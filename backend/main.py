"""
This is the main application file that starts the FastAPI server.

It defines the API endpoints (/upload and /query) and integrates the logic
from the rag_service module. It also handles application startup/shutdown events
and CORS middleware for frontend communication.
"""


from fastapi import Body, FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import time
import database
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from tenacity import RetryError
from sqlalchemy.orm import Session
import uuid
from typing import Generator
import logging
from rich.logging import RichHandler

# Import schemas and service functions
from schemas import Message, QueryRequest, QueryResponse, UploadAcceptResponse, MessageRole, StatusResponse, UserResponse
import rag_service


# Load environment variables at the start.
# This makes .env variables available to os.getenv() for the rag_service.
load_dotenv()

# Configure logging to capture debug information and errors.
logging.basicConfig(
    level="DEBUG",  # Set to DEBUG for more verbose output
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Get a logger for this module.
logger = logging.getLogger(__name__)


# --- Lifespan Events ---
# Use a lifespan context manager to initialize clients on startup and shut down gracefully.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # This block runs on application startup
    logger.info("--- Application Startup ---")

    # Initialize the database on startup
    database.init_db()
    asyncio.create_task(periodic_cleanup_task())  # Start the periodic cleanup task for old sessions

    # Initialize the RAG service clients
    rag_service.initialize_clients()

    yield

    # This block runs on application shutdown
    logger.info("--- Application Shutdown ---")
    await rag_service.close_clients()


# Initialize the FastAPI application with the lifespan manager
app = FastAPI(lifespan=lifespan, title="RAG Application API")


# --- CORS Middleware ---
# Configure CORS (Cross-Origin Resource Sharing) to allow the frontend
# (running on a different domain/port) to communicate with this backend.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Todo: Allows all origins. For production, restrict this to your frontend's domain.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def get_db() -> Generator[Session, None, None]:
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- API Endpoints ---
@app.post("/users/", response_model=UserResponse)
async def create_user(db: Session = Depends(get_db)) -> UserResponse:
    """Create a new anonymous user and return the ID."""
    user_id = str(uuid.uuid4())
    new_user = database.User(id=user_id)
    db.add(new_user)
    db.commit()
    return UserResponse(user_id=user_id)


@app.post("/upload/", status_code=202, response_model=UploadAcceptResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    user_id: str = Body(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> UploadAcceptResponse:
    """
    Endpoint to upload a PDF file for processing and indexing.
    Accepts a file, and starts a background task to process it.
    """
    if file.content_type != "application/pdf":
        logger.error("Invalid file type: '%s'. Expected PDF.", file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    try:
        file_content = await file.read()
        new_document_id = str(uuid.uuid4())
        
        # Find the previous document uploaded by this user, if any.
        previous_doc = (db.query(database.Document)
            .filter(database.Document.user_id == user_id)
            .order_by(database.Document.creation_time.desc())
            .first())

        # Create a new document record in the database
        new_doc = database.Document(id=new_document_id, filename=file.filename, user_id=user_id)
        db.add(new_doc)
        db.commit()

        # Add the processing and cleanup tasks to the background
        background_tasks.add_task(process_document_in_background, new_document_id, file_content)
        if previous_doc is not None:
            background_tasks.add_task(rag_service.delete_namespace, previous_doc.id)

        # Immediately return the ID to prevent blocking the frontend.
        return UploadAcceptResponse(
            message="File upload accepted for processing.",
            document_id=new_document_id
        )
    except Exception as e:
        logger.error("Error during file upload acceptance: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not accept file for upload.")


@app.post("/query/", response_model=QueryResponse)
async def answer_query(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """
    Endpoint to receive a user query and return a RAG-generated answer.
    Manages chat history via a session ID.
    """
    if not request.query:
        logger.error("Received an empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    chat_session_id = request.chat_session_id
    current_history = []

    try:
        # 1. Manage State: Look up the chat session or create a new one.
        if chat_session_id:
            chat_session = db.query(database.ChatSession).filter(
                database.ChatSession.id == chat_session_id,
                database.ChatSession.user_id == request.user_id
            ).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Chat session not found.")
            # Load history from the database record
            current_history = [Message(**msg) for msg in chat_session.history]
        else:
            chat_session_id = str(uuid.uuid4())
            chat_session = database.ChatSession(
                id=chat_session_id,
                document_id=request.document_id,
                user_id=request.user_id
            )
            db.add(chat_session)
            logger.info("Created new chat session with ID: %s", chat_session_id)

        # 2. Call Service: Pass the query and history to the stateless service function.
        logger.info("Passing query to RAG service for session: %s", chat_session_id)
        answer = await rag_service.get_rag_answer(request.query, current_history, request.document_id)

        # 3. Update State: Save the new user message and model answer to the session.
        current_history.append(Message(role=MessageRole.USER, text=request.query))
        current_history.append(Message(role=MessageRole.MODEL, text=answer))
        chat_session.history = current_history
        db.commit()
        
        return QueryResponse(answer=answer, chat_session_id=chat_session_id)
    except RetryError:
        logger.error("AI model is overloaded. Retry limit exceeded.")
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="The AI model is currently overloaded. Please try again in a moment."
        )
    except Exception as e:
        # Handle unexpected errors during the RAG process
        logger.error("An unexpected error occurred during query processing: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")



@app.get("/upload/status/{document_id}", response_model=StatusResponse)
async def get_upload_status(document_id: str, db: Session = Depends(get_db)) -> StatusResponse:
    try:
        # Fetch the task status from the database
        doc = db.query(database.Document).filter(database.Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        return StatusResponse(status=doc.status, message=doc.message, filename=doc.filename)
    except Exception as e:
        logger.error("Error fetching status for doc %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail="Could not fetch upload status.")
    

# --- Background Processing ---
async def process_document_in_background(document_id: str, file_content: bytes) -> None:
    """Helper to run the heavy processing of document indexing outside of the main request."""
    # Get a new database session for this background task
    db = database.SessionLocal()

    try:
        doc = db.query(database.Document).filter(database.Document.id == document_id).first()
        if not doc:
            logger.error("Background task started for non-existent document ID: %s", document_id)
            return
        
        await rag_service.process_and_index_document(file_content, document_id)
        doc.status = 'success'
        doc.message = f'"{doc.filename}" is ready. You can now ask questions.'
        db.commit()
    except Exception as e:
        logger.error("Background processing failed for doc %s: %s", document_id, e)
        db.rollback()
        if doc:
            doc.status = 'failed'
            doc.message = "An error occurred during processing."
            db.commit()
    finally:
        db.close()


async def periodic_cleanup_task(ttl_hours: int = 24) -> None:
    """Periodically cleans up old documents and their namespaces."""
    while True:
        await asyncio.sleep(3600)  # Run once per hour
        db = database.SessionLocal()
        try:
            logger.info("Cleanup Task: Checking for expired documents...")
            TTL_SECONDS = ttl_hours * 3600  # Convert hours to seconds
            cutoff_time = time.time() - TTL_SECONDS
            
            expired_docs = db.query(database.Document).filter(database.Document.creation_time < cutoff_time).all()
            if not expired_docs:
                logger.info("Cleanup Task: No expired documents found.")
                continue

            for doc in expired_docs:
                logger.info(f"Cleanup Task: Deleting expired document and namespace '{doc.id}'...")
                await rag_service.delete_namespace(doc.id)
                db.delete(doc)
            db.commit()
            logger.info("Cleanup Task: Successfully deleted %d expired documents.", len(expired_docs))

        except Exception as e:
            logger.error("Cleanup Task Error: %s", e)
            db.rollback()
        finally:
            db.close()



# --- Uvicorn Server ---
# This block allows running the server directly with `python main.py` for development.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)