"""
This is the main application file that starts the FastAPI server.

It defines the API endpoints (/upload and /query) and integrates the logic
from the rag_service module. It also handles application startup/shutdown events
and CORS middleware for frontend communication.
"""


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from tenacity import RetryError
import uuid
import logging

# Import schemas and service functions
from schemas import QueryRequest, QueryResponse, UploadResponse, ChatSession, MessageRole
import rag_service


# Load environment variables at the start.
# This makes .env variables available to os.getenv() for the rag_service.
load_dotenv()

# Configure logging to capture debug information and errors.
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get a logger for this module.
logger = logging.getLogger(__name__)


# --- STATE INITIALIZATION ---
# --- IMPORTANT: Production Architecture Note ---
# The `chat_sessions` dictionary below stores all conversation data in local memory.
# This makes the application STATEFUL and is not suitable for a production environment.
#
# The Problem with this Stateful Design:
#   1. No Scalability: You cannot run multiple instances of this application behind a
#      load balancer. A user's follow-up requests could be routed to a different
#      server instance that does not have their chat history in its memory.
#   2. No Persistence: If this single server instance restarts or crashes, all active
#      conversation histories are permanently lost.
#
# The Solution for a Stateless, Scalable Backend:
# In a production system, this session state must be externalized to a shared data
# store. Each server instance would then be truly stateless, fetching and writing
# session data from that central store.
# 
# Recommended solutions: Redis (for fast, ephemeral session caching) or a
# database like PostgreSQL (for long-term, persistent storage).
chat_sessions: dict[str, ChatSession] = {}


# --- Lifespan Events ---
# Use a lifespan context manager to initialize clients on startup and shut down gracefully.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # This block runs on application startup
    logger.info("--- Application Startup ---")
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


# --- API Endpoints ---
@app.post("/upload/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Endpoint to upload a PDF file for processing and indexing.
    Accepts a file, extracts text, chunks, embeds, and stores it in Pinecone.
    """
    if file.content_type != "application/pdf":
        logger.error("Invalid file type: '%s'. Expected PDF.", file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    try:
        # Read the file content into memory
        file_content = await file.read()
        
        # Process and index the document using the service function
        result = await rag_service.process_and_index_document(file_content)
        
        return UploadResponse(
            message="File processed and indexed successfully.", 
            filename=file.filename,
            document_id=result["document_id"]
        )
    except ValueError as e:
        # Handle specific value errors from the service (e.g., no text found)
        logger.error("Value error during file upload: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other potential errors during processing
        logger.error("An unexpected error occurred during file upload: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/query/", response_model=QueryResponse)
async def answer_query(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to receive a user query and return a RAG-generated answer.
    Manages chat history via a session ID.
    """
    if not request.query:
        logger.error("Received an empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    chat_session_id = request.chat_session_id

    # 1. Manage State: Look up the session or create a new one.
    if not chat_session_id or chat_session_id not in chat_sessions:
        chat_session_id = str(uuid.uuid4())
        chat_sessions[chat_session_id] = ChatSession()
        logger.info("Created new chat session with ID: %s", chat_session_id)

    chat_session = chat_sessions[chat_session_id]

    try:
        # 2. Call Service: Pass the query and history to the stateless service function.
        logger.info("Passing query to RAG service for session: %s", chat_session_id)
        answer = await rag_service.get_rag_answer(request.query, chat_session.history)
        
        # 3. Update State: Save the new user message and model answer to the session.
        chat_session.add_message(role=MessageRole.USER, text=request.query)
        chat_session.add_message(role=MessageRole.MODEL, text=answer)
        
        return QueryResponse(answer=answer, chat_session_id=chat_session_id)
    except RetryError:
        logger.error("AI model is overloaded. Retry limit exceeded.")
        raise HTTPException(
            status_code=503,
            detail="The AI model is currently overloaded. Please try again in a moment."
        )
    except Exception as e:
        # Handle unexpected errors during the RAG process
        logger.error("An unexpected error occurred during query processing: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Uvicorn Server ---
# This block allows running the server directly with `python main.py` for development.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)