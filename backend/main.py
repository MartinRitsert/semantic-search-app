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

# Import schemas and service functions
from schemas import QueryRequest, QueryResponse, UploadResponse
import rag_service


# Load environment variables at the start.
# This makes .env variables available to os.getenv() for the rag_service.
load_dotenv()


# --- Lifespan Events ---
# Use a lifespan context manager to initialize clients on startup and shut down gracefully.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # This block runs on application startup
    print("--- Application Startup ---")
    rag_service.initialize_clients()

    yield

    # This block runs on application shutdown
    print("--- Application Shutdown ---")
    # Todo: I could add cleanup logic here if needed (e.g., closing client connections)


# Initialize the FastAPI application with the lifespan manager
app = FastAPI(lifespan=lifespan, title="RAG Application API")


# --- CORS Middleware ---
# Configure CORS (Cross-Origin Resource Sharing) to allow the frontend
# (running on a different domain/port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Todo: Allows all origins. For production, restrict this to your frontend's domain.
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
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    try:
        # Read the file content into memory
        file_content = await file.read()
        
        # Process and index the document using the service function
        await rag_service.process_and_index_document(file_content)
        
        return UploadResponse(
            message="File processed and indexed successfully.", 
            filename=file.filename,
            pinecone_index_name='semantic-search-app-index'  # Returning the index name for confirmation
        )
    except ValueError as e:
        # Handle specific value errors from the service (e.g., no text found)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other potential errors during processing
        print(f"An unexpected error occurred during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/query/", response_model=QueryResponse)
async def answer_query(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to receive a user query and return a RAG-generated answer.
    Manages chat history via a session ID.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # Get the answer from the RAG service, passing the query and session ID
        result = await rag_service.get_rag_answer(request.query, request.chat_session_id)
        return QueryResponse(answer=result["answer"], chat_session_id=result["chat_session_id"])
    except Exception as e:
        # Handle unexpected errors during the RAG process
        print(f"An unexpected error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Uvicorn Server ---
# This block allows running the server directly with `python main.py` for development.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)