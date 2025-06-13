"""
This file defines the Pydantic models for API data validation and serialization.

These models ensure that data sent to and from the API has the correct
structure and data types, providing robustness and clear contracts for the API.
"""


from pydantic import BaseModel


# Pydantic model for the incoming query request from the frontend.
class QueryRequest(BaseModel):
    query: str
    chat_session_id: str | None = None  # Optional session ID for multi-turn conversations

# Pydantic model for the response sent back to the frontend.
class QueryResponse(BaseModel):
    answer: str
    chat_session_id: str  # Always return a session ID to maintain conversation state

# Pydantic model for the response after a successful file upload.
class UploadResponse(BaseModel):
    message: str
    filename: str
    pinecone_index_name: str