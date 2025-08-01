"""
This file defines the Pydantic models for API data validation and serialization,
as well as internal data structures for managing application state.

These models ensure that data sent to and from the API has the correct
structure and data types, providing robustness and clear contracts for the API.
"""


from pydantic import BaseModel, Field
from enum import Enum


# --- Internal Schemas ---

# Pydantic model for the role of a message in a chat.
class MessageRole(str, Enum):
    """Enumeration for the possible roles in a chat message."""
    USER = "user"
    MODEL = "model"

# Pydantic model for a message, which includes the role and text.
class Message(BaseModel):
    """Represents a single message in the chat history."""
    role: MessageRole
    text: str

# Reusable utility function to format message history for LLM prompts.
def format_history_for_prompt(history: list[Message], limit: int | None = None) -> str:
    """Creates a formatted string of the message history for an LLM prompt."""
    messages = history
    if limit is not None and len(history) > limit:
        messages = history[-limit:]
        
    return "\n\n".join([f"{message.role.value}:\n{message.text}" for message in messages])


#--- API Request / Response Schemas ---

class QueryRequest(BaseModel):
    """
    Pydantic model for the incoming query request from the frontend.
    """
    query: str
    chat_session_id: str | None = None  # Optional session ID for multi-turn conversations
    document_id: str
    user_id: str

class QueryResponse(BaseModel):
    """
    Pydantic model for the response sent back to the frontend after a query.
    """
    answer: str
    chat_session_id: str  # Always return a session ID to maintain conversation state

class UploadAcceptResponse(BaseModel):
    """
    Response model for when a file upload is successfully accepted for processing.
    """
    message: str
    document_id: str

class StatusResponse(BaseModel):
    """
    Response model for the upload status check endpoint.
    """
    status: str
    message: str | None = None
    filename: str | None = None

class UserResponse(BaseModel):
    """
    Response model for when a new user is created.
    """
    user_id: str