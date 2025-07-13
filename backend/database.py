"""
This module defines the database schema and initializes the SQLite database.
It uses SQLAlchemy for ORM and is designed for local development and single-container deployments.
"""


import time
import json
from typing import Generator
from sqlalchemy import create_engine, Column, String, Float, Text, ForeignKey
from sqlalchemy.orm import Session, sessionmaker, declarative_base

# Import schemas
from schemas import Message


# --- Production Architecture Note ---
# This application uses SQLite, a simple file-based database, which is ideal for
# local development, demos, and single-container deployments.
#
# For a production environment that requires horizontal scaling (running multiple
# instances of the application, e.g., in Kubernetes), SQLite is not suitable.
# Each container instance would have its own separate database file, leading to
# data inconsistency.
#
# The production-ready solution is to use a centralized, client-server database
# like PostgreSQL or MySQL, typically managed by a cloud provider (e.g., AWS RDS,
# Google Cloud SQL). All application instances would connect to this single
# database over the network, ensuring a single source of truth.


# Database connection URL
DATABASE_URL = "sqlite:///./rag_app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Model (Table) ---
class User(Base):
    """Represents an anonymous user session."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    
    
class Document(Base):
    """Represents an uploaded document, its namespace, and processing status."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    status = Column(String, default="processing")
    message = Column(String, nullable=True)
    creation_time = Column(Float, default=time.time)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)


class ChatSession(Base):
    """Represents a single conversation related to a document."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Store the chat history as a JSON string for simplicity
    _history = Column("history", Text, default="[]")

    @property
    def history(self) -> list[dict]:
        """Load the JSON string from the DB and parse it into a list of dicts."""
        return json.loads(self._history)

    @history.setter
    def history(self, messages: list[Message]) -> None:
        """Serialize a list of Pydantic Message objects into a JSON string for the DB."""
        self._history = json.dumps([msg.model_dump() for msg in messages])


def init_db():
    """Create the database and table on application startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()