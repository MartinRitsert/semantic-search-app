"""
This module contains the integration tests for the FastAPI application defined in main.py.

It uses pytest for the testing framework and FastAPI's TestClient for making requests
to the application in a controlled environment. External service calls within rag_service
are mocked using pytest's monkeypatch fixture to ensure tests are isolated, fast,
and do not rely on external APIs (like Google or Pinecone) or secrets.
"""


import pytest
from pytest import MonkeyPatch
from fastapi.testclient import TestClient
import os
import sys

# Add the parent directory (backend/) to the Python path.
# This allows the test module to find and import from 'main', 'rag_service', etc.
# which are located in the parent directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
import rag_service


# Create a TestClient instance that will be used to make simulated
# HTTP requests to the FastAPI application without running a live server.
client = TestClient(app)


# --- Test Fixtures ---
# Use a fixture with monkeypatch to mock our external service calls before any
# tests run, ensuring isolation.
@pytest.fixture(autouse=True)
def mock_rag_service(monkeypatch: MonkeyPatch) -> None:
    """
    This fixture automatically runs for every test. It uses monkeypatch
    to replace the functions in the `rag_service` module with mock objects.
    This prevents actual calls to external APIs (Google, Pinecone) during tests.
    """
    # Create mock async functions to replace the real ones in rag_service.
    async def mock_process_and_index_document(file_content: bytes) -> None:
        # This mock simply confirms it was called, without doing any real work.
        return

    async def mock_get_rag_answer(query: str, chat_session_id: str | None) -> dict[str, str]:
        # This mock returns a predictable dictionary, simulating a successful RAG response.
        return {"answer": f"This is a mocked answer to '{query}'.", "chat_session_id": "mock_session_123"}

    # Use monkeypatch to replace the real functions with our mocks for the duration of a test.
    monkeypatch.setattr(rag_service, "initialize_clients", lambda: None) # Mock initialization to do nothing
    monkeypatch.setattr(rag_service, "process_and_index_document", mock_process_and_index_document)
    monkeypatch.setattr(rag_service, "get_rag_answer", mock_get_rag_answer)


# --- Test Cases for the /upload/ Endpoint ---
def test_upload_document_success() -> None:
    """
    Tests the successful upload of a valid PDF file.
    Verifies that the API returns a 200 OK status and the expected JSON response.
    """
    # Simulate a dummy PDF file using an in-memory byte stream.
    # The content doesn't need to be a real PDF, just bytes.
    dummy_pdf_content = b"%PDF-1.5 test file"
    files = {"file": ("test.pdf", dummy_pdf_content, "application/pdf")}
    
    # Make a POST request to the /upload/ endpoint.
    response = client.post("/upload/", files=files)
    
    # Assertions: Check if the API behaved as expected.
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "File processed and indexed successfully."
    assert response_data["filename"] == "test.pdf"
    assert "pinecone_index_name" in response_data


def test_upload_document_invalid_file_type() -> None:
    """
    Tests the behavior of the /upload/ endpoint when a non-PDF file is provided.
    Verifies that the API correctly returns a 400 Bad Request error.
    """
    # Simulate uploading a text file instead of a PDF.
    dummy_txt_content = b"this is not a pdf"
    files = {"file": ("test.txt", dummy_txt_content, "text/plain")}
    
    response = client.post("/upload/", files=files)
    
    # Assert that the status code is 400 and the detail message is correct.
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid file type. Please upload a PDF."}


# --- Test Cases for the /query/ Endpoint ---
def test_answer_query_success() -> None:
    """
    Tests a successful query to the /query/ endpoint.
    Verifies a 200 OK status and that the response matches the expected structure,
    using the data returned from our mocked get_rag_answer function.
    """
    # Define the request payload.
    request_data = {"query": "What is the capital of France?", "chat_session_id": "test_session_1"}
    
    response = client.post("/query/", json=request_data)
    
    # Assertions: Check for success status and correct response content.
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["answer"] == "This is a mocked answer to 'What is the capital of France?'."
    assert response_data["chat_session_id"] == "mock_session_123"


def test_answer_query_empty_query() -> None:
    """
    Tests the API's validation for an empty query string.
    Verifies that it correctly returns a 400 Bad Request error.
    """
    request_data = {"query": "", "chat_session_id": "test_session_2"}
    
    response = client.post("/query/", json=request_data)
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Query cannot be empty."}


def test_answer_query_processing_error(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the API's error handling when the underlying service raises an exception.
    Verifies that a generic 500 Internal Server Error is returned to the client.
    """
    # Re-mock the get_rag_answer function for this specific test to raise an error.
    async def mock_raise_exception(*args, **kwargs):
        raise Exception("A simulated service error occurred.")

    monkeypatch.setattr(rag_service, "get_rag_answer", mock_raise_exception)
    
    request_data = {"query": "This query will cause an error."}
    response = client.post("/query/", json=request_data)
    
    assert response.status_code == 500
    assert "An unexpected error occurred" in response.json()["detail"]
