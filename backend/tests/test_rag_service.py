"""
This module contains the unit tests for the rag_service.py module.

Mocking is used extensively to replace external dependencies (like Google AI and
Pinecone clients) with controlled, fake objects. This allows a function's
internal logic to be tested without making actual network calls.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import os
import sys

# Add the parent directory to the Python path to allow importing from rag_service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rag_service
from rag_service import genai_types


# --- Unit Tests for process_and_index_document ---
@pytest.mark.asyncio
@patch('rag_service.RecursiveCharacterTextSplitter')
@patch('rag_service.PdfReader')
@patch('rag_service.pinecone_index', new_callable=AsyncMock)
@patch('rag_service.google_ai_client', new_callable=AsyncMock)
async def test_process_and_index_document_success_two_pages(
    mock_google_client: AsyncMock,
    mock_pinecone_index: AsyncMock,
    mock_pdf_reader: MagicMock,
    mock_text_splitter: MagicMock
) -> None:
    """
    Tests the successful execution with a 2-page document producing 2 chunks.
    """
    # --- Arrange ---
    # 1. Mock the PDF Reader to return two pages.
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is the text from a page."
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page, mock_page]
    mock_pdf_reader.return_value = mock_reader_instance

    # 2. Mock the Text Splitter to explicitly return two chunks.
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_text.return_value = ["This is the first chunk.", "This is the second chunk."]
    mock_text_splitter.return_value = mock_splitter_instance

    # 3. Mock the Google AI Client to return two embeddings.
    mock_embedding_1 = MagicMock(values=[0.1] * 768)
    mock_embedding_2 = MagicMock(values=[0.2] * 768)
    mock_embed_response = MagicMock(embeddings=[mock_embedding_1, mock_embedding_2])
    mock_google_client.aio.models.embed_content.return_value = mock_embed_response

    # 4. Mock the Pinecone Index.
    mock_pinecone_index.delete.return_value = None
    mock_pinecone_index.upsert.return_value = None

    # --- Act ---
    await rag_service.process_and_index_document(b"dummy_pdf_content")

    # --- Assert ---
    # Assert that the reader was used and text was extracted for both pages.
    assert mock_page.extract_text.call_count == 4

    # Assert that the splitter was used on the combined text.
    mock_splitter_instance.split_text.assert_called_once_with("This is the text from a page.This is the text from a page.")

    # Assert that the embeddings and upsert reflect the two chunks.
    mock_google_client.aio.models.embed_content.assert_awaited_once()
    mock_pinecone_index.delete.assert_awaited_once_with(delete_all=True)
    mock_pinecone_index.upsert.assert_awaited_once()

    upsert_call_args = mock_pinecone_index.upsert.call_args
    vectors = upsert_call_args.kwargs['vectors']
    assert len(vectors) == 2
    assert vectors[0]['id'] == 'chunk_0'
    assert vectors[1]['id'] == 'chunk_1'


@pytest.mark.asyncio
@patch('rag_service.PdfReader')
@patch('rag_service.pinecone_index', new_callable=AsyncMock)
@patch('rag_service.google_ai_client', new_callable=AsyncMock)
async def test_process_and_index_document_no_text_extracted(
    mock_google_client: AsyncMock,
    mock_pinecone_index: AsyncMock,
    mock_pdf_reader: MagicMock
) -> None:
    """
    Tests that a ValueError is raised if the PDF contains no extractable text.
    """
    # --- Arrange ---
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader_instance

    # --- Act & Assert ---
    with pytest.raises(ValueError, match="Could not extract any text from the provided PDF."):
        await rag_service.process_and_index_document(b"dummy_pdf_content")


@pytest.mark.asyncio
@patch('rag_service.pinecone_index', new_callable=AsyncMock)
@patch('rag_service.google_ai_client', new_callable=AsyncMock)
async def test_get_rag_answer_success(
    mock_google_client: AsyncMock,
    mock_pinecone_index: AsyncMock
) -> None:
    """
    Tests the successful execution of the get_rag_answer function.
    """
    # --- Arrange ---
    mock_embedding_value = MagicMock()
    mock_embedding_value.values = [0.2] * 768
    mock_embed_response = MagicMock(embeddings=[mock_embedding_value])
    mock_google_client.aio.models.embed_content.return_value = mock_embed_response
    
    mock_chat_session = AsyncMock()
    mock_chat_session.send_message.return_value = MagicMock(text="This is the final answer from the LLM.")
    mock_google_client.aio.chats.create.return_value = mock_chat_session

    mock_match = MagicMock()
    mock_match.metadata = {'text': 'This is the retrieved context from the document.'}
    mock_query_response = MagicMock(matches=[mock_match])
    mock_pinecone_index.query.return_value = mock_query_response

    # --- Act ---
    query = "What is the key information?"
    result = await rag_service.get_rag_answer(query, chat_session_id=None)

    # --- Assert ---
    mock_google_client.aio.models.embed_content.assert_awaited_once_with(
        model="models/text-embedding-004",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )

    mock_pinecone_index.query.assert_awaited_once()
    mock_google_client.aio.chats.create.assert_awaited_once()
    mock_chat_session.send_message.assert_awaited_once()

    assert result["answer"] == "This is the final answer from the LLM."
