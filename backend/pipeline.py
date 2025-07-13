"""
This module contains the core business logic for the RAG application.

It handles document processing, embedding, storage, and the RAG chain
for answering queries. Refactoring this logic into a separate service makes
the API endpoints in main.py clean and focused on request/response handling.
"""


import io
import asyncio
import httpx
import logging
from typing import Any
from fastapi import HTTPException
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.genai import types as genai_types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# Import clients
import clients
import pinecone_utils

# Import schemas
from schemas import Message, format_history_for_prompt


# Get a logger for this module.
logger = logging.getLogger(__name__)


# --- CONSTANTS ---
LLM_MODEL_IDENTIFIER = "gemini-2.0-flash"
EMBEDDING_MODEL_IDENTIFIER = "models/text-embedding-004"
CONSISTENCY_CHECK_RETRIES = 45  # Number of retries to check index consistency (1 second per retry)



# --- Private Helper Functions ---
def _extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extracts text from a PDF file content in bytes.
    
    NOTE: pypdf's extract_text() is used here for simplicity. For a production system handling
    diverse and complex PDFs, a more robust library like PyMuPDF or pdfplumber would be a 
    better choice to handle varied layouts and ensure consistent text extraction order.
    """
    logger.info("Extracting text from PDF...")
    try:
        reader = PdfReader(io.BytesIO(file_content))  # Read from in-memory bytes
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text:
            logger.error("Could not extract any text from the provided PDF.")
            raise ValueError("Could not extract any text from the provided PDF.")
        return text
    except Exception as e:
        logger.error("Failed to parse PDF: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Failed to process PDF. Please ensure it is a valid file."
        )

def _chunk_text(text: str, document_id: str) -> list[dict[str, Any]]:
    """Splits a long text into smaller chunks and prepares them for embedding."""
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(text)
    logger.info("Document split into %d chunks.", len(text_chunks))
    return [
        {"id": f"{document_id}_chunk_{i}", "metadata": {"text": chunk}, "values": []} 
        for i, chunk in enumerate(text_chunks)]

async def _embed_chunks(vectors: list[dict[str, Any]]) -> None:
    """Generates embeddings for text chunks in batches using the Google AI API."""
    logger.info("Generating embeddings for text chunks...")
    google_batch_size = 100  # Google's text embedding API limit for texts in a batch
    try:
        for i in range(0, len(vectors), google_batch_size):
            batch_items = vectors[i : i + google_batch_size]
            texts_to_embed = [item['metadata']['text'] for item in batch_items]
            
            response = await clients.google_ai_client.aio.models.embed_content(
                model=EMBEDDING_MODEL_IDENTIFIER,
                contents=texts_to_embed,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            for j, embedding in enumerate(response.embeddings):
                vectors[i + j]['values'] = embedding.values
    except Exception as e:
        logger.error("Google AI embedding call failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not generate document embeddings. The embedding service may be down."
        )

async def _upsert_vectors(vectors: list[dict[str, Any]], namespace: str) -> int | None:
    """Upserts vectors to Pinecone in batches and returns the final LSN."""
    logger.info("Upserting %d vectors to Pinecone namespace '%s'...", len(vectors), namespace)
    pinecone_batch_size = 100  # Pinecone's recommended upsert batch size
    target_lsn = None
    try:
        for i in range(0, len(vectors), pinecone_batch_size):
            batch = vectors[i : i + pinecone_batch_size]
            # Use httpx client to get response headers for LSN
            response = await clients.pinecone_http_client.post(
                url="/vectors/upsert",
                json={"vectors": batch, "namespace": namespace},
            )
            response.raise_for_status()
            lsn_str = response.headers.get('x-pinecone-request-lsn')
            if lsn_str:
                target_lsn = int(lsn_str)
        return target_lsn
    except httpx.HTTPStatusError as e:
        logger.error("Pinecone upsert failed with status %d: %s", e.response.status_code, e.response.text)
        raise HTTPException(
            status_code=503,
            detail="Could not save document to the vector database. The database may be down."
        )
    except Exception as e:
        logger.error("Pinecone upsert failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not save document to the vector database. The database may be down."
        )

async def _verify_index_consistency(target_lsn: int, namespace: str) -> None:
    """Waits for the Pinecone index to achieve consistency for a given LSN.
    
    NOTE: To handle Pinecone's eventual consistency, this code block performs a direct
    queryability test using the Log Sequence Number (LSN). If the check fails after the
    timeout, a warning is printed before continuing, prioritizing a smooth demo experience.
    """
    logger.info("Verifying index consistency...")

    if not target_lsn:
        logger.error("Could not retrieve LSN from upsert response.")
        raise RuntimeError("Failed to retrieve LSN after upserting document.")

    logger.info("Last upsert returned LSN: %d. Waiting for index to catch up...", target_lsn)
    is_ready = False

    for i in range(CONSISTENCY_CHECK_RETRIES):
        is_ready = await pinecone_utils.check_index_consistency(target_lsn, namespace)
        if is_ready:
            logger.info("Consistency met. Index is ready for queries.")
            return
        logger.debug("Attempt %d/%d: Index not yet consistent. Waiting for index to catch up...", i + 1, CONSISTENCY_CHECK_RETRIES)
        await asyncio.sleep(1)

    logger.warning("Index consistency could not be confirmed after %d seconds.", CONSISTENCY_CHECK_RETRIES)
    raise TimeoutError("Failed to prepare document for queries in time. Please upload document again. If this issue persists, try again later.")


async def _rewrite_query_with_history(history: list[Message], query: str) -> str:
    """
    Rewrites the user's query using chat history to create a standalone question.
    """
    # If no history exists, return the original query.
    if not history:
        return query

    logger.info("Rewriting query with chat history...")
    
    # Retrieve the last 10 messages (e.g. 5 user + 5 model) from the chat history.
    formatted_history = format_history_for_prompt(history, limit=10)
    logging.debug("Formatted chat history for rewriting: %s", formatted_history)

    prompt = f"""
    Your sole task is to rephrase a follow-up question into a standalone question based on a chat history.
    If the question is already standalone, return it as is.
    DO NOT add any conversational text, preamble, or explanation. Your output must ONLY be the rewritten query.

    ---
    EXAMPLE:
    Chat History:
    user:
    Tell me about your RAG application.

    model:
    It's a full-stack application using FastAPI and Next.js.

    New Question:
    and the database?

    Standalone Question for Vector Search:
    What database does the RAG application use?
    ---

    YOUR TASK:
    Chat History:
    {formatted_history}

    New Question:
    {query}

    Standalone Question for Vector Search:
    """
    try:
        response = await clients.google_ai_client.aio.models.generate_content(
            model=LLM_MODEL_IDENTIFIER,
            contents=prompt
        )
        rewritten_query = response.text
        logger.info("Original query: '%s' | Rewritten query: '%s'", query, rewritten_query)
        return rewritten_query
    except Exception as e:
        logger.warning("Error during query rewriting: '%s'. Falling back to original query.", e)
        return query
    

async def _retrieve_context(query: str, namespace: str) -> str:
        """Embeds a query and retrieves relevant context chunks from Pinecone."""
        logger.info("Embedding query: '%s'...", query)
        embedding_response = await clients.google_ai_client.aio.models.embed_content(
            model=EMBEDDING_MODEL_IDENTIFIER,
            contents=query,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = embedding_response.embeddings[0].values

        logger.info("Querying Pinecone...")
        query_results = await clients.pinecone_index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=5,  # Using 5 for more consistent results based on earlier experiments
            include_metadata=True
        )

        context_chunks = []
        if query_results.matches:
            for match in query_results.matches:
                if match.metadata and 'text' in match.metadata:
                    context_chunks.append(match.metadata['text'])
            logger.debug("Retrieved %d chunks.", len(context_chunks))
        else:
            logger.warning("No relevant chunks found in Pinecone.")

        return "\n\n---\n\n".join(context_chunks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _generate_final_answer(query: str, history: list[Message], context_string: str) -> str:
    """Generates the final answer using the original query, history, and retrieved context."""
    logger.info("Generating final answer with Gemini...")

    formatted_history = format_history_for_prompt(history, limit=10)

    prompt = f"""
    You are a helpful and conversational assistant.
    Answer the user's question based on the conversation history and the provided context.
    The context below contains relevant information from a document.
    Your answer should be based on the information in the context. If the context does not contain the information to answer the question, state that you cannot answer based on the provided document.

    Conversation History:
    {formatted_history}

    Context from the document:
    {context_string}

    User's Current Question: {query}

    Helpful Answer:
    """

    try:
        logger.info("Generating answer with Gemini...")
        response = await clients.google_ai_client.aio.models.generate_content(
            model=LLM_MODEL_IDENTIFIER,
            contents=prompt
        )
        logger.info("Generated answer: '%s'", response.text)
        return response.text
    except APIError as e:
        if e.code == 429:
            logger.error("Gemini API error - Rate limit exceeded: %s", e)
            raise HTTPException(
                status_code=429,
                detail="API rate limit exceeded. This is likely due to the free tier limit. Please try again later."
            )
    except Exception as e:
        logger.error("Error generating answer with Gemini: %s", e)
        # Re-raise other exceptions to allow the @retry decorator to work
        raise


# --- Public Pipeline Functions ---
async def process_and_index_document(file_content: bytes, document_id: str) -> None:
    """
    Orchestrates the document ingestion pipeline:
    Uploading PDF, chunking, embedding, upserting to Pinecone, consistency check.

    The method completes successfully by returning None if the document is
    indexed and passes the consistency check.
    """
    logger.info("BG Task: Starting ingestion pipeline for document ID: %s...", document_id)

    # Load and Extract Text from PDF
    full_document_text = _extract_text_from_pdf(file_content)

    # Chunk Text
    vectors_to_upsert = _chunk_text(full_document_text, document_id)

    # Embed Text Chunks
    await _embed_chunks(vectors_to_upsert)

    # Upsert Vectors to Pinecone
    target_lsn = await _upsert_vectors(vectors_to_upsert, document_id)

    # Verify Index Consistency
    await _verify_index_consistency(target_lsn, document_id)

    logger.info("Document with ID %s processed and indexed successfully.", document_id)


async def get_rag_answer(query: str, history: list[Message], document_id: str) -> str:
    """
    Orchestrates the RAG retrieval and generation pipeline:
    Gets an answer from the RAG pipeline for a given query and chat session.
    """
    logger.info("Starting RAG pipeline for query in namespace: %s...", document_id)

    try:
        # If history exists, rewrite the query to include relevant context from the history.
        # This is important for follow-up questions that may depend on previous interactions.
        rewritten_query = await _rewrite_query_with_history(history, query)

        # Retrieve
        context_string = await _retrieve_context(rewritten_query, document_id)
        
        # Augment and Generate
        answer = await _generate_final_answer(query, history, context_string)

        logger.info("Answer generated successfully.")
        return answer

    except RetryError as e:
        # Tenacity wraps the last exception. We need to unwrap it.
        last_exception = e.last_attempt.result()
        if isinstance(last_exception, HTTPException):
            # If an HTTPException was already raised (e.g., 429 due to API rate limit),
            # it is re-raised to let FastAPI handle it and send it to the frontend.
            raise last_exception
        else:
            # The retries failed on an unexpected error.
            logger.error("Error during RAG answer generation: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Failed to get an answer due to an external service error. Please try again later."
            )
    except Exception as e:
        # For any other unexpected error that didn't involve retries.
        logger.error("Error during RAG answer generation: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        )


async def delete_previous_document_namespace(document_id: str) -> None:
    """A high-level function to trigger the deletion of an old namespace."""
    await pinecone_utils.delete_namespace(document_id)