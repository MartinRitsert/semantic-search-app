"""
This module contains the core business logic for the RAG application.

It handles document processing, embedding, storage, and the RAG chain
for answering queries. Refactoring this logic into a separate service makes
the API endpoints in main.py clean and focused on request/response handling.
"""


import os
import uuid
import io
import asyncio
from fastapi import HTTPException
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from typing import Any
import logging
from google.genai.errors import APIError
import httpx

# Import schemas
from schemas import Message, format_history_for_prompt


# Get a logger for this module.
logger = logging.getLogger(__name__)


# --- CONSTANTS ---
LLM_MODEL_IDENTIFIER = "gemini-2.0-flash"
EMBEDDING_MODEL_IDENTIFIER = "models/text-embedding-004"
CONSISTENCY_CHECK_RETRIES = 45  # Number of retries to check index consistency (1 second per retry)


# --- CLIENT AND STATE INITIALIZATION ---
# These global variables will be initialized on application startup via the lifespan manager.
# This avoids re-initializing clients on every request.
google_ai_client: genai.Client | None = None  # Google Gen AI Python SDK client
pinecone_client: Pinecone | None = None  # Pinecone Python SDK client 
pinecone_index = None
pinecone_http_client: httpx.AsyncClient | None = None  # Pinecone HTTPX client - used for direct API calls to Pinecone to get headers like LSN (Log Sequence Number) for consistency checks. 


def initialize_clients() -> None:
    """Initializes the Pinecone and Google AI clients using environment variables."""
    global pinecone_client, google_ai_client, pinecone_index, pinecone_http_client
    
    # Load API Keys from environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        logger.critical("GOOGLE_API_KEY not found in environment variables. Application cannot start.")
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    if not pinecone_api_key:
        logger.critical("PINECONE_API_KEY not found in environment variables. Application cannot start.")
        raise ValueError("PINECONE_API_KEY not found in environment variables.")

    # Initialize Google Gen AI Client
    logger.info("Initializing Google Gen AI Client...")
    try:
        google_ai_client = genai.Client(api_key=google_api_key)
        logger.info("Google Gen AI Client initialized.")
    except Exception as e:
        logger.critical("Failed to initialize Google Gen AI Client: %s", e)
        raise RuntimeError(f"Failed to initialize Google Gen AI Client: {e}")

    # Initialize Pinecone Connection
    logger.info("Initializing Pinecone connection...")
    try:
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        logger.info("Pinecone client initialized.")
    except Exception as e:
        logger.critical("Failed to initialize Pinecone client: %s", e)
        raise RuntimeError(f"Failed to initialize Pinecone client: {e}")

    # Connect to the Pinecone Index or create it if it doesn't exist
    index_name = 'semantic-search-app-index'
    embedding_dim = 768  # Dimension for Google's embedding-004 model

    try:
        if index_name not in [index_info["name"] for index_info in pinecone_client.list_indexes()]:
            logger.info("Index '%s' does not exist. Creating...", index_name)
            pinecone_client.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            logger.info("Index '%s' created successfully.", index_name)
        else:
            logger.info("Index '%s' already exists.", index_name)
    except Exception as e:
        logger.critical("Failed to create or connect to Pinecone index '%s': %s", index_name, e)
        raise RuntimeError(f"Failed to create or connect to Pinecone index '{index_name}': {e}")

    # Initialize an async-capable index object and Pinecone HTTPX client.
    try:
        # Initialize an async-capable index object.
        index_host = pinecone_client.describe_index(index_name).host  # Get host URL
        pinecone_index = pinecone_client.IndexAsyncio(host=index_host)
        logger.info("Connected to Pinecone index '%s' at %s.", index_name, index_host)

        # Initialize an httpx client for direct API calls to get headers containing LSN.
        pinecone_http_client = httpx.AsyncClient(
            base_url=f"https://{index_host}",
            headers={"Api-Key": pinecone_api_key}
        )
        logger.info("Initialized httpx client for direct Pinecone API calls.")

    except Exception as e:
        logger.critical("Failed to get host and connect to Pinecone index '%s': %s", index_name, e)
        raise RuntimeError(f"Failed to get host and connect to Pinecone index '{index_name}': {e}")


async def close_clients() -> None:
    """
    Closes the Pinecone client. 
    The Google AI client is closed automatically by the library.
    """
    global pinecone_index, pinecone_http_client
    if pinecone_index:
        logger.info("Closing Pinecone index connection...")
        try:
            await pinecone_index.close()
        except Exception as e:
            logger.error("Error while closing Pinecone index connection: %s", e)
        finally:
            pinecone_index = None
    
    if pinecone_http_client:
        logger.info("Closing Pinecone httpx client...")
        await pinecone_http_client.aclose()
        pinecone_http_client = None


async def process_and_index_document(file_content: bytes, document_id: str) -> None:
    """
    Processes an uploaded PDF file, chunks it, generates embeddings, and upserts to Pinecone.
    
    The method completes successfully by returning None if the document is
    indexed and passes the consistency check.

    Raises:
    TimeoutError: If the index readiness check fails to confirm
                    consistency within the defined retry limit.
    HTTPException: If any other error occurs during the process
                    (e.g., embedding failure, database connection error).
    """
    if not pinecone_index or not google_ai_client or not pinecone_http_client:
        logger.error("Clients are not initialized during background task.")
        # Raise a standard error to be caught by the background task handler
        raise RuntimeError("Clients are not initialized.")

    logger.info("BG Task: Processing document with ID: %s...", document_id)

    # --- 1. Load and Extract Text from PDF ---
    # NOTE: pypdf's extract_text() is used here for simplicity. For a production system handling
    # diverse and complex PDFs, a more robust library like PyMuPDF or pdfplumber would be a 
    # better choice to handle varied layouts and ensure consistent text extraction order.
    try:
        reader = PdfReader(io.BytesIO(file_content))  # Read from in-memory bytes
        full_document_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not full_document_text:
            logger.error("Could not extract any text from the provided PDF.")
            raise ValueError("Could not extract any text from the provided PDF.")
    except Exception as e:
        logger.error("Failed to parse PDF: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Failed to process PDF. Please ensure it is a valid file."
        )

    # --- 2. Chunk Text ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(full_document_text)
    logger.info("Document split into %d chunks.", len(text_chunks))

    # --- 3. Embed Text Chunks ---
    model_name = EMBEDDING_MODEL_IDENTIFIER
    google_batch_size = 100  # Google's text embedding API limit for texts in a batch

    all_vectors_to_upsert = [
        {"id": f"{document_id}_chunk_{i}", "metadata": {"text": chunk}, "values": []}
        for i, chunk in enumerate(text_chunks)
    ]

    logger.info("Generating embeddings for %d chunks...", len(text_chunks))
    try:
        for i in range(0, len(all_vectors_to_upsert), google_batch_size):
            batch_items = all_vectors_to_upsert[i : i + google_batch_size]
            texts_to_embed = [item['metadata']['text'] for item in batch_items]
            
            response = await google_ai_client.aio.models.embed_content(
                model=model_name,
                contents=texts_to_embed,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            for j, embedding in enumerate(response.embeddings):
                all_vectors_to_upsert[i + j]['values'] = embedding.values
    except Exception as e:
        logger.error("Google AI embedding call failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not generate document embeddings. The embedding service may be down."
        )

    # --- 4. Upsert Vectors to Pinecone ---
    pinecone_upsert_batch_size = 100  # Pinecone's recommended upsert batch size

    logger.info("Upserting %d vectors to Pinecone...", len(all_vectors_to_upsert))
    target_lsn = None
    try:
        for i in range(0, len(all_vectors_to_upsert), pinecone_upsert_batch_size):
            batch = all_vectors_to_upsert[i : i + pinecone_upsert_batch_size]
            
            # Use httpx client to get response headers for LSN
            response = await pinecone_http_client.post(
                url="/vectors/upsert",
                json={"vectors": batch, "namespace": document_id},
            )
            response.raise_for_status()

            lsn_str = response.headers.get('x-pinecone-request-lsn')
            if lsn_str:
                target_lsn = int(lsn_str)

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

    # --- 5. Verify Index Consistency ---
    # NOTE: To handle Pinecone's eventual consistency, this code block performs a direct
    # queryability test using the Log Sequence Number (LSN). If the check fails after the
    # timeout, a warning is printed before continuing, prioritizing a smooth demo experience.
    logger.info("Verifying index consistency...")

    if not target_lsn:
        logger.error("Could not retrieve LSN from upsert response.")
        raise RuntimeError("Failed to retrieve LSN after upserting document.")

    logger.info("Last upsert returned LSN: %d. Waiting for index to catch up...", target_lsn)
    is_ready = False

    for i in range(CONSISTENCY_CHECK_RETRIES):
        is_ready = await check_index_consistency(target_lsn, document_id)
        if is_ready:
            logger.info("Consistency met. Index is ready for queries.")
            break

        logger.debug("Attempt %d/%d: Index not yet consistent. Waiting for index to catch up...", i + 1, CONSISTENCY_CHECK_RETRIES)
        await asyncio.sleep(1)

    else:
        logger.warning("Index consistency could not be confirmed after %d seconds.", CONSISTENCY_CHECK_RETRIES)
        raise TimeoutError("Failed to prepare document for queries in time. Please upload document again. If this issue persists, try again later.")

    # --- Document Processing Complete ---
    logger.info("Document with ID %s processed and indexed successfully.", document_id)


async def get_rag_answer(query: str, history: list[Message], document_id: str) -> str:
    """
    Gets an answer from the RAG pipeline for a given query and chat session.
    Manages chat sessions in-memory.
    """
    if not pinecone_index or not google_ai_client:
        logger.error("Clients are not initialized during a request.")
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: clients are not available."
        )

    try:
        # If history exists, rewrite the query to include relevant context from the history.
        # This is important for follow-up questions that may depend on previous interactions.
        query_for_retrieval = query
        if history:
            query_for_retrieval = await _rewrite_query_with_history(history, query)

        # 1. Retrieve
        logger.info("Embedding query: '%s'...", query_for_retrieval)
        embedding_response = await google_ai_client.aio.models.embed_content(
            model=EMBEDDING_MODEL_IDENTIFIER,
            contents=query_for_retrieval,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = embedding_response.embeddings[0].values

        logger.info("Querying Pinecone...")
        query_results = await pinecone_index.query(
            vector=query_embedding,
            namespace=document_id,
            top_k=5,  # Using 5 for more consistent results based on earlier experiments
            include_metadata=True
        )
        
        # 2. Augment
        context_chunks = []
        if query_results.matches:
            for match in query_results.matches:
                if match.metadata and 'text' in match.metadata:
                    context_chunks.append(match.metadata['text'])
            logger.debug("Retrieved %d chunks.", len(context_chunks))
        else:
            logger.warning("No relevant chunks found in Pinecone.")

        context_string = "\n\n---\n\n".join(context_chunks)
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

        # 3. Generate
        answer = await _send_message_with_retry(prompt)

        logger.info("Generated answer: '%s'", answer)
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


async def _rewrite_query_with_history(history: list[Message], query: str) -> str:
    """
    Rewrites the user's query using chat history to create a standalone question.
    """
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
        response = await google_ai_client.aio.models.generate_content(
            model=LLM_MODEL_IDENTIFIER,
            contents=prompt
        )
        rewritten_query = response.text
        logger.info("Original query: '%s' | Rewritten query: '%s'", query, rewritten_query)
        return rewritten_query
    except Exception as e:
        logger.warning("Error during query rewriting: '%s'. Falling back to original query.", e)
        return query
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _send_message_with_retry(prompt: str) -> str:
    """Helper function to send a message to Gemini with retry logic."""
    try:
        logger.info("Generating answer with Gemini...")
        response = await google_ai_client.aio.models.generate_content(
            model=LLM_MODEL_IDENTIFIER,
            contents=prompt
        )
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


async def check_index_consistency(target_lsn: int, namespace: str) -> bool:
    """Checks if the index has reached or passed a target Log Sequence Number (LSN).
    If it has reached or exceeded the target LSN, it is considered consistent and ready for queries."""
    if not pinecone_http_client:
        return False
    
    try:
        # Query with a dummy vector to get the current max indexed LSN from headers.
        dummy_vector = [0.0] * 768 # Dimension for Google's embedding-004 model
        response = await pinecone_http_client.post(
            url="/query",
            json={
                "vector": dummy_vector,
                "topK": 1,
                "namespace": namespace,
            }
        )
        response.raise_for_status()
        
        current_lsn_str = response.headers.get('x-pinecone-max-indexed-lsn')
        
        if current_lsn_str:
            current_lsn = int(current_lsn_str)
            if current_lsn >= target_lsn:
                logger.info("Index is consistent. Current LSN %d >= Target LSN %d.", current_lsn, target_lsn)
                return True
            else:
                logger.debug("Index not yet consistent. Current LSN: %d < Target LSN: %d.", current_lsn, target_lsn)
                return False
        else:
            logger.info("Header 'x-pinecone-max-indexed-lsn' not found. This is expected if the index entries recently deleted.")
            return False
    except httpx.HTTPStatusError as e:
        logger.error("API error during index consistency check: Status %d, Response: %s", e.response.status_code, e.response.text)
        return False
    except Exception as e:
        logger.error("Error during index consistency check: %s", e)
        return False
        

async def delete_namespace(namespace: str):
    """Deletes all vectors from a specific namespace in Pinecone."""
    try:
        logger.info(f"Issuing delete command for namespace: '{namespace}'")
        await pinecone_index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Delete command for namespace '{namespace}' acknowledged.")
    except Exception as e:
        logger.error(f"Failed to issue delete for namespace '{namespace}': {e}")        
