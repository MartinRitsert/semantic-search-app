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
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any
import logging

# Import schemas
from schemas import Message, format_history_for_prompt


# Get a logger for this module.
logger = logging.getLogger(__name__)


# --- CONSTANTS ---
LLM_MODEL_IDENTIFIER = "gemini-2.0-flash"
EMBEDDING_MODEL_IDENTIFIER = "models/text-embedding-004"


# --- CLIENT AND STATE INITIALIZATION ---
# These global variables will be initialized on application startup via the lifespan manager.
# This avoids re-initializing clients on every request.
google_ai_client: genai.Client | None = None
pinecone_client: Pinecone | None = None
pinecone_index = None


def initialize_clients() -> None:
    """Initializes the Pinecone and Google AI clients using environment variables."""
    global pinecone_client, google_ai_client, pinecone_index
    
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

    # Initialize an async-capable index object.
    try:
        index_host = pinecone_client.describe_index(index_name).host  # Get host URL
        pinecone_index = pinecone_client.IndexAsyncio(host=index_host)
        logger.info("Connected to Pinecone index '%s' at %s.", index_name, index_host)
    except Exception as e:
        logger.critical("Failed to get host and connect to Pinecone index '%s': %s", index_name, e)
        raise RuntimeError(f"Failed to get host and connect to Pinecone index '{index_name}': {e}")


async def close_clients() -> None:
    """
    Closes the Pinecone client. 
    The Google AI client is closed automatically by the library.
    """
    global pinecone_index
    if pinecone_index:
        logger.info("Closing Pinecone index connection...")
        try:
            await pinecone_index.close()
        except Exception as e:
            logger.error("Error while closing Pinecone index connection: %s", e)
        finally:
            pinecone_index = None


async def process_and_index_document(file_content: bytes) -> dict[str, Any]:
    """
    Processes an uploaded PDF file, chunks it, generates embeddings, and upserts to Pinecone.
    """
    if not pinecone_index or not google_ai_client:
        logger.error("Clients are not initialized. The application might not have started correctly.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: clients are not available."
        )

    document_id = str(uuid.uuid4())
    logger.info("Processing document with unique ID: %s...", document_id)

    # 1. Load and Extract Text from in-memory bytes
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

    # 2. Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(full_document_text)
    logger.info("Document split into %d chunks.", len(text_chunks))

    # 3. Embed and Upsert to Pinecone
    model_name = EMBEDDING_MODEL_IDENTIFIER
    google_batch_size = 100  # Google's text embedding API limit for texts in a batch
    pinecone_upsert_batch_size = 100  # Pinecone's recommended upsert batch size

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

    # Before upserting a new document, clear all old data from the index.
    # This keeps the demo simple, focusing on one document at a time.
    # NOTE: In a multi-user app, one would use namespaces to isolate data.
    try:
        index_stats = await pinecone_index.describe_index_stats()
        if index_stats.total_vector_count > 0:
            logger.info("Clearing previous data from index...")
            await pinecone_index.delete(delete_all=True)
    except Exception as e:
        logger.error("Failed to clear Pinecone index: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not prepare the vector database for new document. Please try again later."
        )

    logger.info("Upserting %d vectors to Pinecone...", len(all_vectors_to_upsert))
    try:
        for i in range(0, len(all_vectors_to_upsert), pinecone_upsert_batch_size):
            batch_to_upsert = all_vectors_to_upsert[i : i + pinecone_upsert_batch_size]
            await pinecone_index.upsert(vectors=batch_to_upsert)
    except Exception as e:
        logger.error("Pinecone upsert failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not save document to the vector database. The database may be down."
        )

    # Verify that the upserted vectors are queryable.
    # NOTE: To handle Pinecone's eventual consistency, this readiness check performs a direct
    # queryability test. It repeatedly tries to `fetch()` the first vector that was just upserted,
    # as a successful fetch confirms the index is ready. If the check fails after the timeout,
    # a warning is printed before continuing, prioritizing a smooth demo experience.
    logger.info("Verifying index readiness...")
    test_vector_id = all_vectors_to_upsert[0]['id']
    max_retries = 20
    retry_delay = 1  # seconds

    for i in range(max_retries):
        try:
            fetch_response = await pinecone_index.fetch(ids=[test_vector_id])
            if fetch_response.vectors.get(test_vector_id):
                logger.info("Index is ready. Found test vector '%s'.", test_vector_id)
                break  # Exit the loop on success
        except Exception as e:
            logger.warning("Attempt %d/%d: Error during fetch: %s", i + 1, max_retries, e)

        logger.debug("Attempt %d/%d: Waiting for index update... Test vector not found yet.", i + 1, max_retries)
        await asyncio.sleep(retry_delay)
    else:
        # This 'else' block runs only if the loop finishes without a 'break'
        logger.warning("Index readiness could not be confirmed after %d seconds.", max_retries * retry_delay)

    logger.info("Document processing and indexing complete.")
    return {"document_id": document_id}


async def _rewrite_query_with_history(history: list[Message], query: str) -> str:
    """
    Rewrites the user's query using chat history to create a standalone question.
    """
    logger.info("Rewriting query with chat history...")
    
    # Retrieve the last 10 messages (e.g. 5 user + 5 model) from the chat history.
    formatted_history = format_history_for_prompt(history, limit=10)
    logging.debug("Formatted chat history for rewriting: %s", formatted_history)

    prompt = f"""
    You are an expert at rephrasing user questions to be optimized for a vector database search.
    Your task is to take the chat history and the user's new, potentially vague question, and rewrite it into a clear, standalone question.
    This new question should be rich in keywords and context, making it ideal for finding relevant text chunks through semantic search.

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
    logger.info("Generating answer with Gemini...")
    response = await google_ai_client.aio.models.generate_content(
        model=LLM_MODEL_IDENTIFIER,
        contents=prompt
    )
    return response.text


async def get_rag_answer(query: str, history: list[Message]) -> str:
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

    except Exception as e:
        logger.error("Error during RAG answer generation: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Failed to get an answer due to an external service error. Please try again later."
        )
