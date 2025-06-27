"""
This module contains the core business logic for the RAG application.

It handles document processing, embedding, storage, and the RAG chain
for answering queries. Refactoring this logic into a separate service makes
the API endpoints in main.py clean and focused on request/response handling.
"""


import os
import uuid
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any


# --- CLIENT AND STATE INITIALIZATION ---
# These global variables will be initialized on application startup via the lifespan manager.
# This avoids re-initializing clients on every request.
google_ai_client: genai.Client | None = None
pinecone_client: Pinecone | None = None
pinecone_index = None

# In-memory storage for chat sessions.
# For a production application, this should be replaced with a more persistent
# and scalable solution like Redis, a database, or another caching mechanism
# to handle multiple server instances and user sessions.
chat_sessions: dict[str, Any] = {}


def initialize_clients():
    """Initializes the Pinecone and Google AI clients using environment variables."""
    global pinecone_client, google_ai_client, pinecone_index
    
    # Load API Keys from environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")

    # Initialize Google Gen AI Client
    print("Initializing Google Gen AI Client...")
    google_ai_client = genai.Client(api_key=google_api_key)
    print("Google Gen AI Client initialized.")

    # Initialize Pinecone Connection
    print("Initializing Pinecone connection...")
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    print("Pinecone client initialized.")

    # Connect to the Pinecone Index or create it if it doesn't exist
    index_name = 'semantic-search-app-index'
    embedding_dim = 768  # Dimension for Google's embedding-004 model

    if index_name not in [index_info["name"] for index_info in pinecone_client.list_indexes()]:
        print(f"Index '{index_name}' does not exist. Creating...")
        pinecone_client.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    # Initialize an async-capable index object.
    index_host = pinecone_client.describe_index(index_name).host  # Get host URL
    pinecone_index = pinecone_client.IndexAsyncio(host=index_host)
    print(f"Connected to Pinecone index '{index_name}' at {index_host}.")


async def close_clients():
    """
    Closes the Pinecone client. 
    The Google AI client is closed automatically by the library.
    """
    global pinecone_index
    if pinecone_index:
        print("Closing Pinecone index connection...")
        await pinecone_index.close()
        pinecone_index = None


async def process_and_index_document(file_content: bytes):
    """
    Processes an uploaded PDF file, chunks it, generates embeddings, and upserts to Pinecone.
    """
    if not pinecone_index or not google_ai_client:
        raise RuntimeError("Clients are not initialized. The application might not have started correctly.")

    print("Processing document...")
    
    # 1. Load and Extract Text from in-memory bytes
    # NOTE: pypdf's extract_text() is used here for simplicity. For a production system handling
    # diverse and complex PDFs, a more robust library like PyMuPDF or pdfplumber would be a 
    # better choice to handle varied layouts and ensure consistent text extraction order.
    try:
        reader = PdfReader(io.BytesIO(file_content))  # Read from in-memory bytes
        full_document_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        raise ValueError(f"Failed to read or extract text from PDF: {e}")

    if not full_document_text:
        raise ValueError("Could not extract any text from the provided PDF.")

    # 2. Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(full_document_text)
    print(f"Document split into {len(text_chunks)} chunks.")

    # 3. Embed and Upsert to Pinecone
    model_name = "models/text-embedding-004"
    google_batch_size = 100  # Google's text embedding API limit for texts in a batch
    pinecone_upsert_batch_size = 100  # Pinecone's recommended upsert batch size

    all_vectors_to_upsert = [
        {"id": f"chunk_{i}", "metadata": {"text": chunk}, "values": []}
        for i, chunk in enumerate(text_chunks)
    ]

    print(f"Generating embeddings for {len(text_chunks)} chunks...")
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

    # Before upserting a new document, clear all old data from the index.
    # This keeps the demo simple, focusing on one document at a time.
    # NOTE: In a multi-user app, one would use namespaces to isolate data.
    index_stats = await pinecone_index.describe_index_stats()
    if index_stats.total_vector_count > 0:
        print("Clearing previous data from index...")
        await pinecone_index.delete(delete_all=True)

    print(f"Upserting {len(all_vectors_to_upsert)} vectors to Pinecone...")
    for i in range(0, len(all_vectors_to_upsert), pinecone_upsert_batch_size):
        batch_to_upsert = all_vectors_to_upsert[i : i + pinecone_upsert_batch_size]
        await pinecone_index.upsert(vectors=batch_to_upsert)

    print("Document processing and indexing complete.")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _send_message_with_retry(chat_session: Any, prompt: str) -> Any:
    """Helper function to send a message to Gemini with retry logic."""
    print("Generating answer with Gemini...")
    return await chat_session.send_message(prompt)


async def get_rag_answer(query: str, chat_session_id: str | None) -> dict[str, str]:
    """
    Gets an answer from the RAG pipeline for a given query and chat session.
    Manages chat sessions in-memory.
    """
    if not pinecone_index or not google_ai_client:
        raise RuntimeError("Clients are not initialized.")

    # --- Chat Session Management ---
    # If no session ID is provided, create a new one. This allows for conversation history.
    if not chat_session_id or chat_session_id not in chat_sessions:
        print("Creating a new chat session...")
        chat_session_id = str(uuid.uuid4())
        llm_model_identifier = 'gemini-2.0-flash'
        chat_sessions[chat_session_id] = google_ai_client.aio.chats.create(model=llm_model_identifier)

    chat_session = chat_sessions[chat_session_id]

    # 1. Retrieve
    print(f"Embedding query: '{query}'")
    embedding_response = await google_ai_client.aio.models.embed_content(
        model="models/text-embedding-004",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = embedding_response.embeddings[0].values

    print("Querying Pinecone...")
    query_results = await pinecone_index.query(
        vector=query_embedding,
        top_k=5,  # Using 5 for more consistent results based on earlier experiments
        include_metadata=True
    )
    
    # 2. Augment
    context_chunks = [match.metadata['text'] for match in query_results.matches if match.metadata and 'text' in match.metadata]
    context_string = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
    Based ONLY on the following context, answer the question.
    If the answer is not found in the context, state "I cannot answer this question based on the provided information."

    Context:
    {context_string}

    Question: {query}

    Answer:
    """

    # 3. Generate
    response = await _send_message_with_retry(chat_session, prompt)

    return {"answer": response.text, "chat_session_id": chat_session_id}
