"""
This module contains the clients used to interact with external services.
It initializes the Pinecone and Google AI clients, and handles the closing of these clients.
"""


import os
import httpx
from pinecone import Pinecone, ServerlessSpec
from google import genai
import logging


# Get a logger for this module.
logger = logging.getLogger(__name__)


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


