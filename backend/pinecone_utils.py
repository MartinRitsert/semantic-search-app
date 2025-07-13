"""
This module contains utility functions for interacting with Pinecone.
It includes functions for checking index consistency and deleting namespaces.
"""


import httpx
import logging

# Import clients
import clients


# Get a logger for this module.
logger = logging.getLogger(__name__)


async def check_index_consistency(target_lsn: int, namespace: str) -> bool:
    """Checks if the index has reached or passed a target Log Sequence Number (LSN).
    If it has reached or exceeded the target LSN, it is considered consistent and ready for queries."""
    if not clients.pinecone_http_client:
        return False
    
    try:
        # Query with a dummy vector to get the current max indexed LSN from headers.
        dummy_vector = [0.0] * 768 # Dimension for Google's embedding-004 model
        response = await clients.pinecone_http_client.post(
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
        await clients.pinecone_index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Delete command for namespace '{namespace}' acknowledged.")
    except Exception as e:
        logger.error(f"Failed to issue delete for namespace '{namespace}': {e}")        
