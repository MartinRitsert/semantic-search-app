# Backend (FastAPI)
This directory contains the Python backend for the semantic search application, built with FastAPI.

## Prerequisites
*   Python 3.9+
*   A virtual environment tool (e.g., `venv`)

## Setup
1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create an environment file:**
    Create a `.env` file in this directory and add your API keys:
    ```env
    # .env
    GOOGLE_API_KEY="your_google_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    ```

## Running the Application
To start the development server, run the following command from the `backend` directory:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.