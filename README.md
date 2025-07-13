# 🔎 Semantic Search RAG Application
Status: Work in Progress 👨‍💻 (Actively Under Development)

## 🎬 Application Preview
https://github.com/user-attachments/assets/8a92bdc7-c560-4633-85e7-439a44024673

## 🎯 Purpose
This project focuses on developing an advanced semantic search and question-answering application. The primary goal is to enable users to effectively query custom or lengthy documents, overcoming common limitations of standard Large Language Models (LLMs) related to context window sizes and access to specific, private knowledge sources. This is achieved by implementing a robust Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Core Features (Implemented or In Development)
* **Advanced Document Querying:** Users can query documents using natural language, moving beyond simple keyword matching to understand semantic meaning.
* **Retrieval-Augmented Generation (RAG) Pipeline:** At the heart of the application, this Python-based pipeline integrates an LLM of the Google Gemini family and vector embeddings for superior contextual understanding and relevant answer generation.
* **Backend API:** A comprehensive API built with FastAPI (Python) manages:
  * Document ingestion and processing (including text splitting).
  * Generation of vector embeddings.
  * Efficient storage and retrieval of vectors (Vector Database).
  * Processing of user queries and interaction with an LLM.
* **Responsive Web Interface:** A user-friendly frontend, allowing for:
  * Easy document uploads.
  * Displaying question-answering results.

## 🛠️ Technology Stack
* **Core ML & Backend:**
  * Python
  * FastAPI
  * Google AI (Gemini LLM & Embeddings)
  * Pinecone (Vector Database)
  * LangChain (for Text Splitting and RAG components)
* **Frontend:**
  * TypeScript
  * React
  * Next.js
* **Version Control & DevOps:**
  * Git
  * Docker
  * Google Cloud Platform (GCP)

## 🏁 Getting Started
This project is split into a `frontend` and a `backend` directory, each with its own setup instructions.
1.  **Backend Setup**: See the [backend/README.md](./backend/README.md) for instructions on setting up the Python environment and API.
2.  **Frontend Setup**: See the [frontend/README.md](./frontend/README.md) for instructions on setting up the Next.js application.

## Usage
1.  Start the backend server.
2.  Start the frontend development server.
3.  Open your browser to the frontend URL.
4.  Upload a PDF document.
5.  Once the document is processed, you can ask questions about its content.

*This README provides an overview based on the project's current development stage (as of July 2025). More details and code will be added very soon.*