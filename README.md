# 🔎 Semantic Search Application with RAG
Status: Work in Progress 👨‍💻 (Actively Under Development)

## 🎯 Purpose
This project focuses on developing an advanced semantic search and question-answering application. The primary goal is to enable users to effectively query custom or lengthy documents, overcoming common limitations of standard Large Language Models (LLMs) related to context window sizes and access to specific, private knowledge sources. This is achieved by implementing a robust Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Core Features (Implemented or In Development)

* **Advanced Document Querying:** Users can query documents using natural language, moving beyond simple keyword matching to understand semantic meaning.
* **Retrieval-Augmented Generation (RAG) Pipeline:** At the heart of the application, this Python-based pipeline integrates LLMs (specifically Google's Gemini) and vector embeddings (via Google's models) for superior contextual understanding and relevant answer generation.
* **Backend API:** A comprehensive API built with FastAPI (Python) manages:
  * Document ingestion and processing (including text splitting using LangChain).
  * Generation of vector embeddings.
  * Efficient storage and retrieval of vectors using Pinecone (Vector Database).
  * Processing of user queries and interaction with the LLM.
* **Responsive Web Interface:** A user-friendly frontend developed with TypeScript, Next.js, and React, allowing for:
  * Easy document uploads.
  * Displaying search and question-answering results.
* **Evaluation Framework (Planned)** A system to evaluate the semantic search capabilities against traditional keyword search methods, employing qualitative analysis and potentially quantitative metrics.

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
  * Google Cloud Platform (GCP) for future deployment

## 📄 Note on Local Document Testing
To test the document querying functionality inside the .ipynb notebooks with the sample document "Attention Is All You Need.pdf", please follow these steps:
* **Paper Link:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. https://arxiv.org/abs/1706.03762
* **Required Local Path:** For the application's current configuration to process this specific document (or others you wish to test locally), please download the PDF and place it in a data directory at the root of this project. For this example paper, the path would be: "data/Attention Is All You Need.pdf".

*This README provides an overview based on the project's current development stage (as of July 2025). More details and code will be added very soon.*
