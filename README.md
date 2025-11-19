#  Document Question Answering (QA) using RAG Architecture


##  Project Overview

This repository features the implementation of a Document Question Answering (DQA) system, developed as the final project for the Natural Language Processing course.

The system is built upon a Retrieval-Augmented Generation (RAG) architecture to overcome the context limitations of Large Language Models (LLMs). The application acts as a "Financial Analyst" capable of extracting, synthesizing, and comparing critical data from complex corporate filings (e.g., 10-Q reports), specifically designed to investigate the "AI Bubble" hypothesis by comparing CapEx and Cloud Growth between major tech companies (Microsoft vs. Alphabet).

###  NLP Context and Core Areas

This project sits at the intersection of three fundamental NLP subfields:

1. Question Answering (QA): The primary goal is to answer natural language questions based on the provided documents (Generative QA).  
2. Information Retrieval (IR): Utilized for the embedding and retrieval phases (mxbai-embed-large and ChromaDB) to find semantically relevant text fragments (chunks) using Vector Search.  
3. Information Extraction (IE): The LLM performs complex extraction and synthesis of facts (e.g., specific CapEx figures, growth percentages) from the retrieved text.

###  Key Technical Features

* RAG Pipeline: End-to-end implementation of an Ingestion, Retrieval, and Generation pipeline.  
* Local LLM: Utilizes 100% local, open-source models (Llama 3.2 or Mistral) via Ollama, ensuring data privacy.  
* Multi-Document Analysis: Capable of ingesting and comparing data from multiple distinct PDF reports.  
* High Fidelity: Employs source citation (document name and page number) to prevent hallucinations.  
* Interactive Interface: A professional web interface built with Streamlit for dynamic demonstrations.

##  Tech Stack and Dependencies

The project is built using Python and the following tools:

| Component | Technology | Description |
| :---- | :---- | :---- |
| LLM Server | [Ollama](https://ollama.com/) | Local serving of LLM and embedding models. |
| Orchestration | [LangChain](https://www.langchain.com/) | Framework managing the RAG chain components. |
| Vector Store | [ChromaDB](https://www.trychroma.com/) | Persistent local storage for document embeddings. |
| Embeddings | mxbai-embed-large | High-quality vectorization model for semantic search. |
| Interface | [Streamlit](https://streamlit.io/) | Frontend for the chat application. |
| PDF Parsing | pypdf | Text and metadata extraction from PDF files. |

##  Installation and Setup

### 1\. Prerequisites

Ensure the following tools are installed on your system:

* Python 3.8 or higher.  
* [Ollama](https://ollama.com/) (required for local model execution).

### 2\. Model Configuration (Ollama)

Download the necessary models using your terminal:

\# Language Model (Chat)  
ollama pull llama3.2

\# Embedding Model (Vectorization)  
ollama pull mxbai-embed-large

### 3\. Dependency Installation

It is recommended to create a virtual environment. Install the necessary Python libraries:

pip install langchain langchain-community langchain-ollama langchain-chroma streamlit pypdf

##  Usage

The application workflow has two stages: Data Ingestion and Interactive Querying.

### Step 1: Data Ingestion

1. Place your target PDF files (e.g., 10-Q reports) into the ./data/ folder.  
2. Execute the ingestion script to create the vector database:

python vector.py

*(Run this step only the first time or whenever you add/change documents.)*

### Step 2: Run the Application

Start the web application using Streamlit:

streamlit run app.py

The app will automatically open in your browser at http://localhost:8501.

##  Implementation Details

### Chunking Strategy

To handle extensive documents, the RecursiveCharacterTextSplitter was used with the following parameters, optimized for financial reports:

* Chunk Size: 1500 characters (large enough to capture context around numbers).  
* Overlap: 300 characters (to maintain semantic continuity).

### Retrieval Mechanism

The retrieval phase utilizes the MMR (Maximal Marginal Relevance) algorithm (search\_type="mmr") to prioritize both relevance to the query and diversity among the retrieved documents, preventing bias towards one single source when comparing multiple companies.
