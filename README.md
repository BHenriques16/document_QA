# ** Document QA with RAG Architecture (Retrieval-Augmented Generation)**

**Natural Language Processing (NLP)** **University of Beira Interior (UBI)** **Academic Year:** 2025/2026

## ** About the Project**

This repository contains the implementation of a **Question Answering (QA)** system for unstructured documents, developed as the final project for the Natural Language Processing course.

The main objective was to create a robust application capable of overcoming the context limitations of traditional LLMs by utilizing a **RAG (Retrieval-Augmented Generation)** architecture. The system ingests long technical documents (e.g., financial reports, technical manuals, scientific articles), fragments them, and allows the user to interact with them using natural language, with guaranteed information traceability (source citation).

### ** Key Features**

* **Multi-Document Ingestion:** Support for batch reading and processing of PDF files.  
* **Persistent Vectorization:** Creation of a local Vector Store to avoid reprocessing data.  
* **100% Local LLM:** Utilization of Open Source models (Llama 3.2 or Mistral) via Ollama, ensuring data privacy.  
* **Interactive Interface:** Web application developed in Streamlit with chat history and context management.  
* **Hallucination Prevention:** The system explicitly indicates the document and page number from which the answer was extracted.

## ** Tech Stack**

The project was developed in **Python** using the following libraries and tools:

| Component | Technology | Description |
| :---- | :---- | :---- |
| **LLM Server** | [Ollama](https://ollama.com/) | Local execution of Llama/Mistral models. |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Framework to connect the LLM to the data. |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Database for storing embeddings. |
| **Embeddings** | mxbai-embed-large | High-performance text vectorization model. |
| **Interface** | [Streamlit](https://streamlit.io/) | Frontend for user interaction. |
| **PDF Parsing** | pypdf | Extraction of text and metadata from files. |

## ** Installation and Setup**

### **1\. Prerequisites**

Ensure you have installed:

* Python 3.8 or higher.  
* [Ollama](https://ollama.com/) (to run the model locally).

### **2\. Model Configuration (Ollama)**

In your terminal, pull the necessary models:

\# Language Model (Chat)  
ollama pull llama3.2

\# Embedding Model (Vectorization)  
ollama pull mxbai-embed-large

### **3\. Dependency Installation**

It is recommended to create a virtual environment. Install the necessary Python libraries:

pip install langchain langchain-community langchain-ollama langchain-chroma streamlit pypdf

## ** How to Use**

The workflow is divided into two phases: **Data Ingestion** and **Interaction**.

### **Step 1: Data Preparation (Ingestion)**

1. Place your PDF files in the data/ folder.  
2. Run the vectorization script. This script will read the documents, split the text into chunks, and save the embeddings in ChromaDB.

python vector.py

*Note: Run this step only for the first time or whenever you add new documents to the data folder.*

### **Step 2: Run the Application**

Start the Streamlit web interface:

streamlit run app.py

The application will be available in your browser at http://localhost:8501.

## ** Repository Structure**

.  
├── data/                     \# Directory to place input PDF files  
├── chroma\_db\_financas/       \# Vector database (automatically generated)  
├── vector.py                 \# ETL Pipeline Script (Extract, Transform, Load)  
├── app.py                    \# Main Application (Frontend Streamlit \+ RAG Chain)  
└── README.md                 \# Project documentation

## ** Implementation Details**

### **Chunking Strategy**

To handle extensive documents, the RecursiveCharacterTextSplitter was used with:

* **Chunk Size:** 1500 characters (optimized to capture financial tables and complete paragraphs).  
* **Overlap:** 300 characters (to maintain semantic context between splits).

### **Retrieval**

The **MMR (Maximal Marginal Relevance)** algorithm is used for document retrieval to ensure diversity in the cited sources and prevent a single document from dominating the LLM context window.

## ** Author**

**Student Name** Student Number: XXXXX

Computer Science / Data Science

University of Beira Interior