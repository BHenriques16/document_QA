from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

# Configs
DATA_PATH = "./data_software"  # documents
DB_PATH = "./chroma_db_finance"
EMBEDDING_MODEL = "mxbai-embed-large"

def create_vector_db():
    # Check for PDFs in the folder
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"ERROR: The folder '{DATA_PATH}' doesnt exist or is empty.")
        return None

    print(" Loading financial reports (PDFs)...")
    
    # DirectoryLoader uses PyPDFLoader for every file .pdf
    loader = DirectoryLoader(
        path=DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f" Loaded {len(documents)} documents.")

    # Text splitting in chunks
    print('Splittig text in chunks')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,    # Bigger chunks for financial reports
        chunk_overlap=300   # Makes sure that numbers and context dont go missing when the text is split
    )
    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} text chunks created.")

    # Create/Update vector store
    print('Saving vectorial base')
    
    # Incializes the embeddings model
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Creates the database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,                       
        persist_directory=DB_PATH       # Avoid having to reprocess documents every time you run the program.                                  
    )
    print('Database created successfully.')
    return vector_store

def get_retriever():
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # If the database folder doesnt exist, create it first
    if not os.path.exists(DB_PATH):
        create_vector_db()
        
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )
    
    # k = i recovers the i most relevant excerpts
    return vector_store.as_retriever(search_kwargs={"k": 10})

if __name__ == "__main__":
    create_vector_db()