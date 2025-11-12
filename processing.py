import streamlit as st
import PyPDF2
from io import BytesIO

# Importações do LangChain necessárias para a lógica
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Novas importações necessárias para o LLM (Fase 3) ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # <--- MUDANÇA AQUI
from langchain_community.llms import HuggingFacePipeline 
import torch
# --- Funções de Processamento em Cache ---
# Movemos toda a lógica para aqui

@st.cache_resource
def carregar_modelo_embeddings():
    """
    Carrega o modelo de embeddings (all-MiniLM-L6-v2) do Hugging Face.
    Isto transforma texto em vetores (números).
    
    (Nota: @st.cache_resource significa que esta função precisa de 'st',
    por isso importamos streamlit as st aqui também)
    """
    with st.spinner("A carregar o modelo de embeddings... (Isto pode demorar na primeira vez)"):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Forçar o uso de CPU
        )
    return embeddings

def extrair_texto_pdf(ficheiro_pdf):
    """
    Extrai o texto de um ficheiro PDF carregado pelo Streamlit.
    """
    try:
        pdf_file_object = BytesIO(ficheiro_pdf.read())
        pdf_reader = PyPDF2.PdfReader(pdf_file_object)
        
        texto = ""
        for page in pdf_reader.pages:
            texto += page.extract_text()
            
        if not texto:
            st.error("Não foi possível extrair texto do PDF. O ficheiro pode estar corrompido ou ser uma imagem.")
            return None
            
        return texto
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
        return None

def criar_base_vetorial(texto, embeddings):
    """
    Divide o texto em chunks e cria a base de dados vetorial (ChromaDB).
    """
    # 1. Dividir o Texto (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200  
    )
    chunks = text_splitter.split_text(texto)
    
    if not chunks:
        st.error("Não foi possível dividir o texto em 'chunks'.")
        return None

    # 2. Criar a Base Vetorial (Indexação)
    with st.spinner(f"A criar a base de dados vetorial com {len(chunks)} 'chunks' de texto..."):
        db = Chroma.from_texts(chunks, embeddings)
    return db


# --- Novas importações necessárias para o LLM (Fase 3) ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline # Wrapper do LangChain para LLMs do HuggingFace
import torch

# --- Variável Global para o ID do Modelo ---
# Vamos começar com o GPT-2. É pequeno, rápido de descarregar e corre em CPU.
# AVISO: A sua qualidade de resposta será MUITO FRACA! É apenas para testar o pipeline.
MODELO_ID = "microsoft/Phi-3-mini-4k-instruct"
# MODELO_ID = "google/flan-t5-base" # Uma opção melhor (e ainda pequena)
# MODELO_ID = "mistralai/Mistral-7B-Instruct-v0.1" # Uma opção excelente (requer boa GPU)


@st.cache_resource
def carregar_llm():
    """
    Carrega o LLM (Modelo de Linguagem) do Hugging Face.
    Esta versão usa o Phi-3-mini, um modelo de Chat CausalLM.
    """
    with st.spinner(f"A carregar o LLM ({MODELO_ID})... Isto vai demorar e requer VRAM."):
        
        tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)
        
        # --- MUDANÇA IMPORTANTE ---
        # Voltamos ao 'AutoModelForCausalLM' como o gpt2,
        # mas este é um modelo moderno.
        model = AutoModelForCausalLM.from_pretrained(
            MODELO_ID,
            device_map="auto",             # Tenta usar a GPU (cuda) se disponível
            torch_dtype="auto",            # Usa o melhor tipo de dados (ex: float16)
            trust_remote_code=True,        # Obrigatório para o Phi-3
        )

        # --- MUDANÇA IMPORTANTE ---
        # A "tarefa" (task) do pipeline volta a ser "text-generation"
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Limita a *nova* geração
            temperature=0.6,     
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        return HuggingFacePipeline(pipeline=pipe)