import streamlit as st
import time 

# Importar a nossa lógica do outro ficheiro
# (Assume que o processing.py é o "Plano Final" com Mistral-7B em CPU 
# e return_full_text=False)
from processing import (
    carregar_modelo_embeddings, 
    extrair_texto_pdf, 
    criar_base_vetorial,
    carregar_llm  
)

from langchain_core.prompts import PromptTemplate

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Document-QA (Multi-Documento)",
    page_icon="📄",
    layout="wide"
)

# --- 2. Título Principal ---
st.title("📄 Document-QA: Pergunte aos seus Documentos")
st.markdown("""
Bem-vindo ao Document-QA. 
Carregue um ou mais documentos PDF e faça perguntas sobre o seu conteúdo.
""")

# --- 3. Barra Lateral (Sidebar) para Upload (COM SUPORTE MULTI-DOCUMENTO) ---
with st.sidebar:
    st.header("Os seus Documentos")
    
    # --- MUDANÇA (Multi-Documento) ---
    uploaded_files = st.file_uploader(
        "Carregue os seus ficheiros .pdf aqui:", 
        type=["pdf"],
        accept_multiple_files=True  # A mudança mágica
    )

    # Inicializar o 'session_state'
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "llm" not in st.session_state: 
        st.session_state.llm = None

    # --- MUDANÇA (Loop de Processamento) ---
    if uploaded_files: # Se a lista não estiver vazia
        
        # Mostra o nome de todos os ficheiros carregados
        for f in uploaded_files:
            st.info(f"Ficheiro carregado: {f.name}")
        
        if st.button("Processar Documentos"): # Texto do botão atualizado
            
            # 1. Carregar o modelo de Embeddings
            embeddings = carregar_modelo_embeddings()
            
            # 2. Extrair o texto de TODOS os ficheiros
            texto_completo = ""
            with st.spinner("A ler todos os ficheiros..."):
                for pdf_file in uploaded_files:
                    # Acumula o texto de cada ficheiro
                    texto_completo += extrair_texto_pdf(pdf_file) 
            
            if texto_completo:
                # 3. Criar a Base de Dados Vetorial UMA SÓ VEZ com todo o texto
                st.session_state.vector_db = criar_base_vetorial(texto_completo, embeddings)
                
                # 4. Carregar o LLM (Mistral-7B na CPU)
                st.session_state.llm = carregar_llm() 
                
                if st.session_state.vector_db and st.session_state.llm:
                    st.success("Documentos processados e IA carregada. Pronto para perguntas!")
    else:
        st.warning("Por favor, carregue um ou mais documentos PDF para começar.")


# --- 4. Área Principal de Chat ---
# (Esta secção é IDÊNTICA à versão "Plano Final")

st.header("Faça uma Pergunta")

if st.session_state.vector_db is None or st.session_state.llm is None:
    st.info("Por favor, carregue e processe um ou mais documentos na barra lateral para começar.")
    query = st.text_input(
        "Sobre o que quer saber?", 
        placeholder="Aguardando processamento dos documentos...",
        disabled=True,
        label_visibility="collapsed"
    )
else:
    query = st.text_input(
        "Sobre o que quer saber?", 
        placeholder="Escreva a sua pergunta aqui...",
        disabled=False,
        label_visibility="collapsed"
    )

# Espaço reservado para a Resposta
st.subheader("Resposta")
answer_container = st.container(border=True, height=200)

# Espaço reservado para as Fontes
st.subheader("Fontes dos Documentos")
source_container = st.container(border=True, height=250)

# Lógica RAG (Plano Final: Manual, Mistral, CPU-safe)
if query and st.session_state.vector_db and st.session_state.llm:
    
    with st.spinner("A pensar..."):
        
        # 1. O Prompt (Formato Mistral)
        template = """
        [INST]
        Você é um assistente prestável. Use os seguintes trechos de um documento para responder à pergunta.
        A sua resposta deve ser clara e concisa, em Português.
        Se os trechos não contiverem a resposta, diga "Peço desculpa, mas não encontrei essa informação no documento."
        
        Contexto (Trechos do Documento):
        {context}
        
        Pergunta:
        {question}
        [/INST]
        Resposta: 
        """
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # 2. O Retriever (k=3 está ótimo para Mistral)
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

        # 3. Executar o "Retrieval"
        docs = retriever.invoke(query)
        
        # 4. Formatar o Contexto
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 5. Formatar o Prompt Final
        prompt_final = PROMPT.format(context=context_text, question=query)
        
        # 6. Chamar o LLM (O Cérebro)
        # (Assumindo que o processing.py tem 'return_full_text=False')
        resultado_limpo = st.session_state.llm.invoke(prompt_final)
        
        # 7. Mostrar os Resultados
        answer_container.markdown(resultado_limpo)
        
        source_container.empty()
        for i, doc in enumerate(docs):
            source_container.markdown(f"**Fonte {i+1} (dos Documentos):**")
            source_container.info(f"_{doc.page_content}_")
            
elif not query:
    answer_container.write("A resposta do documento aparecerá aqui.")
    source_container.write("Os trechos do documento usados para a resposta aparecerão aqui.")