import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever

# Page configuration
st.set_page_config(
    page_title="Wall St. Analyst (Llama)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    h1 {
        color: #1E3A8A; /* Dark Blue */
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: Context and Controls
with st.sidebar:
    st.title("Financial Report Analyzer")
    
    # ----------------------------------------

    st.markdown("---")
    st.markdown("###  Data Sources")
    st.info(
        """
        Analyzing **10-Q Filings** (Quarterly Reports)
        focusing on the **AI Bubble Hypothesis**.
        
        - **Microsoft** (Q1 FY26)
        - **Alphabet** (Q3 2025)
        """
    )
    st.markdown("---")
    if st.button(" Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Main Header 
st.title(" Financial Report Analyst (Llama 3.2)")

# Caching Resources (Model & DB)
@st.cache_resource
def load_llm_and_chain():
    # Explicitly using Llama 3.2 here
    model = OllamaLLM(model='llama3.2')
    
    template = '''
    You are a and Financial Analyst and expert analyzing 10-Q reports from Wall Street. 
    Your goal is to investigate the "AI Bubble" hypothesis by analyzing the provided 10-Q reports.

    Focus on:
    1. Capital Expenditures (CapEx): massive spending on servers/infrastructure.
    2. Cloud Revenue Growth: Is Azure (Microsoft) or Google Cloud growing faster?
    3. AI Monetization: Specific mentions of revenue from "Copilot" or "Gemini".

    Warning: Microsoft's "Q1 FY26" corresponds to the same calendar period as Alphabet's "Q3 2025" (Quarter ending Sept 30). 
    Treat them as comparable.

    Here is the context: {context}
    Here is the user Question: {question}

    '''
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain

@st.cache_resource
def load_database_retriever():
    return get_retriever()

# Load resources with error handling
try:
    chain = load_llm_and_chain()
    retriever = load_database_retriever()
except Exception as e:
    st.error(f"CRITICAL ERROR: Could not load database. Details: {e}")
    st.stop()

# Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I am running on Llama 3.2 with RAG enabled. I have analyzed the latest 10-Q reports."}
    ]


# Chat Input & Processing
user_input = st.chat_input("Ask your question about the reports...")

if user_input:
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Llama is analyzing financial documents..."):
            try:
                # Retrieval
                retrieved_docs = retriever.invoke(user_input)
                formatted_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generation
                response = chain.invoke({'context': formatted_context, 'question': user_input})
                
                # Formatting Sources
                unique_sources = set()
                for doc in retrieved_docs:
                    source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page_num = doc.metadata.get('page', '?')
                    source_info = f"-  **{source_name}** (Page {int(page_num) + 1})"
                    unique_sources.add(source_info)
                
                # Display the main answer
                st.markdown(response)
                
                # Display sources
                with st.expander(" View Cited Sources"):
                    for source in unique_sources:
                        st.markdown(source)

                # Save full response
                full_response_to_save = response
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response_to_save = "I encountered an error processing your request."

    st.session_state.messages.append({"role": "assistant", "content": full_response_to_save})