import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# --- Streamlit Styles ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    ai_response = response_chain.invoke({
        "user_query": user_query,
        "document_context": context_text
    })
    return ai_response.strip()

def process_question(question):
    relevant_docs = st.session_state.vector_store.similarity_search(question)
    ai_response = generate_answer(question, relevant_docs)

    # Clean response
    ai_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()
    return question, ai_response


# --- UI Configuration ---
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# --- File Upload ---
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

# --- Load Questions from CSV ---
df = pd.read_csv('Combined(Mani).csv', encoding='latin1')

# Ensure result column exists
if 'deepseek-r1:1.5b' not in df.columns:
    df['deepseek-r1:1.5b'] = None

# --- PDF Processing (With FAISS Caching) ---
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)

    if "vector_store" not in st.session_state or st.session_state.last_uploaded != uploaded_pdf.name:
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)

        # Create FAISS Vector Store
        vector_store = FAISS.from_documents(processed_chunks, EMBEDDING_MODEL)
        vector_store.save_local(VECTOR_STORE_PATH)

        # Cache in session state
        st.session_state.vector_store = vector_store
        st.session_state.last_uploaded = uploaded_pdf.name
    else:
        # Load from FAISS cache if already processed
        st.session_state.vector_store = FAISS.load_local(VECTOR_STORE_PATH, EMBEDDING_MODEL)

    st.success("✅ Document processed successfully! Ask your questions below.")

    # --- Process All Questions (Parallel) ---
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_question, df['Question']))

    # --- Update DataFrame ---
    for question, ai_response in results:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

        df.loc[df['Question'] == question, 'deepseek-r1:1.5b'] = ai_response

    # --- Save Results ---
    df.to_csv('deepseek-r1:1.5b.csv', index=False)

    st.success("✅ All answers saved to 'deepseek-r1:1.5b.csv'!")

