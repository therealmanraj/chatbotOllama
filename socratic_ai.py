import streamlit as st
import time
import re
import os

# Import LangChain components for document loading, splitting, embeddings, vector storage, and LLM prompting.
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ----------------------------
# UI & Styling Configuration
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    /* File Uploader Styling */
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------
# Global Configurations & Models
# ----------------------------
# Prompt template for the RAG pipeline. The LLM is instructed to act as an expert research assistant.
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Define storage path for PDFs.
PDF_STORAGE_PATH = 'document_store/pdfs/'

# Initialize the embedding model and vector store.
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Initialize the language model.
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# ----------------------------
# Utility Functions
# ----------------------------

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded PDF file to a local storage directory.
    """
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

@st.cache_data(show_spinner=False)
def load_pdf_documents(file_path):
    """
    Load PDF documents using PDFPlumberLoader.
    Caching ensures that the same file is not reprocessed repeatedly.
    """
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents, chunk_size=1000, chunk_overlap=200):
    """
    Split raw documents into smaller chunks using RecursiveCharacterTextSplitter.
    The chunk size and overlap are configurable. This step may slow down on very large documents
    because each chunk is processed sequentially. For production use, consider batching or parallel processing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    """
    Index the document chunks in the vector store.
    Note: Sequential processing of a high number of chunks can be slow.
    For large PDFs (300+ pages), consider asynchronous processing or batching.
    """
    start_time = time.time()
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    elapsed = time.time() - start_time
    st.write(f"Indexing Time (Embeddings creation): {elapsed:.2f} seconds")
    return elapsed

def find_related_documents(query):
    """
    Perform a similarity search to retrieve relevant document chunks from the vector store.
    """
    start_time = time.time()
    results = DOCUMENT_VECTOR_DB.similarity_search(query)
    elapsed = time.time() - start_time
    st.write(f"Similarity Search Time: {elapsed:.2f} seconds")
    return results

def generate_answer(user_query, context_documents):
    """
    Generate an answer using a RAG pipeline.
    Concatenate the text from context documents, apply the prompt template, and use the LLM to generate an answer.
    """
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# ----------------------------
# Streamlit Interface & Processing Pipeline
# ----------------------------
st.title("Socratic AI")
st.markdown("Learning Starts Here")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    # Save the file locally and display processing steps with timing
    saved_path = save_uploaded_file(uploaded_pdf)
    st.info("Processing document...")

    # Measure PDF loading time
    t0 = time.time()
    raw_docs = load_pdf_documents(saved_path)
    pdf_load_time = time.time() - t0
    st.write(f"PDF Loading Time: {pdf_load_time:.2f} seconds")

    # Measure text splitting time
    t1 = time.time()
    processed_chunks = chunk_documents(raw_docs)
    split_time = time.time() - t1
    st.write(f"Text Splitting Time: {split_time:.2f} seconds")

    # Index the document chunks into the vector database and log the time taken.
    indexing_time = index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")

    # Chat input for user queries about the document
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Display user's question in the chat interface
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            # Retrieve relevant document chunks based on the query.
            relevant_docs = find_related_documents(user_input)
            # Generate an answer using the RAG pipeline.
            ai_response = generate_answer(user_input, relevant_docs)

        # Remove any unwanted model annotations (if present)
        ai_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()
        
        # Display the assistant's response in the chat interface.
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

# ----------------------------
# Notes on Performance Optimizations
# ----------------------------
st.markdown("---")
st.markdown("**Performance Notes:**")
st.markdown("""
- **Embedding Bottleneck:** Large PDFs may result in hundreds or thousands of text chunks, causing slow embedding processing because each chunk is processed sequentially.  
- **Optimizations:** Consider using batching or asynchronous/multi-threaded processing to create embeddings faster.  
- **Caching:** This app uses Streamlit’s caching (`st.cache_data`) to avoid reprocessing the same PDF, which helps in reducing repeated processing times.
""")
