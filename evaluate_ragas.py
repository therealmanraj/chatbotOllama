import time
import os
import re

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the prompt for the LLM.
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

# Set up paths and model configurations.
PDF_STORAGE_PATH = 'document_store/pdfs/'
PDF_FILE = os.path.join(PDF_STORAGE_PATH, "iesc101.pdf")  # Replace with your test PDF file

# Initialize embedding and language models.
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def load_pdf_documents(file_path):
    """Load and extract text from a PDF document."""
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    """Split the loaded documents into manageable text chunks."""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    """Add text chunks to the vector database."""
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    """Retrieve the most relevant text chunks for the query."""
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    """Generate a concise answer using the provided context."""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def evaluate_pipeline():
    print("Starting RAG evaluation pipeline...\n")
    
    # Step 1: Load the PDF.
    start_time = time.time()
    raw_docs = load_pdf_documents(PDF_FILE)
    load_time = time.time() - start_time
    print(f"PDF loading time: {load_time:.2f} seconds")
    
    # Step 2: Chunk documents.
    start_time = time.time()
    processed_chunks = chunk_documents(raw_docs)
    chunk_time = time.time() - start_time
    print(f"Text chunking time: {chunk_time:.2f} seconds")
    print(f"Total chunks created: {len(processed_chunks)}")
    
    # Step 3: Index documents (vector database creation).
    start_time = time.time()
    index_documents(processed_chunks)
    index_time = time.time() - start_time
    print(f"Indexing time: {index_time:.2f} seconds")
    
    # Step 4: Evaluate on sample queries.
    sample_queries = [
        "What is the main idea of the document?",
        "Summarize the key findings in the document.",
        "What conclusions does the document reach?"
    ]
    
    for query in sample_queries:
        print("\n------------------------")
        print(f"Query: {query}")
        start_time = time.time()
        relevant_docs = find_related_documents(query)
        answer = generate_answer(query, relevant_docs)
        # Clean up any debugging or extraneous tokens.
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        query_time = time.time() - start_time
        print(f"Response time: {query_time:.2f} seconds")
        print("Answer:")
        print(answer)
    
    print("\nRAG evaluation completed.")

if __name__ == "__main__":
    evaluate_pipeline()
