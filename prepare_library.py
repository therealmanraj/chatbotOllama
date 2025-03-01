import os
from tqdm import tqdm
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Paths and Constants
PDF_DIRECTORY = "document_library/"

MODELS = {
    # "phi2": "persistent_faiss_phi2",
    # "orca-mini-3b": "persistent_faiss_orca3b",
    "llama3.2:latest": "persistent_faiss_llama3_2",
    "gemma:2b": "persistent_faiss_gemma2b",
    "llama2:7b": "persistent_faiss_llama2_7b",
    "deepseek-r1:1.5b": "persistent_faiss_deepseek"
}

BATCH_SIZE = 50  # Adjust based on available memory


def batch_embed_texts(texts, batch_size, model_name):
    """Batch embed texts using the specified embedding model."""
    embedding_model = OllamaEmbeddings(model=model_name)

    vector_store = None
    for i in tqdm(range(0, len(texts), batch_size), desc=f"⚡ Embedding in Batches ({model_name})"):
        batch = texts[i:i + batch_size]
        batch_vectors = FAISS.from_texts(batch, embedding_model)

        if vector_store is None:
            vector_store = batch_vectors
        else:
            vector_store.merge_from(batch_vectors)

    return vector_store


def process_and_store_documents():
    all_documents = []
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]

    print(f"📚 Found {len(pdf_files)} PDFs to process...")

    for filename in tqdm(pdf_files, desc="📄 Loading PDFs"):
        loader = PDFPlumberLoader(os.path.join(PDF_DIRECTORY, filename))
        docs = loader.load()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        all_documents.extend(chunks)

    print(f"📄 Total chunks created: {len(all_documents)}")

    chunk_texts = [chunk.page_content for chunk in tqdm(all_documents, desc="🔎 Preparing text for embedding")]

    # Process and save for each model
    for model_name, vector_store_path in MODELS.items():
        print(f"\n🚀 Processing embeddings for model: {model_name}")

        os.makedirs(vector_store_path, exist_ok=True)
        vector_store = batch_embed_texts(chunk_texts, BATCH_SIZE, model_name)

        vector_store.save_local(vector_store_path)
        print(f"✅ Vector store saved for {model_name} with {len(chunk_texts)} chunks in '{vector_store_path}'.")

if __name__ == "__main__":
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    process_and_store_documents()
