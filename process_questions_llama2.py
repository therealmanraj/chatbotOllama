
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

# Constants
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

VECTOR_STORE_PATH = "persistent_faiss_llama2_7b"
EMBEDDING_MODEL = OllamaEmbeddings(model="llama2:7b")
LANGUAGE_MODEL = OllamaLLM(model="llama2:7b")

# Load FAISS Vector Store with safe deserialization enabled
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH, 
    EMBEDDING_MODEL, 
    allow_dangerous_deserialization=True
)

# Load questions from CSV
df = pd.read_csv('Combined(Mani).csv', encoding='latin1')

# Ensure column exists for storing answers
if 'llama2:7b' not in df.columns:
    df['llama2:7b'] = None

# Function to generate answers using the vector store and language model
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    ai_response = response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # Clean up <think> sections if present
    return re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()

# Function to process a single question
def process_question(question):
    related_docs = vector_store.similarity_search(question)
    return question, generate_answer(question, related_docs)

# Process all questions in parallel with progress bar
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_question = {executor.submit(process_question, question): question for question in df['Question']}
    
    for future in tqdm(as_completed(future_to_question), total=len(future_to_question), desc="Processing Questions"):
        question, ai_response = future.result()
        results.append((question, ai_response))

# Update DataFrame with answers
for question, ai_response in results:
    df.loc[df['Question'] == question, 'llama2:7b'] = ai_response

# Save to CSV
df.to_csv('llama2:7b', index=False)

print("✅ All answers saved to 'llama2:7b.csv'")
