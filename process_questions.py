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

# Models and corresponding FAISS directories
MODELS = {
    # "llama3.2:latest": "persistent_faiss_llama3_2",
    # "gemma:2b": "persistent_faiss_gemma2b",
    "llama2:7b": "persistent_faiss_llama2_7b",
    # "deepseek-r1:1.5b": "persistent_faiss_deepseek"
}

# Load all vector stores and language models
MODEL_CONFIGS = {}

for model_name, vector_store_path in MODELS.items():
    embedding_model = OllamaEmbeddings(model=model_name)
    language_model = OllamaLLM(model=model_name)

    vector_store = FAISS.load_local(
        vector_store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    MODEL_CONFIGS[model_name] = {
        "vector_store": vector_store,
        "language_model": language_model
    }

# Load questions from CSV
df = pd.read_csv('Combined(Mani).csv', encoding='latin1')

# Ensure all model columns exist (but don't remove anything already there)
for model_name in MODELS.keys():
    if model_name not in df.columns:
        df[model_name] = None  # Add blank column if missing

# Function to generate answers using a given model's vector store and LLM
def generate_answer(model_name, user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])

    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | MODEL_CONFIGS[model_name]["language_model"]

    ai_response = response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # Clean up <think> sections if present
    return re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()

# Function to process a single question across all models
def process_question_for_all_models(question):
    answers = {"Question": question}

    for model_name, config in MODEL_CONFIGS.items():
        vector_store = config["vector_store"]
        related_docs = vector_store.similarity_search(question)

        ai_response = generate_answer(model_name, question, related_docs)

        answers[model_name] = ai_response

    return answers

# Process all questions in parallel with progress bar
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_question = {executor.submit(process_question_for_all_models, question): question for question in df['Question']}

    for future in tqdm(as_completed(future_to_question), total=len(future_to_question), desc="Processing Questions"):
        result = future.result()
        results.append(result)

# Convert results into DataFrame
results_df = pd.DataFrame(results)

# Merge new answers into the original DataFrame, keeping existing columns intact
df = df.merge(results_df, on="Question", how="left")

# Save to CSV (keeping all original columns + new answers)
df.to_csv('Combined(Mani)_with_answers.csv', index=False)

print("✅ All answers saved to 'Combined(Mani)_with_answers.csv'")
