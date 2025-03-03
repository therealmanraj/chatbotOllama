import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Force sequential processing (important for MacBooks)
os.environ['RAGAS_MAX_WORKERS'] = '1'

# Paths
CSV_FILE = 'deepseek-r1:1.5b.csv'
VECTOR_STORE_PATH = "persistent_faiss_deepseek"

# Load Data
df = pd.read_csv(CSV_FILE, encoding='latin1')
df = df.head(5)

# Load Vector Store
embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Use smaller model for evaluation if deepseek is too heavy
# local_llm = OllamaLLM(model="gemma:2b", timeout=300)
local_llm = OllamaLLM(model="deepseek-r1:1.5b", timeout=300)


# Prepare Data for RAGAS
ragas_data = []

for _, row in df.iterrows():
    question = row['Question']
    answer = row['deepseek-r1:1.5b']
    reference = row['Answers']

    # Retrieve contexts from FAISS
    related_docs = vector_store.similarity_search(question)
    contexts = [doc.page_content for doc in related_docs]

    # Pre-check for debugging
    if pd.isna(reference) or not reference.strip():
        print(f"⚠️ Missing reference for question: {question}")

    if pd.isna(answer) or not answer.strip():
        print(f"⚠️ Missing AI answer for question: {question}")

    if not contexts:
        print(f"⚠️ No contexts found for question: {question}")

    # Append to RAGAS dataset
    ragas_data.append({
        "question": question,
        "answer": answer,
        "reference": reference,
        "contexts": contexts
    })

# Convert to Hugging Face Dataset
ragas_dataset = Dataset.from_list(ragas_data)

# ✅ Evaluate using local model
scores = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy, answer_correctness],
    llm=local_llm
)

# Print and Save Scores
print(scores)

scores_df = pd.DataFrame([scores])
scores_df.to_csv("ragas_evaluation_scores.csv", index=False)

print("✅ Evaluation complete. Scores saved to 'ragas_evaluation_scores.csv'")
