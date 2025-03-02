import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Path Constants
CSV_FILE = 'deepseek-r1:1.5b.csv'
VECTOR_STORE_PATH = "persistent_faiss_deepseek"

# Load Data
df = pd.read_csv(CSV_FILE, encoding='latin1')

# Load Vector Store for retrieval (to get contexts)
embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")

vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# Prepare Data for RAGAS - requires question, answer, contexts, and reference (ground truth)
ragas_data = []

# for _, row in df.iterrows():
#     question = row['Question']
#     answer = row['deepseek-r1:1.5b']
#     reference = row['Answers']  # Ground truth answer from your CSV
#     contexts = [doc.page_content for doc in vector_store.similarity_search(question)]

#     ragas_data.append({
#         "question": question,
#         "answer": answer,
#         "reference": reference,
#         "contexts": contexts
#     })

for _, row in df.iterrows():
    question = row['Question']
    answer = row['deepseek-r1:1.5b']
    reference = row['Answers']  # This is your ground truth column

    related_docs = vector_store.similarity_search(question)
    contexts = [doc.page_content for doc in related_docs]

    if not reference or pd.isna(reference):
        print(f"⚠️ Skipping question due to missing reference: {question}")
        continue

    if not contexts:
        print(f"⚠️ No contexts found for question: {question}")
        continue

    print(f"✅ Processing question: {question}")
    print(f"Contexts: {contexts[:2]}")  # Print first 2 for sanity check

    ragas_data.append({
        "question": question,
        "answer": answer,
        "reference": reference,
        "contexts": contexts
    })


# Convert to Hugging Face Dataset
ragas_dataset = Dataset.from_list(ragas_data)

# Run Ragas Evaluation (note: we include `answer_correctness` only if we have reference answers)
scores = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy, answer_correctness]
)

# Print Scores
print(scores)

# Optionally save scores to CSV
scores_df = pd.DataFrame([scores])
scores_df.to_csv("ragas_evaluation_scores.csv", index=False)

print("✅ Evaluation complete. Scores saved to 'ragas_evaluation_scores.csv'.")
