from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv('deepseek-r1:1.5b.csv', encoding='latin1')

vectorizer = TfidfVectorizer()

def compute_similarity(answer, reference):
    if pd.isna(answer) or pd.isna(reference):
        return 0  # No similarity if either is missing

    tfidf = vectorizer.fit_transform([answer, reference])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

df['cosine_similarity'] = df.apply(lambda row: compute_similarity(row['deepseek-r1:1.5b'], row['Answers']), axis=1)

df.to_csv('evaluation_with_cosine.csv', index=False)

print("✅ Evaluation complete. Scores saved to 'evaluation_with_cosine.csv'")
