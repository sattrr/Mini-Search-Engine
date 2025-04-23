import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_RESULTS_PATH = PROCESSED_DATA_DIR / "tfidf_results.json"
SIMILARITY_RESULTS_PATH = PROCESSED_DATA_DIR / "cosine_similarity_results.json"

with open(TFIDF_RESULTS_PATH, "r", encoding="utf-8") as f:
    tfidf_results = json.load(f)

corpus_tf_idf = tfidf_results['corpus_tf_idf']
query_tf_idf = tfidf_results['query_tf_idf']

corpus_tokens = list(corpus_tf_idf[next(iter(corpus_tf_idf))].keys())
query_tokens = list(query_tf_idf[next(iter(query_tf_idf))].keys())

corpus_matrix = np.array([[corpus_tf_idf[doc_id].get(token, 0) for token in corpus_tokens] for doc_id in corpus_tf_idf])
query_matrix = np.array([[query_tf_idf[query_id].get(token, 0) for token in query_tokens] for query_id in query_tf_idf])

corpus_matrix_normalized = normalize(corpus_matrix, norm="l2", axis=1)
query_matrix_normalized = normalize(query_matrix, norm="l2", axis=1)

def calculate_cosine_similarity(query_matrix, corpus_matrix):
    similarity_scores = cosine_similarity(query_matrix, corpus_matrix)
    return similarity_scores

similarity_scores = {}
for i, query_vector in enumerate(query_matrix_normalized):
    similarity_scores[i] = calculate_cosine_similarity(query_vector.reshape(1, -1), corpus_matrix_normalized)

similarity_scores_list = {query_idx: scores.tolist() for query_idx, scores in similarity_scores.items()}

with open(SIMILARITY_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(similarity_scores_list, f, ensure_ascii=False, indent=4)

for query_idx, scores in similarity_scores_list.items():
    print(f"Query {query_idx+1} Similarities:")
    for doc_idx, score in enumerate(scores):
        print(f"Document {doc_idx+1}: {score[0]:.4f}")

print(f"Cosine similarity results have been saved to: {SIMILARITY_RESULTS_PATH}")