import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import normalize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

INVERTED_INDEX_PATH = CLEANED_DATA_DIR / "inverted_index.json"
CLEANED_DOCS_PATH = CLEANED_DATA_DIR / "cleaned_documents.json"
TFIDF_RESULTS_PATH = PROCESSED_DATA_DIR / "tfidf_results.json"

with open(CLEANED_DOCS_PATH, "r", encoding="utf-8") as f:
    cleaned_articles = json.load(f)

query_docs = [cleaned_articles[0], cleaned_articles[1]]
corpus_docs = [cleaned_articles[2], cleaned_articles[3], cleaned_articles[4]]

all_tokens = set()
for doc in query_docs + corpus_docs:
    all_tokens.update(doc["tokens"])

def compute_tf(doc, all_tokens):
    tf = {}
    total_terms = len(doc["tokens"])
    for token in all_tokens:
        tf[token] = doc["tokens"].count(token) / total_terms
    return tf

def compute_idf(corpus_docs, all_tokens):
    idf = {}
    num_docs = len(corpus_docs)
    for token in all_tokens:
        doc_count = sum(1 for doc in corpus_docs if token in doc["tokens"])
        idf[token] = math.log(num_docs / (1 + doc_count))
    return idf

idf = compute_idf(corpus_docs, all_tokens)

def compute_tf_idf(doc, tf, idf):
    tf_idf = {}
    for token, term_freq in tf.items():
        tf_idf[token] = term_freq * idf.get(token, 0)
    return tf_idf

corpus_tf_idf = {}
for i, doc in enumerate(corpus_docs):
    tf = compute_tf(doc, all_tokens)
    tf_idf = compute_tf_idf(doc, tf, idf)
    corpus_tf_idf[i] = tf_idf

query_tf_idf = {}
for i, doc in enumerate(query_docs):
    tf = compute_tf(doc, all_tokens)
    tf_idf = compute_tf_idf(doc, tf, idf)
    query_tf_idf[i] = tf_idf

corpus_matrix = np.array([list(corpus_tf_idf[i].values()) for i in corpus_tf_idf])
query_matrix = np.array([list(query_tf_idf[i].values()) for i in query_tf_idf])

corpus_matrix_normalized = normalize(corpus_matrix, norm="l2", axis=1)
query_matrix_normalized = normalize(query_matrix, norm="l2", axis=1)

combined_matrix = np.vstack([corpus_matrix_normalized, query_matrix_normalized])

num_top_words = 20

sorted_tokens_by_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)[:num_top_words]
top_tokens = [token for token, _ in sorted_tokens_by_idf]

top_tokens_set = set(top_tokens)
corpus_matrix_top = np.array([list(corpus_tf_idf[i].get(token, 0) for token in top_tokens) for i in corpus_tf_idf])
query_matrix_top = np.array([list(query_tf_idf[i].get(token, 0) for token in top_tokens) for i in query_tf_idf])

corpus_matrix_normalized_top = normalize(corpus_matrix_top, norm="l2", axis=1)
query_matrix_normalized_top = normalize(query_matrix_top, norm="l2", axis=1)

combined_matrix_top = np.vstack([corpus_matrix_normalized_top, query_matrix_normalized_top])

corpus_labels = ['Corpus Doc ' + str(i+1) for i in range(len(corpus_docs))]
query_labels = ['Query ' + str(i+1) for i in range(len(query_docs))]

plt.figure(figsize=(12, 8))
ax = sns.heatmap(combined_matrix_top, xticklabels=top_tokens, yticklabels=corpus_labels + query_labels, cmap='Blues', cbar=True)

ax.set_xlabel("Terms")
ax.set_ylabel("Documents")
ax.set_title("Top 20 TF-IDF Normalized Weights of Terms in Corpus and Queries")

plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(PROCESSED_DATA_DIR / "top_tfidf_heatmap.png")
plt.show()

with open(TFIDF_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump({"corpus_tf_idf": corpus_tf_idf, "query_tf_idf": query_tf_idf}, f, ensure_ascii=False, indent=4)

print(f"Hasil TF-IDF telah disimpan di: {TFIDF_RESULTS_PATH}")