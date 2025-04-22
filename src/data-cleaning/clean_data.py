import json
import re
import nltk
from pathlib import Path
from collections import defaultdict, Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

nltk.download('punkt')

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RAW_DATA_DIR / "scraped_documents.json"
INVERTED_INDEX_PATH = CLEANED_DATA_DIR / "inverted_index.json"
CLEANED_DOCS_PATH = CLEANED_DATA_DIR / "cleaned_documents.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

tokenized_docs = []
all_tokens = []

for doc in articles:
    text = doc["isi"].lower()
    tokens = word_tokenize(re.sub(r"[^a-zA-z0-9]", " ", text))
    stems = [stemmer.stem(token) for token in tokens if token.isalpha()]
    tokenized_docs.append(stems)
    all_tokens.extend(stems)

word_freq = Counter(all_tokens)

top_n = 50
top_stopwords = set([word for word, _ in word_freq.most_common(top_n)])

stopwords_combined = stopwords.union(top_stopwords)


filtered_docs = [
    [token for token in doc if token not in stopwords_combined]
    for doc in tokenized_docs
]

cleaned_articles = []
for i, doc in enumerate(articles):
    cleaned_articles.append({
        "judul": doc["judul"],
        "url": doc["url"],
        "tokens": filtered_docs[i]
    })

with open(CLEANED_DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)

inverted_index = defaultdict(set)

for doc_id, tokens in enumerate(filtered_docs):
    for token in tokens:
        inverted_index[token].add(doc_id)

inverted_index = {token: list(doc_ids) for token, doc_ids in inverted_index.items()}

with open(INVERTED_INDEX_PATH, "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=2)

print(f"Inverted index disimpan di: {INVERTED_INDEX_PATH}")
print(f"Dokumen yang telah dibersihkan disimpan di: {CLEANED_DOCS_PATH}")