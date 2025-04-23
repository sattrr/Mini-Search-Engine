import json
import re
import nltk
from pathlib import Path
from collections import Counter, defaultdict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RAW_DATA_DIR / "scraped_documents.json"
CLEANED_DOCS_PATH = CLEANED_DATA_DIR / "cleaned_documents.json"
INVERTED_INDEX_PATH = CLEANED_DATA_DIR / "inverted_index.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())
custom_stopwords = {
    "hingga", "hampir", "tahu", "mengetahui", "perlu", "luar", "biasa", "satu", "1", "2", "ajar", 
    "bantu", "mudah", "mulai", "kena", "lalu", "melalui", "jadi", "sering", "milik", "hasil", "banyak", "terus", "lebih", "sama"
}
stopwords = stopwords.union(custom_stopwords)

manual_corrections = {
    "covid-2019": "covid-19",
    "ataucoronavirus": "atau coronavirus",
    "disease2019": "disease 2019",
    "menerapkansocial": "menerapkan social",
    "distancingdengan": "distancing dengan",
    "anjuranworld": "anjuran world",
    "danup-to-date": "dan up-to-date",
    "menjadikantwitterrelevan": "menjadikan twitter relevan",
    "wabahcovid-19perlu": "wabah covid-19 perlu",
    "mediasmartphoneyang": "media smartphone yang",
    "bahayacovid-19": "bahaya covid-19",
    "masyaraka": "masyarakat",
    "fataluntukkesehatan": "fatal untuk kesehatan"
}

def fix_manual_combinations(text):
    for wrong, correct in manual_corrections.items():
        text = text.replace(wrong, correct)
    return text

def clean_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text, stopwords, stemmer):
    tokens = word_tokenize(text)
    filtered_tokens = [
        token for token in tokens
        if token.lower() not in stopwords and re.match(r'^[a-zA-Z0-9\-]+$', token)
    ]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

tokenized_docs = []
representative_sentences = []
filtered_docs = []

for doc in articles:
    text = doc["isi"].lower()
    text = fix_manual_combinations(text)
    sentences = sent_tokenize(text)
    representative_sentences.append(sentences[:2] if len(sentences) > 1 else sentences)
    selected_text = " ".join(representative_sentences[-1])

    cleaned_text = clean_text(selected_text)
    stemmed_tokens = tokenize_and_filter(cleaned_text, stopwords, stemmer)
    
    word_freq = Counter(stemmed_tokens)
    threshold = 2 
    top_stopwords = set([word for word, count in word_freq.items() if count > threshold])

    stopwords_combined = stopwords.union(top_stopwords)
    filtered_tokens = [
        token for token in stemmed_tokens
        if token not in stopwords_combined or token in {"covid-19"}
    ]

    tokenized_docs.append({
        "representative_sentences": representative_sentences[-1],
        "tokens": filtered_tokens
    })
    filtered_docs.append(filtered_tokens)

cleaned_articles = [{"representative_sentences": rep, "tokens": tokens} for rep, tokens in zip(representative_sentences, filtered_docs)]
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