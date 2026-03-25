"""
preprocess_pipeline.py
----------------------
Comprehensive data preprocessing pipeline for IIT Jodhpur corpus.

This script:
    • Loads data from:
        - .txt files
        - .pdf files
        - URLs (from urls.txt)
    • Cleans and preprocesses the text
    • Tokenizes and removes noise
    • Computes dataset statistics
    • Saves clean corpus for Word2Vec training

Preprocessing steps:
    1. Remove URLs, emails, unwanted symbols
    2. Sentence splitting
    3. Lowercasing
    4. Tokenization (NLTK)
    5. Stopword removal (with domain whitelist)
    6. Remove short tokens
    7. Save one sentence per line

Author  : [Your Name]
Roll No : [Your Roll No]
Course  : CSL 7640 - Natural Language Understanding
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import nltk
import pdfplumber
import requests
from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ──────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# STOPWORDS + DOMAIN WHITELIST
# ──────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

DOMAIN_WHITELIST = {
    "ug", "pg", "phd", "btech", "mtech", "msc",
    "exam", "research", "student", "faculty",
    "course", "degree", "department", "engineering",
    "science", "lab", "professor", "institute",
    "academic", "program", "admission", "semester",
    "credit", "grade", "cgpa","ug","pg"
}
PDF_GARBAGE_WORDS = {
    "pdf", "obj", "endobj", "stream", "endstream",
    "flatedecode", "procset", "resources",
    "xobject", "image", "imageb", "imagec", "imagei",
    "font", "type", "xref", "trailer","cid","also"
}

STOP_WORDS -= DOMAIN_WHITELIST


# ──────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    Each page is read sequentially.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    except Exception as e:
        print(f"[PDF ERROR] {pdf_path}: {e}")
    return text


# ──────────────────────────────────────────────────────────────────────────────
# WEB SCRAPING
# ──────────────────────────────────────────────────────────────────────────────
def extract_text_from_url(url):
    """
    Fetch and extract visible text from a webpage.
    Removes navigation elements like header/footer.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        return soup.get_text(separator=" ")

    except Exception as e:
        print(f"[URL ERROR] {url}: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# LOAD URL LIST (OPTIONAL BUT RECOMMENDED)
# ──────────────────────────────────────────────────────────────────────────────
def load_urls():
    """
    Load URLs from urls.txt inside data directory.
    Each line should contain one URL.
    """
    url_file = os.path.join(DATA_DIR, "urls.txt")

    if not os.path.exists(url_file):
        return []

    with open(url_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    """
    Clean raw text by removing noise and normalizing.
    """
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)              # remove emails
    # Remove common PDF artifact words
    pdf_noise = [
        "obj", "endobj", "stream", "endstream",
        "flatedecode", "type", "xref", "trailer",
        "startxref", "catalog", "pages", "font",
        "resources", "procset", "xobject",
        "imagec", "imagei", "imageb", "xmpg","cid"
    ]
    

    for word in pdf_noise:
        text = re.sub(rf"\b{word}\b", " ", text)
    
    # Remove 1-2 character tokens EXCEPT domain-critical short terms.
    # ug, pg, ai, ml, cs etc. are meaningful in the IITJ corpus and
    # must survive cleaning so they appear in the Word2Vec vocabulary.
    PROTECTED = {"ug", "pg", "ai", "ml", "cs", "it"}
    placeholders = {}
    for i, word in enumerate(PROTECTED):
        ph = f"PROT{i}PROT"
        placeholders[ph] = word
        text = re.sub(rf"\b{word}\b", ph, text, flags=re.IGNORECASE)

    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)

    # Restore protected tokens
    for ph, word in placeholders.items():
        text = text.replace(ph, word)

    # Keep alphabets, digits, spaces, dots
    text = re.sub(r"[^a-zA-Z0-9.\n\s]", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ──────────────────────────────────────────────────────────────────────────────
def split_sentences(text):
    """
    Split text using '.' and newline as delimiters.
    """
    return re.split(r"[.\n]+", text)


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZATION + FILTERING
# ──────────────────────────────────────────────────────────────────────────────
def process_sentence(sentence):
    """
    Convert sentence → clean tokens
    """
    sentence = sentence.strip().lower()

    if len(sentence) < 5:
        return []

    tokens = word_tokenize(sentence)

    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in PDF_GARBAGE_WORDS]
    tokens = [t for t in tokens if len(t) > 2 or t in DOMAIN_WHITELIST]

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# MAIN DATA LOADER (TXT + PDF + URL)
# ──────────────────────────────────────────────────────────────────────────────
def load_all_data():
    """
    Loads and processes all sources:
        • TXT files
        • PDF files
        • URLs
    """
    all_sentences = []
    num_docs = 0

    files = os.listdir(DATA_DIR)

    print("\n📂 Processing local files...\n")

    for filename in sorted(files):
        path = os.path.join(DATA_DIR, filename)
        raw_text = ""

        if filename.endswith(".txt") and filename != "urls.txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        elif filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(path)

        else:
            continue

        sentences = []
        cleaned = clean_text(raw_text)

        for sent in split_sentences(cleaned):
            tokens = process_sentence(sent)
            if len(tokens) > 2:
                sentences.append(tokens)

        print(f"  {filename:30s} → {len(sentences):4d} sentences")

        all_sentences.extend(sentences)
        num_docs += 1

    # ── Process URLs ───────────────────────────────
    urls = load_urls()

    if urls:
        print("\n🌐 Processing URLs...\n")

    for url in urls:
        raw_text = extract_text_from_url(url)
        cleaned = clean_text(raw_text)

        sentences = []
        for sent in split_sentences(cleaned):
            tokens = process_sentence(sent)
            if len(tokens) > 2:
                sentences.append(tokens)

        print(f"  {url[:50]:50s} → {len(sentences):4d} sentences")

        all_sentences.extend(sentences)
        num_docs += 1

    return all_sentences, num_docs

#remove duplicates and redundant sentences
def deduplicate_sentences(sentences):
    """
    Remove duplicate sentences from corpus.
    """
    unique = []
    seen = set()

    for sent in sentences:
        key = " ".join(sent)
        if key not in seen:
            seen.add(key)
            unique.append(sent)

    return unique


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_stats(sentences, num_docs):
    total_sentences = len(sentences)
    total_tokens = sum(len(s) for s in sentences)

    vocab = set(token for sent in sentences for token in sent)
    freq = Counter(token for sent in sentences for token in sent)

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Documents       : {num_docs}")
    print(f"Sentences       : {total_sentences:,}")
    print(f"Tokens          : {total_tokens:,}")
    print(f"Vocabulary Size : {len(vocab):,}")

    print("\nTop 20 Words:")
    for w, c in freq.most_common(20):
        print(f"{w:15s} {c}")

    return {
        "documents": num_docs,
        "sentences": total_sentences,
        "tokens": total_tokens,
        "vocab_size": len(vocab)
    }


# ──────────────────────────────────────────────────────────────────────────────
# SAVE CORPUS
# ──────────────────────────────────────────────────────────────────────────────
def save_corpus(sentences):
    path = os.path.join(OUTPUT_DIR, "corpus.txt")

    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")

    print(f"\n✅ Saved corpus → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Starting preprocessing pipeline...\n")

    sentences, num_docs = load_all_data()

    print("\n🧹 Removing duplicate sentences...\n")
    before = len(sentences)

    sentences = deduplicate_sentences(sentences)

    after = len(sentences)
    print(f"Removed {before - after} duplicate sentences")
    compute_stats(sentences, num_docs)
    save_corpus(sentences)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()