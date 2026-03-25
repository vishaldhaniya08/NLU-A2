"""
wordcloud_gen.py
----------------
Generates a word cloud from the cleaned IITJ corpus.

The word cloud visualizes the most frequent words after preprocessing,
giving a quick visual summary of the dominant topics in the corpus.
Saved to both ../outputs/ and ../plots/.

Author  : [Your Name]
Roll No : [Your Roll No]
Course  : CSL 7640 - Natural Language Understanding
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_PATH = "outputs/corpus.txt"
OUTPUT_DIR  = "outputs"
PLOTS_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def load_corpus_text(path: str) -> str:
    """
    Load the clean corpus and return as a single string.
    Word cloud library expects a plain text input.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_word_frequencies(text: str) -> dict:
    """
    Compute word frequencies from the corpus text.
    Returns a dict {word: count} for the top words.
    """
    words = text.split()
    freq  = Counter(words)
    print(f"  Total unique tokens in corpus : {len(freq):,}")
    print(f"  Top-10 words:")
    for w, c in freq.most_common(10):
        print(f"    {w:20s} {c:5d}")
    return dict(freq)


def generate_wordcloud(freq: dict) -> WordCloud:
    """
    Generate a WordCloud object from word frequency dict.
    Uses a color scheme fitting for an academic/institute context.
    """
    wc = WordCloud(
        width            = 1200,
        height           = 600,
        background_color = "white",
        colormap         = "viridis",      # green-blue palette suits academic theme
        max_words        = 150,
        min_font_size    = 10,
        max_font_size    = 120,
        collocations     = False,          # avoid duplicate bigrams
        random_state     = 42,
    ).generate_from_frequencies(freq)

    return wc


def save_wordcloud(wc: WordCloud):
    """Plot and save the word cloud to both output directories."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Word Cloud — IIT Jodhpur Corpus (most frequent terms)",
        fontsize=14, fontweight="bold", pad=12
    )
    plt.tight_layout()

    for save_dir in [OUTPUT_DIR, PLOTS_DIR]:
        path = os.path.join(save_dir, "wordcloud.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved → {path}")

    plt.close()


def main():
    print("\n🚀 Generating word cloud...\n")

    text = load_corpus_text(CORPUS_PATH)
    freq = get_word_frequencies(text)
    wc   = generate_wordcloud(freq)
    save_wordcloud(wc)

    print("\n✅ Word cloud generation complete.\n")


if __name__ == "__main__":
    main()