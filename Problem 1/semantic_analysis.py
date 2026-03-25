"""
semantic_analysis.py
--------------------
Task-3: Semantic Analysis using cosine similarity.

    1. Reports top-5 nearest neighbors for required words:
         research, student, phd, exam
       — for BOTH CBOW and Skip-gram models (side by side)

    2. Performs analogy experiments using vector arithmetic:
         word2 - word1 + word3 ≈ answer
       e.g.  UG : BTech :: PG : ?
             → model.wv.most_similar(positive=[btech, pg], negative=[ug])

    3. Discusses whether results are semantically meaningful.

Author  : [Your Name]
Roll No : [Your Roll No]
Course  : CSL 7640 - Natural Language Understanding
"""

from gensim.models import Word2Vec

# ── Load both models ──────────────────────────────────────────────────────────
print("Loading models...")
cbow_model = Word2Vec.load("outputs/cbow.model")
sg_model   = Word2Vec.load("outputs/skipgram.model")
print("  ✅ cbow.model     loaded")
print("  ✅ skipgram.model loaded")


# ── Task 3.1 : Nearest neighbors ──────────────────────────────────────────────
def nearest_neighbors(word: str, model: Word2Vec, topn: int = 5) -> list:
    """
    Return top-n most similar words for a given query word.
    Uses cosine similarity over word vectors.
    Returns list of (word, score) tuples, or empty list if word not in vocab.
    """
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    else:
        return []


def report_neighbors(words: list):
    """
    Print a side-by-side comparison of nearest neighbors
    from CBOW and Skip-gram for each query word.
    """
    print("\n" + "=" * 70)
    print("  TASK 3.1 — TOP-5 NEAREST NEIGHBORS (Cosine Similarity)")
    print("=" * 70)

    for word in words:
        cbow_nbrs = nearest_neighbors(word, cbow_model)
        sg_nbrs   = nearest_neighbors(word, sg_model)

        print(f"\n  Query word : '{word}'")
        print(f"  {'─'*65}")
        print(f"  {'Rank':<6} {'CBOW':<28} {'Skip-gram':<28}")
        print(f"  {'─'*65}")

        # zip_longest ensures we print even if one model lacks the word
        from itertools import zip_longest
        for rank, (c, s) in enumerate(zip_longest(cbow_nbrs, sg_nbrs), 1):
            c_str = f"{c[0]} ({c[1]:.3f})" if c else "—"
            s_str = f"{s[0]} ({s[1]:.3f})" if s else "—"
            print(f"  {rank:<6} {c_str:<28} {s_str:<28}")

        # Brief semantic interpretation
        if cbow_nbrs:
            print(f"\n  Interpretation (CBOW)  : "
                  f"{', '.join(w for w, _ in cbow_nbrs[:3])} are semantically "
                  f"close to '{word}' in the IITJ corpus.")
        if sg_nbrs:
            print(f"  Interpretation (SG)    : "
                  f"{', '.join(w for w, _ in sg_nbrs[:3])} are semantically "
                  f"close to '{word}' in the Skip-gram embedding space.")


# ── Task 3.2 : Analogy experiments ────────────────────────────────────────────
def analogy(word1: str, word2: str, word3: str,
            model: Word2Vec, model_name: str, topn: int = 3):
    """
    Perform vector analogy:  word1 : word2 :: word3 : ?

    Vector arithmetic:  result ≈ word2 - word1 + word3
    Uses Gensim's most_similar with positive=[word2, word3], negative=[word1].

    Prints results and a semantic interpretation.
    """
    print(f"\n  Analogy  : {word1} : {word2} :: {word3} : ?   [{model_name}]")

    # Check all three words are in vocabulary
    missing = [w for w in [word1, word2, word3] if w not in model.wv]
    if missing:
        print(f"  ⚠️  Words not in vocabulary: {missing}")
        print(f"  Skipping this analogy for {model_name}.")
        return

    results = model.wv.most_similar(
        positive=[word2, word3],
        negative=[word1],
        topn=topn
    )

    print(f"  Answer(s) : ", end="")
    for w, score in results:
        print(f"{w} ({score:.3f})", end="  ")
    print()

    # Semantic interpretation
    top_answer = results[0][0]
    print(f"  → Best answer '{top_answer}' | "
          f"Semantically meaningful: "
          f"{'Yes ✓' if results[0][1] > 0.4 else 'Weak — low similarity score'}")


def report_analogies():
    """
    Run all required and additional analogy experiments
    on both CBOW and Skip-gram, with interpretation.
    """
    print("\n" + "=" * 70)
    print("  TASK 3.2 — ANALOGY EXPERIMENTS")
    print("=" * 70)

    # Define analogy triples: (word1, word2, word3)
    # Interpretation: word1 : word2 :: word3 : ?
    analogies = [
        # Required by assignment
        ("ug",         "btech",       "pg"),         # UG:BTech :: PG:?  → mtech
        # Additional analogies
        ("student",    "course",      "research"),   # student:course :: research:?
        ("department", "engineering", "science"),    # dept:engg :: science:?
        ("btech",      "undergraduate","mtech"),     # btech:ug :: mtech:?
    ]

    for w1, w2, w3 in analogies:
        print(f"\n  {'─'*65}")
        analogy(w1, w2, w3, cbow_model, "CBOW")
        analogy(w1, w2, w3, sg_model,   "Skip-gram")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Running semantic analysis...\n")

    # Task 3.1 — Nearest neighbors for required words
    required_words = ["research", "student", "phd", "exam"]
    report_neighbors(required_words)

    # Task 3.2 — Analogy experiments
    report_analogies()

    print("\n" + "=" * 70)
    print("  NOTES FOR REPORT")
    print("=" * 70)
    print("""
  1. Skip-gram generally finds more specific/rare word neighbors
     because it trains on individual word-context pairs, giving
     rare words more training signal.

  2. CBOW tends to produce smoother, more averaged representations
     because it predicts the center word from the full context window.

  3. Analogy quality depends heavily on corpus size. With a small
     IITJ corpus, some analogies may fail or return weak results —
     this is expected and should be discussed in the report.

  4. A cosine similarity > 0.5 is generally considered meaningful
     for domain-specific small corpora.
  """)

    print("✅ Semantic analysis complete.\n")


if __name__ == "__main__":
    main()