"""
train_word2vec.py
-----------------
Trains CBOW and Skip-gram Word2Vec models on the cleaned IITJ corpus.

Task-2 requirements:
    - Train both CBOW (sg=0) and Skip-gram with Negative Sampling (sg=1)
    - Experiment with: embedding dimension, context window, negative samples
    - Report results formally (printed table + saved to outputs/)

Three configurations are tested per model type:
    Config A  — small  : vector_size=50,  window=3, negative=5
    Config B  — medium : vector_size=100, window=5, negative=10  (default)
    Config C  — large  : vector_size=200, window=7, negative=15

The best configuration (Config B) is saved as the final model for
downstream tasks (semantic analysis, visualization).

Author  : [Your Name]
Roll No : [Your Roll No]
Course  : CSL 7640 - Natural Language Understanding
"""

import os
import time
from gensim.models import Word2Vec

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_PATH = "outputs/corpus.txt"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameter configurations to experiment with ─────────────────────────
# Each dict is one experimental run. 'name' is used for logging and filenames.
CONFIGS = [
    {
        "name":        "config_A_small",
        "vector_size": 50,
        "window":      3,
        "negative":    5,
        "min_count":   2,
        "epochs":      100,
        "workers":     4,
    },
    {
        "name":        "config_B_medium",
        "vector_size": 100,
        "window":      5,
        "negative":    10,
        "min_count":   2,
        "epochs":      100,
        "workers":     4,
    },
    {
        "name":        "config_C_large",
        "vector_size": 200,
        "window":      7,
        "negative":    15,
        "min_count":   2,
        "epochs":      100,
        "workers":     4,
    },
]

# Probe words used to compare configs qualitatively
PROBE_WORDS = ["student", "research", "engineering"]


# ── Load corpus ───────────────────────────────────────────────────────────────
def load_corpus(path: str) -> list:
    """
    Load the preprocessed corpus from corpus.txt.
    Each line → list of tokens.
    Returns a list of token lists (sentences).
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:                     # skip blank lines
                sentences.append(tokens)

    print(f"  Loaded {len(sentences):,} sentences from corpus.")
    return sentences


# ── Train one model ───────────────────────────────────────────────────────────
def train_one(sentences: list, cfg: dict, sg: int) -> Word2Vec:
    """
    Train a single Word2Vec model.
    sg=0 → CBOW
    sg=1 → Skip-gram with Negative Sampling
    """
    model = Word2Vec(
        sentences    = sentences,
        vector_size  = cfg["vector_size"],   # dimensionality of embeddings
        window       = cfg["window"],         # context window on each side
        negative     = cfg["negative"],       # negative samples per positive pair
        min_count    = cfg["min_count"],      # ignore words below this frequency
        sg           = sg,                    # 0=CBOW, 1=Skip-gram
        epochs       = cfg["epochs"],         # training passes over corpus
        workers      = cfg["workers"],        # parallel threads
        seed         = 42,                    # reproducibility
    )
    return model


# ── Quantitative quality metric ───────────────────────────────────────────────
# Semantic pairs that SHOULD be similar in the IITJ domain.
# We compute average cosine similarity across these pairs as a proxy
# for embedding quality — higher = better semantic capture.
EVAL_PAIRS = [
    ("student",  "research"),
    ("phd",      "thesis"),
    ("course",   "semester"),
    ("faculty",  "professor"),
    ("exam",     "grade"),
]

def evaluate_model(model: Word2Vec) -> float:
    """
    Compute average cosine similarity over known semantic pairs.
    Both words of a pair must be in vocabulary to contribute.
    Returns a float score in [-1, 1]; higher is better.
    This gives an objective quality metric to compare configurations.
    """
    scores = []
    for w1, w2 in EVAL_PAIRS:
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            scores.append(sim)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── Quick qualitative check ───────────────────────────────────────────────────
def probe_model(model: Word2Vec, label: str):
    """
    Print top-3 nearest neighbors for each probe word.
    Used to qualitatively compare configurations.
    """
    print(f"\n  [{label}] Nearest neighbors:")
    for word in PROBE_WORDS:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=3)
            nbr_str   = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
            print(f"    {word:15s} → {nbr_str}")
        else:
            print(f"    {word:15s} → (not in vocabulary)")


# ── Run all experiments ───────────────────────────────────────────────────────
def run_experiments(sentences: list) -> dict:
    """
    Train CBOW and Skip-gram for each configuration.
    Print a comparison table.
    Returns the best models (Config B) as a dict.
    """
    results = []   # store (label, vocab_size, train_time) for the table

    best_models = {}   # will hold final cbow and skipgram models

    for cfg in CONFIGS:
        for sg, model_type in [(0, "CBOW"), (1, "Skipgram")]:

            label = f"{model_type}  {cfg['name']}"
            print(f"\n{'─'*55}")
            print(f"  Training : {label}")
            print(f"  Params   : vector_size={cfg['vector_size']}, "
                  f"window={cfg['window']}, negative={cfg['negative']}")

            start = time.time()
            model = train_one(sentences, cfg, sg)
            elapsed = time.time() - start

            vocab_size = len(model.wv)
            qual_score = evaluate_model(model)   # avg cosine sim over eval pairs
            print(f"  Vocab    : {vocab_size:,} words")
            print(f"  Quality  : {qual_score:.4f}  (avg cosine sim over semantic pairs)")
            print(f"  Time     : {elapsed:.1f}s")

            probe_model(model, label)

            results.append({
                "model_type":  model_type,
                "config":      cfg["name"],
                "vector_size": cfg["vector_size"],
                "window":      cfg["window"],
                "negative":    cfg["negative"],
                "vocab_size":  vocab_size,
                "quality":     qual_score,
                "train_time":  round(elapsed, 2),
            })

            # Save the medium config (Config B) as the final models
            if cfg["name"] == "config_B_medium":
                if model_type == "CBOW":
                    model.save(os.path.join(OUTPUT_DIR, "cbow.model"))
                    best_models["cbow"] = model
                    print(f"  ✅ Saved as cbow.model")
                else:
                    model.save(os.path.join(OUTPUT_DIR, "skipgram.model"))
                    best_models["skipgram"] = model
                    print(f"  ✅ Saved as skipgram.model")

    return best_models, results


# ── Print summary table ───────────────────────────────────────────────────────
def print_summary_table(results: list):
    """
    Print a formatted table of all experimental results.
    Copy this into your report.
    """
    print("\n" + "=" * 82)
    print("  HYPERPARAMETER EXPERIMENT RESULTS")
    print("=" * 82)
    header = (f"{'Model':<12} {'Config':<22} {'Dim':>5} "
              f"{'Window':>7} {'Neg':>5} {'Vocab':>7} {'Quality':>8} {'Time(s)':>8}")
    print(header)
    print("─" * 82)
    for r in results:
        row = (f"{r['model_type']:<12} {r['config']:<22} {r['vector_size']:>5} "
               f"{r['window']:>7} {r['negative']:>5} {r['vocab_size']:>7} "
               f"{r['quality']:>8.4f} {r['train_time']:>8.2f}")
        print(row)
    print("=" * 82)
    print("  Quality = avg cosine similarity over semantic pairs (higher is better)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Starting Word2Vec training experiments...\n")

    sentences = load_corpus(CORPUS_PATH)

    best_models, results = run_experiments(sentences)

    print_summary_table(results)

    print("\n✅ Training complete.")
    print(f"   Final models saved to {OUTPUT_DIR}/cbow.model and skipgram.model\n")

    return best_models


if __name__ == "__main__":
    main()