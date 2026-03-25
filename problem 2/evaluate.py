"""
evaluate.py
-----------
Quantitative evaluation of the three trained name generation models.

Metrics (both computed from scratch — no sklearn):
  - Novelty Rate   : fraction of generated names NOT in the training set
  - Diversity Score: fraction of generated names that are unique

Reads:
    data-2/TrainingNames.txt            (ground truth)
    outputs/generated/*_names.txt       (model outputs)

Prints a formatted comparison table and saves a CSV summary.

Usage:
    python evaluate.py          # must run AFTER generate.py
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_names, compute_metrics   # our from-scratch metrics


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "TrainingNames.txt"
GENERATED_DIR  = "outputs/generated"
METRICS_DIR    = "outputs/metrics"

MODEL_NAMES = ["rnn_vanilla", "blstm", "rnn_attention"]


# ─────────────────────────────────────────────────────────────────────────────
# Table printer (from scratch — no pandas)
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], col_order: list[str]):
    """
    Print a clean ASCII table from a list of row-dicts.

    Args:
        rows      : list of dicts, each dict is one row
        col_order : column names in display order
    """
    # Compute column widths (max of header and all values)
    col_widths = {col: len(col) for col in col_order}
    for row in rows:
        for col in col_order:
            col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))

    # Build separator and header lines
    sep    = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    header = "|" + "|".join(
        f" {col:<{col_widths[col]}} " for col in col_order
    ) + "|"

    print(sep)
    print(header)
    print(sep)
    for row in rows:
        line = "|" + "|".join(
            f" {str(row.get(col,'')):<{col_widths[col]}} " for col in col_order
        ) + "|"
        print(line)
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Length statistics (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def length_stats(names: list[str]) -> dict:
    """
    Compute min, max, mean length of a list of names.
    Implemented from scratch using Python built-ins.
    """
    if not names:
        return {"min": 0, "max": 0, "mean": 0.0}
    lengths = [len(n) for n in names]
    return {
        "min":  min(lengths),
        "max":  max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Character n-gram overlap (measures how "similar" generated names are
# to training names in terms of character patterns)
# ─────────────────────────────────────────────────────────────────────────────

def char_ngram_overlap(
    generated: list[str],
    training:  list[str],
    n:         int = 2,
) -> float:
    """
    Compute the fraction of character n-grams in the generated names
    that also appear in the training set.

    A high overlap means the model has learned training-like character
    patterns; a low overlap may indicate random or broken generation.

    Implemented entirely with Python sets — no sklearn.
    """
    def get_ngrams(name: str, n: int) -> set:
        """Extract all character n-grams from a single name."""
        return {name[i:i+n] for i in range(len(name) - n + 1)}

    training_ngrams  = set()
    for name in training:
        training_ngrams |= get_ngrams(name.lower(), n)

    if not training_ngrams:
        return 0.0

    generated_ngrams = set()
    for name in generated:
        generated_ngrams |= get_ngrams(name.lower(), n)

    if not generated_ngrams:
        return 0.0

    overlap = len(generated_ngrams & training_ngrams) / len(generated_ngrams)
    return round(overlap, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load training names ───────────────────────────────────────────────
    training_names = load_names(DATA_PATH)
    print(f"Training set: {len(training_names)} names\n")

    os.makedirs(METRICS_DIR, exist_ok=True)

    results = []   # list of dicts, one per model

    for model_name in MODEL_NAMES:
        gen_path = os.path.join(GENERATED_DIR, f"{model_name}_names.txt")

        if not os.path.exists(gen_path):
            print(f"WARNING: {gen_path} not found. Run generate.py first.")
            continue

        # Load generated names
        generated_names = load_names(gen_path)

        # ── Core metrics (from utils.py — no sklearn) ─────────────────
        metrics = compute_metrics(generated_names, training_names)

        # ── Additional statistics ──────────────────────────────────────
        len_stats   = length_stats(generated_names)
        bigram_ovlp = char_ngram_overlap(generated_names, training_names, n=2)
        trigram_ovlp = char_ngram_overlap(generated_names, training_names, n=3)

        row = {
            "Model":           model_name,
            "Total Gen.":      metrics["total_generated"],
            "Unique Gen.":     metrics["unique_generated"],
            "Novel Count":     metrics["novel_count"],
            "Novelty Rate":    f"{metrics['novelty_rate']:.2%}",
            "Diversity Score": f"{metrics['diversity_score']:.2%}",
            "Bigram Overlap":  f"{bigram_ovlp:.2%}",
            "Trigram Overlap": f"{trigram_ovlp:.2%}",
            "Avg Length":      len_stats["mean"],
        }
        results.append(row)

    if not results:
        print("No results to display. Make sure generate.py has been run.")
        return

    # ── Print formatted table ─────────────────────────────────────────────
    col_order = [
        "Model", "Total Gen.", "Unique Gen.", "Novel Count",
        "Novelty Rate", "Diversity Score", "Bigram Overlap",
        "Trigram Overlap", "Avg Length",
    ]
    print("=" * 80)
    print("  QUANTITATIVE EVALUATION RESULTS")
    print("=" * 80)
    print_table(results, col_order)

    print("""
Metric Definitions:
  Novelty Rate    : % of generated names not seen in the training set
                    Formula: |generated - training_set| / |generated|
  Diversity Score : % of generated names that are unique (no duplicates)
                    Formula: |unique generated| / |total generated|
  Bigram Overlap  : % of character bigrams in generated names that also
                    appear in training names (measures pattern similarity)
  Trigram Overlap : Same as above for trigrams (3-char sequences)
""")

    # ── Save CSV summary ──────────────────────────────────────────────────
    csv_path = os.path.join(METRICS_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_order)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved → {csv_path}")

    # ── Model comparison: find best on each metric ─────────────────────
    print("\nBest model per metric:")
    for metric_key, metric_label in [
        ("Novelty Rate", "Novelty Rate"),
        ("Diversity Score", "Diversity Score"),
    ]:
        # Parse percentage strings back to floats for comparison
        best = max(results, key=lambda r: float(r[metric_key].strip("%")) / 100)
        print(f"  {metric_label:20s}: {best['Model']}  ({best[metric_key]})")


if __name__ == "__main__":
    main()