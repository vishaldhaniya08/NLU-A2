"""
visualize_embeddings.py
-----------------------
Task-4: Visualize word embeddings in 2D using PCA and t-SNE.

Generates 4 plots saved to ../outputs/ and ../plots/:
    1. cbow_pca.png      — CBOW     embeddings reduced with PCA
    2. cbow_tsne.png     — CBOW     embeddings reduced with t-SNE
    3. skipgram_pca.png  — Skip-gram embeddings reduced with PCA
    4. skipgram_tsne.png — Skip-gram embeddings reduced with t-SNE

Top-60 vocabulary words are selected, then grouped into semantic
clusters so the plot colors reveal clustering behavior.

Author  : [Your Name]
Roll No : [Your Roll No]
Course  : CSL 7640 - Natural Language Understanding
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gensim.models      import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
PLOTS_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Load both models ──────────────────────────────────────────────────────────
print("Loading models...")
cbow_model = Word2Vec.load(os.path.join(OUTPUT_DIR, "cbow.model"))
sg_model   = Word2Vec.load(os.path.join(OUTPUT_DIR, "skipgram.model"))
print("  ✅ Both models loaded.")

# ── Semantic cluster definitions ──────────────────────────────────────────────
# Words are grouped by domain category.
# Colors help visually identify if embeddings cluster by semantic group.
# Only words present in the model vocabulary will be plotted.
CLUSTER_GROUPS = {
    "academic_program": {
        "color": "#E74C3C",   # red
        "words": ["btech", "mtech", "phd", "msc", "undergraduate",
                  "postgraduate", "degree", "program", "admission", "ug", "pg"],
    },
    "people": {
        "color": "#3498DB",   # blue
        "words": ["student", "faculty", "professor", "researcher",
                  "scholar", "advisor", "supervisor", "candidate"],
    },
    "academics": {
        "color": "#2ECC71",   # green
        "words": ["course", "semester", "credit", "exam", "grade",
                  "cgpa", "sgpa", "thesis", "curriculum", "syllabus",
                  "lecture", "assignment", "evaluation"],
    },
    "research": {
        "color": "#9B59B6",   # purple
        "words": ["research", "lab", "project", "publication",
                  "paper", "journal", "conference", "experiment",
                  "algorithm", "data", "analysis", "result"],
    },
    "departments": {
        "color": "#F39C12",   # orange
        "words": ["department", "engineering", "science", "computer",
                  "electrical", "mechanical", "chemical", "physics",
                  "mathematics", "chemistry", "bioscience"],
    },
}


# ── Select words to plot ──────────────────────────────────────────────────────
def get_plot_words(model: Word2Vec, top_n: int = 60) -> tuple:
    """
    Build a list of (word, cluster_label) pairs for plotting.

    Strategy:
        1. First include all cluster-group words found in vocabulary.
        2. Fill remaining slots up to top_n with the most frequent
           vocabulary words not already included.

    Returns:
        words  : list of word strings
        colors : list of hex color strings (one per word)
        labels : list of cluster label strings (for legend)
    """
    words_out  = []
    colors_out = []
    labels_out = []
    seen       = set()

    # Priority: cluster words
    for cluster_name, info in CLUSTER_GROUPS.items():
        for w in info["words"]:
            if w in model.wv and w not in seen:
                words_out.append(w)
                colors_out.append(info["color"])
                labels_out.append(cluster_name)
                seen.add(w)

    # Fill up with frequent vocab words (gray)
    for w in model.wv.index_to_key:
        if len(words_out) >= top_n:
            break
        if w not in seen:
            words_out.append(w)
            colors_out.append("#95A5A6")   # gray for uncategorized
            labels_out.append("other")
            seen.add(w)

    return words_out, colors_out, labels_out


# ── Dimensionality reduction ──────────────────────────────────────────────────
def reduce_pca(vectors: np.ndarray) -> np.ndarray:
    """Reduce high-dimensional vectors to 2D using PCA."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


def reduce_tsne(vectors: np.ndarray, n_words: int) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 2D using t-SNE.
    Perplexity must be < number of samples; auto-set to safe value.
    """
    perplexity = min(30, n_words - 1)   # t-SNE constraint

    # n_iter was renamed to max_iter in scikit-learn >= 1.2 — handle both
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    iter_kwarg = "max_iter" if sk_ver >= (1, 2) else "n_iter"

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",              # PCA init is more stable than random
        **{iter_kwarg: 1000},    # compatible with scikit-learn < 1.2 and >= 1.2
    )
    return tsne.fit_transform(vectors)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_embeddings(coords:   np.ndarray,
                    words:    list,
                    colors:   list,
                    labels:   list,
                    title:    str,
                    savepath: str):
    """
    Scatter plot of 2D word embeddings with word annotations.
    Color-coded by semantic cluster for easy interpretation.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    x = coords[:, 0]
    y = coords[:, 1]

    # Scatter points
    ax.scatter(x, y, c=colors, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Word annotations (slightly offset to avoid overlap)
    for i, word in enumerate(words):
        ax.annotate(
            word,
            (x[i], y[i]),
            fontsize=8,
            alpha=0.9,
            xytext=(4, 4),
            textcoords="offset points",
        )

    # Legend for semantic clusters
    legend_items = []
    seen_labels  = set()
    color_map    = {info["color"]: name for name, info in CLUSTER_GROUPS.items()}
    color_map["#95A5A6"] = "other"

    for color, label in zip(colors, labels):
        if label not in seen_labels:
            patch = mpatches.Patch(color=color, label=label.replace("_", " "))
            legend_items.append(patch)
            seen_labels.add(label)

    ax.legend(handles=legend_items, loc="upper left",
              fontsize=9, framealpha=0.7, title="Semantic group")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {savepath}")


# ── Generate all 4 plots ──────────────────────────────────────────────────────
def generate_plots(model: Word2Vec, model_name: str, top_n: int = 60):
    """
    Generate PCA and t-SNE plots for one model.
    Saves to ../outputs/.
    """
    print(f"\n  Generating plots for {model_name}...")

    words, colors, labels = get_plot_words(model, top_n=top_n)
    n = len(words)
    print(f"  Words selected for plotting: {n}")

    # Get embedding vectors
    vectors = np.array([model.wv[w] for w in words])

    # ── PCA ──
    pca_coords = reduce_pca(vectors)
    for save_dir in [OUTPUT_DIR, PLOTS_DIR]:
        plot_embeddings(
            pca_coords, words, colors, labels,
            title    = f"{model_name} — PCA Projection (2D)",
            savepath = os.path.join(save_dir, f"{model_name.lower()}_pca.png"),
        )

    # ── t-SNE ──
    tsne_coords = reduce_tsne(vectors, n)
    for save_dir in [OUTPUT_DIR, PLOTS_DIR]:
        plot_embeddings(
            tsne_coords, words, colors, labels,
            title    = f"{model_name} — t-SNE Projection (2D)",
            savepath = os.path.join(save_dir, f"{model_name.lower()}_tsne.png"),
        )


# ── Interpretation notes ──────────────────────────────────────────────────────
def print_interpretation():
    print("""
  INTERPRETATION NOTES (for report):
  ─────────────────────────────────────────────────────────────
  PCA:
    - Linear projection; preserves global structure (variance).
    - Clusters are less tight but global layout is more faithful.
    - Good for seeing broad separations between semantic groups.

  t-SNE:
    - Non-linear; preserves local neighborhood structure.
    - Clusters appear tighter and more visually distinct.
    - Does NOT preserve global distances — cluster positions
      relative to each other are NOT meaningful.

  CBOW vs Skip-gram differences:
    - Skip-gram embeddings tend to show tighter, more distinct
      clusters because each word gets more direct training signal
      from its individual contexts.
    - CBOW embeddings are more averaged/smoothed, so clusters
      may appear more spread out or overlapping.
    - Both should show that 'research', 'phd', 'thesis' cluster
      together, and 'btech', 'mtech', 'degree' cluster together
      if the corpus has sufficient coverage.
  ─────────────────────────────────────────────────────────────
  """)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Generating embedding visualizations...\n")

    generate_plots(cbow_model, model_name="CBOW",      top_n=60)
    generate_plots(sg_model,   model_name="Skipgram",  top_n=60)

    print_interpretation()
    print("✅ All 4 visualizations saved.\n")


if __name__ == "__main__":
    main()