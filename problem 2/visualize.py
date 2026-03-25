"""
visualize.py
------------
Generate all visualisations for Problem 2's report.

Produces:
  1. Loss curves     — training loss over epochs for all 3 models (one plot)
  2. Metrics bar chart — novelty rate and diversity score side by side
  3. Name length distribution — histogram of generated vs training name lengths
  4. Attention weights heatmap — for one sample name from rnn_attention

Usage:
    python visualize.py          # run AFTER train.py and generate.py

Output files (all in outputs/plots/):
    training_loss_curves.png
    metrics_comparison.png
    length_distribution.png
    attention_heatmap.png        (requires rnn_attention checkpoint)
"""

import os
import sys
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils import CharVocab, load_names, temperature_sample


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "TrainingNames.txt"
LOSS_LOG_DIR   = "outputs/loss_logs"
GENERATED_DIR  = "outputs/generated"
CHECKPOINT_DIR = "outputs/checkpoints"
PLOTS_DIR      = "outputs/plots"

MODEL_NAMES  = ["rnn_vanilla", "blstm", "rnn_attention"]
MODEL_LABELS = {
    "rnn_vanilla":   "Vanilla RNN",
    "blstm":         "Bidirectional LSTM",
    "rnn_attention": "RNN + Attention",
}
MODEL_COLORS = {
    "rnn_vanilla":   "#4C72B0",   # muted blue
    "blstm":         "#DD8452",   # muted orange
    "rnn_attention": "#55A868",   # muted green
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves():
    """
    Plot training loss (per epoch) for all three models on the same axes.

    Reads the tab-separated loss log files written by train.py.
    Each file has lines:  epoch_number <TAB> loss_value
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    any_plotted = False

    for model_name in MODEL_NAMES:
        log_path = os.path.join(LOSS_LOG_DIR, f"{model_name}_loss.txt")
        if not os.path.exists(log_path):
            print(f"  WARNING: {log_path} not found, skipping.")
            continue

        epochs = []
        losses = []
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))

        ax.plot(
            epochs, losses,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            linewidth=2.0,
        )
        any_plotted = True

    if not any_plotted:
        print("  No loss logs found. Run train.py first.")
        plt.close()
        return

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss Curves — Character-Level Name Generation", fontsize=13, pad=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(PLOTS_DIR, "training_loss_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics bar chart  (novelty + diversity)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar():
    """
    Side-by-side bar chart comparing novelty rate and diversity score
    across the three models.

    Reads the CSV produced by evaluate.py.
    """
    csv_path = "outputs/metrics/evaluation_results.csv"
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found. Run evaluate.py first.")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader  = list(csv.DictReader(f))

    labels    = [r["Model"] for r in reader]
    novelty   = [float(r["Novelty Rate"].strip("%"))   / 100 for r in reader]
    diversity = [float(r["Diversity Score"].strip("%")) / 100 for r in reader]

    x      = list(range(len(labels)))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar([xi - width/2 for xi in x], novelty,   width,
                   label="Novelty Rate",    color="#4C72B0", alpha=0.85)
    bars2 = ax.bar([xi + width/2 for xi in x], diversity, width,
                   label="Diversity Score", color="#DD8452", alpha=0.85)

    # Add value labels on top of each bar
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.1%}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(l, l) for l in labels], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Novelty Rate & Diversity Score by Model", fontsize=13, pad=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Name length distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_length_distribution():
    """
    Histogram comparing the character-length distribution of names
    generated by each model vs. the training names.

    All length computation is pure Python — no sklearn.
    """
    training_names = load_names(DATA_PATH)
    training_lengths = [len(n) for n in training_names]

    fig, axes = plt.subplots(1, len(MODEL_NAMES) + 1, figsize=(14, 4), sharey=True)

    bins = list(range(1, 20))

    # Training set distribution (leftmost panel)
    axes[0].hist(training_lengths, bins=bins, color="#888888", alpha=0.8, edgecolor="white")
    axes[0].set_title("Training set", fontsize=10)
    axes[0].set_xlabel("Name length")
    axes[0].set_ylabel("Count")

    for ax, model_name in zip(axes[1:], MODEL_NAMES):
        gen_path = os.path.join(GENERATED_DIR, f"{model_name}_names.txt")
        if not os.path.exists(gen_path):
            ax.set_title(MODEL_LABELS[model_name] + "\n(no data)", fontsize=9)
            continue

        gen_names   = load_names(gen_path)
        gen_lengths = [len(n) for n in gen_names]

        ax.hist(gen_lengths, bins=bins,
                color=MODEL_COLORS[model_name], alpha=0.8, edgecolor="white")
        ax.set_title(MODEL_LABELS[model_name], fontsize=9)
        ax.set_xlabel("Name length")

    fig.suptitle("Name Length Distribution: Training vs Generated", fontsize=12, y=1.02)

    out = os.path.join(PLOTS_DIR, "length_distribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention weights heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap():
    """
    Visualise attention weights for a few sample names generated by
    the RNN-with-Attention model.

    For each name:
      - Feed the encoded name through the model
      - Retrieve the attention weight vector α (shape: T)
      - Display as a horizontal heatmap with characters as x-axis labels

    This plot is useful for the report's qualitative analysis section —
    it shows which characters the model "attends to" most when predicting.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, "rnn_attention.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: {ckpt_path} not found. Skipping attention heatmap.")
        return

    from models.rnn_attention import RNNWithAttention

    device = torch.device("cpu")

    # Load checkpoint and rebuild model
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RNNWithAttention(
        vocab_size  = ckpt["vocab_size"],
        embed_dim   = ckpt["embed_dim"],
        hidden_size = ckpt["hidden_size"],
        num_layers  = ckpt["num_layers"],
        dropout     = 0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load vocab
    training_names = load_names(DATA_PATH)
    vocab = CharVocab(training_names)

    # Pick a few sample names to visualise
    gen_path = os.path.join(GENERATED_DIR, "rnn_attention_names.txt")
    if not os.path.exists(gen_path):
        print(f"  WARNING: {gen_path} not found. Skipping attention heatmap.")
        return

    gen_names  = load_names(gen_path)
    # Pick 5 names of reasonable length for visualisation
    sample_names = [n for n in gen_names if 4 <= len(n) <= 12][:5]

    if not sample_names:
        print("  WARNING: No suitable names found for attention heatmap.")
        return

    fig, axes = plt.subplots(
        len(sample_names), 1,
        figsize=(10, 2.5 * len(sample_names)),
    )
    if len(sample_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, sample_names):
        # Encode the name and pass through the model
        encoded = vocab.encode(name)
        x = torch.tensor([encoded[:-1]], dtype=torch.long)   # (1, T)

        with torch.no_grad():
            logits, _, alpha = model(x)   # alpha: (1, T)

        alpha_np = alpha[0].cpu().numpy()   # (T,)

        # Character labels for the x-axis (skip <sos> in display)
        char_labels = [vocab.idx2char[idx] for idx in encoded[:-1]]

        im = ax.imshow(
            alpha_np.reshape(1, -1),
            aspect="auto",
            cmap="YlOrRd",
            vmin=0, vmax=alpha_np.max(),
        )
        ax.set_xticks(range(len(char_labels)))
        ax.set_xticklabels(char_labels, fontsize=11)
        ax.set_yticks([])
        ax.set_title(f'"{name.capitalize()}"', fontsize=11, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle("Attention Weights — RNN + Attention Model", fontsize=13)
    out = os.path.join(PLOTS_DIR, "attention_heatmap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Generating visualisations...\n")

    print("1. Training loss curves")
    plot_loss_curves()

    print("2. Metrics bar chart")
    plot_metrics_bar()

    print("3. Name length distribution")
    plot_length_distribution()

    print("4. Attention weights heatmap")
    try:
        plot_attention_heatmap()
    except:
        print("Skipping attention heatmap (no attention weights available)")

    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()