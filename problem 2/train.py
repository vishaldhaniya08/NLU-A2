"""
train.py
---------
Trains all three character-level name generation models:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Attention

For each model:
  - Loads TrainingNames.txt and builds a CharVocab from scratch
  - Creates a NameDataset and DataLoader with our custom collate_fn
  - Trains using cross-entropy loss (ignoring <pad> tokens)
  - Saves the trained model checkpoint to outputs/checkpoints/
  - Saves per-epoch training loss to outputs/loss_logs/

Usage:
    python train.py

Outputs:
    outputs/checkpoints/rnn_vanilla.pt
    outputs/checkpoints/blstm.pt
    outputs/checkpoints/rnn_attention.pt
    outputs/loss_logs/rnn_vanilla_loss.txt
    outputs/loss_logs/blstm_loss.txt
    outputs/loss_logs/rnn_attention_loss.txt
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Import our from-scratch components ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils import CharVocab, NameDataset, collate_fn, load_names
from models.rnn_vanilla   import VanillaRNN
from models.blstm_model   import BLSTM
from models.rnn_attention import RNNWithAttention


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — change these to experiment
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Data
    "data_path":  "TrainingNames.txt",

    # Training
    "batch_size":  32,
    "num_epochs":  100,
    "lr":          0.0005,
    "clip_grad":   5.0,        # gradient clipping threshold (prevents explosion)

    # Architecture (shared defaults; feel free to vary per model)
    "embed_dim":   64,
    "hidden_size": 128,
    "num_layers":  2,
    "dropout":     0.5,

    # Output paths
    "checkpoint_dir": "outputs/checkpoints",
    "loss_log_dir":   "outputs/loss_logs",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute masked cross-entropy loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits:  torch.Tensor,   # (B, T, vocab_size)
    targets: torch.Tensor,   # (B, T)
    pad_idx: int,
) -> torch.Tensor:
    """
    Cross-entropy loss that ignores <pad> positions.

    We flatten B*T into one dimension so nn.CrossEntropyLoss can process
    the whole batch at once.  The ignore_index=pad_idx argument tells
    PyTorch to skip pad tokens when computing the average loss.

    Args:
        logits  : raw model output  (B, T, vocab_size)
        targets : ground-truth ids  (B, T)
        pad_idx : index of the <pad> token — these positions are masked

    Returns:
        scalar loss tensor
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape: (B*T, vocab_size) and (B*T,)
    logits_flat  = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits_flat, targets_flat)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:       nn.Module,
    model_name:  str,
    train_loader: DataLoader,
    vocab:       CharVocab,
    config:      dict,
    device:      torch.device,
) -> list[float]:
    """
    Train a single model for config['num_epochs'] epochs.

    Steps per batch:
      1. Zero gradients
      2. Forward pass  (handles both RNNWithAttention's 3-tuple and others' 2-tuple)
      3. Compute masked cross-entropy loss
      4. Backward pass
      5. Gradient clipping  (prevents exploding gradients)
      6. Optimizer step

    Args:
        model        : one of VanillaRNN / BLSTM / RNNWithAttention
        model_name   : string label used for saving files
        train_loader : DataLoader yielding (inputs, targets, lengths)
        vocab        : CharVocab (needed for pad_idx)
        config       : hyperparameter dictionary
        device       : cpu or cuda

    Returns:
        epoch_losses : list of average loss per epoch
    """
    model.to(device)
    model.train()

    # Adam optimiser — allowed (we are not implementing the optimiser from scratch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epoch_losses = []

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs: {config['num_epochs']}  |  Batch size: {config['batch_size']}")
    print(f"  Hidden size: {config['hidden_size']}  |  Embed dim: {config['embed_dim']}")
    print(f"{'='*60}")

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        total_loss  = 0.0
        num_batches = 0

        for inputs, targets, lengths in train_loader:
            # Move tensors to the compute device
            inputs  = inputs.to(device)   # (B, T)
            targets = targets.to(device)  # (B, T)

            optimizer.zero_grad()

            # ── Forward pass ─────────────────────────────────────────
            # RNNWithAttention returns (logits, hidden, alpha)
            # VanillaRNN and BLSTM return (logits, hidden/states)
            output = model(inputs)
            logits = output[0]           # always the first element

            # ── Loss (masked cross-entropy) ───────────────────────────
            loss = compute_loss(logits, targets, vocab.pad_idx)

            # ── Backward pass ─────────────────────────────────────────
            loss.backward()

            # ── Gradient clipping — prevents exploding gradients ──────
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])

            # ── Parameter update ──────────────────────────────────────
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        elapsed = time.time() - epoch_start

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{config['num_epochs']}]  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Time: {elapsed:.1f}s")

    # ── Save checkpoint ───────────────────────────────────────────────────
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["checkpoint_dir"], f"{model_name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name":       model_name,
        "vocab_size":       vocab.__len__(),
        "embed_dim":        config["embed_dim"],
        "hidden_size":      config["hidden_size"],
        "num_layers":       config["num_layers"],
        "dropout":          config["dropout"],
        "epoch_losses":     epoch_losses,
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")

    # ── Save loss log (one float per line) ───────────────────────────────
    os.makedirs(config["loss_log_dir"], exist_ok=True)
    log_path = os.path.join(config["loss_log_dir"], f"{model_name}_loss.txt")
    with open(log_path, "w") as f:
        for ep, loss_val in enumerate(epoch_losses, start=1):
            f.write(f"{ep}\t{loss_val:.6f}\n")
    print(f"  Loss log   saved → {log_path}")

    return epoch_losses


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    names = load_names(CONFIG["data_path"])
    print(f"Loaded {len(names)} names from {CONFIG['data_path']}")

    # ── Build vocabulary from scratch ─────────────────────────────────────
    vocab = CharVocab(names)
    print(f"Vocabulary size: {len(vocab)} characters")
    print(f"Characters: {' '.join(vocab.idx2char[3:])}")  # skip special tokens

    # ── Dataset and DataLoader ─────────────────────────────────────────────
    dataset = NameDataset(names, vocab)

    # We pass a lambda to collate_fn so it has access to vocab.pad_idx
    # without needing to make collate_fn a class.
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=vocab.pad_idx),
    )

    # ── Instantiate models ─────────────────────────────────────────────────
    V = len(vocab)   # shorthand for vocab_size

    models = {
        "rnn_vanilla": VanillaRNN(
            vocab_size  = V,
            embed_dim   = CONFIG["embed_dim"],
            hidden_size = CONFIG["hidden_size"] //2 , #128 to reduce overfitting
            num_layers  = CONFIG["num_layers"],
            dropout     = CONFIG["dropout"],
        ),
        "blstm": BLSTM(
            vocab_size  = V,
            embed_dim   = CONFIG["embed_dim"],
            hidden_size = CONFIG["hidden_size"] // 2,  # //2 so total ≈ same params
            num_layers  = CONFIG["num_layers"],
            dropout     = CONFIG["dropout"],
        ),
        "rnn_attention": RNNWithAttention(
            vocab_size  = V,
            embed_dim   = CONFIG["embed_dim"],
            hidden_size = CONFIG["hidden_size"],
            num_layers  = CONFIG["num_layers"],
            dropout     = CONFIG["dropout"],
        ),
    }

    # ── Print parameter counts ────────────────────────────────────────────
    print("\nModel parameter counts:")
    for name, model in models.items():
        print(f"  {name:20s}: {model.count_parameters():>10,} parameters")

    # ── Train each model ──────────────────────────────────────────────────
    all_losses = {}
    for model_name, model in models.items():
        losses = train_model(model, model_name, loader, vocab, CONFIG, device)
        all_losses[model_name] = losses

    print("\n\nAll models trained successfully.")
    print("Run generate.py to sample names, then evaluate.py for metrics.")


if __name__ == "__main__":
    main()