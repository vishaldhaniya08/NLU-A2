"""
generate.py
-----------
Generates character-level names from the three trained models using
temperature-based sampling (implemented from scratch in utils.py).

For each model:
  - Loads the saved checkpoint from outputs/checkpoints/
  - Generates `NUM_SAMPLES` names by autoregressively sampling one
    character at a time, starting from the <sos> token
  - Saves generated names to outputs/generated/<model_name>_names.txt
  - Prints a sample of generated names to stdout

Temperature variations are also generated and saved for the report's
qualitative analysis (T=0.5, T=1.0, T=1.5).

Usage:
    python generate.py          # must run AFTER train.py

Output files:
    outputs/generated/rnn_vanilla_names.txt
    outputs/generated/blstm_names.txt
    outputs/generated/rnn_attention_names.txt
    outputs/generated/temperature_comparison.txt
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils import CharVocab, load_names, temperature_sample
from models.rnn_vanilla   import VanillaRNN
from models.blstm_model   import BLSTM
from models.rnn_attention import RNNWithAttention


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH       = "TrainingNames.txt"
CHECKPOINT_DIR  = "outputs/checkpoints"
OUTPUT_DIR      = "outputs/generated"
NUM_SAMPLES     = 500    # names generated per model (for evaluation)
MAX_LEN         = 25     # maximum characters before forcing <eos>
TEMPERATURE     = 0.75    # default sampling temperature

TEMPERATURE_VALUES = [0.5, 1.0, 1.5]   # for the qualitative temperature study


# ─────────────────────────────────────────────────────────────────────────────
# Single-name generator (autoregressive loop)
# ─────────────────────────────────────────────────────────────────────────────

def generate_one_name(
    model:       torch.nn.Module,
    vocab:       CharVocab,
    temperature: float,
    device:      torch.device,
    model_type:  str = "rnn",      # "rnn" | "blstm" | "attention"
) -> str:
    """
    Autoregressively generate one name character by character.

    Algorithm:
      1. Start with the <sos> token as the first input
      2. Feed the current input token through the model
      3. Take the output logits for the current timestep
      4. Sample the next character using temperature sampling
      5. Append sampled character to the output; use it as the next input
      6. Stop when <eos> is sampled or max_len is reached

    The hidden state is carried between steps so the model has memory
    of all previously generated characters (pure autoregressive decoding).

    Args:
        model       : trained model (any of the three)
        vocab       : CharVocab with encode/decode methods
        temperature : float controlling creativity vs. realism
        device      : compute device
        model_type  : controls how we unpack the model's output tuple

    Returns:
        Generated name as a string (without special tokens)
    """
    model.eval()   # disable dropout during generation

    with torch.no_grad():
        # ── Initialise: feed <sos> as the first input ────────────────────
        # Shape: (batch=1, seq_len=1)
        current_input = torch.tensor(
            [[vocab.sos_idx]], dtype=torch.long, device=device
        )

        hidden      = None      # will be initialised inside the model
        blstm_states = None    # separate state carrier for BLSTM
        generated   = []        # accumulate character indices

        for step in range(MAX_LEN):
            # ── One forward step ─────────────────────────────────────────
            if model_type == "blstm":
                # Pass previous states so the BLSTM retains memory across steps
                logits, blstm_states = model(current_input)
            elif model_type == "attention":
                logits, hidden, alpha = model(current_input, hidden)
            else:   # vanilla rnn
                logits, hidden = model(current_input, hidden)

            # logits: (1, 1, vocab_size)
            # Take the last timestep's logits (only one timestep here)
            step_logits = logits[0, -1, :]   # (vocab_size,)

            # ── Temperature sampling (from utils.py — no sklearn) ────────
            next_idx = temperature_sample(step_logits, temperature)

            #  Prevent repetition (last 3 chars same → stop)
            if len(generated) >= 3 and len(set(generated[-3:])) == 1:
                break

            #  Avoid very short meaningless names
            if len(generated) < 3 and next_idx == vocab.eos_idx:
                continue

            # Stop if the model generated the end-of-sequence token
            if next_idx == vocab.eos_idx:
                break

            # Stop if the model generated a padding token (shouldn't happen,
            # but guard against it to avoid polluted output)
            if next_idx == vocab.pad_idx:
                break

            generated.append(next_idx)

            # The sampled character becomes the next input
            current_input = torch.tensor(
                [[next_idx]], dtype=torch.long, device=device
            )

        # ── Decode integer ids back to a name string ─────────────────────
        name = vocab.decode(generated)
        return name


# ─────────────────────────────────────────────────────────────────────────────
# Batch generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_names(
    model:       torch.nn.Module,
    vocab:       CharVocab,
    n:           int,
    temperature: float,
    device:      torch.device,
    model_type:  str,
) -> list[str]:
    """
    Generate `n` names from a model.  Filters out empty strings that
    can occasionally appear if the model immediately samples <eos>.

    Returns:
        list of generated name strings, length ≤ n
    """
    names = []
    attempts = 0

    while len(names) < n and attempts < n * 3:
        name = generate_one_name(model, vocab, temperature, device, model_type)
        if name:   # skip empty names
            names.append(name)
        attempts += 1

    return names


# ─────────────────────────────────────────────────────────────────────────────
# Load a checkpoint and reconstruct the model
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    model_name: str,
    vocab:      CharVocab,
    device:     torch.device,
) -> torch.nn.Module:
    """
    Load a saved checkpoint and reconstruct the model with the same
    hyperparameters that were used during training.

    Args:
        model_name : one of "rnn_vanilla", "blstm", "rnn_attention"
        vocab      : CharVocab (to get vocab_size)
        device     : compute device

    Returns:
        Loaded model in eval mode.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    vocab_size  = ckpt["vocab_size"]
    embed_dim   = ckpt["embed_dim"]
    if model_name == "rnn_attention":
        hidden_size = ckpt["hidden_size"]
    else:
        hidden_size = ckpt["hidden_size"]//2
    num_layers  = ckpt["num_layers"]
    dropout     = 0.0   # no dropout at inference

    if model_name == "rnn_vanilla":
        model = VanillaRNN(vocab_size, embed_dim, hidden_size, num_layers, dropout)
    elif model_name == "blstm":
        model = BLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
    elif model_name == "rnn_attention":
        model = RNNWithAttention(vocab_size, embed_dim, hidden_size, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load training names to build the same vocabulary
    training_names = load_names(DATA_PATH)
    vocab = CharVocab(training_names)
    print(f"Vocabulary size: {len(vocab)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Map model names to their type strings (controls output unpacking)
    model_configs = {
        "rnn_vanilla":   "rnn",
        "blstm":         "blstm",
        "rnn_attention": "attention",
    }

    # ── Generate names from each model ────────────────────────────────────
    for model_name, model_type in model_configs.items():
        print(f"\n{'─'*50}")
        print(f"Generating from: {model_name}")

        model = load_checkpoint(model_name, vocab, device)

        names = generate_names(model, vocab, NUM_SAMPLES, TEMPERATURE, device, model_type)

        # Save generated names (one per line)
        out_path = os.path.join(OUTPUT_DIR, f"{model_name}_names.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for name in names:
                f.write(name.capitalize() + "\n")

        print(f"  Generated {len(names)} names → {out_path}")
        print(f"  Sample (first 20): {', '.join(n.capitalize() for n in names[:20])}")

    # ── Temperature study: show effect of temperature on rnn_attention ────
    print(f"\n{'─'*50}")
    print("Temperature comparison study (model: rnn_attention)")

    attention_model = load_checkpoint("rnn_attention", vocab, device)
    temp_lines = []

    for T in TEMPERATURE_VALUES:
        temp_names = generate_names(
            attention_model, vocab, 50, T, device, "attention"
        )
        temp_lines.append(f"\n--- Temperature = {T} ---")
        temp_lines.extend(n.capitalize() for n in temp_names)
        print(f"  T={T}: {', '.join(n.capitalize() for n in temp_names[:10])}")

    temp_path = os.path.join(OUTPUT_DIR, "temperature_comparison.txt")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(temp_lines))
    print(f"\nTemperature comparison saved → {temp_path}")


if __name__ == "__main__":
    main()