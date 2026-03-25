"""
utils.py
--------
Shared utilities for Problem 2: Character-Level Name Generation.

Provides:
  - CharVocab  : builds char<->index mappings from a list of names
  - NameDataset: torch Dataset wrapping encoded name sequences
  - collate_fn : pads variable-length sequences in a batch (no external lib)
  - sample_name: temperature-based character sampling from a trained model
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CharVocab
# ─────────────────────────────────────────────────────────────────────────────

class CharVocab:
    """
    Builds a character-level vocabulary from a list of name strings.

    Special tokens:
      <pad>  index 0  – used to pad short sequences in a batch
      <sos>  index 1  – start-of-sequence marker prepended to every name
      <eos>  index 2  – end-of-sequence marker appended to every name

    Usage:
        vocab = CharVocab(names)
        ids   = vocab.encode("Arjun")   # list[int]
        name  = vocab.decode(ids)       # str  (strips <sos>/<eos>/<pad>)
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, names: list[str]):
        # Collect every unique character that appears in the training names
        unique_chars = sorted(set("".join(names)))

        # Reserve indices 0-2 for special tokens, then assign to characters
        self.idx2char = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] + unique_chars
        self.char2idx = {ch: i for i, ch in enumerate(self.idx2char)}

        self.pad_idx = self.char2idx[self.PAD_TOKEN]
        self.sos_idx = self.char2idx[self.SOS_TOKEN]
        self.eos_idx = self.char2idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.idx2char)

    def encode(self, name: str) -> list[int]:
        """
        Convert a name string to a list of integer indices.
        Wraps the name with <sos> at the start and <eos> at the end.
        """
        return (
            [self.sos_idx]
            + [self.char2idx[ch] for ch in name if ch in self.char2idx]
            + [self.eos_idx]
        )

    def decode(self, indices: list[int]) -> str:
        """
        Convert a list of indices back to a name string.
        Strips all special tokens (<pad>, <sos>, <eos>) from the output.
        """
        special = {self.pad_idx, self.sos_idx, self.eos_idx}
        return "".join(
            self.idx2char[i] for i in indices if i not in special
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NameDataset
# ─────────────────────────────────────────────────────────────────────────────

class NameDataset(Dataset):
    """
    PyTorch Dataset for character-level name sequences.

    Each item is a pair (input_ids, target_ids):
      input_ids  : [<sos>, c1, c2, ..., cn]          (model reads these)
      target_ids : [c1,    c2, ..., cn, <eos>]        (model predicts these)

    The input/target shift-by-one structure is the standard language-modelling
    objective: at every timestep t, predict the next character.
    """

    def __init__(self, names: list[str], vocab: CharVocab):
        self.vocab = vocab
        # Pre-encode every name once; store as plain Python lists (not tensors)
        # so that collate_fn can do the padding in one vectorised call.
        self.sequences = [vocab.encode(name) for name in names]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Returns (input_tensor, target_tensor) for one name.
        input  = sequence[:-1]  (everything except the last token)
        target = sequence[1:]   (everything except the first token)
        """
        seq = self.sequences[idx]
        input_seq  = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:],  dtype=torch.long)
        return input_seq, target_seq


# ─────────────────────────────────────────────────────────────────────────────
# 3.  collate_fn  (manual padding — no torchtext or sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list, pad_idx: int = 0):
    """
    Custom collate function passed to DataLoader.

    Pads all sequences in a batch to the length of the longest sequence
    using the <pad> index (0 by default).

    Args:
        batch    : list of (input_tensor, target_tensor) pairs from __getitem__
        pad_idx  : integer index to use for padding (CharVocab.pad_idx)

    Returns:
        inputs  : LongTensor of shape (batch_size, max_len)
        targets : LongTensor of shape (batch_size, max_len)
        lengths : list[int] — actual (unpadded) length of each sequence
    """
    inputs, targets = zip(*batch)

    # Find the longest sequence in this batch
    max_len = max(seq.size(0) for seq in inputs)

    # Manually pad each sequence to max_len
    padded_inputs  = []
    padded_targets = []
    lengths        = []

    for inp, tgt in zip(inputs, targets):
        seq_len = inp.size(0)
        lengths.append(seq_len)

        # Amount of padding needed
        pad_amount = max_len - seq_len

        # torch.nn.functional.pad pads the last dimension from the right
        padded_inputs.append(F.pad(inp, (0, pad_amount), value=pad_idx))
        padded_targets.append(F.pad(tgt, (0, pad_amount), value=pad_idx))

    # Stack into (batch_size, max_len) tensors
    return (
        torch.stack(padded_inputs),   # (B, T)
        torch.stack(padded_targets),  # (B, T)
        lengths,                      # list[int]  — useful for masking the loss
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  load_names  (reads TrainingNames.txt)
# ─────────────────────────────────────────────────────────────────────────────

def load_names(filepath: str) -> list[str]:
    """
    Read names from a plain-text file (one name per line).
    Strips whitespace and skips empty lines.
    Lowercases everything so the vocabulary stays small.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 5.  temperature_sample  (from-scratch softmax sampling)
# ─────────────────────────────────────────────────────────────────────────────

def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Sample the next character index from a raw logits vector.

    Temperature controls creativity vs. accuracy:
      T < 1.0  → sharper distribution  → more predictable / realistic names
      T = 1.0  → unchanged distribution
      T > 1.0  → flatter distribution  → more creative / surprising names

    Implementation:
      1. Divide logits by temperature   (scales the sharpness)
      2. Apply softmax to get probs     (from-scratch: no sklearn)
      3. torch.multinomial for sampling (single draw)

    Args:
        logits      : raw output tensor of shape (vocab_size,)
        temperature : float > 0

    Returns:
        Sampled character index as a Python int.
    """
    # Guard against degenerate temperature values
    temperature = max(temperature, 1e-6)

    # Step 1: scale logits by temperature
    scaled_logits = logits / temperature

    # Step 2: softmax — convert to probability distribution
    probs = F.softmax(scaled_logits, dim=-1)

    # Step 3: multinomial sampling (one sample)
    sampled_idx = torch.multinomial(probs, num_samples=1)

    return sampled_idx.item()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  compute_metrics  (novelty + diversity — no sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(generated: list[str], training: list[str]) -> dict:
    """
    Compute novelty rate and diversity score from scratch.

    Novelty Rate:
        Fraction of generated names that do NOT appear in the training set.
        novelty = |generated_set - training_set| / |generated|

    Diversity Score:
        Fraction of generated names that are unique (no repeats).
        diversity = |unique generated names| / |generated|

    Both metrics are computed using only Python built-ins (sets, len).
    No sklearn or external library is used.

    Args:
        generated : list of generated name strings
        training  : list of training name strings (ground-truth set)

    Returns:
        dict with keys: 'novelty_rate', 'diversity_score',
                        'total_generated', 'unique_generated', 'novel_count'
    """
    if not generated:
        return {"novelty_rate": 0.0, "diversity_score": 0.0,
                "total_generated": 0, "unique_generated": 0, "novel_count": 0}

    training_set  = set(name.lower() for name in training)
    generated_lower = [name.lower() for name in generated]

    # Novel names = generated names not seen in training
    novel_count      = sum(1 for name in generated_lower if name not in training_set)
    unique_generated = len(set(generated_lower))
    total_generated  = len(generated_lower)

    novelty_rate   = novel_count      / total_generated
    diversity_score = unique_generated / total_generated

    return {
        "novelty_rate":      round(novelty_rate,    4),
        "diversity_score":   round(diversity_score, 4),
        "total_generated":   total_generated,
        "unique_generated":  unique_generated,
        "novel_count":       novel_count,
    }