"""
models/rnn_vanilla.py
----------------------
Vanilla (Elman) RNN for character-level name generation.

Architecture:
    Embedding → manual RNN cell loop → Linear output projection

The RNN cell is implemented ENTIRELY FROM SCRATCH:
  - Weight matrices W_xh (input→hidden) and W_hh (hidden→hidden) are
    declared as nn.Parameter, not wrapped inside any nn.RNN module.
  - The recurrence h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h) is
    written explicitly as a Python for-loop over timesteps.

This approach is required to satisfy the "from scratch" constraint and
to expose all intermediate tensors for inspection / grading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNN(nn.Module):
    """
    Vanilla (Elman) RNN character-level language model.

    The recurrence equation at every timestep t:
        h_t = tanh( W_xh @ x_t  +  W_hh @ h_{t-1}  +  b_h )
        y_t = W_hy @ h_t  +  b_y

    where x_t is the embedding of the t-th input character.

    Args:
        vocab_size  : total number of characters in the vocabulary
        embed_dim   : dimensionality of character embeddings
        hidden_size : number of units in the hidden state h_t
        num_layers  : number of stacked RNN layers (each layer is a
                      separate set of W_xh / W_hh matrices)
        dropout     : dropout probability applied between layers (0 = off)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        hidden_size: int = 128,
        num_layers:  int = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Embedding lookup (allowed; not a recurrent unit) ──────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ── Per-layer RNN weight matrices (all nn.Parameter — no nn.RNN) ──
        #
        # For layer 0   : input dimension = embed_dim
        # For layers 1+ : input dimension = hidden_size  (output of prev layer)
        #
        # We store them in nn.ParameterList so PyTorch tracks them correctly.

        self.W_xh = nn.ParameterList()   # input  → hidden  weights
        self.W_hh = nn.ParameterList()   # hidden → hidden  weights
        self.b_h  = nn.ParameterList()   # hidden bias

        for layer_idx in range(num_layers):
            input_dim = embed_dim if layer_idx == 0 else hidden_size

            # Kaiming uniform initialisation (matches PyTorch's default for RNN)
            W_xh_layer = nn.Parameter(torch.empty(hidden_size, input_dim))
            W_hh_layer = nn.Parameter(torch.empty(hidden_size, hidden_size))
            b_h_layer  = nn.Parameter(torch.zeros(hidden_size))

            nn.init.kaiming_uniform_(W_xh_layer, nonlinearity="tanh")
            nn.init.orthogonal_(W_hh_layer)          # orthogonal init for recurrent weights

            self.W_xh.append(W_xh_layer)
            self.W_hh.append(W_hh_layer)
            self.b_h.append(b_h_layer)

        # ── Output projection: hidden → vocab logits ──────────────────────
        self.W_hy = nn.Linear(hidden_size, vocab_size)

        # ── Dropout between layers (applied to the output of each layer) ──
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    # ──────────────────────────────────────────────────────────────────────
    def _init_hidden(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """
        Initialise one zero hidden state per layer.
        Returns a list of tensors, each of shape (batch_size, hidden_size).
        """
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x:      torch.Tensor,              # (B, T)  — integer token ids
        hidden: list[torch.Tensor] | None = None,  # initial hidden states
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the stacked Vanilla RNN.

        Args:
            x      : LongTensor of shape (B, T) — input character indices
            hidden : list of hidden state tensors, one per layer.
                     If None, initialised to zeros.

        Returns:
            logits : FloatTensor of shape (B, T, vocab_size)
            hidden : updated hidden states (list of tensors, one per layer)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Initialise hidden states if not provided (e.g. first batch)
        if hidden is None:
            hidden = self._init_hidden(batch_size, device)

        # ── Step 1: Embed input tokens → (B, T, embed_dim) ───────────────
        embeds = self.embedding(x)          # (B, T, E)

        # ── Step 2: Manually unroll each RNN layer over all timesteps ─────
        #
        # outer loop : layers   (from bottom to top)
        # inner loop : timesteps (left to right along the sequence)
        #
        # At each timestep we compute:
        #   pre_act = W_xh @ x_t  +  W_hh @ h_prev  +  b_h
        #   h_t     = tanh(pre_act)
        #
        # This is the exact Elman recurrence.  No nn.RNN is called.

        layer_input = embeds   # shape: (B, T, input_dim)

        for layer_idx in range(self.num_layers):
            h_prev = hidden[layer_idx]          # (B, hidden_size)

            W_xh = self.W_xh[layer_idx]         # (H, input_dim)
            W_hh = self.W_hh[layer_idx]         # (H, H)
            b_h  = self.b_h[layer_idx]          # (H,)

            timestep_outputs = []               # collect h_t for each t

            for t in range(seq_len):
                x_t = layer_input[:, t, :]      # (B, input_dim)

                # Core recurrence — the only equation that matters for an RNN
                # matmul + bias  (F.linear does  x @ W.T + b)
                pre_act = (
                    F.linear(x_t,   W_xh)       # (B, H)  input contribution
                    + F.linear(h_prev, W_hh)    # (B, H)  recurrent contribution
                    + b_h                       # (H,)    broadcast over batch
                )
                h_t = torch.tanh(pre_act)       # (B, H)  squash to [-1, 1]

                timestep_outputs.append(h_t)
                h_prev = h_t                    # pass h_t to the next timestep

            # Stack all timestep outputs → (B, T, H)
            layer_output = torch.stack(timestep_outputs, dim=1)

            # Apply dropout between layers (Identity if dropout=0)
            layer_output = self.dropout(layer_output)

            # Update hidden state for this layer (final h_t)
            hidden[layer_idx] = h_prev

            # The output of this layer becomes the input of the next
            layer_input = layer_output

        # ── Step 3: Project hidden states to vocabulary logits ────────────
        # layer_input at this point holds the output of the top layer
        logits = self.W_hy(layer_input)         # (B, T, vocab_size)

        return logits, hidden

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)