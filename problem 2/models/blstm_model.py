"""
models/blstm_model.py
----------------------
Bidirectional LSTM (BLSTM) for character-level name generation.

Architecture:
    Embedding → manual BLSTM cell loop → Linear output projection

The LSTM cell is implemented ENTIRELY FROM SCRATCH:
  - All four gate weight matrices (input i, forget f, cell g, output o)
    for BOTH directions are declared as nn.Parameter.
  - No nn.LSTM, nn.LSTMCell, or any PyTorch LSTM wrapper is used.
  - The forward pass manually loops over timesteps in both directions
    and concatenates the resulting hidden states.

LSTM equations (standard formulation):
    i_t = sigmoid( W_xi @ x_t  +  W_hi @ h_{t-1}  +  b_i )  # input gate
    f_t = sigmoid( W_xf @ x_t  +  W_hf @ h_{t-1}  +  b_f )  # forget gate
    g_t = tanh(    W_xg @ x_t  +  W_hg @ h_{t-1}  +  b_g )  # cell gate
    o_t = sigmoid( W_xo @ x_t  +  W_ho @ h_{t-1}  +  b_o )  # output gate
    c_t = f_t * c_{t-1}  +  i_t * g_t                        # cell state
    h_t = o_t * tanh(c_t)                                     # hidden state

Bidirectional:
    Forward  LSTM: processes sequence left  → right
    Backward LSTM: processes sequence right → left
    Final h_t = concat(h_t_fwd, h_t_bwd)  → shape (B, 2*hidden_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Single LSTM cell  (the building block — used twice per layer)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMCellFromScratch(nn.Module):
    """
    A single LSTM cell whose weights are all nn.Parameter.

    Unlike nn.LSTMCell, this class makes all four gate matrices explicit
    so the forward() shows every equation of the LSTM cell.

    Args:
        input_size  : dimensionality of the input vector x_t
        hidden_size : dimensionality of the hidden/cell state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size

        # ── Input-gate weights ─────────────────────────────────────────────
        self.W_xi = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_i  = nn.Parameter(torch.zeros(hidden_size))

        # ── Forget-gate weights ───────────────────────────────────────────
        self.W_xf = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_f  = nn.Parameter(torch.ones(hidden_size))   # init to 1 → remember by default

        # ── Cell-gate weights (often called "g" or "c~") ──────────────────
        self.W_xg = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_g  = nn.Parameter(torch.zeros(hidden_size))

        # ── Output-gate weights ───────────────────────────────────────────
        self.W_xo = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_o  = nn.Parameter(torch.zeros(hidden_size))

        # Kaiming uniform for input weights, orthogonal for recurrent weights
        for W in [self.W_xi, self.W_xf, self.W_xg, self.W_xo]:
            nn.init.kaiming_uniform_(W, nonlinearity="sigmoid")
        for W in [self.W_hi, self.W_hf, self.W_hg, self.W_ho]:
            nn.init.orthogonal_(W)

    def forward(
        self,
        x_t:    torch.Tensor,   # (B, input_size)
        h_prev: torch.Tensor,   # (B, hidden_size)
        c_prev: torch.Tensor,   # (B, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One step of the LSTM cell.

        Computes all four gates explicitly, updates c_t and h_t.

        Returns:
            h_t : new hidden state  (B, hidden_size)
            c_t : new cell state    (B, hidden_size)
        """
        # Input gate  — how much of the new input do we let through?
        i_t = torch.sigmoid(
            F.linear(x_t, self.W_xi) + F.linear(h_prev, self.W_hi) + self.b_i
        )

        # Forget gate  — how much of the old cell state do we keep?
        f_t = torch.sigmoid(
            F.linear(x_t, self.W_xf) + F.linear(h_prev, self.W_hf) + self.b_f
        )

        # Cell gate  — candidate new content for the cell state
        g_t = torch.tanh(
            F.linear(x_t, self.W_xg) + F.linear(h_prev, self.W_hg) + self.b_g
        )

        # Output gate  — how much of the cell state becomes the hidden state?
        o_t = torch.sigmoid(
            F.linear(x_t, self.W_xo) + F.linear(h_prev, self.W_ho) + self.b_o
        )

        # Update cell state: blend old (via forget gate) with new (via input gate)
        c_t = f_t * c_prev  +  i_t * g_t

        # Compute hidden state from the updated cell state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# ─────────────────────────────────────────────────────────────────────────────
# Full BLSTM model
# ─────────────────────────────────────────────────────────────────────────────

class BLSTM(nn.Module):
    """
    Bidirectional LSTM language model built from two LSTMCellFromScratch
    instances per layer — one for each direction.

    The hidden states from both directions are concatenated at every timestep:
        h_t = concat(h_t_forward, h_t_backward)   → (B, 2*hidden_size)

    Because the model is bidirectional, it sees the full sequence before
    predicting, which gives it richer context than the unidirectional RNN.

    Args:
        vocab_size  : total vocabulary size
        embed_dim   : character embedding dimension
        hidden_size : hidden size PER DIRECTION (total = 2 * hidden_size)
        num_layers  : number of stacked BLSTM layers
        dropout     : dropout probability between layers
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 64,
        hidden_size: int   = 256,
        num_layers:  int   = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Character embedding ───────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ── Per-layer forward and backward LSTM cells ─────────────────────
        #
        # Layer 0 input: embed_dim
        # Layer 1+ input: 2 * hidden_size  (concatenated fwd + bwd output)
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = embed_dim if layer_idx == 0 else 2 * hidden_size
            self.fwd_cells.append(LSTMCellFromScratch(in_dim, hidden_size))
            self.bwd_cells.append(LSTMCellFromScratch(in_dim, hidden_size))

        # ── Output projection: 2*hidden_size → vocab_size ─────────────────
        self.output_proj = nn.Linear(2 * hidden_size, vocab_size)

        # ── Dropout between layers ────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    # ──────────────────────────────────────────────────────────────────────
    def _init_states(self, batch_size: int, device: torch.device):
        """
        Initialise zero hidden and cell states for all layers and directions.

        Returns:
            fwd_h, fwd_c : lists of (B, H) tensors for the forward  direction
            bwd_h, bwd_c : lists of (B, H) tensors for the backward direction
        """
        zeros = lambda: torch.zeros(batch_size, self.hidden_size, device=device)
        fwd_h = [zeros() for _ in range(self.num_layers)]
        fwd_c = [zeros() for _ in range(self.num_layers)]
        bwd_h = [zeros() for _ in range(self.num_layers)]
        bwd_c = [zeros() for _ in range(self.num_layers)]
        return fwd_h, fwd_c, bwd_h, bwd_c

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,   # (B, T) — integer token ids
    ) -> tuple[torch.Tensor, dict]:
        """
        Full forward pass through the stacked BLSTM.

        For each layer:
          1. Forward  LSTM: loop t = 0 → T-1
          2. Backward LSTM: loop t = T-1 → 0
          3. Concatenate fwd and bwd hidden states at each timestep
          4. Apply dropout and pass to the next layer

        Returns:
            logits : (B, T, vocab_size)
            states : dict with final hidden/cell states (for inspection)
        """
        batch_size, seq_len = x.shape
        device = x.device

        fwd_h, fwd_c, bwd_h, bwd_c = self._init_states(batch_size, device)

        # Embed input characters → (B, T, embed_dim)
        layer_input = self.embedding(x)

        for layer_idx in range(self.num_layers):
            fwd_cell = self.fwd_cells[layer_idx]
            bwd_cell = self.bwd_cells[layer_idx]

            h_fwd = fwd_h[layer_idx]   # (B, H)
            c_fwd = fwd_c[layer_idx]
            h_bwd = bwd_h[layer_idx]
            c_bwd = bwd_c[layer_idx]

            # ── Forward pass: left → right ─────────────────────────────
            fwd_outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]          # (B, input_dim)
                h_fwd, c_fwd = fwd_cell(x_t, h_fwd, c_fwd)
                fwd_outputs.append(h_fwd)           # each: (B, H)

            # ── Backward pass: right → left ────────────────────────────
            bwd_outputs = [None] * seq_len          # pre-allocate correct order
            for t in reversed(range(seq_len)):
                x_t = layer_input[:, t, :]
                h_bwd, c_bwd = bwd_cell(x_t, h_bwd, c_bwd)
                bwd_outputs[t] = h_bwd              # store at original position

            # ── Concatenate fwd and bwd at every timestep ──────────────
            # fwd_outputs[t] : (B, H)   bwd_outputs[t] : (B, H)
            # combined[t]    : (B, 2*H)
            combined = [
                torch.cat([fwd_outputs[t], bwd_outputs[t]], dim=-1)
                for t in range(seq_len)
            ]
            layer_output = torch.stack(combined, dim=1)  # (B, T, 2*H)

            # Apply dropout between layers
            layer_output = self.dropout(layer_output)

            # Update final states (used for generation at inference)
            fwd_h[layer_idx] = h_fwd
            fwd_c[layer_idx] = c_fwd
            bwd_h[layer_idx] = h_bwd
            bwd_c[layer_idx] = c_bwd

            layer_input = layer_output   # next layer reads this layer's output

        # Project to vocabulary logits
        logits = self.output_proj(layer_input)   # (B, T, vocab_size)

        states = {"fwd_h": fwd_h, "fwd_c": fwd_c, "bwd_h": bwd_h, "bwd_c": bwd_c}
        return logits, states

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)