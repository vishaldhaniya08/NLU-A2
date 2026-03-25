"""
models/rnn_attention.py
------------------------
Vanilla RNN augmented with a Basic (Bahdanau-style) Attention Mechanism
for character-level name generation.

Architecture:
    Embedding → RNN hidden states → Attention → context vector → output

Attention mechanism (from scratch — no nn.MultiheadAttention):
    1. Collect all hidden states H = [h_1, h_2, ..., h_T]  shape (B, T, H)
    2. Compute alignment scores for each timestep t using two learnable
       weight matrices W_a and W_b:
           score_t = W_a @ tanh( W_b @ h_t )     shape (B, T)
    3. Normalise scores with softmax → attention weights α_t
    4. Compute context vector as weighted sum of all hidden states:
           context = Σ_t  α_t * h_t               shape (B, H)
    5. At each timestep, concatenate h_t and context → project to vocab

All weight matrices (W_a, W_b, plus the RNN weights) are nn.Parameter.
No nn.RNN or nn.MultiheadAttention is used anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNWithAttention(nn.Module):
    """
    RNN + Additive (Bahdanau-style) Attention language model.

    The attention mechanism lets every output timestep "look at" all
    other hidden states before making its prediction — giving the model
    a global view of the sequence, unlike the plain RNN which only
    has access to the current hidden state.

    Args:
        vocab_size  : total vocabulary size
        embed_dim   : character embedding dimension
        hidden_size : RNN hidden state size
        num_layers  : number of stacked RNN layers
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

        # ── RNN cell weights per layer (same from-scratch approach as
        #    VanillaRNN — all nn.Parameter, no nn.RNN) ─────────────────────
        self.W_xh = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_h  = nn.ParameterList()

        for layer_idx in range(num_layers):
            in_dim = embed_dim if layer_idx == 0 else hidden_size
            W_xh_l = nn.Parameter(torch.empty(hidden_size, in_dim))
            W_hh_l = nn.Parameter(torch.empty(hidden_size, hidden_size))
            b_h_l  = nn.Parameter(torch.zeros(hidden_size))
            nn.init.kaiming_uniform_(W_xh_l, nonlinearity="tanh")
            nn.init.orthogonal_(W_hh_l)
            self.W_xh.append(W_xh_l)
            self.W_hh.append(W_hh_l)
            self.b_h.append(b_h_l)

        # ── Attention weight matrices (all nn.Parameter) ──────────────────
        #
        # W_b : projects each hidden state h_t into "key" space
        #       shape (attention_size, hidden_size)
        # W_a : collapses the tanh-activated key into a scalar score
        #       shape (1, attention_size)
        #
        # We use attention_size = hidden_size as a common choice.
        attention_size = hidden_size

        self.W_b = nn.Parameter(torch.empty(attention_size, hidden_size))
        self.W_a = nn.Parameter(torch.empty(1, attention_size))
        nn.init.xavier_uniform_(self.W_b.unsqueeze(0)).squeeze(0)
        nn.init.xavier_uniform_(self.W_a.unsqueeze(0)).squeeze(0)

        # ── Output projection ─────────────────────────────────────────────
        # Input to projection = h_t (hidden_size) + context (hidden_size)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)

        # ── Dropout between layers ────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    # ──────────────────────────────────────────────────────────────────────
    def _init_hidden(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Initialise per-layer hidden states to zero."""
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    # ──────────────────────────────────────────────────────────────────────
    def _compute_attention(
        self, H: torch.Tensor   # (B, T, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global additive attention over all hidden states H.

        Steps (all matrix operations, no nn.MultiheadAttention):
          1. Key projection: keys = tanh( H @ W_b.T )     shape (B, T, attn_size)
          2. Score:          score = keys @ W_a.T          shape (B, T, 1)
          3. Squeeze + softmax → attention weights α       shape (B, T)
          4. Context:        ctx = Σ_t α_t * H_t           shape (B, hidden_size)
             (implemented as batch matrix multiply)

        Returns:
            context : (B, hidden_size)  — single weighted summary of H
            alpha   : (B, T)            — attention weights (for visualisation)
        """
        # Step 1: project every hidden state through W_b, then tanh
        # H     : (B, T, H)
        # W_b   : (attn_size, H) → effectively a linear layer without bias
        keys = torch.tanh(
            torch.matmul(H, self.W_b.T)        # (B, T, attn_size)
        )

        # Step 2: collapse to a scalar score per timestep
        # W_a   : (1, attn_size)
        scores = torch.matmul(keys, self.W_a.T)   # (B, T, 1)
        scores = scores.squeeze(-1)               # (B, T)

        # Step 3: softmax over T → attention weights α
        alpha = F.softmax(scores, dim=-1)          # (B, T)

        # Step 4: context = weighted sum of hidden states
        # alpha : (B, T)   →  unsqueeze → (B, 1, T)
        # H     : (B, T, H)
        # bmm   : (B, 1, T) × (B, T, H) = (B, 1, H)
        context = torch.bmm(alpha.unsqueeze(1), H)  # (B, 1, H)
        context = context.squeeze(1)                 # (B, H)

        return context, alpha

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x:      torch.Tensor,
        hidden: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Forward pass: RNN unroll → global attention → output projection.

        The attention is applied ONCE over the entire sequence output of
        the top RNN layer, producing a single context vector that is
        broadcast to all timesteps before the final projection.

        Returns:
            logits  : (B, T, vocab_size)
            hidden  : updated hidden states per layer
            alpha   : (B, T) attention weights (for report / visualisation)
        """
        batch_size, seq_len = x.shape
        device = x.device

        if hidden is None:
            hidden = self._init_hidden(batch_size, device)

        # ── Step 1: embed input ───────────────────────────────────────────
        embeds     = self.embedding(x)   # (B, T, embed_dim)
        layer_input = embeds

        # ── Step 2: run through stacked RNN layers ────────────────────────
        for layer_idx in range(self.num_layers):
            h_prev   = hidden[layer_idx]
            W_xh     = self.W_xh[layer_idx]
            W_hh     = self.W_hh[layer_idx]
            b_h      = self.b_h[layer_idx]

            timestep_outputs = []

            for t in range(seq_len):
                x_t   = layer_input[:, t, :]   # (B, in_dim)
                pre   = F.linear(x_t, W_xh) + F.linear(h_prev, W_hh) + b_h
                h_t   = torch.tanh(pre)         # (B, H)
                timestep_outputs.append(h_t)
                h_prev = h_t

            layer_output = torch.stack(timestep_outputs, dim=1)  # (B, T, H)
            layer_output = self.dropout(layer_output)

            hidden[layer_idx] = h_prev
            layer_input = layer_output

        # layer_input is now the top-layer hidden states H : (B, T, H)
        H = layer_input

        # ── Step 3: compute attention over H ──────────────────────────────
        # Uses W_a and W_b to compute additive (Bahdanau-style) attention
        # over all timestep hidden states, producing a single context vector
        # and the attention weight distribution alpha for visualisation.
        context, alpha = self._compute_attention(H)  # (B, H), (B, T)

        # ── Step 4: augment each hidden state with the context vector ─────
        # Broadcast context across all T timesteps
        context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, H)

        # Concatenate [h_t || context]  along the feature dimension
        augmented = torch.cat([H, context_expanded], dim=-1)   # (B, T, 2*H)

        # ── Step 5: project to vocabulary logits ──────────────────────────
        logits = self.output_proj(augmented)   # (B, T, vocab_size)

        return logits, hidden, alpha

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)