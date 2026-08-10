"""
models/grounding_decoder.py

Fast Module — GroundingDecoder: reconstruct Skill text from z_tilde + state_emb.

Grounding loss = CrossEntropy(logits, target_ids), where the target is the
skill's "title: principle" text tokenised by the LLM's tokenizer.

Purpose: forces z_tilde to remain interpretable (decodable back to a human
sentence) rather than collapsing to an arbitrary latent representation.
The decoder is a lightweight LSTM text head — intentionally small so it doesn't
dominate training.

Training mode:  pass target_ids  → returns scalar CE loss.
Inference mode: target_ids=None  → returns generated token ids (for debugging).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GroundingDecoder(nn.Module):
    """
    LSTM-based text decoder that reconstructs the skill text from z_tilde.

    The initial LSTM hidden state is derived from (z_tilde, state_emb).
    Teacher-forcing is used during training for stable gradients.
    """

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_len: int = 64,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.max_len      = max_len
        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size

        # Project (z_tilde ‖ state_emb) → initial hidden state for every LSTM layer
        self.fc_init = nn.Linear(latent_dim + state_dim, hidden_dim * num_layers)

        # Token embedding (shared with output head weight via weight tying)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Weight tying: output projection shares weights with embedding
        self.fc_out.weight = self.embedding.weight

    def _init_hidden(
        self,
        z_tilde: torch.Tensor,    # [B, latent_dim]
        state_emb: torch.Tensor,  # [B, state_dim]
    ):
        """Create initial (h_0, c_0) for the LSTM from the latent code."""
        B = z_tilde.size(0)
        feat = torch.cat([z_tilde, state_emb], dim=-1)             # [B, latent+state]
        h_flat = torch.tanh(self.fc_init(feat))                     # [B, hidden*layers]
        h_0 = h_flat.view(B, self.num_layers, self.hidden_dim)      # [B, L, H]
        h_0 = h_0.permute(1, 0, 2).contiguous()                    # [L, B, H]
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

    def forward(
        self,
        z_tilde:    torch.Tensor,            # [B, latent_dim]
        state_emb:  torch.Tensor,            # [B, state_dim]
        target_ids: Optional[torch.Tensor] = None,  # [B, seq_len] — for teacher forcing
    ) -> torch.Tensor:
        """
        Training mode  (target_ids provided):
            Returns scalar CrossEntropy loss averaged over non-padding tokens.
        Inference mode (target_ids=None):
            Returns generated token ids [B, max_len] (greedy, for debugging).
        """
        h, c = self._init_hidden(z_tilde, state_emb)

        if target_ids is not None:
            return self._forward_train(h, c, target_ids)
        else:
            return self._forward_infer(h, c, z_tilde.device, z_tilde.size(0))

    def _forward_train(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        target_ids: torch.Tensor,  # [B, seq_len]
    ) -> torch.Tensor:
        """Teacher-forcing forward; returns scalar CE loss."""
        B, seq_len = target_ids.shape

        # Shift right: input = [<BOS>, t0, t1, ..., t_{N-2}]
        #              target = [t0,   t1, ..., t_{N-1}]
        bos = torch.full((B, 1), self.pad_token_id, dtype=torch.long,
                         device=target_ids.device)
        decoder_input = torch.cat([bos, target_ids[:, :-1]], dim=1)  # [B, seq_len]

        embedded = self.embedding(decoder_input)                      # [B, seq_len, H]
        lstm_out, _ = self.lstm(embedded, (h, c))                    # [B, seq_len, H]
        logits = self.fc_out(lstm_out)                               # [B, seq_len, V]

        # Flatten for cross-entropy; ignore padding positions
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def _forward_infer(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Greedy auto-regressive generation; returns token ids [B, max_len]."""
        current = torch.full(
            (batch_size, 1), self.pad_token_id,
            dtype=torch.long, device=device,
        )
        generated = []
        for _ in range(self.max_len):
            emb = self.embedding(current)              # [B, 1, H]
            out, (h, c) = self.lstm(emb, (h, c))      # [B, 1, H]
            logit = self.fc_out(out)                   # [B, 1, V]
            next_tok = logit.argmax(dim=-1)            # [B, 1]
            generated.append(next_tok)
            current = next_tok
        return torch.cat(generated, dim=1)             # [B, max_len]
