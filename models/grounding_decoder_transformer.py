"""
models/grounding_decoder_transformer.py

Fast Module — GroundingDecoder（Transformer 版）：与 models/grounding_decoder.py
（LSTM 版）功能等价，接口完全一致，仅内部架构不同。用于 exp3 模块消融实验
（"LSTM → 2层Transformer" 是唯一被测变量，其余设计尽量对齐 LSTM 版）：

  - decoder-only + causal mask（不用 cross-attention/memory，条件向量当作
    prefix token 拼进目标序列最前面，和 LLM 侧 soft prefix 的设计同构）
  - [z_tilde‖state_emb] 投影成 1 个 prefix token（对齐 LSTM 版 h_0 的信息量，
    不引入"prefix token 数"这个新变量）
  - 可学习位置 embedding（序列短、任务轻量，不需要 RoPE）
  - 2 层，8 头，hidden_dim=512（与 LSTM 版一致），ffn=2048，dropout=0.1
  - 输出层与 embedding 权重共享（weight tying，与 LSTM 版一致）

Grounding loss = CrossEntropy(logits, target_ids)，target 是 skill 的
"title: principle" 文本 tokenised by the LLM's tokenizer.

Training mode:  pass target_ids  → returns scalar CE loss.
Inference mode: target_ids=None  → returns generated token ids (for debugging).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GroundingDecoder(nn.Module):
    """
    2层 decoder-only Transformer，把 [z_tilde‖state_emb] 投影成 1 个 prefix
    token 拼在目标序列最前面，用 causal self-attention 重建 skill 文本。

    接口与 models/grounding_decoder.py 的 LSTM 版完全一致：
      forward(z_tilde, state_emb, target_ids=None)
        target_ids 给定 → 返回 scalar CE loss（训练模式，teacher forcing）
        target_ids=None → 返回 [B, max_len] 生成的 token ids（推理模式，贪心）
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
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.max_len      = max_len
        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size

        # 条件向量 [z_tilde‖state_emb] → 1 个 prefix token（对齐 LSTM 版 h_0
        # 的信息注入方式：一次性把条件信息映射成固定长度的表示）
        self.fc_prefix = nn.Linear(latent_dim + state_dim, hidden_dim)

        # Token embedding（与输出头 weight tying，与 LSTM 版一致）
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        # 可学习位置 embedding：序列 = [prefix token] + [seq_len 个目标 token]，
        # 长度上限 max_len + 1
        self.pos_embedding = nn.Embedding(max_len + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        # decoder-only：用 TransformerEncoder + causal mask 实现自回归，不需要
        # cross-attention 的 memory 参数（方案B：prefix token 而非 memory）
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        # Weight tying: 输出投影与 embedding 共享权重（与 LSTM 版一致）
        self.fc_out.weight = self.embedding.weight

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = masked out), for nn.TransformerEncoder."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def _prefix_token(
        self,
        z_tilde: torch.Tensor,    # [B, latent_dim]
        state_emb: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:
        """[B, latent_dim+state_dim] → [B, 1, hidden_dim]."""
        feat = torch.cat([z_tilde, state_emb], dim=-1)
        prefix = self.fc_prefix(feat)               # [B, hidden_dim]
        return prefix.unsqueeze(1)                  # [B, 1, hidden_dim]

    def forward(
        self,
        z_tilde:    torch.Tensor,
        state_emb:  torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prefix = self._prefix_token(z_tilde, state_emb)   # [B, 1, H]

        if target_ids is not None:
            return self._forward_train(prefix, target_ids)
        else:
            return self._forward_infer(prefix)

    def _forward_train(
        self,
        prefix:     torch.Tensor,  # [B, 1, H]
        target_ids: torch.Tensor,  # [B, seq_len]
    ) -> torch.Tensor:
        """Teacher-forcing forward; returns scalar CE loss."""
        B, seq_len = target_ids.shape
        device = target_ids.device

        # Shift right: decoder_input = [<BOS>, t0, ..., t_{N-2}]
        #              target        = [t0,    t1, ..., t_{N-1}]
        bos = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos, target_ids[:, :-1]], dim=1)   # [B, seq_len]

        tok_emb = self.embedding(decoder_input)          # [B, seq_len, H]
        full_seq = torch.cat([prefix, tok_emb], dim=1)    # [B, 1+seq_len, H]

        total_len = full_seq.size(1)
        positions = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]
        full_seq = full_seq + self.pos_embedding(positions)

        mask = self._causal_mask(total_len, device)
        out = self.transformer(full_seq, mask=mask)       # [B, 1+seq_len, H]

        # 丢掉 prefix 位置对应的输出，只保留预测目标 token 的那部分
        logits = self.fc_out(out[:, 1:, :])                # [B, seq_len, V]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def _forward_infer(self, prefix: torch.Tensor) -> torch.Tensor:
        """Greedy auto-regressive generation; returns token ids [B, max_len]."""
        B = prefix.size(0)
        device = prefix.device

        current_ids = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        generated = []
        for _ in range(self.max_len):
            tok_emb = self.embedding(current_ids)             # [B, t, H]
            full_seq = torch.cat([prefix, tok_emb], dim=1)    # [B, 1+t, H]
            total_len = full_seq.size(1)
            positions = torch.arange(total_len, device=device).unsqueeze(0)
            full_seq = full_seq + self.pos_embedding(positions)

            mask = self._causal_mask(total_len, device)
            out = self.transformer(full_seq, mask=mask)
            logit = self.fc_out(out[:, -1:, :])               # [B, 1, V]
            next_tok = logit.argmax(dim=-1)                   # [B, 1]
            generated.append(next_tok)
            current_ids = torch.cat([current_ids, next_tok], dim=1)

        return torch.cat(generated, dim=1)                    # [B, max_len]
