"""
models/grounding_decoder_t5.py

Fast Module — T5 热启动版 GroundingDecoder（exp4 模块消融实验）。

用共享 T5 的 decoder + lm_head（与 encoder 权重绑定，见 models/encoder_t5.py）
通过原生 cross-attention 重建 skill 文本。[z_tilde‖pooled_state] 投影成
NUM_MEMORY_TOKENS 个向量喂给 T5 decoder 的 cross-attention 作为 memory
（决策记录：1 个 token 会让 cross-attention 的 softmax 退化成恒定权重，
发挥不出"选择性"；4 个才能让这套预训练权重真正按原本方式工作。这个数字
后续可能还会调整，和 exp3 的 prefix token 数一并归入"条件信息容量"消融）。

全程使用 T5 自己的 tokenizer/vocab（与 Qwen tokenizer 完全独立并存），
grounding loss 的监督目标也用 T5 tokenizer 编码——保持 T5 训练时
embedding/decoder/输出头的原生对齐关系，真正利用热启动权重（若换成
Qwen vocab，等于只保留了中间层，输出头/embedding层的热启动优势归零，
跟随机初始化没有本质区别）。
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

logger = logging.getLogger(__name__)


class TruncationMonitor:
    """见 models/encoder_t5.py 的同名类，独立一份避免跨文件依赖。"""

    def __init__(self, name: str, log_every: int = 200) -> None:
        self.name = name
        self.log_every = log_every
        self.total = 0
        self.truncated = 0

    def record(self, tokenizer, texts: List[str], max_length: int) -> None:
        for t in texts:
            self.total += 1
            ids = tokenizer(t, truncation=False)["input_ids"]
            if len(ids) > max_length:
                self.truncated += 1
        if self.total and self.total % self.log_every == 0:
            rate = self.truncated / self.total
            logger.info(
                "[TruncationMonitor:%s] %d/%d (%.2f%%) texts exceeded max_length=%d so far",
                self.name, self.truncated, self.total, rate * 100, max_length,
            )


class T5GroundingDecoder(nn.Module):
    """
    用共享 T5 的 decoder + lm_head 重建 skill 文本，[z_tilde‖pooled_state]
    投影成 NUM_MEMORY_TOKENS 个向量，喂给 T5 原生 cross-attention 作为 memory。

    接口：
      forward(z_tilde, pooled_state, target_texts=None)
        target_texts 给定 → 返回 scalar CE loss（训练模式，teacher forcing，
                              内部用 T5 tokenizer 编码 target_texts）
        target_texts=None → 返回 [B, max_len] 生成的 token ids（推理/调试模式，
                              贪心；解码请用 self.tokenizer.decode，不能用
                              Qwen tokenizer，否则是乱码）
    """

    NUM_MEMORY_TOKENS = 4

    def __init__(
        self,
        t5_decoder: nn.Module,      # shared_t5.decoder（外部构造，只持有引用）
        lm_head: nn.Module,         # shared_t5.lm_head（与 embedding weight tying）
        t5_tokenizer,               # T5 自己的 tokenizer
        d_model: int,                # T5 hidden size（t5-efficient-tiny: 256）
        latent_dim: int,
        pooled_state_dim: int,       # = d_model（来自 T5StateConditionalEncoder）
        max_len: int = 64,
    ) -> None:
        super().__init__()
        # 不注册成子模块，规避 encoder/decoder 共享权重被优化器重复添加
        # （见 models/encoder_t5.py 的同一注释）。
        self._t5_decoder_holder = [t5_decoder]
        self._lm_head_holder = [lm_head]
        self.tokenizer = t5_tokenizer
        self.d_model = d_model
        self.max_len = max_len

        # [z_tilde‖pooled_state] → NUM_MEMORY_TOKENS 个 d_model 维向量
        self.fc_memory = nn.Linear(
            latent_dim + pooled_state_dim, self.NUM_MEMORY_TOKENS * d_model
        )

        self.pad_token_id = t5_tokenizer.pad_token_id or 0
        self._trunc_monitor = TruncationMonitor("grounding_target")

    @property
    def _t5_decoder(self) -> nn.Module:
        return self._t5_decoder_holder[0]

    @property
    def _lm_head(self) -> nn.Module:
        return self._lm_head_holder[0]

    def train(self, mode: bool = True) -> "T5GroundingDecoder":
        """
        覆写 train()：_t5_decoder/_lm_head 存在 list 里，不是注册的子模块，
        nn.Module.train() 默认的递归遍历不到它们（同 models/encoder_t5.py
        的 T5StateConditionalEncoder.train() 注释）。目前项目里没有任何
        地方调用 grounding_decoder.eval()/train()（grounding_decoder 只在
        训练路径的 _compute_grounding_loss 里被调用），但覆写一下避免
        以后有人加了 eval() 调用却踩到同一个坑。
        """
        super().train(mode)
        self._t5_decoder.train(mode)
        return self

    def _build_memory(self, z_tilde: torch.Tensor, pooled_state: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([z_tilde, pooled_state], dim=-1)             # [B, latent+pooled]
        flat = self.fc_memory(feat)                                    # [B, NUM*d_model]
        B = z_tilde.size(0)
        return flat.view(B, self.NUM_MEMORY_TOKENS, self.d_model)      # [B, NUM, d_model]

    def forward(
        self,
        z_tilde: torch.Tensor,
        pooled_state: torch.Tensor,
        target_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        memory = self._build_memory(z_tilde, pooled_state)   # [B, NUM, d_model]
        if target_texts is not None:
            return self._forward_train(memory, target_texts)
        return self._forward_infer(memory)

    def _forward_train(self, memory: torch.Tensor, target_texts: List[str]) -> torch.Tensor:
        device = memory.device
        self._trunc_monitor.record(self.tokenizer, target_texts, self.max_len)
        enc = self.tokenizer(
            target_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_len,
        ).to(device)
        target_ids = enc.input_ids                     # [B, seq_len]

        B = target_ids.size(0)
        bos = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)  # shift right

        # T5Stack（is_decoder=True，来自 shared_t5.decoder）内部自动应用
        # causal self-attention mask + cross-attention over encoder_hidden_states。
        out = self._t5_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
        )
        logits = self._lm_head(out.last_hidden_state)   # [B, seq_len, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def _forward_infer(self, memory: torch.Tensor) -> torch.Tensor:
        """Greedy auto-regressive generation; returns token ids [B, max_len]."""
        B = memory.size(0)
        device = memory.device
        current = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        generated = []
        for _ in range(self.max_len):
            out = self._t5_decoder(input_ids=current, encoder_hidden_states=memory)
            logits = self._lm_head(out.last_hidden_state[:, -1:, :])   # [B, 1, V]
            next_tok = logits.argmax(dim=-1)                            # [B, 1]
            generated.append(next_tok)
            current = torch.cat([current, next_tok], dim=1)
        return torch.cat(generated, dim=1)
