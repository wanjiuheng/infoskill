"""
models/encoder_t5.py

Fast Module — T5 热启动版 Encoder（exp4 模块消融实验）。

与 models/encoder.py 的冷启动 MLP 版不同，这里用一个真实预训练过的
google/t5-efficient-tiny 的 encoder 部分（热启动权重）替代，直接吃原始
文本（不再依赖 utils/embedding.py 的 Qwen mean-pool 词袋向量）。

T5 encoder 由外部构造一次（shared_t5，见 train.py 的 build_fast_modules），
本模块只持有对 shared_t5.encoder 的引用（不注册为子模块，见
_t5_encoder_holder 注释），从而与 T5GroundingDecoder
（models/grounding_decoder_t5.py）共享同一份 embedding 权重，符合 T5
预训练时的权重绑定关系，且避免这份共享权重在优化器里被重复添加。

PriorNetwork / RewardPredictor 不需要新的 T5 版本：它们的输入从旧的
Qwen 3584维 mean-pool 向量换成本模块输出的 pooled_state（d_model=256），
构造时只需要把 state_dim 参数改成 d_model，直接复用 models/encoder.py 的
PriorNetwork 和 models/reward_predictor.py 的 RewardPredictor 即可，不需要
新建 T5 版本。
"""

import logging
import torch
import torch.nn as nn
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TruncationMonitor:
    """
    跟踪 tokenize 时发生截断的次数（决策：加监控而非假设 max_length 够用，
    因为旧的 get_text_embedding 一直静默截断，没人验证过是否丢信息）。
    每 log_every 次 forward 打印一次累计截断率。
    """

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


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding tokens. last_hidden_state: [B,L,D], attention_mask: [B,L]."""
    mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)        # [B, D]
    counts = mask.sum(dim=1).clamp(min=1e-9)              # [B, 1]
    return summed / counts


class T5StateConditionalEncoder(nn.Module):
    """
    posterior q(z|s,skill)：用共享 T5 encoder 分别编码 state_text / skill_text
    （各自一次前向），mean-pool 后拼接，过一个小 MLP 头得到 (mu, log_var)。

    额外返回 pooled_state（state 侧单独池化后的 d_model 维向量），供
    PriorNetwork / RewardPredictor / GroundingDecoder 统一使用——决策：整条
    链路的"state"表示统一用 T5 产出的池化向量，不再用旧的 Qwen 词袋向量，
    保证 posterior/prior 的条件变量一致（KL 才有意义）。
    """

    def __init__(
        self,
        t5_encoder: nn.Module,      # shared_t5.encoder（外部构造，本类只持有引用）
        t5_tokenizer,               # T5 自己的 tokenizer（与 Qwen tokenizer 独立并存）
        d_model: int,                # T5 encoder 的 hidden size（t5-efficient-tiny: 256）
        latent_dim: int,             # z_tilde 维度（64，与其余分支一致）
        max_length: int = 256,
    ) -> None:
        super().__init__()
        # 不注册成子模块：用单元素 list 持有引用，规避 nn.Module.__setattr__
        # 对直接赋值的 nn.Module 自动注册子模块的行为。这样 encoder 与
        # grounding_decoder 共享同一份 T5 权重时，各自的 .parameters() 都不
        # 包含它，build_optimizer 只需手动把 shared_t5.parameters() 加一次，
        # 不会被两个 wrapper 重复注册。
        self._t5_encoder_holder = [t5_encoder]
        self.tokenizer = t5_tokenizer
        self.d_model = d_model
        self.max_length = max_length

        self.fc_mu = nn.Linear(d_model * 2, latent_dim)
        self.fc_logvar = nn.Linear(d_model * 2, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        self._trunc_monitor_state = TruncationMonitor("state_text")
        self._trunc_monitor_skill = TruncationMonitor("skill_text")

    @property
    def _t5_encoder(self) -> nn.Module:
        return self._t5_encoder_holder[0]

    def train(self, mode: bool = True) -> "T5StateConditionalEncoder":
        """
        覆写 train()：_t5_encoder 存在 list 里，不是注册的子模块，
        nn.Module.train() 默认的 self.children() 递归遍历不到它，
        shared_t5 内部的 dropout 层会一直停留在 training=True（即使
        eval() 时也不会真正关闭）。手动同步一次，保证 encoder.eval()/
        train() 对 shared_t5 生效（nn.Module.eval() 内部就是调
        self.train(False)，所以只需覆写这一个方法）。
        """
        super().train(mode)
        self._t5_encoder.train(mode)
        return self

    def _encode_pool(self, texts: List[str], monitor: "TruncationMonitor") -> torch.Tensor:
        device = next(self._t5_encoder.parameters()).device
        monitor.record(self.tokenizer, texts, self.max_length)
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        ).to(device)
        out = self._t5_encoder(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask
        )
        return mean_pool(out.last_hidden_state, enc.attention_mask)  # [B, d_model]

    def forward(
        self,
        state_text: List[str],
        skill_text: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_text: 长度 B 的文本列表（"Task: ...\\nObservation: ..." 格式）
            skill_text: 长度 B 的技能 grounding_text 列表

        Returns:
            mu, log_var:   [B, latent_dim]
            pooled_state:  [B, d_model] — state 侧池化向量，供
                           PriorNetwork/RewardPredictor/GroundingDecoder 使用
        """
        pooled_state = self._encode_pool(state_text, self._trunc_monitor_state)  # [B, d_model]
        pooled_skill = self._encode_pool(skill_text, self._trunc_monitor_skill)  # [B, d_model]

        feat = torch.cat([pooled_state, pooled_skill], dim=-1)   # [B, 2*d_model]
        mu = self.fc_mu(feat)
        log_var = self.fc_logvar(feat).clamp(-10.0, 2.0)
        return mu, log_var, pooled_state
