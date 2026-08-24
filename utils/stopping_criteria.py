"""
utils/stopping_criteria.py

生成提前停止工具：当 LLM 输出的文本里出现 </action> 标签（动作已经给出）时
立即停止继续生成，省掉动作之后多余的 token（rollout / eval 提速）。

为什么用自定义 StoppingCriteria 而不是 generation_config.stop_strings：
- stop_strings 需要较新的 transformers 版本才支持，训练机版本不确定；
- 显式控制检测窗口与批量语义，行为完全可控。

注意：transformers 的 StoppingCriteria 是整批共享的——任一序列触发即整批
停止。rollout 中同一批是同一 GRPO 组（同任务、同 step 同步推进），组内输出
长度高度相似，连带截断风险很小；即使个别序列被截断，只要 </action> 已完整
生成（这正是触发条件），parse_action 仍能提取出动作，后续 is_valid 处理逻辑
不变。
"""

from __future__ import annotations

from typing import Any

from transformers import StoppingCriteria, StoppingCriteriaList

# </action> 只有 8 个字符，tokenizer 一般切成 2-4 个 token；检测窗口给足
# 32 个 token，覆盖任何跨 token 切分方式，同时避免每步解码整条序列的开销。
_ACTION_END_TAG = "</action>"
_WINDOW_TOKENS = 32


class _StopAtActionEnd(StoppingCriteria):
    """检测生成文本中出现 </action> 即返回 True（停止生成）。"""

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_str = _ACTION_END_TAG
        self.window = _WINDOW_TOKENS

    def __call__(self, input_ids, scores, **kwargs: Any) -> bool:
        # input_ids: [batch, cur_len]，只看每个序列最近 window 个 token 的文本。
        # 任一序列检测到 stop_str → 整批停止（transformers 语义）。
        recent = input_ids[:, -self.window:]
        for row in recent:
            if self.stop_str in self.tokenizer.decode(row, skip_special_tokens=True):
                return True
        return False


def build_action_stop_criteria(tokenizer) -> StoppingCriteriaList:
    """构造用于检测 </action> 的 StoppingCriteriaList，传给 model.generate()。"""
    return StoppingCriteriaList([_StopAtActionEnd(tokenizer)])
