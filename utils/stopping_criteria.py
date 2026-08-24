"""
utils/stopping_criteria.py

生成提前停止工具：当 LLM 输出的文本里出现 </action> 标签（动作已经给出）时
立即停止继续生成，省掉动作之后多余的 token（rollout / eval 提速）。

为什么用自定义 StoppingCriteria 而不是 generation_config.stop_strings：
- stop_strings 需要较新的 transformers 版本才支持，训练机版本不确定；
- 显式控制检测窗口与批量语义，行为完全可控。

注意：transformers 的 StoppingCriteria 返回值是整批共享的一个布尔值——一
旦返回 True，batch 里所有序列（包括还没写完 </action> 的）都会被同时截断
（这是 transformers 的已知限制，见
https://github.com/huggingface/transformers/issues/22340）。

2026-08-24 debug run 实测踩到了这个坑：批内 4 个序列逐 token 同步生成
（lockstep），一旦有序列先写完 </action>（哪怕只有 1 条），就把同批里还在
写 think/action 的其它序列全部腰斩在词中间（如 "go to countertop" 缺
"1</action>"、"navigate to the sidetable, (" 断在括号里）。这些被腰斩的
序列 parse_action 解析不出合法 <action>，判 is_valid=False，导致那次 run
120 条 step 里只有 30 条（25%）进入 update——训练信号被砍掉四分之三。

修复：改成"批内全部序列都已生成 </action>"才停（all 语义，而非 any）。
这样不会有任何序列被过早截断，只是当批内最慢的那条也写完动作后，才把
多余的收尾 token 一起省掉——退化到"不省"的最坏情况就是等于没加这个优化，
但绝不会牺牲正确性。
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
        # 必须批内所有序列都已生成 </action> 才停止（all 语义）——只要有一条
        # 还没写完，就继续生成，避免把它腰斩在词中间（is_valid 会判它无效）。
        recent = input_ids[:, -self.window:]
        for row in recent:
            if self.stop_str not in self.tokenizer.decode(row, skip_special_tokens=True):
                return False
        return True


def build_action_stop_criteria(tokenizer) -> StoppingCriteriaList:
    """构造用于检测 </action> 的 StoppingCriteriaList，传给 model.generate()。"""
    return StoppingCriteriaList([_StopAtActionEnd(tokenizer)])
