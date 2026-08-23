"""
scripts/check_eval_corruption.py

诊断脚本：判定"第一次 eval 结束后紧跟的 rollout 是否输出退化乱码"。

背景：run lr1e-06_bs8_mnt512_ms30_run_20260820_211252 里观察到的现象——
eval 前 rollout 每步只生成 47-147 token（think+action 长度合理），eval
结束后紧跟的第一个 rollout group 从 Step 0 起每步都打满 max_new_tokens，
且持续到后续所有 group（loss 全 0，模型再也没更新过，说明不是权重问题）。
本脚本把"每步 token 数是否顶到 max_new_tokens 上限"作为退化信号，从
train_progress.log 里切分出 rollout group、定位每次 eval 的位置，输出每个
group 的 token 范围与饱和比例，并以【第一次 eval】为分界判定 eval 前/后
rollout 是否发生退化，不需要人工翻日志。

注意：
- 以第一次 eval 为基准。多次 eval 时，后面的 eval 位置也会标出来，但
  判定只看第一次 eval 前后。
- 饱和信号是内容无关的：不管退化后落进"一堆 0"还是"中文/HTML/emoji"
  哪种先验吸引子，共同特征都是"没产生 EOS、刷满整个生成预算"。

用法：
    python scripts/check_eval_corruption.py <run_log_dir>/train_progress.log
    python scripts/check_eval_corruption.py <run_log_dir>/train_progress.log --max-new-tokens 128

配合 configs/alfworld_debug_eval_repro.yaml 使用：
    python train.py --config configs/alfworld_debug_eval_repro.yaml --run-name eval_repro_debug
    python scripts/check_eval_corruption.py logs_debug_eval_repro/eval_repro_debug/train_progress.log --max-new-tokens 128
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# "2026-08-21 01:04:27,495 [INFO] rollout: Step 0: env_0=512tok, env_1=512tok, ..."
STEP_LINE_RE = re.compile(
    r"^(?P<ts>\S+ \S+) \[INFO\] rollout: Step (?P<step>\d+): (?P<detail>env_.*)$"
)
TOK_RE = re.compile(r"env_(\d+)=(\d+)tok")

# "2026-08-21 01:04:00,564 [INFO] eval.evaluate: Eval Results: 140 episodes"
EVAL_RESULTS_RE = re.compile(r"^(?P<ts>\S+ \S+) \[INFO\] eval\.evaluate: Eval Results: \d+ episodes")


class RolloutGroup:
    def __init__(self, first_ts: str):
        self.first_ts = first_ts
        self.steps: List[Tuple[int, List[int]]] = []  # (step_idx, [tok_counts])

    def max_tok(self) -> int:
        return max((t for _, toks in self.steps for t in toks), default=0)

    def min_tok(self) -> int:
        return min((t for _, toks in self.steps for t in toks), default=0)

    def saturated_step_ratio(self, cap: int) -> float:
        """有多大比例的 (step, env) 组合打满了 max_new_tokens 上限。"""
        total = sum(len(toks) for _, toks in self.steps)
        if total == 0:
            return 0.0
        saturated = sum(1 for _, toks in self.steps for t in toks if t >= cap)
        return saturated / total


def parse_groups_and_eval_markers(
    log_path: Path,
) -> Tuple[List[RolloutGroup], List[int]]:
    """
    扫描 train_progress.log，把连续的 "Step 0, Step 1, ..." 序列切成一个个
    rollout group（Step 计数回到 0 就是新的一组），并记录每次 "Eval Results"
    出现在第几个 group 之后。
    """
    groups: List[RolloutGroup] = []
    current: Optional[RolloutGroup] = None
    eval_markers: List[int] = []   # 每次 eval 完成时已结束的 group 数
    eval_marker_pending = False

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if EVAL_RESULTS_RE.match(line):
                eval_marker_pending = True
                continue

            m = STEP_LINE_RE.match(line)
            if not m:
                continue

            step_idx = int(m.group("step"))
            toks = [int(v) for _, v in TOK_RE.findall(m.group("detail"))]

            if step_idx == 0:
                if current is not None:
                    groups.append(current)
                current = RolloutGroup(first_ts=m.group("ts"))
                if eval_marker_pending:
                    eval_markers.append(len(groups))  # eval 结束于 group{len(groups)} 之后
                    eval_marker_pending = False
            if current is not None:
                current.steps.append((step_idx, toks))

    if current is not None:
        groups.append(current)

    return groups, eval_markers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_path", type=str, help="train_progress.log 路径")
    ap.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="rollout.max_new_tokens 配置值。不传则从数据里自动猜测"
             "（取所有 group 里出现过的最大 token 数）。",
    )
    ap.add_argument(
        "--saturation-threshold", type=float, default=0.8,
        help="判定'退化'的饱和比例阈值（该组里有多大比例的 step*env 打满上限），默认 0.8。",
    )
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"找不到日志文件: {log_path}", file=sys.stderr)
        sys.exit(1)

    groups, eval_markers = parse_groups_and_eval_markers(log_path)

    if not groups:
        print("日志里没有解析到任何 'rollout: Step N: env_X=Ytok' 记录，"
              "确认 log_path 指向的是 train_progress.log 而不是 rollout_actions.log。")
        sys.exit(1)

    cap = args.max_new_tokens
    if cap is None:
        cap = max(g.max_tok() for g in groups)
        print(f"[自动推断] 未传 --max-new-tokens，猜测上限为观测到的最大值: {cap}")

    print(f"共解析到 {len(groups)} 个 rollout group，max_new_tokens 上限 = {cap}。")
    if not eval_markers:
        print("日志里没有出现 'Eval Results' —— 说明这次跑还没触发过 eval，"
              "或者 eval_freq 设置得比 num_episodes 还大，检查 config 里的 "
              "training.eval_freq。")
        sys.exit(2)

    eval_after = set(eval_markers)
    print()
    print("== 每个 group 的 token 概况 ==")
    for idx, g in enumerate(groups, 1):
        ratio = g.saturated_step_ratio(cap)
        mark = ""
        if (idx - 1) in eval_after:
            mark = "  <-- eval 在这里结束"
        print(
            f"  group{idx:<3} 起始 {g.first_ts}  token范围[{g.min_tok():>4}, "
            f"{g.max_tok():>4}]  饱和比例 {ratio:>3.0%}{mark}"
        )
    print()

    first_eval = eval_markers[0]   # 以第一次 eval 为判定基准
    print(f"第一次 eval 在 group{first_eval} 之后完成 "
          f"（group{first_eval} 是 eval 前最后一组，"
          f"group{first_eval + 1} 是 eval 后第一组）。")
    print()

    def describe(idx: int, label: str) -> Optional[RolloutGroup]:
        if idx < 1 or idx > len(groups):
            print(f"  {label}: group{idx} 不存在（日志里还没跑到这一组）。")
            return None
        g = groups[idx - 1]
        ratio = g.saturated_step_ratio(cap)
        print(
            f"  {label}: group{idx} (起始于 {g.first_ts}) "
            f"— token 范围 [{g.min_tok()}, {g.max_tok()}]，"
            f"饱和比例 {ratio:.0%}（阈值 {args.saturation_threshold:.0%}）"
        )
        return g

    print("== eval 前基线 ==")
    baseline = describe(first_eval, "基线组")
    print()
    print("== eval 后待检验组 ==")
    suspect = describe(first_eval + 1, "eval 后第一组")
    # 多看一组，确认崩坏是否持续（不是偶发抖动）
    suspect2 = describe(first_eval + 2, "eval 后第二组")
    print()

    if baseline is None or suspect is None:
        print("数据不完整，无法判定。等训练跑到 eval 后至少一组再重跑本脚本。")
        sys.exit(2)

    baseline_bad = baseline.saturated_step_ratio(cap) >= args.saturation_threshold
    suspect_bad = suspect.saturated_step_ratio(cap) >= args.saturation_threshold

    print("=" * 70)
    if baseline_bad:
        print("⚠️  基线组本身就已经饱和退化了 —— 这次复现里模型在 eval 之前就已经"
              "输出垃圾，不能把'eval 后崩坏'归因于 eval。检查 backbone/config 是否"
              "选对了（比如误用了没对齐过的原版权重），并对照上面的一览表确认"
              "第一次 eval 的位置（多次 eval 时脚本以第一次为基准）。")
        sys.exit(3)

    if suspect_bad:
        msg = "✅ 复现成功：eval 前正常，eval 后第一组立即饱和退化"
        if suspect2 is not None and suspect2.saturated_step_ratio(cap) >= args.saturation_threshold:
            msg += "，且持续到第二组（不是偶发抖动）"
        print(msg + "。")
        print("   → 坐实触发点是'第一次 eval 结束'本身，与模型权重更新无关。")
        print("   → 下一步可以按诊断建议，把 rollout 的 generate 改成在 model.eval() "
              "模式下跑（rollout 本来就在 torch.no_grad() 里，不需要梯度路径）。")
        sys.exit(0)
    else:
        print("❌ 没有复现：eval 后第一组仍然正常。")
        print("   可能原因：debug config 简化掉了某个必要条件（比如 group_size、"
              "max_steps 减小后没跑够步数让退化显现），或者本次改动已经修复了问题、"
              "或者该现象需要更长的 eval（更多 generate 调用次数）才会触发 —— 可以"
              "把 configs/alfworld_debug_eval_repro.yaml 里的 eval_episodes 调大一些"
              "再试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
