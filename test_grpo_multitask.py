"""
test_grpo_multitask.py

离线单元测试：验证多任务 batch（rollout.tasks_per_batch > 1）下的按组切分逻辑。

核心正确性：一个 rollout batch 里可能有多个任务组（每个组 group_size 局），
GRPO advantage / 零方差判断 / fidelity mask 都必须按组独立计算，不能把不同
任务的 reward 混在一起归一化——否则任务难度差异会污染 advantage。

可跑的环境：
- 本机（无 torch）：自动退化为 numpy 参考实现，验证同一套切分数学性质
- 训练机（有 torch）：直接 import 真实的 compute_grpo_advantages_grouped
  做交叉验证

用法：python test_grpo_multitask.py
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
    from training.grpo import compute_grpo_advantages_grouped as real_grouped
    from training.grpo import compute_grpo_advantages as real_plain
except ImportError:
    HAVE_TORCH = False
    # numpy 参考实现：镜像 training/grpo.py 的数学，本地无 torch 时用
    def real_plain(rewards, eps=1e-8):
        rewards = np.asarray(rewards, dtype=np.float32)
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    def real_grouped(rewards, group_size, eps=1e-8):
        rewards = np.asarray(rewards, dtype=np.float32)
        assert group_size > 0
        assert rewards.size % group_size == 0
        return np.concatenate([
            real_plain(rewards[i:i+group_size], eps)
            for i in range(0, rewards.size, group_size)
        ])


def arr(x):
    if HAVE_TORCH:
        return x.numpy() if isinstance(x, torch.Tensor) else x
    return x


# ── 场景1：多任务组，任务A 有方差、任务B 零方差 ──────────────────────────────
# 任务A 4局: [10, 8, -5, -3]  → 非零方差，正常归一化
# 任务B 4局: [2, 2, 2, 2]     → 零方差，advantage 应为 0
rewards = [10.0, 8.0, -5.0, -3.0, 2.0, 2.0, 2.0, 2.0]
GS = 4

adv = arr(real_grouped(rewards, GS))

# 1a. 逐组分别算，拼接 → grouped 结果必须一致
exp_a = arr(real_plain(rewards[:GS]))
exp_b = arr(real_plain(rewards[GS:]))
assert np.allclose(adv[:GS], exp_a), "任务A 的 advantage 应等于单独算任务A"
assert np.allclose(adv[GS:], exp_b), "任务B 的 advantage 应等于单独算任务B"

# 1b. 任务B 零方差 → advantage 全 0（GRPO 的 std=0 分支）
assert np.allclose(adv[GS:], 0.0), "零方差组的 advantage 应为 0"

# 1c. grouped ≠ 整批混算（证明切片真的生效，而不是悄悄退化）
adv_mixed = arr(real_plain(rewards))
assert not np.allclose(adv, adv_mixed), "按组分 vs 整批混算必须不同（切片要生效）"

# 1d. 任务A 的 advantage 相对自身均值为正/负
assert adv[0] > 0 and adv[2] < 0, "任务A 高于均值的局 advantage>0，低于的<0"

# ── 场景2：zero_variance_per_group 逐组判断 ──────────────────────────────────
# 任务A std>0 → False；任务B std=0 → True
gstd = [float(np.asarray(rewards[i:i+GS], dtype=np.float32).std()) for i in range(0, len(rewards), GS)]
zero_var_group = [s < 1e-6 for s in gstd]
assert zero_var_group == [False, True], f"逐组零方差判断错误: {zero_var_group}"

# 展开成逐 episode 标记（镜像 trainer.py 的 zero_variance_per_ep）
n_ep = len(rewards)
zvar_ep = [zero_var_group[ep // GS] for ep in range(n_ep)]
assert zvar_ep == [False]*GS + [True]*GS, f"逐 episode 展开错误: {zvar_ep}"

# ── 场景3：fidelity_mask_full 展开（镜像 trainer.py 逻辑）────────────────────
# active_records 里每条带 ep_idx；任务B 的 ep 全置 0（跳过 fidelity），任务A 全 1
active_ep_idx = [0, 1, 2, 3, 5, 6, 7]   # 模拟：任务A 4条 + 任务B 部分有效
fidelity_mask_full = np.array(
    [0.0 if zvar_ep[ep] else 1.0 for ep in active_ep_idx], dtype=np.float32
)
exp_mask = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
assert np.array_equal(fidelity_mask_full, exp_mask), (
    f"fidelity_mask_full 展开错误: {fidelity_mask_full} vs {exp_mask}"
)

# ── 场景4：tasks_per_batch=1 回归（不改变原行为）────────────────────────────
# 单任务组时 grouped == 整批一次算（compute_grpo_advantages 本身）
rewards1 = [10.0, 8.0, -5.0, -3.0]
adv_single = arr(real_grouped(rewards1, 4))
assert np.allclose(adv_single, arr(real_plain(rewards1))), "单任务组应退化为普通 GRPO"

# 多个任务、每任务 1 局（group_size=1）也是合法输入（退化）
rewards_n1 = [3.0, 1.0, 2.0, 4.0]
adv_n1 = arr(real_grouped(rewards_n1, 1))
assert np.allclose(adv_n1, 0.0), "group_size=1 时每局 self-normalize → advantage 全 0"

print("ALL PASS" + (" (torch 路径)" if HAVE_TORCH else " (numpy 参考路径)"))
