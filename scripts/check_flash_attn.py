"""
scripts/check_flash_attn.py

换机器装好 flash-attn 后的一键验证。依次检查：
  1. torch / CUDA / GPU 环境（flash-attn 依赖它们，先于 flash-attn 装）
  2. flash_attn 版本
  3. 一次真实的 FA2 forward（验证内核与当前 torch/CUDA 组合匹配，能跑通）

用法：python scripts/check_flash_attn.py
输出末尾 "ALL PASS: flash-attn 可用" 即通过；FAIL 说明安装/版本匹配有问题。

2026-08 训练机实测通过的环境组合：
  python 3.11.15 / torch 2.6.0+cu124 / cuda 12.4 / A800 80GB / flash-attn 2.7.4.post1
"""

import sys


def check_torch():
    try:
        import torch
    except ImportError:
        print("FAIL: torch 未安装（先装 torch 再装 flash-attn）")
        sys.exit(1)
    print(f"torch            : {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU              : {torch.cuda.get_device_name(0)}  ({p.total_memory / 2**30:.0f} GB)")
    else:
        print("FAIL: torch 看不到 CUDA。先修 torch/CUDA（驱动 vs torch build 版本），再装 flash-attn")
        sys.exit(1)
    return torch


def check_flash_attn():
    try:
        import flash_attn
    except ImportError:
        print("FAIL: flash_attn 未安装（pip install flash-attn==2.7.4.post1）")
        sys.exit(1)
    print(f"flash_attn       : {flash_attn.__version__}")


def check_forward(torch):
    print("-- 跑一次真实 FA2 forward --")
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
        res = flash_attn_func(q, k, v)
        o = res[0] if isinstance(res, tuple) else res   # 兼容新版返回 (out, softmax_lse)
        print(f"FA2 forward      : OK, output={tuple(o.shape)}")
    except Exception as e:
        print(f"FAIL: FA2 forward 报错 -> {type(e).__name__}: {e}")
        print("torch/CUDA/flash-attn 版本组合不匹配，需换成匹配的 flash-attn 版本。")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("flash-attn 环境验证")
    print("=" * 60)
    torch = check_torch()
    check_flash_attn()
    check_forward(torch)
    print("=" * 60)
    print("ALL PASS: flash-attn 可用，可直接跑训练")
