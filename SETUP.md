# InfoSkill 部署与训练环境配置

本仓库是 InfoSkill（ALFWorld 文字版 + Qwen2.5-7B-Instruct LoRA 策略）的训练代码。本文件给出从零拉取代码、配置环境、启动训练的完整命令流程，面向训练机（Linux + NVIDIA H20 + CUDA 12.4 驱动）。

## 1. 前置条件

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（训练机） |
| GPU | NVIDIA H20（CUDA 12.4 驱动，Driver 支持 cu124 构建） |
| conda | 已安装 |
| Python | 3.10 / 3.11（torch 2.6.0 cu124 构建需要；Python ≥3.12 无 cu124 兼容构建） |
| ALFWorld 数据 | 含 `json_2.1.1/{train,valid_seen,valid_unseen}` 与 `logic/alfred.pddl`、`logic/alfred.twl2` |
| LLM 权重 | 原版 Qwen2.5-7B-Instruct 或 SFT 后 checkpoint（见 §4） |

## 2. 拉取代码

```bash
git clone https://github.com/wanjiuheng/infoskill.git
cd infoskill
git checkout option1-text-alfredtw        # 训练分支，当前 HEAD 为 5ee449d
git log --oneline -1                       # 确认拉到最新
```

已有旧代码想更新到最新时：

```bash
git fetch origin
git reset --hard origin/option1-text-alfredtw   # ⚠️ 丢弃本地未提交改动，先 git stash
```

## 3. 配置 Python 环境

```bash
conda create -n infoskill python=3.11 -y
conda activate infoskill

# 关键：torch 必须【先于】requirements.txt 单独安装。
# torch 故意不写进 requirements.txt：需要匹配驱动的 CUDA 构建。
# PyPI 的 torch==2.6.0 本身就是 cu124 构建，无需 pytorch.org 源
# （download.pytorch.org 不可达时尤其有用）。
pip install torch==2.6.0

# 其余依赖（transformers / accelerate / peft / alfworld / textworld 等）
pip install -r requirements.txt
```

> **textworld 版本提醒**：`requirements.txt` 声明 `textworld[pddl]>=1.6.1`，但 eval 无放回覆盖逻辑是依据 textworld **1.5.0** 的 `shuffled_cycle` 行为实现的（第一轮按 seed 洗牌后的顺序逐个 yield、不重复，耗尽后才重洗；eval 固定 `EVAL_ORDER_SEED` seed 一次后连续 `reset()`）。装完后执行 `pip show textworld` 确认实际版本；若为 1.6.x，需确认 `shuffled_cycle` 行为未变，否则 eval 无放回覆盖会失效。

## 4. 数据与模型路径

### 4.1 ALFWorld 数据（`$ALFWORLD_DATA`）

`envs/alfworld_base_config.yaml` 用 `$ALFWORLD_DATA` 占位数据根目录，必须包含：

```bash
export ALFWORLD_DATA=/你的/alfworld/data/路径   # 建议写入 ~/.bashrc，否则每次 SSH 都要 export
ls $ALFWORLD_DATA/json_2.1.1/train              # 训练集（3553 个任务目录）
ls $ALFWORLD_DATA/json_2.1.1/valid_seen         # 分布内 eval（140 个游戏）
ls $ALFWORLD_DATA/json_2.1.1/valid_unseen       # 分布外 eval（134 个游戏）
ls $ALFWORLD_DATA/logic/alfred.pddl             # PDDL 领域文件
ls $ALFWORLD_DATA/logic/alfred.twl2             # TextWorld 语法文件
```

### 4.2 LLM backbone

`configs/alfworld.yaml` 中 `model.backbone` 默认指向原版 Qwen：

```yaml
model:
  backbone: "/models/wanjh2/models/Qwen/Qwen2.5-7B-Instruct"
```

> **⚠️ 训练前必须确认**：本实验实际使用的是 **SFT 后 checkpoint**（backbone=SFT checkpoint-140），而非原版。把 `model.backbone` 改成训练机上 SFT 产物的实际路径，否则会用原版模型从头训，行为与预期不符。

### 4.3 SFT checkpoint 与续训

- 训练产物（checkpoint、日志、曲线）默认写在仓库内 `checkpoints/`、`logs/`（被 `.gitignore` 忽略，不会入库）。
- 从 checkpoint 续训：`--resume checkpoints/<run 目录>`。

## 5. 环境自检

```bash
# 依赖与 CUDA
python -c "import torch, transformers, peft, textworld, alfworld; print('deps ok'); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0))"

# 数据与模型可达
python -c "import os; print('ALFWORLD_DATA =', os.environ['ALFWORLD_DATA'])"
ls $ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen} >/dev/null && echo "json data ok"

# textworld 实际版本
pip show textworld
```

## 6. 启动训练

```bash
# 单卡
python train.py --config configs/alfworld.yaml

# 单机多卡（2 卡示例；4 卡改 nproc_per_node=4）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml

# 从 checkpoint 续训
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml --resume checkpoints/<run 目录>
```

`train.py` 的 `--run-name` 默认按 `lr_bs_maxnewtok_maxsteps` 自动命名 run 目录（`configs/alfworld.yaml` 中 `training.learning_rate`、`training.mini_batch_size`、`rollout.max_new_tokens`、`rollout.max_steps`），无需手动重复这些参数。

## 7. 关键配置速览（`configs/alfworld.yaml`）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `training.learning_rate` | `1.0e-6` | 论文值；曾用 `1e-4` 疑似致训练崩溃。注意 PyYAML 会把 `1e-4` 解析成字符串，必须写 `1.0e-6` 风格 |
| `rollout.group_size` | 4 | GRPO 组内并行 episode 数 |
| `rollout.max_steps` | 30 | 每 episode 步数上限 |
| `rollout.max_new_tokens` | 512 | 每步 LLM 生成预算 |
| `training.mini_batch_size` | 10 | `_fast_update()` 重算 log_prob 的 mini-batch |
| `training.eval_freq` | 200 | 每 N episode 跑一次 eval（必须是 group_size 的倍数） |
| `training.eval_episodes` | 140 | eval 覆盖 game 池（无放回全量，见 §3 提醒） |
| `training.slow_update_interval` | 200 | Slow Module 技能库维护间隔（global_step） |
| `skill_library.success_reward_threshold` | 7.0 | reward ≥ 7.0 的成功轨迹才收集去生成新技能（对应 ≤30 步） |

## 8. 常见问题

- **OOM（训练崩溃）**：rollout 阶段必须全程 `torch.no_grad()`，梯度只在 `_fast_update` 单独 forward+backward 时构建。若仍 OOM，检查是否误在 rollout 里开了梯度。
- **eval 后 rollout 输出乱码（一堆 0）**：已知现象，触发点是第一次 eval 的 `model.eval()/train()` 切换与全局 gradient checkpointing、sampling generate 的交互，与权重无关。候选修复：rollout 的 generate 用 `model.eval()` 跑。详见诊断记录。
