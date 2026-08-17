# DDP (DistributedDataParallel) 使用说明

## 概述

InfoSkill 训练代码现已支持 PyTorch DDP 多 GPU 训练。DDP 通过数据并行实现加速：
- 每个 GPU 持有完整模型副本
- 每个 GPU 处理不同的数据批次
- 梯度自动在所有 GPU 间同步并平均
- 理论加速比：N GPU → N 倍吞吐量（实际约 0.85-0.95N）

## 启动命令

### 单 GPU 训练（保持不变）
```bash
python train.py --config configs/alfworld.yaml
```

### 多 GPU DDP 训练

使用 `torchrun` 启动器 + `CUDA_VISIBLE_DEVICES` 环境变量选择 GPU：

#### 2 GPU 示例
```bash
# 使用卡 0 和卡 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml

# 使用卡 2 和卡 3
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml
```

#### 4 GPU 示例
```bash
# 使用所有 4 张卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/alfworld.yaml
```

## 参数说明

- `CUDA_VISIBLE_DEVICES=0,1,2,3`: 指定使用哪些物理 GPU（逗号分隔，无空格）
- `--nproc_per_node=N`: 每个节点启动 N 个进程（通常等于 GPU 数量）
- `--config`: 配置文件路径（与单 GPU 模式相同）

## 工作原理

### 自动检测
代码会自动检测是否在 DDP 模式下运行：
- 检查环境变量 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`
- 如果存在 → 初始化 DDP 进程组
- 如果不存在 → 退化到单 GPU 模式

### Rank 分工
- **Rank 0**（主进程）：
  - 创建日志目录
  - 保存 checkpoints
  - 运行 evaluation
  - 保存训练曲线图和 CSV
  - 执行 Slow Module 更新（skill library maintenance）
  
- **其他 Rank**（工作进程）：
  - 等待 Rank 0 完成 I/O 操作
  - 参与梯度计算和同步
  - 在 barrier 点同步

### 同步点（Barrier）
在以下关键操作后插入 `dist.barrier()` 确保所有进程同步：
1. Checkpoint 保存后
2. Evaluation 完成后
3. Slow Module 更新后

## 内存和性能

### 内存使用
- **每个 GPU 的显存占用与单 GPU 模式相同**
- DDP 不会合并显存（每个 GPU 都是完整模型副本）
- 如果单 GPU 能装下 Qwen2.5-7B，则 2 GPU DDP 也能跑

### 性能预期
| GPU 数量 | 理论加速 | 实际加速（预估） |
|---------|---------|----------------|
| 1       | 1x      | 1x             |
| 2       | 2x      | 1.7-1.9x       |
| 4       | 4x      | 3.4-3.8x       |

实际加速比略低于理论值，原因：
- 梯度同步通信开销
- Rank 0 的 I/O 操作（其他 rank 等待）
- 环境交互（rollout）无法并行（每个进程独立 rollout）

## 测试 DDP 设置

运行测试脚本验证 DDP 配置是否正确：

```bash
# 单 GPU 测试
python test_ddp.py

# 2 GPU 测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_ddp.py
```

应该看到：
- 每个 rank 的设备分配信息
- Broadcast 操作成功
- Barrier 同步成功

## 恢复训练（Resume）

DDP 模式下恢复训练的命令与单 GPU 相同：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld.yaml \
    --resume checkpoints/ep200
```

注意：
- 只有 Rank 0 加载 checkpoint，然后通过 DDP 自动同步模型参数
- 必须使用相同数量的 GPU 恢复训练（例如用 2 GPU 训练的不能用 4 GPU 恢复）

## 常见问题

### 1. NCCL 初始化失败
```
RuntimeError: NCCL error in: ...
```
**解决方法**：
- 检查 CUDA 驱动版本是否支持 NCCL
- 确保所有 GPU 在同一节点且可见
- 尝试设置 `export NCCL_DEBUG=INFO` 查看详细信息

### 2. 挂在 barrier 点
某个进程卡死，无响应。

**可能原因**：
- 某个 rank 遇到异常提前退出
- 代码路径不一致（某个 rank 没到达 barrier）

**解决方法**：
- 检查日志，查看哪个 rank 最后的输出
- 确保所有 rank 走相同的代码路径（if 条件一致）

### 3. Out of Memory
DDP 不会增加显存容量。

**解决方法**：
- 减小 `batch_accumulation` 或 mini_batch_size
- 单 GPU 能跑的配置，DDP 也能跑（显存需求相同）

### 4. 速度没提升
如果 2 GPU 速度与 1 GPU 相近：

**检查**：
- 是否真的在 DDP 模式运行（查看日志："DDP initialized: rank=0, world_size=2"）
- 是否用了 `torchrun` 启动（不要直接 `python train.py`）
- GPU 利用率是否都接近 100%（用 `nvidia-smi` 监控）

## 日志和输出

### 日志文件
只有 Rank 0 写入日志文件：
- `logs/run_YYYYMMDD_HHMMSS/train_progress.log`
- `logs/run_YYYYMMDD_HHMMSS/eval_episodes.log`

### 终端输出
所有 rank 都会输出到终端，建议重定向：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld.yaml \
    2>&1 | tee training.log
```

### Checkpoint
只有 Rank 0 保存 checkpoint：
- `checkpoints/ep200/`
- `checkpoints/final/`

## 配置文件无需修改

`configs/alfworld.yaml` 保持不变：
- `device: "cuda"` 保持不变（代码内部会自动分配 `cuda:0`, `cuda:1` 等）
- 所有超参数（learning_rate, batch_size 等）保持不变
- GPU 选择完全通过命令行控制（CUDA_VISIBLE_DEVICES）

## 示例完整训练流程

```bash
# 1. 检查可用 GPU
nvidia-smi

# 2. 测试 DDP 设置（可选）
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 test_ddp.py

# 3. 启动训练（使用卡 2 和卡 3）
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld.yaml \
    2>&1 | tee logs/training_2gpu.log

# 4. 监控（另一个终端）
watch -n 1 nvidia-smi
```

## 性能监控

训练过程中可以监控：
1. **GPU 利用率**：`nvidia-smi` 应显示所有 GPU 都在 90%+ 利用率
2. **日志文件**：`tail -f logs/run_*/train_progress.log` 查看训练进度
3. **训练速度**：对比单 GPU 的 `steps/sec` 和 `episodes/hour`

预期结果：
- 2 GPU：吞吐量约为单 GPU 的 1.7-1.9 倍
- 4 GPU：吞吐量约为单 GPU 的 3.4-3.8 倍
