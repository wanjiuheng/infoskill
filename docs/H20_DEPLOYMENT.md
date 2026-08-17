# H20 服务器部署指南

## 环境信息对比

### 当前 H20 环境
- **CUDA Version**: 13.1
- **GPU**: H20 (可能是 H100 系列)

### 项目原始配置
- **推荐 CUDA**: 12.4 (根据 requirements.txt 注释)
- **推荐 Python**: 3.10 或 3.11
- **PyTorch**: 2.6.0 (cu124 build)

## 兼容性分析

### ⚠️ 关键问题：CUDA 13.1 vs 项目需求

**CUDA 13.1 的情况**：
- CUDA 13.x 是较新的版本（2024 年后发布）
- PyTorch 2.6.0 官方只支持到 CUDA 12.4
- **PyTorch 没有预编译的 cu131 wheel**

**向后兼容性**：
- ✅ **好消息**：PyTorch 的 CUDA 12.x build 通常可以在 CUDA 13.x 上运行
- ✅ NVIDIA 保持向后兼容（新驱动支持旧 CUDA runtime）
- ⚠️ 但可能有性能损失或未充分利用新硬件特性

## 推荐部署方案

### 方案 1：使用 CUDA 12.4 build（推荐）

**优点**：
- 稳定，经过充分测试
- 与项目原始配置一致
- 不需要修改代码

**步骤**：
```bash
# 1. 创建 conda 环境
conda create -n infoskill python=3.11 -y
conda activate infoskill

# 2. 安装 PyTorch 2.6.0 (CUDA 12.4 build)
pip install torch==2.6.0 torchvision torchaudio

# 3. 验证 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 4. 如果上面显示 CUDA available: False，可能需要：
# - 检查 nvidia-smi 确认驱动正常
# - 设置 LD_LIBRARY_PATH（如果 CUDA 库路径不在默认位置）

# 5. 安装其他依赖
cd /path/to/infoskill
pip install -r requirements.txt

# 6. 测试 DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_ddp.py
```

### 方案 2：使用 PyTorch Nightly（CUDA 12.6 或更新）

如果方案 1 出现兼容性问题，可以尝试 nightly build：

```bash
# 安装 PyTorch nightly (可能支持更新的 CUDA)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**风险**：
- Nightly 版本不稳定，可能有 bug
- API 可能与 2.6.0 有差异

### 方案 3：降级 CUDA 驱动（不推荐）

如果服务器允许，可以安装 CUDA 12.4 toolkit，但这通常：
- 需要 root 权限
- 可能影响其他用户
- 不推荐在共享服务器上操作

## 部署检查清单

### 1. 环境检查
```bash
# 查看 CUDA 驱动版本
nvidia-smi

# 查看 GPU 信息
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 查看可用 GPU
nvidia-smi -L
```

### 2. PyTorch 安装验证
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version (torch build): {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### 3. DDP 功能测试
```bash
# 单 GPU 测试
python test_ddp.py

# 2 GPU 测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_ddp.py

# 4 GPU 测试
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test_ddp.py
```

### 4. 小规模训练测试
```bash
# 修改配置文件，减小训练量进行快速测试
cp configs/alfworld.yaml configs/alfworld_test.yaml

# 编辑 alfworld_test.yaml，修改：
# training:
#   num_episodes: 40  # 原来是 14212
#   save_freq: 8
#   eval_freq: 8
#   eval_episodes: 8

# 运行测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld_test.yaml
```

## 常见问题排查

### 问题 1: torch.cuda.is_available() 返回 False

**可能原因**：
1. PyTorch 的 CUDA runtime 与驱动不兼容
2. CUDA 库路径未设置

**解决方法**：
```bash
# 检查驱动支持的 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 如果是 CUDA 13.1，尝试设置环境变量
export CUDA_HOME=/usr/local/cuda-13.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# 重新测试
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 2: NCCL 错误

```
RuntimeError: NCCL error in: ...
```

**解决方法**：
```bash
# 设置 NCCL 调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 如果是 NCCL 版本不兼容，尝试
export NCCL_P2P_DISABLE=1  # 禁用 P2P 通信（可能降低性能）

# 重新运行
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml
```

### 问题 3: OOM (Out of Memory)

**原因**：
- H20 显存可能与原环境不同

**解决方法**：
```yaml
# 在 configs/alfworld.yaml 中调整：
rollout:
  max_new_tokens: 256  # 从 512 降低到 256
  
training:
  batch_accumulation: 2  # 从 1 增加到 2（减小有效 batch size）
```

### 问题 4: 训练速度异常慢

**排查步骤**：
1. 检查 GPU 利用率：`watch -n 1 nvidia-smi`
2. 检查是否在 DDP 模式：日志中应有 "DDP initialized: rank=0, world_size=2"
3. 检查是否有 I/O 瓶颈：`iotop -o`
4. 检查 CPU 利用率：`htop`

## 性能基准参考

### 单 GPU 基准（假设）
- **吞吐量**：~10-15 episodes/hour
- **GPU 利用率**：85-95%
- **显存占用**：~20-25 GB (Qwen2.5-7B + LoRA + 辅助模块)

### 2 GPU DDP 预期
- **吞吐量**：~17-26 episodes/hour (1.7-1.9x)
- **GPU 利用率**：80-90% (略低于单 GPU，因为同步开销)
- **显存占用**：每个 GPU ~20-25 GB (与单 GPU 相同)

### 4 GPU DDP 预期
- **吞吐量**：~34-51 episodes/hour (3.4-3.8x)
- **GPU 利用率**：75-85%
- **显存占用**：每个 GPU ~20-25 GB

## 推荐训练命令

```bash
# 1. 激活环境
conda activate infoskill
cd /path/to/infoskill

# 2. 查看可用 GPU
nvidia-smi

# 3. 根据空闲 GPU 选择训练配置

# 使用卡 0 和 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld.yaml \
    2>&1 | tee logs/train_gpu01_$(date +%Y%m%d_%H%M%S).log

# 使用卡 2 和 3
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --config configs/alfworld.yaml \
    2>&1 | tee logs/train_gpu23_$(date +%Y%m%d_%H%M%S).log

# 4. 在另一个终端监控
watch -n 1 nvidia-smi
tail -f logs/run_*/train_progress.log
```

## 总结

1. **CUDA 13.1 兼容性**：PyTorch 2.6.0 (cu124) 应该能在 CUDA 13.1 上运行，但可能有小概率出现兼容性问题
2. **推荐方案**：先尝试安装 PyTorch 2.6.0 官方 wheel，如果 `torch.cuda.is_available()` 返回 True 就没问题
3. **验证步骤**：
   - 安装 PyTorch → 验证 CUDA 可用 → 测试 DDP → 小规模训练测试 → 全量训练
4. **备用方案**：如果 cu124 build 有问题，可以尝试 PyTorch nightly 或联系服务器管理员安装 CUDA 12.4 toolkit

**需要修改代码吗？**
- ✅ **不需要**！代码已经支持 DDP，只需要正确安装环境即可
- DDP 实现已经完成，直接用 `torchrun` 启动即可

有任何问题随时反馈，我可以帮助调整！
