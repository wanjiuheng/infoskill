#!/bin/bash
# scripts/mbs_sweep.sh
#
# 显存 sweep：依次用不同 mini_batch_size 跑一次短训练（configs/alfworld_debug_memsweep.yaml
# 模板 + sed 替换 __MBS__），每次单独起 nvidia-smi 采样，跑完立即杀掉采样器，
# 避免相邻两组互相干扰。跑完把 logs/gpu_sample_mbs*.csv 和
# logs/mbs_sweep_*/train_progress_rank0.log 发回去分析。
#
# 用法：bash scripts/mbs_sweep.sh
# 需要在项目根目录、已 conda activate 对应环境后运行。

set -e
cd "$(dirname "$0")/.."

mkdir -p logs /tmp

MBS_LIST="4 8 16 24 32"
TEMPLATE="configs/alfworld_debug_memsweep.yaml"

for MBS in $MBS_LIST; do
  echo "=== mini_batch_size=$MBS ==="
  TMP_CFG="/tmp/memsweep_${MBS}.yaml"
  sed "s/__MBS__/$MBS/" "$TEMPLATE" > "$TMP_CFG"

  TS=$(date +%Y%m%d_%H%M%S)
  nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv -l 1 > "logs/gpu_sample_mbs${MBS}_${TS}.csv" 2>&1 &
  SAMPLER_PID=$!
  sleep 2

  torchrun --nproc_per_node=2 train.py \
    --config "$TMP_CFG" \
    --run-name "mbs_sweep_${MBS}" \
    > "logs/memsweep_mbs${MBS}_${TS}.log" 2>&1

  kill "$SAMPLER_PID" 2>/dev/null || true
  sleep 3
  echo "=== mini_batch_size=$MBS done ==="
done

echo "全部跑完。把 logs/gpu_sample_mbs*.csv 和 logs/mbs_sweep_*/train_progress_rank0.log 发回去分析。"
