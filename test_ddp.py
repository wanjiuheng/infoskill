"""
test_ddp.py

Quick test script to verify DDP setup works correctly.

Usage:
    # Single GPU (should work as before)
    python test_ddp.py

    # Multi-GPU DDP
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_ddp.py
"""

import os
import torch
import torch.distributed as dist
from train import setup_ddp, cleanup_ddp

def main():
    # Test DDP setup
    is_ddp, rank, local_rank, world_size = setup_ddp()

    print(f"Process info: is_ddp={is_ddp}, rank={rank}, local_rank={local_rank}, world_size={world_size}")

    if is_ddp:
        # Test device assignment
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}: assigned to {device}")

        # Test tensor creation and broadcast
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
            print(f"Rank 0: created tensor {tensor}")
        else:
            tensor = torch.zeros(3).to(device)
            print(f"Rank {rank}: initialized tensor {tensor}")

        # Broadcast from rank 0 to all ranks
        dist.broadcast(tensor, src=0)
        print(f"Rank {rank}: after broadcast {tensor}")

        # Test barrier
        print(f"Rank {rank}: before barrier")
        dist.barrier()
        print(f"Rank {rank}: after barrier")

        cleanup_ddp(is_ddp)
        print(f"Rank {rank}: DDP cleanup done")
    else:
        print("Single GPU mode: no DDP operations")

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
