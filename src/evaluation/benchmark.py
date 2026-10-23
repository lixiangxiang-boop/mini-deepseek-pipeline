"""
性能基准测试 — 吞吐量和显存占用

测试指标:
  - tokens/sec (吞吐量)
  - 峰值VRAM使用量
  - KV缓存显存占用
  - MLA vs MHA 对比

Usage:
  python -m src.evaluation.benchmark --checkpoint checkpoints/final.pt
"""

import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.inference.generate import generate_text


def benchmark(
    checkpoint_path: str,
    batch_sizes: list = None,
    seq_lengths: list = None,
    device: str = "cuda",
):
    """运行性能基准测试

    Args:
        checkpoint_path: 模型checkpoint
        batch_sizes: 测试的batch sizes
        seq_lengths: 测试的prompt长度
        device: 推理设备
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256]

    print("=" * 60)
    print("  Performance Benchmark")
    print("=" * 60)

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", DeepSeekConfig())
    model = DeepSeekForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    model.print_model_summary()

    # 显存占用
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU Memory:")
        print(f"    Total VRAM:   {total_vram:.1f} GB")
        print(f"    Reserved:     {reserved:.3f} GB")
        print(f"    Allocated:    {allocated:.3f} GB")

    # KV缓存分析
    print(f"\n  MLA KV Cache Analysis:")
    kv_cache_info = model.model.get_kv_cache_memory(512)
    for k, v in kv_cache_info.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v}")

    # 吞吐量测试
    print(f"\n  Throughput Test (generate 50 tokens):")
    print(f"    {'Batch':>6} {'Prompt':>7} {'Tokens/s':>10} {'Latency(ms)':>12}")

    for bs in batch_sizes:
        for sl in seq_lengths:
            # 创建随机输入
            input_ids = torch.randint(4, 1000, (bs, sl), device=device)

            # 预热
            _ = generate_text(model, input_ids, max_new_tokens=10, use_cache=True)

            # 计时
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()

            output = generate_text(
                model,
                input_ids,
                max_new_tokens=50,
                temperature=0.0,  # 贪婪解码 (确定性)
                use_cache=True,
            )

            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - t0

            new_tokens = output.shape[1] - sl
            tokens_per_sec = (new_tokens * bs) / elapsed
            latency_per_token = (elapsed / new_tokens) * 1000

            print(f"    {bs:>6} {sl:>7} {tokens_per_sec:>10.1f} {latency_per_token:>12.1f}")

    # Peak memory
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n  Peak GPU Memory: {peak:.3f} GB")
        print(f"  Remaining headroom: {total_vram - peak:.3f} GB")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mini-DeepSeek")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    benchmark(checkpoint_path=args.checkpoint, device=args.device)
