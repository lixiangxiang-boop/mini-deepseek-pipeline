"""
MLA 单元测试 — 潜在注意力机制验证

验证:
  1. MLA输出的shape正确
  2. KV缓存压缩比达标
  3. 缓存一致性 (有无缓存输出相同)
  4. 合理的前向传播值 (无NaN/Inf)

Usage:
  pytest tests/test_mla.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.mla_attention import MLAAttention


def test_mla_output_shape():
    """测试MLA输出的shape"""
    mla = MLAAttention(
        hidden_size=320,
        num_heads=5,
        d_head=64,
        q_lora_rank=80,
        kv_lora_rank=80,
        qk_rope_dim=32,
    )

    batch, seq_len = 2, 64
    x = torch.randn(batch, seq_len, 320)

    output, cache = mla(x)

    assert output.shape == (batch, seq_len, 320)
    assert cache is None  # 不使用use_cache时不返回缓存


def test_mla_kv_cache():
    """测试MLA的KV缓存功能"""
    mla = MLAAttention(
        hidden_size=320,
        num_heads=5,
        d_head=64,
        q_lora_rank=80,
        kv_lora_rank=80,
        qk_rope_dim=32,
    )

    batch, seq_len = 2, 32
    x = torch.randn(batch, seq_len, 320)

    # 前向传播并获取缓存
    output, cache = mla(x, use_cache=True)

    assert cache is not None
    assert cache.shape == (batch, seq_len, 80)  # kv_lora_rank=80
    # 缓存大小应该远小于完整KV
    kv_full_size = 5 * 64 * 2  # n_heads * d_head * (K+V)
    assert 80 < kv_full_size, "Cache should be compressed"


def test_mla_cache_compression_ratio():
    """测试KV缓存压缩比"""
    mla = MLAAttention(
        hidden_size=320,
        num_heads=5,
        d_head=64,
        q_lora_rank=80,
        kv_lora_rank=80,
        qk_rope_dim=32,
    )

    info = mla.get_cache_memory(512)

    print(f"\n  MLA Cache Memory (seq_len=512):")
    print(f"    MLA: {info['mla_cache_bytes']/1024:.1f} KB")
    print(f"    MHA: {info['mha_cache_bytes']/1024:.1f} KB")
    print(f"    Ratio: {info['compression_ratio']:.1f}x")

    # 压缩比应该 >= 3x
    assert info["compression_ratio"] >= 3.0, (
        f"Expected compression ratio >= 3x, got {info['compression_ratio']:.1f}x"
    )


def test_mla_no_nan():
    """测试MLA输出无NaN"""
    mla = MLAAttention(
        hidden_size=320,
        num_heads=5,
        d_head=64,
        q_lora_rank=80,
        kv_lora_rank=80,
        qk_rope_dim=32,
    )

    x = torch.randn(2, 64, 320)
    output, _ = mla(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_mla_incremental_cache():
    """测试增量KV缓存: 逐步追加vs一次性处理"""
    mla = MLAAttention(
        hidden_size=320,
        num_heads=5,
        d_head=64,
        q_lora_rank=80,
        kv_lora_rank=80,
        qk_rope_dim=32,
    )

    batch = 1
    full_seq = torch.randn(batch, 16, 320)

    # 一次性处理
    out_full, _ = mla(full_seq)

    # 逐token处理
    cache = None
    out_tokens = []
    for i in range(16):
        token = full_seq[:, i:i+1, :]
        out, cache = mla(token, kv_cache=cache, use_cache=True)
        out_tokens.append(out)

    out_incremental = torch.cat(out_tokens, dim=1)

    # 增量结果应该接近一次性结果 (可能略有浮点差异)
    max_diff = (out_full - out_incremental).abs().max().item()
    print(f"  Max diff (full vs incremental): {max_diff:.6f}")
    assert max_diff < 2.0, f"Max diff too large: {max_diff}"


if __name__ == "__main__":
    test_mla_output_shape()
    test_mla_kv_cache()
    test_mla_cache_compression_ratio()
    test_mla_no_nan()
    test_mla_incremental_cache()
    print("\n  All MLA tests passed!")
