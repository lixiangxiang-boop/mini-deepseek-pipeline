"""
Latent KV Cache — MLA推理加速的核心

DeepSeek-V2的MLA最大的优势是大幅减少KV缓存的显存占用。
标准MHA需要缓存完整的K和V，而MLA只需缓存低维的潜在向量c_kv。

显存对比 (以本项目的配置为例):
  MHA:  seq_len × num_heads × d_head × 2 (K+V) × 2 bytes (fp16)
       = 512 × 5 × 64 × 2 × 2 = 655,360 bytes ≈ 640 KB (每层)
  MLA:  seq_len × kv_lora_rank × 2 bytes
       = 512 × 80 × 2 = 81,920 bytes ≈ 80 KB (每层)
  压缩比: 8x

这是MLA的核心价值: 用更少的显存支持更长的上下文。
"""

import torch
from typing import Optional, List


class LatentKVCache:
    """MLA潜在向量KV缓存

    每层缓存一个 [batch, seq_len, kv_lora_rank] 的张量。
    新的token到来时追加到缓存中。

    Args:
        num_layers: Transformer层数
        kv_lora_rank: 潜在空间维度
    """

    def __init__(self, num_layers: int, kv_lora_rank: int):
        self.num_layers = num_layers
        self.kv_lora_rank = kv_lora_rank
        self.caches: List[Optional[torch.Tensor]] = [None] * num_layers

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取某层的缓存"""
        return self.caches[layer_idx]

    def append(
        self,
        layer_idx: int,
        new_latent_kv: torch.Tensor,
    ) -> torch.Tensor:
        """追加新的潜在KV向量到缓存

        Args:
            layer_idx: 层索引
            new_latent_kv: [batch, new_seq_len, kv_lora_rank]

        Returns:
            更新后的完整缓存 [batch, total_seq_len, kv_lora_rank]
        """
        if self.caches[layer_idx] is None:
            self.caches[layer_idx] = new_latent_kv
        else:
            self.caches[layer_idx] = torch.cat(
                [self.caches[layer_idx], new_latent_kv], dim=1
            )
        return self.caches[layer_idx]

    def reset(self):
        """清空所有缓存 (新序列开始前调用)"""
        self.caches = [None] * self.num_layers

    def get_memory_usage(self, dtype_size: int = 2) -> int:
        """计算当前缓存的总显存占用 (bytes)"""
        total = 0
        for cache in self.caches:
            if cache is not None:
                batch, seq_len, _ = cache.shape
                total += batch * seq_len * self.kv_lora_rank * dtype_size
        return total

    def get_equivalent_mha_memory(
        self,
        seq_len: int,
        num_heads: int,
        d_head: int,
        dtype_size: int = 2,
    ) -> int:
        """计算等价的MHA KV缓存显存"""
        # K: seq_len × num_heads × d_head, V: 同上
        return seq_len * num_heads * d_head * 2 * dtype_size

    def print_comparison(self, seq_len: int, num_heads: int, d_head: int):
        """打印MLA vs MHA的显存对比"""
        mla_mem = self.get_memory_usage() or (1 * seq_len * self.kv_lora_rank * 2)
        mha_mem = self.get_equivalent_mha_memory(seq_len, num_heads, d_head)

        print(f"\n  KV Cache Memory Comparison (seq_len={seq_len}, per layer):")
        print(f"    MHA:  {mha_mem/1024:.1f} KB (K+V, fp16)")
        print(f"    MLA:  {mla_mem/1024:.1f} KB (latent c_kv, fp16)")
        print(f"    Compression: {mha_mem/mla_mem:.1f}x")
        print(f"    Total (all layers):")
        print(f"      MHA: {mha_mem * self.num_layers / 1024:.1f} KB")
        print(f"      MLA: {mla_mem * self.num_layers / 1024:.1f} KB")
