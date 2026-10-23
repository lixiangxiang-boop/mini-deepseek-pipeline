"""
Rotary Position Embedding (RoPE)
DeepSeek-V2 §2.1.1: 对Q/K的部分维度施加RoPE

DeepSeek的MLA中，RoPE只作用于Q/K的前 qk_rope_dim 个维度。
剩余的 (d_head - qk_rope_dim) 维度不使用位置编码，由潜在空间直接学习。
这种设计是MLA能够压缩KV的关键。

Implementation based on:
  RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """预计算旋转位置编码的sin/cos表

    Args:
        dim: RoPE作用的维度数 (qk_rope_dim)
        max_seq_len: 最大序列长度
        theta: 频率基数 (DeepSeek-V2默认10000.0)
    """

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率: 1 / (theta^(2i/d))
        # dim个维度，相邻两个一组 (cos, sin)，所以需要 dim/2 个频率
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # 位置索引: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()

        # 外积: [max_seq_len, dim/2]
        freqs = torch.outer(positions, freqs)

        # 预计算cos和sin表: [max_seq_len, dim]
        # 每组cos/sin重复一次以覆盖相邻两个维度
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]

        # 注册为buffer (不参与训练，但随模型移动device)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回cos和sin表，截取到当前序列长度

        Args:
            x: 用于确定device/dtype的参考张量
            seq_len: 需要截取的长度

        Returns:
            (cos, sin): 各为 [seq_len, dim]
        """
        return (
            self.cos_cached[:seq_len].to(x.device).to(x.dtype),
            self.sin_cached[:seq_len].to(x.device).to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量的后半维度取反后与前半互换: [x1, x2] -> [-x2, x1]

    这是RoPE旋转操作的核心步骤。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对Q和K施加旋转位置编码

    Args:
        q: query张量 [batch, seq_len, ...] 或 [batch, num_heads, seq_len, d_head]
        k: key张量 (同shape)
        cos: [seq_len, rope_dim]
        sin: [seq_len, rope_dim]

    Returns:
        (q_rotated, k_rotated)
    """
    # cos/sin调整为广播维度: [1, 1, seq_len, rope_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated
