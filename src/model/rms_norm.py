"""
RMSNorm — Root Mean Square Normalization
DeepSeek-V2 §2.1.1: 使用RMSNorm替代LayerNorm

RMSNorm 比 LayerNorm 更高效:
  - 只计算均方根(RMS)，去掉了均值中心化
  - 减少一次规约操作，训练/推理更快
  - 效果与LayerNorm相当

Formula: y = x * weight / sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Args:
        hidden_size: 隐藏层维度
        eps: 数值稳定性常数 (DeepSeek使用1e-6)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size) or (batch, hidden_size)
        dtype = x.dtype

        # 计算RMS: sqrt(mean(x^2))
        # 使用float32计算以获得更高精度
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化 + 可学习缩放
        x = x.float() / rms
        return (x * self.weight).to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"
