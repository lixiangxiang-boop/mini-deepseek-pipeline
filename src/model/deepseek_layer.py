"""
DeepSeek Decoder Layer — 单层Transformer Decoder
组合 MLA Attention + MoE FFN，使用 Pre-Norm 架构

架构 (与DeepSeek一致):
  x -> RMSNorm -> MLA (with residual) -> RMSNorm -> MoE (with residual)

这是Transformer decoder的标准Pre-Norm布局，DeepSeek直接使用。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .rms_norm import RMSNorm
from .mla_attention import MLAAttention
from .deepseek_moe import DeepSeekMoE


class DeepSeekLayer(nn.Module):
    """单个Decoder层: MLA + MoE

    可通过config开关切换:
      - use_mla=False -> 标准Multi-Head Attention
      - use_moe=False -> 标准FFN
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Pre-norm for attention
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # MLA Attention (也可切换为标准MHA — 这里我们只实现MLA)
        self.attention = MLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            d_head=config.d_head,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_dim=config.qk_rope_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # MoE FFN (也可切换为标准FFN)
        if config.use_moe:
            self.moe = DeepSeekMoE(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_intermediate_size,
                num_experts=config.num_experts,
                num_shared_experts=config.num_shared_experts,
                top_k=config.top_k,
                dropout=config.dropout,
            )
            self.use_moe = True
        else:
            # 退化为标准FFN (与一个共享专家的MoE等价)
            from .deepseek_moe import ExpertFFN
            self.ffn = ExpertFFN(
                config.hidden_size,
                config.hidden_size * 4,  # 标准FFN expansion ratio = 4
                config.dropout,
            )
            self.use_moe = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: causal mask
            kv_cache: 之前的KV缓存
            use_cache: 是否返回新的KV缓存

        Returns:
            (output, new_kv_cache, expert_counts)
        """
        # ===== Attention sub-layer (Pre-Norm) =====
        normed = self.attn_norm(x)
        attn_out, new_cache = self.attention(
            normed,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out  # Residual

        # ===== FFN sub-layer (Pre-Norm) =====
        normed = self.ffn_norm(x)
        expert_counts = None
        if self.use_moe:
            ffn_out, expert_counts = self.moe(normed)
        else:
            ffn_out = self.ffn(normed)
        x = x + ffn_out  # Residual

        return x, new_cache, expert_counts
