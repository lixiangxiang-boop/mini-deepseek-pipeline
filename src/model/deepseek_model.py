"""
DeepSeek Model — 完整的Decoder-Only Transformer
DeepSeek-V2 §2: 基于MLA + MoE的Transformer架构

架构:
  Embedding -> [DeepSeekLayer x num_layers] -> RMSNorm -> LM Head

特性:
  - 输入/输出Embedding权重共享 (weight tying) — DeepSeek V2/V3均使用
  - Pre-Norm (RMSNorm)
  - Causal (自回归) 注意力掩码
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .rms_norm import RMSNorm
from .deepseek_layer import DeepSeekLayer


class DeepSeekModel(nn.Module):
    """DeepSeek Decoder-Only Transformer

    Args:
        config: DeepSeekConfig 实例
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder Layers
        self.layers = nn.ModuleList([
            DeepSeekLayer(config) for _ in range(config.num_layers)
        ])

        # Final LayerNorm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # KV Cache 存储 (推理时使用)
        self._kv_caches: List[Optional[torch.Tensor]] = [None] * config.num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] padding mask (1=有效, 0=padding)
            use_cache: 是否使用KV缓存

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            new_kv_caches: 每层的KV缓存列表
            expert_counts_per_layer: 每层的专家token计数列表
        """
        batch, seq_len = input_ids.shape

        # Token Embedding
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        # Causal Attention Mask
        if attention_mask is not None:
            # 扩展padding mask: [B, 1, 1, seq_len]
            padding_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
        else:
            padding_mask = None

        # Causal mask: 下三角矩阵
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.float) * -1e9,
            diagonal=1,
        )
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]

        # 合并 causal + padding masks
        if padding_mask is not None:
            combined_mask = causal_mask + padding_mask
        else:
            combined_mask = causal_mask

        # 逐层前向传播
        new_kv_caches = []
        all_expert_counts = []

        for i, layer in enumerate(self.layers):
            kv_cache = self._kv_caches[i] if use_cache else None
            x, new_cache, expert_counts = layer(
                x,
                attention_mask=combined_mask,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
            if use_cache:
                self._kv_caches[i] = new_cache
                new_kv_caches.append(new_cache)
            if expert_counts is not None:
                all_expert_counts.append(expert_counts)

        # Final normalization
        x = self.norm(x)

        return x, new_kv_caches, all_expert_counts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def reset_kv_cache(self):
        """重置KV缓存 (新序列开始时调用)"""
        self._kv_caches = [None] * self.config.num_layers

    def get_kv_cache_memory(self, seq_len: int) -> dict:
        """计算KV缓存的显存占用"""
        layer = self.layers[0]
        return layer.attention.get_cache_memory(seq_len)
