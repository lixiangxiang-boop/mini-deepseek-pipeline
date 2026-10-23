"""
Multi-head Latent Attention (MLA) — DeepSeek-V2 §2.1.2

MLA 是 DeepSeek-V2 最核心的创新，通过低秩压缩大幅减少 KV Cache 的内存占用。

原理:
  标准 MHA: 每个 token 缓存完整的 K 和 V，显存 = 2 × n_heads × d_head × seq_len
  MLA: 将 K 和 V 压缩到一个低维潜在向量 c_kv，只缓存 c_kv
       显存 = kv_lora_rank × seq_len

  压缩比 = n_heads × d_head × 2 / kv_lora_rank
  例: 5×64×2 / 80 = 8x 压缩

前向传播流程:
  1. x -> Linear(hidden, q_lora_rank)    -> latent_q
  2. x -> Linear(hidden, kv_lora_rank)   -> latent_kv (潜在KV向量 = 这就是我们要缓存的!)
  3. latent_q  -> Linear(q_lora_rank, n_heads*d_head)   -> Q
  4. latent_kv -> Linear(kv_lora_rank, n_heads*d_head)  -> K
  5. latent_kv -> Linear(kv_lora_rank, n_heads*d_head)  -> V
  6. Q的前qk_rope_dim维 + K的前qk_rope_dim维 施加RoPE
  7. 标准 Scaled Dot-Product Attention
  8. 输出投影

推理时:
  - 只缓存 latent_kv (80维)，而不是完整 K (320维) 和 V (320维)
  - 每层节省 8x KV Cache 显存

简化说明:
  论文中 V2 使用双级压缩 (d_c + d_h)。本项目简化为单级压缩，
  将 K 和 V 压缩到同一个潜在空间中。概念完全一致，实现更易理解。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rotary_embedding import RotaryEmbedding, apply_rotary_emb, rotate_half


class MLAAttention(nn.Module):
    """Multi-head Latent Attention

    Args:
        hidden_size: 输入/输出维度 (d_model)
        num_heads: 注意力头数
        d_head: 每个头的维度
        q_lora_rank: Q的低秩压缩维度 (d_q_compress)
        kv_lora_rank: KV的潜在空间维度 (d_kv_compress)
        qk_rope_dim: Q/K中施加RoPE的维度数
        max_seq_len: 最大序列长度
        rope_theta: RoPE频率基数
        dropout: attention dropout
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        d_head: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_dim: int,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = d_head
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_dim = qk_rope_dim
        self.scale = math.sqrt(d_head)

        # ===== 低秩压缩投影 =====
        # Q压缩: hidden -> q_lora_rank (W^Q_A in paper)
        self.W_q_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
        # Q扩展: q_lora_rank -> num_heads * d_head (W^Q_B in paper)
        self.W_q_b = nn.Linear(q_lora_rank, num_heads * d_head, bias=False)

        # KV压缩: hidden -> kv_lora_rank (W^{KV}_A)
        # 将K和V压缩到同一个潜在空间 — 这是与V2的关键简化
        self.W_kv_a = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        # K扩展: kv_lora_rank -> num_heads * d_head (W^K_B)
        self.W_k_b = nn.Linear(kv_lora_rank, num_heads * d_head, bias=False)
        # V扩展: kv_lora_rank -> num_heads * d_head (W^V_B)
        self.W_v_b = nn.Linear(kv_lora_rank, num_heads * d_head, bias=False)

        # ===== 输出投影 =====
        self.W_o = nn.Linear(num_heads * d_head, hidden_size, bias=False)

        # ===== RoPE (作用于Q/K的前qk_rope_dim维) =====
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] or None
            kv_cache: 之前的潜在KV缓存 [batch, prev_len, kv_lora_rank] or None
            use_cache: 是否返回新的KV缓存

        Returns:
            (output, new_kv_cache)
        """
        batch, seq_len, _ = x.shape

        # ===== Step 1: 计算潜在向量 =====
        latent_q = self.W_q_a(x)    # [B, L, q_lora_rank]
        latent_kv = self.W_kv_a(x)  # [B, L, kv_lora_rank] ★ 这是我们要缓存的!

        # 如果有之前的KV缓存，拼接到前面
        if kv_cache is not None:
            latent_kv = torch.cat([kv_cache, latent_kv], dim=1)  # [B, prev+L, kv_lora_rank]
            total_len = latent_kv.shape[1]
        else:
            total_len = seq_len

        # ===== Step 2: 从潜在空间扩展出Q, K, V =====
        q = self.W_q_b(latent_q)  # [B, L, n_heads * d_head]
        q = q.view(batch, seq_len, self.num_heads, self.d_head)

        k = self.W_k_b(latent_kv)  # [B, total_len, n_heads * d_head]
        k = k.view(batch, total_len, self.num_heads, self.d_head)

        v = self.W_v_b(latent_kv)  # [B, total_len, n_heads * d_head]
        v = v.view(batch, total_len, self.num_heads, self.d_head)

        # ===== Step 3: Transpose to [B, n_heads, L, d_head] =====
        q = q.transpose(1, 2)  # [B, n_heads, L, d_head]
        k = k.transpose(1, 2)  # [B, n_heads, total_len, d_head]
        v = v.transpose(1, 2)  # [B, n_heads, total_len, d_head]

        # ===== Step 4: RoPE — 只作用于前qk_rope_dim维 =====
        # Q和K可能有不同的序列长度 (增量解码时: q=L, k=total_len)
        # 需要分别获取对应位置的RoPE编码
        cos_k, sin_k = self.rotary_emb(x, total_len)  # for K: 所有历史位置
        cos_q, sin_q = cos_k[-seq_len:], sin_k[-seq_len:]  # for Q: 仅当前token位置

        # 分别对Q和K的RoPE部分施加旋转编码
        q_rope_dim = q[:, :, :, :self.qk_rope_dim]
        k_rope_dim = k[:, :, :, :self.qk_rope_dim]

        # 使用统一的apply_rotary_emb对Q和K分别处理
        # 注意: apply_rotary_emb期望q和k长度一致，这里分开调用
        cos_q_4d = cos_q.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rope_dim]
        sin_q_4d = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k_4d = cos_k.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, rope_dim]
        sin_k_4d = sin_k.unsqueeze(0).unsqueeze(0)

        q_rope = (q_rope_dim * cos_q_4d) + (rotate_half(q_rope_dim) * sin_q_4d)
        k_rope = (k_rope_dim * cos_k_4d) + (rotate_half(k_rope_dim) * sin_k_4d)

        # 拼接RoPE部分和非RoPE部分
        q = torch.cat([q_rope, q[:, :, :, self.qk_rope_dim:]], dim=-1)
        k = torch.cat([k_rope, k[:, :, :, self.qk_rope_dim:]], dim=-1)

        # ===== Step 5: Scaled Dot-Product Attention =====
        # scores: [B, n_heads, L, total_len]
        scores = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # output: [B, n_heads, L, d_head]
        output = attn_weights @ v

        # ===== Step 6: Concatenate heads + output projection =====
        output = output.transpose(1, 2).contiguous()  # [B, L, n_heads, d_head]
        output = output.view(batch, seq_len, self.num_heads * self.d_head)
        output = self.W_o(output)

        # ===== Step 7: Return KV cache (只缓存潜在向量!) =====
        new_cache = latent_kv[:, -seq_len:] if use_cache else None

        return output, new_cache

    def get_cache_memory(self, seq_len: int) -> dict:
        """计算KV缓存的显存占用 (用于对比MLA vs 标准MHA)"""
        # MLA: 只存 kv_lora_rank 维
        mla_bytes = seq_len * self.kv_lora_rank * 2  # fp16 = 2 bytes
        # 标准MHA: 存完整的 K + V
        mha_bytes = seq_len * self.num_heads * self.d_head * 2 * 2  # K + V, fp16
        compression_ratio = mha_bytes / mla_bytes if mla_bytes > 0 else 1
        return {
            "mla_cache_bytes": mla_bytes,
            "mha_cache_bytes": mha_bytes,
            "compression_ratio": compression_ratio,
            "kv_lora_rank": self.kv_lora_rank,
            "full_kv_dim": self.num_heads * self.d_head,
        }
