"""
DeepSeekConfig — 模型架构配置
对应 DeepSeek-V2/V3 论文中的超参数设计

Reference:
  - V2 §2.1: Multi-Head Latent Attention (MLA)
  - V2 §2.2: DeepSeekMoE with fine-grained expert segmentation
  - V3 §2.4: Multi-Token Prediction (MTP)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepSeekConfig:
    """DeepSeek风格的Transformer配置

    所有参数都有合理的默认值，适合27M参数级别的小模型学习。
    通过 toggle 开关可以单独启用/禁用每个创新组件，方便做消融实验。
    """

    # ========== Vocabulary ==========
    vocab_size: int = 16384
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ========== Model Dimensions ==========
    hidden_size: int = 320          # 隐藏层维度 (d_model)
    num_layers: int = 8             # Decoder层数
    num_heads: int = 5              # 注意力头数 (需整除 hidden_size)
    d_head: int = 64                # 每个头的维度
    max_seq_len: int = 512          # 最大序列长度

    # ========== MLA (Multi-head Latent Attention) — V2 §2.1.2 ==========
    use_mla: bool = True            # 是否启用MLA (False则退化为标准MHA)
    q_lora_rank: int = 80           # Q的低秩压缩维度 (d_q_compress)
    kv_lora_rank: int = 80          # KV的低秩压缩维度 (d_kv_compress)
    qk_rope_dim: int = 32           # Q/K中施加RoPE的维度数 (RoPE只作用于部分维度)

    # ========== MoE (Mixture of Experts) — V2 §2.2, V3 §2.2 ==========
    use_moe: bool = True            # 是否启用MoE (False则退化为标准FFN)
    num_experts: int = 8            # 路由专家数量
    num_shared_experts: int = 1     # 共享专家数量 (始终激活)
    top_k: int = 2                  # 每个token激活的专家数
    expert_intermediate_size: int = 640  # 每个专家的FFN中间层维度

    # ========== Auxiliary-Loss-Free Load Balancing — V3 §2.2 ==========
    use_load_balancing: bool = True
    load_balance_bias_lr: float = 0.01  # 偏置更新学习率

    # ========== MTP (Multi-Token Prediction) — V3 §2.4 ==========
    use_mtp: bool = True            # 是否启用多token预测
    num_pred_tokens: int = 2        # 预测未来token数 (包含next-token, 所以D=2)
    mtp_loss_weight: float = 0.3    # 辅助预测loss的权重

    # ========== Training ==========
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # ========== RoPE ==========
    rope_theta: float = 10000.0     # RoPE的base频率 (V2/V3使用更大的值如100000)

    # ========== Inference ==========
    use_kv_cache: bool = True

    def __post_init__(self):
        # 验证配置
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )
        if self.use_mla:
            assert self.q_lora_rank > 0 and self.kv_lora_rank > 0
            assert self.qk_rope_dim <= self.d_head

    @property
    def num_parameters(self) -> int:
        """估算总参数量"""
        # 这是一个粗略估计，实际参数量由各部分累加
        emb = self.vocab_size * self.hidden_size

        # MLA per layer (rough)
        mla = (self.hidden_size * self.q_lora_rank +  # W_q_a
               self.hidden_size * self.kv_lora_rank + # W_kv_a
               self.q_lora_rank * self.num_heads * self.d_head +  # W_q_b
               self.kv_lora_rank * self.num_heads * self.d_head * 2 +  # W_k_b, W_v_b
               self.num_heads * self.d_head * self.hidden_size)  # W_o

        # MoE per layer (rough)
        expert_size = self.hidden_size * self.expert_intermediate_size * 2  # gate + up
        expert_size += self.expert_intermediate_size * self.hidden_size     # down
        moe = (self.num_experts + self.num_shared_experts) * expert_size
        moe += self.hidden_size * self.num_experts  # router

        total = emb  # embedding
        total += self.num_layers * (mla + moe)  # layers
        total += self.hidden_size * 2  # final RMSNorm
        # LM head weight-tied with embedding, so no extra params
        # MTP heads
        if self.use_mtp:
            total += self.hidden_size * self.vocab_size  # one extra head for t+2

        return total
