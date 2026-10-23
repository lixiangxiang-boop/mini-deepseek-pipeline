"""
训练配置 — 各阶段的超参数
"""

from dataclasses import dataclass


@dataclass
class PretrainConfig:
    """预训练配置: Next-token prediction on TinyStories"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    max_seq_len: int = 512
    log_every: int = 50
    save_every: int = 500
    eval_every: int = 500
    max_grad_norm: float = 1.0
    use_amp: bool = True  # Automatic Mixed Precision (fp16)


@dataclass
class SFTConfig:
    """SFT配置: Supervised Fine-Tuning on GSM8K CoT data"""
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 20
    num_epochs: int = 100
    max_steps: int = 1200
    max_seq_len: int = 512
    log_every: int = 20
    save_every: int = 400
    eval_every: int = 400
    max_grad_norm: float = 1.0
    use_amp: bool = True


@dataclass
class GRPOConfig:
    """GRPO配置: Group Relative Policy Optimization (DeepSeek-R1 §2.2.3)

    GRPO 核心参数说明:
      - G: 每个prompt生成多少个候选回答 (DeepSeek-R1使用64，我们使用4)
      - epsilon: PPO-style clipping范围 (小模型用更保守的值)
      - beta: KL惩罚系数 (防止策略偏离ref模型太远)
      - grpo_steps: 每个batch的GRPO迭代步数
    """
    G: int = 4                     # 每组候选回答数
    batch_size: int = 4            # 每次训练的prompt数
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_steps: int = 100
    max_seq_len: int = 256         # prompt最大长度
    max_gen_len: int = 256         # 生成最大长度
    log_every: int = 10
    save_every: int = 50
    eval_every: int = 50
    max_grad_norm: float = 1.0
    # GRPO 特定参数
    epsilon: float = 0.1           # Clipping范围 (更保守)
    beta: float = 0.01             # KL惩罚系数
    temperature: float = 0.8       # 生成温度 (trade-off exploration vs exploitation)
    top_p: float = 0.9
    use_amp: bool = True


@dataclass
class RejectionSamplingConfig:
    """拒绝采样配置 (DeepSeek-R1 §2.2.2)"""
    G: int = 8                     # 每个prompt生成的候选数
    batch_size: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    max_seq_len: int = 256
    max_gen_len: int = 256
