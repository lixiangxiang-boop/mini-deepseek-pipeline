"""
Auxiliary-Loss-Free Load Balancing — DeepSeek-V3 §2.2

DeepSeek-V3 的关键创新: 不使用辅助损失来做负载均衡。
传统的MoE使用 auxiliary loss 强制token均匀分配到各专家，但这会损害模型性能。

V3的方法: 维护每个专家的偏置值 b_i
  - 在每个训练step后统计每个专家处理的token数
  - token数 > 平均数的专家: 减小b_i (降低被选中的概率)
  - token数 < 平均数的专家: 增大b_i (提高被选中的概率)

bias更新规则:
  b_i = b_i + lr * sign(capacity_target - count_i)
  其中 capacity_target = total_tokens / num_experts

注意: 偏置更新发生在step之间，不在前向传播的计算图中。
      这意味着它不会产生梯度，纯粹是一个启发式调节机制。

这是一个非常巧妙的设计: 它分离了"模型优化"(梯度下降)和"负载均衡"(偏置调节)。
"""

import torch
import torch.nn as nn


class AuxFreeLoadBalancer(nn.Module):
    """辅助损失自由的负载均衡器

    Args:
        num_experts: 专家数量
        bias_lr: 偏置更新学习率 (默认0.01)
    """

    def __init__(self, num_experts: int, bias_lr: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.bias_lr = bias_lr

    def update_bias(
        self,
        expert_counts: torch.Tensor,
        expert_bias: torch.Tensor,
    ) -> torch.Tensor:
        """根据专家负载更新偏置值

        这个函数在每次训练step后被调用。

        Args:
            expert_counts: [num_experts] 每个专家处理的token数
            expert_bias: [num_experts] 当前偏置值

        Returns:
            new_bias: [num_experts] 更新后的偏置值
        """
        total_tokens = expert_counts.sum()
        if total_tokens == 0:
            return expert_bias

        # 理想情况下每个专家应处理的token数
        capacity_target = total_tokens / self.num_experts

        # 偏置更新: 处理太多的专家->降低偏置, 处理太少的->提高偏置
        delta = torch.where(
            expert_counts > capacity_target,
            -self.bias_lr,  # 过载 -> 降低
            self.bias_lr,   # 欠载 -> 提高
        )

        # 保持偏置的均值为0 (防止整体drift)
        new_bias = expert_bias + delta
        new_bias = new_bias - new_bias.mean()

        return new_bias

    def get_load_balance_metrics(
        self,
        expert_counts: torch.Tensor,
    ) -> dict:
        """计算负载均衡指标

        Returns:
            dict with:
              - cv: 变异系数 (std/mean, 越小越均衡)
              - max_imbalance: 最不均衡的专家偏离理想的倍数
              - ideal_tokens_per_expert: 理想情况下每个专家的token数
        """
        total = expert_counts.sum().item()
        if total == 0:
            return {"cv": 0, "max_imbalance": 0, "ideal": 0}

        ideal = total / self.num_experts
        counts = expert_counts.float()

        cv = counts.std() / (counts.mean() + 1e-8)
        max_imbalance = counts.max() / (ideal + 1e-8)

        return {
            "cv": cv.item(),
            "max_imbalance": max_imbalance.item(),
            "ideal": ideal,
            "counts": counts.tolist(),
        }
