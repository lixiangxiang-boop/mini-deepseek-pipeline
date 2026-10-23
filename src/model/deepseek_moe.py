"""
DeepSeekMoE — Fine-Grained Mixture of Experts
DeepSeek-V2 §2.2 和 DeepSeek-V3 §2.2

DeepSeekMoE 与标准 MoE (如 Mixtral) 的关键区别:

1. 细粒度专家分割 (Fine-Grained Expert Segmentation):
   传统MoE: 每个专家是一个完整的FFN (大但少)
   DeepSeekMoE: 将FFN拆成更多更小的专家 (小但多)
   好处: 每个专家更专精，组合更灵活

2. 共享专家隔离 (Shared Expert Isolation):
   除路由专家外，还有一个"共享专家"始终激活
   共享专家学习所有token的通用知识
   路由专家学习各自专长领域

3. 辅助损失自由的负载均衡 (V3):
   不使用辅助loss，改为动态调整每个专家的偏置
   见 load_balancer.py

架构:
  x -> Router -> Top-K gate -> routed_expert_1(x) * gate_1
                             -> routed_expert_2(x) * gate_2
                             -> shared_expert(x)  (始终激活)
  output = sum(routed_outputs) + shared_output

每个专家FFN使用Swish-Gated结构 (SiLU激活):
  gate = x @ W_gate
  up   = x @ W_up
  output = (SiLU(gate) * up) @ W_down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertFFN(nn.Module):
    """单个专家的FFN — SwiGLU结构

    SwiGLU = SiLU(W_gate @ x) * (W_up @ x) 然后 W_down 投影回 hidden_size
    这与DeepSeek使用的gated FFN一致。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class SharedExpert(ExpertFFN):
    """共享专家 — 始终激活，学习通用知识

    与路由专家的区别:
      - 只有一个 (不是多个)
      - 不对输入进行路由选择
      - 学习所有token共享的模式
    """
    pass


class Router(nn.Module):
    """Top-K路由器

    输入: hidden states [B, L, hidden]
    输出: top_k 个被选中的专家索引 + 对应的gate权重
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由权重 (与V2的expert gate一致)
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B*L, hidden]
            expert_bias: [num_experts] 动态偏置 (用于负载均衡)

        Returns:
            topk_weights: [B*L, top_k] softmax归一化后的gate权重
            topk_indices: [B*L, top_k] 选中的专家索引
        """
        # 计算路由logits
        logits = F.linear(x, self.weight)  # [B*L, num_experts]

        # 添加动态偏置 (DeepSeek-V3的auxiliary-loss-free balancing)
        if expert_bias is not None:
            logits = logits + expert_bias

        # Top-K选择
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax归一化 (只对被选中的专家)
        topk_weights = F.softmax(topk_weights, dim=-1)

        return topk_weights, topk_indices


class DeepSeekMoE(nn.Module):
    """DeepSeek风格的MoE层

    组成:
      - num_experts 个细粒度路由专家
      - 1 个共享专家 (始终激活)
      - 1 个 Top-K 路由器
      - 辅助损失自由的负载均衡器 (外部传入)

    Args:
        hidden_size: 隐藏层维度
        intermediate_size: 每个专家的FFN中间维度
        num_experts: 路由专家数量
        num_shared_experts: 共享专家数量 (推荐1)
        top_k: 每个token激活的路由专家数
        dropout: FFN中的dropout率
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k

        # 路由专家 (多个细粒度专家)
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, intermediate_size, dropout)
            for _ in range(num_experts)
        ])

        # 共享专家 (始终激活)
        self.shared_experts = nn.ModuleList([
            SharedExpert(hidden_size, intermediate_size, dropout)
            for _ in range(num_shared_experts)
        ])

        # 路由器
        self.router = Router(hidden_size, num_experts, top_k)

        # 负载均衡偏置 (外部更新)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
        update_load_balance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            update_load_balance: 是否更新负载统计（训练时为True，推理时为False）

        Returns:
            output: [batch, seq_len, hidden_size]
            expert_counts: [num_experts] 每个专家处理的token数
        """
        batch, seq_len, hidden = x.shape

        # 展平为 [B*L, hidden]
        x_flat = x.view(-1, hidden)

        # ===== 1. 路由专家 =====
        topk_weights, topk_indices = self.router(x_flat, self.expert_bias)
        # topk_weights: [B*L, top_k], topk_indices: [B*L, top_k]

        # 初始化输出
        routed_output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        # 对每个被选中的专家，聚合其输出
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]    # [B*L]
            expert_weight = topk_weights[:, i]  # [B*L]

            # 对每个专家单独处理
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # 哪些token被路由到这个专家
                if mask.any():
                    token_inputs = x_flat[mask]  # 该专家需要处理的token
                    token_outputs = self.experts[e](token_inputs)
                    routed_output[mask] += token_outputs * expert_weight[mask].unsqueeze(-1)
                    expert_counts[e] += mask.sum()

        # ===== 2. 共享专家 (始终激活) =====
        shared_output = torch.zeros_like(x_flat)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(x_flat)
        # 共享专家的贡献权重为 1/num_shared_experts
        shared_output = shared_output / self.num_shared_experts

        # ===== 3. 合并输出 =====
        output = routed_output + shared_output

        return output.view(batch, seq_len, hidden), expert_counts

    def get_expert_utilization(self, expert_counts: torch.Tensor) -> dict:
        """计算专家利用率统计 (用于监控负载均衡)"""
        total_tokens = expert_counts.sum().item()
        if total_tokens == 0:
            return {"utilization": [], "max_min_ratio": 0, "std": 0}

        utilization = expert_counts / total_tokens
        max_count = expert_counts.max().item()
        min_count = expert_counts.min().item()
        max_min_ratio = max_count / max(min_count, 1)

        return {
            "utilization": utilization.tolist(),
            "max_min_ratio": max_min_ratio,
            "std": expert_counts.std().item(),
        }
