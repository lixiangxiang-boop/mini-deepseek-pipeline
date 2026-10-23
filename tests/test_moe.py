"""
MoE 单元测试 — 专家路由和负载均衡

验证:
  1. 路由选择正确的专家数 (top_k)
  2. 共享专家始终有输出
  3. 负载均衡器减少不均衡
  4. 输出无NaN

Usage:
  pytest tests/test_moe.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.deepseek_moe import DeepSeekMoE, Router, ExpertFFN
from src.model.load_balancer import AuxFreeLoadBalancer


def test_router_topk():
    """测试路由器选择top-k个专家"""
    router = Router(hidden_size=320, num_experts=8, top_k=2)

    x = torch.randn(16, 320)  # [B*L, hidden]

    weights, indices = router(x)

    assert weights.shape == (16, 2)
    assert indices.shape == (16, 2)
    # 权重应该被softmax归一化
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_moe_forward():
    """测试MoE前向传播"""
    moe = DeepSeekMoE(
        hidden_size=320,
        intermediate_size=640,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
    )

    x = torch.randn(2, 32, 320)
    output, expert_counts = moe(x)

    assert output.shape == (2, 32, 320)
    assert expert_counts.shape == (8,)
    assert not torch.isnan(output).any()


def test_shared_expert_always_active():
    """测试共享专家总是激活"""
    moe = DeepSeekMoE(
        hidden_size=320,
        intermediate_size=640,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
    )

    # 测试两次，确保共享专家总是有非零输出
    for _ in range(5):
        x = torch.randn(4, 16, 320)
        output, _ = moe(x)
        # 输出应该非零 (因为有共享专家的贡献)
        assert output.abs().sum() > 0


def test_load_balancer():
    """测试负载均衡器减少不均衡"""
    balancer = AuxFreeLoadBalancer(num_experts=8, bias_lr=0.01)

    bias = torch.zeros(8)

    # 模拟极端不均衡
    counts = torch.tensor([100, 5, 5, 5, 5, 5, 5, 5], dtype=torch.float)

    for _ in range(50):
        bias = balancer.update_bias(counts, bias)

    # 经过多轮后，过载专家(索引0)的偏置应该降低
    assert bias[0] < 0, f"Overloaded expert bias should be negative, got {bias[0]}"
    # 欠载专家的偏置应该提高
    assert bias[1:].mean() > bias[0], "Underloaded experts should have higher bias"


def test_expert_utilization():
    """测试专家利用率统计"""
    moe = DeepSeekMoE(
        hidden_size=320,
        intermediate_size=640,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
    )

    x = torch.randn(4, 64, 320)
    _, expert_counts = moe(x)

    info = moe.get_expert_utilization(expert_counts)
    print(f"\n  Expert utilization: {info['utilization']}")
    print(f"  Max/Min ratio: {info['max_min_ratio']:.2f}")
    print(f"  Std: {info['std']:.4f}")

    assert len(info["utilization"]) == 8
    assert info["max_min_ratio"] >= 1.0


def test_expert_ffn():
    """测试单个专家FFN"""
    ffn = ExpertFFN(hidden_size=320, intermediate_size=640)

    x = torch.randn(4, 16, 320)
    output = ffn(x)

    assert output.shape == (4, 16, 320)
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    test_router_topk()
    test_moe_forward()
    test_shared_expert_always_active()
    test_load_balancer()
    test_expert_utilization()
    test_expert_ffn()
    print("\n  All MoE tests passed!")
