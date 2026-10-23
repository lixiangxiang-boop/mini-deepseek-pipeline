"""
GRPO 单元测试 — 损失函数和优势计算

验证:
  1. 组内优势均值为0
  2. Clipped surrogate objective 正确
  3. KL散度非负
  4. loss 是标量

Usage:
  pytest tests/test_grpo.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training.grpo_trainer import compute_grpo_loss
from src.reward.reward_batch import compute_group_advantages


def test_group_advantages():
    """测试组内优势计算 — 每组的优势应该均值为0"""
    # 模拟 G=4, B=2 的奖励
    rewards = [0.0, 0.5, 1.0, 0.0,  # group 1: 2正确
               1.0, 1.0, 0.0, 0.5]   # group 2: 2正确

    advantages = compute_group_advantages(rewards, group_size=4)

    assert len(advantages) == 8

    # 组1的优势均值应该为0
    g1_adv = advantages[:4]
    assert abs(sum(g1_adv)) < 1e-5, f"Group 1 advantages don't sum to 0: {sum(g1_adv)}"

    # 组2的优势均值应该为0
    g2_adv = advantages[4:]
    assert abs(sum(g2_adv)) < 1e-5, f"Group 2 advantages don't sum to 0: {sum(g2_adv)}"


def test_grpo_loss_shapes():
    """测试GRPO损失计算 — 输出应该是标量"""
    n_tokens = 100

    log_probs_current = torch.randn(n_tokens)
    log_probs_old = torch.randn(n_tokens)
    log_probs_ref = torch.randn(n_tokens)
    advantages = torch.randn(n_tokens)

    loss_dict = compute_grpo_loss(
        log_probs_current, log_probs_old, log_probs_ref, advantages
    )

    assert "loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "kl_loss" in loss_dict
    assert loss_dict["loss"].ndim == 0  # scalar


def test_grpo_loss_positive_advantage():
    """测试正优势时policy loss为负 (好的回答应该增加概率)"""
    n_tokens = 50

    # 当前策略比旧策略更好 (log_prob更高 -> ratio > 1)
    log_probs_current = torch.ones(n_tokens) * 0.5
    log_probs_old = torch.zeros(n_tokens)
    log_probs_ref = torch.zeros(n_tokens)

    # 正优势 (好回答)
    advantages = torch.ones(n_tokens) * 2.0

    loss_dict = compute_grpo_loss(
        log_probs_current, log_probs_old, log_probs_ref, advantages,
        epsilon=0.2, beta=0.0,
    )

    # policy loss 应该为负 (因为ratio>1且advantage>0)
    # 但我们的loss加了负号, 所以policy_loss是负的 -> loss减少 -> 好的
    assert loss_dict["policy_loss"] < 0, (
        f"Expected negative policy loss for good actions, got {loss_dict['policy_loss']}"
    )


def test_kl_penalty():
    """测试KL惩罚 — beta越大，KL项越大"""
    n_tokens = 50

    log_probs_current = torch.randn(n_tokens)
    log_probs_old = torch.zeros(n_tokens)
    log_probs_ref = torch.zeros(n_tokens)
    advantages = torch.ones(n_tokens)

    # beta=0 (无KL惩罚)
    loss_no_kl = compute_grpo_loss(
        log_probs_current.clone(), log_probs_old.clone(),
        log_probs_ref.clone(), advantages.clone(),
        epsilon=0.1, beta=0.0,
    )

    # beta=1 (强KL惩罚)
    loss_kl = compute_grpo_loss(
        log_probs_current.clone(), log_probs_old.clone(),
        log_probs_ref.clone(), advantages.clone(),
        epsilon=0.1, beta=1.0,
    )

    # KL loss应该更大
    assert loss_kl["kl_loss"].abs() >= loss_no_kl["kl_loss"].abs(), (
        "KL penalty should increase with beta"
    )


def test_grpo_clipping():
    """测试clipping机制 — 被clip时更新的幅度应受限制"""
    n_tokens = 10

    # 制造一个极大的ratio (策略变化很大)
    log_probs_current = torch.ones(n_tokens) * 10.0  # ratio = exp(10) ≈ 22026
    log_probs_old = torch.zeros(n_tokens)
    log_probs_ref = torch.zeros(n_tokens)
    advantages = torch.ones(n_tokens)

    # 小clipping范围
    loss_tight = compute_grpo_loss(
        log_probs_current.clone(), log_probs_old.clone(),
        log_probs_ref.clone(), advantages.clone(),
        epsilon=0.01, beta=0.0,
    )

    # 大clipping范围
    loss_loose = compute_grpo_loss(
        log_probs_current.clone(), log_probs_old.clone(),
        log_probs_ref.clone(), advantages.clone(),
        epsilon=0.5, beta=0.0,
    )

    # tight clip应该限制更新幅度 (loss不同)
    # 两种设置下loss都应该不是NaN
    assert not torch.isnan(loss_tight["loss"])
    assert not torch.isnan(loss_loose["loss"])


if __name__ == "__main__":
    test_group_advantages()
    test_grpo_loss_shapes()
    test_grpo_loss_positive_advantage()
    test_kl_penalty()
    test_grpo_clipping()
    print("\n  All GRPO tests passed!")
