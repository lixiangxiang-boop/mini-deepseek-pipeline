"""
批量奖励计算 — 用于GRPO训练的高效奖励评估

对多个(文本, 标准答案)对并行计算奖励。
"""

from typing import List, Tuple
from .math_reward import compute_math_reward, has_cot_format


def compute_reward_batch(
    generated_texts: List[str],
    ground_truths: List[float],
    format_reward_weight: float = 0.05,
) -> List[float]:
    """批量计算奖励

    Args:
        generated_texts: 模型生成的文本列表
        ground_truths: 对应的标准答案
        format_reward_weight: CoT格式奖励的权重

    Returns:
        每个样本的奖励值
    """
    rewards = []
    for text, gt in zip(generated_texts, ground_truths):
        # 数学正确性奖励
        math_r = compute_math_reward(text, gt)

        # 格式奖励: 正确的CoT格式
        format_r = 1.0 if has_cot_format(text) else 0.0

        # 总奖励 = 数学奖励 + 小的格式奖励
        total_r = math_r + format_reward_weight * format_r
        rewards.append(total_r)

    return rewards


def compute_group_advantages(
    rewards: List[float],
    group_size: int,
) -> List[float]:
    """计算组内相对优势 (GRPO核心操作)

    Args:
        rewards: 所有样本的奖励 [G * B]
        group_size: 每组的大小 (G)

    Returns:
        advantages: 每个样本的优势值
    """
    num_groups = len(rewards) // group_size
    advantages = []

    for g in range(num_groups):
        group_rewards = rewards[g * group_size:(g + 1) * group_size]
        group_mean = sum(group_rewards) / len(group_rewards)

        # 标准差 (防止除0)
        variance = sum((r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)
        group_std = variance ** 0.5 + 1e-8

        for r in group_rewards:
            advantages.append((r - group_mean) / group_std)

    return advantages
