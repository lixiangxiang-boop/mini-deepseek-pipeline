"""
数学题奖励函数 — DeepSeek-R1 风格

从模型生成的CoT回答中提取最终答案，与GSM8K标准答案比较。

DeepSeek-R1的奖励设计:
  - 规则化奖励 (Rule-based): 数学题用答案正确性作为二值奖励
  - 格式奖励: CoT放在<think>...</think>中，答案放在<answer>...</answer>中

我们简化为:
  - 从文本中提取 Answer: 后面的数字
  - 与ground truth进行数值比较
  - 支持精确匹配和容差匹配
"""

import re
from typing import Tuple, Optional


def extract_answer(text: str) -> Optional[float]:
    """从模型输出中提取数值答案

    支持多种格式:
      - "Answer: 42"
      - "answer: 42.5"
      - "The answer is 42"
      - "答案是42"
      - "= 42"

    Returns:
        提取的数字，或在无法提取时返回None
    """
    if not text:
        return None

    # 优先级: Answer: > answer is > 最后一个数字
    patterns = [
        r"[Aa]nswer:\s*(-?\d+\.?\d*)",       # Answer: 42
        r"[Aa]nswer\s+is\s*(-?\d+\.?\d*)",    # answer is 42
        r"=\s*(-?\d+\.?\d*)\s*$",              # = 42 (行尾)
        r"=\s*(-?\d+\.?\d*)",                  # = 42
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])  # 取最后一个匹配

    # 最后fallback: 取文本中最后一个数字
    all_numbers = re.findall(r"-?\d+\.?\d*", text)
    if all_numbers:
        # 过滤掉太小的数字 (可能是步骤中的中间数字，如 "step 1", "step 2")
        numbers = [float(n) for n in all_numbers if float(n) > 0.1]
        if numbers:
            return numbers[-1]

    return None


def compute_math_reward(
    generated_text: str,
    ground_truth: float,
    tolerance: float = 0.01,
    partial_credit: bool = True,
) -> float:
    """计算数学题回答的奖励

    Args:
        generated_text: 模型生成的完整文本 (包含CoT)
        ground_truth: 标准答案数值
        tolerance: 容差 (相对误差, 默认1%)
        partial_credit: 是否给予部分奖励

    Returns:
        奖励值:
          1.0  — 精确匹配
          0.5  — 在容差范围内 (partial credit)
          0.0  — 错误 / 无法提取答案
    """
    extracted = extract_answer(generated_text)

    if extracted is None:
        return 0.0

    if ground_truth == 0:
        if abs(extracted) < 1e-6:
            return 1.0
        return 0.0

    relative_error = abs(extracted - ground_truth) / (abs(ground_truth) + 1e-8)

    if relative_error < 1e-8:  # 精确匹配
        return 1.0
    elif partial_credit and relative_error < tolerance:
        return 0.5
    else:
        return 0.0


def has_cot_format(text: str) -> bool:
    """检查是否有CoT格式 (用于格式奖励)"""
    has_think = "<think>" in text.lower() or "think:" in text.lower()
    has_answer = "<answer>" in text.lower() or "answer:" in text.lower()
    return has_think and has_answer
