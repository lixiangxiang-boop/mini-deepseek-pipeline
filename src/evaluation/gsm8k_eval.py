"""
GSM8K评估 — 数学题准确率测试

从GSM8K测试集中采样100题，评估模型准确率。
用于对比各训练阶段的效果。

Usage:
  python -m src.evaluation.gsm8k_eval --checkpoint checkpoints/final.pt --num_samples 100
"""

import os
import sys
import json
import argparse
from typing import Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.inference.generate import generate_text
from src.reward.math_reward import compute_math_reward, extract_answer
from configs.data_config import DataConfig


def evaluate_gsm8k(
    checkpoint_path: str,
    num_samples: int = 100,
    model_config: DeepSeekConfig = None,
    data_config: DataConfig = None,
    device: str = "cuda",
) -> dict:
    """评估GSM8K准确率

    Args:
        checkpoint_path: 模型checkpoint
        num_samples: 评估样本数
        model_config: 模型配置
        data_config: 数据配置
        device: 推理设备

    Returns:
        dict: {
            "accuracy": float,
            "correct": int,
            "total": int,
            "avg_reward": float,
            "format_accuracy": float,
        }
    """
    if model_config is None:
        model_config = DeepSeekConfig()
    if data_config is None:
        data_config = DataConfig()

    print("=" * 60)
    print("  GSM8K Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples: {num_samples}")

    # 加载模型
    model = DeepSeekForCausalLM(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded from step {checkpoint.get('step', 'unknown')}")

    # 加载测试数据
    test_data = _load_test_data(data_config, num_samples)
    print(f"  Test samples loaded: {len(test_data)}")

    # 评估
    correct = 0
    partial_correct = 0
    format_correct = 0
    total_reward = 0.0
    total = 0

    results = []

    print(f"\n  Evaluating...\n")
    for item in tqdm(test_data, desc="GSM8K Eval"):
        prompt_text = item.get("prompt_text", "")
        ground_truth = item.get("ground_truth", 0.0)

        # Tokenize (简化)
        prompt_ids = torch.tensor(
            [[ord(c) % (model_config.vocab_size - 10) + 10 for c in prompt_text[:model_config.max_seq_len]]],
            device=device,
        )

        with torch.no_grad():
            generated = generate_text(
                model,
                prompt_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=model_config.eos_token_id,
                use_cache=True,
            )

        # 解码
        generated_part = generated[0, prompt_ids.shape[1]:]
        chars = [chr(t.item() % 128) for t in generated_part if 32 <= t.item() % 128 < 127]
        generated_text = "".join(chars)

        # 评估
        reward = compute_math_reward(generated_text, ground_truth)

        if reward >= 1.0:
            correct += 1
        if reward >= 0.5:
            partial_correct += 1

        # 格式检查
        has_think = "<think>" in generated_text.lower() or "think:" in generated_text.lower()
        has_answer = "<answer>" in generated_text.lower() or "answer:" in generated_text.lower()
        if has_think and has_answer:
            format_correct += 1

        total_reward += reward
        total += 1

        results.append({
            "prompt": prompt_text[:100],
            "ground_truth": ground_truth,
            "generated": generated_text[:200],
            "reward": reward,
        })

    # 汇总
    metrics = {
        "accuracy": correct / max(total, 1),
        "partial_accuracy": partial_correct / max(total, 1),
        "format_accuracy": format_correct / max(total, 1),
        "avg_reward": total_reward / max(total, 1),
        "correct": correct,
        "partial_correct": partial_correct,
        "total": total,
        "format_correct": format_correct,
    }

    print(f"\n  {'='*40}")
    print(f"  Results:")
    print(f"    Exact Accuracy:   {metrics['accuracy']:.1%} ({correct}/{total})")
    print(f"    Partial Accuracy:  {metrics['partial_accuracy']:.1%} ({partial_correct}/{total})")
    print(f"    Format Accuracy:   {metrics['format_accuracy']:.1%} ({format_correct}/{total})")
    print(f"    Avg Reward:        {metrics['avg_reward']:.3f}")
    print(f"  {'='*40}\n")

    return metrics


def _load_test_data(data_config: DataConfig, num_samples: int) -> list:
    """加载GSM8K测试数据"""
    test_file = data_config.gsm8k_test_file
    if os.path.exists(test_file):
        if test_file.endswith(".jsonl"):
            with open(test_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            return data[:num_samples]
        elif test_file.endswith(".pt"):
            return torch.load(test_file)[:num_samples]

    # Fallback: dummy test data
    print(f"  [WARNING] Test data not found. Using dummy data.")
    dummy = []
    math_problems = [
        ("If John has 5 apples and gives 2 to Mary, how many does he have left?", 3.0),
        ("What is 15 + 27?", 42.0),
        ("A train travels 60 miles per hour for 3 hours. How far?", 180.0),
        ("Sarah has 24 cookies. Share among 6 friends. How many each?", 4.0),
        ("Area of rectangle with sides 8 and 5?", 40.0),
    ]
    for i in range(num_samples):
        p, a = math_problems[i % len(math_problems)]
        dummy.append({"prompt_text": p, "ground_truth": a})
    return dummy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_gsm8k(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device,
    )
