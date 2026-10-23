"""
管道对比 — 4阶段效果评估

对比各训练阶段模型的GSM8K准确率:
  1. Pretrain only (baseline, no instruction following)
  2. Cold-Start SFT (basic CoT reasoning)
  3. SFT + GRPO Round 1 (RL improved reasoning)
  4. Full R1 Pipeline (SFT -> GRPO -> RS -> SFT -> GRPO)

这是项目的最终教育输出: 展示每个DeepSeek技术的增量贡献。

Usage:
  python -m src.evaluation.compare_pipelines --checkpoints_dir checkpoints/ --num_samples 100
"""

import os
import sys
import argparse
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.gsm8k_eval import evaluate_gsm8k
from src.model.deepseek_config import DeepSeekConfig


def compare_pipelines(
    checkpoints: Dict[str, str],
    num_samples: int = 100,
    device: str = "cuda",
) -> List[Dict]:
    """对比多个管道的效果

    Args:
        checkpoints: {"stage_name": "checkpoint_path"} 字典
        num_samples: 每个checkpoint评估的样本数
        device: 推理设备

    Returns:
        每个阶段的评估指标列表
    """
    print("=" * 60)
    print("  Pipeline Comparison — DeepSeek R1 Full Pipeline")
    print("=" * 60)
    print(f"  Stages to compare: {len(checkpoints)}")
    print(f"  Samples per stage: {num_samples}")
    print()

    all_metrics = []

    for stage_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {stage_name}: checkpoint not found ({ckpt_path})")
            all_metrics.append({
                "stage": stage_name,
                "error": "checkpoint not found",
            })
            continue

        print(f"\n  --- {stage_name} ---")
        try:
            metrics = evaluate_gsm8k(
                checkpoint_path=ckpt_path,
                num_samples=num_samples,
                device=device,
            )
            metrics["stage"] = stage_name
            metrics["checkpoint"] = ckpt_path
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [ERROR] {stage_name}: {e}")
            all_metrics.append({
                "stage": stage_name,
                "error": str(e),
            })

    # 汇总表
    print(f"\n\n{'='*70}")
    print(f"  Final Comparison Table")
    print(f"{'='*70}")
    print(f"  {'Stage':<25} {'Accuracy':>10} {'Avg Reward':>12} {'Format':>10}")
    print(f"  {'-'*57}")

    for m in all_metrics:
        if "error" in m:
            print(f"  {m['stage']:<25} {'ERROR':>10} {'-':>12} {'-':>10}")
        else:
            print(f"  {m['stage']:<25} {m['accuracy']:>9.1%} {m['avg_reward']:>11.3f} {m['format_accuracy']:>9.1%}")

    print(f"{'='*70}\n")

    # 预期趋势说明
    print("  Expected Trend (matching DeepSeek-R1 findings):")
    print("    Pretrain only:   ~2-5%   (no instruction following)")
    print("    Cold-Start SFT:  ~10-15% (basic CoT, learns format)")
    print("    SFT + GRPO:      ~15-25% (RL improves reasoning)")
    print("    Full R1 Pipeline: ~20-30% (self-improvement loop)")
    print()
    print("  Note: These are EXPECTED numbers for a 27M model.")
    print("  The point is to observe the TREND of improvement")
    print("  across stages, matching DeepSeek-R1's methodology.")
    print()

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pipeline stages")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(project_root, args.checkpoints_dir)

    checkpoints = {
        "1_Pretrained": os.path.join(base, "pretrain", "final.pt"),
        "2_ColdStart_SFT": os.path.join(base, "cold_sft", "final.pt"),
        "3_SFT_GRPO_R1": os.path.join(base, "grpo_r1", "final.pt"),
        "4_Full_R1_Pipeline": os.path.join(base, "grpo_r2", "final.pt"),
    }

    compare_pipelines(
        checkpoints=checkpoints,
        num_samples=args.num_samples,
        device=args.device,
    )
