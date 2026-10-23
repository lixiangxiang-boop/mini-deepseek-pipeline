#!/usr/bin/env python
"""
一键运行完整 DeepSeek 训练管道

Usage:
  python run_pipeline.py                    # 运行全部6个阶段
  python run_pipeline.py --stage pretrain   # 只运行预训练
  python run_pipeline.py --stage grpo       # 从GRPO开始 (需要已有SFT checkpoint)
  python run_pipeline.py --device cpu       # 强制使用CPU
  python run_pipeline.py --skip-tests       # 跳过测试

Stages:
  1. data      — 准备数据 + 训练tokenizer
  2. pretrain  — 预训练 on TinyStories
  3. sft       — Cold-Start SFT on GSM8K CoT
  4. grpo      — GRPO Round 1
  5. rs        — Rejection Sampling + Rejection SFT
  6. grpo2     — GRPO Round 2
  7. eval      — 评估 + 管道对比
"""

import os
import sys
import argparse
import time

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_stage(name: str, cmd: str, skip: bool = False):
    """执行单个管道阶段"""
    print(f"\n{'#'*70}")
    print(f"#  STAGE: {name}")
    print(f"{'#'*70}")

    if skip:
        print(f"  [SKIPPED] {name}")
        return True

    t0 = time.time()
    ret = os.system(f"cd {PROJECT_ROOT} && {cmd}")
    elapsed = time.time() - t0

    if ret != 0:
        print(f"\n  [FAILED] {name} (exit code {ret}, took {elapsed:.0f}s)")
        return False

    print(f"\n  [OK] {name} (took {elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run mini-DeepSeek full pipeline")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device (cuda or cpu)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "data", "pretrain", "sft", "grpo", "rs", "grpo2", "eval", "test"],
                        help="Run specific stage or 'all'")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer steps for testing")
    args = parser.parse_args()

    print("=" * 70)
    print("  mini-deepseek-pipeline — Full Pipeline Runner")
    print("  Based on DeepSeek-V2, V3, R1 papers")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  Mode: {'Quick (reduced steps)' if args.quick else 'Full'}")
    print(f"  Skip tests: {args.skip_tests}")

    total_start = time.time()
    success = True

    # Stage 0: 测试
    if args.stage in ("all", "test") and not args.skip_tests:
        success = run_stage(
            "Unit Tests",
            f"{sys.executable} -m pytest tests/ -v --tb=short 2>&1",
        )
        if not success:
            print("  [WARNING] Tests failed, but continuing...")

    if args.stage == "test":
        return

    # 快速模式: 减少步数
    pretrain_steps = 200 if args.quick else 2000
    sft_steps = 100 if args.quick else 1200
    grpo_steps = 20 if args.quick else 100

    # Stage 1: 数据准备
    if args.stage in ("all", "data"):
        success = run_stage(
            "Data Preparation",
            f"{sys.executable} -m src.training.prepare_data",
        )
        if not success:
            print("Data preparation failed. Aborting.")
            return

    # Stage 2: 预训练
    if args.stage in ("all", "pretrain"):
        success = run_stage(
            "Pretraining (TinyStories)",
            f"{sys.executable} -m src.training.pretrain --device {args.device} --steps {pretrain_steps}",
        )
        if not success and args.stage != "all":
            return

    # Stage 3: Cold-Start SFT
    pretrain_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pretrain", "final.pt")
    if args.stage in ("all", "sft"):
        pretrain_arg = f"--pretrained {pretrain_ckpt}" if os.path.exists(pretrain_ckpt) else ""
        success = run_stage(
            "Cold-Start SFT",
            f"{sys.executable} -m src.training.sft_trainer {pretrain_arg} --device {args.device} --steps {sft_steps}",
        )
        if not success and args.stage != "all":
            return

    # Stage 4: GRPO Round 1
    sft_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "cold_sft", "final.pt")
    if args.stage in ("all", "grpo"):
        if os.path.exists(sft_ckpt):
            success = run_stage(
                "GRPO Round 1",
                f"{sys.executable} -m src.training.grpo_trainer --sft_checkpoint {sft_ckpt} --device {args.device} --steps {grpo_steps}",
            )
            if not success and args.stage != "all":
                return
        else:
            print(f"  [SKIP] GRPO: SFT checkpoint not found at {sft_ckpt}")

    # Stage 5: Rejection Sampling + Rejection SFT
    grpo_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "grpo_r1", "final.pt")
    rs_output = os.path.join(PROJECT_ROOT, "data", "prepared", "rejection_sft")
    if args.stage in ("all", "rs"):
        if os.path.exists(grpo_ckpt):
            success = run_stage(
                "Rejection Sampling",
                f"{sys.executable} -m src.training.rejection_sampling --checkpoint {grpo_ckpt} --output {rs_output} --device {args.device}",
            )
            if success:
                print("\n  Rejection sampling done. Run rejection SFT manually if enough samples collected.")
        else:
            print(f"  [SKIP] RS: GRPO checkpoint not found at {grpo_ckpt}")

    # Stage 6: GRPO Round 2 (使用rejection SFT + GRPO后的模型)
    # (简化: 如果有足够时间可以继续迭代)
    if args.stage in ("all", "grpo2"):
        print("\n  [INFO] GRPO Round 2 — run manually after rejection SFT:")
        print(f"    python -m src.training.grpo_trainer --sft_checkpoint checkpoints/rejection_sft/final.pt --device {args.device}")

    # Stage 7: 评估
    if args.stage in ("all", "eval"):
        success = run_stage(
            "Pipeline Evaluation",
            f"{sys.executable} -m src.evaluation.compare_pipelines --device {args.device} --num_samples 50",
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Pipeline complete! Total time: {total_elapsed/60:.1f} min")
    print(f"  Checkpoints saved in: {os.path.join(PROJECT_ROOT, 'checkpoints')}/")
    print(f"  TensorBoard logs:     {os.path.join(PROJECT_ROOT, 'outputs')}/")
    print(f"  Run 'tensorboard --logdir outputs/' to view training curves")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
