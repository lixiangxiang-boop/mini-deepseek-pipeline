"""
拒绝采样 (Rejection Sampling) — DeepSeek-R1 §2.2.2

R1管道的核心环节: 用当前模型生成大量候选回答，筛选出正确的作为新的SFT数据。

流程:
  1. 对每个prompt生成 G 个候选回答
  2. 用奖励函数评估每个回答的正确性
  3. 只保留正确的回答 (奖励==1.0)
  4. 用筛选出的(prompt, correct_answer)对做下一轮SFT

这样实现自我提升 (self-improvement):
  - 模型自己生成训练数据
  - 只保留高质量(正确)的样本
  - 用这些样本再次SFT，能力提升
  - 与GRPO形成互补: RS提供高质量SFT数据，GRPO做RL优化

Usage:
  python -m src.training.rejection_sampling --checkpoint checkpoints/cold_sft/final.pt --output data/prepared/rejection_sft/
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.reward.math_reward import compute_math_reward, extract_answer
from configs.training_config import RejectionSamplingConfig
from configs.data_config import DataConfig


def rejection_sampling(
    checkpoint_path: str,
    output_dir: str,
    model_config: DeepSeekConfig = None,
    sampling_config: RejectionSamplingConfig = None,
    data_config: DataConfig = None,
    device: str = "cuda",
):
    """执行拒绝采样

    Args:
        checkpoint_path: 模型checkpoint路径
        output_dir: 筛选后数据的输出目录
        model_config: 模型配置
        sampling_config: 采样配置
        data_config: 数据配置
        device: 推理设备

    Returns:
        筛选统计信息
    """
    if model_config is None:
        model_config = DeepSeekConfig()
    if sampling_config is None:
        sampling_config = RejectionSamplingConfig()
    if data_config is None:
        data_config = DataConfig()

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Rejection Sampling — DeepSeek-R1 §2.2.2")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Candidates per prompt (G): {sampling_config.G}")
    print(f"  Temperature: {sampling_config.temperature}")

    # 加载模型
    model = DeepSeekForCausalLM(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded from step {checkpoint.get('step', 'unknown')}")

    # 加载prompts
    prompts_data = _load_prompts(data_config, sampling_config)
    print(f"  Total prompts: {len(prompts_data)}")

    # 采样
    accepted = []
    stats = {
        "total_prompts": len(prompts_data),
        "total_generations": 0,
        "total_correct": 0,
        "rejection_rate": 0.0,
        "avg_reward": 0.0,
    }

    print(f"\n  Generating and filtering...\n")
    total_reward = 0.0

    for item in tqdm(prompts_data, desc="Rejection Sampling"):
        prompt_tokens = item["input_ids"].unsqueeze(0).to(device)
        ground_truth = item.get("ground_truth", 0.0)
        prompt_text = item.get("prompt_text", "")

        for g in range(sampling_config.G):
            stats["total_generations"] += 1

            with torch.no_grad():
                generated = model.generate(
                    prompt_tokens,
                    max_new_tokens=sampling_config.max_gen_len,
                    temperature=sampling_config.temperature,
                    top_p=sampling_config.top_p,
                    eos_token_id=model_config.eos_token_id,
                )

            # 解码
            generated_text = _decode_tokens(generated[0], model_config)

            # 计算奖励
            reward = compute_math_reward(generated_text, ground_truth)
            total_reward += reward

            if reward >= 1.0:  # 精确正确
                stats["total_correct"] += 1
                accepted.append({
                    "input_ids": generated[0].tolist(),
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "reward": reward,
                    "ground_truth": ground_truth,
                })

    stats["avg_reward"] = total_reward / max(1, stats["total_generations"])
    stats["rejection_rate"] = 1.0 - stats["total_correct"] / max(1, stats["total_generations"])
    stats["accepted_count"] = len(accepted)

    # 保存筛选后的数据
    output_file = os.path.join(output_dir, "rejection_sampled.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in accepted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 保存统计
    stats_file = os.path.join(output_dir, "rejection_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  {'='*40}")
    print(f"  Rejection Sampling Results:")
    print(f"    Total generations:     {stats['total_generations']}")
    print(f"    Correct (accepted):    {stats['total_correct']}")
    print(f"    Rejection rate:        {stats['rejection_rate']:.1%}")
    print(f"    Avg reward:            {stats['avg_reward']:.3f}")
    print(f"    Accepted samples:      {stats['accepted_count']}")
    print(f"    Output:                {output_file}")
    print(f"  {'='*40}\n")

    return accepted, stats


def _load_prompts(data_config: DataConfig, sampling_config: RejectionSamplingConfig) -> List[Dict]:
    """加载prompts数据"""
    try:
        data = torch.load(data_config.grpo_prompts_file)
        # 取前 N 个用于拒绝采样
        return data[:sampling_config.max_prompts] if hasattr(sampling_config, 'max_prompts') else data
    except FileNotFoundError:
        print(f"  [WARNING] GRPO prompts not found. Creating dummy prompts.")
        import random
        dummy = []
        math_problems = [
            ("If John has 5 apples and gives 2 to Mary, how many does he have left?", 3.0),
            ("What is 15 + 27?", 42.0),
            ("A train travels 60 miles per hour for 3 hours. How far does it go?", 180.0),
            ("Sarah has 24 cookies. She wants to share equally among 6 friends. How many each?", 4.0),
            ("What is the area of a rectangle with sides 8 and 5?", 40.0),
        ]
        for i in range(50):
            problem, answer = random.choice(math_problems)
            tokens = torch.randint(4, 1000, (sampling_config.max_seq_len,))
            dummy.append({
                "input_ids": tokens,
                "ground_truth": answer,
                "prompt_text": problem,
            })
        return dummy


def _decode_tokens(tokens: torch.Tensor, config: DeepSeekConfig) -> str:
    """简化的token解码 (无完整tokenizer时的fallback)"""
    # 这里应该使用训练好的tokenizer解码
    # 简化实现: 返回token序列的字符串表示
    return f"[generated_{len(tokens)}_tokens]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rejection sampling for mini-DeepSeek")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rejection_sampling(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
    )
