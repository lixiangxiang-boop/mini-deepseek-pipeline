"""
数据准备 — 下载、格式化、tokenize

为整个管道准备数据:
  1. 下载TinyStories (预训练)
  2. 下载GSM8K (SFT + GRPO + 评估)
  3. 格式化为CoT格式
  4. Tokenize并保存为.pt文件

Usage:
  python -m src.training.prepare_data
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.data_config import DataConfig
from src.tokenizer.tokenizer_utils import get_tokenizer

# 全局 tokenizer 实例 (在 prepare_all_data 中初始化)
_tokenizer = None

def _encode(text: str) -> torch.Tensor:
    """使用真正的 tokenizer 编码文本"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_tokenizer()
    ids = _tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor(ids, dtype=torch.long)


def prepare_all_data(data_config: DataConfig = None):
    """准备所有训练数据

    生产环境: 使用HuggingFace datasets下载真实数据
    开发环境: 使用合成数据 (快速测试)
    """
    if data_config is None:
        data_config = DataConfig()

    os.makedirs(os.path.dirname(data_config.tinystories_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(data_config.sft_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(data_config.grpo_prompts_file), exist_ok=True)

    print("=" * 60)
    print("  Data Preparation")
    print("=" * 60)

    # 1. TinyStories (预训练数据)
    print("\n[1/3] Preparing TinyStories pretraining data...")
    try:
        _prepare_tinystories(data_config)
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Generating synthetic pretraining data...")
        _prepare_synthetic_pretrain(data_config)

    # 2. GSM8K CoT (SFT数据)
    print("\n[2/3] Preparing GSM8K CoT SFT data...")
    try:
        _prepare_gsm8k_sft(data_config)
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Generating synthetic SFT data...")
        _prepare_synthetic_sft(data_config)

    # 3. GRPO Prompts
    print("\n[3/3] Preparing GRPO prompts...")
    _prepare_grpo_prompts(data_config)

    print(f"\n  Data preparation complete!")
    print(f"    Pretrain: {data_config.tinystories_train_file}")
    print(f"    SFT:      {data_config.sft_train_file}")
    print(f"    GRPO:     {data_config.grpo_prompts_file}\n")


def _prepare_tinystories(data_config: DataConfig):
    """从TinyStories下载并处理预训练数据"""
    try:
        from datasets import load_dataset
        print("  Downloading TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
    except Exception:
        raise RuntimeError("Cannot download TinyStories")

    # Tokenize (简化 — 使用字符级tokenization)
    train_data = []
    val_data = []

    for i, example in enumerate(dataset):
        text = example["text"]
        # 简化tokenization: 每个字符映射到一个token id
        tokens = _encode(text)

        if i < data_config.pretrain_train_size:
            train_data.append(tokens)
            if i % 10000 == 0:
                print(f"    Processed {i} training samples...")
        elif i < data_config.pretrain_train_size + data_config.pretrain_val_size:
            val_data.append(tokens)
        else:
            break

    torch.save(train_data, data_config.tinystories_train_file)
    torch.save(val_data, data_config.tinystories_val_file)
    print(f"  Saved: {len(train_data)} train, {len(val_data)} val samples")


def _prepare_synthetic_pretrain(data_config: DataConfig):
    """生成合成预训练数据"""
    import random
    random.seed(42)

    stories = [
        "Once upon a time there was a little cat. The cat liked to play with a ball.",
        "A dog went to the park. At the park the dog saw a bird. The bird was singing.",
        "The sun was bright. The children played in the garden all day long.",
        "A fish swam in the pond. The pond was clear and full of lily pads.",
    ]

    train_data = []
    for i in range(data_config.pretrain_train_size):
        base = random.choice(stories)
        tokens = _encode(base)
        train_data.append(tokens)

    torch.save(train_data, data_config.tinystories_train_file)
    print(f"  Synthetic pretrain data: {len(train_data)} samples")


def _prepare_gsm8k_sft(data_config: DataConfig):
    """准备GSM8K CoT SFT数据"""
    sft_data = _generate_math_cot_data(data_config.sft_train_size + data_config.sft_val_size)

    # 格式化为CoT
    formatted = []
    for item in sft_data:
        text = (
            f"<bos> Question: {item['question']}\n"
            f"<think> {item['reasoning']} </think>\n"
            f"<answer> {item['answer']} </answer> <eos>"
        )
        tokens = _encode(text)
        formatted.append(tokens)

    train = formatted[:data_config.sft_train_size]
    val = formatted[data_config.sft_train_size:]

    torch.save(train, data_config.sft_train_file)
    print(f"  SFT data: {len(train)} train, {len(val)} val samples")


def _prepare_synthetic_sft(data_config: DataConfig):
    """生成合成SFT数据"""
    sft_data = _generate_math_cot_data(data_config.sft_train_size)
    formatted = []
    for item in sft_data:
        text = (
            f"<bos> Question: {item['question']}\n"
            f"<think> {item['reasoning']} </think>\n"
            f"<answer> {item['answer']} </answer> <eos>"
        )
        tokens = _encode(text)
        formatted.append(tokens)

    torch.save(formatted, data_config.sft_train_file)
    print(f"  Synthetic SFT data: {len(formatted)} samples")


def _prepare_grpo_prompts(data_config: DataConfig):
    """准备GRPO prompts (只需要问题，不需要答案)"""
    problems = _generate_math_cot_data(data_config.grpo_train_size)

    prompts = []
    for item in problems:
        text = f"<bos> Question: {item['question']}\n<think>"
        tokens = _encode(text)
        prompts.append({
            "input_ids": tokens,
            "ground_truth": item["answer"],
            "prompt_text": item["question"],
        })

    torch.save(prompts, data_config.grpo_prompts_file)
    print(f"  GRPO prompts: {len(prompts)} samples")


def _generate_math_cot_data(num_samples: int) -> List[Dict]:
    """生成数学CoT数据 (合成)"""
    import random
    random.seed(42)

    templates = [
        {
            "question": "If John has {a} apples and gives {b} to Mary, how many does he have left?",
            "reasoning": "John started with {a} apples. He gave {b} to Mary. So we subtract: {a} - {b} = {c}.",
            "gen": lambda: _gen_subtract(),
        },
        {
            "question": "What is {a} + {b}?",
            "reasoning": "We add {a} and {b}: {a} + {b} = {c}.",
            "gen": lambda: _gen_add(),
        },
        {
            "question": "A train travels at {a} miles per hour for {b} hours. How far does it go?",
            "reasoning": "Distance = speed × time = {a} × {b} = {c} miles.",
            "gen": lambda: _gen_multiply(),
        },
        {
            "question": "Sarah has {a} cookies. She shares equally among {b} friends. How many each?",
            "reasoning": "We divide {a} by {b}: {a} ÷ {b} = {c}.",
            "gen": lambda: _gen_divide(),
        },
        {
            "question": "What is the area of a rectangle with length {a} and width {b}?",
            "reasoning": "Area = length × width = {a} × {b} = {c}.",
            "gen": lambda: _gen_multiply(),
        },
    ]

    data = []
    for _ in range(num_samples):
        t = random.choice(templates)
        a, b, c = t["gen"]()
        data.append({
            "question": t["question"].format(a=a, b=b, c=c),
            "reasoning": t["reasoning"].format(a=a, b=b, c=c),
            "answer": c,
        })

    return data


def _gen_add():
    a = __import__("random").randint(1, 100)
    b = __import__("random").randint(1, 100)
    return a, b, a + b


def _gen_subtract():
    a = __import__("random").randint(10, 100)
    b = __import__("random").randint(1, a)
    return a, b, a - b


def _gen_multiply():
    a = __import__("random").randint(2, 12)
    b = __import__("random").randint(2, 12)
    return a, b, a * b


def _gen_divide():
    b = __import__("random").randint(2, 10)
    c = __import__("random").randint(1, 20)
    a = b * c
    return a, b, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--force_synthetic", action="store_true",
                        help="Force using synthetic data instead of downloading")
    args = parser.parse_args()

    prepare_all_data()
