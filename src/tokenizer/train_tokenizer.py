"""
训练BPE Tokenizer — 用于mini-DeepSeek模型

使用HuggingFace `tokenizers` 库训练一个BPE tokenizer，
词汇表大小16384，训练语料来自TinyStories。

特殊Token:
  <pad> (id=0), <bos> (id=1), <eos> (id=2), <unk> (id=3),
  <think>, <answer>, 用于CoT格式化

Usage:
  python -m src.tokenizer.train_tokenizer
"""

import os
import sys
import json
import argparse

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from configs.data_config import DataConfig


def get_training_corpus(data_config: DataConfig):
    """生成训练语料的迭代器

    从TinyStories数据集读取文本，逐批yield。
    如果网络不可用，使用本地合成数据。
    """
    try:
        from datasets import load_dataset
        print("Downloading TinyStories dataset...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            trust_remote_code=True,
        )
        # 取前50K条用于tokenizer训练
        for i, example in enumerate(dataset):
            if i >= 50000:
                break
            yield example["text"]
            if i % 5000 == 0:
                print(f"  Processed {i} stories...")
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Using synthetic corpus for tokenizer training...")
        yield from _generate_synthetic_corpus()


def _generate_synthetic_corpus():
    """生成合成训练语料 (fallback)"""
    # 基础数学文本
    math_texts = [
        "Question: If John has 5 apples and gives 2 to Mary, how many does he have left? Think: John started with 5 apples. He gave 2 to Mary. So 5 - 2 = 3 apples remain. Answer: 3",
        "Question: What is 15 + 27? Think: 15 + 27 = 42. Answer: 42",
        "Question: A train travels 60 miles per hour for 3 hours. How far does it go? Think: Distance = speed * time = 60 * 3 = 180 miles. Answer: 180",
        "Question: Sarah has 24 cookies. She wants to share them equally among 6 friends. How many cookies per friend? Think: 24 divided by 6 equals 4. Answer: 4",
        "Question: What is the area of a rectangle with length 8 and width 5? Think: Area = length * width = 8 * 5 = 40. Answer: 40",
    ]

    for i in range(50000):
        base = math_texts[i % len(math_texts)]
        # 添加变化
        yield base + f" [sample_{i}]"


def train_tokenizer(
    vocab_size: int = 16384,
    save_dir: str = None,
    data_config: DataConfig = None,
):
    """训练BPE tokenizer

    Args:
        vocab_size: 词汇表大小
        save_dir: 保存目录
        data_config: 数据配置
    """
    if data_config is None:
        data_config = DataConfig()
    if save_dir is None:
        save_dir = data_config.tokenizer_dir

    print(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # 创建BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # 预分词器: 按字节分割
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 解码器: 字节级
    tokenizer.decoder = decoders.ByteLevel()

    # 后处理器: 添加BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos>:0 $A:0 <eos>:0",
        pair=None,
        special_tokens=[
            ("<bos>", 1),
            ("<eos>", 2),
        ],
    )

    # 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>",    # 0
            "<bos>",    # 1
            "<eos>",    # 2
            "<unk>",    # 3
            "<think>",  # 4
            "<answer>", # 5
        ],
        min_frequency=2,
        show_progress=True,
    )

    # 训练
    print("Training on corpus...")
    tokenizer.train_from_iterator(
        get_training_corpus(data_config),
        trainer=trainer,
    )

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # 保存配置
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "bos_token_id": tokenizer.token_to_id("<bos>"),
        "eos_token_id": tokenizer.token_to_id("<eos>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
        "think_token_id": tokenizer.token_to_id("<think>"),
        "answer_token_id": tokenizer.token_to_id("<answer>"),
    }
    config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Tokenizer config saved to {config_path}")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {config}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    train_tokenizer(vocab_size=args.vocab_size, save_dir=args.save_dir)
