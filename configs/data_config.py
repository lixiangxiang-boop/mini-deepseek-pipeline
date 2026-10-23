"""
数据配置 — 路径、切分、格式
"""

from dataclasses import dataclass
import os


# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
    """数据路径和配置"""

    # ========== 数据根目录 ==========
    data_dir: str = os.path.join(ROOT_DIR, "data")
    tokenizer_dir: str = os.path.join(ROOT_DIR, "data", "tokenizer")

    # ========== TinyStories 预训练数据 ==========
    tinystories_train_file: str = os.path.join(ROOT_DIR, "data", "prepared", "pretrain", "train.pt")
    tinystories_val_file: str = os.path.join(ROOT_DIR, "data", "prepared", "pretrain", "val.pt")

    # ========== GSM8K 数据 ==========
    gsm8k_train_file: str = os.path.join(ROOT_DIR, "data", "gsm8k_train.jsonl")
    gsm8k_test_file: str = os.path.join(ROOT_DIR, "data", "gsm8k_test.jsonl")

    # ========== 预处理后的数据 ==========
    sft_train_file: str = os.path.join(ROOT_DIR, "data", "prepared", "sft_cold", "train.pt")
    grpo_prompts_file: str = os.path.join(ROOT_DIR, "data", "prepared", "grpo_prompts", "prompts.pt")

    # ========== 数据切分 ==========
    # 预训练数据量
    pretrain_train_size: int = 50000    # TinyStories故事数
    pretrain_val_size: int = 1000

    # SFT数据量
    sft_train_size: int = 200           # Cold-start SFT样本数
    sft_val_size: int = 20

    # GRPO prompts
    grpo_train_size: int = 200          # GRPO训练prompt数
    grpo_val_size: int = 50

    # Rejection sampling prompts
    rejection_prompts: int = 500         # 用于拒绝采样的prompt数

    # 评估
    eval_test_size: int = 100           # GSM8K测试样本数

    # ========== Tokenizer ==========
    vocab_size: int = 16384

    # ========== 序列长度 ==========
    max_seq_len: int = 512

    # ========== 特殊Token ==========
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    think_token: str = "<think>"
    answer_token: str = "<answer>"
    question_token: str = "Question:"
    answer_marker: str = "Answer:"

    # ========== HuggingFace镜像 (国内加速) ==========
    hf_endpoint: str = "https://hf-mirror.com"
