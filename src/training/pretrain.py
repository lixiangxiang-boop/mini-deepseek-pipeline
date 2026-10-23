"""
预训练 — Next-Token Prediction on TinyStories

使用TinyStories数据集训练基础语言模型能力。
这是DeepSeek管道的第一步: 让模型学会基本的语言理解和生成。

Training loop:
  for step in range(max_steps):
    batch = next(dataloader)
    loss = model(input_ids, labels).loss
    loss.backward()
    optimizer.step()

记录: loss, perplexity, 学习率
保存: checkpoint每save_every步
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.training.data_collator import pretrain_collate_fn
from configs.training_config import PretrainConfig
from configs.data_config import DataConfig


class TinyStoriesDataset(Dataset):
    """TinyStories数据集封装"""

    def __init__(self, data_file: str, max_seq_len: int = 512):
        self.data = torch.load(data_file)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_seq_len]
        return {"input_ids": tokens}


def pretrain(
    model_config: DeepSeekConfig = None,
    train_config: PretrainConfig = None,
    data_config: DataConfig = None,
    checkpoint_dir: str = None,
    log_dir: str = None,
    device: str = "cuda",
):
    """运行预训练

    Args:
        model_config: 模型配置
        train_config: 训练配置
        data_config: 数据配置
        checkpoint_dir: checkpoint保存目录
        log_dir: TensorBoard日志目录
        device: 训练设备
    """
    if model_config is None:
        model_config = DeepSeekConfig()
    if train_config is None:
        train_config = PretrainConfig()
    if data_config is None:
        data_config = DataConfig()
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "checkpoints", "pretrain")
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "outputs", "pretrain")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    print("=" * 60)
    print("  Pretraining — Next-Token Prediction")
    print("=" * 60)
    print(f"  Model: ~{model_config.num_parameters/1e6:.1f}M params")
    print(f"  Device: {device}")
    print(f"  Steps: {train_config.max_steps}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")

    # 模型
    model = DeepSeekForCausalLM(model_config).to(device)
    model.print_model_summary()

    # 数据
    try:
        train_dataset = TinyStoriesDataset(
            data_config.tinystories_train_file, train_config.max_seq_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=lambda b: pretrain_collate_fn(b, model_config.pad_token_id, train_config.max_seq_len),
            drop_last=True,
        )
        print(f"  Training samples: {len(train_dataset)}")
    except FileNotFoundError:
        print(f"\n  [WARNING] Training data not found at {data_config.tinystories_train_file}")
        print("  Run data preparation first: scripts/02_prepare_data.sh")
        print("  Using dummy data for testing...\n")
        train_loader = _create_dummy_loader(train_config, model_config)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    # 学习率调度: cosine + warmup
    def get_lr(step):
        if step < train_config.warmup_steps:
            return train_config.learning_rate * (step / max(1, train_config.warmup_steps))
        progress = (step - train_config.warmup_steps) / max(1, train_config.max_steps - train_config.warmup_steps)
        return train_config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    # 混合精度
    scaler = torch.amp.GradScaler("cuda") if train_config.use_amp and device == "cuda" else None

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0
    best_loss = float("inf")

    print(f"\n  Training started...\n")
    pbar = tqdm(total=train_config.max_steps, desc="Pretrain")

    while global_step < train_config.max_steps:
        for batch in train_loader:
            if global_step >= train_config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 学习率
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1
            total_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr:.2e}"})

            # Logging
            if global_step % train_config.log_every == 0:
                avg_loss = total_loss / train_config.log_every
                perplexity = math.exp(avg_loss)
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/perplexity", perplexity, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                total_loss = 0.0

                # Expert utilization
                if hasattr(output, "expert_counts") and output["expert_counts"]:
                    for i, counts in enumerate(output["expert_counts"]):
                        writer.add_histogram(f"experts/layer_{i}", counts.float(), global_step)

            # Save checkpoint
            if global_step % train_config.save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step-{global_step:06d}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": model_config,
                }, checkpoint_path)
                print(f"\n  Checkpoint saved: {checkpoint_path}")

    pbar.close()
    writer.close()

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "config": model_config,
    }, final_path)
    print(f"\n  Final model saved: {final_path}")
    print(f"  Pretraining complete!\n")

    return model


def _create_dummy_loader(train_config, model_config):
    """创建dummy数据用于测试"""
    dummy_data = []
    for _ in range(1000):
        # 随机token序列
        tokens = torch.randint(4, 1000, (train_config.max_seq_len,))
        dummy_data.append({"input_ids": tokens})

    return DataLoader(
        dummy_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: pretrain_collate_fn(b, model_config.pad_token_id, train_config.max_seq_len),
        drop_last=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain mini-DeepSeek model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    model_cfg = DeepSeekConfig()
    train_cfg = PretrainConfig()
    if args.steps:
        train_cfg.max_steps = args.steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size

    pretrain(
        model_config=model_cfg,
        train_config=train_cfg,
        device=args.device,
    )
