"""
Supervised Fine-Tuning (SFT) — Cold-Start阶段

对应DeepSeek-R1 §2.2.1 "Cold Start"阶段:
  - 使用人工标注的高质量CoT数据进行监督微调
  - 让模型学会:
    1. 生成正确格式的CoT推理 (<think>...</think><answer>...</answer>)
    2. 基础数学推理能力
  - 这是R1管道的起点: 先有一个基础好的模型，再通过RL提升

数据格式:
  <bos> Question: {question}
  <think> {step-by-step reasoning} </think>
  <answer> {final answer} </answer> <eos>
"""

import os
import sys
import math
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.training.data_collator import sft_collate_fn
from configs.training_config import SFTConfig
from configs.data_config import DataConfig


class SFTDataset(Dataset):
    """SFT数据集 — GSM8K CoT格式"""

    def __init__(self, data_file: str, max_seq_len: int = 512):
        self.data = torch.load(data_file)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_seq_len]
        return {"input_ids": tokens}


def sft_train(
    model: Optional[DeepSeekForCausalLM] = None,
    pretrained_checkpoint: Optional[str] = None,
    model_config: DeepSeekConfig = None,
    train_config: SFTConfig = None,
    data_config: DataConfig = None,
    checkpoint_dir: str = None,
    log_dir: str = None,
    device: str = "cuda",
):
    """运行SFT训练

    如果提供了pretrained_checkpoint，则从预训练checkpoint加载。
    否则需要传入已初始化的model。
    """
    if model_config is None:
        model_config = DeepSeekConfig()
    if train_config is None:
        train_config = SFTConfig()
    if data_config is None:
        data_config = DataConfig()
    if checkpoint_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoint_dir = os.path.join(project_root, "checkpoints", "cold_sft")
    if log_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, "outputs", "cold_sft")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    print("=" * 60)
    print("  Cold-Start SFT — GSM8K Chain-of-Thought")
    print("=" * 60)

    # 加载/创建模型
    if model is None:
        model = DeepSeekForCausalLM(model_config)

        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            print(f"  Loading pretrained weights from {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"  Loaded from step {checkpoint.get('step', 'unknown')}")
        else:
            print("  Starting from scratch (no pretrained checkpoint)")

    model = model.to(device)
    model.train()

    # 数据
    try:
        train_dataset = SFTDataset(data_config.sft_train_file, train_config.max_seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=lambda b: sft_collate_fn(b, model_config.pad_token_id, train_config.max_seq_len),
            drop_last=True,
        )
        print(f"  Training samples: {len(train_dataset)}")
    except FileNotFoundError:
        print(f"\n  [WARNING] SFT data not found at {data_config.sft_train_file}")
        print("  Run data preparation first")
        print("  Using dummy data for testing...\n")
        train_loader = _create_dummy_sft_loader(train_config, model_config)

    # 优化器 (SFT使用较小的学习率)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR schedule
    def get_lr(step):
        if step < train_config.warmup_steps:
            return train_config.learning_rate * (step / max(1, train_config.warmup_steps))
        progress = (step - train_config.warmup_steps) / max(1, train_config.max_steps - train_config.warmup_steps)
        return train_config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler("cuda") if train_config.use_amp and device == "cuda" else None

    global_step = 0
    total_loss = 0.0

    print(f"\n  Training started...\n")
    pbar = tqdm(total=train_config.max_steps, desc="SFT")

    while global_step < train_config.max_steps:
        for batch in train_loader:
            if global_step >= train_config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

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

            if global_step % train_config.log_every == 0:
                avg_loss = total_loss / train_config.log_every
                writer.add_scalar("sft/loss", avg_loss, global_step)
                writer.add_scalar("sft/perplexity", math.exp(avg_loss), global_step)
                writer.add_scalar("sft/lr", lr, global_step)
                total_loss = 0.0

            if global_step % train_config.save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step-{global_step:06d}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": model_config,
                }, checkpoint_path)

    pbar.close()
    writer.close()

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "config": model_config,
    }, final_path)
    print(f"\n  SFT model saved: {final_path}")
    print(f"  Cold-Start SFT complete!\n")

    return model


def _create_dummy_sft_loader(train_config, model_config):
    dummy_data = []
    for _ in range(50):
        tokens = torch.randint(4, 1000, (train_config.max_seq_len,))
        dummy_data.append({"input_ids": tokens})
    return DataLoader(
        dummy_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, model_config.pad_token_id, train_config.max_seq_len),
        drop_last=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT train mini-DeepSeek model")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    train_cfg = SFTConfig()
    if args.steps:
        train_cfg.max_steps = args.steps

    sft_train(
        pretrained_checkpoint=args.pretrained,
        train_config=train_cfg,
        device=args.device,
    )
