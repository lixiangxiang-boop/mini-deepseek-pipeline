"""
数据整理器 — 将raw样本转换为训练batch

各阶段不同的collate逻辑:
  - pretrain: padding + causal LM labels
  - SFT: padding + full sequence labels (所有token都参与loss)
  - GRPO: 只整理prompts (生成在GRPO trainer内部完成)
"""

import torch
from typing import List, Dict


def pad_sequences(
    sequences: List[torch.Tensor],
    pad_token_id: int,
    max_len: int = 512,
) -> Dict[str, torch.Tensor]:
    """将变长序列padding到相同长度

    Args:
        sequences: 每个序列的token ids
        pad_token_id: 填充token
        max_len: 最大长度 (超过则截断)

    Returns:
        {"input_ids": [batch, max_len], "attention_mask": [batch, max_len]}
    """
    batch = len(sequences)

    # 找到batch内的最大长度 (不超过max_len)
    batch_max_len = min(max_len, max(len(s) for s in sequences))

    input_ids = torch.full((batch, batch_max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, batch_max_len), dtype=torch.float)

    for i, seq in enumerate(sequences):
        seq = seq[:batch_max_len]
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1.0

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def create_causal_lm_labels(input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """创建因果语言模型的labels (shift by 1)

    labels = input_ids shifted left by 1 position
    padding positions (-100) are ignored in loss computation
    """
    labels = input_ids.clone()
    labels[input_ids == 0] = ignore_index  # 忽略padding
    return labels


def pretrain_collate_fn(
    batch: List[Dict],
    pad_token_id: int,
    max_seq_len: int = 512,
) -> Dict[str, torch.Tensor]:
    """预训练batch整理

    Returns:
        {"input_ids", "attention_mask", "labels"}
    """
    sequences = [item["input_ids"].clone().detach() for item in batch]
    result = pad_sequences(sequences, pad_token_id, max_seq_len)
    result["labels"] = create_causal_lm_labels(result["input_ids"])
    return result


def sft_collate_fn(
    batch: List[Dict],
    pad_token_id: int,
    max_seq_len: int = 512,
) -> Dict[str, torch.Tensor]:
    """SFT batch整理 — 与pretrain相同，但所有token参与loss"""
    return pretrain_collate_fn(batch, pad_token_id, max_seq_len)


def grpo_collate_fn(
    batch: List[Dict],
    pad_token_id: int,
    max_seq_len: int = 256,
) -> Dict[str, torch.Tensor]:
    """GRPO batch整理 — 只整理prompts (生成由trainer处理)"""
    sequences = [item["input_ids"].clone().detach() for item in batch]
    result = pad_sequences(sequences, pad_token_id, max_seq_len)
    # GRPO不需要labels
    return result
