"""
Multi-Token Prediction (MTP) — DeepSeek-V3 §2.4

MTP 让模型同时预测未来多个token，提供更密集的训练信号。

标准语言模型: 只预测 next token (position t+1)
MTP: 同时预测 next token (t+1), token after next (t+2), ..., (t+D)

原理:
  模型在位置t的输出hidden state除了预测t+1外，还用于预测t+2,...t+D
  每个额外预测位置有独立的输出头 (Linear layers)

DeepSeek-V3使用D=4 (预测4个future tokens)
我们使用D=2 (简化)

架构:
  hidden[t] -> LM_Head_1 -> logits[t+1] (next token)
  hidden[t] -> LM_Head_2 -> logits[t+2] (token after next)

Loss: L_mtp = L_next_token + λ * L_future_tokens
  其中 λ = 0.3 (V3的做法 — 主任务next-token为主导)

简化:
  论文中V3的MTP是顺序式的: hidden[t] -> 预测t+1 -> 将t+1的embedding注入 -> 预测t+2
  我们简化为并行式: hidden[t] 同时预测所有未来位置
  并行式在概念上等价但实现更简单，适合学习目的
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenPredictionHead(nn.Module):
    """Multi-Token Prediction Head

    Args:
        hidden_size: 隐藏层维度
        vocab_size: 词汇表大小
        num_pred_tokens: 预测的future token数量 (包含next token, 所以D>=2)
        loss_weight: 辅助预测loss权重 (默认0.3, 匹配V3)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_pred_tokens: int = 2,
        loss_weight: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_pred_tokens = num_pred_tokens  # D = 2, 预测 t+1 和 t+2
        self.loss_weight = loss_weight

        # 主预测头: t+1 (next token) — 这是标准的LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 辅助预测头: 对未来token (t+2, ..., t+D) 的预测
        # num_pred_tokens - 1 = D-1 个额外头
        if num_pred_tokens > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_pred_tokens - 1)
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            labels: [batch, seq_len] 目标token ids (shifted)

        Returns:
            dict with:
              - logits: [batch, seq_len, vocab_size] (主预测, t+1)
              - mtp_logits: List of [batch, seq_len, vocab_size] (辅助预测)
              - loss: total loss (训练时)
              - next_token_loss: 主任务loss
              - mtp_loss: 辅助任务loss (各头之和)
        """
        result = {}

        # 主预测: t+1 (next token prediction)
        logits = self.lm_head(hidden_states)
        result["logits"] = logits

        # 辅助预测: t+2, ..., t+D
        if self.num_pred_tokens > 1:
            mtp_logits = []
            for head in self.mtp_heads:
                mtp_logits.append(head(hidden_states))
            result["mtp_logits"] = mtp_logits

        # 计算损失
        if labels is not None:
            batch, seq_len, vocab = logits.shape

            # Next-token loss (标准的因果LM loss)
            next_token_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            result["next_token_loss"] = next_token_loss

            # MTP losses (辅助预测)
            total_mtp_loss = 0.0
            if self.num_pred_tokens > 1:
                mtp_losses = []
                for i, mtp_logit in enumerate(mtp_logits):
                    # 预测位置 t+i+2
                    # hidden[t] 预测 label[t+i+1]
                    shift = i + 2  # t -> t+2, t+3, ...
                    if seq_len > shift:
                        mtp_loss = F.cross_entropy(
                            mtp_logit[:, :seq_len - shift].reshape(-1, vocab),
                            labels[:, shift:].reshape(-1),
                            ignore_index=-100,
                        )
                        mtp_losses.append(mtp_loss)
                        total_mtp_loss += mtp_loss

                result["mtp_losses"] = mtp_losses if self.num_pred_tokens > 1 else []
                result["mtp_loss"] = total_mtp_loss

            # Total loss: 主任务 + 加权辅助任务
            total_loss = next_token_loss
            if self.num_pred_tokens > 1:
                total_loss = total_loss + self.loss_weight * total_mtp_loss
            result["loss"] = total_loss

        return result
