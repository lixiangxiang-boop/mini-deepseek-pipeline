"""
MTP 单元测试 — Multi-Token Prediction

验证:
  1. MTP head输出正确的logit形状
  2. 总loss = next_token_loss + lambda * sum(mtp_losses)
  3. 无MTP时行为退化为标准LM head

Usage:
  pytest tests/test_mtp.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.mtp_head import MultiTokenPredictionHead


def test_mtp_output_shapes():
    """测试MTP输出的形状"""
    head = MultiTokenPredictionHead(
        hidden_size=320,
        vocab_size=16384,
        num_pred_tokens=2,  # D=2: t+1 and t+2
        loss_weight=0.3,
    )

    batch, seq_len = 4, 64
    hidden = torch.randn(batch, seq_len, 320)
    labels = torch.randint(0, 16384, (batch, seq_len))

    result = head(hidden, labels=labels)

    # 主预测logits
    assert result["logits"].shape == (batch, seq_len, 16384)

    # MTP logits (辅助预测)
    assert "mtp_logits" in result
    assert len(result["mtp_logits"]) == 1  # D-1 = 1 extra head
    assert result["mtp_logits"][0].shape == (batch, seq_len, 16384)

    # Losses
    assert "loss" in result
    assert "next_token_loss" in result
    assert "mtp_loss" in result


def test_mtp_loss_composition():
    """测试MTP损失 = next_token + weighted * mtp"""
    head = MultiTokenPredictionHead(
        hidden_size=320,
        vocab_size=16384,
        num_pred_tokens=2,
        loss_weight=0.3,
    )

    hidden = torch.randn(4, 64, 320)
    labels = torch.randint(0, 16384, (4, 64))

    result = head(hidden, labels=labels)

    expected_loss = result["next_token_loss"] + 0.3 * result["mtp_loss"]
    assert torch.allclose(result["loss"], expected_loss, atol=1e-5), (
        f"Loss mismatch: {result['loss']:.4f} vs expected {expected_loss:.4f}"
    )


def test_mtp_without_mtp():
    """无MTP模式 (D=1) — 退化为标准LM head"""
    head = MultiTokenPredictionHead(
        hidden_size=320,
        vocab_size=16384,
        num_pred_tokens=1,  # 只有next-token
        loss_weight=0.0,
    )

    hidden = torch.randn(4, 64, 320)
    labels = torch.randint(0, 16384, (4, 64))

    result = head(hidden, labels=labels)

    # 应该只有logits和next_token_loss
    assert "logits" in result
    assert "mtp_logits" not in result or len(result.get("mtp_logits", [])) == 0
    assert result["loss"] == result["next_token_loss"]


def test_mtp_logits_non_nan():
    """测试MTP logits无NaN"""
    head = MultiTokenPredictionHead(
        hidden_size=320,
        vocab_size=16384,
        num_pred_tokens=2,
    )

    hidden = torch.randn(2, 32, 320)
    labels = torch.randint(0, 16384, (2, 32))

    result = head(hidden, labels=labels)

    assert not torch.isnan(result["logits"]).any()
    if "mtp_logits" in result:
        for mtp_logit in result["mtp_logits"]:
            assert not torch.isnan(mtp_logit).any()


if __name__ == "__main__":
    test_mtp_output_shapes()
    test_mtp_loss_composition()
    test_mtp_without_mtp()
    test_mtp_logits_non_nan()
    print("\n  All MTP tests passed!")
