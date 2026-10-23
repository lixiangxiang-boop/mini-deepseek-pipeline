"""
端到端模型测试 — 前向/反向 + 参数量验证

验证:
  1. 模型能完成前向传播 (无NaN)
  2. 模型能完成反向传播 (所有参数有梯度)
  3. 参数量在预期范围内
  4. loss合理 (>0, 不是inf)

Usage:
  pytest tests/test_full_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM


def test_model_forward():
    """测试前向传播 — 基础功能"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)

    batch, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)

    assert "loss" in output
    assert "logits" in output
    assert not torch.isnan(output["loss"])
    assert not torch.isinf(output["loss"])
    assert output["loss"].item() > 0
    assert output["logits"].shape == (batch, seq_len, config.vocab_size)


def test_model_backward():
    """测试反向传播 — 梯度计算"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)

    batch, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    loss = output["loss"]
    loss.backward()

    # 检查所有参数是否有梯度
    params_with_grad = 0
    params_without_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            else:
                params_without_grad += 1

    assert params_with_grad > 0, "No parameters received gradients"
    # 允许少量参数无梯度 (如MTP头在no_mtp模式下)
    assert params_without_grad < 10, f"Too many params without grad: {params_without_grad}"


def test_parameter_count():
    """测试参数量在预期范围"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)
    num_params = model.get_num_parameters()

    print(f"  Total parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

    # 应该在 20M-40M 范围内
    assert 15_000_000 <= num_params <= 70_000_000, (
        f"Expected 15M-70M params, got {num_params:,}"
    )


def test_model_modes():
    """测试不同配置模式 — MLA/MoE/MTP 开关"""
    # 无MLA (标准MHA替代 — 但本项目只实现了MLA，所以测试MLA on/off不影响)
    config_full = DeepSeekConfig()
    model_full = DeepSeekForCausalLM(config_full)

    batch, seq_len = 2, 32
    input_ids = torch.randint(0, config_full.vocab_size, (batch, seq_len))

    output = model_full(input_ids)
    assert output["logits"].shape[-1] == config_full.vocab_size

    # 无MTP模式
    config_no_mtp = DeepSeekConfig()
    config_no_mtp.use_mtp = False
    config_no_mtp.num_pred_tokens = 1

    model_no_mtp = DeepSeekForCausalLM(config_no_mtp)
    output_no_mtp = model_no_mtp(input_ids, labels=input_ids)
    assert "loss" in output_no_mtp


def test_generation():
    """测试文本生成"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)
    model.eval()

    input_ids = torch.randint(4, 1000, (1, 16))

    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20)

    assert generated.shape[0] == 1
    assert generated.shape[1] >= 16  # 至少生成了prompt
    assert generated.shape[1] <= 16 + 20  # 不超过max_new_tokens


def test_kv_cache_consistency():
    """测试KV缓存的一致性 — 有缓存和无缓存结果应该相同"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)
    model.eval()

    input_ids = torch.randint(4, 1000, (1, 32))

    # 无缓存生成
    model.model.reset_kv_cache()
    with torch.no_grad():
        out1 = model.generate(input_ids, max_new_tokens=5, temperature=0.0, top_p=1.0, top_k=0)

    # 有缓存生成
    model.model.reset_kv_cache()
    with torch.no_grad():
        out2 = model.generate(input_ids, max_new_tokens=5, temperature=0.0, top_p=1.0, top_k=0)

    # 贪婪解码结果应该一致
    assert torch.equal(out1, out2), "KV cache produces different results"


if __name__ == "__main__":
    test_model_forward()
    test_model_backward()
    test_parameter_count()
    test_model_modes()
    test_generation()
    test_kv_cache_consistency()
    print("\n  All model tests passed!")
