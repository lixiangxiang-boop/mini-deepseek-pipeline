"""
自回归文本生成 — 使用MLAS的潜在KV缓存

提供:
  - 贪婪解码 (temperature=0)
  - 温度采样
  - Top-K / Top-P filtering
  - KV缓存加速

Usage:
  from src.inference.generate import generate_text
  output = generate_text(model, prompt_ids, max_new_tokens=200)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from .mlk_kv_cache import LatentKVCache


@torch.no_grad()
def generate_text(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
    use_cache: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """自回归文本生成

    Args:
        model: DeepSeekForCausalLM 实例
        input_ids: [batch, prompt_len] prompt token ids
        max_new_tokens: 最多生成token数
        temperature: 采样温度 (0 = 贪婪)
        top_p: nucleus sampling
        top_k: top-k sampling
        eos_token_id: 终止token
        pad_token_id: 忽略的token
        use_cache: 是否使用KV缓存 (推荐True)
        verbose: 是否打印生成过程

    Returns:
        [batch, prompt_len + generated_len] 完整token序列
    """
    model.eval()
    device = next(model.parameters()).device
    batch = input_ids.shape[0]

    generated = input_ids.clone()

    if use_cache:
        model.model.reset_kv_cache()

    for step in range(max_new_tokens):
        # Prefill: 第一步处理完整prompt
        # Decode: 后续步骤只处理最新token
        if step == 0 or not use_cache:
            current = generated
        else:
            current = generated[:, -1:]

        output = model(current, use_cache=use_cache)
        logits = output["logits"][:, -1, :]  # [batch, vocab]

        # 采样
        if temperature > 0 and temperature is not None:
            logits = logits / temperature

            # Top-K
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1:]
                logits[logits < threshold] = -float("inf")

            # Top-P (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs > top_p] = -float("inf")
                # 恢复原始顺序
                logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
                logits[logits == 0] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            # 处理全-inf的情况 (所有token被过滤)
            if torch.isnan(probs).any():
                probs = F.softmax(logits.masked_fill(logits == -float("inf"), -1e9), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        if verbose:
            print(f"  Step {step}: token={next_token.item()}")

        # EOS检查: 所有batch都生成EOS则停止
        if (next_token == eos_token_id).all():
            break

    if use_cache:
        model.model.reset_kv_cache()

    return generated


def generate_with_sampling_params(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
) -> list:
    """生成多个候选回答 (用于GRPO的G个候选生成)

    Args:
        model: DeepSeekForCausalLM
        input_ids: prompt
        num_return_sequences: 返回多少个不同的回答 (类似GRPO中的G)
        temperature: 温度 (越高越多变)
        其他参数同generate_text

    Returns:
        list of tensors, 每个 [1, seq_len]
    """
    results = []
    for _ in range(num_return_sequences):
        output = generate_text(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        results.append(output)
    return results
