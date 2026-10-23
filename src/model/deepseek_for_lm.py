"""
DeepSeekForCausalLM — 语言模型训练/推理的顶层封装

组合:
  DeepSeekModel (骨干网络) + MultiTokenPredictionHead (输出头)

提供:
  - 训练: forward(input_ids, labels) -> loss
  - 推理: generate(prompt) -> tokens
  - 参数统计与设备管理
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .deepseek_model import DeepSeekModel
from .mtp_head import MultiTokenPredictionHead


class DeepSeekForCausalLM(nn.Module):
    """DeepSeek Causal LM 完整模型

    用法:
        model = DeepSeekForCausalLM(config)
        model.to("cuda")

        # 训练
        output = model(input_ids, labels=labels)
        loss = output["loss"]
        loss.backward()

        # 推理
        generated = model.generate(prompt_ids, max_new_tokens=100)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 骨干网络
        self.model = DeepSeekModel(config)

        # 输出头 (MTP)
        self.head = MultiTokenPredictionHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_pred_tokens=config.num_pred_tokens if config.use_mtp else 1,
            loss_weight=config.mtp_loss_weight if config.use_mtp else 0.0,
        )

        # 权重绑定: LM head 与 embedding 共享权重
        self.head.lm_head.weight = self.model.embed_tokens.weight

        # 总参数量
        self._num_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> dict:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] 训练标签 (-100 for ignored positions)
            use_cache: 是否使用KV缓存

        Returns:
            dict: {
                "logits": [B, L, V],
                "loss": scalar (if labels provided),
                "next_token_loss": scalar,
                "mtp_loss": scalar (if use_mtp),
                ...
            }
        """
        # 骨干网络前向
        hidden_states, kv_caches, expert_counts = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        # 输出头
        result = self.head(hidden_states, labels=labels)

        # 附加信息
        result["kv_caches"] = kv_caches
        result["expert_counts"] = expert_counts

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """自回归文本生成

        Args:
            input_ids: [batch, seq_len] prompt tokens
            max_new_tokens: 最多生成token数
            temperature: 采样温度 (0=贪婪)
            top_p: nucleus sampling
            top_k: top-k sampling
            eos_token_id: 终止token
            pad_token_id: 填充token

        Returns:
            generated_ids: [batch, seq_len + generated_len]
        """
        self.eval()
        self.model.reset_kv_cache()

        batch = input_ids.shape[0]
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            # 只前向传播最新token (使用KV缓存)
            if step == 0:
                current = generated
            else:
                current = generated[:, -1:]

            output = self.forward(current, use_cache=True)
            logits = output["logits"][:, -1, :]  # [B, V]

            # 温度缩放
            if temperature > 0:
                logits = logits / temperature
                logits_raw = logits.clone()  # 保存未过滤的logits用于fallback
                # Top-K filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_values[:, -1:]] = -float('inf')
                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_logits[cumulative_probs > top_p] = -float('inf')
                    logits = torch.zeros_like(logits).scatter_(
                        -1, sorted_indices, sorted_logits
                    )
                    logits[logits == 0] = -float('inf')

                probs = torch.softmax(logits, dim=-1)
                # 处理全-inf边界情况 (所有token被top-p/top-k过滤)
                if torch.isnan(probs).any() or (probs.sum(-1) == 0).any():
                    # Fallback: 使用未过滤的logits
                    safe_probs = torch.softmax(logits_raw / temperature, dim=-1)
                    probs = torch.where(probs.isnan() | (probs.sum(-1, keepdim=True) == 0),
                                       safe_probs, probs)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # 检查EOS
            if (next_token == eos_token_id).all():
                break

        self.model.reset_kv_cache()
        return generated

    def get_num_parameters(self) -> int:
        """返回总参数量"""
        return self._num_params

    def print_model_summary(self):
        """打印模型结构总结"""
        print(f"\n{'='*60}")
        print(f"  DeepSeekForCausalLM — Model Summary")
        print(f"{'='*60}")
        print(f"  Total parameters:     {self._num_params:,} (~{self._num_params/1e6:.1f}M)")
        print(f"  Hidden size:          {self.config.hidden_size}")
        print(f"  Num layers:           {self.config.num_layers}")
        print(f"  Num heads:            {self.config.num_heads}")
        print(f"  Head dim:             {self.config.d_head}")
        print(f"  Vocab size:           {self.config.vocab_size}")
        print(f"  Max seq len:          {self.config.max_seq_len}")
        print(f"  MLA:                  {'enabled' if self.config.use_mla else 'disabled'}")
        if self.config.use_mla:
            print(f"    q_lora_rank:        {self.config.q_lora_rank}")
            print(f"    kv_lora_rank:       {self.config.kv_lora_rank}")
            print(f"    qk_rope_dim:        {self.config.qk_rope_dim}")
        print(f"  MoE:                  {'enabled' if self.config.use_moe else 'disabled'}")
        if self.config.use_moe:
            print(f"    num_experts:        {self.config.num_experts}")
            print(f"    shared_experts:     {self.config.num_shared_experts}")
            print(f"    top_k:              {self.config.top_k}")
        print(f"  MTP:                  {'enabled' if self.config.use_mtp else 'disabled'}")
        if self.config.use_mtp:
            print(f"    num_pred_tokens:    {self.config.num_pred_tokens}")
            print(f"    mtp_loss_weight:    {self.config.mtp_loss_weight}")
        print(f"{'='*60}\n")
