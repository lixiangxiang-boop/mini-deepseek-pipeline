"""
交互式REPL — 测试训练好的模型

用于手动测试数学推理能力:
  $ python -m src.inference.interactive --checkpoint checkpoints/final.pt

Usage:
  > What is 15 + 27?
  <think> 15 + 27 = 42 </think>
  <answer> 42 </answer>
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.inference.generate import generate_text


def interactive(
    checkpoint_path: str,
    device: str = "cuda",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
):
    """启动交互式REPL

    Args:
        checkpoint_path: 模型checkpoint路径
        device: 推理设备
        temperature: 生成温度
        max_new_tokens: 最大生成长度
    """
    # 加载模型
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", DeepSeekConfig())
    model = DeepSeekForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    model.print_model_summary()
    print(f"  Temperature: {temperature}")
    print(f"  Max new tokens: {max_new_tokens}")

    print(f"\n{'='*50}")
    print(f"  Mini-DeepSeek Interactive REPL")
    print(f"  Type a math problem, or 'quit' to exit")
    print(f"  Type 'temp=0.5' to change temperature")
    print(f"  Type 'tokens=100' to change max tokens")
    print(f"{'='*50}\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # 更改温度
        if user_input.lower().startswith("temp="):
            try:
                temperature = float(user_input.split("=")[1])
                print(f"Temperature set to {temperature}")
            except ValueError:
                print("Invalid temperature value")
            continue

        # 更改max tokens
        if user_input.lower().startswith("tokens="):
            try:
                max_new_tokens = int(user_input.split("=")[1])
                print(f"Max tokens set to {max_new_tokens}")
            except ValueError:
                print("Invalid value")
            continue

        # 格式化prompt
        prompt = f"<bos> Question: {user_input}\n<think>"

        # Tokenize (简化 — 使用随机token模拟演示)
        # 完整实现需要加载tokenizer
        from configs.data_config import DataConfig
        dc = DataConfig()

        # 简易tokenize: 用ASCII值模拟
        prompt_ids = torch.tensor(
            [[ord(c) % (config.vocab_size - 10) + 10 for c in prompt[:config.max_seq_len]]],
            device=device,
        )

        print(f"\n  Generating...")
        try:
            output = generate_text(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=config.eos_token_id,
                use_cache=True,
            )

            # 简易解码
            generated_part = output[0, prompt_ids.shape[1]:]
            chars = [chr(t.item() % 128) for t in generated_part if 32 <= t.item() % 128 < 127]
            generated_text = "".join(chars)

            print(f"\n  {generated_text[:500]}")
            print()
        except Exception as e:
            print(f"  Generation error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive mini-DeepSeek REPL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    interactive(
        checkpoint_path=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
