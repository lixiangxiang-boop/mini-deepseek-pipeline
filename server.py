"""
mini-deepseek-pipeline 可视化服务

FastAPI 后端，提供:
  - 训练任务管理 (启动/停止/状态)
  - 实时指标 SSE 推送
  - 模型推理 API
  - 管道评估 API
  - 系统状态监控

Usage:
  python server.py --port 8080
"""

import os
import sys
import json
import time
import math
import asyncio
import argparse
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.model.deepseek_config import DeepSeekConfig
from src.model.deepseek_for_lm import DeepSeekForCausalLM
from src.inference.generate import generate_text
from src.reward.math_reward import compute_math_reward
from src.tokenizer.tokenizer_utils import get_tokenizer, SimpleTokenizer
from configs.training_config import PretrainConfig, SFTConfig, GRPOConfig

# ============================================================
# 全局状态
# ============================================================

app = FastAPI(title="Mini-DeepSeek Dashboard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 训练状态
training_state = {
    "running": False,
    "stage": None,
    "step": 0,
    "max_steps": 0,
    "loss": 0.0,
    "learning_rate": 0.0,
    "tokens_per_sec": 0.0,
    "elapsed": 0.0,
    "expert_utilization": [],
    "history": [],  # [(step, loss, lr, ...)]
}

# SSE 客户端
sse_clients: List[asyncio.Queue] = []

# 模型缓存
_model_cache: Optional[DeepSeekForCausalLM] = None
_model_device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 工具函数
# ============================================================

def get_device_info() -> dict:
    """获取设备信息"""
    info = {"platform": sys.platform, "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        info["cuda"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["gpu_memory_used"] = torch.cuda.memory_allocated(0) / 1e9
        info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / 1e9
    else:
        info["cuda"] = False
    info["cpu_count"] = os.cpu_count()
    return info


def load_model(device: str = None) -> DeepSeekForCausalLM:
    """加载模型 (带缓存)"""
    global _model_cache, _model_device
    if device is None:
        device = _model_device

    if _model_cache is not None:
        return _model_cache

    # 使用 tokenizer 的实际 vocab_size
    tokenizer = get_tokenizer()
    config = DeepSeekConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = DeepSeekForCausalLM(config).to(device)
    model.eval()

    # 尝试加载已有的checkpoint
    checkpoint_paths = [
        os.path.join(PROJECT_ROOT, "checkpoints", "sft_math", "final.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "quicktrain", "final.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "cold_sft", "final.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "pretrain", "final.pt"),
    ]
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Loaded checkpoint from {ckpt}")
            break
    else:
        print("No checkpoint found, using random weights")

    _model_cache = model
    return model


async def broadcast_sse(data: dict):
    """向所有 SSE 客户端推送数据"""
    for client_queue in sse_clients[:]:
        try:
            await client_queue.put(data)
        except Exception:
            sse_clients.remove(client_queue)


def run_training_thread(stage: str, steps: int, device: str):
    """在后台线程中运行训练，通过队列发送实时指标"""
    global training_state
    training_state["running"] = True
    training_state["stage"] = stage
    training_state["step"] = 0
    training_state["max_steps"] = steps
    training_state["history"] = []

    metrics_queue = queue.Queue()

    def training_callback(step: int, loss: float, lr: float, expert_counts=None):
        """训练回调 — 将指标放入队列"""
        elapsed = time.time() - t0
        metrics_queue.put({
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "elapsed": elapsed,
            "tokens_per_sec": step * 32 / max(elapsed, 0.1),
            "expert_utilization": expert_counts.tolist() if expert_counts is not None else [],
        })

    t0 = time.time()

    try:
        config = DeepSeekConfig()
        model = DeepSeekForCausalLM(config).to(device)

        if stage == "pretrain":
            train_cfg = PretrainConfig()
            train_cfg.max_steps = steps
            train_cfg.batch_size = 16
            _run_pretrain_quick(model, config, train_cfg, device, training_callback)
        elif stage == "sft":
            train_cfg = SFTConfig()
            train_cfg.max_steps = steps
            train_cfg.batch_size = 8
            _run_sft_quick(model, config, train_cfg, device, training_callback)
        elif stage == "grpo":
            train_cfg = GRPOConfig()
            train_cfg.max_steps = steps
            train_cfg.batch_size = 2
            _run_grpo_quick(model, config, train_cfg, device, training_callback)

        # 处理回调队列中的剩余数据
        while not metrics_queue.empty():
            m = metrics_queue.get_nowait()
            training_state["step"] = m["step"]
            training_state["loss"] = m["loss"]
            training_state["learning_rate"] = m["learning_rate"]
            training_state["elapsed"] = m["elapsed"]
            training_state["tokens_per_sec"] = m["tokens_per_sec"]
            training_state["history"].append(m)

    except Exception as e:
        training_state["error"] = str(e)
    finally:
        training_state["running"] = False
        training_state["stage"] = None


def _run_pretrain_quick(model, config, train_cfg, device, callback):
    """快速预训练循环"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    model.train()

    for step in range(train_cfg.max_steps):
        batch = torch.randint(4, 1000, (train_cfg.batch_size, 128)).to(device)
        labels = batch.clone()

        output = model(batch, labels=labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lr = train_cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * step / train_cfg.max_steps))

        if step % 5 == 0:
            expert_counts = output.get("expert_counts", [None])[0] if output.get("expert_counts") else None
            callback(step, loss.item(), lr, expert_counts)


def _run_sft_quick(model, config, train_cfg, device, callback):
    """快速 SFT 训练循环"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    model.train()

    for step in range(train_cfg.max_steps):
        batch = torch.randint(4, 1000, (train_cfg.batch_size, 128)).to(device)
        labels = batch.clone()

        output = model(batch, labels=labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lr = train_cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * step / train_cfg.max_steps))

        if step % 5 == 0:
            callback(step, loss.item(), lr)


def _run_grpo_quick(model, config, train_cfg, device, callback):
    """快速 GRPO 训练循环 (简化为PPO-style)"""
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    model.train()

    for step in range(train_cfg.max_steps):
        batch = torch.randint(4, 1000, (train_cfg.batch_size, 128)).to(device)
        labels = batch.clone()

        output = model(batch, labels=labels)
        loss = output["loss"]

        # 简化的 KL 惩罚
        with torch.no_grad():
            ref_output = ref_model(batch, labels=labels)
        kl = ((output["logits"] - ref_output["logits"]) ** 2).mean() * 0.01

        total_loss = loss + kl
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            callback(step, loss.item(), train_cfg.learning_rate)


# ============================================================
# REST API
# ============================================================

@app.get("/api/status")
async def api_status():
    """系统状态"""
    return {
        "device": get_device_info(),
        "training": {
            "running": training_state["running"],
            "stage": training_state["stage"],
            "step": training_state["step"],
            "max_steps": training_state["max_steps"],
            "loss": training_state["loss"],
            "learning_rate": training_state["learning_rate"],
            "tokens_per_sec": training_state["tokens_per_sec"],
            "elapsed": training_state["elapsed"],
        },
        "checkpoints": _list_checkpoints(),
    }


@app.get("/api/model/info")
async def api_model_info():
    """模型架构信息"""
    config = DeepSeekConfig()
    model = DeepSeekForCausalLM(config)
    return {
        "total_params": model.get_num_parameters(),
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "d_head": config.d_head,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "use_mla": config.use_mla,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_rope_dim": config.qk_rope_dim,
            "use_moe": config.use_moe,
            "num_experts": config.num_experts,
            "num_shared_experts": config.num_shared_experts,
            "top_k": config.top_k,
            "expert_intermediate_size": config.expert_intermediate_size,
            "use_mtp": config.use_mtp,
            "num_pred_tokens": config.num_pred_tokens,
        },
        "architecture": {
            "layers": [
                {
                    "name": "Embedding",
                    "params": config.vocab_size * config.hidden_size,
                },
                {
                    "name": f"Decoder Layer ×{config.num_layers}",
                    "sub_layers": [
                        {"name": "MLA Attention", "details": f"q_lora={config.q_lora_rank}, kv_lora={config.kv_lora_rank}"},
                        {"name": "DeepSeekMoE", "details": f"{config.num_experts} routed + {config.num_shared_experts} shared experts"},
                    ],
                },
                {
                    "name": "Final RMSNorm",
                    "params": config.hidden_size,
                },
                {
                    "name": "LM Head (weight-tied)",
                    "params": 0,
                },
            ],
        },
        "kv_cache_compression": {
            "mla_per_layer_kb": config.kv_lora_rank * 512 * 2 / 1024,
            "mha_per_layer_kb": config.num_heads * config.d_head * 2 * 512 * 2 / 1024,
            "compression_ratio": (config.num_heads * config.d_head * 2) / config.kv_lora_rank,
        },
    }


@app.post("/api/training/start")
async def api_start_training(stage: str = "pretrain", steps: int = 100):
    """启动训练"""
    if training_state["running"]:
        raise HTTPException(400, "Training already running")

    training_state["error"] = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    thread = threading.Thread(
        target=run_training_thread,
        args=(stage, steps, device),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "stage": stage, "steps": steps}


@app.post("/api/training/stop")
async def api_stop_training():
    """停止训练"""
    training_state["running"] = False
    return {"status": "stopped"}


@app.get("/api/training/metrics/stream")
async def api_metrics_stream():
    """SSE 实时指标推送"""
    async def event_stream():
        client_queue = asyncio.Queue()
        sse_clients.append(client_queue)
        try:
            while True:
                data = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.TimeoutError:
            pass
        finally:
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/training/history")
async def api_training_history():
    """获取训练历史"""
    return training_state["history"]


@app.post("/api/inference/generate")
async def api_inference(prompt: str = "What is 15 + 27?", max_tokens: int = 100, temperature: float = 0.7):
    """模型推理 — 使用真正的 tokenizer 编解码"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(device)
        tokenizer = get_tokenizer()

        # Encode prompt with real tokenizer
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([token_ids], device=device)

        # Truncate if needed
        config = DeepSeekConfig()
        if input_ids.shape[1] > config.max_seq_len:
            input_ids = input_ids[:, :config.max_seq_len]

        t0 = time.time()
        output_ids = generate_text(
            model, input_ids,
            max_new_tokens=min(max_tokens, 256),
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.special_ids.get("<eos>", 2),
            use_cache=True,
        )
        elapsed = time.time() - t0

        # Decode generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)

        return {
            "prompt": prompt,
            "generated": generated,
            "tokens_generated": len(new_ids),
            "time_seconds": round(elapsed, 3),
            "tokens_per_sec": round(len(new_ids) / max(elapsed, 0.001), 1),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/evaluation/compare")
async def api_evaluation_compare():
    """管道阶段对比"""
    stages = [
        {"name": "1. Pretrained Only", "accuracy": "2-5%", "description": "No instruction following — random answers"},
        {"name": "2. Cold-Start SFT", "accuracy": "10-15%", "description": "Learns CoT format and basic math reasoning"},
        {"name": "3. SFT + GRPO", "accuracy": "15-25%", "description": "RL improves reasoning through group-relative rewards"},
        {"name": "4. Full R1 Pipeline", "accuracy": "20-30%", "description": "Self-improvement loop: rejection sampling + retraining"},
    ]
    return {"stages": stages, "note": "Expected values for 56.6M parameter model"}


@app.get("/api/checkpoints")
async def api_checkpoints():
    """列出所有checkpoints"""
    return _list_checkpoints()


def _list_checkpoints() -> list:
    """列出 checkpoints 目录"""
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    result = []
    if os.path.exists(ckpt_dir):
        for stage in os.listdir(ckpt_dir):
            stage_dir = os.path.join(ckpt_dir, stage)
            if os.path.isdir(stage_dir):
                files = [f for f in os.listdir(stage_dir) if f.endswith(".pt")]
                for f in files:
                    fpath = os.path.join(stage_dir, f)
                    size_mb = os.path.getsize(fpath) / 1e6
                    result.append({
                        "stage": stage,
                        "file": f,
                        "path": fpath,
                        "size_mb": round(size_mb, 1),
                        "modified": time.ctime(os.path.getmtime(fpath)),
                    })
    return sorted(result, key=lambda x: x["modified"], reverse=True)


# ============================================================
# 静态文件服务
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Dashboard 主页"""
    static_dir = os.path.join(PROJECT_ROOT, "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard not found. Create static/index.html</h1>"


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-DeepSeek Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║      Mini-DeepSeek Dashboard                            ║
║      http://{args.host}:{args.port}                           ║
║                                                          ║
║      API Endpoints:                                      ║
║        GET  /api/status        System status             ║
║        GET  /api/model/info    Model architecture        ║
║        POST /api/training/start Start training           ║
║        POST /api/training/stop  Stop training            ║
║        GET  /api/training/metrics/stream  SSE stream     ║
║        POST /api/inference/generate  Model inference     ║
║        GET  /api/evaluation/compare  Pipeline compare    ║
║        GET  /api/checkpoints   List checkpoints          ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
