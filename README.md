# mini-deepseek-pipeline

基于 DeepSeek 论文(V2/V3/R1)的全链路大模型开发学习项目。  
从零实现 MLA、MoE、GRPO 等核心方法，小模型(~27M参数)，适用于 RTX 4060 8GB。

## DeepSeek 方法全景图

| 论文 | 核心创新 | 本项目对应 | 行数 |
|------|---------|-----------|------|
| **V2** §2.1.2 | MLA (Multi-head Latent Attention) | `src/model/mla_attention.py` | ~150 |
| **V2** §2.2 | DeepSeekMoE (细粒度专家+共享专家) | `src/model/deepseek_moe.py` | ~170 |
| **V3** §2.2 | 辅助损失自由的负载均衡 | `src/model/load_balancer.py` | ~60 |
| **V3** §2.4 | Multi-Token Prediction (MTP) | `src/model/mtp_head.py` | ~100 |
| **R1** §2.2.3 | GRPO (Group Relative Policy Optimization) | `src/training/grpo_trainer.py` | ~300 |
| **R1** §2.2.2 | 拒绝采样 (Rejection Sampling) | `src/training/rejection_sampling.py` | ~150 |
| **R1** Fig2 | 全训练管道 (SFT→GRPO→RS→SFT→GRPO) | `run_pipeline.py` | ~150 |

## 快速开始

### 1. 安装依赖

```bash
# CUDA版PyTorch (重要!)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 项目依赖
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# 完整管道 (~2小时)
python run_pipeline.py

# 快速测试 (~15分钟，减少步数)
python run_pipeline.py --quick --skip-tests

# 指定阶段
python run_pipeline.py --stage pretrain
python run_pipeline.py --stage eval
```

### 3. 分步运行

```bash
# Step 1: 数据准备
python -m src.training.prepare_data

# Step 2: 预训练 (~20 min)
python -m src.training.pretrain --device cuda

# Step 3: Cold-Start SFT (~8 min)
python -m src.training.sft_trainer --pretrained checkpoints/pretrain/final.pt --device cuda

# Step 4: GRPO (~30 min)
python -m src.training.grpo_trainer --sft_checkpoint checkpoints/cold_sft/final.pt --device cuda

# Step 5: 拒绝采样 (~15 min)
python -m src.training.rejection_sampling --checkpoint checkpoints/grpo_r1/final.pt --output data/prepared/rejection_sft/

# Step 6: 评估
python -m src.evaluation.compare_pipelines --device cuda
```

### 4. 运行测试

```bash
# 模型测试
python tests/test_full_model.py
python tests/test_mla.py

# MoE测试
python tests/test_moe.py

# GRPO测试
python tests/test_grpo.py
```

## 项目结构

```
mini-deepseek-pipeline/
├── configs/              # 配置 (模型/训练/数据)
├── src/
│   ├── model/            # ★ 核心架构 (MLA, MoE, MTP)
│   ├── training/         # 训练管道 (pretrain, SFT, GRPO, RS)
│   ├── reward/           # 奖励函数
│   ├── inference/        # 推理 (KV缓存, 生成)
│   └── evaluation/       # 评估 (GSM8K, 对比)
├── tests/                # 单元测试
├── checkpoints/          # 模型保存
├── outputs/              # TensorBoard日志
└── run_pipeline.py       # 一键运行
```

## 模型规模

| 参数 | 值 |
|------|-----|
| layers | 8 |
| hidden | 320 |
| heads | 5 |
| experts | 8 (+1 shared) |
| vocab | 16384 |
| **总计** | **~27M** |
| VRAM (fp16, bs=8) | ~312 MB |

## 学习路径

1. **先看论文笔记**: 理解 DeepSeek-V2/V3/R1 的核心思想
2. **读模型代码**: `src/model/` 顺序: config → rms_norm → rotary → mla → moe → layer → model → mtp → for_lm
3. **读训练代码**: pretrain → sft → grpo → rejection_sampling
4. **运行管道**: `python run_pipeline.py --quick`
5. **看评估结果**: `compare_pipelines.py` 输出各阶段准确率对比

## 预期结果

对于一个 27M 参数的小模型，预期 GSM8K 准确率：

| 阶段 | 预期准确率 | 说明 |
|------|-----------|------|
| Pretrain only | ~2-5% | 无指令遵循能力 |
| Cold-Start SFT | ~10-15% | 学会CoT格式 |
| SFT + GRPO | ~15-25% | RL提升推理 |
| Full R1 Pipeline | ~20-30% | 自我提升循环 |

**重点**: 观察各阶段的**提升趋势**，而非绝对数值。这印证了 DeepSeek-R1 的核心发现。

## 技术要点

- **MLA**: 8x KV缓存压缩 (80维 vs 640维)
- **MoE**: 细粒度专家分割, 共享专家始终激活
- **负载均衡**: 无辅助损失的偏置调节 (V3核心创新)
- **MTP**: 同时预测t+1和t+2, 更密集的训练信号
- **GRPO**: 无Critic的PPO, 组内相对排名替代baseline
- **拒绝采样**: 自我生成→自我筛选→自我提升

## License

MIT — 仅供学习使用
