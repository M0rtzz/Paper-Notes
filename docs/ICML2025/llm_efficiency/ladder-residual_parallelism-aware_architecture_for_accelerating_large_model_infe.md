---
title: >-
  [论文解读] Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference
description: >-
  [ICML 2025][LLM效率][张量并行] 本文提出 Ladder Residual，一种简单的架构修改——将每个模块的输入从上一层的输出改为上上层的输出（错位残差），使模块计算与 AllReduce 通信解耦，从而实现通信与计算的重叠，在 70B 模型 8 卡 TP 推理中实现 29% 的端到端加速…
tags:
  - "ICML 2025"
  - "LLM效率"
  - "张量并行"
  - "通信隐藏"
  - "残差连接"
  - "推理加速"
  - "模型架构"
---

# Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference

**会议**: ICML 2025  
**arXiv**: [2501.06589](https://arxiv.org/abs/2501.06589)  
**代码**: 无（基于 gpt-fast 实现）  
**领域**: LLM效率  
**关键词**: 张量并行, 通信隐藏, 残差连接, 推理加速, 模型架构

## 一句话总结
本文提出 Ladder Residual，一种简单的架构修改——将每个模块的输入从上一层的输出改为上上层的输出（错位残差），使模块计算与 AllReduce 通信解耦，从而实现通信与计算的重叠，在 70B 模型 8 卡 TP 推理中实现 29% 的端到端加速，且模型性能与标准 Transformer 持平。

## 研究背景与动机
**领域现状**：随着 LLM 规模增长，张量并行 (Tensor Parallelism, TP) 是跨 GPU 推理的关键技术，将权重和激活分区到多个设备上并行计算。

**现有痛点**：TP 需要在每层进行 AllReduce 同步操作，这是阻塞式的——在 70B 模型 TP=8 的设置下，AllReduce 通信占推理延迟的 38%，P2P 禁用时甚至超过 50%。

**核心矛盾**：现有通信优化方案（如 Flux、CoCoNet）依赖底层 kernel 融合或自定义编译器，难以跨硬件迁移，且受限于模型架构的固有顺序依赖（h_{i+1} 依赖 x_i 的通信结果）。

**本文目标**：通过模型架构层面的修改（而非底层系统优化）来解耦通信和计算，实现通信延迟的隐藏。

**切入角度**：基于 Transformer 中激活变化缓慢的观察（每层更新 h_i(x) 的范数相对残差 x 较小），用"陈旧"输入代替最新输入不会显著影响质量。

**核心 idea**：将标准残差连接 x_{i+1} = h_{i+1}(x_i) + x_i 改为 x_{i+1} = h_{i+1}(x_{i-1}) + x_i，使 h_{i+1} 的计算不依赖 x_i 的 AllReduce 结果，从而可以并行执行。

## 方法详解

### 整体框架
Ladder Residual 修改 Transformer 的残差连接方式：每个模块（Attention 或 MLP）的输入不再是紧接其前的残差流输出，而是前两步的残差流输出。残差流本身仍然按照标准方式累加，保证信息不丢失。这使得每个模块的计算可以与前一个模块的 AllReduce 并行执行。

### 关键设计

1. **错位残差连接（Ladder Residual）**:

    - 功能：将标准 Transformer 的 x_{i+1} = h_{i+1}(x_i) + x_i 修改为 x_{i+1} = h_{i+1}(x_{i-1}) + x_i
    - 核心思路：模块 h_{i+1} 使用的是 x_{i-1}（已经在上一步完成了 AllReduce），而不需要等待 x_i 的 AllReduce 完成。AllReduce(x_i*) 可以与 h_{i+1}(x_{i-1}) 并行执行
    - 设计动机：Transformer 中每层更新的范数相对残差流较小，用"陈旧一步"的输入对最终表示的影响有限
    - 与标准 Transformer 的区别：残差流仍正常累加（保证 block i 可以处理前 i-2 个 block 的所有信息），仅模块输入使用陈旧值

2. **异步 AllReduce 实现**:

    - 功能：利用 NCCL 异步通信 + handle 传递实现非阻塞 AllReduce
    - 核心思路：Attention 计算完成后立即启动异步 AllReduce 并返回 handle；接着用上一层 MLP 的 handle 同步并执行当前 MLP 计算
    - 实现在 PyTorch 中非常简洁，无需自定义 kernel

3. **混合 Ladder 适配（Post-training Adaptation）**:

    - 功能：将预训练好的 Llama-3.1-8B-Instruct 的上半层转换为 Ladder Residual，仅用 3B token 轻量微调
    - 核心思路：下半层保持不变（避免破坏难以恢复的底层知识），仅修改上半层（16-32层）
    - 3B token 微调即可恢复到与原始模型相当的性能

### 损失函数 / 训练策略
- 从头训练：标准语言建模损失，cosine scheduler，peak LR 3e-4，warmup 8B tokens
- 后训练适配：在 Infinity-Instruct 数据集 (3B tokens) 上 SFT，LR 5e-6，200 步 warmup
- 训练配置：DDP/HSDP，BF16 混合精度，batch size 4M tokens

## 实验关键数据

### 主实验（不同模型规模的推理加速）

| 模型规模 | P2P 禁用加速 | P2P 启用加速 |
|---------|------------|------------|
| 1B | 1.39x | 1.56x |
| 3B | 1.50x | 1.57x |
| 8B | 1.40x | 1.46x |
| 34B | 1.47x | 1.44x |
| 70B | 1.59x | 1.29x |
| 176B | 1.54x | 1.35x |
| 405B | 1.57x | 1.31x |

### 70B 模型详细延迟分解（TP=8, batch=1）

| 模型 | Prefill 改善 | Decode 改善 | Token/sec 改善 |
|------|------------|------------|--------------|
| Ladder-70B (P2P=1) | 5.78% | 23.71% | **30.79%** |
| Ladder-70B (P2P=0) | 6.94% | 37.71% | **59.87%** |
| Parallel-70B (P2P=1) | 5.42% | 18.04% | 21.75% |

### 从头训练质量对比（100B tokens, FineWeb-edu）

| 模型 | ARC-C | HellaSwag | PIQA | SciQ | Winograde | 平均 | WikiText PPL |
|------|-------|----------|------|------|-----------|------|-------------|
| Standard-1.2B | 34.22 | 41.10 | 71.49 | 87.30 | 55.41 | 59.98 | 18.54 |
| Ladder-1.2B | 31.31 | 41.18 | 71.49 | 86.60 | 55.17 | 58.92 | 18.42 |
| Standard-3.5B | 38.99 | 46.48 | 74.59 | 92.00 | 58.48 | 64.11 | 14.48 |
| Ladder-3.5B | 36.77 | 45.66 | 73.72 | 89.90 | 58.96 | 62.91 | 14.90 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Hybrid-Ladder-8B-16L (微调后) | 平均 57.61 vs 原始 56.11 | 仅修改上半 16 层+3B token 微调，性能持平甚至略优 |
| Hybrid-Ladder-8B-20L (微调后) | 平均 53.86 | 修改 20 层过于激进，性能有一定下降 |
| 30% 更大 Ladder | 优于同大小 Standard | 利用加速省出的计算预算放大模型是更优策略 |

### 关键发现
- 通信越慢加速越大：P2P 禁用时 70B 加速从 29% 增至 60%
- Decode 阶段受益最大（因为计算强度最低，通信占比最高）
- Ladder Residual 与 Pipeline Parallelism 完全兼容
- 后训练适配仅需 3B tokens，远少于其他架构转换方法（如 Llama-to-Mamba 需要 50B tokens）

## 亮点与洞察
- 极其简洁的架构修改带来显著的推理加速，无需任何底层 kernel 工程
- 利用 Transformer 中"激活变化缓慢"的内在性质，用陈旧输入换取通信隐藏
- 后训练适配路线非常实用——现有模型只需轻微调整即可获得加速
- 30% 更大的 Ladder Transformer 性能优于同 FLOP 的标准 Transformer，说明用加速换更大模型是有效策略

## 局限与展望
- 3.5B 规模下 Ladder Transformer 比 Standard 略差 (平均准确率差 1.2 点)
- 更激进的 Ladder 适配（20 层）导致性能下降，最优适配层数需要探索
- Prefill 阶段加速有限（因为 prefill 计算密集度高，通信占比小）
- 目前仅验证了语言模型，视觉模型或多模态模型的适用性未知

## 相关工作与启发
- 与 Parallel Attention+MLP (PaLM) 的区别：PaLM 融合 Attn 和 MLP 减少一半通信，但牺牲了表达能力；Ladder 保持原有计算图，仅改变输入路由
- 与 Flux/CoCoNet 等系统优化的区别：这些方法需要底层 kernel 重写，Ladder 在 PyTorch 上层即可实现
- 对未来模型设计的启发："并行感知架构"是一个值得深入探索的方向
- Cross-Layer Attention 等参数共享方法是互补的效率优化

## 评分
- 新颖性: ⭐⭐⭐⭐（idea 极简但有效，利用了 Transformer 的内在性质）
- 实验充分度: ⭐⭐⭐⭐⭐（覆盖 1B-405B 多种规模，从头训练+后训练适配两种场景）
- 写作质量: ⭐⭐⭐⭐⭐（动机清晰，实验详尽，benchmark 设置合理）
- 价值: ⭐⭐⭐⭐⭐（极高实用价值，适用于所有基于残差的模型）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](../../ICLR2026/llm_efficiency/semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](../../NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](../../ACL2025/llm_efficiency/accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](../../ACL2025/llm_efficiency/squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)

</div>

<!-- RELATED:END -->
