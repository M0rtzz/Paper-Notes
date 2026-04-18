---
title: >-
  [论文解读] Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity
description: >-
  [NeurIPS 2025][LLM/NLP][LLM inference] 揭示了 LLM 推理中稀疏性的"极性转移"现象——MLP 层稀疏性随 batch 增大而消失，而 attention head 稀疏性保持稳定且与 batch 无关，据此设计了 Selective Head Attention 及对应 GPU kernel，在大 batch 推理中实现高达 2.2x 的端到端加速。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - LLM inference
  - contextual sparsity
  - 注意力机制
  - batched inference
  - GPU kernel
---

# Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity

**会议**: NeurIPS 2025  
**arXiv**: [2505.14884](https://arxiv.org/abs/2505.14884)  
**代码**: [susavlsh10/Polar-Sparsity](https://github.com/susavlsh10/Polar-Sparsity)  
**领域**: llm_nlp  
**关键词**: LLM inference, contextual sparsity, attention head sparsity, batched inference, GPU kernel

## 一句话总结

揭示了 LLM 推理中稀疏性的"极性转移"现象——MLP 层稀疏性随 batch 增大而消失，而 attention head 稀疏性保持稳定且与 batch 无关，据此设计了 Selective Head Attention 及对应 GPU kernel，在大 batch 推理中实现高达 2.2x 的端到端加速。

## 研究背景与动机

上下文激活稀疏性（contextual sparsity）是加速 LLM 推理的有前景方向：每个 token 仅激活模型参数的一小部分。但**现有方法不能扩展到大 batch 推理**——这在现实部署中至关重要。

核心矛盾在于：

**MLP 层稀疏性随 batch 增大快速消失**：batch 中不同序列的活跃神经元取并集后，稀疏性迅速趋近稠密计算

**Attention 层随 batch 和序列长度增加变成瓶颈**：batch 增大后 attention 延迟线性增长，逐渐主导端到端延迟

**现有工作仅优化单 query 推理**：DejaVu、PowerInfer 等方法在大 batch 下收益消失

以 OPT-66B 为例，batch size 从 1 增大到 64 时，MLP 层因 batch 化而变高效，但 attention 层延迟几乎线性增长，成为新的瓶颈。

## 方法详解

### 整体框架

Polar Sparsity 的核心洞察：**稀疏性的重要性从 MLP 层极性转移到 Attention 层**。

- **小 batch**：MLP 层稀疏性有效，attention 开销较低 → 传统激活稀疏方法有效
- **大 batch**：MLP 层稀疏性消失，attention 变成主要瓶颈 → 需要 attention head 稀疏性

系统包含两个核心组件：
1. **MLP 层的动态稀疏**：Selective GEMM kernel + 动态逐层 top-k 策略
2. **Attention 层的稳定稀疏**：Selective Head Attention (SHA) kernel

### 关键设计

**MLP 层动态稀疏**：

对于 batch 输入 $\mathbf{x} \in \mathbb{R}^{B \times 1 \times d}$，MLP 的稀疏化计算为：

$$\text{MLP}_{S_B}(\mathbf{x}) = \sigma(\mathbf{x} W_{1, S_B}) W_{2, S_B}^\top$$

其中 $S_B \subseteq [D]$ 是 batch 内所有序列活跃神经元的并集。

- 使用轻量级两层前馈网络作为 router 预测神经元激活
- 提出**动态 top-k 机制**：不同层使用不同的 k 值，通过贪心算法离线优化每层的 k 以达到目标 recall（99%）
- 设计融合 indexing 和矩阵乘法的 GPU kernel（Selective GEMM），避免 gather-scatter 开销

**Attention Head 稀疏性**：

关键观察：对每个 token，只有少数 attention head 对输出有显著贡献，其余 head 的影响可忽略。

$$\text{Attention}(Q_{b,i}, K_{b,i}, V_{b,i}) = \text{softmax}\left(\frac{Q_{b,i} K_{b,i}^\top}{\sqrt{d_h}}\right) V_{b,i}$$

由于每个序列独立计算 attention，**head 稀疏性与 batch 大小无关**——这是其相比 MLP 稀疏性的根本优势。

实验发现：
- 激活最重要的 head 时，perplexity 在 50% head 稀疏度内增长缓慢
- **模型越大，head 稀疏性越高**：OPT-66B 在 30% head 激活时 perplexity 仅增 5%
- Layer 0 的 attention importance 始终最高 → 对 Layer 0 使用 dense attention

**Selective Head Attention (SHA) Kernel**：

基于 FlashAttention 修改的稀疏感知 kernel：
- 输入包含 batch head index tensor，记录每个 batch 的活跃 head 索引
- 每个 CUDA thread-block 处理一个 batch 和一个 head
- 仅对活跃 head 执行 read/write，减少 memory I/O 和计算量
- 对 GQA 模型采用 group sparsity

### 损失函数 / 训练策略

Router 训练：
- 从 Wikitext-2 训练集采集 400K token 样本
- MLP router：两层前馈网络，二元交叉熵损失，AdamW 优化器
- Attention router：单层全连接网络，基于 attention 输出 L2 范数的 top-k 作为监督目标

## 实验关键数据

### 主实验

**零样本基准评估**（关键阈值处）：

| 模型 | COPA | OBQA | PIQA | RTE | WG | HS | MMLU | ARC-E | ARC-C | Avg |
|------|------|------|------|-----|-----|-----|------|-------|-------|-----|
| OPT 66B | 0.85 | 0.304 | 0.787 | 0.603 | 0.690 | 0.557 | 0.263 | 0.711 | 0.369 | 0.570 |
| OPT 66B + PS-0.3 | 0.83 | 0.296 | 0.769 | 0.592 | 0.677 | 0.546 | 0.264 | 0.693 | 0.361 | 0.560 |
| LLaMA 2 7B | 0.87 | 0.314 | 0.781 | 0.628 | 0.690 | 0.572 | 0.418 | 0.763 | 0.433 | 0.608 |
| LLaMA 2 7B + PS-0.5 | 0.89 | 0.312 | 0.779 | 0.552 | 0.687 | 0.568 | 0.356 | 0.762 | 0.439 | 0.594 |
| LLaMA 3.1 70B | 0.92 | 0.370 | 0.831 | 0.697 | 0.799 | 0.665 | 0.753 | 0.872 | 0.606 | 0.724 |
| LLaMA 3.1 70B + PS-0.625 | 0.91 | 0.340 | 0.823 | 0.729 | 0.793 | 0.650 | 0.732 | 0.853 | 0.590 | 0.712 |

所有模型在关键稀疏度阈值处平均准确率差异 <1%。

**与其他稀疏方法对比**（LLaMA-2-7B）：

| 方法 | COPA | PIQA | WG | HS | MMLU(5) | ARC-E | ARC-C |
|------|------|------|-----|-----|---------|-------|-------|
| Dense | 0.87 | 0.781 | 0.690 | 0.572 | 0.458 | 0.763 | 0.433 |
| ReLUfication | 0.83 | 0.779 | 0.686 | 0.548 | 0.386 | 0.738 | 0.396 |
| CATS-50% | — | 0.769 | 0.675 | 0.571 | 0.421 | 0.744 | 0.412 |
| TEAL-50% | — | 0.778 | 0.673 | — | 0.405 | — | — |
| **PolarSparse-50%** | **0.89** | 0.779 | **0.687** | 0.568 | 0.381 | **0.762** | **0.439** |

### 消融实验

**吞吐量提升**：

| 模型 | 配置 | Batch=1 加速 | 大 Batch 加速 |
|------|------|-------------|--------------|
| OPT-66B | PS-0.3 | ~1x | **2.2x** |
| LLaMA-2-7B | PS-0.5 | ~1x | **1.85x** |
| LLaMA-3.1-70B | PS-0.625 | ~1x | **1.51x** |

**GPU Kernel 性能**：
- Selective GEMM：最高 **5.5x** 加速（vs dense baseline）
- Selective FlashAttention：30% 稀疏度下 **2.8x** 加速
- 两个 kernel 均展现近线性的稀疏度-加速关系

### 关键发现

1. **稀疏性极性转移已被量化验证**：OPT-66B 在 batch=64 时，MLP 层 union 激活率接近 100%，而 head 稀疏性保持不变
2. **模型越大越适合 head 稀疏**：OPT-66B 在 30% 激活下仅损失 5% perplexity
3. **首次证明上下文稀疏性可扩展到大 batch**：之前所有工作仅在 batch=1 下有效
4. **GQA 模型的稀疏度阈值较高**（62.5% vs 30-50%），因 group 内 head 共享 KV cache 导致 group 稀疏本质上更弱

## 亮点与洞察

1. **核心洞察简洁而深刻**：稀疏性从 MLP→Attention 的极性转移是一个直观但被忽视的观察，具有很强的指导意义
2. **系统设计完整**：从 router 训练到 GPU kernel 到端到端系统，形成完整的工程方案
3. **广泛的模型覆盖**：OPT、LLaMA-2/3、Mistral、Qwen 均验证有效
4. **与现有推理框架无缝集成**：基于 FlashAttention Triton kernel 构建，使用 CUDA Graphs
5. **batch-invariant 性质**使其天然适合生产环境的 batched serving

## 局限与展望

1. 在小 batch 推理中收益有限，GPU 工作量不足以体现稀疏优势
2. 固定 top-k 策略不够灵活，动态的输入/层自适应策略可能更优
3. Head 稀疏与 token 稀疏的组合可能带来乘法级加速，值得探索
4. 仅评估了不超过 16K 的上下文长度，百万 token 上下文场景待验证
5. GQA 模型的 group sparsity 效果弱于 MHA 的 head sparsity
6. 仅支持 greedy decoding，beam search 和 speculative decoding 下的稀疏模式可能不同

## 相关工作与启发

- 与 DejaVu（MLP 稀疏为主）互补：Polar Sparsity 在大 batch 下优势更明显
- MoA/MoH 等 MoE 风格的 head 路由仅有理论 FLOP 减少，无实际加速；本文提供了 kernel 级别的实际加速
- 启发方向：task-aware query-sensitive routing（同一 batch 内为困难/简单 query 分配不同数量的 head）可能实现无损稀疏推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 极性转移观察本身不难发现，但系统性地将其转化为可工程化的加速方案是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多个模型家族和规模，包含准确率+吞吐量+kernel 微基准，与多个 baseline 详细对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验充实，但技术细节可以更精炼
- 价值: ⭐⭐⭐⭐⭐ 直接可部署的推理加速方案，首次解决了大 batch 下上下文稀疏性失效的问题
