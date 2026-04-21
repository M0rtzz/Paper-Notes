---
title: >-
  [论文解读] ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping
description: >-
  [ICLR2026][模型压缩][扩散模型] 针对扩散大语言模型（dLLM）推理中大量 token 计算冗余的问题，提出无需训练的 Early-Skipping 加速框架 ES-dLLM，通过估计 token 重要性并在早期层跳过低重要性位置，在 LLaDA-8B 和 Dream-7B 上实现 5.6×–16.8× 加速且不损失生成质量。
tags:
  - ICLR2026
  - 模型压缩
  - 扩散模型
  - Inference Acceleration
  - Token Skipping
  - KV Cache
  - training-free
---

# ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping

**会议**: ICLR2026  
**arXiv**: [2603.10088](https://arxiv.org/abs/2603.10088)  
**代码**: [zhuzj19/ES-dLLM](https://github.com/zhuzj19/ES-dLLM)  
**领域**: model_compression  
**关键词**: Diffusion LLM, Inference Acceleration, Token Skipping, KV Cache, training-free

## 一句话总结

针对扩散大语言模型（dLLM）推理中大量 token 计算冗余的问题，提出无需训练的 Early-Skipping 加速框架 ES-dLLM，通过估计 token 重要性并在早期层跳过低重要性位置，在 LLaDA-8B 和 Dream-7B 上实现 5.6×–16.8× 加速且不损失生成质量。

## 背景与动机

扩散大语言模型（dLLM）如 LLaDA、Dream 通过迭代去噪生成文本，支持双向注意力和并行解码，是自回归模型（ARM）的有力替代。然而开源 dLLM 的推理效率远低于同等规模的 ARM，核心瓶颈在于：

1. **每轮迭代处理全序列**：dLLM 每次迭代需要对整个输入序列做前向计算，计算开销巨大
2. **大量无效计算**：每轮仅 unmask 少量高置信度 token，绝大多数 mask token 的计算结果并未被使用
3. **相邻迭代输入几乎不变**：相邻迭代仅在新 unmask 位置有差异，中间状态变化微小

作者通过实验分析发现：(a) 大多数位置的置信度变化近似指数分布且集中在零附近，超过 90% 的位置变化 < 0.05；(b) 隐藏状态在连续迭代间的变化同样微小。这揭示了大量可消除的冗余计算。

## 核心问题

如何在不引入额外训练的前提下，识别并跳过 dLLM 推理中低重要性 token 的计算，从而显著降低每轮迭代的计算量？

## 方法详解

ES-dLLM 包含两个核心组件：

### 1. 重要性分数估计（Importance Score Estimation）

对每个 token 位置 $i$ 在第 $l$ 层计算重要性分数：

$$I_{l,i} = \alpha \cdot c_i^{(t-1)} + (1-\alpha) \cdot \frac{\| H_{l,i}^{(t)} - H_{l,i}^{(t-1)} \|_1}{\sqrt{d} \cdot \| H_{l,i}^{(t-1)} \|_2}$$

- **置信度项** $c_i^{(t-1)}$：上一轮迭代中该位置的最大 softmax 概率，高置信度 token 更有可能在当前轮被 unmask
- **变化量项**：当前层隐藏状态与上轮缓存的 L1 差异（经 L2 归一化），反映 token 对新生成内容的语义依赖
- **权重** $\alpha$：平衡两项的超参数，默认设为 0.5

根据重要性分数保留 top-$k$（$k = (1 - r_l) \cdot |S|$）个位置继续推理，其余位置跳过。

### 2. 部分缓存更新与早期跳过（Partial Cache Update & Early Skip）

推理流程（每个 Transformer block）：

1. 对当前位置集合 $S$ 中的 token 做 normalization 和 QKV 投影
2. 更新 KV 缓存中对应位置（scatter 操作），读取完整 KV Cache 做注意力计算
3. 经 FFN 得到隐藏状态 $H_S$，同时更新隐藏状态缓存
4. 基于公式计算重要性分数，选出 top-$k$ 位置组成新集合 $S'$
5. 仅将 $S'$ 对应的隐藏状态传递给下一层

**关键设计**：
- 跳过位置越早层越省计算（后续所有层的矩阵乘法规模都缩小），但过早跳过可能影响变化量估计的可靠性
- 默认在 1/8 和 1/4 深度处各设 skip ratio = 0.5，约减少 60% FLOPs
- 为防止误差累积，以固定周期对 prompt token 或当前 block 的 KV Cache 做全量刷新

### 3. 与 DualCache 的对比

DualCache（Fast-dLLM）缓存当前 block 之外的 KV，但仍需计算整个 block；ES-dLLM 在 block 内部进一步跳过不重要位置，实现更深层次的计算节省。

## 实验关键数据

在 NVIDIA H200 GPU 上的主要结果：

**LLaDA-8B-Instruct**：

| 基准 | 原始 TPS | ES-dLLM TPS | 加速比 | 精度变化 |
|------|---------|-------------|--------|---------|
| GSM8K | 8.56 | 143.93 | 16.8× | +0.23 |
| MATH | 14.04 | 103.63 | 7.4× | -0.90 |
| BBH | 11.06 | 159.89 | 14.5× | -2.24 |
| HumanEval | 23.65 | 226.57 | 9.6× | +1.21 |
| MBPP | 8.98 | 145.99 | 16.3× | -1.60 |

**Dream-7B-Instruct**：

| 基准 | 原始 TPS | ES-dLLM TPS | 加速比 | 精度变化 |
|------|---------|-------------|--------|---------|
| GSM8K | 19.80 | 267.13 | 13.5× | -1.06 |
| MATH | 26.38 | 147.44 | 5.6× | -0.50 |
| HumanEval | 44.34 | 308.51 | 7.0× | -1.83 |
| MBPP | 21.68 | 276.12 | 12.7× | -3.40 |

相比 DualCache，ES-dLLM 额外取得 1.20×–1.85× 加速，且多个基准上精度更优。

**消融实验要点**：
- $\alpha=0.5$（两项等权）整体最优；仅用置信度（$\alpha=1$）退化明显
- 隐藏状态作为变化指示器略优于 QKV 张量，但后者内存开销更小
- 额外内存开销极小：LLaDA-8B 每 output token 仅 528KB，Dream-7B 仅 70KB

## 亮点

1. **无需训练**：纯推理阶段优化，即插即用，兼容现有 dLLM
2. **观察驱动设计**：基于对 dLLM 生成过程中置信度和隐藏状态变化的系统分析，动机充分
3. **加速显著**：最高 16.8× 加速，在多数基准上精度变化在 ±2% 以内
4. **与现有方法正交**：可与稀疏注意力、并行解码等技术组合使用
5. **内存开销可忽略**：相比模型权重 10GB+，额外缓存仅需数百 MB

## 局限与展望

1. **重要性估计依赖简单启发式**：置信度 + L1 变化的线性组合可能不够精确，可训练轻量级模型预测重要性
2. **部分 KV 更新偏离训练假设**：dLLM 训练时假设完整状态更新，跳过可能引入分布偏移
3. **实际加速低于理论值**：减少 60% FLOPs 但仅提速 1.2×–1.85×（相比 DualCache），因推理进入 memory-bound 模式，权重和 KV 的内存访问未减少
4. **skip ratio 需手动调节**：不同任务可能需要不同的跳过比例，缺乏自适应调整机制

## 与相关工作的对比

| 方法 | 类型 | 是否需要训练 | 加速来源 | 局限 |
|------|------|-------------|---------|------|
| Semi-AR（LLaDA） | 生成策略 | 否 | 分块顺序生成 | 限制生成顺序 |
| BD3-LM | 模型设计 | 是 | 单向注意力+KV Cache | 丧失双向建模 |
| dKV-Cache | KV 缓存 | 否 | 延迟更新新 unmask 的 KV | 未利用 semi-AR |
| DualCache | KV 缓存 | 否 | block 外 KV 复用 | 仍计算整个 block |
| Sparse-dLLM | 稀疏注意力 | 否 | 稀疏化历史 token | 优化注意力范围，正交 |
| **ES-dLLM** | **token 跳过** | **否** | **block 内跳过低重要性位置** | **memory-bound 瓶颈** |

## 启发与关联

- **跳过与缓存的互补性**：ES-dLLM 与 DualCache 可叠加使用，前者减少 block 内计算，后者减少 block 外计算，思路可推广到其他迭代式生成模型
- **重要性分数的通用性**：置信度 + 变化量的双因子评估可迁移到扩散图像生成中的 token pruning
- **dLLM 推理优化发展迅速**：从 KV 缓存 → 稀疏注意力 → token 跳过 → 并行解码，各类技术正交互补，系统层面的整合是未来关键

## 评分
- 新颖性: 7/10（核心思路直觉明确，但技术设计相对简单）
- 实验充分度: 8/10（多模型、多基准、丰富消融，H200 实测吞吐）
- 写作质量: 8/10（动机清晰，实验组织有条理）
- 价值: 7/10（实用性强但实际加速受 memory-bound 限制，理论加速潜力未充分释放）

<!-- RELATED:START -->

## 相关论文

- [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](../../AAAI2026/model_compression/skipcat_rank-maximized_low-rank_compression_of_large_language_models_via_shared_.md)
- [Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)
- [Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)

<!-- RELATED:END -->
