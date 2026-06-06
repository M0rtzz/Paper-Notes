---
title: >-
  [论文解读] PaTH Attention: Position Encoding via Accumulating Householder Transformations
description: >-
  [NeurIPS 2025][LLM评测][position encoding] 提出 PaTH（Position encoding via accumulating Householder Transformations），一种数据依赖的乘法位置编码方案…
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "position encoding"
  - "Householder transformation"
  - "注意力机制"
  - "state tracking"
  - "RoPE"
---

# PaTH Attention: Position Encoding via Accumulating Householder Transformations

**会议**: NeurIPS 2025  
**arXiv**: [2505.16381](https://arxiv.org/abs/2505.16381)  
**代码**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)  
**领域**: LLM评测  
**关键词**: position encoding, Householder transformation, attention mechanism, state tracking, RoPE

## 一句话总结

提出 PaTH（Position encoding via accumulating Householder Transformations），一种数据依赖的乘法位置编码方案，通过累积 Householder 变换替代 RoPE 的静态旋转矩阵，在理论表达力和实际语言建模性能上均优于 RoPE。

## 研究背景与动机

Attention 机制本身是置换不变的，位置编码对序列建模至关重要。RoPE 已成为主流 LLM 的标准位置编码方案，其核心是通过旋转矩阵 $\mathbf{R}$ 编码相对位置，使得注意力 logit 为 $\mathbf{q}_i^\top \mathbf{R}^{i-j} \mathbf{k}_j$。

然而 RoPE 存在两个根本性局限：

**数据无关性**：旋转矩阵 $\mathbf{R}$ 仅依赖相对位置 $i-j$，完全不受输入内容影响，限制了表达力

**计算复杂度受限**：RoPE Transformer 仍被限制在 $\mathsf{TC}^0$ 复杂度类中，无法解决需要序列推理的简单合成任务（如 flip-flop 语言建模和状态追踪任务）

这些局限意味着 RoPE 可能无法充分建模现实世界中需要的序列推理能力，例如代码中的变量追踪、长上下文中的实体追踪等。

## 方法详解

### 整体框架

PaTH 将乘法位置编码推广为数据依赖的形式。注意力 logit 形如：

$$\mathbf{A}_{ij} \propto \exp\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^{i} \mathbf{H}_s\right) \mathbf{q}_i\right)$$

其中 $\mathbf{H}_s$ 是数据依赖的 Householder-like 变换矩阵，沿路径从位置 $j$ 到位置 $i$ 累积。RoPE 是其特例（$\mathbf{H}_s = \mathbf{R}$，静态旋转矩阵）。

### 关键设计

**Householder-like 变换矩阵**：

$$\mathbf{H}_t = \mathbf{I} - \beta_t \mathbf{w}_t \mathbf{w}_t^T$$

其中：
- $\mathbf{w}_t \in \mathbb{R}^d$：通过低秩线性层 + 短卷积（filter size 3）+ L2 归一化得到
- $\beta_t = 2 \times \text{sigmoid}(\mathbf{u}^\top \mathbf{x}_t + b) \in (0, 2)$：数据依赖的缩放因子

$\beta_t$ 取值范围 $(0,2)$ 允许变换矩阵具有负特征值，这被证明能提升状态追踪性能。

**与线性 RNN 的联系**：PaTH 可以看作是 DeltaNet 式线性 RNN 的 softmax 版本。展开 RNN 递推：

- RNN: $\mathbf{o}_t = \sum_{j=1}^{t} \mathbf{v}_j \left(\mathbf{k}_j^\top \prod_{s=j+1}^{t} \mathbf{H}_s \cdot \mathbf{q}_t\right)$
- PaTH: $\mathbf{o}_t = \frac{1}{Z_t} \sum_{j=1}^{t} \mathbf{v}_j \exp\left(\mathbf{k}_j^\top \prod_{s=j+1}^{t} \mathbf{H}_s \cdot \mathbf{q}_t\right)$

这种关系使得 PaTH 兼具 softmax attention 的关联检索能力和线性 RNN 的状态追踪能力。

**理论保证**（Theorem 2.1）：一层 PaTH transformer（2 个头，$\log n$ 精度）可以求解 $\mathsf{NC}^1$-complete 问题，即能将 Transformer 扩展到 $\mathsf{TC}^0$ 复杂度类之外。

**PaTH-FoX 扩展**：将 PaTH 与 Forgetting Transformer (FoX) 结合：

$$\mathbf{A}_{ij} \propto \left(\prod_{s=j+1}^{i} f_s\right) \exp\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^{i} \mathbf{H}_s\right) \mathbf{q}_i\right)$$

类似于 Gated DeltaNet 将 DeltaNet 与 Mamba2 结合的成功。

### 损失函数 / 训练策略

利用 **UT 变换**将 Householder 矩阵的累积乘积紧凑表示为：

$$\prod_{t=0}^{L-1} \mathbf{H}_t = \mathbf{I} - \mathbf{W}^\top \mathbf{T}^{-1} \mathbf{W}$$

其中 $\mathbf{T}^{-1} = (\mathbf{I} + \text{strictLower}(\mathbf{D}\mathbf{W}\mathbf{W}^\top))^{-1} \mathbf{D}$。

全矩阵形式下，注意力矩阵可表示为：

$$\widetilde{\mathbf{A}} = \text{lower}(\mathbf{Q}\mathbf{K}^\top) - \text{lower}(\mathbf{Q}\mathbf{W}^\top) \mathbf{T}^{-1} \text{strictLower}(\mathbf{W}\mathbf{K}^\top)$$

设计了 FlashAttention 风格的分块算法，将全局逆运算分解为局部逆运算，总复杂度为 $\mathcal{O}(L^2 d + Ld^2/B)$，当 $B \approx d$ 时与标准 attention 相当。

推理时可原地更新历史 key 缓存：$\mathbf{k}_i^{(t)} \leftarrow (\mathbf{I} - \beta_t \mathbf{w}_t \mathbf{w}_t^\top) \mathbf{k}_i^{(t-1)}$，使得解码阶段退化为标准 softmax attention 解码，兼容 FlashDecoding、PagedAttention 等现有推理优化。

## 实验关键数据

### 主实验

| 模型 | Wiki. ppl↓ | LMB. ppl↓ | LMB. acc↑ | PIQA↑ | Hella.↑ | Wino.↑ | ARC-e↑ | ARC-c↑ | Avg.↑ |
|------|-----------|-----------|----------|-------|---------|--------|--------|--------|-------|
| RoPE | 19.01 | 19.77 | 40.4 | 70.2 | 50.3 | 54.9 | 67.2 | 33.3 | 52.7 |
| FoX | 18.33 | 18.28 | 41.7 | 70.8 | 50.9 | 57.1 | 65.7 | 32.6 | 53.1 |
| PaTH | **18.03** | **16.79** | **44.0** | 70.5 | **51.5** | 56.0 | **68.9** | **34.4** | **54.2** |
| PaTH-FoX | **17.35** | **16.23** | 44.1 | **70.8** | **52.2** | **57.1** | 67.3 | 33.9 | 54.2 |

760M 参数模型，50B tokens 训练。PaTH 在几乎所有任务上超越 RoPE，PaTH-FoX 实现最低 perplexity。

### 合成任务实验

| 模型 | FFLM ID Sparse | FFLM ID Dense | FFLM OOD |
|------|---------------|--------------|----------|
| RoPE | 6.9% | 40.3% | 0.01% |
| SBA | 9.6% | 38.9% | 0% |
| FoX | 8.3% | 36.3% | 0% |
| **PaTH** | **0%** | **0.0001%** | **0%** |

PaTH 在 FFLM 任务上几乎完美解决（错误率趋近 0），而其他方法错误率高达 40%。

### 消融实验

**长上下文基准**（训练长度 4096，测试更长序列）：

| 模型 | RULER 4K/8K/16K | BABILONG 4K/8K/16K | PhoneBook 2K/4K/8K |
|------|----------------|--------------------|--------------------|
| RoPE | 35.7/1.3/0.0 | 13.8/0.0/0.0 | 15.6/0.0/0.0 |
| FoX | 41.6/29.5/4.9 | 20.2/8.2/4.4 | 38.5/17.7/— |
| PaTH | 44.6/34.8/18.7 | 24.6/16.8/11.6 | 20.8/0.0/— |
| PaTH-FoX | 42.3/34.0/22.6 | 25.6/19.2/10.0 | **93.8/66.6**/— |

**RoPE→PaTH 模型转换**：

| 模型 | GSM8K | HumanEval | MBPP+ |
|------|-------|-----------|-------|
| RoPE | 19.9 | 23.1 | 47.1 |
| FoX | 15.5 | 21.3 | 48.2 |
| PaTH | **20.1** | **25.6** | **51.3** |

### 关键发现

1. PaTH 在状态追踪类任务上优势巨大，这类任务是 RoPE 的理论盲区
2. PaTH-FoX 可泛化到 64K tokens（训练长度仅 4096），代码领域提升尤为显著
3. RoPE→PaTH 转换在充分训练的模型上效果参差，需在模型训练早期进行

## 亮点与洞察

1. **从线性 RNN 到 attention 的优雅桥梁**：将 DeltaNet 的状态追踪能力注入 softmax attention
2. **UT 变换的巧妙应用**：利用 Householder 矩阵的紧凑表示实现分块并行训练
3. **推理时 key 缓存的原地更新**消除了额外存储需求，保持与现有推理基础设施的兼容性
4. **额外参数开销极低**：仅增加低秩线性层 + 短卷积 + sigmoid 门控

## 局限与展望

1. 训练速度下降约 1.5-2x，需进一步的 kernel 优化（如 ThunderKittens）
2. 实验主要在 760M 参数模型上进行，缺乏 7B+ 规模的从头训练验证
3. 蒸馏转换在已充分训练的模型上效果不稳定，需要探索更好的转换策略
4. 尚未探索与 Sliding Window Attention 等其他变体的组合

## 相关工作与启发

- 与 FoX（加法修改 logit）互补，组合后的 PaTH-FoX 效果更佳
- 为 "key 缓存迭代优化" 研究方向提供了新思路
- 代码已集成到 flash-linear-attention 库，降低了社区复现门槛

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从线性 RNN 视角重新理解位置编码，Householder 变换兼顾表达力和计算效率
- 实验充分度: ⭐⭐⭐⭐ 合成任务+语言建模+长上下文+模型转换全覆盖，但缺乏大规模训练验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，从动机到算法到实验逻辑清晰流畅
- 价值: ⭐⭐⭐⭐⭐ 有望成为 RoPE 的替代方案，代码已开源且集成到主流库

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](../../ICML2025/llm_evaluation/position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ICML 2025\] Position: AI Evaluation Should Learn from How We Test Humans](../../ICML2025/llm_evaluation/position_ai_evaluation_should_learn_from_how_we_test_humans.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_evaluation/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)

</div>

<!-- RELATED:END -->
