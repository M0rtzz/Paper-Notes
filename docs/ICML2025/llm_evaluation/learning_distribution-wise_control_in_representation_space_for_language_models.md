---
title: >-
  [论文解读] Learning Distribution-Wise Control in Representation Space for Language Models
description: >-
  [ICML 2025][LLM评测][表示干预] 将表示微调（Representation Fine-tuning）中的确定性节点替换为随机节点，通过重参数化技巧学习潜在分布而非单点变换，在常识推理和数学推理任务上取得了一致性能提升，尤其在早期层的干预效果最为显著。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "表示干预"
  - "分布级控制"
  - "重参数化"
  - "概念子空间"
  - "表示微调"
---

# Learning Distribution-Wise Control in Representation Space for Language Models

**会议**: ICML 2025  
**arXiv**: [2506.06686](https://arxiv.org/abs/2506.06686)  
**代码**: [chili-lab/D-Intervention](https://github.com/chili-lab/D-Intervention)  
**领域**: LLM评测  
**关键词**: 表示干预, 分布级控制, 重参数化, 概念子空间, 表示微调

## 一句话总结

将表示微调（Representation Fine-tuning）中的确定性节点替换为随机节点，通过重参数化技巧学习潜在分布而非单点变换，在常识推理和数学推理任务上取得了一致性能提升，尤其在早期层的干预效果最为显著。

## 研究背景与动机

**表示干预的兴起**：近年来，表示微调（如 ReFT、LoFiT）作为一种新兴高层控制方法，通过直接修改语言模型层间的隐藏表示来操控模型行为。相比 LoRA 等参数高效微调方法（PEFT），表示微调仅使用 1/10 到 1/100 的参数量即可达到甚至超过 PEFT 的表现。

**核心观察**：已有研究表明，概念空间是连续的（Gandikota et al., 2023）——当找到一个干预向量后，调整其幅度可以控制干预效果。这意味着干预点的邻域区域也应该产生相关效果。然而，现有方法（如 ReFT）仅学习**单点变换（pointwise）**，没有探索干预点周围的分布信息。

**类比启发**：这个问题类似于自编码器（AE）到变分自编码器（VAE）的演进——VAE 通过将确定性节点替换为随机采样，使模型能直接学习潜在分布，从而获得更好的生成能力和更平滑的潜在空间。

**核心问题**：如何有效地探索干预向量周围的概念空间区域？

## 方法详解

### 整体框架

本文提出 **Distribution-wise Intervention（D-Intervention）**，核心思想极为简洁：将表示干预中的确定性 MLP 节点替换为随机节点，使其能够学习一个分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ 而非单一变换点。

**信息论视角**：作者首先从信息论角度统一理解干预方法。对于插入层 $l$ 的干预函数 $f_\phi$，最小化交叉熵损失等价于最大化干预后表示与目标输出之间的互信息：

$$\arg\min_\phi \mathcal{L}_{CE} \equiv \arg\max_\phi I(Y; f_\phi(Z^{(l)}))$$

这揭示了干预的本质目标：变换内部表示使其对目标输出最具信息量，同时过滤掉无关信息。

### 关键设计

#### 1. 随机干预重参数化

原始确定性 MLP 干预为：
$$\hat{Z} = \text{MLP}(Z) = W^T Z + b$$

本文将其替换为两个独立网络，分别学习均值和对数方差：

- **均值网络**：$\boldsymbol{\mu} = \text{MLP}_\mu(Z)$
- **方差网络**：$\log \sigma^2 = \text{MLP}_{\log\sigma^2}(Z)$
- **标准差计算**：$\boldsymbol{\sigma} = \exp(\frac{1}{2}\log\sigma^2)$
- **噪声采样**：$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
- **最终输出**：$\hat{Z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$

通过重参数化技巧，随机性来自外部噪声 $\boldsymbol{\epsilon}$，梯度可以正常通过 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 反向传播。

**与 VAE 的关键区别**：本文移除了 KL 散度损失项，不再是变分推断。这使得模型可以无约束地学习分布，不强制要求分布接近标准正态先验，从而给予模型更大的自由度来探索有利于任务的分布形态。

#### 2. 模型特定的 Clamping 机制

为解决大采样方差导致的数值不稳定问题，作者提出基于目标语言模型权重分布的 clamping 策略。对于层 $l$ 的干预，clamping 边界定义为相邻层权重矩阵的极值：

$$v_{\min} = \min(\min(W^{(l)}), \min(W^{(l+1)}))$$
$$v_{\max} = \max(\max(W^{(l)}), \max(W^{(l+1)}))$$

这确保干预值始终在模型权重的自然范围内，边界在训练前一次性计算，后续固定不变。

#### 3. 混合干预策略

基于逐层实验的发现，作者提出了最优实践：**前几层使用随机节点（分布级干预），后续层保留确定性节点（点级干预）**。这种混合策略在性能和鲁棒性上均表现最佳。

### 损失函数 / 训练策略

**训练目标**：冻结基础语言模型 $\mathcal{M}$，仅训练随机干预层 $\{\mathcal{I}_l\}_{l=1}^L$，最小化下一 token 预测的交叉熵损失：

$$\mathcal{L} = -\mathbb{E}_{(X,Y)}[\log P_{\mathcal{M} \circ \mathcal{I}}(Y | f_\phi(Z^{(l)}))]$$

**关键特点**：

- 基础模型完全冻结，仅优化 $\{\phi_\mu^{(l)}, \phi_\sigma^{(l)}\}_{l=1}^L$
- 无 KL 散度正则，直接通过任务损失学习分布
- 参数量极少（比 PEFT 少 10x-100x）
- 推理时同样采样噪声，保持分布级干预

**实验配置**：

- 模型：Llama-7B, Llama-13B, Llama-3-8B
- 硬件：单张 NVIDIA RTX A6000，bfloat16 混合精度
- 两种设置：逐层（layer-wise）和全层（all-layer）
- 基线方法：RED, ReFT, MLP, SwiGLU 的确定性与分布级变体

## 实验关键数据

### 主实验

**常识推理（8个基准，全层设置）**：

| 方法 | 参数量 | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | 平均 |
|------|--------|-------|------|------|-----------|------------|-------|-------|------|------|
| Prefix-tuning | 6.1M | 65.4 | 76.4 | 73.1 | 42.1 | 59.7 | 72.1 | 47.0 | 60.2 | 62.0 |
| LoRA | 25.2M | 68.9 | 80.7 | 79.2 | 83.6 | 80.6 | 77.8 | 61.9 | 74.8 | 75.9 |
| DoRA | 25.2M | 69.7 | 83.5 | 78.9 | 91.2 | 83.6 | 81.9 | 67.5 | 79.0 | 79.4 |
| ReFT | 0.26M | 65.2 | 78.1 | 77.2 | 64.1 | 73.6 | 75.8 | 58.1 | 73.0 | 70.6 |
| **D-ReFT** | **0.52M** | **67.8** | **80.5** | **78.3** | **72.3** | **75.9** | **78.6** | **60.5** | **75.4** | **73.7** |

> D-ReFT 相比 ReFT 平均提升 **+3.1%**，仅用 0.52M 参数（LoRA 的 ~2%）。

**数学推理（7个基准）**：

| 方法 | GSM8K | SVAMP | MAWPS | AQuA | SAT | MMLU-Math | SAT-M | 平均 |
|------|-------|-------|-------|------|-----|-----------|-------|------|
| ReFT | 42.3 | 52.1 | 88.7 | 25.6 | 53.2 | 31.8 | 48.6 | 48.9 |
| **D-ReFT** | **45.8** | **55.3** | **90.1** | **27.1** | **55.9** | **33.5** | **51.2** | **51.3** |
| 提升 | +3.5 | +3.2 | +1.4 | +1.5 | +2.7 | +1.7 | +2.6 | **+2.4** |

### 消融实验

**逐层干预分析（不同层的干预效果）**：

| 干预层位置 | Pointwise 准确率 | Distribution-wise 准确率 | 提升幅度 | 学习方差 |
|-----------|-----------------|------------------------|---------|---------|
| 第 1-4 层（早期） | 61.2 | 67.1 | **+5.9%** | 高 |
| 第 5-8 层（中早期） | 63.5 | 67.8 | +4.3% | 中高 |
| 第 9-16 层（中期） | 65.1 | 66.9 | +1.8% | 中 |
| 第 17-24 层（中后期） | 64.8 | 65.5 | +0.7% | 低 |
| 第 25-32 层（后期） | 63.9 | 64.1 | +0.2% | 极低 |

**混合策略对比（前 k 层随机 + 其余确定性）**：

| 配置 | 常识推理平均 | 数学推理平均 | 说明 |
|------|-----------|-----------|------|
| 全层 Pointwise | 70.6 | 48.9 | ReFT 基线 |
| 全层 D-wise | 72.1 | 50.5 | 全部替换 |
| 前 25% D-wise + 后 75% Point | **73.7** | **51.3** | 最优配置 |
| 前 50% D-wise + 后 50% Point | 73.2 | 50.9 | 次优 |
| 前 75% D-wise + 后 25% Point | 72.5 | 50.6 | 接近全层 |

### 关键发现

1. **早期层增益最大**：分布级干预在前 1-4 层带来 +4% 到 +6% 的性能提升，但在后期层几乎无增益。这表明早期层的概念空间更宽广，更适合分布级探索。

2. **方差与性能强相关**：学习到的标准差越大，性能提升越明显。早期层自然学习到更大的方差，反映了更广泛的邻域探索。

3. **混合策略最优**：前几层随机 + 后续层确定性的混合方案在 **所有 15 个基准** 上一致优于纯点级干预，且鲁棒性显著提升。

4. **无 KL 正则更优**：移除 VAE 中的 KL 散度约束后，模型可以自由学习任务最优分布，性能优于带 KL 的变体。

## 亮点与洞察

- **极简设计，即插即用**：方法本身极为简单——把一个 MLP 换成两个 MLP（一个学均值，一个学方差），加上采样噪声。可作为任何表示干预方法的 drop-in 替换。
- **信息论统一视角**：将干预方法用互信息最大化统一理解，为后续研究提供了理论基础。最小化 CE 损失等价于最大化干预表示与输出之间的互信息。
- **揭示层间差异性**：早期层概念空间更宽泛、更适合探索性干预，后期层已形成较固定的表示，确定性干预即可。这一发现对理解 Transformer 内部表示组织有启发意义。
- **参数效率极高**：即使参数量翻倍（0.26M → 0.52M），仍远低于 LoRA（25.2M），但性能差距明显缩小。

## 局限与展望

1. **推理开销增加**：推理时仍需采样噪声，引入额外计算和非确定性。可以考虑训练完成后固化均值做确定性推理。
2. **仅验证 Llama 系列**：实验限于 Llama-7B/13B 和 Llama-3-8B，未在更大规模模型（70B+）或非 Llama 架构（Mistral, Qwen 等）上验证。
3. **方差学习缺乏显式引导**：方差完全由任务损失隐式学习，可能存在优化方差的更好策略（如课程学习、退火等）。
4. **任务范围有限**：仅评估推理类任务（常识 + 数学），未涉及生成、对齐、对话等更开放的场景。
5. **理论分析不够深入**：虽然提供了信息论视角，但对"为什么早期层方差更大"缺乏因果解释。

## 相关工作与启发

- **ReFT (Wu et al., 2024b)**: 本文的主要基线，基于分布式对齐搜索（DAS）理论的表示微调方法，D-Intervention 是其直接扩展。
- **LoFiT (Yin et al., 2024)**: 另一表示微调方法，侧重局部化-编辑范式，与 ReFT 互补。
- **VAE / VIB**: 重参数化技巧的经典来源。本文展示了移除 KL 约束后的"非变分"版本在干预场景下更优。
- **Representation Engineering (Zou et al., 2023)**: 表示工程开创性工作，发现干预效果可通过幅度调节，直接启发了本文的分布级探索思路。
- **LoRA / DoRA**: 主流 PEFT 方法，参数量远大于表示干预，但仍是重要对比基线。

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐ | 思路简单直觉，但从 AE→VAE 的类比迁移到干预领域有一定新意 |
| 理论深度 | ⭐⭐⭐⭐ | 信息论统一视角扎实，层间分析有洞察 |
| 实验充分度 | ⭐⭐⭐⭐ | 15 个基准，多模型，逐层分析详尽 |
| 实用性 | ⭐⭐⭐⭐⭐ | 即插即用，代码开源，实现简单 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，从动机到方法到实验环环相扣 |
| **综合** | **⭐⭐⭐⭐** | 思路简洁有效，实验扎实，是表示干预领域的有价值改进 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](../../CVPR2025/llm_evaluation/sufficient_invariant_learning_for_distribution_shift.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/llm_evaluation/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)

</div>

<!-- RELATED:END -->
