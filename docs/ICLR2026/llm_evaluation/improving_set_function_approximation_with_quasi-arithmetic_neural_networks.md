---
title: >-
  [论文解读] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks
description: >-
  [ICLR 2026][LLM评测][集合函数] 提出QUANN（准算术神经网络），用可逆神经网络实现可学习的Kolmogorov均值作为池化操作，首次实现机器学习版本的广义中心趋势度量，QUANN是均值可分解集合函数的通用近似器，且学到的嵌入跨任务迁移性更强。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "集合函数"
  - "Kolmogorov均值"
  - "可逆网络"
  - "可学习池化"
  - "置换不变性"
---

# Improving Set Function Approximation with Quasi-Arithmetic Neural Networks

**会议**: ICLR 2026  
**arXiv**: [2602.04941](https://arxiv.org/abs/2602.04941)  
**代码**: 无  
**领域**: 深度学习理论/集合函数  
**关键词**: 集合函数, Kolmogorov均值, 可逆网络, 可学习池化, 置换不变性

## 一句话总结
提出QUANN（准算术神经网络），用可逆神经网络实现可学习的Kolmogorov均值作为池化操作，首次实现机器学习版本的广义中心趋势度量，QUANN是均值可分解集合函数的通用近似器，且学到的嵌入跨任务迁移性更强。

## 研究背景与动机

### 领域现状

**领域现状**：集合函数学习要求置换不变性。DeepSets用sum池化，PointNet用max池化——两种固定、不可训练的池化操作将近似负担推给编码器和估计器。

**现有痛点**：(1) 固定池化迫使编码器学习既适应下游任务又适应特定池化操作的嵌入→限制了嵌入迁移性；(2) sum和max是Kolmogorov均值的极端特例→大量中间形式（几何均值、调和均值等）未被利用；(3) 已有可学习池化方法要么复杂难用、要么表达力有限（如Power DeepSets仅学一个指数）。

**核心矛盾**：需要一个理论有保证、实现简洁、足够表达的可学习池化操作。

**切入角度**：Kolmogorov均值 $M_f = f^{-1}(\frac{1}{n}\sum_i f(x_i))$ 通过选择不同可逆函数 $f$ 统一了各种均值。用可逆神经网络实现 $f$ →可学习的广义中心趋势。

## 方法详解

### 整体框架
QUANN: $\hat{F}(X) = \rho(\psi^{-1}(\frac{1}{|P_k(X)|}\sum_{\pi} \psi(\phi(\pi))))$，其中 $\phi$ 是编码器，$\psi$ 是可逆神经网络（生成函数），$\rho$ 是估计器。

### 关键设计

1. **Neuralized Kolmogorov Mean (NKM)**:

    - 功能：用可逆神经网络 $\psi$ 实现Kolmogorov均值的生成函数
    - 核心思路：$M_\psi(X) = \psi^{-1}(\frac{1}{n}\sum_{i=1}^n \psi(x_i))$，用RevNet做 $\psi$
    - 设计动机：NKM是首个可学习的Kolmogorov均值实现——$\psi$ 的形式决定了均值类型（线性=算术平均，log=几何平均，幂=幂均值），可逆网络足够表达

2. **理论保证**:

    - QUANN-1是均值可分解集合函数的通用近似器
    - QUANN-2（考虑元素对交互）更加强大
    - 在温和条件下也可近似max可分解函数

3. **嵌入质量**:

    - 可逆 $\psi$ 使NKM保持输入的结构信息→编码器学到更通用的嵌入
    - 实证：QUANN编码器迁移到非集合任务也效果好

### 损失函数 / 训练策略
- 标准监督学习，端到端训练
- $\psi$ 用RevNet架构实现可逆性

## 实验关键数据

### 集合函数任务


### 主实验

| 方法 | 集合分类 | 集合回归 | 点云分类 | 平均 |
|------|---------|---------|---------|------|
| DeepSets (sum) | 基线 | 基线 | 基线 | 基线 |
| PointNet (max) | 中 | 中 | 中 | 中 |
| HPDS (幂均值) | 好 | 好 | 好 | 好 |
| **QUANN-1** | **最优** | **最优** | **最优** | **SOTA** |

### 编码器迁移性


### 消融实验

| 配置 | 在非集合任务上的表现 |
|------|-------------------|
| DeepSets编码器 | 差→嵌入与sum池化强耦合 |
| PointNet编码器 | 差→嵌入与max池化强耦合 |
| **QUANN编码器** | **好→嵌入通用** |

### 关键发现
- NKM学到的池化形式介于sum和max之间→自动适应任务
- 可逆 $\psi$ 确保信息不丢失→编码器不需要为特定池化"补偿"
- QUANN在所有基准上超越SOTA，包括需要高阶交互的任务

## 亮点与洞察
- **Kolmogorov均值的神经化**：首次将百年数学概念（准算术均值）与现代深度学习结合。用可逆网络做生成函数，既有理论美感又实用。
- **解耦编码器与池化**：固定池化→编码器必须"适配"池化→嵌入不通用。可学习池化→编码器只需学好的嵌入→池化自动适配→嵌入迁移性增强。
- **可逆性的双重价值**：(1) 保证Kolmogorov均值有定义（需要可逆生成函数）；(2) 保持信息→不像max那样丢弃信息。

## 局限与展望
- RevNet增加了计算开销（虽然可逆性让我们不需要存储中间激活）
- QUANN-2对元素对的二次复杂度限制了大集合
- 仅在有限集合上实验，函数集合（连续集合）的情况未考虑
- 没有与Slot Attention等非Janossy方法充分比较

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Kolmogorov均值的神经化是优美的理论贡献
- 实验充分度: ⭐⭐⭐⭐ 多种任务+迁移性验证+消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论框架清晰，统一表格一目了然
- 价值: ⭐⭐⭐⭐ 对集合函数学习有基础性改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving the Effective Receptive Field of Message-Passing Neural Networks](../../ICML2025/llm_evaluation/improving_the_effective_receptive_field_of_message-passing_neural_networks.md)
- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](../../ICML2025/llm_evaluation/glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)
- [\[ICML 2026\] CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting](../../ICML2026/llm_evaluation/capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting.md)

</div>

<!-- RELATED:END -->
