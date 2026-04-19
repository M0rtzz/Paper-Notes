---
title: >-
  [论文解读] P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials
description: >-
  [NeurIPS 2025 (Workshop: ML4PS)][图学习][不确定性量化] 提出 P-DRUM，一种简单高效的事后（post-hoc）不确定性量化框架，利用已训练图神经网络势的描述子来估计预测残差，作为不确定性代理，无需修改原模型架构或训练流程。
tags:
  - "NeurIPS 2025 (Workshop: ML4PS)"
  - 图学习
  - 不确定性量化
  - 机器学习原子间势
  - 残差建模
  - MACE
  - 分布外检测
---

# P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials

**会议**: NeurIPS 2025 (Workshop: ML4PS)  
**arXiv**: [2509.02927](https://arxiv.org/abs/2509.02927)  
**代码**: 无  
**领域**: 图神经网络 / 计算化学  
**关键词**: 不确定性量化, 机器学习原子间势, 残差建模, MACE, 分布外检测

## 一句话总结

提出 P-DRUM，一种简单高效的事后（post-hoc）不确定性量化框架，利用已训练图神经网络势的描述子来估计预测残差，作为不确定性代理，无需修改原模型架构或训练流程。

## 研究背景与动机

**机器学习原子间势（MLIPs）** 正在变革材料科学，以量子力学精度但高出数个数量级的计算效率实现原子尺度模拟。然而，MLIPs 的**预测可靠性**仍是关键问题，特别是对训练分布外的原子构型。

**现有不确定性量化（UQ）方法的局限**：

| 方法 | 优势 | 局限 |
|------|------|------|
| **集成方法** | 效果最好，被视为金标准 | 需训练和运行多个模型，计算成本高 |
| **MC Dropout** | 利用推理时 dropout | 部分模型（如 MACE）不默认支持；可能影响精度 |
| **深度核学习** | 结合 NN 和高斯过程 | 需要修改训练流程 |
| **kNN / GMM** | 事后方法，利用描述子空间 | 不利用预测误差信息 |

**核心动机**：能否设计一种**事后**方法，仅使用已训练模型的描述子来估计预测误差，既不需要修改模型也不需要训练日志？

## 方法详解

### 整体框架

P-DRUM 的核心思路分三步：

1. **提取描述子**：给定已训练的 MACE 模型，对每个结构 $X$ 中的每个原子 $j$ 提取描述子 $D_{ij} \in \mathbb{R}^{d_{\text{desc}}}$
2. **计算残差**：能量残差 $\Delta E = E - \hat{E}$，力残差 $\Delta \mathbf{F} = \mathbf{F} - \hat{\mathbf{F}}$
3. **训练轻量 MLP**：用描述子作为输入、残差作为目标，训练 MLP 预测残差，残差大小作为不确定性指标

训练数据构造：$\mathcal{S}_\Delta = \{D(X_i), \Delta E_i, \Delta \mathbf{F}_i\}_{i=1}^N$

### 关键设计

#### 1. 能量残差学习

为保持**置换不变性**和对不同原子数的**灵活性**，设计了逐原子的标量函数 $r^s: \mathbb{R}^{d_{\text{desc}}} \to \mathbb{R}$，能量残差建模为原子贡献之和。

**误差范数学习（norm）**：
$$\mathcal{L}_{\text{E-norm}}(X_i) = \left(\sum_j^{n_i} r_{\text{E-norm}}^s(D_{ij}) - |\Delta E_i|\right)^2$$

**偏差学习（diff）**：
$$\mathcal{L}_{\text{E-diff}}(X_i) = \left(\sum_j^{n_i} r_{\text{E-diff}}^s(D_{ij}) - \Delta E_i\right)^2$$

区别在于是否保留误差的符号信息。

#### 2. 力残差学习

直接在逐原子级别预测力残差：

**误差范数学习（norm）**：预测力误差的欧几里得范数
$$\mathcal{L}_{\text{F-norm}}(X_{ij}) = \left(r_{\text{F-norm}}^s(D_{ij}) - \|\Delta \mathbf{F}_{ij}\|\right)^2$$

**偏差学习（diff）**：预测三维力误差向量
$$\mathcal{L}_{\text{F-diff}}(X_{ij}) = \frac{1}{3}\sum\left(r_{\text{F-diff}}^v(D_{ij}) - \Delta \mathbf{F}_{ij}\right)^2$$

其中 $r_{\text{F-diff}}^v: \mathbb{R}^{d_{\text{desc}}} \to \mathbb{R}^3$ 是向量值函数。

### 损失函数 / 训练策略

- **基础模型**：MACE，32 通道，5Å 截断，2 个交互层，64 维描述子
- **P-DRUM MLP**：1-2 个隐藏层，ReLU 激活（norm 版本输出前用 softplus）
- **学习率调度**：初始 $10^{-3}$，patience=10 减半，最低 $10^{-7}$
- **最大训练**：1000 epochs + 早停
- **批大小**：64 原子（力训练），64 结构（能量训练）；大数据集用 2048
- **计算开销**：仅需一次 MACE 前向传播 + 微小额外开销（vs 集成方法的 5 次前向传播）

## 实验关键数据

### 主实验：域内不确定性-误差相关性（Spearman 相关）

| 误差类型 | 方法 | Uracil | Salicylic | Malondialdehyde | Ni₃Al | HME21 |
|---------|------|--------|-----------|-----------------|-------|-------|
| 能量 | Ensemble | 0.04 | 0.08 | -0.01 | 0.39 | 0.27 |
| | MC-dropout | -0.02 | -0.02 | -0.03 | -0.05 | 0.20 |
| | GMM | 0.07 | 0.07 | 0.13 | 0.64 | 0.06 |
| | kNN | 0.06 | 0.06 | 0.09 | 0.64 | -0.05 |
| | **P-DRUM-norm** | 0.12 | -0.01 | -0.09 | 0.62 | **0.30** |
| | **P-DRUM-diff** | **0.18** | **0.16** | **0.21** | **0.87** | 0.26 |
| 力 | Ensemble | 0.68 | 0.65 | 0.69 | 0.97 | 0.78 |
| | MC-dropout | 0.24 | 0.27 | 0.27 | 0.87 | 0.68 |
| | GMM | 0.58 | 0.67 | 0.68 | 0.96 | 0.64 |
| | kNN | 0.52 | 0.61 | 0.65 | 0.96 | 0.54 |
| | **P-DRUM-norm** | **0.67** | **0.71** | 0.69 | **0.98** | **0.92** |
| | **P-DRUM-diff** | 0.53 | 0.58 | 0.57 | 0.96 | 0.85 |

### OOD 检测实验（Ni₃Al 数据集）

| 方法 | 高温 Corr. | 六方 AUC | 立方 AUC | 原子交换 AUC | 总体 Corr. |
|------|----------|---------|---------|------------|-----------|
| Ensemble | 0.98 | 0.94 | 1.00 | 1.00 | 0.90 |
| MC-dropout | 0.92 | 0.63 | 0.84 | 0.82 | 0.72 |
| GMM | 0.98 | 1.00 | 1.00 | 1.00 | 0.81 |
| kNN | 0.98 | 0.99 | 1.00 | 1.00 | 0.82 |
| **P-DRUM-norm** | 0.99 | 0.82 | 0.82 | 0.99 | 0.82 |
| **P-DRUM-diff** | 0.97 | 0.97 | 1.00 | 1.00 | **0.87** |

OOD 检测包括四种分布外场景：高温分子动力学（2000K/3000K vs 训练时的 500K-1500K）、不同晶相（六方/立方）、随机原子位置交换。

### 关键发现

1. **P-DRUM-diff 在能量 UQ 上表现最佳**：保留误差符号有助于能量残差（标量）学习
2. **P-DRUM-norm 在力 UQ 上表现最佳**：将三维力误差压缩为范数降低了学习难度
3. **HME21（37 种元素）上 P-DRUM 优势突出**：当数据集元素多样性高时，仅靠描述子空间密度（kNN/GMM）不足以捕捉误差相关性，需要显式利用误差信号
4. **OOD 检测中 P-DRUM-norm 表现较弱**：将误差压缩为范数可能丢失对 OOD 检测重要的方向信息
5. **P-DRUM-diff 综合表现最优**：在域内能量 UQ 和 OOD 检测上均表现出色

## 亮点与洞察

- **事后方法的实用性**：不需要修改模型架构、训练流程或保存训练日志，可直接应用于任何已训练的 MACE 模型
- **计算效率高**：仅需一次 MACE 前向传播（vs 集成方法的 5 次），额外开销可忽略
- **PCA 分析揭示了 P-DRUM 优势来源**：在 HME21 中，描述子空间的高密度区域可能具有较高预测误差（与直觉相反），kNN/GMM 无法捕捉这种关系，而 P-DRUM 通过学习误差信号可以
- **置换不变性的自然保持**：通过逐原子操作 + 求和保持了分子系统的对称性

## 局限与展望

1. **训练集复用问题**：用同一数据集训练 MLIP 和 P-DRUM 可能引入偏差，使用独立数据集则减少可用样本
2. **norm vs diff 的选择困境**：两种变体在不同任务上各有优劣，缺乏统一最优方案
3. **仅在 MACE 上验证**：未测试其他 MLIP 架构（如 NequIP、SchNet）
4. **OOD 检测中 P-DRUM-norm 的弱点**：信息压缩可能不利于分布外检测
5. **未探索主动学习应用**：P-DRUM 的不确定性估计可直接用于主动学习选样

## 相关工作与启发

- **LTAU（Loss Trajectory Analysis for UQ）**：需要记录训练过程中每个原子的损失轨迹，P-DRUM 更轻量
- **Orb-v3 的 pLDDT 式方法**：与模型训练联合优化 UQ，P-DRUM 作为事后替代方案
- **AlphaFold 的 pLDDT**：离散化预测误差的思想被 Orb-v3 引入到 MLIP 领域
- **描述子在下游任务中的有效性**：P-DRUM 为描述子的新用途（不确定性估计）提供了证据

## 评分

- 新颖性：⭐⭐⭐⭐ — 简单但有效的事后 UQ 框架
- 理论贡献：⭐⭐⭐ — 更多是实验驱动的工作
- 实验充分度：⭐⭐⭐⭐ — 多数据集、多基线、含 OOD 评估和 PCA 分析
- 实用价值：⭐⭐⭐⭐⭐ — 即插即用，对 MLIP 社区有直接价值
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Generative Graph Pattern Machine](generative_graph_pattern_machine.md)
- [Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](../../ICML2025/graph_learning/neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)

<!-- RELATED:END -->
