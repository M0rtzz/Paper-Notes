---
title: >-
  [论文解读] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games
description: >-
  [AAAI 2026][行为博弈论] 提出 ElementaryNet，一种**可证明不具备策略性推理能力**的神经网络架构，用于建模博弈中人类的"level-0"（非策略性）行为，在预测准确率上与 GameNet（当前 SOTA）无统计差异，同时具备更好的可解释性。
tags:
  - AAAI 2026
  - 行为博弈论
  - 人类行为预测
  - 非策略性神经网络
  - 迭代推理
  - 可解释性
---

# ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games

**会议**: AAAI 2026  
**arXiv**: [2503.05925](https://arxiv.org/abs/2503.05925)  
**代码**: [https://github.com/gregdeon/elementarynet](https://github.com/gregdeon/elementarynet)  
**领域**: 其他  
**关键词**: 行为博弈论, 人类行为预测, 非策略性神经网络, 迭代推理, 可解释性

## 一句话总结

提出 ElementaryNet，一种**可证明不具备策略性推理能力**的神经网络架构，用于建模博弈中人类的"level-0"（非策略性）行为，在预测准确率上与 GameNet（当前 SOTA）无统计差异，同时具备更好的可解释性。

## 研究背景与动机

### 问题场景
在**非重复同时移动博弈（normal-form games）** 中，人类行为通常偏离经典博弈论的理性预测（如囚徒困境中人类可能选择非最优策略）。行为博弈论旨在构建更贴近真实人类行为的预测模型。

### 现有方法及其局限性
当前 SOTA 模型 **GameNet** 将神经网络嵌入 **量子认知层级模型（QCH）** 中，用可学习的 level-0 神经网络替代传统的均匀分布作为非策略基线行为，取得了远超传统模型的性能。然而，GameNet 最佳性能出现在**只使用 level-0（完全没有策略推理层）** 的情况下。这引出了核心问题：

- 是"迭代策略推理"本身就不是人类行为的好模型？
- 还是 GameNet 的 level-0 神经网络**过于灵活**，已经暗中模拟了策略推理？

### 关键发现与动机
本文作者**证明了 GameNet 的 level-0 层确实能表达策略行为**（具体地，它可以精确模拟"对 maxmax 的量子最佳响应"）。这意味着 GameNet 声称的"可解释性"是虚假的。因此，需要设计一个**在架构层面就不可能进行策略推理**的新网络，从而得到真正可信的可解释性结论。

## 方法详解

### 整体框架

ElementaryNet 的核心思路：在 GameNet 的 feature layers 中引入**信息瓶颈（information bottleneck）**，使其在数学上不可能表达策略行为。整体架构为：

$$f_i(G) = \sum_{p=1}^{P} w_p \cdot h_i^p(\Phi^p(G))$$

其中 $\Phi^p$ 是势函数（potential function），$h_i^p$ 是响应函数（response function），$w_p$ 是凸组合权重。

### 关键设计

#### 1. **GameNet 的策略性证明（定理2）**
- **核心思路**：构造性地给出 GameNet 参数的特定设置，使其精确计算"对 maxmax 策略的量子最佳响应"。
- **具体做法**：用 3 层隐藏层网络，通过 colmax/rowmax 操作提取对手的 maxmax 动作，然后计算己方的最优响应。
- **关键公式**：$M_c = \text{colmax}(U^2)$, $M_* = \text{rowmax}(M_c)$, $B = \text{relu}(M_c/C_{gap} - M_*/C_{gap} + 1)$
- **设计动机**：证明 GameNet 的"非策略性" level-0 层实际上可以表达策略行为，揭示了架构层面的根本缺陷。

#### 2. **Elementary Model 与信息瓶颈**
- **核心思路**：基于 elementary behavioral model 理论，通过势函数将两个玩家的效用压缩为单一势值，形成信息瓶颈。
- **势函数定义**：$\varphi^p(x, y) = \theta_x^p x + \theta_y^p y$（learned-potential 版本）
- **非编码性质**：线性势函数要么是 dictatorial（只依赖一个输入），要么是 non-encoding（存在任意远的不同输入映射到同一输出），两种情况都阻止了策略推理。
- **设计动机**：利用 elementary model 的数学保证——凸组合的 elementary model 一定是弱非策略性的（定理1, Wright & Leyton-Brown）。

#### 3. **两种实例化：learned-potential 与 fixed-potential**
- **Learned-potential**：$P$ 个可学习的线性势函数 $\varphi^p(x,y) = \theta_x^p x + \theta_y^p y$，参数通过训练学得。
- **Fixed-potential**：使用 4 个固定势函数：
    - $\varphi_{\text{own}}(x,y) = x$（己方收益）
    - $\varphi_{\text{opp}}(x,y) = y$（对方收益）
    - $\varphi_{\text{sum}}(x,y) = x + y$（社会福利）
    - $\varphi_{\text{diff}}(x,y) = x - y$（公平性）
- **设计动机**：fixed-potential 对应认知心理学中已知的启发式策略，可用于研究人类行为到底主要受哪些因素驱动。

#### 4. **非策略性的形式化证明（定理3）**
- **证明方法**：线性势函数的等值线要么轴对齐（dictatorial）要么无界延伸（non-encoding），两种情况都满足 elementary model 的定义。
- **含义**：ElementaryNet 的每个组件都是 elementary model，其凸组合必然是弱非策略性的——它无法表达"对任何 dominance-responsive 模型的量子最佳响应"。

### 损失函数 / 训练策略

- **损失函数**：使用 squared L2 error（预测分布与经验分布之间的误差），遵循已有文献建议。
- **训练方式**：60/20/20 划分（训练/验证/测试），遍历 36 组超参数（L1 正则系数、dropout 概率、初始 QCH 参数），选验证损失最低的模型。
- **统计方法**：50 次随机划分，报告与参考模型的配对差异（消除数据划分带来的方差），使用 BCa 自助法置信区间。

## 实验关键数据

### 数据集
来自 12 项实验研究的聚合数据集，共 26,553 个观测值，366 个不同的博弈。涵盖多种来源包括 Amazon Mechanical Turk 大规模实验。

### 主实验

| 模型 | 配置 | 相对于 Uniform+QCHp 的损失改善 | 备注 |
|------|------|-----|------|
| GameNet + QCHp | 1层, 50单元 | 最佳性能 | 更深的模型过拟合 |
| ElementaryNet + QCHp | 1层, 50单元, 1 learned potential | **无统计差异** | 与 GameNet 最佳性能等价 |
| Uniform + QCHp | — | 基线 | 传统方法 |
| ElementaryNet (无QCH) | 1-3层 | 显著更差 | 证明策略推理是必要的 |

**核心结论**：尽管增加了严格的架构约束，ElementaryNet + QCHp 达到了与 GameNet 不可区分的预测性能。

### 消融实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| ElementaryNet + QCHp（完整模型） | 最佳 | SOTA |
| ElementaryNet 无 QCH（纯非策略） | 极差（比 Uniform baseline 还差） | **证明迭代推理确实是人类行为的好模型** |
| ElementaryNet + QCH1/QCH2/QCH3 | 与 QCHp 无统计差异 | 推理层级的具体分布形式不太重要 |
| Fixed-potential（仅 own） | 接近 Uniform baseline | 仅用己方收益不足以建模非策略行为 |
| Fixed-potential（4 个固定势函数） | 优于 own，但不如 learned | 认知心理学启发式有价值但不够 |

### 关键发现

1. **迭代推理是好的人类行为模型**：纯非策略模型（没有 QCH）表现极差，这是在排除了 GameNet 的"伪 level-0 可模拟策略"混淆因素后的可靠结论。
2. **70%+ 的人是 level-0**：与 ElementaryNet 配合训练的 QCH 模型将 70%+ 概率分配给 level-0，远高于 Uniform + QCHp 的 33%，说明大量人类受试者进行丰富的非策略性推理。
3. **对手收益信息很重要**：仅使用己方收益的简单 level-0 模型表现差，证明人类的非策略行为确实会考虑对手的收益。
4. **更丰富的势函数有价值**：learned-potential 优于 fixed-potential，暗示存在超越福利/公平性之外的更细致的非策略推理模式。

## 亮点与洞察

1. **理论-实践闭环**：先证明 GameNet 的缺陷（theory），再设计新架构（design），最后实验验证（empirical），逻辑链条完整。
2. **可解释性有实质内容**：不是简单地声称"因为架构简单所以可解释"，而是通过数学证明网络无法表达特定行为类别，从而使后续分析具有因果推断性质。
3. **负面结果也有价值**：证明 GameNet 的 level-0 可模拟策略，解释了为什么添加策略推理层级反而没有收益——这是对先前"反直觉"实验结果的有力解释。

## 局限与展望

1. **仅限 2 人 normal-form 博弈**：不含序贯交互、随机事件或不完全信息，距离真实场景有显著差距。
2. **响应函数仍是黑箱**：虽然势函数可解释，但后面接的响应函数 $h_i^p$ 仍然是不可解释的神经网络。
3. **势函数限于线性**：非线性势函数可能更强但也可能破坏非策略性保证，需要进一步理论研究。
4. **数据规模有限**：366 个博弈、26K 观测值相对于模型复杂度来说不大，导致高方差。

## 相关工作与启发

- **QCH 模型（Camerer 2004）**：迭代推理的经典框架，本文在此基础上替换了 level-0 组件。
- **GameNet**：使用排列等变神经网络替代手工 level-0，是本文的直接改进目标。
- **Elementary models（Wright & Leyton-Brown）**：提供了"非策略性"的形式化定义和凸组合保持非策略性的定理，是 ElementaryNet 的理论基础。
- **启发**：在需要可解释性的场景中，与其事后解释网络行为，不如在架构层面**内嵌理论保证**（design for interpretability）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将形式博弈论概念嵌入神经网络架构设计中，理论与实践结合非常巧妙
- 实验充分度: ⭐⭐⭐⭐ — 消融实验全面，但数据集有限
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，实验逻辑通顺
- 价值: ⭐⭐⭐⭐ — 对行为博弈论领域有重要意义，但应用范围相对狭窄

<!-- RELATED:START -->

## 相关论文

- [A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](../../NeurIPS2025/interpretability/fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)
- [FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)

<!-- RELATED:END -->
