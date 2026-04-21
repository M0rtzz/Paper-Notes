---
title: >-
  [论文解读] LaT-IB：面向标签噪声鲁棒的信息瓶颈学习
description: >-
  [AAAI 2026][信息瓶颈] 本文揭示了信息瓶颈（IB）原理在标签噪声下的固有脆弱性，提出 LaT-IB 方法，通过将表征解耦为干净标签空间和噪声标签空间两部分，结合"最小-充分-干净"（MSC）准则和三阶段训练框架，在多种噪声条件下实现了对现有 IB 方法的显著超越。
tags:
  - AAAI 2026
  - 信息瓶颈
  - 标签噪声
  - 表示学习
  - 鲁棒性
  - 互信息
---

# Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning

**会议**: AAAI 2026  
**arXiv**: [2512.10573](https://arxiv.org/abs/2512.10573)  
**代码**: https://github.com/RingBDStack/LaT-IB  
**领域**: 模型压缩  
**关键词**: 信息瓶颈, 标签噪声, 表示学习, 鲁棒性, 互信息  
# LaT-IB：面向标签噪声鲁棒的信息瓶颈学习

## 一句话总结

本文揭示了信息瓶颈（IB）原理在标签噪声下的固有脆弱性，提出 LaT-IB 方法，通过将表征解耦为干净标签空间和噪声标签空间两部分，结合"最小-充分-干净"（MSC）准则和三阶段训练框架，在多种噪声条件下实现了对现有 IB 方法的显著超越。

## 研究背景与动机

**信息瓶颈的局限性**：IB 原理通过最大化 $I(Y;Z)$ 保留任务相关信息、最小化 $I(X;Z)$ 压缩冗余信息，但其强依赖准确标签使其在标签噪声下极其脆弱——当 $Y$ 被污染时，最大化 $I(Y;Z)$ 实际上会让表征 $Z$ 捕获噪声 $Y_n$。

**实验验证脆弱性**：在 CIFAR-10 上，VIB 在 50% 对称噪声下准确率骤降至 10.0%；在 Cora 图上，GIB 在 40% 噪声下从 69.5% 持续衰退至 55.1%，表明现有 IB 方法无法有效应对标签噪声。

**两阶段方法不理想**：先去噪再应用 IB 的级联策略存在理论上的累积退化问题——作者证明了级联模型的误差下界高于端到端模型（Theorem 1.1），信息路径延长导致不可避免的信息损失。

**噪声普遍存在**：现实场景中标签噪声极为常见，无论是图像标注还是图数据的节点标注，都不可避免地受到噪声和意外因素的干扰，严重影响模型性能。

**表征层面约束缺失**：现有标签噪声学习方法（样本选择、鲁棒损失、数据增强等）大多忽略了表征层面的约束，难以在严重噪声或分布偏移下学习到任务相关且噪声不变的特征。

**端到端解决需求**：需要一种统一的方法，能够在 IB 框架内部同时完成去噪和特征提取，而非依赖外部去噪模块——这要求重新设计 IB 目标函数和优化策略。

## 方法详解

### 核心思想：最小-充分-干净（MSC）准则

LaT-IB 的核心是将潜在表征分解为两部分：$S$（对齐干净标签空间）和 $T$（对齐噪声空间），目标函数为：

$$\min \underbrace{-I(Y;S,T)}_{\text{充分性}} + \beta \underbrace{I(\mathcal{D};S,T)}_{\text{最小性}} + \gamma \underbrace{I(S;T|Y)}_{\text{干净性}}$$

约束条件：$\max(I(Y_n;S), I(Y_c;T)) \leq K$

### 理论基础

- **Lemma 4.1（冗余不变性）**：优化预测项和压缩项可减少模型对 $Y$ 无关特征 $\mathcal{D}_n$ 的学习
- **Lemma 4.2（特征收敛）**：当 $I(Y_n;S)$ 和 $I(Y_c;T)$ 足够小时，优化解耦项可强化 $S \to Y_c$ 和 $T \to Y_n$ 的映射
- 通过上下界分析（Proposition 4.1-4.4），将不可直接优化的多变量互信息转化为可实现的损失函数

### 双编码器架构

采用双编码器-单解码器架构：两个编码器分别将输入映射到高维高斯分布，通过重参数化技巧采样得到 $S$ 和 $T$，共享解码器进行预测。

### 三阶段训练框架

**阶段一：预热（Warmup）**——使用全数据集预训练 $\text{encoder}_S$，建立基本判别能力，损失为标准交叉熵 $\mathcal{L}_{CE}(\hat{y}_S, y)$。

**阶段二：知识注入（Knowledge Injection）**——通过 InfoJS 选择器（基于互信息 $I(S;Y)$ 和 JS 散度 $D_{JS}(S \| T)$）将样本分为干净集、噪声集和不确定集。对干净集和噪声集最大化编码器间散度（促进差异化），对不确定集最小化散度（引导学习），同时加入最小表征正则项。

**阶段三：鲁棒训练（Robust Training）**——优化完整目标函数，引入 ConCE 损失（$\sum \min(\mathcal{L}_{CE}(\hat{y}_S, y), \mathcal{L}_{CE}(\hat{y}_T, y))$ 的光滑近似）促进编码器一致性，并通过判别器交替训练最小化条件互信息 $I(S;T|Y)$。

## 实验结果

### 实验一：真实标签噪声下的图像分类（CIFAR-10N/100N）

| 方法 | CIFAR-10N aggre | CIFAR-10N worst | CIFAR-100N noisy100 | Animal-10N |
|------|:-:|:-:|:-:|:-:|
| VIB | 86.11 | 73.80 | 53.29 | 76.28 |
| (ELR+)+VIB | 92.65 | 86.68 | 61.06 | 85.87 |
| Promix+VIB | 92.35 | **91.24** | **63.91** | 85.47 |
| **LaT-IB** | **94.17** | 87.95 | 63.59 | **88.49** |

LaT-IB 在 CIFAR-10N aggre 上以 94.17% 大幅超越所有基线，在 Animal-10N 上提升 2.6 个百分点。

### 实验二：对抗攻击下的鲁棒性（CIFAR-10N + FGSM）

| 方法 | aggre (无攻击) | aggre (ε=0.1) | worst (无攻击) | worst (ε=0.1) |
|------|:-:|:-:|:-:|:-:|
| VIB | 86.11 | 43.18 | 73.80 | 36.56 |
| Promix+VIB | 92.35 | 36.43 | 91.24 | 36.05 |
| **LaT-IB** | **94.17** | **60.66** | **87.95** | **54.18** |

两阶段方法在对抗攻击下性能暴跌（Promix+VIB 从 92.35→36.43），而 LaT-IB 展现出极强的鲁棒性（94.17→60.66），在 ε=0.1 下仍领先约 14-18 个百分点。

### 图节点分类（Pubmed，40% 均匀噪声）

LaT-IB 达到 73.40%，大幅超越 GIB（64.30%）和各种改进方法，表明跨域泛化性强。

## 亮点与创新

- **首次系统揭示 IB 对标签噪声的脆弱性**，并从理论上证明两阶段方法的次优性
- **MSC 准则统一了去噪和表征学习**，无需外部去噪模块即可在 IB 框架内实现噪声分离
- **三阶段渐进训练策略设计精巧**，从预热→注入→鲁棒分阶段引导表征解耦
- **在对抗攻击+标签噪声的复合场景下表现卓越**，展示了"最小充分"属性带来的天然鲁棒性
- **跨任务跨域有效**，同时适用于图像分类和图节点分类

## 局限性

- **训练管线复杂**：三阶段训练需要调节多个阶段相关超参数（$E_{Warmup}$、$E_{Injection}$、$\beta$、$\gamma$、$\delta$），调参成本较高
- **InfoJS 选择器依赖阈值**：样本划分质量受 $\delta$ 影响大，过小则训练数据不足，过大则两个编码器趋于收敛
- **噪声假设有限**：主要验证了对称噪声、非对称噪声和实例相关噪声，对更复杂的噪声模式（如开放集噪声）未做讨论
- **图任务数据规模较小**：图分类实验仅在 Cora、Citeseer、Pubmed 等小规模图上验证，大规模图的可扩展性有待考察
- **理论假设的实际满足度**：Lemma 4.2 要求 $\max(I(Y_n;S), I(Y_c;T))$ 足够小，实际中此条件的满足程度难以衡量

## 相关工作对比

| 维度 | LaT-IB | VIB/GIB |
|------|--------|---------|
| 噪声处理 | 内置噪声感知解耦机制 | 无噪声处理，直接最大化 $I(Y;Z)$ |
| 表征结构 | 双编码器分离干净/噪声 | 单一表征空间 |
| 对抗鲁棒 | 保留最小充分性 + 噪声分离 | 仅最小充分性，无噪声分离 |

| 维度 | LaT-IB | Denoise + IB（如 Promix+VIB） |
|------|--------|------|
| 架构 | 端到端统一框架 | 级联两阶段 |
| 理论保证 | 累积退化定理证明优于级联 | 存在信息传递损失 |
| 对抗场景 | 一次攻击面 | 两阶段均可被攻击，脆弱性翻倍 |

## 评分

- ⭐⭐⭐⭐ 新颖性：首次从理论和实验两方面揭示 IB 对标签噪声的脆弱性，MSC 准则设计巧妙
- ⭐⭐⭐⭐ 技术深度：理论推导严谨（多个命题+引理），上下界分析和渐进训练框架配合良好
- ⭐⭐⭐⭐ 实验充分性：覆盖图像和图两大领域、真实和合成噪声、对抗攻击等多种设置
- ⭐⭐⭐ 实用性：三阶段训练管线较复杂，超参数较多，工程落地需要较多调优工作

<!-- RELATED:START -->

## 相关论文

- [EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](../../ICLR2026/knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](../../ACL2025/knowledge_editing/memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [Context-Robust Knowledge Editing for Language Models](../../ACL2025/knowledge_editing/context-robust_knowledge_editing_for_language_models.md)
- [Rote Learning Considered Useful: Generalizing over Memorized Training Examples](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_data_in_llms.md)

<!-- RELATED:END -->
