---
title: >-
  [论文解读] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering
description: >-
  [AAAI 2026][多重聚类] 本文提出Multi-DProxy框架，通过门控跨模态融合、双约束代理优化和动态候选词管理三大创新机制，利用可学习的文本代理实现个性化多重聚类，在全部公开基准上达到SOTA。
tags:
  - AAAI 2026
  - 多重聚类
  - 跨模态融合
  - 代理学习
  - 动态候选词管理
  - CLIP
---

# Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering

**会议**: AAAI 2026  
**arXiv**: [2511.07274](https://arxiv.org/abs/2511.07274)  
**代码**: 无（论文提到Supplementary Material中有匿名代码仓库）  
**领域**: 多模态VLM  
**关键词**: 多重聚类, 跨模态融合, 代理学习, 动态候选词管理, CLIP

## 一句话总结
本文提出Multi-DProxy框架，通过门控跨模态融合、双约束代理优化和动态候选词管理三大创新机制，利用可学习的文本代理实现个性化多重聚类，在全部公开基准上达到SOTA。

## 研究背景与动机

**领域现状**：聚类是无监督学习的基石，旨在根据内在相似性发现数据的潜在结构。传统聚类仅产生单一分区，忽略了数据可以从多个视角进行有意义分组的固有复杂性。多重聚类（Multiple Clustering）试图发现互补的多样化分区，但现有方法穷举所有可能的聚类，不关注用户兴趣，导致用户需要人工筛选——这是一个显著的实用瓶颈。

**现有痛点**：近期Multi-MaP和Multi-Sub利用CLIP进行代理学习，用文本提示引导兴趣偏向的嵌入提取。然而这些方法存在两个核心缺陷：

**静态语义刚性**：预定义的候选词（如为"颜色"概念生成"红"、"蓝"、"绿"）无法适应特定数据集的概念，当LLM的建议与真实类别不匹配时产生语义对齐偏差

**不灵活的特征融合**：固定融合策略（拼接或简单平均）忽视了模态间不断演化的特征交互，产生次优的联合表示

**核心矛盾**：用户期望通过简单的概念关键词（如"颜色"）获取与兴趣一致的聚类结果，但静态的文本代理和固定的融合策略无法捕捉数据集特有的语义结构和动态的模态交互。

**本文切入角度**：
- 用**可学习**的文本代理替代静态候选词
- 用**门控机制**实现自适应跨模态融合
- 用**迭代反馈**让候选词集随聚类结构动态演化

## 方法详解

### 整体框架
Multi-DProxy基于冻结的CLIP编码器（视觉 $f_v(\cdot)$ 和文本 $f_t(\cdot)$），核心流程：
1. 用户指定兴趣概念 $u$（如"color"）
2. GPT-4生成初始候选词集 $\mathcal{C}$
3. 为每张图像初始化可学习代理 $\mathbf{w}_i$
4. 通过门控跨模态融合得到联合表示 $\mathbf{F}$
5. 双约束优化代理嵌入
6. 动态候选词管理每 $R$ 个epoch更新候选集
7. 最终用融合特征 $\mathbf{F}$ 做K-means聚类

### 关键设计

#### 1. **门控跨模态融合（Gated Cross-Modal Fusion）**

核心思路是通过分层双向注意力和自适应特征重校准，动态合成判别性联合表示。

**双向交叉注意力**：视觉特征注意文本，文本特征注意视觉：
$$\mathbf{V}_{\text{attn}}^l = \text{MultiHead}(\mathbf{V}^{l-1}, \mathbf{T}^{l-1}, \mathbf{T}^{l-1})$$
$$\mathbf{T}_{\text{attn}}^l = \text{MultiHead}(\mathbf{T}^{l-1}, \mathbf{V}^{l-1}, \mathbf{V}^{l-1})$$

**门控残差融合**：通过sigmoid门控控制注意力信息的融入程度：
$$\mathbf{V}^l = \mathbf{V}^{l-1} + \sigma(\mathbf{W}_g^{\mathbf{V}}[\mathbf{V}^{l-1}; \mathbf{V}_{\text{attn}}^l]) \odot \mathbf{V}_{\text{attn}}^l$$

**自适应特征融合**：基于温度缩放的余弦相似度动态平衡两个模态的贡献：
$$\mathbf{F} = \lambda \mathbf{T}^L + (1-\lambda)\mathbf{V}^L, \quad \lambda = \sigma\left(\frac{\langle \mathbf{T}^L, \mathbf{V}^L \rangle}{\tau}\right)$$

设计动机：固定融合（拼接/平均）无法捕捉不同样本中模态重要性的差异。门控机制让模型根据模态间一致性动态调整权重，$\lambda$ 随训练自适应变化。

#### 2. **双约束代理优化（Dual-Constraint Proxy Optimization）**

**用户兴趣约束**：确保代理与领域概念语义一致。每个代理通过注意力加权组合候选词嵌入：
$$\mathbf{w}_i = \sum_{k=1}^{|\mathbf{C}|} \alpha_{ik} \mathbf{c}_k, \quad \alpha_{ik} = \frac{\exp(\mathbf{w}_i^{\prime \top} \mathbf{c}_k / \tau_\alpha)}{\sum_j \exp(\mathbf{w}_i^{\prime \top} \mathbf{c}_j / \tau_\alpha)}$$

语义一致性损失最小化代理与候选词质心的偏差：
$$\mathcal{L}_u = \frac{1}{D} \sum_{i=1}^{D} \|\mathbf{w}_i - \bar{\mathbf{c}}\|_2^2$$

**概念判别约束**：通过对比学习增强聚类可分离性：
$$\mathcal{L}_c = \frac{1}{B} \sum_{i=1}^{B} \log \sum_{j \neq i} \exp(\mathbf{f}_i^{\top} \mathbf{w}_j / \sigma)$$

设计动机：$\mathcal{L}_u$ 将代理锚定在有意义的语义空间中（防止漂移），$\mathcal{L}_c$ 通过困难样本挖掘拉开不同聚类的表示距离。二者互补，前者保证语义相关性，后者保证聚类判别性。

#### 3. **动态候选词管理（Dynamic Candidate Management）**

每 $R$ 个epoch执行一次更新：
1. 收集所有代理嵌入 $\mathbf{W}$
2. 对代理做K-means聚类，得到 $M$ 个聚类中心
3. 计算每个候选词与所有聚类中心的平均余弦相似度
4. 保留Top-K（$K=|\mathcal{C}|/2$）得分最高的候选词
5. 重新计算候选词嵌入

初始生成 $2^\beta M$ 个候选词（$\beta = E/R$），经过 $E$ 个epoch训练后最终收敛到 $M$ 个，与真实类别数对齐。

设计动机：LLM生成的候选词可能包含与数据集无关的概念。通过与聚类结构的迭代对齐，逐步淘汰无关候选词，保留数据集特有的语义概念。

### 损失函数 / 训练策略
统一损失函数：
$$\mathcal{L} = \underbrace{\frac{1}{D}\sum_{i=1}^{D}(1-\cos(\mathbf{f}_i, \mathbf{v}_i))}_{\text{跨模态对齐} \mathcal{L}_a} + \alpha(t)\mathcal{L}_u + \beta(t)\mathcal{L}_c$$

约束权重采用自适应调度：
- $\alpha(t) = \min(0.5, 0.1 + 0.4 \cdot t/E)$：线性增长，逐步增强语义约束
- $\beta(t) = 0.1 \times (1 - \cos(\pi t / E))$：余弦调度，平稳增强判别约束

训练1000个epoch，Adam优化器（动量0.9），RTX 4090 GPU。

## 实验关键数据

### 主实验（NMI / RI 指标，越高越好）

| 方法 | Fruit-Color NMI | Fruit-Species NMI | Card-Suits NMI | CMUface-Identity NMI | CIFAR10-Type NMI |
|------|----------------|-------------------|----------------|---------------------|------------------|
| MSC | 0.6886 | 0.1627 | 0.0497 | 0.3892 | 0.1547 |
| ENRC | 0.7103 | 0.3187 | 0.0676 | 0.5607 | 0.1826 |
| AugDMC | 0.8517 | 0.3546 | 0.0873 | 0.5875 | 0.2855 |
| Multi-MaP | 0.8619 | 1.0000 | 0.2734 | 0.6625 | 0.4969 |
| Multi-Sub | 0.9693 | 1.0000 | 0.3104 | 0.7441 | 0.5271 |
| **Multi-DProxy** | **1.0000** | **1.0000** | **0.5008** | **0.7609** | **0.5863** |

### 消融实验

| 配置 | 主要变化 | 说明 |
|------|---------|------|
| w/o-Dynamic | 去除动态候选词管理 | 性能下降，候选词无法适应数据 |
| w/o-UConstraints | 去除用户兴趣约束 | 代理缺乏语义锚定 |
| w/o-CConstraints | 去除概念判别约束 | 聚类判别性不足 |
| w/o-GFusion | 拼接替代门控融合 | **影响最大**，验证跨模态融合核心地位 |
| -T（仅文本） | 只用文本模态 | 单模态可独立聚类但效果有限 |
| -V（仅视觉） | 只用视觉模态 | 融合表示显著优于单模态 |

### Zero-shot对比

| 方法 | Fruit-Color | Stanford Cars-Color | CIFAR10-Type |
|------|------------|--------------------|----|
| CLIP_GPT (zero-shot) | 0.7912 | 0.6539 | 0.4935 |
| CLIP_label (真实标签) | 0.8629 | 0.6830 | 0.5087 |
| **Multi-DProxy** | **1.0000** | **0.7610** | **0.5863** |

### 关键发现
1. Multi-DProxy在所有数据集和聚类维度上一致超越现有SOTA，NMI和RI均显著提升
2. 门控跨模态融合是最关键的组件——去除后性能下降最大
3. 即使与使用真实标签的CLIP zero-shot对比，Multi-DProxy在多数场景仍更优，说明学习到的表示更全面
4. 动态候选词管理从初始的泛化概念逐步收敛到数据特定语义，有效解决了静态刚性问题
5. 理论分析证明了代理更新的稳定性（Proposition 1）和视觉特征门控文本学习的机制（Theorem 1）

## 亮点与洞察
1. **首个统一可学习代理+动态候选词+自适应融合的多重聚类框架**，三个创新点互相配合
2. **理论分析扎实**：Proposition 1量化了候选词更新对代理漂移的控制，Theorem 1揭示了视觉特征如何作为门控信号调制文本表示学习
3. **候选词从"宽"到"精"的演化过程**：初始生成$2^\beta M$个→每R个epoch减半→最终收敛到M个，与真实类别数自然对齐
4. **自适应权重调度**：避免了手动调参，在不同数据集上表现一致

## 局限与展望
1. 依赖GPT-4生成初始候选词，候选词质量受LLM能力影响
2. 用户需要指定聚类数量$M$，实际应用中可能未知
3. 训练需要1000个epoch，计算成本较高
4. 仅在视觉数据集上验证，未扩展到文本或其他模态的多重聚类
5. 候选词更新间隔$R$需要手动设置，可考虑自适应调整策略

## 相关工作与启发
- 代理学习（Proxy Learning）思路可以推广到其他需要对齐用户意图的无监督任务
- 动态候选词管理的"生成-评估-筛选"闭环可以启发其他LLM辅助的特征学习方法
- 门控跨模态融合机制对其他多模态任务（如检索、分类）也有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [Evaluating Multiple Models Using Labeled and Unlabeled Data](../../NeurIPS2025/social_computing/evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](../../ACL2025/social_computing/llm_personalized_disinformation.md)

<!-- RELATED:END -->
