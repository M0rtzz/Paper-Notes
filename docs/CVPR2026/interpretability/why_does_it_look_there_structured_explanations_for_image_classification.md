---
title: >-
  [论文解读] Why Does It Look There? Structured Explanations for Image Classification
description: >-
  [CVPR 2026][结构化解释] 提出 I2X 框架，通过在训练检查点追踪从 GradCAM 提取的原型强度与模型置信度的协同演化，将非结构化的可解释性（显著性图）转化为结构化的可解释性，揭示模型"为什么关注那里"的推理结构，并利用这种理解指导微调提升性能。
tags:
  - CVPR 2026
  - 结构化解释
  - 原型
  - GradCAM
  - 模型训练动态
  - XAI
---

# Why Does It Look There? Structured Explanations for Image Classification

**会议**: CVPR 2026  
**arXiv**: [2603.10234](https://arxiv.org/abs/2603.10234)  
**代码**: 无  
**领域**: others / 可解释AI  
**关键词**: 结构化解释, 原型, GradCAM, 模型训练动态, XAI

## 一句话总结

提出 I2X 框架，通过在训练检查点追踪从 GradCAM 提取的原型强度与模型置信度的协同演化，将非结构化的可解释性（显著性图）转化为结构化的可解释性，揭示模型"为什么关注那里"的推理结构，并利用这种理解指导微调提升性能。

## 研究背景与动机

**领域现状**：XAI 方法主要产生三类输出——显著性图（GradCAM）、概念向量（TCAV）和反事实样本。这些都是**非结构化的可解释性**，只告诉"模型看哪里"，不告诉"模型怎么组织这些信息来推理"。

**现有痛点**：
   - 现有方法提供的是碎片化的解释，无法回答"为什么模型看那里"和"模型如何在类间做决策"
   - 一些方法用 GPT/CLIP 等辅助模型描述行为，但解释不忠实于原始模型，可能产生幻觉
   - 训练过程中模型如何逐步建立决策策略的动态过程完全不透明

**核心矛盾**：可解释性（interpretability）≠ 可说明性（explainability），前者是描述现象，后者需要结构化归因

**切入角度**：模型的决策不是静态的——它在训练过程中逐步建立原型证据和置信度的关联，追踪这个过程就能构建结构化解释

**核心 idea**：在训练检查点间追踪原型强度变化与置信度变化的映射关系，将非结构化解释升级为结构化解释

## 方法详解

### 整体框架

I2X 包含五个步骤：
1. 在最终训练模型上用 K-Means 聚类所有隐藏特征向量→抽象原型
2. 在选定的训练检查点上用 GradCAM 生成显著性图
3. 将显著性图与原型对齐→计算原型强度
4. 用 HDBSCAN 聚类置信度变化模式→样本分组
5. 用岭回归建模原型强度变化→置信度变化的映射

### 关键设计

1. **抽象原型提取**:

    - 功能：从模型学到的特征中提取代表性模式
    - 核心思路：对最终模型的所有训练样本提取隐藏特征 $\mathbf{F} \in \mathbb{R}^{(N \cdot h \cdot w) \times d}$，先 PCA 降维再 K-Means聚类得到 $K$ 个聚类中心作为原型
    - 每个特征位置分配到最近原型：$A_i = (a_1, a_2, ..., a_{hw}), a_j \in \{1,...,K\}$
    - 设计动机：将高维特征空间压缩为有限个可解释的"模式"

2. **原型强度追踪**:

    - 功能：量化模型在每个训练检查点对每个原型的关注程度
    - 核心公式：将显著图和原型对齐，计算每个原型的平均强度：
    $P_k^t = \frac{\sum_{j=1}^{hw} \mathbf{1}[a_j = k] \cdot \text{Flatten}(I_j^t)}{\sum_{j=1}^{hw} \mathbf{1}[a_j = k]}$
    - 变化量 $\Delta \mathbf{P}^t = \mathbf{P}^{t+1} - \mathbf{P}^t$ 刻画原型证据的演化
    - 设计动机：显著性图告诉"看哪里"，原型告诉"看什么"，强度变化告诉"学习进程"

3. **置信度-原型映射**:

    - 功能：建立原型强度变化与模型置信度变化之间的定量关系
    - 核心思路：
        - 用 HDBSCAN 聚类所有样本的置信度变化 $\Delta \hat{Y}^t$，识别共同的学习模式
        - 用岭回归建模映射：$\beta^t = (\pi^{t\top}\pi^t + \lambda \mathbf{I})^{-1}\pi^{t\top}C^t \in \mathbb{R}^{K \times M}$
        - $\beta^t$ 量化了在训练步骤 $t$ 时，原型强度变化如何驱动置信度变化
    - 设计动机：聚合全部检查点的 $[\beta_t]$ 就能看到模型如何组织原型证据来支持/区分各类

4. **结构化解释的组装**:

    - 功能：从共享原型和特化原型两个角度分析每个类的决策过程
    - 共享原型：所有样本中都存在的原型，如数字7的横笔和斜笔
    - 特化原型：仅在子组中出现的原型，用于区分类内变体
    - 关键发现：模型不是同时区分所有类，而是**渐进式地**先解决原型差异明显的类，再处理模糊的类

### 损失函数 / 训练策略

I2X 是分析框架，不引入新的训练损失。但发现的"不确定原型"可以通过**扰动微调策略**提升性能：先在去除含不确定原型的样本上微调一轮，再在完整数据上微调一轮。

## 实验关键数据

### 主实验 — 微调提升

| 微调策略 | Accuracy(%) | 2↔7 混淆数 | 说明 |
|---------|-------------|-----------|------|
| 完整数据 → 完整数据 | 98.46±0.31 | 9.60±2.87 | 基线 |
| 筛选数据 → 筛选数据 | 98.31±0.63 | 9.00±4.90 | 混淆少但不稳定 |
| **筛选数据 → 完整数据** | **98.64±0.12** | **8.40±1.85** | 最优：混淆少且稳定 |

### CIFAR-10 / InceptionV3 泛化

| 模型/数据集 | 微调策略 | Accuracy(%) | 混淆数 |
|------------|---------|-------------|--------|
| ResNet-50 / CIFAR-10 | full→full | 81.43±2.79 | cat↔dog: 261.2 |
| ResNet-50 / CIFAR-10 | **curated→full** | **84.02±2.70** | **238.6** |
| InceptionV3 / MNIST | full→full | 99.13±0.29 | 4↔9: 12.6 |
| InceptionV3 / MNIST | **curated→full** | **99.11±0.27** | **10.8** |

### 关键发现
- 模型学习是渐进式的：先区分原型差异大的类（如 7 vs 6），再处理相似类（如 7 vs 1）
- **不确定原型**（如 P-26/P-17）在训练中在两个类之间摆动，是导致混淆的根因
- 训练数据顺序的随机性会改变原型选择策略——不同训练运行可能学到不同的推理策略
- 扰动微调策略（先去除含不确定原型的样本微调）能减少 MNIST 上约 5 个、CIFAR-10 上约 23 个混淆样本

## 亮点与洞察
- **将非结构化解释升级为结构化解释**：从"模型看了什么"到"模型为什么看那里以及如何做决策"，概念层次提升。
- **揭示模型学习的渐进式策略**：类似人类学习——先区分容易的，再处理困难的。
- **不确定原型的发现**：找到了跨类摆动的原型是混淆的直接原因，且可以据此设计改进策略，有实际价值。
- **训练随机性的结构化分析**：第一次用原型追踪解释不同训练运行间的策略差异。

## 局限与展望
- 仅在 MNIST 和 CIFAR-10 上验证，复杂数据集（ImageNet）上是否仍可解释有待验证
- K-Means 聚类数 $K$ 需要手动选择（MNIST 32, CIFAR-10 128），增大数据集时的选择策略不明确
- 依赖 GradCAM，对 Transformer 架构需要替换为 TokenTM 等方法
- 岭回归是线性模型，可能无法捕捉复杂的非线性原型-置信度关系
- 微调提升虽一致但幅度有限（MNIST <0.2%, CIFAR-10 ~2.6%）
- 分析成本较高：需要保存多个训练检查点并逐个分析

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从可解释性到可说明性的概念升级非常有洞察力
- 实验充分度: ⭐⭐⭐ 仅 MNIST 和 CIFAR-10，数据集规模和复杂度偏低
- 写作质量: ⭐⭐⭐⭐ 概念阐述清晰，图表信息密度高
- 价值: ⭐⭐⭐⭐ 提供了理解和改进模型的新视角，有实用潜力

<!-- RELATED:START -->

## 相关论文

- [DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_imageinimage_steg.md)
- [Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)

<!-- RELATED:END -->
