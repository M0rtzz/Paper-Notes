---
title: >-
  [论文解读] Why Does It Look There? Structured Explanations for Image Classification
description: >-
  [CVPR 2025][结构化解释] 本文提出 I2X 框架，通过在训练检查点上追踪从 GradCAM 显著性图中提取的抽象原型（prototype）的强度变化与模型置信度的对应关系，将非结构化的可解释性转化为结构化的可解释性，并利用识别出的"不确定原型"来指导微调、减少类间混淆、提升分类精度。
tags:
  - CVPR 2025
  - 结构化解释
  - 原型学习
  - GradCAM
  - 训练轨迹分析
  - 可解释人工智能
---

# Why Does It Look There? Structured Explanations for Image Classification

**会议**: CVPR 2025  
**arXiv**: [2603.10234](https://arxiv.org/abs/2603.10234)  
**代码**: 无  
**领域**: 可解释性 / XAI  
**关键词**: 结构化解释, 原型学习, GradCAM, 训练轨迹分析, 可解释人工智能

## 一句话总结

本文提出 I2X 框架，通过在训练检查点上追踪从 GradCAM 显著性图中提取的抽象原型（prototype）的强度变化与模型置信度的对应关系，将非结构化的可解释性转化为结构化的可解释性，并利用识别出的"不确定原型"来指导微调、减少类间混淆、提升分类精度。

## 研究背景与动机

**领域现状**：当前 XAI 方法主要提供非结构化的解释形式——如显著性图（saliency map）、概念向量（concept vector）或反事实样本，这些方法告诉我们模型"看了哪里"，但无法回答"为什么看那里"。

**现有痛点**：一些尝试提供结构化推理的工作（如 HybridCBM、GPT 辅助解释）依赖于辅助模型来描述行为，导致解释并非来自原始模型本身，存在不忠实甚至产生幻觉的风险。DiffCAM 等方法虽然对比了激活模式，但仍未揭示模型在推理和训练过程中如何组织和使用这些差异。

**核心矛盾**：现有方法提供的是"可解读性"（interpretability）而非"可解释性"（explainability）——它们描述了模型的行为表象，但缺少对模型内部决策结构的因果责任归因。

**本文目标** (1) 如何从非结构化解释中构建结构化解释？(2) 如何追踪模型在训练过程中的决策演化？(3) 如何利用这些结构化解释来指导模型优化？

**切入角度**：作者观察到模型在训练过程中，其注意力区域（通过显著性图量化）和预测置信度的变化是同步的、有规律的。通过在多个训练检查点上提取抽象原型并关联置信度变化，可以构建模型的结构化学习轨迹。

**核心 idea**：通过在训练检查点上追踪原型强度与置信度的共变关系，将 GradCAM 等非结构化解释方法的输出转化为结构化的决策解释。

## 方法详解

### 整体框架

I2X 的输入是一个训练好的分类模型和其训练过程中保存的多个检查点。输出是一个结构化的解释图谱，描述模型如何通过抽象原型进行类内和类间的决策。整个 pipeline 分为两大步骤：(1) 从最终模型的特征中聚类提取抽象原型集合；(2) 在每个训练检查点上计算原型强度变化与置信度变化的映射关系，最终汇聚成完整的结构化解释。

### 关键设计

1. **抽象原型提取（Abstract Prototypes）**:

    - 功能：从最终模型对所有训练样本的隐层特征中聚类出 K 个代表性模式
    - 核心思路：对特征提取器 $f$ 输出的所有 $N \times h \times w$ 个特征向量先 PCA 降维再 K-Means 聚类，得到 K 个聚类中心作为抽象原型。每个空间位置的特征被分配到某个原型，从而将图像区域与原型关联
    - 设计动机：与 ProtoPNet 等"原型学习"方法不同，这里的原型完全是后验提取的，不改变模型训练过程，保证了对原始模型的忠实性

2. **原型强度追踪（Prototype Intensity Tracking）**:

    - 功能：在每个训练检查点上，将 GradCAM 显著性图与原型空间对齐，量化每个原型的激活强度
    - 核心思路：对于样本 $x$ 在检查点 $t$，计算原型强度 $P_k^t = \frac{\sum_{j} \mathbf{1}[a_j=k] \cdot I_j^t}{\sum_{j} \mathbf{1}[a_j=k]}$，即将属于同一原型的空间位置上的显著性值取均值。原型强度的变化 $\Delta \mathbf{P}^t$ 刻画了模型在训练中如何调整"看"的策略
    - 设计动机：GradCAM 单独只给热力图，但结合原型分配后可以量化"注意力分配给了哪些结构性模式"，实现从像素级到语义级的跃迁

3. **原型-置信度映射（Prototype-Confidence Mapping）**:

    - 功能：建立原型强度变化与模型置信度变化之间的定量关系
    - 核心思路：先用 HDBSCAN 对所有样本的置信度变化 $\Delta \hat{Y}^t$ 进行聚类，找出具有相似置信度变化模式的样本群组。然后在每个群组内，用 ridge regression 拟合原型强度变化 $\pi^t$ 与置信度变化 $C^t$ 的线性关系：$\beta^t = (\pi^{t\top}\pi^t + \lambda I)^{-1}\pi^{t\top}C^t$，系数矩阵 $\beta^t$ 量化了每个原型对每个类别置信度变化的贡献
    - 设计动机：直接分析原型与置信度的关系维度太高，通过先聚类再回归的两步策略，既降低了复杂度，又能发现群体性的模式

### 损失函数 / 训练策略

I2X 本身不修改模型训练过程。在微调阶段，作者构建"策展数据集"（curated dataset），排除包含不确定原型的样本，先用策展数据集微调一个 epoch，再用完整数据集微调一个 epoch，实现扰动引导优化。

## 实验关键数据

### 主实验

| 数据集 / 模型 | 微调策略 | 准确率 (%) | 2↔7混淆数 | 说明 |
|--------------|---------|-----------|----------|------|
| MNIST / ResNet-50 | full→full | 98.46±0.31 | 9.60±2.87 | 传统两轮微调 |
| MNIST / ResNet-50 | curated→full | 98.64±0.12 | 8.40±1.85 | I2X指导微调，最优 |
| MNIST / ResNet-50 | full (1 epoch) | 98.52±0.34 | 14.80±6.31 | 基线单轮微调 |
| MNIST / ResNet-50 | curated (1 epoch) | 98.67±0.18 | 9.80±2.93 | 策展集微调 |

| 数据集 / 模型 | 微调策略 | 准确率 (%) | 混淆对 | 混淆数 |
|--------------|---------|-----------|--------|-------|
| CIFAR10 / ResNet-50 | full→full | 81.43±2.79 | cat↔dog | 261.20±30.77 |
| CIFAR10 / ResNet-50 | curated→full | 84.02±2.70 | cat↔dog | 238.60±21.90 |
| MNIST / InceptionV3 | full→full | 99.13±0.29 | 4↔9 | 12.60±3.07 |
| MNIST / InceptionV3 | curated→full | 99.11±0.27 | 4↔9 | 10.80±2.71 |

### 消融实验

| 配置 | 混淆数 (2↔7) | 准确率 (%) | 说明 |
|------|-------------|-----------|------|
| full 1 epoch | 14.80±6.31 | 98.52±0.34 | 完整数据微调 |
| curated 1 epoch | 9.80±2.93 | 98.67±0.18 | 去除不确定原型样本 |
| curated→curated | 9.00±4.90 | 98.31±0.63 | 两轮策展集，方差变大 |
| curated→full | 8.40±1.85 | 98.64±0.12 | 先扰动后恢复，最稳定 |

### 关键发现

- curated→full 策略在 MNIST 上减少约 5 个混淆样本，CIFAR-10 上减少约 23 个，同时提升整体准确率
- 纯策展数据集两轮微调虽然混淆数最低（9.00），但方差接近翻倍（4.90 vs 2.93），说明模型开始探索新策略但缺乏支撑
- 训练数据顺序的随机性会导致模型学到完全不同的原型选择序列和推理策略，量化可观测于混淆矩阵差异

## 亮点与洞察

- **从可解读到可解释的理论跃迁**：I2X 明确区分了 interpretability 和 explainability，并提出了一条从前者到后者的系统化路径；这个思路比方法本身更有长远价值
- **训练轨迹分析**：通过检查点追踪来揭示模型"先学什么、后学什么"的渐进策略——发现模型先分离容易区分的类（如 7 vs 6），后处理困难类（如 7 vs 1）
- **扰动引导微调**：识别不确定原型后通过数据集策展来绕过有害样本，思路简单但效果显著，可以迁移到任何使用 post-hoc XAI 方法的场景

## 局限与展望

- 仅在 MNIST 和 CIFAR-10 上验证，数据集复杂度有限；在 ImageNet 等大规模数据集上原型数量和检查点分析的计算开销可能是瓶颈
- 依赖 GradCAM 作为底层解释方法，对 Transformer 架构需要切换到 AttnLRP 等方法，框架的通用性有待验证
- 原型数量 K 的选择（32 for MNIST, 128 for CIFAR-10）目前是手动设定的，缺乏自适应选择机制
- ridge regression 的线性假设可能在复杂场景下不够，原型-置信度的关系可能是非线性的

## 相关工作与启发

- **vs ProtoPNet**: ProtoPNet 通过前向设计让模型学习原型，I2X 则完全后验提取原型，不修改训练过程，保证忠实性但牺牲了原型的可控性
- **vs DiffCAM**: DiffCAM 对比不同样本/组的激活模式来提升忠实度，但仍然是非结构化的解释；I2X 在此基础上进一步关联了训练动态
- **vs LLM-based XAI (HybridCBM等)**: 使用 GPT/CLIP 辅助解释虽然输出更易读，但解释来源于辅助模型而非被解释模型本身，I2X 避免了这个忠实性问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 从训练检查点角度构建结构化解释的思路新颖，但核心技术（PCA+KMeans+ridge regression）较为基础
- 实验充分度: ⭐⭐⭐ 仅在 MNIST 和 CIFAR-10 上验证，缺乏大规模数据集实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，从 interpretability 到 explainability 的逻辑链条完整
- 价值: ⭐⭐⭐⭐ 框架思路有启发意义，但实验规模限制了实际影响力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](../../ACL2025/interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[CVPR 2025\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](learning_visual_composition_through_improved_semantic_guidance.md)

<!-- RELATED:END -->
