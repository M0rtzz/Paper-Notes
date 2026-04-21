---
title: >-
  [论文解读] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets
description: >-
  [NeurIPS 2025][偏差放大] 本文提出 Directional Predictability Amplification (DPA)，一种基于可预测性的偏差放大度量指标，是唯一同时满足方向性、适用于平衡/非平衡数据集、能正确识别正负偏差放大的一站式指标，通过测量模型偏差与数据集偏差的相对变化来量化偏差放大程度。
tags:
  - NeurIPS 2025
  - 偏差放大
  - 公平性度量
  - 可预测性
  - 方向性偏差
  - 分类公平
---

# DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets

**会议**: NeurIPS 2025  
**arXiv**: [2412.11060](https://arxiv.org/abs/2412.11060)  
**代码**: 有  
**领域**: others  
**关键词**: 偏差放大, 公平性度量, 可预测性, 方向性偏差, 分类公平

## 一句话总结
本文提出 Directional Predictability Amplification (DPA)，一种基于可预测性的偏差放大度量指标，是唯一同时满足方向性、适用于平衡/非平衡数据集、能正确识别正负偏差放大的一站式指标，通过测量模型偏差与数据集偏差的相对变化来量化偏差放大程度。

## 研究背景与动机

**领域现状**：ML 模型不仅会学习训练数据中的偏差，还会放大它们。例如 ImSitu 数据集中 67% 的 cooking 图片是女性，模型预测时比例可能升到 90%。现有度量分为共现率方法（BA, Multi, BA_MALS）和可预测性方法（LA）。

**现有痛点**：共现率方法（BA, BA_MALS）在平衡数据集上失效；Multi 虽适用于平衡数据集但无法检测负偏差放大；LA 可处理平衡数据集但不具方向性。没有任何一个指标同时具备三个关键性质。

**核心矛盾**：共现率方法天然支持方向性但受限于标注信息，可预测性方法可捕捉隐藏偏差但缺乏方向性——如何结合两者优势？

**本文目标**：设计一个一站式指标，同时满足：(1) 方向性；(2) 适用于平衡和非平衡数据集；(3) 正确识别正负偏差放大。

**切入角度**：将 LA 的可预测性框架扩展为双向的，用相对变化（而非绝对变化）来量化放大程度。

**核心 idea**：用方向性可预测性的归一化相对变化衡量偏差放大，一个指标解决三个问题。

## 方法详解

### 整体框架
给定分类数据集（图片 I、任务标签 T、受保护属性 A）和模型 M 的预测，DPA 分别测量 A->T 和 T->A 两个方向的偏差放大。每个方向通过训练攻击者函数来量化可预测性。

### 关键设计

1. **方向性可预测性度量**:

    - 功能：分别量化 A->T 和 T->A 方向的数据偏差和模型偏差
    - 核心定义（A->T 方向）：
        - 数据偏差：$\Psi_A^D = Q(f_A^T(A), T)$ — 攻击者从真实属性 A 预测真实任务 T 的质量
        - 模型偏差：$\Psi_A^M = Q(f_A^{\hat{T}}(A), \hat{T})$ — 攻击者从真实属性 A 预测模型任务预测的质量
    - T->A 方向类似，交换 A 和 T 的角色

2. **归一化相对变化（DPA 公式）**:

    - 功能：将偏差放大量化为有界的 [-1, 1] 区间值
    - 核心公式：$DPA_{A \to T} = \frac{\Psi_A^M - \Psi_A^D}{\Psi_A^M + \Psi_A^D}$，$DPA_{T \to A} = \frac{\Psi_T^M - \Psi_T^D}{\Psi_T^M + \Psi_T^D}$
    - 设计动机：归一化确保 DPA 在 [-1, 1] 范围内；相对变化比绝对变化更合理——无偏数据集上 0.05 的增加远比高偏差数据集上更严重

3. **准确率对齐处理**:

    - 功能：消除模型预测误差对偏差度量的干扰
    - 核心思路：如果模型任务预测准确率为 70%，则在真实标签 T 上随机翻转 30% 使其准确率也对齐
    - 设计动机：防止将模型的预测误差误判为偏差放大

### 损失函数 / 训练策略
攻击者函数可以是任意 ML 模型（SVM、决策树、MLP），用质量函数 Q（如准确率、F1 score）评估。论文中使用两层隐藏层的 MLP 作为攻击者。

## 实验关键数据

### 主实验：COCO 数据集 T->A 偏差放大（ViT_b_16，逐步遮挡人物）

| 指标 | 原始 | 部分遮挡 | 完全遮挡分割 | 完全遮挡边框 |
|------|------|---------|-------------|-------------|
| Attribution Score | 0.3827 | 0.4327 | 0.4461 | 0.5247 |
| **DPA (ours)** | **-0.0152** | **0.1365** | **0.6015** | **0.8085** |
| BA_-> | -0.0227 | 0.0097 | 0.0188 | 0.0601 |
| Multi_-> | 0.1506 | 0.1179 | 0.3606 | 0.5607 |
| LA | 0.0368 | 0.0715 | 0.0942 | 0.0265 |
| BA_MALS | 0.0001 | 0.0004 | -0.0016 | 0.015 |

### 指标能力对比

| 方法 | 平衡数据集 | 方向性 | 负偏差放大 |
|------|-----------|--------|----------|
| BA_MALS | 否 | 否 | 是 |
| BA_-> | 否 | 是 | 是 |
| Multi_-> | 是 | 是 | 否 |
| LA | 是 | 否 | 是 |
| **DPA (ours)** | **是** | **是** | **是** |

### ImSitu A->T 模型排名一致性

| 模型 | Sen 真实排名 | DPA 排名 | LA 排名 | BA_-> 排名 |
|------|------------|----------|---------|----------|
| MaxViT | 1 | 1 | 4 | 3 |
| ViT_b_32 | 2 | 2 | 6 | 5 |
| VGG16 | 9 | 9 | 7 | 9 |

### 关键发现
- COCO 实验中随着人物被逐步遮挡，模型越来越依赖背景物体预测性别，只有 DPA 和 BA_-> 正确反映趋势
- 平衡 COCO 数据集上，BA_-> 和 BA_MALS 始终报告零偏差放大（错误），而 DPA 正确捕捉到模型利用未标注物体的偏差
- ImSitu 数据集上，DPA 的模型排名与真实偏差排名完全一致，其他指标均有多处排名错误
- LA 在遮挡加重时反而下降（0.0942->0.0265），因为绝对变化度量无法处理这种情况

## 亮点与洞察
- **一站式解决三个长期问题**：方向性 + 平衡数据集 + 负偏差放大，之前任何单一指标都做不到
- **相对变化 vs 绝对变化的洞察**：在已有高偏差的数据集上增加少量偏差 vs 在无偏差数据集上引入偏差，严重程度完全不同
- **对攻击者函数的鲁棒性**：由于度量相对变化，系统性偏差被抵消，LA 的绝对变化则对攻击者超参数高度敏感

## 局限与展望
- 质量函数 Q 的选择需注意方向性（0=最差），交叉熵等损失需取倒数
- 准确率对齐通过随机翻转标签实现，引入随机性需要置信区间
- 仅针对分类任务，回归和生成任务的偏差放大需要额外扩展
- 攻击者函数的训练仍需计算资源，尤其在大规模多属性场景下

## 相关工作与启发
- **vs BA_-> (Wang & Russakovsky 2021)**: 首个方向性指标但仅适用非平衡数据集，DPA 扩展到平衡场景
- **vs LA (Wang et al. 2019)**: 首个可预测性方法但无方向性且度量绝对变化，DPA 加入方向性和归一化
- **vs Multi_-> (Zhao et al. 2023)**: 支持多属性和平衡数据集但无法检测负偏差放大，DPA 通过符号变化自然支持

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义清晰，解决方案简洁而完整，归一化相对变化设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ COMPAS+COCO+ImSitu 三个数据集，9个模型，设计了控制实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，指标设计逻辑严密，实验设计巧妙
- 价值: ⭐⭐⭐⭐ 对 ML 公平性实践有直接帮助，消除了多指标交叉验证的困扰

<!-- RELATED:START -->

## 相关论文

- [A Measure of the System Dependence of Automated Metrics](../../ACL2025/others/a_measure_of_the_system_dependence_of_automated_metrics.md)
- [One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves](../../ICML2025/others/improved_learning_via_k-dtw_a_novel_dissimilarity_measure_for_curves.md)

<!-- RELATED:END -->
