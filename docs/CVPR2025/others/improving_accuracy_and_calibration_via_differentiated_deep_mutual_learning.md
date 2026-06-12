---
title: >-
  [论文解读] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning
description: >-
  [CVPR 2025][深度互学习] 提出 Diff-DML（Differentiated Deep Mutual Learning），通过差异化训练策略（DTS）和多样性保持学习目标（DPLO）两个核心设计，在保持集成模型预测多样性的同时，同时提升准确率和不确定性校准质量。
tags:
  - "CVPR 2025"
  - "深度互学习"
  - "不确定性校准"
  - "集成方法"
  - "预测多样性"
  - "过度自信"
---

# Improving Accuracy and Calibration via Differentiated Deep Mutual Learning

**会议**: CVPR 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 模型校准与集成学习  
**关键词**: 深度互学习, 不确定性校准, 集成方法, 预测多样性, 过度自信

## 一句话总结

提出 Diff-DML（Differentiated Deep Mutual Learning），通过差异化训练策略（DTS）和多样性保持学习目标（DPLO）两个核心设计，在保持集成模型预测多样性的同时，同时提升准确率和不确定性校准质量。

## 研究背景与动机

**领域现状**：深度神经网络在各类任务中取得了优异的预测准确率，但在安全关键应用（如自动驾驶、医疗诊断）中，仅有高准确率是不够的，还需要可靠的不确定性估计。

**现有痛点**：
- 现代 DNN 使用交叉熵损失训练时容易出现**过度自信**（overconfidence）现象，尤其在模糊样本上
- 许多校准技术（如温度缩放、标签平滑等）虽然能改善校准，但往往以**牺牲准确率**或**增加计算开销**为代价
- 传统的 Deep Mutual Learning (DML) 虽然通过多模型互相学习能提升性能，但模型间会逐渐趋同，**丧失预测多样性**，反而不利于校准

**核心矛盾**：集成方法的校准增益来源于成员模型的预测多样性，但互学习过程会使模型趋同，导致多样性丧失，这构成了一个根本性矛盾。

**本文目标** 如何在互学习框架中保持成员模型的预测多样性，从而同时获得准确率和校准的双重提升。

**切入角度**：从互学习的多样性丧失问题入手，通过差异化的训练策略和学习目标来维持模型间的差异。

**核心 idea**：通过差异化的数据增强和差异化的KL散度学习目标，让互学习的各个模型保持足够的预测多样性，从而实现集成的校准增益。

## 方法详解

### 整体框架

Diff-DML 基于 Deep Mutual Learning（DML）框架，训练多个网络互相学习，但引入了两个关键创新来维持预测多样性：差异化训练策略（DTS）和多样性保持学习目标（DPLO）。

### 关键设计

1. **差异化训练策略 (Differentiated Training Strategy, DTS)**:
    - 功能：通过给不同模型施加不同的数据增强策略，从源头上保证模型接收到差异化的训练信号
    - 核心思路：每个成员模型使用不同的数据增强组合（如不同的裁剪策略、颜色变换等），使得即使在互学习过程中，各模型也会因为看到不同视角的数据而保持差异
    - 设计动机：传统 DML 中所有模型看到相同的输入，导致互学习后快速趋同；差异化输入是维持多样性最直接的手段

2. **多样性保持学习目标 (Diversity-Preserving Learning Objective, DPLO)**:
    - 功能：修改互学习的 KL 散度目标，在鼓励模型互相学习的同时，惩罚模型预测分布过于相似
    - 核心思路：在标准的 KL 散度互学习损失基础上，引入多样性项，当模型预测过于相似时给予惩罚，从而在优化目标层面维持多样性
    - 设计动机：仅靠不同的数据增强可能不足以维持长期的多样性，需要在损失函数层面提供明确的多样性约束

3. **理论分析支撑**:
    - 功能：从理论上证明 Diff-DML 的多样化学习框架能够利用集成优势，同时避免传统 DML 中的预测多样性丧失
    - 核心思路：通过分析集成模型的方差分解，证明预测多样性对校准质量的关键作用
    - 设计动机：为方法的有效性提供理论保证

### 损失函数 / 训练策略

总体损失包含三部分：
- **分类损失**：标准交叉熵损失，确保每个模型的分类准确率
- **互学习损失**：修改后的 KL 散度，鼓励模型间相互学习软标签知识
- **多样性正则化**：惩罚模型预测过于相似，维持集成多样性

训练过程中，多个模型同步训练，每个模型使用差异化的数据增强，并通过 DPLO 目标相互学习。

## 实验关键数据

### 主实验

在 CIFAR-100 数据集上使用 ResNet34 模型的结果：

| 指标 | Diff-DML vs MDCA (SOTA) | 改进幅度 |
|------|------------------------|---------|
| 准确率 | 绝对提升 | +1.3% / +3.1% |
| ECE | 相对降低 | -49.6% / -43.8% |
| Classwise-ECE | 相对降低 | -7.7% / -13.0% |

在多个基准数据集上进行了广泛评估，均验证了方法的有效性。

### 消融实验

- DTS 和 DPLO 各自单独使用都能带来提升，但联合使用效果最佳
- 差异化数据增强的效果随增强策略差异程度增大而提升
- 多样性正则化的权重需要仔细调整

### 关键发现

- 传统 DML 在训练后期模型趋同速度加快，多样性急剧下降
- Diff-DML 能在整个训练过程中保持稳定的预测多样性
- 预测多样性与校准质量之间存在强正相关关系
- 该方法在不同架构（ResNet、WideResNet等）上均表现一致

## 亮点与洞察

1. **问题洞察深刻**：准确指出了传统互学习中多样性丧失这一被忽视的问题，并给出了理论和实验上的充分论证
2. **方案简洁有效**：DTS 和 DPLO 两个设计都很简洁，不需要额外的复杂模块，实现成本低
3. **理论与实验统一**：提供了理论分析证明多样性对校准的重要性，并通过实验验证了理论预测
4. **同时提升两个指标**：在不增加推理开销的前提下，同时改善准确率和校准质量

## 局限与展望

1. **集成推理开销**：推理时仍需运行多个模型，计算开销随成员数量线性增长
2. **数据增强策略选择**：差异化增强策略的设计目前缺乏自动化方法
3. **大规模验证**：主要在 CIFAR-100 等中等规模数据集上验证，大规模数据集表现待确认
4. **与后处理校准的结合**：可以探索与温度缩放等后处理方法的组合效果

## 相关工作与启发

- **Deep Mutual Learning (DML)**：本工作的基础框架
- **MDCA**：之前的 SOTA 校准方法
- **集成方法的多样性理论**：双重分解定理表明集成效果取决于成员模型的多样性
- **对后续研究的启发**：维持互学习中的多样性思路可推广到知识蒸馏、联邦学习等场景

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)
- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](../../NeurIPS2025/others/mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)
- [\[CVPR 2025\] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)

</div>

<!-- RELATED:END -->
