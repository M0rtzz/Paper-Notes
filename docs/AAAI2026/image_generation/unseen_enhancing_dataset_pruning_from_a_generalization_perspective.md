---
title: >-
  [论文解读] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective
description: >-
  [AAAI 2026][图像生成][数据集剪枝] 本文提出 UNSEEN，从泛化角度改进数据集剪枝方法——不仅考虑保留样本对训练损失的贡献，还考虑其对测试泛化的贡献，通过优化训练集与未见测试分布的对齐来选择更有利于泛化的核心子集。 领域现状：数据集剪枝（dataset pruning / coreset selection）…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "数据集剪枝"
  - "泛化性"
  - "训练效率"
  - "样本选择"
  - "核心集"
---

# UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective

**会议**: AAAI 2026  
**arXiv**: [2511.12988](https://arxiv.org/abs/2511.12988)  
**代码**: 无  
**领域**: 图像生成 / 数据集优化  
**关键词**: 数据集剪枝, 泛化性, 训练效率, 样本选择, 核心集

## 一句话总结

本文提出 UNSEEN，从泛化角度改进数据集剪枝方法——不仅考虑保留样本对训练损失的贡献，还考虑其对测试泛化的贡献，通过优化训练集与未见测试分布的对齐来选择更有利于泛化的核心子集。

## 研究背景与动机

**领域现状**：数据集剪枝（dataset pruning / coreset selection）旨在从大规模训练集中选择一个小的核心子集，使得在子集上训练可以接近在全集上训练的性能。这对降低训练成本至关重要。

**现有痛点**：（1）大多数数据集剪枝方法优化训练集上的损失，但这可能选出"容易拟合"而非"有利泛化"的样本；（2）忽略了未见数据的分布——选出的核心集可能在训练集上表现好但在测试集上泛化差；（3）冗余样本和边界样本的价值在不同场景下不同，需要更精细的衡量。

**核心矛盾**：训练效率 vs 泛化能力——为训练损失优化选出的子集不一定最有利于泛化。

**本文目标**：从泛化角度而非训练效率角度指导数据集剪枝。

**切入角度**：考虑核心集与未见（unseen）数据分布的对齐程度。

**核心 idea**：选择核心子集时不仅最小化训练误差，还最大化对未见数据分布的覆盖——确保选出的样本能帮助模型更好地泛化。

## 方法详解

### 关键设计

1. **泛化感知样本评分**：不仅用梯度/影响函数衡量样本对训练损失的贡献，还估计其对泛化误差的贡献。利用代理模型或验证集来估计泛化贡献。

2. **分布对齐约束**：选择核心集时加入约束——核心集的特征分布应与估计的全数据分布（包括未见数据的特征空间）尽量一致。

3. **自适应剪枝比例**：不同数据区域的剪枝比例不同——冗余区域可以大幅剪枝，稀疏但重要的区域应尽量保留。

### 损失函数 / 训练策略

核心集选择的优化目标 = 训练误差最小化 + 分布对齐正则项 + 多样性约束。

## 实验关键数据

### 主实验

| 数据集 | 剪枝比例 | UNSEEN泛化精度 | 传统剪枝精度 | 随机子集精度 |
|--------|---------|---------------|-------------|-------------|
| CIFAR-10 | 50% | 最佳 | 次优 | 最差 |
| ImageNet | 30% | 最佳 | 次优 | 最差 |

### 消融实验

| 配置 | 泛化精度 | 说明 |
|------|---------|------|
| UNSEEN (Full) | 最佳 | 泛化感知+分布对齐 |
| 仅训练损失优化 | 次优 | 传统剪枝 |
| 随机子集 | 最差 | 无选择策略 |
| 无分布对齐 | 下降 | 忽视分布覆盖 |

### 关键发现

- 泛化感知的样本选择比训练损失导向的选择在测试集上一致更好。
- 分布对齐约束的贡献在高剪枝比例下更加明显。
- 稀疏区域的样本对泛化至关重要——冗余区域可以大幅剪枝。

## 亮点与洞察

- **从泛化视角重新审视数据集剪枝**改变了问题的优化方向——不是选"好训练的"样本，而是选"帮助泛化的"样本。
- **分布对齐约束**使核心集更好地覆盖输入空间。

## 局限与展望

- 泛化贡献的估计本身需要额外计算。
- 对未见数据分布的估计依赖假设或代理。

## 相关工作与启发

- **vs 基于梯度的剪枝（如 GraNd）**: GraNd 优化训练损失，UNSEEN 优化泛化。
- **vs 核心集选择（如 Herding）**: 经典方法不考虑模型训练的特定需求。

## 评分

- 新颖性: ⭐⭐⭐⭐ 泛化视角的数据集剪枝新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集多比例验证
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰
- 价值: ⭐⭐⭐⭐ 对数据高效训练有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective](enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](../../NeurIPS2025/image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[ICML 2026\] Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](../../ICML2026/image_generation/enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)

</div>

<!-- RELATED:END -->
