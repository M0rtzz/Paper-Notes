---
title: >-
  [论文解读] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning
description: >-
  [NeurIPS 2025][多模态VLM][modality imbalance] 首次提出将跨模态不对齐样本作为有监督训练信号（而非噪声/干扰）来缓解多模态学习中的模态不平衡问题，设计 MIDAS 数据增强框架：通过置信度标注不对齐样本 + 弱模态加权 + 难样本加权三重机制，在四个多模态分类基准上显著超越现有方法。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "modality imbalance"
  - "数据增强"
  - "misaligned samples"
  - "weak-modality weighting"
  - "hard-sample weighting"
---

# MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.25831](https://arxiv.org/abs/2509.25831)  
**代码**: 待确认  
**领域**: 多模态VLM / 多模态学习 / 数据增强  
**关键词**: modality imbalance, data augmentation, misaligned samples, weak-modality weighting, hard-sample weighting

## 一句话总结
首次提出将跨模态不对齐样本作为有监督训练信号（而非噪声/干扰）来缓解多模态学习中的模态不平衡问题，设计 MIDAS 数据增强框架：通过置信度标注不对齐样本 + 弱模态加权 + 难样本加权三重机制，在四个多模态分类基准上显著超越现有方法。

## 研究背景与动机

**领域现状**：多模态学习广泛应用于视觉-语言建模、医疗诊断、自动驾驶等领域，但**模态不平衡**问题始终存在——模型倾向于依赖信息量更大的主导模态，忽视弱模态，甚至可能退化到比单模态更差。

**现有方法局限**：
   - **优化策略类**（OGM、AGM）：通过调节梯度/权重来抑制主导模态，但引入额外计算开销
   - **数据/特征类**（AMCo、SMV）：通过掩码主导模态特征或重采样来平衡，但信息利用不充分
   - **对比/无监督类**（LFM、MCR）：将不对齐数据用于负样本或互信息估计，但缺乏直接监督信号

**核心洞察**：不对齐样本（如猫图+狗文本）蕴含丰富的模态特异性信息，能暴露模型对主导模态的过度依赖——在不对齐输入上，标准模型仅有 6.3% 准确率（对齐为 65.5%），且总是高置信度预测主导模态对应类别。

**本文思路**：将不对齐样本从噪声转化为有监督训练信号，迫使模型从矛盾信号中学习平衡利用各模态。

## 方法详解

### 整体框架
MIDAS 在对齐样本和不对齐样本上同时训练，包含三个核心组件：置信度标注、弱模态加权、难样本加权。

### 1. 生成不对齐样本
从同一 mini-batch 中随机选择标签不同的两个样本 $(x_i, y_i)$ 和 $(x_j, y_j)$（$y_i \neq y_j$），交换一个模态：
$$\tilde{x}_i = (\tilde{x}_i^1, \tilde{x}_i^2) = (x_i^1, x_j^2)$$
例如：猫图像 + 狗文本。对称地也生成 $(x_j^1, x_i^2)$。

### 2. 单模态置信度标注
不对齐样本的标签不能简单取任一类的硬标签。用预训练的单模态分类器评估每个模态对其原始类别的置信度：

归一化置信度：
$$\tilde{c}_i^1 = \frac{(p_i^1)_{y_i}}{(p_i^1)_{y_i} + (p_j^2)_{y_j}}, \quad \tilde{c}_i^2 = \frac{(p_j^2)_{y_j}}{(p_i^1)_{y_i} + (p_j^2)_{y_j}}$$

软标签为加权平均：
$$\tilde{y}_i = \tilde{c}_i^1 \mathbf{y}_i + \tilde{c}_i^2 \mathbf{y}_j$$

例：若视觉置信 0.9、文本置信 0.3，则 $\tilde{c}^1=0.75$, $\tilde{c}^2=0.25$。

### 3. 弱模态加权（Weak-Modality Weighting）
置信度标注仍然偏向强模态。解决方案：动态增大最不自信模态的损失权重。

识别批次中平均置信最低的模态：
$$\hat{m} = \arg\min_{m \in \{1,2\}} \mathbb{E}_{(\tilde{x}_i, \tilde{y}_i) \sim \tilde{B}}[\tilde{c}_i^m]$$

比较目标标签中弱模态的贡献与多模态模型的实际预测：
$$\Delta_{\alpha} = \text{sign}\left(\mathbb{E}[\tilde{c}_i^{\hat{m}}] - \mathbb{E}[(\tilde{c}_i)_{\tilde{y}_i^{\hat{m}}}]\right)$$

更新规则：
$$\alpha_{\hat{m}}^{(t+1)} = \max(1, \alpha_{\hat{m}}^{(t)} + \eta \cdot \Delta_{\alpha})$$

若模型低估了弱模态贡献，$\alpha$ 增大，放大其损失权重。

### 4. 难样本加权（Hard-Sample Weighting）
不是所有不对齐样本同等有价值——交换进来的特征与原特征越相似，语义冲突越微妙，越有训练价值。

$$\tilde{s}_i = \frac{(f_i^2)^\top f_j^2}{\|f_i^2\|_2 \|f_j^2\|_2}$$

### 5. 总损失
$$\mathcal{L}_{mis}(\tilde{x}_i, \tilde{y}_i; \alpha^{(t)}, \tilde{s}_i) = \left(1 + \frac{\tilde{s}_i + 1}{2}\right) \cdot \left[-\sum_{c=1}^{C}(\alpha_1^{(t)}\tilde{c}_i^1(\mathbf{y}_i)_c + \alpha_2^{(t)}\tilde{c}_i^2(\mathbf{y}_j)_c)\log((\tilde{p}_i)_c)\right]$$

$$\mathcal{L}_{total} = \frac{1}{|B|}\sum_{(x_i, y_i) \in B}\left[\mathcal{L}_{align}(x_i, y_i) + \mathcal{L}_{uni}(x_i, y_i) + \lambda \mathcal{L}_{mis}(\tilde{x}_i, \tilde{y}_i; \alpha^{(t)}, \tilde{s}_i)\right]$$

训练前有 warm-up 阶段预训练编码器和单模态分类器。计算复杂度保持 $O(N)$。

## 实验关键数据

### 与基线对比（4个数据集）

| 方法 | K-S Acc | K-S F1 | CREMA-D Acc | CREMA-D F1 | UCF-101 Acc | Food-101 Acc |
|------|---------|--------|-------------|------------|-------------|--------------|
| Joint | 63.92 | 55.54 | 60.28 | 58.60 | 90.07 | 91.35 |
| SMV | 65.76 | 57.59 | 67.94 | 66.83 | **95.24** | 91.64 |
| OPM | 67.35 | 59.29 | 63.97 | 62.71 | 91.73 | 92.40 |
| AMCo | 67.04 | 58.41 | 69.91 | 68.85 | 93.77 | 92.00 |
| MCR | 71.75 | 64.23 | 70.91 | 70.19 | 91.84 | 90.58 |
| **MIDAS** | **74.88** | **67.18** | **74.99** | **73.82** | 95.20 | **93.46** |

- Kinetics-Sounds: +3.13%p over best baseline（MCR）
- CREMA-D: +4.08%p over best baseline

### 消融实验

| W | WM | HS | K-S Acc | CREMA-D Acc | UCF-101 Acc | Food-101 Acc |
|---|----|----|---------|-------------|-------------|--------------|
| ✗ | ✗ | ✗ | 71.70 | 72.32 | 94.16 | 93.39 |
| ✓ | ✓ | ✓ | **74.88** | **74.99** | **95.20** | 93.46 |

三个组件单独使用提升有限，组合后产生明显协同效应。

### 与数据增强方法对比

| 方法 | CREMA-D Acc | Food-101 Acc |
|------|-------------|--------------|
| Mixup | 61.84 | 91.36 |
| PowMix | 63.66 | 89.59 |
| LeMDA | 58.13 | 91.19 |
| **MIDAS** | **74.99** | **93.11** |

### 三模态实验（CMU-MOSI）
MIDAS: 74.00% Acc / 73.64 F1 vs Joint: 71.13% Acc / 70.86 F1，证明可扩展到三模态。

## 亮点
- ⭐⭐⭐⭐ **视角新颖**：将不对齐样本从噪声翻转为有价值的监督信号，概念转换令人印象深刻
- ⭐⭐⭐⭐ **机制完整**：置信度标注+弱模态加权+难样本加权形成互补闭环
- ⭐⭐⭐⭐ **理论清晰**：每个组件的设计动机、数学推导和消融验证都很充分
- ⭐⭐⭐ **通用性好**：模态无关设计，已验证音频-视频、图像-文本、RGB-光流、三模态场景

## 局限与展望
1. 仅在分类任务上验证，生成、检索等任务的适用性需进一步探索
2. 当两种模态信息量差异极端时，不对齐样本的标注质量可能下降
3. warm-up 阶段需要单独训练单模态分类器，增加了训练流程复杂度
4. 随机配对策略简单高效但可能非最优，基于语义相似度的智能配对策略值得研究

## 总评
⭐⭐⭐⭐ 极具启发性的工作，"不对齐即资源"的核心 idea 优雅地串联起三个技术组件。实验覆盖全面，消融分析透彻。对多模态学习中模态不平衡问题提供了全新的数据驱动方案。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[NeurIPS 2025\] Learning Shared Representations from Unpaired Data](learning_shared_representations_from_unpaired_data.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)

</div>

<!-- RELATED:END -->
