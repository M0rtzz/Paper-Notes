---
title: >-
  [论文解读] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models
description: >-
  [NeurIPS 2025][foundation model] 提出 ComBo，一种基于 probing 的轻量级 adapter，通过仿射投影压缩多个冻结基础模型多层激活，再用小型 transformer 融合，无需反向传播即可高效整合多模型互补表征，在 VTAB-1k 上超越先前 probing 方法并匹配蒸馏方法。
tags:
  - NeurIPS 2025
  - foundation model
  - probing
  - multi-backbone
  - feature combination
  - model selection
---

# Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2512.01405](https://arxiv.org/abs/2512.01405)  
**代码**: 有 (bramtoula.github.io/combo)  
**领域**: 模型压缩 / 多模型融合  
**关键词**: foundation model, probing, multi-backbone, feature combination, model selection

## 一句话总结

提出 ComBo，一种基于 probing 的轻量级 adapter，通过仿射投影压缩多个冻结基础模型多层激活，再用小型 transformer 融合，无需反向传播即可高效整合多模型互补表征，在 VTAB-1k 上超越先前 probing 方法并匹配蒸馏方法。

## 研究背景与动机

不同基础模型（CLIP、DINOv2、MAE、SAM 等）因训练目标和数据不同，学到的表征各有优劣。对于特定下游任务，最优模型可能不同，甚至最优层也不同（中间层有时优于最后一层）。现有方法存在三个问题：

**PEFT 方法**（如 LoRA、Adapter）仅适配单模型，且需要反向传播通过 backbone，多模型组合计算代价高

**蒸馏方法**（如 RADIO）需将多个 FM 蒸馏到学生模型，代价高、易出现 "mode switch" 问题

**现有 probing 方法**（Head2Toe、SMP）需要数据集特定的超参调优、对 feature map 做平均池化丢失空间信息、扩展性差

核心问题：如何在不需要反向传播通过任何大模型的前提下，高效组合多个 FM 的互补表征？

## 方法详解

### 整体框架

ComBo 的设计分为三个阶段：

1. **Feature Map 提取**：从 K 个冻结模型的每一层提取特征图 $\mathbf{F}_{k,l} \in \mathbb{R}^{T_k \times D_k}$，通过双线性插值统一 token 数量 T，并做均值-标准差归一化
2. **层嵌入压缩**：在每个空间位置 i，将所有模型所有层的特征拼接为 $\mathbf{S}_i \in \mathbb{R}^D$（$D = \sum_k \sum_l D_k$），学习一个共享的仿射投影 $\Lambda = \{\mathbf{W}, \mathbf{b}\}$ 将其压缩到 $D' \ll D$
3. **Transformer 处理**：压缩后的 token 加上可学习 cls token，送入 6 层小型 transformer（128维，2头，1.7M参数），cls 输出接线性分类头

### 关键设计

**空间信息保留**：与 Head2Toe/SMP 对 feature map 做平均池化不同，ComBo 对每个 token 位置独立做层维度压缩，完整保留空间布局。这对 Structured 类任务（如物体计数、距离估计）至关重要。

**仿射投影的层选择功能**：投影矩阵 $\mathbf{W}$ 在层维度上做特征选择——学会保留各模型中与任务最相关的层特征，丢弃冗余层。

**模型任务相关性评估**：在 $\mathbf{W}$ 上加 L2 正则，计算每个 backbone $M_k$ 的重要性分数 $s_k$（对应列的 L2 范数）。损失为：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{k=1}^K s_k$$

训练后 $s_k$ 直接反映各模型的任务相关性，可指导模型子集选择。选定后用子集重新训练（无正则）。

### 训练策略

- **优化器**：AdamW，lr=0.001，weight decay=0.0001
- **训练**：100 epochs，10 epochs 线性 warmup + cosine schedule
- **输入**：224×224，无数据增强
- **所有数据集使用完全相同超参**，无需逐任务调优（vs. Head2Toe/SMP 需要数据集特定超参）
- 正则系数 λ=0.01（仅在评估模型相关性时使用）

## 实验关键数据

### 主实验：单模型 Probing（VTAB-1k，ViT-B/16 ImageNet-21K）

| 方法 | 类型 | Natural | Specialised | Structured | 全局平均 |
|------|------|---------|-------------|------------|----------|
| Adapter+ | PEFT | 83.3 | 86.2 | 63.3 | **77.6** |
| Full Fine-tuning | Tuning | 78.6 | 86.3 | 57.8 | 74.2 |
| **ComBo (ours)** | Probing | 79.7 | 84.5 | **59.5** | **74.6** |
| SMP | Probing | 80.7 | 84.8 | 55.4 | 73.6 |
| Head2Toe | Probing | 80.2 | 84.7 | 47.7 | 70.9 |
| Linear Probing | Probing | 73.9 | 79.5 | 29.6 | 61.0 |

ComBo 在 probing 方法中最优，在 Structured 任务上优势巨大（59.5 vs. 55.4），超越全量微调。

### 主实验：多模型 Probing

| 方法 | Natural | Specialised | Structured | 全局平均 |
|------|---------|-------------|------------|----------|
| ComBo Top-2 模型 | 84.0 | 86.3 | **65.3** | **78.6** |
| ComBo 全部4模型 | 83.3 | 86.5 | 64.8 | 78.2 |
| RADIOv2.5 + Adapter+ (蒸馏) | 83.8 | 86.6 | — | ~78 |
| 最优单模型 (DINOv2) | 82.6 | 85.9 | 65.0 | 77.9 |

ComBo Top-2（78.6）超越了需要昂贵蒸馏的 RADIOv2.5+Adapter+，且不需要反向传播通过任何大模型。

### 消融实验

**模型相关性评估有效性**：通过正则化训练得到的重要性分数可准确识别各任务最相关的模型。选择 Top-2 模型（78.6）优于使用全部4模型（78.2），说明去除不相关模型可减少噪声、提升性能。

**空间信息保留的关键性**：ComBo 在 Structured 任务上大幅领先 Head2Toe/SMP（59.5 vs. 47.7/55.4），核心原因是保留了完整空间 feature map 而非池化。

### 关键发现

1. 不同 FM 的互补性确实存在：DINOv2 在 Natural/Structured 强，CLIP/SigLIP 在部分 Specialised 强
2. 中间层可能比最后一层更有用，多层 probing 至关重要
3. 固定超参的 ComBo 即可在 19 个多样任务上稳定表现，泛化性强
4. Probing 方法首次能匹配甚至超越蒸馏融合方法

## 亮点与洞察

- **极简但有效**：仅 1.7M 参数的 transformer + 仿射投影，不需要 backbone 梯度
- **对计算受限场景友好**：多个大模型只需前向推理提取特征，训练仅在小 adapter 上
- **统一框架**：同时适用于单模型和多模型场景，且不需逐任务调超参
- **模型选择功能**：通过 probing 权重的 L2 范数自动评估各 backbone 的任务相关性，避免逐一试验

## 局限与展望

1. **仅验证了 ViT-B 级别**：更大模型（ViT-L/G）的扩展性待验证
2. **仅分类任务**：VTAB-1k 全是分类问题，检测/分割等密集预测任务未验证
3. **需要所有模型的前向推理**：虽然不需反向传播，但存储多模型的多层特征图仍有显存开销
4. **DINOv2 单模型已很强**：多模型组合的增益有限（77.9→78.6），成本效益比需权衡
5. 可探索将 ComBo 与 PEFT 方法结合（先 ComBo 选模型→对最优模型做 Adapter+）

## 相关工作与启发

- **Head2Toe / SMP**：多层 probing 的先驱，但需超参调优和池化，ComBo 解决了这两个问题
- **RADIO / SAM-CLIP**：蒸馏融合路线，ComBo 提供了更轻量的替代方案
- **Platonic Representation**：不同 FM 是否趋同？本文实验表明当前一代模型仍有显著差异
- 启发：多模型融合不一定需要蒸馏或联合训练，冻结 probing 就能有效利用互补性

## 评分

- 新颖性：★★★★☆（首次在 probing 框架下实现高效多模型融合+模型选择）
- 技术深度：★★★☆☆（方法简洁，核心是投影+小transformer，理论分析较少）
- 实验充分度：★★★★☆（VTAB-1k 19任务全面评估，多模型组合、消融完整）
- 实用价值：★★★★★（计算友好，无需反向传播大模型，固定超参即用）

<!-- RELATED:START -->

## 相关论文

- [Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [Probing Subphonemes in Morphology Models](../../ACL2025/interpretability/probing_subphonemes_in_morphology_models.md)
- [Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Grammar](../../ICML2025/interpretability/foundation_molecular_grammar_multi-modal_foundation_models_induce_interpretable_.md)
- [LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)

<!-- RELATED:END -->
