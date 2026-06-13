---
title: >-
  [论文解读] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning
description: >-
  [CVPR 2025][自监督学习][类增量学习] 提出TagFex框架，通过持续自监督学习捕获任务无关(task-agnostic)特征，并利用merge attention将其与任务特定特征融合后蒸馏回推理模型，缓解扩展式类增量学习中的特征碰撞问题。 扩展式类增量学习（如DER）为每个新任务扩展新的特征提取器…
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "类增量学习"
  - "特征扩展"
  - "任务无关特征"
  - "特征碰撞"
---

# Task-Agnostic Guided Feature Expansion for Class-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: [2503.00823](https://arxiv.org/abs/2503.00823)  
**代码**: [GitHub](https://github.com/bwnzheng/TagFex_CVPR2025)  
**领域**: 其他  
**关键词**: 类增量学习, 特征扩展, 任务无关特征, 自监督学习, 特征碰撞

## 一句话总结

提出TagFex框架，通过持续自监督学习捕获任务无关(task-agnostic)特征，并利用merge attention将其与任务特定特征融合后蒸馏回推理模型，缓解扩展式类增量学习中的特征碰撞问题。

## 研究背景与动机

扩展式类增量学习（如DER）为每个新任务扩展新的特征提取器，保持旧模型不变。虽然有效抵抗遗忘，但面临**特征碰撞**问题：新任务学到的特征可能与旧任务特征重叠（如两个任务都依赖颜色特征区分不同类别），导致跨任务误分类。

现有解决方案（DER的辅助分类器）依赖少量rehearsal samples来鼓励捕获多样特征，但rehearsal samples数量有限导致训练不均衡，效果次优。CKA相似度分析显示DER各模型学到的特征相似度较高（~0.35），而GradCAM可视化也表明它们关注相似区域。

核心洞察：**分类任务只要求模型捕获最小必要特征**（task-specific），大量有用但与当前分类无关的特征（task-agnostic）被忽略。如果这些任务无关特征能被捕获并传递给后续任务，新模型就能学到更多样化的特征。

## 方法详解

### 整体框架

TagFex由三部分组成：(1) 通过持续自监督学习(CSSL)训练独立模型持续捕获任务无关特征；(2) merge attention自适应融合任务无关和任务特定特征；(3) 通过KL散度蒸馏将融合后的丰富特征迁移回任务特定模型（推理只用任务特定模型）。

### 关键设计1：持续自监督学习捕获任务无关特征

- **功能**: 在每个任务中持续学习与分类无关的丰富视觉表示
- **核心思路**: 使用CaSSLe（基于SimCLR的持续自监督方法）训练独立的任务无关模型$f_{\text{ta}}$。初始任务用标准InfoNCE损失，增量任务增加预测性损失——训练predictor $g(\cdot)$使当前特征能预测上一轮特征$f'_{\text{ta}}$，保证表达能力逐任务递增
- **设计动机**: 自监督学习不受分类目标约束，能发现分类任务忽略的特征（如纹理、形状等），且通过持续学习确保随增量任务积累越来越丰富的表示

### 关键设计2：Merge Attention自适应特征融合

- **功能**: 从任务无关特征中提取对当前分类任务有用的信息
- **核心思路**: 将task-specific特征图作为Query，与task-specific和task-agnostic的Key/Value拼接后做多头注意力：$O^{(h)} = \text{Softmax}(\frac{Q^{(h)}[K_{\text{ts}}^{(h)}, K_{\text{ta}}^{(h)}]^T}{\sqrt{d/h}})[V_{\text{ts}}^{(h)}, V_{\text{ta}}^{(h)}]$。任务无关模型梯度停止，不受分类影响
- **设计动机**: 两种特征在不同空间，直接拼接不合适。注意力机制允许task-specific特征选择性地从task-agnostic特征中"取用"有价值的信息。训练过程中注意力从ta侧逐渐转移到ts侧，说明信息被逐步吸收

### 关键设计3：知识迁移（蒸馏回推理模型）

- **功能**: 将融合后的丰富特征信息迁移到任务特定模型，使其在推理时无需task-agnostic模型
- **核心思路**: 使用KL散度$\mathcal{L}_{\text{trans}} = D_{\text{KL}}(\text{StopGrad}(p_m) \| p_{\text{ts}})$将merge classifier的logits蒸馏到任务特定分类器。推理时只使用task-specific模型，保持与DER相同的参数量
- **设计动机**: 直接使用融合特征推理会受task-agnostic模型持续更新的分布漂移影响。蒸馏回task-specific模型既保证稳定性又传递了多样特征信息

### 损失函数

$\mathcal{L} = \lambda_{\text{ta}}\mathcal{L}_{\text{ta}} + \lambda_{\text{mcls}}\mathcal{L}_{\text{mcls}} + \mathcal{L}_{\text{ts}}$，其中$\mathcal{L}_{\text{ts}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{aux}} + \mathcal{L}_{\text{trans}}$。

## 实验关键数据

### 主实验：各数据集类增量学习结果

| 方法 | CIFAR100 10-10 (Last/Avg) | ImageNet100 10-10 (Last/Avg) | ImageNet1000 100-100 (Last/Avg) |
|------|---------------------------|-----------------------------|---------------------------------|
| iCaRL | 49.52/64.64 | 50.98/67.11 | 40.47/57.55 |
| DER | 64.35/75.36 | 66.71/77.18 | 58.83/66.87 |
| BEEF | 60.98/71.94 | 68.78/77.62 | 58.67/67.09 |
| **TagFex** | **68.23/78.45** | **70.84/79.27** | **61.45/68.32** |
| TagFex-P | 67.34/78.02 | 69.21/78.56 | 60.14/67.65 |

### 消融实验（CIFAR100）

| Task-agnostic | Merge Attn | Knowledge Transfer | 10-10 Last/Avg |
|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✓ | 64.45/75.34 |
| ✓ | ✓ | ✗ | 65.86/76.32 |
| ✓ | ✓ | ✓ | **68.23/78.45** |

### 关键发现

- TagFex相比DER稳定提升3-4%准确率，且推理参数量相同
- CKA相似度从DER的~0.35降到~0.2，证明特征多样性显著提升
- 注意力可视化显示训练初期关注task-agnostic侧，后期迁移到task-specific侧
- 剪枝版本TagFex-P参数从61.6M降至11.6-14.4M，准确率仅微降
- 更换SSL方法（SimCLR→BYOL）可进一步提升，说明框架对SSL方法不敏感

## 亮点与洞察

1. **特征碰撞问题的清晰阐述**: 用CKA和GradCAM可视化证明现有方法特征多样性不足
2. **训练-推理解耦设计**: 训练时用task-agnostic模型辅助，推理时完全不需要，无额外推理开销
3. **Merge Attention的注意力演化**: 从ta→ts的注意力迁移直观展示了知识吸收过程

## 局限与展望

- 训练时需要额外维护task-agnostic模型（约300样本等值的存储），虽已在memory-aligned实验中验证仍有优势
- 目前仅在CNN骨干(ResNet18)上验证，对ViT等架构的适用性待探索
- 自监督学习的额外训练开销可能在边缘场景中不可忽视

## 相关工作与启发

- 用自监督学习发现分类任务忽略的特征的思路新颖，可推广到其他需要特征多样性的场景
- "辅助训练但不参与推理"的设计范式值得在更多任务中探索

## 评分

⭐⭐⭐⭐ — 问题分析透彻，解决方案优雅且有原则性（特征碰撞→多样性→task-agnostic特征）。推理零额外开销是重要的实用性优势。实验全面，包含memory-aligned公平对比。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)
- [\[CVPR 2025\] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](../../AAAI2026/self_supervised/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](../../CVPR2026/self_supervised/hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)

</div>

<!-- RELATED:END -->
