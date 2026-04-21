---
title: >-
  [论文解读] Targeted Forgetting of Image Subgroups in CLIP Models
description: >-
  [CVPR 2025][模型压缩][CLIP遗忘] 提出三阶段 CLIP 模型子群遗忘方法：遗忘阶段用相对 Fisher 信息定位关键层+LoRA 微调遗忘目标类；提醒阶段用分布对齐在保留集上恢复；恢复阶段用模型汤（model souping）恢复零样本能力。在 ImageNet-1K 上遗忘分数达 91.0（基线 68.9）。
tags:
  - CVPR 2025
  - 模型压缩
  - CLIP遗忘
  - 子群遗忘
  - 相对Fisher信息
  - LoRA遗忘
  - 模型汤
---

# Targeted Forgetting of Image Subgroups in CLIP Models

**会议**: CVPR 2025  
**arXiv**: [2506.03117](https://arxiv.org/abs/2506.03117)  
**代码**: 有（项目主页）  
**领域**: 模型压缩 / 机器遗忘  
**关键词**: CLIP遗忘, 子群遗忘, 相对Fisher信息, LoRA遗忘, 模型汤

## 一句话总结

提出三阶段 CLIP 模型子群遗忘方法：遗忘阶段用相对 Fisher 信息定位关键层+LoRA 微调遗忘目标类；提醒阶段用分布对齐在保留集上恢复；恢复阶段用模型汤（model souping）恢复零样本能力。在 ImageNet-1K 上遗忘分数达 91.0（基线 68.9）。

## 研究背景与动机

**领域现状**：CLIP 等基础模型包含从大规模互联网数据中学到的丰富知识。隐私法规（如 GDPR）要求在用户请求时删除特定类别的知识，但 CLIP 的视觉-语言对齐使遗忘更复杂——不仅要遗忘视觉特征，还要断开视觉-文本对齐。

**现有痛点**：直接微调遗忘会导致灾难性过遗忘——不仅目标类被遗忘，相邻类也被连带影响。且遗忘后 CLIP 的零样本泛化能力严重退化。

**核心矛盾**：需要精准遗忘目标子群（如"ImageNet中的猴子里的狨猴"）同时不影响父类（"猴子的其他种类"）和零样本能力。

**切入角度**：用相对 Fisher 信息（目标/保留数据的 Hessian 比）定位只对目标类重要的层，仅在这些层上用 LoRA 做精准遗忘。

**核心 idea**：相对 Fisher 定位关键层 + LoRA 精准遗忘 + 分布对齐提醒 + 模型汤恢复 = 子群级 CLIP 遗忘。

## 方法详解

### 关键设计

1. **遗忘阶段**：相对 Fisher 信息 $\mathcal{I}^l = \frac{\mathbb{E}[\nabla^2_{\theta^l} \text{loss}_f]}{\mathbb{E}[\nabla^2_{\theta^l} \text{loss}_r]}$ 找到对目标类重要、对保留类不重要的层，用 LoRA 在这些层上微调

2. **提醒阶段**：在保留集上分布对齐微调+EMA 参数稳定，恢复被遗忘过程中受损的保留类性能

3. **恢复阶段**：将遗忘后模型与原始模型做模型汤（加权平均），平衡遗忘效果和零样本泛化

### 损失函数 / 训练策略

遗忘用对比损失破坏目标类的视觉-文本对齐，提醒用分布匹配损失。模型汤系数 $\alpha$ 控制遗忘/保留平衡。

## 实验关键数据

| 场景 | 方法 | 遗忘分数 |
|------|------|---------|
| ImageNet 狨猴 in 猴子 | **三阶段** | **91.0** |
| ImageNet 狨猴 in 猴子 | 单阶段 GA | 45.3 |
| ImageNet 狨猴 in 猴子 | 基线 | 68.9 |

### 关键发现
- 提醒阶段不可缺——无提醒则灾难性过遗忘
- 三阶段 vs 单阶段：91.0 vs 45.3，结构化流程远优于直接遗忘
- 相对 Fisher 信息有效定位关键层

## 亮点与洞察
- **子群级遗忘是比类级更精细的需求**——不删除"猴子"而只删除"特定种类的猴子"
- **模型汤恢复零样本能力**——简洁有效的后处理技巧

## 局限与展望
- 需要精心构建保留集
- 模型汤的混合系数需要调参
- 分布差距大时恢复不完美

## 评分
- 新颖性: ⭐⭐⭐⭐ 子群级 CLIP 遗忘的新问题定义
- 实验充分度: ⭐⭐⭐⭐ CIFAR-10/ImageNet/风格遗忘多场景
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为 CLIP 隐私合规提供了实用方案

<!-- RELATED:START -->

## 相关论文

- [Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)
- [Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation](parameter_efficient_mamba_tuning_via_projector-targeted_diagonal-centric_linear_.md)
- [MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [Layered Image Vectorization via Semantic Simplification](layered_image_vectorization_via_semantic_simplification.md)
- [CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)

<!-- RELATED:END -->
