---
title: >-
  [论文解读] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][AIGC detection] 提出AIGI-Holmes，通过构建包含解释性标注的Holmes-Set数据集、三阶段训练管线（视觉专家预训练→SFT→DPO）和协同解码策略，实现可解释且可泛化的AI生成图像检测，在三个基准上达到SOTA检测精度同时提供人类可验证的解释。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "AIGC detection"
  - "explainable AI"
  - "MLLM"
  - "DPO"
  - "collaborative decoding"
---

# AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/wyczzy/AIGI-Holmes)  
**领域**: 多模态VLM  
**关键词**: AIGC detection, explainable AI, MLLM, DPO, collaborative decoding

## 一句话总结

提出AIGI-Holmes，通过构建包含解释性标注的Holmes-Set数据集、三阶段训练管线（视觉专家预训练→SFT→DPO）和协同解码策略，实现可解释且可泛化的AI生成图像检测，在三个基准上达到SOTA检测精度同时提供人类可验证的解释。

## 研究背景与动机

AI生成图像（AIGI）越来越逼真，被滥用于传播虚假信息，威胁公共信息安全。现有检测方法面临两大问题：(1) 缺乏可解释性——黑盒模型无法提供人类可验证的证据；(2) 缺乏泛化性——难以应对最新生成技术（如FLUX、SD3.5、VAR等）。MLLM具备常识理解和自然语言生成能力，是实现可解释检测的理想候选，但直接SFT效果次优——MLLM在图像分类和低级感知上能力不足，且SFT模型容易机械复制解释模板而非真正理解伪造原因。此外，缺乏适用于SFT阶段的解释性训练数据集。

## 方法详解

### 整体框架

Holmes Pipeline包含三个训练阶段和一个推理策略。训练阶段：(1) 视觉专家预训练——在Holmes-SFTSet上通过二分类快速适配视觉编码器；(2) 监督微调(SFT)——训练MLLM生成带解释的检测结果；(3) 人类对齐DPO——在Holmes-DPOSet上进行直接偏好优化。推理阶段采用协同解码策略，融合视觉专家的感知和MLLM的语义推理。

### 关键设计

1. **Holmes数据集构建（Multi-Expert Jury标注法）**: Holmes-SFTSet包含65K图像，标注涵盖高级语义维度（物理不一致性、解剖学错误、文本渲染缺陷）和低级伪影（整体色调、纹理、边缘）。采用跨模型验证和专家引导过滤确保标注质量。Holmes-DPOSet通过正/负提示构造对比解释对，加上4K专家修正样本，实现与人类判断的对齐。使用18种AIGC方法生成图像，涵盖GAN和扩散模型。

2. **三阶段Holmes Pipeline**: 第一阶段通过二分类任务让视觉编码器快速获取域特定特征提取能力（类似adapter预训练）。第二阶段SFT使MLLM不仅检测还能生成解释——解决"黑盒"限制。第三阶段DPO从偏好样本中学习，根本性地重塑MLLM的推理模式，确保解释与人类判断标准对齐而非停留在次优SFT。

3. **协同解码策略**: 推理时将视觉专家的模型感知与MLLM的语义推理集成，创建双通道验证过程。视觉专家提供低级伪影检测信号，MLLM提供高级语义分析，两者互补增强泛化能力，特别是面对未见过的生成方法时。

### 损失函数 / 训练策略

三阶段顺序训练：视觉专家预训练（二分类交叉熵）→ SFT（标准next token prediction）→ DPO（偏好优化损失）。DPO阶段使用Holmes-DPOSet的对比解释对作为偏好数据。

## 实验关键数据

### 主实验

在三个基准数据集上验证，面对未见生成方法的泛化设置：

| 方法 | 未见GAN检测 | 未见扩散模型检测 | 未见自回归模型检测 |
|------|-----------|---------------|-----------------|
| UnivFD | 基线水平 | 基线水平 | 基线水平 |
| DRCT | 中等 | 中等 | 中等 |
| **AIGI-Holmes** | **SOTA** | **SOTA** | **SOTA** |

对最新生成器（FLUX、SD3.5、VAR等）的泛化性能显著优于现有方法。

### 消融实验

- 仅SFT vs 完整Pipeline：DPO阶段显著提升检测准确率和解释质量
- 视觉专家预训练：为MLLM提供域特定特征提取能力，消除则性能明显下降
- 协同解码：融合视觉专家和MLLM输出优于单独使用任一

### 关键发现

- 直接SFT训练MLLM进行AIGC检测效果次优，DPO对齐至关重要
- 视觉编码器的域适配（通过二分类预训练）是泛化的关键
- 高级语义分析（物理不一致、解剖错误）和低级伪影分析互补
- 解释性不仅提升可信度，还间接提升检测泛化能力

## 亮点与洞察

- "像福尔摩斯一样"的设计理念——不仅识别真假，还提供证据链
- 三阶段训练设计环环相扣，每阶段解决特定问题
- Holmes-DPOSet是首个用于AIGC检测的人类对齐偏好数据集
- 协同解码策略巧妙融合了传统分类器和MLLM的各自优势

## 局限与展望

- 数据集构建仍依赖专家过滤和人工修正，扩展成本不低
- MLLM推理开销大，难以用于实时大规模检测
- 当前解释主要基于视觉伪影，对完全无伪影的高质量生成图像可能失效
- 未评估对抗攻击（如针对检测器的对抗样本）的鲁棒性

## 相关工作与启发

- UnivFD使用CLIP-ViT特征、DRCT引入重建对比学习、NPR分析邻域像素关系
- DD-VQA、FFAA等开创了MLLM用于深度伪造检测
- DPO在LLM对齐中的成功被有效迁移到视觉检测领域
- 数据集构建中的Multi-Expert Jury思路可扩展到其他标注任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三阶段Pipeline和协同解码设计新颖
- 技术深度: ⭐⭐⭐⭐ — 数据集构建和训练流程设计严谨
- 实验充分性: ⭐⭐⭐⭐ — 三基准、泛化测试、消融完整
- 写作质量: ⭐⭐⭐⭐ — 类比福尔摩斯，叙事生动
- 实用价值: ⭐⭐⭐⭐ — 可解释检测在实际场景中价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)

</div>

<!-- RELATED:END -->
