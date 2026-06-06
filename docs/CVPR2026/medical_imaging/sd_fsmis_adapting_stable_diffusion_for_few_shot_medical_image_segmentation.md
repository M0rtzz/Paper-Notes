---
title: >-
  [论文解读] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation
description: >-
  [CVPR 2026][医学图像][图像分割] 提出 SD-FSMIS，一个将预训练 Stable Diffusion 适配到少样本医学图像分割的框架，通过支持-查询交互模块和视觉到文本条件转换器实现高效适配，在跨域场景中表现尤为突出。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "图像分割"
  - "扩散模型"
  - "cross-domain"
  - "foundation model"
---

# SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2604.03134](https://arxiv.org/abs/2604.03134)  
**代码**: 无  
**领域**: 医学图像分割  
**关键词**: few-shot segmentation, medical imaging, stable diffusion, cross-domain, foundation model

## 一句话总结

提出 SD-FSMIS，一个将预训练 Stable Diffusion 适配到少样本医学图像分割的框架，通过支持-查询交互模块和视觉到文本条件转换器实现高效适配，在跨域场景中表现尤为突出。

## 研究背景与动机

少样本医学图像分割（FSMIS）旨在仅用极少量标注样本就能分割新类别，解决医学影像中数据稀缺和领域迁移的核心挑战。现有方法主要聚焦设计更精巧的匹配网络，如原型网络和注意力机制，但这些从头训练的架构在跨域场景中表现脆弱。

作者提出范式转换：不再设计越来越复杂的任务特定架构，而是利用大规模预训练生成模型（如 Stable Diffusion）中蕴含的丰富视觉先验。扩散模型从海量数据（如 LAION-5B）中学习了丰富的纹理、形状和上下文表示，这些先验对密集预测任务有巨大潜力但在 FSMIS 中尚未被充分探索。

核心问题：如何高效、直接地将 SD 的通用视觉先验适配到 FSMIS 任务？

## 方法详解

### 整体框架

SD-FSMIS 重新利用 Stable Diffusion 的条件生成架构，在潜空间中处理支持集和查询集的交互。支持集和查询集先通过冻结的 VAE 编码到潜空间，U-Net 在文本嵌入条件下生成查询掩码。

### 关键设计

1. **支持-查询交互模块 (SQI)**：通过最小化修改 SD 的 U-Net 自注意力层实现少样本适配。在标准自注意力后插入额外的交叉注意力层，让查询特征注意到支持特征（作为 Key 和 Value），再经过原始的文本交叉注意力。还包含查询增强（QE）策略，利用支持集原型和余弦相似度增强查询表示。

2. **视觉到文本条件转换器 (VTCT)**：将支持集的视觉线索转换为"文本样"嵌入来条件化扩散模型。用冻结的图像编码器提取支持图像特征，通过掩码平均池化得到类别原型，再经可学习 MLP 投射到文本嵌入空间。这让模型用 SD "理解的语言"进行精确引导。

3. **单步推理设计**：推理时不需要迭代扩散过程，而是单步生成查询掩码潜表示，通过 VAE 解码器映射回像素空间并取三通道平均得到最终分割掩码。

### 损失函数 / 训练策略

使用查询掩码潜表示与预测之间的 MSE 损失训练。采用基于 episode 的元学习训练策略，1-way 1-shot 设置。利用超体素聚类生成伪标签作为训练标注，无需显式数据标注。VAE 权重冻结，仅训练少量新增参数。

## 实验关键数据

### 主实验

| 数据集 | 指标 (Dice %) | 本文 | 之前 SOTA (DIFD) | 提升 |
|--------|-------------|------|-----------------|------|
| Abd-MRI Setting 1 | Mean Dice | 83.16 | 84.12 | 接近 |
| Abd-CT Setting 1 | Mean Dice | 83.66 | 80.19 | +3.47 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 SQI | Dice 下降 | 支持-查询交互对适配至关重要 |
| 无 VTCT（null text） | Dice 下降 | 视觉条件比空文本嵌入信息量更大 |
| 无 QE | Dice 下降 | 查询增强提供了有益的原型匹配信号 |

### 关键发现

- 在标准 FSMIS 设置下取得有竞争力的结果
- 在更具挑战性的跨域场景中显著优于 SOTA 方法，展现了出色的泛化能力
- 验证了大规模生成模型的视觉先验对数据高效医学分割的巨大潜力

## 亮点与洞察

- 范式创新：从设计任务特定网络转向适配预训练基础模型，是 FSMIS 领域的重要转变
- 框架设计极简但有效，仅对 SD 做最小修改就实现了 FSMIS 适配
- VTCT 模块将视觉线索翻译为 SD "理解的语言"的思路巧妙
- 跨域泛化能力突出，说明 SD 的通用视觉先验在医学领域确实有价值

## 局限与展望

- 依赖于 SD 对灰度医学图像的适配（通过通道复制），可能不够优雅
- 目前仅在腹部 MRI/CT 数据上验证，需更多器官和模态的验证
- 单步推理虽然高效但可能牺牲了迭代精修的机会

## 相关工作与启发

- 与 DiffewS 类似利用扩散模型做少样本分割，但 DiffewS 面向自然图像，SD-FSMIS 专门适配医学场景
- 对其他需要跨域泛化的密集预测任务有启发价值

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次系统探索 SD 在 FSMIS 中的应用
- 技术深度：⭐⭐⭐⭐ — SQI 和 VTCT 设计合理
- 实验充分度：⭐⭐⭐ — 数据集范围可以更广
- 实用价值：⭐⭐⭐⭐ — 跨域泛化能力有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)

</div>

<!-- RELATED:END -->
