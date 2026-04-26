---
title: >-
  [论文解读] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation
description: >-
  [CVPR 2025][图像生成][漫画生成] 提出 DiffSensei，结合扩散模型与MLLM实现多角色定制化漫画生成，并发布首个大规模漫画数据集 MangaZero（43K页/427K标注面板）。
tags:
  - CVPR 2025
  - 图像生成
  - 漫画生成
  - 扩散模型
  - MLLM
  - 故事可视化
---

# DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation

**会议**: CVPR 2025  
**arXiv**: [2412.07589](https://arxiv.org/abs/2412.07589)  
**代码**: https://jianzongwu.github.io/projects/diffsensei/ (项目页)  
**领域**: 图像生成  
**关键词**: 漫画生成, 扩散模型, MLLM, 角色定制, MangaZero

## 一句话总结
本文提出新任务"定制化漫画生成"并引入 DiffSensei 框架，用 MLLM 作为文本兼容的角色适配器连接扩散模型，通过 masked cross-attention 实现精确布局控制，在自建的大规模 MangaZero 数据集（43K页/427K标注面板）上显著超越现有方法。

## 研究背景与动机

**领域现状**：故事可视化（从文本生成视觉叙事）近年取得进展，但漫画生成因其独特需求（角色跨面板一致性、精确布局控制、对话整合）一直是未被充分探索的方向。现有方法主要做低层次的图像风格转换，无法从头生成定制化漫画。

**现有痛点**：(1) 零样本角色定制方案容易产生"复制粘贴"效果，角色表情/姿态缺乏变化；(2) 现有故事可视化数据集缺少角色标注和布局控制信息；(3) 扩散模型生成的角色图像倾向于僵化追随输入角色图的像素分布，即使在训练数据中看到同一角色的多种状态。

**核心矛盾**：需要在保持角色身份一致的同时让其表情、姿态、动作根据面板文本动态变化，单纯的图像条件注入难以实现这种灵活控制。

**本文目标**：提出定制化漫画生成任务并构建完整的数据集+框架解决方案。

**切入角度**：借鉴 MLLM 在图像编辑中的文本理解能力，用 MLLM 作为角色特征的文本兼容适配器，让角色特征在输入扩散模型前就已经根据文本提示进行了调整。

**核心 idea**：MLLM 不直接生成图像，而是调整角色特征向量使其与面板文本对齐，然后通过 masked cross-attention 注入扩散模型的正确空间位置。

## 方法详解

### 整体框架
DiffSensei 由两个核心组件组成：(1) 基于扩散模型的图像生成器——以 SD 为基础，接收面板文本和角色特征生成漫画面板；(2) MLLM角色适配器——接收角色参考图和面板描述文字，输出根据文本调整过的角色特征向量。此外包含布局控制（masked attention）和对话嵌入模块。

### 关键设计

1. **MLLM 角色适配器（Text-Compatible Identity Adapter）**:

    - 功能：根据面板文本描述动态调整角色特征，实现表情/姿态/动作的灵活变化
    - 核心思路：将角色参考图通过视觉编码器提取特征，与面板描述文本一起输入 MLLM。MLLM 的文本理解能力使其能够理解"角色应该在这个面板中展现什么状态"，输出修改后的角色特征向量。这些向量再通过 cross-attention 注入扩散模型
    - 设计动机：直接将角色图像特征注入扩散模型会导致僵化的像素复制，MLLM作为中间层能根据语义进行特征调整

2. **Masked Cross-Attention**:

    - 功能：精确控制多角色在面板中的空间布局
    - 核心思路：对每个角色定义空间 mask，在扩散模型的 cross-attention 层中限制角色特征只影响其对应的空间区域。不同于直接像素级别的条件注入，masked attention 在特征级别工作，允许扩散模型在约束范围内自由生成细节
    - 设计动机：漫画需要多角色精确定位，简单的全局条件注入无法支持多角色场景

3. **MangaZero 数据集**:

    - 功能：提供首个支持多角色、多状态漫画生成的大规模训练数据
    - 核心思路：收集日本黑白漫画，包含 43,264 页和 427,147 个标注面板。每个面板标注了角色ID、角色区域、面板描述和对话文字。同一角色在不同面板中展现不同表情和姿态
    - 设计动机：缺乏多角色多状态标注数据是该领域的瓶颈

### 损失函数 / 训练策略
基于扩散模型的标准去噪损失，配合角色一致性损失确保生成的角色与参考图身份一致。MLLM 组件在联合训练中学习文本条件调整。

## 实验关键数据

### 主实验

| 方法 | 角色一致性 | 文本遵循度 | 视觉质量 |
|------|-----------|-----------|---------|
| DiffSensei | **最优** | **最优** | **最优** |
| IP-Adapter + ControlNet | 次优 | 中等 | 良好 |
| StoryDiffusion | 中等 | 中等 | 中等 |

### 关键发现
- MLLM 适配器有效解决了"像素复制"问题，生成的角色表情和姿态能根据面板文本动态变化
- Masked attention 成功支持多达4个角色的同时精确布局控制
- MangaZero 数据集的多状态标注对训练角色变化能力至关重要
- 可应用于真实人脸照片的漫画化

## 亮点与洞察
- 将 MLLM 作为"语义调节器"而非生成器的设计思路巧妙——不改变扩散模型的生成过程，只改变输入条件
- MangaZero 数据集的构建对推动漫画/故事可视化研究有重要基础设施价值
- Masked cross-attention 的布局控制机制可迁移到其他需要精确多实体控制的生成任务

## 局限与展望
- 仅针对黑白日本漫画风格，彩色漫画和其他风格适用性未验证
- 角色姿态变化仍受限于训练数据中的姿态分布
- 对话气泡的位置和样式控制相对粗糙

## 相关工作与启发
- **vs IP-Adapter**: 通过图像提示注入实现角色一致，但缺乏文本驱动的动态调整。DiffSensei的MLLM适配器增加了语义级别的控制
- **vs StoryDiffusion**: 关注连贯图像序列但缺少角色标注和布局控制。DiffSensei提供漫画特化的精确控制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 新任务定义+MLLM作角色适配器+大规模数据集，贡献全面
- 实验充分度: ⭐⭐⭐⭐ 定性结果丰富，定量评估覆盖多维度
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，可视化效果出色
- 价值: ⭐⭐⭐⭐⭐ 对漫画/故事生成领域有开创性贡献

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[CVPR 2025\] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction](syncvp_joint_diffusion_for_synchronous_multi-modal_video_prediction.md)

<!-- RELATED:END -->
