---
title: >-
  [论文解读] TokenLight: Precise Lighting Control in Images using Attribute Tokens
description: >-
  [CVPR 2026][图像生成][relighting] 提出 TokenLight，将图像重光照表述为以属性 token（强度、颜色、环境光、漫反射级别、3D 光源位置）为条件的端到端图像生成任务，在扩散 Transformer 框架中实现精确、连续、可解释的光照控制。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "relighting"
  - "attribute tokens"
  - "Transformer"
  - "lighting control"
  - "synthetic data"
---

# TokenLight: Precise Lighting Control in Images using Attribute Tokens

**会议**: CVPR 2026  
**arXiv**: [2604.15310](https://arxiv.org/abs/2604.15310)  
**代码**: [vrroom.github.io/tokenlight/](https://vrroom.github.io/tokenlight/)  
**领域**: 图像生成/重光照  
**关键词**: relighting, attribute tokens, diffusion transformer, lighting control, synthetic data

## 一句话总结

提出 TokenLight，将图像重光照表述为以属性 token（强度、颜色、环境光、漫反射级别、3D 光源位置）为条件的端到端图像生成任务，在扩散 Transformer 框架中实现精确、连续、可解释的光照控制。

## 研究背景与动机

现有重光照方法的光照表示各有局限：文本驱动不够精确、背景图像信息有限、全景环境图无法建模近场照明、逆渲染方法依赖高质量 3D 重建。缺乏一种既精确可解释又空间定位灵活的表示来直接在 2D 图像域进行光照操控。核心需求是：在不需要 3D 重建的前提下，将 3D 光照工具的直觉灵活性与 2D 图像编辑的可访问性结合。

## 方法详解

### 整体框架

基于预训练的扩散 Transformer（文生图/视频基础模型）微调，输入为控制信号（属性 token + 输入图像）直接重渲染期望输出。大规模合成数据集提供精确光照标注，少量真实数据增强泛化。

### 关键设计

1. **属性 token 光照表示**: 每个 token 编码一个物理有意义的光照属性——强度、颜色（色温）、漫反射级别、3D 空间位置和环境光参数。各属性独立可控、连续可调。这种分解式表示天然支持解耦编辑。

2. **大规模合成数据训练**: 在 Blender 中使用路径追踪渲染器对 3D 资产在系统化变化的光照条件下生成数据集，提供每个光照属性的精确真值监督。包含单光源渲染、环境光图像和光源可见性掩码。

3. **三种实用光照控制**: (1) 添加空间虚拟光源（在任意 3D 位置放置新光源）；(2) 编辑/扩散环境光照（全局光照调节）；(3) 控制场景内光源（通过空间掩码开关场景内发光体），各种组合实现丰富创意效果。

### 损失函数 / 训练策略

扩散模型标准去噪目标。少量真实数据（场景内光源开关拍摄）补充训练以提升真实感和泛化。预训练基础模型的视觉先验在微调中被保留。

## 实验关键数据

### 主实验

在合成和真实图像上验证，与 GenLit、LightLab 等先前方法对比：

| 任务 | 指标 | 先前SOTA | TokenLight |
|------|------|---------|-----------|
| 虚拟光源添加 | 定量+定性 | 较差 | **SOTA** |
| 环境光编辑 | 定量+定性 | 较差 | **SOTA** |
| 场景内光源控制 | 定量+定性 | 较差 | **SOTA** |

### 关键发现

- 无需逆渲染监督，模型展现了对光-场景交互的内在理解
- 可将虚拟光源放置在物体内部（如南瓜灯笼效果）
- 对透明材质的重光照同样产生可信阴影
- 仅从合成数据学习的推理能力即可迁移到真实场景

## 亮点与洞察

- 属性 token 将光照控制从黑箱变为可解释的物理操控
- 端到端方法无需 3D 重建即展现 3D 光照理解能力
- 合成训练数据的规模化策略对其他需要精确标注的生成任务有指导意义

## 局限与展望

- 3D 光源位置与相机视点耦合，多视角一致性未保证
- 对极端光照条件（如高动态范围场景）的鲁棒性需测试
- 实时交互编辑的推理速度可能受限于扩散模型

## 相关工作与启发

- 属性 token 的分解式控制设计可推广到其他条件生成任务
- 合成+少量真实的训练策略平衡了精度和泛化
- 扩散 Transformer 的光照推理能力暗示了端到端物理理解的可能

## 评分

8/10 — 表示设计优雅，控制精度和视觉质量均优秀，是重光照领域的重要进展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)
- [\[CVPR 2026\] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)
- [\[CVPR 2026\] Guiding a Diffusion Model by Swapping Its Tokens](guiding_a_diffusion_model_by_swapping_its_tokens.md)
- [\[CVPR 2026\] All-in-One Slider for Attribute Manipulation in Diffusion Models](all_in_one_slider_attribute_manipulation.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)

</div>

<!-- RELATED:END -->
