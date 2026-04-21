---
title: >-
  [论文解读] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models
description: >-
  [CVPR 2025][图像生成][3D头发重建] 提出 DiffLocks，通过自动化 Blender 管线构建 4 万种发型的最大合成头发数据集，用发丝 VAE 将 10 万根发丝编码为 256×256 头皮纹理图，再训练 image-conditioned HDiT 从单张图像扩散去噪生成头皮纹理，实现 2.33 秒 10 万发丝重建，速度比基线快 8-23×。
tags:
  - CVPR 2025
  - 图像生成
  - 3D头发重建
  - 头发扩散
  - 发丝VAE
  - 头皮纹理图
  - 大规模合成数据
---

# DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2505.06166](https://arxiv.org/abs/2505.06166)  
**代码**: https://radualexandru.github.io/difflocks/ (代码+数据)  
**领域**: 3D视觉 / 扩散模型  
**关键词**: 3D头发重建, 头发扩散, 发丝VAE, 头皮纹理图, 大规模合成数据

## 一句话总结
提出 DiffLocks，通过自动化 Blender 管线构建 4 万种发型的最大合成头发数据集，用发丝 VAE 将 10 万根发丝编码为 256×256 头皮纹理图，再训练 image-conditioned HDiT 从单张图像扩散去噪生成头皮纹理，实现 2.33 秒 10 万发丝重建，速度比基线快 8-23×。

## 研究背景与动机

**领域现状**：单图 3D 头发重建对数字人、VR、游戏等应用至关重要。现有方法（NeuralHDHair, HairStep）依赖朝向图（orientation maps）作为中间表示，再优化 guide strands 并上采样。

**现有痛点**：(1) 朝向图依赖脆弱——需要精确的 2D 头发朝向估计，对卷发/爆炸头等复杂发型失败；(2) guide-strand 上采样的后处理管线慢且复杂（18-53 秒/图）；(3) 缺乏大规模多样化的 3D 头发训练数据。

**核心矛盾**：需要处理 10 万根发丝的高维输出，但直接在发丝空间做生成不可行；朝向图丢失了 3D 信息（如背面结构）。

**本文目标** 从单张图像直接生成 10 万根 3D 发丝，无需朝向图或后处理，支持卷发、爆炸头等复杂发型。

**切入角度**：将发丝编码为低维 latent code 排列在 2D 头皮纹理图上——将 3D 发丝重建转化为 2D 纹理图生成问题，天然适合扩散模型。

**核心 idea**：发丝 VAE 编码 10 万根发丝为 256×256 头皮纹理图 + DINOv2 条件的 HDiT 扩散去噪生成纹理 + 密度图采样并行解码发丝。

## 方法详解

### 关键设计

1. **4 万发型合成数据集**：Blender geometry-node 管线，110 个可随机化参数（长度/卷曲/密度/颜色等）。每种发型约 10 万根发丝 + 配对 RGB 渲染图。自动化管线首次实现大规模 3D 头发数据

2. **发丝 VAE + 头皮纹理编码**：每根发丝编码为 64 维 latent code（带曲率损失保留卷曲细节）。按发丝根部在头皮上的位置排列到 256×256 纹理图 + 密度图。将 10 万发丝的高维问题转为标准 2D 图像生成

3. **Image-conditioned HDiT**：Hourglass Diffusion Transformer 在头皮纹理空间去噪。DINOv2 提取输入图像的局部特征图（cross-attention）和全局 CLS token，替代了脆弱的朝向图。DINOv2 在合成/真实图像上都泛化良好

## 实验关键数据

| 方法 | F-score (4mm/40°)↑ | 运行时间(s)↓ |
|------|-------------------|------------|
| NeuralHDHair | 68.73 | 18.32 |
| HairStep | 57.20 | 53.40 |
| **DiffLocks** | **76.20** | **2.33** |

F-score 提升 11%，速度快 8-23×。在自身评估集上更是达到 91.17 F-score（基线仅 45-49）。

### 关键发现
- DINOv2 特征比朝向图更鲁棒——对卷发、爆炸头等复杂发型效果显著更好
- 头皮纹理表示使 10 万发丝的并行解码成为可能（2.33s 含所有步骤）
- 仅在合成数据训练但泛化到真实照片（得益于 DINOv2 的域泛化能力）

## 局限性
- 数据集缺少辫子、马尾、丸子头等复杂发型
- 优先局部发丝细节而非完美像素对齐
- 评估集来自同一 Blender 管线存在偏差

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 头皮纹理表示+大规模合成数据+扩散生成的组合高度原创
- 实验充分度: ⭐⭐⭐⭐ 多基准+速度分析+真实图像泛化
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 首次实现爆炸头等复杂发型的 3D 重建，速度实用

<!-- RELATED:START -->

## 相关论文

- [DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](../../ICCV2025/image_generation/facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)

<!-- RELATED:END -->
