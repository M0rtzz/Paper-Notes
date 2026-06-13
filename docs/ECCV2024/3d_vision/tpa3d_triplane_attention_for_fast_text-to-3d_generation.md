---
title: >-
  [论文解读] TPA3D: Triplane Attention for Fast Text-to-3D Generation
description: >-
  [ECCV 2024][3D视觉] 提出TPA3D，一个基于GAN的文本引导3D生成框架，通过三平面注意力（TPA）模块对句子级和词级文本特征进行逐层细化，实现快速且细粒度的文本到3D纹理网格生成。 - 现有文本到3D方法主要依赖2D扩散模型进行SDS优化（如DreamFusion、Magic3D），推理时间长达数十分钟甚至…
tags:
  - "ECCV 2024"
  - "3D视觉"
---

# TPA3D: Triplane Attention for Fast Text-to-3D Generation

**会议**: ECCV 2024  
**arXiv**: [2312.02647](https://arxiv.org/abs/2312.02647)  
**代码**: 无  
**领域**: 3D视觉

## 一句话总结

提出TPA3D，一个基于GAN的文本引导3D生成框架，通过三平面注意力（TPA）模块对句子级和词级文本特征进行逐层细化，实现快速且细粒度的文本到3D纹理网格生成。

## 研究背景与动机

- 现有文本到3D方法主要依赖2D扩散模型进行SDS优化（如DreamFusion、Magic3D），推理时间长达数十分钟甚至数小时
- 基于视觉-语言模型的GAN方法（如TAPS3D）仅使用CLIP全局句子特征，无法捕获文本中的细粒度描述信息
- 缺乏大规模文本-3D配对数据，需要在无监督设置下实现文本引导的3D生成
- **核心问题**：现有GAN方法仅利用全局语义特征，导致不同细粒度描述生成的形状和纹理高度相似

## 方法详解

### 整体框架

TPA3D构建在GET3D之上，包含两个核心模块：
1. **句子级三平面生成器G**：将CLIP句子特征与随机噪声拼接，通过调制卷积生成句子级几何/纹理三平面
2. **三平面注意力块TPA**：对句子级三平面进行词级细化，生成包含3D空间信息和词级信息的精细三平面

使用InstructBLIP为渲染图像自动生成伪标题，无需人工标注的文本-3D配对数据。

### 关键设计

**TPA模块包含三种注意力机制**：
- **平面内自注意力**：对每个平面特征独立做自注意力，保持各平面内部一致性
- **跨平面注意力**：融合三平面特征作为key/value，各平面内容特征作为query，建立3D空间连通性
- **跨词注意力**：以自精炼特征为query，CLIP词特征为key/value，注入细粒度词级信息

纹理TPA额外引入几何三平面（权重α=0.5）作为输入，确保纹理与几何对应。判别器使用句子特征+相机位姿作为条件，并引入错配目标函数增强对不匹配文本的敏感性。

### 损失函数

总损失 = RGB对抗损失 + Mask对抗损失 + 错配损失 + CLIP相似度损失

错配损失使用不匹配的句子特征构建负样本，增强判别器的区分能力。CLIP损失计算生成图像与文本的相似度，稳定训练过程。

## 实验关键数据

### 主实验

| 方法 | Car(FID↓) | Chair(FID↓) | Motorbike(FID↓) | Vehicle(FID↓) | Acc.(FID↓) |
|------|-----------|-------------|-----------------|---------------|------------|
| GET3D | 11.50 | 22.75 | 49.98 | 98.15 | 145.66 |
| TAPS3D | 26.37 | 44.70 | 84.83 | 152.34 | 172.14 |
| **TPA3D** | **18.50** | **38.11** | **77.69** | **68.80** | **83.31** |

| 方法 | Car(R-Prec@5↑) | Chair | Motorbike | Vehicle | Acc. |
|------|----------------|-------|-----------|---------|------|
| TAPS3D | 12.55 | 7.52 | 5.00 | 9.47 | 6.67 |
| **TPA3D** | **80.94** | **38.58** | **24.76** | **65.26** | **64.44** |

### 消融实验

| 方法 | 设备 | 输出类型 | 推理时间 |
|------|------|----------|----------|
| DreamFusion | TPUv4 | Rendering | 90 min |
| Magic3D | A100 x8 | Rendering | 40 min |
| TAPS3D | V100-32G | Mesh | 1.03 sec |
| **TPA3D** | V100-32G | Mesh | 2.87 sec |
| **TPA3D** | V100-32G | Rendering | 0.09 sec |

### 关键发现

- CLIP R-Precision@5指标上TPA3D大幅领先TAPS3D（Car: 80.94% vs 12.55%），词级细化极其有效
- 在高多样性OmniObject3D数据集上FID优于无条件GET3D，文本引导有利于多类别生成
- 推理速度与GAN方法相当（毫秒级渲染），比SDS方法快3-4个数量级
- 固定随机种子仅修改文本可增量操控3D物体细节

## 亮点与洞察

- 首次在GAN框架中引入词级三平面注意力，突破了全局特征的信息瓶颈
- 通过InstructBLIP消除了对人工标注文本-3D数据的依赖
- 对比SDS方法，在复杂多属性文本（如"红色轮毂+蓝色座椅"）下能更准确分离颜色到不同部位

## 局限性

- 受限于ShapeNet/OmniObject3D的类别多样性，扩展到开放世界场景具有挑战
- 纹理细节相比真实3D扫描仍有差距
- 三平面表示的分辨率对几何/纹理上限有约束

## 评分

- 新颖性：⭐⭐⭐⭐
- 有效性：⭐⭐⭐⭐
- 实用性：⭐⭐⭐⭐⭐
- 推荐度：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

</div>

<!-- RELATED:END -->
