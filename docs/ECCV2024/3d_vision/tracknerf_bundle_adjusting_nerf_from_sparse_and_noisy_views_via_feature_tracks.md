---
title: >-
  [论文解读] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks
description: >-
  [ECCV 2024][3D视觉] 提出TrackNeRF，将SfM中的特征轨迹（feature tracks）引入NeRF训练，通过全局多视角重投影一致性损失替代传统的成对对应损失，显著提升稀疏+有噪声位姿下的NeRF重建质量和位姿优化精度。
tags:
  - "ECCV 2024"
  - "3D视觉"
---

# TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks

**会议**: ECCV 2024  
**arXiv**: [2408.10739](https://arxiv.org/abs/2408.10739)  
**代码**: [项目页](https://tracknerf.github.io/)  
**领域**: 3D视觉

## 一句话总结

提出TrackNeRF，将SfM中的特征轨迹（feature tracks）引入NeRF训练，通过全局多视角重投影一致性损失替代传统的成对对应损失，显著提升稀疏+有噪声位姿下的NeRF重建质量和位姿优化精度。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：NeRF通常假设大量精确位姿的图像，但实际场景中视角常稀疏、位姿含噪声

### 领域现状

**领域现状**：BARF仅做频率调制而不使用多视角约束，本质上并非真正的bundle adjustment

### 现有痛点

**现有痛点**：SPARF引入了成对对应损失，但仅考虑两视角间的局部一致性，忽视了全局几何约束

### 解决思路

**解决思路**：核心问题**：所有视角来自同一个3D场景，对应像素应投影回同一个3D地标——需要全局一致性而非成对一致性

## 方法详解

### 整体框架

1. 使用PDCNet++提取所有图像对之间的密集对应关系
2. 通过链式传递构建跨所有视角的特征轨迹
3. 按照PixSfM的方式对特征轨迹进行关键点调整（TKA）
4. 设计轨迹重投影损失（Track Reprojection Loss）联合优化NeRF参数和相机位姿

### 关键设计

**特征轨迹提取**：如果（u,v）在图像i和j之间匹配，（v,q）在j和k之间匹配，则建立传递性关系形成连通轨迹T_k = {u, v, q, ...}

**轨迹关键点调整**：最小化轨迹内所有成对特征的加权特征距离，通过数值梯度优化关键点位置，获得更准确的监督信号

**轨迹重投影损失（核心）**：对轨迹T_k中的每对对应(u_i, v_j)，将v_j通过渲染深度反投影到3D再重投影到i的图像平面，最小化与u_i的距离。使用Huber损失增强鲁棒性。

**深度正则化**：鼓励深度梯度与渲染图像梯度对齐，缓解稀疏视角下的几何歧义和浮点伪影

### 损失函数

$$\mathcal{L} = \mathcal{L}_{Photometric} + \lambda_{Depth}\mathcal{L}_{Depth} + \lambda_{Track}\mathcal{L}_{Track}$$

光度损失保证外观一致；深度正则化缓解几何歧义；轨迹损失强制全局几何一致性。

## 实验关键数据

### 主实验

DTU数据集，15%高斯噪声位姿：

| 设置 | 方法 | Rot.↓ | Trans.↓ | PSNR↑ | SSIM↑ | DE↓ |
|------|------|-------|---------|-------|-------|-----|
| 3-view | SPARF | 1.81 | 5.0 | 17.74 | 0.71 | 0.12 |
| 3-view | **TrackNeRF** | **1.12** | **2.48** | **18.53** | **0.73** | **0.11** |
| 6-view | SPARF | 1.31 | 2.7 | 21.39 | 0.81 | 0.09 |
| 6-view | **TrackNeRF** | **0.24** | **0.65** | **22.78** | **0.84** | **0.06** |
| 9-view | SPARF | 1.15 | 2.55 | 24.69 | 0.88 | 0.06 |
| 9-view | **TrackNeRF** | **0.25** | **0.70** | **25.57** | **0.89** | **0.05** |

### 消融实验

DTU 3-view GT位姿对比（部分）：

| 方法 | PSNR↑(masked) | SSIM↑(masked) | LPIPS↓(masked) |
|------|---------------|---------------|----------------|
| FreeNeRF | 20.46 | 0.83 | 0.17 |
| CorresNeRF | 20.58 | 0.77 | - |
| SPARF | 21.22 | 0.85 | 0.12 |
| **TrackNeRF** | **21.70** | **0.85** | **0.12** |

### 关键发现

- 6-view场景改善最大（PSNR +1.65, 位姿误差降低约80%），因为更多视角形成更长的特征轨迹
- 3-view场景位姿旋转误差几乎减半（1.81→1.12）
- 在GT位姿的3-view设置下也优于基于扩散先验的ReconFusion等方法
- 全局特征轨迹约束是位姿优化加速和精度提升的关键因素

## 亮点与洞察

- 将经典SfM中的bundle adjustment思想（轨迹级）无缝融入NeRF优化，理论动机清晰
- 不依赖任何生成先验（扩散模型），仅利用通用几何线索，泛化性更强
- 特征轨迹的关键点调整步骤进一步提升对应质量，为特征噪声提供了额外的鲁棒性
- 6-view设置下的巨大提升说明：中等稠密视角下全局一致性约束的价值最大

## 与BARF/SPARF的方法论对比

BARF首先提出频率调制来优化位姿，但不使用任何多视角对应约束，因此并非真正意义上的bundle adjustment。SPARF引入成对对应损失，是显著进步，但仍局限于两两视角间的局部一致性。TrackNeRF直接在NeRF框架内实现了轨迹级的BA目标函数，一条特征轨迹可能连接所有可见视角的像素，强制全局一致性。

LLFF数据集上TrackNeRF也展示了稳定的提升，在3-view有噪声位姿设置下渲染深度图更平滑、floater更少。这主要归功于深度正则化损失，它鼓励深度梯度与图像梯度对齐。

## 局限与展望

- 依赖密集对应网络（PDCNet++）的质量，错误匹配会传播到轨迹中
- 训练成本与SPARF相当，实际部署仍偏慢
- 极端稀疏（2-view）场景下特征轨迹退化为成对对应，优势减弱
- 特征轨迹链长度与视角数成正比，开销随视角数增长

## 评分

- 新颖性：⭐⭐⭐⭐ — 经典idea的优雅迁移
- 有效性：⭐⭐⭐⭐⭐ — 全面SOTA，位姿和PSNR双提升
- 实用性：⭐⭐⭐⭐
- 推荐度：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)

</div>

<!-- RELATED:END -->
