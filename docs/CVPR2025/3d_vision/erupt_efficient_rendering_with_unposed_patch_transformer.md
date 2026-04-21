---
title: >-
  [论文解读] ERUPT: Efficient Rendering with Unposed Patch Transformer
description: >-
  [CVPR 2025][3D视觉][新视角合成] ERUPT 提出了一种高效的潜在视角合成模型，通过 patch-based 解码器替代像素级解码、可学习的潜在相机位姿以及冻结 DINOv2 特征提取器，在不需要精确相机位姿的情况下仅用 5 张无位姿图像即可实现 600fps 的新视角合成，在 MSN 数据集上达到 SOTA 性能。
tags:
  - CVPR 2025
  - 3D视觉
  - 新视角合成
  - 场景表示
  - 无位姿渲染
  - Patch解码器
  - 潜在视角合成
---

# ERUPT: Efficient Rendering with Unposed Patch Transformer

**会议**: CVPR 2025  
**arXiv**: [2503.24374](https://arxiv.org/abs/2503.24374)  
**代码**: 无（提供数据集 MSVS-1M）  
**领域**: 3D视觉  
**关键词**: 新视角合成, 场景表示, 无位姿渲染, Patch解码器, 潜在视角合成

## 一句话总结
ERUPT 提出了一种高效的潜在视角合成模型，通过 patch-based 解码器替代像素级解码、可学习的潜在相机位姿以及冻结 DINOv2 特征提取器，在不需要精确相机位姿的情况下仅用 5 张无位姿图像即可实现 600fps 的新视角合成，在 MSN 数据集上达到 SOTA 性能。

## 研究背景与动机

1. **领域现状**：新视角合成领域主要依赖 NeRF 和 3D Gaussian Splatting 两大框架，它们通过对每个场景训练独立模型来实现高质量渲染，但需要密集图像和精确的相机位姿。近期 SRT 和 RUST 等方法探索了基于潜在场景表示的通用化重建方案。

2. **现有痛点**：NeRF/3DGS 每个新场景都需要重新训练，且依赖大量带精确位姿的输入图像。SRT 需要所有图像的精确相机参数；RUST 虽然支持无位姿训练，但推理时需要目标图像的一部分来查询模型，无法直接控制相机。二者都采用逐像素解码，计算量巨大。

3. **核心矛盾**：现有潜在场景表示方法在三个维度上存在瓶颈——（1）无法在无位姿数据上有效训练，（2）无法在推理时直接控制相机，（3）逐像素解码导致计算效率极低。

4. **本文目标** 设计一个同时支持有位姿和无位姿训练、推理时可直接指定相机位姿、且计算效率提升一个数量级的通用化新视角合成模型。

5. **切入角度**：将编码器中的自注意力和交叉注意力交替使用来区分不同图像的 token，并引入 patch-based 解码替代像素级解码。

6. **核心 idea**：通过"patch光线查询+可学习潜在位姿+交替注意力场景Transformer"三位一体的架构设计，同时解决了无位姿训练、直接相机控制和计算效率三个问题。

## 方法详解

### 整体框架
ERUPT 的输入是一组无序的（可能无位姿的）场景图像（通常5张），输出是从任意指定相机位姿渲染的新视角图像。整个 pipeline 分为三个阶段：（1）使用冻结的 DINOv2 提取每张图像的特征 token；（2）通过交替自注意力和交叉注意力的场景 Transformer 生成紧凑的场景表示和每张图像的相机位姿估计；（3）使用 patch-based 解码器从场景表示中高效渲染目标图像。

### 关键设计

1. **交替注意力场景 Transformer**:

    - 功能：从无位姿输入图像中提取紧凑的场景表示，同时估计每张图像的相对相机位姿。
    - 核心思路：与 SRT 直接拼接所有 token 做全局注意力不同，ERUPT 交替进行图像内自注意力（同一图像 token 之间混合）和全场景交叉注意力（每张图像 token 与所有场景 token 混合）。每张图像附加一个可学习的 camera token，在交替注意力过程中自然聚合相机信息。场景 Transformer 共 6 个 block。
    - 设计动机：SRT 的简单 token 混合无法区分不同图像的 token，RUST 只用简单标记区分参考帧和非参考帧。交替注意力的设计使模型即使在不知道相机参数的情况下也能区分不同图像并构建鲁棒的场景表示。

2. **目标相机切换（Target Camera Switching）**:

    - 功能：在训练时支持无位姿数据，在推理时支持直接相机控制。
    - 核心思路：将估计的潜在位姿和地面真值的正弦编码拼接，训练时以 1/3 的概率随机采样三种模式——仅潜在位姿、仅真值位姿、两者同时提供。推理时只使用编码的目标位姿，实现直接相机控制。消融实验显示仅用 5% 真值位姿性能下降极小。
    - 设计动机：RUST 需要目标图像的一半来估计位姿，限制了可生成的视角。这种切换机制让模型能在位姿不准确时通过潜在通道生成正确输出，同时保持推理时的精确控制。

3. **Patch-Based 解码器**:

    - 功能：将渲染效率提升一个数量级。
    - 核心思路：使用 8×8 的 patch 光线替代逐像素光线查询场景表示，每次查询恢复 64 个像素而非 1 个。解码器由 4 个标准 Transformer decoder block 组成，交替自注意力和与场景表示的交叉注意力，最后接 3 个卷积 pixel unshuffle 上采样块。额外的 token 解码器匹配 DINOv2 backbone 的语义 embedding。
    - 设计动机：SRT 和 RUST 的逐像素解码在 224 分辨率下 VRAM 不足（RUST 在 48GB A6000 上 OOM），patch 解码使 VRAM 需求降低 64 倍。

### 损失函数 / 训练策略
- 图像解码器使用 $L_2$ 像素损失；token 解码器使用 ArcGeo 辅助损失匹配 DINOv2 语义特征；相机位姿使用 $L_2$位置损失 + 负余弦视角损失。训练 5 个目标图像/场景以复用场景表示。AdamW 优化器，160 epochs，batch size 128。GAN 微调替换 $L_2$ 为 $L_1$+感知+GAN 损失；Stable Diffusion 渲染使用 token 解码器输出作为 prompt，微调 SD U-Net 20 epochs。

## 实验关键数据

### 主实验

| 方法 | 输入位姿 | 目标位姿 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|---------|---------|-------|-------|--------|------|
| SRT | ✓ | ✓ | 23.41 | 0.697 | 0.369 | - |
| SRT* | ✓ | ✓ | 25.93 | - | 0.237 | 67.29 |
| RUST | ✗ | ✗ | 23.49 | 0.703 | 0.351 | - |
| DORSal | ✗ | ✗ | 18.99 | - | 0.265 | 9.00 |
| ERUPT L+LORA | ✗ | 部分 | **25.26** | **0.769** | 0.340 | 91.1 |
| ERUPT B+GAN | ✗ | 部分 | 23.38 | 0.713 | **0.204** | 7.45 |
| ERUPT B+SD | ✗ | 部分 | 21.06 | 0.637 | 0.234 | **6.89** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | 说明 |
|------|-------|-------|------|
| ERUPT B (baseline) | 23.85 | 0.718 | 完整模型 |
| 单目标训练 | 23.43 | 0.700 | 多目标复用场景表示有收益 |
| Patch RUST | 23.20 | 0.690 | 简单 token 混合不如交替注意力 |
| ERUPT B+LORA | 24.69 | 0.749 | LORA 微调 backbone 提升显著 |
| ERUPT L+LORA | 25.26 | 0.769 | 增大模型进一步提升 |
| 5% known poses | 23.55 | 0.706 | 仅需极少真值位姿 |

### 关键发现
- LORA 微调 DINOv2 backbone 贡献最大（PSNR +0.84），说明即使是强大的基础模型也需要针对 3D 场景合成任务做适配。
- 仅使用 5% 的真值目标位姿，性能下降仅 0.3 PSNR，证明了位姿切换策略的鲁棒性。
- Patch 解码器使训练速度比 RUST 快 5 倍（224 分辨率），VRAM 降低 64 倍，RUST 在 224 分辨率 48GB GPU 上直接 OOM。
- GAN 和 SD 微调大幅改善感知质量（FID 从 ~100 降到 7-9），但 SD 的多视角一致性仍有问题。

## 亮点与洞察
- **Patch 光线查询**是最核心的效率创新——每次解码 64 像素而非 1 像素，几乎不损失质量但效率提升一个数量级，这个设计思路可以迁移到任何基于光线查询的场景表示方法。
- **随机位姿切换训练**非常巧妙——以 1/3 概率在三种模式间切换，让模型同时学会使用真值位姿和潜在位姿，推理时可以选择性使用。这种"训练时混合多种输入模式"的策略可以迁移到其他多模态任务。
- 引入 MSVS-1M 真实世界数据集（Mapillary 街景 100 万图像）填补了该领域缺乏大规模真实数据集的空白。

## 局限与展望
- $L_2$ 损失在高不确定性场景下产生模糊输出，GAN/SD 微调虽有改善但引入新的伪影（GAN）或多视角不一致（SD）。
- SD 渲染速度仅约 1fps（512 分辨率），无法实时使用。
- 每帧独立渲染，缺乏帧间一致性，可考虑引入多视角扩散模型（如 DORSal 的方式）。
- 在真实世界数据集 MSVS-1M 上性能明显下降（PSNR 20.64 vs MSN 24.69），说明模型在复杂真实场景中的泛化能力仍需提升。

## 相关工作与启发
- **vs SRT**: SRT 需要所有图像精确位姿，ERUPT 不需要输入位姿且仅需少量目标位姿。ERUPT 通过交替注意力和相机 token 自然解决了位姿问题。
- **vs RUST**: RUST 推理时需要目标图像的一半，无法直接控制相机。ERUPT 的位姿切换机制同时兼顾无位姿训练和推理时相机控制。
- **vs DORSal**: DORSal 用多视角扩散保证一致性，FID 更好但 PSNR 差很多。ERUPT+SD 在 FID 上超过 DORSal 且保持更高 PSNR。

## 评分
- 新颖性: ⭐⭐⭐⭐ patch 解码和位姿切换训练有较强创新性，但整体框架仍是编码器-解码器范式
- 实验充分度: ⭐⭐⭐⭐ MSN 和真实数据集的实验全面，消融充分，计算效率对比详实
- 写作质量: ⭐⭐⭐⭐ 结构清晰，技术细节完整，但部分公式和符号较多
- 价值: ⭐⭐⭐⭐ patch 解码的效率提升和真实数据集贡献有实际价值，位姿切换训练思路有启发性

<!-- RELATED:START -->

## 相关论文

- [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](../../ICCV2025/3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [VGGT: Visual Geometry Grounded Transformer](vggt_visual_geometry_grounded_transformer.md)

<!-- RELATED:END -->
