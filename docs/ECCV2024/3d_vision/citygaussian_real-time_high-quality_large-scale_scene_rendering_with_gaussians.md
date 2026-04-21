---
title: >-
  [论文解读] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians
description: >-
  [ECCV2024][3D视觉][3D Gaussian Splatting] 提出 CityGaussian (CityGS)，通过分治训练策略和 block-wise Level-of-Detail 机制，首次实现了城市级大规模场景（>1.5 km²）的高质量 3D Gaussian Splatting 训练与跨尺度实时渲染。
tags:
  - ECCV2024
  - 3D视觉
  - 3D Gaussian Splatting
  - Large-Scale Scene Reconstruction
  - Level-of-Detail
  - Divide-and-Conquer
  - novel view synthesis
---

# CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians

**会议**: ECCV2024  
**arXiv**: [2404.01133](https://arxiv.org/abs/2404.01133)  
**代码**: [dekuliutesla/citygs](https://github.com/DekuLiuTesworworla/citygs)  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, Large-Scale Scene Reconstruction, Level-of-Detail, Divide-and-Conquer, novel view synthesis

## 一句话总结

提出 CityGaussian (CityGS)，通过分治训练策略和 block-wise Level-of-Detail 机制，首次实现了城市级大规模场景（>1.5 km²）的高质量 3D Gaussian Splatting 训练与跨尺度实时渲染。

## 背景与动机

3D Gaussian Splatting (3DGS) 凭借显式高斯原语和高效光栅化，在小场景新视角合成中取得了实时高质量效果。但将其直接应用于大规模场景（如城市级区域）面临两个核心瓶颈：

1. **训练显存溢出**：覆盖 1.5 km² 的城市区域需要超过 2000 万个高斯原语，即便在 40GB A100 上也会 OOM。单张 24GB RTX3090 在高斯数超过 1100 万时即崩溃。
2. **渲染速度退化**：高斯数量从百万级增至千万级后，深度排序成为瓶颈。MatrixCity 场景（2300 万高斯）仅有 21 FPS，无法满足实时需求。

已有的 NeRF 类大规模方法（Block-NeRF、Mega-NeRF、Switch-NeRF）虽然采用分治策略，但基于隐式表征，细节保真度不足且渲染速度慢。VastGaussian 虽将 3DGS 扩展至大场景，但仍未解决实时渲染问题。

## 核心问题

如何在有限显存下高效训练城市级大规模 3DGS，并在跨尺度（从近景到高空鸟瞰）切换时始终保持实时渲染？

## 方法详解

### 1. 全局高斯先验生成

首先用所有训练图像对 COLMAP 点云进行 30,000 次迭代的标准 3DGS 训练，得到粗糙的全局高斯先验 $\mathbf{G}_K$。该先验提供：

- 全局几何分布感知，避免后续分块训练出现几何不准确的 floater
- 更干净的渲染图像，便于后续数据划分

### 2. 分治训练策略

**空间收缩与分块**：大规模场景通常无界，直接均匀网格划分会导致许多空块、负载不均。方法：

- 定义前景区域 $[\mathbf{p}_{min}, \mathbf{p}_{max}]$，对高斯位置归一化到 $[-1, 1]$
- 对超出前景的背景区域使用非线性 contraction 映射（L∞ 范数），将无界空间压缩到 $[-2, 2]$ 立方体
- 在收缩空间中均匀网格划分，实现更均衡的高斯分配

**自适应数据划分**：不是简单按距离选训练视角，而是基于 SSIM 损失判断视角对该 block 的贡献：

- **原则一（公式 3）**：渲染包含/不包含该 block 高斯的图像，若 SSIM 差异 > 阈值 ε，则该视角对 block 有显著贡献，保留
- **原则二（公式 4）**：若相机位置落入 block 范围内，直接保留（防止 block 边缘处的 artifact）
- 两条原则取并集作为最终数据分配

**并行微调与融合**：

- 每个 block 以全局先验为初始化，独立训练 30,000 次迭代
- 使用 L1 + SSIM 组合损失
- 训练完成后按空间边界裁剪高斯，直接拼接即可无缝融合（全局先验消除了块间干扰）

### 3. Block-wise Level-of-Detail (LoD)

**多级细节生成**：使用 LightGaussian 压缩策略对融合后的高斯生成 3 个细节等级：

- LoD 2（50% 压缩）：最精细
- LoD 1（34% 压缩）：中等
- LoD 0（25% 压缩）：最粗糙

**Block-wise 可见性判断与等级选择**：

- 以训练时划分的 block 为单位，计算 block 八个角点与视锥体的交集（IoU），判定 block 是否可见
- 使用 MAD（中位绝对偏差）算法去除 floater 对 block 包围盒的干扰，得到更紧致的边界
- 根据 block 八角点到相机中心的最小距离决定细节等级：0-200m 用 LoD 2，200-400m 用 LoD 1，>400m 用 LoD 0
- 同一 block 内所有高斯共享相同 LoD，避免逐点距离计算

**融合渲染**：不同 LoD 的高斯直接拼接送入光栅化器，几乎无可见的不连续性。

## 实验关键数据

### 渲染质量（无 LoD 版 vs SOTA）

| 方法 | MatrixCity PSNR↑ | MatrixCity SSIM↑ | Rubble PSNR↑ | Building PSNR↑ |
|------|:-:|:-:|:-:|:-:|
| Mega-NeRF | - | - | 24.06 | 20.93 |
| Switch-NeRF | - | - | 24.31 | 21.54 |
| 3DGS† | 23.67 | 0.735 | 25.47 | 20.46 |
| **CityGS** | **27.46** | **0.865** | **25.77** | **21.55** |

- 在 MatrixCity（2.7 km²合成城市）上 PSNR 提升 **+3.79 dB**，SSIM 提升 +0.13
- 首次成功重建完整 MatrixCity（相机海拔 150m-500m），此前方法均无法稳定训练

### LoD 效果

| 模式 | SSIM | PSNR | FPS |
|------|:-:|:-:|:-:|
| 无 LoD | 0.865 | 27.46 | 21.6 |
| 仅 LoD 2 | 0.863 | 27.54 | 45.6 |
| 仅 LoD 0 | 0.825 | 26.57 | 69.4 |
| **LoD（混合）** | **0.855** | **27.32** | **53.7** |

- LoD 策略将 FPS 从 21.6 提升至 53.7（**2.5× 加速**），PSNR 仅降 0.14 dB
- 在极端高空视角下，唯有混合 LoD 能在全部高度保持最低 FPS > 25 的实时性

### 训练策略消融

| 配置 | PSNR | SSIM | 高斯数量 |
|------|:-:|:-:|:-:|
| baseline（邻近相机选择） | 23.98 | 0.779 | 12.2M |
| + 全局先验 | 25.01 | 0.801 | 15.4M |
| CityGS（完整策略） | **25.77** | **0.813** | 9.7M |

- 全局先验显著提升质量（+1.03 dB）
- 自适应数据划分（公式 3+4）在少用 37% 高斯的情况下进一步提升 +0.76 dB

### LoD 策略消融

- Block-wise 选择 vs 逐点选择：FPS 53.7 vs 30.3，质量相当但速度快 77%
- 距离区间影响：[0,200],[200,400],[400,∞] 取得质量与速度的最佳平衡

## 亮点

1. **全局先验 + 分治微调**：先粗后精的两阶段策略优雅解决了块间干扰和 floater 问题，确保无缝融合
2. **基于 SSIM 的自适应数据划分**：相比简单空间距离选择，能精确筛选出对 block 有实质贡献的训练视角，减少无关数据干扰的同时降低高斯消耗
3. **Block-wise LoD**：以 block 而非逐点为单位选择细节等级，避免了逐点距离计算的开销，实现了跨尺度一致的实时渲染
4. **MAD 去 floater**：用中位绝对偏差估计 block 包围盒，有效排除 floater 对视锥体裁剪的干扰

## 局限与展望

1. **静态场景假设**：无法处理动态物体（行人、车辆），限制了实际城市场景的适用性
2. **混合视角训练退化**：论文承认同时使用航拍和街景视角训练反而降低性能，该问题尚未解决
3. **压缩策略依赖外部方法**：LoD 细节生成直接借用 LightGaussian，未针对大规模场景特点设计专用压缩
4. **Block 边界处理**：虽然全局先验缓解了块间不连续，但在极端情况下仍可能存在接缝
5. **训练开销**：需要先训全局先验再逐块微调，总训练时间较长

## 与相关工作的对比

| 方法 | 表征 | 分治 | LoD | 实时 | 大规模质量 |
|------|:----:|:----:|:---:|:----:|:----------:|
| Mega-NeRF | 隐式 MLP | ✓ | ✗ | ✗ | 一般 |
| Switch-NeRF | 隐式 MLP | ✓（可学习） | ✗ | ✗ | 一般 |
| BungeeNeRF | 隐式 MLP | ✗ | ✓（渐进式） | ✗ | 一般 |
| VastGaussian | 3DGS | ✓ | ✗ | ✗ | 较好 |
| **CityGS** | **3DGS** | **✓** | **✓** | **✓** | **SOTA** |

- 相比 VastGaussian：CityGS 额外引入 LoD 实现实时渲染，且通过全局先验避免了外观变化处理的需求
- 相比 NeRF 类方法：渲染质量大幅领先（PSNR +1.5~3.8 dB），且具备实时能力

## 启发与关联

1. **分治 + 全局先验**是处理大规模显式表征的通用范式，可推广至大规模 mesh 重建、点云补全等任务
2. Block-wise LoD 的思想可与 hierarchical Gaussian（如 octree-GS）结合，实现更细粒度的多尺度表示
3. 动态场景扩展是明确的后续方向——可结合 4D Gaussian / deformable Gaussian 处理时变内容
4. 自适应数据划分策略（基于 SSIM 贡献度）对其他需要数据高效利用的 3DGS 变体也有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ （分治训练 + block-wise LoD 的组合方案系统性强，各模块设计合理）
- 实验充分度: ⭐⭐⭐⭐⭐ （5 个不同规模场景、完整消融、多角度 FPS 分析、street view 泛化验证）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富，动机阐述充分）
- 价值: ⭐⭐⭐⭐ （首次实现城市级 3DGS 实时渲染，工程与学术价值兼具）

<!-- RELATED:START -->

## 相关论文

- [VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](../../CVPR2025/3d_vision/sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](../../NeurIPS2025/3d_vision/lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](../../CVPR2025/3d_vision/matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)

<!-- RELATED:END -->
