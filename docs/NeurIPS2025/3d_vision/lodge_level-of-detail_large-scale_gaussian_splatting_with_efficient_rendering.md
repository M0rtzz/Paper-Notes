---
title: >-
  [论文解读] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering
description: >-
  [NeurIPS 2025][3D视觉][Gaussian Splatting] 提出 LODGE，通过层次化 LOD（Level-of-Detail）策略对 3D Gaussian Splatting 进行多尺度管理，根据相机距离动态选择合适粒度的 Gaussian 表示，实现大规模场景的高质量实时渲染。
tags:
  - NeurIPS 2025
  - 3D视觉
  - Gaussian Splatting
  - Level-of-Detail
  - 大规模场景
  - 高效渲染
  - LOD
---

# LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering

**会议**: NeurIPS 2025  
**arXiv**: [2505.23158](https://arxiv.org/abs/2505.23158)  
**代码**: 有  
**领域**: 3D 视觉 / 大规模场景渲染  
**关键词**: Gaussian Splatting, Level-of-Detail, 大规模场景, 高效渲染, LOD

## 一句话总结

提出 LODGE，通过层次化 LOD（Level-of-Detail）策略对 3D Gaussian Splatting 进行多尺度管理，根据相机距离动态选择合适粒度的 Gaussian 表示，实现大规模场景的高质量实时渲染。

## 研究背景与动机

**领域现状**：3DGS 在小场景中表现出色，但扩展到大规模场景（城市级别）时面临 Gaussian 数量爆炸问题。

**现有痛点**：百万级 Gaussian 导致 (1) GPU 内存溢出；(2) 光栅化开销巨大；(3) 远处细节浪费计算资源。

**核心矛盾**：高质量需要密集 Gaussian vs 效率需要稀疏表示。

**切入角度**：传统图形学中的 LOD 技术——远处物体用低分辨率表示，近处用高分辨率。

**核心 idea**：构建 Gaussian 的多层 LOD 金字塔，渲染时按距离选择合适层级。

## 方法详解

### 整体框架

场景分块 → 每块构建 LOD 金字塔（从精细到粗糙）→ 渲染时按距离选择层级 → 混合光栅化。

### 关键设计

1. **LOD 金字塔构建**

    - 功能：将精细 Gaussian 逐级合并为更粗糙的表示
    - 核心思路：空间聚类 + 属性融合——邻近 Gaussian 合并为更大的 Gaussian，颜色和不透明度加权平均
    - 设计动机：远处观察时小 Gaussian 被像素覆盖，合并后效果等价但数量减少

2. **距离自适应选择**

    - 功能：根据相机到每个块的距离选择合适的 LOD 层级
    - 核心思路：$\text{LOD}(d) = \lfloor \log_2(d / d_{min}) \rfloor$，距离翻倍则粗一级
    - 设计动机：人眼对远处细节不敏感，Screen Space Error 控制视觉质量

3. **跨层级平滑过渡**

    - 功能：在 LOD 层级切换边界处避免突变（popping artifact）
    - 核心思路：在边界区域混合相邻两级的 Gaussian，用线性插值权重
    - 设计动机：突然切换会导致可见的闪烁

### 损失函数 / 训练策略

各层级独立训练：$\mathcal{L} = \lambda_1 \mathcal{L}_{photometric} + \lambda_2 \mathcal{L}_{SSIM}$。从最精细层开始训练，逐级合并构建粗糙层。

## 实验关键数据

### 主实验

| 方法 | Mill19 PSNR↑ | 渲染 FPS↑ | GPU 内存↓ | Gaussian 数量↓ |
|------|------------|---------|---------|-------------|
| 3DGS | 26.8 | 15 | 24GB | 45M |
| Mega-NeRF | 25.1 | 0.5 | 8GB | N/A |
| **LODGE** | **26.5** | **45** | **8GB** | **12M (渲染时)** |

### 消融实验

| 配置 | PSNR | FPS | 说明 |
|------|------|-----|------|
| 单层级 (最精细) | 26.8 | 15 | 内存爆 |
| LOD 无平滑过渡 | 26.1 | 42 | popping artifact |
| **LOD + 平滑过渡** | **26.5** | **45** | **最优** |

### 关键发现

- LODGE 渲染 FPS 提升 3 倍（15→45），PSNR 仅降 0.3
- GPU 内存从 24GB 降至 8GB，可在消费级 GPU 上运行大规模场景
- 平滑过渡贡献 +0.4 PSNR，有效消除 popping

## 亮点与洞察

- **经典 LOD 在 3DGS 中的回归**：图形学的老技术在新表示形式下焕发活力。
- **实用性**：使大规模 3DGS 在消费级硬件上可行，降低了部署门槛。

## 局限与展望

- LOD 合并损失信息——某些细节在近距离观察时可能退化
- 动态场景的 LOD 更新策略未涉及
- 分块边界的接缝问题需进一步处理

## 相关工作与启发

- **vs Mega-NeRF**：Mega-NeRF 基于 NeRF 做分块，LODGE 在 3DGS 上更高效
- **vs CityGaussian**：CityGaussian 也做大规模 3DGS，但无 LOD 策略

## 评分
- 新颖性: ⭐⭐⭐⭐ LOD 概念成熟但在 3DGS 中的适配有创新
- 实验充分度: ⭐⭐⭐⭐ 大规模场景验证
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 大规模 3DGS 的实用方案

<!-- RELATED:START -->

## 相关论文

- [ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](../../ICCV2025/3d_vision/resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)
- [VertexRegen: Mesh Generation with Continuous Level of Detail](../../ICCV2025/3d_vision/vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](../../ICCV2025/3d_vision/occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](../../ECCV2024/3d_vision/citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)

<!-- RELATED:END -->
