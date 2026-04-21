---
title: >-
  [论文解读] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction
description: >-
  [ICCV 2025][3D视觉][新视角合成] MuGS 是首个支持多基线设置（小基线到大基线）的泛化3D高斯泼溅方法，通过融合 MVS 和 MDE 特征、投影-采样深度一致性网络和参考视图损失，在不同基线数据集上均达到 SOTA。
tags:
  - ICCV 2025
  - 3D视觉
  - 新视角合成
  - 3D高斯泼溅
  - 多基线
  - 泛化重建
  - 深度融合
  - 多视图立体
  - 单目深度估计
---

# MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2508.04297](https://arxiv.org/abs/2508.04297)  
**代码**: [GitHub](https://github.com/EuclidLou/MuGS)  
**领域**: 3d_vision  
**关键词**: 新视角合成, 3D高斯泼溅, 多基线, 泛化重建, 深度融合, 多视图立体, 单目深度估计

## 一句话总结

MuGS 是首个支持多基线设置（小基线到大基线）的泛化3D高斯泼溅方法，通过融合 MVS 和 MDE 特征、投影-采样深度一致性网络和参考视图损失，在不同基线数据集上均达到 SOTA。

## 研究背景与动机

3DGS 以高质量实时渲染著称但需逐场景优化。泛化方法通过数据驱动避免此问题，但面临基线适配挑战：

**基线特化**：现有方法专注小基线或大基线，无法跨基线泛化

**小基线→大基线**：因遮挡和重叠不足导致深度误差和变形

**大基线→小基线**：缺乏匹配线索导致深度不准和渲染模糊

**核心洞察**：精确深度引导可统一解决大/小基线方法的共性挑战

## 方法详解

### 整体框架

核心思路："从深度精度出发统一多基线"

### 关键设计一：MVS + MDE 特征融合

- **MVS 特征**：代价体积，擅长有足够重叠时的精确深度
- **MDE 特征**：预训练单目深度模型（Depth Anything），鲁棒但有尺度歧义
- **互补性**：MDE 鲁棒性弥补 MVS 在挑战区域的不足

### 关键设计二：投影-采样深度一致性网络

核心创新，三步融合：
1. **投影深度**：每个深度候选点的空间位置深度（MVS 认为的深度）
2. **采样深度**：从 MDE 特征中采样期望深度值（MDE 认为的深度）
3. **一致性建模**：3D U-Net 分析两者一致性
4. **概率体积精细化**：一致性信息作为查询，轻量级注意力网络精细化深度概率体积

两者一致 → 深度候选可信；不一致 → 需调整。优先选择靠近表面的候选。

### 关键设计三：参考视图损失（Reference-View Loss）

- 传统方法仅在目标新视图上计算损失
- MuGS 额外在参考源视图上计算重建损失
- 提供上下文监督，更有效学习几何对应

### 3D高斯表示

从精细化深度图和特征回归高斯参数（位置、协方差、SH系数、不透明度）。

## 实验关键数据

### 跨基线性能

首个在大/小基线同时 SOTA 的单一模型：

| 基线设置 | 数据集 | 表现 |
|---------|--------|------|
| 小基线 | RealEstate10K | SOTA |
| 大基线 | DTU | SOTA |
| 零样本 | LLFF | 有竞争力 |
| 零样本 | Mip-NeRF 360 | 有竞争力 |

### 关键发现

1. 首个在大/小基线**同时**超越所有方法的统一模型
2. LLFF 和 Mip-NeRF 360 上有竞争力的零样本泛化
3. 投影-采样一致性显著改善挑战区域深度估计
4. 参考视图损失明显提升几何对应学习

### 消融关键结论

- MVS + MDE vs 仅 MVS：大基线场景提升明显
- 投影-采样一致性 vs 简单拼接（DepthSplat）：一致性建模更有效
- 参考视图损失：稳定提升各基线设置

## 亮点与洞察

1. **多基线统一**：首次将"多基线"从立体深度延伸到泛化NVS，实际应用中用户不会切换模型
2. **投影-采样一致性**：相比 DepthSplat 简单拼接，显式建模两个深度线索的一致性更有效
3. **参考视图损失简洁有效**：在已有源图像上加重建监督，提供"已知答案"的额外约束
4. **对 DepthSplat 的改进**：从特征级拼接到一致性建模的深层融合
5. **从根本问题出发**：回到"深度精度"这个根本问题而非渲染端复杂设计

## 局限性

1. 依赖预训练 MDE 模型质量
2. 3D U-Net 增加计算开销
3. 缓存缺少具体数值表格（PSNR/SSIM/LPIPS）
4. 两视图输入限制几何信息量
5. 极端大基线（完全无重叠）退化为单目深度

## 相关工作

- **多基线立体**：传统多基线深度 → MuRF（NeRF级）→ MuGS（3DGS级）
- **泛化高斯**：pixelSplat/MVSplat（小基线）, DepthSplat（大基线）→ MuGS 统一
- **MVS**：MVSNet 代价体积 → MuGS 融合代价体积与 MDE
- **MDE**：Depth Anything → 作为先验输入

## 评分

- **新颖性**：8/10 — 多基线泛化NVS是有价值的问题定义
- **技术深度**：8/10 — 投影-采样-一致性三步设计扎实
- **实验充分性**：6/10 — 缓存缺完整表格但评估数据集全面
- **实用性**：8/10 — 单一模型适应多种基线，无需逐场景优化
- **总评**：7.5/10

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](../../ECCV2024/3d_vision/mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)
- [BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)

<!-- RELATED:END -->
