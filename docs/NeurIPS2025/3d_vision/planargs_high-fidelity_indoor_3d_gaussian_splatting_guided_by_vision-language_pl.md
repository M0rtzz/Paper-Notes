---
title: >-
  [论文解读] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors
description: >-
  [NeurIPS 2025][3D视觉][3D高斯溅射] 利用视觉语言基础模型（GroundedSAM）检测平面区域，结合DUSt3R多视图深度先验，通过共面约束和几何先验监督优化3DGS，实现室内场景的高保真表面重建。 室内场景的3D重建在AR/VR和机器人领域有广泛需求，但室内环境的核心挑战是大面积低纹理区域：（如墙壁、…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "室内重建"
  - "平面先验"
  - "视觉语言模型"
  - "DUSt3R"
---

# PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors

**会议**: NeurIPS 2025  
**arXiv**: [2510.23930](https://arxiv.org/abs/2510.23930)  
**代码**: [项目主页](https://planargs.github.io)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 室内重建, 平面先验, 视觉语言模型, DUSt3R

## 一句话总结

利用视觉语言基础模型（GroundedSAM）检测平面区域，结合DUSt3R多视图深度先验，通过共面约束和几何先验监督优化3DGS，实现室内场景的高保真表面重建。

## 研究背景与动机

室内场景的3D重建在AR/VR和机器人领域有广泛需求，但室内环境的核心挑战是**大面积低纹理区域**（如墙壁、地板、天花板）。3DGS依赖光度损失训练，在这些区域产生严重的几何歧义：
- 现有方法（如PGSR）使用多视图几何一致性约束，只能获得局部平滑；
- 引入单目深度/法线先验（如DN-Splatter）存在局部错位问题；
- 这些方法都只做**局部约束**，无法保证全局的平面平坦性——产生"局部平滑但全局弯曲"的典型问题。

理想解决方案是**显式检测平面区域并强制平面几何**。但传统的平面检测器（如PlaneRCNN等专用小模型）泛化能力差、分割精度低。

本文的核心idea：
1. 利用**视觉语言基础模型**（GroundedSAM）通过文本提示（如"wall"、"floor"）检测平面区域——利用基础模型的泛化能力+文本提示的灵活性
2. 通过**跨视图融合和几何验证**修正检测结果中的错误
3. 使用共面约束**整体性地**约束平面区域的高斯分布

## 方法详解

### 整体框架

输入多视图图像 → DUSt3R提取多视图深度/法线先验 → GroundedSAM+LP3流水线生成平面先验 → 3DGS优化（平面先验监督+几何先验监督）→ TSDF融合提取网格

### 关键设计

1. **LP3：语言提示的平面先验流水线**

    - **跨视图融合**：单张图像中大平面可能超出视野被漏检。LP3利用先验深度将邻近帧的平面mask反投射到当前帧，补充遗漏的检测框
    - **几何验证**：GroundedSAM有时会把两个垂直墙面合并为一个mask。LP3先从深度先验计算法线图$N_{dr}$，然后：(1) 用K-means聚类法线图分离非平行平面；(2) 用平面距离图$\delta_r = P \cdot N_{dr}$的异常值检测几何边缘，分离平行但不同的平面
    - 设计动机：单纯依赖VL模型的分割结果有两类错误——漏检（视野限制）和合并（两个平面被当作一个），LP3用多视图互补+几何验证修正

2. **平面先验监督**

    - **平面引导初始化**：SfM在低纹理区域产生的点云稀疏，用深度先验反投射平面区域像素为稠密3D点补充初始化
    - **高斯平面化**：最小化每个高斯最小缩放因子$L_s = \|min(s_1,s_2,s_3)\|_1$，使高斯趋向扁平面片
    - **共面约束**（核心）：将渲染深度图反投射为3D点，对每个平面区域$p_m$用最小二乘拟合平面参数$A_m$（$A_m^T P = 1$），再从拟合平面计算平面深度$D_p(p) = (A_m^T K^{-1} \tilde{p})^{-1}$，约束渲染深度$\hat{D}$与平面深度$D_p$一致：$L_p = \frac{1}{N_p}\sum \|D_p - \hat{D}\|_1$

3. **几何先验监督**

    - **先验深度约束**：将DUSt3R的深度与SfM稀疏深度做尺度对齐（scale-shift），在低纹理区域约束渲染深度
    - **先验法线约束**：在平面区域约束渲染表面法线与DUSt3R的先验法线对齐
    - **深度-法线一致性**：在低纹理区域约束渲染的GS-法线与深度导出的表面法线一致

### 损失函数 / 训练策略

$$L_{total} = L_{RGB} + L_s + \lambda_1 L_{dn} + \lambda_2 L_p + \lambda_3 L_{rd} + \lambda_4 L_{rn}$$

$\lambda_1=0.05, \lambda_2=0.5, \lambda_3=0.05, \lambda_4=0.2$。训练30K步，在RTX 3090上1小时内完成。

## 实验关键数据

### 主实验：MuSHRoom数据集（5个复杂真实场景）

| 方法 | Acc↓ | Comp↓ | CD↓ | F1↑ | NC↑ | PSNR↑ |
|------|------|-------|-----|-----|-----|-------|
| 3DGS | 12.01 | 11.85 | 11.92 | 38.53 | 62.00 | 25.79 |
| DN-Splatter | 6.25 | 5.29 | 5.77 | 61.86 | 77.13 | 24.80 |
| **PlanarGS** | **3.95** | **5.02** | **4.49** | **77.14** | **83.35** | **26.42** |

### ScanNet++和Replica数据集

| 方法 | ScanNet++ CD↓ | ScanNet++ F1↑ | Replica CD↓ | Replica F1↑ |
|------|--------------|--------------|-------------|-------------|
| DUSt3R | 8.17 | 38.17 | 7.35 | 44.89 |
| PGSR | 7.22 | 53.73 | 8.56 | 62.98 |
| DN-Splatter | 4.16 | 75.86 | 5.60 | 68.12 |
| **PlanarGS** | **3.66** | **82.78** | **4.13** | **81.90** |

### 消融实验（MuSHRoom coffee room）

| 配置 | Acc↓ | F1↑ | 说明 |
|------|------|-----|------|
| ZeroPlane做平面先验 | - | 较低 | 专用小模型泛化差，错误先验引入噪声 |
| GroundedSAM（无LP3） | - | 中等 | 未经几何验证的先验不够准确 |
| 无共面约束 | - | 降低 | 大平面区域出现表面凹凸 |
| 无几何先验 | - | 降低 | 缺少尺度监督，平面整体倾斜 |
| 无深度-法线一致性 | - | 降低 | 表面粗糙 |
| 完整PlanarGS | 最佳 | 最佳 | 各模块互补 |

### 关键发现

- 单独的共面约束贡献最大（尤其在无几何先验时），但缺少几何先验会导致大平面**整体倾斜偏移**
- LP3流水线的跨视图融合和几何验证对平面先验质量至关重要——ZeroPlane和裸GroundedSAM都不够用
- 训练时间与其他3DGS方法相当（<1小时），具有实用性

## 亮点与洞察

1. **基础模型赋能传统任务**的范例——将VL模型的泛化能力用于几何重建的先验提取
2. 文本提示的灵活性：教室加"blackboard"即可检测黑板，无需重训练
3. 共面约束的"整体性"约束思路值得借鉴——不是逐像素约束法线，而是拟合整个平面后约束深度

## 局限与展望

- 依赖DUSt3R和GroundedSAM两个大模型做预处理，部署复杂度高
- 文本提示需要人工设定（虽然通用性较好）
- 非平面区域仍然依赖传统光度损失，对复杂非平面物体（如植物、布料）改进有限

## 相关工作与启发

- LP3流水线（VL检测+跨视图融合+几何验证）可迁移至其他需要语义先验的3D任务
- 共面约束的最小二乘平面拟合→反向约束深度的思路简洁有效

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将VL基础模型的平面先验引入3DGS，LP3流水线设计精巧
- 实验充分度: ⭐⭐⭐⭐ 三个室内数据集+消融，覆盖合成和真实场景
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐⭐ 室内重建的实用方法，大幅领先SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](langsplatv2_high-dimensional_3d_language_gaussian_splatting_with_450_fps.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](../../ICCV2025/3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
