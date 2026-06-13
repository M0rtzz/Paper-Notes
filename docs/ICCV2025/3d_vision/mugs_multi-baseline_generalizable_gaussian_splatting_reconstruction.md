---
title: >-
  [论文解读] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 本文提出 MuGS，首个面向多基线设定的泛化 3D 高斯溅射方法，通过融合多视角立体（MVS）和单目深度估计（MDE）特征，并设计投影-采样深度一致性网络，实现在小基线和大基线场景下的 SOTA 新视角合成。 领域现状：3D高斯溅射（3D-GS）因高效实时渲染而成为新视角合…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "多基线泛化"
  - "单目深度估计"
  - "多视角立体"
  - "新视角合成"
---

# MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2508.04297](https://arxiv.org/abs/2508.04297)  
**代码**: [https://github.com/EuclidLou/MuGS](https://github.com/EuclidLou/MuGS)  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D高斯溅射、多基线泛化、单目深度估计、多视角立体、新视角合成

## 一句话总结
本文提出 MuGS，首个面向多基线设定的泛化 3D 高斯溅射方法，通过融合多视角立体（MVS）和单目深度估计（MDE）特征，并设计投影-采样深度一致性网络，实现在小基线和大基线场景下的 SOTA 新视角合成。

## 研究背景与动机

**领域现状**：3D高斯溅射（3D-GS）因高效实时渲染而成为新视角合成的主流方法，但需要逐场景优化。泛化方法通过数据驱动实现对未见场景的前馈推理，但现有方法要么专精小基线（图像重叠大）要么专精大基线（图像重叠小）。

**现有痛点**：小基线方法在大基线数据上因遮挡和重叠不足导致深度误差大；大基线方法在小基线数据上因缺乏匹配线索导致深度估计不准。核心瓶颈在于深度估计策略随基线而变。

**核心矛盾**：小基线依赖 MVS 匹配，大基线依赖单目几何先验，两种策略本质不同，难以用一个模型统一。

**本文目标**：构建首个能同时处理多种基线设定的泛化高斯溅射方法。

**切入角度**：准确的深度引导可以统一解决两种基线设定的挑战。MVS 在重叠充足时精度高但在挑战区域易错，MDE 提供更鲁棒平滑的深度但缺乏多视角一致性，两者互补。

**核心 idea**：融合 MVS 的匹配深度（projected depth）和 MDE 的预测深度（sampled depth），通过3D U-Net 计算一致性来构建精细概率体，指导深度回归。

## 方法详解

### 整体框架
输入稀疏多视角图像，提取图像特征并用预训练单目深度模型获取辅助深度。构建 MVS 代价体积时同时计算投影深度和采样深度的一致性，用3D U-Net 精炼概率体积，回归深度和高斯参数进行新视角渲染。

### 关键设计

1. **投影-采样深度一致性网络**:

    - 功能：融合 MVS 匹配信息和单目深度先验
    - 核心思路：对每个深度候选点计算两种深度：projected depth（基于多视角特征变换的空间位置）和 sampled depth（基于单目深度模型的预期深度）。3D U-Net 计算两种深度的一致性分数，用于精炼代价体积。一致性高的深度候选获得更高概率。
    - 设计动机：MVS 在重叠区域精确但在遮挡区域失败，MDE 到处都较鲁棒但有尺度歧义，一致性网络能自适应选择可靠信息。

2. **精细概率体积引导**:

    - 功能：将一致性信息转化为深度回归的引导信号
    - 核心思路：一致性分数作为注意力网络的 query，通过轻量注意力机制精炼深度概率体积，使概率集中在真实表面附近。MLP 网络利用每个源视角采样的特征和颜色进行高斯参数回归。
    - 设计动机：优先选择表面附近的深度候选，减少错误深度处采样的噪声特征对渲染质量的影响。

3. **参考视角损失**:

    - 功能：提供上下文监督以改善几何一致性
    - 核心思路：除了对目标视角渲染结果的监督外，额外渲染参考视角（输入视角）的图像并与原始图像计算损失，利用已知视角的精确对应关系来更有效地学习几何。
    - 设计动机：稀疏输入下目标视角监督信号有限，参考视角损失提供额外的几何约束。

### 损失函数 / 训练策略
渲染损失（L1 + SSIM）应用于目标视角和参考视角。使用 3D 高斯表示加速训练和推理。

## 实验关键数据

### 主实验

| 数据集 | 基线类型 | 本文 PSNR | 之前SOTA PSNR | 提升 |
|--------|---------|-----------|-------------|------|
| DTU (3-view) | 小基线 | SOTA | 次优 | 显著 |
| RealEstate10K | 大基线 | SOTA | 次优 | 显著 |
| LLFF (zero-shot) | 小基线 | 有竞争力 | - | 泛化验证 |
| Mip-NeRF 360 (zero-shot) | 大基线 | 有竞争力 | - | 泛化验证 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 MVS | 基线 | 大基线下深度误差大 |
| + MDE 特征 | 提升 | 单目深度辅助 |
| + 深度一致性网络 | 进一步提升 | 智能融合两种深度 |
| + 参考视角损失 | 最佳 | 额外几何监督 |

### 关键发现
- 首个在小基线和大基线上同时达到 SOTA 的泛化高斯方法
- 深度一致性网络是核心贡献——移除后两种基线下性能均显著下降
- 零样本泛化能力良好，证明方法学到了通用的多基线深度推理能力

## 亮点与洞察
- **MVS + MDE 互补融合**：用一致性网络智能选择可靠信息源，而非简单拼接特征，可迁移到其他需要融合多种深度估计的任务。
- **统一多基线**：首次在泛化高斯溅射中解决了基线泛化问题，有很强的实用价值。

## 局限与展望
- 需要预训练单目深度模型作为辅助，增加了系统复杂度
- 极端大基线（完全无重叠）下效果仍受限
- 可以探索自适应基线检测来选择不同的融合策略

## 相关工作与启发
- **vs MuRF**: 依赖 MVS 密度体积，遮挡下密度分散；本文从深度精度角度统一解决
- **vs DepthSplat**: 通过特征级拼接融合 MVS 和 MDE，本文深入建模两种深度线索的关系

## 评分
- 新颖性: ⭐⭐⭐⭐ 多基线泛化是新问题，一致性网络设计有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估+零样本验证
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰
- 价值: ⭐⭐⭐⭐ 对实际应用价值大，多基线是实际场景常见问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs_lm_faster_gaussian_splatting_optimization_with_levenberg_marquardt.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)
- [\[ECCV 2024\] MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](../../ECCV2024/3d_vision/mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)

</div>

<!-- RELATED:END -->
