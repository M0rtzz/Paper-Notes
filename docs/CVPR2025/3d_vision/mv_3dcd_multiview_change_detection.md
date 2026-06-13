---
title: >-
  [论文解读] Multi-View Pose-Agnostic Change Localization with Zero Labels
description: >-
  [CVPR 2025][3D视觉][多视角变化检测] 提出首个无标签、姿态无关的多视角变化检测方法，通过构建变化感知的 3DGS 表示融合多视角变化信息，在 mIoU 上比基线提升 1.7 倍，并能为未见视角生成变化掩码。 领域现状：变化检测通常依赖精确对齐的前后图像对，限制了应用场景；少数方法支持视角不一致但需要标注训练…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多视角变化检测"
  - "3D高斯泼溅"
  - "无标签"
  - "姿态无关"
  - "DINOv2"
---

# Multi-View Pose-Agnostic Change Localization with Zero Labels

**会议**: CVPR 2025  
**arXiv**: [2412.03911](https://arxiv.org/abs/2412.03911)  
**代码**: [https://MV-3DCD.github.io](https://MV-3DCD.github.io)  
**领域**: 3D视觉 / 变化检测  
**关键词**: 多视角变化检测, 3D高斯泼溅, 无标签, 姿态无关, DINOv2

## 一句话总结

提出首个无标签、姿态无关的多视角变化检测方法，通过构建变化感知的 3DGS 表示融合多视角变化信息，在 mIoU 上比基线提升 1.7 倍，并能为未见视角生成变化掩码。

## 研究背景与动机

**领域现状**：变化检测通常依赖精确对齐的前后图像对，限制了应用场景；少数方法支持视角不一致但需要标注训练。

**现有痛点**：现有无标签方法（如 OmniPoseAD、SplatPose）仅在单视角逐图比较，受视角依赖的伪变化（反射、阴影）影响严重。

**核心矛盾**：单视角比较易产生大量假阳性，而多视角融合缺乏有效的 3D 表示。

**本文目标**：利用多视角信息构建 3D 变化表示，抑制视角依赖的假阳性。

**切入角度**：在 3DGS 中嵌入变化通道，利用球谐系数零阶建模视角无关的变化。

**核心 idea**：在推理场景的 3DGS 中学习额外的变化通道（变化幅度 + 变化不透明度），融合多视角特征和结构感知的变化掩码。

## 方法详解

### 整体框架

(1) 用参考场景图像构建 3DGS_ref；(2) 从推理场景视角渲染参考场景图像；(3) 比较渲染与实际图像生成特征+结构感知变化掩码；(4) 将变化信息嵌入推理场景的 Change-3DGS_inf；(5) 从任意视角渲染多视角变化掩码。

### 关键设计

1. **特征与结构感知变化掩码**:

    - 功能：从单视角检测候选变化区域
    - 核心思路：用 DINOv2 提取特征差异得到特征感知掩码 $M_F^k$；用 SSIM 得到结构感知掩码 $M_S^k$；两者逐元素相乘得到组合掩码 $M_{F,S}^k$
    - 设计动机：特征和结构信息互补——DINOv2 捕捉语义变化，SSIM 捕捉像素级变化

2. **3DGS 变化通道嵌入**:

    - 功能：在 3D 表示中编码变化信息，实现多视角融合
    - 核心思路：在每个高斯点上添加变化幅度 $\tilde{c}$ 和变化不透明度 $\tilde{\alpha}$ 两个参数，使用零阶球谐系数建模变化幅度（视角无关），用 L1 + D-SSIM 损失监督变化通道渲染
    - 设计动机：零阶球谐系数使变化建模为视角无关，有效抑制反射/阴影等视角依赖的假阳性

3. **数据增强策略**:

    - 功能：增加用于学习变化通道的训练掩码数量
    - 核心思路：利用 Change-3DGS_inf 从参考场景视角渲染推理场景图像，反向计算变化掩码，与正向掩码合并增强训练数据
    - 设计动机：推理场景图像数量可能很少（最少 5 张），数据增强提升变化通道学习质量

### 损失函数 / 训练策略

变化通道学习使用 L1 + D-SSIM 损失。最终变化掩码通过 alpha 通道过滤未见区域：$M^k = M_{ren}^k \cdot \mathbf{1}(A_{ren}^k \geq 0.5)$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | SplatPose | 提升 |
|--------|------|------|-----------|------|
| MAD-Real | mIoU | 0.132 | 0.077 | 1.7× |
| MAD-Real | F1 | 0.210 | 0.123 | 1.7× |
| ChangeSim | mIoU(C) | 0.407 | - | 1.7× vs CSCDNet |
| PASLCD | 平均 mIoU | 最高 | 次高 | 全场景领先 |

### 关键发现

- 仅需 5 张推理场景图像即可学习有效的变化通道
- 能为推理和参考场景都未见过的新视角生成变化掩码
- 零阶球谐系数比高阶球谐系数效果更好，验证了变化的视角无关假设

## 亮点与洞察

- 首次将变化检测提升到 3D 表示层面，实现真正的多视角融合
- 贡献了包含 10 个真实场景的 PASLCD 数据集
- 方法可作为任何单视角变化检测方法的多视角扩展

## 局限与展望

- 依赖 COLMAP 注册推理场景图像到参考场景
- 对极端外观变化（如完全黑暗）可能导致 COLMAP 注册失败
- 变化通道训练需要为每个场景单独优化

## 评分

- 新颖性：9/10 — 首次在 3DGS 中嵌入变化通道
- 技术深度：8/10 — 零阶球谐系数建模变化的设计有理论支撑
- 实验充分度：8/10 — 三个数据集 + 新贡献数据集
- 写作质量：8/10 — 方法描述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[CVPR 2025\] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)

</div>

<!-- RELATED:END -->
