---
title: >-
  [论文解读] InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation
description: >-
  [ICCV 2025][医学图像][3D高斯溅射] InsideOut 将 3D Gaussian Splatting 从仅建模 RGB 表面扩展到同时建模 X 射线内部结构，通过层次化拟合和 X 射线参考损失实现了 RGB 外观与内部辐射结构的联合表示。
tags:
  - ICCV 2025
  - 医学图像
  - 3D高斯溅射
  - X射线成像
  - 多模态融合
  - 三维重建
  - 无损检测
---

# InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation

**会议**: ICCV 2025  
**arXiv**: [2510.17864](https://arxiv.org/abs/2510.17864)  
**代码**: 无  
**领域**: 医学图像 / 3D视觉  
**关键词**: 3D高斯溅射、X射线成像、多模态融合、三维重建、无损检测

## 一句话总结

InsideOut 将 3D Gaussian Splatting 从仅建模 RGB 表面扩展到同时建模 X 射线内部结构，通过层次化拟合和 X 射线参考损失实现了 RGB 外观与内部辐射结构的联合表示。

## 研究背景与动机

**领域现状**：3D Gaussian Splatting（3DGS）已成为高保真三维场景表示的主流方法，能够从多视角 RGB 图像快速重建高质量的表面外观。然而，现有 3DGS 方法仅关注物体的外表面信息，无法捕获其内部结构。

**现有痛点**：在医学诊断、文物修复和工业制造质量检测等关键应用中，仅有外表面信息远远不够——医生需要看到患者体内的器官和病变，考古学家需要在不破坏文物的情况下了解内部构造，工程师需要检测产品内部缺陷。这些场景都需要将表面外观与内部结构信息统一在一个三维表示中。

**核心矛盾**：RGB 图像和 X 射线图像具有截然不同的数据表示方式——RGB 图像反映表面反射特性，而 X 射线图像基于辐射透射原理反映内部密度分布。这种数据异质性使得简单合并两种模态变得困难。此外，配对的 RGB-X 射线数据集极为稀缺。

**本文目标**：设计一个统一的 3DGS 框架，能同时表示物体的 RGB 表面细节和 X 射线透视的内部结构，并解决两种模态之间的数据不一致性。

**切入角度**：作者观察到 3D 高斯原语本质上是空间中的三维分布，可以同时赋予表面外观属性和辐射透射属性。关键在于如何让同一组高斯分布既能准确渲染 RGB 外观，又能正确模拟 X 射线的穿透成像过程。

**核心 idea**：通过层次化拟合策略分阶段对齐 RGB 和 X 射线高斯溅射，并引入 X 射线参考损失确保内部结构的物理一致性，将 3DGS 扩展为一个"内外兼修"的综合三维表示。

## 方法详解

### 整体框架

InsideOut 的输入是同一物体的多视角 RGB 图像和多角度 X 射线图像。方法首先分别建立 RGB 高斯和辐射高斯（Radiative Gaussian）的独立表示，然后通过层次化拟合流程将两者对齐到统一的三维空间中。最终输出是一组"双属性"高斯原语，每个高斯同时携带 RGB 颜色/不透明度信息和 X 射线辐射衰减信息，可以同时渲染出逼真的 RGB 图像和物理正确的 X 射线图像。

### 关键设计

1. **辐射高斯溅射（Radiative Gaussian Splatting）**:

    - 功能：将 X 射线成像过程纳入高斯溅射框架
    - 核心思路：X 射线图像的形成是射线穿过物体时沿路径累积衰减的结果。作者将 Beer-Lambert 定律与高斯溅射结合，为每个高斯原语额外赋予一个辐射衰减系数 $\mu$。渲染 X 射线图像时，通过累积路径上所有高斯的衰减贡献来模拟透射成像，而非 RGB 渲染中的 alpha blending。
    - 设计动机：直接使用现有 3DGS 的 alpha 合成方式无法正确模拟 X 射线物理过程，需要根据辐射传输原理重新定义渲染方程。

2. **层次化拟合策略（Hierarchical Fitting）**:

    - 功能：将 RGB 高斯和辐射高斯对齐到同一三维空间
    - 核心思路：采用由粗到精的三阶段拟合过程。第一阶段独立优化 RGB 高斯和辐射高斯的空间位置和协方差参数；第二阶段通过全局刚性变换将两组高斯对齐，使它们共享相同的空间坐标系；第三阶段在对齐后的空间中联合微调所有参数，使每个高斯同时满足两种模态的渲染约束。这种渐进策略避免了直接联合优化时因模态差异过大导致的训练不稳定。
    - 设计动机：RGB 和 X 射线相机的几何标定通常独立进行，初始坐标系不同，需要通过对齐再联合优化。

3. **X 射线参考损失（X-ray Reference Loss）**:

    - 功能：确保重建的内部结构与 X 射线观测物理一致
    - 核心思路：除了标准的 RGB 重建损失和 X 射线重建损失外，引入额外的参考约束。该损失通过比较从不同角度渲染的 X 射线图像，强制高斯场的内部辐射分布在空间上保持一致性——即从任意角度透视时，内部结构的衰减分布都应与物理真实相符。这类似于一种多视角一致性正则化，具体作用于辐射属性。
    - 设计动机：仅靠有限角度的 X 射线图像监督，容易导致内部结构表示出现模糊或不一致的伪影，参考损失提供了额外的几何约束。

### 损失函数 / 训练策略

总损失函数包含三部分：RGB 重建损失（L1 + SSIM）、X 射线辐射重建损失（衡量渲染 X 射线与真实 X 射线的差异）、以及 X 射线参考损失。训练按层次化策略分阶段进行，每个阶段使用不同的损失组合和学习率。

## 实验关键数据

### 主实验

作者收集了新的 RGB-X 射线配对数据集，包含多种物体（如医学模型、工业零件、文物复制品等），评估了 RGB 渲染质量和 X 射线渲染质量：

| 任务 | 指标 | InsideOut | 仅RGB 3DGS | 仅X射线重建 |
|------|------|-----------|------------|------------|
| RGB 渲染 | PSNR (dB) | ~30+ | ~31 | N/A |
| RGB 渲染 | SSIM | ~0.95 | ~0.96 | N/A |
| X射线渲染 | PSNR (dB) | ~28+ | N/A | ~25 |
| X射线渲染 | SSIM | ~0.92 | N/A | ~0.88 |

InsideOut 在保持 RGB 渲染质量接近纯 RGB 3DGS 的同时，X 射线渲染质量显著超过单独使用 X 射线数据的重建方法。

### 消融实验

| 配置 | X射线 PSNR | RGB PSNR | 说明 |
|------|-----------|----------|------|
| Full Model | 最优 | 接近最优 | 完整 InsideOut |
| w/o Hierarchical Fitting | 下降 ~2dB | 下降 ~1dB | 直接联合优化导致对齐不准 |
| w/o X-ray Reference Loss | 下降 ~1.5dB | 基本不变 | 内部结构一致性变差 |
| w/o Radiative Rendering | 下降显著 | 不变 | 用 alpha blending 代替辐射传输 |

### 关键发现

- 层次化拟合是性能提升的关键，直接联合优化由于两种模态的梯度冲突导致收敛困难
- X 射线参考损失主要改善内部结构的多角度一致性，对 RGB 质量几乎无负面影响
- 在稀疏视角设定下，InsideOut 的优势更加明显，因为内外信息可以互相补充

## 亮点与洞察

- **统一多模态 3D 表示的思路非常新颖**：将 3DGS 从视觉表面扩展到物理内部，打破了"只能看到外面"的局限。这个 idea 本身就开辟了一个新方向。
- **层次化拟合策略巧妙解决了异质模态对齐问题**：与其强行端到端联合训练，不如分阶段各个击破。这种"先独立、再对齐、后联合"的策略对任何多模态融合任务都有参考价值。
- **设计思路可迁移到其他模态组合**：例如 RGB + 超声波、RGB + 热成像、RGB + CT 扫描等，任何需要同时建模表面和内部信息的场景都可借鉴此框架。

## 局限与展望

- 需要同一物体的配对 RGB 和 X 射线多视角数据，采集成本高,限制了大规模应用
- X 射线成像涉及辐射安全问题，实际部署需要考虑辐射剂量控制
- 当前方法假设物体静态，无法处理软体组织等形变物体
- 层次化拟合的超参数（如各阶段迭代次数、学习率调度）需要针对不同物体类型调整
- 未来可探索将该方法与 CT 重建结合，利用 3DGS 的实时渲染优势实现交互式医学可视化

## 相关工作与启发

- **vs 3D Gaussian Splatting**：标准 3DGS 只处理 RGB 图像，InsideOut 通过引入辐射高斯将其扩展到 X 射线模态，是 3DGS 在多物理场建模方向的首次尝试
- **vs NeRF-based CT 重建**：Neural Attenuation Fields 等工作用 NeRF 做 CT 重建，但无法同时建模 RGB 外观；InsideOut 的双模态设计更加全面
- **vs 传统多模态融合**：传统方法通常在 2D 层面融合 RGB 和 X 射线，InsideOut 在 3D 表示层面实现了统一，能从任意视角生成两种模态的图像

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 3DGS 扩展到 RGB+X射线双模态的内外兼顾三维表示，思路原创性很强
- 实验充分度: ⭐⭐⭐ 新建的数据集规模有限，缺少与更多baseline的对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述流畅，但部分细节需要参考补充材料
- 价值: ⭐⭐⭐⭐ 开辟了3DGS在无损检测和医学成像的新应用方向，潜力巨大但离实际部署还有距离

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](../../ECCV2024/medical_imaging/radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](../../CVPR2026/medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [\[ICCV 2025\] Predict-Optimize-Distill: A Self-Improving Cycle for 4D Object Understanding](predict-optimize-distill_a_self-improving_cycle_for_4d_object_understanding.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)

<!-- RELATED:END -->
