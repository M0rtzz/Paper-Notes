---
title: >-
  [论文解读] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training
description: >-
  [ICCV 2025][3D视觉][4D重建] 提出 Easi3R，一种免训练的即插即用方法，通过分析和操控 DUSt3R 交叉注意力层中隐含的运动信息，实现动态物体分割、相机位姿估计和 4D 密集点云重建。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "4D重建"
  - "DUSt3R"
  - "注意力解耦"
  - "动态分割"
  - "免训练"
---

# Easi3R: Estimating Disentangled Motion from DUSt3R Without Training

**会议**: ICCV 2025  
**arXiv**: [2503.24391](https://arxiv.org/abs/2503.24391)  
**代码**: [https://easi3r.github.io](https://easi3r.github.io)  
**领域**: 3D视觉 / 动态场景重建  
**关键词**: 4D重建, DUSt3R, 注意力解耦, 动态分割, 免训练

## 一句话总结
提出 Easi3R，一种免训练的即插即用方法，通过分析和操控 DUSt3R 交叉注意力层中隐含的运动信息，实现动态物体分割、相机位姿估计和 4D 密集点云重建。

## 研究背景与动机

**领域现状**：DUSt3R 在静态场景中实现了鲁棒的密集点云和相机参数估计。MonST3R、CUT3R 等工作通过在动态数据集上微调来扩展到动态场景，但需要大量训练数据或光流/深度等先验模型。

**现有痛点**：(1) 4D 数据集规模和多样性有限，制约了高泛化性 4D 模型的训练；(2) 现有动态方法依赖光流估计器、深度预测器等外部先验，增加了系统复杂度；(3) DUSt3R 在动态视频上性能显著下降，因为移动物体违反了其静态场景的极线一致性假设。

**核心矛盾**：要处理动态场景需要识别和解耦物体运动与相机运动，但这通常需要在大规模动态数据上训练——然而此类数据集稀缺且昂贵。

**本文目标**：不训练、不微调，直接从预训练的 DUSt3R 中提取运动信息来处理动态视频。

**切入角度**：类比人脑的注意力机制——人类能从视觉中分离自身运动和物体运动，而 DUSt3R 的交叉注意力层也隐式学习了类似的机制。分析发现，动态区域在交叉注意力中会获得低关注值。

**核心 idea**：将 DUSt3R 交叉注意力图在空间和时间维度聚合，提取四种语义有意义的注意力图（源/参考视图的均值/方差），通过组合推导出动态物体分割，再用分割结果重加权注意力进行第二次推理以获得鲁棒的 4D 重建。

## 方法详解

### 整体框架
给定动态视频 → 用滑动时间窗成对输入 DUSt3R → 提取并聚合交叉注意力图 → 分解出动态分割 $M^t$ → 用分割结果重加权交叉注意力层 → 第二次推理获得鲁棒的点云和相机位姿。整个过程无需训练。

### 关键设计

1. **交叉注意力分解与聚合**:

    - 功能：从 DUSt3R 的注意力层中提取运动和结构信息
    - 核心思路：对每帧的所有成对推理，计算四种时序聚合注意力图：(a) $A_\mu^{a=\text{ref}}$ 参考视图均值→纹理少/观测不足区域低值；(b) $A_\sigma^{a=\text{ref}}$ 参考视图方差→相机运动模式；(c) $1-A_\mu^{a=\text{src}}$ 源视图反转均值→动态物体+纹理少区域；(d) $A_\sigma^{a=\text{src}}$ 源视图方差→相机运动+物体运动
    - 设计动机：DUSt3R 训练时学习了刚体视图变换，动态物体违反此假设导致低注意力值——这个"失败模式"恰好可以被利用来检测动态区域

2. **动态物体分割**:

    - 功能：从注意力图中提取免训练的动态物体掩码
    - 核心思路：$A^{a=\text{dyn}} = (1 - A_\mu^{a=\text{src}}) \cdot A_\sigma^{a=\text{src}} \cdot A_\mu^{a=\text{ref}} \cdot (1 - A_\sigma^{a=\text{ref}})$。该公式的逻辑：源视图低均值+高方差标记动态+纹理少区域，乘以参考视图高均值排除纹理少区域，乘以参考视图低方差排除相机运动
    - 设计动机：四种注意力图各有侧重，通过乘积组合精准隔离出"仅物体运动"——不需要光流或分割模型

3. **注意力重加权推理**:

    - 功能：利用动态分割结果重新推理以获得鲁棒的 4D 重建
    - 核心思路：第二次推理时在交叉注意力层中降低动态区域的权重，使模型聚焦于静态背景进行精确的相机位姿估计和点云对齐。动态物体的点云从各帧独立保留
    - 设计动机：直接处理动态视频时 DUSt3R 会因动态区域的错误匹配而崩溃，重加权让模型"忽略"动态部分

### 损失函数 / 训练策略
完全免训练，不涉及任何损失函数或优化。仅在推理时操控注意力层。

## 实验关键数据

### 主实验

| 任务 | Easi3R | MonST3R | CUT3R | DAS3R |
|------|--------|---------|-------|-------|
| 相机位姿估计 | **最优/接近最优** | 次优 | 较低 | 较低 |
| 动态物体分割 | 有效 | 依赖光流 | 不支持 | 依赖训练 |
| 4D 点云重建 | 鲁棒 | 需微调 | 需微调 | 需微调 |

在多个真实动态视频数据集上超越了需要在动态数据上训练/微调的方法。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 原始 DUSt3R | 动态场景崩溃 | 基线 |
| + 注意力聚合分割 | 有效分割动态 | 核心创新 |
| + 重加权推理 | 位姿和重建显著改善 | 完整流程 |
| 不同backbone (DUSt3R/MonST3R) | 均有提升 | 即插即用 |

### 关键发现
- DUSt3R 的交叉注意力中确实编码了丰富的运动信息，仅通过聚合和分析即可获得高质量的动态分割
- 这种免训练方法在多个数据集上超越了在大量动态数据上训练过的方法（MonST3R等），说明预训练 3D 模型中的implicit knowledge比显式训练更具泛化性

## 亮点与洞察
- **从失败中发现机会**：DUSt3R 在动态场景下因低注意力值而"失败"——但正是这种失败模式被转化为动态检测信号，思路极其巧妙
- **免训练 > 有训练**的反直觉结果：说明在 4D 数据稀缺的现状下，利用大规模 3D 预训练模型的隐式知识比在小型 4D 数据上微调更有效
- 即插即用设计：可直接应用于 DUSt3R 和 MonST3R 等多种backbone

## 局限与展望
- 依赖 DUSt3R 的预训练质量——如果预训练模型在某些场景下注意力模式不佳则失效
- 动态分割基于简单阈值 $\alpha$，对复杂场景（部分遮挡、缓慢运动）可能不够鲁棒
- 滑动窗口大小的选择影响分割质量
- 尚未处理快速大幅运动导致的完全无重叠区域

## 相关工作与启发
- **vs MonST3R**: MonST3R 在动态数据上微调 DUSt3R+用光流做分割；Easi3R 免训练直接从注意力层提取
- **vs CUT3R**: CUT3R 在静态+动态数据上微调但不做分割，动静纠缠；Easi3R 显式解耦
- **vs RoMo**: RoMo 用 COLMAP+光流+SAM2 做动态分割，Easi3R 完全从注意力内部获取

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 免训练从注意力中提取运动信息的想法极其新颖且优雅
- 实验充分度: ⭐⭐⭐⭐ 三个任务、多数据集、与训练方法的公平对比
- 写作质量: ⭐⭐⭐⭐⭐ 注意力可视化分析非常直观，"secrets behind DUSt3R"的叙述引人入胜
- 价值: ⭐⭐⭐⭐⭐ 为4D重建提供了免训练的高效方案，具有很强的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[CVPR 2025\] Estimating Body and Hand Motion in an Ego-sensed World](../../CVPR2025/3d_vision/estimating_body_and_hand_motion_in_an_ego-sensed_world.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)

</div>

<!-- RELATED:END -->
