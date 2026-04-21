---
title: >-
  [论文解读] SAM4D: Segment Anything in Camera and LiDAR Streams
description: >-
  [ICCV 2025][自动驾驶][多模态分割] 提出 SAM4D，首个面向相机和 LiDAR 流的可提示多模态分割基础模型，通过统一多模态位置编码（UMPE）实现跨模态提示与交互，通过运动感知跨模态记忆注意力（MCMA）确保时序一致性，并构建包含 30 万+ masklet 的 Waymo-4DSeg 数据集，在跨模态分割和数据标注方面展示了强大能力。
tags:
  - ICCV 2025
  - 自动驾驶
  - 多模态分割
  - 基础模型
  - 相机-LiDAR融合
  - 时序分割
  - SAM
---

# SAM4D: Segment Anything in Camera and LiDAR Streams

**会议**: ICCV 2025  
**arXiv**: [2506.21547](https://arxiv.org/abs/2506.21547)  
**代码**: [SAM4D-Project.github.io](https://SAM4D-Project.github.io)  
**领域**: 自动驾驶  
**关键词**: 多模态分割, 基础模型, 相机-LiDAR融合, 时序分割, SAM

## 一句话总结

提出 SAM4D，首个面向相机和 LiDAR 流的可提示多模态分割基础模型，通过统一多模态位置编码（UMPE）实现跨模态提示与交互，通过运动感知跨模态记忆注意力（MCMA）确保时序一致性，并构建包含 30 万+ masklet 的 Waymo-4DSeg 数据集，在跨模态分割和数据标注方面展示了强大能力。

## 研究背景与动机

### 问题定义

在自动驾驶中，相机和 LiDAR 相互补偿各自的局限性（如弱光条件、深度精度），实现稳健的多模态感知至关重要。现有的分割模型局限于单一模态（图像或点云）且通常只在单帧上工作，无法利用跨模态的空间一致性和时序连续性。

### 已有方法的不足

**SAM/SAM2**：仅面向图像/视频分割，不支持 LiDAR 或其他传感器模态

**LiDAR 分割方法**（SAL、PointSAM）：直接在点云上构建 SAM-like 模型，但**仅限单模态**

**投影方法**（CLIP2Scene 等）：将 2D 分割投影到 3D，但受限于传感器视角差异和同步问题

**多模态感知方法**（BEVFusion 等）：输出仅为 3D 预测，缺少跨模态交互和统一的 2D-3D 分割

**帧级 LiDAR 分割**：未利用 LiDAR 的精确深度进行时序特征关联

### 核心动机

需要一个**统一的多模态时序分割框架**，能够：
- 在相机和 LiDAR 两个模态中同时生成分割mask
- 支持跨模态提示（如用图像上的点击指导 LiDAR 分割）
- 保持长序列的时序一致性
- 大幅降低多模态数据标注成本

## 方法详解

### 整体框架

SAM4D 在 SAM2 的基础上扩展到多模态领域，主要包含：
1. **多模态特征提取**：图像编码器（Hiera）+ LiDAR 编码器（MinkUNet）
2. **统一多模态位置编码（UMPE）**：在共享 3D 空间中对齐图像和 LiDAR 特征
3. **运动感知跨模态记忆注意力（MCMA）**：跨模态融合 + 自车运动补偿的时序注意力
4. **掩码解码器**：同时输出 2D 和 3D 分割掩码

### 关键设计

#### 1. **统一多模态位置编码（UMPE）**

- **功能**：在共享 3D 空间中对齐图像 patch token 和 LiDAR 体素 token 的位置表示。
- **核心思路**：

  UMPE 由两个互补部分组成：（i）模态特定的位置先验；（ii）共享 3D 空间表示。

  **图像位置编码**：
  - 2D 正弦位置编码保留图像平面结构：$\mathcal{P}_{\text{img\_sin}} = \text{SinPE2D}(u, v)$
  - 通过深度估计将像素提升到 3D 空间（类似 Lift-Splat-Shoot）：
  $$\mathbf{x}_{\text{img}} = T_c^l K^{-1} [u \cdot D(u,v), v \cdot D(u,v), D(u,v), 1]^T$$
  - MLP 编码 3D 位置：$\mathcal{P}_{\text{img\_mlp}} = \text{MLP}(\mathbf{x}_{\text{img}})$

  **LiDAR 位置编码**：
  - 3D 正弦位置编码：$\mathcal{P}_{\text{LiDAR\_sin}} = \text{SinPE3D}(x, y, z)$
  - 共享 MLP 编码 3D 位置：$\mathcal{P}_{\text{LiDAR\_mlp}} = \text{MLP}(\mathbf{x}_{\text{LiDAR}})$

  最终位置编码 $\mathcal{P}_{\text{img}}$ 和 $\mathcal{P}_{\text{LiDAR}}$ 各由两部分组成，确保跨模态位置对齐。

- **设计动机**：
    - 双阶段编码兼顾了模态特性（图像的 2D 结构、LiDAR 的 3D 结构）和跨模态对齐（共享 3D MLP）
    - 共享 3D MLP 使图像和 LiDAR 特征在同一空间中可比较，从而实现跨模态提示
    - 对稀疏提示（点、框）也使用相同的双阶段编码，保证提示与特征在同一空间

#### 2. **运动感知跨模态记忆注意力（MCMA）**

- **功能**：整合跨模态特征融合和自车运动补偿的时序记忆注意力。
- **核心思路**：

  三阶段注意力流程：

  **Step 1 — 模态内自注意力**：
  $$\mathcal{F}'_{\text{img}} = \text{SelfAttn}(\mathcal{F}_{\text{img}} + \mathcal{P}_{\text{img}})$$
  $$\mathcal{F}'_{\text{LiDAR}} = \text{SelfAttn}(\mathcal{F}_{\text{LiDAR}} + \mathcal{P}_{\text{LiDAR}})$$

  **Step 2 — 跨模态交叉注意力**：
  $$\mathcal{F}''_{\text{img}} = \text{CrossAttn}(\mathcal{F}'_{\text{img}}, \mathcal{F}'_{\text{LiDAR}} + \mathcal{P}_{\text{LiDAR}})$$
  $$\mathcal{F}''_{\text{LiDAR}} = \text{CrossAttn}(\mathcal{F}'_{\text{LiDAR}}, \mathcal{F}'_{\text{img}} + \mathcal{P}_{\text{img}})$$

  **Step 3 — 自车运动补偿的时序记忆注意力**：
  对历史帧特征和位置进行自车运动变换对齐到当前帧坐标系：
  $$\mathcal{M}_{\text{img}}^{t \leftarrow t'} = \mathcal{M}_{\text{img}}^{t'} + \Phi_{\text{img}}(T_{t \leftarrow t'}(\mathbf{x}_{\text{img}}))$$
  $$\mathcal{M}_{\text{LiDAR}}^{t \leftarrow t'} = \mathcal{M}_{\text{LiDAR}}^{t'} + \Phi_{\text{LiDAR}}(T_{t \leftarrow t'}(\mathbf{x}_{\text{LiDAR}}))$$

  然后用交叉注意力融合当前帧特征和对齐后的记忆特征：
  $$\mathcal{F}_{\text{img}}^{\text{final}} = \text{CrossAttn}(\mathcal{F}''_{\text{img}}, (\mathcal{M}_{\text{img}}^{t \leftarrow t'}, \mathcal{O}_{\text{img}}^{t'}))$$

  其中 $T_{t \leftarrow t'} \in SE(3)$ 来自车辆里程计。

- **设计动机**：
    - 与 SAM2 仅考虑短时运动不同，MCMA 显式引入自车运动补偿来处理自动驾驶中的大尺度场景变化
    - 自车运动变换确保历史帧特征在空间上与当前帧对齐，避免因自车行驶导致的特征错配
    - 记忆库采用 FIFO 队列，分别存储 $N$ 个非提示帧和 $M$ 个提示帧，确保关键帧被保留

#### 3. **多模态自动数据引擎**

- **功能**：自动生成高质量的相机-LiDAR 对齐伪标签，构建 Waymo-4DSeg 数据集。
- **核心思路**：

  三步流程：
  1. **VFM 驱动的视频 masklet 生成**：使用 Grounding-DINO 在关键帧检测物体 + SAM 分割 → SAM2 传播至中间帧
  2. **4D 体素重建**：利用 LiDAR 帧和 3D 框构建 4D 体素表示，建立像素-体素映射表
  3. **跨模态标签融合**：通过映射表将视频 masklet 投影到体素，DBSCAN 过滤噪声，合并跨视频重叠的 masklet，最终转移到 LiDAR 帧

  结果：跨模态 IoU 为 0.56。

- **设计动机**：
    - 不存在同时支持 2D 和 3D 分割且保证时序实例一致性的数据集
    - 利用 VFM（SAM、Grounding-DINO）的强零样本能力自动生成高质量标签
    - 通过 4D 重建作为中介桥接 2D 图像标签和 3D 点云标签

### 损失函数 / 训练策略

- 对图像和 LiDAR 预测使用相同的损失函数以强制跨模态一致性
- 训练时模拟交互式提示过程（类似 SAM2 的策略）
- 在 16 张 A100 上训练 36 个 epoch，每次最多处理 6 个物体
- 图像编码器使用 Hiera-S + SA-V 预训练，LiDAR 编码器使用 Mink-34
- 图像分辨率 768×768，LiDAR 体素化大小 0.15m

## 实验关键数据

### 主实验

跨模态帧级分割（Image-Prioritized Prompting）：

| 提示类型 | Image mIoU↑ | LiDAR mIoU↑ |
|----------|------------|-------------|
| 1-click | 68.0% | 42.3% |
| 3-click | 73.6% | **53.1%** |
| 边界框 | **74.7%** | 47.0% |

半监督流式分割（首帧提示 → 序列传播）：

| 提示类型 | Image mIoU↑ | J&F↑ | NMP↓ | LiDAR mIoU↑ | NMP↓ |
|----------|------------|------|------|-------------|------|
| 1-click | 61.4% | 72.2 | 398 | 50.1% | 784 |
| 3-click | 65.6% | 76.3 | 327 | 52.8% | 711 |
| 5-click | 67.1% | 77.7 | 315 | 52.6% | 702 |
| GT mask | **69.8%** | **80.1** | **280** | **55.7%** | **582** |

跨数据集泛化（nuScenes，半监督流式分割）：

| 设置 | Image mIoU↑ | J&F↑ | LiDAR mIoU↑ |
|------|------------|------|-------------|
| 零样本 | 58.4% | 65.8 | 25.9% |
| 微调 | **67.5%** | **75.4** | **44.8%** |

### 消融实验

输入模态消融：

| 配置 | Image mIoU↑ | J&F↑ | NMP↓ | LiDAR mIoU↑ | NMP↓ |
|------|------------|------|------|-------------|------|
| SAM2+投影 | 68.2% | 79.7 | 383 | 32.0% | - |
| SAM4D-仅相机 | 68.6% | 80.4 | 301 | - | - |
| SAM4D-仅LiDAR | - | - | - | 47.0% | 799 |
| **SAM4D（完整）** | **69.8%** | 80.1 | **280** | **55.7%** | **582** |

自车运动补偿消融：

| 配置 | Image mIoU↑ | LiDAR mIoU↑ | LiDAR NMP↓ |
|------|------------|-------------|-----------|
| 无自车运动 | 69.7% | 52.2% | 746 |
| **有自车运动** | **69.8%** | **55.7%** | **582** |

### 关键发现

1. **跨模态提示有效**：在图像上给提示可以在 LiDAR 上得到 53.1% mIoU，证明 UMPE 成功实现了跨模态对齐
2. **多模态融合显著提升 LiDAR**：LiDAR mIoU 从单模态 47.0% 提升到多模态 55.7%（+8.7%），说明图像的语义信息对点云分割有重大帮助
3. **SAM2+投影方案 LiDAR 仅 32.0%**：证明简单投影无法解决跨模态分割，需要深度融合
4. **自车运动补偿主要帮助 LiDAR**：LiDAR NMP 从 746 降至 582（-22%），说明 LiDAR 的稀疏性使其对空间对齐更敏感
5. **零样本 nuScenes 性能合理**：Image mIoU 58.4%，证明模型有一定泛化能力
6. **数据引擎效率**：平均每个 clip 生成 300 个 masklet，跨模态 IoU 0.56，速度远超人工标注

## 亮点与洞察

1. **首个相机+LiDAR 统一提示式分割模型**：填补了多模态分割基础模型的空白
2. **跨模态提示的创新**：用一个模态的提示指导另一个模态的分割，极大提升了标注效率
3. **UMPE 的巧妙设计**：通过深度估计将图像提升到 3D 空间，使两个模态在同一空间中交互
4. **数据引擎的工程价值**：结合 VFM + 4D 重建 + 跨模态融合，构建了大规模高质量伪标签
5. **Waymo-4DSeg 数据集**：30万+ masklet，覆盖车辆、行人、建筑等多类别，物体规模从 10 体素到 200K 体素

## 局限与展望

1. **依赖深度估计质量**：UMPE 中图像到 3D 的提升依赖深度估计，深度误差会降低跨模态对齐精度
2. **LiDAR 性能仍有差距**：LiDAR mIoU 55.7% 远低于 Image 69.8%，说明点云模态仍需更强的特征提取
3. **跨模态 IoU 仅 0.56**：数据引擎的标签质量仍有提升空间
4. **仅在 Waymo 上训练**：尽管测试了 nuScenes 泛化，但训练数据有限
5. **未评估下游任务影响**：未验证 SAM4D 标注数据对 3D 检测或轨迹预测等任务的迁移效果

## 相关工作与启发

- 与 **SAM2** 的区别：SAM2 仅支持视频分割，SAM4D 扩展到相机+LiDAR 多模态流
- 与 **PointSAM/SAL** 的区别：后者仅在点云上构建提示式分割，SAM4D 统一了 2D 和 3D
- 与 **BEVFusion** 等的区别：多模态感知方法输出 3D 预测，SAM4D 同时输出 2D 和 3D 分割
- **Lift-Splat-Shoot** 的思路被用于 UMPE 中，将 2D 特征提升到 3D 空间
- 数据引擎的设计思路（VFM→4D重建→跨模态融合）为自动驾驶数据自举提供了新范式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个统一相机+LiDAR 流的提示式分割基础模型，任务定义和模型设计都有突破
- **实验充分度**: ⭐⭐⭐⭐ — 多种评估设置，消融充分，但 LiDAR 分割基线对比不够
- **写作质量**: ⭐⭐⭐⭐ — 三大贡献（任务、模型、数据）结构清晰
- **价值**: ⭐⭐⭐⭐⭐ — 对多模态数据标注效率有巨大潜在影响，是 SAM 系列在自动驾驶方向的重要延伸

<!-- RELATED:START -->

## 相关论文

- [Segment Anything, Even Occluded](../../CVPR2025/autonomous_driving/segment_anything_even_occluded.md)
- [Detect Anything 3D in the Wild](detect_anything_3d_in_the_wild.md)
- [MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)

<!-- RELATED:END -->
