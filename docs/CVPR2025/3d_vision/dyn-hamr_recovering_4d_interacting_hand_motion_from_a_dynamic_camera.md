---
title: >-
  [论文解读] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera
description: >-
  [3D视觉] Dyn-HaMR 提出首个从动态相机单目视频中恢复双手 4D 全局运动轨迹的优化框架，通过三阶段流水线（层级初始化 → SLAM 引导全局运动优化 → 交互运动先验优化）解耦相机运动与手部运动，在多个数据集上大幅超越现有方法。
tags:
  - "3D视觉"
---

# Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera

## 一句话总结

Dyn-HaMR 提出首个从动态相机单目视频中恢复双手 4D 全局运动轨迹的优化框架，通过三阶段流水线（层级初始化 → SLAM 引导全局运动优化 → 交互运动先验优化）解耦相机运动与手部运动，在多个数据集上大幅超越现有方法。

## 研究背景与动机

从单目视频重建 3D 手部 mesh 是理解人类行为的关键任务，在 AR/VR 中有重要应用。但现有方法面临严重限制：

1. **弱透视相机假设**：HaMeR、IntagHand、ACR 等方法假设弱透视相机模型，只能在有限的相机视锥内模拟运动，无法恢复**全局 3D 轨迹**
2. **相机-手部运动耦合**：在自我中心（egocentric）场景中，相机随身体移动，手部运动和相机运动纠缠在一起，现有方法无法解耦二者
3. **深度估计噪声**：仅依赖 2D 线索导致深度模糊，产生噪声或错误的深度估计
4. **交互遮挡**：双手交互频繁导致遮挡、截断和检测缺失
5. **数据集缺乏**：没有足够的时序数据集用于学习 4D 全局交互

核心挑战：如何在动态相机和复杂手部交互的条件下，完成从相机坐标系到世界坐标系的手部全局运动恢复？

## 方法详解

### 整体框架

Dyn-HaMR 是一个三阶段多目标优化流水线：Stage I 进行层级初始化和运动补全；Stage II 利用 SLAM 估计相机运动并优化全局轨迹；Stage III 引入手部运动先验和生物力学约束精炼交互。使用 MANO 参数化手部模型，全局运动表示为手部状态序列 $\mathbf{Q}^h = \{q_t^h\}_{t=1}^T$。

### 关键设计

#### 1. 层级初始化与生成式运动补全（Stage I）

- **功能**：从原始视频初始化逐帧手部状态，并填补遮挡丢失的检测
- **核心思路**：融合 ViTPose、ACR、HaMeR、MediaPipe 四个手部检测/重建方法进行层级初始化——先用 ViTPose 获取手部边界框序列，再用 ACR/HaMeR/MediaPipe 提取逐帧 MANO 参数。对缺失帧使用 HMP 手部运动先验的潜在空间优化进行生成式补全
- **设计动机**：单帧方法缺乏时序一致性且频繁检测失败，单一方法不够鲁棒。层级融合多方法提高覆盖率，生成式补全比简单插值更符合运动学约束

#### 2. SLAM 引导的 4D 全局运动优化（Stage II）

- **功能**：解耦相机运动和手部运动，恢复世界坐标系下的全局手部轨迹
- **核心思路**：使用 DPVO（SLAM 系统）估计相对相机运动 $\mathbf{C}_t = \{\mathbf{R}_t, \boldsymbol{\tau}^c_t\}$，通过组合相机运动和手部运动得到全局轨迹。关键引入**世界尺度因子** $\omega$ 建模相机位移与手部运动的相对尺度。联合优化全局轨迹、朝向、局部姿态和相机外参
- **设计动机**：SLAM 提供的相机运动尺度天然不确定，而手部运动有自然的合理范围约束。优化 $\omega$ 利用双手运动进一步约束相机尺度，解决单目深度模糊。分两步优化：先 20 步优化全局朝向和平移，再 60 步优化局部姿态、形状、尺度因子和相机外参

#### 3. 交互运动先验优化（Stage III）

- **功能**：利用学习的运动先验和生物力学约束精炼双手交互，产生更真实的运动
- **核心思路**：在 HMP 运动先验的潜在空间中优化，引入三类额外约束——(a) 运动先验损失确保运动的似然性；(b) 穿透损失防止双手 mesh 穿透；(c) 生物力学约束确保关节角度、骨骼长度和掌部形态在生理合理范围内
- **设计动机**：Stage II 的重投影损失缺乏足够约束会产生不合理姿态。运动先验提供运动学先验知识，穿透和生物力学约束解决交互中的物理不合理问题。分两阶段：先 200 步优化全局运动，再 200 步加入潜在码、局部姿态和相机参数

### 损失函数

**Stage II** 全局优化目标：

$$E_I = \lambda_{2d}\mathcal{L}_{2d} + \lambda_s\mathcal{L}_{smooth} + \lambda_{cam}\mathcal{L}_{cam} + \lambda_J\mathcal{L}_J + \lambda_\beta\mathcal{L}_\beta$$

**Stage III** 交互优化目标：

$$E_{II} = \mathcal{L}_{prior} + \mathcal{L}_{pen} + \mathcal{L}_{bio} + \lambda_{2d}\mathcal{L}_{2d} + \lambda_s\mathcal{L}_{smooth} + \lambda_{cam}\mathcal{L}_{cam} + \lambda_J\mathcal{L}_J + \lambda_\beta\mathcal{L}_\beta$$

其中 $\mathcal{L}_{prior} = \lambda_z\mathcal{L}_z + \lambda_\phi\mathcal{L}_\phi + \lambda_\tau\mathcal{L}_\tau$（运动先验似然 + 全局朝向一致性 + 平移一致性），$\mathcal{L}_{bio}$ 包含关节角度、骨骼长度和掌部约束，$\mathcal{L}_{pen}$ 基于双手穿透顶点的 Chamfer 距离。

## 实验关键数据

### 主实验表

**H2O 数据集（动态相机，Tab. 2）**：

| 方法 | G-MPJPE↓ | GA-MPJPE↓ | MPJPE↓ | Acc Err↓ |
|------|----------|-----------|--------|----------|
| ACR | 113.6 | 88.5 | 46.8 | 14.3 |
| IntagHand | 105.5 | 81.5 | 45.6 | 13.5 |
| HaMeR | 96.9 | 75.7 | 32.9 | 9.21 |
| Ours (w/o III) | 51.9 | 41.2 | 24.9 | 9.5 |
| **Dyn-HaMR** | **45.6** | **34.2** | **22.5** | **4.2** |

**InterHand2.6M（静态相机，Tab. 1）**：

| 方法 | MPJPE↓ | MPVPE↓ | Acc Err↓ |
|------|--------|--------|----------|
| ACR | 8.75 | 9.01 | 3.99 |
| HaMeR | 9.84 | 10.13 | 5.13 |
| **Dyn-HaMR** | **7.94** | **8.15** | **2.76** |

### 消融实验（Tab. 2 & Tab. 6）

| 变体 | G-MPJPE↓ | MPJPE↓ | Acc Err↓ |
|------|----------|--------|----------|
| w/o Stage III | 51.9 | 24.9 | 9.5 |
| w/o 生物力学约束 | - | 改善但不如完整版 | - |
| w/o 穿透损失 | - | 手部穿透严重 | - |
| w/o 生成式补全 | - | 性能下降 | - |
| **完整版** | **45.6** | **22.5** | **4.2** |

### 关键发现

1. **全局运动恢复大幅领先**：在 H2O 上 G-MPJPE 从 HaMeR 的 96.9 降至 45.6（**53% 降幅**），GA-MPJPE 从 75.7 降至 34.2（**55% 降幅**）
2. **静态相机也有效**：即使在静态相机的 InterHand2.6M 上也刷新 SOTA，MPJPE 7.94 vs ACR 的 8.75
3. Stage III 的运动先验和交互约束贡献显著，加速度误差从 9.5 降至 4.2

## 亮点与洞察

1. **首次解决动态相机下双手 4D 全局运动恢复**：填补了一个重要的研究空白，对 AR/VR 第一人称手势交互意义重大
2. **巧妙的尺度因子设计**：通过优化 $\omega$ 解耦相机位移和手部运动的相对尺度，利用双手运动的合理性约束反过来帮助估计相机尺度
3. **生成式运动补全优于插值**：利用 HMP 先验在潜在空间中进行运动补全，同时处理时序平滑和缺失检测
4. **三阶段渐进式优化设计合理**：从粗到细，每个阶段引入更强的约束

## 局限性与可改进方向

1. **优化效率**：三阶段优化管线（L-BFGS 共480步+）计算耗时，难以实时应用
2. **依赖 SLAM 质量**：DPVO 在极端运动模糊或低纹理场景中可能失败，影响下游全局运动恢复
3. **HMP 先验局限**：手部运动先验仅在 Arctic 数据集上训练，对未见过的交互类型泛化能力有限
4. **物体交互未建模**：虽然在包含物体的数据集上评估，但未显式建模手-物交互约束
5. 缺少在超长视频（>1000帧）上的评估和可扩展性分析

## 相关工作与启发

- **HaMeR / ACR / IntagHand**：单帧手部重建方法，Dyn-HaMR 以它们为初始化基础
- **HuMoR / WHAM**：人体全局运动恢复，Dyn-HaMR 将类似思路引入手部领域
- **HMP**：手部运动先验模型，为 Stage I/III 提供运动学约束
- **DPVO**：数据驱动 SLAM 系统，提供相机运动估计
- 启发：手部 4D 重建可以参考人体运动恢复的方法论框架（SLAM + 运动先验 + 分阶段优化）

## 评分：⭐⭐⭐⭐

问题定义明确且实用（动态相机手部重建是 AR/VR 刚需），三阶段优化设计完整，实验覆盖面广（6+ 数据集）。全局运动恢复上的大幅领先令人信服。扣一星因为优化效率限制了实际部署，且 HMP 先验的泛化性未被充分验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)

</div>

<!-- RELATED:END -->
