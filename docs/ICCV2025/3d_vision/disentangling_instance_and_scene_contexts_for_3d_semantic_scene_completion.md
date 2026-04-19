---
title: >-
  [论文解读] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion
description: >-
  [3D视觉] 提出 DISC，一种基于类别感知的双流架构用于 3D 语义场景补全，通过将实例类别和场景类别解耦到独立的查询流中并设计针对性的解码模块，在 SemanticKITTI 上仅用单帧输入即超越多帧 SOTA 方法，实例类别 mIoU 提升 17.9%。
tags:
  - 3D视觉
---

# Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion

## 论文信息

- **会议**: ICCV 2025
- **arXiv**: 2507.08555
- **代码**: [https://github.com/Enyu-Liu/DISC](https://github.com/Enyu-Liu/DISC)
- **领域**: 3D 感知 / 语义场景补全
- **关键词**: semantic scene completion, BEV, instance-scene disentanglement, dual-stream, autonomous driving

## 一句话总结

提出 DISC，一种基于类别感知的双流架构用于 3D 语义场景补全，通过将实例类别和场景类别解耦到独立的查询流中并设计针对性的解码模块，在 SemanticKITTI 上仅用单帧输入即超越多帧 SOTA 方法，实例类别 mIoU 提升 17.9%。

## 研究背景与动机

3D 语义场景补全（SSC）旨在从稀疏输入联合预测场景几何和语义。现有体素方法（VoxFormer、CGFormer、Symphonies 等）存在三个关键限制：

**实例类别预测差**：遮挡和投影误差导致类别遗漏和语义歧义（如行人被误识别）

**场景类别结构不连贯**：视野外区域导致拓扑错误（如道路出现在地形区域）

**体素方法的固有局限**：以体素为基本交互单元破坏了类别结构信息，统一模块难以应对实例与场景的差异化挑战

核心观察：实例类别（car, person, bicycle）和场景类别（road, building, vegetation）面临截然不同的挑战，需要差异化的处理策略。DISC 首次提出类别级别的双流范式来系统解决此问题。

## 方法详解

### 整体框架

DISC 在 BEV 空间操作，包含两个核心模块：

1. **判别性查询生成器 (DQG)**：为实例和场景分别生成带有几何和语义先验的查询
2. **双注意力类别解码器 (DACD)**：针对实例和场景的不同挑战设计专用解码层

### 关键设计 1：判别性查询生成器 (DQG)

**粗到精 BEV 生成**：通过 LSS 提升生成粗体素特征 $V_{\text{coarse}}$，用深度引导的表面体素精炼得到 $V_{\text{fine}}$，Z 轴最大池化得到 BEV 特征 $C$。

**实例查询**：利用图像空间分割检测潜在实例，投影到 BEV。通过 $k \times k$ 邻域抑制策略选择 $N_{\text{ins}}$ 个参考位置，在这些位置采样 BEV 特征初始化实例查询：

$$X_{\text{ins}} = \{CT(\mathbf{g}_n) \mid \mathbf{g}_n \in \text{Top-}N(\{\text{Max}(B_{k \times k}^i)\}_{i=1}^s)\}$$

$$\mathbf{Q}_{\text{ins}} = C[X_{\text{ins}}]$$

**场景查询**：采用 patch 设计捕获连续空间分布。将 BEV 特征分成等大 patch，每个 patch 中心作为参考点，通过上采样卷积将 patch 压缩到 $1 \times 1$ 作为场景查询初始化。

### 关键设计 2：自适应实例层 (AIL)

解决实例类别的高度信息丢失和遮挡问题：

1. **自适应高度采样**：对每个实例查询 $q_{\text{ins}}$，预测 N 个最可能的高度，构成 3D 参考点 $P_j = (x_{\text{ins}}, h_j)$
2. **图像交叉注意力**：将参考点投影到图像空间，通过可变形交叉注意力采样多层级特征：

$$q_{\text{ins}} = \sum_{j=1}^N w_j \text{DA}(q_{\text{ins}}, F^{2D}, \mathcal{T}^{WI}(x_{\text{ins}}, h_j))$$

3. **场景上下文融合**：实例查询从场景特征中提取感兴趣区域信息（如"人行道上的柱状物更可能是交通灯而非树干"）
4. **UNet 传播**：实例特征通过 UNet 网络传播到整个 BEV 平面

### 关键设计 3：全局场景层 (GSL)

解决场景类别的全局推理不足问题：

1. **全局语义聚合**：从最小尺度图像特征构建 $\mathbf{Q}_{\text{img}}$，通过交叉注意力聚合全局信息
2. **随机掩码**：丢弃部分 $\mathbf{Q}_{\text{img}}$ 模拟遮挡信息缺失，帮助网络推理场景布局
3. **自注意力**：扩展全局感受野，将可视区域特征传播到远处和视野外区域

### 特征融合与损失函数

类别解耦的高度预测融合：

$$V = (C_{\text{ins}} \bigotimes H_{\text{ins}}) + (C_{\text{scn}} \bigotimes H_{\text{scn}})$$

总损失包含 SSC 损失（Scene-Class Affinity + Cross-Entropy）、BEV 分割/高度增强损失和深度损失：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{ssc} + \lambda_2 \mathcal{L}_{aug} + \lambda_d \mathcal{L}_d$$

## 实验关键数据

### 主实验：SemanticKITTI 测试集

| 方法 | 输入 | IoU | InsM | ScnM | mIoU |
|------|------|-----|------|------|------|
| VoxFormer-T | 多帧 | 43.21 | 4.79 | 22.97 | 13.41 |
| HTCL | 多帧 | 44.23 | 6.48 | 28.86 | 17.09 |
| VoxFormer-S | 单帧 | 42.95 | 4.39 | 20.89 | 12.20 |
| Symphonize | 单帧 | 42.19 | 6.14 | 24.93 | 15.04 |
| CGFormer | 单帧 | 44.41 | 6.15 | 28.24 | 16.63 |
| **DISC (ours)** | **单帧** | **45.32** | **7.25** | **28.56** | **17.35** |

- 单帧 DISC 在 mIoU (17.35) 上超越所有多帧方法（含 HTCL 17.09）
- 实例 mIoU (InsM) 达到 7.25，较单帧 SOTA 提升 17.9%，较多帧 SOTA 提升 11.9%
- SSCBench-KITTI-360 上 mIoU 达 20.55，超越所有相机和激光雷达方法

### 消融实验（SemanticKITTI val）

| 配置 | IoU | InsM | ScnM | mIoU |
|------|-----|------|------|------|
| Baseline (VoxFormer) | 43.13 | 3.59 | - | 12.18 |
| + DQG (实例查询) | 43.78 | 5.22 | - | 14.31 |
| + DQG (场景查询) | 44.01 | 3.98 | - | 14.76 |
| + DACD (AIL + GSL) | 44.85 | 6.15 | - | 16.10 |
| **Full DISC** | **45.32** | **7.25** | **28.56** | **17.35** |

**关键发现**：
- 判别性查询（替换体素查询）贡献了主要的实例性能提升
- 双注意力解码器的针对性设计进一步提升 InsM 至 7.25
- 训练 epoch 仅需 20 轮，少于大多数现有方法（收敛更快）

## 亮点与洞察

1. **首次类别级范式**：从体素级交互转向类别级交互，根本性改变了 SSC 的处理范式
2. **BEV 空间的可行性论证**：通过实例-场景解耦缓解了 BEV 中高度轴的特征耦合问题（如行人和道路在同一 BEV 格中的高度歧义）
3. **单帧超多帧**：证明了充分利用单帧类别信息的潜力大于简单的多帧融合
4. **场景上下文辅助实例推理**：验证了"人行道上的柱状物→交通灯"这类先验关系的有效性

## 局限性

- 依赖预训练 MaskDINO 的实例分割质量，分割失败会影响实例查询初始化
- BEV 空间的高度预测仍是近似的，极端高度差（如高楼与地面）可能不够精确
- 仅在 KITTI 系列数据集验证，未扩展到 nuScenes 或 Waymo 等大规模数据集

## 相关工作与启发

- 与 Symphonies 的关系：Symphonies 隐式使用实例先验，DISC 显式解耦实例和场景流
- DQG 中的实例查询设计（图像空间检测 → BEV 投影 → 邻域抑制）可推广到其他 BEV 感知任务
- 双流架构思想类似 DETR 系列中对不同目标类型的差异化处理

## 评分

⭐⭐⭐⭐⭐ — 问题洞察深刻（实例 vs 场景的差异化挑战），方案设计优雅（类别级双流），实验结果突出（单帧超多帧），对 SSC 任务有范式级别的贡献。

<!-- RELATED:START -->

## 相关论文

- [CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

<!-- RELATED:END -->
