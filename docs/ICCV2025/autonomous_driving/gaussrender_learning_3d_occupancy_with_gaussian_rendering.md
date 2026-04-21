---
title: >-
  [论文解读] GaussRender: Learning 3D Occupancy with Gaussian Rendering
description: >-
  [ICCV 2025][自动驾驶][3D Occupancy Prediction] 提出 GaussRender，一个即插即用的可微高斯渲染模块，通过将预测和真值的 3D occupancy 投影到 2D 视图并施加语义和深度一致性约束，消除浮空体素等视觉伪影，在多个 benchmark 上显著提升几何保真度，尤其在 RayIoU 等表面敏感指标上提升突出。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D Occupancy Prediction
  - Gaussian Splatting
  - Differentiable Rendering
  - Projective Consistency
---

# GaussRender: Learning 3D Occupancy with Gaussian Rendering

**会议**: ICCV 2025  
**arXiv**: [2502.05040](https://arxiv.org/abs/2502.05040)  
**代码**: https://github.com/valeoai/GaussRender (有)  
**领域**: 自动驾驶  
**关键词**: 3D Occupancy Prediction, Gaussian Splatting, Differentiable Rendering, Projective Consistency, autonomous driving

## 一句话总结

提出 GaussRender，一个即插即用的可微高斯渲染模块，通过将预测和真值的 3D occupancy 投影到 2D 视图并施加语义和深度一致性约束，消除浮空体素等视觉伪影，在多个 benchmark 上显著提升几何保真度，尤其在 RayIoU 等表面敏感指标上提升突出。

## 研究背景与动机

3D 占据预测（3D Occupancy Prediction）是自动驾驶感知的核心任务，需要从多视角相机图像推断驾驶场景的三维几何和语义结构。现有方法（如 SurroundOcc、TPVFormer）通常使用逐体素的交叉熵、Dice 或 Lovász 等损失函数进行训练，这类损失对所有体素一视同仁，**不考虑相邻体素之间的空间一致性**。

这导致了一个关键问题：模型虽然在 voxel-wise 的 IoU 指标上表现不错，但预测结果存在大量**视觉伪影**——浮空体素（floating voxels）、不连续表面、边界错位等。这些伪影虽然对体素分割损失影响甚微（因为损失给所有体素相同权重），但对下游任务（如自由空间估计、运动规划）可能造成严重影响。

本文的核心洞察是：如果把 3D 预测投影到 2D 视图中，这些不合理的空间排列会在投影中显现出来。因此，**将投影一致性（projective consistency）引入训练目标**，可以鼓励模型学习到一致的、物理合理的几何结构。与 NeRF-based 渲染方法相比，本文利用 Gaussian Splatting 实现高效渲染，无需时序监督或 LiDAR 重投影。

## 方法详解

### 整体框架

GaussRender 的流程如下：(1) 将预测和真值的体素网格"高斯化"——每个体素变为一个球形高斯原语；(2) 在场景中放置虚拟相机（包括固定的鸟瞰视图和动态随机相机）；(3) 利用 Gaussian Splatting 将 3D 高斯投影到 2D，产生语义和深度渲染图；(4) 对比预测与真值的渲染结果，计算 L1 损失。整个模块只作用于训练阶段，推理时无任何额外计算。

总损失为：$L = L_{3D} + \lambda L_{2D}$，其中 $L_{3D}$ 是标准逐体素损失，$L_{2D}$ 是渲染一致性损失。

### 关键设计

1. **体素高斯化（Voxel Gaussianization）**:

    - 功能：将每个体素转换为一个简单的球形高斯原语
    - 核心思路：每个体素的高斯参数极度简化——位置 $\mu$ 固定为体素中心，尺度 $S = \text{Diag}(s)$ 基于体素尺寸固定，旋转 $R = I$（球形无需方向），语义 "颜色" $c$ 取自模型最终预测的 logits，仅不透明度 $o$ 从体素特征中学习（或从 empty 类的 logit 推导）
    - 设计动机：极度简化是为了减轻学习负担。与 GaussianOcc、GaussTR 等将高斯表示与模型架构耦合的方法不同，GaussRender 完全在预测层面操作，不要求底层 3D 表示是高斯形式

2. **虚拟相机放置策略（Camera Placement Strategy）**:

    - 功能：在 3D 场景中灵活放置虚拟渲染相机
    - 核心思路：采用两类虚拟相机——(a) 固定正交鸟瞰相机（BeV），提供全局场景俯视约束；(b) 动态"抬高+平移"相机，沿 z 轴抬高并在 xy 平面小范围随机平移，扩大视野覆盖遮挡区域
    - 设计动机：传统方法受限于传感器相机视角或 LiDAR 重投影，无法约束遮挡区域。GaussRender 同时渲染预测和真值，不依赖 RGB 图像或 LiDAR 伪标签，可从任意视角渲染。抬高的相机能"看到"被地面物体水平遮挡的区域

3. **高斯渲染与 2D 损失（Gaussian Rendering & L2D Loss）**:

    - 功能：将 3D 高斯投影到 2D 并计算语义和深度的渲染损失
    - 核心思路：3D 协方差矩阵通过相机参数投影到图像平面：$\Sigma_{2D} = J \cdot W \cdot \Sigma_{3D} \cdot W^T \cdot J^T$。对每个像素 $p$，渲染语义值 $C_p = \sum_i T_i \alpha_i \mathbf{c}_i$ 和深度 $D_p = \sum_i T_i \alpha_i \mathbf{d}_i$，其中 $T_i = \prod_{j<i}(1-\alpha_j)$ 为累积透射率。最终每个虚拟相机的损失为：$L_{2D}^* = L_{depth}^* + L_{sem}^*$，总渲染损失为 $L_{2D} = L_{2D}^{bev} + L_{2D}^{cam}$
    - 设计动机：语义渲染损失强化局部语义一致性，深度渲染损失惩罚破坏遮挡关系的伪影。深度损失用 $d_{range}^*$ 归一化以保证跨场景的尺度一致性

### 损失函数 / 训练策略

- 语义损失采用 L1 距离比较预测和真值的语义渲染：$L_{sem}^* = \|I_{sem}^* - \tilde{I}_{sem}^*\|_1$
- 深度损失同样 L1 距离并做归一化：$L_{depth}^* = \frac{1}{d_{range}^*}\|I_{depth}^* - \tilde{I}_{depth}^*\|_1$
- 真值渲染时，占据体素不透明度设为 1，空体素为 0
- 训练时与标准 3D 损失并行计算，无需修改模型架构
- 推理时完全移除渲染模块，零额外开销

## 实验关键数据

### 主实验

| 数据集 / 模型 | 指标 | 原始 | +GaussRender | 提升 |
|---|---|---|---|---|
| SurroundOcc-nuSc / TPVFormer | IoU / mIoU | 30.86 / 17.10 | 32.05 / 20.85 | +1.19 / +3.75 |
| SurroundOcc-nuSc / SurroundOcc | IoU / mIoU | 31.49 / 20.30 | 32.61 / 20.82 | +1.12 / +0.52 |
| Occ3D-nuSc / TPVFormer | mIoU | 27.83 | 30.48 | +2.65 |
| Occ3D-nuSc / SurroundOcc | mIoU | 29.21 | 30.38 | +1.17 |
| SSCBench-KITTI360 / Symphonies | IoU / mIoU | 43.40 / 17.82 | 44.08 / 18.11 | +0.68 / +0.29 |
| Occ3D-nuSc / TPVFormer | RayIoU | 37.2 | 38.3 | +1.1 |
| Occ3D-nuSc / SurroundOcc | RayIoU | 35.5 | 37.5 | +2.0 |

### 消融实验

| 配置 | IoU | mIoU | 说明 |
|------|-----|------|------|
| Cam 语义 | 26.3 | 14.3 | 仅相机语义渲染损失 |
| + Cam 深度 | 26.8 | 15.1 | 加入相机深度损失 |
| + BeV 语义 | 27.2 | 15.6 | 加入鸟瞰语义渲染 |
| + BeV 深度（完整） | 27.5 | 16.4 | 所有损失组件 |
| 相机策略: Sensor (2D+3D) | - | 25.9 | 传感器位置 |
| 相机策略: Elevated+Around (2D+3D) | - | 26.3 | 最佳策略 |
| 相机策略: Fully Random (2D+3D) | - | 25.4 | 随机视点效果差 |

### 关键发现

- GaussRender 在三个数据集、三种不同架构上均带来一致提升，验证了其通用性
- 在表面敏感的 RayIoU 指标上提升更显著（SurroundOcc +2.0），说明渲染约束有效消除浮空伪影
- 在仅 2D 监督设置下（无 3D GT），GaussRender 达到 25.3 mIoU，超越所有已有渲染方法（包括使用时序帧的 RenderOcc 23.9）
- 抬高+环绕相机策略在 2D+3D 训练中最优，但纯 2D 训练时传感器位置最优——说明虚拟相机放置需要根据监督信号调整

## 亮点与洞察

- **即插即用设计**：不修改任何模型架构，仅在训练时增加渲染损失，推理零开销。能让老架构（TPVFormer）超越新方法（GaussianFormerV2），说明好的训练策略比复杂架构更重要
- **摆脱 LiDAR 依赖**：与 RenderOcc、GSRender 不同，GaussRender 同时渲染预测和真值，无需 LiDAR 重投影或伪标签
- **任意视角渲染能力**：打破了现有方法只能从传感器视角或时序帧渲染的限制

## 局限与展望

- 虚拟相机策略目前基于简单的规则（抬高 + 随机平移），尚未自适应场景复杂度
- 渲染分辨率和高斯参数的影响未充分探索
- 未与时序信息结合——整合动态视角合成和时序序列可能进一步改善遮挡推理
- 未拓展到开放词汇场景理解

## 相关工作与启发

- **RenderOcc / GSRender**：用 NeRF 或高斯渲染做 2D 监督，但依赖 LiDAR 重投影和时序帧，GaussRender 完全摆脱这些限制
- **GaussianOcc / GaussTR**：将高斯表示嵌入模型架构，GaussRender 则在预测层面操作，更灵活
- **SparseOcc**：提出 RayIoU 指标，GaussRender 在该指标上提升最大，验证了投影约束的价值

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

<!-- RELATED:END -->
