---
title: >-
  [论文解读] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance
description: >-
  [NeurIPS 2025][自动驾驶][3D Semantic Scene Completion] 提出 FlowScene，利用光流引导时序特征聚合并结合遮挡掩码进行体素细化，在仅使用2帧历史输入的条件下，在 SemanticKITTI 和 SSCBench-KITTI-360 基准上达到 SOTA（mIoU 17.70 / 20.81）。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 3D Semantic Scene Completion
  - 光流引导
  - 时序建模
  - 遮挡感知
  - 体素细化
---

# FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance

**会议**: NeurIPS 2025  
**arXiv**: [2502.14520](https://arxiv.org/abs/2502.14520)  
**作者**: Meng Wang, Fan Wu, Ruihui Li, Yunchuan Qin, Zhuo Tang, Kenli Li (湖南大学)
**代码**: https://github.com/willemeng/FlowScene (有)  
**领域**: 自动驾驶 / 3D语义场景补全  
**关键词**: 3D Semantic Scene Completion, 光流引导, 时序建模, 遮挡感知, 体素细化

## 一句话总结

提出 FlowScene，利用光流引导时序特征聚合并结合遮挡掩码进行体素细化，在仅使用2帧历史输入的条件下，在 SemanticKITTI 和 SSCBench-KITTI-360 基准上达到 SOTA（mIoU 17.70 / 20.81）。

## 研究背景与动机

3D 语义场景补全（SSC）是自动驾驶感知的核心任务，目标是从稀疏观测中联合推断 3D 场景的几何结构和语义标签。现有方法存在两类局限：

**单帧方法**（MonoScene、CGFormer 等）：仅依赖当前帧的有限观测恢复 3D 几何与语义，信息量不足

**现有时序方法**（VoxFormer-T、HTCL 等）：简单堆叠历史帧特征或通过估计相机位姿对齐特征，忽略了场景运动上下文，无法实现时序一致性

核心问题：**如何准确识别历史帧与当前帧的关联，以有效引导时序 SSC 建模？**

光流天然编码了帧间像素级的运动对应关系，包含运动、不同视角、遮挡、形变等多种信息。本文提出利用光流引导时序建模，将运动感知与遮挡信息注入到 SSC 流程中。

## 方法详解

### 整体框架

FlowScene 包含四个核心模块：

1. **图像编码器**：RepViT + FPN 提取当前帧特征 $F_t$ 和历史帧特征 $F_{temp}$
2. **光流估计（OFE）**：使用预训练 GMFlow 生成双向光流，通过前后一致性检测获得遮挡掩码 $M$
3. **光流引导时序聚合（FGTA）**：在 2D 特征空间中基于光流对齐与聚合时序特征
4. **遮挡引导体素细化（OGVR）**：在 3D 体素空间中利用遮挡掩码自适应融合体素特征

### 光流估计与遮挡检测

**光流引导 Warping**：利用 GMFlow 估计从当前帧到历史帧的光流场 $Flow^{t \to t-i}$，通过双线性插值将历史帧特征 warp 到当前帧坐标：

$$F_{warp}^{t-i \to t} = \mathcal{W}arp(F_{t-i}, Flow^{t \to t-i})$$

**遮挡检测**：利用经典的前后向一致性检测（forward-backward consistency check）。对于每个像素，先通过前向光流映射到历史帧，再通过后向光流映射回来。若往返残差超过阈值 $\tau$，则标记为遮挡：

$$M(x) = \begin{cases} 1 & \text{if } \|\Delta(x)\| > \tau \quad \text{(occluded)} \\ 0 & \text{otherwise} \end{cases}$$

这种检测方式零参数、零训练开销，但能有效识别遮挡区域。

### 光流引导时序聚合（FGTA）

FGTA 模块在 2D 图像特征空间中完成时序特征的对齐与聚合，包含两个子模块：

**（1）时序聚合（Temporal Aggregation）**：对 warp 后的历史帧特征，通过余弦相似度计算空间位置权重：

$$w_{t-i \to t}(P) = \text{similarity}(F_{warp}^{t-i \to t}(P), F_t(P))$$

加权聚合得到聚合特征 $F_{agg} = \sum_{i=0}^{t} w_{t-i \to t} \cdot F_{warp}^{t-i \to t}$。直觉上，warp 后与当前帧越相似的历史特征获得更高权重，实现 motion-aware 的特征融合。

**（2）遮挡交叉注意力（Occlusion Cross-Attention）**：利用邻域交叉注意力（Neighborhood Cross-Attention）机制，选择性地从历史帧的**非遮挡区域**补充当前帧的特征：

$$F_t = \mathcal{NCA}(F_t, (1 - M) \cdot F_{warp})$$

其中当前帧特征作为 query，历史帧非遮挡区域的 warp 特征作为 key/value。这避免了将不可靠的遮挡区域信息注入当前帧，同时利用历史帧中可见但当前帧被遮挡的纹理与上下文信息。

### 遮挡引导体素细化（OGVR）

FGTA 在 2D 空间完成时序融合后，仍缺乏 3D 空间的显式几何建模。OGVR 模块将遮挡感知提升到 3D 体素空间。

首先，通过 LSS（Lift-Splat-Shoot）视图变换，将 $F_t$、$F_{agg}$ 和 $M$ 分别投影到 3D 体素空间，得到 $V_t$、$V_{agg}$ 和 $V_{mask}$。

然后，利用遮挡掩码自适应融合两种体素特征：

$$V_{fine} = \frac{(1 - V_{mask}) \cdot V_{agg} + V_t}{(1 - V_{mask}) + 1}$$

设计思想：
- **非遮挡区域**：优先使用聚合特征 $V_{agg}$（融合了多帧信息）
- **遮挡区域**：使用当前帧特征 $V_t$（历史帧在此区域信息不可靠）
- **归一化**：确保遮挡边界处特征平滑过渡，避免突变

OGVR 模块**零额外参数开销**，仅通过遮挡掩码的加权即可实现显著提升。

### 训练损失

总体损失函数包含四个部分：

$$\mathcal{L} = \lambda_{sem}\mathcal{L}_{scal}^{sem} + \lambda_{geo}\mathcal{L}_{scal}^{geo} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_d\mathcal{L}_d$$

- $\mathcal{L}_{scal}$：来自 MonoScene 的场景类亲和力损失（语义+几何）
- $\mathcal{L}_{ce}$：类别频率加权的交叉熵损失
- $\mathcal{L}_d$：深度分布的二值交叉熵监督（使用 LiDAR 投影）

## 实验结果

### SemanticKITTI 测试集

| 方法 | 输入 | IoU(%) | mIoU(%) |
|------|------|--------|---------|
| MonoScene | S | 34.16 | 11.08 |
| CGFormer | S | 44.41 | 16.63 |
| VoxFormer-T | T(5帧) | 43.21 | 13.41 |
| HTCL | T(3帧) | 44.23 | 17.09 |
| **FlowScene** | **T(2帧)** | **45.20** | **17.70** |

FlowScene 仅用 2 帧历史输入，相比 HTCL（3帧）提升 mIoU +0.61%、IoU +0.97%；相比最优单帧方法 CGFormer 提升 mIoU +1.07%。

### SSCBench-KITTI-360 测试集

| 方法 | IoU(%) | mIoU(%) |
|------|--------|---------|
| CGFormer | 48.07 | 20.05 |
| Symphonies | 44.12 | 18.58 |
| **FlowScene** | **46.95** | **20.81** |

在运动物体（car、truck、person 等）上优势尤为明显，动态物体 mIoU 达 14.87%，远超 CGFormer 的 11.37%。

### 不同距离范围性能（SemanticKITTI 验证集）

| 方法 | 12.8m | 25.6m | 51.2m |
|------|-------|-------|-------|
| VLScene | 26.51 | 24.37 | 17.83 |
| **FlowScene** | **27.63** | **24.65** | **18.13** |

在所有距离范围上均显著优于现有方法。

### 效率对比

| 方法 | mIoU(%) | 推理时间(s) | 参数量(M) |
|------|---------|------------|----------|
| HTCL | 17.13 | 0.297 | 181.4 |
| BRGScene | 15.43 | 0.285 | 161.4 |
| **FlowScene** | **18.13** | 0.301 | **52.4** |

参数量仅 52.4M，远低于 HTCL（181.4M），推理速度相当。

### 消融实验

| 配置 | IoU(%) | mIoU(%) |
|------|--------|---------|
| Baseline（简单堆叠） | 43.98 | 15.89 |
| +光流 Warping | 44.13 | 16.21 |
| +遮挡检测 | 44.38 | 16.43 |
| +时序聚合 | 44.63 | 17.23 |
| +遮挡交叉注意力 | 44.42 | 17.08 |
| Full（+OGVR） | **45.01** | **18.13** |

每个模块都有清晰贡献，总提升 mIoU +2.24%、IoU +1.03%。

**时序帧数消融**：2帧为最优平衡点。帧数增加时光流预测质量下降（帧间间隔过大），4-5帧反而降低性能。

**光流网络消融**：GMFlow > FlowFormer > RAFT > PWC-Net，且参数量最少（4.7M）。

**Backbone 消融**：RepViT-M2.3 > EfficientNetB7 > ResNet50，参数最少性能最好。

## 核心创新与局限

### 创新点

1. **光流引导的时序 SSC**：首次将光流信息系统性地引入 SSC 任务，而非简单的帧堆叠或位姿对齐
2. **遮挡感知的双空间建模**：在 2D（FGTA）和 3D（OGVR）空间中分别利用遮挡信息，形成完整的遮挡处理流水线
3. **高效设计**：仅需 2 帧历史输入和 52.4M 参数即达 SOTA，OGVR 模块零额外参数

### 局限性

1. 依赖预训练光流模型（GMFlow），光流估计误差会传播到下游
2. 仅在 KITTI 系列数据集验证，缺少 nuScenes 等多相机场景的实验
3. 历史帧数受限于光流质量——远距离帧的光流不可靠

## 个人思考

- 将光流引入 SSC 的思路直觉明确——光流编码了精确的像素级运动对应，比简单堆叠或位姿估计更能保持时序一致性
- 遮挡掩码的利用方式设计精巧：2D 空间用于过滤不可靠特征，3D 空间用于自适应融合，两个模块互补
- OGVR 的加权公式（公式7）虽然简单但有效，核心在于"非遮挡区用历史、遮挡区用当前"的分治策略
- 一个值得关注的趋势：相比需要 5 帧输入的方法，FlowScene 用 2 帧就达到更好效果，说明"质量>数量"——高质量的时序对齐比堆叠更多帧更重要
- 未来可探索将光流引导与 BEV 感知（如 BEVFormer）或占据网络（Occ3D）结合

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion](../../ECCV2024/autonomous_driving/hierarchical_temporal_context_learning_for_camera-based_semantic_scene_completio.md)
- [Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [Unleashing Semantic and Geometric Priors for 3D Scene Completion](../../AAAI2026/autonomous_driving/unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)
- [SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)

<!-- RELATED:END -->
