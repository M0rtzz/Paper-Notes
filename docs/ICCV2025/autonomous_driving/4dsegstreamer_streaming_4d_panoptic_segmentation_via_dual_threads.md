---
title: >-
  [论文解读] 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads
description: >-
  [ICCV 2025][自动驾驶][4D全景分割] 提出4DSegStreamer，一种基于双线程系统（预测线程+推理线程）的流式4D全景分割框架，通过几何与运动记忆维护、自车位姿预测和逆向前向光流迭代实现实时高质量4D全景分割。
tags:
  - ICCV 2025
  - 自动驾驶
  - 4D全景分割
  - 流式感知
  - 双线程系统
  - 运动对齐
  - 点云序列
---

# 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads

**会议**: ICCV 2025  
**arXiv**: [2510.17664](https://arxiv.org/abs/2510.17664)  
**代码**: https://github.com/llada60/4DSegStreamer (有)  
**领域**: 自动驾驶  
**关键词**: 4D全景分割, 流式感知, 双线程系统, 运动对齐, 点云序列

## 一句话总结

提出4DSegStreamer，一种基于双线程系统（预测线程+推理线程）的流式4D全景分割框架，通过几何与运动记忆维护、自车位姿预测和逆向前向光流迭代实现实时高质量4D全景分割。

## 研究背景与动机

4D全景分割（4D Panoptic Segmentation）旨在对连续点云序列进行实例级别和语义级别的密集感知。在自动驾驶等高度动态场景中，需要在严格的时间预算内处理每一帧点云数据，即**流式感知**（Streaming Perception）。

现有方法面临的核心挑战：

**计算延迟问题**：现有4D分割方法（Mask4Former、Eq-4D-StOP等）计算量大，无法满足实时约束，在streaming设置下性能严重退化

**粒度不足**：已有的streaming感知研究主要集中在2D/3D目标检测和跟踪上，提供的bounding box信息不足以支撑下游决策（如识别施工区域、人行道等）

**动态对象处理不当**：简单的时序特征融合方法无法有效处理高度动态场景中的运动对象，尤其在高帧率条件下性能下降严重

作者认为，设计一个通用的系统将已有segmentation方法"包装"成streaming模式，比重新训练一个实时model更高效、更灵活。

## 方法详解

### 整体框架

4DSegStreamer采用**双线程系统**（Dual-Thread System），将streaming帧分为关键帧和非关键帧：

- **预测线程（Predictive Thread）**：在关键帧上提取几何和运动特征，持续更新记忆，并利用历史信息预测未来动态
- **推理线程（Inference Thread）**：对每个到来的非关键帧，通过与最新记忆的几何对齐快速检索特征，实现实时推理

两个线程并行运行、共享记忆，推理延迟主要取决于轻量级的推理线程。

### 关键设计

**1. 几何记忆更新（Geometric Memory Update）**

使用稀疏ConvGRU机制更新几何记忆。当新关键帧到来时：
- 先通过运动对齐将前一帧的记忆状态转换到当前帧坐标系
- 再用当前帧特征通过GRU门控机制更新记忆（包含激活门z_t和重置门r_t）

支持与现有3D和4D分割backbone的无缝集成。

**2. 自车位姿未来对齐（Ego-pose Future Alignment）**

提供两种设置：
- **已知位姿**：直接使用相对位姿对齐
- **未知位姿**：使用Suma++估计关键帧间的相对位姿，通过LSTM维护位姿记忆，多头预测器预测未来位姿

**3. 动态目标未来对齐（Dynamic Object Future Alignment）**

- **未来光流预测**：训练时使用FastNSF获取监督GT光流，推理时使用轻量级zeroFlow估计关键帧间的光流，输入LSTM预测未来光流
- **逆向前向光流迭代（Inverse Forward Flow Iteration）**：核心创新点。直接用前向光流对memory做变换需要重建KD-Tree，代价高；直接预测后向光流因为未来点位置未知，效果差。本文提出迭代方法：对查询点y，迭代求解 x_{n+1} = y - flow(x_n)，在Lipschitz连续条件下收敛到不动点

### 损失函数 / 训练策略

- 分两阶段训练：先正常训练backbone分割模型，然后冻结backbone，训练位姿预测、光流预测和记忆聚合模块
- 逆向光流迭代的最大迭代次数设为10，收敛阈值控制精度
- 对移动物体使用moving mask，仅对动态对象赋予非零光流

## 实验关键数据

### 主实验

SemanticKITTI未知位姿streaming设置：

| 方法 | sLSTQ | sPQ | sPQ_d | sPQ_s |
|------|-------|-----|-------|-------|
| StreamYOLO | 0.415 | 0.373 | 0.429 | 0.371 |
| Mask4Former | 0.515 | 0.485 | 0.571 | 0.413 |
| PTv3 | 0.536 | 0.567 | 0.638 | 0.464 |
| **4DSegStreamer (M4F)** | **0.688** | **0.634** | **0.744** | **0.486** |

nuScenes已知位姿streaming设置：

| 方法 | sLSTQ | sPQ | sPQ_d | sPQ_s |
|------|-------|-----|-------|-------|
| Eq-4D-StOP | 0.695 | 0.673 | 0.654 | 0.693 |
| **4DSegStreamer (M4F)** | **0.765** | **0.751** | **0.734** | **0.786** |

### 消融实验

组件逐步添加消融（SemanticKITTI未知位姿）：

| 配置 | sLSTQ | sLSTQ_d | sLSTQ_s |
|------|-------|---------|---------|
| P3Former baseline | 0.304 | 0.265 | 0.357 |
| + Memory | 0.349 | 0.292 | 0.408 |
| + Memory + Pose | 0.497 | 0.488 | 0.501 |
| + Memory + Pose + Flow | 0.591 | 0.667 | 0.514 |
| + Memory + Pose + Moving Flow | **0.613** | **0.682** | **0.516** |

光流预测策略对比：

| 方法 | sLSTQ | sLSTQ_d | sLSTQ_s |
|------|-------|---------|---------|
| Backward flow | 0.565 | 0.637 | 0.483 |
| Forward flow | 0.589 | 0.667 | 0.497 |
| Inverse flow iteration | **0.613** | **0.682** | **0.516** |

### 关键发现

1. 相比PTv3，在SemanticKITTI未知位姿设置下sLSTQ提升15.2%（0.536到0.688），提升巨大
2. 动态目标感知（sPQ_d）提升尤为显著，说明运动对齐对动态对象效果突出
3. 逆向光流迭代相比直接前向/后向光流有明显优势，平衡了精度与效率
4. 框架具有通用性：可无缝集成到P3Former、Mask4Former、Eq-4D-StOP等多种backbone
5. 室内数据集HOI4D上同样领先runner-up 6.6% sLSTQ

## 亮点与洞察

- **双线程设计的工程智慧**：将heavy computation放入predictive thread，轻量推理放入inference thread，天然适合实时系统
- **逆向前向光流迭代**是本文最核心的技术贡献，巧妙解决了前向光流需重建数据结构、后向光流预测困难的两难问题
- **即插即用**：可以为任意3D/4D分割方法赋予streaming能力，实用价值高
- 首次系统化定义和评估了streaming 4D panoptic segmentation任务

## 局限与展望

1. 位姿预测依赖Suma++，在极端运动场景下可能不够准确
2. 光流预测的zeroFlow是蒸馏模型，精度有上限
3. 逆向光流迭代的收敛性依赖Lipschitz条件，在极快运动场景可能不满足
4. 仅在LiDAR点云上验证，未扩展到camera-based方法
5. 记忆大小随时间增长，long-term场景可能需要记忆淘汰策略

## 相关工作与启发

- 与DriveVLM-Dual的快慢系统思想类似，但4DSegStreamer将快慢组件统一为一个pipeline
- 记忆机制（ConvGRU）借鉴了NSM4D和MemorySeg的设计
- 流式感知与2D任务（StreamYOLO、DAMO-StreamNet）的思路一脉相承，但扩展到更难的密集分割任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双线程系统和逆向光流迭代设计新颖，但整体框架是已有组件的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 3个数据集、多种backbone、详细消融、多种setting
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 首次定义streaming 4D panoptic segmentation，实用价值很高

<!-- RELATED:START -->

## 相关论文

- [DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models](eta_efficiency_through_thinking_ahead_a_dual_approach_to_self-driving_with_large.md)
- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)

<!-- RELATED:END -->
