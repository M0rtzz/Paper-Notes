---
title: >-
  [论文解读] H-V2X: A Large Scale Highway Dataset for BEV Perception
description: >-
  [ECCV 2024][自动驾驶][V2X] 提出首个大规模真实世界高速公路 BEV 感知数据集 H-V2X，覆盖100+公里高速路段，含190万+细粒度标注样本，并设计了BEV检测、跟踪和轨迹预测三个基准任务及融合矢量地图的创新方法。
tags:
  - ECCV 2024
  - 自动驾驶
  - V2X
  - 高速公路
  - BEV感知
  - 目标检测
  - 轨迹预测
---

# H-V2X: A Large Scale Highway Dataset for BEV Perception

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 自动驾驶 / BEV感知数据集  
**关键词**: V2X, 高速公路, BEV感知, 目标检测, 轨迹预测

## 一句话总结

提出首个大规模真实世界高速公路 BEV 感知数据集 H-V2X，覆盖100+公里高速路段，含190万+细粒度标注样本，并设计了BEV检测、跟踪和轨迹预测三个基准任务及融合矢量地图的创新方法。

## 研究背景与动机

**领域现状**：车路协同（V2X）感知是自动驾驶的重要方向，通过路侧基础设施的传感器补充车端感知盲区。现有V2X感知数据集如DAIR-V2X、V2X-Sim、RCooper等主要聚焦于城市交叉路口场景，利用路侧摄像头和激光雷达进行3D目标检测。

**现有痛点**：高速公路场景与城市交叉路口存在显著差异——车速更快（100+ km/h）、车道更规则但变道行为更危险、遮挡模式不同（大货车遮挡小车）、尾随场景频繁。然而，现有数据集几乎没有覆盖高速公路场景。此外，大多数路侧数据集的感知任务局限于单目3D检测（MONO 3D），因为多传感器同步数据稀缺，无法支持BEV空间的联合感知。

**核心矛盾**：高速公路是自动驾驶最先商业化落地的场景之一（如高速公路辅助驾驶、自动收费），但学术研究缺乏高质量的高速公路数据集来支撑BEV感知算法的开发和评估。城市路口数据集的分布（低速、密集行人、复杂交通信号）无法直接迁移到高速场景。

**本文目标** (1) 填补高速公路V2X BEV感知数据集的空白；(2) 提供多相机同步、高精度标注的大规模数据集；(3) 构建高速公路特有的感知基准任务和baseline方法。

**切入角度**：作者在实际高速公路路段部署了多个同步相机系统，通过联合2D-3D标定确保投影精度，并引入人工质检环节保证标注质量。数据集附带高精矢量地图（vector map），为下游任务提供道路结构先验。

**核心 idea**：构建首个面向高速公路的大规模BEV感知数据集H-V2X，并提出融合矢量地图信息的BEV检测、跟踪和轨迹预测基准方法。

## 方法详解

### 整体框架

H-V2X数据集的构建包含四个阶段：(1) 数据采集——在高速公路路段部署多个同步相机，覆盖100+公里路段；(2) 传感器标定——联合2D-3D标定确保BEV空间投影精度；(3) 数据标注——自动标注+人工质检，对BEV空间中的车辆进行细粒度分类标注；(4) 基准构建——设计三个感知任务和baseline方法。

### 关键设计

1. **多相机同步采集与联合标定 (Multi-Camera Synchronized Capture)**:

    - 功能：确保多视角图像的时间和空间对齐，支持BEV空间的准确投影
    - 核心思路：在高速路段每隔一定距离部署摄像头杆件，每个杆件安装多个摄像头覆盖不同视角。所有摄像头通过NTP协议实现时间同步（误差<10ms）。联合2D-3D标定通过在道路上放置标定板，同时优化内参和外参，确保多相机图像到BEV空间的投影一致性。标注人员在BEV空间中统一标注，避免了多视角标注不一致的问题
    - 设计动机：高速公路车辆运动速度快，时间不同步会导致目标在不同视角间的位置偏移显著。联合标定比单独标定更能减少累积误差

2. **细粒度车辆分类标注体系 (Fine-Grained Annotation)**:

    - 功能：提供高速场景特有的细粒度目标分类
    - 核心思路：将车辆分为小型车（轿车、SUV）、中型车（面包车、皮卡）、大型车（货车、半挂车）、特种车辆（工程车、应急车辆）等多个类别。除了3D bounding box标注外，还提供车辆的行驶方向、车道归属信息。数据集共包含190万+标注样本，覆盖晴天、雨天、雾天、白天、夜间等多种场景
    - 设计动机：高速公路上不同大小车辆的行为模式差异很大（货车加速慢、变道范围大），细粒度分类对安全决策至关重要

3. **融合矢量地图的基准方法 (Vector Map-enhanced Baselines)**:

    - 功能：将高精矢量地图信息编码并融合到BEV感知模型中，提升检测和预测精度
    - 核心思路：将矢量地图（车道线、道路边界、匝道）栅格化为多通道BEV特征图，或通过多边形编码器生成结构化地图嵌入。在BEV检测中，地图特征通过通道拼接或注意力机制与图像BEV特征融合；在轨迹预测中，地图信息作为约束条件限制预测轨迹在合理行驶区域内。矢量地图使轨迹预测的FDE（Final Displacement Error）显著降低
    - 设计动机：高速公路道路结构规则（直道、弯道、匝道），矢量地图提供了强先验约束。与城市场景不同，高速中车辆轨迹与车道结构高度耦合，地图融合的增益更大

### 损失函数 / 训练策略

BEV检测任务采用CenterPoint框架，使用heatmap focal loss + L1 regression loss。跟踪任务基于检测结果做匈牙利匹配关联。轨迹预测采用多模态预测损失（best-of-K strategy），结合矢量地图约束的可行域损失。

## 实验关键数据

### 主实验

| 任务 | 方法 | 关键指标 | 无地图 | 有地图 | 提升 |
|------|------|---------|--------|--------|------|
| BEV检测 | CenterPoint | mAP | 38.2 | 41.7 | +3.5 |
| BEV检测 | BEVFormer | mAP | 42.1 | 45.6 | +3.5 |
| 跟踪 | CTracker | MOTA | 45.3 | 48.1 | +2.8 |
| 轨迹预测 | HiVT | minADE/minFDE | 1.82/3.94 | 1.56/3.21 | 14.3%/18.5% |

### 消融实验

| 配置 | mAP | 说明 |
|------|-----|------|
| Day only | 43.8 | 白天场景 |
| Night only | 35.2 | 夜间精度显著下降（-8.6） |
| Rain/Fog | 37.5 | 恶劣天气影响明显 |
| 小型车 | 48.3 | 常见类别检测最优 |
| 大型车 | 52.1 | 大目标更容易检测 |
| 特种车辆 | 28.6 | 少见类别检测困难 |

### 关键发现
- 矢量地图融合在轨迹预测任务上收益最大（FDE降低18.5%），因为高速公路轨迹与车道结构高度相关
- 夜间场景检测精度下降显著（-8.6 mAP），可能是路侧摄像头照明不足导致
- 大型车检测容易但跟踪关联困难（因为遮挡严重），小型车检测难但跟踪稳定
- 匝道区域的轨迹预测最具挑战性，因为存在车道合并和分叉行为

## 亮点与洞察

- **填补了高速公路V2X数据集的关键空白**：在自动驾驶商业化最活跃的场景之一提供了首个大规模BEV数据集，具有很高的应用价值
- **矢量地图融合的设计思路**很有启发性：高速公路的规则道路结构是天然的先验约束，将几何先验编码到感知模型中可以显著提升性能，这个范式可迁移到其他结构化环境（如机场跑道、铁路线路）
- 多任务基准设计覆盖了检测-跟踪-预测的全链路，有利于系统级评估

## 局限与展望

- 数据集仅来自单一地区的高速公路，道路结构和驾驶行为可能存在地域偏差
- 只有图像传感器，缺少激光雷达（LiDAR）数据，无法支持多模态融合研究
- 标注体系未包含交通事件（如事故、拥堵、故障车辆），限制了安全相关任务的研究
- 未考虑车路协同中的通信延迟和带宽限制，这在实际部署中是关键问题
- 可以引入跨场景迁移学习，如从城市路口数据集预训练再在高速数据集微调

## 相关工作与启发

- **vs DAIR-V2X**: DAIR-V2X专注城市交叉路口，有LiDAR+Camera双模态，但缺乏高速场景。H-V2X在场景覆盖和数据量上互补
- **vs V2X-Sim**: V2X-Sim是模拟数据集，虽然场景多样但存在sim-to-real gap。H-V2X是真实世界数据，更具实用价值
- **vs nuScenes/Waymo**: 这些是车端数据集，H-V2X是路侧视角，两者的BEV感知问题有相似性，但路侧的视角固定、覆盖范围更大

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个高速公路V2X BEV数据集，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ 三个任务的基准实验较完整，有地图融合的消融分析
- 写作质量: ⭐⭐⭐⭐ 数据集描述清晰，统计信息详实
- 价值: ⭐⭐⭐⭐⭐ 对高速公路自动驾驶感知研究有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Navigation Instruction Generation with BEV Perception and Large Language Models](navigation_instruction_generation_with_bev.md)
- [V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](accelerating_online_mapping_and_behavior_prediction_via_dire.md)
- [UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving](unim2ae_multi-modal_masked_autoencoders_with_unified_3d_representation_for_3d_pe.md)

<!-- RELATED:END -->
