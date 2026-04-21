---
title: >-
  [论文解读] OmniTrack: Omnidirectional Multi-Object Tracking
description: >-
  [CVPR 2025][视频理解][全向跟踪] 提出首个面向 360° 全景图像的多目标跟踪框架 OmniTrack，统一 TBD 和 E2E 两种跟踪范式，通过 CircularStatE 模块缓解全景畸变、FlexiTrack 实例引入时序先验、Tracklet Management 提供轨迹反馈，并构建 QuadTrack 四足机器人全景 MOT 数据集。
tags:
  - CVPR 2025
  - 视频理解
  - 全向跟踪
  - 全景图像
  - 四足机器人
  - 几何畸变
  - 多目标跟踪
---

# OmniTrack: Omnidirectional Multi-Object Tracking

**会议**: CVPR 2025  
**arXiv**: [2503.04565](https://arxiv.org/abs/2503.04565)  
**代码**: [GitHub](https://github.com/xifen523/OmniTrack)  
**领域**: 视频理解/多目标跟踪  
**关键词**: 全向跟踪, 全景图像, 四足机器人, 几何畸变, 多目标跟踪

## 一句话总结

提出首个面向 360° 全景图像的多目标跟踪框架 OmniTrack，统一 TBD 和 E2E 两种跟踪范式，通过 CircularStatE 模块缓解全景畸变、FlexiTrack 实例引入时序先验、Tracklet Management 提供轨迹反馈，并构建 QuadTrack 四足机器人全景 MOT 数据集。

## 研究背景与动机

全景相机以 360° 视场角提供全面的环境信息，对自动驾驶、机器人导航等应用极具价值。然而全景 MOT 面临独特挑战：

- **几何畸变**：全景图像展开（等矩形投影）后高纬度区域存在严重的几何形变
- **分辨率损失**：360° 信息压缩到单帧图像，有效分辨率降低
- **光照/色彩不均**：不同方向的光照条件差异导致图像特征不一致
- **剧烈运动**：四足机器人的仿生步态引入复杂的非线性相机运动

现有 MOT 算法（如 ByteTrack, OC-SORT）针对针孔相机设计，直接应用于全景图像时性能显著下降。此前缺乏专门的全景 MOT 框架和包含剧烈运动的全景数据集。

## 方法详解

### 整体框架

OmniTrack 采用反馈机制设计：FlexiTrack 实例利用历史轨迹信息引导检测器聚焦关键区域，CircularStatE 模块处理全景畸变，Tracklet Management 管理轨迹数据并提供先验知识。统一设置下——禁用数据关联得到 E2E 跟踪器 OmniTrackE2E，启用则得到 TBD 跟踪器 OmniTrackDA。

### 关键设计一：CircularStatE 模块

- **功能**：缓解全景图像的广角畸变、分辨率损失和光照不一致
- **核心思路**：利用全景图像的循环特性（左右边缘实际上是相连的），在特征层面进行循环填充和统计特征增强，使不同区域的检测一致性得到改善
- **设计动机**：标准 CNN 的零填充在全景图像边缘引入不连续性，破坏了全景的循环几何。循环填充利用全景的天然拓扑结构

### 关键设计二：FlexiTrack 实例

- **功能**：利用时序先验引导当前帧的目标检测和关联
- **核心思路**：从 Tracklet Management 获取上一帧的轨迹信息 $\mathcal{I}_F$，与 CircularStatE 处理后的特征图 $\mathcal{I}_L$ 共同输入 Decoder，分别输出 FlexiTrack 检测 $\mathcal{D}_k^F$（基于轨迹先验）和学习检测 $\mathcal{D}_k^L$（基于当前帧特征）
- **设计动机**：全景视场角大，搜索空间庞大。利用历史轨迹作为先验可以显著缩小搜索范围，提高在快速运动场景下的跟踪稳定性

### 关键设计三：QuadTrack 数据集

- **功能**：提供包含复杂运动动态的全景 MOT 基准
- **核心思路**：使用 $360° \times 70°$ 全景相机安装在四足机器人上，在两个城市五个校区采集 19,200 张图像。四足机器人的仿生步态引入现实复杂的非线性运动特征
- **设计动机**：现有 MOT 数据集使用静态或线性运动平台，无法评估全景视场角+剧烈运动下的跟踪性能

### 损失函数

使用标准的检测损失（分类+回归）和匹配损失进行端到端训练。数据关联采用匈牙利算法。

## 实验关键数据

### 主实验：JRDB 数据集

| 方法 | HOTA ↑ | MOTA ↑ | IDF1 ↑ |
|------|--------|--------|--------|
| ByteTrack | 18.40 | - | - |
| OC-SORT | 23.49 | - | - |
| **OmniTrackDA** | **26.92** | - | - |

### QuadTrack 数据集

| 方法 | HOTA ↑ | 提升 |
|------|--------|------|
| Baseline | 16.64 | - |
| **OmniTrackDA** | **23.45** | +6.81% |

### 消融实验

| 配置 | HOTA |
|------|------|
| 基线（标准检测器） | 最低 |
| + CircularStatE | +提升 |
| + FlexiTrack | +显著提升 |
| + 完整框架 | **最优** |

### 关键发现

- JRDB 上 HOTA 26.92%，超越 OC-SORT 3.43 个百分点
- QuadTrack 上超越基线 6.81%，在剧烈运动场景下优势更明显
- OmniTrackDA（TBD 范式）持续优于 OmniTrackE2E（E2E 范式），数据关联仍然重要
- CircularStatE 对高纬度区域目标检测的改善最为显著

## 亮点与洞察

1. **首个全景 MOT 框架**：系统性地解决了全景图像 MOT 的关键挑战
2. **统一 TBD 和 E2E**：通过开关设计实现两种范式的统一框架，便于公平比较
3. **QuadTrack 数据集**：四足机器人的剧烈非线性运动提供了极具挑战性的新基准

## 局限与展望

- HOTA 绝对值仍不高（26.92%），全景 MOT 是一个极其困难的问题
- 四足机器人的运动模式特殊，泛化性有待验证
- 未探索 3D MOT 或深度信息的融合
- QuadTrack 数据集规模相对较小

## 相关工作与启发

- **ByteTrack**：基于置信度的分阶段关联策略
- **OC-SORT**：优化的运动估计模块
- **360VOT**：全向单目标跟踪基准
- 全景感知的循环特性处理策略可推广到其他全景视觉任务

## 评分

⭐⭐⭐⭐ — 首次系统性地解决全景 MOT 问题，框架设计合理。QuadTrack 数据集填补了重要空白。但绝对性能仍有很大提升空间。

<!-- RELATED:START -->

## 相关论文

- [MITracker: Multi-View Integration for Visual Object Tracking](mitracker_multi-view_integration_for_visual_object_tracking.md)
- [FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)
- [Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](../../CVPR2026/video_understanding/occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)

<!-- RELATED:END -->
