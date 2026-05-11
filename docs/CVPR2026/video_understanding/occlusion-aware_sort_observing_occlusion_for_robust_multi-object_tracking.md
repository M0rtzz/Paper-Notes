---
title: >-
  [论文解读] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking
description: >-
  [CVPR 2026][视频理解][多目标跟踪] 提出遮挡感知跟踪框架 OA-SORT，通过显式建模目标遮挡状态来缓解位置代价混淆和 Kalman Filter 估计不稳定问题，在 DanceTrack/SportsMOT/MOT17 上均取得 SOTA 级提升，且组件可即插即用地集成到多种跟踪器中。
tags:
  - CVPR 2026
  - 视频理解
  - 多目标跟踪
  - 遮挡感知
  - Kalman Filter
  - 数据关联
  - 即插即用
---

# Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking

**会议**: CVPR 2026  
**arXiv**: [2603.06034](https://arxiv.org/abs/2603.06034)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 多目标跟踪, 遮挡感知, Kalman Filter, 数据关联, 即插即用

## 一句话总结

提出遮挡感知跟踪框架 OA-SORT，通过显式建模目标遮挡状态来缓解位置代价混淆和 Kalman Filter 估计不稳定问题，在 DanceTrack/SportsMOT/MOT17 上均取得 SOTA 级提升，且组件可即插即用地集成到多种跟踪器中。

## 研究背景与动机

**2D MOT 中遮挡带来位置代价混淆**：当同类别目标发生部分遮挡时，检测器难以区分前景与背景，导致不准确检测，进而使 IoU 代价矩阵产生歧义，引发身份交换（ID switch）。

**Kalman Filter 对不准确检测敏感**：离散线性 KF 在频繁接收遮挡导致的不准确检测后会累积误差，估计不稳定，尤其在非线性姿态变化场景более加严重。

**现有附加线索对遮挡仍脆弱**：外观特征在遮挡下可靠性下降（被前方目标特征污染）；运动方向虽帮助减少匹配失败但代价混淆仍在；检测置信度也对遮挡敏感。

**缺乏对遮挡状态的显式建模**：现有方法要么间接建模遮挡（如用活动/非活动状态），要么使用运动补偿恢复丢失目标，但都没有直接估计遮挡严重程度并利用它来修正关联代价。

**深度信息利用不充分**：虽有方法（PD-SORT, SparseTrack）利用伪深度设计关联策略，但仍受到遮挡引起的代价混淆影响。

**需要一个通用、无需训练的遮挡感知框架**：目标是设计 plug-and-play、training-free 的组件，可以轻松集成到各种跟踪器架构中提升鲁棒性。

## 方法详解

### 整体框架

OA-SORT 以 Hybrid-SORT 为基线，采用三阶段关联策略（高分检测关联→低分检测关联→丢失轨迹重连），在此基础上加入三个遮挡感知组件：

- **OAM (Occlusion-Aware Module)**：分析目标遮挡状态，生成遮挡系数
- **OAO (Occlusion-Aware Offset)**：将遮挡系数融入位置代价，缓解代价混淆
- **BAM (Bias-Aware Momentum)**：利用遮挡系数优化 KF 更新阶段，抑制估计波动

工作流程：(1) KF 预测后，OAM 基于估计计算遮挡系数；(2) 关联过程中 OAO 将遮挡系数融入空间一致性度量；(3) 与低分检测关联的轨迹通过 BAM 优化 KF 运动参数；(4) 当前帧结束前 OAM 基于最新观测计算遮挡系数供后续 BAM 使用。

### 关键设计

**1. 深度排序 (Depth Ordering)**：利用俯视相机下 bounding box 底边位置推断目标前后关系——底边 y 坐标越小说明越靠近相机。设置 5 像素阈值减少灵敏度。

**2. 遮挡系数 (Occlusion Coefficient)**：基于深度排序，计算被遮挡目标的遮挡面积占自身面积比例 $Oc_i = A(\mathcal{O}_i) / A(\mathcal{D}_i)$，支持多目标同时遮挡的并集处理。

**3. 高斯图精炼 (Gaussian Map, GM)**：遮挡系数可能高估遮挡严重程度（因 bbox 边缘包含大量背景像素），引入 2D 高斯权重图使靠近目标中心的像素具有更大权重，精炼后系数 $\hat{Oc}_i = \sum_{(x,y) \in \mathcal{O}_i} GM_{x,y} / A(\mathcal{D}_i)$。

**4. OAO 空间一致性评分**：将遮挡系数施加于 KF 估计（而非检测，因检测在遮挡下不稳定），与 IoU 加权组合：$S = \tau \cdot (1 - \hat{Oc}^X) + (1-\tau) \cdot C_{IoU}(\mathcal{D}, X)$，仅在第一阶段高分检测关联中触发。

**5. BAM 自适应动量**：对低分检测，利用轨迹最新观测的遮挡系数和 IoU 度量生成动量 $BAM = C_{IoU}(X_{t|t-1}, Z_t) \cdot (1 - \hat{Oc}^{Z_{t-1}})$，用动量对当前观测加权：$\hat{Z}_t = BAM \cdot Z_t + (1-BAM) \cdot H_t X_{t|t-1}$。遮挡越严重或检测偏差越大，越依赖 KF 估计。

### 损失函数/训练策略

本方法为 **training-free** 框架，无需额外训练或微调。所有超参数通过经验设定：

- GM 的 $\sigma^x, \sigma^y$ 按数据集运动模式调整（DanceTrack: $w/3\sqrt{2}, h/3$；SportsMOT: $w/4, h/3$；MOT17: $w/2, h/2$）
- OAO 的平衡系数 $\tau$ 在 0.1-0.2 之间调节（DanceTrack: 0.15, SportsMOT: 0.2, MOT17: 0.1）
- BAM 统一使用 HMIoU 作为空间一致性度量

## 实验关键数据

### 主实验

**DanceTrack test set（非线性运动+频繁遮挡）**

| 方法 | HOTA | AssA | MOTA | IDF1 |
|------|------|------|------|------|
| Hybrid-SORT | 62.2 | 47.4 | 91.6 | 63.0 |
| **OA-SORT** | **63.1** | **48.5** | **91.7** | **64.2** |
| OA-Byte (ByteTrack+OA) | 49.0 | 33.7 | 89.6 | 55.9 |
| OA-OC (OC-SORT+OA) | 56.5 | 39.6 | 91.2 | 57.6 |
| OA-Sparse (SparseTrack+OA) | 57.8 | 41.8 | 91.5 | 60.2 |
| OA-PD (PD-SORT+OA) | 60.4 | 44.9 | 91.4 | 60.8 |

**SportsMOT test set（变速运动+相机运动）**

| 方法 | HOTA | AssA | MOTA | IDF1 |
|------|------|------|------|------|
| Hybrid-SORT | 73.0 | 61.6 | 94.3 | 73.3 |
| **OA-SORT** | **73.4** | **62.3** | **94.4** | **74.1** |
| Hybrid-SORT* | 74.8 | 63.2 | 96.2 | 75.1 |
| **OA-SORT*** | **75.2** | **63.8** | **96.3** | **75.8** |

**MOT17 test set（线性运动+长期遮挡）**

| 方法 | HOTA | AssA | MOTA | IDF1 |
|------|------|------|------|------|
| Hybrid-SORT | 63.6 | 63.2 | 80.6 | 78.4 |
| **OA-SORT** | **64.2** | **64.0** | 79.6 | **79.1** |
| Hybrid-SORT-REID | 64.0 | 63.5 | 79.9 | 78.7 |

### 消融实验

**组件消融（DanceTrack-val）**

| OAO | BAM | GM | HOTA | AssA | IDF1 | 耗时增量 |
|-----|-----|----|------|------|------|----------|
| - | - | - | 59.4 | 44.9 | 60.7 | 9.57ms |
| ✓ | - | - | 59.9 | 45.6 | 60.9 | +1.96ms |
| - | ✓ | - | 60.5 | 46.5 | 62.2 | +3.81ms |
| ✓ | ✓ | - | 60.6 | 46.7 | 62.0 | +9.25ms |
| ✓ | ✓ | ✓ | **61.5** | **48.0** | **63.7** | +14.99ms |

**跨跟踪器泛化（DanceTrack-val，均加 OAO+BAM+GM）**

| 跟踪器 | 原 HOTA→新 HOTA | 原 IDF1→新 IDF1 |
|--------|-------------------|-------------------|
| SORT | 48.4→50.4 (+2.0) | 49.6→53.3 (+3.7) |
| ByteTrack | 47.1→49.3 (+2.2) | 52.7→55.7 (+3.0) |
| OC-SORT | 52.3→53.8 (+1.5) | 52.0→53.9 (+1.9) |
| TrackTrack | 59.3→59.9 (+0.6) | 61.1→62.0 (+0.9) |

### 关键发现

- GM 是最关键的组件，单独引入带来 +2.1 HOTA 提升，因为它有效抑制了 bbox 边缘背景像素对遮挡估计的干扰
- BAM 单独使用比 OAO 单独使用效果更好（+1.1 vs +0.5 HOTA），说明优化 KF 估计比修正关联代价更重要
- $\tau$ 超过约 0.2 后性能下降，过大的遮挡偏移会破坏空间一致性表达
- OA-SORT 无 ReID 甚至超过 Hybrid-SORT-REID（+0.5 AssA, +0.3 IDF1），说明遮挡感知可部分替代外观特征
- 遮挡越严重的视频序列获得的提升越大（见 Fig.1）

## 亮点与洞察

- **直接建模遮挡严重程度**：区别于间接遮挡处理方法，OAM 显式计算遮挡系数并将其注入关联和 KF 更新两个关键环节
- **Gaussian Map 精炼遮挡估计**：创新性地用 2D 高斯权重降低 bbox 边缘背景像素影响，是全文最大增益来源
- **完全即插即用、无需训练**：三个模块均可独立使用，已验证对 SORT/ByteTrack/OC-SORT/SparseTrack/PD-SORT/TrackTrack/BOT-SORT 七种跟踪器有效
- **思路清晰的问题分析**：Sec.3 从检测误差和估计误差累积的角度严格分析了遮挡如何引起代价混淆，为方法设计提供了坚实理论基础

## 局限性

- **底边深度排序假设受限**：当目标下半部被遮挡或目标腾空（如跳跃）时，底边位置无法准确反映深度关系，框架性能下降
- **缺乏长时遮挡建模**：遮挡状态及其时间变化是连续过程，当前方法仅利用瞬时遮挡信息，未建模遮挡的长期时序演变
- **GM 计算开销**：引入 GM 后每帧平均跟踪时间增加约 15ms，虽仍满足实时需求但在超大规模场景可能成为瓶颈
- **超参数需按数据集调整**：$\sigma^x, \sigma^y, \tau$ 需针对不同场景经验调参，泛化到全新场景需额外调试

## 相关工作

- **Position-Association 系列**：SORT → OC-SORT → Hybrid-SORT → PD-SORT/SparseTrack（利用伪深度），OA-SORT 在此链条上进一步引入显式遮挡建模
- **Feature-Association 系列**：StrongSORT++/BOT-SORT-REID 等利用外观特征增强关联，但遮挡下特征可靠性同样受损，OA-SORT 证明遮挡感知可部分替代 ReID
- **遮挡处理方法**：Stadler (活动/非活动状态间接建模)、Hibo (运动补偿恢复丢失目标)、DiffMOT (扩散模型预测)，均未直接估计遮挡程度

## 评分

- 新颖性: ⭐⭐⭐⭐ — 显式遮挡系数+高斯图精炼+双路注入思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 三数据集+七跟踪器集成+详细消融+参数分析
- 写作质量: ⭐⭐⭐⭐ — 问题分析到位，公式推导清晰，图表规范
- 价值: ⭐⭐⭐⭐ — plug-and-play 设计实用性强，提升幅度合理且一致

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/video_understanding/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)

</div>

<!-- RELATED:END -->
