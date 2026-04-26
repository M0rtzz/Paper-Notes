---
title: >-
  [论文解读] EBS-EKF: Accurate and High Frequency Event-based Star Tracking
description: >-
  [CVPR 2025][视频理解][事件相机] 本文提出 EBS-EKF，通过建模事件相机在低光条件下的电路行为来获得亮度依赖的质心偏移校正，结合 3D 扩展卡尔曼滤波进行星跟踪，在真实夜空数据上比现有方法精确一个数量级。
tags:
  - CVPR 2025
  - 视频理解
  - 事件相机
  - 星跟踪
  - 扩展卡尔曼滤波
  - 低光信号建模
  - 姿态估计
---

# EBS-EKF: Accurate and High Frequency Event-based Star Tracking

**会议**: CVPR 2025  
**arXiv**: [2503.20101](https://arxiv.org/abs/2503.20101)  
**代码**: 有（开源代码和数据集）  
**领域**: 视频理解 / 事件相机  
**关键词**: 事件相机, 星跟踪, 扩展卡尔曼滤波, 低光信号建模, 姿态估计

## 一句话总结
本文提出 EBS-EKF，通过建模事件相机在低光条件下的电路行为来获得亮度依赖的质心偏移校正，结合 3D 扩展卡尔曼滤波进行星跟踪，在真实夜空数据上比现有方法精确一个数量级。

## 研究背景与动机

**领域现状**：星跟踪器是航天器姿态确定的标准传感器，传统方法使用主动像素传感器（APS）相机，精度可达角秒级别。近年来，事件相机（EBS）因其微秒级延迟和低功耗，成为星跟踪的有前途的新技术。

**现有痛点**：APS 跟踪器受曝光时间和帧处理开销限制，更新率通常为 2-10 Hz，在快速姿态调整时受限。现有 EBS 星跟踪方法（ICP、Hough、2D-KF）全部仅在仿真数据上评估，且使用简化的信号模型。Chin et al. 的 ICP 方法容易产生噪声姿态估计和漂移；Ng et al. 的 2D-KF 方法仅建模 2D 平移，忽略了 roll 的非线性效应。

**核心矛盾**：现有 EBS 方法未考虑事件相机在低光条件下的电路特性——暗星会导致事件质心相对于真实质心产生与亮度相关的偏移，这在传统的高斯信号模型中被忽略。此外，2D 状态估计无法正确处理真实数据中的 3D 旋转动力学。

**本文目标**：(1) 建模 EBS 低光电路行为，提出亮度依赖的质心校正；(2) 设计 3D EKF 进行高频高精度姿态估计；(3) 在真实夜空数据上量化评估。

**切入角度**：作者发现 EBS 像素在低光下带宽与光电流成正比，导致暗星的事件分布滞后于真实位置。通过分析电路微分方程，可以预测这一偏移并进行校正。

**核心 idea**：用 EBS 电路低光模型推导亮度依赖的质心偏移校正函数，配合 3D 扩展卡尔曼滤波实现毫秒级更新的高精度星跟踪。

## 方法详解

### 整体框架
输入为事件相机的正极性事件流，输出为相机的 3D 姿态（四元数 + 角速度）。系统分为三步：(1) 初始化：用短时间窗口积分事件，DBSCAN 聚类找到星簇，通过 Astrometry.net 匹配星表获取初始姿态；(2) 逐事件更新：每收到一个正事件，EKF 预测当前状态，检查事件是否在某颗已知星附近，若是则施加亮度偏移校正后进行 EKF 更新；(3) 输出 1 KHz 的姿态估计。

### 关键设计

1. **低光事件似然模型**:

    - 功能：建模事件相机在低光条件下事件发生的空间分布
    - 核心思路：传统模型假设事件似然 $E(t) = d\log(I(t)/I_0 + 1)/dt$，但在低光下 EBS 电路带宽与光电流成正比，像素电压变为一阶低通滤波器。作者推导出低光事件似然 $E_{LL}(t) = 2\pi \cdot f_c(\tilde{I}(t)) \cdot [\tilde{I}(t) - V(t)]$，其中 $f_c \approx b + a\tilde{I}(t)$。关键发现：暗星正事件质心滞后于真实位置，偏移量 $z(m_s)$ 是星等的函数。最终用高斯分布近似，加入偏移校正项。
    - 设计动机：这解释了为何现有方法质心估计不准——它们忽略了低光电路效应。校正项使质心精度从 ~3 像素提升到 ~0.4 像素。

2. **3D 扩展卡尔曼滤波（EKF）**:

    - 功能：从异步事件流递归估计相机的 3D 旋转和角速度
    - 核心思路：状态 $\mathbb{S} = [\mathbf{q}, \omega]$ 包含四元数和 3D 角速度。使用常速度先验 $q_{t+1} = \exp(\Delta t \omega) \cdot q_t$ 作为前向模型。测量模型将星的 3D 坐标通过四元数旋转 + 针孔投影映射到 2D 像素，推导出完整的雅可比矩阵 $\mathbf{F}$、过程噪声 $\mathbf{Q}$ 和测量矩阵 $\mathbf{H}$。使用 boxplus 算子处理 $SO(3) \times \mathbb{R}^3$ 的复合流形。
    - 设计动机：2D-KF 仅估计平移和速度，忽略 roll，在真实 3D 旋转数据上频繁漂移。3D EKF 直接建模旋转群，无需频繁的绝对测量重置。

3. **亮度依赖质心偏移校正**:

    - 功能：消除亮星和暗星质心估计的系统性偏差
    - 核心思路：从低光似然模型数值求解得到偏移曲线 $z(m_s)$，将其作为查表函数在 EKF 更新时校正事件位置：$\mathcal{L}(e_i | \mathbf{x_0}, \mathbf{v}) \sim \mathcal{N}(\mathbf{x}_i - [\mathbf{x}_0 - \bar{\mathbf{v}} \cdot z(m_s)], \sigma_s^2 \mathbf{I})$。校正参数通过夜空数据标定（$I_0=1, a=20$ Hz/intensity, $b=2$ Hz），并在实验室 LCD 数据上验证泛化。
    - 设计动机：偏移可达 1-2 像素（~30-60 角秒），亮星尤其严重（存在"bow-shock"效应），不校正会导致姿态跳变。

### 损失函数 / 训练策略
本方法为非学习方法，核心是贝叶斯估计：$p(\mathbb{S}_i | \mathcal{E}_i) \propto \mathcal{L}(e_i | \mathbb{S}_i) \cdot p(\mathbb{S}_i | \mathbb{S}_{i-1})$，通过 EKF 递归求解。

## 实验关键数据

### 主实验：真实夜空跟踪精度（角秒，越小越好）

| 方法 | Vel. Sweep 1 (across/about) | Multipose 1 (across/about) | Tilt Ladder (across/about) |
|------|------|------|------|
| ICP | 3300/395 | 1600/2200 | 2300/6200 |
| Hough | 8600/87000 | 11000/57000 | 11000/65000 |
| 2D-KF | 229/1300 | 337/7300 | 485/9200 |
| EBS-EKF (ours) | 25.8/60.3 | 57.4/52.8 | 49.1/64.5 |

### 消融实验：偏移校正效果

| 配置 | 典型精度变化 | 说明 |
|------|---------|------|
| EBS-EKF w/ offset | 25.8/60.3 (Vel.1) | 完整模型 |
| EBS-EKF w/o offset | 27.0/83.1 (Vel.1) | 去掉偏移校正后 about 方向增加 ~20 角秒 |
| Vel. Sweep 7（含亮星 Vega） | 170.1/139.0 vs 172.1/322.0 | 亮星场景下偏移校正使 about 误差降低 180+ 角秒 |

### 关键发现
- 本方法在真实夜空上通常在 100 角秒以内，而现有方法偏差达千角秒甚至度级别
- 2D-KF 是最接近的竞争者，但因 2D 建模不足而周期性漂移，需频繁绝对测量重置
- 偏移校正在视场内有亮星时效果显著（Vel. Sweep 7 中 Vega 出现时，约改善 180 角秒）
- 在 7.5°/s 高速轨迹上（超出 APS 跟踪器 3°/s 限制），本方法仍能正确重建轨迹，总误差 80.4 角秒 vs 2D-KF 的 774.3 角秒

## 亮点与洞察
- **EBS 电路物理建模**：从底层电路行为推导事件似然，发现亮度依赖偏移并提出校正——这是一个"从物理到算法"的精彩示范，比纯数据驱动的方法更具可解释性。
- **首个真实数据基准**：之前所有 EBS 星跟踪工作均在 LCD 屏幕模拟评估，本文首次在真实夜空上量化性能并提供同步数据集，对该领域具有里程碑意义。
- **可迁移思路**：亮度依赖的质心偏移校正可推广到其他低光 EBS 应用场景（如夜间自动驾驶中的弱光点目标检测）。

## 局限与展望
- **地面数据局限**：地面采集的数据包含大气折射变化（尤其靠近地平线时），空间中不会有这个问题。作者通过与 APS 跟踪器对比获得间接评估
- **偏移校正是全局参数**：当前偏移曲线对所有星速度使用同一曲线，理论上星速度也会影响偏移，但实验中差异不大
- **初始化依赖 Astrometry.net**：初始姿态需通过静态积分 + 星表匹配获得，快速启动场景可能受限
- **仅处理正事件**：暗星负事件熵高、延迟大，直接丢弃。未来可探索利用负事件进一步提升性能

## 相关工作与启发
- **vs Ng et al. (2D-KF)**：2D-KF 用 2D 平移 + 速度建模，忽略 roll 和 3D 非线性效应，需频繁绝对测量；本文 3D EKF 直接建模旋转群，仅需初始化一次
- **vs Chin et al. (ICP)**：ICP 将事件积分为帧后配准，精度受噪声影响大且容易漂移；本文逐事件异步更新，更新率更高
- **vs APS 星跟踪器**：APS 精度 5 角秒 across / 55 角秒 about，但更新率仅 2 Hz 且受限于 3°/s。本文 EBS 方法在 1 KHz 更新率下实现可比精度，且容忍 7.5°/s 高速旋转

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从 EBS 电路物理出发推导质心校正模型，结合 3D EKF，思路清晰且创新性强
- 实验充分度: ⭐⭐⭐⭐⭐ 14 条真实夜空轨迹 + LCD 实验室对照，与 3 个基线方法全面对比，消融分析充分
- 可复现性: ⭐⭐⭐⭐⭐ 代码和数据集全部开源，实验细节详尽
- 写作质量: ⭐⭐⭐⭐⭐ 从物理建模到算法推导到实验评估逻辑严密，补充材料极为详尽
- 价值: ⭐⭐⭐⭐ 对 EBS 星跟踪领域有重要推动，首次提供真实数据基准；但应用场景局限于航天领域

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] ETAP: Event-based Tracking of Any Point](etap_event-based_tracking_of_any_point.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](../../ICCV2025/video_understanding/hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](../../CVPR2026/video_understanding/event6d_event-based_novel_object_6d_pose_tracking.md)

<!-- RELATED:END -->
