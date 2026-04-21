---
title: >-
  [论文解读] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision
description: >-
  [CVPR 2026][自动驾驶][事件相机] 提出 FlashCap，首个基于闪烁 LED + 事件相机的运动捕捉系统，通过为每个 LED 配置不同的闪烁频率实现身份识别，构建了首个 1000Hz 标注精度的人体运动数据集 FlashMotion（715 万帧），并提出 ResPose 基线方法，将运动定时误差从 ~50ms 降至 ~5ms，姿态估计 MPJPE 降低约 40%。
tags:
  - CVPR 2026
  - 自动驾驶
  - 事件相机
  - 人体运动捕捉
  - LED标记
  - 高时间分辨率
  - 脉冲神经网络
---

# FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision

**会议**: CVPR 2026  
**arXiv**: [2603.19770](https://arxiv.org/abs/2603.19770)  
**代码**: 即将公开  
**领域**: 自动驾驶 / 人体姿态估计  
**关键词**: 事件相机, 人体运动捕捉, LED标记, 高时间分辨率, 脉冲神经网络

## 一句话总结

提出 FlashCap，首个基于闪烁 LED + 事件相机的运动捕捉系统，通过为每个 LED 配置不同的闪烁频率实现身份识别，构建了首个 1000Hz 标注精度的人体运动数据集 FlashMotion（715 万帧），并提出 ResPose 基线方法，将运动定时误差从 ~50ms 降至 ~5ms，姿态估计 MPJPE 降低约 40%。

## 研究背景与动机

1. **领域现状**：精确运动定时（PMT）在竞技体育等场景至关重要——雪橇比赛中 2ms 差距可能决定奖牌归属。当前人体姿态估计（HPE）主要关注空间准确性，对时间精度关注不足。现有运动捕捉系统如 Vicon（光学标记，~330Hz）、Xsens（IMU，60-240Hz）、标准 RGB 相机（30-60Hz）的时间分辨率均无法满足毫秒级需求。
2. **现有痛点**：(a) 高速 RGB 相机（≥1000Hz）虽可实现高帧率但成本极高（NAC HX-7s 超过 45,000 美元，是事件相机的 9 倍）、需要极强照明、带宽和存储需求比事件相机高两个数量级；(b) 公开的人体运动数据集最高标注帧率仅 120Hz（BEAHM），距离毫秒级精度差一个量级；(c) 现有时间标注方法受辅助模态采样上限或插值误差限制，无法突破 120Hz。
3. **核心矛盾**：如何以低成本、低带宽实现 1000Hz 的高时间分辨率人体运动捕捉和标注？
4. **本文目标** (a) 构建一种新型低成本运动捕捉系统，绕过高速相机的瓶颈；(b) 收集首个 1000Hz 标注精度的多模态人体运动数据集；(c) 提出并评估高时间分辨率的 HPE 基线方法。
5. **切入角度**：事件相机拥有微秒级时间分辨率和极低带宽，但如何从事件流中获取高频地面真值标注是关键挑战。作者创造性地使用不同闪烁频率的 LED 作为身体标记——事件相机能精确捕获 LED 的闪烁模式，通过频率分析自动匹配 LED 身份和位置，直接从事件流生成 1000Hz 的 2D 关节位置标注。
6. **核心 idea**：用不同闪烁频率的 LED 编码关节身份，事件相机天然高时间分辨率地捕获闪烁模式，频率匹配算法自动生成 1000Hz 的姿态标注——低成本、低带宽、无需高速相机。

## 方法详解

### 整体框架

FlashCap 系统分三部分：(1) **运动捕捉装（MoCap Outfit）**：包含 17 个 LED 和 17 个 IMU 的穿戴服装；(2) **多模态采集设备**：事件相机（Prophesee 1280×720）+ RGB 相机（Hikrobot 1920×1200, 20fps）+ LiDAR（Ouster OS-1 128线, 20fps），通过分光镜实现事件-RGB 像素对齐和时间同步；(3) **数据标注流水线**：从事件流中自动识别 LED 闪烁模式并匹配身份，生成 1000Hz 的 2D 关节位置标注。基于此系统收集 FlashMotion 数据集，并提出 ResPose 基线方法进行高时间分辨率 HPE。

### 关键设计

1. **LED 闪烁编码与身份识别**

    - 功能：通过独特的闪烁频率为每个关节 LED 赋予身份标识，使事件相机能自动区分不同关节。
    - 核心思路：每个 LED $i$ 以可配置频率（~4000Hz）闪烁，具有不同的开时间 $t_i^p$ 和关时间 $t_i^n$（100-300μs 范围），形成独特的闪烁签名。事件相机异步触发事件 $e=(h,w,t,p)$，其中高密度事件区域对应 LED 位置。标注流水线四步：(a) **事件聚类识别**：将事件流分割为 1ms 事件帧，用 DBSCAN 聚类识别高密度区域；(b) **频率识别**：分析每个聚类的正负极性事件序列，计算平均开/关时间 $\bar{t_j^p}$、$\bar{t_j^n}$ 和闪烁周期 $\bar{T_j}$；(c) **噪声过滤**：时域平滑 + 异常值过滤；(d) **LED-聚类匹配**：计算距离 $d_{ji} = \alpha \cdot d_{ji}^t + \beta \cdot d_{ji}^p$（开关时间距离 + 周期距离），用二部图匹配算法找最优对应。
    - 设计动机：不同于传统光学标记（需相机高帧率捕获）或 RFID（精度不足），LED 闪烁频率编码天然适配事件相机的异步工作方式——只要亮度变化超过阈值就会触发事件，且事件时间戳精度可达微秒级。

2. **FlashMotion 数据集**

    - 功能：提供首个 1000Hz 标注精度的多模态人体运动数据集。
    - 核心思路：20 名志愿者（10 男 10 女），4 个场景（室内外），11 大类 19 小类动作，240 个序列。包含 144,350 RGB 帧、144,350 LiDAR 点云帧、2 小时事件和 IMU 数据。**2D 标注 1000Hz**（由标注流水线自动生成 + 人工修正），**3D 标注 60Hz**（IMU + LiDAR 融合得到 SMPL 参数）。总标注帧数 715 万帧，是现有数据集的数量级提升。
    - 设计动机：现有 HPE 数据集的标注帧率受限于传统光学系统（最高 120Hz），而 FlashCap 的 LED 方案直接从事件流生成原生 1000Hz 标注，绕过了传统光学 MoCap 的频率瓶颈。

3. **ResPose：残差姿态估计基线**

    - 功能：利用低帧率 RGB 提供的结构先验和高帧率事件捕获的微运动，实现 1000Hz 姿态估计。
    - 核心思路：高分辨率姿态 $P_i = P_{rgb} + P_i^{\Delta}$，其中 $P_{rgb}$ 是低帧率 RGB 分支（如 ViTPose）提供的锚定姿态，$P_i^{\Delta}$ 是事件分支估计的残差姿态。事件分支使用 **SNN-CNN 混合编码器**：以 RGB 锚点为中心动态裁剪 $32 \times 32$ 局部事件补丁，用 Leaky Integrate-and-Fire (LIF) 脉冲神经元做时序积分，$1 \times 1$ 卷积抑制背景噪声。**多模态残差 Transformer**：将 RGB 锚点特征和事件特征拼接后输入 Transformer 编码器，利用全局自注意力建模 17 个关节间的运动学约束。端到端 L2 损失训练。
    - 设计动机：高频事件流包含的是微小运动变化（残差信号），而非完整的空间结构。将 RGB 帧作为结构锚点、事件作为残差修正是自然且高效的分解方式。SNN 天然适合处理异步事件数据。

### 损失函数 / 训练策略

ResPose 使用端到端的 L2 距离损失，最小化预测姿态与 1000Hz 地面真值之间的误差。RGB 分支使用预训练的 ViTPose，事件分支从零训练。

## 实验关键数据

### 主实验

精确运动定时（PMT）——估计关节穿越线的时间误差（ms）：

| 方法 | Kicking | Punching | Jumping |
|------|---------|----------|---------|
| ViTPose (RGB) | 48.5 | 62.3 | 31.4 |
| Hybrid ANN-SNN (Event) | 85.2 | 54.1 | 66.7 |
| LEIR (RGB+Event) | 112.4 | 135.8 | 78.2 |
| **ResPose (Ours)** | **7.2** | **4.8** | **6.5** |

高时间分辨率 HPE（1000Hz）：

| 方法 | MPJPE↓ | PCK0.3↑ | PCK0.5↑ |
|------|--------|---------|---------|
| ViTPose (linear interp.) | 10.06 | 0.96 | 0.98 |
| Hybrid ANN-SNN | 22.48 | 0.82 | 0.91 |
| EventPointPose | 51.61 | 0.48 | 0.74 |
| EvSharp2Blur | 8.78 | 0.95 | 0.96 |
| ResPose (ANN Variant) | 8.12 | 0.95 | 0.96 |
| **ResPose (Ours, SNN)** | **5.66** | **0.97** | **0.99** |

### 消融实验

标注流水线消融（精度 / 召回率）：

| 配置 | Kicking Precision | Kicking Recall | 说明 |
|------|-------------------|----------------|------|
| w/o $d_{ji}^t$ | 43.34% | 97.80% | 去掉开关时间距离→大量误匹配 |
| w/o $d_{ji}^p$ | 69.70% | 97.56% | 去掉周期距离→匹配质量下降 |
| w/o 异常值过滤 | 96.52% | 95.69% | 噪声干扰导致漏检 |
| w/o 跟踪 | 98.38% | 98.16% | 遮挡时无法恢复 |
| **完整流水线** | **99.99%** | **98.99%** | 几乎完美的精度 |

### 关键发现

- **ResPose 在 PMT 任务上实现了量级提升**：时间误差从纯 RGB 的 ~50ms、纯事件的 ~55-86ms 降至 ~5-7ms。这证明了结合 RGB 结构锚点和事件残差修正的有效性。
- **现有纯事件方法在 PMT 上反而失败**（LEIR 误差 78-136ms），说明高时间分辨率输入不等于高时间分辨率输出——需要配合 1000Hz 地面真值训练才行。
- **SNN 编码器优于 ANN 变体**：MPJPE 从 8.12 降至 5.66，证明脉冲神经网络处理异步事件数据的天然优势。
- 标注流水线精度达 99.99%、召回率 98.82%，与人工标注高度一致，验证了 LED 频率编码方案的鲁棒性。
- 即使 100Hz 高速相机的样条插值在快速动作中仍有显著误差（跳跃 28.5px），验证了 1000Hz 原生标注的必要性。

## 亮点与洞察

- **LED 频率编码 + 事件相机**的组合极其巧妙：用硬件设计绕过了软件算法的局限。不同于给每个 LED 贴不同颜色（RGB 相机方案），用不同闪烁频率编码身份天然适配事件相机的工作原理，成本极低。这种"hardware-in-the-loop"标注思路可以迁移到任意需要高频标注的场景。
- **残差分解（RGB 锚点 + 事件残差）**是处理跨时间分辨率融合的优雅框架：RGB 提供低频结构先验，事件提供高频运动增量。这个分解方式不仅适用于 HPE，也可推广到高速物体追踪、高频表面形变估计等任务。
- 系统设计的完整性令人印象深刻——从硬件（LED服装+多模态设备）到软件（标注流水线+基线方法）到数据集，形成了完整的闭环。

## 局限与展望

- LED 标记仍需穿戴特制服装，限制了自然场景下的使用。未来可探索无标记方案（结合事件相机的高动态范围直接估计高频姿态）。
- 17 个 LED 对应粗粒度关节，无法捕获手指等精细运动。增加 LED 数量可能导致频率冲突——闪烁模式的唯一性空间有限。
- 当前 3D 标注仅 60Hz（受 IMU+LiDAR 限制），1000Hz 标注仅限 2D。未来可结合多视角事件相机实现 1000Hz 3D 标注。
- FlashMotion 数据集规模和场景多样性仍有限（20 人、4 场景），扩展到更多受试者和运动类型（如体操、格斗）是必要的。
- SNN-CNN 混合编码器较简单，更复杂的事件表示学习方法（如精细时间分辨率的 Transformer）可能进一步提升性能。

## 相关工作与启发

- **vs BEAHM**: BEAHM 是此前最高帧率的事件 HPE 数据集（120Hz，基于 4 台标定 RGB 相机多视角重建）。FlashMotion 将标注帧率提升 8 倍至 1000Hz，且标注方式更原生（直接从事件流生成，非依赖 RGB 帧率瓶颈）。
- **vs DHP19**: DHP19 使用 100Hz Vicon 做地面真值，受限于 Vicon 的采样率。FlashCap 的 LED 方案不依赖外部 MoCap 系统，时间分辨率提升 10 倍。
- **vs EventCap**: EventCap 用事件相机做 HPE 但地面真值来自 100Hz 无标记 MoCap。FlashCap 的创新在于标注本身就是事件原生的，时间分辨率不受其他系统限制。
- 高速 RGB 相机（如 Basler，用于验证）成本高、带宽大，FlashCap 用约 1/9 的成本实现了接近甚至超越的时间精度。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ LED 频率编码 + 事件相机的创造性组合是极具开创性的高频运动捕捉范式
- 实验充分度: ⭐⭐⭐⭐⭐ 系统验证、数据集质量验证、两个新任务、完整消融，极其充分
- 写作质量: ⭐⭐⭐⭐ 从系统到数据集到方法层层推进，逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 开创了毫秒级运动捕捉的新方向，数据集和系统对整个 HPE 社区都有重大价值

<!-- RELATED:START -->

## 相关论文

- [SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](sharp_short-window_streaming_for_accurate_and_robust_prediction_in_motion_foreca.md)
- [MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](../../AAAI2026/autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)
- [LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment](../../ECCV2024/autonomous_driving/livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)
- [LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)

<!-- RELATED:END -->
