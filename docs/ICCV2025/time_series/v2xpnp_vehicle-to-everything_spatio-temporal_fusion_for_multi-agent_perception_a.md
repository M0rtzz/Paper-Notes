---
title: >-
  [论文解读] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction
description: >-
  [ICCV 2025][时间序列][车路协同] 提出 V2XPnP，一个基于统一 Transformer 架构的 V2X 时空融合框架，在单步通信策略下实现多智能体端到端感知与预测，同时构建了首个支持所有 V2X 协作模式的大规模真实世界时序数据集，在感知和预测任务上达到 SOTA。 领域现状 自动驾驶系统需要准确感知周围道…
tags:
  - "ICCV 2025"
  - "时间序列"
  - "车路协同"
  - "时空融合"
  - "端到端感知预测"
  - "V2X数据集"
  - "Transformer"
---

# V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction

**会议**: ICCV 2025  
**arXiv**: [2412.01812](https://arxiv.org/abs/2412.01812)  
**代码**: [mobility-lab.seas.ucla.edu/v2xpnp](https://mobility-lab.seas.ucla.edu/v2xpnp)  
**领域**: 时序预测 / 自动驾驶  
**关键词**: 车路协同, 时空融合, 端到端感知预测, V2X数据集, Transformer

## 一句话总结

提出 V2XPnP，一个基于统一 Transformer 架构的 V2X 时空融合框架，在单步通信策略下实现多智能体端到端感知与预测，同时构建了首个支持所有 V2X 协作模式的大规模真实世界时序数据集，在感知和预测任务上达到 SOTA。

## 研究背景与动机

### 领域现状

自动驾驶系统需要准确感知周围道路用户并预测其未来轨迹。单车系统受限于感知范围和遮挡问题，车路协同（V2X）技术通过多智能体信息共享弥补了这些不足。

### 现有痛点

**仅关注单帧协同感知**：现有 V2X 工作主要做逐帧协同检测，将不同空间位置的智能体信息融合，但忽略了跨帧的时序线索

**时序任务缺失**：短期时序线索（0.5s）仅用于缓解异步问题，长期时序任务（如运动预测）几乎未被探索

**真实世界时序数据集匮乏**：现有 V2X 数据集多为非时序的、仅支持单种协作模式，缺乏支持所有协作模式（V2V、V2I、I2I、VC、IC）的时序数据集

**端到端 PnP 框架缺失**：感知和预测的解耦流水线存在误差传播问题

### 核心问题

多智能体多帧协同中的三个关键问题：(1) 传输什么信息？(2) 何时传输？(3) 如何融合多智能体的时间和空间维度信息？

## 方法详解

### 整体框架

V2XPnP 包含六个模块：V2X 元数据共享、LiDAR 特征提取（PointPillar）、多帧时序融合、压缩与共享、多智能体空间融合、地图特征提取，最后接检测头和预测头。采用中间特征融合 + 单步通信策略，每个智能体先在本地融合历史 BEV 特征，再将压缩后的单帧特征传输给 ego 智能体。

### 关键设计

#### 1. **单步通信策略（One-step Communication）**

- **功能**：每个智能体在单次通信中共享所有历史数据的融合结果，而非多步逐帧传输
- **核心思路**：每个智能体先在本地将历史 BEV 特征序列 $\mathbf{F}_i^{seq} \in \mathbb{R}^{T \times H \times W \times C}$ 融合为单帧 $\mathbf{F}'_i \in \mathbb{R}^{H \times W \times C}$，传输量与单帧协同感知相当
- **设计动机**：多步通信存在延迟累积、数据丢失、以及相邻智能体可能在历史帧中不在通信范围内等问题。单步通信在保留完整时空信息的同时将传输量从 $5 \times 0.269$ Mb 降至 $0.269$ Mb，延迟约 10-20ms

#### 2. **时空融合 Transformer**

- **功能**：通过三种注意力模块实现统一的时空融合
- **核心思路**：

**时序注意力**：仅融合同一空间位置跨帧的特征，加入可学习的时间戳嵌入：
$$\mathbf{F}_i^{tem} = \text{MHSA}(Q: \text{MLP}(\mathbf{F}_i^{seq'}), K: \text{MLP}(\mathbf{F}_i^{seq'}), V: \text{MLP}(\mathbf{F}_i^{seq'}))$$

**自空间注意力**：使用多尺度窗口注意力（局部/中间/全局窗口）捕获单智能体内不同尺度的 BEV 空间交互

**多智能体空间注意力**：异构设计，为不同交互对（V-I, V-V, I-V, I-I）使用独立可学习权重：
$$\mathbf{F}_{i,m}^{sp} = \sum_j \text{Softmax}(\mathbf{Q}_i^m \cdot \mathbf{W}_{att}^{(e_{i,j})} \cdot \mathbf{K}_j^n) \cdot \mathbf{V}_j^n$$

- **设计动机**：时序和空间信息需要分别建模以保留各自的结构特性。异构注意力权重考虑了车辆和基础设施传感器的部署差异

#### 3. **地图特征注入**

- **功能**：将高精地图的矢量化多段线编码并注入 BEV 特征
- **核心思路**：对每个 BEV 网格的周围地图多段线进行 MLP 编码，再通过 BEV-地图自注意力融合：

$$\mathbf{F} = \text{MHSA}(Q: [\mathbf{F}_{bm}, \mathbf{P}_m], K: [\mathbf{F}_{bm}, \mathbf{P}_m], V: \mathbf{F}_{bm})$$

- **设计动机**：地图信息为轨迹预测提供道路结构约束，引导预测轨迹沿道路方向运动

### 损失函数 / 训练策略

- **感知损失**：Smooth L1 回归损失（位置、尺寸、朝向）+ Focal Loss 分类损失
- **预测损失**：L2 损失（预测轨迹点 vs 真值轨迹）
- **总损失**：三项加权求和
- 训练/验证/测试划分：76/6/14 个场景
- 通信范围：50m，评估范围 x∈[-70,70]m, y∈[-40,40]m
- 历史长度：2s (2Hz)，预测时域：3s (2Hz)

## 实验关键数据

### 主实验

| 协作模式 | 方法 | 端到端 | 地图 | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|---------|------|-------|------|---------|------|------|-----|------|
| VC | No Fusion | ✓ | | 43.9 | 1.87 | 3.24 | 33.8 | 24.3 |
| VC | Late Fusion | | ✓ | 58.1 | 1.59 | 2.82 | 32.4 | 33.0 |
| VC | V2X-ViT* | ✓ | ✓ | 69.6 | 1.39 | 2.56 | 35.2 | 44.7 |
| VC | **V2XPnP** | **✓** | **✓** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |
| V2V | No Fusion | ✓ | | 40.8 | 1.99 | 3.38 | 34.0 | 19.8 |
| V2V | V2X-ViT* | ✓ | ✓ | 64.6 | 1.68 | 3.13 | 39.8 | 36.7 |
| V2V | **V2XPnP** | **✓** | **✓** | **70.5** | 1.78 | 3.28 | 39.9 | **40.6** |
| IC | V2X-ViT* | ✓ | ✓ | 69.3 | 1.27 | 2.39 | 35.4 | 43.3 |
| IC | **V2XPnP** | **✓** | **✓** | **71.0** | **1.18** | **2.16** | **34.0** | **46.0** |

V2XPnP 在所有协作模式的 EPA 指标上都是最佳（VC +3.5, IC +2.7, V2V +3.9, I2I +1.2）。

### 消融实验

| 时序融合 | 空间融合 | 地图融合 | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|---------|---------|---------|---------|------|------|-----|------|
| | | | 43.9 | - | - | - | - |
| ✓ | | | 57.2 | 1.52 | 2.76 | 35.5 | 33.8 |
| ✓ | ✓ | | 71.3 | 1.48 | 2.70 | 36.2 | 44.4 |
| **✓** | **✓** | **✓** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |

通信策略对比：

| 策略 | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|------|---------|------|------|-----|------|
| Multi-step | 68.2 | 1.56 | 2.84 | 31.8 | 43.0 |
| **One-step** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |

### 关键发现

- **时序融合是关键基础**：加入时序融合后 AP 从 43.9 提升至 57.2（+13.3），甚至超过了解耦的 Late Fusion（55.3-61.3）
- **单步通信全面优于多步通信**：AP +3.4, EPA +5.2，且传输量相当于单帧协同感知
- **地图融合主要提升预测性能**：AP 几乎不变（71.3→71.6），但 ADE 从 1.48 降至 1.35，EPA 从 44.4 升至 48.2
- **端到端优于解耦**：FaF*（端到端无融合）在检测上优于解耦无融合模型，且性能可比拟 Late Fusion
- **异构注意力的必要性**：车辆和基础设施传感器的部署位置和能力不同，统一权重会降低性能
- **V2XPnP 在 128× 压缩率下仍保持优异性能**，鲁棒性优于 V2X-ViT*

## 亮点与洞察

1. **系统性的 V2X 时空融合分析**：首次全面探讨 V2X 场景下"传什么、何时传、怎么融"三个维度的设计空间
2. **单步通信的优越性**：反直觉地，一次性传输融合结果优于多步逐帧传输，因为避免了累积误差和通信不稳定
3. **EPA 指标的引入**：联合评估感知和预测性能，避免了检测模块差但偶然获得简单轨迹导致预测指标虚高的问题
4. **首个全协作模式真实世界时序数据集**：96 个场景、40K 帧、4 个智能体、支持 V2V/V2I/I2I/VC/IC

## 局限与展望

1. **仅使用 LiDAR 数据**：相机数据的融入可能进一步提升性能
2. **固定通信范围 50m**：更长距离的通信和更多智能体的场景未探索
3. **数据集规模有限**：96 个场景可能不足以覆盖所有复杂的交通场景
4. **预测时域仅 3s**：更长时域的预测对安全驾驶更有价值但更具挑战性
5. **未考虑通信失败**：现实中的丢包、延迟抖动等问题需要鲁棒性设计
6. **PointPillar backbone 较旧**：VoxelNet 或 CenterPoint 等更先进的 backbone 可能带来提升

## 相关工作与启发

- FaF 和 PnPNet 是经典的端到端感知预测框架，V2XPnP 将其扩展到多智能体场景
- V2X-ViT 是当前最强的 V2X 中间融合模型，V2XPnP 通过时空融合全面超越
- V2X-Seq 是唯一的现有 V2X 时序数据集，但仅限 V2I 且下载受限
- CoBEVFlow 和 FFNet 利用短期历史（0.5s）解决异步问题，V2XPnP 将时序扩展到 2s 并支持预测任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — V2X 场景下的端到端时空融合框架设计有系统性创新，但各组件技术较成熟
- **实验充分度**: ⭐⭐⭐⭐⭐ — 11 个基线模型、4 种协作模式、详尽消融和鲁棒性测试
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，"传什么、何时传、怎么融"的分析框架有条理
- **价值**: ⭐⭐⭐⭐⭐ — 数据集填补了真实世界 V2X 时序数据的关键空白，框架和 benchmark 对社区价值巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](../../AAAI2026/time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](../../ICML2026/time_series/nested_spatio-temporal_time_series_forecasting.md)
- [\[ACL 2026\] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](../../ACL2026/time_series/streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)

</div>

<!-- RELATED:END -->
