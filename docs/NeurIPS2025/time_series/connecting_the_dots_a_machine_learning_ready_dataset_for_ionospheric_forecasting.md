---
title: >-
  [论文解读] Connecting the Dots: 面向电离层预测的机器学习数据集
description: >-
  [NeurIPS 2025][时间序列][电离层预测] 本文构建了一个开放的、机器学习就绪的电离层预测数据集，融合了8种异构数据源（太阳观测、地磁指数、TEC地图等），覆盖2010-2024年约14年时间，并基于此训练了LSTM、SFNO、GraphCast三种时空模型作为基准，实现了最长12小时的TEC预测…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "电离层预测"
  - "数据集"
  - "太阳活动"
  - "时间序列预测"
  - "空间天气"
---

# Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.15743](https://arxiv.org/abs/2511.15743)  
**代码**: 无  
**领域**: 时间序列/空间天气  
**关键词**: 电离层预测, 数据集, 太阳活动, 时间序列预测, 空间天气

## 一句话总结

本文构建了一个开放的、机器学习就绪的电离层预测数据集，融合了8种异构数据源（太阳观测、地磁指数、TEC地图等），覆盖2010-2024年约14年时间，并基于此训练了LSTM、SFNO、GraphCast三种时空模型作为基准，实现了最长12小时的TEC预测。

## 研究背景与动机

- **电离层预测的重要性**：现代社会高度依赖GNSS导航、LEO卫星通信、航空网络和电力系统，而太阳耀斑、日冕物质抛射等太阳活动会直接扰动电离层，导致GNSS精度下降、无线电通信中断甚至电网瘫痪
- **数据碎片化的痛点**：
    - 电离层观测数据来自不同平台（卫星、地面站、智能手机众包），格式、时间分辨率和空间覆盖差异巨大
    - 现有数据产品并非为机器学习设计，缺失值表示不统一（OMNI数据集不同通道使用不同的哨兵值），需要大量预处理
    - 缺乏标准化的ML-ready数据集，导致模型间无法系统比较
- **本文定位**：作为2025 NASA Heliolab合作项目，构建首个将稀疏/密集TEC地图与太阳和地磁驱动数据对齐的统一数据集，填补该领域空白

## 方法详解

### 整体框架

数据集的核心设计思路是"异构对齐"：将时间分辨率从15秒到每日不等的8种数据源，统一对齐到公共时间轴（2010-05-13至2024-08-01），并以模块化结构存储。最终产品支持多种时间步长查询，内置PyTorch Dataset和归一化方案，可直接用于模型训练。

### 关键设计

1. **数据采集与融合**：集成8大异构数据源，覆盖太阳-地球耦合系统的完整链路

    | 数据源 | 特征 | 原始频率 | 时间范围 |
    |--------|------|----------|----------|
    | OMNI2 (NASA) | AU/AL/AE, SYM-H, IMF Bx/By/Bz, 太阳风速度 | 1 min | 2010.05-2024.08 |
    | NOAA/GFZ Kp | Ap, Kp 地磁指数 | 3 h | 1997-2025 |
    | JPL Dense TEC | 1°×1° 全球TEC网格 | 15 min | 2010.05-2024.07 |
    | Madrigal Sparse TEC | 1°×1° GNSS稀疏TEC | 5 min | 2010-2024 |
    | SDO-FM | EUV辐照度嵌入(NVAE) | 15 s | 2010.05-2024.08 |
    | Space Env. Tech. | F10.7/S10.7/M10.7/Y10.7太阳通量 | 每日 | 1997-2025 |
    | 轨道力学 | 太阳/月球天顶角、日地距离 | 可变 | 可变 |
    | 准偶极坐标 | 磁场投影的地理参考坐标 | 每年 | 2010-2024 |

2. **缺失值处理与时间对齐**：
    - 所有缺失值统一为NaN，解决OMNI等数据源使用不同哨兵值的问题
    - 采用前向填充（forward-filling）策略处理短间断，每个数据流定义最大回溯时间（max rewind time）
    - 大多数数据流的回溯时间等于原生采样频率，OMNI例外设为50分钟
    - 超出回溯时间的间断直接跳过，避免陈旧数据传播
    - 同一前向填充逻辑也作为插值策略，将所有特征重采样到统一频率

3. **地磁风暴事件目录**：
    - 基于Kp时间序列阈值划分地磁活动等级，采用NOAA G-level标准

    | 事件ID格式 | NOAA G-level (Kp范围) | 含义 |
    |-----------|----------------------|------|
    | G0Hℓ | Kp < 5 | 平静期 |
    | G1Hℓ | 5 ≤ Kp < 6 | 轻微风暴 |
    | G2Hℓ | 6 ≤ Kp < 7 | 中等风暴 |
    | G3Hℓ | 7 ≤ Kp < 8 | 强风暴 |
    | G4Hℓ | 8 ≤ Kp < 9 | 严重风暴 |
    | G5Hℓ | Kp ≥ 9 | 极端风暴 |

    - 事件ID编码方式：G级别 + "H" + 持续时长ℓ（小时），如G2H6表示达到G2级持续至少6小时的事件
    - 该目录用于确保训练/验证集不会将同一风暴事件拆分到不同集合，防止数据泄漏

### 损失函数 / 训练策略

- 训练数据使用15分钟频率对齐的数据产品，JPL Dense TEC作为预测目标
- 代码库内置PyTorch Dataset类，支持用户指定起止时间范围和数据集特定的归一化方案
- 模型以自回归方式进行预测，支持最长12小时的lead time
- 基于事件目录进行数据划分，保证地磁平静期和活跃期均被覆盖

## 实验关键数据

### 主实验

训练了三个IonCast基准模型进行全球TEC预测：

| 模型 | 架构基础 | 特点 |
|------|----------|------|
| IonCast-LSTM | LSTM | 经典序列建模基线 |
| IonCast-SFNO | Spherical Fourier Neural Operator | 球面频域建模，适配地球曲面 |
| IonCast-GraphCast | GraphCast | 借鉴天气预测最新进展的图网络架构 |

- 所有模型均超越持续性预测（persistence forecast）基线
- 在最长12小时的预测时间窗口内表现出色
- 模型在地磁平静和活跃条件下均进行了评估

### 消融实验

论文作为数据集论文，未设计传统消融实验，但提供了以下设计选择的灵活性供后续研究：
- 前向填充的最大回溯时间可由用户自行调整
- 时间频率可选（15秒到每日不等）
- 数据源可模块化增减

### 关键发现

- 将异构数据对齐到统一时间轴后，三种不同架构的模型均能成功训练并优于基线，验证了数据集的实用性
- 基于物理的事件目录划分有效防止了数据泄漏，确保模型在罕见地磁风暴事件上的评估可靠性
- 数据集覆盖约14年（2010-2024），横跨一个完整的太阳活动周期（第24-25周期），为长期趋势研究提供了充足数据

## 亮点与洞察

- **首个完整的太阳-电离层ML数据集**：填补了从太阳表面观测到电离层响应的完整数据链路空白，将SDO EUV嵌入、太阳风参数、地磁指数和TEC地图统一成单一数据产品
- **工程价值突出**：提供完整的开源pipeline（GitHub + Google Cloud公开数据桶），包含数据对齐、预处理、PyTorch数据加载和模型训练示例代码，极大降低该领域的入门门槛
- **事件目录设计精巧**：MESTICI分类方案同时编码风暴强度（G-level）和持续时间，比单纯的Kp阈值更能刻画地磁事件特征，为模型评估提供了物理意义明确的数据划分
- **众包数据整合**：将Android智能手机测量的TEC数据纳入稀疏TEC源，展示了低成本传感扩展的可能性

## 局限性 / 可改进方向

- **论文不含定量指标**：作为NeurIPS ML4PS Workshop论文，仅定性描述模型"优于持续性基线"，缺乏具体的RMSE/MAE数值和各模型间的完整对比
- **缺失值处理偏简单**：前向填充策略对具有快速变化特征（如太阳风突变）的数据流可能不够准确，可考虑基于物理约束的插值或学习型插补
- **空间分辨率受限**：TEC地图为1°×1°网格，对区域性电离层扰动（如行进式电离层扰动TID）的捕捉能力有限
- **SDO嵌入的黑盒性**：使用NVAE压缩的EUV辐照度嵌入丢失了原始光谱细节，且嵌入质量依赖于预训练模型的性能
- **时间覆盖局限**：虽覆盖约14年，但极端地磁事件（G4/G5级）样本极少，模型在极端事件上的泛化性无法充分验证

## 相关工作与启发

- **JPL GIM-TEC**：提供密集TEC全球地图，但仅是单一数据产品，未与驱动数据对齐
- **Madrigal/MIT Haystack**：提供稀疏TEC和等离子体参数，需额外处理才能用于ML
- **SDO Foundation Model**（Walsh et al., 2024）：将SDO全盘太阳观测压缩为低维嵌入，本文直接使用其输出作为特征
- **GraphCast**（Lam et al., 2023）：原用于中期天气预报，本文将其图网络架构适配到球面电离层预测
- **FourCastNet/SFNO**（Bonev et al., 2025）：球面傅里叶神经算子，天然适合处理球面上的全球场预测
- **启发点**：该数据集的模块化设计理念适用于其他多源时间序列融合场景（如海洋监测、气候建模），事件目录防数据泄漏的方法值得在所有事件驱动预测任务中推广

## 评分

- **新颖性**: ⭐⭐⭐ — 数据集构建而非方法创新，但"首个ML-ready电离层数据集"的定位具有独特价值
- **技术深度**: ⭐⭐⭐ — 数据工程细节充分（缺失值处理、时间对齐策略），但模型部分较为薄弱
- **实验充分度**: ⭐⭐ — 缺乏定量实验结果，仅提供定性描述
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据源表格和事件分类方案描述详尽
- **实用价值**: ⭐⭐⭐⭐ — 开源数据+代码+PyTorch接口的完整pipeline对空间天气ML社区有很强的推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/time_series/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
