---
title: >-
  [论文解读] Competition-Aware CPC Forecasting with Near-Market Coverage
description: >-
  [CVPR 2026][时间序列][CPC 预测] 将付费搜索广告中的 CPC（每次点击成本）预测重新定义为**部分竞争可观测性**问题，通过语义邻域、DTW 行为邻域和地理意图三类竞争代理信号，结合时序基础模型（Chronos-2/TimeGPT/Moirai）和时空 GNN，在 1,811 条关键词序列上实现了中长期预测精度的显著提升。
tags:
  - CVPR 2026
  - 时间序列
  - CPC 预测
  - 竞争代理
  - 时序基础模型
  - 时空图神经网络
  - 部分可观测性
---

# Competition-Aware CPC Forecasting with Near-Market Coverage

**会议**: CVPR 2026  
**arXiv**: [2603.13059](https://arxiv.org/abs/2603.13059)  
**代码**: 无  
**领域**: 时间序列预测 / 广告竞价  
**关键词**: CPC 预测, 竞争代理, 时序基础模型, 时空图神经网络, 部分可观测性

## 一句话总结

将付费搜索广告中的 CPC（每次点击成本）预测重新定义为**部分竞争可观测性**问题，通过语义邻域、DTW 行为邻域和地理意图三类竞争代理信号，结合时序基础模型（Chronos-2/TimeGPT/Moirai）和时空 GNN，在 1,811 条关键词序列上实现了中长期预测精度的显著提升。

## 研究背景与动机

在付费搜索广告中，CPC 是通过实时拍卖生成的价格，受竞争对手出价、平台质量信号和查询特定条件共同决定。广告商面临的核心困境是**部分可观测性**：

- 广告商能观察到自身的 CPC、点击量、展示量和花费
- 但**无法**直接观察竞争对手的出价、预算约束和质量分数
- 这意味着 CPC 序列仅包含竞争环境的间接噪声痕迹

纯自回归预测方法在中长期（竞争格局变化、需求转移发生时）表现不佳。现有文献在拍卖机制理论上很丰富，但对如何在**间接竞争可观测性**下进行预测提供的指导有限。本文提出通过可观测信号构建"竞争代理"来近似潜在竞争状态。

## 方法详解

### 整体框架

本文构建了一个**竞争感知预测设计空间**，将潜在竞争通过三族代理信号进行操作化，并沿两条路径评估：（1）作为外生协变量直接输入预测模型；（2）作为关系先验通过语义邻接矩阵输入时空图神经网络。预测目标为 $h \in \{1, 6, 12\}$ 周的周度 CPC。

### 关键设计

1. **语义邻域与语义图**：使用预训练 Transformer（all-MiniLM-L6-v2）将每个关键词 $k_i$ 编码为 $e_i \in \mathbb{R}^{384}$ 的嵌入向量，通过余弦相似度识别语义相关关键词。语义邻域服务双重目的：提供邻域竞争协变量，并定义固定语义关键词图 $A^{sem} \in \mathbb{R}^{N \times N}$（$k=10$ 近邻，行归一化）。

2. **DTW 行为邻域**：文本相似性不能穷尽竞争关联性——词汇重叠很弱的关键词可能因共同需求冲击而经历类似 CPC 动态。使用**动态时间规整（DTW）** 度量 CPC 轨迹相似性（带 Sakoe-Chiba 带约束），构建行为邻域并转化为无泄漏的行为竞争特征。

3. **地理意图代理**：在租车市场中，拍卖压力高度地域化。从关键词文本中提取地理意图（大洲、国家、城市层级），每个关键词被分配结构化位置指标。保留多层级地理分辨率而非假设更细粒度总是更好。

### 预测架构

本文评估三大类模型：

- **经典/神经基线**：SARIMAX、XGBoost、Random Forest、LightGBM、MLP、LSTM、GRU、TabPFN
- **协变量增强的时序基础模型**：Chronos-2、TimeGPT、Moirai——竞争代理作为外生协变量输入
- **时空图神经网络（STGNN）**：DCRNN（扩散卷积 RNN）、GConvLSTM（图卷积 LSTM）、GraphWaveNet（自适应图结构）——消费语义图 $A^{sem}$

STGNN 的输入为时空张量 $X \in \mathbb{R}^{N \times T \times F}$，输出多步预测 $\hat{Y}_{t+h} \in \mathbb{R}^{N}$，使用 MAE 损失训练（对重尾分布更鲁棒）。

### 损失函数 / 训练策略

- 目标变量定义：$\text{cpc\_week}_{k,t} = \frac{\text{adcost\_sum}_{k,t}}{\text{adclicks\_sum}_{k,t}}$
- 严格时间序列划分：最后 20% 数据作为测试集，防止时间泄漏
- 所有邻域特征使用无泄漏（leakage-free）的历史数据构建
- 评估指标：sMAPE（主指标）和 RMSE（次指标）

## 实验关键数据

### 主实验（6 周预测期）

| 模型族 | 架构 | 竞争增强配置 | sMAPE (%) | RMSE |
|--------|------|-------------|----------|------|
| 统计/ML 基线 | SARIMAX | 单变量滞后 | 43.93±23.55 | 1.660±1.759 |
| 统计/ML 基线 | XGBoost | 核心运营特征 | 36.64±17.51 | 1.301±1.119 |
| 统计/ML 基线 | TabPFN | 核心运营特征 | 35.04±17.77 | 1.250±1.133 |
| 协变量 TSFM | Moirai | 无泄漏滞后+日历稳定化 | 30.14±18.24 | 1.000±0.970 |
| 协变量 TSFM | TimeGPT | 日历条件+增长钳制 | 29.29±17.07 | 1.002±1.008 |
| 协变量 TSFM | **Chronos-2** | **地理意图协变量** | **27.14±15.04** | **0.841±0.846** |
| 时空 GNN | GraphWaveNet | 语义图+搜索组合 | 30.57±20.57 | 1.005±0.941 |
| 时空 GNN | GConvLSTM | 语义图+大洲地理 | 30.69±20.42 | 1.001±0.955 |
| 时空 GNN | DCRNN | 语义图+地理+语义CPC | 30.42±20.42 | 1.000±0.926 |

### 跨期汇总

| 模型族 | 1 周 sMAPE | 6 周 sMAPE | 12 周 sMAPE |
|--------|----------|----------|-----------|
| 最佳基线 | 30.42 | 35.04 | 40.23 |
| 最佳协变量 TSFM | 27.94 | **27.14** | **29.14** |
| 最佳时空 GNN | **25.82** | 30.42 | 37.46 |

### 关键发现

- **最优方法随预测期变化**：1 周 → STGNN 最佳（25.82%），6/12 周 → 协变量增强 TSFM 最佳（27.14%/29.14%）
- **粗粒度地理是最鲁棒的竞争先验**：大洲级编码在所有骨干和预测期上均改善稳定性；更细的国家/城市级编码反而分散信号
- **特征堆叠有害**：在 6 周期，朴素堆叠所有代理产生最差表现（34.0% sMAPE），比最优选择性配置差 3.3 个百分点
- **竞争前沿分析**：高 CPC + 高波动性关键词（402 个）上改进最显著，Core + Geo + Sem CPC 在该区域降低误差 1.3 个百分点

## 亮点与洞察

- **问题重构**：将 CPC 预测从"时序外推"问题重构为"部分竞争可观测性下的预测"问题，这一视角转变是核心贡献
- **选择性胜于穷举**：竞争代理的价值在于有选择地组合，而非无脑堆叠——这与特征工程的一般规律一致
- **基础模型 + 领域代理**：Chronos-2 仅需粗粒度地理意图作为协变量即可在中长期击败所有方法，说明预训练大模型与领域先验的组合潜力巨大
- **竞争前沿**的概念很有启发性：关注预测误差的**分布**而非仅关注平均值

## 局限与展望

- 仅在单一垂直领域（欧洲租车市场）验证，竞争集中度高，泛化性需验证
- 语义图是静态的，无法捕获关键词关系的动态演变、市场进入/退出
- 数据来自单一广告商视角，"近市场覆盖"仍非完全市场可观测
- 未探索动态图构建和更丰富的竞争方信号
- 周度聚合可能平滑掉日内竞价动态

## 相关工作与启发

- **Chronos-2 [Amazon 2024]**：从单变量到通用时序预测的基础模型
- **TimeGPT [Garza et al. 2024]**：零样本时序基础模型
- **DCRNN [Li et al. 2018]**：扩散卷积 RNN，用于交通预测的经典 STGNN
- **GraphWaveNet [2019]**：自适应图 + 固定图结构的时空预测
- 启发：竞争代理构建思路可推广至任何多方博弈场景的预测（如出行平台定价、电商竞价）

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)
- [Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](../../ICLR2026/time_series/test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](../../AAAI2026/time_series/recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)

<!-- RELATED:END -->
