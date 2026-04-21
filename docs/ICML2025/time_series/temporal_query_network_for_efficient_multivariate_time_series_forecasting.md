---
title: >-
  [论文解读] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting
description: >-
  [ICML 2025][时间序列][多变量时序预测] 提出Temporal Query（TQ）技术——使用周期性移位的可学习向量作为注意力机制的query来捕获全局变量间相关模式，同时keys/values来自原始数据以保留样本级局部信息，在此基础上构建的TQNet仅使用单层多头注意力和浅层MLP，即在12个真实数据集上达到整体SOTA，且效率接近线性方法DLinear。
tags:
  - ICML 2025
  - 时间序列
  - 多变量时序预测
  - Temporal Query
  - 变量相关性
  - 注意力机制
  - 周期性参数
  - 轻量模型
---

# TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting

**会议**: ICML 2025  
**arXiv**: [2505.12917](https://arxiv.org/abs/2505.12917)  
**代码**: [GitHub - TQNet](https://github.com/ACAT-SCUT/TQNet)  
**领域**: 时间序列预测  
**关键词**: 多变量时序预测, Temporal Query, 变量相关性, 注意力机制, 周期性参数, 轻量模型

## 一句话总结

提出Temporal Query（TQ）技术——使用周期性移位的可学习向量作为注意力机制的query来捕获全局变量间相关模式，同时keys/values来自原始数据以保留样本级局部信息，在此基础上构建的TQNet仅使用单层多头注意力和浅层MLP，即在12个真实数据集上达到整体SOTA，且效率接近线性方法DLinear。

## 研究背景与动机

**领域现状**: 多变量时序预测（MTSF）的核心挑战在于准确建模变量间相关性。方法发展经历了三个阶段：Channel Mixing（CM）——混合变量但关系不可区分（如Informer、Autoformer）；Channel Independence（CI）——独立建模各变量忽略相关性但更鲁棒（如PatchTST、DLinear）；Channel Dependence（CD）——显式建模变量依赖但需要有效机制（如iTransformer、Crossformer）。

**现有痛点**: 非平稳干扰（极端值、缺失数据、噪声）导致单个样本中观察到的变量间相关性与全局训练集上的真实相关性存在显著差距。标准自注意力从样本数据中同时生成Q、K、V，容易受噪声扰动，学到的相关模式不稳定。

**核心矛盾**: 需要一种机制既能学习全局稳定的变量间相关模式（跨样本一致），又能保留每个样本的局部特异性信息。纯全局（可学习参数）会丢失样本特征，纯局部（原始数据）会被噪声干扰。

**本文目标**: 设计一种能在注意力机制中自然融合全局和局部相关性的技术，并基于此构建极简高效的预测模型。

**切入角度**: 受CycleNet中可学习周期参数的启发，将可学习向量作为注意力的query（编码全局模式），而keys/values保留原始数据（编码局部特征），通过注意力的Q-K交互实现全局-局部融合。

**核心 idea**: 用周期性移位的可学习向量替代标准自注意力中从数据生成的query，使attention score的计算$\frac{QK^\top}{\sqrt{L}}$天然融合了全局先验（Q来自可学习参数）和样本特征（K来自原始数据），单层即可捕获鲁棒的变量间依赖。

## 方法详解

### 整体框架

TQNet由三个组件组成：(1) TQ增强的单层多头注意力（TQ-MHA）捕获变量间相关性；(2) 浅层MLP建模时间依赖；(3) 线性层+Dropout投射到预测目标。输入$X_t \in \mathbb{R}^{C \times L}$经TQ-MHA处理后（C为变量数，L为回看窗口长度），残差连接送入MLP，最后投射输出$\bar{Y}_t \in \mathbb{R}^{C \times H}$（H为预测步长）。可选的Instance Normalization处理分布漂移。

### 关键设计

1. **Temporal Query（TQ）技术**
    - 功能：用可学习参数编码全局变量间相关模式，替代从噪声数据中生成的query
    - 核心思路：初始化可学习参数$\theta_{TQ} \in \mathbb{R}^{C \times W}$（$W$为数据的周期长度），对时间步$t$，按$t \mod W$周期性地从$\theta_{TQ}$中取出长度$L$的片段$\theta_{TQ}^{t,L}$作为query。周期性保证$\theta_{TQ}^{t,L} = \theta_{TQ}^{(t+i \cdot W),L}, i \in \mathbb{N}$，即间隔$W$步的样本共享相同TQ向量
    - 设计动机：可学习的query向量在训练过程中自动捕获最优的变量关系表示，周期性移位实现参数复用，通过对多个样本的梯度平均化消除局部噪声干扰

2. **TQ增强的多头注意力（TQ-MHA）**
    - 功能：融合全局和局部相关性的注意力机制
    - 核心思路：$Q_h = \theta_{TQ}^{t,L} W_h^Q$（来自TQ向量），$K_h = X_t W_h^K$, $V_h = X_t W_h^V$（来自原始数据）。注意力计算 $\text{Head}_h = \text{Softmax}(\frac{Q_h K_h^\top}{\sqrt{L}}) V_h$。Q编码全局模式、K/V编码局部信息，Q-K的点积自然融合了两个层次
    - 设计动机：相比标准自注意力（Q/K/V都来自数据，易受噪声干扰）和纯全局（Q/K都来自可学习参数，丢失样本特异性），TQ-MHA在两者间取得最佳平衡。消融实验证实(Q=TQ,K=Raw)配置显著优于其余两种

3. **周期长度超参数$W$的对齐**
    - 功能：控制TQ向量的周期性移位间隔，需与数据固有周期对齐
    - 核心思路：$W$应设为数据的最大稳定周期长度（如小时数据设为168对应一周，15分钟数据设为96对应一天）。可通过领域知识或自相关函数（ACF）确定。$W$为真实周期整数倍时仍有效（仅减少每个参数的训练样本数），但不对齐时会引入语义不一致导致性能下降
    - 设计动机：时序数据的周期性是先验知识，TQ通过与之对齐使得可学习参数能代表同一周期位相上的全局模式

### 损失函数 / 训练策略

使用L2 loss（MSE）训练，评估指标包括MSE和MAE。优化器未特别说明。Instance Normalization采用iTransformer中的简单方案：输入减均值除标准差，输出逆变换。Dropout可选添加在输出投射层前。

## 实验关键数据

### 12个数据集整体对比（平均4个预测步长）

| 方法 | ETTh1 MSE | Electricity MSE | Traffic MSE | PEMS03 MSE | Top2次数/24 |
|------|----------|----------------|------------|-----------|-----------|
| **TQNet** | **0.441** | **0.164** | 0.445 | **0.097** | **22** |
| TimeXer | 0.437 | 0.171 | 0.466 | 0.112 | 11 |
| CycleNet | 0.457 | 0.168 | 0.472 | 0.118 | 9 |
| iTransformer | 0.454 | 0.178 | 0.428 | 0.113 | 4 |
| PatchTST | 0.469 | 0.205 | 0.481 | - | 0 |
| DLinear | 0.456 | 0.212 | 0.625 | - | 0 |

### 消融实验：Query-Key配置影响

| 配置 | Electricity MSE | PEMS03 MSE | PEMS04 MSE | PEMS07 MSE |
|------|----------------|-----------|-----------|-----------|
| Q=Raw, K=Raw（标准自注意力） | 0.175 | 0.114 | 0.112 | 0.094 |
| **Q=TQ, K=Raw（TQNet默认）** | **0.164** | **0.097** | **0.091** | **0.075** |
| Q=TQ, K=TQ（纯全局） | 0.179 | 0.111 | 0.113 | 0.092 |

### 组件消融（Electricity数据集平均）

| 配置 | MSE | MAE | 说明 |
|------|-----|-----|------|
| TQNet完整 | **0.164** | **0.259** | MLP + TQ & MHA |
| 去除MHA | 0.169 | 0.262 | 仅MLP+通道标识符 |
| 去除TQ | 0.175 | 0.267 | 标准自注意力+MLP |
| 纯MLP | 0.190 | 0.276 | 无通道相关性建模 |

### TQ技术的跨架构可迁移性（Electricity数据集）

| 基础模型 | 原始MSE | +TQ后MSE | 变化 |
|---------|--------|---------|------|
| iTransformer | 0.175 | **0.163** | -6.9% |
| PatchTST | 0.191 | **0.171** | -10.5% |
| DLinear | 0.210 | **0.182** | -13.3% |

### 关键发现

- TQNet在24个指标中22个进入Top2，以压倒性优势取得整体SOTA
- Q=TQ, K=Raw配置在高维数据集上显著优于标准自注意力和纯全局注意力
- TQ是性能提升的最关键组件——去除TQ导致的性能下降（MSE +0.011）大于去除MHA的下降（+0.005）
- TQ技术可零修改地迁移到其他模型（iTransformer、PatchTST、DLinear），均带来显著提升
- t-SNE可视化显示TQ学到的表示与真实通道模式高度一致：相似通道在表示空间中聚集
- 超参数$W$与数据周期对齐时效果最佳，$W$为周期整数倍时仍保持竞争力
- 效率接近DLinear：在862通道Traffic数据集上训练时间与DLinear可比，远快于iTransformer

## 亮点与洞察

- **"用可学习query编码全局模式"的设计极其简洁优雅**：将复杂的全局-局部融合问题简化为Q的来源选择问题，单层注意力即可实现
- **周期性移位的设计直觉清晰**：时序数据本身具有周期性，TQ向量的周期性移位与数据的周期模式自然对齐，兼顾了参数复用和语义一致性
- **极简架构VS复杂模型的胜利**：单层注意力+浅层MLP击败了多层Transformer等复杂架构，再次说明在时序预测中设计比规模更重要
- **TQ技术的跨架构可迁移性**：作为一个独立技术组件可以提升多种现有方法，具有很强的实用价值

## 局限与展望

- 强依赖数据周期性来设定超参数$W$，对无明确周期或多周期重叠的数据可能不适用
- 当变量间相关性很弱时，强制建模多变量依赖可能引入不必要的复杂度
- 随着回看窗口$L$增大，多变量建模的收益下降（更长的时间信息可部分替代变量间线索）
- 注意力机制的二次复杂度$O(C^2 L)$在变量数$C > 1000$时可能成为瓶颈
- 仅使用L2 loss训练，未探索概率预测或分位数损失等更丰富的预测范式

## 相关工作与启发

- **iTransformer (Liu et al., 2024)**: 翻转注意力维度，将变量作为token，是TQNet最直接的对比——TQNet用TQ替换其query, 效果更好
- **CycleNet (Lin et al., 2024)**: 用可学习参数捕捉周期模式的灵感来源，TQ的周期性设计直接继承了这一思想
- **TimeXer (Wang et al., 2024)**: 引入外部变量的跨注意力方法，性能次于TQNet但同属CD方法阵营
- **DLinear (Zeng et al., 2023)**: CI方法的效率标杆，TQNet以同等效率超越其精度
- 启发：注意力机制中Q/K/V的来源选择是一个被低估的设计空间，TQ表明仅改变Q的来源即可带来显著改善

## 评分

- **新颖性**: ⭐⭐⭐⭐（TQ技术的全局-局部融合思路新颖简洁，周期性移位设计巧妙）
- **实验充分度**: ⭐⭐⭐⭐⭐（12个数据集、完整消融、跨架构迁移、表示可视化、效率分析）
- **写作质量**: ⭐⭐⭐⭐（结构清晰，图表丰富，消融逻辑完整）
- **价值**: ⭐⭐⭐⭐⭐（SOTA性能+极简架构+DLinear级效率+可迁移性，综合价值极高）

<!-- RELATED:START -->

## 相关论文

- [HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [Winner-takes-all for Multivariate Probabilistic Time Series Forecasting](winner-takes-all_for_multivariate_probabilistic_time_series_forecasting.md)
- [Learning Soft Sparse Shapes for Efficient Time-Series Classification](learning_soft_sparse_shapes_for_efficient_time-series_classification.md)

<!-- RELATED:END -->
