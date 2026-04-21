---
title: >-
  [论文解读] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting
description: >-
  [ICLR 2026][图像生成][因果推断] 提出DoFlow，一种基于连续正则化流（CNF）的因果生成模型，在因果DAG上统一实现观测、干预和反事实时间序列预测，并可通过显式似然进行异常检测，在合成和真实医疗数据上验证了有效性。
tags:
  - ICLR 2026
  - 图像生成
  - 因果推断
  - 连续正则化流
  - 时间序列预测
  - 反事实推理
  - 异常检测
---

# DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting

**会议**: ICLR 2026  
**arXiv**: [2511.02137](https://arxiv.org/abs/2511.02137)  
**代码**: [有](https://github.com/StatFusion/DoFlow_Causal_Time_Series)  
**领域**: 医学图像  
**关键词**: 因果推断, 连续正则化流, 时间序列预测, 反事实推理, 异常检测

## 一句话总结

提出DoFlow，一种基于连续正则化流（CNF）的因果生成模型，在因果DAG上统一实现观测、干预和反事实时间序列预测，并可通过显式似然进行异常检测，在合成和真实医疗数据上验证了有效性。

## 研究背景与动机

时间序列预测是统计学和机器学习的核心问题。传统预测模型（ARIMA、LSTM、Transformer等）是**纯观测性的**——学习历史相关性并外推。但实际应用中常需回答因果性的"what if"问题：

**干预查询**："如果改变控制变量，系统如何演化？"例如水电站中改变涡轮控制信号，功率输出如何变化。观测预测器对固定历史只能给出固定预测，无法模拟不同控制方案。

**反事实查询**："如果当时采取不同干预，已观测到的轨迹会如何改变？"例如在医疗中，观察到患者的治疗和结局轨迹后，问在不同用药方案下这个特定患者的结局是否会更好。

**核心挑战**：现有因果生成模型主要面向静态数据，时间序列的因果反事实预测尚无通用框架。需要一个既具因果结构又具生成能力的模型。

## 方法详解

### 整体框架

DoFlow建模 $K$ 维多变量时间序列，节点在因果DAG上拓扑排序。每个节点 $X_{i,t}$ 依赖于自身历史 $X_{i,t-}$ 和父节点历史 $X_{\text{pa}(i),t-}$，由结构因果模型（SCM）定义：

$$X_{i,t} := f_i(X_{i,t-}, X_{\text{pa}(i),t-}, U_{i,t})$$

其中 $U_{i,t}$ 是独立的外生噪声。

序列分为上下文窗口 $\{1,...,\tau\}$（条件）和预测窗口 $\{\tau+1,...,T\}$（预测目标）。

### 关键设计

**1. 时间条件连续正则化流（Time-Conditioned CNF）**

为每个DAG节点 $i$ 学习一个CNF，共享跨时间步，通过Neural ODE定义数据分布与基底分布 $\mathcal{N}(0,1)$ 之间的连续变换：

$$\frac{dx_{i,t}(s)}{ds} = v_i(x_{i,t}(s), s; H_{i,t-1}), \quad s \in [0,1]$$

其中 $H_{i,t-1}$ 是由RNN聚合的历史隐状态：

$$H_{i,t-1} = \text{concat}(h_{i,t-1}, h_{\text{pa}(i),t-1})$$

**2. 正向过程（编码）**：将观测值 $x_{i,t}^F$ 映射到潜空间 $z_{i,t}^F$：

$$z_{i,t}^F = \Phi_\theta(x_{i,t}^F; H_{i,t-1}^F) = x_{i,t}^F + \int_0^1 v_i(x_{i,t}(s), s; H_{i,t-1}^F) ds$$

**3. 逆向过程（解码）**：从潜空间生成预测值：

$$\hat{x}_{i,t} = \Phi_\theta^{-1}(z_{i,t}; \hat{H}_{i,t-1}) = z_{i,t} - \int_0^1 v_i(x_{i,t}(s), s; \hat{H}_{i,t-1}) ds$$

**4. 三种预测模式**：

- **观测预测**：$z \sim \mathcal{N}(0,1)$，按拓扑序逆向解码
- **干预预测**：对 $(i,t) \in \mathcal{I}$，直接设 $\hat{x}_{i,t} \leftarrow \gamma_{i,t}$；非干预节点正常解码，但隐状态包含干预信息
- **反事实预测**（三步法）：
    - **溯因**：将事实观测编码为 $z_{i,t}^F$（用事实隐状态 $H^F$）
    - **行动**：应用干预
    - **预测**：用反事实隐状态 $\hat{H}^{CF}$ 解码 $z_{i,t}^F$ 得到反事实轨迹

**5. 似然异常检测**：CNF提供显式对数密度：

$$\log p_{\theta}(\hat{x}_{\tau+1:T} | \hat{H}_\tau) = \sum_{t=\tau+1}^T \left[\log q(z_t) + \int_0^1 \nabla \cdot v_\theta(x_t(s), s; \hat{H}_{t-1}) ds\right]$$

异常上下文导致低密度预测轨迹。

### 损失函数 / 训练策略

使用**条件流匹配（CFM）**损失训练，参考路径为直线插值：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}\left[\frac{1}{K(T-\tau)}\sum_{i,t} \|v_i(\phi(x_{i,t},z;s), s; H_{i,t-1}) - (z - x_{i,t})\|_2^2\right]$$

训练时使用观测值自回归更新隐状态（teacher forcing）。

**理论保证**（Corollary 4.5）：在单调SCM和精确训练假设下，DoFlow的反事实预测可精确恢复真实反事实轨迹。

## 实验关键数据

### 主实验

**合成数据观测/干预/反事实预测RMSE（多种DAG结构）**

| 方法 | Tree-Obs | Tree-Int | Tree-CF | Diamond-Obs | Diamond-Int | Diamond-CF |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DoFlow | **0.57** | **0.54** | **0.11** | **0.55** | **0.57** | **0.12** |
| GRU | 0.65 | 1.01 | NA | 0.58 | 0.94 | NA |
| TFT | 0.58 | 0.97 | NA | 0.63 | 1.18 | NA |
| TiDE | 0.60 | 1.15 | NA | 0.50 | 1.05 | NA |

**关键观察**：
- DoFlow在干预预测上大幅领先（RMSE差距~0.5），因为基线不具备因果结构
- 反事实预测是DoFlow独有能力，基线无法实现
- 非线性非加性（NLNA）场景也表现稳健

**真实数据：水电站干预预测**

DoFlow在水电站数据上成功预测不同涡轮控制方案下游信号的变化，因果结构与物理一致。

**真实数据：癌症治疗效果估计**

| 方法 | 均方根政策误差（RMSE of PEHE）↓ |
|------|:---:|
| DoFlow | **最优** |
| CRN | 次优 |

### 消融实验

- 加性 vs 非线性非加性噪声模型：DoFlow在两种设置下均表现良好
- 不同DAG结构（Chain/Tree/Diamond/FC-Layer）：一致性优异
- 异常检测AUROC：DoFlow在合成和真实水电站数据上有效检测异常

### 关键发现

1. DoFlow统一了观测、干预、反事实三种查询，是首个通用时间序列反事实框架
2. CNF的可逆性是核心：编码→修改条件→解码，天然支持反事实
3. 隐状态传播干预效应，下游节点自然感知上游干预
4. 显式似然密度提供异常检测的额外能力

## 亮点与洞察

- **统一框架**：用一个模型同时支持三种因果查询，架构设计自然优雅
- **CNF的因果对齐**：流的可逆性与因果推断的溯因-行动-预测三步法完美契合
- **理论支持**：证明了在单调SCM下的反事实精确恢复性质
- **显式似然**：除预测外免费获得异常检测能力，增加实用价值
- **RNN+CNF的组合**：RNN编码时序上下文，CNF处理不确定性和可逆映射

## 局限与展望

- 假设因果DAG已知，实际中可能需要因果发现
- 假设无同时步内因果效应（所有因果影响至少有一步时滞）
- 反事实恢复理论需要SCM单调性和精确训练假设
- 每个节点一个独立CNF，节点数多时扩展性待验证
- 真实场景中反事实真值不可观测，仅能在合成数据上定量评估
- 未与更复杂的时间序列因果效应估计方法深入对比

## 相关工作与启发

- **vs 传统因果效应方法**：后者关注离散动作的短期期望差异，DoFlow支持连续变量在任意时间的干预
- **vs 静态因果生成模型**（Javaloy等）：DoFlow扩展到时间序列，捕获跨时间的因果依赖
- **vs 现代预测器**（TFT/TiDE/TSMixer）：它们是观测性的，无法回答因果问题
- **医疗应用潜力**：个体化治疗方案比较、药物剂量优化、临床决策支持

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)
- [SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)

<!-- RELATED:END -->
