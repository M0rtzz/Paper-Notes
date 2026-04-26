---
title: >-
  [论文解读] Causal Discovery from Conditionally Stationary Time Series
description: >-
  [ICML 2025][时间序列][因果发现] 提出 SDCI（State-Dependent Causal Inference）——处理条件平稳时间序列的因果发现方法，通过离散潜状态变量建模非平稳行为，实现状态依赖的因果结构恢复，在粒子交互、基因调控网络和 NBA 球员运动预测中验证有效性。
tags:
  - ICML 2025
  - 时间序列
  - 因果发现
  - 非平稳时间序列
  - 条件平稳
  - 图神经网络
  - 隐状态
---

# Causal Discovery from Conditionally Stationary Time Series

**会议**: ICML 2025  
**arXiv**: [2110.06257](https://arxiv.org/abs/2110.06257)  
**代码**: 无  
**领域**: 时间序列  
**关键词**: 因果发现, 非平稳时间序列, 条件平稳, 图神经网络, 隐状态

## 一句话总结
提出 SDCI（State-Dependent Causal Inference）——处理条件平稳时间序列的因果发现方法，通过离散潜状态变量建模非平稳行为，实现状态依赖的因果结构恢复，在粒子交互、基因调控网络和 NBA 球员运动预测中验证有效性。

## 研究背景与动机

**领域现状**：传统时间序列因果发现方法（Granger 因果、TiMINo 等）假设平稳性——因果结构不随时间变化。但现实数据经常是非平稳的。

**现有痛点**：
   - 非平稳噪声建模方法（Huang et al.）假设因果结构固定，仅噪声变化
   - 时变效应方法假设因果图不变，仅效应强度变化
   - 离散隐变量方法（Saggioro et al.）仅处理"状态独立/可观测"的简单情况

**核心矛盾**：真实世界的因果关系通常是"状态依赖"的——例如篮球球员的行为取决于之前的比赛态势，碰撞前后粒子的交互规则完全不同。

**本文目标**：在因果结构随离散隐状态变化的非平稳时间序列中恢复因果图。

**切入角度**：将非平稳性建模为"条件平稳"——给定隐状态后系统是平稳的。三种场景：(a) 状态可观测; (b) 状态由观测决定; (c) 状态依赖历史（递归）。

**核心 idea**：条件汇总图（Conditional Summary Graph）作为条件平稳时间序列的紧凑因果表示 + 基于 GNN 和离散潜变量的实用算法。

## 方法详解

### 整体框架
SDCI 的核心组件：
1. **因果图表示**：条件汇总图 $\mathcal{G}^{CS}_{1:K} = \{\mathcal{G}^{CS}_k, k=1,...,K\}$——每个状态 $k$ 对应一个因果图
2. **隐状态推断**：基于观测推断当前状态 $u_t$
3. **因果结构学习**：用 GNN 为每个状态学习因果邻接矩阵
4. **预测**：基于推断的状态和对应因果图预测下一步

### 关键设计

1. **条件汇总图（Conditional Summary Graph）**:

    - 功能：紧凑表示状态依赖的因果结构
    - 核心思路：全时间图（Full Time Graph）在非平稳情况下是时变的，直接学习其复杂度指数级增长；条件汇总图将所有时间步按状态分组，每组一个静态因果图
    - 形式化：$i \to j \in \mathcal{E}^{CS}_k$ 当且仅当在状态 $k$ 下 $x_t^{(i)}$ 因果影响 $x_{t+1}^{(j)}$
    - 设计动机：将指数级复杂度降到 $O(K \cdot N^2)$，其中 $K$ 是状态数，$N$ 是变量数

2. **状态推断机制**:

    - 功能：从观测中推断离散隐状态
    - "状态决定"场景：状态直接由当前观测决定，$u_t = g(x_t)$
    - "状态递归"场景：状态依赖历史，用 RNN 编码器推断 $q(u_t | x_{1:t})$
    - 核心思路：变分推断——用 Gumbel-Softmax 使离散状态可微分
    - 设计动机：不同场景需要不同的推断机制，统一在变分框架下

3. **可识别性证明**:

    - 功能：严格证明在满足条件下条件汇总图可被唯一恢复
    - 核心思路：利用加性噪声模型（Additive Noise Model）的可识别性，扩展到状态依赖设定
    - 定理内容：在条件平稳性 + ANM + 充分激励条件下，条件汇总图可识别
    - 设计动机：为实用算法提供理论保障

4. **GNN 因果结构学习**:

    - 功能：参数化状态依赖的因果转移函数
    - 核心思路：每个状态 $k$ 有独立的邻接矩阵 $A_k$，GNN 层根据 $A_k$ 传播信息
    - 训练目标：重建损失 + 稀疏性正则化（$L_1$ on $A_k$）+ KL 散度（潜状态先验）
    - 设计动机：GNN 的消息传递自然对应因果影响的传播

### 损失函数 / 训练策略
- 总损失 = 重建损失（预测 $x_{t+1}$）+ 稀疏性惩罚（$\lambda \sum_k \|A_k\|_1$）+ KL 散度（$D_{KL}(q(u_t) || p(u_t))$）
- EM 算法交替更新：E 步推断隐状态，M 步优化因果图和转移函数
- Gumbel-Softmax 温度退火实现离散状态的可微训练

## 实验关键数据

### 主实验
粒子交互数据集（带碰撞的弹簧系统）：

| 方法 | AUROC (因果图) | 状态分类准确率 | 预测 MSE |
|------|-------------|-------------|---------|
| NRI (静态因果图) | 0.78 | - | 0.042 |
| dNRI (可变因果图) | 0.82 | 0.71 | 0.035 |
| **SDCI** | **0.93** | **0.89** | **0.021** |

### 基因调控网络

| 方法 | AUROC | AUPRC |
|------|-------|-------|
| Granger (平稳假设) | 0.62 | 0.48 |
| PCMCI | 0.68 | 0.55 |
| **SDCI** | **0.81** | **0.72** |

### NBA 球员运动预测

| 方法 | 预测误差 (m) ↓ |
|------|-------------|
| LSTM | 0.85 |
| NRI | 0.72 |
| **SDCI** | **0.58** |

### 消融实验

| 配置 | 因果图 AUROC | 说明 |
|------|------------|------|
| 固定因果图（无状态） | 0.78 | 退化为 NRI |
| 连续状态 | 0.85 | 不如离散状态清晰 |
| **离散状态（SDCI）** | **0.93** | 最优 |
| K=2 状态 | 0.90 | 足以捕捉碰撞/非碰撞 |
| K=5 状态 | 0.93 | 更细粒度 |
| 无稀疏性正则化 | 0.81 | 过度连接 |

### 关键发现
- 因果图恢复的 AUROC 从基线的 0.82 提升到 0.93——状态依赖建模的价值显著
- 离散状态比连续状态更适合因果结构变化（因果图的变化是离散的切换）
- NBA 数据上的改进说明因果建模对实际预测有帮助（不仅仅是因果图恢复）
- "状态递归"场景比"状态决定"更具挑战但结果仍然稳健
- 可识别性证明提供了理论保障——算法收敛到正确的因果图（在假设满足时）

## 亮点与洞察
- **条件平稳性假设**是对非平稳时间序列的优雅放松——比"完全非平稳"更可处理，比"平稳"更接近现实
- 条件汇总图作为因果表示兼顾紧凑性和表达力——每个状态一个图比每个时间步一个图高效得多
- 物理启发的实验设计（碰撞改变因果关系）非常直观——碰撞前粒子独立移动，碰撞后互相排斥
- 从因果发现到预测的提升说明学到的因果结构是真正有用的，不仅仅是图恢复指标上的改进
- "离散状态优于连续状态"的发现有独立价值——因为因果结构的变化本质上是离散的

## 局限与展望
- 状态数 $K$ 需要预先指定，自动确定更实用
- 仅考虑一阶马尔可夫假设，高阶依赖未处理
- 无隐混淆变量和无瞬时效应的假设较强
- GNN 的可扩展性在变量数 $N$ 很大时受限
- 可识别性证明基于高斯 ANM，非高斯场景的理论保障需要扩展
- 真实世界基因调控网络中因果关系的"真实标签"本身可能不完全可靠

## 相关工作与启发
- **vs NRI/dNRI**: NRI 假设静态因果图，dNRI 允许时变但无状态概念；SDCI 用离散状态解耦不同的因果模式
- **vs Granger Causality**: Granger 假设平稳+线性，SDCI 处理非平稳+非线性
- **vs PCMCI**: PCMCI 处理时滞因果但假设平稳，SDCI 放松平稳假设
- **vs Markov Switching Models**: 传统 MSM 不做因果发现（仅做状态切换建模），SDCI 将状态切换与因果结构学习联合
- **启发**：条件平稳性 + 离散隐变量的框架可推广到其他需要发现"模式依赖关系"的场景，如金融市场的牛/熊市因果结构差异

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 条件平稳因果发现是重要的理论+算法贡献
- 实验充分度: ⭐⭐⭐⭐ 合成+生物+体育数据，多场景验证
- 写作质量: ⭐⭐⭐⭐ 理论框架清晰，分类法有用
- 价值: ⭐⭐⭐⭐⭐ 推进了非平稳因果发现的前沿

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](../../NeurIPS2025/time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)

<!-- RELATED:END -->
