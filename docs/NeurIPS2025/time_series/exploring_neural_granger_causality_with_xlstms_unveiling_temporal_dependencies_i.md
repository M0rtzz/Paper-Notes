---
title: >-
  [论文解读] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data
description: >-
  [NEURIPS2025][时间序列][Granger Causality] 提出 GC-xLSTM，利用 xLSTM 架构结合新颖的动态稀疏优化策略，在多变量时间序列中挖掘 Granger 因果关系，在多个数据集上取得 SOTA 性能。
tags:
  - "NEURIPS2025"
  - "时间序列"
  - "Granger Causality"
  - "xLSTM"
  - "Sparsity"
  - "Time Series"
  - "Causal Discovery"
---

# Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data

**会议**: NEURIPS2025  
**arXiv**: [2502.09981](https://arxiv.org/abs/2502.09981)  
**代码**: [github.com/harpoonix/GC-xLSTM](https://github.com/harpoonix/GC-xLSTM)  
**领域**: 时间序列  
**关键词**: Granger Causality, xLSTM, Sparsity, Time Series, Causal Discovery  

## 一句话总结

提出 GC-xLSTM，利用 xLSTM 架构结合新颖的动态稀疏优化策略，在多变量时间序列中挖掘 Granger 因果关系，在多个数据集上取得 SOTA 性能。

## 背景与动机

Granger 因果性（Granger Causality, GC）是判定一个时间序列的过去值能否帮助预测另一个的经典框架。传统方法基于向量自回归（VAR）模型进行统计假设检验，但存在以下局限：

1. **线性假设**：经典 GC 方法假设变量间关系为线性，无法捕捉非线性依赖
2. **短距离依赖**：现有基于 MLP 或标准 LSTM 的 Neural GC 方法在捕捉长程依赖时表现有限
3. **稀疏性提取不够严格**：传统 Group Lasso 正则化只能将参数缩小到接近零，后续层可以重新放大这些微弱信号，导致稀疏性不够严格，还需人为设定阈值 $\tau$

近期 xLSTM（Extended LSTM）通过引入指数门控和矩阵关联记忆，在序列建模中展现了强大能力，特别适合时间序列预测。这为利用更强的序列建模能力来发现 GC 关系提供了契机。

## 核心问题

如何利用具有更强建模能力的 xLSTM 架构，在非线性、存在长距离依赖和噪声的复杂时间序列数据中，鲁棒地发现 Granger 因果关系？

## 方法详解

### 整体架构

GC-xLSTM 采用 component-wise 架构，为每个变量 $v$ 分别建模：

1. **稀疏特征投影**：对每个变量 $v$ 学习一个投影矩阵 $\mathbf{W}_v \in \mathbb{R}^{D \times V}$，将 $V$ 个变量投影到 $D$ 维隐空间：$\mathbf{x}_v = \mathbf{W}_v \mathbf{S} + \mathbf{b}_v$
2. **xLSTM 预测**：使用单个 sLSTM block（含一个 sLSTM 层）进行自回归预测，隐藏维度为 32
3. **因果关系提取**：从投影矩阵 $\mathbf{W}_v$ 的稀疏结构直接读取 GC 关系

整个系统共需训练 $V$ 个独立模型，每个负责预测一个变量并从稀疏投影中提取该变量的入边。

### 联合优化策略（核心创新）

论文提出了一种交替优化方案，同时优化预测模型和严格稀疏性。每步包含两个阶段：

**阶段一：梯度下降更新**。联合优化投影权重 $\phi$、缩减系数 $\boldsymbol{\alpha}$ 和 xLSTM 参数 $\theta$，最小化：

$$\mathcal{L}_{\text{pred}}(\mathbf{S}; \phi_v, \theta_v) + \lambda \log\left(\sum_{w=1}^{V} \alpha_v^w \| \text{sg}(\mathbf{W}_v^w) \|_2 \right)$$

其中 $\text{sg}(\cdot)$ 表示 stop-gradient，意味着 $\mathbf{W}_v$ 在缩减损失中不被更新，仅用于引导缩减系数 $\boldsymbol{\alpha}$ 的学习。缩减系数通过 softmax 参数化 $\boldsymbol{\alpha}_v = \text{softmax}(\boldsymbol{\beta}_v)$ 以确保非负性和归一化。

**阶段二：近端梯度下降压缩**。对 $\mathbf{W}_v$ 执行基于 $\boldsymbol{\alpha}_v$ 加权的 Group Lasso 近端梯度步骤，并进行软阈值化（soft thresholding），将不重要的列严格压缩到零。

### 关键设计细节

- **对数缩减损失**：使用 $\log$ 使得对不同列范数的减小给予更均衡的权重，经验上显著提升了对噪声和超参 $\lambda$ 的鲁棒性
- **梯度动态的自增强特性**：当 $\|\mathbf{W}_v^w\|_2$ 较大时，对应的 $\alpha_v^w$ 会减小，从而在压缩步中保留重要特征；反之则加速消除，形成自增强循环
- **分阶段训练**：$\boldsymbol{\alpha}$ 在前 $K=1500$ 步不参与训练，先让预测损失引导模型获得合理初始化
- **无需阈值**：区别于传统方法需人为选择阈值 $\tau$，GC-xLSTM 通过近端梯度直接实现严格稀疏性
- **sLSTM 而非 mLSTM**：选择 sLSTM 是因为其 memory mixing 在时间序列预测中更有效

### 理论分析

论文论证了 sLSTM block 至少具有与 RNN 相同的表达能力（universal function approximator），因此 GC-xLSTM 架构足以以任意精度逼近底层生成过程 $g_v$，保证了模型类的充分性。

## 实验关键数据

在 6 个数据集上进行了广泛验证：

### Lorenz-96（混沌非线性系统）

| 模型 | F=10 Acc. | F=10 BA | F=40 Acc. | F=40 BA |
|------|-----------|---------|-----------|---------|
| cMLP | 97.2 | 95.6 | 68.3 | 80.5 |
| GVAR | 98.2 | 98.2 | 94.5 | 88.5 |
| **GC-xLSTM** | **99.1** | **98.5** | **96.3** | **96.6** |

在高混沌度 $F=40$ 设置下优势尤为明显，BA 超越次优 GVAR 达 8.1 个百分点。

### fMRI 脑连接

| 模型 | BA |
|------|-----|
| TCDF | 72.8±6.3 |
| cLSTM | 65.5±5.3 |
| **GC-xLSTM** | **73.3±3.0** |

### 消融实验（F=40 Lorenz / fMRI）

| 配置 | Lorenz BA | fMRI BA |
|------|-----------|---------|
| GC-xLSTM（完整） | 96.6 | 73.3 |
| 替换为标准 LSTM | 93.0 | 62.8 |
| 替换为标准 Group Lasso | 73.0 | 65.4 |

两个组件（xLSTM 架构 + 联合优化）均不可或缺，其中联合优化策略贡献更大。

### 实际数据定性分析

- **Moléne 天气**：从温度观测中学习到空间依赖，无需地理先验即可发现局部和远程气象模式
- **人体动作捕捉**：在 Salsa 舞蹈中发现脚→膝→手臂的驱动关系；在跑步中发现下肢作为主要运动驱动源
- **公司财务指标**：提取的因果边被金融专家验证为经济上合理

## 亮点

1. **方法设计精巧**：动态学习缩减系数实现严格稀疏而非依赖阈值，避免了传统 Lasso 方法的根本缺陷
2. **统一且鲁棒**：除 $\lambda$ 外所有超参在六个数据集上基本共享，体现了方法的鲁棒性
3. **理论与实践兼顾**：梯度动态分析清晰直观，消融实验充分验证了各组件贡献
4. **高效**：单 GPU 训练不超过 1.5 小时，时间和空间复杂度相对变量数近似线性增长
5. **真实数据上的可解释性**：在动作捕捉和天气数据上提取的因果关系具有直观物理意义

## 局限与展望

1. **缺乏收敛性理论保证**：虽有模型类充分性分析，但无法对算法收敛性给出严格数学证明
2. **变量规模有限**：实验最多数十个变量，未验证在高维度（数百/数千变量）场景的可扩展性
3. **超参 $\lambda$ 仍需调**：虽然只需调一个超参，但不同数据集需要不同的 $\lambda$，且 AUROC 是通过扫描 $\lambda \in \{5,...,15\}$ 得到的
4. **假设平稳性**：方法要求时间序列严格平稳，对非平稳场景的适用性待探索
5. **sLSTM 选择缺乏系统对比**：仅简单表述 sLSTM 优于 mLSTM，未提供详细对比

## 与相关工作的对比

| 方法 | 非线性 | 长程依赖 | 严格稀疏 | 无需阈值 |
|------|--------|---------|---------|---------|
| VAR (经典) | ✗ | ✗ | ✗ | ✗ |
| cMLP | ✓ | ✗ | ✗ | ✗ |
| cLSTM | ✓ | 有限 | ✗ | ✗ |
| GVAR | ✓ | ✓ | ✗ | ✗ |
| GC-KAN | ✓ | ✓ | ✗ | ✗ |
| **GC-xLSTM** | ✓ | ✓ | ✓ | ✓ |

GC-xLSTM 在所有维度上均具有优势，特别是「严格稀疏 + 无需阈值」的组合是其独特卖点。

## 启发与关联

1. **稀疏优化范式可迁移**：动态缩减系数 + 近端梯度的联合优化框架不限于 GC 发现，可推广到任何需要严格输入选择的场景（如特征选择、注意力稀疏化）
2. **xLSTM 在时间序列的潜力**：相比 Transformer，sLSTM 在 GC 发现中展现了更好的计算-效果权衡，暗示循环架构在因果任务中的独特优势
3. **可扩展到时间滞后特异性**：论文展示了按 lag 学习不同投影矩阵 $\mathbf{W}^{(\ell)}$ 的能力，这对理解不同时间尺度的因果关系很有价值
4. **与因果推断的桥接**：虽然 Granger 因果不等于 Pearl 因果，但论文引用了联系二者的理论工作，为后续研究留下了接口

## 评分
- 新颖性: 8/10 — xLSTM 用于 GC 发现本身新颖，动态稀疏优化策略设计精巧
- 实验充分度: 8/10 — 6 个数据集覆盖模拟和真实场景，消融完整；高维可扩展性不足
- 写作质量: 8/10 — 结构清晰，理论-方法-实验逻辑连贯，梯度动态分析直观
- 价值: 7/10 — 在时间序列因果发现领域有价值，但应用范围相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] How Patterns Dictate Learnability in Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)

</div>

<!-- RELATED:END -->
