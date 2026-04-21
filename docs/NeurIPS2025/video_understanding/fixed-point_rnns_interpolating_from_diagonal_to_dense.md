---
title: >-
  [论文解读] Fixed-Point RNNs: Interpolating from Diagonal to Dense
description: >-
  [NeurIPS 2025][视频理解][Fixed-Point Iteration] 提出 Fixed-Point RNN 框架，将稠密线性 RNN 参数化为对角线性 RNN 的不动点，通过迭代次数在对角（高效）与稠密（表达力强）之间动态插值，首次在状态跟踪（$A_5$/$S_5$）和拷贝任务上同时取得最优结果。
tags:
  - NeurIPS 2025
  - 视频理解
  - Fixed-Point Iteration
  - Dense Linear RNN
  - State-Space Model
  - Mamba
  - State Tracking
---

# Fixed-Point RNNs: Interpolating from Diagonal to Dense

**会议**: NeurIPS 2025  
**arXiv**: [2503.10799](https://arxiv.org/abs/2503.10799)  
**代码**: 无（暂未公开）  
**领域**: 序列建模 / 状态空间模型  
**关键词**: Fixed-Point Iteration, Dense Linear RNN, State-Space Model, Mamba, State Tracking

## 一句话总结

提出 Fixed-Point RNN 框架，将稠密线性 RNN 参数化为对角线性 RNN 的不动点，通过迭代次数在对角（高效）与稠密（表达力强）之间动态插值，首次在状态跟踪（$A_5$/$S_5$）和拷贝任务上同时取得最优结果。

## 研究背景与动机

**领域现状**: 线性 RNN（如 Mamba）因对角状态转移矩阵可高效并行训练，已成为 Transformer 的主流替代。理论研究表明，对角结构严重限制了模型的状态跟踪能力，而稠密转移矩阵理论上具有完整的非线性 RNN 表达力。

**现有痛点**:
   - **对角 RNN 表达力不足**：Mamba 无法学习 $A_5$/$S_5$ 等状态跟踪任务
   - **稠密 RNN 效率问题**：参数化 $A_t \in \mathbb{R}^{d \times d}$ 需要 $O(d^3)$ 时间和 $O(d^2)$ 参数
   - **结构化方法的局限**：DeltaProduct 等用 Householder 结构增强表达力，但选择何种结构仍是 ad-hoc 的

**核心矛盾**: 并行性与表达力之间的根本性权衡——对角 RNN 可并行但不表达，稠密 RNN 表达但不可并行。如何系统性地在两者之间导航？

**本文目标**: 设计一种通用框架，允许模型**自适应地**在效率和表达力之间权衡，而非在设计时固定结构。

**切入角度**: 借鉴 Deep Equilibrium Models 思想，将稠密 RNN 隐式表示为对角 RNN 的不动点。

**核心 idea**: 用 $h = f_\theta(x, h)$ 的不动点迭代（交替进行通道混合与序列混合）来实现稠密 RNN，迭代次数可自适应于任务难度。

## 方法详解

### 整体框架

目标稠密 RNN：$h_t^* = Q_t^{-1} \Lambda_t h_{t-1}^* + B_t x_t$

将其改写为不动点方程：

$$h_t^* = \Lambda_t h_{t-1}^* + Q_t B_t x_t + (I - Q_t) h_t^*$$

对应的对角 RNN：

$$h_t^\ell = \Lambda_t h_{t-1}^\ell + Q_t B_t x_t + (I - Q_t) h_t^{\ell-1}$$

迭代 $\ell$ 次后收敛到稠密 RNN 的隐状态 $h^*$。

### 关键设计

#### 1. 稳定性保证（Banach 不动点定理）

需要满足两个条件：
- **时间维度**：$\|\Lambda\|_2 < 1$（对角转移的收缩性）+ 输入归一化 $(I - \Lambda_t)$
- **深度维度**：$\|I - Q_t\|_2 < 1$（通道混合的收缩性）

满足条件时，迭代从任意初始化 $h^0 = 0$ 出发均收敛到唯一不动点，所有中间状态均不爆炸。

#### 2. 通道混合器 $Q_t$ 的结构选项

| 结构 | 参数化 | 参数量 |
|------|--------|--------|
| DPLR | $Q_t = I - \sum_{i=1}^r \alpha_{it} \bar{u}_{it} \bar{u}_{it}^\top$ | $O(d^2 r)$ |
| Householder (H) | $Q_t = \prod_{i=1}^r (I - \alpha_{it} \bar{u}_{it} \bar{u}_{it}^\top)$ | $O(d^2 r)$ |
| Kronecker (K) | $Q_t = I - (\bar{K}_t^1 \otimes \bar{K}_t^2)$ | $O(d^2)$ |

实验表明 Kronecker 结构最适合状态跟踪，Householder 结构在记忆任务上也表现良好。

#### 3. 隐状态依赖的通道混合

$Q_t = \mathcal{M}(x_t + h_{t-1}^{\ell-1})$：通道混合器不仅依赖输入，还依赖前一次迭代的隐状态。这对序列长度泛化至关重要。

#### 4. FP-Mamba：矩阵值隐状态

将框架扩展到 Mamba 的矩阵值隐状态 $H_t \in \mathbb{R}^{d_{\text{state}} \times d_{\text{inner}}}$：

$$H_t^\ell = \lambda_t \odot H_{t-1}^\ell + \bar{b}_t^\ell (\Delta_t \tilde{x}_t^\ell)^\top$$

其中 $\tilde{x}_t^\ell = Q_t^\ell (x_t - y_t^{\ell-1}) + y_t^{\ell-1}$。

每次不动点迭代 = 一次通道混合（$Q_t$） + 一次 Mamba 序列混合。

#### 5. 自适应迭代次数

收敛条件：$\frac{\|h^\ell - h^{\ell-1}\|_\infty}{\|h^\ell\|_\infty} < 0.1$

模型自动适应任务难度：
- 状态跟踪 $A_5$（困难）：$\ell^* \approx$ 序列长度
- 语言建模（简单通道混合需求）：$\ell^* \ll T$
- 拷贝任务：$\ell^*$ 远低于最大序列长度

### 损失函数/训练策略

- **截断反向传播**：仅在不动点处计算梯度（$k=0$），无需存储迭代计算图
- **无额外内存开销**：相比单层对角 RNN，仅增加前向传播的顺序开销
- **随机化训练**：可将 $\ell_{\max} \sim \Gamma(4,1)$ 采样以加速训练，平均仅需 4 次迭代

## 实验关键数据

### 主实验：状态跟踪 $A_5$/$S_5$

- 1 层 FP-Mamba-K（Kronecker）在训练长度 16 后泛化到 **80+** 长度（$A_5$）
- 1 层 FP-Mamba-H 在 $S_5$ 上也实现了显著泛化
- 对比：2 层 Mamba/Mamba-2 在训练长度上就无法学会，2 层 Gated DeltaNet 可学会训练长度但泛化受限
- LSTM 作为上界可完美泛化

### 主实验：拷贝任务

| 模型 | $2\times$ 长度泛化准确率 |
|------|------------------------|
| Mamba (2层) | 低 |
| Mamba-2 (2层) | 中等 |
| Gated DeltaNet (2层) | **高** |
| **FP-Mamba-H (1层)** | **高（匹配 GDN）** |

FP-Mamba 是**唯一一个同时在状态跟踪和拷贝任务上取得最优结果的模型**。

### 消融实验

**隐状态依赖的贡献**（表 1，拷贝任务）：

| $\lambda_t$ 依赖 | $Q_t$ 依赖 | $b_t$ 依赖 | $c_t$ 依赖 | $2\times$ 泛化准确率 |
|:-:|:-:|:-:|:-:|---:|
| | | | | 0.11 |
| | ✓ | | | 0.53 |
| | | ✓ | ✓ | 0.81 |
| ✓ | ✓ | ✓ | ✓ | **0.94** |

$b_t$ 和 $c_t$ 的隐状态依赖是解锁拷贝能力的关键。

### 训练时间 vs 性能

训练时间（wall clock）与可泛化的最长序列长度的关系：
- **基线模型**（增加深度）：无论训练多久，泛化长度不超过训练长度 16
- **FP-Mamba**（增加迭代次数）：训练时间越长，泛化长度持续增长

### 关键发现

1. **不动点迭代次数自适应于任务难度**：简单任务（语言建模）$\ell^*$ 很小，困难任务（$A_5$/$S_5$）$\ell^*$ 接近序列长度
2. **深度 vs 宽度**：堆叠更多层无法解决对角 RNN 的根本表达力限制，而同参数量的不动点迭代可以
3. **语言建模中的收敛**：FP-Mamba 在语言预训练过程中，各层的有效迭代次数 $\ell^*$ 很快稳定在较小值
4. **反向传播简化**：仅在不动点处计算梯度（不需回传整个迭代过程）就足以稳定训练

## 亮点与洞察

1. **框架的通用性**：不限于特定的通道混合结构（DPLR/Householder/Kronecker），可以"即插即用"各种结构
2. **优雅的效率-表达力权衡**：迭代次数从 0（纯对角）到 $\infty$（稠密 RNN）提供了连续的旋钮
3. **与经典理论的深刻联系**：Banach 不动点定理保证收敛，Deep Equilibrium Model 的隐式微分消除内存开销
4. **同时解决两类任务**：打破了"状态跟踪需要稠密结构"和"拷贝需要线性注意力记忆"之间的二选一局面
5. **自适应计算**：模型自动鉴别任务难度并调整计算量，类似 Graves (2016) 的自适应计算时间思想

## 局限与展望

1. **最坏情况复杂度**：$O(\ell^* \cdot \log T)$，若 $\ell^* \sim T$ 则退化为 $O(T^2)$
2. **高效 GPU 实现缺失**：不动点迭代的重复计算有融合为单 kernel 的潜力，但尚未实现
3. **大规模语言建模验证不足**：语言预训练实验规模较小（上下文长度 2048），未与 DeltaProduct 等在 1.3B 规模对比
4. **非唯一不动点**：隐状态依赖版本的收敛保证较弱，不动点可能不唯一
5. **训练时间开销**：虽然反向传播无额外开销，前向传播的迭代仍然增加了训练时间

## 相关工作与启发

- **Deep Equilibrium Models (Bai et al.)**: 不动点迭代的灵感来源，但本文应用于线性 RNN 上下文
- **DeltaNet/DeltaProduct**: 通过 Householder 增加表达力，是一种"显式"的结构化方法；FP-RNN 通过"隐式"迭代达到稠密
- **Mamba/Mamba-2**: 对角 RNN 的代表，FP-Mamba 以其为基础层
- **LSTM**: 非线性 RNN 的上界，FP-RNN 试图在线性框架内逼近其表达力
- **Adaptive Computation Time (Graves)**: 自适应计算量的思想先驱

## 评分

⭐⭐⭐⭐⭐

理论框架优雅（不动点 + Banach 定理 + 隐式微分），实验全面（同时解决状态跟踪和拷贝），洞察深刻（自适应计算量揭示了任务难度与顺序计算需求的关系）。是线性 RNN 表达力问题的一个里程碑式工作。

<!-- RELATED:START -->

## 相关论文

- [Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)
- [AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [Dense SAE Latents Are Features, Not Bugs](dense_sae_latents_are_features_not_bugs.md)
- [TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video](tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)

<!-- RELATED:END -->
