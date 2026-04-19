---
title: >-
  [论文解读] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions
description: >-
  [NeurIPS 2025][视频理解][Stochastic Differential Equations] 提出 Neural Stochastic Flows（NSF），通过条件归一化流直接学习 SDE 的转移分布 $p(x_t \mid x_s)$，在架构上约束满足随机流性质（恒等、Markov、Chapman-Kolmogorov），实现了无需数值求解器的单步采样，在远距时间点上加速高达两个数量级。
tags:
  - NeurIPS 2025
  - 视频理解
  - Stochastic Differential Equations
  - Normalising Flow
  - Solver-Free
  - Transition Distribution
  - State-Space Model
---

# Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions

**会议**: NeurIPS 2025  
**arXiv**: [2510.25769](https://arxiv.org/abs/2510.25769)  
**代码**: [项目页面](https://nkiyohara.github.io/nsf-neurips2025/) (有)  
**领域**: 序列建模 / 随机微分方程  
**关键词**: Stochastic Differential Equations, Normalising Flow, Solver-Free, Transition Distribution, State-Space Model

## 一句话总结

提出 Neural Stochastic Flows（NSF），通过条件归一化流直接学习 SDE 的转移分布 $p(x_t \mid x_s)$，在架构上约束满足随机流性质（恒等、Markov、Chapman-Kolmogorov），实现了无需数值求解器的单步采样，在远距时间点上加速高达两个数量级。

## 研究背景与动机

**领域现状**: 随机微分方程（SDE）广泛应用于金融、物理和机器学习中，用于建模带噪声的时间序列。传统 Neural SDE 方法需要数值求解器逐步模拟轨迹，计算成本与时间间隔成正比。

**现有痛点**:
   - **数值求解器开销大**：Neural SDE 训练和推理均依赖逐步仿真，长时间预测时计算量巨大
   - **Neural Flow 局限**：已有的无求解器方法（Neural Flow for ODE）无法表达随机动态
   - **扩散模型方法不通用**：Consistency Model 等加速方法仅适用于特定边界条件的扩散过程，不能处理一般 SDE

**核心矛盾**: 需要一般 Itô SDE 的转移分布（用于不确定性量化和概率预测），但数值求解器在远距时间点上代价极高，而现有的无求解器方法要么不支持随机动态，要么不具通用性。

**本文目标**: 设计一个神经网络架构，直接参数化 SDE 的弱解（转移分布），满足随机流的关键性质，实现训练和推理均无需求解器。

**切入角度**: 从随机流微分同胚理论出发，将强解条件转化为弱解（概率分布）的条件，用条件归一化流参数化。

**核心 idea**: 用架构约束保证恒等性和 Markov 性，用双向 KL 正则化损失鼓励 Chapman-Kolmogorov 流性质，从而直接学习 SDE 的转移分布。

## 方法详解

### 整体框架

NSF 由两部分组成：
1. 一个以 $(x_{t_i}, \Delta t, t_i)$ 为条件的参数高斯初始化
2. 一系列仿射耦合层组成的双射变换

采样过程：$\varepsilon \sim \mathcal{N}(0, I) \rightarrow z \rightarrow x_{t_j} = f_\theta(z, c)$

### 关键设计

#### 1. 条件归一化流架构

**基础高斯分布**（类似 Euler-Maruyama 离散化）：

$$z = \underbrace{x_{t_i} + \Delta t \cdot \text{MLP}_\mu(c; \theta_\mu)}_{\mu(c)} + \underbrace{\sqrt{\Delta t} \cdot \text{MLP}_\sigma(c; \theta_\sigma)}_{\sigma(c)} \odot \varepsilon$$

漂移项缩放 $\Delta t$，扩散项缩放 $\sqrt{\Delta t}$，与 SDE 的物理直觉一致。

**仿射耦合层**：每层将状态分为 $(z_A, z_B)$，对 $z_B$ 做仿射变换：

$$f_i(z; c, \theta_i) = \text{Concat}(z_A, z_B \odot \exp(\Delta t \cdot \text{MLP}_{\text{scale}}^{(i)}) + \Delta t \cdot \text{MLP}_{\text{shift}}^{(i)})$$

关键：所有变换参数乘以 $\Delta t$，确保 $\Delta t = 0$ 时为恒等映射（**恒等性质**）。

#### 2. 流性质正则化损失

为鼓励 Chapman-Kolmogorov 性质 $p(x_{t_k} \mid x_{t_i}) = \int p(x_{t_k} \mid x_{t_j}) p(x_{t_j} \mid x_{t_i}) dx_{t_j}$，设计双向 KL 散度的变分上界：

- **前向 KL** $\mathcal{L}_{\text{flow}, 1\text{-to-}2}$：用桥分布 $b_\xi(x_{t_j} \mid x_{t_i}, x_{t_k})$ 作为辅助变分分布
- **反向 KL** $\mathcal{L}_{\text{flow}, 2\text{-to-}1}$：反方向匹配

总损失：

$$\mathcal{L}(\theta, \xi) = -\mathbb{E}[\log p_\theta(x_{t_j} \mid x_{t_i})] + \lambda \mathcal{L}_{\text{flow}}(\theta, \xi)$$

#### 3. Latent NSF（处理部分观测）

对于噪声或部分观测数据，引入变分状态空间模型框架：
- **生成模型**：$p(x_{t_0:T}, o_{t_0:T}) = p(x_{t_0}) \prod_i p_\theta(x_{t_i} \mid x_{t_{i-1}}) p_\psi(o_{t_i} \mid x_{t_i})$
- **变分后验**：GRU 编码器 $q_\phi(x_{t_i} \mid o_{\leq t_i})$
- **跳步 KL 损失** $\mathcal{L}_{\text{skip}}$：利用 NSF 直接跨任意时间间隔采样，无需递归转移

总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\beta\text{-NELBO}} + \lambda \mathcal{L}_{\text{flow}} + \beta_{\text{skip}} \mathcal{L}_{\text{skip}}$

### 损失函数/训练策略

- 桥分布 $b_\xi$ 单独优化 $K$ 步内循环
- 主模型参数与桥参数交替优化
- 时间三元组 $(t_i, t_j, t_k)$ 从数据分布中采样

## 实验关键数据

### 主实验：随机 Lorenz 吸引子

| 方法 | KL ($t=0.25$) | KL ($t=1.0$) | kFLOPs ($t=1.0$) |
|------|--------------|-------------|------------------|
| Latent SDE | 2.1±0.9 | 1.5±0.5 | 3,760 |
| Neural LSDE | 1.3±0.4 | 53.1±29.3 | 6,699 |
| SDE matching (Δt=0.0001) | 4.3±0.7 | 3.8±1.0 | 737,354 |
| **NSF (H_pred=1.0)** | **0.8±0.7** | **0.2±0.6** | **53** |

NSF 单步采样仅需 **53 kFLOPs**，比 SDE matching 少 **~14,000 倍**。运行时间：NSF 0.3ms vs Latent SDE 124-148ms/batch。

### 主实验：CMU 动作捕捉

| 方法 | Setup 1 MSE | Setup 2 MSE |
|------|-------------|-------------|
| Latent ODE | 5.98±0.28 | 31.62±0.05 |
| Latent SDE | 12.91±2.90 | 9.52±0.21 |
| SDE matching | 5.20±0.43 | 4.26±0.35 |
| **Latent NSF** | **8.62±0.32** | — |

运行时间：Latent NSF 3.5ms vs Latent SDE 75ms/batch（加速 ~21 倍）。

### 消融实验

- 不同 $H_{\text{pred}}$ 的影响：$H_{\text{pred}}=1.0$（单步）在长时间 KL 最低；$H_{\text{pred}}=0.25$（需递归）在短时间更精确但长时间 FLOPs 增加
- 流损失 $\mathcal{L}_{\text{flow}}$ 的作用：显著提升分布一致性

### 关键发现

1. NSF 在随机 Lorenz 吸引子上的分布精度超越所有 solver-based 基线，同时 FLOPs 减少 1-2 个数量级
2. 单步采样在长时间预测上优势最大（正是数值求解器最昂贵的场景）
3. 流损失确保了单步与多步转移的分布一致性

## 亮点与洞察

1. **理论优雅**：从随机流微分同胚理论到弱解条件的转化严谨且自然
2. **架构设计精巧**：$\Delta t$ 乘以所有变换参数确保恒等性，自治 SDE 省略 $t_i$ 确保平稳性
3. **通用性**：处理一般 Itô SDE（不限于扩散模型的特定边界条件），填补了重要空白
4. **双向 KL + 桥分布**：巧妙解决了 Chapman-Kolmogorov 方程中边缘化不可处理的问题
5. **加速效果最显著的场景**恰好是最需要的场景（远距时间点预测）

## 局限与展望

1. **CMU 数据集性能未达最佳**：Latent NSF 的 MSE（8.62）不如 SDE matching（5.20）和 NCDSSM（5.69），可能是变分推断的瓶颈
2. **训练复杂度**：桥分布的内循环优化和时间三元组采样增加了训练开销
3. **条件归一化流的表达力上限**：对高维或多模态转移分布可能不够灵活
4. **未验证高维系统**：实验维度较低（Lorenz 3D、Motion Capture 50D），更高维系统需进一步验证

## 相关工作与启发

- **Neural Flow (Biloš et al.)**: ODE 的无求解器方法，本文扩展到 SDE
- **Consistency Models**: 扩散模型加速，但限于特定边界条件
- **Latent SDE (Li et al.)**: solver-based 基线，NSF 在精度相当的情况下大幅加速
- **SDE Matching (Bartosh et al.)**: solver-free 训练但 solver-dependent 推理，NSF 训练和推理均 solver-free
- **变分状态空间模型**: Latent NSF 的框架基础

## 评分

⭐⭐⭐⭐

理论框架完整，方法优雅。在合成数据上效果出色，但在真实数据（CMU）上的性能优势不够突出。总体上是 SDE 建模领域的重要贡献。

<!-- RELATED:START -->

## 相关论文

- [Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](../../ICCV2025/video_understanding/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)
- [GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)
- [FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks](../../ICML2025/video_understanding/fastcav_efficient_computation_of_concept_activation_vectors_for_explaining_deep_.md)

<!-- RELATED:END -->
