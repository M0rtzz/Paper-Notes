---
title: >-
  [论文解读] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown
description: >-
  [NeurIPS 2025][Thompson Sampling] 首次系统性实证评估 Feel-Good Thompson Sampling (FG-TS) 及其平滑变体 SFG-TS 在近似后验下的表现，横跨线性/逻辑/神经三类上下文赌博机设置和十四个基准，发现 FG-TS 在精确后验场景（线性/逻辑）下优于标准 TS，但在神经赌博机中反而退化，揭示了乐观偏差与采样噪声之间的关键权衡。
tags:
  - NeurIPS 2025
  - Thompson Sampling
  - 上下文赌博机
  - MCMC
  - 探索-利用
  - 后验采样
---

# Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown

**会议**: NeurIPS 2025  
**arXiv**: [2507.15290](https://arxiv.org/abs/2507.15290)  
**代码**: [GitHub](https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown)  
**领域**: reinforcement_learning  
**关键词**: Thompson Sampling, 上下文赌博机, MCMC, 探索-利用, 后验采样

## 一句话总结
首次系统性实证评估 Feel-Good Thompson Sampling (FG-TS) 及其平滑变体 SFG-TS 在近似后验下的表现，横跨线性/逻辑/神经三类上下文赌博机设置和十四个基准，发现 FG-TS 在精确后验场景（线性/逻辑）下优于标准 TS，但在神经赌博机中反而退化，揭示了乐观偏差与采样噪声之间的关键权衡。

## 研究背景与动机

**领域现状**：Thompson Sampling (TS) 是上下文赌博机中最流行的探索-利用算法之一，实用性强、实现简单。但在高维场景下，TS 的探索力度不足，理论 regret 为 $O(d\sqrt{dT})$，未达到信息论下界 $\Omega(d\sqrt{T})$。

**现有痛点**：FG-TS（Zhang, 2022）通过在似然中添加"feel-good bonus"增加乐观偏差强制更积极探索，在线性设置下取得了最优 $O(d\sqrt{T})$ regret。但其理论分析假设精确后验，而实际中大规模或神经网络场景必须使用近似后验（如 MCMC），**FG-TS 在近似后验下的表现完全未知**。

**核心矛盾**：乐观偏差在精确后验时帮助探索，但当后验采样本身有噪声时，乐观偏差可能放大错误——两种噪声叠加可能导致决策退化。

**本文目标**：近似后验如何影响 FG-TS 的性能？在什么条件下 FG-TS 优于/劣于标准 TS？bonus 幅度、先验强度、预处理等超参数如何交互？

**切入角度**：构建从精确到粗糙后验的谱系——线性（闭式后验）→ 逻辑（近高斯）→ 神经（高度非线性），系统对比多种 MCMC 采样器。

**核心 idea**：第一个系统性 FG-TS benchmark，揭示"bonus 幅度 × 后验精度"的权衡是性能的决定性因素。

## 方法详解

### 整体框架
在每轮 $t$，智能体观测上下文 $\mathcal{X}_t \subseteq \mathbb{R}^d$，从后验分布采样模型参数 $\theta_t$，贪心选择动作 $x_t = \arg\max_{x \in \mathcal{X}_t} f_{\theta_t}(x)$，观测奖励 $r_t$ 并更新后验。核心区别在于后验的似然函数。

### 关键设计

1. **Feel-Good Thompson Sampling (FG-TS)**:

    - 功能：在标准 TS 的似然中加入对高奖励模型的偏好（乐观偏差）
    - 核心思路：修改似然函数为 $L^{\text{FG}}(\theta, x, r) = \eta(f_\theta(x) - r)^2 - \lambda \min(b, f_\theta(x))$，其中 $\lambda > 0$ 控制 bonus 幅度，$b$ 是截断上界。负号使高 $f_\theta$ 的模型更被倾向采样
    - 设计动机：标准 TS 在高维中探索不足，加入乐观偏差让采样偏向"乐观"方向，理论上将 regret 从 $O(d\sqrt{dT})$ 降到最优 $O(d\sqrt{T})$

2. **Smoothed Feel-Good TS (SFG-TS)**:

    - 功能：将 FG-TS 的 min 截断平滑化，使后验更适合 MCMC 采样
    - 核心思路：用 softplus 替换 min：$L^{\text{SFG}}(\theta, x, r) = \eta(f_\theta(x) - r)^2 - \lambda(b - \Phi_s(b - f_\theta^\star))$，其中 $\Phi_s(u) = \log(1 + \exp(su))/s$
    - 设计动机：min 操作导致后验不光滑，MCMC 采样困难；softplus 平滑化后保持理论 regret 保证的同时改善采样质量

3. **MCMC 采样器家族**:

    - **LMC (Langevin MC)**：$\theta_{t,k+1} = \theta_{t,k} - \eta_t \nabla \mathcal{L}_t(\theta_{t,k}) + \sqrt{2\eta_t \beta_t^{-1}} \epsilon_{t,k}$，最基础的梯度+噪声采样
    - **MALA (Metropolis-Adjusted LMC)**：在 LMC 基础上加 Metropolis 接受/拒绝步骤修正离散化偏差
    - **HMC (Hamiltonian MC)**：引入动量变量模拟哈密顿动力学，$H(\theta, v) = \mathcal{L}_t(\theta) + \frac{1}{2}\|v\|^2$，理论上混合更快
    - **Preconditioned 变体**：用设计矩阵 $\mathbf{V}_t^{-1}$ 预处理梯度，将条件数 $\kappa_t$ 的影响降一个量级
    - **SVRG 变体**：方差缩减梯度估计，用控制变量降低随机梯度噪声

4. **Underdamped LMC (ULMC)**:

    - 功能：引入动量项和阻尼系数的 Langevin 变体
    - 核心思路：$v_{t,k+1/2} = (1-\gamma\eta)v_{t,k} - \eta\nabla U(\theta_{t,k}) + \sqrt{2\gamma\eta}\xi_{t,k}$，额外的速度变量让采样器在粗糙landscape上混合更快
    - 设计动机：对比 overdamped LMC 在复杂后验上的表现差异

### 实验设置
- 线性赌博机：$d=20/40$, $K=5$ 臂, $T=10000$, 高斯噪声 $\sigma=0.5$
- 逻辑赌博机：$d=20$, $K=50$ 臂, $T=10000$, Bernoulli 奖励
- 神经赌博机：UCI 数据集（Adult, Mushroom, Shuttle, MNIST 等），2/4 层 MLP
- 10 seeds（线性/逻辑）/ 5 seeds（神经）

## 实验关键数据

### 主实验（部分线性与逻辑结果）

| 算法 | Linear-20d ($\beta=10^3$) | Linear-40d | Logistic-20d |
|------|--------------------------|------------|-------------|
| LinUCB | 73.0 ± 13.8 | 126.3 ± 19.3 | 176.9 ± 41.9 |
| LinTS | 114.7 ± 8.8 | 204.6 ± 19.1 | 179.9 ± 53.2 |
| LMCTS | 62.6 ± 9.5 | 129.1 ± 16.1 | 202.7 ± 44.1 |
| MALATS | **61.3 ± 26.6** | **100.6 ± 10.0** | 194.0 ± 76.9 |
| FGLMCTS | 213.0 ± 126.0 | 163.6 ± 22.4 | **184.8 ± 38.0** |
| PHMCTS | 90.0 ± 9.2 | 162.2 ± 12.5 | 218.9 ± 16.1 |
| SFGMALATS | 189.3 ± 135.3 | 142.1 ± 19.5 | 198.4 ± 52.3 |

### 神经赌博机关键结果

| 数据集 | LMCTS | FGLMCTS | SFGLMCTS | Neural-εGreedy | NeuralUCB |
|--------|-------|---------|----------|---------------|-----------|
| Adult | 2456.6 | 3505.0 | 4505.6 | 2658.0 | **2444.4** |
| Mushroom | 324.6 | **283.2** | 440.6 | 124.0 | 145.6 |
| Shuttle | **210.2** | 214.4 | 1503.0 | 372.4 | 2981.2 |
| MNIST | **2854.6** | 2542.6 | 2935.0 | 3248.0 | 5442.8 |

### 消融实验

| 消融维度 | 发现 |
|---------|------|
| Feel-good bonus $\lambda$ | $\lambda=0.5$ 在线性设置无改善；$\lambda=0.01$ 时 SFGMALATS 降到 56.2±22.8 优于所有 vanilla TS |
| 预处理（Preconditioning） | 预处理 HMC 在线性中有效（90.0 vs 241.2），但预处理 LMC 反而更差（134.4 vs 62.6） |
| 逆温度 $\beta$ | $\beta=10^3$ 配合衰减调度通常优于 $\beta=1$ |
| SVRG | 在 $d=20$ 有效（73.2 vs 62.6 可比），但 $d=40$ 崩溃（19236.8） |

### 关键发现
- **FG-TS 在线性/逻辑赌博机中有效**（尤其低 $\lambda$ 时），MALATS 整体表现最佳
- **FG-TS 在神经赌博机中普遍退化**：近似后验的噪声与乐观偏差叠加导致决策质量下降
- **核心权衡**：bonus 越大在精确后验时越好，但在近似后验时越有害
- **预处理是把双刃剑**：改善 HMC 但可能破坏 LMC 的自然扩散行为
- Neural-εGreedy 和 NeuralUCB 在神经场景中更稳健

## 亮点与洞察
- **"bonus × 后验精度"权衡的发现**：这是本文最核心的实证贡献——精确后验越好、bonus 越有帮助；近似后验越差、bonus 越有害。这一洞察对所有乐观探索方法都有指导意义
- **全面的 MCMC 采样器对比**：LMC/MALA/HMC/ULMC/SVRG 及其预处理变体的系统对比，是上下文赌博机领域最全面的 MCMC benchmark
- **开源框架**：PyTorch 实现所有算法，可复现且可扩展，对后续研究有实际价值

## 局限与展望
- **神经场景中 FG-TS 退化的根因分析不够深入**——是梯度估计偏差、后验多模态性还是网络容量问题？
- 未考虑非参数后验方法（如粒子滤波、BNN）作为替代
- 线性和逻辑设置中 FG-TS 的改善幅度不大，部分结果方差很大
- 未测试变化的 bonus 调度策略（如 $\lambda$ 随 $t$ 衰减）
- 实际推荐系统等应用验证缺失

## 相关工作与启发
- **vs Zhang (2022) FG-TS 原论文**：原文只做了精确后验的理论分析，本文补充了近似后验的实证全景，发现理论优势在实践中是有条件的
- **vs Riquelme et al. (2018)**：经典 bandit benchmark 论文，本文沿用其实验协议但聚焦于 FG-TS 维度的扩展评估
- **vs Xu et al. (2022)**：LMC-TS 原论文，本文在其基础上加入 FG 变体和更多采样器对比
- 对自己的研究启发：任何在理论上通过修改目标函数改善性能的方法，都应在近似求解条件下重新评估

## 评分
- 新颖性: ⭐⭐⭐ 方法层面无新设计（HMC for SFG-TS 是新的），核心贡献在实证
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个基准、~20 种算法变体、多维消融，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰、背景详尽，但部分表格太密集难以快速阅读
- 价值: ⭐⭐⭐⭐ "bonus × 后验精度"的权衡是实用且普适的洞察，benchmark + 代码有长期价值

<!-- RELATED:START -->

## 相关论文

- [Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [Exploration via Feature Perturbation in Contextual Bandits](exploration_via_feature_perturbation_in_contextual_bandits.md)

<!-- RELATED:END -->
