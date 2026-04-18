---
title: >-
  [论文解读] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules
description: >-
  [NeurIPS 2025][优化][scaling law] 在幂律核回归模型中建立了 Functional Scaling Law (FSL)，通过引入"内在时间"概念统一刻画任意学习率调度下的完整 loss 轨迹，并推导出常数/指数衰减/WSD 三种调度在数据受限和计算受限条件下的显式 scaling 关系，理论解释了 WSD 优于纯衰减的经验现象。
tags:
  - NeurIPS 2025
  - 优化
  - scaling law
  - 学习率调度
  - 核回归
  - loss 动力学
  - WSD schedule
---

# Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules

**会议**: NeurIPS 2025  
**arXiv**: [2509.19189](https://arxiv.org/abs/2509.19189)  
**代码**: 无  
**领域**: optimization  
**关键词**: scaling law, 学习率调度, 核回归, loss 动力学, WSD schedule

## 一句话总结
在幂律核回归模型中建立了 Functional Scaling Law (FSL)，通过引入"内在时间"概念统一刻画任意学习率调度下的完整 loss 轨迹，并推导出常数/指数衰减/WSD 三种调度在数据受限和计算受限条件下的显式 scaling 关系，理论解释了 WSD 优于纯衰减的经验现象。

## 研究背景与动机

**领域现状**：Kaplan 等人发现 LLM 预训练 loss 服从 $L(M,D) = L_0 + C_M M^{-\alpha_M} + C_D D^{-\alpha_D}$ 的幂律关系，scaling law 已成为指导 LLM 开发的基础原则。理论解释工作主要在核回归/线性回归代理模型上进行。

**现有痛点**：
   - 现有理论仅针对终止 loss，未解释完整 loss 轨迹是否也遵循 scaling law
   - 学习率调度(LRS)的作用未被系统刻画——gradient flow、常数 LR SGD、指数衰减已分别研究，但缺乏统一理论
   - WSD (Warmup-Stable-Decay) 调度在实践中（DeepSeek-V3, Kimi-K2）表现优异，但其优势机制不清

**核心矛盾**：LRS 同时影响信号学习和噪声注入/消散，不同调度策略在这两个效应间的权衡不同，需要统一框架

**切入角度**：引入"内在时间"$t = \int_0^\tau \varphi(r) dr$（学习率累积），将 LRS 的影响从 SDE 的漂移项解耦，仅保留在扩散项中

**核心idea**：FSL 将 loss 分解为不可约误差 + 近似误差 + 信号学习项 + 噪声卷积项，其中 LRS 仅通过卷积泛函进入最后一项

## 方法详解

### 整体框架
考虑幂律核(PLK)回归模型：$y = ⟨\boldsymbol{\phi}(\mathbf{x}), \boldsymbol{\theta}^*⟩ + \epsilon$，特征协方差谱 $\lambda_j \asymp j^{-\beta}$（容量指数 $\beta > 1$），目标系数衰减 $|\theta_j^*|^2 \asymp j^{-1}\lambda_j^{s-1}$（难度指数 $s > 0$）。用宽度 $M$ 的模型通过 one-pass SGD 优化。

### 关键设计

1. **内在时间重参数化**:

    - 功能：将物理时间 $\tau$ 按 LRS 重缩放为内在时间 $t$
    - 核心思路：定义 $t = T(\tau) = \int_0^\tau \varphi(r) dr$，变量替换后原始 SDE $d\bar{\mathbf{v}}_\tau = -\varphi(\tau)\nabla\mathcal{R} d\tau + \varphi\sqrt{h/b \cdot \Sigma} d\mathbf{B}_\tau$ 变为 $d\boldsymbol{\nu}_t = -\nabla\mathcal{R} dt + \sqrt{\gamma(t)\Sigma} d\mathbf{B}_t$，其中 $\gamma(t) = h/(\varphi \cdot b)$ 在内在时间下。LRS 从漂移项完全消失
    - 设计动机：内在时间比迭代步数更忠实地反映训练进度，使得信号学习部分与 LRS 无关

2. **Functional Scaling Law (FSL)**:

    - 功能：刻画完整 loss 轨迹而非仅终止 loss
    - 核心思路：
    $\mathbb{E}[\mathcal{R}(\boldsymbol{\nu}_t)] - \frac{\sigma^2}{2} \asymp M^{-s\beta} + e(t) + \int_0^t \mathcal{K}(t-z)[e(z)+\sigma^2]\gamma(z) dz$
   其中 $e(t) = (1+t)^{-s}$ 为信号学习项，$\mathcal{K}(t) = (1+t)^{-(2-1/\beta)}$ 为遗忘核
    - 设计动机：四项分别对应不可约误差、近似误差、信号学习、噪声积累/消散。LRS 仅通过卷积进入最后一项

3. **噪声结构分析 (Lemma 4.8)**:

    - 功能：精确刻画梯度噪声的各向异性结构
    - 核心思路：$(2\rho_- \mathcal{E}(\mathbf{v}) + \sigma^2)\nabla^2\mathcal{R} \preceq \Sigma(\mathbf{v}) \preceq (2\rho_+ \mathcal{E}(\mathbf{v}) + \sigma^2)\nabla^2\mathcal{R}$，噪声沿各方向的能量正比于 risk × 该方向曲率
    - 设计动机：这一结构使得噪声项可以用 loss 本身来界定，形成自洽的分析

4. **三种 LRS 的显式 scaling**:

    - 功能：对 constant/exponential-decay/WSD 三种调度推导闭式 scaling 关系
    - 核心思路：将 LRS 代入 FSL 的卷积项，分别计算数据受限（固定 $M$）和计算受限（$M$ 与 $D$ 联合优化）的最优分配
    - 设计动机：量化不同 LRS 的 scaling 效率差异，解释 WSD 的经验优势

### 训练策略
- 模型容量 $\beta > 1$，任务难度 $s > 0$，相对难度区分 easy ($s \geq 1-1/\beta$) vs hard ($s < 1-1/\beta$)
- 超收缩性假设(Assumption 2.1)保证四阶矩受二阶矩控制
- 支持 top-$M$ 特征选择和 random-$M$ 特征投影两种设置

## 实验关键数据

### 三种 LRS 的 Data-Optimal Scaling 对比

| LRS | Easy regime | Hard regime |
|-----|------------|------------|
| Constant | $D^{-s/(s+1)}$ | $D^{-s/(s+1)}$ |
| Exp-decay | $D^{-s\beta/(1+s\beta)}(\log D)^{s\beta/(1+s\beta)}$ | $D^{-s}(\log D)^s$ |
| WSD | $D^{-s\beta/(1+s\beta)}(\log D)^{(s\beta-s)/(1+s\beta)}$ | $D^{-s}$ |

### Compute-Optimal Scaling 对比

| LRS | Easy regime | Hard regime |
|-----|------------|------------|
| Constant | — | $C^{-s\beta/(1+s\beta+\beta)}$ |
| Exp-decay | $C^{-s\beta/(2+s\beta)}(\log C)^{...}$ | $C^{-s\beta/(1+\beta)}(\log C)^{...}$ |
| WSD | $C^{-s\beta/(2+s\beta)}(\log C)^{...}$ | $C^{-s\beta/(1+\beta)}$ |

### 关键发现
- **WSD > Exp-decay > Constant**：WSD 在 hard regime 去除了 $\log$ 因子，Constant 调度缺少额外 $\beta$ 因子
- **高容量模型更高效**：固定任务（$\alpha = s\beta$ 不变），增大 $\beta$（降低容量）时 $s$ 更大，信号学习更快
- **数据比模型更需扩展**：compute-optimal 训练中数据应比模型扩展更多
- **峰值学习率需与预算匹配**：最优峰值 LR 应随训练预算适当缩放
- LLM 实验 (0.1B-1B) 验证 FSL 可作为拟合和预测 loss 轨迹的代理模型

## 亮点与洞察
- **内在时间概念**：将 LRS 的影响从信号学习中完全解耦，是实现统一分析的关键创新。这一概念直觉上对应"有效训练步数"
- **遗忘核的物理直觉**：$\mathcal{K}(t) \asymp t^{-(2-1/\beta)}$ 说明高容量模型（小 $\beta$）遗忘注入噪声更慢——解释了为什么大模型需要更仔细的 LRS 设计
- **WSD 的优势机制**：长时间稳定阶段积累足够的信号学习，末尾衰减阶段高效消散累积噪声，两阶段互补
- **与 Multi-Power-Law 模型的联系**：通过分部积分，FSL 可变换为近似等价于 Tissue 等人的 MPL 经验模型

## 局限与展望
- 分析基于核回归（二次损失），与实际 LLM 的交叉熵损失有差距
- 连续时间 SDE 近似要求 step size 足够小，实际训练中该条件不一定成立
- random-$M$ 特征的分析仅覆盖 $s \leq 1$，对简单任务 $s > 1$ 的情况未解决
- FSL 的 $\eqsim$（常数因子）精度在实际拟合中可能不够——LLM 实验需要额外拟合参数

## 相关工作与启发
- **vs Bordelon & Pehlevan (2024)**：仅分析 gradient flow，本文覆盖任意 LRS 的 SGD
- **vs Sorscher et al. (2024) / Paquette et al.**：分析常数 LRS 的 SGD，本文统一并扩展
- **vs Lin et al. (2024)**：分析指数衰减 LRS，本文进一步覆盖 WSD
- **vs Tissue et al. (2024)**：经验 MPL 模型，本文提供理论解释

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 内在时间概念和 FSL 框架是重要的理论创新，统一了分散的先前工作
- 实验充分度: ⭐⭐⭐⭐ 核回归数值验证充分，LLM 实验(0.1B-1B)增强实践相关性
- 写作质量: ⭐⭐⭐⭐⭐ FSL 各项的物理解释极为清晰，理论和实践的桥梁搭建出色
- 价值: ⭐⭐⭐⭐⭐ 为理解和设计 LLM 学习率调度提供了坚实的理论基础
