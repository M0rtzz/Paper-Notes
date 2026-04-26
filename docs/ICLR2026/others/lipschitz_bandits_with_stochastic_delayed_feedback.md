---
title: >-
  [论文解读] Lipschitz Bandits with Stochastic Delayed Feedback
description: >-
  [ICLR 2026][Lipschitz bandit] 首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。
tags:
  - ICLR 2026
  - Lipschitz bandit
  - 延迟反馈
  - zooming 算法
  - 分阶段剪枝
  - 遗憾下界
  - 分位数
---

# Lipschitz Bandits with Stochastic Delayed Feedback

**会议**: ICLR 2026  
**arXiv**: [2510.00309](https://arxiv.org/abs/2510.00309)  
**代码**: 无  
**领域**: 在线学习 / Bandit 算法  
**关键词**: Lipschitz bandit, 延迟反馈, zooming 算法, 分阶段剪枝, 遗憾下界, 分位数

## 一句话总结

首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。

## 研究背景与动机

**领域现状**：Lipschitz bandit 将经典 MAB 扩展到连续度量空间 $(\mathcal{A}, \mathcal{D})$，奖励函数 $\mu$ 满足 1-Lipschitz 条件。经典 Zooming 算法通过自适应离散化在无延迟设定下达到最优遗憾率 $\tilde{O}(T^{(d_z+1)/(d_z+2)})$，其中 $d_z$ 为 zooming 维度。延迟反馈在 MAB、线性 bandit、kernel bandit 中已有广泛研究，但 Lipschitz bandit 的延迟问题完全空白。

**现有痛点**：(1) 连续臂空间 + 延迟反馈产生双重复杂性——每个采样点代表一个邻域区域，其估计依赖延迟的观测；(2) Zooming 算法核心分析依赖"置信半径仅在拉臂时变化"——延迟下此性质被打破，因为延迟奖励到达时未拉臂的置信半径也会缩小；(3) 有限臂延迟方法（如 Delayed-UCB1）分析不涉及连续空间的覆盖论证，无法直接推广。

**核心矛盾**：如何在反馈延迟且可能永远缺失的情况下，仍然保持对连续度量空间的高效自适应探索？

**本文目标** 设计在有界/无界随机延迟下均能达到近最优遗憾率的 Lipschitz bandit 算法，并证明其最优性。

**切入角度**：将问题按延迟支撑分为两个子问题——有界延迟用"zooming + lazy update"，无界延迟用"分阶段累积可靠反馈 + 淘汰"。

**核心 idea**：通过 lazy update 控制置信半径缩小速度（有界延迟），或通过分阶段 round-robin 采样和 $Q(p)$ 分位数联系拉臂/观测次数（无界延迟），恢复无延迟最优遗憾率。

## 方法详解

### 整体框架

问题定义为三元组 $(\mathcal{A}, \mathcal{D}, \mu)$：$\mathcal{A}$ 是紧致倍增度量空间，$\mathcal{D}$ 为度量，$\mu: \mathcal{A} \to [0,1]$ 是 1-Lipschitz 未知奖励函数。每轮 $t$ 选臂 $x_t$，生成奖励 $y_t = \mu(x_t) + \epsilon_t$（$\epsilon_t$ 为 sub-Gaussian 噪声），但奖励在随机延迟 $\tau_t$ 后才被观测到。延迟 $\tau_t \sim f_\tau$ 独立于臂和奖励。累积遗憾为 $R(T) = \sum_{t=1}^T (\mu^* - \mu(x_t))$。针对有界和无界延迟分别提出两种算法。

### 关键设计

1. **Delayed Zooming 算法（有界延迟 $\tau_t \leq \tau_{\max}$）**:
    - 功能：保持 Zooming 算法的自适应离散化优势，同时处理延迟导致的信息流紊乱
    - 核心思路：用观测次数 $v_t(x)$ 替代拉臂次数 $n_t(x)$ 计算置信半径 $r_t(x) = \sqrt{\frac{4\log T + 2\log(2/\delta)}{1 + v_t(x)}}$，并引入 **lazy update 机制**——为每个活跃臂维护缓存队列 $Q[x]$，当 $v_t(x)+1 > 4 v_s(x)$（$s$ 为上次拉臂时间）时，将到达的反馈缓存而不更新。下次拉该臂时一并处理缓存。这确保 $r_t(x) \geq \frac{1}{2} r_s(x)$，从而恢复子最优 gap 界 $\Delta(x) \leq 6r_t(x)$（经典无延迟为 $3r_t(x)$）
    - 设计动机：延迟导致未拉臂时 $r_t(x)$ 因延迟奖励到达而缩小，破坏经典 zooming 分析核心。Lazy update 通过控制观测次数增长速率解决此问题。遗憾界为 $\tilde{O}\big(T^{\frac{d_z+1}{d_z+2}} + \tau_{\max} T^{\frac{d_z}{d_z+2}}\big)$

2. **DLPP 算法（无界延迟，含反馈缺失 $\tau = \infty$）**:
    - 功能：处理延迟可能无界甚至永远缺失的场景，提供近最优遗憾保证
    - 核心思路：分阶段学习，第 $m$ 阶段维护半径 $r_m = 2^{-m}$ 的覆盖球集合 $\mathcal{B}_m$。对每个球进行均匀 round-robin 采样，直到累积 $v_m = \frac{2\log T + \log(2/\delta)}{2r_m^2}$ 个观测后停止该球的采样。阶段结束时淘汰经验均值远低于最佳球的区域（剪枝规则：$\hat\mu_m^* - \hat\mu_m(B) \geq 8r_m$），对幸存区域进一步细分。通过 Chernoff 不等式建立拉臂次数与延迟后观测次数的概率联系：$\Pr(v_{t+Q(p)}(B) \leq \frac{p}{2} n_t(B)) \leq \exp(-\frac{p}{8} n_t(B))$，将遗憾与延迟分位数 $Q(p)$ 挂钩
    - 设计动机：DLPP 不依赖实时反馈更新，而是等待足够统计量后做决策，即使部分反馈缺失也能工作。遗憾界为 $R(T) \lesssim \min_{p \in (0,1]} \left\{ \frac{1}{p} T^{\frac{d_z+1}{d_z+2}} (c\log\frac{T}{\delta})^{\frac{1}{d_z+2}} + Q(p) \right\}$

3. **实例相关遗憾下界**:
    - 功能：证明 DLPP 的遗憾率是近最优的（至对数因子）
    - 核心思路：构造特殊延迟分布（延迟为固定值 $\tau_0$ 的概率 $p$，否则 $\infty$），通过 Bernoulli 采样耦合将延迟 Lipschitz bandit 归约到无延迟版本。得到下界 $R(T) \gtrsim \frac{T^{(d_z+1)/(d_z+2)}(c\log T)^{1/(d_z+2)}}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$，其中 $\bar\Delta = \int_\mathcal{A} \Delta(x) / \int_\mathcal{A} 1$ 为平均子最优 gap
    - 设计动机：最后一项 $\bar\Delta \cdot Q(p)$ 来自前 $Q(p)$ 轮完全无反馈时的不可避免遗憾，与上界形式匹配

### 损失函数 / 训练策略

两个算法均为理论算法，无需梯度训练。关键参数选择均由理论推导确定：Delayed Zooming 的 lazy update 阈值 $4v_s(x)$、DLPP 每阶段所需观测数 $v_m$ 与剪枝阈值 $8r_m$。遗憾率的最优化由选择 $\rho = (\log T / T)^{1/(d_z+2)}$（有界延迟）或 $M = \frac{\log(T/(c\log T))}{d_z+2}$（DLPP）得到。

## 实验关键数据

### 主实验（三种奖励函数 × 两种延迟分布，$T=60000$，30 次独立试验）

| 奖励函数 | 算法 | 无延迟 | 均匀 $\mathbb{E}[\tau]=20$ | 均匀 $\mathbb{E}[\tau]=50$ | 几何 $\mathbb{E}[\tau]=20$ | 几何 $\mathbb{E}[\tau]=50$ |
|----------|------|:------:|:------:|:------:|:------:|:------:|
| 三角函数(1D) | Delayed Zooming | 138.97 | 154.55 | 171.07 | 159.30 | 152.98 |
| 三角函数(1D) | DLPP | 304.60 | 314.87 | 326.71 | 312.44 | 325.74 |
| 正弦函数(1D) | Delayed Zooming | 130.64 | 137.31 | 148.69 | 132.88 | 144.08 |
| 正弦函数(1D) | DLPP | 178.05 | 195.35 | 209.97 | 186.28 | 208.80 |
| 二维函数(2D) | Delayed Zooming | 1445.86 | 1843.05 | 1858.45 | 1463.38 | 1828.15 |
| 二维函数(2D) | DLPP | **1120.64** | **1159.85** | **1136.46** | **1120.63** | **1142.55** |

### 理论结果对比

| 设定 | 算法 | 遗憾上界 | 退化检验 |
|------|------|----------|----------|
| 有界延迟 $\tau_{\max}$ | Delayed Zooming | $\tilde{O}(T^{(d_z+1)/(d_z+2)} + \tau_{\max} T^{d_z/(d_z+2)})$ | $\tau_{\max}=0$ 退化为经典 Lipschitz bandit |
| 有界延迟 $d_z=0$ | Delayed Zooming | $O(\sqrt{cT\log T} + c\tau_{\max})$ | 退化为有限臂 MAB with delay |
| 无界延迟 | DLPP | $\min_p \{ \frac{1}{p} T^{(d_z+1)/(d_z+2)} (\cdot)^{1/(d_z+2)} + Q(p) \}$ | $p=0.5$ 时依赖中位数 $\tau_{\text{med}}$ |
| 下界 | — | $\frac{R}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$ | 与 DLPP 上界至对数因子匹配 |

### 关键发现

- 两种算法在有界/无界延迟下均保持次线性遗憾，延迟引起的额外遗憾仅为加性项
- 1D 场景 Delayed Zooming 遗憾更低，2D 场景 DLPP 剪枝+离散化策略更优（2D 无延迟 DLPP 已优于 Zooming）
- Delayed Zooming 在几何分布（无界）延迟实验中也工作良好，但理论保证仅限有界延迟——这是一个 open problem
- DLPP 遗憾曲线呈分段线性——源于阶段内均匀采样的阶段性遗憾累积

## 亮点与洞察

- **Lazy Update 的技术贡献**：看似简单的"用 $v_t$ 替代 $n_t$ + 缓存"，但分析极其非平凡。分析证明 $\Delta(x) \leq 6r_t(x)$ 的常数因子翻倍是精确的——当 $v_t(x)+1 > 4v_s(x)$ 时 $r_t(x)$ 恰好可能降至 $r_s(x)/2$ 以下，必须停止更新
- **分位数刻画的优雅性**：DLPP 遗憾界通过 $\min_{p \in (0,1]}$ 优化分位数，适应延迟分布的"中心质量"而非最坏情况——取 $p=0.5$ 得中位数依赖，取更小 $p$ 可处理重尾
- **上下界匹配**：下界构造使用 Bernoulli 耦合技巧——以概率 $p$ 模拟无延迟算法，证明延迟遗憾至少为 $\Omega(R/(p\log T))$，加上前 $Q(p)$ 轮无信息的 $\bar\Delta \cdot Q(p)$ 项，展示 DLPP 几乎不可改进

## 局限与展望

- Delayed Zooming 理论保证仅限有界延迟，扩展到无界延迟是论文明确指出的 open problem
- DLPP 需要覆盖 oracle，高维空间覆盖计算可能代价高昂
- 实验仅在 1D 和 2D 合成函数上验证（三角/正弦/二维双峰），缺乏更现实的应用场景
- 两种算法均需知晓时间窗口 $T$，未讨论 anytime 版本
- 未讨论对抗性延迟（adversarial delay）场景

## 相关工作与启发

- **vs Delayed-UCB1 (Joulani et al., 2013)**: 有限臂延迟 MAB，分析不涉及覆盖-激活机制，直接推广不可行
- **vs BLiN (Feng et al., 2022)**: 首个 batched Lipschitz bandit，但依赖即时/批量反馈且限于 $[0,1]^d$ 空间和轴对齐 cube 划分
- **vs Lancewicki et al. (2021)**: 无界延迟 MAB 通过分位数函数刻画遗憾，本文将此推广到连续空间并证明了 DLPP 的匹配下界

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次研究 Lipschitz bandit + 延迟反馈的交叉问题，但核心技术路线基于已有方法的组合扩展
- 实验充分度: ⭐⭐⭐⭐ 理论完整（上界+匹配下界），实验验证充分但场景偏合成
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，两种算法层次分明，定理陈述简洁优美
- 价值: ⭐⭐⭐⭐ 对 bandit 理论有扎实的基础贡献，填补了重要的理论空白

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Adversarial Combinatorial Semi-bandits with Graph Feedback](../../ICML2025/others/adversarial_combinatorial_semi-bandits_with_graph_feedback.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)

<!-- RELATED:END -->
