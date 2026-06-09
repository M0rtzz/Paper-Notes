---
title: >-
  [论文解读] Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces
description: >-
  [ICML 2026][文本生成][MCMC] SRMC 用一个 $d$ 维的 running score 平均（而不是 $|\mathcal{X}|$ 维的经验测度）来记录历史，再通过指数 score-tilt 把这段历史折成一个"排斥已访问区域"的代理目标 $\pi_\theta$…
tags:
  - "ICML 2026"
  - "文本生成"
  - "MCMC"
  - "非马尔可夫采样"
  - "score-tilt"
  - "自排斥"
  - "随机近似 CLT"
---

# Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces

**会议**: ICML 2026  
**arXiv**: [2604.22948](https://arxiv.org/abs/2604.22948)  
**代码**: 待确认  
**领域**: 科学计算 / MCMC / 概率推断  
**关键词**: MCMC、非马尔可夫采样、score-tilt、自排斥、随机近似 CLT  

## 一句话总结
SRMC 用一个 $d$ 维的 running score 平均（而不是 $|\mathcal{X}|$ 维的经验测度）来记录历史，再通过指数 score-tilt 把这段历史折成一个"排斥已访问区域"的代理目标 $\pi_\theta$，套在任何 base MCMC kernel 外面，就能在通用状态空间下用常数内存实现非马尔可夫、低方差、保持归一化无关性的采样器。

## 研究背景与动机
**领域现状**：MCMC 是从 Bayesian 推断到 EBM 采样的主力工具，但越复杂的目标（多模 posterior、崎岖能量面、大离散配置空间）越容易陷入"理论遍历、实际困住"的窘境：链反复在同一片区域震荡，样本强相关、估计不可靠。近年提升 sampling efficiency 的工作主要在 refine Markov kernel 本身——离散域用 locally informed proposal / balancing（Zanella、GWG），连续域用 Langevin、HMC、non-reversible 采样器。

**现有痛点**：上面这些做法都是 memoryless 的——kernel 不记得"我已经来过这里 100 次了"。Non-Markovian 路线在有限状态空间下有干净的理论：SRRW、HDT 用经验测度 $\hat{\delta}_n = \frac{1}{n+1}\sum_i \delta_{X_i}$ 喂回 kernel，可以在排斥强度 $\alpha$ 增大时达到 near-zero variance，而且保持 normalization-free。但 $\hat{\delta}_n$ 是 $|\mathcal{X}|$ 维对象：在 $\{0,1\}^d$ 上指数大，在连续域上根本是无限维测度，存不下。

**核心矛盾**：要"记住历史"换 variance reduction，就得存东西；要存得下，就只能存 $O(\text{const})$ 量；可历史本质上是关于全状态空间的分布——怎么压缩成常数维度还能保持理论性质（asymptotic unbiasedness + 方差递减）？已有妥协方案要么 buffer 大（Stein self-repulsive 要存历史样本）、要么需要重要性 reweighting（adaptive biasing potentials），都失了 normalization-free 的简洁。

**本文目标**：构造一个 generic wrapper，使得 (i) 内存恒为 $O(d)$；(ii) 兼容任何 base MCMC（MH、Langevin、HMC、GWG）；(iii) 保持 normalization-free；(iv) 有 CLT 与 $\alpha$-scaling 的理论保证，把 SRRW/HDT 的 near-zero-variance 性质拓到一般状态空间。

**切入角度**：Stein 恒等式告诉我们 $\mathbb{E}_{X\sim\pi}[s(X)] = 0$（其中 $s = \nabla\log\pi$），意思是"探索良好的链，score 的时间平均应该接近 0"。如果链总在某区域打转，那个区域的 score 方向就会被累积偏置——score 的 running 平均 $\theta_n$ 就是"链对真分布偏离的探测器"。这立刻把一个 $|\mathcal{X}|$ 维的统计量降到 $d$ 维。

**核心 idea**：用 $\theta_n \in \mathbb{R}^d$（score 时间平均）作为历史摘要，再用指数 tilt $\pi_\theta(x) \propto \pi(x)\exp\{-\alpha \theta^\top s(x)\}$ 把"过去过度访问的方向"惩罚掉，得到 surrogate target；让 base kernel 在 $\pi_{\theta_n}$ 上跑一步、再更新 $\theta_{n+1}$，循环往复。

## 方法详解

### 整体框架
SRMC 是一层薄包装。给定目标 $\pi(x)\propto e^{-U(x)}$、score $s(x) = -\nabla U(x)$、排斥强度 $\alpha \geq 0$、步长序列 $\gamma_n = (n+1)^{-\rho}$（$\rho \in (1/2, 1]$，Robbins-Monro 条件），以及任意 base kernel $P_q$（MH/MALA/HMC/GWG 均可）。在迭代 $n$：(1) 用当前历史 $\theta_n$ 构造 surrogate $\pi_{\theta_n}(x) \propto \pi(x)\exp\{-\alpha\theta_n^\top s(x)\}$；(2) 用 $P_{\pi_{\theta_n}}$ 采样 $X_{n+1}$；(3) 用一阶递推 $\theta_{n+1} = \theta_n + \gamma_{n+1}(s(X_{n+1}) - \theta_n)$ 更新历史。整套机制只增加 $d$ 维内存（$\theta_n$）和一次 score 评估。

### 关键设计

**1. 常数内存的 score-running 历史 $\theta_n$：用 $d$ 维向量替掉 $|\mathcal{X}|$ 维经验测度**

SRRW/HDT 之所以被困在有限状态空间，根因就是要存 $|\mathcal{X}|$ 维经验测度 $\hat\delta_n$——在 $\{0,1\}^d$ 上指数大、连续域上是无限维。SRMC 的破局点来自 Stein 恒等式 $\mathbb{E}_\pi[s]=0$：探索良好的链，score 的时间平均应当趋于 0，所以 score 的 running 平均 $\theta_n\in\mathbb{R}^d$ 天然是"链对真分布偏离的探测器"。$\theta_n$ 是过去 score $\{s(X_i)\}_{i\leq n}$ 的加权移动平均，$\rho=1$ 退化为简单时间平均、$\rho<1$ 偏向近期样本对短暂困住更敏感；它满足 $\theta_n - \mathbb{E}_\pi[s(X)] = \int_\mathcal{X}[\frac{1}{n+1}\sum_i\delta_{X_i}(x) - \pi(x)]s(x)dx$，本质就是"链的经验分布与 $\pi$ 在 score 投影下的偏差"。

一旦把"分布差异"投到 $d$ 维 score 空间，存储与计算复杂度立刻从指数/无限坌缩成常数，而 Stein 恒等式又保证 $\theta^\star=0$ 是平衡点，等于天然校准了估计。

**2. 指数 score-tilt surrogate target $\pi_\theta$：把历史偏差折成"排斥已访问区域"的代理目标**

有了偏差探测器 $\theta$，还得把它变成能驱动采样器避开旧区域的力。作者用指数 tilt $\pi_\theta(x)\propto\pi(x)\exp\{-\alpha\theta^\top s(x)\}$：当链在某 metastable basin 反复打转时 $\theta$ 指向该 basin 的 score 集中方向，于是仍在 basin 内（$\theta^\top s(x)>0$）的 $x$ 被 $\exp\{-\alpha\theta^\top s(x)\}<1$ 下权、外面的被相对抬升。对 MH 只需在接受率上乘一个 $e^{-\alpha\theta^\top[s(y)-s(x)]}$ 因子，归一化常数 $Z_\theta$ 在比值里自动抵消；对 Langevin/HMC 则把 score 换成 surrogate score $s_\theta(x)=s(x)-\alpha\nabla_x s(x)\theta=-\nabla U(x)+\alpha\nabla^2 U(x)\theta$，其中 Hessian-向量乘积可用 autodiff 或有限差分 $(\nabla U(x+\epsilon\theta)-\nabla U(x))/\epsilon$ 廉价近似。

选指数 tilt 有三个理由：它是 score 的线性扰动指数族，形式最自然又保非负；它让整套机制 normalization-free（$Z_\theta$ 始终抵消或不需），保住 classical MCMC 的可用性；它 plug-and-play，不动 base kernel 内部逻辑，任何对 $\pi$ 有效的 kernel 直接替成 $\pi_{\theta_n}$ 即可。Figure 1 的"arrow flipping"可视化正展示它如何动态压低 metastable basin 周围的有效能垒。

**3. 耦合 SA 分析 + CLT 与 $O(1/\alpha)$ 方差递减：把有限态的近零方差推到一般状态空间**

要让这套包装有理论底气，得在 $\mathcal{X}=\mathbb{R}^d$ 上证收敛 + 联合 CLT，并定量给出 $\alpha$ 增大时的方差 scaling。作者把 $\vartheta_n=(\theta_n,\mu_n)$（$\mu_n$ 是 $\mathbb{E}_\pi[f(X)]$ 的 running 估计）写成随机近似形式 $\vartheta_{n+1}=\vartheta_n+\gamma_{n+1}H(\vartheta_n,X_{n+1})$，$H$ 是 controlled Markovian noise。在 Assumption 1（score $L$-Lipschitz + $U$ 超线性尾增长 + Hessian 渐近正规）和 Assumption 2（kernel 一致 drift + 对 $\theta$ Lipschitz）下，Theorem 3.3 给出 $\vartheta_n\to(0,\mu)$ a.s. 且 $\gamma_n^{-1/2}(\vartheta_n-\vartheta^\star)\xrightarrow{d}\mathcal{N}(0,\Sigma_\vartheta)$，$\Sigma_\vartheta$ 解一个 Lyapunov 方程 $(\frac{1_{\rho=1}}{2}I + A^\star)\Sigma_\vartheta + \Sigma_\vartheta(\cdot)^\top + \Sigma_\Delta = 0$。关键 Jacobian $A^\star = \begin{bmatrix}-I_d - \alpha\mathrm{Cov}_\pi(s,s) & 0 \\ -\alpha\mathrm{Cov}_\pi(f,s) & -I_m\end{bmatrix}$ 里 $\alpha$ 只出现在协方差块上，于是 Proposition 3.4 证出 $\theta$-块 $\Sigma_{\theta\theta}(\alpha)=O(1/\alpha)$ 且对 $\alpha$ 单调不增；Gaussian 目标 + 均值估计时这个 scaling 直接传给 $\Sigma_X(\alpha)=V\Sigma_{\theta\theta}(\alpha)V^\top$，得到 sample mean 的近零方差。

技术上的实质升级是：把 Borkar 等人的通用 SA 条件翻译成 MCMC 语境下可验证的 drift + kernel-Lipschitz 条件，并给出 MH/MALA 的具体充分条件，把"稳定性"从假设升级成可证结论——早期 SRRW/HDT 分析还得先假设 iterate 有界。

### 损失函数 / 训练策略
没有训练损失——SRMC 是采样算法。实务超参三件套：$\rho \in \{0.6, 0.8\}$（避免 $\rho=1$ 步长降太快）、$\epsilon \approx \alpha$ 量级（有限差分尺度，过小会数值不稳）、$\alpha$ 选到使 $\alpha|\theta_n^\top s(X_n)|$ 适中以免过度 tilt。复杂目标可用 adaptive-$\alpha$ heuristic：早期小、后期大。离散域用 discrete Stein operator $s_i(x) = \pi(x^{(i,K-x_i)})/\pi(x) - 1$ 保持 $\mathbb{E}_\pi[s] = 0$；高维 EBM（如 Static MNIST）允许用 relaxed gradient 作 score proxy，理论严格性退化但实践仍有效。

## 实验关键数据

### 主实验
10 维连续目标上 SR-MALA / SR-HMC 与 baseline 的 MSE 对比（sample mean estimation，100 次独立重复，节选 $\alpha$ 网格 $\{0,0.01,0.1,1,2,5\}$）：

| 目标 | sampler | 最佳 $\alpha$ | MSE 改善 vs baseline | 备注 |
|------|---------|--------------|---------------------|------|
| Correlated Gaussian (10D, ill-conditioned) | MALA → SR-MALA | $\alpha=2{-}5$ | 适度下降 | step 与 CPU 双视图均成立 |
| Correlated Gaussian | HMC → SR-HMC | $\alpha=2{-}5$ | 显著下降，最低 MSE | $\alpha=5$ 最优 |
| Bayesian Logistic Regression (10D, 100 obs) | MALA → SR-MALA | $\alpha=1{-}2$ | 最大约 5× MSE 减少 | $\alpha=5$ 过激进、拒绝率高 |
| Bayesian Logistic Regression | HMC → SR-HMC | $\alpha \approx 1{-}2$ | 最快 MSE 衰减 | $\alpha=5$ over-tilt 反而更差 |

离散 EBM（Static MNIST，二值 $\{0,1\}^{784}$，100 条并行链全部从单张数字 '7' 初始化跑 10000 步）：

| 指标 | Baseline GWG | SR-GWG ($\alpha=10^{-4}$) | 相对变化 |
|------|--------------|---------------------------|---------|
| 累积 KL (↓) | 4.16 | 0.68 | **−84%** |
| Batch Vendi Score (↑) | 2.6 | 6.4 | **+146%** |
| 逃出初始模态步数 | 始终未逃出 | ~2500 步 | — |
| 链多样性 (10000 步终态) | 几乎全是 '7' | 覆盖多数字类别 | 显著模式探索 |

CIFAR-10 高斯混合 mode-coverage（附录 D.2）：SR-ULA 1035 步完全覆盖全部模态，ULA 仅 2.8%；10 条并行链时 SR-ULA 覆盖 7/10 类、ULA 仅 5/10。

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| $\alpha = 0$ | 退化为 base sampler | 验证 SRMC 是真包装层 |
| $\alpha \uparrow$ 中等区间 | MSE / KL 单调下降 | 符合 Prop 3.4 的 $O(1/\alpha)$ 理论 |
| $\alpha$ 过大（如 5）+ 非线性目标 | 拒绝率飙升或 over-tilt | 实际最优 $\alpha$ 通常 moderate，非 maximal |
| $\rho = 1$ vs $\rho \in \{0.6, 0.8\}$ | 后者 transient 更稳 | $\rho=1$ 步长降太快、$\theta_n$ 适应慢 |
| $\epsilon$ 过小（finite diff） | Hessian-vec 近似变差 | $\epsilon \sim \alpha$ 量级最稳 |
| Adaptive $\alpha$ | 鲁棒兜底 | 当 fixed $\alpha$ 最优值未知时可用 |
| Discrete: relaxed grad proxy vs exact discrete score | 高维实践只能用 proxy | 理论需 exact $s$ 满足 $\mathbb{E}_\pi[s]=0$，但 proxy 在 MNIST 上仍 work |

### 关键发现
- 理论上 $\Sigma_{\theta\theta}(\alpha) = O(1/\alpha)$ 推到 Gaussian 目标可得 $\Sigma_X(\alpha)$ 同等 scaling——experiments 在 correlated Gaussian 上确实看到 MSE 随 $\alpha$ 单调下降直至饱和。
- 在非线性目标（logistic regression）上，$\alpha$ 存在"最佳工作区"，过大反而拒绝率爆炸；这是 plug-and-play wrapper 的核心调参洞察。
- 离散 EBM 上 mode-mixing 的提升远大于变分（KL 降 84% 是 mode coverage 而非细节估计的胜利），说明 SRMC 真正的杀手锏是"系统性强迫探索"。
- 当并行链数减少时 SR-ULA 的优势更明显，意味着 SRMC 可以在算力受限时替代加更多链。

## 亮点与洞察
- **"用 score 的时间平均代替 visit count"是整篇文章最 elegant 的思想**：从 $|\mathcal{X}|$ 维到 $d$ 维只需一次 Stein 恒等式的洞察——score 的期望为 0，所以它的时间平均天然是"偏差探测器"。这种"用低维统计量取代高维分布"的范式可以迁移到任何需要 history 的随机算法（adaptive optimization、bandit、RL exploration）。
- **plug-and-play wrapper 的工程价值**：MH、MALA、HMC、ULA、GWG 全部一键适配，且 normalization-free 保留。换句话说 SRMC 是 MCMC 工具箱里的通用 layer，不需要重做收敛分析。
- **Hessian-vec product 通过有限差分廉价化**：对没有解析 Hessian 的 EBM/Bayesian 模型，$\nabla^2 U(x)\theta \approx (\nabla U(x+\epsilon\theta) - \nabla U(x))/\epsilon$ 只多一次 score 评估，整个 SRMC overhead 约 2×。
- **理论贡献也很扎实**：把 Borkar 的 SA 一般定理"翻译"成 MCMC 可验证条件，并直接证明稳定性（不需事先假设 iterate bounded），这是相对 SRRW/HDT 分析的实质升级。
- **思想可外推到 RL / 优化**：score-based history + 指数 tilt 可以做 PPO 中的 trust region adaptation、SGD 的 anti-collapse 正则、甚至 diffusion 模型 sampling 时的多样性增强。

## 局限与展望
- 一般非 Gaussian 目标 + 任意 base kernel 下，$\Sigma_{\mu\mu}(\alpha)$ 没有 closed-form $\alpha$ 依赖；只有 independent surrogate sampling 和 Gaussian 这两类特殊情况能写显式公式，工程上需要靠经验调 $\alpha$。
- 非线性目标存在 "best $\alpha$ at moderate value" 的现象，过大反而 over-tilt 导致拒绝率爆炸；缺乏完全自动化的 $\alpha$ 调度（adaptive heuristic 仍是 heuristic）。
- 离散域严格理论要求 discrete Stein score $\mathbb{E}_\pi[s] = 0$，但实际 Static MNIST 必须用 relaxed gradient 做 proxy，理论与实践存在 gap；exact discrete score 在 $d=784$ 量级评估成本爆炸。
- 每步多一次 Hessian-vec 或 finite-difference score：对 HMC 等内层多步 leapfrog 的方法 overhead 不容忽视，CPU-time 视图下 variance 优势会被吞噬一部分。
- 当前框架是 single-chain；与 parallel tempering、Replica Exchange 等结构的组合方式还没系统研究。

## 相关工作与启发
- **vs SRRW (Doshi 2023) / HDT-MCMC (Hu 2025)**：同样追求 non-Markovian 自排斥，但 SRRW/HDT 受限有限状态空间（要存 $|\mathcal{X}|$ 维经验测度），SRMC 用 $d$ 维 score 平均把同样的 near-zero-variance 推广到一般状态空间，是直接对应的"连续 / 高维离散版"。
- **vs Stein self-repulsive dynamics (Ye 2020)**：后者用 sample buffer 做 pairwise 排斥，memory 与 buffer 同阶且无穷 buffer 才无偏；SRMC 常数内存、有限 $\alpha$ 也无偏。
- **vs Adaptive Biasing Potentials / Metadynamics**：物理化学界用密度累积 + reweighting 做 history-dependent flatten，需要离散化和重要性校正；SRMC 不需要重新归一化。
- **vs Wang-Landau / Contour SGLD (Deng 2020)**：那类方法 flatten 自身能量面，但绑定 Langevin 框架且需要 stratification；SRMC 通用 wrapper、不限 sampler。
- **vs Adam-style SGLD with adaptive drift (Kim 2022)**：用 gradient running average 修 Langevin drift，SRMC 在精神上类似（"用历史 score 修目标"），但 SRMC 通过 modify target 而非 modify dynamics，因此对任意 sampler 都成立且有 CLT。
- **启发**：score-based history compression 这个 trick 对 VI、normalizing flow training、diffusion guidance、RL exploration bonus 都有潜在迁移价值——任何需要"低维历史摘要 + 不破坏归一化"的场景都值得套用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](../../ACL2026/nlp_generation/adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ICML 2026\] Characterizing the Effect of Noise in Language Generation in the Limit](characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](../../ACL2026/nlp_generation/threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)

</div>

<!-- RELATED:END -->
