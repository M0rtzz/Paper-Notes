---
title: >-
  [论文解读] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation
description: >-
  [ICML 2026][可解释性][Shapley 值] 在评估预算极度受限（如需要重训模型）的代价昂贵游戏上，用带 Hamming 核的 GP 作为价值函数代理、按"对 Shapley 值的期望信息增益（EIG）"自适应挑下一个 coalition…
tags:
  - "ICML 2026"
  - "可解释性"
  - "Shapley 值"
  - "贝叶斯实验设计"
  - "EIG"
  - "高斯过程"
  - "汉明核"
---

# ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation

**会议**: ICML 2026  
**arXiv**: [2606.02247](https://arxiv.org/abs/2606.02247)  
**代码**: 有（论文附 Appendix D.1.5 公开仓库）  
**领域**: 可解释机器学习 / Shapley 值估计 / 贝叶斯实验设计  
**关键词**: Shapley 值, 贝叶斯实验设计, EIG, 高斯过程, 汉明核

## 一句话总结
在评估预算极度受限（如需要重训模型）的代价昂贵游戏上，用带 Hamming 核的 GP 作为价值函数代理、按"对 Shapley 值的期望信息增益（EIG）"自适应挑下一个 coalition，并把 EIG 计算从 $O(4^p t)$ 压到 $O(p^4 + t^3)$。

## 研究背景与动机

**领域现状**：Shapley 值（SV）是可解释 ML 中最常用的公理化归因度量，但精确计算需要枚举 $2^p$ 个 coalition 并对每个调用价值函数 $\nu(S)$。现有方法大体分两支：(1) 蒙特卡洛类（permutation sampling、MSR、SVARM）从一个**固定预设分布**采 coalition；(2) 代理回归类（Kernel SHAP、Leverage SHAP、Regression MSR）拟一个 surrogate 再读 SV，coalition 也是按固定分布采的。

**现有痛点**：当价值函数本身昂贵——例如 TabPFN 特征重要性（每次要重跑 in-context 推理）、Ghorbani-Zou 式 data valuation（每次重训一个 RF/GB）、HyperSHAP 超参重要性（每次跑一轮 HPO）、大视觉模型的局部解释（每次 API 调用花钱）——预算可能只有几百次，固定分布采样把宝贵的 query budget 浪费在"和之前 coalition 提供同质信息"的样本上。

**核心矛盾**：自然想做"自适应挑 coalition"，但 BED 框架下的 EIG 通常没有闭式，常用的 GP 上 EIG 也只是对 surrogate **本身** 的不确定性下手（如 uncertainty sampling，US），并不直接对准 SV 这个下游量；而且即便有了准则，对 $2^p$ 个 coalition 暴力遍历也是指数级。

**本文目标**：(i) 给"对 SV 的 EIG"找出一个 closed-form；(ii) 把 EIG 计算从 $O(4^p t)$ 降到关于 $p$ 多项式；(iii) 在低预算区间真正打过 SOTA 采样/代理方法。

**切入角度**：SV 是价值函数的**线性变换** $\phi=A\nu$（Shapley 公理之 linearity）。把"挑 coalition + 关心 SV"这件事重新装进**贝叶斯线性逆问题 + 目标导向 OED（GOODE）** 的框架里，EIG 就只依赖 GP 后验协方差而非具体观测值，从而有闭式 $-\tfrac12 \log\det(A\Sigma_{\theta\mid y}A^\top)+C$。

**核心 idea**：用 Hamming 核 GP 当 surrogate + 把 SV 当线性 end-goal 写出闭式 EIG + 用初等对称多项式（ESP）展开 Hamming 核乘性结构，使 EIG 多项式时间可算。

## 方法详解

### 整体框架
ShaplEIG 是一个 greedy Bayesian Adaptive Design（BAD）循环。**输入**：玩家集 $P=\{1,\dots,p\}$、待估的价值函数 $\nu:2^P\to\mathbb{R}$、初始 coalition 子集 $\mathcal{C}_0$（按 leverage score 采 $T_0=p+1$ 个）、候选池 $\mathcal{C}$、预算 $T$（论文里 $\le 512$）。**输出**：所有 $p$ 个 SV 的一致估计 $\hat\phi$。每一轮 $t$ 做四件事：(1) 在候选池 $\mathcal{C}$ 中**穷举**或在最多 1024 个候选上算出 $\mathrm{EIG}^{(t)}_\phi(z^{(i)})$，挑最大的；(2) 真去调用 $\nu$ 评估该 coalition；(3) 把新 $(z,\nu(z))$ 加入数据集 $\mathcal{D}_{t+1}$；(4) 重新极大似然/MAP 拟 GP 超参 $\xi$（这一步是让流程"真正自适应"的唯一渠道——只有它能让历史观测值 $y$ 影响下一轮选择）。终止后从 GP 后验均值通过 $\hat\phi = A\mu_{\nu\mid\mathcal{D}_{T+1}}$ 直接读出 SV。

### 关键设计

1. **Hamming 核 GP 作为价值函数代理**:

    - 功能：在 $\{0,1\}^p$ 的二元 coalition 空间上为 $\nu$ 提供一个完全概率化的、低数据机制下也能给出良校准不确定性的 surrogate。
    - 核心思路：把 coalition 表示为指示向量 $z\in\{0,1\}^p$，用加权 Hamming 距离核 $k_\xi(z,z')=\prod_{j=1}^p \xi_j^{\mathbb{1}[z_j\ne z'_j]}$（每个 player 一个可学权重 $\xi_j$）。在固定 $\xi$ 下，$\nu(Z)\mid\mathcal{D}_t,\xi$ 是一个 $2^p$ 维 MVN，**有闭式后验均值与协方差**。
    - 设计动机：传统 Kernel SHAP 用线性 surrogate + 固定权重，其后验不确定性仅依赖已选 coalition 集合而**与观测值无关**，导致设计天然非自适应；GP + 可学 $\xi$ 把观测值的信息通过 kernel 超参注入后续 EIG，使 design 真正 outcome-adaptive。同时 Hamming 核的乘性结构正好让 ESP 展开成立，是后面 $O(p^4)$ 计算的前提。

2. **把 SV 当 GOODE 的线性 end-goal，写出 EIG 闭式**:

    - 功能：把"对 $p$ 维 SV 向量 $\phi$ 的 EIG"从原本需要嵌套 MC 估的难题，化成一行 log-det。
    - 核心思路：Shapley 公理保证 $\phi=A\nu$ 其中 $A\in\mathbb{R}^{p\times 2^p}$ 元素是 $\frac{\mathbb{1}_S}{p}\binom{p-1}{|S|-1}^{-1} - \frac{1-\mathbb{1}_S}{p}\binom{p-1}{|S|}^{-1}$。在 GP 先验下这是一个 Bayesian linear inverse problem 中"参数 → end-goal"的线性投影，EIG 可写成 $\mathrm{EIG}_\phi(z^{(i)}) \propto C' + \log[e_i^\top(\Sigma_{\nu\mid\mathcal{D}_t}+\sigma_\epsilon^2 I)e_i] - \log[e_i^\top(\Sigma_{\nu\mid\mathcal{D}_t}+\sigma_\epsilon^2 I - Q)e_i]$，其中 $Q_{i,i}=(A\Sigma_{\nu\mid\mathcal{D}_t}e_i)^\top (A\Sigma_{\nu\mid\mathcal{D}_t}A^\top)^{-1}(A\Sigma_{\nu\mid\mathcal{D}_t}e_i)$。这是 Attia 2018 的 GOODE 结果在 SV 设定下的具体化。
    - 设计动机：直接对未变换的 $\nu$ 用 EIG（信息论里叫 ITL）在 GP 上会退化成纯 US，只挑"对 surrogate 本身最不确定"的 coalition，完全忽略"这个评估对 $\phi$ 的下游不确定性减少多少"。论文还显示 EPIG 在 $2^p$ 目标分布上算不动。直接把 $A$ 嵌进 log-det 才是对准 SV 这个真正关心的下游量。

3. **用初等对称多项式把 EIG 从 $O(4^p t)$ 压到 $O(p^4 + t^3)$**:

    - 功能：让闭式 EIG 在 $p\le 100$ 量级的游戏上可算，整批候选 vectorize 后是 $O(p^4+t^3+|W|t^2)$。
    - 核心思路：naive 算法需要先构造 $2^p\times 2^p$ 的 $\Sigma_{\nu\mid\mathcal{D}_t}$，再做 $A\cdot, \cdot e_i$ 投影，dominant 项 $O(4^p t)$。论文的 Theorem B.1/B.2 把线性项 $AK_\xi(Z,z^{(i)})\in\mathbb{R}^p$ 和二次项 $AK_\xi(Z,Z)A^\top\in\mathbb{R}^{p\times p}$ 都重写成"跨 coalition 的加权核值求和"，再注意到大量 kernel evaluation 共享权重，最后把同权重的求和与**一元/二元 ESP** 一一对应，整体分别落到 $O(p^2)$ 和 $O(p^4)$。这正是 Hamming 核乘性结构 $k(z,z')=\prod_j \xi_j^{\mathbb{1}[\cdot]}$ 给的便利。
    - 设计动机：没有这一刀，闭式 EIG 也只是"理论上好看"——$p=10$ 就已经 $4^{10}\approx 10^6$ 维矩阵；这一刀把方法推到了 $p=101$（LE on Crime）也能跑的实际尺度。SV 估计本身可顺带 $O(t^3)$，SV 后验协方差 $\Sigma_\phi$ 可 $O(p^4+t^2 p)$。

### 损失函数 / 训练策略
GP 超参 $\xi$ 在每一轮（大 $p$ 改为按 refit schedule）由后验 $p(\xi\mid\mathcal{D}_{t+1})$ 的 MAP 重训；价值函数评估假设是带高斯噪声 $\epsilon\sim\mathcal{N}(0,\sigma_\epsilon^2)$ 的，但论文也讨论了 noiseless GP 下的一致性：当所有 $2^p$ 个 coalition 都被评估时，由 GP 插值性 $\mu_{\nu(z)\mid\mathcal{D}_{2^p+1}}=\nu(z)$ 自动得到 $\mu_\phi=\phi(\nu)$，即估计**构造上一致**——这一性质优于 Regression MSR（树代理默认不一致，要额外残差校正）。

## 实验关键数据

实验设计成 4 类任务共 15 个 game，玩家数 $p\in[8,101]$，每个 game 跑 30 或 100 个 seed。所有 baseline 在每一轮使用与 ShaplEIG 等量的 $\nu$ 评估预算，公平对比。

### 主实验
| 任务类别 | 代表 game | $p$ | ShaplEIG vs SOTA (低预算 MSE) |
|----------|-----------|-----|-------------------------------|
| FI（TabPFN）| Diabetes Reg. | 10 | 全程严格优于 Kernel/Leverage SHAP、Perm. Sampling、Reg. MSR |
| DV（RF on Bike Sharing）| Bike Sharing | 10 | 大幅领先所有 baseline 多个数量级 |
| HPI（XGBoost on Chess）| Chess | 16 | 早期与 Reg. MSR 接近，后续低 MSE 区间领先 |
| LE（ViT 16-patch）| ImageNet | 16 | 全程优于所有竞争者 |
| LE（RF on Crime, 大 $p$）| Crime | 101 | 尽管 GP 超参按 schedule 重训，仍跑通且最终领先 |

文中描述：在大多数 game 上 ShaplEIG 跨所有预算严格 dominant；只有 Regression MSR 偶尔在窄区间打平；其它 baseline 在低预算区基本被甩开。

### 消融实验
| 配置 | 关键发现 |
|------|---------|
| Full ShaplEIG | 最佳整体性能 |
| GP + Random sampling | 多数 game 被 ShaplEIG 大幅领先 |
| GP + Leverage Score Sampling | 偶尔接近 ShaplEIG，但仍被持续超过 |
| GP + Uncertainty Sampling (US) | **比 GP+Random 还差**，说明 BED 经典 US 准则不适合 SV |
| ShaplEIG（大 $p\ge 60$，超参定期重训） | 早期 100 轮被弱 baseline 略胜，之后反超 |

### 关键发现
- "强性能并不主要来自 GP surrogate 本身"——GP+Random、GP+Leverage、GP+US 都被 ShaplEIG 打败，说明 EIG-based 选 coalition 才是核心贡献。
- US 在 SV 估计上反直觉地比 random 还差，因为 US 只挑 surrogate 自己最不确定的 coalition，完全忽略"对 SV 下游影响有多大"；这反过来印证了"对 SV 直接写 EIG"的必要性。
- 计算开销：$p\le 16$ 时，GP 超参重训每轮 ≤2 分钟，EIG 算 <1 秒；$p\le 100$ 时超参重训 ≤25 分钟/轮，EIG ≤30 秒/轮。**超参重训才是 bottleneck**，所以 GP-based 但非 EIG 的 ablation 并不更便宜——这点对反驳"为了 EIG 多花了算力"很关键。
- 因此 ShaplEIG 的甜区是"$\nu$ 真昂贵（动辄重训/调 API）+ $p$ 中等（$\le \sim 16$ 是绝佳，$\sim 100$ 仍可接受）"。

## 亮点与洞察
- **把 SV 当"end-goal"而非把 $\nu$ 当目标**：直接拒绝了"先把 surrogate 学准、SV 自然就准"的朴素观点。证明在 GP 上 EIG 对 $\nu$ 退化成 US，但对线性投影 $A\nu$ 是有效准则，这个 framing 转换是真正的 idea 核心。
- **Hamming 核 + ESP 是一对天作之合**：乘性结构 $\prod_j \xi_j^{\mathbb{1}[\cdot]}$ 让"按 player 子集求和的核加权和"恰好对应初等对称多项式；这把指数复杂度压成多项式，几乎是为这个问题量身定做的——其他常见类别核（如 categorical RBF）拿不到这个红利。
- **可一致性 + closed-form** 是相对于 Regression MSR 的工程级优势：MSR 的树代理默认不一致，需要在残差上额外跑 MSR 校正；ShaplEIG 在 noiseless GP 下"全 coalition 评估 → 精确 SV"自动成立。
- **可迁移设计**：对任何"关心 $\nu$ 的线性 functional"的代价昂贵 game（不仅 SV，还能是 Sobol indices、其它 fANOVA 分解），GOODE + Hamming-GP + ESP 这一套都能复用，只要换 $A$。

## 局限与展望
- 作者承认：GP 超参重训才是大头，$p>100$ 时单轮分钟级到小时级开销使方法只在"$\nu$ 真的更贵"时才划算；当 $\nu$ 廉价（如普通 SHAP 解释）时 ShaplEIG 的 overhead 不值。
- 评估全部依赖**预计算 game**（除 LE-RF 外都是 cached value table）以方便算 ground-truth；论文没在真正 in-the-loop 重训 LLM/扩散模型这种"价值函数动辄数小时"的场景上端到端跑过，工程极限未充分压测。
- 噪声 GP 下的偏差/一致性、贪心 BAD 的次优性都被推到 Appendix；纯探索 US 表现奇差也提示这一族 BED 准则对游戏的"对称性/异构性"敏感，作者的"asymmetric game ShaplEIG 红利更大"猜想缺乏量化指标。
- 可改进方向：(i) 摊销/lazy 超参重训（如 lazy GP、structured kernel learning）以打破 bottleneck；(ii) batch BED 一次挑多个 coalition；(iii) 把 Hamming 核换成可同样 ESP 化的更表达性核以处理交互更强的 game；(iv) 把框架推广到带交互的高阶 Shapley interaction indices。

## 相关工作与启发
- **vs Kernel SHAP / Leverage SHAP / Regression MSR**：它们用固定预设分布采 coalition、surrogate 不感知历史观测；ShaplEIG 用 GP 的 outcome-adaptive 设计直接对准 SV，且天生一致无需残差校正。
- **vs BayesSHAP（Slack 2021）**：BayesSHAP 用贝叶斯线性模型 + US 选 query；ShaplEIG 同样思路但 (a) 换成 GP 拿到 outcome-adaptivity，(b) 把准则从 US 升级到对 SV 的 EIG，实验显示这个组合在 SV 任务上比单纯 GP+US 强不止一档。
- **vs Mitchell 2022 / Nguyen 2025（GP+BQ for SV）**：他们的 GP/核定义在**排列**或**数据分布**上、且通常一次只减一个 player 的不确定性、kernel 固定→ 非自适应；ShaplEIG 的核在 coalition 空间、目标是**所有 player 的 SV 联合不确定性**、超参随数据重训→ 真正自适应。
- **vs Active learning / ITL / EPIG**：ITL 在本设定下塌缩成 US；EPIG 因要对 $2^p$ 目标分布积分而算不动；ShaplEIG 借 SV 的线性可达到 closed-form，回避了 BED 经典的 nested MC 痛点。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 GOODE 视角引进 Shapley 估计，理论与算法两端都给出原创结果（闭式 EIG + ESP 化的 $O(p^4)$ 计算），不是把已有 BED 套壳。
- 实验充分度: ⭐⭐⭐⭐ 跨 FI/DV/HPI/LE 四类共 15 个 game、$p\in[8,101]$、强 baseline 与 4 种 ablation 都覆盖；但全部用预计算 game，缺少端到端"真昂贵 $\nu$ 在线运行"的工程验证。
- 写作质量: ⭐⭐⭐⭐ 把 BED 术语、Shapley 公理与 GP 推导串成一条清晰主线；只是核心公式与 ESP 证明都被放进 appendix，正文读起来略抽象。
- 价值: ⭐⭐⭐⭐ 在"$\nu$ 真昂贵 + 预算极低"这个明确的子场景里给出可直接采用的 SOTA 估计器；对 fANOVA、Sobol、超参重要性等同构问题有清晰的复用接口。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Verified SHAP: 神经网络精确 Shapley 值的可证明界](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)
- [\[ICML 2026\] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)

</div>

<!-- RELATED:END -->
