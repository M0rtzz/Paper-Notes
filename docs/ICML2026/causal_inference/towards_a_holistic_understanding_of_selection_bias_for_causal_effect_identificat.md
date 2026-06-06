---
title: >-
  [论文解读] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification
description: >-
  [ICML 2026][因果推理][选择偏差] 本文给出一个统一的"分布类"框架，刻画了在选择偏差下平均处理效应 (ATE) 全人群可识别的充要条件 (Condition 1)，并证明在 c-overlap 倾向得分 + 多项式指数族 / Gaussian / Laplace / Pareto / Log-n…
tags:
  - "ICML 2026"
  - "因果推理"
  - "选择偏差"
  - "ATE 可识别性"
  - "截断统计"
  - "倾向得分"
  - "分布类"
---

# Towards a Holistic Understanding of Selection Bias for Causal Effect Identification

**会议**: ICML 2026  
**arXiv**: [2605.13430](https://arxiv.org/abs/2605.13430)  
**代码**: https://github.com/EvieQ01/causal_effect_id_selection_bias  
**领域**: 因果推断 / 选择偏差 / ATE 可识别性  
**关键词**: 选择偏差, ATE 可识别性, 截断统计, 倾向得分, 分布类

## 一句话总结
本文给出一个统一的"分布类"框架，刻画了在选择偏差下平均处理效应 (ATE) 全人群可识别的充要条件 (Condition 1)，并证明在 c-overlap 倾向得分 + 多项式指数族 / Gaussian / Laplace / Pareto / Log-normal 等常见分布下都满足该条件，配套提出 MLE 与 Score Matching 两种带选择函数 $\beta(x,y,t)$ 校正的估计器，在合成与 All of Us 半合成实验上显著优于 IPW 与多项式回归。

## 研究背景与动机

**领域现状**：选择偏差在观察性研究中无处不在 —— 大规模生物银行存在"健康志愿者偏差"，临床试验的志愿者与拒绝者系统性不同。现有刻画 ATE 在选择偏差下可识别性的工作大体分两条路线：(i) **图模型路线** (Bareinboim & Pearl, Correa et al. 2019) 给出 selection-backdoor 等图判据，但**必须知道选择发生在 DAG 的哪个节点**；(ii) **SEM 路线** (Zhang et al. 2016) 假设可加噪声模型 (ANM) 且噪声非高斯，**只能处理 outcome-dependent 选择**。

**现有痛点**：(a) 实际中很难精确知道 selection 节点位置；(b) 非高斯 + ANM 的参数假设强且违背高斯/常用重尾分布的实践场景；(c) 两条路线互不相容，没人能把它们装进同一个语言里比较。

**核心矛盾**：刻画 ATE 可识别性要求"选择机制 + 倾向得分 + 协变量-结局分布"三者联合受约束，但既有结果要么把约束打在图结构上、要么打在函数形式上，缺少一个**只通过分布类 (probability class)** 来刻画的统一语言。

**本文目标**：给出一个**不依赖图位置、也不要求 ANM** 的充要条件，同时覆盖既有所有可识别情形并扩展到新的可识别区域；配套设计实际可跑的 ATE 估计算法。

**切入角度**：作者注意到只要在三元组 $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$ 上施加合适的"两个不同分布若 ATE 不等则观测部分必不等"的可分性条件，就能换来可识别性；而**截断统计** (Daskalakis et al. 2021) 给出了从截断样本外推到原分布的工具，正好对接 deterministic selection 下的"完全屏蔽"区域。

**核心 idea**：用 **distribution-class 三元组上的可分性条件 (Condition 1)** 替代图判据 + ANM 假设，再用 **截断统计 + 选择函数 $\beta$ 的联合估计** 把可识别性落到实操算法上。

## 方法详解

### 整体框架
论文分两层：**理论层**给出 ATE 全人群可识别的充要条件 Theorem 3.1（核心是 Condition 1），并实例化到 deterministic / nondeterministic 两类选择机制下的常见分布族 (Proposition 3.3 / 3.5)；**算法层**用三阶段 pipeline 估计 ATE：先估倾向得分定位"有 overlap"的子区域 $\mathcal{B}$，再在 $\mathcal{B}$ 上用 MLE 或 Score Matching 联合估计条件结局密度 $\hat{P}(y\mid x,t)$ 与选择函数 $\hat{\beta}(x,y,t)$，最后通过 $1/\hat{\beta}$ 重加权外推到全人群再计算 $\hat{\tau}_P$。

### 关键设计

1. **基于 distribution-class 的可识别性条件 (Condition 1 + Theorem 3.1)**:

    - 功能：用一条"全局可分性"条件刻画 ATE 全人群可识别的充要条件，**完全不依赖 DAG 中 selection 节点的位置**。
    - 核心思路：对三元组 $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$ 中任意两组兼容分布 $(P, Q)$，若它们的 ATE 不等 ($\tau_{P_{xy(t)}} \neq \tau_{Q_{xy(t)}}$)，则必存在某点 $(x,y,t)$ 使观测密度 $\alpha_P(x,y,t) P_{t\mid x} P_{xy(t)} \neq \alpha_Q(\cdot) Q_{t\mid x} Q_{xy(t)}$，其中 $\alpha_P = P_{s\mid xyt} / P(s)$。Theorem 3.1 证明这一条件就是"ATE 从 $P(V\mid S=1)$ 可识别"的充要条件。
    - 设计动机：传统图判据把"识别性"绑死在 DAG 拓扑上，本条件直接打到分布类上，**既能覆盖图判据（Corollary 3.9-3.14 把 selection-backdoor、selection-backdoor-ext、outcome-dependent selection、S-id 全部装进来），又能突破图判据无法处理的非典型情形**。

2. **常见分布族下的可识别实例化 (Proposition 3.3 + 3.5)**:

    - 功能：把抽象的 Condition 1 落到能直接拿来用的分布族，告诉实验者"在我的数据下到底识别不识别"。
    - 核心思路：(i) **Deterministic selection**（某些 $(x,y,t)$ 处 $P_{s\mid xyt}=0$ 完全屏蔽）下，只要 $\mathbb{P}_{t\mid x}$ 满足 c-overlap ($c < p(t,x) < 1-c$)，且条件结局密度形如 $P_{y(t)\mid x} \propto e^{f(x,y)}$ 与边际 $P_x \propto e^{g(x)}$ ($f, g$ 多项式，记为 $\mathbb{P}_{xy(t)}^{C^\infty}$)，Condition 1 即满足 —— 用截断统计 (Daskalakis et al. 2021) 从截断样本外推出原分布。(ii) **Nondeterministic selection** ($P_{s\mid xyt} > d > 0$) 下，只要 $\mathbb{P}_{xy(t)}$ 属于 Gaussian / Laplace / Pareto / Log-normal 四大常见分布族就足够。
    - 设计动机：(ii) 直接打脸 Zhang et al. 2016 的"必须非高斯"假设 —— 在本框架里**高斯也可识别**；(i) 把多项式指数族这一类涵盖大多数实际分布的家族纳入可识别范围，且**与轻尾/重尾分布都兼容**。

3. **带选择函数 $\beta$ 的 MLE / Score Matching 估计器 (Algorithm 1)**:

    - 功能：在估计阶段把"选择机制未知"这件事编码成一个待学函数 $\beta_\phi(x,y,t)$，与结局密度 $P_\theta(y\mid x,t)$ 联合学习。
    - 核心思路：观测到的条件密度满足 $\tilde{p}(y\mid x) \propto p(y\mid x) \cdot \beta(x,y,t)$。**MLE 路线**：在 $\mathcal{B}$ 内最小化负对数似然 $L(\theta) = -\sum_i (\log \hat{P}(y_i\mid x_i, t_i) + \log \hat{\beta}(\cdot) + \log \hat{P}(x_i) + \log \hat{e}(x_i))$，并加正则 $\lambda \sum_i \log\|\hat{\beta}\|_2^2$ 让 $\log\hat{\beta} \to 0$ 解决不确定性。**Score Matching 路线**：对 $y$ 求梯度得 $\psi(x,y,t) = s_\theta(x,y) + \nabla_y \log\beta_\phi(x,y,t)$（因为难算的 partition function 仅依赖 $x,t$，对 $y$ 求导即消失），用 Hyvärinen 目标 $\frac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi)$ 加 $-\lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2$ 联合优化。最后用 $1/\hat{\beta}$ 重加权得 $\hat{P}_{pop}(x)$，按 $\hat{\tau}_P = \mathbb{E}_{x \sim \hat{P}_{pop}}[\mathbb{E}[Y\mid x, 1] - \mathbb{E}[Y\mid x, 0]]$ 输出 ATE。
    - 设计动机：传统 IPW 完全忽略 selection，只对 deterministic 部分有效的"polynomial 外推"也搞不定 nondeterministic 偏差；而 +$\beta$ 校正版**同时处理两类选择**，且 Score Matching 路线借助 $\nabla_y$ 抹掉 partition function，规避了高维归一化常数难算的痛点。

### 损失函数 / 训练策略
MLE 最终目标：$L(\theta) + \lambda \sum_i \log\|\hat{\beta}(x_i, y_i, t_i)\|_2^2$；
Score Matching 最终目标：$\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{D}_{obs}} [\frac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi) - \lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2]$。 $\beta$ 通常 $\to 1$（最大化"观测样本被选中"的似然），$\lambda_1, \lambda_2$ 是正则超参。

## 实验关键数据

### 主实验
合成数据：$N=5000$ 生成、selection 后约 $3000$，重复 5 次箱线图。设置覆盖 additive / multiplicative × Gaussian / Laplace 共 4 组噪声，且同时施加 deterministic + nondeterministic selection。

| 数据集 | 噪声 / 函数形式 | IPW | Polynomial | MLE | MLE+$\beta$ | SM | SM+$\beta$ |
|--------|----------------|-----|------------|-----|-------------|----|----|
| Synthetic | Additive Gaussian | 严重偏 | 中等偏 | 较小偏 | **最小偏** | 较小偏 | **最小偏 + 低方差** |
| Synthetic | Additive Laplace | 严重偏 | 中等偏 | 较小偏 | **最小偏** | 较小偏 | **最小偏 + 低方差** |
| Synthetic | Multiplicative Gaussian | 严重偏 | 明显偏 (多项式无法外推) | 较小偏 | **最小偏** | 较小偏 | **最小偏** |
| Synthetic | Multiplicative Laplace | 严重偏 | 明显偏 | 较小偏 | **最小偏** | 较小偏 | **最小偏** |

半合成 All of Us：T2D 为暴露、BMI 为协变量、结局 $Y = f(T, X) + E$ 合成；本文方法显著减小偏差但未消除（作者解释为低倾向得分 $\sim 0.05$ 导致 overlap 极弱）。

### 消融实验

| 配置 | 处理 deterministic | 处理 nondeterministic | 主要误差 |
|------|---------------------|------------------------|----------|
| IPW | ❌ | ❌ | 大 |
| Polynomial | ✅ (外推) | ❌ | 中 |
| MLE | ✅ | 部分 ✅ | 较小 |
| MLE+$\beta$ | ✅ | ✅ | 最小（方差稍大）|
| SM | ✅ | 部分 ✅ | 较小 |
| SM+$\beta$ | ✅ | ✅ | 最小（方差最低）|

### 关键发现
- **+$\beta$ 校正在偏差上始终最优**，但相应方差更大 —— 估计选择函数本身引入不确定性，是 bias-variance trade-off 的典型代价。
- **SM+$\beta$ 比 MLE+$\beta$ 方差更低**：score matching 不需要算 partition function，对密度尺度因子不敏感，因此数值上更稳；但偏差与 MLE+$\beta$ 相近。
- **multiplicative 噪声下 Polynomial 直接失效**：它只能拟合可加结构的均值回归，乘性噪声破坏了这一假设；而 MLE/SM 估计完整密度因此鲁棒。
- **All of Us 上残留偏差**主要源于 overlap 弱（倾向得分 $\sim 0.05$），提示真实世界数据下还需要正则化或额外先验来稳住外推。

## 亮点与洞察
- **把图判据 + ANM 假设统一到 distribution-class 语言里**：Corollary 3.9-3.14 一口气把 selection-backdoor、selection-backdoor-ext、outcome-dependent、S-id 全部翻译为 Condition 1 的特例，这种"统一一切既有结果再加新结果"的呈现方式本身就值得借鉴。
- **截断统计被搬进因果识别**：之前截断统计主要用于带审查的高维分布学习，本文把 deterministic selection 直接视作"$\beta = 0$ 区域上的截断"再调用现有外推工具，是非常自然的桥接，可推广到更多 censored / missing-data 因果问题。
- **Score matching 把 selection 函数 $\beta$ 直接吃进梯度**：对 $y$ 求导让 partition function 消失，使得估计高维条件结局密度变得 tractable，这一 trick 在反事实生成、潜变量因果模型里同样可复用。
- **"不需要定位 selection 节点"是真正的实操友好点**：实际研究人员往往只能列出可能影响 $S$ 的变量集合而不知道精确的图结构，本文框架直接绕开这一痛点。

## 局限与展望
- **理论层是充要条件，但充要性靠"分布类"刻画 —— 实际应用者很难验证自己的数据严格属于某个 $\mathbb{P}_{xy(t)}^{C^\infty}$ 多项式指数族**；Proposition 3.5 给的 Gaussian/Laplace/Pareto/Log-normal 四族更实用，但仍是参数假设。
- **All of Us 上残留偏差未消除**，作者归因于低倾向得分导致 weak overlap；这说明在真实世界 selection 强度大、覆盖差时，本框架的可识别性虽然在理论上成立，但**有限样本下方差与外推误差会主导**。
- **未处理 unobserved confounding**：全文假设 $Y(t) \perp T \mid X$，与现实里有隐藏混杂的 EHR / biobank 数据仍有距离；可与 IV、proxy variable、negative control 等方法结合。
- **半合成实验里 $Y$ 是自己生成的**，文中明确不做公共健康结论；要做 fully real-world validation 需要有 RCT 黄金标准对照。
- **改进思路**：(a) 引入 doubly robust / TMLE 思路降低方差；(b) 把 Condition 1 推广到 multi-treatment 与 continuous treatment；(c) 与 graph-discovery 结合 —— 用本框架检测"哪些 selection 模式在分布上不可识别"，反过来约束图搜索空间。

## 相关工作与启发
- **vs Correa et al. 2019 (selection-backdoor)**：他们要求知道 selection 节点 $S$ 在 DAG 中的位置并满足 backdoor 判据；本文不需要图结构、只需分布类满足 Condition 1，Corollary 3.9-3.10 证明 selection-backdoor 是 Condition 1 的特例。
- **vs Zhang et al. 2016 (ANM + 非高斯)**：他们必须假设可加噪声模型 + 非高斯噪声且只能 outcome-dependent；本文 Proposition 3.5 允许 Gaussian / Laplace / Pareto / Log-normal、且 selection 可依赖任意观测变量。
- **vs Bareinboim & Pearl 2012 / Bareinboim et al. 2014**：他们的 transportability / recoverability 框架同样依赖图判据并需识别 $S$ 的入边结构；本文以分布类语言给出"图无关"的等价刻画。
- **vs Cai et al. 2025 (overlap 违背下的 ATE 识别)**：他们处理的是"covariate 支撑不重叠"，本文处理的是"selection 扭曲了观测分布"；机制不同但都用 distribution-class condition，作者明确指出 Cai et al. 的条件可识别 s-ATE-sub 但识别不了 s-ATE-full。
- **vs Abouei et al. 2024 (S-id 判据)**：S-id 是图判据要求 $T \notin Anc(S)$；Corollary 3.14 把它纳入本文 Condition 3 (Appendix) 的特例。
- **启发**：distribution-class 这种"抽象一层、把判据打到分布族而非图结构"的策略可推广到 missing-at-random、measurement error 等其他数据缺陷场景；截断统计 + score matching + $\beta$ 联合估计的 pipeline 也可移植到 RLHF / 偏好学习中处理选择性反馈偏差。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把图判据 + ANM 假设统一到 distribution-class 语言，并把截断统计搬进因果识别，框架级贡献。
- 实验充分度: ⭐⭐⭐⭐ 4 组合成噪声 + All of Us 半合成覆盖了主要场景，但 fully real-world / 大规模 baseline 对比仍有空间。
- 写作质量: ⭐⭐⭐⭐ Condition 1 与各 Corollary 的对应关系讲得很清楚，但符号密度高，Section 3 对非因果背景读者门槛偏陡。
- 价值: ⭐⭐⭐⭐⭐ 给出了"不依赖图位置"的统一框架，理论上覆盖既有所有可识别情形并扩展到新区域，配套算法可直接用于 biobank / 临床数据。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](../../ICML2025/causal_inference/latent_variable_causal_discovery_under_selection_bias.md)
- [\[ICML 2025\] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](../../ICML2025/causal_inference/causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Formalizing and Falsifying Causal Pathways of Rare Events](formalizing_and_falsifying_causal_pathways_of_rare_events.md)

</div>

<!-- RELATED:END -->
