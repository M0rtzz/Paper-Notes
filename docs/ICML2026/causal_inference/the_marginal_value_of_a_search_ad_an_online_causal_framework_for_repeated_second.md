---
title: >-
  [论文解读] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions
description: >-
  [ICML 2026][因果推理][第二价格拍卖] 本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法…
tags:
  - "ICML 2026"
  - "因果推理"
  - "第二价格拍卖"
  - "treatment effect"
  - "contextual bandit"
  - "IPW"
  - "UCB"
---

# The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions

**会议**: ICML 2026  
**arXiv**: [2605.01756](https://arxiv.org/abs/2605.01756)  
**代码**: 无  
**领域**: 因果推断 / 在线学习 / 拍卖与广告  
**关键词**: 第二价格拍卖、treatment effect、contextual bandit、IPW、UCB

## 一句话总结
本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法，得到 $\widetilde\Theta(\sqrt{dT})$ 的极小极大最优 regret，比同设定下的一价拍卖严格更易学。

## 研究背景与动机

**领域现状**：搜索广告（Amazon / Google / Bing 的赞助位）几乎都用二价（Vickrey）拍卖，竞标策略上"如实出价"是 SPA 的理论最优；但广告主真正关心的是这次曝光值多少钱（CTR / CVR），这需要在线估计。最近一条线把"广告价值"建模为 treatment effect：赢拍点击带来的收益 $v_{t,1}$ 减去输拍后用户仍可能从有机搜索点进的收益 $v_{t,0}$。Wen 等之前在 FPA 上做过 binary 反馈下 $\widetilde\Theta_d(T^{2/3})$ 的最优 regret。

**现有痛点**：现有 auto-bidding 把价值等同于"赢拍后的收益"，会系统性高估应竞价 —— 对于已经排在有机结果前列的品牌，赢拍带来的边际收益近乎为零，但传统算法仍会把它当成高价值机会。理论侧 FPA 与 SPA 的最优 regret 差异也没人系统刻画过。

**核心矛盾**：要做因果估计就需要"赢"和"输"两种 outcome 都有观测，但 regret 最小化的 bidder 倾向于赢高价值、输低价值，反而破坏了 propensity overlap 条件；同时 SPA 下"赢家才看到 HOB"的非对称反馈给学习带来便利，但也让置信宽度的设计变得不平凡。

**本文目标**：(i) 把 treatment-effect 视角扩展到 SPA；(ii) 利用 SPA 支付规则带来的额外 HOB 信息，证明 SPA 下最优 regret 是 $\widetilde\Theta(\sqrt{dT})$ 而非 FPA 的 $\widetilde\Theta_d(T^{2/3})$；(iii) 放宽 propensity score 估计假设，允许任意误差形式 + 含原子的 HOB 分布。

**切入角度**：作者抓住 SPA 的核心信息差 —— 出价越高，关于 HOB CDF 的信息越多（赢了就能精确看到 HOB；输了也能推断 $\mathbb{1}[b\geq m_t]$）。利用这种"one-sided + 推断"信息结构，把 HOB CDF 分解成区间概率 $p^i$ 分块估计，可以得到比直接估计 $G(b)$ 更紧的置信宽度。

**核心 idea**：把"二价支付的信息红利"翻译成 propensity score 的 bid-dependent 置信宽度，再设计 "better of two UCBs" 决策规则中和掉 IPW 估计的潜在大方差，最终把 binary 反馈下的 SPA regret 从 $T^{2/3}$ 推到 $\sqrt{T}$。

## 方法详解

### 整体框架
算法在线交互 $T$ 轮，每轮收到 context $x_t\in\mathbb{R}^d$，按线性模型 $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$ 决策出价 $b_t$。整体由三块构成：(1) **HOB 估计模块**：利用二价支付的 one-sided 反馈分块估计 CDF $\widehat G_t$；(2) **价值估计模块**：用基于估计 propensity 的修正 IPW 加权最小二乘解出 $\widehat\theta_t$；(3) **决策模块**：构造两个等价的奖励重写形式 $\widehat r_{t,0}, \widehat r_{t,1}$ 与对应置信宽度 $w_{t,0}, w_{t,1}$，用 "better of two UCBs" 选择更紧的那个去做 UCB 出价。最外层套一个 $L=O(\log T)$ 层的 master routine，按 confidence 宽度把时间分到不同层，并周期性做均匀探索以满足理论条件。

### 关键设计

1. **分块 HOB 估计 + bid-dependent 置信宽度**:

    - 功能：把 CDF 估计从"对每个 $b$ 单独估"改为"按区间概率 $p^i=\mathbb{P}(b^{i-1}<m_t\leq b^i)$ 分块估计后累加"，让较低 bid 享受较多观测、较高 bid 的方差天然更小。
    - 核心思路：bid 离散到 $\mathcal{B}=\{b^j=(j-1)/\sqrt{T}\}$，定义 $\widehat p_t^i=\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]\mathbb{1}[b^i<m_\tau\leq b^{i+1}]/\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]$，CDF 估计 $\widehat G_t(b^j)=\sum_{i\leq j}\widehat p_t^i$。利用二价支付：高 bid 赢拍后能看到 mt，低 bid 时只要更高 bid 看到过赢的判定，就能反推 $\mathbb{1}[b\geq m_t]$，于是每个 $p^i$ 都有 $n_t^i\propto\sum\mathbb{1}[b_\tau\geq b^i]$ 个有效观测，越低 bid 观测越多。置信宽度推导（Lemma 1）给出 $u_t(b^j)\propto\sqrt{\sum_{k\leq j}\log T/n_t^k\cdot(\widehat p_0^k+\log T/\sqrt T)}$，自然实现"$p^i$ 小则估计更准"的 Bernstein 集中。
    - 设计动机：FPA 中 binary 反馈下没有支付信息，HOB 估计是 $T^{2/3}$ 难度的瓶颈；SPA 中支付带来的信息让 HOB 估计变快，但只有按区间分块才能把这种红利完全释放出来。

2. **可宽误差容忍的 IPW + 加权最小二乘**:

    - 功能：构造一个对 $\widehat G_t$ 任意误差形式都鲁棒的 IPW 估计器 $\widetilde e_t(b)$，再用 variance-weighted 最小二乘解 $\widehat\theta_t$，避免依赖 Bernstein-type HOB 误差假设。
    - 核心思路：定义 $\widetilde e_t(b)=\mathbb{1}[b\geq m_t]v_{t,1}/\widehat G_t(b)-\mathbb{1}[b<m_t]v_{t,0}/(1-\widehat G_t(b))$，bias 与 variance 的可计算 proxy 是 $u_t(b)\sigma_t(b)$ 与 $\sigma_t(b)^2$，其中 $\sigma_t(b)=1/(\widehat G_t(b)(1-\widehat G_t(b)))$。然后解 $\widehat\theta_t=\arg\min_\theta\sum_{\tau\in\Phi_t}\sigma_\tau^{-2}(\widetilde e_\tau-\theta^\top x_\tau)^2+\|\theta\|_2^2$，闭式解形如 $A_t^{-1}z_t$。Lemma 3 给出误差界 $|\widehat\theta_t^\top x_t-\theta_*^\top x_t|\leq\gamma\|x_t\|_{A_t^{-1}}$，注意这个界可以任意大。
    - 设计动机：以往 FPA 算法依赖"propensity 误差在小概率区域更紧"的 Bernstein 假设，本文的分块 HOB 估计不满足这种形式；新 IPW 设计接受任意误差形式 $u_t(b)$，让 HOB 模块和价值模块解耦，方法可推广性大幅提升。

3. **"better of two UCBs" + 均匀探索兜底**:

    - 功能：把价值估计的潜在大误差转换成 regret 上的有界损失：在两种等价的 reward 重写形式中自动选择置信宽度更紧的那个，并在两种宽度都过大时触发均匀探索。
    - 核心思路：奖励 $\bar r_t(b)$ 可改写为 $\bar r_{t,0}(b)=G(b)(\theta_*^\top x_t-b)+\int_0^b G(m)\mathrm{d}m$ 或 $\bar r_{t,1}(b)=-(1-G(b))\theta_*^\top x_t-G(b)b+\int_0^b G(m)\mathrm{d}m$，两种形式对 $\theta_*^\top x_t$ 的依赖系数互补：选 $r_{t,0}$ 在 $\widehat G_t(b)$ 小时窗宽 $\propto\widehat G_t(b)$ 小，反之选 $r_{t,1}$。Algorithm 2 通过比较 $\widehat G_t(b_L), \widehat G_t(b_R)$ 与阈值 $1-\lambda/8$ 在两个 UCB 中选出 $q$，把瞬时 regret 压到 $\min\{w_{t,0}(b_t), w_{t,1}(b_t)\}\propto\sigma_t(b_t)^{-1}$。当估计太差导致两个宽度都大时（理论概率小，由 Lemma 6 上界 $|\Phi_{\text{exp}}|=O(d\log^5 T)$），算法均匀随机出 $b^1$ 或 $b^J$ 做 forced exploration。
    - 设计动机：因果估计的难点是赢-输观测不平衡，传统做法靠 forced randomization 但代价是 $T^{2/3}$；本算法通过"对偶 UCB"利用价值估计误差与 $\sigma_t$ 的负相关（误差大时 $\sigma_t$ 也大，但 $\min$ 之后宽度反而小），把绝大多数轮的 regret 控制住，只在小常数置信门限处做少量探索。

### 损失函数 / 训练策略
算法不是训练 ML 模型，而是在线决策。整体 master routine：(i) 用前 $(L+1)T_0$ 轮以 $b_t=1$ 出价获得初始 HOB 观测样本；(ii) 之后每轮按层 $\ell=1,\ldots,L$ 检查置信宽度，把当前轮分配到首个 "wt > 2^{-ℓ}" 的层，对应 bid 用 UCB 计算或 forced exploration；(iii) 用 Lemma 5 的 layered scheme 保持各层观测的条件独立性。

## 实验关键数据

### 主实验

| 设置 | 反馈 | 上界 | 下界 | 结论 |
|------|------|------|------|------|
| SPA + binary feedback (本文 Thm 1+2) | binary | $O(\sqrt{dT}\log^3 T)$ | $\Omega(\sqrt{dT})$ | 严格优于 FPA 的 $T^{2/3}$ |
| SPA + full-info feedback | full | $\widetilde O(\sqrt{dT})$ | $\Omega(\sqrt{dT})$ | 与 binary 同阶 |
| FPA + binary feedback（Wen et al. 2024） | binary | $\widetilde O_d(T^{2/3})$ | $\Omega_d(T^{2/3})$ | 对比 baseline |
| 实证 vs LinUCB | – | LinUCB 线性 regret | NFM-style 算法 $\sqrt{T}$ 收敛 | LinUCB 因忽视 $v_{t,0}$ 持续高估价值 |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| Algorithm 4 (实用变体) vs 主算法 | 多一个 $\sqrt d$ factor | 去掉分层结构换简洁性，符合 linear bandit 常见 trade-off |
| 含原子 HOB 分布 (Definition 1) | regret 保持 $\sqrt{dT}$ | $(\omega,\lambda)$-locally-bounded 假设泛化到点质量 |
| 仅 forced exploration（不用 better-of-two UCBs） | regret 退化到 $T^{2/3}$ | 验证 better-of-two UCBs 是 $\sqrt T$ 关键 |
| 实证 LinUCB | 线性 regret | 把价值等同 $v_{t,1}$ 时持续过价 |

### 关键发现
- SPA 的支付规则贡献了"赢家观测 HOB"的关键信息，把 binary 反馈下的因果学习从 $T^{2/3}$ 直接拉到 $\sqrt T$，这是 SPA 与 FPA 在 marginal value bidding 上的本质差异。
- 价值估计可以"内禀地不可靠"：$\widehat\theta_t$ 误差可以任意大，但只要决策模块在两种 UCB 中选更紧的那个，整体 regret 仍可控 —— 这种"用决策结构吸收估计误差"的思路在因果在线学习里十分罕见。
- 仅靠 forced randomization 显然次优；只有结合分块 HOB 估计 + better-of-two UCBs + 分层独立性维护三件套，才能同时刷下界。

## 亮点与洞察
- 把广告价值建模为 treatment effect 而非"赢拍收益"在产业实践上有立竿见影的意义：对于有机搜索已经表现好的品牌，传统算法会持续过价，本文方法天然回避这种浪费。
- "better of two UCBs" 是个非常巧妙的技术 trick：同一个 reward 函数写成两种形式后置信宽度对 $\widehat G_t$ 互补，选 min 之后即使价值估计很烂，瞬时 regret 仍能 collapse 到 $\sigma_t^{-1}$。
- 把 propensity 估计误差形式放宽到任意 $u_t$，让 SPA 这种"非 Bernstein"的 HOB 估计也能跟因果学习模块拼起来，这一推广本身有方法论价值（适合任何 sponsored auction 变体）。

## 局限与展望
- 假设 HOB 分布是 i.i.d. stationary，实际广告平台中竞争者会随热点漂移，作者在 Appendix B 提了 contextual HOB 推广但主文未深入。
- 价值模型限定为线性 $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$，对深度神经网络打分场景不直接适用，扩展到 kernel 或神经表示是开放方向。
- Algorithm 3 的多层结构在工程上偏复杂，Algorithm 4 把它压扁后 regret 多一个 $\sqrt d$ 因子，实际部署还要在性能/复杂度间取舍。
- 实验只用合成数据，没有真实搜索广告平台数据验证；初始 $O(\sqrt T\log T)$ 轮高出价探索在头部预算受限场景可能不可接受。

## 相关工作与启发
- **vs Wen et al. (2024, FPA + treatment effect)**：本文是其 SPA 自然扩展，关键贡献是利用支付规则把 regret 从 $T^{2/3}$ 推到 $\sqrt T$，并放宽 propensity 估计假设。
- **vs Han et al. 2020 / interval splitting (FPA)**：分块 HOB 估计的灵感来自 FPA 的 interval splitting，本文把它与二价支付带来的非对称信息结合得到 bid-dependent 置信宽度。
- **vs incrementality / lift bidding 经验文献**：那条线主要在工业内部跑 A/B test，理论保证少；本文给出第一篇 minimax 最优、不依赖"overlap"条件的算法。
- **vs linear contextual bandits**（LinUCB）：本文方法在结构上类似 LinUCB，但额外处理 IPW 引入的 heteroskedasticity 和 propensity 不确定性，并通过 better-of-two UCBs 解决因果估计的爆炸方差。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 SPA 支付规则与 treatment-effect 因果学习耦合，得到 $\sqrt{dT}$ 最优 regret，对 FPA-SPA 复杂度差异做出干净的刻画。
- 实验充分度: ⭐⭐⭐ 理论分析极其完整，但实证只在合成数据上对比 LinUCB，缺乏真实拍卖数据验证。
- 写作质量: ⭐⭐⭐⭐ 数学结构清晰，better-of-two UCBs 的推导直观；符号密度高、entry barrier 不低。
- 价值: ⭐⭐⭐⭐ 对搜索广告、推荐系统中"边际价值"竞价的研究方向有方法论意义，工程上能引导厂商改造现有 auto-bidder。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](../../ACL2025/causal_inference/iris_an_iterative_and_integrated_framework.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)

</div>

<!-- RELATED:END -->
