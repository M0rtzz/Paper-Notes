---
title: >-
  [论文解读] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective
description: >-
  [ICML 2025][LLM对齐][迁移学习] 发现 RLHF 中 KL 正则化带来的结构性质——策略对最优策略的 coverage 被其次优性控制（$\text{Cov}^{\pi^*|\pi} \leq 1 + \kappa \cdot (J(\pi^*) - J(\pi))/\beta$），据此提出两条迁移学习原则：(1) 选高 policy value 的 transfer policy，(2) self-transfer 从在线数据蒸馏策略。设计 TPO 算法实现早期 $O(W\sqrt{T})$、后期 $O(\sqrt{T})$ 的 regret，可模块化集成 DPO/IPO/XPO，在 T5 summarization 实验上验证有效。
tags:
  - ICML 2025
  - LLM对齐
  - 迁移学习
  - KL regularization
  - policy coverage
  - online RLHF
  - sample efficiency
  - DPO
  - IPO
  - XPO
  - win rate
---

# Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective

**会议**: ICML 2025  
**arXiv**: [2502.19255](https://arxiv.org/abs/2502.19255)  
**代码**: https://github.com/jiaweihhuang/RLHF_RewardTransfer  
**领域**: 对齐RLHF  
**关键词**: transfer learning, KL regularization, policy coverage, online RLHF, sample efficiency, DPO, IPO, XPO, win rate

## 一句话总结

发现 RLHF 中 KL 正则化带来的结构性质——策略对最优策略的 coverage 被其次优性控制（$\text{Cov}^{\pi^*|\pi} \leq 1 + \kappa \cdot (J(\pi^*) - J(\pi))/\beta$），据此提出两条迁移学习原则：(1) 选高 policy value 的 transfer policy，(2) self-transfer 从在线数据蒸馏策略。设计 TPO 算法实现早期 $O(W\sqrt{T})$、后期 $O(\sqrt{T})$ 的 regret，可模块化集成 DPO/IPO/XPO，在 T5 summarization 实验上验证有效。

## 研究背景与动机

### 在线 RLHF 的样本效率瓶颈

在线 RLHF 需要收集大量人类偏好标签来对齐 LLM，标注成本高昂。已有方法主要通过**探索策略**提升样本效率（如 XPO 的乐观探索），但忽视了一个关键机会：可以利用已有的**不完美但相关的 reward model** 来加速学习。

### 不完美 reward model 的来源

在实际场景中，多种类型的不完美 reward 可用：

**跨语言 reward 迁移**：一种语言的 reward model 可用于另一种

**LLM-as-judge**：GPT/LLaMA/Gemini 的评估与人类偏好部分一致

**启发式 reward**：ROUGE、BERTScore 等基于规则的评价指标，成本极低

### 核心问题

给定 $W$ 个不完美 reward model $\{r_w\}_{w=1}^W$（质量事先未知），如何利用它们减少人类标注量来学到接近最优的策略？关键挑战：如何选择迁移策略、如何避免被低质量 source 误导。

## 方法详解

### 整体框架

**问题设定**：Contextual bandit（prompt 空间 $\mathcal{S}$，response 空间 $\mathcal{A}$），优化 KL 正则目标：

$$\pi_r^* \leftarrow \arg\max_\pi J_\beta(\pi; r) = \mathbb{E}_{s \sim \rho, a \sim \pi}[r(s,a)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

闭式解：$\pi_r^*(a|s) \propto \pi_{\text{ref}}(a|s) \cdot e^{r(s,a)/\beta}$

**迁移设定**：有 $W$ 个 source reward $\{r_w\}$ 及对应最优策略 $\{\pi_{r_w}^*\}$，策略价值差 $\Delta(w) := J_\beta(\pi_{r^*}^*) - J_\beta(\pi_{r_w}^*)$，$\Delta_{\min} := \min_w \Delta(w)$。

### 关键设计

#### 核心发现：KL 正则化带来的 Coverage 结构

**Lemma 3.1（本文最核心结果）**：

$$\text{Cov}^{\pi_{r^*}^* | \pi} \leq 1 + \kappa(e^{2R/\beta}) \cdot \frac{J_\beta(\pi_{r^*}^*) - J_\beta(\pi)}{\beta}$$

其中 $\kappa(x) = \frac{(x-1)^2}{x - 1 - \log x} = O(x)$。

**核心洞察**：在正则化设定下（$\beta > 0$），一个策略的**次优性**直接控制它对最优策略的 **coverage**。这是正则化独有的——在纯 reward 最大化中，两个次优差距为 $2\varepsilon$ 的确定性策略之间的 coverage 可以是无穷大。

原因：正则化排除了接近确定性的策略，利用 $\pi_{\text{ref}}$ 的先验知识确保剩余策略具有良好结构。

#### Principle 1：选高 Policy Value 的 Transfer Policy

由 Lemma 3.1，high policy value ⟹ small sub-optimality ⟹ good coverage for $\pi^*$。因此**利用高价值策略不与探索冲突**——正则化调和了探索-利用的矛盾。纯 reward 最大化中做不到这一点（近最优不意味着好 coverage）。

#### Principle 2：Self-Transfer Learning

**Theorem 3.2**：从在线收集的数据用离线学习（RPO）蒸馏出的策略 $\pi_{\text{Dstl}}$ 的次优性满足：

$$J_\beta(\pi^*) - J_\beta(\pi_{\text{Dstl}}) \leq \tilde{O}\left(e^{2R}\left(1 + \kappa(e^{2R/\beta}) \cdot \frac{\sum_t (J_\beta(\pi^*) - J_\beta(\pi_t))}{\beta T}\right) \cdot \frac{1}{\sqrt{T}}\right)$$

当在线算法 no-regret 时，$\pi_{\text{Dstl}}$ 以 $O(T^{-1/2})$ 速率收敛到 $\pi^*$——**不依赖状态/动作空间大小或策略类复杂度**。这比标准在线 RLHF 的 $O(\sqrt{\mathcal{C}(\Pi)/T})$ 更快（$\mathcal{C}(\Pi)$ 可能很大）。

Self-transfer 的额外好处：$\pi_{\text{Dstl}}$ 持续改进趋近 $\pi^*$，而 source policy 停留在固定非零 gap，避免被次优 source 限制。

### 损失函数/训练策略

**TPO 算法（Alg. 1）**：
- 将 $T$ 步分为 $K = T/N$ 个 block，每个 block 大小 $N$
- 每个 block 前 $\alpha N$ 步：运行在线算法 AlgOL（如 XPO）
- 后 $(1-\alpha)N$ 步：运行 Transfer Policy Selection (TPS, Alg. 2)
- 最终返回用全部数据蒸馏的策略

**Transfer Policy Selection (Alg. 2)**：
1. 用 MLE reward 估计器 $\hat{r}_{\text{MLE}}$ 对每个 source policy 做 **UCB 式乐观价值估计**
2. 用 RPO 目标的内在结构对 self-transfer policy 做**悲观价值估计**（下界）
3. 选择估计价值最高的策略

**Empirical TPO (Alg. 3)**——实际可用版本：
- 用 **win rate** 替代 policy value 估计（计算量大大降低）
- Win rate 下界可控制 coverage（Lemma 5.1）
- 用 UCB 策略在多个 source 间做 bandit 选择
- 可模块化集成任意 policy optimization 方法（DPO/IPO/XPO）作为 AlgPO
- 比较 $\hat{\text{WR}}_{\pi_{r_w}^*}$ vs $\hat{\text{WR}}_{\pi_{\text{OL}}} = 0.55$，仅在 source 明显优于当前在线策略时启用迁移

## 实验关键数据

### 主实验

**XSum Summarization + T5-small (80M)**：

Source reward models：(a) ROUGE-Lsum, (b) BERTScore, (c) T5-base (250M), (d) T5-large (770M)

True reward：sfairXC/FsfairX-LLaMA3-RM-v0.1（蒸馏自 Llama3-8B）

**DPO 作为 AlgPO 时，TPO 对比三个基线的 Win Rate (%)：**

| Iteration | vs 无迁移 (Iter-DPO) | vs 纯用 ROUGE (最差 source) | vs 纯用 T5-Large (最佳 source) |
|-----------|---------------------|---------------------------|-------------------------------|
| Iter 1 | >50% (显著优势) | >50% (显著优势) | ≈50% (持平) |
| Iter 2 | >50% | >50% | ≈50% |
| Iter 3 | >50% | >50% | >50% (超越) |

关键观察：Iter 3 时 TPO 甚至超越纯用最佳 source (T5-Large)——因为 TPO 自动切换回在线学习，避免被 source 的次优性限制。

### 消融实验

**Source Task Selection 过程分析（Figure 2）**：

- Iter 1：UCB 探索阶段，均匀分配各 source
- Iter 2：识别出 T5-Large 是最佳 source，分配最多预算
- Iter 3：$\pi_{\text{OL}}$ 超越所有 source，自动切换为无迁移模式

这精确验证了理论预测的两阶段行为。

**不同 AlgPO 的验证**：IPO 和 XPO 作为 AlgPO 时也观察到类似的改进模式，证明 TPO 的模块化设计确实通用。

### 关键发现

1. **KL 正则化是迁移学习的"祝福"**：它创造了 coverage 与 sub-optimality 的结构关系，使得简单的 policy value based selection 就够用
2. **Self-transfer 的自动超越机制**：蒸馏策略持续改进，最终必然超过固定的不完美 source
3. **Win rate 是 coverage 的有效代理指标**：虽然是下界，但足以指导迁移决策
4. **阈值 0.55 的实用价值**：设定略高于 0.5 的基线 win rate，避免低质量迁移

## 亮点与洞察

1. **深刻的理论洞察**：Lemma 3.1 揭示了 KL 正则化的结构性"祝福"——不仅是防过拟合的工具，更创造了有利于迁移学习的几何结构
2. **理论到实践的优雅过渡**：从精确但计算昂贵的 policy value 估计，转向高效但理论有依据的 win rate 估计
3. **Self-transfer 概念新颖**：用在线数据的离线蒸馏策略作为迁移候选——这是一种"未来的自己"帮助"现在的自己"的范式
4. **标准在线 RLHF 的附带改进**：即使 $W=0$（无 source），纯靠 self-transfer 也能实现 $O(\sqrt{T})$ regret——消除对 $\mathcal{C}(\Pi)$ 的依赖，严格改进现有结果（Corollary 4.5）
5. **模块化设计**：TPO 的迁移模块可与任意 policy optimization 方法组合，极大增加实用性

## 局限与展望

1. **实验规模有限**：仅在 T5-small (80M) 上验证，未在更大 LLM 上测试
2. **Reward model 数量少**：仅 4 个 source rewards，未验证 $W$ 很大时的选择效率
3. **Bradley-Terry 假设**：理论依赖 BT 偏好模型，真实人类偏好可能不服从
4. **Contextual bandit 设定**：未扩展到多轮对话（MDP），虽然作者指出可能扩展
5. **计算开销**：Best-of-N (N=32) 的 BoN 采样在大模型上成本高
6. **理论 TPO 到实践 TPO 的 gap**：理论算法需解 minimax 优化，实践直接用 DPO/IPO loss，gap 未被定量分析
7. **State-level transfer 未探索**：当前是 policy-level 迁移，更细粒度的 prompt-wise 迁移可能更好

## 相关工作与启发

- **在线 RLHF 探索线**：XPO (Xie et al. 2024)、Self-Exploring LM (Zhang et al. 2024) 关注探索策略，本文正交——关注利用已有 reward models
- **RLAIF vs 本文**：RLAIF 用 AI 反馈替代人类反馈作为终极目标，本文用 AI reward 加速人类 reward 的学习
- **Iterative DPO/IPO**（Xiong et al. 2024）：本文的在线学习基础
- **政策 coverage**（Xie et al. 2022）：$L^\infty$ coverability 最初用于在线 RL 复杂度度量，本文将其与 KL 正则化联结产生新发现
- **RPO**（Liu et al. 2024）：离线 RLHF 的 minimax 算法，用于 self-transfer 蒸馏
- **启发**：KL 正则化的结构性质可能在 Nash Learning from Human Feedback 等新范式中也有类似的有用性质可挖掘

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (coverage-suboptimality 联系、self-transfer 概念、理论到实践的转化都很新颖)
- 实验充分度: ⭐⭐⭐ (仅 T5-small 一个规模，且 source 数量有限)
- 写作质量: ⭐⭐⭐⭐⭐ (理论展开层次清晰，从性质到原则到算法到实践版本环环相扣)
- 价值: ⭐⭐⭐⭐⭐ (理论贡献扎实，实践算法通用，附带改进标准在线 RLHF 结果)

<!-- RELATED:START -->

## 相关论文

- [Reward Generalization in RLHF: A Topological Perspective](../../ACL2025/llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)
- [Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](../../ACL2025/llm_alignment/reward_fairness_rlhf.md)
- [Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](../../ICLR2026/llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [Mutual-Taught for Co-adapting Policy and Reward Models](../../ACL2025/llm_alignment/mutual_taught_policy_reward.md)

<!-- RELATED:END -->
