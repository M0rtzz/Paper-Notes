---
title: >-
  [论文解读] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation
description: >-
  [ICML2025][off-policy evaluation] 本文从理论上揭示了"在 OPE 中使用估计的历史依赖行为策略比使用真实行为策略反而更好"这一悖论的根本原因——估计行为策略隐式地将 IS 估计器投影到更约束的空间，降低渐近方差但增加有限样本偏差。
tags:
  - ICML2025
  - off-policy evaluation
  - importance sampling
  - behavior policy estimation
  - history-dependent
  - bias-variance trade-off
---

# Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation

**会议**: ICML2025  
**arXiv**: [2505.22492](https://arxiv.org/abs/2505.22492)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: off-policy evaluation, importance sampling, behavior policy estimation, history-dependent, bias-variance trade-off

## 一句话总结
本文从理论上揭示了"在 OPE 中使用估计的历史依赖行为策略比使用真实行为策略反而更好"这一悖论的根本原因——估计行为策略隐式地将 IS 估计器投影到更约束的空间，降低渐近方差但增加有限样本偏差。

## 研究背景与动机
- **Off-Policy Evaluation (OPE)** 是强化学习的核心问题：利用行为策略 $\pi_b$ 收集的历史数据评估目标策略 $\pi_e$ 的期望回报
- **Importance Sampling (IS)** 是最基础的 OPE 方法，通过重要性比 $\pi_e/\pi_b$ 对回报重加权，理论上无偏但方差可能很大
- **悖论现象**：已有实验表明（Hanna et al., 2021），即使真实行为策略是一阶 Markov 的，用估计的**历史依赖**行为策略构造 IS 比，反而能降低 MSE。更令人困惑的是，使用更长的历史窗口效果更好
- **未解决问题**：为什么"估计"比"已知真值"更好？为什么引入对当前状态无关的历史信息反而有益？本文首次系统地从理论上回答这些问题

## 方法详解

### 核心直觉：从 Bandit 到 MDP
以上下文无关的 bandit 为例，定义三种 IS 估计器：
1. **Oracle IS** $\hat{v}_{IS}^\dagger$：使用真实行为策略 $\pi_b$
2. **Context-Agnostic IS** $\hat{v}_{IS}^{CA}$：用样本频率估计 $\hat{\pi}_b(a) = n(a)/n$
3. **Context-Dependent IS** $\hat{v}_{IS}^{CD}$：用条件频率估计 $\hat{\pi}_b(a|s) = n(s,a)/n(s)$

**Lemma 1** 建立了渐近 MSE 的严格序：

$$\text{MSE}_A(\hat{v}_{IS}^{CD}) \le \text{MSE}_A(\hat{v}_{IS}^{CA}) \le \text{MSE}_A(\hat{v}_{IS}^\dagger)$$

**关键洞察**：使用估计行为策略的 IS 等价于一个 Doubly Robust 估计器。以 $\hat{v}_{IS}^{CD}$ 为例：

$$\hat{v}_{IS}^{CD} = \mathbb{E}_n\left\{\sum_a \pi_e(a)\hat{r}(S,a) + \frac{\pi_e(A)}{\hat{\pi}_b(A|S)}[R - \hat{r}(S,A)]\right\}$$

第一项是直接法估计，第二项是增强项（augmentation），两者结合实现了：(1) 去偏；(2) 通过对比观测奖励与预测奖励来降低方差。

### MDP 中的历史依赖行为策略估计
定义 $k$-步历史 $H_{t-k:t} = (S_{t-k}, A_{t-k}, \ldots, S_{t-1}, A_{t-1}, S_t)$，通过最大似然估计：

$$\hat{\pi}_b^{(k)} = \arg\max_{\pi \in \Pi_k} \mathbb{E}_n\left[\sum_{t=0}^T \log \pi(A_t | H_{t-k:t})\right]$$

要求策略类满足**单调性** $\Pi_0 \subseteq \Pi_1 \subseteq \Pi_2 \subseteq \cdots$（logistic 回归、神经网络等常用模型均满足）。

### 四类 OPE 估计器的统一分析

**Theorem 2（OIS 的 bias-variance 分解）**：

$$\text{MSE}(\hat{v}_{OIS}(k)) = \frac{1}{n}\text{Var}\left(\text{Proj}_{\mathbb{T}(k)}(\lambda_T G_T)\right) + O\left(\frac{(k+1)C^{2T}R_{\max}^2}{n^{3/2}\varepsilon^2}\right)$$

- 第一项（方差）：$O(n^{-1})$，关于历史长度 $k$ **单调递减**
- 第二项（偏差）：$O(n^{-3/2})$，随 $k$ 增大和时间跨度 $T$ 增长而**增大**
- 投影解释：估计行为策略等价于将 $\lambda_T G_T$ 投影到正交于 score 函数张成的切空间的子空间 $\mathbb{T}(k)$ 上

对 SIS、DR、MIS 的影响总结：

| 估计器 | 偏差变化 | 方差变化 |
|--------|---------|---------|
| OIS    | ↑       | ↓       |
| SIS    | ↑       | ↓       |
| DR（Q 错误指定） | ↑ | ↓ |
| DR（Q 正确指定） | ↑ | → 不变 |
| MIS    | -       | ↑ 增大  |

**MIS 的反面结论（Theorem 8）**：对 MIS 估计器，增加历史长度反而**增大** MSE。直觉上，当 $k=T$ 时 MIS 退化为 SIS，失去了 marginalization 的优势。

### 非参数估计的扩展（Section 5）
将参数化策略估计推广到 sieve 非参数估计，放松了 realizability 假设（允许近似误差以 $o(n^{-1/4})$ 收敛），证明了即使在非参数设定下，估计行为策略仍能降低 OIS、SIS、DR 的渐近方差。

### 历史长度选择
提出基于 BIC 的选择准则：

$$h^* = \arg\min_h \left[2n\hat{\text{Var}}(h) - h\log(n)\right]$$

## 实验关键数据

实验环境：CartPole + MuJoCo（Inverted Pendulum, Double Inverted Pendulum, Swimmer）

| 实验发现 | 细节 |
|---------|------|
| SIS 历史依赖 | 更长历史 → 大样本下 MSE 显著降低，验证 Theorem 4 |
| DR（Q 错误指定） | 历史依赖可降低 MSE，但改善幅度受 Q 质量影响 |
| MIS 历史依赖 | 历史长度增加 → MSE **持续恶化**，验证 Theorem 8 |
| 偏差趋势 | 所有估计器中，历史依赖版本的偏差均高于 oracle 版本 |
| 一致性 | 所有估计器的 MSE 随样本量增大而收敛到零 |
| MuJoCo 扩展 | 连续动作空间、高维状态空间下结论一致 |

## 亮点与洞察
- **理论深度**：首次从 bias-variance 分解角度系统解释了 OPE 领域长期存在的"估计优于已知"悖论
- **统一框架**：一套理论覆盖 OIS/SIS/DR/MIS 四大类估计器，揭示了历史依赖对不同估计器的**差异化影响**
- **投影视角**：估计行为策略 ≈ 隐式 doubly robust 估计 ≈ 投影到约束子空间，三种视角统一
- **MIS 的反直觉结论**：历史依赖对 MIS 有害，这是因为 marginalization 本身已消除了轨迹级方差，再引入历史反而破坏了这一优势
- **实用指导**：BIC 准则为实际应用中的历史长度选择提供了原则性方法

## 局限与展望
- 理论分析假设有限状态/动作空间和有限时间跨度，对连续/无限时间跨度的推广尚不完整
- 非参数估计的结论仅证明了渐近优越性，未给出有限样本的显式偏差界
- 历史长度选择准则（BIC）虽有实验验证，但缺乏理论最优性保证
- 实验规模有限（CartPole 和简单 MuJoCo），未在高维复杂任务（如 Atari、机器人操作）上验证
- 未考虑部分可观测（POMDP）设定下历史依赖估计的效果

## 相关工作与启发
- **Hanna et al. (2019, 2021)**：实验发现估计历史依赖行为策略可降低 OIS 的 MSE，但未给出理论解释；本文正式回答了他们遗留的 open question
- **Hirano et al. (2003); Henmi et al. (2007)**：因果推理中估计倾向性得分优于已知得分的现象，本文将之推广到 sequential decision 场景
- **Rowland et al. (2020)**：条件 IS（Conditional IS）与本文的历史依赖 MIS 比有密切联系
- **Liu et al. (2018)**：MIS 方法缓解 horizon curse，本文揭示了在此基础上加入历史依赖的负面效果
- 启发：**估计量的"不完美"有时是优势**——添加噪声/估计误差可起到正则化/降方差的作用，这一原理可推广到其他统计估计问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次从理论上彻底揭示 OPE 中的经典悖论
- 实验充分度: ⭐⭐⭐ — 验证了理论但环境复杂度有限
- 写作质量: ⭐⭐⭐⭐⭐ — 从 bandit 例子逐步构建直觉，结构清晰
- 价值: ⭐⭐⭐⭐ — 对 OPE 实践有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [Wasserstein Policy Optimization](wasserstein_policy_optimization.md)
- [Bootstrap Off-policy with World Model](../../NeurIPS2025/reinforcement_learning/boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)
- [Bootstrap Off-policy with World Model (BOOM)](../../NeurIPS2025/reinforcement_learning/bootstrap_off-policy_with_world_model.md)
- [Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)

<!-- RELATED:END -->
