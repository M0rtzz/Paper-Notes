---
title: >-
  [论文解读] Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect
description: >-
  [NeurIPS 2025][AI安全][weighted controlled direct effect] 针对加权控制直接效应（WCDE）建立三项基础理论：唯一可识别性的充要条件、非参数估计的影响函数推导、以及最小化渐近方差的最优协变量调整集刻画。
tags:
  - NeurIPS 2025
  - AI安全
  - weighted controlled direct effect
  - causal inference
  - mediation analysis
  - optimal adjustment
  - fairness
---

# Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect

**会议**: NeurIPS 2025  
**arXiv**: [2506.09871](https://arxiv.org/abs/2506.09871)  
**代码**: 无  
**领域**: 因果推断 / 统计方法  
**关键词**: weighted controlled direct effect, causal inference, mediation analysis, optimal adjustment, fairness

## 一句话总结
针对加权控制直接效应（WCDE）建立三项基础理论：唯一可识别性的充要条件、非参数估计的影响函数推导、以及最小化渐近方差的最优协变量调整集刻画。

## 研究背景与动机
**领域现状**：因果推断中直接效应（Direct Effect）的估计在公平性分析和中介分析中至关重要。控制直接效应（CDE）固定中介变量取值后测量处理对结果的直接影响。

**现有痛点**：CDE 要求指定中介变量的具体取值，但当处理效应随中介变量水平变化时，单一取值的 CDE 可能误导。加权控制直接效应（WCDE）通过在中介变量分布上取平均解决此问题，但缺乏系统的可识别性理论和最优估计方法。

**核心矛盾**：WCDE 的估计比平均处理效应（ATE）更复杂——中介变量-混杂因子的交互使得最优调整集与 ATE 的经典结果不同。

**切入角度**：从半参数统计理论出发，推导 WCDE 在观测数据中的影响函数和效率界。

**核心 idea**：WCDE 的最优调整集由中介-混杂交互结构唯一决定，且在某些 DAG 中 WCDE 与 CDE 在数值上不同。

## 方法详解

### 整体框架
给定因果 DAG $\mathcal{G}$，处理 $A$，结果 $Y$，中介 $M$。WCDE 定义为：
$$\text{WCDE}(a, a') = \sum_m [E[Y \mid do(A=a, M=m)] - E[Y \mid do(A=a', M=m)]] \cdot P(M=m)$$

### 关键理论贡献

1. **可识别性充要条件**
    - 时机：WCDE 何时能从观测数据唯一识别？
    - 定理：WCDE 可识别当且仅当 (i) 不存在 $A \to M$ 的未观测混杂路径，且 (ii) 不存在 $M \to Y$ 的未观测混杂路径
    - 重要推论：存在 DAG 使得 CDE 可识别但 WCDE 不可识别（反之亦然）

2. **影响函数推导**
    - 在正则渐近线性（RAL）估计器类中推导 WCDE 的有效影响函数
    - 形式：$\psi(O) = \sum_m [\mu(a,m,W) - \mu(a',m,W)] \cdot f(m) + \text{correction terms}$
    - 修正项涉及处理倾向得分 $e(W)$ 和中介密度 $f(m|W)$

3. **最优协变量调整集**
    - 定理：最优调整集 $W^*$ 需包含 (i) $A$ 和 $Y$ 的共同原因，(ii) $M$ 和 $Y$ 的共同原因
    - 核心差异：ATE 的最优调整集仅需 (i)，但 WCDE 因中介-混杂交互额外需要 (ii)
    - 推论：在某些 DAG 结构中，加入更多协变量反而增大方差（"无益混杂"现象）

### 估计策略
- 基于影响函数构造双重鲁棒（doubly robust）估计器
- 使用交叉拟合（cross-fitting）消除 Donsker 条件

## 实验关键数据

### 模拟实验一 — 方差比较 ($n=1000$, 100 次重复)

| 调整集 | MSE (×$10^{-3}$) | 方差 (×$10^{-3}$) | 偏差² (×$10^{-3}$) |
|--------|-------------------|--------------------|--------------------|
| 空集（不调整） | 有偏 | — | — |
| 全集（所有观测） | 8.72 | 8.41 | 0.31 |
| $\{W_1\}$（仅 ATE 最优） | 6.15 | 5.89 | 0.26 |
| $\{W_1, W_2\}$（本文最优） | **3.87** | **3.65** | **0.22** |
| Oracle | 3.52 | 3.52 | 0.00 |

### 模拟实验二 — 样本量对效率的影响

| 样本量 $n$ | 全集 MSE | ATE 最优集 MSE | WCDE 最优集 MSE |
|-----------|----------|---------------|-----------------|
| 500 | 17.3 | 12.1 | **7.8** |
| 1000 | 8.7 | 6.2 | **3.9** |
| 2000 | 4.5 | 3.1 | **1.9** |
| 5000 | 1.8 | 1.2 | **0.8** |

### 消融实验 — 不同 DAG 结构下的差异

| DAG 类型 | WCDE = CDE? | 全集最优? | 本文最优集优于 ATE 集? |
|---------|-------------|----------|----------------------|
| 无中介-混杂交互 | 是 | 否 | 相同 |
| 有中介-混杂交互 | 否 | 否 | **是，显著** |
| M 是 collider | 不可识别 | — | — |

### 关键发现
- 当存在中介-混杂交互时，WCDE 最优调整集比 ATE 最优集的 MSE 低 37-50%
- 使用全部协变量调整不是最优的，某些协变量会增加方差
- WCDE 与 CDE 在有中介-混杂交互的 DAG 中数值不同，验证了 WCDE 的实际意义
- 交叉拟合消除偏差的效果在小样本时尤为显著

## 亮点与洞察
- **理论贡献清晰且完整**：可识别性 → 影响函数 → 最优调整集，三部曲逻辑自洽
- **ATE 与 WCDE 的本质差异**：最优调整集不同的根源在于中介-混杂路径
- **公平性应用直接**：WCDE 隔离处理的直接效应，排除中介路径，是公平性评估的关键量

## 局限与展望
- 假设无未测量混杂（no unmeasured confounding），在实践中可能不满足
- 连续中介变量时需要核密度估计，高维时效率下降
- 多中介变量场景的最优调整集可能组合爆炸
- 仅考虑无参数模型，半参数效率界的实用性有待探索

## 相关工作与启发
- **Pearl (2001)**：因果直接效应和间接效应的定义
- **Henckel et al. (2022)**：ATE 的最优调整集理论
- **VanderWeele (2015)**：中介分析经典教材
- 启发：不同因果量（ATE/CDE/WCDE）的最优估计策略可能完全不同

## 评分
- 新颖性: ⭐⭐⭐⭐ WCDE 的系统理论首次建立
- 实验充分度: ⭐⭐⭐⭐ 多 DAG 结构模拟充分验证理论
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，动机清晰
- 价值: ⭐⭐⭐⭐ 公平性和中介分析的理论基础

<!-- RELATED:START -->

## 相关论文

- [Probabilistic Reasoning with LLMs for K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)
- [Locally Optimal Private Sampling: Beyond the Global Minimax](locally_optimal_private_sampling_beyond_the_global_minimax.md)
- [Beyond Last-Click: An Optimal Mechanism for Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](../../ICML2025/ai_safety/the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)

<!-- RELATED:END -->
