---
title: >-
  [论文解读] Optimization over Sparse Support-Preserving Sets: Two-Step Projection with Global Optimality Guarantees
description: >-
  [ICML 2025][优化][sparse optimization] 针对带有额外支撑保持约束的稀疏优化问题，提出两步投影IHT算法（先硬阈值再投影凸集），在RSC/RSS条件下给出全局目标值保证（无系统误差），揭示稀疏度松弛与次优性间的新trade-off。
tags:
  - ICML 2025
  - 优化
  - sparse optimization
  - IHT
  - l0 constraint
  - support-preserving
  - two-step projection
---

# Optimization over Sparse Support-Preserving Sets: Two-Step Projection with Global Optimality Guarantees

**会议**: ICML 2025  
**arXiv**: [2506.08558](https://arxiv.org/abs/2506.08558)  
**代码**: 无  
**领域**: 优化  
**关键词**: sparse optimization, IHT, l0 constraint, support-preserving, two-step projection

## 一句话总结
针对带有额外支撑保持约束的稀疏优化问题，提出两步投影IHT算法（先硬阈值再投影凸集），在RSC/RSS条件下给出全局目标值保证（无系统误差），揭示稀疏度松弛与次优性间的新trade-off。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**: 稀疏优化中用l0约束相比凸松弛可精确控制稀疏度。IHT算法是标准方法，已有全局收敛保证。

**现有痛点**: 实际中常需额外约束（预算约束、范数约束等），现有算法要么需混合约束闭式投影（常不存在），要么仅提供局部保证。

**核心矛盾**: 全局保证的IHT不适用于混合约束；适用混合约束的方法仅有局部保证。两者之间的gap亟需填补。

**本文目标**: 在支撑保持约束（lp球等）下获得全局目标值保证的IHT变种。

**切入角度**: 定义"支撑保持集"——k-稀疏向量投影后支撑不变（如lp球p>=1）。

**核心idea**: 用两步投影替代混合约束欧几里得投影：(1) 硬阈值选top-k坐标；(2) 投影到凸集。简单且适用面广。

## 方法详解

### 整体框架
三个算法：IHT-2SP（确定性）、HSG-HT-2SP（随机）、HZO-HT-2SP（零阶）。核心步骤：梯度下降 -> 两步投影（硬阈值+凸投影）。

### 关键设计
1. **两步投影算子**: 第一步保留top-k坐标并置零其余，第二步投影到凸集Gamma。由于Gamma支撑保持，第二步不改变支撑。避免了混合约束闭式投影的需求。
2. **三点引理扩展**: 核心技术贡献。将经典三点引理推广到非凸两步投影设定，使全局目标值分析成为可能。该工具也简化了现有IHT分析，有独立的技术价值。
3. **稀疏度-次优性trade-off**: 参数rho控制松弛程度。$R(w_T) \leq (1+2\rho)R(\bar{w}) + \varepsilon$，稀疏度要求 $k = \Omega(\kappa_s^2\bar{k}/\rho^2)$。
4. **零阶改进**: 消除了de Vazelhes et al. (2022)中的非消失系统误差。首个无系统误差的零阶硬阈值收敛结果。

### 损失函数 / 训练策略
在RSC/RSS条件下分析。确定性版本线性收敛到近似解；随机和零阶版本以类似速率收敛。

## 实验关键数据

### 理论结果对比

| 算法 | 额外约束 | 保证 | 系统误差 |
|------|---------|------|---------|
| Jain et al. 2014 (IHT) | 无 | 全局 | 无 |
| Lu 2015 / Beck & Hallak 2016 | 支撑保持 | 仅局部 | - |
| **IHT-2SP (本文)** | **支撑保持** | **全局** | **无** |
| **HZO-HT-2SP (本文)** | **支撑保持** | **全局** | 仅O(mu)平滑误差 |

### 消融分析

| rho值 | 稀疏度k需求 | 次优性乘子 | 说明 |
|-------|-----------|-----------|------|
| rho小 | 大 | (1+2rho)接近1 | 接近精确但需更高稀疏度 |
| rho大 | 小 | (1+2rho)大 | 更稀疏但次优性增大 |

### 关键发现
- 两步投影在支撑保持集上等价于精确投影但计算简单
- 三点引理扩展作为副产品简化了经典IHT分析
- 零阶无系统误差即使在无约束情况下也是新结果

## 亮点与洞察
- **三点引理扩展**是核心技术创新，使非凸两步投影的全局分析成为可能。该工具可推广到其他非凸投影问题。
- **实用性强**：两步投影避免了闭式投影需求，适用于lp球、单纯形等多种常见约束。
- **稀疏度-次优性trade-off**在混合约束设定下全新——实际中可根据需求灵活选择。

## 局限与展望
- 限于支撑保持集，非支撑保持约束（一般多面体）不适用
- RSC/RSS假设在深度学习场景中可能不满足
- 稀疏度松弛k/rho^2可能实际中过大
- 缺少大规模实验验证

## 相关工作与启发
- **vs Jain et al. (2014)**: 经典IHT无约束；本文扩展到混合约束保持全局性
- **vs Lu (2015)**: 支撑保持约束但仅局部收敛；本文全局
- **vs de Vazelhes et al. (2022)**: 零阶有系统误差；本文消除

## 评分
- 新颖性: ⭐⭐⭐⭐ 三点引理扩展和trade-off是新颖贡献
- 实验充分度: ⭐⭐ 纯理论
- 写作质量: ⭐⭐⭐⭐ Table 1全面
- 价值: ⭐⭐⭐⭐ 填补混合稀疏约束全局保证空白
- 总体: ⭐⭐⭐⭐ 扎实优化理论贡献


## 补充分析

### 适用约束类型
支撑保持集（Definition 2.3）包括但不限于：
- **ℓp球** (p >= 1): 包括ℓ1球（Lasso约束）、ℓ2球（岭约束）、ℓ∞球
- **非负约束 + ℓp球**: 非负矩阵分解中常见
- **单纯形**: 概率分布约束
- **sign-free凸集**: Lu (2015)定义的一般类别
- **组合以上的交集**: 只要交集也是支撑保持的

### 计算复杂度
- 两步投影的每步代价：硬阈值O(d log k)，凸投影O(d)（ℓ2球）或O(d log d)（ℓ1球）
- 总复杂度与标准IHT相当——额外的投影步骤开销可忽略
- 零阶版本每步需2d次函数评估（坐标扰动），适用于梯度不可用的场景

### 与正则化方法的比较
本文的硬约束方法与ℓ1正则化的稀疏方法有本质区别：
- 硬约束直接控制稀疏度k，无需调参
- ℓ1正则化的稀疏度受正则化系数间接控制，无法精确指定
- 在需要精确稀疏度的场景（如组合优化、传感器选择）中，硬约束更合适

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](../../NeurIPS2025/optimization/evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](../../NeurIPS2025/optimization/optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)

<!-- RELATED:END -->
