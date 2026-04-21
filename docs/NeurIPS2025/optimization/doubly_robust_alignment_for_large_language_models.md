---
title: >-
  [论文解读] Doubly Robust Alignment for Large Language Models
description: >-
  [NeurIPS 2025][优化][RLHF] DRPO 借鉴因果推断中的双重稳健估计方法，提出一种偏好优化算法，当偏好模型或参考策略任一正确指定时即可保持一致性，在理论和实验上均优于 PPO/DPO 及其变体。
tags:
  - NeurIPS 2025
  - 优化
  - RLHF
  - 双重稳健
  - 偏好优化
  - DPO
  - 模型鲁棒性
---

# Doubly Robust Alignment for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.01183](https://arxiv.org/abs/2506.01183)  
**代码**: https://github.com/DRPO4LLM/DRPO4LLM  
**领域**: LLM对齐 / 强化学习  
**关键词**: RLHF, 双重稳健, 偏好优化, DPO, 模型鲁棒性

## 一句话总结
DRPO 借鉴因果推断中的双重稳健估计方法，提出一种偏好优化算法，当偏好模型或参考策略任一正确指定时即可保持一致性，在理论和实验上均优于 PPO/DPO 及其变体。

## 研究背景与动机

**领域现状**：RLHF 是 LLM 对齐的主流范式，现有算法分为 reward-based（PPO 等）和 preference-based（DPO、IPO 等）两大类，均在实践中取得了显著成功。

**现有痛点**：现有算法各自存在模型误指定（misspecification）脆弱性——(a) PPO 依赖 Bradley-Terry（BT）偏好模型假设，对奖励估计高度敏感，容易 reward hacking；(b) DPO 虽绕过奖励估计但对参考策略 $\pi_\text{ref}$ 的指定敏感；(c) 两类方法对模型假设的违反容忍度低。

**核心矛盾**：BT 模型假设人类偏好满足传递性、上下文无关性等，实证表明常被违反。同时参考策略不一定已知或准确。没有哪种现有方法能同时对偏好模型和参考策略两者的误指定保持鲁棒。

**本文目标** 设计一种 RLHF 算法，在偏好模型或参考策略任一正确时就能保证一致性。

**切入角度**：从计量经济学和因果推断中的双重稳健（DR）估计方法出发——DR 在治疗效应估计中只需倾向得分模型或结果模型之一正确即可，类比到 RLHF 中偏好模型和参考策略的角色。

**核心 idea**：将 DR 估计框架引入偏好评估，构建对偏好模型和参考策略"双重稳健"的偏好优化算法。

## 方法详解

### 整体框架
给定数据集 $\mathcal{D} = \{(X, Y^{(1)}, Y^{(2)}, Z)\}$（prompt、两个回答、偏好标签），目标是找到最优策略 $\pi$ 使其总偏好最大。方法分三步：(1) 构建 DR 偏好评估估计器；(2) 基于 DR 估计器进行偏好优化；(3) 通过 KL 正则化稳定训练。

### 关键设计

1. **总偏好定义与基线估计器**:

    - 功能：定义目标策略相对参考策略的总偏好 $p^*(\pi) = \mathbb{P}(\pi \succ \pi_\text{ref})$
    - 核心思路：DM 估计器直接用偏好模型估计 $\hat{g}$ 插入计算，依赖 $\hat{g}$ 的正确性；IS 估计器用重要性采样比 $w(y,x) = \pi(y|x)/\pi_\text{ref}(y|x)$ 转换期望，依赖 $\hat{\pi}_\text{ref}$ 的正确性
    - 设计动机：两种估计器各依赖一个模型的正确性，都不够鲁棒

2. **双重稳健（DR）偏好估计器**:

    - 功能：结合 DM 和 IS 构建对两者误指定都鲁棒的估计器
    - 核心思路：估计函数为 $\psi = \frac{1}{2}\sum_{a=1}^2 \mathbb{E}_{y \sim \pi}[\hat{g}(X,y,Y^{(a)})] + \frac{1}{2}\sum_{a=1}^2 (-1)^{a-1} \frac{\pi(Y^{(a)}|X)}{\hat{\pi}_\text{ref}(Y^{(a)}|X)}[Z - \hat{g}(X,Y^{(1)},Y^{(2)})]$。第一项是 DM，第二项是增广项，用偏好残差 $Z - \hat{g}$ 校正 DM 偏差
    - 设计动机：当 $\hat{g} = g^*$ 时增广项期望为零（DM 已正确）；当 $\hat{\pi}_\text{ref} = \pi_\text{ref}$ 时增广项自动达到 IS 估计的效果。任一正确即保证一致性
    - 与 bandit DR 的区别：配对比较中每个数据元组被使用两次（正反），有效降低方差

3. **DRPO 偏好优化**:

    - 功能：利用 DR 估计器求解最优策略
    - 核心思路：$\hat{\pi} = \arg\max_{\pi \in \Pi} \{\hat{p}_\text{DR}(\pi) - \beta \mathbb{E}_X D_\text{KL}[\pi(\cdot|X) \| \hat{\pi}_\text{ref}(\cdot|X)]\}$
    - 实现细节：(a) IS 比率裁剪防止极大值；(b) 设计伪目标函数支持目标策略 Monte Carlo 采样；(c) 采用 GRPO 的 KL 散度方差缩减

### 理论保证
- **Theorem 2 (MSE)**：DR 估计器的 MSE = SEB + 产品偏差项 $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\|^2 \cdot \|\hat{g}-g^*\|^2)$，偏差为两个误差的乘积
- **Corollary 3 (双重稳健)**：任一模型正确，MSE 趋零
- **Corollary 4 (半参数效率)**：两者都近似正确时，MSE 渐近达到 SEB
- **Theorem 7 (次优性)**：DRPO 次优性界 $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\| \cdot \|\hat{r}-r^*\|)$，误差影响通过乘积出现，比 PPO 的 $O(\|\hat{r}-r^*\|)$ 和 DPO 的 $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\|)$ 都更鲁棒

## 实验关键数据

### 偏好评估（IMDb 对照实验）
在偏好模型和参考策略可控的合成环境中验证 DR 性质：
- 两者都正确：MSE 最低，1500 样本后接近零
- 仅偏好模型正确/仅参考策略正确：MSE 显著降低
- 两者都错误：MSE 大且不随样本增加改善
结果与双重稳健理论完美吻合。

### 主实验（偏好优化）

| 数据集 | 对比 | DRPO-BT 胜率 |
|--------|------|-------------|
| TL;DR | vs Dr. DPO | 72.5% |
| TL;DR | vs rDPO | 65.0% |
| TL;DR | vs cDPO | 63.5% |
| TL;DR | vs CPO | 90.0% |
| TL;DR | vs IPO | 98.5% |
| TL;DR | vs RSO | 69.5% |

| 模型 | LC 胜率 | 胜率 |
|------|---------|------|
| Dr. DPO | 92.16% | 90.93% |
| rDPO | 86.89% | 85.71% |
| cDPO | 85.05% | 84.28% |
| **DRPO** | **86.38%** | **84.84%** |
| IPO | 78.29% | 78.88% |
| RSO | 80.62% | 79.50% |

### 关键发现
- TL;DR 上 DRPO 大幅超越所有 DPO 变体（vs IPO 胜率 98.5%），因该数据集存在参考策略误指定
- HH 上 DRPO-GPM 域内表现最佳；DRPO-BT 对 PPO 胜率 57%，尽管使用同样的偏好模型，体现其对偏好模型误指定的鲁棒性
- 域外（AlpacaEval 2.0）上 DRPO 与专门处理噪声的 DPO 变体（cDPO、rDPO）持平

## 亮点与洞察
- **从因果推断到 RLHF 的跨领域创新**：将 DR 估计的核心思想（偏差校正增广项）完美迁移到偏好优化中，理论根基深厚
- **乘积形式的误差界**：DRPO 次优性界中两种误差以乘积出现，这意味着即使两个模型都不完美但各有中等精度，DRPO 仍优于 PPO/DPO
- **不依赖 BT 假设**：Theorem 5 不需要 BT 模型假设即可给出性能界，对 preference-based RLHF 有普遍意义

## 局限与展望
- IS 比率裁剪引入偏差，裁剪阈值的选择缺乏理论指导
- 当 $\pi$ 与 $\pi_\text{ref}$ 差距过大时（高 IS 比率），方差问题仍然存在
- 实验规模有限（基于较小的开源模型），在 SOTA 级别大模型上的效果未验证
- DRPO 需要单独训练偏好模型 $\hat{g}$ 和参考策略 $\hat{\pi}_\text{ref}$，增加了 pipeline 复杂性

## 相关工作与启发
- **vs DPO**: DPO 将奖励参数化为策略比，消除奖励模型但对参考策略敏感；DRPO 通过增广项校正参考策略偏差
- **vs PPO**: PPO 两阶段（奖励学习+RL），对奖励估计高度敏感；DRPO 无需准确奖励
- **vs IPO**: IPO 同为 preference-based 优化同一目标，但不鲁棒于参考策略误指定；DRPO 增加了 IS 增广项

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将双重稳健方法完整引入 RLHF，理论动机和技术设计都很新颖
- 实验充分度: ⭐⭐⭐⭐ 合成实验完美验证理论，真实任务对比充分，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，方法动机层层递进，可视化辅助理解
- 价值: ⭐⭐⭐⭐⭐ 为 RLHF 提供了理论上更优且实践中更鲁棒的新范式，有望成为重要参考

<!-- RELATED:START -->

## 相关论文

- [VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [Large Language Bayes](large_language_bayes.md)
- [Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [Subspace Optimization for Large Language Models with Convergence Guarantees](../../ICML2025/optimization/subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)

<!-- RELATED:END -->
