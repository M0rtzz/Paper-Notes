---
title: >-
  [论文解读] Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning
description: >-
  [NeurIPS 2025][LLM推理][贝尔曼残差] TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。 领域现状：PPO 和 GRPO 虽然在 LLM 对齐上成功…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "贝尔曼残差"
  - "价值学习"
  - "单次采样"
  - "无批评器"
  - "数学推理"
---

# Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2505.15311](https://arxiv.org/abs/2505.15311)  
**代码**: 待公开  
**领域**: LLM对齐  
**关键词**: 贝尔曼残差, 价值学习, 单次采样, 无批评器, 数学推理

## 一句话总结
TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。

## 研究背景与动机

**领域现状**：PPO 和 GRPO 虽然在 LLM 对齐上成功，但需要批评器模型、多次采样、重要性采样等复杂组件。

**现有痛点**：PPO 需 32 次采样/prompt，GRPO 需 16 次——计算成本高、超参数多、训练不稳定。

**核心矛盾**：RL 中价值方法通常更高效稳定，但在 LLM 高维离散动作空间的应用很少。

**本文目标** 设计一种简洁的价值学习方法，每个 prompt 仅需一次采样。

**切入角度**：将 LLM 的 logits 直接当作 Q 值估计器，最小化轨迹级贝尔曼残差。

**核心 idea**：$\mathcal{L}_{TBRM} = (\log \pi_m(y_w|x) - \log \pi_m(y_l|x) - R^*)^2$，让正确和错误回答的 log-likelihood 差逼近目标 margin。

## 方法详解

### 整体框架
给定 prompt $x$，生成一对响应 $(y_w, y_l)$（正确/错误），最小化 log-likelihood margin 与目标奖励 $R^*$ 的差的平方。无需批评器、无需价值归一化、无需 TD-$\lambda$。

### 关键设计

1. **轨迹级贝尔曼残差**:

    - 功能：最小化 $BR(x) = [\log \pi_m(y_w|x) - \log \pi_m(y_l|x) - R^*]^2$
    - 核心思路：将整个响应序列的 log-likelihood 作为 Q 值的代理
    - 设计动机：相比 token 级 TD，轨迹级更稳定且不需要中间值估计

2. **理论收敛保证**:

    - 收敛目标：$\pi_{opt} = \arg\min_\pi D_{KL}(\pi_{data} \| \pi) - \bar{R}(\pi)$
    - 与 RLHF 原始动机一致——保持与初始策略接近同时最大化奖励

## 实验关键数据

### 主实验

| 方法 | AIME24 | MATH | AMC | Minerva-Math | 平均 |
|------|--------|------|-----|-------------|------|
| PPO | 27.3% | 41.2% | 48.7% | 59.4% | 44.2% |
| GRPO | 29.1% | 43.1% | 49.2% | 61.5% | 45.7% |
| **TBRM** | **30.5%** | **44.8%** | **50.3%** | **62.9%** | **47.1%** |

### 效率对比

| 指标 | PPO | GRPO | TBRM |
|------|-----|------|------|
| 每 prompt 采样数 | 32 | 16 | **1** |
| GPU 内存 (GB) | 48 | 40 | **32** |
| 训练时间 (h) | 24.5 | 22.1 | **19.0** |

### 关键发现
- TBRM 在 AIME24 上超 PPO 3.2%，同时采样效率提升 32x
- 无需批评器，GPU 内存减少 33%
- 方法简洁，核心代码可能不超过 50 行

## 亮点与洞察
- **极致简洁**：无批评器、无价值归一化、无 TD-$\lambda$，仅用 trajectory log-likelihood margin
- **单次采样效率**：每个 prompt 仅需 1 个偏好对，计算效率提升 8-16x
- **理论一致性**：收敛目标与 RLHF 原始 KL 正则化动机一致

## 局限与展望
- 二元奖励（对/错）假设限制了对指令跟随等梯度奖励任务的适用性
- 目标 margin $R^*$ 的选择是启发式的
- 初始策略很弱时，生成数据质量有限

## 相关工作与启发
- **vs PPO/GRPO**：TBRM 用更简洁的价值方法替代了复杂的策略梯度方法
- **vs DPO**：DPO 是对比损失，TBRM 是贝尔曼残差——理论动机不同但形式相近

## 评分
- 新颖性: ⭐⭐⭐⭐ 将贝尔曼残差适配到 LLM 推理
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个数学数据集、详细效率分析
- 写作质量: ⭐⭐⭐⭐ 算法描述简洁易复现
- 价值: ⭐⭐⭐⭐⭐ 显著的效率改进，强烈推荐工业应用
**代码**: 待确认

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Value-Guided Search for Efficient Chain-of-Thought Reasoning](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](../../ICML2026/llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[ACL 2026\] DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping](../../ACL2026/llm_reasoning/dvmap_fine-grained_pluralistic_value_alignment_via_high-consensus_demographic-va.md)

</div>

<!-- RELATED:END -->
