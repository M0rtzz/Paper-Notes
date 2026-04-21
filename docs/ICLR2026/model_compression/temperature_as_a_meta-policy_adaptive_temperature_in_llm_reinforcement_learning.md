---
title: >-
  [论文解读] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning
description: >-
  [ICLR 2026][模型压缩][温度调节] 提出 TAMPO（Temperature Adaptive Meta Policy Optimization），将采样温度重新定义为可学习的元策略，通过双层循环在内环做 LLM 策略优化、外环根据轨迹优势信号自适应更新温度分布，无需额外 rollout，在数学推理基准上一致超越固定温度基线。
tags:
  - ICLR 2026
  - 模型压缩
  - 温度调节
  - 元策略
  - GRPO
  - 自适应探索
  - 数学推理
---

# Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2602.11779](https://arxiv.org/abs/2602.11779)  
**代码**: 无  
**领域**: 模型压缩 / 高效 RL 训练  
**关键词**: 温度调节, 元策略, GRPO, 自适应探索, 数学推理

## 一句话总结

提出 TAMPO（Temperature Adaptive Meta Policy Optimization），将采样温度重新定义为可学习的元策略，通过双层循环在内环做 LLM 策略优化、外环根据轨迹优势信号自适应更新温度分布，无需额外 rollout，在数学推理基准上一致超越固定温度基线。

## 研究背景与动机

- 温度是 LLM 采样中控制探索-利用权衡的核心参数
    - 高温鼓励多样性但引入噪声，低温提高聚焦但可能过早收敛
- 现有 RL 训练（GRPO 等）将温度视为**固定超参数**，忽略了训练过程中的动态需求
- 熵正则化和 KL 惩罚虽也影响探索，但温度直接调制采样分布，更透明可控
- **核心论点**：温度应当是可学习的决策变量，而非手动调节的超参数

## 方法详解

### 整体框架

TAMPO 采用层级双循环结构：

- **内循环**：用选定温度 $T_s$ 生成 rollout，通过 GRPO 更新 LLM 策略 $\pi_\theta$
- **外循环**：复用内循环 rollout，根据轨迹优势信号更新温度元策略 $\pi(T)$

### 关键观察

每条轨迹隐式编码其"偏好温度"——使该轨迹最可能被生成的温度：

$$T^* = \arg\max_{T_k \in \mathcal{T}} \ell_{T_k}(\tau_i)$$

其中 $\ell_T(\tau_i) = \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \log \pi_{\theta,T}(o_{i,t} | s_{i,t})$ 为平均对数似然。

### 温度特定优势

对每条轨迹 $\tau_i$ 和虚拟候选温度 $T_k$：

1. 计算 $\ell_{T_k}(\tau_i)$：轨迹在温度 $T_k$ 下的似然
2. 用 sparsemax 归一化得 $\hat{\ell}_{T_k}(\tau_i)$（跨 $K$ 个候选温度求和=1）
3. 温度特定优势：$\mathcal{A}_i^{(T_k)} = \hat{\ell}_{T_k}(\tau_i) \cdot A_i$

直觉：
- 正优势轨迹 → 强化其最可能生成温度
- 负优势轨迹 → 惩罚其最可能生成温度

### 元策略更新

1. 批次聚合：$\mathcal{A}_\mathcal{B}^{(T_k)} = \frac{1}{|\mathcal{B}|G} \sum_b \sum_i \mathcal{A}_{b,i}^{(T_k)}$
2. EMA 平滑：$\bar{\mathcal{A}}_s^{(T_k)} = (1-\alpha)\bar{\mathcal{A}}_{s-1}^{(T_k)} + \alpha \mathcal{A}_\mathcal{B}^{(T_k)}$
3. Min-max 归一化得概率分布：$\pi_s(T_k) = \frac{\tilde{\mathcal{A}}_s^{(T_k)}}{\sum_j \tilde{\mathcal{A}}_s^{(T_j)}}$

### 温度采样

使用 nucleus sampling（top-p）从元策略中采样温度，$p=0.7$ 提供最佳探索-利用平衡。

### 设计特点

- **零额外 rollout**：完全复用内循环的轨迹数据
- **非可微优化**：温度在 LLM RL 中不可微，TAMPO 通过似然信号绕过此限制
- **可忽略开销**：元策略仅维护温度优势列表，推理时丢弃

## 实验关键数据

### 主实验：数学推理基准（DS-Qwen-1.5B）

| 方法 | Average | AIME24 | MATH-500 | AMC23 | Minerva | OlympiadBench |
|------|---------|--------|----------|-------|---------|---------------|
| DS-Qwen-1.5B (无 RL) | 39.1 | 13.3 | 76.2 | 45.0 | 22.8 | 38.4 |
| GRPO ($T_s$:0.9) | 42.0 | 20.0 | 75.2 | 50.0 | 26.1 | 38.7 |
| GRPO ($T_s$:1.5) | 42.6 | 23.3 | 75.4 | 52.5 | 22.8 | 39.0 |
| GRPO ($T_s$:0.9→1.5) | 42.8 | 16.7 | 76.6 | 55.0 | 24.6 | 41.0 |
| **TAMPO** | **44.5** | **23.3** | **76.8** | **55.0** | **27.9** | **39.6** |

### 消融：EMA 系数 $\alpha$

| $\alpha$ | Average | AIME24 | MATH-500 | AMC23 | Minerva | OlympiadBench |
|---------|---------|--------|----------|-------|---------|---------------|
| 0.01 | 41.6 | 20.0 | 75.2 | 50.0 | 25.4 | 37.5 |
| **0.05** | **44.5** | **23.3** | **76.8** | **55.0** | **27.9** | **39.6** |
| 0.10 | 43.6 | 23.3 | 75.4 | 57.5 | 23.2 | 38.8 |

### 消融：元策略采样策略

| top-p | Average |
|-------|---------|
| 0.9 | 43.0 |
| **0.7** | **44.5** |
| 0.5 | 42.2 |
| 0 (greedy) | 40.9 |

### 跨任务泛化（Qwen2.5-3B-Instruct → ECQA）

| 方法 | Pass@1 | Pass@8 |
|------|--------|--------|
| 无 RL | 73.06% | 77.76% |
| GRPO | 75.07% | 78.94% |
| **TAMPO** | **76.12%** | **79.67%** |

### 关键发现

1. TAMPO 平均超越最优固定温度基线 **+1.9%**（Pass@1）和 **+1.7%**（Pass@8）
2. 元策略学到的温度动态：warmup 后偏好高温 (~1.3) 鼓励探索，随训练逐渐降低
3. 贪心采样（$p=0$）导致最差结果 → 温度探索本身也需要探索
4. 训练耗时与基线完全相同（~9h54min on 8×V100）
5. 在常识推理任务上同样有效

## 亮点与洞察

- **将温度从超参数提升为决策变量**：新颖的问题形式化
- **无需额外 rollout**：通过虚拟温度似然计算巧妙复用已有数据
- **学到的温度策略与直觉一致**：先高后低的探索-利用切换
- **与现有 RL 完美兼容**：可插入 GRPO/DAPO/REINFORCE++ 等任意 critic-free 方法
- **计算开销可忽略**：仅维护 $K$ 个温度的优势估计

## 局限性

- 候选温度集 $\mathcal{T}$ 仍需手动设定范围和粒度
- 轨迹似然 w.r.t. 温度的 unimodal 性质在某些情况下可能不成立
- 仅在 1.5B 模型上做主实验，更大模型验证不足
- 温度元策略在不同 prompt 间共享，未探索 prompt 级别的自适应

## 相关工作

- Critic-free RL：GRPO、DAPO、REINFORCE++
- 探索-利用：ε-greedy、温度退火、UCB、熵正则化
- 元策略：MLSH（层级 RL）、Meta-SAC（自动熵系数）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 温度作为元策略的问题形式化新颖
- **技术深度**: ⭐⭐⭐⭐ — 理论推导清晰，虽方法本身简洁
- **实验充分性**: ⭐⭐⭐⭐ — 5 基准 + 消融全面，但模型规模有限
- **实用性**: ⭐⭐⭐⭐⭐ — 零额外成本即可提升 RL 训练效果

<!-- RELATED:START -->

## 相关论文

- [Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)
- [Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)

<!-- RELATED:END -->
