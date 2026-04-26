---
title: >-
  [论文解读] SUSD: Structured Unsupervised Skill Discovery through State Factorization
description: >-
  [ICLR 2026][无监督技能发现] 提出 SUSD（Structured Unsupervised Skill Discovery），通过将状态空间分解为独立因子并为每个因子分配专属技能变量，结合好奇心驱动的因子加权机制，实现在多物体/多智能体复杂环境中发现覆盖全部可控因子的多样化技能。
tags:
  - ICLR 2026
  - 无监督技能发现
  - 状态分解
  - 距离最大化
  - 好奇心驱动
  - 层次强化学习
---

# SUSD: Structured Unsupervised Skill Discovery through State Factorization

**会议**: ICLR 2026  
**arXiv**: [2602.01619](https://arxiv.org/abs/2602.01619)  
**代码**: [https://github.com/hadi-hosseini/SUSD](https://github.com/hadi-hosseini/SUSD)  
**领域**: 无监督技能发现 / 强化学习  
**关键词**: 无监督技能发现, 状态分解, 距离最大化, 好奇心驱动, 层次强化学习

## 一句话总结

提出 SUSD（Structured Unsupervised Skill Discovery），通过将状态空间分解为独立因子并为每个因子分配专属技能变量，结合好奇心驱动的因子加权机制，实现在多物体/多智能体复杂环境中发现覆盖全部可控因子的多样化技能。

## 研究背景与动机

- **无监督技能发现（USD）的目标**：在无外部奖励下自主学习多样化技能，用于下游任务
- **两大技术路线**：
    - **互信息（MI）方法**（DIAYN 等）：最大化技能变量与状态的互信息，但因变换不变性倾向学习简单静态行为
    - **距离最大化（DSD）方法**（CSD、METRA 等）：通过最大化状态空间距离鼓励动态行为，但在复杂多物体环境中只关注最易控制的因子
- **DSD 的关键局限**：缺乏确保技能多样性覆盖所有可控因子的机制。在 Ant、HalfCheetah 等简单环境中表现好，但在多物体环境（多智能体、厨房）中退化。
- **核心解决思路**：利用环境的组合结构作为归纳偏置，将状态空间分解，为每个因子学习专属技能。

## 方法详解

### 整体框架

SUSD 基于 DSD 框架，包含三个核心组件：分解化嵌入、好奇心因子加权、对偶梯度下降训练。

### 1. 状态空间分解

将状态空间分解为 $N$ 个因子：$\mathcal{S} := \mathcal{S}^1 \times \cdots \times \mathcal{S}^N$，技能空间同步分解为 $\mathcal{Z} := \mathcal{Z}^1 \times \cdots \times \mathcal{Z}^N$。

映射函数 $\phi$ 拆分为 $N$ 个独立网络 $\phi_i(s^i)$，每个仅处理对应因子。优化目标变为：

$$\sup_{\pi, \{\phi_i\}_{i=1}^N} \mathbb{E}_{p(\tau, z)} \sum_{i=1}^N \sum_{t=0}^{T-1} (\phi_i(s_{t+1}^i) - \phi_i(s_t^i))^\top z^i$$

$$\text{s.t.} \sum_{i=1}^N \|\phi_i(s'^i) - \phi_i(s^i)\|_2 \leq 1, \quad \forall (s, s') \in \mathcal{S}_{\text{adj}}$$

### 2. 好奇心驱动的因子加权

训练密度模型 $q_\theta(s'|s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$，利用其因子边际估计每个因子的"好奇心"权重：

$$-\log q_\theta(s_{t+1}^i | s_t) \propto (s_{t+1}^i - \mu_\theta^i(s_t))^\top \Sigma_\theta^i(s_t)^{-1} (s_{t+1}^i - \mu_\theta^i(s_t))$$

高好奇心值 = 低概率转移 = 需要更多关注。$\sqrt{-\log q_\theta(s_{t+1}^i|s_t)}$ 作为合法距离度量整合到目标中。

### 3. 最终优化目标与内在奖励

每个因子的技能奖励：

$$r_i^{\text{SUSD}} := (\phi_i(s_{t+1}^i) - \phi_i(s_t^i))^\top z^i$$

总内在奖励（加权求和）：

$$R := \sum_{i=1}^N \sqrt{-\log q_\theta(s_{t+1}^i | s_t)} \cdot r_i^{\text{SUSD}}$$

映射函数和 Lagrange 乘子通过对偶梯度下降更新，策略使用 SAC 训练。

## 实验结果

### 下游任务表现（Multi-Particle 和 Kitchen 环境）

| 方法 | MP 平均回报 | Kitchen 平均回报 |
|------|------------|-----------------|
| DIAYN | 低 | 低 |
| LSD | 低 | 低 |
| CSD | 中 | 低 |
| METRA | 中 | 低-中 |
| DUSDi | 中 | 低 |
| **SUSD** | **高** | **高** |

SUSD 在复杂分解环境中显著优于所有基线，尤其 Kitchen 环境差距更大。

### 技能学习阶段的偶然任务完成

| 任务 | SUSD | CSD | METRA | LSD | DUSDi |
|------|------|-----|-------|-----|-------|
| BiP (黄油入锅) | **39.9±18.5** | 0.0 | 0.0 | 0.0 | 0.0 |
| MiP (肉丸入锅) | **58.9±25.8** | 0.0 | 0.0 | 0.0 | 2.5 |
| PoS (锅上灶) | **20.5±18.0** | 0.0 | 0.0 | 0.0 | 1.3 |

SUSD 在技能学习阶段就能偶然完成下游任务，其他方法完全做不到。

### 因子解码误差

| 方法 | Multi-Particle | Kitchen | 2D-Gunner |
|------|---------------|---------|-----------|
| **SUSD** | **0.060** | **0.014** | **0.080** |
| METRA | 0.147 | 0.028 | 0.186 |
| CSD | 0.313 | 0.049 | 0.404 |
| LSD | 0.308 | 0.038 | 0.224 |

SUSD 的潜在技能嵌入包含最丰富的因子信息。

### 关键发现

- SUSD 实现了显著更好的状态覆盖——尤其是最差智能体的覆盖率
- 在非分解环境（Ant、HalfCheetah）中仍保持竞争力
- 好奇心加权机制有效引导注意力到欠探索因子

## 亮点与洞察

1. **分解化 DSD 的首创**：将状态分解这一归纳偏置首次引入 DSD 框架
2. **细粒度好奇心加权**：不同于 CSD 的粗粒度（整个状态转移一个权重），SUSD 为每个因子独立计算权重
3. **可组合技能**：分解化技能表征天然支持技能组合和链接
4. **引理支撑**：通过 Lemma 4.1 严格证明距离项可作为内在奖励的系数

## 局限性

- 需要预先知道状态的分解结构（哪些维度属于哪个因子）
- 在像素输入场景中需要额外的解耦表征学习
- 技能空间维度随因子数量线性增长
- 在非分解环境中优势不明显

## 相关工作

- **MI-based USD**：DIAYN、DADS、DUSDi（分解化 MI）
- **DSD-based USD**：LSD、CSD、METRA
- **状态分解 in RL**：FMDP、因果分解、DUSDi

## 评分

- **创新性**: ⭐⭐⭐⭐ — 分解化 DSD + 好奇心因子加权的组合新颖有效
- **技术深度**: ⭐⭐⭐⭐ — 理论推导扎实（Lemma 4.1），优化框架完整
- **实验充分性**: ⭐⭐⭐⭐ — 多环境评估，消融和定性分析丰富
- **实用价值**: ⭐⭐⭐ — 依赖状态分解假设，适用场景有限但效果突出

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Periodic Skill Discovery](../../NeurIPS2025/reinforcement_learning/periodic_skill_discovery.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)

<!-- RELATED:END -->
