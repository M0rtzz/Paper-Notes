---
title: >-
  [论文解读] Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems
description: >-
  [ICLR 2026][多模态][多智能体系统] 提出 Agora 框架，将多智能体 VLM 协调问题重构为去中心化的不确定性交易市场——将认知不确定性铸造为可量化的三维可交易资产（感知/语义/推理），通过利润驱动的交易协议和市场感知的 Thompson Sampling Broker 实现成本高效的均衡分配，在 5 个多模态基准上一致超越启发式方法（如 MMMU 上 +8.5% 准确率同时成本降低 3 倍以上）。
tags:
  - ICLR 2026
  - 多模态
  - 多智能体系统
  - VLM协调
  - 不确定性交易
  - 市场机制
  - Thompson Sampling
  - 成本优化
---

# Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems

**会议**: ICLR 2026  
**arXiv**: [2601.18735](https://arxiv.org/abs/2601.18735)  
**代码**: 无  
**领域**: 多模态VLM/多智能体协调  
**关键词**: 多智能体系统, VLM协调, 不确定性交易, 市场机制, Thompson Sampling, 成本优化

## 一句话总结

提出 Agora 框架，将多智能体 VLM 协调问题重构为去中心化的不确定性交易市场——将认知不确定性铸造为可量化的三维可交易资产（感知/语义/推理），通过利润驱动的交易协议和市场感知的 Thompson Sampling Broker 实现成本高效的均衡分配，在 5 个多模态基准上一致超越启发式方法（如 MMMU 上 +8.5% 准确率同时成本降低 3 倍以上）。

## 研究背景与动机

**领域现状**：VLM 驱动的多智能体系统（MAS）在视觉理解任务中展现出强大的集体智能潜力，但在实际部署中面临**经济可行性危机**——操作成本急剧上升，阻碍大规模部署。

**问题根源**：现有范式将智能视为"蛮力商品"而非稀缺经济资源。当认知不确定性（主要成本驱动因素）缺乏经济纪律管理时，产生冗余计算，导致决策成本过高。

**现有方法缺陷**：
   - **聚合类启发式（MoA）**：通过统计共识等同认知真理，假设误差独立同分布（i.i.d.），但 VLM 共享架构偏差导致相关误差被放大，产生系统性幻觉。
   - **路由类启发式（KABB）**：依赖历史性能和语义相似度的代理评分 $S = \alpha \cdot P_{\text{hist}} + \beta \cdot \text{Sim}_{\text{sem}}$，混淆过去表现与未来成本效益，既**不感知成本**（代价向量 $\mathbf{c}$ 缺失）也**不感知不确定性结构**（向量 $\mathbf{u}(t)$ 被压缩为标量）。

**理论证明**：论文正式证明了"不可知协调低效定理"（Theorem 1）——任何同时满足成本不可知和不确定性结构不可知的协调机制对于优化目标 $\min_\pi \mathbb{E}[\mathcal{C}(\pi, \mathbf{u}(t), \mathbf{c}, \Xi)]$ 而言是可证明次优的。

**核心动机**：需要从启发式补丁转向一种拥抱去中心化本质的新范式——基于市场机制的协调，通过价格信号和经济激励实现信息不对称下的高效协调。

## 方法详解

### 整体框架

Agora 框架将多智能体协调重新设计为**去中心化微经济**，包含三个核心模块：

1. **不确定性资产铸造（§3.1）**：将认知不确定性形式化为结构化可交易资产
2. **利润驱动交易协议（§3.2）**：基于经济理性规则执行不确定性交易
3. **市场感知 Broker（§3.3）**：基于扩展 Thompson Sampling 的智能初始化与调度

系统设置：$N$ 个异构 VLM 智能体 $\mathcal{A} = \{a_1, \dots, a_N\}$，每个智能体有单位处理成本 $c_i > 0$ 和专长向量 $\boldsymbol{\xi}_i = [\xi_{i,\text{perc}}, \xi_{i,\text{sem}}, \xi_{i,\text{inf}}]^T$。优化目标为：

$$\min_\pi \mathbb{E}_{t \sim \mathcal{T}}[\mathcal{C}(\pi, \mathbf{u}(t), \mathbf{c}, \Xi)] \quad \text{s.t.} \quad \|\mathbf{u}_{\text{final}}\| \leq \epsilon$$

### 关键设计1：三维不确定性资产模型

**核心思路**：将总不确定性分解为可交易的认知不确定性（Epistemic）和不可交易的偶然不确定性（Aleatoric）。可交易部分进一步分解为三维向量：

$$\mathbf{u}_{\text{epis}} = [u_{\text{perc}}, u_{\text{sem}}, u_{\text{inf}}]^T$$

- **感知不确定性** $u_{\text{perc}}$：视觉感知层面的不确定性（如 OCR、物体识别）
- **语义不确定性** $u_{\text{sem}}$：语义理解层面的不确定性（如上下文推断）
- **推理不确定性** $u_{\text{inf}}$：逻辑推理层面的不确定性（如数学推理、因果推断）

每个智能体维护一个不确定性组合：

$$\mathbf{U}(a_i, t) = \mathbf{U}_{\text{base}}(a_i, t) + \sum_{j \neq i} \mathbf{U}_{\text{transfer}}(a_j \to a_i, t)$$

**为什么有效**：将单一标量不确定性向量化，使系统能够将不同类型的不确定性路由到最擅长处理该类型的专家智能体，避免了"一刀切"的冗余计算。

### 关键设计2：利润驱动交易协议

**核心机制**：每笔交易必须满足经济理性——只有当交易降低系统总成本时才执行。将不确定性包 $T_{ij}(t)$ 从智能体 $a_i$ 转移到 $a_j$ 的成本变化为：

$$\Delta \mathcal{C}(T_{ij}(t)) = T_{ij}(t) \cdot [c_j(1 - \xi_j) - c_i]$$

交易准入规则：

$$\text{Execute trade}(i \to j, T_{ij}(t)) \iff (\Delta \mathcal{C} < 0) \wedge (U_j(t) + T_{ij}(t) \leq C_j(t))$$

即交易必须**有利可图**（$\Delta \mathcal{C} < 0$）且**可行**（接收方有足够认知容量）。每次合法交易都是对全局成本函数的贪心下降步。

### 关键设计3：市场感知 Broker

Broker 基于扩展 Thompson Sampling 选择初始智能体，最大化市场感知的期望效用：

$$\tilde{\theta}_S^{(t)} = (\mathbb{E}[\text{Reward}_S^{(t)}] - \text{Cost}_S^{(t)}) \cdot \exp(-\lambda \cdot \text{Dist}(S, t)) \cdot U_{\text{strategic}}(S)^\omega \cdot \text{Synergy}(S)^\eta \cdot \gamma^{\Delta t}$$

各因子含义：
- **净回报**：期望奖励减去成本，确保经济理性
- **任务距离** $\text{Dist}(S, t)$：衡量智能体与任务的匹配度
- **战略不确定性** $U_{\text{strategic}}$：探索-利用平衡
- **协同效应** $\text{Synergy}$：智能体组合的互补性
- **时间衰减** $\gamma^{\Delta t}$：权重随时间递减

### Agora 算法流程

1. **Phase 1（初始化）**：Broker 选择最优初始智能体 $a_{\text{handler}}$，分配初始不确定性
2. **Phase 2（迭代交易）**：反复寻找最有利可图的交易并执行，直到无法找到更多有利交易，市场收敛到局部最优均衡

## 实验关键数据

### 五大基准综合性能

| 模型 | MMMU(Val) | MMBench_V11 | MathVision | InfoVQA | CC-OCR |
|------|-----------|-------------|------------|---------|--------|
| qwen2.5vl-72b | 70.2% | 88.4% | 39.3% | 87.3% | 79.8% |
| gemini-2.0-flash | 70.7% | 83.0% | 41.3% | 83.2% | 73.1% |
| qwen2.5vl-7b | 58.6% | 82.6% | 25.1% | 82.6% | 77.8% |
| gpt-4o-mini | 60.0% | 76.3% | 26.3% | 68.7% | 64.2% |
| gemini-2.5-pro | 81.7% | 88.3% | 63.5% | 81.0% | 73.0% |
| InternVL3-78B | 72.2% | 87.7% | 43.1% | 84.1% | 80.3% |
| **Agora (Ours)** | **79.2% (+8.5%)** | **89.5% (+1.1%)** | **44.3% (+3.0%)** | **88.9% (+1.6%)** | **81.2% (+1.4%)** |

Agora 在 MMBench、InfoVQA、CC-OCR 上取得 SOTA，在 MMMU 上仅次于专用推理模型 gemini-2.5-pro。

### MAB 策略消融（MMMU Val）

| 方法 | Acc (%) | $U_{\text{final\_epis}}$ ↓ | COI ↓ | UAPS (%) ↑ |
|------|---------|--------------------------|-------|------------|
| **Agora (Ours)** | **79.0** | **0.15** | **1.2** | **70.5** |
| Agora (No Trading) | 75.5 | 0.22 | 1.0 | 65.0 |
| KABB Selector + Trading | 76.0 | 0.25 | 1.5 | 65.5 |
| PPO Selector + Trading | 74.0 | 0.28 | 1.6 | 62.0 |
| MCTS Selector + Trading | 74.5 | 0.26 | 1.4 | 63.0 |
| DQN Selector + Trading | 73.0 | 0.30 | 1.7 | 60.0 |

### 模块消融（MMBench_V11）

| 变体 | Acc (%) ↑ | $U_{\text{final}}$ ↓ | COI ↓ | UAPS (%) ↑ | Rel. Cost ↓ |
|------|-----------|---------------------|-------|------------|-------------|
| **Agora (Full)** | **89.50** | **0.16** | **1.25** | **78.33** | 1.00 |
| w/o $U_{\text{strategic}}$ | 86.42 | 0.23 | 1.45 | 71.58 | 1.06 |
| w/o Synergy | 87.91 | 0.19 | 1.30 | 74.88 | 1.03 |
| w/o Dist | 88.53 | 0.18 | 1.27 | 76.21 | 1.01 |
| w/o $\Delta t$ | 89.05 | 0.17 | 1.26 | 77.14 | 1.00 |
| Only Net Return | 82.15 | 0.31 | 1.08 | 60.72 | 0.92 |

去除战略不确定性因子 $U_{\text{strategic}}$ 导致最大性能下降（Acc -3.08%, UAPS -6.75），证明其在引导智能体选择中的关键作用。

## 亮点与洞察

1. **经济学视角的范式创新**：首次将多智能体VLM协调建模为去中心化不确定性交易市场，从微观经济学（市场均衡、套利、资产交易）角度系统解决了成本-性能权衡问题，而非简单堆叠模型。

2. **理论基础扎实**：正式证明了现有启发式方法（MoA、KABB）的次优性定理，不是凭直觉而是通过严格的数学论证建立了新框架的必要性。

3. **成本效率优势显著**：Agora 在 N=1 时就能达到 87.5% 准确率（成本比 0.02057），甚至优于昂贵的 SOTA 模型；N=8 时准确率达到 89.6%，且成本比始终显著低于所有基线。

4. **可扩展性良好**：性能随智能体池多样化平滑上升，在 N=8 左右达到边际收益递减点——符合经济学规律，不需要无限扩展代价高昂的智能体池。

5. **不确定性分解有效**：三维不确定性（感知/语义/推理）分解使系统能根据任务特性精准分配专家，最终残余不确定性显著低于 KABB（0.16 vs 0.21）。

## 局限性

1. **智能体池依赖**：Agora 的性能上界受限于可用智能体池的质量和多样性，若池中所有模型在某类任务上都弱，市场交易机制也无法弥补。

2. **不确定性量化假设**：三维不确定性分解假设感知/语义/推理可清晰分离，但实际中这些维度可能存在耦合和交互。

3. **基准覆盖度**：实验仅覆盖多选和简答类视觉理解任务，未验证在开放生成、视觉对话等场景的有效性。

4. **成本模型简化**：当前成本模型假设每个智能体的单位处理成本固定，未考虑延迟、并发、网络开销等实际部署因素。

5. **Broker 收敛性**：Thompson Sampling 的探索-利用权衡依赖历史交互积累，在冷启动阶段可能效率较低。

## 相关工作

- **多智能体LLM/VLM系统**：MoA（Guo et al., 2024）通过层级聚合多模型输出；KABB（Zhang et al., 2025）基于知识库路由；FrugalGPT（Chen et al., 2024c）和 RouteLLM（Ong et al., 2024）聚焦成本优化路由但牺牲准确率。
- **不确定性量化**：传统 UQ 方法（MC Dropout、Deep Ensemble）聚焦单模型不确定性估计，Agora 将 UQ 扩展到多智能体间的不确定性交易与传递。
- **多臂老虎机（MAB）**：Thompson Sampling 是经典的探索-利用框架，Agora 将其扩展为市场感知的多因子效用函数。

## 评分

| 维度 | 分数 (1-10) |
|------|-------------|
| 创新性 | 9 |
| 理论深度 | 9 |
| 实验充分性 | 8 |
| 实用价值 | 7 |
| 写作质量 | 8 |
| **总体** | **8.5** |

创新性和理论深度是本文最大亮点，从经济学角度重新定义多智能体协调问题具有范式创新意义。实验覆盖多个基准、消融充分，但缺少开放生成类任务验证。实用价值受限于部署复杂性和智能体池依赖。

<!-- RELATED:START -->

## 相关论文

- [Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)

<!-- RELATED:END -->
