---
title: >-
  [论文解读] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs
description: >-
  [ICLR 2026][推理语言模型] 本文提出 Radar 框架，将推理语言模型（RLM）的自适应推理问题建模为多目标优化，利用项目反应理论（IRT）联合估计可解释的查询难度和模型配置能力参数，实现轻量级、可扩展的查询级路由，在 8 个推理基准上优于 SOTA 路由方法，且仅增加约 7ms 延迟。
tags:
  - ICLR 2026
  - 推理语言模型
  - 模型路由
  - 项目反应理论
  - 多目标优化
  - 自适应推理
---

# RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs

**会议**: ICLR 2026  
**arXiv**: [2509.25426](https://arxiv.org/abs/2509.25426)  
**代码**: 无  
**领域**: 其他  
**关键词**: 推理语言模型, 模型路由, 项目反应理论, 多目标优化, 自适应推理

## 一句话总结

本文提出 Radar 框架，将推理语言模型（RLM）的自适应推理问题建模为多目标优化，利用项目反应理论（IRT）联合估计可解释的查询难度和模型配置能力参数，实现轻量级、可扩展的查询级路由，在 8 个推理基准上优于 SOTA 路由方法，且仅增加约 7ms 延迟。

## 研究背景与动机

近年来推理语言模型（RLMs）如 DeepSeek-R1、o4-mini、Qwen3 等展示了在数学、科学和编程等挑战性任务上的卓越能力。选择合适的 RLM 涉及**性能-成本权衡**的两个关键层面：(1) 模型大小——更大的模型性能更好但成本更高；(2) 推理预算——更多的思考 token 提升性能但增加延迟和费用。

关键发现：MATH-500 上超过 50% 的查询可以用 Qwen3-0.6B 以极少推理预算正确解答，而一些困难查询则需要更强的 RLM 配置。更强的 RLM 还可能在简单问题上"过度思考"（overthinking）反而降低性能。这激发了一个核心问题：如何为每个查询选择恰好"足够强"的 RLM 配置，从而在不牺牲性能的前提下最大化成本效益？

## 方法详解

### 整体框架

Radar 框架包含以下核心组件：
1. **离散化技巧**：将每个 RLM 按可用推理预算离散化为多个配置 $g = (m, u) \in \mathcal{G}$
2. **多目标优化（MOO）**：将路由形式化为最大化性能、最小化成本的双目标优化
3. **IRT 校准**：使用 2PL IRT 模型联合估计查询难度和配置能力
4. **自适应测试**：通过少量动态选择的查询快速估计新模型配置的能力

### 关键设计

1. **离散化与配置路由**: 将每个 RLM $m \in \mathcal{M}$ 按推理预算 $u \in \mathcal{U}_m$ 离散化为配置 $g = (m, u)$。对开源 RLM，通过计数思考 token 并在超出预算时附加中断消息来强制执行预算。设计动机：统一模型选择和预算选择为单一配置路由问题，使得可以在配置空间上进行优化。

2. **多目标优化路由**: 对每个查询 $q$，求解 $g^* = \arg\max_{g \in \mathcal{G}} f(p_q(g), c_q(g))$，其中 $p_q(g)$ 是性能预测函数，$c_q(g)$ 是成本预测函数。本文探索两种标量化技术：

    - **线性标量化**: $\text{LSP}_q^{w_1} = \arg\max_{g \in \mathcal{G}} w_1 p_q(g) - (1-w_1) c_q(g)$
    - **切比雪夫标量化**: $\text{CSP}_q^{w_1} = \arg\min_{g \in \mathcal{G}} \max\{w_1|1-p_q(g)|, (1-w_1)c_q(g)\}$
   设计动机：切比雪夫标量化可以发现 Pareto 前沿的非凸部分，而线性标量化仅能覆盖凸部分。这是首次在 LLM 路由中引入线性标量化之外的 MOO 技术。

3. **2PL IRT 模型**: 使用二参数逻辑斯蒂模型参数化性能预测函数。为实现 OOD 泛化，将查询难度 $b_j = \mathbf{w}_b^\top \mathbf{e}_j$ 和区分度 $a_j = \mathbf{w}_a^\top \mathbf{e}_j$ 参数化为查询嵌入 $\mathbf{e}_j$ 的线性变换，每个配置 $g_i$ 有标量能力参数 $\theta_i$。正确回答概率为 $p_{ij} = \sigma(a_j(\theta_i - b_j))$。设计动机：标量能力值可捕获模型配置间的可解释排序，参数量少于多维 IRT（MIRT），且通过嵌入泛化到未见查询。

4. **自适应测试扩展**: 为新模型配置估计能力参数时，迭代选择 Fisher 信息最大的查询进行评估：$j_t = \arg\max_{j \in \mathcal{Q} \setminus \mathcal{S}_{t-1}} I(\hat{\theta}_{t-1}, a_j, b_j)$，其中 $I(\theta, a_j, b_j) = a_j^2 \sigma(a_j(\theta-b_j))[1-\sigma(a_j(\theta-b_j))]$。设计动机：仅需评估约 12% 的训练集即可准确估计新配置能力，实现即插即用。

### 损失函数 / 训练策略

训练 IRT 模型使用二元交叉熵损失：
$$\mathcal{L}_{2PL} = -\frac{1}{nk} \sum_{i=1}^n \sum_{j=1}^k [y_{ij} \log p_{ij} + (1-y_{ij}) \log(1-p_{ij})]$$

其中 $y_{ij} \in \{0,1\}$ 表示配置 $g_i$ 在查询 $q_j$ 上是否正确。总共收集了 175 万条二值响应数据，覆盖 35 个配置和 50,139 个查询。

## 实验关键数据

### 主实验（ID 设置，Hypervolume 指标，越高越好）

| 基准数据集 | Random-Pair | RouterBench | IRT-Router | Radar (本文) | 改进 |
|-----------|-------------|-------------|------------|-------------|------|
| GPQA-Diamond | 0.5545 | 0.6866 | 0.6942 | **0.7513** | +8% vs 次优 |
| MMLU | 0.6905 | 0.8592 | 0.8604 | **0.8720** | +1.3% |
| MMLU-Redux | 0.7281 | 0.9053 | 0.9117 | **0.9230** | +1.2% |
| LSAT | 0.6913 | 0.9125 | 0.9163 | **0.9188** | +0.3% |
| FRAMES | 0.6589 | 0.8325 | 0.8501 | **0.8762** | +3.1% |

### 消融实验

| 配置 | Hypervolume | 说明 |
|------|-----------|------|
| 线性标量化 (ID) | 略优 | ID 场景下边际领先 |
| 切比雪夫标量化 (OOD) | 更优 | OOD 场景下优势明显 |
| 20% 训练数据 | ~相当 | 仅用 20% 数据即可达到相似性能 |
| Radar (35 配置) | 基线 | 原始 35 个配置 |
| Radar++ (43 配置) | 提升 | 通过自适应测试加入 Qwen3-14B 后提升 |

### 关键发现

- 在 MATH-500 上，Radar 可以仅用 o4-mini（高预算）1.31% 的成本达到其 90% 的性能
- 在 FRAMES（长文本多文档 QA）上，Radar 以 10% 的成本达到 90% 性能，次优方法需要 30% 成本
- Radar 的路由延迟仅约 7ms，相比最小 RLM 配置约 870ms 的生成时间可忽略不计
- 自适应测试仅需 12% 的训练集（5k 查询）即可准确估计新配置能力
- 估计的查询难度与 MATH-500 的 5 级人工标注难度呈中等 Pearson 相关（0.509）

## 亮点与洞察

- **首次将 MOO（超越线性标量化）引入 LLM 路由**：切比雪夫标量化能发现 Pareto 前沿的非凸部分
- **心理测量学启发的 IRT 模型**：将查询类比为考试题目、模型配置类比为考生，自然且可解释
- **极端成本节约**：在 MATH-500 上 1.31% 成本达到 90% 性能的结果令人印象深刻
- **即插即用设计**：无需微调 RLM，黑盒使用，新模型快速接入
- **强 OOD 泛化**：在长文本多文档 QA 上的泛化能力尤为突出

## 局限与展望

- 成本预测使用简单启发式（平均 token 数 × 单价），未考虑查询特异的成本差异
- 在 AIME 等高难度 OOD 基准上泛化能力稍弱，倾向于分配能力偏低的配置
- 仅处理文本模态，多模态推理场景有待扩展
- 2PL IRT 的线性参数化可能不足以捕获复杂的难度-能力交互关系
- 未考虑批量查询下的总预算约束场景

## 相关工作与启发

- IRT-Router（Song et al., 2025）：使用多维 IRT (MIRT)，参数更多但能力非标量；Radar 用标量能力值实现可解释排序
- RouterBench（Hu et al., 2024）：传统模型路由，本文扩展到 RLM 配置级路由
- L1/S1 等高效推理方法：与 Radar 互补，可作为额外配置加入路由池
- 教育测评领域的自适应测试：Fisher 信息选题策略的成功借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ IRT + MOO 的组合新颖，但单个组件（IRT、路由）不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个基准、35 个配置、175 万条数据，全面且严谨
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，公式推导完整，图表直观
- 价值: ⭐⭐⭐⭐⭐ 直接面向 RLM 实际部署的核心问题，节约成本效果显著

<!-- RELATED:START -->

## 相关论文

- [The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_advanced_jai.md)
- [ActivationReasoning: Logical Reasoning in Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_driven_reasoning.md)

<!-- RELATED:END -->
