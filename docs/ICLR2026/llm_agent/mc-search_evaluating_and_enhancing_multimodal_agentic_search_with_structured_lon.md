---
title: >-
  [论文解读] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
description: >-
  [ICLR 2026][LLM Agent][多模态RAG] 提出 MC-Search，首个面向 agentic 多模态 RAG 的 benchmark，包含 3,333 个高质量样本（平均 3.7 跳），覆盖 5 种推理拓扑结构，通过 HAVE 验证确保每步必要性，并引入 Search-Align 过程监督微调框架使开源模型的检索规划能力大幅提升（Qwen2.5-VL-7B F1 提升 +13.7）。
tags:
  - ICLR 2026
  - LLM Agent
  - 多模态RAG
  - Agentic Search
  - 多跳推理
  - 过程级评估
  - 检索增强推理
---

# MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains

**会议**: ICLR 2026  
**arXiv**: [2603.00873](https://arxiv.org/abs/2603.00873)  
**代码**: https://mc-search-project.github.io (有)  
**领域**: LLM Agent  
**关键词**: 多模态RAG, Agentic Search, 多跳推理, 过程级评估, 检索增强推理

## 一句话总结
提出 MC-Search，首个面向 agentic 多模态 RAG 的 benchmark，包含 3,333 个高质量样本（平均 3.7 跳），覆盖 5 种推理拓扑结构，通过 HAVE 验证确保每步必要性，并引入 Search-Align 过程监督微调框架使开源模型的检索规划能力大幅提升（Qwen2.5-VL-7B F1 提升 +13.7）。

## 研究背景与动机
**领域现状**：多模态大语言模型（MLLM）正从固定的"检索-生成"范式向更复杂的 agentic 多模态检索增强生成（MM-RAG）演进。模型需要迭代分解查询、自适应跨模态检索、整合多模态证据。

**现有痛点**：现有 MM-RAG benchmark 存在三个关键局限——(a) 大多采用简单 QA 格式，将多模态证据压缩为纯文本通道（如 MRAG）；(b) 仅评估 1-2 跳的浅层检索，缺乏长推理链（如 Dyn-VQA）；(c) 缺少逐步标注和显式推理拓扑，无法分析不同模态在推理中的角色。

**核心矛盾**：实际查询通常是模糊且复杂的，需要多步、跨模态、知识密集的推理。但没有合适的 benchmark 来评估 MLLM 是否真正能进行长链、结构化的多模态搜索推理。

**本文目标**：(a) 构建首个支持长推理链（≥4跳）的多模态 agentic RAG benchmark；(b) 提供逐步标注和多种推理拓扑；(c) 设计过程级评估指标；(d) 利用验证过的推理链改善开源模型。

**切入角度**：从 Wikipedia 知识库出发构建多模态知识集群，设计 5 种有代表性的推理拓扑结构（串行/并行、图像启动/文本启动/多图分叉等），通过 HAVE 过滤确保每个推理步骤既必要又非冗余。

**核心idea**：长链多跳 + 5种推理拓扑 + HAVE验证 + 过程级指标 + Search-Align 微调 = 全面评估和提升 agentic MM-RAG。

## 方法详解

### 整体框架
MC-Search 包含两大部分：(1) Benchmark 构建——从 Wikipedia 构建多模态知识库，生成覆盖5种推理拓扑的多跳 QA，经 HAVE 过滤和质量验证得到 3,333 个高质量样本；(2) 评估与训练——设计统一的 agentic MM-RAG pipeline 和过程级指标进行公平评估，并通过 Search-Align 利用验证链微调开源模型。

### 关键设计

1. **5种搜索增强推理拓扑**:

    - 功能：定义 5 种代表性的多跳推理图结构，每种结构的推理链形式化为 $\mathcal{G}(Q,A) = \{(q_t, m_t, r_t, a_t)\}_{t=1}^{T}$，其中 $q_t$ 是子问题，$m_t$ 是检索模态，$r_t$ 是证据，$a_t$ 是中间答案
    - 5种结构：(i) Image-Initiated Chain（图像启动+后续文本检索）；(ii) Text-Initiated Chain（文本启动+后续图像验证）；(iii) Parallel Image-Text Fork（图文并行检索，无跨步依赖）；(iv) Multi-Images Fork（多图视觉比较+文本支持）；(v) Text-Only Chain（纯文本基线）
    - 设计动机：捕捉现实世界中的串行/并行推理模式和不同模态组合，使评估更全面

2. **HAVE（Hop-wise Attribution and Verification of Evidence）**:

    - 功能：过滤推理链中的幻觉步骤和冗余步骤
    - 核心思路：对每个步骤计算上下文效用 $\text{Util}(t) = \text{F1}(\mathcal{C}) - \text{F1}(\mathcal{C} \setminus r_t)$——移除该步证据后答案准确率的下降。同时检查导航角色：$\text{Nav}(t)=1$ 如果该步中间答案的实体出现在下游子问题中。若 Util 低于阈值且 Nav=0，则该步为冗余
    - 设计动机：LLM 生成的长推理链常含虚构步骤（看似合理但无证据支持）或多余步骤（对答案无贡献）。HAVE 的双重检查（直接效用 + 导航角色）确保保留的每一步都是不可或缺的

3. **过程级评估指标**:

    - 功能：超越答案准确率，评估推理过程质量
    - 核心思路：(i) **Hit per Step (HPS)**——金标推理步被预测图成功覆盖的比例；(ii) **Rollout Deviation (RD)**——预测链和金标链的步数差，$\text{RD} = ||{\hat{\mathcal{G}}}| - |{\mathcal{G}}||$，反映过度/不足检索程度；(iii) **LLM-as-a-Judge (LJ)**——从答案准确、推理连贯、实体覆盖、步骤对齐四个维度评分
    - 设计动机：仅看最终答案无法诊断检索规划或模态选择的问题

4. **Agentic MM-RAG Pipeline**:

    - 功能：统一的迭代搜索推理管线，支持公平评估
    - 核心思路：每轮迭代：(a) 生成子查询和检索动作（文本搜索/图像搜索/图像查图）；(b) 从多模态知识库检索 top-1 证据；(c) 生成子答案并判断是否继续搜索。全程记录模态和证据，支持链级评估
    - 设计动机：现有工作各用不同 pipeline，缺乏公平对比基础

5. **Search-Align 过程监督微调**:

    - 功能：利用 HAVE 验证过的推理链对开源 MLLM 进行 SFT
    - 核心思路：将推理图转化为对话形式（assistant 提子问题+推理，user 执行检索+返回结果），用 Gemini-2.5-Flash 为每步生成推理思路（reasoning thoughts），连接相邻跳。然后在这种对话式 trace 上做 supervised fine-tuning
    - 设计动机：传统 SFT 只监督最终答案，Search-Align 提供步级监督信号，教会模型如何规划、选择检索模态、整合跨步证据

### 损失函数 / 训练策略
Search-Align 使用标准的 next-token prediction loss 在对话式推理 trace 上微调。训练数据来自 HAVE 验证后的 3,333 条推理链。

## 实验关键数据

### 主实验（Image-Initiated Chain 拓扑为例）

| 模型 | F1(↑) | ΔF1(↑) | LJ(↑) | HPS(↑) | RD(↓) | Golden F1 |
|------|-------|--------|--------|---------|-------|-----------|
| GPT-4o-Mini | 36.49 | 34.18 | 2.63 | 27.51 | 1.46 | 68.29 |
| Gemini-2.5-Flash | 44.10 | 37.38 | 3.01 | 31.46 | 2.91 | 72.39 |
| Gemini-2.5-Pro | **47.61** | **42.76** | **3.18** | 25.90 | 1.05 | 69.83 |
| Claude-3.7-Sonnet | 37.80 | 33.09 | 2.60 | 27.31 | 1.18 | 72.62 |
| InternVL3.5-8B | 39.11 | 29.49 | 2.27 | 22.59 | 1.58 | - |
| + Search-Align | 42.27 | 32.65 | 2.53 | **32.49** | **0.94** | 63.86 |
| Qwen2.5-VL-7B | 26.30 | 8.65 | 1.34 | 16.51 | 4.04 | - |
| + Search-Align | 45.70 | 28.05 | 2.23 | **33.59** | **0.70** | 60.95 |

### 消融实验（模态覆盖分析）

| 查询类型 | 模态 | Gemini-2.5-Pro 覆盖率 | InternVL-3.5-8B 覆盖率 |
|---------|------|---------------------|----------------------|
| 含图查询 | Image | 87.35% | 63.84% |
| 含图查询 | Text | 78.61% | 82.67% |
| 无图查询 | Image | **29.50%** | **0.66%** |
| 无图查询 | Text | 83.55% | 89.78% |

### 关键发现
- **Search-Align 效果显著**：Qwen2.5-VL-7B 经微调后 F1 平均提升 +13.7，HPS 提升 +16.0，RD 降低 3.1，几乎追平 Gemini-2.5-Pro
- **Parallel Image-Text Fork 最难**：需要同时覆盖文本和图像两个分支，所有模型在此拓扑上 F1 和 HPS 最低
- **严重的模态偏差**：当查询中无显式图像线索时，InternVL 的图像检索覆盖率从 63.84% 暴跌至 0.66%，说明模型默认偏向文本检索
- **链越长越难**：4-5 跳的推理链上所有模型性能急剧下降，复合检索错误和不稳定规划是主因
- **适度过度检索有益**：多检索 1-2 步（ΔStep=1~2）通常能提高准确率，但过度检索 ≥4 步会引入噪音导致性能骤降
- **主要瓶颈在检索规划**：错误分析显示 Retrieval-Failure（84.7%）、Hallucinated Entity（75.8%）和 Step-Omission（74.3%）是最常见错误类型

## 亮点与洞察
- **5种推理拓扑的设计非常系统**：不是随意组合多跳问题，而是从实际 MM-RAG 需求出发定义了串行/并行×图像/文本的完整组合空间，为后续研究提供了清晰的分析框架
- **HAVE 过滤机制巧妙**：用"移除某步后答案准确率下降"来验证必要性，用"中间答案实体是否出现在下游子问题"来捕捉导航性步骤，双重标准避免了既不过滤也不误删的平衡问题
- **过程级指标填补空白**：HPS 和 RD 可以精确定位模型是"检索不够"还是"检索过多"，对调试 agentic RAG 系统非常实用
- **模态偏差的发现很有启发**：无图线索时图像检索几乎为零，说明模型还远未具备"根据问题需要主动选择模态"的能力

## 局限与展望
- 知识库基于 Wikipedia，领域覆盖有限（未涉及科学、数学等专业领域）
- 数据生成依赖 Gemini-2.5-Flash，引入了模型特定偏差
- 评估仅用 6 个 MLLM，未包含更强的推理模型（如 GPT-5 系列、Gemini-2.5-Pro with thinking）
- Search-Align 仅使用 SFT，未探索 RL 或 DPO 等强化学习方法
- top-1 检索约束可能过于严格，实际应用中通常检索多条结果

## 相关工作与启发
- **vs MMSearch**：MMSearch 仅 1 跳，关注搜索引擎的图文混合结果。MC-Search 关注长链多跳，强调推理结构和过程评估
- **vs WebQA**：WebQA ≤2 跳且缺乏逐步标注。MC-Search 平均 3.7 跳并提供完整的推理图标注
- **vs Agentic RAG 系统（如 ReAct-style）**：这些系统大多仅用于纯文本场景。MC-Search 将 agentic RAG 扩展到多模态，并首次系统评估了模态规划能力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个长链多模态 agentic RAG benchmark，5种推理拓扑+HAVE验证+过程级指标，系统性很强
- 实验充分度: ⭐⭐⭐⭐ 6个MLLM + 多维度分析（链长/过检索/模态偏差/错误类型），但模型覆盖可以更广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，形式化完整，图表丰富，但内容密度大导致部分细节需要多次阅读
- 价值: ⭐⭐⭐⭐⭐ 为多模态 agentic search 领域提供了急需的评估基础设施和训练方法，Search-Align 的效果也验证了数据的训练价值

<!-- RELATED:START -->

## 相关论文

- [LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_fresh_news.md)
- [ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)

<!-- RELATED:END -->
