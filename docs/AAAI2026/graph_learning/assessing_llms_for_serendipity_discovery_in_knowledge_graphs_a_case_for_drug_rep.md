---
title: >-
  [论文解读] Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing
description: >-
  [AAAI 2026][图学习][Serendipity] 提出 SerenQA 框架，首次形式化定义知识图谱问答中的"意外发现"(serendipity)任务，包含基于信息论的 RNS 度量、专家标注的药物重定位基准数据集和三阶段评估流水线，揭示当前 LLM 在检索任务上表现尚可但在意外发现探索上仍有巨大改进空间。
tags:
  - AAAI 2026
  - 图学习
  - Serendipity
  - Knowledge Graph Question Answering
  - LLM Evaluation
  - Drug Repurposing
  - information theory
---

# Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing

**会议**: AAAI 2026  
**arXiv**: [2511.12472](https://arxiv.org/abs/2511.12472)  
**代码**: [cwru-db-group/serenQA](https://cwru-db-group.github.io/serenQA)  
**领域**: graph_learning  
**关键词**: Serendipity, Knowledge Graph Question Answering, LLM Evaluation, Drug Repurposing, information theory

## 一句话总结

提出 SerenQA 框架，首次形式化定义知识图谱问答中的"意外发现"(serendipity)任务，包含基于信息论的 RNS 度量、专家标注的药物重定位基准数据集和三阶段评估流水线，揭示当前 LLM 在检索任务上表现尚可但在意外发现探索上仍有巨大改进空间。

## 研究背景与动机

- **LLM+KGQA 的局限**：现有 LLM 增强的知识图谱问答系统专注于返回高相关但"可预测"的答案，缺乏发现意外但有价值连接的能力
- **Serendipity 的科学意义**：科学史上许多重大突破源于意外发现（如青霉素），赋予 LLM 从已有知识库中挖掘惊喜发现的能力是迈向真正 AI 驱动科学发现的关键一步
- **药物重定位的需求**：药物重定位（发现已有药物的新适应症）是医学研究的核心任务，天然适合作为 serendipity 的应用场景——例如 Journavx 作为首个非阿片类重度急性疼痛药物，其通过全新机制（NaV1.8 钠通道）实现镇痛，正是 serendipitous 发现的典型案例
- **度量缺失**：现有 serendipity 研究（推荐系统、网页搜索）主要依赖主观人工标注或 LLM 自评，缺乏可解释、可扩展、可复现的量化方法
- **评估空白**：社区尚无专门针对科学 KGQA 中 serendipitous 发现能力的基准数据集和系统评估方案
- **多维度挑战**：serendipity 本身是相关性、新颖性和意外性的复合体验，如何在保持与查询相关的同时发现真正新颖且令人惊讶的答案，是一个兼具理论和实践难度的问题

## 方法详解

### 整体框架

SerenQA 由三个核心组件构成：(1) RNS 度量——基于图的信息论 serendipity 量化指标；(2) Serendipity-aware Benchmark——基于 Clinical Knowledge Graph 的专家标注药物重定位数据集（1529 条查询，15M+ 实体，201M+ 关系）；(3) Assessment Pipeline——包含知识检索、子图推理和 serendipity 探索的三阶段 LLM 评估流水线。对于给定查询 Q，系统返回有序分区 $\mathcal{A} = (\mathcal{A}_e, \mathcal{A}_s)$，其中 $\mathcal{A}_e$ 是图中可直接推导的已知答案集，$\mathcal{A}_s$ 是超越直接知识的意外发现集。

### 关键设计一：RNS Serendipity 度量

- **功能**：量化答案集 $\mathcal{A}_s$ 相对于 $\mathcal{A}_e$ 的 serendipity 程度
- **核心思路**：将 serendipity 分解为三个信息论维度的加权组合 $\text{RNS}(\mathcal{A}_e, \mathcal{A}_s) = \alpha R + \beta N + \gamma S$。其中 Relevance $R$ 基于 GCN 嵌入的归一化欧氏距离衡量上下文相似性；Novelty $N = 1 - MI(\mathcal{A}_e, \mathcal{A}_s)$ 通过互信息衡量 $\mathcal{A}_s$ 相对于 $\mathcal{A}_e$ 的信息增量；Surprise $S$ 通过 Jensen-Shannon 散度衡量实体分布的不可预测性
- **设计动机**：相比依赖人工标注或 LLM 自评的主观方法，基于图概率模型的信息论方法具有可解释性、可扩展性和可复现性。作者通过公理化分析证明 RNS 满足 scale invariance、consistency、non-monotonicity 和 independence 四项性质

### 关键设计二：3-Hop 图概率建模

- **功能**：为 RNS 计算建立高效的概率模型
- **核心思路**：构建 3-hop 条件概率矩阵 $P_k = \sum_{h=1}^k \alpha_h P_1^h$，其中 $P_1$ 是归一化的单跳转移概率矩阵，权重 $\alpha_h$ 随跳数增加（优先考虑更远的连接）。边际概率通过 PageRank 风格的阻尼迭代近似计算，将复杂度从 $O(V^3)$ 降至 $O(V^2 \log V)$
- **设计动机**：实证发现 99% 的 serendipitous 答案可在 3 跳内从已知答案到达，因此 3-hop 是充分的；概率矩阵"一次计算、多次复用"，可适配不同领域图

### 关键设计三：基准数据集构建与三种分区策略

- **功能**：基于 Clinical Knowledge Graph 构建带 serendipity 标注的 ground-truth 数据集
- **核心思路**：对每条查询的完整候选答案集 $\mathcal{A}_c$，采用三种互补策略进行 $(\mathcal{A}_e, \mathcal{A}_s)$ 分区——(1) LLM Ensemble：4 个 SOTA LLM 打分后取 top-20% 为 $\mathcal{A}_s$；(2) Expert Crowdsourced：6 位领域专家（3 位医生 + 1 位药学家 + 2 位标注员）精炼排序；(3) RNS Guided：通过贪心交换算法（Algorithm 1）优化 RNS 分数。三种策略的 Pearson 相关性 > 85%，专家与 RNS 引导分区相关性达 ~99%
- **设计动机**：三种策略互相校验，确保评估稳健性；通过从 $\mathcal{G}_c$ 中删除选定边构造评估图 $\mathcal{G}$，使 $\mathcal{A}_e$ 可推导而 $\mathcal{A}_s$ 不可达，模拟真实发现场景

### 关键设计四：三阶段 LLM 评估流水线

- **功能**：系统评估 LLM 在 serendipity 发现各环节的能力
- **核心思路**：(T1) Knowledge Retrieval——LLM 将自然语言查询转为 Cypher 查询并从 KG 检索 $\mathcal{A}_e$；(T2) Subgraph Reasoning——LLM 将检索到的子图结构化信息总结为领域感知的自然语言；(T3) Serendipity Exploration——LLM 通过 beam search（宽度 30，深度 3）从 $\mathcal{A}_e$ 出发探索 $\mathcal{A}_s$，在每步根据证据强度、交互力、生物效应方向和表达水平选择 top-w 节点
- **设计动机**：三个任务分别评估 LLM 的"基石"能力：精确知识检索、结构化推理和创造性探索，形成完整的能力画像

## 损失函数与训练

本文是评估框架而非训练方法，不涉及新模型训练。RNS 度量的权重 $\alpha, \beta, \gamma$ 通过与专家标注分区对齐来校准。系统部署在 5 台 AWS c6a.24xlarge 实例上进行分布式计算，支持 500 并发 LLM 推理任务。

## 实验

### Table 2: Knowledge Retrieval (T1) 各模型在不同查询模式下的表现

| 模型 | One-Hop F1(%) | Two-Hop F1(%) | 3+-Hop F1(%) | Intersection F1(%) |
|------|:---:|:---:|:---:|:---:|
| DeepSeek-V3 | **78.71** | 10.71 | 6.22 | 7.15 |
| GPT-4o | 77.16 | 6.36 | 4.20 | 4.65 |
| Llama-3.3-70B | 70.67 | **44.34** | **10.16** | **9.60** |
| DeepSeek-R1-70B | 69.07 | 37.00 | 8.06 | 6.16 |
| Med42-V2-70B | 69.43 | 19.12 | 0.51 | 0.13 |
| Qwen3-8B | 37.24 | 2.87 | 2.01 | 1.91 |

**关键发现**：前沿大模型在简单单跳查询上 F1 ~78%，但多跳查询（3+ hops）性能急剧下降至 <10%，暴露推理深度不足。70B 模型（Llama、DeepSeek-R1）在多跳查询上明显优于闭源前沿模型。

### Table 3: Serendipity Exploration (T3) Expert Crowdsourced 分区结果

| 模型 | Relevance | TypeMatch | SerenHit |
|------|:---:|:---:|:---:|
| Llama-3.3-70B | **2.559** | **0.483** | 0.067 |
| DeepSeek-V3 | 2.494 | 0.462 | 0.061 |
| Gamma-2-27B | 2.379 | 0.414 | 0.057 |
| Qwen-2.5-72B | 2.345 | 0.406 | 0.041 |
| Qwen-2.5-32B | 2.331 | 0.426 | 0.045 |
| DeepSeek-R1-70B | 2.000 | 0.409 | 0.034 |
| Mixtral-8x7B | 2.033 | 0.254 | 0.015 |

**关键发现**：所有模型的 SerenHit（与 ground-truth serendipity 集的匹配率）均极低（<10%），表明当前 LLM 在真正的意外发现探索上存在巨大差距。去除子图摘要（w.o. summary）反而提升了多数模型的性能，暗示摘要过程中的幻觉可能误导探索路径。

## 亮点

- **问题定义新颖**：首次形式化定义科学 KGQA 中的 serendipity-aware 任务，填补该领域评估空白
- **度量理论扎实**：RNS 度量基于信息论并经公理化分析验证，比主观评估方法更严谨、可复现
- **数据集构建严谨**：三种互补分区策略（LLM 集成、专家标注、RNS 引导）互相校验，相关性 >85%
- **发现有洞察力**：揭示了子图推理中 faithfulness 与 serendipity coverage 的 trade-off，以及摘要幻觉对探索的负面影响

## 局限性

- **领域局限**：仅在药物重定位（Clinical Knowledge Graph）上验证，未证明框架对其他科学领域（如材料科学、物理学）的泛化性
- **RNS 度量依赖嵌入质量**：Relevance 维度依赖 GCN 嵌入的质量，嵌入方法的选择可能影响度量结果
- **3-hop 假设的局限**：虽然 99% 的 serendipitous 答案在 3 跳内可达，但真正突破性的发现可能需要更远的路径
- **未考虑药理学因素**：如作者所述，框架未考虑药物可行性的关键因素（如物化性质），发现结果需临床验证
- **评估成本高**：需要 5 台大型 AWS 实例和 500 并发推理任务，可复现性受计算资源限制

## 相关工作

- **Serendipity 推荐系统**：先前工作（Bordino et al., Fu et al.）主要在推荐系统和网页搜索中研究 serendipity，依赖主观标注——本文转向科学 KGQA 并提出客观度量
- **LLM + KGQA**：GraphLingo 等方法通过 RAG 和 prompt engineering 提升 KGQA 精度，但仅关注"正确"答案——本文关注"意外但有价值"的答案
- **LLM 科学发现**：AI4Science、Si et al. 等工作探索 LLM 的科学假设生成能力——本文提供首个系统评估框架
- **药物重定位**：传统方法基于相似性网络或知识图谱推理——本文将其重新定位为 serendipity 发现问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次形式化科学 KGQA 中的 serendipity 问题，问题定义和度量设计均有原创贡献
- 实验充分度: ⭐⭐⭐⭐ — 涵盖 12+ 模型、3 种分区策略、3 个评估任务，分析详尽；但仅限单一领域
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、公理化分析严谨、示例直观；部分符号较繁重
- 价值: ⭐⭐⭐⭐ — 开辟了 serendipity-aware KGQA 的新研究方向，对 AI4Science 社区有重要启示

<!-- RELATED:START -->

## 相关论文

- [Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [Deliberation on Priors: Trustworthy Reasoning of LLMs on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)

<!-- RELATED:END -->
