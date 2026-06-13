---
title: >-
  [论文解读] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering
description: >-
  [ACL 2025][图学习][知识图谱问答] 提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。 领域现状：LLM 在复杂推理任务中表现出色…
tags:
  - "ACL 2025"
  - "图学习"
  - "知识图谱问答"
  - "LLM推理"
  - "束搜索"
  - "演绎验证"
  - "Path-RAG"
---

# FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering

**会议**: ACL 2025  
**arXiv**: [2405.13873](https://arxiv.org/abs/2405.13873)  
**代码**: [https://github.com/Y-Sui/FiDeLiS](https://github.com/Y-Sui/FiDeLiS)  
**领域**: 图学习  
**关键词**: 知识图谱问答, LLM推理, 束搜索, 演绎验证, Path-RAG

## 一句话总结

提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。

## 研究背景与动机

**领域现状**：LLM 在复杂推理任务中表现出色，但容易产生幻觉或事实不一致的输出。利用知识图谱（KG）作为外部知识源是缓解这一问题的可行方案，KG 的结构化格式支持显式、可追溯的推理。

**现有痛点**：
   - **检索方法**（如 RoG）：生成的推理步骤仅 67% 有效，33% 包含格式错误或引用不存在的 KG 事实
   - **Agent方法**（如 ToG）：虽然能增强推理准确性，但计算成本高、延迟大、可扩展性差
   - 核心困境：如何在忠实性（faithfulness）和效率（efficiency）之间取得平衡？

**核心矛盾**：检索方法快但不够准确（无法保证推理路径有效），Agent 方法准确但太慢（需要多轮 LLM-KG 交互）。

**本文目标**：设计一个统一框架，既能保证推理路径的每一步都可验证（忠实性），又能通过缩小搜索空间降低计算成本（效率）。

**切入角度**：结合检索和探索的优势——Path-RAG 负责高效预选候选集（检索的优势），DVBS 负责逐步验证推理路径（Agent 的优势）。

**核心 idea**：用 RAG 缩小搜索空间 + 用演绎验证确保每步推理合法，在不训练的前提下实现忠实且高效的 KG 推理。

## 方法详解

### 整体框架

FiDeLiS 由两个核心组件组成：
1. **Path-RAG**：预选小规模候选集，缩小 KG 探索的搜索空间
2. **DVBS（Deductive-Verification Beam Search）**：逐步构建推理路径，每一步通过演绎验证保证逻辑一致性

数学形式化：
$$P(a|q,\mathcal{G}) = \sum_{\mathcal{P}} P_\theta(a|q,\mathcal{P}) \prod_{k=1}^n P_\phi(t_k|q,t_{<k},\mathcal{G})$$

### 关键设计

#### 1. Path-RAG：推理路径检索增强生成

- **功能**：为 beam search 的每一步预选高质量的候选推理步骤
- **三步流程**：
    - **初始化**：用预训练语言模型将 KG 中所有实体 $e_i$ 和关系 $r_i$ 编码为稠密向量 $z(e_i) = \text{LM}(e_i) \in \mathbb{R}^d$，存入最近邻索引
    - **关键词驱动检索**：LLM 从用户问题中提取关键词，编码后检索 top-$m$ 相似的实体和关系
    - **候选推理步骤构建**：定义评分函数结合语义相似度和图连通性
- **评分函数**（核心公式）：
  $$S((r,e)) = S_0((r,e)) + \alpha \max_{\forall(r_j,e_j) \in N(e)} S_0((r_j,e_j))$$
  其中 $S_0((r,e)) = S_{\text{rel}}(r) + S_{\text{ent}}(e)$ 是基础语义得分，第二项考虑下一跳候选的最大得分
- **$\alpha$ 的作用**：平衡即时语义相关性和未来连通潜力，较高 $\alpha$ 偏好长远收益
- **设计动机**：避免 Agent 方法的多轮 LLM-KG 交互，以一次检索获取高质量候选

#### 2. DVBS：演绎验证 Beam Search

- **功能**：逐步构建推理路径，每步验证逻辑一致性，并在问题可推导时终止
- **三步流程**：
    - **计划生成**：LLM 先生成回答问题的规划步骤 $w$，作为后续决策的额外提示
    - **Beam Search**：在时间步 $t$，LLM 从候选集 $\mathcal{S}^t$ 中选择推理步骤：
    $$\mathcal{H}_t = \text{Top}_k \{h \oplus \text{LM}(s^t | q, s^{1:t-1}, w, \mathcal{S}^t) : h \in \mathcal{H}_{t-1}\}$$
    - **演绎验证**：两层验证确保推理质量
- **两层验证**：
    - **全局验证** $C_{\text{global}}$：检查 $(s^t \land s^{1:t-1}) \models q'$，即当前推理路径是否足以推导出问题的答案
    - **局部验证** $C_{\text{local}}$：检查 $s^t$ 是否逻辑上跟从 $s^{1:t-1}$，确保步间一致性
- **终止条件**：当全局和局部验证都通过时终止搜索
- **设计动机**：解决过早终止和过度延伸的问题，确保推理路径完整且有效

### 损失函数/训练策略

**FiDeLiS 是无需训练的框架（training-free）**，完全依赖 LLM 的零/少样本能力和 KG 结构。默认使用 5 个示例。

## 实验关键数据

### 主实验

三个基准数据集上的对比（使用 gpt-3.5-turbo / gpt-4-turbo）：

| 方法 | 后端 | WebQSP Hits@1 | CWQ Hits@1 | CR-LT Acc |
|------|------|--------------|------------|-----------|
| Zero-shot CoT | gpt-3.5 | 57.42 | 43.21 | 37.42 |
| RoG (Finetuning) | - | 83.15 | 61.39 | 60.32 |
| ToG | gpt-3.5 | 75.13 | 57.59 | 62.48 |
| **FiDeLiS** | gpt-3.5 | **79.32** | **63.12** | **67.34** |
| ToG | gpt-4 | 81.84 | 68.51 | 67.24 |
| **FiDeLiS** | gpt-4 | **84.39** | **71.47** | **72.12** |

FiDeLiS (gpt-4) 在所有数据集上均达到最佳，且作为无训练方法甚至优于需要微调的 RoG 和 DeCAF。

### 消融实验

各组件消融（gpt-3.5-turbo）：

| 消融设置 | WebQSP | CWQ | CR-LT |
|---------|--------|-----|-------|
| 完整 FiDeLiS | 79.32 | 63.12 | 67.34 |
| 用 vanilla retriever 替代 Path-RAG | 72.35 | 57.11 | 59.78 |
| 用 ToG 替代 Path-RAG | 75.11 | 59.47 | 63.47 |
| 去掉 beam search | 60.35 | 49.78 | 61.87 |
| 去掉演绎验证 | 74.13 | 57.23 | 63.89 |
| 去掉计划 | 76.23 | 60.14 | 64.13 |

### 关键发现

1. **Beam search 是最关键组件**：移除后 WebQSP 从 79.32% 暴跌至 60.35%（-18.97%）
2. **Path-RAG 优于所有替代方案**：比 vanilla retriever 高 ~7%，比 ToG 作为检索器高 ~4%
3. **推理路径长度更接近 ground truth**：FiDeLiS 平均深度 2.4（WebQSP），ToG 为 3.1，GT 为 2.3
4. **效率优势明显**：比 ToG 减少约 1.7x 运行时间（43.83s vs 74.26s per question on WebQSP）
5. **Path-RAG 覆盖率**：深度1时 CR=72.61%，深度2时 69.38%，深度>3时 62.78%，均远超 vanilla retriever
6. **演绎验证 > adequacy验证 > logit评分**：WebQSP 上 79.32 vs 74.13 vs 73.47
7. **RoG 的推理路径仅 67% 有效**，而 FiDeLiS 通过逐步验证确保 100% 有效

## 亮点与洞察

1. **填补检索与Agent的空白**：Path-RAG 继承检索的效率，DVBS 继承 Agent 的准确性，组合后两全其美
2. **演绎验证的双层设计**：全局验证判断是否足够、局部验证判断是否合法，同时解决了过早终止和过度延伸问题
3. **无需训练**：完全依赖 LLM 的 in-context 能力，通用性强，可直接迁移到新 KG
4. **计划生成的工程价值**：虽是 engineering trick，但释放了 LLM 的高阶思考能力，带来稳定提升
5. **案例分析很有说服力**：伊朗政体的例子清楚展示了 FiDeLiS 如何比 CoT、RoG、ToG 给出更全面准确的答案

## 局限与展望

1. 依赖外部 KG 的质量和完整性，KG 不完整或过时会影响效果
2. 多轮 LLM 调用仍有一定开销，虽比 ToG 好但距离实时仍有差距
3. 未在开放域或大规模 KG 上充分验证
4. 演绎验证依赖 LLM 的逻辑推理能力，LLM 本身的推理错误可能传播
5. Path-RAG 的嵌入质量受限于预训练 LM 的能力

## 相关工作与启发

- **ToG**（Sun et al.）：Agent 方法的代表，FiDeLiS 的效率提升了 1.7x
- **RoG**（Luo et al.）：检索方法的代表，但 33% 推理步骤无效
- **KD-CoT**（Wang et al.）：通过外部 KG 验证子推理步骤，思路与 FiDeLiS 的验证相近
- 启发：将演绎推理作为"校准工具"引入搜索过程是提升 LLM 推理忠实性的有效策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Path-RAG + DVBS 的组合设计巧妙，演绎验证的引入有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三数据集、多后端、消融、鲁棒性、效率分析、案例分析非常全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式和算法描述规范
- **价值**: ⭐⭐⭐⭐⭐ — 无需训练即可超越微调方法，实用价值很高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)

</div>

<!-- RELATED:END -->
