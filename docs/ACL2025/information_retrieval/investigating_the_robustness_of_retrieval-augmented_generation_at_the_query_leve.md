---
title: >-
  [论文解读] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level
description: >-
  [ACL 2025][信息检索/RAG][RAG] 提出首个查询级 RAG 鲁棒性模块化分析框架，通过 5 种扰动类型 × 4 种检索器 × 3 种 LLM × 3 个数据集共 1092+ 实验，揭示 dense 与 sparse 检索器对不同扰动类型的互补鲁棒性，并给出可操作的工程建议。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "RAG"
  - "查询扰动"
  - "鲁棒性评估框架"
  - "dense/sparse 检索器"
  - "模块解耦分析"
---

# Investigating the Robustness of Retrieval-Augmented Generation at the Query Level

**会议**: ACL 2025  
**arXiv**: [2507.06956](https://arxiv.org/abs/2507.06956)  
**代码**: 即将公开  
**领域**: 信息检索 / RAG 鲁棒性  
**关键词**: RAG, 查询扰动, 鲁棒性评估框架, dense/sparse 检索器, 模块解耦分析  

## 一句话总结

提出首个查询级 RAG 鲁棒性模块化分析框架，通过 5 种扰动类型 × 4 种检索器 × 3 种 LLM × 3 个数据集共 1092+ 实验，揭示 dense 与 sparse 检索器对不同扰动类型的互补鲁棒性，并给出可操作的工程建议。

## 研究背景与动机

- **领域现状**：RAG 已成为缓解 LLM 幻觉和知识更新成本的主流方案，通过检索外部文档为生成提供事实性上下文，广泛应用于企业 QA、知识管理等场景。
- **现有痛点**：已有 RAG 鲁棒性研究主要关注检索文档端的噪声（不相关段落、错误信息），或者孤立地研究检索器/LLM 单模块的鲁棒性，缺乏从查询端出发对整个 RAG 管线各模块进行系统性的解耦评估。
- **核心矛盾**：实际用户的查询存在天然的多样性——不同的措辞习惯、拼写错误、冗余表述、语气差异——但我们不清楚这些变化分别对 RAG 管线中的哪个模块造成多大影响，无法针对性地进行优化。
- **本文目标**：量化 RAG 管线各组件（检索器、生成器、端到端）对各类查询扰动的敏感度，提供模块级诊断方法论，帮助从业者定位管线瓶颈。
- **切入角度**：设计 5 种保持语义不变的查询扰动（冗余信息、正式语气、歧义引入、10%/25% 拼写错误），在隔离和联合设置下分别测量各模块的性能变化，再通过 Pearson 相关性解耦主导因素。
- **核心 idea**：通过大规模受控实验和模块解耦相关性分析，首次揭示查询级扰动对 RAG 管线各模块的差异化影响并提出系统诊断框架。

## 方法详解

### 整体框架

本文的分析框架包含三层递进实验：

1. **检索器隔离分析**：对 4 种检索器（BGE-base-en-v1.5、Contriever 为 dense 检索器；BM25 Flat、BM25 Multi-field 为 sparse 检索器）分别施加 5 种扰动，通过 Recall@k 度量检索性能变化
2. **生成器隔离分析**：在两种极端设置下评估 3 种 LLM（Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.2、Qwen2.5-7B-Instruct）——Closed-book（纯参数知识，无检索）和 Oracle（假设完美检索，仅提供正确文档）
3. **端到端管线分析**：将检索器和生成器组合成 12 种 RAG 管线，测量联合性能并通过 Pearson 相关系数解耦各模块对端到端性能变化的贡献

数据集覆盖三个不同特性的 QA 数据集：NQ（通用单跳 QA，2.68M 文档）、HotpotQA（多跳 QA，5.23M 文档）、BioASQ（生物医学领域，14.91M 文档）。评估指标使用 Recall@k（检索端）和 Match（生成端，判断输出是否包含答案片段），采用 model-free 评估避免 LLM 评估器引入不稳定性。

### 关键设计一：五种查询扰动生成策略

- **冗余信息插入（Redundancy）**：通过 GPT-4o 向查询中插入不含答案线索的相关冗余描述，模拟用户输入过多背景信息的场景
- **正式语气变化（Formal Tone）**：通过 GPT-4o 将查询改写为更正式的表达，仅改变表面形式
- **歧义引入（Ambiguity）**：通过 GPT-4o 引入模糊表述（如插入 "might"、用更笼统的词替换具体词），模拟用户表达不精确的情况
- **拼写错误 10%/25%**：使用 TextAttack 库按 QWERTY 键盘邻近键替换词中字符，保留停用词以维持核心语义
- 每种扰动为每个原始样本生成 5 个扰动版本，并通过 GPT2-Large 的困惑度和 multilingual-e5-base 的语义相似度验证扰动样本的质量

### 关键设计二：模块解耦的 Pearson 相关性分析

- **功能**：定量区分端到端 RAG 性能波动中检索器和生成器各自的贡献比例
- **核心思路**：先逐样本计算每个扰动样本与原始样本之间的性能差异（检索器用 ΔRecall@5，生成器用 ΔMatch），然后分别计算「ΔRetriever — ΔRAG」和「ΔGenerator — ΔRAG」的 Pearson 相关系数
- **设计动机**：端到端 RAG 性能是检索器和生成器的耦合结果，直接观察无法区分瓶颈。通过解耦分析，从业者可针对特定扰动类型精准定位问题模块

### 关键设计三：LLM 内部表示可视化

- **功能**：从模型内部表示层面揭示查询扰动如何干扰 LLM 的推理过程
- **核心思路**：提取 Llama-3.1-8B 最后隐藏层所有注意力头的平均表示（最后一个非填充 token），PCA 降维到二维可视化
- **设计动机**：在 BioASQ 上发现即使 Oracle 设置下冗余和歧义扰动仍导致 LLM 内部表示明显分散，解释了为什么正确文档也无法完全弥补查询扰动的负面影响

### 训练策略

本文为纯分析性工作，所有检索器和 LLM 均直接使用预训练权重，不进行 fine-tuning。生成采用贪心解码（temperature=0），最大输入长度 4096 tokens，最大生成长度 128 tokens，每个检索文档截断至 100 词。实验在 RTX 3090 GPU 上完成，使用 vLLM 推理框架。

## 实验关键数据

### 主实验：检索器 Recall@5 (%) 在不同扰动下的性能

| 数据集 | 检索器 | Original | Redundancy | Formal | Ambiguity | Typo 10% | Typo 25% |
|--------|--------|----------|------------|--------|-----------|----------|----------|
| HotpotQA | BGE Base | **71.82** | 66.92 | 69.34 | 64.45 | 62.94 | 47.75 |
| HotpotQA | Contriever | **60.84** | 59.12 | 59.34 | 56.42 | 53.37 | 39.06 |
| HotpotQA | BM25 Flat | **60.81** | 44.72 | 54.20 | 49.38 | 54.34 | 41.92 |
| HotpotQA | BM25 MF | **58.00** | 47.20 | 53.90 | 50.17 | 50.59 | 37.36 |
| NQ | BGE Base | **64.59** | 55.10 | 61.65 | 51.60 | 50.04 | 34.35 |
| NQ | Contriever | **58.60** | 52.11 | 56.58 | 47.33 | 45.39 | 30.95 |
| BioASQ | BGE Base | **36.06** | 33.01 | 34.82 | 30.24 | 30.43 | 27.83 |
| BioASQ | BM25 Flat | **45.22** | 25.01 | 37.89 | 33.37 | 35.94 | 29.87 |

### 消融实验：Pearson 相关性解耦分析（BGE Base + Llama-3.1-8B）

| 数据集 | 相关类型 | Redundancy | Formal | Ambiguity | Typo 10% | Typo 25% |
|--------|---------|------------|--------|-----------|----------|----------|
| BioASQ | Retriever-RAG | 0.05 | 0.04 | 0.15 | **0.21** | **0.23** |
| BioASQ | Closed-book-RAG | 0.21 | 0.08 | 0.23 | 0.05 | 0.10 |
| BioASQ | Oracle-RAG | **0.35** | **0.15** | **0.33** | 0.04 | 0.12 |
| NQ | Retriever-RAG | **0.31** | **0.27** | **0.30** | **0.35** | **0.40** |
| NQ | Closed-book-RAG | 0.03 | 0.04 | 0.11 | 0.08 | 0.16 |
| NQ | Oracle-RAG | 0.11 | 0.14 | 0.15 | 0.06 | 0.03 |

### 关键发现

- Dense 检索器对冗余信息更鲁棒但对拼写错误敏感（BGE 在 NQ 上 Typo 25% 导致 Recall 从 64.59% 暴跌至 34.35%），Sparse 检索器则相反（BM25 Flat 在 BioASQ 上冗余信息导致从 45.22% 降至 25.01%）
- NQ 上端到端性能与检索器性能变化强相关（Pearson 0.27-0.40），检索器是主导因素；BioASQ 上冗余/歧义时 Oracle 相关性更高（0.33/0.35），生成器的上下文利用能力成为瓶颈
- Closed-book 性能不能预测 RAG 性能——Mistral 参数知识有限但 Oracle 设置下表现最佳，说明 RAG 中生成器评估不能脱离检索上下文
- 正式语气变化在所有模块和数据集上影响最小，Typo 25% 最具破坏性

## 亮点与洞察

- 首个查询级 RAG 鲁棒性的模块化诊断框架，三层递进分析使结论层次清晰且可复现
- Dense/Sparse 互补性发现具有直接工程价值——可考虑混合检索策略来对冲查询变体风险
- 模块解耦方法论实用性强：从业者可针对自身管线和数据运行该框架，快速定位瓶颈模块
- 「Closed-book ≠ RAG 鲁棒性」的结论纠正了一个常见误区
- 实验规模大（1092+ 实验）但成本可控（7-8B 模型 + RTX 3090），方法论可广泛复现

## 局限与展望

- 未包含 reranker 模块，实际 RAG 管线中 reranker 可能改变扰动的影响分布
- 仅使用 7-8B 参数 LLM，更大模型（70B+）的鲁棒性表现可能不同
- 未探索缓解策略的实际效果（查询改写/扩展、对抗训练、查询扰动增强训练）
- LLM 内部表示分析仅停留在 PCA 可视化层面，缺乏定量的机制性解释
- 评估指标仅使用 surface matching (Match)，未采用语义级指标如 BERTScore
- 管线超参数空间（上下文长度、文档截断策略）未充分探索

## 相关工作与启发

- **检索器鲁棒性**：Zhuang & Zuccon (2022) 研究 BERT 检索器对拼写错误的编码策略与训练方法；Arabzadeh et al. (2023) 通过扰动 dense 表示直接评估稳定性。本文扩展到多种扰动类型的全面对比
- **LLM 鲁棒性**：Shi et al. (2023) 发现 LLM 容易被不相关上下文分散注意力；Zhu et al. (2024a,b) 在多粒度层面进行 prompt 扰动分析。本文独特之处在于区分了 closed-book 和 oracle 设置，揭示了两者结论不一致的问题
- **管线鲁棒性**：Chen et al. (2023) 测试 RAG 对不相关/错误信息的处理能力；Fang et al. (2024) 和 Yoran et al. (2024) 提出训练方法增强噪声鲁棒性。本文从查询端出发且提供模块解耦分析，是补充性视角
- **启发**：Dense + Sparse 混合检索可能是最直接的鲁棒性提升策略；在 RAG joint training 中加入查询扰动增强（perturbation-aware training）是值得探索的方向

## 评分

- 新颖性: ⭐⭐⭐ 首个查询级 RAG 鲁棒性的模块解耦分析框架，但核心方法是标准的受控实验+相关性分析，方法新颖性有限
- 实验充分度: ⭐⭐⭐⭐⭐ 1092+ 实验覆盖 4 检索器 × 3 LLM × 5 扰动 × 3 数据集，分析全面深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，建议部分有实用价值
- 价值: ⭐⭐⭐⭐ 对 RAG 工程实践有较强指导意义，模块解耦方法论可复用性高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
