---
title: >-
  [论文解读] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition
description: >-
  [NeurIPS 2025][检索增强生成] 提出Routing-to-RAG (R2RAG)系统，通过LLM查询分类器将简单查询路由到单轮Vanilla RAG、复杂查询路由到迭代式Vanilla Agent，全部基于Qwen3-4B（未量化）和Qwen3-Reranker-0.6B两个轻量模型在单块消费级GPU上运行，获NeurIPS 2025 MMU-RAG竞赛开源赛道Best Dynamic Evaluation奖。
tags:
  - NeurIPS 2025
  - 检索增强生成
  - 查询分类
  - 自适应检索
  - 轻量级RAG
  - 动态评估
---

# RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition

**会议**: NeurIPS 2025  
**arXiv**: [2602.20735](https://arxiv.org/abs/2602.20735)  
**代码**: [GitHub](https://github.com/rmit-ir/NeurIPS-MMU-RAG)  
**领域**: 信息检索/RAG  
**关键词**: 检索增强生成, 查询分类, 自适应检索, 轻量级RAG, 动态评估

## 一句话总结

提出Routing-to-RAG (R2RAG)系统，通过LLM查询分类器将简单查询路由到单轮Vanilla RAG、复杂查询路由到迭代式Vanilla Agent，全部基于Qwen3-4B（未量化）和Qwen3-Reranker-0.6B两个轻量模型在单块消费级GPU上运行，获NeurIPS 2025 MMU-RAG竞赛开源赛道Best Dynamic Evaluation奖。

## 研究背景与动机

**领域现状**：检索增强生成（RAG）已成为提升LLM可靠性的标准方法。LiveRAG（ACM SIGIR 2025）和MMU-RAG（NeurIPS 2025）等竞赛提供标准化评估设置。MMU-RAG竞赛包含四个评估维度：静态/动态评估 × 开源/闭源系统，最终排名综合自动指标的鲁棒性聚合和人工Likert评分。

**现有痛点**：
- 现有RAG系统对所有查询使用统一检索策略，忽略查询复杂度差异——简单事实问题和多步推理问题需要完全不同的处理方式
- 合成LLM评估（LLM-as-a-Judge）存在分布偏差和评估偏见，与真实用户偏好存在gap
- 高性能RAG通常依赖大模型和昂贵硬件，限制了可复现性和实际部署

**本文切入点**：基于SIGIR 2025 LiveRAG冠军系统G-RAG进行扩展，引入查询复杂度分类和自适应路由机制，结合N=20人的定性用户研究反馈来指导系统优化，实现轻量级但高效的动态RAG。

## 方法详解

### 整体框架

R2RAG是一个"分类-路由-生成"三阶段管线：(1) 查询分类器（LLM-based）判断查询是简单还是复杂；(2) 简单查询走Vanilla RAG单轮检索路径，复杂查询走Vanilla Agent迭代检索路径；(3) 收集充分证据后由LLM生成带引用的回答。所有组件共享Qwen3-4B（推理）和Qwen3-Reranker-0.6B（重排序），通过vLLM部署在单块GPU上。

### 关键设计

1. **查询分类器 (Query Classifier)**:

    - 功能：将输入查询分为simple（事实性/单步）和complex（多面/多步推理）两类
    - 实现方案对比：测试了两种方案——LR分类器（在175,850条查询上训练，128维语义嵌入+19维语言特征=147维特征向量）和LLM-based分类器（用结构化prompt指导Qwen3-4B判断"是否需要多次Google搜索"）
    - 最终选择LLM-based分类器，原因：(a) 能推理查询语义而非仅依赖表面特征，(b) 更好处理OOD查询，(c) 可通过修改prompt灵活调整分类标准。代价是推理开销更大，但在竞赛10分钟/查询的时限内可接受
    - 设计动机：定性用户研究发现Vanilla Agent对复杂查询效果好但对简单查询过度冗长，Vanilla RAG对简单查询简洁但对复杂任务信息不足——因此需要路由机制取长补短

2. **Vanilla RAG（简单查询路径）**:

    - 四阶段单轮管线：查询变体生成 → 并行检索 → 重排序 → 答案生成
    - 查询变体：Qwen3-4B开启thinking mode生成3个变体，用不同关键词/表述扩展检索覆盖
    - 检索：每个变体在ClueWeb22-A索引上并行检索，每个返回≤10篇，去重后约<30篇
    - 重排序：Qwen3-Reranker-0.6B作为Pointwise重排序器，对每个query-document对预测"yes"概率作为相关性分数。选取top文档并截断到5K词限制
    - 答案生成：Qwen3-4B基于检索文档合成回答，要求附带[ID]格式行内引用，对争议性话题鼓励提供平衡观点

3. **Vanilla Agent（复杂查询路径）**:

    - 核心机制：在Vanilla RAG基础上引入迭代搜索循环，参数更激进——生成≤5个查询变体（而非3个），文档token上限25K（而非5K词）
    - 状态维护：跨迭代累积(1)已使用查询（避免重复）和(2)有用文档摘要（指导后续搜索）
    - 停止条件 $S$: $(T > 20{,}000) \lor (Cov = 1) \lor (i \geq 5)$，即累积token > 20K、覆盖度完整、或迭代≥5次时终止
    - 文档评审与覆盖度判断（Cov）：每轮重排序后由Qwen3-4B评估5个维度——是否需要多视角、子问题是否覆盖、争议话题是否平衡、新有用文档识别、剩余知识空白。不充分则生成新查询针对空白搜索

### 模型选择与部署策略

| 配置项 | 具体设置 | 说明 |
|--------|---------|------|
| 推理模型 | Qwen3-4B（未量化） | 先前实验表明4-bit量化的Qwen3-8B效果弱于未量化的4B |
| 重排序模型 | Qwen3-Reranker-0.6B | 尺寸适合硬件预算，与主模型共存于单GPU |
| 推理框架 | vLLM | 高效内存管理与推理加速 |
| 上下文窗口 | 25,000 tokens（原生32,768） | 缩减以节省GPU内存 |
| 解码参数 | temperature=0.6, top_p=0.95 | 开发者推荐配置 |
| 硬件要求 | 单块消费级GPU | - |

### 损失函数 / 训练策略

系统本身不需要端到端训练。LR分类器用175,850条查询数据训练（TREC Deep Learning + Deep-Research Questions + TREC RAG 2025 + Natural Questions），但最终未被采用。核心组件均通过精心设计的prompt驱动，系统优化主要依赖定性用户研究反馈来迭代调整prompt和组件参数。

## 实验关键数据

### 定性用户研究

- 参与者：N=20人，来自RMIT大学CHAI中心，覆盖硕士生到教授，专业涵盖IR、ML、HCI、NLP、数据科学
- 方法：2小时自由探索，每人平均提交17条查询，提供二元反馈（👍/👎）、开放式评论和口头反思
- 分析方式：Preference Ratio（👍/👎比率）+ 开放式评论人工检查

| 系统变体 | 简单查询表现 | 复杂查询表现 | 主要问题 |
|----------|-------------|-------------|---------|
| Vanilla RAG | 简洁准确 | 信息不充分 | 多面问题覆盖不足 |
| Vanilla Agent | 过度冗长 | 有效处理 | 简单问题不必要的多轮检索 |
| R2RAG（路由） | 简洁准确 | 有效处理 | 取两者之长 |

### 竞赛结果

R2RAG获NeurIPS 2025 MMU-RAG竞赛**开源赛道Best Dynamic Evaluation奖**。竞赛评估框架结合：
- 鲁棒性感知的标准化自动指标聚合
- 基于Likert量表的人工序数评分
- 静态和实时动态（RAG-Arena）两种评估模式

### 关键发现

- **小模型可行**：Qwen3-4B（4B参数，未量化）在查询分类、查询重构、文档评审、答案生成等多个任务上表现强劲，4-bit量化的更大模型（Qwen3-8B）反而效果更差
- **查询路由有效**：统一的Preference Ratio下两个变体看似相近，但定性分析揭示了互补优势——路由机制让系统能在不同查询类型上都发挥最优策略
- **定性评估不可替代**：用户反馈揭示了自动指标无法捕捉的维度——系统偏见感知、回答结构偏好、详细程度期望、语调敏感性等
- **全文档推理优于分块**：检索和推理都在完整文档上进行，而非chunk-level操作

## 亮点与洞察

- 系统工程的精细打磨比模型规模更重要——4B模型+精心设计的路由/prompt在竞赛中击败了更大/更贵的系统
- 停止条件设计（token阈值+覆盖度+迭代上限三重保障）简洁有效，既防止上下文溢出又确保信息充分
- 定性用户研究驱动系统改进的方法论值得借鉴——从N=20人的反馈中提取可操作的设计改进，而非盲目优化自动指标
- 与竞赛主办方结论一致："Live evaluation matters"——动态实时评估揭示了静态指标无法捕捉的定性差异

## 局限与展望

- 论文为竞赛系统报告，缺乏与其他参赛系统的直接定量对比（无分数对比表）
- 查询分类为二元（simple/complex），更细粒度的多级分类可能进一步提升路由效果
- 依赖竞赛提供的ClueWeb22-A检索API，未涉及检索索引本身的优化
- 覆盖度判断完全依赖LLM主观评估，缺乏可量化的覆盖度度量指标
- 未讨论延迟/吞吐量——单GPU约束下复杂查询的多轮迭代可能导致较长响应时间

## 相关工作与启发

- 直接继承自G-RAG（SIGIR 2025 LiveRAG冠军），展示了竞赛系统跨赛道迭代改进的路径
- 查询复杂度分类借鉴了Researchy Questions数据集的定义——非事实、需要长篇推理或多步合成
- Pointwise重排序方法来自"Beyond Yes and No"（NAACL 2024），用细粒度相关性标签替代二元判断
- Qwen3系列模型（4B推理+0.6B重排序）的组合表明，任务特化的小模型组合在资源受限场景下可优于单一大模型

## 评分

- 新颖性: ⭐⭐⭐ 查询路由不算新，但在竞赛场景下的工程优化和定性评估驱动改进有价值
- 实验充分度: ⭐⭐⭐ 竞赛获奖证明效果，但缺乏与其他系统的定量对比表
- 写作质量: ⭐⭐⭐⭐ 竞赛报告结构清晰，设计选择有理有据，prompt全部公开
- 价值: ⭐⭐⭐⭐ 对资源受限RAG部署有很强的实践参考价值，完整开源

<!-- RELATED:START -->

## 相关论文

- [HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)
- [RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction](../../ACL2025/information_retrieval/empaths_at_semeval-2025_task_11_retrieval-augmented_approach_to_perceived_emotio.md)
- [SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](../../ACL2025/information_retrieval/refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)

<!-- RELATED:END -->
