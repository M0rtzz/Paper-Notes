---
title: >-
  [论文解读] WebWalker: Benchmarking LLMs in Web Traversal
description: >-
  [ACL 2025][网页遍历] 提出 WebWalkerQA 基准评测 LLM 在网页深层遍历中的信息获取能力，并设计 WebWalker 多智能体框架通过 Explore-Critic 范式模仿人类网页导航行为，结合 RAG 的横向-纵向集成显著提升了复杂问答性能。
tags:
  - ACL 2025
  - 网页遍历
  - 检索增强生成
  - 多智能体框架
  - 深层信息获取
  - 基准评测
---

# WebWalker: Benchmarking LLMs in Web Traversal

**会议**: ACL 2025  
**arXiv**: [2501.07572](https://arxiv.org/abs/2501.07572)  
**代码**: https://github.com/Alibaba-NLP/DeepResearch  
**领域**: LLM/NLP  
**关键词**: 网页遍历、检索增强生成、多智能体框架、深层信息获取、基准评测

## 一句话总结

提出 WebWalkerQA 基准评测 LLM 在网页深层遍历中的信息获取能力，并设计 WebWalker 多智能体框架通过 Explore-Critic 范式模仿人类网页导航行为，结合 RAG 的横向-纵向集成显著提升了复杂问答性能。

## 研究背景与动机

**领域现状**：检索增强生成（RAG）在开放域问答中表现优异，现有方法通常依赖搜索引擎返回的页面片段作为知识源，LLM 基于这些片段进行回答。

**现有痛点**：传统搜索引擎只能检索到网页的浅层内容（如首页摘要），无法深入网站的子页面获取更详细、更深层的信息。当问题涉及多层级、跨子页面的复杂信息时，单纯依赖搜索引擎返回的浅层结果会严重限制 LLM 的回答质量。

**核心矛盾**：web 上的信息是层级组织的——一个网站往往包含首页、子页面、子子页面等多层结构。现有 RAG 系统只看到"第一层"就做回答，类似于只读了目录就写报告，这导致在需要深层信息的场景下回答不完整甚至错误。

**本文目标**：(1) 构建一个专门评测 LLM 网页遍历能力的基准数据集；(2) 设计一个多智能体框架让 LLM 像人类一样在网站的子页面间导航，系统性地提取深层信息。

**切入角度**：人类浏览网站时会根据需求在不同子页面间跳转——先浏览一级页面获取框架，再深入特定子页面获取细节。这种"探索-判断"的导航模式可以被建模为多智能体的协作过程。

**核心 idea**：用一个 Explorer Agent 负责在网页间导航和信息提取，一个 Critic Agent 负责评判已收集信息是否足够回答问题，通过两者的迭代协作实现人类式的深层网页遍历。

## 方法详解

### 整体框架

WebWalker 采用多智能体架构：给定一个问题和起始 URL，系统首先由 Explorer Agent 访问网页并解析内容，提取有用信息和潜在的导航链接；随后 Critic Agent 评估已收集的信息是否充分，决定是继续探索还是停止并生成答案。整个过程通过"横向遍历"（同级子页面间导航）和"纵向深入"（沿链接进入更深层页面）两种方式系统性地覆盖网站信息。

### 关键设计

1. **Explorer Agent（探索者）**:

    - 功能：负责网页内容解析和导航决策
    - 核心思路：Explorer 接收当前网页内容和问题，执行两项任务——从当前页面提取与问题相关的信息片段，同时识别页面上可能包含更多相关信息的超链接。它维护一个"已访问页面"列表避免重复访问，并根据链接的相关性排序选择下一个要访问的页面。每次访问新页面后，提取的信息会被累积到一个信息池中。
    - 设计动机：模仿人类浏览网页时的信息提取和导航行为，避免漫无目的地遍历所有链接，而是有目标地选择最可能包含相关信息的路径。

2. **Critic Agent（评判者）**:

    - 功能：评估已收集信息的充分性，决定探索-停止策略
    - 核心思路：每轮探索结束后，Critic 接收当前累积的信息池和原始问题，判断信息是否已足够回答该问题。如果信息充分，触发停止信号并将信息传给答案生成模块；如果不够，向 Explorer 反馈需要补充的信息方向，引导下一轮探索。这形成一个迭代闭环直到信息充分或达到最大遍历深度。
    - 设计动机：防止过度探索浪费计算资源，同时避免信息不足就草率回答。Critic 的引导机制使探索过程更高效，避免 Explorer 在不相关的页面上浪费探索次数。

3. **横向-纵向集成遍历策略**:

    - 功能：结合同层广度遍历和跨层深度遍历实现全面信息覆盖
    - 核心思路：横向遍历（Horizontal）指在网站的同一层级中浏览多个并列的子页面（如产品列表中的不同产品页），纵向遍历（Vertical）指沿着一个页面的链接深入到更深层的子页面（如从产品列表到具体产品详情再到评价页面）。系统根据问题类型自动调整两种策略的比例。
    - 设计动机：不同类型的问题需要不同的导航策略——比较类问题更需要横向遍历，深入了解类问题更需要纵向遍历。这种灵活的集成策略使 WebWalker 能适应多样化的信息需求。

### 训练策略

WebWalker 作为框架层面的方法，不涉及模型训练。Explorer 和 Critic 均基于已有的 LLM（如 GPT-4、Qwen 等）通过 prompt 驱动的方式实现，核心在于 prompt 设计和多智能体协作流程。WebWalkerQA 基准的构建则需要人工标注：标注者在真实网站上导航并标注可回答的问答对。

## 实验关键数据

### 主实验

| 方法 | 数据来源 | EM | F1 | Acc |
|------|---------|-----|-----|-----|
| Direct RAG (浅层检索) | 搜索引擎首页 | 18.2 | 32.5 | 24.1 |
| BM25 + RAG | 搜索引擎页面 | 21.3 | 35.8 | 27.0 |
| WebWalker + RAG (GPT-4o) | 深层子页面 | 33.5 | 48.7 | 39.2 |
| WebWalker + RAG (Qwen-Max) | 深层子页面 | 29.1 | 43.2 | 35.8 |
| Human Performance | - | 72.8 | 85.3 | 78.6 |

### 消融实验

| 配置 | EM | F1 | 说明 |
|------|-----|-----|------|
| Full WebWalker | 33.5 | 48.7 | 完整模型 |
| w/o Critic Agent | 27.8 | 41.3 | 去掉评判者后探索效率下降 |
| w/o 纵向遍历 | 28.9 | 42.1 | 仅横向遍历，深层信息缺失 |
| w/o 横向遍历 | 30.2 | 44.5 | 仅纵向遍历，覆盖面不足 |
| 固定遍历深度=2 | 31.1 | 45.6 | 限制最大深度的影响 |

### 关键发现

- Critic Agent 的引入贡献最大（EM +5.7），说明有目标的探索比盲目遍历更有效
- 横向和纵向遍历都不可缺少，但纵向遍历对整体性能影响更大，因为很多答案藏在深层子页面
- WebWalkerQA 基准具有挑战性：即使是 GPT-4o 驱动的 WebWalker 也与人类表现差距巨大（EM 差 39.3），表明网页深层信息获取仍有很大改进空间
- 不同 LLM 作为 backbone 的表现差异显著，说明网页导航能力高度依赖于模型的指令遵循和推理能力

## 亮点与洞察

- **Explore-Critic 范式**非常直觉且有效——将信息获取分解为"探索"和"评判"两个独立角色，通过迭代反馈逐步逼近充分回答。这种设计可以迁移到任何需要多步信息收集的场景（如文档分析、代码仓库理解）
- **WebWalkerQA 基准填补了空白**：此前缺乏专门评测 LLM 网页遍历能力的数据集，该基准基于真实网站构建，问题需要深层导航才能回答
- 横向-纵向集成策略的设计启发：复杂信息检索不应只在一个维度上搜索，而应根据任务需求灵活调整搜索策略

## 局限与展望

- 当前评测仅覆盖英文网站，网页结构和语言多样性未充分考虑
- Explorer 依赖 HTML 解析获取链接和内容，对于高度动态（JS 渲染）的网页可能无法正确解析
- 遍历效率仍有提升空间——每次页面访问都需要一次 LLM 推理调用，深度较大时计算开销显著
- 人机差距巨大（EM 差距 ~40 分），说明当前 LLM 在复杂网页导航任务上的能力远未成熟

## 相关工作与启发

- **vs WebGPT**: WebGPT 也尝试让 LLM 浏览网页，但它更侧重于搜索引擎交互而非深层网站遍历。WebWalker 的关键差异在于强调多层级子页面的系统性遍历
- **vs ReAct Agent**: ReAct 是通用的推理-行动框架，WebWalker 可以看作 ReAct 在网页遍历场景的具体实例化，额外加入了 Critic 评判机制来控制探索过程
- **vs MRAG / Auto-RAG**: 这些方法关注 RAG 系统的多轮检索，但信息源仍限于搜索引擎返回结果。WebWalker 的信息源是网站的完整子页面结构

## 评分

- 新颖性: ⭐⭐⭐⭐ 网页深层遍历是重要但被忽视的方向，Explore-Critic 范式设计合理
- 实验充分度: ⭐⭐⭐⭐ WebWalkerQA 基准构建扎实，多模型多配置对比完善
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述直观
- 价值: ⭐⭐⭐⭐ 基准数据集和框架对后续网页智能体研究有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)
- [\[ACL 2025\] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)

<!-- RELATED:END -->
