---
title: >-
  [论文解读] Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation
description: >-
  [ACL 2026][代码智能][跨语言代码生成] 首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。
tags:
  - "ACL 2026"
  - "代码智能"
  - "跨语言代码生成"
  - "检索增强生成"
  - "知识迁移"
  - "多语言编程"
  - "代码检索"
---

# Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation

**会议**: ACL 2026  
**arXiv**: [2506.03535](https://arxiv.org/abs/2506.03535)  
**代码**: [GitHub](https://github.com/icip-cas/Cross-Lingual-RACG)  
**领域**: Code Intelligence / Cross-Lingual Code Generation  
**关键词**: 跨语言代码生成, 检索增强生成, 知识迁移, 多语言编程, 代码检索

## 一句话总结

首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。

## 研究背景与动机

**领域现状**：检索增强代码生成（RACG）通过检索相关代码片段增强LLM的代码生成能力，但现有研究主要聚焦Python和Java等单一语言设置。

**现有痛点**：编程语言间的代码知识分布严重不均——Python拥有丰富的文档和社区资源，而Scala等小众语言资源匮乏。企业技术栈迁移也产生了大量跨语言代码转换需求。

**核心矛盾**：RACG能否有效地将一种编程语言的代码知识迁移到另一种语言？这种迁移是否对所有语言对都同样有效？

**本文目标**：系统研究RACG中的跨编程语言知识迁移机制，回答三个关键研究问题。

**切入角度**：设计三种检索实验设置（oracle注入、实际检索、无自然语言代码检索），控制变量分析跨语言迁移效果。

**核心 idea**：跨语言代码知识迁移是可行但不对等的，效果取决于语言对的亲缘性和LLM预训练语料的多样性。

## 方法详解

### 整体框架

本文是一项围绕"跨编程语言检索增强代码生成（RACG）"的实证研究，核心是先造数据、再设实验、最后做对照分析。输入是某种编程语言的 NL 需求，中间从一个覆盖 13 种编程语言、约 14K 实例的统一数据集（每个实例含 NL 描述、验证过的参考解和可执行测试用例）里检索另一种语言的相关代码作为上下文，输出是目标语言的代码并用测试用例判定 Pass@1。围绕这一流程，本文用三种检索设置和五个代码 LLM 交叉评估，把"知识能否跨语言迁移、迁移是否对等、迁移能力从何而来"三个问题逐一拆开。

### 关键设计

**1. 三种检索实验设置：用控制变量把检索和生成两阶段的影响分离开。**

为了定位跨语言迁移的真正瓶颈，本文设计了从理想到现实的三档检索设置。Golden Solution Document 用 oracle 直接注入目标解文档，模拟检索完美的理想条件，测量的是跨语言迁移的能力上界；Top-k Retrieved Documents 是完整 RACG 管道的端到端评估，反映真实检索质量下的表现；Top-k without NL 进一步去掉文档里的自然语言描述、只留纯代码片段，逼近企业代码库中"有代码没注释"的现实场景。三档对照之下，若 oracle 设置仍然提升有限，就说明瓶颈出在生成阶段而非检索阶段——这正是本文得到的关键判断。

**2. 大规模多语言代码数据集：用统一格式的 14K 实例支撑 13 语言的跨语言研究。**

现有 RACG 数据集大多只覆盖 2–5 种语言，无法支撑系统的跨语言对比，因此本文构建了覆盖 C++、Go、Java、JavaScript、Python、Rust 等 13 种编程语言的数据集。每个实例统一包含 NL 描述、参考解和可执行测试用例三件套，保证不同语言之间可比、可执行、可量化，从而让"源语言 → 目标语言"的任意配对都能在同一标准下测量增益。

**3. 多语言 vs Python 专用 LLM 对比：把迁移能力的来源归因到预训练多样性。**

跨语言迁移能力究竟来自模型架构还是预训练数据的语言多样性，本文用一组对照模型来回答。一边是多语言代码 LLM（CodeLlama、DeepSeek-Coder、Qwen2.5-Coder），另一边是几乎只见过 Python 的专用模型（Phi-1、Phi-1.5），在完全相同的跨语言 RACG 设置下比较二者利用异语言上下文的能力。结果是 Python 专用模型几乎无法从跨语言上下文获益，从而把迁移能力清晰地归因于预训练语料的语言多样性而非架构本身。

### 损失函数 / 训练策略

本文为纯实证研究，不涉及任何模型训练或微调；所有模型均采用贪心解码（temperature=0.0）以保证结果可复现，评估指标统一为 Pass@1。

## 实验关键数据

### 主实验（Oracle注入，多语言LLM平均）

| 源语言→目标 | C++ | Go | Java | JS | Python | 平均增益 |
|------------|------|-----|------|-----|--------|--------|
| C++ | - | +4.47 | +20.33 | +18.90 | +15.04 | +14.68 |
| Go | +9.15 | - | - | - | - | - |
| Baseline(无检索) | 54.27 | 42.68 | 61.79 | 58.33 | 59.35 | 55.28 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 去除NL信息 | 性能仅轻微下降 | 代码检索器不强依赖自然语言 |
| Python专用LLM | 跨语言迁移差 | 预训练多样性是跨语言迁移的关键 |
| 代码专用检索器 | 显著优于通用检索 | 专用检索器更有效桥接NL意图和代码语义 |

### 关键发现
- 跨语言知识迁移即使在oracle条件下也非trivial，说明生成阶段本身存在跨语言gap
- 迁移效果呈不对等性——与语言对的语法亲缘性相关（如Java→JavaScript效果好于Java→Go）
- Python专用LLM几乎无法利用跨语言上下文，强调了预训练多样性的重要性
- 去除NL后检索性能下降很小，说明代码语义本身足以支撑检索

## 亮点与洞察
- 首次将"跨语言"概念从自然语言扩展到编程语言的RACG场景，开辟了新的研究方向
- 实验设计严谨：三种检索设置形成从理想到现实的梯度，清晰揭示迁移机制
- "Python专用LLM无法跨语言迁移"的发现对模型训练策略有重要指导意义

## 局限与展望
- 仅测试约7B参数的LLM，更大模型的跨语言能力可能不同
- 数据集构建依赖现有benchmark的翻译，可能引入偏差
- 未探索fine-tuning对跨语言迁移能力的影响
- 未来可研究跨语言检索策略的优化和混合语言检索

## 相关工作与启发
- **vs 单语RACG**: 揭示了跨语言场景的独特挑战——不对等迁移和语言亲缘性
- **vs 代码翻译任务**: RACG不是直接翻译而是利用源语言知识增强目标语言生成
- **vs 多语言NLP**: 编程语言的"跨语言"与自然语言有相似机制（亲缘性影响迁移），但也有独特性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究跨编程语言RACG
- 实验充分度: ⭐⭐⭐⭐⭐ 13语言×5模型×3设置的大规模实验
- 写作质量: ⭐⭐⭐⭐ 三个RQ组织清晰
- 价值: ⭐⭐⭐⭐ 为多语言代码工具设计提供实证指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)

</div>

<!-- RELATED:END -->
