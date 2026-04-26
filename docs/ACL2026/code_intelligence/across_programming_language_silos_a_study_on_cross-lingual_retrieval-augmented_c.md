---
title: >-
  [论文解读] Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation
description: >-
  [ACL 2026][跨语言代码生成] 首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。
tags:
  - ACL 2026
  - 跨语言代码生成
  - 检索增强生成
  - 知识迁移
  - 多语言编程
  - 代码检索
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

构建覆盖13种编程语言的大规模数据集（约14K实例），包含NL prompt、验证过的参考解和可执行测试用例。通过三种检索设置和5个代码LLM进行系统评估。

### 关键设计

1. **三种检索实验设置**:

    - 功能：从不同角度评估跨语言知识迁移
    - 核心思路：(1) Golden Solution Document——oracle检索模拟理想条件，测量跨语言迁移的上界；(2) Top-k Retrieved Documents——完整RACG管道的端到端评估；(3) Top-k without NL——去除自然语言描述，模拟现实中的纯代码片段场景
    - 设计动机：通过控制变量分离检索和生成阶段的影响，明确跨语言迁移的瓶颈所在

2. **大规模多语言代码数据集**:

    - 功能：提供跨13种编程语言的统一评测基准
    - 核心思路：每个实例包含NL描述、参考解和测试用例，覆盖C++、Go、Java、JavaScript、Python、Rust等13种语言
    - 设计动机：现有数据集仅覆盖2-5种语言，无法支撑大规模跨语言研究

3. **多语言vs Python专用LLM对比**:

    - 功能：揭示预训练多样性对跨语言迁移能力的影响
    - 核心思路：对比多语言LLM（CodeLlama, DeepSeek-Coder, Qwen2.5-Coder）和Python专用LLM（Phi-1, Phi-1.5）在跨语言RACG中的表现
    - 设计动机：区分跨语言迁移能力来源——是来自架构还是预训练数据的多样性

### 损失函数 / 训练策略

本文为实证研究，不涉及模型训练。使用贪心解码（temperature=0.0）确保可复现性，评估指标为Pass@1。

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

## 相关论文

- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)

<!-- RELATED:END -->
