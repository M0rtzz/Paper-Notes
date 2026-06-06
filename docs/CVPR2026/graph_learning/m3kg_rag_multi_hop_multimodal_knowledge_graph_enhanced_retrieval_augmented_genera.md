---
title: >-
  [论文解读] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
description: >-
  [CVPR 2026][图学习][多模态知识图谱] 提出M3KG-RAG，通过轻量多Agent流水线构建多跳多模态知识图谱（M3KG），并设计GRASP机制进行实体定位和选择性剪枝，仅保留查询相关且有助回答的知识，大幅提升MLLM的音视觉推理能力。
tags:
  - "CVPR 2026"
  - "图学习"
  - "多模态知识图谱"
  - "检索增强生成"
  - "音视觉推理"
  - "图剪枝"
  - "多跳推理"
---

# M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation

**会议**: CVPR 2026  
**arXiv**: [2512.20136](https://arxiv.org/abs/2512.20136)  
**代码**: [项目页面](https://kuai-lab.github.io/cvpr2026m3kgrag/)  
**领域**: 图学习  
**关键词**: 多模态知识图谱, 检索增强生成, 音视觉推理, 图剪枝, 多跳推理

## 一句话总结

提出M3KG-RAG，通过轻量多Agent流水线构建多跳多模态知识图谱（M3KG），并设计GRASP机制进行实体定位和选择性剪枝，仅保留查询相关且有助回答的知识，大幅提升MLLM的音视觉推理能力。

## 研究背景与动机

现有多模态RAG存在两大瓶颈：1）现有MMKG主要覆盖图文模态，音视觉覆盖有限，且大多为单跳图谱，缺乏时间/因果依赖的多跳连接；2）基于共享嵌入空间的相似度检索存在模态鸿沟，无法过滤离题或冗余知识，即使检索到相关上下文也可能注入噪声。

M3KG-RAG的核心创新在于：构建跨音视觉的多跳知识图谱 + 按模态检索绕过模态鸿沟 + GRASP精准保留回答有用的子图。

## 方法详解

### 整体框架

原始多模态语料 → 三步Agent流水线构建M3KG（上下文增强三元组提取→知识锚定→上下文感知描述精炼 + 自反思循环） → 按模态检索候选子图 → GRASP定位+剪枝 → 图增强的MLLM生成。

### 关键设计

1. **多Agent M3KG构建流水线**:
    - 功能：从原始多模态语料构建多跳、跨模态知识图谱
    - 核心思路：Rewriter增强caption → Extractor提取三元组 → Normalizer标准化实体 → Searcher查询知识库获取描述 → Selector选择上下文相关描述 → Refiner适配原始表述 → Inspector自反思循环确保质量
    - 设计动机：仅用Qwen3-8B等轻量LLM即可完成，且自反思循环防止幻觉描述

2. **GRASP（Grounded Retrieval And Selective Pruning）**:
    - 功能：确保检索的知识既与查询相关又对回答有用
    - 核心思路：视觉定位（GroundingDINO检测实体在视频帧中的存在→mask IoU阈值过滤）+ 音频定位（TAG模型评估三元组与查询音频的匹配度）+ 轻量LLM二值掩码剪枝无用三元组
    - 设计动机：相似度检索只捕获broad语义，GRASP通过定位和剪枝提供fine-grained过滤

3. **按模态检索（Modality-Wise Retrieval）**:
    - 功能：绕过跨模态嵌入空间的模态鸿沟
    - 核心思路：视频查询用InternVL2匹配视觉项，音频查询用CLAP匹配音频项，然后通过图链接提升到三元组级别
    - 设计动机：共享嵌入空间中视频查询匹配文本知识库常失败

### 损失函数 / 训练策略

无模型训练，纯pipeline方案。M3KG构建在评估基准的训练集上完成，单张H100 GPU。

## 实验关键数据

### 主实验（Model-as-Judge评分）

| MLLM | 方法 | Audio QA | Video QA | AV QA |
|------|------|----------|----------|-------|
| Qwen2.5-Omni | None | 49.00 | 42.21 | 32.42 |
| Qwen2.5-Omni | VAT-KG | 51.30 | 43.50 | 35.44 |
| Qwen2.5-Omni | M3KG-RAG | 60.77 | 44.35 | 44.67 |

### Win-rate对比（vs VAT-KG）

| 基准 | VAT-KG胜率 | M3KG-RAG胜率 |
|------|-----------|-------------|
| AudioCaps-QA | 25.6% | 74.4% |
| VCGPT | 47.6% | 52.4% |
| VALOR | 41.8% | 58.2% |

### 关键发现

- 文本KG+简单RAG经常导致性能下降（Wikidata在多个设置上比无检索更差）
- 单跳MMKG（VAT-KG）改进有限，多跳结构关键
- 即使GPT-4o也能从M3KG-RAG获益，说明外部知识对大模型仍有价值
- GRASP的每个组件（定位+剪枝）都贡献了性能提升

## 亮点与洞察

- 端到端的多模态知识图谱构建和检索框架，覆盖音视觉文本三模态
- GRASP的"定位→剪枝"两步过滤设计直觉简洁且有效
- 仅用轻量级Qwen3-8B即可构建高质量知识图谱，成本可控

## 局限与展望

- 按模态检索的阈值τ和GRASP阈值η需要按数据集手动调整
- 知识图谱构建依赖训练集，泛化到新领域需重新构建
- GRASP的定位模型（GroundingDINO/TAG）本身可能有误差
- 仅评估了开放式QA，未覆盖其他多模态任务

## 相关工作与启发

- **vs VAT-KG**: 单跳概念图+简单检索；M3KG-RAG多跳图+GRASP精准过滤
- **vs GraphRAG/LightRAG**: 纯文本图RAG；M3KG-RAG扩展到音视觉多模态

## 评分

- 新颖性: ⭐⭐⭐⭐ 多跳多模态知识图谱+GRASP的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 三个基准、多个MLLM、win-rate和MJ双评估
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，流水线步骤详细
- 价值: ⭐⭐⭐⭐ 为多模态RAG提供了实用的知识图谱增强方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](../../ACL2026/graph_learning/megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](../../ACL2026/graph_learning/stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](../../ACL2026/graph_learning/legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
