---
title: >-
  [论文解读] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
description: >-
  [ACL 2025][信息检索/RAG][Information Retrieval] 提出AIR-Bench——首个利用LLM自动生成测试数据的异构IR基准，覆盖2个任务（QA/长文档）、9个领域、13种语言共69个数据集，三阶段质量控制管线确保生成数据与人工标注高度一致，解决了传统IR基准领域覆盖有限和更新成本高的问题。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "Information Retrieval"
  - "Automated Benchmark"
  - "LLM Data Generation"
  - "Multilingual Retrieval"
  - "RAG Evaluation"
---

# AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark

**会议**: ACL 2025  
**arXiv**: [2412.13102](https://arxiv.org/abs/2412.13102)  
**代码**: [GitHub](https://github.com/AIR-Bench/AIR-Bench)  
**领域**: 信息检索 / 基准评估  
**关键词**: Information Retrieval, Automated Benchmark, LLM Data Generation, Multilingual Retrieval, RAG Evaluation

## 一句话总结

提出AIR-Bench——首个利用LLM自动生成测试数据的异构IR基准，覆盖2个任务（QA/长文档）、9个领域、13种语言共69个数据集，三阶段质量控制管线确保生成数据与人工标注高度一致，解决了传统IR基准领域覆盖有限和更新成本高的问题。

## 研究背景与动机

**领域现状**：IR评估经历了从单语QA（MS MARCO、Natural Questions）到多语言（Mr.TyDi、MIRACL）再到通用跨领域（BEIR、MTEB）的演进。这些基准推动了检索模型的快速发展。

**现有痛点**：(1) 领域覆盖有限——现有基准受限于预定义领域和人工标注，无法高效扩展到新兴领域（如特定行业的法律、金融IR）；(2) 数据泄露风险——热门基准的测试数据可能已被检索模型的训练集覆盖，导致评估失真；(3) 更新成本高——人工标注新领域数据需要大量时间和专家资源，无法跟上领域演进速度。

**核心矛盾**：IR模型越来越通用和强大，但评估基准的覆盖面和更新速度远远跟不上——导致许多热门基准出现饱和现象（如MTEB/C-MTEB上的密集领域内微调使分数接近上限）。

**本文目标** 构建一个自动化、异构、动态的IR基准，能够低成本地不断扩展到新领域和语言，同时保证测试数据质量与人工标注一致。

**切入角度**：利用LLM（GPT-4）自动生成测试数据——但不是简单地让LLM编造QA对，而是基于真实语料库，通过"角色→场景→查询→重写→难负例"的多阶段管线生成高质量、高多样性的测试数据。

**核心 idea**：用LLM自动化IR基准的构建，使其能低成本地覆盖任意新领域，同时通过严格质量控制保证与人工标注基准的一致性。

## 方法详解

### 整体框架

AIR-Bench的数据生成管线分三个阶段：(1) **语料准备**——收集真实世界多领域多语言语料并预处理；(2) **候选生成**——基于语料用LLM迭代生成查询、正例和难负例；(3) **质量控制**——过滤低质量查询和修正错误标注。最终生成的数据集包含语料库 $\mathcal{D}$、查询集 $\mathcal{Q}$ 和相关性标签集 $\mathcal{R}$。

### 关键设计

1. **多阶段查询生成管线**:

    - 功能：从真实语料出发，自动生成多样、高质量的检索查询
    - 核心思路：六步迭代循环——(1) 从语料中采样正例文档 $d_i^+$；(2) 用LLM生成可能需要该文档的"角色"（character）；(3) 生成该角色可能使用文档的"场景"（scenario）；(4) 基于角色+场景生成原始查询 $ori\_q_i$，同时控制查询长度、类型、信息需求类型和表达风格；(5) 用LLM重写查询去除与原文重复的词汇，得到最终 $q_i$；(6) 基于查询和正例生成难负例文档 $\{d_i^{-}(j)\}$
    - 设计动机：角色+场景的中间步骤比直接生成查询更透明可控，且显著提升多样性；查询重写增加检索难度；难负例生成使评估更有区分度

2. **双层质量控制机制**:

    - 功能：过滤LLM生成的低质量数据，修正错误的相关性标签
    - 核心思路：两部分——(a) **低质量查询过滤**：用LLM判断 $q_i$ 与 $d_i^+$ 的相关性，预测为负则丢弃该查询；(b) **错误标签修正**：三步管线——用嵌入模型召回top-1000 $\to$ 多个重排序模型预标注（投票制）$\to$ LLM最终判断。对三类文档采取不同策略：原始正例（保留）、生成难负例被判正（丢弃）、语料库中被遗漏的正例（添加到 $\mathcal{D}_+$）
    - 设计动机：LLM生成不可避免产生低质量查询和错误标签，多模型投票+LLM验证的组合最大程度保证数据质量

3. **双任务异构评估设计**:

    - 功能：覆盖传统检索和现代RAG两种核心场景
    - 核心思路：(a) **QA任务**——经典问答检索，语料为大规模文档集，主指标nDCG@10；(b) **Long-Doc任务**——长文档分块检索，贴近RAG应用场景，主指标Recall@10（因为RAG中正例召回比排序更重要）。两类任务覆盖9大领域（新闻、Web、Wiki、科学、金融、医疗、法律、ArXiv、书籍）和13种语言
    - 设计动机：区分两种检索场景的评估需求——传统QA关注排序质量，RAG关注召回完整性

### 数据生成的设计考量

- **依赖真实语料**：基于真实世界语料生成，确保评估贴近实际场景且成本可控
- **查询重写**：变换形式但保留语义，增加检索难度
- **难负例生成**：提升评估的区分度
- **GPT-4 temperature=1.0**：鼓励更大多样性

## 实验关键数据

### 生成数据与人工标注数据一致性验证

以MS MARCO为例，对比原始人工标注（R-MSMARCO）和AIR-Bench生成（G-MSMARCO）：

| 数据集 | 语料规模 | 查询数 | 正例标签数 |
|--------|---------|--------|-----------|
| R-MSMARCO（人工） | 8,841,823 | 6,980 | 7,437 |
| G-MSMARCO（生成+质量控制） | 8,872,840 | 6,319 | 31,447 |
| G-MSMARCO（无质量控制） | 8,878,865 | 7,429 | 7,429 |

### 模型排名一致性

| 模型 | 参数量 | R-MSMARCO排名 | G-MSMARCO排名 | 排名一致性 |
|------|--------|-------------|-------------|-----------|
| repllama-v1-7b | 6.74B | 1 | 1 | ✅ |
| e5-large-v2 | 335M | 2 | 4 | ≈ |
| multilingual-e5-large | 560M | 3 | 5 | ≈ |
| bge-large-en-v1.5 | 335M | 5 | 2 | ≈ |

质量控制的关键作用：无质量控制时模型排名与人工标注严重不一致（如repllama排名从1降到2），质量控制后排名高度一致。

### 消融分析

| 消融项 | 影响 | 说明 |
|--------|------|------|
| 移除质量控制 | 模型排名一致性显著下降 | 验证质量控制模块不可或缺 |
| 移除查询重写 | 检索难度降低 | 重写增加了词汇多样性 |
| 移除难负例 | 评估区分度下降 | 难负例使基准更具挑战性 |
| 查询类型分布 | what最多(30-34%)，claim次之(22-26%) | 覆盖多种查询类型 |

### 关键发现

- **LLM生成的测试数据与人工标注高度一致**：关键前提是质量控制——无质量控制时数据质量和模型排名一致性都大幅下降
- **AIR-Bench对现有模型仍有区分度**：避免了MTEB等基准的饱和问题，因为生成数据不太可能被训练集覆盖
- **查询多样性高**：覆盖how/what/when/where/which/who/why/Yes-No/claim等多种类型，分布相对均衡
- **动态更新是核心优势**：24.04→24.05版本覆盖从较少到69个数据集，持续扩展中

## 亮点与洞察

- **"角色→场景→查询"三步生成比直接生成更可控更多样**：这个中间步骤的设计灵感来自于现实中不同用户有不同信息需求的事实，使得生成的查询不会陷入LLM的"默认模式"
- **质量控制用"嵌入召回+多模型投票+LLM判断"三级验证**：既不完全依赖LLM（可能与生成时共享偏差），也不完全依赖嵌入模型（可能有召回盲区），组合使用互补优势。这个管线设计可迁移到其他自动化评估基准的构建中
- **动态基准解决了AI评估的"军备竞赛"问题**：传统固定基准越来越容易被过拟合，动态生成+定期更新的模式使基准保持挑战性

## 局限与展望

- **生成质量依赖GPT-4能力边界**：在GPT-4理解较弱的领域（如高度专业的医学/法律术语）生成质量可能下降
- **仅二元相关性标注**：只判断相关/不相关，缺乏多级相关性标注（如ACORD的1-5星），对排序质量的评估精度有限
- **Long-Doc任务的分块策略未充分讨论**：不同分块方式对检索性能影响大，但论文未分析这一变量
- **LLM评估LLM的循环偏差风险**：用GPT-4生成数据评估可能偏好GPT-4系列嵌入模型
- **覆盖的13种语言中低资源语言代表不足**：未包含非洲、东南亚等低资源语言

## 相关工作与启发

- **vs BEIR (Thakur et al., 2021)**：BEIR聚合现有人工标注数据集做跨领域评估，但领域固定；AIR-Bench可自动扩展到任意新领域
- **vs MTEB (Muennighoff et al., 2023)**：MTEB是当前最流行的多任务嵌入基准，但面临饱和和数据泄露问题；AIR-Bench的动态特性天然抗过拟合
- **vs MIRACL (Zhang et al., 2023)**：MIRACL专注多语言检索但依赖人工标注（18种语言，成本极高）；AIR-Bench的自动化管线以极低成本覆盖13种语言且可持续扩展

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个完全自动化的异构IR基准，解决了传统基准的扩展性和时效性问题
- 实验充分度: ⭐⭐⭐⭐ 生成数据与人工标注一致性验证充分，消融分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，管线设计描述详尽，符号体系规范
- 价值: ⭐⭐⭐⭐ 为IR社区提供了可持续演进的评估平台，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](any_information_is_just_worth_one_single_screenshot_unifying_search_with_visuali.md)
- [\[ACL 2025\] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)

</div>

<!-- RELATED:END -->
