---
title: >-
  [论文解读] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?
description: >-
  [ACL 2025 (Short Paper)][LLM/NLP][医学问答] 本文探索使用LLM自动生成医学问答的答案关键点（information nuggets），对比多种生成方法与人类专家标注的一致性，发现提供示例+从答案中提取nugget的方法效果最好，但LLM提取原子事实的能力仍然有限…
tags:
  - "ACL 2025 (Short Paper)"
  - "LLM/NLP"
  - "医学问答"
  - "信息nugget"
  - "答案关键点"
  - "LLM评估"
  - "事实提取"
---

# Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?

**会议**: ACL 2025 (Short Paper)  
**arXiv**: N/A  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-short.28/)
**代码**: 无  
**领域**: 医学NLP / 问答系统 / 事实性评估  
**关键词**: 医学问答, 信息nugget, 答案关键点, LLM评估, 事实提取

## 一句话总结

本文探索使用LLM自动生成医学问答的答案关键点（information nuggets），对比多种生成方法与人类专家标注的一致性，发现提供示例+从答案中提取nugget的方法效果最好，但LLM提取原子事实的能力仍然有限，其中Llama 3.3表现最佳。

## 研究背景与动机

**领域现状**：评估LLM生成文本的事实性是NLP的核心挑战，在医学问答领域尤为重要，因为错误信息可能直接影响患者健康。"信息nugget"是一种评估方法——将正确答案分解为原子事实（atomic facts），然后检查被评估文本是否包含这些原子事实。

**现有痛点**：人工提取nugget成本高昂且耗时，不适合大规模评估。近期的RAG评测共享任务开始探索用LLM自动化nugget提取，但在医学领域的效果未被系统评估。医学nugget要求高度精确——一个模糊或错误的nugget可能导致对医学回答的误判。

**核心矛盾**：自动化nugget生成能大幅降低成本，但医学领域对精确性的要求极高，LLM可能无法可靠地从医学文本中提取精确的原子事实。

**本文目标**：(1) 评估多种LLM自动生成医学nugget的方法；(2) 与人类专家标注对比，量化自动方法的可靠性。

**切入角度**：从RAG评测实践出发，系统比较不同的nugget生成策略（如从问题生成 vs 从答案提取，零样本 vs 少样本等），在医学问答场景下建立可靠性基准。

**核心 idea**：医学问答的自动化评估需要可靠的答案关键点，本文系统评估了LLM在这一任务上的能力边界。

## 方法详解

### 整体框架

输入是医学问题及其参考答案，输出是一组information nuggets（原子事实列表）。评估指标是自动生成的nuggets与人类专家标注的nuggets之间的对齐程度。

### 关键设计

1. **多种Nugget生成策略**:

    - 功能：探索不同的nugget生成方法
    - 核心思路：设计了多种生成策略：(a) 从问题生成（question-based）——给定问题，让LLM生成应该包含的关键事实点；(b) 从答案提取（answer-based）——给定参考答案，让LLM提取其中的原子事实；(c) 混合方法——先从问题生成候选，再用答案验证和补充。每种方法又分为零样本和少样本（提供人工nugget示例）设置
    - 设计动机：不同策略各有优劣——从问题生成可能遗漏答案中的细节，从答案提取可能被答案的措辞影响

2. **人类专家Nugget标注**:

    - 功能：建立评估的金标准
    - 核心思路：由医学信息学专家（NIH/NLM的研究人员）手动为医学问题创建nugget。每个nugget是一个简短的事实陈述，且是"原子的"——不应可以进一步分解为更小的独立事实。标注过程遵循严格的指南，包含粒度控制（不能太粗也不能太细）和医学准确性审核
    - 设计动机：高质量的人工标注是评估自动方法不可或缺的参照

3. **对齐度评估框架**:

    - 功能：量化自动nuggets与人工nuggets的一致性
    - 核心思路：使用两种评估方式——(a) 语义相似度匹配，计算自动nugget集合与人工nugget集合的覆盖率和精确率；(b) 人工评判，让评估者判断自动生成的每个nugget是否与某个人工nugget语义等价。报告Precision（生成的nugget中有多少是正确的）、Recall（人工nugget中有多少被覆盖）和F1
    - 设计动机：纯自动的语义匹配可能不够精确（医学术语的同义表达复杂），需要人工评判来验证

### 实验设置

测试了GPT-4、GPT-3.5、Llama 3.3等多个模型。使用来自TREC Health Misinformation Track和BioASQ的医学问答数据。

## 实验关键数据

### 主实验

| 方法 | 模型 | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| 从答案提取+少样本 | Llama 3.3 | 最优 | 高 | 最优 |
| 从答案提取+零样本 | GPT-4 | 高 | 中等 | 次优 |
| 从问题生成+少样本 | GPT-4 | 较高 | 低 | 中等 |
| 从问题生成+零样本 | GPT-4 | 中等 | 低 | 较低 |
| 混合方法 | GPT-4 | 中等 | 中等 | 中等 |

### 消融实验

| 因素 | 影响 | 说明 |
|------|------|------|
| 少样本 vs 零样本 | 少样本显著优于零样本 | 示例帮助模型理解nugget粒度 |
| 从答案 vs 从问题 | 从答案更优 | 答案中包含了具体事实 |
| nugget粒度控制 | 差异显著 | 过粗或过细都降低对齐度 |
| Llama 3.3 vs GPT-4 | Llama 3.3 略优 | 可能与特定提示风格有关 |

### 关键发现
- 提供示例并从答案中提取nugget的策略效果最好——示例帮助模型把握正确的粒度
- LLM生成的nugget存在系统性问题：(1) 粒度不一致——有些过于详细，有些过于笼统；(2) 有时将多个事实合并为一个nugget；(3) 偶尔引入原答案中不存在的信息
- Llama 3.3在该任务上表现最好，可能因为其对指令的遵循更精确
- 总体而言，LLM从文本中提取原子事实的能力仍然**有限**——这是一个值得进一步研究的方向

## 亮点与洞察
- **将nugget生成任务作为LLM能力的测试台**是一个有意义的视角——原子事实提取能力是LLM做事实核查、文本评估的基础能力
- **医学领域的高精度要求**使得该任务成为检验LLM可靠性的良好基准——不同于一般领域，医学nugget的错误是不可接受的
- 发现"从答案提取优于从问题生成"具有实用启示——在设计评估流程时应优先使用答案驱动的方法

## 局限与展望
- 作为Short Paper，实验规模有限（数据集较小，模型数量有限）
- 未评估nugget质量对下游评估任务（如RAG系统评测）的实际影响
- 医学nugget的定义和粒度标准可能因场景而异（如面向患者 vs 面向医生）
- 未来可探索让LLM进行多轮迭代优化nugget质量

## 相关工作与启发
- **vs ARES**: ARES使用LLM评估RAG系统但不显式提取nugget，本文的nugget方法提供了更细粒度的评估
- **vs FActScore**: FActScore提出了从生成文本中提取原子主张并验证的框架，本文将类似思路应用到医学问答评估中
- **vs BioASQ**: BioASQ是医学问答评测的主要基准，本文的nugget方法可作为BioASQ新的评估补充

## 评分
- 新颖性: ⭐⭐⭐ 方法上是对已有nugget提取方法的系统比较，创新性一般
- 实验充分度: ⭐⭐⭐ Short Paper限制了实验规模，但对比维度设计合理
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验分析到位
- 价值: ⭐⭐⭐⭐ 对医学NLP评估方法的可靠性研究有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)

</div>

<!-- RELATED:END -->
