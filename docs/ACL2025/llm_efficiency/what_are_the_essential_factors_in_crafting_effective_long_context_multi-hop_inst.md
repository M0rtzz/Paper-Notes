---
title: >-
  [论文解读] What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices
description: >-
  [ACL 2025][LLM效率][长上下文] 提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。
tags:
  - ACL 2025
  - LLM效率
  - 长上下文
  - 多跳推理
  - 指令微调
  - 数据合成
  - 多智能体
---

# What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices

**会议**: ACL 2025  
**arXiv**: [2409.01893](https://arxiv.org/abs/2409.01893)  
**代码**: [WowCZ/LongMIT](https://github.com/WowCZ/LongMIT)  
**领域**: llm_efficiency  
**关键词**: 长上下文, 多跳推理, 指令微调, 数据合成, 多智能体

## 一句话总结

提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。

## 研究背景与动机

**领域现状**: 长上下文LLM在信息抽取、问答和规划等任务中表现突出，但仅扩展上下文窗口不足以有效利用长上下文，需要高质量指令微调数据来优化长上下文能力。

**现有痛点**: 现有Self-Instruct方法生成的长上下文数据质量差——以Qwen2-72B为例，多跳样本不足35%，超过40%质量低劣，且超过45%存在语义重复。

**核心矛盾**: 长上下文指令数据的人工标注成本极高，但自动合成方法难以保证多跳性、质量和多样性三者兼顾。

**本文目标**: 系统探究"构造高质量长上下文多跳指令数据集的关键因素是什么"——涵盖验证策略、生成策略、采样策略和合并策略。

**切入角度**: 借鉴多智能体协作范式，将数据合成流程分解为四个专职Agent模块，每个模块通过大量对比实验确定最优策略。

**核心 idea**: 用多智能体流水线逐步生成高质量多跳指令数据——先生成单跳问题，再采样组合，最后合并为多跳问题，全程由质量验证Agent把关。

## 方法详解

### 整体框架

MIMG框架包含四个核心模块：质量验证Agent（QVA）、单跳问题生成Agent（SQGA）、多问题采样策略（MQS）和多跳问题合并Agent（MQMA）。QVA全局监督各阶段数据质量，SQGA从单个文档生成单跳QA，MQS采样跨文档相关问题，MQMA将单跳问题合并为连贯的多跳问题。

### 关键设计

**模块一：质量验证Agent（QVA）**

- **功能**: 在数据合成全流程中评估和过滤低质量样本
- **核心思路**: 采用评分策略（scoring）而非分类策略（classification），通过阈值筛选高质量数据；验证条件包含多维度评分标准、辅助上下文信息和推理理由
- **设计动机**: 实验发现评分策略在长上下文场景下显著优于二分类（kappa和precision更高），且能缓解长上下文偏差；增加评分维度和加入rationale可进一步提升鲁棒性

**模块二：单跳问题生成Agent（SQGA）**

- **功能**: 从单个文档中生成多个高质量单跳问答对
- **核心思路**: 采用"先问题后答案"的生成顺序，而非一步生成完整QA；探索了不同规模LLM作为生成骨干
- **设计动机**: 分步生成显著提升数据质量，尤其对开源LLM提升明显；开源LLM在单跳生成上性价比高

**模块三：多问题采样策略（MQS）**

- **功能**: 从多个文档的单跳问题中采样语义相关问题，为多跳合并做准备
- **核心思路**: 使用BGE嵌入相似度构建问题-文档语义关联矩阵，基于问题相似度（而非文档相似度）采样；同时支持文档内和文档间采样
- **设计动机**: 嵌入相似度优于BM25和LDA；基于问题的采样显著优于基于文档的采样，因为问题提供了更精确的上下文线索

**模块四：多跳问题合并Agent（MQMA）**

- **功能**: 将多个采样的单跳问题合并为连贯的多跳问题
- **核心思路**: 仅使用QA对进行合并（不需要原始文档），开源LLM即可胜任；不使用rationale辅助合并
- **设计动机**: 实验发现加入文档不能一致提升性能且增加token消耗；合并阶段使用rationale反而导致模型对推理链的误解

### 损失函数/训练策略

基于合成的LongMIT数据集进行标准指令微调（instruction-tuning），在多个LLM（InternLM2-1.8B/7B、LLaMA3-8B等）上验证效果。

## 实验关键数据

### 主实验

在LongBench上通过GPT-4o评估的QA准确率：

| 模型 | 数据集 | NarrativeQA | 2WikiMQA | DuReader | HotpotQA | MultifieldQA-en | MultifieldQA-zh | MuSiQue | Qasper | AVG |
|---|---|---|---|---|---|---|---|---|---|---|
| InternLM2-1.8B | +LongAlign | 25.00 | 33.00 | 25.00 | 49.50 | 76.00 | 67.50 | 24.50 | 44.00 | 43.06 |
| InternLM2-1.8B | **+LongMIT** | **26.00** | **35.50** | **60.00** | **56.00** | **75.33** | **75.50** | **29.00** | **47.50** | **50.60** |
| LLaMA3-8B | +LongAlign | 29.00 | 44.50 | 56.50 | 56.50 | 79.33 | 80.50 | 21.50 | 55.50 | 52.92 |
| LLaMA3-8B | **+LongMIT** | **36.50** | **67.50** | **74.00** | **71.00** | **87.33** | **84.50** | **39.50** | **54.00** | **64.29** |
| InternLM2-7B | +LongAlign | 45.00 | 40.00 | 60.00 | 65.50 | 74.67 | 86.00 | 34.00 | 56.50 | 57.71 |
| InternLM2-7B | **+LongMIT** | **46.50** | **57.00** | **74.00** | **73.00** | **91.33** | **91.00** | **45.00** | **62.00** | **67.48** |

### 消融实验

MIMG各模块数据质量对比（基于Qwen2-72B生成200样本人工评估）：

| 策略 | 多跳比例(%) | 高质量比例(%) | 非重复比例(%) |
|---|---|---|---|
| 原始Self-Instruct | <35 | ~60 | ~55 |
| MIMG（完整） | >85 | >85 | >85 |

### 关键发现

- LongMIT训练的模型平均提升**7.54%**，在多跳基准（2WikiMQA、MuSiQue、HotpotQA）上提升尤为显著
- 合成高质量数据甚至超越更大规模人工标注数据集（如ChatQA2、LongAlign）
- 评分验证策略在长上下文场景下precision接近完美，但kappa较低——说明LLM不是好的标注者但是好的筛选者
- 增加多跳数（hop数）与模型性能正相关
- MIMG框架在输入端仅增加约3k token（2.5倍），输出端增加不到0.5k token，但质量提升近4倍

## 亮点与洞察

- **系统性极强**：通过17种策略、10个领域、5个LLM的大规模消融实验，逐一确定每个环节的最优策略，研究范式值得借鉴
- **关键洞察**："LLM不是长上下文标注者但是好的筛选者"——在长上下文场景下kappa低但precision高，这一发现对长上下文数据工程有重要指导意义
- **多跳合并不需要rationale**：与单跳生成不同，多跳合并阶段加入rationale反而有害，揭示了CoT在复杂合成场景中的局限

## 局限与展望

- 由于GPT-4级方法大规模蒸馏训练成本高，部分评估仅在200样本上进行人工评估，样本量有限
- 框架依赖多次LLM调用，端到端合成成本虽有优化但仍较高，未来可探索更高效的合成流程
- 仅关注QA任务形式，未探索摘要、信息抽取等其他长上下文任务类型
- 文档拼接生成长文本的方式与真实长文档场景存在分布差异

## 相关工作与启发

- 与Quest（Gao et al., 2024）最接近，但Quest构建闭源单跳QA，本文先建模文档关联再生成多跳QA
- 多智能体协作思路可扩展到其他数据合成场景（如多轮对话、代码生成数据）
- "先分解再组合"的多跳数据构造策略对RAG检索增强生成的数据构造也有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多智能体协作合成多跳数据的框架设计新颖，系统性探究各因素的贡献独到
- **实验充分度**: ⭐⭐⭐⭐⭐ 17种策略×10领域×5 LLM的消融实验极为全面，结论可信
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，每个实验发现都有明确归纳，但公式符号较多略显复杂
- **价值**: ⭐⭐⭐⭐ 对长上下文数据工程有直接的实践指导意义，LongMIT数据集和最佳实践可供社区复用

<!-- RELATED:START -->

## 相关论文

- [What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)
- [NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)
- [Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)

<!-- RELATED:END -->
