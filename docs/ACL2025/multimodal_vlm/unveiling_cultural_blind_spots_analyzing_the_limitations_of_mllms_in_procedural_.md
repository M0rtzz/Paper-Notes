---
title: >-
  [论文解读] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension
description: >-
  [ACL 2025][多模态][多语言大模型] 提出 CAPTex 基准，通过跨 7 个国家/语言的文化程序性文本理解任务（步骤排序、选择题、对话推理等），系统揭示了多语言大模型在文化特定程序性文本理解上的盲区和局限。
tags:
  - ACL 2025
  - 多模态
  - 多语言大模型
  - 程序性文本
  - 文化理解
  - 基准评测
  - 低资源语言
---

# Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension

**会议**: ACL 2025  
**arXiv**: [2502.14315](https://arxiv.org/abs/2502.14315)  
**代码**: [HuggingFace](https://huggingface.co/datasets/AmirHossein2002/CAPTex)  
**领域**: 多语言 / 文化理解  
**关键词**: 多语言大模型, 程序性文本, 文化理解, 基准评测, 低资源语言

## 一句话总结

提出 CAPTex 基准，通过跨 7 个国家/语言的文化程序性文本理解任务（步骤排序、选择题、对话推理等），系统揭示了多语言大模型在文化特定程序性文本理解上的盲区和局限。

## 研究背景与动机

尽管多语言大模型（mLLMs）在多种 NLP 任务中表现优异，但其理解程序性文本（按步骤顺序的指导性文本）的能力，尤其是涉及特定文化内容的程序性文本，仍未被充分探索。存在以下关键问题：

1. **文化嵌入的隐性知识**：程序性文本中嵌入了大量文化特定的默认知识，如宗教仪式步骤、传统工艺流程等，不同文化的同类程序差异巨大（如伊朗与印尼的葬礼程序完全不同）
2. **WEIRD 偏差**：现有 LLM 倾向于西方、受教育、工业化、富裕和民主社会的文化规范，对其他文化语境的表示不足
3. **低资源语言的挑战**：在低资源语言中，模型对程序性文本的理解能力更弱
4. **缺乏评估基准**：之前的工作主要局限于食谱领域或英语单语言，缺乏跨文化、多语言的全面评估基准

核心研究问题：mLLM 能否正确理解和推理嵌入文化语境的程序性文本？

## 方法详解

### 整体框架

CAPTex (Culturally-Aware Procedural Texts) 是一个多任务评估基准，包含三个基础组件：(1) 来自 7 个文化区域的程序性文本集合（10 个类别），(2) 精心设计的选择题，(3) 基于程序的对话语料。评估覆盖 4 种任务形式。

### 关键设计

1. **数据构建——文化原真性保障**：每种语言/文化由两名本地母语者手工撰写程序性文本（严禁使用 AI 生成工具），覆盖 7 个国家（中国、印度、印尼、伊朗、日本、尼日利亚、巴基斯坦）和 10 个文化领域（食物烹饪、庆典节日、社交礼仪、手工艺、传统服饰、农业实践、宗教仪式、人生里程碑、体育竞技、自然实践）。每国每领域 10 个程序，共 1,400 篇程序文本（含母语版和英语译文）。质量控制包括自动检查和交叉人工审核。

2. **四种评估任务**：(Task 1) 步骤排序——打乱步骤顺序后让模型重建正确顺序，用 Spearman 相关、Levenshtein 距离和 Kendall's Tau 评估；(Task 2) PB-MCQ——基于程序的选择题，覆盖后续/前置步骤的肯定/否定四种变体，共 5,600 道题；(Task 3) CB-MCQ——基于对话框架的选择题，模拟两人对话中的程序性知识问答；(Task 4) CB-QA——类似 CB-MCQ 但要求生成回答而非选择，用 ROUGE-L、BERTScore 和语义相似度评估。

3. **语言资源多样性设计**：选择的 7 种语言覆盖从"Winners"（中文、日语）到"Hopefuls"（豪萨语）的不同资源层级，确保系统评估 mLLM 在不同资源水平下的表现。

### 损失函数 / 训练策略

本文是评估性工作，不涉及模型训练。评估采用零样本设置，全部使用英语 prompt 模板（因为研究表明英语 prompt 通常能获得最佳性能）。共评估了 31 个模型，包括 DeepSeek-R1、Gemma-2、Llama-3、Qwen2.5、GPT-4o、O3-mini 等。

## 实验关键数据

### 主实验

| 模型 | 排序 ρ↑ | PB-MCQ Acc | CB-MCQ Acc | CB-QA (R-L) |
|------|---------|------------|------------|-------------|
| Random | 0.00 | 0.25 | 0.25 | 0.00 |
| GPT-4o | 最佳 | 最佳 | 最佳 | 最佳 |
| Gemma-2-9b-it | 0.75 | 0.43 | 0.46 | 0.50 |
| Qwen2.5-14B-Instruct | - | 最佳开源 | - | - |
| Qwen2.5-7B | - | - | - | 最佳开源 |
| Llama-3.1-8B-Instruct | 0.43 | 0.37 | 0.48 | 0.59 |
| DeepSeek-R1-Qwen-14B | 0.43 | 0.42 | 0.54 | 0.55 |
| Mamba | 显著落后 | 显著落后 | 显著落后 | 显著落后 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 英语 vs 母语 | 英语性能一致优于母语 | 低资源语言差距更大 |
| PB-MCQ vs CB-MCQ | CB-MCQ 通常更高 | 对话框架有助于模型推理 |
| 不同文化领域 | 某些领域（宗教仪式）更难 | 文化嵌入越深越难 |
| Instruct vs Base | Instruct 通常更好 | 例外：Gemma-2-2B 和 Qwen2.5-1.5B |

### 关键发现

- mLLM 在文化程序性文本上表现欠佳，低资源语言下性能显著下降
- 不同文化领域的模型表现波动大，部分领域（如宗教仪式、传统手工艺）难度更高
- 对话框架中的选择题表现优于直接提问，说明对话上下文有助于模型推理
- Mamba 架构在程序性文本理解上显著弱于 Transformer 架构
- 排序、PB-MCQ、CB-MCQ 三个任务高度相关（Kendall 0.8-0.9），但 CB-QA 生成任务相关性较低（0.4-0.5），捕捉了不同能力维度
- 增大参数量在同系列模型内通常能提升性能
- 四个任务间 Kendall 相关分析表明排序和选择类任务衡量相似能力，生成任务衡量不同能力

## 亮点与洞察

- **真正的文化多样性**：数据集由 7 国本地人手工编写，确保文化原真性，不同于机器生成或翻译的数据
- **多维评估设计**：4 种任务形式覆盖排序理解、知识回忆、对话推理和自由生成，全面刻画了模型的程序性文本理解能力
- **对话框架的启发**：模型在对话上下文中的选择题表现优于直接提问，暗示适当的对话框架可以激活模型的隐性文化知识
- **资源级别与表现的关系**：系统验证了语言资源量与模型文化理解能力的正相关

## 局限与展望

- 仅覆盖 7 个国家/文化，亚洲偏重（中国、印度、日本、巴基斯坦），缺少拉美、欧洲非英语国家和撒哈拉以南非洲更多代表
- 程序性文本的步骤数固定在 5-10 步，未涉及更复杂的长序列程序
- 评估仅使用英语 prompt，不同语言 prompt 可能影响结果
- 未探索 few-shot 设置或 RAG 增强下的表现
- 对话部分由 GPT-4o 生成（虽经人工审核），可能引入风格偏差

## 相关工作与启发

- 与 IndoCulture (Koto et al., 2024) 的分类法互补扩展
- 菜谱跨文化适配 (Cao et al., 2023) 仅限食物领域，本文扩展到 10 个文化领域
- 对多语言 LLM 的评估从通用能力深入到文化特定推理能力
- 启发：文化程序性知识可作为 LLM 对齐和知识增强的重要维度

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3 | 基准构建工作，任务设计有新意但方法创新有限 |
| 实用性 | 4 | 为多语言/多文化 LLM 评估提供重要基准 |
| 实验充分度 | 4 | 31 个模型、4 种任务、7 种语言的全面评估 |
| 写作质量 | 4 | 结构清晰，分析细致 |
| 总分 | 3.5 | 有价值的评估工作，揭示了 mLLM 文化理解的重要盲区 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](punchbench_mllm_punchline.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ACL 2025\] MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimm_cultural_metaphor.md)
- [\[ACL 2025\] CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](cosyn_code_guided_synthetic_data.md)

<!-- RELATED:END -->
