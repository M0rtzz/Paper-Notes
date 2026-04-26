---
title: >-
  [论文解读] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark
description: >-
  [ACL 2025 (Findings)][挪威语评测] 本文提出 NorEval，一个包含 24 个人工创建数据集、覆盖 9 类任务的挪威语综合评测套件，系统地评测了 19 个开源挪威语语言模型在语言理解和生成上的能力，发现模型在常识推理、真实性和指令遵循上仍显著落后于人类。
tags:
  - ACL 2025 (Findings)
  - 挪威语评测
  - 语言模型基准
  - 低资源语言
  - 多任务评估
  - 人类基线
---

# NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark

**会议**: ACL 2025 (Findings)  
**arXiv**: [2504.07749](https://arxiv.org/abs/2504.07749)  
**代码**: https://github.com/ltgoslo/noreval  
**领域**: NLP理解 / 评测基准  
**关键词**: 挪威语评测、语言模型基准、低资源语言、多任务评估、人类基线

## 一句话总结

本文提出 NorEval，一个包含 24 个人工创建数据集、覆盖 9 类任务的挪威语综合评测套件，系统地评测了 19 个开源挪威语语言模型在语言理解和生成上的能力，发现模型在常识推理、真实性和指令遵循上仍显著落后于人类。

## 研究背景与动机

**领域现状**：语言模型的进步离不开标准化基准测试，但低资源语言（如挪威语）的评测资源严重不足。现有的挪威语基准如 NorBench、ScandEval、SEB 和 NLEBench 各有侧重，但覆盖范围有限。

**现有痛点**：现有基准存在四大问题：(1) 任务覆盖率低且数据集高度重叠；(2) NLEBench 和 ScandEval 包含未经校对的机器翻译数据集，引入评测偏差；(3) 挪威语的少数书写标准 Nynorsk 严重被忽视，现有基准几乎只覆盖 Bokmål；(4) 没有基准建立过人类基线来衡量模型上界。

**核心矛盾**：要全面评估挪威语生成式语言模型的能力，需要一个覆盖理解与生成、同时关注两种官方书写标准（Bokmål 和 Nynorsk）、以高质量人工数据为核心的大规模基准，但目前这样的资源完全缺失。

**本文目标**：构建一个全面、高质量的挪威语评测套件，覆盖广泛的任务类别，建立人类基线，并为 Nynorsk 提供充分的评测支持。

**切入角度**：从零开始创建 5 个新数据集，整合 19 个已有同行评审数据集，编写 100+ 人工撰写的提示词，全部集成到 LM Evaluation Harness 框架中，确保可复现评测。

**核心 idea**：打造挪威语最大多任务基准——24 个数据集覆盖 9 类任务，同时关注 Bokmål 和 Nynorsk，附带人类基线和 LLM-as-a-judge 评估。

## 方法详解

### 整体框架

NorEval 是一个评测套件而非模型方法。其整体设计流程是：(1) 收集和创建 24 个数据集，覆盖 9 类任务；(2) 为每个数据集编写 4-6 个 Bokmål 和 Nynorsk 提示词；(3) 全部集成到 LM Evaluation Harness；(4) 在 k-shot（k∈{0,1,16}）场景下评测 19 个模型；(5) 建立 5 个任务的人类基线；(6) 用 LLM-as-a-judge 评估指令遵循能力。

### 关键设计

1. **九类任务全覆盖的数据集架构**:

    - 功能：构建覆盖挪威语理解和生成的全面评测体系
    - 核心思路：将任务分为 9 个高层类别——情感分析、挪威语语言知识（语法纠错、标点、成语）、挪威国情/世界知识（多选题 QA）、阅读理解、常识推理、机器翻译、文本摘要、指令遵循、真实性。24 个数据集中，16 个覆盖 Bokmål，8 个覆盖 Nynorsk。5 个全新数据集包括 NCB（标点基准）、NorIdiom（成语补全）、NorRewrite-Instruct、NorSummarize-Instruct 和系列 QA 数据集
    - 设计动机：现有基准任务类型单一且重叠严重，NorEval 需要填补挪威语语言知识、真实性、指令遵循等评测空白

2. **多提示词 + 双书写标准评测策略**:

    - 功能：减少模型对特定提示词表述的敏感性，同时评估 Bokmål 和 Nynorsk
    - 核心思路：通过两阶段标注流程创建 100+ 提示词。Stage 1 由 3 位母语者手动翻译/撰写 Bokmål 提示词，Stage 2 由语言学专业学生将其适配为 Nynorsk。评测时对每个模型取所有提示词中的最高分来减轻提示敏感性
    - 设计动机：研究表明 prompt 的表述方式显著影响 LM 表现，多提示策略结合取最优可以得到更可靠的评估

3. **混合性能聚合方法**:

    - 功能：将多任务异构指标合理聚合为总分
    - 核心思路：采用三种互补方法——(1) 多提示聚合（取最高分）；(2) 归一化平均分（将各任务分数在随机基线和上限之间归一化后取平均）；(3) Borda 计数（基于排序的社会选择理论方法，对每个任务按模型排名赋分再求和），替代简单算术平均来处理异构评测指标的问题
    - 设计动机：传统的平均聚合无法公平对待不同量纲的指标，Borda 计数基于排名而非绝对分数更鲁棒，两种方法互相验证

### 损失函数 / 训练策略

本文是评测工作，不涉及模型训练。评测策略包括：log-likelihood 方式（用于分类/多选题任务，选概率最高的选项）和 generation 方式（用于生成任务，贪心解码或遵循 HuggingFace 推荐超参）。LLM-as-a-judge 使用 Llama-3.3-70B-Instruct 做裁判，采用 HREF 框架配合人类参考答案。

## 实验关键数据

### 主实验

| 模型 | Borda总分 | 情感分析 | 语言知识 | 知识QA | 阅读理解 | 常识推理 | 翻译 | 摘要 | 指令遵循 | 真实性 |
|------|-----------|---------|---------|--------|---------|---------|------|------|---------|--------|
| NorMistral-11B | 54.4 | 82.2 | 94.0 | 64.7 | 43.0 | 59.5 | 45.4 | 23.4 | 46.3 | 73.4 |
| AI-Sweden/Llama-3-8B | 51.3 | 80.3 | 84.0 | 54.8 | 51.0 | 47.1 | 34.8 | 31.4 | 38.1 | 71.5 |
| Mistral-Nemo-12B-IT | 52.1 | 82.9 | 33.0 | 58.8 | 16.1 | 67.3 | 44.1 | 42.7 | 55.7 | 43.7 |
| NB-GPT-6B | 33.0 | 34.2 | 42.0 | 29.6 | 30.6 | 7.8 | 27.9 | 33.0 | 39.1 | 55.1 |
| 人类基线 | — | — | 92.0 | — | 90.0 | — | — | — | — | 83.3 |

### 消融实验 — 模型 vs 人类基线

| 任务 | 最佳模型 | 最佳模型得分 | 人类基线 | 差距 |
|------|---------|------------|---------|------|
| Belebele (阅读理解) | Mistral-Nemo-12B-IT | 80.2 | 90.0 | -9.8% |
| NorOpenBookQA (世界知识) | AI-Sweden/Llama-3-8B-IT | 84.8 | 100.0 | -15.2% |
| NorCommonsenseQA (常识推理) | AI-Sweden/Llama-3-8B-IT | 72.2 | 90.0 | -17.8% |
| NorTruthfulQA MC (真实性) | Mistral-7B | 74.6 | 83.3 | -8.7% |
| NCB (标点知识) | NorwAI-Llama2-7B | 90.0 | 88.0 | +2.0% |

### 关键发现

- **没有单一模型在所有任务类别上一致领先**。NorMistral-11B 预训练模型总体最强，AI-Sweden/Llama-3-8B 紧随其后，但指令微调效果因任务而异
- **Bokmål 通常优于 Nynorsk**：在知识问答和常识推理任务上，模型在 Bokmål 上的表现系统性地高于 Nynorsk，但在 NRK-Quiz-QA 和 NorIdiom 上反而 Nynorsk 更好
- **指令微调的双刃剑效应**：IT 版本在多选 QA 和序列生成任务上提升明显，但在挪威语语言知识（尤其是成语补全）和英翻挪翻译上会退化——AI-Sweden/Llama-3-8B-IT 的 NorIdiom 分数直接从 31.3 降到 0.0
- **语言偏移问题严重**：在指令遵循评测中，只有 NorMistral-7B-warm-IT 能稳定用挪威语回答，其他模型（尤其 Mistral-7B-IT 和 Meta/Llama-3-8B-IT）高达 60-66% 的回复是英文

## 亮点与洞察

- **首个全面的挪威语评测套件**：24 个人工创建数据集 + 100+ 人工提示词 + 5 个人类基线，对低资源语言评测方法论有模板意义。这种"从数据到提示到评测框架"的完整设计可以迁移到其他低资源语言
- **Borda 计数替代简单平均**：利用社会选择理论中的排序投票方法来聚合异构评测指标，比算术平均更鲁棒。这个思路在任何需要聚合多个异质指标的评测场景中都有参考价值
- **指令微调的 Nynorsk 退化现象**：数据揭示了一个重要的实践问题——英文指令微调数据会让模型丧失少数语言变体的能力，这对多方言/多变体语言的模型开发有警示作用

## 局限与展望

- **缺少测试数据去污染机制**：作者承认模型的预训练语料可能包含 NorEval 测试数据，这会高估模型性能
- **人类基线覆盖有限**：只在 5 个 Bokmål 任务上建立了基线，每个基线仅 50 个样本，没有 Nynorsk 人类基线
- **LLM-as-a-judge 在低资源语言的可靠性未验证**：用英文训练的大模型来评判挪威语生成质量是否可靠仍需进一步研究
- 模型规模限制在 7-13B，缺少更大模型（如 GPT-4）的参照。未来可以加入闭源大模型的评测结果作为上界参考

## 相关工作与启发

- **vs NorBench**: NorBench 只覆盖 10 个传统 NLP 任务（POS、NER 等），NorEval 扩展到 24 个数据集和 9 类任务，尤其增加了真实性、指令遵循等生成式评测
- **vs ScandEval**: ScandEval 是跨北欧语言的基准，数据集有重叠但包含机器翻译数据。NorEval 几乎全部使用人工创建数据，质量更有保障
- **vs NLEBench**: NLEBench 关注挪威语生成能力但完全不覆盖 Nynorsk，且 7/9 数据集是未校对的机翻。NorEval 在数据质量和语言多样性上均有优势

## 评分

- 新颖性: ⭐⭐⭐ 评测基准工作创新性有限，但对挪威语 NLP 社区有重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 19 个模型、24 个数据集、多种评测场景、人类基线、偏差分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据详实，附录极其完善
- 价值: ⭐⭐⭐⭐ 填补了挪威语评测空白，方法论可迁移到其他低资源语言

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

<!-- RELATED:END -->
