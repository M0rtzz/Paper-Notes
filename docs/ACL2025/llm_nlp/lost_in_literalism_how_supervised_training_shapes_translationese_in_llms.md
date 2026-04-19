---
title: >-
  [论文解读] Lost in Literalism: How Supervised Training Shapes Translationese in LLMs
description: >-
  [ACL 2025][LLM/NLP][机器翻译] 本文系统研究了大语言模型在机器翻译中产生翻译腔（translationese）的现象，揭示了监督微调（SFT）数据中的翻译腔偏差是导致LLM翻译不自然的根本原因，并提出了通过润色训练参考译文和过滤不自然训练实例来缓解翻译腔的方法。
tags:
  - ACL 2025
  - LLM/NLP
  - 机器翻译
  - 翻译腔
  - 监督微调
  - 数据质量
  - 自然度
---

# Lost in Literalism: How Supervised Training Shapes Translationese in LLMs

**会议**: ACL 2025  
**arXiv**: [2503.04369](https://arxiv.org/abs/2503.04369)  
**代码**: [github](https://github.com/yafuly/LLM_Translationese)  
**领域**: LLM/NLP  
**关键词**: 机器翻译, 翻译腔, 监督微调, 数据质量, 自然度

## 一句话总结

本文系统研究了大语言模型在机器翻译中产生翻译腔（translationese）的现象，揭示了监督微调（SFT）数据中的翻译腔偏差是导致LLM翻译不自然的根本原因，并提出了通过润色训练参考译文和过滤不自然训练实例来缓解翻译腔的方法。

## 研究背景与动机

尽管大语言模型在机器翻译任务上已经取得了卓越的表现，但"翻译腔"问题仍然普遍存在。翻译腔是指翻译文本在短语或句子层面过于直译，偏离目标语言的自然表达习惯，使得译文听起来对母语者不自然。

一个关键的矛盾在于：LLM在预训练阶段接触了大量的自然语言数据，理论上应该具备生成自然翻译的能力，但实际上仍然会产生不自然的译文。例如，将英语"suffer night blindness"翻译为中文时，模型生成了"遭受夜盲症"（直译），而非更自然的"患上夜盲症"。本文作者认为，这种"意料之外"的不自然翻译源于监督微调阶段引入的偏差。

更有趣的是，当让LLM自己"润色"已有的翻译时，它们能够产生明显更自然的输出，这进一步证实了LLM本身具备生成自然译文的潜力，但在"翻译"任务格式下这种潜力未被释放。

## 方法详解

### 整体框架

本文的工作分为三个核心部分：(1) 系统评估LLM翻译中的翻译腔现象；(2) 追溯翻译腔在监督训练数据中的根源；(3) 提出缓解翻译腔的训练策略。

### 关键设计

1. **翻译腔标注与量化（TSR指标）**: 作者收集了来自新闻、学术论文、维基百科和社交媒体四个写作领域的文档，使用ALMA、GPT-3.5/4、Mistral等模型进行英译中和德译英翻译。邀请三位专业翻译人员在Label Studio平台上标注翻译腔错误（包括不自然的句子流和不自然的短语流），计算翻译腔跨度比率（Translationese Span Ratio, TSR）来量化翻译腔的严重程度。

2. **困惑度与翻译腔的关联验证**: 使用Llama-3.1-8B计算翻译的困惑度（PPL），发现困惑度与人工标注的TSR呈正相关——困惑度越高，翻译腔越严重。这既验证了LLM本身偏好自然生成的假设，也提供了一种自动检测翻译腔的指标。

3. **训练数据翻译腔分析**: 从ALMA训练集中抽样500个英中和德英翻译实例，让专业翻译员标注翻译腔。结果揭示，超过34%的训练实例存在翻译腔模式（英中40.4%，德英34.2%），说明LLM在SFT过程中被引导将"翻译"理解为从源语言到目标语言的直接映射，过度强调忠实度而牺牲了自然度。

4. **SFT-Polished（润色训练参考）**: 利用GPT-4的"润色"能力来改善训练数据中的翻译参考。与直接让GPT-4翻译（SFT-KD）不同，SFT-Polished让GPT-4对现有翻译进行润色，保留原有的翻译质量同时提高自然度。这种方法的核心洞察是：LLM在"翻译"任务下可能产生翻译腔，但在"润色"任务下能够发挥其自然语言生成的优势。

5. **过滤不自然训练实例**: 以困惑度作为自然度度量，对训练实例排序后移除最不自然的子集。实验发现过滤20%的实例可以同时改善翻译自然度和翻译质量。

### 损失函数 / 训练策略

训练采用标准的监督微调（SFT）流程，基于ALMA的训练配置，使用WMT'17至WMT'21和Flores-200等平行训练数据（共31,621条实例）。基座模型选用Llama-3.1-8B和Qwen-2.5-7B。核心策略差异在于训练数据的预处理：
- **SFT**: 使用原始参考译文训练
- **SFT-KD**: 使用GPT-4直接翻译结果替换参考译文
- **SFT-Polished**: 使用GPT-4润色后的参考译文

## 实验关键数据

### 主实验

| 训练方法 | 指标 | Llama-3.1-8B (En-Zh) | Qwen-2.5-7B (En-Zh) |
|---------|------|----------------------|---------------------|
| SFT | PPL(Doc) | 13.8 | 13.8 |
| SFT-KD | PPL(Doc) | 14.3 | 13.9 |
| SFT-Polished | PPL(Doc) | **11.9** | **12.1** |
| SFT | PPL(Sent) | 103.3 | 101.6 |
| SFT-Polished | PPL(Sent) | **90.0** | **87.3** |

翻译质量评估（COMET-QE）：
| 训练方法 | Llama En-Zh | Llama De-En | Qwen En-Zh | Qwen De-En |
|---------|------------|------------|------------|------------|
| SFT | 80.0 | 80.5 | 73.8 | 74.0 |
| SFT-Polished | **81.8** | 81.0 | 74.2 | **75.6** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 过滤0%（基线） | PPL基线 | 无过滤 |
| 过滤20% | 自然度↑ 翻译质量↑ | 最佳平衡点 |
| 过滤40% | 自然度↑ 翻译质量↓ | 自然度持续改善但质量开始下降 |
| SFT-KD vs SFT-Polished | KD无改善，Polished显著改善 | 证明关键在于润色而非蒸馏 |

### 关键发现

- 所有LLM都存在显著的翻译腔，即使GPT-4也有超过40%的翻译表现出明显的翻译腔模式
- 让LLM自己润色翻译（GPT-4 Polishing）可将翻译腔文档比例从43%降至25%
- 句子级翻译腔比短语级翻译腔更普遍（标注计数约2:1）
- 简单在prompt中加入风格要求（Specified）无法有效减少翻译腔，甚至可能恶化
- SFT-Polished在两个基座模型和两个翻译方向上均一致性地改善翻译自然度
- 训练阶段的润色比推理阶段的后处理润色（Post-Polishing）效果更好

## 亮点与洞察

1. **深刻洞察**：翻译腔问题的根源不在于LLM的能力不足，而在于SFT阶段训练数据的偏差。LLM本身通过预训练获得了生成自然文本的能力，但"翻译"任务格式激活了过于强调忠实度的直译模式。
2. **简单有效的解决方案**：润色训练数据这一方法既直观又有效，无需修改模型架构或训练流程。
3. **困惑度作为翻译腔指标**：建立了困惑度与翻译腔之间的定量关联，为自动检测翻译腔提供了实用工具。
4. **任务格式的重要性**：发现"翻译"和"润色"两种任务格式对LLM生成自然度有本质不同的影响，这启示我们任务framing的重要性。

## 局限与展望

- 主要实验在英中和德英两个翻译方向上进行，虽然有更多语言的泛化实验，但覆盖范围仍有限
- 润色依赖GPT-4等强模型，增加了数据准备的成本
- 困惑度作为翻译腔指标虽然有效，但可能与其他文本质量维度存在混淆
- 未深入探讨不同类型翻译腔（句子级vs短语级）的差异化处理策略
- 过滤不自然实例可能同时移除了有价值的训练信号

## 相关工作与启发

- 本文将传统机器翻译质量研究中的翻译腔概念引入LLM时代，是首个系统研究LLM翻译腔的工作
- 与ALMA等翻译专用LLM的工作互补，关注翻译风格而非仅关注翻译准确度
- 对数据中心的AI（Data-centric AI）范式有启发意义：训练数据质量不仅影响准确性，也影响生成风格
- 为SFT数据清洗和预处理提供了新的视角和方法论

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究LLM翻译腔问题，视角独特但解决方案相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖人工评估、自动指标、多模型、多语言方向的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑清晰，从现象观察到原因分析再到解决方案，层层递进
- 价值: ⭐⭐⭐⭐ 揭示了LLM翻译中一个被忽视的重要问题，方法实用且易于复现

<!-- RELATED:START -->

## 相关论文

- [How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

<!-- RELATED:END -->
