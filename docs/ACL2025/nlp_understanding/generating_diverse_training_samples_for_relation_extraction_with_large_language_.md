---
title: >-
  [论文解读] Generating Diverse Training Samples for Relation Extraction with Large Language Models
description: >-
  [ACL 2025][NLP理解][关系抽取] 研究如何用LLM为关系抽取（RE）生成高质量且多样化的训练样本，提出基于ICL的逐条生成策略和基于DPO的多样性微调方法，生成的训练数据可有效提升few-shot RE模型性能。
tags:
  - ACL 2025
  - NLP理解
  - 关系抽取
  - 数据增强
  - LLM生成训练数据
  - DPO微调
  - 多样性
---

# Generating Diverse Training Samples for Relation Extraction with Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.23108](https://arxiv.org/abs/2505.23108)  
**代码**: [github.com/Lzx-ZBC](https://github.com/Lzx-ZBC) (有)  
**领域**: NLP理解  
**关键词**: 关系抽取, 数据增强, LLM生成训练数据, DPO微调, 多样性  

## 一句话总结

研究如何用LLM为关系抽取（RE）生成高质量且多样化的训练样本，提出基于ICL的逐条生成策略和基于DPO的多样性微调方法，生成的训练数据可有效提升few-shot RE模型性能。

## 研究背景与动机

**领域现状**: 关系抽取（RE）是信息抽取的核心任务，用于从文本中识别实体对之间的关系类型。基于PLM的RE模型（如KnowPrompt、RetrievalRE）在有充足标注数据时表现良好，但实际应用中标注数据稀缺是普遍问题。

**现有痛点**: 直接用LLM生成RE训练样本时，生成结果呈现**高度结构相似性**——LLM倾向于使用有限的动词短语和几乎相同的句子结构来表达实体间关系。例如对"死于某国"关系，LLM总是反复生成"died in [Country]"这类模式。

**核心矛盾**: LLM的强大生成能力与其生成结果的低多样性之间存在矛盾。生成的训练样本如果高度相似，不仅浪费了计算资源，而且训练出的RE模型泛化能力也会受限。

**本文目标**: 如何在保证正确性的前提下，有效提升LLM生成的RE训练样本的多样性。

**切入角度**: 双管齐下——先通过ICL策略（逐条生成+多样性指令）提升生成多样性，再用DPO微调让LLM内化"多样化生成"偏好。

**核心 idea**: 通过构造"相似/错误样本为负例、真实标注为正例"的偏好数据，用DPO微调LLM，使其在生成RE训练样本时同时考虑多样性和正确性。

## 方法详解

### 整体框架

方法分为两部分：
1. **ICL策略**: 设计包含Task Description、Relation Explanation、Sample Demonstration三个模块的prompt，支持All-At-Once (AAO)和One-By-One (OBO)两种生成模式
2. **DPO微调**: 构造偏好数据对，用DPO优化LLM的生成策略

### 关键设计

**逐条生成策略 (One-By-One, OBO)**: 每次只让LLM生成一条训练样本，然后将其加入Prompt的Sample Demonstration Module中。下次生成时，LLM可以看到所有已生成的样本，被指令要求生成与已有样本"尽可能不同"的新样本。这种递增式策略比一次性生成所有样本(AAO)更能保证多样性。

**DPO微调目标**: 给定偏好数据集 $\mathcal{D} = \{(x_i, y_{i,1}, y_{i,2})\}$，优化：

$$\max_{\pi} \Sigma_i \log \sigma\left(\frac{1}{\beta} \log \frac{\pi(y_{i,1}|x_i)}{\pi(y_{i,2}|x_i)}\right)$$

其中 $y_{i,1}$ 是偏好数据（从真实标注集中随机采样), $y_{i,2}$ 是非偏好数据。

**非偏好数据的三种构造方式**:
1. **错误标注**: 取其他关系类型的样本，把标签改为目标关系——构造标注错误的样本
2. **高相似**: 取Prompt中的demonstration样本，替换头尾实体并微调上下文——构造过于相似的样本
3. **直接复制**: 与demonstration样本完全相同——构造完全重复的样本

**模拟OBO过程**: DPO训练数据中，Sample Demonstration Module的样本数量逐步递增，将前一条训练数据的正例输出放入下一条的prompt中，模拟真实的逐条生成过程。

### 损失函数/训练策略

- 底层LLM: LLaMA2-7b-Chat，使用LoRA进行DPO微调
- 超参数: temperature=0.4, top_p=0.9, top_k=20, repetition_penalty=1.15
- DPO微调: truncation length=1024, lr=5e-5, batch size=4, epoch=20
- **关键防作弊设计**: 将关系类别一分为二，一半用于DPO微调、另一半用于生成样本，交换后完成所有类别

## 实验关键数据

### 主实验 (生成样本+真实样本混合, KnowPrompt, Micro F1%)

| 方法 | TACRED K=8 | TACRED K=16 | Re-TACRED K=8 | SemEval K=8 |
|------|:---:|:---:|:---:|:---:|
| 纯手工标注 (KnowPrompt) | 22.07 | 30.00 | 44.77 | 61.39 |
| 纯LLM生成 (Ours pure) | 22.48 | 27.99 | 34.77 | 42.02 |
| **混合-OBO** | **33.48** | **34.93** | **57.41** | **67.92** |
| **混合-AAO** | **34.30** | **35.56** | **56.01** | **70.32** |
| 直接LLM做RE | 21.17 | — | 33.30 | 25.55 |

### 消融实验 (KnowPrompt, Micro F1%)

| 变体 | TACRED K=8 | TACRED K=32 | Re-TACRED K=32 | SemEval K=32 |
|------|:---:|:---:|:---:|:---:|
| OBO + DPO | 22.48 | 30.41 | 50.72 | 47.70 |
| OBO (无DPO) | 18.35 | 23.70 | 40.45 | 30.03 |
| OBO (无DPO, 无多样性指令) | 21.86 | 25.66 | 32.74 | 25.88 |
| AAO + DPO | 21.86 | 26.39 | 46.50 | 56.34 |
| AAO (无DPO) | 18.97 | 36.04 | 36.66 | 49.58 |

### 关键发现

1. **混合训练最优**: 将LLM生成样本与手工标注混合使用，在大多数设置下优于仅用其中一种，TACRED K=8从22.07提升到34.30
2. **训练非LLM模型 > 直接用LLM做RE**: Ours(pure)在所有4个数据集上都优于Direct-RE，这意味着用LLM生成训练数据再训练小模型比直接用LLM更好
3. **DPO微调有效**: 在大多数OBO设置下，DPO微调使生成样本的质量显著提升
4. **多样性指令有效**: "Please make samples as different as possible"使平均余弦相似度降低0.051
5. **固定demonstration数量反而有害**: 当demonstration被替换时，LLM会高概率生成与被替换样本相似的内容
6. **生成样本数量存在上限**: 性能在K≈32时达到饱和，继续增加无显著提升

## 亮点与洞察

- **发现并解决了一个实际问题**: LLM生成训练数据的低多样性问题是一个被广泛忽视但实际影响很大的issue
- **DPO用于训练数据质量控制**: 创造性地将DPO从对齐领域迁移到数据生成质量控制，用自动构造的偏好对代替人工标注
- **"训练小模型比直接用大模型"更好**: 这一发现对实际部署有重要指导意义——可以降低持续调用LLM的成本
- **防作弊设计**: 分离DPO训练和生成使用的关系类别，避免信息泄露

## 局限与展望

- 仅在LLaMA2-7b上实验，未验证更大或更新的LLM（如LLaMA3, GPT-4等）
- DPO微调的计算开销未详细报告
- 当手工标注样本增多时（K=32），生成样本的边际收益递减
- 未探索更复杂的关系类型（如跨句关系、嵌套关系）
- 生成样本的多样性上限受LLM训练语料库的限制，K≈32饱和

## 相关工作与启发

- Self-Instruct (Wang等, 2023)开创了用LLM生成指令数据的方向，但未关注多样性问题
- Xu等(2023)的Data Generation方法是直接对比base，本文方法在大多数设置下都有优势
- DPO (Rafailov等, 2023)原用于摘要生成和对话，本文展示了其在数据增强中的新用途
- 启示：LLM的"模仿行为"（Imitation Behavior）是一个值得深入研究的现象

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将DPO应用于控制生成数据多样性是新颖的切入点，ICL的OBO策略设计也有巧思
- **实验充分度**: ⭐⭐⭐⭐ — 4个数据集、多种设置、消融实验+多样性分析+case study，非常充分
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法描述详细，图示直观
- **价值**: ⭐⭐⭐⭐ — 对few-shot RE实践有直接指导意义，方法可推广到其他任务的数据增强

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2025\] A Variational Approach for Mitigating Entity Bias in Relation Extraction](variational_approach_mitigating_entity_bias_relation_extraction.md)
- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [\[ACL 2025\] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)

<!-- RELATED:END -->
