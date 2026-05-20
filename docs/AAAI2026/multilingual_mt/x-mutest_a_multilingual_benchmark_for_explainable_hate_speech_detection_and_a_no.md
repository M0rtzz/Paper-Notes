---
title: >-
  [论文解读] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework
description: >-
  [AAAI 2026][多语言/翻译][仇恨言论检测] 本文提出X-MuTeST框架，结合LLM语义推理和n-gram attention增强的两阶段训练方法，用于可解释的多语言仇恨言论检测，并提供了印地语和泰卢固语的首个token级人工标注理据基准数据集。
tags:
  - "AAAI 2026"
  - "多语言/翻译"
  - "仇恨言论检测"
  - "可解释性"
  - "多语言"
  - "LLM解释"
  - "人工标注理据"
---

# X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework

**会议**: AAAI 2026  
**arXiv**: [2601.03194](https://arxiv.org/abs/2601.03194)  
**代码**: [https://github.com/ziarehman30/X-MuTeST](https://github.com/ziarehman30/X-MuTeST)  
**领域**: NLP理解 / 多语言  
**关键词**: 仇恨言论检测, 可解释性, 多语言, LLM解释, 人工标注理据

## 一句话总结
本文提出X-MuTeST框架，结合LLM语义推理和n-gram attention增强的两阶段训练方法，用于可解释的多语言仇恨言论检测，并提供了印地语和泰卢固语的首个token级人工标注理据基准数据集。

## 研究背景与动机

**领域现状**：仇恨言论检测已从单纯分类发展到需要提供可解释理据（rationale）的阶段。英语有HateXplain等少量数据集提供人工标注理据，但低资源语言（如印地语、泰卢固语）几乎没有此类资源。

**现有痛点**：LLM虽然能识别显式仇恨词汇（如"狗"），但对文化语境中的隐式仇恨表达（如印地语中"吠叫"作为侮辱语）识别不足。机器生成的理据与人类理据之间存在显著差距，尤其是低资源语言。同时传统attention-based解释方法缺乏语义推理能力。

**核心矛盾**：模型分类准确率与解释可信度之间的断裂——模型可能"答对但理由错"，特别是在低资源语言的文化语境下。

**本文目标**：(1) 为印地语、泰卢固语、英语提供token级人工理据标注基准；(2) 设计融合LLM解释和模型自身attention的混合可解释框架。

**切入角度**：利用人工理据作为训练信号引导模型attention对齐人类判断，再用n-gram扰动方法生成模型驱动的解释，最后与LLM解释取并集得到最终理据。

**核心 idea**：两阶段训练——第一阶段用人工理据引导attention对齐，第二阶段用模型自身n-gram可解释性得分替换人工理据进一步调优，最终解释取LLM和n-gram方法的并集以兼顾语义和语法覆盖。

## 方法详解

### 整体框架
输入文本经预训练编码器（Muril或XLMR）编码，训练分两个阶段：Stage-1使用人工理据引导attention对齐（3个epoch），Stage-2使用n-gram可解释性方法生成的attention mask继续训练。最终解释由X-MuTeST n-gram方法和LLaMA-3.1生成的解释取并集得到。

### 关键设计

1. **人工理据引导的Attention对齐（Stage-1）**:

    - 功能：让模型关注的token与人类判断对齐
    - 核心思路：计算每个token的attention分数 $a_i = \text{softmax}(h_i^\top h_{[CLS]})$，然后通过attention对齐损失 $\mathcal{L}_{att} = -\sum_i R_i \log a_i$ 引导模型关注人工标注的理据token。总损失为 $\mathcal{L}_1 = \alpha \mathcal{L}_{att} + (1-\alpha)\mathcal{L}_{cl}$，其中 $\alpha=0.3$
    - 设计动机：直接训练模型的attention关注理据token，弥合机器理据与人类理据的差距

2. **N-gram扰动可解释性方法（Stage-2）**:

    - 功能：生成模型驱动的token重要性得分
    - 核心思路：对原始文本的每个unigram/bigram/trigram分别输入模型，计算与原文预测概率的差值（logit drop），然后对每个token汇聚其参与的所有n-gram的重要性得分：$E[t] = \sum_{n=1}^{3} w_n \cdot \frac{1}{N_t^{(n)}} \sum_{ng \ni t} \Delta P_{ng}$，权重为0.5/0.3/0.2。选top-k token作为Stage-2的attention target
    - 设计动机：人工理据作为初始引导，但模型自身学到的特征可能捕获更多语境相关的重要token，两阶段交替使用可平衡人类知识和模型洞察

3. **LLM-consulted混合解释**:

    - 功能：生成最终的可解释理据
    - 核心思路：最终解释 $\mathcal{E}_{final} = \mathcal{E}_X \cup \mathcal{E}_{LLM}$，其中 $\mathcal{E}_X$ 来自n-gram方法的top token，$\mathcal{E}_{LLM}$ 来自LLaMA-3.1对仇恨词的识别。取并集确保语法层面和语义层面的双重覆盖
    - 设计动机：LLM擅长语义推理但缺乏文化敏感性，n-gram方法擅长捕捉task-specific显著token但缺乏语义理解，互补结合

### 损失函数 / 训练策略
两阶段损失结构相同：$\mathcal{L} = \alpha \mathcal{L}_{att} + (1-\alpha)\mathcal{L}_{cl}$。Stage-1中 $\alpha=0.3$；Stage-2中 $\alpha$ 因语言而异（泰卢固语/英语0.6，印地语0.7）。编码器选择：泰卢固语和英语用Muril，印地语用XLMR。

## 实验关键数据

### 主实验

| 模型 | Accuracy | F1 | Token-F1↑ | IOU-F1↑ | Comp↑ | Suff↓ |
|---|---|---|---|---|---|---|
| Muril-XMuTeST | 0.860 | 0.860 | 0.558 | 0.292 | 0.685 | 0.095 |
| Muril-Rationale-XMuTeST | **0.874** | **0.864** | **0.575** | **0.301** | **0.724** | **0.075** |
| XLMR-Rationale-XMuTeST | 0.846 | 0.836 | 0.563 | 0.242 | 0.727 | 0.104 |
| GPT-4o | 0.648 | 0.593 | 0.389 | 0.182 | - | - |
| LLaMA-3.1 | 0.636 | 0.672 | 0.515 | 0.281 | - | - |

### 消融实验

| 配置 | Token-F1 | IOU-F1 | 说明 |
|---|---|---|---|
| LIME解释 | 0.552 | 0.284 | 传统方法 |
| X-MuTeST解释 | 0.558 | 0.292 | +0.6%/+0.8% |
| +Rationale训练+LIME | 0.561 | 0.294 | 人工理据有帮助 |
| +Rationale训练+XMuTeST | **0.575** | **0.301** | 完整框架最优 |

### 关键发现
- 人工理据引导训练显著提升分类和解释性能（Acc从0.860→0.874）
- X-MuTeST方法在所有评估指标上优于LIME解释方法
- LLM（GPT-4o、LLaMA-3.1）分类准确率远低于fine-tuned模型，但LLaMA-3.1的理据质量（Token-F1=0.515）优于GPT-4o
- 迭代标注流程将标注者一致性从中等提升至高水平（Kappa 81-85）

## 亮点与洞察
- **低资源语言理据数据集**：首次为印地语和泰卢固语提供token级仇恨理据标注，填补了跨语言可解释仇恨检测的数据空白。这类资源的价值超越单篇论文
- **两阶段训练的巧妙设计**：先用人工理据"热启动"attention方向，再用模型自身发现的重要token微调，避免了纯人工理据的偏见和纯模型理据的不可靠
- **LLM作为解释辅助而非主力**：认识到LLM在分类上不如fine-tuned模型但语义推理能力可互补利用

## 局限与展望
- 数据规模有限（三语言加起来约17K样本），难以覆盖仇恨言论的所有表达形式
- LLM选择（LLaMA-3.1）基于经验性对比，缺乏系统性评估
- 仅处理token级解释，缺乏句子级或discourse级的理据推理
- 可扩展到更多低资源语言（如孟加拉语、乌尔都语等南亚语言）

## 相关工作与启发
- **vs HateXplain (Mathew et al., 2021)**: 首个英语仇恨理据数据集，本文将其扩展到多语言场景，并提出更强的训练框架
- **vs LIME/SHAP**: 传统事后解释方法，本文的n-gram方法更直接且针对NLP任务优化
- 两阶段"人工→模型"理据切换的训练策略可迁移到其他需要可解释性的NLP任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 多语言理据数据集和两阶段训练结合LLM解释的框架较新颖
- 实验充分度: ⭐⭐⭐⭐ 三语言评测，多种baseline对比，含LLM基线
- 写作质量: ⭐⭐⭐ 结构完整但部分描述冗长，公式符号可更简洁
- 价值: ⭐⭐⭐⭐ 数据集贡献对低资源语言研究有持久价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Comparative Analysis of Multilingual Hate Speech Detection](../../ACL2025/multilingual_mt/comparative_analysis_of_multilingual_hate_speech_detection.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2025\] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models](../../ACL2025/multilingual_mt/cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](../../ACL2026/multilingual_mt/the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](../../ACL2026/multilingual_mt/mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)

</div>

<!-- RELATED:END -->
