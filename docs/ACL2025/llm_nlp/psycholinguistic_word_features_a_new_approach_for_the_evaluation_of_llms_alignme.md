---
title: >-
  [论文解读] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans
description: >-
  [ACL 2025][LLM/NLP][心理语言学] 首次系统提出使用心理语言学词汇规范（Glasgow 5,553词 × 7特征 + Lancaster 39,707词 × 6感知模态，共13种词汇特征）评估LLM与人类对齐，发现GPT-4o在Glasgow情感/概念特征上相关性较高，但所有模型在Lancaster感知觉特征上对齐极差，定量揭示LLM缺乏具身认知的根本局限。
tags:
  - ACL 2025
  - LLM/NLP
  - 心理语言学
  - LLM评估
  - 人类对齐
  - 词汇特征
  - 具身认知
---

# Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans

**会议**: ACL 2025  
**arXiv**: [2506.22439](https://arxiv.org/abs/2506.22439)  
**代码**: https://zenodo.org/records/14866800  
**领域**: LLM NLP / LLM评估  
**关键词**: 心理语言学, LLM评估, 人类对齐, 词汇特征, 具身认知

## 一句话总结

首次系统提出使用心理语言学词汇规范（Glasgow 5,553词 × 7特征 + Lancaster 39,707词 × 6感知模态，共13种词汇特征）评估LLM与人类对齐，发现GPT-4o在Glasgow情感/概念特征上相关性较高，但所有模型在Lancaster感知觉特征上对齐极差，定量揭示LLM缺乏具身认知的根本局限。

## 研究背景与动机

**已有问题**: 当前LLM评估主要聚焦任务性能（数学、推理、问答等可客观度量的能力），但LLM生成的文本被人类阅读和交互，因此对齐评估（情感、感知、偏好）同样重要——甚至更难做好。

**现有方案的不足**: (1) 人工评估不可扩展，每个模型需上万条评估；(2) LLM-as-Judge受限于评判模型自身偏差；(3) Chatbot Arena依赖社区众包，问题/参与者均不受控，只能提供排名而非深度分析；(4) 现有benchmark侧重任务完成，忽略语言深层语义属性如arousal、concreteness、感知觉关联等。

**核心洞察**: 心理语言学领域已积累数十年人类评分数据——对数万单词在唤醒度、效价、具体性、感知觉强度等维度的Likert评分。这些数据集现成可用、规模大（Lancaster覆盖近4万词）、经过大量研究验证，完全可以零成本复用来评估LLM与人类的语义对齐。

**深层动机**: 认知科学中的具身认知理论认为，语言意义不仅来自词语共现（LLM的学习方式），更源于感知经验、身体交互和情感体验。符号接地问题（symbol grounding problem）指出纯文本无法完全学习语言——至少约1%的词（约400词）必须通过真实世界经验来奠基。因此，评估LLM在感知维度上的对齐可以定量揭示"纯文本学习"的天花板。

## 方法详解

### 整体框架

复用现有心理语言学词汇评分数据集作为benchmark → 用与人类实验**相同的指示语**对LLM提问 → LLM输出每个词的数值评分 → 计算LLM评分与人类均值评分之间的4种相关系数 → 衡量对齐程度。整个流程全自动、可扩展，无需任何新的人工标注。

### 关键设计

1. **双数据集互补设计**：选择Glasgow规范（5,553词 × 7特征：arousal、valence、dominance、concreteness、imageability、familiarity、gender）和Lancaster规范（39,707词 × 6感知模态：touch、hearing、smell、taste、vision、interoception）。Glasgow包含已知对齐较好的情感/概念特征（验证方法有效性），Lancaster聚焦感知觉（预期对齐较差，用于检验具身认知假设），两者互补构成完整的13维评估空间。

2. **基于logprob的概率加权评分**：LLM评分不取单一输出数字，而是提取评分量表上每个值（如0-5或1-9）的输出概率，计算概率加权平均值。例如一个词在1-9量表上若P(7)=0.6, P(8)=0.3, P(6)=0.1，则估计值=7×0.6+8×0.3+6×0.1=7.2。此方法利用了概率分布信息，比直接取argmax更稳定，已被先前研究验证效果更优。

3. **四维相关指标体系**：同时计算Pearson和Spearman相关系数，分别在原始数据和四舍五入整数数据上各算一次（共4个指标）。Pearson更重视离群值区域的对齐（如高concreteness词），Spearman等权处理全分布。在偏态分布（如味觉/嗅觉评分集中在低端）上，两者差异显著——同时报告可避免单一指标的误导性结论。

## 实验关键数据

评估8个LLM：GPT-4o、GPT-4o-mini（闭源），LLama-3.2-3B、LLama-3.1-8B、LLama-3.2-11B（多模态）、Gemma-2-9B、Yi-1.5-9B、Occiglot-7B（开源）。

### Glasgow规范对齐（7种情感/概念特征）

| 词汇特征 | 量表 | GPT-4o (Pearson) | GPT-4o-mini | LLama-3.1-8B | LLama-3.2-3B | Gemma-2-9B |
|---------|------|-----------------|-------------|-------------|-------------|-----------|
| Arousal | 1-9 | ~0.75 | ~0.70 | ~0.60 | ~0.45 | ~0.55 |
| Valence | 1-9 | ~0.80 | ~0.75 | ~0.65 | ~0.50 | ~0.60 |
| Concreteness | 1-7 | ~0.80 | ~0.75 | ~0.55 | ~0.40 | ~0.55 |
| Imageability | 1-7 | ~0.75 | ~0.70 | ~0.55 | ~0.40 | ~0.50 |
| Familiarity | 1-7 | ~0.70 | ~0.65 | ~0.50 | ~0.35 | ~0.50 |
| Gender | 1-7 | ~0.50 | ~0.45 | ~0.35 | ~0.30 | ~0.55 |
| Dominance | 1-9 | ~0.55 | ~0.50 | ~0.40 | ~0.30 | ~0.40 |

*注：具体相关系数从论文雷达图读取，标~表示近似值*

- **具体案例**: "bicycle"（具体词）人类评分6.81，GPT-4o给出7.00（甚至更极端），但LLama-3.2-3B仅4.73；"bid"（抽象词）人类3.42，GPT-4o给2.96，LLama-3.2-3B给4.50——小模型缺乏区分度

### Lancaster规范对齐（6种感知觉特征）

| 感知模态 | 量表 | GPT-4o (Pearson) | Spearman差异 | 其他模型趋势 | 典型案例 |
|---------|------|-----------------|-------------|-------------|---------|
| Gustatory（味觉） | 0-5 | ~0.40 | Pearson>>Spearman | 极低(0.1-0.3) | "Lemon" 人类4.45 / GPT-4o 4.49 / Gemma-2-9B **0.01** |
| Olfactory（嗅觉） | 0-5 | ~0.45 | Pearson>Spearman | 低(0.15-0.35) | 偏态分布导致指标分歧 |
| Haptic（触觉） | 0-5 | ~0.35 | 小 | 低(0.1-0.3) | — |
| Auditory（听觉） | 0-5 | ~0.40 | 小 | 低(0.15-0.3) | — |
| Visual（视觉） | 0-5 | ~0.35 | 小 | 低(0.1-0.3) | 多模态LLama-3.2-11B **未见优势** |
| Interoceptive | 0-5 | ~0.30 | 小 | 低(0.1-0.25) | — |

*理想对齐范围为0.8-1.0，Lancaster所有特征均远低于此阈值*

### 分析维度汇总

| 分析维度 | 关键发现 |
|---------|---------|
| 模型规模 | LLama系列3B→8B→11B对齐逐步提升，但非决定性因素 |
| 多模态 | LLama-3.2-11B/GPT-4o/GPT-4o-mini均为多模态，但视觉特征对齐并不优于纯文本模型 |
| Pearson vs Spearman | 味觉/嗅觉等偏态分布上两者差异大；Pearson因为加权离群值表现更好 |
| 模型家族 | GPT-4o系列整体最优；开源模型在个别特征上有竞争力（如Gemma-2在gender上） |
| 直接输出 vs logprob | logprob概率加权估计普遍优于直接取argmax输出 |

### 关键发现
- Glasgow规范（情感/概念特征）对齐尚可，但距理想区间（0.8-1.0）仍有明显差距
- Lancaster规范（感知觉特征）对齐极差，证实了LLM在缺乏具身认知下的根本局限
- 多模态模型在视觉相关特征上并未展现出预期优势
- 同一个词（如"bicycle"），LLama-3.2-3B给出的具体性评分几乎无区分（4.73 vs 4.50），而GPT-4o能给出比人类更极端的区分（7 vs 2.96）

## 亮点与洞察

- **跨学科方法论创新**：不做新任务或新模型，而是提出全新的LLM评估视角——零成本复用心理语言学几十年积累的人类评分数据集。这种"用已有人文社科数据评估AI"的范式具有很强的可推广性
- **理论与实验的深度绑定**：将Glasgow和Lancaster的对齐落差与符号接地问题、具身认知理论紧密关联——LLM对arousal/valence对齐好（可从文本共现学到），但对taste/smell对齐差（需要身体经验），完美验证了认知科学的预测
- **logprob概率加权技巧**：LLM做Likert量表评分时，从logprobs计算加权平均（而非直接取argmax）可显著提升与人类评分的一致性，这个技巧对任何需要LLM做数值评分的场景都有实用价值
- **评估指标设计的审慎**：同时报告Pearson和Spearman在原始/取整数据上的4个系数，揭示了单一指标的陷阱（如嗅觉/味觉上Pearson远好于Spearman，因为分布极度偏态）
- **反直觉发现**：多模态模型（LLama-3.2-11B、GPT-4o）在视觉特征对齐上并不优于纯文本模型，说明当前多模态训练未有效传递感知接地能力

## 局限性

- 仅使用2个英语数据集，缺乏多语言覆盖；翻译测试可能引入文化和语义偏差
- 评估的8个LLM未包含最新模型（如Claude、Gemini Pro、Qwen等），代表性有限
- 评估指标仅用相关系数，未探索MAE、RMSE等绝对误差指标或分布层面的对齐度量
- 仅报告整体相关性，未对词进行分组分析（如高频词vs低频词、具体词vs抽象词的对齐差异）
- 未分析对齐差异的根因（如训练数据中感知觉描述文本的比例），也未探索改善路径（如用合成感知体验描述做后训练）

## 相关工作与启发

- **Trott (2024); Martínez et al. (2025)**：已证明GPT-4在valence和concreteness上与人类高度相关，但那些工作是"用LLM生成心理语言学数据"来辅助研究；本文视角相反——"用心理语言学数据评估LLM"，将评估权交还人文学科
- **Ivanova et al. (2024)**：提出logprob概率加权估计比prompt-based评估更接近人类判断，本文直接采用此技术获取LLM评分
- **具身认知理论 (Barsalou 2008; Borghi et al. 2024)**：认为认知根植于身体与环境的交互，本文用Lancaster感知觉数据为此提供了可量化的LLM证据
- **启发方向**：如果感知觉对齐差距源于缺乏身体经验，那么多模态训练（视频/音频/触觉传感器数据）能否弥补？本文提供了可测量的评估目标。另外，用合成感知描述文本做后训练是否有效？这些未探索的方向值得跟进

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨学科视角新颖，首次将心理语言学评分数据系统化用于LLM对齐评估；但方法本身（计算相关性）较为直接
- 实验充分度: ⭐⭐⭐ 覆盖13种特征×8个模型，有具体案例佐证（bicycle/bid/lemon），但缺乏精确数值表格和分组分析
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，理论（具身认知/符号接地）与实验结论紧密闭环
- 价值: ⭐⭐⭐⭐ 开辟了LLM评估新方向，提供可复用的benchmark方法论，对心理语言学社区参与AI研究有推动作用

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](moral_values_western.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean](polishing_every_facet_of_the_gem.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)

<!-- RELATED:END -->
