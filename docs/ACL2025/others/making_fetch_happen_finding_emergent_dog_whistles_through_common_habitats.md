---
title: >-
  [论文解读] Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats
description: >-
  [ACL 2025][隐式仇恨言论] 提出 FETCH! 基准和 EarShot 系统，用于在大规模社交媒体语料库中发现新兴的"狗哨"（dog whistle，即具有双重含义的编码表达），利用向量数据库和 LLM 的结合实现了比现有方法高 2-20 个 F-score 百分点的提升。
tags:
  - ACL 2025
  - 隐式仇恨言论
  - 暗语检测
  - 新兴词汇发现
  - 向量数据库
  - LLM
---

# Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats

**会议**: ACL 2025  
**arXiv**: [2412.12072](https://arxiv.org/abs/2412.12072)  
**代码**: [https://github.com/KuleenS/FETCH-Dog-Whistle](https://github.com/KuleenS/FETCH-Dog-Whistle)  
**领域**: 其他  
**关键词**: 隐式仇恨言论, 暗语检测, 新兴词汇发现, 向量数据库, LLM

## 一句话总结

提出 FETCH! 基准和 EarShot 系统，用于在大规模社交媒体语料库中发现新兴的"狗哨"（dog whistle，即具有双重含义的编码表达），利用向量数据库和 LLM 的结合实现了比现有方法高 2-20 个 F-score 百分点的提升。

## 研究背景与动机

"狗哨"是一种具有双重含义的编码表达：对一般公众（外群体）传递一种含义，对目标受众（内群体）则传递具有争议性的政治观点，同时保持"可否认性"。例如 "dual citizen"（双重公民）表面上指双重国籍，但在某些语境下是一种反犹主义的暗语。

当前检测狗哨的方法主要依赖人工策划的词汇表，但存在几个关键问题：

**维护成本高**：词汇表需持续人工更新，无法跟上语言的动态演化。

**滞后性**：语言缩略、变体和新兴表达不断出现，如 "cosmopolitan" 可能被缩写为 "cosmos"。

**绕过内容审核**：狗哨的良性表面含义使其能轻松绕过现有的毒性和仇恨言论检测器。

本文将问题从"狗哨检测"（给定已知词汇表判断文本是否含狗哨）推进到更具挑战性的"狗哨发现"（在语料中发现未知的新兴狗哨），这是一个全新的任务定义。

## 方法详解

### 整体框架

该工作的贡献包含两部分：FETCH! 基准（定义任务和评估协议）和 EarShot 系统（提出强基线方法）。

任务定义：给定一个语料库和一组已知的种子狗哨（seed dog whistles），系统需要利用种子词和语料库来发现新的狗哨。

### 关键设计

1. **FETCH! 基准的三个案例研究**：

    - **Synthetic（Reddit）**：理想化场景，每条帖子都含有 GPT-4 标注的狗哨，约 16,000 条帖子。
    - **Balanced（Gab）**：中等密度场景，来自右翼平台 Gab 的约 300,000 条帖子，狗哨出现频率高于平均水平。
    - **Realistic（Twitter）**：真实场景，来自 Twitter API 的约 700 万条推文，狗哨稀疏分布。

2. **种子狗哨划分**：使用 Mendelsohn 等人整理的约 340 个英语根狗哨词汇表。按 n-gram 长度分层抽样，20% 作为种子，80% 作为测试。使用精确率、数据潜在召回率（DPR）和 $F_{0.5}$ 作为评估指标（偏重精确率以减轻人工审核负担）。

3. **EarShot 系统**：包含三个阶段：

    - **第一阶段（向量化）**：使用 all-MiniLM-L6-v2 将所有帖子编码为向量并存入 ChromaDB 向量数据库。
    - **第二阶段（近邻检索）**：获取包含种子狗哨的帖子的向量，找到最近邻帖子（不包含自身），捕获语义相关但不共享精确词汇的帖子。
    - **第三阶段（两条路径）**：
        - **DIRECT 路径**：直接将近邻帖子传给 LLM（LLaMA 8B/13B、Mistral 7B），提示模型提取其中的狗哨并以 JSON 格式输出。
        - **PREDICT 路径**：先用仇恨言论分类器（BERT 系列）或 LLM 过滤帖子，再用关键词提取算法（KeyBERT、RAKE、YAKE、TextRank、TF-IDF）提取候选词。

### 损失函数 / 训练策略

EarShot 本身不涉及端到端训练，而是一个管道式系统。各组件使用已有预训练模型：
- 句子编码器：all-MiniLM-L6-v2（轻量快速）
- 过滤器：ToxiGen BERT、RoBERTa R4、HateXplain BERT
- LLM：LLaMA 8B/13B、Mistral 7B（选择小型开源模型以保证可复现性）
- 基线方法中的 Word2Vec 使用 Gensim 训练，词汇量上限 500K，窗口大小 5，维度 100，训练 10 epochs

## 实验关键数据

### 主实验

| 方法 | 场景 | Precision | DPR | F₀.₅ |
|------|------|-----------|-----|------|
| 最优 Word2Vec | Synthetic | 5.50 | 8.40 | 5.91 |
| 最优 MLM | Synthetic | 2.00 | 0.42 | 1.14 |
| EarShot-PREDICT | Synthetic | **19.13** | 7.14 | **14.32** |
| EarShot-DIRECT | Synthetic | 20.31 | **56.30** | 23.29 |
| EarShot-PREDICT | Balanced | **14.81** | 1.65 | 5.70 |
| EarShot-DIRECT | Balanced | 2.97 | **13.58** | 3.52 |
| EarShot-PREDICT | Realistic | **10.00** | 1.47 | **4.63** |
| EarShot-DIRECT | Realistic | 0.94 | **60.29** | 1.17 |

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| BERT vs LLM 过滤 | BERT 高 0.1-0.5 F₀.₅ | 小型任务专用模型优于通用 LLM |
| KeyBERT vs RAKE/YAKE | KeyBERT 在小数据集优 | RAKE/YAKE 在大数据集更好 |
| Unigram vs Bigram/Trigram W2V | Unigram 全胜 | 但牺牲了多词组短语的召回 |
| DIRECT LLaMA 13B vs 8B | 13B 在 2/3 场景更优 | 但在 Realistic 中过度预测导致失败 |

### 关键发现

- 所有现有方法（Word2Vec、MLM、EPD）在三个场景中 F₀.₅ 均低于 6%，任务极具挑战性。
- EarShot-PREDICT 在精确率上大幅领先（9-20%），而 EarShot-DIRECT 在召回率上更强（13-60%），两条路径互补。
- Word2Vec 在 Balanced 和 Realistic 中的 DPR 反而优于 PREDICT，因为 PREDICT 受关键词提取器限制，提取的是"重要词"而非"狗哨"。
- Emoji 类狗哨（如 OK 白人至上手势）、上下文依赖型（如 "Federal Reserve"）和新兴型（如 2020 年后的 "jogger"）是所有方法的盲区。

## 亮点与洞察

- **任务定义创新**：将"狗哨"问题从检测推向发现，这是一个更贴近真实内容审核需求的任务范式。
- **FETCH! 基准设计合理**：三个案例研究覆盖了从理想到现实的不同场景，种子集按 n-gram 长度分层抽样避免偏见。
- **EarShot 的 two-path 设计**：PREDICT 路径高精确率适合减少人工审核负担，DIRECT 路径高召回率适合全面扫描，实际部署中可根据需求选择。
- **向量数据库的巧妙应用**：通过语义近邻而非精确匹配，能发现词汇形式完全不同但语义相关的新狗哨。

## 局限与展望

1. 仅在英语数据集上验证，狗哨在多语言场景中的研究尚为空白。
2. 缺乏专门的人工标注语料库，依赖正则表达式匹配可能引入假阳性。
3. LLM 可能在预训练时已接触过狗哨词汇表，存在数据污染风险。
4. 运行大型 LLM 需要较大计算资源，限制了方法的可及性。
5. 系统在 Realistic 场景中精确率仍仅 10%，实际部署需大量人工复核。
6. 未来可考虑混合 Word2Vec 和 LLM、对噪声预测后处理、集成多个 LLM、加入链式推理等方向。

## 相关工作与启发

- 与委婉语检测（Euphemism Detection）任务高度相关但存在关键区别：狗哨有明确的内群体/外群体二元含义，且通常涉及仇恨内容。
- Word2Vec/Phrase2Vec 方法来源于 Magu & Luo (2018) 的委婉语检测工作，MLM 方法来自 Zhu et al. (2021)。
- 本文揭示了一个重要洞察：传统的分布式语义方法（Word2Vec、BERT）在发现编码语言中表现不佳，但向量数据库+LLM 组合能捕捉更深层的语义关联。

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务定义新颖，FETCH! 基准填补空白，EarShot 设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三种场景、四类基线方法、多个模型变体、详尽的阈值分析
- 写作质量: ⭐⭐⭐⭐ 任务定义和动机阐述清晰，伦理考量充分
- 价值: ⭐⭐⭐⭐ 对内容审核和社交媒体治理有重要实际意义，但精确率仍需大幅提升

<!-- RELATED:START -->

## 相关论文

- [Voting or Consensus? Decision-Making in Multi-Agent Debate](voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs](understanding_common_ground_misalignment_in_goal-oriented_dialog_a_case-study_wi.md)
- [Decision Making under the Exponential Family DRO](../../ICML2025/others/decision_making_under_the_exponential_family_distributionally_robust_optimisatio.md)
- [Finding Diverse Solutions Parameterized by Cliquewidth](../../AAAI2026/others/finding_diverse_solutions_parameterized_by_cliquewidth.md)
- [Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)

<!-- RELATED:END -->
