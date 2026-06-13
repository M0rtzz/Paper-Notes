---
title: >-
  [论文解读] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries
description: >-
  [ACL 2025][多语言/翻译][词汇迁移] 本文提出一种基于双语词典的跨语言词汇迁移方法，利用BPE分词器"删除子词后回退到更短子词"的特性，通过迭代删除-重分词-对齐的过程最大化目标语言子词的映射覆盖率，在低资源语言上显著优于依赖单语语料或平行语料的现有方法。 跨语言词汇迁移旨在将预训练语言模型适配到新语言…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "词汇迁移"
  - "双语词典"
  - "BPE"
  - "嵌入初始化"
  - "低资源语言"
---

# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries

**会议**: ACL 2025  
**arXiv**: [2506.01535](https://arxiv.org/abs/2506.01535)  
**代码**: [GitHub](https://github.com/sj-h4/dict_trans_tokenizer)  
**领域**: 低资源语言 / 跨语言迁移  
**关键词**: 词汇迁移, 双语词典, BPE, 嵌入初始化, 低资源语言

## 一句话总结

本文提出一种基于双语词典的跨语言词汇迁移方法，利用BPE分词器"删除子词后回退到更短子词"的特性，通过迭代删除-重分词-对齐的过程最大化目标语言子词的映射覆盖率，在低资源语言上显著优于依赖单语语料或平行语料的现有方法。

## 研究背景与动机

跨语言词汇迁移旨在将预训练语言模型适配到新语言，是解决token过度碎片化（over-fragmentation）问题的关键技术。现有方法面临以下困境：

**Wechsel**：需要大量单语语料和双语词典

**Focus**：依赖源语言和目标语言的子词重叠，对使用不同书写系统的语言无效

**UniBridge**：同样依赖子词重叠，对跨脚本语言有局限

**Trans-Tokenizer**：需要平行语料，低资源语言通常无法获取

**核心洞察**：对于许多低资源语言，虽然单语语料和平行语料稀缺，但双语词典往往是可用的——这要归功于描述语言学家在语言文档化工作中的产出。词典虽然体量小，但包含了精确的词汇对应关系，可以有效地建立跨语言的子词映射。

BPE分词器有一个关键特性：当一个子词从词表中删除时，包含该子词的词会回退到更短的子词组合。本文巧妙利用这一特性，通过迭代删除已映射的子词来逐步覆盖更短的子词。

## 方法详解

### 整体框架

方法分为三步：(1) 用词典数据训练目标语言的BPE分词器；(2) 迭代对齐源语言和目标语言的子词；(3) 用对齐结果初始化目标语言子词嵌入。之后可选进行语言自适应预训练（LAPT）。

### 关键设计

1. **分词器训练**: 
   使用词典条目（目标语言词汇）训练byte-level BPE分词器。采用byte-level确保即使在有限资源下也不会出现OOV（out of vocabulary）问题。

2. **迭代子词映射算法（核心创新）**:
   重复以下四个步骤直到无新子词被映射：
   
    - **(1) 分词（Tokenization）**：分别用目标语言分词器和源模型分词器对词典的条目（entry）和定义（definition）进行分词
    - **(2) 对齐（Alignment）**：将条目-定义对视为平行语料，使用fast_align（基于IBM Model 2）进行子词级对齐
    - **(3) 映射（Mapping）**：基于对齐结果创建目标子词→源子词的一对多映射（type级别），记录映射计数
    - **(4) 删除（Removal）**：从目标分词器词表中删除已映射的子词及包含它们的合并规则
   
   **删除步骤的关键作用**：BPE分词器默认只映射最长子词，其组成子词不会被映射。删除长子词后，BPE回退到更短片段，使得下一轮迭代可以映射这些更短的子词。

3. **嵌入初始化**:
   对于目标子词t，其嵌入初始化为对应源子词嵌入的加权平均：
   
    $\boldsymbol{e}_t^T = \sum_{s \in \mathcal{M}_t} c(s|t) \cdot \boldsymbol{e}_s^S$
   
   其中 $c(s|t)$ 是子词s在映射中的相对频次。特殊token、数字和标点直接从源模型复制。未映射的子词使用UNK token嵌入（Llama 3则随机初始化）。

4. **与现有方法的差异**:

    - 与Trans-Tokenizer的区别：使用子词级对齐（因为词典条目短），而非词级对齐；通过迭代删除估计更短子词的映射
    - 与Focus的区别：使用词典训练的映射，而非子词重叠+FastText静态嵌入
    - 与Wechsel的区别：不需要大量单语语料

### 损失函数 / 训练策略

- MLM（掩码语言模型）：微调所有层用于NER下游任务
- CLM（因果语言模型）：训练顶部和底部各两层，使用LoRA
- 采用multi-token prediction提升训练效率
- LAPT最多使用3000个样本

## 实验关键数据

### NER性能——Micro F1（表3）

| 模型 | 德语 | 日语 | 古英语 | 维吾尔语 | 梵语 | 高棉语 | 满语 |
|------|------|------|--------|---------|------|--------|------|
| RoBERTa | 89.61 | 75.33 | 62.39 | 38.73 | 51.48 | 27.58 | 73.52 |
| XLM-R | 90.27 | 81.28 | 37.59 | 28.30 | 48.85 | 34.78 | 65.32 |
| Focus(XLM-R)+LAPT | 90.00 | 77.46 | 37.57 | 37.16 | 12.33 | 12.33 | 28.03 |
| Ours(RoBERTa)+LAPT | 76.43 | 73.60 | **52.71** | **64.52** | 42.08 | **62.96** | **92.87** |
| Ours(XLM-R)+LAPT | 75.98 | 74.73 | 40.94 | **59.41** | **56.99** | **58.37** | 91.39 |

### 困惑度——Llama 3.1（表4，节选）

| 模型 | 德语 | 维吾尔语 | 高棉语 | 满语 |
|------|------|---------|--------|------|
| Llama 3.1 | 655 | 2.07×10²⁴ | ∞ | 2.30×10¹⁹ |
| Focus+LAPT | 2376 | 7.53×10²⁰ | ∞ | 1.29×10⁶ |
| Ours | 444877 | 18053 | 64508 | 144818 |
| **Ours+LAPT** | **88.61** | **168.43** | **4.32** | **502.02** |

### 数据效率对比（表5）

| 语言 | Focus所需词数 | 本方法所需词数 |
|------|-------------|--------------|
| 德语 | 21,582,818 | 101,997 |
| 维吾尔语 | 2,771,058 | **1,131** |
| 梵语 | 2,812,121 | **5,282** |
| 高棉语 | 1,937,229 | **5,656** |
| 满语 | 397,659 | **21,620** |

### 关键发现

1. **低资源语言上大幅领先**：在维吾尔语(+27.36 F1)、高棉语(+50.63 F1)、满语(+64.84 F1)上，本方法远超Focus
2. **极致数据效率**：仅用Focus不到10%的数据量就实现了更好的性能。维吾尔语仅需1131个词典条目
3. **高资源语言上不占优**：德语和日语上Focus表现更好，因为这些语言有充足的单语语料和子词重叠
4. **映射覆盖率高**：大多数语言的子词映射率超过85%（梵语97.27%），只有满语较低（77.35%）
5. **困惑度改善惊人**：高棉语从∞降到4.32，满语从10¹⁹降到502——说明嵌入初始化质量极高
6. **适用语言特性**：在谱系上远离英语、具有孤立语或黏着语特征的语言（维吾尔语、高棉语、满语）上效果最佳

## 亮点与洞察

- **巧妙利用BPE特性**：删除子词→回退到更短子词→迭代映射的思路极为精巧，将BPE的"缺陷"变为优势
- **描述语言学的NLP价值**：词典数据通常被NLP社区忽视，本文展示了即使很小的词典（如维吾尔语仅1131条）也能产生巨大价值
- **数据效率的极端体现**：用几千个词典条目替代数百万词语料，对真正的低资源场景意义重大
- **方法的通用性**：适用于MLM和CLM两种架构（RoBERTa、XLM-R、Llama 3.1、Gemma 2），证明了方法的架构无关性

## 局限与展望

- 高资源语言上不如Focus，说明词典信息量终究有限，不能完全替代大规模语料
- 未映射的子词使用UNK嵌入（或随机初始化），可能影响极低覆盖率的语言
- 词典质量和覆盖范围直接影响性能，不同词典来源（Wiktionary vs 专业词典）的影响未探究
- 仅评估了NER和困惑度，更多下游任务（如MT、QA）的验证将更有说服力
- LAPT的数据量上限为3000样本，更大规模LAPT的影响未知
- 对多义词的处理可能不够精细（一对多映射取加权平均）

## 相关工作与启发

- **Trans-Tokenizer (Remy et al., 2024)**：使用平行语料的词级对齐进行词汇迁移，本文的词典方法是其在低资源场景的替代方案
- **Focus (Dobler & de Melo, 2023)**：依赖子词重叠的经典方法，本文在跨脚本语言上大幅超越
- **Wechsel (Minixhofer et al., 2022)**：需要大量单语语料和双语词典，本文仅需词典
- **ZeTT (Minixhofer et al., 2024)**：训练超网络零样本预测嵌入，与词典方法互补
- 启发：自然语言处理不应局限于大规模数据范式，小而精确的语言学资源在特定场景下可能更有效

## 评分

- **新颖性**: 9/10 — BPE删除-回退的迭代映射策略是原创性很强的技术贡献，词典在词汇迁移中的应用也很新颖
- **实验充分度**: 8/10 — 7种语言、4种模型架构、NER和困惑度双指标，覆盖面广。但下游任务种类可以更多
- **写作质量**: 8/10 — 算法描述清晰，图示直观，消融分析到位
- **价值**: 9/10 — 对低资源语言社区有极大实用价值，方法简洁优雅且易于实现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)

</div>

<!-- RELATED:END -->
