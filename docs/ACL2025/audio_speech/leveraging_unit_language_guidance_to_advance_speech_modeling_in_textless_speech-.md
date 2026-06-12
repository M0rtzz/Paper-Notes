---
title: >-
  [论文解读] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation
description: >-
  [ACL 2025 (Findings)][无文本语音翻译] 本文提出"单元语言"（unit language）的概念，通过n-gram语言建模将语音离散单元构造为类文本表示，并利用多任务学习引导无文本语音到语音翻译（S2ST）模型的训练过程，同时提出任务提示建模来缓解源端和目标端单元语言同时使用时的冲突…
tags:
  - "ACL 2025 (Findings)"
  - "无文本语音翻译"
  - "单元语言"
  - "多任务学习"
  - "任务提示"
  - "跨模态建模"
---

# Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation

**会议**: ACL 2025 (Findings)  
**arXiv**: [2505.15333](https://arxiv.org/abs/2505.15333)  
**代码**: 无  
**领域**: 其他  
**关键词**: 无文本语音翻译, 单元语言, 多任务学习, 任务提示, 跨模态建模

## 一句话总结

本文提出"单元语言"（unit language）的概念，通过n-gram语言建模将语音离散单元构造为类文本表示，并利用多任务学习引导无文本语音到语音翻译（S2ST）模型的训练过程，同时提出任务提示建模来缓解源端和目标端单元语言同时使用时的冲突，在VoxPopuli四语数据集上取得显著提升。

## 研究背景与动机

**领域现状**：无文本语音到语音翻译（Textless S2ST）是近年备受关注的方向。其核心思路是绕过文本中间表示，直接从源语言语音生成目标语言语音。典型做法是先用自监督语音模型（如HuBERT）将语音编码为离散单元序列，然后用序列到序列模型进行翻译。

**现有痛点**：无文本S2ST面临两大建模挑战：(1) **跨模态（Cross-Modal, CM）挑战**——如何从连续的语音信号中有效提取语言学特征，离散单元虽然压缩了信息但也丢失了部分语言结构；(2) **跨语言（Cross-Lingual, CL）挑战**——语音离散单元序列通常比对应的文本序列长得多（一段话的语音单元可能是文本token的5-10倍），这使得长序列上的语言对齐变得困难。相比使用文本作为中间表示的级联系统，端到端的无文本系统性能仍有明显差距。

**核心矛盾**：语音离散单元虽然包含语言信息，但其序列过长且缺乏文本那样清晰的词边界和语言结构，模型难以有效建模。使用文本辅助可以提升性能，但这与"无文本"的目标相悖。

**本文目标**：在不使用文本转录的前提下，构造一种类文本的语音表示来弥合离散单元与文本之间的信息差距。

**切入角度**：作者观察到语音离散单元的重复模式中蕴含着语言结构信息，可以通过n-gram统计建模来发现和利用这些结构，类似于从字符序列中发现词汇的过程。

**核心 idea**：将语音离散单元序列通过n-gram语言建模压缩为"单元语言"（unit language）表示——一种更短、更接近文本结构的表示，然后通过多任务学习将其作为辅助监督信号引导S2ST模型训练。

## 方法详解

### 整体框架

整体pipeline分三步：(1) 用HuBERT等自监督模型将语音转为离散单元序列；(2) 在离散单元上进行n-gram语言建模，构造"单元语言"表示（将频繁出现的n-gram合并为新token，类似BPE）；(3) 在序列到序列翻译模型的训练中，加入源端和/或目标端的单元语言预测作为辅助任务，通过多任务学习引导主翻译任务。

### 关键设计

1. **单元语言构造（Unit Language Construction）**:

    - 功能：将冗长的语音离散单元序列压缩为更短的、类文本的表示
    - 核心思路：在离散单元序列上进行n-gram频率统计，将高频出现的n-gram合并为单一token（类似BPE分词算法的思想）。迭代多轮后，原始的数百个单元序列可以被压缩为数十个"单元词"的序列。这种压缩后的序列称为"单元语言"
    - 设计动机：离散单元序列过长是S2ST的核心瓶颈。通过模拟文本分词的过程，单元语言保留了语言结构信息的同时大幅缩短了序列长度，使模型更容易学习跨语言对齐

2. **多任务学习框架**:

    - 功能：利用单元语言作为辅助监督信号来增强翻译模型
    - 核心思路：在encoder端添加源语言单元语言的预测任务（帮助encoder更好地提取语言特征，解决CM挑战），在decoder端添加目标语言单元语言的预测任务（帮助decoder更好地生成目标语言结构，解决CL挑战）。总损失为主翻译损失和辅助任务损失的加权和 $L = L_{s2u} + \alpha L_{src\_ul} + \beta L_{tgt\_ul}$
    - 设计动机：直接在翻译目标上训练的模型缺乏对语言结构的显式建模，辅助任务提供了额外的归纳偏置

3. **任务提示建模（Task Prompt Modeling）**:

    - 功能：缓解同时使用源端和目标端单元语言辅助任务时的冲突
    - 核心思路：初步实验发现，同时加入源端和目标端的单元语言辅助任务反而会降低性能，因为两个辅助任务对共享参数的优化方向可能冲突。作者提出在模型输入前添加可学习的任务提示（task prompt），不同的任务使用不同的提示向量，使模型能根据提示区分不同的任务目标，从而缓解梯度冲突
    - 设计动机：多任务学习中的任务冲突是常见问题，任务提示是一种轻量级的解决方案，不需要增加独立的task-specific网络

### 损失函数 / 训练策略

总训练损失为三部分的加权和：主翻译任务的交叉熵损失 $L_{s2u}$、源端单元语言预测损失 $L_{src\_ul}$、目标端单元语言预测损失 $L_{tgt\_ul}$。使用任务提示建模时，三个任务共享主干参数但通过不同的prompt前缀来区分。训练分阶段进行，先预热翻译任务，再加入辅助任务。

## 实验关键数据

### 主实验

在VoxPopuli数据集的四语言翻译方向上评测（Es→En, Fr→En, Es→Fr, Fr→Es等方向）：

| 方法 | Es→En BLEU | Fr→En BLEU | 平均BLEU | ASR-BLEU |
|------|-----------|-----------|---------|----------|
| 基线 (不使用文本的S2U) | ~15 | ~17 | ~16 | ~20 |
| + 源端单元语言 | ~17 | ~19 | ~18 | ~23 |
| + 目标端单元语言 | ~16.5 | ~18.5 | ~17.5 | ~22 |
| + 双端 (无prompt) | ~16 | ~18 | ~17 | ~21 |
| + 双端 + 任务提示 | ~18 | ~20 | ~19 | ~24 |
| 使用文本的级联系统 | ~19 | ~21 | ~20 | ~25 |

### 消融实验

| 配置 | 平均BLEU | 说明 |
|------|---------|------|
| 完整模型（双端+任务提示） | ~19 | 最佳无文本方案 |
| 仅源端单元语言 | ~18 | 单独帮助encoder有效 |
| 仅目标端单元语言 | ~17.5 | 单独帮助decoder也有效 |
| 双端无任务提示 | ~17 | 冲突导致性能退化 |
| 无单元语言 (基线) | ~16 | 纯S2U翻译 |
| 不同n-gram大小 | 4-gram最优 | 过小则压缩不足，过大则丢失信息 |

### 关键发现

- **单元语言有效弥合了无文本和有文本系统的差距**：使用单元语言引导后，无文本系统的性能接近使用文本的级联系统，证明了语音离散单元中确实蕴含可利用的语言结构
- **源/目标端辅助任务存在冲突**：直接叠加两个辅助任务反而不如只用一个，但任务提示建模能有效缓解此冲突
- **任务提示是轻量但有效的解决方案**：仅增加少量可学习参数就能解决多任务冲突，且不影响推理效率
- **n-gram大小的选择对效果有影响**：4-gram是一个较好的平衡点，过小的n-gram无法有效压缩序列，过大则可能引入过多噪声

## 亮点与洞察

- **"单元语言"概念的提出**：将语音离散单元通过统计方法构造为类文本表示是一个直观且有效的想法。这种表示不依赖任何文本数据，却能模拟文本的结构特性，为无文本语音处理提供了新的工具。
- **多任务冲突的识别和解决**：论文诚实地报告了源/目标端辅助任务冲突的现象，并提出了任务提示这一简洁的解决方案。这种"发现问题-分析原因-提出方案"的研究思路值得学习。
- **无文本语音翻译的新上限**：证明即使不使用文本转录，通过挖掘语音单元自身的结构也能达到接近有文本系统的效果，为低资源语言的S2ST提供了希望。

## 局限与展望

- 仅在VoxPopuli数据集上实验，该数据集主要是欧洲语言的议会演讲，语种和领域多样性有限
- 单元语言的构造依赖HuBERT等预训练模型的离散化质量，对于资源极低的语言可能不适用
- 未与最新的大规模语音语言模型（如AudioPaLM、SeamlessM4T）进行比较
- 任务提示建模虽有效，但论文未深入分析其学到的表示与任务特性的关系

## 相关工作与启发

- **vs Textless NLP (Lakhotia et al.)**: 早期无文本NLP工作开创了语音离散单元的使用，本文在其基础上进一步提出通过n-gram压缩来构造更高层次的"单元语言"
- **vs Translatotron (Jia et al.)**: Translatotron系列是端到端S2ST的代表，但依赖频谱图而非离散单元，本文基于离散单元的方案更适合与语言模型结合
- **vs SeamlessM4T**: Meta的大规模模型使用了文本辅助训练，本文的无文本方案虽然规模小但思路更有挑战性

## 评分

- 新颖性: ⭐⭐⭐⭐ 单元语言的概念新颖，将BPE思想应用到语音单元是有趣的跨领域迁移
- 实验充分度: ⭐⭐⭐ 实验数据集单一（仅VoxPopuli），缺少与大规模基线的对比
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法描述条理分明
- 价值: ⭐⭐⭐ 对无文本S2ST有启发意义，但实际应用场景受限于低资源语言

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ACL 2025\] (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](rsatexttwosuperior_a_rhetorical-strategy-aware_rational_speech_act_framework_for.md)
- [\[ACL 2025\] Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)
- [\[ACL 2025\] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)

</div>

<!-- RELATED:END -->
