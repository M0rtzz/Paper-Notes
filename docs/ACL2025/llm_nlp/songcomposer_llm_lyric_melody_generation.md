---
title: >-
  [论文解读] SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition
description: >-
  [LLM/NLP] SongComposer是首个能够同时生成歌词和旋律的音乐专用大语言模型，通过词级对齐的元组格式、基于音乐知识的标量音高初始化、以及渐进式结构感知训练（motif→独立全曲→短语级配对），在歌词配旋律、旋律配歌词、歌曲续写和文本生成歌曲等任务上全面超越GPT-4。
tags:
  - LLM/NLP
---

# SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2402.17645](https://arxiv.org/abs/2402.17645) |
| 代码 | [项目主页](https://pjlab-songcomposer.github.io/) |
| 领域 | llm_nlp |
| 关键词 | 歌曲创作, 大语言模型, 旋律生成, 歌词生成, 符号音乐表示 |

## 一句话总结

SongComposer是首个能够同时生成歌词和旋律的音乐专用大语言模型，通过词级对齐的元组格式、基于音乐知识的标量音高初始化、以及渐进式结构感知训练（motif→独立全曲→短语级配对），在歌词配旋律、旋律配歌词、歌曲续写和文本生成歌曲等任务上全面超越GPT-4。

---

## 研究背景与动机

**研究背景：** 符号歌曲创作旨在以符号序列生成包含歌词和旋律的人声音轨，是歌曲生成的核心任务。近年来，歌词生成、旋律生成、歌词→旋律和旋律→歌词等子任务各有进展，但缺乏统一框架同时处理歌词和旋律。

**现有方法的局限性：** (1) SongMASS、TeleMelody等传统模型只能处理单一子任务，无法一个模型覆盖所有创作需求；(2) 直接使用LLM进行歌曲创作面临三大挑战——歌词与旋律的对齐方式未被探索、歌曲的层次化结构（motif和phrase）难以建模、高质量配对数据集稀缺。

**核心动机：** 歌曲的符号表示与自然语言在结构上有相似性，LLM的指令跟随能力使其可以将多个子任务整合到一个模型中。但需要解决符号表示设计、音高理解和结构建模三大技术难题。

---

## 方法详解

### 整体框架

SongComposer基于InternLM2-7B构建，包含三个核心创新：词级元组表示格式、标量音高初始化、以及三阶段渐进式训练。训练数据来自自建的SongCompose数据集（280K纯歌词 + 20K纯旋律 + 8K配对数据）。

### 关键设计

**1. 词级元组格式：** 将旋律分解为音高 $p$、音符时长 $d$、休止时长 $r$ 三个属性。配对数据中每个音符与对应歌词按 `<p>,d,w` 格式对齐。时长以1/16拍为单位量化：
$$d_k = \phi\left(\frac{\text{bpm}}{60}(\text{note-end}_k - \text{note-start}_k) \times 16\right)$$

**2. 标量音高初始化：** 先用高斯初始化中心音高 `<66>`，其余音高设为中心嵌入的标量倍数：
$$\text{emb}(\langle p \rangle) = c_p \cdot \text{emb}(\langle 66 \rangle)$$
其中倍数范围为 $[-\ln(e+17), \cdots, -\ln(e), \ln(e), \cdots, \ln(e+17)]$，显式编码了音高间的数值关系。

**3. 渐进式结构感知训练：**
- **阶段1 - Motif级旋律训练：** 提取高重复性短音符序列作为motif数据，让模型学习基本的重复模式
- **阶段2 - 独立全曲训练：** 分别在纯歌词和纯旋律数据上训练全曲理解能力
- **阶段3 - 短语级配对训练：** 在配对数据中引入5种短语特殊token（intro/verse/chorus/bridge/outro），让模型学习歌曲的段落结构

### 损失函数

标准的自回归next-token prediction损失，最大化给定上下文的token对数似然。

---

## 实验

### 主实验：歌词配旋律与旋律配歌词

| 方法 | PD(%)↑ | DD(%)↑ | MD↓ | Cosine Dist.↑ | ROUGE-2↑ | BS↑ |
|------|--------|--------|-----|---------------|----------|-----|
| SongMASS | 30.34 | 48.98 | 2.95 | 0.568 | 0.204 | 0.532 |
| TeleMelody | 46.81 | 51.77 | 2.60 | - | - | - |
| GPT-3.5 | 31.24 | 38.52 | 3.01 | 0.641 | 0.142 | 0.603 |
| GPT-4 | 36.43 | 42.94 | 2.87 | 0.654 | 0.158 | 0.610 |
| **SongComposer** | **50.75** | **57.71** | **2.20** | **0.697** | **0.234** | **0.657** |

### 消融实验

| 消融维度 | 具体结果 |
|----------|----------|
| 音高初始化 | Scalar > Interpolation > Average > Gaussian（MD: 2.33 vs 3.07/3.41/2.90） |
| Motif重复阈值 | 阈值=10时RR-MD平衡最优（RR=2.03, MD=2.33） |
| 对齐粒度 | Word-level > Line-level > Song-level（MD: 2.12 vs 2.42 vs 3.71） |
| 短语级token | 有短语token的续写效果更好（MD: 2.12 vs 2.58, BS: 0.662 vs 0.612） |

### 主观评估

| 方法 | 歌词配旋律 HMY. | MLC. | 文本生成歌曲 OVL. | REL. |
|------|----------------|------|-------------------|------|
| GPT-3.5 | 1.68 | 1.88 | 2.53 | 2.95 |
| GPT-4 | 2.82 | 2.79 | 2.43 | 3.27 |
| **SongComposer** | **3.82** | **3.76** | **3.41** | **3.88** |

### 关键发现

- **统一框架有效：** 单一模型在四个子任务上全面超越GPT-4和专用模型
- **标量初始化关键：** 显式编码音高间数值关系比随机初始化显著更优，Recall Rate从1.44提升到2.03
- **词级对齐必要：** 相比行级和曲级，词级对齐将Melody Distance从3.71降至2.12
- **结构感知有效：** Motif训练提升了旋律的重复性和结构感，短语token提升了段落组织能力

---

## 亮点

- 首次使用LLM同时生成歌词和旋律，实现统一的歌曲创作框架
- 标量音高初始化巧妙利用数值关系编码音高语义，实质上是一种音乐知识注入
- 渐进式训练策略从motif到phrase逐步引入结构信息，符合人类作曲认知
- 自建的SongCompose数据集（8K精确词级对齐配对数据）填补了该领域数据空白

## 局限性

- 音高范围限制在C3-B5（MIDI 48-83），无法覆盖更宽音域的创作需求
- 基于InternLM2-7B的模型规模限制了创作复杂度和多样性
- 配对数据集仅8K首歌曲，规模仍然有限
- 评估偏重旋律相似度指标，对音乐创意性和可听性的评估不足
- 仅支持单声部（人声音轨），无法生成伴奏等多声部内容

## 相关工作

- **歌曲子任务：** SongMASS (Sheng et al., 2021) Lyric↔Melody双向生成，TeleMelody (Ju et al., 2022) 模板旋律生成
- **LLM音乐生成：** ChatMusician (Yuan et al., 2024) 纯音乐符号生成，本文扩展到歌词+旋律
- **符号表示：** REMI (Huang & Yang, 2020) 基于拍的音乐表示
- **配对数据集：** M4Singer (Zhang et al., 2022b) 提供约700首中文歌曲

---

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model](infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)
- [A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [Improve Language Model and Brain Alignment via Associative Memory](improve_language_model_and_brain_alignment_via_associative_memory.md)
- [SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)

<!-- RELATED:END -->
