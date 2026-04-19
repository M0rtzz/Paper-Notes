---
title: >-
  [论文解读] Facial Affective Behavior Analysis with Instruction Tuning
description: >-
  [ECCV 2024][人体理解][面部情感分析] 提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。
tags:
  - ECCV 2024
  - 人体理解
  - 面部情感分析
  - 指令微调
  - 多模态大语言模型
  - Action Unit识别
  - LoRA
---

# Facial Affective Behavior Analysis with Instruction Tuning

**会议**: ECCV 2024  
**arXiv**: [2404.05052](https://arxiv.org/abs/2404.05052)  
**代码**: [有](https://johnx69.github.io/FABA/)  
**领域**: 人体理解  
**关键词**: 面部情感分析, 指令微调, 多模态大语言模型, Action Unit识别, LoRA

## 一句话总结

提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。

## 研究背景与动机

面部情感行为分析（FABA）是理解人类心理状态的关键技术，涵盖面部表情识别（FER）和动作单元识别（AUR）两大任务。传统方法将其视为判别式多分类/多标签问题，存在三大局限：

**粗粒度情绪描述**：仅将表情归入 7 类基本情绪（高兴、悲伤、愤怒等），无法表达複合情绪（如"带着勉强微笑的悲伤"）、夸张表情、情绪程度等细微差别。

**缺乏推理能力**：二值 AU 标注只能标记是否激活，不能解释肌肉运动的程度和因果关系，如 AU6（提颧肌）激活程度差异无法体现。

**无法迁移 MLLM 优势**：多模态大语言模型在通用视觉理解任务上表现出色，但直接用于 FABA 面临**缺少合适数据集**、**缺乏评估基准**和**难以捕获面部结构先验**三大挑战。

作者认为，自然语言描述（而非离散标签）是表征面部情感行为的更优方式——它既能捕捉情感的复杂性与细微差别，又对人类更加直观可量化。这一洞察驱动了整个工作的设计。

## 方法详解

### 整体框架

EmoLA 基于 LLaVA-1.5 构建，包含四个核心组件：视觉专家（CLIP-L/14 + MLP）、面部先验专家（预训练面部关键点检测器 + MLP）、语言专家（tokenizer + word embedding）和语言解码器（Vicuna LLM + LoRA）。输入面部图像经两个专家分别提取视觉 token $H_v$ 和先验 token $H_p$，与指令文本 token $H_q$ 拼接后送入冻结的 LLM 解码器，以自回归方式生成描述。

### 关键设计

1. **FABA-Instruct 指令微调数据集**：首个面向 FABA 的指令微调数据集。从 AffectNet 随机采样 19,877 张野外人脸图像，使用 100 个精心设计的模板通过 GPT-4V 标注情绪和 AU 的细粒度描述。情绪描述平均长度 50.47 词，AU 描述平均 207.35 词，包含肌肉运动原因、可能对应的情绪、AU 间关系等推理信息。描述可以表达複合情绪、夸张表情、情绪程度和未定义情绪等传统标签无法覆盖的内容。

2. **面部先验专家模块（Facial Prior Expert）**：CLIP 视觉编码器在通用图像-文本对上训练，难以捕获面部结构信息（如关键点位置）。因此引入预训练的 InsightFace 面部关键点检测器 $f_p$ 提取面部先验特征 $Z_P = f_p(X_V)$，再通过 MLP 投影到 token 嵌入空间：

$$H_p = g_\theta(Z_P)$$

该先验 token 为 LLM 提供视觉编码器可能忽略的面部结构细节（如关键点拓扑关系），实验证明**仅用单个先验 token 也能保持一定识别能力**，说明关键点先验对 FABA 任务具有高度代表性。

3. **REGE 评测指标**：传统 FABA 指标（准确率/F1）仅关注识别能力，NLG 指标（BLEU/ROUGE）仅关注文本质量，均不完整。REGE 同时评估识别与生成：

$$S_{rege} = S_{re} + S_{ge}$$

其中 $S_{re}$ 为识别分数（FER 用准确率，AUR 用 F1），$S_{ge}$ 为 ROUGE 分数。对于情绪识别，通过预定义同义词表从自由文本中提取情绪类别再计算准确率。

### 训练策略

训练中冻结视觉编码器、先验编码器、word embedding 和 LLM 解码器，仅优化三组参数 $\Theta = \{\theta, \gamma, \phi\}$（先验投影器、视觉投影器、LoRA）。优化目标为自回归语言建模损失：

$$p(X_A|X_V, Z_P, X_Q) = \prod_{i=1}^{L} p_\Theta(x_i | X_V, Z_P, X_Q, X_{A,<i})$$

使用 AdamW 优化器，学习率 1e-4，LoRA rank 128，仅训练 1 个 epoch，8 块 A6000 GPU。EmoLA 仅微调约 10% 参数即超越全量微调的基线。

## 实验关键数据

### 主实验

**表情识别（RAF-DB）**：

| 方法 | Accuracy (%) | 备注 |
|------|-------------|------|
| APViT | 91.98 | 之前 SOTA |
| POSTER | 92.05 | 之前 SOTA |
| **EmoLA** | **92.05** | 持平 SOTA |

**AU 识别（DISFA，8 AU 平均 F1）**：

| 方法 | Avg. F1 (%) | 备注 |
|------|------------|------|
| PIAP | 63.8 | 之前 SOTA |
| **EmoLA** | **65.1** | +1.3% |

**AU 识别（GFT，10 AU 平均 F1）**：

| 方法 | Avg. F1 (%) | 备注 |
|------|------------|------|
| EmoCo | 58.6 | 之前 SOTA |
| **EmoLA** | **62.1** | +3.5% |

**FABA-Bench（综合 REGE 评分）**：

| 方法 | Emotion $S_{rege}$ | AU $S_{rege}$ |
|------|-------------------|--------------|
| MiniGPT4-v2 | 77.8 | 37.8 |
| mPLUG-Owl2 | 82.0 | 55.7 |
| Shikra | 94.6 | 86.6 |
| LLaVA-1.5 | 93.9 | 91.4 |
| **EmoLA** | **96.2** | **91.5** |

### 消融实验

**先验 token 的影响**：

| 配置 | Emotion $S_{re}$ | AU $S_{re}$ | 说明 |
|------|-----------------|------------|------|
| 仅 $H_p$ | 41.2 | 40.5 | 单 token 仍有识别力 |
| 仅 $H_v$ | 62.5 | 55.3 | 视觉 token 基线 |
| $H_v + H_p$ | **64.5** | **56.3** | 先验提供互补信息 |

**微调策略的影响**：

| 配置 | Emotion $S_{re}$ | AU $S_{re}$ | 说明 |
|------|-----------------|------------|------|
| 仅 $g_\theta$ | 44.9 | 47.7 | 仅调先验投影器 |
| 仅 $h_\phi + h_\gamma$ | 63.0 | 55.6 | 仅调 LoRA + 视觉投影器 |
| 两者都调 | **64.5** | **56.3** | 协同最优 |

### 关键发现

- EmoLA 仅用 10% 可训练参数即超越全量微调的 LLaVA-1.5，证明 LoRA + 面部先验的高效性
- 面部先验 token 即使作为唯一输入（无视觉 token）也保持一定识别能力，说明关键点特征对 FABA 高度有效
- 在 BP4D 上与 ReCoT 仅差 0.6%（64.2 vs 64.8），差距来自 ReCoT 的一致性正则化和协同训练

## 亮点与洞察

- **数据层面的突破**：首次构建 FABA 指令微调数据集，将情感分析从"分类"升级为"描述+推理"
- **轻量高效**：面部先验 token 仅增加 1 个 token 输入，却带来显著提升，设计极简
- **统一评测**：REGE 指标首次将识别能力和文本生成质量纳入同一评测框架

## 局限与展望

- 目前仅使用面部关键点作为先验，未探索面部识别特征（ArcFace）、面部解析等其他先验
- 仅处理单张图像，未扩展到视频流的时序情感分析
- FABA-Instruct 的 GPT-4V 标注可能存在幻觉或不一致，未进行大规模人工校验
- 训练数据仅约 19K 图像，规模较小

## 相关工作与启发

- 与 LLaVA-1.5 架构一致，但通过增加面部先验专家和 LoRA 适配实现任务特化
- 启发：领域特定的先验知识可通过轻量级 token 注入方式融入通用 MLLM，值得推广到其他细粒度视觉理解任务（医学影像、遥感等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个 FABA 指令微调工作，数据集+基准+模型三位一体
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 个传统数据集 + 自建基准，消融实验完整
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图表丰富，描述分析部分很有说服力
- **价值**: ⭐⭐⭐⭐ 为 FABA 社区打开了 MLLM 时代的大门

<!-- RELATED:START -->

## 相关论文

- [Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)
- [Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](../../AAAI2026/human_understanding/behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)
- [Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](../../CVPR2026/human_understanding/breaking_the_tuning_barrier_zerohyperparameters_yi.md)
- [MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/human_understanding/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)

<!-- RELATED:END -->
