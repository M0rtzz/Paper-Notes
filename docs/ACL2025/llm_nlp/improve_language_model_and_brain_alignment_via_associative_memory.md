---
title: >-
  [论文解读] Improve Language Model and Brain Alignment via Associative Memory
description: >-
  [ACL 2025][LLM/NLP][brain alignment] 通过模拟联想记忆对文本进行数据增强，以及对 LLM 进行联想记忆指令微调，本文证明两种方式均能显著提升语言模型与人脑在语音理解任务中的对齐程度，尤其在内侧颞叶等联想记忆相关脑区。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "brain alignment"
  - "associative memory"
  - "fMRI"
  - "指令微调"
  - "GPT-2"
  - "LLaMA-2"
---

# Improve Language Model and Brain Alignment via Associative Memory

**会议**: ACL 2025  
**arXiv**: [2505.13844](https://arxiv.org/abs/2505.13844)  
**代码**: [GitHub](https://github.com/lemonsis/Association)  
**领域**: 认知神经科学 / 语言模型  
**关键词**: brain alignment, associative memory, fMRI, 指令微调, GPT-2, LLaMA-2  

## 一句话总结

通过模拟联想记忆对文本进行数据增强，以及对 LLM 进行联想记忆指令微调，本文证明两种方式均能显著提升语言模型与人脑在语音理解任务中的对齐程度，尤其在内侧颞叶等联想记忆相关脑区。

## 研究背景与动机

- **核心问题**: 语言模型的激活可以线性映射到人脑 fMRI 活动（即 brain score），但现有研究很少探索联想记忆在这种对齐中的作用。联想记忆是人类语言理解的关键认知过程，能够将相关概念和信息联系起来。
- **研究动机**: 人类在听故事时会自动进行联想（如听到"医院"联想到"医生""护士"），而语言模型缺乏这种机制。如果在模型输入中模拟联想记忆内容，或训练模型生成联想内容，是否能提升模型与大脑的对齐？
- **两个研究问题**: ① 模拟联想记忆（数据增强）是否能提升 brain score？② 指导 LLM 生成联想内容（指令微调）是否能提升 brain score？

## 方法详解

### 整体框架

三阶段实验设计：**Brain Score 计算**（基线对齐度）→ **联想记忆数据增强**（模拟联想内容）→ **指令微调**（训练 LLM 生成联想内容）。使用 Narratives fMRI 数据集，包含 345 名被试听 27 个英文故事的 fMRI 记录。

### 关键设计

1. **Brain Score 计算**: 选用自回归模型（GPT-2 / LLaMA-2），提取各层激活，通过 FIR 模型对齐时间维度，使用 Ridge 回归将模型激活映射到 fMRI 信号，Pearson 相关系数作为 brain score。创新性地设计了 brain score ceiling test，用一半被试预测另一半以评估可解释上限。

2. **联想记忆数据增强**: 将原始文本扩展为包含联想内容的增强文本。两种粒度——句子级（完整语义句）和词级（名词/形容词/动词短语）；两种标注方式——人工标注（人决定在哪里添加联想）和 GPT-4 标注（每 4 句自动生成联想）。设计随机增强作为对照组证明提升源于联想记忆而非数据量。

3. **指令微调 (Association 数据集)**: 构建 1000 样本的 Association 数据集，输入为故事段落+鼓励联想记忆的指令，输出为联想内容。使用 LoRA 和冻结层微调两种方式对 LLaMA-2 进行 SFT，微调后重新计算 brain score。

### 核心公式

- **Associative Memory Score**: $\mathcal{F}(X^{(l)}) = \mathcal{R}(X_{mem}^{(l)}) - \mathcal{R}(X^{(l)})$
- **Instruction Tuning Score**: $\mathcal{M}(X^{(l)}) = (\mathcal{R}(X_{sft}^{(l)}) - \mathcal{R}(X^{(l)})) / \mathcal{R}(X^{(l)})$

## 实验

### Brain Score 基线

| 模型 | 最佳层 | 最高 Brain Score |
|------|:---:|:---:|
| GPT-2 | 第 9 层 (of 12) | 0.126 |
| LLaMA-2 | 第 14 层 (of 32) | 0.146 |

LLaMA-2 因参数量更大、训练数据更多，对齐度更高。左脑半球 brain score 高于右脑。

### 联想记忆增强结果

| 增强方式 | Brain Score 提升范围 | 最佳设置 |
|------|:---:|:---:|
| 词级-人工标注 | 0.0014 — 0.05 | 最优 |
| 句级-人工标注 | 正增长但弱于词级 | 次优 |
| 词级-GPT-4 | 正增长但弱于人工 | — |
| 随机增强 | 0 或负增长 | 对照组 |

### 指令微调结果

| 方法 | MTL 脑区提升 | 顶叶区提升 |
|------|:---:|:---:|
| LoRA | 2%—7% | 50%—60% |
| 冻结层微调 | 2%—7% | 50%—60% |

### 关键发现

- 词级联想优于句级联想——名词/动词/形容词携带更密集信息，与神经科学研究一致
- 人工标注优于 GPT-4——GPT-4 无法判断何时触发联想
- 随机数据增强无效甚至负面影响——证明提升确实源于联想记忆
- 联想记忆指令微调在内侧颞叶 (MTL) 和工作记忆相关脑区均有显著提升
- LLaMA-2 在大多数 ROI 中表现优于 GPT-2

## 亮点

- 首次系统研究联想记忆对语言模型-大脑对齐的影响，填补了认知神经科学与 NLP 交叉领域的空白
- 构建了 Association 数据集，证明"指导模型联想"可增强脑-模型对齐，具有理论新颖性
- 设计了完善的对照实验（随机增强），排除了数据量带来的混淆因素

## 局限性

- 仅使用英文故事和英文被试的 fMRI 数据，无法泛化到其他语言和文化
- 标注者与 fMRI 被试可能背景不同，联想内容与实际脑活动存在固有不匹配
- Association 数据集仅 1000 样本，规模有限
- 联想记忆的生物学机制极其复杂，本文的模拟方式仍是高度简化的

## 相关工作

- **语言模型与大脑对齐**: Jain & Huth (2018); Caucheteux & King (2020); Goldstein et al. (2022) — 模型激活可线性映射到 fMRI/MEG 信号
- **联想记忆研究**: Anderson & Bower (2014); Eichenbaum (2017) — 联想记忆的认知理论，海马体在记忆编码/检索中的作用
- **LLM 微调与大脑**: Moussa et al. (2024) — 用脑相关语义微调语音模型提升对齐

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★☆ |
| 实用性 | ★★★☆☆ |
| 实验充分度 | ★★★★☆ |
| 写作质量 | ★★★★☆ |
| 总评 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)

</div>

<!-- RELATED:END -->
