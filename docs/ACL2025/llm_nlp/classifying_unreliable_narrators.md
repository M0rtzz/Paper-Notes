---
title: >-
  [论文解读] Classifying Unreliable Narrators with Large Language Models
description: >-
  [LLM/NLP] 借鉴文学叙事学理论，定义三种不同层次的不可靠叙事者（intra-narrational / inter-narrational / inter-textual），构建专家标注数据集 TUNa，系统评估 LLM 在分类不可靠叙事者任务上的表现。
tags:
  - LLM/NLP
---

# Classifying Unreliable Narrators with Large Language Models

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2506.10231](https://arxiv.org/abs/2506.10231) |
| 代码 | [GitHub](https://github.com/adbrei/unreliable-narrators) |
| 领域 | NLP / 文本分类 / 叙事分析 |
| 关键词 | 不可靠叙事者, 叙事学, LLM分类, 课程学习, 专家标注数据集 |

## 一句话总结

借鉴文学叙事学理论，定义三种不同层次的不可靠叙事者（intra-narrational / inter-narrational / inter-textual），构建专家标注数据集 TUNa，系统评估 LLM 在分类不可靠叙事者任务上的表现。

## 研究背景与动机

- **问题定义**：在日常生活中，我们频繁接触第一人称叙事（评论、社交媒体、求职信等），判断叙事者是否可靠是信息安全传递的关键问题。不可靠叙事者（unreliable narrator）指那些**无意中**误导读者的叙事者（区别于故意欺骗）。
- **现有不足**：此前没有任何工作使用自动化方法分析叙事者的不可靠性，也没有可用的标注数据集。
- **叙事学基础**：作者借鉴 Hansen (2007) 的叙事学分类法，将不可靠性分为三个层次，从具象到抽象逐渐递增：
    - **(1) Intra-narrational（叙事内不可靠）**：叙事者表现出"verbal tics"，如对冲语言（hedging）、选择性记忆、承认偏见等文本线索
    - **(2) Inter-narrational（叙事间不可靠）**：存在第二声音与叙事者矛盾（如他人对话中的反驳），或叙事者在过去与现在表现一致的不可靠性
    - **(3) Inter-textual（文本间不可靠）**：叙事者符合经典不可靠角色原型——天真者（naïf）、疯子（madman）、骗子（pícaro）、小丑（clown）
- **核心挑战**：不可靠性线索往往是微妙的、上下文依赖的，可能散布在文本各处，有时需要对叙事者的情感和心理状态进行深层推理。

## 方法详解

### 整体框架

本文将不可靠叙事者识别建模为三个独立的分类任务：
- **Intra-narrational**：二分类（有verbal tics vs 可靠）
- **Inter-narrational**：三分类（same-unreliable-character-over-time / other-character-contradiction / 可靠）
- **Inter-textual**：五分类（naïf / madman / pícaro / clown / 可靠）

### 关键设计

1. **TUNa 数据集构建**：收集来自博客（PersonaBank）、Reddit帖子（r/AITA）、酒店评论（Deceptive Opinion）和文学作品（Project Gutenberg）四个领域的第一人称叙事文本。每个样本由至少2名英语文学专业的专家标注，标注者间Cohen's Kappa一致性达到 κ=0.71~0.75（substantial agreement）。不一致标签通过讨论解决。
2. **课程学习（Curriculum Learning）**：将训练样本按"歧义程度"排序——LLM先统计每个标签类型的特征数量，候选标签越少的样本越容易。先在简单子集上用LoRA微调，再在困难子集上微调，逐步提升模型能力。
3. **从文学到现实的迁移**：使用文学领域（Fiction）的训练样本进行模型训练，在博客、Reddit、评论等现实领域上进行out-of-domain测试，验证从文学中学习不可靠性知识的可迁移性。

### 损失函数

使用标准分类交叉熵损失，配合LoRA适配器（8-bit量化）进行参数高效微调，训练3个epoch。

## 实验关键数据

### 主实验（Llama3.1-8B，F1 macro）

| 任务 | 方法 | Fiction | Blog | Subreddit | Review |
|------|------|---------|------|-----------|--------|
| Intra-nar | CL | 58.51 | 53.94 | 50.04 | **67.17** |
| Intra-nar | Fine-tuned | 50.09 | 50.63 | 49.00 | 55.85 |
| Intra-nar | Zero-Shot | 45.17 | 45.56 | 47.41 | 58.46 |
| Inter-nar | CL | 34.59 | **35.92** | 30.91 | **35.29** |
| Inter-nar | Fine-tuned | 34.63 | 28.73 | 25.59 | 36.59 |
| Inter-tex | CL | 27.42 | 19.58 | 13.49 | 16.72 |
| Inter-tex | Fine-tuned | 28.59 | 18.99 | 10.85 | 17.54 |

### 消融：不同模型规模对比（所有领域平均 F1 macro）

| 模型 | Intra-nar (CL) | Inter-nar (CL) | Inter-tex (CL) |
|------|---------------|----------------|----------------|
| Llama3.1-8B | 57.42 | 34.18 | 19.30 |
| Llama3.3-70B | 51.26 | 33.49 | 21.04 |
| Mistral-7B | 55.76 | 31.15 | **29.68** |
| ModernBERT | 39.94 | 27.07 | 16.98 |

### 关键发现

1. **任务难度递增**：Intra-narrational 最容易，Inter-textual 最难（需要更抽象的推理）
2. **课程学习有效**：CL在小模型上显著优于普通微调，但大模型（70B）的few-shot已可与CL媲美
3. **跨领域迁移可行**：从Fiction学到的知识可以合理迁移到Blog/Subreddit/Review等真实领域
4. **性别偏差**：男性叙事者被正确预测的比例高于女性叙事者
5. **叙事风格影响**：对话风格有助于intra-narrational预测，描述风格有助于inter-narrational和inter-textual预测

## 亮点与洞察

- 将文学理论（叙事学）与NLP任务创新结合，定义了一个全新的有意义的任务——不可靠叙事者自动识别
- TUNa 数据集覆盖多领域（文学/博客/Reddit/评论），专家标注质量高（每条样本约5分钟标注时间）
- 课程学习策略设计巧妙：基于"歧义候选标签数量"定义样本难度，而非传统的loss大小
- 发现从文学作品中学到的不可靠性知识可以迁移到真实世界文本，打开了"从小说学习现实世界理解"的新方向

## 局限性

- 仅处理短文本（最长1050 tokens），未考虑完整小说或长本文
- 数据集仅包含英语，未覆盖其他语言
- 数据集规模较小（817条样本），受限于专家标注的高成本
- 即使最好的方法，Inter-textual任务的F1仍然很低（~30%），说明任务极具挑战性

## 相关工作与启发

- 与虚假信息检测（misinformation detection）和欺骗检测（deception detection）相关但本质不同：本文关注的是**无意的**不可靠，而非**故意的**欺骗
- 角色理解（character understanding）、情感分析（protagonist emotion）等叙事AI方向的自然延伸
- 课程学习的"基于歧义度定义难度"策略可推广到其他含有模糊标签的NLP任务

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

<!-- RELATED:END -->
