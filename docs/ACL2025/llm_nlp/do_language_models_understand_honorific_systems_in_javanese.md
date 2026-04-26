---
title: >-
  [论文解读] Do Language Models Understand Honorific Systems in Javanese?
description: >-
  [ACL 2025][LLM/NLP][honorifics] 构建首个爪哇语敬语语料库 Unggah-Ungguh（4,024 句，覆盖四个敬语层级），通过分类/风格转换/跨语言翻译/对话生成四个任务系统评估 LLM 对爪哇语敬语系统的理解能力，发现即使最强闭源模型（GPT-4o）的零样本分类准确率也仅 53.5%，且普遍偏向特定敬语层级。
tags:
  - ACL 2025
  - LLM/NLP
  - honorifics
  - Javanese
  - low-resource
  - Unggah-Ungguh
  - linguistic evaluation
---

# Do Language Models Understand Honorific Systems in Javanese?

**会议**: ACL 2025  
**arXiv**: [2502.20864](https://arxiv.org/abs/2502.20864)  
**代码**: https://github.com/JavaneseHonorifics/Unggah-Ungguh  
**领域**: LLM/NLP - 低资源语言评估  
**关键词**: honorifics, Javanese, low-resource, Unggah-Ungguh, linguistic evaluation

## 一句话总结
构建首个爪哇语敬语语料库 Unggah-Ungguh（4,024 句，覆盖四个敬语层级），通过分类/风格转换/跨语言翻译/对话生成四个任务系统评估 LLM 对爪哇语敬语系统的理解能力，发现即使最强闭源模型（GPT-4o）的零样本分类准确率也仅 53.5%，且普遍偏向特定敬语层级。

## 研究背景与动机

**领域现状**：爪哇语拥有超过 9,800 万使用者，其核心特征之一是称为 Unggah-Ungguh Basa 的复杂敬语系统，包含四个层级——Ngoko（最非正式）、Ngoko Alus（稍正式）、Krama（正式）、Krama Alus（最正式）。敬语选择取决于说话者、听者和被提及者之间的社会关系。

**现有痛点**：(1) 现有爪哇语语料库的敬语层级分布严重不平衡，大多偏向 Ngoko；(2) 缺乏专门标注的敬语语料库用于 NLP 任务；(3) 随着 LLM 越来越多地充当个人助手，其理解和生成恰当敬语的能力直接影响文化敏感度和用户信任。

**核心矛盾**：敬语系统要求模型不仅理解语义，还要捕捉社会等级、对话角色和语境等语用信息——这对现有模型是极大挑战，尤其在低资源语言场景下。

**本文目标** 系统评估 LLM 对爪哇语四级敬语系统的理解和生成能力，识别其偏向和局限。

**切入角度**：构建平衡分布的敬语语料库，设计覆盖理解和生成的四项基准任务。

**核心 idea**：通过构建首个标注爪哇语四级敬语的平衡语料库和四项评估任务，揭示当前 LLM 对复杂敬语系统的理解严重不足。

## 方法详解

### 整体框架
构建 Unggah-Ungguh 语料库 → 设计四个评估任务 → 使用微调模型和零/少样本通用模型进行对比评估。微调模型用于分类任务并作为后续任务的自动评估工具；通用模型涵盖闭源（GPT-4o、Gemini 1.5 Pro）和开源（Llama 3.1 8B、Gemma2 9B、Sailor2 8B、SahabatAI）两类。

### 关键设计

1. **Unggah-Ungguh 语料库构建**:

    - 功能：从《Kamus Unggah-Ungguh Basa Jawa》等四本权威参考书中手动构建 4,024 句标注语料库
    - 核心思路：原始资料未数字化，需扫描→OCR→母语者两阶段校正。第二阶段独立审核发现并修正了 58 个错误（1.5%）。最终 Shannon 信息熵达 1.88，超过其他 9 个已有数据集，表明分布最平衡
    - 设计动机：现有爪哇语语料库的敬语分布极度不平衡（多数集中在 Ngoko），无法公平评估模型能力

2. **Task 1：敬语层级分类**:

    - 功能：将输入文本分类到四个敬语层级之一
    - 核心思路：微调 Javanese BERT/DistilBERT/GPT-2 和 LSTM/规则基线。Javanese DistilBERT 达到最高 95.65% 准确率，作为后续 Task 4 的自动评估器
    - 设计动机：评估模型对敬语层级的识别能力——这是理解敬语系统的基础

3. **Task 2：敬语风格转换**:

    - 功能：将给定文本从一个敬语风格转换到另一个（如 Ngoko → Krama Alus）
    - 核心思路：零样本翻译，评估模型是否能在保持语义的同时改变敬语层级
    - 设计动机：敬语转换需要精确的词汇替换和语法调整，是理解敬语系统深度的直接测试

4. **Task 3：跨语言敬语翻译**:

    - 功能：在特定敬语层级的爪哇语和印尼语之间进行翻译
    - 核心思路：印尼语缺乏显式敬语系统，而爪哇语有丰富的敬语层级，两者之间的 KL 散度高达 2.26，词汇分布差异大
    - 设计动机：测试模型是否能在跨语言场景中保留敬语信息

5. **Task 4：对话生成**:

    - 功能：给定两个说话者的社会地位（如学生和老师）和对话上下文，生成使用恰当敬语的对话
    - 核心思路：手动构建 160 个评估场景，使用微调的 DistilBERT 自动评估生成文本的敬语层级是否正确
    - 设计动机：最具挑战性的任务——模型必须同时理解角色关系、敬语规则和对话连贯性

## 实验关键数据

### 主实验（Task 1：敬语分类）

| 模型 | 准确率 | F1 |
|------|--------|-----|
| Dictionary-Based | 88.37 | 88.64 |
| LSTM | 93.47 | 91.34 |
| Javanese BERT (微调) | 93.91 | 93.97 |
| Javanese DistilBERT (微调) | **95.65** | **95.66** |
| GPT-4o (零样本) | 53.50 | 40.70 |
| Gemini 1.5 Pro (零样本) | 50.70 | 45.40 |
| Llama 3.1 8B (零样本) | 43.00 | 24.00 |

### 消融实验（GPT-4o 逐层级分类性能）

| 敬语层级 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Ngoko | 78.00 | 91.10 | 84.00 |
| Ngoko Alus | 0 | 0 | 0 |
| Krama | 53.50 | 26.00 | 35.00 |
| Krama Alus | 29.90 | 82.40 | 43.80 |

### 关键发现
- 微调的专用模型（DistilBERT 95.65%）远超通用 LLM（GPT-4o 53.5%），说明爪哇语敬语仍是低资源难题
- GPT-4o 完全无法识别 Ngoko Alus 层级（F1=0），存在严重的层级偏向
- 闭源模型在分类中偏向 Ngoko 和 Krama Alus 两个极端层级，忽略中间层级
- 规则基线（88.37%）已经很强，因为敬语很大程度上通过词汇替换实现
- 跨语言翻译中，KL 散度和 Jensen 分数显示爪哇语-印尼语之间存在显著的词汇鸿沟

## 亮点与洞察
- 首次系统评估 LLM 在复杂敬语系统上的能力，填补了低资源语言语用学评估的空白
- GPT-4o 对 Ngoko Alus 完全失明的发现极具警示意义——表面上的"多语言能力"在文化细粒度上完全不足
- 语料库构建过程严谨（扫描→OCR→两阶段母语者校验），为其他低资源语言的数字化提供了范本

## 局限与展望
- 语料库规模较小（4,024 句），可能不足以训练更大模型
- 仅评估了四个敬语层级，实际使用中还有更细粒度的区分
- 未测试微调通用 LLM（如用 Unggah-Ungguh 微调 Llama）的效果

## 相关工作与启发
- **vs Japanese Honorific Corpus (Liu & Kobayashi, 2022)**：本文的爪哇语敬语系统更复杂（四级 vs 日语的"尊敬语/谦让语"二分），且 Yule's K 值更低（105.43 vs 125.54），表明词汇多样性更高
- **vs Wongso et al. (2021)**：后者做爪哇语预训练模型，但未涉及敬语系统
- **vs Marreddy et al. (2022)**：后者指出低资源语言模型因缺乏标注数据而表现不佳，本文正是通过构建专门语料库来解决这一问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个爪哇语敬语评估基准，问题定义清晰独特
- 实验充分度: ⭐⭐⭐⭐ 四个任务全面覆盖，多类模型对比，但语料规模有限
- 写作质量: ⭐⭐⭐⭐ 语言学背景介绍详实，实验组织清晰
- 价值: ⭐⭐⭐⭐ 对低资源语言 NLP 和文化敏感 AI 研究有重要参考意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)
- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)

<!-- RELATED:END -->
