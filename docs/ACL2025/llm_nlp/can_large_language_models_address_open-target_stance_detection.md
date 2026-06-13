---
title: >-
  [论文解读] Can Large Language Models Address Open-Target Stance Detection?
description: >-
  [ACL 2025][LLM 其他][stance detection] 提出开放目标立场检测（OTSD）任务——目标在训练时未见且不作为输入提供，系统评估了 GPT、Gemini、LLaMA、Mistral 四个系列共 8 个 LLM 在目标生成和立场检测两阶段的表现，发现 LLM 整体优于现有 TSE 方法，但在目标未显式出现时表现明显下降。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "stance detection"
  - "open-target"
  - "zero-shot"
  - "LLM evaluation"
  - "target generation"
  - "NLP"
---

# Can Large Language Models Address Open-Target Stance Detection?

**会议**: ACL 2025  
**arXiv**: [2409.00222](https://arxiv.org/abs/2409.00222)  
**代码**: [AbuUbaida/opentarget](https://github.com/AbuUbaida/opentarget)  
**领域**: LLM/NLP  
**关键词**: stance detection, open-target, zero-shot, LLM evaluation, target generation, NLP

## 一句话总结
提出开放目标立场检测（OTSD）任务——目标在训练时未见且不作为输入提供，系统评估了 GPT、Gemini、LLaMA、Mistral 四个系列共 8 个 LLM 在目标生成和立场检测两阶段的表现，发现 LLM 整体优于现有 TSE 方法，但在目标未显式出现时表现明显下降。

## 研究背景与动机
**领域现状**: 立场检测（Stance Detection）旨在判断文本对某个目标的态度（支持/反对/中立），是社交媒体分析和舆情监控的核心任务。现有研究主要聚焦零样本立场检测（ZSSD），即目标在训练时未见但在推理时作为输入提供。

**现有痛点**: 几乎所有 ZSSD 方法假设目标已知或人工标注后提供给模型，但在真实场景中目标往往未知、不常见或未在文本中显式表达，且对每个可能目标进行标注成本极高。

**核心矛盾**: 唯一尝试解决此问题的 TSE（Li et al., 2023）方法虽然能从文本生成目标，但依赖预定义的目标列表进行映射（如将"Religious diversity"映射到列表中最近的"Atheism"），这在开放场景中不切实际。

**本文解决**: 定义 OTSD 任务——直接从文本生成目标并基于生成的目标检测立场，完全不依赖预定义目标列表，公式化为 $x \xrightarrow{\text{generate}} t', \; x + t' \rightarrow y$。

**切入角度**: 利用 LLM 的零样本生成能力替代传统的关键短语提取+目标列表映射流程，本质上将 OTSD 作为 LLM 能力评估的 testbed。

**核心 idea**: LLM 的强大上下文理解能力使其天然适合 OTSD 的两步流程（目标生成+立场检测），且联合单步提示（TG&SD）优于两步分离提示（TG+SD）。

## 方法详解

### 整体框架
- **功能**: 利用 LLM 在无任何目标信息输入的条件下，从文本中识别讨论目标并判断文本对该目标的立场（支持/反对/中立）。
- **为什么**: 现实应用中用户不会预先告知目标是什么，模型需要自行理解文本在讨论什么主题并给出立场判断，TSE 依赖预定义目标列表的方式无法扩展到开放领域。
- **怎么做**: 采用 "Task Definition" 提示策略（先命名任务、定义输入输出、再要求模型执行），设计两种提示方案：(1) TG+SD 两步法——先生成目标再检测立场，让 LLM 分步聚焦；(2) TG&SD 单步联合法——在一个 prompt 中同时完成目标生成和立场检测，让模型更好理解文本-目标-立场的关系。

### 关键设计
1. **OTSD 任务定义与形式化**

    - **功能**: 将开放目标立场检测定义为两阶段问题：目标生成（TG）和立场检测（SD），移除 TSE 中的目标列表映射步骤。
    - **为什么**: TSE 的映射步骤要求拥有所有可能目标的完整列表，这在新闻评论、社交媒体等目标高度多样化的场景中完全不现实（VAST 数据集有 2,145 个唯一目标，EZSTANCE 有 6,873 个）。
    - **怎么做**: 形式化为 $x \xrightarrow{\text{generate}} t'$，$x + t' \rightarrow y$，其中 $t'$ 是直接生成的目标（非映射结果），立场 $y \in \{\text{favor, against, none}\}$ 基于生成的目标预测。

2. **BTSD 目标质量评估指标**

    - **功能**: 提出基于 BERTweet 的立场分类器作为目标生成质量的自动评估指标（BTSD score）。
    - **为什么**: OTSD 生成的目标与 gold target 措辞不同但语义相关（如 "gun control" vs "permit to carry gun"），传统精确匹配指标不适用；语义相似度（SemSim）与人类判断的相关性仅 0.57-0.59，不够可靠。
    - **怎么做**: 用 4 个经典立场检测数据集（SemEval, AM, COVID-19, P-Stance，含 19 个目标）训练 BERTweet 分类器，将生成的目标+文本输入分类器，用 F1-macro 作为目标质量代理指标。该指标与人类判断的 Kendall τ 相关性达 0.74-0.85，远优于 SemSim。

3. **显式 vs 非显式目标场景区分**

    - **功能**: 将数据集样本划分为目标在文本中显式提及（explicit）和未显式提及（non-explicit）两种情况，分别评估。
    - **为什么**: TSE 原始工作未区分这两种情况，但二者难度差异极大，非显式目标需要模型进行深层推理和隐含语义理解。
    - **怎么做**: 通过停用词去除、特殊字符清理、词形还原后检查目标词是否出现在文本中进行划分。

## 实验关键数据

### 表1: 三个数据集统计

| 数据集 | 来源 | 样本数 | 唯一目标数 | 显式/非显式 | 立场类别 |
|--------|------|--------|-----------|------------|---------|
| TSE | 推文 | 3,000 | 6 | 1,804/1,196 | 3 |
| VAST | 新闻评论 | 5,100 | 2,145 | 3,120/1,980 | 3 |
| EZSTANCE | 推文 | 9,313 | 6,873 | 9,313/149 | 3 |

### 表2: TSE 数据集上目标生成与立场检测表现（显式 vs 非显式）

| 模型 | 显式 BTSD↑ | 显式 SC↑ | 非显式 BTSD↑ | 非显式 SC↑ |
|------|-----------|---------|-------------|-----------|
| TSE-Mapped | 36.63 | 38.10 | 30.56 | 32.00 |
| TSE-BestGen | 35.80 | 37.81 | 29.32 | 31.00 |
| GPT-3.5 (TG&SD) | 39.60 | 47.61 | 31.32 | 33.94 |
| GPT-4o (TG&SD) | **41.92** | 46.83 | **36.12** | 37.50 |
| Gemini-pro (TG&SD) | 40.92 | 45.71 | 34.85 | 35.96 |
| Llama-3-70B (TG&SD) | 41.52 | **49.84** | 34.67 | **35.50** |
| Mistral-large (TG&SD) | 41.39 | 49.76 | 35.42 | 34.70 |

### 表3: VAST 数据集上目标生成与立场检测表现

| 模型 | 显式 BTSD↑ | 显式 SC↑ | 非显式 BTSD↑ | 非显式 SC↑ |
|------|-----------|---------|-------------|-----------|
| GPT-4o (TG&SD) | **44.25** | 49.38 | 39.84 | 43.84 |
| Gemini-pro (TG&SD) | 42.78 | **51.46** | **40.53** | **48.53** |
| Llama-3-70B (TG&SD) | 42.50 | 48.73 | 42.02 | 46.57 |
| Mistral-large (TG&SD) | 43.13 | 51.30 | 39.98 | 46.55 |

### 关键发现
- **LLM 全面优于 TSE**: 在目标生成和立场检测两个阶段上，所有测试的 LLM 在显式和非显式场景中均超过 TSE 的两种变体（TSE-Mapped 和 TSE-BestGen）
- **非显式场景是瓶颈**: 所有模型在非显式目标场景下表现显著下降（BTSD 下降约 5-10 个百分点），因为文本缺乏足够的隐式线索让模型推断出正确目标
- **联合提示优于分步提示**: TG&SD（单步联合）在大多数模型和数据集上优于 TG+SD（两步分离），说明同时建模目标-立场关系有助于两个子任务
- **闭源 vs 开源无绝对优势**: GPT-4o 在目标生成上整体最优，但 Llama-3-70B 和 Mistral-large 在立场检测上常常超越 GPT 系列
- **"反义目标"问题**: GPT-4o 有时生成语义高度相关但立场方向相反的目标（如 gold target "permit to carry gun" → 生成 "gun control"），导致立场反转

## 亮点与洞察
- **任务定义有意义**: OTSD 比 ZSSD 更贴近真实应用场景，去掉预定义目标列表的假设后问题难度和现实性大幅提升
- **BTSD 评估指标设计巧妙**: 用立场分类器的 F1 作为目标质量的代理指标，与人类判断相关性高达 0.74-0.85，解决了开放生成不便精确匹配评估的痛点
- **显式/非显式分析揭示盲区**: 区分显式和非显式目标后发现了 TSE 原始论文未报告的性能差距，非显式场景是未来需要重点突破的方向
- **实验覆盖广**: 8 个 LLM × 3 个数据集 × 2 种提示策略 × 2 种场景，实验矩阵设计系统全面

## 局限性
- 仅关注单目标场景，现实文本常包含多个讨论目标且目标间可能相互关联
- 仅在英语上实验，跨语言 OTSD 尚未探索
- 存在数据泄露风险——LLM 预训练数据可能包含测试集中的部分内容
- "反义目标"问题（语义相关但立场方向相反的生成目标）未在评估指标层面得到充分解决，BTSD 对此不够敏感
- 人类评估规模较小（每数据集 500 条样本，3 位标注者），对评估指标可靠性的验证强度有限

## 相关工作对比
- **vs TSE** (Li et al., 2023): TSE 是最接近的先驱工作，但其目标生成后需映射到预定义列表（如 "Religious diversity" → 从列表中找最近的 "Atheism"），本质上仍是半开放设置；OTSD 完全取消预定义列表，更具挑战性也更实用。实验表明 LLM 在 OTSD 设置下的目标生成和立场检测均优于 TSE。
- **vs ZSSD 方法** (Zhang et al., 2023; Allaway et al., 2021 等): 传统零样本立场检测假设推理时目标已知（仅训练时未见），而 OTSD 在推理时也不提供目标，相当于在 ZSSD 基础上增加了目标识别的前置任务。OTSD 可视为 ZSSD 的自然推广。
- **vs Cross-target SD** (Zhang et al., 2020): 跨目标立场检测的目标虽在训练时未见但通常属于相似领域，而 OTSD 中目标可来自完全不同的领域，泛化要求更高。

## 评分
- 新颖性: ⭐⭐⭐⭐ OTSD 任务定义有意义，BTSD 评估指标设计新颖；但核心方法仅为 prompt engineering
- 实验充分度: ⭐⭐⭐⭐ 8 个模型 × 3 个数据集 × 2 种策略，显式/非显式分析细致，人类评估验证指标可靠性
- 写作质量: ⭐⭐⭐⭐ 任务动机和定义清晰，TSE vs OTSD 的区别用实例说明直观易懂
- 价值: ⭐⭐⭐ 为立场检测领域提供了更现实的评估框架，但实际落地仍受非显式场景性能瓶颈制约

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](bilingual_zero-shot_stance_detection.md)
- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)

</div>

<!-- RELATED:END -->
