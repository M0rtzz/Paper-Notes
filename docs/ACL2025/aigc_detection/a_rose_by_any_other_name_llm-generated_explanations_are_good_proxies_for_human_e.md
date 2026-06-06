---
title: >-
  [论文解读] A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI
description: >-
  [ACL 2025][AIGC检测][人工判断分布] 提出用 LLM 生成的 NLI 解释替代昂贵的人工解释来近似人工判断分布（HJD），实验表明在提供人工标签引导的条件下，LLM 生成的解释与人工解释在 KL 散度、JSD 等指标上效果相当，并可推广到无人工解释的数据集（MNLI）和域外测试集（ANLI）。
tags:
  - "ACL 2025"
  - "AIGC检测"
  - "人工判断分布"
  - "NLI"
  - "LLM解释生成"
  - "标签变异"
  - "标注分歧"
---

# A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI

**会议**: ACL 2025  
**arXiv**: [2412.13942](https://arxiv.org/abs/2412.13942)  
**代码**: [https://github.com/mainlp/MJD-Estimator](https://github.com/mainlp/MJD-Estimator)  
**领域**: 标注分歧 / 人工判断分布  
**关键词**: 人工判断分布, NLI, LLM解释生成, 标签变异, 标注分歧

## 一句话总结

提出用 LLM 生成的 NLI 解释替代昂贵的人工解释来近似人工判断分布（HJD），实验表明在提供人工标签引导的条件下，LLM 生成的解释与人工解释在 KL 散度、JSD 等指标上效果相当，并可推广到无人工解释的数据集（MNLI）和域外测试集（ANLI）。

## 研究背景与动机

**领域现状**：NLI 中标注者之间的分歧是真实的语义歧义现象而非噪声——同一前提-假设对，不同标注者可能合理地给出蕴含/中性/矛盾的不同判断。人工判断分布（HJD）通过大量标注者（如 ChaosNLI 的 100 人/实例）捕捉这种分布信息，对 NLI 分类器训练有价值。

**现有痛点**：Chen et al. (2024) 证明 LLM 可通过少量人工标签+人工解释有效近似 HJD（称为 MJD Estimator），但收集人工解释的成本远高于仅收集标签——每条解释需要标注者详细阐述推理过程，且大多数 NLI 数据集根本没有解释标注。

**核心矛盾**：MJD Estimator 依赖人工解释才能有效工作，但人工解释是整个流程中最昂贵的瓶颈。

**本文目标** LLM 生成的解释能否在质量上足以替代人工解释来近似 HJD？

**切入角度**：让 LLM 为每个 NLI 标签穷举所有可能的解释理由，然后根据人工标签分布选择对应数量的解释，构建标签-解释对输入 MJD Estimator。

**核心 idea**：解释的来源（人工 vs LLM）不重要，重要的是解释的存在——"A rose by any other name would smell as sweet"。

## 方法详解

### 整体框架

分三步：(1) LLM 为每个前提-假设对的每个 NLI 标签生成多条解释；(2) 用无标签或标签引导策略选择对应的解释子集；(3) 将选中的标签-解释对输入 MJD Estimator（基于首 token 概率），输出模型判断分布（MJD）并与真实 HJD 对比。

### 关键设计

1. **模型解释生成**:

    - 功能：提示 LLM（Llama3 / GPT-4o）为给定前提-假设对的每个标签（蕴含/中性/矛盾）列出所有可能的解释理由
    - 核心思路：NLI 标签变异意味着同一标签可有多种合理理由（如多个标注者都选"蕴含"但原因不同），因此要求 LLM 穷举而非只给一条
    - 设计动机：穷举式生成提供了足够的候选池供后续选择策略使用

2. **解释选择策略**:

    - 功能：从候选解释池中选择合适的子集组成 MJD Estimator 的输入
    - 核心思路：
        - **无标签选择（Label-Free）**：每个标签均匀选 1 条解释，共 3 条，不使用任何人工标签——作为基线
        - **标签引导选择（Label-Guided）**：根据人工标签分布选择对应数量的解释。如某实例 5 个标注者中 3 人选蕴含、1 人中性、1 人矛盾，则选 3 条蕴含解释 + 1 条中性 + 1 条矛盾
    - 选择模式：首选模式（First，取 LLM 输出的前 k 条）vs 最长模式（Longest，取最长的 k 条）
    - 设计动机：标签信息是低成本的（大多数数据集有少量标签），用它引导解释选择可弥补无人工解释的缺失

3. **MJD 估计与评估**:

    - 功能：将选中的标签-解释对通过 MCQA 提示输入 LLM，用首 token 概率提取 MJD
    - 核心思路：对标签排列、解释排列和组合进行排列平均以消除位置偏见（A 偏好、长度偏见、序列偏见）
    - 评估指标：KL 散度、JSD、TVD 衡量分布距离；用 MJD 训练 BERT/RoBERTa 后在 ANLI 上评估下游性能

## 实验关键数据

### 主实验（Llama3，VariErr标签引导，vs 人工解释）

| 解释来源 | KL ↓ | JSD ↓ | TVD ↓ | D.Corr ↑ |
|---------|------|-------|-------|----------|
| 无解释（Llama3 only） | 0.259 | 0.262 | 0.284 | 0.689 |
| + 人工解释 | 0.238 | 0.250 | 0.269 | 0.771 |
| + 模型解释（无标签） | 0.295 | 0.278 | 0.310 | 0.744 |
| + 模型解释（VariErr引导） | 0.234 | 0.247 | 0.266 | 0.760 |
| + 模型解释（MNLI引导） | 0.242 | 0.251 | 0.275 | **0.849** |

### 下游任务（RoBERTa 在 ANLI 上的表现）

| 训练分布来源 | CE Loss ↓ | Weighted F1 ↑ |
|-------------|-----------|---------------|
| ChaosNLI HJD | 0.922 | 0.653 |
| Llama3 + 人工解释 MJD | 1.019 | 0.616 |
| Llama3 + 模型解释 (MNLI引导) MJD | 1.018 | **0.645** |

### 关键发现
- 标签引导的模型解释与人工解释在 HJD 近似质量上基本相当，甚至在 MNLI 引导下 D.Corr 超越人工解释（0.849 vs 0.771）
- 无标签选择作为基线效果较差但仍优于完全无解释
- 首选模式和最长模式表现相似（重叠率仅 18.9%），说明结果对选择模式稳健
- 用 MJD 训练的分类器在 ANLI 域外测试上优于单标签训练，证明 HLV 信息的下游价值

## 亮点与洞察
- **"名字不重要"的核心发现**——解释的来源（人工 vs LLM）不如解释的存在本身重要，这意味着大多数 NLI 数据集可以零额外人工成本获得 HJD 近似。该发现对所有涉及标注分歧的任务都有潜在推广价值。
- **成本大幅降低的实际路径**——从需要 100 名标注者/实例（ChaosNLI）降低到只需 4-5 个标签+LLM 自动生成解释，这使 HJD 方法在实际部署中可行。
- **域外泛化有效**——方法不仅在有解释的数据集上有效，还成功推广到无解释（MNLI）和域外（ANLI）场景，增强了方法的实用性。

## 局限与展望
- 仅在 NLI 任务上验证，其他标签变异显著的任务（如情感分析、毒性检测）未测试
- 仍需少量人工标签（4-5 个/实例）做引导选择，完全无标签模式效果较差
- 解释选择启发式（首选/最长）过于简单，可探索基于语义相似度的选择
- HJD 真值依赖 ChaosNLI（100 标注者），该真值本身的质量和代表性未被质疑
- 仅测试了 Llama3 和 GPT-4o 两个模型作为解释生成器

## 相关工作与启发
- **vs Chen et al. (2024)**: 需要人工解释做 MJD 估计；本文证明 LLM 生成的解释可替代，效果相当
- **vs Lee et al. / Madaan et al.**: 直接让 LLM 预测分布效果不一致；本文的解释引导方法更稳健
- **vs Pavlovic & Poesio (2024)**: 也尝试 LLM 近似 HJD 但效果参差不齐；本文通过解释机制获得更好的一致性

## 评分
- 新颖性: ⭐⭐⭐⭐ LLM 解释替代人工解释近似 HJD 是新颖且有影响力的发现
- 实验充分度: ⭐⭐⭐⭐ ChaosNLI+MNLI+VariErr+ANLI，多场景多模型验证，排列平均消除偏见
- 写作质量: ⭐⭐⭐⭐ 隐喻标题贴切，方法阐述清晰，实验组织合理
- 价值: ⭐⭐⭐⭐ 对标注成本优化和 HLV 建模有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment](../../ACL2026/aigc_detection/from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based.md)
- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ACL 2025\] Low-Perplexity LLM-Generated Sequences and Where To Find Them](low-perplexity_llm-generated_sequences_and_where_to_find_them.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)

</div>

<!-- RELATED:END -->
