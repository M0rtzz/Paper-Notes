---
title: >-
  [论文解读] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts
description: >-
  [ACL 2025][自主数据选择] 提出 AutoDS——用基座语言模型自身作为零样本生成分类器，通过 YES/NO token 的 logits 计算连续 LM-Score 来自动评估数学文本质量，筛选高质量语料做持续预训练，在 MATH/GSM8K/BBH 上实现约 2 倍 token 效率提升。
tags:
  - ACL 2025
  - 自主数据选择
  - 零样本生成分类器
  - 数学文本
  - 持续预训练
  - LM-Score
---

# AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts

**会议**: ACL 2025  
**arXiv**: [2402.07625](https://arxiv.org/abs/2402.07625)  
**代码**: [GitHub](https://github.com/yifanzhang-pro/AutoMathText) (有)  
**领域**: 数据筛选 / 数学推理  
**关键词**: 自主数据选择, 零样本生成分类器, 数学文本, 持续预训练, LM-Score

## 一句话总结

提出 AutoDS——用基座语言模型自身作为零样本生成分类器，通过 YES/NO token 的 logits 计算连续 LM-Score 来自动评估数学文本质量，筛选高质量语料做持续预训练，在 MATH/GSM8K/BBH 上实现约 2 倍 token 效率提升。

## 研究背景与动机

**数学推理是 LLM 的核心挑战之一，但高质量数学语料稀缺且质量参差。** 数学文本包含符号公式、多步推导和严格证明结构，与常规语言任务差异巨大。尽管社区对构建数学能力强的 LLM 热情高涨，但缺乏精心策划的数学语料始终是关键瓶颈。

**现有数据筛选方法存在明显不足。** Phi-1/Phi-2 方法用 GPT-4 标注代码片段的教育价值后训练随机森林分类器做数据过滤——这需要昂贵的 GPT-4 标注且只产生离散标签（"好"/"坏"），丢弃了质量的细粒度信息。在数学场景下，教育价值 0.95 和 0.001 的文本被同样处理显然不合理。关键词启发式方法（如计算 LaTeX 符号数量）更是无法捕获深层数学推理。

**核心 insight：基座 LLM 本身可以充当数据质量判官。** 受 DPO 中 Bradley-Terry 模型的启发，作者提出直接用基座模型的 YES/NO logits 计算连续评分，无需任何人工标注、SFT、RLHF 或额外分类器。这种"让模型自己选择学什么"的自主学习范式既高效又可扩展。

## 方法详解

### 整体框架

AutoDS 的核心流程：(1) 设计 meta-prompt 向基座模型提两个 YES/NO 诊断问题；(2) 从模型 logits 计算连续 LM-Score；(3) 按阈值筛选高分文档构建 AutoMathText 数据集；(4) 用筛选数据做持续预训练。整个过程零标注、零额外模型、全自动。

### 关键设计

1. **零样本生成分类器——基于 logits 的连续评分**:
    - 功能：用基座 LLM 的 YES/NO token logits 为每篇数学文本计算连续质量评分
    - 核心思路：设计包含两个诊断问题的 meta-prompt——Q1: "这段文本是否具备数学智能？" Q2: "它是否对未来的数学学习有用？"。从模型输出提取 logits 计算每个问题的评分：$\text{LM-Score}(Q) = \frac{\exp(\text{logit}(\text{YES}))}{\exp(\text{logit}(\text{YES})) + \exp(\text{logit}(\text{NO}))}$，最终评分为两问题分数的乘积：$\text{LM-Score}(Q_1, Q_2) = \text{LM-Score}(Q_1) \times \text{LM-Score}(Q_2)$，确保文本必须在数学智能和教育价值两个维度上都获得高分
    - 设计动机：连续评分比二分类保留了质量的细粒度——0.95 和 0.50 的文本可以被不同对待，提升 token 效率。评分公式本质上是 softmax 归一化，与 RLHF 中的 Bradley-Terry 奖励模型形式一致，但无需任何监督信号

2. **自主数据筛选管线**:
    - 功能：从三大数据源（OpenWebMath、arXiv、Algebraic Stack）筛选高质量数学文本
    - 核心思路：使用 Qwen-72B 基座模型计算 LM-Score，处理 1126 万文档（200+ GB）。按分数阈值（如 0.50-1.00 或 0.75-1.00）保留文档。高分文档主要来自 math.stackexchange.com 等数学密集站点
    - 设计动机：人工标注 1126 万文档成本超过 1000 万美元；AutoDS 使用 4 张 A100-80G 约 750 GPU 小时，按云计算定价不到 1 万美元——成本降低 1000 倍

3. **自主持续预训练**:
    - 功能：基座模型自身选择训练数据后做持续预训练
    - 核心思路：模型不仅从数据中学习，还主动决定"学什么数据"，实现自导向学习。随着新数据到来可持续评估和动态更新语料库
    - 设计动机：在数学等专业领域，人工标注稀缺且昂贵，关键词启发式不可靠（如 OpenWebMath 的分类器主要看 LaTeX 符号密度），让模型自主判断更准确且可扩展

### 损失函数 / 训练策略

持续预训练使用标准 causal LM 目标（next-token prediction），筛选策略通过数据质量间接影响训练效果。用 Qwen-72B 评分，在 Gemma-2B、LLaMA2-7B、Mistral-7B 上做持续预训练验证。

## 实验关键数据

### 主实验——跨模型持续预训练对比

| 模型 + 数据选择方法 | MATH (5-shot) | GSM8K (5-shot) | BBH (3-shot) |
|-------------------|---------------|----------------|--------------|
| Mistral-7B Base | 12.88 | 38.82 | 55.92 |
| + Uniform (OpenWebMath) | 14.26 | 44.12 | 56.50 |
| + DSIR | 12.30 | 42.00 | 55.97 |
| + QuRating | 12.90 | 36.32 | 55.63 |
| **+ AutoDS** | **16.14** | **45.41** | **58.61** |
| LLaMA2-7B Base | 2.94 | 12.51 | 39.89 |
| + Uniform (OpenWebMath) | 5.14 | 19.79 | 41.53 |
| **+ AutoDS** | **7.74** | **21.99** | **42.76** |
| Gemma-2B Base | 10.96 | 17.29 | 34.19 |
| **+ AutoDS** | **11.02** | **18.88** | **34.88** |

### 消融实验——AutoDS vs 二分类过滤

| 方法 | 数据量 (M tokens) | MATH CPT (%) | MATH SFT (%) |
|------|-------------------|-------------|-------------|
| 无预训练 | 0 | 12.88 | 27.20 |
| OpenWebMath (全量) | 328.9 | 10.50 | 26.98 |
| AutoDS (0.75-1.00) | 328.9 | **13.68** | **28.06** |

### 关键发现

- AutoDS 在所有模型（Gemma-2B、LLaMA2-7B、Mistral-7B）和所有基准（MATH、GSM8K、BBH）上一致超越 Uniform、DSIR、QuRating
- 相同 token 数量下，AutoDS 预训练效果约为 Uniform 的 2 倍——即"2× token 效率"
- DSIR 和 QuRating 在某些配置下甚至不如基座模型，说明不当的数据选择可能有害
- OpenWebMath 分类器主要依赖 LaTeX 符号密度，容易选入含大量数字但无数学内容的文本（如物流追踪页面）
- StackExchange 站点（尤其 math.stackexchange.com）在高分段贡献了大量高质量数学文本

## 亮点与洞察

- **"模型自己选数据"的自主学习范式**——无需任何外部标注、分类器或对齐步骤，真正实现自导向数据策划
- **连续评分 > 二分类**——保留质量细粒度信息，提升 token 效率约 2 倍
- **与 Bradley-Terry / DPO 的理论联系**：LM-Score 的 softmax 形式与 RLHF 奖励模型一致，暗示数据选择和偏好优化可能在理论上统一
- **成本效率极高**：用 4 张 A100 约 750 GPU 小时处理 1126 万文档，成本不到 1 万美元（vs 人工标注 1000 万+美元）

## 局限与展望

- 评分质量依赖基座模型能力——弱模型可能无法准确评估数学内容价值
- 仅验证了数学领域，其他专业领域（医学/法律/代码）的 meta-prompt 需重新设计
- YES/NO 二值假设可能不足以捕获更丰富的质量维度（如创新性、严谨性、难度等）
- 仅在英语数学文本上验证，多语言数学语料的评分有效性未知
- 评分模型（Qwen-72B）与目标模型（Mistral-7B 等）不同——是否存在最优的评分模型选择策略？

## 相关工作与启发

- **vs Phi-1/Phi-2 数据筛选**：需 GPT-4 标注 + 训练随机森林分类器，成本高且仅产出离散标签；AutoDS 零标注零分类器且产出连续评分
- **vs DSIR (Importance Resampling)**：基于 n-gram 统计特征做分布匹配；AutoDS 基于 LLM 语义理解，在数学场景下更准确
- **vs QuRating**：使用 LLM 评估数据"质量"但方式不同；AutoDS 的 logits 方案更简洁高效
- **启发**：连续质量评分比二分类更高效这一原则应适用于所有数据筛选场景——不仅限于数学

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 零样本生成分类器概念新颖，连续 LM-Score 设计优雅，自主数据选择范式有深远意义
- 实验充分度: ⭐⭐⭐⭐ 三模型×三基准，对比 DSIR/QuRating/Uniform，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法简洁，数据可视化（treemap、分布图）直观
- 价值: ⭐⭐⭐⭐⭐ AutoMathText 数据集已开源，方法可直接复用，2× 效率提升有实际意义

<!-- RELATED:START -->

## 相关论文

- [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)
- [ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)

<!-- RELATED:END -->
