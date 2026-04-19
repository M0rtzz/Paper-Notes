---
title: >-
  [论文解读] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge
description: >-
  [ACL 2025][模型压缩][幻觉纠正] 系统性地探究了两种自纠正方法（CoVE 和 RARR）在新闻摘要幻觉纠正中的表现，比较了三种搜索引擎、多种检索设置和提示策略，发现 Bing 搜索片段 + RARR（few-shot）组合效果最佳，且 G-Eval 与人类评估高度一致。
tags:
  - ACL 2025
  - 模型压缩
  - 幻觉纠正
  - 自纠正
  - 新闻摘要
  - 搜索引擎
  - LLM后处理
---

# Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge

**会议**: ACL 2025  
**arXiv**: [2506.19607](https://arxiv.org/abs/2506.19607)  
**代码**: [GitHub](https://github.com/jvladika/HalluCorrect) (有)  
**领域**: 模型压缩  
**关键词**: 幻觉纠正, 自纠正, 新闻摘要, 搜索引擎, LLM后处理  

## 一句话总结

系统性地探究了两种自纠正方法（CoVE 和 RARR）在新闻摘要幻觉纠正中的表现，比较了三种搜索引擎、多种检索设置和提示策略，发现 Bing 搜索片段 + RARR（few-shot）组合效果最佳，且 G-Eval 与人类评估高度一致。

## 研究背景与动机

### 1. 领域现状

大语言模型虽然能生成流畅文本，但普遍存在幻觉问题——生成事实不正确或误导性的内容。后处理纠正（post-hoc correction）方法，尤其是多步自纠正方法，通过生成验证问题、检索证据、迭代修正来纠正初始响应中的错误。

### 2. 现有痛点

- 自纠正方法主要在百科生成（传记、列表等）任务上验证，**新闻摘要领域应用不足**
- 证据检索是关键环节，但关于**搜索引擎选择、片段 vs 全文、检索策略**等问题缺乏系统性对比
- **忠实性与纠正力度之间的权衡**被忽视——过度纠正可能改变原文风格和信息

### 3. 核心矛盾

新闻摘要既要准确纠正事实错误，又要保持忠实于原文风格，这两个目标之间存在张力。自纠正系统的各个环节（问题生成、证据检索、答案生成、文本重写）都依赖 LLM，每个环节都可能引入新的错误。

### 4. 本文目标

自纠正方法在新闻摘要幻觉纠正中效果如何？不同搜索引擎、检索设置和提示策略对纠正质量有何影响？

### 5. 切入角度

选取两个代表性自纠正系统（CoVE, RARR），增强其外部搜索能力，在新闻摘要数据集上进行全面比较实验。

### 6. 核心 idea

对 CoVE 和 RARR 两种自纠正系统进行系统性的实证研究，揭示搜索引擎选择、检索粒度和提示策略对新闻摘要幻觉纠正的影响。

## 方法详解

### 整体框架

两种系统共享相同的四步工作流：

1. **获取初始响应** $b$（含幻觉的摘要）
2. **生成验证问题** $q_1, ..., q_k$（用提示 $M_q$ 引导 LLM 生成）
3. **回答问题**：从证据源 $s$ 检索证据 $e$，通过 $M_a(q, e)$ 生成答案 $a_1, ..., a_k$
4. **重写响应**：将初始响应和答案输入 $M_r(b, a)$，输出修正后的响应 $r$

### 关键设计

#### 两种自纠正系统的差异

| 特性 | CoVE | RARR |
|------|------|------|
| 提示方式 | Zero-shot | Few-shot（6个示例） |
| 纠正风格 | 大胆修改，添加额外信息 | 保守忠实，接近原文 |
| 输出长度 | 通常较长 | 与原文接近 |

#### 证据检索设置

- **内部知识**：仅依赖 LLM 自身知识回答验证问题
- **搜索引擎**：Google、Bing、DuckDuckGo
- **检索粒度**：
    - **片段模式（Snippets）**：使用搜索引擎返回的 top-5 摘要片段
    - **全文模式（Full-article）**：解析文章 HTML，分块嵌入，用 SimCSE 余弦相似度选 top-5 段落
- **Gold 基线**：使用原始新闻文章作为证据上限

### 损失函数/训练策略

无训练——完全基于 LLM 推理的后处理方法。基础模型为 GPT-4o-mini，另外测试了 LLaMA 3.1 70B 和 Mixtral 8x7B。

## 实验关键数据

### 主实验

| 系统 | 证据来源 | NED↓ | 语义相似度↑ | NLI-矛盾↓ | G-Eval Overall↑ | G-Eval Factual↑ |
|------|----------|------|------------|----------|-----------------|-----------------|
| CoVE | GPT 内部 | 0.51 | 81 | 42 | 50 | 45 |
| RARR | GPT 内部 | 0.10 | 94 | 40 | 65 | 62 |
| CoVE | Google 片段 | 0.51 | 84 | 34 | 56 | 50 |
| RARR | **Bing 片段** | **0.14** | **95** | **35** | **69** | **60** |
| RARR | Google 片段 | 0.24 | 93 | 32 | 67 | 56 |
| RARR | Bing 全文 | 0.32 | 92 | 32 | 63 | 50 |
| CoVE | Gold 文章 | 0.49 | 88 | 18 | 70 | 63 |
| RARR | Gold 文章 | 0.21 | 94 | 19 | **75** | **67** |

### 人类评估

| 系统 | 人类评分 | G-Eval 评分 | 相关系数 |
|------|---------|------------|----------|
| RARR | 0.68 | 0.65 | Pearson r = 0.87 (p<1%) |
| CoVE | 0.54 | 0.52 | — |

### 关键发现

1. **外部搜索 > 内部知识**：无论 CoVE 还是 RARR，使用搜索引擎都比仅靠 LLM 内部知识效果好
2. **Bing 片段表现最佳**：RARR + Bing snippets 在 6 个指标中达到最优
3. **片段 > 全文**：搜索片段比全文检索更精准，全文模式引入大量无关信息（NLI-neutral 偏高）
4. **Few-shot (RARR) > Zero-shot (CoVE)**：few-shot 示例帮助保持忠实性，zero-shot 倾向大幅改写
5. **G-Eval 与人类评估高度对齐**：Pearson 相关 0.87，平均差异仅 3%
6. **DuckDuckGo 是免费可行替代**：虽然性能略低，但 API 完全免费
7. **Gold 文章是上限**：使用原始新闻文章作为证据时 G-Eval 最高（75），说明精确证据是关键

## 亮点与洞察

- **全面的实证研究**：系统地比较了 2 个系统 × 3 个搜索引擎 × 2 种检索模式 × 3 个 LLM，实验设计严谨
- **片段优于全文的反直觉发现**：进一步的检索不一定更好，简洁的搜索片段反而更切题
- **G-Eval 的高可靠性**：0.87 的相关系数为未来无法进行大规模人类评估的研究提供了有力的替代方案
- **Zero-shot vs Few-shot 的深层差异**：不仅是性能差异，更是纠正策略的根本不同——大胆修改 vs 忠实修正

## 局限与展望

1. 所有模块都依赖 LLM 调用，计算成本高、速度慢
2. 验证问题生成质量不稳定，可能遗漏关键错误或生成无关问题
3. 仅在新闻领域验证，其他领域（医学、法律）的迁移性未知
4. 未探索结构化查询或可信域名过滤等检索优化策略
5. CoVE 的零样本设置导致输出风格不可控，实际应用中可能需要更多约束

## 相关工作与启发

- **CoVE (Dhuliawala et al., 2024)**：链式验证方法，零样本纠正
- **RARR (Gao et al., 2023)**：研究与修订方法，few-shot 纠正
- **CRITIC**：使用工具交互的自纠正
- **SummEdits**：论文使用的幻觉摘要基准数据集
- **启发**：检索增强的自纠正是一个有潜力的方向，但精确的证据检索比复杂的纠正策略更重要

## 评分

- **新颖性**: ⭐⭐⭐ — 方法本身无创新（使用现有系统），但实证分析有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ — 极其全面的对比实验，多维度评估，包含人类评估
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，案例分析直观
- **综合价值**: ⭐⭐⭐⭐ — 对选择自纠正方法和检索策略提供了实用指导，实证发现可直接指导系统设计

<!-- RELATED:START -->

## 相关论文

- [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/model_compression/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)
- [Entropy-based Exploration Conduction for Multi-step Reasoning](entropy-based_exploration_conduction_for_multi-step_reasoning.md)

<!-- RELATED:END -->
