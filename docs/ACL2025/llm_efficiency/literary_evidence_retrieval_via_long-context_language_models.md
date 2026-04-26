---
title: >-
  [论文解读] Literary Evidence Retrieval via Long-Context Language Models
description: >-
  [ACL 2025][LLM效率][长上下文理解] 将 RELiC 数据集改造为长上下文文学证据检索 benchmark（292 个高质量样本），要求模型在完整小说文本（45k-125k tokens）中为文学评论找到缺失引用；Gemini Pro 2.5 以 62.5% 准确率首次超越人类专家（55%），但最佳开源模型 DeepSeek-R1 仅 29.1%，揭示了闭源/开源模型在解释性推理上的巨大鸿沟。
tags:
  - ACL 2025
  - LLM效率
  - 长上下文理解
  - 文学证据检索
  - benchmark
  - 推理模型评测
  - 开源vs闭源
  - 文学分析
---

# Literary Evidence Retrieval via Long-Context Language Models

**会议**: ACL 2025  
**arXiv**: [2506.03090](https://arxiv.org/abs/2506.03090)  
**代码**: [katherinethai/long_context_relic](https://github.com/katherinethai/long_context_relic)  
**领域**: LLM效率  
**关键词**: 长上下文理解, 文学证据检索, benchmark, 推理模型评测, 开源vs闭源, 文学分析

## 一句话总结

将 RELiC 数据集改造为长上下文文学证据检索 benchmark（292 个高质量样本），要求模型在完整小说文本（45k-125k tokens）中为文学评论找到缺失引用；Gemini Pro 2.5 以 62.5% 准确率首次超越人类专家（55%），但最佳开源模型 DeepSeek-R1 仅 29.1%，揭示了闭源/开源模型在解释性推理上的巨大鸿沟。

## 研究背景与动机

**长上下文评测的不足**：现有长上下文 benchmark（如 Needle-in-Haystack）主要测试简单信息检索，无法衡量模型对长文本的深度理解和推理能力。

**文学分析的独特需求**：文学证据检索要求模型同时具备对叙事的全局理解（global narrative reasoning）和对具体段落的精细阅读能力（close textual examination），是检验长上下文真正理解能力的理想测试场景。

**RELiC 数据集的局限**：原始 RELiC 数据集（Thai et al., 2022）包含 78k 条文学分析摘录，但数据较嘈杂，存在 OCR 错误、引用泄露、数据污染等问题，不适合直接用于 LLM 评测。

**核心动机**：在长上下文 LLM 能处理百万 token 的今天，探索它们对文学小说的理解到底达到了什么水平。

## 方法详解

### 整体框架

任务定义：给定一部完整小说的全文和一段文学评论（其中某个引用被遮盖），模型需要从小说原文中生成/检索被遮盖的引用段落。这模拟了文学学者在分析中选择支撑性引用的过程。

### 数据集构建流程

从 RELiC 的 78k 条样本出发，经过多步过滤和人工审核，最终得到 292 个高质量样本：

1. **CLEAN（GPT-4o-mini）**：清理 OCR 错误、移除暴露引用页码的行内引注
2. **LEAKAGE（启发式）**：使用模糊匹配（阈值 95）排除上下文与原文重复的样本，防止模型利用词汇重叠作弊
3. **LIT ANALYSIS（GPT-4o-mini）**：分类判断是否为文学分析，排除被错误标注的样本
4. **LOCATION（GPT-4o-mini）**：检测上下文是否泄露了引用的位置信息
5. **FIRST/LAST SENT（启发式）**：标记首句/末句（可能是名句，容易被记忆）
6. **OUTLIER（启发式）**：标记被过度引用的段落
7. **EZ2MEM（GPT-4）**：不给原文直接让模型答题，排除仅凭记忆就能回答的样本
8. **人工审核**：英语文学学位持有者审查 400 条，最终标注 292 条为高质量样本

过滤方案的 F1-score 为 89.8（100 条手工验证，57 TP / 30 TN / 6 FP / 7 FN）。

### 数据集特点

| 统计量 | 原文（7部小说） | 样本（292条） |
|--------|-----------------|---------------|
| 平均 token 数 | 85,526 | 254.9 |
| 最大 token 数 | 124,544 | 492.0 |
| 最小 token 数 | 45,038 | 116.0 |
| 平均每书样本数 | — | 36.0 |

涵盖 7 部经典英语小说：《了不起的盖茨比》《弗兰肯斯坦》《红字》《勇敢新世界》《梅西所知》《伊桑·弗罗姆》《觉醒》。

### 评测子集设计

- **Human Eval Set（40 条）**：由英语文学专家亲自作答并给出推理过程的子集
- **Close Reading Set（39 条）**：标注为"细读"类型的样本——评论中反复引用了目标段落的部分词句，存在词汇重叠

### 提示策略

- **Simple Prompt**：直接要求模型输出引用
- **Explanation Prompt**：先要求模型给出选择理由，再输出引用（借鉴自 Nocha benchmark）

### Embedding 基线

使用 MTEB 排行榜第一的 gte-Qwen2-7B-instruct 作为 embedding 检索基线，计算 recall@1。

## 实验结果

### 主实验结果

| 模型 | Prompt | ALL (n=292) | Human Eval (n=40) | Close Reading (n=39) |
|------|--------|-------------|---------------------|----------------------|
| Gemini Pro 2.5 | Explanation | **64.7%** | **62.5%** | **79.5%** |
| GPT-4.1 | Explanation | 51.0% | 47.5% | 69.2% |
| o3 | Explanation | 50.7% | 50.0% | 66.7% |
| Gemini Pro 1.5 | Explanation | 38.5% | 40.0% | 50.0% |
| Claude Sonnet 3.7 | Explanation | 37.0% | 32.5% | 48.7% |
| DeepSeek-R1（开源最佳） | Explanation | 29.1% | 15.0% | 38.5% |
| GPT-4o | Explanation | 24.3% | 22.5% | 31.8% |
| Qwen 3 (32B) | Explanation | 19.2% | 20.0% | 33.3% |
| Qwen 3 (8B) | Explanation | 8.9% | 5.0% | 10.3% |
| o3-mini | Explanation | 8.3% | 10.0% | 13.6% |
| gte-Qwen2-7B（embedding基线） | — | 4.5% | 2.5% | 6.8% |
| **人类专家** | — | — | **55.0%** | — |

**关键发现**：Gemini Pro 2.5 在全部指标上超越人类专家，且每次调用平均仅需 45 秒（人类平均 12 分钟）。

### 过度生成分析

| 模型 | 准确率 | 平均长度比 |
|------|--------|------------|
| 人类专家 | 55.0% | 2.1 |
| Gemini Pro 2.5 | 62.5% | 3.0 |
| o3 | 50.0% | 2.7 |
| GPT-4.1 | 47.5% | 4.8 |
| Claude Sonnet 3.7 | 32.5% | 4.0 |
| DeepSeek-R1 | 15.0% | 3.6 |
| Llama 3.1 (8B) | 5.0% | 5.9 |
| Llama 3.3 (70B) | 0.0% | 5.7 |

长度比 = 模型输出长度 / 真实引用长度（字符数），越接近 1.0 越好。所有模型都超过人类的 2.1，小模型（Llama 系列 >5.7）过度生成更严重——弱模型倾向于用更长的输出来弥补不确定性。

### 关键实验发现

1. **LLM 远超 embedding 基线**：gte-Qwen2-7B 仅 4.5%，比三年前 RELiC 论文的最佳 recall@1 仅高 1.6%，说明此任务需要全文推理，而非简单语义匹配
2. **闭源远超开源**：最佳开源 DeepSeek-R1（29.1%）不到最佳闭源 Gemini Pro 2.5（64.7%）的一半
3. **小模型无法利用细读线索**：>8B 模型在 Close Reading 子集上普遍提升 10-20%+，但 7B/8B 模型几乎无提升甚至下降
4. **代际进步显著**：Gemini Pro 2.5 vs 1.5（64.7% vs 38.5%），GPT-4.1 vs GPT-4o（51.0% vs 24.3%），o3 vs o1（50.7% vs 32.2%）
5. **Explanation prompt 效果不一**：对 Gemini Pro 1.5 有帮助，对 GPT-4o 反而下降；推理模型（如 o3、Gemini Pro 2.5）自带内部推理 token，额外的解释提示影响不大

## 案例分析

### 案例一：模型失败 / 人类成功

在《红字》（>80k 词）中，评论暗指对 Roger Chillingworth 认出 Hester Prynne 的场景描写。人类专家考虑了两个候选段落，通过判断哪个更好地展现评论提到的"melodrama"（夸张的人物描写、暗示动作的词汇、情感张力）最终选对。三个顶级 LLM 都选了人类专家排除的那个候选段。

### 案例二：模型与人类同选"错误"答案

在《弗兰肯斯坦》中，人类和 Gemini Pro 2.5 选了同一个"错误"引用——两者都将评论解读为对陪审团反应的引入，但真实引用实际上支持的是 Frankenstein 自身的心理状态。说明：(1) 文学证据检索本质上具有多义性；(2) 模型能找到合理的替代证据，对文学学者有辅助价值。

## 亮点与洞察

- **高质量的真实长上下文 benchmark**：比 Needle-in-Haystack 等合成任务更有说服力，需要真正的叙事理解+解释性推理
- **首次在文学任务上 LLM 超越人类专家**：Gemini Pro 2.5 以 62.5% vs 55.0% 胜出，且速度提升 16 倍
- **开源/闭源鸿沟的定量证据**：不是长上下文能力不够，而是解释性推理能力不足
- **严谨的数据清理流程**：8 步过滤 + 人工审核，对 benchmark 构建有方法论参考价值
- **过度生成现象的系统性量化**：所有模型都倾向输出过长，弱模型尤为严重

## 局限性

1. **数据规模有限**：仅 292 条测试样本，7 部小说，统计效力有限
2. **仅限英语/西方文学**：主要来自公版经典，不代表全球文学传统
3. **单人标注**：人类基线由单一标注者（论文作者之一）完成，存在个人偏差
4. **任务与真实场景有差距**：真实文学分析中学者是逐步形成论点并选择引用，而非给定评论去填空
5. **评估依赖模糊匹配**：自动评估使用 partial ratio fuzzy matching，可能遗漏表述略有不同但语义正确的答案

## 相关工作

- **长上下文 Benchmark**：Nocha (Karpinska et al., 2024)、FABLES (Kim et al., 2024)
- **原始数据集**：RELiC (Thai et al., 2022) — 78k 文学分析摘录及引用
- **计算文学分析**：BookWorm（角色分析）、STORYSUMM（故事摘要忠实度）、Reading Subtext（短篇理解）
- **叙事理解**：HEART-felt（叙事元素提取）、Agents' Room（叙事生成）

## 评分

- 新颖性: ⭐⭐⭐⭐ 将文学分析引入长上下文评测，角度独特
- 实验充分度: ⭐⭐⭐⭐ 15+ 模型对比 + 人类基线 + 过度生成分析 + 案例研究
- 写作质量: ⭐⭐⭐⭐ 结构清晰，案例分析生动
- 价值: ⭐⭐⭐⭐ 对长上下文理解和解释性推理研究有重要参考意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](../../ICML2025/llm_efficiency/efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)

<!-- RELATED:END -->
