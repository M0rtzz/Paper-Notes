---
title: >-
  [论文解读] DRS: Deep Question Reformulation With Structured Output
description: >-
  [ACL 2025][问题重构] 提出 DRS（Deep Question Reformulation with Structured Output），一种零样本方法，通过实体驱动的 DFS 搜索 + 结构化输出约束，将 GPT-3.5 的问题重构准确率从 23.03% 提升至 70.42%，使 LLM 能有效帮助用户将不可回答的问题转化为可回答的形式。
tags:
  - ACL 2025
  - 问题重构
  - 不可回答问题
  - DFS搜索
  - 结构化输出
  - 零样本方法
---

# DRS: Deep Question Reformulation With Structured Output

**会议**: ACL 2025  
**arXiv**: [2411.17993](https://arxiv.org/abs/2411.17993)  
**代码**: [有](https://github.com/Lizhecheng02/DRS)  
**领域**: NLP / 问答系统  
**关键词**: 问题重构, 不可回答问题, DFS搜索, 结构化输出, 零样本方法

## 一句话总结

提出 DRS（Deep Question Reformulation with Structured Output），一种零样本方法，通过实体驱动的 DFS 搜索 + 结构化输出约束，将 GPT-3.5 的问题重构准确率从 23.03% 提升至 70.42%，使 LLM 能有效帮助用户将不可回答的问题转化为可回答的形式。

## 研究背景与动机

当用户面对不熟悉领域的文档时，经常会提出该文档无法回答的问题。例如文本描述了芥末的营养成分，用户却问"芥末有多少卡路里？"——实际文本只包含碳水化合物/脂肪/蛋白质等信息。

现有工作三个方向：

**检测方法**：识别不可答问题（SQuAD 2.0 等）

**澄清方法**：向用户追问信息

**重构方法**：将不可答问题改写为可答形式

**关键挑战**是同时满足两个标准：
- 重构后的问题必须**能从给定文本中回答**（answerability）
- 必须**保留原始问题的核心实体和意图**（intent preservation）

即使 GPT-3.5 在 zero-shot CoT 下也仅达 23.03% 准确率，暴露了 LLM 在平衡这两个目标上的不足。现有方法（few-shot、CoT）要么过度关注保留原问题实体导致仍不可答，要么追求可答性但偏离用户意图。

## 方法详解

### 整体框架

DRS 分三步：(1) 实体提取与过滤 → (2) DFS 组合搜索 + 结构化问题生成 → (3) 候选问题重评估。

### 关键设计

1. **实体提取与过滤（Entity Extraction and Filtering）**：

    - 第一步：用零样本提示让 LLM 提取问题中所有重要实体（尽量不遗漏）
    - 第二步：让 LLM 将实体分类为五类：主语(subject)、宾语(object)、谓语(predicate)、属性(attribute)、其他(others)
    - 仅保留主语、宾语、属性三类（核心意图承载者），丢弃谓语等
    - 设计动机：LLM 直接提取实体时常混入动词短语（如"step down"），导致后续生成错误率升高

2. **DFS 组合搜索 + 结构化生成**：

    - 用 DFS 算法系统性探索实体组合（组合中实体数 > 过滤后总数的一半时才进入生成）
    - 对每个合格组合，依次执行：
        - 让 LLM 基于文档和选定实体生成一个**结构化陈述**（statement）
        - 基于该陈述生成包含所有选定实体的**结构化问题**
        - 验证问题是否包含所有必要实体
        - 验证问题是否可从文档回答
    - 若满足条件则存储候选，达到阈值后停止搜索
    - **关键洞察**：先生成陈述再生成问题，比直接让 LLM 重构问题成功率高得多
    - 控制搜索深度和迭代次数实现效率与准确率的平衡

3. **候选问题重评估（Candidate Re-evaluation）**：

    - 对所有候选问题评估可答性（LLM 验证）
    - 计算实体重叠分数：$\text{Score} = \frac{|\text{Entities}_{cand} \cap \text{Entities}_{orig}|}{|\text{Entities}_{orig}|}$
    - 在可答的候选中选择实体重叠最高的作为最终输出

### 评估框架改进

- 发现之前的 Llama2-7B 评估器泛化性极差（平均准确率仅 52.78%，接近随机）
- 提出用 GPT-4o-mini 作为评估器，平均准确率 90.70%，且跨数据集一致性好
- 评估条件：(1) 重构问题可从文本回答 + (2) 实体重叠 ≥ 50%，两个条件同时满足才算成功

## 实验关键数据

### 主实验 — 四模型六数据集（表格）

| 模型 | 方法 | QA2 | BanditQA | BBC | Reddit | Yelp | SQuADv2 | 平均 |
|------|------|-----|---------|-----|--------|------|---------|------|
| GPT-3.5 | Zero-Shot CoT | 29.15 | 15.63 | 13.56 | 20.35 | 15.69 | 43.79 | 23.03 |
| GPT-3.5 | Few-Shot CoT | 44.94 | 48.78 | 16.95 | 18.58 | 25.49 | 41.22 | 32.66 |
| GPT-3.5 | **DRS (ours)** | **81.80** | **73.20** | **62.71** | **75.22** | **66.67** | **62.90** | **70.42** |
| GPT-4o-mini | Zero-Shot CoT | 49.39 | 37.50 | 42.37 | 17.70 | 35.29 | 50.10 | 38.73 |
| GPT-4o-mini | **DRS (ours)** | **88.26** | **80.16** | **79.66** | **83.19** | **78.43** | **78.30** | **81.33** |
| Gemma2-9B | Zero-Shot | 36.03 | 21.33 | 20.34 | 18.58 | 19.61 | 42.21 | 26.35 |
| Gemma2-9B | **DRS (ours)** | **59.92** | **60.73** | **55.93** | **59.29** | **49.02** | **55.62** | **56.75** |

DRS 在所有模型和数据集上显著优于所有基线，GPT-3.5 上实现约 3 倍提升。

### 参数敏感性 — 候选问题数量（总结表格）

| 候选数 | GPT-3.5 | GPT-4o-mini | Gemma2-9B |
|--------|---------|-------------|-----------|
| 1 | 63.57 | 78.56 | ~52 |
| 2 | ~69 | ~81 | ~56 |
| 3 | **70.42** | **81.33** | **56.75** |
| 4 | ~69 | ~80 | ~53 |
| 5 | ~68 | ~80 | ~55 |

2-3 个候选即可达到最优，即使仅 1 个候选也远超所有基线。

### 关键发现

- DRS 在零样本下超越 few-shot CoT，GPT-3.5: 70.42% vs 32.66%（翻倍以上）
- 温度变化对 DRS 影响极小（不同温度下差异 ≤ 3 个百分点），方法鲁棒性强
- 人工评估确认生成的重构问题几乎 100% 有意义且与文档相关（6 个数据集中仅 1 条无意义）
- 推理时间增加有限：GPT-3.5 上 DRS(2 候选) 耗时 10.07s vs CoT 6.80s，但准确率翻倍
- 额外实验：GPT-4 上 DRS 平均 70.28%，Llama3.1-70B 上 68.48%，验证泛化性

## 亮点与洞察

- **结构化输出约束**是关键创新：先陈述后问题的两步生成确保输出质量
- DFS 搜索将组合爆炸问题转化为可控的搜索空间，剪枝策略有效
- 实体分类（五类过滤）解决了 LLM 实体提取中动词短语混入的实际问题
- 评估器升级（Llama2-7B → GPT-4o-mini）本身就是重要贡献

## 局限与展望

- DFS 搜索需要多次文档遍历，计算成本高于单pass方法
- 六个数据集未覆盖特定学科领域文档
- 实体分类依赖 LLM 的准确性，可能在复杂领域出错
- 未来可探索让 LLM 一次完成重构的高效方法

## 相关工作与启发

- 基于 SQuAD 2.0、CouldAsk 等不可答问题数据集的系列研究
- 与 Chain-of-Thought 形成互补：CoT 提升推理但不约束输出结构，DRS 通过结构化约束提升生成质量
- 实体驱动的方法思路可推广到其他需要保持意图一致性的文本改写任务

## 评分

- **新颖性**: 7/10 — DFS+结构化输出的组合有效但非突破性
- **实验充分度**: 9/10 — 4+3 模型、6 数据集、参数敏感性、人工评估、推理时间分析
- **写作质量**: 7/10 — Case study 清晰，但方法描述偏冗长
- **价值**: 7/10 — 问题重构场景实用，提升幅度显著（3 倍），但应用面相对窄

<!-- RELATED:START -->

## 相关论文

- [Graph-Structured Trajectory Extraction from Travelogues](graph-structured_trajectory_extraction_from_travelogues.md)
- [Follow-up Question Generation for Enhanced Patient-Provider Conversations](follow-up_question_generation_for_enhanced_patient-provider_conversations.md)
- [Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)
- [Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

<!-- RELATED:END -->
