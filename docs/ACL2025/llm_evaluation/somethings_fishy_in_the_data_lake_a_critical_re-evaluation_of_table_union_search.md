---
title: >-
  [论文解读] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks
description: >-
  [ACL 2025][LLM评测][表联合搜索] 系统性分析了主流表联合搜索 (Table Union Search, TUS) 基准测试的三大结构性缺陷——过度重叠、语义简单、真值噪声，揭示简单的词袋 (BoW) 和预训练嵌入基线就能在这些基准上达到或超越复杂 SOTA 方法的效果…
tags:
  - "ACL 2025"
  - "LLM评测"
  - "表联合搜索"
  - "benchmark"
  - "数据湖"
  - "表示学习"
  - "语义理解"
---

# Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks

**会议**: ACL 2025  
**arXiv**: [2505.21329](https://arxiv.org/abs/2505.21329)  
**代码**: [有](https://github.com/Allaa-boutaleb/fishy-tus)  
**领域**: NLP / 数据发现 / 基准测试评估  
**关键词**: 表联合搜索, benchmark, 数据湖, 表示学习, 语义理解

## 一句话总结

系统性分析了主流表联合搜索 (Table Union Search, TUS) 基准测试的三大结构性缺陷——过度重叠、语义简单、真值噪声，揭示简单的词袋 (BoW) 和预训练嵌入基线就能在这些基准上达到或超越复杂 SOTA 方法的效果，调研结论指出当前基准无法有效评估语义理解能力。

## 研究背景与动机

表联合搜索 (TUS) 旨在从数据湖中检索可与查询表进行纵向拼接（union）的候选表。近年来涌现了多种复杂的深度学习方法（如 Starmie、HEARTS、TabSketchFM），它们在现有基准上取得了高分，但这些高分是否真正反映了语义理解能力值得质疑。

核心问题：如果简单的非语义方法也能取得高分，那么基准测试本身可能存在问题——高性能可能归因于数据集特有的统计和结构属性，而非模型的语义理解能力。

三个假设性问题：
1. 基于分区的基准可能引入过度的 Schema 和值重叠
2. 从公开语料构建的基准可能语义过于简单
3. LLM 或人工标注的真值可能存在一致性问题

## 方法详解

### 整体框架

通过三步策略检验基准质量：
1. 分析基准构建过程，识别潜在结构性弱点
2. 使用简单基线方法作为诊断工具，测试基准对不同信息类型的敏感性
3. 对真值标签进行可靠性验证

### 关键设计

1. **基准构建分析**：识别三类结构性问题

    - **过度重叠**：分区式基准（TUS_Small、TUS_Large、SANTOS）通过切分种子表创建样本，导致查询表和真值候选表之间存在大量 Schema 和值重叠。测量显示 >90% 的真值对共享 ≥50% 的精确列名
    - **语义简单**：语料衍生基准（PYLON、LakeBench）虽避免了分区重叠，但聚焦于通用主题（如 GitTables 中的常见话题），这些内容很可能已包含在预训练语料中
    - **真值噪声**：LLM 生成的基准（UGEN）可能因 LLM 对 unionability 的解释不一致而产生噪声标签

2. **简单基线方法选择**

    - **词袋向量化器**（HashingVectorizer、TfidfVectorizer、CountVectorizer）：测试高重叠基准能否仅靠词频信号成功
    - **预训练句子嵌入**（SBERT all-mpnet-base-v2）：测试通用嵌入是否已足以应对语义简单的基准
    - 三种列序列化方式：值+列名 (V+C)、仅列名 (C)、仅值 (V)

3. **重叠度量化**

   使用 Szymkiewicz–Simpson 系数量化列名重叠度和值重叠度：

    $Overlap_c(Q,C) = \frac{|Cols_Q \cap Cols_C|}{\min(|Cols_Q|, |Cols_C|)}$

### 损失函数 / 训练策略

本文为分析性研究，简单基线不需要训练（词袋和预训练嵌入直接使用）。对比的 SOTA 方法（Starmie、HEARTS）按照原始设置重新训练。

## 实验关键数据

### 主实验（P@k / R@k 对比）

| 方法 | SANTOS P@10 | SANTOS R@10 | TUS P@60 | PYLON P@10 |
|------|------------|------------|---------|-----------|
| IDEAL | 1.00 | 0.75 | 1.00 | 1.00 |
| TFIDF | **0.99** | **0.74** | **1.00** | 0.70 |
| SBERT (V+C) | **0.99** | **0.74** | **0.98** | **0.94** |
| Starmie | 0.99 | 0.74 | 0.99 | 0.77 |
| HEARTS | 0.98 | 0.73 | 0.96 | 0.82 |

### 分区式基准上的表现

| 基准 | TFIDF P@k | Starmie P@k | 差距 |
|------|-----------|-------------|------|
| SANTOS | 0.99 | 0.99 | 无 |
| TUS_Small | 1.00 | 0.99 | TFIDF 更优 |
| TUS_Large | 0.99 | 0.99 | 持平 |

### LLM生成基准 (UGEN) 上的真值问题

| UGEN 版本 | SBERT P@10 | 发现 |
|-----------|-----------|------|
| V1 | 0.94 | SBERT 接近理想值 |
| V2 | 0.80 | 存在噪声但 SBERT 仍强 |

### 关键发现

1. **分区式基准上，TFIDF 已达 SOTA**：在 SANTOS、TUS 基准上，简单词袋方法与 Starmie 等深度学习方法不分伯仲甚至更优
2. **SBERT 在几乎所有基准上表现优异**：无需任何微调的预训练嵌入就能达到高性能，说明基准语义挑战性不足
3. **仅用列名 SBERT(C) 就能达到很好效果**：在高重叠基准上，甚至不需要看值内容
4. **UGEN 存在真值不一致问题**：LLM 在不同生成中对 unionability 的判断标准不一
5. **LakeBench 衍生基准继承了分区式基准的问题**：大量合成查询导致高重叠

## 亮点与洞察

1. **反直觉结论**：精心设计的深度学习方法在当前基准上并不比 TFIDF 好多少——这不是说方法不好，而是基准不够区分
2. **诊断而非否定**：论文定位为"用简单基线诊断基准"，而非"简单方法比复杂方法好"
3. **提出未来基准标准**：
    - 最小化 Schema 和值重叠
    - 包含需要深度推理的语义挑战
    - 确保真值标签的高质量和一致性
4. **计算效率证据**：简单方法在效率上远优于 SOTA，但这里效率不是重点

## 局限与展望

1. 仅分析了已有基准的缺陷，未提出具体的新基准
2. 部分 LakeBench 真实查询与合成查询无法区分，影响了分析精度
3. TabSketchFM 因预训练权重不可用而使用论文原始数据
4. 未讨论如何在保持可用性的前提下构建更具挑战性的基准
5. 简单基线的局限——SBERT 在语义高度复杂的真实企业数据上可能表现差

## 相关工作与启发

- Starmie (Fan et al., 2023)：基于对比学习的列嵌入方法，TUS 领域代表性工作
- HEARTS (Boutaleb et al., 2025)：超图表示学习方法
- TabSketchFM (Khatiwada et al., 2025)：数据草图方法，在可扩展性方面有优势
- SANTOS (Khatiwada et al., 2023)：引入列间关系用于 TUS，但基准本身存在分区重叠

## 评分

- **新颖性**: ⭐⭐⭐⭐ 用简单基线诊断基准的方法论视角独特，发现了领域内普遍忽视的问题
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 9 个基准、多种基线和 SOTA 方法、重叠度量化分析
- **写作质量**: ⭐⭐⭐⭐ 论述逻辑严密，从假设到验证的结构清晰
- **价值**: ⭐⭐⭐⭐⭐ 对 TUS 领域的基准建设具有警示和指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](../../ICCV2025/llm_evaluation/rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ACL 2025\] Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation](beyond_one-size-fits-all_tailored_benchmarks_for_efficient_evaluation.md)

</div>

<!-- RELATED:END -->
