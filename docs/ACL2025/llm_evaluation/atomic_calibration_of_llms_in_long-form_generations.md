---
title: >-
  [论文解读] Atomic Calibration of LLMs in Long-Form Generations
description: >-
  [LLM评测] 系统研究长文本生成中的原子级校准(atomic calibration)，将置信度获取方法分为判别式和生成式两类，发现两者互补且提出基于置信度一致性的融合策略，揭示了模型在生成过程中置信度变化的有趣模式。
tags:
  - "LLM评测"
---

# Atomic Calibration of LLMs in Long-Form Generations

- **会议**: ACL 2025
- **arXiv**: [2410.13246](https://arxiv.org/abs/2410.13246)
- **代码**: 未提供
- **领域**: LLM Evaluation / 不确定性估计 / 事实性校准
- **关键词**: Atomic Calibration, Confidence Elicitation, Long-Form Generation, Hallucination, Confidence Fusion

## 一句话总结

系统研究长文本生成中的原子级校准(atomic calibration)，将置信度获取方法分为判别式和生成式两类，发现两者互补且提出基于置信度一致性的融合策略，揭示了模型在生成过程中置信度变化的有趣模式。

## 研究背景与动机

- **问题**: LLM 的置信度校准(calibration)对检测幻觉至关重要，但现有研究主要关注短文本 QA 任务的**响应级校准(macro calibration)**——对整个回答给一个置信度分数。在长文本生成中，一个回答可能同时包含准确和不准确的声明，单一分数无法反映细粒度的事实性。
- **关键问题**: (1) 为什么需要在原子声明(atomic claim)级别评估校准？(2) 什么因素影响原子级校准？(3) 原子级分析能揭示哪些宏观分析看不到的模式？
- **核心定义**: **原子级校准** = 将长回答分解为原子声明(每个包含单一事实)，为每个声明分配置信度，评估置信度与实际事实性的对齐程度。

## 方法详解

### 整体框架

1. 给定查询 q，LLM 生成长回答 x
2. 使用 GPT-4o 将 x 分解为 N 个原子声明 {c_1, ..., c_N}
3. 使用 GPT-4o + Wikipedia/Google Search 验证每个声明的事实性标签 y_i ∈ {0, 1}
4. 使用不同置信度获取方法为每个声明估计置信度 f(c_i)
5. 使用 ECE、Brier Score、AUROC 评估原子级校准质量

### 关键设计

1. **判别式(Discriminative)置信度方法** — 让模型自我评估：
    - **Dis-Single**: 直接问模型单个声明是否为真，取 P(True) 作为置信度
    - **Dis-Context**: 同上但提供原文上下文辅助判断
    - **Dis-Rating**: 让模型直接给出 0-10 的数值置信度

2. **生成式(Generative)置信度方法** — 基于采样一致性：
    - **Gen-Binary**: 额外采样 K 个回答，用 NLI 模型判断原子声明是否被支持，置信度 = |K_s| / |K|
    - **Gen-Multi**: 区分"冲突"和"未提及"，置信度 = |K_s| / (|K_s| + |K_c|)

3. **置信度融合策略**(本文提出):
    - **AdjustedAlpha**: 根据两个置信度的差异 d = B - A 动态调整融合权重 α' = α + γ_a · d
    - **DampedFusion**: 基于一致性施加阻尼 γ(d) = 1 - k · |d|，不一致时降低整体置信度
    - 核心思想: 传统加权平均无法区分 (0, 1) 和 (0.4, 0.6) 两种情况，前者不一致性更高应降低最终置信度

### 损失函数 / 优化目标

无训练过程。评估使用 Expected Calibration Error (ECE)、Brier Score (BS) 和 AUROC。

## 实验

### 主实验 — 原子级校准 (ECE ↓, BS ↓, AUROC ↑)

| 方法 | Llama3-8B ECE | Mistral-7B ECE | Qwen2-7B ECE |
|------|-------------|---------------|-------------|
| Dis-Context | 35.5 / 11.9 / 12.5 | 24.8 / 15.7 / 20.6 | 26.5 / 13.9 / 17.2 |
| Dis-Single | 32.6 / 14.3 / 19.2 | 30.2 / 20.4 / 24.0 | 29.3 / 16.1 / 18.7 |
| Gen-Binary | **10.0 / 8.5 / 11.1** | **13.7 / 8.4 / 12.7** | **10.9 / 6.3 / 9.5** |
| Gen-Multi | 37.4 / 12.6 / 21.9 | 42.2 / 13.4 / 26.6 | 41.7 / 11.6 / 21.0 |

(三列分别对应 Bios / LongFact / WildHallu 数据集)

### 消融实验 — 模型大小对校准的影响

| 方法 | Qwen2-7B ECE | Qwen2-57B ECE | Qwen2-72B ECE |
|------|-------------|---------------|-------------|
| Gen-Binary | 10.9 | 10.5 | 11.2 |
| Dis-Rating | 41.5 | 23.2 | **11.4** |

(Bios 数据集)

### 置信度融合结果 (跨类别: Gen-Binary + Dis-Context)

| 融合方法 | Llama3-8B ECE | Mistral-7B ECE |
|---------|-------------|---------------|
| WAvg (加权平均) | 15.2 | 12.8 |
| MinConf | 14.0 | 11.5 |
| **AdjustedAlpha** | **9.3** | **10.2** |
| **DampedFusion** | **9.5** | **10.4** |

### 关键发现

1. **原子级校准显著差于响应级**: 所有模型在原子级的 ECE 远高于响应级(数据点一致落在 identity line 之上)，即使响应级看似校准良好的模型在原子级仍表现差
2. **Gen-Binary 是最可靠的单一方法**: 在几乎所有模型和数据集上 ECE 最低，但 AUROC 并非最高——说明校准与区分能力是不同维度
3. **判别式和生成式方法互补**: 跨类别融合(Gen + Dis)显著提升校准，同类别融合(Dis + Dis)收益有限
4. **大模型不一定校准更好**: 生成式方法对模型大小不敏感；判别式方法中大模型显著更好(Qwen2-7B Dis-Rating ECE 41.5 → Qwen2-72B 11.4)
5. **置信度随生成过程变化**: 判别式方法中模型置信度随生成进行而递减；生成式方法在生成中段置信度最低——暗示不同方法捕捉了不同类型的不确定性

## 亮点

- **系统性框架**: 首次对原子级校准给出形式化定义(Definition 2)，明确区分 macro vs atomic calibration，证明二者不可互换
- **方法分类学**: 将置信度获取方法划分为判别式/生成式两类，揭示了互补性，为后续研究提供了清晰的方法论指导
- **深刻的分析洞察**: 置信度随生成位置变化的模式分析和跨方法对齐分析提供了对 LLM 内在不确定性的新理解

## 局限性

- 原子声明分解和事实性验证依赖 GPT-4o，引入管道误差
- 未考虑后处理校准方法(如 temperature scaling, Platt scaling)，仅评估"原始"校准
- 仅使用 7 个模型(3 个家族)，对更大规模模型(如 GPT-4, Claude)的分析缺失
- 置信度融合策略(AdjustedAlpha, DampedFusion)的超参数 γ_a 和 k 需要验证集调优
- 仅聚焦事实性维度，连贯性、创造性等其他质量维度的原子级校准未探索

## 相关工作

- **原子事实分解**: FActScore (Min et al., 2023), VeriScore (Song et al., 2024), D-FActScore (Chiang & Lee, 2024)
- **不确定性估计**: Semantic Entropy (Kuhn et al., 2022), P(true) (Kadavath et al., 2022), Self-Rating (Tian et al., 2023)
- **长文本校准**: Luq (Zhang et al., 2024b), Linguistic Calibration (Band et al., 2024)
- **置信度融合**: Rivera et al. (2024) 加权平均

## 评分

- **新颖性**: 8/10 — 原子级校准的形式化定义及判别/生成方法互补性的发现具有重要意义
- **技术深度**: 7/10 — 融合策略简洁有效但技术复杂度不高，形式化定义严谨
- **实验充分度**: 8/10 — 7 个模型、3 个数据集、5 种方法、多种融合策略，分析维度丰富(模型大小、生成位置、方法对齐)
- **清晰度**: 8/10 — 概念定义清晰，实验结果呈现系统化，图表辅助理解
- **总分**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)

</div>

<!-- RELATED:END -->
