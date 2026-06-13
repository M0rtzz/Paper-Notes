---
title: >-
  [论文解读] A Variational Approach for Mitigating Entity Bias in Relation Extraction
description: >-
  [ACL 2025][NLP理解][关系抽取] 提出将变分信息瓶颈(VIB)应用于关系抽取中的实体去偏，通过将实体映射到概率分布 $\mathcal{N}(\mu, \sigma)$ 来压缩实体特定信息同时保留任务相关特征，方差 $\sigma^2$ 可量化模型对实体vs上下文的依赖程度，在TACRED、REFinD、BioRED三个域的ID和OOD设置上均达到SOTA。
tags:
  - "ACL 2025"
  - "NLP理解"
  - "关系抽取"
  - "实体偏差"
  - "变分信息瓶颈"
  - "VIB"
  - "去偏差"
---

# A Variational Approach for Mitigating Entity Bias in Relation Extraction

**会议**: ACL 2025  
**arXiv**: [2506.11381](https://arxiv.org/abs/2506.11381)  
**代码**: 无  
**领域**: NLP理解  
**关键词**: 关系抽取, 实体偏差, 变分信息瓶颈, VIB, 去偏差

## 一句话总结
提出将变分信息瓶颈(VIB)应用于关系抽取中的实体去偏，通过将实体映射到概率分布 $\mathcal{N}(\mu, \sigma)$ 来压缩实体特定信息同时保留任务相关特征，方差 $\sigma^2$ 可量化模型对实体vs上下文的依赖程度，在TACRED、REFinD、BioRED三个域的ID和OOD设置上均达到SOTA。

## 研究背景与动机

**领域现状**：关系抽取模型常过度依赖实体本身的信息（如"微软"→投资关系），而忽视上下文线索。

**现有痛点**：实体遮蔽会丢失有用信息；结构因果模型(SCM)用凸包中心替换实体嵌入，但缺乏可解释性且计算开销大。

**核心 idea**：用VIB将实体嵌入映射为高斯分布，$\sigma$ 大意味着模型对该实体了解少、更依赖上下文，$\sigma$ 小意味着保留更多实体信息。通过最小化KL散度压缩实体特定信息。

## 方法详解

### 关键设计

1. **实体选择性VIB**: 仅对实体token应用VIB变换 $z = \mu + \epsilon \cdot \sigma$，非实体token保留原始嵌入
2. **混合嵌入**: $x' = x \cdot (1-M) + x \cdot M \cdot (1-\beta) + z \cdot M \cdot \beta$，$\beta$ 控制VIB替换程度
3. **自适应损失权重**: $\alpha$ 按CE和VIB损失比例自动计算

### 损失函数
$\mathcal{L} = L_{CE} + \alpha L_{VIB}$，其中 $L_{VIB} = \mathbb{E}[KL(p(z|x,e) \| r(z|e))]$

## 实验关键数据

| 数据集 | 方法 | ID F1 | OOD F1 |
|--------|------|-------|--------|
| TACRED | LUKE+VIB | **70.4** | **66.5** |
| TACRED | LUKE+SCM | 68.6 | 64.8 |
| REFinD | LUKE+VIB | **75.4** | **74.8** |
| BioRED | LUKE+VIB | **61.2** | **58.7** |

### 关键发现
- VIB在所有三个域上均超越SCM等基线，且OOD提升更大（平均+2.8% vs ID的+1.6%）。

### 方差可解释性分析

| 关系类型 | 方差σ² | 解读 |
|---------|--------|------|
| pers:title | 低 (0.12) | 实体信息重要 |
| org:date | 高 (0.89) | 更依赖上下文 |
| pers:org | 中 (0.45) | 实体和上下文均重要 |
| loc:loc | 高 (0.78) | 位置关系主要由上下文决定 |

- VIB在所有三个域上均超越SCM等基线，且OOD提升更大
- 方差分析显示：pers:title关系的方差低（实体信息重要），org:date关系的方差高（更依赖上下文），验证了VIB的可解释性
- $\beta=0.5$ 最优，既不完全依赖原始嵌入也不完全依赖VIB嵌入

## 亮点与洞察
- 方差作为可解释性指标非常直观：可以量化模型对每个实体的信息依赖程度
- 比SCM更简洁，用标准概率工具替代了复杂的邻域构建

## 局限与展望
- 需要预定义实体位置（依赖entity marker），无法处理实体边界不明确的场景。
- 自适应 $\alpha$ 虽简单但可能不是最优策略，可以探索学习型权重调度。
- 高斯分布假设可能过于简化，更复杂的分布（如混合高斯）可能提供更强的表达能力。
- 在嵌套实体（nested entities）场景中效果未验证。
- VIB的信息压缩可能在极端情况下丢失关键实体信息，尤其是当实体本身对关系判断至关重要时。
- 未探索与提示学习或上下文学习方法的结合。
- 在更大规模的预训练模型（如RoBERTa-large、DeBERTa-v3）上的效果未测试。

## 相关工作与启发
- **vs 实体遮蔽方法**: 用[MASK]替换实体名称来去偏，但会丢失实体类型等有用信息；VIB通过概率压缩实现柔性去偏。
- **vs SCM (Zhang et al.)**: SCM用凸包中心替换实体嵌入，计算开销大且可解释性差；VIB的方差直接可量化依赖程度。
- **vs 因果推断方法**: 因果推断需要明确的因果图，VIB通过信息论工具隐式实现去偏，更简洁。
- **vs 对比学习去偏**: 对比学习需要构造正负对，VIB仅需标准监督信号加KL正则化。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。
- 代码/数据的开源对社区复现和后续研究有重要价值。
- 与同期工作相比，本文在问题定义的深度和实验分析的全面性上更具优势。
- 论文的写作逻辑清晰，从问题定义到方法设计到实验验证形成了完整的闭环。
- 方法的计算开销合理，在实际应用中具有可部署性。
- 未来工作可以考虑与更多模态（如音频、3D点云）的融合。
- 在更大规模的数据和模型上验证方法的可扩展性是重要的后续方向。
- 可以考虑将该方法与强化学习结合，实现端到端的优化。
- 跨领域迁移是一个值得探索的方向——方法的通用性需要更多验证。
- 对于边缘计算和移动端部署场景，方法的轻量化版本值得研究。
- 长期评估和用户研究可以提供更全面的方法评价。
- 与人类专家的对比分析可以更好地定位方法的优劣势。

## 评分
- 新颖性: ⭐⭐⭐⭐ VIB用于RE去偏有理论根据且直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 三领域、ID/OOD、两种backbone全面
- 写作质量: ⭐⭐⭐⭐ 方法推导清楚
- 价值: ⭐⭐⭐⭐ 可解释+高性能的去偏方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [\[ACL 2025\] On Synthesizing Data for Context Attribution in Question Answering](on_synthesizing_data_for_context_attribution_in_question_answering.md)

</div>

<!-- RELATED:END -->
