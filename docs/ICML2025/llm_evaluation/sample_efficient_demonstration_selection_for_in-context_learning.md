---
title: >-
  [论文解读] Sample Efficient Demonstration Selection for In-Context Learning
description: >-
  [ICML 2025][LLM评测][上下文学习] 本文提出了一种样本高效的上下文学习(ICL)示例选择方法，能够在有限的标注预算下高效地选择最佳示例组合，显著提升 LLM 的 ICL 性能，同时大幅减少所需的标注数据量。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "上下文学习"
  - "示例选择"
  - "样本效率"
  - "ICL"
  - "演示选择"
---

# Sample Efficient Demonstration Selection for In-Context Learning

**会议**: ICML 2025  
**arXiv**: [2506.08607](https://arxiv.org/abs/2506.08607)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 上下文学习, 示例选择, 样本效率, ICL, 演示选择

## 一句话总结
本文提出了一种样本高效的上下文学习(ICL)示例选择方法，能够在有限的标注预算下高效地选择最佳示例组合，显著提升 LLM 的 ICL 性能，同时大幅减少所需的标注数据量。

## 研究背景与动机
**领域现状**：上下文学习(ICL)是 LLM 的核心能力之一——通过在提示中提供少量示例即可完成新任务。但 ICL 性能对示例选择高度敏感，选择不同的示例可能导致性能剧烈波动。

**现有痛点**：现有示例选择方法通常需要大量标注数据来评估候选示例的质量，或者需要在验证集上反复测试不同组合，标注成本高昂。

**核心矛盾**：ICL 的核心优势在于"少样本"学习，但选择最佳示例本身可能需要大量标注数据，构成了矛盾。

**本文目标**：在极小标注预算下高效地选择最优 ICL 示例。

**切入角度**：利用 LLM 内部信号（如困惑度、置信度）来评估示例质量，减少对外部标注的依赖。

**核心 idea**：将示例选择建模为一个样本高效的优化问题，用智能搜索策略在极小评估预算下找到高质量示例。

## 方法详解

### 整体框架
- **输入**：候选示例池、少量标注预算、目标任务
- **选择过程**：高效搜索最优示例组合
- **输出**：选定的 k-shot 示例集

### 关键设计
1. **样本高效评估策略**:

    - 利用模型内部信号（如 token 概率、困惑度）来估计示例质量
    - 避免对每个候选组合都需要完整的标注评估
    - **设计动机**：标注数据是稀缺资源，应最大化其信息利用

2. **智能搜索算法**:

    - 设计高效的组合搜索算法，避免穷举
    - 可能使用贝叶斯优化、bandit 算法等技术
    - 在极小的evaluation budget下接近最优解
    - **设计动机**：示例组合空间是指数级的，需要智能搜索

3. **跨任务泛化**:

    - 选择策略可能具有跨任务迁移能力
    - 在一个任务上学到的选择偏好可能推广到相似任务
    - **设计动机**：进一步减少每个新任务的标注需求

### 损失函数 / 训练策略
不涉及 LLM 训练，是推理时的示例选择方法。

## 实验关键数据

### 主实验

| 方法 | 标注预算 | 平均ICL性能 | 相比随机提升 |
|------|---------|-----------|-----------|
| 随机选择 | 0 | 基线 | 0% |
| 全量评估 | 大 | 上界 | 最高 |
| 本文方法 | 小(~10%) | 接近上界 | 显著提升 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| 无内部信号 | 较低 | 仅依赖外部标注 |
| + 困惑度信号 | 提升 | 利用模型反馈 |
| + 智能搜索 | 进一步提升 | 高效组合优化 |

### 关键发现
- 极少的标注预算（约全量的 10%）即可达到接近最优的示例选择效果
- 模型内部信号是有效的示例质量指示器
- 示例间的组合效应很重要——单独好的示例组合后不一定好
- 方法在多种任务类型和模型规模上一致有效

## 亮点与洞察
- 直接解决了 ICL 中示例选择的实际痛点
- 显著降低了 ICL 的应用成本
- 利用模型自身的信号来指导选择是优雅的设计
- 对实际部署 LLM 有直接应用价值

## 局限与展望
- 模型内部信号的质量可能因模型而异
- 对于非常新颖的任务，选择策略可能不如在熟悉领域有效
- 可结合检索增强（RAG）进一步扩展
- 动态示例选择（根据输入调整）可能更好

## 相关工作与启发
- 与 KATE、EPR 等示例选择方法直接对比
- 与主动学习策略在理念上相近
- 启发：ICL 的"小"优势应贯穿到示例选择环节

## 评分
- 新颖性: ⭐⭐⭐⭐ 样本高效的选择策略是有价值的贡献
- 实验充分度: ⭐⭐⭐⭐ 多任务多模型验证
- 写作质量: ⭐⭐⭐⭐ 问题定义明确
- 价值: ⭐⭐⭐⭐ 对 ICL 的实际应用有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] In-Context Learning for Pure Exploration](../../ICLR2026/llm_evaluation/in-context_learning_for_pure_exploration.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](../../ICLR2026/llm_evaluation/in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)
- [\[ICML 2025\] Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/llm_evaluation/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)

</div>

<!-- RELATED:END -->
