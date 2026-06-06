---
title: >-
  [论文解读] TabStruct: Measuring Structural Fidelity of Tabular Data
description: >-
  [ICLR 2026][LLM评测][表格数据生成] 提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "表格数据生成"
  - "结构保真度"
  - "因果结构"
  - "全局效用"
  - "条件独立性"
---

# TabStruct: Measuring Structural Fidelity of Tabular Data

**会议**: ICLR 2026  
**arXiv**: [2509.11950](https://arxiv.org/abs/2509.11950)  
**代码**: [https://github.com/SilenceX12138/TabStruct](https://github.com/SilenceX12138/TabStruct)  
**领域**: 数据生成 / 表格数据 / 因果结构  
**关键词**: 表格数据生成, 结构保真度, 因果结构, 全局效用, 条件独立性

## 一句话总结

提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。

## 研究背景与动机

**领域现状**：表格数据生成是增强训练、缺失填补等任务的基础。现有评估主要关注三个维度：密度估计（分布是否接近）、ML 效用（下游预测性能）、隐私保护（合成数据与真实数据距离）。

**现有痛点**：这三个维度都来自同质模态（文本/图像），没有针对表格数据的异质特性做专门评估。一个典型问题是 SMOTE 在密度估计和 ML 效用上可以很好，但生成的数据严重违反了变量间的因果关系——比如物理定律所蕴含的条件独立性被破坏了。

**核心矛盾**：表格数据的核心先验是结构因果模型（SCM），变量间存在因果依赖关系。传统指标只评估边缘分布或某个 target 的预测，无法捕捉特征间的全局因果交互。且现有唯一考虑结构保真度的 CauTabBench 仅限于玩具 SCM 数据集，因为量化结构保真度需要真实因果图，而真实世界数据集几乎不可能获取。

**本文目标**
   - 如何在没有真实因果图的情况下评估结构保真度？
   - 现有生成器在结构保真度上表现如何？
   - 结构保真度与传统评估维度是什么关系？

**切入角度**：作者观察到如果一个生成器真正学到了数据的因果结构，那么用合成数据训练的模型应该能以每个变量为目标、用其他变量来预测，且性能都与真实数据接近。这个"全变量可预测"的性质与 SCM 中的 Markov blanket 概念紧密相关。

**核心 idea**：将每个变量轮流作为预测目标，聚合所有变量的预测性能比值作为 global utility，无需因果图即可评估表格数据的全局结构保真度。

## 方法详解

### 整体框架

TabStruct 是一个统一的评估框架，输入是参考数据集 $\mathcal{D}_{\text{ref}}$ 和合成数据集 $\mathcal{D}_{\text{syn}}$，输出是四个维度的评估分数：密度估计、隐私保护、ML 效用、结构保真度。其中结构保真度是本文新增的核心维度。

评估流程分两种场景：
- **有 SCM 数据集**：用条件独立性（CI）测试直接量化结构保真度
- **无 SCM 真实数据集**：用 global utility 间接衡量结构保真度

### 关键设计

1. **条件独立性（CI）评分——有因果图时的结构保真度**

    - 功能：在已知真实 SCM 的数据集上，通过对比真实数据和合成数据中 CI 语句的一致性来量化结构保真度
    - 核心思路：根据真实 SCM 的 CPDAG 枚举所有 CI 语句 $\mathcal{C}_{\text{global}}$，包括 d-separation 和 d-connection。对每个 CI 语句，在合成数据上做统计检验（$\alpha=0.01$），计算通过率 $CI(\mathcal{C}, \mathcal{D}) = \frac{1}{|\mathcal{C}|}\sum \mathbb{1}[\hat{\mathcal{I}}_\alpha = 1]$
    - 设计动机：在 CPDAG 层面评估而非 DAG 层面，因为现有因果发现方法在特征数 >10 时就不可靠；也不在骨架层面，因为会丢失方向信息
    - 区分局部和全局：local CI 只考虑与预测目标 $y$ 相关的 CI 语句，global CI 考虑所有变量对

2. **Global Utility——无因果图时的结构保真度代理指标**

    - 功能：在没有真实 SCM 的数据集上衡量全局结构保持程度
    - 核心思路：将每个变量 $x_j$ 轮流作为预测目标，用其余变量预测它。定义单变量效用为预测性能与参考数据的比值（分类用 balanced accuracy，回归用 RMSE 取倒数），全局效用为所有变量效用的平均：$\text{Global Utility}(\mathcal{D}) = \frac{1}{D+1}\sum_{j=1}^{D+1}\text{Utility}_j(\mathcal{D})$
    - 设计动机：解决两个问题：一是避免 local utility（仅预测 $y$）对特定目标的偏差；二是聚合归一化后的性能比值使不同类型任务可比较。使用 AutoGluon 集成 9 种预测器来减少单一模型的偏差
    - 理论基础：高保真生成器应使每个变量的条件分布 $p(x_j | \mathcal{X} \setminus \{x_j\})$ 与真实数据一致，这与 Markov blanket 概念一致

3. **Local Utility vs. Global Utility 的区别**

    - 功能：通过对比说明 local utility = ML efficacy 的局限
    - 核心思路：local utility 仅关注预测目标 $y$。实验证明 local utility 与 local CI 强相关（$r_s=0.78$），但与 global CI 几乎无关（$r_s=0.14$）。而 global utility 与 global CI 强相关（$r_s=0.84$）
    - 设计动机：表明传统 ML efficacy 只能反映局部结构，不适合全面评估生成器

### 评估维度整合

框架统一考虑四个维度：密度估计（Shape/Trend）、隐私保护（$\alpha$-precision/$\beta$-recall/DCR/$\delta$-Presence）、ML 效用（local utility）、结构保真度（CI score/global utility），形成对生成器的全面评价。

## 实验关键数据

### 主实验——13 种生成器在 SCM 数据集上的结构保真度

| 生成器 | Global CI ↑ | Global Utility ↑ | Local Utility ↑ | Shape ↑ |
|--------|------------|-------------------|-----------------|---------|
| $\mathcal{D}_\text{ref}$ | 1.00 | 0.99 | 0.99 | 1.00 |
| TabSyn | **0.70** | 0.76 | 0.76 | 0.50 |
| TabDDPM | **0.69** | **0.80** | 0.29 | 0.62 |
| TabDiff | 0.57 | 0.75 | 0.80 | 0.69 |
| SMOTE | 0.30 | 0.39 | **0.92** | 0.82 |
| CTGAN | 0.08 | 0.26 | 0.80 | 0.46 |
| GReaT | 0.16 | 0.25 | 0.27 | 0.62 |
| NRGBoost | 0.11 | 0.16 | 0.75 | 0.65 |

关键发现：SMOTE 的 local utility 最高（0.92），但 global CI 仅 0.30——说明传统 ML 效用会误导评估。扩散模型（TabDDPM/TabSyn/TabDiff）在全局结构保真度上一致最优。

### 真实数据集上 Global Utility 排名

| 生成器 | Global Utility ↑ | Local Utility ↑ |
|--------|-------------------|-----------------|
| $\mathcal{D}_\text{ref}$ | 0.99 | 0.96 |
| TabSyn | **0.73** | 0.76 |
| TabDiff | **0.73** | 0.78 |
| TabDDPM | **0.72** | 0.27 |
| ARF | 0.56 | 0.54 |
| TVAE | 0.53 | 0.70 |
| BN | 0.44 | 0.38 |
| SMOTE | 0.41 | **0.91** |
| CTGAN | 0.13 | 0.70 |
| GReaT | 0.20 | 0.23 |

在真实数据集上，Global Utility 的 Top-3 仍然是扩散模型（TabSyn/TabDiff/TabDDPM），与 SCM 数据集一致，验证了 global utility 的泛化能力。

### 相关性分析

| 指标对 | Spearman $r_s$ | p 值 |
|--------|----------------|------|
| Global Utility ↔ Global CI | **0.84** | <0.001 |
| Local Utility ↔ Local CI | 0.78 | <0.001 |
| Local Utility ↔ Global CI | 0.14 | <0.001 |

Global utility 与 global CI 的强相关性（0.84）是核心实证结果，证明其作为无 SCM 代理指标的有效性。

## 亮点与创新

- **填补评估空白**：首次系统性地将结构保真度纳入表格生成器评估框架，且提出的 global utility 不需要真实因果图
- **规模化基准**：13 种生成器 × 29 个数据集 × 超 15 万次评估，远超现有基准的覆盖范围
- **揭示扩散模型优势的本质**：扩散模型因噪声独立添加到每个特征、去噪时同时重建所有特征，天然学习置换不变的条件分布，与表格数据结构先验对齐
- **打破 SMOTE 的"幻觉"**：实验清楚展示 SMOTE 在传统指标上表现好但严重违反因果结构，揭示了评估偏差

## 局限性

- Global utility 与 global CI 的强相关是经验发现，缺乏理论证明
- 评估在 CPDAG 层面进行，比完整 DAG 层面粗糙，可能遗漏某些方向性因果关系
- 依赖 AutoGluon 集成预测器，计算开销随特征数增长（虽然 Tiny-default 变体效率较好，0.64s/1000 样本）
- SCM 数据集使用专家验证的因果图，但这类数据集数量有限（仅 6 个），泛化性有待扩展

## 相关工作

- **表格生成基准**：Synthcity（Qian et al., 2024）、SynMeter（Du & Li, 2024）覆盖密度/隐私/ML 效用，但无结构保真度；CauTabBench（Tu et al., 2024）评估结构保真度但仅限玩具 SCM
- **表格生成器**：从 SMOTE、BN 到 TVAE、CTGAN、NFlow、ARF，再到扩散模型 TabDDPM/TabSyn/TabDiff/TabEBM，以及 LLM-based 的 GReaT 和树模型 NRGBoost
- **因果发现**：DAG 学习方法在特征数 >10 时困难重重（Zanga et al., 2022），本文因此选择 CPDAG 层面评估
- **表格基础模型**：Hollmann et al. (2025) 经验证明 SCM 是表格数据的有效结构先验

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次提出无需因果图的全局结构保真度指标 |
| 技术深度 | ⭐⭐⭐⭐ | 理论分析清晰，CI 框架 + global utility 设计合理 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 13 模型 × 29 数据集，15 万次评估，极其充分 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，动机示例直观 |
| 实用价值 | ⭐⭐⭐⭐ | 开源框架，可直接用于评估新生成器 |
| 总评 | ⭐⭐⭐⭐ | 表格数据生成评估的重要贡献，global utility 有望成为标准指标 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)

</div>

<!-- RELATED:END -->
