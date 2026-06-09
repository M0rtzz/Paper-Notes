---
title: >-
  [论文解读] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought
description: >-
  [NeurIPS 2025][预训练][learning curves] 构建了大规模高分辨率学习曲线数据库 LCDB 1.1，证明样本学习曲线的"病态行为"（非单调、非凸）比此前认为的普遍两倍，约 15% 的曲线显著不良，且特征缩放难以修复。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "learning curves"
  - "scaling laws"
  - "model selection"
  - "ill-behavior"
  - "tabular data"
  - "benchmark"
---

# LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought

**会议**: NeurIPS 2025  
**arXiv**: [2505.15657](https://arxiv.org/abs/2505.15657)  
**代码**: [GitHub](https://github.com/learning-curve-research/LCDB-1.1)  
**领域**: LLM评测  
**关键词**: learning curves, scaling laws, model selection, ill-behavior, tabular data, benchmark

## 一句话总结

构建了大规模高分辨率学习曲线数据库 LCDB 1.1，证明样本学习曲线的"病态行为"（非单调、非凸）比此前认为的普遍两倍，约 15% 的曲线显著不良，且特征缩放难以修复。

## 研究背景与动机

样本学习曲线（sample-wise learning curve）描绘模型性能随训练集大小的变化。它在多保真度超参搜索、scaling law 估计和数据需求预测方面至关重要。长期以来，学界普遍假设学习曲线是**单调递减**且**凸**的（well-behaved），即更多数据总能带来更好泛化、且边际收益递减。

前作 LCDB 1.0 声称"绝大多数学习曲线表现良好"。然而作者指出该结论过于草率，原因如下：

**分辨率不足**：LCDB 1.0 的锚点（anchor）间距过大，难以捕捉微妙的病态行为

**缺乏统计检验**：没有量化哪些不良行为是显著的

**未做特征缩放**：特征缩放是标准实践，缺失可能导致伪病态

**数据问题**：存在数据泄露和缺失值问题

这些局限促使作者构建 LCDB 1.1 来回答四个核心问题：病态行为有多普遍？哪些学习器负责？特征缩放能否修复？病态对下游任务（曲线拟合、模型选择）影响多大？

## 方法详解

### 整体框架

LCDB 1.1 是一个大规模学习曲线数据库，覆盖 265 个 OpenML 数据集（含精心筛选的 CC-18 子集 72 个）和 32 种学习器，包括现代方法 CatBoost、TabNet、RealMLP 和 TabPFN v2。

### 数据库设计改进

**数据分割**：采用 5 × 5 的外/内种子嵌套划分：

$$D \xrightarrow{\text{outer split}} (D_{\text{train-val}}^{(m_o)}, D_{\text{test}}^{(m_o)}) \xrightarrow{\text{inner split}} (D_{\text{train}}^{(m_o,m_i)}, D_{\text{val}}^{(m_o,m_i)})$$

**锚点分辨率提升 4 倍**：从 $n_k = \lceil 16 \cdot 2^{k/2} \rceil$ 提升到 $n_k = \lceil 16 \cdot 2^{k/8} \rceil$

**双版本**：提供有/无数据泄露两个版本，适配不同使用场景（模型选择 vs 数据需求估计）

**三种特征缩放**：无缩放（noFS）、min-max 缩放、标准化

### 病态行为的严格检测方法

**单调性违反**：定义最大违反误差：

$$\epsilon_{\text{mono}} = \max\left(0, \max_{1 \leq i < j \leq N} (C(n_j) - C(n_i))\right)$$

找到最大化的锚点对 $(i^*, j^*)$ 后，使用配对单侧 t 检验 + Bonferroni 校正（$\alpha' = \frac{\alpha}{N(N-1)/2}$），在 25 次重复上验证显著性。

**凸性违反**：基于线性插值定义违反误差：

$$\epsilon_{\text{conv}} = \max\left(0, \max_{1 \leq h < i < j \leq N} (C(n_i) - C_{\text{interpolated}}(n_i; n_h, n_j))\right)$$

三元组比较用 Bonferroni 校正 $\alpha' = \frac{\alpha}{N(N-1)(N-2)/6}$。注意与 LCDB 1.0 不同，本文正确考虑了锚点在对数尺度上的分布。

**Peaking**：结合凸性违反和单调性违反——在凸性违反点 $(h^*, i^*, j^*)$ 上，验证 $h^*$ 到 $i^*$ 的单调性违反且 $i^*$ 到 $j^*$ 的显著改善。

**Dipping**：单调性违反中 $j$ 固定为最后一个锚点 $N$，说明性能退化不可恢复。

### 下游影响分析

- **曲线拟合**：使用 POW4 ($\hat{C}(n) = a - b(d+n)^{-c}$) 等参数模型做插值，比较 MSE
- **模型选择**：使用 Successive Halving（SH）以训练集大小为保真度进行模型选择

## 实验关键数据

### 主实验：病态行为统计 (DA1)

| 指标 | LCDB 1.1 CC-18 (noFS) | LCDB 1.1 FULL (noFS) | LCDB 1.0 |
|------|----------------------|---------------------|----------|
| 非单调 (¬M) | **9.9%** | 9.6% | 5.1% |
| 非凸 (¬C) | **11.5%** | 12.3% | 5.7% |
| 病态 (¬M ∪ ¬C) | **14.9%** | 15.4% | 8.1% |
| Peaking | **5.0%** | 5.7% | 2.5% |
| Dipping | **6.1%** | 6.9% | 4.6% |

关键发现：约 15% 的学习曲线显著病态，接近 LCDB 1.0 估计的**两倍**。

**各学习器病态比例**（部分极端案例）：

| 学习器 | 病态比例 |
|--------|----------|
| CatBoost | 1.5% |
| TabPFN v2 | 1.5% |
| Decision Tree | 1.5% |
| KNN | 3.8% |
| MLP | 27.9% |
| LDA | 37.7% |
| QDA | 45.7% |
| Sigmoid SVM | 58.1% |
| TabNet | **74.3%** |

### 消融实验：特征缩放 (DA2)

| 缩放方式 | 病态比例 (CC-18) |
|----------|-----------------|
| 无缩放 | 14.9% |
| Min-max | 13.5% |
| 标准化 | 11.2% |

特征缩放仅轻微减少病态行为。Ridge 和 MLP 是少数显著受益的学习器，但 Nearest Centroid 在缩放后反而更差。

### 曲线拟合实验 (DA3)

非单调曲线的 POW4 拟合 MSE 均值（对数尺度）比单调曲线大 **10 倍以上**。违反误差与 MSE 呈明显正相关，确认参数模型难以拟合病态曲线。

### 模型选择实验 (DA4)

学习曲线频繁交叉的学习器组中 Successive Halving 的最优选择率和 regret 均显著劣于曲线很少交叉的组，表明交叉曲线使多保真度模型选择更加困难。

### 关键发现

1. 集成方法（随机森林、梯度提升）和 CatBoost 学习曲线最良好
2. TabNet 病态率高达 74.3%，主要表现为相变（phase transition），可能因默认超参不适合小数据集
3. Sigmoid SVM 是经典方法中最不稳定的，主要表现为 dipping
4. LDA 和 Ridge 的 peaking 与 Fisher 分类器理论一致——训练集大小约等于维度时 peak 最严重
5. 首次在真实数据集上观察到 MLP 的相变现象

## 亮点与洞察

1. **打破"学习曲线总是良好"的迷思**：用严格统计方法证明约 15% 曲线显著病态，是 NeurIPS 社区关于 scaling law 讨论的重要补充
2. **高质量 benchmark**：4 倍分辨率 + 双版本 + 三种缩放 + 32 种学习器 + 公开数据，对未来学习曲线建模研究价值巨大
3. **统计方法的严谨性**：Bonferroni 校正 + 配对检验确保不会将噪声误判为病态，结论更具说服力
4. **现代学习器的纳入**：CatBoost/TabPFN 等表现良好，与 TabNet 的鲜明对比为表格数据方法选择提供指导
5. **实际影响量化**：不仅指出病态存在，还量化了对曲线拟合和模型选择的实际危害

## 局限性

1. **仅覆盖表格分类任务**：未涉及回归、NLP、CV 等领域的学习曲线
2. **默认超参**：所有学习器使用默认超参，调参后病态行为可能减少（尤其 TabNet）
3. **计算开销巨大**：已消耗 80 万 CPU 小时，带超参调优的版本更加昂贵
4. **Bonferroni 过于保守**：可能低估真实病态比例（用 Holm 方法可达 19%）
5. **QDA 不可复现**：SVD 的数值非确定性导致部分结果不可重现

## 相关工作与启发

- 与 scaling law 文献（Kaplan et al. 2020）互补：后者关注大规模深度学习，本文关注表格数据经典方法
- 为 multi-fidelity 方法（如 BOHB、Hyperband）提供更真实的 benchmark
- 病态行为可能启发新的学习曲线参数模型设计（现有 power law 假设 well-behaved）
- Dipping 和 phase transition 现象值得理论分析

## 评分

- **创新性**: ⭐⭐⭐ — 核心贡献是数据库和分析方法而非全新算法，但统计检测框架设计精良
- **实用性**: ⭐⭐⭐⭐⭐ — 对超参搜索、数据预算估计、学习器选择有直接指导价值
- **实验严谨度**: ⭐⭐⭐⭐⭐ — 32 种学习器 × 265 数据集 × 3 种缩放 + 严格统计检验
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，但内容密度较高
- **推荐阅读指数**: ⭐⭐⭐⭐ — 做表格数据、AutoML、scaling law 研究必读

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ICML 2025\] In-Context Adaptation to Concept Drift for Learned Database Operations](../../ICML2025/llm_pretraining/in-context_adaptation_to_concept_drift_for_learned_database_operations.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)

</div>

<!-- RELATED:END -->
