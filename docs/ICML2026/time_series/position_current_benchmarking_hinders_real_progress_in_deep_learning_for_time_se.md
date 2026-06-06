---
title: >-
  [论文解读] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series
description: >-
  [ICML 2026][时间序列][时间序列预测] 这篇位置论文系统揭示了当前时间序列预测基准测试的核心问题——**设计选择的差异**（全局 / 局部参数、预处理、外源变量、时间和空间处理）往往被当作"实现细节"忽视，导致不同论文的对比不公平；通过 44 数据集 × 7 SOTA × 多个参考架构的受控实验…
tags:
  - "ICML 2026"
  - "时间序列"
  - "时间序列预测"
  - "基准测试"
  - "设计维度"
  - "模型对比"
  - "评估方法论"
---

# Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series

**会议**: ICML 2026  
**arXiv**: [2512.22702](https://arxiv.org/abs/2512.22702)  
**代码**: 待确认  
**领域**: 时间序列 / 基准测试方法论  
**关键词**: 时间序列预测, 基准测试, 设计维度, 模型对比, 评估方法论

## 一句话总结
这篇位置论文系统揭示了当前时间序列预测基准测试的核心问题——**设计选择的差异**（全局 / 局部参数、预处理、外源变量、时间和空间处理）往往被当作"实现细节"忽视，导致不同论文的对比不公平；通过 44 数据集 × 7 SOTA × 多个参考架构的受控实验，证明这些差异的影响（5-15%）常常**超过具体序列建模层的贡献**（1-3%）。

## 研究背景与动机

**领域现状**：时间序列预测在深度学习中很受欢迎，每年大量新架构（PatchTST、iTransformer、TimeMixer、Crossformer）。然而这些新方法在发表的基准上结果不一致——同一数据集同一模型，不同论文报告差异很大。

**现有痛点**：新 SOTA 很快被推翻，说明我们对性能改进的根本来源缺乏理解。社区陷入循环：新结果不断被新证据否定，但没人能清楚解释矛盾根源。

**核心矛盾**：论文对比时通常只报告最终性能数字，却忽略隐藏在实现细节中的关键设计选择——（1）模型配置（全局 / 局部 / 混合）、（2）预处理 + 外源变量、（3）时间依赖建模、（4）空间依赖处理。对比两个复杂架构时，无法确定哪部分性能提升来自新型注意力机制、哪部分来自其他设计维度的变化。

**本文目标**：通过严格的实验设计，量化四个关键设计维度对模型性能的影响，证明当前的基准测试实践系统性地失效。

**切入角度**：使用"**参考架构**"（精简的、成分明确的基线如纯 MLP / TCN / Transformer / RNN）进行受控对比，通过逐步引入或移除特定设计选择直接评估每个维度的真实影响。

**核心 idea**：基准测试实践需要彻底改进——不仅要确保超参一致，还要显式地在**模型卡（model card）**中声明关键设计选择，以便未来对比能"苹果对苹果"。

## 方法详解

### 整体框架
**诊断阶段**：选择 7 个 SOTA 模型（PatchTST / DLinear / TimeMixer / iTransformer / Crossformer / ModernTCN / Linear 基线）在 44 个真实数据集上进行受控实验；同时实现一套"参考架构"（MLP / TCN / RNN / Transformer / Pyramidal Attention 等）。通过在参考架构上系统启用 / 禁用特定设计特性隔离地测量影响。

**分析阶段**：沿四个设计维度逐一分析性能差异根源；对每个维度通过对比表格量化不同设计选择的影响幅度（MSE / MAE 变化）。

**产出**：一份"时间序列预测模型卡"模板，为每个新提出的预测架构定义必须显式声明的关键字段。

### 关键设计

1. **设计维度 D1 + D2（模型配置 + 预处理 / 外源变量）**:

    - 功能：决定模型参数如何在多条时间序列间共享，以及如何处理数据规范化、季节性、外源协变量。
    - 核心思路：D1 三种极端——全局模型（全部参数共享）、局部模型（每条序列单独训练）、混合模型（共享大部分参数但留出局部参数）。D2 涉及外源变量、日历特征、规范化方法。通过对比记录每个模型是否使用了时间序列特定的参数（可学习序列嵌入、规范化层中的局部参数）来消除歧义。
    - 设计动机：当前很多论文模糊这些选择。Table 1 显示同一模型加入局部参数可使 MSE 改变 7%-30%（Electricity 数据集上 Transformer 从 0.151 → 0.136）；Table 2 显示仅加入 / 移除日历特征 Traffic 上 Transformer MSE 从 0.479 → 0.417（降 13%）——这些差异堪比任何"架构创新"。

2. **设计维度 D3 + D4（时间处理 + 空间处理）**:

    - 功能：决定如何在时间轴上建模单条序列内的依赖，以及如何建模多条序列间相互依赖。
    - 核心思路：D3 是传统意义"模型选择"的核心（MLP / TCN / RNN / Transformer / 金字塔注意力）；D4 涉及通道独立性、跨序列注意力等。通过参考架构的消融实验量化每个维度单独的贡献。
    - 设计动机：令人惊讶的是 Table 3 发现在保持其他配置一致前提下，不同时间处理算子之间的性能差异被夸大了——MLP 与 Transformer 在 Electricity / Weather 上几乎无法区分；Table 4 对比有无空间注意力的模型差异 < 5%。这意味着许多论文声称的"新算子带来的性能提升"很大一部分可能来自隐藏的设计维度差异。

3. **时间序列预测模型卡（Forecasting Model Card）**:

    - 功能：强制要求每个新提出的预测架构显式声明关键设计选择，为未来论文投稿的元数据标准。
    - 核心思路：包含 D1-D4 四个维度的明确声明，以及超参调优、数据预处理、随机种子等元数据字段。
    - 设计动机：从源头杜绝隐藏差异，建立可重复、可比较的基准。

## 实验关键数据

### 主实验（四个设计维度的影响量化）

| 维度 | 模型 | 配置 A | 配置 B | 影响 (%) |
|------|------|--------|--------|---------|
| D1 | Transformer | 全局参数 | 局部参数 | -10.0 (Electricity) |
| D1 | iTransformer | 混合 | 全局 | +9.6 |
| D2 | Transformer | 有外源 | 无外源 | +13.6 (Traffic) |
| D2 | PatchTST | 无外源 | 有外源 | -4.7 (Electricity) |
| D3/D4 | MLP | — | — | 0.129 ± 0.000 (Electricity) |
| D3/D4 | Transformer | — | — | 0.129 ± 0.001 |
| D3/D4 | TCN | — | — | 0.130 ± 0.000 |

**关键发现**：D1 / D2 的影响（5-15%）远大于 D3（通常 < 5%）——直接否定了当前基准测试的隐含假设。

### 消融实验

| 配置 | Electricity | Weather | Traffic | Solar |
|------|------------|---------|---------|-------|
| 全局 + 无外源 + MLP | 0.129 | 0.148 | 0.376 | 0.194 |
| 全局 + 有外源 + MLP | 0.127 | 0.146 | 0.342 | 0.191 |
| 混合 + 有外源 + Transformer | 0.136 | 0.153 | 0.362 | 0.196 |
| 混合 + 有外源 + TCN | 0.130 | 0.148 | 0.364 | 0.193 |

简单的参考模型（如 MLP）在设计选择对齐后，性能与多数 SOTA 模型相当。

### 关键发现
- **设计维度比架构创新更重要**：在保持其他条件一致前提下，改变预处理或参数共享策略的影响（5-15%）超过改变序列建模层（1-3%），完全颠覆过去十年论文的隐含宣传。
- **隐藏的异质性**：A 论文和 B 论文报告的"PatchTST"虽然同名，但由于隐藏的 D1-D4 差异，实际上在测试不同的东西——解释了新论文发表的 SOTA 很快被推翻。
- **参考架构的启示**：简单架构（甚至单变量线性模型）在设计选择对齐后，往往能与复杂 SOTA 模型竞争——复杂模型的性能提升被伪装成了"架构优越性"。

## 亮点与洞察
- **方法论突破**：通过参考架构和设计维度的系统分解，让每个维度的贡献变得透明可见；这种"隔离变量法"简单但高效，可直接应用到其他领域（NLP、CV）的基准测试中。
- **最大的"啊哈"时刻**：仅通过改变参数共享方式性能就能改变 10%，揭示了一个令人不安的真相——过去十年对"架构创新"的津津乐道，可能有相当大一部分源于对基准测试实践的忽视。
- **可复用的思路**："模型卡"概念是一份强制要求声明所有关键设计选择的模板——可直接推广到时间序列外的其他领域，成为未来论文投稿的元数据标准。

## 局限与展望
- 本文仅分析长期预测（96 / 336 步）场景，短期预测可能有不同规律。
- 实验覆盖 44 个数据集，相对深度学习广阔应用范围仍有限。
- 没有给出"多少差异算过大"的明确阈值。
- 没有展示如何回溯性地为已发表论文评估它们的"隐藏设计维度得分"。
- 改进：建立"时间序列论文元数据数据库"自动提取或要求作者声明每篇论文的 D1-D4 取值；联合期刊编辑部制定时间序列论文的投稿检查清单。

## 相关工作与启发
- **vs Brigato et al. (2026)**：Brigato 聚焦于评估程序本身（交叉验证设置）；本文从架构设计维度深入，指出问题根源是设计选择本身的异质性。
- **vs Zeng et al. (2023)**：Zeng 发现线性模型有时胜过 Transformer；本文在更系统的框架下证实并解释——不是线性模型真的更优，而是不同论文对它们采用了不同的设计选择。
- **启发**：本文方法论借鉴了经典的科学实验设计原则；ML 社区在追求新架构时有时缺乏对基础实验设计的反思。

## 评分
- 新颖性: ⭐⭐⭐⭐  不是第一篇指出基准测试问题的论文，但首次系统识别并量化四个关键设计维度的独立影响。
- 实验充分度: ⭐⭐⭐⭐⭐  44 数据集 + 7 SOTA + 多参考架构 + 3 次独立运行 + 明确消融，实验规模和严谨度顶级。
- 写作质量: ⭐⭐⭐⭐  逻辑清晰，从问题诊断 → 框架提出 → 实验验证 → 解决方案一气呵成。
- 价值: ⭐⭐⭐⭐⭐  对整个时间序列预测社区的影响可能很深远，有潜力推动该领域走向更成熟、更规范的评估标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](../../NeurIPS2025/time_series/selective_learning_for_deep_time_series_forecasting.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)

</div>

<!-- RELATED:END -->
