---
title: >-
  [论文解读] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models
description: >-
  [ICML 2025][自监督学习][world model] 本文提出"归纳偏置探针"（Inductive Bias Probe），通过在合成数据上反复微调基础模型来测试其外推行为是否符合预设的世界模型，发现在轨道力学、Othello、格问题等领域中，基础模型虽然能准确预测序列但未真正学到底层世界模型，而是发展出特定于任务的启发式策略。
tags:
  - ICML 2025
  - 自监督学习
  - world model
  - foundation model
  - inductive bias
  - sequence prediction
  - Newtonian mechanics
---

# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models

**会议**: ICML 2025  
**arXiv**: [2507.06952](https://arxiv.org/abs/2507.06952)  
**代码**: https://github.com/keyonvafa/inductive-bias-probes  
**领域**: 自监督学习  
**关键词**: world model, foundation model, inductive bias, sequence prediction, Newtonian mechanics

## 一句话总结

本文提出"归纳偏置探针"（Inductive Bias Probe），通过在合成数据上反复微调基础模型来测试其外推行为是否符合预设的世界模型，发现在轨道力学、Othello、格问题等领域中，基础模型虽然能准确预测序列但未真正学到底层世界模型，而是发展出特定于任务的启发式策略。

## 研究背景与动机

**领域现状**：基础模型的核心假设是"学会预测序列就能发现更深层的领域理解"——正如开普勒的行星运动预测最终引向牛顿力学的发现。当前评估基础模型是否学到了世界模型的方法主要有两条路径：(1) 机制性探针（probing 内部表示）；(2) 静态行为测试（在单任务上评估输出）。

**现有痛点**：机制性探针难以扩展到大型模型，且内部表示未必反映实际行为；静态行为测试只观察一个任务，无法捕捉基础模型在新任务上的适应能力——而这恰恰是基础模型的核心使用场景。

**核心矛盾**：无免费午餐定理告诉我们，学习算法的优劣取决于它偏好学习哪类函数（归纳偏置）。如果一个基础模型真的学到了世界模型，它的归纳偏置应该朝向该世界模型允许的函数，即在少量数据上外推时自动遵循世界模型的结构。但现有方法无法系统性地测试这一点。

**本文目标** 给定一个基础模型和一个假定的世界模型，如何量化地测试该模型是否将世界模型作为其归纳偏置？

**切入角度**：科学家用世界模型从少量数据中做推断——同样地，基础模型的世界模型应该通过它在少量数据上的外推行为暴露出来。

**核心 idea**：通过反复在与世界模型一致的合成小数据集上微调基础模型，观察其外推是否"尊重"和"区分"世界模型的状态结构。

## 方法详解

### 整体框架

归纳偏置探针是一个通用的评估流程：(1) 给定假设的世界模型（包含状态空间和状态映射）；(2) 生成大量与世界模型一致的小型合成数据集；(3) 对每个合成数据集，用基础模型在训练集上学习后对测试输入做外推；(4) 比较基础模型的外推行为与"知道真实状态"的 oracle 的外推行为——如果基础模型的归纳偏置指向世界模型，二者应该高度一致。

### 关键设计

1. **R-IB 和 D-IB 指标（有限状态空间+二元输出的特例）**:

    - 功能：量化基础模型的归纳偏置是否与世界模型一致
    - 核心思路：定义两个互补指标：**R-IB（Respecting State）** = $\mathbb{E}[\mathbf{1}(\hat{m}_D(X_i), \hat{m}_D(X_j)) | \phi(X_i)=\phi(X_j)]$，即同一状态的输入是否得到相同预测；**D-IB（Distinguishing State）** = $1 - \mathbb{E}[\mathbf{1}(\hat{m}_D(X_i), \hat{m}_D(X_j)) | \phi(X_i) \neq \phi(X_j)]$，即不同状态的输入是否得到不同预测。类比精确率与召回率，两者缺一不可——模型总输出相同值可得到完美 R-IB 但 D-IB 为零
    - 设计动机：仅靠准确率无法区分模型是学到了状态结构还是恰好猜对了答案

2. **一般化的归纳偏置探针（连续输出+一般状态空间+一般任务）**:

    - 功能：将探针推广到轨道力学等连续域
    - 核心思路：定义**外推可预测性** $\hat{I}(x_i, x_j) = -\min_{h \in \mathcal{H}} \mathbb{E}_D[\ell(h(\hat{m}_D(x_i)), \hat{m}_D(x_j))]$，衡量模型在多个合成数据集上对两个输入的外推之间的可预测程度。然后定义 oracle 版本并比较：如果模型行为像 oracle，IB 曲线应落在45度线上。这本质上是一个校准曲线
    - 设计动机：在连续域中，"状态是否相同"需要推广为"状态有多相似"，因此需要更灵活的度量

3. **"下一 Token 分区"假说的验证**:

    - 功能：揭示基础模型在未学到世界模型时，实际使用了什么归纳偏置
    - 核心思路：提出并验证假说——基础模型的归纳偏置指向"具有相同合法下一 token 集合"的状态分区，而非真实的世界状态。通过将 D-IB 分解为 $\text{D-IB}_{q=}$（不同状态但相同合法下一 token）和 $\text{D-IB}_{q\neq}$（不同状态且不同合法下一 token），发现 $\text{D-IB}_{q=} < \text{D-IB}_{q\neq}$ 在所有模型中均成立，表明模型确实按下一 token 分区而非真实状态进行分组
    - 设计动机：回答"如果不是世界模型，那模型学到了什么？"这一关键问题

### 损失函数

本文不提出新的训练损失函数，而是在评估阶段使用 MSE 作为外推可预测性的损失函数 $\ell$。

## 实验关键数据

### 主实验表格

R-IB 和 D-IB 在格问题（5状态）和 Othello 上的表现（1为完美，0为无信息模型）：

| 模型 | 格问题 R-IB↑ | 格问题 D-IB↑ | Othello R-IB↑ | Othello D-IB↑ |
|---|---|---|---|---|
| RNN (untrained) | 0.346 | 0.749 | 0.228 | 0.990 |
| RNN (NTP trained) | 0.574 | 0.803 | 0.632 | 0.797 |
| Transformer (untrained) | 0.268 | 0.742 | 0.708 | 0.843 |
| Transformer (NTP trained) | 0.483 | 0.677 | 0.703 | 0.624 |
| Mamba (NTP trained) | 0.571 | 0.866 | 0.682 | 0.728 |
| LSTM (NTP trained) | 0.782 | 0.921 | 0.563 | 0.610 |

### 消融表格

轨道力学中 Transformer 通过符号回归恢复的力定律（对比真实 $F \propto m_1 m_2/r^2$）：

| 数据分片 | 恢复的力定律 |
|---|---|
| Galaxy 1 | $F \propto (\sin(1/\sin(r-0.24))+1.45) \times 1/(1/r+m_2)$ |
| Galaxy 2 | $F \propto \cos(\cos(2.19 \times m_1))$ |
| Galaxy 3 | $F \propto \cos(\sin(0.48/m_1))$ |
| Galaxy 4 | $F \propto \sin(r+8569.2+1/m_1)$ |
| Galaxy 5 | $F \propto \cos(\cos(e^{m_2}))$ |

### 关键发现

- 轨道力学模型预测精度 $R^2 > 0.9999$，但归纳偏置探针显示其完全未学到牛顿力学——对不同数据子集恢复出完全不同且无意义的引力定律
- Transformer 在所有架构中归纳偏置最差——格问题上从未训练到训练后 D-IB 反而下降，说明预训练可能损害了状态区分能力
- LSTM 在格问题上表现最好（R-IB=0.782），循环/状态空间架构总体优于 Transformer
- Othello 中即使预测的棋盘不完全正确，合法走子集合的匹配率仍远高于棋盘匹配率，证实模型学到的是"足够恢复合法走子"的粗糙表示而非完整棋盘状态

## 亮点与洞察

- 将"基础模型是否学到世界模型"这一哲学问题转化为可量化测试的统计框架，学术贡献显著
- "从开普勒到牛顿"的类比极其贴切——预测准确不等于理解规律
- 发现模型学到的是"下一 token 分区"而非真实状态，这一洞察对理解 LLM 的局限性具有深远意义
- 实验设计跨越物理学、博弈论、组合学三个领域，说服力强

## 局限性

- 测试的世界模型需要事先假定，无法自动发现模型内部实际使用的表示
- 实验模型规模较小（109M参数），大规模 LLM 是否有类似行为尚不明确
- 归纳偏置探针的计算开销非平凡——需要多次微调并收集外推结果
- 目前的结论偏向"负面发现"（模型没学到世界模型），缺少如何改善的建设性建议
- 对于"为什么 LSTM 比 Transformer 表现更好"缺乏深入解释——隐状态的归纳偏置是否天然更适合追踪状态？
- 合成数据集的取样分布 $P_D$ 和输入分布 $P_X$ 的选择可能显著影响结论，但缺乏敏感性分析

## 相关工作与启发

- 与 Nanda et al.（2023）的 Othello 内部探针形成互补——后者检查内部表示，本文检查行为表现
- Lemos et al.（2023）表明修改模型架构以显式编码牛顿定律可恢复引力定律，暗示领域特定归纳偏置是关键
- Li et al.（2023）对 Othello 的合法走子预测研究与本文的"下一 token 分区"发现高度一致
- 启发：评估基础模型不应仅看预测准确度，还应检查其外推是否遵循我们期望的世界模型结构

## 评分

⭐⭐⭐⭐ （8/10）

这是一篇视角独特、方法严谨的研究工作。归纳偏置探针框架的提出是对基础模型评估工具箱的重要补充，跨领域实验设计精妙，"模型学的是下一 token 分区而非世界模型"这一发现深刻且具有广泛影响力。不足在于模型规模较小、无建设性改进方案、部分实验（如轨道力学的符号回归）可复现性待考量。

进一步来看，本文的发现对 AI Safety 有重要启示：如果模型仅学到表面启发式而非真正的世界模型，那么在面对分布外场景时可能做出看似合理但本质错误的决策。这也和"Grokking"现象（模型在过拟合后突然泛化）形成有趣对比——本文的模型虽然达到了极高的预测精度（$R^2 > 0.9999$），但泛化测试表明它并未真正理解底层规律。

<!-- RELATED:START -->

## 相关论文

- [AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad](../../CVPR2025/self_supervised/ocrt_boosting_foundation_models_in_the_open_world_with_object-concept-relation_t.md)
- [Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

<!-- RELATED:END -->
