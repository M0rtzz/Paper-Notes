---
title: >-
  [论文解读] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models
description: >-
  [ACL 2025][知识神经元] > 本文从结构和功能两个角度全面定义了退化知识神经元 (DKN)，提出基于持久同调的神经拓扑聚类方法 (NTC) 获取 DKN，并通过 34 个实验揭示了 DKN 与 LLM 鲁棒性、可进化性和复杂性之间的关系。
tags:
  - ACL 2025
  - 知识神经元
  - 退化知识神经元
  - 持久同调
  - 事实知识存储
  - 鲁棒性
---

# Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2402.13731](https://arxiv.org/abs/2402.13731)  
**代码**: 即将公开  
**领域**: 可解释性 (Interpretability)  
**关键词**: 知识神经元, 退化知识神经元, 持久同调, 事实知识存储, 鲁棒性  

## 一句话总结

> 本文从结构和功能两个角度全面定义了退化知识神经元 (DKN)，提出基于持久同调的神经拓扑聚类方法 (NTC) 获取 DKN，并通过 34 个实验揭示了 DKN 与 LLM 鲁棒性、可进化性和复杂性之间的关系。

## 研究背景与动机

**研究问题：** 大语言模型存储了大量事实知识，但其底层存储机制仍不清楚。先前研究发现 MLP 权重中存在知识神经元 (KN)，且部分 KN 表现出退化现象 (degeneracy)，即不同 KN 子集可独立表达同一事实。然而，退化知识神经元 (DKN) 尚未被严格定义或系统研究。

**现有方法的不足：**
- **数量限制：** 先前工作将 DKN 中每个元素限制为仅包含两个 KN，但事实知识可能需要更多神经元表示
- **连接忽视：** 先前工作仅考虑神经元本身，忽略了神经元之间的连接权重，而知识表达需要多个神经元的交互

**核心动机：** 受认知科学中退化性研究的启发，作者希望从结构与功能两个维度全面定义 DKN，并探索 DKN 与 LLM 三大性质（鲁棒性、可进化性、复杂性）之间的关系。

## 方法详解

### 整体框架

本文首先给出 DKN 的完整定义，然后提出 NTC 方法获取 DKN，最后通过三组实验验证 DKN 与 LLM 性质的关系：

1. **DKN 定义** → 从功能和结构两方面定义
2. **NTC 方法** → 基于持久同调的聚类 + 过滤
3. **性质探索** → 鲁棒性、可进化性、复杂性

### 关键设计

**1. DKN 的功能定义：** 定义基础退化组件 (BDC) 为能独立表达同一事实的 KN 子集。DKN 是所有 BDC 的集合，满足：激活任意单个 BDC 即可表达事实 ($Prob(\mathcal{D}) \approx Prob(\mathcal{B}_i)$)，而抑制所有 BDC 则无法表达事实 ($Prob(\emptyset) \ll Prob(\mathcal{B}_i)$)。

**2. DKN 的结构定义：** 基于连接权重定义神经元距离，对相邻层神经元取权重的倒数，对跨多层神经元用动态规划求最短路径，同层神经元距离设为无穷大（因 PLM 信息在层间传递而非层内）。

**3. NTC 方法（核心创新）：**
- **聚类阶段：** 使用持久同调 (Persistent Homology) 捕捉 KN 集合的持续时间和连接紧密度。随着距离阈值 $R$ 从 0 递增，记录所有 BDC 及其持续时间 $R_p$
- **过滤阶段：** 选择 $R_p > \tau_1$ 且 $Prob(\mathcal{B}_i) > \tau_2$ 的 BDC，构成最终 DKN

### 损失函数/优化目标

本文不涉及模型训练，而是通过抑制/增强实验验证 DKN 的性质。关键指标为预测概率下降率：$\Delta Prob(\%) = \frac{Prob_b - Prob_a}{Prob_b}$

## 实验

### 主实验结果

| 方法 | GPT-2 平均 (部分→全部抑制) | LLaMA2 平均 (部分→全部抑制) |
|------|---|---|
| DBSCAN | 23.90→34.72 | 22.26→18.24 |
| Hierarchical | 20.78→30.81 | 27.50→44.04 |
| K-Means | 34.42→38.86 | 20.08→39.24 |
| AMIG | -0.8→0.9 | -1.0→2.0 |
| **NTC (Ours)** | **9.32→55.60** | **14.11→28.78** |

NTC 方法在部分抑制 BDC 时概率下降最小，全部抑制时下降最大，差距最显著，表明获取的 DKN 退化性最好。

### 消融实验

| 实验设置 | 关键发现 |
|------|------|
| 事实检查 (DKNs vs KNs vs PLMs) | DKN 的 F1 分数在 Golden 设置下达 0.667/0.682，优于 KN (0.618/0.643) 和 PLM (0.015/0.036) |
| 查询扰动-抑制 | 抑制 DKN 导致预测概率显著下降，损害鲁棒性 |
| 查询扰动-增强 | 增强 DKN 可恢复被干扰查询的正确回答 |
| 可进化性-微调重叠 | 微调后参数变化区域与 DKN 高度重叠 |
| 可进化性-高效微调 | 仅微调 DKN 即可高效学习新知识且不遗忘旧知识 |

### 关键发现

1. **退化性验证：** NTC 获取的 DKN 最符合退化性定义，部分抑制 BDC 不影响知识表达但全部抑制则完全破坏
2. **抑制增强补偿：** 抑制部分 DKN 可能反而增强知识表达（概率负值），说明其他 BDC 可补偿
3. **DKN 与鲁棒性正相关：** 抑制/增强 DKN 会损害/改善 PLM 对输入干扰的鲁棒性
4. **DKN 与可进化性正相关：** PLM 利用 DKN 区域学习新知识，冻结非 DKN 参数仍可高效学习
5. **退化性与复杂性正相关：** 更大规模的 PLM 表现出更强的退化性

## 亮点

- 首次从结构和功能两方面全面定义 DKN，开创了 PLM 事实知识存储单元的结构研究
- 基于持久同调的 NTC 方法允许任意数量和结构的 BDC 形成，克服了先前方法的数量限制
- 通过 34 个实验系统揭示了 DKN 与鲁棒性、可进化性、复杂性的关系，具有认知科学启发
- 提出了基于 DKN 的事实检查方法，仅使用神经元激活信息即可判断事实真伪

## 局限性

- 仅在 GPT-2 和 LLaMA2-7B 上验证，缺乏对更大规模 LLM 的实验
- NTC 方法依赖多个超参数 ($\tau_1$, $\tau_2$)，选择方式未充分讨论
- 退化性和复杂性的正相关关系需要在更多模型上验证
- 事实检查实验基于关系级 DKN 聚合，可能存在 DKN 与具体查询不匹配的问题

## 相关工作

- **知识神经元：** Dai et al. (2022) 首次提出知识神经元概念；Chen et al. (2024a) 发现 DKN 现象但定义不完善
- **知识编辑：** Meng et al. (2022) 的 ROME 方法利用 MLP 层间信息传递编辑知识
- **认知科学启发：** Edelman & Gally (2001) 提出生物系统中退化性与鲁棒性、可进化性的关系
- **持久同调：** Edelsbrunner et al. (2008) 的拓扑数据分析方法被创新性地应用于神经网络分析

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ |
| 综合评价 | 8.5/10 |

<!-- RELATED:START -->

## 相关论文

- [Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)
- [Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)
- [Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis](../../ICML2025/interpretability/supernova_event_dataset_interpreting_large_language_models_personality_through_c.md)
- [Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)

<!-- RELATED:END -->
