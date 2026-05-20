---
title: >-
  [论文解读] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis
description: >-
  [ACL 2025][时间序列][电子健康记录] 提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。
tags:
  - "ACL 2025"
  - "时间序列"
  - "电子健康记录"
  - "跨模态时序模式"
  - "注意力机制"
  - "对比学习"
  - "临床预测"
---

# CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis

**会议**: ACL 2025  
**arXiv**: [2411.00696](https://arxiv.org/abs/2411.00696)  
**代码**: [https://github.com/HKU-MedAI/CTPD](https://github.com/HKU-MedAI/CTPD)  
**领域**: 医学信息 / 时间序列  
**关键词**: 电子健康记录, 跨模态时序模式, Slot Attention, 对比学习, 临床预测

## 一句话总结
提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。

## 研究背景与动机

**领域现状**：EHR 数据天然是多模态时序数据——包含结构化的生命体征/检验结果（不规则时间序列）和非结构化的临床笔记。多模态融合已被证明有助于临床结果预测。

**现有痛点**：(1) 现有方法主要关注单样本内的时序交互和多模态融合，忽视了跨患者的高层时序模式（如异常心率趋势、呼吸功能逐步改善等）；(2) 这些模式在时间序列和临床笔记中有语义对应关系，但缺少有效的跨模态模式发现方法；(3) 关键时序模式在不同时间尺度上展开（急性变化 vs 长期趋势），现有方法难以同时捕获。

**核心矛盾**：timestamp-level 的嵌入丢失了高层时序语义——需要一种机制从数据中"发现"具有临床意义的时序模式。

**本文目标**：如何从多模态 EHR 中发现并对齐跨模态时序模式？如何利用这些模式改进临床预测？

**切入角度**：借鉴计算机视觉中 object-centric learning（Slot Attention）的思想，将"时序原型"类比为"对象槽"，从数据中迭代提取有意义的时序模式。

**核心 idea**：共享初始化的可学习时序原型 + Slot Attention 迭代精炼 + TP-NCE 跨模态对齐 + 多尺度时序编码。

## 方法详解

### 整体框架
输入：不规则多变量时间序列（MITS）+ 临床笔记序列 → 分别编码为规则时间嵌入 → 多尺度时间序列提取 → 共享原型向量通过 Slot Attention 分别聚合两模态 → TP-NCE 对齐 + 重建损失 → Transformer 融合原型+timestamp 嵌入 → 分类预测。

### 关键设计

1. **多模态 EHR 编码**:

    - 功能：将不规则时间序列和临床笔记编码为统一的时间对齐嵌入
    - 核心思路：时间序列用门控机制融合插值嵌入和 mTAND 嵌入（$z^{TS} = g \odot e_{imp}^{TS} + (1-g) \odot e_{mTAND}^{TS}$）；临床笔记用 BERT-Tiny 编码 [CLS] token，再通过 mTAND 做时间对齐
    - 设计动机：门控机制让模型自适应选择插值和注意力两种时序表示的优点

2. **跨模态时序模式发现（核心）**:

    - 功能：用 Slot Attention 从多尺度时间嵌入中提取 K 个跨模态时序原型
    - 核心思路：
        - **多尺度**：对时间序列嵌入用 3 个卷积块+mean pooling（stride=2）生成 3 个尺度（T, T/2, T/4），拼接得到 $z_{MS}^{TS} \in \mathbb{R}^{T' \times D}$
        - **共享初始原型**：$P^{Shared} \in \mathbb{R}^{K \times D}$，从 $\mathcal{N}(\mu, \text{diag}(\sigma))$ 初始化，训练中优化
        - **Slot Attention 迭代**：原型作为 query，模态嵌入作为 key/value，计算注意力权重 → 加权聚合更新 → GRU 精炼原型。迭代 3 次
    - 设计动机：Slot Attention 能自动将数据分组到不同"槽"中，每个槽对应一个时序模式。多尺度确保捕获不同时间粒度的模式

3. **TP-NCE 对比损失**:

    - 功能：对齐两模态发现的时序原型
    - 核心思路：InfoNCE 变体，对 batch B 中的样本，同一 ICU stay 的 $P^{TS}$ 和 $P^{Text}$ 为正对，不同 stay 为负对。相似度为 K 个原型的加权 cosine 之和：$\text{sim}(\cdot) = \sum_{k=1}^K \beta_k \langle P^{TS}(k), P^{Text}(k) \rangle$，权重 $\beta$ 由全局嵌入 MLP 生成
    - 设计动机：确保两模态的时序原型具有一致的语义，同时加权机制允许不同原型有不同重要度

4. **辅助重建损失**:

    - 功能：从 $P^{TS}$ 重建时间序列，从 $P^{Text}$ 重建文本嵌入
    - 核心思路：两个 Transformer decoder 分别做重建，MSE 损失
    - 设计动机：确保原型保留了核心模态信息，防止对比学习导致信息丢失

### 损失函数 / 训练策略
- $\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{TPNCE} + \lambda_2 \mathcal{L}_{Recon}$
- 分类损失：交叉熵；batch size=128，lr=4e-5，Adam，cosine annealing，early stopping
- 单张 RTX-3090，约 1 小时/run

## 实验关键数据

### 主实验（MIMIC-III）

| 方法 | 48-IHM AUROC↑ | 48-IHM F1↑ | 24-PHE AUROC↑ | 24-PHE AUPR↑ |
|------|-------------|-----------|-------------|-------------|
| UTDE (时序 SOTA) | 86.14 | 49.29 | 73.62 | 36.80 |
| mTAND (笔记 SOTA) | 85.40 | 35.76 | 82.14 | 54.57 |
| MMTM (多模态 SOTA) | 87.88 | 51.54 | 81.46 | 51.88 |
| DAFT | 87.53 | 51.95 | 81.18 | 50.91 |
| **CTPD (Ours)** | **88.15** | **53.85** | **83.34** | **56.39** |

### 消融实验

| 配置 | 48-IHM AUROC | 48-IHM F1 | 24-PHE AUROC |
|------|-------------|-----------|-------------|
| CTPD Full | 88.15 | 53.85 | 83.34 |
| w/o 原型 | 86.89 | 48.47 | 82.24 |
| w/o timestamp 嵌入 | 87.18 | 45.85 | 82.41 |
| w/o 多尺度 | 87.59 | 49.74 | 83.11 |

### 关键发现
- **跨模态时序原型对 F1 贡献最大**：去掉原型后 48-IHM F1 从 53.85 降到 48.47（-5.38 绝对值），说明发现的时序模式对预测极为重要
- **多模态融合优于单模态**：纯时间序列最好 AUROC 86.14，纯笔记 82.14，多模态 CTPD 88.15，说明两种数据互补
- **统计显著性强**：6 个指标中 5 个 p<0.05（其中 4 个 p<0.001），24-PHE 的提升尤其显著
- **多尺度的贡献**：去掉多尺度后 F1 从 53.85 降到 49.74，说明不同时间粒度的模式确实互补

## 亮点与洞察
- **Slot Attention 用于时序模式发现**：将 CV 中物体分割的 Slot Attention 巧妙迁移到时序数据，每个"槽"自动学习一个时序模式。这个思路可推广到任何需要从序列中提取高层概念的任务
- **共享初始化原型确保跨模态对应**：两模态从相同的初始原型出发分别精炼，再用对比学习对齐，保证了模式的语义一致性
- **多尺度时序编码**：用简单的卷积+池化生成多尺度，不增加太多计算量但有效捕获急性变化和长期趋势

## 局限与展望
- 仅在 MIMIC-III/IV 上验证，ICU 数据可能不代表门诊/社区场景
- 要求配对的时间序列+临床笔记，缺失模态场景未处理
- BERT-Tiny 作为文本编码器能力有限，更强的 LLM（如临床 LLaMA）可能提升文本表示
- 原型数量 K 是超参数，最优值可能因任务而异
- 未分析学出的时序原型的可解释性（如哪些原型对应哪些临床事件）

## 相关工作与启发
- **vs UTDE**: UTDE 也做不规则时间序列编码（门控 mTAND），CTPD 在此基础上加了跨模态原型发现，AUROC 从 86.14 提升到 88.15
- **vs DrFuse**: DrFuse 用 Transformer 做跨模态 token 交互，但缺少高层时序模式发现，所有指标都低于 CTPD
- **vs Slot Attention（Locatello 2020）**: 原始 Slot Attention 发现图像中的物体，CTPD 将其推广到时序+文本的跨模态场景

## 评分
- 新颖性: ⭐⭐⭐⭐ Slot Attention→时序原型发现的迁移很有新意，TP-NCE 设计合理
- 实验充分度: ⭐⭐⭐⭐ 消融全面，统计检验严谨，但数据集有限
- 写作质量: ⭐⭐⭐⭐ 公式化清晰，Fig.1 的临床动机很直观
- 价值: ⭐⭐⭐⭐ 对临床 EHR 分析提供了新的模式发现范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ICML 2025\] Causal Discovery from Conditionally Stationary Time Series](../../ICML2025/time_series/causal_discovery_from_conditionally_stationary_time_series.md)
- [\[ACL 2025\] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)
- [\[ICML 2025\] Risk and Cross Validation in Ridge Regression with Correlated Samples](../../ICML2025/time_series/risk_and_cross_validation_in_ridge_regression_with_correlated_samples.md)

</div>

<!-- RELATED:END -->
