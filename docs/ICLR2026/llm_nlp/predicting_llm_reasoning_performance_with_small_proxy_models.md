---
title: >-
  [论文解读] Predicting LLM Reasoning Performance with Small Proxy Models
description: >-
  [ICLR 2026][LLM/NLP][小模型代理] 提出 rBridge 方法，通过结合前沿模型推理轨迹 (reasoning trace) 的 NLL 评估与 token 级任务对齐权重，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，数据排序计算成本降低 100 倍以上。
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "小模型代理"
  - "推理能力预测"
  - "预训练数据"
  - "scaling law"
  - "NLL"
---

# Predicting LLM Reasoning Performance with Small Proxy Models

**会议**: ICLR 2026  
**arXiv**: [2509.21013](https://arxiv.org/abs/2509.21013)  
**代码**: 未开源  
**领域**: LLM/NLP  
**关键词**: 小模型代理, 推理能力预测, 预训练数据, scaling law, NLL

## 一句话总结

提出 rBridge 方法，通过结合前沿模型推理轨迹 (reasoning trace) 的 NLL 评估与 token 级任务对齐权重，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，数据排序计算成本降低 100 倍以上。

## 研究背景与动机

### 预训练中的代理模型需求

预训练大语言模型的成本极高（7B 模型 500B tokens 训练成本超 5 万美元），因此用小模型作为代理 (proxy) 来优化数据集和预测大模型表现是关键研究方向。现有方法（scaling law、数据排名不变性）在非推理任务上有效，但**推理能力的涌现特性**使小模型代理面临根本挑战。

### 核心挑战：推理的涌现性

推理能力展现出明显的涌现行为——仅在 7B+ 模型中可靠出现：

- 1B 模型在 MATH500 上噪声大且斜率方向错误（$R^2$ 极低）
- 相比之下，TriviaQA、HellaSwag 等非推理任务在小规模即有平滑改善
- 这使得从业者被迫使用 7-15B 的 "代理模型"，成本高昂

### 问题定义

设 $\pi^{\text{p}}, \pi^{\text{t}}$ 分别为小代理和大目标模型。目标是设计代理评估指标 $\text{metric}^{\text{p}}$ 使得：

$$\text{metric}^{\text{p}} := \max_{\text{metric}} \text{corr}(\text{metric}(\pi^{\text{p}}), \text{metric}^{\text{t}}(\pi^{\text{t}}))$$

即小模型指标的改善应可靠地预测大模型指标的改善。

## 方法详解

### 整体框架

rBridge 的核心洞察是现有方法在两个轴上失败：**评估目标对齐**和**目标任务对齐**。

### 先导分析：现有方法的局限

**(1) 评估目标不对齐**

- **Acc./Pass@K 与预训练目标不匹配**：小预训练模型缺乏强泛化能力，Accuracy 在 1B 模型上噪声极大
- **NLL 的分布对齐问题**：不是所有 NLL 都等价。若 gold label $Y^*$ 是 OOD 的，NLL 信号也会噪声化。ScalingBench 的 $Y^*$ 包含 "\\n"、"Final Answer:" 等格式文本，对小模型是 OOD 的

**(2) 目标任务不对齐**

标准 NLL 对所有 token 等权处理，无法区分任务关键 token（如 "sum modulo 9"）和格式 token（如换行、编号）。

### 关键设计：rBridge NLL

**步骤 1：使用前沿模型推理轨迹 $R^\phi$ 作为 gold label**

仅使用前沿模型 $\pi^\phi$ 的推理轨迹（不含最终答案格式），因为：
- $R^\phi$ 更接近预训练分布（ID），平均 NLL 下降 74.7%
- $R^\phi$ 是通向正确答案的推理过程，天然任务对齐

**步骤 2：token 级任务对齐权重**

$$\text{rBridge NLL}(\text{token}_i) := \underbrace{-\text{log}(p^{\text{p}}(\text{token}_i))}_{\text{标准 NLL}} \cdot \underbrace{\frac{1}{|\text{token}_i|} \sum_{\text{letter} \in \text{token}_i} p^\phi(\text{letter})}_{\text{自动 tokenizer 无关任务对齐权重}}$$

每个 token 的 NLL 按前沿模型对该 token 的置信度加权。为处理不同 tokenizer，在字母级别计算权重后在 token 内取平均。最后对权重因子做 MinMax 归一化以放大效果。

### 损失函数

rBridge 不训练新模型，而是一个评估指标。其计算流程：

1. 用前沿模型生成推理轨迹 $R^\phi$
2. 用小代理模型计算每个 token 的 NLL
3. 用前沿模型的 token 概率计算任务对齐权重
4. 加权求和得到 rBridge score

## 实验关键数据

### 主实验 1：数据集排名（<100M → 1.2B）

在 DataDecide 协议下，用代理模型排列 25 个预训练数据集：

| 方法 | 最佳 DAcc. | 计算节省 |
|------|-----------|---------|
| CF Accuracy（最大代理） | ~基线 | 1× |
| Norm Correct Prob | ~50%（随机水平） | - |
| **rBridge (3.7M 模型)** | 比基线高 27% | **100.2×-733.4×** |

rBridge 在最小代理规模（3.7M，训练 87.3M tokens）即达到远超基线的排名准确率。

### 主实验 2：代理-目标关系（1B → 13B/32B）

6 个推理基准上的 5-fold 交叉验证结果：

| 方法 | 1B→13B Avg Train $R^2$ ↑ | 1B→13B Avg Test MAE ↓ | 1B→32B Avg $R^2$ ↑ |
|------|--------------------------|----------------------|---------------------|
| Acc./p@1 | 0.304 | 3.709 | 0.312 |
| iSFT | 0.290 | 5.123 | 0.349 |
| TED | 0.375 | 3.377 | 0.352 |
| MPCA | 0.194 | 302.642 | 0.205 |
| NLL | 0.485 | 5.173 | 0.488 |
| $R^\phi$ | 0.867 | 1.455 | 0.820 |
| **rBridge** | **0.874** | **1.384** | **0.826** |

rBridge 在 10/12 个指标上取得最优，平均 $R^2$ 从基线 0.304 提升到 0.874。

### 消融实验

| 设置 | 1B→13B $R^2$ | 1B→13B+SFT $R^2$ | 1B→32B $R^2$ |
|------|-------------|-------------------|-------------|
| 标准 NLL | 0.485 | 0.413 | 0.488 |
| NLL on $R^\phi$ | 0.867 | 0.820 | 0.820 |
| **rBridge (+ 权重)** | **0.874** | **0.846** | **0.826** |

每个组件（推理轨迹、权重、归一化）都带来一致提升。

### 零样本迁移（1B → 7B 跨数据集）

在一个数据集上拟合的 rBridge-目标函数可零样本迁移到另一数据集：

| 方法 | 数据集排名正确率 | 平均 MAE |
|------|----------------|---------|
| $R^\phi$ | 4/5 | 3.425 |
| **rBridge** | **5/5** | **2.490** |

### 关键发现

1. **rBridge 性能超越 7-13 倍大的代理模型**：1B rBridge 优于 7B/13B 模型直接用 Acc. 评估
2. **计算节省巨大**：数据排名任务中节省 100-733 倍 FLOPs
3. **跨数据集零样本迁移有效**：一次拟合可复用
4. **非连续指标（Acc./p@k、iSFT）表现最差**：验证了评估目标对齐的重要性

## 亮点与洞察

1. **深刻的方法论洞察**：问题不在于小模型 "不够大"，而在于评估方式与预训练目标和任务不对齐
2. **优雅的解决方案**：不训练新模型，仅改变评估指标即可达到数量级的改善
3. **前沿模型概率的巧妙使用**：将前沿模型的 token 概率作为任务重要性的自动度量
4. **处理 tokenizer 不匹配**：字母级概率平均的设计简洁实用
5. **理论动机清晰**：从预训练目标对齐和 ID/OOD 分析出发，每一步设计都有理论支撑

## 局限性

1. 需要前沿模型生成推理轨迹，虽然成本低（每基准 <$10）但引入了对前沿模型的依赖
2. 实验中最大目标规模为 32B，更大模型（70B+）是否成立未验证
3. 零样本迁移仅在 2 个数据集间验证，泛化性有待进一步确认
4. 方法专注推理任务，对非推理任务（知识、理解）的应用未探索

## 相关工作与启发

- **Scaling Laws (Kaplan et al. 2020)**：rBridge 可视为推理领域的改进版 scaling law
- **ScalingBench**：rBridge 是 ScalingBench 的改进版，关键改进是去掉答案格式只用推理轨迹
- **DataDecide (Magnusson et al. 2025)**：rBridge 在其协议上验证了数据排名能力
- **启发**：评估指标的设计比模型规模更重要——对齐评估目标和任务是利用小模型的关键

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统解决小模型预测大模型推理性能的问题
- **理论深度**: ⭐⭐⭐⭐ — ID/OOD 分析和评估对齐框架完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三类实验（排名/关系/迁移）覆盖 6 个基准，对比 6+ 基线
- **实用价值**: ⭐⭐⭐⭐⭐ — 100× 计算节省的实际意义巨大
- **总评**: ⭐⭐⭐⭐☆ — 问题重要、方法优雅、实验全面，对预训练数据优化有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](../../ACL2026/llm_nlp/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](../../ACL2025/llm_nlp/collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](../../ACL2026/llm_nlp/cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)

</div>

<!-- RELATED:END -->
