---
title: >-
  [论文解读] DataDecide: How to Predict Best Pretraining Data with Small Experiments
description: >-
  [模型压缩] > 本文构建了 DataDecide——迄今最大规模的开放模型套件（25 种数据配方 × 14 种模型规模 × 3 个随机种子），系统研究如何用小规模实验预测最佳预训练数据，发现单一小规模排名（如 150M 参数）即可达到约 80% 的成对决策准确率，且连续似然代理指标仅需目标计算量 0.01% 即可让多个基准任务的预测准确率超过 80%。
tags:
  - 模型压缩
---

# DataDecide: How to Predict Best Pretraining Data with Small Experiments

| 属性 | 内容 |
|------|------|
| **会议** | ICML 2025 |
| **arXiv** | [2504.11393](https://arxiv.org/abs/2504.11393) |
| **代码** | [HuggingFace DataDecide Collection](https://huggingface.co/collections/datadecide) |
| **领域** | 数据选择 |
| **关键词** | 预训练数据选择, Scaling Laws, 小规模实验, 决策准确率, 代理指标 |

## 一句话总结

> 本文构建了 DataDecide——迄今最大规模的开放模型套件（25 种数据配方 × 14 种模型规模 × 3 个随机种子），系统研究如何用小规模实验预测最佳预训练数据，发现单一小规模排名（如 150M 参数）即可达到约 80% 的成对决策准确率，且连续似然代理指标仅需目标计算量 0.01% 即可让多个基准任务的预测准确率超过 80%。

## 研究背景与动机

### 核心问题

大语言模型（LLM）预训练成本极高，在不同数据集上进行全规模训练来选择最佳数据不现实。实践中通常依赖小规模实验做数据决策，但关键问题是：**哪些基准测试和决策方法能最准确地从小规模实验预测出大模型的最优数据？**

### 现有方法的不足

**缺乏反事实验证**：现有方法仅通过产出单个"表现出色"的大模型来间接验证决策正确性，无法观察到如果选择其他数据会怎样

**数据配方覆盖不足**：先前工作最多只考虑 2 种（Pythia）或 6 种（Paloma、Brandfonbrener 等）数据配方，不足以系统评估决策方法

**Scaling Law 验证不充分**：已有 scaling law 方法虽然能降低预测误差，但缺少从预测误差到实际决策准确率的转换评估

**评估指标与规模不匹配**：离散的准确率指标在小规模模型上表现不稳定，可能影响预测质量

### 本文动机

为使"数据选择决策"的研究可量化、可复现，作者构建了 DataDecide 模型套件——在 25 种数据配方上进行受控预训练实验，规模覆盖 4M 到 1B 参数，每个配置 3 个随机种子，共计超过 30K 个模型检查点，全部开源。

## 方法详解

### 整体框架

DataDecide 的核心思路可概括为三步：

1. **构建大规模受控模型套件**：在固定架构、优化器和超参数的条件下，仅改变数据配方和模型规模
2. **定义预测任务**：对于每对数据配方 $(A, B)$，预测哪个在目标规模（1B）上表现更好
3. **度量预测准确率**：用 Decision Accuracy 衡量预测方法在所有配方对上的正确率

### DataDecide 模型套件

- **数据配方**：25 种，涵盖 Dolma 1.7、DCLM-Baseline、FineWeb、Falcon/RefinedWeb、C4 等主流开源语料，以及不同的去重、过滤、领域消融和混合策略
- **模型规模**：14 种，从 4M（3.7M 参数）到 1B（1176.8M 参数），使用 OLMo 的 model ladder 框架自动配置超参数
- **token-参数比**：固定为 100（即 $5 \times$ Chinchilla 最优比例），反映当前主流的 overtrain 策略
- **随机种子**：每个配置 3 个种子；1B 模型所有种子均完整训练，其他规模在第 2、3 个种子上训练 25% 计算量后提前终止
- **总模型数量**：$25 \times 14 \times 3 = 1{,}050$ 个模型

### 预测方法

#### 方法一：Single Scale Ranking（单一规模排名）

在某个固定小规模上（如 150M 参数）训练所有 25 种数据配方的模型，直接用小模型的下游性能排名作为大模型的预测排名。

$$\hat{y}_A > \hat{y}_B \iff \text{Acc}_{\text{small}}(A) > \text{Acc}_{\text{small}}(B)$$

#### 方法二：Multi Scale Extrapolation（多规模 Scaling Law 外推）

在多个小规模上训练模型，拟合 scaling law 曲线，外推到目标规模的性能。采用 Bhagia et al. (2024) 的两步法：

**Step 1 - 计算量到 Loss 的映射：**

$$L(C) = \frac{A}{C^{\alpha}} + E$$

其中 $A, \alpha, E$ 为待优化参数，$C = 6ND$（理论 FLOPs）。

**Step 2 - Loss 到准确率的映射：**

$$\text{Acc}(L) = \frac{a}{1 + e^{-k(L - L_0)}} + b$$

其中 $a, b, k, L_0$ 为待优化参数。共测试了 8 种 scaling law 变体（2 参数、3 参数、5 参数、单步拟合、加 helper point、过滤早期 checkpoint 等）。

### 评估体系

#### Decision Accuracy（决策准确率）

核心评估指标。对于所有数据配方对 $(A, B)$，测量预测是否正确识别目标规模上的赢家：

$$\text{Decision Accuracy} = \frac{1}{|\mathcal{P}|} \sum_{(A,B) \in \mathcal{P}} \mathbb{I}\big(\text{sign}(\hat{y}_A - \hat{y}_B) = \text{sign}(y_A - y_B)\big)$$

目标规模的"金标准"排名基于 3 个随机种子的平均性能。

#### 计算预算比例 (%C)

$$\%C = \frac{c}{C} \times 100\%$$

其中 $c$ 为预测实验的 FLOPs，$C$ 为目标规模的 FLOPs。

#### 下游评估

使用 OLMES 框架的 10 个多选题基准：MMLU、HellaSwag、ARC Challenge、ARC Easy、PIQA、CommonsenseQA、SocialIQA、OpenBookQA、BoolQ、WinoGrande。

### 代理指标（Proxy Metrics）

为改善小规模下离散准确率不稳定的问题，引入 5 种连续代理指标：

| 指标 | 定义 |
|------|------|
| **Correct Prob** | 正确选项的平均概率 |
| **Margin** | 正确选项与最高错误选项概率之差 |
| **Norm Correct Prob** | 正确选项概率占所有选项概率之和的比例 |
| **Total Prob** | 所有选项（含正确和错误）的概率总和 |
| **Accuracy** | 正确选项概率最高的比例（离散） |

每种指标均有 per_token 和 per_char 两种长度归一化变体，实验发现 per_char 在多数任务上表现最优。

## 实验与关键数据

### 主实验：计算预算 vs 决策准确率

**核心发现**（Figure 1）：

- 在 OLMES 10 任务聚合指标上，计算预算与决策准确率呈近似对数线性关系
- 150M 参数的小模型即可达到约 **80%** 的成对决策准确率
- 使用连续似然指标仅需目标 **0.01%** 的计算量即可让 MMLU、ARC、HellaSwag、MBPP、HumanEval 等基准达到 >80% 的决策准确率

### 各任务预测难度差异（Figure 2）

| 任务 | 预测难度 | 特征 |
|------|----------|------|
| ARC Easy | 极易预测 | 少量计算即可达到高决策准确率 |
| MMLU | 较易预测 | 运行间方差低 |
| ARC Challenge | 中等 | 数据配方间性能分布广 |
| HellaSwag | 较难 | 需要更多计算才开始有效预测 |
| SocialIQA, WinoGrande | 难 | 存在明显的不敏感期，之后才呈对数线性增长 |
| BoolQ | 极难 | 仅在接近目标计算量的中间检查点才有非平凡的决策准确率 |

### Scaling Law 对比（Figure 3）

**关键结论**：8 种 scaling law 变体均未能超越 single-scale ranking 的计算-决策准确率前沿。

| Scaling Law 变体 | 相对误差 | 绝对误差 |
|------------------|----------|----------|
| 3 参数 + helpers + >50% checkpoints | 5.6 | 2.6 |
| 3 参数 + helpers | 6.0 | 2.8 |
| 3 参数 | 6.5 | 3.1 |
| 2 参数 | 6.5 | 3.2 |
| 5 参数单步 | 42.8 | 17.4 |
| 5 参数 | 230.8 | 65.4 |

2 参数和 3 参数变体预测误差相近且最优，5 参数变体由于过拟合而显著恶化。

### 代理指标实验（Figure 4）

**关键发现**：

- **Correct Prob 和 Total Prob** 在小规模下（0.01%~1% 目标计算量）提供最佳或等价的决策准确率
- 在 MMLU、ARC Easy、PIQA 等 5 个任务上，连续代理指标在小规模时显著优于离散 Accuracy
- 接近目标规模时（最后一个数量级），Accuracy 和惩罚错误答案的连续指标（Norm Correct Prob、Margin）反超 Correct Prob 和 Total Prob
- **代码任务突破**（Figure 6）：HumanEval 和 MBPP 从使用 Accuracy 时几乎不可预测，切换到 Correct Prob 后决策准确率上升至约 80%

### 可预测性的两个关键因素（Figure 5）

决策准确率取决于：
1. **运行间方差（噪声）**：3 个随机种子间性能的标准差越低，预测越准
2. **配方间性能分布（spread）**：不同数据配方之间性能差异越大，越容易区分

例如：MMLU 之所以好预测，主要因为运行间方差低；ARC Easy 之所以好预测，因为不同数据配方的性能分布很广。Correct Prob 代理指标的优势往往体现在改善了这两个特征中的至少一个。

## 亮点与洞察

1. **反直觉发现**：简单的单一规模排名即可达到约 80% 的决策准确率，复杂的 scaling law 外推方法不仅计算成本更高，而且并未带来显著改善。这挑战了"越复杂越好"的直觉
2. **代理指标的重要价值**：连续似然指标可以将某些原本在小规模下不可预测的任务（如代码生成）变为可预测，核心原因是它们降低了评估噪声或增大了配方间的性能差异
3. **实用建议明确**：(a) 选择易预测的基准（如 MMLU、ARC）进行小规模数据决策；(b) 使用 Correct Prob 作为代理指标；(c) 150M 参数模型是性价比最高的预测规模
4. **开放生态构建**：释放 30K+ 个检查点、25 种数据配方的完整模型、全部评估结果，社区可以零成本运行新评估、尝试新预测方法
5. **Decision Accuracy 指标新颖**：不同于传统 scaling law 工作关注预测误差的绝对大小，DataDecide 直接度量"预测是否能正确做出二选一决策"，更贴合实际应用需求

## 局限性

1. **固定 token-参数比**：仅测试了 100:1（5× Chinchilla optimal）这一比例，虽然符合当前主流但不能覆盖所有场景
2. **目标规模有限**：最大目标规模为 1B 参数，无法直接验证发现是否推广到更大模型（如 7B、70B）
3. **模型架构单一**：固定使用 OLMo 架构，未测试架构变化对数据选择决策的影响
4. **评估任务有限**：仅使用 10 个 OLMES 多选题基准；虽然论文展示了代码任务的可行性，但对数学推理任务的扩展性仍存疑（数学任务即使换代理指标仍然难以预测）
5. **Scaling law 实现可能非最优**：8 种 scaling law 变体都是直接基准线实现，更精心设计的拟合方法可能表现更好

## 相关工作

- **Scaling Law 预测**：Kaplan et al. (2020)、Hoffmann et al. (2022) 奠定了 LM loss 随规模变化的 scaling law 基础；Gadre et al. (2024)、Bhagia et al. (2024) 将其扩展到下游性能预测；Choshen et al. (2024) 提供了 scaling law 估计的实用指南
- **数据选择套件**：Pythia (Biderman et al., 2023) 首次提供 2 种数据配方的受控比较；Paloma (Magnusson et al., 2024) 扩展到 6 种；DCLM (Li et al., 2024) 大量使用单一规模排名但未开源消融实验模型
- **数据混合优化**：Kang et al. (2024)、Ye et al. (2024) 通过 scaling law 优化数据源混合比例，是 DataDecide 工作方向的特殊情形
- **涌现能力与指标选择**：Schaeffer et al. (2023) 指出离散指标可能导致"涌现幻觉"，连续指标更能反映渐进提升——本文的代理指标发现与此呼应

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 严谨性 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 清晰度 | ⭐⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐** |

> 本文是一项扎实的实验型贡献，通过大规模受控实验回答了一个非常实用的问题。实验设计严谨，结论清晰且可操作，开源资源丰富。虽然目标规模局限于 1B 和架构/比例的单一性略显不足，但其方法论框架可直接推广，对工业界和学术界的预训练数据选择实践都有重要参考价值。

<!-- RELATED:START -->

## 相关论文

- [Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)

<!-- RELATED:END -->
