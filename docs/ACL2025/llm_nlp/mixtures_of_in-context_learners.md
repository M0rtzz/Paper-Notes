---
title: >-
  [论文解读] Mixtures of In-Context Learners
description: >-
  [ACL2025][LLM/NLP][in-context learning] 提出 MoICL 方法，将 ICL 的 demonstration 集合划分为多个子集（专家），通过可学习的权重函数融合各专家的 next-token 分布，在不修改 LLM 参数的前提下显著提升 ICL 的准确率、鲁棒性和效率。
tags:
  - ACL2025
  - LLM/NLP
  - in-context learning
  - mixture of experts
  - demonstration selection
  - product of experts
  - robustness
---

# Mixtures of In-Context Learners

**会议**: ACL2025  
**arXiv**: [2411.02830](https://arxiv.org/abs/2411.02830)  
**代码**: -  
**领域**: LLM/NLP  
**关键词**: in-context learning, mixture of experts, demonstration selection, product of experts, robustness  

## 一句话总结

提出 MoICL 方法，将 ICL 的 demonstration 集合划分为多个子集（专家），通过可学习的权重函数融合各专家的 next-token 分布，在不修改 LLM 参数的前提下显著提升 ICL 的准确率、鲁棒性和效率。

## 研究背景与动机

In-context learning（ICL）是当前 NLP 中的变革性技术，仅通过在 LLM 的上下文中提供示例（demonstrations）就可以完成各种任务，无需微调模型参数。然而，ICL 存在几个关键限制：

**上下文长度受限**：Transformer 的最大上下文长度限制了可使用的 demonstration 数量，而更多的 demonstration 通常能带来更好的性能。

**二次复杂度**：随着 demonstration 数量增加，self-attention 的计算和内存开销呈二次增长。

**对 demonstration 选择敏感**：ICL 的效果高度依赖于选择了哪些 demonstration，不同选择可能导致截然不同的结果。

**缺乏质量区分**：传统 ICL 将所有 demonstration 等权重对待，无法区分高质量和低质量（甚至噪声）示例。

现有的 demonstration 选择方法大多基于启发式规则，无法量化每个示例对泛化性能的贡献。这些问题共同促使作者提出了一种更具表达力的学习方法来组合多组 demonstration 的输出。

## 方法详解

### 整体框架

MoICL（Mixtures of In-Context Learners）的核心思想是：

1. **划分 demonstration**：将 demonstration 集合 D 划分为 k 个不相交的子集 D₁, D₂, ..., Dₖ
2. **创建专家**：每个子集 Dᵢ 连同输入文本 x 一起送入 LLM，产生一个 next-token 分布 p(y|Dᵢ, x)，称为一个"专家"
3. **加权融合**：通过可学习的权重 w ∈ ℝᵏ 融合所有专家的输出分布

最终的预测分布为 Product of Experts 形式：

$$p(y|D, x) \propto \exp\left[\sum_{i=1}^{k} w_i \log p(y|D_i, x)\right]$$

其中 wᵢ 代表第 i 个专家的贡献权重。这种形式允许权重为负值，使得某些专家可以充当"反专家"（anti-expert）。

### 关键设计

**权重函数的两种实现**：

1. **标量权重（Scalar）**：直接学习 k 个标量权重参数 w ∈ ℝᵏ，初始化为 1/k。优点是简单高效，缺点是权重绑定到固定的 demonstration 子集。

2. **超网络（Hyper-network）**：使用一个小型 T5 模型作为超网络 h_ϕ(·)，以所有 demonstration 子集的拼接为输入，动态生成权重。优点是可以泛化到训练时未见过的 demonstration。

**稀疏化混合权重**：

为减少推理开销（需要对每个 token 调用 LLM k 次），作者提出稀疏化权重：

$$w = w' \odot \text{top-}k'(m)$$

其中 m 是可学习的 masking 系数，top-k' 函数只保留最大的 k' 个权重。由于 top-k' 是离散操作，使用 IMLE（Implicit Maximum Likelihood Estimation）进行梯度估计，实现端到端训练。

**反专家机制**：

允许权重为负值是 MoICL 的一个重要设计。负权重意味着该子集的 demonstration 不仅无帮助，还被用作反面参考。实验证明，限制权重为正会导致显著的性能下降（从 81.33% 降至 76.05%）。

### 损失函数

通过最大化训练集 D_T 上的条件对数似然来学习权重参数。训练过程仅更新权重函数的参数（标量权重或超网络参数），无需修改 LLM 的参数。

## 实验

### 主实验结果

在 Llama-3-8B-Instruct 上，使用 30 个 demonstration 进行分类任务评估：

| 方法 | Offensive | Hate | SST2 | PAWS | QNLI |
|------|-----------|------|------|------|------|
| Concat-based ICL | 76.44 | 53.54 | 95.46 | 78.12 | 89.08 |
| Random Search | 77.88 | 58.09 | 95.76 | 78.88 | 89.99 |
| LENS | 78.70 | 53.20 | 93.81 | 75.60 | 89.04 |
| LoRA (PEFT) | 79.79 | 53.76 | 85.89 | 54.82 | 57.24 |
| **MoICL scalar k=10** | **79.42** | **66.52** | 95.32 | **79.42** | **90.44** |
| **MoICL scalar k=30** | **81.33** | 63.45 | 94.79 | **79.50** | 90.11 |

MoICL 在 7 个数据集中的 5 个上超越了所有基线方法，其中在 Hate 数据集上提升最大（+13%）。值得注意的是，LoRA 需要访问模型权重进行微调，而 MoICL 不需要。

### 鲁棒性实验

**OOD demonstration（域外示例）**：当 70% 的 demonstration 来自不同数据集（SST2）时，MoICL scalar 仍保持 80.19% 的准确率，而 concat-based ICL 降至 68.49%，MoICL 提升高达 +11%。权重可视化显示，域内 demonstration 获得正权重，域外 demonstration 获得负权重，说明 MoICL 成功学会了区分域内外示例。

**标签不平衡**：在 30 个 demonstration 中仅 1 个为 "neutral"、29 个为 "offensive" 的极端不平衡设置下，concat-based ICL 准确率从 76.44% 骤降至 28.49%，而 MoICL scalar 仅从 81.33% 降至 77.77%，提升高达 +49%。

**噪声 demonstration（标签噪声）**：在 NQ-Open 生成任务中，即使 10/12 个 demonstration 的答案被改为随机值，MoICL scalar 仍保持稳定性能（+35% 以上），而 concat-based ICL 性能持续下降。权重分析显示噪声 demonstration 获得了接近 0 或负的权重。

### 消融实验

1. **子集数量的影响**：scalar 权重下，增加子集数量 k（即每个子集更少 demonstration）反而提升性能。虽然每个子集的 demonstration 减少导致单个专家变弱，但更多可调权重增加了灵活性，补偿了这一损失。

2. **稀疏化效果**：IMLE top-k' mask 在仅选择 5 个专家时就达到 76.07% 的准确率，展现了良好的稀疏化性能和稳定性。

3. **模型规模影响**：在 Llama-2 的 7B、13B、70B 版本上，MoICL scalar 均一致优于 concat-based ICL，70B 模型上提升最大（82.26% vs 69.42%）。

### 效率分析

- **数据效率**：仅需约 20 个标注 demonstration 即可超越 concat-based ICL
- **时间效率**：在相同性能下，MoICL 需要的推理时间更少
- **计算复杂度**：MoICL 的复杂度为 k·(n/k+1)²·C_LLM，优于 concat-based ICL 的 (n+1)²·C_LLM

## 亮点与洞察

1. **反专家的价值**：允许负权重使 MoICL 能够从"反面教材"中学习，这在 OOD 和噪声场景中尤为重要——坏的 demonstration 不仅被忽略，还被利用
2. **无需模型权重**：MoICL 仅需访问 LLM 的输出 logits，无需修改模型参数，适用于半黑盒场景
3. **从 MoE 到 MoICL 的思路迁移**：将 Mixture of Experts 的思想迁移到 ICL 中，demonstration 子集充当专家，权重通过梯度学习，是一个优雅的类比
4. **Pareto 前沿改进**：MoICL 在准确率-效率的 Pareto 前沿上具有优势，用更少的上下文长度达到更好的性能

## 局限性

1. **需要 logits 访问**：MoICL 需要访问 LLM 的 vocabulary 分布 logits，不适用于纯黑盒 API（如 GPT-4 等不暴露 logits 的模型）
2. **需要训练数据**：权重调优需要一些标注数据作为训练集
3. **实验规模有限**：仅在 Llama-2 和 Llama-3 系列上实验，尚未在更大模型或闭源模型上验证
4. **权重与 demonstration 子集绑定**：scalar 权重方法要求固定的 demonstration 划分，超网络虽解决这个问题但增加了少量计算开销

## 相关工作

- **Concat-based ICL**（Brown et al., 2020）：标准 ICL 方法，将所有 demonstration 拼接到上下文中
- **Ensemble-based ICL**（Min et al., 2022）：每个 demonstration 独立输入 LLM，输出分布做乘积融合
- **LENS**（Li and Qiu, 2023）：基于 retrieval 的 ICL 改进方法
- **Mixtures of In-Context Experts**（Le et al., 2022）：用余弦相似度计算权重，本文的直接前驱工作

## 评分

⭐⭐⭐⭐ (4/5)

创新性强，将 MoE 思想优雅地应用于 ICL；实验设计丰富、分析深入，尤其是鲁棒性分析令人印象深刻。但局限于需要 logits 访问，在纯黑盒 LLM 时代的适用性受限。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ExpeTrans: LLMs Are Experiential Transfer Learners](expetrans_llms_are_experiential_transfer_learners.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)

<!-- RELATED:END -->
