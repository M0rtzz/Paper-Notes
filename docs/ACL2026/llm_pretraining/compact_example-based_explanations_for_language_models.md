---
title: >-
  [论文解读] Compact Example-Based Explanations for Language Models
description: >-
  [ACL 2026][预训练][训练数据影响力] 本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。
tags:
  - "ACL 2026"
  - "预训练"
  - "训练数据影响力"
  - "示例解释"
  - "选择相关性"
  - "梯度重构"
  - "冗余消除"
---

# Compact Example-Based Explanations for Language Models

**会议**: ACL 2026  
**arXiv**: [2601.03786](https://arxiv.org/abs/2601.03786)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 训练数据影响力, 示例解释, 选择相关性, 梯度重构, 冗余消除

## 一句话总结

本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。

## 研究背景与动机

**领域现状**：训练数据影响力估计方法（如影响函数）可量化每个训练文档对模型输出的贡献，是示例解释的有前景的信息源。但人类无法处理数千个文档，实际中只能选择少量训练样本作为解释。

**现有痛点**：(1) 选择最高影响力的 k 个样本作为解释是当前默认策略，但高影响力样本往往是全局异常值（如标注错误的数据），不一定与当前测试实例最相关；(2) 最高影响力样本之间高度冗余，严格选择可能收益递减；(3) 现有评估要么在嵌入空间操作（而排序在梯度空间），要么依赖类标签（不适用于生成任务），要么需要重训练（对 LLM 不可行）。

**核心矛盾**：影响力估计方法为每个训练样本生成独立的影响力分数，但作为解释时需要考虑样本之间的互补性和冗余性——一组好的解释样本应该共同覆盖模型决策的关键方面。

**本文目标**：(1) 提出评估选择质量的无重训练指标；(2) 揭示常见选择策略的不足；(3) 设计更好的选择策略。

**切入角度**：将示例解释视为梯度重构任务——好的解释样本应该能用其梯度的线性组合重构测试实例的梯度。

**核心 idea**：选择相关性 = 选中样本的梯度重构测试实例梯度的能力，高质量解释集应最大化重构精度。

## 方法详解

### 整体框架

将选择质量评估形式化为梯度重构问题：给定测试实例的损失梯度 $\nabla\mathcal{L}'$ 和 k 个选中训练样本的梯度矩阵 $A$，计算最优线性组合 $\hat{\nabla\mathcal{L}}' = At$ 的重构误差。选择相关性分数 $\xi^{SR}$ 为原始梯度范数与重构误差的比值（以 dB 表示）。

### 关键设计

1. **选择相关性分数（Selection Relevance Score）**:

    - 功能：量化选中训练样本集作为解释的综合质量
    - 核心思路：$\xi^{SR} = \frac{\mathbb{E}[\|G(\omega)\|^2]}{\mathbb{E}[\|G(\omega) - At_\omega\|^2]}$，即期望梯度平方范数与期望重构误差平方范数的比值。值 >0 dB 表示选中样本提供了有用信息，<0 dB 表示不如零向量基线
    - 设计动机：梯度空间中的重构能力直接反映了训练样本对模型决策的解释力度；考虑样本组合而非独立评分

2. **约束投影（Constrained Projection）**:

    - 功能：确保线性组合系数满足解释语义
    - 核心思路：对系数 $t$ 施加非负性约束（防止不相关样本通过抵消获得权重）和归一化约束（$\sum t = 1$，使 $t$ 可解释为相对重要性）。先计算无约束最小二乘解，再投影到单位单纯形
    - 设计动机：无约束的最小二乘可能产生负系数，意味着某些"解释"样本实际上与预测相矛盾

3. **平衡影响力与代表性的选择策略**:

    - 功能：替代朴素的"选最高影响力"策略
    - 核心思路：在选择过程中同时考虑影响力分数和样本之间的多样性/代表性，避免冗余选择和全局异常值主导
    - 设计动机：实验证明朴素的 top-k 选择常不如随机选择，因为全局异常值和冗余信息降低了解释质量

### 损失函数 / 训练策略

本文不涉及模型训练。选择相关性分数通过解析方法计算（最小二乘+单纯形投影），无需梯度更新。验证实验通过微调（fine-tuning）对比来确认分数的有效性。

## 实验关键数据

### 主实验

**不同选择策略的选择相关性分数（dB，越高越好）**

| 选择策略 | k=1 | k=5 | k=10 | k=25 |
|----------|-----|-----|------|------|
| 随机选择 | 基线 | 基线 | 基线 | 基线 |
| Top-k（最高影响力） | < 随机 | < 随机 | ≈ 随机 | > 随机 |
| 平衡策略（本文） | > 随机 | > 随机 | > 随机 | > 随机 |

### 消融实验

| 影响力估计方法 | 与 Top-k 结合效果 | 与平衡策略结合效果 |
|---------------|------------------|------------------|
| 影响函数 | 差（全局异常值多） | 显著提升 |
| TracIn | 中等 | 提升 |
| TRAK | 较好 | 进一步提升 |

### 关键发现

- Top-k 选择策略在小预算（k≤10）下常不如随机选择——全局异常值和冗余是主因
- 选择相关性分数与微调验证指标高度相关，证明其作为代理评估指标的有效性
- 不同影响力估计方法对选择质量有显著影响：TRAK 比影响函数更适合选择任务
- 平衡策略在所有预算大小和估计方法组合下均优于 Top-k 和随机选择

## 亮点与洞察

- 揭示了一个被忽视的重要问题：示例解释的质量不仅取决于影响力估计的准确性，更取决于选择策略
- "Top-k 不如随机"的发现挑战了领域内的默认假设
- 选择相关性分数提供了首个无重训练、任务无关的选择质量评估工具

## 局限与展望

- 梯度重构作为解释质量的代理可能不完全捕捉用户的实际需求
- 约束投影（非负+归一化）可能排除了某些有效的重构方案
- 在大规模 LLM 上的梯度计算仍然昂贵
- 仅在分类任务上验证，生成任务上的效果待确认

## 相关工作与启发

- **vs Bhatt et al. (2021)**: 他们通过多样性+影响力的加法目标减少冗余，但可能偏好异常值；本文提出代表性作为替代
- **vs Bae et al. (2022)**: 提出预测约束影响力的概念，本文的分数与之高度兼容
- **vs 影响函数**: 影响函数的全局异常值问题在选择任务中尤为突出，本文定量证实了这一点

## 评分

- 新颖性: ⭐⭐⭐⭐ 梯度重构视角和选择相关性分数是新颖的评估工具
- 实验充分度: ⭐⭐⭐⭐ 多种影响力方法×选择策略×预算大小的系统评估
- 写作质量: ⭐⭐⭐⭐⭐ 形式化严谨，动机清晰，分析深入
- 价值: ⭐⭐⭐⭐ 为示例解释领域提供了重要的评估工具和实践建议

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](../../NeurIPS2025/llm_pretraining/learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/llm_pretraining/model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
