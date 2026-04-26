---
title: >-
  [论文解读] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions
description: >-
  [ACL 2025][LLM/NLP][公众舆论] 提出直接在大规模公众意见调查数据（SubPOP，含 3362 道题目、70K 子群体-响应对）上微调 LLM，使其预测不同人口统计子群体的意见分布，相比 prompt engineering 基线将 Wasserstein 距离降低 32-46%，且泛化到未见过的调查和子群体。
tags:
  - ACL 2025
  - LLM/NLP
  - 公众舆论
  - 分布预测
  - 调查数据
  - LoRA微调
  - Wasserstein距离
---

# Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions

**会议**: ACL 2025  
**arXiv**: [2502.16761](https://arxiv.org/abs/2502.16761)  
**代码**: [GitHub](https://github.com/JosephJeesungSuh/subpop)  
**领域**: 公众舆论预测 / LLM微调  
**关键词**: 公众舆论, 分布预测, 调查数据, LoRA微调, Wasserstein距离

## 一句话总结

提出直接在大规模公众意见调查数据（SubPOP，含 3362 道题目、70K 子群体-响应对）上微调 LLM，使其预测不同人口统计子群体的意见分布，相比 prompt engineering 基线将 Wasserstein 距离降低 32-46%，且泛化到未见过的调查和子群体。

## 研究背景与动机

**领域现状**: 公众意见调查是探测社会舆论的重要工具，但成本高、耗时长。LLM 被寄望于在调查设计早期阶段预测不同子群体的回答分布，辅助问卷设计和试点测试。

**现有痛点**: 现有方法主要通过 prompt engineering 引导 LLM 模拟特定人群的回答，但效果有限——开箱即用的 LLM 倾向于反映富裕、受教育群体的观点，对少数群体产生刻板印象或偏见，且无法捕捉群体内部的意见方差。

**核心矛盾**: prompt 方法无法有效"引导"LLM 到特定子群体的意见分布上；现有微调数据集规模太小（OpinionQA 仅约 500 题），不足以支撑有效微调。

**本文目标**: 如何让 LLM 准确预测多元子群体的意见**分布**（而非只是最可能的选项）。

**切入角度**: 大规模高质量调查数据天然适合微调——(1) 提供明确的子群体-响应对；(2) 经过后分层校准具有代表性；(3) 可以直接对分布建模。

**核心idea**: 构建大规模调查数据集 SubPOP，使用 Forward KL 散度作为损失函数微调 LLM，让模型输出的 token 概率分布匹配人类调查响应分布。

## 方法详解

### 整体框架

给定一道多选题 $q$ 和一个子群体 $g$，微调后的 LLM 对每个选项 token (A/B/C/D/E) 输出概率，该概率分布应匹配真实人类调查中该群体的回答分布。

### 关键设计

1. **SubPOP 数据集**: 从 Pew 的 ATP（American Trends Panel）wave 61-132 收集 3229 道训练题目（排除 OpinionQA 已含波次），从 NORC 的 GSS（General Social Survey）收集 133 道用于跨调查家族的 OOD 评估。共计 70K 子群体-响应分布对，是 OpinionQA 的 6.5 倍。覆盖 22 个子群体维度（年龄、教育、政治倾向等）。
2. **Forward KL 损失**: 使用 Forward KL 散度 $D_{\text{KL}}(p_H \| p_\theta)$ 作为训练目标，其中 $p_H$ 是人类调查的经验分布，$p_\theta$ 是模型对各选项 token 的输出概率。Forward KL 的"均值追踪"特性确保模型不会忽略人类群体中概率较高的选项。
3. **LoRA 微调**: 对 Llama-2-7B/13B、Mistral-7B、Llama-3-70B 使用 LoRA 微调，使用预训练基座模型而非指令微调版本（实验显示预训练模型效果更好）。
4. **分布建模 vs 替代方案**: 对比了 (1) one-hot 编码——只关注最可能选项，丢失分布信息；(2) 按频率复制数据——计算效率低。直接分布建模更优。

### 评估指标

- **Wasserstein 距离 (WD)**: 考虑选项间的序数关系，衡量预测分布与人类分布的"搬运距离"。比 one-hot accuracy 更细粒度。
- 下界：通过 bootstrap 采样估计人类调查自身的抽样方差。

## 实验与关键数据

### 主实验结果 (Table 1, WD ↓)

| 方法 | OpinionQA (Llama-2-7B) | OpinionQA (Mistral-7B) | SubPOP-Eval (Llama-2-7B) |
|------|------------------------|------------------------|--------------------------|
| 上界 (Uniform) | 0.178 | 0.178 | 0.208 |
| 下界 (Human) | 0.031 | 0.031 | 0.033 |
| Zero-shot QA | 0.173 | 0.153 | 0.206 |
| Few-shot | 0.186 | 0.174 | 0.217 |
| Modular Pluralism | 0.285 | 0.279 | — |
| **SubPOP-FT (Ours)** | **0.106** | **0.096** | **0.121** |

- 微调后 WD 降低 32-46%，所有模型、所有评估集上一致优于基线
- SubPOP-Eval（GSS 调查家族）上同样有效，证明跨机构泛化能力

### 子群体一致性 (Figure 3)

- 22 个子群体的相对改进幅度为 38%-54%（均值 46.7%，标准差 4.4%），非常一致
- 改进不偏向任何特定人口统计群体

### 未见子群体泛化 (Table 2)

- 对未参与训练的子群体（如具体年龄段 18-29、65+ 等），相对改进平均 44.7%，与训练中见过的群体相当
- 模型成功学会了基于人口统计 prompt 的"引导性"——交叉群体分歧模式与人类数据一致

### 数据规模效应 (Figure 5)

- 25% 数据即可达到总改进的 72-78%
- 性能随数据量增长持续提升，且不同模型的提升斜率相似
- 估算需约 25 倍当前数据量才能接近人类下界

## 亮点与洞察

1. **分布建模而非点预测**: 意见天然是分布的，即使同一群体内部也存在多样性，one-hot 方法本质上忽略了这一点
2. **Forward KL 的合理性**: 确保模型覆盖人类分布的高概率区域，与最大似然训练一致
3. **跨调查家族泛化**: SubPOP-Train 基于 ATP，但在 GSS 上同样有效，说明学到了通用的子群体-意见映射
4. **可引导性验证**: 通过"交叉群体分歧矩阵"优雅地证明模型确实根据 prompt 中的群体标签调整输出，而非简单趋近整体均值
5. **训练数据设计**: 每个子群体均匀分配训练样本是一致性改进的关键

## 局限性

1. 仅聚焦美国调查数据，跨文化泛化未验证
2. 子群体定义基于粗粒度人口统计标签，无法捕捉更细粒度的个体差异
3. 选项序数映射假设选项之间有自然顺序（WD 评估的前提），但并非所有调查问题都满足
4. Forward KL 鼓励模式覆盖但可能导致"过于平滑"的分布预测
5. 伦理风险——微调后的模型可能被滥用于合成"假调查结果"

## 相关工作

- **公众意见数据集**: OpinionQA、GlobalOpinionQA、PRISM 等
- **LLM 预测人类意见**: 从 rule-based prompt 到个人叙事 prompt，直至直接微调
- **多元对齐 (Pluralistic Alignment)**: DPO 等偏好学习方法关注成对偏好排序，本文关注分布匹配

## 评分

⭐⭐⭐⭐ — 问题定义清晰，数据集贡献突出（6.5 倍扩展），方法简洁有效。对"可引导性"的验证实验设计精巧。不足之处在于仅限美国调查场景，且方法创新性一般（本质是 LoRA + KL 微调）。

## 补充细节

- 使用预训练模型而非指令微调模型效果更好，可能是因为指令微调模型在 token 概率分布上过于尖锐，不利于分布建模
- ATP 和 GSS 之间的分布转移包括受访者池、校准技术和方法论差异，在这种条件下的泛化说明模型学到的是通用的子群体-意见映射能力
- 每个子群体均匀分配训练样本是一致性改进的关键设计决策
- 数据规模实验中各模型的提升斜率相似，暗示性能增益主要由数据和任务决定而非模型架构

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](salience_sparse_fine_tuning.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)

<!-- RELATED:END -->
