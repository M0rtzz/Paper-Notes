---
title: >-
  [论文解读] Reranking-based Generation for Unbiased Perspective Summarization
description: >-
  [ACL 2025][视角摘要] 针对政治视角摘要任务，构建了受控测试集验证现有评估指标的可靠性，发现 LLM-based 指标远优于传统指标，并证明基于重排序（Reranking）的方法及在重排序数据上的 DPO 训练能显著提升摘要的覆盖性和忠实性。
tags:
  - ACL 2025
  - 视角摘要
  - 重排序
  - DPO
  - LLM评估指标
  - 无偏摘要
---

# Reranking-based Generation for Unbiased Perspective Summarization

**会议**: ACL 2025  
**arXiv**: [2506.15925](https://arxiv.org/abs/2506.15925)  
**代码**: [narutatsuri/Unbiased-Perspective-Summarization](https://github.com/narutatsuri/Unbiased-Perspective-Summarization) (有)  
**领域**: NLP-文本摘要  
**关键词**: 视角摘要、重排序、DPO、LLM评估指标、无偏摘要  

## 一句话总结

针对政治视角摘要任务，构建了受控测试集验证现有评估指标的可靠性，发现 LLM-based 指标远优于传统指标，并证明基于重排序（Reranking）的方法及在重排序数据上的 DPO 训练能显著提升摘要的覆盖性和忠实性。

## 研究背景与动机

- **领域现状**: LLM 在文本摘要方面取得突破，但在意见型文章（如政治新闻）的摘要中，模型常因位置偏差、输入覆盖不均和幻觉问题导致无法公正地呈现多元观点
- **现有痛点**: (1) 现有评估框架源于新闻摘要领域，未验证其在视角摘要任务中的适用性；(2) 除零样本推理外，其他 LLM 方法（如提示工程、偏好训练）在视角摘要中的效果未被充分探索
- **核心矛盾**: 好的视角摘要需要同时满足"覆盖性"（包含所有关键要点）和"忠实性"（不包含对立观点/幻觉），这两个目标存在内在张力
- **本文目标**: (1) 识别可靠的视角摘要质量评估指标；(2) 探索超越零样本推理的 LLM 方法来提升摘要质量
- **切入角度**: 通过构建带有真实分数的受控测试集来评估指标，然后用验证过的可靠指标来指导 Reranking 和 DPO 训练
- **核心 idea**: 先验证指标可靠性，再用可靠指标做 Reranking 选择最优摘要，最后用 Reranking 数据做 DPO 训练

## 方法详解

### 整体框架

分为两个阶段：(1) **指标评估阶段** — 构建受控测试集，验证哪些指标能可靠衡量覆盖性和忠实性；(2) **方法评估阶段** — 对比 Prompting、机械式注意力修改（PINE）、Reranking、DPO+Reranking 四类方法。

### 关键设计

#### 1. 受控测试集构建（评估指标可靠性）
- **功能**: 为每篇文章创建多个具有不同覆盖性/忠实性分数的摘要
- **核心思路**: 标注者从文章中提取关键摘录 $E_{t,\theta}$，用 LLM 改写为关键要点 $K_{t,\theta}$。生成对立关键要点 $\bar{K}_{t,\theta}$（来自对立视角或语义反转）。从 $K_{t,\theta}$ 选 $k_g$ 个正确要点、从 $\bar{K}_{t,\theta}$ 选 $k_b$ 个错误要点组合为摘要
    - 覆盖性 = $k_g / |K_{t,\theta}|$
    - 忠实性 = $k_g / (k_g + k_b)$
- **规模**: 50 篇文档、5 位标注者、370 个文章-摘要对
- **设计动机**: 通过控制关键要点的选取组合，可精确知道每个摘要的真实覆盖性和忠实性分数

#### 2. Reranking 方法
- **功能**: 生成多个候选摘要，用代理指标选择最佳摘要
- **核心思路**: 使用骨干模型（Llama-3.1-8B-Instruct）生成 9 个候选摘要，使用 LLM-Coverage 和 LLM-Faithfulness 指标评分，取最高分摘要
- **评分模型**: 使用 Qwen2.5-14B-Instruct 作为评分器（避免与自动评估指标的模型重合）
- **设计动机**: Reranking 利用了骨干模型已有的生成多样高质量摘要的能力，无需额外训练

#### 3. DPO + Reranking（偏好训练）
- **功能**: 用 Reranking 标记的偏好对训练模型
- **核心思路**: 迭代执行——模型生成摘要 → 用代理指标评分 → 高分为 preferred、低分为 rejected → DPO 训练骨干模型。在 PoliSum 训练集（1716 篇）上迭代 10 个 epoch
- **设计动机**: 利用模型自生成的数据 + 自动评分构建偏好对，无需人工标注即可进行偏好训练

### 损失函数

DPO 损失：标准的 Direct Preference Optimization 损失，将高分摘要作为偏好样本、低分摘要作为拒绝样本进行偏好训练。

## 实验关键数据

### 主实验：指标可靠性评估

| 指标 | 覆盖性 Spearman ρ | 覆盖性 Winrate | 忠实性 Spearman ρ | 忠实性 Winrate |
|------|-------------------|---------------|-------------------|---------------|
| ROUGE_L (R) | 0.473 | 0.780 | -0.038 | 0.393 |
| BERTScore (R) | 0.527 | 0.815 | -0.032 | 0.415 |
| **LLM-Coverage** | **0.707** | 0.739 | 0.393 | 0.431 |
| AlignScore | 0.261 | 0.503 | **0.650** | **0.773** |
| **LLM-Faithfulness** | 0.462 | 0.398 | **0.706** | 0.537 |

- LLM-Coverage 是最佳覆盖性指标（ρ=0.707），AlignScore 是最佳忠实性指标（ρ=0.650）
- 传统指标（ROUGE、BERTScore）在忠实性上几乎无效

### 消融实验：不同方法的自动评估和人工评估

**自动评估**（覆盖性/忠实性）:

| 方法 | 覆盖性得分 | 忠实性得分 |
|------|-----------|-----------|
| Zero-Shot | baseline | baseline |
| Self-Refine | 微小提升 | 略降 |
| Debate | 微小提升 | 略降 |
| PINE | 无改善 | 无改善 |
| Reranking | **显著提升** | **显著提升** |
| DPO+RR | **最佳** | **最佳** |

**人工评估**:

| 方法 | 覆盖性 | 忠实性 |
|------|--------|--------|
| Zero-Shot | 0.347 | 0.642 |
| Reranking | 0.410 | 0.673 |
| **DPO+RR** | **0.437** | **0.724** |

- DPO+RR 覆盖性提升约 **12%**，忠实性提升约 **8%**

### 关键发现

1. **传统指标不可靠**: ROUGE、BERTScore 在衡量忠实性时完全失效（甚至负相关），BLEURT 和 SummaC 也表现不佳
2. **Prompting 方法效果有限**: Multi-Agent Debate 和 Self-Refine 仅轻微提升覆盖性，忠实性反而下降
3. **Reranking 是强基线**: 无需训练即可大幅超越所有推理时方法
4. **DPO 在合成数据上有效**: 仅用模型自生成的数据就能持续提升两个指标，尤其忠实性改善最大
5. **抽象性无损失**: DPO+RR 的 novel 4-gram ratio（0.953）和 extractive fragment density（1.415）均优于多数基线

## 亮点与洞察

- **"先验证指标再优化方法"** 的研究范式值得借鉴——直接用未验证的指标优化可能在错误方向上努力
- 发现 Reranking 一致优于复杂的 Prompting 方法（Self-Refine、Debate），挑战了"更复杂推理 = 更好结果"的假设
- DPO 在自生成的 Reranking 数据上有效，说明模型的生成空间中已包含高质量摘要，关键在于选择机制
- 覆盖性和忠实性的分离评估发现常用指标的盲区——这对其他摘要任务同样有启示

## 局限与展望

- 受控测试集规模较小（50 篇文档、370 对），可能影响统计显著性
- 仅在政治视角摘要（PoliSum）上验证，其他观点密集领域（如产品评论、医疗争议）未测试
- DPO 训练在小数据上迭代 10 个 epoch，可能存在过拟合风险
- 评分器使用不同模型（Qwen2.5-14B）可能引入模型间偏差
- 人工评估样本较少，仍需更大规模验证

## 相关工作与启发

- **PoliSum**: 本文使用的政治视角摘要数据集和任务定义
- **AlignScore / SummaC**: 事实一致性指标，AlignScore 在此任务中表现突出
- **DPO (Rafailov et al., 2023)**: 本文扩展 DPO 到无人工标注的自动偏好学习场景
- **Self-Refine / Multi-Agent Debate**: 流行的推理时增强方法，但在此任务中效果有限
- **启发**: 对于生成质量优化，选择（Reranking）可能比生成策略的改进（Prompting）更高效

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐: 将指标验证与方法优化结合的研究范式新颖；DPO+Reranking 的自训练循环简洁有效
- **实验** ⭐⭐⭐⭐: 自动+人工评估双验证，指标评估也有专门实验
- **实用性** ⭐⭐⭐⭐: Reranking 方法简单易用，无需额外训练即可提升摘要质量
- **写作** ⭐⭐⭐⭐: 结构清晰，问题定义精确

<!-- RELATED:START -->

## 相关论文

- [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [Gumbel Reranking: Differentiable End-to-End Reranker Optimization](gumbel_reranking.md)
- [RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

<!-- RELATED:END -->
