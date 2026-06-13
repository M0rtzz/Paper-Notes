---
title: >-
  [论文解读] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale
description: >-
  [ACL 2025][推理路径缩减] 提出 VARR 框架，以句子为单位并利用基于似然度的"冗余度（verbosity）"标准识别和移除推理路径中的冗余句子，在多种推理任务上平均提升 7.71% 准确率同时减少 19.87% 的 token 生成。 LLM 通过生成冗长的中间推理步骤来提升最终答案质量…
tags:
  - "ACL 2025"
  - "推理路径缩减"
  - "冗余句子"
  - "冗余度"
  - "CoT微调"
  - "token节省"
---

# Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale

**会议**: ACL 2025  
**arXiv**: [2412.21006](https://arxiv.org/abs/2412.21006)  
**代码**: 无  
**领域**: 其他  
**关键词**: 推理路径缩减, 冗余句子, 冗余度, CoT微调, token节省

## 一句话总结

提出 VARR 框架，以句子为单位并利用基于似然度的"冗余度（verbosity）"标准识别和移除推理路径中的冗余句子，在多种推理任务上平均提升 7.71% 准确率同时减少 19.87% 的 token 生成。

## 研究背景与动机

LLM 通过生成冗长的中间推理步骤来提升最终答案质量，但不可避免地增加了推理成本和延迟。更关键的是，使用完整推理路径微调 LLM 并不一定保证性能提升——部分推理句子可能是冗余甚至有害的。

已有推理路径缩减方法存在两大问题：（1）以 token 为单位进行缩减（如 ICoT-SI），缺乏语言学合理性，可能破坏句子语义；（2）缺乏有原则的标准判断哪些内容应被移除，多为启发式方法。这些方法主要在简单算术任务（如多位数乘法）上验证，泛化性不足。

## 方法详解

### 整体框架

VARR 包含三个阶段：（1）预热阶段用完整推理路径正常 CoT 微调；（2）冗余度评估阶段，从前向后检查每个推理句子；（3）根据冗余度标准移除句子后继续训练。

### 关键设计

1. **早期推理句子更冗余的经验发现**：通过计算移除不同位置句子后答案的 NLL 变化，发现移除前部句子对 NLL 影响最小（marginal NLL difference），说明早期句子对生成正确答案的贡献最小。这为"从前向后移除"提供了经验依据。

2. **冗余度（Verbosity）定义**：verbosity(y_g) = log(p(y_g|R',x) / p(y_g|R,x))。本质是 KL 散度之差，衡量移除句子 r_i 后正确答案概率是否提升。 verbosity ≥ 0 意味着移除后正确答案概率不降，该句子可安全移除。

3. **用错误答案增强（VARR+）**：引入 verbosity(y_w) = (1/K) Σ log(p(y_w^k|R',x) / p(y_w^k|R,x))，用 in-batch 负样本采样 K 个错误答案。当 verbosity(y_w) - verbosity(y_g) ≤ 0 时，说明移除该句子使正确答案概率增益大于错误答案概率增益，进一步确认可安全移除。

4. **线性移除调度**：r(t) = ⌊N_t · (t/T)⌋，随训练进行逐步增加可移除句子数量上限，但实际移除受冗余度标准约束（不强制移除）。

### 损失函数 / 训练策略

标准 CoT 训练损失 -log p(y_g, R|x)。预热阶段占 10% 总训练步。每个 epoch 开始时重新初始化优化器以稳定训练。

## 实验关键数据

### 主实验 — Mistral 7B

| 方法 | MathQA | GSM8K | CommonQA | TriviaQA | StrategyQA |
|------|--------|-------|----------|----------|------------|
| Explicit-CoT | 55.84 | 55.26 | 84.33 | 82.94 | 74.70 |
| ICoT-SI | 35.84 | 28.27 | 67.82 | 77.09 | 61.33 |
| Coconut | - | - | - | - | - |
| **VARR+** | **56.95** | **54.98** | **89.56** | **83.45** | **78.19** |

（VARR+ 平均提升 7.71%，token 减少 19.87%）

### 消融实验 — 缩减单位对比

| 方法 | 平均准确率 | 平均 token |
|------|----------|----------|
| ICoT-SI (token, 无标准) | 最低 | 低 |
| VARR-Tok (token + 冗余度标准) | 中 (+24.74% vs ICoT-SI) | 中 |
| **VARR-Sent (句子 + 冗余度标准)** | **最高 (+15.98% vs VARR-Tok)** | 较低 |

### 消融实验 — 移除位置

| 移除位置 | 平均准确率 |
|---------|----------|
| No Rule (随机不受控) | 最低 |
| Random (随机 + 冗余度) | 中 |
| Back (后向 + 冗余度) | 中 |
| **Front (前向 + 冗余度)** | **最高** |

### 关键发现

1. **句子 >> token 作为缩减单位**：句子级别移除比 token 级别多带来 15.98% 的性能提升，因为 token 级移除可能截断语义。
2. **冗余度标准是关键**：即用 token 级别，加入冗余度标准也能比 ICoT-SI 提升 24.74%。
3. **ICoT-SI 和 Coconut 性能严重下降**：平均下降 21.98% 和 25.20%，表明启发式移除方法破坏了推理能力。
4. **VARR+ 的错误答案对比进一步提升鲁棒性**：在多数数据集上 VARR+ 优于 VARR。
5. **实际移除率自适应**：大部分冗余句子在训练早期被移除，后期自动趋于稳定，框架具有自我调节能力。

## 亮点与洞察

- 核心洞察：推理路径中的冗余不是随机分布的，而是集中在前部——这可能因为 LLM 在推理初始阶段生成了大量"铺垫性"语句，实际推理在后续步骤完成。
- 冗余度概念的形式化（KL 散度差→似然比）优雅且可解释。
- 突破了已有方法仅在简单算术任务上验证的局限，覆盖了数学推理、常识推理、阅读理解等多种任务。

## 局限与展望

- 实验仅在 7B 模型和较小 batch size (单张 A100 GPU) 上进行。
- 未在长序列推理任务上测试。
- 冗余度计算需要额外的前向传播（R 和 R' 各一次），增加训练时间。
- 线性移除调度可能不是最优的，可探索自适应调度。

## 相关工作与启发

- 与 Self-Consistency、Tree of Thoughts 等"通过更多推理提升性能"的方法形成互补——本文证明"精简推理也能提升性能"。
- ICoT-SI 的 token 级移除思路被本文系统性地证明为次优方案。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 冗余度概念和句子级缩减的组合是全新的方法论贡献
- **实验充分度**: ⭐⭐⭐⭐ — 5 个数据集、多种消融、跨模型验证（Mistral+Llama3.2）
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，实验逻辑严密
- **价值**: ⭐⭐⭐⭐⭐ — 同时提升性能和效率的方法在实践中极具吸引力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](../../ICML2025/others/randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](../../ECCV2024/others/domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ACL 2025\] FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)
- [\[ACL 2025\] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)

</div>

<!-- RELATED:END -->
