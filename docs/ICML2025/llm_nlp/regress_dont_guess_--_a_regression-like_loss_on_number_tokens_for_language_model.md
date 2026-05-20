---
title: >-
  [论文解读] Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models
description: >-
  [ICML 2025][LLM/NLP][number token loss] 提出 Number Token Loss (NTL)，一种纯 token 级别的回归式损失函数，通过最小化数值 token 之间的 $L_p$ 范数或 Wasserstein 距离，为 LLM 注入数值邻近性归纳偏置。
tags:
  - "ICML 2025"
  - "LLM/NLP"
  - "number token loss"
  - "Wasserstein distance"
  - "numerical reasoning"
  - "regression loss"
  - "LLM"
---

# Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models

**会议**: ICML 2025  
**arXiv**: [2411.02083](https://arxiv.org/abs/2411.02083)  
**代码**: [GitHub (ntloss)](https://github.com/ai4sd/number-token-loss)  
**领域**: Language Modeling / Numerical Reasoning  
**关键词**: number token loss, Wasserstein distance, numerical reasoning, regression loss, LLM  

## 一句话总结

提出 Number Token Loss (NTL)，一种纯 token 级别的回归式损失函数，通过最小化数值 token 之间的 $L_p$ 范数或 Wasserstein 距离，为 LLM 注入数值邻近性归纳偏置。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：LLM 处理数字的四个根本性问题：

**分词**：标准子词分词将数字切割为任意 token

**嵌入**：数字 token 嵌入像普通词一样学习，无法保留数值关系

**顺序预测**：逐 token 解码忽略高位数字的更大重要性

**训练目标**：交叉熵 (CE) 损失假设名义尺度，预测 3 vs 预测 9 对真实值 2 的惩罚相同

## 方法详解

### NTL-MSE（$L_p$ 范数族）

将预测概率映射为实值输出，然后计算与真实值的 MSE：

$$\mathcal{L}_{\text{NTL-MSE}} = \frac{1}{N}\sum_i^N \left(y_i - \hat{\mathbf{y}}_i^{s:t} \circ \mathcal{V}^{s:t}\right)^2$$

其中 $\mathcal{V}: V \to \mathbb{R}$ 将 token 映射为数值，$s...t$ 为数字 token 的索引范围。

**问题**：NTL-MSE 存在**非唯一最小值**——例如标签为 4 时，50% 概率在 0 和 50% 在 8 也会得到零损失。

### NTL-WAS（Wasserstein 距离）

使用离散 Wasserstein-1 距离衡量预测分布与真实分布的差异：

$$\mathcal{L}_{\text{NTL-WAS}} = \frac{1}{N}\sum_{i=1}^N \sum_{j=s}^t \hat{\mathbf{y}}_i^j |y_i - \mathcal{V}^j|$$

当标签为 one-hot 时简化为加权绝对差之和，解决了非唯一最小值问题。

### 联合训练

两种 NTL 均与标准 CE 联合使用：

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{NTL}$$

默认 $\lambda = 0.3$。对非数字 token，NTL 损失为零，不影响文本任务。

### 关键特性

- **模型无关**：适用于任何 LM 架构（Transformer、Mamba 等）
- **即插即用**：仅需 token→数值映射，兼容数字级和多数字分词
- **零开销**：NTL-WAS 仅使损失计算慢 1%，在整体训练步骤中可忽略

## 实验结果

### 数学数据集 (DeepMind, T5-Base, 220M)

| 模型 | 损失 | 准确率 | MAE | R² |
|------|------|--------|-----|-----|
| T5 | CE | 0.64 | 0.13 | 0.97 |
| T5 | NTL-MSE | 0.72 | 0.11 | 0.97 |
| T5 | **NTL-WAS** | **0.75** | **0.10** | **0.98** |
| Regression Transformer | CE | 0.71 | 0.11 | 0.97 |
| xVal | MSE | 0.10 | 0.26 | 0.97 |

- NTL-WAS 在插值和外推测试上均最优
- 在回归任务上，NTL 匹配专用回归头的性能，比标准 CE 提升 10%

### 大规模扩展 (T5-3B, 33B 参数)

NTL 在大规模模型上性能提升一致，不影响纯文本任务表现。

### 消融：分词支持

| 分词方式 | CE 准确率 | NTL-WAS 准确率 |
|----------|----------|---------------|
| 标准子词 | 0.43 | 0.51 |
| 数字级 | 0.64 | **0.75** |

NTL 与数字级分词组合效果最佳。

## 亮点与洞察

- 优雅地解决了 CE 损失在数值 token 上的名义尺度问题
- NTL-WAS 通过 Wasserstein 距离完美克服了朴素回归损失的非唯一最小值缺陷
- 真正的即插即用：发布为 PyPI 包 `ntloss`，一行代码集成
- 灵活的代价函数设计支持非欧几里得空间（如模运算）
- 不影响文本任务 + 零运行时开销 → 没有理由不在预训练中使用

## 局限与展望

- 仅在数学相关任务上验证，未涵盖科学计算、金融等领域
- 对多数字分词器的支持需要非欧代价矩阵，增加了使用复杂度
- $\lambda$ 超参数的选择缺乏理论指导
- 未与近期的 chain-of-thought 等推理增强方法联合验证
- 主要在 T5 和 GPT-2 上实验，缺少对主流 decoder-only LLM 的验证

## 评分

⭐⭐⭐⭐⭐ — 方法简洁优雅、理论完备、即插即用、零开销，应该成为 LLM 预训练的标准组件。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](../../ACL2026/llm_nlp/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2025\] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles](../../ACL2025/llm_nlp/library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](../../ACL2025/llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)

</div>

<!-- RELATED:END -->
