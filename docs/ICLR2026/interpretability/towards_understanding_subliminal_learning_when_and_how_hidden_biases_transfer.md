---
title: >-
  [论文解读] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer
description: >-
  [ICLR 2026][subliminal learning] 本文通过受控实验和机制分析揭示了潜意识学习（subliminal learning）的本质——教师模型的隐藏偏好通过少量"分歧token"（divergence tokens）传递给学生模型，且早期层是关键，同时发现该现象非常脆弱，简单的同义改写即可抑制。
tags:
  - ICLR 2026
  - subliminal learning
  - 知识蒸馏
  - divergence tokens
  - hidden bias transfer
  - AI safety
---

# Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer

**会议**: ICLR 2026  
**arXiv**: [2509.23886](https://arxiv.org/abs/2509.23886)  
**代码**: [GitHub](https://github.com/lmb-freiburg/divergence-tokens)  
**领域**: 模型压缩 / 知识蒸馏 / AI安全  
**关键词**: subliminal learning, knowledge distillation, divergence tokens, hidden bias transfer, AI safety

## 一句话总结

本文通过受控实验和机制分析揭示了潜意识学习（subliminal learning）的本质——教师模型的隐藏偏好通过少量"分歧token"（divergence tokens）传递给学生模型，且早期层是关键，同时发现该现象非常脆弱，简单的同义改写即可抑制。

## 研究背景与动机

知识蒸馏是压缩模型或转移知识的核心技术。传统观点认为，传递的内容取决于训练数据的语义内容——如果教师的输出不显示某种特质（如对某种动物的偏好），学生就不应学到这种特质。

Cloud et al. (2025) 的研究挑战了这一观点：教师的隐藏偏好可以传递给学生，即使训练数据与该偏好完全无关（例如数字序列、代码等）。这种现象被称为**潜意识学习**。

在**软蒸馏**（学生看到教师的完整 next-token 分布）下，潜意识学习可以预期。但令人惊讶的是，**硬蒸馏**（学生只看到采样的 token）下也会发生。先前的解释将其归因于 token 纠缠（token entanglement）和 logit 泄漏（logit leakage），但本文发现这些解释并不充分。

核心研究问题：**潜意识学习究竟何时、如何发生？**

## 方法详解

### 整体框架

本文围绕"分歧 token"（divergence tokens）这一核心概念展开分析：

1. **排除先前假说**：证明 token 纠缠和 logit 泄漏都不是必要条件
2. **发现分歧 token**：识别关键的偏好传递载体
3. **机制分析**：定位关键层并分析脆弱性

### 关键设计 1：排除 Token 纠缠和 Logit 泄漏

- **Logit 泄漏**：使用贪心采样（greedy sampling）生成微调数据，完全避免 logit 泄漏。实验表明偏好仍可传递，甚至某些之前无法传递的偏好（如 Qwen 上的 'dog'）在贪心采样下反而成功传递
- **Token 纠缠**：移除所有包含 50 个最纠缠 token 的训练样本后，隐藏偏好仍可传递

$$\text{Finding 1: 即使没有 logit 泄漏和纠缠 token，隐藏偏好仍可转移}$$

### 关键设计 2：分歧 Token（Divergence Tokens）

在贪心采样下，不同偏好的教师对同一 prompt 常常产生大量相同的 token，然后在某些位置突然分歧。

**定义**：给定由偏好 $b$ 的教师生成的前缀 $x_{<k}$，token $x_k$ 是分歧 token 当且仅当存在另一偏好 $b' \neq b$ 的教师使得：

$$\arg\max_t p_b(t \mid x_{<k}) = x_k \quad \text{且} \quad \arg\max_t p_{b'}(t \mid x_{<k}) \neq x_k$$

分歧 token 非常稀少（Qwen 约 7.5%，Gemma 约 18.3%），但因果效应显著。

### 关键设计 3：损失遮蔽实验

- **仅在分歧 token 上训练**（约 4.7% 的 token）：通常保留甚至增强偏好传递
- **遮蔽分歧 token**（在其余 ~95% token 上训练）：基本消除偏好传递

$$\text{Finding 2: 分歧 token 是潜意识学习的关键驱动力}$$

### 关键设计 4：关键层定位

通过因果中介分析（causal mediation analysis）和归因补丁（attribution patching），发现：

- **早期层**在分歧 token 的首次出现位置具有很强的因果影响
- 仅微调单个早期层（如 layer 0 或 layer 7）就足以诱导潜意识学习
- 微调中后期层（layer 14, 21, 27, 33）则几乎没有传递效果

$$\text{Finding 3: 早期层是关键，微调单个早期层即可实现潜意识学习}$$

### 关键设计 5：脆弱性分析

- **同义改写 prompt**：随机替换 "look at these numbers" 为 "examine these numbers" 等等，通常即可抑制偏好传递，且任务性能不受影响
- **混合教师数据**：混入 10% 无偏教师数据显著降低传递；25% 基本消除
- **即使偏好教师自己改写 prompt**，也通常能抑制传递

$$\text{Finding 4 \& 5: 潜意识学习是脆弱的}$$

## 实验

### 主实验

| 设置 | 方法 | 偏好传递效果 |
|------|------|-------------|
| Qwen 2.5-7B | 温度采样 (FT) | 部分动物成功传递 |
| Qwen 2.5-7B | 贪心采样 (FT greedy) | 多数动物成功传递（甚至更强） |
| Qwen 2.5-7B | 去除纠缠token | 部分动物仍可传递 |
| Gemma 3-4B | 温度采样 (FT) | 多数动物成功传递 |
| Gemma 3-4B | 贪心采样 (FT greedy) | 传递效果一致 |

### 消融实验：分歧 Token 的作用

| 方法 | 分歧 token 比例 | 偏好传递 |
|------|----------------|---------|
| 仅分歧 token（贪心） | ~7.5% (Qwen) | 保留或增强 |
| 非分歧 token（贪心） | ~92.5% | 基本消除 |
| 仅分歧 token（温度） | ~4.7% (Qwen) | 保留或增强 |
| 非分歧 token（温度） | ~95.3% | 基本消除 |

### 关键发现

1. 不需要 logit 泄漏或 token 纠缠即可发生潜意识学习
2. 分歧 token 虽稀少但因果效应显著
3. 早期层最关键，单层微调即可
4. 同义改写即可抑制
5. 混合多教师数据也可抑制

### 错位倾向（Misalignment）实验

使用在危险金融建议上训练的 Qwen 模型，验证分歧 token 在错位倾向传递中同样起关键作用。

## 亮点

- 首次揭示潜意识学习的核心机制：少量分歧 token 驱动，而非全局 token 纠缠
- 发现单个早期层即可实现潜意识学习，提供了精确的机制定位
- 证明潜意识学习的脆弱性，为防御提供了简单有效的方法
- 方法论上的创新：利用贪心采样消除随机性干扰，实现可控分析

## 局限性

- 使用的蒸馏任务（数字序列等）较为程式化，可能不完全反映实际前沿模型的特质传递
- 某些例外情况（如 'penguin'）机制不完全清楚
- 部分模型从未成功传递隐藏偏好，原因尚不明确
- 防御方法虽简单有效但可能不够鲁棒，更强的防御方法有待开发

## 相关工作

- **潜意识学习**：Cloud et al. (2025) 首次发现；Zur et al. (2025) 归因于 token 纠缠（本文否定）
- **清洁标签中毒攻击**：类似但不依赖优化的隐藏信号
- **蒸馏中的暗知识**：Hinton et al. (2015) 的经典工作
- **AI 安全**：与欺骗性对齐、隐藏目标检测等问题密切相关

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次揭示分歧 token 作为潜意识学习核心机制
- 理论深度：⭐⭐⭐⭐ — 因果分析和层定位深入，但缺少形式化理论保证
- 实验充分性：⭐⭐⭐⭐⭐ — 多模型、多偏好、多设置的全面验证
- 实用价值：⭐⭐⭐⭐ — 为蒸馏安全提供了简单有效的防御思路
- 写作质量：⭐⭐⭐⭐⭐ — 结构清晰，逐步推进，结论明确

<!-- RELATED:START -->

## 相关论文

- [When Machine Learning Gets Personal: Evaluating Prediction and Explanation](when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [Base Models Know How to Reason, Thinking Models Learn When](../../NeurIPS2025/interpretability/base_models_know_how_to_reason_thinking_models_learn_when.md)
- [When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)

<!-- RELATED:END -->
