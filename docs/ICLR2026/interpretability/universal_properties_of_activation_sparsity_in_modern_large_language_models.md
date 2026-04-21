---
title: >-
  [论文解读] Universal Properties of Activation Sparsity in Modern Large Language Models
description: >-
  [ICLR2026][activation sparsity] 对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。
tags:
  - ICLR2026
  - activation sparsity
  - LLM acceleration
  - GLU architecture
  - critical sparsity
  - top-p sparsification
  - 扩散模型
---

# Universal Properties of Activation Sparsity in Modern Large Language Models

**会议**: ICLR2026  
**arXiv**: [2509.00454](https://arxiv.org/abs/2509.00454)  
**代码**: [GitHub](https://github.com/fszatkowski/activation-sparsity-benchmarking)  
**领域**: llm_nlp  
**关键词**: activation sparsity, LLM acceleration, GLU architecture, critical sparsity, top-p sparsification, diffusion LLM

## 一句话总结
对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。

## 研究背景与动机

**激活稀疏的历史**：ReLU 网络天然产生精确零激活，围绕此性质的效率优化、鲁棒性增强、可解释性分析已有大量工作。

**现代 LLM 的问题**：主流 LLM（Gemma3、LLaMA3、Qwen2.5）使用 GLU 架构 + SiLU/GELU 激活，不产生严格零值——ReLU 时代的方法无法直接迁移。

**现有方案碎片化**：
   - **改造方案**（将 SiLU 替换为 ReLU）需额外训练且可能损害模型质量
   - **近似稀疏方案**缺乏 ReLU 严格零值的原则性保证，需校准阈值，可能过拟合校准集
   - 不同方法分别针对 FFN 的输入、门控、中间激活，设计选择缺乏统一指导

**本文目标**：建立一个通用、简单、无需训练的框架来系统研究和利用现代 LLM 的激活稀疏性。

## 方法详解

### Top-p 稀疏化规则

对任意激活向量 $v \in \mathbb{R}^n$，保留绝对值最大的条目使其 L1 范数占比达到 $p$：

$$\text{top-p}(v) = m_p \odot v; \quad m_p = \arg\min_m \|m\|_0 \quad \text{s.t.} \quad \|m \odot v\|_1 \geq p \cdot \|v\|_1, \quad m \in \{0,1\}^n$$

诱导的稀疏度为：$S_p(v) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}(m_p^{(i)} = 0)$

**优势**：
- 适用于任意 FFN 模块，无需架构假设或额外训练
- 不存在校准过拟合问题——无需辅助校准数据集
- 简单且易于解释，允许模型间和模块间的公平比较

### 临界稀疏度（Critical Sparsity）

定义为模型保持 ≥99% 原始性能时的最大稀疏度。这提供了一个锚定在实际性能约束上的量化指标，使不同模型/模块的稀疏承受力可直接比较。

### GLU FFN 中的四类激活向量

对 GLU 架构 $\mathcal{FFN}(x) = W_d((W_u x) \odot \sigma(W_g x))$，定义：

| 激活类型 | 定义 | 说明 |
|----------|------|------|
| 输入 $x$ | FFN 输入向量 | 可加速所有三个线性层 |
| 上投影 $u$ | $W_u x$ | 无激活函数的线性投影 |
| 门控 $g$ | $\sigma(W_g x)$ | 经激活函数后的门控信号 |
| 中间 $i$ | $(W_u x) \odot \sigma(W_g x)$ | 逐元素乘积后的中间表示 |

### 三类加速策略比较

| 策略 | 目标激活 | 优点 | 缺点 |
|------|---------|------|------|
| 输入稀疏化 | $x$ | 无需预测器，加速全部 FFN 模块 | 输入无自然稀疏性 |
| 门控稀疏化 | $g$ | 激活函数自然压缩 | 计算门控本身占 FFN 1/3 成本 |
| 预测器方法 | $i$ | 理论最高加速 | 需训练预测器，有近似误差 |

## 实验关键数据

### 模型规模与临界稀疏度（Gemma3 系列）

| 模型 | 参数量 | 中间激活稀疏度 | 输入稀疏度 | 门控稀疏度 |
|------|--------|---------------|-----------|-----------|
| Gemma3-1B | 1B | ~50% | ~35% | ~35% |
| Gemma3-4B | 4B | ~55% | ~40% | ~40% |
| Gemma3-12B | 12B | ~62% | ~48% | ~48% |
| Gemma3-27B | 27B | ~70% | ~55% | ~55% |

**核心发现**：临界稀疏度随模型规模单调递增——更大的模型有更多冗余神经元可以安全跳过。

### 有效秩分析

有效秩（effective rank）随模型规模一致下降，表明大模型的激活表示更低秩、更冗余。但门控激活的有效秩与中间激活类似，虽然其经验稀疏化承受力更差——说明有效秩不足以完全刻画稀疏化鲁棒性。

### 跨模型家族趋势

| 模型家族 | 规模范围 | 临界稀疏度趋势 |
|---------|---------|---------------|
| Gemma3 | 1B–27B | 线性增长最明显 |
| LLaMA3.1/3.2 | 1B–70B | 一致增长，宽度/深度缩放较均匀 |
| Qwen2.5 | 0.5B–72B | 整体增长但较波动，维度增长不均匀 |

### 训练方式的影响

| 模型变体 | 临界稀疏度变化 |
|---------|--------------|
| 预训练 → 指令微调 | 大规模时 IT 模型稀疏度更高 |
| Qwen3-4B Instruct vs Thinking | 推理模型在 GSM8K 上更鲁棒，MMLU 上退化更快 |

### 扩散型 LLM（LLaDA-8B）首次分析

| 任务 | 中间激活临界稀疏度 | All-Inputs 临界稀疏度 |
|------|------------------|---------------------|
| MMLU | 69.46% | 62.72% |
| HumanEval | 81.25% | 77.89% |
| HellaSwag | 71.21% | 67.92% |
| MBPP | 66.67% | 59.18% |
| **平均** | **68.13%** | **56.79%** |

LLaDA-8B 的临界稀疏度显著高于同规模自回归 LLaMA3.1-8B——扩散模型的去噪特性使其对稀疏化引入的噪声更鲁棒。

### 扩散步内的时序稳定性

- 连续扩散步之间的 Jaccard 相似度稳定但不高（~0.6–0.7）
- 与初始步的漂移相似度快速下降——稀疏模式随去噪逐步变化
- 结论：扩散 LLM 的稀疏掩码**不能跨步复用**（与自回归模型在 prompt 后掩码可复用不同）

### MoE 模型分析（Qwen3-30B-A3B）

层内平均临界稀疏度稳定，但个别专家的稀疏度远超平均值。128 个专家中的异常值稀疏度甚至超过同等规模的稠密模型——MoE 专家同样普遍展现激活稀疏性。

## 亮点与洞察
- **"功能性稀疏是 LLM 的普遍性质"**：跨架构（GLU/MoE）、跨训练方式（PT/IT/Thinking）、跨生成范式（自回归/扩散）一致成立
- **输入稀疏化是最实用方案**：不需要预测器、不需要计算门控就能加速全部 FFN 模块——在研究的规模范围内门控并无优势
- **校准的风险**：临界稀疏度在不同任务间差异显著，基于校准数据集的阈值方法存在过拟合风险，应追求真正无数据的稀疏化方案
- **扩散 LLM 的潜力**：首次实证表明扩散型 LLM 的激活稀疏度高于自回归模型，但须针对扩散特性设计专用方法

## 局限与展望
- **仅关注 FFN**：未分析多头注意力中的激活稀疏性，虽然 FFN 主导长上下文以外的计算
- **加速上限有限**：激活稀疏加速约 1.3–1.5x，不如投机解码（~4x），应定位为互补技术
- **top-p 是稀疏度下界**：更复杂的层级/模块特定方法可能达到更高稀疏度
- **未提供具体加速实现**：论文聚焦于稀疏性的表征而非部署优化

## 相关工作与启发
- **vs Mirzadeh et al. (2024)**：先前改造 ReLU 方案需额外训练，本文证明免训练的 top-p 已达实用水平
- **vs Liu et al. (2025a/b)**：输入稀疏化加速方法的经验前提在本文得到系统验证
- **vs Song et al. (2024a) / Lee et al. (2024)**：门控稀疏化在研究规模范围内**不优于**输入稀疏化——重要的实践指南
- **启发**：随着模型持续增大，激活稀疏度持续增长，frontier 模型可能天然拥有 70%+ 的可利用稀疏度（Gemma3n 已开始在架构中集成稀疏感知层）

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一框架 + 临界稀疏度定义 + 首次扩散 LLM 稀疏分析，但核心方法（top-p）简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 Gemma3/LLaMA3/Qwen2.5 多规模 + PT/IT/Thinking + MoE + 扩散模型，9 个基准
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表信息量大，结论明确
- 价值: ⭐⭐⭐⭐ 为 LLM 激活稀疏加速提供了全面的基础参考，实用指导意义强

<!-- RELATED:START -->

## 相关论文

- [The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)
- [ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning](../../AAAI2026/interpretability/hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)
- [Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](../../AAAI2026/interpretability/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)

<!-- RELATED:END -->
