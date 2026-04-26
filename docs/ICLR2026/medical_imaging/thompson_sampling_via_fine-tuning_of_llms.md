---
title: >-
  [论文解读] Thompson Sampling via Fine-Tuning of LLMs
description: >-
  [医学图像] 提出 ToSFiT，通过微调大语言模型直接参数化最大概率（Probability of Maximality），将 Thompson Sampling 扩展到大规模非结构化离散空间，避免了获取函数最大化的难题。
tags:
  - 医学图像
---

# Thompson Sampling via Fine-Tuning of LLMs

- **会议**: ICLR 2026
- **arXiv**: [2510.13328](https://arxiv.org/abs/2510.13328)
- **代码**: [GitHub](https://github.com/IBM/thompson-sampling-via-fine-tuning-of-llms)
- **领域**: 贝叶斯优化 / LLM微调
- **关键词**: Thompson Sampling, Bayesian Optimization, LLM Fine-Tuning, Probability of Maximality, VBOS

## 一句话总结

提出 ToSFiT，通过微调大语言模型直接参数化最大概率（Probability of Maximality），将 Thompson Sampling 扩展到大规模非结构化离散空间，避免了获取函数最大化的难题。

## 研究背景与动机

贝叶斯优化在大规模非结构化离散空间（如氨基酸序列、量子电路设计）中面临核心挑战：由于缺乏梯度信息，获取函数（acquisition function）的最大化在组合级别大的离散域中不可行。例如，20种氨基酸、长度100的蛋白质序列空间已超过宇宙原子数。

Thompson Sampling（TS）是一种经典的贝叶斯优化策略，通过从奖励后验中采样并选择最大化该样本的点来进行探索-利用平衡。其采样分布本质上就是最大概率（Probability of Maximality, PoM）。然而在大规模离散域中，直接从 PoM 采样同样需要遍历所有点。

**核心思路**：既然 LLM 已经通过预训练编码了丰富的先验知识，能否直接用 LLM 的生成分布来参数化 PoM，从而将 Thompson Sampling 转化为 LLM 微调问题？

## 方法详解

### 整体框架

ToSFiT 的核心思想是将候选生成视为 Thompson 采样，用预训练 LLM 参数化 PoM，并通过 VBOS 目标函数增量式地将 LLM 适配到后验 PoM。整个流程：

1. 用 prompt-conditioned LLM 生成初始候选并观测
2. 拟合高斯过程奖励模型
3. 迭代：生成候选 → 估计 VBOS 梯度 → 微调 LLM → 观测新候选

### 变分贝叶斯乐观采样 (VBOS)

PoM 可以通过最大化 VBOS 目标来近似：

$$\mathcal{V}(\pi) = \mathbb{E}_{x \sim \pi}\left[\mu_x + \sqrt{-2\ln(\pi_x)} \cdot \sigma_x\right]$$

其中 $\mu_x$ 是后验均值，$\sigma_x$ 是后验标准差。$\sqrt{-2\ln(\pi_x)}$ 项充当自适应 UCB 探索奖励。

### VBOS 梯度推导（Proposition 1）

$$\frac{d}{d\theta}\mathcal{V}(\pi^\theta) = \mathbb{E}_{x \sim \pi^\theta}\left[(\mu_x - \xi - v^{-1}(\pi_x^\theta) \cdot \sigma_x) \cdot \frac{d}{d\theta}\ln\pi_x^\theta\right]$$

其中 $v^{-1}(u) = \sqrt{-2\ln u} - 1/\sqrt{-2\ln u}$。这个梯度具有能量模型的解释：当 LLM 隐含的期望奖励 $\mu_x^\theta$ 低估了真实 $\mu_x$ 时，生成概率被提升。

### 梯度稳定化

- 使用 **RLOO 基线**（Reinforce Leave-One-Out）减少方差
- 通过优势函数的经验标准差做归一化
- 数学上等价于 GRPO（Group Relative Policy Optimization）

### 高斯过程可扩展性

通过特征映射 $\phi: X \to \mathbb{H}$ 将核函数转化为线性核，复杂度为 $\Theta(\dim(\mathbb{H})^2)$，与观测数量无关。

### 损失/目标函数

$$\frac{d}{d\theta}\mathcal{V}(\pi^\theta) \approx \frac{1}{B}\sum_i \frac{\hat{\hat{r}}_{x_i}^\theta - \xi_i}{\widehat{\text{advantage std}}} \cdot \frac{d}{d\theta}\ln\pi_{x_i}^\theta$$

## 理论分析

**Theorem 1（核心理论贡献）**：将精确 VBOS 的累积遗憾上界从 $\tilde{\mathcal{O}}(\sqrt{T|X|})$ 改进到 $\tilde{\mathcal{O}}(\sqrt{T\gamma^T})$（$\gamma^T$ 为最大信息增益），并首次给出近似 VBOS 的遗憾上界：

$$\mathbb{E}\left[\sum_{t=1}^T R^* - R_{x^t}\right] \leq \sqrt{C_{\sigma_n} H T \gamma^T} + \mathbb{E}\sum_{t=1}^T D_{\sigma^t}(\pi^t, \tilde{\pi}^t)$$

关键 insight：策略初始化必须接近先验（预训练+上下文），微调需要谨慎（小学习率）以保持先验知识。

## 实验

### 三个任务

| 任务 | 模型 | 搜索空间 | 奖励 |
|------|------|----------|------|
| FAQ 回答优化 | Qwen3-1.7B/8B | 所有 token 序列 | 语义对齐分数 |
| 蛋白质搜索 | ProtGPT2-0.7B | 氨基酸序列 | 热稳定性指数 |
| 量子电路设计 | Qwen2.5-Coder-1.5B/7B | Qiskit 电路代码 | 能量负值 |

### 主要结果

ToSFiT 在所有三个任务中均取得 SOTA 的样本效率和计算效率，显著优于7个基线方法（包括上下文BO、强化学习、进化搜索）。

### 关键发现

- **强先验的重要性**：去除 prompt 中的关键信息（如量子比特数）会显著降低性能
- **谨慎微调**：过大学习率会导致遗忘先验并陷入停滞
- **批量优化有效**：批量大小增大会降低样本效率但提升迭代效率
- **计算-样本效率权衡**：增加每轮梯度步数可进一步提升样本效率

### 消融实验

| 消融 | 效果 |
|------|------|
| 去除先验上下文 | 性能显著下降 |
| 大学习率 | 初始提升但后续停滞 |
| 增加梯度步数 | 样本效率提升 |
| 增大批量 | 迭代效率提升 |

## 亮点

1. 理论与实践完美结合：新的遗憾上界直接指导了算法设计
2. 巧妙利用 LLM 预训练先验，避免了离散空间获取函数最大化
3. VBOS 梯度的能量模型解释优雅且直观
4. 三个高度多样化的实验任务（NLP、蛋白质、量子计算）验证了通用性

## 局限性

1. 使用固定特征映射，未与 GP 联合学习
2. 微调全模型带来计算和内存开销
3. 假设线性核的可扩展 GP，限制了奖励模型的表达能力
4. 仅评估了序列生成任务，未涉及图结构等其他离散空间

## 相关工作

- **离散域 BO**：Bal et al. (2025) 假设笛卡尔积分解；Swersky et al. (2020) 通过局部突变策略优化
- **VAE 松弛**：Kusner et al. (2017) 等将离散空间松弛到连续空间
- **深度核学习**：Ranković & Schwaller (2025) 在线学习特征映射

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 将 Thompson Sampling 与 LLM 微调结合，理论和方法上都有重要贡献
- **实用性**: ⭐⭐⭐⭐ — 适用于蛋白质设计、电路优化等实际场景
- **清晰度**: ⭐⭐⭐⭐⭐ — 理论推导清晰，实验设计well-motivated
- **意义**: ⭐⭐⭐⭐⭐ — 为 LLM 与贝叶斯优化结合开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)

<!-- RELATED:END -->
