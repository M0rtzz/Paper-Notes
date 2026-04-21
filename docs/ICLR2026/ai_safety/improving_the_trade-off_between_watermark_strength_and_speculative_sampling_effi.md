---
title: >-
  [论文解读] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models
description: >-
  [ICLR 2026][AI安全][LLM 水印] 提出水印强度的量化度量（期望 KL 散度）并完整刻画其与推测采样效率的 Pareto 权衡曲线，进而通过将接受决策伪随机化实现最大水印强度和最优采样效率的同时达成。
tags:
  - ICLR 2026
  - AI安全
  - LLM 水印
  - 推测采样
  - 水印强度
  - 采样效率
  - Pareto 前沿
  - 伪随机接受
---

# Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models

**会议**: ICLR 2026  
**arXiv**: [2602.01428](https://arxiv.org/abs/2602.01428)  
**代码**: [GitHub](https://github.com/hwq0726/watermark-tradeoff)  
**领域**: AI安全  
**关键词**: LLM 水印, 推测采样, 水印强度, 采样效率, Pareto 前沿, 伪随机接受  

## 一句话总结

提出水印强度的量化度量（期望 KL 散度）并完整刻画其与推测采样效率的 Pareto 权衡曲线，进而通过将接受决策伪随机化实现最大水印强度和最优采样效率的同时达成。

## 背景与动机

1. **LLM 水印是溯源的关键技术**：水印通过在 token 采样过程中注入可恢复的伪随机信号实现文本溯源，是保证生成内容可追踪的原则性方法。

2. **推测采样加速 LLM 推理**：Speculative sampling 利用轻量级 draft 模型快速生成候选 token、大模型并行验证，接受率越高加速越明显。效率用期望接受率衡量：$\text{SE} = \mathbb{E}_\zeta[\sum_w \mathcal{A}_\zeta(w|w) Q_{\zeta,w}]$。

3. **水印与推测采样存在根本性权衡**：Hu & Huang (2024) 证明不可能同时保持最高接受率和最强水印——更强的水印使 draft/target 分布偏离更多，降低接受率。这一"不可能"结果令人沮丧。

4. **先前水印强度定义过于粗糙**：现有定义是二值的（精确匹配水印分布才算"保持"），忽略了中间水印强度级别，无法进行精细的权衡分析。本文需要一个连续的、可量化的水印强度度量来完整刻画权衡曲线。

## 方法详解

### 整体框架

- **功能**：(1) 定义水印强度的连续量化度量；(2) 完整刻画水印强度 vs 采样效率的 Pareto 前沿；(3) 提出打破权衡的新机制。
- **为什么**：先前二值定义无法揭示权衡的连续性质，且接受/拒绝中的真随机性是导致水印强度损失的根源。
- **怎么做**：用期望 KL 散度量化水印强度 → 推导 Pareto 曲线的凸优化公式 → 将接受决策伪随机化使整个生成过程成为伪随机数的确定性函数。

### 关键设计 1：水印强度的量化度量

定义无偏水印方案的水印强度为水印分布与原始分布间的期望 KL 散度：

$$\text{WS}(\bm{P}_\zeta) = \mathbb{E}_\zeta[D_{\mathrm{KL}}(\bm{P}_\zeta \| \bm{P})] = \text{Ent}(\bm{P}) - \mathbb{E}_\zeta[\text{Ent}(\bm{P}_\zeta)]$$

这一度量具有深刻的信息论含义：等价于 token $w$ 与伪随机数 $\zeta$ 之间的互信息 $I(w; \zeta)$。其关键性质包括：

- **上界为原始分布的熵**：$\text{WS}(\bm{P}_\zeta) \leq \text{Ent}(\bm{P})$，当且仅当 $\bm{P}_\zeta$ 几乎处处退化（所有概率集中在一个 token 上）时取等
- **决定 p-value 指数衰减率**：在似然比检验下，$n$ 个 token 的 p-value 满足 $\lim_{n\to\infty} -\frac{1}{n}\log(\text{p-value}) = \bar{D}$，直接决定检测所需的样本复杂度
- **Gumbel-max 和 SynthID（$m\to\infty$）均达到最大水印强度**

### 关键设计 2：Pareto 权衡曲线的完整刻画

将权衡曲线定义为约束优化问题：给定效率要求 $r$，求可达到的最大水印强度：

$$L(r) = \max_{\mathcal{S}_{\text{draft}}, \mathcal{S}_{\text{target}}} \text{WS}(\bm{P}_\zeta) \quad \text{s.t.} \quad \text{SSE}(\bm{Q}_\zeta, \bm{P}_\zeta) \geq r$$

对于线性水印类 $\mathcal{Q} = \{(1-\theta)\text{Id} + \theta \mathcal{S} : \theta \in [0,1]\}$，逆曲线可化为凸优化问题，其中目标是最小化 $\ell_1$ 范数（对应总变差距离），约束是熵不超过阈值。论文为 Gumbel-max 和 SynthID 两种水印方案分别绘制了完整的 Pareto 曲线。

### 关键设计 3：伪随机接受机制打破权衡

核心洞察：标准推测采样中，接受/拒绝 draft token 使用的是**真随机**硬币翻转，即使知道 draft 和 target 模型的伪随机数，最终 token 仍不确定。这种残余随机性削弱了水印强度。

解决方案：将接受变量 $u_t = G(\zeta_t^R)$ 也伪随机化，使整个生成过程成为伪随机变量的确定性函数。理论证明（Theorem 4.1）：

- **无偏性**：$\mathbb{E}_\zeta[\bm{P}'_\zeta(w)] = \bm{P}(w)$
- **最大采样效率**：$\text{SE} = 1 - \text{TV}(\bm{Q}, \bm{P})$
- **最大水印强度**：$\text{WS}(\bm{P}'_\zeta) = \text{Ent}(\bm{P})$

检测时，伪随机接受变量 $u_t$ 提供了额外信息，可更准确地选择正确的测试统计量。对 Gumbel-max 提出 Ars-τ 检测器（用阈值 $\tau$ 选择 draft/target 统计量），对 SynthID 提出 Bayes-MLP 检测器（用 MLP 替代简单加权平均选择统计量）。

## 实验

### 实验设置

- **模型对**：Llama-68M (draft) + Llama-7B (target)；Gemma-2B (draft) + Gemma-7B (target)
- **数据集**：ELI5 问答任务，C4 开放生成任务
- **水印方案**：Gumbel-max（温度 0.5）、SynthID（m=30，温度 0.7）
- **lookahead**：K ∈ {2, 3, 4}
- **指标**：AATPS（平均每步接受 token 数）、TPR@FPR=1%

### 主实验：采样效率与检测性能

| 方法 | 水印方案 | AATPS (K=4) | TPR@100 tokens | TPR@200 tokens |
|---|---|---|---|---|
| Std. SpecSampl | 无水印 | ~3.2 | - | - |
| Ars-Prior | Gumbel-max | ~3.2 | ~65% | ~88% |
| **Ars-τ（本文）** | **Gumbel-max** | **~3.2** | **~78%** | **~95%** |
| Bayes-Prior | SynthID | ~3.2 | ~50% | ~75% |
| **Bayes-MLP（本文）** | **SynthID** | **~3.2** | **~62%** | **~85%** |
| Oracle（理论上界） | 两者 | ~3.2 | ~85% | ~98% |

### 消融：权衡曲线验证

| 效率要求 (r) | Gumbel-max WS | SynthID WS (m=30) | SynthID WS (m=∞) |
|---|---|---|---|
| 最大效率 | 0 | 0 | 0 |
| 0.9 × 最大效率 | 中等 | 中等偏低 | 中等 |
| 0.7 × 最大效率 | 高 | 中等 | 高 |
| 无效率约束 | 最大 = Ent(P) | < Ent(P) | 最大 = Ent(P) |

实验验证了 Gumbel-max 和 SynthID（m=∞）达到相同的最大水印强度，但有限 m 的 SynthID 水印强度低于 Gumbel-max。

### 关键发现

1. **采样效率完全保持**：伪随机接受机制下 AATPS 与标准推测采样几乎一致，证实 Theorem 4.1 的最大效率性质。
2. **检测能力显著提升**：在相同 token 数下，Ars-τ 比 Ars-Prior TPR 提升约 10-15 个百分点，Bayes-MLP 比 Bayes-Prior 提升约 10-12 个百分点。
3. **逼近 Oracle 上界**：在 200 tokens 时本文方法已接近理想 Oracle 检测器的性能，说明伪随机接受变量有效降低了测试统计量选择的不确定性。
4. **无偏性验证**：Log Perplexity 与无水印版本一致，确认不降低输出质量。

## 亮点

- 首次提出连续的、有意义的水印强度量化度量，将其与 p-value 衰减率和样本复杂度直接联系。
- 完整刻画 Pareto 权衡曲线，将经验观察转化为严格的约束优化问题。
- 伪随机接受机制的设计极为优雅——一个微小的改动（将真随机硬币替换为伪随机）即同时突破效率和强度的界限。
- 理论分析深入且完整（无偏性、最大效率、最大强度的同时证明），实验验证与理论预测高度一致。

## 局限

- 直接适用于无偏退化水印（Gumbel-max、SynthID），对非退化水印和有偏水印的扩展仍是开放问题。
- 检测需要训练数据（1000 条水印文本），且 SynthID 还需非水印文本作为负样本，增加了部署复杂度。
- 仅评估了标准推测采样，对 tree-based 等变体的扩展未验证。
- 人为编辑对水印的影响未探讨，实际部署中水印鲁棒性是重要考量。

## 评分

| 维度 | 评分 |
|---|---|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 可复现性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](../../ACL2025/ai_safety/from_tradeoff_to_synergy_a_versatile.md)
- [Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret](../../ICML2025/ai_safety/connecting_thompson_sampling_and_ucb_towards_more_efficient_best-fixed_action_.md)
- [Improved Unbiased Watermark for Large Language Models](../../ACL2025/ai_safety/improved_unbiased_watermark_for_large_language.md)
- [Watermark-based Detection and Attribution of AI-Generated Content](watermark-based_attribution_of_ai-generated_content.md)
- [Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](../../AAAI2026/ai_safety/breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)

<!-- RELATED:END -->
