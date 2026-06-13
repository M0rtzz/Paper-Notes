---
title: >-
  [论文解读] Theoretical Guarantees on the Best-of-n Alignment Policy
description: >-
  [ICML2025][图像生成][best-of-n采样] 本文推翻了文献中广泛使用的 best-of-n 策略 KL 散度公式 $\log(n) - (n-1)/n$ 的精确性声明，证明它只是一个上界，并提出了更紧的 KL 散度估计器和 win rate 理论界。 Best-of-n 采样是一种简单高效的推理时对齐方法：从…
tags:
  - "ICML2025"
  - "图像生成"
  - "best-of-n采样"
  - "KL散度"
  - "对齐策略"
  - "推理时计算"
  - "win rate"
  - "理论保证"
---

# Theoretical Guarantees on the Best-of-n Alignment Policy

**会议**: ICML2025  
**arXiv**: [2401.01879](https://arxiv.org/abs/2401.01879)  
**代码**: 无（纯理论工作）  
**领域**: 图像生成  
**关键词**: best-of-n采样, KL散度, 对齐策略, 推理时计算, win rate, 理论保证

## 一句话总结

本文推翻了文献中广泛使用的 best-of-n 策略 KL 散度公式 $\log(n) - (n-1)/n$ 的精确性声明，证明它只是一个上界，并提出了更紧的 KL 散度估计器和 win rate 理论界。

## 研究背景与动机

Best-of-n 采样是一种简单高效的推理时对齐方法：从参考策略 $\pi_{\text{ref}}$ 中采 $n$ 个样本，用奖励函数排序后选最优。该方法在 Llama 2、推理时计算扩展等场景中广泛应用，且在 win rate vs KL 权衡曲线上常优于 RLHF 和 DPO 等更复杂的方法。

文献中普遍使用的一个"公式"声称 best-of-n 策略与参考策略之间的 KL 散度**等于**：

$$D_{\text{KL}}(\pi^{(n)} \| \pi_{\text{ref}}) = \log(n) - \frac{n-1}{n}$$

该公式被 Stiennon et al. (2020)、Gao et al. (2023)、Coste et al. (2024) 等多篇重要工作引用。然而作者通过一个简单的二元输出反例证明**该公式是错误的**——在二元均匀分布下，真实 KL 有上界 $\log 2$，而该公式却随 $n$ 无界增长。

## 方法详解

### Best-of-n 策略的概率质量函数

作者首先严格推导了 best-of-n 策略的 PMF。令输出按奖励从低到高排序，则：

$$\pi^{(n)}(\mathbf{y}|\mathbf{x}) = \mathcal{F}_{\pi_{\text{ref}}}(\mathbf{y}|\mathbf{x})^n - \mathcal{F}_{\pi_{\text{ref}}}^{-}(\mathbf{y}|\mathbf{x})^n$$

其中 $\mathcal{F}$ 和 $\mathcal{F}^{-}$ 分别是基于奖励的 CDF 和严格 CDF。核心思路是将 best-of-n 等价为 $n$ 个均匀随机变量取最大值的分位数变换。

### KL 散度的上下界

**定理 3.1（上界）**：对任意 $n$ 和上下文 $\mathbf{x}$，

$$D_{\text{KL}}(\pi^{(n)}(\cdot|\mathbf{x}) \| \pi_{\text{ref}}(\cdot|\mathbf{x})) \leq \log(n) - \frac{n-1}{n}$$

即经典公式实际是**上界**而非等式。

**定理 3.4（Gap 上界）**：Gap $G_{\text{KL}}^{(n)}$ 受模型 2 阶 Rényi 熵控制：

$$G_{\text{KL}}^{(n)}(\mathbf{x}) \leq 2n(n-1) e^{-H_2(\pi_{\text{ref}}|\mathbf{x})}$$

对 $\delta$-bound 模型（所有输出概率 $\leq \delta$），Gap $\leq 2n(n-1)\delta$。当 $n^2\delta \ll 1$ 时经典公式近似成立。

**定理 3.6（Gap 下界）**：当 $n \cdot \varepsilon_\infty \gg 1$（$\varepsilon_\infty$ 为最高奖励输出的概率）时，Gap 无界增长，经典公式严重高估。

### 新 KL 估计器

作者提出基于观测量 $\varepsilon_n = \pi_{\text{ref}}(\mathbf{y}|\mathbf{x})$（best-of-n 选中样本的参考概率）的估计器：

$$\hat{D}_{\text{KL}}(\varepsilon_n) = (1-\varepsilon_n)^n \left(\log n + (n-1)\log(1-\varepsilon_n) - \frac{n-1}{n}\right) + (1-(1-\varepsilon_n)^n)\log\frac{1-(1-\varepsilon_n)^n}{\varepsilon_n}$$

该估计器满足 $0 \leq \hat{D}_{\text{KL}}(\varepsilon_n) \leq \log(n) - (n-1)/n$，且在数值实验中紧密跟踪真实 KL。

### Win Rate 理论

**定理 5.3**：win rate 上界为 $\mathcal{W}_r \leq \frac{n}{n+1}$。

**定理 5.4**：win rate gap 受 Rényi 熵控制，$G_\mathcal{W}^{(n)} \leq \frac{n-1}{2} e^{-H_2(\pi_{\text{ref}}|\mathbf{x})}$。

当 $n\delta \ll 1$ 时，win rate $\approx n/(n+1)$。

## 理论结果总览

| 指标 | 经典声明 | 本文纠正 | 紧致条件 |
|------|---------|---------|---------|
| KL 散度 | $= \log n - (n-1)/n$ | $\leq \log n - (n-1)/n$（上界） | $n^2\delta \ll 1$ 时近似成立 |
| KL Gap 上界 | — | $\leq 2n(n-1)\delta$ | $\delta$-bound 模型 |
| KL Gap 下界 | — | $\geq \log(n\varepsilon_\infty) + o(\log n)$ | $n\varepsilon_\infty \gg 1$ 时 Gap 无界 |
| Win Rate | 无理论 | $\leq n/(n+1)$ | 对任意模型成立 |
| Win Rate Gap | — | $\leq (n-1)\delta/2$ | $n\delta \ll 1$ 时近似成立 |
| 新 KL 估计器 | — | $\hat{D}_{\text{KL}}(\varepsilon_n) \in [0, \log n - (n-1)/n]$ | 数值实验显示紧致 |

## 实验验证

- **合成实验**：均匀分布（字母表大小 $L=10, 10^2, 10^3, 10^4$）验证 KL 在 $n \approx L$ 时饱和，且新估计器全程紧密跟踪真实 KL
- **Cherry-picked 分布**：展示当高奖励输出概率很小时（如 $10^{-4}$），KL 出现阶梯式增长，经典公式完全失效
- **Alpaca + Gemma 9B**：用对数似然和负长度作为奖励函数，验证在低熵 prompt（如"法国首都是什么"）下新估计器显著优于经典公式
- **Win Rate vs KL 权衡**：best-of-n 的权衡曲线接近 KL 正则化 RL 的最优解，优于 rewind-and-repeat 策略

## 亮点与洞察

1. **纠正广泛误用**：推翻了 6 年来多篇顶会工作中对 best-of-n KL 散度"公式"的误用，影响了 Gao et al. (2023)、Llama 2 等重要工作中的对比曲线
2. **好消息**：经典公式是上界意味着 best-of-n 的实际 reward-KL 权衡**比文献报告的更好**，进一步巩固了其作为强基线的地位
3. **实用估计器**：新估计器仅需观测被选样本的参考概率 $\varepsilon_n$，计算简单且方差可控（$M = O(\log n \cdot \log(1/\delta))$ 次采样即可）
4. **统一理论框架**：同时分析了 best-of-n 和 rewind-and-repeat，证明 best-of-n 在 win rate vs KL 权衡上近似最优
5. **关键洞察**：Rényi 熵 $H_2$ 是控制所有 gap 的核心量——高 Rényi 熵（低概率输出多）时经典公式好用，低 Rényi 熵时失效

## 局限与展望

1. **猜想未证**：Conjecture 4.4（估计器期望是 KL 上界）和 Conjecture 7.2（win rate-KL 权衡上界）仅有数值验证，缺乏严格证明
2. **离散假设**：分析假设有限输出空间，对连续输出（如扩散模型）的推广依赖后续工作（Mroueh 2024）
3. **奖励唯一性假设**：Assumption 2.1 要求奖励值唯一，实际中可能存在大量同分样本
4. **无实际 LLM 对齐实验**：所有实验均为合成或小规模，未在实际 RLHF pipeline 中验证估计器的实用性
5. **估计器需要 $\pi_{\text{ref}}(\mathbf{y}|\mathbf{x})$**：在自回归模型中计算完整序列的概率虽可行但需额外前向传播

## 评分

- 新颖性: ⭐⭐⭐⭐ — 推翻广泛使用的"公式"，提出更紧的理论界和实用估计器
- 实验充分度: ⭐⭐⭐ — 合成实验充分但缺乏大规模实际对齐场景验证
- 写作质量: ⭐⭐⭐⭐⭐ — 论证清晰严谨，从反例到定理到估计器层层递进
- 价值: ⭐⭐⭐⭐ — 对 LLM 对齐社区有重要纠偏意义，后续已被多篇工作引用和推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[ICML 2025\] Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations](video_prediction_policy_a_generalist_robot_policy_with_predictive_visual_represe.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[ICML 2025\] LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces](livs_a_pluralistic_alignment_dataset_for_inclusive_public_spaces.md)

</div>

<!-- RELATED:END -->
