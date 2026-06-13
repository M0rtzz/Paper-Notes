---
title: >-
  [论文解读] Provable Maximum Entropy Manifold Exploration via Diffusion Models
description: >-
  [ICML2025][预训练][扩散模型] 提出 S-MEME 算法，将扩散模型的探索问题形式化为近似数据流形上的熵最大化，通过利用 score 函数与熵一阶变分的内在联系绕开密度估计，以镜像下降方式迭代微调预训练扩散模型，并证明收敛到最优探索策略。 - 核心问题：生成模型（尤其是扩散模型）擅长拟合数据分布…
tags:
  - "ICML2025"
  - "预训练"
  - "扩散模型"
  - "最大熵探索"
  - "流形探索"
  - "镜像下降"
  - "score function"
  - "微调"
---

# Provable Maximum Entropy Manifold Exploration via Diffusion Models

**会议**: ICML2025  
**arXiv**: [2506.15385](https://arxiv.org/abs/2506.15385)  
**代码**: 待确认  
**领域**: 扩散模型探索  
**关键词**: 扩散模型, 最大熵探索, 流形探索, 镜像下降, score function, 微调

## 一句话总结

提出 S-MEME 算法，将扩散模型的探索问题形式化为近似数据流形上的熵最大化，通过利用 score 函数与熵一阶变分的内在联系绕开密度估计，以镜像下降方式迭代微调预训练扩散模型，并证明收敛到最优探索策略。

## 研究背景与动机

- **核心问题**：生成模型（尤其是扩散模型）擅长拟合数据分布，但在科学发现等场景中，目标不是模仿数据分布而是**探索数据流形上的新区域**。如何利用生成模型的表征能力引导探索，是一个基本挑战。
- **现有局限**：
    - 传统生成模型只能近似 $p_{\text{data}}$，倾向于在高密度区域采样，无法覆盖低密度区域
    - 显式不确定性量化在高维空间中计算代价高昂
    - 密度估计（$p_T^{\text{pre}}(x)$）在高维实际场景中极具挑战性
- **关键洞察**：预训练扩散模型隐式定义了一个近似数据流形 $\Omega_{\text{pre}} = \text{supp}(p_T^{\text{pre}})$，可以在该流形上做熵最大化来实现均匀探索

## 方法详解

### 1. 最大熵流形探索问题

将探索形式化为在近似数据流形 $\Omega_{\text{pre}}$ 上的熵最大化：

$$\pi^* \in \arg\max_{\pi} \; \mathcal{H}(p_T^\pi), \quad \text{s.t.} \; p_T^\pi \in \mathbb{P}(\Omega_{\text{pre}})$$

其中 $\mathcal{H}(\mu) = -\int d\mu \log \frac{d\mu}{dx}$ 为微分熵泛函。$\Omega_{\text{pre}}$ 的紧致性由 Proposition 1 保证（当 score 函数 Lipschitz 且噪声分布为截断高斯时）。

### 2. 熵一阶变分与 Surprise 最大化

对熵泛函做一阶变分，得到**正则化 surprise 最大化**原则：

$$\pi^* \in \arg\max_\pi \; \mathbb{E}_{x \sim \pi}\left[-\log p_T^{\text{pre}}(x)\right] - \alpha \, D_{KL}(p_T^\pi, p_T^{\text{pre}})$$

- 第一项 $-\log p_T^{\text{pre}}(x)$ 即 surprise：鼓励在预训练模型低密度区域采样
- 第二项 KL 正则化：约束微调模型不偏离数据流形

### 3. 绕开密度估计的关键联系

论文的核心理论贡献——熵一阶变分的梯度等于负 score 函数：

$$\nabla_x \delta\mathcal{H}(p_T^\pi)(x) = -\nabla_x \log p_T^\pi(x) = -s^\pi(x, T)$$

这意味着无需估计概率密度 $p_T^{\text{pre}}(x)$，直接使用预训练模型的 score 函数 $s^{\text{pre}}(x, T)$ 即可作为奖励梯度，配合 Adjoint Matching 等一阶微调方法求解。

### 4. S-MEME 算法

**Score-based Maximum Entropy Manifold Exploration (S-MEME)** 将非线性熵优化分解为迭代线性微调：

```
输入: 预训练模型 π_pre, 迭代次数 K, 正则化系数 {α_k}
初始化: π_0 = π_pre
for k = 1 to K:
    设置奖励梯度: ∇f_k = -s^{k-1}(·, T)
    求解微调: π_k = LinearFineTuningSolver(∇f_k, α_k, π_{k-1})
返回 π_K
```

每一步迭代对应一步镜像下降（Mirror Descent），KL 散度扮演 Bregman 散度的角色。

### 5. 收敛保证

- **理想情形（Theorem 5.2）**：在精确 score 估计和精确优化假设下，单步微调即可达到最优探索策略
- **现实情形（Theorem 7.1）**：在噪声和偏差满足 Assumption 7.3 的条件下（偏差 $\|b_k\|_\infty \to 0$，步长调度满足 Robbins-Monro 条件），S-MEME 几乎必然收敛到最优探索策略
- 收敛率为 $\tilde{\mathcal{O}}((\log\log k)^{-1})$

**关键理论性质**：负熵 $\mathcal{F} = -\mathcal{H}$ 相对于自身是 **1-smooth 且 1-strongly convex** 的（Lemma 5.1），这保证了镜像下降框架的适用性。

## 实验关键数据

### 合成数据实验

| 指标 | 预训练模型 $\pi^{\text{pre}}$ | S-MEME $\pi_4$（4步） |
|------|------|------|
| 熵估计 | 低（集中在高密度区域） | 显著提升 |
| 低密度区域覆盖 | 差 | 好（密度更均匀） |
| 数据支撑集保持 | — | ✓ 保持 |

- 80000 个样本上的 Monte Carlo 熵估计显示，仅 4 步 S-MEME 即可大幅提升熵

### 文本到图像实验

| 设置 | 预训练模型 $\pi^{\text{pre}}$ | S-MEME $\pi_3$（3步） |
|------|------|------|
| 提示词 | "A creative architecture." | 同上 |
| 生成多样性 | 常规建筑设计 | 更高复杂度和原创性 |
| 语义保真度 | ✓ | ✓ 保持 |

- 基于 Stable Diffusion 预训练模型，S-MEME 微调后生成的建筑图像呈现更高的创意复杂度
- 固定初始噪声对比可见，微调模型倾向在预训练模型低密度区域采样

## 亮点与洞察

1. **优雅的理论联系**：将熵的一阶变分梯度与 score 函数建立等价关系（Eq. 12），从根本上绕开了高维密度估计这个难题
2. **镜像下降视角**：将扩散模型微调解释为概率空间上的镜像下降，KL 散度作为 Bregman 散度，这一视角非常自然且理论优美
3. **探索与有效性的权衡**：正则化系数 $\alpha$ 可灵活控制探索（低 $\alpha$）与保守（高 $\alpha$）之间的平衡
4. **完整的收敛理论**：从理想化的单步最优性到现实情形下的渐近收敛，理论层次递进且完整
5. **无需外部奖励**：纯粹利用模型自身的 score 函数作为内在奖励，实现自引导探索

## 局限与展望

1. **收敛率较慢**：$\tilde{\mathcal{O}}((\log\log k)^{-1})$ 在理论上是目前最优的，但实际中可能需要较多迭代
2. **计算开销**：每步迭代需要完整的扩散模型微调流程（Adjoint Matching），多步迭代的累计计算成本较高
3. **高维实验有限**：文本到图像实验虽展示了定性效果，但缺乏系统的定量评估指标（如 FID、多样性度量等）
4. **流形假设的强度**：方法依赖于预训练模型隐式定义的流形 $\Omega_{\text{pre}}$ 的质量，若预训练不充分则探索范围受限
5. **仅验证了连续域**：未在分子设计等离散/结构化域上验证，而这些恰恰是科学发现的核心场景
6. **Support Compatibility 假设**（Assumption 7.1）在实际中难以精确验证

## 相关工作与启发

- **扩散模型微调**：DPPO、DDPO、Adjoint Matching (Domingo-Enrich et al., 2024)——本文使用 Adjoint Matching 作为 LinearFineTuningSolver
- **探索与内在奖励**：Count-based exploration、RND——本文的 surprise 最大化可视为密度估计版内在奖励
- **连续时间 RL**：Doya (2000)、Zhao et al. (2024)——将扩散模型反向过程视为连续时间 RL 策略
- **镜像下降/镜像流**：Lu et al. (2018)、Hsieh et al. (2019)——提供了概率空间上优化的理论框架
- **启发**：该框架可推广到任何需要在生成模型定义的隐式空间中做最优探索的场景（如药物发现、材料设计）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (熵一阶变分 = 负score的联系非常漂亮)
- 实验充分度: ⭐⭐⭐ (合成实验充分但高维验证偏弱)
- 写作质量: ⭐⭐⭐⭐⭐ (理论层次清晰，动机到方法到保证环环相扣)
- 价值: ⭐⭐⭐⭐ (为扩散模型的探索提供了坚实的理论基础)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/llm_pretraining/composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[CVPR 2025\] Exploration-Driven Generative Interactive Environments](../../CVPR2025/llm_pretraining/exploration-driven_generative_interactive_environments.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] A Practical Guide for Incorporating Symmetry in Diffusion Policy](../../NeurIPS2025/llm_pretraining/a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)

</div>

<!-- RELATED:END -->
