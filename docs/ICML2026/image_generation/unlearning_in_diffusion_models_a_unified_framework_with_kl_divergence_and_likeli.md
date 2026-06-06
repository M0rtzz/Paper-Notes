---
title: >-
  [论文解读] 扩散模型中的遗忘：基于 KL 散度和似然约束的统一框架
description: >-
  [ICML 2026][图像生成][机器遗忘] 本文提出统一的约束优化框架——将扩散模型中的机器遗忘问题形式化为最小化与预训练模型的偏差，同时受约束于明确的与遗忘分布的分离条件，通过三种约束形式（反向 KL、前向 KL、似然约束）统一处理概念遗忘和数据遗忘，并证明强对偶性。
tags:
  - "ICML 2026"
  - "图像生成"
  - "机器遗忘"
  - "扩散模型"
  - "KL 散度约束"
  - "似然约束"
  - "强对偶性"
---

# 扩散模型中的遗忘：基于 KL 散度和似然约束的统一框架

**会议**: ICML 2026  
**arXiv**: [2605.30825](https://arxiv.org/abs/2605.30825)  
**代码**: 待确认  
**领域**: 扩散模型 / 图像生成 / 机器遗忘  
**关键词**: 机器遗忘, 扩散模型, KL 散度约束, 似然约束, 强对偶性

## 一句话总结
本文提出统一的约束优化框架——将扩散模型中的机器遗忘问题形式化为最小化与预训练模型的偏差，同时受约束于明确的与遗忘分布的分离条件，通过三种约束形式（反向 KL、前向 KL、似然约束）统一处理概念遗忘和数据遗忘，并证明强对偶性。

## 研究背景与动机

**领域现状**：扩散模型因高质量图像生成能力被广泛应用，但可能生成有害内容、侵犯版权或包含不当概念。机器遗忘成为重要研究方向。

**现有痛点**：现有经验方法（如概念擦除）采用简单权重组合方式平衡两个冲突目标（保留模型能力 vs 移除不当内容），但权重设置本质上是启发式的，且对不同场景的泛化性差。

**核心矛盾**：保留预训练模型的实用性与防止生成特定有害数据/概念是根本冲突的目标，需要系统的权衡机制。

**本文目标**：建立原理性的约束优化框架，显式刻画这一权衡。

**切入角度**：将保留模型能力形式化为最小化与预训练模型的距离，将遗忘形式化为从不良分布中分离的约束条件，用拉格朗日乘数法处理冲突目标。

**核心 idea**：用三个约束优化问题（RU/FU/LU）统一概念与数据遗忘，利用非原子向量测度的凸性证明强对偶性，得到显式最优解。

## 方法详解

### 整体框架
给定预训练模型 $q$、$m$ 个待遗忘分布 $q_u^i$，论文提出三个约束优化问题——（1）**反向 KL 约束遗忘（RU）**：最小化 $D_{KL}(p \| q)$，约束 $D_{KL}(p \| q_u^i) \geq b_i$；（2）**前向 KL 约束遗忘（FU）**：最小化 $D_{KL}(q \| p)$，约束 $D_{KL}(q_u^i \| p) \geq b_i$；（3）**似然约束遗忘（LU）**：最小化 $D_{KL}(p \| q)$，约束 $\mathbb{E}_p[q_u^i] \leq \epsilon_i$。

### 关键设计

1. **约束优化框架 vs 权重启发式**:

    - 功能：系统化编码遗忘-保留权衡，用约束阈值 $b_i$ 或 $\epsilon_i$ 直接指定目标分离程度或似然上界。
    - 核心思路：RU/FU 通过 KL 散度约束推开不良分布；LU 通过似然约束限制高概率采样。三者对应不同的遗忘语义。
    - 设计动机：相比权重法的 ad-hoc 性，约束法提供可解释、可重复的目标；对多目标遗忘更易控制。

2. **强对偶性通过非原子向量测度**:

    - 功能：证明三个问题的强对偶性（包括 KL 约束的非凸问题）。
    - 核心思路：利用 Lyapunov 凸性定理，证明非原子向量测度的像的凸性；将凸性性质从概率测度空间迁移到分数函数空间，克服 KL 约束非凸性。
    - 设计动机：强对偶性使得可在对偶域求解；给出显式解（RU 中的分布比 $p^* \propto q/\prod_i (q_u^i)^{\alpha_i}$、FU 中的分布差 $p^* \propto q - \sum_i \lambda_i q_u^i$、LU 中的指数倾斜 $p^* \propto q \cdot e^{-\sum_i \lambda_i q_u^i}$）。

3. **原对偶算法对接扩散模型**:

    - 功能：将分布空间的最优解映射到实际扩散模型的分数函数参数化。
    - 核心思路：原始步用梯度下降最小化拉格朗日函数；对偶步用梯度上升更新拉格朗日乘数 $\lambda$；对 RU/FU/LU 分别设计相应损失。
    - 设计动机：扩散模型参数化与最优分布的"参数化间隙"总是存在，但原对偶法在约束满足时收敛到近似最优解。

## 实验关键数据

### 三个场景对比

| 场景 | 基准模型 | 对比方法 | 主要指标 | 结论 |
|------|---------|---------|---------|------|
| 似然约束遗忘 | 三高斯混合 + Stable Diffusion | 概念擦除 | KL(保留模式) | 同等遗忘程度下约束法偏离更小 |
| 前向 KL 遗忘 | CelebA-HQ DDPM | 无约束 baseline | KID（保留）/ max SSCD（遗忘） | 相同遗忘程度下 KID 更好 |
| 反向 KL 多概念 | Stable Diffusion | 无约束 baseline | KL(原模型) vs CLIP 分数 | 同等 CLIP 分数下 KL 偏差更小 |

### 遗忘-保留权衡

| 约束类型 | 遗忘程度 | 保留能力 | 超越 baseline 的幅度 |
|---------|---------|------------------|-----------------|
| 似然约束（高） | 低似然 | KL↓ 明显 | 显著优于 |
| 前向 KL（强） | max SSCD 高 | KID↓ 明显 | 约束学会动态权重 |
| 反向 KL（多概念） | KL(unlearn)↑ | KL(origin)↓ 显著 | 同等遗忘下更接近原模型 |

### 关键发现
- 约束学到最优权重分配——前向 KL 中约束方法自动给难遗忘样本大权重、易遗忘样本小权重。
- 似然约束更精细的保留——LU 在抑制不良概念时不会像 RU 那样强行推开其他保留模式。
- 参数化间隙影响——当约束太激进或保留/遗忘分布严重纠缠时，参数化间隙会增大导致约束违反。

## 亮点与洞察
- **三个统一的角度**：RU/FU/LU 通过强对偶得到显式最优解，提供遗忘的统一视图。
- **Lyapunov 定理的新应用**：巧妙地应用 Lyapunov 凸性定理到非原子向量测度，突破 KL 约束的非凸性屏障。
- **可迁移性强**：框架不限于图像生成，原则上可扩展到语言、音频等生成任务。

## 局限与展望
- 参数化间隙问题——理论给出分布空间最优解但实际扩散模型实现中参数化间隙可能较大。
- 计算成本未明确——相比 baseline 需要多次迭代更新对偶变量。
- 扩展性验证不足——仅在文本-图像模型上验证；多个高度纠缠的不良概念同时遗忘表现需评估。

## 相关工作与启发
- **vs 概念擦除（Gandikota et al., 2023）**：概念擦除用启发式权重组合；本文通过约束优化得最优权重。
- **vs 数据遗忘（Wu et al., 2025）**：本文从对偶角度统一 FU 情形，提供显式最优解。
- **vs 正则化方法**：约束法约束阈值更可解释，对多目标遗忘控制更精细。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  将遗忘问题重新形式化为约束优化，强对偶性 + 显式最优解都是新颖贡献。
- 实验充分度: ⭐⭐⭐⭐  三个场景比较充分；扩展任务（文本/音频）缺失。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑清晰，问题定义精准，理论推导详细。
- 价值: ⭐⭐⭐⭐  为遗忘提供原理性框架与算法；对安全可控生成模型有重要意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[ICML 2026\] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)

</div>

<!-- RELATED:END -->
