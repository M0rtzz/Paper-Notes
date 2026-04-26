---
title: >-
  [论文解读] Calibrated Multi-Preference Optimization for Aligning Diffusion Models
description: >-
  [CVPR 2025][LLM对齐][校准偏好优化] 本文提出 Calibrated Preference Optimization（CaPO），通过 win-rate 校准将不同奖励模型的分数统一为期望胜率，并设计基于 Pareto 前沿的配对采样策略（FRS）来处理多奖励信号间的冲突，在 SDXL 和 SD3-Medium 上一致地超越 DPO 和 IPO 方法。
tags:
  - CVPR 2025
  - LLM对齐
  - 校准偏好优化
  - 多奖励对齐
  - Pareto前沿
  - 扩散模型微调
  - win-rate校准
---

# Calibrated Multi-Preference Optimization for Aligning Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2502.02588](https://arxiv.org/abs/2502.02588)  
**代码**: https://kyungmnlee.github.io/capo.github.io/  
**领域**: 对齐RLHF / 扩散模型  
**关键词**: 校准偏好优化, 多奖励对齐, Pareto前沿, 扩散模型微调, win-rate校准

## 一句话总结

本文提出 Calibrated Preference Optimization（CaPO），通过 win-rate 校准将不同奖励模型的分数统一为期望胜率，并设计基于 Pareto 前沿的配对采样策略（FRS）来处理多奖励信号间的冲突，在 SDXL 和 SD3-Medium 上一致地超越 DPO 和 IPO 方法。

## 研究背景与动机

**领域现状**：文本到图像（T2I）扩散模型与人类偏好对齐是一个重要研究方向。Diffusion-DPO 将 DPO 成功应用于大规模扩散模型，但需要昂贵的人类标注偏好数据集。一个替代方案是使用多个奖励模型来模拟人类偏好并生成训练数据，这样可以避免人工标注成本。

**现有痛点**：在使用多个奖励模型进行偏好优化时存在三大问题。**第一，pairwise 信息利用不足**：当前方法只用奖励模型来判断二元偏好（win/lose），丢弃了奖励分数中的丰富信息（如质量差距的大小）。**第二，缺乏多奖励泛化**：现有方法主要针对单一奖励设计，直接用奖励加权求和来处理多奖励会导致次优结果，因为不同奖励模型的分数范围和分布差异大。**第三，奖励间存在冲突**：例如优化美学奖励往往会降低 prompt 对齐质量，不同奖励信号之间的不一致性使得简单线性组合失败。

**核心矛盾**：多个奖励模型的黑盒评分分布不一致——有的值域是 [0,1]，有的可能是 [-10,10]，分数的绝对值不反映样本质量的真实好坏程度（Bradley-Terry 模型下的奖励值可能高但实际质量并不好）。

**本文目标** 1）如何将不同奖励模型的分数校准到统一尺度？2）如何在多奖励冲突时选择训练样本对？3）如何避免单一奖励过度优化（reward hacking）？

**切入角度**：作者提出用"对参考模型的期望胜率"（expected win-rate）代替原始奖励值。胜率天然在 [0,1] 范围内，为不同奖励模型提供了统一的衡量标准。通过 pairwise 比较近似计算胜率，得到校准后的奖励信号。对于多奖励场景，使用非支配排序找到 Pareto 前沿来选择训练对。

**核心 idea**：用 win-rate 校准代替原始奖励值、用 Pareto 前沿采样代替加权求和来实现多奖励扩散模型对齐。

## 方法详解

### 整体框架

CaPO 的训练流程如下：（a）用预训练 T2I 模型对每个 prompt 生成 $N$ 张图像，用多个奖励模型打分；（b）对每张图像计算校准奖励——通过与其他 $N-1$ 张图像的 pairwise 比较得到 win-rate 近似值；（c）选择训练对——单奖励时取最高/最低校准奖励的样本，多奖励时用 Pareto 前沿采样；（d）训练时用回归损失最小化校准奖励差与隐式奖励差的差距。

### 关键设计

1. **Win-Rate 校准奖励（Calibrated Rewards）**:

    - 功能：将不同奖励模型的分数统一到 [0,1] 范围，消除分布不一致性
    - 核心思路：对于给定 prompt $c$，从参考模型生成 $N$ 个样本 $\{x_i\}_{i=1}^N$。定义数据 $x$ 对分布 $p(\cdot|c)$ 的胜率为 $\mathbb{P}(x \succ p | c) = \mathbb{E}_{x' \sim p}[\mathbb{P}(x \succ x'|c)]$。在 Bradley-Terry 模型下，pairwise 胜率 $\mathbb{P}(x \succ x'|c) = \sigma(R(x,c) - R(x',c))$。因此，样本 $x_i$ 的校准奖励为所有其他样本的平均 pairwise 胜率：$R_{\text{ca}}(x_i, c) = \frac{1}{N-1}\sum_{j \neq i} \sigma(R(x_i, c) - R(x_j, c))$。当 $N$ 足够大时，$R_{\text{ca}}$ 近似等于对参考模型的真实胜率。胜率天然有界、可比较，解决了不同奖励模型分数范围不一致的问题
    - 设计动机：原始 Bradley-Terry 奖励值即使预测准确性高，也不一定反映样本的绝对质量。例如，在一批低质量样本中，"最好的"得分可能很高但实际质量差。胜率更准确地反映"比参考模型好多少"

2. **基于 Pareto 前沿的配对采样（Frontier-based Rejection Sampling, FRS）**:

    - 功能：在多奖励场景下选择平衡各奖励信号的训练对
    - 核心思路：给定 $L$ 个奖励模型的校准分数，为每个 prompt 的 $N$ 个生成样本构建 $L$ 维向量。使用非支配排序（non-dominated sorting）算法找到上 Pareto 前沿（positive set $X^+$）和下 Pareto 前沿（negative set $X^-$）。定义支配关系：$x$ 支配 $x'$ 当且仅当 $R_{\text{ca}}^{(j)}(x) \geq R_{\text{ca}}^{(j)}(x')$ 对所有 $j=1,...,L$ 成立。上 Pareto 集是非支配点集合，下 Pareto 集是被支配点集合。训练时从 $X^+$ 采正样本、从 $X^-$ 采负样本。优化目标使用多奖励的平均校准分数：$R_{\text{ca}}(x,c) = \frac{1}{L}\sum_{j=1}^{L} R_{\text{ca}}^{(j)}(x,c)$
    - 设计动机：加权求和方法需要手动调权重，且固定权重可能对某些 prompt 不合适。Pareto 前沿是多目标优化的经典概念，自动识别在所有维度上都好/差的样本，无需预设权重。推动模型远离下 Pareto 前沿、趋近上 Pareto 前沿，自然实现多目标平衡

3. **CaPO 回归损失**:

    - 功能：利用校准奖励差作为动态目标，指导扩散模型的偏好学习
    - 核心思路：不同于标准 DPO 的 log-sigmoid 损失（隐式假设 $\Delta R = 1$），CaPO 使用回归损失来匹配校准奖励差：$\mathcal{L}_{\text{CaPO}}(\theta) = \mathbb{E}_{t,\epsilon,\epsilon'}\left[\left(R_{\text{ca}}(x^+, c) - R_{\text{ca}}(x^-, c) - \beta(R_\theta(x_t^+, c, t) - R_\theta(x_t^-, c, t))\right)^2\right]$。其中 $R_\theta$ 是扩散模型的隐式奖励，由 $\epsilon$-prediction 损失差定义。CaPO 是 IPO 的泛化（IPO 固定 $\Delta R = 1$），通过动态目标避免过度优化——当校准奖励差小时，模型不会被迫学习过大的偏好差距
    - 设计动机：固定 $\Delta R = 1$ 的 IPO 对所有偏好对施加相同力度的学习信号，忽略了对之间质量差距的大小。CaPO 的动态目标使学习更精细——大差距对给更强信号，小差距对给更温和信号

### 损失加权

采用单调递减的 sigmoid 权重函数 $w_t = \sigma(-\lambda_t + b)$，其中 $\lambda_t$ 是 log-SNR，$b$ 是偏置参数。高噪声（低 $\lambda_t$）时权重大，低噪声（高 $\lambda_t$）时权重小。等价于用加权 ELBO 代替 KL 散度作为正则化项。

## 实验关键数据

### 单奖励实验（SDXL，对基线的 win-rate %）

| 微调奖励 → | MPS | VQAscore | VILA |
|------------|-----|----------|------|
| DPO | 58.5 / 49.3 / 61.7 | 53.1 / 50.6 / 55.9 | 52.6 / 46.4 / 81.8 |
| IPO | 56.8 / 50.1 / 64.1 | 53.1 / 51.9 / 53.8 | 53.3 / 48.5 / 76.1 |
| **CaPO** | **61.1** / 49.7 / **64.9** | **55.5** / **53.2** / **58.7** | **54.1** / **49.6** / **83.1** |

### 多奖励实验（SDXL）

| 目标 | 方法 | MPS Win% | VQA Win% | VILA Win% |
|------|------|----------|----------|-----------|
| DPO | SUM | 57.2 | 52.1 | 71.9 |
| DPO | FRS | 58.1 | 52.9 | 78.6 |
| CaPO | SUM | 61.2 | 52.5 | 75.0 |
| **CaPO** | **FRS** | **61.2** | **54.6** | **79.2** |

### GenEval 基准（图文对齐）

| 模型 | Overall |
|------|---------|
| SDXL | 0.55 |
| CaPO+SDXL | **0.59** |
| SD3-M | 0.68 |
| CaPO+SD3-M | **0.71** |

### 关键发现

- CaPO 在单奖励和多奖励设置下均一致优于 DPO 和 IPO，说明 win-rate 校准确实提供了更好的训练信号
- FRS（Pareto 前沿采样）在多奖励场景下比简单加权求和（SUM）和模型汤（SOUP）都更有效，尤其在 VILA（美学）维度上提升最明显
- CaPO+FRS 在 SDXL 上实现了所有三个奖励维度的同时提升，避免了单一奖励优化时典型的"跷跷板"效应
- 在 GenEval 上 CaPO 对 SDXL 和 SD3-M 都有提升，说明在改善视觉质量的同时没有牺牲 prompt 对齐
- SD3-Medium 上的表现同样稳定，证明方法对不同架构的扩散模型（U-Net vs DiT）都有效

## 亮点与洞察

- **Win-rate 校准是一个简洁而强大的想法**：将不可比较的奖励分数转化为统一的"比参考模型好多少"的度量，方法优雅且理论有据
- **Pareto 前沿避免了人工调权重**：非支配排序是多目标优化的经典工具，而此前扩散模型偏好优化领域几乎没有使用，是一个有价值的引入
- **CaPO 作为 IPO 的泛化**：动态目标 $\Delta R$ 比固定目标更合理，既避免了 DPO 的不稳定性，又比 IPO 更精细
- **不需要人类标注数据**：完全依赖奖励模型打分，大幅降低了数据成本

## 局限与展望

- 依赖多个奖励模型的质量——如果奖励模型本身不准确或存在系统偏差,校准后的结果可能仍然次优
- 生成 $N$ 个样本和多奖励评分的数据准备成本较高（GPU 开销大）
- 实验主要在 SDXL 和 SD3-M 上验证，尚未在更大规模模型（如 Flux、DALL-E 3）上测试
- Pareto 前沿在高维奖励空间（$L$ 很大时）可能退化——大部分点都是非支配的，采样策略可能失效
- 未与 RLHF 方法（如 DDPO、ReFL）做直接对比
- 缺乏大规模人类评估，自动指标与真实人类偏好的一致性未充分验证

## 相关工作与启发

- **Diffusion-DPO (Wallace et al., 2023)**：将 DPO 引入扩散模型，CaPO 在此基础上解决了奖励校准和多奖励问题
- **IPO (Azar et al., 2024)**：提出一般性偏好优化框架，CaPO 是其扩展（动态化了固定的 $\Delta R=1$ 目标）
- **Rewarded Soups (Rame et al., 2024)**：通过合并单奖励微调模型来实现多奖励优化，CaPO 的联合优化方法更优
- **DPOK / DDPO**：基于策略梯度的扩散模型微调方法，计算代价更高
- CaPO 的 win-rate 校准思路可推广到 LLM 对齐——任何涉及多奖励 DPO 的场景都可以受益于这种统一校准方法

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)
- [\[ICML 2025\] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](../../ICML2025/llm_alignment/d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)

<!-- RELATED:END -->
