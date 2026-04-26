---
title: >-
  [论文解读] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models
description: >-
  [ICLR 2026][图像生成][策略梯度] 提出 PCPO，通过稳定目标重构和原则性时间步重加权，修正扩散/流模型策略梯度中固有的不成比例信用分配问题，显著加速收敛并缓解模型崩溃。
tags:
  - ICLR 2026
  - 图像生成
  - 策略梯度
  - 信用分配
  - 扩散模型
  - 流匹配
  - 模型崩溃
---

# PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models

**会议**: ICLR 2026  
**arXiv**: [2509.25774](https://arxiv.org/abs/2509.25774)  
**代码**: [GitHub](https://github.com/jaylee2000/pcpo/)  
**领域**: 扩散模型对齐 / 强化学习  
**关键词**: 策略梯度, 信用分配, 扩散模型, 流匹配, 模型崩溃

## 一句话总结

提出 PCPO，通过稳定目标重构和原则性时间步重加权，修正扩散/流模型策略梯度中固有的不成比例信用分配问题，显著加速收敛并缓解模型崩溃。

## 研究背景与动机

- GRPO 成为 T2I 模型对齐的 SOTA 框架，但训练不稳定且易出现模型崩溃
- **根因分析**：
  1. 标准目标易受数值精度误差影响
  2. 采样器数学结构导致**不成比例的信用分配**——不同时间步的梯度贡献被任意缩放

## 方法详解

### 问题诊断

**Proposition 1**：对于 DDIM 采样，log 策略比率分解为：

$$\log \rho_t = -[w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)}) \cdot \boldsymbol{\epsilon}_{\text{old}}^{(t)} + \frac{1}{2}\|w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)})\|^2]$$

其中原生权重 $w(t) = C(t)/\sigma_t$ 高度非均匀（跨越数个量级），是训练不稳定的主要来源。

### PCPO 核心设计

**步骤 1：稳定 log-hinge 目标**

用 $\log \rho_t$ 替换不稳定的 $\rho_t - 1$（Taylor 近似误差 < 1.2%）：

$$\mathcal{L}_{\text{PCPO-base}}(\theta) = \mathbb{E}\left[\sum_{t=1}^T \max\{0, \xi|A| - A\log\rho_t\}\right]$$

**步骤 2：成比例信用分配**

- **扩散模型**：重新设计 DDIM 方差调度 $\tilde{\sigma}_t$，使 $w(t) = w^*$（常数），$w^*$ 缩放以匹配原始权重均值
- **流模型**（Proposition 2）：直接重加权训练目标，使信用与积分区间成正比：$w(t_i) = \zeta \Delta t_i$

### 与 REINFORCE 的类比

标准 REINFORCE 中各动作贡献均匀缩放。扩散采样器的梯度公式类似，但引入了非均匀权重 $w(t)$——这是采样器数学的产物而非刻意设计的信用分配策略。PCPO 通过强制均匀权重恢复正确的信用分配。

## 实验关键数据

### 训练效率

| 基线 | 奖励 | 目标水平 | 基线 Epochs | PCPO Epochs | 加速 |
|------|------|---------|------------|------------|------|
| DDPO | Aesthetics | 6.90 | 147 | 118 | **24.6%** |
| DDPO | BERTScore | 0.52 | 191 | 146 | **30.8%** |
| DanceGRPO (SD1.4) | HPS | 0.370 | 236 | 188 | **25.5%** |
| DanceGRPO (FLUX) | HPS | 0.360 | 209 | 148 | **41.2%** |

### 图像质量（匹配奖励水平下）

| 模型 | 方法 | FID(↓) | FD_DINO(↓) | LPIPS(↑) |
|------|------|--------|-----------|----------|
| SD1.5 (batch=512) | 基线 | 24.09 | 451.19 | 0.6321 |
| SD1.5 (batch=512) | **PCPO** | **22.06** | **391.30** | **0.6525** |
| FLUX | 基线 | 46.23 | 539.83 | 0.5736 |
| FLUX | **PCPO** | **40.38** | **438.88** | 0.5708 |

### 关键发现

1. PCPO 在所有设定下均显著降低裁剪比率（clipping fraction），直接改善收敛
2. FLUX 上加速最显著（41.2%），因为时间步偏移使原生权重更不均匀
3. LMM 统计分析证实 PCPO 对 FID 的改善具有统计显著性（p=0.047）
4. 在 MSCOCO-2017 和 MJHQ-30K 未见提示上泛化良好

## 亮点与洞察

1. **根因分析精准**：将训练不稳定归因于采样器数学结构导致的不成比例信用分配
2. **统一处理扩散和流模型**：针对不同采样器提供了原则性的修正方案
3. **实现简单但效果显著**：仅需修改方差调度或重加权目标，即插即用
4. **缓解模型崩溃**：不仅加速收敛，还维持了图像多样性和保真度

## 局限性

- $\log \rho_t \approx \rho_t - 1$ 的 Taylor 近似假设小策略更新，大步长更新时可能失效
- 方差调度修改可能改变采样轨迹特性（尽管实验表明影响可忽略）
- 与 TempFlow-GRPO、MixGRPO 等并发工作的深入比较有限
- 仅在 Aesthetics、BERTScore、HPSv2.1 三种奖励下验证

## 相关工作

- **T2I 对齐**：DDPO, DPO-Diffusion, DanceGRPO, Flow-GRPO
- **LLM 对齐**：PPO, GRPO, DPO
- **模型崩溃**：Shumailov et al. (2024), 奖励黑客

## 评分

- 新颖性：⭐⭐⭐⭐ — 诊断精准但修正方法相对简单
- 技术深度：⭐⭐⭐⭐ — 从 REINFORCE 视角的分析深刻，数学推导完整
- 实验完整性：⭐⭐⭐⭐⭐ — 多框架多奖励验证 + LMM 统计验证
- 实用价值：⭐⭐⭐⭐⭐ — 即插即用，对现有对齐流程有直接提升

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](../../ICML2025/image_generation/discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [\[ICLR 2026\] From Parameters to Behaviors: Unsupervised Compression of the Policy Space](from_parameters_to_behaviors_unsupervised_compression_of_the_policy_space.md)

<!-- RELATED:END -->
