---
title: >-
  [论文解读] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization
description: >-
  [ICCV 2025][图像生成][训练后量化] 提出 DMQ 框架，结合学习型等价缩放（LES）和通道级 Power-of-Two 缩放（PTS）来处理扩散模型量化中的异常值问题，首次在 W4A6 低比特设定下实现稳定的高质量图像生成。 扩散模型虽然在图像生成中取得了卓越成果，但迭代去噪过程带来的巨大计算开销严重限制了其在…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "训练后量化"
  - "扩散模型量化"
  - "异常值处理"
  - "等价缩放"
  - "Power-of-Two缩放"
---

# DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization

**会议**: ICCV 2025  
**arXiv**: [2507.12933](https://arxiv.org/abs/2507.12933)  
**代码**: [GitHub](https://github.com/LeeDongYeun/dmq)  
**领域**: 扩散模型/图像生成  
**关键词**: 训练后量化, 扩散模型量化, 异常值处理, 等价缩放, Power-of-Two缩放

## 一句话总结

提出 DMQ 框架，结合学习型等价缩放（LES）和通道级 Power-of-Two 缩放（PTS）来处理扩散模型量化中的异常值问题，首次在 W4A6 低比特设定下实现稳定的高质量图像生成。

## 研究背景与动机

扩散模型虽然在图像生成中取得了卓越成果，但迭代去噪过程带来的巨大计算开销严重限制了其在资源受限环境中的部署。量化是降低计算和内存需求的核心技术，但扩散模型的量化面临独特挑战：

**激活分布动态变化**：扩散模型在不同时间步共享参数，导致激活分布随时间步剧烈变化，难以用固定量化参数准确表示

**量化误差累积**：去噪过程中每步引入的量化误差会在后续步骤中累积放大，尤其是早期时间步的误差影响最大

**通道异常值问题**：某些通道（特别是 skip connection 层）存在极端异常值，拉伸了量化范围，导致非异常通道的量化精度大幅降低

现有 PTQ 方法（如 Q-Diffusion、TFMQ-DM）主要关注校准数据组成和时间步适配，但**忽视了异常值问题**。直接应用 LLM 领域成功的 SmoothQuant 方法也不可行——因为扩散模型激活远大于权重，SmoothQuant 产生的大缩放因子会严重放大权重量化误差（如 Table 1 所示，FID 从 36 飙升至 454）。

## 方法详解

### 整体框架

DMQ 统一了两个关键技术：(1) LES 在所有层上细粒度调整异常值分布以平衡量化难度；(2) PTS 在极端异常值层（如 skip connection）上用 power-of-two 因子直接消除异常值。两者协同工作，确保在低比特约束下仍能准确量化。

### 关键设计

1. **学习型等价缩放（Learned Equivalent Scaling, LES）**:

    - 功能：学习通道级缩放因子 $\tau \in \mathbb{R}^{C_{in}}$，将量化难度在权重和激活之间重新分配
    - 核心思路：对矩阵乘法引入缩放变换：
    $Y = (X/\tau)(\tau^T \odot W) = \hat{X}\hat{W}$
      然后优化 $\tau$ 最小化量化误差：
    $\mathcal{L}_i = \|X_i W - Q(\hat{X}_i)Q(\hat{W})\|^2$
    - 设计动机：与 SmoothQuant 基于启发式规则不同，LES 通过梯度优化学习最优缩放因子，避免了扩散模型中激活远大于权重导致的缩放因子过大问题。同时将 $\tau$ 融合到量化尺度中消除推理开销。

2. **自适应时间步加权（Adaptive Timestep Weighting）**:

    - 功能：根据时间步对损失函数进行自适应加权，优先优化关键时间步
    - 核心思路：加权因子定义为：
    $\lambda_{t_i} = \left(1 - \frac{\Lambda_{t_i}}{\sum_{t' \in T}\Lambda_{t'}}\right)^\alpha$
      其中 $\Lambda_t$ 是累积损失的移动平均（动量 $\xi=0.95$）。
    - 设计动机：分析发现一个关键矛盾——后期时间步量化误差更大，但早期时间步的小误差因累积效应对最终质量影响更大。简单的单调加权反而不如均匀加权，因为不同层在不同时间步的误差趋势各异。自适应策略借鉴 Focal Loss，动态平衡误差大小与误差影响。

3. **通道级 Power-of-Two 缩放（PTS）**:

    - 功能：对高层间方差的激活通道施加 $2^\delta$ 因子缩放，直接消除极端异常值
    - 核心思路：活化量化公式修改为：
    $\tilde{X} = \text{clamp}\left(\lfloor \frac{X}{2^\delta \odot s^{(X)}} \rceil, l, u\right)$
      乘以 PTS 因子等价于对权重进行位移操作：$\tilde{W}_{kj} \ll \delta_k$，硬件实现高效。
    - 设计动机：LES 通过重分配异常值减轻量化难度，但不能消除异常值。PTS 用 power-of-two 因子直接缩放异常通道，通过位移操作以最小计算开销实现。配合投票算法（Voting Algorithm）从小校准集中鲁棒选择缩放因子，避免过拟合。

### 损失函数 / 训练策略

- 逐层优化等价缩放因子，借鉴 AdaRound 的层级优化策略
- 学习完缩放因子后使用 BRECQ 进行权重量化
- PTS 仅选择性应用于高层间方差的层（如 skip connection），不引入全局开销
- 投票算法的共识阈值 $\kappa$ 控制缩放的保守程度

## 实验关键数据

### 主实验

无条件图像生成（FFHQ 256×256, LDM-4）：

| 方法 | 比特(W/A) | FID↓ | sFID↓ |
|------|----------|------|-------|
| 全精度 | 32/32 | 31.34 | 25.88 |
| Q-Diffusion | 4/8 | 36.17 | 28.75 |
| TFMQ-DM | 4/8 | 36.08 | 33.06 |
| **DMQ (Ours)** | **4/8** | **30.37** | **22.72** |
| Q-Diffusion | 4/6 | 71.16 | 75.70 |
| TFMQ-DM | 4/6 | 29.76 | 27.07 |
| **DMQ (Ours)** | **4/6** | **26.38** | **20.01** |

类条件生成（ImageNet 256×256, LDM-4）：

| 方法 | 比特(W/A) | IS↑ | FID↓ | sFID↓ | LPIPS↓ |
|------|----------|-----|------|-------|--------|
| 全精度 | 32/32 | 366.8 | 11.34 | 7.81 | — |
| TFMQ-DM | 4/8 | 342.1 | 9.51 | 8.10 | 0.181 |
| **DMQ (Ours)** | **4/8** | **350.8** | **9.68** | **7.19** | **0.124** |
| TFMQ-DM | 4/6 | 225.6 | 9.61 | 10.19 | 0.336 |
| **DMQ (Ours)** | **4/6** | **320.6** | **7.81** | **7.26** | **0.194** |

### 消融实验

组件逐步叠加效果（FFHQ W4A8）：

| 方法 | FID↓ | sFID↓ | 说明 |
|------|------|-------|------|
| 全精度 | 31.34 | 25.88 | 上界 |
| Baseline | 36.08 | 33.06 | MinMax量化 |
| +LES | 33.46 | 26.29 | 等价缩放分配异常值 |
| +Timestep Weighting | 31.83 | 24.39 | 自适应时间步加权 |
| **+PTS** | **30.37** | **22.72** | Power-of-Two缩放 |

投票算法 vs MSE选择（FFHQ W4A8）：

| PTS因子选择 | 应用范围 | FID↓ | sFID↓ |
|------------|---------|------|-------|
| MSE-based | 所有层 | 33.87 | 25.40 |
| MSE-based | Skip层 | 32.35 | 25.07 |
| **投票算法** | **Skip层** | **30.37** | **22.72** |

### 关键发现

- W4A6 是扩散模型量化的"硬骨头"，现有方法通常失败（FID 飙升），而 DMQ 首次实现稳定量化
- 早期时间步量化误差虽小，但因误差累积对最终质量影响巨大——简单地优先早期步并非最优策略
- Skip connection 层的层间方差远高于其他层，是量化的主要瓶颈
- 投票算法比直接 MSE 优化更鲁棒，因为小校准集下 MSE 容易过拟合

## 亮点与洞察

- **LES + PTS 的互补设计**：LES 做精细调整（浮点缩放因子，全局使用），PTS 做粗糙但有效的去除（power-of-two，仅用于极端层），两者分工明确
- **自适应时间步加权**：不同于简单的早期优先策略，通过动态累积损失实现自适应，兼顾了不同层在不同时间步的异质行为
- **投票算法**：巧妙地用统计共识替代直接优化来选择离散缩放因子，解决了小样本过拟合问题
- **实用性强**：$\tau$ 融合到量化尺度中无额外推理开销，PTS 的位移操作硬件友好

## 局限与展望

- 目前主要在 UNet 架构（Stable Diffusion）上验证，DiT 架构的适用性虽有讨论但实验较少
- PTS 仅应用于 skip connection 层，其他存在异常值的层可能也能受益
- 校准集大小对投票算法的影响值得进一步研究
- 可探索与 QAT 方法的结合，在更低比特（如 W3A4）下实现量化

## 相关工作与启发

- SmoothQuant 在 LLM 量化中的成功启发了等价缩放思路，但需针对扩散模型特点（激活远大于权重、迭代误差累积）做重要调整
- 与 ViDiT-Q 等 DiT 量化方法不同，LES 学习**静态**缩放因子可融合到权重中，而非随时间步变化
- 启示：扩散模型量化需要同时考虑空间维度（哪些层/通道有异常值）和时间维度（哪些时间步的误差更关键）

## 评分

- 新颖性: ⭐⭐⭐⭐ LES+PTS 的组合设计新颖，自适应时间步加权有理论深度
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多架构、多比特设定的全面评测，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 分析透彻，从问题到方案的逻辑链清晰
- 价值: ⭐⭐⭐⭐⭐ 首次实现 W4A6 稳定量化，对扩散模型部署有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](../../CVPR2025/image_generation/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](../../CVPR2025/image_generation/finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[CVPR 2025\] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](../../CVPR2025/image_generation/dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)
- [\[ICML 2025\] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](../../ICML2025/image_generation/modulated_diffusion_accelerating_generative_modeling_with_modulated_quantization.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](../../CVPR2026/image_generation/segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)

</div>

<!-- RELATED:END -->
