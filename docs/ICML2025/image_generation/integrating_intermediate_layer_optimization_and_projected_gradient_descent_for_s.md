---
title: >-
  [论文解读] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models
description: >-
  [ICML2025][图像生成][逆问题] 提出 DMILO 和 DMILO-PGD 两种方法，通过中间层优化（ILO）分解扩散模型采样过程以大幅降低显存，并结合投影梯度下降（PGD）避免次优收敛，在线性和非线性逆问题上全面超越 DMPlug 等 SOTA 方法。 逆问题（Inverse Problems）的目标是从噪声观测…
tags:
  - "ICML2025"
  - "图像生成"
  - "逆问题"
  - "扩散模型"
  - "中间层优化"
  - "投影梯度下降"
  - "图像重建"
  - "ILO"
  - "PGD"
---

# Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models

**会议**: ICML2025  
**arXiv**: [2505.20789](https://arxiv.org/abs/2505.20789)  
**代码**: [StarNextDay/DMILO](https://github.com/StarNextDay/DMILO)  
**领域**: 逆问题求解 / 扩散模型  
**关键词**: 逆问题, 扩散模型, 中间层优化, 投影梯度下降, 图像重建, ILO, PGD

## 一句话总结

提出 DMILO 和 DMILO-PGD 两种方法，通过中间层优化（ILO）分解扩散模型采样过程以大幅降低显存，并结合投影梯度下降（PGD）避免次优收敛，在线性和非线性逆问题上全面超越 DMPlug 等 SOTA 方法。

## 研究背景与动机

逆问题（Inverse Problems）的目标是从噪声观测 $\boldsymbol{y} = \mathcal{A}(\boldsymbol{x}^*) + \boldsymbol{\epsilon}$ 中恢复信号 $\boldsymbol{x}^*$，广泛应用于医学成像、压缩感知和遥感等领域。扩散模型（DM）作为强大的生成先验已在逆问题求解中取得 SOTA 表现，但现有方法存在两大痛点：

**显存瓶颈**：DMPlug 等 CSGM 类方法需要在整个采样过程中保留完整计算图用于反向传播，随采样步数增加显存线性增长，4 步时即超出 RTX 4090 显存

**次优收敛**：依赖初始向量选择，容易陷入局部最优

此前的 ILO 方法虽在 GAN 上有效，但其层分解依赖特定网络架构，难以泛化。本文发现，扩散模型的采样过程天然由多个函数复合而成，非常适合 ILO 分解。

## 方法详解

### 核心思想：将扩散采样视为函数复合

扩散模型的生成过程可表示为 $N$ 个采样步的函数复合：

$$\mathcal{G}(\cdot) = g_1 \circ g_2 \circ \cdots \circ g_N(\cdot)$$

每个 $g_i$ 对应 DDIM 采样的一步：

$$g_i(\boldsymbol{x}) = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} \boldsymbol{x} + \sigma_{t_{i-1}} \left( \frac{\alpha_{t_{i-1}}}{\sigma_{t_{i-1}}} - \frac{\alpha_{t_i}}{\sigma_{t_i}} \right) \boldsymbol{x}_\theta(\boldsymbol{x}, t_i)$$

这种分解与去噪网络架构无关，可无缝适配任何 DM。

### DMILO：中间层优化 + 稀疏偏差

对第一层（与观测直接关联），优化：

$$\hat{\boldsymbol{x}}_{t_1}, \hat{\boldsymbol{\nu}}_{t_1} = \arg\min_{\boldsymbol{x}, \boldsymbol{\nu}} \| \boldsymbol{y} - \mathcal{A}(g_1(\boldsymbol{x}) + \boldsymbol{\nu}) \|_2^2 + \lambda \|\boldsymbol{\nu}\|_1$$

对后续每层，以上一层优化结果为目标：

$$\hat{\boldsymbol{x}}_{t_i}, \hat{\boldsymbol{\nu}}_{t_i} = \arg\min_{\boldsymbol{x}, \boldsymbol{\nu}} \| \hat{\boldsymbol{x}}_{t_{i-1}} - (g_i(\boldsymbol{x}) + \boldsymbol{\nu}) \|_2^2 + \lambda \|\boldsymbol{\nu}\|_1$$

其中 $\boldsymbol{\nu}$ 为稀疏偏差项（$\ell_1$ 正则），用于探索扩散模型范围之外的信号。每次只需保留单步梯度信息，显存恒定。

### DMILO-PGD：引入投影梯度下降

在 DMILO 基础上交替执行：

1. **梯度下降步**：$\boldsymbol{x}_{t_0}^{(e)} = \boldsymbol{x}_{t_0}^{(e-1)} - \eta \nabla \|\boldsymbol{y} - \mathcal{A}(\boldsymbol{x}_{t_0}^{(e-1)})\|_2^2$
2. **投影步**：用 DMILO 将更新后的信号投影回扩散模型的扩展范围

关键区别：投影时最小化 $\|\mathcal{A}(\mathcal{G}(\boldsymbol{x}_{t_N})) - \mathcal{A}(\hat{\boldsymbol{x}}_{t_0})\|_2^2$，利用前向算子 $\mathcal{A}$ 引导投影方向，而非传统 PGD 的纯距离投影，理论上保证更优重建。

### 理论保证

在 Lipschitz 连续和低维流形假设下，Theorem 4.4 保证当测量数 $m = \Omega(k_2 \log \frac{L_1 n}{\delta} + k^2 \log(3n))$ 时，利用前向算子的测量最优解接近真实最优解：

$$\|g_1(\hat{\boldsymbol{x}}_1) - \boldsymbol{x}^*\|_2 \leq \left(1 + \frac{3}{\gamma}\right) \|g_1(\bar{\boldsymbol{x}}_1) - \boldsymbol{x}^*\|_2 + \frac{\delta}{\gamma}$$

## 实验关键数据

实验覆盖 CelebA、FFHQ、LSUN-bedroom、ImageNet，含 4 种线性任务和 2 种非线性任务。

### 显存对比（RTX 4090，模型 2.75GB）

| 采样步数 | DMPlug | DMILO | DMILO-PGD |
|---------|--------|-------|-----------|
| 1 | 10.53 GB | 10.53 GB | 10.53 GB |
| 2 | 15.72 GB | 10.53 GB | 10.54 GB |
| 3 | 20.83 GB | 10.53 GB | 10.54 GB |
| 4 | N/A (OOM) | 10.54 GB | 10.54 GB |

### 超分辨率 & 修复（CelebA, σ=0.01）

| 方法 | SR PSNR↑ | SR SSIM↑ | Inpaint PSNR↑ | Inpaint SSIM↑ |
|------|---------|---------|--------------|--------------|
| DMPlug | 32.38 | 0.875 | 35.51 | 0.935 |
| DCPS | 29.47 | 0.834 | 35.42 | 0.940 |
| DMILO-PGD | **33.58** | **0.906** | **36.42** | **0.952** |

### 运动去模糊（CelebA, σ=0.01）

| 方法 | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|------|--------|-------|-------|
| DMPlug | 78.57 | 0.164 | 30.25 | 0.824 |
| DCPS | 35.19 | 0.054 | 31.05 | 0.856 |
| DMILO | **31.08** | **0.044** | **34.15** | **0.908** |

### 非线性去模糊（FFHQ, σ=0.01）

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|-------|
| DMPlug | 0.099 | 31.37 | 0.866 |
| DMILO-PGD | **0.047** | **34.02** | **0.919** |

## 亮点与洞察

1. **自然分解**：扩散采样过程的函数复合结构天然适配 ILO，无需依赖特定架构，这一观察非常优雅
2. **显存恒定**：无论采样步数如何增加，DMILO 显存几乎不变（~10.5 GB），而 DMPlug 线性增长直至 OOM
3. **稀疏偏差扩展范围**：$\ell_1$ 正则允许探索 DM 生成范围之外的信号，对于真实信号不完全在生成分布内的场景很关键
4. **前向算子引导投影**：DMILO-PGD 中利用 $\mathcal{A}$ 引导投影比纯距离投影效果更好，有直观的理论支撑
5. **非线性任务大幅领先**：PSNR 提升 2.5–3.5 dB，显示方法在复杂退化下的优势

## 局限与展望

1. **PGD 对盲去模糊效果有限**：DMILO-PGD 在 BID 任务上不如 DMILO，作者推测是朴素梯度更新不适合核估计，需要设计专门的核更新策略
2. **计算效率未充分讨论**：虽然显存降低，但多轮外迭代 × 多轮内迭代的计算量可能很大（如超分辨率需 400 内迭代 × 10 外迭代）
3. **高斯去模糊在 ImageNet 上表现一般**：FID 和 LPIPS 在某些配置下劣于 DCPS，泛化性有待加强
4. **超参数较多**：$\lambda$、内外学习率、内外迭代次数等超参数需针对不同任务分别调优
5. **仅验证了 DDIM 采样**：未探索与其他采样器（DPM-Solver 等）的结合

## 相关工作与启发

- **DMPlug** (Wang et al., 2024)：本文直接改进的基线，优化初始隐变量的 CSGM 方法
- **ILO** (Daras et al., 2021)：中间层优化最初为 GAN 设计，本文将其自然推广到 DM
- **稀疏偏差** (Dhar et al., 2018)：允许生成范围外的信号探索
- **PGD for IP** (Shah & Hegde, 2018)：投影梯度下降框架
- **DCPS** (Janati et al., 2024)：另一个强竞争对手，在某些任务上与本文方法各有胜负

## 评分

- 新颖性: ⭐⭐⭐⭐ — ILO 与 DM 的结合自然且有效，前向算子引导投影有理论新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 种任务、4 个数据集、9+ 个 baseline，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论与实验衔接好
- 价值: ⭐⭐⭐⭐ — 显存问题的解决对实际部署有重要意义，但计算效率仍有提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](learning_single_index_models_with_diffusion_priors.md)
- [\[AAAI 2026\] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes](../../AAAI2026/image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](../../CVPR2025/image_generation/improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)

</div>

<!-- RELATED:END -->
