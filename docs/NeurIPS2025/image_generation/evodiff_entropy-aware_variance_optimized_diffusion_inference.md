---
title: >-
  [论文解读] EVODiff: Entropy-aware Variance Optimized Diffusion Inference
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 从信息论角度分析扩散模型推理过程，提出通过优化条件方差来减少条件熵的 EVODiff 方法，在不修改模型的前提下显著加速采样并提升生成质量。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散模型
  - 推理加速
  - 信息论
  - 条件熵
  - 方差优化
---

# EVODiff: Entropy-aware Variance Optimized Diffusion Inference

**会议**: NeurIPS 2025  
**arXiv**: [2509.26096](https://arxiv.org/abs/2509.26096)  
**代码**: [ShiguiLi/EVODiff](https://github.com/ShiguiLi/EVODiff)  
**领域**: image_generation  
**关键词**: 扩散模型, 推理加速, 信息论, 条件熵, 方差优化

## 一句话总结

从信息论角度分析扩散模型推理过程，提出通过优化条件方差来减少条件熵的 EVODiff 方法，在不修改模型的前提下显著加速采样并提升生成质量。

## 背景与动机

扩散模型在图像生成中表现优异，但推理过程缓慢且存在训练-推理不一致问题。现有加速方法（如 DPM-Solver、UniPC 等）将去噪过程视为 ODE 求解，但缺乏信息论基础——忽略了信息传输效率这一关键因素。作者认为，成功的去噪本质上是减少反向转移中的条件熵，而现有方法并未从这一原理出发设计算法。

## 核心问题

1. 现有 ODE 求解器缺乏信息论指导，无法最优地恢复去噪过程中的信息
2. 数据预测参数化为何优于噪声预测参数化，缺乏理论解释
3. 条件方差优化在无参考情况下如何同时减少转移误差和重建误差

## 方法详解

### 信息论框架

将扩散推理视为条件熵减少过程。反向转移中相邻状态的互信息为：

$$I_p(\mathbf{x}_{t_i}; \mathbf{x}_{t_{i+1}}) = H_p(\mathbf{x}_{t_i}) - H_p(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}})$$

在高斯假设下，条件熵与条件方差直接相关：

$$H_p(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}}) \propto \log\det(\mathrm{Var}(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}}))$$

因此 **最小化条件方差等价于最大化信息传输效率**。

### 重建误差分解

将重建误差分解为方差项和偏差项：

$$\mathbb{E}_q[\|\mathbf{x}_{t_i} - \mathbf{x}_0\|^2] = \underbrace{\mathbb{E}_q[\|\mathbf{x}_{t_i} - \boldsymbol{\mu}_{t_i|t_{i+1}}\|^2]}_{\text{方差项}} + \underbrace{\mathbb{E}_q[\|\boldsymbol{\mu}_{t_i|t_{i+1}} - \mathbf{x}_0\|^2]}_{\text{偏差项}}$$

由于推理时无法获得真实 $\mathbf{x}_0$，优化条件方差成为唯一可操作的机制。

### 数据预测 vs 噪声预测

**定理 3.4**：数据预测参数化比噪声预测更有效地减少重建误差和条件熵。数据参数化直接以数据分布为目标，避免了 $\boldsymbol{\epsilon}_t \mapsto \mathbf{x}_t \mapsto \mathbf{x}_0$ 的误差累积链路。

### EVODiff 算法

基于数据预测的多步迭代，通过优化两个关键参数 $\zeta_i$ 和 $\eta_i$ 实现熵感知推理：

**步骤 1 — 统一迭代**：将显式和隐式改进统一为：

$$\frac{\mathbf{x}_{t_{i-1}}}{\sigma_{t_{i-1}}} - \frac{\mathbf{x}_{t_i}}{\sigma_{t_i}} = h_{t_i} \mathbf{x}_\theta(\mathbf{x}_{t_i}, t_i) + \frac{1}{2} h_{t_i}^2 \zeta_i \bar{B}_\theta(t_i; u_i)$$

**步骤 2 — 求解 $\zeta_i$**：通过最小化前向和反向估计差异，得到闭式解：

$$\zeta_i^* = -\frac{\text{vec}^T(D_i) \text{vec}(\tilde{P}_i)}{\sigma_{t_i} h_{t_i} \text{vec}^T(D_i) \text{vec}(D_i)}$$

**步骤 3 — 求解 $\eta_i$**：平衡隐式和显式梯度误差：

$$\eta_i^* = -\frac{\text{vec}^T(\tilde{B}_i) \text{vec}(B_\theta(t_i, l_i))}{\text{vec}^T(\tilde{B}_i) \text{vec}(\tilde{B}_i)}$$

**步骤 4 — 映射为可用参数**：

$$\eta_i = \text{Sigmoid}(|\eta_i^*|), \quad \zeta_i = \text{Sigmoid}(-(|\zeta_i^*| - \mu))$$

算法具有二阶全局收敛性，局部误差为 $\mathcal{O}(h_{t_i}^3)$。

## 实验关键数据

### CIFAR-10（EDM，50k 样本）

| 方法 | NFE=5 FID↓ | NFE=8 FID↓ | NFE=10 FID↓ | NFE=12 FID↓ |
|------|-----------|-----------|------------|------------|
| DPM-Solver++ | 27.96 | 8.40 | 5.10 | 3.70 |
| UniPC | 27.03 | 7.67 | 3.98 | 2.76 |
| **EVODiff** | **17.84** | **3.98** | **2.78** | **2.30** |

### FFHQ-64（EDM，50k 样本）

| 方法 | NFE=5 FID↓ | NFE=10 FID↓ | NFE=15 FID↓ | NFE=20 FID↓ |
|------|-----------|------------|------------|------------|
| DPM-Solver++ | 25.08 | 6.81 | 3.80 | 3.00 |
| UniPC | 28.87 | 6.65 | 3.40 | 2.69 |
| **EVODiff** | **19.65** | **5.31** | **3.04** | **2.66** |

### ImageNet-256（ADM，10k 样本）

| 方法 | NFE=5 FID↓ | NFE=10 FID↓ | NFE=15 FID↓ | NFE=20 FID↓ |
|------|-----------|------------|------------|------------|
| DPM-Solver++ | 16.62 | 8.68 | 7.80 | 7.51 |
| DPM-Solver-v3 | 14.92 | 8.14 | 7.70 | 7.42 |
| **EVODiff** | **13.98** | **8.14** | **7.48** | **7.25** |

- CIFAR-10 NFE=10 时 FID 从 5.10 降至 2.78，**降幅 45.5%**
- ImageNet-256 上将 NFE 从 20 降至 15 即可获得高质量样本，**节省 25% 计算**
- 文本到图像生成中也减少了伪影

## 亮点

- ⭐ 首次从信息论角度系统分析扩散推理，建立条件熵减少框架
- ⭐ 理论证明数据预测优于噪声预测参数化
- ⭐ 方差优化参数 $\zeta_i$、$\eta_i$ 具有闭式解，计算开销极小
- 无需额外训练或参考数据（不同于 DPM-Solver-v3）
- 统一解释了 DPM-Solver 和 EDM Heun 迭代的加速机制

## 局限与展望

- 假设去噪步间估计噪声的独立性，实际中共享参数可能引入依赖
- 理论分析主要基于高斯假设，对非高斯分布的适用性有待验证
- 仅关注确定性采样（ODE），尚未扩展到随机采样（SDE）路径
- 位移参数 $\mu$ 需手动调节

## 与相关工作的对比

| 特性 | DDIM | DPM-Solver | UniPC | DPM-Solver-v3 | EVODiff |
|------|------|------------|-------|---------------|---------|
| 梯度方法 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 需要参考 $\tilde{\mathbf{x}}_0$ | ✗ | ✗ | ✗ | ✓ | ✗ |
| 方差项优化 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 熵感知 | ✗ | ✗ | ✗ | ✗ | ✓ |

## 启发与关联

- 信息论视角可推广到视频生成、3D 生成等多步扩散场景
- 条件熵框架可能用于指导采样调度（time schedule）的自动设计
- 方差优化思想可与 consistency model、rectified flow 结合

## 评分

- 新颖性: ⭐⭐⭐⭐ (信息论视角切入点新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (多数据集、多模型、多 NFE 全面对比)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，符号体系统一)
- 价值: ⭐⭐⭐⭐ (即插即用的推理加速方法，实用性强)

<!-- RELATED:START -->

## 相关论文

- [Neural Entropy](neural_entropy.md)
- [Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)

<!-- RELATED:END -->
