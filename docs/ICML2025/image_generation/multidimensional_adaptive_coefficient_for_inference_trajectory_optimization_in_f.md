---
title: >-
  [论文解读] Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion
description: >-
  [ICML2025][图像生成][Flow Matching] 提出多维自适应系数 MAC（Multidimensional Adaptive Coefficient），作为 flow/diffusion 模型的即插即用模块，将传统的一维时间调度系数扩展为多维、样本自适应的系数，通过对抗训练优化推理轨迹…
tags:
  - "ICML2025"
  - "图像生成"
  - "Flow Matching"
  - "扩散模型"
  - "推理轨迹优化"
  - "多维自适应系数"
  - "对抗训练"
---

# Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion

**会议**: ICML2025  
**arXiv**: [2404.14161](https://arxiv.org/abs/2404.14161)  
**代码**: 待确认  
**领域**: Flow模型加速 / image_generation  
**关键词**: Flow Matching, Diffusion加速, 推理轨迹优化, 多维自适应系数, 对抗训练

## 一句话总结

提出多维自适应系数 MAC（Multidimensional Adaptive Coefficient），作为 flow/diffusion 模型的即插即用模块，将传统的一维时间调度系数扩展为多维、样本自适应的系数，通过对抗训练优化推理轨迹，在 CIFAR-10 条件生成上以 5 NFE 取得 FID 1.37 的 SOTA 结果。

## 研究背景与动机

Flow 和 diffusion 模型在生成任务中展现了优异性能和训练稳定性，但相比基于仿真的方法（如 NeuralODE）缺少两个关键属性：

**维度自由度**：传统 flow/diffusion 中的插值系数 $\alpha_0(t), \alpha_1(t) \in \mathbb{R}$ 是标量，对所有数据维度施加相同的时间调度

**轨迹自适应性**：推理时所有样本共享相同的步长和轨迹方向，无法根据不同样本动态调整

现有的轨迹优化方法（如 Rectified Flow 的直线性约束、OT 配对）都是预定义最优性标准，且在推理计划上缺乏维度灵活性。本文的动机是将仿真方法的优势融入 flow/diffusion 框架，同时保持训练效率。

## 方法详解

### 核心思路：从一维系数到多维自适应系数

传统 flow/diffusion 的插值路径为：

$$x(t) = \alpha_0(t) x_0 + \alpha_1(t) x_1, \quad \alpha_0(t), \alpha_1(t) \in \mathbb{R}$$

MAC 将系数从标量扩展为对角矩阵 $\gamma(t) \in \mathbb{R}^{d \times 2}$，允许不同维度拥有不同的时间调度：

$$x(t) = \gamma_0(t) \odot x_0 + \gamma_1(t) \odot x_1, \quad \gamma_0(t), \gamma_1(t) \in \mathbb{R}^d$$

进一步引入参数化的 MAC $\gamma_\phi(t, \mathbf{x}_{\theta,\phi}^{\mathcal{S}})$，使系数能根据不同推理轨迹自适应调整。

### MAC 的参数化设计

MAC 使用加权正弦基函数（类似 Fourier 展开）建模：

- **基函数**：$b_m(t) = \sin(\pi m (t/T)^{1/q})$
- **权重网络**：$w_\phi(x_T) = s \cdot \text{LPF} \circ \tanh(\text{nn}_\phi(x_T))$，其中 $\text{nn}_\phi$ 为 U-Net
- **低通滤波 LPF**：高斯卷积排除高频噪声，保证系数平滑
- **tanh 约束**：输出范围限制在 $(-1, 1)$，支持预训练时直接从均匀分布采样权重

关键设计：$\gamma_\phi$ 在 $t=T$ 时只需一次前向传播即可计算整个推理调度。

### 推理轨迹优化

**问题一（仅优化 MAC）**：固定向量场 $\theta$，优化 $\phi^* = \arg\min_\phi \mathbb{D}(\rho_0, \hat{\rho}_{0,\theta,\phi})$

**问题二（联合优化）**：同时优化向量场和推理计划 $\theta^*, \phi^* = \arg\min_{\theta,\phi} \mathbb{D}(\rho_0, \hat{\rho}_{0,\theta,\phi})$

### 对抗训练策略

使用 hinge loss + StyleGAN-XL 判别器 $D_\psi$，三组损失函数分别更新：

- $\mathcal{L}_\phi$：通过仿真 $G_{\theta,\phi}$ 的完整推理过程，对抗优化 MAC 参数
- $\mathcal{L}_\theta$：对抗优化向量场模型 $H_\theta$
- $\mathcal{L}_\psi$：优化判别器

### 可选的 γ-预训练

用随机采样的多维系数 $\gamma \sim \Gamma_h$ 预训练 $H_\theta$，使其适应多维输入。预训练阶段在 $t$ 较大时约束多维性，对抗阶段完全放开。此步骤可选——MAC 可直接兼容标准 $\alpha$ 预训练模型。

## 实验关键数据

### 2D 合成数据（仅优化 φ，冻结 θ）

| 方法 | Gaussian→8Gaussians (NFE=5) | Gaussian→Moons (NFE=5) |
|------|:---:|:---:|
| SI_α | 0.763 | 0.882 |
| SI_γ + opt φ（MAC） | **0.721** | **0.682** |
| SI_α^OT | 0.457 | 0.245 |
| SI_γ^OT + opt φ（MAC） | **0.399** | **0.230** |

仅优化 MAC 即可在所有配置下降低 $\mathcal{W}_2$ 距离。

### γ-预训练效果（CIFAR-10 FID↓）

| 方法 | NFE=100 | NFE=200 |
|------|:---:|:---:|
| SI_α | 4.75 | 4.30 |
| SI_γ | **3.98** | **3.63** |
| FM_α | 4.52 | 4.07 |
| FM_γ | **3.59** | **3.42** |

多维预训练在所有框架（SI/FM/DDPM）上一致提升。

### CIFAR-10 SOTA 对比

| 模型 | NFE | FID uncond. | FID cond. |
|------|:---:|:---:|:---:|
| EDM_α（Karras 2022） | 35 | 1.98 | 1.79 |
| CTM_α + adv θ（Kim 2024） | 2 | 1.87 | 1.63 |
| **EDM_γ + adv θ,φ（MAC，本文）** | **5** | **1.69** | **1.37** |

CIFAR-10 条件生成 FID **1.37**（5 NFE），刷新 SOTA。

### ImageNet-64 条件生成

| 模型 | NFE | FID | FD_DINOv2 |
|------|:---:|:---:|:---:|
| CTM_α + adv θ | 2 | 1.73 | 157.7 |
| **EDM_α + adv θ,φ（MAC）** | **5** | **1.48** | **70.2** |

FD_DINOv2 显著优于 CTM（70.2 vs 157.7），说明 MAC 在感知质量上优势更大。

### 自适应性消融（CIFAR-10，10 NFE）

| γ_φ 的条件输入 | SI_γ FID | DDPM_γ FID |
|------|:---:|:---:|
| 常数 $\mathbf{1}_d$（无自适应） | 7.84 | 26.09 |
| 随机 $z \sim \rho_T$ | 6.48 | 23.31 |
| 实际起点 $x_T \sim \rho_T$ | **4.14** | **10.04** |

以实际推理起点作为 MAC 输入时效果最佳，验证了样本自适应的重要性。

## 亮点与洞察

1. **视角新颖**：提出推理轨迹最优性不应由预定义标准（如直线性）决定，而应通过仿真后的最终生成质量来衡量
2. **即插即用**：MAC 作为模块可兼容 DDPM/FM/EDM/SI 等任意 flow/diffusion 框架，无需修改骨干网络结构
3. **训练高效**：MAC 网络参数量远小于主模型，γ_φ 仅需 1 NFE 即可计算完整推理调度
4. **搜索空间扩展**：从标量调度扩展到维度级自适应调度，允许非线性弯曲轨迹和样本级自适应步长
5. **理论优雅**：通过 Fourier 基+低通滤波的设计排除了粗糙系数，假设集设计合理

## 局限与展望

1. **FID vs 感知质量**：FID 在 NFE 极少（如 4 步）时性能骤降（DDPM_γ 在 4 NFE 时 FID=72.64），距离实用的 1-2 步生成仍有差距
2. **判别器依赖**：使用 StyleGAN-XL 判别器进行对抗训练，引入了额外的训练复杂度和超参数调优负担
3. **分辨率限制**：实验最高仅在 64×64 分辨率验证，对更高分辨率（256/512）的可扩展性未知
4. **对角矩阵简化**：为计算效率将 $\gamma$ 限制为对角矩阵，可能丢失了维度间的交互信息
5. **U-Net 依赖**：MAC 网络使用 U-Net 架构，未探索更轻量的替代方案

## 相关工作与启发

- **Rectified Flow / OT 配对**：以直线性作为预定义最优标准，MAC 提供了不依赖预定义标准的替代路径
- **CTM (Kim et al., 2024)**：对抗蒸馏的直接竞争对手，MAC 在 5 NFE 下以更好的 FID 和 FD_DINOv2 超越
- **NeuralODE**：MAC 的理论动机源于将仿真方法的维度自由度和自适应性引入 flow/diffusion
- **启发**：该工作暗示 flow/diffusion 的轨迹最优性是一个尚未充分探索的方向，可能推动后续关于自适应采样策略的研究

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多维自适应系数概念新颖，从系数设计角度优化推理轨迹是较少探索的方向
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 4 种框架、4 个数据集、多种消融，SOTA 结果有说服力
- 写作质量: ⭐⭐⭐⭐ — 数学形式化清晰，符号统一，图示直观
- 价值: ⭐⭐⭐⭐ — 即插即用特性和训练效率使其具有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[CVPR 2025\] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](../../CVPR2025/image_generation/rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](sada_stability-guided_adaptive_diffusion_acceleration.md)

</div>

<!-- RELATED:END -->
