---
title: >-
  [论文解读] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models
description: >-
  [ECCV 2024][图像生成] 提出SPDInv——一种源提示解耦反演方法，通过将反演过程建模为不动点搜索问题并利用预训练扩散模型求解，使反演噪声码与源提示解耦，显著提升基于文本驱动的图像编辑质量。
tags:
  - ECCV 2024
  - 图像生成
---

# Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2403.11105](https://arxiv.org/abs/2403.11105)  
**领域**: 图像生成

## 一句话总结

提出SPDInv——一种源提示解耦反演方法，通过将反演过程建模为不动点搜索问题并利用预训练扩散模型求解，使反演噪声码与源提示解耦，显著提升基于文本驱动的图像编辑质量。

## 研究背景与动机

- 文本驱动图像编辑的核心管线：给定源图像 → 通过反演得到潜在噪声码 → 使用目标提示编辑
- DDIM反演是最常用的方法，但存在根本缺陷：**反演噪声码与源提示紧密耦合**
- 现有改进方法（NTI、NPI、DirectInv等）均聚焦于缩小**重建误差** $D_{Rec}$，却忽略了**噪声误差** $D_{Noi}$
- 源提示耦合的噪声码在被目标提示编辑时产生冲突，导致编辑伪影和内容不一致
- **关键洞察**：理想的反演应满足不动点约束 $z_t = C_{t,1} \cdot z_{t-1} + C_{t,2} \cdot \epsilon_\theta(z_t, t, c)$，但DDIM反演用 $(z_{t-1}, t-1)$ 替代了 $(z_t, t)$ 作为网络输入，引入了源提示先验

## 方法详解

### 整体框架

SPDInv的核心思想：**最小化噪声误差 $D_{Noi}$（而非重建误差 $D_{Rec}$）**，使反演噪声码尽可能独立于源提示。

1. 用标准DDIM反演获取初始近似 $z_t$
2. 在每个反演步骤中，将不动点约束转化为优化问题
3. 利用预训练扩散模型的梯度下降求解

### 关键设计

**1. 不动点约束分析**

理想反演方程（Eq.1）要求网络输入为 $(z_t, t)$，但DDIM反演实际使用 $(z_{t-1}, t-1)$。由于 $z_{t-1}$ 是由源提示条件去噪 $z_t$ 得到的，这导致源提示信息被注入到 $z_t$ 中。

将理想反演转化为不动点问题：
$$x = f_\theta(x), \quad x = z_t, \quad f_\theta(x) = C_{t,2} \cdot \epsilon_\theta(x, t, c) + C_{t,1} \cdot z_{t-1}$$

**2. 基于梯度下降的不动点搜索**

- 不同于AIDI的固定轮次迭代（不稳定且次优），SPDInv将不动点约束重写为损失函数：
$$L = \|f_\theta(z_t) - z_t\|_2$$
- 利用预训练扩散模型（固定参数）通过梯度下降优化 $z_t$：
$$z_t := z_t - \eta \nabla L$$
- 引入阈值 $\delta$ 控制提前终止：早期反演步骤需要更多优化轮次，后期（$t > T/2$）几轮即可收敛

**3. 扩展到定制化图像生成**

将SPDInv集成到ELITE等定制化图像生成方法中：
1. 将给定图像通过ELITE转换为文本嵌入空间（"S*"）
2. 使用SPDInv反演图像得到噪声码（保持布局和背景）
3. 用新文本提示（如"a white S*"）生成编辑结果

### 损失函数

每个反演步骤的优化目标：
$$\arg\min_{z_t} L = \|f_\theta(z_t) - z_t\|_2$$

其中 $f_\theta(z_t) = C_{t,2} \cdot \epsilon_\theta(z_t, t, c) + C_{t,1} \cdot z_{t-1}$，预训练网络参数固定，仅更新潜在特征 $z_t$。

## 实验关键数据

### 主实验

在PIE-Bench上不同反演方法在三种编辑引擎下的对比：

| 反演方法 | 编辑引擎 | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ | 时间(s) |
|---------|---------|------------|-------|-------------|------------|------------|------|--------|
| DDIM | P2P | 69.43 | 17.87 | 208.80 | 219.88 | 71.14 | 25.01 | 11.55 |
| NTI | P2P | 13.44 | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 137.54 |
| DirectINV | P2P | 11.65 | 27.22 | 54.55 | 32.86 | 84.76 | 25.02 | 19.94 |
| AIDI | P2P | 12.16 | 27.01 | 56.39 | 36.90 | 84.27 | 24.92 | 87.21 |
| **SPDInv** | **P2P** | **8.81** | **28.60** | **36.01** | **24.54** | **86.23** | **25.26** | 27.04 |
| DDIM | MasaCtrl | 28.38 | 22.17 | 106.62 | 86.97 | 79.67 | 23.96 | 11.55 |
| DirectINV | MasaCtrl | 24.70 | 22.64 | 87.94 | 81.09 | 81.33 | 24.38 | 19.94 |
| **SPDInv** | **MasaCtrl** | **20.48** | **24.12** | **71.74** | **64.77** | **82.54** | **24.61** | 27.04 |
| DDIM | PNP | 28.22 | 22.28 | 113.33 | 83.51 | 79.00 | 24.95 | 11.55 |
| **SPDInv** | **PNP** | **15.58** | **26.72** | **91.55** | **34.69** | **82.04** | **25.14** | 27.04 |

相比第二优方法，SPDInv在P2P引擎下：DINO提升24%、LPIPS降低21%、MSE降低13%。

### 消融实验

超参数对性能的影响（PIE-Bench子集）：

| 超参数 | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ |
|-------|------------|-------|-------------|------------|------------|------|
| K=5 | 8.52 | 31.49 | 22.31 | 10.42 | 90.21 | 26.70 |
| **K=25** | **8.43** | **31.61** | **21.70** | **10.12** | **90.28** | — |
| K=50 | 更优 | 更优 | 更优 | 更优 | 更优 | 略降 |
| δ=5e-4 | 较差 | 较差 | 较差 | 较差 | 较差 | — |
| **δ=5e-6** | **最优** | **最优** | **最优** | **最优** | **最优** | — |
| η=0.005 | 略差 | 略差 | 略差 | 略差 | 略差 | 略升 |
| **η=0.001** | **最优** | **最优** | **最优** | **最优** | **最优** | — |

定制化图像编辑结果（SPDInv-ELITE vs原始ELITE）：

| 方法 | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ |
|------|------------|-------|-------------|------------|------------|------|
| ELITE（原始） | 148.37 | 14.83 | 201.94 | 359.58 | 67.62 | 15.72 |
| BlendDM | 59.21 | 15.51 | 244.07 | 306.75 | 67.45 | 20.21 |
| InstructP2P | 155.49 | 18.19 | 161.20 | 362.01 | 78.05 | 20.06 |
| **SPDInv-ELITE** | **21.23** | **24.14** | **74.36** | **48.73** | **88.90** | **19.18** |

SPDInv-ELITE相比原始ELITE：DINO提升85%、PSNR提升62%、MSE降低86%。

### 关键发现

1. 不动点搜索方法（SPDInv）显著优于直接迭代（AIDI），验证了基于梯度的搜索策略的有效性
2. 反演过程前期（$t < T/2$）需要更多优化轮次来满足不动点约束，后期收敛极快
3. SPDInv是即插即用的：仅需修改约10行代码即可集成到现有编辑管线
4. 在三种不同编辑引擎（P2P、MasaCtrl、PNP）下均表现一致优势，证明了方法的通用性

## 亮点与洞察

- 问题定义精准：从"重建误差"转向"噪声误差"的思路转变是这篇论文最大的洞察
- 理论推导简洁有力：从理想反演到不动点约束再到优化问题，逻辑链完整
- 极高的工程实用性：10行代码修改、支持多种编辑引擎、可扩展到定制化生成
- 反演时间（27秒）远低于NTI（138秒）和AIDI（87秒），同时效果最好
- 将定制化生成方法（如ELITE）赋予局部编辑能力的扩展方向很有价值

## 局限性

- 基于Stable Diffusion v1.4，对更新版本的模型（如SD-XL、SD3）的兼容性未验证
- 每个反演步骤可能需要最多K=25次前向传播，对于需要实时编辑的场景仍有开销
- 不动点搜索的收敛性缺乏严格的理论保证，依赖经验设置的超参数
- 对于极大幅度的编辑（如完全改变场景），源提示解耦的效果有待验证

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 从噪声误差out发重新定义问题，不动点搜索思路新颖
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用、10行代码、多引擎兼容、可扩展
- **实验充分性**: ⭐⭐⭐⭐⭐ — 三种引擎×两种基准数据集、充分消融、定制化编辑扩展
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰，但公式符号较多，部分推导可更简洁

<!-- RELATED:START -->

## 相关论文

- [ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)

<!-- RELATED:END -->
