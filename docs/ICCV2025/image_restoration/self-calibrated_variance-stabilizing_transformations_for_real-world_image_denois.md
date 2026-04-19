---
title: >-
  [论文解读] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising
description: >-
  [ICCV 2025][图像恢复][图像去噪] 提出 Noise2VST 框架，通过自监督学习一个无模型假设的方差稳定化变换（VST），使现成的高斯去噪器无需额外训练即可高效处理真实世界噪声图像。
tags:
  - ICCV 2025
  - 图像恢复
  - 图像去噪
  - 方差稳定化变换
  - 零样本学习
  - 盲点去噪
  - 样条建模
---

# Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising

**会议**: ICCV 2025  
**arXiv**: [2407.17399](https://arxiv.org/abs/2407.17399)  
**代码**: [GitHub](https://github.com/sherbret/Noise2VST)  
**领域**: 图像修复  
**关键词**: 图像去噪, 方差稳定化变换, 零样本学习, 盲点去噪, 样条建模

## 一句话总结

提出 Noise2VST 框架，通过自监督学习一个无模型假设的方差稳定化变换（VST），使现成的高斯去噪器无需额外训练即可高效处理真实世界噪声图像。

## 研究背景与动机

深度学习在图像去噪领域取得了显著成功，但主流方法高度依赖特定场景的训练数据。针对高斯噪声训练的去噪网络在真实场景中表现不佳，因为真实噪声来源复杂（光子散粒噪声、读出噪声等），不遵循简单的高斯分布。当前解决方案面临以下困境：

**数据依赖问题**：需要采集大量特定场景的 clean/noisy 图像对来训练专用模型，这在很多应用场景（如医学成像、天文成像）中极其耗时甚至不可行

**传统 VST 的局限**：经典方差稳定化变换（如 Anscombe 变换、GAT 变换）虽然可以将非高斯噪声映射为近似高斯噪声，但需要事先知道噪声分布的参数化形式，参数估计不准确会导致性能严重下降

**无监督方法的不足**：现有无需真值的去噪方法（Noise2Noise、盲点去噪等）虽然不需要 clean 图像，但性能仍落后于有监督方法

本文的核心洞察是：**预训练的高斯去噪器蕴含了丰富的信号先验知识**，如果能找到合适的变换将真实噪声映射为高斯噪声，就可以直接复用这些强大的去噪器。关键问题在于如何不假设噪声模型地学习这样的变换。

## 方法详解

### 整体框架

Noise2VST 的流程如下：给定一张含噪图像 $\boldsymbol{z}$，首先学习一个逐像素的 VST $f_{\boldsymbol{\theta}}$ 将其变换到高斯噪声域，使用现成的高斯去噪器 $D$ 进行去噪，然后通过学习到的逆变换 $f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta}$ 回到原始域，整个过程可表示为：

$$\hat{\boldsymbol{s}} = (f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta} \circ D \circ f_{\boldsymbol{\theta}})(\boldsymbol{z})$$

训练阶段使用盲点去噪器 $\bar{D}$（权重冻结），推理阶段替换为标准去噪器 $D$ 以获得更好效果。

### 关键设计

1. **连续分段线性（CPWL）VST 建模**:

    - 功能：用增函数族中的分段线性样条来建模方差稳定化变换
    - 核心思路：将 VST $f_\theta$ 参数化为 $n=128$ 个节点的 CPWL 函数。横坐标 $x_i$ 均匀分布在 $[z_{\min}, z_{\max}]$，纵坐标通过 $y_i = \theta_1 + \sum_{j=2}^{i} \exp(\theta_j)$ 参数化，保证严格单调递增。逆变换设计为 $f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta}(z) = f_{\boldsymbol{\theta}}^{-1}(z) + \alpha z + \beta$，其中仿射项用于修正代数逆变换的偏差
    - 设计动机：样条函数是通用逼近器，且单调性保证了像素序关系；总共仅 $n+2=130$ 个可学习参数，极大降低了过拟合风险

2. **盲点自监督训练策略**:

    - 功能：利用盲点去噪器的性质实现完全自监督的 VST 学习
    - 核心思路：盲点去噪器 $\bar{D}$ 对每个像素的输出不依赖该像素本身的值。由于组合 $f^{\text{inv}} \circ \bar{D} \circ f_\theta$ 保持盲点性质，在噪声空间独立的假设下，自监督损失与真值损失仅差一个常数：$\mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{z}) = \mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{s}) + \text{const}$。因此无需任何 clean 图像即可优化 VST 参数
    - 设计动机：盲点方法的理论保证确保了在仅有一张噪声图像的零样本设置下，VST 的学习仍然是有效的；130 个参数足够少，避免了过拟合

3. **推理阶段可见盲点替换**:

    - 功能：训练时使用盲点去噪器，推理时替换为标准去噪器
    - 核心思路：最优 VST 只取决于噪声分布而非去噪器类型。因此训练好的 VST 可以搭配任意高斯去噪器使用。推理时使用 DRUNet 等标准去噪器，避免了盲点去噪器的棋盘伪影问题
    - 设计动机：盲点去噪器故意不使用目标像素信息，导致性能受限；还可以训练时使用快速的盲点去噪器（如基于 FFDNet 的 Noise2VST†），推理时用更强的标准去噪器

### 损失函数 / 训练策略

- **损失函数**：$\ell_2$ 自监督损失 $\mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{z}) = \|(f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta} \circ \bar{D} \circ f_{\boldsymbol{\theta}})(\boldsymbol{z}) - \boldsymbol{z}\|_2^2$
- **优化器**：Adam，初始学习率 0.01，在 1/3 和 2/3 时衰减 10 倍
- **训练细节**：在 $64 \times 64$ 随机裁剪 patch 上训练，batch size 4，数据增强包含随机翻转和 90° 旋转，总迭代次数 2000（raw-RGB 数据为 5000）
- **去噪器选择**：DRUNet（非盲，需指定噪声等级 $\sigma=25/255$）；快速替代版本使用 FFDNet

## 实验关键数据

### 主实验

**合成泊松噪声（sRGB 空间）**

| 方法 | KODAK PSNR/SSIM | BSD300 PSNR/SSIM | SET14 PSNR/SSIM |
|------|----------------|-----------------|----------------|
| Baseline (N2C+GAT) | 31.63/0.865 | 29.92/0.850 | 30.66/0.854 |
| B2UNB | 31.07/0.857 | 29.92/0.852 | 30.10/0.844 |
| SST-GP | 31.39/0.872 | 29.96/0.853 | 30.22/0.848 |
| **Noise2VST** | **31.60/0.865** | 29.89/0.849 | **30.60/0.850** |

**真实世界噪声去噪（SIDD 数据集，raw-RGB 空间）**

| 方法 | SIDD Benchmark PSNR/SSIM | SIDD Validation PSNR/SSIM |
|------|--------------------------|---------------------------|
| Baseline (N2C) | 50.60/0.991 | 51.19/0.991 |
| B2UNB | 50.79/0.991 | 51.36/0.992 |
| SST-GP | 50.87/0.992 | 51.57/0.992 |
| **Noise2VST** | **51.07/0.991** | **51.66/0.992** |

### 消融实验

**计算效率对比**（256×256 图像）

| 方法 | GPU 时间 | CPU 时间 | 可训练参数数 |
|------|---------|---------|------------|
| S2S | 35 min | 4.5 hr | 1M |
| ZS-N2N | 20 sec | 1 min | 22k |
| Noise2VST | 50 sec | 5 min | 130 |
| Noise2VST† | **20 sec** | **40 sec** | 130 |

**荧光显微镜数据（FMD + W2S）**

| 方法 | FMD Confocal Fish | FMD Two-Photon Mice | W2S ch0 avg1 |
|------|-------------------|---------------------|--------------|
| B2UNB | 32.74/0.897 | 34.03/0.916 | - |
| Noise2VST | **32.88/0.904** | **34.06/0.926** | **35.65** |

### 关键发现

1. Noise2VST 与使用真实噪声参数的 oracle baseline（DRUNet+GAT）性能差距仅 0.06 dB 以内，证明学到的无模型 VST 接近最优
2. 在 SIDD benchmark 上超越所有无需真值训练的方法，甚至接近有监督基线
3. 快速版本 Noise2VST†（用 FFDNet）GPU 时间仅 20 秒，性能几乎不损失
4. 传统 GAT 方法在真实场景下性能较差，主要原因是参数估计不准和噪声模型过于简化

## 亮点与洞察

- **极简参数设计**：仅 130 个参数就能学到有效的 VST，这是对"多即是好"范式的有力反驳
- **复用预训练知识**：不训练新的去噪网络，而是通过学习变换来桥接高斯去噪器和真实噪声，充分利用了预训练模型的信号先验
- **零样本 + 高性能**：在不使用任何外部数据的情况下，达到了接近有监督方法的性能，在荧光显微镜数据上甚至超越
- **训练/推理解耦**：训练用盲点去噪器保证理论正确性，推理用标准去噪器提升质量，设计简洁巧妙

## 局限与展望

1. 假设噪声空间独立，对于 sRGB 空间中经过 demosaicking 的图像（噪声已产生空间相关性），效果可能受限
2. 每张图像都需要单独训练 VST，虽然时间可接受（~50 秒），但无法做到即时去噪
3. VST 为全局的像素级映射，未考虑空间变化的噪声特性
4. 分段线性函数虽是通用逼近器，但光滑性不如更高阶的样条

## 相关工作与启发

- **Noise2Noise / Noise2Void 系列**：盲点思路是本方法的理论基础
- **GAT / Anscombe 变换**：传统 VST 提供了方法论灵感，本文将其从参数化推广到无模型
- **DRUNet**：非盲高斯去噪器的泛化能力是方法成功的关键前提
- 启发：预训练去噪器的价值被低估，通过轻量级适配可以释放其在非高斯场景的潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 VST 从参数化模型推广到自监督无模型学习，130 个参数的极简设计极具创新
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖合成噪声、荧光显微镜、智能手机相机三类场景，对比方法全面
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰严谨，但符号较多需要仔细跟读
- 价值: ⭐⭐⭐⭐⭐ 零样本方法达到接近有监督性能，在资源受限场景具有重要实用价值

<!-- RELATED:START -->

## 相关论文

- [TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [Blind2Sound: Self-Supervised Image Denoising without Residual Noise](blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)
- [Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](../../CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)

<!-- RELATED:END -->
