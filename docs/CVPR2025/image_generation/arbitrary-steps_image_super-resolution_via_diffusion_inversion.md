---
title: >-
  [论文解读] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion
description: >-
  [CVPR 2025][图像生成][扩散反演] 本文提出InvSR，通过训练一个噪声预测网络来实现扩散反演（Diffusion Inversion），利用预训练扩散模型的图像先验进行超分辨率，支持1-5步任意步数采样，即使单步采样也能达到或超过现有SOTA方法的效果。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散反演
  - 图像超分辨率
  - 噪声预测
  - 部分噪声预测
  - 任意步采样
---

# Arbitrary-Steps Image Super-Resolution via Diffusion Inversion

**会议**: CVPR 2025  
**arXiv**: [2412.09013](https://arxiv.org/abs/2412.09013)  
**代码**: [https://github.com/zsyOAOA/InvSR](https://github.com/zsyOAOA/InvSR)  
**领域**: 扩散模型 / 图像超分辨率  
**关键词**: 扩散反演, 图像超分辨率, 噪声预测, 部分噪声预测, 任意步采样

## 一句话总结

本文提出InvSR，通过训练一个噪声预测网络来实现扩散反演（Diffusion Inversion），利用预训练扩散模型的图像先验进行超分辨率，支持1-5步任意步数采样，即使单步采样也能达到或超过现有SOTA方法的效果。

## 研究背景与动机

图像超分辨率（SR）是计算机视觉中的基础问题，目标是从低分辨率（LR）图像恢复高分辨率（HR）图像。由于真实场景中退化模型的复杂性和未知性，SR本质上是一个不适定问题。

近年来，大规模文本到图像（T2I）扩散模型展现了强大的图像生成能力，很多研究尝试将其作为先验来缓解SR的不适定性。现有方法主要分两类：一类通过优化中间特征来保证与LR图像的一致性（如DDRM、DDNM），但计算复杂度高且依赖已知退化模型；另一类直接微调T2I模型（如StableSR、DiffBIR、SeeSR），虽效果好但需要修改扩散网络结构。

**核心矛盾**在于：现有方法要么需要修改扩散模型的中间特征/参数，无法完全利用扩散先验；要么采样步数固定，缺乏灵活性。GAN反演在SR中已有应用，但扩散模型的多步随机采样使得反演更加困难——直接优化每步的噪声图计算开销大，且迭代过程会累积预测误差。

**本文的切入角度**是：LR和HR图像仅在高频细节上有差异，加上适当噪声后两者变得不可区分。因此可以用LR图像加上预测的噪声来构造扩散过程的中间状态，从而在不修改扩散网络的前提下实现SR。核心idea是"部分噪声预测"（PnP）——只预测起始步的噪声，后续步使用随机噪声。

## 方法详解

### 整体框架

InvSR的pipeline如下：给定LR图像，用噪声预测网络估计一个噪声图，将其与LR图像按扩散前向过程组合，构造出扩散模型的中间状态作为采样起点。然后用预训练的扩散模型（SD-Turbo）从该中间状态开始反向采样，生成HR图像。整个过程不修改扩散网络参数。

### 关键设计

1. **部分噪声预测策略（PnP）**:

    - 核心观察：LR和HR图像的主要差异在于高频成分，加入适量噪声后两者几乎无法区分
    - 不预测所有扩散步的噪声图，只预测起始步（时间步≤250，对应SNR>1.44）的噪声图
    - 中间步使用随机高斯噪声（因为预训练扩散模型在低噪声水平下已经很稳健）
    - 这样将噪声图集合从T=1000个简化为仅1个，大幅降低预测难度
    - 起始步的噪声预测允许非零均值（因为用LR替代了HR），可视化显示该噪声与LR图像相关

2. **任意步采样机制**:

    - 噪声预测网络通过时间嵌入在多个预选起始步上训练，训练集为 {250, 200, 150, 100}
    - 推理时可自由选择起始步，与加速采样算法（如DDIM）结合实现1-5步采样
    - 起始步越大（如250），采样步数越多，生成更丰富的细节，适合模糊退化
    - 起始步越小（如100），单步即可完成，避免放大噪声，适合重噪声退化
    - 用户可根据退化类型灵活选择步数，实现保真度-真实感的平衡

3. **噪声预测网络**:

    - 基于VQGAN编码器架构，包含两个下采样块，每个配备自注意力层
    - 采用VAE的重参数化技巧，预测高斯分布的均值和方差而非直接预测噪声图
    - 输入为LR图像和时间步，输出为对应的噪声图
    - 参数量仅33.84M，远小于现有扩散SR方法

### 损失函数 / 训练策略

训练损失由三部分组成：
- **L2损失**：基于扩散模型对中间状态的单步去噪结果与GT HR图像之间的MSE
- **LPIPS损失**：感知损失，权重λ_l=2.0，在潜在空间中计算（进行了微调）
- **GAN损失**：对抗损失，权重λ_g=0.1，使用hinge loss，判别器基于扩散UNet加多输入多输出策略

训练细节：
- 基础模型为SD-Turbo，在VQGAN的潜在空间中计算所有损失以节省GPU内存
- 数据集：LSDIR + 20k FFHQ人脸子集，退化用RealESRGAN pipeline合成
- Adam优化器，固定学习率5e-5，batch=64，训练100k+ iterations
- 每次迭代随机从{250, 200, 150, 100}选择起始步训练

## 实验关键数据

### 主实验

| 数据集 | 指标 | InvSR-1 | OSEDiff-1 | SinSR-1 | ResShift-4 | SeeSR-50 | 说明 |
|--------|------|---------|-----------|---------|------------|----------|------|
| ImageNet-Test | LPIPS↓ | 0.2517 | 0.2624 | 0.2209 | 0.1998 | 0.2187 | 单步采样 |
| ImageNet-Test | NIQE↓ | **4.38** | 4.72 | 5.26 | 5.87 | 4.38 | 无参考最优 |
| ImageNet-Test | CLIPIQA↑ | **0.709** | 0.682 | 0.662 | 0.615 | 0.587 | 感知质量最优 |
| ImageNet-Test | MUSIQ↑ | **72.29** | 70.39 | 67.76 | 65.59 | 71.24 | 质量评分最优 |
| RealSR | NIQE↓ | **4.22** | 5.33 | 6.25 | 6.91 | 4.54 | 真实数据最优 |
| RealSR | CLIPIQA↑ | 0.692 | **0.701** | 0.663 | 0.599 | 0.682 | 接近最优 |

InvSR单步即在无参考感知质量指标上全面领先，且参数量（33.84M）远小于SeeSR（751.5M）和DiffBIR（385.4M）。

### 消融实验

| 配置 | PSNR↑ | LPIPS↓ | NIQE↓ | CLIPIQA↑ | 说明 |
|------|-------|--------|-------|----------|------|
| 5步 {250,200,150,100,50} | 22.70 | 0.2844 | 4.88 | 0.673 | 最多步，最多细节 |
| 3步 {250,150,50} | 22.92 | 0.2762 | 4.80 | 0.682 | 平衡配置 |
| 3步 {150,100,50} | 23.84 | 0.2575 | 4.22 | 0.702 | 高SNR起点更佳 |
| 1步 {250} | 23.84 | 0.2575 | 4.53 | 0.713 | 高保真 |
| 1步 {100} | 24.66 | 0.2450 | 4.06 | 0.691 | 高PSNR |

### 关键发现

- 步数越多不一定越好：对于噪声严重的输入，单步反而优于多步（多步可能放大噪声）
- 起始步选择影响保真度-真实感平衡：高起始步(250)更利于恢复细节，低起始步(100)更利于保持保真度
- 中间步不需要噪声预测：在高SNR约束下，预训练扩散模型已足够处理中间步的高斯噪声
- 推理速度优势显著：单步在A100上仅需168ms（128→512），而StableSR-50需1730ms

## 亮点与洞察

- **设计哲学精妙**：核心洞察"LR与HR在加噪后不可区分"简洁而深刻，将复杂的全过程反演简化为单步噪声预测
- **灵活性独特**：首个支持任意步数采样的扩散SR方法，让用户根据退化类型自适应选择
- **效率极高**：完全不修改预训练扩散模型，只增加一个轻量级噪声预测器（33.84M），推理速度快
- **方法论贡献**：PnP策略不仅适用于SR，原则上可推广到其他基于扩散反演的图像恢复任务

## 局限与展望

- 文本提示固定为通用描述，未利用图像内容自适应的语义信息
- SNR阈值1.44（时间步250上限）是手动选定的，未充分讨论阈值选择的敏感性
- 仅在×4 SR任务上评估，未扩展到其他放大倍数或图像恢复任务
- 噪声预测网络架构较简单（基于VQGAN编码器），更强的架构可能带来更好效果
- 退化模型仍然使用合成pipeline（RealESRGAN），在更极端的真实退化上表现有待验证

## 相关工作与启发

- **扩散先验SR**：与StableSR、DiffBIR、SeeSR等微调方法不同，InvSR完全保留扩散模型不变，这种"不修改基础模型"的思路与Adapter/LoRA的设计哲学一致
- **GAN反演 → 扩散反演**：本文成功将GAN反演中"找最优隐空间表示"的思想迁移到扩散模型，是一个有价值的方法论跨越
- **单步蒸馏方法**：与OSEDiff等专用单步方法相比，InvSR同时支持单步和多步，更灵活
- 对图像编辑、图像修复等任务的启发：PnP策略的核心idea（用目标图像的近似来构造扩散中间状态）有广泛适用性

## 评分

- 新颖性: ⭐⭐⭐⭐ 扩散反演用于SR不是全新，但PnP策略和任意步采样的设计很有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多指标、多步数消融、与9种方法对比、运行时间对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、推导严谨、图表规范，整体逻辑流畅
- 价值: ⭐⭐⭐⭐ 方法简洁高效且效果好，有实际应用价值和方法论推广潜力

<!-- RELATED:START -->

## 相关论文

- [Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)

<!-- RELATED:END -->
