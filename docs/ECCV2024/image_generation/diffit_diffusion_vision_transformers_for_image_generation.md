---
title: >-
  [论文解读] DiffiT: Diffusion Vision Transformers for Image Generation
description: >-
  [ECCV 2024][图像生成][Transformer] 提出 DiffiT（Diffusion Vision Transformer），通过引入时间依赖多头自注意力（TMSA）机制，让自注意力在去噪过程的不同阶段动态调整行为，在ImageNet-256上以比DiT/MDT少16-20%的参数量达到了1.73的SOTA FID分数。
tags:
  - ECCV 2024
  - 图像生成
  - Transformer
  - 时间依赖自注意力
  - 扩散模型
  - 参数高效
---

# DiffiT: Diffusion Vision Transformers for Image Generation

**会议**: ECCV 2024  
**arXiv**: [2312.02139](https://arxiv.org/abs/2312.02139)  
**代码**: https://github.com/NVlabs/DiffiT (有)  
**领域**: 扩散模型 / 图像生成  
**关键词**: Vision Transformer, 时间依赖自注意力, 扩散模型, 图像生成, 参数高效

## 一句话总结

提出 DiffiT（Diffusion Vision Transformer），通过引入时间依赖多头自注意力（TMSA）机制，让自注意力在去噪过程的不同阶段动态调整行为，在ImageNet-256上以比DiT/MDT少16-20%的参数量达到了1.73的SOTA FID分数。

## 研究背景与动机

**领域现状**：扩散模型已成为图像生成的主流框架，其核心组件是去噪网络。传统上使用CNN-based U-Net作为去噪骨干，近期DiT和MDT等工作开始探索用Vision Transformer替代U-Net。

**现有痛点**：

**时间条件注入方式粗糙**：DiT和MDT使用AdaLN（Adaptive LayerNorm）进行噪声时间步条件化，通过逐通道的scale和shift参数调制输入。这种机制无法有效建模去噪过程中空间与时间依赖性的联合关系

**去噪过程的时间动态未被充分捕捉**：去噪初期模型主要预测低频内容（整体结构），后期则聚焦于高频细节。AdaLN机制无法让注意力机制根据时间步动态调整关注模式

**参数效率低**：AdaLN每个Transformer块需要学习6个调制分量（self-attention和MLP各需shift、scale、gate），参数开销大

**核心矛盾**：如何设计一种既能精细控制时间-空间交互、又能保持参数高效的去噪网络架构？

**切入角度**：将时间信息直接整合到自注意力的Q/K/V计算中，而非外部调制，使注意力本身具备时间自适应能力。

**核心idea**：Q、K、V均为空间token与时间token的线性组合，自注意力因此能在不同去噪阶段自动改变关注模式——从全局结构到局部细节渐进聚焦。

## 方法详解

### 整体框架

DiffiT提供两种变体：
- **图像空间模型**：采用对称U-Net编解码架构，每个分辨率级别由L个DiffiT块组成，使用卷积层进行上下采样，通过skip connection连接编解码器
- **潜空间模型**：使用预训练VAE将图像编码为潜表示，去噪网络为纯Transformer（无上下采样），类似DiT架构但使用TMSA替代AdaLN

### 关键设计

1. **时间依赖多头自注意力（TMSA）**：

    - **功能**：将时间嵌入直接融入自注意力的Q/K/V计算中
    - **核心思路**：对于空间token $\mathbf{x_s}$ 和时间token $\mathbf{x_t}$，计算时间依赖的Q/K/V：
    $\mathbf{q_s} = \mathbf{x_s}\mathbf{W}_{qs} + \mathbf{x_t}\mathbf{W}_{qt}$
    $\mathbf{k_s} = \mathbf{x_s}\mathbf{W}_{ks} + \mathbf{x_t}\mathbf{W}_{kt}$
    $\mathbf{v_s} = \mathbf{x_s}\mathbf{W}_{vs} + \mathbf{x_t}\mathbf{W}_{vt}$
      自注意力计算为：$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\frac{\mathbf{QK}^\top}{\sqrt{d}} + \mathbf{B})\mathbf{V}$
    - **设计动机**：Q/K/V都是空间和时间token的线性函数，使注意力能在不同时间步自适应调整。相比AdaLN的6个调制分量，TMSA每块仅需3个时间线性投影（$\mathbf{W}_{qt}, \mathbf{W}_{kt}, \mathbf{W}_{vt}$），参数更少。可视化表明TMSA模型的注意力图展现出从全局到局部的渐进聚焦效果。

2. **窗口化TMSA（图像空间模型）**：

    - **功能**：将自注意力限制在不重叠的局部窗口内计算
    - **核心思路**：在不减少局部区域间通信的前提下，通过U-Net瓶颈层实现跨区域信息共享
    - **设计动机**：自注意力的二次复杂度在大特征图时代价昂贵，窗口化大幅降低token序列长度。实验表明窗口大小为4时即可获得大部分性能提升

3. **DiffiT ResBlock（图像空间模型）**：

    - **功能**：组合卷积层与DiffiT Transformer块的混合残差单元
    - **核心思路**：
    $\mathbf{\hat{x}_s} = \text{Conv}_{3\times 3}(\text{Swish}(\text{GN}(\mathbf{x_s})))$
    $\mathbf{x_s} = \text{DiffiT-Block}(\mathbf{\hat{x}_s}, \mathbf{x_t}) + \mathbf{x_s}$
    - **设计动机**：卷积层嵌入图像归纳偏置，与Transformer块互补

4. **三通道Classifier-Free Guidance（潜空间模型）**：

    - **功能**：在潜空间模型中使用三通道的CFG以提升生成质量
    - **设计动机**：直接提升条件生成的保真度，在ImageNet-256上使用4.6的guidance scale达到最优1.73 FID

### 损失函数 / 训练策略

- 标准去噪分数匹配损失：$\mathbb{E}[\lambda(t)\|\epsilon - \epsilon_\theta(\mathbf{z}_0 + \sigma_t\epsilon, t)\|_2^2]$
- 采样使用随机微分方程求解器（SDE/ODE选择），ODE求解器步数更少，SDE求解器对不精确score更鲁棒
- 时间嵌入使用位置编码（优于傅里叶编码）

## 实验关键数据

### 主实验（潜空间模型，ImageNet-256）

| 模型 | 类型 | 参数量(M) | FLOPs(G) | FID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|----------|---------|------|-----|-----------|---------|
| DiT-XL/2-G | Diffusion | 675 | 119 | 2.27 | 278.24 | 0.83 | 0.57 |
| MDT-G | Diffusion | 700 | 121 | 1.79 | 283.01 | 0.81 | 0.61 |
| SiT-XL | Diffusion | 675 | 119 | 2.06 | 270.27 | 0.82 | 0.59 |
| StyleGAN-XL | GAN | - | - | 2.30 | 265.12 | 0.78 | 0.53 |
| **DiffiT** | **Diffusion** | **561** | **114** | **1.73** | **276.49** | **0.80** | **0.62** |

**图像空间模型**：

| 模型 | CIFAR-10 FID↓ | FFHQ-64 FID↓ |
|------|-------------|-------------|
| EDM (VP) | 1.99 | 2.39 |
| LSGM | 2.01 | - |
| **DiffiT** | **1.95** | **2.22** |

### 消融实验

**架构设计消融（CIFAR-10）**：

| 配置 | 编码器 | 解码器 | FID↓ | 说明 |
|-----|--------|--------|------|------|
| A | ViT | SETR-MLA | 5.34 | 等距架构不理想 |
| B | +多分辨率 | SETR-MLA | 4.64 | 多尺度特征有帮助 |
| C | 多分辨率 | +多分辨率 | 3.71 | 对称U-Net进一步提升 |
| D | +DiffiT编码器 | 多分辨率 | 2.27 | TMSA的效果显著 |
| E | +DiffiT编码器 | +DiffiT解码器 | 1.95 | 完整DiffiT |

**TMSA有效性（替换DDPM++自注意力）**：

| 模型 | 无TMSA FID↓ | 有TMSA FID↓ | 提升 |
|------|-----------|-----------|------|
| DDPM++ (VE) | 3.77 | 3.49 | -0.28 |
| DDPM++ (VP) | 3.01 | 2.76 | -0.25 |

**时间依赖注入位置消融**：

| 配置 | 注入方式 | FID↓ | 说明 |
|-----|---------|------|------|
| F | 仅相对位置偏置 | 3.97 | 无法同时编码空间和时间 |
| G | 仅MLP层 | 3.81 | 次优 |
| H | TMSA（Q/K/V） | 1.95 | 最优位置 |

### 关键发现

- TMSA在DiffiT中带来了显著且一致的FID提升，跨数据集和模型配置
- 时间token必须与空间token混合而非分离处理（分离后FID从1.95降为2.28）
- 位置时间嵌入优于傅里叶时间嵌入（1.95 vs 2.02）
- 窗口大小从2到4时性能提升23%，但从4到8仅提升1.5%，存在空间冗余
- DiffiT比DiT-XL参数少16.88%、FLOPs少4.38%，但FID更优

## 亮点与洞察

- TMSA的设计思路极其简洁——仅仅是在Q/K/V投影时加入时间token的线性投影，但效果显著且通用（可直接替换任何Transformer中的self-attention）
- "去噪过程中注意力应随时间步变化"这一观察虽直觉上自然，但之前的方法（如AdaLN）并未真正在注意力层面实现这一点
- 图像空间模型中卷积+Transformer的混合设计利用了两者的互补优势

## 局限与展望

- Latent DiffiT在ImageNet-512上表现不如StyleGAN-XL（FID 2.67 vs 2.41），虽然GAN多样性可能不足
- TMSA的window-based变体缺少跨窗口通信机制（依赖U-Net瓶颈），在纯Transformer架构中可能不够
- 未在文本到图像生成任务上验证，仅做了类条件和无条件生成
- 与后续工作（如DiT-3、Flux等）的比较缺失

## 相关工作与启发

- **DiT**：首个用Transformer替代U-Net做潜空间扩散的工作，使用AdaLN条件化，本文证明TMSA是更优的替代方案
- **MDT**：在DiT基础上引入mask建模以捕获上下文信息，但训练流程更复杂
- **SiT**：将flow matching与DiT结合，与DiffiT的设计正交
- **EDM**：图像空间扩散模型的强基线，DiffiT在CIFAR-10和FFHQ-64上均超越
- 启发：时间条件注入的位置和方式对扩散模型至关重要，直接解耦到注意力的Q/K/V中比外部调制更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ TMSA思路简洁新颖，将时间信息优雅地融入自注意力机制
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其详尽，覆盖架构/TMSA设计/窗口大小/CFG/时间嵌入/效率等多维度
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验组织有条理，可视化注意力图很有说服力
- 价值: ⭐⭐⭐⭐⭐ TMSA可直接替换现有方法中的自注意力层，通用性强，NVIDIA出品代码已开源

<!-- RELATED:START -->

## 相关论文

- [RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](../../CVPR2026/image_generation/razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)
- [EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](../../CVPR2026/image_generation/edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [Latent Guard: a Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](../../ICLR2026/image_generation/contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)

<!-- RELATED:END -->
