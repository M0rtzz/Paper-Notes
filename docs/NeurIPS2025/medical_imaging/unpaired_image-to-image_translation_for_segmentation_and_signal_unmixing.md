---
title: >-
  [论文解读] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing
description: >-
  [NeurIPS 2025][医学图像][无配对图像翻译] 提出 Ui2i 模型，在 CycleGAN 基础上通过 UNet 生成器、近似双向谱归一化替代特征归一化、通道-空间注意力和尺度增强，实现高内容保真度的无配对图像翻译，成功用于 IHC→H&E 域适应核分割及单通道免疫荧光信号解混两大生物医学任务。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "无配对图像翻译"
  - "核分割"
  - "免疫荧光解混"
  - "GAN"
  - "谱归一化"
---

# Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing

**会议**: NeurIPS 2025  
**arXiv**: [2505.20746](https://arxiv.org/abs/2505.20746)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: 无配对图像翻译, 核分割, 免疫荧光解混, CycleGAN, 谱归一化

## 一句话总结

提出 Ui2i 模型，在 CycleGAN 基础上通过 UNet 生成器、近似双向谱归一化替代特征归一化、通道-空间注意力和尺度增强，实现高内容保真度的无配对图像翻译，成功用于 IHC→H&E 域适应核分割及单通道免疫荧光信号解混两大生物医学任务。

## 研究背景与动机

在数字病理和空间蛋白质组学中，不同染色协议（H&E、IHC、IF）之间存在巨大的域差异，导致在某一域上训练的分割模型（如 StarDist）无法直接泛化到其他域。获取新域的标注数据成本极高，因此需要通过**图像到图像翻译**实现跨域适应。

然而，生物医学图像对内容保真度的要求极为严格——核的形状、边界、细胞形态和空间关系必须被精确保留，这与自然场景翻译中允许一定失真的要求不同。现有方法（如 CycleGAN）虽然通过循环一致性损失实现了无配对翻译，但存在严重问题：

**循环一致性的非唯一性**：作者从理论上证明，对于完美解耦内容-风格的生成器对 $(\hat{G}_{AB}, \hat{G}_{BA})$，施加任意可逆变换 $T_A, T_B$ 后仍满足循环一致性，即存在无穷多满足 $\mathcal{L}_{\text{cyc}}$ 的生成器对，无法唯一保证内容保留。

**特征归一化导致的伪影**：Instance Normalization 让局部物体在不同全局上下文中产生显著不同的信号响应，导致翻译图像中出现"液滴状"伪影，在核分割中被误识别为假阳性。

本文的核心思路是：彻底移除特征归一化层，替换为**参数级别的近似双向谱归一化**，并结合 UNet 架构的跳跃连接和注意力机制来更好地保留空间局部内容特征。

## 方法详解

### 整体框架

Ui2i 基于 CycleGAN 架构，包含两个生成器 $G_{AB}$ 和 $G_{BA}$、一个域判别器和一个内容判别器。关键改进包括：UNet 结构的生成器、双向谱归一化、通道-空间注意力和尺度增强。

### 关键设计

1. **近似双向谱归一化 (ABSN)**：移除所有特征归一化（BN、IN 等），对每层权重张量 $\mathbf{w}$ 进行双向谱归一化。权重矩阵同时考虑前向 reshape $W^{FW}$（前向信息流）和后向 reshape $W^{BW}$（梯度传播），使用可微下界估计谱范数：

$$\|W\| \geq \sigma(W) = \frac{\left\|\left(\sum_{j=1}^n w_j w_j^\top\right)r\right\|}{\|(w_1^\top r, \ldots, w_n^\top r)^\top\|}$$

然后用 RMS 聚合前向和后向谱范数：$\sigma_{\text{rms}} = \sqrt{(\sigma^2(W^{FW}) + \sigma^2(W^{BW}))/2}$。这避免了特征归一化导致的上下文依赖响应问题，同时保持训练稳定性。

2. **UNet 生成器 + 跳跃连接**：替代 CycleGAN 的 ResNet 生成器，利用跳跃连接将编码器浅层的局部空间特征传播到解码器深层，更好地保留细粒度结构信息。两个生成器在瓶颈层共享权重。解码器使用 4×4 Lanczos2 核上采样，避免棋盘格伪影。

3. **通道-空间注意力模块**：在编码器模块中集成 ESCA（高效对称空间和通道注意力）+ 空间注意力的串联模块，对特征图进行注意力精炼。配合残差连接，增强内容保留能力。解码器省略注意力模块以提高效率。

4. **堆叠域判别器 + 内容判别器**：域判别器采用单个 PatchGAN 结构，将两个域的图像在通道维拼接后分类为 real/fake/identity 三类，减少过拟合。内容判别器对瓶颈特征分类域来源（A 或 B），鼓励域不变的内容表示。

### 损失函数 / 训练策略

总损失为：

$$\mathcal{L} = \mathcal{L}_{\text{adv}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}} + \lambda_{\text{id}}\mathcal{L}_{\text{id}} + \lambda_{\text{cl}}\mathcal{L}_{\text{cl}}$$

- $\mathcal{L}_{\text{adv}}$：对抗损失，监督域判别器和内容判别器
- $\mathcal{L}_{\text{cyc}}$：循环一致性损失，$\lambda_{\text{cyc}}=10$
- $\mathcal{L}_{\text{id}}$：恒等映射损失，$\lambda_{\text{id}}=1$
- $\mathcal{L}_{\text{cl}}$：N-pair 对比损失，拉近原始-翻译图像对的瓶颈特征，$\lambda_{\text{cl}}=0.1$

训练使用可微数据增强：随机缩放因子 $[0.75, 1.5]$，促进尺度不变特征学习。Adam 优化器，lr=0.0002，50K 迭代，保存最近 50 张生成图像的 buffer 用于稳定判别器训练。

## 实验关键数据

### 主实验：IHC→H&E 核分割

| 方法 | Instance Precision | Instance Recall | Segm. Quality | Panoptic Quality |
|------|-------------------|----------------|---------------|-----------------|
| No translation (StarDist) | 0.92±0.13 | 0.51±0.21 | 0.78±0.08 | 0.50±0.17 |
| InstanSeg (IHC预训练) | 0.76±0.16 | 0.70±0.17 | 0.75±0.08 | 0.55±0.12 |
| CycleGAN | 0.72±0.18 | 0.76±0.16 | 0.80±0.05 | 0.59±0.14 |
| **full Ui2i** | **0.87±0.11** | **0.77±0.14** | 0.80±0.05 | **0.65±0.10** |

### 消融实验

| 配置 | Instance Precision | Instance Recall | Panoptic Quality | 说明 |
|------|-------------------|----------------|-----------------|------|
| Ui2i w/o augment. | 0.83±0.14 | 0.72±0.14 | 0.60±0.11 | 移除尺度增强 |
| Ui2i w/ feature norm. | 0.75±0.18 | 0.74±0.16 | 0.60±0.14 | 恢复特征归一化 |
| Ui2i w/o attention | 0.83±0.14 | 0.73±0.15 | 0.63±0.12 | 移除注意力 |
| full Ui2i | **0.87±0.11** | **0.77±0.14** | **0.65±0.10** | 完整模型 |

### IF 信号解混定量结果

| 指标 | SOX2 通道 | Grasp65 通道 |
|------|----------|-------------|
| MicroMS-SSIM | 0.96±0.03 | 0.96±0.02 |
| PSNR | 38±3 | 32±2 |

与 MicroSplit（需配对数据）对比：MicroMS-SSIM 0.978/0.951，PSNR 40.3/32.8，Ui2i 使用无配对数据接近配对方法性能。

### 关键发现

- Ui2i 在所有 StarDist 分割指标上显著超越 CycleGAN，甚至超过在 IHC 域预训练的 InstanSeg
- 特征归一化是伪影的主要来源，替换为谱归一化后 precision 从 0.75 提升至 0.87
- 首次实现从真实无配对数据学习 IF 信号解混，解决了单荧光团标记两种生物标志物的实际需求

## 亮点与洞察

- 理论分析了循环一致性损失的非唯一性问题，解释了为何仅靠 $\mathcal{L}_{\text{cyc}}$ 不足以保证内容保留
- 从参数归一化角度切入解决特征归一化导致的上下文依赖伪影，思路新颖且简洁
- IF 信号解混应用极具实用价值——能将 mIF 实验的标记容量翻倍

## 局限与展望

- 仅在核分割和 IF 解混两个具体任务上评估，泛化性需进一步验证
- IF 解混的定量评估使用的 HT-T24 数据集并非完全模拟真实单荧光团复用场景
- 未与基于扩散模型的最新 i2i 翻译方法比较

## 相关工作与启发

- 与 MicroSplit 相比，Ui2i 不需要配对数据，更适合实际应用场景
- ABSN 的思路可以推广到其他需要高保真度的医学图像生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论分析+ABSN设计有新意，IF解混是新应用
- 实验充分度: ⭐⭐⭐⭐ 两个应用场景+完整消融，但缺少更多baseline
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，理论-方法-实验衔接紧密
- 价值: ⭐⭐⭐⭐ IF解混的实用价值高，域适应方法实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)

</div>

<!-- RELATED:END -->
