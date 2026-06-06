---
title: >-
  [论文解读] Rotation-Equivariant Self-Supervised Method in Image Denoising
description: >-
  [CVPR 2025][图像恢复][自监督去噪] 首次将旋转等变卷积引入自监督图像去噪，严格分析了上/下采样算子对等变性的影响并给出 U-Net 完整网络的等变误差界，进一步提出自适应旋转等变网络 AdaReNet，通过 Mask 融合模块自动决定图像哪些区域更适合使用旋转等变网络…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "自监督去噪"
  - "旋转等变性"
  - "等变卷积"
  - "U-Net"
  - "自适应融合"
---

# Rotation-Equivariant Self-Supervised Method in Image Denoising

**会议**: CVPR 2025  
**arXiv**: [2505.19618](https://arxiv.org/abs/2505.19618)  
**代码**: [https://github.com/liuhanze623/AdaReNet](https://github.com/liuhanze623/AdaReNet)  
**领域**: 图像修复  
**关键词**: 自监督去噪, 旋转等变性, 等变卷积, U-Net, 自适应融合

## 一句话总结

首次将旋转等变卷积引入自监督图像去噪，严格分析了上/下采样算子对等变性的影响并给出 U-Net 完整网络的等变误差界，进一步提出自适应旋转等变网络 AdaReNet，通过 Mask 融合模块自动决定图像哪些区域更适合使用旋转等变网络，在 N2N、N2V、R2R 三种典型自监督方法上均取得一致性能提升。

## 研究背景与动机

**领域现状**：自监督图像去噪方法（Noise2Noise、Noise2Void、R2R 等）因不需要干净-噪声配对数据集而受到广泛关注。这类方法严重依赖深度网络自身内嵌的先验信息来弥补监督信号的不足。目前几乎所有自监督去噪方法都基于 CNN 架构，因为 CNN 天然捕获了图像最重要的先验之一——**平移等变性**。

**现有痛点**：平移等变性的引入为 CNN 带来了巨大成功，但自然图像还存在另一个重要先验——**旋转等变性**。旋转等变卷积已在超分辨率等监督任务中展现出性能提升，但尚未被引入自监督去噪这一更依赖网络先验的领域。将旋转等变性引入自监督去噪面临两个关键挑战：(1) 自监督去噪常用 U-Net 结构，其中的上/下采样模块对等变性的影响缺乏理论分析；(2) 旋转等变设计引入的参数共享和卷积核参数化会降低表示精度，而去噪任务对高频细节的重建要求很高。

**核心矛盾**：旋转等变先验可以为网络提供更好的归纳偏置，但等变设计的刚性约束与自然图像局部不完全满足旋转对称性之间存在矛盾——不加区分地对整张图像使用旋转等变网络反而可能损害重建性能。

**本文目标**：(1) 为 U-Net 架构中旋转等变性提供严格的理论保证；(2) 设计一个自适应框架，让网络自动决定哪些区域使用旋转等变先验。

**切入角度**：从连续域出发，逐模块推导等变误差，最终得到完整 U-Net 的误差界。然后设计双路融合架构解决刚性等变约束的灵活性问题。

**核心 idea**：用 Fconv 旋转等变卷积替换 U-Net 中所有普通卷积即可获得近似旋转等变网络（有理论保证），再通过自适应 Mask 融合等变网络和普通 CNN 的输出，扬长避短。

## 方法详解

### 整体框架

方法分两个层次：(1) 等变 U-Net 构建——将自监督去噪网络中的所有卷积层替换为旋转等变卷积（Fconv），并通过定理证明整个 U-Net 的等变误差有界；(2) 自适应框架 AdaReNet——包含 Vanilla Module（普通CNN）、EQ Module（等变CNN）、Fusion Module（Mask 网络）和 Self-correcting Module（ResNet 块），通过学习到的像素级 Mask 自适应融合两个网络的输出。

### 关键设计

1. **上/下采样等变误差分析（Theorem 1 & 2）**:

    - 功能：为 U-Net 中不可避免的上/下采样操作提供等变误差理论界
    - 核心思路：对于 Maxpooling 和 Stride 下采样，以及 Nearest Neighbor 和 Bilinear 上采样，在特征图的连续函数梯度有界（$\|\nabla e\| \leq G$）的条件下，证明等变误差上界为 $O(h)$（$h$ 为网格大小），即图像分辨率越高，等变误差越小。虽然这比等变卷积层本身的 $O(h^2)$ 收敛率慢，但仍趋于零。
    - 设计动机：之前的工作只分析了卷积层的等变性，U-Net 的上/下采样是否破坏等变性是个悬而未决的关键问题。本文首次给出了严格证明。

2. **完整 U-Net 等变误差界（Theorem 3）**:

    - 功能：证明将 U-Net 中所有卷积替换为 E-Conv 后，整个网络是近似旋转等变的
    - 核心思路：将 U-Net 分解为多个下采样块（E-Conv + 下采样算子）和上采样块（上采样算子 + 2层 E-Conv），逐块推导误差并组合，得到完整网络的等变误差界 $\leq R_1 h + R_2 h^2$，其中 $R_1, R_2$ 是与网络深度、通道数和卷积核性质相关的常数。对任意旋转角度 $\theta$，误差界增加一项 $R_3 t^{-1} h$（$t$ 为等变子群阶数）。
    - 设计动机：这个定理为"简单替换所有卷积层即可获得等变网络"提供了理论支持，大大降低了工程实现难度。

3. **自适应旋转等变网络 AdaReNet**:

    - 功能：自动判断图像不同区域应更多使用等变网络还是普通 CNN
    - 核心思路：AdaReNet 包含四个模块：Vanilla Module $f_c = \text{VM}(I)$ 使用普通 CNN，EQ Module $f_e = \text{EQ}(I)$ 使用旋转等变 CNN，Fusion Module 通过一个 MaskNetwork 学习像素级融合权重 $M_f = \text{Mask}(I)$，Self-correcting Module 通过 ResNet 块精修融合结果。最终输出 $\bar{I} = S_c(M_f \odot f_c + (1-M_f) \odot f_e)$。观察发现 Mask 在纹理边缘处偏大（使用更多 Vanilla），低频区域偏小（使用更多 EQ），这与"卷积更擅长拟合高频信息"的常识一致。
    - 设计动机：自然图像并非处处严格满足旋转等变性，尤其是边缘和纹理区域。刚性等变约束还会因参数共享降低表示精度。自适应融合既保留了等变先验对低频区域的优势，又保护了高频细节的重建质量。

### 损失函数 / 训练策略

- 损失函数：$L = \|\bar{I} - \text{target}\|_2 + \alpha_1 \|f_c - \text{target}\|_2 + \alpha_2 \|f_e - \text{target}\|_2$
- 子网络的损失作为正则项加入主损失，$\alpha_1 = \alpha_2 = 0.1$
- 训练策略因基础方法而异：N2N 用噪声对训练，N2V 用盲点网络单张图训练，R2R 从噪声图生成训练对

## 实验关键数据

### 主实验

| 方法 | 数据集 | σ=25 (PSNR/SSIM) | σ=50 (PSNR/SSIM) |
|------|--------|------------------|------------------|
| N2N | Kodak | 31.47/0.874 | 28.29/0.778 |
| N2N-EQ | Kodak | 31.60/0.878 | 28.58/0.790 |
| N2N-EQ⁺ (AdaReNet) | Kodak | **31.72/0.880** | **28.69/0.791** |
| N2V | BSD500 | 28.17/0.820 | 26.07/0.725 |
| N2V-EQ | BSD500 | 29.05/0.834 | 26.38/0.735 |
| N2V-EQ⁺ (AdaReNet) | BSD500 | **29.12/0.845** | **26.82/0.755** |

### 消融实验

| 配置 | 等变误差 | 说明 |
|------|---------|------|
| N2V (原始) | 0.233 | 普通 CNN 完全没有旋转等变性 |
| N2V-EQ | 0.068 | 等变卷积大幅降低等变误差 |
| N2V-EQ⁺ (AdaReNet) | 0.076 | 自适应融合略增等变误差，但换来更好的重建质量 |

### 关键发现

- 旋转等变先验对自监督去噪的提升是一致性的：在 N2N、N2V、R2R 三种方法上均有效
- AdaReNet 的 Mask 输出揭示了有趣的规律：纹理边缘区域倾向使用普通 CNN，平滑低频区域倾向使用等变网络
- N2V-EQ⁺ 的提升尤为显著（从 28.17 到 29.12 dB，提升近 1 dB），因为 N2V 是最依赖网络先验的方法（只用单张噪声图训练）
- 在 Poisson、椒盐噪声等多种噪声类型上也有效，说明旋转等变先验的通用性

## 亮点与洞察

- **理论贡献扎实**：首次严格分析了上/下采样对等变网络的影响，并给出了完整 U-Net 的等变误差界——这个理论结果本身对等变网络设计社区有重要参考价值
- **AdaReNet 的自适应融合思路**：用 Mask 网络让模型自己决定何时使用等变先验，比粗暴的全局等变设计更符合图像的实际特性。这个思路可以迁移到其他需要引入先验但又担心先验约束太强的场景
- **可插拔设计**：方法可以直接应用于现有的自监督去噪框架，不需要修改训练流程，只需替换网络结构

## 局限与展望

- 旋转等变卷积增加了参数量和计算成本，论文没有详细讨论效率 overhead
- 自适应框架包含两个独立网络（Vanilla + EQ），参数量近乎翻倍，对于资源受限场景不太友好
- 理论分析假设特征函数光滑且梯度有界，这在实际深度网络中不一定严格成立
- 仅验证了 O(2) 群的旋转等变性，如否扩展到 3D 旋转（SO(3)）用于体数据去噪值得探索

## 相关工作与启发

- **vs Fconv (Xie et al.)**: Fconv 提出了高精度的旋转等变卷积用于图像处理，但仅用于 ResNet 结构且限于监督学习。本文首次将其扩展到 U-Net 结构和自监督学习
- **vs 数据增广方法**: 旋转数据增广是引入旋转不变性的简单方法，但不保证网络本身的等变性。等变卷积从架构层面保证等变性，提供更好的解释性和泛化性
- **vs DnCNN/FFDNet**: 这些监督方法依赖大规模配对数据，AdaReNet 的自监督+等变设计在数据稀缺场景更有优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将旋转等变引入自监督去噪，理论分析是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 三种基础方法、多种噪声类型、多个数据集
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，动机阐述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 为自监督去噪提供了新视角，理论结果对等变网络设计有普适意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2025\] Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise](generalized_recorrupted-to-recorrupted_self-supervised_learning_beyond_gaussian_.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)

</div>

<!-- RELATED:END -->
