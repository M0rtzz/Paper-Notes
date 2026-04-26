---
title: >-
  [论文解读] HVI: A New Color Space for Low-light Image Enhancement
description: >-
  [CVPR 2025][图像恢复][低光图像增强] 本文提出了一种新的色彩空间 HVI（Horizontal/Vertical-Intensity），通过极化的 HS 映射消除红色伪影、可学习的强度分量压缩暗区黑色伪影，并配合 CIDNet 解耦网络在 10 个数据集上超越了现有低光增强 SOTA。
tags:
  - CVPR 2025
  - 图像恢复
  - 低光图像增强
  - 色彩空间
  - HVI色彩空间
  - 颜色解耦
  - CIDNet
---

# HVI: A New Color Space for Low-light Image Enhancement

**会议**: CVPR 2025  
**arXiv**: [2502.20272](https://arxiv.org/abs/2502.20272)  
**代码**: https://github.com/Fediory/HVI-CIDNet (有)  
**领域**: 图像恢复 / 低光增强  
**关键词**: 低光图像增强, 色彩空间, HVI色彩空间, 颜色解耦, CIDNet

## 一句话总结
本文提出了一种新的色彩空间 HVI（Horizontal/Vertical-Intensity），通过极化的 HS 映射消除红色伪影、可学习的强度分量压缩暗区黑色伪影，并配合 CIDNet 解耦网络在 10 个数据集上超越了现有低光增强 SOTA。

## 研究背景与动机

**领域现状**：低光图像增强（LLIE）是计算机视觉中的基础任务，目标是从低光条件下的退化图像中恢复清晰的视觉信息。主流方法大多基于 sRGB 色彩空间直接处理图像，通过深度网络学习从低光到正常光照的映射函数。

**现有痛点**：sRGB 空间存在固有的高色彩灵敏度问题——RGB 三通道高度耦合，低光条件下的增强操作容易引入色偏和亮度伪影。一种自然的改进思路是转换到 HSV 空间处理，因为 HSV 将亮度（V）与色彩（H、S）分离。然而 HSV 空间的 Hue 通道采用角度表示，在红色附近（0° 和 360° 的跳变区域）存在数值不连续性，导致增强结果出现严重的红色噪声伪影；同时 V 通道在极低光区域高度压缩，放大后产生黑色噪声伪影。

**核心矛盾**：sRGB 的通道耦合问题和 HSV 的数值不连续性问题构成了一个两难困境——直接处理 RGB 有色偏问题，转到 HSV 又有红色/黑色伪影问题。现有色彩空间都无法同时满足"通道解耦"和"数值连续稳定"这两个需求。

**本文目标**：设计一个专门为低光增强优化的新色彩空间，同时克服 sRGB 的耦合问题和 HSV 的伪影问题。

**切入角度**：作者观察到 HSV 的两个核心问题——红色伪影来自 Hue 角度的不连续性，黑色伪影来自 Value 在暗区的极端压缩。如果能通过坐标变换消除角度不连续性，并通过可学习映射改善暗区分布，就能同时解决这两个问题。

**核心 idea**：将 HSV 的极坐标 Hue-Saturation 重映射为笛卡尔坐标 H/V（水平/垂直），消除红色跳变区域的距离变大问题；同时用可学习的强度分量 I 替代固定的 Value 通道，自适应地压缩暗区分布以消除黑色伪影。

## 方法详解

### 整体框架
系统分为两个核心组件：HVI 色彩空间转换和 CIDNet（Color and Intensity Decoupling Network）。输入低光 sRGB 图像首先被转换到 HVI 色彩空间，其中 H（水平）和 V（垂直）分量编码色彩信息，I（强度）分量编码亮度信息。CIDNet 在 HVI 空间中分别处理色彩分支和强度分支，学习光照条件下的光度映射函数，最后将结果转换回 sRGB 空间输出增强图像。

### 关键设计

1. **HVI 色彩空间（Horizontal/Vertical-Intensity Color Space）**:

    - 功能：提供一个数值连续、通道解耦的色彩表示，消除 sRGB 的耦合问题和 HSV 的伪影问题
    - 核心思路：从 HSV 出发，将极坐标形式的 Hue-Saturation 映射转化为笛卡尔坐标。具体地，定义 $H_{cart} = S \cdot \cos(2\pi \cdot Hue)$，$V_{cart} = S \cdot \sin(2\pi \cdot Hue)$。这就是所谓的"极化 HS 映射"（polarized HS maps）。在极坐标中，红色（Hue ≈ 0° 和 360°）附近两点的角距离很大，但在笛卡尔坐标中对应点的欧氏距离很小。这从根本上消除了红色区域的数值跳变。而 Intensity 分量采用可学习参数 $\alpha_s$、$\alpha_i$、$\gamma$ 对原始 Value 进行非线性变换，自适应压缩暗区的极端值分布
    - 设计动机：传统 HSV 的 Hue 通道在 0/360° 处不连续导致红色伪影是该空间的固有缺陷。笛卡尔坐标变换是解决角度不连续问题的经典数学方法，但直接用于低光增强此前未被探索

2. **CIDNet（Color and Intensity Decoupling Network）**:

    - 功能：在 HVI 空间中对色彩和强度信息进行解耦处理，学习准确的光度映射
    - 核心思路：网络包含两个并行分支——色彩分支处理 H/V 通道，强度分支处理 I 通道。每个分支采用 U-Net 式的编解码结构，包含多个 Lighten Cross-Attention（LCA）模块。LCA 让色彩分支和强度分支之间进行信息交互：色彩分支的特征作为 Query，强度分支的特征作为 Key/Value（反之亦然），通过交叉注意力实现色彩和亮度信息的互补融合
    - 设计动机：色彩和亮度的退化模式不同——低光条件下色彩偏移和亮度不足是两种不同的退化，需要分别建模处理。但二者又不完全独立（如亮度变化会影响色彩感知），所以需要通过交叉注意力保持适度的信息交互

3. **可学习强度变换（Learnable Intensity Transformation）**:

    - 功能：自适应调整不同光照条件下的亮度映射
    - 核心思路：引入三个可调参数 $\alpha_s$（饱和度缩放）、$\alpha_i$（强度缩放）和 $\gamma$（gamma 校正），对 HVI 空间的 Intensity 分量进行参数化变换。这些参数可以在推理时手动调节以适配不同光照场景，也可以在训练中自动学习最优值
    - 设计动机：不同数据集/场景的低光退化程度差异很大。固定的空间变换无法适应所有场景，可学习参数提供了灵活性。特别是通过 random gamma 增强可以提升跨数据集泛化能力

### 损失函数 / 训练策略
综合使用 L1 重建损失、感知损失（perceptual loss）和 SSIM 损失。L1 保证像素级准确性，感知损失保证高层语义一致性，SSIM 保证结构相似性。训练时可选 warmup 策略。支持 LOLv1、LOLv2-real、LOLv2-syn、LOL-Blur、SID、SICE、FiveK 等多个数据集进行训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CIDNet | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| LOLv1 | PSNR | 27.08 | 25.74 (Retinexformer) | +1.34 |
| LOLv1 | SSIM | 0.870 | 0.845 | +0.025 |
| LOLv2-real | PSNR | 26.03 | 24.81 | +1.22 |
| LOLv2-syn | PSNR | 27.51 | 26.32 | +1.19 |
| LOL-Blur | PSNR | 27.84 | 26.47 | +1.37 |
| DICM/LIME/MEF/NPE/VV (无参考) | NIQE↓ | 3.28 | 3.65 | -0.37 |

### 消融实验

| 配置 | PSNR (LOLv1) | SSIM | 说明 |
|------|-------------|------|------|
| Full CIDNet (HVI) | 27.08 | 0.870 | 完整模型 |
| sRGB 空间 | 24.92 | 0.831 | 直接在 RGB 空间处理 |
| HSV 空间 | 25.41 | 0.842 | 在传统 HSV 空间处理 |
| HVI w/o 可学习 I | 26.23 | 0.856 | 使用固定 Value 替代可学习 Intensity |
| w/o LCA 模块 | 26.15 | 0.851 | 去掉交叉注意力 |
| w/o 感知损失 | 26.67 | 0.862 | 去掉 perceptual loss |

### 关键发现
- HVI 色彩空间相比 sRGB 和 HSV 分别带来了 2.16 和 1.67 dB PSNR 提升，充分验证了新色彩空间的有效性
- 可学习 Intensity 变换贡献了 0.85 dB，LCA 交叉注意力贡献了 0.93 dB，两个核心设计均有显著贡献
- 模型在 5 个无参考数据集上同样表现最优，证明了良好的泛化能力
- NTIRE 2025 低光增强竞赛中基于 HVI-CIDNet 的融合方案获得冠军

## 亮点与洞察
- **从色彩空间层面解决问题**的思路非常根本性——大多数低光增强工作聚焦于网络架构设计，本文回溯到更底层的数据表示问题。这个视角启发我们：有时候换一个表示空间比堆叠网络层更有效
- **笛卡尔坐标消除角度不连续**是一个简洁优雅的数学操作。这个 trick 可以迁移到任何涉及角度/周期性数据的任务（如光流方向、旋转估计等）
- 可学习参数 $\alpha_s$, $\alpha_i$, $\gamma$ 的设计让色彩空间变换本身也变得可微和可适配，这是一个有价值的设计范式

## 局限与展望
- HVI 空间是 HSV 的变体，其设计强依赖于 HS 极坐标→笛卡尔坐标这个特定变换，是否存在更优的色彩空间设计有待探索
- 在极端低光（如 SID 数据集的极暗场景）中，性能提升幅度相对较小，暗示底层色彩空间变换在信噪比极低时作用有限
- 可学习参数在推理时需手动调节以适配不同场景，自动化程度可以进一步提升
- 值得探索将 HVI 色彩空间推广到其他图像恢复任务（去雾、去雨等）

## 相关工作与启发
- **vs Retinexformer**: Retinexformer 基于 Retinex 理论在 sRGB 空间进行分解，CIDNet 则在专门设计的 HVI 空间中解耦处理。HVI 的色彩空间优势使其在相同网络复杂度下取得更好结果
- **vs SNR-Aware**: SNR-Aware 通过估计噪声水平自适应处理，CIDNet 则通过色彩空间变换从源头减少退化影响。两者思路互补，理论上可以结合
- 该工作表明"数据表示优于网络设计"在某些场景下成立，值得在其他低层视觉任务中验证这一观点

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从色彩空间层面提出全新解决方案，极坐标到笛卡尔坐标的映射思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个数据集全面验证，消融实验详尽，NTIRE 竞赛冠军背书
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，色彩空间设计的数学描述完整
- 价值: ⭐⭐⭐⭐⭐ 提出了通用的新色彩空间，797 stars 和竞赛冠军说明了实际影响力

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](../../ICCV2025/image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)

<!-- RELATED:END -->
