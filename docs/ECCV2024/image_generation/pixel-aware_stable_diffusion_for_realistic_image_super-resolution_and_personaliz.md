---
title: >-
  [论文解读] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization
description: >-
  [ECCV 2024][图像生成] 提出 PASD（Pixel-Aware Stable Diffusion），通过像素感知交叉注意力（PACA）模块使扩散模型在像素级感知图像局部结构，结合退化去除模块和可调噪声调度，实现了真实图像超分辨率和个性化风格化的统一框架，只需替换底座模型即可切换风格。
tags:
  - ECCV 2024
  - 图像生成
---

# Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization

**会议**: ECCV 2024  
**arXiv**: [2308.14469](https://arxiv.org/abs/2308.14469)  
**领域**: 图像生成

## 一句话总结

提出 PASD（Pixel-Aware Stable Diffusion），通过像素感知交叉注意力（PACA）模块使扩散模型在像素级感知图像局部结构，结合退化去除模块和可调噪声调度，实现了真实图像超分辨率和个性化风格化的统一框架，只需替换底座模型即可切换风格。

## 研究背景与动机

利用预训练文本到图像扩散模型（如 Stable Diffusion）做真实图像超分（Real-ISR）是一个有前景的方向，但面临核心挑战：

**像素级结构保持困难**：ControlNet 等方法通过"零卷积"简单相加 U-Net 和条件特征，无法传递像素级精确信息，导致输出与输入的结构不一致

**跳跃连接的局限**：StableSR 等方法通过 VAE 编码器-解码器之间的跳跃连接传递像素细节，但需要额外在图像空间训练，限制了在潜空间任务（如风格化）中的应用

**训练-测试不一致**：SD 的噪声调度在训练终端时间步留有残余信号，但测试时从纯高斯噪声采样，产生不一致

PASD 的目标是设计一个灵活模型，同时解决 Real-ISR 和个性化风格化，且不需要跳跃连接。

## 方法详解

### 整体框架

PASD 在预训练 SD 基础上增加四个模块：

1. **Degradation Removal Module**：金字塔网络提取多尺度退化不敏感特征
2. **ControlNet**：提取低层控制特征
3. **PACA（Pixel-Aware Cross Attention）**：在潜空间实现像素级引导
4. **ANS（Adjustable Noise Schedule）**：可调噪声调度，灵活平衡感知-保真度
5. **High-level Nets**：利用 ResNet/YOLO/BLIP 提取分类/检测/描述语义信息

### 关键设计

**像素感知交叉注意力（PACA）**：

替代 ControlNet 的零卷积连接，将 U-Net 特征 $\mathbf{x} \in \mathbb{R}^{h \times w \times c}$ 和 ControlNet 特征 $\mathbf{y}$ 分别展平后，以 $\mathbf{y}$ 为上下文输入做交叉注意力：

$$PACA(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = Softmax(\frac{\mathbf{QK}^T}{\sqrt{d}}) \cdot \mathbf{V}$$

其中 $\mathbf{Q} = to\_q(\mathbf{x}')$, $\mathbf{K} = to\_k(\mathbf{y}')$, $\mathbf{V} = to\_v(\mathbf{y}')$。由于 $\mathbf{y}'$ 长度为 $h*w$（等于潜特征的像素总数），且未经 VAE 编码器转换，能保持原始图像结构，从而实现像素级感知。

**退化去除模块**：

金字塔网络提取 1/2、1/4、1/8 三个尺度的特征图，每个尺度通过"toRGB"卷积层重建 HQ 图像并施加 L1 损失监督：$\mathcal{L}_{DR} = \sum_s \|\mathbf{I}_{hq}^s - \mathbf{I}_{sr}^s\|_1$。这使后续扩散模块专注于恢复真实细节，无需处理退化。

**可调噪声调度（ANS）**：

引入额外高斯噪声 $\mathbf{z}'$ 控制残余信号强度：

$$\mathbf{z}_N = \sqrt{\bar{\alpha}_a \bar{\alpha}_N} \mathbf{z}_{LR} + \sqrt{1 - \bar{\alpha}_a \bar{\alpha}_N} \mathbf{z}''$$

$\bar{\alpha}_a \in [0, 1]$ 控制感知-保真度平衡：$\bar{\alpha}_a$ 越大保真度越高但细节越少。实验选择 $\bar{\alpha}_{900} = 0.1189$ 作为最佳平衡点。

**个性化风格化**：

由于 PASD 冻结了 SD 基座模型参数，仅训练新增模块，因此测试时可直接替换基座模型为个性化模型（如 ToonYou 做卡通化，majicMIX realistic 做老照片修复），无需额外训练。

### 损失函数

总训练损失：$\mathcal{L} = \mathcal{L}_{DF-\epsilon} + \gamma \mathcal{L}_{DR}$

扩散损失：$\mathcal{L}_{DF-\epsilon} = \mathbb{E}_{z_0, t, c, I_{lq}, \epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon_\theta(z_t, t, c, I_{lq})\|_2^2]$

训练时随机替换 50% 文本 prompt 为空文本，鼓励模型从 LQ 图像本身感知语义。训练 500K 次迭代，batch size 4，学习率 $5 \times 10^{-5}$。

## 实验关键数据

### 主实验

**Real-ISR 定量比较（DIV2K / RealSR / DRealSR）**：

| 方法 | FID↓ (RealSR) | LPIPS↓ | MUSIQ↑ | QAlign↑ |
|------|---------------|--------|--------|---------|
| RealESRGAN | 67.02 | 0.2729 | 59.69 | 3.92 |
| StableSR | 109.11 | 0.2565 | 60.71 | 3.87 |
| DiffBIR | 55.17 | 0.3633 | 65.52 | 4.10 |
| SeeSR | 58.32 | 0.2796 | 64.27 | 3.89 |
| **PASD** | **47.34** | **0.2806** | **65.60** | **4.13** |

**卡通化定量比较（FFHQ + Flicker2K）**：

| 方法 | FID↓ (FFHQ) | MUSIQ↑ | QAlign↑ |
|------|-------------|--------|---------|
| CartoonGAN | 53.75 | 71.98 | 3.61 |
| InstructPix2Pix | 39.33 | 72.95 | 3.90 |
| ControlNet | 37.96 | 74.90 | 4.01 |
| **PASD** | **37.67** | **75.02** | **4.00** |

**老照片修复定量比较**：

| 方法 | FID↓ | MUSIQ↑ | QAlign↑ |
|------|------|--------|---------|
| RealESRGAN | 265.54 | 59.47 | 3.67 |
| Wan et al. | 268.35 | 32.15 | 3.01 |
| DiffBIR | 262.18 | 60.39 | 3.92 |
| **PASD** | **240.26** | **64.40** | **3.98** |

### 消融实验

**PASD 各组件消融（RealSR 数据集）**：

| 组件配置 | PSNR↑ | FID↓ | LPIPS↓ | 推理时间(s)↓ |
|---------|-------|------|--------|------------|
| 无 PACA（仅零卷积） | 26.11 | 56.79 | 0.3822 | 14.32 |
| +退化去除 | 27.87 | 53.90 | 0.3080 | 8.04 |
| +退化去除+高层信息 | 27.09 | 52.34 | 0.2851 | 8.74 |
| +退化去除+负提示 | 27.38 | 50.25 | 0.2809 | 13.32 |
| **完整 PASD** | 25.93 | **47.34** | **0.2806** | 14.59 |

### 关键发现

- PACA 是结构保持的关键：无 PACA 时输出与输入在颜色和结构上不一致
- 退化去除模块使输出更干净，PSNR 提升 1.76dB
- 负提示词（"noisy", "blurry", "low resolution"）对视觉质量贡献显著，FID 降低 3.65
- PASD 在三个 Real-ISR 数据集上 QAlign 得分全部最高，用户研究中获最多 rank-1 票
- 风格化任务无需额外训练，直接替换基座模型即可

## 亮点与洞察

1. **PACA 解耦像素控制与潜空间操作**：无需跳跃连接即可实现像素级结构保持，使模型同时适用于 Real-ISR 和潜空间风格化任务
2. **ANS 提供灵活调节**：一个超参数 $\bar{\alpha}_a$ 即可在推理时控制感知-保真度平衡，无需重新训练
3. **一模型多用途**：同一 PASD 模型通过替换基座模型实现 Real-ISR、卡通化、老照片修复，展现了极强的灵活性
4. **高层语义有效**：利用 ResNet 分类 + YOLO 检测 + BLIP 描述提供语义引导，比单纯空文本显著提升效果

## 局限性

- 保真度和感知质量仍存在 trade-off，严重退化或语义信息不准确时可能生成不忠实细节
- 使用 classifier-free guidance 时推理时间近乎翻倍（约 14.6s vs 8.0s）
- 高层信息提取依赖多个预训练模型（ResNet、YOLO、BLIP），增加了系统复杂度

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性：★★★★ — PACA 模块在不引入跳跃连接的前提下实现像素级感知，设计巧妙
- 技术：★★★★★ — 统一框架覆盖超分/风格化/修复，各模块均有理论支撑
- 实验：★★★★★ — 三个任务全面评估，消融完整，用户研究增强说服力
- 实用性：★★★★ — 代码开源，可直接利用社区个性化模型

<!-- RELATED:START -->

## 相关论文

- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)

<!-- RELATED:END -->
