---
title: >-
  [论文解读] gQIR: Generative Quanta Image Reconstruction
description: >-
  [CVPR 2026][图像生成][单光子传感器] 将大规模 text-to-image latent diffusion model 适配到单光子雪崩二极管（SPAD）的极端光子受限成像场景，通过三阶段框架（Quanta-aligned VAE → 对抗微调 LoRA U-Net → FusionViT 时空融合）实现从稀疏二值光子检测到高质量 RGB 图像的重建，在 10K-100K fps 极端条件下显著超越所有现有方法。
tags:
  - CVPR 2026
  - 图像生成
  - 单光子传感器
  - 扩散模型
  - 图像重建
  - burst imaging
  - VAE对齐
---

# gQIR: Generative Quanta Image Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2602.20417](https://arxiv.org/abs/2602.20417)  
**代码**: [GitHub](https://github.com/Aryan-Garg/gQIR)  
**领域**: 图像生成 / 图像重建 / 计算成像  
**关键词**: 单光子传感器, 扩散模型, 图像重建, burst imaging, VAE对齐

## 一句话总结

将大规模 text-to-image latent diffusion model 适配到单光子雪崩二极管（SPAD）的极端光子受限成像场景，通过三阶段框架（Quanta-aligned VAE → 对抗微调 LoRA U-Net → FusionViT 时空融合）实现从稀疏二值光子检测到高质量 RGB 图像的重建，在 10K-100K fps 极端条件下显著超越所有现有方法。

## 研究背景与动机

**领域现状**：SPAD 单光子传感器能在传统相机失败的极低光照和高速场景中成像，但每个像素仅记录二值光子检测（有/无光子到达），单帧数据极度稀疏且噪声主导。现有方法主要分两类：(1) 经典视觉方法如 QBP，通过块匹配+帧对齐+Wiener 滤波重建；(2) 学习方法如 QUIVER 和 QuDI，用光流估计+循环融合或时间条件 U-Net 处理 burst 序列。

**现有痛点**：
- 经典方法在光子极度稀缺时运动估计不可靠，且缺乏语义先验导致高频细节丢失
- 学习方法虽引入 learned modules，但任务特定、从零训练，未利用大规模预训练生成模型的结构和语义知识
- 所有现有方法在极端形变或超高速运动（>10K fps）下严重退化
- color SPAD 的 Bayer mosaic 进一步加剧稀疏性，每个颜色通道光子事件更少

**核心矛盾**：大规模 T2I 扩散模型拥有强大的自然图像先验，但它们假设连续高斯噪声，而 SPAD 数据服从离散 Bernoulli 分布、噪声远超常规摄影——朴素微调会导致 encoder 崩溃（catastrophic forgetting）和 shortcut learning。

**本文切入角度**：不从零训练，而是通过精心设计的分阶段适配策略，将 Stable Diffusion 的结构先验迁移到 quanta 域。关键观察：VAE encoder 直接微调会因 SPAD 的极端噪声而退化为常数输出，必须引入 latent space 对齐约束来防止崩溃。

**核心 idea**：三阶段模块化框架——先对齐 latent space 处理 Bernoulli 统计特性，再蒸馏扩散先验为单步生成器增强感知质量，最后在 latent space 做 burst 级时空融合。

## 方法详解

### 整体框架

输入为 SPAD 传感器的 binary photon frames（或其 aggregated 3-bit nano-burst），输出为高质量 RGB 图像。Pipeline 分三个顺序训练的阶段：Stage 1 训练 quanta-aligned VAE encoder 实现去噪+去马赛克；Stage 2 对抗训练 LoRA U-Net 增强感知保真度；Stage 3 训练 FusionViT 在 latent space 做 burst 级时空融合。每个阶段冻结前序模块的参数。

### 关键设计

1. **图像形成模型（Bernoulli SPAD Simulator）**

    - 功能：建立从 clean sRGB 图像到 SPAD 观测的物理一致模拟管线
    - 核心思路：clean 图像 $x_{gt}$ 经 gamma 校正映射到线性辐射空间 $x_{lin} = x_{gt}^{2.2}$，然后每个像素的光子检测服从 Bernoulli 分布 $x_{spad} = \text{Bern}(1 - e^{-\alpha \cdot x_{lin}})$，其中 $\alpha$ 控制每像素期望光子数（PPP）。对于 color SPAD，施加随机 Bayer pattern 产生 mosaiced binary frame。$N$ 帧平均得到 $\log_2(N+1)$ bit 观测
    - 设计动机：与传统高斯噪声模型完全不同的 Bernoulli 统计特性是本文所有技术创新的出发点。训练时用 $\alpha=1.0$（PPP≈3.5）

2. **Stage 1: Quanta-Aligned VAE**

    - 功能：微调预训练 SD VAE 的 encoder，使其能从极度噪声的 SPAD 输入产生与 clean 图像对齐的 latent code
    - 核心思路：冻结 decoder $\mathcal{D}$，微调 encoder $\mathcal{E}_{\phi^*}$。两个关键修改防止 encoder 崩溃：
        - **确定性均值编码**：不从后验分布采样，直接使用均值 $\mu_\phi(x_{lq})$，避免 Bernoulli 噪声下的方差放大
        - **Latent Space Alignment (LSA) Loss**：$\mathcal{L}_{lsa} = \|\mu_{\phi^*}(x_{lq}) - \mu_\phi(x_{gt})\|_2^2$，其中第二项使用**冻结的预训练 encoder** 对 GT 编码——这与现有方法 SUPIR/DiffBIR 用同一个可训练 encoder 不同，避免了 predegradation removal loss 导致的崩溃
    - 总损失：$\mathcal{L} = \lambda_{lsa}\mathcal{L}_{lsa} + \lambda_{MSE}\mathcal{L}_{MSE} + \lambda_{perc}\mathcal{L}_{perc}$
    - 设计动机：直接用现有方法的微调目标（如 SUPIR 的 predegradation removal loss），encoder 会学到一个 shortcut——不管输入什么都输出近似相同的 latent code，因为训练 encoder 同时控制 supervision 和 prediction 两端，在极端噪声下迅速收敛到退化解

3. **Stage 2: 对抗微调 LoRA U-Net（感知增强）**

    - 功能：将 SD 的扩散 U-Net 蒸馏为单步生成器，增强重建的高频细节和感知质量
    - 核心思路：用 LoRA adapter 初始化 generator $\mathcal{G}_{lora}$（继承扩散权重保证小初始梯度），设计多层 ConvNext-Large 判别器 $\mathcal{V}_\theta$，通过标准 min-max GAN 目标对抗训练：$\min_\phi \max_\theta \mathbb{E}[\log \mathcal{V}_\theta(x)] + \mathbb{E}[\log(1 - \mathcal{V}_\theta(\mathcal{G}(x)))]$
    - 总损失：$\mathcal{L} = \mathcal{L}_{adv} + \mathcal{L}_{perc} + \|\mathcal{D}(\mathcal{G}_{lora}(\mu_{\phi^*}(x_{lq}))) - x_{gt}\|_2^2$
    - 设计动机：SPAD 超高帧率（10K-100K fps）产生海量数据处理需求，多步扩散采样不现实；对抗蒸馏方案将扩散先验压缩为单步推理，既保留了预训练知识又满足实时性需求

4. **Stage 3: FusionViT（Latent Burst 时空融合）**

    - 功能：利用 burst 序列的时间冗余信息，在 latent space 对齐并动态融合多帧，提高重建保真度并减轻时域 artifacts
    - 核心思路：
        - 先用 Stage 1+2 重建所有帧 $Y = \mathcal{D}(\mathcal{G}_{lora}(\mathcal{E}_{\phi^*}(X_{lq})))$，在重建图像上用预训练 RAFT 估计光流（直接在 noisy SPAD 上估计光流会因域差距严重失败）
        - 将所有 burst latent maps 按光流 warp 到 center frame
        - 引入 pseudo-3D miniViT $\mathcal{F}$，用亚二次复杂度窗口注意力跨时间和空间轴做动态融合（naive 平均会因运动产生模糊）
        - FusionViT 输出通过可学习标量 $\delta$（初始 0.05）残差加到 center latent $z_{T/2}$ 上
    - 损失：$\mathcal{L}_{fusion} = \|\mathcal{F}(\mu_{\phi^*}(X_{lq})) - \mu_\phi(x_{gt})\|_2^2 + \text{pixel MSE} + \mathcal{L}_{perc}$
    - 设计动机：朴素 flow-warp + 平均在运动场景下严重模糊；FusionViT 通过注意力机制自适应地根据运动幅度和到 center frame 的距离加权不同帧的贡献

### 损失函数 / 训练策略

三阶段渐进训练，每阶段冻结前序模块：
- **Stage 1**：8×A100，600K steps，$\lambda_{lsa}=0.1, \lambda_{MSE}=10^3, \lambda_{perc}=2$
- **Stage 2**：单张 RTX 4090，100K iterations，256×256，$\lambda_{adv}=0.5, \lambda_{MSE}=500, \lambda_{perc}=5$
- **Stage 3**：FusionViT 仅 20K steps，使用 RAFT 光流（在 FlyingThings3D 预训练）

训练数据：281 万图像 + 44,575 个视频，涵盖图像超分、人脸、视频去模糊等多种数据源。3-bit nano-burst 由 7 帧二值帧平均得到；burst 模式使用 11 个 3-bit nano-burst（共 77 帧二值帧）。

## 实验关键数据

### 主实验

**单帧 3-bit 重建（334 张测试图像，384×384）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | ManIQA↑ | ClipIQA↑ | MUSIQ↑ |
|------|-------|-------|--------|---------|----------|--------|
| InstantIR | 10.79 | 0.178 | 0.651 | 0.187 | 0.346 | 36.65 |
| ft-Restormer | 28.73 | 0.816 | 0.294 | 0.262 | 0.435 | 40.44 |
| ft-NAFNet | 28.28 | 0.830 | 0.261 | 0.300 | 0.473 | 39.13 |
| qVAE (Stage 1) | 28.18 | **0.863** | 0.327 | 0.299 | 0.487 | 44.90 |
| **gQIR (S1+S2)** | 27.28 | 0.839 | **0.318** | **0.331** | **0.547** | **45.61** |

**Burst 重建（极端运动场景）**：

| 数据集 (fps) | QBP PSNR/SSIM | QUIVER PSNR/SSIM | **Burst-gQIR PSNR/SSIM** |
|-------------|---------------|-------------------|--------------------------|
| XVFI (1000) | 12.01 / 0.370 | 23.00 / 0.751 | **25.82 / 0.712** |
| I2-2000fps (2000) | 16.04 / 0.549 | 25.06 / 0.874 | **31.21 / 0.878** |
| XD (2K-100K) | 12.78 / 0.409 | 20.10 / 0.790 | **30.33 / 0.895** |
| **累计** | 13.38 / 0.448 | 22.43 / 0.814 | **29.83 / 0.856** |

**I2-2000fps 完整测试集**：Burst-gQIR 达到 30.81 dB PSNR / 0.868 SSIM，超越前 SOTA QuDI (28.64 dB) **+2.17 dB**。

### 消融实验

**Stage 1 设计选择消融**：

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o 确定性编码 (A) | 20.56 | 0.435 | 0.167 |
| w/o LSA loss (B) | 10.39 | 0.222 | 0.139 |
| w/o (A) + (B) | 10.30 | 0.218 | 0.136 |
| **完整 Stage 1** | **24.78** | **0.665** | **0.194** |

**三阶段渐进提升 - 保真度 vs 时域稳定性**：

| 阶段 | PSNR↑ | SSIM↑ | $E^*_{warp}$↓ |
|------|-------|-------|-------------|
| Alignment (S1) | 20.04 | 0.759 | 9.088 |
| Perceptual (S2) | 24.11 | 0.846 | 8.508 |
| **Fidelity (S3)** | **27.63** | **0.869** | **8.005** |

### 关键发现

- **LSA loss 是 Stage 1 成功的关键**：去掉 LSA 后 PSNR 从 24.78 暴跌到 10.39（接近随机输出），encoder 退化为常数映射——验证了 Bernoulli 噪声下 latent space 对齐的必要性
- 确定性编码也很重要，去掉后 PSNR 降 ~4 dB，但影响小于 LSA
- Stage 2 的对抗训练在提升感知质量的同时略微增加了内容漂移（content drift），但 Stage 3 的时空融合有效缓解了这一问题
- 在极端运动数据集 XD 上，gQIR 相对 QBP 提升 **+17.5 dB**，体现了生成先验在 out-of-distribution 场景的巨大优势
- 在真实 color SPAD 数据上无需 dark count 或 hot pixel 校正即可重建出照片级质量，仅需灰世界白平衡

## 亮点与洞察

- **首次将大规模 T2I 扩散先验成功迁移到单光子成像**：这是一个噪声统计完全不同的域（Bernoulli vs Gaussian），关键突破在于发现并解决了 encoder collapse 问题
- **冻结预训练 encoder 做 LSA 对齐目标的设计非常巧妙**：对比 SUPIR/DiffBIR 的做法，他们用同一个可训练 encoder 同时生成 supervision 和 prediction，在极端噪声下这会导致退化——本文用冻结副本切断了这条 shortcut 路径
- **三阶段解耦设计各有侧重**：S1 解决域适配（结构+色彩）、S2 增强感知（高频细节）、S3 利用时域信息（稳定性+保真度），职责清晰且可独立优化
- **在重建域而非噪声域做光流估计**的 pre-denoising 策略绕过了 SPAD 数据上光流估计失败的问题

## 局限与展望

- 固定 PPP=3.5 训练，在极低光照（PPP≤1）下鲁棒性受限——将 PPP 作为 conditioning signal 显式建模可能提升泛化性
- 预训练 VAE decoder 8-bit 输出限制了 SPAD 原生的 HDR 能力，开发 HDR-capable decoder 是重要方向
- Stage 2 的 content drift 虽被 Stage 3 缓解但未根本解决，视频级扩散先验可能进一步改善时域一致性
- 当前 Stage 3 依赖 RAFT 光流，在超大运动下仍可能失败，可探索 latent space 内的隐式对齐
- 仅在合成数据上定量评估，真实 SPAD 数据缺乏 GT 参考

## 相关工作与启发

- **vs QBP**: QBP 是经典 align-and-merge 管线（块匹配+Wiener 滤波），在光子充足时有效但缺乏语义先验。gQIR 将 align-and-merge 哲学推广到 latent space，用 FusionViT 替代简单平均，在极端运动下优势巨大（+17.5 dB）
- **vs QUIVER/QuDI**: QUIVER 用 SpyNet 光流+循环融合，QuDI 用时间条件 U-Net。两者都是任务特定训练且不利用预训练生成先验。gQIR 在 I2-2000fps 上超 QuDI +2.17 dB
- **vs SUPIR/DiffBIR（通用图像修复）**: 这些方法也微调 VAE encoder 适配退化输入，但在 Bernoulli 噪声下直接崩溃。gQIR 的 LSA loss + 冻结 encoder 目标是专门针对 SPAD 极端噪声的关键改进
- 三阶段解耦训练的思路可迁移到其他非标准噪声模型的传感器修复任务（如 event camera、量子传感器）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将 T2I diffusion 适配到 quanta imaging，发现并解决了 encoder collapse 的关键问题
- **实验充分度**: ⭐⭐⭐⭐ 合成+真实数据、单帧+burst、多 fps 级别、完整消融，还贡献了首个 color SPAD burst 数据集和 XD benchmark
- **写作质量**: ⭐⭐⭐⭐ 物理模型和方法动机交代清晰，encoder collapse 的分析可视化直观
- **价值**: ⭐⭐⭐⭐ 为计算成像领域引入生成先验开辟新方向，代码+数据集开源

<!-- RELATED:START -->

## 相关论文

- [PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)
- [Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)

<!-- RELATED:END -->
