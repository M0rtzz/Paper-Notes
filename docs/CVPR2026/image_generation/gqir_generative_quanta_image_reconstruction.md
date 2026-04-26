---
title: >-
  [论文解读] gQIR: Generative Quanta Image Reconstruction
description: >-
  [CVPR 2026][图像生成][单光子成像] 提出 gQIR，一个模块化三阶段框架，将大规模 T2I 扩散模型适配到 SPAD 传感器的极端光子受限域，通过量子对齐 VAE（冻结编码器副本防坍缩）、对抗微调 LoRA U-Net（单步生成）和潜空间 FusionViT（时空融合），从极稀疏二值光子事件重建高质量彩色图像和视频。
tags:
  - CVPR 2026
  - 图像生成
  - 单光子成像
  - SPAD
  - 扩散模型
  - 去噪去马赛克
  - 低光重建
  - VAE对齐
  - burst融合
---

# gQIR: Generative Quanta Image Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2602.20417](https://arxiv.org/abs/2602.20417)  
**代码**: https://github.com/Aryan-Garg/gQIR  
**领域**: 图像生成 / 计算成像  
**关键词**: 单光子成像, SPAD, 扩散模型, 去噪去马赛克, 低光重建, VAE对齐, burst融合

## 一句话总结
提出 gQIR，一个模块化三阶段框架，将大规模 T2I 扩散模型适配到 SPAD 传感器的极端光子受限域，通过量子对齐 VAE（冻结编码器副本防坍缩）、对抗微调 LoRA U-Net（单步生成）和潜空间 FusionViT（时空融合），从极稀疏二值光子事件重建高质量彩色图像和视频。

## 研究背景与动机

**领域现状**：SPAD 传感器可在极低光照和超高帧率（10k–100k fps）条件下成像，每像素仅记录是否检测到光子（伯努利分布），原始 quanta 帧极度稀疏噪声大。现有重建方法（QBP、QUIVER、QuDI）采用对齐-合并策略或任务特定网络，未利用大规模生成先验。

**现有痛点**：(a) 传统去噪网络（NAFNet、Restormer）针对泊松-高斯噪声，面对伯努利量化噪声严重过度平滑；(b) 已有生成式恢复模型（InstantIR）在常规退化上表现良好但在光子受限域完全失效（PSNR 仅 7.9）；(c) 朴素微调扩散模型的 VAE 编码器会导致**编码器坍缩**——可训练编码器同时控制预测和监督两端，快速收敛到常数输出。

**核心矛盾**：伯努利噪声统计与扩散模型训练数据（连续自然图像）之间存在巨大域差距，直接 fine-tune 会导致 shortcut learning。

**切入角度**：发现退化预移除损失的对称性结构是编码器坍缩的根因，引入冻结编码器副本打破对称性。

**核心idea**：冻结编码器做 LSA 锚点防坍缩 + 对抗 LoRA 单步化 + 潜空间 FusionViT 时空融合。

## 方法详解

### 整体框架
输入为 SPAD 二值帧（nano-burst：7 帧平均得 3-bit），输出为高质量 RGB 图像。三阶段独立训练：Stage 1 微调 VAE 编码器实现潜空间去噪去马赛克对齐；Stage 2 对抗微调 LoRA U-Net 实现一步高感知质量生成；Stage 3 训练 FusionViT 在潜空间做多帧时空融合。

### 关键设计

1. **成像模型**:

    - 将干净 sRGB 做 gamma 校正 $x_{lin} = x_{gt}^{2.2}$ 映射到线性辐照空间
    - SPAD 输出服从伯努利分布：$x_{spad} = \text{Bern}(1 - e^{-\alpha \cdot x_{lin}})$，$\alpha=1.0$ 对应期望 PPP=3.5
    - 施加随机 Bayer 图案后 N 帧平均：$x_{lq} = \frac{1}{N}\sum_{i=1}^{N} M_\pi[\text{Bern}(1-e^{-\alpha \cdot x_{lin}})]$

2. **量子对齐 VAE（Stage 1）**:

    - **确定性均值编码**：用 $\mu_\phi(x_{lq})$ 代替随机采样，避免伯努利噪声下的方差放大
    - **潜空间对齐损失 (LSA)**：$\mathcal{L}_{lsa} = \|\mu_{\phi^*}(x_{lq}) - \mu_\phi(x_{gt})\|_2^2$，其中 $\mu_\phi$ 为**冻结**的原始预训练编码器副本，提供稳定锚点
    - 辅以像素空间 MSE ($\lambda=10^3$) 和 LPIPS ($\lambda=2$) 损失
    - 总损失：$\mathcal{L} = 0.1\mathcal{L}_{lsa} + 10^3\mathcal{L}_{MSE} + 2\mathcal{L}_{perc}$
    - **设计动机**：现有退化预移除让可训练编码器同时出现在预测和监督中导致退化最优，冻结副本从根本上打破对称性

3. **对抗 LoRA 微调（Stage 2）**:

    - 多层级 ConvNeXt-Large 判别器 $\mathcal{V}_\theta$ 对 LoRA 初始化的 U-Net 做标准 min-max GAN 训练
    - 生成器总损失：$\mathcal{L}_{G} = \mathcal{L}_{adv} + \mathcal{L}_{perc} + \|\mathcal{D}(G_{lora}(\mu_{\phi^*}(x_{lq}))) - x_{gt}\|_2^2$
    - **设计动机**：SPAD 极高采集速率要求单步推理；LoRA 初始化继承扩散权重保证 GAN 稳定训练起点

4. **潜空间 Burst 融合（Stage 3 - FusionViT）**:

    - 先用 S1+S2 重建所有帧 $Y$，再用 RAFT（FlyingThings3D 预训练）在重建域估光流，避免低质量输入的域差距
    - pseudo-3D miniViT 用亚二次窗口注意力在时间+空间轴自适应融合，避免朴素平均的运动模糊
    - 输出通过可学习标量 $\delta=0.05$ 与中心帧潜编码残差相加后送入 $\mathcal{G}_{lora}$
    - 训练损失：$\mathcal{L}_{fusion} = \|\mathcal{F}(\mu_{\phi^*}(X)) - \mu_\phi(x_{gt})\|_2^2 + \|\mathcal{D}(\mathcal{G}(\mathcal{F}(\mu_{\phi^*}(X)))) - x_{gt}\|_2^2 + \mathcal{L}_{perc}$

### 训练配置
- Stage 1：8×A100, 600k 步, batch=8, Adam(lr=1e-5), 训练数据 2.81M 图 + 44k 视频
- Stage 2：1×RTX4090, 100k 步, 256×256
- Stage 3：20k 步, 冻结 S1+S2, 仅训练 FusionViT

## 实验关键数据

### 主实验：单帧 3-bit 彩色重建

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | ManIQA↑ | ClipIQA↑ | MUSIQ↑ |
|------|-------|-------|--------|---------|----------|--------|
| InstantIR | 7.93 | 0.101 | 0.736 | 0.197 | 0.358 | 32.21 |
| ft-Restormer | 26.43 | 0.739 | 0.388 | 0.235 | 0.395 | 36.03 |
| ft-NAFNet | 26.88 | 0.757 | 0.338 | 0.251 | 0.431 | 36.73 |
| qVAE (S1) | 26.28 | 0.791 | 0.435 | 0.272 | 0.432 | 38.61 |
| **gQIR (S1+S2)** | 25.48 | 0.766 | **0.361** | **0.313** | **0.490** | **42.04** |

### Burst 重建对比

| 测试集 (fps) | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------------|------|-------|-------|--------|
| I2-2000fps | QBP | 16.04 | 0.549 | 0.468 |
| I2-2000fps | QUIVER | 25.06 | 0.874 | 0.366 |
| I2-2000fps | **Burst-gQIR** | **31.21** | **0.878** | **0.296** |
| XD (2k-100k) | QBP | 12.78 | 0.409 | 0.458 |
| XD (2k-100k) | **Burst-gQIR** | **30.33** | **0.895** | **0.316** |

### 消融实验：Stage 1 设计选择

| 变体 | PSNR↑ | SSIM↑ | ManIQA↑ |
|------|-------|-------|---------|
| w/o det. encoding | 20.56 | 0.435 | 0.167 |
| w/o LSA loss | 10.39 | 0.222 | 0.139 |
| w/o 两者 | 10.30 | 0.218 | 0.136 |
| **完整方法** | **24.78** | **0.665** | **0.194** |

### 消融实验：三阶段递进（视频集）

| 阶段 | PSNR↑ | SSIM↑ | $E_{warp}$↓ |
|------|-------|-------|-------------|
| S1 对齐 | 20.04 | 0.759 | 9.088 |
| S2 感知 | 24.11 | 0.846 | 8.508 |
| S3 融合 | **27.63** | **0.869** | **8.005** |

### 关键发现
- LSA 损失**必须**——去掉后 PSNR 从 24.78 骤降至 10.39，编码器 100% 坍缩
- I2-2000fps 基准上超越 SOTA QuDI +2.17 dB（30.81 vs 28.64）
- 对抗微调使感知指标大幅提升：ClipIQA 0.432→0.490（+13.4%），MUSIQ 38.6→42.0
- FusionViT 在极端运动 XD 数据集贡献最大，burst vs 单帧提升约 +5dB
- 真实彩色 SPAD 数据无需热像素/暗计数校正即可重建，仅需灰世界白平衡

## 亮点与洞察
- **冻结编码器副本防坍缩**是核心贡献，可直接迁移到任何退化感知 VAE 微调场景
- **伯努利噪声下确定性编码**避免方差放大，对非高斯噪声域的扩散适配有通用指导
- 首个**彩色 SPAD burst 数据集**和 XD 视频基准，填补评估空白
- 三阶段解耦训练使每阶段可独立优化，工程实用性强

## 局限性
- 训练假设固定 PPP=3.5，极低光（PPP≤1）泛化受限，需将 PPP 作为条件信号
- 预训练 VAE 解码器 8-bit 限制了 SPAD 原生 HDR 能力
- 对抗训练仅在 256×256 进行，高分辨率靠 VAE tiling
- Stage 3 依赖"先重建再估流"的两步策略，端到端方案可能更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 T2I 扩散先验成功适配到单光子成像
- 实验充分度: ⭐⭐⭐⭐⭐ 单帧+burst+真实数据+新数据集+多指标+完整消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰物理建模严谨
- 实用价值: ⭐⭐⭐⭐ 为极端成像条件下的生成先验适配开辟新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)

<!-- RELATED:END -->
