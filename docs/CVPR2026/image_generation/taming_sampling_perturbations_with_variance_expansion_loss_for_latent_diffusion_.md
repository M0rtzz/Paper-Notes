---
title: >-
  [论文解读] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models
description: >-
  [CVPR 2026][图像生成][潜在扩散模型] 揭示了潜在扩散模型中β-VAE tokenizer因方差坍缩导致潜空间过于紧凑、对扩散采样扰动极敏感的问题，提出Variance Expansion (VE) Loss通过重构与方差扩展的对抗式平衡来自适应学习鲁棒的潜空间方差，在多种扩散架构上一致提升生成质量（FID 1.18）。
tags:
  - CVPR 2026
  - 图像生成
  - 潜在扩散模型
  - 方差扩展损失
  - 采样鲁棒性
  - 方差坍缩
  - VAE tokenizer
---

# Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2603.21085](https://arxiv.org/abs/2603.21085)  
**代码**: https://github.com/CVL-UESTC/VE-Loss  
**领域**: 扩散模型  
**关键词**: 潜在扩散模型、方差扩展损失、采样鲁棒性、方差坍缩、VAE tokenizer

## 一句话总结
揭示了潜在扩散模型中β-VAE tokenizer因方差坍缩导致潜空间过于紧凑、对扩散采样扰动极敏感的问题，提出Variance Expansion (VE) Loss通过重构与方差扩展的对抗式平衡来自适应学习鲁棒的潜空间方差，在多种扩散架构上一致提升生成质量（FID 1.18）。

## 研究背景与动机

1. **领域现状**：潜在扩散模型（LDM）已成为高质量图像生成的主流范式。其核心是先用 autoencoder（通常是β-VAE）将图像编码到潜空间，然后在潜空间训练扩散/flow模型。近期工作如VA-VAE、MAETok、DC-AE 1.5通过对齐语义先验或引入自监督目标来改善潜空间的语义结构。
2. **现有痛点**：重构精度和语义对齐之外，还有一个被忽视的关键因素——潜空间对扩散采样扰动的鲁棒性。一个反直觉的现象：重构更好、扩散loss更低的tokenizer，其生成质量反而更差。原因是标准β-VAE中KL项权重极小（如$10^{-6}$），潜空间方差$\sigma^2$趋近于零，潜流形过于紧凑，采样时微小的随机扰动就能把样本推到流形外，导致解码失败。
3. **核心矛盾**：重构损失天然地驱动方差坍缩（$\partial \mathcal{L}_{rec}/\partial \sigma \approx 2\sigma T(\mu)$，总是把$\sigma$推向0），而扩散采样过程固有的随机性又要求潜空间有足够的鲁棒性容纳扰动。传统KL正则化虽然能增大方差，但对重构质量伤害太大（强制对齐标准高斯先验）。
4. **本文目标** 构建一个既保持高重构保真度、又对扩散采样扰动鲁棒的潜空间。
5. **切入角度**：从一阶Taylor展开分析重构损失对方差的梯度，推导出方差坍缩的理论机制，然后设计一个反向梯度来对抗坍缩。
6. **核心 idea**：用VE Loss ($\mathcal{L}_{var} = 1/(\sigma^2+\delta)$) 的强反向梯度对抗重构损失的方差坍缩趋势，通过两者的自然对抗实现自适应方差平衡。

## 方法详解

### 整体框架
在标准VAE tokenizer训练中，用VE Loss替换KL散度项。编码器输出高斯分布 $\mathcal{N}(\mu, \sigma^2)$，通过重参数化采样后送入解码器重构。训练目标 = 重构损失 + $\lambda_1 \cdot$ 方差扩展损失 + $\lambda_2 \cdot$ 正则化损失。训练好的tokenizer搭配任意扩散模型（DiT/LightningDiT/SiT）使用。

### 关键设计

1. **方差坍缩的理论分析**:

    - 功能：从理论上解释为什么标准β-VAE的潜空间不适合扩散采样
    - 核心思路：对解码器做一阶Taylor展开，$\mathcal{D}(\mu+\sigma\epsilon) \approx \mathcal{D}(\mu) + J(\mu)\sigma\epsilon$，代入期望重构误差得到 $\mathcal{L}_{rec}(\mu,\sigma) \approx \|\mathbf{X}_0 - \mathcal{D}(\mu)\|^2 + \sigma^2 \cdot \text{Tr}(J(\mu)J(\mu)^\top)$。重构损失关于 $\sigma$ 的梯度为 $2\sigma T(\mu)$，始终将 $\sigma$ 推向零。当KL权重极小时，$\sigma^2$ 趋近$10^{-8}$，潜流形变成极细的"针状"结构，扩散采样的随机扰动轻易就超出流形边界。
    - 设计动机：提供从第一性原理出发的方差坍缩解释，为后续VE Loss的设计提供理论基础。

2. **Variance Expansion (VE) Loss**:

    - 功能：对抗方差坍缩，维持健康的潜空间方差
    - 核心思路：设计反方差损失 $\mathcal{L}_{var}(\sigma) = 1/(\sigma^2 + \delta)$，其对$\sigma$的梯度为$-2\lambda/\sigma^3$，在$\sigma$小时提供极强的反向推力。与重构梯度平衡后达到均衡 $\sigma = (\lambda/T(\mu))^{1/4}$，即方差自适应地反比于解码器局部灵敏度$T(\mu)$的四次根。作者比较了三种候选形式：(i) 负方差$-\alpha\sigma^2$：梯度在$\sigma\to 0$时消失无保护力；(ii) 对数熵$\log\sigma^2$：均衡点$\sigma^2 \propto 1/T(\mu)$理论可行但小$\sigma$区保护力不足；(iii) 反方差$1/(\sigma^2+\delta)$（选定）：$\sigma\to 0$时保护力最强。
    - 设计动机：不同于KL正则化强制对齐固定高斯先验（破坏重构），VE Loss只关心方差不要太小，通过与重构损失的自然对抗实现自适应平衡——对解码器敏感区域保持小方差（精确重构），对不敏感区域允许大方差（增强鲁棒性）。

3. **正则化项**:

    - 功能：防止方差扩展导致潜变量整体幅度膨胀
    - 核心思路：$\mathcal{L}_{reg} = e^{|z|-\tau}$，当$|z|$超过阈值$\tau$时施加指数级惩罚。$\tau=1$，$\lambda_2=10^{-6}$。
    - 设计动机：VE Loss增大方差的同时可能导致$z$的绝对值增大，需要一个温和的约束来维持潜变量在合理范围内。

### 损失函数 / 训练策略
- 总损失 $\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{var} + \lambda_2 \mathcal{L}_{reg}$
- $\lambda_1 = 0.1$，$\lambda_2 = 10^{-6}$，$\tau = 1$
- Tokenizer架构和训练策略沿用VA-VAE，降采样因子16
- 消融实验tokenizer训练16 epochs；SOTA版本在VA-VAE基础上微调5 epochs
- 扩散模型用DiT/LightningDiT/SiT，flow matching目标
- SOTA配置：LightningDiT-XL (675M) + Muon优化器（解决DINOv2对齐的训练不稳定）

## 实验关键数据

### 主实验 - SOTA对比 (ImageNet 256×256, CFG)

| 方法 | 参数量 | 训练epochs | gFID↓ | IS↑ | Recall↑ |
|------|--------|-----------|-------|-----|---------|
| DiT | 675M | 1400 | 2.27 | 278.2 | 0.57 |
| MDTv2 | 675M | 1080 | 1.58 | 314.7 | 0.65 |
| REPA | 675M | 800 | 1.42 | 305.7 | 0.65 |
| VA-VAE | 675M | 800 | 1.35 | 295.3 | 0.65 |
| RAE | 675M | 800 | 1.41 | 309.4 | 0.63 |
| **VE Loss (本文)** | **675M** | **530** | **1.18** | 289.8 | **0.66** |

### 消融实验 - VE Loss在不同tokenizer上的效果

| Tokenizer | 训练epochs | rFID↓ | PSNR↑ | FID-10K (DiT-B)↓ | FID-10K (LightningDiT-B)↓ |
|-----------|-----------|-------|-------|-------------------|--------------------------|
| LDM | 10 | 0.55 | 26.05 | 31.93 | 22.25 |
| LDM + VE | 10 | 0.60 | 25.23 | **29.03** | **19.70** |
| VAVAE | 16 | 0.35 | 27.43 | 22.27 | 19.85 |
| VAVAE + VE | 16 | 0.45 | 26.54 | **19.42** | **15.50** |

### KL正则化的局限性

| KL权重β | 潜方差$\sigma^2$ | rFID↓ | PSNR↑ | FID-10K↓ |
|---------|-----------------|-------|-------|----------|
| $10^{-6}$ | $10^{-8}$ | 0.39 | 27.12 | 23.12 |
| $10^{-2}$ | $10^{-5}$ | 0.44 | 26.71 | 22.87 |
| 1 | 0.07 | 0.61 | 25.45 | 23.18 |
| 8 | 0.94 | 2.36 | 22.29 | 27.54 |
| **VE Loss** | **0.06** | **0.46** | **26.31** | **18.90** |

### 关键发现
- **VE Loss一致性提升**：在vanilla LDM和VA-VAE两种tokenizer上都显著降低FID，说明这不是特定于某种tokenizer的技巧，而是普适的潜空间改进。
- **KL正则化的两难**：调大KL权重虽增大方差但严重伤害重构（β=8时rFID从0.39暴涨到2.36），导致FID反升。VE Loss在$\sigma^2=0.06$的适度方差下就实现了FID 18.90的最优性能。
- **训练效率**：SOTA配置只用530 epochs就超越了VA-VAE的800 epochs（FID 1.18 vs 1.35），tokenizer只微调5 epochs vs 从头训50 epochs。
- **微调即可见效**：在已训练好的VA-VAE上追加10 epochs VE Loss微调就改善了重构和生成（rFID 0.28→0.26, PSNR 27.71→28.31）。

## 亮点与洞察
- **被忽视的第三维度**：重构精度、语义对齐之外，"采样鲁棒性"是潜空间设计的第三个关键维度。这个发现对整个LDM社区有重要启示——好的tokenizer不仅要重构好、语义好，还要"扛扰动"。
- **对抗式自适应平衡**：VE Loss和重构损失天然形成对抗关系，无需手动调节每个位置的方差——解码器敏感区自动保持小方差，不敏感区自动放大方差。这种利用损失函数之间内在冲突来实现自适应的思路非常优雅。
- **理论分析完整且有指导性**：从方差坍缩的一阶分析到均衡点的推导，每一步都有清晰的物理直觉。三种候选VE形式的比较也非常instructive。

## 局限与展望
- 当前仅在ImageNet 256×256上验证，更高分辨率（512/1024）和视频生成的效果未知
- VE Loss引入了3个超参数（$\lambda_1$, $\lambda_2$, $\tau$），虽然设置较稳定但缺乏自动调节机制
- 理论分析基于一阶Taylor展开（$\sigma$小时成立），当$\sigma$被显著增大后高阶项的影响未讨论
- 与MAETok、DC-AE 1.5等使用自监督目标的tokenizer的结合效果未探索

## 相关工作与启发
- **vs KL正则化**: KL强制对齐固定高斯先验，重构-方差是刚性trade-off；VE Loss只防方差坍缩，通过自然对抗实现柔性自适应平衡
- **vs RAE (σ-VAE)**: RAE通过固定方差注入噪声来增强鲁棒性，但固定方差是全局的、不自适应的；VE Loss学出的方差随解码器敏感度变化
- **vs GIVT**: GIVT通过增大KL权重来增大方差，但受限于KL的重构惩罚性质；VE Loss脱离了KL框架，更灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 识别出"采样鲁棒性"这一被忽视的维度，VE Loss设计有理论基础
- 实验充分度: ⭐⭐⭐⭐ 多种tokenizer和扩散架构的消融，SOTA对比充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析清晰优雅，toy example的可视化直观有力
- 价值: ⭐⭐⭐⭐⭐ FID 1.18刷新SOTA，且方法简单通用，对LDM社区启示重大

<!-- RELATED:START -->

## 相关论文

- [DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)
- [Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)
- [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)

<!-- RELATED:END -->
