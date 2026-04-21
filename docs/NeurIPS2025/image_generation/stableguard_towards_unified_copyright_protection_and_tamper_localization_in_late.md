---
title: >-
  [论文解读] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][潜在扩散模型] 提出StableGuard，将全局二值水印嵌入LDM生成流程中（通过MPW-VAE），并利用水印扰动模式的变化实现篡改定位（通过MoE-GFN），首次实现端到端的版权保护与篡改检测统一框架。
tags:
  - NeurIPS 2025
  - 图像生成
  - 潜在扩散模型
  - 水印嵌入
  - 篡改定位
  - 混合专家
  - 自监督学习
---

# StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17993](https://arxiv.org/abs/2509.17993)  
**代码**: [GitHub](https://github.com/Harxis/StableGuard)  
**领域**: 图像水印 / 内容安全  
**关键词**: 潜在扩散模型, 水印嵌入, 篡改定位, 混合专家, 自监督学习

## 一句话总结

提出StableGuard，将全局二值水印嵌入LDM生成流程中（通过MPW-VAE），并利用水印扰动模式的变化实现篡改定位（通过MoE-GFN），首次实现端到端的版权保护与篡改检测统一框架。

## 研究背景与动机

潜在扩散模型(LDM)生成的图像越来越逼真，带来两大安全需求：（1）版权保护——确认图像是否由特定模型生成；（2）篡改定位——检测和定位图像被恶意编辑的区域。现有方法存在以下不足：

**传统水印方法**（HiDDeN、SepMark等）是事后处理(post-hoc)方式，在图像生成后单独嵌入水印，引入额外计算开销和图像质量下降。**扩散原生水印**（Stable Signature、WOUAF、WaDiff）虽然将水印集成到生成过程中，但不支持篡改定位。**统一方法**（EditGuard、OmniGuard、WAM）尝试同时解决两个问题，但仍然是事后处理，生成和取证分别优化，无法实现相互增强。

作者的核心洞察有两点：（1）全局分布的水印由于空间冗余性，对局部篡改具有天然鲁棒性；（2）水印引入的微弱扰动模式在被篡改区域会缺失，这种缺失特征可以作为篡改定位的可靠线索。这两个互补性质使得全局水印成为同时解决版权验证和精细篡改分析的理想基础。

## 方法详解

### 整体框架

StableGuard由两个核心组件构成：(1) MPW-VAE在LDM的VAE解码器中嵌入水印；(2) MoE-GFN利用水印扰动模式进行取证分析。两个组件端到端联合训练，实现嵌入质量和取证精度的相互增强。

### 关键设计

1. **复用水印VAE (MPW-VAE)**: 在预训练VAE解码器的每个block后插入轻量级水印适配器。适配器通过两层全连接层编码水印比特，reshape后与解码器特征拼接，再经两层卷积和残差连接进行融合。关键设计是**可切换机制**：水印适配器可以打开或关闭，从同一潜在编码生成视觉上无法区分的水印图像 $Y$ 和无水印图像 $\hat{X}$。这对配对图像通过随机二值掩码 $M$ 融合，以50%概率使用真实图像 $X$ 或VAE重建图像 $\hat{X}$ 作为拼接区域，模拟人工编辑和AI生成两种篡改场景。掩码策略混合使用随机二值掩码和SAM生成的语义掩码以增加多样性。

2. **混合专家引导取证网络 (MoE-GFN)**: 基于UNet架构，解码器中插入MoFE（混合取证专家）模块，包含三个专门化的专家分支：

    - **水印提取专家(WEE)**：基于Transformer捕获全局长距离依赖，即使局部被篡改也能恢复全局水印模式
    - **篡改定位专家(TLE)**：采用子补丁Transformer，将特征图分割为 $n \times n$ 小块进行局部注意力，检测细粒度的操控痕迹
    - **边界增强专家(BEE)**：在傅里叶域进行Transformer操作，通过FFT捕获边界区域的高频异常

   三个专家的输出通过**动态软路由器**(DSR)自适应融合：$x_{\text{unified}} = \sum_{n=1}^{3} R_n \odot \text{Expert}_n(x)$，其中 $R = \text{Softmax}(f(x))$ 根据输入自适应调整融合权重。

3. **自监督端到端训练**: 训练过程无需任何手动篡改标注。MPW-VAE的可切换机制自动生成配对训练数据，MoE-GFN学习区分水印区域和非水印区域。冻结预训练VAE解码器参数，仅训练水印适配器和MoE-GFN，确保图像生成质量不受影响。

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sim}} + \mathcal{L}_{\text{wm}} + \mathcal{L}_{\text{tamper}}$$

- **相似性损失** $\mathcal{L}_{\text{sim}} = \|{\hat{X} - Y}\|_1 + \text{PS}(\hat{X}, Y)$：L1距离+感知相似性，保持水印图像与原图的视觉一致性
- **水印损失** $\mathcal{L}_{\text{wm}}$：二值交叉熵，监督水印比特的准确提取
- **篡改损失** $\mathcal{L}_{\text{tamper}} = \lambda_0 \mathcal{L}_{\text{wbce}} + (1-\lambda_0) \mathcal{L}_{\text{dice}}$：加权二值交叉熵+Dice损失，前者处理前景/背景不平衡（$\lambda_1=2, \lambda_2=0.5$），后者优化区域重叠

基于SD 2.1，Adam优化器，学习率 $1 \times 10^{-4}$，2×RTX 4090D，batch=16，10 epochs。

## 实验关键数据

### 主实验

**水印提取性能（COCO + T2I数据集）**

| 方法 | 比特长度 | PSNR↑ | SSIM↑ | FID↓ | Bit Acc↑ |
|------|----------|-------|-------|------|----------|
| HiDDeN | 32 | 31.95 | 0.879 | 20.0 | 98.80 |
| EditGuard | 64 | 32.75 | 0.937 | 20.0 | 99.77 |
| WAM | 32 | 38.20 | 0.951 | 19.9 | 98.17 |
| OmniGuard | 100 | 37.54 | 0.950 | 20.1 | 98.11 |
| **StableGuard** | **32** | **40.50** | **0.970** | **19.5** | **99.97** |

PSNR提升2.3dB以上，Bit Acc达到99.97%，全面碾压所有基线。

**AIGC篡改定位（SD Inpainting / SDXL / Kandinsky / ControlNet / LAMA）**

| 方法 | F1↑ | AUC↑ | IoU↑ |
|------|-----|------|------|
| MVSS-Net | 0.862 | 0.934 | 0.791 |
| EditGuard | 0.937 | 0.977 | 0.911 |
| WAM | 0.924 | 0.977 | 0.868 |
| **StableGuard** | **0.980** | **0.993** | **0.962** |

### 消融实验

| 配置 | F1↑ | AUC↑ | Bit Acc↑ | 说明 |
|------|-----|------|----------|------|
| 完整模型（Dec位置） | 0.980 | 0.992 | 99.98 | 默认配置 |
| w/o MPW-VAE（用WOUAF替代） | 0.811 | 0.796 | 99.13 | 扩散原生水印至关重要 |
| w/o 整个MoFE模块 | 0.931 | 0.920 | 95.12 | 专家模块整体贡献显著 |
| w/o 水印提取专家 | 0.969 | 0.958 | 98.69 | 全局感知受损 |
| w/o 篡改定位专家 | 0.952 | 0.940 | 98.90 | 局部检测受损 |
| w/o 边界增强专家 | 0.962 | 0.950 | 99.11 | 边界精度下降 |
| w/o 动态软路由 | 0.966 | 0.955 | 98.97 | 简单求和不如自适应融合 |
| w/o 联合优化 | 0.921 | 0.919 | 99.14 | 解耦训练性能大幅下降 |

**鲁棒性（高斯噪声 σ=5 / JPEG Q=70）**

| 方法 | 噪声σ=5 Bit/F1 | JPEG Q=70 Bit/F1 |
|------|----------------|-----------------|
| EditGuard | 98.11/0.866 | 96.77/0.577 |
| StableGuard | **99.69/0.928** | **99.73/0.908** |

### 关键发现

- MPW-VAE的扩散原生设计是性能提升的最大贡献者——不仅生成更高保真度的配对图像，联合优化还带来协同增益
- 三个专家虽然单独移除时下降幅度有限（模型接近性能上限），但整体移除MoFE导致F1从0.980骤降至0.931，说明三者互补而非冗余
- JPEG压缩Q=70时，StableGuard的Bit Acc仍保持99.73%，而EditGuard降至96.77%
- 水印比特长度从32扩展到256时性能优雅下降（Bit Acc: 99.97→99.83）

## 亮点与洞察

- **自监督闭环**：可切换水印机制消除了对篡改标注数据的依赖，同时模拟了人工编辑和AI生成两种篡改类型
- **频域专家设计巧妙**：FFT域的Transformer捕获篡改边界的高频异常，与空间域专家形成互补
- **现实应用价值高**：直接集成到SD生成流程中，无需额外处理步骤，对生成质量影响极小（FID仅19.5）

## 局限与展望

- 目前基于SD 2.1验证，未测试在SDXL、SD3等更新架构上的适配性
- 水印适配器增加的参数量和推理开销未详细讨论
- 对于非LDM生成的图像（如GAN生成），该框架不适用
- 未探索水印安全性（如专门针对水印的攻击）

## 相关工作与启发

- 全局水印+缺失检测的思路可推广到视频扩散模型的逐帧篡改检测
- MoE在取证任务中的应用值得深入，不同专家可能对应不同的篡改类型
- 自监督配对数据生成策略可启发其他需要配对监督的视觉取证任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 端到端统一框架的思路新颖，可切换水印机制和MoFE模块设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖5种AIGC篡改方法、多种图像退化、详尽的消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分公式和符号在正文中较密集
- 价值: ⭐⭐⭐⭐⭐ 解决了扩散模型内容安全的实际痛点，性能大幅领先，实用性强

<!-- RELATED:START -->

## 相关论文

- [Blameless Users in a Clean Room: Defining Copyright Protection for Generative Models](blameless_users_in_a_clean_room_defining_copyright_protection_for_generative_mod.md)
- [Anti-Tamper Protection for Unauthorized Individual Image Generation](../../ICCV2025/image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)
- [Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)

<!-- RELATED:END -->
