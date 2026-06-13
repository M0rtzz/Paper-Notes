---
title: >-
  [论文解读] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models
description: >-
  [CVPR 2025][图像生成][4K图像生成] 本文提出 Diffusion-4K 框架，包含 Aesthetic-4K 基准数据集、GLCM Score/Compression Ratio 评估指标、以及基于小波变换的微调方法，使 SD3-2B 和 Flux-12B 等大规模潜在扩散模型能直接生成具有丰富纹理细节的 4096×4096 高质量图像。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "4K图像生成"
  - "小波变换微调"
  - "分区VAE"
  - "扩散模型"
  - "Aesthetic-4K基准"
---

# Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2503.18352](https://arxiv.org/abs/2503.18352)  
**代码**: [https://github.com/zhang0jhon/diffusion-4k](https://github.com/zhang0jhon/diffusion-4k)  
**领域**: 图像生成 / 超高分辨率  
**关键词**: 4K图像生成、小波变换微调、分区VAE、扩散模型、Aesthetic-4K基准

## 一句话总结
本文提出 Diffusion-4K 框架，包含 Aesthetic-4K 基准数据集、GLCM Score/Compression Ratio 评估指标、以及基于小波变换的微调方法，使 SD3-2B 和 Flux-12B 等大规模潜在扩散模型能直接生成具有丰富纹理细节的 4096×4096 高质量图像。

## 研究背景与动机

**领域现状**：主流潜在扩散模型（SD3、Flux 等）专注于 1024×1024 分辨率的训练和生成，直接4K图像合成基本未探索。PixArt-Σ 和 Sana 虽实现4K级生成，但主要关注效率，忽略了4K图像固有的高频细节和丰富纹理优势。

**现有痛点**：(1) 缺少公开的4K图像合成数据集和基准；(2) 常规评估指标（FID、CLIP Score）在低分辨率下计算，无法评估4K图像的局部细节质量；(3) 直接将 F=8 的 VAE 用于 4096×4096 会导致 OOM。

**核心矛盾**：提升分辨率到4K引入平方级计算开销，同时标准训练目标（噪声预测/速度预测）对高频细节没有显式关注，导致4K图像往往"大而模糊"。

**本文目标**：(1) 建立4K图像合成的完整基准；(2) 提出一种通用的微调方法，使各种LDM在4K下生成具有丰富细节的图像。

**切入角度**：小波变换可以将信号分解为低频近似和高频细节，对速度/噪声预测目标应用小波变换后，高频分量和低频分量同时参与损失计算，从而显式增强对细节的关注。

**核心 idea**：提出 Wavelet-based Latent Enhancement (WLF) 微调方法 + 分区 VAE (F=16) 解决 OOM + Aesthetic-4K 数据集和专用评估指标。

## 方法详解

### 整体框架
收集高质量4K图像构建 Aesthetic-4K 数据集，使用 GPT-4o 生成精确 caption。用分区 VAE 将图像压缩到 F=16 潜在空间。在此潜在空间中用小波变换损失微调预训练扩散模型（SD3/Flux），冻结 VAE 和文本编码器。

### 关键设计

1. **分区 VAE (Partitioned VAE)**:

    - 功能：解决 F=8 VAE 在4096×4096下 OOM 的问题
    - 核心思路：在 VAE 编码器第一层卷积使用 dilation rate=2 来实现额外2倍下采样（F=8→F=16）。解码器最后一层将特征图分区、分别上采样2倍、对每个分区应用同一卷积，再重组输出。完全复用预训练 VAE 参数，无需重新训练。
    - 设计动机：保持与预训练 LDM 的潜在空间一致性（避免分布偏移），同时将显存消耗降低到可行范围。

2. **基于小波变换的潜在增强 (WLF)**:

    - 功能：在微调目标中显式增强对高频细节的关注
    - 核心思路：对扩散模型的预测目标（速度/噪声）和真实目标应用 Haar 小波离散变换 (DWT)，分解为 LL（低频近似）、LH/HL/HH（高频细节）四个子带。损失函数 $\mathcal{L}_{WLF} = \mathbb{E}[w_t \|f(v_\Theta(z_t,t)) - f(\epsilon - x_0)\|^2]$，其中 $f(\cdot)$ 是 DWT。
    - 设计动机：标准 MSE 损失对所有频率一视同仁，在4K场景中高频成分的相对贡献被稀释。DWT 强制模型同时优化低频结构和高频纹理。

3. **Aesthetic-4K 基准**:

    - 功能：提供4K图像合成的训练、评估和指标体系
    - 核心思路：训练集 12,015 张高质量4K图像 + GPT-4o caption。评估集 2,781 张来自 LAION-Aesthetics。新指标 GLCM Score（灰度共生矩阵熵，衡量纹理丰富度）和 Compression Ratio（JPEG 压缩比，衡量细节保留），与人类感知对齐度（SRCC=0.75/0.53）远超 MUSIQ（0.36）和 MANIQA（0.20）。
    - 设计动机：填补4K图像合成评估的空白，提供以人类感知为中心的细节质量指标。

### 损失函数 / 训练策略
WLF 损失基于小波变换域的 MSE。使用 AdamW 优化器（lr=1e-6），混合精度训练。SD3-2B 用 2 张 A800-80G，Flux-12B 用 8 张 A100-80G。

## 实验关键数据

### 主实验
Aesthetic-Eval@2048 评估：

| 模型 | FID↓ | CLIPScore↑ | Aesthetics↑ |
|------|------|------------|-------------|
| SD3-F16 (baseline) | 43.82 | 31.50 | 5.91 |
| SD3-F16-WLF (本文) | 40.18 | 34.04 | 5.96 |
| Flux-F16 (baseline) | 50.57 | - | - |

### 细节质量指标

| 模型 | GLCM Score↑ | Compression Ratio↓ |
|------|-------------|-------------------|
| SD3-F16 | 基线 | 基线 |
| SD3-F16-WLF | 更高 | 更低（更多细节） |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 标准微调（无 WLF） | FID/CLIP 基线 | 4K 图像偏模糊 |
| + WLF 小波损失 | FID降低, CLIP提高 | 细节显著增强 |
| 分区 VAE (F=16) | rFID=1.40, PSNR=28.82 | 接近原 VAE 重建质量 |

### 关键发现
- WLF 在 FID、CLIPScore、Aesthetics 和细节指标上全面优于标准微调
- 分区 VAE 无需重训即可保持潜在空间一致性，4K 重建质量可接受
- GLCM Score 和 Compression Ratio 比现有 NR-IQA 指标更好地对齐人类对细节的感知
- 大模型（Flux-12B）在4K生成上优势更明显，验证了 DiT 在超高分辨率下的可扩展性

## 亮点与洞察
- **小波变换微调的通用性**：WLF 只修改损失函数，可以无缝适配任何 LDM（SD3、Flux、甚至 UNet 架构），是一种轻量但有效的方法。
- **分区 VAE 巧妙解决 OOM**：仅修改第一层和最后一层卷积的策略，无需重训 VAE，实用性极强。
- **以人类感知为中心的评估**：GLCM Score 和 Compression Ratio 填补了4K评估的空白。

## 局限与展望
- 训练集仅 12K 图像，数据规模有限
- 分区 VAE 的 F=16 压缩比更高，可能丢失部分精细信息
- 仅在文生图任务验证，图像编辑、视频生成等下游任务未探索
- 评估指标虽与人类感知对齐更好，但相关系数仍有提升空间

## 相关工作与启发
- **vs PixArt-Σ / Sana**: 关注4K效率但忽略细节质量，本文专注细节增强
- **vs Stable Cascade**: 用多级扩散逐步提升分辨率，可能累积误差；本文直接端到端4K生成

## 评分
- 新颖性: ⭐⭐⭐⭐ 小波微调+分区VAE+4K基准的完整体系
- 实验充分度: ⭐⭐⭐⭐ 多模型验证，新指标人类对齐分析充分
- 写作质量: ⭐⭐⭐⭐ 系统完整，各组件动机清晰
- 价值: ⭐⭐⭐⭐ 4K基准和WLF方法对社区有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion](ultrafusion_ultra_high_dynamic_imaging_using_exposure_fusion.md)
- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[NeurIPS 2025\] RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation](../../NeurIPS2025/image_generation/repldm_reprogramming_pretrained_latent_diffusion_models_for_high-quality_high-ef.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)

</div>

<!-- RELATED:END -->
