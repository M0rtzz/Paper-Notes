---
title: >-
  [论文解读] Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP
description: >-
  [ICCV 2025][模型压缩][HDR Gain Map] 提出使用 10KB 轻量级 MLP 网络替代传统 JPEG/HEIC 压缩来编码 HDR gain map，以 SDR 图像的颜色和位置坐标 (r,g,b,x,y) 作为输入，结合指数残差编码（gamma map），在多个 HDR 重建指标上超越现有方法和传统压缩技术。
tags:
  - ICCV 2025
  - 模型压缩
  - HDR Gain Map
  - MLP
  - Implicit Neural Representation
  - Image Compression
  - Tone Mapping
---

# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP

**会议**: ICCV 2025  
**arXiv**: [2503.11883](https://arxiv.org/abs/2503.11883)  
**代码**: 无（论文提及接收后公开）  
**领域**: 模型压缩  
**关键词**: HDR Gain Map, MLP, Implicit Neural Representation, Image Compression, Tone Mapping

## 一句话总结

提出使用 10KB 轻量级 MLP 网络替代传统 JPEG/HEIC 压缩来编码 HDR gain map，以 SDR 图像的颜色和位置坐标 (r,g,b,x,y) 作为输入，结合指数残差编码（gamma map），在多个 HDR 重建指标上超越现有方法和传统压缩技术。

## 研究背景与动机

现代显示器（智能手机、平板）已普遍支持高动态范围（HDR）内容，现代相机也能原生捕获 HDR 图像。但为了与现有 SDR 工作流和传统显示器兼容，大部分图像仍以 SDR 格式（如 8-bit JPEG/sRGB）分发。为同时支持 SDR 和 HDR 设备，Apple EDR、Android UltraHDR、Samsung SuperHDR、Adobe gain map specification 和 ISO 21496 等新编码格式正在兴起——核心思路是在 SDR 图像中存储一个**像素级增益图（gain map）**作为元数据。当 HDR 显示器可用时，将 gain map 应用于 SDR 图像即可恢复 HDR 版本。

然而现有的 gain map 编码存在固有缺陷：gain map 需要被下采样到 1/4 分辨率、量化为 8-bit、再用 JPEG/HEIC/JPEG-XL 压缩。这一系列步骤不可避免地引入压缩伪影——**带状噪声（banding）、光晕效应（haloing）、块效应（blocking）**以及高频细节丢失。提高压缩质量虽可减少伪影，但会增大文件大小。

本文的核心洞察有两个：(1) 隐式神经表示（INR）作为连续函数逼近可以避免传统量化带来的问题；(2) MLP 不需要从零开始编码整个图像——它只需要编码一个**空间变化的 RGB 变换**（即 gain map），而 SDR 图像提供了大量先验信息。这使得 MLP 可以做得极其轻量（10KB）且训练极快（4秒/图像）。

## 方法详解

### 整体框架

给定 SDR 图像 $S$ 和 HDR 图像 $H$，传统方法计算 gain map $f(x,y) = (H+\epsilon)/(S+\epsilon)$，然后 log 归一化、量化、下采样、JPEG 压缩。解码时反向操作恢复 $H' = (S+\epsilon) \odot f'(x,y) - \epsilon$。本文用 MLP 替代传统压缩步骤，输入 SDR 图像的 $(x,y,r,g,b)$，直接预测对应的 gain map 值。

### 关键设计

1. **指数残差编码（Exponential Residual / Gamma Map）**:

    - 功能：将乘法残差（gain map）替换为指数残差（gamma map）
    - 核心思路：传统 gain map 用乘法关系：$f(x,y) = (H+\epsilon)/(S+\epsilon)$，解码为 $H' = (S+\epsilon) \odot f'(x,y) - \epsilon$。指数残差改为：$f(x,y) = \log(H+\epsilon)/\log(S+\epsilon)$，解码为 $H' = (S+\epsilon)^{f'(x,y)} - \epsilon$
    - 设计动机：指数残差更接近色调映射操作的本质——色调映射通常是非线性的幂函数变换。指数残差相当于一个更准确的预测编码近似，减少了 MLP 需要学习的残差复杂度。实验证明 Gamma-MLP 在所有比特率下都比 Gain-MLP 表现更好且更稳定

2. **轻量级 MLP 架构**:

    - 功能：用极小的 MLP 编码 gain/gamma map
    - 核心思路：5 维输入 $(x,y,r,g,b)$ → 每维 24 维正弦嵌入 → 120 维输入 → 两层 ReLU MLP（每层 16 神经元）→ 3 通道输出（RGB gain/gamma 值）。最终模型大小仅 **10KB**
    - 训练配置：batch size 65,536 随机采样像素，MSE 损失，Adam 优化器（lr=1e-2），1000 迭代，约 4 秒/图像（RTX 6000）
    - 设计动机：MLP 不是从 $(x,y)$ 预测 $(r,g,b)$（那样很慢），而是利用 SDR 图像的颜色作为强先验输入，因为 SDR 颜色与 gain map 高度相关。这使得训练极快，且模型可以极小

3. **色度噪声元初始化（Chromatic Noise Meta-Initialization）**:

    - 功能：为 MLP 提供更好的权重初始化
    - 核心思路：使用 Daly 等人提出的时空色度自然图像统计模型生成 50 张色度噪声图像，这些图像覆盖 Rec. 2020 色域，其 BT.709 SDR 对应版本用 DaVinci Resolve 默认色调映射处理。在这些合成数据上预训练 MLP 10,000 迭代获得元初始化权重
    - 设计动机：与用自然图像做元初始化不同，色度噪声图像保留了自然图像的统计特性（空间和色度相关性），同时避免了特定内容的偏差。这加速了后续单图优化的收敛并提高重建质量

### 损失函数 / 训练策略

- 损失：MSE loss，$\mathcal{L} = \|f'(x,y) - f(x,y)\|_2^2$，其中 $f$ 是真值 gain/gamma map，$f'$ 是 MLP 预测
- 元初始化后，每张图像单独微调 MLP（即 per-image optimization），1000 迭代即收敛
- 推理时：在所有像素坐标查询 MLP 得到完整 gain map，再应用到 SDR 图像恢复 HDR

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | ΔE₀₀↓ | SSIM↑ | ΔE_IPT↓ | HDR-VDP3↑ | 大小(KB) |
|------|-------|--------|-------|---------|-----------|----------|
| Gain-JPEG | 38.29 | 2.16 | 0.968 | 9.63 | 7.92 | 19.0 |
| Gamma-JPEG | 41.45 | 1.37 | 0.979 | 7.12 | 8.62 | 19.4 |
| Gain-HEIC | 39.20 | 1.98 | 0.972 | 8.71 | 8.14 | 18.4 |
| Gamma-HEIC | 42.21 | 1.27 | 0.982 | 6.57 | 8.75 | 18.2 |
| Direct-MLP [Le] | 46.30 | 0.96 | 0.988 | 4.66 | 9.06 | 10 |
| MLP-ITM [Liu] | 47.25 | 0.87 | 0.991 | 4.28 | **9.13** | 34 |
| **Gain-MLP (ours)** | 47.60 | 1.02 | 0.992 | 4.27 | 8.98 | **10** |
| **Gamma-MLP (ours)** | **48.53** | **0.78** | **0.993** | **3.91** | 9.11 | **10** |

### 消融实验（Rate-Distortion 性能，不同 MLP 大小和编码方式）

| 配置 | PSNR 趋势 | 说明 |
|------|----------|------|
| Gamma-MLP, 8 neurons | 最低比特率下仍优于 JPEG/HEIC 全分辨率 | 极低容量下指数残差优势最大 |
| Gamma-MLP, 16 neurons (默认) | 全范围最优 | 最佳性能/大小平衡 |
| Gamma-MLP, 64 neurons | 略升但边际递减 | 网络容量过大时收益有限 |
| Gamma-MLP, 128 neurons | 接近饱和 | 4 秒优化时间不变 |
| Gain-MLP vs Gamma-MLP | Gamma 在所有比特率上领先 | 指数残差一致优于乘法残差 |
| Gain-MLP vs Direct-MLP | Gain-MLP 在低比特率优势更大 | base-residual 范式惠及 MLP |
| JPEG 1/8→full 分辨率 | 接近 Gamma-MLP 16n 但仍不及 | 传统方法需要大幅增加比特率 |

### 关键发现

- Gamma-MLP 以 **10KB** 大小获得 PSNR 48.53dB，超越 34KB 的 MLP-ITM 方法（47.25dB），且大小仅为其 29%
- 指数残差编码（gamma map）在传统压缩和 MLP 编码中都一致优于乘法残差（gain map），验证了其作为预测编码近似的有效性
- MLP 方法的核心优势是**固定内存占用**（10KB），无需根据图像大小或编码参数调整
- 定性分析显示传统方法的 banding/haloing/blocking 伪影在 MLP 方法中大幅减少，gamma map 进一步改善

## 亮点与洞察

- **问题选取精准**：不做通用图像压缩（那需要大 MLP 和长时间优化），而是利用 SDR 图像作为强先验只编码变换残差，使问题变得极其轻量
- **10KB 固定开销**：与传统方法（大小随图像分辨率和质量参数变化）不同，MLP 大小完全固定，非常适合嵌入式元数据场景
- **指数残差的理论洞察**：将 gain map 问题重新框架为幂函数参数编码，找到了问题的更自然表示
- **色度噪声初始化的巧妙性**：用符合自然图像统计的合成数据做元初始化，避免内容偏差

## 局限与展望

- MLP 优化仍需 4 秒/图像，对实时应用仍有延迟
- 对于存在大量裁剪信息的 SDR tone mapping，MLP 重建能力有限——因为这些信息已经不可逆丢失
- 未处理不同分辨率图像的自适应比特率分配——不同分辨率可能需要不同大小的 MLP
- 仅在 HD (1920×1080) 和 UHD (3840×2160) 图像上验证，更高分辨率的扩展性未知
- 解码端需要 MLP 前向传播（虽然很快），与纯解码相比仍有额外计算

## 相关工作与启发

- **Le et al. (Direct-MLP)**：首次用 MLP + (x,y,r,g,b) 输入做嵌入式色域恢复，但直接输出 HDR RGB 值而非残差
- **Liu et al. (MLP-ITM)**：双网络（空间+颜色）+域预训练+困难样本挖掘，34KB 更大但精度略低于 Gamma-MLP
- **Canham et al.**：首次提出指数残差改善传统压缩的 gain map，本文证明该发现同样适用于 MLP
- **启发**：在 base+residual 编码范式中，选择正确的残差表示（线性 vs 非线性）可能比选择更好的编码器更重要

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../../ICLR2026/model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)
- [Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps](../../ACL2025/model_compression/revisiting_lora_through_the_lens_of_parameter_redundancy_spectral_encoding_helps.md)

<!-- RELATED:END -->
