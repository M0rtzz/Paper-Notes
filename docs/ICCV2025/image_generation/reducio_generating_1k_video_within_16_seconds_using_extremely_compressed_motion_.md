---
title: >-
  [论文解读] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents
description: >-
  [ICCV 2025][图像生成][视频生成] 提出 Reducio-VAE，一种以内容帧为条件的 3D 视频自编码器，将视频压缩至比标准 2D VAE 小 64 倍的运动潜空间，配合 Reducio-DiT 在单张 A100 上 15.5 秒内生成 16 帧 1024x1024 视频，训练仅需 3200 A100 GPU 小时。
tags:
  - ICCV 2025
  - 图像生成
  - 视频生成
  - 运动潜变量
  - 视频VAE
  - 极端压缩
  - DiT
  - 高效推理
---

# REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents

**会议**: ICCV 2025  
**arXiv**: [2411.13552](https://arxiv.org/abs/2411.13552)  
**代码**: [GitHub](https://github.com/microsoft/Reducio-VAE)  
**领域**: 图像生成 / 视频生成  
**关键词**: 视频生成, 运动潜变量, 视频VAE, 极端压缩, DiT, 高效推理

## 一句话总结

提出 Reducio-VAE，一种以内容帧为条件的 3D 视频自编码器，将视频压缩至比标准 2D VAE 小 64 倍的运动潜空间，配合 Reducio-DiT 在单张 A100 上 15.5 秒内生成 16 帧 1024x1024 视频，训练仅需 3200 A100 GPU 小时。

## 研究背景与动机

- **核心问题**：当前视频 LDM（如 Sora、Runway Gen-3）训练需数千 GPU 和数百万 GPU 小时，推理生成一秒视频需数分钟——高昂成本严重阻碍研究和大规模应用
- **被忽视的本质**：大多数视频 LDM 沿用图像扩散的范式，直接使用 SD 的 2D VAE（空间 8x 下采样，时间 1x），但视频远比图像冗余——相邻帧大面积重复
- **现有加速方向**：
    - 高效注意力模块（如 Mamba 替代 Transformer）
    - 扩散训练策略优化（更少采样步数、蒸馏）
    - 但都没有从根源解决问题——**潜空间本身就过大**
- **关键洞察**：视频可以被分解为"内容帧 + 极少量运动潜变量"，空间维度可以激进压缩至 32x，时间维度压缩至 4x，总体实现 4096x 下采样，同时重建质量反而优于标准 VAE

## 方法详解

### 整体框架：两阶段生成

1. **第一阶段**：用现成的 T2I LDM（如 PixArt-alpha）生成内容图像
2. **第二阶段**：以文本+内容图像为条件，通过 Reducio-DiT 在极度压缩的运动潜空间中生成视频

### 关键设计 1：Reducio-VAE

**动机**：既然以内容帧为条件重建，视频潜变量只需编码帧间运动信息。

**架构演进**：
- **(a) 标准 2D VAE**（SDXL）：$f_s=8, f_t=1$ -> 每帧独立编码
- **(b) 3D VAE**：$f_s=16, f_t=4$ -> 时空联合压缩，16x 提升
- **(c) Reducio-VAE**：$f_s=32, f_t=4$ -> 以中间帧为内容条件，总下采样 4096x

**具体实现**：
- 3D 编码器将输入视频 $3 \times T \times H \times W$ 压缩至 $|z| \times T/4 \times H/32 \times W/32$
- 选取中间帧 $V_{T/2}$ 作为内容引导
- 3D 解码器融合内容帧的多尺度特征金字塔（$H/8 \times W/8$ 和 $H/4 \times W/4$），通过交叉注意力注入细节
- 对高分辨率视频采用空间分块编解码（tile size 256，overlap 64），重叠区域线性混合

**核心优势**：Reducio-VAE 在 64x 更小的潜空间下，PSNR 反超 SDXL-VAE 达 5dB！

### 关键设计 2：Reducio-DiT

基于 DiT-XL 架构，沿用 PixArt-alpha 的 AdaLN-single + T5 文本交叉注意力设计。

**图像条件模块**：
- **语义编码器**：预训练 OpenCLIP ViT-H，提取图像高层语义
- **内容编码器**：SD2.1-VAE 初始化，提取空间详细信息
- 两类特征与 T5 文本 token 拼接，通过交叉注意力注入去噪过程

**注意力方案**：默认采用全 3D 注意力（2D->3D 直接转换，无额外参数），消融显示优于 2D+1D 分离方案。

**多分辨率训练策略**：
- Stage 1：256^2 视频，batch 512，4xA100，~900 GPU 小时
- Stage 2：512^2 微调，~300 GPU 小时
- Stage 3：1024 + 多宽高比增强，8xMI300，~1000 GPU 小时
- 总计仅 **3200 A100 GPU 小时**

**高分辨率效率策略**：在 Stage 3，内容编码器的 token 从 $H/16 \times W/16$ 压缩至 $H/32 \times W/32$，受 Deepstack 启发将 token 分为 4 组网格迭代拼接。

### 损失函数

- VAE 训练：标准重建损失 + KL 正则
- DiT 训练：标准扩散去噪损失 $\mathcal{L} = \mathbb{E}_{z,\epsilon,t} \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2$

## 实验关键数据

### VAE 重建性能对比

| 模型 | 下采样 | 潜通道 | PSNR | SSIM | LPIPS | rFVD |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| SD2.1-VAE | 1x8x8 | 4 | 29.23 | 0.82 | 0.09 | 25.96 |
| SDXL-VAE | 1x8x8 | 4 | 30.54 | 0.85 | 0.08 | 19.87 |
| OpenSora-1.2 | 4x8x8 | 16 | 30.72 | 0.85 | 0.11 | 60.88 |
| Cosmos-VAE (8x8) | 8x8x8 | 16 | 30.84 | 0.74 | 0.12 | 29.44 |
| Cosmos-VAE (8x16) | 8x16x16 | 16 | 28.14 | 0.65 | 0.18 | 77.87 |
| **Reducio-VAE** | **4x32x32** | **16** | **35.88** | **0.94** | **0.05** | **17.88** |

Reducio-VAE 以 64x 压缩率取得最高 PSNR（35.88）、最低 LPIPS（0.05），全面碾压所有基线。

### 视频生成 FVD 对比

| 方法 | UCF-101 FVD |
|------|:---:|
| VideoComposer | 576.8 |
| I2VGen-XL | - |
| **Reducio-DiT** | **318.5** |

### 效率对比

| 方法 | 生成 1024^2 16帧耗时 | 加速比 |
|------|:---:|:---:|
| Lavie | ~258s | 1x |
| **Reducio-DiT** | **15.5s** | **16.6x** |

### 消融实验

| 消融项 | 结果 |
|--------|------|
| 3D全注意力 vs 2D+1D | 3D 全注意力效果更好 |
| 内容帧选择 | 中间帧优于首帧/末帧 |
| $f_s$=32 vs 16 | 32 最优（配合内容帧条件） |
| 有/无内容编码器 | 内容编码器显著提升细节保留 |

## 亮点与洞察

1. **视频=内容帧+运动潜变量**：这一分解思想是本文最核心的洞察。内容帧已包含绝大部分视觉信息，运动变化可以用极少的潜变量描述
2. **压缩悖论**：64x 压缩反而提升了重建质量（PSNR +5dB），因为内容帧条件提供了充分的空间先验
3. **极致效率**：总训练仅 3200 A100 小时，是 Sora 等商用模型的千分之一量级；推理 15.5 秒/视频是 Lavie 的 16.6 倍加速
4. **工程友好**：分块编解码支持任意分辨率，多宽高比训练支持灵活部署
5. **来自微软研究院**，代码开源，数据集可公开获取

## 局限性

- 两阶段管线（先生图再生视频）导致视频内容受限于首帧质量，错误会级联
- 中间帧作为内容帧在推理时不可用——实际使用时以 T2I 生成的首帧替代，可能与最终视频不完全一致
- 16 帧限制较短，对长视频生成（数十秒以上）需要额外的时间扩展机制
- Reducio-VAE 在 Pexels rFVD 上表现优异，但 UCF-101 rFVD（65.17）高于 SDXL-VAE（23.68），说明对某些视频分布的泛化有待提升
- 运动潜变量的极端压缩可能丢失复杂运动细节（快速运动、遮挡变化等）

## 相关工作

- **视频扩散模型**：AnimateDiff、SVD、Open-Sora、CogVideo 等 UNet/DiT 视频生成
- **两阶段生成**：I2VGen-XL、DynamiCrafter、SparseCtrl 的图像到视频范式
- **高效扩散**：CausVid（蒸馏）、DIM（Mamba 骨干）、少步采样等
- **视频潜空间**：PVDM/CMD（三个 2D 平面分解）、LaMD（运动潜空间分解，本文最直接前身）
- **高压缩图像 VAE**：DC-AE、TiTok 等高下采样因子 VAE

## 评分

| 维度 | 分数 (1-5) |
|------|:---:|
| 创新性 | 4.5 |
| 理论深度 | 3.5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 5 |
| **总评** | **4.2** |

本文的核心洞察——视频可被分解为内容帧+极少运动潜变量——简洁而有力，且实验验证了"更压缩=更高质量"的反直觉结论。64x 压缩率是一个标志性数字，对视频生成效率具有里程碑意义。实用价值极高：3200 GPU 小时训练 + 15.5 秒推理使高质量视频生成触手可及。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device](../../CVPR2025/image_generation/snapgen-v_generating_a_five-second_video_within_five_seconds_on_a_mobile_device.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)

<!-- RELATED:END -->
