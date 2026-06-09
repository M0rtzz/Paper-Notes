---
title: >-
  [论文解读] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution
description: >-
  [NeurIPS 2025][图像生成][单步扩散] 提出 DOVE，基于 CogVideoX 预训练视频生成模型，通过两阶段潜空间-像素空间训练策略和高质量 HQ-VSR 数据集实现单步推理的视频超分辨率，比多步扩散方法快 28 倍且性能相当或更优。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "单步扩散"
  - "视频超分辨率"
  - "CogVideoX"
  - "潜空间-像素空间训练"
  - "视频数据管线"
---

# DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution

**会议**: NeurIPS 2025  
**arXiv**: [2505.16239](https://arxiv.org/abs/2505.16239)  
**代码**: [有](https://github.com/zhengchen1999/DOVE)  
**领域**: 图像生成 / 扩散模型 / 视频超分辨率  
**关键词**: 单步扩散, 视频超分辨率, CogVideoX, 潜空间-像素空间训练, 视频数据管线

## 一句话总结

提出 DOVE，基于 CogVideoX 预训练视频生成模型，通过两阶段潜空间-像素空间训练策略和高质量 HQ-VSR 数据集实现单步推理的视频超分辨率，比多步扩散方法快 28 倍且性能相当或更优。

## 研究背景与动机

现有扩散模型在真实世界视频超分辨率（VSR）中表现优秀，但面临两大瓶颈：

**多步采样效率低**：典型方法需数十步采样，处理 33 帧 720p 视频在 A100 上需 173-425 秒

**额外模块增加开销**：ControlNet、时序层等辅助组件进一步拖慢推理

单步推理在图像 SR 已有成功案例，但在 VSR 中尚未实现，困难在于：
- **视频训练开销过大**：DMD/VSD 等需多网络联合优化，在视频域不可行
- **高保真度要求**：对抗训练的不稳定性在 VSR 中引入不良伪影

## 方法详解

### 整体框架

DOVE 基于 CogVideoX1.5（T2V 预训练模型）微调，核心设计决策：
- **不引入任何额外模块**（无 ControlNet、无光流模块、无时序层），只微调 Transformer
- LR 视频双线性上采样后通过 VAE 编码到潜空间 z_lr
- 将 z_lr 视为时间步 t=399 的噪声潜变量（而非 t=999），因为 LR 已包含足够结构信息
- 单步 v-prediction 去噪：z_sr = √ᾱ_t · z_lr - √(1-ᾱ_t) · v_θ(z_lr, c, t)
- 使用空文本 prompt，预编码以减少推理开销
- VAE 解码得到输出视频 x_sr

### 关键设计

#### Latent-Pixel 两阶段训练策略

核心创新——用回归损失（而非蒸馏或对抗损失）实现高效训练：

**Stage-1: Adaptation（潜空间适应）**
- 在潜空间最小化预测潜变量 z_sr 与 HR 潜变量 z_hr 的 MSE
- 利用 VAE 高压缩比的计算效率优势，可在更长帧序列上训练
- 训练 10,000 步，学习率 2e-5，视频分辨率 320×640，帧长 25

**Stage-2: Refinement（像素空间精炼）**
- 潜空间训练后 z_sr 接近 z_hr，但经 VAE 解码后差距被放大
- 像素空间训练直接优化 x_sr 与 x_hr 的差距
- 引入**图像-视频混合训练**解决视频像素训练内存瓶颈：
    - 图像（单帧视频）占比 φ=0.8，像素训练可行
    - 视频逐帧通过 VAE 编码/解码避免多帧内存峰值，Transformer 在完整潜变量上操作
- 训练仅 500 步，学习率 5e-6

**Stage-2 损失函数：**
- 图像：L_s2-image = MSE(x̂_sr, x̂_hr) + λ₁ · DISTS(x̂_sr, x̂_hr)
- 视频：L_s2-video = MSE + λ₁·DISTS + λ₂·L_frame
- 帧差损失 L_frame 强制帧间一致性：对齐相邻帧差值 Δx_sr 与 Δx_hr
- λ₁ = λ₂ = 1

#### 视频处理管线（HQ-VSR 数据集构建）

四步流水线从 OpenVid-1M 中筛选高质量 VSR 训练数据：
1. **元数据过滤**：短边 > 720px，帧数 > 50
2. **场景过滤**：场景检测+切分，丢弃 < 50 帧的片段
3. **质量过滤**：多指标严格筛选（CLIP-IQA + FasterVQA + DOVER）
4. **运动处理**：光流运动打分 + **运动区域检测算法**

    - 生成运动强度图 M，阈值化得运动掩码
    - 包围盒 B 定位高运动区域，裁剪后丢弃低于 720p 的片段
    - 解决"全局高运动但局部静态"的问题

最终得到 2,055 个高质量视频的 HQ-VSR 数据集。

### 损失函数 / 训练策略

- Stage-1：MSE loss 在潜空间对齐
- Stage-2：MSE + DISTS（感知质量）+ 帧差损失（时序一致性）
- 仅微调 Transformer，VAE 权重冻结
- 4 张 A800-80G GPU，总 batch size 8
- AdamW 优化器，Stage-1 共 10K 步 + Stage-2 共 500 步
- 图像数据 DIV2K（900 张），采用 Real-ESRGAN 退化
- 视频数据 HQ-VSR（2,055 视频），采用 RealBasicVSR 退化

## 实验关键数据

### 主实验

**UDM10 合成数据集（×4 超分）**

| 方法 | 步数 | PSNR↑ | LPIPS↓ | CLIP-IQA↑ | DOVER↑ | E*warp↓ |
|------|------|-------|--------|-----------|--------|---------|
| RealBasicVSR | - | 24.13 | 0.3908 | 0.3494 | 0.7564 | 3.10 |
| MGLD-VSR | 多步 | 24.23 | 0.3272 | 0.4557 | 0.7264 | 3.59 |
| STAR | 多步 | 23.47 | 0.4242 | 0.2417 | 0.4830 | 2.08 |
| **DOVE** | **1步** | **26.48** | **0.2696** | **0.5107** | **0.7809** | **1.77** |

DOVE 在保真度（PSNR +2.25）和感知质量（CLIP-IQA/DOVER）上全面领先。

**SPMCS 合成数据集（×4 超分）**

| 方法 | PSNR↑ | LPIPS↓ | CLIP-IQA↑ |
|------|-------|--------|-----------|
| MGLD-VSR | 22.39 | 0.3263 | 0.4348 |
| **DOVE** | **23.11** | **0.2888** | **0.5690** |

**效率对比（33 帧 720p 视频，1 张 A100）**

| 方法 | 推理时间 |
|------|---------|
| MGLD-VSR | 425.23s |
| STAR | 173.07s |
| **DOVE** | **~15s（28× 加速）** |

### 消融实验

**训练策略消融（UDM10）**

| 策略 | PSNR | LPIPS | CLIP-IQA | DOVER |
|------|------|-------|----------|-------|
| S1（仅潜空间） | 27.20 | 0.3037 | 0.3236 | 0.6154 |
| S1+S2-I（+像素图像） | 26.39 | 0.2784 | 0.5085 | 0.7694 |
| S1+S2-I/V（+像素混合） | 26.48 | 0.2696 | 0.5107 | 0.7809 |

Stage-2 显著提升感知质量（CLIP-IQA: 0.32→0.51），混合训练优于纯图像。

**图像比例 φ 消融**

| φ | LPIPS | CLIP-IQA | DOVER |
|------|-------|----------|-------|
| 0%（纯视频） | 0.2624 | 0.4800 | 0.7647 |
| 80%（最优） | 0.2696 | 0.5107 | 0.7809 |
| 100%（纯图像） | 0.2784 | 0.5085 | 0.7694 |

**HQ-VSR 数据集对比（Stage-1）**

| 数据集 | 视频数 | PSNR | DOVER |
|--------|--------|------|-------|
| YouHQ | 38,576 | 26.88 | 0.3965 |
| OpenVid-1M | ~400K | 27.04 | 0.4363 |
| **HQ-VSR** | **2,055** | **27.20** | **0.6154** |

仅 2K 视频即超越 40 万级数据集，证明数据质量远比数量重要。

### 关键发现

- 单步推理在 VSR 中完全可行，且性能超越多步方法
- 潜空间→像素空间两阶段训练是平衡效率与质量的关键
- 视频逐帧过 VAE 的技巧有效解决了像素空间视频训练的内存瓶颈
- 运动区域检测裁剪比全局运动打分更适合 VSR 场景

## 亮点与洞察

1. **首个单步扩散 VSR 模型**：28 倍加速且不牺牲性能，具有重要实用价值
2. **极简架构哲学**：不加任何额外模块，完全依赖预训练 T2V 模型的先验，简洁高效
3. **超短训练周期**：仅 10K+500 步即完成微调，远低于同类方法
4. **数据质量 >> 数据数量**：2K 高质量视频胜过 40 万视频，运动区域裁剪是关键
5. **t=399 的选择**：不从纯噪声开始，利用 LR 已有的结构信息，减少不必要的重建负担

## 局限与展望

- 基于 CogVideoX 的模型体量较大，单帧推理仍需 GPU
- 仅在 ×4 超分上评估，可扩展到其他倍数和退化类型
- 帧差损失是简单的 L1 差值对齐，更高级的时序一致性约束（如光流）可能进一步提升
- HQ-VSR 仅 2K 视频，更大规模高质量数据可能进一步提升

## 相关工作与启发

- CogVideoX 作为基座模型：3D causal VAE + Transformer denoiser
- OSEDiff 在图像 SR 首次探索单步扩散，DOVE 将其推进到视频域
- DISTS 感知损失在质量评估和训练中同时使用
- 帧差损失是简洁但有效的时序一致性方案，避免了光流计算的额外开销

## 评分

- 新颖性：⭐⭐⭐⭐（首次将单步扩散推进至 VSR，训练策略有创新）
- 技术深度：⭐⭐⭐⭐（两阶段训练+数据管线+架构选择有完整方法论）
- 实验充分性：⭐⭐⭐⭐⭐（6 个测试集、8 个对比方法、多维度消融）
- 实用性：⭐⭐⭐⭐⭐（28× 加速具有重要应用价值）
- 表达清晰度：⭐⭐⭐⭐⭐（结构清晰，图示丰富）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](../../CVPR2025/image_generation/self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)

</div>

<!-- RELATED:END -->
