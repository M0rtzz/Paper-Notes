---
title: >-
  [论文解读] Detail-Preserving Latent Diffusion for Stable Shadow Removal
description: >-
  [CVPR 2025][图像恢复][阴影去除] 提出两阶段阴影去除框架，Stage 1 微调 Stable Diffusion 去噪器以 z₀-prediction 方式生成粗阴影去除结果（低方差高稳定性），Stage 2 用 Detail Injection 模块通过 RRDB 块选择性地将阴影感知高频细节注入冻结 VAE 解码器，在四个基准上达到无掩码方法最优。
tags:
  - CVPR 2025
  - 图像恢复
  - 阴影去除
  - 潜扩散模型
  - 细节注入
  - VAE解码器增强
  - 无掩码推理
---

# Detail-Preserving Latent Diffusion for Stable Shadow Removal

**会议**: CVPR 2025  
**arXiv**: [2412.17630](https://arxiv.org/abs/2412.17630)  
**代码**: 无（未提及）  
**领域**: 图像修复 / 扩散模型  
**关键词**: 阴影去除, 潜扩散模型, 细节注入, VAE解码器增强, 无掩码推理

## 一句话总结
提出两阶段阴影去除框架，Stage 1 微调 Stable Diffusion 去噪器以 z₀-prediction 方式生成粗阴影去除结果（低方差高稳定性），Stage 2 用 Detail Injection 模块通过 RRDB 块选择性地将阴影感知高频细节注入冻结 VAE 解码器，在四个基准上达到无掩码方法最优。

## 研究背景与动机

**领域现状**：阴影去除旨在从带阴影图像恢复无阴影版本。传统方法依赖阴影掩码指导修复，但掩码获取本身是难题。基于扩散模型的方法（如 DeS3）可以处理但推理方差大、细节丢失。

**现有痛点**：(1) 扩散模型的随机采样导致推理方差高（DeS3 方差 1.075 vs 本文 0.160），多次推理结果差异大；(2) VAE 编解码的信息瓶颈损失了高频细节（纹理、边缘信息在 latent 空间中丢失）；(3) 掩码依赖方法在掩码不准确时性能急剧下降。

**核心矛盾**：需要扩散模型的生成能力来修复阴影区域，但也需要保留非阴影区域的原始高频细节，而 VAE 的编解码过程天然丢失这些细节。

**本文目标** 在利用扩散模型的阴影修复能力的同时，保持高频细节并降低推理方差，无需阴影掩码。

**切入角度**：两阶段分离——Stage 1 训练扩散去噪器做粗修复（用 z₀-prediction 减小方差），Stage 2 训练 Detail Injection 模块补偿 VAE 丢失的高频细节。

**核心 idea**：z₀-prediction 的微调 LDM 做低方差粗阴影修复 + RRDB-based 细节注入模块补偿 VAE 解码器损失的高频信息。

## 方法详解

### 关键设计

1. **z₀-prediction 粗修复**：微调 SD 去噪器直接预测 z₀（而非噪声 ε），降低采样随机性。条件输入为阴影图像的 latent code 拼接。推理方差从 DeS3 的 1.075 降到 0.160

2. **Detail Injection (DI) 模块**：RRDB 块从 VAE 编码器特征中提取高频信息，选择性注入冻结 VAE 解码器。输入拼接阴影图/无阴影图特征 + DINOv2 语义特征，使注入过程具有阴影感知性——只注入无阴影的细节

3. **无掩码推理**：不需要阴影掩码，模型自学习哪些区域需要修复

## 实验关键数据

| 方法 | ISTD+ PSNR↑ | SRD PSNR↑ | INS PSNR↑ | WSRD+ PSNR↑ | 推理方差↓ |
|------|-----------|---------|---------|-----------|---------|
| ShadowFormer (有掩码) | 35.46 | 32.90 | 28.62 | 25.44 | — |
| OmniSR (无掩码) | 33.34 | 32.87 | 30.38 | 26.07 | — |
| DeS3 (扩散) | — | — | — | — | 1.075 |
| **Ours (无掩码)** | **35.19** | **33.63** | **30.56** | **26.26** | **0.160** |

四个数据集无掩码方法最优。推理方差比 DeS3 低 6.7×。

## 评分
- 新颖性: ⭐⭐⭐⭐ z₀-prediction 降方差+DI 补细节的组合有效
- 实验充分度: ⭐⭐⭐⭐ 四个数据集+方差分析
- 写作质量: ⭐⭐⭐⭐ 两阶段设计清晰
- 价值: ⭐⭐⭐⭐ 无掩码阴影去除的实用方案

<!-- RELATED:START -->

## 相关论文

- [FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)
- [Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](../../NeurIPS2025/image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)
- [Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)

<!-- RELATED:END -->
