---
title: >-
  [论文解读] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis
description: >-
  [NeurIPS 2025][3D视觉][Flow Matching] 提出 Shallow Flow Matching（SFM），在粗到细 TTS 框架中利用弱生成器输出构建 flow matching 中间状态，使推理从中间状态而非纯噪声出发，同时提升合成质量和加速推理。
tags:
  - NeurIPS 2025
  - 3D视觉
  - Flow Matching
  - TTS
  - 粗到细生成
  - 浅层推理
  - ODE求解
---

# Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2505.12226](https://arxiv.org/abs/2505.12226)  
**代码**: [有](https://ydqmkkx.github.io/SFMDemo/)  
**领域**: 语音合成 / 生成模型  
**关键词**: Flow Matching, TTS, 粗到细生成, 浅层推理, ODE求解

## 一句话总结

提出 Shallow Flow Matching（SFM），在粗到细 TTS 框架中利用弱生成器输出构建 flow matching 中间状态，使推理从中间状态而非纯噪声出发，同时提升合成质量和加速推理。

## 研究背景与动机

当前基于 Flow Matching（FM）的 TTS 模型通常采用粗到细的生成范式：弱生成器先产生粗糙 mel-spectrogram，再由 FM 模块精炼为高质量输出。然而，传统做法仅将粗糙表示作为 FM 的条件输入，生成仍然从纯噪声 $\boldsymbol{X}_0 \sim \mathcal{N}(0, I)$ 出发。由于粗糙表示已经编码了大量语义和声学结构，从纯噪声建模早期阶段是冗余的，浪费了建模能力。

作者受到 DiffSinger 中"浅层扩散"思想的启发：在扩散模型中从浅层步骤开始反向生成。本文将此思想推广到 Flow Matching 框架，提出了 SFM 机制——利用正交投影将弱生成器输出映射到 CondOT 路径上的中间状态，推理时直接从该中间状态出发，跳过早期阶段。

## 方法详解

### 整体框架

SFM 框架包含三个核心部分：(1) 弱生成器 $\boldsymbol{g}_\omega$ 产生粗糙 mel-spectrogram $\hat{\boldsymbol{X}}_g$；(2) 轻量级 SFM head $\boldsymbol{h}_\psi$ 输出缩放后的 mel-spectrogram $\hat{\boldsymbol{X}}_h$、时间 $\hat{t}_h$ 和方差 $\hat{\sigma}_h^2$；(3) FM 解码器从构建的中间状态开始生成。推理时，ODE 求解器从 $\tilde{t}_h$ 积分到 1，而非从 0 到 1。

### 关键设计

1. **正交投影映射（Orthogonal Projection onto CondOT Paths）**: 核心思路是将 SFM head 输出 $\hat{\boldsymbol{X}}_h$ 投影到目标 $\boldsymbol{X}_1$ 上，求投影系数 $t_h$ 作为对应时间。公式为 $t_h = \max(0, \mathbb{E}[\text{sg}[\hat{\boldsymbol{X}}_h] \cdot \boldsymbol{X}_1 / (\boldsymbol{X}_1 \cdot \boldsymbol{X}_1)])$。利用 Theorem 1 将 $\hat{\boldsymbol{X}}_h$ 通过缩放因子 $1/\Delta$ 归到 CondOT 路径上。设计动机是自适应确定 $\hat{\boldsymbol{X}}_h$ 在 FM 路径上的位置，避免手动设定浅层步。

2. **单段分段流（Single-Segment Piecewise Flow）**: 基于 Theorem 2 将 CondOT 路径在中间状态 $\tilde{t}_h$ 处分为两段，训练和推理只关注后半段 $t \geq \tilde{t}_h$。流和向量场为分段定义：$\boldsymbol{X}_t = (1-t_S)\boldsymbol{X}_{\tilde{t}_h} + t_S(\boldsymbol{X}_1 + \sigma_{\min}\boldsymbol{X}_0)$，训练使用 CFM loss 对后半段进行监督。

3. **SFM 推理强度（SFM Strength $\alpha$）**: 训练时自适应确定的 $t_h$ 通常偏小，推理时引入超参数 $\alpha \geq 1$ 放大 $\hat{t}_h$，增强粗糙表示的引导强度。通过在验证集上搜索最优 $\alpha$ 来平衡质量和确定性。

### 损失函数 / 训练策略

总损失为五项之和：

$$\mathcal{L}_{\text{SFM}} = \mathcal{L}_{\text{coarse}} + \mathcal{L}_t + \mathcal{L}_\sigma + \mathcal{L}_\mu + \mathcal{L}_{\text{CFM}}$$

- $\mathcal{L}_{\text{coarse}}$: 粗糙 mel-spectrogram 的 L2 损失
- $\mathcal{L}_\mu$: 引导 $\hat{\boldsymbol{X}}_h$ 靠近 $t_h \boldsymbol{X}_1$
- $\mathcal{L}_t, \mathcal{L}_\sigma$: 预测时间和方差的 MSE 损失
- $\mathcal{L}_{\text{CFM}}$: 条件流匹配损失

## 实验关键数据

### 主实验

在 LJ Speech、VCTK、LibriTTS 上验证，覆盖 Matcha-TTS（U-Net）、StableTTS（DiT）、CosyVoice 等多个骨干模型。

| 系统 | UTMOS↑ | UTMOSv2↑ | Distill-MOS↑ | WER↓ | CMOS↑ |
|------|--------|----------|-------------|------|-------|
| Matcha-TTS Baseline (LJ) | 4.186 | 3.692 | 4.282 | 3.308 | -0.48 |
| Matcha-TTS Ablated (LJ) | 4.217 | 3.763 | 4.311 | 3.355 | -0.27 |
| **Matcha-TTS SFM (LJ)** | **4.257** | **3.848** | **4.386** | 3.413 | **0.00** |
| Ground Truth | 4.380 | 3.964 | 4.241 | 3.566 | +0.22 |

### 消融实验

| $\alpha$ | $\tilde{t}_g$ | $\tilde{\sigma}_g$ | PMOS↑ | UTMOS↑ | WER↓ |
|----------|---------------|---------------------|-------|--------|------|
| 1.0 | 0.099 | 0.092 | 4.036 | 4.194 | 4.641 |
| 2.0 | 0.198 | 0.183 | 4.158 | 4.305 | 3.496 |
| **2.5** | **0.248** | **0.229** | **4.176** | **4.276** | 3.556 |
| 5.0 | 0.496 | 0.458 | 4.025 | 3.977 | 3.376 |
| 10.0 | 0.520 | 0.480 | 3.987 | 3.955 | 3.315 |

### 关键发现

- SFM 在所有测试的 TTS 模型上均带来一致的自然度提升（客观和主观评估均显著）
- 使用自适应步长 ODE 求解器时，SFM 显著加速推理（NFE 大幅减少）
- 最优 $\alpha$ 通常在 2-4 之间，过大会降低质量
- SFM-c（同时使用条件+SFM）的效果不如仅用 SFM，说明中间状态已包含充分信息

## 亮点与洞察

- **核心创新在于将"浅层扩散"从 DDPM 推广到 Flow Matching**，通过正交投影和分段流提供了严格的数学框架
- 轻量级 SFM head 设计使方法可即插即用地集成到多种 TTS 架构中
- 自适应确定中间状态位置（而非手动设定），比浅层扩散更灵活
- 推理时 SFM strength $\alpha$ 提供了质量-速度的灵活权衡

## 局限与展望

- SFM head 的投影假设 $\hat{\boldsymbol{X}}_h \approx t_h \boldsymbol{X}_1$ 可能在某些情况下不准确
- 目前仅在 TTS 上验证，未探索其他 FM 应用（图像、视频生成）
- $\alpha$ 需要在验证集上搜索，增加了部署成本
- 训练阶段早期 $\Delta \geq 1$ 时退化为确定性行为，可能影响训练稳定性

## 相关工作与启发

- DiffSinger 提出的浅层扩散机制是直接灵感来源
- PeRFlow 的分段 reflow 思想为 Theorem 2（分段流）提供了理论基础
- 该方法的思路可推广到其他粗到细生成任务：先用简单模型获取粗糙估计，再用生成模型精炼

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将浅层扩散推广到 FM 并提供严格数学框架
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型、多数据集、主观+客观评估、速度分析
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，实验组织良好
- 价值: ⭐⭐⭐⭐ — 即插即用的 FM 加速方法，实用性强

<!-- RELATED:START -->

## 相关论文

- [GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](../../CVPR2026/3d_vision/geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/3d_vision/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [Flux4D: Flow-based Unsupervised 4D Reconstruction](flux4d_flow-based_unsupervised_4d_reconstruction.md)

<!-- RELATED:END -->
