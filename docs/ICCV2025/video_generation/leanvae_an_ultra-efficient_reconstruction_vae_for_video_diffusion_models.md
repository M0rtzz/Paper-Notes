---
title: >-
  [论文解读] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models
description: >-
  [视频生成] 提出 LeanVAE，基于非重叠 Patch 操作、邻域感知前馈（NAF）模块、小波变换和压缩感知技术，构建超高效视频 VAE，在仅 40M 参数下实现 FLOPs 减少 50 倍、推理速度加快 44 倍，同时保持有竞争力的重建质量。 Video VAE 是视频扩散模型的瓶颈 潜在视频扩散模型 (LVDM) 如…
tags:
  - "视频生成"
---

# LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models

| 信息 | 内容 |
|------|------|
| 会议 | ICCV 2025 |
| arXiv | [2503.14325](https://arxiv.org/abs/2503.14325) |
| 代码 | [GitHub](https://github.com/westlake-repl/LeanVAE) |
| 领域 | 视频生成 · VAE · 扩散模型 |
| 关键词 | Video VAE, wavelet transform, compressed sensing, lightweight architecture, video diffusion |

## 一句话总结

> 提出 LeanVAE，基于非重叠 Patch 操作、邻域感知前馈（NAF）模块、小波变换和压缩感知技术，构建超高效视频 VAE，在仅 40M 参数下实现 FLOPs 减少 50 倍、推理速度加快 44 倍，同时保持有竞争力的重建质量。

## 研究背景与动机

### Video VAE 是视频扩散模型的瓶颈

潜在视频扩散模型 (LVDM) 如 Open-Sora、CogVideoX、HunyuanVideo 等依赖 Video VAE 将高维视频压缩到紧凑的潜在空间。然而：

**计算瓶颈**：现有 Video VAE（如 OD-VAE）处理 5 帧 1080p 视频就需要 ~32GB 显存，在 LVDM 训练中成为主要计算瓶颈

**继承问题**：多数方法直接从 Stable Diffusion 的图像 VAE 膨胀而来（2D→3D 卷积），架构冗余严重

**效率-质量权衡**：轻量化尝试（如 ViT-based）虽参数更少但有二次复杂度问题

### 设计理念

LeanVAE 的核心思路：不是在重型架构上做减法，而是从零开始设计一个极轻量级的 Video VAE，同时利用信号处理领域的经典工具（小波变换、压缩感知）来弥补模型容量的不足。

## 方法详解

### 整体框架

输入视频 $\mathbf{x} \in \mathbb{R}^{(T+1) \times H \times W \times 3}$ 经过以下流程压缩为潜在表示 $\mathbf{z} \in \mathbb{R}^{(T'+1) \times H' \times W' \times d}$，压缩比 $c_t = 4, c_s = 8$。

### 1. Patchify：频域非重叠分块

与传统 ViT 直接在 RGB 空间分块不同，LeanVAE 先做 **Haar 小波变换**，再做非重叠分块：

- 后续帧 $\mathbf{x_{1:T}}$ 执行 3D Haar DWT → 低频分量 LC ($T/2 \times H/2 \times W/2 \times 3$) + 高频分量 HC ($T/2 \times H/2 \times W/2 \times 21$)
- 首帧 $\mathbf{x_0}$ 执行 2D Haar DWT（支持图像-视频联合编码）
- 分别用线性层投影为低频嵌入 $\mathbf{p^L}$ (384 维) 和高频嵌入 $\mathbf{p^H}$ (128 维)

三个区别于标准 ViT：① 首帧与后续帧分开处理（支持图像+视频）；② 频域操作而非 RGB 空间；③ **不做 Patch 归一化**（发现 LayerNorm 导致块状伪影，PSNR 下降 3.27dB）。

### 2. 编码器/解码器：NAF 骨干

核心模块为 **Neighborhood-Aware Feedforward with Residual Connection (ResNAF)**：

$$\text{ResNAF}(x) = x + \text{FFN}(\text{DWConv}_{3\times3\times3}(x))$$

- 用 3D 深度可分离卷积聚合邻域上下文
- 前馈层做特征变换
- 残差连接优化梯度传播

编码器架构采用分离-融合结构：

$$\mathbf{p} = \xi_f(\text{cat}(\xi_l(\mathbf{p^L}), \xi_h(\mathbf{p^H})))$$

其中 $\xi_l, \xi_h$ 各 2 层 ResNAF，$\xi_f$ 为 4 层 ResNAF。时间维度使用 **因果填充**，确保每一帧只与之前帧交互。

### 3. 通道压缩瓶颈：ISTA-Net+ 压缩感知

**首次将压缩感知引入 Video VAE 的通道压缩**。使用感知矩阵 $\Phi \in \mathbb{R}^{d \times D}$ 将特征从 $D=512$ 维压缩到 $d \in \{4, 16\}$ 维。恢复过程使用 ISTA-Net+ 的迭代展开算法：

$$\mathbf{r}^{(k)} = \mathbf{p}^{(k-1)} - \rho^{(k)} \Phi^\top(\Phi \mathbf{p}^{(k-1)} - \mathbf{z})$$

$$\mathbf{p}^{(k)} = \mathbf{r}^{(k)} + \tilde{\mathcal{F}}^{(k)}(\text{soft}(\mathcal{F}^{(k)}(\mathbf{r}^{(k)}), \theta^{(k)}))$$

前向/后向网络 $\mathcal{F}^{(k)}, \tilde{\mathcal{F}}^{(k)}$ 各用 2 层 NAF，迭代步数 $K=2$。

### 4. 训练目标

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{lpips}} \mathcal{L}_{\text{lpips}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{KL} \mathcal{L}_{KL}$$

RGB + 频域 L1 重建损失 + VGG 感知损失 + PatchGAN 对抗损失 + KL 正则。分两阶段训练：600K 步无 GAN + 100K 步含 GAN。

## 实验

### 主实验：视频重建性能

| 方法 | 参数量 | 通道 | DAVIS PSNR↑ | DAVIS LPIPS↓ | DAVIS rFVD↓ | TokenBench PSNR↑ | TokenBench LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CV-VAE | 182M | 4 | 25.75 | 0.1464 | 598.55 | 30.37 | 0.0706 |
| OD-VAE | 239M | 4 | 26.16 | 0.1173 | 407.20 | 30.47 | 0.0618 |
| VidTok | 157M | 4 | 26.50 | 0.1098 | 358.28 | 31.38 | 0.0526 |
| **LeanVAE** | **40M** | 4 | 26.04 | **0.0899** | **322.46** | 31.12 | **0.0432** |
| WF-VAE | 316M | 16 | 29.62 | 0.0628 | 149.27 | 35.11 | 0.0222 |
| VidTok | 157M | 16 | **31.06** | **0.0436** | **103.79** | **36.12** | **0.0166** |
| **LeanVAE** | **40M** | 16 | 30.15 | 0.0461 | 119.48 | 35.71 | 0.0173 |

关键发现：LeanVAE 仅 40M 参数在 4 通道设置下 **LPIPS 和 rFVD 最优**，16 通道下接近 VidTok（参数量仅 1/4）。

### 效率对比（vs. VidTok）

- **FLOPs**：减少 **50×**（768² 分辨率）
- **推理速度**：加速 **8-44×**（VidTok 处理 768² 17 帧需 20.26 秒，LeanVAE 仅 0.46 秒）
- **显存**：可在单 A40 上处理 17 帧 1080p 视频（~15GB FP16）

### 视频生成实验

| 方法 | 吞吐量 | SkyTimelapse FVD↓ | UCF101 FVD↓ |
|------|:---:|:---:|:---:|
| Latte 原版 (4 chn) | 1.60 | 59.82 | 477.97 |
| **Latte+LeanVAE (4 chn)** | **6.64** | **49.59** | **164.45** |

训练吞吐量提升 **315%**（支持 4× 更大 batch），生成质量同时提升。

### 消融实验

| 消融项 | PSNR | LPIPS | rFVD |
|--------|:---:|:---:|:---:|
| Variant 2 (分离处理后融合) ✓ | **26.18** | 0.145 | 470.64 |
| AE 瓶颈 (vs. CS) | 25.79 | 0.163 | 535.18 |
| 有 Patch 归一化 | 22.91 | 0.158 | 599.38 |

压缩感知比 AE 瓶颈提升 0.39dB PSNR；Patch 归一化导致 3.27dB 下降。

## 亮点与洞察

1. **首次将压缩感知引入 Video VAE**，用 ISTA-Net+ 替代简单 AE 瓶颈，取得显著性能提升
2. **发现 Patch 归一化导致块状伪影**，这一洞察可能启发低层视觉任务中的相关模型改进
3. **极致轻量**：40M 参数即可匹敌 150M-300M 级别的对手，且因果设计支持图像-视频联合建模
4. **实际意义重大**：LVDM 训练中 VAE 的编码速度直接影响训练吞吐量，LeanVAE 可以显著加速大规模视频生成模型训练

## 局限性

- 16 通道潜在空间虽然重建质量更好，但在扩散生成中反而效果变差（FVD 增加 45），说明高通道潜在空间的扩散训练仍是开放问题
- 仅验证了 4×8×8 压缩比，更高压缩率有待探索
- 未涉及文本到视频生成的端到端评估

## 相关工作

- **常规 Video VAE**：OD-VAE、CV-VAE、CogVideoX VAE、WF-VAE、VidTok、Cosmos Tokenizer 等
- **高效化方向**：因式分解 3D 卷积（Open-Sora）、小波变换（WF-VAE）、ViT 架构（OmniTokenizer、ViTok）
- **压缩感知**：ISTA-Net+ 展开算法框架

## 评分

| 维度 | 分数 |
|------|:----:|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
