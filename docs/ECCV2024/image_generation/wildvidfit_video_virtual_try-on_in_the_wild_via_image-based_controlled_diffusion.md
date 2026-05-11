---
title: >-
  [论文解读] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models
description: >-
  [ECCV 2024][图像生成] WildVidFit 提出了一个无需视频训练的虚拟试穿框架，利用基于图像的条件扩散模型和扩散引导模块（VideoMAE + DINO-V2），实现了在野外复杂视频中保持时序一致性的服装试穿效果。
tags:
  - ECCV 2024
  - 图像生成
---

# WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2407.10625](https://arxiv.org/abs/2407.10625)  
**领域**: 图像生成

## 一句话总结

WildVidFit 提出了一个无需视频训练的虚拟试穿框架，利用基于图像的条件扩散模型和扩散引导模块（VideoMAE + DINO-V2），实现了在野外复杂视频中保持时序一致性的服装试穿效果。

## 研究背景与动机

视频虚拟试穿旨在生成逼真的序列，使目标服装自然地适配源视频中人物的姿态和体型。现有方法面临两大核心挑战：

**显式变形的局限性**：传统方法依赖光流估计进行服装的 warping 和 blending，在遇到复杂人体动作和肢体遮挡时容易产生像素错位，效果大打折扣

**视频数据和计算开销**：视频级模型需要大量高质量标注视频和庞大的计算资源来训练额外的时序模块，泛化能力也受限于特定数据集

**野外场景的复杂性**：TikTok 等平台的舞蹈视频包含剧烈的肢体运动、频繁的遮挡和复杂的背景，现有方法难以胜任

WildVidFit 的核心思路是将视频试穿分解为两个子任务：（1）构建一个能处理复杂动作和遮挡的细粒度图像试穿模型；（2）通过扩散引导将其扩展到视频领域并保持帧间一致性，全程无需视频级训练。

## 方法详解

### 整体框架

WildVidFit 由两个核心模块组成：

1. **单阶段图像试穿网络**：基于 Stable Diffusion 构建的条件图像生成网络，同时接受人物表示（cloth-agnostic RGB + pose map）和服装表示（服装图像 + 边缘图）作为条件输入
2. **扩散引导模块**：利用预训练的 VideoMAE 和 DINO-V2 模型，在扩散采样过程中引入时序一致性约束，无需额外微调

### 关键设计

**输入预处理**：
- 人物表示：通过分割和姿态估计生成 cloth-agnostic RGB 图像 A 和 pose map P，保留人物身份但移除原始服装
- 服装表示：对服装图像 G 使用 Sobel 算子提取边缘图 $E_g$，并通过 DINO-V2 提取特征向量 $F_g \in \mathbb{R}^{257 \times 2048}$

**单阶段隐式变形**：
- 主 UNet 继承 Stable Diffusion 权重，新增条件分支提取人物多尺度特征
- 用服装特征 $F_g$ 替代文本做 cross attention，实现隐式变形（inspired by TryOnDiffusion），避免显式光流估计的局限性
- 条件特征仅在 UNet 解码器中通过卷积注入，保留预训练先验

**扩散引导的时序一致性**：
- **VideoMAE 引导**（$\mathcal{L}_{MAE}$）：将生成帧序列随机遮盖后输入预训练 VideoMAE 重建，假设越平滑的视频越容易被重建，重建损失越低
- **DINO-V2 引导**（$\mathcal{L}_{SIM}$）：通过球面距离约束相邻帧在 DINO-V2 特征空间中的一致性
- 两种损失共同构成时序损失，通过梯度引导采样过程

**长视频生成策略**：
- 将长视频切分为重叠的短片段，相邻片段偏移 stride $s$（通常为 $L/2$ 或 $L/4$）
- 在每个去噪步中对重叠帧取平均，实现片段间的平滑过渡

### 损失函数

训练阶段使用标准扩散目标：

$$\mathcal{L} = \mathbb{E}_{x,c,\epsilon,t}[w_t \|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2]$$

推理阶段的时序引导：

$$\hat{\epsilon}_t = \epsilon_\theta(z_t; t, c) - w_1 \nabla_{z_t} \mathcal{L}_{MAE}(z_t) - w_2 \nabla_{z_t} \mathcal{L}_{SIM}(z_t)$$

其中 $w_1 = 2000$，$w_2 = 1000$，mask ratio = 0.7。

## 实验关键数据

### 主实验

**表1：VITON-HD 数据集上的图像试穿比较**

| 方法 | SSIM↑ | LPIPS↓ | FID↓ | KID↓ | User↑ |
|------|-------|--------|------|------|-------|
| CP-VTON | 0.785 | 0.2871 | 48.86 | 4.42 | 3.86% |
| HR-VTON | 0.878 | 0.0987 | 11.80 | 0.37 | 6.62% |
| LaDI-VTON | 0.871 | 0.0941 | 13.01 | 0.66 | 16.02% |
| DCI-VTON | 0.882 | 0.0786 | 11.91 | 0.51 | 12.18% |
| **WildVidFit** | **0.883** | **0.0773** | **8.67** | **0.10** | **61.32%** |

**表2：VVT 和 TikTok 数据集上的视频试穿比较**

| 方法 | 数据集 | VFID↓ | User↑ |
|------|--------|-------|-------|
| HR-VTON | VVT | 4.852 | 9.46% |
| LaDI-VTON | VVT | 4.442 | 4.24% |
| ClothFormer | VVT | 4.192 | 46.44% |
| **WildVidFit** | **VVT** | **4.202** | **39.86%** |
| HR-VTON | TikTok | 25.43 | 0.00% |
| LaDI-VTON | TikTok | 14.24 | 26.90% |
| **WildVidFit** | **TikTok** | **9.87** | **73.10%** |

### 消融实验

**表3：边缘图和 CFG 消融（VITON-HD）**

| Edge maps | Guidance scale | FID↓ | KID↓ |
|-----------|---------------|------|------|
| ✗ | 2 | 8.93 | 0.12 |
| ✓ | 1 | 9.47 | 0.17 |
| ✓ | 2 | **8.67** | **0.10** |
| ✓ | 3 | 8.68 | 0.10 |

**表4：时序模块消融（TikTok）**

| 方法 | VFID↓ |
|------|-------|
| Image-based | 13.45 |
| + Fully cross-frame attention | 12.14 |
| + Guidance with $\mathcal{L}_{MAE}$ | 10.64 |
| + Guidance with $\mathcal{L}_{MAE}$ and $\mathcal{L}_{SIM}$ | **9.87** |

### 关键发现

1. WildVidFit 在 VITON-HD 上 FID 仅 8.67，大幅领先 DCI-VTON（11.91），用户偏好达 61.32%
2. 在 TikTok 野外视频上，WildVidFit 以 9.87 的 VFID 远超 LaDI-VTON（14.24），用户偏好 73.10%
3. 在 VVT 上，纯图像方法+扩散引导即可匹敌专门的视频方法 ClothFormer
4. 每个时序模块的贡献都是显著的：cross-frame attention 降低 VFID 1.31，MAE 引导再降 1.50，SIM 引导再降 0.77

## 亮点与洞察

- **无需视频训练的视频生成范式**：通过扩散引导机制巧妙地将预训练视频/图像模型的时序先验注入采样过程，完全避免了视频级训练的高昂成本
- **隐式变形优于显式变形**：用 cross attention 替代光流估计进行服装适配，不依赖像素级严格对齐，天然具备更好的遮挡处理能力
- **多数据集联合训练增强泛化**：将 VITON-HD、DressCode 和 TikTok 三个数据集联合训练，可以跨数据集进行服装迁移

## 局限性

1. 扩散引导需要在每个采样步解码潜在表示计算损失，推理速度较慢
2. 由于内存限制，时序损失仅在服装区域计算，可能对全局一致性有一定影响
3. VideoMAE 的固定片段长度限制了超长视频的处理灵活性
4. 对于极端遮挡或超大幅度动作，性能可能仍有下降

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
