---
title: >-
  [论文解读] Hierarchical Flow Diffusion for Efficient Frame Interpolation
description: >-
  [CVPR 2025][图像生成][视频插帧] 本文提出在视频插帧中用层级扩散模型从粗到细显式去噪双向光流（而非直接去噪潜空间），再用流引导图像合成器生成最终帧，实现比其他扩散方法快 10+ 倍且精度 SOTA。 领域现状：视频插帧旨在给定连续两帧生成中间帧。主流方法基于编码器-解码器范式利用双向光流作为中间监督信号…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频插帧"
  - "扩散模型"
  - "层级光流"
  - "从粗到细"
  - "端到端训练"
---

# Hierarchical Flow Diffusion for Efficient Frame Interpolation

**会议**: CVPR 2025  
**arXiv**: [2504.00380](https://arxiv.org/abs/2504.00380)  
**代码**: [项目页](https://hfd-interpolation.github.io)  
**领域**: 图像生成 / 视频理解  
**关键词**: 视频插帧, 扩散模型, 层级光流, 从粗到细, 端到端训练

## 一句话总结

本文提出在视频插帧中用层级扩散模型从粗到细显式去噪双向光流（而非直接去噪潜空间），再用流引导图像合成器生成最终帧，实现比其他扩散方法快 10+ 倍且精度 SOTA。

## 研究背景与动机

**领域现状**：视频插帧旨在给定连续两帧生成中间帧。主流方法基于编码器-解码器范式利用双向光流作为中间监督信号。最近的扩散方法将其建模为潜空间去噪过程。

**现有痛点**：(1) 非扩散方法（如 SGM-VFI）由于中间帧光流本质上是不适定问题（多解），只能产生过度平滑的均值解；(2) 扩散方法（如 LDMVFI、CBBD）虽能生成更锐利的结果，但直接在潜空间去噪的搜索空间太大，效率低且无法处理复杂运动和大位移。

**核心矛盾**：潜空间维度远大于光流空间（2 通道 × 空间分辨率），直接对潜空间做扩散效率低且不利于建模运动结构。

**切入角度**：光流只有 4 个通道（双向各 2 通道），搜索空间远小于潜空间。从粗到细地估计光流可以自然处理大位移运动。

**核心 idea**：将扩散过程从潜空间转移到光流空间，用层级从粗到细的策略高效去噪光流，再通过流引导合成器产出最终帧。

## 方法详解

### 整体框架

三阶段训练流程：(1) 第一阶段训练流引导图像合成器（编码器-解码器）；(2) 第二阶段冻结合成器训练层级流扩散模型；(3) 第三阶段端到端联合微调合成器和扩散模型。推理时：编码器提取多尺度特征→层级扩散从噪声去噪出多尺度光流→光流引导解码器合成目标帧。

### 关键设计

1. **流引导图像合成器 (Flow-Guided Image Synthesizer)**:

    - 功能：在已知光流条件下从两帧合成中间帧
    - 核心思路：多尺度编码器-解码器架构。在每个尺度上用光流 warp 编码器特征，与解码器特征融合。最终输出包含混合 mask $M$、RGB 残差 $\Delta\mathbf{I}$，合成公式为 $\tilde{\mathbf{I}}_t = M \odot w(\mathbf{I}_0, \tilde{f}_0) + (1-M) \odot w(\mathbf{I}_1, \tilde{f}_1) + \Delta\mathbf{I}$
    - 设计动机：先用预训练光流网络（UniMatch）产出伪 GT 光流训练合成器，使其学会从光流进行高质量图像合成，为后续扩散模型提供强条件信息

2. **层级流扩散模型 (Hierarchical Flow Diffusion)**:

    - 功能：从高斯噪声逐级去噪出多尺度双向光流
    - 核心思路：将 DDPM 去噪过程均匀分配到 3 个金字塔层级（$k_1{=}4$ 到 $k_0{=}2$，即 1/16→1/4 原分辨率）。在每个层级 $i$，U-Net 以该层级的编码器特征 $(\mathbf{F}_0^i, \mathbf{F}_1^i)$ 为条件去噪光流。跨层级过渡时，将当前估计光流 2× 上采样并用 DDPM 前向函数近似下一层级的输入。各层级共享 U-Net 参数，仅 flow projector 和 feature projector 独立
    - 设计动机：从粗到细策略天然适合处理大位移（粗层级捕获大运动，细层级补充细节）。光流空间仅 4 通道，搜索空间远小于潜空间，去噪更高效

3. **端到端联合微调 (End-to-End Joint Fine-tuning)**:

    - 功能：将合成器和扩散模型联合优化，消除两阶段分离训练的信息断裂
    - 核心思路：扩散模型输出的多尺度光流直接用于 warp 编码器特征送入合成器解码器，用光度损失监督最终合成图像质量。合成器和扩散模型同时更新梯度
    - 设计动机：分离训练时合成器针对"完美"伪 GT 光流优化，但实际扩散模型输出的光流有预测误差，联合微调使两者互相适应

### 损失函数 / 训练策略

- **第一阶段**（合成器训练）：光度损失 $\mathcal{L}_{photo} = \mathcal{L}_{pixel} + 0.1 \cdot \mathcal{L}_{LPIPS} + 20 \cdot \mathcal{L}_{style}$，200 epochs，batch 64
- **第二阶段**（扩散训练）：多尺度光流 L1 损失 $\mathcal{L}_{flow} = \sum_i \|\tilde{f}_0^i - f_0^i\|_1 + \|\tilde{f}_1^i - f_1^i\|_1$，200 epochs，1000 去噪步
- **第三阶段**（联合微调）：光度损失，100 epochs，batch 32
- 推理时使用 DDIM（$\sigma_t{=}0$）采样，仅需 6 步

## 实验关键数据

### 主实验

SNU-FILM 基准（LPIPS/FID，↓越低越好）：

| 方法 | easy LPIPS | hard LPIPS | extreme LPIPS | extreme FID |
|------|-----------|-----------|--------------|------------|
| SGM-VFI | 0.0191 | 0.0611 | 0.1182 | 41.078 |
| CBBD (扩散) | 0.0112 | 0.0467 | 0.1040 | 36.729 |
| **Ours** | **0.0098** | **0.0405** | **0.0839** | **27.032** |

Xiph-4K（高分辨率挑战）：

| 方法 | LPIPS | FID |
|------|-------|-----|
| CBBD | 0.0634 | 24.621 |
| **Ours** | **0.0614** | **14.132** |

DAVIS + Vimeo-90k：

| 数据集 | 方法 | LPIPS | FID |
|-------|------|-------|-----|
| DAVIS | CBBD | 0.0919 | 9.220 |
| DAVIS | **Ours** | **0.0753** | **7.237** |
| Vimeo | CBBD | 0.0123 | 1.961 |
| Vimeo | **Ours** | **0.0120** | **1.712** |

### 消融实验

| 配置 | SNUFILM-hard LPIPS | extreme LPIPS |
|------|-------------------|--------------|
| Vanilla（单尺度扩散） | 0.0625 | 0.1199 |
| **层级扩散（Ours）** | **0.0405** | **0.0839** |

### 关键发现

- 在所有 4 个数据集上全面超越现有最佳扩散方法 CBBD 和非扩散方法 SGM-VFI
- 在困难场景（hard/extreme）中优势尤为显著：extreme FID 27.0 vs CBBD 36.7（改善 26%）
- 推理速度 0.20s（1024×1024），与最快的非扩散方法 SGM-VFI 持平，比扩散 CBBD 快 10×
- 层级策略相比单尺度扩散在 hard 子集上 LPIPS 改善 35%

## 亮点与洞察

1. **扩散目标的巧妙转移**：不对潜空间扩散而对光流扩散，将搜索空间从高维潜空间缩减到 4 通道光流，实质性提升效率
2. **层级从粗到细与扩散天然兼容**：扩散本身就是噪声→信号的渐进过程，与光流从粗到细的估计方式完美契合
3. **达成速度-质量的双重 SOTA**：同时在精度和效率上超越所有基线，打破了扩散方法"质量换速度"的固有印象

## 局限与展望

- 依赖预训练光流网络提供伪 GT，光流质量上界受限于该网络
- 仅支持两帧间单帧插值，未讨论多帧插值或任意时间步插值
- 仅用 6 步推理采样，更多步数是否能进一步提升质量未充分探讨
- 可探索将层级扩散策略推广到视频生成或其他对运动敏感的任务

## 相关工作与启发

- **SGM-VFI**：非扩散 SOTA，统一前向/反向光流框架，高效但结果偏平滑
- **CBBD**：基于扩散的插帧方法，本文在其潜空间扩散基础上改为光流扩散
- **FlowDiffuser / DDVM**：将扩散应用于光流估计的工作，但针对的是有 GT 光流的监督设定
- 启发：层级扩散策略可推广到其他需要多尺度结构化预测的视觉任务

## 评分

⭐⭐⭐⭐ — 方法设计简洁有效，动机清晰，实验全面且令人信服。将扩散从潜空间转移到光流空间是关键洞察，速度和质量的双重提升有实际应用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[ICCV 2025\] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](../../ICCV2025/image_generation/tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[CVPR 2025\] Nested Diffusion Models Using Hierarchical Latent Priors](nested_diffusion_models_using_hierarchical_latent_priors.md)
- [\[CVPR 2025\] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention](dig_scalable_and_efficient_diffusion_models_with_gated_linear_attention.md)

</div>

<!-- RELATED:END -->
