---
title: >-
  [论文解读] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography
description: >-
  [ICCV 2025][人体理解][metalens] 提出DMDiff框架，利用预训练扩散模型的自然图像先验，通过正/中/负三路径多提示扩散策略和空间变化退化感知注意力（SVDA）模块，实现毫米级超透镜相机的高保真可调图像重建，在多项指标上超越现有方法。
tags:
  - ICCV 2025
  - 人体理解
  - metalens
  - 扩散模型
  - 图像恢复
  - computational imaging
  - LoRA
---

# Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography

**会议**: ICCV 2025  
**arXiv**: [2506.22753](https://arxiv.org/abs/2506.22753)  
**代码**: https://dmdiff.github.io/  
**领域**: human_understanding  
**关键词**: metalens, diffusion model, image restoration, computational imaging, LoRA

## 一句话总结
提出DMDiff框架，利用预训练扩散模型的自然图像先验，通过正/中/负三路径多提示扩散策略和空间变化退化感知注意力（SVDA）模块，实现毫米级超透镜相机的高保真可调图像重建，在多项指标上超越现有方法。

## 研究背景与动机
超透镜（metalens）作为超紧凑成像系统具有巨大潜力，但面临复杂光学退化的挑战。现有方法要么依赖精确的光学标定（获取困难），要么需要大规模配对数据集（难以收集），要么使用深度学习方法但缺乏对推理过程的控制，导致生成幻觉伪影。→ 核心矛盾在于：如何在缺少大规模训练数据的情况下，利用预训练大模型先验有效恢复超透镜的空间变化退化，同时控制生成过程避免幻觉？→ 本文提出利用预训练扩散模型的自然图像先验替代大数据集，通过多路径提示策略平衡细节生成和结构保真，设计可调解码器控制重建质量。

## 方法详解

### 整体框架
DMDiff基于SD-Turbo（Stable Diffusion的蒸馏版本），包含VAE编码器、潜在扩散UNet、VAE解码器和SVDA模块。输入超透镜拍摄的图像被编码到潜在空间，UNet进行一步去噪（k=1），结合文本提示和SVDA模块的退化线索，生成高质量重建图像。通过LoRA对编码器和UNet进行高效微调。

### 关键设计
1. **空间变化退化感知注意力（SVDA）模块**:

    - 功能：量化超透镜和传感器引起的空间变化退化，引导LoRA微调过程
    - 核心思路：结合两种退化度量——基于PSF的FWHM（光学像差）和基于MUSIQ的无参考图像质量评估（传感器噪声）。将图像划分为n×n个patch，计算每个patch的FWHM和NR-IQA分数，通过注意力网络生成r×r的注意力矩阵Q，嵌入LoRA过程：$W^* = W + AQB$
    - 设计动机：超透镜退化是空间变化的，传统方法假设均匀退化无法处理；精确PSF标定困难且受制造误差影响，需同时考虑光学和电子传感器两种退化源

2. **多路径扩散训练**:

    - 功能：通过正、中、负三条路径分别学习不同目标
    - 核心思路：正路径（输入退化图像→高质量GT）学习高频细节生成；中路径（输入退化图像→低通滤波GT）学习结构保真；负路径（输入GT→退化图像）学习超透镜退化模式并生成伪数据对扩充训练集。三条路径按概率$M \sim \text{Cat}(p_1, p_2, p_3)$随机选择
    - 设计动机：扩散模型虽能生成逼真细节但易产生幻觉，通过中路径保持结构准确性、负路径学习退化特征进行抑制，三路径协同平衡感知质量和重建保真度

3. **即时可调解码器**:

    - 功能：推理时动态调整重建结果在感知质量和客观精度之间的平衡
    - 核心思路：分别获取正路径和中路径的潜在编码$z_{pos}$和$z_{neu}$，通过可调参数α混合后解码：$I^* = D(\alpha \cdot z_{pos} + (1-\alpha) \cdot z_{neu})$
    - 设计动机：不同应用场景对重建质量的需求不同，α越大感知质量越好但可能有过多细节，α越小则保真度更高

### 损失函数 / 训练策略
训练损失为L2损失和LPIPS感知损失的加权组合：$L = L_2 + \lambda \cdot L_{\text{LPIPS}}$，其中$\lambda = 2.5$。在4块A100 80G GPU上训练两天，batch size为16。SVDA中patch数n=7。

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | MUSIQ↑ | CLIP-IQA↑ |
|------|-------|-------|--------|--------|--------|-----------|
| Wiener deconv | 16.06 | 0.5727 | 0.6706 | 0.4393 | 17.41 | 0.2681 |
| Neural nano-optics | 29.25 | 0.8624 | 0.2001 | 0.1765 | 37.26 | 0.2746 |
| SwinIR | 29.46 | 0.8786 | 0.2462 | 0.2111 | 36.86 | 0.3046 |
| SeeSR-s50 | 23.95 | 0.8340 | 0.2315 | 0.1673 | 44.87 | 0.3913 |
| OSEDiff-s1 | 19.69 | 0.8224 | 0.2643 | 0.1868 | 34.52 | 0.3761 |
| **Ours-s1-α0.75** | **30.31** | 0.8731 | 0.1705 | 0.1499 | 44.48 | 0.3869 |
| **Ours-s1-α1.05** | 29.75 | 0.8598 | **0.1485** | **0.1356** | **51.85** | **0.4460** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | MANIQA↑ | MUSIQ↑ | 说明 |
|------|-------|-------|--------|---------|--------|------|
| Base (无任何模块) | 17.12 | 0.7685 | 0.3455 | 0.2332 | 38.27 | 简单LoRA微调无法恢复 |
| w/o FWHM | 26.62 | 0.8414 | 0.1869 | 0.2966 | 50.55 | 去除光学退化建模 |
| w/o Neg prompt | 28.21 | 0.8571 | 0.1953 | 0.2587 | 44.15 | 去除负路径退化学习 |
| **Ours-α1** | **29.89** | **0.8633** | **0.1504** | **0.3078** | **50.63** | 完整方法 |

| α值 | PSNR↑ | LPIPS↓ | MUSIQ↑ | 说明 |
|-----|-------|--------|--------|------|
| 0 | 30.06 | 0.2715 | 31.24 | 纯中路径，保真但缺细节 |
| 0.5 | 30.39 | 0.2039 | 38.96 | 平衡 |
| 0.7 | 30.31 | 0.1705 | 44.48 | 较好平衡点 |
| 1.0 | 29.89 | 0.1504 | 50.63 | 更强感知质量 |
| 1.05 | 29.75 | 0.1485 | 51.85 | 最佳感知质量 |

### 关键发现
- 非扩散方法（如SwinIR）生成模糊图像缺少高频细节，但保持准确色调和低频结构
- 扩散方法能生成丰富细节但缺乏退化建模导致色调错误和错误细节
- 本方法在图像边缘区域（退化最严重）仍保持高性能，其他方法在边缘区域明显退化
- 负路径生成的伪数据有效扩充了训练集，提升泛化能力

## 亮点与洞察
- 将SVDA退化量化与LoRA微调结合的设计非常巧妙，避免了精确PSF标定的需求
- 三路径设计有效解决了扩散模型生成幻觉的固有问题
- 可调解码器无需重新推理即可快速生成不同扩散强度的图像
- 构建了1×1×1 mm³的MetaCamera进行真实硬件验证，工程完整性强

## 局限与展望
- 训练数据仍需显示器-相机对齐采集配对数据，尽管伪数据扩充缓解了这一问题
- 一步扩散可能在极端退化区域仍有不足，多步推理可能进一步提升质量
- SVDA中PSF仍依赖模拟参数，制造误差可能导致实际PSF偏差
- 超透镜的色散特性对基于文本语义的扩散模型构成挑战，本文通过仅使用质量描述提示回避
- 当前仅针对单个超透镜设计优化，迁移到不同超透镜设计可能需要重新微调
- 传感器分辨率仅400×400像素，更高分辨率传感器下的表现有待验证

## 相关工作与启发
- 从OSEDiff/S3Diff的单步扩散思路出发，引入退化建模适应计算成像场景
- SVDA模块的退化感知注意力设计可推广到其他空间变化退化的图像恢复任务（如广角镜头、内窥镜成像）
- 多路径扩散训练策略对控制扩散模型生成质量有普适价值
- 仅使用图像质量描述而非场景内容作为文本提示的策略，巧妙规避了超透镜色散对语义prompt的干扰
- MetaCamera的1mm³超紧凑设计为生物医学植入式成像和AR/VR微型化提供了硬件参考
- 负路径生成伪数据的方法可推广到其他缺乏配对训练数据的计算成像任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 多路径扩散+SVDA退化建模+可调解码的完整设计具有较高创新性
- 实验充分度: ⭐⭐⭐⭐ 合成+真实场景双重验证，消融全面，但缺少与更多计算成像方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，图表设计专业
- 价值: ⭐⭐⭐⭐ 对超紧凑计算成像领域有重要推动，方法论可推广到其他退化恢复任务

<!-- RELATED:START -->

## 相关论文

- [SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [A Practical Guide for Incorporating Symmetry in Diffusion Policy](../../NeurIPS2025/human_understanding/a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)
- [DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration](dynfacerestore_balancing_fidelity_and_quality_in_diffusion-guided_blind_face_res.md)
- [GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)

<!-- RELATED:END -->
