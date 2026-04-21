---
title: >-
  [论文解读] OSDFace: One-Step Diffusion Model for Face Restoration
description: >-
  [CVPR 2025][图像生成][人脸修复] OSDFace 提出了首个专门针对人脸修复的单步扩散模型，通过视觉表示嵌入器（VRE）从低质量人脸中提取丰富先验信息，结合面部身份损失和 GAN 引导，仅需一步推理（约 0.1 秒）即可生成高保真、自然且身份一致的人脸图像，全面超越现有 SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - 人脸修复
  - 单步扩散
  - 向量量化
  - 视觉先验
  - GAN
---

# OSDFace: One-Step Diffusion Model for Face Restoration

**会议**: CVPR 2025  
**arXiv**: [2411.17163](https://arxiv.org/abs/2411.17163)  
**代码**: https://github.com/jkwang28/OSDFace  
**领域**: 图像生成  
**关键词**: 人脸修复, 单步扩散, 向量量化, 视觉先验, GAN引导

## 一句话总结

OSDFace 提出了首个专门针对人脸修复的单步扩散模型，通过视觉表示嵌入器（VRE）从低质量人脸中提取丰富先验信息，结合面部身份损失和 GAN 引导，仅需一步推理（约 0.1 秒）即可生成高保真、自然且身份一致的人脸图像，全面超越现有 SOTA。

## 研究背景与动机

**领域现状**：人脸修复旨在从受模糊、噪声、下采样和 JPEG 压缩等复杂退化影响的低质量（LQ）图像中恢复高质量（HQ）人脸。当前方法主要分为三类：基于 CNN/Transformer 的方法（如 RestoreFormer++）、基于 GAN 的方法（训练不稳定、模式坍塌）、以及基于扩散模型的方法（如 DifFace、DiffBIR，质量好但推理慢）。

**现有痛点**：第一，多步扩散模型推理成本高——PGDiff 需要 1000 步/85.8 秒，DiffBIR 需要 50 步/9.03 秒。第二，现有方法在"和谐性"上表现不佳，即使能恢复基本面部特征（眼睛、嘴巴），头发和复杂背景等细节仍不自然。根本原因是面部先验的融入不充分：有些方法（DifFace、DiffBIR）完全不考虑人脸先验；有些方法（PGDiff）用先验但限制了生成能力。第三，通用的单步扩散修复模型（如 OSEDiff）不是为人脸特别设计的，人类对面部特征极度敏感，细微不协调即可察觉。

**核心矛盾**：快速推理（单步扩散）与高质量人脸修复（需要丰富先验）之间的矛盾；通用图像修复模型无法满足人脸这一特殊领域的高标准要求。

**本文目标**：设计一个专门针对人脸的单步扩散模型，同时实现快速推理、高保真修复和身份一致性。

**切入角度**：作者观察到 VQ 先验方法（如 CodeFormer）能有效利用 codebook 捕捉面部特征，但直接用 codebook 生成图像缺乏细节。如果将 VQ 先验与扩散模型结合——用 VQ 提取丰富先验作为扩散模型的条件——就能兼顾两者优势。

**核心 idea**：设计视觉表示嵌入器（VRE）从低质量图像中直接提取视觉先验 prompt（绕过图像→标签→嵌入的信息损失），配合面部身份损失和 GAN 分布对齐，实现高质量单步人脸修复。

## 方法详解

### 整体框架

OSDFace 的训练分两个阶段。第一阶段：训练视觉表示嵌入器（VRE），通过自重建训练 HQ 和 LQ 两个 VQVAE，建立视觉 token 字典，并对齐两个域的特征类别。第二阶段：将预训练的 VRE 集成到 Stable Diffusion 中——LQ 图像经 VAE 编码器映射到潜在空间，VRE 提取视觉 prompt 嵌入，UNet（仅通过 LoRA 微调）预测一步噪声，VAE 解码器重建 HQ 人脸。整个推理流程仅需 0.1 秒。

### 关键设计

1. **视觉表示嵌入器 (VRE)**:

    - 功能：从低质量人脸中提取丰富的视觉先验 prompt
    - 核心思路：VRE 由两部分组成。**视觉分词器**：LQ VAE 编码器 $E_L$ + VQ 匹配函数 $\mathcal{M}$，将 LQ 人脸映射到可学习的 LQ 字典 $\mathbb{C}_L = \{c_q \in \mathbb{R}^d\}_{q=1}^N$ 中的类别 token $\mathbf{Q}_L = \mathcal{M}(E_L(I_L))$。**VQ 嵌入器**：用 token 作为索引在字典中查找对应的嵌入向量，$z_k = \text{dict}(q)$，时间复杂度 $\mathcal{O}(1)$。与 image-to-tag 方法不同，VRE 直接将人脸分词为视觉嵌入，避免了图像→标签→嵌入的信息损失
    - 设计动机：注意力图可视化表明 VRE 同时关注面部和非面部特征（头发、背景等），捕获了比文本标签更丰富的信息

2. **特征关联训练策略**:

    - 功能：对齐 HQ 和 LQ 两个域的 VQ 字典类别
    - 核心思路：构建 HQ 和 LQ 两个 VQ 字典，分别训练 VQVAE 自重建。然后受 CLIP 启发，构建 HQ 和 LQ 编码特征的相似度矩阵 $M_{\text{assoc}}$，用交叉熵损失增强对角相关性，引导 LQ 编码器的注意力与 HQ 编码器对齐。关联损失 $\mathcal{L}_{\text{assoc}} = (\mathcal{L}_{\text{CE}}^H + \mathcal{L}_{\text{CE}}^L) / 2$
    - 设计动机：LQ 编码器由于强退化可能关注到无意义的类别，通过跨域对齐确保 LQ token 能对应到有意义的 HQ 语义

3. **面部身份损失与 GAN 引导**:

    - 功能：确保修复后人脸的身份一致性和整体真实性
    - 核心思路：**面部身份损失**：使用预训练 ArcFace 模型提取生成人脸和 GT 的身份嵌入，计算余弦相似度损失 $\mathcal{L}_{\text{ID}} = 1 - \cos(\mathcal{F}(I_H), \mathcal{F}(\hat{I}_H))$。**感知损失**：使用边缘感知 DISTS（EA-DISTS），在标准 DISTS 基础上加入 Sobel 边缘处理后的 DISTS，增强纹理和边缘细节还原。**GAN 损失**：使用判别器在扩散潜在空间中对齐生成分布和真实分布，提供比蒸馏更灵活的训练信号。总损失 $\mathcal{L}_{\text{gen}} = \lambda_{\text{dis}} \mathcal{L}_{\mathcal{G}} + \lambda_{\text{ID}} \mathcal{L}_{\text{ID}} + \lambda_{\text{per}} \mathcal{L}_{\text{EA-DISTS}} + \text{MSE}$
    - 设计动机：人类对面部极度敏感，细微不一致即可察觉，需要多方面的损失来确保和谐性：身份损失管身份一致、GAN 损失管分布对齐、感知损失管纹理细节

### 损失函数 / 训练策略

第一阶段 VRE 训练损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \lambda_{\text{per}} \mathcal{L}_{\text{per}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{VQ}} + \lambda_{\text{assoc}} \mathcal{L}_{\text{assoc}}$，其中 $\lambda_{\text{assoc}}$ 初始为 0，后调为 1。第二阶段冻结 VAE 编码器/解码器和 VRE，仅通过 LoRA 微调 UNet，生成器和判别器交替训练。推理时使用预定义的固定时间步 $T_L$，LQ 潜在向量直接作为 UNet 输入（不是纯高斯噪声）。

## 实验关键数据

### 主实验

| 方法 | 步数 | LPIPS↓ | DISTS↓ | MUSIQ↑ | NIQE↓ | FID(HQ)↓ |
|------|------|--------|--------|--------|-------|---------|
| CodeFormer | - | 0.3412 | 0.2151 | 75.94 | 4.52 | 26.86 |
| DifFace (250步) | 250 | 0.3469 | 0.2126 | 66.75 | 4.64 | 22.24 |
| DiffBIR (50步) | 50 | 0.3740 | 0.2340 | 75.64 | 6.28 | 32.51 |
| OSEDiff* (1步) | 1 | 0.3496 | 0.2200 | 69.98 | 5.33 | 37.13 |
| **OSDFace (1步)** | 1 | **0.3365** | **0.1773** | **75.64** | **3.88** | **17.06** |

### 消融实验

论文图 2 展示了综合性能雷达图，OSDFace 在 LPIPS、DISTS、MUSIQ、NIQE、Deg、LMD、FID(FFHQ)、FID(HQ) 8 个指标中大部分取得最佳，尤其在 DISTS（纹理质量）和 NIQE（自然度）上大幅领先。

### 关键发现

- OSDFace 仅需 0.1 秒推理就超越了所有多步扩散方法和非扩散方法，计算量仅 2.132T MACs
- 与通用单步扩散模型 OSEDiff 相比，DISTS 从 0.2200 降到 0.1773（-19.4%），FID(HQ) 从 37.13 降到 17.06（-54%），证明了人脸专用设计的必要性
- 在真实世界数据集（WIDER、WebPhoto、LFW）上的零样本泛化表现优异
- 注意力图可视化证实 VRE 能同时关注面部特征和背景/头发等非面部区域

## 亮点与洞察

- VRE 的设计思路精巧：用 VQ 字典捕捉类别级先验，直接从视觉 token 生成 prompt 嵌入，避免了 image-to-tag 的信息瓶颈
- 跨域特征关联策略让 LQ 字典能找到有意义的语义映射，即使输入严重退化
- 不依赖蒸馏（不需要多步教师模型），而是用 GAN 引导实现分布对齐，训练更灵活
- 0.1 秒完成 512×512 人脸修复，具有很强的实际部署价值

## 局限与展望

- 当前模型针对人脸设计，泛化到其他图像类型需要额外适配
- VQ 字典大小固定，可能在面对极度多样化的人脸风格时覆盖不足
- 单步扩散的生成多样性低于多步扩散，对于人脸修复这种确定性任务影响不大，但在需要多样性的场景下可能受限
- 在极端退化（如极低分辨率或严重遮挡）下的效果有待进一步验证

## 相关工作与启发

- 与 CodeFormer（VQ 先验直接生成图像）和 PGDiff（VQ 先验作为扩散引导目标）不同，OSDFace 将 VQ 先验作为扩散模型的条件 prompt，在灵活性和质量之间取得更好平衡
- 面部身份损失的引入启发了面部修复任务中身份保持的重要性
- VRE 的 LQ 字典方案可能对其他退化图像理解任务（如医学图像增强）有借鉴意义

## 评分

- 新颖性：⭐⭐⭐⭐ — VRE 设计巧妙，VQ 先验 + 单步扩散的结合方式新颖
- 实验充分度：⭐⭐⭐⭐⭐ — 8 个指标全面评估，多个基线对比，真实世界零样本测试
- 写作质量：⭐⭐⭐⭐ — 图表丰富，方法描述清晰，可视化分析有说服力
- 价值：⭐⭐⭐⭐⭐ — 0.1 秒高质量人脸修复，具有直接的工业应用价值

<!-- RELATED:START -->

## 相关论文

- [SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [Unlocking the Potential of Diffusion Priors in Blind Face Restoration](../../ICCV2025/image_generation/unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)

<!-- RELATED:END -->
