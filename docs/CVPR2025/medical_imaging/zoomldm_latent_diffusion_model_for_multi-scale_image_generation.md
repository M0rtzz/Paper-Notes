---
title: >-
  [论文解读] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation
description: >-
  [CVPR 2025][医学图像][多尺度生成] ZoomLDM 提出了一个尺度条件的潜在扩散模型，通过可训练的 Summarizer 模块构建跨倍率潜在空间，实现了病理图像在多个尺度下的高质量生成，并首次支持最大 $4096 \times 4096$ 像素的全局一致大图像合成和无训练超分辨率。
tags:
  - CVPR 2025
  - 医学图像
  - 多尺度生成
  - 潜在扩散模型
  - 病理图像
  - 自监督学习
  - 大图像合成
---

# ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation

**会议**: CVPR 2025  
**arXiv**: [2411.16969](https://arxiv.org/abs/2411.16969)  
**代码**: https://github.com/cvlab-stonybrook/ZoomLDM  
**领域**: 医学图像/扩散模型  
**关键词**: 多尺度生成, 潜在扩散模型, 病理图像, 自监督学习, 大图像合成

## 一句话总结

ZoomLDM 提出了一个尺度条件的潜在扩散模型，通过可训练的 Summarizer 模块构建跨倍率潜在空间，实现了病理图像在多个尺度下的高质量生成，并首次支持最大 $4096 \times 4096$ 像素的全局一致大图像合成和无训练超分辨率。

## 研究背景与动机

**领域现状**：扩散模型在自然图像生成中取得了巨大成功，但在大图像领域（如数字病理学、卫星图像，常达 GB 级别的千兆像素图像）应用受限。现有方法主要训练固定尺寸 patch 的扩散模型，无法捕获大图像的全局结构。

**现有痛点**：(1) 直接在全幅千兆像素图像上训练不可行；(2) 单尺度 patch 模型只能生成局部细节，缺乏全局上下文——在低倍率（如 $0.15625\times$）数据极度稀缺时质量急剧下降；(3) 大图像领域缺乏配对的图像-文本标注，无法像 Stable Diffusion 那样使用文本条件；(4) 现有大图像生成方法要么模糊（$\infty$-Brush）要么缺乏全局一致性（Graikos et al.）。

**核心矛盾**：大图像的全局结构需要低倍率信息理解，局部细节需要高倍率渲染，但单一模型无法同时覆盖从 $20\times$ 到 $0.15625\times$ 的 128 倍倍率跨度，且低倍率数据极为稀缺。

**本文目标**：训练一个统一的多尺度扩散模型，能在所有倍率下生成高质量 patch，并利用多尺度联合采样实现全局一致的大图像合成。

**切入角度**：不同尺度的病理 patch 虽然分辨率相同（$256 \times 256$），但每个像素编码的"缩放级别"不同。如果能训练一个尺度感知的模型，跨尺度共享权重让数据丰富的尺度帮助数据稀缺的尺度，就能解决数据不足问题。利用自监督学习（SSL）编码器（如 UNI）替代缺失的文本标注提供条件信号。

**核心 idea**：训练一个共享权重的尺度条件扩散模型，通过可训练的 Summarizer 将不同尺度的 SSL 嵌入映射到统一的跨倍率潜在空间，实现多尺度协同训练和联合采样。

## 方法详解

### 整体框架

ZoomLDM 的训练流程：(1) 从大图像的初始尺度（$20\times$）提取 $256 \times 256$ patch 并用 UNI 编码器提取 SSL 嵌入；(2) 逐步将大图像下采样 2 倍，每个尺度的 patch 与对应的初始尺度区域的 SSL 嵌入矩阵配对；(3) Summarizer transformer 将变长的 SSL 嵌入矩阵（加上倍率嵌入）压缩为固定大小的条件向量；(4) LDM（VQ-f4 自编码器 + U-Net 去噪器）以 Summarizer 输出为条件进行多尺度统一训练。推理时，通过条件扩散模型（CDM）直接采样潜在空间中的条件向量，无需真实图像。

### 关键设计

1. **跨倍率潜在空间与 Summarizer 模块**:

    - 功能：将不同尺度的 SSL 嵌入统一映射到共享的条件空间，实现多尺度协同训练
    - 核心思路：不同倍率的 patch 对应不同数量的初始尺度 SSL 嵌入（$5\times$ patch 对应 $4 \times 4 = 16$ 个 $20\times$ 嵌入，$1.25\times$ 对应 $16 \times 16 = 256$ 个）。Summarizer 是一个 12 层 ViT-Base transformer，接收变长的嵌入序列加倍率位置嵌入，通过 padding + pooling 输出固定大小的条件 token。倍率嵌入使 Summarizer 感知当前尺度，提取对应尺度所需的信息。大于 $8 \times 8$ 的嵌入矩阵预先池化到 $8 \times 8$ 以控制计算量。
    - 设计动机：SSL 编码器（如 UNI）仅在初始尺度（$20\times$）训练，不能直接对其他尺度的图像提取有意义的特征。通过将初始尺度嵌入作为"内容描述"传递给所有尺度，Summarizer 负责根据倍率提取相应粒度的信息。这比为每个尺度训练独立 SSL 编码器高效得多。

2. **条件扩散模型（CDM）**:

    - 功能：在无需真实图像的情况下采样条件向量，实现完全无条件的新图像生成
    - 核心思路：在 ZoomLDM 训练完成后，训练一个 Diffusion Transformer 来建模 Summarizer 输出空间的分布。以倍率为条件，CDM 可以直接采样新的条件向量用于 LDM 生成。这比建模原始 SSL 嵌入分布简单得多，因为 Summarizer 输出是经过压缩的任务相关表示。
    - 设计动机：实际应用中不一定能获取真实图像的 SSL 嵌入（如生成全新的合成图像），CDM 解耦了生成过程对真实数据的依赖。尤其在数据极度稀缺的低倍率（如 $0.15625\times$ 仅 2500 张），CDM 仍能生成非记忆化的新图像。

3. **联合多尺度采样（Joint Multi-Scale Sampling）**:

    - 功能：生成跨多个尺度的全局一致大图像
    - 核心思路：利用不同尺度图像之间的线性下采样关系 $\mathbf{x}^{s+1} = \mathbf{A}(\mathbf{x}_1^s, \mathbf{x}_2^s, \mathbf{x}_3^s, \mathbf{x}_4^s)$，在扩散过程的每个去噪步中，先估计各尺度的"干净图像"，用高尺度图像作为低尺度的自引导信号，通过有限差分近似计算梯度方向，更新低尺度的 noisy latent 使其与高尺度保持一致。相比原始算法，避免了昂贵的反向传播，使用数值近似高效计算误差方向。
    - 设计动机：独立采样各尺度会导致全局不一致（高尺度看到森林但低尺度生成了沙漠）。联合采样通过"高尺度引导低尺度"确保语义一致性，高尺度提供全局上下文（什么组织类型），低尺度保证局部细节（细胞形态）。

### 损失函数 / 训练策略

LDM 使用标准潜在扩散训练目标。Summarizer 与 LDM 联合端到端训练。CDM 使用 DiT 架构训练。DDIM 采样 50 步，classifier-free guidance scale=2.0。VQ-f4 自编码器和 U-Net 从 ImageNet 预训练检查点初始化。

## 实验关键数据

### 主实验（各倍率 FID）

| 倍率 | 训练样本数 | ZoomLDM | 单尺度 SoTA | CDM |
|------|----------|---------|-----------|-----|
| 20× | 12M | **6.77** | 6.98 | 9.04 |
| 10× | 3M | **7.60** | 7.64 | 10.05 |
| 5× | 750K | **7.98** | 9.74 | 14.36 |
| 2.5× | 186K | **10.73** | 20.45 | 19.68 |
| 1.25× | 57K | **8.74** | 39.72 | 14.06 |
| 0.625× | 20K | **7.99** | 58.98 | 13.46 |
| 0.3125× | 7K | **8.34** | 66.28 | 14.40 |
| 0.15625× | 2.5K | **13.42** | 106.14 | 26.09 |

### 消融实验（大图像生成 $1024 \times 1024$）

| 方法 | CLIP FID↓ | Crop FID↓ | 推理时间 |
|------|----------|----------|---------|
| Graikos et al. | 7.43 | 15.51 | 60s |
| ∞-Brush | 3.74 | 17.87 | 30s |
| **ZoomLDM** | **1.23** | **14.94** | 28s |

### 关键发现

- **跨尺度权重共享的效果惊人**：在数据仅 2500 张的 $0.15625\times$ 倍率，单尺度模型 FID 高达 106.14，ZoomLDM 降至 13.42（7.9 倍改善）。多尺度协同训练让数据丰富的高倍率帮助了低倍率。
- 首次实现了 $4096 \times 4096$ 病理图像的全局一致合成，推理仅需 8 分钟（vs ∞-Brush 的 12 小时）。
- 超分辨率（$4\times$）在 SSIM/PSNR/LPIPS 上均超越 CompVis 和 ControlNet 基线，无需额外训练。
- ZoomLDM 的中间特征在 MIL 实验中超越 SoTA SSL 编码器——多尺度生成学习产生的特征比判别式编码器更具表达力。

## 亮点与洞察

- **多尺度生成用于表示学习**的发现非常有价值：ZoomLDM 仅从 $20\times$ 单尺度的特征就超越了同尺度的 UNI（最先进的病理 SSL 编码器），说明学习生成多尺度图像迫使模型学会更丰富的多尺度内部表示。
- **Summarizer 的跨倍率潜在空间**设计巧妙地解决了 SSL 编码器单尺度训练的问题——不是为每个尺度训练新编码器，而是学习一个映射层从已有嵌入中提取尺度适配的信息。
- 联合采样的"自引导"思路（用模型自身的高尺度预测引导低尺度生成）无需外部信息，是 test-time 的计算开销换质量的有效策略。

## 局限与展望

- 联合采样在 $4096 \times 4096$ 时需要同时处理 $16 \times 16 + 1 = 257$ 个去噪过程，计算开销仍然较大（8 分钟/张）
- $4096 \times 4096$ 的 CLIP FID 和 Crop FID 不如 ∞-Brush，说明全局一致性的代价是牺牲了部分局部细节多样性
- 目前仅在 TCGA-BRCA 病理数据集上进行详细评估，卫星图像结果在补充材料中
- 可探索更高效的联合采样算法，以及将方法扩展到 3D 医学图像（如 CT 的多分辨率生成）

## 相关工作与启发

- **vs Graikos et al.**: 仅训练单尺度 patch 模型再通过 SSL 嵌入空间变化生成大图像，缺乏对全局结构的真正理解。ZoomLDM 通过多尺度联合训练+采样实现了内在的全局一致性。
- **vs ∞-Brush**: 使用无限维扩散模型理论上可以处理任意分辨率，但实际结果模糊且推理极慢（12小时/4k图像）。ZoomLDM 速度快 90 倍且细节更好。
- **vs Harb et al.**: 也训练多尺度病理扩散模型但没有条件机制，严重限制了生成质量和可控性。ZoomLDM 的 Summarizer 条件化设计是关键创新。

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨倍率潜在空间和联合采样思路新颖，多尺度生成在病理领域有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个倍率的 FID、大图像生成、超分辨率、MIL 实验全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整
- 价值: ⭐⭐⭐⭐⭐ 首个实用的千兆像素病理图像多尺度生成方法，对医学 AI 社区价值重大

<!-- RELATED:START -->

## 相关论文

- [Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](../../ICML2025/medical_imaging/ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)
- [Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/medical_imaging/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)

<!-- RELATED:END -->
