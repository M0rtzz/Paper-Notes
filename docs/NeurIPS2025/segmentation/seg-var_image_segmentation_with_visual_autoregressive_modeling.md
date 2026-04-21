---
title: >-
  [论文解读] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling
description: >-
  [NeurIPS 2025][图像分割][视觉自回归建模] Seg-VAR 将图像分割重新定义为条件自回归掩码生成问题，通过引入 seglat（分割掩码的潜在表示）和空间感知颜色映射将分割掩码编码为可由 VAR 模型处理的离散 token，在 COCO、Cityscapes、ADE20K 上的语义/实例/全景分割任务中全面超越 Mask2Former 等判别式方法和 GSS 等生成式方法。
tags:
  - NeurIPS 2025
  - 图像分割
  - 视觉自回归建模
  - 通用图像分割
  - Seglat
  - 空间感知颜色映射
  - 生成式分割
---

# Seg-VAR: Image Segmentation with Visual Autoregressive Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2511.12594](https://arxiv.org/abs/2511.12594)  
**代码**: [GitHub](https://github.com/rkzheng99/Seg-VAR)  
**领域**: 图像分割  
**关键词**: 视觉自回归建模, 通用图像分割, Seglat, 空间感知颜色映射, 生成式分割

## 一句话总结

Seg-VAR 将图像分割重新定义为条件自回归掩码生成问题，通过引入 seglat（分割掩码的潜在表示）和空间感知颜色映射将分割掩码编码为可由 VAR 模型处理的离散 token，在 COCO、Cityscapes、ADE20K 上的语义/实例/全景分割任务中全面超越 Mask2Former 等判别式方法和 GSS 等生成式方法。

## 研究背景与动机

图像分割需要模型捕获从粗粒度语义类别到细粒度实例边界的层次化空间关系。现有方法——无论是 CNN 还是 Transformer 基础架构——通常将分割视为并行预测任务，难以建模迭代式的、上下文依赖的空间和语义关系。

视觉自回归 (VAR) 建模通过将图像序列化为 token 进行生成，其顺序的、上下文累积的特性天然契合分割所需的渐进式精炼。然而现有 VAR 框架主要面向图像合成，忽视了其统一分割任务的潜力。核心障碍在于**表示问题**：

- 大多数自回归框架将图像编码到缺乏显式空间或实例级结构的潜在空间
- 前代生成式分割方法 GSS 虽然引入了 maskige（用 RGB 图像表示分割掩码），但其简单 MLP 变换无法区分重叠实例或保留细粒度位置线索
- 自回归图像生成器通常将像素/patch 视为无序 token，丧失了几何控制

Seg-VAR 通过三项关键创新解决这些问题：空间感知 seglat 编码、层次化自回归解码和多阶段潜在对齐。

## 方法详解

### 整体框架

Seg-VAR 基于变分推断框架，引入离散 $L$ 维潜在分布 $q_\phi(z|c)$ 到对数似然中，利用 ELBO 分解为三个组件：

$$\mathbb{E}_{q_\phi(z|c)}[\log p_\theta(c|z)] - D_{KL}(q_\phi(z|c), p_\psi(z|x))$$

对应三大模块：图像编码器 $\mathcal{I}_\psi$（从输入图像生成潜在 token 的先验分布）、seglat 编码器/解码器 $\mathcal{E}_\phi, \mathcal{D}_\theta$（基于 VAR 结构，将分割掩码编码/解码为离散 token）、潜在编码器/解码器 $\mathcal{T}_\phi, \mathcal{T}_\theta$（在分割掩码和 seglat 之间转换）。

### 关键设计

1. **空间感知 Seglat 编码**：这是最核心的创新。Seglat 是分割掩码的潜在 RGB 图像表示。关键在于颜色映射编码器 $\Psi$：将图像划分为 $a \times a$ 网格，每个网格分配唯一颜色；每个实例根据其重心所在网格获得对应颜色。从 RGB 三通道各选 6 个候选值 $\{0, 51, 102, 153, 204, 255\}$，总共 $6^3 - 1 = 215$ 种颜色（(0,0,0) 表示背景），网格数 $a^2$ 不超过 215。设计动机：Transformer 的位置编码可以帮助预测对应颜色，而手工随机分配颜色则因颜色空间过大而难以区分实例。潜在编码器将分割掩码 $M$ 和颜色映射 $M_c$ 拼接后转换为 seglat $\mathcal{S}$。

2. **层次化自回归解码 (Seglat Encoder/Decoder)**：基于 ControlVAR 设计，在 VAR 的层次化 Transformer 结构的每个尺度 $k$ 上联合建模图像和 seglat。图像 token $X_k$ 和 seglat token $S_k$ 通过共享 tokenizer $\Phi$ 量化到词汇表 $[V]$，然后在同一尺度内使用全注意力：$X_k', S_k' = \text{Attention}(X_k, S_k, S_k)$。引入 `[CLS]` token 提供语义上下文（类别），`[TYP]` token 选择分割任务类型（语义/实例/全景），两者不可消融。设计动机：全注意力使模型既保持空间局部性又能挖掘 seglat 和图像之间的全局联系。

3. **多阶段训练策略**：三阶段渐进训练——(a) **Stage 1 Seglat Learning**：联合训练图像和 seglat 的编码/解码器 $\mathcal{E}_\phi, \mathcal{D}_\theta$，学习 seglat 表示；(b) **Stage 2 Latent Learning**：冻结 seglat 编码/解码器，训练潜在编码/解码器 $\mathcal{T}_\phi, \mathcal{T}_\theta$，优化 $\min \|\mathcal{T}_\theta(\mathcal{S}) - c\|$；(c) **Stage 3 Image Encoder Learning**：冻结其他模块，训练图像编码器 $\mathcal{I}_\psi$ 最小化 KL 散度 $D_{KL}(q_\phi(z|c), p_\psi(z|x))$。设计动机：逐步对齐各组件分布，避免端到端训练的不稳定性。

### 损失函数 / 训练策略

- Seglat 编码/解码器使用标准交叉熵损失监督重建
- 潜在学习阶段使用重建损失
- 图像编码器学习阶段使用交叉熵损失度量 KL 散度
- 推理时使用 top-k (k=900) top-p (p=0.96) 采样
- AdamW 优化器，初始学习率 $10^{-4}$，权重衰减 0.05
- 大规模抖动 (LSJ) 数据增强，随机缩放 0.1-2.0 倍后裁剪到 1024×1024

## 实验关键数据

### 主实验 — 全景分割 (COCO, Swin-L)

| 方法 | PQ | PQ^Th | PQ^St | AP^Th_pan | mIoU_pan |
|------|-----|-------|-------|----------|---------|
| Mask2Former | 57.8 | 64.2 | 48.1 | 48.6 | 67.4 |
| GSS | 44.9 | 50.2 | 32.6 | 36.9 | 54.2 |
| **Seg-VAR** | **59.7** | **65.6** | **50.5** | **49.6** | **68.7** |
| 提升 vs Mask2Former | +1.9 | +1.4 | +2.4 | +1.0 | +1.3 |
| 提升 vs GSS | **+14.8** | — | — | — | — |

### 消融实验

| 配置 | ADE20K mIoU | COCO AP | 说明 |
|------|------------|---------|------|
| 无 Stage 1 + 无 Stage 3 | 78.9 | 46.2 | 基线 |
| + Stage 1 (Seglat Learning) | 83.4 | 52.0 | +4.5 mIoU |
| + Stage 3 (Image Enc. Learning) | 81.6 | 49.3 | +2.7 mIoU |
| Stage 1 + Stage 3 (完整) | **85.8** | **52.7** | 最优 |
| Vanilla VAR (无 seglat 模块) | 77.4 | — | 比 Seg-VAR 低 8.4 |
| VQGAN 替代 VAR | 74.6 | 42.8 | VAR 远优于替代品 |
| SD-XL 替代 VAR | 81.8 | 48.9 | — |

### 关键发现

- **ADE20K 语义分割**：Seg-VAR (Swin-L) 达到 85.82 mIoU，超越 Mask2Former 2.52，超越 GSS 5.77
- **Cityscapes 语义分割**：54.90 mIoU，超越 SegFormer 4.82 mIoU
- **COCO 实例分割**：Swin-L backbone 下 52.7 AP，超越 Mask2Former 2.6 AP
- **VAR 优于所有替代生成模型**：在 ADE20K 上，VAR (85.8) >> SD-XL (81.8) >> DALL·E 2 (80.2) >> VQGAN (74.6)
- **Grid 和 Palette 参数鲁棒**：grid 12×12、palette size 215 为最优，但变化范围内性能稳定
- **参数效率**：在可比参数量下（R50 backbone），Seg-VAR (64.2 PQ) 仍优于 Mask2Former (63.8 PQ)

## 亮点与洞察

- **范式转换**：首次成功将分割从判别式并行预测转化为生成式顺序层次化预测，证明自回归方法在分割精度上可以超越乃至超过并行架构
- **空间感知颜色映射**：简单但有效的设计——利用 Transformer 位置编码预测实例颜色，巧妙地将实例区分问题转化为位置编码可解的颜色预测问题
- **Seglat 的 RGB 图像表示**：将分割掩码表示为 RGB 图像，使其可以利用已有的图像生成预训练模型
- **通用性**：同一架构在语义/实例/全景三种分割任务上都取得 SOTA

## 局限与展望

- **推理速度**：Seg-VAR (Swin-L) 仅 3.2 fps，远慢于 Mask2Former 的 4.0 fps，部分源于自回归解码的固有开销
- **内存消耗**：由于图像生成模型的记忆特性，内存成本高于 Transformer 分割模型
- **视频扩展**：尚未探索视频分割领域的应用
- 颜色映射方案在实例数量极大时可能受限于 215 种颜色上限

## 相关工作与启发

- 与 GSS（首个生成式分割方法）相比，Seg-VAR 的空间感知 seglat 编码解决了实例区分问题，全景 PQ 提升 14.8
- 与 Mask2Former（判别式 SOTA）相比，证明生成式方法在分割任务上具有竞争力甚至优势
- 为自回归模型从生成到感知的统一提供了可行路径，后续可探索将 Seg-VAR 与 LLM 结合实现多模态分割

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将分割重定义为自回归掩码生成，空间感知 seglat 设计独特
- **实验充分度**: ⭐⭐⭐⭐⭐ COCO/ADE20K/Cityscapes 三大数据集、三种分割任务全覆盖
- **写作质量**: ⭐⭐⭐⭐ 框架和公式推导清晰，但部分符号使用不够一致
- **价值**: ⭐⭐⭐⭐⭐ 开辟了自回归建模在分割中的新方向，SOTA 结果有力

<!-- RELATED:START -->

## 相关论文

- [ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](argenseg_image_segmentation_with_autoregressive_image_generation_model.md)
- [Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](../../CVPR2025/segmentation/uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)

<!-- RELATED:END -->
