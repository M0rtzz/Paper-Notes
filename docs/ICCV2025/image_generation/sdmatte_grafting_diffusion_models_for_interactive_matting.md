---
title: >-
  [论文解读] SDMatte: Grafting Diffusion Models for Interactive Matting
description: >-
  [ICCV 2025][图像生成][交互式抠图] 本文提出 SDMatte，基于 Stable Diffusion 的交互式抠图模型，通过视觉提示驱动交叉注意力、坐标/不透明度嵌入和掩码自注意力三项设计，将扩散模型的文本交互能力转化为视觉提示交互能力，在多个数据集上显著超越 SAM-based 方法。
tags:
  - ICCV 2025
  - 图像生成
  - 交互式抠图
  - 扩散模型先验
  - 视觉提示
  - Alpha Matte
  - 注意力机制
---

# SDMatte: Grafting Diffusion Models for Interactive Matting

**会议**: ICCV 2025  
**arXiv**: [2508.00443](https://arxiv.org/abs/2508.00443)  
**代码**: https://github.com/vivoCameraResearch/SDMatte  
**领域**: 扩散模型 / 图像抠图  
**关键词**: 交互式抠图, 扩散模型先验, 视觉提示, Alpha Matte, 注意力机制

## 一句话总结

本文提出 SDMatte，基于 Stable Diffusion 的交互式抠图模型，通过视觉提示驱动交叉注意力、坐标/不透明度嵌入和掩码自注意力三项设计，将扩散模型的文本交互能力转化为视觉提示交互能力，在多个数据集上显著超越 SAM-based 方法。

## 研究背景与动机

**传统方法代价高**：Trimap-based 抠图精度高但标注成本大；自动抠图方法对非显著/透明物体效果差。交互式抠图（点、框、掩码提示）是平衡易用性和精度的理想方案。

**SAM-based 方法的局限**：MAM、MatAny、SEMat 等方法依赖冻结的 SAM 生成粗掩码再细化，但无法纠正 SAM 的错误，导致误差在后续模块中放大。

**扩散模型的潜力**：扩散模型在数十亿图文对上训练，具有出色的泛化能力和细节保持能力（如 Marigold 仅在合成数据上微调就实现了优秀的深度估计）。但现有方法通常用空文本嵌入微调，浪费了强大的文本交互能力。

**SDMatte 的核心思路**：不丢弃扩散模型的交互能力，而是将文本驱动交互转化为视觉提示驱动交互。

## 方法详解

### 整体框架

基于 Stable Diffusion v2，采用单步确定性范式（类似 GenPercept）：
1. VAE 编码器将输入图像和视觉提示映射到潜在空间
2. 拼接后输入 U-Net（首层卷积权重翻倍以适应增加的通道数）
3. VAE 解码器将输出映射回像素空间计算 matting loss

### 关键设计一：视觉提示交叉注意力

将 U-Net 中间块（语义信息最集中）的文本嵌入替换为视觉提示嵌入：
- 对视觉提示的潜在表示应用零卷积层，映射到与文本嵌入相同维度
- 零卷积层初始化确保训练初期不破坏原模型，渐进式地将文本交互能力转化为视觉提示交互能力
- 注意力图可视化表明模型准确聚焦于视觉提示指示的区域

### 关键设计二：坐标嵌入和不透明度嵌入

受 SDXL 将图像尺寸/裁剪坐标作为 U-Net 条件的启发：

**坐标嵌入**：
- Box 提示：对左上和右下角坐标的 4 个数值分别进行正弦位置编码，得到 $\mathbf{E}_{box} \in \mathbb{R}^{B \times 1280}$
- Point 提示：将 $2N$ 个坐标值补齐后统一编码，得到 $\mathbf{E}_{point} \in \mathbb{R}^{B \times 1680}$
- Mask 提示：计算最小包围框后按 box 方式编码

**不透明度嵌入**：对物体透明度信息（透明物体=0，非透明=1）进行正弦编码

最终条件嵌入替代原时间嵌入：$\mathbf{E}_{cond} = f_1(\mathbf{E}_{opacity}) + f_2(\mathbf{E}_{coord})$

### 关键设计三：掩码自注意力

受 Mask2Former 启发，明确引导模型关注视觉提示指示区域：

$$\mathbf{X} = \text{softmax}\left(\mathbf{M} + \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{M}$ 为注意力掩码：
- Box/Mask 提示：生成硬二值掩码（指示区域=1，其余=0）
- Point 提示：生成以点坐标为中心的高斯软掩码
- 非指示区域的 $(\mathbf{M}-1) \times \infty$ 有效抑制了注意力

## 实验

### AIM-500 和 AM-2K 数据集对比

| 方法 | 骨干 | 提示 | MSE↓ | MAD↓ | SAD↓ | Grad↓ |
|------|------|------|------|------|------|-------|
| MAM | SAM | point | 0.0752 | 0.1080 | 186.50 | 37.48 |
| MatAny | SAM | point | 0.0425 | 0.0523 | 87.05 | 33.44 |
| SmartMatting | DINOv2 | point | 0.0302 | 0.0388 | 66.27 | 46.63 |
| **SDMatte** | **SD2** | **point** | **0.0109** | **0.0189** | **31.80** | **26.84** |
| MAM | SAM | box | 0.0116 | 0.0222 | 36.66 | 21.04 |
| SmartMatting | DINOv2 | box | 0.0077 | 0.0151 | 25.33 | 27.16 |
| **SDMatte** | **SD2** | **box** | 最优 | 最优 | 最优 | 最优 |

### 消融实验

| 配置 | MSE | SAD |
|------|-----|-----|
| LiteSDMatte (无掩码自注意力等) | 0.0115 | 34.43 |
| + 视觉提示交叉注意力 | 改善 | 改善 |
| + 坐标/不透明度嵌入 | 进一步改善 | 进一步改善 |
| + 掩码自注意力 (完整 SDMatte) | **0.0109** | **31.80** |

### 关键发现

- SDMatte 在 point 提示下 MSE 降至 SmartMatting 的 36%（0.0109 vs 0.0302），证明扩散先验的强大
- 在 box 提示下超越包括 SEMat (SAM2) 在内的所有方法
- 视觉提示交叉注意力有效继承了文本交互能力——注意力图精确聚焦于目标区域
- 坐标嵌入和不透明度嵌入对透明物体抠图提升尤为显著

## 亮点与洞察

1. **范式创新**：将扩散模型的文本交互能力转化为视觉提示交互，而非简单丢弃
2. **透明物体处理**：不透明度嵌入是针对抠图任务的独特设计
3. **方法可扩展性强**：支持点、框、掩码三种提示类型

## 局限性

- 单步确定性范式虽然高效但丢失了扩散模型的随机性优势
- VAE 编码/解码引入的信息损失可能影响精细边缘
- 依赖 SD2 预训练权重，对新扩散架构（DiT）的迁移性未验证

## 相关工作

- **交互式抠图**: MAM, MatAny, SmartMatting, SEMat
- **扩散模型视觉感知**: Marigold, GenPercept, DiffDIS
- **Trimap-based**: DIM, IndexNet

## 评分

- 新颖性：⭐⭐⭐⭐ — 文本→视觉提示交互转化思路巧妙
- 技术深度：⭐⭐⭐⭐ — 三个组件设计合理互补
- 实验充分度：⭐⭐⭐⭐ — 多数据集、多提示类型全面验证
- 实用价值：⭐⭐⭐⭐ — 边缘细节保持能力强，适合工业应用

<!-- RELATED:START -->

## 相关论文

- [GameFactory: Creating New Games with Generative Interactive Videos](gamefactory_creating_new_games_with_generative_interactive_videos.md)
- [StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](../../ECCV2024/image_generation/lazy_diffusion_transformer_for_interactive_image_editing.md)
- [SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](../../CVPR2025/image_generation/semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)

<!-- RELATED:END -->
