---
title: >-
  [论文解读] FlexTok: Resampling Images into 1D Token Sequences of Flexible Length
description: >-
  [ICML 2025][图像生成][可变长度 Tokenizer] 提出 FlexTok——一种将 2D 图像重采样为可变长度、有序的 1D 离散 token 序列的 tokenizer，通过 nested dropout 学习层次化编码，配合 rectified flow 解码器在任意 token 数量下生成高质量重建，在 ImageNet 上用 8~128 个 token 即可实现 FID<2 的自回归图像生成。
tags:
  - ICML 2025
  - 图像生成
  - 可变长度 Tokenizer
  - 1D Token 序列
  - Nested Dropout
  - Rectified Flow 解码器
  - 粗到细生成
---

# FlexTok: Resampling Images into 1D Token Sequences of Flexible Length

**会议**: ICML 2025  
**arXiv**: [2502.13967](https://arxiv.org/abs/2502.13967)  
**代码**: https://flextok.epfl.ch/ (项目页面)  
**领域**: 图像 Tokenizer / 自回归图像生成  
**关键词**: 可变长度 Tokenizer, 1D Token 序列, Nested Dropout, Rectified Flow 解码器, 粗到细生成

## 一句话总结
提出 FlexTok——一种将 2D 图像重采样为可变长度、有序的 1D 离散 token 序列的 tokenizer，通过 nested dropout 学习层次化编码，配合 rectified flow 解码器在任意 token 数量下生成高质量重建，在 ImageNet 上用 8~128 个 token 即可实现 FID<2 的自回归图像生成。

## 研究背景与动机

**领域现状**：自回归（AR）图像生成已成为与扩散模型并驾齐驱的主流范式。核心技术是图像 tokenizer：将图像编码为离散 token 序列，然后用 GPT 式 Transformer 预测。

**现有痛点**：
   - **2D 网格 tokenizer 的冗余**：传统方法（VQGAN、LlamaGen）将图像编码为 2D token 网格（如 16×16=256 个 token），但很多 token 承载的信息高度冗余（如背景区域）
   - **TiTok 的固定长度限制**：TiTok 证明了 1D tokenizer 的可行性，但对每个压缩率需要训练不同模型，且 token 数量固定
   - **无法适应图像复杂度**：简单图像（纯色背景上的苹果）和复杂图像（拥挤街景）使用相同数量的 token，效率低下

**核心矛盾**：固定 token 数量与图像复杂度的可变性之间的矛盾——简单图像浪费 token，复杂图像 token 不足。

**本文目标**：设计一个单一模型，能将图像编码为 1 到 256 个 token 的任意长度序列，且在所有长度下都能生成合理的重建。

**切入角度**：用 nested dropout 强制编码器按重要性排序 token（从语义到细节），用 rectified flow 解码器确保任意 token 数量下都能生成高质量输出。

**核心 idea**：FlexTok 将图像压缩为有序的 1D token 序列，形成一种"视觉词汇表"——少量 token 描述粗略语义，更多 token 逐步添加细节。

## 方法详解

### 整体框架

三阶段 pipeline：
- **Stage 0**：训练 VAE（类似 SDXL VAE）将图像压缩为连续 2D latent 网格（8× 下采样）
- **Stage 1**：FlexTok tokenizer 将 2D VAE latent 重采样为 1D 离散 token 序列
- **Stage 2**：训练自回归 Transformer 生成 token 序列，再用 FlexTok 解码器重建图像

### 关键设计

1. **ViT 编码器 + Register Token 瓶颈**:

    - 编码器是 Vision Transformer，输入 2D VAE latent patches
    - 使用 256 个 **register tokens** 作为 1D 瓶颈表示
    - 对 register tokens 做 **Finite Scalar Quantization (FSQ)**，levels=[8,8,8,5,5,5]，有效 codebook 大小 64000
    - 编码器和解码器使用 2×2 patchification，结合 VAE 的 8× 下采样实现总共 16× 下采样
    - **设计动机**：Register tokens 机制来自 ViT 研究，天然适合做 1D 信息瓶颈；FSQ 比 VQ 更稳定且无需 codebook collapse 的担忧

2. **Nested Dropout 实现有序编码**:

    - 训练时对 register tokens 做 nested dropout：随机保留前 $k$ 个 token（$k$ 从 1 到 256 均匀采样），丢弃后面的 token
    - 这迫使编码器将最重要的信息放在靠前的 token 中
    - 前几个 token 编码高层语义（如"金毛犬"），后面的 token 逐步补充细节（如毛发纹理、背景）
    - **为什么这样设计**：Nested dropout 是实现 token 排序的最简洁方式——无需显式定义"重要性"，模型自动学习把全局语义前置

3. **Rectified Flow 解码器**:

    - 解码器不是简单的 deterministic decoder，而是一个 **rectified flow 模型**
    - 输入：噪声化的 VAE latent patches + （被随机 mask 的）register tokens
    - 预测：从噪声到干净 latent 的 flow
    - 使用 AdaLN-zero 分别对 patches 和 registers 做时间步条件化
    - 额外应用 **REPA 归纳偏置损失**（使用 DINOv2-L）加速收敛
    - **为什么用生成式解码器**：确定性解码器在极少 token（如 1-8 个）时只能输出模糊的平均图像；rectified flow 可以"想象"缺失细节，在任何 token 数量下都生成清晰且合理的图像

### 损失函数 / 训练策略

- **Rectified Flow 目标**：标准的 flow matching 损失
- **REPA 损失**：中间解码器特征与 DINOv2-L 特征的对齐损失，加速语义学习
- **模型规模**：编码器-解码器 depth 配置 d12-d12、d18-d18、d18-d28，width=64d
- 训练分辨率 256×256，在 ImageNet-1k 和 DFN-2B 上训练

## 实验关键数据

### 主实验：ImageNet 类条件图像生成（1.3B AR 模型）

| 方法 | Token 数 | gFID ↓ | 特点 |
|------|---------|--------|------|
| LlamaGen (2D grid) | 256 | ~2.2 | 固定 256 token，光栅扫描 |
| TiTok-S-128 | 128 | 1.97 | 固定 128 token |
| TiTok-L-32 | 32 | 2.77 | 固定 32 token，另一个模型 |
| **FlexTok d18-d28** | **8** | **<2** | 单一模型，8 token |
| **FlexTok d18-d28** | **32** | **<2** | 单一模型，32 token |
| **FlexTok d18-d28** | **128** | **<2** | 单一模型，128 token |

FlexTok 在 8-128 token 范围内均实现 FID<2，且是**单一模型**。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 nested dropout | Token 无序，少 token 时质量差 | 无法实现可变长度 |
| 确定性解码器 | 少 token 时极度模糊 | 缺乏生成能力补偿缺失信息 |
| 无 REPA 损失 | 收敛慢 2-3× | DINOv2 归纳偏置加速语义学习 |
| 小模型 d12-d12 | rFID 较高 | 模型规模影响重建 FID，但不太影响 MAE |
| 大模型 d18-d28 | rFID 显著改善 | 更强的生成式解码器最关键 |

### 关键发现
- **粗到细的视觉词汇表**：前几个 token 捕获高层语义（类别、构图），后续 token 添加细节（纹理、颜色）
- **条件复杂度决定 token 数**：ImageNet 类标签用 16-32 个 token 即可满足；开放式文本 prompt 需要 256 个 token
- **AR 模型规模的影响**：少 token（1-8）时模型规模无关紧要（粗语义容易学），多 token（128+）时大模型显著更好
- **与 2D tokenizer 比较**：在 256 token 上限时 FlexTok 与 2D 方法持平，但灵活性远超——可以根据任务需求减少 token 数

## 亮点与洞察
- **范式创新**：从"固定长度光栅扫描"到"可变长度粗到细生成"，改变了 AR 图像生成的思路
- **单一模型多分辨率**：一个 tokenizer 适配所有压缩率，部署和训练成本大幅降低
- **Rectified Flow 解码器的妙用**：将不确定性优雅地转化为生成多样性，避免了模糊重建问题
- **64000 种"图像原型"**：第一个 token 本质上将所有可能图像分成 64000 个语义簇

## 局限与展望
- 目前仅支持 256×256 分辨率，高分辨率（512/1024）需要进一步验证
- Rectified flow 解码器引入额外推理成本（需要多步采样），比确定性解码器慢
- Token 排序依赖 nested dropout 的随机性，可能不是最优的信息排序策略
- 尚未探索视频、音频等多模态扩展

## 相关工作与启发
- **TiTok**：1D tokenizer 的先驱，但固定长度。FlexTok 可视为 TiTok 的灵活长度推广
- **ElasticTok/ALIT/One-D-Piece**：同期并行工作，探索类似的可变长度思路
- **REPA**：representation alignment 技术，显著加速 tokenizer 训练
- **启发**：可变长度表示的思想可推广到视频（时间维度的冗余更多）和 3D 生成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 可变长度 1D tokenizer + rectified flow 解码器的组合优雅且实用
- 实验充分度: ⭐⭐⭐⭐ 类条件和文本条件生成均有验证，scaling 分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 项目页面极好，可视化丰富，概念解释清晰
- 价值: ⭐⭐⭐⭐⭐ 对 AR 图像生成领域有重要推进，开启自适应压缩的新方向

<!-- RELATED:START -->

## 相关论文

- [Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](../../NeurIPS2025/image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)
- [ToMA: Token Merge with Attention for Diffusion Models](toma_token_merge_with_attention_for_diffusion_models.md)
- [DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](../../ICCV2025/image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)

<!-- RELATED:END -->
