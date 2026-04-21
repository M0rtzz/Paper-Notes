---
title: >-
  [论文解读] LaVin-DiT: Large Vision Diffusion Transformer
description: >-
  [CVPR 2025][图像生成][大视觉模型] LaVin-DiT 提出一种基于扩散 Transformer 的大视觉基础模型，通过空间-时序 VAE 编码、联合扩散 Transformer 去噪、以及 in-context learning 实现超过 20 种视觉任务的统一处理，从 0.1B 扩展至 3.4B 参数，在多项任务上显著超越自回归式大视觉模型 LVM。
tags:
  - CVPR 2025
  - 图像生成
  - 大视觉模型
  - Transformer
  - In-Context Learning
  - 统一视觉框架
  - 多任务学习
---

# LaVin-DiT: Large Vision Diffusion Transformer

**会议**: CVPR 2025  
**arXiv**: [2411.11505](https://arxiv.org/abs/2411.11505)  
**代码**: https://derrickwang005.github.io/LaVin-DiT/ (有)  
**领域**: 扩散模型 / 多模态VLM  
**关键词**: 大视觉模型, 扩散Transformer, In-Context Learning, 统一视觉框架, 多任务学习

## 一句话总结
LaVin-DiT 提出一种基于扩散 Transformer 的大视觉基础模型，通过空间-时序 VAE 编码、联合扩散 Transformer 去噪、以及 in-context learning 实现超过 20 种视觉任务的统一处理，从 0.1B 扩展至 3.4B 参数，在多项任务上显著超越自回归式大视觉模型 LVM。

## 研究背景与动机

1. **领域现状**：大语言模型（LLM）已能在统一框架内处理多种语言任务。计算机视觉领域也在追求类似的"大视觉模型"（LVM），目前主要有两条路径：图像化生成方法（如 Painter 将密集预测重构为图像修复）和序列建模方法（如 LVM 将视觉数据量化为离散 token 做 next-token prediction）。

2. **现有痛点**：序列建模方法直接从 NLP 架构迁移过来，面临两大问题——(a) 自回归的逐 token 生成对高维视觉数据计算效率低下；(b) 将 2D/3D 视觉数据展平为 1D 序列时破坏了空间关系（如图 1 中红色和蓝色 token 在空间上相邻但序列上相隔很远）。

3. **核心矛盾**：如何在保持空间结构完整的同时，实现高效的多任务统一生成？

4. **本文目标** 设计一个基于扩散模型的大视觉基础模型，避免自回归的效率和空间破坏问题，同时支持 20+ 种图像和视频任务。

5. **切入角度**：扩散模型天然并行去噪所有 token，不破坏空间结构；结合 in-context learning，用输入-目标对作为任务定义，无需任务特定头或微调。

6. **核心 idea**：用 ST-VAE 压缩视觉数据到连续潜在空间，用联合扩散 Transformer（J-DiT）并行去噪，通过 in-context learning 定义和适配多种视觉任务。

## 方法详解

### 整体框架
LaVin-DiT 由三个核心组件构成：(1) **ST-VAE**（空间-时序变分自编码器）将图像/视频从像素空间压缩到紧凑的潜在空间（$4 \times 8 \times 8$ 压缩率）；(2) **J-DiT**（联合扩散 Transformer）在潜在空间中执行条件去噪——将 in-context 的输入-目标对和查询图像作为条件（clean），将目标加噪后迭代去噪；(3) **推理时**随机采样任务相关的输入-目标对作为上下文，和测试查询一起送入 J-DiT 生成预测。

### 关键设计

1. **空间-时序变分自编码器（ST-VAE）**:

    - 功能：将高维图像/视频数据压缩到低维连续潜在空间
    - 核心思路：使用因果 3D 卷积和反卷积，4 个对称阶段交替进行 $2\times$ 下采样/上采样。前两个阶段同时处理空间和时间维度，最后一个阶段仅处理空间维度，总压缩率 $4 \times 8 \times 8$。为防止未来信息泄漏，在时间卷积的起始位置做 padding。视频第一帧独立压缩（仅空间），后续帧同时压缩空间和时间。训练分两阶段：先纯图像，再图像+视频联合，使用 MSE + 感知损失 + 对抗损失。
    - 设计动机：直接在像素空间做扩散计算成本太高。ST-VAE 的因果设计确保了时序一致性，$4 \times 8 \times 8$ 的高压缩率大幅降低后续 J-DiT 的计算量。

2. **联合扩散 Transformer（J-DiT）**:

    - 功能：在潜在空间中执行条件扩散生成，核心是全序列联合注意力（full-sequence joint attention）
    - 核心思路：基于 MM-DiT 架构，为条件序列和目标序列分别构建 patch embedding（patch 大小 $2 \times 2$），并引入条件/目标各自的自适应 RMS 归一化（AdaRN），通过不同的时间步嵌入独立调制两个表示空间。**全序列联合注意力**将条件和目标序列线性投影、拼接后执行双向注意力，使每个序列在自身空间中运算同时感知对方，实现任务特定对齐。使用 grouped-query attention 替代 multi-head attention 减少参数，加入 QK-Norm 稳定长序列训练，sandwich normalization 维持残差连接中的激活幅度。位置编码采用 **3D RoPE** 统一表示空间-时序关系。
    - 设计动机：条件（clean）和目标（noisy）的值域不同，需要独立的嵌入和归一化；全序列联合注意力比分离注意力更好地对齐任务信息；3D RoPE 比 1D 位置编码更准确地捕获视觉数据的空间-时序位置关系。

3. **In-Context Learning 多任务统一**:

    - 功能：通过提供输入-目标对定义任务，推理时无需微调即可适配不同任务
    - 核心思路：训练时从任务数据中采样一组输入-目标对作为任务上下文，与查询拼接送入 J-DiT。目标加噪后通过 flow matching 训练（条件流匹配损失 $\ell_{\text{CFM}} = \int_0^1 \mathbb{E}[|v_\theta(z_t, t) - (z_0 - z_1)|_2^2] \text{d}t$）。推理时用 Euler 方法从噪声逆向积分 $N=20$ 步。增加上下文长度（更多输入-目标对）能持续提升下游任务性能。
    - 设计动机：避免为每个任务设计专门的解码头或微调策略，实现真正的统一框架。上下文对的信息丰富程度直接影响任务理解质量。

### 损失函数 / 训练策略
- J-DiT 使用条件流匹配（CFM）损失在潜在空间训练
- 两阶段训练：先 256×256 分辨率 100K 步，再 512×512 分辨率 20K 步
- AdamW，lr=1e-4，batch=640，64×A100-80G GPU
- 数据集：约 320 万图像 + 60 万视频，覆盖 20+ 任务

## 实验关键数据

### 主实验

| 任务 | 指标 | LaVin-DiT (3.4B) | LVM (7B) | 提升 |
|------|------|-----------------|----------|------|
| 前景分割 Split 1 (unseen) | mIoU ↑ | **67.87** | 48.94 | +18.93 |
| 单目标检测 Split 4 (unseen) | mIoU ↑ | **68.88** | 48.92 | +19.96 |
| NYU-v2 深度估计 | AbsRel ↓ | **6.2** | 30.2 | -24.0 |
| NYU-v2 法线估计 | MAE ↓ | **15.901** | 23.433 | -7.5 |
| ImageNet 修复 | FID ↓ | **1.65** | 4.05 | -2.40 |
| 着色 | MSE ↓ | **0.24** | 0.51 | -0.27 |

在未见过的任务（前景分割、目标检测）上也显著优于 LVM，展现强泛化能力。

### 可扩展性分析

| 模型规模 | 着色 MSE ↓ | 深度 AbsRel ↓ |
|---------|-----------|--------------|
| 0.1B | 0.609 | 7.6 |
| 1.0B | 0.311 | 6.5 |
| 3.4B | **0.273** | **6.2** |

### 关键发现
- LaVin-DiT 3.4B 收敛更快且训练损失更低，体现出明显的 scaling law
- 推理速度比 LVM 快 1.7~2.3 倍（256×256 时 4.67s vs 8.1s，512×512 时 20.1s vs 47.2s），扩散模型的并行去噪优势明显
- 增加 in-context 对数量持续提升性能（如深度到图像的 FID 和去运动模糊的 PSNR 均持续改善）
- 深度估计 AbsRel 6.2 接近专家模型 Marigold（6.0），法线估计 MAE 15.901 甚至超越专家模型 StableNormal（19.707）
- 可处理视频任务（帧预测、视频深度/法线/光流估计、视频实例分割等），生成 12 帧后续预测

## 亮点与洞察
- **扩散模型 vs 自回归模型用于大视觉模型的系统对比**非常有说服力：扩散模型的并行去噪天然保留空间结构且推理更快，而自回归的逐 token 生成既慢又破坏空间关系
- **J-DiT 中条件/目标分离嵌入 + 独立 AdaRN** 的设计精巧地处理了 clean 和 noisy 表示的值域差异问题
- **统一的 in-context learning 范式**使得同一模型无需微调就能处理图像/视频的 20+ 种任务，且上下文越长效果越好，展示了类似 LLM 的 few-shot 涌现能力
- 3D RoPE 比 1D 位置编码更自然地表达视觉数据的空间-时序结构，可推广到其他视频生成/理解模型

## 局限与展望
- 模型的泛化依赖于训练任务分布，当任务定义显著偏离训练分布时泛化困难
- 训练数据规模和多样性仍远不及 LLM，未来需要更大规模的视觉多任务数据集
- 深度和法线的伪标签来自 Depth-anything V2 和 StableNormal，上限受制于这些模型的精度
- 未探索自动选择最优任务上下文（context selection）的方法
- 3.4B 模型在 64×A100 上训练，计算门槛仍然较高

## 相关工作与启发
- **vs LVM (Bai et al.)**: LVM 用自回归序列建模统一视觉任务，但计算慢且破坏空间关系。LaVin-DiT 用扩散 Transformer 替代，在几乎所有任务上显著优于 LVM-7B，且推理快 2 倍
- **vs Painter/PromptDiffusion**: 这些方法通过图像修复/视觉提示实现多任务，但依赖预训练扩散模型的先验。LaVin-DiT 从头训练统一模型，灵活性和可扩展性更强
- **vs DiT/SD3**: LaVin-DiT 将扩散 Transformer 从单一生成任务扩展到 20+ 种理解+生成任务，是 DiT 架构在通用视觉方向的重要探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将扩散 Transformer 扩展为统一的大视觉基础模型，系统性地论证了扩散优于自回归的路径
- 实验充分度: ⭐⭐⭐⭐⭐ 20+ 任务评估、可扩展性分析、推理延迟对比、上下文长度研究，极其全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，自回归 vs 扩散的对比图直观有力
- 价值: ⭐⭐⭐⭐⭐ 为大视觉模型开辟了扩散 Transformer 的新路径，影响力潜力大

<!-- RELATED:START -->

## 相关论文

- [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](../../NeurIPS2025/image_generation/linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](../../NeurIPS2025/image_generation/in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)

<!-- RELATED:END -->
