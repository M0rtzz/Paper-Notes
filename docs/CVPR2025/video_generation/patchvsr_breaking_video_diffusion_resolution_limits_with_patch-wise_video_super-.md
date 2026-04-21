---
title: >-
  [论文解读] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution
description: >-
  [CVPR 2025][视频超分辨率] PatchVSR 首次将预训练视频扩散模型（T2V）用于 patch 级别的视频超分辨率，通过双分支适配器（局部 patch 分支 + 全局上下文分支）和无训练的多 patch 联合调制方案，基于 512×512 分辨率的基础模型实现了高保真的 4K 视频超分辨率，同时大幅提升计算效率。
tags:
  - CVPR 2025
  - 视频超分辨率
  - 扩散模型
  - Patch处理
  - 任意分辨率
  - 双分支适配器
---

# PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2509.26025](https://arxiv.org/abs/2509.26025)  
**代码**: 无  
**领域**: 扩散模型 / 视频超分辨率  
**关键词**: 视频超分辨率, 扩散模型, Patch处理, 任意分辨率, 双分支适配器

## 一句话总结

PatchVSR 首次将预训练视频扩散模型（T2V）用于 patch 级别的视频超分辨率，通过双分支适配器（局部 patch 分支 + 全局上下文分支）和无训练的多 patch 联合调制方案，基于 512×512 分辨率的基础模型实现了高保真的 4K 视频超分辨率，同时大幅提升计算效率。

## 研究背景与动机

**领域现状**：视频超分辨率（VSR）长期是计算机视觉的核心挑战。传统 CNN/Transformer 方法受限于模型容量和数据覆盖，难以生成逼真的细节和纹理。近年来，扩散生成模型因其强大的生成能力为 VSR 带来了新机遇，如 VEnhancer、Upscale-A-Video 等方法利用预训练扩散模型进行视频增强。

**现有痛点**：现有基于扩散模型的 VSR 方法都在全帧级别处理视频，继承了基础模型固定分辨率的限制。由于 Transformer 的全注意力特性，预训练模型通常只支持固定数量的 token（如 512×512 的各种宽高比），扩展到更高分辨率需要大量训练资源和高质量高分辨率数据集，目前都不可用。这导致现有方法无法支持任意分辨率的输出，推理效率低且显存需求大。

**核心矛盾**：超分辨率任务的特征注意力比生成任务更加局部化（因为有低分辨率参考，细节可以基于邻域语义生成，不需要全局一致性），但现有方法仍在全帧上进行昂贵的全注意力计算，浪费了大量资源。

**本文目标**：利用预训练视频扩散模型的生成先验进行 patch 级别的视频超分辨率，在不修改基础模型分辨率限制的情况下实现任意分辨率的高保真 VSR。

**切入角度**：超分辨率中的注意力更加局部化这一洞察使得 patch 处理成为可能——将输入视频切分为与预训练模型兼容尺寸的 patch，独立增强后拼接。但关键挑战在于预训练模型是在完整帧上训练的，patch 级生成会有明显性能下降。

**核心 idea**：通过双分支适配器为 patch 级生成注入局部内容保真性（patch 分支）和全局语义上下文（global 分支），使预训练的全帧 T2V 模型适配 patch 级别的细节生成。

## 方法详解

### 整体框架

给定低分辨率视频 $\mathbf{V}_l \in \mathbb{R}^{F \times H \times W \times 3}$ 和上采样因子 $k$，首先通过双三次插值上采样到目标分辨率，然后将上采样结果切分为 patch $\{\mathbf{P}_i \in \mathbb{R}^{F \times h \times w \times 3}\}_{i=1}^N$，其中 patch 尺寸 $(h, w)$ 与预训练 T2V 模型的生成尺寸匹配。同时将全视频缩放为 $\mathbf{G} \in \mathbb{R}^{F \times h \times w \times 3}$ 作为全局引导。每个 patch 通过双分支适配器增强后，使用多 patch 联合调制拼接为最终结果。

### 关键设计

1. **Patch 条件分支 (Patch Condition Branch)**:

    - 功能：从输入 patch 中提取特征，引导基础模型在保持内容保真的前提下合成细节
    - 核心思路：采用几个 Transformer block 组成的适配器，从输入 patch $\mathbf{P}_i$ 中提取特征并注入到基础模型每个 block 的输出。同时使用 LoRA 对基础模型进行微调，使其适应 patch 级别的数据分布（因为局部 patch 的数据分布与完整视频帧有明显差异）。text prompt 在 patch 分支中被替换为固定 prompt，避免 patch 与全局语义信息的不匹配
    - 设计动机：需要一个轻量级的方式将输入 patch 信息注入预训练模型，指导细节合成遵循输入内容；相比直接拼接到噪声空间，适配器方式更轻量且只需训练新增分支

2. **全局上下文分支 (Global Context Branch)**:

    - 功能：从完整输入视频中提取上下文信息，弥补 patch 语义不完整导致的生成差距
    - 核心思路：使用 Transformer 编码器将缩放后的全视频处理为四分之一 token 数量的上下文特征，通过新增的交叉注意力模块融入基础模型每个 block：$\{\mathbf{Q}, \mathbf{K}_g, \mathbf{V}_g\}$，其中 $\mathbf{Q}$ 与已有的文本交叉注意力共享。关键设计是在全局输入中拼接一个二值位置掩码 $\mathbf{M}_i$，指示目标 patch 在完整帧中的位置，使上下文引导更具针对性
    - 设计动机：预训练模型是在完整帧上训练的，patch 往往包含不完整或模糊的语义（如只有一部分物体）；全局上下文可以帮助模型理解 patch 的整体语境，生成更自然的细节

3. **多 Patch 联合调制 (Multi-Patch Joint Modulation)**:

    - 功能：确保独立增强的 patch 拼接后保持视觉一致性，消除边界伪影
    - 核心思路：基于 MultiDiffusion 的思想进行改进——先将视频切分为不重叠的 patch，然后在相邻 patch 的交界处创建辅助 patch（组合相邻 patch 各一半），形成 50% 重叠率的新 patch 组。在每个去噪步骤中，对重叠区域进行加权融合。不同于简单取平均（会导致黑洞或接缝），采用空间权重图，辅助 patch 的影响从中心分割线向两侧逐渐减弱
    - 设计动机：由于生成过程的随机性，在退化或不确定区域可能产生多种合理的高分辨率解，导致相邻 patch 在边界处纹理不一致

### 损失函数 / 训练策略

- 使用 Rectified Flow 框架，前向过程定义为数据分布和标准高斯分布之间的直线路径 $z_t = (1-t)z_0 + t\epsilon$
- 训练数据为 460K 自采集高质量视频-文本对，分辨率 1024×1024 到 2K
- 降质处理：对 HR 视频先双线性下采样再上采样回原始分辨率，随机裁剪 512×512 区域
- 噪声增强：向输入 latent 注入噪声（时间步 200-300），保持结构
- 下采样因子、裁剪位置、噪声时间步均作为条件编码

## 实验关键数据

### 主实验

SynVideo30 数据集 ×4 超分 (2K)：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | Aesthetics↑ |
|------|-------|-------|--------|--------|-------------|
| RealBasicVSR | 33.507 | 0.776 | 0.185 | 49.557 | 0.496 |
| Upscale-A-Video | 33.432 | 0.728 | 0.205 | 49.839 | 0.494 |
| VEnhancer | 28.856 | 0.697 | 0.199 | 43.538 | 0.503 |
| **PatchVSR** | 30.857 | 0.732 | **0.183** | **50.695** | **0.520** |

计算效率 (2K 视频)：

| 方法 | 时间(s)↓ | 显存(G)↓ |
|------|---------|---------|
| LaVie-SR | 2261 | 68 |
| Upscale-A-Video | 2743 | 47 |
| VEnhancer | 1562 | 62 |
| **PatchVSR** | **680** | **40** |

### 消融实验

VideoGen30 各组件消融：

| 组件 | DOVER↑ | MUSIQ↑ | Aesthetics↑ |
|------|--------|--------|-------------|
| w/o global branch | 0.502 | 46.074 | 0.589 |
| w/o LoRA | 0.582 | 50.496 | 0.600 |
| w/o location embed | 0.574 | 50.084 | 0.601 |
| w/o fixed prompt | 0.562 | 48.133 | 0.597 |
| Full PatchVSR | **0.590** | **50.559** | **0.602** |

全局分支的移除导致最显著的性能下降（DOVER 从 0.590 降至 0.502）。

### 关键发现

- PatchVSR 在 LPIPS（感知保真度）上取得最优，但 PSNR/SSIM 不是最高——因为生成式方法倾向于产生丰富的细节而非像素级对齐
- patch 处理策略使自注意力在 patch 内部进行，计算复杂度从全帧的 $n^2$ 降为 $n \times m$（$m$ 为 patch token 数），效率提升 2-4 倍
- 全局分支中的位置嵌入对补齐 patch 语义至关重要
- 模型对语义不完整的 patch（如仅包含物体局部）表现出良好的鲁棒性

## 亮点与洞察

- 首个将预训练视频扩散模型用于 patch 级 VSR 的工作，优雅地解决了固定分辨率限制
- 效率优势显著：基于 512×512 模型实现 4K 输出，推理时间和显存消耗大幅降低
- 关键洞察——VSR 的注意力比生成任务更局部化——为 patch 处理提供了合理的理论支撑
- 多 patch 联合调制的设计（辅助 patch + 空间权重图）有效解决了拼接伪影问题

## 局限与展望

- 对真实世界低分辨率图像性能下降明显，因为训练数据缺乏专门的退化增强
- 继承了基础模型的迭代推理方案，虽然已比全帧方法高效，但仍有优化空间
- 不同上采样因子的效果展示了灵活性，但极端因子下细节生成的可控性有待探索
- 未来可以结合步数蒸馏等技术进一步压缩推理时间

## 相关工作与启发

- MultiDiffusion 的多区域联合采样思想被本文成功改进并应用于视频超分
- 与 VEnhancer、Upscale-A-Video 等全帧方法形成互补——它们追求全局一致性，PatchVSR 追求高效率和分辨率灵活性
- LoRA 微调在适配 patch 分布上的有效性为类似的迁移学习场景提供参考
- 双分支适配器范式（局部+全局）可推广到其他需要多尺度引导的生成任务

## 评分

- **新颖性**: 8/10 — 首次探索 patch 级视频扩散超分，方向新颖
- **实验充分度**: 8/10 — 多数据集多指标对比充分，消融实验详尽
- **写作质量**: 8/10 — 动机阐述清晰，方法描述详细
- **价值**: 9/10 — 实用价值高，4K VSR 的效率优势对产业应用有直接推动

<!-- RELATED:START -->

## 相关论文

- [VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)
- [VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)
- [Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)

<!-- RELATED:END -->
