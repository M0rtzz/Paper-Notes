---
title: >-
  [论文解读] RefTon: Reference Person Shot Assist Virtual Try-on
description: >-
  [CVPR 2026][人体理解][虚拟试穿] 本文提出 RefTon，一个基于 Flux-Kontext 的人对人虚拟试穿框架，通过引入额外参考图像（其他人穿着目标服装的照片）来提供更准确的服装细节信息，同时通过两阶段训练策略和缩放位置索引机制实现了无需辅助条件（如 DensePose、分割掩码）的端到端试穿，在 VITON-HD 和 DressCode 上达到 SOTA。
tags:
  - CVPR 2026
  - 人体理解
  - 虚拟试穿
  - 参考图像引导
  - Flux-Kontext
  - 无遮罩试穿
  - 扩散模型
---

# RefTon: Reference Person Shot Assist Virtual Try-on

**会议**: CVPR 2026  
**arXiv**: [2511.00956](https://arxiv.org/abs/2511.00956)  
**代码**: https://github.com/360CVGroup/RefTon  
**领域**: 人体理解  
**关键词**: 虚拟试穿, 参考图像引导, Flux-Kontext, 无遮罩试穿, 扩散模型

## 一句话总结
本文提出 RefTon，一个基于 Flux-Kontext 的人对人虚拟试穿框架，通过引入额外参考图像（其他人穿着目标服装的照片）来提供更准确的服装细节信息，同时通过两阶段训练策略和缩放位置索引机制实现了无需辅助条件（如 DensePose、分割掩码）的端到端试穿，在 VITON-HD 和 DressCode 上达到 SOTA。

## 研究背景与动机
1. **领域现状**：虚拟试穿（ViTON）已从 GAN 方法发展到基于扩散模型的方法，后者在服装变形和纹理保真度上有显著进步。
2. **现有痛点**：(a) 大量方法依赖复杂外部模型——姿态估计器、人体解析、分割模型等来处理不同条件输入，增加了框架复杂度且掩码质量直接影响最终结果。(b) 更关键的是，仅从"衣服平铺图"无法完整感知服装的样式、纹理和设计细节——比如无法判断一件衣服是绿色透明面料还是浅绿色不透明面料，无法识别蕾丝领口设计。
3. **核心矛盾**：在真实购物场景中，用户更关注模特穿着效果而非平铺衣物图。但现有方法不支持"参考模特图"作为额外输入，因为公开数据集中缺乏这类配对数据。
4. **本文目标** (a) 去除对外部模型和辅助条件的依赖 (b) 引入参考图像来更准确地传达服装的穿着效果 (c) 构建包含参考图像的训练数据。
5. **切入角度**：利用 Flux-Kontext 强大的图像编辑能力，自动合成不同人穿同一件衣服的参考图像，构建训练数据集。同时改进位置编码以支持多条件多分辨率输入。
6. **核心 idea**：通过参考图像（其他人穿着目标服装的照片）为虚拟试穿提供更直观的视觉指导，配合无掩码两阶段训练和缩放位置索引，实现简洁高效的端到端试穿。

## 方法详解

### 整体框架
RefTon 基于 Flux-Kontext 骨干架构。输入包括源人物图像（或遮罩图）、目标服装图像、可选的参考图像。图像经 VAE 编码为潜变量后拼接成序列，通过 DiT（Diffusion Transformer）去噪生成目标图像。训练分两个阶段：第一阶段训练基于掩码的试穿模型（合成非配对数据），第二阶段训练人对人无掩码模型。

### 关键设计

1. **两阶段训练策略**:

    - 功能：在仅有配对数据的情况下实现人对人直接试穿
    - 核心思路：现有数据集只提供 $[\mathbf{c}_i, \mathbf{p}_{i,\mathbf{c}_i}]$（衣服+穿该衣服的人），但训练人对人模型需要非配对三元组 $[\bar{\mathbf{p}}_{i,\mathbf{c}_j}, \mathbf{c}_i, \mathbf{p}_{i,\mathbf{c}_i}]$（穿其他衣服的人+目标衣服+穿目标衣服的结果）。阶段一：用 agnostic 图像、DensePose、warp mask 等丰富条件训练掩码试穿模型，然后用它给每个人合成穿不同衣服的图像。阶段二：用合成的非配对图像训练人对人模型，以 50% 概率随机使用 agnostic 图像或合成人物图像作为输入。
    - 设计动机：类似 CatVTON 的策略但进一步丰富了条件输入以提高合成图像质量。关键在于阶段一生成足够高质量的非配对数据供阶段二训练。

2. **缩放位置索引（Rescaled Position Index）**:

    - 功能：支持多类型、多分辨率条件输入的统一处理
    - 核心思路：原版 Flux-Kontext 的位置索引为三通道——第一通道用二值标记区分噪声和条件图像，后两通道编码空间坐标。RefTon 将第一通道扩展为离散条件标签（区分 person、garment、reference 等不同输入类型）。对每个条件独立生成位置索引，空间坐标按目标图像与条件图像的分辨率比缩放，保持跨分辨率的空间对齐。
    - 设计动机：原版二值设计无法区分多种异构条件输入。独立生成索引的方式比在像素空间拼接画布（如 Any2AnyTryon）更灵活，支持任意数量和分辨率的条件输入。消融实验证明缩放位置索引在 FID 和 KID 上均优于原始方案。

3. **参考图像引导机制**:

    - 功能：通过额外参考图像传递服装穿着效果的视觉信息
    - 核心思路：在训练时以 25% 的概率提供参考图像 $\mathbf{r}_i$（其他人穿着目标服装的照片），作为额外条件与 person/garment 图像一起输入模型。参考图像通过独立的位置索引集成。推理时可选择是否提供参考图像。
    - 设计动机：平铺服装图像无法展示透明面料、蕾丝细节、服装与人体的交互效果。参考图像弥补了这一信息缺口。实验表明加入参考图像后在所有指标上均有提升（如VITON-HD FID从5.45降至4.69）。

4. **参考图像数据生成流水线**:

    - 功能：自动构建包含参考图像的训练数据集
    - 核心思路：利用 Qwen2.5-VL 描述目标图像中人物外观并生成"相反描述"（不同肤色、发型等），再用 Flux-Kontext 编辑目标图像，将相反描述作为正向 prompt、原始描述作为负向 prompt，生成保持服装不变但改变人物外观的参考图像。非目标服装和动作也从描述库中随机采样以增加多样性。
    - 设计动机：三个约束确保参考图像质量——(i) 忠实保留目标服装 (ii) 人物外观与目标不同（防止模型走捷径直接复制）(iii) 非目标服装不同（增加多样性）。用 CLIP 特征去重和 VLM 质量过滤进一步保证数据质量。

### 损失函数 / 训练策略
使用标准 flow matching 损失训练。冻结 Flux-Kontext 的编码器和解码器，仅用 LoRA（rank=64, $\alpha=128$）微调 Transformer blocks。单数据集实验：VITON-HD 20k steps / DressCode 48k steps，batch=128，8×H100 GPU。混合数据集（VFR）训练用于增强泛化。

## 实验关键数据

### 主实验（VITON-HD + DressCode）

| 方法 | 输入条件 | VITON-HD LPIPS↓ | SSIM↑ | FID↓(paired) | FID↓(unpaired) |
|------|---------|----------------|-------|-------------|---------------|
| CatVTON | Mask | 0.057 | 0.870 | 5.43 | 9.02 |
| IDM-VTON | Mask+Pose | 0.102 | 0.870 | 6.29 | - |
| **RefTon** | **Mask** | **0.057** | 0.873 | 5.45 | 8.58 |
| **RefTon+R** | **Mask+Ref** | **0.049** | **0.879** | **4.69** | **8.43** |
| RefTon/MF | 无掩码 | 0.061 | 0.866 | 5.98 | 8.40 |
| RefTon+R/MF | 无掩码+Ref | 0.053 | 0.872 | 5.11 | **8.32** |

### 消融实验

| 设置 | VITON-HD FID↓ | DressCode FID↓ | 说明 |
|------|-------------|---------------|------|
| 有掩码，无参考 | 5.45 | 3.48 | 基准 |
| 有掩码，有参考 | **4.69** | **2.94** | 参考图显著提升 |
| 无掩码，无参考 | 5.98 | 3.84 | 去掉掩码轻微下降 |
| 无掩码，有参考 | 5.11 | 3.34 | 参考图弥补了掩码缺失 |
| 原版位置索引 (0.5×) | 5.29 | - | 无缩放 |
| 缩放位置索引 (0.5×) | **5.09** | - | 缩放后改善 |

### 关键发现
- **参考图像一致性提升所有指标**：在有掩码设置下，加入参考图像使 VITON-HD paired FID 从 5.45 降至 4.69（↓14%），LPIPS 从 0.057 降至 0.049（↓14%）。DressCode 上 FID 从 3.48 降至 2.94（↓15%）。
- **无掩码模式依然强劲**：即使完全去除 agnostic 掩码，性能仍与需要掩码的基线方法持平或更优（FID 8.40 vs CatVTON 9.02），展示了实际部署的便利性。
- **跨数据集泛化**：在混合 VFR 数据集上训练后，未单独在 VITON-HD/DressCode 上训练也能超越 OOTDiffusion 等基线。
- **StreetTryOn 跨域评估**：在从未训练过的 StreetTryOn 数据集上也取得了 SOTA 的 FID，证明了强泛化能力。
- **掩码质量问题**：消融可视化表明过度裁剪的掩码会丢失人物携带的物品（如手提包），保守掩码会保留不需要的区域。无掩码模式避免了这些问题。

## 亮点与洞察
- **从人类购物行为出发的参考图像设计**：真实用户在网购时确实更关注模特穿着效果而非平铺图片。参考图像捕获了平铺图无法展示的信息——透明材质、蕾丝细节、面料垂坠效果。这个设计直觉精准。
- **参考数据生成流水线**的设计很巧妙——利用 VLM 自动生成外观描述及其反面，配合 Flux-Kontext 进行编辑，在保持目标服装不变的同时改变人物外观和其他服装。三个约束（服装保真、人物不同、服装多样）有效防止了训练时的捷径学习。
- **统一框架处理多种输入模式**：一个模型同时支持有掩码/无掩码 × 有参考/无参考的四种组合，通过条件标签和概率采样优雅实现。

## 局限与展望
- 参考图像生成依赖 Flux-Kontext 的编辑质量，若编辑模型在某些服装类型上表现不佳，参考图像质量会下降。
- 仅评估了静态图片试穿，未扩展到视频试穿场景。
- LoRA 微调的表达能力可能有限，全参数微调（或更大 rank）是否能进一步提升有待探索。
- 参考图像在训练时仅以 25% 概率出现，是否存在更优的采样策略未充分研究。
- 未考虑多视角参考图像的融合。

## 相关工作与启发
- **vs CatVTON**: 同样使用两阶段训练和无掩码设计，但 CatVTON 缺少参考图像机制。RefTon 在此基础上引入参考图像显著提升了细节保真度。
- **vs TryOffDiff/ViTON-GUN**: 采用"先脱再穿"策略，引入误差累积且丢失服装细节。RefTon 直接利用参考图像避免了脱衣阶段。
- **vs Any2AnyTryon**: 也支持人对人试穿，但在拼接画布上生成位置索引。RefTon 对每个条件独立生成索引，更灵活地支持多分辨率输入。
- **vs OmniVTON**: 需要额外姿态和文本条件，RefTon 更简洁。

## 评分
- 新颖性: ⭐⭐⭐⭐ 参考图像引导的虚拟试穿思路新颖且实用，数据生成流水线设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多设置（掩码/无掩码×参考/无参考）、跨域评估、消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 解决了虚拟试穿中服装细节信息不足的实际问题，有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)
- [Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile-vton_high-fidelity_on-device_virtual_try-on.md)
- [UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/human_understanding/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)
- [Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](../../ECCV2024/human_understanding/wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)
- [VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](../../CVPR2025/human_understanding/vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)

<!-- RELATED:END -->
