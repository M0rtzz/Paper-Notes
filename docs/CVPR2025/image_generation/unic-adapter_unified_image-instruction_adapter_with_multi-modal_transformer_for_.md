---
title: >-
  [论文解读] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation
description: >-
  [CVPR 2025][图像生成][统一可控生成] UNIC-Adapter 基于 MM-DiT 架构设计了一个统一的图像-指令适配器，通过跨注意力机制和 RoPE 增强的空间感知注入，使单个 SD3 模型能够处理像素级控制、主题驱动生成和风格迁移等 14 种条件图像生成任务。 领域现状：文本到图像（T2I）扩散模型已经能生…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "统一可控生成"
  - "适配器"
  - "MM-DiT"
  - "跨注意力"
  - "旋转位置编码"
---

# UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation

**会议**: CVPR 2025  
**arXiv**: [2412.18928](https://arxiv.org/abs/2412.18928)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 统一可控生成, 适配器, MM-DiT, 跨注意力, 旋转位置编码

## 一句话总结
UNIC-Adapter 基于 MM-DiT 架构设计了一个统一的图像-指令适配器，通过跨注意力机制和 RoPE 增强的空间感知注入，使单个 SD3 模型能够处理像素级控制、主题驱动生成和风格迁移等 14 种条件图像生成任务。

## 研究背景与动机

**领域现状**：文本到图像（T2I）扩散模型已经能生成高质量图像，但仅靠文本提示难以精确控制像素级布局、物体外观和全局风格。ControlNet 用平行编码器实现像素级控制，IP-Adapter 通过 CLIP 特征做内容控制，但每种控制类型需要训练单独的模型。

**现有痛点**：(1) 每种条件需要专门的适配器，导致训练成本高、部署复杂；(2) Uni-ControlNet 将条件分为 local/global 两组分别训练，不够统一；(3) OmniGen/Instruct-Imagen 虽然统一但需要多阶段训练整个模型，成本极高。

**核心矛盾**：统一性与效率的矛盾——要支持多种条件类型需要灵活的特征交互，但在预训练 T2I 模型上高效适配（不训练整个模型）是具有挑战的。

**本文目标**：设计一个基于预训练 T2I 模型（SD3）的轻量适配器，冻结基础模型参数，仅训练适配器即可实现跨多种条件类型的统一可控生成。

**切入角度**：MM-DiT 架构天然支持多模态特征的全注意力交互，可以同时处理任务指令（文本）和条件图像（图像）两个模态，非常适合作为统一适配器的骨干。

**核心 idea**：用 MM-DiT 块处理任务指令和条件图像的双模态交互，然后通过带 RoPE 的跨注意力将提取的信息注入主生成分支，实现对各种条件类型的统一处理。

## 方法详解

### 整体框架
UNIC-Adapter 建立在 SD3（MM-DiT 架构）之上。输入包括三部分：文本提示（主生成分支）、任务指令（如 "Generate an image from this edge map"）和条件图像（如 Canny 边缘图、参考风格图等）。任务指令通过 CLIP/T5 文本编码器提取特征 $Z_{\text{ist}}$，条件图像通过 VAE 提取特征 $Z_{\text{con}}$。在适配器的 N 个 MM-DiT 块中，$Z_{\text{ist}}$ 和 $Z_{\text{con}}$ 通过全注意力互相交互。然后通过跨注意力将适配器提取的信息注入主生成分支的图像特征 $Z_{\text{img}}$。

### 关键设计

1. **MM-DiT 双模态特征提取**:

    - 功能：让任务指令和条件图像互相理解，提取与任务相关的条件特征
    - 核心思路：将指令和条件图像映射为 QKV，然后做全注意力交互：$Z'_{\text{ist}} = \text{Attn}(Q_{\text{ist}}, [K_{\text{ist}} \| K_{\text{con}}], [V_{\text{ist}} \| V_{\text{con}}])$，$Z'_{\text{con}}$ 同理。在 N 层堆叠中，两个模态的特征被逐层精炼
    - 设计动机：任务指令指定了"如何使用"条件图像（是做边缘控制还是风格迁移），MM-DiT 的双向注意力使指令能调制条件特征的提取方式，实现"一个适配器多种任务"

2. **RoPE 增强的跨注意力注入**:

    - 功能：将适配器提取的条件信息注入主生成分支
    - 核心思路：主分支的图像特征 $Z_{\text{img}}$ 通过新引入的线性层 $L_{\text{cross}}^q$ 生成 query，适配器输出的 $K_{\text{ist}}, K_{\text{con}}$ 作为 key/value，做跨注意力后残差加回：$Z''_{\text{img}} = Z'_{\text{img}} + \text{Attn}(Q'_{\text{img}}, [K_{\text{ist}} \| K_{\text{con}}], [V_{\text{ist}} \| V_{\text{con}}])$。对 Q 和 K 施加 2D RoPE（高度和宽度维度分别编码），使相近像素的注意力分数更高
    - 设计动机：像素级控制任务需要精确的空间对应关系，RoPE 提供了相对位置编码，确保条件图像和生成图像之间的空间对齐。消融显示加入 RoPE 后 Canny 控制的 F1 从 29.27 提升到 31.32

3. **统一的训练策略**:

    - 功能：冻结 SD3 基础模型，仅训练适配器中新引入的参数
    - 核心思路：适配器使用 SD3 的预训练参数初始化（减少学习难度），冻结基础模型和适配器中 attention 后的 FFN 层。用 GPT-4o 为每种任务生成 20 条同义指令增加多样性。总训练参数约 1.2B，在 16 块 H100 上训练 100K 步
    - 设计动机：从预训练参数初始化让适配器继承了 MM-DiT 对多模态交互的先验知识，大幅减少训练难度

### 损失函数 / 训练策略
使用标准的扩散模型去噪损失，混合像素级控制（MultiGen-20M 2.8M 图像）、主题驱动（2.1M 对）和风格迁移（90K 图像）三类数据联合训练。

## 实验关键数据

### 主实验

| 任务/方法 | Canny (F1↑) | HED (SSIM↑) | Seg (mIoU↑) | Depth (RMSE↓) |
|----------|-------------|-------------|-------------|---------------|
| ControlNet (单任务) | 34.65 | 0.7621 | 32.55 | 35.90 |
| ControlNet++ (单任务) | 37.04 | 0.8097 | 43.64 | 28.32 |
| UniControl (多任务) | 30.82 | 0.7969 | 25.44 | 39.18 |
| OmniGen (多任务) | 35.54 | 0.8237 | 44.23 | 28.54 |
| **UNIC-Adapter** (多任务) | **38.94** | **0.8369** | 42.89 | 31.10 |

| 方法 | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| DreamBooth | 0.668 | 0.803 | 0.305 |
| OmniGen | 0.801 | 0.847 | 0.301 |
| **UNIC-Adapter** | **0.816** | 0.841 | **0.306** |

### 消融实验

| 配置 | Canny F1↑ | HED SSIM↑ | DINO↑ |
|------|-----------|-----------|-------|
| 仅条件图像做 Key | 29.01 | 0.7767 | 0.769 |
| 仅指令做 Key | 22.38 | 0.5599 | 0.694 |
| 无 RoPE | 29.27 | 0.7707 | 0.771 |
| 无新 Query 层 | 29.98 | 0.7840 | 0.778 |
| **完整模型** | **31.32** | **0.7934** | 0.769 |

### 关键发现
- 条件图像特征 $K_{\text{con}}$ 是像素级控制的关键，去掉后 Canny F1 从 31.32 暴跌到 22.38
- 同时注入指令和条件特征比单独注入任一者都更好，说明指令引导条件特征的机制有效
- RoPE 对像素级控制提升明显（F1 从 29.27→31.32），对主题驱动生成影响较小
- 在主题驱动生成上，UNIC-Adapter 的 DINO 分数（0.816）超越了需要 test-time tuning 的方法如 SuTI（0.741）

## 亮点与洞察
- **MM-DiT 的多模态天赋**：巧妙利用 MM-DiT 本身就能处理文本-图像双模态的特性，自然地扩展为指令-条件图像的双模态交互，避免了设计新架构的开销
- **指令驱动的统一性**：通过文本指令区分不同的控制任务，使同一个适配器可以动态切换行为，这比为每种条件设计专用路径更优雅且可扩展
- **RoPE 增强空间感知**：在跨注意力中引入旋转位置编码是简单而有效的设计，可以迁移到其他需要空间对应的视觉生成任务

## 局限与展望
- 仅基于 SD3 Medium 实验，在更大的模型（如 FLUX）上的表现有待验证
- 约 1.2B 训练参数量仍然不小，进一步压缩适配器参数是值得探索的方向
- 风格控制的定量评估不足，主要靠定性结果
- 多条件组合（如同时指定边缘图 + 风格图）的能力未充分验证

## 相关工作与启发
- **vs ControlNet**: ControlNet 是单条件单模型，UNIC-Adapter 是多条件单模型。在 Canny 和 HED 控制上 UNIC-Adapter 已超越 ControlNet
- **vs OmniGen**: OmniGen 也做统一生成但需要训练整个模型，UNIC-Adapter 只训练适配器部分，更高效。且在主题驱动上 DINO 分数更高
- 这种"指令 + 条件"的统一范式可以推广到视频生成控制

## 评分
- 新颖性: ⭐⭐⭐⭐ MM-DiT 用于统一可控生成是合理且新颖的设计
- 实验充分度: ⭐⭐⭐⭐ 三大类任务、14 种条件、详细消融
- 写作质量: ⭐⭐⭐⭐ 公式清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为可控生成提供了统一且高效的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICLR 2026\] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](../../ICLR2026/image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)

</div>

<!-- RELATED:END -->
