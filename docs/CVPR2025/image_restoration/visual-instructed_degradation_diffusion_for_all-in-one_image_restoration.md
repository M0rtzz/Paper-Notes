---
title: >-
  [论文解读] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration
description: >-
  [CVPR 2025][图像恢复][全能图像复原] Defusion 提出用"视觉指令"替代文本指令来引导 all-in-one 图像复原，通过将退化效果施加到标准化视觉元素上构建可视化的退化描述，并在退化空间（而非图像空间）进行扩散去噪，在 8 个复原任务上超越 task-specific 和 all-in-one 方法。
tags:
  - CVPR 2025
  - 图像恢复
  - 全能图像复原
  - 视觉指令
  - 退化扩散
  - 退化空间
  - 视觉先验
---

# Visual-Instructed Degradation Diffusion for All-in-One Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2506.16960](https://arxiv.org/abs/2506.16960)  
**代码**: 无（有 Project Page）  
**领域**: 图像复原 / 全能复原  
**关键词**: 全能图像复原, 视觉指令, 退化扩散, 退化空间, 视觉先验

## 一句话总结

Defusion 提出用"视觉指令"替代文本指令来引导 all-in-one 图像复原，通过将退化效果施加到标准化视觉元素上构建可视化的退化描述，并在退化空间（而非图像空间）进行扩散去噪，在 8 个复原任务上超越 task-specific 和 all-in-one 方法。

## 研究背景与动机

**领域现状**：图像复原包括去噪、去模糊、去雾、去雨等多种任务。传统方法为每种退化训练独立模型，all-in-one 方法试图用单一模型处理多种退化。当前 all-in-one 方法主要有两种获取退化先验的路径：隐式先验（通过子网络/可学习 prompts 从 LQ 图像提取）和显式先验（通过语言模型处理文本描述）。

**现有痛点**：(1) 隐式先验（如 PromptIR、AirNet）本质上等同于增加参数，对复杂/组合退化的泛化能力有限；(2) 文本描述的显式先验（如 SUPIR、InstructIR）存在语言-视觉对齐弱的问题——"enhance the image"可以指增亮也可以指去噪，"underwater enhancement"实际包含增亮+去模糊，语言无法准确传达低层视觉细节。

**核心矛盾**：低级视觉退化的具体视觉效果（如雨线的方向密度、噪声的强度分布）很难用语言精确描述，但又需要精确的退化类型/强度信息来指导复原。

**本文目标**：用视觉方式精确描述视觉退化——让视觉为视觉代言（let visual speak for visual）。

**切入角度**：构造"视觉指令"——将退化过程施加到包含各种视觉基元的标准测试图（TE42 色卡）上，生成直观展示退化效果的视觉描述。这些视觉指令与退化模式天然对齐，避免了文本描述的语义鸿沟。

**核心 idea**：用视觉指令替代文本指令引导复原，并在退化空间（LQ-HQ 差值空间）而非图像空间进行扩散，使退化描述与扩散目标天然对齐，提升复原效果和可控性。

## 方法详解

### 整体框架

Defusion 包含三个核心组件：(1) 视觉指令构造：用 TE42 色卡作为"视觉基底"，施加退化得到视觉指令；(2) 退化 tokenizer：用 VQ-VAE 编码器将视觉指令压缩为离散的退化 token；(3) 退化扩散模型：在退化空间 $\mathbf{y}_0 = \mathbf{x}_{LQ} - \mathcal{T}_v(\mathbf{x}_{LQ})$ 上进行扩散去噪，以 LQ 图像（通过 IRB）和视觉指令（通过 VIA）为条件。复原结果为 $\hat{\mathcal{T}}_v(\mathbf{x}_{LQ}) = \mathbf{x}_{LQ} - \hat{\mathbf{y}}_0$。

### 关键设计

1. **视觉指令 (Visual Instructions)**:

    - 功能：精确、直观地描述图像退化的视觉效果
    - 核心思路：从 TE42 色卡中选取四类视觉构件（同心纹理、随机纹理、自然纹理、色块）随机组合形成"视觉基底"（visual ground），对其施加与 LQ 图像相同的退化过程（加雨、加噪等），得到退化后的视觉基底即视觉指令。视觉指令直观展示了退化对各种视觉元素的影响，且与图像语义无关（提高泛化性）。推理时额外编码干净的视觉基底（"null"），用两者的差值作为最终退化 token
    - 设计动机：TE42 包含丰富的纹理和色彩模式，能全面揭示退化效果。与文本不同，视觉指令天然与低级视觉退化模式对齐。组合退化的视觉指令也自然展示组合效果（如图 1 所示）

2. **退化空间扩散 (Degradation Space Diffusion)**:

    - 功能：在 LQ-HQ 差值空间而非图像空间进行扩散，使扩散目标与退化描述对齐
    - 核心思路：定义退化空间 $\mathcal{D}_v = \{\mathbf{x}_{LQ} - \mathcal{T}_v(\mathbf{x}_{LQ})\}$，扩散前向过程 $p_t(\mathbf{y}_t | \mathbf{y}_0, \mathbf{x}_{LQ}, \mathbf{v}) = \mathcal{N}(\alpha_t \mathbf{y}_0, \sigma_t^2 I)$。训练 score model $\mathbf{s}_\theta(\mathbf{y}_t, t, \mathbf{x}_{LQ}, \mathbf{v})$，推理时从 $\mathbf{y}_1 \sim \mathcal{N}(0, I)$ 反向求解 SDE 得到 $\hat{\mathbf{y}}_0$，最终 $\hat{\mathbf{x}}_{HQ} = \mathbf{x}_{LQ} - \hat{\mathbf{y}}_0$
    - 设计动机：三大优势：(1) $\mathbf{y}_t$ 与退化高度相关，复原更准确；(2) 退化空间分布比图像空间更一致，训练更稳定、模型容量需求更低；(3) 当视觉指令不匹配实际退化时模型"什么都不做"，可控性更好。此外，对 $\alpha_t, \sigma_t$ 无特殊要求，可直接适配预训练扩散模型

3. **条件注入机制 (IRB + VIA)**:

    - 功能：将 LQ 图像和视觉指令分别注入扩散 U-Net
    - 核心思路：**Image Restoration Bridge (IRB)**：轻量级卷积网络将 LQ 图像编码为多尺度特征图，通过 AdaLN-Zero 注入 U-Net 各层。**Visual Instruction Adapter (VIA)**：复制预训练 text-to-image U-Net 的 cross-attention 层，query 为 U-Net 特征图，key/value 为退化 token $\mathbf{z}_v$
    - 设计动机：IRB 不用 ControlNet 或 concat 方式是因为退化空间预测目标与 LQ 图像差异大。VIA 采用 IP-Adapter 风格设计保留了文本条件能力，允许文本作为额外引导

### 损失函数 / 训练策略

退化 tokenizer 用 VQ-VAE 损失 $\mathcal{L}_{inst} = \mathcal{L}_{rec} + \mathcal{L}_{VQ}$（含 MSE + LPIPS + hinge adversarial loss）训练。扩散模型用标准 denoising score matching 训练。训练时以 0.1 概率将视觉指令替换为干净的 visual ground。多任务联合训练。

## 实验关键数据

### 主实验

**8 个任务对比（Defusion vs task-specific 最优 / all-in-one 最优）**：

| 任务 | 指标 | Task-specific SOTA | All-in-one SOTA | Defusion |
|------|------|-------------------|----------------|----------|
| 运动去模糊 (GoPro) | PSNR | 33.20 (DiffIR) | 32.49 (MPerceiver) | **34.53** |
| 散焦去模糊 (DPDD) | PSNR | 26.18 (FocalNet) | 28.21 (NAFNet*) | **29.68** |
| 去雪 (Snow100K) | PSNR | 30.43 (WeatherDiff) | 31.02 (MPerceiver) | **32.11** |
| 去雾 (Dense-Haze) | PSNR | 17.07 (FocalNet) | 16.72 (NAFNet*) | **17.55** |
| 去噪 (SIDD) | PSNR | 40.02 (Restormer) | 40.60 (Restormer*) | **40.89** |

Defusion 在**所有 8 个任务**上均超越 task-specific 和 all-in-one 方法，且是 all-in-one 模型。

### 消融实验

从论文结构和数据来看，视觉指令 vs 文本指令、退化空间 vs 图像空间的对比是核心消融。退化空间在训练稳定性和可控性上有明显优势。

### 关键发现

- 视觉指令比文本指令在低级视觉任务中更有效，因为视觉指令与退化模式天然对齐
- 退化空间扩散比图像空间扩散更稳定、更准确
- IRB 通过 AdaLN-Zero 注入优于 ControlNet 和 concat 方式
- "null" visual ground 减法增强了退化 token 的判别性
- 单一 all-in-one 模型超越所有 task-specific 专家模型，体现了跨任务知识共享的价值

## 亮点与洞察

- **"让视觉为视觉代言"**的设计哲学优雅且实用——完全跳出了文本描述的语义鸿沟
- 用 TE42 色卡作为视觉基底是一个聪明的选择——工业标准测试图已经设计好了丰富的纹理和色彩模式
- 退化空间扩散的思路极具启发性——扩散什么、预测什么应该与条件信息对齐
- VQ-VAE tokenizer 将视觉指令压缩为离散 token 的设计与 LLM 的 tokenization 理念一致

## 局限与展望

- 视觉指令的构造依赖已知的退化过程，对"黑箱"退化（如真实世界未知退化）需要额外的退化检测步骤
- TE42 色卡虽然丰富但并非为图像复原专门设计，可能遗漏某些退化模式
- 未讨论视觉指令在超分辨率等生成式复原任务上的效果
- 退化空间扩散的 $\mathbf{y}_0 = \mathbf{x}_{LQ} - \mathbf{x}_{HQ}$ 假设完美配对数据，对非配对场景的推广需研究

## 相关工作与启发

- 与同组工作 **ResFlow** 形成互补——ResFlow 用确定性流做单任务复原，Defusion 用 SDE 扩散做多任务复原
- **IP-Adapter** 的 cross-attention 注入思路被用于视觉指令的注入
- 视觉指令的概念可推广到其他低级视觉任务（如视频复原、3D 复原）

## 评分

- **新颖性**: 9/10 — 视觉指令和退化空间扩散两个创新点均有独到见解
- **实验充分度**: 9/10 — 8 个任务、合成+真实数据集全面评估
- **写作质量**: 8/10 — 动机表述清晰，方法描述详细
- **价值**: 9/10 — all-in-one 模型全面超越专家模型，实用性极强

<!-- RELATED:START -->

## 相关论文

- [Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models](prior_does_matter_visual_navigation_via_denoising_diffusion_bridge_models.md)
- [Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)

<!-- RELATED:END -->
