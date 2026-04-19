---
title: >-
  [论文解读] InsightEdit: Towards Better Instruction Following for Image Editing
description: >-
  [CVPR 2025][image editing] 提出 InsightEdit，通过高质量 AdvancedEdit 数据集和双流桥接机制（文本+图像分支），将 MLLM 的文本推理与视觉语义同时注入扩散模型去噪过程，实现复杂指令跟随与高背景一致性的端到端图像编辑。
tags:
  - CVPR 2025
  - image editing
  - instruction following
  - MLLM
  - diffusion model
  - 数据集构建
---

# InsightEdit: Towards Better Instruction Following for Image Editing

**会议**: CVPR 2025  
**arXiv**: [2411.17323](https://arxiv.org/abs/2411.17323)  
**代码**: [项目页面](https://poppyxu.github.io/InsightEdit_web/)  
**领域**: image_generation  
**关键词**: image editing, instruction following, MLLM, diffusion model, two-stream bridging, AdvancedEdit dataset

## 一句话总结

提出 InsightEdit，构建 250 万级高质量编辑数据集 AdvancedEdit，并设计双流桥接机制将 MLLM 的文本推理特征和图像语义特征同时注入扩散模型，在复杂指令跟随和背景一致性上达到 SOTA。

## 研究背景与动机

**领域现状**: 基于指令的端到端图像编辑近年取得显著进展，InstructPix2Pix、InstructDiffusion、SmartEdit 等方法已探索了各种方案。

**现有痛点**:
1. **数据集质量低**: 现有数据集（如 InstructPix2Pix 的 Prompt2Prompt 方法生成）存在分辨率低（512²）、背景一致性差、指令过于模板化简单等问题。
2. **图像条件利用不足**: 现有方法主要依赖 CLIP 文本编码器或 MLLM 的文本级理解来提供条件，忽略了源图像丰富的视觉语义信息，导致复杂指令跟随能力弱、背景保持差。

**核心矛盾**: 高质量编辑需要同时理解复杂指令的语义和保持未编辑区域的视觉一致性，而现有方法仅依赖文本条件无法兼顾两者。

**本文目标**: 构建高质量数据集并设计能同时利用文本和图像条件的编辑框架，实现复杂指令跟随与高背景一致性。

**切入角度**: 从数据和模型两方面同时入手——用自动化管线构建高质量编辑对数据集，用双流桥接将 MLLM 的文本和图像信息同时注入扩散模型。

## 方法详解

### 整体框架

InsightEdit 由三个模块组成：
1. **理解模块 (Comprehension Module)**: 使用 LLaVA-7B 接收源图像和编辑指令，通过 [MM] 特殊 token 压缩多模态理解结果
2. **桥接模块 (Bridging Module)**: 双流设计，分别对齐文本特征和图像特征到扩散模型空间
3. **生成模块 (Generation Module)**: 使用解耦交叉注意力将文本和图像条件注入 UNet 生成目标图像

### 关键设计

**1. AdvancedEdit 数据集构建管线**
- **功能**: 自动化构建 250 万+ 高质量编辑对，覆盖移除、添加、替换三类任务。
- **核心思路**: 五步管线——① MLLM 提取全局描述和目标 JSON 列表 → ② GroundedSAM 生成 mask → ③ 使用 BrushNet/PowerPaint 等 mask-based 编辑模型生成目标图像 → ④ MLLM 重写指令（简单版+推理版）→ ⑤ VIEScore 质量过滤。
- **设计动机**: mask-based 编辑模型生成质量远优于 Prompt2Prompt；利用 MLLM 重写指令引入推理复杂度；VIEScore 过滤确保高质量；源数据使用 Pexels 高分辨率（~2K+）真实照片。

**2. 双流桥接机制 (Two-Stream Bridging)**
- **文本分支 (Q-Former + BIM)**: 使用 text-aligned Q-Former 从 [MM] token 隐状态提取文本推理信息 $q' = Q_\beta(q, h)$，再通过 BIM 模块实现源图像特征与文本特征的双向信息交换，输出 $f_{txt}$（文本条件）和 $v_{txt}$（文本感知视觉特征，加到 UNet 输入）。
- **图像分支 (IAA)**: 设计 Image Alignment Adapter，用 MLP Mapper 将 [MM] token 隐状态 $h \in \mathbb{R}^{r \times 4096}$ 映射为 $\mathbb{R}^{1 \times 768}$，与目标图像 CLIP 特征对齐监督；再线性扩展为 $\mathbb{R}^{N \times 768}$ 的 token 序列 $f_{img}$。
- **设计动机**: 文本特征提供高层编辑语义，图像特征包含更丰富的细节条件（如期望背景），二者互补能更精确引导编辑。

**3. 解耦交叉注意力生成**
- **功能**: 在 UNet 每个 block 中使用两层独立的交叉注意力分别处理文本条件和图像条件。
- **核心思路**: $\mathbf{Z} = \text{Attention}(\mathbf{Q}, \mathbf{K_{txt}}, \mathbf{V_{txt}}) + \lambda \cdot \text{Attention}(\mathbf{Q}, \mathbf{K_{img}}, \mathbf{V_{img}})$，推理时可调整 $\lambda$ 控制图像条件权重。
- **设计动机**: 受 IP-Adapter 启发，解耦设计既保证两路条件的独立有效性，又提供推理时的灵活控制。

### 损失函数 / 训练策略

- **LLM 损失**: 预测 [MM] token 的负对数似然 $L_{\text{LLM}}$，LLM 参数冻结 + LoRA 微调
- **IAA 对齐损失**: $\mathcal{L}_{\text{IAA}} = \|\text{CLIP}(\mathbf{I}_{\text{tar}}) - \text{Mapper}(h)\|_2^2$
- **扩散损失**: 标准 $\epsilon$-prediction 损失，输入为噪声 latent 与源图像 latent 的拼接 + $v_{txt}$
- **三阶段训练**: 在 8 × H100 GPU 上训练
- 仅使用 AdvancedEdit 中的 202,822 编辑对（受资源限制）

## 实验关键数据

### 主实验（AdvancedEdit-Eval 上对比）

| 方法 | VIEScore↑ | CLIPScore↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| InstructPix2Pix | 0.342 | 19.528 | 20.192 | 0.694 | 0.182 |
| SmartEdit-7B | 0.682 | 20.114 | 20.115 | 0.651 | 0.131 |
| InsightEdit | 0.738 | 20.395 | 21.267 | 0.675 | 0.112 |
| **InsightEdit + AdvancedEdit** | **0.831** | **21.002** | **22.871** | **0.716** | **0.071** |

### Reason-Edit 对比

| 方法 | Understanding VIEScore↑ | Reasoning VIEScore↑ |
|---|---|---|
| SmartEdit-7B | 0.866 | 0.835 |
| InsightEdit | 0.901 | 0.893 |
| **InsightEdit + AdvancedEdit** | **0.934** | **0.947** |

### 消融实验（IAA 模块）

| IAA | PSNR↑ | SSIM↑ | LPIPS↓ | CLIPScore↑ | VIEScore↑ |
|---|---|---|---|---|---|
| ✗ | 22.348 | 0.692 | 0.095 | 20.652 | 7.307 |
| ✓ | **22.871** | **0.716** | **0.071** | **21.002** | **7.545** |

### 关键发现

1. **数据驱动的提升**: InsightEdit + AdvancedEdit 在 VIEScore 上相比仅模型提升明显（0.738→0.831），证明高质量数据的价值。
2. **图像分支的关键作用**: IAA 模块使 LPIPS 从 0.095 降至 0.071，VIEScore 从 7.307 升至 7.545，有效提升背景一致性。
3. **理解+推理双优**: 在 Reason-Edit 的理解和推理场景中均超越 SmartEdit，证明双流机制的全面优势。
4. **AdvancedEdit 增强通用性**: 使用复杂指令数据训练后，在理解和推理场景中均持续提升。

## 亮点与洞察

- 数据构建管线利用 mask-based 编辑模型 + MLLM 重写的组合很巧妙，解决了 Prompt2Prompt 方法的质量问题
- 双流桥接同时利用 MLLM 的文本推理和图像感知能力，是对 SmartEdit 仅用文本条件的自然扩展
- IAA 模块设计简洁——用目标图像 CLIP 特征监督 Mapper 对齐，推理时无需目标图像
- 解耦交叉注意力 + 可调 $\lambda$ 提供了实用的推理灵活性

## 局限与展望

- 仅使用了 AdvancedEdit 中约 20 万对（全集 250 万+），受资源限制，更大规模训练可能带来更多提升
- 数据构建依赖 GPT-4o 等商用模型，成本较高
- 仅在移除/添加/替换三类任务上验证，未涵盖风格迁移、属性编辑等更复杂任务
- 推理速度未报告，包含 LLaVA-7B + UNet 的完整管线推理开销可能较大

## 相关工作与启发

- SmartEdit 首次将 MLLM 用于指令理解但仅用文本嵌入，本文扩展到双流（文本+图像）
- IP-Adapter 的解耦交叉注意力思路被有效复用于编辑任务
- 数据构建中 mask-based 编辑 → 指令重写 → 质量过滤的管线可推广到其他编辑数据集构建

## 评分

⭐⭐⭐⭐ — 数据+模型双线创新，实验充分；数据管线的自动化程度和质量令人印象深刻，双流桥接设计合理有效。

<!-- RELATED:START -->

## 相关论文

- [MagicQuill: An Intelligent Interactive Image Editing System](magicquill_an_intelligent_interactive_image_editing_system.md)
- [SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](../../ICCV2025/image_generation/superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](../../ICCV2025/image_generation/adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)

<!-- RELATED:END -->
