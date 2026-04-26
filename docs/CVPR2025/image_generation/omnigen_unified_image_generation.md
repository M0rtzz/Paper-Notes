---
title: >-
  [论文解读] OmniGen: Unified Image Generation
description: >-
  [CVPR 2025][图像生成][统一图像生成] 首个通用图像生成基础模型，仅由 VAE 和 Transformer 组成，通过统一多模态输入格式实现文生图、图像编辑、可控生成等多任务端到端处理。
tags:
  - CVPR 2025
  - 图像生成
  - 统一图像生成
  - 扩散模型
  - 多任务
  - 知识迁移
  - Rectified Flow
---

# OmniGen: Unified Image Generation

**会议**: CVPR 2025  
**arXiv**: [2409.11340](https://arxiv.org/abs/2409.11340)  
**代码**: https://github.com/VectorSpaceLab/OmniGen  
**领域**: 图像生成 / 统一模型  
**关键词**: 统一图像生成, 扩散模型, 多任务, 知识迁移, Rectified Flow

## 一句话总结

首个通用图像生成基础模型，仅由 VAE 和 Transformer 组成，通过统一多模态输入格式实现文生图、图像编辑、可控生成等多任务端到端处理。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：大语言模型 (LLM) 已统一语言生成任务并革新人机交互，但图像生成领域仍缺乏能在单一框架内处理各种任务的统一模型。当前每个新任务都需设计特定模块并微调（如 ControlNet 用于可控生成、InstantID 用于身份保持、InstructPix2Pix 用于编辑），导致繁琐的多步工作流程。例如，ControlNet 需先用检测器提取条件，InstantID 需人脸检测和编码。核心问题是：能否像 ChatGPT 处理语言任务那样，通过单个模型和用户指令端到端完成任意图像生成任务？本文提出 OmniGen，追求极简架构和灵活指令跟随，消除对额外插件和中间步骤的需求。

### 解决思路

**本文目标**：### 整体框架

OmniGen 仅包含两个组件：冻结的 SDXL VAE 用于图像编解码，Phi-3 初始化的大型 Transformer 模型用于基于条件的图像生成。


## 方法详解

### 整体框架

OmniGen 仅包含两个组件：冻结的 SDXL VAE 用于图像编解码，Phi-3 初始化的大型 Transformer 模型用于基于条件的图像生成。支持任意交错的文本和图像多模态输入，通过 rectified flow 方法训练。推理时从高斯噪声出发，迭代预测目标速度，最终用 VAE 解码器生成图像。整个架构无需 CLIP 文本/图像编码器等额外组件。

### 关键设计

1. **统一输入表示与注意力机制**：文本用 Phi-3 tokenizer 处理，图像用 VAE 编码后线性嵌入为视觉 token 序列，用 `<<<img>>>` 和 `<<</img>>>` 特殊标记封装并插入文本 token 流。注意力机制融合因果注意力和双向注意力：文本 token 使用因果注意力（左到右依赖），但每个图像内部使用双向注意力（patch 间互相注意），不同图像只能注意之前出现过的图像/文本。推理时可用 KV-cache 加速。

2. **X2I 数据集构建**：首个全面图像生成数据集（约1亿图像），统一所有任务为交错图文序列格式。涵盖文生图（Recap-DataComp 56M、LAION-Aesthetic 等）、多模态到图像（图像编辑、虚拟试衣、风格迁移、视觉条件控制等）、主体驱动生成（GRIT-Entity + 网络图像）。所有任务使用统一格式，无需任务特定特殊标记。

3. **编辑区域加权损失**：图像编辑任务中输入输出差异小，模型易学到直接复制的捷径。通过计算输入/目标图像潜表示的差异，对变化区域赋予显著更高的损失权重 $w_{i,j} = \frac{1}{\|\mathbf{x} - \mathbf{x}'\|^2}$（变化区域）vs 1（不变区域），引导模型聚焦需修改区域。

### 损失函数 / 训练策略

- Rectified Flow 目标: $\mathcal{L} = \mathbb{E}[\|(\mathbf{x} - \epsilon) - v_\theta(\mathbf{x}_t, t, c)\|^2]$
- 渐进式分辨率训练：256→512→1024→2240→多分辨率，共5阶段
- AdamW 优化器，104 张 A800 GPU
- 仅 VAE 冻结，Transformer 全参数训练

## 实验关键数据

### 主实验

| 能力 | 对比方法 | OmniGen 表现 |
|------|---------|-------------|
| 文生图 | SD系列, DALL-E, Imagen | 竞争力十足 |
| 可控生成 | ControlNet | 端到端，无需检测器 |
| 主体驱动 | InstantID, IP-Adapter | 无需人脸检测器 |
| 图像编辑 | InstructPix2Pix | 单模型完成 |
| 经典CV任务 | 特定任务模型 | 同样支持 |

### 消融实验

| 设计选择 | 效果 |
|---------|------|
| 无编辑区域加权 | 编辑任务输出直接复制输入 |
| 因果+双向注意力 | 优于纯因果或纯双向 |
| 渐进分辨率 | 低分辨率数据高效，高分辨率提升美学质量 |
| X2I 统一训练 | 显著的跨任务知识迁移 |

### 关键发现

- OmniGen 展现出强大的跨任务知识迁移能力，可处理未见过的任务和领域
- 统一训练培养了推理能力和思维链机制的潜力
- 无需 CLIP 编码器，单一 Transformer 自主编码所有条件信息，不同模态间有交互
- 极简架构（VAE + Transformer）即可覆盖需要多个专门模型的复杂工作流

## 亮点与洞察

- 真正的"一个模型做所有事"哲学，类似 LLM 在语言领域的革命
- 架构极简但功能全面——证明了足够大的 Transformer + 足够多的数据即可涌现通用能力
- X2I 数据集的统一格式设计是关键贡献——使不同任务得以在同一框架下训练
- 编辑区域加权损失解决了一个实际而微妙的训练问题（复制捷径）

## 局限与展望

- 模型参数量大，推理成本高于专用模型
- 在特定任务上可能不如精心设计的专用模型（如 ControlNet 在边缘检测条件控制上）
- 训练数据规模和计算资源需求巨大（104 A800）
- 可探索更高效的架构设计和知识蒸馏方案

## 相关工作与启发

- **vs ControlNet/IP-Adapter**: 需要额外插件和多步流程；OmniGen 端到端完成
- **vs SD3/DALL-E**: 专注文生图；OmniGen 同时支持多模态输入的多种任务
- **vs InstructPix2Pix**: 仅做编辑；OmniGen 将编辑作为统一框架的子任务之一

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个真正意义上的通用图像生成基础模型
- **实验充分度**: ⭐⭐⭐⭐ — 多任务多基准评估，但部分任务对比不够深入
- **写作质量**: ⭐⭐⭐⭐ — 愿景清晰，架构描述简洁
- **实用价值**: ⭐⭐⭐⭐⭐ — 极大简化图像生成工作流，开源贡献大

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation](unic-adapter_unified_image-instruction_adapter_with_multi-modal_transformer_for_.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)

<!-- RELATED:END -->
