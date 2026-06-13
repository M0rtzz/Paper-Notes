---
title: >-
  [论文解读] DreamOmni: Unified Image Generation and Editing
description: >-
  [CVPR 2025][图像生成][统一生成编辑] 构建统一文生图+多种编辑任务（指令编辑/修补/拖拽/参考生成）的 2.5B DIT 模型，用 Qwen2-VL 替换文本编码器实现统一视觉-语言 prompt 理解，通过合成贴纸数据管线高效创建编辑训练数据，在生成和编辑上同时达到 SOTA。 领域现状：图像生成（T2I）和…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "统一生成编辑"
  - "DIT架构"
  - "合成数据"
  - "VLM编码器"
  - "多任务模型"
---

# DreamOmni: Unified Image Generation and Editing

**会议**: CVPR 2025  
**arXiv**: [2412.17098](https://arxiv.org/abs/2412.17098)  
**代码**: [https://zj-binxia.github.io/DreamOmni-ProjectPage/](https://zj-binxia.github.io/DreamOmni-ProjectPage/)  
**领域**: 图像生成  
**关键词**: 统一生成编辑、DIT架构、合成数据、VLM编码器、多任务模型

## 一句话总结
构建统一文生图+多种编辑任务（指令编辑/修补/拖拽/参考生成）的 2.5B DIT 模型，用 Qwen2-VL 替换文本编码器实现统一视觉-语言 prompt 理解，通过合成贴纸数据管线高效创建编辑训练数据，在生成和编辑上同时达到 SOTA。

## 研究背景与动机

**领域现状**：图像生成（T2I）和图像编辑（指令编辑、inpainting、拖拽编辑等）通常是独立模型。统一两者可以共享视觉知识但面临多任务冲突和数据不均衡的挑战。

**现有痛点**：(1) 编辑训练数据获取困难——需要编辑前后的配对图像。(2) 不同编辑任务的 prompt 格式差异大（文本指令/区域 mask/拖拽点/参考图像）。(3) UNet 架构在多任务联合训练时收敛慢。

**核心矛盾**：统一模型需要同时在生成和编辑上表现好，但编辑任务需要精细的空间理解而生成任务需要创意发散。

**本文目标** 设计一个同时处理 T2I 和多种编辑任务的统一框架，解决数据、架构和 prompt 统一三个问题。

**切入角度**：用 DIT 替代 UNet（计算集中在 2× 下采样潜空间更高效）+ VLM 编码器统一理解多种 prompt + 合成贴纸数据管线高效创建编辑训练对。

**核心 idea**：VLM 编码器统一多种 prompt 理解 + DIT+UNet 残差连接加速收敛 + 合成贴纸数据管线解决编辑数据稀缺。

## 方法详解

### 整体框架
VLM（Qwen2-VL 7B）编码多模态 prompt → DIT 自注意力融合 VLM 特征和噪声潜空间 → UNet 式残差连接加速收敛 → Rectified Flow 训练 → 3 阶段渐进分辨率训练（256→512→1024）。

### 关键设计

1. **合成贴纸数据管线**:

    - 功能：高效创建精确的编辑训练数据
    - 核心思路：基于贴纸的合成——将物体作为"贴纸"添加/移除/替换到图像上生成编辑前后对。支持指令编辑（添加/删除/替换）、拖拽编辑（平移/缩放/旋转）、参考生成、分割等。~60M 合成编辑对 + 125M T2I 数据
    - 设计动机：比人工标注编辑对高效 1000×，且覆盖多种编辑类型。关键洞察：编辑训练的目的是教模型"编辑语义"而非"新概念"

2. **VLM 编码器替换文本编码器**:

    - 功能：统一理解文本/图像/区域等多种 prompt
    - 核心思路：Qwen2-VL 7B 可以同时处理文本指令+参考图像+区域标注，输出统一的条件 embedding。DIT 通过自注意力融合 VLM 特征和噪声潜空间
    - 设计动机：传统 T5/CLIP 文本编码器无法处理图像输入。VLM 统一了生成和编辑的 prompt 理解

3. **DIT + UNet 残差连接**:

    - 功能：加速多任务训练收敛
    - 核心思路：DIT 块之间加入类似 UNet 的跳连（encoder→decoder），收敛速度提升 4×。DIT 比 UNet 更适合因为计算集中在 2× 下采样空间
    - 设计动机：消融显示有残差连接在同等训练步数下 FID 更低，收敛更快

### 损失函数 / 训练策略
Rectified Flow 损失。3 阶段渐进分辨率（256→512→1024）。T2I + 编辑联合训练防止概念遗忘。

## 实验关键数据

### 主实验

| 任务 | DreamOmni | SOTA 对比 |
|------|-----------|----------|
| GenEval 总分 | 0.70 | SD3-Medium 0.70 |
| Inpainting FID↓ | **0.837** | SD-inp 1.352 |
| 指令编辑 | 超越 InstructPix2Pix | - |
| 拖拽编辑 | 超越 DragGAN | - |

### 消融实验

| 配置 | 效果 |
|------|------|
| UNet 架构 | DIT 更好（2× 下采样更高效） |
| 无 UNet 残差 | 收敛慢 4× |
| T2I only | 编辑能力丧失 |
| 编辑 only | T2I 概念遗忘 |
| **联合 T2I+编辑** | **两者协同最优** |

### 关键发现
- **联合训练产生协同效应**：T2I 防止概念遗忘，编辑训练提升指令遵循能力
- **合成贴纸数据足以训练高质量编辑**：不需要真实编辑对，关键是教"编辑语义"
- **DIT + 残差连接 = 收敛速度 4× 提升**

## 亮点与洞察
- **"编辑训练教语义不教概念"的洞察**解放了数据构建——合成数据完全够用
- **VLM 统一 prompt 理解**使得一个模型能处理五种编辑任务+生成

## 局限与展望
- VLM 编码器（7B）增加了推理开销
- 合成贴纸数据的编辑质量上限受限于合成方法
- 仅在 1024 分辨率验证，更高分辨率未测

## 相关工作与启发
- **vs InstructPix2Pix**：仅做指令编辑。DreamOmni 统一 5 种编辑+T2I
- **vs SD3/FLUX**：仅做 T2I。DreamOmni 在保持 T2I 质量同时支持编辑

## 评分
- 新颖性: ⭐⭐⭐⭐ VLM编码器+合成数据+统一框架组合有价值
- 实验充分度: ⭐⭐⭐⭐ 多任务评估、收敛消融
- 写作质量: ⭐⭐⭐⭐ 框架描述清楚
- 价值: ⭐⭐⭐⭐ 对统一生成编辑有工程意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation](unic-adapter_unified_image-instruction_adapter_with_multi-modal_transformer_for_.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)

</div>

<!-- RELATED:END -->
