---
title: >-
  [论文解读] Dual Diffusion for Unified Image Generation and Understanding
description: >-
  [CVPR 2025][图像生成][多模态扩散] 提出 D-DiT（Dual Diffusion Transformer），首个完全端到端的多模态扩散模型，在图像分支用连续流匹配、文本分支用离散掩码扩散，统一损失函数下同时训练图像生成和文本理解。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多模态扩散"
  - "图文统一模型"
  - "离散扩散"
  - "流匹配"
  - "视觉问答"
---

# Dual Diffusion for Unified Image Generation and Understanding

**会议**: CVPR 2025  
**arXiv**: [2501.00289](https://arxiv.org/abs/2501.00289)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 多模态扩散, 图文统一模型, 离散扩散, 流匹配, 视觉问答

## 一句话总结

提出 D-DiT（Dual Diffusion Transformer），首个完全端到端的多模态扩散模型，在图像分支用连续流匹配、文本分支用离散掩码扩散，统一损失函数下同时训练图像生成和文本理解。

## 研究背景与动机

**领域现状**：扩散模型在文生图领域占据主导，自回归模型在视觉语言理解方面表现卓越。两者能否统一成一个双向模型？

**现有痛点**：现有多模态扩散模型要么文本推理能力有限（UniDiffuser 需要 AR 解码器将扩散文本 latent 转文本），要么本质上仍依赖自回归模型做文本生成（Show-O、Transfusion）。

**核心矛盾**：之前缺乏经验上可行的离散文本扩散过程，限制了纯扩散模型进行语言建模。

**本文目标**：构建首个纯扩散的端到端多模态模型，同时支持图像生成、图像描述和视觉问答。

**核心 idea**：基于 MM-DiT 架构，图像分支用 flow matching 做连续扩散，文本分支用 masked diffusion 做离散扩散，联合训练。

## 方法详解

### 整体框架

D-DiT 基于 SD3 的 MM-DiT 架构，双分支 Transformer：图像分支输出 velocity field 预测，文本分支输出 denoised token 预测。训练时交替做图像去噪（文本干净）和文本去噪（图像干净），推理时可分别执行 T2I 和 I2T 任务。

### 关键设计

1. **图像-文本联合扩散损失**:

    - 功能：统一训练图像和文本的条件生成
    - 核心思路：$L_{dual} = L_{image} + \lambda_{text} L_{text}$，其中图像用 flow matching MSE 损失，文本用 masked diffusion 的 NELBO 损失。训练时仅对被条件的一侧加噪——做图像扩散时文本干净，做文本扩散时图像干净
    - 设计动机：简单优雅，反向传播通过 DiT 的两个分支联合优化

2. **文本的可控填充推理**:

    - 功能：实现视觉问答任务
    - 核心思路：VQA 任务中，问题 token 保持不变（不加噪），仅对答案位置的 token 进行掩码扩散采样。这利用了掩码扩散天然的文本填充能力
    - 设计动机：前代扩散模型无法做 VQA，而掩码扩散允许条件性填充

3. **从预训练 SD3 初始化**:

    - 功能：快速适配文本生成能力
    - 核心思路：从 SD3 预训练权重初始化 DiT，在文本分支上加线性头用于 token 预测。利用 T5 的 <extra_id0> 特殊 token 作为掩码 token，第二阶段解冻其 embedding
    - 设计动机：仅需约 25B text tokens 即可展现有意义的文本输出，展示了极快的适配能力

### 损失函数 / 训练策略

三阶段训练：(1) 在 Datacomp-1b 上预训练 60K 步；(2) 在高质量理解数据上续训 200K 步，可选 512 分辨率微调；(3) 在 LLaVA 指令数据上微调 50K 步。总共约 40M 图文对。

## 实验关键数据

### 主实验

- VQA 基准：在 MME、GQA、POPE 上超越 Show-O（另一个统一模型）
- 图像生成：GenEval 上保持 SD3 原有性能，部分颜色指标还有提升
- 首个支持完整 VQA 的纯扩散模型

### 关键发现

- 联合扩散训练不会导致图像生成能力的灾难性遗忘
- 文本扩散不需要文本单独训练数据，image-text pair 足够
- D-DiT 在长文本回答中展示了细粒度的多模态理解能力

## 亮点与洞察

- 首次证明扩散模型可以完全替代自回归模型进行多模态建模
- 损失函数设计极其简洁——两个单模态扩散损失的加权和
- 从 SD3 快速适配的能力令人印象深刻

## 局限与展望

- 模型规模和训练数据相对较小，与 SOTA VLM 仍有差距
- 文本生成需要较多扩散步数（256 步），推理速度慢于自回归
- 尚未探索文本独立生成（无条件文本扩散）

## 评分

- 新颖性：9/10 — 首个纯扩散多模态模型
- 技术深度：8/10 — 连续+离散扩散的优雅统一
- 实验充分度：7/10 — 受限于模型规模
- 写作质量：8/10 — 清晰，背景介绍充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
