---
title: >-
  [论文解读] The Power of Context: How Multimodality Improves Image Super-Resolution
description: >-
  [CVPR 2025][图像分割][多模态超分辨率] 提出 MMSR，一种融合深度、语义分割、边缘和文本描述等多模态信息的扩散模型超分辨率方法，通过多模态潜在连接器和多模态 CFG 有效抑制幻觉并提升 SR 质量。
tags:
  - CVPR 2025
  - 图像分割
  - 多模态超分辨率
  - 扩散模型
  - 分类器自由引导
  - 多模态融合
  - 图像修复
---

# The Power of Context: How Multimodality Improves Image Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2503.14503](https://arxiv.org/abs/2503.14503)  
**代码**: [项目主页](https://mmsr.kfmei.com/)  
**领域**: 图像超分辨率 / 图像分割  
**关键词**: 多模态超分辨率, 扩散模型, 分类器自由引导, 多模态融合, 图像修复

## 一句话总结

提出 MMSR，一种融合深度、语义分割、边缘和文本描述等多模态信息的扩散模型超分辨率方法，通过多模态潜在连接器和多模态 CFG 有效抑制幻觉并提升 SR 质量。

## 研究背景与动机

- 单图超分辨率（SISR）是一个不适定问题，从低分辨率输入恢复高频细节极具挑战性
- 近年来基于扩散模型的超分辨率方法取得了显著进展，通常利用文本提示来激活预训练生成先验
- 然而，仅依赖文本描述存在固有缺陷：文本无法准确表达空间关系，导致纹理描述只能全局应用
- 例如，用 "lion" 作为提示词时，模型可能在舌头上生成毛发纹理——因为狮子有毛，但舌头不应长毛
- 深度图、语义分割等空间模态可以提供互补的空间信息，减少超分辨率问题的不确定性
- 从信息论角度，辅助模态 $m$ 的引入可降低条件分布的熵：$H(p(\mathbf{x}|\mathbf{x}_{LR})) \geq H(p(\mathbf{x}|\mathbf{x}_{LR}, m))$
- 现有多模态方法（如 ControlNet、IP-Adapter）为每种模态复制网络组件，计算开销巨大
- 需要一种高效、灵活的架构，能够在不修改扩散网络的前提下融合任意数量的模态信息

## 方法详解

### 整体框架

MMSR 基于预训练文本到图像扩散模型（与 Stable Diffusion v2 同架构）构建。推理时，首先从低分辨率图像提取四种模态：Gemini Flash 生成文本描述、Depth Anything 估计深度图、Mask2Former 生成语义分割、Canny 提取边缘。所有模态通过统一的 token 编码后，经多模态潜在连接器压缩为固定长度的潜在 token，作为扩散模型的跨注意力条件。低分辨率图像则通过拼接方式（类似 InstructPix2Pix）提供额外条件。

### 关键设计

**1. Token-wise 多模态编码**
- **功能**: 将不同模态统一编码为 token 序列，无需修改扩散网络架构
- **核心思路**: 利用预训练 VQGAN 图像分词器将深度/分割/边缘编码为 $16 \times 16$ 的离散 token 序列（特征维度 256，码本大小 1024）。离散 token 比连续 token 更好地保留各模态信息，避免重建伪影。token 序列与文本嵌入拼接后用于跨注意力
- **设计动机**: 离散量化在保留模态特定信息方面优于连续表示，同时统一的 token 格式允许灵活添加/删除模态。引入可学习的空 token $m_\emptyset$ 表示缺失模态（训练时以 10% 概率随机替换），增强鲁棒性

**2. 多模态潜在连接器 (MMLC)**
- **功能**: 高效压缩长多模态 token 序列，降低扩散模型跨注意力的计算复杂度
- **核心思路**: 使用一组可学习的潜在 token（128 个）通过跨注意力从完整多模态序列（$256 \times 3 + 77 = 845$ 个 token）中提取关键信息，再通过自注意力进一步整合，输出固定长度的条件 token
- **设计动机**: 直接使用完整多模态序列的跨注意力复杂度为 $\mathcal{O}(M^2)$，MMLC 将其降至 $\mathcal{O}(MN)$（$N \ll M$），实现线性复杂度。消融实验表明 MMLC 不仅提升效率，还减少幻觉伪影

**3. 多模态分类器自由引导 (m-CFG)**
- **功能**: 在高引导率下抑制幻觉和虚假细节，改善感知质量与保真度的平衡
- **核心思路**: 传统 CFG 的负向引导仅使用空文本 token，导致负向约束较弱。m-CFG 在正向和负向生成过程中同时使用多模态潜在 token 条件：$\tilde{\epsilon}(\mathbf{z}_t, c, m) = (1+w)\epsilon(\mathbf{z}_t, c, \text{pos}, m) - w\epsilon(\mathbf{z}_t, c, \text{neg}, m)$
- **设计动机**: 多模态信息增强负向引导的约束力，在使用高引导率（10-14）时有效抑制颜色偏移和错误纹理，而传统 CFG 在高引导率下性能急剧下降

### 损失函数

采用标准扩散模型训练损失，即预测噪声与真实噪声之间的均方误差：

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \mathbf{x}_{LR}, m, t)\|^2\right]$$

训练数据使用 LSDIR 和 DIV2K 数据集，通过 RealESRGAN 退化策略生成低分辨率图像。

## 实验关键数据

### 主实验：定量比较

| 方法 | LPIPS↓ | DISTS↓ | NIQE↓ | FID↓ | MUSIQ↑ | CLIPIQA↑ |
|------|--------|--------|-------|------|--------|----------|
| R-ESRGAN | 0.3868 | 0.2601 | 4.92 | 53.46 | 58.64 | 0.5424 |
| StableSR | 0.4055 | 0.2542 | 4.66 | 36.57 | 62.95 | 0.6486 |
| SeeSR | 0.3843 | 0.2257 | 4.93 | 31.93 | 68.33 | 0.6946 |
| **MMSR** | **0.3707** | **0.2071** | **4.25** | **29.35** | **70.06** | **0.7164** |

*DIV2K-Val-3k 512×512 基准，MMSR 在所有感知质量指标上全面领先*

### 消融实验

| 消融项 | MUSIQ↑ | NIQE↓ | DISTS↓ | LPIPS↓ |
|--------|--------|-------|--------|--------|
| w/o MMLC | 69.69 | 3.48 | 0.1781 | 0.3929 |
| **w. MMLC** | **72.31** | **3.42** | **0.1492** | **0.2810** |

*DIV2K-Val-100 1024p，MMLC 在所有指标上均有提升*

| 引导方式 | LPIPS@w=2 | LPIPS@w=10 | LPIPS@w=14 |
|----------|-----------|------------|------------|
| cfg | 0.3239 | 0.4491 | 0.5064 |
| $m_\emptyset$-cfg | 0.2815 | 0.4803 | 0.5493 |
| **m-cfg** | **0.2810** | **0.3471** | **0.3772** |

*m-CFG 在高引导率下显著抑制 LPIPS 退化*

### 关键发现

- 深度信息主要提升感知质量（MUSIQ），而分割和边缘更有助于保持身份一致性（DISTS）
- 多模态默认设置在感知质量和身份保持之间取得最佳平衡
- 通过调整各模态的注意力温度 $\delta \in [0.4, 10]$，可以实现细粒度控制：降低深度温度增强景深效果，降低分割温度突出特定物体特征，降低边缘温度增强细节锐度

## 亮点与洞察

1. **信息论视角的动机**: 通过条件互信息的非负性证明多模态信息必然降低超分辨率的不确定性，为方法提供了理论支撑
2. **统一 token 表示**: 利用 VQGAN 将异构模态统一为离散 token，避免了传统方法为每种模态复制网络的高昂代价
3. **模态级可控性**: 首次在超分辨率任务中实现了各模态影响力的独立、连续调节，开辟了新的交互方向

## 局限与展望

- 多模态信息提取引入计算开销（Gemini Flash 仅 0.34 img/s），成为推理速度瓶颈
- 当低分辨率图像退化严重时，提取的多模态信息本身可能不准确（如扭曲的边缘、错误分割）
- 未来可探索更快速的视觉-语言模型和更鲁棒的模态提取模块

## 相关工作与启发

- 与 ControlNet、IP-Adapter 相比，本方法用统一 token 编码替代网络复制，更加高效灵活
- SeeSR、PASD 等文本驱动方法仅使用单一文本模态，缺乏空间引导能力
- 多模态 CFG 的思路可以推广到其他条件生成任务中，增强负向引导的约束力

## 评分

⭐⭐⭐⭐ — 方法设计优雅，从信息论出发的动机清晰有力，多模态融合架构高效实用。实验全面覆盖合成和真实场景，消融充分。模态级可控性是有价值的新特性，但计算开销和对预测模态质量的依赖是实际应用中的制约因素。

<!-- RELATED:START -->

## 相关论文

- [SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning](../../NeurIPS2025/segmentation/srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)
- [Paint by Inpaint: Learning to Add Image Objects by Removing Them First](paint_by_inpaint_learning_to_add_image_objects_by_removing_them_first.md)
- [ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting](rocket-1_mastering_open-world_interaction_with_visual-temporal_context_prompting.md)
- [SmartEraser: Remove Anything from Images using Masked-Region Guidance](smarteraser_remove_anything_from_images_using_masked-region_guidance.md)
- [OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels](overlock_an_overview-first-look-closely-next_convnet_with_context-mixing_dynamic.md)

<!-- RELATED:END -->
