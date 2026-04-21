---
title: >-
  [论文解读] Attention Decomposition for Cross-Domain Semantic Segmentation
description: >-
  [ECCV 2024][图像分割][跨域语义分割] 本文提出 ADFormer，一种用于跨域语义分割的新型 Transformer 架构，通过将解码器中的交叉注意力分解为域无关和域特定两部分，结合梯度反转对抗学习，有效缩小源域和目标域之间的分布差异，在 GTA→Cityscapes 和 SYNTHIA→Cityscapes 两个基准上以显著更低的复杂度超越了现有无 proposal 方法。
tags:
  - ECCV 2024
  - 图像分割
  - 跨域语义分割
  - 注意力分解
  - 对抗学习
  - Transformer
  - 域自适应
---

# Attention Decomposition for Cross-Domain Semantic Segmentation

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: https://github.com/helq2612/ADFormer  
**领域**: 语义分割 / 域自适应  
**关键词**: 跨域语义分割, 注意力分解, 对抗学习, Transformer解码器, 域自适应

## 一句话总结

本文提出 ADFormer，一种用于跨域语义分割的新型 Transformer 架构，通过将解码器中的交叉注意力分解为域无关和域特定两部分，结合梯度反转对抗学习，有效缩小源域和目标域之间的分布差异，在 GTA→Cityscapes 和 SYNTHIA→Cityscapes 两个基准上以显著更低的复杂度超越了现有无 proposal 方法。

## 研究背景与动机

**领域现状**：跨域语义分割旨在将在源域（如合成数据 GTA5、SYNTHIA）上训练的分割模型迁移到目标域（如真实场景 Cityscapes）。随着 Transformer 在分割任务中的广泛应用，SOTA 方法主要分为两类：基于 CNN 的域自适应方法和基于 Transformer 的无 proposal 方法。Transformer 方法通过查询 token（query tokens）预测分割 mask，展现了良好的性能。

**现有痛点**：现有的跨域分割方法存在两个主要问题：(1) 传统 Transformer 分割模型（如 Mask2Former）通常使用复杂的编码器和相对简单的解码器，没有显式考虑域差异；(2) 在跨域场景下，查询 token 与图像 token 的交叉注意力同时吸收了域无关信息（如物体形状、结构等跨域共享的语义）和域特定信息（如纹理风格、光照等域特有的干扰），后者正是导致域差距的关键因素。

**核心矛盾**：Transformer 解码器中的交叉注意力机制无差别地处理来自不同域的图像特征，使得查询 token 不可避免地编码了域特定的噪声信息。如果能让查询 token 只关注域无关的共享语义部分，就能从根本上减少域差距。

**本文目标** (1) 如何在 Transformer 解码器中显式分离域无关和域特定的注意力交互；(2) 如何在保持模型轻量的同时实现有效的域自适应。

**切入角度**：作者提出可以将交叉注意力矩阵数学分解为域无关部分和域特定部分，然后通过约束让查询 token 主要与域无关部分交互。结合梯度反转块（GRL）引入对抗学习，进一步增强域不变性。

**核心 idea**：将 Transformer 解码器的交叉注意力分解为域无关和域特定两部分，让查询 token 只关注域无关语义以实现域自适应分割。

## 方法详解

### 整体框架

ADFormer 采用编码器-解码器架构，设计思路是"轻编码器 + 复杂解码器"。编码器（较轻量）提取多尺度图像特征；解码器通过多层交叉注意力让可学习的查询 token 与图像特征交互，最终每个查询 token 预测一个分割 mask 和对应的类别标签。关键创新在解码器端：(1) 交叉注意力被分解为域无关和域特定两个分支；(2) 梯度反转块控制反向传播使模型学习域不变表示。

### 关键设计

1. **注意力分解（Attention Decomposition）**:

    - 功能：将解码器中查询 token 与图像 token 之间的交叉注意力分解为域无关和域特定两个部分
    - 核心思路：给定查询 token $Q$ 和图像 token $K, V$，标准交叉注意力为 $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$。ADFormer 引入两组投影矩阵：域无关投影 $W_{di}$ 和域特定投影 $W_{ds}$。图像 token 通过这两组投影分别映射为域无关分量 $K_{di}, V_{di}$ 和域特定分量 $K_{ds}, V_{ds}$。查询 token 主要与域无关分量 $K_{di}, V_{di}$ 交互来预测分割结果，而域特定分量 $K_{ds}, V_{ds}$ 用于对抗学习。这样查询 token 获得的信息天然过滤了域特定干扰
    - 设计动机：源域和目标域的图像虽然在纹理、风格上差异很大，但物体的形状、结构等语义信息是共享的。通过显式分解，模型被迫将共享语义和域干扰分离到不同的子空间中

2. **梯度反转对抗学习（Gradient Reverse Adversarial Learning）**:

    - 功能：通过对抗训练约束域特定分支学到真正的域差异信息
    - 核心思路：在域特定注意力分支后接一个域分类器（判断输入来自源域还是目标域），同时使用梯度反转层（Gradient Reversal Layer, GRL）。前向传播时 GRL 不做任何操作；反向传播时 GRL 将梯度取反。这使得域特定分支被"鼓励"编码域差异信息（域分类器要分对），而编码器的特征提取器被"鼓励"减少域差异（梯度反转使其朝混淆域分类器的方向优化）
    - 设计动机：仅靠注意力分解的结构约束还不够——如果没有显式的域分类信号，模型可能将有用的语义信息也错误地放入域特定分支。对抗学习提供了明确的域差异信号，保证分解的质量

3. **轻量编码器-复杂解码器架构**:

    - 功能：在保持模型效率的同时提供足够的跨域自适应能力
    - 核心思路：不同于 Mask2Former 等使用重型编码器（如 Swin-L），ADFormer 使用相对轻量的编码器（如 ResNet-50 或小型 ViT），但在解码器端增加了注意力分解和对抗学习模块。解码器使用多层交叉注意力层，每层都进行注意力分解。查询 token 数量与类别数对应，直接预测每类的分割 mask
    - 设计动机：跨域分割的核心挑战在解码阶段的特征-语义映射，而非编码阶段的特征提取。将自适应能力集中在解码器上可以在不显著增加计算量的前提下有效缩小域差距

### 损失函数 / 训练策略

总训练损失包含：(1) 分割损失 $L_{seg}$：结合交叉熵和 Dice 损失来监督 mask 预测；(2) 对抗损失 $L_{adv}$：域分类器的二元交叉熵损失，通过 GRL 反向传播。总损失 $L = L_{seg} + \lambda L_{adv}$。训练分两阶段：先在源域数据上预训练编码器和解码器，再使用源域-目标域联合训练进行域自适应（目标域无标注，仅参与对抗损失）。推理时只使用域无关注意力分支。

## 实验关键数据

### 主实验

| 数据集迁移 | 指标 | 本文 ADFormer | 之前SOTA | 提升 |
|-----------|------|-------------|----------|------|
| GTA→Cityscapes | mIoU | SOTA级别 | 无 proposal 方法 | 显著提升 |
| SYNTHIA→Cityscapes | mIoU | SOTA级别 | 无 proposal 方法 | 显著提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无分解（baseline） | mIoU 基准 | 标准交叉注意力 |
| + 注意力分解 | mIoU 提升 | 仅结构分解 |
| + 对抗学习 | mIoU 进一步提升 | 加入 GRL 域分类器 |
| 完整 ADFormer | 最佳 | 分解 + 对抗联合 |

### 关键发现

- 注意力分解是最关键的组件，仅通过结构约束就能显著缩小域差距
- 对抗学习与分解结合效果更佳，两者互补
- ADFormer 的模型复杂度显著低于现有 SOTA 的 proposal-free 方法，但性能更优
- 在更轻的编码器上也能保持竞争力，说明域自适应能力主要在解码器中

## 亮点与洞察

- 注意力分解的思路非常优雅——将一个连续的注意力空间显式分为域无关和域特定两个子空间，从机制设计层面解决域自适应问题
- "轻编码器 + 复杂解码器"的设计理念与主流趋势相反，但在跨域场景中确实更合理，因为域差距主要在语义映射环节产生
- 梯度反转与注意力分解的结合自然流畅，没有引入过多的超参数和训练技巧

## 局限与展望

- 本文主要验证了合成→真实的域迁移场景，对于真实→真实（如晴天→雨天）的域迁移效果未充分验证
- 注意力分解的粒度是固定的（域无关 vs 域特定二分法），对于多域场景可能需要更细粒度的分解方式
- 域分类器的设计较简单，更复杂的域判别器或多级域分类可能进一步提升效果
- 未讨论模型在 open-vocabulary 或 zero-shot 跨域场景下的表现

## 相关工作与启发

- **DAFormer** 和 **HRDA** 是基于 Transformer 的域自适应分割代表方法，但它们主要在编码器端做域自适应
- **Mask2Former** 的查询 token + 交叉注意力范式为本文的解码器设计奠定了基础
- **DANN**（Domain Adversarial Neural Networks）的梯度反转思想在域自适应中被广泛使用，本文将其创新性地应用于 Transformer 注意力机制
- 注意力分解的思想或可推广到其他需要"选择性关注"的场景，如跨模态融合、多任务学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 注意力分解用于域自适应是新颖的思路，理论简洁优美
- 实验充分度: ⭐⭐⭐ 两个标准基准验证，但缺乏更多域迁移场景
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，推导逻辑严谨
- 价值: ⭐⭐⭐⭐ 对 Transformer 域自适应领域有较好的启发意义

<!-- RELATED:START -->

## 相关论文

- [Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](../../NeurIPS2025/segmentation/interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)

<!-- RELATED:END -->
