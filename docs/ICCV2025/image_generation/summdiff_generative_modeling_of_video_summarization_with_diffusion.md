---
title: >-
  [论文解读] SummDiff: Generative Modeling of Video Summarization with Diffusion
description: >-
  [ICCV 2025][图像生成][视频摘要] SummDiff 首次将扩散模型引入视频摘要任务，将其定义为条件生成问题，通过学习"好摘要"的分布来生成多种合理摘要，更好地反映视频摘要任务固有的主观性。
tags:
  - ICCV 2025
  - 图像生成
  - 视频摘要
  - 扩散模型
  - 条件生成
  - 主观性建模
  - 背包问题
---

# SummDiff: Generative Modeling of Video Summarization with Diffusion

**会议**: ICCV 2025  
**arXiv**: [2510.08458](https://arxiv.org/abs/2510.08458)  
**领域**: 扩散模型·视频摘要  
**关键词**: 视频摘要, 扩散模型, 条件生成, 主观性建模, 背包问题  

## 一句话总结

SummDiff 首次将扩散模型引入视频摘要任务，将其定义为条件生成问题，通过学习"好摘要"的分布来生成多种合理摘要，更好地反映视频摘要任务固有的主观性。

## 研究背景与动机

视频摘要旨在从长视频中选出关键帧以保留核心内容。然而，该任务本质上具有**强主观性**：不同标注者对于"好摘要"的定义各不相同。

现有方法的主要局限包括：

**忽略标注多样性**：大多数方法将多位标注者的帧级重要性评分取平均作为训练目标，这种回归式方法会丢失不同标注者的独立视角。例如，若一半标注者选择视频前1/4的片段，另一半选择后1/4的片段，简单平均会让两部分获得相似的中间分数，无法区分两种有效的摘要方式。

**确定性输出**：给定视频只能生成一个摘要，无法反映多种合理的摘要可能。

**评估指标不完善**：对背包（knapsack）步骤的分析不够深入，F1 分数对视频分割过度敏感。

作者提出**生成式视角**：将视频摘要视为条件生成任务，让模型学习好摘要的概率分布，从而通过采样生成多种合理摘要。

## 方法详解

### 整体框架

SummDiff 的流程分为三个阶段：(1) 视频编码；(2) 基于扩散的重要性评分去噪；(3) 背包优化生成最终摘要。

### 1. 视频编码

使用预训练图像编码器提取每帧特征 $\mathbf{z}_i \in \mathbb{R}^D$，再通过自注意力进行上下文化，得到整体视觉特征矩阵 $\mathbf{Z} \in \mathbb{R}^{N \times D}$。

### 2. 视频重要性评分去噪器（核心）

**前向过程**：对单个标注者的重要性评分 $\mathbf{s}_0 \in [0,1]^N$ 先做 logit 变换 $\mathbf{u}_0 = \log \frac{\mathbf{s}_0}{1-\mathbf{s}_0}$，然后在 logit 空间添加高斯噪声：

$$\mathbf{s}_t = \sqrt{\bar{\alpha}_t} \mathbf{s}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t$$

**Codebook 量化**：将噪声化的 logit $\mathbf{u}_t$ 通过 sigmoid 映射回 $[0,1]$，等分为 $K$ 段，每段对应一个可学习的 $D$ 维嵌入。这样将标量评分 $\mathbf{u}_t \in \mathbb{R}^N$ 映射为 $\mathcal{C}(\mathbf{u}_t) \in \mathbb{R}^{N \times D}$，以适配交叉注意力。

**Transformer 交叉注意力去噪**：量化嵌入 $\mathcal{C}(\mathbf{u}_t)$ 作为 query，视觉特征 $\mathbf{Z}$ 作为 key/value，利用 AdaLN-Zero 机制注入时间步和位置编码条件，避免信息混叠：

$$\mathbf{X}_1 = \mathbf{A}_1 \odot \text{softmax}(\mathbf{Q}_t' \mathbf{K}'^{\top}) \mathbf{V}' + \mathbf{Q}_t'$$

### 3. 训练与推理

**训练损失**：对每个标注者的独立评分进行 MSE 训练：

$$\mathcal{L}(\mathbf{s}_0, \hat{\mathbf{s}}_0) = \|\mathbf{s}_0 - \sigma(\text{FC}(f_\theta(\mathcal{C}(\mathbf{u}_t), t, \mathbf{Z})))\|_2^2$$

**推理**：从随机噪声 $\mathbf{u}_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，通过 DDIM 反向过程逐步去噪，最终 sigmoid 得到 $\hat{\mathbf{s}}_0 \in (0,1)^N$。

### 4. 摘要生成（KTS + 背包）

使用 KTS 将视频分割为语义片段，以片段均值作为重要性，通过背包动态规划在预算约束（如 $\rho=0.15$）下选择最优片段。

## 实验

### 主要结果

| 方法 | SumMe τ | SumMe ρ | TVSum τ | TVSum ρ |
|------|---------|---------|---------|---------|
| Random | 0.000 | 0.000 | 0.000 | 0.000 |
| Human | 0.205 | 0.213 | 0.177 | 0.204 |
| CSTA | 0.108 | 0.120 | 0.168 | 0.221 |
| **SummDiff** | **0.133** | **0.148** | **0.173** | **0.226** |

在 TVT 设置下，SummDiff 在 SumMe 上 τ 提升 23%，在 TVSum 上也优于所有基线。

### Mr. HiSum 大规模评测

| 方法 | τ | ρ | MAP@50% | MAP@15% |
|------|---|---|---------|---------|
| CSTA | 0.128 | 0.185 | 63.38 | 30.42 |
| **SummDiff** | **0.175** | **0.238** | **65.44** | **33.83** |

在大规模 Mr. HiSum 数据集上（31,892 视频），SummDiff 在所有指标上大幅超越最强基线 CSTA，展现了良好的可扩展性。

### 消融实验

| 配置 | SumMe τ | TVSum τ |
|------|---------|---------|
| 无 codebook（直接用标量） | 0.105 | 0.152 |
| 简单加法注入条件 | 0.118 | 0.161 |
| **AdaLN-Zero + codebook** | **0.133** | **0.173** |

codebook 量化和 AdaLN-Zero 条件注入对性能提升至关重要。

## 亮点与洞察

1. **首次将扩散模型用于视频摘要**，将确定性回归问题转化为条件生成问题，天然适配主观性任务。
2. **对每个标注者的独立评分训练**而非平均评分，有效保留了多样化的摘要视角。
3. **Codebook 量化**巧妙解决了标量评分与高维交叉注意力之间的维度匹配问题。
4. **提出了新的评估指标**，通过背包分析提供更深入的评估视角。

## 局限性

- 在传统小数据集（SumMe/TVSum）上的评估可靠性有限（分别仅 25/50 个视频）。
- 训练需要多标注者的独立评分数据，在大规模数据集（如 Mr. HiSum）上只能使用聚合标注，无法充分发挥生成式优势。
- logit 空间变换需要 clip 处理，可能引入数值偏差。

## 相关工作

- 传统视频摘要：VASNet、PGL-SUM、CSTA 等通过回归方式预测平均评分
- 生成式方法：GAN-based 摘要（SUM-GAN）使用对抗损失但目标不同
- 扩散模型：首次在视频摘要中应用，借鉴了 DiT 的条件生成机制

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.0** |

<!-- RELATED:START -->

## 相关论文

- [Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)
- [Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [Diffusion Generative Modeling on Lie Group Representations](../../NeurIPS2025/image_generation/diffusion_generative_modeling_on_lie_group_representations.md)

<!-- RELATED:END -->
