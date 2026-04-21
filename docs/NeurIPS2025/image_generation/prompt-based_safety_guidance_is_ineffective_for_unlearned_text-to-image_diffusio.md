---
title: >-
  [论文解读] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][概念遗忘] 本文发现训练式概念遗忘（unlearning）与免训练安全引导（negative prompt guidance）两种安全方法组合后效果反而下降，提出用概念反演（Concept Inversion）获得的隐式负向嵌入替换显式负向提示，有效恢复了免训练方法在遗忘模型上的防御能力。
tags:
  - NeurIPS 2025
  - 图像生成
  - 概念遗忘
  - 安全引导
  - 负向提示
  - 概念反演
  - 文生图安全
---

# Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.04834](https://arxiv.org/abs/2511.04834)  
**代码**: 无  
**领域**: AI安全 / 扩散模型安全  
**关键词**: 概念遗忘, 安全引导, 负向提示, 概念反演, 文生图安全

## 一句话总结

本文发现训练式概念遗忘（unlearning）与免训练安全引导（negative prompt guidance）两种安全方法组合后效果反而下降，提出用概念反演（Concept Inversion）获得的隐式负向嵌入替换显式负向提示，有效恢复了免训练方法在遗忘模型上的防御能力。

## 研究背景与动机

文生图扩散模型（如 Stable Diffusion）存在生成有害内容的风险。目前主流的两类安全方案分别是：（1）训练式概念遗忘，通过微调模型权重来"忘记"有害概念（如 ESD、SPM、UCE、DUO）；（2）免训练引导方法，在推理阶段利用负向提示（negative prompt）引导生成远离有害内容（如 SLD、SAFREE）。

这两种方法在设计上是正交的，理论上可以叠加使用以获得更强的安全保障。然而，作者发现了一个关键矛盾：**经过遗忘微调的模型已经不再响应显式的负向提示词**（如"Sexual Acts"、"Nudity"等），因此将免训练方法直接应用于遗忘模型时，效果只有边际提升甚至反而下降。

核心洞察在于：虽然遗忘模型对显式文本提示不再敏感，但概念反演（Concept Inversion）的研究表明，**遗忘模型仍然能通过隐式嵌入生成有害内容**，这意味着在文本嵌入空间中仍存在代表有害概念的向量，只是难以用人工选择的显式词语找到。

## 方法详解

### 整体框架

方法非常简洁：将 SLD 和 SAFREE 等免训练安全方法中原本使用的**手工选取的负向提示嵌入** $\mathbf{C}_n$ 替换为通过**概念反演获得的隐式概念嵌入** $\mathbf{C}_*$。整个流程无需修改遗忘方法或免训练方法本身的任何机制，即插即用。

### 关键设计

1. **免训练安全方法回顾**：SLD 在 CFG 基础上增加了一个利用负向提示嵌入 $\mathbf{c}_n$ 的额外引导项，自适应地减去有害方向的分数；SAFREE 则构造负向子空间，调整与该子空间接近的提示 token 嵌入。两者都依赖于用户手动选择的负向提示词。

2. **概念反演获取隐式嵌入**：采用 Textual Inversion 的技术框架，在遗忘模型上最小化 LDM 的去噪损失来优化一个嵌入向量 $\mathbf{c}_*$，使其能够代表被遗忘的有害概念。具体而言，给定有害图像数据集 $\mathbf{x}$，通过优化 $\mathbf{c}_* = \arg\min_{\mathbf{c}} \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathbf{c}, t)\|_2^2]$ 来找到最优的概念嵌入。使用 Adam 优化器，学习率 $5 \times 10^{-3}$，batch size 1，梯度步数 3000。

3. **嵌入替换**：将获得的隐式概念嵌入 $\mathbf{C}_*$ 直接替代原始的 $\mathbf{C}_n$，分别代入 SAFREE 的嵌入调整函数和 SLD 的负向引导项中。实验中设 $K_* = 1$（即只用一个概念嵌入）。

### 损失函数 / 训练策略

概念反演阶段使用标准的 LDM 去噪损失（L2 loss），只优化嵌入向量 $\mathbf{c}$，冻结模型参数。整个概念反演过程在 NVIDIA RTX 3090 上约需 15 分钟。

## 实验关键数据

### 主实验

使用 DUO 作为基础遗忘模型，在 4 个不同超参数 $\beta$ 的 checkpoint 上评估。

**暴力任务（Violence）- Ring-a-Bell 基准**：

| $\beta$ | 方法 | DSR ↑ | PP ↑ |
|---------|------|-------|------|
| 1000 | DUO | 0.613 | 0.820 |
| 1000 | DUO+SLD | 0.760 | 0.784 |
| 1000 | DUO+SLD+Ours | **0.880** | 0.693 |
| 1000 | DUO+SAFREE | 0.793 | 0.733 |
| 1000 | DUO+SAFREE+Ours | **0.947** | 0.676 |

**裸露任务（Nudity）- I2P 基准**：

| $\beta$ | 方法 | DSR ↑ | PP ↑ |
|---------|------|-------|------|
| 2000 | DUO | 0.442 | 0.802 |
| 2000 | DUO+SLD | 0.484 | 0.777 |
| 2000 | DUO+SLD+Ours | **0.937** | 0.712 |
| 2000 | DUO+SAFREE | 0.568 | 0.678 |
| 2000 | DUO+SAFREE+Ours | **0.947** | 0.632 |

### 消融实验（可迁移性验证）

从 $\beta=500$ checkpoint 提取的概念嵌入直接用于其他 checkpoint：

| $\beta$ | DUO+SLD+Ours (共享CE) DSR | DUO+SAFREE+Ours (共享CE) DSR |
|---------|---------------------------|-------------------------------|
| 250 | 1.000 | 0.990 |
| 1000 | 0.990 | 0.990 |
| 2000 | 0.905 | 0.968 |

### 关键发现

- 原始方法 SLD/SAFREE 直接用于遗忘模型时，nudity 任务上 DSR 甚至不升反降
- 使用隐式概念嵌入后，DSR 在所有 checkpoint 上均大幅提升（nudity 任务最高从 0.44 提升到 0.94）
- 概念嵌入在不同 checkpoint 间具有迁移性，从一个 checkpoint 提取的嵌入可有效用于同一基底模型的其他 checkpoint
- 用提取的概念嵌入提示原始 SD v1.4 时仍能生成有害图像，说明遗忘模型与原始模型共享残留的负向文本嵌入空间

## 亮点与洞察

- **问题发现极为精准**：首次系统性地指出训练式遗忘与免训练引导之间存在根本性的不兼容——遗忘模型已"免疫"显式负向提示
- **方法极度简洁**：只是一个嵌入的替换操作，不修改任何已有方法的结构，展现了"以毒攻毒"的巧妙思路
- **可迁移性发现**：隐式概念嵌入在不同 checkpoint 间的迁移能力暗示遗忘模型中残留的概念表示具有一定程度的共享结构

## 局限与展望

- 需要为每个遗忘模型单独提取概念嵌入，无法做到完全即插即用
- 需要有害图像数据集作为概念反演的输入，在实际部署中可能受限
- 仅在 DUO 这一种遗忘方法上验证，其他遗忘方法（ESD、UCE 等）尚未测试
- 只设置 $K_* = 1$ 的简单情况，多概念嵌入的效果有待探索
- 评估仅覆盖 nudity 和 violence 两个任务

## 相关工作与启发

- DUO 使用偏好优化进行图像级遗忘，是当前较强的遗忘方法
- SLD/SAFREE 代表了免训练安全引导的两种主流范式
- Concept Inversion 原本用于攻击遗忘模型（恢复被遗忘概念），本文创新性地将其用于防御
- 这一工作启示我们：**安全方法的组合不一定带来叠加效果**，需要深入理解各方法的作用机制后再进行整合

## 评分

- 新颖性: ⭐⭐⭐⭐ （问题发现有价值，方法本身较简单）
- 实验充分度: ⭐⭐⭐ （仅一种遗忘方法、两种任务，规模偏小）
- 写作质量: ⭐⭐⭐⭐ （逻辑清晰，动机明确）
- 价值: ⭐⭐⭐⭐ （揭示了安全方法间的兼容性问题，具有实际意义）

<!-- RELATED:START -->

## 相关论文

- [Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

<!-- RELATED:END -->
