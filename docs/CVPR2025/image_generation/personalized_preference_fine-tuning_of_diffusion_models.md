---
title: >-
  [论文解读] Personalized Preference Fine-tuning of Diffusion Models
description: >-
  [CVPR 2025][图像生成][个性化偏好对齐] PPD 提出个性化偏好扩散模型微调框架：利用 VLM 从少量（4 对）偏好示例中提取用户嵌入，通过解耦交叉注意力层注入扩散模型，结合 DPO 目标同时优化多用户的个性化偏好，仅需 4 个偏好对即可为新用户生成匹配其偏好的图像（76% 胜率）。
tags:
  - CVPR 2025
  - 图像生成
  - 个性化偏好对齐
  - DPO
  - 多奖励优化
  - VLM用户嵌入
  - 扩散模型微调
---

# Personalized Preference Fine-tuning of Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2501.06655](https://arxiv.org/abs/2501.06655)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 个性化偏好对齐, DPO, 多奖励优化, VLM用户嵌入, 扩散模型微调

## 一句话总结
PPD 提出个性化偏好扩散模型微调框架：利用 VLM 从少量（4 对）偏好示例中提取用户嵌入，通过解耦交叉注意力层注入扩散模型，结合 DPO 目标同时优化多用户的个性化偏好，仅需 4 个偏好对即可为新用户生成匹配其偏好的图像（76% 胜率）。

## 研究背景与动机

**领域现状**：Diffusion-DPO 等 RLHF 技术已显著提升文本到图像扩散模型的生成质量。然而，这些方法优化的是单一奖励函数，代表群体层面的平均偏好，忽略了个体用户的独特口味——有人喜欢明亮色彩，有人偏好居中前景。

**现有痛点**：(1) 为每个用户单独微调一个模型不可扩展；(2) IP-Adapter 等方法通过参考图像控制生成，但受限于单图输入且不直接学习偏好；(3) 偏好数据集（如 Pick-a-Pic）中虽然有用户 ID 标注，但用户特征信息极度稀缺——用户自述偏好往往不准确。

**核心矛盾**：个性化需要理解"用户喜欢什么"，但这种偏好很难用文字或单张图片精确描述，更适合通过成对比较（"A 比 B 好"）来隐式表达。然而，如何从少量成对比较中提取出可用于条件化生成模型的用户表示，是一个未解决的问题。

**本文目标**：设计一个统一框架，让单个扩散模型同时学习多个用户的个性化偏好，并能泛化到训练时未见的新用户。

**切入角度**：将个性化问题转化为条件生成问题——用户偏好作为额外条件注入扩散模型。关键洞察：VLM（如 LLaVA-OneVision）的中间隐状态可以从少量偏好对中有效编码用户特征。

**核心 idea**：用 VLM 处理 4 组偏好示例（每组含文本+偏好图+非偏好图），提取用户嵌入，通过类似 IP-Adapter 的解耦交叉注意力注入 Stable Cascade，以用户条件化的 Diffusion-DPO 目标联合训练。

## 方法详解

### 整体框架
PPD 分两阶段：阶段一（用户嵌入生成）：从偏好数据集中为每个用户采样 $N=4$ 组偏好示例，送入 VLM 提取中间隐状态作为用户嵌入。阶段二（条件化微调）：在 Stable Cascade 的 Stage C 中添加解耦交叉注意力层处理用户嵌入，以用户条件化的 DPO 目标微调（仅训练新增交叉注意力层，冻结预训练模型）。

### 关键设计

1. **VLM 用户嵌入（VLM User Embedding）**:

    - 功能：从少量偏好示例中提取能有效表征用户偏好的向量。
    - 核心思路：使用 LLaVA-OneVision 处理每个用户的 4 组偏好示例（文本+偏好图+非偏好图），提取中间隐状态作为用户嵌入。验证实验表明，基于冻结嵌入训练的用户分类器在 300 个用户中 Top-16 准确率达 90%，证明嵌入能有效区分不同用户的偏好。
    - 设计动机：One-hot 编码无法泛化到新用户。VLM 嵌入具有语义意义（基于偏好示例的视觉内容），天然支持零样本泛化——新用户提供 4 个偏好对即可生成嵌入。在 Bradley-Terry 模型下，偏好对的特征是奖励函数的充分统计量。

2. **个性化 DPO 目标（Personalized DPO Objective）**:

    - 功能：让单个模型同时优化多个用户的个性化偏好，而非群体平均偏好。
    - 核心思路：将标准 Diffusion-DPO 扩展为用户条件化版本 $L_{PPD}(\theta) = -\mathbb{E}_{c, x_0^+, x_0^-, u}[\log\sigma(-\beta T\omega(\lambda_t)\Delta)]$，其中噪声预测器 $\epsilon_\theta(x_t, c, u, t)$ 额外条件化于用户嵌入 $u$。每个训练样本标注了偏好来自哪个用户，模型学习根据不同用户的嵌入生成不同风格的图像。训练时随机 dropout 用户嵌入（零化）作为正则化。
    - 设计动机：标准 DPO 对所有用户使用同一个偏好方向优化，会导致少数派偏好被压制。用户条件化让模型能为每个人维护独立的偏好方向。

3. **解耦交叉注意力条件化（Decoupled Cross-Attention Conditioning）**:

    - 功能：将用户嵌入注入扩散模型，同时保持文本条件化不受影响。
    - 核心思路：沿用 IP-Adapter 的设计，在每个文本交叉注意力层旁添加新的交叉注意力层处理用户嵌入：$Z' = \text{Softmax}(\frac{QK^T}{\sqrt{d}})V + \text{Softmax}(\frac{Q(K')^T}{\sqrt{d}})V'$，其中 $K' = u_t W_k'$，$V' = u_t W_v'$。仅训练新增的 $W_k', W_v'$ 参数，冻结预训练模型。
    - 设计动机：解耦设计确保用户偏好和文本语义是独立的条件化通道，用户偏好影响风格/色彩/构图等高层属性，文本控制语义内容。仅训练新增层保证了训练效率和基础模型质量。

### 损失函数 / 训练策略
使用 Pick-a-Pic v2 数据集（58K 文本提示、0.8M 图像对、5K 用户），以用户条件化的 Diffusion-DPO 目标仅微调新增交叉注意力层。AdamW 优化器，有效批大小 768 对，学习率 $1\times10^{-5}$，训练 1 个 epoch。

## 实验关键数据

### 主实验
多奖励优化（合成用户实验，每个奖励函数作为一个"用户"）：

| 方法 | CLIP↑ | Aesthetic↑ | HPS↑ |
|------|-------|-----------|------|
| Stable Cascade | 31.97 | 5.33 | 23.87 |
| Diffusion-DPO | 32.48 | 5.46 | 25.96 |
| SFT | 32.26 | 5.56 | 25.78 |
| **PPD (ours)** | **32.66** | **5.92** | **27.51** |
| DPO (CLIP only) | 32.96 | - | - |
| DPO (Aesthetic only) | - | 6.42 | - |
| DPO (HPS only) | - | - | 28.61 |

### 消融实验
| 配置 | 说明 |
|------|------|
| One-hot 用户编码 | 无法泛化到新用户 |
| VLM 用户嵌入 | 可泛化，Top-16 分类准确率 90% |
| w/o 用户 dropout | 过拟合训练用户 |
| w/ 用户 dropout | 更好的泛化 |
| Pick-a-Pic 真实用户 | 76% 胜率 vs Stable Cascade |

### 关键发现
- PPD 同时优化三个奖励函数且接近各自的单独上界，证明单模型可以有效容纳多种偏好
- 可以在推理时通过线性插值奖励权重平滑地在不同偏好之间过渡（图 4）
- 在 Pick-a-Pic 真实用户场景中，仅需 4 个偏好对就达到 76% 对 Stable Cascade 的胜率
- VLM 嵌入在 300 用户中的分类准确率远超随机基线，证明了其偏好区分能力

## 亮点与洞察
- **偏好即条件**的范式转换：将个性化偏好从"优化目标"转变为"生成条件"，使得单模型服务多用户成为可能。推理时只需切换用户嵌入，无需重新训练
- **VLM 作为偏好编码器**：利用 VLM 的多图理解能力从偏好对中提取用户特征，这一思路可以迁移到 LLM 个性化、推荐系统等领域
- **奖励函数插值**：推理时可以在不同偏好之间平滑插值，提供了前所未有的生成控制灵活性

## 局限与展望
- 目前仅在 Stable Cascade 上验证，未测试 SDXL、Flux 等更新架构
- 4 个偏好对对复杂偏好的描述可能不够充分
- 用户嵌入的可解释性有限——难以理解模型"学到了什么偏好"
- 多用户同时训练时的负迁移风险未被深入分析

## 相关工作与启发
- **vs Diffusion-DPO**: DPO 优化群体平均偏好，PPD 优化个体偏好。PPD 在同时优化多个奖励时接近 DPO 的单奖励上界
- **vs IP-Adapter**: IP-Adapter 从参考图像提取风格信息，PPD 从偏好对提取偏好信息。PPD 的条件化信息更抽象（偏好而非风格）
- **vs PRISM/Pluralistic Alignment**: LLM 领域的多元价值对齐工作。PPD 将类似思想引入图像生成领域

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "偏好即条件"的框架设计、VLM 偏好编码器、个性化 DPO 三者结合非常新颖
- 实验充分度: ⭐⭐⭐⭐ 合成和真实用户实验充分，但缺乏大规模用户研究
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法推导严谨
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型个性化开辟了新方向，框架通用且实用

<!-- RELATED:START -->

## 相关论文

- [Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward](reward_fine-tuning_two-step_diffusion_models_via_learning_differentiable_latent-.md)
- [Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)

<!-- RELATED:END -->
