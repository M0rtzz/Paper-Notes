---
title: >-
  [论文解读] AdaWorld: Learning Adaptable World Models with Latent Actions
description: >-
  [ICML 2025][自监督学习][世界模型] 提出 AdaWorld——通过从视频中自监督提取潜在动作（latent actions）进行动作感知预训练，构建高度可适应的世界模型，支持零样本动作迁移和少量交互快速适应新环境。
tags:
  - ICML 2025
  - 自监督学习
  - 世界模型
  - 潜在动作
  - 自监督
  - 视频预训练
  - 动作迁移
---

# AdaWorld: Learning Adaptable World Models with Latent Actions

**会议**: ICML 2025  
**arXiv**: [2503.18938](https://arxiv.org/abs/2503.18938)  
**代码**: https://adaptable-world-model.github.io  
**领域**: 自监督学习  
**关键词**: 世界模型, 潜在动作, 自监督, 视频预训练, 动作迁移

## 一句话总结
提出 AdaWorld——通过从视频中自监督提取潜在动作（latent actions）进行动作感知预训练，构建高度可适应的世界模型，支持零样本动作迁移和少量交互快速适应新环境。

## 研究背景与动机

**领域现状**：世界模型旨在学习动作可控的未来预测，对智能体开发至关重要。现有方法依赖大量动作标注数据和昂贵训练来获得动作可控性。

**现有痛点**：(a) 不同环境的动作格式各异，难以定义统一格式; (b) 适应新环境需要重新收集大量动作标签和训练; (c) 仅用无动作视频预训练的世界模型缺乏动作可控性。

**核心矛盾**：如何在预训练阶段就引入动作信息，同时不依赖显式动作标签？

**本文目标**：构建可快速适应新环境的世界模型。

**切入角度**：从视频帧对中自监督提取潜在动作——用信息瓶颈迫使编码器只保留最关键的帧间变化（即动作信息），去除上下文（颜色、纹理等）。

**核心 idea**：潜在动作是上下文无关的，可跨环境迁移——给一个演示就能将动作迁到新场景。

## 方法详解

### 整体框架
两个组件：
1. **潜在动作自编码器**：从无标签视频中提取潜在动作
2. **自回归世界模型**：以潜在动作为条件预测下一帧

### 关键设计

1. **潜在动作自编码器（Latent Action Autoencoder）**:

    - 功能：从两连续帧 $f_t, f_{t+1}$ 中提取紧凑的潜在动作 $\tilde{a}$
    - 核心思路：编码器用时空 Transformer 从帧对中提取潜在动作，解码器基于 $\tilde{a}$ 和 $f_t$ 预测 $f_{t+1}$。使用 $\beta$-VAE 的信息瓶颈，迫使 $\tilde{a}$ 只编码最关键的帧间变化
    - 设计动机：信息瓶颈使潜在动作自动从上下文中解耦——相比像素空间的巨大维度，潜在动作极其紧凑

2. **动作感知预训练**:

    - 功能：用基于 Stable Video Diffusion 的世界模型，以潜在动作为条件预测下一帧
    - 核心思路：将潜在动作与时间步嵌入和 CLIP 图像嵌入拼接，作为扩散模型的条件
    - 设计动机：预训练中就学会了"不同潜在动作导致不同状态转移"，适应新环境只需找到动作映射

3. **适应机制**:

    - 零样本迁移：给一个演示视频，提取潜在动作并在新场景中复用
    - 少量交互适应：有动作标签时，用潜在动作编码器找到映射，少量微调即可

### 损失函数 / 训练策略
- 潜在动作自编码器：$\beta$-VAE 目标（重建 + KL 散度）
- 世界模型：EDM 扩散损失 + 噪声增强（缓解长期漂移）
- 大规模多样化视频数据集预训练

## 实验关键数据

### 主实验

| 任务 | AdaWorld | 无动作预训练基线 | 改进 |
|------|---------|---------------|------|
| 动作迁移（跨场景） | 可行 | 不可行 | 质的突破 |
| 50次交互适应 | FVD 85.2 | FVD 142.3 | 40%↓ |
| 视觉规划成功率 | 72.4% | 48.1% | +24.3% |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| 无潜在动作（无动作预训练） | 差 | 无动作可控性 |
| 离散潜在动作 | 表达力不足 | 无法组合动作 |
| 连续潜在动作（本文） | 最优 | 支持插值和组合 |
| 大 $\beta$ | 解耦好但表达力弱 | 折中 |
| 小 $\beta$ | 表达力强但解耦差 | 折中 |

### 关键发现
- 潜在动作是上下文无关的——同一动作可从一个场景迁移到完全不同的场景
- 连续潜在空间支持动作组合（两个动作的平均产生组合效果）
- 仅 50 次交互的适应效果远超从头训练

## 亮点与洞察
- **信息瓶颈 = 动作解耦**的设计极其精巧——利用 VAE 的压缩特性自动分离动作和上下文
- 潜在动作的连续空间支持语义插值和组合，暗示了更通用的动作表示
- 与 SVD 的集成使模型继承了强大的视频生成先验

## 局限与展望
- 潜在动作只捕捉两帧间的局部变化，长程规划能力有限
- 预训练数据主要来自自动生成的游戏环境，真实世界场景的泛化待验证
- $\beta$ 的选择需要手动调节

## 相关工作与启发
- **vs UniSim/GameGen**: 需要动作标签训练，AdaWorld 自监督提取
- **vs GAIA-1**: 需要大量标注数据，AdaWorld 从视频自动学习
- 对通用世界模型和具身智能研究有启发

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 潜在动作预训练范式新颖且有力
- 实验充分度: ⭐⭐⭐⭐ 多环境、动作迁移、规划任务
- 写作质量: ⭐⭐⭐⭐⭐ 图示精美，方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 世界模型适应性的重要突破

<!-- RELATED:START -->

## 相关论文

- [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [CheXWorld: Image World Modeling for Radiograph Representation Learning](../../CVPR2025/self_supervised/chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)
- [Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)
- [Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)
- [L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)

<!-- RELATED:END -->
