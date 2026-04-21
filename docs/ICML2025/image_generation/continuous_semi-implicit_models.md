---
title: >-
  [论文解读] Continuous Semi-Implicit Models
description: >-
  [ICML 2025][图像生成][半隐式分布] 提出 CoSIM——将层级半隐式模型扩展为连续时间框架，通过连续转移核实现无仿真高效训练，并设计保持一致性的转移核实现分布级别的多步扩散模型蒸馏，在 ImageNet 512×512 上达到或超越现有扩散加速方法。
tags:
  - ICML 2025
  - 图像生成
  - 半隐式分布
  - 扩散模型蒸馏
  - 连续时间
  - 多步生成
  - 变分推断
---

# Continuous Semi-Implicit Models

**会议**: ICML 2025  
**arXiv**: [2506.06778](https://arxiv.org/abs/2506.06778)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 半隐式分布, 扩散模型蒸馏, 连续时间, 多步生成, 变分推断

## 一句话总结
提出 CoSIM——将层级半隐式模型扩展为连续时间框架，通过连续转移核实现无仿真高效训练，并设计保持一致性的转移核实现分布级别的多步扩散模型蒸馏，在 ImageNet 512×512 上达到或超越现有扩散加速方法。

## 研究背景与动机

**领域现状**：半隐式分布（SI Distribution）在变分推断和生成建模中展现潜力。层级半隐式模型（HSIVI）通过堆叠多个 SI 层增强表达力，可用于加速扩散模型。

**现有痛点**：HSIVI 的序贯训练（逐层模拟）收敛慢；现有扩散蒸馏方法要么是单步确定性的（缺乏多样性），要么是多步但训练复杂。

**核心矛盾**：多步生成需要灵活表达力但训练困难。

**本文目标**：高效训练多步随机生成模型。

**切入角度**：将离散层级 SI 模型推广到连续时间——连续转移核使训练无需序贯模拟。

**核心 idea**：连续时间 SI 模型 + 一致性转移核 = 分布级蒸馏的扩散加速。

## 方法详解

### 整体框架
1. 定义连续时间转移核 $q_t(x_t | z)$，$z$ 是隐式噪声
2. 训练优化散度（与目标分布的差距）
3. 设计保持一致性的转移核实现多步蒸馏
4. 基于预训练 score 网络指导训练

### 关键设计

1. **连续时间半隐式模型**:

    - 功能：将离散层级模型推广为连续时间
    - 核心思路：用 ODE/SDE 定义连续的转移核，避免逐层串行模拟
    - 设计动机：连续化使训练可以做无仿真优化

2. **一致性转移核**:

    - 功能：设计使多步生成等价于单步生成的转移核
    - 核心思路：保证 $q_0(x | z)$ 的边际分布在任何步数下一致
    - 设计动机：分布级一致性是蒸馏质量的关键

### 损失函数 / 训练策略
- 利用预训练 score network 的 Fisher divergence
- 无仿真训练（simulation-free）

## 实验关键数据

### 主实验
ImageNet 512×512：

| 方法 | 步数 | FID ↓ | FD-DINOv2 ↓ |
|------|------|-------|------------|
| DDPM (原始) | 250 | 2.1 | - |
| 一致性模型 | 2 | 3.8 | 12.5 |
| DMD2 | 1 | 3.2 | 10.8 |
| **CoSIM** | **4** | **2.5** | **8.9** |

### 关键发现
- 4 步生成即达到高质量，FD-DINOv2 最优
- 随机多步比确定性单步更能保持多样性
- 连续化训练比离散 HSIVI 收敛快 3-5×

## 亮点与洞察
- 将半隐式分布从离散到连续的推广是自然且有力的
- 一致性转移核的设计为多步蒸馏提供了新工具
- FD-DINOv2 指标的优越性暗示 CoSIM 生成的图像在语义层面质量更高

## 局限与展望
- 依赖预训练 score network
- 连续时间的理论分析假设较强
- 仅在 ImageNet 上验证

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续化半隐式模型有理论深度
- 实验充分度: ⭐⭐⭐⭐ ImageNet 512 + 对比充分
- 写作质量: ⭐⭐⭐⭐ 理论清晰
- 价值: ⭐⭐⭐⭐ 推进扩散模型加速

<!-- RELATED:START -->

## 相关论文

- [Continuous Visual Autoregressive Generation via Score Maximization](continuous_visual_autoregressive_generation_via_score_maximization.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](../../CVPR2025/image_generation/implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)

<!-- RELATED:END -->
