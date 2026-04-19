---
title: >-
  [论文解读] Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales
description: >-
  [CVPR 2025][图像生成][Winograd量化] 提出完全量化 Winograd 卷积的方法——通过可学习的对角缩放矩阵均衡 Winograd 域输出的动态范围差异，仅用随机高斯噪声（无真实数据）微调缩放参数，在 InstaFlow/SD v1.5 等扩散模型上实现近无损的 W8A8 Winograd 加速（CPU 卷积层加速 31.3%）。
tags:
  - CVPR 2025
  - 图像生成
  - Winograd量化
  - 无数据微调
  - 组级量化
  - CPU加速
  - 扩散模型部署
---

# Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales

**会议**: CVPR 2025  
**arXiv**: [2412.19867](https://arxiv.org/abs/2412.19867)  
**代码**: 无（未提及）  
**领域**: 模型压缩 / 扩散模型  
**关键词**: Winograd量化, 无数据微调, 组级量化, CPU加速, 扩散模型部署

## 一句话总结
提出完全量化 Winograd 卷积的方法——通过可学习的对角缩放矩阵均衡 Winograd 域输出的动态范围差异，仅用随机高斯噪声（无真实数据）微调缩放参数，在 InstaFlow/SD v1.5 等扩散模型上实现近无损的 W8A8 Winograd 加速（CPU 卷积层加速 31.3%）。

## 研究背景与动机

**领域现状**：Winograd 变换可将 3×3 卷积的乘法量降至 5/9（F(4,3)）或 4/9（F(6,3)），是 CPU 推理加速的核心技术。但 Winograd 域的中间数据动态范围极大，传统量化在 Winograd 域几乎完全失败。

**现有痛点**：标准 Winograd + W8A8 量化导致灾难性质量崩溃（InstaFlow FID 从 23.00 暴涨到 326.96）。原因是 Winograd 变换后不同像素位置的数值范围差异可达数十倍，统一量化范围无法覆盖。

**核心矛盾**：Winograd 加速需要在变换域做定点乘法，但变换域的极端动态范围使定点量化失效。

**本文目标** 让 Winograd 卷积可以安全地全流程 8-bit 量化，同时不需要任何领域特定数据。

**切入角度**：Winograd 变换矩阵 $B$ 和 $G$ 可以分解为 $S_B \cdot B'$ 和 $S_G \cdot G'$，其中 $S_B, S_G$ 是对角缩放矩阵。通过优化这些缩放参数来均衡变换域的动态范围——只用随机高斯噪声作为输入，完全无需真实数据。

**核心 idea**：微调 Winograd 变换矩阵的对角缩放参数均衡动态范围，用随机噪声做无数据优化，实现全量化 Winograd 卷积。

## 方法详解

### 关键设计

1. **可学习缩放矩阵**：Winograd 变换 $B^T d B$ 中 $B$ 可分解含对角缩放矩阵 $S_B$，优化 $S_B$ 使变换后各位置的数值范围更均衡，从而适合统一量化

2. **无数据优化**：仅用随机高斯噪声作为输入微调 $S_B, S_G$，不需要任何领域特定数据。这使方法适用于基础模型（扩散模型等不应看到特定领域数据以防过拟合）

3. **组级量化**：在 Winograd 域进行分组量化（group-wise），不同组有独立的缩放因子，进一步细化量化精度。开发优化的 CPU 向量指令核实现分组量化的矩阵乘

## 实验关键数据

| 配置 | InstaFlow FID↓ | CLIP↑ |
|------|-------------|-------|
| FP16 基线 | 23.00 | 30.19 |
| W8A8（无 Winograd） | 23.04 | 30.16 |
| W8A8 Winograd（标准） | 326.96 | 5.95 |
| **W8A8 Winograd + 可学习缩放** | **26.58** | **29.65** |

从完全崩溃（FID 327）恢复到近无损（FID 26.58 vs 基线 23.00）。CPU 卷积层加速 31.3%，端到端加速 12.8%。ImageNet 分类比 SOTA Winograd PTQ 高 +1.6-2.6%。

## 评分
- 新颖性: ⭐⭐⭐⭐ 可学习缩放+无数据优化的组合简洁有效
- 实验充分度: ⭐⭐⭐⭐ 扩散模型+分类模型+多种 Winograd 配置
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻
- 价值: ⭐⭐⭐⭐⭐ 解决了 Winograd 量化的核心瓶颈，对 CPU 部署有重大价值

<!-- RELATED:START -->

## 相关论文

- [Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)
- [BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [Improving Editability in Image Generation with Layer-wise Memory](improving_editability_in_image_generation_with_layer-wise_memory.md)
- [MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)

<!-- RELATED:END -->
