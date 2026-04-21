---
title: >-
  [论文解读] Decoupling Training-Free Guided Diffusion by ADMM
description: >-
  [CVPR 2025][图像生成][训练免费引导] 提出 ADMMDiff，用 ADMM 框架将训练免费条件生成中的扩散项和引导项解耦为两个变量（x 扩散, z 引导, 约束 x=z），用近端算子分别处理，消除了手动调权重的需求并提供收敛保证，在分割/素描/文本引导生成上大幅超越 DPS 和 MPGD。
tags:
  - CVPR 2025
  - 图像生成
  - 训练免费引导
  - ADMM优化
  - 条件生成
  - 收敛保证
  - 非线性引导
---

# Decoupling Training-Free Guided Diffusion by ADMM

**会议**: CVPR 2025  
**arXiv**: [2411.12773](https://arxiv.org/abs/2411.12773)  
**代码**: 无（未提及）  
**领域**: 扩散模型  
**关键词**: 训练免费引导, ADMM优化, 条件生成, 收敛保证, 非线性引导

## 一句话总结
提出 ADMMDiff，用 ADMM 框架将训练免费条件生成中的扩散项和引导项解耦为两个变量（x 扩散, z 引导, 约束 x=z），用近端算子分别处理，消除了手动调权重的需求并提供收敛保证，在分割/素描/文本引导生成上大幅超越 DPS 和 MPGD。

## 研究背景与动机

**领域现状**：训练免费的条件扩散生成（如 DPS, FreeDoM, MPGD）通过在去噪过程中加入条件引导梯度来实现条件生成，无需重新训练模型。但这些方法需要手动调节引导权重，且缺乏理论收敛保证。

**现有痛点**：扩散项和引导项目标不同——扩散项生成高质量图像，引导项满足条件约束。两者的权重比例对结果影响巨大，手动调参费时且不同条件需要不同最优权重。现有方法本质上是启发式的。

**核心矛盾**：需要平衡扩散质量和条件满足度，但没有原则性的方法来自动确定这个平衡。

**本文目标** 提供一种有理论保证的、无需手动调权重的训练免费条件生成框架。

**切入角度**：引入辅助变量 z，将问题转化为约束优化 min f(x) + g(z) s.t. x=z，用 ADMM 框架交替优化。f 的近端算子用反向扩散步近似（有理论证明），g 的近端算子用梯度下降求解。

**核心 idea**：用 ADMM 框架将扩散生成的采样步和条件引导的优化步解耦为分别处理的近端算子，自动平衡两者且有亚线性收敛保证。

## 方法详解

### 关键设计

1. **ADMM 双变量解耦**：引入 z 变量和约束 x=z，x 负责扩散质量（近端算子=反向扩散步），z 负责条件满足（近端算子=引导梯度下降）。增广拉格朗日参数 ρ 自动平衡两者（需满足 ρ ≤ 1/(6L)）

2. **扩散近端算子近似**：证明反向扩散步是扩散项 f(x) 的近端算子的合理近似（Proposition 1），建立了理论基础

3. **收敛保证**：在 L-光滑假设下证明亚线性收敛到稳定点，是该领域首个有收敛理论的训练免费方法

## 实验关键数据（CelebA-HQ 非线性引导）

| 方法 | Seg距离↓ | Seg FID↓ | Sketch距离↓ | Sketch FID↓ |
|------|---------|---------|------------|------------|
| DPS | 2199.8 | 57.38 | 50.74 | 67.21 |
| MPGD | 1922.5 | 43.97 | 35.32 | 60.56 |
| **ADMMDiff** | **1586.2** | **30.18** | **32.28** | **42.43** |

CLIP Score (文本引导): ADMMDiff **26.85** vs FreeDoM 25.9, MPGD 25.1

## 局限与展望
- 每步需 $K_t$ 次内循环梯度下降，计算量高于单步方法
- L-光滑假设对某些引导函数可能不成立
- ρ 的约束条件需要预知 L 值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ ADMM 框架解耦扩散和引导是优雅的理论贡献
- 实验充分度: ⭐⭐⭐⭐ 多种引导类型+理论分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，命题和证明清晰
- 价值: ⭐⭐⭐⭐ 为训练免费引导提供了原则性框架

<!-- RELATED:START -->

## 相关论文

- [TKG-DM: Training-Free Chroma Key Content Generation Diffusion Model](tkg-dm_training-free_chroma_key_content_generation_diffusion_model.md)
- [Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)

<!-- RELATED:END -->
