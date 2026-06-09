---
title: >-
  [论文解读] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers
description: >-
  [CVPR 2026 (HOW Workshop)][自监督学习][register tokens] 通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激…
tags:
  - "CVPR 2026 (HOW Workshop)"
  - "自监督学习"
  - "register tokens"
  - "Transformer"
  - "zero-ablation"
  - "DINO"
  - "interpretability"
---

# Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers

**会议**: CVPR 2026 (HOW Workshop)  
**arXiv**: [2604.14433](https://arxiv.org/abs/2604.14433)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: register tokens, vision transformers, zero-ablation, DINO, interpretability

## 一句话总结

通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激活"而非图像特定值。

## 研究背景与动机

零消融（将 token 激活替换为零向量）是探测 ViT 中 token 功能的常用方法。在 DINOv2+registers 和 DINOv3 中，清零 register token 导致分类下降高达 36.6pp、分割下降 30.9pp，表面上表明 register 不可或缺。然而零向量相对于原生 register 激活是不合理的分布外输入，可能夸大了真实的内容依赖性。这类似于神经科学中的损毁研究混淆——损伤通过互联回路级联传播产生过度定位的假象。

## 方法详解

### 整体框架

对 DINOv2、DINOv2+registers、DINOv3 三个模型系列（ViT-S 和 ViT-B）应用 hook-based 消融，在每个 block 输出后替换 [CLS] 或 register 隐藏状态。在分类、检索、对应和分割四个下游任务上对比零消融与三种替换控制。

### 关键设计

1. **三种替换控制实验**: (1) 均值替换：使用 5000 张 ImageNet 图像校准的逐层数据集均值激活；(2) 噪声替换：均值和方差匹配的逐层高斯噪声；(3) 跨图像 register 洗牌：在批次内随机排列 register 激活，保留真实激活结构但打破图像特定内容。

2. **分布内验证**: 通过逐 patch 余弦相似度分析确认三种替换确实扰动了内部表示（余弦相似度 0.95-0.999），排除了"替换未改变特征"的可能性。同时通过 JS 散度量化零消融造成的分布偏移是替换控制的数十到数百倍。

3. **有效秩分析与注意力流**: Register 压缩了 patch 几何（有效秩从 13.5 降至 4.0），DINOv3 压缩最显著。注意力流分析显示 register 注意力从中间层逐渐积累，但分类依赖性在第 10-11 层突然出现。

### 损失函数 / 训练策略

本文为分析性工作，不涉及训练。所有评估在冻结特征上进行。

## 实验关键数据

### 主实验

| 条件 | DINOv2+R 分类 | DINOv3 分类 | DINOv2+R 分割 | DINOv3 分割 |
|------|-------------|------------|-------------|------------|
| Full | 67.3% | 62.0% | 基线 | 基线 |
| Zero registers | -18.9pp | **-36.6pp** | -9.6pp | **-30.9pp** |
| Mean-sub | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |
| Noise-sub | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |
| Shuffle | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |

### 关键发现

- 仅零消融产生性能下降，三种合理替换均保持所有任务的性能
- Register 缓冲了密集特征对 [CLS] 的依赖（分割下降 37pp vs <1pp）
- 结果在 ViT-B 规模上完全复现

## 亮点与洞察

- 优雅地揭示了零消融的方法论缺陷——注入分布外输入而非移除功能
- 与神经科学中的损毁研究类比恰当且有教育意义
- 结论清晰：register 功能如预期的"上下文通道"，精确内容非必需

## 局限与展望

- 仅在冻结特征评估上测试，微调后的模型可能表现不同
- 仅测试了 DINO 系列模型，其他自监督 ViT 的行为可能不同
- Workshop 论文篇幅有限，部分分析深度受限

## 相关工作与启发

- 为所有使用零消融进行功能探测的工作提供了重要方法论警示
- 激活替换的"分布内控制"思想可推广到 NLP 中的机制可解释性
- Register token 的"结构性通道"角色为 ViT 设计提供指导

## 评分

7/10 — 方法论贡献清晰且重要，但作为 Workshop 论文规模有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](../../CVPR2025/self_supervised/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)

</div>

<!-- RELATED:END -->
