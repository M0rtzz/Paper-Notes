---
title: >-
  [论文解读] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation
description: >-
  [AAAI 2026][医学图像][神经架构搜索] 提出 DeNAS-ViT，首次将 NAS 应用于 ViT 的 Token 级搜索实现超声图像分割的多尺度特征提取优化，并设计基于 NAS 约束的半监督学习框架（网络独立性损失+层次对比损失+阶段式优化），在有限标注数据下达到 SOTA。 领域现状：超声图像分割对心脏疾病诊断…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "神经架构搜索"
  - "Transformer"
  - "超声分割"
  - "半监督学习"
  - "Token级搜索"
---

# DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation

**会议**: AAAI 2026  
**arXiv**: [2407.04203](https://arxiv.org/abs/2407.04203)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 神经架构搜索, Vision Transformer, 超声分割, 半监督学习, Token级搜索

## 一句话总结
提出 DeNAS-ViT，首次将 NAS 应用于 ViT 的 Token 级搜索实现超声图像分割的多尺度特征提取优化，并设计基于 NAS 约束的半监督学习框架（网络独立性损失+层次对比损失+阶段式优化），在有限标注数据下达到 SOTA。

## 研究背景与动机

**领域现状**：超声图像分割对心脏疾病诊断至关重要，深度学习方法（UNet++、TransUNet、EfficientViT）取得进展但依赖手动设计架构。NAS 可自动优化架构，但现有 NAS 方法仅在模块级搜索（选择卷积或Transformer块），忽略了模块内部更细粒度的操作。

**现有痛点**：
   - 手工架构设计在先验知识不足时增益有限
   - 超声数据标注稀缺，而 NAS 天然需要大量数据
   - 模块级 NAS 搜索精度低，容易选择最复杂的操作（如 ViT），使搜索失去意义

**核心矛盾**：NAS 需要大量数据来搜索最优架构，但超声分割恰好缺少标注数据——两者对数据需求的冲突如何解决？

**本文目标** 设计高效的token级NAS + 数据高效的半监督学习框架。

**切入角度**：在 ViT 的注意力计算之前做 NAS——搜索最优的多尺度 token 表示（而非搜索整个模块），同时用 NAS 引导的半监督学习减少对标注数据的依赖。

**核心 idea**：ViT 内部的 Token 级 NAS 搜索 + NAS 约束驱动的半监督协同训练。

## 方法详解

### 整体框架
- **Efficient NAS-ViT 模块**：在 ViT 的 Q/K/V token 上做 NAS 搜索多尺度表示
- **NAS Backbone**：编码器（层次NAS+NAS-ViT cell）+ 解码器（NAS cell）
- **NAS-based SSL**：两个共享 NAS backbone 但参数独立的网络做协同训练，加入独立性损失和层次对比损失

### 关键设计

1. **Efficient NAS-ViT 模块**:

    - 功能：在注意力计算前搜索最优的多尺度 token 表示
    - 核心思路：对 Q/K/V token 用部分通道连接+连续松弛做可微搜索：$\{Q'/K'/V'\} = (1-P) \odot \{Q/K/V\} + \sum_{O_i} \frac{\exp(\alpha_i)}{\sum_j \exp(\alpha_j)} O_i(P \odot \{Q/K/V\})$
    - 设计动机：DAST 把整个 ViT 层作为候选操作导致计算量大且搜索无意义（模型总会选最复杂的 ViT）。在 token 级搜索更细粒度且减少参数开销

2. **层次 NAS Backbone**:

    - 编码器：cell级搜索（NAS-ViT处理多尺度token）+ 层级搜索（不同分辨率路径间用 softmax 加权聚合）
    - 解码器：U型结构，每层做独立NAS cell搜索
    - 6种分辨率（r=1,2,4,8,16,32）捕获多尺度特征

3. **NAS-based 约束驱动的半监督学习**:

    - **网络独立性损失**：鼓励两个网络学到互补表示，通过卷积层权重的余弦相似度度量
    - **层次对比损失**：在解码器NAS cell的多分辨率输出上计算不确定性，让低质量特征对齐高质量特征
    - **阶段式优化**：分阶段引入不同约束确保训练稳定

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{uns} + \mathcal{L}_{ind} + \mathcal{L}_{con}$。DARTS 风格的双层优化搜索架构参数 $\alpha, \beta, \gamma$ 和网络权重 $\theta$。

## 实验关键数据

### 主实验（3个数据集，100%标注）

| 方法 | HMC-QU DSC | CAMUS DSC | CETUS DSC |
|------|-----------|-----------|-----------|
| UNet++ | 0.899 | 0.919 | 0.952 |
| nnU-Net | 0.908 | 0.922 | 0.958 |
| TransFuse | 0.903 | 0.923 | 0.957 |
| Se2NAS† | 0.907 | 0.920 | 0.955 |
| **DeNAS-ViT† (Ours)** | **~0.920** | **~0.930** | **~0.962** |

(†表示使用SSL)。DeNAS-ViT 在所有数据集上超过所有基线（含NAS和SSL方法）。

### 消融实验
- Efficient NAS-ViT vs DAST(模块级搜索)：更小参数+更好性能
- 各SSL约束贡献：独立性损失和对比损失各自提升约0.5-1% DSC
- 在10%标注数据下仍保持竞争力

### 关键发现
- Token级搜索比模块级搜索更高效且泛化性更好
- NAS+SSL组合比单独使用任一方法提升更大
- 在非超声数据集上也表现良好，具有通用性

## 亮点与洞察
- **Token级 NAS 搜索**的颗粒度选择很巧妙——比模块级更精细，比像素级又不过于昂贵，恰好在 ViT 的 Q/K/V 层面做搜索
- **NAS 和 SSL 的自然结合**——NAS 生成的不同架构自然提供了协同训练所需的多样性，两者互相增强
- **阶段式优化策略**是实际工程中的重要经验——直接联合优化所有目标往往不稳定

## 局限与展望
- NAS 搜索过程本身计算代价高，论文未详细讨论搜索时间
- 仅在2D超声上验证，3D体积数据的扩展有待探索
- 半监督部分的超参数（各损失权重、阶段切换时机）较多

## 相关工作与启发
- **vs EfficientViT**: EfficientViT 用固定多尺度卷积处理 token，DeNAS-ViT 用 NAS 搜索最优 token 表示——更灵活
- **vs DAST**: DAST 在模块级搜索中 ViT 总被选中，DeNAS-ViT 的 token 级搜索避免了这个问题
- **vs Se2NAS**: Se2NAS 也结合 NAS+SSL 但無额外约束，DeNAS-ViT 的独立性+对比损失带来更好性能

## 评分
- 新颖性: ⭐⭐⭐⭐ Token级NAS + NAS-SSL 统一框架，首次在超声分割中实现
- 实验充分度: ⭐⭐⭐⭐⭐ 3个数据集+12个基线+充分消融+泛化验证
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 为数据稀缺的医学图像分割提供了NAS+SSL的有效方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](../../CVPR2026/medical_imaging/medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[AAAI 2026\] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)
- [\[CVPR 2026\] Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes](../../CVPR2026/medical_imaging/extending_zach-vit_to_robust_medical_imaging_corruption_and_adversarial_stress_t.md)

</div>

<!-- RELATED:END -->
