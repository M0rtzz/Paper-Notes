---
title: >-
  [论文解读] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You
description: >-
  [NeurIPS 2025][多模态VLM][多模态对齐] 提出 STRUCTURE 正则化和基于表示相似度的层选择策略，仅用少量配对数据（数万对，不到常规方法的1%）即可实现冻结单模态基础模型的高质量跨模态对齐，在24个零样本分类和检索基准上平均提升51.6%和91.8%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态对齐"
  - "低数据学习"
  - "结构正则化"
  - "参数冻结对齐"
  - "层选择"
---

# STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You

**会议**: NeurIPS 2025  
**arXiv**: [2506.16895](https://arxiv.org/abs/2506.16895)  
**代码**: [https://brbiclab.epfl.ch/projects/structure](https://brbiclab.epfl.ch/projects/structure)  
**领域**: 多模态VLM  
**关键词**: 多模态对齐, 低数据学习, 结构正则化, 参数冻结对齐, 层选择

## 一句话总结
提出 STRUCTURE 正则化和基于表示相似度的层选择策略，仅用少量配对数据（数万对，不到常规方法的1%）即可实现冻结单模态基础模型的高质量跨模态对齐，在24个零样本分类和检索基准上平均提升51.6%和91.8%。

## 研究背景与动机
单模态基础模型（LLM、DINOv2、AlphaFold 等）在各自领域已非常强大，但许多应用需要将不同模态映射到共享表示空间。CLIP 等多模态模型依赖 4 亿配对数据训练，而在医疗、生物等领域，获取大量高质量配对数据极其昂贵。

**现有对齐方法的困境**：
- 有监督对齐方法（Linear/MLP mapping）需要数千万配对样本
- 无监督方法（CKA 等）不利用配对数据，仅做样本级匹配，无法构建共享嵌入空间
- Platonic 表示假设（模型趋向相似内部表示）为低数据对齐提供理论基础，但如何利用仍是开放问题

**核心问题**：能否用仅数万个配对样本（<1%常规数据量）将两个冻结的单模态编码器对齐到共享空间？

**本文思路**：保留预训练空间中丰富的邻域结构——这些结构编码了数百万/亿样本的关系信息。通过正则化防止对齐过程中的"过度扭曲"，同时选择最适合对齐的中间层而非默认使用最后一层。

## 方法详解

### 整体框架
冻结两个预训练编码器，学习轻量级对齐函数（线性层/MLP）将各模态映射到共享空间。在任何现有对齐目标 $\mathcal{L}_A$ 上添加 STRUCTURE 正则化，并基于表示相似度选择最佳层对齐。

### 关键设计

1. **STRUCTURE 正则化**：保持多尺度邻域几何

    - 对预训练空间 $\mathcal{X}$ 和对齐空间 $\mathcal{A}$ 中的样本，先 $\ell_2$ 归一化再中心化
    - 计算带温度的相似度矩阵 $S_X, S_A$，按行 softmax 得概率分布 $P_X, P_A$
    - 通过矩阵幂运算捕获 $l$-hop 关系：$P_X^{(l)} = (P_X)^l$（类比随机游走）
    - 用 Jensen-Shannon 散度度量各层级的结构差异
    - 最终正则化器为加权平均（低层级权重更大以对抗集中分布）：
    $\mathcal{R}_S^{(L)}(X,A) = \frac{1}{L}\sum_{l=1}^L \frac{\text{JS}(P_X^{(l)}, P_A^{(l)})}{l}$
    - 总损失：$\mathcal{L} = \mathcal{L}_A + \lambda(\mathcal{R}_S(X_1, f_1(X_1)) + \mathcal{R}_S(X_2, f_2(X_2)))$

2. **基于相似度的层选择**：

    - 先前工作默认对齐最后一层，但中间层可能具有更高的跨模态相似度
    - 用mutual kNN 度量不同层对之间的表示相似度（在小规模配对数据上计算）
    - 实验发现层相似度与下游性能高度相关（Spearman 秩相关系数 $\rho$ 很高）
    - 选择最高相似度的层对进行对齐

3. **理论保证**：

    - 泛化界：$|\hat{\mathcal{R}}_N - \mathcal{R}^*| \leq \mathcal{O}(1/\sqrt{N})$
    - STRUCTURE 对全局缩放、平移和正交旋转不变，仅依赖内在层级关系结构

### 损失函数 / 训练策略
- 对齐目标：对称对比损失 $\mathcal{L}_C$（CLIP 风格）+ STRUCTURE 正则化
- 使用 COCO 训练集（80K 配对）进行训练
- 适用于 Linear、MLP、CSA 等不同对齐方法

## 实验关键数据

### 主实验（RoBERTa + DINOv2 ViT-Giant）

| 方法 | STL10 | CIFAR10 | CIFAR100 | ImageNet | Flickr30 I2T | Flickr30 T2I |
|------|-------|---------|----------|----------|-------|-------|
| Linear + Last | 75.6 | 85.5 | 34.0 | 9.9 | 32.5 | 22.1 |
| Linear + Similar + $\mathcal{R}_S$ | **92.6** | **96.3** | **51.3** | **24.7** | **65.8** | **53.7** |
| MLP + Last | 76.6 | 79.2 | 35.3 | 10.6 | 31.6 | 20.3 |
| MLP + Similar + $\mathcal{R}_S$ | **92.7** | **96.3** | **52.1** | **25.1** | **65.9** | **53.8** |
| CSA + Last | 77.9 | 78.5 | 47.4 | 23.2 | 47.0 | 38.3 |
| CSA + Similar + $\mathcal{R}_S$ | **91.7** | **97.2** | **56.4** | **26.8** | **56.1** | **43.1** |

### 消融实验

| 组件 | 分类平均相对提升 | 检索平均相对提升 | 说明 |
|------|----------------|----------------|------|
| 层选择（Last→Similar） | +2.0%~4.8% | +2.7%~18.3% | 不同方法提升不同 |
| STRUCTURE 正则化 | +26.8%~74.0% | +15.9%~137.0% | 对 MLP/Linear 提升最大 |
| 组合 | +51.6% avg | +91.8% avg | 两者协同效果最佳 |

### 关键发现
- **极少数据可行**：仅1000个配对样本+STRUCTURE 仍能显著提升性能
- **数据效率**：CIFAR100 和 Flickr30 上约需 23× 更少样本即可达到无正则化的相同性能（utility 23.1× 和 22.4×）
- **领域样本的力量**：每类仅加 3-4 个目标域样本，即可在 Flowers (95% vs CLIP 93%)、CIFAR100 上超越 CLIP（400M 数据训练）
- **CIFAR10 上超越 CLIP**：仅用 0.02% 数据的对齐方法在某些数据集上已超越端到端 CLIP
- **邻域保持验证**：无正则化时 Trustworthiness 和 Continuity 持续下降；加正则化后稳定在 0.99-1.00

## 亮点与洞察
- **"结构即先验"**：预训练模型中蕴含数亿样本的关系结构，通过正则化保留这些结构等于注入了隐式的大规模数据先验
- **多尺度层级设计**：不仅保持直接邻居关系（1-hop），还保持多步可达关系（$l$-hop），类比随机游走在不同尺度上刻画流形结构
- **实用性极高**：即插即用，可添加到任何现有对齐方法中，计算开销小

## 局限与展望
- 在复杂任务上与 CLIP（亿级数据）仍有差距，需要少量域内样本弥补
- 目前仅探索了两个模态的对齐，三模态及以上是直接扩展但未验证
- 正则化参数 $\lambda$ 和层级数 $L$ 的选择需要一定调节
- STRUCTURE 的计算涉及矩阵幂运算，batch size 较大时可能有内存瓶颈

## 相关工作与启发
- **vs CLIP (Radford et al., 2021)**: CLIP 端到端训练 400M 数据，STRUCTURE 冻结编码器 + 80K 数据，在某些任务上已可比
- **vs FuseMix (Kim et al.)**: FuseMix 也关注低数据对齐但侧重数据增强，STRUCTURE 侧重几何正则化，两者互补
- **vs ASIF (Norelli et al.)**: ASIF 利用邻域结构做直接匹配但不学习对齐函数，STRUCTURE 学习参数化映射

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多尺度邻域保持正则化 + 层选择的组合是新颖且优雅的
- 实验充分度: ⭐⭐⭐⭐⭐ 24个数据集，3种对齐方法，数据缩放/扩展分析，多模型组合
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，核心 idea说明白，实验图表信息密度高
- 价值: ⭐⭐⭐⭐⭐ 对资源受限领域（医疗/生物）的多模态对齐有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[ECCV 2024\] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](../../ECCV2024/multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)

</div>

<!-- RELATED:END -->
