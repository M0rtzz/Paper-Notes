---
title: >-
  [论文解读] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images
description: >-
  [CVPR 2026][医学图像][基因表达估计] 提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。
tags:
  - CVPR 2026
  - 医学图像
  - 基因表达估计
  - 病理图像
  - 单细胞RNA测序
  - 细胞类型原型
  - 多实例学习
---

# Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images

**会议**: CVPR 2026  
**arXiv**: [2603.18461](https://arxiv.org/abs/2603.18461)  
**代码**: [https://github.com/naivete5656/CPNN](https://github.com/naivete5656/CPNN)  
**领域**: 医学图像  
**关键词**: 基因表达估计, 病理图像, 单细胞RNA测序, 细胞类型原型, 多实例学习

## 一句话总结

提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。

## 研究背景与动机

从病理全切片图像（WSI）直接预测基因表达是低成本替代 RNA 测序的重要方向。现有方法分两类：slide 级（bulk transcriptomics，用 MIL 架构）和 patch 级（spatial transcriptomics，用 Transformer/GNN）。但这些方法都只在聚合层面学习，**没有显式建模基因表达的数据生成过程**——即观测到的表达其实是由底层每个细胞的表达聚合而来的。

核心矛盾：单细胞 RNA-seq 数据提供了细胞级表达信息，但它噪声大、有批次效应、没有对应病理图像，不能直接用于 WSI 回归。

本文核心 idea：从单细胞数据中提取**稳定的细胞类型原型**（各类细胞的平均表达 profile），用原型作为先验约束预测空间。模型从图像估计各 patch 的细胞类型组成权重，再通过权重与原型的矩阵乘法得到基因表达预测。

## 方法详解

### 整体框架

CPNN 的核心假设：slide/patch 级基因表达 = 各细胞类型原型表达的加权组合。训练三步：(1) 从 scRNA-seq 数据估计细胞类型原型 $\bar{T}$；(2) 从 WSI patch 图像估计组成权重 $w$；(3) 通过负二项分布似然优化整体模型。

### 关键设计

1. **Batch-Agnostic Prototype Generation（批次无关原型生成）**：单细胞数据存在严重批次效应。用负二项回归模型建模：$\mu_{c,g}^{\mathrm{sc}} = (t_{c,g} + b_d)s_d$，其中 $s_d$ 和 $b_d$ 分别是实验条件的缩放和偏移参数。通过回归拟合后得到归一化原型 $\bar{T}$。设计动机：将技术变异剥离，只保留生物学意义上稳定的基因-基因共变模式。

2. **Compositional Weight Estimation（组成权重估计）**：从 WSI patch 用预训练编码器（CONCH）提取特征 $\mathbf{h}_i^{(n)}$，经 MLP + softmax 得到细胞类型比例 $w(\mathbf{x}_i^{(n)})$。基因表达均值建模为：$\mu_g^{\mathrm{b}}(\mathcal{X}^{(n)}) = \alpha_g \sum_i \sum_c w(\mathbf{x}_i^{(n)})_c \bar{T}_{c,g} + \beta_g$，其中 $\alpha_g, \beta_g$ 是基因特异的缩放/偏移参数，弥合单细胞与 bulk 间的模态差异。

3. **Modality Correction & Prototype Update（模态校正与原型更新）**：原型 $\bar{T}$ 在训练中被微调以适应目标分布，同时正则项约束其不偏离初始值太远：$L_R = \|\bar{T}^0 - \bar{T}\|^2 + \mathbb{E}_n[\|\mathbf{W}^{(n)} - \bar{\mathbf{W}}^{(n)}\|^2]$。正则化保证原型的可解释性——权重仍对应真实的细胞类型组成。

4. **Patch 级扩展**：在空间转录组学（ST）场景下，因 ST 数据噪声更大，用 Pearson 相关损失替换负二项似然。CPNN 可作为即插即用模块嵌入已有 ST 模型（如 STNet、TRIPLEX）。

### 损失函数 / 训练策略

总损失 $L_{\text{total}} = L_{\text{NB}} + \lambda L_R$，其中 $L_{\text{NB}}$ 是负二项分布负对数似然，$L_R$ 是对原型和权重的正则化。AdamW 优化器，batch size 16，500 epochs，$\lambda = 10^3$。4-fold 交叉验证。

## 实验关键数据

### 主实验

**Slide 级基因表达估计**

| 数据集 | 指标(SCC) | CPNN | 之前最佳 | 提升 |
|--------|-----------|------|----------|------|
| BRCA | SCC | **0.338** | 0.314 (MOSBY) | +0.024 |
| KIRC | SCC | **0.318** | 0.292 (HE2RNA) | +0.026 |
| LUAD | SCC | **0.304** | 0.286 (SRMambaMIL) | +0.018 |

**Patch 级基因表达估计（嵌入 TRIPLEX）**

| 数据集 | 指标(SCC) | TRIPLEX+CPNN | TRIPLEX | 提升 |
|--------|-----------|--------------|---------|------|
| CSCC | SCC | **0.1821** | 0.1239 | +0.0582 |
| Her2st | SCC | **0.1194** | 0.0861 | +0.0333 |
| STNet | SCC | **0.0621** | 0.0546 | +0.0075 |

### 消融实验

| 配置 | SCC (BRCA) | 说明 |
|------|-----------|------|
| w/o PI, MC, R（从头训练） | 0.305 | 无原型指导，类似 MOSBY |
| w/o MC, U, R（原型不更新不校正） | 0.174 | 模态差异过大导致崩溃 |
| w/o U, R（加模态校正） | 0.248 | 校正不足以完全弥合差异 |
| w/o R（原型可更新） | 0.336 | 接近完整模型 |
| **完整 CPNN** | **0.338** | 正则保持可解释性 |

### 关键发现

- 细胞类型标签粒度：粗粒度（8类）SCC=0.317，中等（29类）0.336，细粒度（49类）0.338，过粗会损失信息。
- 生物学验证：BRCA 各亚型的组成权重与已知生物学特征一致——Basal-like 的 Cycling 原型权重最高（高增殖），LumB > LumA。

## 亮点与洞察

- "间接利用"单细胞数据：不直接配对，而是提取稳定原型作为先验，回避噪声和模态不匹配。
- 模型自带可解释性：权重可直接解读为"这个 patch 主要由哪种细胞类型驱动"，对病理分析有实际意义。
- 即插即用设计使其可与已有 ST 方法结合，实际应用灵活。

## 局限与展望

- 依赖有标注的单细胞数据集，对无公开 scRNA-seq 的组织类型不适用。
- 原型是对各细胞类型的平均表达，无法捕捉同一类型内的表达变异。
- SCC 绝对值仍然不高（~0.3），说明从形态到表达的映射本身很难。
- 没有尝试更先进的 MIL 聚合器（如 graph-based）。

## 相关工作与启发

- 与 cell deconvolution 的区别：deconvolution 估计细胞比例来重建已知表达，本文反过来用比例和原型从图像预测未知表达。
- 借鉴 PINN 思想：用先验知识（细胞类型原型）约束模型输出空间。
- 可启发其他"利用辅助数据但模态不匹配"的场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 细胞类型原型引入是对基因表达预测的结构性创新
- **实验充分度**: ⭐⭐⭐⭐ 6 个数据集，slide/patch 两种设置，消融完整
- **写作质量**: ⭐⭐⭐⭐ 问题建模清晰，公式推导规范
- **价值**: ⭐⭐⭐⭐ 可解释性+性能提升，对计算病理学有实际意义

<!-- RELATED:START -->

## 相关论文

- [Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [Intrinsic Lorentz Neural Network](../../ICLR2026/medical_imaging/intrinsic_lorentz_neural_network.md)
- [Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)

<!-- RELATED:END -->
