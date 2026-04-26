---
title: >-
  [论文解读] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection
description: >-
  [ICCV 2025][图像分割][目标检测] 本文提出 RISE——一种检索自增强的无监督伪装目标检测范式，通过从训练集本身构建前景/背景原型库并利用 KNN 检索生成伪标签，在无任何标注的条件下大幅超越现有无监督和基于提示的方法。
tags:
  - ICCV 2025
  - 图像分割
  - 目标检测
  - retrieval-augmented
  - KNN
  - prototype library
---

# Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection

**会议**: ICCV 2025  
**arXiv**: [2510.18437](https://arxiv.org/abs/2510.18437)  
**代码**: https://github.com/xiaohainku/RISE  
**领域**: 分割 / 伪装目标检测 / 无监督  
**关键词**: camouflaged object detection, unsupervised segmentation, retrieval-augmented, KNN, prototype library

## 一句话总结

本文提出 RISE——一种检索自增强的无监督伪装目标检测范式，通过从训练集本身构建前景/背景原型库并利用 KNN 检索生成伪标签，在无任何标注的条件下大幅超越现有无监督和基于提示的方法。

## 研究背景与动机

**领域现状**：伪装目标检测（COD）旨在从高度相似的背景中分割出目标物体。主流全监督方法依赖密集的像素级标注，标注一张图像可能耗时一小时。弱监督和半监督方法减轻了标注负担但仍需部分标注。

**现有痛点**：(a) 无监督方法（TokenCut, MaskCut, ProMerge 等）主要基于单图像内特征相似度来区分前背景，但伪装目标和背景的特征高度相似，导致单图像方法效果差；(b) 基于提示的方法用 SAM + 任务提示词，仍需某种形式的监督且对 COD 的特定上下文理解有限；(c) 基于扩散模型或多模态 LLM 生成伪标签的方法（GenSAM, ProMac）需要数天时间且 GPU 开销大。

**核心矛盾**：在单张图像中，伪装目标和背景的 DINOv2 特征非常接近（t-SNE 上几乎重叠）——仅基于图像内部相似度无法有效区分。但如果看整个数据集，前景目标与前景原型库的相似度是高于与背景原型库的。

**本文目标** 如何在完全无标注的情况下利用数据集级别的上下文信息来区分伪装前景和背景。

**切入角度**：从数据集本身挖掘原型——通过先粗后精的策略（先聚类得粗 mask，再检索提纯原型），构建高质量的前景/背景原型库，然后对每张图像用 KNN 检索分类每个特征为前景或背景。

**核心 idea**：不靠单图像内的相似度，而是靠整个训练集的原型库+KNN检索来区分伪装目标与背景，实现无监督 COD。

## 方法详解

### 整体框架

RISE 分两阶段：(1) **Clustering-then-Retrieval (CR)** — 对数据集中每张图像用谱聚类生成粗 mask，提取前景/背景全局特征，通过跨类别检索选择高置信原型，聚合为原型库；(2) **Multi-View KNN Retrieval (MVKR)** — 对每张图像提取 DINOv2 特征，每个局部特征在原型库中 KNN 检索 top-K 最相似原型，投票决定前景/背景，多视角融合消除伪影，生成伪 mask 用于训练 SINet-V2。

### 关键设计

1. **Clustering-then-Retrieval (CR) — 原型库构建**:

    - 功能：从无标注的 COD 数据集中构建高质量的前景/背景原型库
    - 核心思路：
        - **谱聚类生成粗 mask**：构建特征相似图 $\mathcal{G}$，邻接矩阵 $\mathbf{W}_{i,j} = \max(\text{cos}(\mathbf{F}'_i, \mathbf{F}'_j), 0)$，计算归一化拉普拉斯矩阵 $\mathbf{L} = \mathbf{D}^{-1/2}(\mathbf{D}-\mathbf{W})\mathbf{D}^{-1/2}$，取特征向量做 KMeans 二分类，用边界像素比例较低的簇为前景
        - **跨类别检索（Cross-Category Retrieval）**：前景原型不选"与前景全局特征最相似的"，而选"与背景全局特征最不相似的"：$\mathbf{P}^f = \arg\min_{\mathbf{s} \in \mathbf{S}_f} \text{cos}(\mathbf{s}, \mathbf{F}^g_b)$。这增强了前背景原型的区分度
        - **直方图自适应过滤**：计算所有图像前背景全局特征相似度的直方图，以峰值为阈值过滤质量差的图像
    - 设计动机：跨类别检索是关键——直觉上"与对方最不像"比"与自己最像"更能确保原型的判别力，消融实验证实带来 5-8% 提升

2. **Multi-View KNN Retrieval (MVKR)**:

    - 功能：利用原型库对每张图像生成高质量伪 mask
    - 核心思路：对每个特征 $\mathbf{F}_{i,j}$ 在前景/背景原型库中各检索 top-K（K=512）最相似原型，投票决定类别。为消除 DINOv2 特征图中的伪影，对同一图像做翻转/旋转生成多视角，各视角分别检索后逆变换取投票融合
    - 设计动机：DINOv2 特征图有固定位置的伪影（artifact），不同视角下伪影位置不同，多视角融合以不额外训练的方式消除伪影
    - 实现细节：使用 FAISS 库加速检索，图像统一 resize 到 476×476

3. **伪标签训练**:

    - 功能：用生成的伪 mask 训练标准 COD 模型
    - 核心思路：生成的伪 mask 直接作为 ground truth 训练 SINet-V2，训练流程与标准全监督一致
    - 设计动机：RISE 专注于伪标签生成质量，模型训练部分与现有方法正交可替换

### 损失函数 / 训练策略

RISE 本身不需要训练，只做伪标签生成。下游 SINet-V2 使用标准的 COD 训练策略。特征提取器为 DINOv2-ViT-L14（冻结）。

## 实验关键数据

### 主实验

在四个 COD 基准上与无监督方法对比（DINOv2-ViT-L14 特征提取器）：

| 方法 | CHAMELEON $S_\alpha$↑ | COD10K $S_\alpha$↑ | COD10K $F^\omega_\beta$↑ | NC4K $S_\alpha$↑ |
|------|--------|---------|---------|--------|
| RISE | **0.822** | **0.763** | **0.600** | **0.805** |
| ProMerge | 0.741 | 0.674 | 0.435 | 0.726 |
| TokenCut | 0.708 | 0.637 | 0.370 | 0.697 |
| VoteCut | 0.679 | 0.645 | 0.390 | 0.674 |
| DiffCut | 0.574 | 0.628 | 0.372 | 0.693 |

与基于提示的方法对比（集成 SAM）：

| 方法 | CHAMELEON $S_\alpha$↑ | COD10K $S_\alpha$↑ | COD10K $F^\omega_\beta$↑ | NC4K $S_\alpha$↑ |
|------|--------|---------|---------|--------|
| RISE+SAM | **0.823** | **0.790** | **0.643** | **0.825** |
| WS-SAM* | 0.795 | 0.787 | 0.622 | 0.829 |
| ProMac | 0.786 | 0.774 | 0.609 | 0.812 |
| GenSAM | 0.659 | 0.641 | 0.390 | 0.702 |

### 消融实验

| 配置 | COD10K $S_\alpha$ | COD10K $E_\phi$ | COD10K $F^\omega_\beta$ | COD10K $M$ |
|------|--------|---------|---------|------|
| (e) 完整 RISE | **0.763** | **0.840** | **0.600** | **0.049** |
| (a) 仅图像级建模（谱聚类） | 0.641 | 0.662 | 0.414 | 0.169 |
| (b) 无跨类别检索 | 0.710 | 0.781 | 0.518 | 0.065 |
| (c) 无直方图过滤 | 0.744 | 0.822 | 0.575 | 0.055 |
| (d) 无多视角检索 | 0.759 | 0.832 | 0.584 | 0.052 |

### 关键发现

- **数据集级别信息是关键**：从仅图像级建模到完整 RISE，$S_\alpha$ 提升超 12%，证明利用跨图像信息远优于单图像相似度
- **跨类别检索贡献最大**：去掉后 COD10K 上 $S_\alpha$ 下降 5.3%、$F^\omega_\beta$ 下降 8.2%
- 超越使用人工标注弱监督信号的 WS-SAM，且推理时间从数天缩短到数小时
- 对不同 DINO 变体鲁棒：DINO-ViT-S16/B16、DINOv2-S14/B14/L14 均有效

## 亮点与洞察

- **检索自增强范式**：不依靠外部数据源，从数据集自身构建原型库——"自己提拔自己"——这对标注成本极高的 COD 任务极有价值
- **跨类别检索的逆向思维**：选原型时不找"最像自己的"而找"最不像对方的"，显著提升判别力——这个 trick 可以迁移到任何需要构建对比原型的场景
- **多视角消除伪影**：利用 DINOv2 伪影位置随视角变化的特性，通过简单的翻转/旋转+投票消除，比微调模型简单得多
- **直方图自适应阈值**：用相似度分布的峰值做自适应过滤，无需人工设定阈值

## 局限与展望

- 谱聚类的粗 mask 质量是瓶颈——如果初始分割太差，原型质量会受影响
- K=512 的 top-K 参数需要调优（敏感性分析见论文图5）
- 当前只做二分类（前景/背景），不支持多实例检测
- 对极小目标的检测仍有改进空间，尽管定性结果显示比基线好

## 相关工作与启发

- **vs TokenCut/VoteCut**: 这些方法基于单图像的 Normalized Cut，伪装场景下前背景相似度高导致失败；RISE 用数据集级原型克服了这一局限
- **vs ProMac/GenSAM**: 这些方法用扩散模型或多模态 LLM 生成提示耗时数天，RISE 仅需数小时且效果更好
- **vs 检索增强分割（RASS）**: RASS 用外部模型生成原型库，RISE 从数据集自身挖掘原型，避免了域外偏差

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 检索自增强范式在 COD 中首创，跨类别检索策略巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集+八种无监督+三种提示方法对比+全面消融+敏感性分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，t-SNE 可视化很有说服力
- 价值: ⭐⭐⭐⭐⭐ 为无监督 COD 设立新标杆，思路可推广到其他精细分割任务

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](../../ECCV2024/segmentation/learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](../../ECCV2024/segmentation/frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](../../CVPR2026/segmentation/fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[NeurIPS 2025\] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations](../../NeurIPS2025/segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)

<!-- RELATED:END -->
