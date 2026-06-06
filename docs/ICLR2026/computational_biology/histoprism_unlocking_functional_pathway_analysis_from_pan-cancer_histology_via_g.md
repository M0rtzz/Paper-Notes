---
title: >-
  [论文解读] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction
description: >-
  [ICLR 2026][计算生物][空间转录组学] 本文提出 HistoPrism，一个高效的 Transformer 架构，通过交叉注意力注入癌症类型条件来从 H&E 病理图像预测泛癌基因表达，并提出基于 Hallmark/GO 通路的 Gene Pathway Coherence (GPC) 评估框架…
tags:
  - "ICLR 2026"
  - "计算生物"
  - "空间转录组学"
  - "基因表达预测"
  - "Pan-Cancer"
  - "Pathway分析"
  - "Transformer"
---

# HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction

**会议**: ICLR 2026  
**arXiv**: [2601.21560](https://arxiv.org/abs/2601.21560)  
**代码**: [GitHub](https://github.com/susuhu/HistoPrism)  
**领域**: 计算生物
**关键词**: 空间转录组学, 基因表达预测, Pan-Cancer, Pathway分析, Transformer

## 一句话总结

本文提出 HistoPrism，一个高效的 Transformer 架构，通过交叉注意力注入癌症类型条件来从 H&E 病理图像预测泛癌基因表达，并提出基于 Hallmark/GO 通路的 Gene Pathway Coherence (GPC) 评估框架，在通路级别预测上大幅超越 STPath，尤其在低方差核心生物通路上优势显著。

## 研究背景与动机

**领域现状**：空间转录组学（ST）是将高分辨率成像与转录组分析结合、在组织原位映射基因表达分布的技术。但 ST 成本高、劳动密集、难以规模化。H&E 染色的全切片图像（WSI）在临床中常规采集，因此从 H&E 计算推断基因表达成为热门方向。

**现有痛点**：(1) 早期方法（BLEEP、GraphST、TRIPLEX）依赖复杂的多阶段 pipeline，使用对比学习（负样本定义困难）或多分辨率工程（计算开销大）；(2) 生成式方法（STEM、STFlow）虽建模一对多映射，但仅在单癌种验证且计算密集；(3) STPath 使用 BERT 式掩码基因建模在38k基因上学习泛癌预测，但假设基因间相关性跨组织稳定（在异质性大的泛癌设置中易失效），且模型庞大导致训练/推理资源需求高。

**核心矛盾**：现有评估标准只关注 top-N 高变异基因（HVG）的 Pearson 相关性，忽略了功能通路层面的生物学一致性。一个模型可以在 HVG 上达到高分但无法恢复生物学上有意义的协调表达模式，限制了临床转化价值。

**本文目标** (1) 设计高效的直接映射架构替代复杂的重建式方法；(2) 建立通路级别的评估标准来衡量预测的生物学意义。

**切入角度**：作者认为基因表达预测本质上是模态翻译（图像→表达）任务而非重建任务，直接映射比自编码器框架更合适；评估应从孤立基因级方差转向功能通路级一致性。

**核心 idea**：用交叉注意力注入癌症类型条件 + Transformer 编码器捕获 patch 间上下文 + MLP 直接回归基因表达，并用通路级 GPC 基准评估生物学保真度。

## 方法详解

### 整体框架

输入是 H&E WSI 经病理基础模型（UNI PFM）提取的 patch 级特征 $\mathbf{x}_i \in \mathbb{R}^{D_{img}}$ + 癌症类型 one-hot 编码 $\mathbf{c}$ → 交叉注意力条件化 → Transformer 编码器建模 patch 间上下文 → MLP 回归头预测每个 patch 的 $D_{gene}$ 维基因表达。

### 关键设计

1. **泛癌条件化交叉注意力模块**:

    - 功能：将全局癌症类型信息注入局部 patch 表示
    - 核心思路：one-hot 癌症类型向量通过线性层映射为 $\mathbf{c}_{\text{emb}} \in \mathbb{R}^{D_{img}}$，作为 Key 和 Value；patch 特征作为 Query，通过标准交叉注意力计算条件化后的 patch 特征 $\mathbf{X}_{\text{cond}}$
    - 设计动机：使模型能根据癌症类型调制 patch 表示，学习泛癌共享模式和癌种特异性模式。消融实验证实去掉交叉注意力后所有癌种性能一致下降

2. **Transformer 编码器上下文聚合**:

    - 功能：捕获 patch 间的短程和长程空间依赖
    - 核心思路：条件化后的 patch 特征先投影到隐层维度 $D_{hidden}=256$，然后经2层8头 Transformer 编码器处理，输出 $\mathbf{H}_{\text{latent}} \in \mathbb{R}^{N \times D_{hidden}}$
    - 设计动机：建模高层组织结构（如肿瘤边界、免疫浸润模式）。有趣的是，消融实验发现**不加位置编码反而更好**——可能因为 UNI PFM 特征已包含形态信息，Transformer 作为置换不变的集合函数利用全局组成结构而非固定位置

3. **Gene Pathway Coherence (GPC) 评估框架**:

    - 功能：从功能通路层面评估预测的生物学保真度
    - 核心思路：从 MSigDB Hallmark（50条通路）和 GO 数据库筛选87条非冗余通路（50-100基因/条，Jaccard 相似度 < 0.1 去冗余），对每条通路内所有成员基因计算跨 patch 的 Pearson 相关系数再平均：$s_m = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{|P_m|} \sum_{g \in P_m} r_{i,g}$
    - 设计动机：HVG 指标只关注高方差基因，忽略了低方差但生物学上重要的核心通路。GPC 评估协调表达模式的恢复能力，更贴近临床需求

### 损失函数 / 训练策略

端到端训练，目标函数为 MSE 损失：$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i \in N} (\hat{y}_i - y_i)^2$。在 HEST1k 数据集上训练，该数据集聚合了153个队列、36项独立研究的空间转录组数据。HistoPrism 仅需约500张 WSI 训练，约为 STPath 的一半。

## 实验关键数据

### 主实验（Top50 HVG PCC）

| 癌种 | STPath (微均) | HistoPrism (微均) |
|------|-------------|-----------------|
| CCRCC | 0.117 | **0.206** |
| COAD | **0.459** | 0.397 |
| HCC | 0.094 | **0.113** |
| IDC | **0.629** | 0.477 |
| PRAD | 0.255 | **0.317** |
| 全部平均(微均) | 0.292 | **0.318** |

### GPC 通路级别评估

| 通路数据库 | HistoPrism 胜出比例 |
|-----------|-------------------|
| Hallmark 通路 (50条) | **86.0%** |
| GO 通路 (87条) | **74.7%** |

### 聚类质量对比

| 模型 | AMI ↑ | ARI ↑ |
|------|-------|-------|
| STPath | 0.395 | 0.402 |
| **HistoPrism** | **0.623** | **0.521** |

### 关键发现
- HistoPrism 在微均 PCC 上超越 STPath（0.318 vs 0.292），微均更能反映跨癌种的整体预测质量
- **通路级别预测是最大亮点**：在86%的 Hallmark 通路和75%的 GO 通路上优于 STPath，尤其在低方差通路上优势最大——这些通路往往对应核心生物学过程
- 聚类实验中 AMI 从0.395提升到0.623（+57.7%），说明 HistoPrism 的全转录组预测在整体上更具生物学一致性
- 位置编码无益于性能，暗示预测任务主要是局部的，PFM 特征已捕获了形态信息

## 亮点与洞察
- **GPC 评估框架的提出**是本文最重要的贡献——将评估从孤立的高方差基因转向功能通路的协调表达，更符合临床和生物学的真实需求。这比单纯提升 HVG PCC 有更大的方法论意义
- 直接映射架构 vs 自编码器框架的选择非常有洞察——基因表达预测是单向翻译任务，没有输入侧基因信息可供重建，自编码器的归纳偏置反而是负担
- 交叉注意力做泛癌条件化的设计简洁高效，且消融实验中去掉交叉注意力的性能下降验证了其必要性

## 局限与展望
- STPath 在 IDC（乳腺浸润性导管癌）和 COAD（结肠腺癌）上的宏均 PCC 仍然领先，说明 HistoPrism 在某些癌种上的特异性学习还有提升空间
- GPC 框架中通路的筛选标准（50-100基因、Jaccard < 0.1）是人为设定的，不同阈值可能影响评估结论
- 仅使用 UNI 一种 PFM 作为特征提取器，未测试不同 PFM（如 GigaPath、CTransPath）的影响
- 生成式方法（STEM、STFlow）在泛癌设置下表现差，但作者承认受限于计算资源只用了部分基因训练这些基线

## 相关工作与启发
- **vs STPath**: STPath 是当前泛癌基因预测的 SOTA 基础模型，使用 BERT 式掩码基因建模学习基因间依赖。HistoPrism 在通路级别全面超越但在某些癌种的 HVG 上仍不及 STPath。根本差异在于架构哲学：STPath 是重建式（预测缺失基因），HistoPrism 是直接映射式（从图像回归表达）
- **vs BLEEP**: BLEEP 用对比学习对齐 H&E 和基因表达到联合空间，推理时用最近邻检索。检索式推理限制了对未见样本的泛化，且负样本定义在病理中本身就很模糊
- **vs TRIPLEX**: TRIPLEX 引入多分辨率蒸馏架构，计算复杂度高但只在单癌种验证。HistoPrism 在效率和泛化性上远优

## 评分
- 新颖性: ⭐⭐⭐⭐ GPC 评估框架的提出有重要方法论贡献，架构设计简洁但缺乏突破性创新
- 实验充分度: ⭐⭐⭐⭐⭐ 10个癌种、多基线对比、通路级评估、聚类分析、效率对比、消融实验，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题动机和评估框架阐述清晰，方法部分公式化到位
- 价值: ⭐⭐⭐⭐⭐ GPC 评估范式对计算病理学领域有深远影响，HistoPrism 本身也是实用的高效工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](../../AAAI2026/computational_biology/hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)
- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](../../AAAI2026/computational_biology/spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/computational_biology/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)

</div>

<!-- RELATED:END -->
