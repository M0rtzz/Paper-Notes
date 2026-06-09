---
title: >-
  [论文解读] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics
description: >-
  [计算生物] 提出 SToFM，首个多尺度空间转录组学基础模型，通过基因尺度域适应、微观尺度子切片划分和宏观尺度虚拟细胞注入，结合 SE(2) Transformer 和 88M 细胞的大规模预训练语料库，在组织区域语义分割和细胞类型标注等任务上显著超越现有方法。
tags:
  - "计算生物"
---

# SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics

- **会议**: ICML 2025
- **arXiv**: [2507.11588](https://arxiv.org/abs/2507.11588)
- **代码**: [GitHub](https://github.com/PharMolix/SToFM)
- **领域**: 分割
- **关键词**: 空间转录组学, 基础模型, 多尺度学习, SE(2) Transformer, 组织区域分割

## 一句话总结

提出 SToFM，首个多尺度空间转录组学基础模型，通过基因尺度域适应、微观尺度子切片划分和宏观尺度虚拟细胞注入，结合 SE(2) Transformer 和 88M 细胞的大规模预训练语料库，在组织区域语义分割和细胞类型标注等任务上显著超越现有方法。

## 研究背景与动机

空间转录组学（ST）技术在保留细胞空间位置的同时测量基因表达，提供了单细胞 RNA 测序无法获得的组织级信息。然而，ST 数据包含**多尺度生物学信息**，现有模型未充分捕获：

**宏观尺度**（图1a）：组织形态和器官结构信息，如功能区域、解剖层

**微观尺度**（图1b）：细胞微环境和细胞间相互作用

**基因尺度**（图1c）：单个细胞的基因表达谱

现有 ST 基础模型的局限：
- **Nicheformer**：仅利用基因表达，完全忽略空间坐标
- **CellPLM**：通过正弦位置编码整合空间信息，但属于初步尝试，缺乏精细的多尺度设计

核心挑战是：如何在包含数万个细胞的组织切片中，用合适的模型架构和自监督目标**同时捕获并整合**三个尺度的信息。

## 方法详解

### 整体框架

SToFM 分两个阶段：
1. **多尺度信息提取**：对每个 ST 切片进行三个尺度的处理，构建包含多尺度信息的子切片集合
2. **SE(2) Transformer 表示学习**：在子切片上联合建模基因表达和空间信息

### 关键设计

**基因尺度：域适应**

基于 Geneformer（预训练单细胞基础模型）进行增量训练：
- ST 数据质量低（基因覆盖有限、高 dropout 率），直接编码效果差
- 在 ST 数据上继续预训练细胞编码器 $f_{cell}$，实现 scRNA-seq → ST 的域适应
- 使用 Masked Gene Modeling + 对比学习目标

**微观尺度：子切片划分**

将整个 ST 切片按空间位置划分为多个子切片，每个包含约 1000 个细胞：
- 平衡计算效率与局部细胞间交互的保留
- 强调空间局部化的细胞-细胞相互作用

**宏观尺度：虚拟细胞**

通过 Leiden 聚类算法将切片中所有细胞聚类，每个簇聚合为一个虚拟细胞：
- 虚拟细胞的嵌入和位置为簇内所有细胞的平均值
- 保留切片的主要形态和分区信息，作为宏观信息的压缩
- 将虚拟细胞注入每个子切片，使模型在学习微观信息的同时感知宏观结构

**SE(2) Transformer**：

使用 SE(2)-不变的 Transformer 架构联合编码细胞嵌入和位置信息：
- 输入：细胞嵌入 $F^{(i)}$ + 距离矩阵 $D^{(i)}$（通过 Gaussian 模块初始化对表示）
- 距离矩阵作为注意力偏置，类似 Graphformer 和 AlphaFold
- 输出：细胞表示 $Y_{cell}^{(i)}$ 和配对表示 $Y_{pair}^{(i)}$
- 保证对 2D 平移和旋转的不变性

### 预训练目标

**Masked Cell Modeling (MCM)**：随机掩码 10% 的细胞嵌入，用输出表示预测被掩码的嵌入（MSE 损失）

**Pairwise Distance Recovery (PDR)**：随机选 10% 的细胞添加高斯噪声到坐标，用配对表示重建原始距离矩阵（MSE 损失）

$$\mathcal{L}_{MCM} = \frac{1}{|\mathcal{M}_1|}\sum_{j \in \mathcal{M}_1}(\|\hat{F}_j - F_j\|_2)^2$$

$$\mathcal{L}_{PDR} = \frac{1}{|\mathcal{M}_2|}\sum_{(j,k) \in \mathcal{M}_2}(\|\hat{D}_{jk} - D_{jk}\|_2)^2$$

### 训练策略

分阶段训练：前 2 个 epoch 冻结细胞编码器仅训练 SE(2) Transformer → 第 3 个 epoch 联合微调两者。4×A100 GPU，约 20 天。

## 实验关键数据

### 主实验1：组织区域语义分割（F1 Score）

| 模型 | ST 预训练 | 胚胎平均 | 胚胎跨切片 | DLPFC 平均 | DLPFC 跨切片 |
|------|----------|---------|-----------|-----------|-------------|
| scGPT | 无空间 | 0.7450 | 0.3947 | 0.6178 | 0.5885 |
| Geneformer | 无空间 | 0.7467 | 0.3745 | 0.5606 | 0.5440 |
| CellPLM | 表达+位置 | 0.7722 | 0.3985 | 0.6219 | 0.5953 |
| **SToFM** | **多尺度** | **0.8046** | **0.4588** | **0.6535** | **0.6437** |

### 主实验2：细胞类型标注（Mouse Brain）

| 模型 | Brain1 Acc | Brain1 F1 | Brain2 Acc | Brain2 F1 |
|------|-----------|-----------|-----------|-----------|
| CellPLM | 0.6001 | 0.4186 | 0.9256 | 0.7332 |
| **SToFM** | **0.6349** | **0.4951** | **0.9289** | **0.8362** |

### 消融实验

| 消融变体 | 基因 | 微观 | 宏观 | 胚胎跨切片 F1 | Brain1 F1 |
|---------|------|------|------|-------------|-----------|
| Cell encoder w/o DA | ✗ | ✗ | ✗ | 0.3745 | 0.3853 |
| Cell encoder w/ DA | ✔ | ✗ | ✗ | 0.4155 | 0.4725 |
| SToFM w/o VCs | ✔ | ✔ | ✗ | 0.4291 | 0.4893 |
| **SToFM** | **✔** | **✔** | **✔** | **0.4588** | **0.4951** |

### 关键发现

- 跨切片设置中 SToFM 优势更大（胚胎跨切片 F1：0.4588 vs 0.3985），说明多尺度信息显著增强迁移能力
- 三个尺度各自贡献明确：域适应（+4.1%/+8.7%）、微观信息（+1.4%/+1.7%）、宏观信息（+3.0%/+0.6%）
- 在零样本聚类中，SToFM 的 UMAP 可视化显示不同细胞类型形成清晰、紧凑的簇
- SToCorpus-88M 是当前最大的高分辨率 ST 预训练语料库，超过 Nicheformer 1.6 倍

## 亮点与洞察

1. **多尺度设计精巧**：虚拟细胞机制将宏观信息压缩后注入微观子切片，巧妙地在 Transformer 的序列输入中同时编码了三个尺度的信息，避免了处理整个切片的计算爆炸
2. **SE(2) 不变性的合理性**：组织切片的旋转和平移不应改变生物学解读，SE(2) Transformer 天然保证了这一点
3. **数据贡献**：SToCorpus-88M 涵盖 6 种 ST 技术、2000 个切片、88M 细胞，对社区有重要价值

## 局限性

1. **仅考虑三个尺度**：未建模更细粒度的尺度（如基因调控网络层面）或利用图像金字塔方法
2. **未整合多模态信息**：未利用病理图像、已知的配体-受体对等先验生物学知识
3. **低分辨率 ST 未覆盖**：仅选择单细胞分辨率或接近单细胞分辨率的数据，10x Visium 等低分辨率技术未纳入预训练
4. **计算成本高**：细胞编码器的二次前向传播和多尺度处理带来显著计算开销（4×A100 训练 20 天）

## 相关工作与启发

- **Geneformer (Theodoris et al., 2023)**：SToFM 的细胞编码器初始化来源，按相对表达排序的基因序列编码策略
- **CellPLM (Wen et al., 2023)**：首个整合空间信息的 ST 基础模型，仅用正弦位置编码，SToFM 在此基础上大幅扩展
- **Nicheformer (Schaar et al., 2024)**：当前最大的 ST/scRNA-seq 联合预训练模型，但完全忽略空间坐标
- **Uni-Mol/AlphaFold**：SE(2)/SE(3) Transformer 在分子/蛋白质中的成功应用，SToFM 将其迁移到 ST 领域

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多尺度设计（特别是虚拟细胞机制）和 SE(2) Transformer 在 ST 中的应用具有原创性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 涵盖分割、标注、聚类、反卷积、插补 5 类任务 + 消融 + 可视化
- **写作质量**: ⭐⭐⭐⭐ — 多尺度动机清晰，方法描述系统完整
- **价值**: ⭐⭐⭐⭐⭐ — 数据集 + 模型 + 代码全部公开，对 ST 分析领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)

</div>

<!-- RELATED:END -->
