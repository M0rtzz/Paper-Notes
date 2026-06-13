---
title: >-
  [论文解读] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features
description: >-
  [ICCV 2025][多模态VLM][稀疏注意力向量] 提出稀疏注意力向量（SAVs）——一种无需微调的方法，从冻结的生成式大型多模态模型（LMM）的注意力头中提取不到 5% 的头作为强特征表示，仅需约 20 个标注样本即可在视觉语言分类任务上达到 SOTA，平均超越 LoRA 微调 7%（在 BLINK、VLGuard、NaturalBench 等挑战性基准上）。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "稀疏注意力向量"
  - "小样本分类"
  - "特征提取"
  - "免微调"
  - "视觉语言分类"
---

# Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features

**会议**: ICCV 2025  
**arXiv**: [2412.00142](https://arxiv.org/abs/2412.00142)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 稀疏注意力向量, 小样本分类, 特征提取, 免微调, 视觉语言分类

## 一句话总结

提出稀疏注意力向量（SAVs）——一种无需微调的方法，从冻结的生成式大型多模态模型（LMM）的注意力头中提取不到 5% 的头作为强特征表示，仅需约 20 个标注样本即可在视觉语言分类任务上达到 SOTA，平均超越 LoRA 微调 7%（在 BLINK、VLGuard、NaturalBench 等挑战性基准上）。

## 研究背景与动机

### 问题定义

生成式 LMM（如 LLaVA、Qwen2-VL）在开放式视觉语言任务上表现出色，但在视觉语言分类任务（输入为图文、输出为离散标签）上反而不如 CLIP 等编码器模型。本文目标是：从冻结的生成式 LMM 中提取多模态特征，使其能用于任意下游分类任务，无需微调。

### 已有方法的不足

**生成式 LMM 在分类任务上表现差**：数十亿参数的 LMM 在图像分类上甚至不如 CLIP 和 SigLIP 等小模型，原因是其生成输出不适合离散标签任务

**现有特征提取方法有局限**：
   - 精心构造的 prompt（hard/soft prompting）：无法弥补与编码器的差距
   - 微调（LoRA 等）：每个新任务都需要训练规模的数据和计算，效率低
   - 少样本 ICL（in-context learning）：指令微调反而破坏了 LMM 的少样本能力，导致性能一致性下降

**CLIP 特征是单模态的**：CLIP 提取的是视觉特征或文本特征，无法处理交错的图文输入（如 VQA 中的图像+问题）

### 核心动机

**关键洞察**：借鉴神经科学中"大脑特定区域负责特定功能"（functional specificity）的发现，结合 transformer 可解释性研究表明特定注意力头对应特定任务，作者假设：在 LMM 数百个注意力头中，存在极少数（<5%）自然形成了适合特定分类任务的特征表示。这些头的注意力向量可以直接作为判别式分类器使用，无需任何梯度更新。

## 方法详解

### 整体框架

SAVs 方法分三步：
1. **特征提取**：从冻结 LMM 的每个注意力头提取注意力向量
2. **稀疏头选择**：基于少量标注样本，选出分类准确率最高的 $k$ 个头
3. **多数投票分类**：对新样本，用选出的 $k$ 个头分别分类，取多数投票结果

### 关键设计

#### 1. **注意力向量提取（Step 1）**

- **功能**：从 LMM 每个注意力头的输出中提取特征向量
- **核心思路**：
  对于一个有 $L$ 层、每层 $H$ 个注意力头的 LMM，给定输入序列 $x = \{x_1, ..., x_T\}$，每个头 $(l, m)$ 的注意力向量定义为最后一个 token 的输出：

  $$\mathbf{h}_l^m(x_i^T) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_m}}\right)V$$

  其中 $d_m = d/H$。提取所有 $L \times H$ 个头的注意力向量作为候选特征。

- **设计动机**：最后一个 token 的注意力向量聚合了整个输入序列的信息，是生成式模型中最自然的"表示"位置。不同头学到的表示可能对应不同的语义维度

#### 2. **稀疏头选择（Step 2）**

- **功能**：从 $L \times H$ 个头中选出与目标分类任务最相关的 $k$ 个头
- **核心思路**：
  给定少量标注样本 $\{(x_i, y_i)\}_{i=1}^N$（每类约 20 个），对每个头 $(l, m)$：

  1. 计算每个类 $c$ 的质心向量：
  $$\mu_c^{l,m} = \frac{1}{|N_c|}\sum_{j:y_j=c}\mathbf{h}_l^m(x_j^T)$$

  2. 用余弦相似度做最近质心分类：
  $$s_{l,m}(x_i, c) = \frac{\mathbf{h}_l^m(x_i^T) \cdot \mu_c^{l,m}}{\|\mathbf{h}_l^m(x_i^T)\| \|\mu_c^{l,m}\|}$$

  3. 计算该头的分类得分（在标注样本上的准确率）：
  $$\text{score}(l, m) = \sum_{i=1}^N \mathbf{1}[\hat{y} = y_i]$$

  4. 选择得分最高的 $k$ 个头作为 SAVs：
  $$\mathcal{H}_{\text{SAV}} = \{(l,m) \mid \text{score}(l,m) \text{ 在前 } k \text{ 名}\}$$

- **设计动机**：通过在标注数据上直接评估每个头的分类能力来选择，比启发式方法（如选最后几层）更精准。实验证明仅 20 个头（<5% 的总头数）即可捕获任务相关特征

#### 3. **多数投票分类（Step 3）**

- **功能**：对新查询，用选出的 $k$ 个头独立分类后取多数投票
- **核心思路**：
  对查询序列 $Q$，每个头独立预测：
  $$\hat{y}_{l,m} = \arg\max_{c \in \mathcal{C}} s_{l,m}(Q^T, c)$$

  最终预测为多数投票结果：
  $$\arg\max_{y \in \mathcal{C}} \sum_{(l,m) \in \mathcal{H}_{\text{SAV}}} \mathbf{1}[\hat{y}_{l,m} = y]$$

- **设计动机**：多数投票是最简单的集成方法，但由于选出的头已经是高质量分类器，简单投票就能达到强大的效果。这也避免了引入额外的可学习参数

### 损失函数 / 训练策略

SAVs 是完全无训练的方法：
- 模型完全冻结，不进行任何梯度更新
- 每类仅需约 20 个标注样本用于头选择
- 默认选择 $k = 20$ 个注意力头
- 支持 LLaVA-OneVision-7B 和 Qwen2-VL-7B
- 所有实验可在单张 A100 GPU 上运行

## 实验关键数据

### 主实验

**LLaVA-OneVision-7B 上 SAVs vs. 基线（选取关键任务）**：

| 方法 | MHalu↑ | VLGuard↑ | BLINK↑ | NB-Group↑ | EuroSAT↑ | Pets↑ | ImageNet↑ |
|------|--------|----------|--------|-----------|----------|-------|-----------|
| 零样本 | 34.7 | 31.4 | 45.0 | 27.0 | 66.5 | 88.1 | 85.3 |
| 4-shot ICL | 25.0 | 35.0 | 38.9 | 22.2 | 47.1 | 63.9 | 60.6 |
| MTV | 37.3 | 32.9 | 44.5 | 30.7 | 65.5 | 88.5 | 85.6 |
| LoRA | 78.3 | 90.0 | 47.0 | 32.4 | 85.0 | 96.8 | 91.8 |
| **SAVs** | **80.8** | **94.3** | **51.8** | **35.1** | **86.7** | **97.0** | **99.6** |

**Qwen2-VL-7B 上 SAVs vs. 基线**：

| 方法 | MHalu↑ | VLGuard↑ | BLINK↑ | NB-Group↑ | EuroSAT↑ | Pets↑ |
|------|--------|----------|--------|-----------|----------|-------|
| 零样本 | 24.0 | 26.9 | 43.3 | 28.5 | 54.7 | 92.6 |
| LoRA | 84.8 | 87.7 | 46.3 | 28.8 | 72.9 | 98.4 |
| **SAVs** | **85.1** | **96.0** | **47.2** | **32.3** | **79.9** | **98.1** |

SAVs 相对 LoRA 的平均提升：LLaVA-OV 上 +7%，Qwen2-VL 上类似

### 消融实验

**分类策略对比**：

| 分类方法 | MHalu | NaturalBench | EuroSAT |
|---------|-------|--------------|---------|
| KNN | 71.9 | 28.2 | 84.2 |
| 线性探测 (MLP) | 80.6 | 33.1 | 87.0 |
| **质心分类 (ours)** | **80.8** | **35.1** | **86.7** |

**头 vs. 层选择**：

| 特征粒度 | MHalu | NaturalBench | EuroSAT |
|---------|-------|--------------|---------|
| 选择 2 个层 | 68.3 | 31.2 | 83.1 |
| **选择 20 个头** | **80.8** | **35.1** | **86.7** |

**样本数量扩展性**：从每类 5 个样本到 200 个样本，性能持续上升且无饱和迹象

**注意力头数量**：从 5 个到 40 个头，20 个头已接近最优

### 关键发现

1. **ICL 反而降低性能**：4-shot ICL 在几乎所有任务上比零样本更差，证实指令微调破坏了 LMM 的少样本提示能力
2. **SAVs 超越 LoRA**：在不需要任何梯度更新的情况下，SAVs 在大多数任务上超越需要训练的 LoRA
3. **极度稀疏即可**：仅 20 个头（<5% 总头数）就捕获了任务相关特征，暗示分类相关信息在预训练过程中自然集中在少数头
4. **头比层更好**：选择 20 个稀疏头比选择 2 个完整层 效果好得多，证实分类信号分布于模型各层的少数头中
5. **Sample 效率极高**：每类仅 20 个样本即可达到 SOTA，且性能随样本数增加持续提升

## 亮点与洞察

1. **"功能特异性"假说的验证**：在 LMM 中发现了类似大脑功能分区的现象——特定注意力头自然编码了与特定任务相关的特征
2. **将生成模型转变为判别模型**：无需修改模型，仅通过选择内部表示，就能将生成式 LMM 变成高效的分类器
3. **完全免训练**：不需要任何梯度计算，极大降低了使用门槛和计算成本
4. **跨任务泛化**：在安全检测、VQA、图文检索、图像分类等差异极大的任务上均有效
5. **揭示了 ICL 的失效模式**：指令微调和 ICL 之间的冲突是一个重要发现

## 局限与展望

1. **每个新任务需要重新选择头**：虽然不需要微调，但选择 SAVs 仍需对每个新任务运行一次前向传播评估
2. **需要少量标注数据**：每类 20 个标注样本虽然不多，但仍然不是完全零样本的
3. **类别数受限**：目前主要验证在 2-16 类的分类任务上，更大类别数（如 1000 类）下多数投票的有效性可能受限
4. **仅评估选择式分类**：对于需要生成回答的任务（如开放式 VQA），SAVs 方法不直接适用
5. **理论解释不足**：为什么少数头能编码分类信息？这些头与模型的其他能力（如生成）有何关系？缺乏深入分析

## 相关工作与启发

- 与 task vectors 的区别：task vectors 用于增强生成能力，而 SAVs 直接用注意力向量作分类特征
- 与 CLIP 的互补：CLIP 提取单模态特征，SAVs 提取多模态特征（同时编码图像和文本）
- 启发：生成模型内部可能隐藏着大量未被利用的判别式特征，系统性地挖掘这些特征可能是一个重要研究方向

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 提出从生成模型注意力头中提取判别式特征的全新范式，思路巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖安全、VQA、检索、分类四大类共 15+ 数据集，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，上下文动机充分
- **价值**: ⭐⭐⭐⭐⭐ — 为利用生成式 LMM 做判别任务开辟了新路径，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](../../CVPR2025/multimodal_vlm/generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)

</div>

<!-- RELATED:END -->
