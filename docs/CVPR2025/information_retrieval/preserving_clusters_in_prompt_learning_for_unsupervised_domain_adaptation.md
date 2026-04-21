---
title: >-
  [论文解读] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation
description: >-
  [CVPR 2025][无监督域适应] 提出 CRPL 框架，通过源域增强的伪标签和基于最优传输的聚类保持策略，改进 CLIP 在无监督域适应（UDA）中的 prompt 学习，使得目标域 prompt 的文本嵌入能更好地覆盖视觉嵌入的聚类结构。
tags:
  - CVPR 2025
  - 无监督域适应
  - 提示学习
  - 最优传输
  - 聚类保持
  - CLIP
---

# Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation

**会议**: CVPR 2025  
**arXiv**: [2506.11493](https://arxiv.org/abs/2506.11493)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 无监督域适应, Prompt学习, 最优传输, 聚类保持, CLIP

## 一句话总结

提出 CRPL 框架，通过源域增强的伪标签和基于最优传输的聚类保持策略，改进 CLIP 在无监督域适应（UDA）中的 prompt 学习，使得目标域 prompt 的文本嵌入能更好地覆盖视觉嵌入的聚类结构。

## 研究背景与动机

基于 CLIP 的 prompt 学习已成为无监督域适应的主流方法（DAPL、MPA、PGA），但存在三个关键问题：

1. **伪标签质量差**：现有方法依赖 CLIP 零样本预测作为目标域伪标签，但当目标域与 CLIP 预训练数据存在显著分布偏移时，伪标签准确率急剧下降（如 OfficeHome 的 Clipart 域仅 50.4%）

2. **目标 prompt 质量低**：作者通过实验惊人地发现，现有方法学到的目标域 prompt 实际上性能很差（Table 1：仅用 target prompt 推理时准确率仅 47.6%），其看似不错的总性能完全依赖源域 prompt 的贡献。源-目标 prompt对齐策略效果有限

3. **聚类结构未被利用**：CLIP 的视觉嵌入天然呈现强聚类特性（Table 1：有监督训练 target prompt 可达 99.6%），但现有方法忽略了这一几何结构

## 方法详解

### 整体框架

CRPL（Clustering Reinforcement Prompt Learning）包含两个创新组件：(1) 源域增强伪标签（SPL），利用源域 prompt 的预测来改善目标域伪标签质量；(2) 聚类保持正则化（$\mathcal{L}_\mathcal{W}$），通过最小化目标域视觉嵌入分布与文本嵌入分布之间的 Wasserstein 距离，强制文本嵌入成为视觉聚类的质心。

### 关键设计

1. **源域增强伪标签（SPL）**:
    - 功能：提供比 CLIP 零样本预测更准确的目标域伪标签
    - 核心思路：对每个目标域样本，计算其视觉嵌入到各源域类别质心的距离，用距离感知的权重 $w_{i,k}(x) = \frac{\exp(\|z_{pre} - c_k^i\|_2)}{\sum_{i'}\exp(\|z - c_k^{i'}\|_2)}$ 加权聚合多个源域 prompt 的预测，再与基础 prompt 预测取平均：$\boldsymbol{\tau}_{ave}^k(x) = \frac{1}{2}\boldsymbol{\tau}_{base}^k + \frac{w_{k,i}(x)}{2}\sum_i \boldsymbol{\tau}_{S_i}^k$
    - 设计动机：不同源域对目标域的迁移能力不同（Table 1：多源设置中各源域对不同目标域贡献差异显著），需根据样本-源域距离自适应加权

2. **基于最优传输的聚类保持（$\mathcal{L}_\mathcal{W}$）**:
    - 功能：强制目标域文本嵌入成为视觉嵌入聚类的质心
    - 核心思路：将文本嵌入 $\{\boldsymbol{\tau}_T^k\}_{k=1}^K$ 看作 $K$ 个质心的离散分布 $\mathbb{P}_{\tau,\pi}$，将目标域视觉嵌入看作经验分布 $\mathbb{P}^T$，最小化两者之间的 Wasserstein 距离：$\mathcal{L}_\mathcal{W} = \mathcal{W}_{d_z}(\mathbb{P}_{\tau,\pi}, \mathbb{P}^T)$，其中 $d_z$ 使用余弦距离
    - 设计动机：伪标签即使经过增强仍有错误，导致文本嵌入偏向正确标签样本的质心而非整个聚类的质心（Figure 1b）。Wasserstein 距离的最小化等价于最优聚类分配（Lemma 1），可自动校正文本嵌入位置

3. **联合训练策略**:
    - 功能：协调源域监督、目标域伪标签和聚类保持三个目标
    - 核心思路：总损失 $\mathcal{L}_{total} = \mathcal{L}_S + \lambda_T \mathcal{L}_T + \lambda_\mathcal{W} \mathcal{L}_\mathcal{W}$，其中源域使用交叉熵损失，目标域使用增强伪标签的软交叉熵损失
    - 设计动机：SPL 和 $\mathcal{L}_\mathcal{W}$ 相互增强——SPL 提供方向性引导（防止文本嵌入跑到错误类别的聚类），$\mathcal{L}_\mathcal{W}$ 保证文本嵌入覆盖整个聚类而非偏左偏右

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_S + \lambda_T \mathcal{L}_T + \lambda_\mathcal{W} \mathcal{L}_\mathcal{W}$$

- $\mathcal{L}_S$：源域交叉熵损失（有标签）
- $\mathcal{L}_T$：目标域软交叉熵损失（SPL 伪标签）
- $\mathcal{L}_\mathcal{W}$：Wasserstein 聚类正则化
- $\lambda_T = \lambda_\mathcal{W} = 0.5$，$\pi_k = 1/K$（均匀先验）

## 实验关键数据

### 主实验

| 数据集 | 设置 | CRPL | PGA (SOTA) | DAPL | 提升 |
|--------|------|------|----------|------|------|
| ImageCLEF | Source-combined Avg | **90.3%** | 88.2% | 87.1% | +2.1% |
| Office-Home | Source-combined Avg | **77.6%** | 74.5% | 72.8% | +3.1% |
| DomainNet | Source-combined Avg | **55.8%** | 55.4% | 52.0% | +0.4% |
| Office-Home | Multi-source Avg | **78.9%** | 75.5% | - | +3.4% |
| DomainNet | Multi-source Avg | **56.2%** | 55.4% | - | +0.8% |
| Office-Home | 12对单源 Avg | **74.4%** | 73.9% | 73.3% | +0.5% |

### 消融实验（Office-Home）

| 配置 | 推理用Prompt | Ar | Cl | Pr | Rw | 说明 |
|------|---------|------|------|------|------|------|
| CPL (CLIP伪标签) | τ_T | 47.6 | 29.0 | 53.5 | 63.5 | 基线target prompt极差 |
| SPL (源增强伪标签) | τ_T | **75.8** | **62.9** | **86.6** | **87.2** | SPL大幅提升target prompt |
| CPL + $\mathcal{L}_W$ | τ_T | 7.9 | 4.7 | 80.4 | 82.3 | 弱伪标签+OT会崩溃 |
| SPL + $\mathcal{L}_W$ | τ_T | **76.8** | **63.5** | **87.5** | **87.6** | 互补，最优组合 |

### 关键发现

- **现有方法的target prompt是"虚假"的好**：PGA 的 target prompt 单独推理仅 72.3%（Ar），其报告的好成绩依赖 source prompt + target prompt 的平均推理。而 CRPL 的 target prompt 单独就达 76.8%
- **SPL 和 $\mathcal{L}_\mathcal{W}$ 存在强互补关系**：单独用 OT 聚类（CPL + $\mathcal{L}_\mathcal{W}$）在弱伪标签域会崩溃（Ar 从47.6%跌到7.9%），但配合 SPL 后反而锦上添花
- **源域-目标域 prompt 对齐策略效果有限**：这是对现有范式的重要质疑，为该领域提供了新方向

## 亮点与洞察

- **对现有方法的深刻质疑**：通过拆分推理实验（仅用 target prompt vs 平均 prompt），暴露了 DAPL/PGA 等方法 target prompt 形同虚设的事实，这是非常有洞察力的分析
- **聚类假设+最优传输的优雅结合**：利用 CLIP 视觉嵌入的天然聚类性质，通过 Wasserstein 距离将文本嵌入"拉"到聚类质心位置，比简单的对齐损失更有理论保证（Lemma 1）
- **距离感知的跨域迁移权重**：不同源域用不同权重，比简单平均更合理

## 局限与展望

- 在目标域与源域差异极大时（如 DomainNet 的 Quickdraw），SPL 的提升有限（仅 10.6%），因为源域 prompt 本身对该域也不具备迁移能力
- $\lambda_T = \lambda_\mathcal{W} = 0.5$ 的简单设置可能不是所有场景最优
- Wasserstein 距离的计算在类别数 $K$ 很大时代价较高
- 视觉编码器完全冻结，可探索少量微调的效果

## 相关工作与启发

- 与 DAPL、MPA、PGA 的关系：都是基于 CLIP 的 prompt 学习做 UDA，但本文指出了它们隐性依赖 source prompt 的问题
- 最优传输在域适应中的应用（如 DeepJDOT、OTDA），本文将其创新性地用于 prompt 的文本嵌入优化而非特征对齐
- 启发：在其他 prompt 学习场景中，也可检查学到的 prompt 是否真正具有独立表征能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 将最优传输引入 prompt 学习做 UDA 是新颖的，对现有方法的质疑很有价值
- 实验充分度: ⭐⭐⭐⭐ 覆盖ImageCLEF/OfficeHome/DomainNet三个benchmark，消融详尽
- 写作质量: ⭐⭐⭐⭐ 动机分析深刻（Table 1的实验设计），公式推导清晰
- 价值: ⭐⭐⭐⭐ 为 prompt-based UDA 提供了新视角和实用的改进方案

<!-- RELATED:START -->

## 相关论文

- [Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](../../ACL2025/information_retrieval/redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)
- [RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](../../ACL2025/information_retrieval/raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)
- [GOAL: Global-Local Object Alignment Learning](goal_global-local_object_alignment_learning.md)
- [On Synthetic Data Strategies for Domain-Specific Generative Retrieval](../../ACL2025/information_retrieval/on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)
- [Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/information_retrieval/generalized_contrastive_learning_for_universal_multimodal_re.md)

<!-- RELATED:END -->
