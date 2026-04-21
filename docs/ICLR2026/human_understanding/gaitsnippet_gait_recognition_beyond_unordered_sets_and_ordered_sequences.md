---
title: >-
  [论文解读] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences
description: >-
  [ICLR2026][人体理解][gait recognition] 提出 Snippet 范式：将步态轮廓序列组织为若干"片段"（snippet），每个 snippet 由一个连续区间内随机抽取的帧构成，兼顾短程时序上下文与长程时序依赖，在 Gait3D 上以 2D 卷积骨干达到 77.5% Rank-1，超越所有 3D 卷积方法。
tags:
  - ICLR2026
  - 人体理解
  - gait recognition
  - snippet paradigm
  - temporal modeling
  - silhouette
  - 2D convolution
---

# GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences

**会议**: ICLR2026  
**arXiv**: [2508.07782](https://arxiv.org/abs/2508.07782)  
**代码**: 待确认  
**领域**: human_understanding  
**关键词**: gait recognition, snippet paradigm, temporal modeling, silhouette, 2D convolution  

## 一句话总结

提出 Snippet 范式：将步态轮廓序列组织为若干"片段"（snippet），每个 snippet 由一个连续区间内随机抽取的帧构成，兼顾短程时序上下文与长程时序依赖，在 Gait3D 上以 2D 卷积骨干达到 77.5% Rank-1，超越所有 3D 卷积方法。

## 研究背景与动机

步态识别以轮廓（silhouette）序列为输入，当前主流建模范式有两种：

**无序集合（Unordered Set）**：以 GaitSet 为代表，将所有帧视为无序集合用 2D 卷积独立提取特征，再通过 Set Pooling 聚合。优点是高效且对帧顺序扰动鲁棒，但每帧独立处理，**丢失了相邻帧之间的短程时序上下文**。

**有序序列（Ordered Sequence）**：以 GaitGL、DeepGaitV2-3D 为代表，用 3D/P3D 卷积联合建模时空特征。虽然能捕获局部时序，但训练时通常只采样约 30 帧连续段，**难以建模长程时序依赖**（真实场景序列可超过 200 帧）。

**核心问题**：能否找到一种新范式，同时获得短程时序感知和长程时序覆盖？

作者从人类认知中获得启发——识别一个人往往只需观察几个关键动作（甚至不需要完整步态周期），因此提出将步态视为**个体化动作的组合**，每个动作用一个 snippet 表示。

## 方法详解

### 整体思路

将轮廓序列切分为等长 segment，从每个 segment 中随机抽帧组成一个 snippet（代表一个局部动作），整个序列的步态特征由多个 snippet 的特征聚合而得。

### 3.1 Snippet Sampling

**训练阶段**：

- 将序列切分为 $K$ 个等长 segment（长度 $L=16$，约一个步态周期）
- 随机选取 $M=4$ 个 segment，每个 segment 内随机抽 $N=8$ 帧组成一个 snippet
- 总采样帧数 $S = M \times N = 32$
- 第一个 segment 长度 $L_1$ 随机取 $\{1,\ldots,L\}$，增加采样多样性

**推理阶段**：

- 使用所有帧：每个 segment 内全部帧构成一个 snippet，$M=K, N=L$
- 第一段长度固定为 $L$，保证预测稳定性

### 3.2 Snippet Modeling

模型称为 **GaitSnippet**，核心解决三个子问题：

#### (1) Intra-Snippet Modeling（片段内建模）

目标：捕获 snippet 内部的局部时序上下文，增强帧级特征。设计 **Snippet Block**：

- **Gathering**：将 snippet 内帧视为无序集合，通过 Temporal Max Pooling（非参数化）聚合为 snippet 级表征
- **Smoothing**：对聚合后的特征施加 $1\times1$ 卷积，平滑噪声并缩小帧级/snippet 级特征的语义差距
- **Residual**：通过残差连接将 snippet 级上下文信息与帧级特征融合

将 Snippet Block 嵌入标准 2D 残差块的两个空间卷积层之间，形成 **Residual Snippet Block**，作为骨干网络的基本构件。设计灵感来自 P3D——让帧级特征在逐层提取过程中持续感知局部时序上下文。

#### (2) Cross-Snippet Modeling（片段间建模）

- 在骨干网络输出端，先对帧级特征做 Intra-Snippet Gathering 得到 snippet 级表征
- 将所有 snippet 视为无序集合，再次通过 Temporal Max Pooling 聚合为序列级表征
- 形成**层级化无序集合**结构：帧→snippet→序列

**关键区别**：虽然两层都用了 Set Pooling，但 snippet 内经过了 Snippet Block 的时序建模，因此整体**不是**帧级排列不变的——局部时序信息已融入帧级特征。

#### (3) Snippet-Level Supervision（片段级监督）

snippet 范式天然产生两级表征（序列级 + snippet 级），作者为 snippet 级特征引入额外监督分支：

- 序列级损失：Triplet Loss $\mathcal{L}_{tp}$ + Cross-Entropy Loss $\mathcal{L}_{ce}$（配合 BNNeck）
- Snippet 级损失：$\mathcal{L}_{tp}^{\star}$ + $\mathcal{L}_{ce}^{\star}$，在 snippet 粒度构建正负对
- 总损失：$\mathcal{L}_{all} = \mathcal{L}_{tp} + \mathcal{L}_{ce} + \alpha(\mathcal{L}_{tp}^{\star} + \mathcal{L}_{ce}^{\star})$，$\alpha=0.75$
- snippet 级分支**仅用于训练**，不增加推理开销

### 骨干网络

基于 DeepGaitV2-2D（ResNet 风格 2D 卷积骨干），将标准残差块替换为 Residual Snippet Block。使用 Horizontal Pyramid Mapping 提取多粒度局部表征。

## 实验关键数据

### 主实验（真实场景数据集）

| 方法 | 类型 | 骨干 | Gait3D R1 | Gait3D mAP | GREW R1 | GREW R5 |
|------|------|------|-----------|------------|---------|---------|
| GaitSet | Set | 2D | 36.7 | 30.0 | 48.4 | 63.6 |
| GaitBase | Set | 2D | 64.6 | 55.3 | 60.1 | 75.5 |
| DeepGaitV2-2D | Set | 2D | 68.2 | 60.4 | 68.6 | 82.0 |
| DeepGaitV2-3D | Seq | 3D | 72.8 | 63.9 | 79.4 | 88.9 |
| VPNet | Seq | 3D | 75.4 | — | 80.0 | 89.4 |
| SwinGait-3D | Seq | Swin3D | 75.0 | 67.2 | 79.3 | 88.9 |
| **GaitSnippet** | **Snippet** | **2D** | **77.5** | **69.4** | **81.7** | **90.9** |

核心发现：

- GaitSnippet 用 2D 卷积骨干**全面超越**所有 3D 卷积方法
- 相比同骨干的 DeepGaitV2-2D：Gait3D R1 提升 **+9.3%**，mAP 提升 **+9.0%**
- 在 CCPG（换装场景）上 AVG 达 95.1%，CCGR-MINI R1 达 42.4%，均为最佳

### 消融实验（Gait3D）

**Snippet Sampling 的效果**：

- 仅将 DeepGaitV2-2D 的采样策略从 Set 换为 Snippet：R1 从 68.2% 升至 69.5%（+1.3%），说明 snippet 采样本身即有正则化效果
- 最优超参：$L=16, M=4, N=8$

**Snippet Block 各组件**：

- 去掉 Gathering：R1 降至 73.3%（无法做 snippet 级监督）
- 去掉 Smoothing：R1 降至 74.8%（噪声/语义差距增大）
- 去掉 Residual：R1 降至 72.5%（丢失帧级细粒度信息，影响最大）

**Snippet-Level Supervision**：

- $\alpha=0$（无 snippet 级监督）仍有竞争力表现，说明 Snippet Block 本身有效
- 加入 snippet 级监督（$\alpha=0.75$）进一步提升性能

## 优点与局限

**优点**：

- 提出了介于集合和序列之间的第三种范式，概念清晰且有认知科学支撑
- 仅用 2D 卷积骨干即超越所有 3D 方法，计算成本更低
- Snippet Block 设计简洁（非参数化 pooling + 1×1 conv + 残差），易于集成到现有架构
- 层级化监督（序列级 + snippet 级）充分利用了 snippet 范式的结构优势
- 在受控（CCPG）和无约束（Gait3D/GREW）场景均表现最优

**局限**：

- Snippet 长度 $L$ 固定为 16 帧，对步频差异较大的个体可能不够自适应
- Cross-Snippet Modeling 仅用 Max Pooling 聚合，未探索更复杂的 snippet 间关系建模（如 Transformer）
- 推理时需处理所有帧构成的全部 snippet，长序列下推理开销仍可能较高
- 仅在轮廓模态上验证，未扩展到骨架/RGB 等其他模态

## 评分

- 新颖性: ⭐⭐⭐⭐ 提出 snippet 新范式，概念贡献明确
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集 + 详尽消融
- 价值: ⭐⭐⭐⭐ 2D 骨干超越 3D 方法，实用性强

<!-- RELATED:START -->

## 相关论文

- [Cut Out the Middleman: Revisiting Pose-Based Gait Recognition](../../ECCV2024/human_understanding/cut_out_the_middleman_revisiting_pose-based_gait_recognition.md)
- [CarGait: Cross-Attention based Re-ranking for Gait Recognition](../../ICCV2025/human_understanding/cargait_cross_attention_based_re_ranking_for_gait_recognition.md)
- [BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)
- [NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)

<!-- RELATED:END -->
