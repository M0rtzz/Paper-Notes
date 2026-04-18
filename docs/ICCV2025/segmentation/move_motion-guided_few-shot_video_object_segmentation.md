---
title: >-
  [论文解读] MOVE: Motion-Guided Few-Shot Video Object Segmentation
description: >-
  [ICCV 2025][图像分割][图像分割] 本文提出运动引导的少样本视频目标分割新任务及大规模数据集 MOVE（224 类运动、4300 视频、314K mask），并设计解耦运动-外观网络 DMA，通过帧差提取运动原型+外观原型的双分支架构，在新基准上显著优于现有 FSVOS 方法。
tags:
  - ICCV 2025
  - 图像分割
  - motion understanding
  - temporal modeling
  - benchmark
---

# MOVE: Motion-Guided Few-Shot Video Object Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.22061](https://arxiv.org/abs/2507.22061)  
**代码**: https://henghuiding.com/MOVE/  
**领域**: 分割 / 视频理解 / Few-Shot  
**关键词**: few-shot video segmentation, motion understanding, video object segmentation, temporal modeling, benchmark

## 一句话总结

本文提出运动引导的少样本视频目标分割新任务及大规模数据集 MOVE（224 类运动、4300 视频、314K mask），并设计解耦运动-外观网络 DMA，通过帧差提取运动原型+外观原型的双分支架构，在新基准上显著优于现有 FSVOS 方法。

## 研究背景与动机

**领域现状**：少样本视频目标分割（FSVOS）旨在给定少量标注示例，分割查询视频中的新类别目标。现有方法（DANet、HPAN、TTI）都是**语义中心**——根据物体类别关联支持集和查询集，如"给定熊猫图，分割所有熊猫"。

**现有痛点**：(a) 现有 FSVOS 方法忽略了视频中最核心的信息——**运动模式**，等价于在做静态图像匹配；(b) YouTube-VIS 等数据集上，图像级 FSS 方法与视频级 FSVOS 方法性能相近（62.3 vs 63.0 $\mathcal{J}\&\mathcal{F}$），说明现有评估没有考核时序理解能力；(c) 参照视频分割（RVOS）用文本描述运动，但新颖或复杂运动难以用文字准确描述（如 C罗标志性庆祝动作、小丑之舞）。

**核心矛盾**：运动模式是视频的核心特征，但现有少样本分割方法主要提取外观/语义特征，缺乏有效的运动特征提取和匹配机制——导致不能跨物体类别识别相同运动。

**本文目标** (a) 建立一个以运动类别组织的 FSVOS 基准；(b) 设计能从视频中有效提取运动原型的方法，实现"看到一种运动模式 → 在新视频中找做同样运动的目标"。

**切入角度**：将支持集从图像扩展为视频剪辑（因为静态图像无法表征运动），构建运动类别词汇表，通过帧差（frame differencing）显式提取运动特征，与外观特征解耦。

**核心 idea**：把 FSVOS 从"物体类别匹配"转变为"运动模式匹配"，通过解耦运动和外观原型实现跨类别的运动分割。

## 方法详解

### 整体框架

给定支持视频（带 mask）和查询视频：共享编码器提取多尺度特征 → Proposal Generator 生成查询的粗 mask → DMA 模块分别从支持和查询视频提取解耦的运动-外观原型 → Prototype Attention 模块融合支持和查询原型 → Mask Decoder 生成最终分割 mask。同时用 [CLS] token 计算匹配分数判断查询目标是否与支持有相同运动。

### 关键设计

1. **Decoupled Motion-Appearance Module (DMA)**:

    - 功能：从视频中提取解耦的运动原型和外观原型
    - 核心思路：
        - **外观原型** $P_a$：对 $\frac{1}{4}$ 分辨率特征 $F_{l1}$ 做 mask pooling：$P_a = \frac{\sum_{h,w} F_{l1} \odot M}{\sum_{h,w} M} \in \mathbb{R}^{T \times d}$
        - **运动原型** $P_m$：计算相邻帧特征差 $D_{l1,t} = F_{l1,t+1} - F_{l1,t}$，通过 3D 卷积增强时序特征后空间池化得到 $P_m \in \mathbb{R}^{T \times d}$
        - **辅助分类头**：外观原型过 MLP 预测物体类别（$C_o$ 类），运动原型过 MLP 预测运动类别（$C_m$ 类），显式引导解耦
        - **Transformer 精炼**：运动原型通过 cross-attention（注意力到 $P_m$ 和 $P_a$）+ self-attention + FFN 精炼为最终的 $P_{\text{dma}}$
    - 设计动机：帧差是最直接的运动信号提取方式——消除了静态外观信息，保留物体位移/形变；辅助分类确保运动和外观分支各学各的不混淆

2. **Prototype Attention Module**:

    - 功能：融合支持和查询的运动-外观原型
    - 核心思路：$P^q_{\text{dma}}$ 为 query，$P^s_{\text{dma}}$ 为 key/value 做 cross-attention，再做 self-attention，多层迭代精炼
    - 设计动机：让查询原型在支持原型的引导下聚焦于与支持运动匹配的特征

3. **Matching Score**:

    - 功能：判断查询实例是否与支持有相同运动
    - 核心思路：$S_{\text{match}} = \cos([\text{CLS}]_s, [\text{CLS}]_q)$，用支持和查询的 [CLS] token 余弦相似度
    - 设计动机：在实际场景中需要先判断"是否存在目标运动"再分割，避免对不含目标运动的帧误分割

4. **Proposal Generator + Mask Decoder**:

    - 功能：多尺度特征生成粗 mask → 精细分割
    - 核心思路：Proposal Generator 用三组不同尺度卷积块处理查询特征生成粗 mask 提案；Mask Decoder 在原型引导下通过 cross-attention + 自顶向下特征融合生成最终 mask

### 损失函数 / 训练策略

- Mask 预测：Cross-Entropy + IoU Loss
- 辅助分类头：Cross-Entropy Loss（物体类别+运动类别）
- 匹配分数：Cross-Entropy Loss
- 骨干网络：ResNet50（ImageNet预训练）或 VideoSwin-Tiny（Kinetics-400 预训练）
- 学习率 1e-5 + cosine annealing，训练 240K episodes

## 实验关键数据

### 主实验

MOVE 基准 Overlapping Split (OS) 设置，2-way-1-shot：

| 方法 | Backbone | $\mathcal{J}\&\mathcal{F}$ | T-Acc | N-Acc |
|------|----------|----------|-------|-------|
| DMA (Ours) | ResNet50 | **50.1** | 98.6 | **11.5** |
| DANet | ResNet50 | 45.4 | 97.1 | 8.2 |
| TTI | ResNet50 | 45.2 | 97.6 | 9.4 |
| HPAN | ResNet50 | 44.4 | 97.3 | 7.2 |
| SCCAN (FSS) | ResNet50 | 40.6 | 93.9 | 5.8 |
| DMA (Ours) | VSwin-T | **51.5** | 98.9 | **21.2** |
| DANet | VSwin-T | 49.8 | 93.4 | 16.5 |

5-way-1-shot 设置：

| 方法 | $\mathcal{J}\&\mathcal{F}$ | T-Acc | N-Acc |
|------|----------|-------|-------|
| DMA (Ours, R50) | **40.2** | 99.5 | **28.7** |
| TTI | 35.6 | 70.6 | 26.2 |
| HPAN | 34.0 | 99.1 | 3.1 |
| DMA (Ours, VSwin-T) | **41.4** | 99.8 | **31.0** |

### 消融实验

运动提取器对比：

| 配置 | $\mathcal{J}\&\mathcal{F}$ | 说明 |
|------|----------|------|
| Differencing (Ours) | **46.8** | 帧差提取运动 |
| Mask Adapter | 43.4 | adapter 学运动 |
| Mask Pooling | 41.3 | 直接 pooling |

DMA 原型分解：

| 外观 | 运动 | $\mathcal{J}\&\mathcal{F}$ | 说明 |
|------|------|----------|------|
| ✓ | ✗ | 36.5 | 仅外观→无法区分运动 |
| ✗ | ✓ | 43.8 | 仅运动→缺外观辅助 |
| ✓ | ✓ | **46.8** | 互补最优 |

辅助分类消融：

| 物体分类 | 运动分类 | $\mathcal{J}\&\mathcal{F}$ |
|---------|---------|----------|
| ✗ | ✗ | 43.8 |
| ✓ | ✓ | **46.8** |

### 关键发现

- **运动原型 > 外观原型**（43.8 vs 36.5），证实运动是 MOVE 任务的核心线索
- 帧差是最有效的运动提取方式，比 mask adapter (+3.4) 和 mask pooling (+5.5) 都好
- **辅助分类头提升巨大**（43.8→46.8），显式约束解耦是必要的
- MOVE 的必要性：在 YTVIS 上 FSS 和 FSVOS 效果相近（62.3 vs 63.0），但在 MOVE 上差距拉大（40.6 vs 44.4），证明 MOVE 确实需要运动理解
- Oracle 实验：完美运动标签→63.6%，完美 mask→74.3%，说明运动理解和 mask 质量都有大提升空间
- t-SNE 可视化：无 DMA 时原型按物体类别（颜色）聚簇；有 DMA 后按运动类别（形状）聚簇

## 亮点与洞察

- **运动中心的新范式**：将 FSVOS 从"物体类别是什么"转变为"在做什么运动"，开辟了全新研究方向——"一段视频胜过千言万语"
- **帧差+辅助分类的解耦策略**：简洁高效——帧差天然去除静态外观只保留运动，辅助分类头确保两个分支学到各自该学的，可迁移到任何需要运动-外观解耦的任务
- **MOVE 数据集的构建**：224 类运动、4300 视频、314K mask，跨 88 个物体类别，是首个以运动类别组织的 FSVOS 数据集
- **匹配分数机制**：用 [CLS] token 相似度做运动匹配判断，简洁有效地解决了空前景的误分割问题

## 局限与展望

- N-Acc（非目标准确率）所有方法都很低（最好仅 31%），说明背景建模和假阳性抑制是重大瓶颈
- 帧差对微小运动或缓慢动作可能不敏感
- 目前只做 1-shot 设置，多样本（few-shot）下如何聚合多个支持视频的运动原型值得探索
- 复杂运动的分解（如先跳后转）和关系运动建模（多物体交互）尚未涉及
- 数据集以日常动作和体育为主，工业/医学场景的运动尚未覆盖

## 相关工作与启发

- **vs DANet/HPAN/TTI**: 这些都是语义中心的 FSVOS 方法，用图像作为支持，本文证明它们在 MOVE 上表现差，因为缺乏运动建模
- **vs LMPM (RVOS)**: 用文本描述运动，在 MOVE 上仅 41.8%，远差于 DMA 的 50.1%——证明许多细粒度运动无法用文字准确表达
- **vs SAM2**: SAM2 虽是强模型但需要首帧 mask，不适用于少样本设置——MOVE 是正交的任务设定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 运动引导的 FSVOS 是全新任务定义，帧差解耦策略简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个基线方法×2 种设置×2 种 backbone×2 种 split，消融全面，oracle 分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰（YouTube-VIS 对比实验很有说服力），可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 新数据集+新任务+强基线，有望推动运动中心视频理解的发展
