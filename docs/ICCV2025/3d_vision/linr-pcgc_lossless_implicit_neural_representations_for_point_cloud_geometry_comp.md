---
title: >-
  [论文解读] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression
description: >-
  [ICCV 2025][3D视觉][点云无损压缩] LINR-PCGC 提出了首个基于隐式神经表征（INR）的点云几何无损压缩方法，通过设计轻量级多尺度 SparseConv 网络（含尺度上下文提取 SCE 和子节点预测 CNP 模块），结合 GoP 级帧共享解码器和初始化策略，在不依赖特定训练数据分布的前提下，在 MVUB 数据集上比 G-PCC TMC13v23 降低 21.21% 码率，比 SparsePCGC 降低 21.95%。
tags:
  - ICCV 2025
  - 3D视觉
  - 点云无损压缩
  - 隐式神经表征
  - 多尺度稀疏卷积
  - GoP编码
  - 模型压缩
---

# LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression

**会议**: ICCV 2025  
**arXiv**: [2507.15686](https://arxiv.org/abs/2507.15686)  
**代码**: https://huangwenjie2023.github.io/LINR-PCGC/  
**领域**: 3D视觉 / 点云压缩  
**关键词**: 点云无损压缩, 隐式神经表征, 多尺度稀疏卷积, GoP编码, 模型压缩

## 一句话总结

LINR-PCGC 提出了首个基于隐式神经表征（INR）的点云几何无损压缩方法，通过设计轻量级多尺度 SparseConv 网络（含尺度上下文提取 SCE 和子节点预测 CNP 模块），结合 GoP 级帧共享解码器和初始化策略，在不依赖特定训练数据分布的前提下，在 MVUB 数据集上比 G-PCC TMC13v23 降低 21.21% 码率，比 SparsePCGC 降低 21.95%。

## 研究背景与动机

**领域现状**：点云压缩方法分为传统方法（G-PCC、V-PCC）和 AI 驱动方法（PCGCv2、SparsePCGC）。传统方法依赖手工设计的工具和参数，AI 方法利用神经网络建模空间相关性达到 SOTA 压缩效果

**现有痛点**：
   - AI 方法严重依赖训练数据分布，数据分布偏移会导致性能显著下降（如 SparsePCGC 在 MVUB 上性能差于 G-PCC）
   - INR 方法通过对目标数据过拟合解决了分布依赖问题，但面临两个挑战：(1) 解码器网络参数需要编码进码流，限制了网络大小和拟合能力；(2) 过拟合时间过长
   - 现有 INR 方法仅限于有损压缩，无损压缩领域尚无 INR 方案

**核心矛盾**：AI 方法的高压缩效率vs分布泛化性，INR 方法的泛化性vs网络容量和编码效率之间存在根本矛盾

**本文目标**：如何在 INR 框架下实现点云几何无损压缩，同时控制解码器大小和编码时间？

**切入角度**：借鉴视频编码中 GoP（Group of Pictures）概念——相邻帧共享一个轻量解码器网络，分摊参数开销；前一个 GoP 的过拟合网络初始化下一个 GoP，加速收敛

**核心 idea**：通过 GoP 级网络共享降低参数开销 + 多尺度 SparseConv 的子节点预测实现高效无损压缩 + 初始化策略节省约 65% 编码时间

## 方法详解

### 整体框架

输入为点云序列 $S = \{x_1, ..., x_M\}$，按 GoP 分组编码（GoP 大小 T=32 帧）。每个 GoP 的编码包含三步：
1. **初始化**：用前一个 GoP 过拟合后的网络参数初始化当前 GoP
2. **编码**：过拟合网络参数 → 分离为 pc-encoder 和 pc-decoder → 编码点云 + 量化压缩 pc-decoder 参数
3. **解码**：解压 pc-decoder 参数 → 逐尺度解码点云

最终码流包含：最低尺度点云坐标 + 解码器网络参数 + 各尺度占用编码信息。

### 关键设计

1. **多尺度 SparseConv 网络**:

    - 功能：逐步下采样点云直到仅剩几十到几百个点，然后从低尺度到高尺度逐步预测占用概率
    - 核心思路：使用 MaxPooling 进行下采样 $x_t^{i+1} = DS(x_t^i)$，然后在每个尺度上预测子节点的占用概率，用算术编码压缩
    - 设计动机：多尺度架构使得高尺度（含大量细节）的信息可以利用低尺度的结构先验，逐步精细化预测

2. **尺度上下文提取模块（Scale Context Extraction, SCE）**:

    - 功能：为不同空间尺度的点云提供区分信息
    - 核心思路：将尺度嵌入（SEMB，8 通道隐式特征扩展尺度索引 $i$）作为全局信息，与邻域占用（"前后左右上下自身"7 个位置的占用状态）作为局部信息拼接，经 MLP 融合生成尺度上下文特征 $l_t^{i+1}$
    - 公式：$l_t^{i+1} = MLP_i(Concat(Nb^{i+1}, SEMB(i)))$
    - 设计动机：所有尺度共享同一套网络参数，需要机制让网络知道当前处于哪个尺度，否则无法针对性地提取空间特征

3. **子节点预测模块（Child Node Prediction, CNP）**:

    - 功能：从低尺度到高尺度上采样点云，即预测八叉树的子节点占用
    - 核心思路：将上采样问题转化为八叉树子节点占用预测（8 个通道对应 8 个子节点）。采用通道顺序（channel-wise）8 阶段预测——已解码的子节点作为后续阶段的上下文。使用 GDFE（全局深度特征提取）和 LDFE（局部深度特征提取）两个模块，GDFE 提取全局特征，LDFE 从已解码子节点提取局部特征，融合后预测占用概率
    - 与转置卷积对比：转置卷积内存占用和时间复杂度高；CNP 直接在八叉树结构上操作更高效
    - 设计动机：通道顺序预测类似自回归思路，已解码的子节点为待解码子节点提供额外上下文，提高预测准确率

4. **自适应量化（AQ）与模型压缩（MC）**:

    - AQ：归一化解码器参数到 [0,1] 后量化到 B=8 位
    - MC：训练中加入 L2 正则化使参数分布趋近拉普拉斯分布，然后用拉普拉斯分布的参数（均值 $\mu$ 和尺度 $b$）做算术编码

### 损失函数 / 训练策略

$$\mathcal{L} = \sum_{i=0}^{N} \sum_{j=0}^{7} L_{BCE}^{i,j} + \lambda \|\boldsymbol{\theta}\|_2^2$$

- $L_{BCE}^{i,j}$ 为第 $i$ 尺度第 $j$ 阶段的二值交叉熵，估计当前阶段的码流大小
- $\lambda \|\boldsymbol{\theta}\|_2^2$ 为 L2 正则化，使参数分布更集中以便压缩
- 使用 Adam 优化器，学习率从 0.01 衰减至 0.0004
- 第一个 GoP 训练 6 epochs，后续 GoP 训练 1-6 epochs
- 单张 RTX 3090 GPU

## 实验关键数据

### 主实验

**8iVFB 数据集（Tab.1）**:

| 方法 | bpp (avg) | 相对bpp | 编码时间(s) | 解码时间(s) |
|------|-----------|---------|-----------|-----------|
| G-PCC v23 | 0.743 | 100% | 2.72 | 0.923 |
| SparsePCGC | 0.625 | 84.0% | 2.202 | 1.048 |
| V-PCC v23 | 1.415 | 190.4% | 194.261 | 2.304 |
| **Ours** | **0.616** | **82.9%** | 2.464 | **0.501** |
| **Ours 2** | **0.564** | **75.9%** | 16.423 | **0.459** |

**MVUB 数据集（Tab.3）— 分布偏移场景**:

| 方法 | bpp (avg) | 相对bpp | 编码时间(s) | 解码时间(s) |
|------|-----------|---------|-----------|-----------|
| G-PCC v23 | 0.921 | 100% | 3.951 | 1.284 |
| SparsePCGC | 0.930 | **100.9%** | 3.06 | 1.456 |
| V-PCC v23 | 1.543 | 167.6% | 213.192 | 3.071 |
| **Ours** | **0.806** | **87.5%** | 2.712 | **0.554** |
| **Ours 2** | **0.725** | **78.8%** | 18.564 | **0.544** |

注意 MVUB 数据集上 SparsePCGC 甚至差于 G-PCC（100.9%），但 LINR-PCGC 仍保持 78.8% 的优异表现，体现了 INR 方法的分布无关性。

### 消融实验

**初始化策略消融（Tab.5）**:

| 初始化方式 | 相对时间 (8iVFB) | 相对时间 (Owlii) | 相对时间 (MVUB) | 平均 |
|-----------|----------|----------|----------|------|
| 随机初始化 (rand.) | 100% | 100% | 100% | 100% |
| 前一GoP初始化 (ini.) | 36.0% | 34.4% | 33.7% | 34.7% |
| 相似序列初始化 (fur. ini.) | 22.9% | 29.2% | 20.0% | 24.0% |

初始化策略平均节省 65.3% 编码时间（ini.）和 76.0%（fur. ini.）。

**模块消融（Tab.6）**:

| 配置 | 相对bpp↓ |
|------|---------|
| 仅 CNP | 100.0% |
| CNP + AQ&MC | 91.9% |
| **CNP + AQ&MC + SCE (完整)** | **88.8%** |

AQ&MC 降低 8.1% bpp，SCE 进一步降低 3.1% bpp。

**码流分配与时间组成（Tab.4，MVUB）**:

| 组成部分 | 码流占比 | 编码时间占比 | 解码时间占比 |
|---------|---------|-----------|-----------|
| 解码器参数 | 0.73% | 0.47% | 0.00% |
| 最低尺度点云 | 0.17% | 8.58% | - |
| 高尺度 (scale 2-6) | 5.83% | 30.47% | 31.60% |
| 中尺度 (scale 1) | 18.10% | 14.92% | 16.25% |
| 最高尺度 (scale 0) | 75.17% | 45.56% | 51.63% |

### 关键发现

- **INR 方法的关键优势是分布无关性**：在 MVUB 数据集上，SparsePCGC（在 ShapeNet 上训练）性能反而差于 G-PCC，而 LINR-PCGC 因为对每个序列独立过拟合，不受训练数据分布限制
- **解码器参数开销极小**：仅占总码流 0.73%（因为 GoP 级共享），不会成为瓶颈
- **编码时间与压缩率的权衡**：编码 1 epoch（~2.5s/帧）即可达到与 SparsePCGC 相当的压缩率；编码 6 epoch（~16s/帧）可进一步降低 15-20% 码率
- **解码速度快**：约为 G-PCC 或 SparsePCGC 的一半，因为轻量网络设计

## 亮点与洞察

- **GoP 级 INR 框架** — 借鉴视频编码 GoP 概念到 INR 点云压缩，一箭双雕地解决了参数开销和编码速度问题。这个思路可以迁移到其他 INR 压缩场景（如 NeRF 场景压缩）
- **子节点预测替代转置卷积** — 将八叉树上采样问题建模为分阶段的子节点占用预测，既省内存又利用了已解码信息的上下文，是一个精巧的工程设计
- **L2 正则化 → 拉普拉斯分布 → 高效参数编码** — 简单的训练技巧使参数分布更易压缩，展现了对 INR 参数特性的深刻理解

## 局限与展望

- 未利用帧间预测（inter-frame prediction），各帧在 GoP 内独立压缩，未消除时域冗余
- 编码时间仍然较长（完整编码约 16s/帧），不适用于实时应用
- 仅处理几何信息，未扩展到属性（颜色）压缩
- 未与最新的 Unicorn-Part I 对比（虽然给出了合理理由）
- 网络架构固定，未探索 NAS 或自适应架构选择

## 相关工作与启发

- **vs G-PCC**: 传统方法在所有数据集上表现稳健但压缩率有限；LINR-PCGC 在充分编码时间下降低 21-28% 码率
- **vs SparsePCGC**: SparsePCGC 在训练分布内表现好但分布外急剧退化（MVUB 甚至差于 G-PCC）；LINR-PCGC 因 INR 特性天然分布无关
- **vs V-PCC**: V-PCC 码率最高且编码极慢（194-213s），不适合稀疏点云
- **INR 压缩方法（Hu & Wang 2022 等）**: 之前 INR 方法仅做有损压缩，LINR-PCGC 首次扩展到无损场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 INR 无损点云压缩方法，GoP 框架和 CNP 模块设计新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集全面对比，含编码时间-码率曲线和详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但公式较密集
- 价值: ⭐⭐⭐⭐ 填补了 INR 无损点云压缩的空白，分布无关性有实际部署价值

<!-- RELATED:START -->

## 相关论文

- [Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [End-to-End Implicit Neural Representations for Classification](../../CVPR2025/3d_vision/end-to-end_implicit_neural_representations_for_classification.md)
- [SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)

<!-- RELATED:END -->
