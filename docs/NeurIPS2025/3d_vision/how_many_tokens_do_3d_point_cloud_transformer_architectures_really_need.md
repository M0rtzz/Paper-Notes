---
title: >-
  [论文解读] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?
description: >-
  [NeurIPS 2025][3D视觉][Transformer] 本文系统性地揭示了 3D 点云 Transformer（如 PTv3、Sonata）中存在 90-95% 的 token 冗余，并提出 gitmerge3D——一种全局信息感知的图 token 合并方法，通过能量分数自适应合并策略实现了高达 5.3× FLOPs 降低和 6.4× 显存节省而几乎不损失精度。
tags:
  - NeurIPS 2025
  - 3D视觉
  - Transformer
  - Token冗余
  - Token合并
  - 3D语义分割
  - 计算效率
---

# How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?

**会议**: NeurIPS 2025  
**arXiv**: [2511.05449](https://arxiv.org/abs/2511.05449)  
**代码**: [https://gitmerge3d.github.io](https://gitmerge3d.github.io)  
**领域**: 3d_vision  
**关键词**: 点云Transformer, Token冗余, Token合并, 3D语义分割, 计算效率

## 一句话总结
本文系统性地揭示了 3D 点云 Transformer（如 PTv3、Sonata）中存在 90-95% 的 token 冗余，并提出 gitmerge3D——一种全局信息感知的图 token 合并方法，通过能量分数自适应合并策略实现了高达 5.3× FLOPs 降低和 6.4× 显存节省而几乎不损失精度。

## 研究背景与动机

**领域现状**：Transformer 架构（特别是 Point Transformer v3/PTv3）已成为 3D 点云理解的主流骨干网络，在语义分割、目标检测、3D 重建等任务上达到 SOTA。PTv3 通过空间填充曲线将点云序列化为 1D 序列，在局部分区内做自注意力。

**现有痛点**：尽管 PTv3 在架构上做了优化（用序列化替代 KNN 降低 28% 推理时间、去除相对位置编码节省 26% 计算），它仍然在自注意力中处理大量冗余的 token。每个分区默认用 1024 个 token，注意力复杂度为 $\mathcal{O}(N^2)$，导致高 FLOPs 和高显存消耗。

**核心矛盾**：人们普遍认为密集的 token 化对 3D Transformer 性能至关重要，但实际上 3D 点云数据在空间上稀疏且细粒度——同一物体表面的相邻点特征高度相似，存在大量冗余。

**本文目标** (1) 量化 3D 点云 Transformer 中 token 冗余的程度；(2) 设计专门针对 3D 数据的高效 token 合并策略。

**切入角度**：将 2D 视觉 Transformer 中的 token 合并/剪枝技术引入 3D 点云，并通过实验发现通用方法已经能合并 50% token 不掉点。进而设计了利用 3D 空间局部性和注意力显著性的专用合并策略，实现更激进的压缩（95-99%）。

**核心 idea**：3D 点云 Transformer 严重"过度 token 化"，保留 5-10% 最有信息量的 token 就足以维持几乎相同的性能。

## 方法详解

### 整体框架
在 PTv3 的每个注意力层之前插入 token 合并操作，合并后的 token 参与注意力计算，之后通过反合并（unmerge）恢复原始分辨率以兼容后续 MLP 层。方法分为两部分：(1) 全局信息能量分数计算；(2) 基于能量的自适应合并策略。

### 关键设计

1. **全局信息能量分数 (Global-Informed Energy Score)**:

    - 功能：为每个 token 评估其"信息含量"，指导哪些 token 应该被合并。
    - 核心思路：构建二部图 $G = (\mathcal{V}, \mathcal{E})$，顶点集为所有 token $x_i$ 和所有分区质心 $\bar{P}_j = \frac{1}{|\mathcal{P}_j|} \sum_{x_k \in \mathcal{P}_j} x_k$。每个 token 的能量分数为其与所有分区质心的平均余弦相似度的负值：$E(x_i) = -\frac{1}{|\mathcal{N}(x_i)|} \sum_{\bar{P}_j \in \mathcal{N}(x_i)} \cos(x_i, \bar{P}_j)$。能量低的 token（与全局结构更一致）被认为冗余可合并；能量高的 token（与全局结构不一致）包含更多独特信息，应保留。
    - 设计动机：通用合并方法（ToMe 等）只看局部 token 相似度，不考虑全局结构。能量分数通过与所有分区质心的关系捕获 token 的全局独特性，更适合 3D 场景中密度不均匀的情况。

2. **自适应分区级合并策略**:

    - 功能：根据分区的平均能量自适应选择合并程度——高能量分区温和合并，低能量分区激进合并。
    - 核心思路：分区重要度 $E(\mathcal{P}) = \frac{1}{|\mathcal{P}|} \sum_{x \in \mathcal{P}} E(x)$。若 $E(\mathcal{P}) > \tau$，应用中等合并比率 $r$；否则应用激进合并 $r^+$（$r^+ \gg r$）。实验中固定 $\tau = 0.2$，对所有数据集通用。
    - 设计动机：3D 场景中不同区域的信息密度差异极大（物体边界 vs 大面积墙面/地板）。统一合并比率要么在高信息区域丢失细节，要么在低信息区域不够激进。自适应策略在两者间取得平衡。

3. **空间感知的 Token 合并执行**:

    - 功能：在每个分区内执行实际的 token 合并操作。
    - 核心思路：每个分区被均匀分成若干 bin，在 bin 内随机选择目标 token（destination），将源 token（source）通过特征平均合并到目标 token。使用 Value 特征作为合并度量、按独立注意力头分别合并效果最好。合并后用反合并函数 $f^{-1}$ 恢复原始分辨率。
    - 设计动机：在 bin 内随机选择目标保证了空间局部性——合并的 token 在 3D 空间中是相近的，避免了将远距离 token 错误合并。

### 训练策略
方法主要设计为 off-the-shelf（即推理时直接应用无需重训练）。对于要求最高精度的场景，只需原始训练周期 10% 的微调（仅更新注意力层前后的 MLP），模型不仅恢复还可能超过基线性能。

## 实验关键数据

### 主实验 — 3D 语义分割（mIoU）

| 方法 | ScanNet Val | ScanNet200 Val | S3DIS Area5 |
|------|-----------|---------------|-------------|
| PTv3 (原始) | 77.6 | 35.2 | 74.7 |
| - Random Drop (80%) | 70.1 | 31.1 | 73.4 |
| - FPS (80%) | 71.2 | 32.4 | 70.9 |
| - VoxelGrid Down. (80%) | 72.1 | 32.2 | 69.1 |
| - **Ours (off-the-shelf, 80%)** | **77.0** | **34.4** | **72.3** |
| - **Ours (微调, 80%)** | **77.4** | **35.2** | **74.3** |
| PTv3-Sonata (原始) | 79.0 | 30.4 | 72.2 |
| - **Ours (off-the-shelf, 80%)** | **77.5** | **28.8** | **72.8** |
| - **Ours (微调, 80%)** | **78.9** | **30.9** | **73.5** |

### 效率提升（PTv3，合并 90%）

| 指标 | PTv3 原始 | PTv3 + Ours | 改善 |
|------|----------|------------|------|
| FLOPs | 107.5 GFLOPs | 19.9 GFLOPs | **5.3×** |
| 显存 | 10.12 GB | 1.6 GB | **6.4×** |
| NuScenes 显存 | 6.20 GB | 0.92 GB | **6.7×** |
| NuScenes FLOPs | 101.68 | 32.45 | **3.1×** |

### 3D 重建（SplatFormer + Ours，合并 90%）

| 数据集 | SplatFormer 原始 PSNR | + Ours PSNR | SSIM 差异 |
|--------|---------------------|-------------|----------|
| GSO-OOD | 24.71 | ~24.6 | < 0.01 |
| Objaverse-OOD | 22.43 | ~22.3 | < 0.01 |
| RealWorld-OOD | 24.33 | ~24.2 | < 0.01 |

合并 90% token 后 3D 重建质量几乎无变化，仍大幅超越 3DGS、2DGS 等方法。

### 关键发现
- **核心发现**：即使合并 90-95% 的 token，PTv3 的 PCA 特征表示几乎不变。大部分预测的变化集中在物体边界极少数点上。
- 使用 Value 特征作为合并度量 + 独立头合并的组合最优（ScanNet mIoU 76.98 vs Q 的 76.27）。
- 能量阈值 $\tau = 0.2$ 在所有任务和数据集上通用，无需针对性调参。
- 在 SpatialLM 目标检测中，合并 80% token 不仅不掉点（F1 O50 从 0.1894 提升到 0.2006），还将显存从 12.36 GB 降到 2.53 GB。

## 亮点与洞察
- **挑战了核心假设**：密集 token 化对 3D Transformer 至关重要——这个领域普遍接受的假设被本文用系统实验推翻。90% 的 token 可以丢弃，这个发现对 3D 基础模型的设计方向有深远影响。
- **全局能量分数的设计**：通过 token 与所有分区质心的相关性衡量信息含量，比局部相似度度量更能反映 3D 场景的全局结构特点。这个思路可推广到视频 Transformer 等其他具有空间冗余的模态。
- **off-the-shelf 可用性**：无需重训练即可直接应用到任何基于 PTv3 的模型（Sonata、SplatFormer、SpatialLM），实用价值极高。

## 局限与展望
- 方法主要验证在 PTv3 及其变体上，对其他 3D Transformer 架构（如 OctFormer、Swin3D）的 token 冗余情况未验证。
- $\tau = 0.2$ 声称对所有数据集通用，但极端场景（如非常密集的工业点云或极度稀疏的航空 LiDAR）可能需要调整。
- 合并后的反合并（unmerge）采用简单复制策略恢复原始分辨率，可能在精细边界处丢失局部细节。
- 99% 合并率虽然可行但精度开始明显下降，实际部署中 80-95% 是更稳妥的选择。

## 相关工作与启发
- **vs ToMe (Token Merging)**: ToMe 使用二部图软匹配合并 2D token，是通用方法。本文在 3D 上改进了合并度量（全局能量分数）和自适应策略（分区级），在 95% 合并率下明显优于 ToMe 直接移植。
- **vs 传统下采样 (Random, FPS, VoxelGrid)**: 这些在输入层操作的方法在高压缩率下精度大幅下降（ScanNet 70.1-72.1 vs 本文 77.0），因为它们丢弃了整个点的几何信息。本文在特征层操作，保留了更多语义信息。
- **vs PiToMe/ALGM**: 这些进阶 token 合并方法也被测试，但它们为 2D 分类设计，不支持密集预测的特征恢复，在 3D 分割任务上效果不如本文。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统揭示 3D Transformer token 冗余并提出专用合并方案，发现本身有重大价值
- 实验充分度: ⭐⭐⭐⭐⭐ 横跨分割、重建、检测三大任务，多个骨干网络和数据集，消融详尽
- 写作质量: ⭐⭐⭐⭐ 观察清晰，实验丰富，但方法描述中图表引用略有混乱
- 价值: ⭐⭐⭐⭐⭐ 对 3D 基础模型效率优化有直接且显著的指导意义，off-the-shelf 的特性极具实用价值

<!-- RELATED:START -->

## 相关论文

- [Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [When Do We Not Need Larger Vision Models?](../../ECCV2024/3d_vision/when_do_we_not_need_larger_vision_models.md)
- [Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [LitePT: Lighter Yet Stronger Point Transformer](../../CVPR2026/3d_vision/litept_lighter_yet_stronger_point_transformer.md)
- [Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model](galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)

<!-- RELATED:END -->
