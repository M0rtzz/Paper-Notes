---
title: >-
  [论文解读] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images
description: >-
  [ECCV 2024][模型压缩][图像联合对齐] 提出 SpaceJAM，一种仅约 16K 可训练参数的无监督图像联合对齐方法，无需正则化项或 atlas 维护，在 SPair-71K 和 CUB 数据集上匹配现有方法的对齐能力同时实现 10 倍以上加速。
tags:
  - ECCV 2024
  - 模型压缩
  - 图像联合对齐
  - Congealing
  - 无正则化
  - 空间变换网络
  - 轻量级
---

# SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images

**会议**: ECCV 2024  
**arXiv**: [2407.11850](https://arxiv.org/abs/2407.11850)  
**代码**: [有](https://bgu-cs-vil.github.io/SpaceJAM/)  
**领域**: 模型压缩  
**关键词**: 图像联合对齐, Congealing, 无正则化, 空间变换网络, 轻量级

## 一句话总结

提出 SpaceJAM，一种仅约 16K 可训练参数的无监督图像联合对齐方法，无需正则化项或 atlas 维护，在 SPair-71K 和 CUB 数据集上匹配现有方法的对齐能力同时实现 10 倍以上加速。

## 研究背景与动机

图像联合对齐（Joint Alignment, JA）是指在无监督条件下，将一组包含同类物体的图像对齐到共同坐标系的任务，也称为 "congealing"。该任务面临多重挑战：

**高复杂度**：现有方法依赖昂贵的模型和大量参数

**几何变形**：需要处理复杂的空间变换

**局部/全局最优收敛问题**：优化过程容易陷入不良解

**超参数调优困难**：大量正则化项导致超参数空间庞大

虽然 Vision Transformer（ViT）近期为 JA 提供了有价值的特征表示，但它们并未完全解决上述问题。现有方法如 Neural Congealing、GANgealing 等通常依赖：
- 昂贵的生成模型（如 GAN、扩散模型）
- 多种正则化项（流场平滑、atlas 一致性等）
- 长时间训练和复杂的超参数调优

**核心动机**：能否设计一种轻量级、无正则化的方法来高效完成联合对齐？

## 方法详解

### 整体框架

SpaceJAM 的核心设计理念是"简单高效"：

| 设计维度 | SpaceJAM | 现有方法 (如 Neural Congealing) |
|----------|----------|-------------------------------|
| 可训练参数 | ~16K | 数百万 |
| 正则化项 | 无 | 多个（流场、atlas 等） |
| Atlas 维护 | 不需要 | 需要 |
| 训练速度 | 快 (10x+) | 慢 |
| 特征提取 | 冻结的 ViT | 冻结的 ViT / 可训练 |

SpaceJAM 利用紧凑的网络架构，基于以下关键观察：
1. 预训练 ViT（特别是 DINO/DINOv2）的特征已经足够好，无需额外学习特征表示
2. 只需要学习一个从 ViT 特征到空间变换参数的轻量映射
3. 通过在特征空间中直接优化对齐目标，可以避免正则化的需求

### 关键设计

**1. 紧凑架构（~16K 参数）**

SpaceJAM 采用极简架构：
- 输入：冻结的预训练 ViT 提取的 patch-level 特征
- 中间层：轻量级 MLP/卷积层，将 ViT 特征映射到变换参数
- 输出：空间变换网络（STN）风格的变换参数

这种设计的关键是将 ViT 特征视为已经包含了丰富的语义和几何信息，只需要学习一个简单的参数预测头。

**2. 无正则化设计**

传统方法需要多种正则化来防止变形场退化（如折叠、过度变形）。SpaceJAM 通过以下方式避免：
- 工作在 ViT 特征空间而非像素空间，特征天然具有语义平滑性
- 使用参数化变换族（如仿射或薄板样条），而非自由形式变形场
- 紧凑的参数空间本身就限制了变换的复杂度

**3. 无 Atlas 维护**

很多 JA 方法需要维护和更新一个共享模板（atlas），SpaceJAM 完全不需要，直接在特征空间中成对优化对齐目标，简化了优化流程。

### 损失函数 / 训练策略

SpaceJAM 的优化目标是在 ViT 特征空间中最大化一组图像变换后的特征一致性。核心损失函数基于：
- 特征空间中的相似性度量（如余弦相似度或 L2 距离）
- 不包含任何额外正则化项

训练策略：
- 使用 Adam 优化器
- 无需预训练或分阶段训练
- 收敛速度极快（10x 加速的来源之一）

## 实验关键数据

### 主实验（表格）

在 SPair-71K 数据集上的语义对应评估：

| 方法 | 参数量 | 训练时间 | PCK@0.1 | 需要正则化 |
|------|--------|----------|---------|-----------|
| Neural Congealing | ~数百万 | 数小时 | 较高 | 是 |
| GANgealing | ~数百万 | 数小时 | 较高 | 是 |
| ASIC | 大规模 | 长 | 高 | 是 |
| **SpaceJAM** | **~16K** | **数分钟** | **可比** | **否** |

### CUB 数据集上的结果（表格）

| 方法 | 参数量 | 加速倍数 | 对齐质量 | Atlas 维护 |
|------|--------|----------|----------|-----------|
| 传统 JA 方法 | 大 | 1x | 基础 | 需要 |
| ViT-based 方法 | 大 | ~1x | 高 | 需要 |
| **SpaceJAM** | **最小** | **≥10x** | **可比** | **不需要** |

### 关键发现

- SpaceJAM 以约 16K 可训练参数匹配了参数量大数个量级的方法的对齐质量
- 实现至少 10 倍的训练加速，使过程更加可及和高效
- 无需正则化项极大简化了超参数调优过程
- 证明了 ViT 特征的强大表征能力——复杂的对齐任务不需要复杂的模型
- 在 SPair-71K 和 CUB 两个标准数据集上验证了有效性

## 亮点与洞察

1. **极致简约**：仅 16K 参数实现 SOTA 级别对齐，挑战了"越大越好"的范式
2. **无正则化**：完全去掉正则化项是一个大胆的设计选择，说明在合适的特征空间中，正则化可能是多余的
3. **实践友好**：无需 atlas 维护、无需复杂超参数调优、快速训练，极大降低了使用门槛
4. **ViT 特征的价值**：进一步证实 DINO/DINOv2 等自监督 ViT 特征是极强的视觉语义表示
5. **启发式思考**：当上游特征足够好时，下游任务的建模可以极大简化

## 局限与展望

- 对齐能力"match"而非"surpass"现有方法，在极端变形场景可能不如非参数化方法灵活
- 紧凑的参数化变换可能限制了处理高度非刚性变形的能力
- 依赖高质量的预训练 ViT 特征，模型选择（DINO vs DINOv2）可能影响性能
- 未在医学影像等需要精细配准的领域验证
- 可以探索结合少量可学习正则化进一步提升极端情况下的鲁棒性

## 相关工作与启发

- **Neural Congealing [Ofri-Amar et al., CVPR 2023]**：基于 ViT 特征的联合对齐方法，构建语义 atlas
- **GANgealing [Peebles et al., CVPR 2022]**：GAN 监督的稠密视觉对齐
- **ASIC [Gupta et al., ICCV 2023]**：对齐稀疏的野外图像集合
- **STN [Jaderberg et al., NeurIPS 2015]**：空间变换网络，是参数化空间变换的基石
- **DINO/DINOv2**：提供了文章依赖的高质量冻结 ViT 特征
- 作者团队在正则化-free 方法上有系列工作（CPAB 变换、Regularization-free DTAN 等）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 理论深度 | 3.5 |
| 实验充分度 | 3.5 |
| 实用性 | 4.5 |
| 写作质量 | 4 |
| 总体 | 3.5 |

<!-- RELATED:START -->

## 相关论文

- [PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)
- [JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [FreestyleRet: Retrieving Images from Style-Diversified Queries](freestyleret_retrieving_images_from_style-diversified_queries.md)
- [Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [Modality-free Graph In-context Alignment](../../ICLR2026/model_compression/modality-free_graph_in-context_alignment.md)

<!-- RELATED:END -->
