---
title: >-
  [论文解读] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter
description: >-
  [CVPR 2025][3D视觉][点云理解] 提出 Point Mamba Adapter (PMA)，通过 Mamba 架构将预训练点云模型所有中间层的互补特征构造为有序序列并进行融合，配合几何约束门控提示生成器 (G2PG) 动态优化 3D 空间的序列排序，在仅训练 1% 参数的情况下达到甚至超越全量微调的性能。
tags:
  - CVPR 2025
  - 3D视觉
  - 点云理解
  - 参数高效微调
  - Mamba
  - 中间层特征融合
  - 状态空间模型
---

# PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter

**会议**: CVPR 2025  
**arXiv**: [2505.20941](https://arxiv.org/abs/2505.20941)  
**代码**: https://github.com/zyh16143998882/PMA  
**领域**: 3D视觉  
**关键词**: 点云理解, 参数高效微调, Mamba, 中间层特征融合, 状态空间模型

## 一句话总结

提出 Point Mamba Adapter (PMA)，通过 Mamba 架构将预训练点云模型所有中间层的互补特征构造为有序序列并进行融合，配合几何约束门控提示生成器 (G2PG) 动态优化 3D 空间的序列排序，在仅训练 1% 参数的情况下达到甚至超越全量微调的性能。

## 研究背景与动机

**领域现状**：点云理解领域已从端到端监督学习转向"自监督预训练 + 下游微调"范式，Point-MAE、Point-BERT、PointGPT-L 等预训练模型成为主流。微调策略分为全量微调 (FFT) 和参数高效微调 (PEFT)。

**现有痛点**：现有 PEFT 方法（如 IDPT、DAPT、PointGST）在预训练模型的每一层引入少量可学习参数进行适应，但最终只使用最后一层的输出特征送入下游任务头，**完全丢弃了中间层的丰富信息**。这在分割等需要细粒度点级理解的任务上尤为致命。

**核心矛盾**：作者实验发现，预训练模型的中间层特征蕴含着与最终层几乎等量的语义信息（例如仅用前 3 层，分类精度仅下降 2.6%），但现有 PEFT 方法完全忽略了这些互补信息。问题的根源在于：(1) 如何高效融合所有层的特征？(2) 3D 空间的各向同性使得构建有序序列非常困难。

**本文目标**：设计一个正交于预训练骨干的 PEFT 方法，能够高效融合所有中间层特征，同时解决 3D 点云的序列排序问题。

**切入角度**：中间层特征具有"时序有序性"（随层深度递增），且总 token 数量远超单层（$L \times M$），传统 Attention 因二次复杂度不可行，而 Mamba 的线性复杂度和序列建模能力恰好适配。

**核心 idea**：用 Mamba 作为 Adapter，将预训练模型所有层的 token 拼接为有序序列进行全面特征融合，通过几何约束动态优化序列排序。

## 方法详解

### 整体框架

PMA 的 pipeline 如下：输入点云经 FPS+KNN 分为点块，通过 PointNet 提取嵌入和位置编码，加上 CLS token 后送入冻结的 L 层 Transformer 骨干。每层输出的 token 特征被送入共享的 G2PG 模块生成几何提示和排序索引，然后所有层的 token 按层序拼接为长序列，送入 Mamba Adapter 进行全面融合。最终将融合后的前 N-1 层特征、最后一层特征和 CLS token 拼接送入任务头。训练时仅更新 CLS token、G2PG、Mamba Adapter 和任务头。

### 关键设计

1. **Mamba Adapter（正交适配器）**:

    - 功能：将预训练模型所有中间层特征融合为统一表示
    - 核心思路：将 L 层 × M 个 token 拼接为长度 $L \times M$ 的序列，利用 Mamba 的状态空间模型进行序列建模。Mamba 的线性复杂度 $O(L \times M)$ 使得处理这样的长序列成为可能。输出矩阵 $C$ 被增强为 $C + P$，其中 $P$ 是几何提示。
    - 设计动机：与传统 Attention 的 $O((L \times M)^2)$ 相比，Mamba 的线性复杂度使全层融合变得可行。同时 Mamba 的序列依赖建模天然适合层间递进的语义信息。

2. **几何约束门控提示生成器 (G2PG)**:

    - 功能：为 Mamba 的输出门生成几何提示，同时生成 token 排序索引
    - 核心思路：对每层输出 token，基于中心坐标用 KNN 构建连通图，通过 Down Linear + Max Pooling 聚合邻域特征以强化几何约束，再通过 Up Linear 映射到 Mamba 输出矩阵 $C$ 的维度（如 $S=128$），Softmax 得到概率分布 $T_i^D$。$T_i^D$ 有两个用途：(1) 通过 One-hot + Argmax 为每个 token 分配唯一索引实现几何语义感知排序；(2) 映射生成几何提示 $P_i$ 注入 Mamba 的输出矩阵。
    - 设计动机：3D 空间具有各向同性，没有天然方向，不能像 NLP 那样简单按位置排序。G2PG 利用点云的空间邻域约束来学习排序和提示，使 Mamba 能基于空间结构（而非仅依赖前序输入）来调整输出。

3. **特征聚合与任务头**:

    - 功能：将 Mamba 融合后的多层特征与最终层特征进行拼接，送入下游任务头
    - 核心思路：最终预测 $y = f([C_N; F_{last}; F_{pre}])$，其中 $F_{pre}$ 是前 N-1 层融合特征，$F_{last}$ 是最后一层特征，$C_N$ 是最终 CLS token。这种设计同时保留了全层融合信息和最终层的全局语义。
    - 设计动机：既利用了 Mamba 的全层融合能力，又保留了最终层本身的高质量表示，形成互补。

### 损失函数 / 训练策略

- 采用标准的交叉熵损失进行分类，使用标准分割损失进行分割
- 训练时冻结整个 Transformer 骨干的所有参数，仅更新 CLS token、G2PG、Mamba Adapter 和下游任务头
- 相比 PointGPT-L 的 360.5M 参数，PMA 仅需 4.9M 可训练参数（减少 99%）

## 实验关键数据

### 主实验

| 数据集 | 指标 | PMA (Ours) | PointGPT-L (FFT) | PointGST | 提升 |
|--------|------|-----------|------------------|----------|------|
| ScanObjectNN OBJ-BG | OA(%) | 98.97 | 97.2 | 98.97 | +1.77 vs FFT |
| ScanObjectNN PB-T50-RS | OA(%) | 95.18 | 93.4 | 94.83 | +1.78 vs FFT |
| ModelNet40 (w/ Vote) | OA(%) | 95.4 | 94.9 | 95.3 | +0.5 vs FFT |
| ShapeNetPart | mIoU_C | 84.52 | - | 83.87 | +0.65 vs PointGST |

### 消融实验

| 配置 | ScanObjectNN PB-T50-RS | 说明 |
|------|----------------------|------|
| Point-MAE + FFT | 85.18 | 全量微调基线 |
| Point-MAE + IDPT | 84.94 | 对比PEFT |
| Point-MAE + DAPT | 85.08 | 对比PEFT |
| Point-MAE + PointGST | 85.29 | 对比PEFT |
| Point-MAE + PMA | **86.43** | 本文方法，+1.25 vs FFT |

### 关键发现

- 中间层特征的价值被严重低估：仅用前 3 层的分类精度仅比全部 12 层低 2.6%，说明中间层携带了大量互补信息
- G2PG 的几何排序对 Mamba 的有效性至关重要：解决了 3D 各向同性带来的序列构建难题
- PMA 在分割任务上的提升尤为显著（如 Point-BERT 上 mIoU_I 从 85.7 提到 86.1），验证了中间层融合对细粒度理解的重要性
- 仅需 1% 的参数即可超越全量微调，极大降低了部署成本

## 亮点与洞察

- **Mamba 作为特征融合器的新范式**：不是把 Mamba 用于序列建模本身，而是作为跨层特征融合的工具——这个视角很新颖，将 PEFT 从"在每层加小模块"升级为"用一个全局模块融合所有层"
- **G2PG 的双重功能设计**：一个模块同时解决排序和提示两个问题，且跨层共享参数，非常经济
- **中间层特征实验**：Figure 1 的实验（逐步增加使用的层数）为"中间层特征有价值"提供了直接证据，这个观察可以迁移到 2D 视觉和 NLP 的 PEFT 研究中

## 局限与展望

- 目前仅在 Transformer 架构的点云预训练模型上验证，是否适用于其他架构（如 PointMamba 本身就用 SSM 的模型）还不清楚
- G2PG 的 KNN 图构建增加了一定的计算开销，对于超大规模点云场景的效率有待评估
- 序列排序策略依赖于 Softmax + Argmax 的离散化，可能存在梯度不连续问题
- 论文没有讨论 Mamba Adapter 中不同层特征的加权策略，是否某些层更重要值得探索

## 相关工作与启发

- **vs IDPT**: IDPT 首次引入实例感知动态提示到点云 PEFT，但仍只用最终层特征，PMA 通过全层融合大幅超越
- **vs PointGST**: PointGST 用更少参数（0.6M vs 4.9M）在部分数据集上接近 PMA，但 PMA 在 PB-T50-RS 等难数据集上优势明显
- **vs PointMamba/Mamba3D**: 这些工作把 Mamba 用作骨干网络，PMA 则将 Mamba 用作适配器——两条路线可以结合

## 评分

- 新颖性: ⭐⭐⭐⭐ 中间层融合+Mamba Adapter 的组合有新意，G2PG 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 分类、分割、少样本学习全面覆盖，多个预训练模型对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，Figure 1 的观察实验很有说服力
- 价值: ⭐⭐⭐⭐ 为 3D PEFT 提供了新思路，99% 参数减少有实际部署价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)

<!-- RELATED:END -->
