---
title: >-
  [论文解读] Parallel Sequence Modeling via Generalized Spatial Propagation Network
description: >-
  [CVPR 2025][图像生成][注意力机制] GSPN 提出广义空间传播网络，通过行/列线扫描的 2D 线性传播和稳定性-上下文条件，实现原生 2D 空间感知的亚二次注意力机制，将有效序列长度降至 $\sqrt{N}$，在 16K 图像生成中加速 SD-XL 达 84 倍。
tags:
  - CVPR 2025
  - 图像生成
  - 注意力机制
  - 空间传播网络
  - 线性复杂度
  - 2D结构
  - 高分辨率生成
---

# Parallel Sequence Modeling via Generalized Spatial Propagation Network

**会议**: CVPR 2025  
**arXiv**: [2501.12381](https://arxiv.org/abs/2501.12381)  
**代码**: [项目页面](https://whj363636.github.io/GSPN/)  
**领域**: Image Generation / Architecture Design  
**关键词**: 注意力机制, 空间传播网络, 线性复杂度, 2D结构, 高分辨率生成

## 一句话总结

GSPN 提出广义空间传播网络，通过行/列线扫描的 2D 线性传播和稳定性-上下文条件，实现原生 2D 空间感知的亚二次注意力机制，将有效序列长度降至 $\sqrt{N}$，在 16K 图像生成中加速 SD-XL 达 84 倍。

## 研究背景与动机

Transformer 在视觉任务中取得巨大成功，但面临两大核心局限：
- **二次计算复杂度**：处理高分辨率图像时计算量巨大，特别是 16K 等超高分辨率任务
- **空间结构忽视**：将 2D 图像展平为 1D 序列丢失了空间连贯性
- 线性注意力方法（$Q(K^\top V)$）虽降低复杂度但同样忽视空间结构
- SSM（如 Mamba）使用 1D 光栅扫描处理 2D 数据，牺牲固有空间结构
- 2D 线性传播的核心挑战：权重矩阵的连乘积——特征值过大导致指数增长（不稳定），过小导致信号衰减（信息消失）
- 需要同时保证稳定性和远程依赖建模的平衡

## 方法详解

### 整体框架

GSPN 通过行/列线扫描实现 2D 线性传播，每个像素与前一行/列的 3 个邻近像素连接（三对角矩阵），从 4 个方向（左→右、上→下及反向）传播后融合。提供全局和局部两种变体，可无缝替换现有架构中的注意力模块。

### 关键设计1：稳定性-上下文条件（Stability-Context Condition）

**功能**：确保 2D 传播在长距离上既稳定又保持有效的上下文信息。

**核心思路**：2D 传播公式 $h_i^c = w_i^c h_{i-1}^c + \lambda_i^c \odot x_i^c$，累积权重 $W_{ij} = \prod_{\tau=j+1}^i w_\tau$。为保证 $h_i$ 是所有先前 $x'_j$ 的加权平均，需要：(1) $W_{ij}$ 是稠密矩阵，(2) $\sum_{j=0}^{n-1} W_{ij} = 1$。**定理1**：若所有 $w_\tau$ 是行随机矩阵（非负且行和为1），则 $\sum W_{ij} = 1$ 成立。**定理2**：行随机约束同时保证传播稳定性。实现方式：对每行的非零元素施加 sigmoid 后行归一化。

**设计动机**：在不引入衰减因子的前提下同时实现稳定传播和长程依赖，传统方法必须在二者间做出妥协。

### 关键设计2：三对角矩阵 + 4 方向线扫描

**功能**：以参数高效的方式建立所有像素间的稠密成对连接。

**核心思路**：每个像素仅连接前一行/列的 3 个邻近像素（左上、正上、右上），$w_\tau$ 为三对角矩阵。关键数学性质：**多个三对角矩阵的乘积为稠密矩阵**，因此经过多行传播后自然建立远距离连接。从 4 个方向分别传播，最后通过可学习合并器聚合。使用定制 CUDA 核心并行化：行间传播串行、列间/通道间/batch 间并行，有效序列长度仅为 $\sqrt{N}$。

**设计动机**：直接学习 $n \times n$ 全连接矩阵参数量过大；三对角连接 + 累积乘积的组合以 $O(3n)$ 参数实现等效全连接。

### 关键设计3：全局/局部变体与任务适配

**功能**：根据任务需求灵活选择全局或局部传播范围。

**核心思路**：局部 GSPN 将一个空间维度分为 $g$ 个非重叠组，组内独立传播，复杂度降低 $g$ 倍（极端情况 $g=n$ 时为 $O(1)$）。分类任务：低层用局部、高层用全局（需语义理解）。生成任务：主要用局部（需空间细节和局部一致性）。T2I 生成：直接替换 SD-XL 中的自注意力层，用预训练的 Q/K/V 权重初始化 GSPN 参数（利用 GSPN 与线性注意力的数学关系）。

**设计动机**：不同视觉任务对全局 vs 局部信息的需求不同，灵活切换最大化效率。无需位置编码（扫描本身隐含位置信息）消除了常见的混叠问题。

### 损失函数

随任务变化：分类用交叉熵，DiT 用扩散损失，T2I 用 SD 标准损失。

## 实验关键数据

### 主实验：ImageNet 分类

| 模型 | 类型 | 参数(M) | MAC(G) | Top-1 Acc |
|------|------|---------|--------|----------|
| **GSPN-T** | Line scan | 30 | 5.3 | **83.0** |
| VMamba-T | Raster | 22 | 5.6 | 82.2 |
| Swin-T | Transformer | 29 | 4.5 | 81.3 |
| ConvNeXT-T | ConvNet | 29 | 4.5 | 82.1 |
| LocalVMamba-T | Raster | 26 | 5.7 | 82.7 |

### 消融实验：推理速度对比（SD-XL 16K 生成）

| 注意力类型 | 16K 推理时间 | 加速比 |
|-----------|------------|--------|
| Softmax Attention | 极慢 | 1× |
| **GSPN (Local)** | 极快 | **84×** |
| GSPN (Global) | 快 | 中等 |

### 关键发现

- GSPN-T (83.0%) 在 ImageNet 分类上超越所有同规模 Mamba/Transformer/ConvNet 模型
- 在 DiT 类条件生成中，GSPN 仅用 65.6% 参数即超越 SOTA 扩散 Transformer
- SD-XL 替换自注意力后，16K 图像生成加速 84 倍，匹配原始性能
- 稳定性-上下文条件的理论保证在实验中得到验证——远距离依赖有效建模

## 亮点与洞察

- **数学优雅**：通过行随机矩阵的性质同时解决稳定性和长程依赖，理论保证清晰
- **实用性极强**：84 倍加速使超高分辨率生成成为可能
- **无需位置编码**：扫描顺序本身隐含位置信息，避免了外推和混叠问题

## 局限与展望

- 4 方向扫描引入的计算开销是常数倍但不可忽略
- 三对角连接的稀疏性可能在某些需要精确全局对应的任务中不足
- 目前主要验证在 2D 图像上，向 3D（如视频）和多模态扩展有待探索
- 未来可探索自适应方向数量和连接模式

## 相关工作与启发

- 与 SPN 的关系：GSPN 将单层模块级 SPN 提升为可堆叠的基础架构，并解决了长程传播问题
- 与 Mamba 的对比：GSPN 原生保持 2D 结构而非展平为 1D
- $\sqrt{N}$ 的有效序列长度可能启发其他需要处理 2D 数据的高效架构设计

## 评分

⭐⭐⭐⭐ — 理论优雅、实验充分的新型注意力机制。稳定性-上下文条件的数学推导令人信服，84 倍加速的实用价值巨大。在分类、条件生成和 T2I 生成三类任务上均有竞争力的表现。

<!-- RELATED:START -->

## 相关论文

- [GSPN-2: Efficient Parallel Sequence Modeling](../../NeurIPS2025/image_generation/gspn-2_efficient_parallel_sequence_modeling.md)
- [HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [Generation of Maximal Snake Polyominoes Using a Deep Neural Network](generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models](pcm_picard_consistency_model_for_fast_parallel_sampling_of_diffusion_models.md)

<!-- RELATED:END -->
