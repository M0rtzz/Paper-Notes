---
title: >-
  [论文解读] End-to-End HOI Reconstruction Transformer with Graph-based Encoding
description: >-
  [CVPR 2025][3D视觉][人物交互重建] 提出 HOI-TG 框架，用 Transformer 的自注意力机制隐式学习人物交互关系，并在编码器中嵌入图残差模块分别增强人体和物体的拓扑结构建模，在 BEHAVE 和 InterCap 数据集上实现 SOTA 的 HOI 三维重建。
tags:
  - CVPR 2025
  - 3D视觉
  - 人物交互重建
  - Transformer
  - 图卷积
  - 隐式交互建模
  - 网格重建
---

# End-to-End HOI Reconstruction Transformer with Graph-based Encoding

**会议**: CVPR 2025  
**arXiv**: [2503.06012](https://arxiv.org/abs/2503.06012)  
**代码**: https://hoi-tg.github.io/ (有)  
**领域**: 3D视觉  
**关键词**: 人物交互重建, Transformer, 图卷积, 隐式交互建模, 网格重建

## 一句话总结
提出 HOI-TG 框架，用 Transformer 的自注意力机制隐式学习人物交互关系，并在编码器中嵌入图残差模块分别增强人体和物体的拓扑结构建模，在 BEHAVE 和 InterCap 数据集上实现 SOTA 的 HOI 三维重建。

## 研究背景与动机

**领域现状**：从单张图像重建人物交互（HOI）的 3D 网格是 AR/VR 和机器人操作的关键任务。现有方法如 StackFLOW、CHORE、CONTHO 通常显式建模人和物体之间的接触约束（offset、contact map 等），以此指导联合重建。

**现有痛点**：显式交互建模在全局与局部之间存在天然矛盾。网格重建关注人和物体的整体相对位置关系，而接触约束（offset、contact map）关注的是局部区域。要同时优化两者非常困难，例如 StackFLOW 必须依赖耗时的后优化过程才能得到合理结果。

**核心矛盾**：全局结构重建与局部接触精度之间存在 trade-off，显式建模方式难以平衡二者。此外，直接将人体 Transformer 方法（如 METRO）迁移到 HOI 任务会面临三个问题：所有 3D 点共享相同特征导致区分度不足、从静态模板学习交互姿态困难、以及自注意力会混淆人体与物体的独立拓扑边界。

**本文目标** 如何在不使用显式接触约束的情况下，利用 Transformer 隐式建模人物交互，并同时保持人体和物体各自拓扑结构的完整性。

**切入角度**：作者认为 Transformer 的自注意力天然适合捕捉全局交互，而图卷积网络（GCN）擅长建模局部拓扑。将二者结合，让自注意力处理全局交互、图卷积处理各自的局部结构，可以隐式而自然地学习 HOI。

**核心 idea**：用 Transformer 自注意力隐式建模人物全局交互，用嵌入在编码器中的图残差模块分别维护人体和物体的局部拓扑结构。

## 方法详解

### 整体框架
输入为一张包含 HOI 的 RGB 图像以及人体/物体的分割 mask，输出为人体 3D 网格（6890 顶点）和物体的 6D 位姿（旋转+平移）。整体流程分三步：(1) 通过预训练 ResNet50 提取图像特征并生成初始人体/物体网格；(2) 将初始顶点通过 grid sampling 与图像特征拼接构建 3D query（关节 query、人体顶点 query、物体顶点 query）；(3) 将所有 query 送入三层 HOI 重建 Transformer 编码器联合重建，最终通过上采样矩阵恢复人体全分辨率网格，通过刚体变换求解物体位姿。

### 关键设计

1. **3D Query 特征构建（Grid Sampling + 初始网格坐标）**:

    - 功能：为 Transformer 提供高区分度的输入 query
    - 核心思路：先用预训练 backbone 生成粗略的 SMPLH 参数和初始物体位姿，得到初始网格顶点。然后将每个 3D 顶点投影回 2D 图像坐标，通过 grid sampling 从特征图中提取对应特征，再与顶点 3D 坐标拼接。最终每个 query 维度为 $(2048+3)$。
    - 设计动机：解决"所有 3D 点共享相同特征"的问题——grid sampling 让每个顶点拥有独特的视觉特征；使用初始网格而非静态模板作为起点，降低从固定模板直接学习复杂交互的难度。消融实验证实这比用全局 pooling 特征+静态模板显著好。

2. **Human Graph Residual Block（人体图残差模块）**:

    - 功能：在 Transformer 编码器内部增强人体顶点的局部拓扑关系建模
    - 核心思路：在每个 Transformer encoder block 的多头注意力后，对人体顶点特征额外做一次图卷积：$Q'_{hv} = \sigma(\bar{A} Q^{mid}_{hv} W_G)$，其中 $\bar{A}$ 是人体网格的预定义邻接矩阵，保持了 SMPLH 模型的拓扑结构。采用残差连接。
    - 设计动机：Transformer 的自注意力是全局的，会混淆人体和物体的独立拓扑边界。图卷积利用预定义邻接矩阵在局部邻域内融合信息，帮助模型区分属于人体的顶点、保持人体拓扑完整性。

3. **Object Graph Residual Block（物体图残差模块）**:

    - 功能：针对不同物体模板构建专属的图结构，增强物体局部建模
    - 核心思路：与人体图残差模块结构类似，但邻接矩阵 $\bar{A}$ 根据不同物体模板用 KNN 图算法（K=10）动态构建。不同物体有不同拓扑（如椅子 vs 雨伞），需要不同的图结构。
    - 设计动机：不同物体拓扑差异很大，统一用自注意力难以准确建模对称物体或复杂形状的局部关系。KNN 图能自适应各种物体拓扑，帮助准确预测物体姿态。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{human} + \mathcal{L}_{object} + \mathcal{L}_{hbox}$：
- $\mathcal{L}_{human}$：包括多尺度顶点 L1 损失（431→1723→6890 三个尺度）、关节 L1 损失（初始+精炼的 3D/2D 坐标）、边长一致性损失、SMPLH 参数 L1 损失
- $\mathcal{L}_{object}$：物体顶点 L1 损失 + 旋转平移 L1 损失
- $\mathcal{L}_{hbox}$：手部 bounding box L1 损失
端到端训练，Transformer 编码器的三层隐藏维度分别为 1024、512、256 逐层递减。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HOI-TG | CONTHO (之前SOTA) | 提升 |
|--------|------|--------|-------------------|------|
| BEHAVE | CD_human ↓ | **4.59** | 4.99 | 8.0% |
| BEHAVE | CD_object ↓ | **8.00** | 8.42 | 5.0% |
| BEHAVE | Contact_p ↑ | **0.662** | 0.628 | +3.4% |
| BEHAVE | Contact_r ↑ | **0.554** | 0.496 | +5.8% |
| InterCap | CD_human ↓ | **5.43** | 5.96 | 8.9% |
| InterCap | CD_object ↓ | **8.68** | 9.50 | 8.6% |
| InterCap | Contact_p ↑ | **0.700** | 0.661 | +3.9% |
| InterCap | Contact_r ↑ | **0.473** | 0.432 | +4.1% |

### 消融实验

| 配置 | CD_human ↓ | CD_object ↓ | Contact_p ↑ | Contact_r ↑ |
|------|-----------|------------|------------|------------|
| Transformer only | 4.73 | 8.55 | 0.606 | 0.559 |
| +Human GRB | 4.61 | 8.11 | 0.651 | 0.539 |
| +Human GRB + Object GRB | **4.59** | **8.00** | **0.662** | **0.554** |
| Static query (全局特征+模板) | 4.95 | 8.90 | 0.632 | 0.472 |
| Initial query (grid sampling) | **4.59** | **8.00** | **0.662** | **0.554** |

### 关键发现
- 图残差模块贡献显著：仅加 Human GRB 就让 CD_object 从 8.55 降到 8.11，说明人体拓扑建模对物体重建也有间接帮助
- 初始网格 vs 静态模板差异巨大：使用 grid sampling + 初始网格的 query 在所有指标上大幅优于全局特征+静态模板，特别是 Contact_r 从 0.472 提升到 0.554
- KNN 邻居数 K=10 为最优，过少无法充分建模邻接关系，过多引入冗余导致性能下降
- 注意力可视化显示：简单交互场景中模型只关注局部身体部位；复杂交互中模型成功关注到非局部身体部位来推断物体位置

## 亮点与洞察
- **隐式交互建模替代显式约束**：用 Transformer 自注意力自然学习人物交互，无需手工设计 contact map 或 offset 约束，降低了工程复杂度同时效果更好。这个思路表明很多"显式约束"可以被足够强的注意力机制隐式覆盖。
- **在 Transformer 中嵌入 GCN 的范式**：图残差模块作为 Transformer block 的组件，兼顾全局注意力和局部拓扑。这种混合架构设计可以迁移到任何需要同时处理全局关系和局部结构的 3D 任务中。
- **针对不同物体的自适应图结构**：用 KNN 动态构建物体邻接矩阵，使框架能泛化到不同拓扑的物体。

## 局限与展望
- 作者承认在躺卧姿态和完全对称物体上表现不佳
- 方法依赖于物体 3D 模板的先验知识，无法处理未知物体
- 推理时需要人体和物体的 segmentation mask 作为输入，实际应用中需要额外的分割模型
- 仅在室内数据集上验证，户外复杂背景下的泛化能力未知
- 可考虑引入多帧时序信息来处理更复杂的动态交互场景

## 相关工作与启发
- **vs CONTHO**: 共享相同初始网格，但 CONTHO 用显式 contact map 约束来精炼，HOI-TG 用隐式自注意力+图残差块。HOI-TG 在所有指标上更优，证明隐式方式更合理。
- **vs Graphormer**: Graphormer 将 GCN 集成到 Transformer 用于单人体重建，HOI-TG 将此思路扩展到 HOI 双主体场景，并为人和物体设计了不同的图残差块。
- **vs StackFLOW**: StackFLOW 依赖耗时的后优化过程，HOI-TG 端到端推理效率更高。

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐式交互建模思路清晰，但 Transformer+GCN 混合架构非首创
- 实验充分度: ⭐⭐⭐⭐ 两个数据集全面对比+多组消融+注意力可视化
- 写作质量: ⭐⭐⭐⭐ 动机分析清楚，方法描述条理分明
- 价值: ⭐⭐⭐⭐ HOI 重建的实用改进，隐式建模思路有启发性

<!-- RELATED:START -->

## 相关论文

- [End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)
- [ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)
- [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)

<!-- RELATED:END -->
