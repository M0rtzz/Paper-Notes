---
title: >-
  [论文解读] PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior
description: >-
  [ICCV 2025][图学习][草图到3D生成] 提出PASTA框架，通过集成VLM文本先验补偿草图的语义缺失，并设计ISG-Net（IndivGCN+PartGCN双图卷积）建模零件间结构关系，实现SOTA的草图到3D形状生成和零件级编辑。 1. 领域现状：条件3D形状生成主要有草图和文本两种输入，草图提供几何精确控制但…
tags:
  - "ICCV 2025"
  - "图学习"
  - "草图到3D生成"
  - "零件级编辑"
  - "视觉语言模型"
  - "图卷积网络"
  - "高斯混合模型"
---

# PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior

**会议**: ICCV 2025  
**arXiv**: [2503.12834](https://arxiv.org/abs/2503.12834)  
**代码**: 无  
**领域**: 3D视觉 / 图学习  
**关键词**: 草图到3D生成, 零件级编辑, 视觉语言模型, 图卷积网络, 高斯混合模型

## 一句话总结
提出PASTA框架，通过集成VLM文本先验补偿草图的语义缺失，并设计ISG-Net（IndivGCN+PartGCN双图卷积）建模零件间结构关系，实现SOTA的草图到3D形状生成和零件级编辑。

## 研究背景与动机

1. **领域现状**：条件3D形状生成主要有草图和文本两种输入，草图提供几何精确控制但缺乏语义，文本提供语义但缺乏精确几何控制。
2. **现有痛点**：单一草图信息过于简化和模糊，导致零件丢失（如椅子缺扶手）和结构不准确；现有方法如SENS、DY3D仅用视觉特征，无法补偿草图中缺失的语义线索。
3. **核心矛盾**：如何从高度简化的2D草图中准确推断出完整的3D零件结构和语义属性。
4. **本文切入角度**：用VLM对草图进行文本描述（如"椅子有4条腿、有扶手"），在零件级表示（GMM）上建图做结构推理。
5. **核心idea**：文本先验补偿视觉缺失+图卷积网络建模零件关系=更准确完整的3D形状生成。

## 方法详解

### 整体框架
输入草图→视觉backbone提取视觉嵌入$\mathcal{V}$→VLM提取文本嵌入$\mathcal{T}$（描述零件组成）→Text-Visual Transformer Decoder融合两种模态到N个可学习查询→ISG-Net（IndivGCN+PartGCN）细化结构关系→MLP映射到SPAGHETTI潜向量→形状解码器生成3D mesh。

### 关键设计

1. **Text-Visual Transformer Decoder**:
    - 功能：将视觉和文本条件融合到可学习查询中
    - 核心思路：N个可学习查询先做self-attention→再与视觉嵌入做visual cross-attention→再与文本嵌入做text cross-attention，共迭代12次。$\mathbf{Q}_{\mathcal{TV}} = Attn(W_Q^T \cdot \mathbf{Q}_\mathcal{V}, W_K^T \cdot \mathcal{T}, W_V^T \cdot \mathcal{T})$
    - 设计动机：文本先验提供草图中不易观察的语义信息（如零件数量、是否有扶手等），弥补视觉backbone的不足

2. **IndivGCN（细粒度特征处理）**:
    - 功能：建模个体GMM之间的空间关系
    - 核心思路：用MLP从查询预测邻接矩阵$\tilde{\mathbf{A}}_I$（基于GMM中心距离的伪GT监督），然后做图卷积$\mathbf{Q}_{indiv} = \sigma(\tilde{\mathbf{A}}_I \mathbf{Q}_{\mathcal{TV}} \mathbf{W}_I)$
    - 设计动机：让每个GMM感知其空间邻居的信息，细化局部几何细节

3. **PartGCN（零件级结构聚合）**:
    - 功能：将GMM聚类为零件并建模零件间关系
    - 核心思路：用层次聚类将N个GMM分为K个零件组→平均池化得到零件级查询→预测零件邻接矩阵→零件级图卷积→unpool回个体级
    - 设计动机：零件级的粗粒度结构建模保证全局一致性

最终融合：$\mathbf{Q}_{final} = norm(\alpha \mathbf{Q}_{indiv} + (1-\alpha)\mathbf{Q}_{part} + \mathbf{Q}_{\mathcal{TV}})$

### 损失函数 / 训练策略
$\mathcal{L} = \lambda_{align}\mathcal{L}_{align} + \lambda_{indiv}\mathcal{L}_{indiv} + \lambda_{part}\mathcal{L}_{part}$，其中$\mathcal{L}_{align}$是预测潜向量与GT潜向量的L1距离，$\mathcal{L}_{indiv}$和$\mathcal{L}_{part}$是邻接矩阵预测的MSE损失。

## 实验关键数据

### 主实验

| 方法 | AmateurSketch-3D | | ProSketch-3D | |
|------|------|------|------|------|
| | CD↓ | EMD↓ | CD↓ | EMD↓ |
| Sketch2Mesh | 0.257 | 0.211 | 0.228 | 0.171 |
| LAS-D | 0.159 | 0.128 | 0.195 | 0.147 |
| SENS | 0.121 | 0.096 | 0.116 | 0.076 |
| DY3D | 0.109 | 0.091 | 0.093 | 0.087 |
| **PASTA** | **0.090** | **0.071** | **0.055** | **0.049** |

| 方法 | Airplane CD↓ | Lamp CD↓ |
|------|-------------|----------|
| SENS | 0.240 | 0.253 |
| **PASTA** | **0.188** | **0.195** |

### 消融实验

| 配置 | CD↓ | EMD↓ |
|------|-----|------|
| 仅视觉backbone | 0.115 | 0.092 |
| + 文本先验 | 0.098 | 0.078 |
| + IndivGCN | 0.095 | 0.075 |
| + PartGCN (完整PASTA) | **0.090** | **0.071** |

### 关键发现
- 在ProSketch-3D上CD相比DY3D降低41%，EMD降低44%，提升巨大
- 文本先验贡献最大（CD从0.115降到0.098），证实VLM语义信息对补偿草图模糊性至关重要
- 双图卷积进一步带来稳定提升，PartGCN的零件级建模比IndivGCN的贡献更大
- 可扩展到真实图片输入，展示系统鲁棒性

## 亮点与洞察
- **VLM作为草图语义增强器的思路非常实用**：VLM能识别出草图中"有几条腿"、"有无扶手"等人类肉眼也难从简笔画中判断的信息
- **双粒度GCN设计精巧**：IndivGCN负责细节、PartGCN负责结构，两者互补覆盖了不同尺度的几何关系
- **支持零件级编辑**：基于GMM表示天然支持添加/删除/变换零件

## 局限与展望
- 仅在ShapeNet的椅子/飞机/台灯上训练和评估，类别有限
- 依赖SPAGHETTI预训练形状解码器，受限于解码器的表示能力
- VLM对草图的描述质量可能不稳定
- 可探索将框架扩展到更复杂形状（如多部件机械、人体等）

## 相关工作与启发
- **vs SENS**: SENS仅用视觉特征，本文加入文本先验和图结构推理，所有指标显著更优
- **vs DY3D**: DY3D也使用零件级表示，但没有文本增强和图卷积建模零件关系
- **vs 文本到3D方法**: 文本条件缺乏几何控制，本文用草图+文本结合了两者优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本+草图融合+双GCN的组合新颖但非颠覆性
- 实验充分度: ⭐⭐⭐⭐ 多数据集定量+定性，有消融但类别有限
- 写作质量: ⭐⭐⭐⭐ 图文清晰，架构描述详细
- 价值: ⭐⭐⭐⭐ 对交互式3D内容创建有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching](../../ECCV2024/graph_learning/synchronous_diffusion_for_unsupervised_smooth_non-rigid_3d_shape_matching.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[NeurIPS 2025\] Nonlinear Laplacians: Tunable Principal Component Analysis under Directional Prior Information](../../NeurIPS2025/graph_learning/nonlinear_laplacians_tunable_principal_component_analysis_under_directional_prio.md)

</div>

<!-- RELATED:END -->
