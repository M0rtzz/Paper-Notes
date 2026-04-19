---
title: >-
  [论文解读] PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior
description: >-
  [ICCV 2025][图学习][草图到3D生成] 提出PASTA框架，通过集成VLM文本先验补偿草图的语义缺失，并设计ISG-Net（IndivGCN+PartGCN双图卷积）建模零件间结构关系，实现SOTA的草图到3D形状生成和零件级编辑。
tags:
  - ICCV 2025
  - 图学习
  - 草图到3D生成
  - 零件级编辑
  - 视觉语言模型
  - 图卷积网络
  - 高斯混合模型
---

# PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior

**会议**: ICCV 2025  
**arXiv**: [2503.12834](https://arxiv.org/abs/2503.12834)  
**代码**: 无  
**领域**: 图学习 / 3D生成  
**关键词**: 草图到3D, 文本先验, 图卷积网络, 部件感知, VLM

## 一句话总结
提出PASTA框架，通过VLM文本先验补充草图缺失的语义信息，并用ISG-Net（IndivGCN+PartGCN）建模部件间关系，实现了草图到3D形状生成的SOTA性能，支持部件级编辑。

## 研究背景与动机

**领域现状**：条件3D生成主要基于草图或文本，但两者各有明显局限——文本缺乏精确几何控制，草图缺乏语义信息且存在歧义。

**现有痛点**：纯草图方法（LAS-D、SENS）难以从简化草图恢复完整结构（如缺少扶手、腿数不对）；纯文本方法无法精确控制几何。

**核心矛盾**：如何同时利用草图的几何控制力和文本的语义表达力来生成准确的3D形状。

**本文切入角度**：利用VLM从草图中自动提取部件描述（如"椅背形状、座椅、4条腿、无扶手"），作为文本先验补充草图缺失的语义线索。

**核心idea**：文本先验 + 视觉-文本Transformer解码器融合两种条件 + ISG-Net双GCN建模部件间结构关系。

## 方法详解

### 整体框架
输入草图 → 视觉骨干提取视觉嵌入$\mathcal{V}$ + VLM提取文本嵌入$\mathcal{T}$ → Text-Visual Transformer Decoder融合两种条件到可学习查询$\mathbf{Q}$ → ISG-Net精化部件结构 → SPAGHETTI形状解码器生成3D mesh。

### 关键设计

1. **Text-Visual Transformer Decoder**:

    - 功能：将视觉和文本条件融合到$N$个可学习查询中，每个查询对应一个GMM
    - 核心思路：查询先自注意力 → 与视觉嵌入交叉注意力$\mathbf{Q}_\mathcal{V}$ → 再自注意力 → 与文本嵌入交叉注意力$\mathbf{Q}_{\mathcal{TV}}$，迭代12次
    - 设计动机：草图抽象简化导致视觉信息不足，文本先验能指明"有几条腿、有无扶手"等部件组成

2. **IndivGCN（细粒度特征处理）**:

    - 功能：建模个体GMM之间的空间关系
    - 核心思路：用MLP预测邻接矩阵$\tilde{\mathbf{A}}_I$（用GMM中心间距离作为伪ground truth监督），然后做图卷积$\mathbf{Q}_{indiv} = \sigma(\tilde{\mathbf{A}}_I \mathbf{Q}_{\mathcal{TV}} \mathbf{W}_I)$
    - 设计动机：不同GMM之间的距离关系反映了它们在3D空间的连接关系

3. **PartGCN（部件级结构聚合）**:

    - 功能：将GMM聚类到部件级别，建模部件间结构关系
    - 核心思路：用层次聚类将$N$个GMM分为$K$个部件，平均池化得部件查询$\mathbf{Q}_P$，预测部件邻接矩阵并做图卷积，再反池化回原尺寸
    - 设计动机：部件级关系（如"腿和座面连接"）比个体GMM关系更能保证结构完整性

最终输出：$\mathbf{Q}_{final} = norm(\alpha \mathbf{Q}_{indiv} + (1-\alpha)\mathbf{Q}_{part} + \mathbf{Q}_{\mathcal{TV}})$

### 损失函数 / 训练策略
$\mathcal{L} = \lambda_{align}\mathcal{L}_{align} + \lambda_{indiv}\mathcal{L}_{indiv} + \lambda_{part}\mathcal{L}_{part}$，其中$\mathcal{L}_{align}$用L1损失对齐预测向量与SPAGHETTI反演的GT向量，$\mathcal{L}_{indiv}$和$\mathcal{L}_{part}$分别用MSE监督邻接矩阵预测。

## 实验关键数据

### 主实验

| 方法 | AmateurSketch CD↓ | EMD↓ | FID↓ | ProSketch CD↓ | EMD↓ | FID↓ |
|------|-------------------|------|------|----------------|------|------|
| Sketch2Mesh | 0.257 | 0.211 | 392.2 | 0.228 | 0.171 | 297.8 |
| LAS-D | 0.159 | 0.128 | 197.5 | 0.195 | 0.147 | 193.5 |
| SENS | 0.121 | 0.096 | 171.3 | 0.116 | 0.076 | 160.5 |
| DY3D | 0.109 | 0.091 | - | 0.093 | 0.087 | - |
| **PASTA** | **0.090** | **0.071** | **143.9** | **0.055** | **0.049** | **112.2** |

### 消融实验

| 配置 | CD↓ | EMD↓ | FID↓ |
|------|-----|------|------|
| 无文本先验 | 0.121 | 0.096 | 171.3 |
| 无ISG-Net | 0.105 | 0.083 | 156.2 |
| PASTA完整 | **0.090** | **0.071** | **143.9** |

### 关键发现
- PASTA在所有指标上全面超越现有方法，ProSketch上CD从0.093降到0.055（41%提升）
- 文本先验有效补充了草图缺失的部件信息（如缺少扶手、腿数错误）
- ISG-Net的双GCN设计显著改善结构一致性
- 方法可扩展到飞机、台灯等其他类别

## 亮点与洞察
- **用VLM自动从草图提取部件描述作为文本先验**：避免了人工文本标注，VLM能识别"有4条腿、圆形座面"等结构信息
- **IndivGCN + PartGCN的层次图建模**：类似从像素到区域的层次推理，既保留细节又保证全局结构
- **支持部件级编辑**：基于GMM的部件分解允许添加、删除、变换部件

## 局限与展望
- 依赖SPAGHETTI预训练解码器，受限于其支持的物体类别（椅子、飞机、台灯）
- VLM对复杂草图的理解能力有限
- 仅支持单视角草图输入
- 可尝试扩展到更多物体类别和开放词汇3D生成

## 相关工作与启发
- **vs SENS**: SENS是纯视觉草图到3D，本文加入文本先验显著提升
- **vs DY3D**: DY3D支持用户交互编辑但不用文本，本文用VLM自动补充语义
- **vs 文本到3D方法**: 文本到3D缺乏精确几何控制，本文用草图提供几何约束

## 评分
- 新颖性: ⭐⭐⭐⭐ VLM文本先验+双GCN结构建模思路清晰且有效
- 实验充分度: ⭐⭐⭐⭐ 多类别多数据集+消融+编辑演示
- 写作质量: ⭐⭐⭐⭐ 图表清晰，方法描述条理分明
- 价值: ⭐⭐⭐⭐ 草图3D生成的实用方法，支持部件编辑
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

## 相关论文

- [Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [Nonlinear Laplacians: Tunable Principal Component Analysis under Directional Prior Information](../../NeurIPS2025/graph_learning/nonlinear_laplacians_tunable_principal_component_analysis_under_directional_prio.md)
- [Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](../../ACL2025/graph_learning/can_graph_neural_networks_learn_language.md)
- [Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](../../AAAI2026/graph_learning/format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)

<!-- RELATED:END -->
