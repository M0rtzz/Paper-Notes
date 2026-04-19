---
title: >-
  [论文解读] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs
description: >-
  [NeurIPS 2025][目标检测][文本属性图] 提出 SSTAG，通过双重知识蒸馏将 LLM 和 GNN 的互补知识联合蒸馏到结构感知的 MLP 中，结合内存库机制存储原型表示，实现高效、可扩展的文本属性图跨域自监督预训练。
tags:
  - NeurIPS 2025
  - 目标检测
  - 文本属性图
  - 自监督学习
  - 知识蒸馏
  - 跨域迁移
  - 图基础模型
---

# SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs

**会议**: NeurIPS 2025  
**arXiv**: [2510.01248](https://arxiv.org/abs/2510.01248)  
**代码**: 无  
**领域**: Graph Learning / Self-Supervised Learning  
**关键词**: 文本属性图, 自监督学习, 知识蒸馏, 跨域迁移, 图基础模型

## 一句话总结

提出 SSTAG，通过双重知识蒸馏将 LLM 和 GNN 的互补知识联合蒸馏到结构感知的 MLP 中，结合内存库机制存储原型表示，实现高效、可扩展的文本属性图跨域自监督预训练。

## 研究背景与动机

NLP 和 CV 领域的大规模预训练模型已展现出卓越的跨域泛化能力，但图学习领域仍停留在**单图训练**的范式中。这种方式存在两大局限：（1）模型局限于单任务或窄任务，缺乏跨图知识迁移能力；（2）模型性能严重依赖大量标注数据，而高质量标注成本高昂，在低资源场景下形成瓶颈。

构建图基础模型面临独特挑战：不同于 NLP 的统一词汇空间或 CV 的一致像素空间，图数据具有**领域异质性**（不同图域有不同特征空间和标签体系）和**结构多样性**（引文网络 vs 知识图谱的结构截然不同）。

SSTAG 的核心洞察是：利用**文本作为统一表示媒介**，因为许多真实图天然是文本属性图 (TAG)。LLM 擅长文本理解但不擅长拓扑推理，GNN 擅长结构建模但缺乏开放世界知识。SSTAG 通过蒸馏将两者的互补能力融合到轻量级 MLP 中。

## 方法详解

### 整体框架

SSTAG 包含三个核心模块：（1）统一图任务 (UGT) 模块——基于子图采样统一节点/边/图级任务；（2）LLM 知识提取 (KEL) 模块——结合 LM 和 GNN 的掩码自编码预训练；（3）知识蒸馏 (KD) 模块——将 LM+GNN 教师的知识蒸馏到结构感知 MLP 学生中。

### 关键设计

1. **统一图任务 (UGT)**: 采用基于 Personalized PageRank (PPR) 的子图采样策略，为每个目标节点/边构建上下文子图。对节点 $v$，PPR 重要性得分为 $\pi_v = \alpha(\mathbf{I} - (1-\alpha)\tilde{\mathbf{A}})^{-1}\mathbf{e}_v$，k跳邻居 $u$ 的采样概率正比于 $\pi_{vu}/\sum_{w \in \mathcal{N}_k(v)} \pi_{vw}$。这种策略消弭了不同域图结构的差异，且对大规模图有更好的可扩展性。边级任务取两端点子图的并集，图级任务直接使用完整图。

2. **LLM 知识提取与掩码预训练**: 教师模型由语言模型 (Sentence Transformer) + GCN 级联组成。采用掩码语言建模 (MLM) 目标——随机掩码节点文本中的 token，利用文本上下文和邻域信息重建。编码过程：文本通过 LM 得到逐 token 嵌入 $\mathbf{E}_v$，[CLS] token 经 GNN 传播得到 $\mathbf{H}^{\text{cls}}$，两者拼接后通过线性层融合：$\mathbf{H}_v = \text{Linear}(\mathbf{E}_v \oplus (\mathbf{H}_v^{\text{cls}} \otimes \mathbf{1}_{n_v+2}^\top))$，最后通过 MLM Head 预测被掩码的 token。这驱使模型同时学习语义关联和结构模式。

3. **知识蒸馏到结构感知 MLP**: 学生模型为轻量级 MLP，将节点的 [CLS] 嵌入与 PPR 得分拼接作为输入：$\tilde{\mathbf{H}}_v^{\text{cls}} = f_{\text{MLP}}([\tilde{\mathbf{E}}_v^{\text{cls}} \| p_v])$。通过 PPR 分数注入结构信息，无需显式消息传递，大幅降低计算开销。推理时只需 LM + MLP，不需要 GNN 组件。

4. **内存库 (Memory Bank)**: 维护 $L$ 个可学习的原型锚点 $\{\mathbf{a}_j\}_{j=1}^L$，通过注意力机制与输入图表示交互。节点表示通过 softmax 注意力与锚点加权求和得到重构嵌入 $\hat{\mathbf{H}}_v = \sum_j s'_{vj} \mathbf{a}_j$。内存库保存跨训练实例的不变知识，通过对齐促使模型关注稳定、一致的特征，增强泛化能力。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{ST}} + \mathcal{L}_{\text{ME}}$：

- **掩码损失** $\mathcal{L}_{\text{mask}}$：标准 MLM 交叉熵，迫使模型通过邻节点文本信息重建被掩码 token
- **师生一致性损失** $\mathcal{L}_{\text{ST}}$：学生与教师表示的余弦相似度对齐
- **内存一致性损失** $\mathcal{L}_{\text{ME}}$：节点嵌入与内存库重构嵌入的 L2 距离，引导内存锚点捕获图的不变特征

在 ogbn-Paper100M 上预训练后，通过线性探测协议在 12 个跨域目标数据集上评估。

## 实验关键数据

### 主实验

**节点分类（跨域迁移, Table 1, 预训练在 ogbn-Paper100M）**:

| 方法 | Cora | Pubmed | ogbn-Arxiv | WikiCS | Products |
|------|------|--------|------------|--------|----------|
| GCN (监督) | 57.62 | 55.18 | 60.85 | 53.24 | 61.95 |
| GraphMAE2 | 73.92 | 68.76 | 69.07 | 58.04 | 74.05 |
| UniGraph | 74.65 | 70.84 | 70.89 | 65.47 | 76.58 |
| **SSTAG** | **75.09** | **72.65** | **72.85** | **68.76** | **78.27** |

**链接预测 + 图分类（Table 1 & 2）**:

| 方法 | FB15K237 | WN18RR | HIV | BACE |
|------|----------|--------|-----|------|
| UniGraph | 85.01 | 80.55 | 77.27 | 79.23 |
| Graph-LLM | 82.47 | 73.46 | 76.43 | 80.65 |
| **SSTAG** | **88.64** | **82.42** | **79.52** | **82.06** |

### 消融实验

**关键组件消融 (Table 3)**:

| 配置 | WikiCS | ogbn-Arxiv | FB15K237 | MUV |
|------|--------|------------|----------|-----|
| SSTAG (完整) | **68.76** | **72.85** | **88.64** | **79.86** |
| W/o $\mathcal{L}_{\text{mask}}$ | 67.02 | 70.51 | 85.84 | 76.22 |
| W/o $\mathcal{L}_{\text{ST}}$ | 67.75 | 71.86 | 87.12 | 78.65 |
| W/o $\mathcal{L}_{\text{ME}}$ | 66.53 | 71.14 | 85.96 | 76.43 |
| W/o GNN | 64.34 | 69.53 | 84.32 | 70.57 |

### 关键发现

- 移除 GNN 组件导致最大性能下降（WikiCS -4.42, MUV -9.29），说明结构信息至关重要
- 内存一致性损失 ($\mathcal{L}_{\text{ME}}$) 的移除影响较大，验证了内存库在跨域泛化中的重要性
- 在 BACE 数据集上微调后达到 82.06%，比监督基线 GCN 高出 12.21 个百分点
- 作为纯自监督预训练模型，在多个数据集上匹配或超越完全监督方法

## 亮点与洞察

- **统一多粒度任务的子图表示方法**实用且优雅，PPR 采样兼顾了重要性和可扩展性
- **双重蒸馏设计**巧妙：教师模型获取完整的 LM+GNN 能力，学生模型通过 PPR 注入结构信息，推理时仅需 LM+MLP，大幅降低部署成本
- **内存库机制**新颖，为跨域图学习提供了不变知识锚定

## 局限与展望

- 预训练仅在 ogbn-Paper100M（引文网络）上进行，对非引文结构（如社交网络、分子图）的泛化有待验证
- 学生模型通过 PPR 得分隐式注入结构，丢失了部分精确拓扑信息
- 内存库大小 $L$ 的选择缺乏理论指导
- 未与最新的图基础模型方法（如 GraphMAE 的扩展版本）进行更全面对比

## 相关工作与启发

- 与 UniGraph 等方法思路相似但技术路线不同，可考虑结合已有的图 prompt 方法
- 蒸馏到 MLP 的思路与 GLNN 等 GNN-to-MLP 蒸馏方向一脉相承，但加入了 LLM 维度
- 可扩展到异质图或动态图场景

## 评分

- 新颖性: ⭐⭐⭐⭐ (LLM+GNN 双蒸馏到 MLP 具有新意，内存库机制锦上添花)
- 实验充分度: ⭐⭐⭐⭐ (12个跨域数据集，多任务评测)
- 写作质量: ⭐⭐⭐⭐ (方法描述清晰，公式推导完整)
- 价值: ⭐⭐⭐⭐ (为图基础模型提供了可行的技术路径)

<!-- RELATED:START -->

## 相关论文

- [M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)
- [WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](../../ECCV2024/object_detection/walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](../../ICML2025/object_detection/discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](../../ECCV2024/object_detection/wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)
- [Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](../../ICML2025/object_detection/self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)

<!-- RELATED:END -->
