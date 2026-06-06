---
title: >-
  [论文解读] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation
description: >-
  [NeurIPS 2025][图学习][开放词汇场景图生成] 本文提出ACC框架，通过交互驱动范式（而非传统以对象为中心的范式）来解决开放词汇场景图生成中的关键匹配问题：在知识注入阶段用双向交互提示生成更准确的伪监督，在知识迁移阶段用交互引导的查询选择和交互一致性知识蒸馏来减少不匹配…
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "开放词汇场景图生成"
  - "交互建模"
  - "知识蒸馏"
  - "视觉语言模型"
  - "伪监督"
---

# Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation

**会议**: NeurIPS 2025  
**arXiv**: [2511.05935](https://arxiv.org/abs/2511.05935)  
**代码**: [GitHub](https://github.com/HKUST-LongGroup/ACC)  
**领域**: 图学习 / 场景图生成  
**关键词**: 开放词汇场景图生成, 交互建模, 知识蒸馏, 视觉语言模型, 伪监督

## 一句话总结
本文提出ACC框架，通过交互驱动范式（而非传统以对象为中心的范式）来解决开放词汇场景图生成中的关键匹配问题：在知识注入阶段用双向交互提示生成更准确的伪监督，在知识迁移阶段用交互引导的查询选择和交互一致性知识蒸馏来减少不匹配，在VG、GQA、PSG三个基准上达到SOTA。

## 研究背景与动机
场景图生成（SGG）旨在将图像映射为结构化语义表示，其中对象是节点、关系是边。开放词汇场景图生成（OVSGG）进一步要求模型能识别训练中未见过的新对象和关系类别，通常利用预训练视觉语言模型（VLM）的知识。

**现有痛点**：
- 当前OVSGG采用两阶段流程：(1) 知识注入——在大规模数据上预训练VLM；(2) 知识迁移——用全标注数据微调。两个阶段都采用以对象为中心的范式，忽略了同类别对象中交互实例和非交互实例的区别
- **知识注入阶段**：仅用对象类别名（如"man", "surfboard"）定位对象，导致大量候选配对中无法正确关联交互关系，产生噪声伪监督
- **知识迁移阶段**：大量对象查询候选中，非交互的"man"查询可能被错误匹配到标注中参与"riding"的"man"，导致关系分类混乱

**核心矛盾**：如何让模型在两个阶段都区分"参与交互的对象"和"未参与交互的对象"？

**切入角度**：从对象中心范式转向交互中心范式，在知识注入和迁移两个阶段都显式建模交互关系。

## 方法详解

### 整体框架
ACC（interACtion-Centric）是一个端到端的OVSGG框架，基于双编码器-单解码器架构（类似GroundingDINO）。视觉编码器提取多尺度特征，文本编码器处理类别提示，DETR式解码器通过自注意力和交叉注意力优化对象查询。ACC的核心改进在知识注入和知识迁移两个阶段引入交互中心设计。

### 关键设计

1. **交互中心知识注入——双向交互提示（BIP）**:

    - 传统方法用孤立的对象名（如"man. surfboard."）作为检测提示，缺少交互上下文
    - BIP构建双向提示：正向（"man hold surfboard"）+ 反向（"surfboard held by man"）
    - **上下文建模**：通过文本编码器的注意力机制，"man"这个token能吸收"hold surfboard"的交互语义，使定位更精准——优先定位参与交互的那个"man"
    - **角色感知增强**：反向提示将关系客体（"surfboard"）变成句法主语，在注意力中获得更高权重，提升其定位精度
    - 配合基于IoU的规则组合策略，将重叠的主体/客体框组合为三元组伪监督

2. **交互引导的查询选择（IGQS）**:

    - **第一步**：计算每个视觉token的交互相关性得分$s_i = (\max(\mathbf{v}_i \mathbf{T}_o^\top))^\gamma \cdot (\max(\mathbf{v}_i \mathbf{T}_r^\top))^{1-\gamma}$，同时考虑对象和关系语义的相似性，选择top-K个查询
    - **第二步**：利用第一步预测的关系三元组，分解为⟨subject, predicate⟩和⟨predicate, object⟩交互对，编码为交互token。用交互相关性得分选择top-L个查询，剩余K-L个用对象相关性补充
    - 两步策略的优势：优先选择参与交互的对象查询，同时保留未出现在初始预测中但对场景理解重要的对象
    - 分解三元组为配对避免了不同对象token之间的直接干扰

3. **交互一致性知识蒸馏（ICKD）**:

    - **视觉概念保留蒸馏（VRD）**：对负样本的边特征，用L1损失保持学生和教师的逐点语义一致：$\mathcal{L}_{VRD} = \frac{1}{|\mathcal{N}|}\sum \|\mathbf{e}_S - \mathbf{e}_T\|_1$
    - **相对交互保留蒸馏（RRD）**：建模三元组嵌入之间的结构相似性矩阵，用Frobenius范数对齐教师和学生：$\mathcal{L}_{RRD} = \frac{1}{|\mathcal{N}|^2}\|\mathbf{M}_S - \mathbf{M}_T\|_F^2$
    - VRD保持逐点语义一致，RRD保持配对之间的相对关系一致（交互对 vs 背景对），两者互补
    - 整体避免灾难性遗忘，增强对新类别三元组的泛化

### 损失函数 / 训练策略
最终损失函数组合了定位损失（L1回归+GIoU）、分类损失（对象+关系的交叉熵）和蒸馏损失：
$$\mathcal{L} = \mathcal{L}_{reg} + \mathcal{L}_{giou} + \mathcal{L}_{obj} + \mathcal{L}_{rel} + \beta_1 \mathcal{L}_{VRD} + \beta_2 \mathcal{L}_{RRD}$$

预训练阶段使用COCO captions的图文对生成伪监督，微调阶段在VG/GQA/PSG数据集上有监督训练。

## 实验关键数据

### 主实验（VG OvD+R-SGG设置）

| 方法 | Backbone | Joint B+N R@100 | Novel(Obj) R@100 | Novel(Rel) R@100 |
|------|----------|----------------|-----------------|-----------------|
| ACC (Ours) | Swin-T | **19.55** | **19.65** | **17.83** |
| ESGG | Swin-T | 16.37 | 17.48 | 11.18 |
| VS³ | Swin-T | 11.56 | 11.97 | 8.82 |
| OvSGTR | ResNet-50 | 11.12 | 12.09 | 9.19 |
| Faster R-CNN | Swin-T | 14.58 | 15.92 | 10.93 |

### 消融实验

| 配置 | Joint B+N R@100 | Novel(Rel) R@100 | 说明 |
|------|----------------|-----------------|------|
| Baseline | 16.37 | 11.18 | 无IGQS和ICKD |
| +IGQS | 19.37 | 17.38 | 查询选择优先交互对象 |
| +ICKD | 19.20 | 17.32 | 交互一致性蒸馏 |
| +IGQS+ICKD | **19.55** | **17.83** | 组合最优 |
| w/o BIP | 17.82 | 16.20 | 无双向交互提示 |
| w/ BIP | **19.55** | **17.83** | BIP提升1.73% |

### 关键发现
- IGQS贡献最大（R@100提升3.00%），说明减少非交互候选的匹配是核心瓶颈
- ICKD的RRD组件通过保持交互对与背景对的相对关系，显著提升新关系类别的泛化
- BIP在预训练阶段就提供更干净的伪监督，为后续微调奠定更好的基础
- 两个组件各自独立有效，但组合后的增量提升略低于预期——因为两者都在减少非交互对象，存在递减效应

## 亮点与洞察
- **范式转换**：从"对象中心"到"交互中心"的范式转换简洁有力，识别了OVSGG两个阶段的共性问题（同类别不同实例的交互区分）
- **双向交互提示**：巧妙利用文本编码器的注意力机制将交互上下文注入对象定位，无需额外模型
- **两步查询选择**：先用交互语义筛选，再用对象语义补充，兼顾精度和召回
- 将知识蒸馏从简单的逐点对齐升级为结构感知的交互一致性对齐

## 局限与展望
- 依赖语言解析器提取初始三元组，解析质量制约上游伪监督质量
- IGQS的两步流程增加了推理时的计算开销（需要额外的前向传播）
- 仅在VLM-based方法上验证，未涉及MLLM-based方法（如LLaVA系列）
- 反向提示的动词转换依赖LLM或规则库，对复杂关系可能不够鲁棒
- 实验主要在VG/GQA/PSG上进行，这些数据集的类别偏向长尾分布

## 相关工作与启发
- **vs ESGG**: ESGG也使用GroundingDINO和知识蒸馏，但采用对象中心范式；ACC通过交互中心设计在所有指标上大幅领先
- **vs VS³**: VS³关注视觉-语义预训练对齐，但缺少对交互的显式建模
- **vs OvSGTR**: OvSGTR提出开放词汇SGG的统一框架，但同样未区分交互/非交互对象
- **启发**：交互中心范式不仅适用于场景图生成，在人物交互检测（HOI）、动作识别等需要建模对象间关系的任务中也可能有价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 交互中心范式的提出清晰有力，但各个组件（交互提示、查询选择、结构蒸馏）均有已有工作的影子
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、两种设置、充分消融、预训练对比均有覆盖
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法设计的叙述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为OVSGG提供了有效的交互中心范式，但场景图生成领域本身的应用影响力相对有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[CVPR 2025\] Universal Scene Graph Generation](../../CVPR2025/graph_learning/universal_scene_graph_generation.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](../../ICML2026/graph_learning/when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[CVPR 2025\] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](../../CVPR2025/graph_learning/unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
