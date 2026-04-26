---
title: >-
  [论文解读] Beyond Caption-Based Queries for Video Moment Retrieval
description: >-
  [CVPR 2026][目标检测][视频时刻检索] 揭示了VMR中caption-based查询与真实用户搜索查询之间的巨大鸿沟，提出了三个搜索查询基准，并通过移除自注意力+查询Dropout两项架构修改来缓解DETR中的解码器查询坍塌问题，在多时刻搜索查询上提升高达21.83% mAPm。
tags:
  - CVPR 2026
  - 目标检测
  - 视频时刻检索
  - 搜索查询泛化
  - DETR解码器查询坍塌
  - 多时刻检索
  - 查询欠规范化
---

# Beyond Caption-Based Queries for Video Moment Retrieval

**会议**: CVPR 2026  
**arXiv**: [2603.02363](https://arxiv.org/abs/2603.02363)  
**代码**: 有（项目主页提供代码、模型和数据）  
**领域**: 目标检测  
**关键词**: 视频时刻检索, 搜索查询泛化, DETR解码器查询坍塌, 多时刻检索, 查询欠规范化

## 一句话总结

揭示了VMR中caption-based查询与真实用户搜索查询之间的巨大鸿沟，提出了三个搜索查询基准，并通过移除自注意力+查询Dropout两项架构修改来缓解DETR中的解码器查询坍塌问题，在多时刻搜索查询上提升高达21.83% mAPm。

## 研究背景与动机

### 1. 领域现状
视频时刻检索（Video Moment Retrieval, VMR）旨在根据文本查询定位视频中的时间片段。当前主流方法基于DETR架构，使用K个可学习的解码器查询，每个映射到一个候选时刻及对应置信度。现有基准（HD-EPIC、YouCook2、ActivityNet-Captions等）均采用标注员**观看视频后**写出的描述性文本作为查询。

### 2. 痛点
现有数据集的文本查询是**caption-based**的——标注员在看完视频后撰写细粒度描述。这导致了"视觉偏差"：查询过于详细、与视觉内容高度对齐。例如标注者会写"a man in a yellow jersey intercepts a loose pass..."，但真实用户可能只搜索"when are goals being scored?"。这两种查询在语言细粒度和语义覆盖上存在根本差异。

### 3. 核心矛盾
- **训练时**：每个caption-based查询仅对应**单个GT时刻**，且语言高度具体
- **推理时**：真实搜索查询往往更抽象和欠规范化，可能对应视频中的**多个时刻**
- 这种不匹配导致模型在真实搜索场景下性能大幅下降（最高达77.4% Rm@0.3退化）

### 4. 要解决什么
（1）量化caption-based查询与search查询之间的性能差距；（2）定位退化的两个根因：**语言差距（language gap）**和**多时刻差距（multi-moment gap）**；（3）缓解多时刻差距导致的解码器查询坍塌。

### 5. 切入角度
纯从**模型架构**角度出发，不改变训练数据或训练范式，仅通过结构性修改使模型能在单时刻训练数据上泛化到多时刻搜索场景。

### 6. 核心idea
DETR模型中存在**活跃解码器查询坍塌（active decoder-query collapse）**——仅少数查询参与预测，其余保持沉默。这由两个结构性原因导致：（i）自注意力引起的**协调坍塌**，查询间"商量好"只让少数查询激活；（ii）**索引坍塌**，固定的少量查询索引垄断了激活。通过移除自注意力（-SA）和引入查询Dropout（+QD），可同时解决这两个问题。

## 方法详解

### 整体框架

本文工作包含两大部分：

1. **基准构建**：基于LLM的搜索查询生成管线，将现有caption-based数据集转化为搜索查询基准
2. **架构改进**：针对DETR-based VMR模型的两项修改（-SA + QD），缓解活跃解码器查询坍塌

### 关键设计

#### 设计1：搜索查询生成管线（Search-Query Pipeline）

**功能**：将fine-grained的caption转化为under-specified的搜索查询，并自动建立多时刻对应关系。

**核心思路**：两阶段管线——
- **Per-query欠规范化阶段**：使用Gemma-12B构建rewriter-validator双代理系统。Rewriter将详细caption改写为模糊版本（如"a man tying his running shoes before starting a marathon" → "a person getting ready to exercise"），Validator检测不一致并交由人工修正
- **Query-grouping阶段**：对所有欠规范化查询计算句子嵌入的两两相似度，高相似度查询合并为一组（对应多时刻），再由LLM聚合器生成该组的代表性搜索查询

**设计动机**：真实搜索查询无法通过简单的数据收集获得（难以将文本标注与视频观看解耦），因此复用现有密集标注数据集，通过可控的欠规范化模拟搜索查询的分布偏移。

#### 设计2：移除自注意力（Remove Self-Attention, -SA）

**功能**：在DETR解码器层中直接移除解码器查询之间的自注意力模块。

**核心思路**：标准解码器层为 $\hat{Q}^{l+1} = \text{FFN}(\text{CA}(\text{SA}(\hat{Q}^l), M))$，修改后变为 $Q^{l+1} = \text{FFN}(\text{CA}(Q^l, M))$。移除SA后用NMS做后处理来去除冗余预测。

**设计动机**：SA的作用是推动解码器查询彼此远离以减少冗余。但在单时刻训练中，这种协调机制使查询"商量好"只让少数查询处理GT，其余主动关闭——即**协调坍塌（coordination collapse）**。移除SA使每个查询独立运作，打破这种协调捷径。

#### 设计3：查询Dropout（Query Dropout, QD）

**功能**：在训练时随机将k%的可学习解码器查询置零。

**核心思路**：$\hat{Q} = Q \odot M, \quad M \sim \mathbb{B}(1-k)$，其中$\mathbb{B}$为Bernoulli分布，k=0.25效果最佳。

**设计动机**：即使移除SA，模型仍会出现**索引坍塌（index collapse）**——固定的少量查询索引（如索引1-4）反复获得高置信度，其余永久沉默。QD通过随机屏蔽部分查询，迫使模型将监督信号分散到更多查询上，防止对固定子集的过度依赖。

### 损失函数 / 训练策略

- 损失函数保持与基线（CG-DETR、LD-DETR）完全一致，使用标准**一对一匈牙利匹配**
- 关键发现：保持1-to-1匹配至关重要——它在查询间引入竞争，确保被-SA+QD额外激活的查询保持多样性而非生成冗余预测
- 查询Dropout仅在训练时使用，推理时全部查询激活
- 后处理增加NMS步骤以替代被移除的SA去冗余功能

## 实验关键数据

### 主实验

**表1：HD-EPIC-S{1,2,3}基准结果（CG-DETR & LD-DETR）**

| 模型 | 输入 | 方法 | Rm@0.1 | Rm@0.3 | Rm@0.5 | mAPm@0.1 | mAPm@0.3 | mAPm@0.5 |
|------|------|------|--------|--------|--------|----------|----------|----------|
| CG-DETR | S1 | base | 28.61 | 17.95 | 8.99 | 36.21 | 22.84 | 11.59 |
| CG-DETR | S1 | -SA+QD | 29.87 | 19.69 | 10.86 | 39.74 | 26.49 | 14.87 |
| CG-DETR | S2 | base | 24.71 | 15.52 | 7.89 | 32.15 | 20.10 | 10.29 |
| CG-DETR | S2 | -SA+QD | 26.17 | 17.00 | 9.40 | 35.38 | 23.39 | 13.04 |
| CG-DETR | S3 | base | 9.50 | 4.61 | 2.08 | 16.20 | 8.01 | 3.58 |
| CG-DETR | S3 | -SA+QD | 10.57 | 6.52 | 3.45 | 17.27 | 10.65 | 5.54 |
| LD-DETR | S2 | base | 25.23 | 16.38 | 8.46 | 32.42 | 21.11 | 10.93 |
| LD-DETR | S2 | -SA+QD | 26.36 | 16.98 | 8.87 | 36.37 | 23.75 | 12.54 |

**表2：YC2-S和ANC-S基准结果**

| 模型 | 数据集 | 方法 | Rm@0.3 | mAPm@0.1 | mAPm@0.3 | mAPm@0.5 |
|------|--------|------|--------|----------|----------|----------|
| CG-DETR | YC2-S | base | 19.87 | 38.83 | 26.96 | 15.21 |
| CG-DETR | YC2-S | -SA+QD | 20.32 | 41.00 | 29.40 | 17.21 |
| LD-DETR | YC2-S | base | 23.48 | 41.69 | 30.04 | 15.58 |
| LD-DETR | YC2-S | -SA+QD | 24.76 | 45.66 | 33.09 | 18.74 |
| CG-DETR | ANC-S | base | 40.89 | 72.12 | 54.92 | 36.42 |
| CG-DETR | ANC-S | -SA+QD | 43.12 | 74.00 | 56.42 | 38.20 |

### 消融实验

**组件消融（HD-EPIC-S2, CG-DETR）**

| -SA | +QD | Rm (avg) | mAPm (avg) | #active queries |
|-----|-----|----------|------------|-----------------|
| ✗ | ✗ | 16.04 | 20.84 | 3.64±1.18 |
| ✓ | ✗ | 15.31 | 21.02 | 3.72±1.16 |
| ✗ | ✓ | 16.50 | 21.43 | 3.77±1.28 |
| ✓ | ✓ | **17.52** | **23.93** | **6.43±2.16** |

**替代查询激活方法对比**

| 方法 | Rm | mAPm | #active | %match GT |
|------|-----|------|---------|-----------|
| base | 16.04 | 20.84 | 3.64 | 0.36 |
| +1-to-5 matching | 14.66 | 16.30 | 9.56 | 0.21 |
| +1-to-k matching | 10.78 | 11.01 | 20.00 | 0.07 |
| +group matching | 15.34 | 17.97 | 8.69 | 0.27 |
| -SA+QD (ours) | **17.52** | **23.93** | 6.43 | **0.42** |

### 关键发现

1. **两项修改缺一不可**：单独使用-SA或+QD仅带来边际提升（mAPm从20.84到~21），两者结合才能将活跃查询数从3.64翻倍至6.43，mAPm提升3.09
2. **单纯增加活跃查询无效**：1-to-k matching把活跃查询增至20但mAPm反而暴跌到11.01——激活的查询生成了冗余预测（%match GT从0.36降至0.07）
3. **1-to-1匹配的关键保障作用**：保持匈牙利1-to-1匹配确保新激活的查询彼此竞争而非冗余
4. **QD率敏感**：k=0.25最优，k=0.50导致性能崩溃（mAPm从23.93降到3.84）
5. **多时刻查询受益最大**：-SA+QD在多时刻实例上提升高达34.3% mAPm@0.3，单时刻也有温和提升
6. **方法恢复了约70%的oracle差距**（oracle指直接用搜索查询训练的模型）

## 亮点与洞察

- **问题定义极具洞察力**：指出VMR领域长期忽视的根本问题——训练用的caption与真实用户搜索查询的分布偏移，这是一个被整个社区忽视但对实际部署至关重要的议题
- **提出新的多时刻评估指标**：Rm和mAPm解决了传统R1/mAP在多时刻评估中的不公平问题
- **解码器查询坍塌的诊断**非常精准：通过协调坍塌和索引坍塌两个正交维度分析问题，方案简洁有效
- **不改数据只改结构**的思路具有很高的实用性——避免了昂贵的重新标注

## 局限与展望

1. **语言差距未解决**：本文仅解决了multi-moment gap，language gap被留作future work，作者建议用更强的视觉-语言模型来应对不同粒度的语义推理
2. **搜索查询由LLM生成而非真实用户**：虽然经过了验证，但与真正的用户搜索行为可能仍有差异
3. **QD率敏感性过高**：k从0.25到0.50导致性能从23.93崩溃到3.84，鲁棒性有待提升
4. **基准仅涵盖烹饪/运动等场景**：开放域、长视频场景的泛化性未探索
5. **NMS后处理依赖**：移除SA后需NMS去冗余，这引入了额外超参数和计算开销

## 相关工作与启发

- **与DETR查询坍塌文献的关联**：目标检测（[53,28,21]）、时序动作检测（[17]）和3D检测（[44,52]）中均报告了查询坍塌，但原因不同——它们由稀疏one-to-one匹配导致，而VMR中由单时刻先验导致
- **对搜索/检索领域的启发**：Liang et al.[24]研究模糊查询对ranked retrieval的影响，本文则处理单视频内的多时刻检索，两者互补
- **方法可推广**：-SA+QD的设计思路可应用于任何存在decoder-query collapse的DETR变体任务

## 评分

⭐⭐⭐⭐ 问题定义新颖、分析深入、方案简洁有效，是推动VMR走向真实应用场景的重要工作，但语言差距未解决且QD敏感性较高是明显短板。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/object_detection/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[AAAI 2026\] Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](../../AAAI2026/object_detection/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)

<!-- RELATED:END -->
