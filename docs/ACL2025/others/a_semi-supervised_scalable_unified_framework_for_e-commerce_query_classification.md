---
title: >-
  [论文解读] SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification
description: >-
  [ACL 2025][查询分类] 提出电商查询分类统一框架 SSUF，通过三个可插拔模块——标签增强（BERT 语义编码标签）、知识增强（LLM 世界知识 + 后验点击 + 半监督标签生成）、结构增强（共现/语义/层级三图融合 GCN）——解决短查询信息不足和"马太效应"恶性循环问题…
tags:
  - "ACL 2025"
  - "查询分类"
  - "电商搜索"
  - "半监督学习"
  - "知识增强"
  - "图神经网络"
---

# SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification

**会议**: ACL 2025  
**arXiv**: [2506.21049](https://arxiv.org/abs/2506.21049)  
**代码**: 无  
**领域**: 其他  
**关键词**: 查询分类, 电商搜索, 半监督学习, 知识增强, 图神经网络

## 一句话总结

提出电商查询分类统一框架 SSUF，通过三个可插拔模块——标签增强（BERT 语义编码标签）、知识增强（LLM 世界知识 + 后验点击 + 半监督标签生成）、结构增强（共现/语义/层级三图融合 GCN）——解决短查询信息不足和"马太效应"恶性循环问题，在 JD.COM 意图分类和品类分类任务上 Macro F1 分别达到 49.46 和 41.22（均超 SMGCN 等 SOTA），已上线服务带来显著商业价值。

## 研究背景与动机

**领域现状**：电商平台（京东/淘宝/Amazon）的查询分类（意图/品类/品牌预测）是搜索系统核心。深度学习方法（XML-CNN、LSAN、DPHA）和近期的层级感知方法（HCL4QC、SMGCN、HQC）已有不少进展。

**现有痛点**：(1) **查询短且歧义**——电商查询平均仅 6-8 个字符（如"黑 16pro"），语义信息极度不足，直接编码无法与"手机"类目关联。(2) **马太效应恶性循环**——工业方法依赖用户点击行为构造训练样本，热门查询获得过多关注，偏差数据导致长尾查询泛化差。(3) **子任务孤立**——意图/品类/品牌预测各自独立建模，无统一框架支持共享优化。

**核心矛盾**：如何在信息极度稀缺（短查询）和标签极度倾斜（马太效应）的双重约束下提升分类性能？

**本文目标** 构建统一框架，通过先验知识注入和结构信息传播，打破对后验点击标签的过度依赖。

**切入角度**：三管齐下——用 LLM 生成世界知识补充查询语义、用标签语义编码生成半监督信号、用图结构传播长尾标签梯度。

**核心 idea**：知识增强解决"信息不足"，半监督标签解决"马太效应"，图结构增强解决"长尾标签"。

## 方法详解

### 整体框架

SSUF 的核心是一个共享 BERT 文本编码器，叠加三个高度可插拔的增强模块。在线推理仅需查询文本 + GCN 标签嵌入，知识增强分支的重计算离线完成。

### 关键设计

1. **标签增强模块（Label-Enhanced Module）**:

    - 功能：用 BERT 编码标签的语义表示，替代传统的标签 index 嵌入
    - 核心思路：标签输入 = 标签名 $n$ + 增强侧信息 $m$（产品词、高频搜索词、LLM 知识描述）。通过共享 BERT 编码：$\mathbf{C}_j = \text{BERT}_{\text{CLS}}([n_1,...,n_L, m_1,...,m_{L_m}])$
    - 设计动机：传统 index 嵌入无法捕获标签间语义关系，语义编码使标签可做相似度计算，促进知识迁移

2. **知识增强模块（Knowledge-Enhanced Module）**:

    - 功能：用外部知识补充短查询的语义信息，并生成半监督训练信号
    - 核心思路：
        - **知识来源**：(1) 后验知识——用户高频点击/购买的产品标签，(2) 世界知识——将查询和相关产品送入开源 LLM 生成简短描述（含相关查询/品类/产品）
        - **知识融合**：注意力机制融合查询表示和知识嵌入: $\alpha = \text{softmax}(\mathbf{Q}_i \mathbf{K}^T)$, $\mathbf{q}'_i = \mathbf{Q}_i + \sum_j \alpha_j \mathbf{K}_j$
        - **半监督标签生成**：计算融合后查询与标签的余弦相似度，超过阈值 $\tau$ 的作为半监督标签 $y^{semi}_{ij} = s_{ij} \cdot \mathbb{1}_{s_{ij} \geq \tau}$
        - **关键设计**：对半监督分支做 stop_gradient，防止循环依赖导致模型坍塌
    - 设计动机：如"黑 16pro"通过 LLM 知识可补充为"苹果手机 iPhone 16 Pro 黑色"，从而与"手机"类目匹配

3. **结构增强模块（Structure-Enhanced Module）**:

    - 功能：通过标签关系图传播梯度到长尾标签
    - 核心思路——三种图构建：
        - **共现图** $\mathbf{A}^{coo}$：标签共现条件概率 $a_{ij} = N(c_i, c_j) / N(c_i)$，阈值 $\alpha$ 过滤低频边
        - **语义相似图** $\mathbf{A}^{sim}$：标签 BERT 嵌入的余弦相似度，阈值 $\beta$ 过滤
        - **层级结构图** $\mathbf{A}^{hier}$：父子标签关系，边权 = $\max(1/|Child(k)|, m_i / \sum_{j \in Child(k)} m_j)$
    - **图融合与学习**：$\mathbf{A} = \frac{1}{2}(\mathbf{A}^{coo} + \mathbf{A}^{sim}) \rightarrow \mathbf{A}^{hier}$，归一化后用 GCN 学习标签表示
    - 设计动机：长尾标签训练样本少但可通过图连接与热门标签关联，获得梯度传播

### 损失函数 / 训练策略

- 最终预测：$\hat{\mathbf{y}}_i = \text{sigmoid}(\mathbf{q}_i \mathbf{H}_l^T + \mathbf{b})$，仅对叶标签做预测
- 标签融合：$\mathbf{y}_i = \min(\mathbf{y}_i^{click} + \mathbf{y}_i^{semi}, 1.0)$，后验+半监督标签联合
- 损失函数：Binary Cross-Entropy Loss
- 推理优化：知识增强分支（LLM 世界知识生成 + 注意力融合）离线预计算，在线仅需查询编码 + 标签嵌入交互

## 实验关键数据

### 主实验（JD.COM 数据集, Micro/Macro F1）

| 模型 | 意图任务 Micro F1 | 意图任务 Macro F1 | 品类任务 Micro F1 | 品类任务 Macro F1 |
|------|------------------|------------------|------------------|------------------|
| XML-CNN | 45.58 | 27.24 | 38.34 | 20.16 |
| LSAN | 47.98 | 31.71 | 37.15 | 22.84 |
| SMGCN | 59.72 | 48.54 | 53.92 | 40.15 |
| HQC | 49.58 | 36.77 | 44.85 | 33.98 |
| **SSUF** | **61.81** | **49.46** | **56.45** | **41.22** |

### 消融实验

| 配置 | 意图 Macro F1 | 品类 Macro F1 | 说明 |
|------|--------------|--------------|------|
| SSUF 完整 | 49.46 | 41.22 | 基线 |
| w/o SE（结构增强全去） | 43.30 (-6.16) | 38.52 (-2.70) | 图传播贡献显著 |
| w/o KE（知识增强） | 45.82 (-3.64) | 39.24 (-1.98) | 知识增强主要提升Macro（长尾） |
| w/o LE&KE | 42.36 (-7.10) | 36.47 (-4.75) | 标签+知识联合移除影响最大 |
| w/o SE-S（去语义图） | 45.21 (-4.25) | 39.72 | 语义图贡献最大 |
| w/o SE-C（去共现图） | 44.92 (-4.54) | 39.24 | 共现图同样重要 |
| w/o SE-H（去层级图） | 47.29 (-2.17) | 39.95 | 层级图贡献相对较小 |
| 纯 BERT | 36.84 | 33.80 | 三模块联合提升 +12.62 Macro F1 |

### 关键发现

- 三个模块各有独立贡献，联合效果远超单独使用——纯 BERT 到 SSUF，意图 Macro F1 从 36.84 提升到 49.46（+34%）
- 知识增强模块对 Macro F1（长尾标签）提升大于 Micro F1（热门标签），验证了打破马太效应的设计目标
- 已通过 JD.COM 在线 A/B 实验验证，带来显著商业价值
- 三种图的贡献中，语义图和共现图各有 ~4-5 点 Macro F1 贡献，层级图约 2 点

## 亮点与洞察

- 统一框架的模块化设计——三个模块高度可插拔，可根据子任务的数据特点灵活组合。这种工程化的框架设计在工业界有很高的实用价值
- LLM 知识离线注入小模型是实用的知识蒸馏范式——不需要在线调用 LLM，将 LLM 世界知识预计算为查询特征，成本可控
- stop_gradient 防循环依赖的半监督设计巧妙——查询和标签共享编码器，直接传梯度会导致半监督信号和编码器互相强化坍塌

## 局限与展望

- LLM 生成的世界知识质量不可控，错误信息可能反向污染分类
- 半监督阈值 $\tau$ 和图过滤阈值 $\alpha$/$\beta$ 的敏感性分析不够充分
- 仅在 JD.COM 中文电商数据上验证，其他平台和多语言场景未测试
- 三图融合策略较简单（均值叠加+层级赋值），更复杂的注意力融合可能更好
- 品类任务标签空间 6,634 个类别，GCN 的可扩展性在更大标签空间下需要验证

## 相关工作与启发

- **vs HCL4QC/SMGCN**: 利用层级结构但缺乏知识增强和半监督信号；SSUF 三合一
- **vs 纯 LLM 分类**: LLM 直接做分类成本和延迟不可接受；SSUF 将 LLM 知识蒸馏为离线特征
- **vs LEAM/LSAN**: 标签感知但无图结构和外部知识；SSUF 的标签表示更丰富

## 评分

- 新颖性: ⭐⭐⭐ 三个模块各自不算新（LLM知识增强/半监督/GCN），但组合设计和工程化统一框架有实用贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模真实数据（6700万+训练样本）、完整消融、线上A/B验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，公式推导完整，但动机部分偏工业化
- 价值: ⭐⭐⭐⭐ 已上线工业系统，对电商搜索有直接价值，模块化设计有迁移参考性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](learning_to_align_multi-faceted_evaluation_a_unified_and_robust_framework.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](../../NeurIPS2025/others/prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/others/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[ACL 2025\] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)

</div>

<!-- RELATED:END -->
