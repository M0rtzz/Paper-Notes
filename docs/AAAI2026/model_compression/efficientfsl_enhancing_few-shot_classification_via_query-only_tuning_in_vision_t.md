---
title: >-
  [论文解读] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers
description: >-
  [AAAI 2026][模型压缩][少样本学习] 提出 EfficientFSL，一种针对 ViT 少样本分类的 query-only 参数高效微调框架，通过 Forward Block（解耦的主动/冻结子块）、Combine Block（自适应多层特征融合）和 SQ Attention Block（支持-查询分布对齐）三个模块，仅用1.25M~2.48M可训练参数即可在4个域内+6个跨域基准上达到 SOTA。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "少样本学习"
  - "参数高效微调"
  - "Transformer"
  - "查询式调优"
  - "原型网络"
---

# EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers

**会议**: AAAI 2026  
**arXiv**: [2601.08499](https://arxiv.org/abs/2601.08499)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 少样本学习, 参数高效微调, Vision Transformer, 查询式调优, 原型网络

## 一句话总结
提出 EfficientFSL，一种针对 ViT 少样本分类的 query-only 参数高效微调框架，通过 Forward Block（解耦的主动/冻结子块）、Combine Block（自适应多层特征融合）和 SQ Attention Block（支持-查询分布对齐）三个模块，仅用1.25M~2.48M可训练参数即可在4个域内+6个跨域基准上达到 SOTA。

## 研究背景与动机

### 领域现状
少样本学习（FSL）需要模型从极少量标注样本中学习新类别。近年来，基于 ViT 的方法用预训练的大规模视觉模型替代 CNN，获得了显著的性能提升。常见策略是先在大规模数据集上预训练，再适配到少样本任务。

### 现有痛点

**全参数微调代价高**：ViT-B 有85.8M参数，全微调需要为每个任务存储完整参数副本，GPU 显存和训练时间成本高

**PETL方法的耦合问题**：现有参数高效方法（Adapter, LoRA, Prompt Tuning 等）通过在冻结骨干中插入小模块来适配——但这些模块**修改了特征流**并与骨干权重**耦合**，在FSL数据极少的情况下容易过拟合

**支持集-查询集分布偏移**：少样本中支持集和查询集的图像在背景、光照、拍摄角度等方面存在差异，导致从支持集计算的原型偏离查询集的分布中心

### 核心矛盾
大模型+少数据 = 高过拟合风险。如何在利用预训练知识的同时，以极少参数实现任务适配并避免过拟合？

### 切入角度

**Query-only 范式**：完全冻结骨干，不修改特征流，而是引入轻量级查询模块从骨干中间表示中**选择性提取**任务相关信息。这与传统 PETL 方法"修改骨干"的思路本质不同——EfficientFSL 是"旁路查询"而非"内部修改"。

## 方法详解

### 整体框架
EfficientFSL 的输入是预训练 ViT 各层的中间表示，通过以下模块顺序处理：
1. **Forward Block**（×n层）：每层包含 Active Block（学任务知识）+ Frozen Block（查询预训练知识）
2. **Combine Block**：自适应融合多层特征
3. **SQ Attention Block**：调整原型位置以对齐查询集分布
4. **PN分类器**：基于余弦相似度的原型网络分类

### 关键设计

1. **Forward Block — Active Block（主动子块）**:

    - **功能**：轻量级网络，从零开始训练以学习任务特定知识，生成自适应查询
    - **核心结构**：
        - 输入 $H_{i-1}$ 加上可训练 prompt $P_i$，经过瓶颈投影层生成 $Z_i$
        - $Z_i$ 经过自注意力层和 MLP，所有投影层使用瓶颈结构（隐藏维度仅8或48）
    - **关键公式**：
    $Z_i = \text{Proj}(H_{i-1} + P_i)$
    $Z'_i = \xi \cdot \text{Att}(Q^A_i, K^A_i, V^A_i) + Z_i$
    $F_i = \zeta \cdot \text{MLP}(\text{LN}(Z'_i)) + Z'_i$
    - **设计动机**：
        - Prompt $P_i$ 提供每层的任务引导信号
        - 缩放因子 $\xi, \zeta$ 控制任务知识的注入强度
        - 瓶颈结构（注意力头维度仅8）极大减少参数量

2. **Forward Block — Frozen Block（冻结子块）**:

    - **功能**：复用并冻结预训练 ViT 的参数，以 query-only 方式从骨干提取知识
    - **核心思路**：
        - **Query**来自 Active Block 的输出 $F_i$（任务特定）
        - **Key和Value**来自预训练 ViT 第 $i$ 层的输出 $X_i$（通用知识）
        - 这实现了**解耦**：任务知识→查询，通用知识→键值
    - **关键公式**：
    $F^{att}_i = \text{Att}'(Q^F_i, K^F_i, V^F_i) + F_i$
    $F^{mlp}_i = \text{MLP}'(\text{LN}(F^{att}_i))$
    $H_i = F^{mlp}_i + F^{att}_i$
    - **设计动机**：冻结骨干权重完全保留了预训练的通用知识，避免在少样本数据上过拟合；query-only 模式允许任务特定查询有选择地"检索"相关信息

3. **Combine Block（多层特征融合）**:

    - **功能**：自适应地将所有层的三种特征（$F^{att}_i, F^{mlp}_i, H_i$）融合为统一表示
    - **核心思路**：
        - 所有特征通过共享的瓶颈 MLP 投影到统一空间（对齐）
        - 以最后一层 $H_n$ 为条件，通过 Weight MLP 生成自适应权重
    $F^{agg} = \sum_{i=1}^{n}(w^{att}_i \cdot \hat{F}^{att}_i + w^{mlp}_i \cdot \hat{F}^{mlp}_i + w^H_i \cdot \hat{H}_i)$
    - **设计动机**：不同任务可能需要不同层级的特征——浅层的纹理信息 vs 深层的语义信息。条件式自适应权重比简单平均或固定权重效果更好

4. **SQ Attention Block（支持-查询对齐）**:

    - **功能**：调整原型位置，使其更接近对应类别查询样本的分布中心
    - **核心公式**：
    $s^{att} = \alpha(s \cdot \text{Proj}(q)^T) \cdot q + (1-\alpha) \cdot s$
    - **核心思路**：
        - 对查询 $q$ 做可学习投影以实现类感知对齐
        - 通过注意力权重 $s \cdot \text{Proj}(q)^T$ 让原型向与之关联的查询方向移动
        - 混合系数 $\alpha$ 控制调整幅度
    - **设计动机**：FSL 中支持集很小（1-5shot），计算的原型可能因随机性偏离真实类中心，SQ Attention 利用查询集信息动态修正

### 训练策略
- 优化器：AdamW，学习率 0.0001，余弦调度
- 训练 5 epochs，batch size 64
- 数据增强：resize 256→center crop 224
- 所有实验在单张 NVIDIA V100 GPU 上完成

## 实验关键数据

### 主实验

**域内少样本分类**（ViT-B, ImageNet-21K预训练）：

| 数据集 | 设定 | EfficientFSL | FewVS (MM'24) | MetaFormer (ICML'24) | SemFew (CVPR'24) |
|--------|------|-------------|---------------|---------------------|-----------------|
| miniImageNet | 5w-1s | **98.34** | 86.80 | 84.78 | 78.94 |
| miniImageNet | 5w-5s | **99.12** | 90.32 | 91.39 | 86.49 |
| tieredImageNet | 5w-1s | **93.27** | 87.87 | 88.38 | 82.37 |
| FC100 | 5w-1s | **80.13** | 61.01 | 58.04 | 54.27 |
| FC100 | 5w-5s | **88.81** | 70.37 | 70.80 | 65.02 |

**参数量对比**：EfficientFSL 仅 2.48M 参数，而 SemFew 用 88.0M，FewVS 用 21.7M。

**与 PETL 方法对比**（ViT-B, FC100）：

| 方法 | 精度(%) | 训练时间(s/epoch) | 峰值显存(GB) | 推理速度(img/s) |
|------|--------|------------------|-------------|----------------|
| Adapter | 67.48 | 74.19 | 1.09 | 134.21 |
| AdaptFormer | 68.83 | 76.87 | 1.19 | 129.98 |
| LoRA | 74.19 | 76.35 | 1.10 | 131.84 |
| **EfficientFSL** | **80.13** | **59.62** | **1.08** | **134.29** |

### 消融实验

**训练模块移除消融**（ViT-B, FC100, 21K预训练）：

| 配置 | 参数(M) | 1-shot | 5-shot | 说明 |
|------|---------|--------|--------|------|
| 去除 Proj 层 | 1.58 | 51.15 | 68.95 | 性能骤降，Proj几乎不可或缺 |
| 去除 Att & MLP | 1.05 | 72.15 | 87.81 | 参数减半，轻微下降 |
| 去除 Combine Block | 2.37 | 75.55 | 88.60 | 多尺度融合有帮助 |
| **完整模型** | **2.48** | **80.13** | **88.81** | 最优 |

**SQ Attention Block 消融**：

| 配置 | mini-1s | FC100-1s | FC100-5s |
|------|---------|----------|----------|
| 无 SQ Attention | 95.95 | 75.42 | 87.82 |
| 无 Proj(q) | 98.34 | 80.13 | 88.81 |
| **完整** | **98.49** | **80.20** | **89.53** |

**特征融合策略对比**（FC100）：

| 融合方式 | 1-shot | 5-shot | 说明 |
|----------|--------|--------|------|
| 简单平均 | 69.58 | 85.38 | 所有层等权 |
| 固定权重 | 69.97 | 85.49 | 可学习但输入无关 |
| **条件权重** | **80.13** | **88.81** | 基于最后层动态生成，显著更优 |

### 关键发现
1. **参数效率惊人**：仅用全微调 ~3% 的参数（2.48M vs 85.8M），在所有基准上大幅超越全微调
2. **miniImageNet 上接近完美**：5w-5s 达到99.12%，几乎无提升空间
3. **跨域泛化强**：在6个跨域数据集（CUB, Cars, Places, Plantae, EuroSAT, CropDiseases）上均超越现有 SOTA
4. **效率全面领先**：训练速度最快、显存最低、推理速度最快（相比 Adapter/LoRA/AdaptFormer）
5. **Proj 层是关键**：去除投影层导致1-shot精度从80.13骤降至51.15，说明高质量特征映射的重要性
6. **条件式权重远优于简单融合**：自适应权重比均匀平均提升 ~10% 绝对精度

## 亮点与洞察
1. **Query-only 范式的核心价值**：完全不修改骨干的特征流，避免了 PETL 方法的耦合过拟合问题。这是一种更温和、更安全的知识提取策略
2. **解耦设计**：Active Block（学什么好查询）和 Frozen Block（保留预训练知识）的分工明确
3. **SQ Attention 的启发性**：利用查询集信息反过来修正原型，这在少样本场景中是合理的——原型从 K 个样本估计可能有偏
4. **t-SNE 可视化有说服力**：从混乱→粗通聚类→清晰分离，每加一个模块都可视化出明确改善
5. **极致的参数效率**：瓶颈结构+权重共享将参数压到极低，同时保持高性能

## 局限与展望
1. miniImageNet 上的99%+性能可能暗示该基准已接近饱和，需要更难的少样本基准来区分方法
2. 仅在分类任务上验证，未扩展到少样本检测、分割等更复杂任务
3. SQ Attention Block 的额外参数量很小（0.07M），其性能提升也相对有限（~0.7%），成本效益比不够高
4. 需要预训练 ViT 各层的中间表示作为输入，增加了推理时的计算和存储开销
5. 条件式权重高度依赖最后一层特征，若最后一层质量差则可能影响全局

## 相关工作与启发
- **FewTURE**（NeurIPS'22）：早期 ViT 少样本方法，参数21.7M但精度仅68%
- **MetaFormer-A**（ICML'24）：元学习+ViT，参数24.5M，精度84.78%
- **LoRA**（Hu et al. 2022）：低秩适配方法，在FSL上精度74.19%，被 EfficientFSL 大幅超越
- **启发**：Query-only 的思想可推广到其他模态的少样本学习（如 NLP few-shot with LLM），以及多模态模型的高效适配

## 评分
- 新颖性: ⭐⭐⭐⭐ （query-only范式有新意，但具体组件比较标准）
- 实验充分度: ⭐⭐⭐⭐⭐ （4+6个数据集，3种骨干，完整消融和可视化）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表充实）
- 价值: ⭐⭐⭐⭐ （参数效率极高，但FSL应用场景相对有限）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/model_compression/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[AAAI 2026\] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching](quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)

</div>

<!-- RELATED:END -->
