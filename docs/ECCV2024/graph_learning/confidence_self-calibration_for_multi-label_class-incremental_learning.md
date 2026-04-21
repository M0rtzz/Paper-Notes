---
title: >-
  [论文解读] Confidence Self-Calibration for Multi-Label Class-Incremental Learning
description: >-
  [ECCV2024][图学习][multi-label class-incremental learning] 针对多标签类增量学习(MLCIL)中部分标签导致的过度自信预测和假阳性错误问题，提出 Confidence Self-Calibration (CSC) 框架，通过类增量图卷积网络(CI-GCN)校准标签关系 + 最大熵正则化校准置信度，在 MS-COCO 和 VOC 上大幅超越 SOTA。
tags:
  - ECCV2024
  - 图学习
  - multi-label class-incremental learning
  - confidence calibration
  - graph convolutional network
  - max-entropy regularization
  - partial label
---

# Confidence Self-Calibration for Multi-Label Class-Incremental Learning

**会议**: ECCV2024  
**arXiv**: [2403.12559](https://arxiv.org/abs/2403.12559)  
**作者**: Kaile Du, Yifan Zhou, Fan Lyu, Yuyang Li, Chen Lu, Guangcan Liu (东南大学, 中科院自动化所)  
**领域**: graph_learning  
**关键词**: multi-label class-incremental learning, confidence calibration, graph convolutional network, max-entropy regularization, partial label

## 一句话总结

针对多标签类增量学习(MLCIL)中部分标签导致的过度自信预测和假阳性错误问题，提出 Confidence Self-Calibration (CSC) 框架，通过类增量图卷积网络(CI-GCN)校准标签关系 + 最大熵正则化校准置信度，在 MS-COCO 和 VOC 上大幅超越 SOTA。

## 背景与动机

多标签类增量学习(MLCIL)的核心挑战是**任务级部分标签问题**：在每个增量任务中，只有当前任务的新类被标注，过去和未来的类标签均缺失。这导致新旧标签空间天然隔离(disjoint)。

现有方法（如 AGCN、KRT）忽视了一个关键现象：**模型在部分标签设定下输出过度自信的预测分布**，产生大量假阳性(false-positive)错误。例如，测试图像中只有"person"，但模型对旧类"dog"也输出了很高的置信度。随着标签空间不断增大，这种过度自信问题会加剧灾难性遗忘。

作者动机很清晰：既然问题出在置信度校准上，就应该从**标签关系校准**和**置信度校准**两个层面同时解决。

## 核心问题

1. **标签关系断裂**：部分标签设定下，无法获取完整的标签共现统计，跨任务标签关系难以构建
2. **置信度过度自信**：模型在缺失标签情况下容易混淆新旧类特征，输出分布呈多峰过度自信状态，精确率远低于召回率，假阳性率居高不下
3. **灾难性遗忘加剧**：以上两点相互叠加，使旧类性能严重退化

## 方法详解

### 整体框架 CSC

CSC 包含两大组件：

**（一）Class-Incremental Graph Convolutional Network (CI-GCN)** — 标签关系校准

CI-GCN 是一个两层堆叠的 GCN 结构，不依赖先验统计信息：

1. **General GCN**：使用可学习的 general correlation matrix (CM) $A_g$，通过梯度更新自动学习跨任务标签关系。$A_g$ 分为旧任务部分 $A_g^{1:t-1}$（继承自前任务，保留旧标签关系）和新任务部分 $A_g^t$（建立新标签空间关系）。关键创新在于 CM 通过梯度下降联合使用真实标签和伪标签更新，避免了固定统计矩阵的误差累积问题。

2. **Specific GCN**：从 General GCN 的输出 $V_1$ 自适应生成每张图像独有的 specific CM $A_s$。具体做法是对 $V_1$ 做全局池化和卷积得到全局特征 $v$，拼接后通过卷积层计算 $A_s = \sigma(V_1' W)$。这提供了样本级别的细粒度标签关系。

图节点 $V_0$ 由 CNN 骨干提取的特征图 $F$ 和类激活图 $M$ 解耦得到：$V_0 = M^\top \otimes F$。两层 GCN 计算：

$$V_1 = \text{LReLU}(A_g V_0 W_g), \quad V_2 = \text{LReLU}(A_s V_1 W_s)$$

**（二）Max-Entropy Regularization** — 置信度校准

观察到模型输出分布过度自信（低熵），作者用 Shannon 熵量化旧类预测的不确定性：

$$H = -\mathbb{E}_{c \in \mathcal{C}^{1:t-1}} [\hat{y}_c^t \log(\hat{y}_c^t)]$$

训练时取负号实现最大熵正则化，惩罚过度自信的输出分布：

$$L = L_{\text{cls}} - \beta H$$

其中 $L_{\text{cls}}$ 结合了交叉熵（新类）和知识蒸馏（旧类），$\beta$ 控制正则化强度。最终预测融合分类器输出和图表示：$\hat{y}^t = \hat{y}_{\text{cls}}^t + \hat{y}_{\text{gcn}}^t$。

### 设计亮点

- General CM 可学习而非统计固定，避免伪标签累积误差
- Specific CM 为每张图像自适应生成，处理罕见标签组合更灵活
- 两种 CM 随类数增加自动扩展，无需手动调整
- 最大熵正则化仅作用于旧类，定向减少假阳性

## 实验关键数据

### MS-COCO 2014

| 设定 | 方法 | Buffer | Last mAP | CF1 | OF1 |
|------|------|--------|----------|-----|-----|
| B0-C10 | KRT (SOTA) | 0 | 65.9 | 55.6 | 56.5 |
| B0-C10 | **CSC** | 0 | **72.8** | **64.9** | **66.8** |
| B0-C10 | KRT-R | 5/class | 68.3 | 60.0 | 61.0 |
| B0-C10 | **CSC-R** | 5/class | **73.7** | **67.3** | **68.1** |
| B0-C10 | **CSC-R** | 20/class | **74.8** | **67.8** | **68.6** |

无 buffer 的 CSC (72.8%) 甚至超过了 KRT-R 使用 20/class buffer 的结果 (70.2%)。

### PASCAL VOC 2007

| 设定 | 方法 | Buffer | Last mAP | Avg. mAP |
|------|------|--------|----------|----------|
| B0-C4 | KRT-R | 2/class | 83.4 | 90.7 |
| B0-C4 | **CSC-R** | 2/class | **87.9** | **92.4** |
| B4-C2 | AGCN-R | 2/class | 59.3 | 74.3 |
| B4-C2 | **CSC-R** | 2/class | **86.6** | **90.4** |

CSC-R 在最困难的 B4-C2 场景中比 AGCN-R 高出 27.3%，展现出极强的鲁棒性。

### 消融实验

| 组件 | mAP (B0-C10) | CF1 | OF1 |
|------|-------------|-----|-----|
| Baseline (KD only) | 42.4 | 45.3 | 43.7 |
| + Max-Entropy | 47.6 | 50.3 | 49.5 |
| + CI-GCN | 69.3 | 59.0 | 59.5 |
| + 两者结合 (CSC) | **72.8** | **64.9** | **66.8** |

CI-GCN 贡献最大（mAP +26.9%），Max-Entropy 在此基础上再提升 3.5%。Max-Entropy 将假阳性率从 35% 降至 19%。

### CM 结构消融

G → S (Softmax) 组合最优(mAP 72.8%)，优于固定统计 CM 的 Z → Z (64.1%)，验证了可学习 CM 的优势。

## 亮点

- **问题洞察深刻**：首次明确指出 MLCIL 中过度自信输出分布和假阳性错误的关联，并从校准角度提出解决方案
- **CI-GCN 设计精巧**：General + Specific 双层结构从宏观到细粒度校准标签关系，且 CM 可学习扩展，避免统计误差累积
- **最大熵正则简洁有效**：仅需一个额外正则项就能显著降低假阳性率(35% → 19%)，且对不同方法具有正交提升效果
- **实验结果压倒性**：无 buffer 的 CSC 甚至超越有 20/class buffer 的 SOTA
- **鲁棒性强**：在不同场景(步长大小)间性能波动极小

## 局限与展望

- 仅在 MS-COCO (80类) 和 VOC (20类) 上验证，未测试更大规模数据集（如 Open Images）
- 骨干网络基于 CNN (TResNetM)，未探索 Vision Transformer 架构下 CI-GCN 的适配
- 最大熵正则仅作用于旧类，对新类的置信度校准未做探讨
- General CM 的随机初始化可能影响首个任务的性能，可探索更好的初始化策略
- 类增量顺序固定为字典序，对随机/困难优先排序的敏感性未分析

## 与相关工作的对比

| 方法 | 标签关系建模 | 置信度校准 | 核心思路 |
|------|------------|-----------|---------|
| AGCN | 固定统计 CM + 伪标签 | 无 | 用伪标签构建固定图 |
| KRT (前 SOTA) | 知识恢复与迁移 token | 无 | old/new knowledge token 框架 |
| **CSC (本文)** | 可学习双层 CM (CI-GCN) | Max-Entropy 正则 | 标签关系校准 + 置信度校准 |

与 KRT 相比，CSC 的优势在于：(1) CM 通过梯度学习而非统计固定，适应性更强；(2) 显式解决了 KRT 忽略的置信度过度自信问题。与 L2P (ViT-B/16, 86M 参数) 相比，CSC 用更少参数 (TResNetM, 29.4M) 取得更好结果。

## 启发与关联

- **校准视角的普适性**：过度自信问题普遍存在于增量学习中，Max-Entropy 思路可迁移到单标签 CIL、语义分割增量学习等任务
- **可学习图结构**：CI-GCN 中 CM 通过梯度学习的设计思路可推广到其他需要动态构建关系图的场景
- **与 Focal Loss 的互补**：Focal Loss 聚焦难样本，Max-Entropy 正则化聚焦校准，两者可能具有互补性
- **与 label smoothing 的对比**：两者都在调节输出分布的"锐度"，但 Max-Entropy 直接优化信息熵，理论上更直接

## 评分

- 新颖性: ⭐⭐⭐⭐ (校准视角切入 MLCIL 有新意，CI-GCN 设计合理)
- 实验充分度: ⭐⭐⭐⭐⭐ (多数据集、多场景、详细消融、可视化齐全)
- 写作质量: ⭐⭐⭐⭐ (问题阐述清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (MLCIL 方向重要推进，校准思路具有启发性)

<!-- RELATED:START -->

## 相关论文

- [GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)
- [Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [SENC: Handling Self-collision in Neural Cloth Simulation](senc_handling_self-collision_in_neural_cloth_simulation.md)
- [EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](../../AAAI2026/graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [Posterior Label Smoothing for Node Classification](../../AAAI2026/graph_learning/posterior_label_smoothing_for_node_classification.md)

<!-- RELATED:END -->
