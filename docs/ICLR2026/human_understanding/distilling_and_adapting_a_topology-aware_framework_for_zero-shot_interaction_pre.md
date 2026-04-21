---
title: >-
  [论文解读] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks
description: >-
  [ICLR 2026][人体理解][多重生物网络] 提出CAZI-MBN框架，通过融合领域特定LLM序列嵌入、拓扑感知图分词器、上下文感知跨层注意力和教师-学生蒸馏，实现多重生物网络中未见实体的零样本交互预测，在5个基准数据集上AUROC较最优baseline提升3.1-20.4%。
tags:
  - ICLR 2026
  - 人体理解
  - 多重生物网络
  - 零样本预测
  - 知识蒸馏
  - Transformer
  - 多模态表示学习
---

# Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks

**会议**: ICLR 2026  
**arXiv**: [2603.06618](https://arxiv.org/abs/2603.06618)  
**代码**: [有](https://github.com/alanadeng/CAZI-MBN)  
**领域**: 人体理解  
**关键词**: 多重生物网络, 零样本预测, 知识蒸馏, 图Transformer, 多模态表示学习

## 一句话总结

提出CAZI-MBN框架，通过融合领域特定LLM序列嵌入、拓扑感知图分词器、上下文感知跨层注意力和教师-学生蒸馏，实现多重生物网络中未见实体的零样本交互预测，在5个基准数据集上AUROC较最优baseline提升3.1-20.4%。

## 研究背景与动机

多重生物网络（Multiplex Biological Networks, MBNs）通过多层结构表示同一组实体间的不同交互类型（如药物-基因的多种作用机制、蛋白质在不同细胞类型中的交互），是理解复杂生物系统的关键工具。现有方法存在三个核心缺陷：

**多重性处理不足**：大多采用单层网络分析，丢失了交互类型间的关系异质性和语义区分

**多模态整合能力弱**：难以将生物/化学序列特征与网络拓扑有效结合

**零样本预测困难**：对训练中未见的新实体（无先验邻域信息），现有GNN方法难以泛化

本文提出CAZI-MBN（Context-Aware and Zero-shot Interaction prediction in MBNs），旨在系统性地解决上述三个挑战。

## 方法详解

### 整体框架

CAZI-MBN由四个核心模块组成：

1. **特征表示**：序列嵌入（领域LLM）+ 拓扑嵌入（统一图分词器UGT）
2. **上下文感知增强（CAE）**：跨层注意力 + 对比学习
3. **混合专家（MoE）**：多标签分类的自适应预测
4. **知识蒸馏**：教师→学生的零样本泛化能力迁移

训练由混合损失驱动：$\mathcal{L} = \mathcal{L}_{disc} + \mathcal{L}_{reg} + \mathcal{L}_{cls} + \beta\|\Theta\|^2$

### 关键设计

**1. 特征表示层**

**序列嵌入**：针对不同生物实体类型使用对应的预训练LLM：
- ChemBERTa → 药物/代谢物的SMILES表示
- DNABERT-2 → 基因序列
- ESM-2 → 蛋白质序列

**统一图分词器（UGT）**：从超邻接矩阵 $\hat{A}$ 生成拓扑/多重性/高阶连接感知的节点嵌入：
- 构建平滑高阶矩阵：$\tilde{A} = \bar{A} + \bar{A}^2 + \cdots + \bar{A}^O$，其中 $\bar{A} = D^{-1/2}\hat{A}D^{-1/2}$
- 对 $\tilde{A}$ 做SVD分解得到 $U, \Sigma, V$
- 节点嵌入：$e_v = \tilde{A}_{v,:} \cdot \text{LN}(U\sqrt{\Sigma} \| V\sqrt{\Sigma})$

超邻接矩阵 $\hat{A}$ 通过直和编码层内连接，加上层间连接矩阵 $C$，完整保留多重网络结构。

**2. 上下文感知增强（CAE）模块**

CAE模块通过两级注意力机制精化多重嵌入：

**节点级跨层注意力**：允许实体在不同层的表示之间自适应地加权传递信息：

$$a_n^{(p \leftarrow q)} = \text{softmax}\left(\frac{\sigma(\theta^{(p)} \cdot (H_n^{(p)} \otimes H_n^{(q)}))}{\sum_{l \neq p} \sigma(\theta^{(p)} \cdot (H_n^{(p)} \otimes H_n^{(l)}))}\right)$$

**层级注意力聚合**：通过注意力模块将各层特征聚合为统一表示 $H$。

**对比学习框架**：
- 对每层图进行负采样生成扰动图 $\tilde{G}_i$
- 判别器区分真实边嵌入与负边嵌入
- 共识正则化器学习共识嵌入 $Z$：最大化与 $H$ 的一致性，最小化与 $\tilde{H}$ 的对齐
- 正则化损失：$\mathcal{L}_{reg} = 1 + \text{CosineSim}(H, Z) - \text{CosineSim}(\tilde{H}, Z)$

**3. 混合专家（MoE）**

MBN中的交互预测是多标签分类任务。MoE框架中：
- K个专家 $f_k$ 各自捕捉不同交互模式
- 门控网络根据输入分配权重：$\boldsymbol{a} = \text{softmax}(W_g \mathbf{h} + b_g)$
- 最终预测：$\hat{\mathbf{Y}} = \sum_{k=1}^{K} a_k f_k(\mathbf{h})$

**4. 知识蒸馏实现零样本**

- **教师模型**：使用序列+拓扑嵌入，依赖邻域上下文
- **学生模型**：仅使用序列数据，拓扑无关
- 蒸馏损失：MSE对齐师生潜在表示 + 分类损失
- 推理时学生模型可为完全未见的实体（无图结构信息）做预测

### 损失函数 / 训练策略

- **教师损失**：$\mathcal{L} = \mathcal{L}_{disc} + \mathcal{L}_{reg} + \mathcal{L}_{cls} + \beta\|\Theta\|^2$
    - $\mathcal{L}_{disc}$：判别器二元交叉熵损失
    - $\mathcal{L}_{reg}$：共识正则化损失（余弦相似度）
    - $\mathcal{L}_{cls}$：多标签软间隔损失
- **学生损失**：$\mathcal{L}_{distill}(\text{MSE}) + \mathcal{L}_{cls}$

## 实验关键数据

### 主实验（5个多重生物网络，13个baseline）

**DGIdb（药物-基因，1846节点，5种交互类型）**：

| 设置 | 模型 | AUROC | AUPRC | HS | SA |
|------|------|-------|-------|-----|-----|
| 转导 | Graph Transformer | 0.505 | 0.514 | 0.493 | 0.508 |
| 转导 | HDMI | 0.551 | 0.557 | 0.540 | 0.511 |
| 转导 | **CAZI-MBN** | **0.715** | **0.729** | **0.687** | **0.684** |
| 零样本 | DMGI | 0.524 | 0.528 | 0.529 | 0.502 |
| 零样本 | **CAZI-MBN** | **0.671** | **0.709** | **0.688** | **0.663** |

**ChEMBL（化合物-细菌，9368节点，3种交互类型）**：

| 设置 | 模型 | AUROC | AUPRC | HS | SA |
|------|------|-------|-------|-----|-----|
| 转导 | HDMI | 0.663 | 0.762 | 0.789 | 0.730 |
| 转导 | **CAZI-MBN** | **0.812** | **0.863** | **0.889** | **0.757** |
| 零样本 | DMGI | 0.652 | 0.745 | 0.756 | 0.711 |
| 零样本 | **CAZI-MBN** | **0.791** | **0.839** | **0.857** | **0.723** |

**PINNACLE（蛋白质-蛋白质，7044节点，12种交互类型）**：

| 设置 | 模型 | AUROC | AUPRC |
|------|------|-------|-------|
| 转导 | xCAPT5 | 0.781 | 0.804 |
| 转导 | **CAZI-MBN** | **0.831** | **0.845** |
| 零样本 | xCAPT5 | 0.785 | 0.791 |
| 零样本 | **CAZI-MBN** | **0.812** | **0.820** |

### 消融实验

各模块移除后的平均性能下降（跨5个数据集）：
- **LLMs**：AUROC下降 15-20%（贡献最大）
- **CAE**：AUROC下降 7-10%
- **UGT**：AUROC下降 5-8%
- **MoE**：AUROC下降 5-8%

### 关键发现

1. CAZI-MBN在所有5个数据集的AUROC和AUPRC上一致优于13个baseline，提升幅度**3.1-20.4%**
2. 零样本设置下性能衰退小，表明知识蒸馏有效迁移了拓扑知识到序列空间
3. 领域特定LLM嵌入是最关键的组件（15-20%贡献），说明序列级语义信息对生物交互预测至关重要
4. 在稀少交互类型上表现稳定，MoE有效处理了标签不平衡问题
5. IBD案例研究中恢复了82.7%（DGIdb）和85.7%（TRRUST）的已知交互

## 亮点与洞察

1. **首个系统性的零样本MBN预测框架**：通过蒸馏将拓扑知识压缩到序列空间，使得完全未见实体的预测成为可能
2. **UGT的设计巧妙**：通过超邻接矩阵的SVD和高阶平滑，一步编码了拓扑、多重性和高阶连接性
3. **模块化设计合理**：序列→拓扑→跨层→预测的流水线使各组件贡献可分离、可消融
4. **自建5个高质量基准数据集**：填补了MBN评估标准化的空白

## 局限与展望

1. 未整合**3D结构数据**（蛋白质/化合物的三维结构），限制了精细化建模
2. 零样本仅实现实体级别泛化，对全新交互类型的泛化尚未验证
3. CAE模块中的多层注意力计算在大规模网络上的扩展性有待验证
4. 领域LLM的选择需要先验知识，框架的通用性依赖于合适的预训练模型
5. 未探索跨物种网络的迁移学习

## 相关工作与启发

- **单层GNN（GCN/GraphSAGE）**：处理多重数据能力不足
- **多重网络模型（DMGI/HDMI）**：保留层内/层间依赖但多模态整合弱
- **知识图谱方法**：扁平化层结构导致交互类型特异性丢失
- **知识蒸馏在NLP/CV中广泛应用**，但在MBN和零样本泛化中几乎未探索
- 启发点：拓扑→序列的蒸馏范式可推广到其他需要零样本泛化的图学习任务

## 评分

- **新颖性**: ★★★★☆ — 多重网络的零样本预测是新问题，框架设计系统完整
- **技术深度**: ★★★★☆ — UGT+CAE+MoE+蒸馏的组合有技术含量
- **实验说服力**: ★★★★★ — 5数据集×13baseline×2设置，覆盖全面
- **实用价值**: ★★★★☆ — 在药物发现和精准医疗中有直接应用前景
- **表达清晰度**: ★★★☆☆ — 模块众多，部分细节过于分散在附录中

<!-- RELATED:START -->

## 相关论文

- [Conformal Prediction for Zero-Shot Models](../../CVPR2025/human_understanding/conformal_prediction_for_zero-shot_models.md)
- [Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/human_understanding/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [Maximizing Asynchronicity in Event-based Neural Networks](maximizing_asynchronicity_in_event-based_neural_networks.md)
- [Rapid Training of Hamiltonian Graph Networks using Random Features](rapid_training_of_hamiltonian_graph_networks_using_random_features.md)
- [DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

<!-- RELATED:END -->
