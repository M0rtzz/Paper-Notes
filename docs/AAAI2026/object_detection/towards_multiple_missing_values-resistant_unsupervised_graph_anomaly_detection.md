---
title: >-
  [论文解读] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection
description: >-
  [AAAI 2026][目标检测][图异常检测] 提出 M2V-UGAD 框架，首次解决节点属性和图拓扑同时缺失下的无监督图异常检测问题，通过双通路独立填补、超球潜空间融合和伪异常生成三个核心机制，克服跨视图干扰和填补偏差，在7个基准数据集上一致超越现有方法。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "图异常检测"
  - "缺失值"
  - "无监督学习"
  - "伪异常生成"
  - "Sinkhorn散度"
---

# Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection

**会议**: AAAI 2026  
**arXiv**: [2511.09917](https://arxiv.org/abs/2511.09917)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 图异常检测, 缺失值, 无监督学习, 伪异常生成, Sinkhorn散度

## 一句话总结

提出 M2V-UGAD 框架，首次解决节点属性和图拓扑同时缺失下的无监督图异常检测问题，通过双通路独立填补、超球潜空间融合和伪异常生成三个核心机制，克服跨视图干扰和填补偏差，在7个基准数据集上一致超越现有方法。

## 研究背景与动机

图异常检测（GAD）旨在从图结构数据中识别偏离正常模式的节点/子结构，应用于社交网络刷评检测、金融反欺诈等关键场景。由于真实异常极其稀少且标注成本高，**无监督 GAD** 成为研究重点。

现有无监督 GAD 方法主要分两类：

**基于重建**：训练图自编码器重建节点属性/结构，用重建误差识别异常（如 ADA-GAD、DiffGAD）

**基于对比学习**：通过节点表示与其局部邻域的一致性/不一致性区分异常（如 CoLA、PREM）

**然而，所有现有方法都隐含假设输入图是完整的**——即节点属性矩阵完整、邻接矩阵捕获了全部拓扑。现实中，节点属性可能因隐私保护或采集错误而缺失，边关系可能因新节点加入或交互未记录而丢失。

直接使用数据填补（imputation）再做异常检测面临两个根本挑战：

### 挑战一：跨视图干扰（Cross-view Interference）

当节点属性和边同时缺失时，基于不完整拓扑做属性填补会引入错误的邻域上下文，基于不完整属性做边预测会传播重建误差——两个视图的错误相互污染，形成恶性循环。

### 挑战二：填补偏差（Imputation Bias）

由于异常节点极其稀少，填补模型几乎完全在正常节点上训练。在推理时，它会将缺失的异常属性/边填充为典型的正常值，**将异常节点"修复"得看起来像正常节点**，从而掩盖异常信号。填补越精准，偏差反而越严重。

## 方法详解

### 整体框架

M2V-UGAD 包含三个关键模块，通过预训练 + 微调两阶段训练：

1. **双通路不完整图填补**：独立重建属性和拓扑，避免跨视图污染
2. **联合潜空间建模**：在共享超球面上融合两个视图，用 Sinkhorn 散度约束正常节点分布紧凑，用重建解码器保持语义
3. **图伪异常生成**：在正常区域外壳采样，解码生成逼真的伪异常节点和子图，作为困难负样本微调边界

### 关键设计

#### 1. **双通路不完整图填补（Dual-pathway Imputation）**

**属性重建通路**：使用自监督 MLP 对缺失属性进行填补。MLP 独立于图拓扑，对输入稀疏性鲁棒，避免在不完整拓扑上做消息传递引入误差。

$$\mathcal{L}_{feat} = \text{MSE}(\mathbf{M}_X \odot \hat{\mathbf{X}}, \mathbf{M}_X \odot \mathbf{X})$$

仅在观测到的属性位置上计算MSE损失。

**结构重建通路**：采用确定性 Personalized PageRank（PPR）扩散来恢复缺失拓扑。PPR 通过多跳传播在全局尺度上填补边信息，缓解因结构稀疏导致的节点孤立问题。

$$\mathbf{P}^{(t+1)} = \beta \tilde{\mathbf{A}} \mathbf{P}^{(t)} + (1-\beta)\mathbf{I}$$

最终增强邻接矩阵：$\hat{\mathbf{A}} = \mathbf{A}_{obs} + \mathbf{A}_{ppr}$

**设计动机**：属性和结构通路完全独立运行，从根本上阻断了跨视图的错误传播。属性通路不依赖拓扑，结构通路不依赖属性，实现了真正的解耦。

#### 2. **联合潜空间建模（Joint Latent-Space Modeling）**

虽然双通路解耦了填补过程，但属性和结构之间的内在共依赖关系（邻居属性可以推断缺失特征，节点特征相似性可以推断缺失边）需要被利用。

**超球面融合**：使用 L 层 GCN 将填补后的 $(\hat{\mathbf{X}}, \hat{\mathbf{A}})$ 投影到共享潜空间 $\mathbf{Z}$。通过 Sinkhorn 散度将 $\mathbf{Z}$ 的经验分布对齐到截断高斯先验 $\mathcal{N}_r(\mathbf{0}, \mathbf{I})$，使正常嵌入紧凑在半径 $r$ 的球内，异常自然分布在外部。

$$\mathcal{L}_{dist} = \text{Sinkhorn}(\mathbf{Z}, \mathcal{N}_r(\mathbf{0}, \mathbf{I}))$$

**为什么用 Sinkhorn 而非 SVDD？** SVDD 是点到中心的距离目标，可能导致所有嵌入坍缩到一点。Sinkhorn 做全局分布匹配：不仅拉近正常节点，还鼓励正常模式在球内保持有意义的分散，防止平凡坍缩。

**语义保持重建**：附加 MLP 解码器 $r_\omega$ 从嵌入重建节点属性：

$$\mathcal{L}_{recon} = \text{MSE}(\mathbf{M}_X \odot \tilde{\mathbf{X}}, \mathbf{M}_X \odot \mathbf{X})$$

联合优化 Sinkhorn 损失和重建损失，在紧凑性和语义保真之间取得平衡。

#### 3. **图伪异常生成（Graph Pseudo-Anomaly Generation）**

这是对抗填补偏差的核心机制。

**潜空间采样**：在正常球 $r$ 之外的环形壳 $\mathcal{S}_{shell} = \{r_a < \|z\|_2 < r_b\}$ 中均匀采样 $M$ 个潜向量（$r_a=1.2r, r_b=2r$），确保采样点：
- 足够接近正常流形——解码后特征逼真
- 位于决策边界附近——提供最有信息量的困难负样本

**合成图构建**：
1. 用已训练的解码器将潜向量解码为节点属性 $\tilde{x}^{(a)}$
2. 基于解码特征的余弦相似度建立伪异常内部子图 $\mathbf{A}^{(a)}$
3. 构建块对角增广图——伪异常子图与真实图断开，避免引入虚假混合边：

$$\mathbf{A}_{aug} = \begin{bmatrix} \mathbf{A}_{obs} & \mathbf{0} \\ \mathbf{0} & \mathbf{A}^{(a)} \end{bmatrix}$$

**设计动机**：传统方法中，异常信号被填补偏差掩盖；通过显式生成位于正常-异常边界的伪样本，模型可以学习到清晰的决策边界，而非被大量正常样本淹没。

### 损失函数 / 训练策略

**预训练阶段**（在原始不完整图上）：

$$\mathcal{L}_{pretrain} = \mathcal{L}_{dist} + \alpha \mathcal{L}_{feat} + \lambda \mathcal{L}_{recon}$$

**微调阶段**（在增广图上，含伪异常）：

$$\mathcal{L}_{finetune} = \mathcal{L}'_{dist} + \alpha \mathcal{L}_{feat} + \lambda \mathcal{L}_{recon}$$

其中混合 Sinkhorn 损失 $\mathcal{L}'_{dist}$ 同时约束正常嵌入在球内、伪异常嵌入在环壳上：

$$\mathcal{L}'_{dist} = \text{Sinkhorn}(\mathbf{Z}, \mathcal{N}_r) + \text{Sinkhorn}(\mathbf{Z}_{pseudo}, \mathcal{N}_{\{r_a < \|z\| \leq r_b\}})$$

- 超参默认值：$\alpha=0.01, \lambda=0.001$，预训练和微调各 100 epochs
- 优化器：Adam，lr 从 {0.0001, 0.001, 0.0005} 中选取
- 推理时异常打分：$s_i = \|z_i\|_2$（距原点越远越异常）

## 实验关键数据

### 主实验

**30% 缺失率下 AUROC 对比（最佳填补方法 + 最佳检测器 vs M2V-UGAD）**

| 数据集 | 最佳基线 | M2V-UGAD | 提升 |
|-------|---------|----------|------|
| Cora | 0.86 (ASD-VAE+ADA-GAD) | **0.93** | +0.07 |
| Citeseer | 0.90 (Mean/GAIN+ADA-GAD) | **0.92** | +0.02 |
| Books | 0.70 (ASD-VAE+ADA-GAD) | **0.63** | - |
| Disney | 0.78 (ASD-VAE+ADA-GAD) | **0.81** | +0.03 |
| Flickr | 0.75 (GAIN+ADA-GAD) | **0.93** | +0.18 |
| ACM | 0.82 (GAIN+ADA-GAD) | **0.92** | +0.10 |
| Reddit | 0.56 (ASD-VAE+PREM) | **0.58** | +0.02 |

在 25 种 (填补方法×检测器) 基线组合中，M2V-UGAD 在多数数据集上显著领先。

**不同缺失率下 Cora 数据集 AUROC**

| 缺失率 | 最佳基线 | M2V-UGAD |
|-------|---------|----------|
| 10% | 0.86 | **0.93** |
| 20% | 0.85 | **0.93** |
| 30% | 0.84 | **0.93** |
| 40% | 0.83 | **0.92** |
| 50% | 0.83 | **0.92** |

M2V-UGAD 在缺失率从 10% 到 50% 的范围内性能几乎不变（0.93→0.92），展现出极强的鲁棒性。

### 消融实验

| 变体 | Cora | Citeseer | Books | Disney |
|------|------|----------|-------|--------|
| 完整模型 | 0.93 | 0.92 | 0.63 | 0.81 |
| 去掉 $\mathcal{L}_{feat}$ | ~0.91 | ~0.90 | ~0.61 | ~0.77 |
| 去掉 $\mathcal{L}_{recon}$ | ~0.91 | ~0.90 | ~0.61 | ~0.77 |
| 去掉属性填补通路 | ~0.90 | ~0.89 | ~0.53 | ~0.74 |
| 去掉结构填补通路 | ~0.83 | ~0.82 | ~0.53 | ~0.46 |
| **去掉伪异常生成** | **~0.58** | **~0.57** | **~0.53** | **~0.60** |

- 去掉伪异常生成导致最严重下降（10-35%），说明反填补偏差机制是最核心组件
- 去掉结构通路比去掉属性通路影响更大，说明拓扑信息对异常检测更关键
- 两个约束损失的贡献相对温和但一致

### 关键发现

1. **伪异常生成是核心驱动力**：去掉后 AUROC 暴跌 10-35%，说明填补偏差是不完整图 GAD 的主要瓶颈
2. **双通路解耦至关重要**：去掉任一通路（尤其是结构通路）都显著降低性能，验证了跨视图干扰假设
3. **对缺失率不敏感**：10%-50% 缺失率下 AUROC 仅波动 1%，展现出真正的鲁棒性
4. **超参不敏感**：$\alpha, \lambda, \eta, r$ 在较大范围内性能稳定
5. **t-SNE 可视化**：M2V-UGAD 的嵌入中正常/异常节点清晰分离，基线方法则严重混叠

## 亮点与洞察

- **问题定义新颖**：首次研究属性和拓扑同时缺失的无监督 GAD，是一个更贴近现实的重要问题设定
- **填补偏差的识别与对策精妙**：通过在潜空间边界采样生成伪异常，巧妙地将填补偏差转化为可学习的对比信号
- **Sinkhorn vs SVDD 的选择有理论基础**：全局分布匹配优于点到中心距离，避免平凡坍缩
- **块对角增广图的设计细节**：伪异常子图与真实图断开连接，既提供了对比信号又不引入拓扑噪声

## 局限与展望

1. **异常注入实验的局限**：4个数据集的异常是按 CoLA 协议合成的（结构异常+上下文异常），可能无法反映真实异常模式
2. **计算效率未充分讨论**：Sinkhorn 散度计算（1000次迭代）和 PPR 扩散的计算开销在大规模图上可能成为瓶颈
3. **缺失机制假设**：实验采用随机缺失（MCAR），但现实中缺失可能是非随机的（如异常节点更可能缺失属性）
4. **仅支持节点级异常检测**：未扩展到边级或子图级异常
5. **Books 数据集上未超过最佳基线**（0.63 vs 0.70），小规模、低维数据集上可能存在过拟合风险

## 相关工作与启发

- 该工作填补了"不完整图 + 无监督异常检测"交叉领域的空白
- 伪异常生成思想可迁移到其他存在类不平衡的检测任务
- 双通路解耦 + 下游融合的范式对多视图学习有通用参考价值
- Sinkhorn 散度在超球面上的应用为异常检测的潜空间设计提供了新思路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 问题定义新颖，三个核心机制环环相扣，设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 7个数据集、25种基线组合、多缺失率、消融和敏感性分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，motivation 有说服力，公式符号统一
- 价值: ⭐⭐⭐⭐ — 解决了一个实际且重要的问题，但工业落地需更多大规模验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[AAAI 2026\] FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI](fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/object_detection/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/object_detection/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->
