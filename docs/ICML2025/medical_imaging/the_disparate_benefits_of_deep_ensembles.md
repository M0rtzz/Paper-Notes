---
title: >-
  [论文解读] The Disparate Benefits of Deep Ensembles
description: >-
  [ICML 2025][医学图像][Deep Ensembles] 系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因，Hardt 后处理可有效缓解。
tags:
  - ICML 2025
  - 医学图像
  - Deep Ensembles
  - 算法公平性
  - 差异化收益
  - 预测多样性
  - 后处理
---

# The Disparate Benefits of Deep Ensembles

**会议**: ICML 2025  
**arXiv**: [2410.13831](https://arxiv.org/abs/2410.13831)  
**代码**: 无  
**领域**: 算法公平性 / 深度学习  
**关键词**: Deep Ensembles、算法公平性、差异化收益、预测多样性、后处理

## 一句话总结

系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因，Hardt 后处理可有效缓解。

## 研究背景与动机

- **Deep Ensembles 广泛应用**：通过简单集成多个DNN提升预测性能和不确定性估计，被大量部署在医疗、金融等高风险场景
- **算法公平性关注**：当模型在受保护群体（如性别、种族、年龄）间表现不均时，可能加剧社会不公
- **研究空白**：
    - 单个DNN的公平性问题已被广泛研究
    - 浅层模型集成的公平性有部分探索
    - **但 Deep Ensembles 对群体公平性的影响几乎未被研究过**
- **最相关工作 Ko et al. (2023)**：研究了子群性能变化但未使用标准公平性指标，且结论偏积极（认为集成仅有正面影响）

**核心发现预告**：集成的性能提升并非"水涨船高"式普惠所有群体，而是不成比例地偏向某些群体。

## 方法详解

### 整体框架

**实验驱动的研究**，非提出新模型，而是系统分析现象、原因和缓解策略：

1. **现象发现**：量化 Deep Ensemble 在多个公平性指标上的影响
2. **原因分析**：通过预测多样性（predictive diversity）解释差异化收益
3. **缓解策略**：验证 Hardt 后处理的有效性

### 关键设计

**1. 公平性指标体系**

三种标准群体公平性指标：
- **统计平权差异（SPD）**：$\text{PR}_{A=1} - \text{PR}_{A=0}$（正预测率差异）
- **机会均等差异（EOD）**：$\text{TPR}_{A=1} - \text{TPR}_{A=0}$（真正例率差异）
- **平均赔率差异（AOD）**：$\frac{1}{2}|\text{TPR}_{A=1} - \text{TPR}_{A=0}| + \frac{1}{2}|\text{FPR}_{A=1} - \text{FPR}_{A=0}|$

其中 $A=1$ 表示优势群体，所有指标取值 $\in [0, 1]$，0为最公平。

**2. 预测多样性分析**

利用 Jeffares et al. (2023) 的预测多样性定义：

$$\overline{\text{DIV}} = \frac{1}{K}\sum_{k=1}^K \left[\log\left(\frac{1}{N}\sum_{n=1}^N p(y_k|\mathbf{x}_k, \mathbf{w}_n)\right) - \frac{1}{N}\sum_{n=1}^N \log p(y_k|\mathbf{x}_k, \mathbf{w}_n)\right]$$

- 等式右侧 = 集成对数似然 - 平均成员对数似然
- 直觉：若集成成员对某群体的预测更"分散"（多样性更高），该群体从集成中获益更多
- **核心发现**：群体间预测多样性差异大 → 差异化收益效应显著；差异小 → 效应不显著

**3. Hardt 后处理**

- Deep Ensembles 因更好的校准性（calibration），其预测分布对**决策阈值更敏感**
- Hardt后处理通过为不同群体设置不同阈值来满足公平性约束
- 集成的良好校准使得阈值调整更有效

### 损失函数 / 训练策略

- 5种架构：ResNet18/34/50、RegNet-Y 800MF、EfficientNetV2-S
- 交叉熵损失，SGD+momentum=0.9
- FairFace训练100 epochs，CheXpert训练30 epochs
- 每种配置独立训练10个模型，5个随机种子，共1000个模型

## 实验关键数据

### 主实验

**数据集**：FairFace（人脸分析）、UTKFace（分布偏移测试）、CheXpert（医学影像）

**FairFace + ResNet50**（核心结果表，Δ = 集成 - 平均成员）：

| 目标/群体 | Δ Accuracy↑ | Δ SPD↓ | Δ EOD↓ | Δ AOD↓ |
|-----------|------------|--------|--------|--------|
| age/gender | +0.022 | -0.022 | -0.017 | -0.017 |
| age/race | +0.022 | -0.009 | -0.012 | -0.007 |
| gender/age | +0.014 | -0.001 | -0.007 | -0.004 |
| race/age | +0.015 | -0.004 | +0.005 | -0.001 |

注：Δ SPD/EOD/AOD为负表示公平性**恶化**

- 15个任务中的大多数出现统计显著的差异化收益效应
- 效应在age作为target时最强（SPD恶化达0.022）
- 效应在分布偏移下（UTKFace）同样存在且幅度相似

**CheXpert医学影像**：
- Protected group = age时，SPD/EOD/AOD三个指标均显著恶化
- 效应幅度较小（AUROC提升仅0.005），但公平性指标仍受影响

### 消融实验

- **模型大小影响**：较大模型（ResNet50 > ResNet34 > ResNet18）的差异化收益效应更强
- **架构一致性**：5种架构的结果一致
- **异构集成**：混合不同架构的集成同样出现差异化收益
- **集成成员数量**：添加第一个成员（1→2）时公平性恶化最剧烈，之后趋于平缓

### 关键发现

**预测多样性解释**（核心因果链）：
- 当存在显著差异化收益时，优势群体的 $\overline{\text{DIV}}$ 系统性高于劣势群体
- 当不存在差异化收益时，两群体的 $\overline{\text{DIV}}$ 差异很小
- 合成实验进一步验证：人工制造的 per-group 多样性差异导致可控的公平性变化

**Hardt后处理效果**：
- 单模型上Hardt效果有限
- Deep Ensemble上Hardt效果显著——因为集成的预测分布更好校准，阈值更可靠
- 后处理后可以同时保留集成的性能增益和满足公平性约束

## 亮点与洞察

1. **首次定义"差异化收益效应"**：揭示了一个被忽视的现象——广泛使用的Deep Ensembles可能系统性地加剧群体不公平
2. **因果解释清晰**：通过预测多样性的 per-group 差异，从理论层面解释了效应产生的机制
3. **实践影响重大**：在医疗影像、人脸识别等高风险领域，性能提升伴随公平性恶化的隐患必须被意识到
4. **缓解策略简单有效**：不需要修改训练过程，仅通过Hardt后处理即可兼顾性能和公平性
5. **实验规模大**：1000个模型、15个任务、5种架构的全面评估

## 局限与展望

- 仅考虑二元保护属性和二元分类任务，多类别场景（如多种族分组）未涉及
- 公平性指标有其固有局限——满足数学定义不等于真实公平
- 未分析差异化收益效应与训练数据不平衡的具体关系（虽然隐含相关）
- 后处理方法需要保护属性在推断时可用，某些场景下不现实
- 未探索in-processing方法（如对抗去偏、公平约束训练）能否在集成中更有效
- 缺乏对回归任务和连续公平性指标的分析

## 相关工作与启发

- **Lakshminarayanan et al. (2017)**：Deep Ensembles原始论文
- **Hardt et al. (2016)**：Equalized Odds后处理方法
- **Jeffares et al. (2023)**：预测多样性的理论分析（Theorem 4.3）
- **Ko et al. (2023)**：最相关前期工作，但结论不同（本文发现负面影响）
- **Abe et al. (2022, 2024)**：Deep Ensembles性能提升的机制分析

**启发**：性能提升≠所有群体均匀受益。在部署集成模型到高风险场景前，必须进行per-group的公平性审计。预测多样性可作为预警指标：若 per-group 多样性差异大，集成后公平性可能恶化。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究Deep Ensembles对群体公平性的影响，发现重要新现象
- 实验充分度: ⭐⭐⭐⭐⭐ — 1000个模型、3个数据集、5种架构、3个公平性指标，极其全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验详实，但理论分析深度可进一步加强
- 价值: ⭐⭐⭐⭐⭐ — 直接影响Deep Ensembles在高风险场景的部署决策

<!-- RELATED:START -->

## 相关论文

- [Learning Conformational Ensembles of Proteins Based on Backbone Geometry](../../NeurIPS2025/medical_imaging/learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](../../NeurIPS2025/medical_imaging/jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)
- [Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [Efficient Noise Calculation in Deep Learning-based MRI Reconstructions](efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)
- [DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](../../NeurIPS2025/medical_imaging/dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)

<!-- RELATED:END -->
