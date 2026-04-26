---
title: >-
  [论文解读] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning
description: >-
  [NeurIPS 2025][图学习][不确定知识图谱] ssCDL 通过将三元组置信度从标量转换为高斯分布形式的置信度分布以捕获邻近置信度的监督信号，并利用元自训练（meta self-training）为负采样三元组生成高质量伪置信度标签来重平衡训练数据，在不确定知识图谱补全的置信度预测和链接预测上显著超过所有基线方法。
tags:
  - NeurIPS 2025
  - 图学习
  - 不确定知识图谱
  - 置信度分布学习
  - 半监督学习
  - 元自训练
  - 知识图谱补全
---

# Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.16601](https://arxiv.org/abs/2510.16601)  
**代码**: https://github.com/seucoin/unKR/tree/main/unKR_ssCDL (有)  
**领域**: 图学习 / 知识图谱  
**关键词**: 不确定知识图谱, 置信度分布学习, 半监督学习, 元自训练, 知识图谱补全

## 一句话总结

ssCDL 通过将三元组置信度从标量转换为高斯分布形式的置信度分布以捕获邻近置信度的监督信号，并利用元自训练（meta self-training）为负采样三元组生成高质量伪置信度标签来重平衡训练数据，在不确定知识图谱补全的置信度预测和链接预测上显著超过所有基线方法。

## 研究背景与动机

**领域现状**：不确定知识图谱（UKG）为每个三元组关联一个置信度分数（0-1之间），比确定性知识图谱提供更精确的知识表示。典型 UKG 包括 NELL、ConceptNet、Probase 等。现有 UKG 补全方法（UKGE、PASSLEAF、BEUrRE、UPGAT 等）通过嵌入学习进行链接预测和置信度预测。

**现有痛点**：真实 UKG 中三元组的置信度分布极度不均衡——例如 NELL 中几乎所有存储的三元组置信度都大于 0.9，因为低置信度的三元组通常是错误的不会被存储。在如此偏斜的数据上学习嵌入，模型严重偏向高置信度区间，无法准确预测低置信度样本。

**核心矛盾**：嵌入学习需要在不同置信度水平上有充分样本，但 UKG 的本质决定了低置信度数据极度稀缺。存在两个挑战：Challenge 1——如何从已有的不均衡标签数据中提取未见置信度的监督信号；Challenge 2——如何为负采样产生的未标注三元组生成可靠的置信度标签。

**本文目标** 同时解决上述两个挑战，通过标签数据增强和未标注数据扩充双管齐下提升 UKG 嵌入质量。

**切入角度**：置信度本质是模糊概念（0.77 和 0.78 没有本质区别），受面部年龄估计中标签分布学习的启发，将单一置信度扩展为高斯分布。

**核心 idea**：将三元组置信度转为高斯分布以引入邻近置信度信号（解决 Challenge 1），用元自训练为负样本生成可靠伪置信度分布（解决 Challenge 2）。

## 方法详解

### 整体框架

ssCDL 由两个核心组件组成：**CDL-RL**（基于置信度分布学习的关系学习器）和 **PCDG**（伪置信度分布生成器）。CDL-RL 在标注数据和 PCDG 生成的伪标注数据上迭代学习 UKG 嵌入。PCDG 通过元学习优化——以 CDL-RL 在标注数据上的表现作为元目标来评估伪标签质量。两者通过元自训练交替优化。

### 关键设计

1. **置信度分布学习（CDL）**:

    - 功能：将单一置信度标量转换为离散分布，引入邻近置信度的监督信号
    - 核心思路：对于置信度为 $s$ 的三元组，以 $s$ 为均值、$\sigma$ 为标准差的高斯分布生成 101 维的置信度分布向量 $\boldsymbol{s}$（粒度 1/100）。预测侧拼接 $(h, r, t)$ 嵌入后经两层 FCN + Softmax 输出预测分布 $\hat{\boldsymbol{s}}$，用 KL 散度（分布相似度）+ MSE（期望与真实值偏差）联合优化。同时设计独立的链接预测分支（FCN + Sigmoid + 基于间隔的排序损失），以不确定性权重动态平衡两任务
    - 设计动机：置信度 0.78 附近的 0.76、0.77、0.79 也能提供监督信号，有效缓解数据稀疏。分布化理论上等价于对标签空间做了平滑化，类似 Label Smoothing 的效果但更加结构化

2. **伪置信度分布生成器（PCDG）**:

    - 功能：为负采样未标注三元组生成高质量伪置信度标签
    - 核心思路：PCDG 与 CDL-RL 共享相同结构但参数独立。元学习循环：PCDG 先为未标注数据生成 $\mathcal{D}_{tmp}$，CDL-RL 在 $\mathcal{D} \cup \mathcal{D}_{tmp}$ 上更新一步得 $\theta^+$，以 $\mathcal{L}(\mathcal{D}, \theta^+)$ 为元目标更新 PCDG 参数 $\eta$。选择策略：若伪分布的最大描述度超过阈值则选入训练集
    - 设计动机：传统自训练存在渐进漂移问题，元学习通过反向验证"伪标签是否真的帮到了 CDL-RL"来筛选质量

3. **三阶段元自训练**:

    - 功能：协调两个组件的训练过程，确保稳定性
    - 核心思路：阶段①（$< T_{PCDG}$）：仅用标注数据训练 CDL-RL 使嵌入稳定。阶段②（$T_{PCDG} \le \cdot < T_{CDLRL}$）：开始训练 PCDG 但不将伪标签用于 CDL-RL。阶段③（$\ge T_{CDLRL}$）：PCDG 伪标签经筛选后参与 CDL-RL 训练
    - 设计动机：渐进引入组件避免早期噪声累积，阈值筛选进一步保证质量

### 损失函数 / 训练策略

CDL-RL 总损失：$\mathcal{L} = \frac{1}{2\lambda_{CP}^2}\mathcal{L}_{CP} + \frac{\phi}{2\lambda_{LP}^2}\mathcal{L}_{LP} + \log(\lambda_{CP} \cdot \lambda_{LP})$，其中 $\phi=0.1$ 限制负样本过多导致 $\mathcal{L}_{LP}$ 过大。每正样本生成 50 个负样本用于链接预测。伪标签仅参与 $\mathcal{L}_{CP}$ 优化。NL27k 嵌入维度 128，CN15k 嵌入维度 512。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | ssCDL | 最佳基线 | 改进 |
|--------|------|------|-------|----------|------|
| NL27k | 置信度预测 | MSE↓ | **0.009** | 0.019 (PASSLEAF-RotatE) | **-52.6%** |
| NL27k | 置信度预测 | MAE↓ | **0.042** | 0.051 (PASSLEAF-DistMult) | -17.6% |
| NL27k | 链接预测 | WMRR↑ | **0.727** | 0.715 (PASSLEAF-RotatE) | +1.7% |
| NL27k | 链接预测 | Hits@1↑ | **0.636** | 0.586 (PASSLEAF-ComplEx) | +8.5% |
| CN15k | 置信度预测 | MSE↓ | **0.034** | 0.094 (PASSLEAF-RotatE) | **-63.8%** |
| CN15k | 置信度预测 | MAE↓ | **0.116** | 0.248 (PASSLEAF-RotatE) | -53.2% |
| CN15k | 链接预测 | Hits@1↑ | **0.133** | 0.086 (PASSLEAF-ComplEx) | +54.7% |

### 消融实验

| 配置 | NL27k MSE↓ | NL27k MAE↓ | NL27k WMRR↑ | CN15k MSE↓ | CN15k MAE↓ |
|------|-----------|-----------|-------------|-----------|-----------|
| ssCDL (完整) | **0.009** | **0.042** | **0.727** | **0.034** | **0.116** |
| w/o CDL | 0.015 | 0.057 | 0.586 | 0.044 | 0.141 |
| w/o 元自训练 | 0.010 | 0.045 | 0.718 | 0.035 | 0.118 |

### 关键发现

- CDL 贡献远大于元自训练：去除 CDL 后 NL27k MSE 从 0.009 升至 0.015（+67%），去除元自训练仅升至 0.010（+11%），说明标注数据的监督信号增强比引入未标注数据更重要
- 在低置信度三元组（<0.5）的预测上，ssCDL 显著优于所有基线（低置信度 MAE 最低），验证了分布学习有效缓解不均衡
- 所有方法在 NL27k 上表现优于 CN15k，因为 ConceptNet 置信度定义基于数据源频率，区分度不足
- 链接预测提升不如置信度预测显著（间接受益 vs 直接优化），但仍全面超越基线

## 亮点与洞察

- **标签分布学习的跨领域迁移**：将面部年龄估计中的 Label Distribution Learning 迁移到知识图谱置信度建模，利用了置信度天然的连续模糊性。这个洞察简单但有效——启示我们在任何数值标签不均衡的场景都可以尝试"分布化"策略
- **元学习做伪标签质量控制**：不是简单用模型预测当伪标签，而是让元目标反向验证伪标签的有效性，比固定阈值筛选更优雅。PCDG 本身也在不断进化，形成良性循环
- **三阶段渐进策略**：先稳定嵌入→再训练生成器→最后引入伪标签的渐进方案，简单有效地避免了早期噪声干扰

## 局限与展望

- 仅在 NL27k（17.5万四元组）和 CN15k（24.1万四元组）两个小规模数据集实验，更大规模 UKG 的效果未知
- PCDG 与 CDL-RL 结构完全相同（两层 FCN），未探索更轻量或更专门的生成器架构
- 高斯标准差 $\sigma$ 固定为 0.6，未探索自适应调整（不同置信度区间可能需要不同的平滑程度）
- 元学习涉及二阶梯度计算，训练开销较大，可扩展性存疑
- 论文指出 ConceptNet 的置信度定义存在质量问题，呼吁建设更好的 UKG 评测基准

## 相关工作与启发

- **vs UKGE**: 使用 DistMult 评分函数直接预测标量置信度，忽略了置信度不均衡。ssCDL 正面解决此问题
- **vs PASSLEAF**: 也用半监督学习但仅缓解假阴性问题，用传统自训练易渐进漂移。ssCDL 的元自训练框架更稳健
- **vs BEUrRE**: Box embedding 建模实体不确定性（不同维度），ssCDL 关注三元组置信度层面
- **vs UPGAT**: 用子图特征+GAT，未处理置信度不均衡。ssCDL 提供了正交且互补的改进方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 置信度分布学习 + 元自训练的组合有新意，跨领域迁移值得肯定
- 实验充分度: ⭐⭐⭐ 仅两个小数据集，规模有限，但消融、低置信度分析和超参敏感性分析较完整
- 写作质量: ⭐⭐⭐⭐ 两个 Challenge 引出对应方案的结构清晰，方法推导细致
- 价值: ⭐⭐⭐ UKG 补全是细分方向，但置信度分布学习的思路有更广泛的迁移潜力

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](../../ACL2025/graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)

<!-- RELATED:END -->
