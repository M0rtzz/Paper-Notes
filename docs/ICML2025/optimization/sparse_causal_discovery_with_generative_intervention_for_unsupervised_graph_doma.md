---
title: >-
  [论文解读] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation
description: >-
  [优化] 提出 SLOGAN 框架，通过稀疏因果图构建与信息瓶颈解耦因果/虚假特征，结合跨域虚假特征交换的生成式干预机制和类别自适应伪标签动态校准，实现无监督图域自适应中稳定的因果特征迁移。
tags:
  - 优化
---

# Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation

- **会议**: ICML 2025
- **arXiv**: [2507.07621](https://arxiv.org/abs/2507.07621)
- **代码**: 未公开
- **领域**: 图域自适应 / 因果发现 / 稳定学习
- **关键词**: 无监督图域自适应, 稀疏因果建模, 生成式干预, 伪标签校准, 互信息瓶颈

## 一句话总结

提出 SLOGAN 框架，通过稀疏因果图构建与信息瓶颈解耦因果/虚假特征，结合跨域虚假特征交换的生成式干预机制和类别自适应伪标签动态校准，实现无监督图域自适应中稳定的因果特征迁移。

## 研究背景与动机

无监督图域自适应（UGDA）旨在利用有标签的源域图数据在无标签的目标域中取得良好性能。现有方法面临两个核心挑战：

**因果-虚假特征纠缠**：图数据天然包含因果关系和统计相关性。传统方法仅依赖语义标签训练模型，无法区分二者。例如在 PTC 毒性预测数据集中，分子骨架结构是致癌性的因果因素，但性别、物种等实验变量仅是统计相关。未显式解耦时，虚假因素通过残余相关干扰跨域泛化。

**全局对齐策略失效**：现有对抗学习方法采用全局域分布对齐策略，容易导致信息坍塌——丢弃关键的稀有子结构，同时无法有效抑制虚假因素。图数据因结构复杂和高维稀疏性使这些问题更加严重。

这些局限性促使作者设计显式的因果-虚假解耦和局部稳定性保持机制。

## 方法详解

### 整体框架

SLOGAN 由三个互补的稳定性增强组件组成：
1. 基于稀疏因果发现的特征解耦
2. 带置信度校准的渐进式稳定对齐判别学习
3. 带协方差约束的生成式干预机制

### 关键设计一：稀疏因果发现与特征解耦

**结构因果模型（SCM）构建**：建立稀疏因果图，其中 $L$ 为标签、$C$ 为因果特征、$S$ 为虚假特征、$PL$ 为伪标签。通过三条关键因果路径实现稳定性：
- 稀疏特征生成：$C \rightarrow \mathcal{G} \leftarrow S$
- 标签稳定性：$L \rightarrow C \leftarrow PL$
- 域稀疏性：$D^{so} \rightarrow S \leftarrow D^{ta}$

**稳定性感知解耦学习**：基于稀疏变量独立性（SVI）原则，优化目标为：

$$\underbrace{\max I(Y; Z^c)}_{\text{稳定预测}} - \underbrace{\beta I(Z^s; Z)}_{\text{残余控制}} + \underbrace{\min I(Z^s; Y)}_{\text{虚假抑制}}$$

**因果特征提取**：使用 InfoNCE 最大化因果特征 $\mathbf{z}^c$ 与标签 $\mathbf{y}$ 的互信息：

$$\min \mathcal{L}_{MI}^c = \mathbb{E}_{p(\mathbf{z}^c, \mathbf{y})}[\xi] - \log \mathbb{E}_{p(\mathbf{z}^c)p(\mathbf{y})}[e^\xi]$$

其中 $\xi = F^c(\mathbf{z}^c, \mathbf{y}) = {\mathbf{z}^c}^T W \mathbf{y}$ 是双线性映射。

**虚假特征抑制**：通过变分信息瓶颈（VIB）最小化虚假特征与标签互信息并控制残余信息：

$$\min \mathcal{L}_{MI}^s = I(\mathbf{z}^s, \mathbf{y}) - \beta I(\mathbf{z}^s, \mathbf{z})$$

### 关键设计二：无偏判别学习

为避免伪标签过度自信问题，设计类别自适应动态校准策略：

1. 基于因果特征计算目标域预测分布和置信度分数 $s_i^{ta} = \max_c \mathbf{p}_i^{ta}[c]$
2. 计算类别自适应系数 $\mathcal{M}_c$ 和阈值 $\tau_c = \mathcal{M}_c \cdot \tau$（$\tau = 0.95$）
3. 构建置信样本集 $\mathcal{C}$ 进行交叉域联合优化

### 关键设计三：生成式干预与协方差约束

使用两层 MLP 作为生成模型 $G(\cdot, \cdot)$，通过 L2 距离优化重建：

$$\mathcal{L}_{ge} = \mathbb{E} \|\mathbf{z} - G(\mathbf{z}^c, \mathbf{z}^s)\|_2^2$$

**跨域虚假特征交换**：在 mini-batch 中交换不同域样本的虚假特征，生成组合样本 $\mathbf{z}_i^{+k} = G(\mathbf{z}_i^c, \mathbf{z}_k^s)$，迫使模型仅依赖因果特征进行重建。

**干预不变性约束**：

$$\mathcal{L}_{inv} = \mathcal{L}_{re} + \mathbb{E}_{\mathbf{z}_i \in \mathcal{B}^{so}, \mathbf{z}_k \in \mathcal{B}^{ta}} \|\mathbf{z}_i^{+k} - \mathbf{z}_i\|^2$$

### 损失函数

总体优化目标：

$$\mathcal{L} = \mathcal{L}_{sup} + \gamma \mathcal{L}_{dis} + \eta \mathcal{L}_{inv}$$

其中 $\mathcal{L}_{sup} = \mathcal{L}_{so} + \mathcal{L}_{ta}$，$\gamma = 0.003$，$\eta = 0.1$。先在源域预热训练，再加入目标域优化。

## 理论保证

**定理 3.1**：在稳定因果图构建下，满足因果充分性、虚假抑制和生成干预三个条件时，目标域误差以高概率被界定为：

$$\epsilon_T(h) \leq \hat{\epsilon}_S(h) + C\sqrt{\epsilon_1} + L\sqrt{\epsilon_2} + C(n_S, \delta)$$

即目标域误差受源域经验误差、虚假相关抑制程度和重建精度共同约束。

## 实验关键数据

### 主实验结果

| 数据集 | SLOGAN Avg. | MTDF (SOTA) Avg. | 提升 |
|--------|------------|------------------|------|
| PTC | **67.8%** | 65.5% | +2.3% |
| NCI1 | **70.6%** | 69.5% | +1.1% |
| Twitter | **64.7%** | 64.2% | +0.5% |
| Letter-Med | **73.5%** | 71.3% | +2.2% |

在 4 个基准数据集共 48 个域迁移场景中，SLOGAN 均取得最优或接近最优的成绩。

### 消融实验（NCI1 数据集）

| 变体 | Avg. | 下降 |
|------|------|------|
| 完整模型 | 70.6% | - |
| w/o $\mathcal{L}_{sup}$ | 67.5% | -3.1% |
| w/o $\mathcal{L}_{inv}$ | 69.5% | -1.1% |
| w/o $\mathcal{L}_{dis}$ | 69.6% | -1.0% |
| Baseline (GCN) | 63.3% | -7.3% |

每个组件都对最终性能有贡献，$\mathcal{L}_{sup}$ 的影响最大。

### 关键发现

1. **可视化验证**：t-SNE 显示因果特征跨域分布一致（域不可知），且按语义标签清晰分离
2. **可扩展性**：SLOGAN 在参数量和延迟最小的情况下取得最佳性能
3. **超参数鲁棒性**：对 $\gamma$ 和 $\eta$ 的变化不敏感，性能稳定

## 亮点与洞察

1. **因果视角切入 UGDA 问题**：不同于全局对齐策略，从因果推断角度分解特征，更加合理
2. **生成式干预是亮点**：通过跨域虚假特征交换打破局部虚假耦合，比对抗学习更精细
3. **类别自适应伪标签校准**：有效缓解长尾/不平衡场景中的误差传播问题
4. **理论保证完备**：提供了目标域误差的概率上界

## 局限性

1. **实验规模有限**：数据集规模偏小（PTC、NCI1 等），未在大规模图数据上验证
2. **假设较强**：因果图结构需要预先假定，实际中因果关系可能更复杂
3. **仅限图分类任务**：未扩展至节点分类和链接预测等图学习任务
4. **计算开销分析不充分**：虽然声称可扩展，但缺少与 baseline 的详细耗时对比

## 相关工作

- **图域自适应**：CoCo、MTDF 等全局对齐方法，DUA、DARE-GRAM 等域差异最小化方法
- **因果发现与解耦**：IRM（Arjovsky et al., 2019）、IDEA（Wang et al., 2024a）等因果表示学习
- **图神经网络分类**：GCN、GIN、GAT 等消息传递架构

## 评分

⭐⭐⭐⭐ (4/5)

方法设计完整，因果解耦+生成干预+伪标签校准三位一体。理论分析和实验覆盖较全。但实验数据集偏小，提升幅度有限（0.5%~2.3%），且因果图的先验假设可能限制方法的泛化能力。

<!-- RELATED:START -->

## 相关论文

- [Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [GCAL: Adapting Graph Models to Evolving Domain Shifts](gcal_adapting_graph_models_to_evolving_domain_shifts.md)
- [Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](../../NeurIPS2025/optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)

<!-- RELATED:END -->
