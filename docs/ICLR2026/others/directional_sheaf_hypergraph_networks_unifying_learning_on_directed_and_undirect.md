---
title: >-
  [论文解读] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs
description: >-
  [ICLR 2026][sheaf neural networks] 本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子…
tags:
  - "ICLR 2026"
  - "sheaf neural networks"
  - "directed hypergraphs"
  - "Laplacian"
  - "spectral methods"
  - "heterophily"
---

# Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs

**会议**: ICLR 2026  
**arXiv**: [2510.04727](https://arxiv.org/abs/2510.04727)  
**代码**: [GitHub](https://github.com/EmaMule/DirectionalSheafHypergraphs)  
**领域**: 其他  
**关键词**: sheaf neural networks, directed hypergraphs, Laplacian, spectral methods, heterophily

## 一句话总结

本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

## 研究背景与动机

**超图的高阶交互建模**：许多真实系统存在多实体间的高阶关系，传统图只能表达成对关系。超图通过超边连接多个节点来建模多路交互。

**无向超图的局限**：大多数 HGNN 仅处理无向超图，忽略了超边中可能存在的方向性（化学反应中反应物→产物、因果交互的发起方→接收方）。

**Sheaf 理论缓解异质性**：通过为节点和边分配向量空间及可学习的 restriction map，能有效缓解过平滑和异质性问题。但已有 Sheaf 超图方法不支持有向超图。

**已有 SHN 的谱性质缺陷**：Duta et al. (2023) 的 Sheaf Hypergraph Laplacian 不满足正半定性，无法作为合格卷积算子。

**有向图方法的成功经验**：Magnetic Laplacian 用复数相位编码方向，但未推广到超图。

## 方法详解

### 整体框架

DSHN 的核心是为有向超图重新定义一个合格的谱卷积算子。它先把方向信息以复数相位编码进 sheaf 的 restriction map，由此构造出一个 Hermitian 且正半定的 Directed Sheaf Hypergraph Laplacian，再以该 Laplacian 为基础搭建多项式滤波的扩散卷积网络。整个设计的关键在于让"方向"与"sheaf 的异质性建模"在同一个复值算子里相容，并保证它仍满足谱卷积所需的全部性质。

### 关键设计

**1. 方向性矩阵 $\mathcal{S}^{(q)}$：用复数相位区分头尾节点**

无向超图里节点对超边是对称的，无法表达"谁是反应物、谁是产物"这类方向角色。DSHN 借鉴 Magnetic Laplacian 的思路，为每个节点-超边对赋一个复值系数：头节点取 $1$，尾节点取 $e^{-2\pi i q}$，由参数 $q$ 统一控制方向信息的强度。当 $q=0$ 时所有系数退化为实数 $1$，模型回到无向情形；当 $q=1/4$ 时尾节点系数变为纯虚数 $e^{-\pi i/2}$，方向差异被编码进虚部，恰好与有向图上的 Magnetic Laplacian 对齐。这样一个标量参数就把方向性以可控、可退化的方式注入了整套算子，既保留了对无向数据的兼容，又给有向数据提供了相位上的判别能力。

**2. Directed Sheaf Hypergraph Laplacian：修正对角项得到合格算子**

有了方向编码后，DSHN 把它和 sheaf 的 restriction map 一起组装成 Laplacian $\mathbf{L}^{\vec{\mathcal{F}}} = \mathbf{D}_V - \mathbf{B}^{(q)\dagger}\mathbf{D}_E^{-1}\mathbf{B}^{(q)}$，其中 $\mathbf{B}^{(q)}$ 是带方向相位的关联块矩阵、$\mathbf{D}_V$ 与 $\mathbf{D}_E$ 分别为节点和超边的度矩阵。该算子的对角块为实值，承载节点自身信息；非对角块在超边有向时取复值，承载带方向的邻居耦合。值得强调的是，本文专门修正了 Duta et al. (2023) Sheaf Hypergraph Laplacian 中对角项系数的设置——正是那个系数导致其原算子不满足正半定，无法作为合格卷积核。修正之后，$\mathbf{L}^{\vec{\mathcal{F}}}$ 成为一个 Hermitian 算子，为后续谱性质打下基础。

**3. 谱性质保证：让 Fourier 变换与多项式滤波良定义**

谱卷积要成立，Laplacian 必须可对角化、特征值实且非负、整体正半定，并有有界谱。DSHN 证明 $\mathbf{L}^{\vec{\mathcal{F}}}$ 满足全部这些条件：因为它 Hermitian 所以可酉对角化、特征值为实数；通过把二次型写成 Dirichlet 能量并证其非负，得到正半定与非负特征值；进一步给出谱上界为 $1$。这组性质确保了图 Fourier 变换有良好定义、多项式滤波器在谱域稳定，从而 DSHN 的扩散卷积可以安全地堆叠多层而不发散。

**4. 统一泛化：一个框架退化出多种已有 Laplacian**

DSHN 的另一价值在于它的普适性。论文证明，在恰当取特殊参数时，$\mathbf{L}^{\vec{\mathcal{F}}}$ 可退化为多种已有算子：取平凡 sheaf 退化为标准超图 Laplacian，限制为图结构时退化为 Graph Sheaf Laplacian，去掉 sheaf 而保留方向相位时退化为 Magnetic Laplacian，此外还涵盖 Zhou 超图 Laplacian、GeDi Laplacian 等。这说明它不是又一个孤立定义，而是把"方向 / sheaf / 超图"三个维度的已有工作统一在同一个复值 Hermitian 算子之下。

**5. DSHNLight：detach 梯度换取可扩展性**

完整 DSHN 中 restriction map 需要随训练学习，每步都要重建 Laplacian，开销不小。DSHNLight 把 Laplacian 构建过程的梯度 detach、固定 restriction map 参数（相当于一次随机投影），只训练后续卷积与分类部分。这样大幅降低了计算成本，而实验上它在多个数据集上的性能与完整版相当、个别情况甚至更好，呼应了极限学习机里"随机特征也能很有效"的观察。

### 损失函数 / 训练策略

训练用标准交叉熵节点分类损失。restriction map 由一个 MLP 学习，输入是节点特征与超边特征的拼接。由于 Laplacian 与中间表示为复值，最终送入分类头前会先做 unwind，即把实部和虚部拼接还原成实向量，再接全连接层输出类别。

## 实验关键数据

### 主实验

7 个数据集上对比 13 个 baseline 的节点分类准确率：

| 数据集 | DSHN 相对最佳 baseline 提升 |
|--------|--------------------------|
| Cora (co-auth) | ~2% |
| Citeseer (co-auth) | ~5% |
| Senate-committees | ~8% |
| House-committees | ~4% |
| Walmart-trips | ~20% |
| Zoo | ~3% |
| 20Newsgroups | ~2% |

### 消融实验

| 变体 | 效果 |
|------|------|
| $q=0$（无方向） | 退化为无向 sheaf 方法，性能下降 |
| $q=1/4$（标准相位） | 有向数据上表现最佳 |
| Trivial sheaf ($\mathcal{F}=I$) | 退化为有向超图 Laplacian，性能大幅下降 |
| DSHNLight | 计算效率高，多数数据集性能接近 |

### 关键发现

- 方向性 + Sheaf 联合使用效果显著优于单独使用任一
- Duta et al. (2023) 的 Laplacian 确实存在负特征值（附录给出反例）
- DSHNLight 的"随机投影"策略出乎意料地有效
- 在异质性数据集上优势最为明显

## 亮点与洞察

1. 一个复值 Hermitian 算子统一了多种已有 Laplacian 定义
2. 严格纠正了 Duta et al. (2023) 的谱性质错误
3. 方向信息用复数相位编码的思路从有向图自然推广到超图
4. DSHNLight 与极限学习机思想呼应，说明随机特征在图学习中有效

## 局限与展望

- $nd \times nd$ Laplacian 的可扩展性问题
- $q$ 作为全局参数，未能为每条超边学习不同 $q$
- 实验仅覆盖节点分类
- 有向超图真实数据集稀缺
- 缺少 WL 层级等表达能力分析

## 相关工作与启发

- **Hansen & Gebhart (2020)**：Graph Sheaf NN → 本文推广到有向超图
- **Zhang et al. (2021)**：Magnetic Laplacian → 本文推广到超图 + sheaf
- **Duta et al. (2023)**：SheafHyperGNN → 本文修正其谱性质缺陷
- **启发**：复值 Hermitian + Sheaf 范式可推广到 simplicial complex 等更一般拓扑结构

## 评分

- **新颖性**: ⭐⭐⭐⭐ Sheaf + 有向超图的结合和统一性结果有重要理论价值
- **实验充分度**: ⭐⭐⭐⭐ 7 个数据集、13 个 baseline、完整消融
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，符号系统较重但定义精确
- **价值**: ⭐⭐⭐⭐ 修正了已有方法缺陷并提供统一框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](../../NeurIPS2025/others/sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)

</div>

<!-- RELATED:END -->
