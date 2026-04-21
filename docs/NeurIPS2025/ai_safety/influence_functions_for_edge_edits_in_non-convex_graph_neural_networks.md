---
title: >-
  [论文解读] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks
description: >-
  [NeurIPS 2025][AI安全][影响函数] 提出适用于非凸 GNN 的边编辑影响函数，通过 proximal Bregman 响应函数放松凸性假设，并同时考虑参数偏移和消息传播两方面的影响，支持边的删除和插入。
tags:
  - NeurIPS 2025
  - AI安全
  - 影响函数
  - 图神经网络
  - 边编辑
  - 非凸优化
  - 图重连
---

# Influence Functions for Edge Edits in Non-Convex Graph Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2506.04694](https://arxiv.org/abs/2506.04694)  
**代码**: 无  
**领域**: 图神经网络 / AI安全  
**关键词**: 影响函数, 图神经网络, 边编辑, 非凸优化, 图重连

## 一句话总结

提出适用于非凸 GNN 的边编辑影响函数，通过 proximal Bregman 响应函数放松凸性假设，并同时考虑参数偏移和消息传播两方面的影响，支持边的删除和插入。

## 研究背景与动机

**领域现状**：GNN 中各边对模型行为的影响尚不明确，理解边的贡献对可解释性和鲁棒性至关重要。影响函数可以高效地估计移除训练数据对模型的影响，无需重训练。

**现有痛点**：现有图影响函数（GIF）依赖严格凸性假设（实际 GNN 是非凸的），仅考虑边删除而忽略边插入，且不能捕捉边编辑导致的消息传播路径变化。

**核心矛盾**：标准影响函数要求损失函数严格凸，但常用 GNN 架构天然非凸；边编辑改变了 GNN 的计算图结构，而传统影响函数仅建模参数变化。

**本文目标**：为非凸 GNN 设计准确的边编辑影响函数，同时支持删除和插入，并显式建模消息传播效应。

**切入角度**：结合 proximal Bregman 响应函数（PBRF）和链式法则分解，将影响分解为参数偏移项和消息传播项。

**核心 idea**：通过 edge-edit PBRF 放松凸性假设，并将影响函数分解为参数偏移+消息传播两部分，统一处理边的删除与插入。

## 方法详解

### 整体框架

给定图 $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathbf{X})$ 和评估函数 $f(\theta, \mathcal{G})$，方法通过链式法则将边编辑的影响分解为两部分：

$$\frac{df(\theta^*_\epsilon, \mathcal{G}^\epsilon)}{d\epsilon}\bigg|_{\epsilon=0} = \underbrace{\nabla_\theta f \cdot \frac{\partial \theta^*_\epsilon}{\partial \epsilon}}_{\text{参数偏移}} + \underbrace{\frac{\partial f}{\partial A^\epsilon}\frac{\partial A^\epsilon}{\partial \epsilon}}_{\text{消息传播}}$$

### 关键设计

1. **Edge-edit PBRF（边编辑 Proximal Bregman 响应函数）**：扩展 Bae et al. 的 PBRF 到图边编辑场景：

    $\theta^*_\epsilon := \arg\min_\theta \frac{1}{N}\sum_{v \in \mathcal{V}_{train}} D_\mathcal{L}(h_v^{\mathcal{G},\theta}, h_v^{\mathcal{G},\theta_s}) + \frac{\lambda}{2}\|\theta - \theta_s\|^2 + \sum_v \epsilon(\mathcal{L}(h_v^{\mathcal{G},\theta}) - \mathcal{L}(h_v^{\mathcal{G}^{-1/N},\theta}))$

    - 前两项约束参数不远离参考点 $\theta_s$（输出空间和参数空间）
    - 第三项鼓励参数在原图上失败、在编辑后的图上成功，从而响应边编辑
    - 不再要求损失函数严格凸，只需损失相对输出凸（如交叉熵、MSE自然满足）

2. **参数偏移项**：利用广义 Gauss-Newton Hessian $\mathbf{G} = \mathbf{J}_{h\theta_s}^\top \mathbf{H}_{h_s} \mathbf{J}_{h\theta_s} + \lambda\mathbf{I}$：

    $-\nabla_\theta f(\theta_s, \mathcal{G})^\top \mathbf{G}^{-1} \sum_v (\nabla_\theta \mathcal{L}(h_v^{\mathcal{G},\theta_s}) - \nabla_\theta \mathcal{L}(h_v^{\mathcal{G}^{-1/N},\theta_s}))$

3. **消息传播项**：显式计算边权重变化对评估函数的直接影响（不经过参数变化）：

    $(2\mathbb{I}[\{u,v\} \in \mathcal{E}] - 1) \cdot N \cdot \left(\frac{\partial f(\theta_s, \mathcal{G})}{\partial A_{uv}} + \frac{\partial f(\theta_s, \mathcal{G})}{\partial A_{vu}}\right)$

   这一项被 GIF 等先前方法完全忽略，但实验证明其与参数偏移的相关性低且量级相当。

4. **多种评估指标**：

    - **Over-squashing 度量**：通过掩蔽 L-hop 邻域特征衡量信息传播影响
    - **Over-smoothing（Dirichlet 能量）**：量化邻接节点表示的差异度
    - **验证损失**：标准交叉熵

### 损失函数 / 训练策略

不涉及专门训练。使用 LiSSA 算法近似 $\mathbf{G}^{-1}$ 的乘积。影响函数在已训练的 GNN 上计算，$\theta_s$ 为训练后的参数。

## 实验关键数据

### 主实验

影响预测准确性（预测影响 vs 实际影响的相关性，Cora 数据集，4 层 GCN）：

| 方法 | Over-squashing 相关性 | Dirichlet 能量相关性 | 验证损失相关性 |
|------|---------------------|--------------------|--------------| 
| GIF（先前方法） | 0.09 | 0.14 | — |
| **本文方法（删除）** | **~0.95** | **~0.95** | **~0.95** |
| **本文方法（插入）** | **~0.95** | **~0.95** | **~0.95** |

基于影响函数的图重连对测试准确率的影响：

| 方法 | Cora | CiteSeer | PubMed |
|------|------|----------|--------|
| GCN（原始） | 81.0±0.3 | 69.3±0.5 | 75.6±1.0 |
| Random | 81.1±0.4 | 69.2±0.4 | 75.7±0.8 |
| GIF | 80.9±0.5 | 69.2±0.5 | 75.6±0.9 |
| **Ours (VL)** | **82.1±0.5** | **69.6±0.7** | **76.4±1.3** |

### 消融实验

多边同时编辑的影响预测准确性（Cora，GCN）：

| 同时插入边数 | 预测 vs 实际相关性 |
|-------------|------------------|
| 10 | ~0.90 |
| 20 | ~0.87 |
| 100 | 0.84 |

### 关键发现

- **消息传播项不可忽略**：参数偏移和消息传播的影响相关性低、量级相当，两者必须同时建模。
- **验证损失是最有效的攻击/改进指标**：基于验证损失的边编辑在三个数据集上均能提升测试准确率，而 Dirichlet 能量和 over-squashing 的优化不一定转化为更好的分类性能。
- **同质性分析**：无论同质图还是异质图，增加同质性边（连接同类节点）总是有益的，这与 GCN 的同质性偏好一致。
- **图重连方法的副作用**：BORF 和 FoSR 的边插入虽缓解 over-squashing，但加剧 over-smoothing。

## 亮点与洞察

- **统一框架**：首次在同一影响函数框架下处理边的删除和插入，且适用于非凸 GNN。
- **分解思想优雅**：将图编辑影响分解为参数偏移和消息传播两个独立且互补的维度。
- **多视角分析工具**：提供 over-squashing、over-smoothing、验证损失三个视角的统一分析，揭示了现有图重连方法的未察觉副作用。
- **对抗攻击能力**：基于验证损失的影响函数引导的对抗攻击优于专门的攻击方法（DICE、PRBCD）。

## 局限与展望

- 同时编辑大量边时精度下降（一阶近似误差），在 100 条边时相关性降至 0.84。
- 对深层 GNN 的可扩展性有待验证。
- LiSSA 近似逆 Hessian 的计算仍有一定开销，对大规模图可能成为瓶颈。
- 目前仅验证 GCN/GAT/ChebNet，对更复杂的 GNN 架构（如 Graph Transformer）未涉及。

## 相关工作与启发

- **经典影响函数** (Koh & Liang)：要求严格凸性，对深度网络不可靠。
- **PBRF** (Bae et al.)：通过 Bregman 散度和阻尼项放松凸性假设，本文将其扩展到图场景的边编辑。
- **GIF** (Chen et al., Wu et al.)：图上的影响函数先驱，但仅考虑参数变化，本文补充了消息传播部分，关键性地提高了预测精度（0.09→0.95）。
- 影响函数的多元化应用：模型可解释性、数据估值、对抗分析、机器遗忘。

## 评分

- 新颖性: ⭐⭐⭐⭐ 边编辑影响分解和 edge-edit PBRF 的设计新颖且有理论深度
- 实验充分度: ⭐⭐⭐⭐ 5 个数据集、3 种 GNN、3 种评估指标、多种应用场景
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，实验设计与理论紧密对应
- 价值: ⭐⭐⭐⭐ 为 GNN 的可解释性和鲁棒性分析提供了实用工具

<!-- RELATED:START -->

## 相关论文

- [ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [Lyapunov Stable Graph Neural Flow](../../CVPR2025/ai_safety/lyapunov_stable_graph_neural_flow.md)
- [Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)

<!-- RELATED:END -->
