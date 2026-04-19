---
title: >-
  [论文解读] Enhancing Graph Classification Robustness with Singular Pooling
description: >-
  [NeurIPS 2025][AI安全][图神经网络] 首次系统分析 flat pooling（Sum/Avg/Max）对图分类对抗鲁棒性的影响，推导各自的对抗风险上界，并提出 RS-Pool——利用节点嵌入矩阵的主奇异向量构建图级表示，在不牺牲 clean accuracy 的前提下显著提升对抗鲁棒性。
tags:
  - NeurIPS 2025
  - AI安全
  - 图神经网络
  - graph classification
  - pooling
  - adversarial attacks
  - singular value decomposition
---

# Enhancing Graph Classification Robustness with Singular Pooling

**会议**: NeurIPS 2025  
**arXiv**: [2510.22643](https://arxiv.org/abs/2510.22643)  
**代码**: [GitHub](https://github.com/king/rs-pool)  
**领域**: AI Safety / 图神经网络鲁棒性  
**关键词**: GNN robustness, graph classification, pooling, adversarial attacks, singular value decomposition

## 一句话总结
首次系统分析 flat pooling（Sum/Avg/Max）对图分类对抗鲁棒性的影响，推导各自的对抗风险上界，并提出 RS-Pool——利用节点嵌入矩阵的主奇异向量构建图级表示，在不牺牲 clean accuracy 的前提下显著提升对抗鲁棒性。

## 研究背景与动机

**领域现状**：GNN 在图表示学习中表现出色，但对抗鲁棒性研究主要集中在**节点分类**，对**图分类**场景关注不足。现有防御方法（GNNGuard、Jaccard 预处理、GCORN 等）主要针对 message-passing 阶段。

**现有痛点**：图分类需要将节点特征聚合为图级表示（pooling），但 pooling 操作对鲁棒性的影响几乎未被研究。不同 pooling 策略（Sum/Avg/Max）在对抗攻击下表现差异巨大，且与图结构和攻击类型强相关。

**核心矛盾**：图分类的图通常较小，节点级防御（如删除节点）可能破坏信息传播、损害 clean accuracy。需要一种在 pooling 阶段就能抵御扰动、同时保持表达力的方案。

**本文目标**：(1) 量化不同 pooling 对图分类鲁棒性的影响；(2) 设计一种兼顾鲁棒性和 clean accuracy 的 pooling 方法。

**切入角度**：矩阵扰动理论（Davis-Kahan / Wedin 定理）表明，当谱间隙充分大时，主奇异向量在有界扰动下保持稳定——这正好可以用来构建鲁棒的图级表示。

**核心 idea**：用节点嵌入矩阵 $H$ 的主右奇异向量作为图级表示，天然过滤对抗噪声中的不稳定分量。

## 方法详解

### 整体框架
- **输入**：图 $G=(V,E)$ 及节点特征 $X$
- **中间过程**：$L$ 层 GCN/GIN message passing 得到节点嵌入矩阵 $H \in \mathbb{R}^{n \times d}$
- **RS-Pool**：从 $H$ 提取主右奇异向量 $v_1$，缩放后作为图级表示
- **输出**：分类器对 $\tau v_1$ 的预测

### 关键设计

1. **Flat Pooling 鲁棒性理论分析**:

    - 功能：对 GCN 下的 Sum/Avg/Max pooling 推导对抗风险上界
    - 核心思路（Theorem 4.2）：设 $L$ 层 GCN，权重矩阵 $W^{(\ell)}$，扰动预算 $\epsilon$，$\hat{w}_u$ 为节点 $u$ 出发的长度 $L-1$ 归一化游走总和：
        - Sum pooling: $\gamma = (\prod_\ell \|W^{(\ell)}\|) \sum_{u} \hat{w}_u \epsilon$
        - Average pooling: $\gamma = \frac{1}{n}(\prod_\ell \|W^{(\ell)}\|) \sum_{u} \hat{w}_u \epsilon$
        - Max pooling: $\gamma = \sqrt{\min\{n,d_L\}}(\prod_\ell \|W^{(\ell)}\|) \max_{u} \hat{w}_u \epsilon$
    - 设计动机：Sum 的上界随图密度线性增长（脆弱），Avg 除以 $n$ 缓解，Max 只依赖单个最大节点（攻击目标明确时脆弱，随机攻击时可能更强）

2. **RS-Pool (Robust Singular Pooling)**:

    - 功能：用 $H$ 的主右奇异向量 $v_1$ 生成图级表示 $\tau v_1(H)$
    - 核心思路：$H = U\Sigma V^\top$，RS-Pool 映射 $H \mapsto \tau v_1(H)$，其中 $\tau > 0$ 为缩放因子
    - 鲁棒性保证（Theorem 5.1）：

    $\gamma = \frac{\tau\sqrt{2}\epsilon}{\sigma_1 - \sigma_2} \left(\prod_{\ell=1}^L \|W^{(\ell)}\|\right) \sum_{u=1}^n (\hat{w}_u)^2$

   关键：分母 $\sigma_1 - \sigma_2$（谱间隙）越大，上界越紧，鲁棒性越强
    - 设计动机：对抗扰动通常是低秩和局部的，主奇异方向最稳定，次要分量更容易被噪声污染

3. **Power Iteration 高效实现**:

    - 功能：避免计算完整 SVD，用幂迭代法近似主奇异向量
    - 核心思路：先计算 $S = H^\top H$，随机初始化 $v$，迭代 $v \leftarrow Sv/\|Sv\|_2$，$K$ 次后返回 $\tau Hv$
    - 复杂度 $O(K \times n \times d)$，$K$ 通常 2-5 次即收敛（GNN 嵌入的谱间隙通常较大）
    - 收敛速率 $\sigma_2/\sigma_1$ 越小越快，与鲁棒性上界形成**双重优势**：谱间隙大 → 既鲁棒又快

### 缩放因子 $\tau$ 的控制
- 参数化为 $\tau = \sigma_1(X)/\alpha$，$\alpha$ 可调
- Corollary 5.2：即使 message-passing 权重范数极大，$\gamma' = \min\{\gamma, 2\tau\}$ 给出有限上界
- 增大 $\alpha$（减小 $\tau$）→ 降低攻击成功率，但 clean accuracy 在中间值最优

## 实验关键数据

### 主实验：Clean 和 Attacked 分类准确率（GCN, $\epsilon=0.3$）

| 数据集 | 攻击 | Sum | Average | Max | RS-Pool |
|--------|------|-----|---------|-----|---------|
| PROTEINS | Clean | 74.2±3.1 | 70.8±2.2 | 73.2±2.3 | **73.5±2.9** |
| PROTEINS | PGD | 45.8±2.9 | 34.2±2.1 | 28.2±3.5 | **51.9±3.6** |
| PROTEINS | Random | 68.3±3.1 | 65.7±2.3 | 42.9±4.1 | **70.4±3.1** |
| D&D | Clean | 75.1±0.6 | 70.1±0.5 | 74.1±0.6 | 74.6±0.7 |
| D&D | PGD | 6.7±2.0 | 12.6±3.6 | 8.7±5.2 | **30.4±3.2** |
| D&D | Genetic | 60.5±4.2 | 63.1±2.6 | 21.4±1.5 | **67.2±2.9** |
| NCI1 | Clean | **70.6±0.8** | 67.9±1.6 | 68.2±1.2 | 70.1±1.2 |
| NCI1 | PGD | 23.4±1.1 | 14.1±2.1 | 19.7±1.7 | **26.3±1.8** |

### 与高级 pooling 方法的对比

| 数据集 | 攻击 | SAG | TopK-P | PAN-P | Sort-P | RS-Pool |
|--------|------|-----|--------|-------|--------|---------|
| PROTEINS | PGD | 45.5±2.1 | 48.5±1.2 | 33.3±1.5 | 31.8±2.9 | **51.9±3.6** |
| D&D | PGD | 9.0±3.8 | 5.9±2.9 | 9.6±2.8 | 16.6±1.9 | **30.4±3.2** |
| D&D | Genetic | 62.5±2.8 | 57.7±0.6 | 56.1±3.6 | 57.7±4.3 | **67.2±2.9** |

### 关键发现
- RS-Pool 在**所有攻击类型**下一致优于或持平所有 pooling 方法，clean accuracy 损失极小
- D&D 数据集上 PGD 攻击的提升最为显著：RS-Pool (30.4%) vs Sum (6.7%) vs Max (8.7%)，提升巨大
- Max pooling 在 Random/Genetic 攻击下表现最差（高度局部化，容易被针对），验证理论分析
- $\alpha$ 参数的实验验证了理论预测：增大 $\alpha$ 提高鲁棒性但降低 clean accuracy，存在最优平衡点

## 亮点与洞察
- **pooling 视角的鲁棒性分析**填补了重要空白：之前只关注 message-passing 的防御，忽略了最终聚合步骤的关键作用
- **理论→方法的推导链**非常优雅：从推导各 pooling 上界的弱点 → 利用矩阵扰动理论 → 自然导出 RS-Pool
- **谱间隙的双重优势**：$\sigma_1 - \sigma_2$ 大既保证鲁棒性（Theorem 5.1 分母），又保证计算效率（幂迭代快速收敛）
- RS-Pool 是**模型无关**的，可作为即插即用模块替换任何 GNN 的 pooling 层

## 局限与展望
- 理论分析限于 GCN/GIN，对 GAT 等基于注意力的 GNN 未覆盖
- 假设主奇异值无重复（$\sigma_1 \neq \sigma_2$），虽然实践中罕见但理论上不完备
- 仅考虑边扰动（结构攻击），缺少对节点特征攻击和联合攻击的深入分析
- 小图上 SVD 主成分可能不够稳定，论文未充分讨论图规模极小时的行为

## 相关工作与启发
- **vs GNNGuard**: GNNGuard 通过注意力机制在 message-passing 阶段过滤恶意边，属于不同防御层面；RS-Pool 在 pooling 阶段补充防御，两者可组合
- **vs GCORN**: GCORN 通过正交权重正则化稳定 message-passing；RS-Pool 通过谱方法稳定 pooling，互补
- **vs 对抗训练**: Gosch et al. 在扰动图上训练提升鲁棒性，计算代价高；RS-Pool 无需额外训练开销

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性地从 pooling 角度分析图分类鲁棒性，理论+方法都有贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 8个数据集、3类攻击、8种pooling对比，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机→理论→方法→实验环环相扣
- 价值: ⭐⭐⭐⭐ 提供了互补的防御视角和即插即用的方案，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)
- [Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)
- [Improved Balanced Classification with Theoretically Grounded Loss Functions](improved_balanced_classification_with_theoretically_grounded_loss_functions.md)
- [Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)

<!-- RELATED:END -->
