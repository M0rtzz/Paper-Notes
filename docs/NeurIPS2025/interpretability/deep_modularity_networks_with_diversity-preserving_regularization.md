---
title: >-
  [论文解读] Deep Modularity Networks with Diversity-Preserving Regularization
description: >-
  [NEURIPS2025][graph clustering] 在 Deep Modularity Networks (DMoN) 基础上引入三项多样性保持正则化（距离、方差、熵），显式促进特征空间中的簇间分离和分配多样性，在特征丰富的图数据集上显著提升聚类质量。
tags:
  - NEURIPS2025
  - graph clustering
  - modularity maximization
  - diversity regularization
  - 图神经网络
---

# Deep Modularity Networks with Diversity-Preserving Regularization

**会议**: NEURIPS2025  
**arXiv**: [2501.13451](https://arxiv.org/abs/2501.13451)  
**代码**: [YasminSalehi/DMoN-DPR](https://github.com/YasminSalehi/DMoN-DPR)  
**领域**: self_supervised  
**关键词**: graph clustering, modularity maximization, diversity regularization, GNN pooling  

## 一句话总结

在 Deep Modularity Networks (DMoN) 基础上引入三项多样性保持正则化（距离、方差、熵），显式促进特征空间中的簇间分离和分配多样性，在特征丰富的图数据集上显著提升聚类质量。

## 背景与动机

图聚类是图表示学习的核心问题，广泛应用于社交网络社区发现、生物网络功能模块识别等场景。近年来基于 GNN 的图池化方法（DiffPool、MinCutPool 等）取得了进展，但存在计算开销大或收敛困难的问题。DMoN 通过将谱模块度最大化与坍缩正则化结合，在端到端框架中实现了较好的社区检测效果。

然而 DMoN 的优化目标存在两个关键缺陷：

1. **缺乏特征空间分离机制**：目标函数不包含任何直接奖励簇间特征分离的项，导致结构上不同的簇在特征空间中可能高度重叠
2. **缺乏分配置信度控制**：没有显式的熵/温度控制机制，软分配可能过早硬化，影响探索阶段的簇平衡

## 核心问题

如何在 DMoN 的模块度最大化框架中引入显式的多样性保持机制，使聚类结果不仅在图结构上有意义，还能在特征空间中实现良好的簇间分离和分配多样性？

## 方法详解

### 基础框架：DMoN 回顾

DMoN 使用 GCN 编码器生成软分配矩阵 $C = \text{softmax}(\text{GCN}(\tilde{A}, X))$，优化目标包含模块度项和坍缩正则化项：

$$L_{\text{DMoN}} = -\frac{1}{2m}\text{Tr}(C^\top B C) + \frac{\sqrt{k}}{n}\left\|\sum_i C_i^\top\right\|_F - 1$$

其中 $B = A - \frac{dd^\top}{2m}$ 为模块度矩阵。

### DMoN-DPR：三项多样性保持正则化

在 DMoN 基础上增加三个正则化项：

$$L_{\text{DMoN-DPR}} = L_{\text{DMoN}} + W_{\text{dist}} L^{\text{distance}} + W_{\text{var}} L^{\text{variance}} + W_{\text{entropy}} L^{\text{entropy}}$$

#### 1. 距离正则化（Distance-Based）

受 SimCLR 对比学习启发，惩罚特征空间中距离过近的簇质心对：

$$L^{\text{distance}} = \frac{1}{k(k-1)} \sum_{i=1}^{k} \sum_{j \neq i}^{k} \text{ReLU}(\epsilon - \|\mu_i - \mu_j\|_2^2)$$

其中 $\mu_i$ 为簇 $i$ 的加权质心，$\epsilon$ 为最小距离阈值。当两个质心距离小于 $\epsilon$ 时产生惩罚，将簇推离彼此。

#### 2. 方差正则化（Variance-Based）

最大化每个簇的分配概率在所有节点上的方差，防止均匀分配：

$$L^{\text{variance}} = -\frac{1}{k} \sum_{i=1}^{k} \text{Var}(C_{:i})$$

高方差意味着簇对不同节点有明确的偏好，实现分配的专业化。

#### 3. 熵正则化（Entropy-Based）

以较小的正权重最小化每个节点分配的 Shannon 熵：

$$L^{\text{entropy}} = -\frac{1}{n} \sum_{v=1}^{n} \sum_{i=1}^{k} C_{vi} \log(C_{vi} + \delta)$$

使用小权重（0.001–0.1）确保熵缓慢下降，避免过早硬化分配，在训练早期保持较高不确定性以支持探索。

### 设计直觉

- **D 项**推动簇质心在特征空间中互相远离
- **V 项**确保每个簇的分配概率分布有足够的区分度
- **E 项**温和地引导分配逐渐变得更确定，但不破坏探索
- 三者组合实现了"簇间分离 + 簇内专注 + 渐进确信"的协同效果

## 实验关键数据

在 5 个基准数据集上与 DiffPool、MinCutPool、DMoN 对比，使用 10 个随机种子取平均：

**特征稀疏数据集**（Cora / CiteSeer / PubMed）：

- DPR 变体在 NMI 和 F1 上小幅领先 DMoN（如 Cora 上 DPR(DV) NMI 44.40% vs DMoN 43.92%），但改进未达统计显著（$p > 0.10$）
- 图结构指标（Conductance、Modularity）几乎不受影响

**特征丰富数据集**（Coauthor CS / Coauthor Physics）——改进显著：

| 方法 | CS NMI↑ | CS F1↑ | Physics NMI↑ | Physics F1↑ |
|------|---------|--------|-------------|-------------|
| DMoN | 69.26% | 59.26% | 53.50% | 47.51% |
| DPR(DVE) | 71.28% | **62.67%** | 53.50% | **57.96%** |
| DPR(E) | **71.58%** | 61.33% | 52.83% | 51.09% |
| DPR(DV) | 70.72% | 61.35% | **55.84%** | 57.99% |

- Coauthor CS 上 F1 提升超 3 个百分点，Physics 上 F1 提升超 10 个百分点
- 配对 t 检验确认 NMI 和 F1 改进在 $p \leq 0.05$ 水平显著

## 亮点

1. **简洁有效的正则化设计**：三个正则项各有明确直觉，即插即用地增强 DMoN，无需修改模型架构
2. **理论与实践一致**：在特征丰富数据集上的显著提升验证了"特征空间多样性能改善聚类"的假设
3. **不损害结构指标**：添加正则化几乎不影响 Conductance 和 Modularity，说明方法兼顾了图结构与特征空间
4. **严格的统计验证**：使用配对双尾 t 检验在相同种子下对比，比单纯报告均值更有说服力

## 局限与展望

1. **对特征稀疏图效果有限**：在 Cora、CiteSeer、PubMed 上改进不显著，说明方法依赖于节点特征的丰富性
2. **超参数敏感性**：三个权重 $W_{\text{dist}}, W_{\text{var}}, W_{\text{entropy}}$ 和阈值 $\epsilon$ 需要针对每个数据集调优
3. **可扩展性未验证**：最大数据集仅约 34K 节点（Coauthor Physics），在大规模图上的表现未知
4. **簇数 $k$ 固定**：需要预设簇数，未探索自适应确定簇数的方案
5. **距离正则化的质心计算**依赖软分配加权，当分配接近均匀时质心区分度低

## 与相关工作的对比

| 方法 | 特点 | 局限 |
|------|------|------|
| DiffPool | 端到端学习软分配 | 二次计算开销，大图上不稳定（PubMed 坍缩） |
| MinCutPool | 归一化割 + 正交约束 | 可能阻碍收敛 |
| DMoN | 模块度最大化 + 坍缩正则化 | 无特征空间分离机制 |
| **DMoN-DPR** | DMoN + 距离/方差/熵正则化 | 超参需调优，对特征稀疏图提升有限 |

## 启发与关联

- **多样性正则化的通用性**：距离和方差正则化的思路可推广到其他需要簇/表示多样性的场景，如多样化推荐、集成学习中的基学习器多样性
- **特征丰富度作为方法选择依据**：论文揭示了一个实用原则——当节点特征丰富且异质时，值得投入额外正则化来利用特征信息；反之则可能得不偿失
- **与对比学习的联系**：距离正则化与对比学习中的 pushing apart 机制异曲同工，暗示图聚类可能受益于更深入地融合对比学习损失

## 评分

- 新颖性: 3/5（正则化设计合理但各项技术本身不新颖）
- 实验充分度: 4/5（多数据集、统计检验、消融实验齐全）
- 写作质量: 4/5（结构清晰，动机阐述充分）
- 价值: 3/5（特征丰富图上有效，但适用场景受限）
