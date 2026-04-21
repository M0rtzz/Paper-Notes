---
title: >-
  [论文解读] t-SNE Exaggerates Clusters, Provably
description: >-
  [ICLR 2026][t-SNE] 从理论上严格证明 t-SNE 存在两个根本性失败模式：（1）无法从输出推断输入聚类的强度，（2）无法忠实地展示极端离群点——即使输入毫无聚类结构或存在极端离群点，t-SNE 也可能产生完美聚类的可视化。
tags:
  - ICLR 2026
  - t-SNE
  - 聚类夸大
  - 降维
  - 可视化误导
  - 离群点
---

# t-SNE Exaggerates Clusters, Provably

**会议**: ICLR 2026  
**arXiv**: [2510.07746](https://arxiv.org/abs/2510.07746)  
**代码**: [https://github.com/njbergam/tsne-exaggerates-clusters](https://github.com/njbergam/tsne-exaggerates-clusters)  
**领域**: 数据可视化 / 理论分析  
**关键词**: t-SNE, 聚类夸大, 降维, 可视化误导, 离群点

## 一句话总结

从理论上严格证明 t-SNE 存在两个根本性失败模式：（1）无法从输出推断输入聚类的强度，（2）无法忠实地展示极端离群点——即使输入毫无聚类结构或存在极端离群点，t-SNE 也可能产生完美聚类的可视化。

## 研究背景与动机

- **t-SNE 的广泛使用**：t-SNE 是探索性数据分析的标配工具，广泛用于单细胞基因组学、语言模型可解释性等领域
- **现有理论**：已证明 t-SNE 对良好分离的聚类输入能产生保持聚类结构的输出（真阳性保证）
- **理论空白**：关于假阳性（无聚类输入但输出有聚类）和假阴性（有聚类输入但输出无聚类）的理论分析一直缺失
- **科学影响**：t-SNE 输出直接影响假设生成、实验设计和科学结论

## 方法详解

### t-SNE 形式化

输入亲和度矩阵 $P$ 基于高斯核构建：
$$P_{j|i}(X; \sigma_i) := \frac{\exp(-\|x_j - x_i\|^2 / (2\sigma_i^2))}{\sum_{k \neq i} \exp(-\|x_k - x_i\|^2 / (2\sigma_i^2))}$$

输出亲和度矩阵 $Q$ 基于 t-分布：
$$Q_{ij}(Y) := \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k,l; k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

目标：最小化 $\mathcal{L}_X(Y) := \text{KL}(P(X) \| Q(Y))$

### 核心发现 1：聚类强度不可推断

**定理 3**（不同输入，相同输出）：对任意 $0 < \epsilon \leq 1$，存在数据集 $X_\epsilon$，使得：
$$\bar{\mathcal{S}}(X_\epsilon; C_{m \in [k]}) = \epsilon \cdot \bar{\mathcal{S}}(X; C_{m \in [k]})$$
但对任意 perplexity $\rho$：
$$\text{t-SNE}_\rho(X) = \text{t-SNE}_\rho(X_\epsilon)$$

即任意弱聚类的"冒名"数据集可产生与强聚类数据集完全相同的 t-SNE 输出！

**推论 4**：对任意均分的两类数据，存在一系列轮廓系数从 $\epsilon$ 到 1 的数据集，它们共享**完全相同的 t-SNE 驻点集合**。

### 核心发现 2：微小扰动导致巨大变化

**定理 5**：对任意 $\epsilon > 0$，存在数据集 $X, X'$，使得所有点对距离的比值在 $[1-\epsilon, 1+\epsilon]$ 内（即距离几乎相同），但 t-SNE 输出完全不同。

**引理 6**（惊人结论）：任意 $\epsilon > 0$ 近似正则单纯形的数据集集合 $\Delta_\epsilon$，就能生成**所有可能的 t-SNE 驻点输出**！

### 关键机制：加法不变性

t-SNE 对输入的平方距离不仅具有乘法尺度不变性，还具有**加法平移不变性**。即若 $\|x'_i - x'_j\|^2 = \|x_i - x_j\|^2 + C$，则 $\text{t-SNE}_\rho(X) = \text{t-SNE}_\rho(X')$。这是上述定理的核心原因。

### 核心发现 3：离群点被压制

**定理 9**：对**任何** t-SNE 输出 $Y$，离群数 $\alpha(Y) \leq 3.266 + o_n(1)$。

无论输入中离群点多极端，t-SNE 都无法在输出中表现出超过约 3.6 的离群程度。原因在于输入/输出亲和度矩阵的不对称性。

### 单点毒化攻击

仅添加一个"毒化点"（放在数据均值位置）就能破坏整个聚类可视化结构。在高维数据中尤为严重——毒化点成为大多数点的最近邻，极大改变亲和度矩阵。

## 实验验证

### 冒名数据集实验

| 度量 | 原始 PBMC3k | 冒名数据集 |
|------|-----------|-----------|
| t-SNE 可视化 | 清晰聚类 | **几乎相同的聚类** |
| 轮廓系数 | 高（原始） | **极低** |
| 最近邻排序 | 正常 | **保持不变** |

### 毒化攻击实验

- 400 点 × 2000 维高斯混合 → 添加 1 个毒化点 → 聚类结构**完全消失**
- BBC 新闻数据集：注入 10% 毒化点 → 轮廓系数**减半**
- 相比之下：注入 50% 的离群点对聚类结构**几乎无影响**

### 离群点实验

| 数据集 | t-SNE 中的 α | PCA 中的 α |
|--------|-------------|-----------|
| 金融欺诈数据 | ~0.2 | 保持分离 |
| 高斯+离群点 | ~0.1 | 忠实还原 |

## 亮点与洞察

1. **首个 t-SNE 失败模式的理论分析**：此前仅有经验观察，本文提供严格证明
2. **加法不变性的发现**：揭示了 t-SNE 误导行为的根本原因
3. **实用启示**：
    - 不能从 t-SNE 的聚类可视化推断输入的聚类强度
    - t-SNE 不适合离群点检测
    - 高维数据上 t-SNE 尤其不稳定（因接近正则单纯形）
4. **PCA 作为互补**：在离群点检测和稳定性方面 PCA 显著优于 t-SNE

## 局限性

- 理论结果基于驻点分析，实际 t-SNE 输出依赖优化路径（可能避开某些驻点）
- 贡献偏数学理论，算法改进建议较少
- 主要关注 t-SNE，对 UMAP 等方法仅做了初步实验

## 相关工作

- **t-SNE 理论**：Arora et al. 2018（聚类保持保证）、Cai & Ma 2022（优化阶段分析）
- **t-SNE 批评**：Chari & Pachter 2023（不可靠的探索性分析工具）
- **一般降维理论**：Snoeck et al. 2026（任何常数维嵌入必然失真）

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首个严格的 t-SNE 失败模式理论分析
- **技术深度**: ⭐⭐⭐⭐⭐ — 证明精美，加法不变性的发现深刻
- **实验充分性**: ⭐⭐⭐⭐ — 理论与实验紧密结合
- **实用价值**: ⭐⭐⭐⭐ — 对实际使用 t-SNE 的科研人员有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](../../ICML2025/others/provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)
- [Optimizer Choice Matters for the Emergence of Neural Collapse](optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)
- [Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)
- [Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)

<!-- RELATED:END -->
