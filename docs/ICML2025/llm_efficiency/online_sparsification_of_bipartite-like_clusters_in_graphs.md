---
title: >-
  [论文解读] Online Sparsification of Bipartite-Like Clusters in Graphs
description: >-
  [ICML 2025][LLM效率][图稀疏化] 提出了一种**近线性时间的在线图稀疏化算法**，能在保留图的二部图式聚类（bipartite-like clusters）结构的前提下，将边数压缩到 $\widetilde{O}(n)$，同时适用于无向图和有向图，显著加速现有聚类算法。
tags:
  - ICML 2025
  - LLM效率
  - 图稀疏化
  - 二部图聚类
  - 在线算法
  - 谱方法
  - 有向图
---

# Online Sparsification of Bipartite-Like Clusters in Graphs

**会议**: ICML 2025  
**arXiv**: [2508.05437](https://arxiv.org/abs/2508.05437)  
**代码**: [GitHub](https://github.com/suranjande4/Online-Sparsification-of-Bipartite-Like-Clusters-in-Graphs)  
**领域**: 图算法/图聚类  
**关键词**: 图稀疏化, 二部图聚类, 在线算法, 谱方法, 有向图

## 一句话总结

提出了一种**近线性时间的在线图稀疏化算法**，能在保留图的二部图式聚类（bipartite-like clusters）结构的前提下，将边数压缩到 $\widetilde{O}(n)$，同时适用于无向图和有向图，显著加速现有聚类算法。

## 研究背景与动机

### 经典图聚类的局限

传统图聚类关注的是找到**低导出度（low conductance）**的顶点集——即集合内部边密集、与外部连接稀疏。然而许多现实场景中，更关键的是挖掘**两组顶点之间的密集互连关系**，例如：

- **迁移/贸易网络**：区域间的人口流动或贸易往来形成二部图式结构
- **金融市场**：不同类型资产间的领先-滞后关系
- **社交网络**：交通和社交图中的链式结构

这类结构被称为 **bipartite-like cluster**——一对不相交的顶点集 $(A, B)$ 之间的边占据 $A \cup B$ 中绝大多数边。

### 现有算法的瓶颈

现有的聚类算法（如 Macgregor & Sun 2021 的局部算法）在大规模图上运行开销大。而传统的**谱稀疏化**只保留 $w(S, V \setminus S)$ 的切割值，**不保证保留** $w(A, B)$ 这类跨两个子集的切割值。此外，大多数稀疏化算法依赖 Laplacian 求解器，实现复杂，且只适用于无向图。

### 本文动机

能否设计一种**简单、在线、近线性时间**的稀疏化算法，既保留二部图式聚类结构，又能直接作为现有算法的加速子程序？

## 方法详解

### 整体框架

本文核心分两大部分：

1. **无向图稀疏化**（Theorem 1）：基于边采样的在线算法，直接在无向图上保留 $k$ 个 bipartite-like clusters
2. **有向图稀疏化**（Theorem 2）：通过"半双覆盖"归约将有向图问题转化为无向图问题，再反向构造稀疏有向图

### 关键定义

#### Dual Conductance

对于无向图 $G=(V,E)$ 和两个不相交子集 $A, B \subset V$，定义**双向导出度**：

$$\overline{\phi}_G(A,B) \triangleq \frac{2 w_G(A,B)}{\text{vol}_G(A \cup B)}$$

$\overline{\phi}_G(A,B)$ 越大，说明 $A$ 和 $B$ 之间的边占比越高，$(A,B)$ 构成一个越强的二部图式聚类。

#### k-way Dual Cheeger 常数

$$\bar{\rho}_G(k) \triangleq \max_{(A_1,B_1),\ldots,(A_k,B_k)} \min_{1 \le i \le k} \overline{\phi}_G(A_i, B_i)$$

量化了图 $G$ 中 $k$ 个二部图式聚类的强度，$\bar{\rho}_G(k)$ 越高说明 $G$ 拥有越明显的 $k$ 个二部图式结构。

### 关键设计

#### 无向图稀疏化算法

**算法思想**：以精心设计的采样概率对每条边进行独立采样，被采样的边以逆概率重加权。

对于每条边 $e = \{u,v\}$，定义采样概率：

$$p_e = p_u(v) + p_v(u) - p_u(v) \cdot p_v(u)$$

其中：

$$p_u(v) = \min\left\{\frac{w_G(u,v) \cdot C \cdot \log^3 n}{d_G(u) \cdot (2 - \lambda_{n-k})}, 1\right\}$$

- $C$ 是常数，$\lambda_{n-k}$ 是归一化 Laplacian $\mathcal{L}_G$ 的第 $(n-k)$ 个特征值
- $(2 - \lambda_{n-k})$ 越小（即 $\lambda_{n-k}$ 越接近 2），说明二部图式结构越明显，采样概率越高
- 被采样的边权重重设为 $w_{G^*}(e) = w_G(e) / p_e$（无偏估计）

**关键保证**（Theorem 5）：

- 输出图 $G^*$ 仅含 $O\left(\frac{n \cdot \log^3 n}{2-\lambda_{n-k}}\right)$ 条边
- $\bar{\rho}_{G^*}(k) = \Omega(\bar{\rho}_G(k))$ — 聚类质量保持
- $\lambda_{k+1}(\mathcal{J}_{G^*}) = \Theta(\lambda_{k+1}(\mathcal{J}_G))$ — 聚类数量不变

**证明核心**：利用 Chebyshev 不等式证明切割值 $w_H(A_i, B_i) = \Omega(w_G(A_i, B_i))$ 以高概率成立，再结合 union bound 推广到所有 $k$ 个聚类。

#### 有向图稀疏化（三步归约）

有向图的难点在于：自然的矩阵表示会产生复数特征值，且不存在有向图版本的高阶 dual Cheeger 不等式。本文通过三步归约巧妙解决：

**Step 1：半双覆盖（Semi-Double Cover）**

将有向图 $\overrightarrow{G}$ 转化为无向二部图 $H$：
- 每个顶点 $v$ 复制为 $v_1, v_2$
- 每条有向边 $(u,v)$ 变为无向边 $\{u_1, v_2\}$

关键引理（Lemma 7）：$f_{\overrightarrow{G}}(A,B) = \phi_H(A_1 \cup B_2)$，即有向图的 flow ratio 等价于无向覆盖图中的 conductance。

**Step 2：稀疏化无向图 $H$**

利用已有的聚类保持稀疏化算法（Lemma 9, Sun & Zanetti 2019）将 $H$ 稀疏化为 $H^*$，保持 $k$ 个低 conductance 集合。

**Step 3：反向半双覆盖（Reverse Semi-Double Cover）**

从 $H^*$ 反向构造有向图 $\overrightarrow{G^*}$：
- $H^*$ 中每对 $(u_1, u_2)$ 合并回顶点 $v$
- 边 $\{u_1, v_2\} \in H^*$ 恢复为有向边 $(u,v)$

核心难点：反向构造时，$H^*$ 中的最优 $k$-way partition 不一定是 simple set。论文证明了对任意（非 simple）集合 $S$，存在 simple 集合 $T$ 使得 $\phi_H(S) \ge (1-c) \cdot \phi_H(T)$，从而建立不等式链：

$$1 - \frac{1}{1-c} \cdot \rho_{H^*}(k) \le \bar{\rho}_{\overrightarrow{G^*}}(k) \le 1 - \rho_{H^*}(k)$$

### 损失函数 / 训练策略

本文是理论算法工作，不涉及传统的损失函数或训练策略。其核心优化目标是：

- **最小化边数**：输出 $\widetilde{O}(n)$ 条边（相比输入的 $m$ 条）
- **最大化结构保持**：$\bar{\rho}_{G^*}(k) = \Omega(\bar{\rho}_G(k))$
- **在线处理**：算法只需 degree oracle，可以边探索边稀疏化

与基于 Laplacian 求解器的方法不同，本文**仅使用随机采样**，实现简单且高效。

## 实验关键数据

### 主实验

#### 无向图 - 合成数据（SBM, p=0.3, q=0.1p）

| 顶点数（每分区） | MS 运行时间 | 本文运行时间 | MS Bipartiteness | 本文 Bipartiteness |
|---|---|---|---|---|
| 500 | ~0.3s | ~0.05s | ~0.18 | ~0.18 |
| 1500 | ~2.5s | ~0.3s | ~0.18 | ~0.18 |
| 2500 | ~8s | ~0.5s | ~0.18 | ~0.18 |

#### 无向图 - 真实数据（Militarised Interstate Disputes）

| 种子国家 | MS 时间 | 本文时间 | MS Bipartiteness | 本文 Bipartiteness |
|---|---|---|---|---|
| USA | 0.034s | **0.0044s** (7.7×) | 0.292 | 0.285 |
| Netherlands | 0.0351s | **0.0042s** (8.4×) | 0.307 | 0.281 |
| Lithuania | 0.0336s | **0.0043s** (7.8×) | 0.303 | 0.165 |

#### 有向图 - 真实数据（US Migration）

| 种子县 | 目标 $\phi$ | ECD 时间 | 本文时间 | ECD Flow-ratio | 本文 Flow-ratio |
|---|---|---|---|---|---|
| Maricopa County | 0.2 | 20.66s | **13.43s** | 0.414 | 0.417 |
| Virginia Beach City | 0.2 | 15.31s | **12.29s** | 0.546 | 0.621 |
| Kanawha County | 0.2 | 9.32s | **8.48s** | 0.330 | 0.330 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| $\eta=0.7$（弱二部图）| 加速约 1.3-2× | η 越低二部图结构越弱，加速倍数相对小 |
| $\eta=0.8$（中等结构）| 加速约 2-3× | 结构更明显，稀疏化效果更好 |
| $\eta=0.9$（强二部图）| 加速约 3-5× | 强结构下采样概率更集中，效果最显著 |
| 顶点 500→2500 | 加速随规模增大 | 稀疏化的优势在大图上更明显 |

### 关键发现

1. **加速显著**：在无向图合成数据上，随顶点数增加，本文算法相比 MS 的加速倍数从 ~6× 增长到 ~16×
2. **质量保持**：bipartiteness ratio 和 flow-ratio 与原始算法几乎一致
3. **η 越大效果越好**：二部图结构越强，稀疏化保留信息越多，加速越显著
4. **真实数据验证**：在军事冲突、人口迁移等多领域数据上均有效

## 亮点与洞察

1. **采样概率设计精妙**：$p_u(v)$ 与 $w(u,v)/d_G(u)$ 成正比，本质上是对"边在其端点看来的重要性"进行加权——端点度数大则采样概率低，避免高度节点的边被过度保留
2. **$(2-\lambda_{n-k})$ 的自适应性**：这个因子自动适应图的二部图式结构强度，结构越强采样越激进，非常优雅
3. **半双覆盖归约**：将有向图问题完美映射为无向图问题，是本文最具创意的技术贡献。这一归约允许复用无向图的谱理论工具
4. **在线特性**：算法只需 degree oracle，可以边探索图边稀疏化，天然适合与局部算法结合
5. **实现简洁**：相比依赖 Laplacian 求解器的稀疏化方法，本文仅用随机采样，工程友好

## 局限与展望

1. **需要知道 $\lambda_{n-k}$**：采样概率依赖特征值 $(2-\lambda_{n-k})$，实际中通常以 $O(\log^c n)$ 近似，但精确估计可能影响效果
2. **$k$ 需预先给定**：算法假设 $k$ 已知，自动确定 $k$ 是开放问题
3. **稀疏化比率有限**：对于二部图结构不明显的图（$\bar{\rho}_G(k)$ 接近 0），稀疏化空间有限
4. **实验规模偏小**：最大测试规模为 5000 顶点，百万/十亿级图上的表现有待验证
5. **有向图条件较强**：Theorem 2 要求 $\bar{\rho}_{\overrightarrow{G}}(k) = 1 - o(1/k)$，即需要非常强的二部图结构
6. **仅保留二部图式聚类**：不保证保留其他类型的聚类结构（如低 conductance 社区）

## 相关工作与启发

- **Trevisan (2009)**, **Soto (2015)**：用谱方法找单个 bipartite-like cluster，本文推广到 $k$ 个
- **Macgregor & Sun (2021a)**：提出 LocBipartDC 和 ECD 局部算法，本文直接作为其加速模块
- **Sun & Zanetti (2019)**：cluster-preserving 稀疏化，本文在此基础上扩展到 dual Cheeger 设定
- **Cucuringu et al. (2020)**：Hermitian 矩阵谱聚类有向图，本文提供了更简洁的半双覆盖归约视角
- **Liu (2015)**：高阶 dual Cheeger 不等式，是本文理论基础的关键工具
- **启发**：这种"归约-稀疏化-反向构造"的三步范式可能推广到其他类型的结构化聚类问题

## 评分

| 维度 | 评分 (1-5) | 说明 |
|---|---|---|
| 新颖性 | ⭐⭐⭐⭐ | 半双覆盖归约+在线稀疏化的组合非常新颖 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 严格的理论保证，完整的证明链 |
| 实验充分度 | ⭐⭐⭐ | 合成+真实数据均有，但规模偏小 |
| 实用性 | ⭐⭐⭐⭐ | 实现简单，可直接加速现有算法 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，定义和定理陈述严谨 |
| **总评** | **⭐⭐⭐⭐** | 扎实的理论工作，有明确的实用价值 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/llm_efficiency/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](../../CVPR2026/llm_efficiency/learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)
- [Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity](../../AAAI2026/llm_efficiency/connectivity-guided_sparsification_of_2-fwl_gnns_preserving_full_expressivity_wi.md)

<!-- RELATED:END -->
