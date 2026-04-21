---
title: >-
  [论文解读] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm
description: >-
  [NeurIPS 2025][可解释聚类] 本文提出首个适用于一般 $\ell_p$ 范数的可解释 k-medians 聚类算法，实现 $\tilde{O}(p(\log k)^{1+1/p-1/p^2})$ 近似比（改进了 p=2 的已知最优界），并给出首个动态版本：在中心集合的插入/删除下，以 $O(d \log^3 k)$ 摊还更新时间和 $O(\log k)$ 重分配次数维护可解释聚类。
tags:
  - NeurIPS 2025
  - 可解释聚类
  - k-medians
  - 决策树
  - 动态算法
  - 近似比
---

# Dynamic Algorithm for Explainable k-medians Clustering under lp Norm

**会议**: NeurIPS 2025  
**arXiv**: [2512.01150](https://arxiv.org/abs/2512.01150)  
**代码**: 无  
**领域**: others  
**关键词**: 可解释聚类, k-medians, 决策树, 动态算法, 近似比

## 一句话总结
本文提出首个适用于一般 $\ell_p$ 范数的可解释 k-medians 聚类算法，实现 $\tilde{O}(p(\log k)^{1+1/p-1/p^2})$ 近似比（改进了 p=2 的已知最优界），并给出首个动态版本：在中心集合的插入/删除下，以 $O(d \log^3 k)$ 摊还更新时间和 $O(\log k)$ 重分配次数维护可解释聚类。

## 研究背景与动机

**领域现状**：可解释聚类由 Dasgupta et al. (2020) 提出——用阈值决策树做聚类划分，每个内部节点对单个特征做阈值比较，使聚类分配透明可追溯。核心问题是可解释性的代价：决策树聚类成本与最优无约束成本的比值能有多好？

**现有痛点**：$\ell_1$ 范数下竞争比已被紧确定为 $1 + H_{k-1}$；$\ell_2$ 下最佳已知为 $\tilde{O}(\log^{3/2} k)$，仍有提升空间；对于一般 $\ell_p$（$p > 2$）完全没有算法；所有算法都是静态的。

**核心矛盾**：更一般的 $\ell_p$ 范数需要更精细的随机切割分布设计——均匀随机切割会导致 $\Theta(k^{1-1/p})$ 竞争比。动态设置下决策树的层次结构使局部更新更具挑战。

**本文目标**：(a) 设计适用于所有有限 $p \geq 1$ 的可解释 k-medians 算法并改进 p=2 的界；(b) 首次给出动态版本。

**切入角度**：使用非均匀随机切割（CDF 为 $x^p/R_t^p$），结合 Poisson 过程时间戳实现动态更新。

**核心 idea**：通过 $\ell_p$ 适配的非均匀随机切割和 Poisson 时间戳驱动的动态维护，首次实现所有 $\ell_p$ 范数的近最优可解释聚类和高效动态更新。

## 方法详解

### 整体框架
算法接收 $k$ 个参考中心 $C$ 作为输入，输出 $k$ 叶阈值决策树 $\mathcal{T}$。核心过程 Partition_Leaf 递归地将每个包含多个中心的节点分裂为子节点。算法与数据点无关——仅依赖参考中心集合。

### 关键设计

1. **非均匀随机切割（Partition_Leaf）**:

    - 功能：将包含中心集合 $C_u$ 的空间递归划分，直到每个叶子至多含一半中心
    - 核心思路：选择锚点 $m^u$（坐标中位数），迭代采样随机坐标 $i$、符号 $\sigma$ 和参数 $Z_t \sim \text{Uniform}[0, R_t^p]$，令 $\theta_t = Z_t^{1/p}$，切割阈值 $\vartheta_t = m^u_i + \sigma \cdot \theta_t$
    - 关键性质：阈值的 CDF 为 $x^p / R_t^p$——靠近锚点的切割密度低，远离的高。确保分离概率与 $\ell_p$ 距离的 p 次方成比例
    - 为什么不用均匀切割：均匀切割对所有距离一视同仁，导致竞争比退化到 $\Theta(k^{1-1/p})$

2. **三类切割分析框架**:

    - 功能：精确分析数据点 $x$ 被错误切割的期望代价
    - 核心分类：
        - **安全切割**：回退距离很小（$A_k M_t \leq R_t / (6^p \log^2 k)$），总贡献 $O(1) \cdot \|x-c\|_p$
        - **轻切割**：半径较小（$R_t \leq 6 \log^\alpha k \cdot \max\{\|x-m^t\|, \|c-m^t\|\}$），贡献 $O(p \log^{1+\alpha} k) \cdot \|x-c\|_p$
        - **重且不安全切割**：用 Holder 不等式界定，贡献 $O((\log k)^{2-1/p-\alpha(p-1)}) \cdot \|x-c\|_p$
    - 取 $\alpha = 1/p - 1/p^2$ 平衡两者，得到递推 $A_k \leq O(p (\log k)^{1+1/p-1/p^2} \log\log k)$

3. **动态算法（Poisson 时间戳）**:

    - 功能：在中心集合发生插入/删除时高效更新决策树
    - 核心思路：为每个随机切割分配 Poisson 过程到达时间 $\rho_t$，时间戳在中心变化时保持不变
        - 插入：找最早将新中心与锚点分离的切割，在树中相应位置插入
        - 删除：定位叶子并移除对应切割
        - 重建：更新次数超过初始中心数 1/4 时触发
    - 复杂度：$O(d \log^3 k)$ 摊还更新时间，$O(\log k)$ 摊还重分配次数

### 理论保证

定理 3.1（静态）：$\mathbf{E}[\text{cost}_p(X, \mathcal{T})] \leq O(p \cdot (\log k)^{1+1/p-1/p^2} \log\log k) \cdot \text{cost}_p(X, C)$

定理 5.1（动态）：相同近似比，摊还更新时间 $O(d \log^3 k)$，摊还重分配次数 $O(\log k)$

## 实验关键数据

### 近似比理论对比

| 范数 | 之前最佳上界 | 本文上界 | 下界 | Gap |
|------|------------|---------|------|-----|
| $\ell_1$ | $1 + H_{k-1}$ (紧) | $O(\log k \log\log k)$ | $\Omega(\log k)$ | $O(\log\log k)$ |
| $\ell_2$ | $\tilde{O}(\log^{3/2} k)$ | $\tilde{O}(\log^{5/4} k)$ | $\Omega(\log k)$ | $\tilde{O}(\log^{1/4} k)$ |
| 一般 $\ell_p$ | 无 | $\tilde{O}(p (\log k)^{1+1/p-1/p^2})$ | $\Omega(\log k)$ | 对数多项式 |

### 动态算法复杂度

| 操作 | 更新时间（摊还） | 重分配次数（摊还） | 近似比保证 |
|------|----------------|------------------|----------|
| 插入中心 | $O(d \log^3 k)$ | $O(\log k)$ | 与静态相同 |
| 删除中心 | $O(d \log^3 k)$ | $O(\log k)$ | 与静态相同 |

### 关键发现
- $\ell_2$ 情况下指数从 3/2 改进到 5/4，通过更精确的三类切割分析实现
- 对数指数 $1 + 1/p - 1/p^2$ 总在 [1, 1.25] 范围内，最大值在 p=2 处取得
- 下界定理证明不存在对所有 p 同时最优的 p-无关算法（存在 $\Omega(d^{1/4})$ 下界）
- 动态算法可与 Bhattacharya et al. (2025) 的动态 k-medians 结合实现全动态可解释聚类

## 亮点与洞察
- **非均匀切割的精妙设计**：CDF 为 $x^p/R_t^p$ 是 $\ell_p$ 几何的自然匹配，使分离概率与距离的 p 次方成比例，是得到对数级竞争比的关键
- **Poisson 时间戳的启发性用法**：原本只用于分析的指数时钟技巧被提升为算法设计工具，时间戳使每个切割具有全局唯一标识
- **从静态到动态的通用桥梁**：先分配时间戳再按时间戳维护的范式可能推广到其他递归随机算法的动态化

## 局限与展望
- $p=1$ 下竞争比为 $O(\log k \log\log k)$，与紧界差 $O(\log\log k)$ 因子
- $p=2$ 下仍有 $\tilde{O}(\log^{1/4} k)$ 的 gap
- 动态算法在重建阈值处需完全重建子树，可能导致偶发高延迟
- 纯理论工作，实际聚类性能的实验评估完全缺失

## 相关工作与启发
- **vs Dasgupta et al. (2020)**: 开创性工作只给 $O(k)$ 上界，本文在所有 $\ell_p$ 下达到对数级
- **vs Gupta et al. (2023)**: $\ell_1$ 紧界 $1 + H_{k-1}$，本文仅差 $O(\log\log k)$ 但覆盖所有 $\ell_p$
- **vs Makarychev & Shan (2021)**: $\ell_2$ 的 $\tilde{O}(\log^{3/2} k)$，本文改进到 $\tilde{O}(\log^{5/4} k)$ 并推广
- **vs Bhattacharya et al. (2025)**: 动态无约束 k-medians，本文的动态可解释聚类可直接叠加其上

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个一般 $\ell_p$ 可解释聚类算法加首个动态版本，理论贡献重大
- 实验充分度: ⭐⭐ 纯理论工作无实验验证
- 写作质量: ⭐⭐⭐⭐ 理论推导缜密，三类切割分析框架清晰
- 价值: ⭐⭐⭐⭐ 对可解释 AI 理论基础贡献显著，动态化对流数据场景有实际意义

<!-- RELATED:START -->

## 相关论文

- [SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)
- [Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [Empowering Decision Trees via Shape Function Branching](empowering_decision_trees_via_shape_function_branching.md)
- [The Anatomy of Evidence: An Investigation Into Explainable ICD Coding](../../ACL2025/interpretability/the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding.md)

<!-- RELATED:END -->
