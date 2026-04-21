---
title: >-
  [论文解读] Graph Alignment via Birkhoff Relaxation
description: >-
  [NeurIPS 2025][graph alignment] 首次为 Birkhoff 松弛（QAP 的紧凸松弛）在高斯 Wigner 模型下建立理论保证：当噪声 $\sigma = o(n^{-1})$ 时松弛解近似真实排列，$\sigma = \Omega(n^{-0.5})$ 时松弛解远离真实排列，揭示了相变现象。
tags:
  - NeurIPS 2025
  - graph alignment
  - quadratic assignment
  - Birkhoff polytope
  - convex relaxation
  - phase transition
  - Gaussian Wigner Model
---

# Graph Alignment via Birkhoff Relaxation

**会议**: NeurIPS 2025  
**arXiv**: [2503.05323](https://arxiv.org/abs/2503.05323)  
**作者**: Sushil Mahavir Varma (University of Michigan), Irène Waldspurger (CNRS/INRIA, Université Paris Dauphine), Laurent Massoulié (INRIA, PSL Research University)
**代码**: [GitHub](https://github.com/smv30/convex_rel_for_graph_alignment)  
**领域**: 图匹配 / 组合优化 / 凸松弛

---

## 一句话总结

本文首次为图对齐问题的 Birkhoff 松弛（将排列矩阵约束松弛为双随机矩阵约束）提供了理论保证，在高斯 Wigner 模型下证明了最优解的相变行为：当噪声 $\sigma = o(n^{-1})$ 时松弛解接近真实排列，当 $\sigma = \Omega(n^{-0.5})$ 时松弛解远离真实排列。

---

## 研究背景与动机

### 问题定义

图对齐（Graph Alignment）问题的目标是找到两个图之间的顶点映射，使边重叠最大化。形式化为二次分配问题（QAP）：

$$\Pi^{\star} = \arg\min_{X \in \mathcal{P}_n} \|AX - XB\|_F^2$$

其中 $\mathcal{P}_n$ 是 $n \times n$ 排列矩阵集合，$A, B$ 是两个图的（加权）邻接矩阵。QAP 在最坏情况下是 NP-hard 的，甚至近似求解也困难。

### 现有方法局限

图对齐在网络去匿名化、计算生物学、模式识别等领域有广泛应用。现有凸松弛方法包括：

**GRAMPA**（Fan et al., 2023）：进一步松弛 Birkhoff 约束并添加二次正则化项，理论上在 $\sigma = O(1/\log n)$ 时成功，但需要调参 $\eta$，且实证表现不如 Birkhoff 松弛

**Simplex 松弛**（Araya & Tyagi, 2024）：保留非负约束但放松行/列和约束，仅在 $\sigma = 0$（无噪声）时有理论保证

**谱方法 EIG1**（Ganassali et al., 2022）：噪声阈值仅为 $\sigma = \Theta(n^{-7/6})$

### 核心动机

Birkhoff 松弛在实践中表现最优，但缺乏理论分析。本文填补这一空白，将排列矩阵集合 $\mathcal{P}_n$ 替换为其凸包——Birkhoff 多面体 $\mathcal{B}_n$（所有双随机矩阵的集合）：

$$X^{\star} = \arg\min_{X \in \mathcal{B}_n} \|AX - XB\|_F^2$$

---

## 方法详解

### 高斯 Wigner 模型

输入矩阵 $A, B$ 来自相关的高斯正交系综（GOE）：$B = (\tilde{\Pi}^{\star})^T (A + \sigma Z) \tilde{\Pi}^{\star}$，其中 $A, Z$ 是独立的 GOE 矩阵，$\sigma$ 控制噪声强度。相关性为 $1/\sqrt{1+\sigma^2}$。

### 主定理：相变行为（Theorem 1）

对任意固定 $\epsilon > 0$，当 $n$ 足够大时，以概率 $1-o(1)$：

- **Well-Separation**：当 $\sigma \geq n^{-0.5+\epsilon}$ 时，$\|X^{\star} - \Pi^{\star}\|_F^2 \geq \delta n$，即松弛解远离真实排列
- **Small-Perturbation**：当 $\sigma \leq n^{-1-\epsilon}$ 时，$\|X^{\star} - \Pi^{\star}\|_F^2 \leq 10 n^{1-\epsilon/32}$，即松弛解是真实排列的小扰动

### 推论：简单取整即可恢复

当 $\sigma = o(n^{-1})$ 时，无论是逐行取最大值（$\hat{\pi}(i) = \arg\max_j X_{ij}^{\star}$）还是 Hungarian 投影，都能正确匹配 $1-o(1)$ 比例的顶点。

### 证明核心技术

**Part I（Well-Separation）的证明思路**：
1. 用 $J/n$（均匀矩阵）的目标值上界 Birkhoff 松弛的最优值：$\|AX^{\star} - X^{\star}B\|_F^2 \leq 9n^{\epsilon}$
2. 构造下界为 $\|X^{\star} - I\|_F$ 的函数，利用 $\|Z\|_F^2 \geq n/2$ 和交换子 $\max_{i \neq j}|(AZ-ZA)_{ij}| \leq 8n^{\epsilon/2-0.5}$ 的集中不等式
3. 结合上下界推出 $\|I - X^{\star}\|_F \geq \sqrt{n}/(16c)$

**Part II（Small-Perturbation）的证明思路**：
1. 考虑 $\sigma=0$ 时的对偶问题，构造对偶可行解 $(R, M, \mu, \tilde{\mu})$
2. 关键设置 $R = J - I$（全1矩阵减单位矩阵），利用 GOE 矩阵的特征值/特征向量性质构造 $M$ 和 $\mu$
3. 通过 Lemma 6 建立 off-diagonal 和 与 $\|AX - XA\|_F$ 的关系：$\sum_{j \neq i} X_{ij} \leq 2n^{3/2+7\epsilon/8}\|AX-XA\|_F + 4n^{1-\epsilon/32}$
4. 通过 Lemma 7 证明 $\|AX^{\star} - X^{\star}A\|_F \leq c\sigma\sqrt{n}$
5. 对特征值间距的精细分析（Claim 10）：$\sum_{i \neq j} \frac{1}{(|\lambda_i - \lambda_j| + n^{-1-\epsilon})^2} \leq n^{3+3\epsilon/2}$

### Population 版本的直觉

当用目标函数的期望替代样本时，最优解有闭式：$\bar{X}^{\star} = \epsilon I + \frac{1-\epsilon}{n}J$，其中 $\epsilon = \frac{2}{2+\sigma^2(n+1)}$。这表明相变阈值在 $\sigma \sim n^{-0.5}$ 处，暗示当前 $\sigma = o(n^{-1})$ 的充分条件可能还不够紧。

---

## 实验关键数据

### 表1：不同凸松弛方法的噪声阈值对比

| 方法 | 算法类型 | 噪声阈值 |
|------|----------|----------|
| EIG1 (Ganassali et al.) | 顶特征向量对齐 | $\sigma = \Theta(n^{-7/6})$ |
| GRAMPA (Fan et al.) | 正则化凸松弛 | $\sigma = O(1/\log n)$ |
| Simplex (Araya & Tyagi) | Simplex 松弛 | $\sigma = 0$ |
| **Birkhoff（本文）** | Birkhoff 松弛 | $\sigma = O(n^{-1})$ |

> GRAMPA 虽然理论阈值最宽松，但需调参 $\eta$，实际表现不如 Birkhoff。

### 表2：实验结果汇总（$n=400$, 10次平均）

| $\sigma$ 范围 | Birkhoff | GRAMPA | Simplex |
|---------------|----------|--------|---------|
| $0 \sim 0.2$ | 100% 匹配 | 开始退化 | 100% 匹配 |
| $0.3 \sim 0.5$ | 100% 匹配 | 大幅退化 | 退化 |
| $0.5 \sim 0.7$ | 开始退化 | 接近失败 | 接近失败 |
| $0.8 \sim 1.0$ | 退化 | 失败 | 失败 |

> Birkhoff 松弛在 $\sigma$ 高达 0.5 时仍能精确匹配所有顶点，远超 GRAMPA（$\sigma \approx 0.2$ 开始退化）和 Simplex（$\sigma \approx 0.4$ 开始退化）。

**关键实验发现**：
- log-log 回归显示相变阈值的斜率约为 $-0.45$，支持 $\sigma = \Theta(n^{-0.5})$ 的相变猜测
- 即使 $\|X^{\star} - \Pi^{\star}\|_F / \sqrt{n}$ 接近 1（$\sigma \in [0.3, 0.5]$），后处理（Hungarian 投影）仍能成功——说明 $X^{\star}$ 虽然整体远离 $\Pi^{\star}$，但相对其他排列矩阵仍有偏好

---

## 亮点与洞察

1. **首个 Birkhoff 松弛的理论保证**：此前没有关于这一「紧」凸松弛的理论结果，本文首次证明其在非零噪声下的成功条件，填补了理论与实践之间的鸿沟
2. **优雅的相变刻画**：$\sigma$ 从 $o(n^{-1})$ 到 $\Omega(n^{-0.5})$ 的渐变行为揭示了松弛紧度的本质——不是非黑即白的成功/失败，而是平滑过渡
3. **对偶证书构造的技巧**：利用 GOE 矩阵特征分解作为基底，设置 $R = J - I$ 来约束 off-diagonal 元素之和，特征值间距的精细分析（区分 $|i-j|$ 大小的不同处理）是关键技术贡献
4. **实践启示**：Birkhoff 松弛无需调参（vs GRAMPA 需要选 $\eta$），且后处理步骤（Hungarian）在理论边界之外仍然有效，说明实际可用范围比理论保证更宽

---

## 局限性

1. **理论与实践的 gap**：充分条件 $\sigma = o(n^{-1})$ 距离猜测的最优阈值 $\sigma = o(n^{-0.5})$ 还有 $\sqrt{n}$ 的差距，population 分析暗示可以改进
2. **模型假设较强**：仅分析了高斯 Wigner 模型（GOE 矩阵），对更实际的 Erdős-Rényi 图模型的推广受限于特征向量集中性（Claim 7 依赖 GOE 的均匀分布特征向量）
3. **未分析后处理的额外收益**：Well-Separation 结果（$\sigma = \Omega(n^{-0.5})$ 时 $X^{\star}$ 远离 $\Pi^{\star}$）不等于失败——实验表明 Hungarian 投影在此范围仍可能成功，但论文未给出理论分析
4. **计算可扩展性**：SCS 求解器需要 50GB 内存和最多 3 小时才能处理 $n=500$ 的实例，在大规模应用中可能不实用

---

## 相关工作

- **凸松弛方法**：GRAMPA (Fan et al., 2023) 通过正则化换取更宽松的理论保证但牺牲实际性能；Simplex 松弛 (Araya & Tyagi, 2024) 仅限 $\sigma=0$；SDP 松弛 (Ling, 2024) 分析了 QAP 的精确性
- **谱方法**：Umeyama (1988) 的特征分解方法；Feizi et al. (2019) 的谱对齐；Ganassali et al. (2022) 的顶特征向量对齐 (EIG1)
- **随机图匹配**：Erdős-Rényi 模型下的精确匹配 (Mao et al., 2023)、树匹配 (Ganassali & Massoulié, 2020)、度数分析 (Ding et al., 2021)
- **应用驱动**：形状匹配 (Aflalo et al., 2015)、图像排列 (Dym et al., 2017)、基因测序 (Fogel et al., 2013)、计算机视觉 (Birdal & Simsekli, 2019)

---

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|:-----------:|------|
| 理论深度 | 9 | 对偶证书构造精巧，特征值间距分析技术扎实 |
| 创新性 | 8 | 首次分析 Birkhoff 松弛在有噪声条件下的行为 |
| 实验验证 | 7 | 仅限 GOE 模型的数值验证，缺少真实数据集 |
| 写作质量 | 8 | 结构清晰，直觉解释充分（population 版本分析） |
| 实用价值 | 6 | 理论结果优雅但实际可扩展性受限，$n=500$ 已达上限 |
| **总评** | **7.5** | 理论贡献扎实的图匹配松弛分析工作，相变刻画优雅 |
<!-- 由 src/gen_stubs.py 自动生成 -->
# Graph Alignment via Birkhoff Relaxation

**会议**: NeurIPS 2025  
**arXiv**: [2503.05323](https://arxiv.org/abs/2503.05323)  
**代码**: [smv30/convex_rel_for_graph_alignment](https://github.com/smv30/convex_rel_for_graph_alignment)  
**领域**: 图算法 / 组合优化  
**关键词**: graph alignment, quadratic assignment, Birkhoff polytope, convex relaxation, phase transition, Gaussian Wigner Model

## 一句话总结

首次为 Birkhoff 松弛（QAP 的紧凸松弛）在高斯 Wigner 模型下建立理论保证：当噪声 $\sigma = o(n^{-1})$ 时松弛解近似真实排列，$\sigma = \Omega(n^{-0.5})$ 时松弛解远离真实排列，揭示了相变现象。

## Problem

**图对齐问题 (Graph Alignment)**：给定两个 $n$ 个顶点的无向图 $G_1, G_2$，要找到一个顶点映射使得边重叠最大化。数学上等价于 Quadratic Assignment Problem (QAP)：

$$\Pi^\star = \arg\min_{X \in \mathcal{P}_n} \|AX - XB\|_F^2$$

其中 $\mathcal{P}_n$ 是 $n \times n$ 排列矩阵集合。QAP 在最坏情况下是 NP-hard 的，甚至近似求解也很困难。

**现有方法的不足**：

- **GRAMPA** (Fan et al.)：在 $\sigma = O(1/\log n)$ 时成功，但需要调参正则化参数 $\eta$，且实际性能弱于 Birkhoff 松弛
- **Simplex 松弛** (Araya & Tyagi)：仅在 $\sigma = 0$（无噪声）时有理论保证
- **EIG1** (谱方法)：阈值为 $\sigma = \Theta(n^{-7/6})$，条件更严格
- Birkhoff 松弛在实践中表现最优，但此前缺乏任何理论保证

## Core Idea

将 QAP 的可行域从排列矩阵集 $\mathcal{P}_n$ 松弛为其凸包——**Birkhoff 多面体** $\mathcal{B}_n$（即所有双随机矩阵的集合），得到 Birkhoff 松弛：

$$X^\star = \arg\min_{X \in \mathcal{B}_n} \|AX - XB\|_F^2$$

这是 QAP 的**最紧凸松弛**（tight convex relaxation）。核心贡献是证明了松弛解 $X^\star$ 与真实排列 $\Pi^\star$ 之间的距离存在**相变 (phase transition)**：

| 噪声条件 | 行为 |
|---|---|
| $\sigma = o(n^{-1})$ | $\|X^\star - \Pi^\star\|_F^2 = o(n)$，小扰动 |
| $\sigma = \Omega(n^{-0.5})$ | $\|X^\star - \Pi^\star\|_F^2 = \Omega(n)$，远离 |

## Method

### Gaussian Wigner Model 设定

输入图的（加权）邻接矩阵来自相关的 Gaussian Orthogonal Ensemble (GOE)：

- $A$ 是 GOE 矩阵：$A_{ii} \sim N(0, 2/n)$，$A_{ij} = A_{ji} \sim N(0, 1/n)$
- $B_{\pi^\star(i), \pi^\star(j)} = A_{ij} + \sigma Z_{ij}$，其中 $Z$ 是独立的 GOE 矩阵
- 相关系数为 $1/\sqrt{1 + \sigma^2}$，$\sigma$ 控制噪声强度

### Part I: Well-Separation 证明 ($\sigma = \Omega(n^{-0.5+\epsilon})$)

证明思路是建立 $\|AX^\star - X^\star B\|_F^2$ 的上下界：

1. **上界**：用 $J/n$（均匀矩阵）作为可行解，得到 $\|AX^\star - X^\star B\|_F^2 \leq 9n^\epsilon$
2. **下界**：展开 $\|AX^\star - X^\star B\|_F^2$，利用 $B = A + \sigma Z$ 得到关于 $\|I - X^\star\|_F$ 的下界：
   $$\|AX^\star - X^\star B\|_F^2 \geq \frac{\sigma^2 n}{4} - 2c\sigma^2 \sqrt{n} \|I - X^\star\|_F$$
3. **合并**：两个界结合推出 $\|I - X^\star\|_F \geq \sqrt{n}/(16c)$

关键技术点：利用 GOE 矩阵的 $\|Z\|_F^2 \geq n/2$ 的集中性，以及 $\max_{i \neq j} |(AZ - ZA)_{ij}| \leq 8n^{\epsilon/2 - 0.5}$ 的对角元素界。

### Part II: Small-Perturbation 证明 ($\sigma = o(n^{-1-\epsilon})$)

这是论文最具技术性的部分，基于**对偶构造 (dual certificate)**：

1. **简化版本**：先考虑 $\sigma = 0$ 的情况，此时 $X = I$ 是最优解且 $AX - XA = 0$
2. **对偶问题**：引入对偶变量 $(R, \mu, \tilde{\mu}, M)$，构造近似可行对偶解
3. **关键构造**：
    - 设 $R = J - I$（对角为 $-1$，非对角为 $1$）
    - 利用 GOE 矩阵的特征向量 $\{u_i\}$ 构造 $M = \sum_{i \neq j} \frac{\tilde{w}_{ij}}{\lambda_i - \lambda_j} u_i u_j^T$
    - $\mu$ 也基于特征向量构造，确保 $\langle \mu, \mathbf{1} \rangle = 0$（强对偶性）
4. **核心不等式 (Lemma 6)**：对任意 $X \in \mathcal{B}_n$：
   $$\sum_{j \neq i} X_{ij} \leq 2n^{3/2 + 7\epsilon/8} \|AX - XA\|_F + 4n^{1 - \epsilon/32}$$
5. **辅助界 (Lemma 7)**：$\|AX^\star - X^\star A\|_F \leq c\sigma\sqrt{n}$
6. **合并**：当 $\sigma = n^{-1-\epsilon}$ 时得到 $\sum_{i \neq j} X_{ij}^\star \leq 5n^{1-\epsilon/32}$，进而 $\|X^\star - I\|_F^2 \leq 10n^{1-\epsilon/32}$

### 特征值间距的精细分析

证明中需要控制 $\sum_{i \neq j} \frac{1}{(|\lambda_i - \lambda_j| + n^{-1-\epsilon})^2}$。直接界给出 $n^{4+2\epsilon}$（太松），论文通过分离 $|i-j|$ 大小两种情况，结合 Nguyen-Vu 的特征值间距尾界和 Markov 不等式，得到更紧的 $n^{3+3\epsilon/2}$ 上界。引入 $n^{-1-\epsilon}$ 正则化项是确保期望有限的关键技巧。

### Rounding 过程

- **简单 rounding**：$\hat{\pi}(i) = \arg\max_j X_{ij}^\star$，可恢复 $1 - o(1)$ 比例的正确对齐 (Corollary 2)
- **Hungarian projection**：$\tilde{\Pi} = \arg\max_{\Pi \in \mathcal{P}_n} \langle X^\star, \Pi \rangle$，同样成功 (Corollary 3)

## Training/Inference

本文是纯理论工作，无训练过程。实际求解 Birkhoff 松弛使用：

- **凸优化求解器**：cvxpy + SCS (Splitting Conic Solver)，`use_indirect=True` 以适应大规模实例
- **后处理**：Hungarian 算法将双随机矩阵投影到排列矩阵
- **复杂度**：Birkhoff 松弛是凸二次规划，可在多项式时间内求解；但实际中 $n$ 较大时（$n > 500$）计算代价较高

## Experiments

### 实验设置

- **模型**：Gaussian Wigner Model
- **图大小**：$n = 400$（默认），$n \in \{100, 200, 300, 400, 500\}$（规模实验）
- **噪声范围**：$\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$
- **对比方法**：GRAMPA（$\eta = 0.2$）、Simplex 松弛、Birkhoff 松弛
- **重复次数**：每组参数 10 次取平均
- **硬件**：CPU + 50GB 内存，每实例最大运行时间 3 小时

## Results

### 三种松弛方法对比 ($n = 400$)

| 方法 | 100% 对齐的最大 $\sigma$ | 性能退化起始 $\sigma$ |
|---|---|---|
| **Birkhoff** | **0.5** | 0.6 |
| Simplex | 0.3 | 0.4 |
| GRAMPA | 0.1 | 0.2 |

Birkhoff 松弛在所有噪声水平上显著优于其他两种方法。

### 相变行为验证

- 当 $\sigma \in [0.3, 0.5]$ 时，虽然 $\|X^\star - \Pi^\star\|_F / \sqrt{n}$ 已接近 1（松弛解远离真排列），但 Hungarian rounding 后仍能完美对齐
- 这说明 $X^\star$ 虽然整体不接近 $\Pi^\star$，但仍有向 $\Pi^\star$ 的微弱偏好，足以支撑 rounding

### 相变阈值的经验估计

- 对 $n \in \{100, \ldots, 500\}$ 做 log-log 回归，$\|X^\star - \Pi^\star\|_F / \sqrt{n} = 0.5$ 的相变点斜率为 $-0.45$
- 与理论预测的 $\sigma = \Theta(n^{-0.5})$ 吻合（Theorem 1）

### 不同 $n$ 的性能

随 $n$ 增大，对齐成功率随 $\sigma$ 的衰减加速，但仍然是渐进的，暗示 Birkhoff + rounding 可能在几乎常数的 $\sigma$ 下成功。

## Limitations

1. **理论与实践的 gap**：理论保证 small-perturbation 需要 $\sigma = o(n^{-1})$，但实验中 $\sigma$ 可达 $O(1)$；population 版本的相变在 $n^{-0.5}$，理论仍有 $n^{-0.5}$ 到 $n^{-1}$ 的缺口
2. **模型限制**：仅分析 Gaussian Wigner Model（连续权重），未涉及实际更常见的 Erdős-Rényi 图（二值边）
3. **特征向量集中性**：Small-perturbation 证明强依赖 GOE 特征向量在球面上均匀分布的性质，扩展到一般 Wigner 矩阵是瓶颈
4. **Well-separation 非失败条件**：$\|X^\star - \Pi^\star\|_F = \Omega(\sqrt{n})$ 不意味着 rounding 失败，论文未能刻画 rounding 的精确阈值
5. **计算可扩展性**：SCS 求解 $n > 500$ 的实例耗时超过 3 小时，大规模场景下不实际
6. **无实际数据验证**：仅合成数据，未在社交网络去匿名化、蛋白质对齐等实际场景中测试

## My Notes

- **理论贡献扎实**：首次为 Birkhoff 松弛提供非平凡的理论保证（此前只有 $\sigma = 0$ 的结果），对偶证书的构造技巧精妙
- **对偶构造的直觉**：设 $R = J - I$ 意味着"惩罚所有非对角元素"，然后通过 $M$ 和 $\mu$ 补偿近似可行性残差。这种构造思路可能迁移到其他组合优化的凸松弛分析中
- **相变现象的 population 版本**很优雅：$\bar{X}^\star = \epsilon I + \frac{1-\epsilon}{n}J$，其中 $\epsilon = \frac{2}{2 + \sigma^2(n+1)}$，直接给出 $\sigma \sqrt{n}$ 处的相变
- **实验中 rounding 的"救赎效应"**值得深入：$X^\star$ 虽然 $\|X^\star - \Pi^\star\|_F$ 大，但 Hungarian 投影仍能成功。这暗示分析 rounding 后的性能（而非松弛解本身）可能给出更优的阈值
- **实际价值**：对于 ML 中的图匹配应用（如 GNN 的 graph matching layer），Birkhoff 松弛是标准组件，本文的理论分析有助于理解其鲁棒性边界
- **扩展方向**：(1) 填补 $n^{-1}$ 到 $n^{-0.5}$ 的 gap；(2) 分析 rounding 后的精确阈值；(3) 推广到 Erdős-Rényi 模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 Birkhoff 松弛在噪声模型下的相变分析，对偶证书构造方法新颖
- 实验充分度: ⭐⭐⭐ 理论+合成实验互相印证，但缺乏真实数据验证
- 写作质量: ⭐⭐⭐⭐ 数学严谨，结构清晰，直觉解释充分
- 价值: ⭐⭐⭐⭐ 图匹配/QAP 凸松弛理论的重要进展，可能启发后续更优阈值的分析

<!-- RELATED:START -->

## 相关论文

- [On Topological Descriptors for Graph Products](on_topological_descriptors_for_graph_products.md)
- [Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](../../ICLR2026/others/learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)

<!-- RELATED:END -->
