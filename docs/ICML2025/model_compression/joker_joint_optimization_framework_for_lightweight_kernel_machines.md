---
title: >-
  [论文解读] Joker: Joint Optimization Framework for Lightweight Kernel Machines
description: >-
  [ICML2025][模型压缩][核方法] 提出 Joker 框架，通过对偶块坐标下降 + 信赖域 (DBCD-TR) 和随机傅里叶特征近似，以 ~2GB 内存实现多种大规模核模型（KRR / KLR / SVM 等）的统一高效训练，内存节省高达 90% 且性能不降。
tags:
  - ICML2025
  - 模型压缩
  - 核方法
  - 大规模学习
  - 对偶优化
  - 块坐标下降
  - 信赖域方法
  - 随机傅里叶特征
  - 低内存
---

# Joker: Joint Optimization Framework for Lightweight Kernel Machines

**会议**: ICML2025  
**arXiv**: [2505.17765](https://arxiv.org/abs/2505.17765)  
**代码**: 待确认  
**领域**: 核方法优化  
**关键词**: 核方法, 大规模学习, 对偶优化, 块坐标下降, 信赖域方法, 随机傅里叶特征, 低内存

## 一句话总结

提出 Joker 框架，通过对偶块坐标下降 + 信赖域 (DBCD-TR) 和随机傅里叶特征近似，以 ~2GB 内存实现多种大规模核模型（KRR / KLR / SVM 等）的统一高效训练，内存节省高达 90% 且性能不降。

## 研究背景与动机

核方法是非线性学习的经典范式，理论基础深厚，近年与深度学习（Neural Tangent Kernel 等）的联系进一步突显其潜力。但大规模核方法面临两大瓶颈：

**内存开销过高**：当前 SOTA 方法 Falkon 基于 Nyström 近似 + Cholesky 预条件，需要 $O(M^2)$ 内存存储预条件器（$M$ 为 Nyström 中心数）。在 HIGGS 数据集上使用 $M=1.2 \times 10^5$ 需 **>55GB** 内存，普通用户难以承受。

**模型多样性不足**：已有方法主要聚焦 KRR，核逻辑回归 (KLR) 和 SVM 缺乏高效的大规模实现。LIBSVM/ThunderSVM 训练时间超一周，LogFalkon 同样有高内存问题。

Joker 的目标：**用统一优化框架覆盖多种核模型，同时大幅降低内存需求**。

## 方法详解

### 1. 统一对偶问题 (Joint Optimization by Duality)

对于一般核机器的原始问题：

$$\min_{\boldsymbol{\theta}} \frac{\lambda}{2} \|\boldsymbol{\theta}\|_{\mathcal{H}}^2 + \sum_{i=1}^{n} \ell(y_i, \langle \boldsymbol{\theta}, \boldsymbol{\varphi}(\boldsymbol{x}_i) \rangle)$$

通过 Fenchel 共轭 $\xi_y^*(v)$ 推导其对偶形式（Theorem 1）：

$$\min_{\boldsymbol{\alpha} \in \Omega} \frac{1}{2} \boldsymbol{\alpha}^\top \boldsymbol{K} \boldsymbol{\alpha} + \frac{1}{\lambda} \sum_{i=1}^{n} \xi_{y_i}^*(-\lambda \alpha_i)$$

**关键性质**：
- 对偶问题始终为**凸优化**，且约束为简单的**箱约束** $\alpha_i \in [\tau_i^L, \tau_i^U]$；
- 强对偶性成立（Slater 条件），对偶最优即原始最优；
- 对偶 Hessian 线性依赖 $\boldsymbol{K}$，而原始 Hessian 是 $\boldsymbol{K}$ 的二次形式，因此**对偶优化收敛更快**；
- 通过 Proposition 2（infimal convolution 的共轭分解），可轻松处理 Huber 损失等复合损失函数。

不同损失函数对应不同核模型，通过同一对偶框架统一处理：

| 模型 | 损失函数 | 对偶共轭 $\xi_y^*(v)$ |
|------|---------|----------------------|
| KRR | 平方损失 $(y-u)^2/2$ | $v^2/2 + vy$ |
| KLR | Logistic 损失 | 二元熵 $\text{bEnt}(-vy)$ |
| L1-SVC | Hinge 损失 | $vy$，$-1 \le vy \le 0$ |
| L2-SVC | 平方 Hinge 损失 | $v^2/2 + vy$，$vy \le 0$ |
| SVR | $\varepsilon$-insensitive | $\varepsilon|v| + vy$，$-1 \le v \le 1$ |
| Huber 回归 | Huber 损失 | $v^2/2 + vy$，$-\delta \le v \le \delta$ |

### 2. DBCD-TR 求解器

核心求解器为 **Dual Block Coordinate Descent with Trust Region (DBCD-TR)**：

**块坐标下降**：每次选取索引子集 $\mathcal{B} \subset [n]$，固定其余变量，求解子问题：

$$\min_{\boldsymbol{\alpha}_{\mathcal{B}}} \frac{1}{2} \boldsymbol{\alpha}_{\mathcal{B}}^\top \boldsymbol{K}_{\mathcal{B},\mathcal{B}} \boldsymbol{\alpha}_{\mathcal{B}} + \boldsymbol{\alpha}_{\mathcal{B}_\complement}^\top \boldsymbol{K}_{\mathcal{B}_\complement,\mathcal{B}} \boldsymbol{\alpha}_{\mathcal{B}} + f(\boldsymbol{\alpha}_{\mathcal{B}}), \quad \text{s.t. } \boldsymbol{\tau}^L \le \boldsymbol{\alpha}_{\mathcal{B}} \le \boldsymbol{\tau}^U$$

**信赖域方法**：构造二次模型 $\mu_k(\boldsymbol{s}) = J(\boldsymbol{\alpha}_k) + \boldsymbol{g}_k^\top \boldsymbol{s} + \frac{1}{2} \boldsymbol{s}^\top \boldsymbol{Q}_k \boldsymbol{s}$，其中 $\boldsymbol{Q}_k = \boldsymbol{K}_{\mathcal{B},\mathcal{B}} + \nabla^2 f$。通过比率 $\rho_k$ 评估步长质量，自适应调整信赖域半径 $\Delta_k$。

**截断 CG-Steihaug**：求解信赖域子问题时使用截断共轭梯度法（Algorithm 2），遇到箱约束违反或超出信赖域边界时提前终止，最后投影回可行域。相比投影 Newton 法更高效。

**复杂度**：空间 $O(|\mathcal{B}|^2)$（存储子核矩阵），时间 $O(T_{\text{TR}} \cdot T_{\text{CG}} \cdot |\mathcal{B}|^2)$，实际 $T_{\text{TR}} \le 50$，$T_{\text{CG}} \le 10$。

### 3. Inexact Joker（随机傅里叶特征近似）

精确核评估 $\boldsymbol{K}_{\mathcal{B},:}\boldsymbol{\alpha}$ 的时间复杂度为 $O(nd|\mathcal{B}|)$，$n \ge 10^6$ 时不可接受。通过 Random Fourier Features (RFF) 近似：

$$\boldsymbol{\psi}(\boldsymbol{x}) = \sqrt{\frac{2}{M}} \cos(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}), \quad \boldsymbol{w}_i \sim p_{\mathcal{K}}, \quad b_i \sim U_{[0,2\pi]}$$

维护低维权重向量 $\boldsymbol{\theta} = \sum_i \alpha_i \boldsymbol{\psi}(\boldsymbol{x}_i) \in \mathbb{R}^M$，将核梯度计算简化为：

$$\boldsymbol{K}_{\mathcal{B},:}\boldsymbol{\alpha} = \boldsymbol{\psi}(\boldsymbol{X}_{\mathcal{B}})^\top \boldsymbol{\theta}$$

时间复杂度从 $O(nd|\mathcal{B}|)$ 降至 $O(Md|\mathcal{B}|)$（$M \ll n$），空间仅需存储 $\boldsymbol{\theta} \in \mathbb{R}^M$。

## 实验关键数据

在 HIGGS 数据集（$n=11M$，$d=28$）上使用**单张 RTX 3080 (10GB)** 进行对比：

| 方法 | 类型 | 内存 | 时间 | 支持模型 |
|------|------|------|------|---------|
| Falkon | 近似核 | >50GB | <1h | KRR |
| LogFalkon | 近似核 | >50GB | <1h | KLR |
| EigenPro3 | 近似核 | ~7GB | >15h | KRR |
| LIBSVM | 精确核 | <2GB | >1周 | SVC/SVR |
| ThunderSVM | 精确核 | ~8GB | >1周 | SVC/SVR |
| **Joker** | **混合** | **~2GB** | **~1h** | **KRR/KLR/SVM 等** |

- **内存节省高达 90%**：相比 Falkon 从 >50GB 降至 ~2GB
- 训练时间与 Falkon 可比（~1 小时），远优于 LIBSVM/ThunderSVM（>1 周）
- 性能与 SOTA 持平甚至更优
- 统一框架覆盖 KRR、KLR、L1-SVC、L2-SVC、SVR、Huber 回归、$L_p$ 回归等多种模型

## 亮点与洞察

1. **统一对偶视角**：通过 Fenchel 共轭将不同损失的核模型映射到结构一致的对偶问题，箱约束 + 可分离结构天然适合坐标下降，设计优雅。
2. **信赖域嵌入 BCD**：巧妙解决了箱约束下步长调节的困难，无需手动调参，CG-Steihaug + 截断策略高效且鲁棒。
3. **RFF 加速的精巧实现**：通过维护低维权重 $\boldsymbol{\theta}$ 避免每轮重新计算核梯度，增量更新 $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{\psi}(\boldsymbol{X}_{\mathcal{B}})(\boldsymbol{\alpha}^{\text{new}} - \boldsymbol{\alpha})$，开销极低。
4. **Proposition 2 处理复合损失**：利用 infimal convolution 的共轭分解性质，自然处理 Huber 等原始形式难优化的损失。
5. **对偶优化的条件数优势**：对偶 Hessian 线性依赖 $\boldsymbol{K}$，原始 Hessian 二次依赖，对偶收敛更快。

## 局限与展望

1. **仅支持移不变核**：当前 RFF 实现基于 Bochner 定理，对非移不变核（如多项式核）需额外扩展。
2. **RFF 的近似误差**：$M$ 较小时近似质量有限，$M$ 较大时内存/时间增加，如何自适应选择 $M$ 值得研究。
3. **缺少深度学习对比**：作为核方法论文，未与相似规模的神经网络在相同任务上充分对比。
4. **块大小 $|\mathcal{B}|$ 的选择**：需要平衡子问题精度和并行效率，论文未充分讨论自适应策略。
5. **非光滑损失处理**：虽然提到信赖域兼容非光滑优化，但详细的非光滑场景实验不充足。

## 相关工作与启发

- **Falkon / LogFalkon**：Nyström + Cholesky 预条件的 SOTA，性能优但内存瓶颈严重，Joker 的直接竞争对手
- **EigenPro3**：投影梯度下降避免高空间复杂度，但时间代价高
- **LIBSVM / ThunderSVM**：精确核 SVM 实现，SMO 求解器在大规模场景过时
- **Random Fourier Features (Rahimi & Recht, 2007)**：核近似的基石方法，Joker 的关键组件
- **对偶坐标上升 (Shalev-Shwartz & Zhang, 2013)**：线性模型的对偶优化框架，Joker 将其推广至核模型
- **信赖域方法 (Sorensen, 1982)**：经典非线性优化技术，Joker 首次将其与 BCD 结合用于核优化

## 评分
- 新颖性: ⭐⭐⭐⭐ — 对偶 + BCD + 信赖域 + RFF 的组合设计新颖，统一多模型框架有实用价值
- 实验充分度: ⭐⭐⭐⭐ — 在大规模数据集上与多方法全面对比，90%内存节省的结论有说服力
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨清晰，算法伪代码完整，理论与实践结合好
- 价值: ⭐⭐⭐⭐ — 填补了大规模核方法低资源训练的空白，对资源受限场景（单 GPU）很有意义
