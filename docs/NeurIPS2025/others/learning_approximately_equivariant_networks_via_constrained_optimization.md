---
title: >-
  [论文解读] Learning (Approximately) Equivariant Networks via Constrained Optimization
description: >-
  [NeurIPS 2025][等变性] 提出ACE（Adaptive Constrained Equivariance）框架，将等变神经网络训练建模为约束优化问题，通过对偶方法自动从灵活的非等变模型渐进过渡到等变模型，无需手动调参即可适应完全和部分对称数据。
tags:
  - NeurIPS 2025
  - 等变性
  - 约束优化
  - 同伦方法
  - 近似对称
  - 对偶方法
---

# Learning (Approximately) Equivariant Networks via Constrained Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.13631](https://arxiv.org/abs/2505.13631)  
**代码**: 无  
**领域**: 机器学习理论 / 等变神经网络  
**关键词**: 等变性, 约束优化, 同伦方法, 近似对称, 对偶方法

## 一句话总结

提出ACE（Adaptive Constrained Equivariance）框架，将等变神经网络训练建模为约束优化问题，通过对偶方法自动从灵活的非等变模型渐进过渡到等变模型，无需手动调参即可适应完全和部分对称数据。

## 研究背景与动机

等变神经网络通过架构编码对称性来提升泛化和样本效率，但训练面临三个核心挑战：

1. **复杂损失景观**：等变约束即使在数据完全对称时也会使损失景观复杂化，减缓优化
2. **部分对称的现实数据**：噪声、测量偏差、动力学相变等效应打破完美对称，严格等变模型可能欠拟合
3. **手动调参负担**：现有放松方法（如REMUL的惩罚权重、PennPaper的退火schedule）需要大量领域特定调参

已有方法的不足：

- **REMUL**：向损失添加等变性惩罚并自适应权重 $\alpha, \beta$，但无法保证最终解的等变性程度
- **PennPaper**：手动减小扰动参数 $\gamma$ 至零，但对schedule敏感，且额外使用Lie导数惩罚增加超参

## 方法详解

### 整体框架

构造同伦架构 $f_{\theta,\gamma} = f_{\theta,\gamma}^L \circ \cdots \circ f_{\theta,\gamma}^1$，其中每层 $f_{\theta,\gamma}^i = f_\theta^{\text{eq},i} + \gamma_i f_\theta^{\text{neq},i}$。当 $\gamma = 0$ 时模型等变，$|\gamma_i| > 0$ 时允许偏离。

将训练建模为约束优化问题，通过对偶方法（梯度下降-上升）自动调节 $\gamma$。

### 关键设计

1. **等式约束方案（完全对称数据，Algorithm 1）**:
    - 功能：约束 $\gamma_i = 0$，通过对偶变量 $\lambda_i$ 自动控制过渡速度
    - 核心思路：初始化 $\gamma^{(0)} = 1$（非等变），对偶变量 $\lambda_i$ 在 $\gamma_i > 0$ 时持续增长，逐渐将模型推向等变。关键更新：$\gamma_i^{(t+1)} = \gamma_i^{(t)} - \eta_p(\nabla_{\gamma_i} J_0^{(t)} + \lambda_i^{(t)})$，$\lambda_i^{(t+1)} = \lambda_i^{(t)} + \eta_d \gamma_i^{(t)}$
    - 设计动机：对偶方法等价于自适应退火——根据施加等变性对下游性能的实际影响调节收紧速度

2. **弹性不等式约束方案（部分对称数据，Algorithm 2）**:
    - 功能：替换等式约束为 $|\gamma_i| \leq u_i$，松弛变量 $u_i$ 也是优化变量
    - 核心思路：添加 $\frac{\rho}{2}\|u\|^2$ 到目标函数惩罚大的松弛。$u_i^* = \lambda^*/\rho$，约束越紧的层对应越大的 $\lambda_i$。投影更新 $\lambda_i^{(t+1)} = [\lambda_i^{(t)} + \eta_d(|\gamma_i^{(t)}| - u_i^{(t)})]_+$
    - 设计动机：当数据部分对称时，$\gamma_i$ 在某些层不会消失——通过对偶变量大小自动检测哪些层需要放松等变性

3. **理论保证（Theorem 4.1 & 4.2）**:
    - 功能：给出去除 $\gamma$ 后的近似误差和等变违反程度的显式界
    - 核心思路：Thm 4.1 — $\|f_{\theta,\gamma}(x) - f_{\theta,0}(x)\| \leq [\sum_{k=0}^{L-1}(1+\bar{\gamma})^k] \bar{\gamma} B M^{L-1} \|x\|$；Thm 4.2 — $\|\rho_Y(g)f_{\theta,\gamma}(x) - f_{\theta,\gamma}(\rho_X(g)x)\| \leq 2\bar{\gamma}(M + C\bar{\gamma})^{L-1}LB^2\|x\|$
    - 设计动机：保证当 $\gamma_i$ 足够小时，截断为等变模型的误差可控

### 损失函数 / 训练策略

等式约束版的拉格朗日函数：$\hat{L}(\theta, \gamma, \lambda) = \frac{1}{N}\sum_{n=1}^N \ell_0(f_{\theta,\gamma}(x_n), y_n) + \sum_{i=1}^L \lambda_i \gamma_i$

关键：不使用任何等变性惩罚（$\beta = 0$），完全依赖约束+对偶方法。只需两个学习率 $\eta_p, \eta_d$ 和弹性常数 $\rho = 1$。

## 实验关键数据

### 主实验（表格）

CMU MoCap 运动预测 MSE（$\times 10^{-2}$）：

| 模型 | Run | Walk |
|------|-----|------|
| EGNN | 50.9±0.9 | 28.7±1.6 |
| EGNO (原文) | 33.9±1.7 | 8.1±1.6 |
| EGNO + ACE (等式) | 改进 | 改进 |
| EGNO + ACE (弹性不等式) | **最优** | **最优** |

N-Body物理仿真：SEGNN + ACE在验证MSE和样本效率上均优于标准SEGNN

### 消融实验

- 等式约束（Alg. 1）在完全对称数据上改善收敛轨迹：早期灵活探索 → 后期收紧为等变
- 弹性不等式（Alg. 2）在含噪/对称破缺数据上保持部分等变同时提升性能
- 验证等变误差在训练过程中逐渐趋近零（Figure 4）
- 理论界（Thm 4.2）与实际观测的等变违反程度趋势一致

### 关键发现

- ACE在多个架构（SEGNN、EGNN、EGNO、p4m-CNN）和任务（N-Body、运动预测、图像分类）上一致性地改善表现
- 在完全对称数据上，ACE的优势来自优化景观的平滑化
- 在部分对称数据上，ACE自动发现哪些层需要放松等变性（$\lambda_i$ 大的层）
- 样本效率提升显著：相同样本数下ACE达到更低误差
- 对输入扰动的鲁棒性也得到改善

## 亮点与洞察

- **无需手动调参**：完全自动的等变性过渡，无需schedule、惩罚权重或领域知识
- **理论+实践一致**：理论界的预测与实验行为吻合
- **通用性强**：适用于任何可微分的 $f_{\theta,\gamma}$ 满足 $f_{\theta,0}$ 等变的架构
- 对偶方法与同伦/模拟退火的深层联系提供了优化视角的理论洞察

## 局限与展望

- $\gamma_i$ 在有限迭代内不会精确为零，需要最终截断（引入 Thm 4.1 的误差）
- 非等变分支 $f_\theta^{\text{neq},i}$ 增加了参数量和计算开销
- 在大规模模型（如大型GNN）上的扩展性未验证
- 对偶方法的收敛性在非凸设定下的理论保证依赖"足够丰富"的参数化假设
- 目前仅测试了离散群（如p4m）和连续群（如SE(3)），更大群的适用性待验证

## 相关工作与启发

- **REMUL (EquivMTL)**：惩罚式方法，需调 $\alpha, \beta$ 且不保证等变性
- **PennPaper**：手动schedule + Lie导数惩罚，本文通过对偶方法自动化取代
- **Residual Pathway Priors**：添加不变分支增加灵活性，但无渐进收紧机制
- ACE的约束优化视角可推广到其他结构化神经网络约束（如稀疏性、低秩等）

## 评分

⭐⭐⭐⭐ — 理论和方法均有贡献，提供了等变网络训练的通用、无需调参的解决方案，实验全面

<!-- RELATED:START -->

## 相关论文

- [On Universality Classes of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)
- [FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](../../ICML2025/others/the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)

<!-- RELATED:END -->
