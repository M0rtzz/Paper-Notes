---
title: >-
  [论文解读] Regression Trees Know Calculus
description: >-
  [NeurIPS 2025][回归树] 揭示常叶回归树中隐含的梯度信息——通过相邻节点均值差的有限差分类比，高效提取梯度估计，进而将活跃子空间（Active Subspace）和集成梯度（Integrated Gradient）等微分工具引入树模型，拓展了树模型的可解释性和预测改进能力。
tags:
  - NeurIPS 2025
  - 回归树
  - 梯度估计
  - 活跃子空间
  - 集成梯度
  - 可解释性
---

# Regression Trees Know Calculus

**会议**: NeurIPS 2025  
**arXiv**: [2405.13846](https://arxiv.org/abs/2405.13846)  
**代码**: 暂无  
**领域**: 机器学习理论 · 回归树可解释性  
**关键词**: 回归树, 梯度估计, 活跃子空间, 集成梯度, 可解释性

## 一句话总结

揭示常叶回归树中隐含的梯度信息——通过相邻节点均值差的有限差分类比，高效提取梯度估计，进而将活跃子空间（Active Subspace）和集成梯度（Integrated Gradient）等微分工具引入树模型，拓展了树模型的可解释性和预测改进能力。

## 研究背景与动机

回归树（包括随机森林和梯度提升）是数据科学中最广泛使用的工具之一，以分段常数函数逼近目标函数。由于其主要设计目标是处理非线性、交互效应和尖锐不连续性，对于树模型在逼近光滑可微函数时的性质，研究者关注较少。

现有树模型的可解释性工具主要有：
- **MDI（不纯度均值降低）**：对变量比例分配分裂时的代价降低，但已知存在偏差问题
- **特征置换**：打乱某特征后衡量性能下降，但在特征相关时表现不佳
- **SHAP**：通用的解释框架，但计算量大

另一方面，神经网络和高斯过程等可微模型拥有丰富的基于梯度的分析工具：
- **集成梯度（IG）**：沿参考点到目标点的路径积分梯度，提供局部归因
- **活跃子空间（AS）**：通过梯度外积的期望矩阵的特征分解，发现重要线性方向

这些方法无法直接用于树模型，因为分段常数函数的梯度几乎处处为零。本文的核心洞察是：虽然树的形式梯度为零，但**分裂节点两侧的均值差隐含着梯度信息**，可以构造出类比有限差分的梯度估计器。

## 方法详解

### 整体框架

将深度为$K$的回归树的每个内部节点视为一个有限差分采样点，从中提取关于分裂变量的偏导数估计。通过自根向叶的遍历，逐层更新梯度向量，最终为每个叶节点关联一个完整的梯度估计。在此基础上进一步构建活跃子空间矩阵和集成梯度等积分-微分量的估计器。

### 关键设计

1. **树基梯度估计器（TBGE）**：

   对节点$i$，其沿变量$\sigma_i$将数据分为左右子节点。左右子节点的响应变量均值分别为$\hat{\mu}_l^i$和$\hat{\mu}_r^i$，节点$i$在维度$\sigma_i$上的范围为$[l_{\sigma_i}^i, u_{\sigma_i}^i]$。定义：

   $$\gamma_i := \frac{2(\hat{\mu}_r^i - \hat{\mu}_l^i)}{u_{\sigma_i}^i - l_{\sigma_i}^i} \approx \frac{\partial f}{\partial x_{\sigma_i}}(\mathbf{x}), \quad \forall \mathbf{x} \in [\mathbf{l}^i, \mathbf{u}^i]$$

   这本质上是用子节点均值差除以节点宽度——与经典有限差分完全类比。每个节点只能估计一个偏导数（沿分裂变量的方向），但通过从根到叶的遍历，可以组合不同深度节点的估计形成完整梯度向量$\mathbf{G}^i$。

   **设计动机**：当节点足够小且函数梯度变化平滑时，这个有限差分逼近是合理的。计算只需遍历树一次，复杂度与标准预测相同。

2. **树基活跃子空间（TBAS）**：

   利用分区估计器（PBE），对倒数第二层所有节点的梯度估计加权求和，构造活跃子空间矩阵：

   $$\hat{C}_\mu^f = \sum_{i \in \mathcal{N}_{K-1}} \mathbf{G}^i (\mathbf{G}^i)^\top \mu([\mathbf{l}^i, \mathbf{u}^i])$$

   其中$\mu([\mathbf{l}^i, \mathbf{u}^i])$是节点$i$在度量$\mu$下的测度（均匀分布时即体积比）。对该矩阵做特征分解即可获得重要的线性方向组合。

   **设计动机**：无需额外采样或辅助模型，直接复用树结构中已有信息。且树天然倾向于在重要变量上分裂，TBAS具有内建的稀疏性归纳偏置。

3. **树基集成梯度（TBIG）**：

   在参考点$\mathbf{x}^*$和目标点$\mathbf{x}$之间的路径上做蒙特卡洛采样，用TBGE替代真实梯度：

   $$\hat{IG}(\mathbf{x}) = (\mathbf{x} - \mathbf{x}^*) \odot \frac{1}{M} \sum_{m=1}^{M} \tilde{\nabla}f(u_m \mathbf{x} + (1-u_m)\mathbf{x}^*)$$

   **设计动机**：将神经网络的局部归因方法移植到树模型，为随机森林分类器提供可视化的特征重要性解释。

### 理论保证

**定理4.1**（梯度估计一致性）：当分裂数$S(N)$增长慢于$\log N$时，TBGE收敛到真实梯度，误差率为$O_P(P(\log N)^{-1})$。虽然收敛速率较慢，但利用了树的计算效率优势。

**定理4.2**（积分量一致性）：在有界条件下，PBE估计的积分-微分量随样本量趋于无穷而收敛到真实值。

## 实验关键数据

### 主实验：TBAS旋转对预测的影响

| 数据集 | 方法 | 深度4单树 | 深度8单树 | 深度4随机森林 |
|-------|------|----------|----------|-------------|
| kin40k | TBAS | **0.856** | **0.586** | **0.802** |
| kin40k | 无旋转(Id) | 0.963 | 0.862 | 0.954 |
| kin40k | PCA | 0.964 | 0.872 | 0.954 |
| keggu | TBAS | **0.194** | **0.078** | **0.161** |
| keggu | 无旋转(Id) | 0.350 | 0.121 | 0.344 |
| concrete | TBAS | **0.470** | **0.350** | **0.406** |
| concrete | 无旋转(Id) | 0.537 | 0.403 | 0.462 |

（数值为100-fold RMSE，越低越好。TBAS在多个数据集上显著优于无旋转和PCA旋转。）

### 消融实验：活跃子空间估计效率对比

| 维度 | 方法 | 执行时间趋势 | 估计误差趋势 | 说明 |
|-----|------|------------|------------|------|
| 2-4D | TBAS | 远快 | 可比 | 构成Pareto前沿的大部分 |
| 2-4D | GP | 小样本快，大样本不可行 | 小样本好 | >150样本计算爆炸 |
| 2-4D | PRA | 类似GP | 类似GP | 多项式岭近似 |
| 2-4D | DASM | 中等 | 中等 | 神经网络方法 |
| 10-100D | TBAS | 远优 | 远优 | 稀疏归纳偏置显著 |
| 10-100D | DASM | 唯一可比方法 | 需更大样本 | GP/PRA不可行 |

### 关键发现

1. TBAS在8个基准数据集中至少与其他旋转方法持平，在kin40k和keggu上RMSE降低达10-50%
2. 在低维（2-4D）场景中，TBAS以数量级更快的速度达到与UQ专用方法可比的精度
3. 在高维（10-100D）稀疏场景中，TBAS天然偏好逐坐标的活跃子空间，远优于DASM
4. TBIG在MNIST分类任务上成功识别出数字的判别性像素区域

## 亮点与洞察

1. **概念优雅**：基于一个极简观察——分裂节点两侧均值差即有限差分——打通了树模型与微积分工具的连接
2. **计算高效**：梯度估计仅需一次树遍历，无需额外数据、无需辅助模型
3. **双向价值**：不仅将微分工具引入树模型（提升可解释性），也将树模型引入UQ领域（提供高效估计器）
4. **稀疏归纳偏置**：树的分裂机制天然偏好重要变量，TBAS无需额外正则化即具有稀疏性

## 局限与展望

- 理论分析基于简化假设（按变量循环分裂、中位数分裂），与CART等实际算法有差距
- 梯度估计收敛速率为$O((\log N)^{-1})$，较慢，适合"粗略估计足够"的场景
- 高维时需要非常深的树才能获得高精度梯度估计
- 分类任务和类别型特征的扩展尚在初步阶段
- 每一层覆写上一层的梯度估计，可能通过多层聚合获得更好的收敛率

## 相关工作与启发

- **树可解释性**: SHAP, MDI, 特征置换
- **梯度归因**: Integrated Gradients, Active Subspace Method
- **活跃子空间估计**: GP方法, PRA, DASM
- 启发：其他非可微模型（如KNN、kernel方法）是否也能从类似的"隐含梯度"视角中获益？树模型在物理信息机器学习（PIML）中的应用也值得探索

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 极具洞察力的发现，首次揭示常叶回归树中隐含梯度的估计方法
- 实验充分度：⭐⭐⭐⭐ — 覆盖预测改进、低/高维活跃子空间、可视化等多维度验证
- 写作质量：⭐⭐⭐⭐⭐ — 记号清晰，从直觉到理论到实验层层推进
- 价值：⭐⭐⭐⭐ — 连接树模型与微分工具的桥梁性工作，开启多个有趣的后续方向

<!-- RELATED:START -->

## 相关论文

- [Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [Improving Decision Trees through the Lens of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [Prediction via Shapley Value Regression (ViaSHAP)](../../ICML2025/others/prediction_via_shapley_value_regression.md)
- [Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination](information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)
- [An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)

<!-- RELATED:END -->
