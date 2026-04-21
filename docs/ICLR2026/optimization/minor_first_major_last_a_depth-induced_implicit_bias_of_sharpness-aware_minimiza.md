---
title: >-
  [论文解读] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][优化][Sharpness-Aware Minimization] 深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。
tags:
  - ICLR 2026
  - 优化
  - Sharpness-Aware Minimization
  - 隐式偏差
  - 线性对角网络
  - 特征放大
  - 深度诱导
---

# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization

**会议**: ICLR 2026  
**arXiv**: [2603.08290](https://arxiv.org/abs/2603.08290)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: Sharpness-Aware Minimization, 隐式偏差, 线性对角网络, 特征放大, 深度诱导

## 一句话总结

深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。

## 研究背景与动机

### 问题背景
Sharpness-Aware Minimization (SAM) 通过最小化邻域内最坏情况损失来寻找平坦最小值，在实践中显著提升泛化能力。先前理论工作主要分析 SAM 在有限极小值设定（如平方损失）下的隐式偏差，而在损失下确界位于无穷处（如逻辑损失）的情形尚未被充分理解。

### 动机
作者考察 SAM 在 $L$ 层线性对角网络上训练线性可分二分类数据（逻辑损失）时的隐式偏差。令人惊讶的发现是：

- **深度 $L=1$（线性模型）**：$\ell_\infty$-SAM 和 $\ell_2$-SAM 都收敛到 $\ell_2$ 最大间隔分类器，与梯度下降（GD）一致
- **深度 $L=2$**：行为发生质变——即使在单样本数据集 $\{(\boldsymbol\mu, +1)\}$，$\boldsymbol\mu=(1,2)$ 上，SAM 的轨迹就可能偏离 GD 的 $\ell_1$ 最大间隔方向

这一观察揭示了仅增加一层就能从根本上改变 SAM 的隐式偏差。

## 方法详解

### 整体框架

研究采用**理论分析+实验验证**的框架：
- **模型**：$L$ 层线性对角网络 $f(\mathbf{x}) = \langle \boldsymbol\beta(\boldsymbol\theta), \mathbf{x}\rangle$，其中 $\boldsymbol\beta(\boldsymbol\theta) = \bigodot_{\ell=1}^L \mathbf{w}^{(\ell)}$
- **数据**：线性可分数据集 + 单样本数据集 $\mathcal{D}_{\boldsymbol\mu} = \{(\boldsymbol\mu, +1)\}$
- **损失**：逻辑损失 $\ell(u) = \log(1+\exp(-u))$
- **分析工具**：连续时间 SAM 流（ODE）和重标度 SAM 流

### 关键设计与理论结果

1. **深度1的结果（$L=1$）**

   **定理 3.1**：对几乎所有线性可分数据集，任意扰动半径 $\rho$ 和任意初始化，$\ell_\infty$-SAM 流方向收敛到 $\ell_2$ 最大间隔方向。

   意义：深度1时，SAM 不改变 GD 的隐式偏差。

2. **深度 $L\geq 2$ 的 $\ell_\infty$-SAM**

   **定理 3.2**：在单样本数据集上，每个坐标 $\beta_j(t)$ 的行为完全由初始化 $\alpha_j$ 与扰动半径 $\rho$ 的关系决定：
   - $\alpha_j < \rho$：$\beta_j(t) \to 0$（偶数 $L$）或 $\to \rho^L$（奇数 $L$）
   - $\alpha_j = \rho$：$\beta_j(t) = \rho^L$，保持常数
   - $\alpha_j > \rho$：$\beta_j(t)$ 发散，$L=2$ 时指数增长

   **推论 3.5**：极限方向由 $j^* = \arg\max_{j: \alpha_j > \rho} \mu_j(\alpha_j - \rho)^{L-2}$ 决定。这意味着初始化可以让 SAM 收敛到**任意标准基向量方向**——包括次要特征方向，与 GD 始终选择主要特征形成鲜明对比。

3. **深度2的 $\ell_2$-SAM：序列特征放大**

   **定理 4.2**（极限方向）：$\ell_2$-SAM 流的极限方向是 $\ell_1$ 最大间隔解，与 GD 相同。

   然而，**有限时间**动态却截然不同。论文发现了"序列特征放大"（Sequential Feature Amplification）现象：

   - **时间维度**（定理 4.4）：预测器 $\boldsymbol\beta(t)$ 初期**依赖次要坐标**，随着训练推进**逐渐转向主要坐标**
   - **初始化维度**（定理 4.5）：初始化规模增大时，也观察到从次要到主要特征的类似转变

   **根本原因**：$\ell_2$-SAM 的梯度归一化因子在扰动中起作用——归一化后，梯度中的小坐标被放大，导致对应的 $\beta_j$ 在早期获得更大的增长率。随着训练进行，主要坐标最终凭借更大的特征权重 $\mu_j$ 占主导，但这一过程是渐进的。

### 重标度流技术

针对单样本数据集，通过去除损失导数项 $-\ell'(\langle\boldsymbol\beta(\hat{\boldsymbol\theta}(t)),\boldsymbol\mu\rangle) > 0$ 得到重标度流，这仅是时间重参数化，保留了原始SAM流的空间轨迹，但使分析更加简洁。

## 实验关键数据

### 合成实验
- 2D 单样本数据集 $\boldsymbol\mu=(1,2)$，清晰展示了 GD vs $\ell_\infty$-SAM vs $\ell_2$-SAM 的轨迹差异
- 多样本数据集验证了理论预测在实际设定中的适用性
- 深度 $L=3$ 的实验验证了更深网络中 $\ell_\infty$-SAM 对初始化更加敏感

### 实际网络实验（MNIST + CNN）

| 方法 | Grad-CAM 观察 | 说明 |
|------|-------------|------|
| GD | 聚焦于主要数字像素 | 传统行为 |
| $\ell_2$-SAM | 强调背景/弱像素区域 | 与"先次要后主要"理论一致 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $L=1$, 任意 $\rho$ | 与GD一致 | SAM 不改变偏差 |
| $L=2$, $\ell_\infty$, $\alpha_j < \rho$ | $\beta_j \to 0$ | 坐标被压制 |
| $L=2$, $\ell_\infty$, $\alpha_j > \rho$ | 指数增长 | 可选择次要特征 |
| $L=2$, $\ell_2$ | 先弱后强 | 序列特征放大 |

### 关键发现
- $\ell_\infty$-SAM 在 $L\geq 2$ 时对初始化极度敏感，扰动半径 $\rho$ 充当坐标级别的"门控"阈值
- $\ell_2$-SAM 的有限时间行为与无穷时间极限存在本质差异——极限分析不够
- 离散 SAM 更新的行为与连续时间 SAM 流高度一致，验证了理论的实用性

## 亮点与洞察

1. **"minor first, major last"现象**：揭示了 $\ell_2$-SAM 一个反直觉的有限时间行为——优化器先关注弱特征再关注强特征，这是由梯度归一化导致的
2. **有限时间 vs 无穷时间**：给出了一个明确的例子，说明仅看 $t\to\infty$ 的隐式偏差分析可能遗漏关键动态信息，呼吁更多有限时间分析
3. **深度的质变效应**：仅增加一层（$L=1\to L=2$）就能彻底改变 SAM 的行为，揭示了深度与优化算法之间的深层交互
4. **精确的轨迹刻画**：$\ell_\infty$-SAM 的坐标独立演化性质允许精确的轨迹表征，这是一个优美的理论简化

## 局限与展望

- 理论分析限于**线性对角网络**这一简化模型，与实际深度非线性网络有差距
- 多样本数据集的分析面临额外技术困难，目前仅限于实验验证
- $\ell_2$-SAM 的极限方向定理（定理 4.2）依赖方向收敛假设
- 序列特征放大对实际深度学习中的泛化有何影响仍不清楚
- 未涉及 SAM 变体（如 ASAM、GSAM）的分析

## 相关工作与启发

- **Soudry et al. (2018)**：GD 在线性模型上收敛到 $\ell_2$ 最大间隔方向的经典结果
- **Gunasekar et al. (2018)**：GD 在线性对角网络上偏向 $\ell_1$ 稀疏解
- **Pesme & Flammarion (2023)**：GD 的 saddle-to-saddle 动态——类似的阶段性学习，但机制不同
- **Foret et al. (2020)**：SAM 算法的原始论文
- 启发：有限时间分析对于理解优化器行为至关重要，未来应更多关注训练过程中的动态变化而非仅关注收敛结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — "sequential feature amplification"是一个全新且反直觉的发现
- 实验充分度: ⭐⭐⭐⭐ — 合成实验与MNIST/CNN验证，但缺乏大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，图示清晰，有限时间 vs 无穷时间的对比引人深思
- 价值: ⭐⭐⭐⭐ — 对 SAM 的理论理解有重要推进，但实际应用指导有限

<!-- RELATED:START -->

## 相关论文

- [Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)
- [Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [Generalization Below the Edge of Stability: The Role of Data Geometry](generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)

<!-- RELATED:END -->
