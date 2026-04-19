---
title: >-
  [论文解读] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification
description: >-
  [ICML2025][优化][SDP松弛] 本文首次识别了SDP松弛用于深度神经网络验证时的"内点消失"(interior-point vanishing)问题——随着网络深度增加，SDP问题丧失严格可行性导致数值不稳定和求解失败——并提出五种缓解方法，其中B-Remove（移除层边界约束）最有效，解决了88%原本无法求解的问题。
tags:
  - ICML2025
  - 优化
  - SDP松弛
  - 神经网络验证
  - 内点法
  - 严格可行性
  - ReLU
  - 对抗鲁棒性
---

# Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification

**会议**: ICML2025  
**arXiv**: [2506.10269](https://arxiv.org/abs/2506.10269)  
**代码**: 未公开  
**领域**: 神经网络验证 / 优化  
**关键词**: SDP松弛, 神经网络验证, 内点法, 严格可行性, ReLU, 对抗鲁棒性

## 一句话总结

本文首次识别了SDP松弛用于深度神经网络验证时的"内点消失"(interior-point vanishing)问题——随着网络深度增加，SDP问题丧失严格可行性导致数值不稳定和求解失败——并提出五种缓解方法，其中B-Remove（移除层边界约束）最有效，解决了88%原本无法求解的问题。

## 研究背景与动机

**DNN验证问题**：给定分类模型 $\boldsymbol{f}: \mathbb{R}^d \to \mathbb{R}^m$、输入 $\bar{\boldsymbol{x}}$ 及扰动半径 $\rho$，判定在 $\|\boldsymbol{x}_0 - \bar{\boldsymbol{x}}\|_\infty \le \rho$ 内预测标签是否不变。这等价于求解优化问题：

$$\gamma^* = \min_{\{\boldsymbol{x}_i\}} \boldsymbol{c}^\top \boldsymbol{x}_L + c_0 \quad \text{s.t. ReLU约束、扰动约束、边界约束}$$

当 $\gamma^* > 0$ 时模型对该输入鲁棒。由于ReLU使问题非凸（NP-hard），实践中采用**凸松弛**近似求解。

**SDP松弛的优势与困境**：SDP松弛通过polynomial lifting引入矩阵变量 $\boldsymbol{P} = \boldsymbol{v}\boldsymbol{v}^\top$ 并放松rank-1约束，被公认为凸松弛中最紧的方法之一。然而SDP方法在SOTA验证器中几乎不被使用，通常归因于计算开销大。本文揭示了一个**更根本的原因**：当网络深度增加时，SDP问题的严格可行性（Slater条件）丧失，导致：

- 强对偶性不成立 → 原始-对偶间隙不收敛到零
- 内点法数值不稳定 → 求解器返回错误结果或直接失败
- 一阶方法更为严重，因为缺乏二阶Hessian信息

## 方法详解

### 内点消失的理论分析

**定义**：内点消失指SDP验证问题随网络深度增加而失去可行内点（$\boldsymbol{P} \succ \boldsymbol{O}$ 的解不存在）。

**原因一：不活跃神经元**。若某神经元上界 $(u_i)_j = 0$，则由边界约束 (5e) 和半正定约束可推出 $(\boldsymbol{P}[\boldsymbol{x}_i \boldsymbol{x}_i^\top])_{jj} = 0$，由Proposition 3.3知 $\lambda_{\min}(\boldsymbol{P}) = 0$，即 $\boldsymbol{P}$ 不可能正定。一个不活跃神经元就足以触发内点消失，且预处理无法完全消除所有不活跃神经元。

**原因二：权重范数约束**（Theorem 3.6）。定义 $\widetilde{\boldsymbol{W}}_i = (\boldsymbol{b}_i \; \boldsymbol{W}_i)$，递推常数 $T_0 = (\|\bar{\boldsymbol{x}}\|_2 + \rho\sqrt{n_0})^2$，$T_{i+1} = (1+T_i)\|\widetilde{\boldsymbol{W}}_i\|_F^2$，则：

$$\lambda_{\min}(\boldsymbol{P}) \le \min_{i \in [L]} \min_{j} (1+T_i) \cdot \|\widetilde{\boldsymbol{W}}_i(j,:)\|_2^2$$

当任一层存在小范数行向量时，所有可行解的最小特征值被压到接近零。

**严格可行性验证**（Proposition 3.2）：引入辅助变量 $\lambda$，令 $\boldsymbol{X} + \lambda\boldsymbol{I}$ 替代原变量，最大化 $\lambda$；若最优值 $\lambda^* > 0$ 则原问题严格可行。

### 五种缓解方法

| 方法 | 核心思路 | 修改的约束 |
|------|---------|-----------|
| **ε-SDP** | 将ReLU等式约束松弛为 $\pm\varepsilon$ 容差的不等式 | (5d) → $\|...\| \le \varepsilon$ |
| **LeakySDP** | 用Leaky ReLU（$\alpha \cdot x, x<0$）替代ReLU约束 | (5b)(5c)(5d) → Leaky ReLU约束 |
| **D-Scale** | 对角缩放 $\boldsymbol{D} = \text{diag}(1, \boldsymbol{u}_1, ..., \boldsymbol{u}_L)$ | 等价变换所有约束矩阵 |
| **W-Scale** | 基于Theorem 3.6，按最小行范数缩放权重矩阵 | $\widetilde{\boldsymbol{W}}_i' = \widetilde{\boldsymbol{W}}_i / \check{w}_i$ |
| **B-Remove** | 直接移除中间层的上下界约束（$i > 0$ 的 $\boldsymbol{l}_i, \boldsymbol{u}_i$）| 删除 (5e) 中 $i > 0$ 的部分 |

**关键洞察**：B-Remove之所以有效，是因为中间层的上下界约束对SDP松弛并非必要——SDP松弛ReLU本身不需要这些边界（见公式(2)-(3)）。这些约束源自LP松弛的"三角区域"，被先前工作不加甄别地沿用，实际上不仅无益反而有害。

## 实验关键数据

**实验设置**：MNIST数据集，全连接ReLU网络，层数 $L \in \{2,4,...,16\}$，每层20个神经元，5个随机种子，10张图片。对比方法：SDP-IP、LayerSDP。

### 求解成功率（%）

| 深度L | ε-SDP | LeakySDP | B-Remove | D-Scale | W-Scale | LayerSDP | SDP-IP |
|-------|-------|----------|----------|---------|---------|----------|--------|
| 2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 6 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| 8 | 100 | 82 | 100 | 88 | 88 | 82 | 98 |
| 10 | 100 | 4 | 100 | 8 | 8 | 2 | 14 |
| 12 | 78 | 0 | **100** | 0 | 0 | 0 | 0 |
| 16 | 24 | 0 | **66** | 0 | 0 | 0 | 0 |

### 严格可行性实证（高精度SDPA-GMP，512位）

| 深度L | MNIST Solved(%) | Avg. Obj. | FashionMNIST Solved(%) | Avg. Obj. |
|-------|-----------------|-----------|------------------------|-----------|
| 2 | 98% | 2.13E-05 | 100% | 5.79E-05 |
| 8 | 98% | 3.52E-09 | 94% | 4.98E-09 |
| 10 | 18% | **-4.09E-10** | 26% | **-2.57E-10** |
| 16 | **0%** | -1.20E-09 | **0%** | -9.35E-10 |

$L \ge 10$ 时，最优值趋近零甚至为负（数值误差），验证了严格可行性完全丧失。

### 松弛质量

B-Remove几乎不损失验证能力（与LayerSDP目标函数值差距极小），因为中间层边界约束本就不影响SDP松弛质量。ε-SDP验证能力损失相对较大。

## 亮点与洞察

1. **首次识别内点消失问题**：揭示了SDP验证方法在深度网络上失败的根本原因，不仅是计算开销
2. **理论与实证双重验证**：通过最小特征值界（Theorem 3.6）和高精度求解器实验，严格证明了问题的普遍性
3. **"有效约束反而有害"的反直觉发现**：中间层上下界约束来自LP松弛的惯例，在SDP中不仅不提升松弛质量，还摧毁严格可行性
4. **B-Remove的极简有效性**：仅删除非必要约束就解决88%的失败案例，且几乎不损失验证精度——方法越简单越深刻

## 局限与展望

1. **实验规模有限**：仅在20个神经元/层的小网络上验证，实际验证场景网络更大（100-1000+神经元/层）
2. **仅针对全连接ReLU网络**：未涉及卷积网络、残差连接、其他激活函数
3. **SDP求解器本身的可扩展性瓶颈**：即使解决了内点消失，SDP的 $O(n^6)$ 计算复杂度仍是实用化的根本障碍
4. **缺乏与SOTA验证器的端到端对比**：如α,β-CROWN + BaB，实际中是否因初始松弛更紧而在整体pipeline中受益？
5. **B-Remove的理论保证**：虽然实验效果好，但缺乏关于松弛质量损失的理论下界分析

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统识别并分析内点消失问题，反直觉发现有价值
- 实验充分度: ⭐⭐⭐ — 理论分析扎实但实验规模偏小，缺乏大网络验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论推导严谨，图表直观
- 价值: ⭐⭐⭐⭐ — 对SDP验证领域有重要理论贡献，但实用影响可能受限于SDP本身的可扩展性

<!-- RELATED:START -->

## 相关论文

- [SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)
- [The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product](emergence_in_non-neural_models_grokking_modular_arithmetic_via_average_gradient_.md)

<!-- RELATED:END -->
