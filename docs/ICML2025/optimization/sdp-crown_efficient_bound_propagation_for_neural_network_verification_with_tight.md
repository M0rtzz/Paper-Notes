---
title: >-
  [论文解读] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming
description: >-
  [ICML2025][优化][神经网络验证] 提出 SDP-CROWN，将半定规划（SDP）松弛的紧致性融入线性界传播框架，每层仅增加一个参数 λ，便可在 ℓ₂ 扰动下将验证松弛度最多收紧 √n 倍，同时保持与 α-CROWN 同级的可扩展性。
tags:
  - ICML2025
  - 优化
  - 神经网络验证
  - 半定规划
  - 线性界传播
  - ℓ₂ 鲁棒性
  - ReLU 松弛
  - CROWN
---

# SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming

**会议**: ICML2025  
**arXiv**: [2506.06665](https://arxiv.org/abs/2506.06665)  
**代码**: [Hong-Ming/SDP-CROWN](https://github.com/Hong-Ming/SDP-CROWN)  
**领域**: optimization / neural network verification  
**关键词**: 神经网络验证, 半定规划, 线性界传播, ℓ₂ 鲁棒性, ReLU 松弛, CROWN

## 一句话总结

提出 SDP-CROWN，将半定规划（SDP）松弛的紧致性融入线性界传播框架，每层仅增加一个参数 λ，便可在 ℓ₂ 扰动下将验证松弛度最多收紧 √n 倍，同时保持与 α-CROWN 同级的可扩展性。

## 研究背景与动机

- **线性界传播**（CROWN / α-CROWN / β-CROWN）是当前最可扩展的神经网络验证方法，在 VNN-COMP 竞赛中表现优异，尤其擅长 ℓ∞ 扰动。
- **核心痛点**：面对 ℓ₂ 扰动时，界传播方法必须先将 ℓ₂ 球放松为外接 ℓ∞ 盒，这会让攻击半径放大 √n 倍（n 为输入维度），导致界极度松弛。例如在 ConvLarge（≈2.47M 参数、65k 神经元）上，α-CROWN 仅验证 2.5% 准确率。
- **SDP 方法**可通过 n×n 耦合矩阵精确捕捉神经元间依赖关系，但 $O(n^3)$ 复杂度限制了其只能应用于小规模模型（<10k 神经元）。
- **目标**：在不牺牲可扩展性的前提下，把 SDP 的紧致性注入界传播框架。

## 方法详解

### 核心思路

标准界传播为目标函数 $c^T f(x)$ 构造线性松弛 $g(\alpha)^T x + h(\alpha)$，其中 $h(\alpha) = -\rho \|\min\{g(\alpha), 0\}\|_1$（基于 ℓ∞ 盒）。SDP-CROWN 将偏移量替换为基于 ℓ₂ 球的新表达式 $h(g,\lambda)$，同时保持斜率 $g(\alpha)$ 不变。

### 主定理（Theorem 4.1）

对任意 $\lambda \ge 0$ 和 $g \in \mathbb{R}^n$：

$$c^T \operatorname{ReLU}(x) \ge g^T x + h(g,\lambda), \quad \forall\, x \in \mathcal{B}_2(\hat{x}, \rho)$$

其中：

$$h(g,\lambda) = -\frac{1}{2}\left(\lambda(\rho^2 - \|\hat{x}\|_2^2) + \frac{1}{\lambda}\|\phi(g,\lambda)\|_2^2\right)$$

$$\phi_i(g,\lambda) = \min\{c_i - g_i - \lambda\hat{x}_i,\; g_i + \lambda\hat{x}_i,\; 0\}$$

### 紧致性保证（Theorem 5.2）

当 $\hat{x}=0$ 时，最优 λ 下的偏移量为：

$$h^* = -\rho \|\min\{c-g, g, 0\}\|_2$$

相比传统界传播的 $h = -\rho \|\min\{g, 0\}\|_1$，由 ℓ₁ 范数变为 ℓ₂ 范数，直接获得最多 **√n 倍**的紧致性提升。且 SDP 松弛本身是精确紧的（Theorem 5.3），即 SDP 对偶值等于原始非凸问题最优值。

### 实现细节

1. 先用标准 α-CROWN 在 ℓ∞ 外接盒上计算斜率 $g(\alpha)$ 和初始偏移 $h(\alpha)$
2. 用 Theorem 4.1 中的公式替换偏移为 $h(g(\alpha), \lambda)$
3. 对 α 和 λ 联合做投影梯度上升求最紧界
4. **每层仅增加 1 个标量参数 λ**，相比 SDP 的 n² 参数极大降低开销

## 实验关键数据

在 MNIST 和 CIFAR-10 上对比多种验证器，200 张图片的 **ℓ₂ 验证准确率**（%）：

| 模型 | 参数 | SDP-CROWN（本文） | α-CROWN | β-CROWN | GCP-CROWN | BICCOS | LipNaive | 上界 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MNIST MLP | ρ=1.0 | **32.5%** (2.5s) | 1.5% (1.2s) | 36% (302s) | 38% (198s) | 41% (173s) | 29% | 54% |
| MNIST ConvSmall | ρ=0.3 | **81.5%** (12s) | 0% (17s) | 16% (257s) | 17% (265s) | 19.5% (248s) | 77.5% | 84.5% |
| MNIST ConvLarge | ρ=0.3 | **79.5%** (88s) | 0% (66s) | 0% (307s) | 0% (304s) | 0% (309s) | 77% | 84% |
| CIFAR ConvLarge | ρ=8/255 | **63.5%** (73s) | 2.5% (47s) | 5% (289s) | 6% (282s) | 6% (286s) | 47.5% | 72.5% |
| CIFAR CNN-A | ρ=24/255 | **49%** (12s) | 7.5% (3.8s) | 20% (201s) | 20% (224s) | 20% (210s) | 39% | 55.5% |
| CIFAR CNN-B | ρ=24/255 | **49.5%** (16s) | 0% (8.7s) | 3% (193s) | 3% (302s) | 3% (290s) | 33% | 59.5% |

**关键发现**：

- 在 ConvLarge（65k 神经元）上，α-CROWN/β-CROWN/BICCOS 的验证准确率几乎为 0%，而 SDP-CROWN 达到 63.5%–79.5%
- SDP-CROWN 比 LipNaive 基线高出 2%–16% 验证准确率，且远优于基于分支定界的方法
- 传统 SDP (BM-Full/LP-All) 在中大规模模型上无法运行（标记为 "-"），SDP-CROWN 成功扩展到 2.47M 参数
- 运行时间适中：ConvLarge 上 73–88s/样本，远低于 BICCOS 的 173–309s

## 亮点与洞察

1. **理论优雅**：从 SDP 对偶推导出仅含一个额外参数 λ 的闭式线性界，证明其在 $\hat{x}=0$ 时精确紧
2. **√n 倍提升有理论保证**：将 ℓ₁ 范数界改为 ℓ₂ 范数界，精确对应 ℓ₂ 球与 ℓ∞ 盒的体积比
3. **即插即用**：可无缝集成到任何线性界传播管线中（CROWN、α-CROWN 等），不需要改变底层架构
4. **首次将 SDP 松弛高效应用于 ℓ₂ 验证**，填补了大规模 ℓ₂ 鲁棒性认证的空白
5. **对比惊艳**：在 ℓ₂ 扰动下，β-CROWN 和 BICCOS 使用复杂的分支定界与切割平面仍几乎为 0% 验证率，而 SDP-CROWN 仅通过一个参数即达到 60%+ 验证率
6. **LipNaive 基线的启示**：简单的逐层 Lipschitz 常数乘积在 ℓ₂ 下反而优于复杂的界传播方法，说明 ℓ₂ 场景下神经元耦合的建模至关重要

## 局限与展望

1. **理论分析限于 $\hat{x}=0$**：一般情况（$\hat{x} \neq 0$）下的紧致性需通过实验验证，缺乏解析保证
2. **仅考虑 ReLU 激活函数**：对 GeLU、Swish 等非分段线性激活尚未讨论扩展方式
3. **λ 为每层标量**：更细粒度的逐神经元 λ 可能进一步提升效果，但会增加参数量
4. **对 ℓ∞ 扰动无额外增益**：该方法专为 ℓ₂ 场景设计，ℓ∞ 下退化为标准界传播
5. **与分支定界的结合未深入探索**：β-CROWN 的 split + SDP-CROWN 的 λ 联合优化值得研究
6. **缺少大规模视觉模型验证**：实验最大模型仅 2.47M 参数，未测试 ResNet / ViT 等现代架构
7. **未探讨对抗训练协同**：SDP-CROWN 的紧致界是否可反馈用于 ℓ₂ 对抗训练值得研究

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — SDP 与界传播融合的全新视角，闭式推导优雅
- 实验充分度: ⭐⭐⭐⭐ — 涵盖多架构多数据集，与主流验证器全面对比；缺少更大规模（如 ImageNet）验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，Figure 1/2 直观展示核心思想
- 价值: ⭐⭐⭐⭐⭐ — 解决了 ℓ₂ 验证的重要瓶颈，实用价值高

## 相关工作

- **SDP 验证**: Raghunathan et al. 2018, Brown et al. 2022, Chiu & Zhang 2023 — 紧致但 $O(n^3)$ 不可扩展
- **界传播**: CROWN (Zhang 2018), α-CROWN (Xu 2021), β-CROWN (Wang 2021) — 可扩展但 ℓ₂ 下松弛
- **Lipschitz 方法**: LipSDP (Fazlyab 2019), 1-Lipschitz 网络 (Singla & Feizi 2022) — 全局常数过于保守
- **切割平面**: GCP-CROWN (Zhang 2022), BICCOS (Zhou 2024) — 增加 MIP 约束但对大 ℓ₂ 扰动仍失效

<!-- RELATED:START -->

## 相关论文

- [Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)
- [The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [Integer Programming for Generalized Causal Bootstrap Designs](integer_programming_for_generalized_causal_bootstrap_designs.md)
- [Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)

<!-- RELATED:END -->
