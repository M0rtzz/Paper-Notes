---
title: >-
  [论文解读] Robust Fitting on a Gate Quantum Computer
description: >-
  [ECCV2024][quantum computing] 首次在真实门量子计算机（IonQ Aria）上实现鲁棒拟合：提出用于一维 $\ell_\infty$ 可行性检验的量子电路，填补了 Bernstein-Vazirani（BV）电路计算 Boolean influence 的关键空缺，并展示如何将一维 influence 累积到高维非线性模型（如基础矩阵估计）。
tags:
  - ECCV2024
  - quantum computing
  - robust fitting
  - consensus maximization
  - Boolean influence
  - gate quantum computer
  - Bernstein-Vazirani circuit
---

# Robust Fitting on a Gate Quantum Computer

**会议**: ECCV2024  
**arXiv**: [2409.02006](https://arxiv.org/abs/2409.02006)  
**代码**: 待确认  
**领域**: physics  
**关键词**: quantum computing, robust fitting, consensus maximization, Boolean influence, gate quantum computer, Bernstein-Vazirani circuit

## 一句话总结

首次在真实门量子计算机（IonQ Aria）上实现鲁棒拟合：提出用于一维 $\ell_\infty$ 可行性检验的量子电路，填补了 Bernstein-Vazirani（BV）电路计算 Boolean influence 的关键空缺，并展示如何将一维 influence 累积到高维非线性模型（如基础矩阵估计）。

## 背景与动机

- **鲁棒拟合**是计算机视觉中的核心问题：从含噪声和离群值的数据中估计几何模型（如基础矩阵、单应矩阵、三维三角化等），通常采用 consensus maximization 框架
- 该问题已被证明是 NP-hard 且难以近似的，RANSAC 及其变体不提供最优性保证
- 量子计算为解决此类难题提供了替代路线：
    - **绝热量子计算（AQC）** 方面：Doan et al. 将其建模为超图顶点覆盖（HVC）在 D-Wave 量子退火器上求解；Farina et al. 用量子退火做多模型拟合
    - **门量子计算（GQC）** 方面：Chin et al. 提出用 BV 电路计算 Boolean influence 来度量离群程度，理论上可实现 $O(N)$ 级并行加速
- **关键差距**：Chin et al. 的方法假设 $\ell_\infty$ 可行性检验拥有量子实现，但从未给出具体电路设计，因此无法在真实 GQC 上验证

## 核心问题

1. 如何设计一个量子电路来实现 $\ell_\infty$ 可行性检验，从而让 BV 电路真正可运行？
2. 如何将针对一维 point fitting 计算的 Boolean influence 扩展到高维非线性鲁棒拟合任务？

## 方法详解

### Boolean Influence 回顾

给定数据集 $\mathcal{D} = \{p_i\}_{i=1}^N$，二值向量 $\mathbf{z} \in \{0,1\}^N$ 表示子集选择。定义 minimax 值：

$$g(\mathbf{z}) = \min_{\mathbf{x} \in \mathcal{M}} \max_{i: z_i=1} r_i(\mathbf{x})$$

$\ell_\infty$ 可行性检验：$f(\mathbf{z}) = 0$ 当且仅当 $g(\mathbf{z}) \le \epsilon$，即该子集中所有点到某个模型的残差均不超过阈值 $\epsilon$。

第 $i$ 个数据点的 Boolean influence 定义为翻转 $z_i$ 改变可行性的概率：

$$\alpha_i = \Pr[f(\mathbf{z}) \neq f(\mathbf{z} \oplus \mathbf{e}_i)]$$

已有理论证明：在一定条件下，内点的 influence 严格小于离群点，使其成为可靠的离群程度度量。

### BV 量子电路加速 Influence 计算

- 经典方法需对每个样本 $\mathbf{z}_j$ 逐一遍历 $N$ 个数据点，计算复杂度为 $O(MN)$
- BV 电路使用 $N+1$ 个量子比特，利用量子叠加与干涉，一次测量即可并行获取所有 $N$ 个数据点的 influence 采样
- 运行 $M$ 次即可近似 influence，采样误差界为 $\Pr(|\hat{\alpha}_i - \alpha_i| < \delta) > 1 - 2e^{-2M\delta^2}$
- **瓶颈**：BV 电路中的 oracle $U_f$ 需实现 $\ell_\infty$ 可行性检验，此前未有具体量子电路

### 本文贡献：一维 $\ell_\infty$ 可行性检验量子电路

针对**一维 point fitting** 问题（$r_i(x) = |x - b_i|$），可行性简化为：

$$f(\mathbf{z}) = 0 \iff \max(\{b_i | z_i=1\}) - \min(\{b_i | z_i=1\}) \le 2\epsilon$$

电路设计（$U_f$）包含以下关键子电路：

| 子电路 | 功能 |
|--------|------|
| **A / A⁻¹** | 从排序后的子集中选出最大值（一位热编码） |
| **B** | 从排序后的子集中选出最小值 |
| **V₁ / V₂** | 将数据数值编码到量子比特中 |
| **S₁** | 基于 QFT 的量子减法器，计算最大值与最小值之差 |
| **S₂** | 与 $2\epsilon$ 比较，输出可行性判定 |

预处理：数据降序排列，偏移使最小值为零。整个电路消耗 $3N + 2C + 1$ 个量子比特（$N$ 为数据点数，$C$ 为数据位精度）。通过 $D$ 和 $D^{-1}$（uncomputation）确保辅助比特正确恢复，满足 BV 电路要求。

### Influence 累积：扩展到基础矩阵估计

将基础矩阵估计通过线性化（8-point 方法去齐次化）转换为线性回归问题，然后通过 RANSAC 式假设采样框架累积 influence：

1. 采样 $T$ 组最小子集（各 8 个对应点），用 8-point 方法估计假设基础矩阵 $\mathbf{F}_t$
2. 计算所有数据点在 $\mathbf{F}_t$ 下的线性化残差，作为一维 point fitting 的输入数据
3. 对每个假设运用 Alg. 1/2 计算一维 influence
4. 通过对数平均累积各假设的 influence

模型估计阶段：对归一化 influence 均匀采样阈值 $\gamma_h$，对 influence $\le \gamma_h$ 的点做最小二乘拟合，返回共识集最大的模型。

## 实验关键数据

### 量子模拟器验证（Amazon Braket SV1）

- 数据 $\mathcal{D} = \{7,5,3,2\}$，$\epsilon=1$，20 个量子比特
- 随着迭代次数 $M$ 增加，近似 influence 收敛到真实值 $\{0.50, 0.25, 0.25, 0.50\}$

### 真实量子硬件验证（IonQ Aria，25 量子比特）

| 数据点数 $N$ | 位精度 $C$ | 量子比特数 | Aria 结果 | SV1 结果 | 真实值 |
|:---:|:---:|:---:|---:|---:|---:|
| 2 | 1 | 9 | 0.50, 0.56 | 0.51, 0.49 | 0.50, 0.50 |
| 3 | 3 | 16 | 0.49, 0.23, 0.53 | 0.50, 0.26, 0.52 | 0.50, 0.25, 0.50 |
| 4 | 3 | 19 | 0.48, 0.39, 0.41, 0.53 | 0.50, 0.25, 0.25, 0.49 | 0.50, 0.25, 0.25, 0.50 |

小规模电路在真实硬件上结果良好；随电路增大（接近 25 量子比特上限），误差增加，主要因为迭代次数不足（$M=50$）和缺乏纠错。

### 基础矩阵估计（Influence 累积）

在 TUM、KITTI、T&T、CPC 四个数据集上的 ROC AUC 分类得分：

| 数据集 | RANSAC | 本文 Influence (Alg.3) | Chin et al. |
|--------|:------:|:-----:|:-----:|
| TUM | 0.71 | **0.83** | 0.76 |
| KITTI | 0.86 | **0.94** | 0.86 |
| T&T | 0.72 | **0.82** | 0.82 |
| CPC | 0.75 | **0.87** | 0.85 |

SGD $\le 0.05$ 的召回率（%）：

| 数据集 | 本文方法 | RANSAC | USAC-openCV |
|--------|:-------:|:------:|:-----------:|
| TUM | **62.8** | 60.9 | 54.0 |
| KITTI | 85.4 | 87.0 | 83.8 |
| T&T | **85.1** | 77.6 | 88.1 |
| CPC | **50.3** | 42.0 | 52.5 |

## 亮点

- **首次真实 GQC 证明**：在 IonQ Aria 上首次实现量子鲁棒拟合，将理论可能性变为实际可行
- **关键空缺填补**：提出了 $\ell_\infty$ 可行性检验的量子电路设计，解决了 BV 电路缺少 oracle 实现的根本问题
- **从一维到高维的桥梁**：通过 RANSAC 式假设采样 + 对数平均累积，将简单的一维 influence 扩展到复杂的高维估计任务
- **实验验证充分**：模拟器确认正确性，真实硬件确认可行性，基准数据集确认实用性

## 局限与展望

- 当前量子电路仅适用于**一维 point fitting**，高维问题仍需经典方法辅助
- IonQ Aria 仅 25 量子比特，只能处理极小规模数据（$N \le 4$），远未达到实用规模
- 当前处于 NISQ 时代，量子噪声显著，大规模电路误差增长明显
- 电路资源消耗为 $3N + 2C + 1$ 量子比特，随数据量线性增长
- 与成熟的 RANSAC 类方法相比，量子方法暂无性能优势，价值在于探索替代技术路线
- Influence 累积依赖 RANSAC 式采样（$T=1000$ 假设），仍有较高经典计算开销

## 与相关工作的对比

| 方法 | 量子范式 | 问题建模 | 硬件验证 | 适用规模 |
|------|---------|---------|---------|---------|
| Chin et al. (2020) | GQC（BV 电路） | Boolean influence | 仅理论 | 理论上可扩展 |
| Doan et al. (2022) | AQC（D-Wave） | 超图顶点覆盖 + QUBO | D-Wave 上运行 | 受 QUBO 变量限制 |
| Farina et al. (2023) | AQC | 不相交集覆盖 + QUBO | D-Wave 上运行 | 受 QUBO 变量限制 |
| **本文** | **GQC（BV + 新 $U_f$）** | **Boolean influence** | **IonQ Aria** | 一维 $N \le 4$，可累积到高维 |

与 AQC 方法的核心差异：GQC 可实现任意量子算法（如 Shor、Grover），理论上更通用，但当前硬件限制更严格。

## 启发与关联

- 量子-经典混合框架是 NISQ 时代的务实路线：利用量子加速子问题，经典方法处理整体优化
- Influence 这一度量本身具有独立价值，可脱离量子计算框架，在经典方法中用作离群检测工具
- 一维到高维的累积策略（假设采样 + 对数平均）可能推广到其他鲁棒估计任务
- IBM 量子路线图（2029 年 200 量子比特全纠错系统）若实现，该方法可扩展到实用规模

## 评分
- 新颖性: 8/10 — 首次给出 $\ell_\infty$ 可行性检验的量子电路并在真实 GQC 上验证，填补重要理论空缺
- 实验充分度: 7/10 — 模拟器、真实硬件、基准数据集三方面验证较全面，但受硬件限制规模极小
- 写作质量: 8/10 — 问题定义清晰，理论推导严谨，量子-经典方法对比合理
- 价值: 6/10 — 目前实际应用价值有限，但作为量子计算在视觉中的探索具有前瞻意义

<!-- RELATED:START -->

## 相关论文

- [Quantum Doubly Stochastic Transformers](../../NeurIPS2025/physics/quantum_doubly_stochastic_transformers.md)
- [Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [Data Verification is the Future of Quantum Computing Copilots](../../AAAI2026/physics/data_verification_is_the_future_of_quantum_computing_copilots.md)
- [Feedback-driven Recurrent Quantum Neural Network Universality](../../ICLR2026/physics/feedback-driven_recurrent_quantum_neural_network_universality.md)
- [Compact Matrix Quantum Group Equivariant Neural Networks](../../ICML2025/physics/compact_matrix_quantum_group_equivariant_neural_networks.md)

<!-- RELATED:END -->
