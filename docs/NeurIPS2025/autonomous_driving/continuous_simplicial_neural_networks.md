---
title: >-
  [论文解读] Continuous Simplicial Neural Networks
description: >-
  [NEURIPS2025][自动驾驶][Simplicial Neural Networks] 提出 COSIMO，首个基于偏微分方程（PDE）的连续单纯形神经网络，通过在 Hodge Laplacian 上定义热扩散动力学实现连续信息流，比离散 SNN 具有更好的稳定性和过平滑控制能力。
tags:
  - NEURIPS2025
  - 自动驾驶
  - Simplicial Neural Networks
  - 偏微分方程
  - Over-smoothing
  - Hodge Laplacian
  - 拓扑深度学习
---

# Continuous Simplicial Neural Networks

**会议**: NEURIPS2025  
**arXiv**: [2503.12919](https://arxiv.org/abs/2503.12919)  
**代码**: [ArefEinizade2/COSIMO](https://github.com/ArefEinizade2/COSIMO)  
**领域**: autonomous_driving  
**关键词**: Simplicial Neural Networks, PDE, Over-smoothing, Hodge Laplacian, 拓扑深度学习  

## 一句话总结

提出 COSIMO，首个基于偏微分方程（PDE）的连续单纯形神经网络，通过在 Hodge Laplacian 上定义热扩散动力学实现连续信息流，比离散 SNN 具有更好的稳定性和过平滑控制能力。

## 背景与动机

- 图神经网络（GNN）只能建模节点间的成对交互，无法刻画高阶关系（如三角形、四面体等多体交互）
- 单纯形复形（Simplicial Complex）通过引入 k-单纯形和 Hodge Laplacian 扩展了图的表达能力，但现有的单纯形神经网络（SNN）主要依赖**离散滤波**（矩阵多项式），存在两个核心限制：
    - **滤波器阶数需手动调参**：离散多项式阶数 $T_d, T_u$ 是超参数，调参成本高
    - **过平滑控制困难**：随层数增加特征趋同，且离散 SNN 只能通过修改拓扑结构来缓解，不可行
- 在 GNN 领域，连续模型（如 Neural ODE / 图扩散方程）已被证明能更好地控制过平滑并提升对结构扰动的鲁棒性，但**连续 SNN 尚未被探索**

## 核心问题

如何设计一种连续的单纯形神经网络，使其：(1) 具备动态可学习的感受野而非固定的多项式阶数；(2) 对拓扑扰动保持稳定；(3) 能有效控制过平滑速率？

## 方法详解

### 单纯形复形基础

- **k-单纯形**：0-单纯形=节点，1-单纯形=边，2-单纯形=三角形
- **关联矩阵** $\mathbf{B}_k$：描述 $(k-1)$-单纯形与 $k$-单纯形之间的关联关系
- **Hodge Laplacian**：$\mathbf{L}_k = \mathbf{B}_k^\top \mathbf{B}_k + \mathbf{B}_{k+1}\mathbf{B}_{k+1}^\top$，分解为下 Laplacian $\mathbf{L}_{k,d}$ 和上 Laplacian $\mathbf{L}_{k,u}$
- **Dirichlet 能量**：$E(\mathbf{x}_k) = \mathbf{x}_k^\top \mathbf{L}_k \mathbf{x}_k$，衡量信号的平滑程度

### COSIMO 的 PDE 体系

核心思想是在解耦的上下 Hodge Laplacian 上定义热扩散 PDE，实现连续时间的信息传播：

1. **独立下扩散**：$\frac{\partial \mathbf{x}_{k,d}(t_d)}{\partial t_d} = -\mathbf{L}_{k,d} \mathbf{x}_{k,d}(t_d)$
2. **独立上扩散**：$\frac{\partial \mathbf{x}_{k,u}(t_u)}{\partial t_u} = -\mathbf{L}_{k,u} \mathbf{x}_{k,u}(t_u)$
3. **联合扩散**：耦合上下空间的交互动力学
4. **积分输出**：组合独立和联合动力学的解

### COSIMO 层的定义

PDE 的解析解为矩阵指数形式，第 $l$ 层定义为：

$$\mathbf{X}_k^l = \sigma\left(e^{-t_d \mathbf{L}_{k,d}} \mathbf{X}_{k,d}^{l-1} \boldsymbol{\Theta}_{k,d}^l + e^{-t_u \mathbf{L}_{k,u}} \mathbf{X}_{k,u}^{l-1} \boldsymbol{\Theta}_{k,u}^l + e^{-t_d \mathbf{L}_{k,d}} \mathbf{X}_k^{l-1} \boldsymbol{\Psi}_{k,d}^l + e^{-t_u \mathbf{L}_{k,u}} \mathbf{X}_k^{l-1} \boldsymbol{\Psi}_{k,u}^l\right)$$

- $t_d, t_u$ 是**可学习的连续感受野参数**（核心优势），取代了离散滤波器中需手动调的阶数
- 支持多分支聚合（$M$ 个分支）以增强表达能力

### 高效实现

利用 Hodge Laplacian 的特征值分解（EVD），取前 $K$ 个主导特征对近似矩阵指数，将复杂度从 $\mathcal{O}(|\mathcal{X}_k|^3)$ 降至 $\mathcal{O}(|\mathcal{X}_k|^2 (K_k^{(d)} + K_k^{(u)} + K_k))$。

### 稳定性分析

在关联矩阵存在加性扰动 $\tilde{\mathbf{B}}_k = \mathbf{B}_k + \mathbf{E}_k$ 时，COSIMO 输出误差有界：

$$\delta_{\mathbf{X}_k} \leq t_d \delta_{k,d} e^{t_d \delta_{k,d}}(\|\mathbf{x}_{k,d}(0)\| + \|\mathbf{x}_k(0,0)\|) + t_u \delta_{k,u} e^{t_u \delta_{k,u}}(\|\mathbf{x}_{k,u}(0)\| + \|\mathbf{x}_k(0,0)\|)$$

当扰动足够小时，$\delta_{\mathbf{X}_k} = \mathcal{O}(\epsilon_k) + \mathcal{O}(\epsilon_{k+1})$，推广了连续 GNN 的稳定性结果。

### 过平滑分析

- **离散 SNN**：Dirichlet 能量上界仅受拓扑结构 $\tilde{\lambda}_{\max}$ 控制，需修改拓扑才能缓解过平滑
- **COSIMO**：上界中引入了 $e^{-2\varphi}$ 衰减因子（$\varphi = \min_k\{t_d \lambda_{\min}(\mathbf{L}_{k,d}), t_u \lambda_{\min}(\mathbf{L}_{k,u})\}$），通过调小感受野参数 $t$ 即可减缓过平滑速率，无需修改拓扑

## 实验关键数据

| 任务 | 数据集 | COSIMO | 最强基线 |
|------|--------|--------|----------|
| 轨迹预测 | ocean-drifts | **0.550** | SCCNN 0.545 |
| 网格回归 | Shrec-16 (small) | **0.010** MSE | SCCNN 0.020 |
| 网格回归 | Shrec-16 (full) | **0.027** MSE | SCCNN 0.063 |
| 节点分类 | high-school | **0.90** | SCCNN/GSAN 0.88 |
| 节点分类 | senate-bills | **0.69** | GCN 0.67 |
| 图分类 | proteins | **0.79** | SaNN/GSAN 0.77 |

- 过平滑实验验证：$t=10^{-2}$ 时 COSIMO 的过平滑速率慢于离散 SNN；增大 $t$ 则加快
- 稳定性实验：在 SNR 从 -5dB 到 20dB 变化时，模型性能稳定衰减，验证了理论界

## 亮点

- **首创性**：首个在单纯形复形上定义连续 PDE 动力学的神经网络，填补了拓扑深度学习在连续模型方面的空白
- **理论扎实**：同时给出了稳定性界和过平滑收敛率的严格分析，且均有实验验证
- **感受野可学习**：$t_d, t_u$ 可作为可学习参数，避免了离散方法中多项式阶数的超参搜索
- **Shrec-16 上的显著优势**：MSE 较 SCCNN 降低了 50%+，展示了连续模型在网格处理中的潜力

## 局限与展望

- **EVD 开销大**：虽然通过截断特征值降低了复杂度，但对大规模单纯形复形仍需 $\mathcal{O}(K N^2)$ 的预处理
- **作者提到的未来方向**：探索非负矩阵分解、Cholesky 分解或隐式 Euler 方法来替代 EVD
- **轨迹预测提升有限**：在 synthetic 数据集上不如 SCNN，ocean-drifts 上仅略优于 SCCNN
- **连续时间参数的物理意义**：$t_d, t_u$ 的可解释性和最优取值范围缺乏深入讨论
- **未涉及异质/动态拓扑**：当前假设固定拓扑结构，对时变单纯形复形的扩展尚不清楚

## 与相关工作的对比

| 方法 | 类型 | 感受野 | 过平滑控制 | 稳定性分析 |
|------|------|--------|-----------|-----------|
| SNN/SCNN | 离散滤波 | 固定阶数 | 困难（需改拓扑） | 无 |
| SCCNN | 离散 Hodge-aware | 固定阶数 | 部分分析 | 有（离散） |
| 连续 GNN | 图上 PDE | 可学习时间 | 可控 | 有 |
| **COSIMO** | **单纯形上 PDE** | **可学习 $t_d, t_u$** | **可控（调 $t$）** | **有（推广到高阶）** |

- COSIMO 是连续 GNN 从图到单纯形复形的自然推广，核心贡献在于处理上下 Hodge Laplacian 的解耦 PDE
- 与 SCCNN 最为对标：COSIMO 用矩阵指数 $e^{-t\mathbf{L}}$ 替代了 SCCNN 的矩阵多项式 $\sum \alpha_i \mathbf{L}^i$

## 启发与关联

- 连续化思路具有通用性：可尝试将类似的 PDE 框架扩展到 cell complex、hypergraph 等其他高阶结构
- 过平滑的控制机制（通过感受野参数 $t$）为深层拓扑模型设计提供了实用指导
- 在自动驾驶场景中，轨迹预测的单纯形建模（节点=位置、边=路径、三角形=区域）是一个有潜力的应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个连续 SNN，理论框架完整
- 实验充分度: ⭐⭐⭐⭐ — 覆盖多任务+理论验证实验，但轨迹预测提升有限
- 写作质量: ⭐⭐⭐⭐ — 数学表述清晰，理论和实验组织合理
- 价值: ⭐⭐⭐⭐ — 为拓扑深度学习的连续化奠定了基础

<!-- RELATED:START -->

## 相关论文

- [Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](../../ICCV2025/autonomous_driving/unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](../../CVPR2025/autonomous_driving/solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [I-INR: Iterative Implicit Neural Representations](../../AAAI2026/autonomous_driving/i-inr_iterative_implicit_neural_representations.md)
- [Neural Volumetric World Models for Autonomous Driving](../../ECCV2024/autonomous_driving/neural_volumetric_world_models_for_autonomous_driving.md)
- [Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2025/autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)

<!-- RELATED:END -->
