---
title: >-
  [论文解读] Deep Learning for Subspace Regression
description: >-
  [ICLR2026][科学计算][subspace regression] 将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。
tags:
  - ICLR2026
  - 科学计算
  - subspace regression
  - Grassmann manifold
  - reduced order modeling
  - 神经算子
  - eigenspace
---

# Deep Learning for Subspace Regression

**会议**: ICLR2026  
**arXiv**: [2509.23249](https://arxiv.org/abs/2509.23249)  
**代码**: [GitHub](https://github.com/VLSF/subreg)  
**领域**: 科学计算  
**关键词**: subspace regression, Grassmann manifold, reduced order modeling, neural operator, eigenspace

## 一句话总结

将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。

## 研究背景与动机

**领域现状**：缩减阶建模（Reduced Order Modelling, ROM）通过识别线性子空间来丢弃无信息的自由度，从而简化系统的分析和模拟。这一思路在参数化偏微分方程（PDE）、最优控制、不确定性量化等反复求解相关问题的场景中极具价值。典型方法如 Proper Orthogonal Decomposition（POD）通过对代表性参数的解做最佳低秩逼近来构建缩减基。

**现有痛点**：当子空间显式依赖参数时（即 local POD），需要从已知参数对应的子空间"插值"到未知参数对应的子空间。然而，Grassmann 流形上的经典插值方法（如 Riemannian 正规坐标插值）在高维参数空间中极不可靠——观测稀疏导致切空间近似质量低下，且计算代价随维度快速增长。

**核心矛盾**：子空间数据具有特殊的代数结构——右 $\text{GL}(k)$ 不变性，即矩阵 $W_1$ 和 $W_2$ 若列空间相同则等价。这使得标准的 $\ell_2$ 回归损失完全不适用。同时，参数化特征问题的复杂度（不同子空间配置数）随参数维度和特征值序号急剧增长，直接学习单个特征向量面临组合爆炸。

**本文方案**：将插值问题放松为回归问题，设计满足 $\text{GL}(k)$ 不变性的损失函数，用神经网络参数化高维参数到子空间的映射。核心创新是引入**子空间嵌入**——预测包含目标子空间的更大维度子空间，理论证明可降低映射的导数范数（更平滑），从而契合神经网络的 F-principle（频谱偏差），大幅提升精度。

## 方法详解

### 整体框架

给定参数 $r \in \mathbb{R}^p$ 及其对应的 $k$ 维子空间 $V(r) \in \text{Gr}(k, n)$ 的数据集 $\mathcal{D} = \{(r_i, V_i)\}_{i=1}^m$，用神经网络 $Y_\theta: \mathbb{R}^p \to \text{Gr}(r, n)$（允许 $r \geq k$）逼近映射 $V(r)$，通过最小化子空间损失函数来训练：

$$\theta^\star = \arg\min_\theta \frac{1}{m} \sum_{i=1}^m L(Y_\theta(r_i), V_i)$$

其中损失函数 $L$ 需满足两个关键性质：（1）对等价类的不变性 $L(A,B) = L(\tilde{A}, \tilde{B})$，对任意 $\tilde{A} \in \lceil A \rceil, \tilde{B} \in \lceil B \rceil$；（2）$L(A,B) = 0$ 当且仅当 $\mathcal{S}(B) \subset \mathcal{S}(A)$。网络架构采用 FFNO（Factorized Fourier Neural Operator），输出经过处理后映射到 Grassmann 流形上的元素。

### 关键设计一：子空间损失函数

**Theorem 1** 给出了两种满足不变性要求的损失函数：

- **确定性损失 $L_1$**：$L_1(A, B) = p - \|Q_B^\top Q_A\|_F^2$，其中 $A = Q_A R_A$, $B = Q_B R_B$ 为 QR 分解。本质上度量正交投影子之差的 Frobenius 范数。当且仅当 $\mathcal{S}(B) \subset \mathcal{S}(A)$ 时 $L_1 = 0$。

- **随机化损失 $L_2$**：$L_2(A, B; z) = \min_u \|Au - Q_B z\|_2^2$，其中 $z \sim \mathcal{N}(0, I_k)$。用最小二乘问题替换投影子计算，结合 Hutchinson 迹估计消除第二个 QR 分解。关键理论结果：$\mathbb{E}_z[L_2] = L_1$。

$L_2$ 的优势在于可通过正规方程求解最小二乘问题，在子空间维度增大时训练效率远优于需要 QR 分解的 $L_1$。但 $L_2$ 基于正规方程可能产生数值不稳定，引入 Cholesky-QR2 稳定化后（$L_2^{\text{stab}}$）可恢复精度。

### 关键设计二：子空间嵌入（Subspace Embedding）

本文最核心的创新。策略很简单：允许模型预测 $r > k$ 维的子空间来逼近 $k$ 维目标，只要 $\mathcal{S}(V) \subset \mathcal{S}(Y_\theta)$ 即可。这一"冗余"策略从两个角度获得理论支撑：

**Theorem 2（平滑性降低）**：对于连续可微函数 $V: \mathbb{R} \to \text{Gr}(k, n)$，总可以构造 $W: \mathbb{R} \to \text{Gr}(r, n)$（$r > k$）使得 $\|\dot{W}(t)\|_F^2 \leq \|\dot{V}(t)\|_F^2$，且在非零导数点处严格小于。这意味着嵌入到更大子空间后映射变得更平滑，完美契合神经网络的 F-principle——网络倾向学习低频/平滑的函数。

**Theorem 3（复杂度降低）**：对于常系数椭圆特征问题 $-\sum a_i \partial^2 \phi / \partial x_i^2 = \lambda \phi$，参数到第 $k$ 个特征向量的映射是分段常数函数，不同常数区域数为 $\#_{F_k}(k, D) \sim \frac{1}{(D-1)!} k(\log k)^{D-1}$，而子空间映射的区域数更大：$\#_{G_k} \geq \frac{1}{(D-1)!} k^{D-1}$。但子空间嵌入可利用"有限种特征向量的不同组合"来减少区域数——极端情况下，若预测维度等于所有可能特征向量数 $\#_{F_k}$，映射退化为常数函数。

### 关键设计三：多应用场景的统一覆盖

子空间回归框架统一覆盖了多个科学计算核心场景：

| 应用场景 | 映射关系 | 子空间含义 |
|---------|---------|-----------|
| 参数化特征问题 | $U(x) \to \text{span}\{\phi_1, \dots, \phi_K\}$ | 前 $K$ 个特征函数 |
| 本地 POD（Burgers 方程） | PDE 参数 $\to \{\psi_1, \dots, \psi_k\}$ | POD 缩减基 |
| 共轭梯度 Deflation | $k(x) \to \mathcal{S}(V)$ | 小特征值特征空间 |
| 粗网格修正（Jacobi） | $k(x) \to \mathcal{S}(V)$ | 误差传播矩阵主特征空间 |
| 平衡截断（最优控制） | $A, B, C \to \mathcal{S}(\bar{\mathcal{T}})$ | 可控可观性约简基 |

## 实验关键数据

### 量子力学特征空间预测

| 数据集 | Riemannian 插值 | $\mathbb{Z}_2$-adjusted $\ell_2$ | 子空间回归 $L_1$ |
|--------|:--------:|:--------:|:--------:|
| $D=1$ Schrödinger | 4.69% | 2.33% | **0.09%** |
| $D=2$, 数据集 a | 31.9% | 19.52% | **0.65%** |
| $D=2$, 数据集 b | 92.64% | 48.56% | **15.58%** |

子空间回归在所有数据集上均大幅优于经典插值和直接特征向量预测。

### 子空间嵌入效果（$D=2$ 椭圆问题，前 10 特征向量）

| 预测子空间维度 $N_{\text{sub}}$ | 测试误差 | 说明 |
|:---:|:---:|:---|
| 10（无嵌入） | ~30% | 与目标维度相同 |
| 20 | ~10% | 嵌入开始生效 |
| 30 | ~5% | 显著改善 |
| 40 | **~2%** | 仅占总自由度的 0.4% |

嵌入技术将测试误差从 30% 降至 2%，同时训练-测试泛化差距也系统性缩小。

### $D=3$ 椭圆问题损失函数对比

| $N_{\text{sub}}$ | $L_1(A,B)$ | $L_2(A,B;z)$ | $L_2^{\text{stab}}(A,B;z)$ |
|:---:|:---:|:---:|:---:|
| 6 | 24.77% | 31.46% | 28.28% |
| 12 | 13.69% | 17.12% | 15.88% |
| 24 | 9.71% | 失败 | 9.49% |
| 48 | 7.54% | 16.3% | **7.4%** |

$L_2$ 在大维度下因正规方程病态条件数而不稳定，Cholesky-QR2 稳定化后恢复甚至略优于 $L_1$。

## 亮点与洞察

- **子空间嵌入思想**极为精妙：利用子空间包含关系的单调性，用冗余换平滑性，完美契合神经网络的频谱偏差（F-principle）。代价是学到的子空间维度远大于最优，但缩减模型的额外计算量可忽略
- **Theorem 3 的复杂度刻画**将组合数学与逼近论联系，为子空间学习提供了精确的困难度量
- **随机化损失 $L_2$** 将 Hutchinson 迹估计引入子空间学习，大维度下训练时间显著优于 QR 方案
- **迭代法实验的意外发现**：标准 Jacobi 方法的主特征空间包含高低频混合函数导致学习完全失败，改用阻尼 Jacobi ($\omega=0.9$) 后变为纯低频函数，学习难度骤降——问题公式化对学习难度有决定性影响

## 局限与展望

- 子空间嵌入的最优冗余维度 $r - k$ 缺乏自动选择机制，目前需手动网格搜索
- 仅在线性子空间上验证，向非线性流形（如 Stiefel 流形上的约束逼近）的扩展尚未涉及
- 结果为单次运行无误差条（虽然附录有方差分析表明影响小）
- 所有神经网络学到的表示效率远低于最优子空间，这一根本性效率差距是否可弥补仍是开放问题

## 相关工作与启发

- **vs Grassmann 流形插值**：经典方法在高维参数空间中因稀疏观测而失效，本文用回归+神经网络克服维度灾难
- **vs 直接特征向量预测**：$\mathbb{Z}_2$-adjusted $\ell_2$ 虽处理符号不定性，但无法利用子空间结构避免 Theorem 3 的组合爆炸
- **vs 神经算子（FNO/DeepONet）**：互补关系——神经算子直接预测 PDE 解，本文预测缩减基然后求解缩减模型
- **vs DeepPOD**：类似思路但用投影损失直接从快照矩阵提取基，子空间回归在精度上匹配或略优

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 子空间回归问题形式化 + 嵌入技术 + Theorem 3 复杂度分析的理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 多种应用场景覆盖广，与多种 baseline 比较全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，数学符号规范，结构清晰
- 价值: ⭐⭐⭐⭐⭐ 为缩减阶建模提供了强大的新工具，子空间嵌入思想具有广泛的方法论启发

<!-- RELATED:START -->

## 相关论文

- [DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](../../NeurIPS2025/scientific_computing/symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)
- [Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)
- [Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [Integration Matters for Learning PDEs with Backward SDEs](../../NeurIPS2025/scientific_computing/integration_matters_for_learning_pdes_with_backward_sdes.md)

<!-- RELATED:END -->
