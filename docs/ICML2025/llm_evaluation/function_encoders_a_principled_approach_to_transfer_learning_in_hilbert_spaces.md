---
title: >-
  [论文解读] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces
description: >-
  [ICML2025][迁移学习] 提出基于 Hilbert 空间几何视角的迁移学习分类体系（凸包插值 / 线性张成外推 / 全空间外推），并设计 Function Encoder 方法利用可学习神经网络基函数实现三种迁移，在多项基准上超越 MAML、Transformer 等方法。
tags:
  - ICML2025
  - 迁移学习
  - Hilbert Space
  - Basis Functions
  - Function Encoder
  - Least Squares
---

# Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces

**会议**: ICML2025  
**arXiv**: [2501.18373](https://arxiv.org/abs/2501.18373)  
**代码**: [tyler-ingebrand/FEtransfer](https://tyler-ingebrand.github.io/FEtransfer)  
**领域**: 迁移学习  
**关键词**: Transfer Learning, Hilbert Space, Basis Functions, Function Encoder, Least Squares

## 一句话总结

提出基于 Hilbert 空间几何视角的迁移学习分类体系（凸包插值 / 线性张成外推 / 全空间外推），并设计 Function Encoder 方法利用可学习神经网络基函数实现三种迁移，在多项基准上超越 MAML、Transformer 等方法。

## 研究背景与动机

- **核心问题**：迁移学习算法何时能有效迁移到新任务？现有方法缺乏对"迁移成功条件"的刻画
- **不足**：MAML 等元学习方法需要对新任务微调，当源任务与目标任务仅弱相关时容易失败；大规模预训练依赖海量数据而非结构性洞察；核方法随数据量增长 Gram 矩阵膨胀
- **动机**：在 Hilbert 空间框架下，将迁移学习问题转化为几何问题——目标任务相对于源任务集合的几何位置决定了迁移难度
- 作者基于已有 Function Encoder 理论 (Ingebrand et al., 2024b)，进一步推广至全 Hilbert 空间的迁移学习场景

## 方法详解

### 1. 迁移学习的几何分类

将归纳迁移（inductive transfer）问题建模在 Hilbert 空间 $\mathcal{H}$ 中，按目标函数 $f_T$ 与源函数集 $\{f_{S_1}, \ldots, f_{S_n}\}$ 的几何关系分为三类：

| 类型 | 名称 | 定义 | 难度 |
|------|------|------|------|
| Type 1 | 凸包插值 | $f_T \in \text{Conv}(f_{S_1}, \ldots, f_{S_n})$，即 $f_T = \sum \alpha_i f_{S_i}$，$\alpha_i \ge 0$，$\sum \alpha_i = 1$ | 最易 |
| Type 2 | 线性张成外推 | $f_T \in \text{span}\{f_{S_1}, \ldots, f_{S_n}\}$，系数无约束 | 中等 |
| Type 3 | 全空间外推 | $f_T \in \mathcal{H}$ 但 $f_T \notin \text{span}\{f_{S_1}, \ldots, f_{S_n}\}$ | 最难 |

### 2. Function Encoder 架构

学习一组神经网络参数化的基函数 $\{g_1, \ldots, g_k\}$，将任意函数 $f \in \mathcal{H}$ 表示为：

$$f(x) = \sum_{j=1}^{k} c_j g_j(x \mid \theta_j)$$

**系数计算——最小二乘法（LS，本文新提出）**：

$$c = G^{-1} b, \quad G_{ij} = \langle g_i, g_j \rangle_{\mathcal{H}}, \quad b_j = \langle f, g_j \rangle_{\mathcal{H}}$$

其中内积通过 Monte Carlo 积分近似：$\langle f, g_j \rangle \approx \frac{1}{m} \sum_{i=1}^{m} y_i \cdot g_j(x_i)$

相比原始内积法（IP），LS 方法的关键优势：
- **不要求基函数正交**，仅需线性无关（更弱条件）
- 提供理论最优投影（最小二乘意义下）
- 训练收敛更快、精度更高

### 3. 训练损失

$$L = \frac{1}{n} \sum_{\ell=1}^{n} \|f_{S_\ell} - \sum_{j=1}^{k} c_j^\ell g_j\|_{\mathcal{H}}^2 + \sum_{i=1}^{k} (\|g_i\|_{\mathcal{H}}^2 - 1)^2$$

第二项为正则化项，防止基函数幅值发散。

### 4. 万能函数空间逼近定理

**Theorem 1**：对任意可分 Hilbert 空间 $\mathcal{H}$，存在一组神经网络基函数，使得 $\mathcal{H}$ 中任意函数均可被任意精度逼近。

证明思路：可分 Hilbert 空间拥有可数正交基 → 神经网络万能逼近定理保证每个正交基可被 NN 逼近 → 误差按几何级数衰减 → 整体有限精度逼近。

### 5. 在线推理

给定目标任务的少量数据 $D_{f_T}$，直接用 LS 公式计算系数即可，无需重训练。Gram 矩阵大小为 $k \times k$（超参数），与数据量无关，推理极快。

## 实验关键数据

在 4 个基准任务上对比 FE (LS)、FE (IP)、AutoEncoder、Transformer、TFE、MAML、BF、BFB 等方法：

| 基准任务 | Type 1 (插值) | Type 2 (张成外推) | Type 3 (全空间外推) |
|----------|---------------|-------------------|---------------------|
| 多项式回归 | FE(LS) 最优，其他方法尚可 | FE(LS) 领先数个量级 | FE(LS) 领先数个量级 |
| CIFAR-100 分类 | FE(LS) 略优于 Siamese Network | — | FE(LS) 最优，与 Siamese 接近 |
| 7-Scenes 位姿估计 | FE(LS) 最优 | — | FE(LS) 最优 |
| MuJoCo Ant 动力学 | FE(LS) 最优 | FE(LS) 显著领先 | FE(LS) 最优且稳定 |

**关键发现**：

- 多项式回归中，FE(LS) 在 Type 2/3 上比其他方法低数个量级的 $L^2$ 误差
- CIFAR-100 中，FE 尽管是通用方法，性能与专用的 Siamese/Prototypical Network 相当甚至略优
- MuJoCo 动力学任务中，AutoEncoder 在训练早期 Type 3 较好但随训练推进急剧退化，FE(LS) 始终稳定
- 增加基函数数量（如从 3 到 100）可显著提升 Type 3 迁移——多余维度被 LS 最优利用

## 亮点与洞察

1. **几何分类体系新颖**：首次从 Hilbert 空间几何角度系统分类三种迁移类型，提供直觉理解
2. **LS 计算系数是核心创新**：不依赖正交性假设，使得基函数训练更灵活、收敛更快
3. **万能逼近定理**：为 Function Encoder 表达能力提供理论保证
4. **不需要微调**：与 MAML 不同，推理时仅需解最小二乘，无梯度计算
5. **冗余维度的利用**：当基函数数量 $k$ 大于源任务数时，LS 能自动利用多余维度适配 Type 3 任务，这是其他方法不具备的

## 局限与展望

1. **内积选择**：不同问题需手动设计内积（如 $L^2$、概率分布内积），通用性受限
2. **基函数数量 $k$ 需调参**：$k$ 太小限制表达能力，太大增加计算和正则化难度
3. **Monte Carlo 近似误差**：数据量少时内积估计不准，影响系数计算质量
4. **可扩展性**：在超大规模任务空间（如数千个源任务）上的效率未验证
5. **Type 3 的理论保证有限**：万能逼近定理是存在性证明，未给出 $k$ 与逼近误差的定量关系
6. **仅考虑归纳迁移**：未涉及领域自适应（domain adaptation）等跨域场景

## 相关工作与启发

- **MAML (Finn et al., 2017)**：学习好初始化以快速微调，但需梯度步骤；FE 无需微调
- **Kernel Methods**：同样使用基函数思想，但基函数数量随数据增长且需预选核
- **Dictionary Learning**：在离散点上做原子分解，FE 的基函数可在连续域上求值
- **Transformer/TFE**：在简单多项式回归中即表现不佳，缺乏结构先验

## 评分

- 新颖性: ⭐⭐⭐⭐ — 几何分类体系 + LS 训练方案 + 万能逼近定理，理论贡献扎实
- 实验充分度: ⭐⭐⭐⭐ — 4 个不同领域基准、消融分析充分，但缺少更大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ — 框架清晰，图表直观，理论与实验结合紧密
- 价值: ⭐⭐⭐⭐ — 为迁移学习提供新的几何理解视角，LS-based FE 实用性强
