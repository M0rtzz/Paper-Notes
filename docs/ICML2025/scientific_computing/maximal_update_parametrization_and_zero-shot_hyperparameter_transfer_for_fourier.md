---
title: >-
  [论文解读] Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators
description: >-
  [ICML2025][科学计算][μP] 首次为 Fourier Neural Operator (FNO) 推导了 Maximal Update Parametrization (μP)，使得在小模型上调优的超参数可以零样本迁移到十亿参数级 FNO，将 Navier-Stokes 问题的调参计算量降至 0.30×。
tags:
  - ICML2025
  - 科学计算
  - μP
  - μTransfer
  - FNO
  - 超参数迁移
  - 偏微分方程
  - Fourier 模式缩放
---

# Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators

**会议**: ICML2025  
**arXiv**: [2506.19396](https://arxiv.org/abs/2506.19396)  
**代码**: [LithiumDA/muTransfer-FNO](https://github.com/LithiumDA/muTransfer-FNO)  
**领域**: 科学计算 / Fourier Neural Operator  
**关键词**: μP, μTransfer, FNO, 超参数迁移, PDE 求解, Fourier 模式缩放

## 一句话总结

首次为 Fourier Neural Operator (FNO) 推导了 Maximal Update Parametrization (μP)，使得在小模型上调优的超参数可以零样本迁移到十亿参数级 FNO，将 Navier-Stokes 问题的调参计算量降至 0.30×。

## 研究背景与动机

FNO 通过在频域对低频分量做核积分来学习 PDE 的解算子映射，是当前最主流的 neural operator 架构。其表达能力与 Fourier 模式数 $K$ 直接相关：

- 对 $d$ 维 PDE，核积分参数量为 $\mathcal{O}(K^d)$，增大 $K$ 会使参数量爆炸
- 例：4 层 FNO-3D 将 $K$ 从 3 增至 24，参数从 1.7M 增长到 906M
- 大模型上的超参数调优（学习率、batch size、优化器参数）计算代价不可接受

**核心问题**：能否在小 FNO 上调参，再无损迁移到大 FNO？

已有 μP 理论覆盖了 MLP 的宽度缩放（$\Theta(m^{-1})$）和深度缩放（$\Theta(L^{-1/2})$），但 FNO 的核积分算子工作机制与标准线性层截然不同，此前无人推导过其 μP。

## 方法详解

### 1. FNO 架构回顾

FNO 的前向过程：

$$\mathcal{G}_\theta = \mathcal{Q} \circ \phi(W_L + \mathcal{K}_L) \circ \cdots \circ \phi(W_1 + \mathcal{K}_1) \circ \mathcal{P}$$

其中核积分算子为：

$$(\mathcal{K}v)(x) = \mathcal{F}^{-1}\left[\boldsymbol{R} \cdot \mathcal{T}_K(\mathcal{F}v)\right](x)$$

- $\mathcal{F}$/$\mathcal{F}^{-1}$：$d$ 维 FFT 及逆变换
- $\mathcal{T}_K$：截断到最低 $K$ 个 Fourier 模式
- $\boldsymbol{R} \in \mathbb{R}^{K^d \times m \times m}$：可学习参数张量

### 2. abc-参数化泛化

将 $\boldsymbol{R}$ 的参数化定义为三个缩放函数 $a(K), b(K), c(K)$：

| 函数 | 含义 |
|------|------|
| $a(K)$ | 参数缩放：$\boldsymbol{R} = a(K)\boldsymbol{r}$ |
| $b(K)$ | 初始化方差：$\boldsymbol{r}_{ij} \sim \mathcal{N}(0, b(K)^2)$ |
| $c(K)$ | 学习率缩放：$\eta = c(K)\eta_0$ |

标准参数化下 $a=1, b=\Theta(1), c=1$，最优学习率随 $K$ 漂移。

### 3. 主定理：FNO 的 μP

**定理 3.5（主要理论结果）**：在 tanh/GELU 激活、sub-Gaussian 梯度更新假设下，FNO 在 Adam 优化器下的 μP 为：

$$a(K)=1, \quad b(K) = c(K) = \Theta\!\left(\frac{1}{\sqrt{d\log K}}\right)$$

即：

- **初始化方差**按 $\Theta(1/(d\log K))$ 缩放
- **学习率**按 $\Theta(1/\sqrt{d\log K})$ 缩放

这与已有的宽度缩放 $\Theta(m^{-1})$ 和深度缩放 $\Theta(L^{-1/2})$ 完全不同。差异根源：已有推导依赖随机变量均值的中心极限定理，而 FNO 核积分的谱范数取决于 $K^d$ 个 sub-Gaussian 随机变量的**最大值**，自然引出 $\sqrt{d\log K}$ 项。

### 4. μTransfer-FNO 算法

1. 在小代理模型（$K_{\text{proxy}}$）上网格搜索最优超参数 $\xi^*$
2. 按缩放规则迁移到目标模型（$K^*$）：
    - 学习率：$\eta_{K^*} = \sqrt{\frac{\log K_{\text{proxy}}}{\log K^*}} \cdot \eta_{K_{\text{proxy}}}$
    - 初始化方差：$\sigma^2_{K^*} = \frac{\log K_{\text{proxy}}}{\log K^*} \cdot \sigma^2_{K_{\text{proxy}}}$
3. 用迁移后的超参数直接训练大模型

关键性质：缩放规则与输入离散化 $N_1 \times \cdots \times N_d$ 无关，保持了 FNO 的分辨率无关性。

## 实验关键数据

### 实验设置

| PDE 问题 | 模型 | 维度 $d$ | $K$ 范围 | 参数量范围 |
|----------|------|---------|---------|-----------|
| Burgers 方程 | FNO-1D | 1 | 3→512 | — |
| Darcy Flow | FNO-2D | 2 | 3→24 | — |
| Navier-Stokes | FNO-3D | 3 | 3→24 | 1.7M→906M |

所有模型：4 层、64 隐藏维度、GELU 激活、Adam 优化器。

### 学习率迁移效果

- **标准参数化**：最优学习率随 $K$ 显著漂移（如 FNO-3D 从 $1.8\times10^{-3}$ 漂至 $7.4\times10^{-4}$）
- **μTransfer-FNO**：最优学习率在不同 $K$ 下保持稳定（如 FNO-3D 约 $4.2\times10^{-3}$）

### 端到端性能对比

| 方法 | Darcy Flow $L^2$ 误差 | 训练成本 | NS 方程 $L^2$ 误差 | 训练成本 |
|------|----------------------|---------|-------------------|---------|
| 直接调大模型 | 1.25% | 1× | 5.69% | 1× |
| **μTransfer-FNO** | **1.22%** | 1.38× | **5.34%** | **0.30×** |

在 Navier-Stokes 上实现了更低误差 + 仅 0.30× 计算量的双赢。

### 更多超参数迁移

在 Darcy Flow (FNO-2D) 上验证了 batch size 和 Adam $\beta_2$ 的迁移：

- **Batch size**：μTransfer 下最优 batch size 始终为 20，标准参数化下大模型需要更大 batch
- **$\beta_2$**：μTransfer 下最优 $\beta_2$ 始终为 0.98，标准参数化下随 $K$ 变化

### 对 PINO 的泛化

在 Physics-Informed Neural Operator (PINO) 训练模式下同样有效：
- PINO 引入额外的物理约束损失，训练动态更复杂
- μTransfer-PINO 在 Darcy Flow 上最优学习率稳定在 $5.6\times10^{-3}$

## 亮点与洞察

1. **理论新颖**：首次将 μP 框架扩展到 FNO 核积分算子，发现缩放率 $\Theta(1/\sqrt{d\log K})$ 与已有结果（宽度/深度缩放）本质不同
2. **技术深度**：推导核心在于分析 $K^d$ 个 sub-Gaussian 变量的最大值而非均值，丰富了 μP 理论工具箱
3. **实用价值**：在近 10 亿参数 FNO 上验证了零样本超参数迁移，NS 方程上节省 70% 计算量
4. **泛化性强**：对学习率、batch size、$\beta_2$ 等多种超参数均有效，且兼容 PINO 训练范式
5. **分辨率无关**：缩放规则不依赖空间离散化，保持 FNO 的核心优势

## 局限与展望

1. **仅针对 $K$ 缩放**：宽度 $m$ 和深度 $L$ 的联合缩放未覆盖，实际中三者可能同时变化
2. **sub-Gaussian 假设**：需要对梯度做 element-wise clipping（clip=0.01）来保证理论假设成立，引入额外调参
3. **PDE 类型有限**：仅验证了 Burgers、Darcy、NS 三类经典 PDE，对更复杂/高维问题的适用性待验证
4. **小 $K$ 不稳定**：实验显示 $K$ 较小时最优学习率偶尔偏移，说明 asymptotic 理论在小模型上有误差
5. **仅限 FNO 架构**：未扩展至 DeepONet、Transformer-based operator 等其他架构

## 相关工作与启发

- **μP/μTransfer 系列**：Yang & Hu (2021) → Yang et al. (2022, 2024)，本文是该理论在 neural operator 领域的首次应用
- **iFNO** (George et al., 2024)：渐进增加 Fourier 模式来提高训练效率，可与本文互补
- **启发**：对于其他频域参数化的模型（如 spectral methods），类似的 $\log K$ 缩放规律可能同样成立

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — μP 理论首次应用于 neural operator，缩放率与已有结果本质不同
- 实验充分度: ⭐⭐⭐⭐ — 覆盖三类 PDE、多种超参数、PINO 扩展，但 PDE 类型可更丰富
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨、实验清晰、动机明确
- 价值: ⭐⭐⭐⭐ — 对大规模 FNO 训练有直接实用价值，但受限于 FNO 单一架构
