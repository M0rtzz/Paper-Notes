---
title: >-
  [论文解读] Discrepancy Minimization in Input-Sparsity Time
description: >-
  [ICML2025][discrepancy minimization] 提出首个实值矩阵差异最小化的输入稀疏时间算法，组合版 $\widetilde{O}(\mathrm{nnz}(A)+n^3)$、快速矩阵乘法版 $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$，逼近 herdisc 的对数因子保证不变，几乎弥合了实值矩阵与二值矩阵之间的计算鸿沟。
tags:
  - ICML2025
  - discrepancy minimization
  - input-sparsity time
  - leverage score sampling
  - sketching
  - Edge-Walk
  - partial coloring
  - fast matrix multiplication
---

# Discrepancy Minimization in Input-Sparsity Time

**会议**: ICML2025  
**arXiv**: [2210.12468](https://arxiv.org/abs/2210.12468)  
**代码**: 无（纯理论工作）  
**领域**: 其他/理论（组合优化、随机线性代数）  
**关键词**: discrepancy minimization, input-sparsity time, leverage score sampling, sketching, Edge-Walk, partial coloring, fast matrix multiplication

## 一句话总结

提出首个实值矩阵差异最小化的输入稀疏时间算法，组合版 $\widetilde{O}(\mathrm{nnz}(A)+n^3)$、快速矩阵乘法版 $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$，逼近 herdisc 的对数因子保证不变，几乎弥合了实值矩阵与二值矩阵之间的计算鸿沟。

## 研究背景与动机

**差异理论 (Discrepancy Theory)** 是组合学和理论计算机科学中的基础课题：给定集合系统 $S_1,\dots,S_m \subseteq [n]$（等价于 $m\times n$ 矩阵 $A$），寻找一个二着色 $x\in\{-1,+1\}^n$ 使得最大不平衡度 $\mathrm{disc}(A,x)=\|Ax\|_\infty$ 最小。该问题在计算几何、差分隐私、bin-packing、机器学习等方向有广泛应用。

经典 Spencer 定理证明了 $\mathrm{disc}(A)\le 6\sqrt{n}$ 的存在性，但长期缺乏高效构造性算法。Bansal (FOCS 2010) 首次给出多项式时间 SDP 算法，运行时间 $\widetilde{O}(mn^{4.5})$。Larsen (SODA 2023) 提出组合算法将时间改进到 $\widetilde{O}(mn^2+n^3)$，但对稀疏矩阵而言仍远非最优。对于**二值矩阵**，JSS (SODA 2023) 已实现输入稀疏时间 $\widetilde{O}(\mathrm{nnz}(A))$，但其方法不能推广到实值矩阵。

**核心问题**：实值矩阵的差异最小化能否达到输入稀疏时间？这是 2018 年差异理论与整数规划 workshop 提出的主要开放问题之一。

## 方法详解

### 整体框架

算法沿用 Lovett-Meka 的迭代部分着色 (Iterated Partial Coloring) 架构，分为两个核心子程序：

1. **ProjectToSmallRows**：找到一个"遗传投影"子空间 $V$，使得 $A$ 投影到 $V^\perp$ 后每行 $\ell_2$ 范数受控于 $O(\mathrm{herdisc}(A)\log(m/n))$。
2. **PartialColoring**：在 $V^\perp$ 上执行 Edge-Walk 随机游走，每轮将常数比例坐标舍入到 $\{-1,+1\}$。

### 关键设计 1：JL Sketch 加速行范数估计

Larsen 算法需显式计算 $B_t = A(I-V_t^\top V_t)$ 的每行范数，代价为 $O(mn^2)$。本文用 JL 随机矩阵 $R\in\mathbb{R}^{n\times O(\log n)}$ 构造压缩矩阵：

$$\widehat{B}_t = A(I-V_t^\top V_t)R$$

由 JL 引理保证对所有行 $j$：

$$\|e_j^\top \widehat{B}_t\|_2 \in (1\pm\epsilon_0)\|e_j^\top B_t\|_2$$

取 $\epsilon_0=\Theta(1)$ 即可保持算法正确性。行范数查询从 $O(mn^2)$ 降到 $\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$。

### 关键设计 2：隐式杠杆分数采样 (Implicit Leverage Score Sampling)

为了避免显式计算 $\bar{B}_t^\top\bar{B}_t$ 的 SVD（代价为 $O(mn^2)$），本文提出 **ImplicitLeverageScore** 算法：

- 输入原始矩阵 $A$ 和正交基 $V$，通过稀疏嵌入矩阵 $S_1, S_2$ 间接计算杠杆分数
- 生成对角采样矩阵 $\widetilde{D}$，构造 $\widetilde{B}_t = \widetilde{D}_t B_t$
- 证明 $\widetilde{B}_t^\top\widetilde{B}_t$ 是 $\bar{B}_t^\top\bar{B}_t$ 的谱近似，添加其特征向量到 $V$ 仍满足"过采样"前提

全过程无需显式存储 $B_t$（$m\times n$ 大小），时间 $\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$。

### 关键设计 3：鲁棒性分析

原算法要求精确行范数和精确 SVD，本文证明：

- **近似范数足矣**：JL sketch 导致的常数因子误差不影响正确性，未选中行的范数仍受 $(1+\epsilon_0)\cdot C_0\cdot T\cdot\mathrm{herdisc}(A)$ 控制
- **近似 SVD 足矣**：只需特征向量满足正交性和谱近似，$\epsilon_B=\Theta(1)$ 的谱近似即可保持 ProjectToSmallRows 的行范数保证

### 关键设计 4：打破立方屏障——猜测-修正数据结构

PartialColoring 每轮需计算 $(I-V_t^\top V_t)\mathsf{g}$，由于 $V_t$ 动态更新，$O(n)$ 轮共需 $O(n^3)$（受 Online Matrix-Vector 猜想限制）。本文设计"猜测-修正"(Guess-and-Correct) 数据结构：

1. **预计算**：在 Init 阶段生成高斯矩阵 $\mathsf{G}\in\mathbb{R}^{n\times N}$，批量预计算投影 $\widetilde{G}=V^\top V\cdot\mathsf{G}$
2. **惰性更新**：将列分为大小 $K=n^a$ 的批次，维护计数器 $k_q,k_u$，达到阈值时 Restart 并累积低秩修正
3. **Query**：输出 $g - \widetilde{g} - \sum_{i=1}^{k_u} w_i w_i^\top g$，利用已知低秩修正代替全矩阵重算

设 $K=n^a$，总运行时间为：

$$\widetilde{O}\!\left(\mathrm{nnz}(A)+n^\omega+n^{2+a}+n^{1+\omega(1,1,a)-a}\right)$$

取 $a\approx 0.53$ 平衡两项，得 $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$。

### 关键设计 5：去除验证步骤

Larsen 原算法每轮需验证事件 $\mathcal{E}_\tau$（检查所有 $m$ 行是否越过阈值 $\tau$），代价 $O(mn)$/轮。本文通过略微增大阈值 $\tau'=\tau(\delta)$，利用集中不等式证明 $\mathcal{E}_{\tau'}$ 以 $1-\delta$ 概率成立，从而完全跳过验证步骤。

## 实验/理论结果

### 主定理

| 方法 | 运行时间 | 近似比 |
|------|---------|--------|
| Bansal (SDP) | $\widetilde{O}(mn^{4.5})$ | $O(\log(mn)\cdot\mathrm{herdisc}(A))$ |
| Larsen (组合) | $\widetilde{O}(mn^2+n^3)$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |
| **本文 (组合)** | $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |
| **本文 (FMM)** | $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |

### 关键理论发现

- **高矩阵最优**：当 $m=\mathrm{poly}(n)$ 时，组合版 $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ 对于高矩阵是最优的
- **打破立方屏障**：FMM 版 $n^{2.53}$ 是首个突破 $n^3$ 的方阵差异算法，超越了 LP 方法的限制
- **快速遗传投影 (Theorem 1.5)**：$\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$ 时间找到遗传投影矩阵，在某种意义上是最优的（读入 $A$ 需 $O(\mathrm{nnz}(A))$，计算投影矩阵需 $n^\omega$）
- **对 FMM 指数不敏感**：即使 $\omega=2$，最终指数仅从 $2.53$ 降至 $2.5$，说明瓶颈已不在矩阵乘法

## 亮点与洞察

1. **首次实现实值矩阵输入稀疏时间差异最小化**，回答了 2018 年 workshop 的公开问题
2. **隐式杠杆分数采样**是独立有价值的贡献：无需显式存储 $m\times n$ 投影矩阵即可完成非遗忘维度缩减
3. **猜测-修正数据结构**巧妙规避了 OMv 猜想的下界，通过预计算+惰性低秩修正将在线矩阵-向量乘法转化为批量操作
4. **鲁棒性分析**证明常数因子近似的范数/SVD 不影响正确性，为 sketch 技术在迭代算法中的应用提供了范式
5. 几乎弥合了实值与二值矩阵差异算法的计算复杂度差距

## 局限与展望

1. **对数因子近似比**：相比 Bansal 的 $O(\log(mn))$ 近似，本文为 $O(\log n\cdot\log^{1.5}m)$，略弱
2. **FMM 依赖**：$n^{2.53}$ 需要快速矩阵乘法，实际不可用；组合版 $n^3$ 虽然更实用但未突破立方
3. **方法极限**：即使 $\omega=2$，本方法也只能达到 $n^{2.5}$，说明需要根本性新思路才能进一步改进
4. **纯理论工作**：虽然附录有实验验证效率，但缺乏大规模实际应用场景的测试
5. **投影步骤不可避免**：对实值矩阵，所有已知算法都需要投影操作，导致 $n^\omega$ 下界；二值矩阵的 projection-free 技术能否推广仍是开放问题

## 相关工作与启发

- **Bansal (FOCS 2010)**：开创性 SDP 差异算法，$O(mn^{4.5})$
- **Lovett & Meka (SICOMP 2015)**：Edge-Walk 部分着色框架，本文核心架构来源
- **Larsen (SODA 2023)**：组合差异算法 $O(mn^{2}+n^{3})$，本文直接改进对象
- **Jain, Sah & Sawhney (SODA 2023)**：二值矩阵输入稀疏时间算法，本文的实值推广
- **Eldan & Singh (RS&A 2018)**：LP 方法，本文结合其框架可得更快 LP 变体
- **JL Sketch / Leverage Score Sampling**：本文将标准 sketch 工具推广到迭代算法场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首次回答实值矩阵输入稀疏差异最小化的公开问题）
- 实验充分度: ⭐⭐⭐ （纯理论为主，附录有验证）
- 写作质量: ⭐⭐⭐⭐ （技术深度极高但图表清晰，流程图辅助理解）
- 价值: ⭐⭐⭐⭐⭐ （差异理论领域重要进展，隐式杠杆分数采样有独立价值）
