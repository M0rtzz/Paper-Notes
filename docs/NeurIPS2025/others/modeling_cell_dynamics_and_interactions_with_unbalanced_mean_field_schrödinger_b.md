---
title: >-
  [论文解读] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge
description: >-
  [NeurIPS 2025][Schrödinger Bridge] 提出 Unbalanced Mean Field Schrödinger Bridge (UMFSB) 框架和 CytoBridge 深度学习算法，从稀疏时间快照数据中同时建模细胞的非平衡随机动力学和细胞间交互。
tags:
  - NeurIPS 2025
  - Schrödinger Bridge
  - 细胞动力学
  - 细胞间交互
  - 单细胞RNA测序
  - 最优传输
---

# Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge

**会议**: NeurIPS 2025  
**arXiv**: [2505.11197](https://arxiv.org/abs/2505.11197)  
**代码**: [GitHub](https://github.com/zhenyiizhang/CytoBridge-NeurIPS)  
**领域**: 计算生物学 / 最优传输  
**关键词**: Schrödinger Bridge, 细胞动力学, 细胞间交互, 单细胞RNA测序, 最优传输

## 一句话总结

提出 Unbalanced Mean Field Schrödinger Bridge (UMFSB) 框架和 CytoBridge 深度学习算法，从稀疏时间快照数据中同时建模细胞的非平衡随机动力学和细胞间交互。

## 研究背景与动机

从高维分布样本中重建动力学是科学和机器学习中的核心挑战。在单细胞生物学中，从 scRNA-seq 快照数据推断细胞轨迹（trajectory inference）是一个基础问题。现有方法已经在以下方面取得了进展：

- **最优传输（OT）方法**：如 Benamou-Brenier 形式可以推断连续的细胞动力学
- **非平衡 OT**：引入 Wasserstein-Fisher-Rao 度量来处理细胞的生长和死亡
- **Schrödinger Bridge**：建模分布间最可能的随机转移路径
- **非平衡随机方法**：如 DeepRUOT，同时处理非平衡和随机效应

然而，**现有方法几乎都忽略了细胞间交互（cell-cell interaction）**。在真实生物场景中，胞间通讯是基本的生命过程，会直接影响细胞状态转移。例如，邻近细胞可能通过信号分子促进或抑制对方的分化。

**核心矛盾**：如何在动力学最优传输框架内，同时建模非平衡随机效应（生长/死亡）和粒子间交互？

**切入角度**：将 Mean Field Schrödinger Bridge（处理交互）与 Regularized Unbalanced OT（处理非平衡）统一为新的 UMFSB 框架，并设计深度学习求解器。

## 方法详解

### 整体框架

CytoBridge 通过四个神经网络参数化 UMFSB 问题中的关键量：转移速度 $\mathbf{v}_\theta$、生长率 $g_\theta$、对数密度函数 $s_\theta$ 和交互势 $\Phi_\theta$。训练采用两阶段策略（预训练+联合训练），损失函数包含能量损失、重建损失和 Fokker-Planck 物理约束。

### 关键设计

1. **UMFSB 框架建立**：将 RUOT（处理非平衡）和 MFSB（处理交互）统一。核心 PDE 约束为：

$$\frac{\partial \rho}{\partial t} = -\nabla_\mathbf{x} \cdot \left[\left(\mathbf{b} - \int k(\mathbf{x},\mathbf{y})\nabla_\mathbf{x}\Phi(\mathbf{x}-\mathbf{y})\rho(\mathbf{y},t)d\mathbf{y}\right)\rho\right] + \frac{\sigma^2}{2}\Delta\rho + g\rho$$

   当 $k=0$ 退化为 RUOT，当 $g=0$ 退化为 MFSB，实现了优雅的统一。

2. **Fisher Information 正则化（Theorem 4.1）**：将原始 SDE 约束转化为 ODE 约束，使计算更可行。关键等价变换引入了新的向量场 $\mathbf{v} = \mathbf{b} - \frac{1}{2}\sigma^2\nabla_\mathbf{x}\log\rho$（即 probability flow ODE），将扩散项内化为 Fisher 信息正则项 $\frac{\sigma^4}{8}\|\nabla_\mathbf{x}\log\rho\|^2$。

3. **加权粒子模拟（Proposition 5.1）**：用加权交互粒子系统近似连续密度演化。每个粒子有位置 $\mathbf{X}_t^i$ 和权重 $w_i(t)$，权重的 ODE 为 $\frac{dw_i}{dt} = g(\mathbf{X}_t^i, t)w_i(t)$，位置遵循含交互项的 SDE。当 $N \to \infty$ 时，加权经验测度弱收敛到 UMFSB 的密度解。

4. **Random Batch Methods (RBM)**：交互项计算复杂度为 $\mathcal{O}(N^2)$，采用随机批次方法将粒子随机分组，仅计算组内交互，复杂度降为 $\mathcal{O}((p-1)N)$，且保持 $\mathcal{W}_2$ 距离的收敛性 $\leq C\sqrt{\tau}$。

### 损失函数 / 训练策略

总损失包含三部分：

- **能量损失 $\mathcal{L}_{\text{Energy}}$**：促进最小作用原理，采用上界近似避免 $\mathbf{v}_\theta$ 和 $s_\theta$ 的耦合优化
- **重建损失 $\mathcal{L}_{\text{Recons}}$**：包含局部/全局质量匹配损失和 Wasserstein 分布匹配损失，对齐生成密度与真实数据密度
- **Fokker-Planck 约束 $\mathcal{L}_{\text{FP}}$**：PINN 风格的物理信息损失，强制四个网络满足 Fokker-Planck 方程

训练分为**预训练阶段**（四步依次初始化 $g_\theta, \mathbf{v}_\theta, \Phi_\theta, s_\theta$）和**联合训练阶段**（最小化总损失联合优化所有网络）。

## 实验关键数据

### 主实验：合成基因调控网络（吸引交互，$\sigma=0.05$）

| 模型 | $\mathcal{W}_1$ (t=1) | TMV (t=1) | $\mathcal{W}_1$ (t=4) | TMV (t=4) |
|------|---------|---------|---------|---------|
| SF2M | 0.146±0.002 | 0.080±0.000 | 0.554±0.005 | 0.930±0.000 |
| DeepRUOT | 0.044±0.002 | 0.014±0.007 | 0.057±0.003 | 0.075±0.044 |
| UOT-FM | 0.051±0.000 | 0.010±0.000 | 0.054±0.000 | 0.095±0.000 |
| **CytoBridge** | **0.015±0.001** | **0.013±0.009** | **0.038±0.003** | **0.058±0.061** |

### 主实验：小鼠造血数据集（$\sigma=0.1$）

| 模型 | $\mathcal{W}_1$ (t=1) | TMV (t=1) | $\mathcal{W}_1$ (t=2) | TMV (t=2) |
|------|---------|---------|---------|---------|
| SF2M | 8.217±0.001 | 2.231±0.000 | 11.086±0.002 | 5.399±0.000 |
| MIOFlow | 6.313±0.000 | 2.231±0.000 | 6.746±0.000 | 5.399±0.000 |
| DeepRUOT | 6.052±0.002 | 0.200±0.001 | 6.757±0.006 | 0.260±0.007 |
| **CytoBridge** | **6.013±0.002** | **0.208±0.001** | **6.644±0.011** | **0.078±0.013** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无交互项 (DeepRUOT) | $\mathcal{W}_1$=0.044 (t=1) | 能捕捉转移模式，无法捕捉方差减小 |
| 无生长项 (SF2M) | TMV=0.930 (t=4) | 无法匹配质量变化，产生错误转移 |
| 完整 CytoBridge | $\mathcal{W}_1$=0.015, TMV=0.013 | 同时捕捉转移和交互模式 |
| Lennard-Jones 势 | 正确识别 | 能恢复既有吸引又有排斥的 LJ 势 |
| 无交互真值 | 正确识别 $k \approx 0$ | 能正确判断不存在交互的情况 |

### 关键发现

- CytoBridge 在所有数据集上的分布匹配（$\mathcal{W}_1$）和质量匹配（TMV）均优于或持平于现有 SOTA
- 能从数据中自动学习交互势的形态（吸引/排斥/无交互）
- 在小鼠造血数据上，学到的高生长率区域与造血干细胞群体一致
- 学到的交互力与细胞分化方向的相关性分析揭示了交互可能促进早期分化、抑制晚期分化

## 亮点与洞察

- **理论贡献扎实**：UMFSB 框架优雅统一了 RUOT 和 MFSB，Fisher information 变换将 SDE 问题化为 ODE 问题是技术亮点
- **可解释性强**：能直接输出交互势、生长率和 Waddington 景观，便于生物学解读
- **从数据学习交互**：不需要预先指定交互势的形式，由神经网络直接从快照数据中学习
- RBM 的引入使方法能扩展到大规模数据（49K+ 细胞）

## 局限与展望

- 当前优化的是能量项上界而非原始 UMFSB 目标
- 交互项的 RBF 展开限制了表达能力，可考虑稀疏表示方法
- 多阶段训练较复杂，可引入 HJB 方程的最优性条件简化
- 未利用生物先验（如配体-受体信息）来约束交互网络
- 开发 simulation-free 训练方法（类似 flow matching）是值得探索的方向

## 相关工作与启发

- 与 DeepRUOT 的关系：CytoBridge 在其基础上增加了交互项，从 RUOT 扩展到 UMFSB
- 与 Meta Flow Matching 的关系：后者用 GCN 建模邻域细胞影响，但仅考虑初始时刻的邻域结构
- Fisher information 正则化的思路可推广到其他涉及 SDE 约束的优化问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ UMFSB 框架是 OT 理论的有意义扩展，首次统一了非平衡+交互+随机
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据均有，消融实验充分，但缺少计算效率对比
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，整体结构合理
- 价值: ⭐⭐⭐⭐ 对计算生物学领域有直接价值，框架也有更广的机器学习应用潜力
