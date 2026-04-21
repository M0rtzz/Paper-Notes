---
title: >-
  [论文解读] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres
description: >-
  [NeurIPS 2025][图像生成][Schrödinger Bridge] 将Iterative Markovian Fitting (IMF)程序推广到树结构Schrödinger Bridge问题，提出TreeDSBM算法，在Wasserstein重心计算中将IMF迭代与不动点迭代优雅合并，仅需廉价的bridge-matching步骤即可高效求解。
tags:
  - NeurIPS 2025
  - 图像生成
  - Schrödinger Bridge
  - Iterative Markovian Fitting
  - Wasserstein Barycentre
  - 最优传输
  - 树结构代价
  - 生成模型
---

# Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres

**会议**: NeurIPS 2025  
**arXiv**: [2506.17197](https://arxiv.org/abs/2506.17197)  
**作者**: Samuel Howard, Peter Potaptchik, George Deligiannidis (Oxford)  
**代码**: [samuel-howard/Tree_SB_Matching_Barycentres](https://github.com/samuel-howard/Tree_SB_Matching_Barycentres)  
**领域**: image_generation  
**关键词**: Schrödinger Bridge, Iterative Markovian Fitting, Wasserstein Barycentre, 最优传输, 树结构代价, 生成模型  

## 一句话总结

将Iterative Markovian Fitting (IMF)程序推广到树结构Schrödinger Bridge问题，提出TreeDSBM算法，在Wasserstein重心计算中将IMF迭代与不动点迭代优雅合并，仅需廉价的bridge-matching步骤即可高效求解。

## 研究背景与动机

### 问题背景
Schrödinger Bridge (SB)是熵正则化最优传输(OT)的动态形式，近年来基于flow的生成模型方法为其提供了可扩展的求解工具。标准SB问题处理两个边际分布之间的传输；更一般地，多边际OT旨在最小化定义在多个分布之上的代价函数，其中**树结构代价**因应用广泛且可利用结构提升效率而备受关注。特别地，星形树对应于重要的**Wasserstein重心**问题——为概率分布定义自然的"平均"。

### 已有工作的不足
- **IPF方法的缺陷**：现有TreeDSB基于Iterative Proportional Fitting (IPF)，存在仅在收敛时才保持两端边际分布、需要昂贵的轨迹缓存、以及"遗忘"原始参考测度等问题
- **不动点方法的瓶颈**：传统Wasserstein-2重心的不动点迭代方法每步需求解完整的OT问题，计算代价高昂
- **文献空白**：IMF已在标准SB问题中胜过IPF，但尚未被推广到树结构SB设定（见Table 1的文献定位）

### 核心动机
将IMF的诸多优势（每步保持边际分布、无需轨迹缓存、训练稳定）迁移到树结构SB和Wasserstein重心计算中，填补文献空白。

## 方法详解

### 树上的随机过程
在树$\mathcal{T}=(\mathcal{V},\mathcal{E},\ell)$上定义随机过程：沿有根树的每条边$e=(u,v)$运行SDE $\mathrm{d}X_t^e = v^e(t,X_t^e)\mathrm{d}t + \sigma_t^e \mathrm{d}B_t^e$，从根节点出发按深度优先遍历顺序依次模拟，构成树上的路径测度$\mathbb{P}\in\mathcal{P}(\mathcal{C}_\mathcal{T})$。

### 树结构Schrödinger Bridge
给定仅在部分顶点$\mathcal{S}\subset\mathcal{V}$上固定的边际分布$\{\mu_i\}_{i\in\mathcal{S}}$，TreeSB问题定义为：

$$\mathbb{P}^{SB} = \arg\min_{\mathbb{P}\in\mathcal{P}(\mathcal{C}_\mathcal{T})} \left\{ \mathrm{KL}(\mathbb{P}\|\mathbb{Q}) \;\middle|\; \mathbb{P}_i=\mu_i\;\forall i\in\mathcal{S} \right\}$$

静态问题等价于树结构二次代价的熵正则化多边际OT。

### TreeIMF核心定理
**Theorem 3.1**：TreeSB的解是唯一同时满足Markov性和属于倒易类$\mathcal{R}_\mathcal{S}(\mathbb{Q})$（且边际正确）的路径测度。这推广了标准IMF的刻画结果，为交替投影算法提供了理论基础。

关键技术工具是**KL散度沿树的分解**（Lemma 3.2）：

$$\mathrm{KL}(\mathbb{P}\|\widetilde{\mathbb{P}}) = \sum_{(u,v)\in\mathcal{E}_r} \mathbb{E}_{X_u\sim\mathbb{P}_u}\left[\mathrm{KL}(\mathbb{P}^{(u,v)}(\cdot|X_u)\|\widetilde{\mathbb{P}}^{(u,v)}(\cdot|X_u))\right]$$

### TreeDSBM算法
算法交替执行两步：
1. **倒易投影**：从当前耦合$\Pi_\mathcal{S}$采样已知顶点值，通过高斯条件分布$\mathbb{Q}_{\mathcal{S}^c|\mathcal{S}}$采样未知顶点值，沿每条边画布朗桥
2. **Markov投影**：对每条边独立训练神经网络向量场$v_{\theta_e}$，使用bridge-matching损失：

$$\mathcal{L}(\theta_e) = \int_0^{T^e} \mathbb{E}\left\| \frac{X_v - X_t}{T^e - t} - v_{\theta_e}(X_t, t) \right\|^2 \mathrm{d}t$$

采用双向训练以减少误差累积，所有边可并行训练。

### 与不动点方法的联系
在星形树（重心问题）中，条件分布简化为$Y_0 \sim \mathcal{N}(\sum\lambda_i Y_i, \sigma^2 I_d)$。TreeDSBM将IMF迭代与不动点迭代合并为单一过程，用廉价的bridge-matching替代了每步完整OT求解。

**Theorem 3.5**：TreeIMF迭代收敛到唯一不动点（即TreeSB解），$\lim_{n\to\infty}\mathrm{KL}(\mathbb{P}^n\|\mathbb{P}^*)=0$。

## 实验关键数据

### 实验1：2D合成数据重心

计算moon、spiral、circle三个数据集的$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$-重心，$\varepsilon=0.1$。

| 方法 | 迭代次数 | Sinkhorn散度 (k=0) | Sinkhorn散度 (k=1) | Sinkhorn散度 (k=2) |
|------|---------|-------------------|-------------------|-------------------|
| **TreeDSBM** | 6 IMF | **1.14±0.07** | **1.05±0.07** | **1.08±0.11** |
| TreeDSB | 50 IPF | 2.35 | 4.04 | 2.35 |
| WIN | - | 1.17 | - | - |

TreeDSBM用6次迭代即显著优于TreeDSB的50次迭代，且与强基线WIN竞争力相当。

### 实验2：子集后验聚合 (BW₂²-UVP, %)

在自行车租赁数据集上，比较Poisson和负二项回归的子集后验聚合：

| 方法 | Poisson (↓) | Negative Binomial (↓) |
|------|-----------|-------------------|
| WIN | 0.014 | 0.009 |
| W2CB | 0.026 | 0.024 |
| NOTWB | 0.023 | 0.018 |
| **TreeDSBM** | **0.008** | **0.012** |

TreeDSBM在Poisson回归上取得最优结果，且训练速度快（约3分钟，WIN需45分钟）。

### 实验3：高维高斯重心

| 方法 | d=64 BW₂² | d=96 BW₂² | d=128 BW₂² | d=64 L² | d=96 L² | d=128 L² |
|------|----------|----------|-----------|--------|--------|---------|
| WIN | 0.20 | 0.30 | 0.38 | 0.96 | 1.20 | 1.46 |
| W2CB | 0.04 | 0.07 | 0.12 | 0.17 | 0.20 | 0.25 |
| NOTWB | 0.08 | 0.10 | 0.14 | 0.10 | 0.10 | 0.13 |
| **TreeDSBM** | 0.14 | 0.15 | 0.27 | 1.18 | 1.13 | 1.23 |

TreeDSBM在BW₂²指标上接近SOTA，与WIN水平相当；L²指标上略逊于W2CB和NOTWB。

### 实验4：MNIST 2,4,6重心
TreeDSB在低正则化$\varepsilon$时训练不稳定；TreeDSBM因每步保持边际匹配，可使用$\varepsilon=0.02$（远小于TreeDSB的0.5），仅4次IMF迭代即生成视觉质量更优的重心样本。

## 亮点

- **优雅的理论推广**：将IMF的收敛理论（倒易类/Markov类交替投影）完整推广到树结构设定，关键利用KL散度沿树分解引理
- **IMF与不动点的统一**：揭示TreeDSBM在星形树中等价于不动点重心算法使用flow-based熵OT求解器的版本，将两层嵌套迭代合并为单一过程
- **计算效率显著提升**：6次IMF迭代大幅超越50次IPF迭代（TreeDSB），每步仅需bridge-matching而非完整OT；边独立训练可并行化
- **训练稳定性**：IMF确保每步保持正确边际，允许使用更小的正则化$\varepsilon$，避免了TreeDSB的训练不稳定问题
- **开源代码**：基于JAX实现，代码公开

## 局限与展望

- **限于二次代价**：仅适用于Wasserstein-2（二次地面代价）的OT问题，不支持一般代价函数
- **熵偏差**：引入熵正则化带来偏差，与真实OT解有差距
- **推理代价较高**：相比单次函数评估的方法（如WIN的组合map），需要模拟SDE多步推理
- **非采样自由**：每次迭代需模拟当前学到的过程以生成样本，不同于解析方法
- **高维L²指标偏高**：在128维高斯实验中L²-UVP指标明显高于W2CB和NOTWB
- **共享结构场景的不足**：附录实验显示当已知边际与重心有简单共享结构时，方法可能不如专门方法

## 与相关工作的对比

- **TreeDSB**：本文的直接前驱——基于IPF的树结构SB求解器。TreeDSBM在所有实验中均优于TreeDSB，且迭代次数更少、训练更稳定
- **DSBM**：标准IMF/SB匹配方法，本文将其推广到树结构，树退化为单边时恢复DSBM
- **WIN**：基于不动点性质的Wasserstein-2重心方法，利用对抗loss学习OT map。TreeDSBM性能可比但训练时间更短
- **W2CB**：基于凸神经网络的重心方法，高维高斯实验中BW₂²指标最优，但对复杂非连续传输的数据可能收敛困难
- **NOTWB**：双层对抗方法，一般代价适用，高维表现优异。TreeDSBM采用非对抗bridge-matching loss，训练更稳定
- **DSB**：基于IPF的SB求解器（双边际），TreeDSB是其树结构推广，TreeDSBM对应IMF路线的推广

## 评分

- 新颖性: ⭐⭐⭐⭐ — 理论推广自然但非平凡，IMF与不动点的统一视角有新意
- 实验充分度: ⭐⭐⭐⭐ — 覆盖合成、图像、后验聚合、高维高斯多场景，与多个SOTA对比
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，理论严谨，图表直观，文献定位精确
- 价值: ⭐⭐⭐⭐ — 填补IMF在树结构SB的空白，对重心计算有实用价值

<!-- RELATED:START -->

## 相关论文

- [Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)
- [Tree-Sliced Wasserstein Distance: A Geometric Perspective](../../ICML2025/image_generation/tree-sliced_wasserstein_distance_a_geometric_perspective.md)
- [Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)

<!-- RELATED:END -->
