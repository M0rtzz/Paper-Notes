---
title: >-
  [论文解读] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry
description: >-
  从权重对称性角度解释彩票假说(LTH)掩码不能迁移到新初始化的原因，并提出通过置换匹配对齐LTH掩码与新初始化的优化盆地来实现稀疏训练。
tags:

---

# Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry

## 元信息
- **会议**: ICML 2025
- **arXiv**: [2505.05143](https://arxiv.org/abs/2505.05143)
- **代码**: GitHub（公开）
- **领域**: 其他
- **关键词**: 彩票假说, 权重对称性, 置换匹配, 稀疏训练, 线性模式连通性

## 一句话总结
从权重对称性角度解释彩票假说(LTH)掩码不能迁移到新初始化的原因，并提出通过置换匹配对齐LTH掩码与新初始化的优化盆地来实现稀疏训练。

## 研究背景与动机

### 领域现状

**领域现状**：LTH**：密集网络中存在稀疏子网络，用原始初始化训练可匹配密集网络性能

### 现有痛点

**现有痛点**：核心问题**：LTH掩码与其发现时的初始化绑定，不能迁移到新的随机初始化

### 核心矛盾

**核心矛盾**：权重对称性**：神经网络存在置换不变性，同一层交换两个神经元不改变功能

### 解决思路

**解决思路**：假说**：掩码迁移失败是因为LTH掩码的优化盆地与新初始化的优化盆地不对齐

## 方法详解

### 1. 核心假说
不同随机初始化训练的模型收敛到同一盆地（模置换），因此LTH掩码需要相应置换才能与新初始化对齐。

### 2. 置换匹配
使用激活匹配（Ainsworth et al., 2023）找到置换映射 $\pi$：
$$\pi_l = \arg\min_\pi \|Z_l^B - \pi Z_l^A\| = \arg\max_\pi \langle \pi, Z^B(Z^A)^\top \rangle_F$$
通过匈牙利算法求解线性分配问题。

### 3. 掩码对齐
- 训练模型A和B到收敛
- 用激活匹配找 $\pi$ 使 $\pi(w_A^{t=T})$ 与 $w_B^{t=T}$ 对齐
- 将LTH掩码 $m_A$ 置换为 $\pi(m_A)$
- 用 $\pi(m_A)$ 从 $w_B^{t=k}$（重绕点）开始稀疏训练

### 4. 方差坍缩修复
使用REPAIR方法修正插值网络激活统计量的方差坍缩问题，验证线性模式连通性。

## 实验

### 主实验结果

| 数据集/模型 | 稀疏度 | naive vs permuted 差距 |
|------------|--------|---------------------|
| ResNet20/CIFAR-10 | 90% | permuted显著优于naive |
| ResNet20/CIFAR-100 | 90% | 一致性优势 |
| VGG11/CIFAR-10 | 90% | permuted接近LTH |
| ResNet50/ImageNet | 95% | permuted优于naive约2% |

### 宽度效应
模型越宽，置换匹配越准确，permuted与LTH的差距越小（从width=1到16逐步缩小）。

### 多样性分析（Table 1）

| 方法 | 测试精度 | 集成精度 | 分歧度 | KL | JS |
|------|---------|---------|--------|-----|-----|
| LTH | 91.15% | 91.43% | 0.035 | 0.038 | 0.011 |
| Permuted | 89.38% | 91.75% | 0.107 | 0.273 | 0.091 |

Permuted方案功能多样性显著高于LTH，集成性能反而更好。

## 亮点与洞察
- 新颖洞察：从权重对称性理解LTH掩码不可迁移性
- 修正了Paul et al. (2023)关于LTH与密集解不在同一盆地的结论（考虑方差坍缩后它们连通）
- 置换后的模型功能多样性远高于LTH，有利于集成
- 方法简洁，仅需标准激活匹配+掩码置换

## 局限与展望
- 需要训练两个密集模型才能得到置换映射，增加计算成本
- 置换匹配是NP难问题，贪心解在ImageNet上不够精确
- 当前硬件无法有效利用非结构化稀疏性
- 模型剪枝可能引入算法偏见（Hooker et al., 2020）

## 评分
⭐⭐⭐⭐ 从对称性视角解释LTH限制，洞察力强，实验充分展示了掩码对齐的有效性和多样性优势。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] Symmetry-Robust 3D Orientation Estimation](symmetry-robust_3d_orientation_estimation.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[ICML 2025\] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability](constrained_hamiltonian_systems_on_observation-induced_fiber_bundles_theory_of_s.md)

</div>

<!-- RELATED:END -->
