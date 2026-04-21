---
title: >-
  [论文解读] Order-One Rolling Shutter Cameras
description: >-
  [CVPR 2025][卷帘快门] 建立了卷帘快门（Rolling Shutter, RS）相机的统一代数理论，定义了"一阶RS相机"（RS1）——将空间点一一映射到图像点的卷帘模型，完整分类了其三种类型（I/II/III），并发现了 31 个 2-5 相机间相对位姿估计的最小问题。
tags:
  - CVPR 2025
  - 卷帘快门
  - 代数几何
  - 有理映射
  - 最小问题
  - 相对位姿
---

# Order-One Rolling Shutter Cameras

**会议**: CVPR 2025  
**arXiv**: [2403.11295](https://arxiv.org/abs/2403.11295)  
**代码**: 无  
**领域**: 其他 / 几何视觉  
**关键词**: 卷帘快门, 代数几何, 有理映射, 最小问题, 相对位姿

## 一句话总结

建立了卷帘快门（Rolling Shutter, RS）相机的统一代数理论，定义了"一阶RS相机"（RS1）——将空间点一一映射到图像点的卷帘模型，完整分类了其三种类型（I/II/III），并发现了 31 个 2-5 相机间相对位姿估计的最小问题。

## 研究背景与动机

**领域现状**：卷帘快门相机因行扫描导致每一行在不同时刻曝光，与全局快门不同。现有 RS 几何建模大多限于特定运动假设（如匀速旋转），缺乏一般性的代数框架。

**现有痛点**：（1）缺乏 RS 相机的完整数学分类——哪些运动函数产生良定义的（一一映射的）相机？（2）不知道有多少种最小配置可以求解多相机间的相对位姿；（3）现有方法仅处理线性/匀速 RS 这一特殊类。

**核心矛盾**：RS 相机的投影是有理映射而非线性映射，标准的多视角几何工具（本质矩阵/基础矩阵）不直接适用。

**切入角度**：用代数几何中的有理映射和双有理映射理论系统研究 RS 投影的结构。

**核心 idea**：RS1 相机的完整代数分类 + 31 个最小问题 = 卷帘快门几何理论的基础。

## 方法详解

### 关键设计

1. **RS1 相机的特征化（Theorem 4）**：RS 相机为 RS1（一一映射）当且仅当：（a）所有卷帘平面相交于一条直线 K；（b）卷帘平面到参数的映射是双有理的；（c）中心轨迹与 K 交于 $\deg(C)-1$ 个点

2. **三种类型分类**：Type I（$\dim = 3d+6$），Type II（$\dim = 2d+6$），Type III（$\dim = d+6$），其中 $d$ 是卷帘运动的多项式次数

3. **31 个最小问题**：系统枚举了 2-5 个 RS1 相机间相对位姿估计所需的最少点/线对应，问题代数度分布在 28-364 之间

### 损失函数 / 训练策略

纯几何/代数理论，无学习过程。用同伦延拓（homotopy continuation）求解高次多项式系统。

## 实验关键数据

| 最小问题 | 相机数 | 点数 | 代数度 |
|---------|--------|------|--------|
| 实用问题 1 | 2 | 7 | 28 |
| 实用问题 2 | 2 | 9 | 48 |
| 混合点线 | 2 | 3点+2线 | 60 |
| 理论极限 | 5 | - | 364 |

### 关键发现
- **31 个最小问题中大部分代数度过高**——只有少数几个（度 28/48/60）具有实际可解性
- **线性 RS（匀速+匀速旋转）形成 8 维参数空间**——是最常用的实际子类

## 亮点与洞察
- **首次完整的 RS 相机代数理论**——从"case-by-case 分析"到"统一分类定理"，为 RS 几何奠定理论基础
- **连接代数几何与计算机视觉**——双有理映射、参数空间维度等概念直接导出实用算法

## 局限与展望
- 高次最小问题（度 140+）计算上不实用
- 理论假设完全可见性
- 真实 RS 相机的有理映射假设可能不精确成立

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论新颖性极高，首次完整分类
- 实验充分度: ⭐⭐⭐ 理论为主，合成实验验证
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨优雅
- 价值: ⭐⭐⭐⭐ 为 RS 多视角几何提供了长期理论基础

<!-- RELATED:START -->

## 相关论文

- [Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)
- [Higher-Order Responsibility](../../AAAI2026/others/higher-order_responsibility.md)
- [ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/others/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [Enhancing Transformers for Generalizable First-Order Logical Entailment](../../ACL2025/others/enhancing_fol_entailment.md)

<!-- RELATED:END -->
