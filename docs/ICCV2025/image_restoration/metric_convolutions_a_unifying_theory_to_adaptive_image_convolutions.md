---
title: >-
  [论文解读] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions
description: >-
  [ICCV 2025][图像恢复][自适应卷积] 从度量几何视角统一解释现有各种自适应卷积（标准/膨胀/平移/可变形），并基于显式 Randers 度量的单位球采样提出 Metric Convolution，以更少参数实现更好的几何正则化和泛化能力。
tags:
  - ICCV 2025
  - 图像恢复
  - 自适应卷积
  - 度量几何
  - Finsler 度量
  - 可变形卷积
  - 去噪
---

# Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions

**会议**: ICCV 2025  
**arXiv**: [2406.05400](https://arxiv.org/abs/2406.05400)  
**代码**: [GitHub](https://github.com/Tommoo/MetricConvolutions)  
**领域**: image_restoration  
**关键词**: 自适应卷积, 度量几何, Finsler 度量, 可变形卷积, 去噪

## 一句话总结

从度量几何视角统一解释现有各种自适应卷积（标准/膨胀/平移/可变形），并基于显式 Randers 度量的单位球采样提出 Metric Convolution，以更少参数实现更好的几何正则化和泛化能力。

## 研究背景与动机

标准卷积是深度学习的基石，但固定的各向同性 $k \times k$ 核形状限制了其对形变物体和复杂空间变换的适应性。为此，研究社区提出了多种卷积变体：

- **膨胀卷积**：均匀放大采样间距，改善感受野但缺乏数据自适应性
- **空间变换网络 (STN)**：学习全局参数化变换，但受限于预定义变换族
- **Active Convolution**：学习各向异性偏移，但所有空间位置共享
- **可变形卷积 (DCN)**：每个像素每个核位置独立学习偏移，灵活但缺乏理论约束

**核心矛盾**：这些方法虽然经验上有效，但缺乏统一的理论框架来理解它们的能力和局限。各种变形策略看起来像是一堆性能驱动的 trick，没有内在联系。

本文的关键 insight 是：**将图像视为配备度量的二维流形**。在这个视角下，每种卷积的核采样邻域可以被重新解释为某个隐式度量的单位球采样。这一观察引出两个方向：(1) 为现有卷积提供几何可解释性；(2) 基于显式度量设计新的 Metric Convolution。

## 方法详解

### 整体框架

1. **统一理论**：证明所有现有卷积（标准、膨胀、平移、可变形）都可以表达为某个隐式度量单位球上的加权信号平均
2. **Metric Convolution**：显式构造信号相关的参数化度量（Randers 度量），从其单位球采样构造核位置
3. 可用于单层去噪滤波器或嵌入 CNN 中替换标准卷积层

### 关键设计

1. **统一度量理论 (Theorem 1-2)**：

    - 任何标准/膨胀/平移/可变形卷积都可以表达为 $(f*g)(x) = \int_{\Delta_x} f(x+y) g(y) dm_x(y)$，其中 $\Delta_x$ 是依赖 $x$ 的局部支撑
    - 度量由其单位切球唯一确定（Theorem 2）
    - 因此各种卷积本质上是隐式度量单位球上的加权平均

2. **Randers 度量参数化**：

    - Riemannian 度量：$R_x(u) = \sqrt{u^\top M(x) u}$，由 2×2 正定矩阵 $M$ 定义
    - Randers 度量（Finsler 子类）：$F_x(u) = \sqrt{u^\top M(x) u} + \omega(x)^\top u$，增加线性漂移项 $\omega$ 以允许**不对称邻域**
    - 不对称性对边缘保持特别有用：在物体边界附近，邻域不应跨越到背景

3. **单位切球采样 (UTB)**：

    - 由度量正齐性得到单位球上角度 $\theta$ 处的点为 $y_x(\theta, \gamma) = \frac{1}{F_x^\gamma(u_\theta)} u_\theta$
    - 使用极坐标离散化 UTB 得到 $k^2$ 个采样点
    - 关键优势：度量参数 $\gamma = (M, \omega)$ 仅需 5-7 个数（Cholesky 分解 3 个 + $\omega$ 2 个），而可变形卷积需要 $2k^2$ 个偏移参数

4. **度量参数的获取**：

    - 启发式设计：$M$ 的特征向量取为图像梯度 $\nabla f$ 及其正交方向，特征值控制各向异性
    - 可学习方式：通过中间标准卷积从输入信号预测度量参数，保持位移等变性

### 损失函数 / 训练策略

- 去噪任务使用 MSE 损失
- 分类任务在 ResNet18 中将 layer2-layer4 的标准 3×3 卷积替换为 Metric Convolution
- 支持固定核权重 (FKW) 和可学习核权重 (LKW) 两种模式
- 使用 Adam 优化器，学习率根据任务调整

## 实验关键数据

### 主实验

在 BSDS300 和 PascalVOC 数据集上的去噪对比（学习滤波器，$k=5$, 噪声 $\sigma_n=0.1$）：

| 方法 | MSE (BSDS300) | MSE (PascalVOC) | 参数效率 |
|------|--------------|-----------------|---------|
| Deformable (FKW) | 1.12e-4 | 8.59e-5 | $2k^2$ 通道 |
| Deformable (LKW) | 1.72e-4 | 1.02e-4 | $2k^2$ 通道 |
| **Metric UTB ε=0.1 (FKW)** | **1.19e-4** | **1.01e-4** | **5 通道** |
| **Metric UTB ε=0.1 (LKW)** | **1.64e-4** | **1.06e-4** | **5 通道** |

CNN 分类（ResNet18, CIFAR-10, LKW-TL）：

| 方法 | TOP1 准确率 | 标准差 |
|------|-----------|--------|
| Standard Conv | 92.64% | ±0.18% |
| Deformable Conv | 93.10% | ±0.17% |
| Shifted Conv | 92.58% | ±0.28% |
| **Metric UTB (Ours)** | **93.07%** | **±0.13%** |

### 消融实验

单图去噪中不同核大小的泛化间隙 $\delta_{\text{MSE}}$（噪声 $\sigma_n=0.3$）：

| 方法 | k=5 | k=11 | k=31 | k=51 | k=121 |
|------|-----|------|------|------|-------|
| Deformable | 265 | 74 | 28 | 18 | 6.6 |
| **Metric UTB (ε=0.9)** | **1.1** | **0.9** | **1.1** | **0.8** | **1.2** |
| **Metric UTB (ε=0.1)** | **1.3** | **1.1** | **1.3** | **1.4** | **1.5** |

可变形卷积的泛化间隙随 $k$ 增大急剧增长（过拟合），而 Metric Convolution 始终维持低且一致的泛化间隙。

### 关键发现

- **几何先验提供强正则化**：Metric CNN 在从头训练 (SC) 时几乎不受影响，而 DCN 和 Shifted Conv 性能大幅下降
- **固定核权重仍然有效**：Metric CNN 在 FKW 设置下仍保持合理性能，而 DCN 接近随机预测
- GradCAM 可视化显示 Metric CNN 更好地聚焦于相关目标和有意义的部分，而非背景
- 不对称度量（小 $\varepsilon_\omega$）通常优于对称度量，因为它允许邻域在边缘处不对称延伸

## 亮点与洞察

- 度量几何视角非常优雅地统一了看似不相关的各种卷积变体
- Metric Convolution 用 5-7 个参数描述核位置变形，而 DCN 需要 $2k^2$ 个，参数效率极高
- Finsler/Randers 度量引入的不对称性在边缘保持上有天然优势
- 移位等变性证明完整（Theorem 3），理论基础扎实

## 局限与展望

- Randers 度量约束单位切球为椭圆形，无法表达更复杂的凸形状
- 测地球 (UGB) 版本计算太慢，无法在实际 CNN 中使用
- 与所有非标准卷积一样，需要存储每个像素的偏移量，高分辨率图像计算成本大
- 目前仅在较小分辨率基准上验证，高分辨率数据和更复杂任务待探索

## 相关工作与启发

- 可变形卷积 v2 (DCNv2) 的调制思想可以视为单位球上的非均匀采样概率分布
- InternImage 将大规模可变形卷积推向基础模型，度量卷积的参数效率可能在此方向更有优势
- 图/曲面上的各向异性卷积（如 Boscaini 等人的工作）与度量卷积在概念上相通
- Randers 度量的不对称性在活动轮廓等经典视觉任务中也有应用

## 评分

- 新颖性：⭐⭐⭐⭐⭐ 度量几何统一视角非常原创
- 技术深度：⭐⭐⭐⭐⭐ 理论框架完整严密
- 实验充分度：⭐⭐⭐⭐ 多任务验证，消融全面
- 实用价值：⭐⭐⭐ 高分辨率实用性待验证
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions](alocc_adaptive_lifting-based_3d_semantic_occupancy_and_cost_volume-based_flow_pr.md)
- [Adaptive Discretization for Consistency Models](../../NeurIPS2025/image_restoration/adaptive_discretization_for_consistency_models.md)
- [Adaptive Estimation and Learning under Temporal Distribution Shift](../../ICML2025/image_restoration/adaptive_estimation_and_learning_under_temporal_distribution_shift.md)

<!-- RELATED:END -->
