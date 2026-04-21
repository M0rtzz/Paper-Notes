---
title: >-
  [论文解读] Locally Orderless Images for Optimization in Differentiable Rendering
description: >-
  [CVPR 2025][differentiable rendering] 提出利用局部无序图像（LOI）的三维尺度空间（内尺度 σ、色调尺度 β、范围尺度 α）进行直方图匹配的逆渲染优化方法，无需修改可微渲染器即可扩展稀疏梯度的支持范围，有效避免局部最优。
tags:
  - CVPR 2025
  - differentiable rendering
  - inverse rendering
  - locally orderless images
  - histogram matching
  - Wasserstein distance
  - scale space
---

# Locally Orderless Images for Optimization in Differentiable Rendering

**会议**: CVPR 2025  
**arXiv**: [2503.21931](https://arxiv.org/abs/2503.21931)  
**代码**: 待确认  
**领域**: others  
**关键词**: differentiable rendering, inverse rendering, locally orderless images, histogram matching, Wasserstein distance, scale space

## 一句话总结

提出利用局部无序图像（LOI）的三维尺度空间（内尺度 σ、色调尺度 β、范围尺度 α）进行直方图匹配的逆渲染优化方法，无需修改可微渲染器即可扩展稀疏梯度的支持范围，有效避免局部最优。

## 研究背景与动机

**领域现状**: 可微渲染通过分析合成（analysis-by-synthesis）迭代优化场景参数，但对于引起图像空间运动的参数（如光源位置、几何位置），图像梯度极度稀疏（仅在轮廓边界非零）。

**现有痛点**: 代理梯度方法（拓扑导数、拉格朗日导数、变分导数）计算昂贵、仅支持隐式几何或仅处理一次光传输效果。多分辨率金字塔匹配被证明不可靠。

**核心矛盾**: 标准 RGB 梯度在远离目标时为零（梯度消失），但代理梯度方案引入了额外约束和计算开销。

**本文切入角度**: 不修改渲染器和梯度计算，而是改变匹配目标——从像素值匹配改为局部直方图匹配。

**核心 idea**: 将图像视为"局部直方图的族"而非像素值函数，在三个正交的尺度空间中匹配分布而非均值，保留了空间平滑化的梯度扩展效果同时保持分布模态信息。

## 方法详解

### 整体框架

1. 渲染图像 $\mathcal{I}(x;\theta)$
2. 在三个尺度空间中构建局部无序图像表示 $\mathcal{H}(x, k; \theta, \sigma, \beta, \alpha)$
3. 计算渲染图像与参考图像的局部直方图的 Wasserstein 距离
4. 对所有尺度求和得到总误差 $\mathcal{E}_{\text{total}}(\theta) = \sum_{\alpha,\beta,\sigma} \mathcal{E}(\theta, \alpha, \beta, \sigma)$
5. 反向传播更新场景参数 $\theta$

### 关键设计

**1. 内尺度空间（Inner Scale, σ）——分辨率模糊**
- **功能**: 使用高斯滤波器 $G(x;\sigma)$ 对渲染图像进行多分辨率模糊: $\mathcal{I}(x;\theta,\sigma) = (G * I)(x;\sigma)$。
- **核心思路**: 尺度越大，图像特征变成更大的重叠区域，产生非零梯度。类似高斯金字塔但在连续尺度空间中操作。
- **设计动机**: 当目标与初始化不重叠时，标准分辨率下梯度为零；模糊后两者重叠产生梯度信号。但单纯模糊会改变局部外观并抑制高频细节。

**2. 色调尺度空间（Tonal Scale, β）——强度不确定性建模**
- **功能**: 将每个像素的辐射值从确定值放松为概率分布 $\mathcal{P}(x, k; \theta, \sigma, \beta) = \frac{1}{\sqrt{2\pi\beta^2}} \exp(-\frac{(k-\mathcal{I}(x;\theta,\sigma))^2}{2\beta^2})$。
- **核心思路**: 对于给定强度 $k$，$\mathcal{P}(x,k)$ 是一个"软等照度线"（soft isophote），将不同强度的像素分离为独立的图像层，避免了多目标场景中的强度混淆。
- **设计动机**: 蒙特卡罗渲染和传感器噪声导致辐射值本身就有不确定性；色调分离使不同外观的物体在优化时不会相互干扰。

**3. 范围尺度空间（Extent Scale, α）——直方图空间积分**
- **功能**: 使用空间窗 $A(x;\alpha)$ 对局部概率分布进行积分: $\mathcal{H}(x, k; \theta, \sigma, \beta, \alpha) = \int A(x-y;\alpha) \mathcal{P}(y, k) dy$。
- **核心思路**: 与内尺度直接对像素值求平均不同，范围尺度对直方图贡献求平均——保留了分布的模态信息。即使在粗尺度下，强度分布的峰值仍然稳定。
- **设计动机**: 高斯金字塔在粗尺度下丢失了局部外观的多模态特性（只保留均值），而直方图积分保留了完整分布，对噪声也更鲁棒。

### 损失函数 / 训练策略

- 使用 1D Wasserstein 距离（CDF 逐点误差之和）: $\mathcal{E} = \int \int [\text{cdf}_{\mathcal{H}'}(x,k) - \text{cdf}_{\mathcal{H}^{gt}}(x,k)]^{1/p} dk dx$
- 总误差为所有尺度的求和: $\mathcal{E}_{\text{total}} = \sum_{\alpha,\beta,\sigma} \mathcal{E}(\theta,\alpha,\beta,\sigma)$
- 典型参数: $\alpha=[1,5,15]$, $\sigma=[1,5,15,45]$, $\beta=0.125$, $p=1$
- 使用标准梯度下降优化，兼容任意可微渲染器

## 实验关键数据

### 主实验——2D 可微矢量化（恢复 n 个圆盘位置）

| 方法 | n=4 PSNR | n=16 PSNR | n=64 PSNR | n=256 PSNR |
|---|---|---|---|---|
| Gaussian Pyramid | 25.05 | 19.83 | 15.34 | 9.60 |
| MS-SSIM | 27.44 | 21.41 | 15.42 | 9.83 |
| LPIPS | 27.79 | 20.94 | 17.04 | 15.2 |
| **本文** | **30.40** | **33.55** | **28.23** | **21.57** |

### 与参数空间模糊方法对比

| 场景 | PSNR (PRDPT) | PSNR (本文) |
|---|---|---|
| 反射球光源恢复 | 局部最优 | **全局最优** |
| 多物体位置恢复 | 部分成功 | **更高成功率** |

### 关键发现

1. **本文方法在所有规模的圆盘恢复任务中大幅领先**：n=256 时 PSNR 21.57 vs LPIPS 15.2 vs GP 9.60，优势随问题难度增大而扩大。
2. **高斯金字塔在复杂场景中不可靠**：均值匹配无法区分不同外观的物体，在多目标场景中频繁陷入局部最优。
3. **LOI 方法与参数空间模糊（PRDPT）互补**：在复杂光传输场景中可组合使用。
4. **对噪声鲁棒**：直方图分布在噪声下保持模态稳定（核密度估计的固有特性）。
5. **仅使用标准 RGB 梯度即可解决需要"长程"特征匹配的逆问题**。

## 亮点与洞察

- 优雅的理论框架：将 Koenderink & van Doorn 的局部无序图像理论引入逆渲染领域
- 三个正交尺度空间的设计具有清晰的物理意义和互补性
- 方法与渲染器完全解耦——不修改渲染器梯度计算，仅改变匹配目标
- 1D Wasserstein 距离有闭式解，实现简洁高效
- 可与其他优化方法（变分优化等）组合使用

## 局限与展望

- 多尺度参数选择（$\alpha, \beta, \sigma$ 的离散取值）需要手动设定，可探索自适应策略
- 直方图 bin 数量和宽度影响性能，需要调参
- 未探索 3D 场景重建等大规模逆渲染问题
- 对于拓扑变化（如物体出现/消失）的处理能力未分析
- 计算开销：多尺度直方图构建和 Wasserstein 距离计算增加了每次迭代的成本

## 相关工作与启发

- 与 PRDPT（参数空间模糊）形成互补：LOI 在图像空间操作，PRDPT 在参数空间操作，可组合
- Koenderink & van Doorn 的局部无序图像理论原用于图像拓扑分析，本文巧妙地将其重新诠释为逆渲染优化工具
- 启发：将信号处理中的尺度空间理论应用于深度学习优化（如损失函数设计）的思路值得推广

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将经典图像理论创新性地引入可微渲染，三尺度框架设计优雅
- **实验充分度**: ⭐⭐⭐⭐ 覆盖矢量化、光线追踪、光栅化三种渲染器，含真实数据实验
- **写作质量**: ⭐⭐⭐⭐⭐ 从 1D 示例逐步构建直觉，理论与实验紧密结合
- **价值**: ⭐⭐⭐⭐ 提供了逆渲染优化的新范式，与现有方法互补

<!-- RELATED:START -->

## 相关论文

- [DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)
- [NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](../../NeurIPS2025/others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)
- [CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)

<!-- RELATED:END -->
