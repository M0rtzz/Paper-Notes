---
title: >-
  [论文解读] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] 将经典最优实验设计中的 P-Optimality 理论引入 3D-GS，推导出基于 Hessian 矩阵的通用协方差矩阵，提出对角和块对角两种近似方案，在 D-Optimality 和 T-Optimality 准则下显著超越 FisherRF 的信息增益量化。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯泼溅
  - 不确定性量化
  - 最优实验设计
  - 下一最优视角
  - Fisher信息
---

# POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality

**会议**: CVPR 2025  
**arXiv**: [2503.07819](https://arxiv.org/abs/2503.07819)  
**代码**: 无  
**领域**: 3d_vision  
**关键词**: 3D高斯泼溅, 不确定性量化, 最优实验设计, 下一最优视角, Fisher信息

## 一句话总结

将经典最优实验设计中的 P-Optimality 理论引入 3D-GS，推导出基于 Hessian 矩阵的通用协方差矩阵，提出对角和块对角两种近似方案，在 D-Optimality 和 T-Optimality 准则下显著超越 FisherRF 的信息增益量化。

## 研究背景与动机

3D-GS 虽然渲染质量高，但不具备原生的不确定性量化能力，限制了其在 SLAM、主动感知等应用中的使用：

1. **FisherRF 的局限**：虽通过 Fisher 信息的对角近似量化信息增益，但忽略了参数间的相关性，且未利用最优实验设计的丰富文献
2. **协方差矩阵太大**：3D-GS 可能包含数百万参数，完整协方差矩阵在内存和计算上不可行
3. **缺乏统一框架**：现有方法各自独立设计信息度量，缺乏系统性的理论框架

本文从最大似然估计出发推导 3D-GS 的协方差矩阵，并应用 P-Optimality 理论提供一族信息度量解。

## 方法详解

### 整体框架

从最大似然角度，3D-GS 参数 $\theta$ 的协方差矩阵为 $\Sigma = \sigma_e^2 (J^TJ)^{-1}$，其中 $J$ 是渲染函数对参数的 Jacobian。添加候选图像 $i$ 后，新 Hessian 为 $H_i = H_- + J_i^T J_i$，通过 P-Optimality 的不同 $p$ 值定义信息度量。

### 关键设计

**1. 基于 P-Optimality 的信息量化框架**

- **功能**：提供一族统一的信息增益度量，不同 $p$ 值对应不同几何含义
- **核心思路**：$U_p(\Sigma_i) = (\frac{1}{l} \text{trace}(\Sigma_i^p))^{1/p}$。T-Optimality ($p=1$) 为平均方差（trace），A-Optimality ($p=-1$) 为调和均值方差，D-Optimality ($p \to 0$) 为协方差超椭球体积（行列式），E-Optimality ($p \to \pm\infty$) 为极端特征值
- **设计动机**：D-Optimality 在主动建图中具有单调性保证（不确定性随探索单调递减），且从信息论角度对应多元高斯的微分熵

**2. 块对角协方差近似**

- **功能**：在对角近似的基础上捕获同一椭球体参数间的相关性
- **核心思路**：将完整 Hessian 矩阵近似为块对角矩阵，每个块包含一个 3D 椭球体的所有参数（位置、旋转、缩放、opacity、颜色）。分通道计算逐像素梯度避免奇异性问题，块矩阵可在 GPU 上并行处理
- **设计动机**：同一椭球体的参数最可能相关（如位置变化会影响颜色贡献），对角近似完全忽略了这些相关性

**3. 批次选择算法**

- **功能**：从候选视图集合中迭代选择信息增益最大的视图子集
- **核心思路**：贪心策略——每次选择使 P-Optimality 度量改善最大的候选图像，更新 Hessian 后重复。无需额外训练，通过 Hessian 的增量更新捕获视图间的冗余
- **设计动机**：单视图选择不考虑视图间冗余，批次选择通过 Hessian 更新协方差自然处理冗余

### 损失函数

不涉及额外训练损失。信息量化基于已训练 3D-GS 模型的 Hessian 计算：

$$H_i = H_- + J_i^T J_i, \quad J = \frac{\partial h}{\partial \theta}\bigg|_{\theta_*, p_i}$$

## 实验关键数据

### Blender 数据集（从 100 候选中选 10 个视图）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Uniform | 25.82 | 0.944 | 0.051 |
| FisherRF | 27.14 | 0.956 | 0.039 |
| Diag T-Opt (Ours) | 27.89 | 0.960 | 0.035 |
| **Block D-Opt (Ours)** | **28.31** | **0.963** | **0.032** |

### Mip-NeRF360 数据集

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Uniform | 22.15 | 0.698 | 0.271 |
| FisherRF | 23.42 | 0.732 | 0.243 |
| **Block D-Opt (Ours)** | **24.18** | **0.756** | **0.221** |

### P-Optimality 不同 $p$ 值对比

| 度量准则 | 近似方式 | Blender PSNR↑ |
|---------|---------|-------------|
| T-Optimality (p=1) | Diagonal | 27.89 |
| D-Optimality (p→0) | Diagonal | 27.95 |
| T-Optimality (p=1) | Block | 28.12 |
| **D-Optimality (p→0)** | **Block** | **28.31** |

### 关键发现

- Block D-Optimality 超越 FisherRF ~1.2 PSNR（Blender），~0.8 PSNR（Mip-NeRF360）
- D-Optimality 一致优于 T/A/E-Optimality，与理论预期一致（单调性保证+信息论意义）
- 块对角近似比简单对角提升 ~0.4 PSNR，验证参数相关性的重要性
- 不需要候选图像内容，仅需要位姿即可评估信息增益

## 亮点与洞察

1. **理论框架优雅**：将 3D-GS 信息量化统一到经典最优实验设计中，提供了一族有理论保证的解
2. **块对角近似实用**：在可接受的计算开销增加下有效捕获参数相关性
3. **不需要额外训练**：纯基于已训练模型的梯度信息，即插即用

## 局限与展望

- 块对角近似的计算成本仍然较高，随椭球体参数维度立方增长
- 未考虑 3D-GS 致密化/剪枝过程中参数数量变化
- 贪心批次选择非全局最优，可探索更高效的组合优化方法
- 未来可扩展到动态场景的主动感知

## 相关工作与启发

- **FisherRF**：3D-GS 信息量化的先驱，但仅用对角 Fisher 信息
- **经典 SLAM 文献**：P-Optimality 在关键帧选择和回环检测中广泛应用
- **3D-GS 剪枝**：块对角近似也被用于识别冗余椭球体

## 评分

⭐⭐⭐⭐ — 理论贡献扎实，将经典优化设计理论成功引入 3D-GS。实验在多个数据集上一致超越基线，块对角近似是实用创新。

<!-- RELATED:START -->

## 相关论文

- [DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [Ref-GS: Directional Factorization for 2D Gaussian Splatting](ref-gs_directional_factorization_for_2d_gaussian_splatting.md)
- [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)

<!-- RELATED:END -->
