---
title: >-
  [论文解读] Preconditioners for the Stochastic Training of Neural Fields
description: >-
  [CVPR 2025][3D视觉][神经场] 本文提出了一个用于神经场随机训练的预条件理论框架，证明了曲率感知对角预条件器（如 ESGD）能显著加速 sine/Gaussian/wavelet 激活神经场的训练，而对 ReLU(PE) 激活则无明显帮助，为神经场优化器选择提供了理论指导。
tags:
  - CVPR 2025
  - 3D视觉
  - 神经场
  - 预条件器
  - 随机优化
  - 曲率感知
  - 激活函数
---

# Preconditioners for the Stochastic Training of Neural Fields

**会议**: CVPR 2025  
**arXiv**: [2402.08784](https://arxiv.org/abs/2402.08784)  
**代码**: https://github.com/sfchng/preconditioner_neural_fields  
**领域**: 3D视觉 / 神经场优化  
**关键词**: 神经场, 预条件器, 随机优化, 曲率感知, 激活函数

## 一句话总结

本文提出了一个用于神经场随机训练的预条件理论框架，证明了曲率感知对角预条件器（如 ESGD）能显著加速 sine/Gaussian/wavelet 激活神经场的训练，而对 ReLU(PE) 激活则无明显帮助，为神经场优化器选择提供了理论指导。

## 研究背景与动机

1. **领域现状**：神经场（Neural Fields / 隐式神经表示）在计算机视觉、机器人和几何建模中广泛应用，包括图像重建、形状建模和 NeRF 等任务。目前 Adam 是训练神经场的默认优化器。
2. **现有痛点**：Adam 虽然有效但训练时间长。传统二阶方法如 L-BFGS 在随机设置中不适用（无法处理 mini-batch 训练），而社区对 "为何 Adam 比 SGD 好" 缺乏严格理论解释。
3. **核心矛盾**：SGD 在高曲率方向需要小学习率以避免 overshooting，导致在低曲率方向上进展缓慢。预条件器可以均衡各方向曲率，但哪种预条件器适合哪种神经场架构缺乏理论指导。
4. **本文目标**：(1) 从预条件角度解释 Adam 的成功；(2) 找到比 Adam 更好的预条件策略来加速训练。
5. **切入角度**：将 Adam 视为使用 Gauss-Newton 矩阵对角线的预条件 SGD，然后分析激活函数对 Hessian 向量积稀疏性的影响。
6. **核心 idea**：sine/Gaussian/wavelet 激活的神经场具有稠密的 Hessian 向量积，因此曲率感知预条件器（如均衡预条件器）能有效降低条件数、加速收敛；而 ReLU(PE) 的 Hessian 向量积是稀疏的，预条件器作用有限。

## 方法详解

### 整体框架

本文不是提出新的网络架构，而是建立了一个理论框架来分析和选择神经场训练的最优预条件器。整体流程为：(1) 将 Adam 解析为基于 Gauss-Newton 矩阵对角线的预条件 SGD；(2) 通过 Hessian 向量积的稀疏性定理区分不同激活函数；(3) 推荐使用均衡预条件器 ESGD 来替代 Adam 训练 sine/Gaussian/wavelet 神经场。

### 关键设计

1. **预条件框架的理论建立**:
    - 功能：统一描述各种优化算法（Adam、ESGD、AdaHessian 等）中预条件器的工作原理
    - 核心思路：通过参数变换 $\tilde{x} = D^{1/2} x$ 将原始优化问题映射到新参数空间，使得新 Hessian 为 $(D^{-1/2})^T H (D^{-1/2})$。目标是选择 $D$ 使新 Hessian 的条件数接近 1，即各方向曲率均匀。Adam 的第二阶矩 $v_t$ 本质上是 Gauss-Newton 矩阵对角线 $\text{Diag}(J^T J)$ 的滑动平均，是一种一阶近似的对角预条件器。
    - 设计动机：解释 Adam 优于 SGD 的根本原因，并为寻找更好的预条件器奠定基础

2. **Hessian 向量积稠密性定理（Theorem 4.2 & 4.3）**:
    - 功能：从理论上区分不同激活函数的神经场是否适合曲率感知预条件器
    - 核心思路：Theorem 4.2 证明 sine/Gaussian/wavelet/sinc 激活的神经场在 MSE 损失下，Hessian 向量积 $Hv$ 是稠密向量；Theorem 4.3 证明 ReLU/ReLU(PE) 的 $Hv$ 是稀疏向量。稠密的 Hessian 向量积意味着预条件器能有效缩放梯度的多个分量，而稀疏的 Hessian 向量积意味着只有少数梯度分量被缩放，预条件效果有限。
    - 设计动机：ReLU 的分段线性性质导致二阶导数在大部分区域为零，因此 Hessian 稀疏；而 sine/Gaussian 等光滑激活函数的二阶导数处处非零产生稠密 Hessian

3. **ESGD 均衡预条件器的实际应用**:
    - 功能：提供一种计算高效的曲率感知优化算法作为 Adam 的替代
    - 核心思路：ESGD 使用均衡预条件器 $D^E$，即 Hessian 每行的 2-范数作为对角预条件矩阵。为提高效率，每 $N=100$ 次迭代才重新计算一次预条件器。每次 Hessian 向量积的计算成本与一次梯度计算相当，存储和求逆为线性复杂度 $O(n)$。
    - 设计动机：均衡预条件器比 Jacobi 预条件器更好地降低条件数（实验验证条件数下降幅度更大），同时避免了全 Hessian $O(n^3)$ 的存储和计算开销

### 损失函数 / 训练策略

- 2D 图像重建和 NeRF 使用 MSE 损失
- 3D 二值占用场使用 BCE（二值交叉熵）损失
- 理论结果对 MSE 和 BCE 损失均成立
- ESGD 每 100 次迭代更新一次预条件器，使用指数移动平均（ESGD）或无穷范数（ESGD-max）两种变体

## 实验关键数据

### 主实验

**2D 图像重建（DIV2K, Gaussian 激活）：**

| 方法 | 收敛速度 | 计算复杂度 | 说明 |
|------|---------|-----------|------|
| ESGD | 最快 | 与 Adam 相当 | 最优平衡 |
| AdaHessian(E) | 较快 | 略高于 Adam | 使用均衡预条件 |
| AdaHessian(J) | 中等 | 略高于 Adam | 使用 Jacobi 预条件 |
| Shampoo | 较快 | 显著高于 Adam | Kronecker 分解开销大 |
| Adam | 基线 | 基线 | 默认优化器 |

**NeRF（LLFF 数据集, Gaussian 激活, 平均结果）：**

| 场景 | 指标 | Adam | ESGD | 说明 |
|------|------|------|------|------|
| fern | Test PSNR | 24.38 | 24.41 | ESGD 用 120K iter, Adam 200K |
| flower | Test PSNR | 25.67 | 25.65 | ESGD 用 120K iter, Adam 200K |
| room | Test PSNR | 31.60 | 30.52 | Adam 略优 |
| trex | Test PSNR | 22.04 | 22.21 | ESGD 用 120K iter, Adam 80K |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| Gaussian + ESGD | 收敛最快 | 稠密 Hessian, 预条件有效 |
| Sine + ESGD | 快于 Adam | 稠密 Hessian, 预条件有效 |
| Wavelet + ESGD | 快于 Adam | 稠密 Hessian, 预条件有效 |
| ReLU(PE) + ESGD | 不如 Adam | 稀疏 Hessian, 预条件无效 |
| Gaussian + Adam | 基线 | Gauss-Newton 对角近似 |
| 无预条件 (SGD) | 最慢 | 所有激活都远不如 Adam |

### 关键发现

- **激活函数是决定性因素**：ESGD 对 Gaussian/sine/wavelet 神经场一致优于 Adam，但对 ReLU(PE) 反而不如 Adam，完美验证了理论预测
- **ESGD 计算高效**：Hessian 向量积的计算时间与单次梯度计算相当，且每 100 步才更新一次预条件器，实际额外开销很小
- **3D 任务中 AdaHessian/Shampoo 退化**：在高模态信号（3D 二值占用场）中，局部 Hessian 变得噪声较大，影响了这些方法的性能；ESGD 因间隔更新预条件器而更鲁棒
- **条件数直接可验证**：实验表明均衡预条件器比 Jacobi 预条件器更大幅度地降低了 Hessian 的条件数

## 亮点与洞察

- **将 Adam 解构为预条件 SGD** 是非常优雅的理论贡献——Adam 的第二阶矩本质上是 Gauss-Newton 矩阵对角线的滑动平均，这个视角将 Adam 纳入了预条件优化的统一框架
- **Hessian 稀疏性与激活函数的关联**非常有洞察力：ReLU 的分段线性性质导致 Hessian 稀疏，这不仅解释了预条件器的有效性差异，也为神经场的激活函数选择提供了新视角
- **ESGD 每 N 步更新预条件器的策略**在高噪声场景（3D 任务）中反而成为优势，这是一个有趣的发现，可迁移到其他需要鲁棒预条件的场景

## 局限与展望

- **对 ReLU(PE) 无改进**：未能找到优于 Adam 的 ReLU(PE) 预条件策略，而 ReLU(PE) 仍是实际应用中最广泛使用的激活组合
- **仅考虑对角预条件器**：全矩阵预条件器可能对 ReLU(PE) 也有效，但计算成本过高
- **未扩展到大规模场景**：instant-NGP、3D Gaussian Splatting 等现代高效表示未被考虑
- 改进方向：探索非对角但稀疏的预条件器，或针对 ReLU(PE) 设计专门的预条件策略

## 相关工作与启发

- **vs Adam**: Adam 使用 Gauss-Newton 矩阵对角线作为预条件器，是一阶近似；本文的 ESGD 使用真正的 Hessian 信息作为预条件，对非 ReLU 激活更有效
- **vs L-BFGS**: L-BFGS 等二阶方法不适合随机训练（mini-batch），本文的框架专为随机设置设计
- **vs AdaHessian**: AdaHessian 结合了 Adam 的移动平均和 Hessian 对角信息，但实际实现使用均衡预条件器而非论文声称的 Jacobi 预条件器

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论视角新颖，将 Adam 与预条件框架统一，Hessian 稀疏性与激活函数的关联是原创贡献
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个任务（2D 重建、3D 占用场、NeRF），多种优化器对比，理论与实验一致性好
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验展示直观，但数学符号较多对非优化方向读者有一定门槛
- 价值: ⭐⭐⭐ 对使用 sine/Gaussian/wavelet 激活的神经场有直接实用价值，但对主流 ReLU(PE) 和现代高效表示帮助有限

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](../../NeurIPS2025/3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)

<!-- RELATED:END -->
