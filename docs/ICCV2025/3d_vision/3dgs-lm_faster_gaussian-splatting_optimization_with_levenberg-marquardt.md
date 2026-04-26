---
title: >-
  [论文解读] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 本文提出3DGS-LM，用定制的Levenberg-Marquardt优化器替换3DGS中的ADAM优化器，通过高效的GPU缓存驱动并行化方案实现Jacobian-向量积的快速计算，在保持相同重建质量的前提下将3DGS优化速度提升20%。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - Levenberg-Marquardt
  - 优化加速
  - 新视角合成
  - CUDA并行
---

# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**会议**: ICCV 2025  
**arXiv**: [2409.12892](https://arxiv.org/abs/2409.12892)  
**代码**: [https://lukashoel.github.io/3DGS-LM/](https://lukashoel.github.io/3DGS-LM/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, Levenberg-Marquardt, 优化加速, 新视角合成, CUDA并行

## 一句话总结
本文提出3DGS-LM，用定制的Levenberg-Marquardt优化器替换3DGS中的ADAM优化器，通过高效的GPU缓存驱动并行化方案实现Jacobian-向量积的快速计算，在保持相同重建质量的前提下将3DGS优化速度提升20%。

## 研究背景与动机
3D高斯溅射（3DGS）是当前最流行的3D场景表示方法之一，通过可微分光栅化器将3D高斯投影为2D splats进行渲染，实现了高质量实时渲染。然而，3DGS的优化过程仍然需要数万次迭代和长达一小时的时间来收敛。

现有加速方法主要从两个方向入手：改进可微分光栅化器的实现效率（如DISTWAR的warp reduction），以及改进densification策略减少高斯数量（如Taming-3DGS）。但这些方法仍然依赖ADAM这一一阶优化器，需要大量迭代才能收敛。

核心矛盾在于：一阶优化器（ADAM）每步更新方向仅基于梯度，收敛速度受限；而二阶优化器（如LM）通过求解正规方程获得更高质量的更新方向，可以大幅减少迭代次数。但LM需要高效的Jacobian-向量积计算，在3DGS的大规模参数和高分辨率图像场景下实现起来极具挑战。

本文的切入角度是：设计一种基于缓存的高效并行化方案，使LM优化器能在3DGS中实际运行。核心idea：缓存α-blending的中间梯度，将并行模式从"逐像素"改为"逐像素-逐splat"，大幅加速PCG算法中的Jacobian-向量积计算。

## 方法详解

### 整体框架
两阶段优化pipeline：第一阶段使用原始ADAM优化器运行约20K次迭代完成高斯densification和初步优化；第二阶段切换到LM优化器，仅需5次LM迭代即可完成收敛。这种两阶段设计利用了梯度下降在初期快速进步而LM在精细优化阶段更高效的特性。

### 关键设计
1. **LM优化器适配3DGS**:

    - 功能：将3DGS的渲染损失重构为平方和形式，适配LM算法
    - 核心思路：将L1和SSIM损失取平方根得到残差形式 $r_i^{abs} = \sqrt{\lambda_1|c_i - C_i|}$ 和 $r_i^{SSIM} = \sqrt{\lambda_2(1 - \text{SSIM}(c_i, C_i))}$，目标函数保持不变但符合LM的平方和要求。每次迭代通过求解正规方程 $(\mathbf{J}^T\mathbf{J} + \lambda_{reg}\text{diag}(\mathbf{J}^T\mathbf{J}))\Delta = -\mathbf{J}^T\mathbf{F}(\mathbf{x})$ 获得更新方向
    - 设计动机：LM通过近似二阶更新，一次迭代的更新质量远高于ADAM，只需5次迭代vs 10K次

2. **缓存驱动的并行化方案**:

    - 功能：加速PCG算法中反复需要的Jacobian-向量积计算
    - 核心思路：将梯度计算分解为三个独立阶段 $\frac{\partial r}{\partial x_i} = \frac{\partial r}{\partial c} \frac{\partial c}{\partial s} \frac{\partial s}{\partial x_i}$。关键观察是中间量 $T_s$, $\frac{\partial c}{\partial \alpha_s}$ 在PCG迭代中被重复计算多达18次。方案：一次性缓存这些中间梯度（buildCache），然后将并行模式从"逐像素"改为"逐像素-逐splat"，每个线程处理一条射线上的一个splat
    - 设计动机：原始3DGS的逐像素并行需要每个线程遍历射线上所有splat，导致中间量的重复计算；缓存后解耦了splat，实现了更细粒度的并行

3. **图像子采样方案**:

    - 功能：控制缓存大小以适配GPU显存限制
    - 核心思路：将图像分成 $n_b$ 个batch，独立求解正规方程，然后用加权平均合并更新方向：$\Delta = \sum_{i=1}^{n_b} \frac{\mathbf{M}_i \Delta_i}{\sum_k \mathbf{M}_k}$，其中权重 $\mathbf{M}_i = \text{diag}(\mathbf{J}_i^T\mathbf{J}_i)$ 反映各高斯参数对渲染的贡献
    - 设计动机：高分辨率稠密捕获场景的缓存可能超出GPU显存；加权平均比简单平均更合理

### 损失函数 / 训练策略
- 使用与原始3DGS完全相同的L1+SSIM损失函数
- 第一阶段：ADAM优化器20K迭代，完成densification（前15K迭代）
- 第二阶段：LM优化器5次迭代，每次8轮PCG内循环
- 每次LM迭代后通过line search找最佳步长 $\gamma$（在30%图像子集上），并根据更新质量 $\rho$ 自适应调整正则化强度 $\lambda_{reg}$

## 实验关键数据

### 主实验

| 数据集 | 基线方法 | PSNR | 时间(s) | +Ours PSNR | +Ours时间(s) | 加速比 |
|--------|----------|------|---------|------------|-------------|--------|
| MipNeRF360 | 3DGS | 27.40 | 1271 | 27.39 | 972 | 23.5% |
| MipNeRF360 | gsplat | 27.42 | 1064 | 27.42 | 818 | 23.1% |
| Deep Blending | 3DGS | 29.51 | 1222 | 29.72 | 951 | 22.2% |
| Deep Blending | Taming-3DGS | 29.84 | 447 | 29.91 | 347 | 22.4% |
| Tanks&Temples | gsplat | 23.50 | 646 | 23.68 | 414 | 35.9% |

### 消融实验

| 配置 | PSNR | 时间(s) | 说明 |
|------|------|---------|------|
| 3DGS + Ours (L1/SSIM) | 27.29 | 1175 | 完整模型 |
| 3DGS + Ours (L2 only) | 27.48 | 1131 | 仅L2损失 |
| 3DGS ADAM (L1/SSIM) | 27.23 | 1573 | 基线 |
| Batch=100 | 33.77 | 242 | GPU 32.5GB |
| Batch=60 | 33.69 | 223 | GPU 22.6GB |
| Batch=40 | 33.51 | 212 | GPU 15.4GB |

### 关键发现
- LM仅需5次迭代即可完成10K ADAM迭代的工作，更新质量显著更高
- 即使给ADAM相同的多视角batch size（75张图），130次迭代也只能达到LM 5次迭代的水平
- 图像子采样对LM收敛影响很小，40张图即可获得接近全量的效果
- L1/SSIM损失在LM和ADAM下均优于纯L2损失

## 亮点与洞察
- 将传统的二阶优化方法成功应用到大规模3DGS优化中，关键在于缓存驱动的并行化设计
- 方法与其他3DGS加速技术正交可叠加——可以直接插入任何3DGS变体的优化器
- 两阶段策略巧妙利用了一阶和二阶优化器各自的优势：ADAM擅长从差初始化快速收敛，LM擅长精细调整
- 图像子采样几乎不影响LM收敛质量，说明3DGS优化具有良好的局部性质
- 仅5次LM迭代 vs 10K ADAM迭代即可达到相同质量，更新效率提升1000倍以上

## 局限与展望
- 内存开销大：缓存需要约53GB GPU显存，而基线仅需6-11GB，限制了在消费级GPU上的使用
- 目前不支持在densification阶段使用LM优化器，需要先用ADAM完成densification
- SSIM梯度计算做了简化近似（忽略邻域像素的贡献），可能影响收敛精度
- 20%的加速幅度虽然稳定但不算巨大，对于短时间优化场景收益有限
- 需要手动设定batch size和batch数量等超参数，不同数据集需要不同配置

## 相关工作与启发
- **vs 原始3DGS (ADAM)**: 质量相同但加速20%，体现了二阶优化在精细调优阶段的优势
- **vs Taming-3DGS**: 通过减少高斯数量加速，3DGS-LM通过改进优化器加速，两者正交可叠加（叠加后447s→347s）
- **vs NeRF中的GN/LM方法**: RGB-D fusion中LM已被广泛使用，但3DGS的显式高斯表示为高效Jacobian-向量积提供了独特机会
- **vs DISTWAR**: warp reduction加速梯度下降的backward pass，本文直接替换优化器，两种策略可以组合
- **启发**: 缓存中间梯度以加速重复计算的思路可以推广到其他需要多次Jacobian计算的优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将LM优化器成功应用于3DGS，缓存并行化方案设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 3个数据集13个场景，4个基线方法，详细的运行时分析和消融实验
- 写作质量: ⭐⭐⭐⭐ 表述清晰，算法伪代码完整，并行化策略图示直观
- 价值: ⭐⭐⭐⭐ 20%的无损加速具有实际意义，与其他加速方法的正交性增强了实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)

<!-- RELATED:END -->
