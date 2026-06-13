---
title: >-
  [论文解读] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 本文提出 3DGS-LM，通过将 3D Gaussian Splatting 的 ADAM 优化器替换为定制的 Levenberg-Marquardt (LM) 二阶优化器，并设计了高效的 GPU 并行化方案和梯度缓存结构，在保持相同重建质量的前提下实现了 20% 的训练加速。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "Levenberg-Marquardt"
  - "优化加速"
  - "新视角合成"
  - "CUDA并行"
---

# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**会议**: ICCV 2025  
**arXiv**: 无 (CVF Open Access)  
**代码**: [https://lukashoel.github.io/3DGS-LM/](https://lukashoel.github.io/3DGS-LM/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, Levenberg-Marquardt, 优化加速, 新视角合成, CUDA并行

## 一句话总结

本文提出 3DGS-LM，通过将 3D Gaussian Splatting 的 ADAM 优化器替换为定制的 Levenberg-Marquardt (LM) 二阶优化器，并设计了高效的 GPU 并行化方案和梯度缓存结构，在保持相同重建质量的前提下实现了 20% 的训练加速。

## 研究背景与动机

1. **领域现状**：3D Gaussian Splatting (3DGS) 是目前最流行的新视角合成方法之一，通过可微分光栅化器将场景表示为一组 3D 高斯体，能够实现实时渲染和高质量图像合成。现有的加速方法主要从减少高斯数量（如改进致密化策略）和加速光栅化器实现两个方向入手。

2. **现有痛点**：尽管在光栅化和致密化方面已有诸多改进，3DGS 的优化仍需数千次梯度下降迭代才能收敛，在高分辨率真实世界数据集上可能需要长达一小时。所有现有方法都依赖 ADAM 这一一阶优化器来拟合高斯参数，这成为了训练时间的主要瓶颈。

3. **核心矛盾**：一阶优化器（如 ADAM）虽然实现简单，但每次迭代的更新质量有限，需要大量迭代才能收敛。而在 RGB-D 融合等传统 3D 重建任务中，Gauss-Newton / Levenberg-Marquardt 等二阶方法能以数量级更少的迭代次数快速收敛，但直接将 LM 应用于 3DGS 面临内存和计算效率的严峻挑战——Jacobian 矩阵过大无法显式存储。

4. **本文目标**：设计一个高效的 LM 优化器，替换 3DGS 中的 ADAM，使其在保持重建质量的同时显著减少优化时间。

5. **切入角度**：作者观察到 LM 在初始化较好时收敛极快（仅需 5-10 次迭代），因此提出两阶段策略：先用 ADAM 完成致密化和粗略优化，再用 LM 快速收敛到最终结果。

6. **核心 idea**：通过预条件共轭梯度 (PCG) 算法以矩阵无关方式求解正规方程，并设计缓存驱动的 per-pixel-per-splat 并行化方案来高效计算 Jacobian-向量积。

## 方法详解

### 整体框架

3DGS-LM 采用两阶段优化流程。输入为 SfM 点云和已标定的图像集。第一阶段使用标准 3DGS 的 ADAM 优化器进行 20K 次迭代，完成高斯致密化和初步参数拟合。第二阶段切换到 LM 优化器，仅需 5 次迭代即可收敛到最终结果。输出为高质量的 3D 高斯场景表示。

### 关键设计

1. **LM 优化器用于 3DGS**:

    - 功能：用二阶近似的 Levenberg-Marquardt 算法替代 ADAM，实现更高效的参数更新。
    - 核心思路：将渲染损失重写为平方和形式 $E(x) = \sum_i r_i^2$，其中 $r_i^{abs} = \sqrt{\lambda_1}|c_i - C_i|$ 和 $r_i^{SSIM} = \sqrt{\lambda_2}(1 - SSIM(c_i, C_i))$。每次迭代通过求解正规方程 $(J^TJ + \lambda_{reg}\text{diag}(J^TJ))\Delta = -J^TF(x)$ 获取更新方向 $\Delta$，再通过线搜索确定最优步长 $\gamma$。使用 PCG 算法以矩阵无关方式求解，避免显式构建和存储巨大的 Jacobian 矩阵。
    - 设计动机：LM 的每次迭代利用了曲率信息，更新方向质量远高于 ADAM 的一阶梯度方向，因此能以极少的迭代次数（5-10 次）完成收敛，而 ADAM 需要 10K+ 次。

2. **缓存驱动的 CUDA 并行化方案**:

    - 功能：高效实现 PCG 算法中的 Jacobian-向量积运算。
    - 核心思路：传统 3DGS 的梯度计算采用 per-pixel 并行化（每个线程处理一条光线），在 PCG 中需要多次重复光线行进，导致中间状态（$T_s$、$\partial c / \partial \alpha_s$ 等）被反复重算。本文提出改为 per-pixel-per-splat 并行化：先构建一次梯度缓存，存储每个像素-splat 对的中间梯度 $\partial c / \partial s$；然后在 PCG 迭代中，通过读取缓存来并行计算 $u = Jp$ 和 $g = J^Tu$，将 splat 沿光线解耦，实现大规模并行。
    - 设计动机：per-pixel 并行化在 PCG 中效率极低（同一中间状态被重算 18 次），缓存方案将重复计算变为一次缓存构建 + 多次快速读取，代价是额外的 GPU 内存消耗。

3. **图像子采样方案**:

    - 功能：控制缓存内存消耗，使方法可扩展到高分辨率、大规模数据集。
    - 核心思路：将所有图像分成 $n_b$ 个批次，对每个批次独立求解正规方程得到更新向量 $\Delta_i$，然后按加权平均合并：$\Delta = \sum_i M_i \Delta_i / \sum_k M_k$，权重 $M_i = \text{diag}(J_i^T J_i)$ 反映每个高斯参数在该批图像中的贡献度。实践中使用 25-70 张图像 / 批，最多 4 个批次。
    - 设计动机：完整数据集的缓存太大无法放入 GPU 显存。子采样方案通过分批处理和加权合并，在不显著影响收敛性的前提下将内存消耗降低到可控范围。

### 损失函数 / 训练策略

- 目标函数与原始 3DGS 完全一致：$L(x) = \frac{1}{N}\sum_i (\lambda_1|c_i - C_i| + \lambda_2(1 - SSIM(c_i, C_i)))$，其中 $\lambda_1 = 0.8$，$\lambda_2 = 0.2$
- 两阶段策略：第一阶段 ADAM 运行 20K 次迭代（含致密化），第二阶段 LM 运行 5 次迭代（每次含 8 次 PCG 内循环）
- 正则化强度 $\lambda_{reg}$ 基于更新质量自适应调整

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 时间(s) |
|--------|------|-------|-------|--------|---------|
| MipNeRF-360 | 3DGS | 27.40 | 0.813 | 0.218 | 1271 |
| MipNeRF-360 | 3DGS + LM | 27.39 | 0.813 | 0.221 | **972** |
| MipNeRF-360 | DISTWAR | 27.42 | 0.813 | 0.217 | 966 |
| MipNeRF-360 | DISTWAR + LM | 27.42 | 0.814 | 0.221 | **764** |
| Deep Blending | Taming-3DGS | 29.84 | 0.900 | 0.274 | 447 |
| Deep Blending | Taming-3DGS + LM | 29.91 | 0.901 | 0.275 | **347** |

### 消融实验

| 配置 | PSNR↑ | 时间(s) | 说明 |
|------|-------|---------|------|
| L1/SSIM + ADAM | 27.23 | 1573 | 原始3DGS |
| L1/SSIM + LM | 27.29 | 1175 | 加速25%，质量一致 |
| L2 + ADAM | 27.31 | 1528 | L2损失质量差 |
| Batch=100 | 33.77 | 242 | 最高质量 |
| Batch=40 | 33.51 | 212 | 质量轻微下降，速度更快 |

### 关键发现

- LM 优化器可以**无缝插入**多种 3DGS 基线（3DGS、DISTWAR、gsplat、Taming-3DGS），平均加速 20%，质量几乎不变
- LM 仅需 5 次迭代即可达到 ADAM 需要 10K 次迭代的效果，但每次迭代计算量更大
- 内存消耗显著增加（平均 53GB vs 基线 6-11GB），这是速度-内存权衡

## 亮点与洞察

- **二阶优化在 3DGS 中的成功应用**：证明了 LM 优化器在显式 3D 表示优化中的可行性，每次迭代的更新质量远超一阶方法，5 次 LM 迭代即可等效 10K 次 ADAM 迭代。
- **缓存驱动的并行化设计**：通过将光线行进的中间状态缓存并按高斯排序，巧妙地将 PCG 的 Jacobian-向量积计算从 per-pixel 转为 per-pixel-per-splat 并行化。这个思路可迁移到其他需要频繁计算 Jacobian 相关量的可微渲染任务。
- **正交性优势**：LM 优化器与光栅化加速、致密化改进等方法正交互补，可直接与现有工作组合使用。

## 局限与展望

- 内存消耗大（53GB），高分辨率场景可能需要 CPU offloading
- 致密化阶段仍依赖 ADAM，无法在 LM 框架内统一处理
- 对初始化敏感，直接在 SfM 初始化上运行 LM 效果不佳

## 相关工作与启发

- **vs ADAM 多视图**：增加 ADAM 的 batch size 到相同级别，ADAM 仍需更多迭代和时间才能收敛到相同质量
- **vs gsplat/DISTWAR**：这些方法加速光栅化前后向传播，与本文的优化器替换正交互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 LM 引入 3DGS 优化并解决了 GPU 并行化的工程难题
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个数据集 × 4 个基线，消融全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法细节完整
- 价值: ⭐⭐⭐⭐ 实用的即插即用加速方案，但内存开销是主要限制

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
