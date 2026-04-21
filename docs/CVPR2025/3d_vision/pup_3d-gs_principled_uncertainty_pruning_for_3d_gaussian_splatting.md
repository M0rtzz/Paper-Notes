---
title: >-
  [论文解读] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出基于 Fisher 信息矩阵的有原则的 3D 高斯溅射剪枝方法 PUP 3D-GS，通过空间参数（位置+尺度）的二阶敏感度评分实现 90% 高斯剪枝率，同时保持比现有启发式方法更好的视觉质量和前景细节。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 模型剪枝
  - Fisher信息矩阵
  - 不确定性估计
  - 场景压缩
---

# PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2406.10219](https://arxiv.org/abs/2406.10219)  
**代码**: [项目页面](https://pup3dgs.github.io)  
**领域**: 3D视觉 (3D Vision / Novel View Synthesis)  
**关键词**: 3D高斯溅射, 模型剪枝, Fisher信息矩阵, 不确定性估计, 场景压缩

## 一句话总结

提出基于 Fisher 信息矩阵的有原则的 3D 高斯溅射剪枝方法 PUP 3D-GS，通过空间参数（位置+尺度）的二阶敏感度评分实现 90% 高斯剪枝率，同时保持比现有启发式方法更好的视觉质量和前景细节。

## 研究背景与动机

3D Gaussian Splatting (3D-GS) 在新视角合成中取得了巨大成功，实现了实时渲染和高重建质量。然而，复杂场景通常包含数百万个高斯，导致高存储和内存需求，限制了在资源受限设备上的部署。

现有的 3DGS 剪枝方法（如 LightGaussian、EAGLES、Compact-3DGS）依赖**启发式**标准来决定剪枝哪些高斯——例如基于不透明度、大小、透射率等的人工设计规则。这些启发式方法在高压缩率下（>80%）会严重损失视觉保真度，尤其是前景细节。

本文的核心洞察是：3D 场景重建是一个固有的**欠约束问题**——从有限视角拍摄的图像无法唯一确定每个高斯的 3D 位置和尺度（一个远处的大高斯和一个近处的小高斯在投影空间中可能等价）。这种**空间不确定性**可以通过 Fisher 信息矩阵来精确量化。不确定性高的高斯对场景重建贡献小，是优先剪枝的对象；不确定性低（敏感度高）的高斯对重建至关重要，应保留。

## 方法详解

### 整体框架

PUP 3D-GS 是一个后处理（post-hoc）剪枝流程，可应用于**任何**预训练 3D-GS 模型：（1）在收敛模型上计算每个高斯的空间敏感度评分；（2）移除评分最低的高斯；（3）对剩余高斯做微调（fine-tune）；（4）可选地重复步骤 1-3 进行多轮剪枝。

### 关键设计

1. **基于 Fisher 信息矩阵的敏感度评分**:
    - 功能：为每个高斯计算一个有原则的重要性分数
    - 核心思路：从 $L_2$ 重建误差出发，对高斯参数求二阶 Hessian。在收敛模型上残差项趋近零，Hessian 近似为 Fisher 信息矩阵 $\nabla^2_\mathcal{G} L_2 \approx \sum_\phi \nabla_\mathcal{G} I_\mathcal{G}(\phi) \nabla_\mathcal{G} I_\mathcal{G}(\phi)^T$。取块对角近似得到每个高斯的独立 Hessian $\mathbf{H}_i$，最终敏感度评分为其对数行列式 $U_i = \log|\nabla_{x_i, s_i} I_\mathcal{G} \nabla_{x_i, s_i} I_\mathcal{G}^T|$。这个评分衡量扰动该高斯空间参数对重建误差的影响——值越高，高斯越重要
    - 设计动机：FisherRF 仅使用对角近似和颜色参数，不够准确；块对角近似+空间参数（位置+尺度）捕获了 3D 投影中的几何不变性

2. **仅使用空间参数（位置+尺度）**:
    - 功能：将 Hessian 计算从全参数缩减到 6 维
    - 核心思路：经过消融实验发现，仅需 $x_i \in \mathbb{R}^3$（位置）和 $s_i \in \mathbb{R}^3$（尺度）即可获得有效的敏感度评分。旋转参数不需要，因为在投影不变性下旋转不会引起 3D 几何变化。颜色和球谐参数也可省略。最终每个高斯只需计算 $6 \times 6$ 的块 Hessian
    - 设计动机：减少计算量的同时根据投影几何原理保留最有信息量的参数

3. **Patch-wise 计算 + 多轮剪枝流程**:
    - 功能：降低计算开销并提升最终质量
    - 核心思路：（i）将图像降至 $4\times4$ patch 分辨率计算 Fisher 近似，再在所有视角上求和得到场景级 Hessian——与逐像素计算高度相关但快得多。（ii）多轮剪枝：第一轮剪枝 88.5% 并微调 15K 迭代，第二轮在剩余中再剪枝 ~13% 并微调 15K 迭代，总计达到 90% 剪枝率。比等效的单轮 90% 剪枝质量更好
    - 设计动机：单轮大比例剪枝过于激进，多轮渐进式修剪让模型在每轮之间有机会重新适应

### 损失函数 / 训练策略

- **剪枝后微调**: 使用原始 3DGS 损失 $L = \|I_\mathcal{G}(\phi) - I_{gt}\|_1 + L_{SSIM}(I_\mathcal{G}(\phi), I_{gt})$
- **无致密化**: 微调阶段不进行高斯致密化，防止数量回升
- **两轮流程**: 第一轮保留 11.5%（15K 迭代微调） → 第二轮保留 10%（15K 迭代微调）

## 实验关键数据

### 主实验

Mip-NeRF 360 数据集（90% 剪枝率）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 渲染 FPS↑ |
|------|-------|-------|--------|----------|
| 3D-GS (原始) | 27.49 | 0.815 | 0.214 | 134 |
| LightGaussian | 25.30 | 0.756 | 0.280 | — |
| **PUP 3D-GS** | **26.56** | **0.792** | **0.230** | **477** |

Deep Blending 数据集示例（90% 剪枝）：

| 场景 | 原始 GS 数 | 剪枝后 GS 数 | 原始 FPS | 剪枝后 FPS | 加速比 |
|------|-----------|------------|---------|-----------|-------|
| Playroom | 2.65M | 0.265M | 76.65 | 318.06 | 4.15× |

### 消融实验

| 参数选择 | PSNR | 说明 |
|---------|------|------|
| 位置 + 尺度 (ours) | **26.56** | 最佳，捕获投影不变性 |
| 全参数 | 26.41 | 增加旋转/颜色反而干扰评分 |
| 仅位置 | 26.32 | 缺少尺度信息 |
| 仅颜色 (如 FisherRF) | 25.89 | 颜色不反映空间不确定性 |

### 关键发现

- 90% 剪枝率下 PUP 3D-GS 在 PSNR 上领先 LightGaussian 约 1.26 dB
- 平均渲染速度提升 3.56×，前景细节保留显著更好
- 多轮剪枝比等效单轮剪枝提升约 0.3-0.5 dB
- 可与 Vectree 量化等正交压缩技术结合使用

## 亮点与洞察

- **从启发式到有原则的剪枝**：利用 Fisher 信息矩阵（二阶优化理论）替代人工设计规则，提供了数学上的最优性保证——移除对重建误差影响最小的高斯。这种方法论的提升比单纯的性能提升更有价值
- **空间参数的关键洞察**：发现位置+尺度是最有效的剪枝参数（而非颜色），这与3D重建中投影几何不变性的直觉完美对应——不确定性主要来自深度方向的模糊性

## 局限与展望

- Patch-wise 计算仍需遍历所有训练视角，在大规模场景上计算量较大
- 剪枝后微调需要额外的 15K-30K 迭代训练
- 未探索与自适应致密化策略的联合优化
- 多轮剪枝的最优轮次和比例需要手动选择

## 相关工作与启发

- **vs LightGaussian**: LightGaussian 使用全局显著性分数（启发式组合不透明度、覆盖范围等），在高压缩率下严重丢失前景细节；PUP 3D-GS 的 Fisher 评分更有原则且效果更好
- **vs FisherRF**: FisherRF 也计算 Fisher 信息但仅用对角近似和颜色参数做主动视角选择；PUP 3D-GS 使用块对角近似和空间参数做剪枝，方法和目标都不同
- **vs BayesRays**: BayesRays 在 NeRF 上用 Fisher 信息做不确定性量化，需要假设性扰动场；PUP 3D-GS 在 3DGS 上直接计算且更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 将二阶优化理论引入 3DGS 剪枝，方法有原则性
- 实验充分度: ⭐⭐⭐⭐ 三个数据集多压缩率评估，详细的参数选择消融
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，从不确定性到剪枝的逻辑自然
- 价值: ⭐⭐⭐⭐ 90%剪枝+更好质量，实用价值高，可与其他压缩技术正交使用

<!-- RELATED:START -->

## 相关论文

- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)
- [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)

<!-- RELATED:END -->
