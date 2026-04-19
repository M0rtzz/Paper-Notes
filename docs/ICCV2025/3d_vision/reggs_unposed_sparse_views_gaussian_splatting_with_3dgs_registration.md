---
title: >-
  [论文解读] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration
description: >-
  [ICCV 2025][3D视觉][3D高斯泼溅] 提出 RegGS 框架，通过基于最优传输 MW2 距离的可微 3DGS 配准模块，将前馈网络生成的局部3D高斯增量式地对齐到全局一致的3D表示中，实现无位姿稀疏视角的高质量3D重建。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯泼溅
  - 无位姿重建
  - 稀疏视角
  - 最优传输
  - 高斯配准
---

# RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration

**会议**: ICCV 2025  
**arXiv**: [2507.08136](https://arxiv.org/abs/2507.08136)  
**代码**: [项目页面](https://3dagentworld.github.io/reggs/)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 无位姿重建, 稀疏视角, 最优传输, 高斯配准

## 一句话总结

提出 RegGS 框架，通过基于最优传输 MW2 距离的可微 3DGS 配准模块，将前馈网络生成的局部3D高斯增量式地对齐到全局一致的3D表示中，实现无位姿稀疏视角的高质量3D重建。

## 研究背景与动机

从稀疏无位姿图像重建3D场景是一个极具挑战性的问题，现有方法可分为三类，各有其局限：

**优化式3DGS方法**（如CF-3DGS）：将位姿估计融入3DGS优化循环中，但在稀疏视角下因缺乏几何先验而挣扎——拓扑不连续、尺度歧义严重

**前馈式高斯方法**（如NoPoSplat, pixelSplat）：利用大规模训练数据学习3D先验，可直接预测3D高斯，跨数据集泛化能力强。但只能处理有限数量的输入图像（通常2张），无法扩展到更多视角

**传统方法**（COLMAP + 3DGS）：SfM管线在稀疏视角下经常失败

**核心矛盾**：前馈方法有强3D先验但输入视角受限；优化方法能处理任意数量视角但缺乏先验。能否将前馈模型的局部高斯表示通过配准合并成全局一致的表示？

作者提出的解决方案：**3DGS 配准（Registration）**。将问题重新定义为：给定多张稀疏无位姿图像，用前馈模型为每张（或每对）图像生成局部3D高斯，然后通过配准将它们增量式对齐到统一坐标系中。

关键技术挑战：3DGS 的中心点（centers）不能准确反映场景的几何结构——需要考虑每个高斯的完整分布（均值+协方差）。因此引入高斯混合模型（GMM）的统计框架来度量高斯集合之间的结构相似性。

## 方法详解

### 整体框架

1. 用预训练的前馈高斯模型（NoPoSplat）从两张初始图像生成主高斯(main Gaussians)
2. 对每张新输入图像，生成子高斯(sub Gaussians)
3. 通过 MW2 距离+光度一致性+深度几何 的联合优化估计 Sim(3) 变换
4. 将子高斯变换并合并到主高斯中
5. 完成所有帧注册后，全局精炼

### 关键设计

1. **最优传输 MW2 距离**:

    - 功能：度量两组3D高斯分布之间的结构差异
    - 核心思路：将每组3D高斯建模为 GMM，使用 2-Wasserstein 距离衡量单对高斯间的差异：
    $W_2^2 = \|\mu_i^A - \mu_k^{B'}\|^2 + \text{Tr}(\Sigma_i^A + \Sigma_k^{B'} - 2(\Sigma_i^A \Sigma_k^{B'})^{1/2})$
   直接计算 GMM 间的 $W_2$ 距离需要求解无穷维优化问题，因此约束传输计划到高斯混合子空间，得到可计算的 Mixture W2 (MW2) 距离上界：
    $\text{MW}_2^2(P,Q) = \inf_{\pi \in \Pi(w^A, w^B)} \sum_{i,k} \pi_{ik} C_{ik}$
   使用熵正则化 Sinkhorn 算法高效求解：$W_{2,\epsilon}^2 = \min_\pi [\sum_{i,k} \pi_{ik} C_{ik} + \epsilon \sum_{i,k} \pi_{ik} \log \pi_{ik}]$，通过交替缩放迭代收敛。
    - 设计动机：MW2 距离不仅考虑高斯中心的位置偏移，还考虑协方差矩阵（形状和方向），提供了比仅用中心点ICP更完整的对齐度量。熵正则化避免局部最优，加速收敛，并使整个计算过程可微。计算复杂度 $O(MN)$。

2. **可微联合3DGS配准模块**:

    - 功能：联合优化 Sim(3) 变换参数以对齐子高斯到主高斯
    - 核心思路：用四元数+平移+对数尺度参数化 Sim(3) 变换 $\boldsymbol{\theta} = [\mathbf{q}; \mathbf{t}; \log s] \in \mathbb{R}^8$。联合优化三个损失：
    $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{MW}_2} + \lambda_2 \mathcal{L}_{\text{Photo}} + \lambda_3 \mathcal{L}_{\text{Depth}}$
    - MW2 损失驱动全局分布对齐
    - 光度损失 $\mathcal{L}_{\text{Photo}}$ 通过3DGS渲染管线获取像素级RGB一致性
    - 深度损失 $\mathcal{L}_{\text{Depth}}$ 约束深度一致性，抑制尺度漂移和拓扑变形
    - 设计动机：单一MW2损失容易陷入局部最优（因为Sinkhorn是近似解）；光度损失提供精细的局部对齐；深度损失稳定几何同时缓解尺度问题。三者互补实现从粗到细的配准。

3. **增量式配准与全局精炼**:

    - 功能：逐帧注册并最终全局优化
    - 核心思路：前馈模型产生的子高斯尺度差异大，先进行尺度归一化（基于平均深度）和初始尺度估计。coarse-to-fine的增量配准后，对全局高斯进行自适应修剪和精炼（refinement），提升最终渲染质量。
    - 设计动机：增量方式允许处理任意数量的输入图像，突破前馈模型的输入限制。全局精炼修复注册过程中累积的局部不一致性。

### 损失函数 / 训练策略

配准阶段三个损失的联合优化，梯度通过自动微分传播到四元数旋转参数。计算效率方面，Sinkhorn迭代、Cholesky分解、Wasserstein距离计算均映射到GPU张量操作。协方差矩阵加 $10^{-6}I$ 正则化确保正定性。对数空间Sinkhorn迭代防止指数项溢出。

## 实验关键数据

### 主实验

**RE10K 数据集 NVS 结果**：

| 方法 | 2-view PSNR↑ | 8-view PSNR↑ | 16-view PSNR↑ | 32-view PSNR↑ |
|------|-------------|-------------|--------------|--------------|
| NoPoSplat (仅2view) | 23.247 | - | - | - |
| CF-3DGS | 19.326 | 20.329 | 23.034 | 25.596 |
| MASt3R* | 16.036 | 24.249 | 27.024 | 28.309 |
| VideoLifter | 14.526 | 16.651 | 14.765 | 15.268 |
| **RegGS** | **24.272** | **26.691** | **28.663** | **28.332** |

**位姿估计（ATE↓）**：

| 方法 | RE10K 8x | RE10K 16x | ACID 8x | ACID 16x |
|------|----------|-----------|---------|----------|
| CF-3DGS | 0.237 | 0.254 | 0.278 | 0.195 |
| VideoLifter | 0.335 | 0.291 | 0.272 | 0.206 |
| **RegGS** | **0.023** | **0.041** | **0.020** | **0.038** |

### 消融实验

| 配置 | ATE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | MW2↓ |
|------|------|-------|-------|--------|------|
| w/o Photo | 1.184 | 16.06 | 0.52 | 0.44 | 58.8 |
| w/o Depth | 0.160 | 20.97 | 0.72 | 0.29 | 57.8 |
| w/o MW2 | 1.151 | 19.41 | 0.67 | 0.31 | 67.7 |
| w/o 联合配准模块 | 1.164 | 11.41 | 0.34 | 0.60 | 100.0 |
| **完整RegGS** | **0.098** | **23.09** | **0.79** | **0.23** | **56.5** |

### 关键发现

1. 去除任一损失项都导致显著退化——MW2 和光度损失的去除直接导致 ATE 暴涨 10× 以上
2. 联合配准模块是系统核心——没有它 PSNR 从23.09降到11.41，场景几乎无法重建
3. 位姿估计精度远超竞争方法——RE10K 8-view ATE=0.023 vs CF-3DGS的0.237（10×提升）
4. 在2-view设定下RegGS(24.272)甚至超过NoPoSplat(23.247)，说明全局精炼进一步改善了前馈预测
5. ACID数据集（无人机航拍）上同样领先，证明方法不局限于室内场景
6. MW2距离能有效量化高斯分布的对齐程度，作为配准质量的可靠指标

## 亮点与洞察

1. **统一前馈+优化的优雅方案**：前馈模型提供强3D先验（解决稀疏视角问题），配准机制实现多视角融合（突破输入限制），两者互补
2. **最优传输框架**：将3DGS配准形式化为GMM间的最优传输问题，比朴素的ICP或中心点匹配更有理论基础
3. **Sim(3) 空间配准**：考虑尺度因子对齐，适应前馈模型产生的不同尺度输出，这在实际中非常重要
4. MW2 作为可微的配准质量度量，有潜力推广到其他需要对齐分布集合的任务

## 局限与展望

1. 受前馈高斯模型质量影响——如果局部高斯生成质量差，配准可能失败
2. 随输入视角增多训练时间显著增加（MW2 距离计算 $O(MN)$），需要进一步优化
3. 大帧间运动时配准可能不收敛——需要更好的初始化策略
4. 目前使用固定的前馈backbone（NoPoSplat），联合微调可能进一步提升
5. 未讨论动态场景下的表现——所有实验均为静态场景

## 相关工作与启发

- **NoPoSplat**：提供无位姿前馈高斯预测的基础 → RegGS 将其2-view输出扩展到任意视角
- **CF-3DGS**：优化式无位姿3DGS → RegGS 用配准替代端到端优化，稀疏视角下更稳定
- **MASt3R**：点云匹配+3DGS重建 → RegGS 在分布层面（而非点层面）进行对齐
- **Sinkhorn 算法**：经典的最优传输求解器 → 首次应用于3DGS配准问题
- 启发：3DGS 的高斯属性（均值+协方差+不透明度）天然适合统计学框架下的配准

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将最优传输引入3DGS配准是新颖的思路，前馈+配准的框架设计优雅
- **实验充分度**: ⭐⭐⭐⭐ 两个数据集、多视角设定、完整消融，但缺少户外大场景评估
- **写作质量**: ⭐⭐⭐⭐ 方法阐述清晰，但公式符号较多
- **价值**: ⭐⭐⭐⭐ 为无位姿多视角3DGS重建提供了实用的解决方案，位姿估计精度突出

<!-- RELATED:START -->

## 相关论文

- [Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

<!-- RELATED:END -->
