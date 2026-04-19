---
title: "FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views"
conference: "CVPR 2025"
arxiv: "2502.12138"
code: "https://zhanghe3z.github.io/FLARE/"
domain: "三维重建 / 稀疏视角"
tags: ["sparse view reconstruction", "camera pose estimation", "3D Gaussian", "feed-forward", "cascaded learning"]
---

# FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views

## 一句话总结

FLARE 提出级联学习范式（cascade learning），以相机位姿为桥梁将 3D 重建分解为位姿估计→局部几何→全局几何→高斯外观四个渐进阶段，在 0.5 秒内从 2-8 张未标定稀疏图像实现高质量的相机位姿、几何重建和新视角合成。

## 研究背景与动机

- **核心问题**：从多视角图像重建 3D 场景是计算机视觉的基础问题。传统 SfM + MVS 管线在稀疏视角下严重退化
- **现有方法的局限**：
    - **优化方法**（BARF、NeRF--）：需要好的初始化，泛化能力差
    - **DUSt3R/MASt3R**：仅处理两视角逐对匹配+后处理全局对齐，速度慢且效果次优
    - **PF-LRM**：前馈式但 tri-plane 表示限制了复杂场景性能
    - **NoPoSplat/Splatt3R**：依赖 DUSt3R 的不完美几何估计
- **关键洞察**：
    - 相机位姿是连接 2D 图像和 3D 结构的"桥梁"——即使不完美的位姿也能提供有效的几何先验
    - 直接联合优化所有参数容易陷入局部最优，应该**渐进式分解**
    - 在相机坐标系下先学局部几何，再投影到全局坐标系，比直接预测全局几何更容易收敛

## 方法详解

### 整体框架

输入未标定稀疏视图 → 四阶段级联：
1. **Neural Pose Predictor**：估计粗略相机位姿
2. **Camera-centric Geometry**：在每个相机坐标系下预测局部点云
3. **Global Geometry Projection**：将局部几何投影到全局坐标系
4. **3D Gaussian Head**：在全局点云上预测高斯参数，实现逼真渲染

### 关键设计

#### 1. Neural Pose Predictor（位姿预测器）

- 将位姿估计建模为图像空间→相机空间的直接变换问题，抛弃特征匹配
- 将图像 patch token 与可学习 camera latent $\mathcal{Q}_c$ 拼接成 1D 序列
- 通过小型 decoder-only transformer $F_p$ 直接回归 7 维位姿（3D 平移 + 归一化四元数）
- 关键发现：位姿不需要很精确——只需近似真实分布即可为后续阶段提供有效先验

#### 2. 两阶段几何学习

**Stage 1 — 相机中心几何估计**：
- 在每个相机的局部坐标系下学习几何，符合图像形成过程（每个视角直接观察局部几何）
- 将图像 token 和位姿 token 送入 transformer $F_l$，利用自注意力进行多视角关联
- 通过 DPT 解码器上采样得到局部点云 $\mathcal{G}_l$ 和置信度图 $\mathcal{C}_l$
- 同时引入额外 pose token $\mathcal{Q}_f$ 精炼位姿（多任务学习互补监督）
- **训练时位姿增强**：对预测位姿添加高斯噪声扰动，使网络对推理时的不精确位姿具有鲁棒性

**Stage 2 — 全局几何投影**：
- 不使用几何变换（因位姿不精确会导致不可靠），而是学习一个 **neural scene projector** $F_g$
- 以局部 point tokens $\mathcal{T}_l$ 和精炼位姿 $\mathcal{P}_f$ 为条件，通过 transformer 变换到全局坐标
- 再通过 DPT 解码器得到全局点云 $\mathcal{G}_g$

#### 3. 3D Gaussian 外观建模

- 以全局点云为高斯中心，预测 opacity、rotation、scale、SH 系数
- 引入预训练 VGG 网络提取图像外观特征 $\mathcal{V}$，与几何特征融合后通过 CNN 解码器回归高斯参数
- 归一化处理解决估计几何与 GT 几何的尺度不一致问题
- 使用可微高斯光栅化器 $R(\cdot)$ 进行端到端渲染

### 损失函数

$$\mathcal{L}_{total} = \lambda_{pose}\mathcal{L}_{pose} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{splat}\mathcal{L}_{splat}$$

- **位姿损失**：Huber loss 监督粗略和精炼位姿
- **几何损失**：置信度加权的 3D 回归损失（局部+全局坐标系）
  $$\mathcal{L}_{geo} = \sum_i \sum_j \mathbf{C}_{i,j}^{camera}\ell_{regr}^{camera} - \alpha\log\mathbf{C}_{i,j}^{camera} + \mathbf{C}_{i,j}^{global}\ell_{regr}^{global} - \alpha\log\mathbf{C}_{i,j}^{global}$$
- **渲染损失**：L2 + VGG 感知损失 + 单目深度损失

## 实验关键数据

### 位姿估计（Tab. 1 — RealEstate10K）

| 方法 | RRA@5°↑ | RTA@5°↑ | AUC@30°↑ |
|------|---------|---------|----------|
| DUSt3R (优化) | 0.83 | 0.37 | 54.9 |
| MASt3R (优化) | 0.87 | 0.45 | 61.1 |
| COLMAP | 0.63 | 0.07 | 16.0 |
| VGGSfM | - | - | 72.1 |
| **FLARE (前馈)** | **0.92** | **0.56** | **76.8** |

### 新视角合成（RealEstate10K & ACID）

在 RealEstate10K 上对比 DUSt3R + 3DGS、NoPoSplat、MASt3R 等方法，FLARE在 PSNR、SSIM、LPIPS 上全面领先。

### 关键发现

1. FLARE 的前馈位姿估计超越了需要优化的 DUSt3R/MASt3R，AUC@30° 从 61.1 提升至 76.8
2. 两阶段几何学习（局部→全局）比直接预测全局几何收敛更快，几何失真更小
3. 位姿增强策略使模型对推理时的位姿误差更鲁棒
4. 整体推理时间 <0.5 秒，比基于优化方法（DUSt3R 需要全局对齐）快一到两个数量级
5. 在真实场景（如室内卧室的随意拍摄）中展现出强泛化能力

## 亮点与洞察

1. **级联分解哲学**：将困难的联合优化问题分解为渐进阶段，每一阶段的输出条件化下一阶段的学习。这种"以简驭繁"的思路优雅且有效
2. **位姿作为桥梁**：即使不精确的位姿也能大幅降低后续几何学习的复杂度——这是一个重要的实践洞察
3. **局部→全局的几何策略**：在相机坐标系下学习局部几何符合物理直觉（每个视角看到的是局部结构），全局投影交给学习型模块处理
4. **位姿噪声增强**：简单但高效的鲁棒性提升策略

## 局限性与可改进方向

1. **GPU 内存限制**：transformer 的自注意力对视角数量线性增长，8 视角以上可能面临内存瓶颈
2. **大场景泛化**：主要在室内和物体中心场景上验证，对大尺度户外场景的效果有待评估
3. **textureless 区域**：稀疏视角 + 弱纹理区域的几何估计仍具挑战性
4. **动态场景**：当前假设静态场景，无法处理运动物体

## 相关工作与启发

- **DUSt3R/MASt3R**：point map 表示 + 两视角匹配→本文扩展到多视角前馈 + 全局一致
- **PF-LRM**：4 视角前馈重建→本文改用 point map + 级联学习实现更好泛化
- **VGGSfM**：可微光束法平差→本文直接用 transformer 回归位姿更快
- **启发**：3D 重建中"粗到精"的级联策略是一种通用范式，位姿估计和几何重建可以互相促进

## 评分

⭐⭐⭐⭐ — 方法设计精妙，级联学习范式高效优雅。从未标定稀疏视角在 0.5 秒内完成位姿+几何+外观的联合推理，实用价值极高。在多个任务上全面超越现有方法。

<!-- RELATED:START -->

## 相关论文

- [Speed3R: Sparse Feed-forward 3D Reconstruction Models](../../CVPR2026/3d_vision/speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](../../ICCV2025/3d_vision/fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)
- [Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

<!-- RELATED:END -->
