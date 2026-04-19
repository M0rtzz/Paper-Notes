---
title: >-
  [论文解读] IM360: Large-scale Indoor Mapping with 360 Cameras
description: >-
  [ICCV 2025][3D视觉][室内三维建图] 本文提出 IM360，一个面向稀疏扫描大规模室内环境的三维建图流水线，通过将球面相机模型深度集成到 SfM 各环节、结合稠密特征匹配和可微渲染纹理优化，在 Matterport3D 和 Stanford2D3D 上实现了远超现有方法的相机定位准确率和渲染质量（PSNR 提升 3.5）。
tags:
  - ICCV 2025
  - 3D视觉
  - 室内三维建图
  - 360度全景相机
  - 球面SfM
  - 纹理优化
  - 大规模室内场景
---

# IM360: Large-scale Indoor Mapping with 360 Cameras

**会议**: ICCV 2025  
**arXiv**: [2502.12545](https://arxiv.org/abs/2502.12545)  
**代码**: https://jdk9405.github.io/IM360/ (有项目页)  
**领域**: 3D视觉  
**关键词**: 室内三维建图, 360度全景相机, 球面SfM, 纹理优化, 大规模室内场景

## 一句话总结

本文提出 IM360，一个面向稀疏扫描大规模室内环境的三维建图流水线，通过将球面相机模型深度集成到 SfM 各环节、结合稠密特征匹配和可微渲染纹理优化，在 Matterport3D 和 Stanford2D3D 上实现了远超现有方法的相机定位准确率和渲染质量（PSNR 提升 3.5）。

## 研究背景与动机

室内三维建图和照片级渲染是 AR/VR、机器人导航等应用的核心技术。传统图像采集流程使用普通透视相机，视场角有限，需要大量密集拍摄才能覆盖场景，耗时且劳动密集。

**360度全景相机的优势与挑战**：全景相机单张图像即可覆盖完整球面视场，Matterport3D 和 Stanford2D3D 的采集密度仅约 0.23 images/m²，而 ScanNet 使用普通相机则需 72.35 images/m²。然而，全景相机在三维重建中面临两大挑战：

**挑战一：稀疏视角下的 SfM 失败**。大规模室内场景存在大面积无纹理区域（白墙、地板）和频繁遮挡。传统 SfM（COLMAP、OpenMVG）基于特征点检测和稀疏匹配，在这些区域严重退化。即使结合最新匹配方法（SuperPoint+SuperGlue），在 Matterport3D 上仍有近半数图像无法注册。目前缺乏专门为球面相机模型设计的完整 SfM 流水线。

**挑战二：稀疏视角下的神经渲染质量差**。NeRF 和 3DGS 优化依赖密集重叠视角，在稀疏全景扫描场景中容易过拟合训练视角、新视角渲染质量差。专门为 ERP 图像设计的方法（如 OmniSDF）在大规模场景中甚至无法收敛。高斯溅射方法也会产生严重的浮点伪影。

**核心切入点**：将球面相机模型（unit bearing vector 表示）贯穿到 SfM 的每一步——从稠密匹配、两视图几何、三角化到 Bundle Adjustment，并结合经典纹理映射与可微渲染精调实现高质量建图。关键观察是：在稀疏室内场景中，传统纹理映射 + 神经优化的混合方案反而优于纯神经渲染方法。

## 方法详解

### 整体框架

IM360 由三个阶段构成：(1) 球面 SfM — 从稀疏全景图估计相机位姿和稀疏三维点；(2) 几何重建 — 将 ERP 图像转为 cubemap 后用神经 SDF 方法重建网格；(3) 纹理优化 — 经典纹理映射初始化 + 可微渲染精调漫射和高光纹理。

### 关键设计

1. **球面稠密匹配与 SfM（Spherical Dense Matching SfM）**

    - 功能：构建首个完整的球面 SfM 流水线，所有步骤均在球面流形上操作
    - 核心思路：
        - **特征匹配**：使用专为 360° 图像设计的稠密匹配方法 EDM，直接在 ERP 图像上提取稠密对应。相比将 ERP 转为 cubemap 后在 36 对透视图间稀疏匹配，直接在球面上稠密匹配更高效且准确
        - **球面两视图几何**：利用 unit bearing vector $u \in \mathbb{S}^2$ 表示图像点，传统对极约束扩展为 $u_1^T E u_2 = 0$。采用 8 点法 + DLT 框架求解本质矩阵 E，加入额外归一化增强 SVD 数值稳定性
        - **球面 Bundle Adjustment**：将观测表示为 bearing vector u，重投影误差定义为 $L = \sum_i \sum_j \rho(\|\Pi(P_j; R_i, t_i) - u_{ij}\|^2)$，其中 $\Pi: \mathbb{R}^3 \mapsto \mathbb{S}^2$ 为球面投影
        - 对稠密匹配的结果进行量化舍入和合并策略，确保多视角匹配一致性
    - 设计动机：现有 COLMAP 和 OpenMVG 对全景图的支持碎片化，缺少从匹配到优化一体化的球面方案。球面表示的核心优势是 360° 视场带来的大视角重叠，这对稀疏扫描至关重要

2. **几何重建（Geometry Reconstruction）**

    - 功能：从全景图重建高质量三维网格
    - 核心思路：将 ERP 图像投影为 cubemap 生成 6 张透视图，利用预训练的 Omnidata 模型估计单目深度和法线先验，输入到 DebSDF 网络学习 SDF 场，最后用 Marching Cubes 提取网格
    - 设计动机：直接在 ERP 图像上训练的方法（OmniSDF）因畸变严重无法收敛；3DGS 表面重建方法（SuGaR、VCR-GauS）在稀疏场景下产生碎片化结果。转为 cubemap 可以复用成熟的透视图方法和单目先验模型

3. **纹理优化（Texture Optimization）**

    - 功能：生成高质量纹理贴图，同时建模漫反射和视角相关的高光效果
    - 核心思路：
        - **初始化**：使用经典纹理映射方法 TexRecon 为每个网格面选择最优颜色图像并生成初始纹理贴图 $K_d$
        - **漫射纹理优化**：将纹理贴图参数化后通过可微光栅化器 nvdiffrast 渲染，用 L1+SSIM 光度损失与真实图像对比优化
        - **高光建模**：额外初始化高光特征 $K_s \in \mathbb{R}^3$，训练一个小型 MLP $f_s$ 作为 fragment shader，输入高光特征和观察方向，输出高光颜色。最终渲染色 $\hat{I} = \hat{I_d} + \hat{I_s}$
        - 损失函数：$L_{photo} = (1-\alpha)\|\hat{I} - I\| + \alpha(1 - SSIM(\hat{I}, I))$，$\alpha=0.2$
    - 设计动机：经典纹理映射有接缝和颜色不一致问题，而纯神经渲染在稀疏场景下严重过拟合。混合方案以经典方法提供稳定初始化，再用可微渲染精调，兼顾鲁棒性和质量。添加高光分量比仅漫射进一步提升 0.8 PSNR

### 损失函数 / 训练策略

- SfM 阶段：球面 BA 使用 soft L1 鲁棒损失用于局部 BA，全局 BA 不加鲁棒核
- 几何重建：DebSDF 训练 8 层 MLP（隐藏维度 256），输入图像 384×384，Adam 优化器 lr=5e-4 + 指数衰减
- 纹理优化：高光 MLP 2 层 32 维，Adam lr=5e-4，训练 7000 步，渲染分辨率 512×512

## 实验关键数据

### 主实验

Matterport3D 数据集相机定位性能：

| 方法 | 注册率 (6场景整体) | AUC@5° (Mean) |
|------|-------------------|---------------|
| OpenMVG | 极低 (多场景<50%) | ~12 |
| SPSG COLMAP | 中等 | ~29 |
| DKM COLMAP | 较高 | ~30 |
| SphereGlue COLMAP | 较高 | ~31 |
| **IM360 (Ours)** | **100% (所有6场景全注册)** | **~57** |

Matterport3D 渲染质量对比：

| 方法 | 渲染方式 | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|------|----------|-------------|-------------|--------------|
| ZipNeRF | Volume | 13.9 | 0.51 | 0.68 |
| 3DGS | Splat | 13.4 | 0.47 | 0.55 |
| SparseGS | Splat | 14.3 | 0.46 | 0.53 |
| TexRecon | Mesh | 15.9 | 0.54 | 0.43 |
| **IM360 (Ours)** | **Mesh** | **19.4** | **0.67** | **0.37** |

### 消融实验

纹理优化各组件贡献（Matterport3D PSNR Mean）：

| 配置 | PSNR | 说明 |
|------|------|------|
| TexRecon（无优化） | 15.9 | 经典纹理映射基线 |
| IM360*（仅漫射优化） | 18.6 | 可微渲染精调漫射纹理，+2.7 |
| IM360（漫射+高光） | **19.4** | 加入高光 MLP，额外 +0.8 |

Stanford2D3D 注册率对比：

| 方法 | area 3 (85img) | area 4 (258img) | area 5a (143img) |
|------|----------------|-----------------|------------------|
| OpenMVG | 6/85 | 17/258 | 8/143 |
| SPSG COLMAP | 28/85 | 73/258 | 54/143 |
| **IM360** | **85/85** | **258/258** | **138/143** |

### 关键发现

- IM360 在所有 Matterport3D 场景实现 100% 图像注册，而 OpenMVG 在多个场景仅注册不到 10% 的图像
- 神经渲染方法（NeRF、3DGS）在稀疏全景场景下表现反而不如经典纹理映射方法（TexRecon），说明稀疏视角是核心瓶颈
- 纹理优化贡献了 3.5 PSNR 的提升，其中漫射精调贡献 2.7、高光建模贡献 0.8
- 球面稠密匹配相比 cubemap 上的稀疏匹配/稠密匹配，在遮挡密集场景中优势显著

## 亮点与洞察

- 首个完整的球面 SfM 流水线，从特征匹配到 BA 全部在球面流形上进行，填补了全景相机三维重建的工具链空白
- 关键观察：在稀疏扫描场景中，基于网格的纹理映射 + 可微渲染组合优于纯隐式/高斯方法，这对社区有重要启示
- 漫射+高光分离的纹理参数化简洁有效，用极小的 MLP（2层32维）即可建模视角相关效果
- 实际应用价值高：360度相机已广泛用于房产扫描等场景，本方法可直接服务于这些应用

## 局限与展望

- 当前依赖人工标注确定匹配图像对（知道哪些图像属于同一房间），未实现自动图像检索
- Cubemap 投影+神经 SDF 的几何重建在场景尺度超过一定范围时质量退化
- ERP 图像的单目深度和法线估计目前没有专门模型，依赖 cubemap 转换后使用透视图先验
- 纹理优化仅用固定 7000 步训练，更长的训练或自适应调度可能进一步提升效果

## 相关工作与启发

- COLMAP/OpenMVG：经典 SfM 框架，对全景相机支持有限，在大规模室内场景注册率低
- EDM：首个 360° 稠密匹配方法，是本文 SfM 的特征匹配基础
- DebSDF：基于单目几何先验的神经 SDF，在稀疏场景中比其他方法更鲁棒
- TMO：纹理贴图优化的先驱工作，但仅支持漫射建模，本文扩展为漫射+高光
- 启发：在数据稀疏场景中，经典方法+神经精调的混合路线可能比端到端神经方法更实用

## 评分

- 新颖性: ⭐⭐⭐⭐ （系统集成创新为主，各组件创新度中等但整合完整）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个大规模数据集，SfM+渲染全面评估，多方法对比）
- 写作质量: ⭐⭐⭐⭐ （流水线描述清晰，但部分细节分散在补充材料）
- 价值: ⭐⭐⭐⭐⭐ （填补全景相机室内建图工具链空白，实际应用价值高）

<!-- RELATED:START -->

## 相关论文

- [Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](../../CVPR2026/3d_vision/indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](../../CVPR2025/3d_vision/3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)
- [S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimat.md)

<!-- RELATED:END -->
