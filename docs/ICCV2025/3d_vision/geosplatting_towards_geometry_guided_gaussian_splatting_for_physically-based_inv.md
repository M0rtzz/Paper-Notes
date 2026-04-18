---
title: >-
  [论文解读] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering
description: >-
  [ICCV 2025][3D视觉][逆渲染] 提出 GeoSplatting，通过从可优化的显式网格可微分地生成表面对齐的高斯点，为3DGS提供精确的几何引导，实现SOTA逆渲染性能（材质-光照解耦），同时训练仅需10-15分钟。
tags:
  - ICCV 2025
  - 3D视觉
  - 逆渲染
  - 3D高斯泼溅
  - 材质分解
  - 环境光照
  - 网格引导
---

# GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering

**会议**: ICCV 2025  
**arXiv**: [2410.24204](https://arxiv.org/abs/2410.24204)  
**代码**: [项目页面](https://pku-vcl-geometry.github.io/GeoSplatting/)  
**领域**: 3D视觉  
**关键词**: 逆渲染, 3D高斯泼溅, 材质分解, 环境光照, 网格引导

## 一句话总结

提出 GeoSplatting，通过从可优化的显式网格可微分地生成表面对齐的高斯点，为3DGS提供精确的几何引导，实现SOTA逆渲染性能（材质-光照解耦），同时训练仅需10-15分钟。

## 研究背景与动机

逆渲染旨在从多视角图像中恢复场景的材质属性（albedo、roughness、metalness）和环境光照，这对游戏、电影、AR/VR等下游应用至关重要。核心挑战在于准确建模光传输（light transport）。

现有方法的问题可按表示方法分类：

**隐式场方法**（TensoIR, NeRO）：通过SDF梯度获得精确法线，适合逆渲染，但体渲染中的密集采样导致训练耗时数小时

**网格方法**（NVdiffrec, NVdiffrecmc）：天然支持PBR流程和射线追踪，但可微渲染仅在三角形边缘产生梯度，优化困难

**3DGS方法**（R3DG, GS-IR, GS-Shader）：渲染效率高，但存在两个根本性缺陷：
   - **法线不精确**：通过隐式几何约束（如深度-法线正则化）近似高斯点法线，精度不足
   - **表面不不透明**：高斯点本质上是半透明的，无法定义准确的光-表面交点

作者的核心洞察：准确的光传输建模需要两个条件——（1）精确的法线方向（决定光传播方向）和（2）不透明的表面（定义光-表面交点）。**显式网格天然满足这两个条件**。因此，通过将3DGS与显式网格「桥接」，可以同时获得网格的几何精度和3DGS的渲染效率。

## 方法详解

### 整体框架

GeoSplatting 的流程：标量场 $\boldsymbol{\zeta}$ → FlexiCubes 提取三角网格 $\mathbf{M}$ → MGadapter 生成表面对齐的高斯点 → 多分辨率哈希网格查询PBR属性 → 物理渲染方程计算每个高斯点的PBR颜色 → 3DGS光栅化生成最终图像。整个过程完全可微，支持端到端训练。

### 关键设计

1. **Mesh-to-Gaussian Adapter (MGadapter)**:

    - 功能：从三角网格的每个面可微分地生成结构化的高斯点
    - 核心思路：对每个三角面片 $\mathbf{P}$，生成 $K=6$ 个高斯点，按重心坐标空间中的预定义模式放置。位置 $\boldsymbol{\mu}_i$ 和法线 $\mathbf{n}_i$ 通过重心插值计算，尺度 $\mathbf{S}_i$ 和旋转 $\mathbf{R}_i$ 由三角面片的方向和形状决定。不透明度固定为1（因为网格表面不透明）。
    $\{(\boldsymbol{\mu}_i, \mathbf{S}_i, \mathbf{R}_i, \mathbf{n}_i) | i=1,...,K\} = \mathcal{T}(\mathbf{P})$
    - 设计动机：高斯点的形状参数完全由三角面片决定，保证了mesh与3DGS之间的形状一致性。这意味着mesh法线 = 高斯法线，不需要额外的法线学习或正则化。

2. **基于物理的高斯渲染**:

    - 功能：为每个高斯点计算物理可解释的PBR颜色
    - 核心思路：替代原始3DGS的球谐函数，使用GGX微表面模型的PBR渲染方程：
    $\mathbf{L}_o(\mathbf{x}, \boldsymbol{\omega}_o) = \int_{\mathcal{H}^2} \mathbf{f}_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \mathbf{L}_i(\mathbf{x}, \boldsymbol{\omega}_i) |\mathbf{n} \cdot \boldsymbol{\omega}_i| \mathrm{d}\boldsymbol{\omega}_i$
   材质属性（albedo $\mathbf{a}$, roughness $\rho$, metalness $m$）通过多分辨率哈希网格 $\mathcal{E}_d, \mathcal{E}_s$ 查询。用蒙特卡罗采样评估渲染方程积分。
    - 设计动机：物理可解释的材质表示支持真实的重光照（relighting），球谐函数无法做到这一点。

3. **基于网格的高效光传输建模**:

    - 功能：利用显式网格进行高效的自遮挡评估和间接光照建模
    - 核心思路：将入射光分解为直接光 $\mathbf{L}_{\text{dir}}$ 和间接光 $\mathbf{L}_{\text{ind}}$，通过遮挡因子 $O(\mathbf{x}, \boldsymbol{\omega}_i)$ 加权。关键创新：用网格上的二值遮挡 $O_{\text{mesh}} \in \{0,1\}$ 替代高斯点上的连续遮挡 $O_{\text{3dgs}} \in [0,1]$，利用BVH加速的网格射线追踪进行高效遮挡评估。
    $\mathbf{L}_i(\mathbf{x}, \boldsymbol{\omega}_i) = (1-O)\mathbf{L}_{\text{dir}}(\boldsymbol{\omega}_i) + O \cdot \mathbf{L}_{\text{ind}}(\mathbf{x}, \boldsymbol{\omega}_i)$
    - 设计动机：MGadapter 保证了mesh与3DGS的形状一致性，因此 $O_{\text{mesh}} \approx O_{\text{3dgs}}$，替换不引入明显误差。BVH加速的网格射线追踪远比在3D高斯点上累积不透明度更高效。

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{\text{img}} + \lambda_{\text{entropy}}\mathcal{L}_{\text{entropy}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{light}}\mathcal{L}_{\text{light}}$

其中 $\mathcal{L}_{\text{img}} = \mathcal{L}_1 + \lambda_{\text{ssim}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$。

**两阶段训练策略**：初始阶段使用 Split-Sum 近似（无自遮挡，快速预计算）进行 warm-up，几何稳定后切换到蒙特卡罗采样模式实现完整光传输。训练末期可选切换到 Deferred Shading 进行外观细化，改善高频反射效果。

得益于几何引导，不需要 dist loss 或 pseudo depth normal loss 等额外正则化。

## 实验关键数据

### 主实验

| 数据集 | 指标 | GeoSplatting | R3DG | TensoIR | NVdiffrecmc | GS-IR |
|--------|------|-------------|------|---------|-------------|-------|
| Synthetic4Relight | Relighting PSNR↑ | **34.10** | 31.00 | 29.94 | 30.23 | 23.81 |
| TensoIR Synthetic | Relighting PSNR↑ | **29.95** | 28.55 | 28.51 | 26.51 | 24.35 |
| TensoIR Synthetic | Albedo PSNR↑ | **29.41** | 28.74 | 28.35 | 27.71 | 26.80 |
| Shiny Blender | NVS PSNR↑ | **31.14** | 28.83 | 27.89 | 28.03 | 27.01 |
| 训练时间(min) | - | **14** | ~110 | ~270 | 82 | 20 |

**法线质量（MAE↓）**：

| 数据集 | GeoSplatting | R3DG | TensoIR | GS-IR | NVdiffrecmc |
|--------|-------------|------|---------|-------|-------------|
| TensoIR Synthetic | **4.08** | 5.45 | 4.10 | 5.41 | 4.81 |
| Shiny Blender | **2.15** | 7.04 | 4.42 | 4.42 | 9.76 |

### 消融实验

| 配置 | NVS PSNR↑ | Relighting PSNR↑ | Albedo PSNR↑ | Normal MAE↓ |
|------|----------|-----------------|-------------|------------|
| w/o 形状对齐 | 35.95 | 26.39 | 26.72 | 8.29 |
| w/o 外观细化 | 35.07 | 27.95 | 29.31 | 4.42 |
| w/o 遮挡建模 | 35.87 | 27.36 | 27.80 | 6.17 |
| w/o 间接光 | 36.01 | 28.92 | 29.18 | 4.97 |
| 完整模型 | **36.45** | **29.95** | **29.41** | **4.08** |

### 关键发现

1. 形状对齐（MGadapter）是性能提升的最大因素——没有它法线MAE从4.08退化到8.29
2. 遮挡建模对NVS影响不大但对分解至关重要——albedo PSNR从29.41降到27.80
3. GeoSplatting 训练仅14分钟，比隐式场方法（TensoIR ~270min, NeRO ~800min）快一个数量级
4. 在反射性表面（Shiny Blender）上优势最大，法线MAE比R3DG低 69%

## 亮点与洞察

1. **显式+隐式的最佳结合**：网格提供精确几何，3DGS提供高效渲染，MGadapter无缝连接两者。这种混合表示比任一单独使用都更强
2. **无需法线学习**：摆脱了之前3DGS逆渲染方法对法线近似的依赖，从根本上解决了法线不准导致的材质分解噪声
3. **14分钟训练**：在保持SOTA性能的同时达到最快训练速度，对迭代式设计工作流极具价值
4. Split-Sum warm-up + MC采样的两阶段策略平衡了早期稳定性和后期精度

## 局限与展望

1. 依赖等值面技术（FlexiCubes），受网格分辨率限制，难以处理细薄结构和复杂几何
2. 训练需要物体mask，限制了在无mask场景中的直接应用
3. 前向着色（per-Gaussian shading）的细节受高斯点密度限制，虽然通过 Deferred Shading 缓解但增加了复杂性
4. 目前仅支持物体级，扩展到场景级是重要的未来方向

## 相关工作与启发

- **R3DG**：学习额外法线属性并用深度图正则化 → GeoSplatting 直接使用mesh法线，更准确
- **NVdiffrecmc**：同样使用mesh+蒙特卡罗渲染 → GeoSplatting 结合3DGS实现更快训练和更好渲染
- **2DGS/SuGaR**：几何增强的3DGS → GeoSplatting 将几何增强推向逆渲染任务
- 启发：显式几何引导是3DGS走向物理精确渲染的关键路径

## 评分

- **新颖性**: ⭐⭐⭐⭐ MGadapter是优雅的桥接设计，但网格+3DGS混合的思路并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集覆盖漫反射/高光/真实场景，消融全面
- **写作质量**: ⭐⭐⭐⭐⭐ 图示清晰，motivation阐述深入，技术细节完整
- **价值**: ⭐⭐⭐⭐⭐ 14分钟SOTA逆渲染具有高实用价值，混合表示思路有广泛启发性
