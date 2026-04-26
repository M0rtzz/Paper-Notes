---
title: >-
  [论文解读] Generative Gaussian Splatting for Unbounded 3D City Generation
description: >-
  [CVPR 2025][自动驾驶][3D高斯溅射] 提出 GaussianCity，首个将 3D 高斯溅射应用于无界 3D 城市生成的框架，通过引入 BEV-Point 紧凑中间表示使显存占用与场景规模解耦（保持恒定），并设计 Point Serializer 将无序 BEV 点转为有序序列以捕获结构和上下文特征，在无人机视角和街景视角的城市生成中达到 SOTA，渲染速度比 CityDreamer（基于 NeRF）快 60 倍。
tags:
  - CVPR 2025
  - 自动驾驶
  - 3D高斯溅射
  - 无界城市生成
  - BEV-Point表示
  - 高效渲染
  - 场景合成
---

# Generative Gaussian Splatting for Unbounded 3D City Generation

**会议**: CVPR 2025  
**arXiv**: [2406.06526](https://arxiv.org/abs/2406.06526)  
**代码**: https://haozhexie.com/project/gaussian-city  
**领域**: 3D视觉 / 场景生成  
**关键词**: 3D高斯溅射, 无界城市生成, BEV-Point表示, 高效渲染, 场景合成

## 一句话总结

提出 GaussianCity，首个将 3D 高斯溅射应用于无界 3D 城市生成的框架，通过引入 BEV-Point 紧凑中间表示使显存占用与场景规模解耦（保持恒定），并设计 Point Serializer 将无序 BEV 点转为有序序列以捕获结构和上下文特征，在无人机视角和街景视角的城市生成中达到 SOTA，渲染速度比 CityDreamer（基于 NeRF）快 60 倍。

## 研究背景与动机

**领域现状**：3D 城市生成是 3D 内容创作中最具挑战性的任务之一，在游戏、动画、影视和 VR 等领域有广泛应用。InfiniCity 和 CityDreamer 使用 NeRF 实现了无界城市生成，取得了不错的效果。近一年来，3D 高斯溅射（3D-GS）因其高效的 GPU 光栅化渲染和灵活的细节表达能力，在 3D 生成中获得广泛关注。

**现有痛点**：NeRF 基方法对所有点以相同密度采样并沿光线聚合，推理效率低且丢失细节。而现有 3D-GS 生成模型仅能处理有限规模的物体或场景。当场景扩展到城市尺度时，高斯点数需要膨胀到数十亿级别，一个 10km² 的城市场景可能需要数百 GB 的显存，这使得直接将 3D-GS 用于无界城市生成完全不可行。

**核心矛盾**：3D-GS 的高效渲染优势与大规模场景下显存和存储的爆炸性增长之间存在根本冲突。

**本文目标** 如何在保留 3D-GS 高效渲染的同时，使显存消耗不随场景规模增长，从而实现无界 3D 城市的生成？

**切入角度**：作者观察到在固定相机参数下，无论场景多大，每一帧中可见的 BEV 点数量是恒定的。因此可以将高斯属性解耦为位置相关部分（编码到 BEV 地图中）和风格相关部分（编码到查找表中），只对可见点做渲染和优化。

**核心 idea**：用 BEV-Point 作为紧凑中间表示，仅保留可见点并通过 Point Serializer 恢复空间结构，实现显存恒定的无界城市高斯生成。

## 方法详解

### 整体框架

GaussianCity 的流程分为四个步骤：(1) 从 BEV 地图（高度图 $\mathbf{H}$、语义图 $\mathbf{S}$、密度图 $\mathbf{D}$）的局部 patch 生成 BEV 点并筛选可见点；(2) 为每个点生成 BEV-Point 属性（实例标签、坐标、场景特征）和为每个实例生成风格查找表；(3) BEV-Point Decoder 通过 Point Serializer + Point Transformer + Modulated MLP 生成高斯属性；(4) Gaussian Rasterizer 渲染最终图像。

### 关键设计

1. **BEV-Point 紧凑表示**:

    - 功能：将高斯属性解耦并压缩，使显存占用与场景规模无关
    - 核心思路：通过 BEV 地图的语义图和高度图，将像素沿高度方向拉伸生成 3D 点集 $\mathbf{C}_F$。引入二值密度图 $\mathbf{D}$ 对不同语义类别自适应调整采样密度（路面等简单纹理降低密度，建筑立面等复杂纹理增加密度）。关键步骤是通过射线求交得到可见性图 $\mathcal{V}$，只保留可见点 $\mathbf{C}_A$。由于固定相机参数下可见点数恒定，显存消耗不随场景扩大而增长。
    - 设计动机：与直接用 3D-GS（显存随点数线性增长）相比，BEV-Point 将全场景 ~2000 万个 BEV 点压缩到可见的 ~几十万个，且文件存储也从属性维度（每点 ~60 维）压缩为 BEV 地图 + 查找表的形式。

2. **Point Serializer**:

    - 功能：将无序的 BEV 点转为有空间局部性的有序序列
    - 核心思路：设计 linearization 函数 $\mathcal{L}(x,y,z,g) = \lfloor x/g^2 + y/g + z \rfloor$，将每个点的 3D 坐标映射为一个整数序号。通过栅格大小 $g$ 控制离散化粒度。排序后，数据结构中相邻的点在空间中也是相邻的，从而使后续的 Transformer 可以在局部窗口内有效捕获上下文关系。
    - 设计动机：NeRF 中沿光线采样的点天然保持空间相关性，但 BEV 点是无序的点云，直接用 MLP 处理会丢失空间结构。Serializer 通过空间填充曲线的思想恢复了这种空间局部性。

3. **风格查找表与 Modulated MLP**:

    - 功能：以低成本控制不同实例的外观多样性
    - 核心思路：为每个实例（建筑、车辆等）学习一个风格向量 $\mathbf{z}_T^i \sim \mathcal{N}(0,1)$，存入查找表 $\mathcal{T}$。在生成高斯属性时，通过 Modulated MLP 将 BEV 点特征在风格向量的调制下生成最终的颜色、缩放、旋转等属性。这样同一栋建筑的所有 BEV 点共享一个风格码，大幅减少参数量。
    - 设计动机：城市场景中建筑和车辆的外观多样性需要实例级控制，但全局 3D-GS 存储每个点的独立属性太昂贵。风格表将高维属性压缩为低维风格码 + 预测网络。

### 损失函数 / 训练策略

使用混合损失：$\ell = \lambda_{L1} \|\hat{\mathbf{R}} - \mathbf{R}\| + \lambda_{VGG} \text{VGG}(\hat{\mathbf{R}}, \mathbf{R}) + \lambda_{GAN} \text{GAN}(\hat{\mathbf{R}}, \mathbf{S}_G)$，其中 L1 损失保证像素级重建，VGG 感知损失保证语义一致性，GAN 对抗损失（以语义图为条件）保证真实感。

## 实验关键数据

### 主实验

在 GoogleEarth 和 KITTI-360 数据集上的城市生成质量对比：

| 方法 | FID↓ | KID↓ | 渲染速度 (FPS) | 数据集 |
|------|------|------|---------------|--------|
| CityDreamer | 66.05 | 3.06 | 0.18 | GoogleEarth (Drone) |
| **GaussianCity** | **57.92** | **2.58** | **10.72** | GoogleEarth (Drone) |
| CityDreamer | 78.41 | 6.35 | 0.17 | KITTI-360 (Street) |
| **GaussianCity** | **69.23** | **5.14** | **9.89** | KITTI-360 (Street) |

GaussianCity 在 FID/KID 指标上全面优于 CityDreamer，且渲染速度快 **60 倍**（10.72 vs 0.18 FPS）。

### 消融实验

| 配置 | FID↓ | KID↓ | 说明 |
|------|------|------|------|
| Full model | 57.92 | 2.58 | 完整模型 |
| w/o Point Serializer | 63.18 | 3.12 | 无序列化，质量明显下降 |
| w/o 密度图 $\mathbf{D}$ | 60.45 | 2.89 | 简单区域过度采样 |
| w/o 风格查找表 | 62.37 | 3.04 | 实例多样性降低 |
| w/o Scene Feature $\mathbf{F}_S$ | 61.93 | 2.96 | 缺少上下文信息 |

### 关键发现

- Point Serializer 对生成质量贡献最大（去掉后 FID 上升 5+），说明将无序点云结构化处理对于生成任务至关重要。
- 密度图 $\mathbf{D}$ 的自适应采样策略有效平衡了质量和效率——简单纹理区域（路面）少采样，复杂区域（建筑立面）密采样。
- BEV-Point 表示在显存方面真正实现了"恒定"——当场景从 1km² 扩大到 10km² 时，显存仅从 3.2GB 到 3.3GB，而直接用 3D-GS 显存从 12GB 增长到超过 120GB。

## 亮点与洞察

- **恒定显存的无界场景生成**：通过"只看可见点"这一简洁 insight，将显存从随场景线性增长变为恒定，这个思路可以迁移到任何基于点表示的大规模场景任务。
- **Point Serializer 的空间局部性恢复**：用一维排序函数恢复三维空间局部性，使标准 Transformer 可以直接处理无序点云，比复杂的图注意力或点云网络更简洁高效。
- 60 倍加速的实际意义巨大——从 0.18 FPS（不可用于交互）到 10.72 FPS（接近实时），使得 3D 城市生成首次具备了交互式应用的可能。

## 局限与展望

- 目前依赖预定义的 BEV 地图（语义图、高度图）作为输入，不具备从头生成布局的能力，可考虑结合布局生成模型。
- 建筑内部结构完全不可见，仅为外壳渲染，对于街景模式近距离观察时可能穿帮。
- Point Serializer 使用简单的线性映射，对于复杂拓扑（如桥梁、高架）的空间填充可能不够理想，可尝试 Z-order 曲线或 Hilbert 曲线。
- 对动态物体（行人、车辆运动）没有建模，当前仅能生成静态城市。

## 相关工作与启发

- **vs CityDreamer**: CityDreamer 用 NeRF 实现无界城市生成但渲染极慢，GaussianCity 用 3D-GS 获得 60 倍加速且质量更好。核心区别在于 BEV-Point 解决了 3D-GS 的显存瓶颈。
- **vs InfiniCity**: InfiniCity 同样基于 NeRF 且通过分块生成实现无界，但分块边界处可能不连续；GaussianCity 的 BEV-Point 天然支持跨块一致性。
- **vs LGM/GS-LRM**: 这些 3D-GS 生成方法仅适用于单个物体级别，GaussianCity 首次扩展到城市级别。

## 评分

- 新颖性: ⭐⭐⭐⭐ BEV-Point 解耦思路新颖且工程上有重要意义
- 实验充分度: ⭐⭐⭐⭐ 显存分析、效率对比和消融都很充分
- 写作质量: ⭐⭐⭐⭐ 图示精美，方法阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 60 倍加速使城市级 3D 生成首次实用化

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](../../ICCV2025/autonomous_driving/3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)
- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](pansplat_4k_panorama_synthesis_with_feed-forward_gaussian_splatting.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)

<!-- RELATED:END -->
