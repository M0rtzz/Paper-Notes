---
title: >-
  [论文解读] 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians
description: >-
  [CVPR 2025][3D视觉][4D SLAM] 本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。
tags:
  - CVPR 2025
  - 3D视觉
  - 4D SLAM
  - 非刚性重建
  - 2D高斯溅射
  - 变形场
  - 动态场景
---

# 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians

**会议**: CVPR 2025  
**arXiv**: [2505.22859](https://arxiv.org/abs/2505.22859)  
**代码**: [https://muskie82.github.io/4dtam/](https://muskie82.github.io/4dtam/)  
**领域**: 3D视觉  
**关键词**: 4D SLAM, 非刚性重建, 2D高斯溅射, 变形场, 动态场景

## 一句话总结
本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。

## 研究背景与动机
视觉SLAM技术在过去二十年取得了巨大进展，但绝大多数方法假设场景是静态的。现实世界中充满了运动的元素——河流、树木、行人等。虽然可以检测并排除动态物体来聚焦静态部分，但这样做丢失了场景的时空信息。真正的4D重建（三维空间+时间）需要同时处理相机运动和场景非刚性形变，这是一个极其困难的问题，因为优化空间维度极高，单相机的观测又非常稀疏。

现有的非刚性SLAM方法（如DynamicFusion系列）通常依赖TSDF体表示和特定的形变模型，重建精度有限。近年来3D高斯溅射（3DGS）在静态SLAM中展现了强大能力，但其blob-like的表示不适合精确的表面重建。另一方面，大多数动态重建方法要求已知相机位姿或多相机系统，限制了实际应用。

本文的核心切入点是：利用2D高斯溅射（2DGS）作为表面基元，结合MLP变形场来建模非刚性运动，同时推导解析的相机位姿雅可比矩阵实现高效的位姿估计，从而构建一个完整的4D SLAM系统。

## 方法详解

### 整体框架
4DTAM采用经典的跟踪-建图（Tracking-Mapping）双线程架构。跟踪模块负责快速在线位姿估计；建图模块在滑动窗口内联合优化相机位姿、规范空间高斯、变形场参数。输入为单目RGB-D视频流，输出为完整的4D时空重建模型。

### 关键设计

1. **2DGS表面基元表示**:

    - 与3DGS不同，2DGS将每个高斯约束在一个2D切平面上，天然具有明确的表面法向量方向
    - 每个2D高斯由3D均值位置、旋转矩阵（分解为两个切向量和一个法向量）、颜色、不透明度和2D缩放向量表示
    - 通过射线-溅射交叉（ray-splat intersection）实现高效渲染，避免了数值不稳定的矩阵求逆
    - 这种表面表示能更好地利用深度信号，对单相机非刚性重建至关重要

2. **解析相机位姿雅可比矩阵**:

    - 这是论文的一个重要技术贡献：为2DGS推导了完整的解析位姿梯度
    - 使用李代数参数化SE(3)位姿，推导了变换矩阵M^T对位姿参数τ的偏导数
    - 同时推导了渲染法向量对相机位姿的雅可比矩阵
    - 通过CUDA内核实现，保持了高斯溅射的实时渲染优势
    - 这个公式化具有广泛的应用潜力，不仅限于SLAM

3. **MLP变形场（Warp Field）**:

    - 使用一个紧凑的MLP网络作为变形场，将规范空间的高斯映射到各时间步的变形空间
    - 输入为时间t和高斯中心位置x的频率位置编码，输出位移δx、旋转偏移δr和缩放偏移δs
    - MLP的连续性天然提供了运动的平滑性先验
    - 使用CUDA优化的MLP实现（tiny-cuda-nn）保证运行效率

4. **跟踪模块**:

    - 最小化当前帧与变形后高斯模型渲染之间的光度误差和深度误差
    - 关键设计：相机位姿相对于最新关键帧时刻的变形高斯进行估计，假设变形场景在时间上连续变化
    - 每N帧选取关键帧送入建图模块

5. **建图模块与正则化**:

    - 新关键帧到来时，根据RGB-D观测反投影生成新的规范空间高斯
    - 创新性地使用深度图的有限差分计算表面法向量来初始化2DGS的法向量，这比随机初始化效果更好
    - 引入基于传感器法向量的监督损失，避免了2DGS原始方法中每步计算渲染深度差分法向量的高开销
    - ARAP（As-Rigid-As-Possible）正则化约束相邻高斯之间的相对位移保持刚性
    - 创新性的法向量刚性损失：约束相邻高斯的法向量在不同时间步之间保持类似的相对关系

### 损失函数 / 训练策略
总损失函数：$L_{total} = \lambda_p L_p + \lambda_g L_g + \lambda_n L_n + \lambda_{iso} L_{iso} + L_{ARAP} + L_{ARAP\_n}$

- $L_p$: 光度渲染损失（L1）
- $L_g$: 深度渲染损失（L1）
- $L_n$: 法向量一致性损失（基于传感器测量）
- $L_{iso}$: 各向同性损失
- $L_{ARAP}$: 位置刚性正则化
- $L_{ARAP\_n}$: 法向量刚性正则化（新提出）

全局优化阶段：跟踪完成后，固定位姿和高斯数量，随机选择关键帧进行全局优化，在RTX 4090上约1分钟。

## 实验关键数据

### 主实验
| 数据集 | 指标 | 4DTAM | SurfelWarp | 提升 |
|--------|------|-------|------------|------|
| Sim4D (curtain) | ATE RMSE (cm) | **0.25** | 6.10 | 24x |
| Sim4D (flag) | ATE RMSE (cm) | **1.00** | 31.9 | 32x |
| Sim4D (mercedes) | PSNR (dB) | **32.13** | 25.7 | +6.4 |
| Sim4D (shoe_rack) | L1 Depth (cm) | **0.99** | 4.25 | 4x |
| Sim4D (平均) | ATE RMSE (cm) | **~0.35** | ~7.0 | ~20x |

### 消融实验
| 配置 | ATE (cm) | Depth L1 (cm) | F1 (%) | 说明 |
|------|---------|---------------|--------|------|
| MonoGS (3DGS) | 0.59 | 4.52 | 31.9 | 基线静态SLAM |
| MonoGS-2D (本文2DGS) | **0.36** | **0.54** | **88.8** | 2DGS表面基元大幅提升几何重建 |

离线非刚性重建消融：
| 配置 | 指标 | 本文 | Morpheus | 说明 |
|------|------|------|----------|------|
| iPhone数据集 | Depth L1 (cm) | **0.57** | 2.4 | 几何更准 |
| iPhone数据集 | LPIPS | **0.26** | 0.63 | 渲染质量大幅提升 |

### 关键发现
- 2DGS作为SLAM表面表示比3DGS在几何重建上有质的飞跃（F1从31.9%到88.8%）
- 法向量初始化对2DGS至关重要，基于深度传感器的初始化远优于随机初始化
- 法向量刚性正则化有效防止了非刚性变形中的表面撕裂
- 相机位姿估计速度约1.5 fps

## 亮点与洞察
- 首次将2DGS引入SLAM，并推导了完整的解析位姿雅可比，这对整个GS-SLAM领域都有参考价值
- 法向量刚性损失的提出很巧妙——利用2DGS的表面法向量特性来约束变形的局部刚性，这是3DGS做不到的
- Sim4D数据集的构建思路值得借鉴：利用大规模开源3D模型和动画建模，通过Blender渲染生成带完整标注的4D数据
- 从公式推导到CUDA实现，工程完整度很高

## 局限与展望
- 跟踪速度仅1.5fps，离实时（30fps）还有距离
- 全局优化仍需额外1分钟后处理
- 目前依赖RGB-D输入，纯RGB扩展仅在补充材料中简要展示
- Sim4D数据集主要是单个物体的动态，尚未涵盖大规模多物体动态场景
- 拓扑变化（如物体断裂）尚未有效处理

## 相关工作与启发
- DynamicFusion系列开创了基于TSDF的非刚性SLAM，但几何精度受限于体素分辨率
- MonoGS验证了高斯溅射在SLAM中的可行性，本文将其扩展到2DGS和动态场景
- 变形场的MLP表示借鉴了D-NeRF和Nerfies的思想，但首次将其与2DGS和SLAM框架结合
- 与DyNoMo相比，本文同时支持位姿优化和高质量几何重建

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个基于2DGS的4D SLAM，多项技术创新（解析位姿雅可比、法向量刚性损失）
- 实验充分度: ⭐⭐⭐⭐ 新数据集+多角度消融，但real-world评估主要是定性
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导完整，论文结构清晰
- 价值: ⭐⭐⭐⭐⭐ 开辟了现代4D-SLAM的研究方向，数据集和评估协议将促进后续研究

<!-- RELATED:START -->

## 相关论文

- [GauSTAR: Gaussian Surface Tracking and Reconstruction](gaustar_gaussian_surface_tracking_and_reconstruction.md)
- [GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](../../ECCV2024/3d_vision/animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)
- [Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)

<!-- RELATED:END -->
