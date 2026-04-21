---
title: >-
  [论文解读] VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians
description: >-
  [ICML2025][3D视觉][SLAM] 提出视图绑定3D高斯（View-Tied 3D Gaussians），将高斯绑定到深度像素上并简化为球形，大幅节省存储开销，配合仅优化最近视图相关高斯的tracking/mapping策略，实现面向大规模场景的可扩展RGBD SLAM系统。
tags:
  - ICML2025
  - 3D视觉
  - SLAM
  - 3D高斯泼溅
  - 视图绑定高斯
  - 大规模场景
  - RGBD
  - 相机追踪
  - 场景重建
---

# VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians

**会议**: ICML2025  
**arXiv**: [2506.02741](https://arxiv.org/abs/2506.02741)  
**代码**: [项目主页](https://machineperceptionlab.github.io/VTGaussian-SLAM-Project)  
**领域**: 3d_vision  
**关键词**: SLAM, 3D高斯泼溅, 视图绑定高斯, 大规模场景, RGBD, 相机追踪, 场景重建

## 一句话总结

提出视图绑定3D高斯（View-Tied 3D Gaussians），将高斯绑定到深度像素上并简化为球形，大幅节省存储开销，配合仅优化最近视图相关高斯的tracking/mapping策略，实现面向大规模场景的可扩展RGBD SLAM系统。

## 研究背景与动机

- **SLAM的核心任务**：从RGBD图像中同时估计相机位姿并建图，是计算机视觉、3D重建和机器人领域的基础问题
- **NeRF时代的局限**：基于NeRF的SLAM方法通过体渲染实现连续场景表示，但光线追踪效率低，难以满足实时性要求
- **3DGS的引入与优势**：3D高斯泼溅（3DGS）通过可微光栅化实现高质量实时渲染，为SLAM提供了新的场景表示范式
- **现有3DGS-SLAM的瓶颈**：现有方法需要在有限GPU内存中维护覆盖整个场景的所有3D高斯，并在训练全程优化所有高斯以保持与历史帧的颜色和几何一致性，导致无法扩展到极大规模场景
- **存储效率问题**：原始3DGS每个高斯需要存储位置(3)、旋转(4)、多维方差(6)、颜色(球谐系数)等大量参数，高斯数量受限
- **核心矛盾**：场景越大需要的高斯越多，但GPU内存有限；同时需要保持全局一致性又增加了优化负担，二者形成根本矛盾

## 方法详解

### 整体框架

VTGaussian-SLAM包含两大核心组件：**视图绑定3D高斯表示**和**基于此表示的新型tracking/mapping策略**。系统接收RGBD流作为输入，将每帧的深度像素与简化的球形高斯绑定，仅在需要时加载和优化与当前视图相关的高斯子集，实现对大规模场景的处理能力。

### 关键设计一：视图绑定3D高斯（View-Tied 3D Gaussians）

| 属性 | 做什么 | 核心思路 | 设计动机 |
|------|--------|----------|----------|
| 位置绑定 | 将高斯绑定到深度像素，位置由深度值和相机位姿唯一确定 | 位置 = R * unproj(u,v,d) + t | 无需学习和存储位置参数，省去density control |
| 球形简化 | 将椭球形高斯简化为球形，去除旋转和多维方差参数 | 各向同性高斯，仅需单一半径参数 | 进一步节省存储，在SLAM场景中球形已足够描述局部几何 |
| 属性精简 | 每个高斯仅保留颜色、不透明度和半径 | 去除位置(3)、旋转(4)、协方差(6)共13个参数 | 有限GPU内存中可容纳更多高斯，描述更多局部细节 |

### 关键设计二：新型Tracking策略

| 属性 | 做什么 | 核心思路 | 设计动机 |
|------|--------|----------|----------|
| 局部优化 | 仅渲染和优化与最近视图相关的高斯 | 不再维护全场景高斯的全局一致性 | 避免大场景中优化全部高斯导致的内存溢出和效率问题 |
| 位姿估计 | 通过最小化当前帧渲染图与观测图差异来优化相机位姿 | 利用view-tied高斯的可微渲染梯度反传到位姿参数 | 高斯位置由位姿决定，位姿更新自动带动高斯位置更新 |
| 关键帧选择 | 选取代表性关键帧子集用于tracking参考 | 基于视角变化和overlap判断 | 减少需要加载到内存的历史帧高斯数量 |

### 关键设计三：新型Mapping策略

- **功能**：对每个新的关键帧，仅优化与其相关的视图绑定高斯的颜色、不透明度和半径参数
- **核心思路**：映射过程不需要维护对所有历史关键帧的一致性约束，而是通过view-tied的绑定关系自然保证局部一致性
- **设计动机**：传统方法需要在mapping时将所有高斯保持可学习状态以维持全局一致性，这是内存消耗的主要来源；view-tied设计将高斯绑定到特定视图后，移出内存的高斯不影响当前帧的优化

### 损失函数与训练策略

- **渲染损失**：结合RGB图像和深度图的光度损失，监督高斯属性的优化
- **增量式处理**：按帧顺序处理RGBD流，每帧先tracking估计位姿，再mapping优化局部高斯
- **内存管理**：仅将当前相关的高斯子集加载到GPU，处理完毕后可释放，实现对场景规模的解耦

## 实验关键数据

### 主实验：渲染质量与追踪精度对比

| 方法 | 场景表示 | 可扩展性 | 渲染质量 | 追踪精度 | 需维护全局高斯 |
|------|----------|----------|----------|----------|----------------|
| SplaTAM | 3DGS | 差 | 中等 | 中等 | 是 |
| MonoGS | 3DGS | 差 | 较好 | 较好 | 是 |
| Gaussian-SLAM | 3DGS | 中等 | 较好 | 较好 | 是 |
| Photo-SLAM | 3DGS+隐式 | 中等 | 好 | 较好 | 是 |
| **VTGaussian-SLAM** | **View-Tied 3DGS** | **强** | **最好** | **最好** | **否** |

### 消融实验：各组件贡献

| 配置 | 渲染质量变化 | 追踪精度变化 | 说明 |
|------|-------------|-------------|------|
| 完整模型 | 基准 | 基准 | VTGaussian-SLAM全部组件 |
| 去除球形简化（改用椭球） | 未显著提升 | 相当 | 说明球形简化足够，且节省大量参数 |
| 去除view-tied绑定（学习位置） | 下降 | 下降 | 恢复为传统3DGS，丧失可扩展性优势 |
| 改用全局优化策略 | 轻微提升 | 相当 | 但内存消耗大幅增加，不具可扩展性 |

### 关键发现

- 视图绑定策略使得在相同GPU内存下可使用数倍于传统方法的高斯数量，弥补了简化带来的表达力下降
- 在大规模场景（多房间/长序列）上，传统3DGS-SLAM方法因内存溢出无法运行，VTGaussian-SLAM仍可正常工作
- 球形简化在SLAM场景中几乎不损失渲染质量，说明SLAM中高斯主要起局部颜色/几何采样作用而非精确椭球建模

## 亮点与洞察

1. **表示与算法的协同设计**：view-tied高斯不仅是存储优化，更从根本上改变了tracking/mapping的优化范式——从全局一致性优化转为局部增量优化
2. **以约束换自由度**：通过将高斯位置绑定到深度像素（强约束），换来了内存使用上的极大自由度，是做减法的典型范例
3. **简化不等于退化**：球形高斯看似是对模型表达力的妥协，但在GPU内存限制下可以使用更多高斯反而提升了整体质量，体现了数量vs质量的trade-off

## 局限性与可改进方向

- **依赖深度输入**：方法强依赖RGBD传感器提供的深度图，无法直接应用于纯RGB SLAM场景
- **无回环检测**：当前系统未集成loop closure模块，长序列轨迹可能累积漂移
- **动态场景**：论文未讨论动态物体的处理，view-tied策略在动态场景中可能产生伪影
- **球形假设的边界**：对于细长结构（如栏杆、电线），球形高斯可能需要更多数量来逼近，存在效率边界
- **场景编辑能力**：view-tied绑定使得高斯难以独立于视图进行编辑或重用

## 相关工作与启发

- **SplaTAM/MonoGS**：代表当前3DGS-SLAM方法的主流范式，需要维护全局高斯——本文正是针对其可扩展性瓶颈提出改进
- **NICE-SLAM/Co-SLAM**：基于NeRF的SLAM方法，使用分层特征网格实现局部更新，启发了本文的局部优化思路
- **Dynamic 3DGS**：将高斯绑定到实体的思想在动态场景重建中也有应用，但本文面向SLAM的绑定策略（绑定到深度像素）更为简洁
- **启发**：view-tied的核心理念——将表示与观测绑定——可迁移到其他需要大规模场景处理的3DGS应用中，如自动驾驶场景重建

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐ | view-tied高斯是对3DGS-SLAM范式的本质性改进 |
| 技术深度 | ⭐⭐⭐⭐ | 表示设计与优化策略协同，从理论到系统层面都有贡献 |
| 实验充分度 | ⭐⭐⭐⭐ | 在标准benchmark上超越最新方法，消融实验验证各组件 |
| 写作清晰度 | ⭐⭐⭐⭐ | 论文结构清晰，图示直观，motivation明确 |

<!-- RELATED:START -->

## 相关论文

- [SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](../../CVPR2026/3d_vision/sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)
- [WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](../../ECCV2024/3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [4D Gaussian Splatting SLAM](../../ICCV2025/3d_vision/4d_gaussian_splatting_slam.md)
- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](../../CVPR2025/3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)

<!-- RELATED:END -->
