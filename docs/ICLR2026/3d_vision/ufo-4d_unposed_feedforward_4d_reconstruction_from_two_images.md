---
title: >-
  [论文解读] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images
description: >-
  [3D视觉] 提出 UFO-4D，一个统一的前馈框架，仅从两张无位姿图像直接预测动态 3D 高斯表示，实现 3D 几何、3D 运动和相机位姿的联合一致估计，在几何和运动基准上比现有方法提升达 3 倍。
tags:
  - 3D视觉
---

# UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images

- **会议**: ICLR 2026
- **arXiv**: [2602.24290](https://arxiv.org/abs/2602.24290)
- **代码**: [项目页面](https://ufo-4d.github.io/)
- **领域**: 3D 视觉 / 4D 重建
- **关键词**: 4D Reconstruction, Dynamic 3D Gaussians, Feedforward, Scene Flow, Unposed, Self-Supervised

## 一句话总结

提出 UFO-4D，一个统一的前馈框架，仅从两张无位姿图像直接预测动态 3D 高斯表示，实现 3D 几何、3D 运动和相机位姿的联合一致估计，在几何和运动基准上比现有方法提升达 3 倍。

## 研究背景与动机

从随意拍摄的图像进行相机位姿、3D 几何和 3D 运动的联合估计（4D 场景重建）是计算机视觉的基础挑战。现有方法存在以下问题：

**测试时优化方法**速度慢（数小时），依赖预计算的深度和光流

**前馈模型**（DUST3R, MonST3R, DynaDUSt3R）在单独任务上效果好，但缺乏统一架构

**4D 训练数据稀缺**：合成数据有域差距，真实数据标注稀疏且噪声大

**核心 insight**：从单一动态 3D 高斯表示进行可微渲染多种信号（外观、深度、运动）可以提供自监督训练信号，并通过几何耦合使各监督信号相互正则化。

## 方法详解

### 整体架构

给定两张无位姿图像 $\mathbf{I}_t, \mathbf{I}_{t+1}$ 和相机内参：

$$f_\theta(\mathbf{I}_t, \mathbf{I}_{t+1}) \mapsto (\mathcal{G}, \mathbf{P})$$

输出动态 3D 高斯集合 $\mathcal{G}$ 和相对相机位姿 $\mathbf{P}$。每个动态高斯包含：
- 3D 中心 $\boldsymbol{\mu} \in \mathbb{R}^3$
- 3D 运动 $\mathbf{v} \in \mathbb{R}^3$
- 协方差参数（四元数旋转 $\mathbf{r}$，尺度 $\mathbf{s}$）
- 球谐颜色 $\mathbf{h}$，不透明度 $o$

### 网络架构

- **编码器**：权重共享 ViT 分别处理两张图像
- **解码器**：ViT with 交叉注意力层融合两图信息
- **头部**：
    - 中心头（DPT）→ 3D 位置
    - 属性头（DPT）→ 旋转、尺度、颜色、不透明度
    - 速度头（DPT）→ 3D 运动向量
    - 位姿头（3层 MLP）→ 相对位姿（平移+四元数）
- **初始化**：NoPoSplat（高斯头）+ MASt3R（其余）

### 可微 4D 光栅化

**关键创新**：扩展标准 3DGS 光栅化器，统一渲染图像、点图和场景流。

时间插值（线性运动假设）：

$$\mathcal{G}(t') = \{(\boldsymbol{\mu} + \Delta t \cdot \mathbf{v}, \mathbf{v}, \mathbf{r}, \mathbf{s}, \mathbf{h}, \mathbf{c}, o)_\mathbf{p}\}$$

统一 $\alpha$-blending 渲染点图和运动图：

$$\mathbf{X}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \boldsymbol{\mu}_i o_i \prod_{j=1}^{i-1}(1-o_j)$$

$$\mathbf{V}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \mathbf{v}_i o_i \prod_{j=1}^{i-1}(1-o_j)$$

### 损失函数

**总损失** = 监督损失 + 自监督损失：

$$L_{total} = L_{sup} + L_{self}$$

**监督损失**（场景流 + 点图 + 位姿）：

$$L_{sup} = L_{motion} + w_{point} L_{point} + w_{pose} L_{pose}$$

- $L_{motion}$：同时约束高斯中心运动 $\mathbf{v}$ 和渲染运动 $\mathbf{V}$
- $L_{point}$：同时约束高斯位置 $\boldsymbol{\mu}$ 和渲染点图 $\mathbf{X}$
- $L_{pose}$：分别约束平移和四元数

**自监督损失**（光度 + 平滑）：

$$L_{self} = L_{photo} + w_{smooth} L_{smooth}$$

- $L_{photo} = \text{MSE} + w_{lpips} \text{LPIPS}$
- $L_{smooth}$：边缘感知平滑正则化

### 下游任务

- 深度 = 点图最后一个通道
- 光流 = 3D 场景流的 2D 投影
- 运动分割 = 场景流阈值化
- 4D 插值 = 任意时间和视角的渲染

## 实验

### 训练数据

混合使用：Stereo4D (60%) + PointOdyssey (20%) + Virtual KITTI 2 (20%)

### 主要结果

**几何估计**（点图 EPE, 深度指标）：

| 方法 | Stereo4D EPE | KITTI EPE | Sintel EPE |
|------|-------------|-----------|-----------|
| DynaDUSt3R | ~0.15 | ~0.80 | - |
| ZeroMSF | ~0.12 | ~0.65 | - |
| **UFO-4D** | **~0.05** | **~0.25** | **最优** |

UFO-4D 在 Stereo4D 和 KITTI 上比竞争方法提升 **3×以上**。

**运动估计**（场景流 EPE）：同样大幅领先。

### 关键发现

1. **自监督损失**极大改善了几何和运动估计质量
2. 直接位姿估计优于后处理回归（DUSt3R 方式）
3. 合成+真实混合训练有效缓解域差距
4. 4D 插值在新视角和时间步均表现良好

### 4D 插值应用

首次实现从前馈输出的时空插值：在任意中间时间点和视角渲染图像、深度和运动。

## 亮点

1. **统一表示**：单一动态 3D 高斯表示同时解决几何、运动、位姿估计
2. **自监督训练**：光度重建损失无需标注，有效克服数据稀缺
3. **耦合正则化**：几何和运动共享高斯基元，各监督信号互相正则化
4. **新应用**：前馈 4D 插值（图像+几何+运动的时空插值）
5. **性能跃升**：EPE 指标提升 3 倍以上

## 局限性

1. 线性运动假设限制了复杂非刚体运动的建模
2. 仅处理两帧输入，无法建模长期时序依赖
3. 依赖相机内参作为输入（虽通常可获取）
4. 训练数据混合策略的最优比例需手动调整
5. 大面积遮挡区域的重建可能不够准确

## 相关工作

- **静态 3D 重建**：DUSt3R (Wang et al., 2024b), MASt3R (Leroy et al., 2024) 学习强先验实现端到端重建
- **动态 3D 重建**：MonST3R (Zhang et al., 2025a) 微调静态模型处理动态场景，但缺乏时间对应
- **密集 4D 重建**：测试时优化方法质量高但慢；现有前馈方法需位姿输入或分离的任务头

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 统一 4D 表示 + 自监督框架是重要贡献
- **实用性**: ⭐⭐⭐⭐ — 前馈推理、实时应用潜力大
- **清晰度**: ⭐⭐⭐⭐⭐ — 方法描述系统清晰，公式推导完整
- **意义**: ⭐⭐⭐⭐⭐ — 将密集 4D 重建从优化推向前馈时代

<!-- RELATED:START -->

## 相关论文

- [Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](../../CVPR2025/3d_vision/dyn_hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)

<!-- RELATED:END -->
