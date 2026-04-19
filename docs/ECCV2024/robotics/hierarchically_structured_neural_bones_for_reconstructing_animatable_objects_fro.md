---
title: >-
  [论文解读] Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos
description: >-
  [ECCV2024][机器人][3D重建] 提出层次化神经骨骼（Hierarchical Neural Bones）框架，通过树状结构的骨骼系统以粗到细的方式分解物体运动，从随手拍摄的视频中重建可操控的高质量 3D 模型。
tags:
  - ECCV2024
  - 机器人
  - 3D重建
  - Neural Bones
  - Hierarchical Deformation
  - NeRF
  - Manipulation
---

# Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos

**会议**: ECCV2024  
**arXiv**: [2408.00351](https://arxiv.org/abs/2408.00351)  
**代码**: [subin6/HSNB](https://github.com/subin6/HSNB)  
**领域**: robotics  
**关键词**: Animatable 3D Reconstruction, Neural Bones, Hierarchical Deformation, NeRF, Manipulation

## 一句话总结

提出层次化神经骨骼（Hierarchical Neural Bones）框架，通过树状结构的骨骼系统以粗到细的方式分解物体运动，从随手拍摄的视频中重建可操控的高质量 3D 模型。

## 背景与动机

从随手拍摄的视频中构建可操控的 3D 模型具有重要的应用价值（影视、混合现实、游戏等），但现有方法存在两大问题：

1. **依赖先验模板的方法**（如骨架、3D 人体模型）需要大量 3D 扫描数据或标注，缺乏对任意物体的通用性
2. **无模板方法**（如 BANMo）虽然使用高斯椭球体作为控制点具有通用性，但骨骼分布缺乏结构，散落在物体表面，不考虑运动粒度，导致：
    - 骨骼之间缺乏关联性，用户难以理解和操控
    - 需要大量输入视频才能产生合理结果
    - 重建质量有提升空间

## 核心问题

如何在不依赖任何先验结构知识的前提下，学习一组有层次结构的控制点，使其既能准确捕获物体运动、提升重建质量，又能为用户提供直观、易操控的 3D 模型编辑接口？

## 方法详解

整体框架基于 BANMo 构建，核心改进为 **层次化变形模型** 和 **骨骼占用函数**。

### 1. 基础框架（BANMo）

- **Canonical Model**：用 NeRF 表示物体的形状和外观，输出颜色和 SDF 值
- **Deformation Model**：使用高斯椭球体骨骼和 Linear Blend Skinning (LBS) 将每帧 3D 点变形到 canonical 空间
- 通过可微体渲染联合优化所有组件

### 2. 层次化神经变形模型（Hierarchical Neural Deformation Model）

**核心思想**：将骨骼组织为树状结构，父骨骼捕获大区域的粗运动，子骨骼表示更细粒度部位的精细运动。

- **骨骼层次公式**：深度为 $d$ 的骨骼变换通过递归左乘所有父级变换得到：
$$T^d = \hat{T}^1 \hat{T}^2 \cdots \hat{T}^{d-1} \hat{T}^d$$
子骨骼在父骨骼的局部坐标系中定义，自然继承父级运动。

- **神经骨骼表示**：每根骨骼由旋转 $R$、中心 $\mathbf{t}$（逐帧变化）和共享尺度 $\mathbf{s}$ 组成。不同深度使用独立 MLP $f^d$，输入为父骨骼的 embedding 和根 embedding。

- **蒙皮权重**：基于 Mahalanobis 距离计算每个 3D 点与叶骨骼之间的权重，结合 MLP 预测的 delta 权重进行 LBS 变形。

- **粗到细优化**：先优化 depth-1 骨骼（粗运动），训练过程中逐步添加子骨骼捕获更细运动。初始骨骼数：动物 5 根，人类 6 根；每根骨骼后续扩展 2 根子骨骼。

### 3. 骨骼占用函数（Bone Occupancy Function）

解决骨骼位置和形状欠约束问题，将骨骼正则化到物体部件中心。

- **骨骼占用**：$g_b(\mathbf{x}) = d_M(\mathbf{x}, b) - \gamma$，点在骨骼内部为负值，外部为正值
- **骨骼掩码损失** $\mathcal{L}_{bone}$：将所有骨骼的占用聚合后渲染出 2D 掩码，与真实前景掩码对比，约束骨骼形状与物体对齐
- **重叠损失** $\mathcal{L}_{overlap}$：限制每个表面点被不超过 $\lambda$ 个骨骼覆盖，避免骨骼过度重叠
- **覆盖损失** $\mathcal{L}_{cover}$：确保每根骨骼占据一定比例的表面区域，防止骨骼退化

### 4. 总体损失函数

$$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{cycle} + \mathcal{L}_{bone} + \mathcal{L}_{overlap} + \mathcal{L}_{cover}$$

### 5. 操控方式

优化完成后提取 canonical mesh，用户通过调整父骨骼控制大范围运动，调整子骨骼实现细粒度运动，还可交互式添加/删除控制点。

## 实验关键数据

| 方法 | Eagle CD↓ | Eagle F2↑ | Swing CD↓ | Swing F2↑ | Samba CD↓ | Samba F2↑ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| ViSER | 19.22 | 24.76 | 16.29 | 19.95 | 23.28 | 22.47 |
| BANMo (复现) | 4.66 | 81.44 | 7.33 | 64.88 | 7.22 | 64.99 |
| CAMM* | 4.50 | 81.21 | 9.02 | 56.00 | 7.50 | 62.17 |
| RAC* | - | - | 6.10 | 70.33 | 6.63 | 67.71 |
| **本文** | **4.64** | **81.59** | **7.11** | **65.88** | **6.15** | **72.07** |

- 在所有数据集上超越所有无模板方法，在 Samba 上超越需要骨架的 RAC
- Eagle 数据集仅使用 10 根叶骨骼（BANMo 用 25 根以上），仍达到可比结果
- 神经渲染质量全面领先：Samba 上 PSNR 31.74 / SSIM 0.942（BANMo 为 30.72 / 0.916）

## 亮点

1. **无需先验的层次结构学习**：完全无监督地从视频中发现物体的部件层次和运动关联，无需预定义骨架或模板
2. **粗到细的运动分解**：树状骨骼结构提供多粒度运动表示，既提升重建质量又增强可解释性
3. **骨骼占用正则化**：借鉴 part-based generative model 的思想，通过掩码、重叠和覆盖三个损失项确保骨骼与物体形状对齐
4. **更少控制点更好效果**：用更少的骨骼数量达到甚至超越使用大量控制点的基线方法
5. **用户友好的操控接口**：层次化骨骼支持粗到细操控，用户可交互式增删控制点

## 局限与展望

1. 训练耗时较长（每个深度 20k 迭代，2×RTX 3090 每阶段约 3 小时），多阶段训练累计时间可观
2. 子骨骼数量（每父骨骼 2 个）为手动设定，未根据运动复杂度自适应调整
3. 仅在人体和动物上验证，对拓扑变化较大的物体（如布料、流体）适用性未知
4. 依赖预处理（PointRend 分割、VCN 光流、CSE 特征），级联误差可能影响结果
5. 3D Gaussian Splatting 等更高效的表示方式可能替代 NeRF 进一步提升效率

## 与相关工作的对比

| 维度 | BANMo | CAMM | RAC | 本文 (HSNB) |
|------|-------|------|-----|-------------|
| 先验知识 | 无 | 需要 RigNet 骨架 | 需要预定义骨架 | 无 |
| 控制点结构 | 无结构散布 | 运动链 | 固定骨架 | 层次化树结构 |
| 运动分解 | 无 | 有限 | 有限 | 粗到细多粒度 |
| 骨骼数量 | 25 | 25+ | 按类别定义 | 最少 10 根叶骨骼 |
| 可操控性 | 困难 | 较好 | 较好 | 直观且可交互增删 |
| 通用性 | 任意物体 | 需匹配骨架 | 单类别 | 任意物体 |

## 启发与关联

- **Bone Occupancy Function** 的思路可扩展到其他需要部件对齐的任务（如 part-aware 人体重建）
- 层次化控制点思想可与 3D Gaussian Splatting 结合，构建更高效的可操控重建管线
- 粗到细的训练策略对其他分层表示学习任务有借鉴意义（如场景级别的层次化表示）
- 骨骼占用函数中的重叠/覆盖损失设计可应用于其他 primitive-based 方法的正则化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层次化骨骼结构 + 骨骼占用正则化的组合在无模板动画重建中较新颖
- 实验充分度: ⭐⭐⭐⭐ — 多数据集定量/定性比较充分，含消融实验和操控演示
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 在实用性（操控接口）和学术贡献（层次化运动分解）上均有价值

<!-- RELATED:START -->

## 相关论文

- [Neural Graph Navigation for Intelligent Subgraph Matching](../../AAAI2026/robotics/neural_graph_navigation_for_intelligent_subgraph_matching.md)
- [Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](../../AAAI2026/robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)
- [UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](../../ICLR2026/robotics/urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)

<!-- RELATED:END -->
