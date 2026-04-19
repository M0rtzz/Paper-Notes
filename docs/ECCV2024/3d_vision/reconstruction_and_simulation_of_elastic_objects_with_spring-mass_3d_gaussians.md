---
title: >-
  [论文解读] Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians
description: >-
  [ECCV 2024][3D视觉][3D高斯] 提出 Spring-Gaus，将3D弹簧-质点模型集成到3D高斯表示中，从多视角视频中重建弹性物体的外观、几何和物理动力学属性，实现未来预测和不同环境条件下的仿真。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D高斯
  - 弹性物体
  - 弹簧-质点模型
  - 物理仿真
  - 系统辨识
---

# Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians

**会议**: ECCV 2024  
**arXiv**: [2403.09434](https://arxiv.org/abs/2403.09434)  
**代码**: [有](https://zlicheng.com/spring_gaus)  
**领域**: 3D视觉  
**关键词**: 3D高斯, 弹性物体, 弹簧-质点模型, 物理仿真, 系统辨识

## 一句话总结

提出 Spring-Gaus，将3D弹簧-质点模型集成到3D高斯表示中，从多视角视频中重建弹性物体的外观、几何和物理动力学属性，实现未来预测和不同环境条件下的仿真。

## 研究背景与动机

从视觉观测中重建和仿真弹性物体是计算机视觉和机器人领域的基础挑战。现有方法如3D高斯和动态扩展方法可以捕捉外观和几何的时序变化，但无法估计物体的物理属性。PAC-NeRF等使用MPM的方法假设已知材料模型，只能为整个物体分配全局物理参数，限制了对真实异质物体的适应性。此外MPM需要大量密集点导致计算成本极高。核心挑战在于开发一个既表达力强又高效、适合基于梯度优化的物理动力学模型。

## 方法详解

### 整体框架

Spring-Gaus 分为三个阶段：
1. **静态重建**：从多视角视频首帧重建物体的外观和几何（3D高斯）
2. **高斯精炼**：提取锚点用于仿真，精炼高斯核的缩放、颜色和不透明度参数
3. **动态重建**：通过可微仿真和可微渲染优化物理参数

### 关键设计

**3D弹簧-质点模型**：
- 通过体采样从高斯核中心生成锚点集合 A
- 锚点之间通过KNN连接弹簧，每个弹簧具有刚度 k 和阻尼因子 ζ
- 每个时间步计算弹簧力、阻尼力和重力，通过半隐式欧拉积分更新位置和速度
- 高斯核位置通过反距离加权（IDW）插值从锚点运动中更新

**软向量机制**：引入可学习软向量 η 调节连接弹簧数量，通过参数 κ 控制物体的刚性/柔软行为，避免超参数 n_k 的选择对仿真结果的显著影响。

**参数简化**：统一质量为常数 m₀，统一阻尼因子为 ζ₀，每个锚点一个刚度参数 k_i，将优化参数从 O(n_k·N_A) 降至 O(N_A)。

### 损失函数

采用 L₁ 范数和 D-SSIM 的加权组合：

$$\mathcal{L} = (1-\lambda_\text{d-ssim})\mathcal{L}_1 + \lambda_\text{d-ssim}\mathcal{L}_\text{d-ssim}$$

对真实数据额外加入中心损失和感知损失（VGG16），λ_d-ssim=0.8。

## 实验关键数据

### 主实验

**合成数据动态重建（Chamfer Distance ×10³mm², EMD）**：

| 方法 | Torus CD | Cross CD | Apple CD | Mean CD | Mean EMD |
|------|----------|----------|----------|---------|----------|
| **Spring-Gaus** | **0.17** | **0.48** | **0.38** | **0.85** | **0.040** |
| PAC-NeRF | 4.92 | 1.10 | 1.11 | 2.11 | 0.052 |
| Dy-Gaus | 579 | 773 | 727 | 1305 | 1.116 |
| 4D-Gaus | 11.12 | 1.77 | 2.23 | 4.43 | 0.095 |

**合成数据未来预测**：

| 方法 | Mean CD↓ | Mean EMD↓ | Mean PSNR↑ | Mean SSIM↑ |
|------|----------|-----------|------------|------------|
| **Spring-Gaus** | **5.16** | **0.095** | **17.06** | **0.897** |
| PAC-NeRF | 17.94 | 0.134 | 15.77 | 0.870 |

### 消融实验

**泛化到新条件**（定性实验）：

| 编辑类型 | 具体操作 | 效果 |
|---------|---------|------|
| 边界条件 | 改变地面位置、使用光滑/粘性地面 | 物体在不同地面上展现合理弹跳行为 |
| 物理条件 | 调节物体软硬度 | 更软的物体形变更大、回弹更慢 |
| 初始条件 | 改变初始速度 | 不同速度下的运动轨迹合理 |
| 环境条件 | 改变重力方向 | 物体按新重力方向正确运动 |

Spring-Gaus 在动态重建中CD指标比PAC-NeRF降低约60%，同时保持更好的渲染质量（PSNR提升1.29dB，SSIM提升0.027）。

### 关键发现

- Spring-Gaus 在合成和真实数据上均能准确重建弹性物体的几何、外观和物理动力学
- 解耦外观/几何重建和物理参数重建是高效优化的关键
- 可以编辑边界条件（地面位置、粘性）、物理条件（软硬度）、初始速度和重力进行新条件下的仿真

## 亮点与洞察

1. **表达力与效率的平衡**：弹簧-质点模型不假设任何材料类型，可建模异质弹性物体，同时全可微、计算高效
2. **解耦优化策略**：将外观/几何重建与物理参数重建分离，避免优化干扰
3. **创建可仿真数字资产**：从视频观测中获取的模型可在不同环境条件下进行动态仿真

## 局限性

- 仅处理弹性物体，不支持塑性变形或流体
- 真实数据需要多视角设置和手动相机标定
- 弹簧-质点模型对某些复杂材料行为的表达力有限

## 相关工作与启发

- PhysGaussian 将MPM集成到3D高斯实现前向仿真，但不支持逆向物理参数估计
- DANO 实现刚体的可微仿真和NeRF重建
- 启发：物理先验与神经渲染的结合是创建交互式数字资产的重要方向

## 评分

- 创新性：★★★★☆ 将弹簧-质点模型巧妙集成到3DGS框架，实现外观+物理联合重建
- 实用性：★★★☆☆ 需要多视角视频输入，应用场景有限
- 实验质量：★★★★☆ 合成和真实数据验证充分，消融实验完整

<!-- RELATED:START -->

## 相关论文

- [3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)
- [Human Hair Reconstruction with Strand-Aligned 3D Gaussians](human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)
- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)
- [Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)

<!-- RELATED:END -->
