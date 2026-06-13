---
title: >-
  [论文解读] Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians
description: >-
  [ECCV 2024][3D视觉][3D高斯] 提出 Spring-Gaus，将可学习的 3D 弹簧-质点模型集成到 3D Gaussian Splatting 中，从多视角视频重建弹性物体的外观、几何和物理动力学参数，支持未来预测和不同条件下的仿真。 领域现状：3D Gaussian Splatting 及其动态扩展可以重…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D高斯"
  - "弹性物体"
  - "弹簧质点模型"
  - "系统辨识"
  - "物理仿真"
---

# Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians

**会议**: ECCV 2024  
**arXiv**: [2403.09434](https://arxiv.org/abs/2403.09434)  
**代码**: [https://zlicheng.com/spring_gaus](https://zlicheng.com/spring_gaus)  
**领域**: 3D视觉 / 物理仿真  
**关键词**: 3D高斯, 弹性物体, 弹簧质点模型, 系统辨识, 物理仿真

## 一句话总结
提出 Spring-Gaus，将可学习的 3D 弹簧-质点模型集成到 3D Gaussian Splatting 中，从多视角视频重建弹性物体的外观、几何和物理动力学参数，支持未来预测和不同条件下的仿真。

## 研究背景与动机

**领域现状**：3D Gaussian Splatting 及其动态扩展可以重建物体外观和几何的时序变化，但不捕获物理属性。PAC-NeRF 尝试将 MPM 物理先验集成到 NeRF 中，但假设已知材料模型且只有全局物理参数。

**现有痛点**：(1) 动态场景重建方法（D-NeRF、4D-GS）只拟合运动轨迹，不理解物理规律，无法预测未来或模拟新条件；(2) PAC-NeRF 假设已知材料类型（如弹性体），不适用于真实异构物体；(3) 为 MPM 的每个粒子学习物理参数计算代价极高，且 NeRF 隐式网格分辨率受限。

**核心矛盾**：需要一个既有表达力（能建模异构弹性物体的复杂变形）又计算高效（支持基于梯度的逆优化）的物理动力学模型。

**本文目标**：从多视角视频重建物体的外观、形状和物理动力学参数，并支持在不同初始条件和环境下的仿真预测。

**切入角度**：弹簧-质点系统是经典的物理模型，不假设特定材料，通过可学习的拓扑和参数（刚度、阻尼）建模各种弹性行为，且天然可微分。

**核心 idea**：用 3D Gaussian 表示外观和几何，用稀疏的锚点+弹簧系统表示物理动力学，将两者解耦重建后组合成可仿真的 3D 物体。

## 方法详解

### 整体框架
重建分三步：(1) 从首帧多视角图像重建静态 3D Gaussians；(2) 体积采样生成稀疏锚点并精细化 Gaussians；(3) 在锚点间建立弹簧连接，通过可微仿真+渲染损失优化物理参数（刚度、阻尼、质量、初始速度）。

### 关键设计

1. **3D 弹簧-质点模型**:

    - 功能：表示弹性物体的物理动力学，不假设材料类型
    - 核心思路：从 Gaussian 中心点体积采样 $N_A$ 个锚点，每个锚点有质量 $m_i$ 和速度 $v_i$，通过 KNN 连接到 $n_k$ 个最近邻居形成弹簧网络。弹簧力 $F_k = -\eta \cdot k_{i,j}(\|x_i - x_{i,j}\| - l_{i,j}) \cdot |\Delta l|^{p_k}$（$p_k > 0$ 时为非线性弹簧），加上阻尼力和重力，用半隐式 Euler 积分更新位置
    - 设计动机：相比 MPM 需要密集粒子和已知材料本构，弹簧-质点模型用稀疏锚点和逐弹簧可学习参数就能建模异构物体的复杂变形，且完全可微分

2. **软连接向量 (Soft Vector)**:

    - 功能：自动学习每个锚点的有效弹簧连接数
    - 核心思路：引入衰减向量 $\eta = [\eta_0, \ldots, \eta_{n_k}]$，近邻弹簧权重为 1，远邻弹簧通过可学习参数 $\kappa$ 控制衰减 $\eta_j = \text{clamp}(2 - \exp(\text{softplus}(\kappa))^{j-n_c}, 0, 1)$
    - 设计动机：$n_k$ 太大物体过刚、太小过软，软连接向量让模型自动学习合适的刚度，避免手动调参

3. **解耦重建流程**:

    - 功能：将外观/几何重建与物理参数重建分离，降低优化难度
    - 核心思路：先冻结物理参数重建静态 3D Gaussians → 提取锚点并精细化 Gaussians → 冻结外观参数，用可微仿真+渲染损失优化物理参数
    - 设计动机：联合优化外观+物理参数空间太大且高度非凸，解耦后各阶段优化目标更明确

### 损失函数 / 训练策略
静态重建：光度损失 + SSIM + 不透明度正则。动态重建：多视角多帧光度损失反传通过可微仿真到物理参数。

## 实验关键数据

### 主实验

| 场景 | Spring-Gaus PSNR | PAC-NeRF PSNR | 备注 |
|------|-----------------|---------------|------|
| 合成弹性球 | 更优 | 较低 | PAC-NeRF 假设已知材料 |
| 合成异构物体 | 显著更优 | 不适用 | PAC-NeRF 无法处理异构 |
| 真实橡胶鸭 | 有效 | 有限 | 真实数据验证 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 完整 Spring-Gaus | 最优 | 弹簧-质点+解耦重建 |
| 无软连接向量 | 下降 | 固定 $n_k$ 不灵活 |
| 非线性弹簧 ($p_k>0$) | 提升 | 非线性力更贴合真实材料 |
| 联合优化（不解耦） | 收敛困难 | 解耦策略是关键 |

### 关键发现
- 弹簧-质点模型虽然简单但在真实弹性物体上表现出色，可以捕捉碰撞、变形和反弹
- 软连接向量有效地自动调节了"刚度-柔度"的平衡
- 在合成和真实数据上均能进行未见条件（不同初始高度、不同重力方向）的前瞻仿真

## 亮点与洞察
- **模型选择的智慧**：不用复杂的 MPM/FEM，而用经典的弹簧-质点模型——足够灵活（逐弹簧可学习参数可处理异构材料）又足够简单（完全可微、高效）
- 将外观重建和物理重建解耦是关键工程洞察——避免了高维联合优化的困难
- 从视频中逆向学习物理参数（系统辨识）的范式可迁移到机器人的物体操作中

## 局限与展望
- 弹簧-质点模型无法精确建模流体、布料等需要连续介质力学的材料
- 依赖多视角同步视频，单视角场景需要额外的深度估计
- 碰撞处理使用简单的边界条件，复杂接触场景（多物体交互）处理有限
- 锚点数量和弹簧连接数的选择仍依赖经验

## 相关工作与启发
- **vs PAC-NeRF**: PAC-NeRF 用 MPM 但假设全局材料参数；Spring-Gaus 用弹簧-质点允许逐弹簧异构参数
- **vs PhysGaussian**: PhysGaussian 用 MPM 做正向仿真但不做逆向参数估计；Spring-Gaus 完成了视频→物理参数的完整逆问题
- **vs DANO**: DANO 处理刚体；Spring-Gaus 处理弹性体，是更具挑战性的设置

## 评分
- 新颖性: ⭐⭐⭐⭐ 弹簧-质点+3DGS 的组合简洁有效，软连接向量设计巧妙
- 实验充分度: ⭐⭐⭐ 合成+真实数据，但真实场景种类有限
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，流程图直观
- 价值: ⭐⭐⭐⭐ 首次从视频重建可仿真的弹性 3DGS 物体

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] 3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)
- [\[ECCV 2024\] Human Hair Reconstruction with Strand-Aligned 3D Gaussians](human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](../../CVPR2025/3d_vision/riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)

</div>

<!-- RELATED:END -->
