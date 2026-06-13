---
title: >-
  [论文解读] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders
description: >-
  [NeurIPS 2025][3D视觉][可微物理模拟] 提出 AS-DiffMPM，一种支持任意形状刚体碰撞体的可微物质点法（MPM）框架，结合多种新视角合成方法实现从视觉观测中估计物体物理参数的系统辨识。 从视频观测中估计物体的物理参数（如粘度、弹性模量）在机器人、VR 和图形学中有重要应用。现有方法（如 PAC-NeR…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "可微物理模拟"
  - "系统辨识"
  - "Material Point Method"
  - "3D Gaussian Splatting"
  - "碰撞处理"
---

# Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders

**会议**: NeurIPS 2025  
**arXiv**: [2511.06846](https://arxiv.org/abs/2511.06846)  
**代码**: [有](https://as-diffmpm.github.io)  
**领域**: 3D视觉  
**关键词**: 可微物理模拟, 系统辨识, Material Point Method, 3D Gaussian Splatting, 碰撞处理

## 一句话总结

提出 AS-DiffMPM，一种支持任意形状刚体碰撞体的可微物质点法（MPM）框架，结合多种新视角合成方法实现从视觉观测中估计物体物理参数的系统辨识。

## 研究背景与动机

从视频观测中估计物体的物理参数（如粘度、弹性模量）在机器人、VR 和图形学中有重要应用。现有方法（如 PAC-NeRF）将可微渲染与可微 MPM 结合，通过梯度优化从图像反推物理参数。然而，这些方法存在关键局限：

**碰撞体限制**：现有系统辨识方法仅支持平面碰撞体（如地面），无法处理物体与非平面表面的复杂碰撞

**MPM 模拟器不可微**：能处理任意形状碰撞体的 MPM 模拟器（如 CPIC）是不可微的，无法用于梯度优化
3. 真实场景中物体常与各种形状的刚体碰撞（如掉在盒子边缘、撞到复杂物体），仅支持平面交互严重限制了适用范围

## 方法详解

### 整体框架

AS-DiffMPM 的流程分三步：

1. **刚体碰撞体重建**：从多视角图像用 2DGS 重建刚体，或直接使用 mesh
2. **物理模拟**：用 AS-DiffMPM（可微 MPM + 任意形状碰撞处理）模拟连续体材料的运动
3. **渲染与优化**：将模拟后的粒子位置映射回渲染原语，通过与真实视频的光度损失反传梯度优化物理参数

关键梯度传播链：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_p \left(\frac{\partial \mathcal{L}}{\partial I} \cdot \frac{\partial I}{\partial \mathbf{x}_p}\right) \cdot \frac{\partial \mathbf{x}_p}{\partial \theta}$$

其中 $\theta$ 为物理参数，$\mathbf{x}_p$ 为粒子位置，$I$ 为渲染图像。

### 关键设计

#### 1. MPM 基础与碰撞处理

MPM 三阶段循环：
- **P2G（粒子到网格）**：粒子的质量和动量通过权重函数传递到邻近网格节点
- **G-OP（网格操作）**：在网格上施加外力（如重力）并处理边界条件
- **G2P（网格到粒子）**：网格速度场插值回粒子，更新粒子状态

传统方法在 G-OP 阶段处理碰撞（对碰撞体内部网格节点速度置零），但对复杂几何体失效——同一网格单元内的所有粒子共享单一速度场。

#### 2. CPIC 方法的可微化（核心贡献）

AS-DiffMPM 基于 Compatible Particle-in-Cell (CPIC) 实现**粒子级碰撞处理**：

**Step 1 — 刚体投影到碰撞网格**：
- 在刚体表面采样**刚性粒子** $\mathbf{x}_{rp}$
- 识别**亲和网格节点**：将刚性粒子的 $3\times3\times3$ 邻域节点投影到所属面上，判断是否在面内
- 重建碰撞网格：每个节点存储亲和标记 $A_g$、无符号距离 $d_g$、侧向标记 $T_g$、法线 $\mathbf{n}_g$

**Step 2 — 碰撞网格到材料粒子**：
- 通过插值将碰撞属性从网格传递到粒子：$d_p = \sum_{g} w_g A_g T_g d_g$，$\mathbf{n}_p = \sum_g w_g A_g \mathbf{n}_g$
- **穿透修正**：若粒子穿透边界，施加惩罚力 $\mathbf{f}_p = -k_h d_p \mathbf{n}_p$
- **相容性判断**：比较粒子 $T_p$ 与邻域节点 $T_g$，位于边界两侧则不相容

**Step 3 — P2G/G2P 中的碰撞处理**：
- P2G：粒子仅向**相容节点**传递速度（G-OP 阶段无需碰撞处理）
- G2P：对不相容节点，直接复用粒子当前速度并投影到表面：$\mathbf{v}_p^{proj} = \mathbf{v}_p - (\mathbf{v}_p \cdot \mathbf{n}_p)\mathbf{n}_p$

#### 3. 2D 高斯作为碰撞体

创新性地支持**用 2D Gaussian Splatting 重建的刚体**作为碰撞体。每个 2D 高斯对应一个平面圆盘（planar disk），替代 mesh 面片作为碰撞原语 $\xi^i$。其余流程与 mesh 碰撞体完全一致，仅在亲和节点识别时需要判断投影是否落在圆盘内。

#### 4. 渲染-物理集成

支持三种渲染方法：
- **Voxel-based NeRF (DVGO)**：需要额外的体素-粒子映射
- **2D Gaussian Splatting (2DGS)**：天然与 MPM 对齐（都是粒子表示）
- **MDyn-3DGS**：先做动态场景重建再优化物理参数

### 损失函数 / 训练策略

- **粒子轨迹监督**：粒子级 MSE 损失
- **视觉观测监督**：渲染图像与真实帧的光度损失
- Adam 优化器，物理参数从初始猜测迭代优化
- 数据集：每个碰撞体/材料组合 10 个序列（16 时间步），11 相机视角

## 实验关键数据

### 主实验（粒子轨迹监督）

从粒子轨迹估计物理参数（绝对误差×100）：

| 材料类型 | 碰撞体 | 参数 | RP-DiffMPM | GOP-DiffMPM | **AS-DiffMPM** |
|---------|--------|------|:----------:|:-----------:|:-------------:|
| 牛顿流体 | Bunny | log₁₀(μ) | 6.69±5.97 | 4.35±5.24 | **0.43±0.30** |
| 牛顿流体 | Armadillo | log₁₀(κ) | 34.61±46.5 | 0.22±0.24 | **0.22±0.24** |
| 非牛顿流体 | Box | log₁₀(τ_Y) | 20.13±25.8 | 7.28±3.75 | **5.86±3.21** |
| 颗粒 | Bunny | θ_fric | 1.91±0.42 | 0.06±0.07 | **0.11±0.15** |

### 视觉观测监督

不同渲染方法 + AS-DiffMPM 的系统辨识结果：

| 材料 | 碰撞体 | 参数 | DVGO | 2DGS | **MDyn-3DGS** |
|------|--------|------|:----:|:----:|:------------:|
| 牛顿流体 | Armadillo | log₁₀(μ) | 1.99±1.59 | 5.28±2.55 | **0.97±0.86** |
| 牛顿流体 | Armadillo | log₁₀(κ) | 32.94±42.2 | 49.66±35.8 | **14.99±17.1** |
| 颗粒 | Bunny | θ_fric | 4.02±0.68 | **0.50±0.29** | 3.33±0.62 |
| 颗粒 | Armadillo | θ_fric | 4.11±1.05 | **0.36±0.13** | 2.89±0.78 |

### 消融实验

三种碰撞处理策略对比：
- **RP-DiffMPM**（刚性粒子）：一致表现最差，缺乏定制碰撞策略
- **GOP-DiffMPM**（网格级 SDF）：对复杂几何体精度受限（同一格子共享速度场）
- **AS-DiffMPM**（粒子级 CPIC）：在多数场景下最优或持平

### 关键发现

1. 粒子级碰撞处理（CPIC）在系统辨识中显著优于网格级方法
2. MDyn-3DGS 通常表现最佳（利用动态重建作为额外约束），但 2DGS 在颗粒材料上更优
3. 非牛顿流体最具挑战，无单一渲染方法能在所有碰撞体上一致领先
4. 复杂碰撞体的系统辨识本质上比平面碰撞更困难，但对迈向更少假设的模型至关重要

## 亮点与洞察

- **填补重要空白**：首次实现任意形状碰撞体下的可微 MPM 系统辨识
- **通用碰撞接口**：支持 mesh 和 2D 高斯两种碰撞体表示，形成即插即用模块
- **渲染方法无关**：可与 NeRF、3DGS、2DGS 等多种方法集成
- **真实世界验证**：在面团碰撞盒子边缘的实验中成功估计非牛顿材料参数

## 局限与展望

1. 长时间模拟后高斯偏离原始位置，造成视觉伪影——可引入时变高斯属性
2. 碰撞体目前假设为静态刚体，未支持可变形碰撞体或动态碰撞体
3. 非牛顿流体的估计精度仍有较大方差，需要更强的正则化
4. 75 个合成序列的 benchmark 规模较小，需更大规模真实数据验证

## 相关工作与启发

- **PAC-NeRF**：开创性地将 NeRF + DiffMPM 用于系统辨识，但仅支持平面碰撞
- **PhysGaussian**：用 MPM 驱动 3DGS 的物理动画，但不做系统辨识
- **CPIC (hu2018moving)**：原始的非可微任意碰撞体 MPM 方法，本文的可微化基础
- 启发：可微物理引擎 + 可微渲染的端到端管线是理解物理世界的关键路径

## 评分

- 新颖性: ⭐⭐⭐⭐ 任意碰撞体可微 MPM 是切实的技术突破
- 实验充分度: ⭐⭐⭐⭐ 三种材料×三种碰撞体×三种渲染方法，覆盖广
- 写作质量: ⭐⭐⭐ 技术细节充分但公式较密，需要一定背景知识
- 价值: ⭐⭐⭐⭐ 显著拓展了可微物理模拟的适用范围

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](../../ICML2026/3d_vision/stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)
- [\[CVPR 2025\] MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](../../CVPR2025/3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)
- [\[NeurIPS 2025\] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation](every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)

</div>

<!-- RELATED:END -->
