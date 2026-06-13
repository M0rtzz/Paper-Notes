---
title: >-
  [论文解读] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids
description: >-
  [AAAI 2026][物理/科学计算][physics-informed dataset] 构建了 Phys-Liquid 数据集（97,200 张物理仿真图像 + 3D mesh），基于 Navier-Stokes 方程模拟透明容器内液体的动态形变，并提出四阶段重建管线（分割→多视角 mask 生成→3D 重建→缩放），在仿真和真实场景中实现高精度液体几何与体积估计。
tags:
  - "AAAI 2026"
  - "物理/科学计算"
  - "physics-informed dataset"
  - "transparent liquid"
  - "3D重建"
  - "liquid simulation"
  - "deformable objects"
---

# Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids

**会议**: AAAI 2026  
**arXiv**: [2511.11077](https://arxiv.org/abs/2511.11077)  
**代码**: [https://dualtransparency.github.io/Phys-Liquid/](https://dualtransparency.github.io/Phys-Liquid/)  
**领域**: 3D 视觉 / 科学计算  
**关键词**: physics-informed dataset, transparent liquid, 3D reconstruction, liquid simulation, deformable objects

## 一句话总结

构建了 Phys-Liquid 数据集（97,200 张物理仿真图像 + 3D mesh），基于 Navier-Stokes 方程模拟透明容器内液体的动态形变，并提出四阶段重建管线（分割→多视角 mask 生成→3D 重建→缩放），在仿真和真实场景中实现高精度液体几何与体积估计。

## 研究背景与动机

自主实验室机器人在执行液体操作（移液、吸取、混合）时，需要准确感知容器运动引起的液体动态形变。然而现有数据集存在明显不足：

- **Objaverse** 等大规模 3D 数据集主要是刚体/不透明物体，不含液体
- **ClearGrasp、ClearPose** 等透明物体数据集关注 6D 位姿而忽略内部液体
- **DTLD** 包含液体但仅有静态状态，**Narasimhan et al.** 仅单视角、无形变

核心矛盾：**缺乏包含物理真实形变的动态液体仿真数据集**，阻碍了精确液体感知算法的发展。

本文的切入点是利用 Blender + Mantaflow 基于 Navier-Stokes 方程进行物理仿真，系统地模拟容器旋转引起的液体形变，构建涵盖多场景、多光照、多视角、时间连续的大规模数据集。

## 方法详解

### 整体框架

本工作包含两部分：(1) **数据集构建**——基于物理仿真生成 97,200 张图像和 8,100 个液体 3D mesh；(2) **四阶段验证管线**——从单张图像重建液体 3D 几何并估计真实尺寸体积，用于验证数据集的有效性。

### 关键设计

**1. 基于 Navier-Stokes 方程的物理仿真**

液体动力学由动量方程控制：

$$\frac{D\mathbf{u}}{Dt} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}$$

其中 $\mathbf{u}$ 为速度场，$\rho$ 为密度，$p$ 为压力场，$\nu$ 为运动粘度，$\mathbf{g}$ 为外力。同时施加不可压缩条件 $\nabla \cdot \mathbf{u} = 0$，通过 Mantaflow 求解器在 Blender 中实现。

数据生成覆盖：20 种常见实验室容器、5 种实验室场景、8 种光照条件、5 种液体颜色、6 种旋转模式（沿 X/Y/Z 轴组合，0°~80°）、81 个时间帧、6 个正交相机视角（上下前后左右）。每个容器分配 5 种条件组合，共 100 种配置。

**2. 四阶段液体重建管线**

整体流程形式化为 $S = F(I) = T(R(G(S(I))), s)$：

- **阶段 1：液体分割**——使用 YOLO-World 检测液体区域（"liquid"和"colored liquid"为正类，排除"bottle"），生成 bounding box 引导 SAM2 进行精确分割
- **阶段 2：多视角 mask 生成**——使用 CRM 扩散模型（在 Phys-Liquid 上微调）从单视角 mask 生成 6 个正交视角的液体 mask 和 canonical coordinate maps
- **阶段 3：3D mesh 重建**——基于 triplane 表示，用卷积 U-Net 编码多视角 mask 为三平面特征，MLP 解码为纹理化 3D mesh
- **阶段 4：真实尺寸缩放**——多视角 ViT 架构回归缩放因子 $s$，用 L2 损失监督

缩放因子计算：

$$s = \sqrt[3]{\frac{S_{\text{PI},x}}{V_x} \cdot \frac{S_{\text{PI},y}}{V_y} \cdot \frac{S_{\text{PI},z}}{V_z}}$$

**3. 扩散模型微调策略**

利用 Phys-Liquid 每个时间步的 6 个正交视角作为微调数据，在 2 张 RTX 6000 Ada 48GB GPU 上训练 16 小时、10k 迭代。微调后平均 IoU 从 74.38% 提升到 90.05%。

### 损失函数 / 训练策略

- 扩散模型微调：标准去噪损失
- 缩放模型：L2 回归损失，在 1 张 RTX 6000 Ada GPU 上训练 12 小时、500 迭代
- 数据集划分：按完整时间序列 9:1 划分训练/测试集，确保无帧级泄漏

## 实验关键数据

### 主实验

**与液体特定基线对比**（Phys-Liquid 测试集）：

| 方法 | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|------|------|-----------------|------------|-------------|
| Eppel et al. | 0.0842 | 0.0412 | 0.1216 | 30.91 |
| **本文方法（微调后）** | **0.0192** | **0.0079** | **0.4748** | **75.38** |

RMSE 降低 77.2%，F-Score 提升 44.47 个百分点。

**与通用重建基线对比**（50 张测试图像）：

| 方法 | Chamfer Distance | Volume IoU | F-Score (%) |
|------|-----------------|------------|-------------|
| InstantMesh | 0.0189 | 0.2794 | 46.18 |
| TripoSR | 0.0275 | 0.2275 | 38.06 |
| 本文（无微调） | 0.0128 | 0.3246 | 58.19 |
| **本文（有微调）** | **0.0085** | **0.6236** | **78.57** |

微调后 Volume IoU 从 0.3246 提升到 0.6236，F-Score 从 58.19% 提升到 78.57%。

### 消融实验

**管线模块消融**（逐步替换模块输出为仿真 ground truth）：

| 替换模块 | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|---------|------|-----------------|------------|-------------|
| 完整管线 | 0.0192 | 0.0079 | 0.4748 | 75.38 |
| 替换分割 | 0.0130 | 0.0075 | 0.5504 | 78.42 |
| 替换多视角生成 | 0.0105 | 0.0067 | 0.6532 | 81.36 |
| 替换 mesh 重建 | 0.0085 | 0.0058 | 0.7687 | 85.64 |
| 替换缩放 | 0.0071 | 0.0042 | 0.7511 | 88.47 |

mesh 重建和缩放模块对最终性能影响最大。

**扩散模型微调影响**：

| 设置 | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|------|------|-----------------|------------|-------------|
| 无微调 | 0.0254 | 0.0139 | 0.2850 | 46.19 |
| 微调后（测试集） | 0.0192 | 0.0079 | 0.4748 | 75.38 |

微调使 RMSE 降低 24.4%，Volume IoU 提升 66.6%。

**真实场景泛化**（DTLD 数据集）：

| 数据集 | RMSE | Chamfer Distance | Volume IoU | F-Score (%) |
|--------|------|-----------------|------------|-------------|
| DTLD（真实世界） | 0.0266 | 0.0172 | 0.3861 | 62.43 |
| Phys-Liquid（仿真） | 0.0192 | 0.0079 | 0.4748 | 75.38 |

仅在仿真数据上训练即可在真实数据上取得合理精度。

### 关键发现

- 6 个视角的多视角 mask 生成一致性好：各视角平均 IoU 在 89.21%~91.76% 之间
- 时间一致性高：100 个时间序列上 RMSE 方差仅 0.00038，标准差 0.00644
- 仿真与真实液体形变角度高度吻合，验证了物理仿真的真实性

## 亮点与洞察

- **填补数据空白**：首个包含动态液体形变的物理仿真数据集，从 3D 扩展到 4D（时空域）
- **端到端验证**：数据集 + 重建管线一体化，不仅提供数据还提供可用的基线方法
- **可扩展性**：基于 Blender 可渲染法线、折射流等额外模态，具有工具属性

## 局限与展望

- 数据规模适中（97,200 张），相比大规模视觉数据集仍较小
- 仅模拟旋转引起的形变，未覆盖倾倒、混合等更复杂操作
- 容器均为标准实验室器皿，未涉及不规则容器
- 缩放模型依赖 CAD 模型提供的真实尺寸信息

## 相关工作与启发

- **vs DTLD**: DTLD 有 27,458 张多视角图像但仅静态液体；Phys-Liquid 有 97,200 张且包含动态形变和时间变化
- **vs TransProteus**: TransProteus 用 Mantaflow 模拟液体但不模拟容器旋转引起的形变
- **vs InstantMesh/TripoSR**: 通用 3D 重建方法无法捕捉液体的精细形变特征，本文微调后大幅超越

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个包含动态液体形变的物理仿真数据集，填补领域空白
- 实验充分度: ⭐⭐⭐⭐ 多维度评估（重建精度、泛化、时间一致性、模块消融），设计合理
- 写作质量: ⭐⭐⭐⭐ 图表丰富，管线说明清晰
- 价值: ⭐⭐⭐ 聚焦实验室液体操作场景，应用范围较窄但在目标领域有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](../../ICML2026/physics/eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[AAAI 2026\] Fast 3D Surrogate Modeling for Data Center Thermal Management](fast_3d_surrogate_modeling_for_data_center_thermal_management.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/physics/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)

</div>

<!-- RELATED:END -->
