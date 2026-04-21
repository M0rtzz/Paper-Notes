---
title: >-
  [论文解读] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出TRACE框架，将每个3D高斯核视为刚性粒子并为其学习独立的平移-旋转动力学系统（包含速度、加速度、角速度、角加速度等完整物理参数），无需任何人工标注即可从多视角动态视频中学习3D场景的物理运动规律并准确外推未来帧。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 物理动力学学习
  - 未来帧外推
  - 平移旋转动力学系统
  - 动态场景重建
---

# TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos

**会议**: ICCV 2025  
**arXiv**: [2508.09811](https://arxiv.org/abs/2508.09811)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 物理动力学学习, 未来帧外推, 平移旋转动力学系统, 动态场景重建

## 一句话总结

提出TRACE框架，将每个3D高斯核视为刚性粒子并为其学习独立的平移-旋转动力学系统（包含速度、加速度、角速度、角加速度等完整物理参数），无需任何人工标注即可从多视角动态视频中学习3D场景的物理运动规律并准确外推未来帧。

## 研究背景与动机

动态3D场景的几何、外观和物理属性建模对机器人、混合现实、具身AI等应用至关重要。现有方法存在两大路线的局限：

**物理信息神经网络(PINN)**：将PDE作为软约束加入损失函数，但训练效率低、边界区域精度差，且通常需要前景mask等额外标注

**物理模型编码方法**：将弹簧、流体等特定物理系统嵌入网络，但只适用于特定材料/物体类型，泛化性差

两者的共同问题是——它们要么无法真正学到复杂运动的物理规律，要么需要额外的物体类型/mask标注。另一方面，现有动态3DGS方法（如DefGS、4DGS）虽然能很好地做视角插值，但它们学习的形变场本质上不编码物理先验，只是拟合像素相关性，因此**完全无法外推未来帧**。

核心洞察：多物体/多部件场景中，相邻点的运动可能截然不同（如两个相向运动的物体交界处），因此每个3D点应该有独立的动力学参数。

## 方法详解

### 整体框架

TRACE由两个核心模块和一个辅助模块组成：
1. **3D场景表示模块**：标准3DGS在规范时间戳 $t=0$ 学习静态几何和外观
2. **平移-旋转动力学系统模块**（核心创新）：为每个刚性粒子学习完整的物理参数集
3. **辅助形变场**：借助DefGS/4DGS的形变网络辅助训练稳定性

### 关键设计：平移-旋转动力学系统

根据经典力学定律，任意刚体粒子 $\mathbf{P}$ 在3D空间中的运动可以分解为绕旋转中心的旋转 + 旋转中心自身的平移。为每个粒子学习两组物理参数：

**Group 1 - 旋转中心参数**：
- 中心位置 $\mathbf{P}_c \in \mathbb{R}^3$
- 中心速度 $\mathbf{v}_c \in \mathbb{R}^3$
- 中心加速度 $\mathbf{a}_c \in \mathbb{R}^3$

**Group 2 - 粒子旋转参数**：
- 旋转向量 $\mathbf{w}_p \in \mathbb{R}^3$（相对旋转中心）
- 角加速度 $\boldsymbol{\epsilon}_p \in \mathbb{R}^3$

粒子的组合速度推导为：
$$\mathbf{v}_p^t = \mathbf{w}_p^t \times (\mathbf{P} - \mathbf{P}_c^t) + \mathbf{v}_c^t$$

由于中心速度和位置耦合，实际学习等价参数 $\bar{\mathbf{v}}_c^t = \mathbf{v}_c^t - \mathbf{w}_p^t \times \mathbf{P}_c^t$ 和 $\bar{\mathbf{a}}_c^t$。整个模块用简单的MLP实现：
$$\{(\bar{\mathbf{v}}_c^t, \bar{\mathbf{a}}_c^t), (\mathbf{w}_p^t, \boldsymbol{\epsilon}_p^t)\} = f_{trd}(\mathbf{P}, t)$$

**关键优势**：只需在某一时刻 $t$ 学习动力学系统参数，粒子未来运动即由力学定律导出，无需额外物理先验。

### 关键设计：Runge-Kutta 2阶数值外推

使用RK2方法从 $t'$ 推导到 $t = t' + \Delta t$：
1. 计算中间时刻的等价速度和角速度
2. 更新位置：$\mathbf{x}_t = \mathbf{x}_{t'} + \Delta t (\bar{\mathbf{v}}_c^{mid} + \mathbf{w}_p^{mid} \times \mathbf{x}_{t'})$
3. 利用Rodrigues公式计算旋转矩阵增量 $\Delta \mathbf{R}$
4. 尺度保持不变（刚性假设）

选择2阶而非更高阶的理由：(1) 短时预测（毫秒级）2阶已足够精确；(2) 牛顿第一/二定律都可被2阶关系捕捉；(3) 大量粒子各自2阶的复合效果已能表达复杂变形。

### 辅助形变场的作用

直接端到端训练动力学系统模块存在困难——早期3DGS核的位置不稳定会导致优化困难。因此并行训练一个辅助形变场（如DefGS），为动力学系统提供稳定的粒子位置输入。

### 损失函数

标准的3DGS重建损失：$\ell_1 + \ell_{ssim}$，联合优化规范高斯 $G_0$、形变场 $f_{defo}$ 和动力学系统 $f_{trd}$。

## 实验关键数据

### 主实验：未来帧外推（Table 1）

**Dynamic Object 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|
| D-NeRF | 14.660 | 0.737 | 0.312 |
| NVFi | 27.594 | 0.972 | 0.036 |
| DefGS | 19.849 | 0.949 | 0.045 |
| DefGS_nvfi | 28.749 | 0.984 | 0.013 |
| **TRACE (Ours)** | **31.597** | **0.987** | **0.009** |

**Dynamic Indoor Scene 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|
| NVFi | 29.745 | 0.876 | 0.204 |
| DefGS_nvfi | 31.096 | 0.945 | 0.077 |
| **TRACE (Ours)** | **34.824** | **0.965** | **0.054** |

**Dynamic Multipart 数据集**（新提出，最具挑战）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|
| NVFi | 25.235 | 0.955 | 0.046 |
| DefGS_nvfi | 28.455 | 0.979 | 0.017 |
| **TRACE (Ours)** | **33.481** | **0.990** | **0.007** |

### 关键发现

- 在4个数据集中的3个上大幅超越所有基线（PSNR提升2.8~5.0 dB）
- 在NVIDIA Dynamic Scene数据集（真实世界）上也优于NVFi（29.341 vs 28.462）
- DefGS_nvfi（将NVFi速度场嫁接到3DGS上）是最强基线，TRACE在此基础上仍有显著提升，证明平移-旋转动力学系统的优越性
- 纯形变方法（DefGS、4DGS）在外推任务上远不如有物理约束的方法
- 框架灵活性：替换辅助形变场为4DGS（TRACE_4dgs）也能获得优秀结果

### 附加能力：无监督物体/部件分割

通过聚类学到的物理参数可以自然地分割不同运动模式的物体/部件——无需额外标注。

## 亮点与洞察

1. **物理参数显式学习 vs PINN软约束**：TRACE直接学习速度、加速度等物理量，而非靠PDE损失间接正则化，效率更高、效果更好
2. **刚性粒子假设的优雅性**：将每个高斯核视为有大小和朝向的刚性粒子，实现了3DGS天然粒子表示与经典力学的完美对接
3. **2阶动力学的"刚好够用"哲学**：每个粒子仅有2阶（加速度），但大量粒子各自独立的2阶动力学组合后可表达极其复杂的场景变形
4. **辅助形变场的巧妙设计**：不是替代而是辅助——在训练时提供稳定输入，推理时完全依赖物理参数外推

## 局限性

- 仅建模2阶动力学，对于爆炸、碎裂等突变运动可能不足
- 假设每个高斯核为刚性粒子（尺度不变），对强烈形变的软体可能需要放松假设
- 外推时间越长误差越大，需要滑动窗口机制持续校正
- 在NVIDIA真实数据集上相比合成数据集优势减小，泛化到更复杂真实场景仍需验证

## 相关工作

- **NVFi**：最接近的工作，用PINN损失学习速度场+NeRF骨干；TRACE用显式物理参数+3DGS骨干全面超越
- **DefGS / 4DGS**：3DGS动态场景方法，擅长视角插值但无法外推；TRACE以其为辅助模块
- **D-NeRF / HexPlane / TiNeuVox**：NeRF动态场景方法，同样只能插值
- **PINN系列**：物理信息神经网络（如ScalarFlow），用PDE作为软约束训练效率低
- **FreeGave**：concurrent work，隐式拟合速度网络 vs TRACE显式学习运动变化量（加速度/jerk）

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 平移-旋转动力学系统的提出是全新范式
- **技术深度**: ⭐⭐⭐⭐⭐ — 物理建模扎实，RK2数值推导严谨
- **实验充分性**: ⭐⭐⭐⭐ — 四数据集（含新数据集），但缺少更多真实场景验证
- **实用价值**: ⭐⭐⭐⭐ — 对机器人操作中的短时运动预测有直接应用
- **总体推荐**: ⭐⭐⭐⭐⭐ — 动态场景物理建模的里程碑工作

<!-- RELATED:START -->

## 相关论文

- [MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](../../NeurIPS2025/3d_vision/mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](../../CVPR2025/3d_vision/hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](../../CVPR2025/3d_vision/freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)

<!-- RELATED:END -->
