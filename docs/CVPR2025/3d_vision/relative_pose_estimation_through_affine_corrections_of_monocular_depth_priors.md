---
title: >-
  [论文解读] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors
description: >-
  [CVPR 2025][3D视觉][相对位姿估计] 本文提出三个新的相对位姿求解器，通过显式建模单目深度预测的仿射（尺度+偏移）模糊性来利用深度先验，并设计混合估计框架将深度感知求解器与经典点求解器结合，在标定和非标定设置下均大幅提升位姿估计精度。
tags:
  - CVPR 2025
  - 3D视觉
  - 相对位姿估计
  - 单目深度先验
  - 仿射校正
  - RANSAC
  - 几何求解器
---

# Relative Pose Estimation through Affine Corrections of Monocular Depth Priors

**会议**: CVPR 2025  
**arXiv**: [2501.05446](https://arxiv.org/abs/2501.05446)  
**代码**: https://github.com/MarkYu98/madpose  
**领域**: 3D视觉  
**关键词**: 相对位姿估计, 单目深度先验, 仿射校正, RANSAC, 几何求解器

## 一句话总结
本文提出三个新的相对位姿求解器，通过显式建模单目深度预测的仿射（尺度+偏移）模糊性来利用深度先验，并设计混合估计框架将深度感知求解器与经典点求解器结合，在标定和非标定设置下均大幅提升位姿估计精度。

## 研究背景与动机

**领域现状**：单目深度估计（MDE）模型近年取得巨大进步，从 Marigold 到 Depth Anything v2 再到 MoGe，已能在开放域图像上提供合理的深度预测。然而，如何有效利用这些深度先验来提升经典几何视觉任务（特别是相对位姿估计）仍然欠探索。

**现有痛点**：现有方法假设两幅图像的深度图只差一个全局尺度因子（scale-only），但 SOTA 深度模型通常是仿射不变的（输出与真实深度差一个尺度和偏移）。即使是号称预测"度量深度"的模型，其预测也并非完美一致——仍然存在视图间的偏移差异。忽略偏移建模会导致错误的深度对齐和位姿估计。

**核心矛盾**：深度先验提供了丰富的跨视图约束，但固有的噪声和仿射歧义使其难以简单地改进经典的基于关键点的方法。需要一种既能利用深度先验又不被其误差误导的稳健方案。

**本文目标**：开发能显式处理独立仿射（尺度+偏移）歧义的相对位姿求解器，覆盖标定和非标定场景，并设计混合估计框架兼容深度感知和经典方法。

**切入角度**：将深度图的仿射校正参数（$\alpha, \beta_1, \beta_2$）作为未知量与相对位姿一起联合求解，利用刚体变换的距离不变性消除旋转和平移，先解仿射参数和焦距，再恢复 $R, t$。

**核心 idea**：为相对位姿估计设计三个新求解器，分别适用于标定（3点）、共享焦距（4点）和独立焦距（4点）场景，在 RANSAC 框架中与经典求解器混合使用，同时利用深度和极线约束。

## 方法详解

### 整体框架
输入为两幅图像，分别运行特征匹配（如 SuperPoint+LightGlue 或 RoMa）和单目深度估计得到对应点和深度图 $D_1, D_2$。深度图经仿射变换 $\hat{D}_1 = D_1 + \beta_1$, $\hat{D}_2 = \alpha(D_2 + \beta_2)$ 后用于约束位姿。在混合 LO-MSAC 框架中，交替使用深度感知求解器和经典点求解器，利用深度重投影误差和 Sampson 误差联合评分，通过局部优化联合优化所有参数。

### 关键设计

1. **仿射校正求解器（Affine-Corrected Solvers）**:

    - 功能：在求解相对位姿的同时联合估计两幅深度图的尺度比 $\alpha$ 和偏移 $\beta_1, \beta_2$，以及可能未知的焦距。
    - 核心思路：利用刚体变换的距离保持性 $\|\delta_{jk}^{(1)}\|^2 = \|\delta_{jk}^{(2)}\|^2$，消除旋转和平移，得到仅关于仿射参数的多项式方程组。(a) 标定 3 点求解器：3 个对应点产生 3 个四次方程（重参数化 $\gamma = \alpha^2$ 降为三次），使用 Gröbner 基求解，至多 4 个解，12×12 矩阵消去 + 4×4 特征值问题。(b) 共享焦距 4 点求解器：4 个对应点取 4 个方程，额外未知量 $\omega = 1/f^2$，至多 8 个解，36×36 模板。(c) 双焦距 4 点求解器：5 个方程求解 5 个未知量，意外地仅 4 个解，40×40 消去矩阵。求得仿射参数后，通过 SVD 对齐两视图 3D 点恢复 $R, t$。
    - 设计动机：现有方法（如 2pt+D）要么假设 scale-only 要么存在秩不足的退化。显式建模偏移是关键因为 SOTA MDE 模型（包括度量深度模型）确实存在视图间偏移不一致。

2. **混合估计框架（Hybrid Estimation Pipeline）**:

    - 功能：在 RANSAC 框架中同时利用深度感知约束和经典极线约束，互补提升鲁棒性。
    - 核心思路：每次 RANSAC 迭代随机选择深度感知求解器或经典求解器（5点/6点/7点），初始等概率选择，后续根据数据类型内点率调整。将每个对应点视为三种数据类型：$(p_1, p_2, d_1)$、$(p_1, p_2, d_2)$、$(p_1, p_2)$，分别用深度重投影误差 $E_{r(1\to2)}$、$E_{r(2\to1)}$ 和 Sampson 误差 $E_s$ 评估。对经典求解器的结果也估计仿射参数（通过三角化+最小二乘拟合）。联合 MSAC 评分为 $\bar{E} = \bar{E}_r + 2\lambda_s \frac{\tau_r}{\tau_s}\bar{E}_s$。
    - 设计动机：纯依赖深度先验在深度不可靠时会出错，纯基于点的方法又无法利用深度信息。混合方案让系统自适应地在两种信号源之间权衡。双向深度重投影误差的设计增强了对跨视图深度不一致的鲁棒性。

3. **混合局部优化（Hybrid Local Optimization）**:

    - 功能：在 RANSAC 内对所有参数 $\Theta = (R, t, \alpha, \beta_1, \beta_2)$ 进行联合优化。
    - 核心思路：在三种数据类型的内点集上，最小化 $E(\Theta) = \sum_{I_1} E_{r(1\to2)} + \sum_{I_2} E_{r(2\to1)} + 2\lambda_s \frac{\tau_r}{\tau_s} \sum_{I_3} E_s$。使用 Ceres 求解器的自动微分实现。
    - 设计动机：RANSAC 的局部优化步骤对最终精度影响巨大，混合目标函数确保优化同时考虑深度和极线约束。

### 损失函数 / 训练策略
本文方法是传统几何方法（非学习型），不需要训练。求解器用 C++ 实现并通过 pybind11 导出 Python 接口。可与任意特征匹配器和 MDE 模型组合使用。

## 实验关键数据

### 主实验
在 ScanNet-1500（室内）和 MegaDepth-1500（室外）上评估标定设置：

| 匹配器 | 方法 | 深度模型 | AUC@5° | AUC@10° | AUC@20° |
|--------|------|---------|--------|---------|---------|
| SP+LG | PoseLib-5pt | - | 21.55 | 39.11 | 55.60 |
| SP+LG | PoseLib-PnP | DA-metric | 15.05 | 34.16 | 53.97 |
| SP+LG | **Ours** | **DA-metric** | **显著提升** | **显著提升** | **显著提升** |
| RoMa | PoseLib-5pt | - | 基线 | 基线 | 基线 |
| RoMa | **Ours** | **MoGe** | **显著提升** | **显著提升** | **显著提升** |

### 消融实验
| 配置 | AUC@5°↑ | 说明 |
|------|---------|------|
| 仅 scale 校正 | 中等提升 | 不建模偏移 |
| Scale + shift 校正 | **大幅提升** | 完整仿射建模 |
| 仅深度求解器 | 提升但不稳定 | 缺少极线约束 fallback |
| 仅经典求解器 | 基线 | 不利用深度 |
| **混合估计** | **最高** | 两种约束互补 |
| 对"度量深度"加 shift | 仍有提升 | 证明度量深度也不完美 |

### 关键发现
- 偏移建模（shift）是最关键的贡献——即使对号称"度量深度"的模型（如 Depth Anything v2 metric），加上偏移校正仍能进一步提升位姿精度，说明所有 MDE 模型都存在视图间偏移不一致
- 混合估计显著优于单独使用任一种求解器，证明深度和极线约束在不同场景下的互补性
- 方法与不同特征匹配器（SP+LG、RoMa）和 MDE 模型（Marigold、DA、MoGe）均一致提升，即插即用
- 与 MASt3R 的改进方向正交——可以通过使用 MASt3R 的对应点进一步提升

## 亮点与洞察
- **仿射校正的普适性发现**：证明了即使是度量深度模型也需要仿射校正这一反直觉但重要的发现，对整个利用 MDE 先验做几何任务的社区有指导意义
- **混合估计的优雅设计**：三种数据类型+双向重投影误差+自适应求解器选择，在 RANSAC 框架内自然整合了深度和几何约束
- **实用性极强**：C++ 实现+Python 绑定，兼容任意匹配器和 MDE 模型，可直接集成到现有 SfM/SLAM 流程

## 局限与展望
- 求解器基于代数几何（Gröbner 基），共享焦距和双焦距求解器的模板较大（36×36 和 40×40），数值稳定性可能在极端配置下受影响
- 仅处理固定相机的相对位姿，未扩展到滚动快门或鱼眼相机模型
- 与 DUSt3R/MASt3R 等端到端方法相比，仍需要分别运行匹配和深度估计，pipeline 更长
- 在深度先验完全错误的场景（如镜面反射、透明物体）中效果可能退化

## 相关工作与启发
- **vs PoseLib (5pt/7pt)**: 经典纯几何方法不利用深度先验。本文方法一致性提升约 3-8% AUC@5°
- **vs PnP-RANSAC**: 直接在 MDE 深度上做 PnP 不建模偏移，效果反而可能比纯 5pt 更差——验证了偏移建模的必要性
- **vs Ding et al. (3p3d/4p4d)**: 同样使用深度先验但仅建模尺度不建模偏移。本文的仿射建模在所有设置下更优
- **vs DUSt3R/MASt3R**: 端到端学习方法，但需要大规模训练。本文方法无需训练且改进方向正交

## 评分
- 新颖性: ⭐⭐⭐⭐ 仿射校正的形式化和混合估计框架是扎实的贡献，"度量深度也需要偏移"的发现有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多匹配器、多 MDE 模型的全面评估，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，从求解器推导到混合框架设计逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 即插即用的实用方案，对整个利用深度先验的几何视觉社区有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [RePoseD: Efficient Relative Pose Estimation with Known Depth Information](../../ICCV2025/3d_vision/reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)

<!-- RELATED:END -->
