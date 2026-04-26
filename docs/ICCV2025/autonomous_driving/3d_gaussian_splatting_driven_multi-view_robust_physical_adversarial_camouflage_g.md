---
title: >-
  [论文解读] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation
description: >-
  [ICCV 2025][自动驾驶][physical adversarial attack] 提出PGA，首个基于3DGS的物理对抗攻击框架，通过快速准确重建目标+解决Gaussians互/自遮挡问题+min-max背景对抗优化策略，生成跨视角鲁棒的物理对抗迷彩，在数字和物理域均超越SOTA方法。
tags:
  - ICCV 2025
  - 自动驾驶
  - physical adversarial attack
  - adversarial camouflage
  - 3D Gaussian Splatting
  - multi-view robustness
  - 目标检测
---

# 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation

**会议**: ICCV 2025  
**arXiv**: [2507.01367](https://arxiv.org/abs/2507.01367)  
**代码**: [https://github.com/TRLou/PGA](https://github.com/TRLou/PGA)  
**机构**: Sun Yat-Sen University, NTU, NUS, Peng Cheng Lab
**领域**: 自动驾驶 / 对抗攻击 / 物理对抗  
**关键词**: physical adversarial attack, adversarial camouflage, 3D Gaussian Splatting, multi-view robustness, autonomous driving, object detection

## 一句话总结
提出PGA，首个基于3DGS的物理对抗攻击框架，通过快速准确重建目标+解决Gaussians互/自遮挡问题+min-max背景对抗优化策略，生成跨视角鲁棒的物理对抗迷彩，在数字和物理域均超越SOTA方法。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：物理对抗攻击暴露了DNN在安全关键场景（自动驾驶）的脆弱性。对抗迷彩比对抗patch更有前景，因为它能覆盖整个对象表面，在复杂物理环境中具有更强的对抗效力。然而现有方法存在两大限制：(1) 依赖目标物体的mesh先验和CARLA等模拟器构建虚拟环境，既耗时又与真实世界有不可避免的差距；(2) 训练图像中背景有限，使优化的迷彩难以跨视角鲁棒，容易陷入次优解。

### 解决思路

**本文目标**：如何生成在多视角和多物理环境下都有效且鲁棒的对抗迷彩？三大挑战：(1) 如何快速精确地建模任意目标物体而无需mesh先验？(2) 如何确保不同视角下迷彩图案的一致性？(3) 如何使迷彩在不同背景/天气/距离下保持对抗效力？

## 方法详解

### 整体框架（PGA）
PGA包含三个模块：**重建模块** → **渲染模块** → **攻击模块**。
- 重建模块：用少量多视角图像通过3DGS重建目标场景
- 渲染模块：从指定相机视角渲染图像，用SAM提取目标mask，合成检测用图像
- 攻击模块：迭代优化Gaussians的球谐系数来生成对抗迷彩

### 关键设计

1. **基于3DGS的重建与渲染**：

    - 利用3DGS从少量图像快速准确重建目标车辆和场景，无需手工建mesh
    - 3DGS提供可微的、照片级真实感的多视角渲染能力
    - 仅优化球谐系数k_g（控制表面颜色），不改变形状参数，保证物理可部署性
    - 迭代攻击公式：k^{t+1} = k^t + η∇_k L_det

2. **解决跨视角成像不一致（核心创新之一）**：

    - **互遮挡问题**：vanilla 3DGS中部分Gaussians位于物体内部，视角变化时遮挡关系改变导致迷彩不一致。解决方案：引入SuGaR正则化，将Gaussians对齐到物体表面并降低opacity，防止内部Gaussians遮挡表面。
    - **自遮挡问题**：高阶球谐函数使单个Gaussian在不同视角呈现截然不同的颜色。解决方案：攻击迭代中仅优化零阶球谐系数⟨k⟩_0，确保每个Gaussian表面颜色均匀。
    - 这两项改进保证了跨视角迭代优化时同一迷彩图案被一致优化。

3. **Min-Max鲁棒迷彩优化（核心创新之二）**：

    - 将迷彩优化视为UAP(Universal Adversarial Perturbation)问题
    - 对每个视角逐一迭代优化，设迭代限制，攻击成功即跳到下一视角
    - **背景对抗扰动**：每次优化迷彩前，先用I-FGSM在背景区域添加噪声σ最大化检测损失，然后再优化迷彩最小化检测损失
    - 数学形式：G' = argmin_G max_σ L_det(I_det(θ_c, G) + σ·(1-M))，s.t. ||σ||_∞ ≤ ε
    - 这有效过滤掉依赖特定背景的非鲁棒对抗特征

4. **额外增强技术**：

    - EoT (Expectation over Transformations)：模拟物理世界的随机变换
    - NPS (Non-Printability Score)：确保颜色可打印
    - 主色正则化：提升迷彩视觉自然度

### 攻击目标
检测损失L_det：最小化与GT有最大IoU的预测框的置信度分数，使检测器漏检或误分类。

## 实验关键数据

### 数字域攻击效果（AP@0.5 %, 越低越好 = 攻击越有效）

| 距离 | 方法 | Faster R-CNN | YOLO-V5* | Mask R-CNN* | D-DETR* | 平均 |
|------|------|-------------|----------|-------------|---------|------|
| 5m | Clean | 71.86 | 70.57 | 73.18 | 79.76 | 73.72 |
| 5m | RAUCA | 21.71 | 46.94 | 31.90 | 36.54 | 37.16 |
| 5m | **PGA** | **4.52** | **39.10** | **10.62** | **28.31** | **23.46** |
| 10m | RAUCA | 18.88 | 56.70 | 31.00 | 44.85 | 39.25 |
| 10m | **PGA** | **1.40** | **45.53** | **8.44** | **30.89** | **21.78** |

- PGA将白盒Faster R-CNN的AP降至1-5%，远超所有SOTA方法
- 黑盒迁移到Mask R-CNN效果突出（8-11% vs RAUCA 31%）

### 物理域实验
- 在真实车辆上部署对抗迷彩，多种距离/俯仰角/天气条件下均保持高攻击成功率
- 物理域效果也显著优于所有对比方法

### 消融实验
- 互遮挡正则化 + 自遮挡解决 + Min-max优化，每项贡献显著
- 所有组件组合达到最优效果

## 亮点与洞察
- **首个基于3DGS的物理攻击框架**：利用3DGS的快速重建和照片级渲染能力，彻底摆脱对mesh先验和模拟器的依赖
- **互/自遮挡问题的分析与解决很有洞察力**：精准识别vanilla 3DGS应用于迷彩生成的两个关键问题
- **Min-max背景对抗策略设计巧妙**：通过对抗博弈自动过滤非鲁棒特征
- **全面的实验验证**：数字域多距离多天气 + 物理域真实部署 + 跨模型黑盒迁移 + 红外检测扩展

## 局限与展望
- 仅优化零阶SH丢失了视角相关颜色信息
- 物理部署需要将Gaussians转化为mesh纹理再打印，转换过程可能引入精度损失
- Min-max优化的背景噪声预算ε需要手动调节
- 仅在车辆检测任务上验证
- 防御方法的鲁棒性评估不足

## 相关工作与启发
- **vs. DAS/FCA/ACTIVE/TAS**：传统基于mesh+可微渲染的方法，需要目标mesh先验+模拟器
- **vs. RAUCA**：虽考虑天气因素但仍依赖mesh和neural renderer
- **vs. NeRF-based attacks**：NeRF渲染慢、质量低、内存大；PGA用3DGS全面超越
- **PGA优势**：无需mesh先验 + 快速精确重建 + 照片级渲染 + 跨视角一致性保证

## 相关工作与启发
- 3DGS在对抗攻击中的应用揭示了高保真渲染对物理攻击有效性的重要性
- 互/自遮挡的分析对3DGS在需要跨视角一致性的任务中都有参考价值
- Min-max框架的思路可迁移到其他需要环境鲁棒性的优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将3DGS引入物理攻击，互/自遮挡分析和min-max策略都是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 数字域+物理域+消融+红外扩展，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，但公式符号较多
- 价值: ⭐⭐⭐⭐ 推进physical attack SOTA，也给防御研究提供参考

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_f.md)
- [\[ICCV 2025\] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)
- [\[ICCV 2025\] PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection](pbcat_patch-based_composite_adversarial_training_against_physically_realizable_a.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)

<!-- RELATED:END -->
