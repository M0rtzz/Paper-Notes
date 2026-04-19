---
title: >-
  [论文解读] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation
description: >-
  [CVPR2026][3D视觉][3D Gaussian Splatting] 提出 Physics-Guided Score Distillation 框架，利用物理仿真（MPM）作为运动先验引导 Video-SDS 优化，在静态 3DGS 场景中生成具有物理合理运动和真实感外观的动态天气效果（降雪、降雨、雾、沙尘暴）。
tags:
  - CVPR2026
  - 3D视觉
  - 3D Gaussian Splatting
  - 动态场景编辑
  - 天气效果
  - 物理仿真
  - Score Distillation Sampling
  - MPM
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation

**会议**: CVPR2026  
**arXiv**: [2504.05296](https://arxiv.org/abs/2504.05296)  
**代码**: [项目主页](https://galfiebelman.github.io/let-it-snow/)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 动态场景编辑, 天气效果, 物理仿真, Score Distillation Sampling, MPM

## 一句话总结

提出 Physics-Guided Score Distillation 框架，利用物理仿真（MPM）作为运动先验引导 Video-SDS 优化，在静态 3DGS 场景中生成具有物理合理运动和真实感外观的动态天气效果（降雪、降雨、雾、沙尘暴）。

## 研究背景与动机

**静态场景动态编辑需求大**：3D Gaussian Splatting 已能高效重建静态场景，但向其中添加时间维度的动态效果（如天气）仍是手工操作，门槛高。

**静态编辑方法无法建模时间演化**：ClimateNeRF、GaussCtrl 等只能做静态外观修改，无法表达持续的粒子发射与累积过程。

**物理仿真缺乏真实感外观**：PhysGaussian、PAC-NeRF 等基于物理的方法能提供合理运动，但无法为新引入的动态元素合成逼真外观。

**数据驱动 4D 生成运动不可控**：DreamGaussian4D、Animate124 等依赖扩散模型生成运动，在需要连续粒子发射的复杂多粒子场景中运动不连贯。

**运动和外观的根本矛盾**：物理仿真给出强运动先验但不够真实，Video-SDS 能生成真实感外观但无法独立学习复杂运动——两者需要统一。

**现有 4D 编辑方法作用于固定高斯集合**：不支持天气效果所需的粒子持续发射、累积和移除机制。

## 方法详解

### 整体框架

框架分两阶段：(1) 利用 Material Point Method (MPM) 物理仿真生成参考运动轨迹作为先验；(2) 训练循环神经动力学模型，通过 Physics-Guided Score Distillation 联合优化运动和外观。

### 物理运动先验 (Physics-Based Motion Prior)

- 用 3DGS 重建静态场景后提取 mesh，将静态高斯映射为 MPM 粒子作为障碍物
- 引入动态粒子（设定发射区域、速率、初速度、材料属性），用 MPM 仿真计算运动轨迹
- **主动粒子追踪**：粒子静止或出界后自动从仿真中移除，支持大规模粒子仿真
- **Mesh 碰撞精修**：针对 MPM 粗网格不足，设计不同天气的碰撞处理——雪投影到表面并插值附近高斯实现自然堆积；雨用 3D 湿度网格追踪水分带高斯平滑与时间衰减；沙尘沿法线位移并各向异性缩放

### 循环神经动力学模型 (Recurrent Neural Dynamics Model)

- 输入：上一时刻渲染状态（位置、旋转、外观）+ 物理仿真速度 + 时间步
- 为 active 和 collided 两种物理状态分别设 MLP
- 位置/速度用 Fourier 特征编码，时间用正弦编码
- 输出：速度修正 $\Delta \mathbf{v}$、角速度 $\boldsymbol{\omega}$、外观增量 $\Delta \mathcal{A}$
- 运动更新：$\mathbf{v}_g(t) = \mathbf{v}_g^{\text{init}}(t) + \Delta \mathbf{v}_g$，$\mathbf{x}_g(t) = \mathbf{x}_g(t{-}1) + \mathbf{v}_g(t)$
- 外观参数由 LLM 根据天气文本描述初始化

### 损失函数

总损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Video-SDS}} + \lambda_{\text{xyz}}\mathcal{L}_{\text{xyz}} + \lambda_{\text{vel}}\mathcal{L}_{\text{vel}} + \lambda_{\text{rot}}\mathcal{L}_{\text{rot}} + \lambda_{\text{app}}\mathcal{L}_{\text{app}}$$

- **Video-SDS 损失**：利用文本-视频扩散模型提供真实感监督
- **位置正则** $\mathcal{L}_{\text{xyz}}$：学习轨迹与仿真轨迹的 L2 距离
- **速度正则** $\mathcal{L}_{\text{vel}}$：学习速度与仿真速度的 L2 距离
- **旋转正则** $\mathcal{L}_{\text{rot}}$：四元数角距离，防止旋转漂移
- **外观正则** $\mathcal{L}_{\text{app}}$：惩罚过大的外观增量，抑制循环累积误差
- **SDS 自适应权重**：所有正则项权重乘以 $|\mathcal{L}_{\text{Video-SDS}}|$ 动态缩放——扩散模型不确定时加强物理引导，确信时放松约束

## 实验

### 实验设置

- **数据集**：MipNeRF 360 (Garden, Bicycle, Stump) + Tanks and Temples (Playground, Truck)，共 5 个场景
- **天气效果**：降雪、降雨、雾、沙尘暴 + 创意文本变体（紫色雪、闪光黄沙、魔法粒子）
- **评估指标**：CLIP_Sim、CLIP_Dir、VQAScore、ViCLIP-T、VE-Bench

### 与基线对比 (Table 1)

| 方法 | CLIP_Sim↑ | CLIP_Dir↑ | VQAScore↑ | ViCLIP-T↑ | VE-Bench↑ |
|------|-----------|-----------|-----------|-----------|-----------|
| ClimateNeRF (F+S) | 0.23 | 0.07 | 0.87 | 0.15 | 0.28 |
| GaussCtrl (F+S) | 0.25 | 0.08 | 0.71 | 0.16 | 0.24 |
| **Ours (F+S)** | **0.29** | **0.12** | **0.92** | **0.20** | **0.45** |
| GaussCtrl (All) | 0.24 | 0.07 | 0.64 | 0.15 | 0.21 |
| **Ours (All)** | **0.28** | **0.11** | **0.89** | **0.19** | **0.41** |

在所有图像和视频指标上全面超越静态编辑方法，VE-Bench 提升尤为显著（+61%）。

### 消融实验 (Table 2)

| 变体 | CLIP_Sim | CLIP_Dir | VQAScore | ViCLIP-T | VE-Bench |
|------|----------|----------|----------|----------|----------|
| w/o 碰撞处理 | 0.24 | 0.10 | 0.83 | 0.16 | 0.34 |
| w/o 外观优化 | 0.25 | 0.10 | 0.82 | 0.16 | 0.37 |
| w/o 运动仿真 | 0.18 | 0.03 | 0.35 | 0.08 | 0.13 |
| w/o 物理引导 | 0.26 | 0.10 | 0.85 | 0.17 | 0.37 |
| **完整方法** | **0.28** | **0.11** | **0.89** | **0.19** | **0.41** |

### 关键发现

- **去掉运动仿真 (w/o Motion) 退化最严重**：Video-SDS 单独无法学习多粒子连续发射的物理运动，VQAScore 从 0.89 降至 0.35
- **物理引导是联合优化的关键**：固定物理运动不做联合优化 (w/o PG) 也会阻碍 Video-SDS 外观优化
- **碰撞处理不可或缺**：去掉后粒子悬浮于表面上方，无法实现自然堆积
- **SDS 自适应权重优于固定权重**：固定权重无论大小都有问题——太小产生噪声伪影，太大过约束阻碍外观精修
- 与 4D 编辑基线 Instruct-4DGS 对比，后者缺乏物理先验导致运动不连贯（VQAScore 0.57 vs 0.89）

## 亮点

- **核心洞察优雅**：物理仿真作为"可软约束的运动先验"而非"硬约束"，完美调和了物理合理性与视觉真实感的矛盾
- **SDS 自适应权重**：根据扩散模型不确定度动态调节物理约束强度，免去繁琐的权重调参
- **通用天气框架**：同一框架支持雪/雨/雾/沙尘暴四种截然不同的天气效果，还能响应创意文本提示
- **完整的消融体系**：对碰撞处理、运动/外观优化、物理引导、自适应权重、各正则项逐一消融，论证充分

## 局限性

- **无双向交互**：动态粒子不会引起静态场景元素的形变或运动
- **静态高斯外观不更新**：不会反映天气带来的环境光照/阴影变化
- **轨迹漂移阈值**：若优化轨迹偏离物理先验过远，引导速度信号变得不可靠
- **依赖 MPM 仿真质量**：物理先验的好坏直接影响最终效果
- **计算开销**：MPM 仿真 + Video-SDS 优化的双阶段流程较重

## 相关工作

- **静态 3D 编辑**：ClimateNeRF (ICCV'23) 添加静态天气效果但无时间动态；GaussCtrl (ECCV'24) 文本引导编辑高斯但限于外观修改
- **物理驱动动画**：PhysGaussian (CVPR'24) 用 MPM 驱动已有场景元素运动但无法合成新元素外观；RainyGS 专注降雨但不能泛化到其他天气
- **4D 生成/编辑**：Gaussians2Life、Instruct-4DGS 等基于 Score Distillation 做 4D 编辑，但在多粒子连续发射场景中运动不可控
- **本文定位**：首次将物理仿真先验与 Video-SDS 统一到同一优化框架，填补了"动态天气场景编辑"这一空白

## 评分

- 新颖性: ⭐⭐⭐⭐ — 物理先验引导 SDS 的统一框架思路新颖，SDS 自适应权重设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 场景 × 4 天气 × 多视角，消融极为详尽（碰撞、运动、外观、物理引导、自适应权重、各正则项、轨迹漂移）
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表配合好，问题动机阐述到位
- 价值: ⭐⭐⭐⭐ — 为 3DGS 动态编辑开辟新方向，框架可扩展到更多动态效果场景

<!-- RELATED:START -->

## 相关论文

- [PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](../../ICCV2025/3d_vision/physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)
- [RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](rap_fast_feedforward_rendering-free_attribute-guided_primitive_importance_score_.md)
- [Stable Score Distillation](../../ICCV2025/3d_vision/stable_score_distillation.md)
- [Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)
- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](../../ICCV2025/3d_vision/identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)

<!-- RELATED:END -->
