---
title: >-
  [论文解读] LumiMotion: Improving Gaussian Relighting with Scene Dynamics
description: >-
  [CVPR 2026][3D视觉][逆渲染] LumiMotion 是首个利用场景动态（运动区域）作为监督信号来改善逆渲染的 Gaussian-based 方法，通过动静分离和运动揭示的材质变化来更好地分离光照与材质，albedo 估计 LPIPS 提升 23%，重光照提升 15%。
tags:
  - CVPR 2026
  - 3D视觉
  - 逆渲染
  - 2D高斯溅射
  - 动态场景
  - 材质估计
  - 重光照
---

# LumiMotion: Improving Gaussian Relighting with Scene Dynamics

**会议**: CVPR 2026  
**arXiv**: [2604.10994](https://arxiv.org/abs/2604.10994)  
**代码**: [https://joaxkal.github.io/LumiMotion/](https://joaxkal.github.io/LumiMotion/)  
**领域**: 3D视觉  
**关键词**: 逆渲染, 2D高斯溅射, 动态场景, 材质估计, 重光照

## 一句话总结
LumiMotion 是首个利用场景动态（运动区域）作为监督信号来改善逆渲染的 Gaussian-based 方法，通过动静分离和运动揭示的材质变化来更好地分离光照与材质，albedo 估计 LPIPS 提升 23%，重光照提升 15%。

## 研究背景与动机

**领域现状**：逆渲染旨在从图像中恢复几何、材质和光照。现有 Gaussian Splatting 方法（R3DG、IRGS、GI-GS）主要针对静态场景，在强直射光下容易将阴影和材质颜色混淆。

**现有痛点**：静态场景中难以区分"区域暗是因为阴影还是材质本身深色"，因为缺乏同一表面在不同光照条件下的观测。已有动态场景方法要么只针对人体 avatar，要么需要已知光照或多光照训练。

**核心矛盾**：准确分离材质和光照需要同一表面的多光照观测，但现实中通常只有单一光照条件。

**本文目标**：利用场景中物体的运动（如阴影移动、运动物体光照变化）作为天然的多光照监督信号。

**核心 idea**：运动揭示了同一表面在不同光照条件下的外观，为材质-光照分离提供了更强的约束。

## 方法详解

### 整体框架
两阶段方法：Stage 1 训练动态 2DGS 表示（学习几何+动静分离+时变颜色） → Stage 2 冻结几何和变形网络，联合优化材质参数（albedo、roughness）和环境光照，用光线追踪计算可见性和间接光。

### 关键设计

1. **Binary Concrete 动静模糊分离**:

    - 功能：在 Stage 1 中显式区分静态和动态高斯
    - 核心思路：为每个高斯引入辅助变量 $P$，用 Binary Concrete 分布（Bernoulli 的连续松弛）采样 $\tilde{P}$，乘以变形网络的输出。$\tilde{P}$ 接近 0 的高斯保持静态，接近 1 的跟随变形
    - 设计动机：移动的阴影可以被解释为颜色变化或高斯移动/消失——后者会导致 Stage 2 中无法为其分配稳定的 albedo。必须让阴影区域由静态高斯+颜色变化来解释

2. **乘性时变颜色模型**:

    - 功能：建模阴影移动和动态物体光照变化
    - 核心思路：颜色 $c' = c \cdot (1 - \Delta c)$，乘性形式模拟光照对表面的影响（呼应渲染方程），canonical 颜色 $c$ 近似 pseudo-albedo 作为 Stage 2 的初始估计
    - 设计动机：加性颜色变化不符合物理光照模型，乘性变化更自然且便于正则化

3. **分层采样光线追踪**:

    - 功能：高效计算逆渲染中的可见性和间接光
    - 核心思路：Stage 2 中冻结几何，光栅化 albedo/roughness/normal 为 G-buffer，对环境光图分层采样方向进行光线追踪计算可见性 $V$ 和间接光 $L_{\text{ind}}$，使用 Disney BRDF 模型
    - 设计动机：动态场景的阴影会随时间变化，光线追踪能精确捕获这种变化，为材质估计提供正确的光照信息

### 损失函数 / 训练策略
Stage 1: 重建损失 + 法线一致性 + 深度畸变 + 前景掩码 BCE + 动静分离正则（鼓励 P 趋向 0）+ 颜色变化正则。Stage 2: 渲染方程下的 L1 损失 + albedo 平滑正则。

## 实验关键数据

### 主实验

| 场景/指标 | LumiMotion | IRGS（次优） | 提升 |
|-----------|------------|-------------|------|
| Albedo LPIPS | 最优 | 次优 | -23% |
| Relighting LPIPS | 最优 | 次优 | -15% |
| Relighting PSNR | 最优 | 次优 | 显著 |

### 消融实验

| 配置 | Relighting PSNR | 说明 |
|------|----------------|------|
| Full (动态) | 最优 | 利用动态信息 |
| 静态 baseline | 较差 | 阴影混入 albedo |
| w/o 动静分离 | 下降 | 动态高斯干扰 albedo |

### 关键发现
- 动态场景下 LumiMotion 能成功从 albedo 中去除阴影，静态方法则将阴影烘焙进 albedo
- 在同一场景的静态/动态两个版本上，动态版本的逆渲染结果一致优于静态版本
- Binary Concrete 分离对正确的 albedo 估计至关重要

## 亮点与洞察
- **运动作为监督信号**：这个观察非常有洞察力——运动天然提供了同一表面在不同光照下的样本，是一种"免费"的多光照数据
- **发布对照数据集**：新合成基准含静态/动态对照版本，首次系统评估动态对逆渲染的影响

## 局限与展望
- 假设光照静态不变，不适用于光照也变化的场景
- 需要场景中有足够的运动区域来提供监督
- 间接光建模仍较简化

## 相关工作与启发
- **vs IRGS**: IRGS 仅处理静态场景，阴影去除能力有限
- **vs Relightable Neural Actor**: 仅限人体 avatar，需已知光照

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次利用场景动态改善逆渲染，观察深刻
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据，静态/动态对照评估
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰
- 价值: ⭐⭐⭐⭐ 打开了动态逆渲染的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)
- [\[CVPR 2025\] ReCap: Better Gaussian Relighting with Cross-Environment Captures](../../CVPR2025/3d_vision/recap_better_gaussian_relighting_with_cross-environment_captures.md)
- [\[ICLR 2026\] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](../../ICLR2026/3d_vision/improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)

</div>

<!-- RELATED:END -->
