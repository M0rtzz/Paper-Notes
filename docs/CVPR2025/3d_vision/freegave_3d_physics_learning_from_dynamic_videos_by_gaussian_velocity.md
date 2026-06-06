---
title: >-
  [论文解读] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity
description: >-
  [CVPR 2025][3D视觉][3D物理学习] 提出 FreeGave，一个从多视角动态视频中学习 3D 场景几何、外观和物理速度的通用框架，通过为每个 3D 高斯核引入可学习的物理编码（physics code）并设计无散度（divergence-free）速度场参数化…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D物理学习"
  - "高斯速度场"
  - "无散度速度场"
  - "动态3DGS"
  - "未来帧预测"
---

# FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity

**会议**: CVPR 2025  
**arXiv**: [2506.07865](https://arxiv.org/abs/2506.07865)  
**代码**: [https://github.com/vLAR-group/FreeGave](https://github.com/vLAR-group/FreeGave)  
**领域**: 3D视觉 / 动态场景重建  
**关键词**: 3D物理学习, 高斯速度场, 无散度速度场, 动态3DGS, 未来帧预测

## 一句话总结
提出 FreeGave，一个从多视角动态视频中学习 3D 场景几何、外观和物理速度的通用框架，通过为每个 3D 高斯核引入可学习的物理编码（physics code）并设计无散度（divergence-free）速度场参数化，在不依赖 PINN 损失和目标先验的条件下实现精准的未来帧外推。

## 研究背景与动机

**领域现状**：3DGS 及其动态变体（如 Deformable 3DGS、4DGS）在动态场景的新视角渲染上已达到很高水准，但大多只能在训练时间段内进行插值，无法预测未来——因为它们没有显式学习物理属性，只是在拟合视觉观测。

**现有痛点**：（1）PINN 类方法将 PDE 转为损失函数做软约束，但在边界区域学到的物理不准确，且需要在时空维度密集采样，训练效率低下；（2）显式物理模型方法（如弹簧系统、图神经网络）需要目标先验（如物体掩码、类型），泛化性差，通常只能处理特定类型运动（流体或弹性体）。

**核心矛盾**：物理学习的通用性与准确性之间的权衡——PINN 够通用但边界不准，显式物理模型够准但需要领域先验。

**本文目标** 从纯 RGB 多视角视频出发，在不知道物体数量、类型、掩码的情况下，学习场景中所有物体/部件的 3D 速度场，实现物理上合理的未来帧预测。

**切入角度**：将每个 3D 高斯核视为一个刚体粒子，为其学习一个潜在"物理编码"来描述其运动类型（如质量、力等信息的抽象），然后用精心设计的无散度参数化从编码中解码出速度场，结构性地满足物理约束而非依赖"软"损失。

**核心 idea**：用可学习的物理编码+结构化无散度速度场参数化替代 PINN 损失，实现零先验的 3D 物理学习。

## 方法详解

### 整体框架
三模块流水线：（1）Canonical 3DGS 模块在 t=0 学习场景的静态几何和外观；（2）Neural Divergence-free Gaussian Velocity 模块为每个高斯学习物理编码并解码出无散度的 6-DOF 速度场；（3）Deformation-aided Optimization 模块引入辅助变形场帮助优化收敛，用中点法积分速度场传播高斯核位置并与多视角图像比较生成训练信号。

### 关键设计

1. **Physics Code（物理编码）**:

    - 功能：为每个高斯核建模一个潜在向量，抽象地描述其跨时间不变的运动类型/物理属性
    - 核心思路：用 MLP $f_{code}$ 从高斯的 canonical 位置 $\mathbf{p}_0$ 预测 $L$ 维编码 $\mathbf{z} = f_{code}(\mathbf{p}_0)$。编码 $\mathbf{z}$ 在所有时间戳共享——它描述的是"运动模式"而非"某一时刻的状态"。相比给每个高斯独立学一个自由编码向量，基于位置的 MLP 预测引入了空间平滑正则化，临近的高斯倾向于获得相似编码。
    - 设计动机：仅用位置→速度的映射无法区分相邻但运动完全不同的物体（如静止桌子上滚动的球）。引入物理编码打破了这种"位置连续则速度连续"的限制，让网络能对不同运动模式产生不同的速度场。

2. **Divergence-free Velocity 参数化**:

    - 功能：确保估计的速度场满足无散度物理约束，无需 PINN 损失
    - 核心思路：将每个粒子的速度分解为 6 个基本分量 $\mathbb{V}_t = [v_t^x, v_t^y, v_t^z, w_t^z, w_t^y, w_t^x]$（3 个线速度 + 3 个角速度）和一个位置相关的基矩阵 $\mathcal{B}(\mathbf{p}_t)$，速度 $\mathbf{v} = \mathbb{V}_t \cdot \mathcal{B}(\mathbf{p}_t)$。关键在于 $\mathbb{V}_t$ 被设计为与位置 $\mathbf{p}_t$ 无关——通过 $\mathbb{V}_t = f_{neck}(\mathbf{z}) \cdot f_{weight}(t)$：物理编码经 MLP 解码为 $K$ 维运动模式，时间戳经另一 MLP 生成 $K \times 6$ 权重矩阵来选择/混合运动模式。
    - 设计动机：基矩阵 $\mathcal{B}$ 的形式保证了速度场的无散度性（散度恒为零），这是刚体运动的基本物理约束。无散度从结构上被满足，而非像 PINN 那样作为软惩罚。$f_{neck} \cdot f_{weight}$ 的乘法形式让物理编码和时间动态解耦——编码定义"做什么运动"，时间权重定义"什么时候做"。

3. **Deformation-aided Optimization**:

    - 功能：引入辅助变形场帮助训练收敛，并将速度场与渲染监督连接起来
    - 核心思路：辅助变形场 $f_{deform}(\mathbf{p}_0, t, \mathbf{z})$ 直接预测位移 $\delta\mathbf{p}$、旋转 $\delta\mathbf{r}$、缩放 $\delta\mathbf{s}$，把 canonical 高斯搬到时间 $t'$。然后从 $t'$ 出发，用 interleaved mid-point 积分速度场传播到 $t$，渲染并与真实图像对比。两条路径（变形场 + 速度场）共享物理编码 $\mathbf{z}$，变形场为速度场提供合理的"出发点"，解决了直接端到端训练难以收敛的问题。
    - 设计动机：直接训练速度场需要精确的位置→速度→新位置链条，但初始阶段所有参数都不准确，导致梯度噪声大。变形场相当于"辅助轮"，先让模型学会大致的运动模式，再让速度场负责精确的物理一致传播。

### 损失函数 / 训练策略
损失函数为标准 3DGS 的 $\ell_1 + \ell_{ssim}$ 渲染损失。训练分两阶段：先训练 canonical 3DGS，再联合训练物理编码+速度模块+变形场。在每次迭代中采样时间 $t$ 和间隔 $\Delta t$，用变形场定位 $t'$ 处高斯，再用 mid-point 方法积分到 $t$。

## 实验关键数据

### 主实验（未来帧外推）

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Dynamic Object | NVFi | 27.594 | 0.972 | 0.036 |
| Dynamic Object | DefGS_nvfi | 28.749 | 0.984 | 0.013 |
| Dynamic Object | **FreeGave** | **31.987** | **0.990** | **0.007** |
| Dynamic Indoor | NVFi | 29.745 | 0.876 | 0.204 |
| Dynamic Indoor | DefGS_nvfi | 31.096 | 0.945 | 0.077 |
| Dynamic Indoor | **FreeGave** | **35.019** | **0.966** | **0.051** |
| ParticleNeRF | NVFi | 18.173 | 0.867 | 0.119 |
| ParticleNeRF | DefGS_nvfi | 22.730 | 0.931 | 0.050 |
| ParticleNeRF | **FreeGave** | **26.657** | **0.956** | **0.030** |

### 消融实验（需结合论文 Table 3/4 推断）

| 配置 | 外推 PSNR | 说明 |
|------|----------|------|
| Full model | 最优 | 完整框架 |
| 无 physics code（直接回归速度） | 下降 | 无法区分相邻不同运动物体 |
| 独立 learnable code（无 MLP） | 下降 | 参数过多，过拟合 |
| 直接 MLP 解码 $\mathbb{V}_t$（无乘法分解） | 下降 | 过于灵活，物理/时间耦合 |
| 无 deformation 辅助 | 训练难收敛 | 缺少"辅助轮" |

### 关键发现
- FreeGave 在所有外推任务上大幅超越所有基线——Dynamic Indoor 上比最强基线高 ~4 dB PSNR
- 在新收集的 FreeGave-GoPro 真实数据集上，外推 PSNR 达 28.094，接近插值水平（28.451），展示了强大的真实场景泛化能力
- 物理编码的可视化揭示了有意义的聚类——相同运动模式的高斯获得相似编码，无需任何标签即实现了隐式运动分割
- DefGS（无速度场的变形 GS）外推性能大幅落后，证实了学习速度场相比直接拟合变形的必要性

## 亮点与洞察
- **结构化无散度参数化替代 PINN**：不用 PDE 做软约束，而是从速度场的参数化形式上"硬编码"物理规律——这是一个优雅的设计范式。任何已知的物理不变性都可尝试类似的结构化编码方式。
- **Physics Code 的涌现语义**：物理编码在无监督训练中自发地对运动模式聚类，可直接用于运动分割——这表明 3D 速度场学习不仅能做预测，还能做感知。可作为下游任务（如机器人操作、场景理解）的发现式表示。
- **乘法分解的设计哲学**：$f_{neck}(\mathbf{z}) \cdot f_{weight}(t)$ 的设计让物理和时间动态显式解耦——编码解释"做什么运动"，时间权重控制"什么时候启动/停止"。这种底层物理建模思路简洁有力。

## 局限与展望
- 假设每个高斯是刚体粒子——对高度形变物体（如布料、流体）可能不够准确
- 速度场的无散度假设对于存在质量源/汇（如烟雾消散、物体出现/消失）的场景不适用
- 训练需要多视角视频，单目视频场景未涉及
- GoPro 数据集仅 6 场景，真实数据验证规模偏小
- 可改进：扩展到非刚体运动（引入应变场或可压缩速度场）；结合 diffusion model 做条件未来生成；将物理编码用于下游机器人规划

## 相关工作与启发
- **vs NVFi**：NVFi 也学速度场但依赖 PINN 损失，在边界区域表现差。FreeGave 通过结构化参数化完全绕过 PINN，在所有数据集上大幅超越。
- **vs Deformable 3DGS (DefGS)**：DefGS 学变形场但不学物理，外推性能差。FreeGave 结合辅助变形场和速度场，同时具备拟合和预测能力。
- **vs PhysGaussian/PAC-NeRF**：这些方法需要显式物理模型和物体掩码/类型先验，通用性受限。FreeGave 不需要任何先验，更适合复杂真实场景。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 物理编码 + 无散度速度场参数化是全新设计，理论优雅实用
- 实验充分度: ⭐⭐⭐⭐ 4 个数据集+消融充分，但真实数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰严谨，但数学符号较多可能影响可读性
- 价值: ⭐⭐⭐⭐⭐ 为 3D 物理学习开辟了新范式，物理编码的涌现语义对下游任务有广泛价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Recovering Dynamic 3D Sketches from Videos](recovering_dynamic_3d_sketches_from_videos.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](stereo4d_learning_how_things_move_in_3d_from_internet_stereo_videos.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] PGC: Physics-Based Gaussian Cloth from a Single Pose](pgc_physics-based_gaussian_cloth_from_a_single_pose.md)

</div>

<!-- RELATED:END -->
