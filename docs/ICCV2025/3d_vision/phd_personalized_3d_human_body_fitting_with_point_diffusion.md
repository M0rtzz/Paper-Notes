---
title: >-
  [论文解读] PHD: Personalized 3D Human Body Fitting with Point Diffusion
description: >-
  [ICCV 2025][3D视觉][人体姿态估计] 提出个性化3D人体姿态估计范式PHD——先通过SHAPify校准用户体型，再用体型条件化的点扩散模型PointDiT作为3D先验，结合Point Distillation Sampling损失迭代优化姿态，在绝对姿态精度上达到EMDB数据集SOTA。
tags:
  - ICCV 2025
  - 3D视觉
  - 人体姿态估计
  - 个性化体型
  - 点扩散模型
  - 体型先验
  - SMPL拟合
  - 即插即用
---

# PHD: Personalized 3D Human Body Fitting with Point Diffusion

**会议**: ICCV 2025  
**arXiv**: [2508.21257](https://arxiv.org/abs/2508.21257)  
**代码**: [https://PHD-Pose.github.io](https://PHD-Pose.github.io) (有)  
**领域**: 3D视觉  
**关键词**: 人体姿态估计, 个性化体型, 点扩散模型, 体型先验, SMPL拟合, 即插即用

## 一句话总结
提出个性化3D人体姿态估计范式PHD——先通过SHAPify校准用户体型，再用体型条件化的点扩散模型PointDiT作为3D先验，结合Point Distillation Sampling损失迭代优化姿态，在绝对姿态精度上达到EMDB数据集SOTA。

## 研究背景与动机

### 核心问题
从单目视频恢复精确的3D人体姿态和体型是AR/VR远程呈现、服务机器人等未来AI系统的基础。

### 现有方法的两大核心缺陷

**问题1：体型-姿态纠缠**
现有方法（HMR2.0b、CameraHMR等）逐帧同时估计体型/姿态/骨盆位置，设计为"用户无关"（subject-agnostic）。但同一视频中用户体型不应变化——不稳定的体型估计迫使姿态参数去补偿体型误差以满足优化目标。典型表现：错误的体型导致膝盖不自然弯曲以对齐2D投影。

**问题2：过度依赖2D对齐**
优化方法通过最小化2D关键点投影误差来精化姿态（regress-then-refine范式），但2D投影固有深度歧义——2D对齐好不代表3D准确。更严重的是，用于训练回归器的3D伪标注本身就是通过2D对齐获得的，形成错误传播的闭环。

### 关键洞察

**个性化是必要的**：利用用户身份一致性（体型不变），解耦体型和姿态估计

**3D先验不应基于关节角度**：关节角度表示与图像/体型的相关性弱，导致条件生成模型效果差；应使用体表面点云——与体型条件天然高度相关

## 方法详解

### 整体框架（两阶段解耦）
1. **个性化阶段（SHAPify）**：从单帧校准用户体型 $\beta^*$，仅需一次
2. **姿态拟合阶段**：用 $\beta^*$ 作为条件，通过PointDiT采样+2D关键点约束交替迭代优化姿态 $\theta, \phi, \mathbf{p}$

### 关键设计

#### 1. SHAPify：个性化体型校准
从一张用户站立参考姿态（T-pose/I-pose）图像 + 可选的身高体重信息估计体型。

**优化目标**：
$$\text{argmin}_{\phi,\theta,\beta,\mathbf{p}} \mathcal{L}_{rep} + \mathcal{L}_{reg}$$

重投影损失：
$$\mathcal{L}_{rep} = \|\Pi(J_{3D}(\phi, \theta, \beta) + \mathbf{p}) - J_{2D}\|_1$$

正则项（解决2D关键点对体型约束不足的问题）：
$$\mathcal{L}_{reg} = \lambda_\beta \|\beta\|_2^2 + \lambda_h \|H(\beta) - h\|_1 + \lambda_w \|W(\beta) - w\|_1$$
其中 $H(\cdot), W(\cdot)$ 是从SMPL mesh计算身高/体重的可微函数。$(h, w)$ 可用平均值或用户提供的真实数据。

**关键技巧**：初始化 $\theta$ 为预定义休息姿势，给 $\theta$ 和 $\mathbf{p}$ 设置更小的学习率，优先改变 $\beta$ 和 $\phi$。

#### 2. PointDiT：体型条件化的点扩散先验

**表示选择**：用体表面点云（$S=238$个mesh顶点 + $J=45$个关节点）作为姿态表示，而非关节角度。关节角度与图像/体型条件的相关性弱，在非常见姿态上表现差。

**架构**（基于Diffusion Transformer + Rectified Flow）：
- 前向过程：简单线性插值 $\mathbf{x}_t = (1-t/T)\mathbf{x}_0 + (t/T)\epsilon$
- 反向采样：$\mathbf{x}_{t-1} = \mathbf{x}_t - \hat{\mathbf{u}}(\mathbf{x}_t, t)$
- 训练损失：条件流匹配 $\mathcal{L}_{CFM} = \mathbb{E}[w_t \|\hat{\epsilon}(\mathbf{x}_t, t) - \epsilon\|_2^2]$

**条件注入**：
- 图像条件：ViTPose提取图像token和2D热力图 → 256个条件token → self-attention
- 体型条件：SMPL shape参数 $\beta$ → adaLN-Zero（替换原始的类别embedding）
- Rectified flow调度：仅需 $T=5$ 步去噪，对迭代拟合至关重要

**训练数据**：仅用合成数据集BEDLAM，保证纯净的GT shape和pose。

#### 3. Point Distillation Sampling + 采样-拟合循环

受Score Distillation Sampling启发，提出**Point Distillation Sampling**：

**点云对齐损失**（pelvis-aligned L2）：
$$\mathcal{L}_p = \lambda_p \|\mathbf{x}_0 - P_{3D}(\phi, \theta, \beta^*)\|_2$$

**角度对齐损失**（通过Point Fitter将点云转回SMPL参数后约束）：
$$\mathbf{x}_0 \xmapsto{\text{Point Fitter}} (\phi_g, \theta_g) \quad \Rightarrow \quad \mathcal{L}_a = \lambda_\phi \|\phi_g - \phi\|_2 + \lambda_\theta \|\theta_g - \theta\|_2$$

**采样-拟合交替循环**：
在第$k$次迭代中：
1. 用当前参数 $(\phi^k, \theta^k)$ 计算数据项 $\mathcal{L}_{data}$ 和先验项 $\mathcal{L}_{prior}$
2. 更新参数得 $(\phi^{k+1}, \theta^{k+1})$
3. 生成新点云 $\mathbf{x}^{k+1} = P_{3D}(\phi^{k+1}, \theta^{k+1})$
4. 在小噪声级别（$t/T=0.75$）扰动后重新采样，为下一轮提供更新的3D先验

$$\text{argmin}_{\phi,\theta,\mathbf{p}} \underbrace{\|\Pi(J_{3D}(\phi,\theta,\beta^*)+\mathbf{p}) - J_{2D}\|_1}_{\mathcal{L}_{data}} + \underbrace{\mathcal{L}_p + \mathcal{L}_a}_{\mathcal{L}_{prior}}$$

## 实验

### 主实验：EMDB1 Pelvis-aligned姿态精度

| 方法 | MPJPE↓ | PA-MPJPE↓ | MVE↓ | PA-MVE↓ |
|------|------|------|------|------|
| ScoreHMR Sample init. | 114.0 | 82.3 | 141.3 | 101.9 |
| **PHD Sample init.** | **73.6** | **49.2** | **86.4** | **59.1** |
| HMR2.0b init. | 117.2 | 77.9 | 140.2 | 93.9 |
| + ScoreHMR | 105.5 | 70.0 | 124.5 | 84.7 |
| **+ PHD (Ours)** | **73.2** | **47.4** | **86.4** | **58.5** |
| CameraHMR init. | 70.3 | 43.3 | 81.7 | — |
| + ScoreHMR | 74.9(+4.6) | 45.0(+1.7) | 89.0(+7.3) | — |
| **+ PHD (Ours)** | **62.5(-7.8)** | **42.4(-0.9)** | **74.6(-7.1)** | — |

关键发现：PHD在所有初始化策略下均大幅提升，而ScoreHMR在CameraHMR初始化下反而降低性能。

### 绝对姿态精度（C-MPJPE）

| 方法 | Pelvis Err.↓ | C-MPJPE↓ |
|------|------|------|
| HMR2.0b | 144.0 | 182.0 |
| + ScoreHMR | 180.6(+36.6) | 181.4(-0.6) |
| **+ PHD** | **94.7(-49.3)** | **112.6(-69.4)** |
| CameraHMR | 163.0 | 160.3 |
| **+ PHD** | **130.9(-32.1)** | **135.6(-27.4)** |

局部姿态好不等于绝对姿态好——CameraHMR MPJPE低但Pelvis Err.高。PHD个性化体型显著提升绝对精度。

### 消融：点云 vs 关节角度表示

| 表示 | MPJPE↓ | PA-MPJPE↓ | PA-MVE↓ |
|------|------|------|------|
| 6D Angular | 177.9 | 125.2 | 154.8 |
| ScoreHMR (angular) | 150.0 | 102.3 | 128.0 |
| **Points (Ours)** | **75.6** | **52.1** | **62.1** |

点云表示在相同条件下误差不到关节角度的一半，尤其在非常见姿态上优势明显。

### 体型校准精度

| 方法 | 关节误差均值↓ | 顶点误差均值↓ |
|------|------|------|
| CameraHMR | 30.60 | 31.85 |
| NLF | 19.36 | 20.61 |
| SHAPY | 22.94 | 21.38 |
| **SHAPify (w/ 测量)** | **11.29** | **9.18** |

SHAPify利用身高体重约束将体型精度提升到远超数据驱动方法的水平。

## 亮点与洞察
1. **范式创新**：从"通用估计"到"个性化拟合"——解耦体型和姿态是提升精度的关键
2. **表示选择的深层逻辑**：点云作为姿态表示而非关节角度，核心原因是条件生成模型需要条件（图像/体型）与输出有强相关性，而图像特征与表面点的关系远比与关节旋转角度直观
3. **即插即用设计**：可与任意3D姿态估计器组合使用，作为后处理模块提升精度
4. **仅需合成数据**：PointDiT仅在BEDLAM上训练，避免了伪标注数据的噪声传播

## 局限性
1. SHAPify需要用户配合拍一张参考姿势照片，增加部署门槛
2. 个性化体型假设视频中体型不变——穿脱衣服等场景可能失效
3. 迭代拟合过程（PointDiT采样5步 × 多次拟合迭代）计算开销较大，非实时
4. 评估主要在EMDB数据集上，更多in-the-wild场景验证有限

## 相关工作
- **Regress-then-refine范式**: SMPLify → ScoreHMR → PHD（从固定先验到图像条件先验到体型+图像条件先验）
- **3D姿态先验**: GMM(SMPLify) → GAN/VAE → 扩散模型(ScoreHMR/本文)
- **个性化方法**: SHAPY（体型数据驱动）→ PHD（优化+测量约束）

## 评分
- 新颖性：⭐⭐⭐⭐⭐ — 个性化+点扩散+迭代拟合的组合创新，表示选择洞察深刻
- 技术深度：⭐⭐⭐⭐⭐ — 从体型校准到扩散先验到拟合循环的完整技术链
- 实验充分度：⭐⭐⭐⭐ — 对比和消融充分，但数据集覆盖偏窄
- 实用价值：⭐⭐⭐⭐ — 即插即用提升现有方法，但需个性化校准步骤

<!-- RELATED:START -->

## 相关论文

- [ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)
- [HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)
- [GUAVA: Generalizable Upper Body 3D Gaussian Avatar](guava_generalizable_upper_body_3d_gaussian_avatar.md)
- [Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](../../CVPR2025/3d_vision/ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)

<!-- RELATED:END -->
