---
title: >-
  [论文解读] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion
description: >-
  [CVPR 2025][视频生成][4D视频生成] 提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "4D视频生成"
  - "双流架构"
  - "同步层"
  - "视频扩散"
  - "多视角一致性"
---

# 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion

**会议**: CVPR 2025  
**arXiv**: [2412.04462](https://arxiv.org/abs/2412.04462)  
**代码**: [https://snap-research.github.io/4Real-Video/](https://snap-research.github.io/4Real-Video/) (无独立代码仓库)  
**领域**: 视频生成 / 4D生成  
**关键词**: 4D视频生成, 双流架构, 同步层, 视频扩散, 多视角一致性

## 一句话总结
提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。

## 研究背景与动机

随着视频扩散模型（如Sora）的崛起，4D视频生成——即生成同时沿时间和视角两个轴变化的视频帧网格——成为重要的延伸方向。该能力对动态场景创建、沉浸式体验和基于图像的渲染至关重要。

**本文对4D视频的定义**：一个帧网格，行共享时间戳，列共享视角。区别于"camera-aware视频"（仅生成带相机控制的单路视频），4D视频网格提供完整的时空体验且更易做动态重建。

**现有方法的局限**：

**基于优化的方法**（如4Dfy、Dream-in-4D）：用SDS从预训练模型蒸馏，需要数小时且偏向object-centric/非写实输出。

**直接4D训练方法**（如SV4D、Diffusion4D）：在有限的4D数据（如Objaverse动画）上训练，泛化能力受限于训练数据分布。

**顺序交错方法**（如CVD）：交替执行视角注意力和时间注意力更新，但不能充分考虑两者的相互依赖；视角更新的输出对时间更新来说是out-of-distribution的，导致伪影。

**核心矛盾**：①高质量4D数据极度稀缺；②顺序交错架构破坏预训练视频模型的分布导致质量退化；③这些方法要么太慢（SDS需数小时）要么质量差。

**本文切入角度**：设计双流并行架构复用预训练视频模型权重，通过同步层协调时间和视角一致性，避免分布偏移。核心idea：**把4D生成建模为两个视频任务的联合优化，用变量分裂+同步来保持两流一致。**

## 方法详解

### 整体框架
- **输入**：第一行帧（固定视角视频）+ 第一列帧（冻结时间视频）
- **输出**：完整的 $V \times T$ 帧网格（默认8×8）
- **两阶段**：①训练base视频模型支持冻结时间/动态两种模式；②在此基础上训练双流4D视频模型

### 关键设计

1. **双流架构（Two-Stream Architecture）**:

    - 功能：将视频帧网格的token分成两个独立流并行处理——视角流处理行（freeze-time视频），时间流处理列（fixed-view视频）
    - 核心思路：将帧网格的token复制为 $\mathbf{x}_l^{\text{v}}$ 和 $\mathbf{x}_l^{\text{t}}$ 两份。每层中，视角流用预训练DiT做 $T$ 个并行行更新，时间流做 $V$ 个并行列更新：
    $\mathbf{y}_l^{\text{v}} = \mathbf{x}_l^{\text{v}} + \varphi_l^{\text{v}}(\mathbf{x}_l^{\text{v}}; \mathbf{c}^{\text{v}}); \quad \mathbf{y}_l^{\text{t}} = \mathbf{x}_l^{\text{t}} + \varphi_l^{\text{t}}(\mathbf{x}_l^{\text{t}}; \mathbf{c}^{\text{t}})$
   两流独立计算后通过同步层交换信息。
    - 设计动机：与顺序交错相比，并行双流避免了一个流的输出成为另一个流的out-of-distribution输入。预训练DiT层不微调，仅训练新增的同步层参数，避免破坏预训练视频模型的生成质量。

2. **同步层——Hard Synchronization**:

    - 功能：在每层DiT后严格合并两流token
    - 核心思路：通过可学习加权合并两流以满足约束 $\mathbf{x}^{\text{v}} = \mathbf{x}^{\text{t}}$：
    $\mathbf{x}_{l+1} = \mathbf{W}_l^{\text{v}} \mathbf{y}_l^{\text{v}} + \mathbf{W}_l^{\text{t}} \mathbf{y}_l^{\text{t}}$
   权重初始化为 $\frac{1}{2}\mathbf{I}$，并由扩散时间步 $\sigma$ 调制以适应不同去噪阶段。
    - 设计动机：类比优化中的投影梯度下降——每步将两个变量投影到等式约束流形上。但实验发现在大视角变化时会产生拉伸伪影，因为合并后的token偏离了base model的分布。

3. **同步层——Soft Synchronization**:

    - 功能：保持两流token独立的同时通过软更新使其趋近一致
    - 核心思路：用时间步调制的线性层预测非对称token增量：
    $(\Delta\mathbf{y}_l^{\text{v}}, \Delta\mathbf{y}_l^{\text{t}}) = \text{Mod\_Linear}(\mathbf{y}_l^{\text{v}}, \mathbf{y}_l^{\text{t}}; \sigma)$
    $\mathbf{x}_{l+1}^{\text{v}} = \mathbf{y}_l^{\text{v}} + \Delta\mathbf{y}_l^{\text{v}}, \quad \mathbf{x}_{l+1}^{\text{t}} = \mathbf{y}_l^{\text{t}} + \Delta\mathbf{y}_l^{\text{t}}$
    - 设计动机：类比ADMM等不严格满足约束但逐步趋近的优化算法。给模型更多灵活性，让不同层可以自适应调整同步强度。实验观察到更深层的同步强度自动增大，浅层允许两流一定程度的分歧。

4. **Base视频模型训练**:

    - 功能：训练支持freeze-time和dynamic两种生成模式的base视频模型
    - 核心思路：将训练数据分为动态视频和静态场景视频两类，用不同的context embedding控制生成模式。采用随机遮蔽训练策略，使模型能基于任意帧子集预测未见帧，支持自回归扩展。
    - 设计动机：为4D模型提供高质量的单维度视频生成基础；遮蔽训练使模型能灵活接受不同条件输入。

### 损失函数 / 训练策略
- **Base model**：像素空间扩散（非latent），渐进式分辨率训练（36×64 → 72×128），24×A100训练12天
- **4D model**：velocity matching loss（rectified flow），两阶段训练——先在2D变换伪4D视频上训练20k步，再在Animated Objaverse上微调3k步
- 微调时**仅更新同步层参数**，保持预训练DiT层冻结
- 用扩散上采样器将72×128提升至288×512

## 实验关键数据

### 主实验：4D视频生成质量

| 方法 | FID ↓ | CLIP ↑ | FVD ↓ | Visual Quality ↑ | Temporal Consist. ↑ |
|------|-------|--------|-------|-------------------|---------------------|
| SV4D | 204.81 | 19.46 | 1053.10 | 2.26/2.02 | 2.03/1.68 |
| MotionCtrl | 87.10 | 20.20 | 1556.36 | 2.36/2.30 | 2.38/2.25 |
| Sequential | 96.64 | 28.16 | 1662.54 | 2.30/2.28 | 2.21/2.15 |
| Hard Sync | 79.92 | 28.16 | 972.87 | 2.42/2.40 | 2.40/2.33 |
| **Soft Sync** | **78.36** | **28.22** | **906.16** | **2.43/2.42** | **2.41/2.36** |

### 消融实验：多视角一致性（Dust3R-Confidence）

| 配置 | τ=2.0 ↑ | τ=2.5 ↑ | τ=3.0 ↑ |
|------|---------|---------|---------|
| Sequential | 33.5 | 24.6 | 16.6 |
| Soft w/o Obj | 39.1 | 31.4 | 24.0 |
| Hard Sync | 39.3 | 31.5 | 23.8 |
| **Soft Sync** | **41.0** | **33.4** | **25.7** |

### 关键发现
- **Soft Sync > Hard Sync > Sequential**：在所有指标上soft同步均优于hard同步和顺序交错
- 即使不使用任何4D数据（Soft w/o Obj），仅用2D变换伪数据训练仍能产生有竞争力的结果
- 用户研究中在所有7项指标上大幅超越优化方法（4Dfy、Dream-in-4D、AYG、4Real）
- 生成速度约1分钟（8×8@288×512），而SDS方法需数小时
- Objaverse微调仅需3k步，但过度微调反而降低真实场景质量

## 亮点与洞察
- **优化理论视角的架构设计**：将DiT层类比为隐式优化的迭代求解器，将4D生成建模为变量分裂+约束优化问题，hard/soft同步分别对应投影梯度下降和ADMM，理论直觉优雅
- 仅训练同步层（极少参数），完全冻结预训练DiT权重，最大限度保留base model的泛化能力
- 像素空间扩散模型在小模型规模下训练更快、运动更连贯（对比latent-based）
- 无需显式相机位姿条件，通过条件视频自动推断视角

## 局限与展望
- Base model仅600M参数，限制了视觉质量和分辨率上限
- 不支持360°视角生成
- 对动态元素（如奔跑的马、火焰）的freeze-time视频生成鲁棒性有限
- 需要后处理步骤（如3DGS重建）才能获得显式3D表示
- 伪4D数据（2D仿射变换）训练可能导致前景物体在大视角变化时显得扁平

## 相关工作与启发
- 4Real（前作）使用视频模型生成冻结时间/动态视频后做优化重建，本文将其改为前馈式生成
- CVD提出用伪配对数据微调视频模型生成结构一致视频对，但顺序交错架构有局限
- SV4D直接在Objaverse上训练4D模型但无法泛化到真实场景
- 核心启发：**将4D生成问题分解为两个1D视频问题的联合优化，通过同步层而非共享token来保持一致性，是处理多轴生成的优雅方案**

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双流+同步层架构设计新颖，优化理论类比深刻
- 实验充分度: ⭐⭐⭐⭐ 指标全面（6类度量），ablation清晰，但受限于无真正4D ground truth
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，架构设计的动机推导从优化理论出发逻辑严密
- 价值: ⭐⭐⭐⭐ 首个可泛化到真实场景的高效4D视频生成方法，具有重要应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] Learning from Streaming Video with Orthogonal Gradients](learning_from_streaming_video_with_orthogonal_gradients.md)
- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](../../CVPR2026/video_generation/orbital_video_3d_foundation_priors.md)
- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](../../ICCV2025/video_generation/normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](../../ECCV2024/video_generation/vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
