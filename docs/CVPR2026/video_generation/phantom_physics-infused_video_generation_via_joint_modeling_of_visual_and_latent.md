---
title: >-
  [论文解读] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics
description: >-
  [CVPR 2026][物理一致视频生成] 提出Phantom框架，在预训练视频扩散模型（Wan2.2-TI2V）之上增加一个物理动力学分支，利用V-JEPA2提取的物理感知嵌入作为潜在物理状态，通过双向交叉注意力联合建模视觉内容和物理动力学演化，在物理一致性基准上大幅超越基线（VideoPhy PC提升50.4%），同时保持视觉质量。
tags:
  - CVPR 2026
  - 物理一致视频生成
  - 流匹配
  - 双分支架构
  - V-JEPA2
  - 潜在物理动力学
---

# Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics

**会议**: CVPR 2026  
**arXiv**: [2604.08503](https://arxiv.org/abs/2604.08503)  
**代码**: https://plan-lab.github.io/phantom  
**领域**: 视频生成 / 物理一致性  
**关键词**: 物理一致视频生成, 流匹配, 双分支架构, V-JEPA2, 潜在物理动力学

## 一句话总结
提出Phantom框架，在预训练视频扩散模型（Wan2.2-TI2V）之上增加一个物理动力学分支，利用V-JEPA2提取的物理感知嵌入作为潜在物理状态，通过双向交叉注意力联合建模视觉内容和物理动力学演化，在物理一致性基准上大幅超越基线（VideoPhy PC提升50.4%），同时保持视觉质量。

## 研究背景与动机

**领域现状**：以Sora、HunyuanVideo、Wan2.2为代表的视频生成模型已经能产生视觉逼真的视频，但在物理一致性方面仍存在明显缺陷——生成的物体经常违背重力、惯性、碰撞等基本物理定律。

**现有痛点**：(1) 单纯扩大模型规模和数据量不足以学习可泛化的物理定律，模型倾向于案例式记忆而非抽象物理规则；(2) 现有Physics-aware方法要么依赖外部物理模拟器（受限于模拟器的覆盖范围），要么依赖LLM提示工程在推理时引导（不增加模型内在的物理理解，且带来推理开销），要么通过表示对齐间接注入物理先验（不能显式建模物理状态演化）。

**核心矛盾**：当前视频生成模型主要依赖next-frame prediction目标，该目标优化视觉保真度但不显式强制物理推理，使模型难以内化和遵守真实世界的物理定律。

**本文目标** 如何在视频生成过程中直接整合对潜在物理属性的推理，使模型不仅生成视觉逼真、而且物理一致的视频？

**切入角度**：作者假设——模型无法学习物理动力学源于其仅依赖下一帧预测目标。解决方案是让模型同时预测视频内容和潜在物理参数。

**核心 idea**：在视频生成流程中增加一个专用的物理分支，利用V-JEPA2的自监督表示作为"潜在物理状态"，与视觉分支联合训练，使模型在生成视频的同时推理物理动力学。

## 方法详解

### 整体框架
Phantom基于Wan2.2-TI2V-5B构建，采用双分支并行的latent flow-matching架构：(1) **视频分支**复用预训练的Wan2.2模块处理视觉潜在序列；(2) **物理分支**镜像视频分支的架构但从零初始化，在V-JEPA2潜在空间中预测物理动力学。两个分支通过双向交叉注意力层交换信息——Vis-Attention让视频分支关注物理分支的隐藏状态，Phy-Attention让物理分支关注视频分支的隐藏状态。

输入观测视频 $\mathbf{x}^o$ 被编码到两个互补的潜在空间：(1) 通过视频VAE编码器获得视觉潜在序列 $\mathbf{v}^o$；(2) 通过V-JEPA2获得物理潜在序列 $\mathbf{z}^o$。模型条件化于观测帧和物理状态，联合预测未来视频帧和对应的物理动力学。

### 关键设计

1. **物理感知潜在表示（V-JEPA2嵌入）**:

    - 功能：为视频生成模型提供物理状态的抽象表示
    - 核心思路：利用V-JEPA2（自监督视频编码器）提取的表示作为潜在物理状态。V-JEPA2的表示已被证明能编码物体恒存、碰撞、重力等直觉物理概念。这使得模型无需显式指定物理属性/模拟器/外部推理，而是在一个学到的抽象物理空间中推理
    - 设计动机：与使用显式物理模拟器相比，潜在物理表示不受模拟器假设限制，能覆盖更多样的物理现象；与表示对齐方法相比，Phantom显式建模物理状态的时间演化而非仅做静态对齐

2. **双向交叉注意力耦合**:

    - 功能：在视频和物理分支之间动态交换信息
    - 核心思路：在两个分支对应深度插入交叉注意力层。Vis-Attention以视频隐藏状态为query、物理隐藏状态为key/value：$\mathbf{h}'_v = \text{Softmax}(\frac{\mathbf{W}^Q_v\mathbf{h}_v \cdot (\mathbf{W}^K_v\mathbf{h}_z)^T}{\sqrt{d}}) \mathbf{W}^V_v\mathbf{h}_z$；Phy-Attention对称处理。这让物理线索引导视觉生成，同时视觉证据精炼物理推理
    - 设计动机：相比joint-attention将两种模态混合在一起，双交叉注意力提供更细粒度的控制，避免了视觉和物理特征过度纠缠导致的训练不稳定

3. **选择性冻结训练策略**:

    - 功能：在注入物理推理的同时保留预训练视觉生成能力
    - 核心思路：训练时冻结视频分支的所有预训练参数，仅更新物理分支和双交叉注意力层。50%训练实例不提供条件帧（对应text-to-video），另外50%随机采样1-45帧条件帧（对应video-to-video）
    - 设计动机：保护Wan2.2的强生成先验不被物理分支的梯度破坏

4. **递归损失权重调度**:

    - 功能：稳定视觉和物理分支的联合训练
    - 核心思路：联合损失 $\mathcal{L} = \mathcal{L}_v + \alpha_z \mathcal{L}_z$，其中物理损失的梯度范数远大于视觉损失。初始化 $\alpha_z=0$ 并逐渐增加；当物理分支梯度范数超过阈值 $\eta_z$ 时，重置 $\alpha_z=0$ 并重新开始调度。这种循环加权防止物理分支压倒共享架构
    - 设计动机：直接联合训练会因物理损失过大导致不稳定，循环调度让物理分支逐步贡献有意义的梯度

### 损失函数 / 训练策略
基于标准flow-matching目标扩展为联合预测视觉和物理速度场。训练在OpenVidHD-0.4M（约40万高质量视频-文本对）上进行，支持最多121帧、分辨率480×832。采用递归权重调度策略平衡双分支训练。

## 实验关键数据

### 主实验

| 基准 | 指标 | Phantom | Wan2.2-TI2V | 提升 |
|--------|------|------|----------|------|
| VideoPhy | SA | 47.5 | 41.5 | +14.5% |
| VideoPhy | PC | **37.9** | 25.2 | **+50.4%** |
| VideoPhy-2 | SA | 27.75 | 24.53 | +13.1% |
| VideoPhy-2 | PC | 71.74 | 69.20 | +2.6% |
| Physics-IQ (单帧) | Score | **29.59** | 22.10 | **+33.9%** |
| Physics-IQ (多帧) | Score | 27.53 | - | - |

注：在VideoPhy PC上达到所有方法中最高（37.9），超过PhyT2V(37)和WISA(33)等专用物理方法。

### VBench-2综合评估

| 维度 | Phantom | Wan2.2-TI2V | 变化 |
|------|---------|-------------|------|
| Total | 51.84 | 51.57 | +0.5% |
| Physics | 43.61 | 40.19 | +6.0% |
| Human Fidelity | 88.39 | 86.10 | +2.7% |
| Controllability | 20.23 | 18.50 | +9.4% |
| Commonsense | 61.43 | 60.57 | +1.4% |

### Physics-IQ细分指标（单帧）

| 指标 | Phantom | Wan2.2-TI2V | 提升 |
|------|---------|-------------|------|
| Spatial IoU | 0.245 | 0.164 | +49.4% |
| Spatiotemporal IoU | 0.146 | 0.132 | +10.6% |
| Weighted Spatial IoU | 0.140 | 0.102 | +37.3% |
| MSE↓ | 0.009 | 0.010 | +11.1% |

### 关键发现
- **物理一致性大幅提升的同时未牺牲视觉质量**——VBench-2总分持平甚至略高，说明物理推理和视觉生成可以兼得
- Creativity中Diversity有所下降（64.67→45.95），但Composition从40.35提升到45.07，作者认为物理不合理的视频反而可能"膨胀"多样性指标
- 在Physics-IQ单帧设置下Phantom达到29.59，超过所有方法包括CogVideoX-I2V(27.90)和RDPO(25.21)
- Phantom仅用了40万视频训练（非物理特化数据），却显著提升了物理一致性，说明V-JEPA2物理表示加联合建模是有效的

## 亮点与洞察
- **V-JEPA2作为物理先验的巧妙选择**：不需要构建物理模拟器或标注物理参数，直接利用自监督视觉表示中已编码的直觉物理知识。这是一种"免费午餐"——利用现有大模型的物理感知能力来增强另一个模型
- **双分支flow-matching设计**：视觉和物理两个并行ODE过程通过交叉注意力耦合，在保持各自模态特性的同时实现信息交换。这种设计比把物理信息直接拼接到输入中更优雅，且可扩展性好
- **递归损失权重调度**是一个实用trick——当两个学习目标梯度尺度差异很大时，周期性重置权重比固定比例更稳定。可迁移到其他多任务学习场景
- **推理时零额外物理输入**：text-to-video模式下完全从纯噪声联合去噪，说明模型已内化了物理理解

## 局限与展望
- 物理分支从零初始化，训练效率可能不如用现有物理模型初始化
- V-JEPA2的物理感知能力仍然有限，对复杂流体动力学、可变形物体等可能编码不足
- 仅在40万数据上训练，而基线Wan2.2在更大数据上预训练——更大规模训练可能进一步提升
- 递归权重调度需要手动设置阈值 $\eta_z$，对超参数可能敏感
- VBench-2的Diversity下降值得关注，可能限制创意性应用场景

## 相关工作与启发
- **vs PhyT2V/DiffPhy**: 这些方法在推理时用LLM推理来精化提示引导扩散，是外部的、不增加模型内在物理理解、且有推理overhead。Phantom将物理推理内化到生成过程中
- **vs VideoREPA**: VideoREPA通过表示对齐间接注入物理先验，是静态对齐不建模物理状态演化。Phantom显式预测物理动力学的时序演化
- **vs PhysAnimator/PhysGen**: 依赖外部物理模拟器，受限于模拟器的覆盖范围和保真度。Phantom无需模拟器

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双分支联合建模视觉和物理动力学是全新范式，V-JEPA2作为潜在物理表示的选择巧妙
- 实验充分度: ⭐⭐⭐⭐ 覆盖VideoPhy/VideoPhy-2/Physics-IQ/VBench-2四个基准，但缺少消融实验分析各组件贡献
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法阐述系统
- 价值: ⭐⭐⭐⭐⭐ 为物理一致视频生成开辟了新方向，双分支联合建模+自监督物理表示的范式具有广泛影响力

<!-- RELATED:START -->

## 相关论文

- [SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [Physical Simulator In-the-Loop Video Generation](physical_simulator_in-the-loop_video_generation.md)
- [JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](../../ICCV2025/video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)

<!-- RELATED:END -->
