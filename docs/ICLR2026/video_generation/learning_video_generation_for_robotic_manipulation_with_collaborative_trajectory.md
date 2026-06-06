---
title: >-
  [论文解读] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control
description: >-
  [ICLR 2026][视频生成][机器人操作] 提出RoboMaster框架，通过协作轨迹（collaborative trajectory）将机器人-物体交互过程分解为前交互、交互中、后交互三阶段，结合外观和形状感知的物体嵌入，实现高质量的机器人操作视频生成。
tags:
  - "ICLR 2026"
  - "视频生成"
  - "机器人操作"
  - "协作轨迹"
  - "扩散模型"
  - "交互建模"
---

# Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control

**会议**: ICLR 2026  
**arXiv**: [2506.01943](https://arxiv.org/abs/2506.01943)  
**代码**: [项目页面](https://fuxiao0719.github.io/projects/robomaster/)  
**领域**: 视频生成  
**关键词**: 视频生成, 机器人操作, 协作轨迹, 扩散模型, 交互建模

## 一句话总结

提出RoboMaster框架，通过协作轨迹（collaborative trajectory）将机器人-物体交互过程分解为前交互、交互中、后交互三阶段，结合外观和形状感知的物体嵌入，实现高质量的机器人操作视频生成。

## 研究背景与动机

1. **领域现状**: 视频扩散模型在生成机器人决策数据方面展现出巨大潜力，轨迹条件控制能够实现对机器人运动的细粒度控制。

2. **现有痛点**: 现有轨迹控制方法（如Tora、DragAnything）主要关注单个物体的独立运动，使用分离的轨迹分别控制机械臂和被操作物体。在交互区域（重叠区域）会导致特征纠缠，生成质量下降。

3. **核心矛盾**: 机器人操作本质上是多物体交互过程，但现有方法将其简化为独立运动控制，无法捕获物理上合理的交互。如果合成视频无法准确表示交互阶段，逆动力学模型将提取不可靠的动作标签。

4. **本文目标**: 设计一种能准确建模机器人-物体交互动力学的视频生成框架，使生成的视频可作为高质量的机器人学习演示数据。

5. **切入角度**: 不分解物体，而是分解交互过程——将操作过程分为三个子阶段，每个阶段由主导主体引导，统一为单一的协作轨迹。

6. **核心 idea**: 通过分解交互过程而非分解物体，将多物体轨迹统一为协作轨迹表示，从根本上避免重叠区域的特征纠缠。

## 方法详解

### 整体框架

RoboMaster基于预训练CogVideoX-5B架构。给定初始帧 $\mathbf{I}$、文本提示 $\mathbf{c}$、物体mask $\mathbf{M}_d, \mathbf{M}_s$ 和协作轨迹 $\mathcal{C}$，生成操作视频 $\mathbf{X}$。流程为：(1) 通过外观和形状感知嵌入编码物体表示 $\mathbf{v}_d, \mathbf{v}_s$；(2) 将轨迹分解为三个子阶段并关联对应物体特征；(3) 通过运动注入模块将协作轨迹嵌入注入DiT块。

### 关键设计

**1. 耦合外观-形状物体嵌入（Coupled Appearance and Shape Embedding）**

- **功能**: 在视频序列中保持物体的语义一致性
- **核心思路**: 将初始帧通过VAE编码器投影为latent特征 $\mathbf{z}$，下采样物体mask后提取被mask的latent特征并池化得到 $\tilde{\mathbf{v}}$。在每个时间步以轨迹点为中心、mask面积比例为半径构建圆形体积表示 $\mathbf{v} \in \mathbb{R}^{c \times h \times w}$
- **设计动机**: 相比Tora等方法的点表示，mask-based表示同时编码了物体的外观和空间形状信息，加速训练收敛并提升跨帧身份一致性

**2. 协作轨迹表示（Collaborative Trajectory Representation）**

- **功能**: 统一建模多物体交互动力学，避免特征纠缠
- **核心思路**: 将轨迹分解为三个时间阶段：前交互 $\mathcal{C}_1$（机械臂主导）、交互中 $\mathcal{C}_2$（被操作物体主导）、后交互 $\mathcal{C}_3$（机械臂主导）。利用因果表示将前一时间步的latent传播到后续帧。最终分布分解为三个物体感知子分布的乘积
- **设计动机**: 交互阶段被操作物体的运动隐式引导机械臂轨迹（两者相对动力学受约束）；特征表示的时间变化 $\mathbf{v}_d \rightarrow \mathbf{v}_s \rightarrow \mathbf{v}_d$ 为建模行为转变提供线索

**3. 运动注入模块（Motion Injection Module）**

- **功能**: 将协作轨迹信息注入视频DiT生成过程
- **核心思路**: 协作轨迹latent $\mathbf{V} \in \mathbb{R}^{f \times c \times h \times w}$ 经patchify后通过零初始化的2D空间卷积和1D时间卷积编码，然后与DiT块的隐藏状态相加：$\mathbf{h} = \mathbf{h} + \text{norm}(\tilde{\mathbf{V}}) + \tilde{\mathbf{V}}$
- **设计动机**: 零初始化确保训练初期不破坏预训练模型的生成能力，plug-and-play设计便于集成

### 损失函数 / 训练策略

标准扩散模型去噪损失：$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}[\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{c}, \mathbf{M}_d, \mathbf{M}_s, \mathcal{C}, t)\|_2^2]$

训练设置：8块A800 GPU, AdamW优化器, DiT学习率 $2 \times 10^{-5}$, 运动注入器学习率 $1 \times 10^{-4}$, batch size 16, 训练30K步。推理使用50步DDIM, CFG scale 6.0。

## 实验关键数据

### 主实验

视频生成质量和轨迹精度对比（Bridge数据集，所有基线在相同数据上重新训练）：

| 方法 | FVD↓ | PSNR↑ | SSIM↑ | TrajError_robot↓ | TrajError_obj↓ |
|------|------|-------|-------|------------------|----------------|
| TesserAct | 261.84 | 18.99 | 0.778 | 37.34 | 54.64 |
| IRASim | 159.04 | 20.88 | 0.782 | 19.25 | 34.39 |
| DragAnything | 158.42 | 21.13 | 0.792 | 18.97 | 27.41 |
| Tora | 152.28 | 21.24 | 0.788 | 18.14 | 26.43 |
| **RoboMaster** | **147.31** | **21.55** | **0.803** | **16.47** | **24.16** |

机器人动作规划成功率（RLBench + SIMPLER，各100次试验平均成功率）：

| 方法 | pick up cup | put knife | open microwave | close box | pick coke can |
|------|-------------|-----------|----------------|-----------|---------------|
| OpenVLA | 0.55 | 0.46 | 0.35 | 0.45 | 0.59 |
| Tora | 0.79 | 0.82 | 0.61 | 0.72 | 0.89 |
| RoboMaster | **0.83** | 0.76 | 0.54 | **0.79** | **0.91** |

### 消融实验

| 配置 | FVD↓ | PSNR↑ | TrajError_obj↓ | 说明 |
|------|------|-------|----------------|------|
| 去除因果嵌入 | 151.62 | 21.30 | 27.15 | 时序连贯性下降 |
| 点表示替代mask | 157.49 | 20.87 | 31.41 | 物体身份一致性大幅下降 |
| 分离轨迹 | 152.01 | 21.08 | 25.84 | 交互区域特征纠缠 |
| 交叉注意力注入 | 163.56 | 19.38 | 29.16 | 效果不如additive注入 |
| **完整模型** | **147.31** | **21.55** | **24.16** | 所有组件协同最优 |

### 关键发现

- 协作轨迹设计在10个机器人任务中8个优于Tora，证明交互精确建模对下游策略学习的重要性
- Mask表示在90%稀疏度下仍保持99.81% PSNR，对粗糙用户输入具有很强的鲁棒性
- 即使40%的提示被替换为不精确描述，模型仍保持96%以上的PSNR性能
- 用户研究中45.16%的偏好率远超所有基线（第二名Tora仅17.74%）

## 亮点与洞察

- **"分解交互而非分解物体"** 的核心思路简洁而有效，从根本上解决了特征纠缠问题
- 协作轨迹设计同时简化了用户交互——用户只需定义分段轨迹而非完整的多物体轨迹
- 圆形体积表示兼顾外观和形状信息，是一种优雅的设计
- 下游机器人规划实验验证了视频生成质量与策略学习效果的正相关性

## 局限与展望

- 纯2D像素空间操作，集成深度线索可能实现更精确的3D控制
- 对域外输入可能产生不完整或扭曲的物体
- 对不同机器人形态的泛化仍需扩展训练数据
- 协作轨迹需预先知道交互阶段的时间分段点，自动检测可能更实用

## 相关工作与启发

- 与Tora等分离轨迹方法形成鲜明对比，启发了"交互建模"视角在视频生成中的重要性
- 与TesserAct的4D方法相比，2D方法在数据效率上更有优势
- 视频生成作为世界模拟器的范式在机器人学习中展现出巨大潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ 协作轨迹的交互分解思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 包含视频质量、轨迹精度、机器人规划、消融和用户研究
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示精美
- 价值: ⭐⭐⭐⭐ 对机器人视频生成和数据增强具有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](../../CVPR2025/video_generation/posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_fewstep_controllable_video_generation.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICML 2026\] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](../../ICML2026/video_generation/epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)

</div>

<!-- RELATED:END -->
