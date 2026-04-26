---
title: >-
  [论文解读] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos
description: >-
  [CVPR 2026][4D世界模型] NeoVerse 提出了一个可扩展的 4D 世界模型，通过前馈式无位姿 4DGS 重建和在线单目退化模拟，使整个训练流程可以利用海量野外单目视频（百万级），在 4D 重建和新轨迹视频生成上均达到 SOTA。
tags:
  - CVPR 2026
  - 4D世界模型
  - 高斯泼溅
  - 单目视频
  - 新视角生成
  - 前馈重建
---

# NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos

**会议**: CVPR 2026  
**arXiv**: [2601.00393](https://arxiv.org/abs/2601.00393)  
**代码**: https://neoverse-4d.github.io (即将开源)  
**领域**: 视频理解 / 3D视觉 / 4D重建  
**关键词**: 4D世界模型, 高斯泼溅, 单目视频, 新视角生成, 前馈重建

## 一句话总结

NeoVerse 提出了一个可扩展的 4D 世界模型，通过前馈式无位姿 4DGS 重建和在线单目退化模拟，使整个训练流程可以利用海量野外单目视频（百万级），在 4D 重建和新轨迹视频生成上均达到 SOTA。

## 研究背景与动机

1. **领域现状**：4D 世界建模（重建 + 生成的混合范式）在自动驾驶、数字内容创作等领域潜力巨大。现有方法通常先重建 3D/4D 表示，再用几何先验指导视频生成模型，以实现时空一致性和精确视角控制。

2. **现有痛点**：当前方案的核心瓶颈在于**可扩展性不足**，表现在两个层面：
    - **数据可扩展性差**：如 ViewCrafter 只能处理静态场景，SynCamMaster/ReCamMaster 依赖昂贵的多视角动态视频，数据获取成本高且限制泛化；
    - **训练可扩展性差**：TrajectoryCrafter、FreeSim 等需要离线预处理（重深度估计、离线重建高斯场），计算开销大、存储消耗高、训练方案不灵活。

3. **核心矛盾**：海量廉价的野外单目视频无法被直接利用，因为缺乏多视角监督信号和高效的在线处理流程。

4. **本文目标** 如何让整个 4D 世界模型的训练 pipeline 完全可扩展到多样化的野外单目视频。

5. **切入角度**：作者观察到，如果能实现 (a) 无需位姿的前馈 4D 重建，(b) 在线高效的退化渲染模拟，就可以将任意单目视频变成训练数据。

6. **核心 idea**：通过前馈式 4DGS 重建 + 在线单目退化模拟，让 4D 世界模型的全流程可扩展到百万级野外单目视频。

## 方法详解

### 整体框架

NeoVerse 分为两个阶段：(1) **重建阶段**：基于 VGGT 构建前馈式无位姿 4DGS 重建模型，输入单目视频输出 4D 高斯场；(2) **生成阶段**：利用 4DGS 在新轨迹下渲染退化图像作为条件，输入视频生成模型（Wan-T2V 14B + 控制分支）生成高质量视频。训练时，生成阶段利用在线重建 + 退化模拟，将原始单目视频本身作为监督目标。

### 关键设计

1. **双向运动建模 (Bidirectional Motion Modeling)**:

    - 功能：为 4D 高斯体赋予前向和后向的线速度与角速度，支持任意时刻的高斯插值
    - 核心思路：将 VGGT 输出的帧特征 $\{F_t\}$ 沿时间维切为两组，分别做交叉注意力编码前向运动特征（$t \to t+1$）和后向运动特征（$t \to t-1$）。利用这些特征预测每个高斯的双向速度 $v^+, v^-$ 和角速度 $\omega^+, \omega^-$，使得可以用线性插值将关键帧的高斯传播到非关键帧
    - 设计动机：与 4DGT 的单向运动不同，双向设计支持高效的稀疏关键帧重建（只输入 K 帧即可渲染 N 帧），大幅降低在线重建的计算开销，同时支持时间控制相关的下游应用

2. **在线单目退化模拟 (Monocular Degradation Simulation)**:

    - 功能：在训练时从单目视频自动生成退化渲染-原始视频配对数据
    - 核心思路：包含三种互补的退化模式：(a) **基于可见性的高斯剔除**——对相机轨迹施加随机变换得到新轨迹，用深度信息剔除被遮挡的高斯，再渲染回原始视角，模拟遮挡退化；(b) **平均几何滤波器**——在新视角深度图上施加均值滤波，再据此调整高斯中心，模拟深度不连续边缘的飞行像素；(c) 对 (b) 使用更大滤波核模拟更宽范围的深度误差失真
    - 设计动机：多视角数据集可以直接获取训练对，但单目视频不行。三种退化模式基于几何关系和深度学习的第一性原理设计，简单有效，使任意单目视频都可作为训练数据

3. **稀疏关键帧在线重建 (Efficient On-the-fly Reconstruction)**:

    - 功能：避免对长视频逐帧进行前馈推理，提高训练效率
    - 核心思路：给定 N 帧长视频仅取 K 个关键帧做前馈重建，利用双向运动机制将高斯插值到其余帧，再高效渲染。通过时变透明度衰减函数 $\alpha_i(t_q) = \alpha_i \exp(-\gamma \cdot d(t_q, t)^{1/(1-\tau_i)})$ 实现帧间自然过渡
    - 设计动机：前馈网络推理仍是训练瓶颈，而渲染极其高效；用 11 个关键帧重建 81 帧视频，重建仅需 2 秒

### 损失函数 / 训练策略

- **重建损失**：$\mathcal{L}_{recon} = \mathcal{L}_{rgb} + \lambda_1\mathcal{L}_{camera} + \lambda_2\mathcal{L}_{depth} + \lambda_3\mathcal{L}_{motion} + \lambda_4\mathcal{L}_{regular}$，包括光度损失（L2 + LPIPS）、相机参数损失、深度损失、双向速度监督、透明度正则化
- **生成损失**：采用 Rectified Flow，基于 Wan-T2V 14B，训练控制分支，冻结生成模型主体（兼容蒸馏 LoRA 加速）
- **两阶段训练**：第一阶段 150K iterations 训重建模型，第二阶段 50K iterations 训生成模型，使用 32 张 A800 GPU
- **全局运动追踪**：推理时通过跨帧可见性加权最大速度分离动态/静态高斯，对两类高斯采用不同的时间聚合策略

## 实验关键数据

### 主实验

| 数据集 | 指标 | NeoVerse | AnySplat | NoPoSplat |
|--------|------|----------|----------|-----------|
| VRNeRF (静态) | PSNR↑ | **20.73** | 18.02 | 11.27 |
| VRNeRF (静态) | LPIPS↓ | **0.352** | 0.366 | 0.620 |
| Scannet++ (静态) | PSNR↑ | **25.34** | 22.79 | 8.69 |
| ADT (动态) | PSNR↑ | **32.56** | - | - |
| DyCheck (动态) | PSNR↑ | **11.56** | - | 9.32 |

| 方法 | 推理总时间(s) | Subj. Consist. | Back. Consist. | Imag. Quality |
|------|-------------|----------------|----------------|---------------|
| TrajectoryCrafter | 146 | 83.02 | 88.58 | 54.59 |
| ReCamMaster | 168 | 88.21 | 91.60 | 58.87 |
| NeoVerse (11 key) | **20** | 88.43 | 92.27 | 59.75 |
| NeoVerse (21 key) | **21** | **88.73** | **92.43** | 60.01 |

### 消融实验

| 配置 | DyCheck PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------|-------|--------|
| w/o 双向运动 | 11.27 | 0.285 | 0.570 |
| w/o 透明度正则 | 10.86 | 0.244 | 0.576 |
| 完整重建模型 | 11.56 | 0.293 | 0.558 |
| 完整 pipeline (+ 生成) | **14.59** | **0.323** | **0.501** |

### 关键发现

- 双向运动建模贡献显著，去掉后 DyCheck PSNR 下降 0.29
- 生成阶段对最终质量提升巨大（PSNR 从 11.56 → 14.59），验证了重建-生成混合范式的有效性
- 稀疏关键帧（11 帧 vs 全 81 帧）对生成质量影响很小，但推理时间从 28s 降至 20s（7x 快于 TrajectoryCrafter）
- 在 VBench 评估中，NeoVerse 在主观一致性、背景一致性、图像质量上全面超越 TrajectoryCrafter 和 ReCamMaster

## 亮点与洞察

- **核心洞察**：4D 世界模型的瓶颈不是模型架构而是数据和训练的可扩展性。通过在线退化模拟巧妙地将单目视频变成多视角训练对，避免了对昂贵多视角数据的依赖
- **稀疏关键帧重建**的设计非常巧妙——利用高斯渲染远快于网络推理的特性，将前馈推理成本降低数倍而几乎不影响质量
- **控制分支冻结生成模型**的策略使得 NeoVerse 可直接兼容蒸馏 LoRA，推理时生成仅需 18 秒
- 退化模拟基于第一性原理（几何遮挡、深度均值化），无需学习额外的噪声模型

## 局限与展望

- 当前分辨率固定在 336×560，离实际应用的高分辨率需求有距离
- 全局运动追踪基于阈值分离动态/静态，对于缓慢运动物体可能不够精确
- 退化模拟虽然基于第一性原理但仍是近似，真实新视角渲染的退化模式可能更复杂
- 训练需要 32 张 A800 GPU，对于学术实验室来说成本较高

## 相关工作与启发

- **vs TrajectoryCrafter**：同为重建-生成混合方法，但 TrajectoryCrafter 依赖离线预处理，限制数据规模；NeoVerse 全在线流程，可扩展到百万级视频。NeoVerse 推理速度快 7 倍
- **vs ReCamMaster**：纯生成方法视觉质量好但轨迹控制不精确；NeoVerse 兼具生成质量和精确轨迹控制
- **vs AnySplat**：AnySplat 针对静态场景无位姿重建；NeoVerse 扩展到 4D 动态场景，PSNR 高 2.7dB
- 该工作为利用海量互联网视频训练世界模型提供了一条可行路径

## 评分

- 新颖性: ⭐⭐⭐⭐ 双向运动建模 + 在线退化模拟的组合巧妙，但各个模块单独看并非完全创新
- 实验充分度: ⭐⭐⭐⭐⭐ 静态/动态重建、生成质量、推理效率、消融实验全面覆盖
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation 论述充分，符号系统一致
- 价值: ⭐⭐⭐⭐⭐ 解决了 4D 世界模型训练的数据瓶颈问题，具有很强的实际影响力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[CVPR 2026\] The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)

<!-- RELATED:END -->
