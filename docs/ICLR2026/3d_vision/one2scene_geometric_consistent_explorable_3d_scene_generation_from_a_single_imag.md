---
title: >-
  [论文解读] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image
description: >-
  [ICLR 2026][3D视觉][单图3D场景生成] 提出One2Scene——将单图到可探索3D场景的病态问题分解为三个子任务：(1)全景图生成扩展视觉覆盖 (2)前馈3DGS网络从稀疏锚点视角构建显式3D几何scaffold (3)scaffold引导的新视角合成，通过Dual-LoRA融合高质量锚点视角和几何先验，在大视角变化下实现几何一致且逼真的场景生成，显著超越SOTA。
tags:
  - ICLR 2026
  - 3D视觉
  - 单图3D场景生成
  - 全景深度估计
  - 3D Gaussian Splatting
  - 几何scaffold
  - 新视角合成
---

# One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

**会议**: ICLR 2026  
**arXiv**: [2602.19766](https://arxiv.org/abs/2602.19766)  
**代码**: [项目页面](https://one2scene5406.github.io/)  
**领域**: 3D视觉/场景生成  
**关键词**: 单图3D场景生成, 全景深度估计, 3D Gaussian Splatting, 几何scaffold, 新视角合成

## 一句话总结
提出One2Scene——将单图到可探索3D场景的病态问题分解为三个子任务：(1)全景图生成扩展视觉覆盖 (2)前馈3DGS网络从稀疏锚点视角构建显式3D几何scaffold (3)scaffold引导的新视角合成，通过Dual-LoRA融合高质量锚点视角和几何先验，在大视角变化下实现几何一致且逼真的场景生成，显著超越SOTA。

## 研究背景与动机

**领域现状**：从单张图像生成可探索3D场景是3D视觉的核心挑战。重建方法(NeRF/3DGS)需要大量图像，稀疏视角方法无法外推。生成式方法包括：视频扩散模型(ReconX/ViewCrafter)、全景管线(DreamScene360/DreamCube)、导航+修复(WonderJourney/Pano2Room)。

**现有痛点**：(1) 视频扩散方法缺乏持久3D表示，长序列几何累积误差导致崩溃；(2) 全景方法只从单点观测，缺乏显式3D信息，大视角变化时严重畸变；(3) 迭代导航方法累积误差导致全局语义漂移和拉伸几何。

**核心矛盾**：单图信息极度匮乏 vs 需要全局一致的3D场景。现有方法要么缺乏全局覆盖(单视角方法)，要么缺乏几何约束(生成式方法)，要么累积误差(迭代方法)。

**本文目标**：(a) 如何从单图获得全局视觉覆盖？(b) 如何建立显式3D几何约束？(c) 如何在大视角变化下保持几何一致性和视觉质量？

**切入角度**：将问题分解为三个更容易的子问题——先用全景生成扩展2D覆盖，再用多视角立体匹配建立3D scaffold，最后用scaffold先验约束新视角合成。关键洞察是把单目全景深度估计重新formulate为多视角立体匹配问题，从而利用大规模多视角数据集学到的强几何先验。

**核心 idea**：通过显式3D几何scaffold为单图场景生成提供稳定的全局几何和外观先验，从根本上避免了累积误差和尺度歧义。

## 方法详解

### 整体框架
输入：单张图像。输出：可从任意视角探索的3D场景(高质量新视角图像)。

三个阶段：
- **Stage 1 (全景生成)**：单图 -> cubemap全景 -> 6个锚点视角
- **Stage 2 (3D Scaffold构建)**：6个稀疏锚点视角 -> 前馈3DGS网络 -> 显式3D几何scaffold (0.5秒)
- **Stage 3 (Scaffold引导合成)**：scaffold渲染的粗糙视角 + 高质量锚点视角 -> 扩散模型 -> 逼真新视角

### 关键设计

1. **全景锚点视角生成**：

    - 功能：把单图扩展为360度全景，然后投影为6个cubemap锚点视角
    - 核心思路：使用Hunyuan-Pano-DiT生成全景图，然后投影为6个透视cubemap视角(FoV=95度，相邻视角有2.5度重叠)
    - 设计动机：全景图提供全局语义覆盖，cubemap投影允许利用透视图像的多视角立体匹配先验，比直接处理全景图(等距投影畸变)更稳健

2. **前馈3DGS几何Scaffold (双向融合)**：

    - 功能：从6个稀疏锚点视角前馈预测3D Gaussian参数，构建显式3D scaffold
    - 核心思路：基于VGGT backbone。将全景深度估计重新formulate为多视角立体匹配——6个cubemap视角作为"多视角"输入。关键创新是双向融合模块(Bidirectional Fusion)：6个视角特征$F_i$ -> Cube-to-Equirectangular (C2E)投影到统一等距空间 -> 卷积融合 -> E2C变换回cubemap空间 -> 残差连接：$F_i' = F_i + E2C(H_c(C2E(\{F_i\})))$
    - 设计动机：6个cubemap视角重叠极少(仅2.5度)，现有多视角模型(VGGT)在如此稀疏overlap下性能严重下降。双向融合通过等距空间中间表示强制跨视角一致性，同时残差连接保留视角特有细节。Gaussian中心通过深度反投影计算：$\mu = K^{-1}ud + \Delta$

3. **Scaffold引导新视角合成 (Dual-LoRA)**：

    - 功能：利用scaffold先验从任意视角生成逼真图像
    - 核心思路：基于SEVA架构。Scaffold渲染的视角含丰富几何但有伪影/空洞，锚点视角质量高但缺几何信息——两种异质条件。Dual-LoRA策略：两个独立LoRA模块分别处理锚点视角和scaffold渲染视角，然后通过3D attention机制融合到noisy latent中
    - 设计动机：naive的channel-wise拼接无法有效区分和利用两种异质条件信号。Dual-LoRA让模型分别学习从高质量外观和粗糙几何中提取有用信息。Memory condition(从memory bank选最近生成帧)进一步保证长序列时序一致性

### 损失函数 / 训练策略
- Stage 2 (3DGS): 复合loss = MSE渲染loss + LPIPS感知loss + SILog深度loss。在Structured3D/Deep360/Matterport3D/Stanford2D3D上训练80K iterations
- Stage 3 (合成): 基于SEVA，Adam优化器，lr=1.25e-5，batch=16，40K iterations。训练数据从DL3DV和RealEstate10K用MVSplat稀疏重建获得，刻意模拟稀疏输入的伪影

## 实验关键数据

### 主实验：可探索3D场景生成 (WorldScore benchmark变体)

| 方法 | NIQE↓ | Q-Align↑ | CLIP-I↑ | CamMC↓ | RotErr↓ |
|------|-------|---------|---------|--------|---------|
| DreamScene360 | 8.40 | 1.91 | 74.24 | - | - |
| WonderJourney | 4.97 | 3.02 | 77.92 | - | - |
| SEVA | 4.53 | 3.20 | 87.82 | 0.558 | 0.165 |
| VMem | 6.86 | 2.95 | 75.80 | 0.998 | 0.569 |
| **One2Scene** | **4.43** | **4.13** | **89.95** | **0.389** | **0.107** |

### 消融实验：Scaffold质量对最终生成的影响

| 配置 | NIQE↓ | Q-Align↑ | CLIP-I↑ | CamMC↓ |
|------|-------|---------|---------|--------|
| 替换为AnySplat | 4.96 | 3.61 | 81.96 | 0.616 |
| **Ours (完整)** | **4.43** | **4.13** | **89.95** | **0.389** |

### 关键发现
- **Scaffold质量决定性**：用AnySplat替换本文scaffold后，CLIP-I从89.95降到81.96，CamMC从0.389升到0.616，证明高质量scaffold是核心
- **深度估计领先**：Matterport3D上finetuned AbsRel 0.0391 vs 之前SOTA 0.0850，提升>50%；Stanford2D3D上zero-shot AbsRel 0.0675已超越所有先前方法
- **效率优势**：6个稀疏视角仅0.5秒(H20)重建scaffold，比AnySplat(20视角2.8秒)快5.6倍
- **解决尺度歧义**：SEVA因缺乏3D约束存在严重尺度歧义(相机穿墙)，One2Scene的scaffold提供了稳定的全局尺度约束

## 亮点与洞察
- **全景深度估计 -> 多视角立体匹配的reformulation**非常巧妙：将全景图投影为cubemap后就能利用大量多视角数据训练的模型，避免了全景深度数据稀缺的问题。这个思路可以迁移到任何全景理解任务
- **双向融合模块(C2E-E2C)**：在等距空间做全局融合再投射回透视空间，优雅解决了极稀疏overlap下的跨视角一致性，是全景处理的通用方案
- **Dual-LoRA处理异质条件**：面对质量好但缺几何vs几何好但有伪影的两种条件，用独立LoRA分别编码后融合，比直接拼接效果好得多。可迁移到任何需要融合不同质量/类型条件的生成任务
- **三阶段分解的系统思维**：把一个不可解的问题拆成三个可解的子问题，每个阶段的输出为下一阶段提供越来越强的约束

## 局限与展望
- 生成视角间仍可能存在微妙不一致(可用post-reconstruction进一步优化)
- 全景生成模型的质量直接影响后续所有阶段——如果全景生成失败则无法恢复
- 训练数据构造依赖MVSplat的稀疏重建质量来模拟伪影，可能无法涵盖所有实际情况
- 目前仅处理静态场景，动态场景支持是未来方向

## 相关工作与启发
- **vs SEVA**: SEVA直接从单图做相机控制的新视角合成，缺乏持久3D表示导致尺度歧义和几何不一致。One2Scene通过显式scaffold提供全局约束
- **vs VMem**: VMem用CUT3R做在线重建维持一致性，但低质量生成帧反过来破坏重建——恶性循环。One2Scene的scaffold预先建立避免了这个问题
- **vs Pano2Room**: Pano2Room通过迭代导航+修复建场景，有强室内先验限制泛化。One2Scene是前馈式无场景类型限制

## 评分
- 新颖性: ⭐⭐⭐⭐ 三阶段分解和多视角reformulation有创新，但各组件是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 多维度评估全面，消融充分，深度估计benchmark结果强
- 写作质量: ⭐⭐⭐⭐⭐ 问题分解清晰，motivation链条逻辑流畅
- 价值: ⭐⭐⭐⭐ 对单图3D场景生成有重要推进，三阶段范式可能成为标准流程

<!-- RELATED:START -->

## 相关论文

- [SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)
- [Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)

<!-- RELATED:END -->
