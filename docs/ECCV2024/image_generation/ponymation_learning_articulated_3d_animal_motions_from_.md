---
title: >-
  [论文解读] Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos
description: >-
  [ECCV 2024][图像生成][3D动物运动] 本文提出一种从无标注互联网视频中学习关节式3D动物运动生成模型的方法，通过视频Photo-Geometric自编码框架将视频分解为静态形状、外观和运动隐编码，无需任何姿态标注或参数化形状模型即可在推理时从单张图像生成多样的4D动画。 领域现状：3D人体运动合成借助SMPL等…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "3D动物运动"
  - "无标注学习"
  - "视频自编码"
  - "运动VAE"
  - "可微渲染"
---

# Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos

**会议**: ECCV 2024  
**arXiv**: [2312.13604](https://arxiv.org/abs/2312.13604)  
**代码**: [https://keqiangsun.github.io/projects/ponymation](https://keqiangsun.github.io/projects/ponymation)  
**领域**: 图像生成  
**关键词**: 3D动物运动, 无标注学习, 视频自编码, 运动VAE, 可微渲染

## 一句话总结
本文提出一种从无标注互联网视频中学习关节式3D动物运动生成模型的方法，通过视频Photo-Geometric自编码框架将视频分解为静态形状、外观和运动隐编码，无需任何姿态标注或参数化形状模型即可在推理时从单张图像生成多样的4D动画。

## 研究背景与动机

**领域现状**：3D人体运动合成借助SMPL等参数模型和大规模MoCap数据取得了巨大进展，但动物运动的3D建模严重缺乏数据——没有大规模3D扫描、参数形状模型或运动捕捉数据。现有方法要么需要多视角视频、要么需要2D关键点标注。

**现有痛点**：MagicPony等方法可从单视角图像集合学习3D动物模型，但只处理静态图像，忽略了动态运动信息。LASR等方法需要繁重的优化流程且不建模运动分布。BKinD做自监督关键点发现但限于2D表示。通用4D生成模型（如基于扩散的方法）在运动质量和多样性上仍然有限。

**核心矛盾**：要学习3D动物运动分布，需要将非结构化的互联网视频（每个视频都是不同个体、不同运动、不同视角）配准到统一的3D规范模型上，同时分离出形状、外观和运动——但没有任何外部标注可用。

**本文目标**：仅从原始互联网视频学习动物类别的3D关节运动生成模型，支持从潜空间采样新运动和从单张图像自动生成4D动画。

**切入角度**：利用DINO-ViT的自监督特征提供跨实例语义对应，结合已知的粗略骨架（如"四足"），在视频重渲染目标下端到端学习运动VAE。

**核心 idea**：设计视频Photo-Geometric自编码框架，将视频分解为rest-pose 3D mesh + 纹理 + 关节姿态序列，用可微渲染重建原始视频帧来训练，核心是运动VAE——空间-时间Transformer编码器将视频编码为运动隐码，解码器解码为骨骼旋转序列。

## 方法详解

### 整体框架
输入：动物类别的互联网视频集合。训练流程：(1) DINO-ViT提取每帧特征；(2) 空间Transformer将每帧的骨骼特征聚合为姿态表示，时间Transformer将帧序列编码为运动VAE参数 $(\hat{\mu}, \hat{\Sigma})$；(3) 采样 $z$，经时间-空间解码器生成关节姿态序列；(4) Linear Blend Skinning驱动mesh，可微渲染重建RGB帧和mask。推理时采样 $z$ 即可生成新运动。

### 关键设计

1. **空间-时间Transformer运动编码器**:

    - 功能：从视频中提取关节运动信息编码为VAE隐空间
    - 核心思路：对每帧构建骨骼特征描述子 $\nu_{t,b}$（包含全局DINO特征、局部key token特征、骨骼3D位置、图像投影位置），空间Transformer $E_s$ 将同一帧的所有骨骼特征聚合为单一姿态特征 $\nu_{t,*}$，时间Transformer $E_t$ 将帧序列映射为VAE分布参数。解码器对称设计：时间解码器将 $z$ 解码为帧序列特征，空间解码器将每帧特征解码为各骨骼的旋转
    - 设计动机：空间维度捕获跨骨骼的姿态关系，时间维度捕获运动的时序模式，两层分离使编码更高效

2. **视频Photo-Geometric自编码**:

    - 功能：无需姿态标注，仅用视频重建目标训练整个系统
    - 核心思路：将视频分解为共享的base mesh（SDF MLP + DMTet）、实例形变（条件MLP）、纹理（MLP）和运动序列（VAE），通过LBS驱动mesh后可微渲染重建RGB帧和mask。损失包括mask重建（L2 + distance transform）、RGB重建（L1）、DINO特征重建、时间平滑正则和形状正则
    - 设计动机：这是一种"分析-综合"策略——通过要求系统能从学到的表示重新合成输入视频，迫使系统学会有意义的3D分解，完全绕过标注需求

3. **DINO语义对应驱动的跨实例学习**:

    - 功能：在无标注情况下建立不同动物个体间的对应关系
    - 核心思路：DINO-ViT的PCA特征天然包含跨实例的部件级语义对应。在规范空间学习一个特征场MLP $\psi(\mathbf{x})$，要求渲染出的特征图与DINO特征图匹配，从而在3D层面建立跨实例对应
    - 设计动机：这是将所有不同个体的视频配准到统一3D模型的关键——否则每个视频只能学到自己的几何，无法建模shared motion分布

### 损失函数 / 训练策略
总损失 = mask loss（L2 + distance transform）+ RGB L1 loss + DINO特征匹配loss + VAE KL散度 + 时间平滑正则 + 形状正则（Eikonal + 关节旋转幅度 + 形变幅度）。使用多假设视角预测机制处理视角歧义。

## 实验关键数据

### 主实验

| 方法 | 3D重建质量 | 运动多样性 | 运动质量(FID) | 速度 |
|---|---|---|---|---|
| MagicPony | 仅静态 | 无运动建模 | - | 快 |
| LASR | 视频级优化 | 无生成能力 | - | 极慢 |
| 4D扩散模型 | 一般 | 有限 | 较差 | 慢 |
| **Ponymation** | **好** | **多样** | **最优** | **秒级推理** |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| 无DINO特征 | 形状和运动质量下降 | 跨实例对应缺失 |
| 无时间Transformer | 运动不连贯 | 帧间信息无法交互 |
| 无VAE（直接回归） | 运动多样性下降 | 无法采样新运动 |
| 完整模型 | 最优 | 所有组件互补 |

### 关键发现
- 仅从互联网视频即可学到合理的3D动物运动分布（如马的奔跑、行走、站立等）
- DINO特征对于跨实例姿态配准至关重要——没有它模型完全无法学习共享运动空间
- 推理时单图像到4D动画只需几秒，远快于优化式方法（LASR需数小时）
- 生成的运动在视觉质量和多样性上超越当时的4D生成方法

## 亮点与洞察
- **完全无标注的3D运动学习**：不需要MoCap数据、参数模型、关键点标注，仅从YouTube视频学习。这极大降低了3D动物运动建模的门槛
- **空间-时间分离的Transformer VAE**：将复杂的4D问题分解为空间（骨骼间）和时间（帧间）两个维度分别处理，设计优雅高效
- **DINO特征的妙用**：将预训练视觉特征的语义对应能力用于3D配准，避免了手工标注的需要

## 局限与展望
- 假设已知粗略骨架结构（如"四足"），不能处理完全未知的动物类别
- 形状建模较粗糙，高频几何细节（如毛发）难以恢复
- 受限于DINO特征的质量——在外观差距大的个体间对应可能不准确
- 可扩展到更多动物类别，或结合扩散模型实现更高质量的运动生成

## 相关工作与启发
- **vs MagicPony**: 同样利用DINO特征和SDF mesh，但MagicPony仅处理静态图像，Ponymation扩展到视频运动建模
- **vs MotionDiffuse等人体运动方法**: 人体方法依赖SMPL模型和大规模MoCap数据，Ponymation证明可从原始视频绕过这些需求
- 视频Photo-Geometric自编码的思路可推广到其他非刚体物体的运动学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从无标注视频学习3D动物运动生成模型
- 实验充分度: ⭐⭐⭐⭐ 多种消融和对比，但定量评估指标受限于缺乏GT
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐⭐ 开创性工作，为非人类3D运动建模开辟新路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos](../../CVPR2025/image_generation/videoworld_exploring_knowledge_learning_from_unlabeled_videos.md)
- [\[ECCV 2024\] Generating Human Interaction Motions in Scenes with Text Control](generating_human_interaction_motions_in_scenes_with_text_control.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
