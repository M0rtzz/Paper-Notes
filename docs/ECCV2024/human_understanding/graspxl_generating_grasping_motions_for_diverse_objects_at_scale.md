---
title: >-
  [论文解读] GraspXL: Generating Grasping Motions for Diverse Objects at Scale
description: >-
  [ECCV 2024][人体理解][抓取动作生成] 提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。
tags:
  - ECCV 2024
  - 人体理解
  - 抓取动作生成
  - 灵巧手操作
  - 强化学习
  - 大规模泛化
  - 多目标控制
---

# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

**会议**: ECCV 2024  
**arXiv**: [2403.19649](https://arxiv.org/abs/2403.19649)  
**代码**: [有](https://eth-ait.github.io/graspxl/)  
**领域**: 人体理解  
**关键词**: 抓取动作生成, 灵巧手操作, 强化学习, 大规模泛化, 多目标控制

## 一句话总结

提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。

## 研究背景与动机

人手具有非凡的灵巧性——能够抓取任意形状的物体，精确到特定部位，从特定方向接近，且无需针对每个物体的专门训练。复现这种能力是动画制作和机器人抓取领域的关键挑战。

现有抓取动作合成方法存在**三大瓶颈**：

**依赖昂贵的3D手-物交互数据**：数据驱动方法（如D-Grasp）需要精确的3D标注序列作为训练数据，采集成本极高，且模型受训练分布限制，难以泛化

**泛化能力受限**：多数方法只能处理训练时见过的物体（DexVIP评估0个新物体），或仅支持少量测试物体（UniDexGrasp约100个），远无法满足实际应用需求

**缺乏多目标控制**：现有方法通常只支持单一目标（如仅控制朝向），无法同时满足抓取区域、朝向、手腕旋转和位置等多个运动目标

核心挑战在于：如何在**无需3D手-物交互数据**的前提下，让一个策略学会为**任意形状物体**生成**满足多运动目标**的稳定抓取动作？难点包括：(1) 物体形状千变万化，需要通用的形状感知机制；(2) 多目标可能互相冲突（满足目标导致接触引起物体移动，破坏抓取稳定性）；(3) 不同灵巧手的关节结构差异巨大。

## 方法详解

### 整体框架

GraspXL 在强化学习框架中运作：给定物体和手模型，策略网络 $\boldsymbol{\pi}$ 接收运动目标 $\mathcal{T}$ 和物理状态 $\mathbf{s}$ 作为输入，通过特征提取层 $\Phi$ 转换后，输出PD控制目标作为动作 $\mathbf{a}$，再由物理模拟器（RaiSim）执行。手模型有 $L$ 个关节（link），物体表示为3D点云。

### 关键设计

1. **通用特征提取 $\Phi(\mathbf{s}, \mathcal{T})$**:

    - 输入包括关节角 $\mathbf{q}$、PD跟踪误差 $\mathbf{d}$、手/物速度 $\mathbf{u}_h, \mathbf{u}_o$、接触向量 $\mathbf{c}$、接触力 $\mathbf{f}$
    - **目标差分**：$\tilde{\mathbf{v}}, \tilde{\mathbf{m}}, \tilde{\omega}$ 分别表示当前与目标的朝向、位置、旋转差异
    - **关节距离特征** $\mathbf{l}^+ \in \mathbb{R}^{L \times 3}$：每个关节到**可抓取区域**最近点的向量；$\mathbf{l}^- \in \mathbb{R}^{L \times 3}$：到**不可抓取区域**最近点的向量
    - 设计动机：关节距离特征对物体形状通用（不依赖特定点云编码器），且能感知可/不可抓取区域的空间关系，使策略对任意形状物体具备泛化能力

2. **多目标奖励函数**:

    - 总奖励 $r = r_{\text{goal}} + r_{\text{grasp}}$，解耦目标满足和抓取稳定性
    - 目标奖励：$r_{\text{goal}} = r_{\text{dis}} + r_\mathbf{v} + r_\omega + r_\mathbf{m}$
        - $r_{\text{dis}} = -\sum_i [w_d^+(i)\|\mathbf{h}_i - \mathbf{o}_i^+\|^2 - w_d^-(i)\|\mathbf{h}_i - \mathbf{o}_i^-\|^2]$：鼓励接近可抓取区域，远离不可抓取区域
        - $r_\mathbf{v} = -w_\mathbf{v}\|\mathbf{v} - \bar{\mathbf{v}}\|^2$：朝向对齐
    - 抓取奖励：$r_{\text{grasp}} = r_\mathbf{c} + r_\mathbf{f} + r_{\text{anatomy}} + r_{\text{reg}}$
        - 接触奖励 $r_\mathbf{c}$、力奖励 $r_\mathbf{f}$（都区分可/不可抓取区域）、解剖约束 $r_{\text{anatomy}}$（MANO手用）、正则化 $r_{\text{reg}}$
    - 设计动机：奖励函数完全不依赖特定手模型结构，仅使用关节位置-物体表面的几何关系，因此可直接迁移到不同灵巧手

3. **学习课程(Curriculum)**:

    - **第一阶段**：在**静止物体**上训练，增大 $r_{\text{goal}}$ 权重，学习精确的手指运动以满足目标
    - **第二阶段**：在**可移动物体**上微调，增大 $r_{\text{grasp}}$ 权重，学习稳定抓取
    - **目标驱动引导(Objective-driven Guidance)**：将目标与当前值的差分直接作为手腕6DoF PD控制器的偏置项，加速探索
    - 设计动机：直接在可移动物体上同时学习目标和抓取会导致**局部最优**——手指移动满足目标时产生的接触力使物体翻转，导致抓取失败。课程设计将两个学习任务解耦

### 损失函数 / 训练策略

- 使用PPO算法进行RL训练，RaiSim物理模拟器
- 训练集仅58个物体（26个ShapeNet + 32个PartNet）
- 训练时随机采样目标：随机朝向 $\bar{\mathbf{v}}$、随机旋转 $\bar{\omega} \in [0, 2\pi)$，确保可抓取区域宽度≤12cm
- 单张Nvidia RTX 6000 GPU + 128 CPU核心

## 实验关键数据

### 主实验（PartNet & ShapeNet 对比）

| 方法 | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ | Contact Ratio↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| SynH2R-PD | 26.5% | 4.30cm | 0.767rad | 0.857rad | 13.0% |
| SynH2R | 82.3% | 4.06cm | 0.522rad | 0.568rad | 53.4% |
| **GraspXL** | **95.0%** | **2.85cm** | **0.270rad** | **0.306rad** | **86.7%** |

ShapeNet测试集（完全未见物体）：

| 方法 | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ |
|------|:---:|:---:|:---:|:---:|
| SynH2R | 65.8% | 4.49cm | 0.642rad | 0.688rad |
| **GraspXL** | **81.0%** | **3.22cm** | **0.292rad** | **0.338rad** |

大规模Objaverse泛化（503k物体）：

| 物体大小 | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ |
|----------|:---:|:---:|:---:|:---:|
| Small | 85.9% | 3.20cm | 0.311rad | 0.362rad |
| Medium | 84.5% | 3.16cm | 0.274rad | 0.315rad |
| Large | 79.0% | 3.50cm | 0.271rad | 0.306rad |
| **平均** | **82.2%** | **3.32cm** | **0.279rad** | **0.319rad** |

### 消融实验

| 配置 | Suc. Rate↑ | Mid. Err↓ | Head. Err↓ | Rot. Err↓ | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| w/o Guidance | 90.0% | 3.22 | 0.394 | 0.425 | 去掉目标引导，探索效率下降 |
| w/o Distance | 81.6% | 2.90 | 0.419 | 0.475 | 去掉距离特征，形状感知变差 |
| w/o Curriculum | 96.2% | 4.12 | 0.381 | 0.462 | 去掉课程，成功率高但目标精度差 |
| **Full Model** | **95.0%** | **2.85** | **0.270** | **0.306** | 最佳综合表现 |

不同灵巧手泛化（PartNet）：

| 手模型 | Suc. Rate↑ | Mid. Error↓ | Head. Error↓ | 关节数 |
|--------|:---:|:---:|:---:|:---:|
| MANO | 95.0% | 2.85cm | 0.270rad | 45 |
| Allegro | 95.3% | 4.38cm | 0.291rad | 16 |
| Shadow | 94.0% | 3.57cm | 0.317rad | 22 |
| Faive | 95.8% | 2.85cm | 0.228rad | 30 |

### 关键发现

- 仅用58个物体训练即可泛化到50万+未见物体，成功率82.2%，展现了极强的泛化能力
- SynH2R需要耗时一周生成ShapeNet测试集的参考姿态，而GraspXL能实时推理
- 课程学习虽然在成功率上略低于无课程版本，但目标精度显著提升（Head. Error从0.381降至0.270），说明解耦学习的必要性
- 关节距离特征是泛化的关键——不依赖点云编码器，直接从几何距离获取形状信息
- 对重建物体（含噪声mesh）和AI生成物体同样有效，实际应用无障碍

## 亮点与洞察

- **规模化的突破**：从58个训练物体到50万+测试物体的泛化，这是抓取动作合成领域前所未有的规模
- **零数据依赖**：完全不需要3D手-物交互数据，仅通过RL + 物理仿真学习
- **课程学习的精妙设计**：静止→可移动的物体状态切换 + 目标→抓取的奖励权重切换，有效解决多目标冲突问题
- **通用性**：同一框架无需修改即可支持MANO、Shadow、Allegro、Faive四种不同灵巧手

## 局限与展望

- 仅支持**刚体**物体抓取，不处理柔性物体（如布料、绳索）
- 物体过大或过重时成功率下降（Large物体79%），需要考虑力矩约束
- 当前仅生成**接近+抓取**动作，不包含后续的**操作**（如旋转、放置）
- 关节距离特征对遮挡和自遮挡敏感，未来可引入点云编码器增强形状理解
- 推理需物理仿真器，直接部署到真实机器人需要sim-to-real的工作

## 相关工作与启发

- **SynH2R** (Christen et al., ICRA 2024): 通过优化生成参考姿态再用RL跟随的两阶段方法，但优化耗时且控制目标有限
- **UniDexGrasp/UniDexGrasp++**: 同样用RL做抓取，但需要预生成参考姿态，泛化物体数量仅约100
- **D-Grasp** (Christen et al., CVPR 2022): 利用参考抓取姿态生成物理可信的抓取动作
- **课程学习在RL中的应用**：本文的静止→可移动课程设计可推广到其他接触丰富的RL任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多目标控制框架+课程学习设计有创新，大规模泛化令人印象深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多数据集、多手模型、消融完整、50万+物体规模测试，极为全面
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，符号统一，结构合理
- **价值**: ⭐⭐⭐⭐⭐ — 首次实现50万+物体的抓取动作生成，代码和数据集开源，对机器人和动画领域有直接价值

<!-- RELATED:START -->

## 相关论文

- [StickMotion: Generating 3D Human Motions by Drawing a Stickman](../../CVPR2025/human_understanding/stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)
- [Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [Learning Cross-Hand Policies of High-DOF Reaching and Grasping](learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)
- [PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)
- [PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)

<!-- RELATED:END -->
