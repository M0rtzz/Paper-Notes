---
title: >-
  [论文解读] EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance
description: >-
  [CVPR 2025][医学图像][超声心动图] 本文提出 EchoWorld，一种面向超声心动图探头引导的运动感知世界建模框架：先通过空间世界建模（掩码重建）和运动世界建模（探头运动与视觉变化预测）进行预训练以编码心脏解剖知识，然后在微调阶段引入运动感知注意力机制融合历史视觉-运动序列，在 10 个标准切面的引导任务上显著降低引导误差。
tags:
  - CVPR 2025
  - 医学图像
  - 超声心动图
  - 探头引导
  - 世界模型
  - 运动感知注意力
  - 表征学习
---

# EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance

**会议**: CVPR 2025  
**arXiv**: [2504.13065](https://arxiv.org/abs/2504.13065)  
**代码**: https://github.com/LeapLabTHU/EchoWorld  
**领域**: 医学图像  
**关键词**: 超声心动图、探头引导、世界模型、运动感知注意力、表征学习

## 一句话总结

本文提出 EchoWorld，一种面向超声心动图探头引导的运动感知世界建模框架：先通过空间世界建模（掩码重建）和运动世界建模（探头运动与视觉变化预测）进行预训练以编码心脏解剖知识，然后在微调阶段引入运动感知注意力机制融合历史视觉-运动序列，在 10 个标准切面的引导任务上显著降低引导误差。

## 研究背景与动机

**领域现状**：超声心动图是一种无创、低成本、广泛可用的心脏诊断工具，但其操作高度依赖操作者的经验——需要精确操控探头的位置、角度和压力来获取标准切面图像。全球合格超声技师的短缺限制了超声服务的普及。探头引导系统旨在预测从当前帧到目标标准切面所需的探头运动向量，为辅助扫描或全自动机器人扫描提供支持。

**现有痛点**：超声探头引导本质上是一个视觉-运动序列预测问题，模型不仅需要理解复杂的心脏解剖结构（腔室、瓣膜、血管），还需理解探头运动如何导致视觉信号的变化。现有方法（如基于模仿学习的 US-GuideNet）主要使用通用视觉 backbone（如 ResNet、DINOv2），缺乏对超声领域知识的专门编码，也缺乏有效整合运动信息与视觉特征的机制。标准的交错视觉-动作序列编码方式无法充分利用所有帧之间的运动关系。

**核心矛盾**：探头引导需要同时具备两种能力——理解"图像中有什么"（解剖结构知识）和理解"探头怎么动会看到什么"（运动-视觉动态），而现有方法缺乏将这两种知识系统化学习的框架。同时标准注意力机制对帧间运动关系的建模是隐式的，无法显式利用丰富的探头位姿数据。

**本文目标** (1) 如何设计一个预训练策略来系统地编码超声心脏解剖知识和运动动态？(2) 如何在引导推理时有效整合历史视觉-运动信息来提升预测精度？

**切入角度**：作者类比人类超声技师拥有"心脏内部地图"的心智模型——知道心脏各结构的位置以及探头移动后会看到什么。受世界模型概念启发，设计了空间世界建模（学解剖结构）和运动世界建模（学运动-视觉变化关系）两个预训练任务。在下游引导时，引入运动感知注意力将帧间的 3D 相对位姿编码到 key/value 中，替代标准的位置编码。

**核心 idea**：用空间+运动双世界模型预训练编码超声领域知识，用运动感知注意力显式整合帧间位姿关系来提升探头引导精度。

## 方法详解

### 整体框架

EchoWorld 采用两阶段设计。**预训练阶段**：基于 JEPA（Joint-Embedding Predictive Architecture）框架，同时进行空间世界建模（MAE 式的掩码区域重建）和运动世界建模（给定探头运动预测视觉变化）。**微调阶段**：冻结/微调预训练的视觉和运动编码器，加入运动感知注意力模块处理历史视觉-运动序列 $\{(I_{t_i}, p_{t_i})\}_{i=1}^N$，输出到各目标标准切面的运动向量预测。

### 关键设计

1. **空间世界建模（Spatial World Modeling）**:

    - 功能：编码心脏解剖结构的空间知识——腔室位置、瓣膜形态、组织纹理等
    - 核心思路：基于 JEPA 的掩码重建任务。用 ViT 作为 context 编码器，只处理可见 patch；target 编码器（context 编码器的 EMA 版本）处理完整图像。预测器在特征空间中预测被掩码区域的特征表示。损失函数为掩码位置的 L1 特征重建误差 $\mathcal{L}_{spatial} = \sum_{c \in M} \|g_\phi(f_\theta(x); M)_c - f'_{\theta'}(y)_c\|_1$。
    - 设计动机：通过重建被遮挡的解剖区域，模型必须学习不同心脏结构之间的空间关系和视觉特征，这正是超声技师需要的解剖知识。不同于像素空间重建（MAE），特征空间重建更关注语义层面的理解。

2. **运动世界建模（Motion World Modeling）**:

    - 功能：编码探头运动与视觉变化之间的因果关系
    - 核心思路：给定同一扫描中的两帧 $I_a, I_b$ 和它们之间的相对位姿 $p_{a\to b}$，用运动编码器 $A_\psi$ 编码位姿差异为运动特征 $z_{a\to b}$。预测器以 $I_a$ 的 context 特征和运动特征为条件，预测 $I_b$ 的平均池化特征。使用 InfoNCE 对比损失而非 L1 重建损失，使模型对探头运动的方向和幅度更敏感。公式为 $\mathcal{L}_{motion} = -\frac{1}{B}\sum_i \log \frac{\exp(\hat{h}_{y_i}^\top h_{y_i}/\tau)}{\sum_j \exp(\hat{h}_{y_i}^\top h_{y_j}/\tau)}$。
    - 设计动机：超声技师的核心技能之一就是知道"如果我把探头向左转10度，图像会怎么变"。对比学习让模型精确区分不同运动导致的不同视觉结果。两个任务联合预训练（$\mathcal{L}_{total} = \mathcal{L}_{spatial} + 0.1 \cdot \mathcal{L}_{motion}$），形成既懂解剖又懂运动的统一表征。

3. **运动感知注意力（Motion-Aware Attention）**:

    - 功能：在引导推理时，显式整合帧间的 3D 相对位姿信息到特征交互中
    - 核心思路：标准注意力中 Key 和 Value 是帧特征的线性投影，与帧间关系无关。运动感知注意力将 Key 和 Value 改为帧特征与帧间相对位姿的联合函数：$K_j^{(i)} = \text{MLP}(h_j, z_{i\to j})$，$V_j^{(i)} = \text{MLP}(h_j, z_{i\to j})$，其中 $z_{i\to j}$ 是帧 $i$ 和帧 $j$ 之间的相对运动编码。这样每个 query token $i$ 与其他帧 $j$ 的交互都包含了它们之间的空间关系信息。
    - 设计动机：标准注意力是置换不变的，对从哪一帧看过来的信息不加区分。但在探头引导中，距离当前位置 5cm 处帧的信息和 1cm 处帧的信息有完全不同的参考价值。将位姿差异嵌入 K/V 让模型能根据空间关系加权聚合历史信息。

### 损失函数 / 训练策略

预训练使用超过 100 万张来自 200+ 例常规扫描的超声图像，联合优化空间和运动世界建模损失。微调阶段使用 L1 引导损失 $\mathcal{L}_{guide} = \|a_t - \hat{a}_t\|_1$。探头位姿用 6 自由度表示（3 平移 + 3 旋转）。target 编码器使用 context 编码器的 EMA 更新。

## 实验关键数据

### 主实验（10 个标准切面的探头引导误差）

| 方法 | 预训练 | 平均平移误差 (mm) | 平均旋转误差 (°) |
|------|--------|-----------------|-----------------|
| Scratch | 无 | ~8.86 (推算) | ~9.07 |
| DINOv2 | ImageNet | ~8.62 | ~8.52 |
| EchoCLIP | 超声 | ~8.35 | ~8.37 |
| USFM | 超声 | ~8.39 | ~8.42 |
| US-MAE | 超声 | ~8.45 | ~8.46 |
| **EchoWorld** | **超声(世界建模)** | **~7.85** | **~7.95** |

EchoWorld 在所有 10 个标准切面上的平均引导误差最低，相比最佳超声预训练方法 EchoCLIP 平移误差降低约 0.5mm，旋转误差降低约 0.4°。

### 消融实验

| 配置 | 平均误差 | 说明 |
|------|---------|------|
| 仅空间世界建模 | 中等 | 缺少运动知识 |
| 仅运动世界建模 | 中等 | 缺少解剖知识 |
| 空间 + 运动（完整预训练） | 最优 | 两种知识互补 |
| 标准注意力（无运动感知） | 较差 | 无法显式利用位姿关系 |
| 运动感知注意力 | 最优 | 帧间位姿嵌入到 K/V 有效 |

### 关键发现
- 世界模型预训练显著优于所有通用预训练（ImageNet）和超声专用预训练（USFM、EchoCLIP 等），证明领域特定的世界建模比通用自监督学习更有效
- 空间和运动建模的联合预训练效果优于单独任何一个，两种知识具有互补性
- 运动感知注意力相比标准注意力有明显优势，验证了在 K/V 中嵌入帧间位姿差异的有效性
- 从 scratch 到 DINOv2 到 EchoWorld，预训练质量的提升直接转化为引导精度的提升
- 定性分析显示 EchoWorld 预训练后的预测器可以作为超声模拟器——给定一帧和探头运动，能合理预测移动后会看到什么（配合 diffusion model 映射到像素空间）

## 亮点与洞察
- **世界模型视角应用于医学影像**很有启发性——将超声扫描类比为智能体在世界中的导航，把解剖知识类比为"世界地图"，把运动预测类比为预判移动效果，这一概念迁移非常自然
- **运动感知注意力**的设计将 3D 位姿差异嵌入到注意力的 K/V 中，是一种通用的时空关系建模方式，可以迁移到任何涉及相机/传感器位姿变化的视觉序列任务（如机器人导航、自动驾驶）
- 对比学习用于运动世界建模比回归损失更有效，因为对比损失能更好地区分细微运动差异

## 局限与展望
- 数据集来自单一中心的机器人辅助扫描，探头由机械臂控制，与实际手持扫描的运动模式可能存在差异
- 运动感知注意力中帧间 K/V 的计算量为 $O(N^2)$（N 为序列长度），长序列场景下可能需要更高效的近似
- 当前只在 10 个标准切面上验证，非标准切面或病理性畸形的场景未涉及
- 预训练需要大量带有时间同步位姿标注的超声视频数据，这种数据的采集成本较高

## 相关工作与启发
- **vs US-GuideNet**: US-GuideNet 提供旋转引导但用通用视觉 backbone，EchoWorld 通过领域特定的世界模型预训练实现了更精确的引导
- **vs USFM/EchoCLIP**: 这些超声基础模型使用通用自监督/多模态学习，但未针对视觉-运动动态进行建模；EchoWorld 的运动世界建模填补了这一空白
- **vs JEPA/I-JEPA**: EchoWorld 基于 JEPA 框架但引入了运动条件预测，将其从被动观察扩展为主动交互的世界模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 世界模型视角在超声引导领域是全新尝试，运动感知注意力设计新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个标准切面、大规模数据、对比广泛的 baseline、定性可视化分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 思路清晰，从世界模型到超声引导的类比非常自然，图示优秀
- 价值: ⭐⭐⭐⭐⭐ 对超声自动化有直接推动作用，框架可推广到其他影像引导任务

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2025\] EchoONE: Segmenting Multiple Echocardiography Planes in One Model](echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2026\] EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](../../CVPR2026/medical_imaging/echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)

<!-- RELATED:END -->
