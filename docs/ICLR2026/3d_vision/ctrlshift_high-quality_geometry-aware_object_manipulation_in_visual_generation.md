---
title: >-
  [论文解读] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation
description: >-
  [ICLR 2026][3D视觉][物体操纵] 提出Ctrl&Shift，一个端到端扩散框架，通过将物体操纵分解为物体移除+参考引导修复，并注入相对相机位姿控制，首次在不依赖显式3D重建的情况下实现几何一致的细粒度物体操纵。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "物体操纵"
  - "扩散模型"
  - "几何一致性"
  - "相机位姿控制"
  - "图像编辑"
---

# Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation

**会议**: ICLR 2026  
**arXiv**: [2602.11440](https://arxiv.org/abs/2602.11440)  
**代码**: 待确认  
**领域**: 3D视觉 / 视觉生成  
**关键词**: 物体操纵, 扩散模型, 几何一致性, 相机位姿控制, 图像编辑

## 一句话总结
提出Ctrl&Shift，一个端到端扩散框架，通过将物体操纵分解为物体移除+参考引导修复，并注入相对相机位姿控制，首次在不依赖显式3D重建的情况下实现几何一致的细粒度物体操纵。

## 研究背景与动机

**领域现状**：物体级操纵（重定位、旋转物体并保持场景真实感）是影视后期、AR和创意编辑的基础操作。主流方法分为两大派：几何方法（NeRF/3DGS重建后操作）和扩散方法（文本/轨迹条件编辑）。

**现有痛点**：
   - 几何方法（NeRF/3DGS）提供精确控制但需要显式3D重建，per-scene优化成本高，泛化性差
   - 扩散方法（DragAnything/VACE等）泛化性好但缺乏细粒度几何控制，无法精确指定物体的位姿变换
   - 没有方法能同时实现：背景保持、几何一致的视角变换、用户可控变换这三个目标

**核心矛盾**：几何精度与泛化能力之间存在根本性的trade-off

**本文目标**：在不做显式3D重建的情况下，实现几何一致、细粒度可控的物体操纵

**切入角度**：不把内容提升到3D再编辑，而是把精确视角控制直接注入2D扩散过程

**核心 idea**：将物体操纵分解为"移除+参考修复+相机位姿控制"三个子任务，通过多任务多阶段训练在统一扩散框架内学习

## 方法详解

### 整体框架

Ctrl&Shift 要解决的是「把图里某个物体搬到新位置、换个视角看，而背景和物体本身都不能穿帮」，而且不想付出 per-scene 三维重建的代价。它的做法是把这件事拆成「先把物体从原处干净抹掉、再按指定相机位姿把参考物体画回目标位置」，整个过程在一个扩散框架里端到端完成，不显式重建任何三维结构。具体来说，网络一次性吃进五路条件——源图/视频帧、参考物体图像、源掩码、目标掩码、以及一个描述「源视角到目标视角怎么转」的相对相机位姿描述符，输出就是物体已经移动/旋转到位的目标帧。架构上以 ControlNet 式的 DiT 为骨架，外观类条件走控制分支注入，相机位姿则单独编码后经 cross-attention 注入，让几何信号和外观信号各走各的通道。

### 关键设计

**1. 任务分解与多任务训练：把五路纠缠的条件拆开学**

五个条件信号（源帧、参考图、源掩码、目标掩码、相机位姿）在「完整操纵」这一个目标里高度纠缠，模型很难分清是哪一路在起作用，容易学成捷径。本文把训练目标显式拆成一个主任务加两个辅助任务联合优化：**主任务**做完整操纵，即抹掉源位置物体、再在目标位置以目标视角把它重绘出来；**辅助任务 1（物体移除）**把参考图设成白图、目标掩码置全零、位姿推到画面外，逼模型专心学「怎么干净地去掉物体并补全背景」；**辅助任务 2（参考修复 + 相机控制）**把源掩码置全零、输入换成纯背景帧，逼模型学「在给定位姿下把参考物体合成出来」。

三个任务按 8:1:1 的权重混合采样，等于给每一路条件都安排了一个能单独锻炼它的子场景，外观、位置、位姿的贡献因此被解纠缠，而不是糊成一团。

**2. 相对相机位姿编码：用「相对变换」而非绝对位姿来表达视角**

要做几何一致的视角变换，就得把「从源视角到目标视角怎么转」量化成一个网络能消化的信号。本文采用 look-at 相机模型，每个视角参数化为 $(yaw, pitch, d, r_x, r_y)$，然后只编码两视角之间的相对关系：相对旋转矩阵的 axis-angle 表示 $\text{aa}(\mathbf{R}_{rel})$、相对平移 $\mathbf{t}_{rel}$、以及 NDC 平面上的偏移 $(\Delta r_x, \Delta r_y)$，拼成一个 8 维描述符 $\mathbf{f} \in \mathbb{R}^8$。这个描述符再经 Fourier 位置编码加 MLP 映射成 8 个 token（维度 $d=4096$），通过 cross-attention 注入 DiT。

用相对位姿而非绝对位姿，是因为绝对位姿需要先定义一套场景无关的标准坐标系，这在野外图像上几乎无法统一；而相对位姿天然以输入帧为基准，用户的操作就像在原图上「拖一下、转一下」，既直观又免去了标定标准位姿的麻烦。

**3. 掩码编码策略：让二值掩码对齐潜空间而不被当成外观**

掩码是二值的位置信号，如果直接丢进 VAE 编码，VAE 会把它当成普通图像的外观纹理来处理，语义就被扭曲了。本文改用 space-to-depth（pixel unshuffle）操作，把掩码直接按通道重排降分辨率，对齐到 VAE 的 stride，从而既进入潜空间又保住了二值语义的本意。推理阶段没有现成的目标掩码，则用源掩码的 bbox 做缩放加平移来近似得到，省去了用户手绘目标区域的负担。

**4. 两阶段训练：先在合成数据上学几何先验，再到真实数据上学真实感**

合成数据位姿标注精确但缺真实感，真实数据真实但位姿难标，单靠任一种都不够。本文把训练分成两段：**Stage I** 在约 2M 张合成图像对上预训练，白背景配随机相机位姿，让模型先把物体先验和相对位姿表示学扎实，此阶段主干和控制分支联合更新；**Stage II** 转到 10 万对高质量真实图像/视频对上微调，这时冻结主干、只更新控制分支，把注意力集中到背景保持和真实感上，避免真实数据的噪声破坏已学好的几何能力。

**5. 数据构建流水线：自动从真实图像造出带位姿标注的训练对**

真实世界几乎没有「同一物体两个视角 + 干净背景」的现成配对，所以训练对必须自动合成。流水线先用 Hunyuan3D-2 把物体重建成 mesh，再用可微渲染估计源相机位姿（只保留渲染轮廓与真值 IoU≥0.90 的样本以保证位姿可靠），接着采样一个目标位姿渲染出目标视角；背景一侧用 MiniMax-Remover 抹掉原物体得到干净底图，最后用一个物体粘贴网络把渲染物体和谐化地贴回去，凑成带精确位姿标注的训练对。整条流水线全自动，因此能规模化地覆盖真实图像和视频。

### 损失函数

采用 flow-matching 训练，沿线性路径 $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\boldsymbol{\varepsilon}$ 加噪，优化速度匹配损失 $\|\mathbf{v}_\theta(\mathbf{z}_t, \mathbf{c}, t) - \mathbf{v}^*(\mathbf{z}_t, t)\|_2^2$，其中 $\mathbf{c}$ 即上述五路条件。

## 实验关键数据

### 主实验

ObjectMover-A零样本评测：

| 方法 | PSNR↑ | DINO↑ | CLIP↑ | DreamSim↓ |
|------|-------|-------|-------|-----------|
| ObjectMover | 25.27 | 85.07 | 93.16 | 0.142 |
| **Ctrl&Shift** | **28.69** | **88.07** | **93.58** | **0.075** |

GeoEditBench（自建基准，几何感知编辑评测）：

| 方法 | PSNR↑ | DINO↑ | Pose MAPE↓ | Obj IoU↑ |
|------|-------|-------|-----------|----------|
| VACE | 24.32 | 75.38 | 30.56% | 0.72 |
| Nano-Banana | 26.38 | 78.05 | 24.36% | 0.78 |
| **Ctrl&Shift** | **28.71** | **85.23** | **17.70%** | **0.83** |

### 消融实验
- 去掉Stage 1：Pose MAPE从17.70%升至32.50%，几何理解严重受损
- 去掉Stage 2：PSNR从28.71降至24.83，背景保持和视觉质量下降
- 去掉辅助任务1：CLIP-Score降至86.32，语义一致性受损
- 去掉辅助任务2：Obj IoU降至0.65，Pose MAPE升至28.60%，物体级精度最受影响

## 亮点
- 概念上的关键突破：不需要3D重建即可实现几何一致物体操纵
- 多任务分解思路巧妙，让模型从各任务中学习到解纠缠的信号
- 数据构建流水线可规模化，支持真实世界图像和视频
- GeoEditBench提供了几何感知编辑的系统性评测

## 局限与展望
- 推理时目标掩码的近似（bbox缩放+平移）可能在极端变换下不准确
- 基于Wan-1.3B backbone，模型规模不大，复杂场景可能表现受限
- 目前只支持单物体操纵，多物体协同编辑未探索
- 数据构建依赖Hunyuan3D-2和物体粘贴模型，引入这些模型的误差
- 视频操纵能力虽展示但定量评测偏少

## 与相关工作的对比
- vs DragAnything：基于轨迹控制的扩散方法，泛化性差且缺乏位姿控制
- vs VACE：背景保持好但实际是平移整个画面而非真正操纵物体
- vs Nano-Banana/Qwen-Image-Edit：生成质量好但文本指令驱动的相机位姿控制不精确
- vs 3DiT/GeoDiffuser：依赖3D重建或几何条件，泛化性受限
- vs ObjectMover：视频先验方法，本文在PSNR上+3.42，DreamSim减半

## 启发与关联
- "不做3D重建但注入3D几何控制"的思路可推广到其他编辑任务
- 多任务解纠缠训练策略值得在多条件生成任务中借鉴
- 相对位姿编码比绝对位姿更适合交互式编辑场景

## 评分
- 新颖性: ⭐⭐⭐⭐ (任务分解+位姿注入的概念创新)
- 实验充分度: ⭐⭐⭐⭐ (多基准+消融+自建benchmark)
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐ (首次统一几何精度和扩散泛化)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](../../CVPR2025/3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](../../CVPR2026/3d_vision/qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](../../CVPR2025/3d_vision/matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)

</div>

<!-- RELATED:END -->
