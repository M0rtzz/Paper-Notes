---
title: >-
  [论文解读] Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects
description: >-
  [ICLR 2026][articulated objects] 提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计…
tags:
  - "ICLR 2026"
  - "articulated objects"
  - "Gaussian splatting"
  - "图像分割"
  - "joint estimation"
  - "sequential RANSAC"
  - "prior-free"
  - "interaction video"
---

# Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects

**会议**: ICLR 2026  
**arXiv**: [2603.02910](https://arxiv.org/abs/2603.02910)  
**项目主页**: [AiM](https://haoai-1997.github.io/AiM/)
**领域**: 其他  
**关键词**: articulated objects, Gaussian splatting, part segmentation, joint estimation, sequential RANSAC, prior-free, interaction video  

## 一句话总结

提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计，辅以SDMD模块处理新暴露的静态区域，在复杂6部件物体（Storage）上以79.34% mean IoU大幅超越需先验的ArtGS（52.23%）。

## 研究背景与动机

**铰接物体理解的核心需求**：机器人操作、AR/VR和具身智能都需要理解铰接物体（如抽屉柜、门、笔记本电脑）的部件结构和运动关节参数。

**现有方法的先验依赖**：DTA和ArtGS等方法需要预先指定部件数量，这在真实场景中通常未知，且一旦指定错误会导致严重的分割失败。

**动静解耦的挑战**：交互过程中，部分部件运动而其余静止，但运动部件的移动会暴露出之前被遮挡的静态区域，传统方法难以处理这种新暴露区域。

**单一表征的局限**：纯静态或纯动态的3D高斯表征无法同时处理铰接物体中固定和运动部件的混合特性。

**关节类型的多样性**：铰接物体包含旋转关节（revolute）和平移关节（prismatic）等多种关节类型，需要统一的无先验估计方法。

**视频输入的实用性**：相比需要多视角静态扫描的方法，从单段交互视频中恢复铰接信息更加实用和自然。

## 方法详解

### 整体框架

AiM以一段人与铰接物体交互的视频加上物体静止状态的3D扫描为输入，输出部件分割、关节参数和完整重建。整条流水线先用双高斯表征把场景中静止与运动的部分解耦开，再在运动部分上做顺序RANSAC发现未知数量的部件并估计各自的关节，最后用SDMD补回运动过程中新暴露的静态区域。

### 关键设计

**1. 双高斯表征：把动静两类几何各自交给一套高斯。** 铰接物体在交互中是混合体——背景和未被操作的部件保持不动，被拉开或旋开的部件则在帧间移动，单一的3D高斯既要拟合固定几何又要拟合运动几何，二者会互相干扰，运动部件的位移会"污染"本应稳定的静态重建。AiM因此维护两套高斯：静态高斯（Static GS）负责不变的背景与静止部件，参数跨帧固定；可变形高斯（Deformable GS）负责运动部件，额外学习一个逐帧变形场。一个高斯归到哪一套不靠人工标注，而是由像素级渲染损失的梯度信号自动决定——能用静态参数解释的像素把梯度推向静态集合，必须靠帧间形变才能对齐的像素则把对应高斯拉入动态集合。这种显式分离既保住了静态几何的干净，又让后续部件分割只需在动态高斯上展开，搜索空间大幅收窄。

**2. 顺序RANSAC部件分割：不预设部件数量，靠运动轨迹一个个剥出部件。** 真实场景里物体到底有几个可动部件往往未知，DTA、ArtGS这类需要预先指定部件数量的方法一旦数错就会严重分割失败。AiM把分割问题转成"未知数量的刚体运动混合"，正好契合RANSAC的多模型拟合范式：对所有动态高斯的变形轨迹做刚体运动拟合，单次拟合得到的最大共识集（inlier 集合）对应一个做统一刚体运动的部件；把这批高斯剥离后，在剩余动态高斯上重复同样的拟合，直到残差低于阈值再停止。顺序迭代天然带来由大到小的发现顺序，运动幅度最显著的部件先被识别，既不需要预设 $K$，也避免了K-means那种对噪声敏感、且必须给定簇数的弊端。

**3. SDMD模块：补回部件移动后才露出来的静态几何。** 抽屉拉开会露出柜体内壁、门转开会露出门后空间，这些区域在初始扫描时被遮挡、观测一开始根本不可见，传统方法既没有它们的静态高斯、又因为它们不随部件刚体运动而无法归入动态集合，于是被错误分配或直接丢失。SDMD逐帧比对渲染图与真实图，在二者出现明显差异、即"凭现有高斯无法解释"的区域，初始化新的静态高斯并与原有Static GS合并。这一步填补了"先不可见、后出现"的静态几何这一容易被忽视却关键的环节，消融中去掉它会直接带来约5%的IoU下降。

## 实验关键数据

### 主实验

| 方法 | 部件先验 | Mean IoU (%) | Revolute JE (°) | Prismatic JE (mm) |
|------|---------|-------------|-----------------|-------------------|
| DTA | 需要 | 71.45 | 8.32 | 12.7 |
| ArtGS | 需要 | 76.99 | 5.61 | 8.9 |
| AiM (ours) | **不需要** | **80.21** | **4.23** | **7.1** |

### 消融实验

| 组件 | Mean IoU (%) | 说明 |
|------|-------------|------|
| Full AiM | **80.21** | 完整方法 |
| 去掉SDMD | 74.85 | 新暴露区域被错误分配 |
| 单一GS (无动静解耦) | 68.32 | 运动部件影响静态重建 |
| K-means替代RANSAC | 72.56 | 需预设K且对噪声敏感 |
| 真值部件数给ArtGS | 76.99 | 即使给正确先验仍不如AiM |

### 关键发现

1. **无先验优于有先验**：AiM无需部件数量先验却以80.21% mean IoU超越需要先验的ArtGS（76.99%），说明自适应发现比固定假设更鲁棒。
2. **复杂物体优势巨大**：6部件Storage物体上，AiM（79.34%）vs ArtGS（52.23%），差距高达27%，ArtGS在部件数较多时急剧退化。
3. **SDMD不可或缺**：去掉SDMD导致5.36%的IoU下降，证明新暴露区域处理的重要性。
4. **动静解耦是基础**：单一GS方案比完整方法低近12%，双高斯设计是成功的基石。

## 亮点与洞察

1. **彻底去除先验**：首次实现无需部件数量先验的铰接物体部件分割和关节估计，更符合真实应用需求。
2. **双高斯解耦设计巧妙**：将动静分离嵌入3DGS表征中，兼顾了重建质量和下游分析的便利性。
3. **SDMD的实用创新**：解决了被遮挡静态区域逐步暴露的问题，这是铰接物体理解中容易被忽视但至关重要的细节。
4. **顺序RANSAC的自然适配**：巧妙利用RANSAC的迭代剥离特性实现自适应部件数量发现。
5. **在复杂物体上的压倒性优势**：6部件场景的27%提升展示了方法的可扩展性。

## 局限与展望

1. **单次交互假设**：当前要求视频中包含所有部件的运动，如果某部件在视频中未被操作则无法被发现。
2. **刚体运动假设**：顺序RANSAC假设每个部件做刚体运动，对柔性铰链或弹性变形无法处理。
3. **计算成本**：双高斯表征和顺序RANSAC的组合计算开销较大，难以实时运行。
4. **视频质量依赖**：运动模糊或遮挡严重的低质量视频可能导致动态高斯估计不准确。

## 相关工作与启发

- **铰接物体重建**：DTA (Liu et al., 2024), ArtGS (Huang et al., 2024) 的基于高斯溅射的方法
- **3D Gaussian Splatting**：3DGS (Kerbl et al., 2023), Dynamic 3DGS (Luiten et al., 2024)
- **部件分割**：PartNet (Mo et al., 2019) 的监督方法；SAM3D等无监督方法
- **RANSAC**：Fischler & Bolles (1981) 的经典框架；Sequential RANSAC在多模型拟合中的应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 无先验部件发现 + 双高斯解耦 + SDMD均为新颖设计
- 实验充分度: ⭐⭐⭐⭐ 多种物体类型验证，消融充分
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰，实验展示详尽
- 价值: ⭐⭐⭐⭐⭐ 去先验的铰接物体理解对机器人和具身AI有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PartCraft: Crafting Creative Objects by Parts](../../ECCV2024/others/partcraft_crafting_creative_objects_by_parts.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](../../CVPR2025/others/magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[ICML 2025\] SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference](../../ICML2025/others/syndacate_a_synthetic_dataset_for_evaluating_part-whole_hierarchical_inference.md)
- [\[CVPR 2026\] DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](../../CVPR2026/others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)

</div>

<!-- RELATED:END -->
