---
title: >-
  [论文解读] Extend3D: Town-Scale 3D Generation
description: >-
  [CVPR 2026][3D视觉][3D场景生成] 本文提出 Extend3D，一个无需训练的 3D 场景生成流水线，通过扩展预训练物体级 3D 生成模型（Trellis）的体素隐空间并引入重叠 patch 联合去噪、under-noising SDEdit 初始化和 3D 感知优化，从单张图像生成城镇级大规模 3D 场景，在人类偏好和定量评估中均超越现有方法。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D场景生成
  - 大规模场景
  - 训练无关
  - 扩展隐空间
  - 体素生成
---

# Extend3D: Town-Scale 3D Generation

**会议**: CVPR 2026  
**arXiv**: [2603.29387](https://arxiv.org/abs/2603.29387)  
**代码**: 无（有 project page）  
**领域**: 3D视觉  
**关键词**: 3D场景生成, 大规模场景, 训练无关, 扩展隐空间, 体素生成

## 一句话总结

本文提出 Extend3D，一个无需训练的 3D 场景生成流水线，通过扩展预训练物体级 3D 生成模型（Trellis）的体素隐空间并引入重叠 patch 联合去噪、under-noising SDEdit 初始化和 3D 感知优化，从单张图像生成城镇级大规模 3D 场景，在人类偏好和定量评估中均超越现有方法。

## 研究背景与动机

1. **领域现状**：3D 生成模型（如 Trellis、Hunyuan3D）已能生成高质量 3D 物体，但局限于物体级数据训练，使用固定大小的隐空间表示 3D 数据。
2. **现有痛点**：
    - 固定隐空间大小限制了输出细节，场景越大越模糊（类似低分辨率图像）；
    - 3D 场景数据集稀缺，数据驱动的场景生成方法只能生成有限类别；
    - 外绘式（outpainting）方法（如 SynCity、3DTown）逐块生成导致块间不一致、接缝可见。
3. **核心矛盾**：物体级模型的隐空间不足以表示大规模场景的细节，但缺乏场景级训练数据使得直接训练场景模型不可行。
4. **本文目标**：如何利用预训练的物体级 3D 生成模型实现高保真的大规模 3D 场景生成？
5. **切入角度**：借鉴 2D 高分辨率图像生成中的 MultiDiffusion 思路，在 x/y 方向扩展 3D 隐空间，使用重叠 patch 联合生成，但针对 3D 特有的问题（地面消失、物体旋转错误等）加入结构先验和优化。
6. **核心 idea**：将物体级 3D 模型的隐空间在水平方向扩展，通过重叠 patch 联合去噪+点云先验初始化+3D 感知损失优化，实现城镇级 3D 场景生成。

## 方法详解

### 整体框架

Extend3D 分为两阶段（与 Trellis 一致）：稀疏结构生成和结构化隐变量（SLat）生成。两阶段都使用扩展的隐空间。输入是单张场景图像，先通过单目深度估计器（MoGe-2）获取点云作为先验，然后在扩展隐空间上进行 SDEdit 初始化和优化去噪，输出大规模 3D 场景。

### 关键设计

1. **重叠 Patch-wise Flow (Overlapping Patch-wise Flow)**:
    - 功能：使扩展隐空间中的多个 patch 能同时生成并相互影响
    - 核心思路：将扩展的隐空间 $\mathbf{Z}_t \in \mathbb{R}^{aN \times bN \times N}$（$a,b$ 为扩展因子）用滑动窗口分成重叠 patch，每个 patch 独立计算 vector field 后合并（重叠区域取平均）。图像条件也相应裁切对齐。关键公式：$\bm{v}(\mathbf{Z}_t, \mathcal{I}, t) = \sum_{i,j} \phi_{i,j}^{-1}(\bm{v}_{i,j}) \oslash \sum_{i,j} \mathbf{1}_{\mathbb{W}_{i,j}}$
    - 设计动机：不同于 SynCity 等逐块顺序生成，重叠 patch 使相邻区域互相纠正，滑动窗口的小步长能捕捉局部信息变化，中心物体也能利用物体级模型的优势。消融表明 $d=2$ 时局部结构畸变，$d=4$ 时即可修复

2. **Under-noising SDEdit 初始化**:
    - 功能：从单目深度点云初始化场景结构，并补全遮挡区域
    - 核心思路：将点云体素化后编码为隐变量 $\mathbf{Z}_0^{(g)}$，但不使用标准 SDEdit（$t_{\text{noise}} = t_{\text{start}}$），而是设置 $t_{\text{start}} > t_{\text{noise}}$，即"去噪程度大于加噪程度"。这样模型会将缺失/遮挡区域视为额外噪声并填补。迭代应用 $O_n = \text{SDEdit}(O_{n-1})$ 逐步完善场景
    - 设计动机：标准 SDEdit 面临权衡困境：$t_{\text{start}}$ 太小则无法填补空洞，太大则破坏已有结构。Under-noising 打破了这一权衡，类似超分辨中用高频噪声增强细节的思路

3. **3D 感知优化 (Optimize with Prior)**:
    - 功能：在每个去噪步骤优化 vector field，防止物体级模型的去噪轨迹偏向物体动态
    - 核心思路：两阶段分别设计优化损失。稀疏结构阶段：$\mathcal{L}_{\text{SS}} = -\frac{1}{|\mathbb{P}|}\sum_{\bm{p}\in\mathbb{P}} \log \sigma((\mathcal{D}(\mathbf{Z}_t^{\text{SS}} - t\cdot\hat{\bm{v}}_t))_{\bm{p}})$，约束点云位置的体素不消失。SLat 阶段：$\mathcal{L}_{\text{SLat}} = \text{LPIPS}(\hat{\mathcal{I}}, \mathcal{I}) - \text{SSIM}(\hat{\mathcal{I}}, \mathcal{I})$，通过可微渲染将 3D 结果渲染到输入视角与原图对比
    - 设计动机：物体级模型在去噪过程中会让子场景偏向物体结构（地面消失、物体随机旋转），优化确保去噪路径与场景动态一致，同时消除 patch 间接缝

### 损失函数 / 训练策略

无需训练，所有组件在推理时应用。两个优化损失均使用 Adam 优化器在每个去噪步骤优化 vector field $\hat{\bm{v}}_t$。稀疏结构阶段使用 dilated sampling 保证全局一致性。

## 实验关键数据

### 主实验（定量，100张输入图像）

| 方法 | LPIPS↓ | SSIM↑ | PSNR↑ | CD↓ | F-score↑ |
|------|--------|-------|-------|-----|----------|
| Trellis | 0.650 | 0.239 | 10.0 | 0.0315 | 0.442 |
| Hunyuan3D | 0.683 | 0.255 | 10.4 | 0.0192 | 0.567 |
| EvoScene | 0.482 | 0.310 | 13.2 | 0.0188 | 0.498 |
| **Ours w/o SLat optim** | **0.400** | **0.333** | **13.8** | **0.0078** | **0.708** |
| **Ours (full)** | **0.240** | **0.611** | **20.4** | **0.0086** | **0.694** |

### 消融实验（a=b=2）

| 配置 | LPIPS↓ | SSIM↑ | PSNR↑ | CD↓ | F-score↑ |
|------|--------|-------|-------|-----|----------|
| Patch-wise flow only | 0.606 | 0.209 | 9.63 | 0.0348 | 0.261 |
| + 初始化 | 0.425 | 0.312 | 13.0 | 0.0083 | 0.693 |
| + SS优化 | 0.400 | 0.333 | 13.8 | 0.0078 | 0.708 |
| + SLat优化 (full) | 0.240 | 0.611 | 20.4 | 0.0086 | 0.694 |

### 关键发现

- 人类偏好评估中，Extend3D 在几何、保真度、外观、完整性四个维度上全面胜出（vs Trellis 50-67%、vs Hunyuan3D 73-76%、vs EvoScene 87%）
- 初始化是必需的：无初始化（$t_{\text{start}}=1$）时结构完全崩溃
- Under-noising 相比标准 SDEdit 能自然地填补遮挡区域而不破坏已有结构
- Division factor $d$ 越大结果越好（$d=8$ 最优），但计算开销也更大
- SLat 优化极大提升纹理质量（LPIPS 从 0.400 降到 0.240，PSNR 从 13.8 提升到 20.4）

## 亮点与洞察

- **Under-noising 概念**：巧妙地发现 $t_{\text{start}} > t_{\text{noise}}$ 时模型会将3D结构的不完整性视为噪声进行补全。这是一个简单但深刻的观察，可推广到其他需要"编辑+补全"的生成任务
- **场景级生成无需场景级数据**：完全复用物体级模型的知识，仅通过扩展隐空间+先验引导即可生成城镇级场景，避免了3D场景数据稀缺的瓶颈
- **重叠 patch 联合去噪 vs 顺序外绘**：同时生成所有 patch 使得相邻区域可互相修正，比 SynCity 的顺序方式更一致

## 局限与展望

- 扩展因子 $a,b$ 和 division factor $d$ 的选择需要手动调参，计算开销随之增大
- 依赖单目深度估计器的质量，MoGe-2 估计不准会传播误差
- 生成场景的物理合理性（如重力、遮挡关系）未被显式建模
- 对长走廊等极细长场景的效果未验证（当前主要展示方形/矩形场景）
- SLat 优化略微增加了 CD（0.0078→0.0086），可能对某些几何结构引入轻微偏差

## 相关工作与启发

- **vs SynCity**: 顺序外绘方式导致块间不一致和可见接缝，Extend3D 通过同时生成避免此问题
- **vs 3DTown/EvoScene**: 也使用点云初始化但依赖 RePaint 逐 patch 补全，无法处理物体级模型的系统性偏差（如地面消失）
- **vs MultiDiffusion**: 2D 高分辨率生成的思路扩展到 3D，但 3D 特有问题（物体中心性、空间对齐）需要额外的先验和优化

## 评分

- 新颖性: ⭐⭐⭐⭐ under-noising 和 3D 感知优化是有价值的贡献
- 实验充分度: ⭐⭐⭐⭐ 人类偏好+定量+消融全面，但数据集偏小
- 写作质量: ⭐⭐⭐⭐ 方法论述清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 训练无需场景数据即可生成城镇级3D场景，实用价值高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[CVPR 2025\] SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE](../../CVPR2025/3d_vision/sar3d_autoregressive_3d_object_generation_and_understanding_via_multi-scale_3d_v.md)
- [\[CVPR 2026\] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)

<!-- RELATED:END -->
