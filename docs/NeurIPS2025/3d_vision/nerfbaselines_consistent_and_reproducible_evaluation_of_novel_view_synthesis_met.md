---
title: >-
  [论文解读] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods
description: >-
  [NeurIPS 2025][3D视觉][novel view synthesis] 提出NerfBaselines评测框架，通过统一的评估协议、环境隔离和原始代码封装，解决了新视角合成领域因评估协议差异导致的不公平比较问题，并通过实验揭示了微小的协议差异（如图像缩放方式、背景颜色）可以显著改变方法排名。
tags:
  - NeurIPS 2025
  - 3D视觉
  - novel view synthesis
  - 评测框架
  - NeRF
  - 3D Gaussian Splatting
  - benchmark
---

# NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods

**会议**: NeurIPS 2025  
**arXiv**: [2406.17345](https://arxiv.org/abs/2406.17345)  
**代码**: [https://nerfbaselines.github.io](https://nerfbaselines.github.io)  
**领域**: 3D视觉  
**关键词**: novel view synthesis, 评测框架, NeRF, 3D Gaussian Splatting, benchmark

## 一句话总结
提出NerfBaselines评测框架，通过统一的评估协议、环境隔离和原始代码封装，解决了新视角合成领域因评估协议差异导致的不公平比较问题，并通过实验揭示了微小的协议差异（如图像缩放方式、背景颜色）可以显著改变方法排名。

## 研究背景与动机
新视角合成（NVS）是计算机视觉的重要问题，随着NeRF和3DGS的快速发展，每天都有多篇arXiv论文发表。然而，该领域面临一个被忽视但极其关键的问题：**缺乏统一的评估协议**。

现有方法在评估时存在诸多不一致之处：不同的图像缩放策略（手动缩放 vs 使用预缩放的JPEG）、不同的图像分辨率、是否对图像做畸变校正、LPIPS使用VGG还是AlexNet、半透明图像的背景颜色（白色 vs 黑色）、图像是否四舍五入到uint8范围、指标计算前的归一化方式等。这些看似微小的差异会人为地提升某些方法的性能表现，使得文献中的定量比较变得不可靠。

一个典型的例子是：Gsplat仅通过改变Mip-NeRF 360数据集上的图像缩放方式，PSNR排名就提升了3位。这意味着研究者可以通过"调协议"而非"做研究"来获得更好的数字，这严重损害了领域的科学性。

现有的开发框架如NerfStudio虽然提供了统一的数据加载和评估工具，但其重实现的方法与原始实现往往存在性能差异（如TensoRF在NerfStudio中PSNR从36.46降至33.09），且方法需要在框架内重新实现，工作量大、难以保证一致性。

NerfBaselines的核心思路是：**不重新实现方法，而是封装原始代码**，通过轻量级API接口和环境隔离，确保每个方法的行为与原始发布版本完全一致，同时强制执行统一的评估协议。

## 方法详解

### 整体框架
NerfBaselines框架由四个核心组件构成：统一API、环境隔离、交互式查看器、Web平台。框架将各方法的模型实现与数据加载和评估代码分离，使得任何方法可以在任何数据集上使用相同的评估协议。

### 关键设计
1. **统一API接口（Unified API）**:

    - 识别了光线追踪方法和光栅化方法的共享结构，定义统一的方法类接口
    - 核心函数包括`train_iteration`（执行一步训练）和`render`（渲染单帧）
    - 每个方法只需编写一个轻量级wrapper调用原始代码，而非重新实现
    - 设计动机：最小化集成工作量的同时确保行为与原始代码一致。这是与NerfStudio的本质区别

2. **环境隔离（Environment Isolation）**:

    - 为每个方法创建独立的运行环境，冻结源代码和依赖版本
    - 支持三级隔离：Conda、Docker、Apptainer，适应不同的系统和隔离需求
    - 通过进程间通信（IPC）与隔离环境交互
    - 设计动机：在19个方法中，有12个使用官方说明无法安装——依赖版本冲突和缺失包是长期可复现性的最大障碍。冻结环境可确保方法在未来仍能安装和产出一致结果

3. **标准化评估协议（Standardized Evaluation Protocol）**:

    - 图像在uint8范围内评估，确保可复现性
    - LPIPS默认使用AlexNet（旧数据集使用VGG），版本固定为0.10.1
    - 使用预缩放发布图像，避免不同平台/库的缩放算法差异
    - SSIM参数固定：kernel size=11, σ=1.5, k1=0.01, k2=0.03
    - 数据集特定细节：Blender用白色背景+VGG LPIPS，Mip-NeRF 360用预缩放JPEG等
    - 设计动机：每个细节的选择都经过仔细考虑，目的是与文献中的主流做法保持一致，同时消除歧义

### 损失函数 / 训练策略
NerfBaselines本身不涉及训练，而是使用各方法原始的训练策略。框架的重点在于**评估阶段**的标准化。

## 实验关键数据

### 主实验（复现性验证）

| 方法 (场景) | 论文PSNR | 官方代码PSNR | NerfStudio PSNR | NerfBaselines PSNR |
|-------------|---------|-------------|----------------|-------------------|
| TensoRF (lego) | 36.46 | 36.54 | 33.09 | **36.49** |
| 3DGS (garden) | 27.41 | 27.39 | 27.17 | **27.34** |
| Instant-NGP (lego) | 36.39 | 36.09 | 15.24 | **35.65** |
| SeaThru-NeRF (Panama) | 27.89 | 27.85 | 31.28 | **27.82** |
| Zip-NeRF (garden) | 28.20 | 28.22 | - | **28.19** |

### 消融实验（协议差异影响）

| 场景/协议变化 | 影响 | 说明 |
|-------------|------|------|
| Mip-NeRF 360: 手动缩放 vs 预缩放JPEG | PSNR差异0.2-0.3dB | 3DGS方法排名变化，Gsplat排名升3位 |
| Blender: 白色背景 vs 黑色背景 | PSNR差异0.3-0.5dB | GOF从第1变为第2，Mip-Splatting从第2变为第1 |
| Photo Tourism: 半图 vs 全图外观优化 | PSNR差异1-2dB | WildGaussians从第1变为第3 |

### 关键发现
- 大多数已发表方法的结果可以被复现，说明"恶意刷评估协议"目前不是主要问题
- 同一方法在不同协议下的性能差异可以**超过**不同方法在同一协议下的差异——这意味着协议差异可以制造虚假的进步
- NerfStudio的重实现版本有时与原始实现差距很大（如Instant-NGP lego场景差超20dB）
- 3DGS方法对评估协议的敏感度普遍高于NeRF方法，可能与显式表示的特性有关

## 亮点与洞察
- 这是一篇极具社区价值的系统论文，解决了一个所有研究者都知道但没人系统解决的问题。揭示了评估协议差异可以逆转方法排名这一发现，足以引起整个领域的警惕
- "封装原始代码而非重新实现"是正确的工程哲学。NerfStudio的重实现方式注定无法保证一致性，而NerfBaselines的wrapper方式将集成成本和出错风险都降到最低
- 19个方法中有12个无法用官方说明安装这一数据本身就揭示了该领域的可复现性危机
- 交互式Viewer和Camera Trajectory Editor的设计超越了数字对比，提供了在训练轨迹外进行定性评估的工具
- Web平台允许下载渲染图像和checkpoint，使结果可被独立验证，体现了开放科学精神

## 局限与展望
- 环境隔离虽然解决了兼容性问题，但增加了磁盘和计算开销，每个方法独立环境可能占用数十GB磁盘
- 目前主要覆盖静态场景的NVS方法，动态场景、生成式3D方法的覆盖有限
- Web平台需要持续维护和更新，随着新方法不断涌现，长期可持续性是挑战
- 缺乏对评估指标本身的深入讨论（如PSNR是否是衡量视觉质量的最佳指标）
- 对训练超参数的统一控制有限——不同方法使用不同的训练策略，框架仅统一了评估而非训练
- 未覆盖场景编辑、文本驱动生成等新兴NVS任务

## 相关工作与启发
- **vs NerfStudio**: NerfStudio重新实现方法，性能常有偏差；NerfBaselines封装原始代码，保证一致性
- **vs SDFStudio**: 类似NerfStudio的问题，且专注于SDF表示，覆盖面更窄
- **vs 各方法独立评估**: 独立评估缺乏统一协议，结果不具可比性；NerfBaselines提供了公共标准

## 评分
- 新颖性: ⭐⭐⭐ 框架设计思路不复杂，但对问题的系统分析和揭示的insights有新颖性
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖19个方法、多个数据集、多种协议变体的全面对比，非常详实
- 写作质量: ⭐⭐⭐⭐⭐ 论点清晰有力，用数据说话，每个设计选择都有充分动机说明
- 价值: ⭐⭐⭐⭐⭐ 对整个NVS社区有极大的基础设施价值，有望成为领域标准评测工具

<!-- RELATED:START -->

## 相关论文

- [HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)
- [High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](../../CVPR2025/3d_vision/movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)

<!-- RELATED:END -->
