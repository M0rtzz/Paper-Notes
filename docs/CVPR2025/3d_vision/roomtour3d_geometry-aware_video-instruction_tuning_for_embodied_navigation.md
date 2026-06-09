---
title: >-
  [论文解读] RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation
description: >-
  [CVPR 2025][3D视觉][视觉语言导航] RoomTour3D利用网络房屋参观视频构建了一个几何感知的视频-指令数据集，通过3D重建获取行走轨迹的几何信息，结合GPT-4生成开放词汇指令，显著提升了多个VLN基准任务的性能并支持可训练的零样本导航。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "视觉语言导航"
  - "视频指令微调"
  - "几何感知"
  - "室内导航"
  - "数据集"
---

# RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation

**会议**: CVPR 2025  
**arXiv**: [2412.08591](https://arxiv.org/abs/2412.08591)  
**代码**: [项目页面](https://roomtour3d.github.io)  
**领域**: 3D视觉 / 具身导航  
**关键词**: 视觉语言导航, 视频指令微调, 几何感知, 室内导航, 数据集

## 一句话总结

RoomTour3D利用网络房屋参观视频构建了一个几何感知的视频-指令数据集，通过3D重建获取行走轨迹的几何信息，结合GPT-4生成开放词汇指令，显著提升了多个VLN基准任务的性能并支持可训练的零样本导航。

## 研究背景与动机

视觉语言导航(VLN)任务长期受限于训练数据的多样性和规模，主要依赖人工策划的模拟器环境。现有数据集如R2R、CVDN等在模拟场景中制作，场景多样性不足，无法捕获真实世界的复杂性。

已有尝试利用网络数据的方法各有局限：AirBERT使用离散的Airbnb图像缺少室内场景的连贯性；ScaleVLN依赖人工策划的3D场景成本高且可扩展性差；YTB-VLN虽然使用YouTube视频但忽略了路径几何信息和物体多样性。没有方法同时实现场景多样性、物体开放性和空间几何感知这三个关键属性。

RoomTour3D的核心思路是利用互联网上易获取的房屋参观视频，这些视频以第一人称视角捕获连续运动，天然嵌入了空间几何属性，通过自动化流水线提取几何感知的导航轨迹和空间上下文化的文本指令。

## 方法详解

### 整体框架

RoomTour3D包含一个自动化数据生成流水线和两个训练任务。流水线从房屋参观视频出发，使用COLMAP进行3D重建获取几何信息，结合RAM/Grounding-DINO/Depth-Anything提取物体标签/定位/深度，最终用GPT-4生成两类轨迹：~100K描述增强轨迹（用于预训练）和~17K动作增强轨迹（用于导航微调）。

### 关键设计1: 描述增强轨迹生成

- **功能**: 提供丰富的空间感知描述用于模型预训练，增强模型的空间理解和物体认知能力
- **核心思路**: 以每2秒1帧的速率均匀采样生成行走轨迹。对每帧使用RAM标注物体类别、Grounding-DINO定位、Depth-Anything预测深度，组合成文本模板"There is a [object] to the [position] of current spot in [distance]"。使用BLIP-2预测房间类型（16类预定义房间），经时间平滑去噪。最终将帧级描述整合到GPT-4中，以"Task instruction - In-context examples - Prediction"格式生成可控且开放词汇的导航指令
- **设计动机**: 多源专家模型提供的物体多样性、空间位置和深度信息，使GPT-4能生成比模板化指令更准确的自由形式描述

### 关键设计2: 动作增强轨迹与3D重建

- **功能**: 提供带有导航决策点和候选动作的轨迹数据，支持导航微调
- **核心思路**: 使用COLMAP重建3D场景获取相机姿态。在最大偏航旋转点采样"决策帧"，每~1.5米采样序列帧构成轨迹。利用3D重建测量相机朝向差异和距离，DBSCAN聚类空间近邻帧，在每个路径中选择最近帧作为正样本、角差最大帧作为负样本。视频切分为100秒片段并行重建，通过深度优先搜索合并子模型
- **设计动机**: 与全景节点不同，动作增强轨迹提供了不同位置和朝向的候选视角，更接近真实导航场景中的决策过程

### 关键设计3: 基于NaviLLM的双任务训练

- **功能**: 将RoomTour3D数据集集成到通用导航模型中实现多任务增强
- **核心思路**: (a) **预训练阶段**：将描述增强轨迹用于视觉-指令摘要任务，帧作为候选观察用`<cand>`token封装，模型输出包含物体进展和房间位置的轨迹摘要；(b) **微调阶段**：将动作增强轨迹用于动作-指令导航任务，每帧视为可导航步骤，模型根据历史观察`<hist>`选择下一步动作，最后步骤要求摘要整个导航路径
- **设计动机**: 双阶段策略分别增强空间理解（预训练）和决策能力（微调），与NaviLLM的训练范式无缝对接

### 损失函数

使用标准next-token prediction损失，与语言模型训练方法一致。

## 实验关键数据

### 主实验: 多任务VLN性能提升 (SPL/GP指标)

| 方法 | CVDN(GP) | SOON(SPL) | R2R(SPL) | REVERIE(SPL) |
|------|---------|----------|---------|-------------|
| NaviLLM(原始) | 6.16 | 29.2 | 59 | 35.7 |
| NaviLLM(复现) | 6.09 | 28.0 | 56.7 | 31.4 |
| +RT3D描述 | 提升 | 提升 | 提升 | 提升 |
| **+RT3D完整** | **>6%提升** | **9.8%提升(SOTA)** | **提升** | **提升** |

### 关键数据规模

| 数据类型 | 数量 |
|---------|------|
| 描述增强轨迹 | ~100K |
| 导航指令 | ~200K |
| 动作增强轨迹 | ~17K |
| 房屋环境数 | 1,847 |

### 关键发现

- 整合RoomTour3D数据后，基线性能在所有VLN任务上同时提升超过6%
- SOON任务上提升9.8%并创造新SOTA
- 零样本导航性能超过所有非商业方法，与GPT-3.5基础的商业方法可比
- BLIP-2房间定位准确率达85%（人工标注验证）

## 亮点与洞察

1. **数据驱动的范式转变**：不是设计更好的模型架构，而是利用网络视频的规模和多样性来解决VLN的数据瓶颈
2. **自动化流水线的可扩展性**：整个数据生成过程自动化，可以持续从更多房屋参观视频中扩展
3. **从视频到导航的桥梁**：3D重建将视频的连续帧转化为带几何信息的导航轨迹

## 局限与展望

- COLMAP重建质量不一致，部分视频片段无法成功合并
- 当前数据集规模仍在持续扩展中
- 房间类型限制为16类可能不够覆盖所有场景
- 未来可探索更大规模的视频数据和更多类型的导航任务

## 相关工作与启发

- **YTB-VLN**: 使用YouTube视频但忽略路径几何，用模板化指令
- **ScaleVLN**: 使用人工3D场景但成本高
- **NaviLLM**: 当前SOTA的LLM导航模型，RoomTour3D直接增强其性能
- 启发：网络视频+3D重建+LLM的组合可以低成本地为具身AI提供大规模训练数据

## 评分

⭐⭐⭐⭐ — 数据集构建方法论扎实，自动化流水线设计合理，在多个VLN基准上取得一致且显著的提升。数据集的可扩展性和开放世界导航的潜力特别值得关注。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[ICCV 2025\] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness](../../ICCV2025/3d_vision/ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)
- [\[CVPR 2025\] 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video](4dgc_rate-aware_4d_gaussian_compression_for_efficient_streamable_free-viewpoint_.md)

</div>

<!-- RELATED:END -->
