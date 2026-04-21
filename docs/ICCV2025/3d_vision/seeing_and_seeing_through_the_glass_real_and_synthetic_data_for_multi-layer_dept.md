---
title: >-
  [论文解读] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation
description: >-
  [ICCV 2025][3D视觉][多层深度估计] 提出多层深度估计(multi-layer depth estimation)新任务，构建了包含1500张真实图像的LayeredDepth基准和程序化合成数据生成器，揭示了现有深度估计方法在透明物体上的严重不足。
tags:
  - ICCV 2025
  - 3D视觉
  - 多层深度估计
  - 透明物体
  - 合成数据
  - 相对深度
  - 基准数据集
---

# Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation

**会议**: ICCV 2025  
**arXiv**: [2503.11633](https://arxiv.org/abs/2503.11633)  
**代码**: [项目页](https://layereddepth.cs.princeton.edu)  
**领域**: 3D视觉  
**关键词**: 多层深度估计, 透明物体, 合成数据, 相对深度, 基准数据集

## 一句话总结

提出多层深度估计(multi-layer depth estimation)新任务，构建了包含1500张真实图像的LayeredDepth基准和程序化合成数据生成器，揭示了现有深度估计方法在透明物体上的严重不足。

## 研究背景与动机

透明物体在日常生活中无处不在，理解其多层深度信息对实际应用至关重要：

**感知透明表面本身**：避免走进玻璃门/墙，能够抓取塑料袋

**感知透明表面背后的物体**：从透明容器中取物、识别窗户后的场景

现有深度数据集的核心缺陷：
- **仅有单层深度标注**：要么标注透明表面后的物体深度，要么标注透明表面本身深度，无法同时捕获
- **透明物体数量少或场景单一**：多局限于桌面小物体或室内环境
- **结构光/LiDAR无法可靠测量透明物体深度**：发射光穿透表面

## 方法详解

### 任务定义

多层深度估计：给定图像 $\mathcal{I}$ 和查询像素 $p=(x,y)$，预测有序深度序列 $\hat{\mathcal{D}} = \{\hat{d}_1, \dots, \hat{d}_n\}$，其中 $n$ 因像素而异。每次介质转换（如空气到水）定义一个新层。

### 真实世界基准 (LayeredDepth)

- **数据规模**：1500张CC0图像，涵盖家庭、餐厅、实验室、城市等多样场景
- **标注方式**：相对深度标注（数值深度对透明物体无法准确获取）
    - 标注者绘制单调深度线或选择参考点
    - 每个点标注层ID以指定所在层
    - 包含10%假元组用于惩罚生成过多层的模型
- **标注量**：2.5M对、5.9M三元组、5.8M四元组

### 合成数据生成器 (LayeredDepth-Syn)

基于Infinigen Indoor构建的完全程序化生成器：
- 无限变化的材质、形状和场景组合
- 随机材质分配系统：任何物体可被指定为透明
- 生成15,300张多层深度标注图像

### 基线模型设计

提出三种多层深度估计基线架构：
1. **Multi-head Output**：多个输出头分别预测不同层
2. **Layer Index Concatenation**：层索引拼接到输入中
3. **Recurrent**：循环预测各层深度

## 实验

### 主实验 - 现有方法在透明物体上的表现

| 方法 | 四元组准确率(All) |
|------|-----------------|
| ZoeDepth | 42.98% |
| MiDaS | 52.26% |
| Metric3D V2 | 55.14% |
| Depth Anything V2 | 70.43% |
| **Metric3D ft. (微调后)** | **75.20%** |

所有现有SOTA深度估计方法在透明物体上表现欠佳，Metric3D V2仅55.14%。

### 合成数据微调效果

| 方法 | 微调前 | 微调后 |
|------|--------|--------|
| Metric3D V2 (四元组) | 55.14% | **75.20%** |

在合成数据上微调后，四元组准确率提升约20个百分点(+36.3%相对提升)。

### 基线方法在真实基准上的结果

| 方法 | All-P | All-T | All-Q | Layer1-P | Mixed-P |
|------|-------|-------|-------|----------|---------|
| Multi-head | 63.42 | 42.55 | 25.97 | 65.66 | 74.72 |
| Index Concat | 64.46 | 44.00 | 26.00 | 66.95 | 76.70 |
| Recurrent | 62.36 | 41.88 | 24.64 | 68.08 | 73.51 |

Index Concat方法在总体上表现最佳，而Recurrent在高层数(Layer 5, 7)上性能显著下降。

## 亮点与洞察

1. **新任务定义**：首次系统性定义多层深度估计任务，填补了透明物体理解的重要空白
2. **标注策略巧妙**：使用相对深度标注避开了透明物体数值深度不可测量的根本困难
3. **合成数据有效性**：仅用合成数据训练即可在真实基准上取得良好泛化
4. **假元组设计**：10%假元组防止模型对不存在的层过度预测

## 局限性

- 基线模型的多层预测准确率仍有很大提升空间（四元组仅约26%）
- 合成数据与真实世界的域差距仍然存在
- 标注为手工完成（非众包），规模受限
- 未探索端到端数值多层深度预测

## 相关工作

- Metric3D, Depth Anything: 单层深度估计
- Booster, ClearGrasp: 透明物体数据集（但仅单层）
- Infinigen Indoor: 合成场景生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (开创性新任务+数据集)
- 技术深度: ⭐⭐⭐ (基线方法相对简单)
- 实验充分度: ⭐⭐⭐⭐ (多个SOTA方法对比+基线)
- 实用价值: ⭐⭐⭐⭐⭐ (透明物体理解是关键问题)

<!-- RELATED:START -->

## 相关论文

- [Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)
- [DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [Seeing A 3D World in A Grain of Sand](../../CVPR2025/3d_vision/seeing_a_3d_world_in_a_grain_of_sand.md)
- [Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)

<!-- RELATED:END -->
