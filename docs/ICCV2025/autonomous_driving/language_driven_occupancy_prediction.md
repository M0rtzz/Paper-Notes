---
title: >-
  [论文解读] Language Driven Occupancy Prediction (LOcc)
description: >-
  [ICCV 2025][自动驾驶][open-vocabulary occupancy] 提出LOcc，一个有效且可泛化的开放词汇占据(OVO)预测框架，核心是设计了语义传递标注管线（LVLM+OV-Seg→LiDAR→voxel），生成密集细粒度的3D语言占据伪GT，替代了噪声大且稀疏的传统中间特征蒸馏，在Occ3D-nuScenes上全面超越SOTA。
tags:
  - ICCV 2025
  - 自动驾驶
  - open-vocabulary occupancy
  - language-driven
  - semantic transitive labeling
  - 3D perception
  - occupancy prediction
  - CLIP
---

# Language Driven Occupancy Prediction (LOcc)

**会议**: ICCV 2025  
**arXiv**: [2411.16072](https://arxiv.org/abs/2411.16072)  
**代码**: [https://github.com/pkqbajng/LOcc](https://github.com/pkqbajng/LOcc)  
**机构**: Zhejiang University, CaiNiao (Alibaba)
**领域**: 自动驾驶 / 占据预测 / 开放词汇  
**关键词**: open-vocabulary occupancy, language-driven, semantic transitive labeling, 3D perception, occupancy prediction, CLIP

## 一句话总结
提出LOcc，一个有效且可泛化的开放词汇占据(OVO)预测框架，核心是设计了语义传递标注管线（LVLM+OV-Seg→LiDAR→voxel），生成密集细粒度的3D语言占据伪GT，替代了噪声大且稀疏的传统中间特征蒸馏，在Occ3D-nuScenes上全面超越SOTA。

## 背景与动机
基于视觉的占据预测是自动驾驶3D感知的核心任务，需要从图像输入估计场景的几何和语义。传统有监督方法被限制在固定语义类别集合上，且构建密集GT需要逐帧标注LiDAR点云，人工成本极高。开放词汇占据(OVO)旨在预测任意词汇集的占据状态，仅使用无标注数据训练。

现有OVO方法存在两个关键缺陷：
1. **使用图像特征作为中间媒介**：同类物体在不同图像中特征值不同（编码了语义+外观），语义表示不一致，导致噪声大
2. **基于voxel的model-view投影**：将voxel直接投影到图像平面获取标签，忽略遮挡关系，且仅用单帧图像，导致稀疏粗糙的voxel-to-text对应

## 核心问题
如何生成密集且细粒度的3D语言占据伪GT？

## 方法详解

### 整体框架
LOcc分为两大部分：**语义传递标注管线**（生成伪GT）+ **OVO模型训练**。

### Part 1：语义传递标注管线（核心创新）

#### Step 1：词汇提取 (LVLM)
- 使用Qwen-VL等LVLM对每帧环视图像进行chain-of-thought对话
- 先让LVLM描述场景，再要求列出所有物体类别名
- 合并多帧环视图像的结果得到该帧的完整词汇集

#### Step 2：像素-文本关联 (OV-Seg)
- 使用开放词汇分割模型（如FC-CLIP/SAN/CAT-Seg）
- 每个像素通过余弦相似度匹配最高分的文本标签
- 结果：每个像素都有一致的文本标签（非特征向量）

#### Step 3：LiDAR点云标签转移
- 将无标注LiDAR点投影到图像平面获取对应像素的文本标签
- **关键改进**：建模遮挡关系，防止被遮挡点获取错误标签

#### Step 4：场景重建与体素化
- 合并多帧LiDAR点云进行时序密集场景重建
- 对每个voxel使用**多数投票法**分配最频繁出现的标签
- 降低单帧分割噪声的影响

### Part 2：OVO模型架构

#### 语言自编码器（降维）
- CLIP embedding维度高（512/768），设计文本自编码器压缩到低维潜空间

#### 占据预测模型改造
- 基于现有有监督占据模型（BEVFormer/BEVDet/BEVDet4D）
- 将原始分类预测头替换为：几何头（二值占据状态）+ 语言头（低维语言特征）
- 推理时将预测特征与任意文本embedding做余弦相似度匹配

### 损失函数
- 几何损失：Binary CE用于占据状态预测
- 语言损失：预测语言特征与伪GT语言标签的余弦相似度损失

## 实验关键数据

### Occ3D-nuScenes上的OVO性能

| 方法 | Backbone | 输入分辨率 | mIoU |
|------|---------|-----------|------|
| POP-3D | R101 | 900×1600 | 11.70 |
| VEON | ViT-B | 900×1600 | 16.78 |
| VEON (temporal) | ViT-B | 900×1600 | 17.51 |
| **LOcc-BEVDet** | R50 | 256×704 | **20.29** |
| **LOcc-BEVDet4D** | R50 | 256×704 | **21.07** |
| **LOcc-BEVFormer** | R101 | 900×1600 | **23.15** |

- LOcc-BEVDet仅用R50+256×704分辨率即超越所有SOTA
- LOcc-BEVFormer比VEON高出+6.37 mIoU

### 消融实验
- LVLM词汇提取 vs. 固定类别集合：LVLM更全面
- 遮挡建模：+1.8 mIoU
- 多帧融合+多数投票：显著降低单帧噪声
- 语言自编码器降维（512→64）：mIoU仅下降0.3但计算量大幅减少

## 亮点
- **语义传递标注管线是核心创新**：Text label传递（而非visual feature蒸馏）从根本上解决了语义不一致问题
- **利用LVLM做场景词汇发现**：避免了预定义类别集合的限制
- **遮挡感知的标签传递**：显著提升标注准确性
- **框架泛化性极强**：兼容BEVFormer/BEVDet/BEVDet4D等多种主流模型
- **伪GT质量接近人工标注**：有望大幅降低3D标注成本

## 局限与展望
- 管线依赖LVLM和OV-Seg模型的准确性
- 词汇提取阶段需要逐帧LVLM推理，离线计算量大
- 仅在nuScenes上验证，Waymo等未评估
- 伪GT仍与人工标注有约5 mIoU差距

## 与相关工作的对比
- **vs. POP-3D**：用稀疏LiDAR+LSeg features蒸馏，语义噪声大；LOcc用文本标签传递+密集重建
- **vs. VEON**：用CLIP features+voxel直投，忽略遮挡；LOcc用OV-Seg文本标签+遮挡建模+多帧融合

## 启发与关联
- "文本标签传递优于特征蒸馏"可迁移到其他3D语言理解任务
- OVO的瓶颈不在模型架构而在伪GT质量——数据为王

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 语义传递标注管线理念新颖且实效显著
- 实验充分度: ⭐⭐⭐⭐ 多骨干验证+伪GT质量对比+消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 图示清晰，问题定义精准
- 价值: ⭐⭐⭐⭐⭐ 高质量免标注3D语言GT生成对3D感知社区有深远影响

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

<!-- RELATED:END -->
