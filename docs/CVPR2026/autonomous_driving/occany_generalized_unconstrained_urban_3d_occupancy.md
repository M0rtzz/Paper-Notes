---
title: >-
  [论文解读] OccAny: Generalized Unconstrained Urban 3D Occupancy
description: >-
  [CVPR 2026][自动驾驶][3D占用预测] OccAny 提出了首个泛化无约束城市 3D 占用预测框架，能在无标定、域外场景中从单目/序列/环视图像预测度量级占用体素，通过 Segmentation Forcing 和 Novel View Rendering 两项关键设计，在 KITTI 和 nuScenes 上超越所有视觉几何基线。
tags:
  - CVPR 2026
  - 自动驾驶
  - 3D占用预测
  - 泛化
  - 无约束场景
  - 视觉几何基础模型
  - 语义分割
---

# OccAny: Generalized Unconstrained Urban 3D Occupancy

**会议**: CVPR 2026  
**arXiv**: [2603.23502](https://arxiv.org/abs/2603.23502)  
**代码**: [https://github.com/valeoai/OccAny](https://github.com/valeoai/OccAny)  
**领域**: 自动驾驶  
**关键词**: 3D占用预测, 泛化, 无约束场景, 视觉几何基础模型, 语义分割

## 一句话总结

OccAny 提出了首个泛化无约束城市 3D 占用预测框架，能在无标定、域外场景中从单目/序列/环视图像预测度量级占用体素，通过 Segmentation Forcing 和 Novel View Rendering 两项关键设计，在 KITTI 和 nuScenes 上超越所有视觉几何基线。

## 研究背景与动机

**领域现状**：3D 占用预测（Occupancy Prediction）是自动驾驶场景中的核心感知任务，旨在联合估计密集体素的占用状态和语义标签。现有方法如 SurroundOcc、OccFormer 等已在 nuScenes-Occ 和 SemanticKITTI 上取得不错效果。

**现有痛点**：(1) 现有方法严重依赖域内标注数据和精确的传感器标定参数（内外参），无法泛化到新场景；(2) 视觉几何基础模型（如 DUSt3R、Depth Anything）虽然泛化性强，但缺乏城市场景的几何补全能力（被遮挡区域）和度量级预测精度；(3) 没有统一框架能同时支持序列输入、单目输入和环视输入三种模式。

**核心矛盾**：高精度占用预测需要专有数据和标定，但实际应用中往往无法获取这些先验。视觉基础模型的通用性与占用预测所需的城市场景特化之间存在 gap。

**本文目标**：构建首个"无约束"3D 占用预测框架，能在完全未标定的域外场景中，从任意配置的相机输入生成度量级占用预测和分割特征。

**切入角度**：作者观察到，可以将视觉几何基础模型（MUSt3R/Depth Anything 3）的强泛化重建能力与大规模分割模型（SAM2/SAM3）的语义能力结合，通过专门的训练策略弥补城市场景的 gap。

**核心 idea**：提出 Segmentation Forcing 来强制模型学习与分割一致的占用表征，以及 Novel View Rendering 来通过虚拟视角渲染实现几何补全，从而构建一个既保持泛化能力又适配城市场景的统一框架。

## 方法详解

### 整体框架

OccAny 采用两阶段 pipeline：(1) 重建阶段——基于视觉几何基础模型从输入图像重建场景的 3D 点云，并使用分割模型提取每个点的语义特征；(2) 渲染阶段——通过 Novel View Rendering 在虚拟视角下推断未观测区域的几何，从而实现占用补全。最终将 3D 点云体素化为占用网格。框架有两个变体：OccAny（基于 MUSt3R + SAM2）和 OccAny+（基于 Depth Anything 3 + SAM3）。

### 关键设计

1. **Segmentation Forcing**:

    - 功能：提升占用预测的语义质量，同时实现 mask 级别的实例预测
    - 核心思路：在训练过程中，将 SAM2/SAM3 生成的分割 mask 作为监督信号，强制重建模型的输出点云与分割 mask 在特征空间中对齐。具体地，对分割模型输出的每个 mask，将其覆盖区域内的 3D 点的特征蒸馏为一致的语义向量。这种 "distill" 模式让几何重建和语义分割在统一空间中联合学习
    - 设计动机：直接使用基础模型的原始重建输出缺乏细粒度语义区分。通过 Segmentation Forcing，占用体素不仅有几何信息，还携带与 SAM 一致的分割特征，实现几何-语义的联合提升

2. **Novel View Rendering（虚拟视角渲染）**:

    - 功能：补全遮挡区域和未观测区域的 3D 几何
    - 核心思路：在测试时（test-time），利用训练好的几何模型从已有重建点云出发，随机采样若干虚拟相机视角（通过旋转和平移生成），在这些新视角下渲染深度图和点云。多个虚拟视角的渲染结果与原始重建结果合并，形成更完整的 3D 点云。这是一种 test-time augmentation 策略，不需要额外训练
    - 设计动机：单一或少数真实视角的重建不可避免地存在遮挡盲区。通过虚拟视角渲染，模型可以"想象"被遮挡物体的背面和远处区域，显著提升占用的 recall

3. **Majority Pooling 体素化**:

    - 功能：将多来源的 3D 点云聚合为体素占用网格
    - 核心思路：将来自重建视角和渲染视角的所有 3D 点映射到统一的体素网格中。对每个体素，取其内所有点的语义标签的多数投票作为该体素的最终标签。支持 "separate"（重建和渲染分别投票）和 "unified"（合并投票）两种模式
    - 设计动机：多视角点云在体素空间中可能存在冲突的语义预测，多数投票是最鲁棒的聚合策略

### 损失函数 / 训练策略

训练分两阶段：(1) 重建阶段使用度量深度的回归损失训练几何预测，同时使用 Segmentation Forcing 的特征蒸馏损失对齐语义；(2) 渲染阶段训练虚拟视角的深度预测能力。两阶段均使用多个驾驶数据集（Waymo、VKITTI、DDAD、PandaSet、ONCE）联合训练以增强泛化性。OccAny 使用 16 x A100 40G 训练。

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | OccAny | OccAny+ | 之前最佳基线 | 说明 |
|------------|------|--------|---------|-------------|------|
| KITTI 5帧 几何 | IoU | 25.91 | 27.33 | <25 | 超越所有视觉几何基线 |
| nuScenes 环视 几何 | IoU | 34.15 | 33.49 | <33 | 无需域内标注 |
| KITTI 5帧 语义(distill) | mIoU / mIoU^SC | 7.30/13.54 | 6.49/13.31 | - | Segmentation Forcing |
| nuScenes 环视 语义(distill) | mIoU / mIoU^SC | 6.65/10.31 | 7.20/11.51 | - | OccAny+更优 |
| KITTI 5帧 语义(pretrained) | mIoU / mIoU^SC | 7.62/13.75 | 8.03/13.17 | - | 使用预训练分割 |
| nuScenes 环视 语义(pretrained) | mIoU / mIoU^SC | 7.42/10.78 | 9.45/12.22 | - | OccAny+大幅领先 |

### 消融实验

| 配置 | KITTI IoU | 说明 |
|------|----------|------|
| OccAny Full (5帧) | 25.91 | 完整模型 |
| w/o Novel View Rendering | ~22 | 去掉虚拟视角渲染后大幅下降 |
| w/o Segmentation Forcing | ~24 | 语义质量显著降低 |
| OccAny+ Full (DA3+SAM3) | 27.33 | 更强基础模型带来进一步提升 |
| 1帧 vs 5帧 | 24.03 vs 25.91 | 多帧序列输入提供额外几何线索 |

### 关键发现

- Novel View Rendering 是几何 IoU 提升的最大贡献者，约贡献 3-4 个点的绝对 IoU 提升
- OccAny+ 使用 Depth Anything 3（1.1B）+ SAM3 的组合效果更好，但 OccAny（MUSt3R + SAM2）在某些设置下也具竞争力
- 框架在完全域外（未见过 KITTI/nuScenes）的情况下性能接近域内自监督方法，展示了极强的泛化能力
- 深度估计指标上，OccAny+ recon 1.1B 在 KITTI 上实现 AbsRel 仅 9.58%，远超 DA3 原始的 33.28%
- Ego 轨迹评估 ADE：OccAny+ recon 1.1B 达到 0.90，超越 DA3 1.1B 的 1.12

## 亮点与洞察

- **首个真正泛化的 3D 占用预测框架**：不需要目标域的标注、标定或微调，直接推理即可工作。这对实际部署极有价值
- **虚拟视角渲染的 test-time augmentation** 是一个优雅的设计：它不增加训练复杂度，仅在推理时通过"想象"未见视角来补全几何，思路可迁移到任何 3D 重建任务
- 框架的模块化设计值得学习：几何重建和语义分割分别由专门的基础模型承担，通过 Segmentation Forcing 桥接，升级基础模型即升级整个系统

## 局限与展望

- 推理速度偏慢：部分设置需要渲染 150-180 个虚拟视角，单 GPU 推理耗时较长
- 语义 mIoU 绝对值仍偏低（7-9%），与有监督方法差距较大，主要受限于 SAM 特征与具体语义类别的对齐质量
- 对遮挡严重的密集城市场景（如狭窄巷道），虚拟视角渲染的补全效果可能有限
- 目前仅支持静态场景，动态物体（行人、车辆）的时序一致性未建模

## 相关工作与启发

- **vs SurroundOcc / OccFormer**: 这些方法需要域内 3D 标注训练且依赖精确标定，泛化性差。OccAny 完全零样本泛化，虽然绝对精度略低，但通用性远超
- **vs DUSt3R / MASt3R**: 视觉几何基础模型提供强泛化重建，但缺乏占用补全和语义预测。OccAny 在其之上添加了分割融合和虚拟视角渲染
- **vs Depth Anything 3**: DA3 提供强单目深度，OccAny+ 用它替代 MUSt3R 作为几何 backbone，证明框架的灵活性
- 本文的 Segmentation Forcing 思路类似于知识蒸馏，但应用于几何-语义对齐，可迁移到其他 3D 感知任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个泛化无约束 3D 占用框架，两项核心设计均具新意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 KITTI 和 nuScenes，三种输入模式，含重建和轨迹评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，但方法细节部分略显复杂
- 价值: ⭐⭐⭐⭐⭐ 实际部署价值极高，解决了 3D 占用预测最核心的泛化瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)

</div>

<!-- RELATED:END -->
