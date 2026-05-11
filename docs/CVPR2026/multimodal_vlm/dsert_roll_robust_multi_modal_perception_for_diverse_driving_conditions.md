---
title: >-
  [论文解读] DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions
description: >-
  [CVPR 2026][多模态][多模态] 提出 DSERT-RoLL 驾驶数据集，首次集成立体事件相机、RGB、热成像、4D 雷达和双 LiDAR 六种传感器，覆盖多种天气和光照条件，并提出统一多模态 3D 检测融合框架。
tags:
  - CVPR 2026
  - 多模态
  - event camera
  - 4D radar
  - thermal camera
  - 3D detection
---

# DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions

**会议**: CVPR 2026  
**arXiv**: [2604.03685](https://arxiv.org/abs/2604.03685)  
**代码**: [https://jeongyh98.github.io/dsert-roll](https://jeongyh98.github.io/dsert-roll)  
**领域**: 自动驾驶 / 多模态感知  
**关键词**: multi-modal dataset, event camera, 4D radar, thermal camera, 3D detection

## 一句话总结

提出 DSERT-RoLL 驾驶数据集，首次集成立体事件相机、RGB、热成像、4D 雷达和双 LiDAR 六种传感器，覆盖多种天气和光照条件，并提出统一多模态 3D 检测融合框架。

## 研究背景与动机

自动驾驶感知在恶劣天气（雾、雨、雪）和极端光照条件下仍面临严峻挑战。传统 RGB+LiDAR 方案在这些场景中表现退化。新型传感器如事件相机（对高动态范围和快速运动鲁棒）、热成像（夜间有效）和 4D 雷达（恶劣天气穿透性强）各具互补优势，但现有数据集通常只包含部分传感器组合，缺乏在同一环境下对所有传感器的公平对比和系统研究。

DSERT-RoLL 的核心贡献在于：将所有这些新型传感器与传统传感器集成到同一采集平台，在相同场景下采集数据，使得跨传感器对比和融合研究首次成为可能。

## 方法详解

### 整体框架

数据集包含 22K 帧多模态传感器数据，覆盖高速公路、城市街道、郊区道路等场景。同时提出多模态 3D 检测融合框架：LiDAR 和 4D Radar 体素化特征生成初始 3D 框提议，RGB/热成像/事件相机特征通过置信度融合整合到 3D 空间。

### 关键设计

1. **全面传感器套件**：立体 RGB（2448×2048）、立体事件相机（1280×720）、立体热成像（640×512）、4D 雷达（100m 范围）、长距 LiDAR（150m）和短距高分辨率 LiDAR（100m, 360°），所有相机均为立体配置以覆盖前方视场。

2. **3D 范围传感器融合**：LiDAR 和 4D Radar 分别体素化后提取 BEV 特征，沿通道拼接后卷积融合，生成初始 3D 框提议。4D Radar 在恶劣天气中提供 Doppler 速度信息，弥补 LiDAR 在雾雪中性能退化的不足。

3. **相机-3D 范围传感器融合**：提出体素中心采样策略，从 LiDAR 和 Radar 的非空体素索引出发，建立统一稀疏体素特征空间。将每个非空体素投影到 RGB/热成像/事件相机的图像平面，通过可变形交叉注意力采样邻域图像特征并融合到 3D 空间，实现置信度加权的多模态融合。

### 损失函数 / 训练策略

使用标准 3D 检测损失（回归 + 分类），在多模态特征融合后的统一表示上训练检测头。训练/测试按 7:3 比例划分，确保天气、光照和类别分布在两个集合间平衡。

## 实验关键数据

### 主实验

| 模态组合 | 天气-晴 | 天气-雾 | 天气-大雪 | 光照-HDR |
|---------|--------|--------|---------|---------|
| L (仅LiDAR) | 82.90 | 65.67 | 54.14 | 74.51 |
| R+L | 84.67 | 66.14 | 59.43 | 79.31 |
| 4R+L | 88.26 | 67.41 | 69.96 | 82.98 |
| R+E+T+4R+L (全模态) | 90.30 | 71.42 | 72.94 | 86.33 |

### 关键发现

- 4D Radar 在恶劣天气（大雪 +15.82 vs 仅 LiDAR）中贡献最显著
- 事件相机在 HDR 和过曝光照条件下特别有价值
- 热成像在低光照和夜间场景中补充 RGB 的不足
- 全模态融合在所有条件下均最优，证实了传感器互补性

## 亮点与洞察

- 首个同时包含六种传感器（含新型传感器）并在同一环境采集的驾驶数据集
- 系统性地揭示了不同传感器在不同环境条件下的优势和劣势
- 体素中心采样策略优雅地解决了异构传感器到统一 3D 空间的映射问题
- 数据分布在天气、光照和类别间精心平衡

## 局限与展望

- 数据集规模（22K 帧）相比 Waymo 等大型数据集偏小
- 仅三个目标类别（车辆、行人、自行车），覆盖范围有限
- 传感器标定和时间同步在极端条件下可能存在偏差

## 相关工作与启发

- 与 K-Radar（4D 雷达）、DSEC（事件相机）、KAIST（热成像）等单传感器数据集互补
- 融合框架的模块化设计便于未来探索更多传感器组合

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个全面的多新型传感器驾驶数据集
- 技术深度：⭐⭐⭐⭐ — 融合框架设计合理
- 实验充分度：⭐⭐⭐⭐⭐ — 系统性消融各传感器组合
- 实用价值：⭐⭐⭐⭐⭐ — 填补了多传感器研究的数据空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

</div>

<!-- RELATED:END -->
