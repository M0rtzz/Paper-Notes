---
title: >-
  [论文解读] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions
description: >-
  [AAAI2026][自动驾驶][目标检测] 提出TSBOW——一个基于CCTV的大规模交通监控数据集，包含198个视频、超32小时真实交通数据和320万帧，覆盖全年四季天气（晴/霾/雨/雪含极端灾害场景），涵盖8类交通参与者，重点解决恶劣天气下遮挡车辆检测的挑战。
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "目标检测"
  - "交通监控"
  - "benchmark数据集"
  - "遮挡检测"
  - "恶劣天气"
  - "CCTV"
---

# TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions

**会议**: AAAI2026  
**arXiv**: [2602.05414](https://arxiv.org/abs/2602.05414)  
**代码**: [SKKUAutoLab/TSBOW](https://github.com/SKKUAutoLab/TSBOW)  
**作者**: Ngoc Doan-Minh Huynh, Duong Nguyen-Ngoc Tran, Long Hoang Pham 等 (Sungkyunkwan University)  
**领域**: 自动驾驶  
**关键词**: 目标检测, 交通监控, benchmark数据集, 遮挡检测, 恶劣天气, CCTV  

## 一句话总结

提出TSBOW——一个基于CCTV的大规模交通监控数据集，包含198个视频、超32小时真实交通数据和320万帧，覆盖全年四季天气（晴/霾/雨/雪含极端灾害场景），涵盖8类交通参与者，重点解决恶劣天气下遮挡车辆检测的挑战。

## 背景与动机

全球变暖加剧了极端天气事件的频率和严重程度，这些天气条件会降低CCTV信号和视频质量，扰乱交通流，增加交通事故率。现有交通监控数据集如UAVDT和UA-DETRAC主要覆盖晴天和轻微雨天条件，对极端天气场景（重度降雪、浓雾等）的覆盖严重不足。AAURainSnow数据集虽然包含降雨和降雪，但仅有1.83小时数据且不提供bounding box标注。

从检测任务角度来看，目标检测模型在恶劣天气下面临多重挑战：水坑反射导致bounding box尺寸失真、强风和不稳定连接引起的相机抖动造成运动模糊、雪覆盖的车辆与白色背景融合难以区分等。已有数据集的规模和多样性不足以训练出在真实复杂场景中鲁棒工作的模型。

此外，现有基准数据集通常仅包含3-5个标注类别，缺少微交通工具（micromobility）和行人等重要交通参与者。在城市密集区域，遮挡问题极为严重（交叉路口、建筑施工等场景下车辆大面积相互遮挡），模型往往无法检测到严重遮挡的实例。因此，构建一个覆盖多种天气、多种道路类型、多种遮挡程度的大规模高质量交通监控基准迫在眉睫。

## 核心问题

如何构建一个涵盖全年多种天气条件（特别是极端灾害天气）、多种道路类型和多种遮挡程度的大规模交通监控benchmark，以推进恶劣天气下遮挡车辆检测的研究？

## 方法详解

### 数据采集设计

TSBOW数据来源于韩国水原市固定路线的CCTV系统，按以下四个维度系统分类：

- **场景类型**：道路（Road）、交叉路口（Intersection）、特殊情况（Special case，含共享车道/立交桥/道路施工）、灾害（Disaster，极端天气严重降低视频质量）
- **天气条件**：晴天/多云（Normal）、霾/雾（Haze）、雨（Rain，需可见雨滴）、雪（Snow，需可见雪花）
- **道路类型**：城市道路（Urban，双车道）、标准道路（Standard，四车道）、大道（Boulevard，六车道以上）
- **尺度分类**：细粒度（Fine）、中等（Medium）、粗粒度（Coarse），取决于摄像头高度和距离

### 半自动标注流程

标注采用五阶段pipeline：(1) 视频预处理，定义ROI区域并提取关键帧；(2) 使用X-Anylabeling工具进行人工标注；(3) 使用在韩国车辆特征上fine-tuned的YOLOv12x模型进行半自动标注；(4) 标注验证与质量控制；(5) 后处理生成最终数据集。共完成48,061帧人工标注和超过320万帧半自动标注。

### 标注类别体系

数据集包含8个类别：car、bus、truck、small truck、micromobility、pedestrian、unidentified、others。遮挡程度按IoU分为三级：无遮挡（<15%）、轻度遮挡（15-40%）、重度遮挡（$\geq$40%），分别有721,684、266,420和143,051个实例。

## 实验关键数据

### 基准模型性能对比（测试集，imgsz=1280，100 epochs）

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| YOLOv8x | 0.783 | 0.705 | 0.733 | 0.609 |
| YOLO11x | 0.786 | 0.696 | 0.734 | 0.614 |
| YOLOv12x | **0.806** | 0.662 | **0.744** | **0.615** |
| RT-DETR-x | 0.731 | **0.740** | 0.718 | 0.552 |

### YOLOv12x各类别性能

| 类别 | 实例数 | Precision | Recall | mAP50 | mAP50-95 |
|------|--------|-----------|--------|-------|----------|
| Car | 479,560 | 0.959 | 0.932 | 0.959 | 0.849 |
| Bus | 21,037 | 0.917 | 0.929 | 0.951 | 0.876 |
| Small Truck | 36,152 | 0.878 | 0.830 | 0.870 | 0.750 |
| Truck | 6,690 | 0.824 | 0.575 | 0.720 | 0.643 |
| Pedestrian | 32,779 | 0.833 | 0.605 | 0.715 | 0.447 |
| Micromobility | 18,490 | 0.793 | 0.574 | 0.726 | 0.519 |
| Unidentified | 4,855 | 0.475 | 0.223 | 0.317 | 0.221 |

### 跨场景/天气/尺度性能分析（YOLOv12x）

| 分组 | 条件 | mAP50 | mAP50-95 |
|------|------|-------|----------|
| 场景 | Intersection | 0.759 | 0.633 |
| 场景 | Disaster | 0.656 | 0.510 |
| 天气 | Rain | 0.789 | 0.664 |
| 天气 | Snow | 0.723 | 0.597 |
| 尺度 | Fine | 0.686 | 0.559 |
| 尺度 | Coarse | 0.733 | 0.581 |

### 跨数据集泛化验证（YOLOv12x，仅Car类）

| 训练数据 | Precision | Recall | mAP50 | mAP50-95 |
|----------|-----------|--------|-------|----------|
| UAVDT | 0.647 | 0.141 | 0.383 | 0.328 |
| UA-DETRAC | 0.820 | 0.295 | 0.558 | 0.459 |
| **TSBOW** | **0.743** | **0.869** | **0.846** | **0.792** |

## 亮点

- **极端天气覆盖全面**：首个包含重度雪灾、浓雾等灾害级天气场景的大规模CCTV交通监控数据集，198个视频跨全年四季（52晴/15霾/46雨/85雪）
- **规模突出**：超32小时、320万帧、7100万+标注实例，远超UAVDT（10小时/8万帧）和UA-DETRAC（10小时/14万帧）
- **8类细粒度标注**：涵盖car/bus/truck/small truck/micromobility/pedestrian等8类，类别分布相对均衡（car占69%，而其他benchmark超过83%）
- **系统化遮挡分析**：按遮挡程度三级分类，灾害场景下mAP50降至0.656，量化了极端条件对检测模型的挑战
- **跨数据集泛化优势**：在TSBOW上训练的模型对Car类mAP50达0.846，远超在其他数据集上训练的结果

## 局限与展望

- **仅覆盖白天场景**：当前版本不包含夜间数据，夜间恶劣天气下的检测挑战更加严峻，作者计划后续版本补充
- **地理单一性**：所有数据采集于韩国水原市，车辆类型和交通规则可能不具有全球代表性
- **人工标注比例有限**：48,061帧人工标注仅占总帧数的约1.5%，其余由YOLOv12x半自动标注，可能引入系统性偏差
- **缺少tracking和segmentation标注**：当前仅提供bounding box标注，未涵盖跟踪ID或实例分割mask

## 与相关工作的对比

| 数据集 | 时长 | 总帧数 | 标注类别 | 分辨率 | 天气覆盖 |
|--------|------|--------|---------|--------|----------|
| UAVDT | 10h | 80K | 3 | 1080×540 | 晴/雨 |
| UA-DETRAC | 10h | 140K | 5 | 960×540 | 晴/阴/雨/夜 |
| AAURainSnow | 1.83h | 132K | - | 640×480 | 雨/雪/霾/雾 |
| **TSBOW** | **32.36h** | **3.2M** | **8** | **1280×720** | **晴/霾/雨/雪+灾害** |

TSBOW在规模、分辨率、类别数和天气多样性上全面超越现有基准。UAVDT和UA-DETRAC缺少雪天场景；AAURainSnow不提供bounding box；TSBOW是唯一包含disaster级极端天气的交通监控数据集。

## 启发与关联

该数据集为恶劣天气下的鲁棒目标检测提供了宝贵的验证平台。灾害场景下mAP50仅0.656的结果表明，当前SOTA检测器在极端条件下仍有巨大提升空间。数据集可进一步用于domain adaptation研究（天气域迁移）、image restoration辅助检测、以及多尺度dense object detection等方向。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个涵盖灾害级极端天气的大规模CCTV交通监控benchmark，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ — 提供4款SOTA检测器基准，跨场景/天气/尺度维度的详细分析，跨数据集泛化验证
- 写作质量: ⭐⭐⭐⭐ — 数据集构建流程清晰，统计分析详尽
- 价值: ⭐⭐⭐⭐ — 数据集公开可用，对智能交通系统研究具有较高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation](../../CVPR2026/autonomous_driving/hg-lane_high-fidelity_generation_of_lane_scenes_under_adverse_weather_and_lighti.md)
- [\[AAAI 2026\] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions](walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)
- [\[ICCV 2025\] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes](../../ICCV2025/autonomous_driving/trafficloc_localizing_traffic_surveillance_cameras_in_3d_scenes.md)
- [\[CVPR 2026\] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](../../CVPR2026/autonomous_driving/perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](../../CVPR2026/autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)

</div>

<!-- RELATED:END -->
