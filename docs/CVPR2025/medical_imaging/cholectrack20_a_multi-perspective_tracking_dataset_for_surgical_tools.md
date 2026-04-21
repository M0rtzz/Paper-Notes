---
title: "CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools"
authors: "Chinedu Innocent Nwoye, Kareem Elgharably, Cristians Gonzalez, Tong Yu, Pietro Mascagni, Didier Mutter, Nicolas Padoy"
venue: "CVPR 2025"
date: 2023-12-12
tags: [surgical-tool-tracking, dataset, multi-object-tracking, cholecystectomy, medical-imaging]
arxiv: "2312.07352"
---

# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

**作者**: Chinedu Innocent Nwoye, Kareem Elgharably 等  
**机构**: University of Strasbourg / University of Basel  
**会议**: CVPR 2025  
**arXiv**: 2312.07352  
**许可**: CC-BY-NC-SA 4.0  

## 研究背景与动机

手术器械追踪是计算机辅助手术系统的核心能力之一，对手术导航、技能评估和安全监控至关重要。然而，现有手术器械追踪研究面临多重挑战：

**数据集匮乏**：现有手术器械数据集主要关注检测或分割任务（如 Cholec80、m2cai16-tool），缺乏高质量的追踪标注。已有的追踪数据集规模偏小且标注维度单一

**追踪场景的特殊性**：手术场景与自然场景有本质区别——器械频繁进出视野、多器械交叉遮挡、血液/烟雾干扰、光照剧烈变化

**多视角追踪需求**：外科手术中同一器械从不同视角具有不同的追踪需求：
   - 整台手术层面（术中 intraoperative）：器械在整个手术过程中的使用轨迹
   - 体腔内层面（体内 intracorporeal）：器械在腹腔内的运动轨迹
   - 可见性层面（visibility）：器械在当前视频帧中是否可见

**评测标准缺失**：缺乏针对手术场景特性设计的追踪评测协议，直接使用 MOT 评测指标可能忽略手术安全相关的关键维度

CholecTrack20 旨在填补这些空白，提供首个多视角、大规模的手术器械追踪基准。

## 数据集详解

### 数据规模

| 统计项 | 数值 |
|--------|------|
| 视频数 | 20 |
| 标注帧总数 | ~35,000 |
| 器械实例总数 | ~65,000 |
| 器械种类 | 7 |
| 帧率 | 1 fps |
| 分辨率 | 854×480 或 1920×1080 |

### 七类手术器械

| 器械类型 | 功能 | 出现频率 |
|----------|------|----------|
| Grasper | 抓取/牵引组织 | 高 |
| Bipolar | 双极电凝止血 | 中 |
| Hook | 电切/分离 | 高 |
| Scissors | 剪切 | 低 |
| Clipper | 钛夹施放 | 低 |
| Irrigator | 冲洗/吸引 | 中 |
| Specimen Bag | 标本取出 | 低 |

### 三视角标注体系

CholecTrack20 的核心创新在于提出三种互补的追踪视角：

**术中视角（Intraoperative Perspective）**：
- 跟踪器械在整台手术中的使用模式
- 同一把物理器械即使退出再进入体腔，仍保持相同 ID
- 适用于器械使用统计和手术流程分析

**体内视角（Intracorporeal Perspective）**：
- 仅跟踪器械在腹腔内的连续轨迹
- 器械退出体腔后丢失 ID，重新进入时分配新 ID
- 适用于运动分析和手术技能评估

**可见性视角（Visibility Perspective）**：
- 记录器械在每帧中的可见/遮挡状态
- 处理器械被组织、血液或其他器械遮挡的情况
- 适用于安全监控（确保器械不在视野外操作）

### 标注流程

- 专业标注团队（含外科背景人员）
- 三轮标注+审核质量控制
- Bounding box + 器械类别 + 追踪 ID（三视角各一套 ID）

## 基准实验

### 检测结果

| 检测器 | AP@0.5 ↑ | AP@0.75 ↑ | mAP ↑ |
|--------|----------|-----------|-------|
| Faster R-CNN | 68.3% | 42.1% | 39.7% |
| DETR | 72.5% | 48.3% | 44.2% |
| YOLOv5 | 76.8% | 53.6% | 48.1% |
| **YOLOv7** | **80.6%** | **57.2%** | **51.8%** |

### 追踪结果（Visibility Perspective）

| 追踪器 | HOTA ↑ | MOTA ↑ | IDF1 ↑ | AssA ↑ |
|--------|--------|--------|--------|--------|
| SORT | 32.1% | 45.3% | 38.7% | 28.4% |
| DeepSORT | 38.6% | 52.1% | 45.2% | 35.1% |
| ByteTrack | 41.3% | 56.8% | 49.6% | 38.2% |
| **Bot-SORT** | **44.7%** | **60.2%** | **53.8%** | **41.5%** |

### 跨视角追踪对比

| 视角 | 最佳 HOTA | 主要挑战 |
|------|----------|---------|
| Visibility | 44.7% | 频繁遮挡 |
| Intracorporeal | 39.2% | 进出体腔导致 ID 中断 |
| Intraoperative | 35.8% | 长时间跨度 + 外观变化 |

## 方法详解

### 追踪方法适配

作者对主流 MOT 方法进行了手术场景适配：

1. **检测阶段**：使用 YOLOv7 作为统一检测器，在 CholecTrack20 训练集上微调
2. **关联阶段**：测试了 IoU 匹配、外观特征匹配、运动预测等不同关联策略
3. **评测协议**：基于 TrackEval 框架，分别评估三个视角的追踪性能

### 手术场景特有挑战分析

- **器械交换**：两把 Grasper 在重叠区域交换位置，导致 ID switch
- **烟雾遮挡**：电凝操作产生大量烟雾，造成检测失败
- **快速运动**：器械快速进出导致运动模型预测失败
- **类内外观相似**：同类器械（如两把 Grasper）外观极为相似，纯外观匹配失效

## 核心贡献

1. **首个多视角手术器械追踪数据集**：提出三种追踪视角（术中/体内/可见性），全面覆盖手术场景需求
2. **大规模高质量标注**：20 个视频、35K 帧、65K 实例，专业团队多轮标注
3. **全面的基准评测**：测试了主流检测器和追踪器在手术场景的表现，为后续研究提供 baseline
4. **开放许可**：CC-BY-NC-SA 4.0 许可，促进学术研究

## 局限性

- 仅覆盖腹腔镜胆囊切除术，未扩展到其他术式
- 1 fps 的标注帧率可能遗漏快速运动的细节
- 未包含器械分割和姿态估计标注
- 现有追踪方法在手术场景的表现仍有较大提升空间

## 相关工作

- Cholec80: 腹腔镜手术器械识别数据集
- SurgToolLoc: MICCAI 2022 手术器械定位挑战赛
- MOT17/MOT20: 自然场景多目标追踪基准
- Bot-SORT: 基于 BoT 特征的 SORT 追踪方法

<!-- RELATED:START -->

## 相关论文

- [Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](../../AAAI2026/medical_imaging/bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](../../NeurIPS2025/medical_imaging/the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](../../NeurIPS2025/medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](../../ICCV2025/medical_imaging/progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset](../../AAAI2026/medical_imaging/rethinking_surgical_smoke_a_smoke-type-aware_laparoscopic_video_desmoking_method.md)

<!-- RELATED:END -->
