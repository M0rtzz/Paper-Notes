---
title: >-
  [论文解读] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection
description: >-
  [ICLR 2026][目标检测][人员检测] ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件…
tags:
  - "ICLR 2026"
  - "目标检测"
  - "人员检测"
  - "森林搜救"
  - "无人机"
  - "遮挡感知"
  - "数据集"
---

# ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection

**会议**: ICLR 2026  
**arXiv**: [2603.02541](https://arxiv.org/abs/2603.02541)  
**代码**: [https://huggingface.co/datasets/etri/ForestPersons](https://huggingface.co/datasets/etri/ForestPersons)  
**领域**: 目标检测  
**关键词**: 人员检测, 森林搜救, 无人机, 遮挡感知, 数据集

## 一句话总结

ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件，为下冠层人员检测模型的训练和评估提供了坚实基础。

## 研究背景与动机

**领域现状**：无人机（UAV）已被广泛应用于搜救（SAR）任务中，能够快速覆盖大面积开阔区域。随着硬件微型化和 SLAM 技术发展，微型无人机（MAV）已具备在 GPS 受限的森林环境中安全导航和探索的能力。

**现有痛点**：

1. **视角局限**：现有 SAR 数据集（HERIDAL、WiSARD、SARD、VTSaR）均从高空俯视或斜视角采集，密集树冠遮挡导致人员在图像中仅占几个像素，检测极其困难
2. **场景偏差**：地面人员检测数据集（COCO、CrowdHuman、CityPersons）主要覆盖城市环境中站立行走的人，与森林搜救场景差异巨大——躺卧、坐姿、植被遮挡等情况几乎不涉及
3. **标注缺失**：没有数据集同时提供遮挡等级和姿态标注，无法系统性地评估不同难度条件下的检测能力

**核心矛盾**：森林搜救中最需要检测的失踪人员恰恰处于现有数据集最无法覆盖的场景——树冠之下、植被遮挡、非站立姿态。

**本文方案**：构建首个专注下冠层人员检测的大规模基准数据集 ForestPersons，模拟 MAV 低空视角，辅以姿态、可见性等语义标注，支撑面向真实搜救场景的检测模型开发。

## 方法详解

### 整体框架

ForestPersons 的构建流程为：森林环境视频采集 → 帧采样 → 边界框标注 → 姿态与可见性属性标注 → 人脸匿名化 → 基于难度的数据划分。整个流程围绕"还原真实 SAR 场景"这一核心原则设计。

### 关键设计一：多维度数据采集策略

数据采集旨在尽可能模拟真实搜救场景的复杂性：

- **视角模拟**：手持/三脚架摄像机在 1.5-2.0 米高度拍摄，模拟 MAV 在树冠下飞行的低空地面视角
- **姿态多样性**：志愿者表演疲劳或迷路状态，包括站立、坐着和躺在地面三种姿态，自然受到植被、枝干和地形遮挡
- **环境覆盖**：涵盖四季变化（夏季密集树冠 vs 冬季落叶+积雪）、多种天气（晴/阴/小雨）和不同时段（下午/黄昏）
- **总规模**：从 377 段视频中采样得到 96,482 张图像和 204,078 个标注实例

### 关键设计二：三维度语义标注体系

每个人员实例除边界框外，还标注两个 SAR 相关语义属性：

**姿态分类**（3 类）：

| 类别 | 含义 | SAR 意义 |
|------|------|----------|
| Standing | 站立 | 轻症/清醒状态 |
| Sitting | 坐着 | 疲劳/等待状态 |
| Lying | 躺着 | 受伤/昏迷状态 |

**可见性等级**（4 级）：

| 等级 | 含义 | 遮挡描述 |
|------|------|----------|
| 100 | 完全可见 | 无遮挡 |
| 70 | 轻微遮挡 | 身体大部分清晰可见 |
| 40 | 部分遮挡 | 人员可辨识但遮挡明显 |
| 20 | 严重遮挡 | 几乎无法辨识 |

当姿态因遮挡难以判断时，标注员参考相邻视频帧进行决策。

### 关键设计三：模型驱动的难度感知数据划分

数据按视频序列级别划分（防止时序相邻帧跨集泄露），划分策略基于模型驱动的难度评估：

1. 用 COCO 预训练的 Faster R-CNN 在每段视频上计算 $AP_{50}$
2. 难度分数定义为 $1 - AP_{50}$
3. 按难度将序列分为三组：easy（$< 0.45$）、medium（$0.45 \le \text{score} < 0.75$）、hard（$\ge 0.75$）
4. 按比例均匀分配到训练/验证/测试集

最终划分：训练集 67,686 图 + 145,816 标注，验证集 18,243 图 + 37,395 标注，测试集 10,553 图 + 20,867 标注。

## 实验结果

### 主实验：现有数据集在下冠层场景的迁移表现

用 Faster R-CNN 在不同数据集上训练后在 ForestPersons 测试集上评估，验证现有数据的不足：

| 训练数据 | 类型 | 自有测试集 AP | ForestPersons AP | ForestPersons AP₅₀ |
|----------|------|:---:|:---:|:---:|
| SARD | SAR/俯视 | 58.6 | 3.0 | 7.8 |
| HERIDAL | SAR/俯视 | 35.0 | 0.2 | 0.3 |
| WiSARD | SAR/斜视 | 18.5 | 11.3 | 29.0 |
| COCO-Person | 地面/城市 | 54.0 | 40.8 | 66.9 |
| CrowdHuman | 地面/城市 | 39.4 | 31.9 | 58.8 |
| CityPersons | 地面/城市 | 38.7 | 5.9 | 15.1 |

SAR 数据集在 ForestPersons 上的 AP 均低于 12%，证实高空视角数据无法适应下冠层场景。地面数据集中 COCO 表现最好（AP=40.8），但仍有显著性能衰减，说明城市场景与森林环境的差距。

### 基准实验：ForestPersons 上多检测器性能

| 检测模型 | 骨干类型 | AP | AP₅₀ | AP₇₅ | AR |
|----------|----------|:---:|:---:|:---:|:---:|
| SSD | MobileNetV2 | 45.0 | 83.6 | 43.1 | 53.7 |
| YOLOv3 | YOLO | 50.2 | 86.5 | 53.9 | 58.6 |
| YOLOX | YOLO | 51.0 | 89.0 | 54.4 | 58.2 |
| DETR | Transformer | 53.9 | 88.7 | 59.4 | 67.9 |
| RetinaNet | ResNet-50 | 64.2 | 93.9 | 74.4 | 70.9 |
| Faster R-CNN | ResNet-50 | 64.4 | 92.7 | 75.4 | 70.0 |
| DINO | Transformer | 65.3 | 94.0 | 76.2 | **77.7** |
| YOLOv11 | YOLO | 65.6 | 93.4 | 75.6 | 71.7 |
| CZ Det | 级联缩放 | 65.6 | **96.1** | **77.9** | 71.6 |
| Deformable R-CNN | ResNet-50 | **66.3** | 93.4 | 77.5 | 71.3 |

Deformable R-CNN 取得最高 AP（66.3），但不同指标下最优模型不同：DINO 的 AR 最高（77.7，搜救中更关注召回率），CZ Det 的 AP₅₀ 和 AP₇₅ 最优。

### 消融实验：属性对检测性能的影响

| 训练属性 → 测试属性 | Standing AP | Sitting AP | Lying AP |
|---------------------|:---:|:---:|:---:|
| 仅 Standing 训练 | 45.3-60.1 | 30.0-44.5 | 31.7-46.0 |
| 全姿态训练 | 49.3-65.5 | 50.6-65.7 | 47.5-65.1 |

仅用 Standing 数据训练时，Sitting/Lying 检测性能严重下降（约 -20 AP）；全姿态训练则三种姿态均获得大幅提升，证实多姿态数据的必要性。

**可见性等级与检测性能的相关性**：检测精度随可见性等级增加而稳步提升（从 20 级到 100 级），验证了 ForestPersons 的难度梯度设计与真实 SAR 条件的一致性。

## 论文评价

### 优点

1. **填补空白**：首个专注下冠层视角的大规模人员检测数据集，96K+ 图像规模是此前最大 SAR 数据集（WiSARD, 44K）的两倍以上
2. **标注全面**：边界框 + 姿态 + 可见性三维标注体系为系统性研究遮挡鲁棒性提供了独特基础
3. **实验充分**：不仅评估了多种检测器基准，还通过跨数据集迁移实验定量论证了数据集的必要性

### 不足

1. 数据采集依赖人工模拟搜救场景，可能与真实失踪人员的外观/姿态分布存在偏差
2. 仅提供 RGB 数据（热红外 ForestPersonsIR 仅在附录中简要提及），多模态融合潜力未充分挖掘
3. 最优检测器 AP 仅 66.3%，说明该场景仍有巨大提升空间，但论文未提出针对性的检测方法

### 评分

⭐⭐⭐⭐ — 作为一个数据集论文，ForestPersons 在问题定义的清晰度、数据规模和标注质量上均表现优秀，为森林搜救中的计算机视觉研究开辟了重要方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](../../CVPR2026/object_detection/mmrad_multimodal_anomaly_detection.md)
- [\[ICLR 2026\] InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)

</div>

<!-- RELATED:END -->
