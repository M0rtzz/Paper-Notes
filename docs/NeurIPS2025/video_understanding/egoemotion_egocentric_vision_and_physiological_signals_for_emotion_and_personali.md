---
title: >-
  [论文解读] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks
description: >-
  [NeurIPS 2025][视频理解][egocentric vision] 提出egoEMOTION——首个结合第一人称视觉（Meta Project Aria眼镜）与生理信号的情感与人格识别数据集，涵盖43名被试、50+小时录制、16种任务，发现第一人称视觉信号（尤其眼动特征）在真实场景情感预测中优于传统生理信号。
tags:
  - NeurIPS 2025
  - 视频理解
  - egocentric vision
  - emotion recognition
  - personality
  - physiological signals
  - Project Aria
---

# egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2510.22129](https://arxiv.org/abs/2510.22129)  
**代码**: 有（开源数据集+基线实现）  
**领域**: 情感计算 / 第一人称视觉 / 多模态数据集  
**关键词**: egocentric vision, emotion recognition, personality, physiological signals, Project Aria

## 一句话总结
提出egoEMOTION——首个结合第一人称视觉（Meta Project Aria眼镜）与生理信号的情感与人格识别数据集，涵盖43名被试、50+小时录制、16种任务，发现第一人称视觉信号（尤其眼动特征）在真实场景情感预测中优于传统生理信号。

## 研究背景与动机

**领域现状**：第一人称视觉已建立大型基准（Ego4D、EPIC-KITCHENS），情感识别则依赖实验室环境中的生理信号（DEAP、AMIGOS等）。

**现有痛点**：(a) 第一人称视觉基准忽略参与者情绪状态，假设情感中立；(b) 现有情感数据集局限于实验室，生态效度低；(c) 唯一公开移动眼动情感数据集eSEE-d仅4类情绪、无生理信号、需固定头部。

**核心矛盾**：情绪和人格是行为的内在驱动力，但第一人称视觉系统无法建模这些状态。

**本文目标** 构建生态效度高的多模态情感数据集，证明第一人称视觉信号足以进行情感预测。

**核心 idea**：第一人称眼镜信号（尤其眼动）在真实场景情感预测中比传统生理信号更有效。

## 方法详解

### 整体框架
43名被试佩戴Project Aria眼镜+生理传感器完成16项任务（9个诱导视频+7个自然活动），三个基准：(1)连续情感V/A/D二分类，(2)离散情绪9类，(3)人格Big Five二分类。

### 关键设计

1. **16项任务设计**:

    - Session A：9段~48s视频对应Mikels' Wheel 8种情绪+中性
    - Session B：Flappy Bird（挫折）、品尝难吃软糖（厌恶）、Jenga（紧张社交）、绘画+音乐（放松）、写悲伤信件（悲伤）、Slenderman恐怖游戏（恐惧）、讲笑话（娱乐）

2. **多层次标注体系**:

    - Emoti-SAM 7点量表收集V/A/D
    - 加权Mikels' Wheel：100%分配给9种情绪（10%步长），捕捉混合情绪的相对强度
    - BFI-2人格问卷

3. **传感器配置**:

    - Aria眼镜：眼动视频(640x480@90fps)、POV相机(1408x1408@10fps)、双IMU(1000Hz+800Hz)、鼻垫PPG(128Hz)
    - 外部：ECG(1024Hz)、EDA(256Hz)、RSP(400Hz)

4. **612维特征提取**:

    - ECG/PPG 77维、EDA 31维、RSP 14维
    - 眼动派生：瞳孔大小、像素强度、Fisherface、注视方向、眨眼检测、LBP-TOP微表情
    - 每个信号15个统计描述符

### 基线方法
- 连续情感：SVM-RBF + LOSO
- 离散情绪：Random Forest + SelectKBest(top-10) + LOSO
- 人格：Random Forest + SelectKBest + LOSO
- 深度学习：CNN和WER(Transformer)，5折CV

## 实验关键数据

### 模态对比主实验（F1 score）

| 基准任务 | 可穿戴(ECG/EDA/RSP) | 第一人称眼镜 | 全融合 | 随机基线 |
|----------|-------------------|------------|--------|---------|
| 连续情感(V/A/D均值) | 0.70 | **0.74** | **0.75** | 0.59 |
| 离散情绪(9类均值) | 0.24 | **0.46** | **0.46** | 0.11 |
| 人格特质(Big Five均值) | 0.50 | **0.57** | **0.59** | 0.53 |

### 经典方法 vs 深度学习

| 基准任务 | 经典方法(All) | CNN(All) | Transformer WER(All) |
|----------|-------------|----------|---------------------|
| 连续情感 | **0.75** | 0.68 | 0.60 |
| 离散情绪 | **0.46** | 0.22 | 0.21 |
| 人格特质 | **0.59** | — | 0.47 |

### 关键发现
- **第一人称眼镜信号全面优于传统生理传感器**：离散情绪任务优势最大（0.46 vs 0.24）
- 眼动注视(gaze)对连续情感最有效，像素强度对离散情绪最有效，IMU泛化性最好
- 经典ML大幅优于深度学习——小数据(43人x16任务)上深度模型过拟合严重
- 人格预测最困难，接近随机基线

## 亮点与洞察
- **加权Mikels' Wheel标注**：允许量化混合情绪的相对强度，比简单多选更rich。可迁移到视频情感分析标注
- **眼动视频>传统生理信号**：未来情感识别可能不需ECG/EDA接触式传感器，一副眼动追踪眼镜即可
- **Fisherface特征**：PCA+LDA应用于眼动视频帧作为低成本视觉描述符，效果良好

## 局限与展望
- 43名被试（主要大学生）样本量偏小
- 深度学习表现远不如经典方法，应探索few-shot/meta-learning小样本策略
- 标注仅在任务结束后收集，非连续时序标注
- 活动设计最大化减少身体运动，限制了日常活动场景推广
- 未探索大型视觉基础模型(VideoMAE/InternVideo)的表征能力
- 数据集中实验者在场（坐在帘子后面），可能影响被试的自然行为
- 被试均为健康人群，临床焦虑/抑郁人群的泛化性未知

## 相关工作与启发
- **vs DEAP/AMIGOS/ASCERTAIN**：这些经典情感数据集都在实验室环境、使用台式EEG/ECG设备，生态效度低。egoEMOTION首次用移动第一人称设备在半自然场景收集，更接近真实使用
- **vs Ego4D/EPIC-KITCHENS**：大型第一人称视觉数据集但无情感标注。egoEMOTION填补情感维度但规模远小于它们
- **vs eSEE-d**：唯一公开移动眼动情感数据集，但仅4类情绪、需下巴托固定、无生理信号。egoEMOTION在每个维度上显著扩展
- **vs K-EmoCon/EmoPairCompete**：在社交场景中自然诱导情绪，但缺乏第一人称视觉和眼动数据。egoEMOTION传感器覆盖更全面
- 未来可将egoEMOTION的标注方法用于Ego4D等大规模数据集的情感扩展标注
- 眼动特征的有效性暗示混合现实(MR)设备内置眼动追踪已足以支持实时情感推断

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个第一人称视觉+情感多模态数据集，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 612维特征、多模态对比、经典vs深度学习，但被试量偏小
- 写作质量: ⭐⭐⭐⭐⭐ 数据集论文非常清晰，实验设计描述详尽
- 价值: ⭐⭐⭐⭐ 开源数据集+基线对后续研究有推动，但规模限制直接应用
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

## 相关论文

- [egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks](../../ICCV2025/video_understanding/egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)
- [Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)
- [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/video_understanding/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)

<!-- RELATED:END -->
