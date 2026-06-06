---
title: >-
  [论文解读] Event-Level Detection of Surgical Instrument Handovers in Videos
description: >-
  [CVPR 2026][医学图像][surgical video] 提出面向真实手术视频中器械交接检测的时空视觉框架，结合 ViT 空间特征提取和单向 LSTM 时序建模，通过多任务学习联合预测交接事件和方向，在肾移植手术视频上达到 F1=0.84 的检测性能。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "surgical video"
  - "instrument handover"
  - "ViT-LSTM"
  - "multi-task"
  - "event detection"
---

# Event-Level Detection of Surgical Instrument Handovers in Videos

**会议**: CVPR 2026  
**arXiv**: [2604.07577](https://arxiv.org/abs/2604.07577)  
**代码**: 有  
**领域**: 医学图像  
**关键词**: surgical video, instrument handover, ViT-LSTM, multi-task, event detection

## 一句话总结

提出面向真实手术视频中器械交接检测的时空视觉框架，结合 ViT 空间特征提取和单向 LSTM 时序建模，通过多任务学习联合预测交接事件和方向，在肾移植手术视频上达到 F1=0.84 的检测性能。

## 研究背景与动机

手术器械交接的可靠监测对维持手术流程效率和患者安全至关重要。手术中器械交接失败可能导致残留器械等严重不良事件。从术中视频自动检测交接仍极具挑战：频繁遮挡、背景杂乱、动态光照、交接本身的时序演化特性使得单帧分析不够。

先前 SurgiGuard 利用 CLIP 特征和图推理检测交接，但主要依赖帧级特征，缺乏显式时序建模。本文引入 ViT+LSTM 的时空架构，在真实手术录像（而非模拟环境）上验证。

## 方法详解

### 整体框架

从视频中采样 8 帧序列（步长 4，覆盖 29 帧时域），ViT 独立提取每帧空间特征，线性投影后送入单向 LSTM 进行时序聚合，共享表示送入两个任务头。

### 关键设计

1. **ViT 空间特征提取**：使用预训练 ViT 骨干，冻结前 18 层 transformer，微调上层以适配交接分析任务。帧级特征投影到 64 维嵌入空间。

2. **LSTM 时序聚合**：选择单向 LSTM 而非 Transformer 时序模型，因为有标注数据规模小、事件分布稀疏，LSTM 的强序列归纳偏置更适合短交互序列建模。

3. **多任务联合预测**：共享表示送入二分类交接检测头（sigmoid）和方向分类头（softmax: 助手接收/助手递出）。联合优化避免级联管线的误差累积。

### 损失函数 / 训练策略

L = λ_det · L_det + λ_dir · L_dir。L_det 使用加权 BCE（处理正负样本不平衡），L_dir 使用加权 CE（仅在正样本上计算）。序列标签通过中心 5 帧的多数投票确定（类别：助手接收/助手递出/助手空闲）。事件级评估通过高斯平滑+峰值检测从序列级预测提取离散交接事件。训练时冻结 ViT 前 18 层，仅微调上层，帧级特征投影到 64 维嵌入空间后送入 LSTM。数据增强策略用于减少手术背景杂乱和遮挡的干扰。数据集包含 5 台肾移植手术的术中视频，共 484 个交接事件。

## 实验关键数据

### 主实验

| 模型 | 检测 F1 | 方向 Mean F1 |
|------|--------|------------|
| 多任务 ViT-LSTM | **0.84** | **0.72** |
| 单任务 ViT-LSTM | 0.79 | 0.63 |
| VideoMamba | 0.84 | 0.61 |

### 关键发现

- 多任务学习在检测（F1 0.84 vs 0.79）和方向分类（0.72 vs 0.63）上均优于单任务
- 与 VideoMamba 相比，检测性能相当但方向分类显著更优
- Layer-CAM 可视化显示模型正确关注手部-器械交互区域

## 亮点与洞察

- 在真实肾移植手术视频上的实际验证具有临床价值
- 事件级评估（而非帧级）更符合临床感知
- Layer-CAM 可解释性分析增强了临床可信度
- 统一的多任务损失避免了级联管线的误差累积，检测和方向分类共享统一的时空表示
- 选择单向 LSTM 而非 Transformer 时序模型的关键原因：有标注数据规模小、事件分布稀疏，LSTM 的强序列归纳偏置更适合短交互序列建模
- VideoMamba 基线的专门比较显示了不同时序建模策略的影响

## 局限与展望

- 数据集较小（5台手术、484个交接事件），泛化性需进一步验证
- 仅检测助手与主刀间的交接，未涵盖更复杂的多人交互
- 未与 SurgiGuard 等基于 CLIP+图推理的方法在相同数据集上直接对比
- 事件级评估的高斯平滑参数和峰值检测阈值需要根据手术类型调优
- 未探索双向 LSTM 或 Transformer 时序模型在更大数据集上的潜力
- 未利用器械跟踪等辅助信息增强交接检测
- 数据增强包括裁剪、翻转等策略，减少手术背景杂乱干扰
- 事件级评估方法对临床实际应用更有意义，避免帧级评估的高估问题

## 评分

- 新颖性：⭐⭐⭐ — 方法设计相对标准
- 技术深度：⭐⭐⭐ — ViT+LSTM+多任务组合直接
- 实验充分度：⭐⭐⭐ — 数据集规模有限，仅 5 台手术 484 个交接事件
- 实用价值：⭐⭐⭐⭐ — 手术安全应用场景明确，的临床可转化前景好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
