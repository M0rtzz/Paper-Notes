---
title: >-
  [论文解读] Tracking and Understanding Object Transformations
description: >-
  [NeurIPS 2025][视频理解][目标跟踪] 提出 Track Any State 任务和 TubeletGraph 零样本框架，在视频中跟踪经历外观剧变的物体状态变化（如切苹果、蝴蝶从蛹中羽化），同时检测并描述这些变化。
tags:
  - NeurIPS 2025
  - 视频理解
  - 目标跟踪
  - 状态变化
  - 零样本
  - 时空分割
---

# Tracking and Understanding Object Transformations

**会议**: NeurIPS 2025  
**arXiv**: [2511.04678](https://arxiv.org/abs/2511.04678)  
**代码**: [有](https://tubelet-graph.github.io)  
**领域**: Video Understanding  
**关键词**: 目标跟踪, 状态变化, 视频理解, 零样本, 时空分割

## 一句话总结

提出 Track Any State 任务和 TubeletGraph 零样本框架，在视频中跟踪经历外观剧变的物体状态变化（如切苹果、蝴蝶从蛹中羽化），同时检测并描述这些变化。

## 研究背景与动机

现实世界中，物体频繁经历状态转换——苹果被切成碎片、蝴蝶从蛹中破壳而出。跟踪这些变化对理解物体和动态至关重要，但现有跟踪方法在物体发生变换后通常会丢失目标。

核心问题在于，所有主流的目标跟踪器（模板匹配、光流、SAM2 等）都依赖于物体外观的连续性假设。当物体经历状态变化时，外观可能发生剧烈变化（红苹果→白色果肉碎片、蛹→空壳+蝴蝶），导致跟踪器产生大量假阴性——模型判定原始物体"消失"了。

关键观察：状态变化导致的跟踪错误通常是单向的——当物体外观改变时，模型倾向于预测物体"缺失"（假阴性），而非错误地跟踪其他物体（假阳性）。这为恢复丢失目标提供了机会。

本文提出两个核心问题：
1. 如何在视频的指数级大搜索空间中找到变换后的缺失物体？
2. 如何建模底层变换并解决状态变化后的物体歧义性？

## 方法详解

### 整体框架

TubeletGraph 是一个零样本系统，包含四个步骤：(1) 将视频分割为时空 tubelet 集合；(2) 通过空间近邻性和语义一致性约束推理候选实体；(3) 提示多模态 LLM 描述变换；(4) 构建状态图。

### 关键设计

1. **时空分割（Spatiotemporal Partition）**：首先用 CropFormer 对首帧进行实体分割 $\mathcal{E}_1 = \text{CF}(I_1) \cup \{\mathcal{M}_1\}$，然后用 SAM2 将每个实体向前跟踪，形成初始 tubelet 集合。随着时间推移，出现无 tubelet 覆盖的区域时，在中间帧启动新的跟踪。这将"在每帧每个像素中寻找缺失物体"的连续问题转化为"哪个 tubelet 是真正缺失的物体"的离散问题，大幅缩小搜索空间。

2. **空间近邻性约束（Spatial Proximity）**：利用 SAM2 预测的多个候选掩码来估计变换物体可能出现的空间区域。定义 $S_{\text{prox}}(C,P) = \max_{j} |c_s \cap m_s^j| / |c_s|$，其中 $\{m_s^j\}$ 是 SAM2 在候选出现帧的三个候选掩码。阈值 $\tau_{\text{prox}}=0.3$。动机：变换后的物体在短时间内位置不会剧变。

3. **语义一致性约束（Semantic Consistency）**：使用 CLIP 的掩码池化特征计算语义相似度 $S_{\text{sem}}(C,P) = \max_{i,j} f(p_i, I_i) \cdot f(c_j, I_j)^T$。阈值 $\tau_{\text{sem}}=0.7$。动机：物体的身份和语义不会被变换根本改变（蛹可以变蝴蝶，但不会变鸟）。排除假阳性（如手、工具等）。

4. **状态图构建**：对每个满足约束的新候选 tubelet，将其出现视为状态变换的标记。在 tubelet 开始帧和首帧上绘制轮廓，提示 GPT-4.1 描述变换和物体身份，构建状态图。

### 损失函数 / 训练策略

TubeletGraph 是完全零样本的系统，无需训练。所有组件（SAM2.1-L、CropFormer-Hornet-3X、FC-CLIP-COCO、GPT-4.1）使用默认超参数。仅在 VOST 训练集上通过网格搜索确定 $\tau_{\text{prox}}=0.3$ 和 $\tau_{\text{sem}}=0.7$。

## 实验关键数据

### 主实验

| 方法 | 检测+描述变化 | VOST $\mathcal{J}$ | VOST $\mathcal{J}_{tr}$ | VSCOS $\mathcal{J}$ | M3-VOS $\mathcal{J}$ | DAVIS17 $\mathcal{J}$ |
|------|-------------|---------|-----------|---------|----------|----------|
| SAM2.1 | ✗ | 48.4 | 32.4 | 72.0 | 71.3 | 85.7 |
| SAM2.1 (ft) | ✗ | 54.4 | 36.4 | - | - | - |
| DAM4SAM | ✗ | 48.8 | 33.6 | 71.3 | 72.2 | 86.2 |
| **TubeletGraph** | **✓** | **50.9** | **36.7** | **75.9** | **74.1** | 85.6 |

### 消融实验

| 配置 | VOST $\mathcal{J}$ | 精确率 $\mathcal{P}$ | 召回率 $\mathcal{R}$ |
|------|---------|---------|---------|
| SAM2.1 基线 | 48.4 | 71.3 | 54.5 |
| +时空分割（全部加入） | 25.7 | 18.6 | 71.5 |
| +语义 | 49.2 | 63.7 | 64.8 |
| +近邻 | 50.7 | 67.7 | 63.8 |
| +近邻+语义 | **50.9** | 68.1 | 63.7 |

状态图评估（VOST-TAS）：时间定位精确率 43.1，召回 20.4；动作动词准确率 81.8，物体描述准确率 72.3。

### 关键发现

- SAM2 在变换物体上的精确率 (71.3%) 远高于召回率 (54.5%)，验证了"假阴性为主"的观察
- 时空分割本身能将召回率提升至 71.5（超过微调 SAM2 的 65.5），但精确率大幅下降
- 两个约束可大幅恢复精确率（+49.5）同时最小化召回损失（-7.8）
- 系统对阈值 $\tau_{\text{prox}}$ 和 $\tau_{\text{sem}}$ 高度鲁棒：在多个数据集上扫描后 $\mathcal{J}$ 变化范围小
- 替换 GPT-4.1 为 Qwen-2.5VL 会导致语义准确率剧降（动作 81.8→31.8），说明高质量 VLM 对语义描述至关重要

## 亮点与洞察

- **问题定义有价值**：Track Any State 将跟踪和状态变化理解统一为一个任务，输出兼具跟踪掩码和状态图
- **搜索空间约简巧妙**：时空分割将连续搜索转化为离散选择，是解决"变换后物体在哪"的优雅方案
- **新基准 VOST-TAS**：57 个视频实例、108 个变换、293 个标注结果物体，填补了该方向的评估空白

## 局限与展望

- 计算效率是瓶颈：构建时空分割平均每帧 7 秒（A6000 GPU），不适合实时应用
- 变换检测是被动的——仅在假阴性恢复时触发，无法检测不改变外观的变换
- 时间定位召回较低（20.4%），有较大提升空间
- 模块化设计可能带来系统性错误归因困难

## 相关工作与启发

本文揭示了一个有趣的洞察：现有跟踪器在物体变换时的失败模式是结构化的（以假阴性为主），而非随机的。这种结构化的失败模式可以被系统性地利用。TubeletGraph 的时空分割思想也可以用于多目标跟踪，且几乎不需要额外计算成本。对于机器人操作（如切割、折叠等任务的前后条件建模），这一框架有直接的应用价值。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （新任务定义 + 新方法 + 新基准）
- 实验充分度: ⭐⭐⭐⭐ （4 个跟踪数据集 + 状态图评估，消融清晰）
- 写作质量: ⭐⭐⭐⭐⭐ （行文流畅，问题引导清晰）
- 价值: ⭐⭐⭐⭐ （开辟新的研究方向，但计算成本限制了实际应用）

<!-- RELATED:START -->

## 相关论文

- [General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [MITracker: Multi-View Integration for Visual Object Tracking](../../CVPR2025/video_understanding/mitracker_multi-view_integration_for_visual_object_tracking.md)
- [MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](../../CVPR2025/video_understanding/must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)
- [FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](../../CVPR2025/video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4d-bench_benchmarking_multi-modal_large_language_models_for_4d_object_understand.md)

<!-- RELATED:END -->
