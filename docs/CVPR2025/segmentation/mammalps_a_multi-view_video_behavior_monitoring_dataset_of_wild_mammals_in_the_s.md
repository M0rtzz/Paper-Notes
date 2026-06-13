---
title: >-
  [论文解读] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps
description: >-
  [CVPR 2025][语义分割][相机陷阱] 本文提出 MammAlps——一个来自瑞士国家公园的多模态多视角野生哺乳动物行为监测数据集（8.5 小时稠密标注，5 个物种，11 种活动 + 19 种动作），以及两个基准任务：多模态物种+层级行为识别（B1）和首个多视角长期事件理解（B2），填补了野生动物视频行为分析在层级行为标注、多模态和多视角方面的空白。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "相机陷阱"
  - "野生动物监测"
  - "多视角视频"
  - "层级行为识别"
  - "多模态"
---

# MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps

**会议**: CVPR 2025  
**arXiv**: [2503.18223](https://arxiv.org/abs/2503.18223)  
**代码**: [https://github.com/eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps)  
**领域**: 图像分割  
**关键词**: 相机陷阱, 野生动物监测, 多视角视频, 层级行为识别, 多模态

## 一句话总结
本文提出 MammAlps——一个来自瑞士国家公园的多模态多视角野生哺乳动物行为监测数据集（8.5 小时稠密标注，5 个物种，11 种活动 + 19 种动作），以及两个基准任务：多模态物种+层级行为识别（B1）和首个多视角长期事件理解（B2），填补了野生动物视频行为分析在层级行为标注、多模态和多视角方面的空白。

## 研究背景与动机

1. **领域现状**：野生动物行为监测对生态保护至关重要。相机陷阱（camera traps）是非侵入式监测的主要手段，已积累海量视频数据。深度学习模型被用于自动化分析物种识别和行为分类。
2. **现有痛点**：(1) 现有野生动物视频数据集大多只有单一物种、单一视角、平面行为标注（独立类别无层级）; (2) 网络爬取的数据集（如 MammalNet）与实际相机陷阱数据域差异大; (3) 尚无数据集同时提供多模态（视频+音频+分割图）和多视角信息; (4) 行为标注缺乏层级结构（activity vs action 分层）
3. **核心矛盾**：机器学习社区需要更接近真实场景、标注更丰富的野外数据集来开发实用的动物行为识别模型，但采集和标注成本极高。
4. **本文目标**：构建一个兼具多模态、多视角、层级行为标注的相机陷阱视频数据集，并提出两个互补的基准任务——短片段行为识别和长期事件理解。
5. **切入角度**：利用瑞士国家公园 3 个采样点各 3 个相机陷阱（共 9 个）的部署，获得天然的多视角数据；结合神经行为学的层级行为分类理论（activity 由多个 action 组成）进行标注。
6. **核心 idea**：提供首个集多模态、多视角、层级行为标注于一体的野外相机陷阱数据集，搭配行为识别和长期事件理解双任务基准。

## 方法详解

### 整体框架
MammAlps 的构建流程为：(1) 数据采集——9 个相机陷阱在瑞士国家公园 3 个站点录制 6 周; (2) 数据预处理——视频分组为事件、MegaDetector 动物检测、ByteTrack 跟踪、人工校正; (3) 标注——个体级物种和双层行为标注（activity + action）+ 事件级个体计数和气象条件; (4) 基准任务——B1 多模态行为识别 + B2 多视角长期事件理解。

### 关键设计

1. **层级行为标注体系**:
    - 功能：提供比现有数据集更结构化的行为描述
    - 核心思路：基于动物行为学理论，将行为分为两层：(1) Actions（动作，如 walking, grazing）——通常几帧即可识别的基本运动; (2) Activities（活动，如 foraging, courtship）——需要更长时空上下文、可能由多个 action 组成或涉及个体间交互。每帧标注一个 activity 和一到两个 non-mutually exclusive 的 action
    - 设计动机：现有数据集将所有行为视为独立类别（flat taxonomy），忽略了行为的组合与层级关系。分层标注更符合行为生态学需求，也让模型可以在不同粒度上进行预测

2. **多模态输入设计（视频+音频+参考场景分割图）**:
    - 功能：为行为识别提供多种互补信号
    - 核心思路：(1) 视频——动物个体的裁剪跟踪片段; (2) 音频——与视频同步的声音频谱图，通过 AudioMAE tokenizer 处理; (3) 参考场景分割图——每个相机位置固定，标注一张场景分割图（10 类，如 grass、water 等），然后根据动物轨迹生成与视频同步的分割图 clip。三种模态通过改进的 VideoMAE 架构融合
    - 设计动机：音频可捕获发声、行走等行为信号；场景分割图提供环境上下文（如动物在水源旁→可能在饮水），这些信息在纯视频输入中可能缺失

3. **多视角长期事件理解基准（B2）**:
    - 功能：从多个相机的长期录像中提取事件级别的综合信息
    - 核心思路：(1) 将同一站点的多个相机录像按时间对齐分组为"事件"; (2) 使用 Token Merging（ToME）在空间和时间维度压缩视频 token——先在帧内减少 patch token（reduction factor r），再在时间维度按固定时长块合并; (3) 添加三种位置编码：相机 ID、相对时间、源帧/patch 位置; (4) 所有压缩后 token 输入 Transformer encoder，通过四个输出头分别预测物种、活动、个体数量和气象条件
    - 设计动机：现有模型无法直接处理 multi-view + 长时间（最长 12 分钟）的生态数据。离线 token merging 方案在可控 context length 内保留关键信息，是实用的 workaround

### 损失函数 / 训练策略
B1 任务：基于 VideoMAE 的多任务联合训练，采样策略按标签频率逆加权以处理类别不平衡。测试时对每个 clip 采样 10 次取平均。指标用各任务的 mAP。B2 任务：Transformer encoder + 四头多任务训练，使用 ViT-Base 预训练 encoder。

## 实验关键数据

### 主实验

**B1: 多模态行为识别（VideoMAE, 联合预测）**:

| 模态 | Species mAP | Activity mAP | Action mAP | Avg |
|------|------------|-------------|------------|-----|
| Video (V) | 0.495 | 0.436 | 0.452 | 0.453 |
| Audio (A) | 0.223 | 0.212 | 0.172 | 0.192 |
| V+A | 0.473 | 0.484 | 0.466 | 0.473 |
| V+A+S | 0.531 | 0.485 | 0.437 | 0.466 |

### 消融实验

**B2: 多视角长期事件理解**:

| 配置 | Species | Activity | Met. Cond. | Individuals | Avg |
|------|---------|----------|-----------|-------------|-----|
| Joint, r=14 | 0.415 | 0.479 | 0.618 | 0.499 | 0.489 |
| Joint, r=11 | 0.446 | 0.481 | 0.594 | 0.543 | 0.500 |
| Single task (best) | 0.481 | 0.478 | 0.681 | 0.592 | - |

**位置编码消融（B2, 个体计数任务）**：加入 CamID + 时间 + 源帧位置编码后，mAP 提升最高达 +0.109，证明位置信息对多视角理解至关重要。

### 关键发现
- **视频+音频融合最有效**：V+A 组合比纯视频提升 +0.020 mAP，视频+音频+分割图虽然在物种上最好（0.531），但平均不如 V+A
- **分割图模态有条件有用**：对纯音频有显著增益（+0.111），但对视频+音频组合反而轻微降低，可能是当前融合策略不够优
- **单任务 vs 多任务**：联合训练在大部分情况下不弱于单任务训练，且 B1 中联合训练对所有任务均有提升
- **B2 task 非常有挑战**：最长事件 12 分钟，物种和活动严重不平衡，跨相机计数需要多视角融合

## 亮点与洞察
- **层级行为标注**：首次在野生动物数据集中引入 activity-action 双层标注，与行为学理论对齐。这种分层标注方法可以迁移到人体动作识别——将 "cooking" 分解为 "chopping"、"stirring" 等子动作
- **参考场景分割图作为模态**：利用相机固定位置的特性，用一张分割图为所有视频提供环境上下文信息，成本极低但信息丰富。这个想法可迁移到任何固定摄像头的监控场景
- **离线 Token Merging 处理长视频**：用 ToME 在空间+时间维度压缩视频 token 到可控 context length，简洁有效地解决了长视频理解的工程问题

## 局限与展望
- 数据规模有限（仅 8.5 小时标注视频，5 个物种），相比 MammalNet（394 小时，173 物种）差距明显
- 地理范围窄（仅瑞士国家公园），泛化性有待验证
- 夜间视频使用红外拍摄质量较差，但论文未单独分析夜间数据的性能
- 相机间时间漂移可达 1 分钟，B2 任务中多视角对齐精度受影响
- 基线模型（改进 VideoMAE + Transformer encoder）偏简单，更强的多模态/多视角模型（如 video-LLM）可能大幅提升性能

## 相关工作与启发
- **vs MammalNet**: MammalNet 规模大但来源杂（纪录片等），域差异大; MammAlps 规模小但是真实相机陷阱数据，标注更细（个体级、双层行为、多模态）
- **vs KABR/BaboonLand**: 都有个体级行为标注，但使用无人机而非相机陷阱，且只有单一模态和 flat 行为分类
- **vs PanAf/LoTE**: 相机陷阱数据但行为标注在视频级而非个体级，MammAlps 的标注粒度更细

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个集多模态+多视角+层级行为标注于一体的野外数据集，B2 任务设计新颖
- 实验充分度: ⭐⭐⭐ 基线实验设计合理但模型偏简单，缺少与更多 SOTA 方法的对比
- 写作质量: ⭐⭐⭐⭐ 数据集论文结构完整，术语定义清晰（Table 2），统计分析详尽
- 价值: ⭐⭐⭐⭐ 对野生动物行为监测领域有重要贡献，但数据规模限制了直接影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring](ripvis_rip_currents_video_instance_segmentation_benchmark_for_beach_monitoring_a.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[CVPR 2025\] Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)

</div>

<!-- RELATED:END -->
