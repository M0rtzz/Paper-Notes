---
title: >-
  [论文解读] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants
description: >-
  [NeurIPS 2025][3D视觉][第一人称视觉] 提出IndEgo——首个面向真实工业场景的大规模多模态第一人称视觉数据集，包含3,460段自我中心录像（约197小时）和1,092段外部视角录像（约97小时），覆盖装配/拆卸、物流、检修、木工等五大类任务及协作场景，并建立了错误检测、推理问答和协作理解三项基准。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "第一人称视觉"
  - "工业场景"
  - "多模态数据集"
  - "协作工作"
  - "错误检测"
  - "视频问答"
---

# IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants

**会议**: NeurIPS 2025  
**arXiv**: [2511.19684](https://arxiv.org/abs/2511.19684)  
**作者**: Vivek Chavan (Fraunhofer IPK / TU Berlin), Yasmina Imgrund, Tung Dao, Sanwantri Bai, Bosong Wang, Ze Lu, Oliver Heimann, Jörg Krüger  
**代码**: [项目页](https://indego-dataset.github.io/) / [HuggingFace](https://huggingface.co/datasets/FraunhoferIPK/IndEgo)  
**领域**: 3D视觉  
**关键词**: 第一人称视觉, 工业场景, 多模态数据集, 协作工作, 错误检测, 视频问答  

## 一句话总结

提出IndEgo——首个面向真实工业场景的大规模多模态第一人称视觉数据集，包含3,460段自我中心录像（约197小时）和1,092段外部视角录像（约97小时），覆盖装配/拆卸、物流、检修、木工等五大类任务及协作场景，并建立了错误检测、推理问答和协作理解三项基准。

## 研究背景与动机

### 问题背景
第一人称视觉（Egocentric Vision）和AI助手是当前热点方向，其目标是开发能理解用户行为、意图并提供引导的智能助手。在工业环境中，工人需要完成复杂的手工操作、使用多种工具、在杂乱空间中导航，对第一人称视觉系统提出了独特挑战。

### 已有工作的不足
- 现有数据集（如EPIC-KITCHENS、Ego4D）**严重偏向日常/厨房场景**，工业场景极度缺乏
- 已有类工业数据集（Meccano仅7小时、Assembly101仅42小时）任务局限于桌面操作，未涵盖真实工业中的移动、搬运、协作等需求
- 缺乏**双人协作**数据——未来AI助手和具身智能体需要与人类协作完成任务
- 缺乏**多模态传感数据**（眼动、手势、运动轨迹、3D点云）的综合采集
- 现有数据集录像时长偏短，**长时任务**（20分钟以上）严重不足

### 核心动机
填补工业场景第一人称视觉数据集的空白，构建包含真实工业任务、双人协作、丰富多模态数据和长时录像的大规模基准，并评估SOTA多模态模型在该场景下的表现。

## 方法详解

### 数据集设计与采集

**任务类别**（5大类）：
1. **装配/拆卸**：机械设备和PC机箱的组装拆解，含引导式和无指导式
2. **物流与组织**：工具搬运、物品整理、仓储操作
3. **检修**：设备检查和故障修复（故障由实验人员预设）
4. **木工**：锉削、钻孔、拼接等基础到复杂的木工操作
5. **杂项**：穿戴PPE、急救、包装等

**协作模式**（3种）：
- 合作（Coworking）：双人作为平等伙伴共同完成任务
- 监督（Supervision）：专家指导新手操作
- 教学（Teacher-Student）：教师示范、学生跟随各自的独立设备

**硬件与多模态**：使用Meta Project Aria设备采集，传感器包括8MP RGB相机（2880×2880@10FPS）、SLAM相机、眼动跟踪、IMU等，外部视角使用Sony A6400等相机（1080p）。处理后的输出包括：眼动估计、手部姿态、半稠密3D点云和用户轨迹。

**数据规模**：
- 3,460段第一人称录像，共197.1小时，7.1M RGB帧
- 1,092段外部视角录像，共96.8小时，10.5M帧
- 20名参与者（15男5女），来自10个国家，经验从新手到专家
- 约34k条细粒度动作标注，含动词/名词/形容词的POS标注
- 标注一致性：关键步骤Krippendorff's α=0.97，细粒度动作α=0.54

### 基准任务设计

**错误检测（Mistake Detection）**：1,166段录像覆盖25个任务，包含正确操作和有意/无意的错误。错误类型包括跳步、顺序错误、多余步骤、安全违规等。引入严重性分级：Severe(2.3%)、Process Failure(18.7%)、Impact Future(7%)、Harm(5%)。

**推理问答（Reasoning-based QA）**：3,105个问答对，分为时间理解(14%)、情境推理(28%)、视觉识别(32%)和类比/溯因推理(26%)四类。

**协作任务理解**：预测穿戴者和同事在协作中的各自动作及角色关系。

## 实验关键数据

### 实验1：错误检测基准

| 方法 | 设定 | Precision | Recall | F1 | F1_Severe | F1_ProcessFail | F1_Harm |
|------|------|-----------|--------|-----|-----------|----------------|---------|
| QVL2.5 | Zero-shot | 15.9 | 50.1 | 24.1 | 38.8 | 36.5 | 34.1 |
| GFT (Gemini 2.0) | Zero-shot | 35.6 | 48.2 | **40.9** | **51.2** | **42.2** | **48.0** |
| QVL2.5 | MLP微调 | 31.4 | 51.6 | 39.1 | 42.6 | 39.8 | 44.0 |
| VL3 | Transformer微调 | 34.5 | 33.3 | 33.9 | 39.2 | 35.5 | 38.5 |
| QVL2.5 | 早期检测(50%帧) | 24.1 | 51.0 | 32.7 | 34.2 | 32.0 | 40.1 |

- 零样本设定中Gemini 2.0 Flash Thinking以40.9% F1显著领先，其他VLM仅约24%
- 微调后开源模型F1提升至~39%，但仍落后于Gemini零样本
- 早期检测（仅用50%帧）F1降至~31-33%，说明时序信息很关键
- 联合使用自我中心+外部视角可提升F1（GFT: 0.43→0.44）

### 实验2：推理问答基准

| 模型 | 时间理解 | 情境推理 | 视觉识别 | 类比推理 | 总准确率 |
|------|---------|---------|---------|---------|---------|
| VideoLLaMA3 8B | 52.2 | 60.3 | 59.4 | 57.5 | 58.2 |
| InternVL2.5 | 51.7 | 61.1 | 58.2 | 56.0 | 57.6 |
| Qwen2.5-VL 7B | 53.2 | 60.8 | 59.3 | 56.5 | 58.1 |
| Gemini 2.0 Flash | 55.4 | 62.1 | 67.2 | 68.3 | **64.1** |
| **人类** | **92.6** | **89.6** | **90.4** | **88.6** | **90.0** |

- SOTA模型最高准确率仅64.1%，与人类90%相差约26个百分点
- 时间理解（~52-55%）是所有模型的最大短板
- 纯文本模型Mistral-Large2+标签达到61.4%，说明当前VLM的视觉理解能力提升有限

### 模态消融实验

| 模型 | RGB | +Audio | +Gaze | +Audio+Gaze |
|------|-----|--------|-------|-------------|
| GFT | 0.38 | 0.41 | 0.39 | **0.42** |
| VL3 | 0.27 | 0.26 | 0.28 | **0.30** |
| IVL2.5 | 0.30 | 0.28 | 0.29 | 0.29 |

多模态融合对不同任务的增益高度依赖上下文，全模态组合整体最优。

## 亮点

- **首个真实工业场景大规模第一人称数据集**：覆盖5大工业类别，197小时自我中心+97小时外部视角，规模远超已有工业数据集（Meccano 7h、Assembly101 42h）
- **协作工作数据**：首次系统采集双人协作的多视角第一人称数据，包含合作/监督/教学三种模式，为AI协作助手研究奠定基础
- **丰富多模态标注**：结合眼动、手部姿态、3D点云、语音、动作标注等多模态信息，标注一致性高（α=0.97）
- **三项挑战性基准**：错误检测（含严重性分级）、推理问答、协作理解，均暴露了SOTA模型的显著不足
- **硬件可扩展性**：基于Project Aria设备，轻量穿戴式采集方案可推广到更多工业场景

## 局限与展望

- **单一场地**：数据仅在柏林Fraunhofer IPK采集，缺乏跨厂区和跨行业的泛化性验证
- **参与者数量有限**：20名参与者，性别比15:5反映工业现状但可能引入偏差
- **仅使用英语**：所有标注和口述均为英语，限制了多语言场景研究
- **最大录像时长受限**：受设备存储限制，单段最长约68分钟，无法覆盖真实工业中更长的连续流程
- **基准模型较少**：仅评估了4种VLM，缺乏对更大规模模型和专用视频模型的评估
- **3D点云和手部姿态**仅作为辅助输出提供，未深入用于基准任务的建模

## 与相关工作的对比

- **EPIC-KITCHENS**：厨房场景100小时，单模态（无眼动/运动/协作），IndEgo在场景多样性和模态丰富度上全面超越
- **Ego4D**：规模最大（3670小时）但场景偏向日常生活，无工业场景、无协作标注
- **Ego-Exo4D**：221小时双视角但场景以日常和运动为主，IndEgo专注工业且提供更丰富的错误检测和QA基准
- **Assembly101**：42小时类工业数据，但限于桌面组装，无协作、无眼动/音频
- **HoloAssist**：166小时辅助场景，有协作和关键步骤标注，但缺乏工业特定任务和错误严重性分级
- **Meccano**：仅7小时，规模和任务多样性均不及IndEgo

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个大规模真实工业第一人称数据集，填补重要空白，但数据集论文的方法创新有限
- 实验充分度: ⭐⭐⭐⭐ — 三项基准+模态消融+视角消融，评估全面；但基线模型种类偏少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据统计详实，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 对工业AI助手和具身智能体研究有重大推动作用，数据已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)
- [\[CVPR 2025\] HD-EPIC: A Highly-Detailed Egocentric Video Dataset](../../CVPR2025/3d_vision/hd-epic_a_highly-detailed_egocentric_video_dataset.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](../../CVPR2025/3d_vision/egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)

</div>

<!-- RELATED:END -->
