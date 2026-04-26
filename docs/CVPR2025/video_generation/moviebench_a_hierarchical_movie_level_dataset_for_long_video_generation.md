---
title: >-
  [论文解读] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation
description: >-
  [CVPR 2025][长视频生成] 本文提出MovieBench——首个面向电影级长视频生成的层次化数据集，提供电影-场景-镜头三级标注（含角色肖像、字幕和音频），并基于此定义了四个基准任务（文本到关键帧、身份定制长视频、关键帧条件视频、音频驱动说话人生成），揭示了现有模型在多场景叙事一致性上的重大挑战。
tags:
  - CVPR 2025
  - 长视频生成
  - 电影级数据集
  - 层次化标注
  - 角色一致性
  - 多场景叙事
---

# MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation

**会议**: CVPR 2025  
**arXiv**: [2411.15262](https://arxiv.org/abs/2411.15262)  
**代码**: https://github.com/showlab/MovieBench (有)  
**领域**: 视频理解  
**关键词**: 长视频生成, 电影级数据集, 层次化标注, 角色一致性, 多场景叙事

## 一句话总结
本文提出MovieBench——首个面向电影级长视频生成的层次化数据集，提供电影-场景-镜头三级标注（含角色肖像、字幕和音频），并基于此定义了四个基准任务（文本到关键帧、身份定制长视频、关键帧条件视频、音频驱动说话人生成），揭示了现有模型在多场景叙事一致性上的重大挑战。

## 研究背景与动机

**领域现状**：视频生成领域近年取得了巨大进展，Stable Video Diffusion、CogVideo、SORA等模型能从文本生成高质量短视频片段。数据方面，WebVid-10M、Panda-70M、InternVid等大规模视频-文本数据集推动了模型训练，但这些数据集主要面向短视频（平均几秒到几十秒）。

**现有痛点**：当前视频生成研究聚焦于单场景短视频，而电影级长视频生成面临三大未解问题：(1) 多场景间的叙事连贯性——不同场景需要讲述一个完整故事；(2) 角色外观一致性——同一角色在不同场景中需保持相同外观；(3) 音频连续性——对话和背景音需跨场景协调。关键障碍在于，**没有一个公开数据集**同时提供这三方面的标注。

**核心矛盾**：现有视频数据集要么规模大但标注粗（只有片段级描述），要么标注细但面向理解而非生成。MAD、AutoAD等电影级数据集专注于影片理解（检索、描述），其标注格式不适合指导生成模型训练。

**本文目标**：构建一个专为电影级视频生成设计的层次化数据集，包含丰富的角色信息、连贯的故事线和分层标注结构，并基于此建立标准化评测基准。

**切入角度**：电影天然具备多场景叙事、角色一致性和音频同步等特征，是研究长视频生成的理想素材。作者利用自动化pipeline从电影中提取分层标注，降低人工标注成本。

**核心idea**：用"电影-场景-镜头"三级层次结构组织视频数据和标注，为长视频生成研究提供从粗到细的完整信息链。

## 方法详解

### 整体框架
MovieBench并非一个模型工作，而是一个数据集+基准框架。核心pipeline包括：(1) 电影数据收集与预处理；(2) 三级层次化标注生成；(3) 角色信息库构建；(4) 四个基准任务的定义与评测。输入为完整电影视频和字幕文件，输出为结构化的层次标注数据。

### 关键设计

1. **三级层次化数据结构 (Hierarchical Data Structure)**:

    - 功能：提供从全局到局部的多粒度视频描述信息
    - 核心思路：Movie-level提供电影概述（约43.4K词/部电影，时长约45.6分钟），包含故事梗概、主要角色、风格等全局信息；Scene-level对应电影中的场景段落（平均263.6词，15.4秒），标注场景主题、涉及角色和空间关系；Shot-level是最细粒度（平均66.3词，4.09秒），提供镜头的详细动作、对话和视觉描述。这三级信息通过自动化pipeline生成，利用了字幕对齐和场景检测技术
    - 设计动机：不同粒度的生成任务需要不同层次的信息。关键帧生成需要场景级信息，镜头衔接需要镜头级细节，而角色一致性维护需要电影级角色库。现有数据集只提供单一粒度，无法同时支撑这些需求

2. **角色信息库 (Character Bank)**:

    - 功能：为每部电影建立完整的角色档案，包含角色名、肖像图片和配音音频
    - 核心思路：从电影中自动提取主要角色的面部图像（多角度、多表情），通过人脸检测和聚类算法将不同场景中的同一角色关联起来，形成角色身份库。同时提取角色的对话音频片段，用于音频驱动的生成任务。Ours版本包含肖像和音频，Ours++扩展版进一步扩大了数据规模（116.8小时 vs 69.2小时）
    - 设计动机：角色一致性是电影级视频生成最大的挑战之一。现有数据集不提供角色级标注，模型无法学习到"同一角色在不同场景应保持外观一致"这一关键约束

3. **四个基准任务定义 (Four Benchmark Tasks)**:

    - 功能：全面评估长视频生成的不同能力维度
    - 核心思路：Task 1 (Text→Keyframe/Storyboard) 从电影级文本描述生成关键帧序列，测试叙事理解和视觉规划能力；Task 2 (Identity-Customized Long Video) 给定角色肖像和场景描述，生成保持角色一致性的多场景视频；Task 3 (Keyframe-conditioned Video) 从关键帧序列生成连贯视频，测试帧间过渡和动态生成能力；Task 4 (Audio-driven Talking Human) 用音频驱动角色说话视频生成
    - 设计动机：电影级视频生成不是单一任务，而是多个子能力的综合。分解为四个任务可以独立评估各能力，也便于分步研发

### 数据集统计与对比
MovieBench在关键维度上与现有数据集形成互补：总视频时长69.2小时（Ours++为116.8小时），分辨率1080p，提供角色肖像+音频+字幕三重标注，是唯一同时覆盖电影级/场景级/镜头级三层结构的数据集。相比WebVid-10M（360p、无角色信息）和Panda-70M（720p、无角色信息），MovieBench虽然规模较小但标注密度远高。

## 实验关键数据

### 数据集规模对比

| 数据集 | 总时长 | 分辨率 | 角色信息 | 层次结构 | 文本来源 |
|--------|--------|--------|---------|---------|---------|
| WebVid-10M | 52Khr | 360p | ✗ | 仅Shot级 | Alt-Text |
| Panda-70M | 167Khr | 720p | ✗ | 仅Shot级 | 生成 |
| InternVid | 371.5Khr | 720p | ✗ | 仅Shot级 | 生成 |
| MiraData | 16Khr | 1080p | ✗ | 仅Shot级 | 生成 |
| MovieBench | 69.2hr | 1080p | ✓(肖像+音频) | 三级层次 | 生成 |
| MovieBench++ | 116.8hr | 1080p | ✓(肖像+音频) | 三级层次 | 生成 |

### 基准任务实验结果

| 任务 | 方法 | 角色一致性 | 叙事连贯性 | 视觉质量 |
|------|------|----------|----------|---------|
| Task 2 身份定制 | DreamVideo | 低 | 中 | 中 |
| Task 2 身份定制 | Magic-Me | 低 | 中 | 中 |
| Task 3 关键帧条件 | StoryDiffusion+CogVideoX | 低 | 低-中 | 中 |
| Task 3 关键帧条件 | StoryDiffusion+Kling 1.5 | 中 | 中 | 高 |
| Task 4 音频驱动 | Hallo2 (Source Image) | 高 | - | 高 |
| Task 4 音频驱动 | Hallo2 (Text Conditional) | 低 | - | 中 |

### 关键发现
- 所有现有方法在多场景角色一致性上都表现不佳：即使使用身份定制方法（DreamVideo、Magic-Me），跨场景的角色外观仍难以保持稳定，尤其是在角色姿态和光照变化大的场景中
- 闭源商业产品（Kling 1.5）在视觉质量上显著优于开源模型（CogVideoX），但在角色一致性维持上同样面临困难
- 多角色场景是最大挑战：当一个场景中出现3+角色时，所有方法都容易出现角色混淆或外观漂移
- 文本条件生成比图像条件生成的角色一致性差距很大，说明当前模型从文本到角色特征的映射能力不足

## 亮点与洞察
- **层次化组织思路很有启发**：将电影内容按"电影-场景-镜头"三级组织，为不同粒度的生成和理解任务提供了对应的supervision信号。这种层次化设计思路可以推广到其他长序列生成任务（如长文档、长对话）
- **角色信息库的构建**是关键贡献：以角色为中心的数据组织方式填补了现有数据集的重要空白，是实现角色一致性生成的数据基础。角色面部聚类+跨场景关联的pipeline具有很好的可复用性
- **揭示了"长视频=多个短视频的衔接"范式的根本不足**：实验清楚表明，简单串联短视频生成模型无法解决电影级挑战，需要从本质上重新设计考虑全局一致性的生成架构

## 局限与展望
- 数据集规模相对较小（69.2小时），相比WebVid-10M等大规模数据集在训练方面的作用有限
- 自动标注pipeline的质量依赖于底层模型（场景检测、人脸识别等），可能存在噪声
- 评测以定性对比为主，缺乏定量的一致性和叙事质量指标的标准化定义
- 涉及版权电影内容，数据集的学术使用需当心法律风险
- 四个任务的baseline较简单，缺少专门针对电影生成设计的方法作为strong baseline

## 相关工作与启发
- **vs MAD / AutoAD**: 这些数据集同样基于电影，但面向理解任务（检索、描述）。MovieBench在设计上面向生成，标注格式更适合作为生成模型的训练信号
- **vs MiraData**: MiraData也面向视频生成，提供高质量标注，但缺少角色信息和层次结构。MovieBench在标注丰富度上显著超越
- **vs WebVid-10M / Panda-70M**: 这些大规模数据集适合预训练但粒度粗。MovieBench适合作为微调或评测数据集与它们互补使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个面向电影级长视频生成的层次化数据集，四个任务定义清晰
- 实验充分度: ⭐⭐⭐ 以定性分析为主，定量评估不够系统化
- 写作质量: ⭐⭐⭐⭐ 数据集描述清晰，任务定义明确
- 价值: ⭐⭐⭐⭐ 填补了长视频生成领域数据集和基准的重要空白

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[CVPR 2025\] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](../../ICCV2025/video_generation/long_context_tuning_for_video_generation.md)

<!-- RELATED:END -->
