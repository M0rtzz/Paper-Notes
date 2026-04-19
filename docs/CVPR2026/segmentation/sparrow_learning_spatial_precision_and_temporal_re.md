---
title: >-
  [论文解读] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
description: >-
  [CVPR 2026][图像分割][图像分割] 提出SPARROW框架，通过Target-Specific Tracked Feature注入时序参照一致性和BOX+SEG双提示初始化稳定像素定位，作为即插即用模块在三个视频MLLM基线上跨六个benchmark一致提升。
tags:
  - CVPR 2026
  - 图像分割
  - MLLM grounding
  - temporal consistency
  - 提示学习
---

# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)  
**代码**: 无  
**领域**: 视频理解 / 语义分割  
**关键词**: video segmentation, MLLM grounding, temporal consistency, dual-prompt, referring video object segmentation

## 一句话总结
提出SPARROW框架，通过Target-Specific Tracked Feature注入时序参照一致性和BOX+SEG双提示初始化稳定像素定位，作为即插即用模块在三个视频MLLM基线上跨六个benchmark一致提升。

## 研究背景与动机

**视频MLLM的时序漂移问题**：现有视频MLLM依赖静态文本定位token（如[SEG]）指示需分割的对象，但[SEG]仅提供"做什么"的语义线索，不包含对象位置和外观如何随时间变化的信息。模型必须完全从视觉线索推断运动和外观变化，导致空间漂移、身份切换和不一致的分割。

**首帧初始化不稳定**：[SEG]token只提供语义信息无空间先验，首帧mask经常与目标不对齐，这种错误随序列传播不断累积。一旦漂移开始，对象身份切换和参照不一致随之而来。

**现有方法的共性局限**：VideoGLaMM、UniPixel、GLUS等方法依赖逐帧语义和传播mask而非序列级参照线索，缺乏显式的时序identity维护机制。

## 方法详解

### 整体框架
SPARROW增强基线视频MLLM的两个互补模块：(1) Target-Specific Tracked Feature (TSF)——在训练时注入时序对齐的参照特征，教会模型identity持久性；(2) Dual-Prompt Grounding——联合解码[BOX]和[SEG]token，实现粗到细的空间定位和语义分割。

架构组成：双分支视觉编码器（空间SigLIP+时序InternVideo2）→ V→L适配器 → LoRA微调的LLM → L→V适配器（BOX/SEG） → SAM2像素解码器。所有新增模块即插即用，不修改基础backbone。

### 关键设计

1. **Target-Specific Tracked Feature (TSF)**：
    - 功能：在训练时注入时序对齐的参照对象特征，使模型学会跨帧identity持久性
    - 核心思路：给定文本query，用GroundingDINO在一帧检测对象 → CLDTracker跨序列传播 → K-means聚类（K=4）在联合视觉-空间特征空间中选择代表性样本 → 编码为TSF tokens $Z_{\text{TSF}}$ 拼接到LLM输入。测试时TSF默认关闭（无需外部检测/跟踪器）
    - 设计动机：TSF在训练时提供"这个对象在不同帧长什么样"的监督信号，让模型internalize时序参照能力。离线预计算解耦了重型模块和训练循环。K-means选择确保多样外观表示

2. **Dual-Prompt Grounding（BOX + SEG双提示）**：
    - 功能：LLM同时发射[BOX]和[SEG]token，前者提供空间先验，后者提供语义分割
    - 核心思路：[BOX]嵌入 $e_{\text{BOX}}$ 条件一个轻量回归头——在SAM2的Hiera特征上构建class-agnostic proposer（Deformable-DETR），生成300个候选框 → 通过交叉注意力 $A_i = \text{softmax}((W_q e_{\text{BOX}})(W_k F_i)^T/\sqrt{d})$ 用语言条件筛选 → 细化框坐标。[SEG]嵌入 $e_{\text{SEG}}$ 与筛选后的框 $\hat{b}_i$ 一起送入SAM2 prompt encoder产生mask
    - 设计动机：[BOX]的粗定位先验约束[SEG]的搜索空间，稳定首帧且允许漂移纠正。独立评分机制自然支持多实例查询（如"两个玩家"）。在任意帧重发[BOX]+[SEG]可实现无需外部跟踪器的漂移纠正

3. **两阶段训练策略**：
    - 功能：Stage 1训练TSF注入（多模态适配器+LoRA），Stage 2训练BOX提示（proposer预训练→filtration head微调）
    - 核心思路：Stage 1：在30,646视频/45,231 QA对上训练 $\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{BCE} + \mathcal{L}_{DICE}$，仅更新V→L适配器、L→V SEG适配器和LLM LoRA。Stage 2：先在COCO/Objects365/OpenImages/V3Det上预训练class-agnostic proposer，再微调filtration head $\mathcal{L}_{filter} = \lambda_{cls}\mathcal{L}_{BCE} + \lambda_{box}(\mathcal{L}_{\ell_1} + \mathcal{L}_{GIoU})$
    - 设计动机：两阶段解耦——Stage 1专注时序+语义对齐，Stage 2专注空间精度。渐进式训练避免多目标冲突

### 损失函数 / 训练策略
- Stage 1：$\mathcal{L}_{CE}$（语义对齐）+ $\mathcal{L}_{BCE} + \mathcal{L}_{DICE}$（mask监督）
- Stage 2 proposer预训练：$\mathcal{L}_{obj} + \lambda_1\mathcal{L}_{\ell_1} + \lambda_2\mathcal{L}_{GIoU}$
- Stage 2 filter微调：IoU>0.5为正样本，<0.2为负样本，$\lambda_{cls}=1.0, \lambda_{box}=2.0$
- TSF数据集：30,646视频+45,231 QA对，来自HC-STVG、VID-Sentence、A2D Sentences、LaSOT、MeViS、GOT-10k、Ref-SAV的统一整合

## 实验关键数据

### 主实验（适配三个基线的RVOS任务）

SPARROW作为即插即用模块分别适配到VideoGLaMM、UniPixel和GLUS三个SOTA视频MLLM，在MeViS、Ref-DAVIS17、Ref-YouTube-VOS等6个benchmark上均产生一致且显著的提升。

| 基线 → +SPARROW | 提升效果 |
|----------------|---------|
| VideoGLaMM → +SPARROW | 时序一致性和空间精度均提升 |
| UniPixel → +SPARROW | 身份切换显著减少 |
| GLUS → +SPARROW | 首帧定位稳定性改善 |

### 消融实验

| 组件 | 效果 |
|------|------|
| 基线 (无TSF无Dual-Prompt) | 基线 |
| + TSF (训练时) | 时序一致性显著提升 |
| + Dual-Prompt (BOX+SEG) | 空间精度显著提升 |
| + TSF + Dual-Prompt | **最优** |
| TSF在推理时使用 | 进一步小幅提升（但需额外开销） |

### 关键发现
- TSF训练后测试时默认不使用——模型已internalize时序参照能力，不依赖外部跟踪器
- Dual-Prompt的[BOX]先验对首帧稳定性提升最大，减少了误初始化的error cascade
- 跨三个不同架构的基线均一致提升，验证了模块化设计的通用性
- 多实例场景（如"两条狗"）中独立评分机制自然处理，无需额外标注

## 亮点与洞察
- **训练时注入、测试时无需**的TSF设计很优雅——用离线追踪数据教模型时序感知，部署时不增加开销
- BOX+SEG粗到细的双提示范式为视频MLLM的精确定位提供了新范式
- 即插即用设计使方法可立即应用于任何现有视频MLLM，降低了采用门槛

## 局限与展望
- TSF的离线数据构建依赖GroundingDINO和CLDTracker的质量——检测/追踪失败会引入噪声监督
- Stage 2的proposer预训练使用大规模检测数据集，可能限制在资源受限场景的复现
- 仅在RVOS和GCG任务上评估，扩展到视频QA、moment retrieval等任务的效果待验证
- K-means的K=4是经验值，不同复杂度的视频可能需要不同K

## 相关工作与启发
- **VideoGLaMM**：通过[SEG]token做逐帧SAM解码，SPARROW在此基础上增加时序和空间增强
- **Artemis**：启发了TSF的追踪特征注入思路
- **Groma**：启发了BOX提示的图像定位思路，SPARROW将其扩展到视频
- 启发：视频MLLM的"定位"和"跟踪"能力可通过即插即用模块独立增强，不需要从头设计新架构

## 评分
- 新颖性: ⭐⭐⭐⭐ TSF训练注入+Dual-Prompt双提示的组合设计新颖，即插即用模块化设计有影响力
- 实验充分度: ⭐⭐⭐⭐ 三个基线×六个benchmark的全面验证，模块化消融清晰
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，方法描述详细，图示清晰
- 价值: ⭐⭐⭐⭐ 对视频MLLM的时序一致性和空间精度提供了通用增强方案

<!-- RELATED:START -->

## 相关论文

- [SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)
- [Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)

<!-- RELATED:END -->
