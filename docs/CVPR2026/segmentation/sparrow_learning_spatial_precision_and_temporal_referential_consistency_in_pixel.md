---
title: >-
  [论文解读] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
description: >-
  [CVPR 2026][图像分割][视频像素级定位] 提出 SPARROW 框架，通过 **目标特定追踪特征（TSF）** 注入时间一致性监督、**双提示（[BOX]+[SEG]）粗到细解码** 稳定首帧初始化，以即插即用方式集成到现有视频 MLLM 上，在 6 个基准 3 个任务上取得一致提升。
tags:
  - CVPR 2026
  - 图像分割
  - 视频像素级定位
  - 参考视频目标分割
  - 时间一致性
  - 双提示解码
  - 多模态大语言模型
---

# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)  
**代码**: 无  
**领域**: 语义分割  
**关键词**: 视频像素级定位, 参考视频目标分割, 时间一致性, 双提示解码, 多模态大语言模型

## 一句话总结

提出 SPARROW 框架，通过 **目标特定追踪特征（TSF）** 注入时间一致性监督、**双提示（[BOX]+[SEG]）粗到细解码** 稳定首帧初始化，以即插即用方式集成到现有视频 MLLM 上，在 6 个基准 3 个任务上取得一致提升。

## 研究背景与动机

### 1. 领域现状

多模态大语言模型（MLLMs）在图像级视觉推理与像素级定位方面取得进展，LISA / PixelLM 等方法通过 [SEG] token 实现语言条件化分割。但将这些方法扩展到**视频领域**时面临运动动态、遮挡和时间一致性等额外挑战。

### 2. 痛点

现有视频 MLLM（VideoGLaMM、UniPixel、GLUS）主要依赖**静态 [SEG] token** 进行逐帧推理：

- **时间漂移与身份切换**：文本提示是静态的，但视频是动态的，模型需要完全从视觉线索推断运动和外观变化，导致同一目标在不同帧间分割不一致
- **首帧初始化不可靠**：[SEG] token 仅提供语义线索而无空间先验，首帧 mask 可能与目标错位，误差逐帧累积

### 3. 核心矛盾

静态语义 token 无法编码目标随时间变化的位置与外观信息；首帧定位一旦出错，后续所有帧的分割都会受到误差传播影响。

### 4. 要解决什么

在不修改基础模型架构的前提下，同时解决 **(i) 时间参考一致性**（身份保持）和 **(ii) 首帧空间精度**（减少漂移）两个问题。

### 5. 切入角度

从追踪得到的目标特定特征中提炼时间监督信号（训练时注入，推理时可去除）；引入 [BOX] 几何先验与 [SEG] 语义先验的双提示协同解码机制。

### 6. 核心 idea

- **TSF**：离线用检测+追踪得到目标轨迹，K-means 选代表性子集，编码为 TSF token 注入训练，让模型学习身份持久性
- **双提示**：[BOX] 条件化类无关 proposal 产生空间先验 → [SEG] 在此基础上用 SAM2 精细分割，形成粗到细流程

## 方法详解

### 整体框架

SPARROW 的流程：双分支视觉编码器（空间 ℱg + 时间 ℱh）→ V→L 适配器 → LoRA 微调的 LLM → L→V 适配器 → SAM2 像素解码器。LLM 输出 [BOX] 和 [SEG] token，分别投影回视觉空间驱动 bounding box 回归和 mask 解码。所有新增模块均为即插即用，不修改骨干网络。

### 关键设计 1：目标特定追踪特征（TSF）

**功能**：在训练时为模型提供时间对齐的目标特定参考线索，使模型学会跨帧身份保持。

**核心思路**：
1. 给定文本 query，GroundingDINO 在某一帧检测目标，CLDTracker 跨帧传播得到候选框序列 B'_1...B'_K'
2. 在联合视觉-空间特征空间上做 K-means 聚类（K=4），选取最近质心的样本构成紧凑子集 B_1...B_K
3. 用图像编码器 ℱg 编码这些区域，经 V→L 适配器投影为 Z_TSF token，拼接到多模态输入中

**设计动机**：参考 Artemis 的发现——追踪目标特定特征可改善时间一致性。K-means 选择确保每个代表样本覆盖同一目标的不同外观，且减少冗余。关键点在于：**推理时默认不使用 TSF**（无需外部检测器/追踪器），模型已从训练中内化了时间一致性。

**数据集构建**：整合 HC-STVG、VID-Sentence、A2D Sentences、LaSOT、MeViS、GOT-10k、Ref-SAV 等多个公开数据集，统一为 30,646 个视频序列、45,231 个 Q&A 对，提供时间一致的轨迹、bbox 和分割 mask。

### 关键设计 2：双提示定位（Dual-Prompt Grounding）

**功能**：将 [BOX] 和 [SEG] 结合实现粗到细定位，稳定首帧并缓解漂移。

**[BOX] 分支**：
1. LLM 输出 [BOX] embedding e_BOX，经 L→V 适配器 Wb 投影
2. 基于 SAM2/Hiera 冻结特征构建类无关 proposer（Deformable-DETR 结构，单一 objectness head），生成 K=300 个 proposal
3. e_BOX 与每个 proposal 特征做 cross-attention 融合 → 经 filtration head 打分 → 对 top-M 候选做文本条件化 bbox 回归精修
4. 最终置信度融合语言和视觉分数，阈值过滤得到 B*

**[SEG] 分支**：
LLM 输出 [SEG] embedding e_SEG，与筛选后的 b_hat 组成 mask query 送入 SAM2 prompt encoder，每个空间先验产生实例级 mask。当 |B*|>1 时自然支持多实例输出。

**设计动机**：仅用 [SEG] 时首帧容易定位模糊；[BOX] 先提供几何约束，[SEG] 在此基础上做语义精修，两者互补。在任意帧重新发出 [BOX]+[SEG] 还可实现漂移校正。

### 损失函数 / 训练策略

**两阶段训练**：

**Stage 1 — TSF 信息注入**：训练 V→L 适配器 (Wg, Wh)、L→V SEG 适配器 Ws、LLM LoRA 参数，骨干和像素解码器冻结。损失：L_total = L_CE + L_BCE + L_DICE。

**Stage 2 — Box Prompt 学习**：
- 先独立预训练类无关 proposer（D-DETR head，在 COCO/Objects365/OpenImages/V3Det 上训练，丢弃类别标签）。损失：L_prop = L_obj + λ1·L_ℓ1 + λ2·L_GIoU
- 再微调 filtration head + L→V BOX 适配器 Wb，其余全部冻结。损失：L_filter = λ_cls·L_BCE + λ_box·(L_ℓ1 + L_GIoU)，其中 λ_cls=1.0, λ_box=2.0

## 实验关键数据

### 主实验

在 3 个视频 MLLM 基线（UniPixel、GLUS、VideoGLaMM）上分别集成 SPARROW，涵盖 RVOS、VG、GCG 三个任务。

**表 1：MeViS 参考视频目标分割（运动表达式）**

| 方法 | val J&F | val^u J&F |
|------|---------|-----------|
| UniPixel | 53.1 | 59.7 |
| + SPARROW | **54.4** (+1.3) | **60.7** (+1.0) |
| GLUS | 51.3 | 59.8 |
| + SPARROW | **53.2** (+1.9) | **61.9** (+0.3) |
| VideoGLaMM | 45.2 | 48.5 |
| + SPARROW | **47.5** (+2.3) | **57.4** (+8.9) |

**表 2：Ref-YTVOS & Ref-DAVIS17 参考视频目标分割**

| 方法 | Ref-YTVOS J&F | Ref-DAVIS17 J&F |
|------|---------------|-----------------|
| UniPixel | 70.5 | 74.2 |
| + SPARROW | **70.7** (+0.2) | **76.4** (+2.2) |
| GLUS | 67.3 | 72.9 |
| + SPARROW | **69.1** (+1.8) | **75.5** (+2.6) |
| VideoGLaMM | 66.8 | 69.5 |
| + SPARROW | **68.9** (+2.1) | **76.8** (+7.3) |

VideoGLaMM 在 Ref-DAVIS17 上 F（边界质量）提升高达 +14.5，所有集成 SPARROW 的模型 F 均超过 80。

**表 3：VideoGCG 视频定位对话生成**

| 方法 | mIoU | Recall | CLAIR |
|------|------|--------|-------|
| UniPixel | 52.0 | 0.311 | 26.0 |
| + SPARROW | **54.5** (+2.5) | **0.325** | **29.4** (+3.4) |
| VideoGLaMM | 62.34 | 0.375 | 28.2 |
| + SPARROW | **65.59** (+3.25) | **0.383** | **33.6** (+5.4) |

### 消融实验

基于 Ref-DAVIS17 (val) + VideoGLaMM 基线。

**TSF 与 BOX 联合消融**（J&F）：

| TSF 模式 | BOX OFF | BOX ON |
|----------|---------|--------|
| 无 TSF | 69.5 (baseline) | 72.5 (+3.0) |
| 仅训练时（默认） | 72.4 (+2.9) | **76.8** (+7.3) |
| 训练+推理 | 75.3 (+5.8) | **77.7** (+8.2) |

**提示组合消融**：[SEG] only 69.5、[BOX] only 68.2、[BOX]+[SEG] **72.5**（+3.0），证实双提示互补性。

### 关键发现

1. TSF **仅在训练时使用**即可带来 +2.9 提升，无需推理时的检测器/追踪器开销
2. [BOX] 提示单独贡献 +3.0，与 TSF 叠加呈**近似可加**效果（+7.3）
3. VideoGLaMM 上收益最大（MeViS val^u +8.9, Ref-DAVIS17 +7.3），说明基线越弱，改进空间越大
4. VidSTG（视觉定位）任务上三个基线均获得约 +5 mIoU 的一致提升

## 亮点与洞察

- **即插即用设计**：SPARROW 不修改基线骨干/LLM，仅通过轻量适配器和 proposal head 集成，成功应用于 3 个不同架构的视频 MLLM，具有很强的通用性
- **训练时追踪、推理时免追踪**：TSF 的核心洞察是利用伪追踪监督在训练阶段注入时间一致性先验，模型内化后推理时无需外部追踪器，大幅降低部署成本
- **粗到细双提示**：[BOX] 提供几何约束、[SEG] 提供语义精修，二者在信息维度上正交互补，类似于检测后分割的两阶段思路但通过 token 优雅实现
- **大规模数据集构建**：整合 7 个公开数据源形成 30K+ 视频的统一训练集，填补了目标中心时间定位数据的空白

## 局限与展望

1. **依赖 proposal 召回**：小目标、严重遮挡或未见类别若 proposal 未覆盖则无法恢复，recall 是瓶颈
2. **长视频误差累积**：早期 [BOX] 错误仍可能在长序列中传播，虽然双提示缓解了但未完全消除
3. **TSF 伪标签质量**：依赖 GroundingDINO + CLDTracker 生成伪追踪，严重噪声/ID 切换会影响训练
4. 未来方向：更高召回的 proposal 方法、在线校正机制、更强的追踪监督信号

## 相关工作与启发

- **Artemis**：启发了 TSF 中"追踪目标特定特征提升时间一致性"的思路
- **Groma**：启发了用 box prompting 增强细粒度视觉定位的双提示设计
- **VideoGLaMM / UniPixel / GLUS**：三个不同设计理念的基线，SPARROW 均能集成并提升，验证了通用性
- 与 SAM2 的结合方式值得借鉴：利用 Hiera 冻结特征做 proposal，保持 prompt encoder 接口不变

## 评分

⭐⭐⭐⭐ 工程导向强、方法模块化设计优雅、实验覆盖全面（3基线×6数据集），但核心技术（proposal+追踪伪标签）创新性中等，更多是已有组件的精巧组合。

<!-- RELATED:START -->

## 相关论文

- [Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)

<!-- RELATED:END -->
