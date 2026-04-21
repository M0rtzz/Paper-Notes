---
title: >-
  [论文解读] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark
description: >-
  [ICLR 2026][3D视觉] 提出首个夜间第一人称视觉基准 EgoNight，包含日夜对齐视频和 3658 个人工验证 QA 对，揭示 MLLM 在低光照下存在高达 32.8% 的性能下降。
tags:
  - ICLR 2026
  - 3D视觉
---

# EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark

**会议**: ICLR 2026  
**arXiv**: [2510.06218](https://arxiv.org/abs/2510.06218)  
**代码**: https://github.com/dehezhang2/EgoNight  
**领域**: 3D视觉 / 第一人称视觉  

## 一句话总结

提出首个夜间第一人称视觉基准 EgoNight，包含日夜对齐视频和 3658 个人工验证 QA 对，揭示 MLLM 在低光照下存在高达 32.8% 的性能下降。

## 研究背景与动机

### 领域现状

第一人称视觉理解近年来取得了显著进展，大规模数据集如 EPIC-KITCHENS、Ego4D、Ego-Exo4D 推动了动作识别、物体检测、视频问答等任务的发展。MLLM（如 GPT-4V、Gemini、Qwen-VL）已在视频理解中展现出强大能力，专用的第一人称 MLLM（如 EgoGPT、Exo2Ego）也相继出现。

### 现有痛点

几乎所有现有的第一人称视觉数据集和基准都局限于白天或光线良好的场景，忽视了夜间低光照这一在现实应用中不可避免的场景。这导致当前模型在夜间环境下的鲁棒性完全未知，严重限制了智能助手、导航系统等在夜间场景中的实际部署。

### 核心矛盾

现实中，第一人称视觉系统（如智能导航助手）必须在夜间运行，面临低光照、不均匀照明和严重受限的可见度等挑战。然而，缺乏适当的夜间基准使得研究者既无法评估模型的夜间性能，也无法有针对性地改进。此外，夜间标注本身因低可见度而极其困难，难以保证标注质量。

### 本文方案

提出 EgoNight，首个系统性的夜间第一人称视觉基准。核心创新在于引入日夜对齐视频：利用 Blender 合成精确对齐的日夜视频对（EgoNight-Synthetic），设计视频引导录制策略采集真实世界的日夜对齐视频（EgoNight-Sofia），并整合现有的夜间数据（EgoNight-Oxford）。基于此构建了 EgoNight-VQA 基准及两个辅助任务。

---

## 方法详解

### 整体框架

EgoNight 由三部分组成：(1) 视频源采集——合成（50 对）、真实录制（20 对）和现有数据（20 段）；(2) EgoNight-VQA 基准——12 种 QA 类型、3658 个 QA 对，通过三阶段日增强自动标注流水线生成；(3) 两个辅助基准——日夜对应检索和夜间深度估计。

### 关键设计一：日夜对齐视频采集策略

**合成对齐（EgoNight-Synthetic）**: 使用 Infinigen 生成多样化室内 3D 场景，人工标注员清理场景并模拟行走轨迹，通过 Blender 在相同轨迹下分别渲染白天和夜间版本，确保像素级精确对齐。共 50 对视频，覆盖 100+ 环境素材和 50+ 物体类别。

**真实对齐（EgoNight-Sofia）**: 设计视频引导录制策略——首先录制白天视频，夜间录制时在手机上播放白天视频作为视觉引导，帮助穿戴者匹配步速、视点和动作。后期修剪进一步优化时空一致性。共 20 对视频，覆盖公寓、办公室、超市、街道等多种场景。

### 关键设计二：日增强自动 QA 生成流水线

三阶段流水线利用白天视频辅助夜间标注：

1. **夜间描述生成**: 针对目标 QA 类型，提示 GPT-4.1 为夜间片段生成详细描述
2. **夜间问题生成**: 将描述与夜间片段输入 MLLM 生成多样化问题候选
3. **日增强伪答案合成**: 对配对类型，利用白天片段生成更准确的答案；非配对类型直接从夜间片段推断

所有生成的 QA 对经过人工标注员的三轮操作（删除/修改/新增）精炼，每个 QA 对至少经过一次人工验证，累计投入 300+ 小时人工工作量。

### 关键设计三：多样化 QA 类型体系

定义 12 种 QA 类型，分为：

- **配对类型**（8 种）：物体识别、文本识别、空间推理、场景序列、导航、静态计数、动作识别、非常识推理
- **非配对类型**（4 种）：光照识别、光照变化、动态检测、动态计数

其中导航、场景序列、光照识别/变化和非常识推理是本文新提出的任务类型。

---

## 实验关键数据

### 主实验

评估 10 个 SOTA MLLM 在 EgoNight-VQA 上的表现：

| 模型 | Synthetic (夜) | Sofia (夜) | Oxford (夜) | 平均准确率 |
|------|--------------|-----------|------------|----------|
| GPT-4.1 | 30.73% | 26.33% | 35.72% | **30.93%** |
| Gemini 2.5 Pro | 27.18% | 25.00% | 33.21% | 28.46% |
| InternVL3-8B | 19.28% | 17.10% | 23.80% | **20.06%** |
| Qwen2.5-VL-72B | 18.56% | 16.73% | 22.41% | 19.23% |
| Qwen2.5-VL-7B | 14.58% | 13.28% | 16.71% | 14.86% |
| EgoGPT | 12.88% | 14.03% | 15.95% | **14.29%** |

日夜性能差距：EgoNight-Synthetic 上平均下降 **32.8%**，EgoNight-Sofia 上平均下降 **25.0%**。

### 消融实验 / 深入分析

| 微调策略 | Synthetic 准确率 | Real 准确率 | 提升幅度 |
|---------|----------------|------------|---------|
| Zero-shot（基线） | 14.83% | - | - |
| 全模型微调 | **24.67%** | **21.88%** | +9.84% |
| 仅视觉编码器 | 19.23% | 18.56% | +4.40% |
| 仅 LLM | 21.15% | 19.02% | +6.32% |
| 合成数据训练→真实测试 | - | 20.57% | +5.74% |

**关键发现**：

1. 合成数据与真实数据高度相关（Pearson $r = 0.9359$，$p = 6.847 \times 10^{-5}$），合成数据微调可有效提升真实场景性能
2. 感知类任务在日间表现更好但夜间下降更大，推理类任务整体更难但受光照影响相对较小
3. 新提出的 QA 类型（光照识别、导航、非常识推理）对现有 MLLM 构成极大挑战
4. 辅助任务中，GPT-4.1 在空间检索上达到 80%+ 准确率，但在时间定位上表现不佳；鱼眼深度估计模型优于通用模型

---

## 亮点与洞察

- 填补了夜间第一人称视觉理解的空白，日夜对齐设计精巧，使性能差距可定量分析
- QA 类型设计全面，新增导航、光照识别等多种此前未探索的任务维度
- 日增强标注流水线巧妙利用白天信息辅助夜间标注，兼顾效率与质量
- 合成数据与真实数据高度相关（$r = 0.9359$），验证了合成数据的研究价值

## 局限与展望

- 数据规模较小（90 个视频，3658 QA 对），与大规模基准相比存在差距
- 合成数据占比较高（约 55%），可能不完全反映真实世界复杂性
- 仅评估了 VQA 及两个辅助任务，未涵盖更多夜间第一人称任务
- 日增强标注策略依赖 GPT-4.1，生成质量受限于该模型能力

## 相关工作与启发

- **vs Ego4D/EPIC-KITCHENS**: 这些大规模第一人称数据集均聚焦白天场景，EgoNight 是首个专注夜间的基准
- **vs NightBench**: NightBench 关注一般夜间图像理解，EgoNight 专注第一人称视角且提供日夜对齐
- **vs 低光照增强方法**: 传统方法聚焦像素级增强，本文关注语义级理解差距

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统性夜间第一人称视觉基准
- 实验充分度: ⭐⭐⭐⭐ 覆盖 10 个 MLLM，含微调分析和辅助任务
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据呈现规范
- 价值: ⭐⭐⭐⭐ 填补重要研究空白，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations](egoworld_translating_exocentric_view_to_egocentric_view_using_rich_exocentric_ob.md)

<!-- RELATED:END -->
