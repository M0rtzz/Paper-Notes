---
title: >-
  [论文解读] IDEA-Bench: How Far are Generative Models from Professional Designing?
description: >-
  [CVPR 2025][benchmark][image generation] 提出 IDEA-Bench，包含 100 个真实世界专业设计任务和 275 个测试案例的综合基准，系统评估生成模型在专业级图像设计场景中的能力，揭示当前最强模型仅达 22.48 分。
tags:
  - CVPR 2025
  - benchmark
  - image generation
  - professional design
  - MLLM evaluation
  - generative model
---

# IDEA-Bench: How Far are Generative Models from Professional Designing?

**会议**: CVPR 2025  
**arXiv**: [2412.11767](https://arxiv.org/abs/2412.11767)  
**代码**: [https://github.com/ali-vilab/IDEA-Bench](https://github.com/ali-vilab/IDEA-Bench)  
**领域**: image_generation  
**关键词**: benchmark, professional design, image generation, evaluation, MLLM, visual effects, storyboard

## 一句话总结

提出首个面向专业级图像设计的综合基准 IDEA-Bench，涵盖 100 个真实设计任务（海报、绘本、字体、特效等）和 5 种输入输出模式，揭示当前最强模型仅获 22.48/100 分，距离专业设计仍有巨大鸿沟。

## 研究背景与动机

**领域现状**: DALL-E 3、FLUX-1 等 T2I 模型在学术基准上表现优异，每日吸引百万用户。GenEval、DreamBooth 等基准覆盖了基本的文生图和单图编辑评估。

**现有痛点**: (1) 现有基准仅关注孤立的学术任务（如 T2I 对齐、简单编辑），与真实专业设计需求脱节；(2) 提示词过短（平均<11词），远低于专业设计师使用的长详细指令；(3) 缺乏多图输入/输出的评估维度；(4) FID/CLIPScore 等传统指标无法捕捉审美、上下文和多模态整合的细微差异。

**核心矛盾**: 专业设计师仍然依赖 Photoshop 等传统工具，说明生成模型在处理复杂多元的专业任务上能力严重不足，但缺乏系统性的评估框架来量化这种差距。

**本文切入角度**: 从真实设计平台和专业设计师处收集任务，按模型能力层级分类，建立多层次评估体系。

## 方法详解

### 整体框架

1. **任务收集**: 从互联网设计平台和专业设计师处获取 100 个代表性任务
2. **分类体系**: 按输入/输出模式分 5 大类 — T2I、I2I、Is2I、T2Is、I(s)2Is
3. **标注流程**: GPT-4o 生成任务定义和提示，人工设计 6 道分层评估题（基础→质量→细节）
4. **评估**: 人工评估（全集）+ MLLM 自动评估（18任务子集 IDEA-Bench-mini）

### 关键设计

**1. 五级任务分类体系**
- **T2I（文生图）**: 11 个任务，含海报、名片、游戏 UI、LOGO 等长提示场景（平均 138.68 词，vs 现有基准<11词）
- **I2I（图生图）**: 包装渲染、图片修图、风格迁移、打光调整等 13 个任务
- **Is2I（多图生图）**: 品牌周边生成、角色融合等多参考图输入任务
- **T2Is（文生多图）**: 多视角生成、绘本创作等需要一致性的多图输出
- **I(s)2Is（多图生多图）**: 分镜设计、角色集生成等最复杂的任务
- **设计动机**: 随着统一生成模型的发展，需要涵盖从简单到复杂的完整能力谱。

**2. 分层二值评估体系**
- **功能**: 每个 case 6 道二值判定题（0/1），分 3 层：基础任务理解(Q1-2) → 完成质量(Q3-4) → 细节审美(Q5-6)。
- **核心规则**: 层级依赖 — 如果低层未满分，高层自动为 0 分。
- **设计动机**: 优先考察任务完成度而非审美，符合专业设计标准（先做对再做美）。

**3. MLLM 自动评估（IDEA-Bench-mini）**
- **功能**: 在 18 个代表性任务上使用 Gemini 1.5 Pro 自动评分，每个 case 评 3 次取平均。
- **核心思路**: 针对每个 case 定制评估问题（而非共享），并通过人工迭代校准使 MLLM 评分与人工一致。
- **设计动机**: 解决 MLLM 对图像顺序敏感、多图理解不可靠等问题。

### 提示词策略

对不支持多图生成的模型（如 FLUX-1、SD3），用 GPT-4o 将多模态输入重述为每张图的独立提示，使基础 T2I 模型也能参与多图生成任务的评测。

## 实验关键数据

### 主实验 — 全类别得分

| 模型 | T2I | I2I | Is2I | T2Is | I(s)2Is | **Avg** |
|---|---|---|---|---|---|---|
| FLUX-1† | 46.06 | 12.13 | 4.89 | 20.15 | 29.17 | **22.48** |
| SD3† | 24.04 | 10.79 | 4.69 | 21.59 | 13.06 | 14.83 |
| DALL-E 3† | 24.34 | 6.95 | 5.27 | 14.36 | 14.44 | 13.07 |
| OmniGen† | 21.41 | 8.17 | 2.77 | 23.52 | 21.39 | 15.45 |
| Emu2† | 17.98 | 7.05 | 8.98 | 15.53 | 12.78 | 12.46 |
| Emu2 (原生) | 17.98 | 7.05 | 8.98 | – | – | 6.81 |
| Anole (7B) | 0.00 | 0.64 | 0.00 | 1.74 | 0.00 | 0.48 |

（†表示使用 GPT-4o 重述提示词适配所有任务）

### T2I 子任务

| 模型 | 建筑 | 名片 | 游戏UI | 信息图 | 海报 | **Avg** |
|---|---|---|---|---|---|---|
| FLUX-1 | 100 | 38.89 | 5.56 | 0 | 56.67 | 46.06 |
| DALL-E 3 | 22.22 | 0 | 0 | 0 | 23.33 | 24.34 |
| Anole | 0 | 0 | 0 | 0 | 0 | 0 |

### 关键发现

1. **专业设计能力严重不足**: 最强模型 FLUX-1 仅获 22.48 分，距离及格线（60分）差异巨大。
2. **通用模型 vs 专用模型的反转**: 最优通用模型 Emu2 仅 6.81 分，不如经过提示重述的 T2I 模型。
3. **多图生成是最大短板**: Is2I 类别所有模型得分 < 9 分，说明多参考图理解几乎不可用。
4. **FLUX-1 在 T2I 上一枝独秀**: 在纯文本任务（如建筑风格 100 分）表现出色，但图像引导任务急剧下降。
5. **Anole 几乎全零**: 多模态交错生成模型在专业设计任务上完全不可用。
6. **提示长度对模型是挑战**: IDEA-Bench 平均提示长度 138.68 词，远超现有基准的 <11 词，暴露了长提示遵循能力的不足。

## 亮点与洞察

- 首次系统性地将专业设计任务引入生成模型评估，填补了学术基准与实际需求的鸿沟
- 五级任务分类 + 分层二值评估的设计方法论值得其他基准借鉴
- 通过 GPT-4o 提示重述使基础 T2I 模型也能参与多图任务，扩大了可评估模型范围
- 揭示了一个重要观察：生成模型在"做对"任务前就已失败，审美提升是次要问题
- MLLM 自动评估 + 人工校准的闭环方案具有实用性

## 局限与展望

- 100 个任务中许多对现有模型过难，导致大量 0 分，区分度不够
- 人工评估主观性仍然存在，特别是 Q3-6 的质量和审美判断
- MLLM 自动评估仅覆盖 18 个子任务，覆盖率需扩大
- 缺乏对专业设计师的 human baseline 分数，难以量化人机差距
- 分层评估的严格层级依赖可能过于惩罚小错误

## 相关工作与启发

- **GenEval**: 6 个 T2I 评估任务，提示词简短，本文在复杂度上大幅升级
- **ImagenHub**: 7 种任务但仍局限于学术定义
- **DEsignBench**: 关注设计场景但任务范围和模型能力覆盖不如本文
- **启发**: 生成模型基准需要从"能生成什么"转向"能完成什么设计任务"的思维转变

## 评分

⭐⭐⭐⭐ — 首个面向专业设计的系统性基准，任务设计精心、评估体系合理，对领域发展具有重要指导意义；但部分任务对现有模型难度过高，导致评分缺乏区分度。

<!-- RELATED:START -->

## 相关论文

- [AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [Image Generation Diversity Issues and How to Tame Them](image_generation_diversity_issues_and_how_to_tame_them.md)
- [Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](../../ICLR2026/image_generation/blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)
- [Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)

<!-- RELATED:END -->
