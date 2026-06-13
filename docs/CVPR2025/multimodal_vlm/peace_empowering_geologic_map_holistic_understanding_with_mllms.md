---
title: >-
  [论文解读] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs
description: >-
  [CVPR 2025][多模态VLM][地质图理解] 本文构建了首个地质图理解基准 GeoMap-Bench（5 种能力、25 个任务、3864 个问题），并提出 GeoMap-Agent（层级信息提取 + 领域知识注入 + 增强问答），在地质图理解上以 0.811 的整体得分大幅超越 GPT-4o 的 0.369。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "地质图理解"
  - "多模态大模型"
  - "AI智能体"
  - "领域知识注入"
  - "基准测试"
---

# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs

**会议**: CVPR 2025  
**arXiv**: [2501.06184](https://arxiv.org/abs/2501.06184)  
**代码**: 无  
**领域**: 人体理解 / 多模态VLM  
**关键词**: 地质图理解, 多模态大模型, AI智能体, 领域知识注入, 基准测试

## 一句话总结
本文构建了首个地质图理解基准 GeoMap-Bench（5 种能力、25 个任务、3864 个问题），并提出 GeoMap-Agent（层级信息提取 + 领域知识注入 + 增强问答），在地质图理解上以 0.811 的整体得分大幅超越 GPT-4o 的 0.369。

## 研究背景与动机

**领域现状**：地质图是地质学的基础图表，在灾害检测、资源勘探和土木工程中至关重要。多模态大语言模型（MLLM）在通用图像理解上已展现强大能力，但在地质图理解上表现很差。

**现有痛点**：地质图理解面临三大挑战：(1) 超高分辨率——地质图分辨率可达 $10,000^2$ 像素，远超 MLLM 的处理能力；(2) 多组件关联——图名、图例、主地图、剖面图、柱状图等多个组件相互关联；(3) 领域专业知识——包含符号化的地质对象和多样化的视觉表示，需要地质、地理、地震学等跨学科知识。目前没有任何基准或方法专门针对地质图理解。

**核心矛盾**：即使是经验丰富的地质学家也难以快速从外部数据源检索和关联知识（如地质、地理和地震数据），更不用说 AI 模型。而 MLLM 虽有通用图像理解能力，但缺乏制图概括所需的特殊处理能力。

**本文目标**：(1) 构建地质图理解的全面评估基准；(2) 设计专门的 AI 智能体实现地质图问答和分析。

**切入角度**：受人类科学家跨学科协作的启发，设计一个 AI 专家组作为顾问，利用多样化的工具池（检测、OCR、分割等）综合分析问题。

**核心 idea**：通过层级信息提取将地质图数字化为结构化数据，注入领域知识库增强推理，最后用增强提示驱动 MLLM 回答问题。

## 方法详解

### 整体框架
GeoMap-Agent 包含三个模块串联工作：(1) HIE（层级信息提取）将地质图的各组件（标题、图例、主地图等）检测、裁剪并通过 OCR 和分割工具提取为结构化元数据；(2) DKI（领域知识注入）将提取的元数据与外部地质知识库关联，注入岩性、地层年代等专业信息；(3) PEQA（提示增强问答）将结构化信息和领域知识组织为增强提示，引导 MLLM（GPT-4o）生成准确回答。

### 关键设计

1. **层级信息提取（Hierarchical Information Extraction, HIE）**:

    - 功能：将超高分辨率的地质图数字化为结构化、可查询的元数据。
    - 核心思路：首先检测地质图的各个组件（标题、比例尺、图例、主地图、索引图、剖面图、柱状图）的位置和边界框。然后对每个组件进行细粒度提取：图例中每个单元的颜色、文字、岩性和地层年代；主地图中的断层线和岩层分布；剖面图中的地层结构信息。使用专门的检测、OCR 和颜色分析工具构建层级化的信息结构。
    - 设计动机：地质图的超高分辨率使得直接输入 MLLM 不可行（信息会被压缩丢失），需要先将其分解为组件级别的信息。层级结构保证了信息的完整性和可查询性。

2. **领域知识注入（Domain Knowledge Injection, DKI）**:

    - 功能：将提取的地质元数据与外部地质知识库关联，提供 MLLM 缺乏的专业背景知识。
    - 核心思路：构建地质领域知识图谱，包含地层年代表、岩石分类体系、构造地质学知识等。将 HIE 提取的元数据（如岩性代码、地层符号）映射到知识图谱中对应的条目，获取详细的地质含义。例如，将图例中的 "Qal" 映射为 "第四纪冲积层"，并关联其形成环境和可能含有的矿产资源。
    - 设计动机：地质图使用高度符号化的编码系统（颜色-地层对应、缩写-岩性对应），这些专业知识无法从图像本身推断，必须从外部知识源获取。

3. **提示增强问答（Prompt-enhanced Question Answering, PEQA）**:

    - 功能：将结构化信息和领域知识组织为有效的提示，引导 MLLM 生成准确、详细的回答。
    - 核心思路：根据问题类型（提取/定位/引用/推理/分析），选择性地组织相关的结构化信息和知识作为上下文，构建增强提示。类似"AI 专家组"的设计，不同模块的输出作为不同专家的意见汇总后提供给 MLLM。
    - 设计动机：直接把问题扔给 MLLM 效果很差（GPT-4o 仅 0.369），需要将复杂的地质图理解问题转化为 MLLM 擅长的"给定上下文做推理"的形式。

### 损失函数 / 训练策略
GeoMap-Agent 是推理时的智能体系统，不涉及训练。GeoMap-Bench 包含 124 张地质图和 3864 个问题，来源于 USGS（英文）和 CGS（中文），涵盖提取、定位、引用、推理和分析五种能力的 25 个任务。

## 实验关键数据

### 主实验
GeoMap-Bench 上各模型的表现：

| 模型 | 提取 | 定位 | 引用 | 推理 | 分析 | 整体 |
|------|------|------|------|------|------|------|
| GPT-4o（直接） | 低 | 低 | 低 | 低 | 低 | **0.369** |
| Gemini-1.5-Pro | 低 | 低 | 低 | 低 | 低 | 较低 |
| Qwen-VL | 更低 | 更低 | 更低 | 更低 | 更低 | 更低 |
| **GeoMap-Agent** | **高** | **高** | **高** | **高** | **高** | **0.811** |

### 消融实验

| 配置 | 整体得分 | 说明 |
|------|---------|------|
| GPT-4o baseline | 0.369 | 无辅助直接理解 |
| + HIE | 提升显著 | 结构化信息帮助提取和定位 |
| + HIE + DKI | 进一步提升 | 领域知识帮助推理和分析 |
| + HIE + DKI + PEQA | **0.811** | 提示工程最大化 MLLM 能力 |

### 关键发现
- MLLM 在地质图理解上表现远不如人类——GPT-4o 仅 0.369，暴露了这些模型在专业领域的巨大不足
- HIE 模块贡献最大，将地质图数字化为结构化数据是关键突破
- 领域知识注入对推理和分析类问题的提升最为显著——这些问题需要专业背景知识
- 中文地质图（CGS 源）的理解难度高于英文（USGS 源）

## 亮点与洞察
- **首个地质图理解基准与智能体**：GeoMap-Bench 填补了地质图 AI 理解的评估空白，25 个任务覆盖了从基础提取到高级分析的完整能力谱
- **工具增强的智能体范式**：不是训练一个端到端模型，而是通过检测+OCR+知识图谱等工具组合增强存在的 MLLM，这种思路对其他专业领域图表理解有直接借鉴意义
- **结构化中间表示**：将非结构化的高分辨率图像转化为结构化元数据再进行推理，有效绕过了 MLLM 分辨率限制

## 局限与展望
- GeoMap-Bench 目前仅 124 张地质图，规模有限
- HIE 的组件检测依赖于地质图的标准格式，对非标准格式的适应性未验证
- 仅测试了 USGS 和 CGS 两个来源，其他国家/地区的地质图可能有不同绘制标准
- 智能体的推理延迟较高（需要多步工具调用），实时性有待改善

## 相关工作与启发
- **vs GeoBench/K2**: 这些工作关注纯文本的地质知识问答，不涉及地质图（图像）理解。GeoMap-Bench 是首个多模态地质基准
- **vs GeoGPT**: GeoGPT 使用 GIS 工具处理地理空间任务，但不处理地质图。GeoMap-Agent 专门针对地质图信息提取和问答
- **vs LHRS-Bench**: LHRS-Bench 评估遥感图像理解，但遥感图像与地质图在内容和复杂性上有本质不同

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统化定义地质图理解问题，基准和智能体都是原创贡献
- 实验充分度: ⭐⭐⭐⭐ 多模型对比+消融充分，但数据规模可扩大
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，但方法描述可更精炼
- 价值: ⭐⭐⭐⭐ 对地质学领域的 AI 应用有重要推动作用，智能体范式可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](../../NeurIPS2025/multimodal_vlm/gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[NeurIPS 2025\] Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention](../../NeurIPS2025/multimodal_vlm/dont_just_chase_highlighted_tokens_in_mllms_revisiting_visual_holistic_context_r.md)
- [\[CVPR 2025\] SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](segagent_exploring_pixel_understanding_capabilities_in_mllms_by_imitating_human_.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[NeurIPS 2025\] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](../../NeurIPS2025/multimodal_vlm/dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)

</div>

<!-- RELATED:END -->
