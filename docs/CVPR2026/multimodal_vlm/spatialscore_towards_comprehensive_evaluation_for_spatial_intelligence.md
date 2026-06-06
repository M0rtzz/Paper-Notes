---
title: >-
  [论文解读] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence
description: >-
  [CVPR 2026][多模态VLM][空间智能] 本文提出了目前最全面的多模态空间智能基准 SpatialScore（5K样本/30任务），并通过数据驱动的 SpatialCorpus（331K QA）微调方案和免训练的 SpatialAgent（12个工具）两条互补路径来提升 MLLM 的空间理解能力。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "空间智能"
  - "多模态评测"
  - "空间推理"
  - "Agent系统"
  - "空间语料库"
---

# SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence

**会议**: CVPR 2026  
**arXiv**: [2505.17012](https://arxiv.org/abs/2505.17012)  
**代码**: [https://github.com/haoningwu3639/SpatialScore/](https://github.com/haoningwu3639/SpatialScore/)  
**领域**: 多模态VLM  
**关键词**: 空间智能, 多模态评测, 空间推理, Agent系统, 空间语料库

## 一句话总结
本文提出了目前最全面的多模态空间智能基准 SpatialScore（5K样本/30任务），并通过数据驱动的 SpatialCorpus（331K QA）微调方案和免训练的 SpatialAgent（12个工具）两条互补路径来提升 MLLM 的空间理解能力。

## 研究背景与动机
1. **领域现状**：多模态大语言模型（MLLM）在语义问答、数学推理等任务上表现优异，但在空间智能方面的评估仍然碎片化且范围有限。
2. **现有痛点**：现有空间基准存在两大问题——（i）任务过于简单，主要关注粗粒度的空间关系（如物体存在/位置），忽视了严格的视觉几何感知（如相机位姿、动态感知）；（ii）评估范围窄，仅依赖简单的判断题、单模态输入或单一技能，无法全面衡量空间智能。
3. **核心矛盾**：传统计算机视觉已有成熟的几何优化工具和数学基础，但这些进展仍停留在纯视觉范式内，缺乏与语言的紧密集成和统一评估协议。
4. **本文目标**：（i）构建最全面的空间智能基准；（ii）广泛评估49个代表性MLLM；（iii）通过数据驱动和Agent两条路径提升空间推理能力。
5. **切入角度**：将语义理解和空间感知的融合视为下一个前沿，系统性地研究现有MLLM在何种程度上具备空间智能。
6. **核心idea**：提出覆盖30个任务的综合基准，配合大规模训练语料和多工具Agent系统，从评估和增强两个维度推动空间智能发展。

## 方法详解

### 整体框架
本文包含三部分贡献：（1）SpatialScore 基准——5025个人工验证样本，覆盖多种数据类型（真实/仿真/AIGC）、输入模态（图像/视频）和问答格式（判断/选择/开放）；（2）SpatialCorpus——331K 多模态QA训练资源；（3）SpatialAgent——配备12个专业空间感知工具的多Agent系统。

### 关键设计

1. **SpatialScore 基准构建**:
    - 功能：提供全面、多样化的空间智能评测
    - 核心思路：从已有3D数据集（ScanNet++、Omni3D等）中随机采样500个场景，利用精确3D标注生成开放式QA对，再通过LLM改写增加语言多样性。同时整合23个现有数据集的空间相关样本。最终通过GPT过滤和5位志愿者人工筛选，得到5025个高质量样本，覆盖10个大类30个具体任务。
    - 设计动机：现有基准要么任务过于简单，要么评估范围窄，无法全面衡量空间智能。本基准通过融合自建数据与现有数据集，实现了任务多样性和评估全面性的平衡。

2. **SpatialCorpus 训练资源**:
    - 功能：为MLLM提供大规模空间推理微调数据
    - 核心思路：利用2D模拟器和已有3D标注（ScanNet++、WildRGB-D、Omni3D、PointOdyssey等）构建331K多模态空间QA样本。支持对Qwen3-VL等模型进行有监督微调，显著提升空间推理任务性能。
    - 设计动机：仅靠评测无法提升模型能力，需要足够规模和质量的训练数据来弥合模型与人类在空间理解上的差距。

3. **SpatialAgent 多Agent系统**:
    - 功能：以免训练方式增强现有MLLM的空间推理能力
    - 核心思路：协调12个专业空间感知工具（深度估计器、相机位姿估计器、运动估计器等），支持两种推理范式——Plan-Execute（层次化分解子任务+顺序工具调用）和 ReAct（交错推理-行动的迭代式工具交互）。通过动态工具编排，在不进行额外训练的情况下提升空间理解能力。
    - 设计动机：数据驱动方法需要额外训练成本，Agent方案提供了即插即用的轻量级替代方案，两者互补。

### 损失函数 / 训练策略
SpatialCorpus 微调采用标准的有监督微调范式；SpatialAgent 为免训练方案，不涉及额外的损失函数设计。

## 实验关键数据

### 主实验

| 模型 | Overall | Mental Anim. | Counting | Depth Est. | Obj-Dist | Camera |
|------|---------|-------------|----------|------------|----------|--------|
| Human | 86.60 | 96.87 | 89.72 | 82.33 | 78.96 | 86.89 |
| GPT-5 (Text-only) | 30.62 | 18.79 | 20.34 | 29.36 | 24.20 | 32.01 |
| Qwen3-VL-2B | 41.41 | 35.35 | 52.74 | 34.64 | 35.42 | 30.59 |
| InternVL3-1B | 33.03 | 26.85 | 47.69 | 24.74 | 24.02 | 25.71 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Qwen3-VL + SpatialCorpus | Overall 显著提升 | 数据驱动微调有效 |
| SpatialAgent (Plan-Execute) | 显著优于基础模型 | 免训练范式可行 |
| SpatialAgent (ReAct) | 与Plan-Execute互补 | 适合需要迭代推理的场景 |

### 关键发现
- 即使最强的现有模型在SpatialScore上也远未达到人类水平（86.60 vs 最高约50+），说明空间智能仍是巨大挑战。
- 纯文本GPT-5的表现接近随机水平（30.62 vs 28.29），证实了视觉信息对空间推理的必要性。
- SpatialCorpus微调和SpatialAgent都能显著提升性能，两者互补。
- 相机位姿/运动类任务最具挑战性，模型与人类差距最大。

## 亮点与洞察
- **评测规模前所未有**：30个任务、5025个样本、49个模型的系统评估，为空间智能研究提供了坚实基础。
- **双路径提升策略**：数据驱动和Agent免训练方案互补，可根据场景灵活选择。
- **3D数据重用pipeline**：将3D标注转化为QA格式的流程可迁移到其他领域。

## 局限与展望
- 基准仍以静态评测为主，缺乏交互式空间推理的评估。
- Agent系统依赖外部工具的准确性，工具失败会级联影响结果。
- 未来可扩展到具身AI和自主导航的实际场景评测。

## 相关工作与启发
- **vs VSI-Bench/STI-Bench**: 这些基准仅覆盖少量任务和格式，SpatialScore在规模和多样性上全面超越。
- **vs OmniSpatial**: 虽然任务数量多（50个），但样本量小（1533个），SpatialScore在质量和平衡性上更优。

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统性整合+新构建的评测基准，贡献巨大但核心技术创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 49个模型评测+人类基线+多条提升路径验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，数据翔实
- 价值: ⭐⭐⭐⭐⭐ 空间智能领域的重要基础设施工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICML 2026\] ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](../../ICML2026/multimodal_vlm/revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](../../ICML2026/multimodal_vlm/thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
