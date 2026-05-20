---
title: >-
  [论文解读] From Charts to Code: A Hierarchical Benchmark for Multimodal Models
description: >-
  [ACL 2026][代码智能][图表生成] 本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型…
tags:
  - "ACL 2026"
  - "代码智能"
  - "图表生成"
  - "代码生成"
  - "多模态基准"
  - "层次化评估"
  - "视觉忠实度"
---

# From Charts to Code: A Hierarchical Benchmark for Multimodal Models

**会议**: ACL 2026  
**arXiv**: [2510.17932](https://arxiv.org/abs/2510.17932)  
**代码**: [GitHub](https://github.com/CSU-JPG/Chart2Code)  
**领域**: 代码智能 / 多模态理解  
**关键词**: 图表生成, 代码生成, 多模态基准, 层次化评估, 视觉忠实度

## 一句话总结

本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。

## 研究背景与动机

**领域现状**：图表是科学论文和商业报告中最重要的可视化工具。随着大型多模态模型（LMM）的快速发展，AI 系统不仅能理解图表，还能生成可执行的绑图代码（chart-to-code），有望大幅提升生产力。

**现有痛点**：(1) 现有基准（如 ChartMimic）已趋于性能饱和——GPT-4o 在 ChartMimic 上已达 82.2%，无法区分当前和未来模型的能力；(2) 缺乏覆盖真实使用场景的系统性基准——用户不仅需要复现图表，更常需要编辑图表（换类型、加元素）和从原始长表格生成图表，但这些场景未被充分测试；(3) 现有评估主要关注代码正确性，忽略了渲染出的图表的视觉忠实度。

**核心矛盾**：现有基准报告的高分与模型在实际使用中的表现之间存在巨大差距——模型在简单复现上得分很高，但在更常见的编辑和数据转图表场景中表现大幅下降，现有基准无法暴露这些问题。

**本文目标**：(1) 构建一个从用户视角出发的层次化图表代码生成基准；(2) 设计多层次评估协议（代码级 + 图表级）；(3) 全面评测 29 个 SOTA 多模态模型。

**切入角度**：从真实用户工作流出发设计三个递进难度级别——简单复现→复杂编辑→长表格转图表——逐步增加对模型理解、推理和代码生成能力的要求。

**核心 idea**：将图表代码生成从单一的复现任务扩展为覆盖完整用户工作流的层次化基准，同时引入代码级和图表级双层评估来全面衡量生成质量。

## 方法详解

### 整体框架

Chart2Code 基准的形式化定义为 $C = f(R, I, D)$，其中 $R$ 是参考图表，$I$ 是用户自然语言指令，$D$ 是可选数据源（支持文本、表格截图、Excel 文件三种模态），$C$ 是 LMM ($f$) 生成的可执行 Python 代码。评估分为代码级（规则化 Base 评分 + LLM 评分）和图表级（LMM 视觉评分 + 人类评估）。

### 关键设计

1. **三级层次化任务设计**:

    - 功能：覆盖从简单到复杂的完整图表代码生成场景
    - 核心思路：Level 1（图表复现）分为纯视觉复现 DR（仅给图表图像）和带数据的风格迁移 CRD/CFD（给图表+数据），共 863 个任务；Level 2（图表编辑）要求基于参考图表进行复杂修改（换图表类型、加趋势线、计算相关系数、按类别拆分等），共 1,010 个任务；Level 3（长表格转图表）要求从原始长表格（平均 2,647 行）中提取、计算并生成图表，共 313 个任务
    - 设计动机：用户实际工作流远不止简单复现——编辑和数据可视化是更常见且更困难的需求，现有基准完全忽略了这些场景

2. **多层次评估协议**:

    - 功能：从代码和视觉两个角度全面评估生成质量
    - 核心思路：(a) 代码级 Base 评分——解析 Matplotlib 等库的 Figure 对象，提取颜色、网格、布局、图例、视觉元素、数据、文本、类型等 8 个维度进行规则化评分；(b) 代码级 LLM 评分——使用 GPT-5-mini 从代码层面评估视觉忠实度；(c) 图表级 LMM 评分——使用 GPT-5-mini 对比 GT 图表和生成图表的多维度相似性。同时进行人类评估验证 LMM 评分的一致性
    - 设计动机：代码正确不等于图表正确——相同的数据可以通过不同的代码路径渲染出视觉差异很大的图表。8 维度规则化评分比 ChartMimic 的 4 维度更全面、更快、更准确

3. **丰富的数据多样性**:

    - 功能：确保基准的广泛覆盖和区分度
    - 核心思路：涵盖 22 种图表类型（雷达图、热力图、散点图、箱线图、树图、误差条、饼图、小提琴图等），Level 1 强调图表唯一性（719 个独特图表），Level 2 每个图表至少一个编辑指令（1,010 个独特编辑），Level 3 包含 71 个 Excel 文件（最长 30,427 行）。指令长度 Level 2 平均 267 词最长，反映编辑任务的复杂性
    - 设计动机：多样的图表类型和任务覆盖确保基准不会被单一能力的模型"刷分"

### 损失函数 / 训练策略

本文为基准测试工作，不涉及模型训练。所有模型使用其公开权重或 API 进行推理。开源模型在 NVIDIA V100 GPU 上运行，非 thinking 模型推理长度 4,096 tokens。图像以原始分辨率输入。

## 实验关键数据

### 主实验

**Level 1 图表复现（部分模型）**

| 模型 | DR Exec% | DR Base | DR LMM | CRD Base | CFD Base |
|------|----------|---------|--------|----------|----------|
| GPT-5.2 | 97.08 | 79.91 | 43.73 | 66.31 | 73.02 |
| Gemini-3-Pro | 97.50 | 78.65 | 45.42 | 69.23 | 70.78 |
| Claude-Sonnet-4 | 96.52 | 65.60 | 32.36 | 61.46 | 65.27 |
| Qwen3-VL-32B | - | - | - | - | - |
| InternVL-3-38B | 85.26 | 53.57 | 16.68 | 58.17 | 60.17 |

**Level 2 图表编辑（8 维度代码级评分）**

| 模型 | Exec% | Color | Data | Text | Type | Base | LMM |
|------|-------|-------|------|------|------|------|-----|
| GPT-5.2 | 96.04 | 58.44 | 64.66 | 83.77 | 94.52 | 70.93 | 33.03 |
| Gemini-3-Pro | 97.23 | 52.32 | 62.75 | 77.16 | 93.86 | 70.78 | 33.41 |
| Claude-Sonnet-4 | 90.20 | 47.17 | 54.88 | 80.52 | 93.29 | 63.65 | 25.40 |

### 消融实验

**代码级 vs 图表级评分差距**

| 模型 | Level 2 代码级 Base | Level 2 图表级 LMM | 差距 |
|------|-------------------|-------------------|------|
| GPT-5.2 | 70.93 | 33.03 | -37.90 |
| Gemini-3-Pro | 70.78 | 33.41 | -37.37 |
| Claude-Sonnet-4 | 63.65 | 25.40 | -38.25 |

**开源模型 thinking vs non-thinking（Level 2，MiMo-VL-7B）**

| 配置 | Base | LMM |
|------|------|-----|
| MiMo-VL-7B-SFT (non-thinking) | 63.99 | 22.61 |
| MiMo-VL-7B-RL (non-thinking) | 64.41 | 21.43 |
| MiMo-VL-7B-SFT (thinking) | 65.49 | 16.95 |
| MiMo-VL-7B-RL (thinking) | 70.45 | 30.04 |

### 关键发现

- 从 Level 1 到 Level 2/3，所有模型性能显著下降——即使 GPT-5.2 在编辑任务上的图表质量评分（LMM）仅 33.03，说明编辑和数据转图表远未解决
- 代码级评分与图表级评分存在巨大差距（约 37-38 分），说明代码层面的正确性不能代表视觉层面的忠实度
- 闭源模型（GPT-5.2、Gemini-3-Pro）在所有级别上大幅领先开源模型，开源模型中 Qwen3-VL-32B 和 InternVL-3.5-38B 表现最佳
- Thinking 模式对部分模型有帮助（MiMo-VL-7B-RL thinking LMM 30.04 vs non-thinking 21.43），但不是所有模型都受益
- Level 3 长表格任务极具挑战性，需要长上下文理解、信息检索、数学计算和代码生成的综合能力

## 亮点与洞察

- 层次化设计精准对应了用户工作流的三个阶段——从"看图画图"到"改图"到"看数据画图"，难度递增揭示了模型真正的能力边界
- 代码级和图表级评估的巨大差距是重要发现——提示社区不能仅靠代码相似度来评估图表生成质量
- 8 维度规则化评分可迁移到其他图表/代码生成任务中使用

## 局限与展望

- 评估依赖 GPT-5-mini 作为 judge，存在 LMM 评估偏差
- Level 3 的标注和 GT 代码构建极为耗时，限制了数据规模（仅 313 个任务）
- 仅评估了 Python/Matplotlib 生态的代码生成，未覆盖 R、D3.js 等其他绑图工具
- 未测试模型的迭代修正能力——实际工作中用户会多轮交互修正图表

## 相关工作与启发

- **vs ChartMimic**: ChartMimic 仅测试图表复现（Level 1），且已被 GPT-4o 达到 82.2% 基本饱和；Chart2Code 扩展到编辑和长表格场景，且当前 SOTA 仅 33 分
- **vs ChartEdit**: ChartEdit 仅覆盖简单局部编辑（233 个样本），Chart2Code 的 Level 2 包含 1,010 个复杂编辑任务（加趋势线、算相关系数等）
- **vs Plot2Code**: Plot2Code 仅 132 个样本且无规则化评估，Chart2Code 规模和评估体系都显著更全面

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个层次化图表代码生成基准，任务设计贴合真实用户需求
- 实验充分度: ⭐⭐⭐⭐⭐ 29 个模型的全面评测，多层次评估协议，人类评估验证
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，数据统计丰富，但表格较多需反复查阅
- 价值: ⭐⭐⭐⭐⭐ 揭示了图表代码生成的真正难点，为社区指明了改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](../../NeurIPS2025/code_intelligence/table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

</div>

<!-- RELATED:END -->
