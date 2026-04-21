---
title: >-
  [论文解读] DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation
description: >-
  [ECCV 2024][合成数据生成] 提出利用代码生成合成结构化视觉数据（幻灯片和UI），用于训练理解模型，减少人工标注需求。
tags:
  - ECCV 2024
  - 合成数据生成
  - 幻灯片理解
  - 用户界面理解
  - 代码生成
  - 结构化视觉
---

# DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation

**会议**: ECCV 2024  
**arXiv**: [2410.00201](https://arxiv.org/abs/2410.00201)  
**代码**: 无  
**领域**: NLP Generation  
**关键词**: 合成数据生成, 幻灯片理解, 用户界面理解, 代码生成, 结构化视觉

## 一句话总结

提出利用代码生成合成结构化视觉数据（幻灯片和UI），用于训练理解模型，减少人工标注需求。

## 研究背景与动机

幻灯片（Slides）和用户界面（UI）是日常数字交互中极为常见的结构化视觉内容，对这类内容的机器理解对于辅助残障人士使用数字工具至关重要。然而，现有的结构化视觉理解方法通常依赖大量手工收集和标注的数据，这一过程既耗时又费力。

核心痛点在于：(1) 真实幻灯片和UI数据涉及隐私和版权问题，难以大规模获取；(2) 标注结构化视觉内容需要详细的元素级别标注（如元素类型、位置、层级关系等），标注成本极高；(3) 现有视觉语言模型在理解结构化布局方面能力有限。

本文的切入角度非常巧妙——通过代码生成来创建合成的结构化视觉数据。由于幻灯片和UI本质上是由代码/标记语言定义的，因此可以通过程序化方式生成大量带有精确标签的合成数据。这种方法从根本上解决了数据标注的瓶颈问题，因为合成数据天然携带完整的结构化标签。

## 方法详解

### 整体框架

DreamStruct的核心pipeline分为三个阶段：(1) 利用大语言模型（LLM）生成描述幻灯片或UI布局的代码；(2) 执行生成的代码渲染出合成的结构化视觉图像，同时自动获取所有元素的标签；(3) 使用合成数据与少量真实标注数据混合训练下游理解模型。

### 关键设计

1. **代码驱动的合成数据生成**:
    - 功能：自动生成带有完整标签的结构化视觉数据
    - 核心思路：利用LLM生成HTML/CSS或PPT脚本代码，代码执行后产生视觉图像，代码中的结构信息直接作为标签。通过控制代码模板和参数，可以生成多样化的布局和样式变体
    - 设计动机：结构化视觉内容本质上由代码定义，因此代码即是标注，无需额外人工标注

2. **少样本引导策略**:
    - 功能：确保合成数据的分布与真实数据对齐
    - 核心思路：使用少量真实标注样本作为参考，引导LLM生成风格和结构上与真实数据相似的代码。通过few-shot prompting，让生成的合成数据在视觉外观和结构复杂度上接近真实场景
    - 设计动机：纯随机生成的合成数据可能与真实分布差距过大，少量真实样本的引导能显著提升合成数据的质量和实用性

3. **多任务评估框架**:
    - 功能：在三个核心任务上验证方法有效性
    - 核心思路：覆盖视觉元素识别（element recognition）、视觉内容描述（content description）和内容类型分类（type classification）三个任务，全面评估合成数据对下游任务的增益
    - 设计动机：单一任务可能无法全面反映合成数据的价值，多任务评估能更好地验证方法的通用性

### 损失函数 / 训练策略

训练策略上采用混合训练范式：将大量合成数据与少量真实标注数据按一定比例混合进行训练。合成数据提供大量多样性样本用于特征学习，真实数据帮助模型校准到真实分布。具体比例通过验证集性能进行调节。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 幻灯片元素识别 | mAP | 显著提升 | 仅真实数据训练 | +8-15% |
| UI元素识别 | mAP | 显著提升 | 仅真实数据训练 | +5-12% |
| 内容描述 | CIDEr | 提升明显 | 仅真实数据训练 | +10-20% |
| 内容分类 | Accuracy | 显著提升 | 仅真实数据训练 | +3-8% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅真实数据 | 基线 | 数据量受限 |
| 仅合成数据 | 低于混合 | 存在域差距 |
| 合成+真实混合 | 最优 | 两者互补 |
| 不同合成数据量 | 随量提升 | 合成数据量越大越好，但有饱和点 |

### 关键发现

- 代码生成的合成数据能有效替代大量人工标注，在少标注场景下提升尤为显著
- 合成数据与真实数据的混合训练始终优于单独使用任一数据源
- 方法在幻灯片和UI两个不同领域均有效，展示了良好的通用性
- 合成数据的多样性对模型泛化能力至关重要

## 亮点与洞察

- 核心洞察非常优雅：结构化视觉内容天然与代码对应，代码即是标注
- 利用LLM的代码生成能力来解决计算机视觉的数据标注问题，是一种巧妙的跨模态方法迁移
- 方法简单实用，不需要复杂的模型架构创新，而是从数据角度解决问题
- 对于辅助技术领域有直接的应用价值

## 局限与展望

- 合成数据的视觉真实感仍有提升空间，生成的幻灯片和UI在视觉复杂度上可能不如真实场景
- 代码生成的多样性受限于LLM的能力和prompt设计
- 未探讨方法在更复杂的结构化文档（如学术论文、表格密集型报告）上的适用性
- 可以考虑引入对抗训练或域自适应技术进一步缩小合成-真实域差距

## 相关工作与启发

- **文档理解**: LayoutLM系列通过预训练学习文档布局理解，但依赖大量标注数据
- **合成数据**: SynthText等工作在OCR领域验证了合成数据的有效性，DreamStruct将这一思路扩展到结构化视觉理解
- **LLM辅助数据生成**: 越来越多的工作利用LLM生成训练数据，DreamStruct通过代码生成提供了一种结构化的数据生成范式
- 启发：代码生成范式可以推广到任何可由代码定义的视觉内容（如图表、流程图、网页等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 代码即标注的想法简洁优雅，切入角度新颖
- 实验充分度: ⭐⭐⭐ 三个任务两个领域的评估较为充分，但缺少定量消融细节
- 写作质量: ⭐⭐⭐⭐ 论文逻辑清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 对辅助技术领域有直接的实际价值，方法具有良好的可扩展性

<!-- RELATED:START -->

## 相关论文

- [Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](../../NeurIPS2025/code_intelligence/preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](../../AAAI2026/code_intelligence/towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)
- [DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](../../ICML2025/code_intelligence/dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)
- [LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](../../ACL2025/code_intelligence/benchmarking_long-context_language_models_on_long_code_understanding.md)

<!-- RELATED:END -->
