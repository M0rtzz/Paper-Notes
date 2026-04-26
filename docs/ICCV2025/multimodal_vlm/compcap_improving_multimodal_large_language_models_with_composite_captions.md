---
title: >-
  [论文解读] CompCap: Improving Multimodal Large Language Models with Composite Captions
description: >-
  [ICCV 2025][多模态][复合图像] 提出 CompCap 框架，自动合成6类复合图像（拼贴、图文混合、图表、表格、代码、流程图）及其高质量描述文本，构建 CompCap-118K 数据集，通过在 SFT 阶段引入该数据集显著提升 MLLM 对复合图像的理解能力。
tags:
  - ICCV 2025
  - 多模态
  - 复合图像
  - 图像描述生成
  - 多模态大语言模型
  - 数据合成
  - 视觉-语言对齐
---

# CompCap: Improving Multimodal Large Language Models with Composite Captions

**会议**: ICCV 2025  
**arXiv**: [2412.05243](https://arxiv.org/abs/2412.05243)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 复合图像, 图像描述生成, 多模态大语言模型, 数据合成, 视觉-语言对齐

## 一句话总结

提出 CompCap 框架，自动合成6类复合图像（拼贴、图文混合、图表、表格、代码、流程图）及其高质量描述文本，构建 CompCap-118K 数据集，通过在 SFT 阶段引入该数据集显著提升 MLLM 对复合图像的理解能力。

## 研究背景与动机

当前多模态大语言模型（MLLM）在自然图像理解方面表现出色，但在**复合图像（Composite Images, CIs）**理解上存在明显短板。复合图像是指由多种视觉元素（照片、图表、文字、代码等）组合而成的合成视觉内容，在实际应用中非常普遍（如海报、信息图、网页截图等）。

作者通过实验发现：

**MLLM 对 CI 的理解远差于自然图像**：在 captioning 和 VQA 任务上，LLaVA-1.5、InstructBLIP 等模型在 CI 上的准确率显著低于自然图像。

**captioning 错误与 VQA 错误高度一致**：模型在描述 CI 时产生的错误信息与直接 VQA 时的错误高度重合，说明问题根源在于视觉-语言对齐不充分。

**训练数据中缺乏高质量 CI 描述**：现有训练数据主要包含自然图像的高质量 caption 和 CI 的指令问答数据（如 ChartQA），但缺少 CI 的详细描述文本。

这些发现表明，仅靠指令数据（instruction data）不足以让 MLLM 全面理解 CI——高质量的 CI-caption 数据对于建立更好的视觉-语言对齐至关重要。

## 方法详解

### 整体框架

CompCap 是一个通用的复合图像-描述文本合成框架。核心流程为：**元数据 → 图像生成 + 描述生成 → CI-Caption 对**。框架针对6类 CI 分别设计了专用的生成管线。

### 关键设计

1. **元数据驱动的图像合成**：元数据由原始数据（如图像-描述对、表格数据、代码片段）和配置信息（如布局、图表类型、颜色风格）组成。配置通过随机过程生成，确保多样性。图像使用 Plotly、Matplotlib、OpenCV、PIL、Mermaid、Carbon 等渲染工具生成。

2. **LLM 驱动的描述生成**：使用 LLM 为生成的 CI 编写准确且详细的描述。关键在于 prompt 设计——针对每类 CI 定制指令（如拼贴图需关注子图位置关系和关联，图表需关注数据分析）。采用主动式 in-context example 选择以提升描述质量和多样性。

3. **拼贴图（Collage）生成管线**（典型实例）：

    - **原始数据检索**：三种策略——随机检索（无关图像组合，用于反事实去偏）、相似性检索（基于 DINO-v2 和 CLIP 的视觉/文本相似度）、实体检索（同一实体的多张图片）
    - **布局生成**：Grid layout（先定义网格，再采样图片）和 Auto layout（先采样图片，再自动排列）
    - **描述设计**：prompt 中包含坐标系统、每张子图的位置和描述、in-context example；当图像相关时额外生成关联推断

4. **其他 CI 类型管线**：

    - 图文混合（Image-Text）：使用 OpenCV/PIL/Augraphy 渲染
    - 图表（Chart）：使用 Plotly 从表格数据生成，22K 样本，平均 1468 字符描述
    - 流程图（Diagram）：使用 Mermaid 代码通过 Selenium 渲染
    - 代码（Code）：使用 Carbon 渲染，2K 样本
    - 表格（Table）：使用 Matplotlib 渲染

### 损失函数 / 训练策略

将 CompCap-118K 数据集加入 MLLM 的 SFT 阶段。为保证公平比较，对原始 SFT 数据集进行均匀下采样，使总训练样本数保持一致。基于 LLaVA-NeXT 和 xGen-MM 两种架构训练了 4B/7B/13B 三种规模的模型。

## 实验关键数据

### 主实验

| 模型 | SEEDBench | TextVQA | MMBench | ChartQA | DocVQA | InfoVQA | 平均 | 提升 |
|------|-----------|---------|---------|---------|--------|---------|------|------|
| xGen-MM-4B (基线) | 71.3 | 67.7 | 75.5 | 54.8 | 55.2 | 27.6 | 57.2 | - |
| CompCap-4B | 71.6 | 67.9 | 76.2 | 57.4 | 58.1 | 27.9 | 58.9 | +1.7% |
| LLaVA-NeXT-7B (基线) | 71.2 | 65.2 | 67.6 | 63.5 | 76.5 | 39.2 | 62.5 | - |
| CompCap-7B | 70.5 | 65.6 | 68.9 | 68.9 | 77.6 | 40.8 | 64.5 | +2.0% |
| LLaVA-NeXT-13B (基线) | 71.9 | 67.6 | 68.9 | 68.5 | 79.9 | 43.8 | 65.6 | - |
| CompCap-13B | 72.2 | 67.8 | 70.8 | 73.9 | 81.1 | 47.0 | 68.5 | +2.9% |

### 消融实验

| 累积加入的 CI 类型 | NI 基准平均 | CI 基准平均 | 总平均 | 提升 |
|-------------------|------------|------------|--------|------|
| Baseline (无 CompCap) | 70.9 | 61.3 | 65.6 | - |
| + Collage | 71.5 | 62.4 | 66.4 | +0.8 |
| + Code | 71.3 | 62.8 | 66.6 | +1.0 |
| + Table | 71.7 | 63.0 | 67.0 | +1.4 |
| + Diagram | 71.5 | 63.1 | 67.4 | +1.8 |
| + Chart | 72.2 | 63.9 | 68.0 | +2.4 |
| + Image-Text (全部=CompCap-118K) | 73.1 | 64.6 | 68.5 | +2.9 |

### 关键发现

- 每种 CI 类型的加入都带来了一致的性能提升，证明框架各管线的有效性
- Caption 数据比 instruction 数据更有利于跨领域迁移（如 chart caption 有助于 DocVQA 和 InfoVQA）
- 在 ChartQA 上用 caption 替换 instruction 的实验表明：caption 更有效提升性能，但 100% 替换会损害指令遵循能力，最优比例约为 60-80%
- 即使不含数学专项数据，MathVista 也获得显著提升，说明 CI 理解对数学推理有间接促进

## 亮点与洞察

- **数据缺口的精准定位**：首次系统性地揭示了 MLLM 在 CI 理解上的不足源于训练数据中 CI-caption 的缺失
- **框架化的数据生成方法**：CompCap 是通用框架而非单一管线，可灵活扩展至新的 CI 类型
- **Caption vs. Instruction 的深入对比**：定量证明了 caption 在视觉-语言对齐上的独特价值

## 局限与展望

- 6类 CI 覆盖面有限，更复杂的混合 CI（如信息图、海报）未直接涵盖
- LLM 生成的描述可能存在幻觉，质量控制仅依赖后处理过滤
- 仅在 SFT 阶段引入 CI-caption，未探索在 PT 阶段的效果
- 数据集规模相对较小（118K），扩大规模可能带来进一步收益

## 相关工作与启发

- 与 ShareGPT4V 类似，都强调高质量 caption 对 MLLM 的重要性，但本文聚焦于 CI 领域
- 与 DVQA、PlotQA 等合成数据集不同，CompCap 覆盖更广的 CI 类型且专注 caption 而非 QA
- 对后续 MLLM 训练数据构建有指导意义：应平衡 NI 和 CI 的 caption 数据

## 评分

- **新颖性**: ⭐⭐⭐⭐ 问题定位准确，CI-caption 缺失的分析有价值，但数据合成方法本身不算特别新颖
- **实验充分度**: ⭐⭐⭐⭐ 三种模型规模、11个基准、详细消融，但缺少与更多SoTA的对比
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰、逻辑严密、图表丰富
- **价值**: ⭐⭐⭐⭐ 指出了MLLM训练数据蓝图中的重要缺口，具有实用参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ECCV 2024\] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](../../ECCV2024/multimodal_vlm/sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)

<!-- RELATED:END -->
