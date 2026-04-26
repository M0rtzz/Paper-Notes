---
title: >-
  [论文解读] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays
description: >-
  [NeurIPS 2025][医学图像][胸部X光] 提出 CheXStruct + CXReasonBench，一个基于胸部X光的结构化诊断推理评估框架，通过多路径、多阶段评估揭示现有 LVLM 在中间推理步骤上的严重不足。
tags:
  - NeurIPS 2025
  - 医学图像
  - 胸部X光
  - 诊断推理
  - 视觉语言模型
  - benchmark
  - 结构化评估
---

# CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays

**会议**: NeurIPS 2025  
**arXiv**: [2505.18087](https://arxiv.org/abs/2505.18087)  
**代码**: [GitHub](https://github.com/ttumyche/CXReasonBench)  
**领域**: 医学图像  
**关键词**: 胸部X光, 诊断推理, 视觉语言模型, benchmark, 结构化评估

## 一句话总结

提出 CheXStruct + CXReasonBench，一个基于胸部X光的结构化诊断推理评估框架，通过多路径、多阶段评估揭示现有 LVLM 在中间推理步骤上的严重不足。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLM）在医学影像中的应用日益广泛，包括报告生成和视觉问答（VQA）。胸部X光因其临床相关性和可获取性成为标准的评估基准。

**现有痛点**：现有基准（VQA-RAD、PathVQA、PMC-VQA 等）主要评估最终诊断答案的正确性，对模型是否进行了临床有意义的推理过程缺乏洞察。一些近期工作引入解释或视觉基础（visual grounding），但仍侧重输出而非中间推理步骤。

**核心矛盾**：模型可能给出看似合理的答案（如"心脏区域"显示异常），但无法判断它是否真正识别了相关解剖结构、执行了适当测量并应用了临床规则（如心胸比）。没有对中间步骤的评估，就无法区分模型是真正理解图像还是依赖浅层模式匹配。

**本文目标**：构建一个能够评估模型诊断推理中间步骤的基准——不仅评估"答案对不对"，更评估"推理过程对不对"。

**切入角度**：从解剖分割出发，自动提取诊断测量、计算诊断指标、应用临床阈值，构建完整的结构化推理流程作为参考答案。

**核心 idea**：通过自动化管道从X光中提取结构化推理步骤，并设计多路径多阶段评估框架，系统性地评估LVLM的诊断推理能力。

## 方法详解

### 整体框架

系统由两个互补部分组成：
- **CheXStruct**：全自动管道，从胸部X光中提取结构化临床信息（解剖分割→解剖标志→诊断测量→诊断指标→临床阈值判定）
- **CXReasonBench**：多路径、多阶段评估框架，基于 CheXStruct 的参考答案评估模型在各中间阶段的表现

### 关键设计

#### CheXStruct 管道

**任务定义**：与临床专家合作定义了12个放射学发现和质量评估任务，分为两类标准：
- **标准化可量化标准**：如心脏肥大使用心胸比（CTR），定义为心脏最大水平宽度与胸廓宽度之比
- **专家定义标准**：对于缺乏标准化标准的任务（如纵隔增宽），设计基于比例的替代指标

**解剖分割**：使用 CXAS 分割模型获取必要的解剖掩模（如心脏和肺部掩模）

**质量控制（QC）**：
- 每个任务都有专门的 QC 规则
- 自动过滤低质量样本
- 仅通过 QC 的样本用于基准构建

#### CXReasonBench 评估管道

**初始诊断决策**：对每个案例提出二元诊断问题（如"该患者是否有心脏肥大？"），模型可选择 Yes/No 或 "I don't know"

**Path 1：直接推理过程评估**（模型给出明确答案时）
- Stage 1：诊断标准选择——模型识别使用的诊断标准
- Stage 1.5：精细标准采纳——对需要专家定义标准的任务提供额外标准
- Stage 2：解剖结构识别——从标注的X光中选择相关解剖区域
- Stage 3：测量/识别——执行诊断标准相关的计算或解读
- Stage 4：最终决策——基于 Stage 3 的结果做出判断

**Path 2：引导推理与再评估**（模型回答"I don't know"或拒绝专家标准时）
- Stage 1：解剖结构识别（提示辅助）
- Stage 2：引导式测量/识别（提供详细视觉标注和计算说明）
- Stage 3：最终决策
- **再评估 Path 1**：完成引导后，测试模型能否在新案例上独立应用学到的推理

### 评估指标

- **Final Stage Completion**：成功完成全部推理阶段的比例
- **Average Reasoning Depth**：平均达到的推理阶段数
- **Decision Alignment**：初始决策与最终决策的一致性
- **Measurement Consistency**：Stage 3 和 Stage 4 之间数值的一致性

## 实验关键数据

### 基准规模

| 指标 | 数量 |
|------|------|
| 诊断任务 | 12 |
| 评估案例 | 1,200 |
| QA 对总数 | 18,988 |
| Path 1 QA | 8,044 |
| Path 2 QA | 3,600 |
| 再评估 Path 1 QA | 7,344 |

### 主实验：Path 1 结果（Greedy Decoding）

| 模型 | Completion(↑) | Depth(↑) | Consistency(↑) | Alignment(↑) |
|------|--------------|----------|----------------|-------------|
| Gemini-2.5-Pro | 17.03 (16.24) | 1.96 | 68.4 | 60.88 |
| Gemini-2.5-Flash | 12.83 (8.56) | 1.40 | 43.76 | 50.29 |
| GPT-4.1 | 8.32 | 1.15 | 61.22 | 39.80 |
| Pixtral-Large | 3.73 (2.31) | 1.00 | 28.50 | 36.74 |
| Llama-3.2-90B | 0.38 | 0.53 | 61.27 | 23.32 |
| Qwen2.5-VL-72B | 2.34 (2.12) | 0.67 | 34.67 | 38.45 |
| MedGemma 27B | 3.31 (2.34) | — | — | — |
| HealthGPT-L14 | — | — | — | — |
| RadVLM | — | — | — | — |

### 关键发现

1. **即使最强的 Gemini-2.5-Pro 在 Path 1 中仅 17% 完成全部推理阶段**，平均仅达到 Stage 2
2. **视觉基础是最大瓶颈**：模型在 Stage 2（解剖结构识别）的表现高度依赖任务——单一显著结构（如肺）可达 89%，但抽象参考线任务（如气管偏移）仅 48%
3. **闭源模型显著优于开源模型**，但差距主要在 Stage 2+ 的视觉理解上
4. **医学专用模型（HealthGPT、RadVLM）在识别型任务上相对较好**，但在需要算术计算的测量型任务上表现薄弱
5. **结构化引导（Path 2）能帮助诊断推理**，但多数模型无法将学到的推理泛化到新案例

## 亮点与洞察

- **首个评估诊断推理中间步骤的胸部X光基准**，填补了从"答案正确性"到"推理正确性"的评估空白
- **全自动管道 CheXStruct** 可扩展到大规模数据集，无需人工干预
- **多路径设计**（Path 1 + Path 2 + Re-eval）提供了全面的推理能力画像
- **揭示了 LVLM 的"知识-视觉"断裂**：模型可能知道正确的诊断标准，却无法在图像中定位相关解剖结构
- **Two-round 格式设计巧妙**：先隐藏正确答案测试模型能否识别不足，再给出正确选项

## 局限与展望

1. **仅覆盖结构可推导的发现**：依赖分割模型，无法处理病理特异性模式（如不透明度、气液界面）
2. **12个任务仍有限**：未涵盖全部临床相关的胸部X光诊断
3. **评估格式为多选题**：可能低估模型的开放式推理能力
4. **分割模型本身的误差**可能影响参考答案的质量
5. **缺乏对纵向推理的评估**（如随访对比）

## 相关工作与启发

- 与 Chest ImaGenome、PadChest-GR、GR-Bench 等结构化框架相比，CheXStruct 直接从图像提取（而非报告），粒度更细
- 与 GEMeX 的视觉基础评估互补，CXReasonBench 更强调显式推理过程
- **核心启发**：评估 AI 辅助诊断系统不能仅看"答案对不对"，必须审核推理过程的临床合理性

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：问题定义清晰、管道设计系统全面、实验覆盖12个模型和12个任务；但任务范围局限于结构可推导的发现，且多选评估格式可能无法完全反映实际临床推理能力。作为基准贡献非常有价值。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](../../ACL2026/medical_imaging/dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ECCV 2024\] CheX: Interactive Localization and Region Description in Chest X-rays](../../ECCV2024/medical_imaging/chex_interactive_localization_and_region_description_in_chest_x-rays.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)

<!-- RELATED:END -->
