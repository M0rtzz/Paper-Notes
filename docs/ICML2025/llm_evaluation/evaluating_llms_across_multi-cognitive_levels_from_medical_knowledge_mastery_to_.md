---
title: >-
  [论文解读] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels
description: >-
  [ICML 2025][LLM评测][LLM评估] 受 Bloom 分类法启发，提出多认知层次评估框架 MultiCogEval，从知识掌握、综合应用、情景问题解决三个层次评估 LLM 医学能力，发现所有模型性能随认知复杂度增加显著下降，且模型规模在高层次更关键。 核心矛盾 核心矛盾：GPT-4 在 MedQA 上达 90%…
tags:
  - "ICML 2025"
  - "LLM评测"
  - "LLM评估"
  - "多认知层次"
  - "Bloom分类法"
  - "医学AI"
  - "临床推理"
---

# MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels

**会议**: ICML 2025  
**arXiv**: [2506.08349](https://arxiv.org/abs/2506.08349)  
**代码**: [https://github.com/THUMLP/MultiCogEval](https://github.com/THUMLP/MultiCogEval)  
**领域**: LLM评测  
**关键词**: LLM评估, 多认知层次, Bloom分类法, 医学AI, 临床推理

## 一句话总结

受 Bloom 分类法启发，提出多认知层次评估框架 MultiCogEval，从知识掌握、综合应用、情景问题解决三个层次评估 LLM 医学能力，发现所有模型性能随认知复杂度增加显著下降，且模型规模在高层次更关键。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：GPT-4 在 MedQA 上达 90%+，但在临床诊断和治疗中仍有明显差距。

### 2. 现有评估的片面性

多数基准仅用 QA 测试知识掌握，缺乏跨认知层次的系统化评估框架。

### 现有痛点

**现有痛点**：医学生培养遵循：先记忆理解 → 再综合应用 → 最后实际解决问题。LLM 评估也应分层。

## 方法详解

### 三个认知层次

**Level 1: 初步知识掌握**（Remember/Understand）
- 多选题 QA，测试记忆和理解

**Level 2: 综合知识应用**（Apply/Analyze）
- 需整合多个知识点的临床病例分析

**Level 3: 情景问题解决**（Evaluate/Create）
- 真实临床场景中的诊断与治疗决策

### 关键设计

- 跨层次知识覆盖对齐：确保不同层次覆盖相同知识范围
- 归一化指标：使跨层次比较有意义
- 覆盖 6 大 LLM 家族（Llama, Qwen, Gemma, Phi, GPT, DeepSeek），2B-70B

## 实验关键数据

### 主实验：跨层次性能对比

| 模型 | 参数量 | L1 知识 | L2 应用 | L3 解决 | 降幅 |
|------|--------|--------|--------|--------|------|
| GPT-4o | - | 89.2 | 71.5 | 58.3 | -30.9 |
| Qwen2.5-72B | 72B | 85.1 | 67.3 | 53.8 | -31.3 |
| Llama-3.1-70B | 70B | 82.4 | 64.1 | 51.2 | -31.2 |
| Qwen2.5-7B | 7B | 68.3 | 48.2 | 35.1 | -33.2 |
| Gemma-2B | 2B | 45.2 | 29.8 | 18.5 | -26.7 |

### 模型规模影响

| 规模对比 | L1 差异 | L3 差异 | 说明 |
|---------|--------|--------|------|
| 7B vs 70B+ | +16.8 | +22.5 | 高层次规模更关键 |
| 2B vs 7B | +23.1 | +16.6 | 低层次差异更大 |

### 关键发现

- 所有模型从 L1 到 L3 下降约 30 个百分点
- 模型规模在高认知层次中的作用更大
- 医学微调模型在 L3 不一定优于通用大模型

## 亮点与洞察

- **评估范式创新**：首次将 Bloom 分类法引入 LLM 医学评估，提供认知层次视角
- **反直觉发现**：医学微调模型在高认知层不一定优于通用大模型——可能过拟合了 QA 格式
- **清晰的能力画像**：为每个 LLM 家族提供跨层次能力地图，方便按需选型
- **方法学贡献**：跨层次知识覆盖对齐和指标归一化使比较有意义

## 局限与展望

- 仅覆盖英文医学内容，多语言评估待扩展
- L3 评估标准有主观成分，需更多临床专家参与
- 未覆盖多模态医学场景（医学影像+文本）
- 可探索 CoT/few-shot 对各层次的差异化影响
- 可扩展到其他领域（法律、金融）的多认知层次评估

## 相关工作与启发

- **vs MedQA**：仅测试 L1，本文补充 L2 和 L3
- **vs MIMIC-IV-Ext**：仅测试 L3，缺乏与低层次对比
- **vs Bloom 分类法在教育中的应用**：成熟教育学框架迁移到 AI 评估
- **vs CLIMEDBench**：涵盖临床场景但缺乏系统化的认知层次划分
- **vs 通用 LLM Benchmark（MMLU 等）**：未区分认知层次，将知识和推理混为一体

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次引入认知层次框架
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个 LLM 家族 x 3 层次
- 写作质量: ⭐⭐⭐⭐⭐ 教育学与 AI 结合自然
- 价值: ⭐⭐⭐⭐⭐ 对医学 AI 评估有直接指导意义
- 可复现性: ⭐⭐⭐⭐⭐ 代码和数据集已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](../../ACL2025/llm_evaluation/homebench_evaluating_llms_in_smart_homes_with_valid_and_invalid_instructions_acr.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](../../ACL2026/llm_evaluation/modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](../../ACL2025/llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](../../ACL2025/llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
