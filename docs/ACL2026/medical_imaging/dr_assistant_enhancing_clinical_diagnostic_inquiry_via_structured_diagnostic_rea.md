---
title: >-
  [论文解读] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning
description: >-
  [ACL 2026][医学图像][临床诊断推理] 本文提出临床诊断推理数据（CDRD）结构来捕获从症状到鉴别诊断的抽象临床推理逻辑，并基于 CDRD 通过 SFT+RL 两阶段训练构建 Dr. Assistant 模型（14B），在临床问诊基准上 ICD-Recall 超过 HuatuoGPT-o1-72B 13.59%，达到与 GPT-5 竞争的水平。
tags:
  - ACL 2026
  - 医学图像
  - 临床诊断推理
  - 强化学习
  - 结构化数据
  - 问诊引导
  - CDSS
---

# Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2601.13690](https://arxiv.org/abs/2601.13690)  
**代码**: [GitHub](https://github.com/YGswu/Dr.-Assistant)  
**领域**: 医疗NLP / 临床决策支持  
**关键词**: 临床诊断推理, 强化学习, 结构化数据, 问诊引导, CDSS

## 一句话总结

本文提出临床诊断推理数据（CDRD）结构来捕获从症状到鉴别诊断的抽象临床推理逻辑，并基于 CDRD 通过 SFT+RL 两阶段训练构建 Dr. Assistant 模型（14B），在临床问诊基准上 ICD-Recall 超过 HuatuoGPT-o1-72B 13.59%，达到与 GPT-5 竞争的水平。

## 研究背景与动机

**领域现状**：临床决策支持系统（CDSS）为医生提供推理和问诊指导。LLM 因其广泛的医学知识已被广泛应用于医疗咨询，在医学基准上表现出色。

**现有痛点**：(1) 传统 CDSS 依赖结构化知识库和规则算法，开发维护成本高且适应性差；(2) 现有医疗 LLM（如 Baichuan-M2、HuatuoGPT-o1）主要优化患者咨询体验，缺乏专业的临床诊断推理和问诊技能；(3) 临床指南中的诊断推理逻辑分散在不同章节中，难以直接用于训练；(4) 即使有高质量数据，训练模型掌握临床问诊技能仍然是一个显著挑战。

**核心矛盾**：LLM 拥有广泛的医学知识，但缺乏系统性的临床诊断推理逻辑——在零样本提示下无法像经验丰富的医生那样进行结构化的症状分析和鉴别诊断。

**本文目标**：(1) 设计 CDRD 数据结构来捕获抽象诊断推理逻辑；(2) 构建具备诊断推理和问诊技能的 Dr. Assistant 模型；(3) 构建临床诊断推理与问诊评估基准。

**切入角度**：从临床指南中提取结构化的诊断推理逻辑（CDRD），然后用 CDRD 作为种子合成 SFT 和 RL 训练数据，通过两阶段训练使模型内化临床推理能力。

**核心 idea**：临床诊断推理可以被抽象为（核心症状, 诊断证据, 鉴别诊断）的结构化三元组——以此为种子生成训练数据，再通过包含"逻辑偏差惩罚"的 RL 奖励函数约束模型的推理行为。

## 方法详解

### 整体框架

CDRD 构建管道（LLM+医生协作三阶段：症状提取→疾病匹配→逻辑补全）→ 数据合成（CDRD→QA 对用于 SFT + CDRD→多轮问诊对话用于 RL）→ Dr. Assistant 两阶段训练（SFT 记忆推理逻辑 + RL 强化问诊技能）。

### 关键设计

1. **CDRD 数据结构与构建管道**:

    - 功能：从临床指南中提取抽象诊断推理逻辑
    - 核心思路：定义 CDRD 为三元组 $\mathcal{C} = (\mathcal{S}, \mathcal{E}, \mathcal{D})$——核心症状 $\mathcal{S}$（如头痛）、诊断证据 $\mathcal{E}$（相关症状/检查/化验结果）、鉴别诊断 $\mathcal{D}$（可能疾病及其临床表现和所需检查）。三阶段构建：LLM 提取候选症状→医生精炼标准化→LLM 匹配疾病→医生验证→LLM 补全逻辑→医生审核
    - 设计动机：临床指南的推理逻辑分散在不同章节，CDRD 将其重组为从症状出发的鉴别诊断路径，每阶段都有医生审核确保可靠性

2. **两阶段训练策略（SFT + RL）**:

    - 功能：使模型先记忆推理逻辑，再通过实践强化问诊技能
    - 核心思路：Stage 1 用 CDRD 生成的 QA 对做 SFT，让模型记住初步的诊断推理逻辑。Stage 2 用 CDRD 生成的多轮问诊对话做 RL（双智能体模拟：医生智能体+患者智能体），设计包含两个维度的奖励函数：临床推理与问诊技能评分 + CDRD 逻辑保真度（与 CDRD 的逻辑偏差惩罚）
    - 设计动机：仅 SFT 无法让模型灵活运用推理逻辑进行动态多轮问诊，RL 的逻辑偏差惩罚约束模型在自由探索时不偏离正确的诊断推理路径

3. **结构化推理-问诊模板**:

    - 功能：将每轮问诊的推理过程结构化为六步
    - 核心思路：已知信息→用户意图→已提供信息→诊断假设→待收集信息→回应策略→问诊/诊断输出。这个模板确保模型的每轮推理都是有据可循的
    - 设计动机：非结构化的问诊容易遗漏关键信息或做出无依据的跳跃推理

### 损失函数 / 训练策略

SFT 阶段：标准交叉熵损失。RL 阶段：奖励函数 = 临床推理与问诊技能评分（由 LLM 评估覆盖率、准确性、问诊逻辑性）+ CDRD 逻辑保真度（惩罚与 CDRD 标准逻辑的偏差）。基座模型为 14B 参数。

## 实验关键数据

### 主实验

**诊断推理评估（242 个真实临床案例，8 个二级科室）**

| 模型 | 参数 | ICD-Recall ↑ | 综合评分 |
|------|------|-------------|---------|
| HuatuoGPT-o1 | 72B | 基线 | - |
| GPT-5 | - | 高 | 竞争水平 |
| **Dr. Assistant** | **14B** | **+13.59%** | **与 GPT-5 竞争** |

### 消融实验

| 配置 | ICD-Recall | 问诊质量 |
|------|-----------|---------|
| 仅 SFT | 基础水平 | 中等 |
| SFT + RL（无逻辑惩罚） | 提升 | 提升但有逻辑偏差 |
| SFT + RL（完整奖励） | **最高** | **最高** |

### 关键发现

- Dr. Assistant（14B）以小模型超越 HuatuoGPT-o1（72B），ICD-Recall 提升 13.59%——证明专业化的诊断推理训练比模型规模更重要
- RL 中的 CDRD 逻辑保真度惩罚是关键——没有它模型容易产生看似合理但逻辑不严谨的推理
- 结构化推理模板使模型的每轮问诊都有据可循，提升了问诊的系统性和完整性
- Dr. Assistant 达到与 GPT-5 竞争的水平，为 CDSS 的实际部署提供了可行方案

## 亮点与洞察

- CDRD 数据结构是一个通用的临床知识表示方案，可扩展到更多临床指南
- LLM+医生协作的数据构建管道平衡了效率和可靠性
- RL 奖励函数中"逻辑偏差惩罚"的设计确保模型的自由探索不偏离临床推理正轨

## 局限与展望

- 目前仅基于内科相关临床指南构建 CDRD，覆盖科室有限
- 评估基准规模较小（242 个案例，147 轮问诊），统计效力有限
- 未在真实临床环境中进行前瞻性评估
- RL 奖励函数的权重调节可能需要领域专家参与

## 相关工作与启发

- **vs Baichuan-M2/HuatuoGPT-o1**: 这些模型优化通用医疗咨询体验，本文专注于临床诊断推理和问诊技能的专业化
- **vs 传统 CDSS**: 传统系统依赖规则难以扩展，Dr. Assistant 通过 LLM+结构化推理数据实现灵活适应
- **vs Doctor-R1**: Doctor-R1 侧重推理过程，本文更注重诊断推理逻辑的结构化和问诊技能

## 评分

- 新颖性: ⭐⭐⭐⭐ CDRD 数据结构和逻辑偏差惩罚 RL 的设计新颖
- 实验充分度: ⭐⭐⭐⭐ 对比全面但评估规模有限
- 写作质量: ⭐⭐⭐⭐ 方法清晰系统，临床问题定义准确
- 价值: ⭐⭐⭐⭐⭐ 为 CDSS 实际部署提供了有效的 LLM 解决方案

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)

<!-- RELATED:END -->
