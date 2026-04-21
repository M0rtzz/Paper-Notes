---
title: >-
  [论文解读] Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation
description: >-
  [ACL2025][LLM推理][安全推理] 提出 AIDsafe 多智能体迭代审议框架，自动生成嵌入安全策略的高质量 CoT 数据，微调后的模型在安全泛化和越狱鲁棒性上显著优于传统安全训练，同时引入 ear-whisperer agent 解决 DPO 偏好数据中 selected/rejected 难以区分的问题。
tags:
  - ACL2025
  - LLM推理
  - 安全推理
  - Chain-of-Thought
  - 多智能体审议
  - 安全策略嵌入
  - DPO
  - 越狱防御
  - 偏好数据
---

# Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation

**会议**: ACL2025  
**arXiv**: [2505.21784](https://arxiv.org/abs/2505.21784)  
**代码**: 无（数据集已开源）  
**领域**: llm_reasoning  
**关键词**: 安全推理, Chain-of-Thought, 多智能体审议, 安全策略嵌入, DPO, 越狱防御, 偏好数据

## 一句话总结
提出 AIDsafe 多智能体迭代审议框架，自动生成嵌入安全策略的高质量 CoT 数据，微调后的模型在安全泛化和越狱鲁棒性上显著优于传统安全训练，同时引入 ear-whisperer agent 解决 DPO 偏好数据中 selected/rejected 难以区分的问题。

## 研究背景与动机

**领域现状**：LLM 安全训练正从传统的 SFT/RLHF 向"安全推理（safety reasoning）"范式转变——模型在生成回复前显式推理安全策略。OpenAI o1、Deliberative Alignment 等工作引领了这一方向。

**现有痛点**：(a) 高质量安全 CoT 数据极度稀缺，人工标注主观性强且成本高昂；(b) 直接让 LLM 生成安全推理链会出现幻觉、欺骗性推理、策略冲突等问题；(c) 安全策略本身存在模糊性和相互矛盾，单一模型难以全面覆盖。

**核心矛盾**：安全推理需要高质量 CoT 训练数据 → 生成高质量 CoT 又需要强推理能力的模型 → 训练这样的模型又需要数据——形成**鸡生蛋问题**。

**本文目标**：如何在不依赖昂贵推理模型的情况下，批量生成高质量、策略对齐的安全推理 CoT 数据？

**切入角度**：借鉴多 agent 辩论可以减少幻觉、提升推理可靠性的已有成果，用多 agent 协作 + 迭代审议 + 后处理精炼来替代单一强模型。

**核心 idea**：多 agent 迭代审议 + refiner 过滤 = 用普通模型也能生成高质量安全推理链数据。

## 方法详解

### 整体框架（AIDsafe）
四个阶段：初始化 → 审议 → 精炼 → （可选）偏好数据生成。

### 关键设计 1：初始化阶段

**功能**：解构用户意图 + 生成种子 CoT。
**为什么**：用户查询可能同时包含良性和恶意意图，直接生成容易导致过度拒绝；需要先分离意图再针对性推理。
**怎么做**：
- **意图分解（Intent Decomposition）**：LLM agent 识别查询中的显性和隐性意图，区分良性 vs 潜在恶意
- **初始 CoT 生成**：单 agent 生成基线推理链和回复，作为后续审议的起点

### 关键设计 2：审议阶段（Deliberation）

**功能**：多 agent 迭代扩展安全推理链。
**为什么**：单次生成容易遗漏策略覆盖，且推理可能不完整或有偏差。多 agent 交叉审查可提升覆盖率和可靠性。
**怎么做**：
- 每轮审议中，一个 agent 评估已有推理链，判断是否需要补充新的安全策略分析
- 如需要，agent 提出新 thoughts 并更新 response
- 迭代持续直到 agent 达成共识（如"I agree with previous agent"）或耗尽预算
- 使用 5 类安全策略：仇恨/骚扰/暴力、欺诈/欺骗、人身伤害、非法活动、尊重/有用性

### 关键设计 3：精炼阶段（Refiner）

**功能**：作为独立第三方评估者清理审议输出。
**为什么**：审议可能产生三类噪声——(a) 欺骗性思维（看似合理但结论错误），(b) 冗余/重复思维（导致 overthinking 和过度拒绝），(c) 策略不一致。
**怎么做**：Refiner agent 汇总所有轮次 thoughts → 逐条评估 → 删除噪声 → 输出简洁一致的最终 CoT + response。灵感来自 Irving et al. (2018) 的"AI safety via debate"。

### 关键设计 4：Ear-whisperer 偏好数据生成

**功能**：为 DPO 训练生成高质量 selected/rejected CoT 对。
**为什么**：标准采样方法中，SFT 后的模型生成的 selected 和 rejected CoT 质量差异极小（策略忠实度接近），无法为 DPO 提供有效梯度信号。
**怎么做**：
- 引入"ear-whisperer" agent 生成对抗性不良信念前缀（bad beliefs）
- 生成 rejected CoT 时，将 bad belief 前缀拼接到输入中，引导模型产生策略违反的推理
- 采用迭代 ICL 策略：反复精炼 bad belief 直到能有效诱导策略违反
- 使用 ShieldGemma 作为评分函数评估 bad belief 质量

### 训练策略
- Agent 模型：Mixtral 8x22B（所有 agent 统一使用）
- 数据源：BeaverTails 5000 条安全提示 + Alpagasus 5000 条通用提示
- SFT：QLoRA 4-bit 量化，在 Mixtral-7B 和 Qwen2.5-7B 上微调
- 效率：异步 LLM 查询（AsyncInferenceClient），4×A100，~35 秒/prompt

## 实验关键数据

### 主实验：SFT 后模型安全性评估（Table 2）

| 评估维度 | 数据集 | Mixtral Base | SFTOG | **SFTDB (AIDsafe)** |
|---------|--------|-------------|-------|---------------------|
| **安全性（域内）** | BeaverTails | 76.00% | 79.57% | **96.00%** |
| **安全性（域外）** | WildChat | 31.00% | 33.50% | **85.95%** |
| **越狱鲁棒性** | StrongREJECT | 51.09% | 67.01% | **94.04%** |
| **过度拒绝准确率** | XSTest | 98.80% | 87.60% | **91.84%** |
| **通用能力** | MMLU | 35.42% | 31.38% | **34.51%** |

| 评估维度 | 数据集 | Qwen Base | SFTOG | **SFTDB (AIDsafe)** |
|---------|--------|----------|-------|---------------------|
| **安全性（域内）** | BeaverTails | 94.14% | 87.95% | **97.00%** |
| **安全性（域外）** | WildChat | 95.50% | 59.42% | **96.50%** |
| **越狱鲁棒性** | StrongREJECT | 72.84% | 59.48% | **95.39%** |
| **过度拒绝准确率** | XSTest | 99.20% | 98.00% | 93.60% |
| **通用能力** | MMLU | 75.78% | 55.73% | 60.52% |

### 消融实验：CoT 数据质量评估（Table 1）

| 指标 | LLMZS（单模型） | **AIDsafe** | 提升 |
|------|----------------|------------|------|
| Relevance | 4.66 | **4.68** | +0.43% |
| Coherence | 4.93 | **4.96** | +0.61% |
| Completeness | 4.86 | **4.92** | +1.23% |
| **CoT 策略忠实度** | 3.85 | **4.27** | **+10.91%** |
| Response 策略忠实度 | 4.85 | **4.91** | +1.24% |
| Response-CoT 一致性 | 4.99 | **5.00** | +0.20% |

### 关键发现

1. **安全推理 >> 传统安全训练**：Mixtral SFTDB 域外安全性比 Base 提升 54.95pp（31% → 85.95%），而传统 SFTOG 仅提升 2.5pp
2. **传统安全训练可能"覆盖"原有安全性**：Qwen SFTOG 在 WildChat 上从 95.5% 暴跌到 59.42%，但 SFTDB 保持 96.5%——说明安全推理让模型"理解策略"而非"记住模式"
3. **越狱鲁棒性飞跃**：仅用 5000 安全样本训练，未接触任何越狱数据，Mixtral 越狱安全率从 51% → 94%
4. **CoT 策略忠实度是最大差异点**：AIDsafe 比单模型生成在策略忠实度上提升 10.91%，Pairwise 评测（两个独立 auto-grader）中 AIDsafe 胜率远超 LLMZS
5. **Ear-whisperer 有效拉大偏好差距**：标准采样的 selected/rejected 策略忠实度几乎无差别，ear-whisperer 方法成功制造了显著的分布偏移

## 亮点与洞察

- **"安全推理"vs"安全分类"**：核心洞察是让 LLM 理解"为什么不安全"比记住"什么不安全"更泛化——5000 样本即可泛化到未见过的攻击类型，这在传统安全训练中不可能实现
- **多 agent 审议弥补单模型推理不足**：用普通 Mixtral 8x22B 通过多 agent 协作达到了接近强推理模型的 CoT 质量，是"集体智慧"在安全推理中的成功应用
- **Overthinking 问题的系统性解决**：Refiner 阶段专门处理重复/冗余思维，直接缓解了安全训练中的过度拒绝问题
- **Ear-whisperer 的巧妙设计**：不是简单地生成有害回复，而是通过"信念注入"产生看似合理但推理有缺陷的 CoT——这种"对比学习"对建模"好推理 vs 坏推理"非常有效

## 局限与展望

1. **安全策略覆盖有限**：仅用 5 类策略，现实世界的安全风险远不止这些（隐私、偏见、版权等未覆盖）
2. **仅一种 agent 模型**：所有 agent 都用 Mixtral 8x22B，未探索异构 agent（不同模型各有所长）的效果
3. **审议仅限 2 agents**：未尝试更复杂的圆桌多 agent 设置
4. **SFT 起点不够理想**：直接在 instruction-tuned 模型上做安全 SFT，理想情况应先在 base 模型上做 CoT warm-up
5. **高度安全化模型的审议困难**：如果 agent 模型本身有严格 guardrails，可能在审议有害查询时拒绝配合，导致流程中断
6. **通用能力退化（Qwen）**：MMLU 从 75.78% 降到 60.52%，说明安全性-能力权衡仍未完全解决

## 相关工作与启发

### vs Deliberative Alignment (Guan et al., 2024)
Deliberative Alignment 让模型在推理时参考安全策略文档，但依赖模型自身的推理能力。AIDsafe 的优势在于**将推理能力外包给多 agent 系统**，降低了对单一强模型的依赖。但 DA 是推理时方案、AIDsafe 是数据生成方案，二者可互补。

### vs Constitutional AI (Bai et al., 2022b)
CAI 用 AI 反馈替代人类反馈实现安全对齐，但缺乏显式的策略推理链——模型学到"什么是安全的"但不一定理解"为什么"。AIDsafe 通过嵌入推理链**让模型学习安全决策的推理过程**，泛化能力更强（域外安全性提升 54.95pp vs CAI 的有限泛化）。

### vs DeepSeek-R1 / OpenAI o1
这些强推理模型的安全推理能力来自大规模 RL 训练，成本极高。AIDsafe 证明**用多 agent 协作 + 数据工程也能在 7B 模型上实现接近的安全推理效果**，为开源社区提供了可行路径。

## 评分
- 新颖性: ⭐⭐⭐⭐ 多 agent 审议用于安全 CoT 数据生成是新颖且自然的思路，ear-whisperer 偏好数据方案尤其巧妙
- 实验充分度: ⭐⭐⭐⭐ 数据质量评估（3 维度 + 忠实度 + pairwise）+ 下游训练（2 模型 × 4 基准）+ 偏好数据分析，全链路验证充分
- 写作质量: ⭐⭐⭐⭐ 框架设计清晰，各阶段动机和设计选择解释到位，ethical consideration 部分坦诚讨论了 ear-whisperer 的潜在滥用风险
- 价值: ⭐⭐⭐⭐⭐ 为开源 LLM 的安全推理训练提供了完整的数据生成 pipeline，5000 样本即实现域外安全泛化 54.95pp 提升——实际应用价值极高

<!-- RELATED:START -->

## 相关论文

- [Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](../../NeurIPS2025/llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)
- [笔记1: CoT是幻觉吗？数据分布角度](../../NeurIPS2025/llm_reasoning/is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [Ad-Hoc Human-AI Coordination Challenge (AH2AC2)](../../ICML2025/llm_reasoning/ad-hoc_human-ai_coordination_challenge.md)

<!-- RELATED:END -->
