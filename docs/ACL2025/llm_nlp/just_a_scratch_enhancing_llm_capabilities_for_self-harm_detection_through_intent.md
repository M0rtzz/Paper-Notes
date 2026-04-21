---
title: >-
  [论文解读] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement
description: >-
  [ACL 2025][LLM/NLP][self-harm detection] 提出 SHINES 数据集和 CESM-100 表情符号矩阵，通过区分社交媒体帖子中自伤表达的"随意提及"和"严重意图"，结合表情符号语境解读和多任务微调，将 LLM 自伤检测 F1 从 0.74（zero-shot）提升至 0.88（多任务+CESM-100），同时生成可解释的预测理由。
tags:
  - ACL 2025
  - LLM/NLP
  - self-harm detection
  - intent classification
  - emoji interpretation
  - mental health
  - multitask learning
---

# Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement

**会议**: ACL 2025  
**arXiv**: [2506.05073](https://arxiv.org/abs/2506.05073)  
**代码**: [资源页面](https://www.iitp.ac.in/%7eai-nlp-ml/resources.html#SHINES)  
**领域**: LLM/NLP  
**关键词**: self-harm detection, intent classification, emoji interpretation, mental health, multitask learning

## 一句话总结

提出 SHINES 数据集和 CESM-100 表情符号矩阵，通过区分社交媒体帖子中自伤表达的"随意提及"和"严重意图"，结合表情符号语境解读和多任务微调，将 LLM 自伤检测 F1 从 0.74（zero-shot）提升至 0.88（多任务+CESM-100），同时生成可解释的预测理由。

## 研究背景与动机

**领域现状**：社交媒体上的自伤检测是心理健康干预的重要环节，LLM 在压力检测、抑郁识别等任务上已有应用，但在自伤这一更细粒度的任务上表现仍不理想。

**现有痛点**：（1）现有 LLM 难以区分自伤表达中的"随意提及"（Casual Mention, CM）和"严重意图"（Serious Intent, SI）——例如 "might as well electrocute myself"（夸张修辞）vs "I don't even want to see tomorrow"（真实求助）；（2）表情符号在自伤语境中的语义变异被严重忽视——刀具🔪可能暗示自伤，但笑哭😂可能是用笑容掩饰痛苦；（3）现有 LLM（如 ChatGPT-4o 和 Gemini）对同一帖子的分类结果不一致。

**核心矛盾**：自伤信号具有高度隐含性和多模态性（文字+表情符号），简单的二分类无法捕捉意图层次的差异。

**本文目标** 构建区分 CM/SI 意图的数据集和表情符号资源，用多任务学习框架增强 LLM 对自伤帖子的检测和解释能力。

**切入角度**：将自伤检测分解为三个子任务（分类 + CM/SI span 提取 + 理由生成），并用 CESM-100 表情符号矩阵提供额外的语境信号。

**核心 idea**：通过意图分层（CM vs SI）和表情符号语境化矩阵（CESM-100）增强 LLM 的自伤检测能力，实现检测-解释一体化。

## 方法详解

### 整体框架

整体 pipeline 包含三个阶段：（1）输入增强——将帖子文本与 CESM-100 表情符号解释整合为统一输入；（2）多任务微调——同时训练自伤分类（主任务）和 CM/SI span 提取（辅助任务）；（3）理由生成——基于分类和 span 结果生成可解释的预测理由。

### 关键设计

1. **SHINES 数据集**:

    - 功能：构建包含 5,206 条社交媒体帖子的标注数据集，提供自伤标签、CM/SI span 标注和表情符号解释
    - 核心思路：从 Reddit 心理健康子版块（SuicideWatch、selfharm 等）收集 4,206 条帖子，用 Presidio 去除个人信息；3 名标注者独立标注自伤/非自伤标签（Fleiss' κ = 0.78）；每条帖子标注最多 3 个 CM 和 SI span（macro-F1: CM=0.66, SI=0.69）；额外用 ChatGPT-3.5 生成 1,000 条合成帖子（人工修订表情符号后质量验证 F1=58%）
    - 设计动机：现有数据集缺乏 CM/SI 意图层次的标注，无法训练模型区分修辞夸张和真实求助

2. **CESM-100（Centennial Emoji Sensitivity Matrix）**:

    - 功能：构建 100 个表情符号的自伤语境解释矩阵
    - 核心思路：每个表情符号标注通常含义、自伤语境含义、CM 关联程度（Low/Medium/High）和 SI 关联程度。例如：棺材⚰️（CM: Low, SI: High），笑哭😂（CM: High, SI: Medium），破碎的心💔（CM: Medium, SI: High）
    - 设计动机：表情符号在自伤语境中的语义与日常使用显著不同，模型需要这种领域特定的语义映射来减少误判。标注者间一致性达 Fleiss' κ = 0.71 (CM) / 0.75 (SI)

3. **多任务微调框架**:

    - 功能：同时优化自伤分类、CM/SI span 提取和理由生成三个任务
    - 核心思路：使用 decoder-only LLM（Llama 3、Mental-Alpaca、MentalLlama）的统一输入表示。自伤分类用二元交叉熵损失，span 提取用稀疏类别交叉熵损失（预测起止位置），理由生成用 prompting 引导模型引用 CM/SI span 和 CESM-100 解释
    - 设计动机：多任务学习让模型共享 CM/SI 的语义理解，从而提升主任务（自伤检测）的表现

4. **表情符号使用模式分析**:

    - 功能：分析表情符号在自伤/非自伤帖子中的使用差异
    - 核心思路：自伤帖子中表情符号组合更复杂（多符号组合在 SH 中远多于 NSH），SI 帖子偏好隐喻用法（Metaphorical > Direct），CM 帖子偏好直接用法（Direct > Metaphorical）
    - 设计动机：验证表情符号作为自伤检测信号的可行性和 CESM-100 设计的合理性

## 实验关键数据

### 主实验（Llama 3 各设置下的 F1）

| 设置 | SHC F1 | CMSE F1 | SISE F1 | RG SemSim |
|------|--------|---------|---------|-----------|
| Zero-shot | 0.74 | - | - | 0.75 |
| Few-shot | 0.79 | - | - | 0.80 |
| Fine-tuning (单任务) | 0.83 | - | - | 0.84 |
| MT Fine-tuning + CESM-100 | **0.88** | **0.85** | **0.84** | **0.88** |

### 消融实验（Llama 3）

| 配置 | SHC F1 | 说明 |
|------|--------|------|
| MT + CESM-100（完整） | 0.88 | 最佳 |
| MT w/o CESM-100 | 0.84 | 去除表情符号矩阵，-0.04 |
| 单任务 + CESM-100 | 0.86 | 去除多任务，-0.02 |
| 单任务 w/o CESM-100 | 0.83 | 基线微调 |

### 关键发现
- 多任务微调比单任务提升 +4% F1，比 few-shot 提升 +7%，比 zero-shot 提升 +12%（模型平均）
- CESM-100 贡献约 +4% F1 提升，统计检验显著（p=0.0198 for Llama, p=0.017 for Mental-Alpaca）
- 领域专用模型（Mental-Alpaca: 0.86, MentalLlama: 0.85）略低于通用模型 Llama 3（0.88），可能是因为 Llama 3 参数量更大
- 自伤帖子中表情符号组合呈现 "越复杂越倾向 SH" 的模式：单符号 SH:NSH=7815:5359，4+ 符号 SH:NSH=82:22

## 亮点与洞察
- CESM-100 表情符号矩阵是独特贡献——将表情符号从简单的情感信号提升为具有 CM/SI 维度的语境化自伤指标，可复用于其他心理健康 NLP 任务
- 多任务框架设计合理，CM/SI span 提取作为辅助任务不仅提升检测性能，还为理由生成提供结构化解释素材，实现了"可解释的自伤检测"

## 局限与展望
- 受硬件限制（NVIDIA K80, 24GB）仅评估了 7-8B 参数模型，未测试 GPT-4 等更大模型
- 合成数据（1000 条）的表情符号使用模式需人工修订，说明 LLM 生成的表情符号仍不够自然
- 数据集来源单一（Reddit），可能无法代表 Twitter、Instagram 等不同平台的表达风格
- CESM-100 的 100 个表情符号覆盖有限，新兴表情符号和跨文化差异未涉及

## 相关工作与启发
- **vs McBain et al. 2025**: 发现 GPT-4o 等模型在评估自杀意念严重程度时存在上偏，本文通过 CM/SI 意图分层直接针对这一问题
- **vs Grabb et al. 2024**: 指出 LLM 在心理健康紧急情况中可能造成伤害，本文的可解释理由生成有助于辅助而非替代专业判断
- **vs Yang et al. 2024 (MentalLlama)**: 领域专用模型虽有心理健康知识，但在本任务中不如通用 Llama 3 + 多任务微调

## 评分
- 新颖性: ⭐⭐⭐⭐ CESM-100 和 CM/SI 意图分层是有创意的贡献，但整体框架（多任务微调）较为标准
- 实验充分度: ⭐⭐⭐⭐ 三个模型 × 四种设置 + 两组消融 + 显著性检验，覆盖全面
- 写作质量: ⭐⭐⭐ 结构完整但可读性一般，数据集描述和方法论的章节组织较松散
- 价值: ⭐⭐⭐⭐ 对社交媒体心理健康监测有实际应用价值，SHINES 和 CESM-100 可作为社区资源

<!-- RELATED:START -->

## 相关论文

- [CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)
- [MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)
- [Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

<!-- RELATED:END -->
