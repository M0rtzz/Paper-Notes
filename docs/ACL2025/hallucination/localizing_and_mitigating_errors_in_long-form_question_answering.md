---
title: >-
  [论文解读] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering
description: >-
  [ACL 2025 (Findings)][幻觉检测][长文问答] 本文构建了首个包含 span 级别错误标注的 LFQA 幻觉数据集 HaluQuestQA（698 QA 对，4.7k 错误标注，5 种错误类型），训练了一个自动反馈模型来检测不完整信息的错误 span 并生成解释，最后提出 Error-informed Refinement 方法利用反馈信号精炼答案，将幻觉减少约 3%，且人类评估中 84% 的用户更偏好精炼后的答案。
tags:
  - "ACL 2025 (Findings)"
  - "幻觉检测"
  - "长文问答"
  - "细粒度标注"
  - "反馈模型"
  - "答案精炼"
---

# Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering

**会议**: ACL 2025 (Findings)  
**arXiv**: [2407.11930](https://arxiv.org/abs/2407.11930)  
**代码**: 有（论文提供）  
**领域**: 幻觉检测  
**关键词**: 长文问答, 幻觉检测, 细粒度标注, 反馈模型, 答案精炼

## 一句话总结

本文构建了首个包含 span 级别错误标注的 LFQA 幻觉数据集 HaluQuestQA（698 QA 对，4.7k 错误标注，5 种错误类型），训练了一个自动反馈模型来检测不完整信息的错误 span 并生成解释，最后提出 Error-informed Refinement 方法利用反馈信号精炼答案，将幻觉减少约 3%，且人类评估中 84% 的用户更偏好精炼后的答案。

## 研究背景与动机

**领域现状**：长文问答（LFQA）旨在为复杂问题提供全面深入的回答，LLM 在这方面表现出色但容易产生幻觉和事实不一致。简单的 BLEU/ROUGE 指标无法有效评估长文答案质量。

**现有痛点**：(1) 现有幻觉检测工作主要关注事实性错误，忽略了回答的完整性、相关性等多维度评估需求；(2) 现有评估要么是粗粒度的整体偏好判断，要么需要参考文本；(3) 长文答案的 span 级别错误标注在 LFQA 领域几乎为空白。已有的反馈模型要么无法提供细粒度错误反馈，要么依赖金标准文本。

**核心矛盾**：LLM 生成的长文答案可能在表面上看起来流畅且有说服力，但在信息完整性、参考文献质量等方面存在隐蔽缺陷——这些缺陷需要专家级标注才能发现，而专家标注成本高昂。

**本文目标**：(1) 建立细粒度的 LFQA 错误分类体系和标注数据集；(2) 训练一个无需参考文本的自动反馈模型；(3) 利用反馈信息自动精炼答案以减少幻觉。

**切入角度**：作者从人类专家如何评审长文答案出发，定义了五种关键错误类型，并观察到答案最大的问题不是事实错误，而是缺乏全面性和参考文献质量差。

**核心 idea**：通过"检测错误→生成细粒度反馈→利用反馈精炼答案"的闭环管道来系统性地减少 LFQA 中的幻觉。

## 方法详解

### 整体框架

管道分为三个阶段：(1) 构建 HaluQuestQA 数据集，领域专家对人类和模型生成的答案进行 span 级别错误标注；(2) 在该数据集上训练反馈模型，预测每个句子的错误状态和解释；(3) 用反馈模型的输出驱动精炼模型改善答案质量。

### 关键设计

1. **HaluQuestQA 数据集构建**:

    - 功能：提供首个 LFQA 领域的 span 级别多维度错误标注数据
    - 核心思路：从 Reddit ELI5 论坛抓取 2022.11-2023.03 的新问题（避免数据泄漏），覆盖物理、化学、生物、技术、经济、历史、法律 7 个领域。通过 Prolific 平台招募领域专家（22-32 岁，美英本科/研究生），每个领域 3 位专家标注 35-50 个问答对。人类答案来自 Reddit，模型答案由 GPT-4 零样本生成。定义五种错误类型：问题误解（question misconception）、事实性（factuality）、完整性（completeness）、相关性（relevance）、参考文献质量（references）
    - 设计动机：现有数据集没有针对 LFQA 的细粒度 span 级别标注，这是评估和改进的基础

2. **反馈模型（Error Feedback Model）**:

    - 功能：给定问答对，预测每个句子的 [Complete]/[Incomplete] 标签及错误解释
    - 核心思路：基于 LLaMA2-13B 微调，使用 HQ2A 中的完整性标注数据（509 样本）。推理时采用 nucleus sampling（p=0.9）生成 20 个候选输出，通过两阶段一致性筛选选出最可靠的反馈。第一阶段为标签一致性 $\mathcal{S_{TC}} = \frac{1}{n}\sum_{s=1}^n \mathbf{1}_{t_i = t_s}$（相同标签序列的比例）；第二阶段为理由一致性 $\mathcal{S_{RC}} = \frac{1}{m_i}\sum_{k=1}^{m_i}\sum_{s=1}^n \mathbf{1}_{w_{ik} \in j_s}$（各 token 在其他解释中出现的频率）
    - 设计动机：反馈模型自身也会产生幻觉（约 20% 输出含编造的网页链接），采样+一致性筛选可将幻觉率降低至 5-10%

3. **Error-Informed Refinement (EIR)**:

    - 功能：利用细粒度反馈精炼原始答案
    - 核心思路：用 LLaMA2-13B-chat 作为精炼模型，输入为原始问答对和反馈模型提供的细粒度错误信息（错误位置、原因、置信度分数）。模型零样本生成改进后的答案。对比设置包括：Improve（无反馈直接改进）、Generic（通用反馈）和 EIR（细粒度反馈），以验证反馈粒度的重要性
    - 设计动机：细粒度反馈比粗粒度反馈更能精准指导模型修正答案中的具体不足

### 损失函数 / 训练策略

反馈模型使用标准的序列到序列微调损失，batch size 4，学习率 2e-5，序列长度 1024，训练 5 个 epoch。另外还使用 DPO 算法对 LLaMA2-13B-chat 进行偏好优化，使用 LoRA（r=256, alpha=128）。

## 实验关键数据

### 主实验（答案精炼效果）

| 数据集 | 方法 | TigerScore(↓) | 幻觉样本比例(↓) | F1(↑) |
|--------|------|-------------|---------------|-------|
| HQ2A | Baseline | 19.61 | 0.63 | - |
| HQ2A | Improve | 1.31 | 0.05 | 0.97 |
| HQ2A | Generic | 1.31 | 0.05 | 0.97 |
| HQ2A | **EIR** | **0.65** | **0.03** | **0.97** |
| HQ2A | 人类反馈 | 2.61 | 0.09 | 0.94 |

### 人类评估

| 数据集 | 类型 | 全面性 | 偏好率 |
|--------|------|--------|--------|
| HQ2A | Baseline | 0% | 7.84% |
| HQ2A | Refined | **100%** | **92.16%** |
| ASQA | Baseline | 82% | 40% |
| ASQA | Refined | **100%** | **60%** |
| ELI5 | Baseline | 38% | 0% |
| ELI5 | Refined | **100%** | **100%** |

### 关键发现

- EIR 不仅超越了粗粒度反馈，甚至超越了专家人类反馈（HQ2A 上 TigerScore 0.65 vs 2.61），说明模型化的细粒度反馈在精炼引导上比人类更稳定
- 反馈模型在一致性分数 >0.80 时预测高度对齐人类标注；<0.80 时不确定性显著增加
- 答案最大的问题是完整性和参考文献质量，而非事实错误——GPT-4 答案在事实性和相关性上得分较高，但在全面性上仍有明显不足
- DPO 优化的精炼模型在 ASQA 和 ELI5 上获得更好的幻觉分数，但未减少幻觉样本数，说明 DPO 帮助修正严重错误但不擅长改善覆盖面
- 专家标注中发现 ~40% 的技术和经济领域问题本身含有误解，提示 LFQA 评估需要先评估问题质量

## 亮点与洞察

- **反馈模型超越人类反馈**是一个重要发现：自动化的稳定细粒度反馈比人类专家的一次性点评更适合指导答案精炼。这说明在"评估—改进"循环中，一致性比专业深度更关键
- **采样+一致性筛选**是处理反馈模型自身幻觉的实用技巧，将编造参考文献的比例从 20% 降到 5-10%，可迁移到任何需要可靠生成的场景
- **五维度错误分类**中"完整性"和"参考文献"是新洞察——现有工作过度关注事实性，忽略了答案实用性的其他维度

## 局限与展望

- 仅关注 LFQA 任务，尚未验证在摘要生成、翻译等其他长文生成任务上的效果
- 反馈模型基于 LLaMA2-13B，较新的更大模型可能获得更好的错误检测精度
- 仅实验了 LLaMA2-13B-chat 作为精炼模型，其他指令跟随能力不同的模型可能结果不同
- 数据来源限于 Reddit 平台的英语内容，其他语言和领域的泛化性有待验证
- 可以考虑迭代精炼——用精炼后的答案再次检测错误并进一步改进

## 相关工作与启发

- **vs TigerScore**: TigerScore 是参考无关的评估指标，但不提供可用于精炼的结构化反馈；本文的反馈模型既能评估又能指导改进
- **vs Fine-grained RLHF (Wu et al., 2023)**: 他们训练多个奖励模型对应不同错误类型，复杂且计算密集；本文用单一反馈模型覆盖多维度评估
- **vs Self-Refine (Madaan et al., 2023)**: Self-Refine 依赖模型的自我反馈，容易产生"自我幻觉"；本文用独立训练的反馈模型提供外部监督

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 LFQA span 级别错误标注数据集和"检测-反馈-精炼"管道的系统性整合
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种对比方案、人类评估、DPO 消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图文配合好，错误类型示例直观
- 价值: ⭐⭐⭐⭐ 数据集和方法对 LFQA 社区有重要参考价值，反馈+精炼的范式可广泛应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[ACL 2025\] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)
- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](../../CVPR2026/hallucination/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](../../ICML2026/hallucination/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)

</div>

<!-- RELATED:END -->
