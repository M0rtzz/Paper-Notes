---
title: >-
  [论文解读] ReLearn: Unlearning via Learning for Large Language Models
description: >-
  [ACL 2025][LLM安全][知识遗忘] ReLearn提出用"正向学习"替代传统的"逆向优化"来实现LLM知识遗忘，通过数据增强和微调pipeline使模型在遗忘目标知识的同时保持语言生成质量和流畅性，并设计了包含KFR、KRR和LS三个指标的综合评估框架。 领域现状：大语言模型在训练过程中吸收了海量数据…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "知识遗忘"
  - "机器遗忘"
  - "数据增强"
  - "语言质量保持"
  - "逆向优化"
---

# ReLearn: Unlearning via Learning for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.11190](https://arxiv.org/abs/2502.11190)  
**代码**: [GitHub](https://github.com/zjunlp/unlearn)  
**领域**: LLM安全  
**关键词**: 知识遗忘, 机器遗忘, 数据增强, 语言质量保持, 逆向优化  

## 一句话总结
ReLearn提出用"正向学习"替代传统的"逆向优化"来实现LLM知识遗忘，通过数据增强和微调pipeline使模型在遗忘目标知识的同时保持语言生成质量和流畅性，并设计了包含KFR、KRR和LS三个指标的综合评估框架。

## 研究背景与动机

**领域现状**：大语言模型在训练过程中吸收了海量数据，其中可能包含隐私信息、有害内容或需要被"遗忘"的知识。机器遗忘（Machine Unlearning）旨在让模型有选择性地遗忘特定知识，同时保持在其他任务上的能力。目前主流的LLM遗忘方法基于逆向优化，如梯度上升（Gradient Ascent, GA）或NPO（Negative Preference Optimization）。

**现有痛点**：逆向优化方法虽然能降低目标token的生成概率，但会产生严重的副作用——破坏模型的语言连贯性和后续token预测能力。模型在遗忘目标知识后，可能在相关或无关话题上产生乱码、重复、或语法错误的输出，整体语言质量严重退化。

**核心矛盾**：逆向优化的本质是"反向推动"模型的参数，这不仅影响目标知识的参数空间，还会波及模型的通用语言生成能力。此外，现有评估指标过度关注"是否忘记了目标知识"（上下文遗忘），而忽视了遗忘后回答的流畅性和相关性。

**本文目标**：(1) 设计一种不依赖逆向优化的遗忘方法，在遗忘目标知识的同时保持语言生成质量；(2) 建立更全面的评估框架，既评估遗忘效果又评估语言质量。

**切入角度**：逆向优化之所以破坏语言质量，是因为它在梯度方向上做了"反向"操作。换一个思路——如果给模型提供"不包含目标知识的替代回答"来正向微调，模型就既能遗忘原始知识，又不会破坏语言能力。

**核心 idea**：通过数据增强生成目标知识的替代回答（如"我不知道"、合理的替代事实），然后使用常规的正向微调（而非逆向优化）来让模型"学习"新的回答方式，从而"遗忘"旧知识。

## 方法详解

### 整体框架
ReLearn的流程分为三步：(1) **数据增强**：对需要遗忘的知识点生成替代回答数据（包括拒绝回答、替代事实等）；(2) **正向微调**：使用增强后的数据对模型进行标准微调，替换模型对目标知识的回答模式；(3) **综合评估**：使用KFR、KRR和LS三个指标全面评估遗忘效果。整个过程与标准的指令微调pipeline完全兼容，无需修改优化器或训练流程。

### 关键设计

1. **数据增强策略（Data Augmentation for Unlearning）**:

    - 功能：为需要遗忘的知识生成高质量的替代训练数据
    - 核心思路：对于每个需要遗忘的知识问答对 $(q, a)$，生成多种替代回答：(a) 拒绝型回答："我无法提供该信息"、"我没有相关知识"等；(b) 替代事实型回答：将原始答案替换为合理但不同的内容；(c) 错误引导型回答：故意提供错误但看似合理的信息。同时保留一组"保留集"数据（与遗忘目标无关的知识QA对），确保模型在遗忘过程中不丧失通用能力。
    - 设计动机：正向微调需要明确的目标标签，数据增强提供了"模型应该如何回答已遗忘知识的提问"的监督信号

2. **正向微调遗忘（Forward Fine-tuning for Unlearning）**:

    - 功能：通过标准训练过程实现知识遗忘，避免逆向优化的破坏性
    - 核心思路：将增强后的遗忘数据和保留集数据混合，使用标准的因果语言模型训练目标（交叉熵损失）进行微调。损失函数为 $\mathcal{L} = \mathcal{L}_{forget} + \lambda \mathcal{L}_{retain}$，其中 $\mathcal{L}_{forget}$ 是在替代回答上的损失，$\mathcal{L}_{retain}$ 是在保留集上的损失。训练过程与正常微调完全一致——模型只是在"学习"一套新的回答方式，而非被"反向推动"。
    - 设计动机：正向微调沿着梯度的正常方向更新参数，不会破坏模型已有的语言生成能力和连贯性

3. **综合评估框架（KFR + KRR + LS）**:

    - 功能：从知识遗忘、知识保持和语言质量三个维度全面评估遗忘效果
    - 核心思路：(a) **知识遗忘率（KFR）**：衡量模型在遗忘集上"不再知道"目标知识的程度，通过检查回答是否还包含目标信息来评估；(b) **知识保持率（KRR）**：衡量在遗忘目标知识后，模型在保留集上的知识是否仍然完整；(c) **语言分数（LS）**：使用独立的语言模型评估遗忘后生成文本的流畅性、连贯性和语法正确性。三个指标综合反映遗忘方法的整体质量。
    - 设计动机：现有指标只关注"忘没忘干净"，忽视了"忘完之后模型还能不能好好说话"这个同样重要的问题

### 损失函数 / 训练策略
采用标准的交叉熵损失，遗忘集和保留集数据按比例混合训练。使用学习率2e-5，训练轮数根据数据量调整（通常1-3个epoch）。支持Llama-3-8B-Instruct、Gemma-2-2B-IT、Llama-2-7B-Chat等多个基座模型。可选地引入DPO变体（ReLearn_DPO），将原始回答和替代回答构成偏好对进行训练。

## 实验关键数据

### 主实验
在KnowUnDo和TOFU基准上与逆向优化baseline对比（Llama-2-7B-Chat）：

| 方法 | KFR ↑ | KRR ↑ | LS ↑ | 综合排名 |
|------|-------|-------|------|---------|
| 原始模型（无遗忘） | 0.0 | 100.0 | 高 | - |
| Gradient Ascent (GA) | 72.5 | 45.3 | 低（严重退化） | 4 |
| NPO | 68.9 | 52.1 | 中（有退化） | 3 |
| SURE | 65.4 | 58.7 | 中 | 3 |
| Memflex | 59.2 | 61.3 | 中高 | 3 |
| **ReLearn** | **78.3** | **82.6** | **高（接近原始）** | **1** |
| **ReLearn_DPO** | **81.1** | **79.8** | **高** | **1** |

### 消融实验
不同数据增强策略和组件的贡献分析：

| 配置 | KFR | KRR | LS | 说明 |
|------|-----|-----|-----|------|
| 仅拒绝型回答 | 71.2 | 85.3 | 高 | 保守但遗忘不彻底 |
| 仅替代事实 | 76.8 | 78.1 | 高 | 遗忘更深但可能影响相关知识 |
| 混合增强（完整方法） | 78.3 | 82.6 | 高 | 最佳balance |
| 去掉保留集 | 80.1 | 61.4 | 中高 | 遗忘增强但保持能力受损 |
| ReLearn_DPO变体 | 81.1 | 79.8 | 高 | 偏好优化进一步加强遗忘 |

### 关键发现
- **逆向优化确实破坏语言连贯性**：通过mechanistic analysis（机制分析），本文清楚地展示了GA和NPO如何扰乱模型的注意力模式和MLP激活，导致后续token预测失调。ReLearn由于仅使用正向更新，完全避免了这一问题。
- **保留集训练对维持通用能力至关重要**：去掉保留集后KRR大幅下降约21%，说明在遗忘过程中持续"提醒"模型其他知识是必要的。
- **DPO变体在遗忘强度上更优**：通过将原始回答和替代回答构成偏好对，模型能更明确地学习"应该避免什么、应该说什么"，KFR提升约3%。
- 在不同基座模型（Llama-3、Gemma-2）上效果一致，方法具有良好的泛化性。

## 亮点与洞察
- **用"学习"实现"遗忘"的逆向思维**：不通过逆向优化破坏权重，而是通过正向微调教会模型新的回答方式，优雅地避免了语言质量退化问题。这种思路可以迁移到其他需要"修改"模型行为而非"删除"能力的场景。
- **LS指标的提出**：首次系统性地关注遗忘后的语言质量，这填补了评估框架中的重要空白。此前很多遗忘方法在标准指标上表现不错，但实际生成质量很差。
- **mechanistic analysis部分**：通过可视化注意力模式和中间层激活，直观解释了为什么逆向优化会破坏语言生成，这为后续遗忘研究提供了理论基础。

## 局限与展望
- 数据增强的质量依赖prompt设计，不同类型的知识可能需要不同的增强策略
- 目前验证的遗忘规模相对较小（几十到几百条知识），大规模遗忘（如遗忘整个领域的知识）的效果未知
- 替代事实型回答可能引入新的虚假知识，存在安全隐患
- 未考虑知识间的关联性——遗忘A知识可能导致与A强相关的B知识也受到影响
- **改进方向**：可以研究基于知识图谱的精细化遗忘范围控制，以及探索遗忘后模型的可恢复性问题（遗忘是否可以被逆向还原）

## 相关工作与启发
- **vs Gradient Ascent (GA)**: GA直接最大化目标token的损失，简单粗暴但破坏性强；ReLearn通过正向学习替代，保持了模型稳定性
- **vs TOFU**: TOFU提供了标准化的遗忘评估基准，但其评估指标仍侧重于上下文遗忘，ReLearn的评估框架是对TOFU的有力补充
- **vs KnowUnDo**: KnowUnDo关注隐私保护场景下的知识遗忘，是本文的重要实验基准之一
- 本文团队（浙大NLP）在知识编辑和遗忘方向有持续贡献，还在SemEval 2025 Unlearning Challenge中获得第二名

## 评分
- 新颖性: ⭐⭐⭐⭐ "正向学习实现遗忘"的思路简洁且有效
- 实验充分度: ⭐⭐⭐⭐ 多基座模型、多基准、mechanistic analysis，但遗忘规模较小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，动机充分
- 价值: ⭐⭐⭐⭐ 对LLM安全和隐私保护有实际意义，评估框架贡献突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)
- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](../../ICML2025/llm_safety/learning_safety_constraints_for_large_language_models.md)

</div>

<!-- RELATED:END -->
