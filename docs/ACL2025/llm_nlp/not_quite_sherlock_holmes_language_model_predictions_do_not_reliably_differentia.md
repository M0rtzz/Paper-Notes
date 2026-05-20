---
title: >-
  [论文解读] Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events
description: >-
  [ACL 2025 (Findings)][LLM/NLP][事件可能性] 本文通过精心设计的最小对实验，揭示语言模型无法可靠区分"不可能事件"和"不太可能但可能的事件"——在对抗性条件下（可能句含不相关词、不可能句含相关词）…
tags:
  - "ACL 2025 (Findings)"
  - "LLM/NLP"
  - "事件可能性"
  - "语义相关性"
  - "语言模型预测"
  - "常识推理"
  - "最小对"
---

# Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events

**会议**: ACL 2025 (Findings)  
**arXiv**: [2506.06808](https://arxiv.org/abs/2506.06808)  
**代码**: [https://osf.io/r6xns/](https://osf.io/r6xns/)  
**领域**: LLM / NLP理解  
**关键词**: 事件可能性、语义相关性、语言模型预测、常识推理、最小对

## 一句话总结

本文通过精心设计的最小对实验，揭示语言模型无法可靠区分"不可能事件"和"不太可能但可能的事件"——在对抗性条件下（可能句含不相关词、不可能句含相关词），包括 Llama 3、Gemma 2、Mistral NeMo 在内的所有 35 个模型均表现低于随机水平。

## 研究背景与动机

**领域现状**：大量研究通过常识推理基准（如 HellaSwag、WinoGrande、PIQA）评估语言模型的世界知识，这些基准通常要求模型从多个选项中选出最可能的续写。先前工作（如 Kauf et al., 2023）表明语言模型可以区分可能和不可能的事件。

**现有痛点**：现有基准将"不正确"选项混为一谈——有些是真正不可能的事件，有些只是不太典型的事件。例如 HellaSwag 中的"错误"续写往往还是可能发生的事情，只是不太典型。这意味着当模型表现良好时，我们无法确定它是真的理解了事件的"可能性"，还是仅仅依赖"典型性"和"语义相关性"这两个更表面的线索。

**核心矛盾**：可能性（possibility）、典型性（typicality）和语义相关性（relatedness）在自然文本中高度纠缠——典型事件通常也是可能的，描述可能/典型事件的词也倾向于与上下文语义相关。如果不把这三个因素拆开，就无法确定模型到底学到了什么。

**本文目标**：系统地解开可能性、典型性和语义相关性三个因素的影响，回答一个关键问题——"语言模型能否在典型性和语义相关性不再是有用线索的情况下，仍然区分可能事件和不可能事件？"

**切入角度**：作者借用心理语言学中的最小对范式和结构化刺激材料（来自 Vega-Mendoza et al., 2021 和 Chow & Phillips, 2013），将关键词按 5 类条件操控：可能-典型-相关（PTR）、可能-非典型-相关（PAR）、可能-非典型-不相关（PAU）、不可能-非典型-相关（IAR）、不可能-非典型-不相关（IAU）。

**核心 idea**：通过正交操控可能性、典型性和语义相关性，构造最小对，证明语言模型在预测时严重依赖典型性和语义相关性启发式，当这些线索与可能性冲突时，模型几乎完全失败。

## 方法详解

### 整体框架

本文不是提出新模型，而是一个精心设计的实验研究。实验使用最小对范式：每对句子只有一个关键词不同，该关键词决定事件是否可能。输入两个句子给语言模型，看它是否对可能的那个句子赋予更高概率。通过 4 个实验递进深入：实验 1 测典型 vs 非典型/不可能；实验 2 测非典型但可能 vs 不可能（核心实验）；实验 3 用混合效应回归验证统计可靠性；实验 4 用 Pythia 系列模型研究缩放效应。

### 关键设计

1. **五维条件操控的最小对刺激**:

    - 功能：将可能性、典型性、语义相关性三个因素正交分离
    - 核心思路：以句子"the cure for the disease was discovered by the ___"为例，关键词设计为：doctor（PTR: 可能+典型+相关）、patient（PAR: 可能+非典型+相关）、guest（PAU: 可能+非典型+不相关）、medication（IAR: 不可能+非典型+相关）、stamp（IAU: 不可能+非典型+不相关）。英文刺激来自 Vega-Mendoza et al. (2021)，含 154 对；中文刺激来自 Chow & Phillips (2013)，含 57 对
    - 设计动机：先前研究要么不区分不可能和不典型，要么控制了语义相关性使其不作为混淆因素。本文需要涵盖相关性"帮倒忙"的情况——当不可能词恰好与上下文相关时

2. **逐步升级的对抗难度实验设计**:

    - 功能：从容易到困难逐步揭示模型的脆弱性
    - 核心思路：实验 1 先测最简单情况（PTR vs IAR/IAU），确认模型能完成典型 vs 不可能的区分。实验 2 核心升级为非典型但可能 vs 不可能（PAU vs IAR），这才是关键的 Sherlock Holmes 任务。实验 3 在 item 层面验证语义相关性和典型性的独立贡献。实验 4 使用 Pythia 全系列（14M-12B）10 个规模 × 20 个检查点 = 200 个模型来检验缩放趋势
    - 设计动机：单纯报告一个"模型做不好"的结论不够有说服力，逐步递进的实验设计让读者看到退化是如何逐渐发生的

3. **跨语言 + 跨规模的全面验证**:

    - 功能：验证发现的普遍性
    - 核心思路：在英语 35 个模型（BLOOM、Gemma、Llama、Mistral、OLMo、Qwen、SmolLM、XGLM、Yi、mGPT 等家族）和中文子集上复现实验。使用 LM Evaluation Harness 统一评测。额外使用 Pythia 的训练检查点追踪能力随训练的演化轨迹
    - 设计动机：单一语言或单一模型家族的结论可能是特例，跨语言和跨规模验证是证明结论普遍性的必要步骤

### 损失函数 / 训练策略

本文是纯评测研究，不涉及模型训练。评测方式是直接比较模型对两个句子赋予的对数似然：如果可能句子的概率更高则判定正确。所有模型使用预训练版本（base models），不使用指令微调版本，以确保测试模型的原始预测能力。

## 实验关键数据

### 主实验 — 实验 2: 非典型可能 vs 不可能

| 比较任务 | 35模型平均准确率 (英文) | 中文子集平均 | 说明 |
|---------|----------------------|------------|------|
| PAU vs IAU (不相关可能 vs 不相关不可能) | ~73% | ~70% | 两者都不相关时，模型表现较好 |
| PAR vs IAU (相关可能 vs 不相关不可能) | ~76% | — | 可能词相关时稍有帮助 |
| PAU vs IAR (不相关可能 vs 相关不可能) | **~28%** | **~35%** | **远低于50%随机水平！** |
| PAR vs IAR (相关可能 vs 相关不可能) | ~68% | — | 两者都相关时，模型恢复部分能力 |

### 消融实验 — 实验 3 混合效应回归

| 预测因素 | 对模型准确率的影响 | χ² 统计量 | p 值 |
|---------|------------------|----------|------|
| 可能词语义相关性↑ | 准确率显著提升 | 176.17 | <0.0001 |
| 不可能词语义相关性↑ | 准确率显著下降 | 197.58 | <0.0001 |
| 可能词典型性↑ | 准确率显著提升 | 128.76 | <0.0001 |
| 不可能词典型性↑ | 准确率显著下降 | 127.39 | <0.0001 |
| 可能词频率↑ | 准确率显著提升 | 394.32 | <0.0001 |
| 不可能词频率↑ | 准确率显著下降 | 302.54 | <0.0001 |

### 关键发现

- **最核心发现**：在 PAU vs IAR 条件下（不相关但可能 vs 相关但不可能），所有 35 个模型均表现在 50% 以下——即模型超过半数情况下把不可能事件判断为更可能的。例如模型认为"the car was given a parking ticket by the brake"比"...by the explorer"更可能
- **缩放无法解决问题**：Pythia 实验表明，在 PAU vs IAR 的最难任务上，模型准确率不随参数量或训练数据增加而提升，大模型甚至略差于小模型。这不是简单的"模型不够大"的问题
- **语义相关性是主要干扰因素**：将不可能词替换为与上下文相关的词会导致准确率暴跌。这表明模型在做预测时严重依赖"该词是否与上下文话题相关"这一启发式策略，而非真正理解事件的物理/逻辑约束

## 亮点与洞察

- **"Sherlock Holmes 任务"的精妙命名与实验设计**：以福尔摩斯名言"排除不可能之后，剩下的无论多么不可能，都是真相"为灵感设计任务，巧妙地将深层的认知科学问题转化为 NLP 可操作的评测。这种从心理语言学借鉴刺激材料的方法论值得推广
- **揭示 shortcut learning 的新证据**：模型不是真的"理解"了事件可能性，而是在使用语义相关性和典型性作为捷径。这一发现对医疗、法律等需要模型可靠区分可能/不可能的高风险应用场景有重要警示意义
- **cross-linguistic 一致性**：英文和中文（使用完全不同的刺激和语言结构）呈现几乎相同的模式，说明这是语言模型训练范式的根本限制而非个别语言的特殊情况

## 局限与展望

- **不可能性类型单一**：本文只研究了"动物性违反"（animacy violation）这一种不可能性，其他类型（物理违反、逻辑违反、时间违反等）是否有同样结论尚未验证
- **数据集规模小**：英文 154 对、中文 57 对，样本量有限
- **只测试了预训练模型**：指令微调模型和 RLHF 后的模型是否能通过 prompt 改善这一问题值得探索
- 作为改进方向，可以探索用对比学习或特定的常识推理训练数据来减轻对语义相关性启发式的依赖

## 相关工作与启发

- **vs Kauf et al. (2023)**: Kauf 等人发现模型整体上对不可能事件赋予更低概率，本文在更细粒度上挖掘发现这一能力在语义相关性混淆时完全崩溃，是对其结论的重要修正
- **vs Jones et al. (2022)**: Jones 等人在 GPT-3 上用 Glenberg & Robertson (2000) 的刺激发现模型"平均"能区分可能和不可能，但其刺激有意平衡了语义相关性。本文故意引入相关性不平衡来暴露问题
- **vs HellaSwag / WinoGrande 等基准**: 这些基准中的"错误"选项通常只是不典型而非不可能，因此高分不意味着模型真的有事件理解能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 将可能性、典型性、语义相关性正交分离是 NLP 评测中的新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 35 个模型 + 2 种语言 + Pythia 缩放分析 + 混合效应回归，极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文命名巧妙，实验递进清晰，论证逻辑严密
- 价值: ⭐⭐⭐⭐ 对语言模型"世界知识"能力的边界有重要认识价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](../../AAAI2026/llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[ACL 2025\] Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)
- [\[ACL 2025\] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)
- [\[ACL 2025\] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)

</div>

<!-- RELATED:END -->
