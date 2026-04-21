---
title: >-
  [论文解读] What Makes a Good Natural Language Prompt?
description: >-
  [ACL 2025][LLM/NLP][提示学习] 通过元分析150+篇prompting文献，提出包含6个维度21个属性的以属性为中心的prompt质量评估框架，并通过推理任务实验发现：单属性增强常常优于多属性组合，且在属性增强数据上微调可进一步提升模型推理能力。
tags:
  - ACL 2025
  - LLM/NLP
  - 提示学习
  - 元分析
  - 属性框架
  - 认知负荷理论
  - 指令微调
---

# What Makes a Good Natural Language Prompt?

**会议**: ACL 2025  
**arXiv**: [2506.06950](https://arxiv.org/abs/2506.06950)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: prompt质量评估, 元分析, 属性框架, 认知负荷理论, 指令微调

## 一句话总结

通过元分析150+篇prompting文献，提出包含6个维度21个属性的以属性为中心的prompt质量评估框架，并通过推理任务实验发现：单属性增强常常优于多属性组合，且在属性增强数据上微调可进一步提升模型推理能力。

## 研究背景与动机

**领域现状**：Prompt已成为人与LLM交互的主要接口，但"什么是好prompt"缺乏系统化共识。现有研究多提出零散的prompting技术（CoT、few-shot等），也有OpenAI、Anthropic等公司发布实践指南，但这些都是碎片化的经验总结。

**现有痛点**：
1. 缺乏统一的属性级框架来系统化理解和比较各种prompting策略
2. 现有评估以结果为中心（outcome-centric），仅看任务性能指标，不关注prompt本身质量
3. 各属性是否具有跨模型、跨任务的普适性未被验证
4. 多个属性的交互效应和组合影响几乎未被研究

**核心矛盾**：结果导向的prompt优化可能产生对机器友好但人类难以理解的prompt，带来对齐、透明度和可维护性问题；而属性导向的评估虽可解释，但缺少系统化的理论框架。

**本文目标** 建立一个统一的、以属性为中心的prompt质量评估框架，回答三个问题：(1)好prompt应具备哪些属性？(2)这些属性如何影响不同模型和任务？(3)多属性组合还是单属性增强更有效？

**切入角度**：从Grice会话准则、认知负荷理论、Gagne教学九事件等人文/认知科学理论出发，将prompt属性体系化，用元分析+实证实验验证。

**核心 idea**：自然语言prompt的质量可分解为6个维度21个可独立评估的属性，精准增强单个属性往往比堆砌多属性更有效。

## 方法详解

### 整体框架

研究分四个阶段：(1) 文献元分析：调研150+篇来自ACL/EMNLP/NAACL/ICLR/NeurIPS(2022-2025)的论文和企业博客，提取prompting建议并概念化为属性 → (2) 属性影响分析：统计各属性在不同模型和任务上的研究分布和效果 → (3) 属性相关性分析：在969条高质量prompt上评估21属性间的相关性 → (4) 推理任务案例研究：在prompting和fine-tuning两个设置下验证单/多属性增强的效果。

### 关键设计

1. **6维度21属性分类体系**:
    - 功能：为prompt质量提供完整的评估维度和可操作的属性定义
    - 核心思路：6个维度分别为——**I. 沟通与语言**（token数量、表达方式、交互参与度、礼貌性）、**II. 认知**（管理内在负荷、降低外在负荷、激发相关负荷）、**III. 指令**（目标设定、外部工具、元认知、示例、奖励机制）、**IV. 逻辑与结构**（结构逻辑、上下文逻辑）、**V. 幻觉**（幻觉感知、事实与创意平衡）、**VI. 责任**（偏见、安全、隐私、可靠性、社会规范）
    - 设计动机：借鉴Grice会话准则（沟通维度）、Sweller认知负荷理论（认知维度）、Gagne教学九事件（指令维度）等成熟人文理论，使框架既有理论支撑又可操作

2. **属性相关性分析方法**:
    - 功能：揭示高质量prompt中各属性间的共现和相关模式，推导出实践建议
    - 核心思路：收集969条高质量prompt（来自PE论文、Awesome ChatGPT Prompts、Alpaca、Natural Instructions等），用GPT-4o + Self-consistency在21个属性上逐一打分(1-10)，计算属性间相关系数。对平均分<5的属性对不做相关性分析以避免虚假相关
    - 设计动机：直接分析人类精心设计的prompt中属性的共现规律，为prompt优化提供"哪些属性应联合优化"的实证依据

3. **单/多属性增强的对比实验**:
    - 功能：验证实际推理任务中不同prompt属性增强策略的效果
    - 核心思路：以zero-shot CoT为基线，通过添加简单语句分别增强4个属性——Politeness（加"Please"）、Germane load（要求回忆先验知识）、Metacognition（要求自我验证）、Rewards（给予100美元奖励），测试单独和组合效果。在prompting之外还做了fine-tuning实验：在Alpaca-GPT-4o数据集上分别用礼貌/原始数据微调Qwen-2.5-7B-It
    - 设计动机：回答"属性是否越多越好"这一关键问题；fine-tuning实验验证属性增强是否可内化到模型中

## 实验关键数据

### 主实验

**属性增强Prompting结果**（Table 2，各任务准确率%）：

| 配置 | MMLU | CommonsenseQA | ARC-C | GSM8K |
|------|:----:|:----:|:----:|:----:|
| **Llama-3.1-8B-It** | | | | |
| Zero-shot CoT | 65.00 | 76.00 | 81.50 | 82.0 |
| + Politeness | **68.00**↑ | **83.50**↑ | **84.50**↑ | **87.5**↑ |
| + Germane load | 66.00↑ | 75.50↓ | 82.00↑ | 82.0 |
| + Metacognition | 61.00↓ | 81.50↑ | 81.00↓ | 81.5↓ |
| + Rewards | 64.00↓ | 80.50↑ | 82.00↑ | 84.0↑ |
| + Pol.+Ger.+Met. | 69.50↑ | 75.00↓ | 82.50↑ | 81.5↓ |
| **Qwen-2.5-7B-It** | | | | |
| Zero-shot CoT | 45.50 | 55.00 | 59.50 | 76.5 |
| + Metacognition | **52.50**↑ | **56.50**↑ | **62.00**↑ | 83.5↑ |
| + Germane load | 44.50↓ | 56.50↑ | 53.50↓ | **90.0**↑ |
| + Politeness | 41.00↓ | 45.50↓ | 54.00↓ | 79.0↑ |
| + Rewards | 40.50↓ | 48.00↓ | 52.00↓ | 66.0↓ |
| **o3-mini** | | | | |
| Zero-shot CoT | 92.00 | 88.50 | 94.50 | 97.0 |
| + Politeness | 88.50↓ | 87.00↓ | 93.50↓ | 96.0↓ |
| + Germane load | 88.00↓ | 82.00↓ | 95.00↑ | 96.5↓ |

### 消融实验

**属性增强Fine-tuning结果**（Table 3，Qwen-2.5-7B-It微调后，礼貌数据/原始数据）：

| 配置 | MMLU | CQA | ARC | GSM8K | Avg. |
|------|:----:|:----:|:----:|:----:|:----:|
| Zero-shot CoT | 60.0/67.0 | 67.5/69.0 | 73.5/68.5 | 85.0/85.0 | 71.50/72.38 |
| + Politeness | **69.5**/62.5 | **72.5**/70.0 | **85.0**/79.5 | 85.0/88.5 | **78.00**/75.13 |
| + Metacognition | 61.0/54.0 | 72.0/68.0 | 75.0/71.0 | 86.5/89.0 | 73.63/70.50 |
| + Pol.+Ger.+Met. | 69.0/66.5 | **77.5**/**79.5** | **86.5**/**83.5** | 82.5/81.5 | **78.88**/**77.75** |

### 关键发现

1. **单属性增强常优于多属性组合**：Politeness对Llama在所有4个任务上均有效（+3~+7.5%），但加上Germane load后反而在CommonsenseQA上从83.50降到79.50
2. **不同模型对同一属性响应截然不同**：Politeness对Llama全面有效，对Qwen却在MMLU/CQA/ARC上均下降；Metacognition对Qwen全面有效，对Llama却在MMLU上下降4%
3. **强模型几乎不受属性增强影响**：o3-mini在所有属性增强下性能均下降，推测与其大量CoT训练导致额外属性使prompt偏离训练分布有关
4. **属性增强可通过fine-tuning内化**：在礼貌数据上微调后Qwen对加"Please"的prompt性能从45.5→69.5(MMLU)，平均从71.50→78.00，且在几乎所有属性增强配置下都优于原始数据微调
5. **属性相关性分析**：969条高质量prompt中发现17/210组强相关(≥0.7)，如token数量↔表达方式↔结构逻辑↔外在负荷，目标↔内在负荷↔相关负荷，幻觉感知↔可靠性

## 亮点与洞察

- **理论根基扎实**：从Grice会话准则、认知负荷理论、Gagne教学理论等出发构建框架，不是拍脑袋分类
- **"少即是多"的反直觉发现**：精准匹配1个属性 > 堆砌多个属性，这对prompt工程实践有重要指导意义
- **属性→模型的不对称性**：清楚揭示了"没有万能属性"——不同模型需要不同属性增强，呼应了"没有免费午餐"定理
- **fine-tuning与prompting的协同**：属性增强不仅可在推理时使用，还可通过微调内化进模型，且两者协同效果更好
- **开放问题有价值**：提出8个open questions（Oq1-Oq8）涵盖属性迁移性、因果关系、任务特异性等，为后续研究指明方向

## 局限与展望

- 文献调研虽涵盖150+篇，但人力有限难以覆盖所有相关工作
- 21属性的评估依赖GPT-4o作为judge，开源模型（DeepSeek R1、Mistral）格式遵循率仅65-71%，评估可靠性受限
- 多属性组合实验仅使用最简单的prompt增强形式（如加"Please"），未针对模型优化
- 责任维度（偏见/安全/隐私/社会规范等）过于宽泛，且文献支持极少
- 相关性分析仅在单一prompt集合上进行，不同任务场景下相关性可能不同
- 实验任务仅覆盖推理类（MMLU/CommonsenseQA/ARC-C/GSM8K），未验证生成/NLU等任务

## 相关工作与启发

- **vs 自动prompt优化（APE/OPRO/RLPrompt）**：自动优化关注搜索最优prompt文本，本文提供人类可理解的属性级设计框架，两者互补
- **vs prompt分析（LLMLingua等）**：现有分析聚焦prompt的结构组件或压缩，本文从属性质量的角度提供新视角
- **vs 企业prompt指南（OpenAI/Anthropic）**：实践指南给出具体建议（如"指定输出长度"），本文将其抽象化、体系化为可研究的属性
- **启发**：prompt设计应从经验驱动转向"属性诊断→精准增强"的工程化范式；属性增强+微调的协同路径值得进一步探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统化的prompt属性分类框架，理论根基扎实
- 实验充分度: ⭐⭐⭐ 元分析覆盖广但实证实验仅限推理任务，属性增强方式过于简单
- 写作质量: ⭐⭐⭐⭐ 框架层次清晰，开放问题有深度
- 价值: ⭐⭐⭐⭐ 对prompt工程研究和实践均有直接指导，属性框架可作为后续研究的基准

<!-- RELATED:START -->

## 相关论文

- [Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)
- [AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)

<!-- RELATED:END -->
