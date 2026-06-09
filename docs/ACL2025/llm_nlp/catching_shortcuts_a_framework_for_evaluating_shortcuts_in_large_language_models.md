---
title: >-
  [论文解读] Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models
description: >-
  [ACL 2025][LLM/NLP][捷径学习] 本文提出了一个系统化的框架来检测和评估大语言模型中的捷径学习（shortcut learning）现象，通过构造对比测试集和诊断指标，揭示LLM在多个NLP任务上依赖虚假相关而非真正理解语义的问题。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "捷径学习"
  - "虚假相关"
  - "评估框架"
  - "LLM鲁棒性"
  - "偏差检测"
---

# Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models

**会议**: ACL 2025  
**领域**: LLM/NLP  
**关键词**: 捷径学习, 虚假相关, 评估框架, LLM鲁棒性, 偏差检测

## 一句话总结
本文提出了一个系统化的框架来检测和评估大语言模型中的捷径学习（shortcut learning）现象，通过构造对比测试集和诊断指标，揭示LLM在多个NLP任务上依赖虚假相关而非真正理解语义的问题。

## 研究背景与动机

**领域现状**：大语言模型在各类NLP基准测试上取得了令人瞩目的成绩，但越来越多的研究发现，这些高分可能源于模型对训练数据中统计捷径（shortcuts）的利用，而非对任务的真正理解。捷径学习指模型依赖数据中的虚假相关性（spurious correlations）来做出预测，例如否定词与矛盾标签的关联、特定实体与答案的共现等。

**现有痛点**：虽然已有一些工作研究了特定任务（如NLI、QA）上的捷径现象，但缺乏一个统一的、可扩展的框架来系统性地检测和量化LLM中的捷径依赖。现有方法通常针对单一模型或单一任务，难以跨模型、跨任务进行对比分析。此外，随着LLM规模的增大，捷径行为是否减弱仍是一个未解问题。

**核心矛盾**：模型在标准测试集上的高准确率可能掩盖了其对捷径的依赖，而现有评估体系无法有效区分"真正的能力"和"捷径利用"。

**本文目标**：设计一个通用的捷径评估框架，能够（1）系统识别不同类型的捷径模式，（2）构造针对性的对比测试集，（3）提供可量化的诊断指标来衡量模型的捷径依赖程度。

**切入角度**：作者观察到捷径可以被形式化为输入特征与标签之间的虚假统计关联，通过控制这些特征的存在与否，可以精确度量模型的捷径依赖。

**核心 idea**：构建一个"捷径敏感测试框架"，通过对比原始样本和去除/翻转捷径特征后的样本，量化LLM对各类捷径的依赖程度。

## 方法详解

### 整体框架
框架包含三个核心阶段：（1）捷径模式识别——通过统计分析和启发式规则从训练数据中挖掘潜在的捷径模式；（2）对比测试集构造——针对每种捷径模式生成配对样本（保留捷径 vs 去除/翻转捷径）；（3）诊断评估——使用多维指标综合评估模型的捷径依赖性。

### 关键设计

1. **捷径模式分类体系（Shortcut Taxonomy）**:

    - 功能：对LLM可能利用的捷径进行系统分类
    - 核心思路：将捷径分为词汇级（如特定关键词与标签的关联）、句法级（如句子长度、特定句式与标签的关联）、语义级（如情感词极性、否定词出现与标签的关联）三个层次。对每种捷径类型定义形式化的检测标准和触发条件
    - 设计动机：不同层次的捷径对模型行为的影响机制不同，分类处理能更精准地定位问题

2. **对比样本生成器（Contrastive Sample Generator）**:

    - 功能：为每种捷径模式自动生成配对测试样本
    - 核心思路：给定原始样本 $x$ 和捷径特征 $s$，生成最小编辑的对比样本 $x'$，使得 $x'$ 不再包含捷径特征 $s$，但语义标签不变。使用基于规则的替换和LLM辅助的改写两种策略，确保对比样本的自然性和标签一致性
    - 设计动机：最小编辑保证了性能差异主要来源于捷径特征的有无，而非其他语义变化

3. **捷径依赖度量指标（Shortcut Dependency Score, SDS）**:

    - 功能：量化模型对特定捷径的依赖程度
    - 核心思路：定义 $SDS = \frac{Acc_{with} - Acc_{without}}{Acc_{with}}$，其中 $Acc_{with}$ 是包含捷径样本的准确率，$Acc_{without}$ 是去除捷径样本的准确率。SDS 越高说明模型越依赖该捷径。同时引入聚合指标 Overall SDS 来衡量模型在所有捷径类型上的综合表现
    - 设计动机：归一化的指标使得不同模型、不同任务之间可以公平对比

### 损失函数 / 训练策略
本文是一个评估框架而非训练方法，不涉及专门的损失函数。但作者提出了基于SDS反馈的去偏训练建议：在微调阶段对高SDS捷径类型的样本进行过采样或对抗增强。

## 实验关键数据

### 主实验

| 模型 | NLI-SDS↓ | QA-SDS↓ | Sentiment-SDS↓ | Avg-SDS↓ |
|------|----------|---------|----------------|----------|
| GPT-4 | 8.2 | 5.1 | 3.7 | 5.7 |
| LLaMA-2-70B | 15.6 | 12.3 | 9.8 | 12.6 |
| LLaMA-2-13B | 22.4 | 18.7 | 14.2 | 18.4 |
| LLaMA-2-7B | 28.1 | 24.5 | 19.6 | 24.1 |
| Mistral-7B | 19.8 | 16.2 | 12.1 | 16.0 |

### 消融实验

| 捷径类型 | LLaMA-2-7B SDS | LLaMA-2-70B SDS | 说明 |
|----------|----------------|-----------------|------|
| 词汇级捷径 | 31.2 | 17.8 | 小模型对词汇捷径依赖最严重 |
| 句法级捷径 | 24.5 | 13.2 | 句子长度偏差较为普遍 |
| 语义级捷径 | 22.7 | 11.4 | 否定词捷径影响显著 |
| 混合捷径 | 34.8 | 20.1 | 多种捷径叠加效应 |

### 关键发现
- 模型规模越大，捷径依赖总体越低，但即便是GPT-4级别的模型，在特定任务上仍存在明显的捷径依赖
- 词汇级捷径（如否定词、特定实体名）的影响最大且最普遍，句法级其次
- 经过RLHF训练的模型（如ChatGPT）比基础模型的SDS更低，表明对齐训练在一定程度上缓解了捷径问题
- 不同任务上的捷径模式差异很大，NLI任务的捷径问题最为严重

## 亮点与洞察
- 提出了一个统一的、可扩展的捷径评估框架，首次实现了跨模型、跨任务的系统性对比，这比以往针对单一任务的零散分析要全面得多
- 对比样本生成策略的"最小编辑"原则很巧妙，确保了评估的因果有效性
- SDS 指标简洁有效，可以直接用于任何新模型的捷径诊断，具有很高的实用价值

## 局限与展望
- 框架目前主要关注文本输入的捷径，未涉及多模态场景中的视觉捷径
- 对比样本的自动生成质量依赖于改写模型的能力，可能引入额外噪声
- 未深入探讨捷径的因果机制——模型为何学到特定捷径、如何从架构层面消除
- 可以扩展到代码生成、数学推理等更多任务上，检测LLM在这些领域的捷径行为
- 框架的可扩展性值得进一步验证——当任务类型和捷径类型大规模增长时，维护成本如何变化

## 相关工作与启发
- **vs McCoy et al. (2019) HANS**: HANS 专注于NLI任务中的句法启发式捷径，本文框架覆盖更多任务和捷径类型
- **vs Geirhos et al. (2020)**: Geirhos 关注CV中的纹理偏差，本文将类似思路系统化地引入NLP/LLM评估
- **vs Ribeiro et al. (2020) CheckList**: CheckList 是通用的NLP测试框架，本文专注于捷径这一特定问题，诊断更深入

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架思路清晰但捷径研究本身不算新方向
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个模型和任务，但缺少更大规模LLM的评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐ 对LLM评估领域有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)

</div>

<!-- RELATED:END -->
