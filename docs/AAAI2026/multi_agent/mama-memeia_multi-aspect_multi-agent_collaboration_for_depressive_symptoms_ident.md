---
title: >-
  [论文解读] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes
description: >-
  [AAAI 2026][多智能体][抑郁症检测] 本文提出 MAMAMemeia，一个基于认知分析疗法（CAT）能力框架的多智能体多方面协作讨论框架，用于从社交媒体表情包中识别抑郁症状，同时引入 RESTOREx 资源（含 LLM 生成和人工标注的解释）…
tags:
  - "AAAI 2026"
  - "多智能体"
  - "抑郁症检测"
  - "表情包分析"
  - "多智能体协作"
  - "认知分析疗法"
  - "大语言模型"
---

# MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes

**会议**: AAAI 2026  
**arXiv**: [2512.25015](https://arxiv.org/abs/2512.25015)  
**代码**: 无  
**领域**: 医学图像 / NLP  
**关键词**: 抑郁症检测, 表情包分析, 多智能体协作, 认知分析疗法, 大语言模型

## 一句话总结

本文提出 MAMAMemeia，一个基于认知分析疗法（CAT）能力框架的多智能体多方面协作讨论框架，用于从社交媒体表情包中识别抑郁症状，同时引入 RESTOREx 资源（含 LLM 生成和人工标注的解释），在 macro-F1 上超越 30+ 种方法 7.55%。

## 研究背景与动机

**领域现状**：表情包（memes）已从纯粹的幽默媒介演变为用户表达各种情感（包括抑郁情绪）的工具。社交媒体上越来越多的抑郁主题表情包为心理健康监测提供了新的数据源。

**现有痛点**：（1）表情包是多模态的（图像+文本），且常使用隐喻、反讽等修辞手法，单一模型难以理解其深层含义；（2）抑郁症状的识别需要专业的心理学知识——不同的表情包可能暗示不同的抑郁症状（如无望感、社交退缩、自我贬低等）；（3）现有方法缺乏可解释性——黑盒分类无法告诉临床工作者为什么认为某个表情包表达了抑郁。

**核心矛盾**：表情包的隐晦表达方式与抑郁症状识别所需的专业精确性之间的鸿沟。

**本文目标**：（1）构建可解释的抑郁症状检测框架；（2）引入临床心理学知识指导分析；（3）提供标注解释的数据资源。

**切入角度**：借鉴认知分析疗法（CAT）的临床方法论，将不同的心理学分析维度分配给不同的 LLM 智能体，通过多智能体讨论达成综合判断。

**核心 idea**：用多个 LLM 智能体分别扮演不同的 CAT 能力角色（如情感分析师、认知评估师、行为观察者），通过协作讨论整合多方面分析来识别表情包中的抑郁症状。

## 方法详解

### 整体框架

MAMAMemeia 接收一个表情包（图像+文本）作为输入，由多个 LLM 智能体分别从不同的 CAT 能力维度分析该表情包，然后通过结构化的讨论/辩论过程整合各方面意见，最终输出抑郁症状的分类标签和解释性理由。

### 关键设计

1. **多方面智能体设计（Multi-Aspect Agents）**:

    - 功能：每个智能体专注于一个心理学分析维度。
    - 核心思路：基于 CAT 能力框架，设计多个专业智能体——如情感分析智能体（识别情感基调）、认知模式智能体（检测扭曲的认知模式如灾难化思维）、社交信号智能体（识别社交退缩或求助信号）等。每个智能体配备专门的系统提示和角色描述。
    - 设计动机：单一 LLM 即使能力强大，也难以同时兼顾多个分析维度。专业化分工使每个方面都得到充分关注。

2. **协作讨论机制**:

    - 功能：整合多个智能体的分析结果，达成一致判断。
    - 核心思路：智能体之间进行多轮对话——先各自独立分析，然后分享各自的发现，针对分歧点进行讨论，最终汇总形成综合判断。这模拟了临床团队会诊的过程。
    - 设计动机：临床实践中，心理健康评估通常由多位专业人员从不同角度共同完成。多智能体讨论忠实地模拟了这一协作过程。

3. **RESTOREx 资源**:

    - 功能：提供带解释的抑郁症状标注数据集。
    - 核心思路：对表情包数据进行两层标注：（a）LLM 生成初始解释——为什么这个表情包表达了某种抑郁症状；（b）人工审核和修正这些解释，确保临床准确性。解释涉及视觉元素解读、文本含义分析和隐喻理解。
    - 设计动机：可解释的训练数据是开发可信赖的心理健康AI系统的基础设施。

### 损失函数 / 训练策略

MAMAMemeia 基于 LLM 的 in-context learning 或 fine-tuning，不需要传统的损失函数设计。框架主要通过精心设计的 prompt 和智能体交互协议来实现功能。

## 实验关键数据

### 主实验

| 方法 | Macro-F1 | 对比方法数 | 说明 |
|------|---------|-----------|------|
| MAMAMemeia | 最佳（+7.55%） | 30+ | 新SOTA |
| 最佳单模型基线 | 次优 | -- | 缺乏多维分析 |
| 简单LLM提示 | 中等 | -- | 无结构化分析 |

### 消融实验

| 配置 | Macro-F1 | 说明 |
|------|---------|------|
| 完整多智能体 | 最佳 | 所有CAT维度 |
| 单智能体 | 下降 | 缺乏多维视角 |
| 无讨论（仅投票）| 下降 | 缺乏信息整合 |
| 无CAT框架 | 下降 | 缺乏临床指导 |

### 关键发现

- 多智能体协作相比单智能体有显著提升——验证了多维分析的价值。
- CAT 能力框架的引入为 LLM 提供了有效的分析结构。
- 讨论机制比简单投票更有效——智能体之间的信息交换产生了协同效应。

## 亮点与洞察

- **临床心理学框架与 AI 的结合**是本文最大亮点——不是盲目使用 LLM，而是用专业的心理学方法论来指导智能体设计。
- **可解释性**对心理健康应用至关重要——MAMAMemeia 不仅给出分类结果，还提供详细的分析理由。
- RESTOREx 数据集对社区有长期价值。

## 局限与展望

- 表情包的文化特异性可能限制跨文化泛化。
- LLM 智能体的分析质量依赖于底层模型的能力。
- 未考虑用户发布表情包的上下文（如时间序列、社交网络）。
- 可以与临床标准（如 PHQ-9）对齐以增强临床实用性。

## 相关工作与启发

- **vs 单模型多模态分类**: 缺乏分析维度的分离和可解释性。
- **vs 情感分析方法**: 情感分析过于粗粒度，不区分具体的抑郁症状类型。

## 评分

- 新颖性: ⭐⭐⭐⭐ CAT框架+多智能体协作的组合方案新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 超越30+方法，RESTOREx数据集有长期价值
- 写作质量: ⭐⭐⭐⭐ 跨学科背景阐述清晰
- 价值: ⭐⭐⭐⭐ 对心理健康AI应用有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](../../ACL2026/multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](../../ACL2026/multi_agent/posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](../../ACL2025/multi_agent/multi-agent_collaboration_via_cross-team_orchestration.md)

</div>

<!-- RELATED:END -->
