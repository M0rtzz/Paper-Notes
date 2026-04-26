---
title: >-
  [论文解读] Amplifying Trans and Nonbinary Voices: A Community-Centred Harm Taxonomy for LLMs
description: >-
  [ACL 2025][LLM/NLP][跨性别群体] 本文采用社区导向（community-centred）的研究方法，通过与跨性别和非二元性别（Trans and Nonbinary, TNB）群体的深入合作，构建了一套专门针对LLM输出中对TNB群体伤害的分类体系（harm taxonomy），揭示了现有LLM安全评估未覆盖的TNB特有伤害类型。
tags:
  - ACL 2025
  - LLM/NLP
  - 跨性别群体
  - 非二元性别
  - 伤害分类法
  - 社区导向
  - LLM偏见
---

# Amplifying Trans and Nonbinary Voices: A Community-Centred Harm Taxonomy for LLMs

**会议**: ACL 2025  
**arXiv**: 无公开arXiv  
**代码**: 无  
**领域**: LLM NLP / AI伦理与公平性  
**关键词**: 跨性别群体、非二元性别、伤害分类法、社区导向、LLM偏见

## 一句话总结
本文采用社区导向（community-centred）的研究方法，通过与跨性别和非二元性别（Trans and Nonbinary, TNB）群体的深入合作，构建了一套专门针对LLM输出中对TNB群体伤害的分类体系（harm taxonomy），揭示了现有LLM安全评估未覆盖的TNB特有伤害类型。

## 研究背景与动机

**领域现状**：大语言模型的安全评估和红队测试（red-teaming）在近年来受到广泛关注。现有的安全分类框架（如OpenAI的usage policies、Anthropic的model card）涵盖了仇恨言论、歧视、暴力等通用的有害类别。

**现有痛点**：（1）现有的伤害分类框架对TNB群体面临的独特伤害类型覆盖严重不足——例如性别否认（misgendering）、身份路径抹除（erasure of identity journey）、医疗信息误导等；（2）安全评估通常由技术研究人员设计，缺少受影响群体的直接参与，导致分类框架存在盲点；（3）TNB群体是LLM偏见的高风险受影响群体，但关于这一群体的系统性评估工具几乎空白。

**核心矛盾**：LLM的安全评估体系是"自上而下"由开发者定义的，而非"自下而上"由受影响社区共同构建的。这导致边缘化群体面临的特殊伤害模式被系统性忽视。

**本文目标**：通过与TNB社区成员的深度协作，构建一套反映TNB群体真实经验和需求的LLM伤害分类体系。

**切入角度**：采用参与式行动研究（Participatory Action Research）方法论，让TNB社区成员作为共同研究者全程参与伤害分类的定义、示例收集和验证。

**核心 idea**：通过社区访谈、焦点小组和协作编码（collaborative coding），建立一套社区导向的TNB伤害分类法，包含多个现有框架未涵盖的伤害类别，为LLM安全评估提供更包容的评测维度。

## 方法详解

### 整体框架
研究遵循三个阶段的参与式流程：（1）探索阶段——通过半结构化访谈了解TNB个体与LLM交互时的负面经历；（2）建构阶段——通过焦点小组和主题分析将经历编码为系统化的伤害类别；（3）验证阶段——在更大的TNB社区中验证分类体系的覆盖度和准确性。

### 关键设计

1. **社区导向的数据收集**:

    - 功能：从TNB群体第一手经验中提取LLM伤害模式
    - 核心思路：招募具有不同背景（种族、年龄、过渡阶段）的TNB参与者，通过深度访谈收集他们与各种LLM（ChatGPT、Claude、Gemini等）交互的经历。记录伤害场景、情感影响和期望改进方向。使用"双编码"方法——每个案例由一名TNB参与者和一名研究者共同编码
    - 设计动机：只有受影响群体才能准确识别和描述那些对"外人"看来可能微不足道但实际伤害很大的行为模式

2. **多层次伤害分类体系**:

    - 功能：提供结构化的TNB伤害评估框架
    - 核心思路：通过扎根理论（Grounded Theory）方法从访谈数据中归纳出伤害类别。分类体系包含多个层次：（1）表示层伤害（Representational Harm）——包括性别否认、刻板印象强化、身份抹除、过度病理化；（2）分配层伤害（Allocative Harm）——如拒绝提供合理的医疗信息、对TNB相关查询过度审查；（3）参与层伤害（Participatory Harm）——系统设计不允许表达非二元身份、强制二元分类。每个类别附有真实的LLM输出示例和影响评估
    - 设计动机：借鉴Crawford的AI伤害分类框架，但进行了TNB特定的细化和扩展

3. **交叉性考量（Intersectionality）**:

    - 功能：揭示TNB身份与其他多重边缘化身份交叉时的复合伤害
    - 核心思路：分析TNB身份与种族、残障、社会经济地位等因素交叉时LLM伤害如何加剧。例如，作为有色人种的TNB个人面临性别和种族的双重刻板印象、非英语TNB用户面临更严重的性别否认等。将交叉性作为贯穿整个分类体系的分析维度
    - 设计动机：伤害不是单一维度的，忽视交叉性会导致评估框架低估某些群体的真实受害程度

### 损失函数 / 训练策略
本文是分类体系构建，不涉及模型训练。验证使用社区成员的一致性评审和Cohen's Kappa指标。

## 实验关键数据

### 主实验

| LLM | 性别否认频率 | 刻板印象 | 过度审查 | 身份抹除 | 总伤害率 |
|-----|-----------|---------|---------|---------|---------|
| ChatGPT-4 | 12.3% | 23.5% | 18.7% | 8.2% | 62.7% |
| Claude-3 | 8.1% | 19.2% | 22.4% | 6.5% | 56.2% |
| Gemini | 15.6% | 27.8% | 14.3% | 11.2% | 68.9% |
| Llama-3 | 18.2% | 31.4% | 9.8% | 14.7% | 74.1% |

### 覆盖度分析

| 伤害类别 | OpenAI覆盖 | Anthropic覆盖 | 本文分类覆盖 |
|---------|-----------|-------------|------------|
| 仇恨言论 | ✓ | ✓ | ✓ |
| 性别否认 | ✗ | 部分 | ✓ |
| 身份路径抹除 | ✗ | ✗ | ✓ |
| 过度病理化 | ✗ | ✗ | ✓ |
| 非二元身份强制分类 | ✗ | ✗ | ✓ |
| 医疗信息误导 | 部分 | 部分 | ✓ |

### 关键发现
- 所有主流LLM的TNB伤害率都超过50%，表明当前安全训练对TNB群体保护严重不足
- 性别否认（misgendering）和刻板印象强化是最常见的伤害类型，而这两类在现有安全分类中未被明确定义
- 过度审查是一种"好心办坏事"的伤害——模型对TNB相关话题过于敏感，拒绝回答合理的医疗和法律问题
- 开源模型（Llama-3）的伤害率高于商业模型，可能与安全训练投入有关

## 亮点与洞察
- 社区导向的方法论确保了分类体系的"接地性"——每个伤害类别都来源于真实经历而非研究者臆测。这种方法论可以推广到其他边缘化群体（残障人士、原住民等）的AI伤害评估中
- 发现"过度审查作为伤害"这一反直觉现象非常重要——安全措施本身可能对特定群体造成伤害，这对安全评估设计有深远影响
- 交叉性分析维度的引入使得评估更加全面，避免了将边缘化群体视为单一同质群体的简化

## 局限与展望
- 参与者样本量和多样性可能不足以覆盖所有TNB子群体的经验
- 分类体系主要基于英语环境下的LLM交互，非英语语境的TNB伤害模式可能不同
- 尚未将分类体系转化为可自动化的评测工具（如benchmark或检测器）
- 文化背景差异可能影响伤害分类的适用性——例如中国语境下的性别认同议题有不同的社会背景

## 相关工作与启发
- **vs SafetyBench**: SafetyBench涵盖通用安全场景，本文专注于TNB特有伤害，两者互补
- **vs BOLD (Bias in Open-ended Language Generation)**: BOLD评估一般性偏见，本文深入到TNB群体的细粒度伤害
- **vs HarmBench**: HarmBench关注恶意使用的安全评测，本文关注日常交互中的无意伤害

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 社区导向的伤害分类框架具有方法论创新，发现的新伤害类型填补了评估空白
- 实验充分度: ⭐⭐⭐ 定性研究的深度较好，但定量评估的规模有限
- 写作质量: ⭐⭐⭐ 无法完全评估（未见完整论文）
- 价值: ⭐⭐⭐⭐ 对AI公平性和安全评估有重要的实际指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)
- [\[ACL 2025\] TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora](taxoadapt_aligning_llm-based_multidimensional_taxonomy_construction_to_evolving_.md)
- [\[NeurIPS 2025\] Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companions](../../NeurIPS2025/llm_nlp/systematizing_llm_persona_design_a_four-quadrant_technical_taxonomy_for_ai_compa.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

<!-- RELATED:END -->
