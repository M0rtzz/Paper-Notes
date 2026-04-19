---
title: >-
  [论文解读] HeurekaBench: A Benchmarking Framework for AI Co-scientist
description: >-
  [ICLR 2026][模型压缩][AI co-scientist] 提出 HeurekaBench，一个基于真实科学工作流构建评测基准的框架，通过多LLM流水线从论文中提取可验证的科学洞见并生成开放式研究问题，用于评估AI co-scientist在数据驱动科学发现中的端到端能力。
tags:
  - ICLR 2026
  - 模型压缩
  - AI co-scientist
  - benchmark
  - scientific agents
  - single-cell biology
  - open-ended evaluation
---

# HeurekaBench: A Benchmarking Framework for AI Co-scientist

**会议**: ICLR 2026  
**arXiv**: [2601.01678](https://arxiv.org/abs/2601.01678)  
**代码**: [brbiclab.epfl.ch/projects/heurekabench](https://brbiclab.epfl.ch/projects/heurekabench)  
**领域**: 模型压缩  
**关键词**: AI co-scientist, benchmark, scientific agents, single-cell biology, open-ended evaluation

## 一句话总结
提出 HeurekaBench，一个基于真实科学工作流构建评测基准的框架，通过多LLM流水线从论文中提取可验证的科学洞见并生成开放式研究问题，用于评估AI co-scientist在数据驱动科学发现中的端到端能力。

## 研究背景与动机
LLM推理能力提升催生了大量科学Agent（如CellVoyager、Biomni等），它们旨在自主分析实验数据并产生科学洞见。然而，现有benchmark存在根本性不足：大多数仅测试静态知识检索或单步计算问题（如"有多少miRNA在p≤0.05后显著？"），这些"指令跟随式"任务与真正的co-scientist角色相去甚远——一个合格的co-scientist应能自主规划分析流程、探索数据集并生成新发现。BaisBench虽尝试生成研究问题但仅依赖单个LLM，导致问题质量不可靠。核心矛盾在于：现有benchmark无法评估开放式、数据驱动的科学发现能力。本文的切入角度是将benchmark构建本身植根于科学过程——从同行评审论文中提取经过验证的科学洞见，作为评测的ground truth。

## 方法详解

### 整体框架
HeurekaBench由三个阶段组成：(a) 洞见生成——从论文中提取候选洞见并半自动验证；(b) 问题生成——将验证后的洞见转化为QA对；(c) 问题求解——Agent自主设计多步分析并产生答案，由LLM Judge对照ground truth评分。

### 关键设计

1. **洞见生成流水线 (Insight Generation)**:

    - 功能：从科学论文及其代码仓库中提取可重现的科学洞见
    - 核心思路：设计4个模块化LLM组件——InsightExtractor从论文提取候选洞见（包含摘要、实验技术、原文证据三个结构化组件）；CodeDescriber将代码脚本转为自然语言描述；CodeMatcher将洞见与最相关的代码描述配对；CodeGenerator组合脚本生成多步验证工作流
    - 设计动机：通过代码可重现性验证来过滤不可靠洞见，比单纯依赖LLM生成的问题（如BaisBench）更可靠。使用GPT-4o做InsightExtractor、Claude-4-Sonnet做代码相关模块

2. **问题生成 (Question Generation)**:

    - 功能：将验证后的洞见转化为开放式研究问题(OEQ)和多选题(MCQ)
    - 核心思路：对每个洞见用few-shot prompting生成2个QA对。OEQ允许多种分析路径到达正确答案；MCQ包含高质量干扰项。生成后经两阶段过滤：(1) 自动过滤——剔除仅凭LLM预训练知识就能回答的简单问题；(2) 人工审核——去除幻觉、重复和基于未验证部分的问题
    - 设计动机：OEQ反映真实科研中的开放性，MCQ作为快速Agent原型验证的轻量代理

3. **评估方案 (G-Eval with Atomic Facts)**:

    - 功能：使用GPT-4o作为LLM Judge，对开放式回答进行1-5分评估
    - 核心思路：指导Judge将回答和ground truth均分解为原子事实（条件、趋势、结论），然后逐一比较完整性、部分匹配和缺失情况。只有所有ground truth事实均存在且无矛盾时才给满分，额外的非矛盾发现不扣分
    - 设计动机：避免表面匹配，奖励数据驱动的输出而非事实记忆

### 验证实验
在单细胞生物学领域实例化为sc-HeurekaBench：从22篇Nature/Cell论文中提取，最终41个验证洞见、13篇论文，产生50个OEQ和50个MCQ。InsightExtractor在FlyBase上44/50强相关，CodeMatcher平均74.6%文件正确匹配率。

## 实验关键数据

### 主实验
| Agent | OEQ正确性[1-5] | MCQ准确率(%) | MCQ召回率(%) | MCQ精度(%) |
|-------|---------------|-------------|-------------|-----------|
| BixBench-Agent | 2.34 | 44.44 | 80.56 | 62.96 |
| CellVoyager | 2.03 | 27.78 | 38.89 | 32.41 |
| Biomni | 2.31 | **50.00** | **88.24** | **76.96** |

### Planner消融（Biomni Agent）
| 模型 | 开源 | OEQ正确性 | MCQ准确率(%) |
|------|------|----------|-------------|
| MedGemma-27B | ✓ | 1.53 | 20.41 |
| Qwen3-32B | ✓ | 1.47 | 40.00 |
| Qwen3-235B-thinking | ✓ | 1.85 | 46.00 |
| GPT-OSS-120B | ✓ | 2.08 | 42.00 |
| Claude-4-Sonnet | ✗ | **2.58** | 44.00 |

### 关键发现
- Biomni和BixBench-Agent优于CellVoyager，表明灵活的Agent循环更能构建鲁棒的工作流
- Claude-4-Sonnet作为Planner显著优于其他模型（2.58 vs 2.08），闭源前沿模型在co-scientist任务上仍有明显优势
- End-critic（在Agent循环结束时加入critic）可显著提升开源LLM表现，低分组（得分1-2）的表现从1.32提升至1.91
- 模型参数规模和推理能力（thinking模式）对co-scientist表现至关重要

## 亮点与洞察
- 从"将benchmark植根于科学过程本身"的角度出发非常巧妙——用论文的可重现性作为洞见验证标准
- 多LLM流水线的模块化设计使框架可迁移到其他科学领域
- End-critic的设计能弥补开源与闭源模型的差距达22%，是一个轻量且有效的改进策略

## 局限与展望
- 目前仅在单细胞生物学领域实例化，框架泛化到化学、物理等需额外验证
- sc-HeurekaBench规模较小（50 OEQ + 50 MCQ），可能不足以进行细粒度能力诊断
- 验证过程仍需大量人工参与（运行代码、核对结果），自动化程度有提升空间

## 相关工作与启发
- **vs BaisBench**: BaisBench仅用单个LLM生成问题且无验证，HeurekaBench通过多LLM+代码验证确保可靠性
- **vs BixBench**: BixBench主要测试计算型问题，HeurekaBench测试开放式科学探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架思路新颖但偏benchmark/系统工作
- 实验充分度: ⭐⭐⭐⭐ 多维度消融非常详尽，但数据集规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图表精美
- 价值: ⭐⭐⭐⭐ 为AI for Science领域提供了重要的评测框架

<!-- RELATED:START -->

## 相关论文

- [A Mathematical Framework for AI-Human Integration in Work](../../ICML2025/model_compression/a_mathematical_framework_for_ai-human_integration_in_work.md)
- [Textual Equilibrium Propagation for Deep Compound AI Systems](textual_equilibrium_propagation_for_deep_compound_ai_systems.md)
- [Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](../../AAAI2026/model_compression/hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)

<!-- RELATED:END -->
