---
title: >-
  [论文解读] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification
description: >-
  [ACL 2026][多智能体][表格问答] 本文提出 ODUTQA-MDC 任务和基准，首次系统研究开放域场景下用户查询模糊性的检测与多轮对话澄清问题，构建了包含 25,105 个 QA 对的大规模数据集，并设计了 MAIC-TQA 多智能体框架来完成"检测-澄清-推理"的端到端表格问答。
tags:
  - "ACL 2026"
  - "多智能体"
  - "表格问答"
  - "模糊查询澄清"
  - "多轮对话"
  - "多智能体框架"
  - "Text-to-SQL"
---

# ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification

**会议**: ACL 2026  
**arXiv**: [2604.10159](https://arxiv.org/abs/2604.10159)  
**代码**: [GitHub](https://github.com/jensenw1/ODUTQA-MDC)  
**领域**: LLM评测  
**关键词**: 表格问答, 模糊查询澄清, 多轮对话, 多智能体框架, Text-to-SQL

## 一句话总结
本文提出 ODUTQA-MDC 任务和基准，首次系统研究开放域场景下用户查询模糊性的检测与多轮对话澄清问题，构建了包含 25,105 个 QA 对的大规模数据集，并设计了 MAIC-TQA 多智能体框架来完成"检测-澄清-推理"的端到端表格问答。

## 研究背景与动机

**领域现状**：大语言模型推动了表格问答（Tabular QA）的发展，现有 Text-to-SQL 方法在标准数据集（如 Spider）上表现出色。开放域表格问答需要从大规模数据库中自主检索相关表格，进一步增加了难度。

**现有痛点**：现实场景中用户查询经常是模糊的（underspecified）——存在拼写错误、表述不清或信息不完整。例如用户可能省略城市名（FROM 子句缺失）、用模糊表达替代精确列名（SELECT 意图不明）、或使用简称代替全称（WHERE 条件不匹配）。这些模糊性根本性地阻碍了正确 SQL 的生成。

**核心矛盾**：现有研究要么仅在封闭域下检测模糊性（不解决问题），要么使用静态预设对话（PRACTIQ），无法捕捉真实用户交互的动态和不可预测性。缺乏适当的数据集和评估框架来系统研究"检测-澄清-问答"完整流程。

**本文目标**：定义 ODUTQA-MDC 任务，构建首个综合基准，包括大规模数据集、细粒度标注方案和动态澄清接口，并提出基线系统。

**切入角度**：将模糊性按 SQL 结构分类——表范围模糊（FROM）、查询意图模糊（SELECT）、查询条件模糊（WHERE），以及混合类型。这种分类自然对应 Text-to-SQL 流程的不同阶段。

**核心 idea**：构建"检测-澄清-重检测"闭环评估流程，通过动态用户模拟器实现可扩展的多轮交互评估，同时提出 MAIC-TQA 多智能体框架作为基线。

## 方法详解

### 整体框架
MAIC-TQA 采用模块化多智能体架构，流程为：SLU 模块提取用户意图和槽位信息 → 范围验证智能体（SV Agent）验证并澄清表范围信息 → 表检索智能体（TR Agent）整合原始查询和澄清信息确定目标表 → SQL 生成与验证智能体（SGV Agent）生成、执行和验证 SQL 查询。各智能体在流程中可动态触发与用户模拟器的澄清对话。

### 关键设计

**1. 细粒度模糊性分类与标注体系：把模糊性按 SQL 子句拆开，让"检测到什么"直接告诉系统"该澄清什么"**

现有数据集往往只标一种模糊性，且把它当成一个笼统的二值标签，系统即使知道"查询模糊"也不知道模糊在哪、该问什么。本文把模糊性和 SQL 结构对齐成三类：意图模糊对应 SELECT，用二值分类标注；范围模糊对应 FROM，用三元组 $[\text{slot\_content}, \text{slot\_type}, \text{error\_type}]$ 标注，其中 $\text{error\_type} \in \{\text{Missing}, \text{Error}, \text{Unmatch}\}$ 区分了"漏掉城市名""城市名写错""简称对不上全称"三种情况；条件模糊对应 WHERE，用三元组 $[\text{slot\_content}, \text{slot\_type}, \text{"not exist"}]$ 标注。多类共存时即构成 Mixed 类型。

这种"标签即定位"的设计让检测结果可以直接驱动澄清：系统一旦判定 FROM 槽位是 Missing，就知道该问用户补哪个城市，而不是泛泛地说一句"你的问题不清楚"。

**2. 动态澄清用户模拟器：用一个门控于检测正确性的自动用户，替代昂贵又不可复现的真人交互**

多轮澄清评估的难点其实在"用户"这一侧——真人交互成本高、回复不一致、无法复现。本文把用户实现成一个可调用的 Python 接口，并严格门控于检测准确性：只有当系统正确识别出模糊性类型时，它才返回对应的澄清信息，检测错了就什么提示都拿不到。为了让回复像真人，模拟器用 LLM 把标准模板改写成口语化表达，同时校验城市名、列名等关键信息没有在改写中丢失；它还提供动态模式（多样化回复）和固定模式（标准化回复，用于复现实验）。

门控是这个设计的关键所在：它堵死了"系统蒙对澄清"的泄漏路径，让多轮分数真实反映系统的检测能力，而不是模拟器有多慷慨。

**3. 多智能体协作框架（MAIC-TQA）：四个各管一段 SQL 子句的智能体，把"检测-澄清-推理"串成端到端闭环**

模糊性分散在 SQL 的不同子句上，让单个模型一次性又检测又澄清又生成，负担过重也难以定位错误。MAIC-TQA 把任务拆给四个专注的智能体：SLU 模块用 BERT 分类器联合做意图检测和槽位填充；SV Agent（范围验证）检查必需槽位是否缺失或无效，并调用数据库验证函数确认；TR Agent（表检索）整合对话历史生成表摘要，再用精确匹配或 BM25 检索目标表；SGV Agent（SQL 生成与验证）用 5-shot ICL 生成 SQL，执行后检查结果有效性，必要时触发条件澄清。每个智能体都能在自己负责的环节动态拉起与用户模拟器的对话。

这种分工让每个模块只处理一类模糊性、对应一段 SQL 流程，检测—澄清—重检测的闭环能在对应阶段就地完成，而不是把所有不确定性都堆到最后的 SQL 生成上。

### 一个完整示例：一句模糊查询如何被澄清成可执行 SQL

假设用户问"列出价格最高的几套房"。SLU 模块先做意图检测和槽位填充，发现 SELECT 意图大致清楚（要房产、按价格排序），但 FROM 范围缺了关键槽位——没有城市名。SV Agent 据此判定 $\text{error\_type}=\text{Missing}$，触发一轮澄清；因为系统检测正确，用户模拟器返回"我说的是杭州"。TR Agent 把这条澄清并入对话历史、生成更新后的表摘要，用 BM25 在数据库里检索到杭州的房产表。SGV Agent 用 5-shot ICL 生成 SQL 并执行，若发现 WHERE 里"价格最高"没有对应的可用列或返回为空，会再触发一次条件澄清，确认是按"总价"还是"单价"排序，最终产出可执行查询。整个过程里只要某一步检测错了模糊性类型，模拟器就不给提示、对应澄清也就拿不到，分数自然落回真实的检测能力。

### 损失函数 / 训练策略
SLU 模块使用 BERT 进行意图分类和槽位填充的联合训练。其他智能体使用 LLM 的 in-context learning，不需要额外训练。支持多种 LLM 后端（Qwen3 32B/30B、Kimi K2、GLM 4 等）。

## 实验关键数据

### 主实验（模糊性检测）

| 模型 | FROM Acc. | FROM F1 | WHERE Acc. | WHERE F1 | Mixed Acc. |
|------|-----------|---------|------------|----------|------------|
| Qwen3 32B | 77.66 | 82.82 | 69.59 | 66.02 | 54.96 |
| Qwen3 30B | 75.17 | 85.10 | 75.67 | 78.99 | 58.55 |
| Kimi K2 | 82.60 | 87.95 | 69.02 | 65.54 | 55.51 |
| SELECT (BERT) | 99.78 Acc. | 99.22 F1 | - | - | - |

### 消融实验（MAIC-TQA vs SLUTQA 基线）

| 配置 | 说明 |
|------|------|
| SLUTQA (无澄清) | 直接从模糊查询回答，作为无澄清基线 |
| MAIC-TQA Fixed | 使用标准化澄清回复 |
| MAIC-TQA Dynamic | 使用 LLM 改写的多样化澄清回复 |

### 关键发现
- SELECT 模糊性最容易检测（BERT 达 99%+ F1），FROM 和 WHERE 较难，Mixed 类型最难（~55% 准确率）
- 多轮对话澄清显著提升了问答准确率，验证了动态澄清机制的价值
- 动态模式下的性能略低于固定模式，反映了自然语言变异带来的挑战
- 所有模型在 Mixed 类型上表现较差，说明同时处理多种模糊性仍是开放问题

## 亮点与洞察
- 任务定义非常完整：从数据集构建、标注方案到评估框架（含动态用户模拟器），形成了可复现的闭环研究范式
- 模糊性按 SQL 子句分类的设计直觉且实用，使检测结果可直接指导后续 SQL 生成
- 动态澄清模拟器的门控机制设计巧妙——系统只有正确检测到模糊性才能获得澄清信息，避免了"泄漏"问题

## 局限与展望
- 数据集覆盖领域有限（房产、土地拍卖、金融），泛化到其他领域需要验证
- 模板化数据生成可能导致查询分布与真实用户查询存在差异
- 澄清轮次限制为单轮，对复杂模糊性可能不够
- Mixed 类型性能较低，需要更好的多模糊性联合处理方法
- 未来方向：扩展到更多领域和语言、允许多轮迭代澄清、引入用户满意度评估

## 相关工作与启发
- **vs PRACTIQ**: PRACTIQ 使用静态预设对话，不支持动态交互评估。本文的动态模拟器更贴近真实场景
- **vs AmbiQT/Ambrosia**: 这些工作引入了模糊性但缺少系统化的澄清机制和 QA 评估

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统定义了开放域模糊表格问答+多轮澄清的完整任务
- 实验充分度: ⭐⭐⭐⭐ 数据集规模大、标注细粒度、多模型对比
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 填补了该方向数据集和评估框架的空白，对社区有基础设施价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/multi_agent/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](../../AAAI2026/multi_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](../../ICML2026/multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)

</div>

<!-- RELATED:END -->
