---
title: >-
  [论文解读] Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis
description: >-
  [ACL 2025][多智能体辩论] 提出Tree-of-Debate (ToD)框架，将科学论文转化为LLM persona进行树结构化辩论，通过自我审议、迭代检索和主持人引导的层级子话题扩展，生成细粒度、上下文化的论文对比摘要，在领域专家评估中显著优于基线方法。
tags:
  - ACL 2025
  - 多智能体辩论
  - 论文对比分析
  - 树结构推理
  - 检索增强生成
  - 科学文献摘要
---

# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis

**会议**: ACL 2025  
**arXiv**: [2502.14767](https://arxiv.org/abs/2502.14767)  
**作者**: Priyanka Kargupta, Ishika Agarwal, Tal August, Jiawei Han (UIUC)
**代码**: [GitHub](https://github.com/pkargupta/tree-of-debate)  
**领域**: others  
**关键词**: 多智能体辩论, 论文对比分析, 树结构推理, 检索增强生成, 科学文献摘要

## 一句话总结

提出Tree-of-Debate (ToD)框架，将科学论文转化为LLM persona进行树结构化辩论，通过自我审议、迭代检索和主持人引导的层级子话题扩展，生成细粒度、上下文化的论文对比摘要，在领域专家评估中显著优于基线方法。

## 研究背景与动机

### 问题背景
随着科学文献的爆炸式增长（arXiv 2024年10月单月提交超24,000篇），研究者面临巨大挑战：难以识别相关论文间的新颖发现、增量贡献和等价思想，特别是来自不同研究社区的相似工作。自动生成论文对比摘要对文献综述具有重要价值。

### 已有工作的不足
- **现有对比摘要方法**（如Two-Stage pipeline）通常采用"先提取再对比"的两步策略，但仅关注表面语义差异，无法捕捉最相关的深层区别
- **非结构化LLM生成**倾向于产生提取式摘要，将语义相似/不相似的短语直接列为相似/差异，缺少"为什么"的上下文解释
- **长文本处理困难**：直接提供整篇论文导致细节被淹没，仅用标题和摘要又只能产生高层表面对比
- **缺乏非互引论文对比数据集**：现有工作忽略了不互相引用的论文对比场景，而这在文献爆炸时代极为常见

### 核心动机
设计一个结构化的多persona辩论框架，利用树结构分解论文贡献为独立子话题，通过辩论式批判推理揭示细粒度的相似性和差异性，同时通过迭代检索确保事实准确性。

## 方法详解

### 整体框架
ToD由三个核心角色组成：两个**论文persona**（分别代表待比较的两篇论文）和一个**主持人**（Moderator）。给定两篇论文$p_1, p_2$和一个根话题$n_0$，框架动态构建一棵辩论树$T$，每个节点代表一个具体贡献话题的辩论轮次。

### 段落级检索（Segment-level Retrieval）
- 使用检索嵌入模型将每篇论文切分为约三句话的段落
- 基于余弦相似度按话题$n_i$检索最相关的$\delta$个段落
- 查询格式："[话题名称] : [话题描述]"

### Persona构建
- **论文Persona**：被赋予论文的标题、摘要和与当前话题相关的检索段落，角色是论证自身贡献优于对手
- **主持人**：使用相同底层模型，负责(i)识别关键辩论子话题，(ii)判断辩论进展，(iii)将辩论树综合为对比摘要

### 树节点辩论流程（三阶段）
**阶段1：自我审议（Self-Deliberation）**
1. 每个persona检索与当前话题$n_i$相关的段落$S_i^a$
2. 生成$k$个新颖性声明$C_i^a$，并映射对应证据$E_i^a \subseteq S_i^a$
3. **预判反驳（Preemption）**：将对手的声明$C_i^b$暴露给persona $p_a$，后者从自己论文中检索反驳证据$\widetilde{E}_i^a$，通过LLM过滤步骤判断证据是否支持/反驳/澄清对手声明
4. 主持人基于双方声明和证据生成$k$个子话题

**阶段2：辩论（Debate）**
对每个子话题节点$n_j^i$，每个persona依次：
1. **陈述（Present）**：论证自身在该子话题上优于对手
2. **回应（Respond）**：回应对手论点，提出质疑或澄清问题
3. **修正（Revise）**：基于交互修正自身论点

**阶段3：扩展判定（Determine Expansion）**
主持人评估三个条件：
- 论点是否有足够进展或引入新概念？
- 是否存在未回答的有价值问题？
- 是否已有明确"赢家"无需进一步分解？
满足前两者之一或第三者不成立，则触发新一轮自我审议并向下扩展树。

### 辩论综合
当所有辩论路径充分扩展后，主持人将整棵辩论树综合为段落式对比摘要，先阐述相似性再详述差异性。

## 实验关键数据

### 实验设置
- 基座模型：Llama-3.1-Nemotron-70B-Instruct-HF（开源）
- 数据集：100对论文，涵盖NLP、数据挖掘、电气工程、航空航天四个领域，由5位领域专家构建和评估
- 论文对分类：互引(30对) vs 非互引(70对)；方法差异(45对) vs 任务差异(55对)
- 评估指标：事实性（Factuality, 0/1二值句级）、广度（Breadth, 0-4）、上下文化（Contextualization, 0-4）

### 主实验结果（Table 2，整体性能）

| 方法 | Breadth↑ | Context↑ | Factuality↑ |
|------|----------|----------|-------------|
| Single Stage | 89.04 | 72.80 | 93.59 |
| Two Stage | 86.77 | 75.46 | 94.12 |
| ToD (No Tree) | 80.74 | 70.04 | 89.78 |
| ToD (No SD) | 85.99 | 91.27 | 79.99 |
| **Tree-of-Debate** | **94.92** | **95.28** | **93.87** |

ToD在广度上比最优基线高约6.85%，在上下文化上高约25.98%，同时保持高事实性。

### 领域级性能（Table 5）

| 领域 | ToD Breadth | ToD Context | ToD Fact |
|------|------------|------------|----------|
| NLP | 95.08 | 95.83 | 94.78 |
| 数据挖掘 | 91.67 | **100.00** | 90.99 |
| 电气工程 | 98.08 | 88.46 | 91.36 |
| 航空航天 | **100.00** | 90.00 | 88.64 |

ToD在所有领域均展现强劲性能，在NLP和数据挖掘领域增益最大；电气和航空领域事实性略有下降，推测与模型预训练知识中这些领域信息相对较少有关。

### 消融实验关键发现
- **去除树结构（No Tree）**：上下文化和广度显著下降，说明结构化辩论对细粒度分析至关重要
- **去除自我审议（No SD）**：上下文化仍保持较高（91.59%），但事实性和广度大幅下降，证明迭代检索是减少幻觉、保证广度覆盖的关键
- **辩论深度影响**：深度从1增至3时，事实性和上下文化均显著提升；深度1易产生表面化、过度正面的陈述

## 亮点

- **树结构辩论的创新设计**：将论文贡献分解为层级子话题独立辩论，避免了非结构化辩论中多个论点纠缠导致的推理困难，上下文化评分达95.28%
- **自我审议+预判反驳机制**：persona在辩论前不仅准备自身论点，还预判对手声明并检索反驳证据，模拟了真实学术辩论中的充分准备过程
- **迭代检索保证深度与准确性**：随辩论进展动态更新证据池，确保每个子话题的讨论有充分的论文细节支撑，有效抑制了幻觉
- **鲁棒性强**：在互引/非互引、方法差异/任务差异四种设定下性能标准差仅2.49，远低于基线方法
- **专家构建数据集**：填补了非互引论文对比分析数据集的空白，100对论文涵盖4个跨学科领域

## 局限与展望

- **辩论质量依赖基座模型**：在模型预训练知识较薄弱的领域（电气/航空），事实性有所下降，辩论中"回应"阶段的质量直接受限于模型能力
- **深层辩论可能引入噪声**：当论文中缺乏支撑细粒度声明的证据时，persona会"发明"潜在的未来研究或新方法，引入轻微噪声
- **事实性与上下文化的权衡**：在需要更大推理"跳跃"的任务差异对比中，ToD为识别论文间联系可能混淆两篇论文的事实
- **计算成本高**：树结构多轮辩论需要大量LLM推理调用，扩展到大规模文献比较时成本显著
- **仅限两篇论文对比**：当前框架固定为双方辩论，未扩展至多篇论文的群体对比场景
- **数据集规模有限**：100对论文、5位评估者，规模较小，跨领域泛化性仍需更大规模验证

## 与相关工作的对比

- **Single Stage / Two Stage基线**：直接用LLM生成对比摘要，倾向于产生提取式、缺乏上下文解释的摘要；ToD在上下文化上领先约20-25个百分点
- **Liang et al. (2024) 多智能体辩论**：用正反方辩论+裁判来减少确认偏差、提升推理质量，但辩论作为手段而非目标；ToD将辩论本身作为输出，树结构使贡献分解更精细
- **Portenoy et al. (2022) 作者Persona**：为作者推荐创建persona，基于命名实体匹配；ToD创建的是论文persona，主动辩论和捍卫论文贡献
- **ContributionSum (Liu et al. 2023) / DIR (Wang et al. 2024a)**：前者生成解耦贡献摘要但需要微调，后者需要基于gold standard的结构化微调；ToD纯推理时工作，无需训练，领域无关
- **Graph-based方法 (Chen et al. 2022; Stroehle et al. 2023)**：分类句子为声明/相似/差异并打分生成提取式摘要；ToD生成抽象式、上下文化的摘要

## 评分

- 新颖性: ⭐⭐⭐⭐ — 树结构辩论框架设计新颖，自我审议+预判反驳+迭代检索的组合很有创意
- 实验充分度: ⭐⭐⭐⭐ — 领域专家评估、多维指标、消融实验和案例分析充分，但数据集规模较小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示和案例展示有效，方法描述详尽
- 价值: ⭐⭐⭐⭐ — 对科学文献对比分析提供实用工具，框架思想可扩展至其他复杂推理任务

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)
- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ICLR 2026\] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](../../ICLR2026/others/completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)

<!-- RELATED:END -->
