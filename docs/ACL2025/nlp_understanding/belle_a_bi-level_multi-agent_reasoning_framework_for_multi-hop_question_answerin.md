---
title: >-
  [论文解读] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering
description: >-
  [ACL2025][NLP理解][multi-hop QA] 提出 BELLE 双层多智能体辩论框架，先将多跳问题分类为四种类型，再通过双层辩论机制（第一层正反方辩论 + 第二层快/慢辩论者监督）动态规划 CoT、单步检索、迭代检索等算子的组合方案，实现面向问题类型的自适应多跳推理。
tags:
  - ACL2025
  - NLP理解
  - multi-hop QA
  - multi-agent debate
  - operator combination
  - question type classification
  - retrieval-augmented reasoning
---

# BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering

**会议**: ACL2025  
**arXiv**: [2505.11811](https://arxiv.org/abs/2505.11811)  
**代码**: 未公开  
**领域**: nlp_understanding  
**关键词**: multi-hop QA, multi-agent debate, operator combination, question type classification, retrieval-augmented reasoning

## 一句话总结

提出 BELLE 双层多智能体辩论框架，先将多跳问题分类为四种类型，再通过双层辩论机制（第一层正反方辩论 + 第二层快/慢辩论者监督）动态规划 CoT、单步检索、迭代检索等算子的组合方案，实现面向问题类型的自适应多跳推理。

## 研究背景与动机

多跳问答（Multi-hop QA）需要在多篇文档间进行逐步推理，是 LLM 面临的核心挑战之一。现有方法大致分两类：

- **闭卷推理**（CoT、子问题分解、概率推理树）：完全依赖 LLM 内部知识，面对知识密集型问题容易产生幻觉
- **检索增强推理**（单步检索、迭代检索、自适应检索）：引入外部知识但通常采用固定流程，不区分问题类型，对简单问题过度消耗计算资源，对复杂问题检索不足

作者的关键洞察是：**不同类型的多跳问题对不同推理方法的敏感度差异显著**。具体而言：
- **Inference 类型**（跨文档推理）：需要子问题分解 + 迭代检索的组合
- **Comparison/Temporal 类型**（比较/时间）：只需子问题分解 + 单步检索即可
- **Null 类型**（无固定模式）：需要自适应组合

这一观察促使作者设计了一个能根据问题类型动态选择算子组合的框架，而非为所有问题套用同一套固定流程。

## 方法详解

### 整体框架

BELLE 由三个模块构成：问题类型分类器 → 双层多智能体辩论 → 多跳QA执行器。将 CoT、单步检索、迭代检索、子问题分解、自适应检索五种方法视为"算子"，辩论系统的目标是为给定问题规划最优的算子组合方案。

### 模块1：问题类型分类器

利用 ICL（in-context learning）将多跳问题分为四类：
- **Inference**：需跨多个文档推理，逻辑步骤多
- **Comparison**：比较两个实体的某个属性，答案通常简短（Yes/No）
- **Temporal**：涉及时间戳的查找和比较
- **Null**：不属于上述类型，无固定模式

分类结果作为先验知识注入后续辩论系统的 meta prompt 中，引导算子选择方向。对于缺少类型标注的数据集（HotpotQA、2WikiQA、MuSiQue），使用 GPT-4 标注并经人工验证（95% 准确率）。

### 模块2：双层多智能体辩论系统

这是 BELLE 的核心创新。在传统多智能体辩论（MAD）的基础上，增加了第二层记忆智能体来监控辩论质量。

**第一层 — 正反方辩论**：
- 正方辩论者（affirmative）和反方辩论者（negative）轮流就算子选择方案展开对抗式辩论
- 每轮辩论基于各自的历史记录 $H_{ad}^{t-1}$ 以及上一轮快/慢辩论者的反馈
- 正方第 $t$ 轮输出：$f_{ad}^t = \mathcal{M}(H_{ad}^{t-1}, f_{fast}^{t-1}, f_{slow}^{t-1})$

**第二层 — 快/慢辩论者监督**：
- **快辩论者（Fast Debater）**：关注**当前轮**双方立场是否合理，评估当前算子选择方案，但容易被双方观点牵引  
  $f_{fast}^t = \mathcal{M}(f_{ad}^t, f_{nd}^t, H_{fast}^{t-1})$
- **慢辩论者（Slow Debater）**：整合**全部历史信息**进行综合判断，防止辩论者因相互影响丧失正确立场而产生观点振荡  
  $f_{slow}^t = \mathcal{M}(f_{ad}^t, f_{nd}^t, f_{fast}^t, H_{slow}^{t-1})$

快/慢辩论者的命名灵感来自 Talker-Reasoner 架构（System 1 vs System 2 思维），快辩论者关注即时状态，慢辩论者负责深层综合。

**裁判（Judge）**：监督整个辩论过程，有两种终止模式：
- **Hard Mode**：辩论中达成共识，直接输出算子组合方案
- **Soft Mode**：轮数用尽仍无共识，裁判从慢辩论者历史中提取最合理的算子建议

### 模块3：多跳QA执行器

根据辩论输出的执行计划，依次调用对应算子。例如：先用子问题分解拆分原始问题，再对每个子问题执行迭代检索获取外部知识，最后合并子答案得到最终回答。

## 实验关键数据

### 数据集与基线
- **数据集**：MultiHop-RAG、HotpotQA、2WikiMultiHopQA、MuSiQue（2/3/4跳）
- **基线**：13 种方法，涵盖闭卷推理（SP、CoT）、检索增强（Single-step、IRCoT、FLARE、ProbTree、BeamAggR）、Agent-based（LONGAGENT、GEAR、RopMura）
- **LLM**：GPT-3.5-turbo、Qwen2.5-7B

### Table 1: 主实验结果（F1 %）

| 方法 | MultiHop-RAG | HotpotQA | 2WikiQA | MuSiQue-2hop | MuSiQue-3hop | MuSiQue-4hop |
|------|------|------|------|------|------|------|
| CoT | 50.5 | 46.5 | 42.3 | 30.2 | 22.5 | 13.2 |
| IRCoT | 59.2 | 56.2 | 56.8 | 31.4 | 19.2 | 16.4 |
| ProbTree | 62.5 | 60.4 | 67.9 | 41.2 | 30.9 | 14.4 |
| BeamAggR | 67.2 | 62.9 | 71.6 | 45.9 | 36.8 | 21.6 |
| LONGAGENT | 56.8 | 59.3 | 65.6 | 40.5 | 25.8 | 16.4 |
| **BELLE** | **70.4** (+3.2) | **66.5** (+3.6) | **75.7** (+4.1) | **50.5** (+4.6) | **42.1** (+5.3) | **29.2** (+7.6) |

BELLE 在所有数据集上均取得最优 F1，且**跳数越多提升越大**（4-hop 提升 7.6%），说明动态算子组合对复杂推理尤为有效。

### Table 2: 消融实验（F1 %，GPT-3.5-turbo）

| 配置 | MultiHop-RAG | HotpotQA | 2WikiQA | MuSiQue | Avg. |
|------|------|------|------|------|------|
| BELLE (完整) | 70.4 | 66.5 | 75.7 | 40.6 | 63.3 |
| w/o 类型分类器 | 67.9 | 63.4 | 73.2 | 37.6 | 60.5 |
| w/o 第一层辩论 | 68.2 | 63.7 | 73.5 | 38.1 | 60.9 |
| w/o 第二层辩论 | 66.8 | 62.8 | 72.3 | 36.5 | 59.6 |
| w/o 快辩论者 | 67.3 | 63.2 | 72.9 | 37.4 | 60.2 |
| w/o 慢辩论者 | 67.0 | 63.1 | 72.7 | 36.9 | 59.9 |

**关键发现**：去除第二层辩论影响最大（-3.7% avg.），说明快/慢辩论者的历史信息整合机制是性能核心。慢辩论者比快辩论者更重要。

### Table 3: 计算开销对比（平均 prompt token 数）

| 方法 | Avg. Token |
|------|------|
| BELLE | 20,742 |
| LONGAGENT | 48,493 |
| GEAR | 41,931 |
| RopMura | 56,859 |

BELLE 的 token 消耗仅为其他 agent 方法的 **36%~57%**，因为辩论通常只需 2 轮即可确定算子方案。

## 关键发现

1. **组合算子优于单一算子**：在所有问题类型上，组合算子比单一算子平均提升 3%
2. **问题类型决定最优组合**：Inference 适合子问题+迭代检索，Comparison/Temporal 适合子问题+单步检索
3. **跳数越多优势越明显**：在 MuSiQue 4-hop 上提升达 7.6%，表明动态规划在高复杂度下价值更大
4. **辩论收敛快**：通常 2 轮辩论即可确定算子组合，token 开销远低于其他 agent 方法
5. **慢辩论者比快辩论者更关键**：慢辩论者通过整合全局历史防止观点振荡，对性能的贡献更大

## 亮点与洞察

- **算子组合的视角**：将现有多跳QA方法抽象为"算子"，从方法选择问题转化为规划问题，思路清晰有启发
- **双层辩论的 System 1/2 类比**：快辩论者类似直觉反应（关注当前状态），慢辩论者类似深思熟虑（综合历史信息），借鉴认知科学的双系统理论设计多智能体协作
- **成本与效果的双赢**：不仅性能最优，token 消耗反而最低，核心原因是对问题类型的精准规划避免了冗余的检索轮次
- **Qwen2.5-7B 上的有效性**：在 7B 开源模型上也取得显著提升（avg. 56.2 vs 48.3 BeamAggR），说明框架不依赖超大模型

## 局限性

- **辩论开销的优化空间**：虽然比其他 agent 方法 token 消耗低，但多智能体迭代辩论本身仍有开销，辩论策略和规则还有优化余地
- **问题类型的泛化性**：当前仅定义了 4 种类型，面对全新/未见过的问题格式可能表现不佳，类型体系的可扩展性需要验证
- **分类器的准确性依赖**：整个流程建立在问题类型分类正确的前提上，分类错误会导致错误的算子规划，但论文未详细分析分类错误的级联影响
- **算子池的固定性**：当前算子池包含 5 种方法，未探讨如何动态扩展新的算子或适配不同领域的检索需求
- **仅在英文数据集上评估**：四个数据集均为英文，多语言场景的适用性未知

## 相关工作与启发

- **BeamAggR**（ACL 2024）：多源知识聚合的束搜索推理，是最强基线（avg. F1 59.1），BELLE 在其基础上通过问题类型的先验知识进一步提升
- **Talker-Reasoner 架构**（Christakopoulou et al., 2024）：快/慢辩论者的灵感来源，将 System 1（快速反应）和 System 2（深度推理）的认知科学概念引入多智能体系统
- **MAD 框架发展**：从基础的正反方+裁判三角色，到本文的双层五角色，多智能体辩论系统在结构设计上的演进方向
- **启发**：将方法选择本身建模为可辩论的决策问题，这一元推理思路可推广到其他需要动态选择策略的 NLP 任务（如摘要、翻译中的方法选择）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题类型驱动的算子动态组合视角新颖，双层辩论的快/慢设计有明确的认知科学动机
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集、13 基线、详细消融、计算开销分析、算子选择变化可视化，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 动机分析清晰，方法描述严谨，但符号较多、部分公式可简化
- 价值: ⭐⭐⭐⭐ — 在多跳QA任务上取得一致的 SOTA，且 token 开销更低，对实际应用有指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)

<!-- RELATED:END -->
