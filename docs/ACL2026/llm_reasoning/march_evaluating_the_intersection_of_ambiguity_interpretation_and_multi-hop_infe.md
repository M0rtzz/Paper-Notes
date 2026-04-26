---
title: >-
  [论文解读] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference
description: >-
  [ACL 2026][LLM推理][多跳推理] 提出 MARCH 基准（2,209 个多跳歧义问题）和 CLARION 框架，首次系统研究歧义解析与多步推理交叉场景下的 QA 挑战，揭示现有 SOTA 模型在此类问题上的严重不足。
tags:
  - ACL 2026
  - LLM推理
  - 多跳推理
  - 歧义解析
  - 基准构建
  - 分层不确定性
  - 智能体框架
---

# MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference

**会议**: ACL 2026  
**arXiv**: [2509.22750](https://arxiv.org/abs/2509.22750)  
**代码**: [GitHub](https://github.com/jeonghyunpark2002/MARCH)  
**领域**: LLM Reasoning / Question Answering  
**关键词**: 多跳推理, 歧义解析, 基准构建, 分层不确定性, 智能体框架

## 一句话总结

提出 MARCH 基准（2,209 个多跳歧义问题）和 CLARION 框架，首次系统研究歧义解析与多步推理交叉场景下的 QA 挑战，揭示现有 SOTA 模型在此类问题上的严重不足。

## 研究背景与动机

**领域现状**：多跳 QA 要求模型跨多个文档构建逻辑链；歧义 QA 要求模型处理多义词和不充分上下文。这两类挑战已分别被广泛研究，但它们的**交叉**场景几乎未被探索。

**现有痛点**：在真实用户查询中，48.4% 包含歧义，17.7% 涉及多跳推理，13.3% 同时涉及两者。然而现有基准要么只关注单跳歧义(ASQA)，要么只关注多跳推理(MuSiQue)。当歧义发生在多跳推理的中间步骤时，不确定性会级联放大——早期的歧义解析错误会锁定错误的推理路径。

**核心矛盾**：多跳推理中的歧义可能是**潜在的**(latent)——只有在前面步骤被正确解析后才会显现。例如 "What is the best-selling pickup sold by the manufacturer of the 'Mustang'?" 中，pickup 的歧义（卡车 vs 吉他拾音器）只有在保留 Mustang 的两种解释（汽车 vs 吉他）后才能被发现。

**本文目标**：(1) 构建评估多跳歧义 QA 的专用基准；(2) 提出解决此问题的框架。

**核心idea**：多跳歧义 QA 需要模型在整个推理链中保持多条解释路径的"叠加态"，而非过早提交到单一解释。CLARION 通过将歧义规划与证据检索解耦，防止推理路径的过早剪枝。

## 方法详解

### 整体框架

本文包含两个贡献：(1) MARCH 基准——从 MuSiQue 出发，通过四阶段管线构建 2,209 个多跳歧义问题，覆盖语义/句法/约束三类歧义；(2) CLARION 框架——一个两阶段 Agent 框架，先由 Planning Agent 规划所有可能的解释路径，再由 Reasoning Agent 逐路径检索证据和推理。

### 关键设计

1. **MARCH 基准构建管线**:

    - 功能：提供高质量的多跳歧义 QA 评估基准
    - 核心思路：四阶段构建——(a) 使用4个 LLM（GPT-4.1、Llama-4、Qwen3-235B、Claude-4）全票一致检测歧义类型；(b) 分解澄清问题为原子子问题并从 Wikipedia 检索证据；(c) 为每种解释生成短答案和综合长答案；(d) 使用3个独立 LLM 一致性过滤。最终保留 2,209 个样本
    - 设计动机：多 LLM 全票一致检测减少单模型偏差，人工验证(Fleiss' κ=0.92-0.95)确保标签质量

2. **多跳歧义分类体系**:

    - 功能：为多跳歧义提供系统化的分类和处理指南
    - 核心思路：扩展标准歧义分类到多跳场景——(a) **语义歧义**：同一mention映射到多个实体（如 Mustang → Ford 汽车/Fender 吉他），需要"解释"(Interpret)；(b) **句法歧义**：多种合法解析导致不同的跳间依赖（如介词附着歧义），需要"解析"(Resolve)；(c) **约束歧义**：过度特定的限定导致合法推理路径被剪枝，需要"泛化"(Generalize)
    - 设计动机：不同类型的歧义需要不同的处理策略，分类体系为方法设计提供指导

3. **CLARION 框架(CLarifying Ambiguity with Reasoning and InstructiON)**:

    - 功能：通过解耦歧义规划和证据推理来处理多跳歧义 QA
    - 核心思路：两阶段——(a) **Planning Agent**：接收原始问题，识别所有歧义点，生成多条解释路径的规划图，确保每条合法解释都被保留；(b) **Reasoning Agent**：沿每条规划路径独立进行证据检索和推理，最终综合所有路径的结果生成完整答案
    - 设计动机：标准 RAG/CoT 方法倾向于在第一跳就锁定单一解释（过早提交），CLARION 通过规划-执行分离防止路径的过早剪枝

### 损失函数 / 训练策略

MARCH 和 CLARION 均为无训练的(training-free)方案。MARCH 是构建型基准，CLARION 基于现有 LLM 通过提示工程实现。评估使用 F1-score、EM、D-F1(disambiguation F1)、ROUGE-L 和 LLM-judge 等指标。

## 实验关键数据

### 主实验

| 设置 | MuSiQue(多跳) | ASQA(歧义) | MARCH(交叉) | 说明 |
|------|-------------|-----------|------------|------|
| 现有模型 | 尚可 | 尚可 | **显著下降** | 交叉场景远超单一场景的难度 |
| CLARION | - | - | **显著优于基线** | 验证了解耦策略的有效性 |

### 基准统计

| 指标 | 值 | 说明 |
|------|-----|------|
| 总样本数 | 2,209 | 覆盖三类歧义 |
| 歧义分布 | Sem:734, Syn:739, Const:736 | 均衡分布 |
| 平均跳数 | 2.11-2.95 | 句法歧义跳数最多 |
| 人工验证一致性 | Fleiss' κ=0.92-0.95 | 极高标注一致性 |
| 长答案有效率 | >90% | 整合了所有解释 |

### 关键发现
- 真实用户查询中 13.3% 同时涉及多跳和歧义，这不是罕见边缘情况
- 在单独的多跳或歧义任务上表现合理的模型，在交叉场景(MARCH)上性能急剧下降
- 模型倾向于在第一跳就锁定单一解释（过早提交），导致级联错误
- CLARION 的规划-执行解耦有效防止了推理路径的过早剪枝

## 亮点与洞察
- **问题定义的深度**："潜在歧义"（latent ambiguity，只有前序步骤正确才能显现的歧义）的概念非常深刻
- **三类歧义的分类**：语义/句法/约束歧义各有对应的处理动作(解释/解析/泛化)，分类体系实用
- **严格的基准构建流程**：4个 LLM 全票一致 + 人工验证，质量可靠性高
- **13.3% 的真实世界频率数据**：从 lmsys-chat-1m 中的统计数据有力论证了问题的现实重要性
- **CLARION 的简洁设计**：规划-执行解耦的思路简单但有效

## 局限与展望
- MARCH 基于 MuSiQue 构建，继承了其领域和跳数的限制
- CLARION 目前基于检索增强设置，开放域无检索场景未探索
- 三类歧义的均衡分布是人工控制的，可能不完全反映真实分布
- 未来可探索模型在何种条件下应主动提出澄清问题而非尝试所有解释
- 可扩展到多语言多跳歧义场景

## 相关工作与启发
- **vs ASQA**：ASQA 只关注单跳歧义，MARCH 首次扩展到多跳歧义
- **vs MuSiQue**：MuSiQue 关注多跳但假设无歧义，MARCH 在其基础上引入歧义
- **vs 标准 RAG/CoT**：标准方法在多跳歧义下因过早提交而失败，CLARION 通过规划-执行解耦解决

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义和评估多跳歧义 QA，问题重要且此前未被研究
- 实验充分度: ⭐⭐⭐⭐ 含基准构建、人工验证、模型评估和框架对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，分类体系严谨，实例分析生动
- 价值: ⭐⭐⭐⭐⭐ 基准和框架均有独立贡献，对推理和歧义研究社区有重要影响

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2026\] Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)
- [\[ACL 2025\] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](../../ACL2025/llm_reasoning/beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[NeurIPS 2025\] 笔记6：Self-Evaluating LLMs - 多步任务的步级置信度估计](../../NeurIPS2025/llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
