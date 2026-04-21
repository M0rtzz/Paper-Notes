---
title: >-
  [论文解读] PERSIST: Persistent Instability in LLM's Personality Measurements
description: >-
  [AAAI 2026][LLM/NLP][LLM人格测量] PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。
tags:
  - AAAI 2026
  - LLM/NLP
  - LLM人格测量
  - 行为一致性
  - 推理模式
  - 对齐评估
  - 心理测量
---

# PERSIST: Persistent Instability in LLM's Personality Measurements

**会议**: AAAI 2026  
**arXiv**: [2508.04826](https://arxiv.org/abs/2508.04826)  
**代码**: https://github.com/tosatot/PERSIST  
**领域**: NLP生成 / LLM评估  
**关键词**: LLM人格测量, 行为一致性, 推理模式, 对齐评估, 心理测量

## 一句话总结
PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。

## 研究背景与动机

**领域现状**：LLM 对齐和安全评估需要衡量模型的行为特征（大五人格、暗黑三元组等），通常使用心理学量表。

**现有痛点**：(1) 问题顺序、改写都可导致评分大幅变化；(2) 不清楚不稳定性是人类中心化问卷的伪影还是模型本身的问题；(3) 推理、会话历史等因素对稳定性的影响未被系统研究。

**核心矛盾**：无法可靠测量 LLM 行为特征→无法可靠评估对齐效果。

**本文目标** 全面量化 LLM 人格测量不稳定性的来源和程度。

**切入角度**：全因素设计——25模型×5人设×4问卷×250排列×100改写×推理/非推理×会话历史。

**核心 idea**：LLM 的人格测量不稳定性是持久的、结构性的，不能仅靠模型规模增长来解决。

## 方法详解

### 整体框架
PERSIST 三模块：生成引擎(vLLM)、响应处理(token 级 log 概率提取)、分析管道(层次聚合计算排列间 SD)。

### 关键设计

1. **双版本问卷**：BFI-44 + SD3（原版）+ BFI-LLM + SD3-LLM（替换人类特定概念），验证不稳定性是否为问卷设计的伪影
2. **5种人设**：Assistant(基线)、Buddhist、Teacher、Antisocial、Schizophrenia
3. **稳定性指标**：问题级 SD（跨250个排列）+ 困惑度 $\exp(-\log p)$
4. **温度自适应推理**：τ=0 确定性输出，τ=0.6 推理模式

### 损失函数 / 训练策略
纯评估研究，无训练。25 个模型通过 vLLM 和 HuggingFace Inference API 访问。

## 实验关键数据

### 主实验

| 发现 | Spearman ρ | p值 | 效应 |
|------|-----------|-----|------|
| 模型规模↑ → 正面特质↑ | 显著 | 0.001** | 更"友善" |
| 模型规模↑ → 负面特质↓ | 显著 | <0.001*** | 更"温和" |
| 模型规模↑ → 变异性↓ | 显著 | <0.001*** | 更稳定 |
| 模型规模↑ → 困惑度 | 不显著 | 0.934 | 无关联 |

### 推理悖论

| 条件 | 变异性 | 困惑度 |
|------|--------|--------|
| 无推理 | 低 | 高 |
| 有推理(CoT) | **高** (p<0.001) | **低** (p<0.01) |

### 会话历史的双刃剑效应

| 模型大小 | 有历史的效果 | p值 |
|---------|------------|------|
| <50B (n=19) | **增加**变异性 | <0.001*** |
| ≥50B (n=4) | **降低**变异性 | <0.001*** |

### 关键发现
- 即使 400B+模型仍有 SD>0.3——规模不是解药
- CoT 推理增加变异性但降低困惑度（更自信却更不一致——推理悖论）
- LLM 适配问卷与原版表现出相同的不稳定性——非问卷设计的伪影
- 错位人设(反社会、精神分裂)比基线人设变异性更高、困惑度更高
- 困惑度与变异性仅中等相关(ρ=0.465)——不是完整的稳定性指标

## 亮点与洞察
- **"推理悖论"是核心发现**：CoT 让模型更自信(困惑度↓)却更不一致(SD↑)——挑战"更多推理=更好"的直觉
- **会话历史的规模依赖效应**：帮助大模型但伤害小模型，提示工程需考虑模型规模
- **对对齐评估的警示**：测量工具本身不可靠时，对齐评估结论如何可信？

## 局限与展望
- 人格测量的 construct validity 存疑——LLM 是否真有"人格"？
- 仅使用心理学量表，未覆盖其他行为评估方式（如博弈论场景）
- 未探索微调或 RLHF 对稳定性的影响

## 相关工作与启发
- 对 LLM 安全评估方法论的直接冲击——所有基于问卷的对齐评估都需要重新审视其稳定性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全因素设计揭示多个反直觉发现
- 实验充分度: ⭐⭐⭐⭐⭐ 25模型、200万+响应、系统统计检验
- 写作质量: ⭐⭐⭐⭐⭐ 统计分析严谨，呈现清晰
- 价值: ⭐⭐⭐⭐⭐ 对LLM安全和对齐评估社区有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [Smart: 基于 GNN 与 LLM 融合的 Dragonfly 系统应用运行时代理模型](smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)
- [Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)

<!-- RELATED:END -->
