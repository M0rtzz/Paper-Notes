---
title: >-
  [论文解读] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning
description: >-
  [ICML 2025][人体理解][rulebreaker] 构建首个大规模"规则破坏者"数据集 RULEBREAKERS（25,600 实例），系统评估 7 个 LLM 在形式逻辑推理与事实知识冲突时的表现，发现模型普遍倾向过度刚性地应用逻辑规则而忽略常识，与人类推理行为存在显著偏离。
tags:
  - ICML 2025
  - 人体理解
  - rulebreaker
  - formal logic
  - human-like reasoning
  - modus tollens
  - disjunctive syllogism
---

# RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning

**会议**: ICML 2025  
**作者**: Jason Chan, Robert Gaizauskas, Zhixue Zhao (University of Sheffield)  
**arXiv**: [2410.16502](https://arxiv.org/abs/2410.16502)  
**代码**: [GitHub](https://github.com/jasonchanly/rulebreakers)  
**领域**: LLM评估, 认知科学, 推理  
**关键词**: rulebreaker, formal logic, human-like reasoning, modus tollens, disjunctive syllogism  

## 一句话总结

构建首个大规模"规则破坏者"数据集 RULEBREAKERS（25,600 实例），系统评估 7 个 LLM 在形式逻辑推理与事实知识冲突时的表现，发现模型普遍倾向过度刚性地应用逻辑规则而忽略常识，与人类推理行为存在显著偏离。

## 研究背景与动机

**领域现状**：形式逻辑是 NLP 中实现推理的经典手段——将自然语言转为符号形式，再应用预定义规则推导结论。越来越多的研究利用形式逻辑增强 LLM 推理能力，包括逻辑数据增强、逻辑约束推理、符号求解器集成等。

**现有痛点**：认知科学研究（Johnson-Laird 等）早已发现，纯粹基于逻辑形式的推理与人类实际推理方式存在根本差异。关键问题在于"规则破坏者"（rulebreaker）场景：形式逻辑推导出的结论虽然在符号层面有效，但与前提在事实上矛盾，人类通常不会接受这样的结论。例如："Anne 在斯德哥尔摩或瑞典某处"且"Anne 不在瑞典"，析取三段论推出"Anne 在斯德哥尔摩"——但知道斯德哥尔摩在瑞典的人不会接受这个结论。

**核心矛盾**：形式逻辑规则假设连接词（if、or、and）具有固定语义，但自然语言中连接词的解释高度依赖句子的语义内容。现有 LLM 推理评估都假设只要逻辑形式正确结论就正确，忽视了语义内容应当影响推理判断的情形。

**本文目标** (1) 构建严格受控的 rulebreaker 评估基准；(2) 评估 LLM 能否像人类一样区分 rulebreaker 和 non-rulebreaker；(3) 分析失败的潜在原因。

**切入角度**：从认知科学实验范式出发，设计最小差异的 rulebreaker/non-rulebreaker 对，控制逻辑形式相同但语义内容不同，隔离评估语义感知能力。

**核心 idea**：用认知科学启发的 rulebreaker 数据集检验 LLM 是否能在推理中整合常识知识而非机械套用逻辑规则。

## 方法详解

### 整体框架

RULEBREAKERS 是一个包含 25,600 个实例的评估数据集，由 12,800 个 rulebreaker 和 12,800 个配对的 non-rulebreaker 组成。数据集通过 4 种模板（2 种逻辑规则 × 2 种实体类型）系统生成，并配套设计了配对准确率和模型置信度两类评估指标。

### 关键设计

1. **四模板数据生成系统**:

    - 功能：系统化生成最小差异的 rulebreaker/non-rulebreaker 对
    - 核心思路：组合两种逻辑规则（否定后件 MT："if P then not Q; Q; ∴ not P" 和析取三段论 DS："P or Q; not Q; ∴ P"）与两种实体类型（地理对：183 个国家-首都对来自 WikiData；类别对：91 个类型-实例对来自 ConceptNet，涵盖鸟类、鱼类、乐器等）。每个实体对进一步与 5-6 个动词和 5 个人名排列组合。Non-rulebreaker 通过随机替换国家/类型名使结论不再与前提矛盾
    - 设计动机：模板化生成确保大规模、最小差异、可控变量，避免手工构造的不一致性

2. **配对准确率指标 (Paired Accuracy)**:

    - 功能：评估模型同时正确处理 rulebreaker 和 non-rulebreaker 的能力
    - 核心思路：$\tau = \frac{1}{|D^{paired}|}\sum_{(x^R, x^N)} \mathbb{1}[T(x^R) \wedge T(x^N)]$，要求模型对 rulebreaker 回答"否"且对配对的 non-rulebreaker 回答"是"才算正确。随机猜测基线为 0.25，全部回答"是"得分为 0
    - 设计动机：消除模型固有的回答偏好（如总说"是"）对评估的干扰，比单独看准确率更严格

3. **模型置信度分析**:

    - 功能：检测模型是否具有潜在的区分 rulebreaker 能力
    - 核心思路：提取模型对"yes/true"的输出概率 $p^+(x)$，比较在 non-rulebreaker 上正答置信度 $\Pi_{D^N}^+$ 与在 rulebreaker 上正（但错误）答置信度 $\Pi_{D^R}^+$ 的差异，用 Welch's t-test 检验显著性
    - 设计动机：即使模型最终输出错误，置信度差异可能揭示潜在的区分信号

### 损失函数

本工作为评估性研究，不涉及模型训练。

## 实验

### 主实验——配对准确率

| 模型 | 配对准确率 τ | Rulebreaker 准确率 | Non-rulebreaker 准确率 |
|------|------------|-------------------|----------------------|
| LLaMA-3-8B-Instruct | **~0.609** | 中等 | 高 |
| LLaMA-3-70B-Instruct | ~0.497 | 中等 | 高 |
| Mistral-7B-Instruct | ~0.476 | 中等 | 高 |
| Phi-3-medium-128k | ~0.292 | 低 | 高 |
| Phi-3-mini-128k | ~0.208 | 低 | 非常高 |
| GPT-4o | ~0.14 | 非常低 | **100%** |
| Gemma-2-27b-it | ~0.071 | 非常低 | **100%** |

### 消融实验——模型置信度与失败原因分析

| 模型 | $\Pi_{D^R}^+$ (%) | $\Pi_{D^N}^+$ (%) | 差异显著性 |
|------|-------------------|-------------------|-----------|
| Phi-3-mini | 92.05 | 96.22 | p<0.0001 |
| Phi-3-medium | 93.96 | 97.16 | p<0.0001 |
| LLaMA-3-8B | 77.20 | 90.46 | p<0.0001 |
| LLaMA-3-70B | 96.34 | 99.95 | p<0.0001 |
| Mistral-7B | 92.55 | 98.11 | p<0.0001 |
| Gemma-2-27b | 97.99 | 100.00 | p<0.0001 |

### 关键发现

- **反直觉的负相关**：non-rulebreaker 表现最好的模型（Gemma-2-27b、GPT-4o 达到 100%）在 rulebreaker 上表现最差，说明这些模型过度泛化了逻辑规则
- **潜在区分能力存在**：所有模型在 non-rulebreaker 上的正答置信度显著高于在 rulebreaker 上的正答置信度，说明潜在的区分信号存在但未被利用
- **失败与实体熟悉度相关**：除 LLaMA-3-8B 和 GPT-4o 外，其余模型在正确回答组中的实体熟悉度高于错误回答组
- **注意力分配影响**：准确率最高的 LLaMA-3-8B/70B 对第二前提（事实性前提）的注意力比率也最高
- **DS 优于 MT，类别优于地理**：模型在析取三段论上表现普遍优于否定后件，在类别实体对上优于地理实体对

## 亮点与洞察

- 首个大规模 rulebreaker 数据集，设计精巧，最小差异控制严格，评估指标消除了回答偏好干扰
- 揭示了 LLM 推理的深层盲点：模型可以同时擅长事实检索与逻辑规则应用，却无法识别两者冲突
- 对当前依赖形式逻辑增强 LLM 推理的研究趋势发出重要警示——这可能进一步加剧 LLM 与人类推理的偏离
- 置信度分析表明模型内部存在被压制的区分信号，为改进方向提供了线索

## 局限性

- 仅涵盖两种逻辑规则（MT 和 DS），未覆盖其他推理模式（如假言三段论、归谬法等）
- 实体集合有限（183 个地理对 + 91 个类别对），可能不充分代表真实世界知识的多样性
- 评估限于选择式回答格式，未充分测试开放式生成场景下的推理行为
- 未提出改进 LLM 在 rulebreaker 上表现的具体方法，停留在诊断层面

## 相关工作与启发

- **认知科学基础**：直接继承心理模型理论（Johnson-Laird 1983）的实验范式，将 rulebreaker 概念系统化为 NLP 评估工具
- **与内容效应的关系**：Lampinen et al. (2024) 发现 LLM 在形式逻辑任务上也受语义内容影响，但方向相反——本文关注模型应当利用语义内容但未能利用的场景
- **启发**：可以设计训练策略（如对比学习）让模型学习何时应用逻辑规则、何时依赖常识

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ | 首次将认知科学的 rulebreaker 概念系统引入 LLM 评估 |
| 技术深度 | ⭐⭐⭐⭐ | 指标设计严谨，多维分析深入，但无模型改进 |
| 实验完整度 | ⭐⭐⭐⭐ | 7 个模型 × 10 种表述变体，配对+置信度+归因三层分析 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 问题动机阐述清晰，数据集构造透明 |
| 实用性 | ⭐⭐⭐⭐ | 数据集开源，对逻辑增强方法的警示具有实际指导价值 |

<!-- RELATED:START -->

## 相关论文

- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](../../CVPR2025/human_understanding/freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)
- [SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/human_understanding/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](../../ECCV2024/human_understanding/bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)
- [Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)

<!-- RELATED:END -->
