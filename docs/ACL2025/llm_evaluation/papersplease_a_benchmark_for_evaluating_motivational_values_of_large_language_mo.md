---
title: >-
  [论文解读] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory
description: >-
  [ACL 2025 (GEM2 Workshop)][LLM评测][LLM价值观评估] 提出PapersPlease基准，包含3700个基于ERG动机理论构建的道德困境场景，让LLM扮演移民检查官决定是否放行，揭示了6个LLM在动机价值优先级上的显著差异以及对边缘化身份群体的偏见。 领域现状：通过角色扮演场景评估LLM的行为…
tags:
  - "ACL 2025 (GEM2 Workshop)"
  - "LLM评测"
  - "LLM价值观评估"
  - "ERG理论"
  - "角色扮演"
  - "社会偏见"
  - "道德推理"
---

# PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory

**会议**: ACL 2025 (GEM2 Workshop)  
**arXiv**: [2506.21961](https://arxiv.org/abs/2506.21961)  
**代码**: [https://github.com/yeonsuuuu28/papers-please](https://github.com/yeonsuuuu28/papers-please)  
**领域**: LLM评测  
**关键词**: LLM价值观评估, ERG理论, 角色扮演, 社会偏见, 道德推理

## 一句话总结
提出PapersPlease基准，包含3700个基于ERG动机理论构建的道德困境场景，让LLM扮演移民检查官决定是否放行，揭示了6个LLM在动机价值优先级上的显著差异以及对边缘化身份群体的偏见。

## 研究背景与动机

**领域现状**：通过角色扮演场景评估LLM的行为和偏见已成为主流方法。现有工作如MACHIAVELLI基准通过文本游戏评估策略行为，SANDBOX通过多智能体交互模拟人类社会行为。

**现有痛点**：现有道德推理评估存在两个不足：（1）缺乏基于心理学理论的系统化评估框架，大多数工作只是测试LLM对简单道德判断的一致性；（2）很少将社会身份（种族、性别、宗教）与动机价值相结合进行评估，忽略了身份偏见如何影响LLM在价值敏感场景中的决策。

**核心矛盾**：LLM在角色扮演中展现出与标准问答模式截然不同的行为倾向，但我们缺乏结构化的方法来评估这些隐式编码的价值优先级是否与人类动机价值对齐。

**本文目标**：设计一个基于心理学理论（ERG理论）的角色扮演基准，系统评估LLM如何在竞争性的人类需求之间做出价值判断，以及社会身份如何影响其决策。

**切入角度**：受经典游戏《Papers, Please》启发——在该游戏中玩家扮演移民检查官面对道德困境。作者将这一高压决策情境与Alderfer的ERG理论（将人类需求分为生存、关系、成长三层）结合，创造了一个有强制选择约束的评估环境。

**核心 idea**：让LLM扮演虚构国家Arstotzka的移民检查官，面对基于ERG三层需求构建的申请者叙事做出放行/拒绝决定，同时嵌入社会身份线索来检测偏见。

## 方法详解

### 整体框架
输入是一个移民场景描述（申请者的背景叙事），输出是LLM的决策（批准/拒绝/逮捕）。整个评估分三个维度：（1）个体案例评估：逐个场景独立决策；（2）比较案例评估：三选一强制优先级排序；（3）社会维度评估：在叙事中加入身份线索后观察决策变化。

### 关键设计

1. **基于ERG理论的场景生成**:

    - 功能：生成结构化的道德困境叙事数据集
    - 核心思路：ERG理论将人类动机分为三层——生存需求（Existence，如食物、安全）、关系需求（Relatedness，如家庭团聚）、成长需求（Growth，如职业发展）。为每个类别手工编写5个代表性示例，然后用GPT-4o-mini通过few-shot prompting扩展到每类100个场景，共300个基础场景。所有场景经人工审核确保质量
    - 设计动机：ERG理论提供了心理学上经过验证的层次化人类需求框架，比随机构造道德困境更具理论基础和可解释性

2. **社会身份嵌入机制**:

    - 功能：检测LLM决策中的社会偏见
    - 核心思路：在每个叙事前添加简短的社会身份标注（如"Person's gender: male"），覆盖三个维度——性别（男/女/非二元，3选项）、种族（白人/黑人/西班牙裔/亚裔，4选项）、宗教（基督教/穆斯林/印度教/佛教，4选项）。通过对比有无身份线索时的决策差异来量化偏见。总计产生3700个场景
    - 设计动机：仅评估动机价值优先级不够，还需检测这些优先级是否受社会身份影响——这对评估LLM在实际敏感应用中的公平性至关重要

3. **三种评估范式**:

    - 功能：从不同角度全面评估LLM的价值判断
    - 核心思路：个体案例——独立评估每个场景的接受/拒绝率，观察绝对偏好；比较案例——三个不同ERG类别的场景同时呈现只能批准一个，强制做优先级排序；社会维度——加入身份线索后重复个体评估，通过差值分析偏见方向和幅度
    - 设计动机：个体评估可能受模型整体宽松/严格程度影响，比较评估消除了这一混淆因子；社会维度评估则专门针对公平性问题

### 损失函数 / 训练策略
无训练——纯评估基准。使用temperature=0进行确定性推理，通过Chi-Square检验和事后成对比较进行统计显著性分析。

## 实验关键数据

### 主实验（个体案例评估 - 接受数/100场景）

| 模型 | 生存需求 | 关系需求 | 成长需求 |
|------|----------|----------|----------|
| GPT-4o-mini | 99 | 47 | 74 |
| Claude-3.7-sonnet | 0 | 0 | 0 |
| Gemini-2.0-flash | 41 | 11 | 43 |
| Llama-3.1-8B | 83 | 91 | 96 |
| Llama-4-Maverick | 83 | 11 | 47 |
| Qwen3-14B | 89 | 53 | 63 |

### 比较案例评估（强制三选一优先级）

| 模式 | 代表模型 |
|------|----------|
| 生存>成长>关系（符合ERG层次） | GPT-4o-mini, Claude-3.7, Qwen3 |
| 均衡分布（偏离ERG层次） | Gemini-2.0, Llama-4, Llama-3.1 |

### 关键发现
- Claude-3.7-sonnet在所有个体场景中100%拒绝，严格遵循规则而忽略人道主义考量——这是极端的规则优先行为
- 两个模型集群：GPT/Claude/Qwen优先生存需求（符合ERG层次）；Gemini/Llama更均衡（可能偏离人类直觉的优先级）
- 社会身份影响显著：Llama-4对黑人、亚裔、穆斯林、印度教身份全面降低通过率；GPT-4o-mini在关系和成长类别中对大多数身份提高通过率，但穆斯林身份例外
- Chi-Square检验显示模型间差异和模型组间差异均具有统计显著性（$p<0.05$）

## 亮点与洞察
- **心理学理论驱动的LLM评估**：将Alderfer的ERG理论引入LLM评估是一个聪明的框架选择，比临时构造的道德困境更具理论深度和可重复性
- **游戏化的评估设计**：受《Papers, Please》启发的情境设计创造了自然的强制决策压力，比简单的"你觉得对不对"提问更能暴露LLM的隐式价值偏好
- **身份偏见的精细化分析**：将动机类别×社会身份的交叉分析揭示了更细粒度的偏见模式——例如某些模型只在特定需求类别中对特定身份存在偏见

## 局限与展望
- 仅评估了6个LLM，样本量有限制泛化性
- 场景设计基于虚构极端情境（反乌托邦边境检查），与日常生活中的价值判断可能存在差距
- 决策是二元的（批准/拒绝），未来可用连续评分（0-10）捕获更细微的偏好差异
- 缺乏与人类实验的直接对比——虽然用ERG理论推断人类期望优先级，但没有收集真实人类数据作为基准

## 相关工作与启发
- **vs MACHIAVELLI**: MACHIAVELLI评估策略决策中的道德权衡，PapersPlease专注于动机价值的层次化优先级——更接近心理学评估
- **vs MoCA**: MoCA评估基于认知科学文献的道德规范一致性，但不涉及社会身份的影响。PapersPlease将价值评估与公平性检测结合更全面
- 这个评估框架可以扩展到其他心理学理论（如Maslow需求层次），也可以推广到非移民场景——例如医疗资源分配、灾难救援优先级

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将ERG理论与角色扮演结合评估LLM动机价值，设计思路新颖
- 实验充分度: ⭐⭐⭐ 6个模型3种评估维度覆盖面可以，但缺乏人类基准对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设计合理，统计分析规范
- 价值: ⭐⭐⭐⭐ 为LLM的隐式价值系统评估提供了新视角和可操作的工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](../../ICML2025/llm_evaluation/position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)

</div>

<!-- RELATED:END -->
