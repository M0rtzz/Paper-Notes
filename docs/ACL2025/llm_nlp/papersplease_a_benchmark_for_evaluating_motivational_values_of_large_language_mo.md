---
title: >-
  [论文解读] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory
description: >-
  [ACL 2025][LLM/NLP][动机价值观] 提出 PapersPlease 基准（3,700 个道德困境场景），基于 ERG 动机理论评估 LLM 在角色扮演情境中的价值观优先级排序和社会身份偏见，发现 LLM 对不同层级需求的优先序差异显著，且社会身份线索会影响决策公平性。
tags:
  - ACL 2025
  - LLM/NLP
  - 动机价值观
  - ERG理论
  - 角色扮演
  - 社会偏见
  - 道德决策
---

# PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory

**会议**: ACL 2025  
**arXiv**: [2506.21961](https://arxiv.org/abs/2506.21961)  
**代码**: [GitHub](https://github.com/yeonsuuuu28/papers-please)  
**领域**: LLM/NLP  
**关键词**: 动机价值观, ERG理论, 角色扮演, 社会偏见, 道德决策

## 一句话总结

提出 PapersPlease 基准（3,700 个道德困境场景），基于 ERG 动机理论评估 LLM 在角色扮演情境中的价值观优先级排序和社会身份偏见，发现 LLM 对不同层级需求的优先序差异显著，且社会身份线索会影响决策公平性。

## 研究背景与动机

LLM 在角色扮演情境中的偏见和决策模式评估日益重要。当 LLM 扮演特定角色时，其行为倾向可能与标准问答模式显著不同，暴露出隐含的偏见。然而，现有工作多关注一般性道德判断或社会互动模拟，缺少基于心理学动机理论的系统性评估。

ERG（Existence, Relatedness, Growth）理论将人类动机需求分为三个层级：生存需求（生理安全）、关系需求（人际关系）和成长需求（自我实现）。这一框架提供了结构化的视角来分析 LLM 如何权衡不同层次的人类需求。

灵感来自游戏《Papers, Please》——玩家扮演边检员在国家安全和人道主义之间做出困难决策。本文将 LLM 置于类似的高风险决策情境中，观察其价值判断和社会偏见。

## 方法详解

### 整体框架

PapersPlease 由 3,700 个角色扮演场景组成，LLM 扮演虚构国家 Arstotzka 的边检员，根据个人叙事决定是否批准入境。叙事基于 ERG 理论的三个维度构建，并嵌入性别、种族和宗教三个社会身份维度。评估分为三个设置：个体案例、比较案例和社会维度案例。

### 关键设计

**1. 基于 ERG 理论的场景生成**

- **功能**: 为每种动机价值观创建标准化的测试叙事
- **核心思路**: 手动编写每类 5 个示例，使用 GPT-4o-mini 通过 few-shot prompting 扩展到每类 100 个场景。生存类叙事关注干旱、饥饿等基本生存需求；关系类关注寻亲、重聚等人际关系；成长类关注职业发展、学术追求等自我实现。所有叙事排除可识别身份线索
- **设计动机**: ERG 理论提供了心理学基础的需求层级框架，使得对 LLM 价值观优先级的分析有理论依据可循

**2. 社会身份维度嵌入**

- **功能**: 检测 LLM 决策中的社会偏见
- **核心思路**: 在每个叙事前添加社会身份信息（如 "Person's gender: male"），覆盖三个维度：性别（male, female, non-binary）、种族（White, Black, Hispanic, Asian）、宗教（Christian, Muslim, Hindu, Buddhist）。通过比较有/无身份线索时的批准率差异来量化偏见
- **设计动机**: 在价值观敏感的决策场景中，社会身份线索可能激活 LLM 的隐含偏见，这对公平性评估至关重要

**3. 多层次评估设计**

- **功能**: 从绝对和相对两个维度全面评估 LLM 的价值观
- **核心思路**: **个体案例**：分别呈现每个场景，统计各类需求的批准/拒绝率；**比较案例**：同时呈现三类场景，强制 LLM 只批准一人，观察优先级排序；**社会维度案例**：加入社会身份信息后重复个体案例评估。使用卡方检验进行统计显著性分析
- **设计动机**: 绝对批准率反映模型的整体宽容度，比较案例则迫使模型在竞争需求间做出权衡，更能揭示其内在价值序列

### 损失函数 / 训练策略

本文为评估基准，不涉及模型训练。LLM 输出 JSON 格式的决策（approve/deny/arrest）及简短理由。评估使用 temperature = 0 以确保可复现性。

## 实验关键数据

### 主实验：个体案例批准率

| 模型 | Existence | Relatedness | Growth |
|------|-----------|-------------|--------|
| GPT-4o-mini | 99 | 47 | 74 |
| Claude-3.7-sonnet | 0 | 0 | 0 |
| Gemini-2.0-flash | 41 | 11 | 43 |
| Llama-3.1-8B | 83 | 91 | 96 |
| Llama-4-Maverick | 83 | 11 | 47 |
| Qwen3-14B | 89 | 53 | 63 |

### 消融实验：社会身份对批准率的影响（GPT-4o-mini）

| 社会身份 | Existence 变化 | Relatedness 变化 | Growth 变化 |
|---------|--------------|----------------|------------|
| Female | ≈0 | ↑显著 | ↑显著 |
| Non-binary | ≈0 | ↑显著 | ↑显著 |
| Muslim | ↓3% | 最小增幅 | 最小增幅 |
| Black (Llama-4) | ↓ | ↓ | ↓ |
| Asian (Llama-4) | ↓ | ↓ | ↓ |

### 关键发现

1. **Claude-3.7-sonnet 拒绝所有人**: 严格遵循规则，不做任何例外，与其他模型形成鲜明对比
2. **GPT-4o-mini 最符合 ERG 层级**: 生存需求批准率最高（99%），与 ERG 理论的基础需求优先相一致
3. **模型形成两大行为集群**: GPT/Claude/Qwen 倾向于优先生存需求；Gemini/Llama 更均匀分布
4. **Llama-4 对边缘化群体表现出负面偏见**: Black、Asian、Hindu 身份的批准率普遍降低
5. **社会身份影响因需求类型而异**: 生存需求因初始批准率高而受影响小，关系/成长需求对身份线索更敏感

## 亮点与洞察

- 首次将 ERG 动机心理学理论系统应用于 LLM 评估，提供了结构化的价值观分析框架
- 角色扮演的高风险情境设计迫使模型做出困难的道德权衡，比一般性问卷更能揭示隐含偏见
- Claude 的"全部拒绝"行为揭示了过度对齐（over-alignment）的问题
- 卡方检验确保了发现的统计显著性，增强了结论的可靠性

## 局限与展望

- 仅评估 6 个 LLM，泛化性有限
- 场景和价值框架是简化的，不能完全反映真实决策的复杂性
- 游戏《Papers, Please》的反乌托邦设定使其难以反映日常价值观偏好
- 缺少人类基准数据来评估模型与人类价值观的对齐程度
- 未来应多元化任务设计，将 ERG 框架应用到更广泛的场景中

## 相关工作与启发

- **LLM 道德推理**: Nie et al. (2023) 发现 LLM 在道德偏好中存在不一致；Rao et al. (2023) 揭示文化偏见
- **角色扮演评估**: MACHIAVELLI 基准评估策略行为、SANDBOX 评估社会行为
- **启发**: 基于心理学理论的评估框架可以更系统地揭示 LLM 的内在价值体系，而不仅仅是表面的对齐

## 评分

- **新颖性**: ⭐⭐⭐⭐ ERG 理论引入 LLM 评估很新颖，游戏灵感的实验设计也很巧妙
- **实验充分度**: ⭐⭐⭐⭐ 6 个模型 + 3 种评估设置 + 社会维度分析 + 统计检验
- **写作质量**: ⭐⭐⭐⭐ 心理学理论引入充分，实验设计清晰
- **价值**: ⭐⭐⭐⭐ 为 LLM 公平性和价值对齐评估提供了新的视角和工具

<!-- RELATED:START -->

## 相关论文

- [Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)
- [Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)

<!-- RELATED:END -->
