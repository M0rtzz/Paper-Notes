---
title: >-
  [论文解读] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
description: >-
  [NeurIPS 2025][LLM推理][时间推理] 提出 TimE，一个包含 38,522 个 QA 对的多层级时间推理基准，覆盖知识密集（Wiki）、动态新闻（News）、长对话（Dial）三种真实场景和三级渐进式 11 子任务，全面评估 24 个 LLM 后发现即便最强推理模型在时间线构建和反事实推理等复杂任务上仍有显著短板。
tags:
  - NeurIPS 2025
  - LLM推理
  - 时间推理
  - 多层级基准
  - 真实世界场景
  - 知识密集型
  - 长对话
---

# TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios

**会议**: NeurIPS 2025  
**arXiv**: [2505.12891](https://arxiv.org/abs/2505.12891)  
**代码**: [GitHub](https://github.com/) / [HuggingFace Dataset](https://huggingface.co/)  
**领域**: LLM推理 / 时间推理  
**关键词**: 时间推理, 多层级基准, 真实世界场景, 知识密集型, 长对话

## 一句话总结

提出 TimE，一个包含 38,522 个 QA 对的多层级时间推理基准，覆盖知识密集（Wiki）、动态新闻（News）、长对话（Dial）三种真实场景和三级渐进式 11 子任务，全面评估 24 个 LLM 后发现即便最强推理模型在时间线构建和反事实推理等复杂任务上仍有显著短板。

## 研究背景与动机

**领域现状**：时间推理是 LLM 理解现实世界的关键能力。现有基准如 TimeBench 和 TRAM 主要聚焦于简化场景（如基础时间常识、短文本内的时间关系），任务设计较为简单，对当前 LLM 已不构成足够挑战。

**现有痛点**：真实世界的时间推理面临三大挑战：(1) 知识密集型场景中时间信息密度高且实体关联复杂；(2) 新闻事件快速演变，细节随时间变化；(3) 多轮对话中时间依赖关系复杂、跨越长上下文。现有基准未覆盖这些难点。此外，时间推理本质上是一个层级能力框架（从基础理解到复杂关系推理），但现有工作往往只关注单一维度。

**核心矛盾**：如何设计一个既覆盖真实世界复杂性、又系统化分层评估的时间推理基准？现有基准的任务过于简单（如 TimeBench 的基础任务）或只聚焦特定方面（如 TReMu 只做对话中的时间定位），缺乏统一框架。

**本文目标** (1) 缺乏涵盖多种真实场景的时间推理基准；(2) 缺乏从基础到复杂的系统化分级评估；(3) 缺乏高质量人工标注子集用于可靠评估。

**切入角度**：从三种真实数据源（Wikidata 知识图谱、在线新闻、多轮长对话）出发，设计三级渐进式任务体系，配合人工标注的 TimE-Lite 子集提供高质量评估锚点。

**核心 idea**：构建覆盖"时间理解→时间表达推理→复杂时间关系推理"三级渐进、跨三种真实场景的大规模时间推理基准。

## 方法详解

### 整体框架

TimE 由三个子数据集组成：(1) TimE-Wiki（13,848 QA）——基于 Wikidata 时间知识图谱构建，评估知识密集场景下的时间推理；(2) TimE-News（19,958 QA）——基于在线新闻中的时态复杂事件（TCE），评估动态事件理解；(3) TimE-Dial（4,716 QA）——基于超长多轮对话，评估交互式场景的时间推理。所有子数据集共享三级任务体系。

### 关键设计

1. **三级渐进式任务体系**:

    - 功能：从基础到复杂系统化评估时间推理能力
    - 核心思路：Level-1（基础时间理解与检索）包含 5 个子任务——Extract（时间表达式提取）、Localization（事件-时间映射）、Computation（持续时间计算）、DurationCompare（时间段比较）、OrderCompare（时间序列排序）。Level-2（时间表达式推理）包含 3 个子任务——Explicit Reasoning（未明确提及的时间推理）、Order Reasoning（序数时间定位）、Relative Reasoning（相对时间参照推理）。Level-3（复杂时间关系推理）包含 3 个子任务——Co-temporality（同时性识别）、Timeline（多事件时间排序）、Counterfactual（反事实时间推理）
    - 设计动机：模拟人类处理时间信息的认知过程——先捕获理解、再推理隐含表达、最后厘清复杂关系

2. **多源数据构建管道**:

    - 功能：从真实数据源自动生成高质量 QA 对
    - 核心思路：对于 TimE-Wiki，使用 SLING 解析 Wikidata 提取时间事实四元组并构建多跳时间知识图谱，用 DeepSeek-V3 生成自然语言上下文。对于 TimE-News，基于已有的时态复杂事件数据集通过 RAG 检索相关文段。对于 TimE-Dial，利用 LLM 从长对话中总结事件图谱并标准化时间表达式。QA 生成结合规则模板和 LLM（DeepSeek-V3/R1），对自由格式问题还通过 STARC 框架生成干扰选项
    - 设计动机：规则+LLM 结合保证 QA 的逻辑正确性和自然性；干扰选项提升评估区分度

3. **TimE-Lite 人工标注子集**:

    - 功能：提供高质量评估锚点
    - 核心思路：从完整数据集随机采样 1,071 个 QA，经 3 名专业标注员多轮审核与答案验证，最终得到 943 条高质量 QA。自动生成数据与人工标注的一致率达 89.13%，验证了数据管道质量
    - 设计动机：自动生成数据可能有噪声，人工标注子集提供可靠的评估标准

## 实验关键数据

### 主实验（TimE-Wiki，代表性模型）

| 模型 | Extract | Localiz. | Compute | OrderComp | ExplReas | Timeline | Counterfact |
|------|---------|---------|---------|-----------|----------|----------|-------------|
| Qwen2.5-72B-Inst | 81.70 | 83.84 | 41.37 | 84.22 | 70.13 | 4.08 | 50.68 |
| QwQ-32B | 74.99 | 67.75 | 49.59 | 93.53 | 60.61 | 25.38 | 53.13 |
| o3-mini | 96.67 | 80.83 | 49.17 | 93.33 | 82.24 | 33.33 | 52.07 |
| DeepSeek-R1 | 96.67 | 77.61 | 46.39 | 93.33 | 78.20 | 33.33 | 55.71 |

### 消融分析（TimE-Dial）

| 模型 | Extract | Localiz. | DurComp | OrdComp | Timeline |
|------|---------|---------|---------|---------|----------|
| Qwen2.5-14B-Inst | 38.85 | 30.83 | 42.00 | 47.78 | 0.00 |
| R1-Distill-14B | 40.40 | 18.34 | 53.33 | 72.22 | 0.22 |
| o3-mini | 41.41 | 45.30 | 56.67 | 86.67 | 10.00 |

### 关键发现

- **Timeline 任务是最大瓶颈**：即使最强模型 o3-mini 在 TimE-Wiki 上也只有 33.33%，小模型接近 0%。排列多事件的时间序需要同时处理信息检索和全局排序
- **test-time scaling 对逻辑推理有帮助，对检索任务效果不稳定**：R1-Distill 在 OrderCompare 上比对应非推理模型提升 24.44%，但在 Localization 上反而下降（过度思考导致的循环推理）
- **基础时间检索能力与高级推理任务显著正相关**：聚类分析显示 Extract 和 Localization 与几乎所有其他任务的相关系数 > 0.5
- **检索器选择对新闻场景影响巨大**：同一模型在不同检索器下性能差距可达 10%+
- **长对话场景的时间定位极难**：超过 15k token 的多轮对话 + 基于记忆的相对时间表达使定位准确率骤降

## 亮点与洞察

- **层级化评估框架设计精巧**：三级从认知由浅入深的任务设计，配合三种不同难点的数据源，能精确定位模型的时间推理短板。这种"能力分层 × 场景分层"的评估矩阵可以迁移到其他复杂推理能力的评估
- **Timeline 任务暴露了根本性缺陷**：所有模型在多事件排序上表现极差，说明当前 LLM 缺乏构建全局时间结构的能力，这指向了一个重要的研究方向
- **test-time scaling 的双面性**：推理模型的长 CoT 在逻辑推理上有帮助，但在简单检索上可能过度思考导致错误——这为合理使用 test-time compute 提供了实证依据

## 局限与展望

- QA 主要通过 LLM + 规则生成，可能存在系统性偏差（如偏好特定类型的时间表达）
- TimE-News 使用 RAG 检索会引入检索质量的混淆变量，难以纯粹评估时间推理能力
- 对话数据源来自合成长对话（LoCoMo, RealTalk），与真实多轮对话可能存在分布差异
- 未评估多语言时间推理能力
- Counterfactual 任务的设计较为基础（如"事件延后3年"），更复杂的反事实场景未涉及

## 相关工作与启发

- **vs TimeBench (Chu 2024)**: 聚合 10 个数据集但任务较简单，评估上下文不统一可能引入偏差；TimE 提供统一框架和更具挑战的任务
- **vs TRAM (Wang 2024)**: 专注事件序列理解但对当前模型挑战不足；TimE 的 Level-3 任务难度远高
- **vs TCELongBench (Zhang 2024)**: 仅聚焦新闻场景的时间理解；TimE 跨三种场景且层级更系统
- **vs TReMu**: 仅做对话中的时间定位和远程依赖；TimE-Dial 覆盖更全的时间推理子任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 三级渐进式 × 三种场景的评估矩阵设计有原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 24个模型 × 11个子任务 × 3个数据集的全面评估
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，数据构建流程完整
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 时间推理的根本性缺陷，为社区提供了重要评估工具

<!-- RELATED:START -->

## 相关论文

- [RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs](../../ICML2025/llm_reasoning/putnam-axiom_a_functional_and_static_benchmark_for_measuring_higher_level_mathem.md)
- [No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks](../../ICML2025/llm_reasoning/no_soundness_in_the_real_world_on_the_challenges_of_the_verification_of_deployed.md)
- [Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [笔记6：Self-Evaluating LLMs - 多步任务的步级置信度估计](value-guided_search_for_efficient_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
