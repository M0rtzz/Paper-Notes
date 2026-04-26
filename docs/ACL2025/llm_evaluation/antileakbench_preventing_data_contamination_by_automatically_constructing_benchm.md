---
title: >-
  [论文解读] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge
description: >-
  [ACL 2025][数据污染] 提出 AntiLeakBench 自动化反泄露基准框架，通过追踪 Wikidata 知识更新历史识别 LLM 截止时间后的新知识，自动构建单跳/多跳 QA 测试样本（附真实 Wikipedia 支撑文档），确保知识级严格无污染，12 个 LLM 的大规模实验证实截止后性能普遍下降（EM 跌幅显著）验证了框架有效性。
tags:
  - ACL 2025
  - 数据污染
  - 基准构建
  - 知识更新
  - Wikidata
  - 自动化评测
---

# AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge

**会议**: ACL 2025  
**arXiv**: [2412.13670](https://arxiv.org/abs/2412.13670)  
**代码**: https://github.com/bobxwu/AntiLeakBench (有)  
**领域**: LLM评测 / 数据污染  
**关键词**: 数据污染, 基准构建, 知识更新, Wikidata, 自动化评测

## 一句话总结

提出 AntiLeakBench 自动化反泄露基准框架，通过追踪 Wikidata 知识更新历史识别 LLM 截止时间后的新知识，自动构建单跳/多跳 QA 测试样本（附真实 Wikipedia 支撑文档），确保知识级严格无污染，12 个 LLM 的大规模实验证实截止后性能普遍下降（EM 跌幅显著）验证了框架有效性。

## 研究背景与动机

**领域现状**：MMLU、GSM8K 等静态基准是 LLM 评估的基石，但其公开可访问性导致测试数据可能泄入训练集——数据污染问题。GSM8K 已被证实部分 LLM 存在过拟合。动态基准（如 LiveBench、RealTimeQA）通过收集新发布数据应对，但问题并未彻底解决。

**现有痛点**：(1) **污染免疫性弱**——"新发布"不等于"新知识"。LeetCode 编程题或考试题可能在发布前已有解法被 LLM 训练数据覆盖。(2) **人工维护成本高**——需要人类标注新收集的数据，导致更新频率低。RealTimeQA 和 KoLA 近期几乎停止更新。

**核心矛盾**：如何在"严格保证无污染"和"可持续低成本更新"之间取得平衡？

**本文目标** 构建知识级严格无污染的评测基准，同时实现完全自动化的零人工更新流程。

**切入角度**：不直接使用新发布数据，而是从知识库（Wikidata）中识别截止后真正更新的知识三元组，基于此构建 QA 样本。

**核心 idea**：用知识库的编辑历史追踪"新知识"（而非"新数据"），自动构建保证知识级无污染的评测样本。

## 方法详解

### 整体框架

四步自动化流程：(1) 准备 Wikidata 三元组数据，(2) 识别截止时间后更新的知识，(3) 从 Wikipedia 构建支撑文档，(4) 基于新知识构建无污染 QA 样本。全程无需人工介入。

### 关键设计

1. **知识数据准备（Data Preparation）**:

    - 功能：从 Wikidata 获取实体-关系-实体三元组及其时间限定符
    - 核心思路：提取物理实体相关关系（如 member of sports team），排除虚拟实体的关系（如坐标）。每个三元组附带 start_time 和 end_time 限定符，标示知识的有效时间段
    - 设计动机：Wikidata 提供结构化、时间标注的知识，是追踪知识变化的理想数据源

2. **新知识识别（Identifying Updated Knowledge）**:

    - 功能：找出 LLM 截止时间 $t_1$ 之后、当前时间 $t_2$ 之前发生变化的知识
    - 核心思路：按主体和关系分组所有三元组，按 start_time 时序排列。若某三元组的新值出现在 $t_1$ 之后（即 object 发生了变化），则标记为更新知识。例如：(Messi, member of sports team) 从 PSG → Inter Miami，发生在截止后
    - 关键细节：排除"变回原值"的情况（如球员回归老东家），确认新值与旧值确实不同
    - 设计动机：这是框架的核心洞察——只有知识本身在截止后更新，才能严格保证 LLM 训练集中不存在该知识

3. **支撑文档构建（Building Supporting Documents）**:

    - 功能：为每条新知识提供真实世界上下文
    - 核心思路：检索 Wikipedia 页面修订历史，找到新知识 start_time 之后的修订版本，提取包含主体和客体（或别名）的文章摘要作为支撑文档
    - 设计动机：不用 LLM 生成文档（避免幻觉），而用 Wikipedia 这一维护良好的真实数据源。且修订后文档同样是截止后产生，也不在训练集中

4. **构建无污染 QA 样本（Constructing Contamination-Free Samples）**:

    - 功能：基于新知识和支撑文档构建测试样本
    - 核心思路：支持四种任务格式——
        - **Single-Hop Gold**：直接提问新知识，上下文仅含支撑文档（如"Messi 属于哪支球队？"上下文为其 Wikipedia 页面）
        - **Single-Hop $N_d$**：增加 $N_d$ 个干扰文档，测试长上下文定位能力
        - **Multi-Hop Gold**：构建 $H$ 跳链式知识（$o_i = s_{i+1}$），如"Messi 所在球队的教练是谁？"，需要两步推理
        - **Multi-Hop $N_d$**：多跳 + 干扰文档
    - 问题格式：Generation（开放式生成）和 Multi-Choice（四选一：正确/Unknown/过时答案/噪声）
    - 设计动机：多种任务格式测试不同 LLM 能力维度——知识检索、长上下文理解、多跳推理

### 基准维护

只需下载最新 Wikidata dump 并执行自动化流程即可更新基准，支持多语言（利用 Wikidata/Wikipedia 的多语言特性），整个过程零人工成本。

## 实验关键数据

### 主实验（12 LLM × 8 任务设置，EM/F1）

| 模型 | Single-Hop Gold EM | Multi-Hop Gold EM | 平均 EM | 平均 F1 |
|------|-------------------|-------------------|---------|---------|
| Gemma-2-9B | **85.0** | 57.7 | — | — |
| Mistral-Nemo-12B | 82.7 | **57.7** | **53.9** | **62.0** |
| LongChat-v1.5-7B | 75.5 | 38.8 | 36.4 | 48.9 |
| GPT-4o | 高 | 高 | 最佳之一 | 最佳之一 |
| Llama-2-7B | 40.6 | 33.6 | 19.9 | 36.7 |

### 消融实验（截止前后性能对比）

| 观察 | 说明 |
|------|------|
| 截止后 EM 普遍下降 | 所有 LLM 在截止后样本上性能明显低于截止前，直接证实数据污染存在 |
| 过时答案选择率升高 | Multi-Choice 中过时答案（截止前正确但截止后已更新）被选中比例显著上升 |
| 干扰文档数增加致性能下降 | 从 $N_d=3$ 到 $N_d=7$，EM 持续下降，反映长上下文检索困难 |

### 关键发现

- 数据质量验证：人工评估显示上下文准确率 97.3%（单跳）/98.7%（多跳），答案准确率 96.7%/97.3%
- 截止前后的性能对比是框架有效性的最直接证据——如果基准真正无污染，则截止后性能应下降（因为 LLM 没见过该知识）
- 多跳任务难度显著高于单跳，且干扰文档进一步加大难度

## 亮点与洞察

- "新知识"而非"新数据"的概念区分是核心贡献——LiveBench 从 LeetCode 收集新题但题目知识可能旧，AntiLeakBench 确保底层知识本身是截止后才产生的
- 全自动零人工的构建流程使其具备可持续维护性——只需定期下载 Wikidata dump 即可为新 LLM 生成专属基准
- Multi-Choice 中设计 outdated option（截止前正确答案）巧妙检测 LLM 是否使用过时知识回答

## 局限与展望

- 仅覆盖知识型 QA 任务——推理/数学/代码生成等任务类型无法用此框架评估
- 依赖 Wikidata/Wikipedia 的更新频率和覆盖面，某些领域知识更新可能滞后
- LLM 的知识截止时间本身可能不完全准确——模型可能通过其他渠道接触到截止后知识
- 自动生成的问题模板化较强，多样性可能低于人工编写
- 多语言支持虽然声明但实验中主要在英语上验证

## 相关工作与启发

- **vs LiveBench**: 收集新发布数据但知识可能旧；AntiLeakBench 确保知识本身是新的
- **vs RealTimeQA**: 需人工维护已很少更新；AntiLeakBench 全自动零成本
- **vs ADU (Ying et al. 2024)**: 用 LLM 改写现有基准引入偏差风险；AntiLeakBench 基于真实世界知识更新

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "新知识"概念和全自动知识追踪构建流程是新颖且重要的贡献
- 实验充分度: ⭐⭐⭐⭐ 12个LLM×8种任务设置，截止前后对比有力，人工质量验证充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，但部分公式符号较密集
- 价值: ⭐⭐⭐⭐⭐ 对LLM评测基础设施有重要贡献，全自动可持续更新是实用价值核心

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] TripTailor: A Real-World Benchmark for Personalized Travel Planning](triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)
- [\[ICML 2025\] How Much Can We Forget about Data Contamination?](../../ICML2025/llm_evaluation/how_much_can_we_forget_about_data_contamination.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_realworld_display_inverse_rendering_dataset.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](rulearena_rule_guided_reasoning.md)

<!-- RELATED:END -->
