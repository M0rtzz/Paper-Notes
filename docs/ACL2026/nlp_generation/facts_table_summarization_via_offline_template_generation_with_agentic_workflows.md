---
title: >-
  [论文解读] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows
description: >-
  [ACL 2026][文本生成][表格摘要] 本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板）…
tags:
  - "ACL 2026"
  - "文本生成"
  - "表格摘要"
  - "离线模板"
  - "Agentic工作流"
  - "SQL生成"
  - "隐私合规"
---

# FACTS: Table Summarization via Offline Template Generation with Agentic Workflows

**会议**: ACL 2026  
**arXiv**: [2510.13920](https://arxiv.org/abs/2510.13920)  
**代码**: [GitHub](https://github.com/BorealisAI/FACTS)  
**领域**: 数据分析 / 表格理解  
**关键词**: 表格摘要, 离线模板, Agentic工作流, SQL生成, 隐私合规

## 一句话总结

本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板），实现快速、准确、隐私合规的查询聚焦表格摘要，在 FeTaQA、QTSumm 和 QFMTS 三个基准上全面超越基线。

## 研究背景与动机

**领域现状**：查询聚焦表格摘要（query-focused table summarization）要求根据用户查询从表格数据生成自然语言摘要，不同于简单的表格问答（返回短答案）和通用表格摘要（捕捉所有重要内容）。在金融、医疗、法律等领域，专业人员依赖定制化摘要做决策。

**现有痛点**：(1) 表格到文本模型（如 TAPEX、ReasTAP）需要昂贵的微调，且在数值推理和逻辑忠实度上表现不佳；(2) 基于提示的方法（如 DirectSumm）直接查询 LLM，受 token 限制，暴露敏感数据，且需为每张新表重新生成；(3) 现有 Agentic 框架（如 Binder、Dater）依赖分解规划或手工模板，缺乏鲁棒性和可扩展性。

**核心矛盾**：实用方案必须同时满足四个属性——快速（可复用）、准确（基于执行而非自由生成）、可扩展（不需传递所有行）、隐私合规（不暴露原始数据给 LLM）——但现有方法无一满足全部。

**本文目标**：设计首个自动化离线模板生成的 Agentic 框架，一次生成、多次复用，满足所有四个属性。

**切入角度**：将表格摘要分解为 SQL 查询（提取精确值）+ Jinja2 模板（渲染自然语言），形成可独立于数据值的离线模板。

**核心 idea**：离线模板绑定于表格 schema 和查询语义而非具体数据值——一旦生成，可直接应用于任何共享相同 schema 的新表格，避免重复 LLM 推理。

## 方法详解

### 整体框架

FACTS 由三个互联阶段组成，每个阶段的输出由 LLM Council（多模型集成验证）迭代验证和改进。最终产出为离线模板——SQL 查询集 + Jinja2 渲染模板。LLM 全程仅接触 schema 信息，从不暴露原始数据。

### 关键设计

1. **Schema-Guided Specification and Filtering（模式引导规范和过滤）**:

    - 功能：明确用户查询意图并生成过滤规则
    - 核心思路：给定用户查询和表格 schema，Agent 生成两类输出：(a) 引导问题（guided questions）——识别哪些列、关系和操作相关；(b) 过滤规则（filtering rules）——指定需排除的行或类别值。LLM 不接触原始数据，仅基于 schema 提出抽象过滤规则（如"exclude rows where category='expense'"），后续转化为 SQL WHERE 子句
    - 设计动机：用户查询通常是高层次的自然语言，需要先"翻译"为 schema 级别的具体操作规范

2. **SQL Queries Generation（SQL 查询生成）**:

    - 功能：生成可执行的 SQL 查询以精确提取数据
    - 核心思路：基于 Stage 1 的规范，Agent 生成候选 SQL 查询，将过滤规则转化为约束条件。每条查询在本地数据库上执行验证——若失败或返回空结果，错误信息传递给 LLM Council 反馈，Agent 迭代修正直到可执行。最大 patience 设为 3 轮
    - 设计动机：将摘要建立在可执行程序而非自由文本生成之上，从根本上消除幻觉

3. **Jinja2 Template Generation and Alignment（Jinja2 模板生成与对齐）**:

    - 功能：将 SQL 结果渲染为自然语言摘要
    - 核心思路：Agent 生成 Jinja2 模板，要求引用精确列名、正确迭代返回行、优雅处理空结果。LLM Council 检查 SQL 输出与模板引用的对齐——若存在字段缺失或形状不兼容，SQL 和模板协同修正
    - 设计动机：分离数据提取（SQL）和文本渲染（Jinja2），使两者都可独立验证和复用

### 损失函数 / 训练策略

FACTS 为无训练方法。主 Agent 使用 GPT-4o-mini 作为骨干。LLM Council 由 GPT-4o-mini、Claude-4 Sonnet 和 DeepSeek v3 组成，通过多数投票决定接受/拒绝，聚合反馈指导改进。每样本平均 2.47 个引导问题/过滤规则、1.36 轮 SQL 修正、1.84 轮模板修正。

## 实验关键数据

### 主实验

| 方法 | FeTaQA BLEU/RL/MET | QTSumm BLEU/RL/MET | QFMTS BLEU/RL/MET |
|------|---------------------|---------------------|---------------------|
| CoT | 28.2/51.0/56.9 | 19.3/39.0/47.2 | 31.5/54.3/58.1 |
| DirectSumm | 29.8/51.7/58.2 | 20.7/40.2/50.3 | 33.6/57.0/62.8 |
| SPaGe | 33.8/55.7/62.3 | 20.9/41.3/47.7 | 45.7/68.3/73.4 |
| FACTS (GPT-Only) | 30.8/55.7/66.0 | 20.1/43.1/50.5 | 45.4/70.5/73.2 |
| **FACTS** | **32.6/58.9/67.7** | **21.9/45.8/51.3** | **46.0/70.8/73.2** |

### 消融实验

| 评估维度 | FACTS 得分 |
|----------|-----------|
| 意图匹配 | 97% |
| SQL 执行准确率 | 94% |
| 模板渲染准确率 | 98% |
| Council 共识错误率 | ~3% |
| 整体事实正确率 | ~92% |

### 关键发现

- FACTS 在所有三个数据集上均达到最佳或次佳结果，尤其在 ROUGE-L 和 METEOR 上优势明显
- 人类偏好研究：FACTS vs SPaGe——55% 偏好 FACTS 的完整性，59% 偏好正确性，60% 偏好减少幻觉
- 复用性测试：100 张同 schema 表时，FACTS 因模板复用大幅加速（仅需 SQL 执行 + Jinja2 渲染）
- GPT-Only 变体仍超越大多数基线，证明核心工作流本身有效，Council 多样性进一步增强
- 每样本平均消耗 9,922 输入 token 和 1,045 输出 token，计算成本可控

## 亮点与洞察

- "离线模板"概念是工程上的优雅创新——将一次性的 LLM 推理成本摊薄到无限次复用中，特别适合企业场景（如每年重复的财务报告摘要）
- LLM Council 的多数投票 + 聚合反馈机制提供了轻量级的自我修正能力，~3% 的共识错误率表明多模型集成有效
- 隐私合规设计是该方法的核心优势——LLM 仅接触 schema，原始数据值完全留在本地 SQL 引擎中
- SQL + Jinja2 的组合将"正确性"和"可读性"解耦——前者由程序执行保证，后者由模板渲染实现

## 局限与展望

- 假设模板在相同 schema 下完全复用，未考虑 schema 漂移或列重命名
- 对复杂的多表 JOIN 和嵌套查询可能需要更多修正轮次
- SQL 执行准确率 94% 意味着仍有 6% 的错误——对高风险决策可能不够
- Jinja2 模板的自然语言表达可能在不同语言/文化背景下需要调整

## 相关工作与启发

- **vs DirectSumm**: 后者一次性将全表+查询传给 LLM，暴露数据且不可复用；FACTS 通过离线模板解决了两个问题
- **vs SPaGe**: SPaGe 使用图结构化规划提高可靠性，但其规划仅部分可复用；FACTS 的离线模板完全可复用
- **vs Binder/Dater**: 这些方法将查询转为可执行程序但缺乏模板化和复用能力；FACTS 增加了 Jinja2 渲染层实现自然语言输出

## 评分

- 新颖性: ⭐⭐⭐⭐ 离线模板生成概念新颖且实用，但各组件（SQL 生成、Jinja2、LLM Council）有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、自动+人类评估、复用性/可扩展性分析、消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，四个属性对比表直观，示例具体
- 价值: ⭐⭐⭐⭐⭐ 高度实用——隐私合规+可复用的设计直接解决企业部署痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](../../ACL2025/nlp_generation/theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)
- [\[ACL 2026\] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)

</div>

<!-- RELATED:END -->
