---
title: >-
  [论文解读] Improving Code Localization with Repository Memory
description: >-
  [ICLR 2026][代码定位] 通过利用代码仓库的 commit 历史构建情景记忆（过去 commit）和语义记忆（活跃代码功能摘要），增强语言代理的代码定位能力，在 SWE-bench 上取得显著提升。
tags:
  - ICLR 2026
  - 代码定位
  - 仓库记忆
  - commit历史
  - 语言代理
  - SWE-bench
---

# Improving Code Localization with Repository Memory

**会议**: ICLR 2026  
**arXiv**: [2510.01003](https://arxiv.org/abs/2510.01003)  
**代码**: 无  
**领域**: Software Engineering / LLM Agent  
**关键词**: 代码定位, 仓库记忆, commit历史, 语言代理, SWE-bench

## 一句话总结

通过利用代码仓库的 commit 历史构建情景记忆（过去 commit）和语义记忆（活跃代码功能摘要），增强语言代理的代码定位能力，在 SWE-bench 上取得显著提升。

## 研究背景与动机

代码定位是仓库级软件工程任务（如 bug 修复）的关键第一步，即找到需要修改的文件和代码片段。现有方法，包括检索式（CodeRankEmbed）、过程式（Agentless）和代理式（LocAgent），都存在一个共同局限：它们将每个问题视为全新谜题，从零开始求解，完全没有利用对仓库的先验知识。

相比之下，人类开发者会随时间积累长期仓库记忆——包括对核心模块功能的理解、以及各种 bug 模式与修复位置之间的关联。这种经验积累使开发者能够成为代码库专家。论文通过分析 LocAgent 在 django 仓库上的失败案例来说明这一点：在缺乏仓库知识的情况下，代理需要进行复杂的推理链来追踪数据流和函数调用，最终容易过早停止或推理错误。而有经验的开发者可以利用往期 commit 记忆快速定位相关模块。

本文的核心洞察是：commit 历史是一个丰富但未被充分利用的资源，可以自然地为代理构建仓库记忆。

## 方法详解

### 整体框架

RepoMem 在现有的 LocAgent 框架上扩展了记忆工具，构建两种互补的记忆存储：情景记忆和语义记忆。这些记忆工具以模块化方式集成到 LocAgent 的 ReAct 循环中，与原有的代码导航工具协同工作。

### 关键设计

1. **情景记忆（Episodic Memory of Past Commits）**:

    - **构建方式**: 从仓库的近期 commit 历史（问题创建前的 7000 个 commit）中爬取和预处理结构化语料库，存储代码补丁、commit 消息、时间戳和关联的 issue 链接
    - **过滤机制**: 排除与测试实例有文本重叠的 issue 及其关联 commit，防止数据泄漏
    - **工具接口**:
        - `SearchCommit(query, top_k)`: 使用 BM25 检索与查询匹配的历史 commit，返回 commit SHA、消息和修改文件列表
        - `ExamineCommit(id)`: 根据 commit ID 获取完整上下文，包括 diff 补丁和关联 issue 内容
    - **设计动机**: 模拟开发者的情景记忆，提供具体的过去问题解决范例

2. **语义记忆（Semantic Memory of Active Code Functionality）**:

    - **构建方式**: 分析 commit 频率，识别最常修改的前 200 个文件（开发热点），使用 LLM 为每个文件生成高层功能摘要
    - **工具接口**:
        - `ViewSummary(file_name)`: 检索特定文件的缓存摘要
        - `SearchSummary(query, top_k)`: 对所有文件摘要进行关键词搜索，返回最相关的（文件, 摘要）对
    - **设计动机**: 提供代码库架构级别的上下文知识，引导代理关注最活跃和最相关的代码区域，避免在庞大代码库中迷失

3. **与 LocAgent 的集成**:

    - LocAgent 原有三个核心工具：SearchEntity（搜索代码实体）、TraverseGraph（多跳图遍历）、RetrieveEntity（获取源代码）
    - RepoMem 直接将记忆工具添加到动作空间，形成高层记忆引导与底层结构化代码分析的协同：先通过记忆形成假设，再用 LocAgent 工具进行详细验证

### 检索方法选择

实验比较了三种检索方法：
- BM25 + LLM 分词器（能识别代码实体名，如 `MigrationWriter`）：效果最佳
- BM25 + 空格分词：次之
- 稠密检索（GritLM-7B）：最差

原因在于代码仓库的词汇具有"刚性"特征——语义相近但功能不同的实体（如 `MigrationWriter` 和 `OperationWriter`）需要精确匹配而非语义匹配。

## 实验关键数据

### 主实验

所有实验使用 GPT-4o (2024-05-13) 作为骨干 LLM。

| 方法 | SWE-bench-verified Acc@1 | Acc@3 | Acc@5 | SWE-bench-live Acc@1 | Acc@3 | Acc@5 |
|------|----------|-------|-------|----------|-------|-------|
| CodeRankEmbed | 29.6 | 45.1 | 54.3 | 26.2 | 44.6 | 52.3 |
| Agentless | 53.3 | 67.8 | 71.4 | 40.0 | 60.0 | 62.3 |
| LocAgent | 64.8 | 70.4 | 71.6 | 59.2 | 60.8 | 63.1 |
| RepoMem (episodic-only) | 67.8 | 72.4 | 74.3 | 60.0 | 61.5 | 64.6 |
| RepoMem (semantic-only) | 65.0 | 71.0 | 72.8 | 56.9 | 61.5 | 63.9 |
| **RepoMem (full)** | **68.6** | **74.5** | **76.5** | **60.8** | **63.9** | **66.2** |

在 SWE-bench-verified 上 Acc@5 提升 4.9%，SWE-bench-live 上提升 3.1%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅情景记忆 | Acc@5=74.3 | 引用历史 commit 带来显著增益 |
| 仅语义记忆 | Acc@5=72.8 | 帮助聚焦活跃代码区域 |
| 两者结合 | Acc@5=76.5 | 互补信息，效果最优 |
| BM25 (LLM分词) | django Acc@5=79.7 | 优于稠密检索的65.8 |

### 关键发现

- **per-repo 分析**: commit 历史越丰富的仓库获益越大（如 sympy 提升 16.7%），历史稀疏的仓库反而可能降低性能（"others" 组降低 13.1%）
- **代理行为转变**: 引入记忆后，代理显著减少对 TraverseGraph 和 RetrieveEntity 的依赖，转向更有针对性的、假设驱动的探索策略
- **成本效率分析**: 平均成本增加，但在 per-example 层面表现为高方差——部分问题因记忆提示直接命中而显著便宜，部分问题因记忆无用而增加开销。额外成本主要集中在 LocAgent 本身失败的困难问题上，说明代理策略性地在困难问题上投入更多资源

## 亮点与洞察

1. **仓生记忆的天然优势**: commit 历史是仓库的自然演化记录，无需额外标注即可构建高质量记忆，这是一个优雅且实用的设计
2. **认知科学的类比**: 情景记忆对应开发者的"过去经验回忆"，语义记忆对应"模块功能理解"，两者协同反映了人类开发者的实际工作模式
3. **稀疏检索 > 稠密检索**: 在代码领域，精确关键词匹配优于语义匹配，这一发现对 RAG 在代码任务中的应用有重要启示
4. **模块化设计**: 记忆工具可以轻松集成到任何基于 ReAct 的代理框架中

## 局限与展望

1. **历史稀疏仓库表现差**: 当仓库 commit 历史有限时，记忆检索可能返回无关信息，反而干扰推理
2. **缺乏自适应记忆使用策略**: 代理目前无法判断何时应该使用记忆、何时应该从头推理，未来可以训练代理根据问题新颖性动态决策
3. **仅在文件级定位上验证**: 未展示在函数级或行级定位上的效果
4. **仅限于 bug 修复场景**: 对其他仓库级任务（如特性开发、重构）的适用性未探索

## 相关工作与启发

- **LocAgent** (Chen et al., 2025): 本文的基础框架，通过异构图表示代码结构和依赖关系
- **Agentless** (Xia et al., 2025): 过程式方法的代表，直接用 LLM 和仓库结构进行定位
- **SWE-Exp** (Chen et al., 2025): 从代理过去成功/失败轨迹中提炼过程性知识，与本文正交（本文从 commit 历史获取记忆）
- **Agent Workflow Memory** (Wang et al., 2025): 更广泛的记忆增强代理研究，本文专注于代码特定场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — commit 历史作为记忆源是自然但新颖的想法
- 实验充分度: ⭐⭐⭐⭐ — 两个基准、多角度分析、per-repo 分析、成本分析
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，案例分析生动，分析深入
- 价值: ⭐⭐⭐⭐ — 对代码代理的实际改进，方法简单有效且易于推广

<!-- RELATED:START -->

## 相关论文

- [A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](../../ACL2025/code_intelligence/feabench_repo_code_gen.md)
- [SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)

<!-- RELATED:END -->
