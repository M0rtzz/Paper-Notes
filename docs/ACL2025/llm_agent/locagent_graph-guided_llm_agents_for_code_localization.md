---
title: >-
  [论文解读] LocAgent: Graph-Guided LLM Agents for Code Localization
description: >-
  [LLM Agent][代码定位] LocAgent 将代码库解析为有向异构图（含 contain/import/invoke/inherit 四种关系），并设计统一工具（SearchEntity/TraverseGraph/RetrieveEntity）引导 LLM Agent 进行多跳推理…
tags:
  - "LLM Agent"
  - "代码定位"
  - "图表示"
  - "软件维护"
  - "代码搜索"
---

# LocAgent: Graph-Guided LLM Agents for Code Localization

| 会议 | 领域 | arXiv | 代码 |
|------|------|-------|------|
| ACL2025 | LLM Agent / Software Engineering | [2503.09089](https://arxiv.org/abs/2503.09089) | [GitHub](https://github.com/gersteinlab/LocAgent) |

**关键词**: 代码定位, 图表示, LLM Agent, 软件维护, 代码搜索

## 一句话总结

LocAgent 将代码库解析为有向异构图（含 contain/import/invoke/inherit 四种关系），并设计统一工具（SearchEntity/TraverseGraph/RetrieveEntity）引导 LLM Agent 进行多跳推理，实现高精度代码定位，在文件级达到 92.7% 准确率，同时通过微调开源模型将成本降低 86%。

## 研究背景与动机

代码定位（Code Localization）是软件维护中的基础任务，目标是根据自然语言问题描述（如 GitHub Issue）在代码库中精确找到需要修改的代码片段。开发者花费高达 66% 的调试时间在理解代码上，自动化工具也面临同样挑战。

现有方法面临三方面限制：

**密集检索方法**（如向量嵌入）需要维护和持续更新整个代码库的向量表示，对于大型、快速演化的仓库工程成本很高

**大上下文窗口 LLM** 无法一次性处理整个代码库，需要战略性地导航到相关部分

**现有 Agent 方法**主要通过目录遍历导航，无法理解语义关系，难以在跨文件依赖较多时进行多跳推理

核心问题：Issue 描述往往只提到症状而非根因。例如"用户画像中的 XSS 漏洞"可能需要修改一个共享的验证工具函数，该函数并未在 Issue 中被提及。这种 Issue 描述与实际代码的隐式关联是传统检索方法难以处理的。

## 方法详解

### 整体框架

LocAgent 由三个核心部分组成：

1. **图构建与索引**（离线）：将代码库解析为有向异构图并构建稀疏索引
2. **Agent 引导搜索**（在线）：Agent 使用统一工具在图上进行自主探索和定位
3. **开源模型微调**：通过轨迹蒸馏降低成本

### 图构建（Graph-based Code Representation）

构建有向异构图 $\mathcal{G}(\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R})$：

- **节点类型** $\mathcal{A}$：directory（目录）、file（文件）、class（类）、function（函数）
- **边类型** $\mathcal{R}$：contain（包含）、import（导入）、invoke（调用）、inherit（继承）

构建过程：
1. 所有目录和 Python 文件作为节点
2. 使用 AST 递归解析每个文件，提取内部函数和类作为节点
3. 函数级别为最小节点粒度，函数代码内容作为检索文档
4. contain 边形成树结构，import/invoke/inherit 捕获跨文件依赖

### 稀疏层级实体索引

为图节点构建四级层级索引：
1. **实体 ID 索引**：使用全限定名（如 `src/utils.py:MathUtils.calculate_sum`）
2. **全局名称字典**：实体名称到所有同名节点的映射
3. **实体 ID 的 BM25 倒排索引**：处理模糊匹配
4. **代码块倒排索引**：覆盖所有可能的匹配情况（如全局变量）

### 三个统一工具

| 工具名 | 输入 | 输出 |
|--------|------|------|
| SearchEntity | 关键词 | 相关实体及代码片段（三级详细度：fold/preview/full） |
| TraverseGraph | 起始实体 ID、方向、跳数、实体类型、关系类型 | 遍历的子图（实体和关系） |
| RetrieveEntity | 实体 ID | 完整代码、文件路径、行号 |

**TraverseGraph** 的关键设计：
- 支持类型感知的 BFS 搜索
- Agent 可选择实体类型和关系类型，相当于生成异构图的 meta-path
- 输出采用扩展树格式，通过空间距离编码拓扑关系

### Chain-of-Thought Agent 规划

Agent 按步骤执行：
1. **关键词提取**：从 Issue 中提取不同类别的关键词
2. **关键词链接**：通过 SearchEntity 链接到代码实体
3. **逻辑流生成**：识别入口点，迭代使用 TraverseGraph 和 RetrieveEntity 追踪调用链
4. **定位目标实体**：基于逻辑流定位所有可疑代码实体并排序

### 一致性置信度估计

使用 Reciprocal Rank 作为初始置信度分数，通过多次迭代聚合得到最终置信度，一致性高的位置更可能相关。

### 开源模型微调

- 收集 433 条 Claude-3.5 成功轨迹 + 335 条微调后 Qwen2.5-32B 的成功轨迹
- 使用 LoRA 进行 SFT
- 自改进循环：用微调模型生成新轨迹，筛选成功的用于进一步训练
- 蒸馏到 7B 模型

## 实验

### Loc-Bench 新基准

针对 SWE-Bench 的局限（数据泄露风险、类别不均衡——85% 为 bug 报告），提出 Loc-Bench：
- 560 个样例，覆盖 Bug Report（242）、Feature Request（150）、Security Issue（29）、Performance Issue（139）
- 收集 2024 年 10 月之后的 GitHub Issue，减少预训练数据泄露

### 主实验结果（SWE-Bench-Lite）

| 方法 | 文件 Acc@5 | 模块 Acc@10 | 函数 Acc@10 |
|------|-----------|------------|------------|
| BM25 | 61.68 | 52.92 | 36.86 |
| CodeRankEmbed | 84.67 | 78.83 | 58.76 |
| Agentless (Claude-3.5) | 79.56 | 68.98 | 58.76 |
| OpenHands (Claude-3.5) | 90.15 | 83.58 | 70.07 |
| SWE-agent (Claude-3.5) | 90.15 | 78.10 | 64.60 |
| **LocAgent (Qwen2.5-32B-ft)** | **92.70** | **87.23** | **77.01** |
| **LocAgent (Claude-3.5)** | **94.16** | **87.59** | **77.37** |

LocAgent 在所有级别上全面领先。

### 效率分析

| 方法 | 模型 | 每例成本 | Acc@10/Cost |
|------|------|---------|-------------|
| OpenHands | Claude-3.5 | $0.79 | 0.9 |
| LocAgent | Qwen2.5-7B(ft) | **$0.05** | **13.2** |
| LocAgent | Qwen2.5-32B(ft) | $0.09 | 8.6 |
| LocAgent | Claude-3.5 | $0.66 | 1.2 |

微调模型实现了 86% 的成本降低。

### 消融实验

- 移除 TraverseGraph：函数级准确率从 71.53% 降到 66.06%
- 只保留 contain 关系：66.42%，说明 import/invoke/inherit 至关重要
- 限制跳数为 1：66.79%，多跳探索对深层理解必要
- 移除 SearchEntity：降至 53.28%，稀疏索引是核心贡献

### 下游任务影响

更好的定位直接提升 GitHub Issue 解决率：
- Agentless (Claude-3.5) Pass@10: 33.58%
- LocAgent (Claude-3.5) Pass@10: **37.59%**（+12%）

## 亮点与洞察

1. **图表示的核心价值**：通过 import/invoke/inherit 关系捕获跨文件依赖，使物理距离远但逻辑上紧密的模块在图中变得"近邻"，这是传统目录遍历无法做到的
2. **工具设计哲学**：将所有操作统一为三个工具而非大量碎片化工具，更适合 LLM Agent 使用
3. **索引轻量高效**：索引过程仅需数秒，且不需要维护向量数据库，实用性强
4. **成本 vs 性能的优秀平衡**：微调 7B 模型仅 $0.05/例，性能却超越大多数使用 GPT-4o 的方法

## 局限性

1. 仅聚焦 Python 代码库，未验证其他编程语言
2. 微调依赖 Claude-3.5 的成功轨迹，数据来源有限
3. 评估指标以准确率为主，缺乏对定位质量的细粒度衡量
4. Loc-Bench 虽涵盖多类别，但安全相关样本仍偏少（29 个）

## 相关工作

- **传统检索方法**：BM25、E5、Jina-Code 等，但维护成本高且缺乏结构理解
- **LLM 生成式方法**：Agentless 的层级定位、SWE-Agent 的 ACI 接口
- **图方法**：CodexGraph（Neo4j+Cypher）、RepoGraph（子图检索）、RepoUnderstander（MCTS 搜索）、OrcaLoca（优先级调度）。LocAgent 的图表示最完整，同时兼顾四种节点和四种关系

## 评分

⭐⭐⭐⭐（4/5）

本文解决了代码定位中的核心痛点——跨文件依赖追踪，图表示和统一工具设计简洁有效。微调开源模型降低成本的思路实用。遗憾的是仅限 Python，且 Loc-Bench 的安全和性能类别样本偏少。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](../../NeurIPS2025/llm_agent/the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)
- [\[ACL 2025\] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation](gear_graph-enhanced_agent_for_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](../../NeurIPS2025/llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](../../ACL2026/llm_agent/codestruct_code_agents_over_structured_action_spaces.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)

</div>

<!-- RELATED:END -->
