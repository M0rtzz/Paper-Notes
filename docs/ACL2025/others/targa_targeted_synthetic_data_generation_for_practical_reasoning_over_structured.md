---
title: >-
  [论文解读] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data
description: >-
  [ACL 2025][语义解析] TARGA 提出了一种针对性的合成数据生成框架，无需任何人工标注即可为知识库问答（KBQA）动态生成高相关性的合成示例用于上下文学习，仅用 7B 模型即在 GrailQA（+7.7 F1）和 KBQA-Agent（+12.2 F1）上大幅超越所有非微调方法。
tags:
  - ACL 2025
  - 语义解析
  - 知识库问答
  - 合成数据生成
  - 上下文学习
  - 结构化推理
---

# TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data

**会议**: ACL 2025  
**arXiv**: [2412.19544](https://arxiv.org/abs/2412.19544)  
**代码**: https://github.com/cdhx/TARGA  
**领域**: NLP理解  
**关键词**: 语义解析, 知识库问答, 合成数据生成, 上下文学习, 结构化推理

## 一句话总结

TARGA 提出了一种针对性的合成数据生成框架，无需任何人工标注即可为知识库问答（KBQA）动态生成高相关性的合成示例用于上下文学习，仅用 7B 模型即在 GrailQA（+7.7 F1）和 KBQA-Agent（+12.2 F1）上大幅超越所有非微调方法。

## 研究背景与动机

1. **领域现状**：语义解析（将自然语言问题转换为逻辑形式）是结构化环境中推理的主流方法。现有方法要么需要大规模人工标注数据集进行微调（如 Pangu），要么通过检索标注数据集中的相似示例进行上下文学习（如 KB-Binder、KB-Coder）。

2. **现有痛点**：两大核心挑战——(1) **标注依赖**：在特定环境中收集人工标注的数据耗时费力，现实场景中通常没有预标注数据；(2) **泛化能力有限**：基于静态训练集的方法在遇到未见过的 KB 元素或查询结构（non-I.I.D. 设定）时性能急剧下降，降幅高达 20%+。

3. **核心矛盾**：像 Freebase 这样有 30 亿+三元组的复杂环境，不可能通过预收集的静态数据集覆盖所有可能的查询模式。增加标注覆盖范围又会导致训练和检索成本指数增长。

4. **本文目标** 如何在完全不需要人工标注的情况下，为每个测试问题动态生成高度相关的示例，且在小模型上就能达到优秀性能？

5. **切入角度**：从测试问题相关的 KB 实体和关系出发，通过逐层扩展和跨层组合**在知识库上直接构造有效的查询图**，然后将这些查询转换为自然语言问题作为上下文学习的示例。

6. **核心 idea**：不从静态数据集检索示例，而是从测试问题出发在 KB 上**动态合成针对性的查询-问题对**作为上下文学习的 demonstrations。

## 方法详解

### 整体框架

给定一个自然语言问题 $nlq$，TARGA 分四步：（1）检索候选 KB 实体和关系；（2）通过逐层扩展（multi-hop）和跨层组合（multi-constraint）构造合成查询图；（3）用 bge-reranker 对合成查询排序筛选最相关的示例；（4）将排序后的合成（NLQ, Query）对作为 demonstrations 进行上下文学习生成目标查询。

### 关键设计

1. **逐层扩展（Layer-wise Expansion）**:

    - 功能：构建 multi-hop 链式查询结构（查询图的深度）
    - 核心思路：从最简单的单跳结构 $\mathcal{L}_1 = \{(s,p,o) | s \in E_{nlq}, p \in R_{nlq}, \text{Exec}((s,p,o), \mathcal{G}) \neq \emptyset\}$ 开始，逐层向外扩展——将终端变量节点通过新关系连接到新变量，构造 $\mathcal{L}_2, \mathcal{L}_3$。扩展停止条件为达到复杂度阈值（3 跳），因为合理问题的实体-答案距离通常不超过 3 跳。关键约束：只扩展**执行结果非空**的查询，避免无效组合爆炸
    - 设计动机：从简单到复杂的探索策略既保证覆盖率又控制计算开销；执行验证自动实现了实体-关系联合消歧

2. **跨层组合（Cross-layer Combination）**:

    - 功能：构建 multi-constraint 查询结构（查询图的宽度）
    - 核心思路：选择两个不同层级的查询 $q \in \mathcal{L}_x$ 和 $q' \in \mathcal{L}_y$，找到它们各自变量中执行结果有交集的变量（$\mathcal{E}(o_i) \cap \mathcal{E}(o_j) \neq \emptyset$），将两个查询通过此共享变量合并。组合终止于 5 条边，覆盖当前数据集中绝大多数问题结构。从简单组合开始（$\mathcal{L}_{1\times1}$），逐步扩展到更复杂的模式（$\mathcal{L}_{2\times3}$、$\mathcal{L}_{1\times(1\times2)}$）
    - 设计动机：multi-constraint 是 KBQA 中的重要查询类型（如"出生在X且就读于Y的人"），仅靠链式扩展无法建模

3. **分层排序 + 查询文本化（Re-ranking）**:

    - 功能：从可能的大量合成查询中筛选最相关的 demonstrations
    - 核心思路：先用 **Query Textification** 将 SPARQL 查询通过启发式规则转换为接近自然语言的文本（弥合 embedding 模型和查询之间的语义鸿沟），再用 bge-reranker-v2-m3 计算与问题的相似度。采用**分层排序**（Hierarchical Ranking）——对同一父查询衍生的所有子查询只保留 top-n，防止复杂查询指数增长导致的不平衡
    - 设计动机：合成查询中很多结构虽然有效但与当前问题无关，需要高效筛选；分层排序确保不同复杂度级别的查询都有机会被选中

### 损失函数 / 训练策略

TARGA 是一个零标注、无训练的框架，核心组件均在推理时动态运行：
- 实体链接：直接复用 Gu et al. (2023) 的结果
- 关系检索：用 text-embedding-ada-002 计算问题与 Freebase 关系的相似度，取 top-20
- 示例数量：默认 10 个 demonstrations
- 基座模型：默认 Qwen-2.5-7B-Instruct

## 实验关键数据

### 主实验

| 数据集 | 指标(F1) | TARGA (7B) | 之前 SOTA (非微调) | 提升 |
|--------|----------|------------|-------------------|------|
| GrailQA | F1 | 69.0 | 61.3 (KB-Coder-R, GPT-3.5) | +7.7 |
| GraphQ | F1 | 50.6 | 50.8 (QueryAgent, GPT-3.5) | -0.2 |
| KBQA-Agent | F1 | 46.5 | 34.3 (FUXI, GPT-3.5) | +12.2 |
| MetaQA | F1 | 85.7 | 99.5 (KB-Binder-R) | -13.8 |

**更强基座模型**：

| 模型 | GrailQA | GraphQ | KBQA-Agent |
|------|---------|--------|------------|
| Qwen-2.5-7B | 69.0 | 50.6 | 46.5 |
| Qwen-2.5-72B | 70.6 | 54.1 | 57.3 |
| GPT-3.5-turbo | 68.9 | 51.0 | 52.7 |
| GPT-4-turbo | 69.8 | 52.5 | 51.4 |

### 消融实验

**泛化能力（GrailQA 三种设定）**：

| 方法 | I.I.D. | Compositional | Zero-shot | Avg |
|------|--------|--------------|-----------|-----|
| KB-Binder-R | 80.6 | 53.6 | 50.7 | 58.5 |
| KB-Coder-R | 81.0 | 57.8 | 54.1 | 61.3 |
| TARGA | 68.4 | 62.2 | 71.7 | 69.0 |
| TARGA-R | 80.8 | 63.6 | 71.6 | 71.9 |

**效率分析（GrailQA）**：

| 方法 | TPQ(秒) | QPQ(查询次数) | CPQ(费用$) |
|------|---------|-------------|-----------|
| KB-BINDER | 51.2 | 3,297.7 | 0.010 |
| QueryAgent | 16.6 | 5.2 | 0.019 |
| TARGA | 4.5 | 256.8 | 0.000 |

### 关键发现

- **样本效率极高**：仅 1 个合成 demonstration 就超越了 20 shot 的随机/相似度检索设定，表明合成数据质量远超静态数据集检索
- **泛化能力碾压**：在 Zero-shot 设定下（完全未见的 KB 元素/结构），TARGA 达到 71.7 F1，比 KB-Coder-R 的 54.1 高出 17.6 点。预收集训练集对 Zero-shot 几乎无帮助
- **鲁棒性强**：在对抗设定下（随机替换 demonstrations 中的关系），TARGA 性能仅下降约 25%，而 random 设定下降 75%，证明合成示例间的信息冗余提供了鲁棒性
- **不依赖强大闭源模型**：7B 开源模型即超越所有基于 GPT-3.5-turbo 的非微调方法，这得益于高质量示例降低了任务难度
- **响应速度最快**：4.5 秒/问题，比 KB-BINDER 快 11×，且闭源 LLM 调用成本为零

## 亮点与洞察

- **"动态合成 vs 静态检索"范式转变**：不从固定数据集检索相似例子，而是根据当前问题在 KB 上实时构造针对性示例——这避免了泛化问题的根源（训练集分布无法覆盖所有测试情况）
- **查询构造巧妙利用 KB 结构约束**：通过执行非空验证自动进行实体-关系联合消歧，将组合爆炸控制在每个问题平均只有几十个有效候选查询
- **Query Textification 弥合语义鸿沟**：将 SPARQL 查询转为自然语言的简单启发式规则让 embedding 排序更准确，设计简单却很有效
- **简单到复杂的构造策略**：从 $\mathcal{L}_1$ 到 $\mathcal{L}_3$，从单层到跨层组合，保证了复杂查询一定是由验证过的简单子查询组装而成

## 局限与展望

- MetaQA 上效果不如需要全训练集的 KB-Binder-R（85.7 vs 99.5），说明在简单环境+充足标注的场景下动态合成不一定有优势
- 实体链接直接复用已有方法，如果链接质量差会影响后续查询构造
- 查询构造过程中的 KB 查询次数仍较多（256.8 QPQ），在大规模 KB 上可能有性能瓶颈
- 目前限定于 KBQA（Freebase），未验证在 Text2SQL 等其他结构化推理任务上的效果（虽然论文中提到了 WikiSQL 的初步结果）
- 分层排序的 top-n 参数和最大复杂度阈值需要手动设定

## 相关工作与启发

- **vs KB-Binder/KB-Coder**: 二者从标注训练集检索示例做 ICL，在 non-I.I.D. 下性能暴跌 20%+；TARGA 动态合成示例完全不受训练集分布限制。但在 I.I.D. 设定下加入训练集效果（TARGA-R）可与之匹敌
- **vs BYOKG (Agarwal et al.)**: BYOKG 也不需标注，但离线合成 + 静态检索，仍有泛化问题；TARGA 在线合成，每个问题获得定制化示例
- **vs Agent-based (QueryAgent, AgentBench)**: Agent 方法逐步分解问题，泛化性好但计算成本高（多次 LLM 调用）且严重依赖强模型能力；TARGA 用 7B 模型一次 ICL 调用即完成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 动态 targeted 合成示例的思路非常巧妙，将 KB 结构约束转化为数据质量保证
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集 + 泛化性/鲁棒性/效率/模型大小等详细分析
- 写作质量: ⭐⭐⭐⭐ 方法描述层次分明，公式清晰
- 价值: ⭐⭐⭐⭐⭐ 零标注+小模型就超越闭源大模型方法，非常practical，可直接用于实际KBQA系统

<!-- RELATED:START -->

## 相关论文

- [Theorem Prover as a Judge for Synthetic Data Generation](theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)
- [PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)

<!-- RELATED:END -->
