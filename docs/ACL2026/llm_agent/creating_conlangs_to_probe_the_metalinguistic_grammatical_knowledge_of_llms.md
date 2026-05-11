---
title: >-
  [论文解读] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs
description: >-
  [ACL2026][LLM Agent] 构建人造语言系统IASC来探测LLM的元语言学语法知识，发现LLM对常见语言类型模式处理远优于罕见模式
tags: [人造语言, 元语言知识, 形态句法, LLM评估, 语言类型学]
---

# Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs

**会议**: ACL 2026
**arXiv**: [2510.07591](https://arxiv.org/abs/2510.07591)
**代码**: [https://github.com/SakanaAI/IASC](https://github.com/SakanaAI/IASC)
**领域**: LLM Agent
**关键词**: 人造语言, 元语言知识, 形态句法变换, LLM语言能力探测, 语言类型学

## 一句话总结

本文提出 IASC（Interactive Agentic System for ConLangs），一个模块化的人造语言构建系统，通过让 LLM 按语言学规格执行形态句法变换来探测其元语言知识，发现 LLM 处理常见语言类型模式远优于罕见模式，且不同 LLM 之间能力差异悬殊。

## 研究背景与动机

**领域现状**：大量研究关注 LLM 的语言能力，包括翻译、句法标注等，但这些任务评估的是 LLM 对特定语言的知识，而非对语言学概念本身的理解。LLM 是否真正"理解"抽象的语言学概念（如词序、格标记、一致性等），而不只是记住了训练数据中特定语言的模式？

**现有痛点**：(1) 现有 LLM 语言能力评估多集中于百科知识式的测试（知道某种语言的某个事实），缺少对元语言学推理能力的系统探测；(2) 自然语言测试容易受训练数据泄露影响，LLM 可能只是"记住"了答案而非真正理解规则。

**核心矛盾**：LLM 在训练中接触到大量语言学文献和多语言数据，但这并不意味着它能按照给定的抽象语法规则来操纵语言结构。例如，将英语句子的词序从 SVO 改为 OVS（一种极罕见的词序）在原则上并不比改为 SOV 更难，但 LLM 的表现可能截然不同。

**本文目标**：(1) 提供一个灵活有趣的人造语言构建工具；(2) 利用形态句法变换任务系统探测 LLM 对不同语言类型学特征的元语言知识水平。

**切入角度**：构建人造语言（ConLang）要求 LLM 不只是翻译，而是根据抽象的语法规格重组句子结构、添加形态标记——这直接考验其对语言学概念的理解深度。

**核心 idea**：用一个模块化的人造语言构建系统作为 benchmark，通过让 LLM 将英语句子按不同的形态句法参数（词序、格系统、时态标记等）进行变换，来量化其元语言学能力。

## 方法详解

### 整体框架

IASC 是一个完整的人造语言构建 pipeline，包含音系学（phonology）、形态句法（morphosyntax）、词库（lexicon）、正字法（orthography）和语法手册（grammatical handbook）五个模块。本文重点关注形态句法模块作为 LLM 元语言知识的探测工具。输入为英语源句子 + 目标语法参数，输出为按目标语法变换后的 gloss 标注。采用累积变换策略（cumulative morphosyntax），通过多步 prompt 逐步应用不同语法特征。

### 关键设计

1. **累积形态句法变换（Cumulative Morphosyntax）**:

    - 功能：将源句子逐步变换为符合目标语法规格的形式
    - 核心思路：不是一次性给出所有语法规格让 LLM 变换（preliminary 实验表明效果很差），而是每次只应用一个语法特征（如先改词序，再加格标记，再加时态标记），通过迭代 prompt $s_i = M(s_{i-1}; G; t_i)$ 逐步累积变换。每步的 prompt $t_i$ 只关注一个特定语法特征
    - 设计动机：一次性变换导致 prompt 过长且复杂，LLM 难以同时遵循多个约束。分步累积降低了每步的认知负担

2. **九种类型学多样的语法配置**:

    - 功能：构建覆盖常见到罕见语言类型的评估数据集
    - 核心思路：设计了受八种真实语言启发的语法配置（阿拉伯语、斐济语、法语、希克卡里亚纳语、米佐语、土耳其语、越南语、威尔士语）加一个"hard"配置（极罕见的类型学组合）。每种配置定义了词序、格系统、一致性标记、时态标记等参数。构建了 45 个源句子 × 9 种配置 = 405 个测试样本，gold data 由语言学家手工标注
    - 设计动机：通过控制类型学频率来测试 LLM 是否真正理解抽象规则，还是只能处理训练数据中常见的模式

3. **Agentic 自我改进机制**:

    - 功能：通过自动生成反馈来迭代改进输出
    - 核心思路：部分模块（如音系学）采用 agentic 方法——LLM 先生成初始输出，然后自动生成对输出的评论/反馈，再根据反馈改进输出，迭代进行
    - 设计动机：LLM 的首次输出可能不完全符合规格，通过自我审查和修正机制提高质量

## 实验关键数据

### 主实验

| 模型 | 'french' (常见) | 'turkish' (常见) | 'mizo' (罕见) | 'hard' (极罕见) | 整体表现 |
|------|----------------|-----------------|---------------|----------------|---------|
| GPT-4.1 | TER 低 | TER 低 | TER 中等 | TER 较高 | 最好 |
| Claude 3.7 | TER 低 | TER 低 | TER 中高 | TER 高 | 第二 |
| Gemini 2.5 | TER 中等 | TER 中等 | TER 高 | TER 很高 | 中等 |
| 较小模型 | TER 高 | TER 高 | TER 很高 | TER 极高 | 较差 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 累积变换 vs 一次性变换 | 累积远优 | 一次性变换 LLM 无法同时遵循多约束 |
| 常见类型学特征 vs 罕见特征 | 常见远优 | LLM 对 SVO/SOV 处理好，OVS/OSV 差 |
| 形态标记（前缀 vs 后缀） | 后缀更好 | 与训练数据中后缀更常见一致 |
| 有 agentic refinement vs 无 | 有时改善 | 并非所有模块都受益 |

### 关键发现

- LLM 对常见语言类型学模式（如 SVO、SOV 词序、后缀式形态）的处理明显优于罕见模式（如 OVS 词序、前缀式形态），与该特征在世界语言中的分布频率高度相关
- 不同 LLM 之间能力差异巨大：GPT-4.1 在大多数配置上表现最好，而较小模型在罕见配置上几乎完全失败
- "hard" 语言配置（含极罕见类型学组合）对所有模型都极具挑战性，说明 LLM 的元语言知识仍受训练数据分布强烈约束

## 亮点与洞察

- **用人造语言作为探测工具**：极其巧妙的实验设计——人造语言避免了训练数据泄露问题，且能精确控制语言学变量，使得评估结果可解释性极强
- **揭示了 LLM "语言知识"的本质**：LLM 不是真正"理解"语言学概念，而是依赖训练数据中的模式分布。常见的语言类型处理好、罕见的处理差，说明其能力本质上是统计相关性而非抽象规则理解
- **累积变换策略**：将复杂的多约束问题分解为逐步单约束变换，是一种通用的 prompt engineering 策略，可迁移到其他需要多步推理的场景

## 局限与展望

- 评估数据集（405 个样本）相对较小，可能不足以捕捉所有语法特征的交互效应
- 仅以英语为源语言，未探索从其他语言出发的变换效果
- 形态句法模块的 gold data 由单个语言学家标注，可能引入标注者偏差
- 作者也尝试了将方法应用于低资源语言翻译，但结果大多为负面，距实际应用还有距离
- 53 页的论文包含大量附录，核心贡献可以更集中

## 相关工作与启发

- **vs ConlangCrafter (Alper et al., 2025)**: 也做 LLM 驱动的人造语言构建，但 IASC 的形态句法模块更细粒度，支持逐特征探测
- **vs 传统 LLM 语言能力测试**: 如 BLiMP、SyntaxGym 等测试 LLM 对特定语言现象的判断，IASC 则要求 LLM 主动进行语言结构操纵，难度更高
- **vs Diamond (2023)**: 仅用 ChatGPT 通过简单 prompt 生成人造语言，未做系统的模块化控制和类型学评估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 用人造语言构建来探测元语言知识是非常新颖且有深度的研究视角
- 实验充分度: ⭐⭐⭐⭐ 九种语法配置覆盖了丰富的类型学多样性，但样本量偏小
- 写作质量: ⭐⭐⭐⭐ 论文极为详尽（53页），语言学背景介绍充分，但过于冗长
- 价值: ⭐⭐⭐⭐⭐ 对理解 LLM 的语言知识本质提供了关键洞察，IASC 工具本身也有独立价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](../../AAAI2026/llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

</div>

<!-- RELATED:END -->
