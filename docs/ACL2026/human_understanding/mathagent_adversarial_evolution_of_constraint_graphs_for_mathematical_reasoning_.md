---
title: >-
  [论文解读] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis
description: >-
  [ACL 2026][人体理解][数学推理] 提出基于约束图对抗进化的分层数据合成框架 MathAgent，将数据合成从文本生成任务重构为约束图的无监督优化问题，通过 Legislator 三Agent系统进化问题骨架再由 Executor 实例化为自然语言，仅 1K 合成样本即超越 LIMO 和 s1K 在八个数学基准上的表现。
tags:
  - ACL 2026
  - 人体理解
  - 数学推理
  - 数据合成
  - 约束图
  - 对抗进化
  - Legislator-Executor
---

# MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis

**会议**: ACL 2026  
**arXiv**: [2604.11188](https://arxiv.org/abs/2604.11188)  
**代码**: 无  
**领域**: 数据合成 / LLM推理  
**关键词**: 数学推理, 数据合成, 约束图, 对抗进化, Legislator-Executor

## 一句话总结
提出基于约束图对抗进化的分层数据合成框架 MathAgent，将数据合成从文本生成任务重构为约束图的无监督优化问题，通过 Legislator 三Agent系统进化问题骨架再由 Executor 实例化为自然语言，仅 1K 合成样本即超越 LIMO 和 s1K 在八个数学基准上的表现。

## 研究背景与动机

**领域现状**：高质量数学推理数据是提升 LLM 推理能力的关键驱动力之一。随着人工标注数据的规模瓶颈日益凸显，合成数据生成已成为主流研究方向。

**现有痛点**：(1) 种子扩展方法（如 Self-Instruct、WizardMath）受限于初始种子的"语义半径"，多样性有上界；(2) 零样本方法（如 Magpie）直接探测模型分布，缺乏结构引导，容易模式坍缩和逻辑幻觉；(3) 现有方法将数据合成视为直接的文本生成任务，模型往往停留在表面叙事模仿，未能掌握核心推理能力。

**核心矛盾**：直接在 token 空间进行数据合成，无法有效控制问题的逻辑复杂度和结构多样性——高难度、高质量的长尾样本恰恰是锻造复杂推理能力的关键，但标准方法难以发现这些样本。

**本文目标**：设计无需人工种子数据、可自动探索结构空间的合成框架，生成兼具高复杂度和高多样性的数学推理数据。

**切入角度**：将数据合成解耦为结构进化（meta-level）和语义实例化（base-level）两个阶段——先优化问题的逻辑骨架（约束图），再将骨架转化为自然语言题目。

**核心 idea**：用约束图表示数学问题的逻辑结构，通过三Agent对抗进化机制（Proposer-Critic-Moderator）持续优化图结构的复杂度和多样性，再由 Executor 生成自然语言问题和推理链。

## 方法详解

### 整体框架
MathAgent 分为两个解耦阶段：(1) Meta-Level 结构进化：Legislator 三Agent系统在约束图空间中对抗进化，产出优化的问题骨架 $\mathcal{G}^*$；(2) Base-Level 语义实例化：Executor 将 $\mathcal{G}^*$ 和风格令牌 $\mathcal{S}$ 转化为自然语言问题 Q 和推理链 A。最后通过外部模型验证筛选合格样本。

### 关键设计

1. **约束图表示 (Constraint Graph)**:

    - 功能：形式化描述数学问题的逻辑骨架
    - 核心思路：将问题建模为图 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ 加风格令牌 $\mathcal{S}$。节点 $\mathcal{V}$ 表示数学概念，边 $\mathcal{E}$ 表示逻辑关系，$\mathcal{S}$ 控制全局属性（问题类别、难度级别等）。优化目标为 $\mathcal{G}^* = \arg\max_{\mathcal{G}} \mathcal{H}(\mathcal{G})$，其中 $\mathcal{H}$ 估计复杂度，约束 $\mathbb{I}_{\text{valid}}(\mathcal{G}|\mathcal{S})=1$ 保证可解性
    - 设计动机：将结构规范与文本实现解耦，使框架可以专注于构建复杂且多样的逻辑结构，而非被表面语言模式束缚

2. **Legislator 三Agent进化系统**:

    - 功能：通过对抗动态迭代优化约束图结构
    - 核心思路：三个角色协同工作——Proposer ($\mathcal{A}_P$) 根据前轮反馈优化 $\mathcal{G}_t \to \mathcal{G}_{t+1}$，解决逻辑矛盾并扩展结构深度；Critic ($\mathcal{A}_C$) 从内部一致性、规范对齐、优化潜力三个维度审查新图，生成改进报告；Moderator ($\mathcal{A}_M$) 作为战略决策者，要么终止进化（已收敛），要么指导 Proposer 实施改进。初始化阶段也通过对抗机制自动构建概念分类学和风格令牌池
    - 设计动机：对抗进化驱动系统持续探索结构空间的前沿，能发现标准数据集中缺失的高难度样本；自适应截断避免过度进化

3. **Executor 语义实例化**:

    - 功能：将优化后的约束图转化为自然语言数学题和推理链
    - 核心思路：条件生成模型 $(Q, A) \sim P_{\text{executor}}(\cdot | \mathcal{G}^*, \mathcal{S})$，基于线性化的图表示生成问题和答案。由于复杂度和多样性已由 Legislator 保证，Executor 只需专注于语言本身。生成后通过外部模型（judge）验证逻辑正确性和问答一致性
    - 设计动机：解耦后 Executor 免于探索复杂度空间的负担，能更高效地生成多样化的文本表述

### 损失函数 / 训练策略
合成数据用于微调目标模型，采用标准 SFT 训练。验证阶段使用外部 LLM 作为 judge 评估合成 QA 对的逻辑正确性和一致性，仅保留通过验证的样本。

## 实验关键数据

### 主实验

| 模型 | 数据集 | GSM8K | MATH500 | AIME24 | AIME25 | 平均 |
|------|--------|-------|---------|--------|--------|------|
| Qwen3-14B | LIMO | 91.8 | 86.2 | 33.8 | 27.5 | 59.5 |
| Qwen3-14B | s1K | 87.5 | 86.4 | 37.9 | 25.0 | 60.3 |
| Qwen3-14B | **Ours** | **95.4** | **91.8** | **38.8** | **30.0** | **63.9** |
| Qwen2.5-Math-7B | LIMO | 87.4 | 72.2 | 10.8 | 14.6 | 45.6 |
| Qwen2.5-Math-7B | **Ours** | **91.6** | **82.2** | **18.8** | **18.3** | **53.5** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 MathAgent | 63.9 (Qwen3-14B Avg) | 全部组件 |
| w/o Critic | ~60.5 | 无对抗审查，结构质量下降 |
| w/o 自适应截断 | ~61.2 | 固定进化轮数，效率降低 |
| 直接文本生成 | ~58.0 | 不使用约束图，模式坍缩 |

### 关键发现
- 仅 1K 合成样本即可超越同规模的 LIMO 和 s1K，展现出极强的数据效率
- 在 AIME 等高难度竞赛基准上提升尤为显著，验证了框架在长尾高难度样本上的优势
- 跨模型系列泛化能力强：在 Qwen、Llama、Mistral、Gemma 四个系列共 10 个模型上均有效
- 越小的模型从 MathAgent 数据中受益越大，Qwen3-4B 从 base 的 42.8 提升至 53.5

## 亮点与洞察
- 将数据合成从文本空间提升到结构空间是关键创新——约束图作为中间表示，有效分离了"问题有多难"和"问题怎么说"两个正交维度
- 对抗进化机制不依赖种子数据，从模型内在的概念原语出发即可构建高质量数据，真正实现了"无中生有"
- 自适应截断机制类似于 early stopping，避免了过度进化导致的不可解问题，体现了对合成数据质量与复杂度的精细平衡

## 局限与展望
- 当前仅在数学推理领域验证，能否推广到代码生成、逻辑推理等需要结构化数据的场景尚未探索
- Legislator 系统需要多轮 LLM 交互进行进化，合成成本可能高于简单的种子扩展方法
- 外部 judge 验证可能存在自身盲点，对极端困难问题的正确性判断可能不完全可靠

## 相关工作与启发
- **vs LIMO/s1K**: 这些方法依赖精心筛选的种子数据，MathAgent 完全自动化生成且以更少数据量超越它们
- **vs Self-Instruct**: Self-Instruct 在 token 空间扩展，多样性受限于种子语义半径；MathAgent 在结构空间探索，能发现更远的分布区域
- **vs Magpie**: Magpie 零样本但缺乏结构引导易模式坍缩；MathAgent 通过约束图提供结构骨架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 约束图+对抗进化的分层合成框架是全新范式
- 实验充分度: ⭐⭐⭐⭐⭐ 10个模型、8个基准、跨系列验证
- 写作质量: ⭐⭐⭐⭐ 形式化清晰，但部分符号略重
- 价值: ⭐⭐⭐⭐⭐ 1K数据超越主流方法，数据效率惊人

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/human_understanding/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ICLR 2026\] Scaling Generalist Data-Analytic Agents](../../ICLR2026/human_understanding/scaling_generalist_data-analytic_agents.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](../../AAAI2026/human_understanding/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](../../CVPR2026/human_understanding/graph2eval_multimodal_task_generation_agents.md)

<!-- RELATED:END -->
