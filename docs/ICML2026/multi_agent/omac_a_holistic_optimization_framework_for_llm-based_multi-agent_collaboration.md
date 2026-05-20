---
title: >-
  [论文解读] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration
description: >-
  [ICML 2026][LLM Agent / 多智能体系统 / 代码生成][多智能体系统] 本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在…
tags:
  - "ICML 2026"
  - "LLM Agent / 多智能体系统 / 代码生成"
  - "多智能体系统"
  - "协同优化"
  - "对比推理"
  - "提示进化"
  - "监督优化"
---

# OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration

**会议**: ICML 2026  
**arXiv**: [2505.11765](https://arxiv.org/abs/2505.11765)  
**代码**: https://github.com/xiwenchao/OMAC  
**领域**: LLM Agent / 多智能体系统 / 代码生成  
**关键词**: 多智能体系统、协同优化、对比推理、提示进化、监督优化

## 一句话总结
本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在每个维度上做监督式优化，再迭代联合优化多个维度，在 HumanEval / MMLU / MATH 上稳定打败 DyLAN、ADAS、AFlow 等基线。

## 研究背景与动机

**领域现状**：多智能体系统（MAS）已在代码生成（AgentVerse）、推理（LLM Debate）、决策（Sun 2024）等任务上展示出超越单 agent 的能力。但现有 MAS 大都靠人工设计——agent 角色用人类先验或 LLM 直接生成，协作结构用固定的中心化/去中心化/层次化拓扑。少数自动化工作如 DyLAN（动态选 agent 团队组成）、ADAS（提示进化）、AFlow（MCTS 架构搜索）、G-Designer/MaAS（架构搜索）也只优化单一方面。

**现有痛点**：（1）DyLAN 用无监督的 "agent importance score"，缺乏从训练数据导出的监督信号；（2）ADAS / AFlow / G-Designer / MaAS 只优化某一个方面（提示 *或* 架构）；（3）这些方法都不能同时改 agent 功能（提示/few-shot）和协作结构（候选选择 / 动态参与 / 通信流），缺少统一框架。

**核心矛盾**：MAS 本质上是一张信息流图——节点（agent）和边（通信链路）都可优化，但现有方法要么只动节点要么只动边，并且没有一个通用算法可以同时处理这两类。

**本文目标**：构造一个统一框架，能用同一个算法监督式地优化 MAS 的任何一个维度，并能迭代联合优化多个维度。

**切入角度**：把 MAS 的协作过程概念化为信息流图后，作者识别出 5 个核心可优化维度——2 个关于节点（agent）功能，3 个关于图（结构）；并发现这 5 个维度都可以归结为"优化一段 LLM 的指令提示（外加可选的 few-shot 示例）"。

**核心 idea**：用一对 LLM 驱动的 actor——Semantic Initializer 探索语义空间生成多样化的初始提示集合，Contrastive Comparator 对比高低分对子的差异生成更优提示——在每个维度上做监督式对比推理优化，再用"一次一维度"的迭代策略联合多个维度。

## 方法详解

### 整体框架
OMAC 的输入是一个已有的 MAS 配置（多个 agent + 协作拓扑）和一个有监督评估指标的训练集（如 HumanEval 的 Pass@1）。框架由两层组成：（1）**单维度优化算法**——选定 5 个维度之一，用 Semantic Initializer 生成 $n$ 个候选 agent/controller 提示，在训练集上跑完整 MAS 协作得到性能分；按上/下阈值 $h, l$ 抽样一对正负样本送入 Contrastive Comparator；后者做对比推理生成新候选加入集合；循环至预设迭代次数。（2）**多维度联合优化**——选定要联合的多个维度，按"固定其他、单优一个"的方式串行迭代，避免多维度同时变动让 Contrastive Comparator 难以归因。

### 关键设计

1. **五维优化空间的划分**:

    - 功能：把 MAS 的所有可优化部分系统地分成 5 个互不重叠的维度——Fun-1（优化已有 agent 的提示 / few-shot）、Fun-2（构造新 agent）、Str-1（候选 agent 选择控制器）、Str-2（每步动态参与控制器）、Str-3（agent 间通信流控制器）。
    - 核心思路：把 MAS 看作信息流图，节点是 agent、边是通信链路；Fun-1/2 优化节点能力（改已有节点 / 加新节点），Str-1/2/3 优化图结构（全局团队选择 / 每步局部选择 / 边路由）。所有 5 个维度都是"优化一段 LLM 提示"的问题，所以同一个算法可以套用，仅需替换上下文描述。
    - 设计动机：以前的工作各自只覆盖某一维（DyLAN 优化 Str-1，ADAS/AFlow 主要做 Str-3 + Fun-2 的某种组合等），缺乏统一坐标系。明确分维度后，作者发现 5 维都有独立的提升空间，且可以叠加。

2. **Semantic Initializer + Contrastive Comparator 双 actor 算法**:

    - 功能：用两个 LLM 驱动的 actor 在每个维度上做监督式优化——前者负责探索（生成多样初始集合），后者负责利用（对比正负差异生成改进）。
    - 核心思路：Semantic Initializer 接收上下文（任务描述 + 当前 MAS 配置 + 维度规格 + 一个 one-shot 示例）和数量 $n$，让 LLM 在保持功能不变的前提下做语义空间的多样化生成。每个候选放入 MAS 跑训练集得到分数，按 top-$\lfloor n h \rfloor$ / bottom-$\lfloor n l \rfloor$ 阈值抽一对正负样本送给 Contrastive Comparator，后者做对比推理——"为什么 A 比 B 好？把 A 里的优点强化、把 B 里的缺点去掉，生成一个比 A 还好的新版本"，新候选加入集合，循环。
    - 设计动机：单纯让 LLM 生成提示是无监督探索，没充分利用训练集分数。对比推理把"性能差"作为监督信号，让 LLM 用自身推理能力做归因——这是一种轻量级、不需要梯度的"伪 RL"，且对比样本一次只变一个维度避免归因混乱。

3. **迭代式多维度联合优化**:

    - 功能：在多个维度间轮转优化，每轮只动一维、固定其他，避免组合爆炸和 Comparator 多变量归因失败。
    - 核心思路：先对每个维度独立跑单维优化，选出收益最大的几个；联合优化时，对 dim-1 跑完单维优化保留最优 → 切换到 dim-2 在固定 dim-1 的前提下再跑单维优化 → 回到 dim-1 ……直到达到迭代上限。论文显示选两个收益最大的功能维度联合，或者最强功能 + 最强结构维度联合，效果最佳。
    - 设计动机：迭代式而非并行式，是为了保证 Contrastive Comparator 的归因清晰——每次对比对子里只有一个维度在变，其他维度保持当前最优，差异原因可解释，避免"多个变量同时影响性能"的混淆。论文用 ablation 验证：让 Comparator 同时推理多个变化维度，性能提升显著更小、方差更大。

### 损失函数 / 训练策略
OMAC 不更新模型参数，所有"优化"都是提示工程。监督信号是 MAS 在训练集上的任务指标（HumanEval Pass@1 / MMLU Accuracy / MATH Accuracy）。基础 LLM 用 gpt-3.5-turbo-1106，温度 0.8；Semantic Initializer 每轮生成 3 个候选；对比推理迭代上限 3；阈值 $l=h=0.5$；每个实验跑 3 次报均值±标差。

## 实验关键数据

### 主实验
基准任务：HumanEval（代码生成，Pass@1）、MMLU（通用推理，Accuracy）、MATH（算术推理，Accuracy）。默认 MAS 配置来自 DyLAN（代码 7 agent、推理 7 agent、算术 4 agent），全连通拓扑。

| 任务 | 最强基线 | 基线分数 | OMAC 单维最高 | 维度 |
|------|---------|---------|--------------|------|
| HumanEval (Pass@1) | AFlow | 85.63 | 89.25 ± 1.30 | Fun-1.4 |
| MMLU (Accuracy) | ADAS | 69.02 | 74.22 ± 2.22 | Fun-1.4 |
| MATH (Accuracy) | AFlow | 32.49 | 35.17 ± 1.96 | Fun-1.1 |

联合优化（MATH 任务上 3 次迭代）：

| 优化策略 | 性能提升（相对默认 MAS） |
|---------|----------------------|
| 单维度最佳 | ~2.9% |
| 两个最强维度联合迭代 3 轮 | ~9.6% |

### 消融实验
OMAC-C：去掉 Contrastive Comparator，只用 Semantic Initializer 生成初始集合后选最高分。

| MATH 维度 | OMAC-C | OMAC | Comparator 收益 |
|----------|--------|------|---------------|
| Str-1 | 32.64 | 33.34 | +0.70 |
| Str-2 | 32.67 | 33.41 | +0.74 |
| Str-3 | 32.76 | 33.70 | +0.94 |
| Fun-1.1 | 34.20 | 35.17 | +0.97 |
| Fun-2 | 32.71 | 33.95 | +1.24 |

### 关键发现
- **5 个维度都有独立提升空间**：单维度优化平均收益分别是 HumanEval 3.6%、MMLU 2.8%、MATH 4.9%，证明 5 维划分确实捕获了正交的优化方向。
- **功能维度通常比结构维度收益更大**：Fun-1.x 在三个任务上都给出了最高的单维度提升，说明改 agent 提示比改协作拓扑更直接。
- **联合优化收益超过单维度之和**：MATH 上单维度最高 2.9% → 联合 9.6%，说明结构和功能优化存在协同。"选最强两个维度" 策略稳定优于随机两个或最弱组合。
- **Comparator 不可或缺**：去掉对比推理后 OMAC-C 在所有维度都掉点，证明监督式对比推理比单纯探索更高效；但 OMAC-C 仍超过 DyLAN，说明 Semantic Initializer 的多样化探索本身就有价值。
- **降低推理成本**：动态 agent 选择 + 通信流优化让 OMAC 在推理时通常使用比基线更少的 agent，显著降低 token 消耗。

## 亮点与洞察
- **把 MAS 优化空间形式化为 5 维坐标系**：以前每篇 MAS 优化论文都在自己定义"什么是优化"，本文给出一个清晰的图论坐标系（节点能力 × 图结构）覆盖既有工作，分类干净易扩展。
- **对比推理作为轻量监督信号**：不需要梯度、不需要 RL 训练，只用 LLM 本身的推理能力把"性能差"翻译成"提示改进"，几乎是个通用范式，可以迁移到任何"优化某段自然语言"的场景。
- **"一次一维度"的迭代准则**：这个看似简单的工程选择背后是对 LLM 归因能力局限的清醒认识——同时变多个变量会让 LLM 给出错误的因果归因，论文用 ablation 量化了这一点。
- **可与任何 MAS 基础设计互补**：OMAC 的输入是"已有 MAS 配置"，输出是"优化后的 MAS"，所以任何手工设计的 MAS（包括未来更复杂的）都可以套上 OMAC 再提升。

## 局限与展望
- **依赖训练集监督信号**：对没有明确指标的开放性任务（如开放对话质量、长程 agent 规划）不直接适用，作者只在 GAIA 等开放任务上做了少量补充实验。
- **优化质量受 LLM 自身推理能力限制**：作者用 GPT-3.5 做主实验，更弱的 LLM 当 Comparator 时对比推理可能不稳；附录有 GPT-4 实验但篇幅不够展开。
- **联合优化的组合爆炸**：5 维全部联合的算力是单维的 $5^k$ 倍（$k$ 是迭代轮数），所以只能选最优两三维度，没法做真正全面的联合优化。
- **没考虑 agent 之间的角色冲突**：Fun-2（新建 agent）后 MAS 里可能出现职责重叠或矛盾，论文没讨论如何检测和处理。
- **训练数据子集采样策略经验化**：附录提到用训练数据子集而非全集评估候选可以省算力，但子集规模和采样策略缺乏理论分析。

## 相关工作与启发
- **vs DyLAN (2024)**：DyLAN 用无监督的 Agent Importance Score 仅优化候选选择（Str-1）；OMAC 用监督式对比推理覆盖 5 个维度，且对 Str-1 也比 DyLAN 强。
- **vs ADAS / AFlow**：这两者主要做架构搜索（提示进化 / MCTS），优化 Fun-2 + 部分 Str-3；OMAC 用更轻量的对比推理算法在更广维度上一致更优。
- **vs G-Designer / MaAS**：纯结构搜索方法；OMAC 在结构维度上不弱（Str-1/2/3 都打败它们），且额外能优化功能维度。
- **vs gradient-based 方法（SFT / prompt tuning）**：单 agent 优化思路，无法处理 MAS 多步协作；OMAC 不需要梯度、用 LLM 推理能力替代。
- **vs RL-based MAS 优化（Shao 2024, Liu 2025）**：往往局限于单步交互或共享策略；OMAC 处理多步、角色专用 MAS 更自然。

## 评分
- 新颖性: ⭐⭐⭐⭐ 五维划分和对比推理双 actor 算法是清晰的概念贡献，虽然单个组件不算颠覆但组合产生新范式。
- 实验充分度: ⭐⭐⭐⭐ 3 个经典 benchmark + 2 个困难 benchmark（MBPP/GAIA）+ ablation + 多基线对比，超参研究也比较完整。
- 写作质量: ⭐⭐⭐⭐ 框架清晰、图示直观，公式不多但每个设计的"为什么"都说清楚了。
- 价值: ⭐⭐⭐⭐ 对实战 MAS 是即插即用的优化套件，且范式可迁移到任何"优化自然语言指令"的场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](../../ACL2026/multi_agent/masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](../../ACL2026/multi_agent/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)
- [\[ACL 2026\] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](../../ACL2026/multi_agent/mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)

</div>

<!-- RELATED:END -->
