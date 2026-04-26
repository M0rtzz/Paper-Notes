---
title: >-
  [论文解读] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][约束规划] 本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。
tags:
  - ACL 2026
  - 约束规划
  - LLM-as-Planner
  - LLM-as-Formalizer
  - 基准测试
  - PDDL
---

# Language Model as Planner and Formalizer under Constraints

**会议**: ACL 2026  
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)  
**代码**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)  
**领域**: LLM规划与评估  
**关键词**: 约束规划, LLM-as-Planner, LLM-as-Formalizer, 基准测试, PDDL

## 一句话总结

本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。

## 研究背景与动机

**领域现状**：LLM 在规划领域有两种主流范式——LLM-as-Planner 直接端到端生成动作序列，LLM-as-Formalizer 将自然语言描述转为 PDDL 等形式语言再用求解器推导方案。两种方法在标准规划基准上均展现了不俗的能力。

**现有痛点**：然而，现有基准（如 BlocksWorld、Gripper 等）大多诞生数十年，环境描述简单、同质化严重，且高度可能被 LLM 训练数据覆盖。这种简单性可能导致对 LLM 规划能力的**过度高估**，在下游安全敏感场景中构成隐患。

**核心矛盾**：真实世界的规划指令通常包含用户或资源施加的**个性化需求与约束**，而标准基准完全缺少这些元素。已有的增强方法仅加入噪声或词汇扰动，未改变语义本身。

**本文目标**：构建一个语义层面增强的约束规划基准，系统评估 LLM 在约束条件下的规划和形式化能力。**切入角度**：将约束按语言学和实用主义方法形式化为四类（Initial、Goal、Action、State），确保分类的完备性。**核心 idea**：简单的一句话约束即可大幅降低 LLM 性能，且这种性能下降在问题复杂度增加和词汇混淆时进一步加剧。

## 方法详解

### 整体框架

CoPE (Constrained Planning Environments) 在 BlocksWorld 和 CoinCollector 两个域上，为每个问题手动标注自然语言约束及其四种形式语言的 ground-truth 编码。评估流程：给定域描述 $D_d$、问题描述 $D_p$、PDDL 头部 $\mathcal{DF}'$ 和约束 $\mathcal{C}$，LLM 生成计划（Planner）或形式化代码（Formalizer），最终用 VAL 验证器验证计划正确性。

### 关键设计

1. **四类约束的形式化定义**：

    - 功能：将自然语言约束严格分为 Initial（修改初始状态）、Goal（修改目标状态）、Action（限制合法动作序列）、State（限制合法状态轨迹）四类
    - 核心思路：基于原始动作/状态空间（primitive）与修改后空间（modified）的集合关系定义，证明 State 约束子类涵盖所有可能约束，确保分类**完备性**
    - 设计动机：不同形式语言（PDDL、PDDL3、LTL、SMT）对不同类别约束的表达能力各异，形式化分类可支撑系统性分析

2. **多形式语言对比评估**：

    - 功能：将约束分别编码为 PDDL 1.2、PDDL3、LTL 和 SMT（Z3），评估各形式语言的表达和求解能力
    - 核心思路：Generation（直接生成）、Editing（先生成无约束代码再编辑）、Revision（最多 3 次语法错误修正）三种技术路线
    - 设计动机：不同约束类型天然适合不同形式语言，如 PDDL3 擅长状态约束语法、SMT 擅长状态谓词建模，系统对比可为未来工具链选择提供指导

3. **鲁棒性拓展实验**：

    - 功能：通过 BlocksWorld-XL（50个方块）和 MysteryBlocksWorld（词汇混淆）评估复杂度扩展和数据污染
    - 核心思路：XL 版本测试实体空间增大后的性能，Mystery 版本将所有类型/谓词/动作名替换为无意义占位符
    - 设计动机：验证约束是否会放大 LLM 在复杂问题和词汇扰动下的已有脆弱性

### 损失函数 / 训练策略

本文为评估型工作，不涉及模型训练。核心评估指标为 **plan correctness**——预测计划在 ground-truth PDDL 环境中能否成功从初始状态转移到目标状态。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 无约束 | 有约束 | 下降幅度 |
|--------|------|--------|--------|----------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | ~30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | ~30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | ~30% |
| BlocksWorld | PDDL3 Formalizer | 低于 PDDL | 更低 | 语法/编译错误多 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Generation | 基线 | 直接生成约束代码 |
| Editing | 部分提升 | 先生成无约束版本再编辑 |
| Revision | 进一步提升 | 基于求解器错误迭代修正 |
| BlocksWorld-XL (50 blocks) | 性能骤降 | 复杂度放大后约束影响更严重 |
| MysteryBlocksWorld | Formalizer 鲁棒性消失 | 约束 + 词汇混淆双重打击 |

### 关键发现
- 一句话约束一致性地将性能减半，所有 LLM、方法、语言组合均受影响
- LLM-as-Planner 在无约束时整体优于 Formalizer，但 Formalizer 对问题复杂度更鲁棒
- PDDL3 虽然有约束语法支持，但因训练数据稀缺表现反而不如标准 PDDL
- 约束引入后，Formalizer 原有的复杂度鲁棒性和词汇扰动鲁棒性**完全消失**

## 亮点与洞察
- 约束分类的形式化定义非常严谨，证明了完备性，可作为后续工作的理论基础
- 实验设计覆盖 4 个 LLM × 4 种形式语言 × 3 种技术 × 4 类约束 × 4 个数据集，分析维度丰富
- 揭示了一个重要结论：**简单的语义修改比词汇噪声更能有效挑战 LLM**，为基准设计提供新思路
- CoPE 的设计理念——通过语义增强而非数据扰动来对抗数据污染——值得在其他 NLP 评估任务中借鉴

## 局限与展望
- 约束类型仅考虑单约束，未讨论约束的合取、否定和歧义，真实场景的约束更为多样
- BlocksWorld 和 CoinCollector 域仍较简单，与真实世界规划场景（如机器人操作、资源调度）有较大差距
- 评估指标（plan correctness）可能存在 false positive——计划碰巧正确但代码未真正编码约束，不过验证显示比例可忽略
- 未来方向：支持更复杂的约束组合、扩展到更多域、开发约束感知的规划工具链
- 自主 Agent 在下游任务中的安全风险值得关注，形式化表示可提供人类审计和形式验证的透明性

## 相关工作与启发
- **vs 标准 IPC 基准**: CoPE 通过语义修改而非仅加噪声来挑战 LLM，更能暴露真实能力
- **vs LLM+P (Liu et al., 2023)**: 同为 Formalizer 路线但未考虑约束，CoPE 揭示其局限
- **vs Mystery BlocksWorld**: CoPE 表明约束比词汇混淆更能削弱 Formalizer 的鲁棒性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化的约束规划 LLM 评估基准，形式化分类严谨
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多模型 × 多语言 × 多技术 × 多域，分析极为详尽
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，结构合理，图表丰富
- 价值: ⭐⭐⭐⭐ 为 LLM 规划研究敲响警钟，指明了从简单基准到现实约束的重要研究方向

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/llm_evaluation/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)
- [\[ACL 2025\] CFBench: A Comprehensive Constraints-Following Benchmark for LLMs](../../ACL2025/llm_evaluation/cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_model_task_adaptation.md)

<!-- RELATED:END -->
