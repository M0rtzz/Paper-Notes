---
title: >-
  [论文解读] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models
description: >-
  [ACL 2025 (Long Paper)][代码智能][Code Instruction Synthesis] 提出Tree-of-Evolution (ToE)——一种树结构代码指令合成框架，通过多路径进化和质量驱动优化克服Code Evol-Instruct和OSS-Instruct的单向合成与随机生…
tags:
  - "ACL 2025 (Long Paper)"
  - "代码智能"
  - "Code Instruction Synthesis"
  - "Tree-Structured Evolution"
  - "Optimization-Driven Generation"
  - "Code LLM"
  - "Data Quality"
---

# Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models

**会议**: ACL 2025 (Long Paper)  
**arXiv**: 无  
**代码**: 无  
**领域**: 代码生成 / 数据合成 / 指令微调  
**关键词**: Code Instruction Synthesis, Tree-Structured Evolution, Optimization-Driven Generation, Code LLM, Data Quality  

## 一句话总结
提出Tree-of-Evolution (ToE)——一种树结构代码指令合成框架，通过多路径进化和质量驱动优化克服Code Evol-Instruct和OSS-Instruct的单向合成与随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

## 研究背景与动机

**数据瓶颈**: 高质量代码指令数据的人工标注成本极高（平均每条需15-30分钟专家审核），LLM驱动的自动合成方法成为主流解决方案。

**现有方法的两大缺陷**: (1) **单向合成**——Code Evol-Instruct和OSS-Instruct等方法对每条种子指令只生成一条进化链（A->B->C），限制了数据多样性和进化空间的探索；(2) **随机驱动**——进化方向随机选择（如"增加复杂度"、"改变算法"等），不考虑前一步的生成质量，导致进化路径中大量低质量死胡同。

**核心挑战**: 在固定的合成预算（如75K条）下，如何同时最大化代码指令的**质量**和**多样性**？

## 方法详解

### 整体框架
将代码指令合成建模为**树搜索优化问题**：每条种子指令作为树的根节点，可沿多条进化路径分支发展（树的宽度），每条路径可多步迭代（树的深度）。通过质量评估驱动的剪枝和扩展策略，在有限预算下最优地探索进化空间。

### 关键设计

1. **树结构多路径进化**

    - 传统方法将指令进化建模为线性链（A->B->C），ToE将其建模为多叉树（A->{B1,B2,B3}->{C1,...,C9}->...）
    - 每个节点可探索多个进化方向：增加算法复杂度、改变编程范式、添加边界约束、引入新数据结构等
    - 树的分支因子（branching factor）和最大深度可调，平衡探索广度与深度
    - 同一根节点的不同分支天然保证了主题相关性下的多样性（如同一个排序问题可分化出归并排序优化、并行排序、外部排序等方向）

2. **优化驱动的进化策略**

    - 每一步进化后立即进行质量评估：代码可执行性检查、测试用例通过率、代码复杂度分析等
    - 基于前一轮质量反馈动态调整下一轮的进化策略：高质量节点获得更多分支预算，低质量节点被剪枝
    - 类似蒙特卡洛树搜索(MCTS)中的UCB引导思想，但用代码质量信号替代奖励信号
    - 有效避免了随机进化导致的资源浪费——传统方法中30-50%的合成数据因质量不达标被丢弃

3. **多维质量评估**

    - 渐进式筛选：语法正确性 -> 可执行性 -> 功能正确性（测试用例） -> 代码质量（复杂度、可读性）
    - 每层评估淘汰不达标的进化分支，集中资源于高潜力方向
    - 保留进化路径中的最优节点（不一定是叶子节点），构建最终训练数据集

### 训练策略
- 在75K合成数据上对base model进行标准指令微调（SFT），不需要额外的RLHF或DPO训练
- 合成数据格式为标准 (指令, 参考代码) 对

## 实验

### 主实验：与SOTA代码LLM对比

| 模型 | HumanEval | MBPP | EvalPlus | LiveCodeBench | BigCodeBench |
|------|-----------|------|----------|---------------|-------------|
| Qwen2.5-Coder-Instruct (百万级) | 基线 | 基线 | 基线 | 基线 | 基线 |
| **ToE (75K)** | **>=基线** | **>=基线** | **~=基线** | **~=/>=基线** | **~=基线** |
| Code Evol-Instruct (75K) | <基线 | <基线 | <基线 | <基线 | <基线 |
| OSS-Instruct (75K) | <基线 | <基线 | <基线 | <基线 | <基线 |

核心亮点：以约**1/15~1/50的数据量**（75K vs 百万级），在5个主流代码benchmark上全面达到或超越SOTA。

### 消融实验

| 消融维度 | 结论 |
|----------|------|
| 树结构 vs 链式进化 | 树结构在数据多样性（唯一解+23%）和下游性能上显著更优 |
| 优化驱动 vs 随机进化 | 优化驱动使合成数据测试通过率提升15-20%，直接转化为下游提升 |
| 分支因子=2 vs 4 vs 8 | 分支因子4为最优平衡点，过大分支导致计算成本指数增长收益递减 |
| 树深度=2 vs 3 vs 5 | 深度3-4最优，更深层级的进化数据难度过高且质量下降 |
| 质量筛选阈值 | 严格筛选(top 30%)略优于宽松筛选(top 70%)，数据质量比数量更重要 |

### 关键发现
- 树结构的多样性优势在困难benchmark（如LiveCodeBench）上更加突出——困难题需要更多样的代码模式
- 进化过程中"死胡同"比例：随机进化约35%，优化驱动仅约12%
- ToE生成的75K数据在代码模式覆盖度上接近人工curated的500K数据集

## 亮点
- **数据效率极高**: 75K数据微调即达百万级SOTA水平，对资源受限的研究者极为实用
- **树结构思想自然且优雅**: 指令进化天然可以多路径分叉，树结构比链式更好地捕捉了这一特性
- **优化驱动避免资源浪费**: 每步都有质量反馈，避免大量低质量数据的后期筛选开销
- **通用框架**: 树结构+优化驱动可推广到其他领域的数据合成（数学推理、对话系统等）

## 局限性
- 树结构探索的计算成本随分支因子指数增长，需合理剪枝策略控制总预算
- 质量评估依赖代码可执行性测试，对非代码领域需要新的质量信号源
- 未探索更大规模（500K+）数据合成时的性能scaling行为
- 种子指令的选择策略未详细讨论，不同种子可能导致不同的进化树质量
- 仅在Qwen系列base model上验证，跨模型泛化性有待确认

## 相关工作
- **vs Code Evol-Instruct**: 链式单向进化+随机方向选择，ToE改为树结构+质量引导，数据效率显著更高
- **vs OSS-Instruct**: 从开源代码提取种子后单向进化，ToE的树结构可从少量种子生成更丰富变体
- **vs Self-Instruct**: 通用指令合成框架，未针对代码领域的质量评估进行特化
- **vs Qwen2.5-Coder-Instruct**: 需百万级数据+多阶段训练，ToE以1/15数据量实现可比性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 树结构进化+优化驱动的组合思路新颖，但单个组件并非全新
- 实验充分度: ⭐⭐⭐⭐ 5个主流benchmark全面评估，消融实验覆盖主要设计选择
- 写作质量: ⭐⭐⭐⭐ 从单向链到多路径树的递进逻辑自然清晰
- 对我的价值: ⭐⭐⭐⭐ 数据高效合成的通用思路可迁移到其他领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
