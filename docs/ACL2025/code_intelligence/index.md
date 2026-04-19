---
title: >-
  ACL2025 代码智能方向 20篇论文解读
description: >-
  20篇ACL2025 代码智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**💬 ACL2025** · **20** 篇论文解读

**[LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)**

:   提出 LongCodeU 基准，从代码单元感知、单元内理解、单元间关系理解和长文档理解四个维度设计 8 个任务，评估 9 个长上下文语言模型在真实仓库级长代码上的理解能力，揭示 32K token 是当前 LCLM 长代码理解的实际上限。

**[CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)**

:   提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。

**[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)**

:   提出 CodeReviewQA 基准，将代码审查自动修正（ACR）任务分解为三个中间推理步骤——变更类型识别（CTR）、变更定位（CL）、解决方案识别（SI），各自设计为不同难度的多选题探测，在 900 个人工验证的高质量样例（9 种语言）上评测 72 个 LLM，揭示了模型在代码审查理解中的具体弱点。

**[CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)**

:   提出 CompileAgent，首个面向仓库级代码编译的 LLM Agent 框架，集成五种专用工具和流程化 Agent 策略，在 100 个 C/C++ 真实项目的 CompileAgentBench 上将编译成功率最高提升 71%，平均每个项目仅需 $0.22。

**[CoRet: Improved Retriever for Code Editing](coret_improved_retriever_for_code_editing.md)**

:   提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。

**[DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)**

:   提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。

**[ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)**

:   提出 Entity Tracing Framework (ETF)，一种通过静态程序分析提取代码实体、再用 LLM 验证这些实体在生成摘要中是否被正确描述的幻觉检测框架，配合首创的 CodeSumEval 数据集（~10K样本），在代码摘要幻觉检测上达到 73% F1。

**[ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration](exploracoder_advancing_code_generation_for_multiple_unseen_apis_via_planning_and.md)**

:   提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。

**[FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)**

:   本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。

**[OASIS: Order-Augmented Strategy for Improved Code Search](oasis_order-augmented_strategy_for_improved_code_search.md)**

:   提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。

**[Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)**

:   用 GPT-4o 为每个编程任务动态生成适配的 MBTI 人格类型和详细描述，再让目标 LLM 以该人格角色扮演程序员生成代码，在 7 个 LLM × 4 个数据集的 28 个组合中 23 个取得 pass rate 提升，最高达 12.9%，关键因素是人格多样性而非某个特定人格。

**[Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)**

:   系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。

**[TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)**

:   提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。

**[Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)**

:   提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。

**[Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构代码指令合成框架，通过多路径进化和质量驱动优化克服Code Evol-Instruct和OSS-Instruct的单向合成与随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)**

:   本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。
