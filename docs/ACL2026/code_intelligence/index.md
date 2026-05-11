---
title: >-
  ACL2026 代码智能方向20篇论文解读
description: >-
  20篇ACL2026的代码智能方向论文解读，涵盖代码智能、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**💬 ACL2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/code_intelligence/index.md) · [🔬 ICLR2026 (21)](../../ICLR2026/code_intelligence/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (22)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md) · [🧪 ICML2025 (11)](../../ICML2025/code_intelligence/index.md)

🔥 **高频主题：** 代码智能 ×9 · LLM ×3 · 推理 ×2

**[Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation](across_programming_language_silos_a_study_on_cross-lingual_retrieval-augmented_c.md)**

:   首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。

**[CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)**

:   本文提出 CodeRL+，将执行语义对齐集成到 RLVR 训练管道中，通过让模型推断变量级执行轨迹来弥合代码文本表示与执行语义之间的差距，在代码生成上平均 pass@1 提升 4.6%，在代码推理和测试输出生成基准上分别提升 15.5% 和 4.4%。

**[CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)**

:   提出 CodeWiki，一个基于层次化分解和递归多智能体处理的开源框架，用于自动生成仓库级代码文档，并构建了 CodeWikiBench 基准，在七种编程语言上以 68.79% 的质量分数超越了闭源系统 DeepWiki（64.06%）。

**[CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)**

:   本文提出 CollabCoder，一个计划-代码共演化框架，通过协作决策模块（CDM）判断错误应在计划层还是代码层修复，结合推理轨迹模块（RT）实现从错误中学习的自改进调试，在复杂编程基准上比强基线提升 11-20%，同时减少 4-10 次 API 调用。

**[DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)**

:   提出 DeepGuard，通过注意力机制聚合 Transformer 上层多层表示克服"最终层瓶颈"问题，结合多目标训练和轻量推理时安全引导策略，在 5 个代码 LLM 上将安全-正确生成率平均提升 11.9%。

**[EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)**

:   提出 EET——一种基于历史经验驱动的早停方法，在补丁生成和补丁选择阶段识别无效迭代并提前终止，将 SE Agent 总成本降低 19%-55%（平均 32%），同时几乎不损失任务性能（最多 0.2%）。

**[From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)**

:   本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。

**[From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation](from_if-statements_to_ml_pipelines_revisiting_bias_in_code-generation.md)**

:   揭示LLM代码生成的偏差评估严重低估了实际风险：在ML流水线生成中，敏感属性出现在87.7%的特征选择中（vs 条件语句中的59.2%），且模型能正确排除无关特征但仍选择保留种族、性别等敏感属性，显示出系统性的隐性歧视。

**[LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)**

:   本文构建了首个针对逻辑漏洞的修复评估框架 LogicEval 和数据集 LogicDS（61 个真实逻辑漏洞 + 61 个合成 Java 样本），系统评估了传统 AVR 工具和 LLM 在修复逻辑漏洞上的能力，发现 LLM 在提供辅助信息时表现最佳但整体修复率仍然很低（61 个真实样本中仅正确修复 5 个），并识别了提示敏感性、上下文丢失和补丁定位困难等关键瓶颈。

**[MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)**

:   MARS2 提出多智能体强化树搜索框架，将多个独立优化的策略嵌入共享搜索树中协作探索，通过 Thompson 采样选择智能体-节点对、树一致性奖励塑形和路径级组优势估计，在代码生成基准上一致提升单模型 Pass@1 最高 8.0%、系统级 Pass@1(MCTS) 最高 6.5%。

**[OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)**

:   本文提出 OmniDiagram，一个统一的图表代码生成框架，覆盖 LaTeX/Mermaid/PlantUML 三种语言和图表转代码/图表编辑/文本转代码三种任务，并引入基于视觉问答的 Viva 奖励机制来指导 RL 训练，在多个基准上达到 SOTA。

**[Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)**

:   本文揭示前沿 LLM 在调试任务中的"重生成"倾向——通过引入 PDB 框架和编辑级精度/bug 级召回指标，发现 GPT-5.1-Codex 等模型虽能通过 76% 以上单元测试，但编辑精度不足 45%，且迭代和 agent 调试策略也无法显著改善精度。

**[QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)**

:   本文识别了 LLM 代码修复中的"过度编辑"问题——模型倾向于重写大量代码而非精确定位和修复 bug，提出 PRepair 框架，通过 Self-Breaking（多样化 bug 注入）和 Self-Repairing（编辑感知 GRPO 训练），显著提升修复精确度同时保持正确性，并加速推测解码推理。

**[ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)**

:   本文提出 ReFEree，一种针对真实世界代码摘要的无参考、细粒度事实一致性评估方法，定义四类不一致标准并在句段级别评估，结合依赖信息搜索机制，在 Python 和 Java 上相比前 SOTA 提升 15-18% 的人类判断相关性。

**[River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)**

:   本文提出 River-LLM，一个无需训练的框架，通过构建轻量级 KV 共享退出通道（Exit River）解决了 decoder-only 架构中 Early Exit 的 KV Cache 缺失问题，利用状态转换相似度引导退出决策，实现 1.71×-2.16× 的实际推理加速且保持近无损生成质量。

**[Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)**

:   本文提出区分词汇召回（逐字检索代码）和语义召回（理解代码运行语义）两种能力，发现前沿 LLM 在长上下文中词汇召回近乎完美但语义召回严重退化，并引入 SemTrace 基准揭示现有评估严重低估了语义理解失败的程度。

**[SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)**

:   本文提出 SOCIA-EVO，一种将自动化模拟器构建重新定义为双锚进化过程的 LLM 智能体框架，通过静态蓝图（Blueprint）锚定经验约束、双层优化解耦结构修正与参数校准、自我策划的策略剧本（Playbook）管理修复假说并通过执行反馈进行贝叶斯加权检索，在用户建模、口罩佩戴扩散和个人出行三个模拟任务上显著超越 Reflexion、G-SIM 等基线。

**[SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)**

:   SolidCoder 通过 S.O.L.I.D. 架构（Shift-left Planning、Oracle-based Assertions、Live Execution、Intermediate Simulation、Defensive Accumulation）将代码验证从 LLM 的"想象执行"转变为"真实执行"，在 GPT-4o 上达到 HumanEval 95.7%、CodeContests 77.0%、APPS 26.7% 的 pass@1 性能。

**[StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)**

:   本文提出 StoryCoder，一种将代码生成问题重构为连贯自然语言叙事的提示框架，通过任务概述、约束条件和示例三个叙事组件引导 LLM 进行结构化推理，在 11 个模型上平均提升零样本 pass@10 达 18.7%。

**[The Path Not Taken: Duality in Reasoning about Program Execution](the_path_not_taken_duality_in_reasoning_about_program_execution.md)**

:   本文提出程序执行推理的对偶性概念，通过DexBench基准（445个配对实例）联合评估LLM的正向执行推理（预测给定输入下的代码覆盖）和反向反事实推理（推断使执行流转向目标分支的输入变异），发现单一方向上的强表现不能转化为联合评估下的成功，揭示了模型对程序因果理解的不足。
