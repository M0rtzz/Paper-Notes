---
title: >-
  AAAI2026 因果推理方向10篇论文解读
description: >-
  10篇AAAI2026的因果推理方向论文解读，涵盖 LLM、多模态、推理、Agent、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
  - "推理"
  - "Agent"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🤖 AAAI2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (3)](../../ICML2026/causal_inference/index.md) · [💬 ACL2026 (6)](../../ACL2026/causal_inference/index.md) · [📷 CVPR2026 (3)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/causal_inference/index.md) · [🧠 NeurIPS2025 (21)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

**[CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)**

:   提出 CaDyT，结合高斯过程连续时间动力学建模（Adams-Bashforth 积分器实现精确推断）和 MDL 最小描述长度原则进行结构搜索，同时解决不规则采样和因果结构识别两个挑战，在双质点弹簧/菱形图/Rössler 振荡器上大幅超越所有基线（AUPRC 0.79 vs 次优 0.39）。

**[Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[From Theory of Mind to Theory of Environment: Counterfactual Simulation of Latent Environmental Dynamics](from_theory_of_mind_to_theory_of_environment_counterfactual_simulation_of_latent.md)**

:   本文提出"环境理论"（Theory of Environment）概念，认为人类可能通过与心智理论（Theory of Mind）共享的计算机制来推断环境中隐含的动态规律，从而扩展运动探索的维度空间并促进行为创新。

**[Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)**

:   提出 I-CAM-UV 方法，通过对多个变量集不同的 CAM-UV 因果图结果进行一致性约束枚举，恢复因未观测变量而丢失的因果关系，并设计基于不一致代价单调性的最优优先搜索算法高效求解。

**[KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)**

:   提出 KTCF，一种面向知识追踪（KT）的反事实解释生成方法，通过考虑知识概念间关系生成稀疏且可操作的反事实解释，并将其后处理为顺序化的教学指令，在有效性、稀疏性和可操作性指标上全面超越基线方法。

**[Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)**

:   在 SCM 框架下证明最大处理效应子群必须具有同质点效应（定理1），在分区模型假设下证明最优子群发现可化简为标准监督学习（定理2），用 CART+Gini 指数即可实现——在 77 个 ACIC-2016 半合成数据集上均值处理效应 10.54（vs 次优 7.84），51.9% 排名第一。

**[MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

**[Skill Path: Unveiling Language Skills from Circuit Graphs](skill_path_unveiling_language_skills_from_circuit_graphs.md)**

:   提出 Skill Path 概念及三步框架（分解-剪枝-因果中介），从电路图中提取语言模型特定技能的线性路径，定量验证了技能的分层性（Stratification）和包容性（Inclusiveness）两大猜想。

**[Sparse Additive Model Pruning for Order-Based Causal Structure Learning](sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)**

:   提出 SARTRE 框架，利用随机化树嵌入与组稀疏回归学习稀疏加性模型，替代 CAM-pruning 中基于假设检验的冗余边修剪，在基于拓扑序的因果结构学习中实现显著加速且精度不降。
