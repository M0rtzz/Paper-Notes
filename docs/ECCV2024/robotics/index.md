---
title: >-
  ECCV2024 机器人/具身智能方向 10篇论文解读
description: >-
  10篇ECCV2024 机器人/具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🎞️ ECCV2024** · 共 **10** 篇

**[Aff-Ttention Affordances And Attention Models For Short-Term Object Interaction ](aff-ttention_affordances_and_attention_models_for_short-term_object_interaction_.md)**

:   提出 STAformer 架构和两个基于 affordance 的模块（环境 affordance 数据库 + 交互热点），将第一人称视频中的短期物体交互预测（STA）在 Ego4D 和 EPIC-Kitchens 上提升了 30-45% 的相对性能。

**[Disco Embodied Navigation And Interaction](disco_embodied_navigation_and_interaction.md)**

:   提出 DISCO，通过可微分场景语义表征（包含物体和 affordance）实现动态场景建模，结合全局-局部双层粗到细控制策略实现高效移动操作，在 ALFRED benchmark 的 unseen scenes 上以 +8.6% 成功率超越使用分步指令的 SOTA，且无需分步指令。

**[Disco Embodied Navigation And Interaction Via Differentiable Scene Semantics And](disco_embodied_navigation_and_interaction_via_differentiable_scene_semantics_and.md)**

:   提出 DISCO 框架，通过可微分场景语义表示和双层粗-细动作控制，在 ALFRED 基准上实现具身导航与交互的显著性能提升（未见场景成功率超越 SOTA +8.6%，且无需逐步指令）。

**[Hierarchically Structured Neural Bones For Reconstructing Animatable Objects Fro](hierarchically_structured_neural_bones_for_reconstructing_animatable_objects_fro.md)**

:   提出层次化神经骨骼（Hierarchical Neural Bones）框架，通过树状结构的骨骼系统以粗到细的方式分解物体运动，从随手拍摄的视频中重建可操控的高质量 3D 模型。

**[Prioritized Semantic Learning For Zero-Shot Instance Navigation](prioritized_semantic_learning_for_zero-shot_instance_navigation.md)**

:   提出Prioritized Semantic Learning (PSL)方法，通过语义增强的Agent架构、优先语义训练策略和语义扩展推理方案，显著提升零样本目标/实例导航中Agent的语义感知能力，在ObjectNav和新提出的InstanceNav任务上实现SOTA。

**[Prioritized Semantic Learning For Zeroshot Instance Navigation](prioritized_semantic_learning_for_zeroshot_instance_navigation.md)**

:   提出 Prioritized Semantic Learning (PSL) 方法，通过语义感知智能体架构、优先语义训练策略和语义扩展推理方案，显著提升导航智能体的语义感知能力，在零样本 ObjectNav 上超越 SOTA  66%（SR），并提出了更具挑战性的 InstanceNav 任务。

**[Realfred An Embodied Instruction Following Benchmark In Photo-Realistic Environm](realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)**

:   提出 ReALFRED 基准，使用 150 个真实世界 3D 扫描的多房间可交互环境替代 ALFRED 的合成单房间场景，提供 30,696 条自由格式语言指令，揭示了现有具身指令跟随方法在真实环境中性能显著下降的问题。

**[See And Think Embodied Agent In Virtual Environment](see_and_think_embodied_agent_in_virtual_environment.md)**

:   提出 STEVE，一个基于视觉感知、语言指令和代码动作三大组件的 Minecraft 开放世界具身智能体，通过 STEVE-21K 数据集微调 LLaMA-2 并结合视觉编码器和技能数据库，在科技树解锁和方块搜索任务上大幅超越现有方法。

**[Semgrasp Semantic Grasp Generation Via Language Aligned](semgrasp_semantic_grasp_generation_via_language_aligned.md)**

:   提出 SemGrasp，通过层次化 VQ-VAE 将抓取姿态离散化为三个语义对齐的 token（方向/方式/精修），并微调多模态大语言模型实现基于语言指令的语义抓取生成。

**[Semgrasp Semantic Grasp Generation Via Language Aligned Discretization](semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)**

:   提出SemGrasp方法，设计层次化VQ-VAE将抓取姿态离散为"方向-方式-精修"三个语义token，然后微调多模态大语言模型(MLLM)在统一语义空间中融合物体、抓取与语言，实现根据自然语言指令生成物理合理且语义一致的人类抓取姿态。
