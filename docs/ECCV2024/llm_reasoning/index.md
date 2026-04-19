---
title: >-
  ECCV2024 LLM推理方向 2篇论文解读
description: >-
  2篇ECCV2024 LLM推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM推理

**🎞️ ECCV2024** · **2** 篇论文解读

**[Controllable Navigation Instruction Generation with Chain of Thought Prompting](controllable_navigation_instruction_generation_with_chain_of_thought_prompting.md)**

:   提出 C-Instructor，利用 LLM 的思维链提示实现风格和内容可控的导航指令生成，通过 CoTL（带地标的思维链）、STMT（空间拓扑建模）和 SMT（混合风格训练）三大机制，在四个室内外导航数据集上全面超越已有方法。

**[RoadPainter: Points Are Ideal Navigators for Topology Transformer](roadpainter_points_are_ideal_navigators_for_topology_transformer.md)**

:   提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。
