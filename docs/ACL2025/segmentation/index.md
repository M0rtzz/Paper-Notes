---
title: >-
  ACL2025 语义分割方向 3篇论文解读
description: >-
  3篇ACL2025 语义分割方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**💬 ACL2025** · **3** 篇论文解读

**[DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)**

:   提出 DEF-DTS，一种基于 LLM 多步演绎推理的对话话题分割方法——通过双向上下文摘要 → 话语意图分类（5 类） → 演绎话题转移判断三步 pipeline，在 TIAGE、SuperDialseg、Dialseg711 三个数据集上取得无监督/prompt 方法 SOTA，在 Dialseg711 上超越监督方法。

**[InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)**

:   提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

**[Pixel-Level Reasoning Segmentation via Multi-turn Conversations](pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)**

:   提出像素级推理分割 (Pixel-level RS) 新任务，通过多轮对话逐步理解用户意图实现细粒度分割，构建了包含 24k 对话轮次的 PRIST 数据集，并设计 MIRAS 框架在分割精度和推理能力上均超越现有基线。
