---
title: >-
  ICCV2025 LLM Agent方向 4篇论文解读
description: >-
  4篇ICCV2025 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**📹 ICCV2025** · **4** 篇论文解读

**[Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions](embodied_image_captioning_self-supervised_learning_agents_for_spatially_coherent.md)**

:   提出一个三阶段自监督框架，通过agent自主导航收集多视角观测、LLM共识机制生成伪标注、对比学习微调captioner，显著提升室内环境中同一物体跨视角描述的一致性和准确性。

**[GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)**

:   发现RL训练VLM Agent时的"思维坍塌"现象——CoT推理迅速退化为与状态无关的模板化思维并导致无效动作，提出GTR框架用VLM纠正器自动修正思维(SFT) + PPO优化动作的双目标训练，在24点游戏和ALFWorld上实现3-5倍的成功率提升。

**[Less is More: Empowering GUI Agent with Context-Aware Simplification](less_is_more_empowering_gui_agent_with_context-aware_simplification.md)**

:   提出 SimpAgent——一种上下文感知的简化框架，通过基于遮挡的元素剪枝（训练时随机遮挡无关元素区域）和一致性引导的历史压缩（在 LLM 中间层直接丢弃历史视觉 token + KL散度一致性约束），在降低27% FLOPs 的同时取得多个 GUI 导航基准的 SOTA。

**[UIPro: Unleashing Superior Interaction Capability for GUI Agents](uipro_unleashing_superior_interaction_capability_for_gui_agents.md)**

:   提出 UIPro，通过构建 2060 万 GUI 理解样本进行预训练并提出统一动作空间整合异构 GUI agent 任务数据，实现跨移动端、Web 端和桌面端的 SOTA GUI 交互性能。
