---
title: >-
  CVPR2026 代码智能方向 4篇论文解读
description: >-
  4篇CVPR2026 代码智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**📷 CVPR2026** · **4** 篇论文解读

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过系统性缩放分析发现感知（perception）而非推理（reasoning）是 MLLM 在 STEM 领域的真正瓶颈，提出以可执行 Python 代码为锚定媒介的 CodePercept 范式——构建 100 万级 ICC-1M 数据集和 STEM2Code-Eval 基准，在 SFT+RL 两阶段训练后显著提升 MLLM 的 STEM 视觉感知和下游推理能力。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_codegrounded_visual_stem_perception_fo.md)**

:   通过系统性缩放分析揭示**感知而非推理**是 MLLM 在 STEM 视觉任务上的真正瓶颈，提出以可执行代码为媒介增强感知能力的范式，构建 100 万级 Image-Caption-Code 三元组数据集 ICC-1M，包含代码锚定的标题生成和 STEM 图到代码翻译两个训练任务。

**[GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)**

:   GeoTikzBridge 通过构建最大的 2.5M 图像-TikZ 代码数据集和首个辅助线指令数据集，训练出能精准重建几何图形的代码生成模型，并可作为即插即用模块增强任意 MLLM/LLM 的几何推理能力。

**[MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)**

:   提出 MM-ReCoder，首个具备自我纠错能力的图表转代码多模态 LLM，通过两阶段多轮 GRPO 强化学习（先共享首轮优化纠错能力，再全轨迹优化编码能力），在 ChartMimic 上以仅 7B 参数达到 86.5% low-level score，媲美 Qwen3-VL-235B。
