---
title: >-
  ICCV2025 LLM / NLP方向8篇论文解读
description: >-
  8篇ICCV2025的 LLM / NLP 方向论文解读，涵盖 LLM、持续学习、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**📹 ICCV2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (36)](../../ACL2026/llm_nlp/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (39)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_nlp/index.md) · [🧪 ICML2025 (28)](../../ICML2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×3 · 持续学习 ×2

**[Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag.md)**

:   提出Analytic Subspace Routing（Any-SSR）框架，通过为每个任务分配独立的LoRA子空间消除任务间干扰，并利用递归最小二乘（RLS）闭式解训练一个零遗忘的解析路由器，实现LLM的无回放持续学习。

**[Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models.md)**

:   提出Analytic Subspace Routing (Any-SSR)，为每个新任务分配独立的LoRA子空间以消除知识干扰，同时使用基于递归最小二乘(RLS)闭式解的分析路由器动态选择子空间，在理论上保证不遗忘先前任务知识，实现LLM的无重放持续学习。

**[Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)**

:   TITA 提出了一种无需任务标识的统一图像融合框架，通过交互增强像素注意力（IPA）模块探索任务不变的互补信息提取，并通过基于操作的自适应融合（OAF）模块动态适配任务特定需求，同时采用 FAMO 策略缓解多任务梯度冲突。

**[Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)**

:   提出 DiffBrush，首个基于扩散模型的手写文本行生成方法，通过内容解耦的风格学习（列/行掩码）和多尺度内容判别器（行/词级别），在风格模仿和内容准确性上大幅超越现有方法。

**[FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)**

:   将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。

**[ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)**

:   提出ShadowHack框架，将阴影去除分解为亮度恢复和颜色修复两个子任务，通过带有纠偏外展注意力的LRNet恢复亮度和纹理，再用跨注意力驱动的CRNet重建准确颜色，在ISTD+和SRD数据集上取得SOTA。

**[VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)**

:   提出 VA-GPT，一个面向视频异常事件理解的多模态大模型，通过空间有效token选择(SETS)和时间有效token生成(TETG)两个模块，让MLLM在空间和时间维度上精准对齐异常相关信息，在域内和跨域异常检测基准上均达到SOTA。

**[VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)**

:   提出 VIM，首个能在统一框架内同时理解和生成双人交互运动与文本的多模态大模型，配合82.7K多轮交互运动指令数据集 Inter-MT²，支持文本到运动、运动到文本、反应生成、运动编辑和运动推理等多种任务。
