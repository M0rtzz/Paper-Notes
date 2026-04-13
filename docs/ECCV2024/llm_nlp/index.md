---
title: >-
  ECCV2024 LLM/NLP方向 7篇论文解读
description: >-
  7篇ECCV2024 LLM/NLP方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM/NLP

**🎞️ ECCV2024** · 共 **7** 篇

**[Adaclip Adapting Clip With Hybrid Learnable Prompts For Zero](adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)**

:   在CLIP中同时引入静态（全局共享）和动态（逐图生成）两种可学习提示，用辅助异常检测数据训练后，在14个工业+医学异常检测数据集上实现零样本SOTA，核心在于"任务级+实例级"双层自适应的混合提示设计。

**[Funqa Towards Surprising Video Comprehension](funqa_towards_surprising_video_comprehension.md)**

:   构建了大规模反直觉视频问答基准 FunQA（4.3K 视频、312K QA 对），覆盖幽默/创意/魔术三类令人惊讶的视频，并提出 FunMentor 智能体通过多轮对话增强 VLM 的反常识推理能力。

**[Promptiqa Boosting The Performance And Generalization For No-Reference Image Qua](promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)**

:   提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

**[Propose Assess Search Harnessing Llms For Goal-Oriented Planning In Instructiona](propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)**

:   VidAssist提出"提议-评估-搜索"三步框架，利用LLM作为知识库和评估工具，结合广度优先搜索算法，在教学视频的目标导向规划任务中以零/少样本方式超越全监督SOTA，few-shot在COIN上比全监督VLaMP高+7.7% SR。

**[Reprojection Errors As Prompts For Efficient Scene Coordinate Regression](reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)**

:   本文提出 EGFS（Error-Guided Feature Selection）机制，利用低重投影误差区域作为 SAM 的 point prompts 扩展为语义掩码，迭代地筛选可靠训练样本，在 Cambridge Landmarks 和 Indoor6 数据集上以更小模型和更少训练时间超越现有无 3D 信息依赖的 SCR 方法。

**[Stripe Observation Guided Inference Cost-Free Attention Mechanism](stripe_observation_guided_inference_cost-free_attention_mechanism.md)**

:   本文通过深入分析Transformer中注意力权重矩阵的条纹（stripe）模式现象，提出一种推理阶段完全无额外计算开销的注意力增强机制——仅在训练阶段通过辅助模块学习条纹引导的注意力修正，并在推理时将其重参数化融入标准注意力权重中，实现"免费午餐"式的性能提升。

**[Zeroshot Object Counting With Good Exemplars](zeroshot_object_counting_with_good_exemplars.md)**

:   提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。
