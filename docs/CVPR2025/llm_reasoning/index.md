---
title: >-
  CVPR2025 LLM推理方向 3篇论文解读
description: >-
  3篇CVPR2025 LLM推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM推理

**📷 CVPR2025** · 共 **3** 篇

**[Cot-Vla Visual Chain-Of-Thought Reasoning For Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   提出 CoT-VLA，将视觉思维链推理引入视觉-语言-动作模型，通过两阶段推理——先预测子目标图像再生成动作序列——结合混合注意力和动作分块策略，在 LIBERO 基准上实现 81.13% 平均成功率，显著超越现有方法。

**[Interleaved-Modal Chain-Of-Thought](interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Style Evolving Along Chain-Of-Thought For Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。
