---
title: >-
  CVPR2025 LLM推理方向 7篇论文解读
description: >-
  7篇CVPR2025 LLM推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM推理

**📷 CVPR2025** · **7** 篇论文解读

**[Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)**

:   Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。

**[CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   提出 CoT-VLA，将视觉思维链推理引入视觉-语言-动作模型，通过两阶段推理——先预测子目标图像再生成动作序列——结合混合注意力和动作分块策略，在 LIBERO 基准上实现 81.13% 平均成功率，显著超越现有方法。

**[Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework](learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte.md)**

:   提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。

**[Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval](reason-before-retrieve_one-stage_reflective_chain-of-thoughts_for_training-free_.md)**

:   OSrCIR提出单阶段反思性链式思维推理，让MLLM同时处理参考图像和修改文本（避免两阶段caption→推理的信息丢失），通过"描述→思考→反思→目标描述"四步CoT生成准确的目标图像描述，在CIRCO上mAP@5达23.87%超越CIReVL 26.2%，且完全免训练。

**[Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。

**[VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)**

:   VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。
