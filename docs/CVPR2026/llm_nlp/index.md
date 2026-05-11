---
title: >-
  CVPR2026 LLM / NLP方向9篇论文解读
description: >-
  9篇CVPR2026的 LLM / NLP 方向论文解读，涵盖少样本学习、异常检测、LLM、推理、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**📷 CVPR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (36)](../../ACL2026/llm_nlp/index.md) · [🔬 ICLR2026 (39)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_nlp/index.md) · [🧪 ICML2025 (28)](../../ICML2025/llm_nlp/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo](bi_cmpstereo_bidirectional_cross_modal_prompting_for_event_frame_asymmetric_stereo.md)**

:   提出 Bi-CMPStereo，一种双向跨模态提示框架，交替将事件和帧设为目标域进行立体规范化约束和跨域嵌入适配，同时利用两个方向的代价体实现鲁棒的事件-帧非对称立体匹配。

**[Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)**

:   提出QICA框架解决零样本目标计数中的数量感知缺失和空间不敏感问题，通过数量条件化的协同提示策略（SPS）联合适配视觉-语言编码器，结合在相似度图上直接操作的代价聚合解码器（CAD）保持零样本迁移能力，在FSC-147上达到零样本SOTA（MAE 12.41）并展现强跨域泛化。

**[Composing Concepts from Images and Videos via Concept-prompt Binding](composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)**

:   提出 Bind & Compose (BiCo)，一种one-shot方法，通过层次化binder结构将视觉概念绑定到prompt token，并通过token组合实现图像-视频概念的灵活组合，在概念一致性、prompt保真度和运动质量上全面超越前作。

**[CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)**

:   本文提出 CoPS 框架，通过显式状态token合成（ESTS）和隐式类别token采样（ICTS）两种视觉条件化机制动态生成提示，配合空间感知对齐（SAGA），在13个工业和医学数据集上实现零样本异常检测SOTA。

**[GUIDE: Guided Updates for In-context Decision Evolution in LLM-Driven Spacecraft Operations](guide_guided_updates_for_in-context_decision_evolution_in_llm-driven_spacecraft_.md)**

:   提出GUIDE框架，利用LLM的in-context学习能力为航天器自主操作提供引导式决策进化，通过结构化的上下文信息和反馈机制让LLM在无需微调的情况下逐步改善航天任务规划和故障诊断决策的质量。

**[Perception Programs: Unlocking Visual Tool Reasoning in Language Models](perception_programs_visual_tool_reasoning.md)**

:   提出 Perception Programs (P2)，一种训练免费、模型无关的方法，将视觉工具（深度、光流、对应等）的原始输出转换为紧凑的语言原生结构化摘要，使 MLLM 能直接"阅读"视觉模态而非从密集像素推断，在 BLINK 6 个任务上平均提升 19.66%。

**[PhysVid: Physics Aware Local Conditioning for Generative Video](physvid_physics_aware_local_conditioning_for_generative_video_models.md)**

:   提出 PhysVid，一种物理感知的局部条件化方案——将视频分为时间片段（chunk），由 VLM 为每个 chunk 标注物理现象描述，通过 chunk 级交叉注意力注入生成模型；推理时引入"负物理提示"（反事实引导）引导生成远离物理违规，在 VideoPhy 上将物理常识分数提升约 33%。

**[Sign Language Recognition in the Age of LLMs](sign_language_recognition_llms.md)**

:   首个系统评估现代 VLM 在零样本孤立手语识别（ISLR）上能力的研究，发现开源 VLM 远落后于专用分类器，但大型商用模型（GPT-5）展现出令人惊讶的潜力。

**[SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)**

:   提出SketchDeco，一种无需训练的线稿上色方法，通过全局-局部两阶段策略将区域蒙版和调色板作为精确控制信号，利用扩散模型反演和自注意力注入在隐空间中实现区域精准着色与全局和谐过渡，在消费级GPU上15-20步即可完成。
