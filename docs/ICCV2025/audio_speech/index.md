---
title: >-
  ICCV2025 音频/语音方向 10篇论文解读
description: >-
  10篇ICCV2025 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📹 ICCV2025** · **10** 篇论文解读

**[2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](25_years_in_class_a_multimodal_textbook_for_vision-language_pretraining.md)**

:   从YouTube教学视频中提取关键帧和文本（ASR+OCR），构建高质量交错图文格式的"多模态教材"数据集，用于VLM预训练，在知识密集型和推理任务上大幅领先网页爬取的交错数据集。

**[2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)**

:   从YouTube收集2.5年(22,000课时)的教学视频，通过LLM驱动的多级抽取与过滤管线构建高质量交错图文"多模态教科书"语料(6.5M关键帧 + 0.75B文本token)，显著提升VLM在知识密集型和推理任务上的预训练效果，尤其在ScienceQA和MathVista上带来大幅提升。

**[How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)**

:   提出材质可控的声学特征生成任务（M-CAPA），给定室内场景的音视觉观测和用户定义的新材质配置，生成反映材质变化的目标房间脉冲响应（RIR），并构建了配套的 Acoustic Wonderland 数据集。

**[Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)**

:   提出SaFa（Swap Forward），一种模态无关的高效方法，通过两种潜空间交换算子（Self-Loop Latent Swap和Reference-Guided Latent Swap）替代传统联合扩散中的均值化操作，解决频谱混叠问题并保持跨视图一致性，在长音频和全景图生成中显著优于现有方法。

**[Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](lyra_an_efficient_and_speech-centric_framework_for_omni-cognition.md)**

:   提出 Lyra，一个以语音为中心的高效全模态 MLLM 框架，通过多模态 LoRA、潜在跨模态正则化器和潜在多模态提取器三大策略，使用更少的训练数据实现视觉-语言-语音多模态的 SOTA 性能，并首次支持长达数小时的语音输入。

**[Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](lyra_an_efficient_and_speechcentric_framework_for_omnicognit.md)**

:   提出Lyra，一个以语音为中心的全模态MLLM框架，通过三大核心组件（DTW-based跨模态正则化器、多模态LoRA、Latent多模态提取器）和首个12K长语音SFT数据集，在仅用2.7M数据和少量训练的情况下，同时在视觉-语言、视觉-语音、语音-语言benchmark上达到SOTA，并能处理长达2小时的语音输入。

**[MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)**

:   提出 MemoryTalker，通过两阶段训练策略（Memorizing + Animating）利用键值记忆网络存储通用面部运动，并通过音频驱动的风格化记忆实现仅凭音频即可生成个性化 3D 面部动画，无需任何额外先验信息。

**[MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)**

:   提出MUG框架，通过伪标签增强的跨模态随机组合数据增强策略和音视频Mamba网络，同时提升弱监督音视频解析任务中段级和事件级的预测性能。

**[VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)**

:   针对 VGGSound 数据集在多标签缺失、类别重叠和模态错位方面的局限性，构建了 VGGSounder——一个带有模态标注的多标签音视频分类基准，并提出"模态混淆"度量来揭示基础模型在多模态融合上的不足。

**[Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)**

:   提出 Zero-AVSR 框架，通过将语音转写为语言无关的罗马化文本（Roman text），再利用 LLM 将罗马文本转换为目标语言文字，实现无需目标语言语音数据的零样本视听语音识别，并构建了覆盖 82 种语言、2916 小时的 MARC 数据集。
