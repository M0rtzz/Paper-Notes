---
title: >-
  CVPR2025 音频/语音方向 4篇论文解读
description: >-
  4篇CVPR2025 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2025** · 共 **4** 篇

**[Contextual Ad Narration With Interleaved Multimodal Sequence](contextual_ad_narration_with_interleaved_multimodal_sequence.md)**

:   提出 Uni-AD 统一框架，以交错多模态序列（视频特征+文本+角色库+上下文）作为输入，通过视觉映射网络对齐特征 + 角色精化模块识别主要角色 + 对比损失增强上下文一致性，在 MAD-eval-Named 上达到 SOTA。

**[Crab A Unified Audio-Visual Scene Understanding Model With Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)**

:   提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

**[Distinctad Distinctive Audio Description Generation In Contexts](distinctad_distinctive_audio_description_generation_in_contexts.md)**

:   生成上下文中有区分度的音频描述（AD），避免生成泛化无特色的描述，通过对比学习鼓励与前后AD的差异性

**[Team Leya In 10Th Abaw Competition Multimodal Ambivalencehesitancy Recognition A](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   本文提出面向视频级矛盾/犹豫（A/H）识别的多模态方法，整合场景（VideoMAE）、面部（EmotionEfficientNetB0）、音频（EmotionWav2Vec2.0+Mamba）和文本（EmotionDistilRoBERTa）四种模态，通过原型增强的 Transformer 融合模型实现 83.25% 平均 MF1，最终以五模型集成在测试集达到 71.43%。
