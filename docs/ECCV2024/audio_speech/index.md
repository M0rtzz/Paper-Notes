---
title: >-
  ECCV2024 音频/语音方向 9篇论文解读
description: >-
  9篇ECCV2024 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🎞️ ECCV2024** · 共 **9** 篇

**[Beat-It Beat-Synchronized Multi-Condition 3D Dance Generation](beat-it_beat-synchronized_multi-condition_3d_dance_generation.md)**

:   提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。

**[Coleaf A Contrastive-Collaborative Learning Framework For Weakly Supervised Audi](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)**

:   提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。

**[Controlllm Augment Language Models With Tools](controlllm_augment_language_models_with_tools.md)**

:   提出 ControlLLM 框架，通过任务分解、Thoughts-on-Graph (ToG) 图搜索范式和执行引擎三大组件，让 LLM 在预构建的工具图上搜索最优解决方案路径，准确高效地调用多模态工具完成复杂任务，在困难任务上达到 93% 的解决方案成功率。

**[Controlllm Augment Language Models With Tools By Searching On Graphs](controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)**

:   提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

**[Edtalk Efficient Disentanglement For Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)**

:   提出基于正交可学习基向量的高效解耦框架 EDTalk，将人脸动态分解为嘴型、头部姿态和情感表情三个独立潜空间，同时支持视频驱动和音频驱动的情感说话人头像生成。

**[Label-Anticipated Event Disentanglement For Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)**

:   提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。

**[Latent-Inr A Flexible Framework For Implicit Representations Of Videos With Disc](latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)**

:   提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。

**[Listen To Look Into The Future Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)**

:   提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。

**[Siamese Vision Transformers Are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)**

:   提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。
