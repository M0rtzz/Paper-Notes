---
title: >-
  ICML2025 视频理解方向 7篇论文解读
description: >-
  7篇ICML2025 视频理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🧪 ICML2025** · **7** 篇论文解读

**[FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks](fastcav_efficient_computation_of_concept_activation_vectors_for_explaining_deep_.md)**

:   提出 FastCAV，通过计算概念样本激活的归一化均值向量来替代 SVM 训练提取概念激活向量（CAV），在理论上等价于 Fisher 判别分析的简化形式，实测加速高达 63.6 倍（平均 46.4 倍），同时保持与 SVM-CAV 相当的分类精度和下游解释质量。

**[Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)**

:   提出 SGVC 框架，通过将视频各段的文本描述解析为场景图、用 Hungarian 算法迭代合并为统一图表示、再用轻量图到文本解码器生成视频级描述，以极低计算开销实现了超越 LLM-based 方法的零样本长视频描述性能。

**[MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition](moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition.md)**

:   提出 MoMa 框架，通过序列调制操作 (SeqMod) 将 Mamba 的线性复杂度 SSM 以 scale-bias 方式注入冻结的 CLIP Transformer，实现高效全时空动态建模，在多个视频识别基准上以更少计算量达到 SOTA 水平。

**[Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs](parity_requires_unified_input_dependence_and_negative_eigenvalues_in_ssms.md)**

:   从理论上证明了线性SSM（如S4/Mamba）无法计算奇偶校验(parity)函数——即使允许输入依赖参数化——除非状态转移矩阵包含负特征值，为SSM的表达力瓶颈提供了精确的数学刻画。

**[REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization](revolve_optimizing_ai_systems_by_tracking_response_evolution_in_textual_optimiza.md)**

:   REVOLVE 通过跟踪 LLM 系统中响应在迭代过程中的"演化"趋势来指导优化，比 TextGrad 等基于即时反馈的方法更稳定高效，在提示优化、方案改进和代码优化上分别提升 7.8%、20.72% 和 29.17%。

**[Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](scaling_video-language_models_to_10k_frames_via_hierarchical_differential_distil.md)**

:   ViLaMP 提出差分蒸馏 (Differential Distillation) 原则，通过层次化的帧级差分关键帧选择 (DKS) 和 patch 级差分特征融合 (DFM) 两种机制实现"混合精度"视频处理——关键帧保留全部视觉 token，非关键帧压缩为单个 token，成功在单张 A100 GPU 上处理长达 10K 帧（约 2.7 小时）的超长视频。

**[Unifying Specialized Visual Encoders for Video Language Models](unifying_specialized_visual_encoders_for_video_language_models.md)**

:   MERV 提出了多编码器视频表示方法，将四种专长不同的视觉编码器（DINOv2、ViViT、SigLIP、LanguageBind）通过时空对齐和跨注意力融合整合到单一 VideoLLM 中，在视频推理基准上比基线 Video-LLaVA 提升最高 4.62%，并验证了不同编码器的互补专长。
