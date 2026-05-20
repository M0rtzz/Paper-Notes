---
title: >-
  AAAI2026 目标检测方向15篇论文解读
description: >-
  15篇AAAI2026的目标检测方向论文解读，涵盖目标检测、遥感、目标跟踪等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "遥感"
  - "目标跟踪"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🤖 AAAI2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/object_detection/index.md) · [📷 CVPR2026 (36)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×8 · 遥感 ×3 · 目标跟踪 ×2

**[AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)**

:   构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

**[An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)**

:   提出一个实时大米品质评估整体机制，整合改进的 YOLO-v5（品种检测）、改进的 ConvNeXt-Tiny（完整度分级）和 K-means（垩白区域量化）三个模块，在自建的六品种两万张图像数据集上实现了 99.14% mAP 和 97.89% 检测准确率。

**[Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](beyond_boundaries_leveraging_vision_foundation_models_for_so.md)**

:   提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

**[Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](connecting_the_dots_training-free_visual_grounding_via_agent.md)**

:   提出 GroundingAgent，一个完全不需要任务特定微调的视觉定位框架，通过组合预训练的开放词汇检测器（YOLO World）、MLLM（Llama-3.2-11B-Vision）和 LLM（DeepSeek-V3）进行结构化迭代推理，在 RefCOCO/+/g 上实现 65.1% 的零样本平均准确率，大幅超越之前的 zero-shot 方法。

**[LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](lampq_towards_accurate_layer-wise_mixed_precision_quantization_for_vision_transf.md)**

:   本文提出 LampQ，一种基于度量（metric-based）的逐层混合精度量化方法，通过类型感知的 Fisher 信息度量衡量 ViT 各层对量化的敏感度，结合整数线性规划优化比特宽度分配并迭代更新，在图像分类、目标检测和零样本量化等多个任务上取得 SOTA 性能。

**[REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)**

:   将 DiffusionDet 的 2D BBox 扩散范式提升到 3D 雷达空间，提出 REXO 框架：通过含噪 3D BBox 的投影引导显式跨视图雷达特征关联，并引入地面约束减少扩散参数，在 HIBER 和 MMVR 两个室内雷达数据集上分别超越 SOTA +4.22 AP 和 +11.02 AP。

**[SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)**

:   提出SAGA方法，通过学习与提示词对齐的高斯分布来改进文本到图像生成模型的语义对齐，无需重新训练且支持文本和空间双条件生成，在SD 1.4和SD 3上大幅提升对齐性能（TIAM-3从8.4%提升到50.7%）。

**[SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements](simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)**

:   提出SimROD，一种极其轻量（仅0.003M参数）的RAW图像目标检测方法，通过全局Gamma增强（4个可学习参数）和绿色通道引导的局部增强，在多个RAW检测基准上超越了复杂的SOTA方法。

**[SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)**

:   SM3Det提出了遥感领域的M2Det新任务（多模态数据集+多任务目标检测），通过网格级稀疏MoE骨干网络和动态子模块优化（DSO）机制，用单一模型同时处理SAR/光学/红外三种模态的水平/旋转框检测，显著超越各模态独立训练的三个专用模型组合。

**[T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)**

:   提出T-Rex-Omni框架，首次将负视觉提示（negative visual prompts）系统性地引入开放集目标检测，通过训练免费的NNC模块和NNH损失，显著缩小了视觉提示和文本提示检测方法之间的性能差距，在长尾场景中表现尤为突出（LVIS-minival APr达到51.2）。

**[Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)**

:   提出一种对象感知的时序建模框架，通过选择性传播高置信度检测特征实现跨帧时序一致性，结合预训练视觉-语言编码器（OWL-ViT）和少样本检测头，在四个视频少样本检测基准上平均提升3.7%-5.3% AP。

**[TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)**

:   提出 TubeRMC 框架，利用文本条件化的候选 tube 生成 + 从时间/空间/时空三个维度进行 tube 条件化重建，并引入空间-时间互约束来增强弱监督时空视频定位性能。

**[VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)**

:   提出 VK-Det 框架，仅利用 VLM 的视觉知识（无需额外监督信号），通过自适应选择知识蒸馏（ASKD）+ 原型感知伪标签（PAPL）+ 综合匹配推理（SMI），在航空遥感开放词汇目标检测中达到 SOTA，甚至超越使用额外监督的方法。

**[When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)**

:   提出 MFT25 大规模水下多鱼跟踪数据集（15 序列, 408K 标注）和 SU-T 跟踪框架（UKF + FishIoU），实现 34.1 HOTA 和 44.6 IDF1 的 SOTA 性能，并通过统计分析揭示鱼类跟踪与陆地目标跟踪的本质差异。

**[YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)**

:   首次系统性地将增量目标检测（IOD）引入 YOLO 实时框架，识别三种知识冲突类型，提出 CPR + IKS + CAKD 三模块协同解决方案，并引入更真实的 LoCo COCO 基准评估。
