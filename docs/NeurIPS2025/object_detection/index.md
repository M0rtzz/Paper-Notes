---
title: >-
  NeurIPS2025 目标检测方向17篇论文解读
description: >-
  17篇NeurIPS2025的目标检测方向论文解读，涵盖目标检测、布局/合成、Agent、动态场景、图像恢复、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🧠 NeurIPS2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (5)](../../ACL2026/object_detection/index.md) · [📷 CVPR2026 (39)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×6 · 布局/合成 ×2

**[Ascent Fails to Forget](ascent_fails_to_forget.md)**

:   本文从遗忘集与保留集之间的统计依赖出发，理论结合实验证明广泛使用的梯度上升/Descent-Ascent（DA）类机器遗忘方法在存在数据相关性时会系统性失败——在 logistic 回归中 DA 解甚至会比原始模型更远离 oracle，且在非凸设置下会将模型困在劣质局部最小值中。

**[Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)**

:   提出一个自反思 agent 框架，通过迭代的假设生成-测试-验证-反思循环来自动检测视觉模型中的属性依赖（如 CLIP 识别 teacher 依赖教室背景、YOLOv8 检测行人依赖人行横道），在 130 个注入已知属性依赖的模型 benchmark 上显示自反思显著提升检测准确性。

**[BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)**

:   提出首个面向多帧闪烁去除（MFFR）的大规模 benchmark 数据集 BurstDeflicker，包含基于 Retinex 的合成数据、真实静态数据和绿幕动态数据三个互补子集，系统解决了动态场景下闪烁-干净图像对难以获取的核心瓶颈。

**[CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)**

:   针对大规模类别（>10K）目标检测中分类头的正梯度稀释和难负样本梯度稀释问题，提出 CQ-DINO：用可学习类别查询替代分类头，通过图像引导的 Top-K 类别选择将负空间缩小 100 倍，在 V3Det（13204 类）上超越前 SOTA 2.1% AP，同时保持 COCO 竞争力。

**[DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)**

:   DetectiumFire 构建了最大的多模态火灾理解数据集——14.5K 真实图像 + 2.5K 视频 + 8K 合成图像 + 12K RLHF 偏好对，低重复率（0.03 PHash vs D-Fire 0.15），配合 4 级严重性分类标准和详细场景描述，微调 YOLOv11m 达 mAP 43.74，微调 LLaMA-3.2-11B 火灾严重性分类 83.84%。

**[DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)**

:   提出 DETree 框架，通过构建层次亲和树（HAT）建模不同人机协作文本生成过程之间的层次关系，并设计树结构对比损失（TSCL）对齐表示空间，在混合文本检测和 OOD 场景下取得了显著优势。

**[DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)**

:   DitHub 将开放词汇目标检测的增量适配问题重新构造为"版本控制"问题——为每个类别训练独立的 LoRA 专家模块，通过 branch（分支）、fetch（检索）、merge（合并）三个原语管理不断扩展的模块库，在 ODinW-13 全量数据上以 62.19 mAP 超越 ZiRa 4.21 个点，同时保持 47.01 的零样本 COCO 性能。

**[FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)**

:   提出 FlexEvent 框架，通过自适应事件-图像融合模块 FlexFuse 和频率自适应微调机制 FlexTune，实现事件相机在不同操作频率下的灵活目标检测，在 20Hz 到 180Hz 范围内保持鲁棒性能，显著超越现有方法。

**[Generalizable Insights for Graph Transformers in Theory and Practice](generalizable_insights_for_graph_transformers_in_theory_and_practice.md)**

:   提出 Generalized-Distance Transformer (GDT)，一种基于标准注意力（无需修改注意力机制）的图 Transformer 架构，理论证明其表达力等价于 GD-WL 算法，并通过覆盖 800 万图/2.7 亿 token 的大规模实验首次建立了 PE 表达力的细粒度经验层次，在 few-shot 迁移设置下无需微调即可超越 SOTA。

**[InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](instanceassemble_layoutaware_image_generation_via_instance_a.md)**

:   提出 InstanceAssemble，在 DiT-based T2I 模型（SD3 和 Flux）的 Transformer 块中注入"实例组装注意力"机制，通过将每个 bounding box 区域的 image token 独立与对应的 layout hidden state 做 cross-attention 来实现精确的实例级空间控制，同时以 LoRA 轻量适配方式保持与现有风格 LoRA 的兼容性，并提出包含 5K 图像/90K 实例的 DenseLayout 基准和多维度的 Layout Grounding Score（LGS）评估指标。

**[Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy](lr_yolo_lipschitz_continuity_image_restoration_object_detection.md)**

:   从 Lipschitz 连续性视角分析图像复原与目标检测级联框架的不稳定性根源，发现两个网络在平滑性上存在量级差异，提出 LR-YOLO 通过将复原任务集成到检测backbone的特征学习中来正则化检测器的Lipschitz常数，在去雾和低光增强基准上持续提升检测稳定性。

**[MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)**

:   提出 MSTAR，首个无需边界框标注的多查询场景文本检索方法，通过渐进式视觉嵌入（PVE）逐步将注意力从显著区域转移到不显著区域，结合风格感知指令和多实例匹配模块，实现了对单词、短语、组合和语义四种查询类型的统一检索，并构建了首个多查询文本检索基准 MQTR。

**[OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)**

:   OverLayBench 构建了首个聚焦密集重叠场景的 Layout-to-Image 基准（4052 样本 + OverLayScore 难度指标），揭示 SOTA 方法在复杂重叠下 mIoU 从 71%→54% 急剧退化，提出 Amodal Mask 监督在重叠 IoU 上提升 15.9%。

**[ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction](recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)**

:   提出 ReCon-GS，通过连续性保持的 Gaussian 流式处理实现增量式 3D 重建，在保持渲染质量的同时大幅减少存储需求和训练时间，支持大规模场景的实时重建。

**[ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)**

:   ReCon 提出无需额外训练的区域可控数据增强框架，通过区域引导校正（RGR）和区域对齐交叉注意力（RACA）增强现有结构可控生成模型的目标检测数据质量，在 COCO 上实现 35.5 mAP（超过需 fine-tune 的 GeoDiffusion）。

**[Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)**

:   提出无需源域数据的开放词汇测试时自适应目标检测框架（TTAOD），通过多模态 Prompt Tuning + Mean-Teacher + 实例动态记忆（IDM）+ 记忆增强/幻觉策略，在 Pascal-C 上 AP50 达 56.2%（+11.0 vs SOTA），在 13 个跨域数据集上一致有效。

**[Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)**

:   本文提出Video-RAG，一个免训练、即插即用的RAG管道，通过从视频中提取视觉对齐的辅助文本（OCR、ASR、目标检测）并经检索筛选后输入LVLM，在仅增加约2K token的条件下将7个开源LVLM的Video-MME平均性能提升2.8%，72B模型超越GPT-4o。
