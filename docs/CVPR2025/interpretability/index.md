---
title: >-
  CVPR2025 可解释性方向 11篇论文解读
description: >-
  11篇CVPR2025 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**📷 CVPR2025** · **11** 篇论文解读

**[Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)**

:   提出基于可解释基 BRDF 的可微逆渲染方法，将材质分解为有物理意义的基函数组合，实现可解释的材质估计

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)**

:   提出一个从基准构建、诊断到注入的完整框架，通过 3D 基础模型（VGGT）提取相机运动线索并以结构化提示注入 VideoLLM，实现无需训练的相机运动感知增强。

**[Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)**

:   本文提出一种基于非参数原型学习的可解释图像分类框架，通过对自监督ViT特征进行最优传输聚类来发现语义上不同的物体部件原型，解决了现有ProtoPNet方法中原型重复冗余的问题，同时引入了Distinctiveness和Comprehensiveness两个新指标来量化解释质量。

**[KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)**

:   KVQ 受人类视觉系统启发，将视频全局质量显式解耦为视觉显著性和局部纹理两个因素，通过 Fusion-Window Attention 提取跨区域显著性、Local Perception Constraint 增强独立区域的纹理感知，在五个 VQA benchmark 上显著超越 SOTA。

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。

**[Learning on Model Weights using Tree Experts](learning_on_model_weights_using_tree_experts.md)**

:   发现公开模型大多属于少数 Model Tree（从共同祖先微调而来），在同一 Tree 内学习权重远比跨 Tree 简单；提出 ProbeX——首个针对单隐藏层权重的轻量 probing 方法，通过 Tucker 张量分解实现参数量 30 倍压缩，并首次实现了将模型权重与文本表示对齐的零样本模型分类（89.8% 准确率）。

**[Learning Visual Composition through Improved Semantic Guidance](learning_visual_composition_through_improved_semantic_guidance.md)**

:   本文提出通过改善训练数据的语义监督信号（使用基础模型重新生成高质量描述+使用预训练文本编码器替代从头训练）来大幅提升标准 CLIP 模型的视觉组合理解能力，在 ARO 基准上从CLIP的59%/63%提升到92%/94%，在DOCCI图像检索上从58.4%提升到94.5% recall@1，且无需任何架构改动。

**[Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)**

:   提出 Prompt-CAM，通过为预训练 ViT 注入类别特定的可学习 prompt token，利用最后一层的多头注意力图来识别和定位区分细粒度类别的关键特征（traits），实现了近乎"免费"的可解释细粒度分析。

**[Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)**

:   本文提出PS3（Pre-training with Scale-Selective Scaling），通过局部区域与局部caption的对比学习代替全图对比，以近常数的计算开销将CLIP式视觉预训练扩展到4K分辨率，并结合top-down/bottom-up patch选择机制构建VILA-HD多模态大模型，在高分辨率感知任务上大幅超越GPT-4o和Qwen2.5-VL。

**[Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。

**[Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   本文提出 I2X 框架，通过在训练检查点上追踪从 GradCAM 显著性图中提取的抽象原型（prototype）的强度变化与模型置信度的对应关系，将非结构化的可解释性转化为结构化的可解释性，并利用识别出的"不确定原型"来指导微调、减少类间混淆、提升分类精度。
