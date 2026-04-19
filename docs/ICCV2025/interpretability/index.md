---
title: >-
  ICCV2025 可解释性方向 8篇论文解读
description: >-
  8篇ICCV2025 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**📹 ICCV2025** · **8** 篇论文解读

**[AIM: Amending Inherent Interpretability via Self-Supervised Masking](aim_amending_inherent_interpretability_via_self-supervised_masking.md)**

:   本文提出 AIM，一种基于自监督二值掩码的 top-down 特征选择机制，无需额外标注即可引导 CNN 聚焦真实判别特征、抑制虚假相关，同时获得内在可解释性和更强的 OOD 泛化能力。

**[ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)**

:   提出 ArgoTweak，首个提供"旧地图先验 + 当前传感器数据 + 最新真值地图"完整三元组的 HD 地图数据集，通过双射映射框架将大规模地图修改分解为元素级原子变化，并引入可解释的评测指标（mAPC/mACC），将模型在 ArgoTweak 上训练后的 sim2real 差距降低 10 倍以上。

**[CE-FAM: Concept-Based Explanation via Fusion of Activation Maps](ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)**

:   提出CE-FAM概念解释方法，通过训练与图像分类器共享激活图的分支网络来模拟VLM嵌入，实现概念预测→概念区域（激活图加权和）→概念贡献（对分类分数影响）的一一对应，并提出新的NRA评估指标，在零样本概念推理上超越现有方法。

**[Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)**

:   提出 Granular Concept Circuit (GCC) 方法，通过迭代评估神经元间的功能依赖性（Neuron Sensitivity Score）和语义一致性（Semantic Flow Score），自动发现深度视觉模型中编码特定概念的细粒度视觉电路——这是首个能在单个query中发现多个概念级电路的方法。

**[Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond](learnable_fractional_reaction-diffusion_dynamics_for_under-display_tof_imaging_a.md)**

:   LFRD² 提出一种混合框架，将可学习的时间分数阶反应-扩散方程与神经网络结合，用于屏下 ToF（UD-ToF）深度图恢复。通过分数阶微积分捕获迭代过程中的长期记忆依赖，并引入高效的连续卷积算子替代离散卷积，在 UD-ToF 深度恢复、ToF 去噪和深度超分辨率任务上均取得最优性能。

**[Minerva: Evaluating Complex Video Reasoning](minerva_evaluating_complex_video_reasoning.md)**

:   提出 Minerva——一个包含 1515 个手工标注的复杂视频推理问答数据集，每题配有 5 个选项和详细推理链（reasoning trace），用于评估多模态大模型的视频推理能力，并建立了视频推理错误分类体系（Temporal/Perceptual/Logical/Completeness）和 MiRA 自动评估框架。

**[SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)**

:   提出SVIP框架，通过在**输入阶段**识别并替换语义无关的图像patch（用属性级word embedding初始化的可学习嵌入替代），从根源上解决零样本学习中的语义错位问题。

**[VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)**

:   提出VITAL方法，通过将特征可视化重新定义为真实图像特征分布对齐问题（而非传统的激活最大化），并结合相关性评分过滤无关特征，生成对人类更易理解的神经元可视化结果。
