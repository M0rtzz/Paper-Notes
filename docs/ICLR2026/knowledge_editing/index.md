---
title: >-
  ICLR2026 知识编辑方向 9篇论文解读
description: >-
  9篇ICLR2026 知识编辑方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🔬 ICLR2026** · 共 **9** 篇

**[Bilinear Representation Mitigates Reversal Curse And Enables Consistent Model Ed](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[Eamet Robust Massive Model Editing Via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)**

:   发现大规模模型编辑失败的根本原因是 key embedding 和 residual embedding 之间的结构不一致（embedding misalignment），提出 EAMET 通过 KL+MSE 双损失渐进式对齐优化，在 6 个 LLM 上平均提升编辑成功率 14%（CounterFact）。

**[Energy-Regularized Sequential Model Editing On Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[Fine-Tuning Done Right In Model Editing](fine-tuning_done_right_in_model_editing.md)**

:   揭示模型编辑中 fine-tuning 被低估的根因是错误的训练 pipeline（深度优先逐样本优化），修正为标准的广度优先 mini-batch 训练后，配合局部化参数调优形成 LocFT-BF，首次支持 10 万次连续编辑和 72B 模型规模。

**[Got-Edit Geometry-Aware Generic Object Tracking Via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[Pics Pairwise Image Compositing With Spatial Interactions](pics_pairwise_image_compositing_with_spatial_interactions.md)**

:   提出 PICS——一种并行成对图像合成方法，通过 Interaction Transformer 中的掩码引导 MoE 和自适应 α-blending 策略，在单次推理中同时合成两个对象并显式建模遮挡、接触等空间交互关系，全面超越现有序列合成方法。

**[Rote Learning Considered Useful Generalizing Over Memorized Data In Llms](rote_learning_considered_useful_generalizing_over_memorized_data_in_llms.md)**

:   提出"记忆-再泛化"（memorize-then-generalize）框架，通过先用无语义合成 token 死记硬背事实关联、再用少量语义提示微调的两阶段策略，揭示 LLM 能从死记硬背数据中泛化，且记忆越深泛化越好，同时指出该机制可被恶意利用的安全隐患。

**[Rote Learning Considered Useful Generalizing Over Memorized Training Examples](rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)**

:   本文提出"先记忆再泛化"两阶段框架，证明 LLM 可以在死记硬背合成关键 token 后，通过极少量语义微调实现泛化，挑战了"记忆阻碍泛化"的传统观点。

**[When Large Multimodal Models Confront Evolving Knowledge Challenges And Explorat](when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)**

:   提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。
