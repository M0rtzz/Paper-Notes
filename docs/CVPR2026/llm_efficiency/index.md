---
title: >-
  CVPR2026 LLM效率方向 7篇论文解读
description: >-
  7篇CVPR2026 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**📷 CVPR2026** · **7** 篇论文解读

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)**

:   首个面向3D几何计算机视觉的PhD级代码生成基准GeoCodeBench，包含100个从2025年顶会论文+代码库中精选的函数补全任务，配套自动化多样化单元测试，最强模型GPT-5仅36.6%通过率，揭示LLM在科学级3D代码实现上的巨大差距。

**[Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)**

:   提出QICA框架解决零样本目标计数中的数量感知缺失和空间不敏感问题，通过数量条件化的协同提示策略（SPS）联合适配视觉-语言编码器，结合在相似度图上直接操作的代价聚合解码器（CAD）保持零样本迁移能力，在FSC-147上达到零样本SOTA（MAE 12.41）并展现强跨域泛化。

**[Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)**

:   提出 AL-GCD 框架，通过模拟人类类比推理机制设计"类比文本概念生成器"（ATCG）——从已知类别的视觉-文本知识库中类比生成未知样本的文本概念，将类别发现转化为视觉-文本联合推理任务，在六个基准上平均提升 5.0%，细粒度数据集提升 7.1%。

**[Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)**

:   提出 ESM 框架，通过对参数更新引起的激活偏移做 PCA 构建"本质子空间"（而非直接对参数做 SVD），并用三级极化缩放增强关键参数、抑制噪声，在 ViT-B/32 的 20 任务合并中比 Iso-CTS 提升 3.2%（绝对准确率）。

**[SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)**

:   对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块，实现8B模型1024×1024生成降至1秒级（1.57×加速），且几乎不损失高频细节。

**[StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)**

:   提出StoryTailor零样本视觉叙事生成管线，通过高斯中心注意力（GCA）缓解主体重叠和背景泄漏、动作增强奇异值重加权（AB-SVR）放大动作语义、选择性遗忘缓存（SFC）维护跨帧背景连续性，在单张RTX 4090上实现多主体、动作丰富的图像叙事生成，CLIP-T较基线提升10-15%。
