---
title: >-
  ICLR2026 目标检测方向9篇论文解读
description: >-
  9篇ICLR2026的目标检测方向论文解读，涵盖目标检测、模型压缩、扩散模型、少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
  - "扩散模型"
  - "少样本学习"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🔬 ICLR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/object_detection/index.md) · [📷 CVPR2026 (36)](../../CVPR2026/object_detection/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×3

**[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)**

:   提出 AdaRank，用可学习二值掩码自适应选择 task vector 的奇异分量（取代启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰，在 ViT-B/32 上达到 89.4% 准确率。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。

**[CORDS: Continuous Representations of Discrete Structures](cords_continuous_representations_of_discrete_structures.md)**

:   提出 CORDS 框架，通过将变大小离散集合（检测框、分子原子）双射映射为连续的密度场和特征场，使模型可在场空间中学习并精确解码回离散集合，避免了固定 slot 或 padding 的限制。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件，为下冠层人员检测模型的训练和评估提供了坚实基础。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)**

:   提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

**[SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)**

:   提出 SPWOOD 框架统一处理稀疏标注和弱标注（HBox/Point）的旋转目标检测问题，通过自适应旋转目标检测器(SAOD)和空间布局学习策略，在 DOTA 基准上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。
