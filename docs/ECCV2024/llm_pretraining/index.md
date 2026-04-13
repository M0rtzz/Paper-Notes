---
title: >-
  ECCV2024 预训练/数据方向 6篇论文解读
description: >-
  6篇ECCV2024 预训练/数据方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练/数据

**🎞️ ECCV2024** · 共 **6** 篇

**[Cross-Domain Learning For Video Anomaly Detection With Limited Supervision](cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)**

:   提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。

**[Dragapart Learning A Part-Level Motion Prior For Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)**

:   DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。

**[Learning To Obstruct Few-Shot Image Classification Over Restricted Classes](learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)**

:   提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。

**[Prompting Language-Informed Distribution For Compositional Zero-Shot Learning](prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)**

:   本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。

**[Scaling Backwards Minimal Synthetic Pre-Training](scaling_backwards_minimal_synthetic_pre-training.md)**

:   提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。

**[Scantalk 3D Talking Heads From Unregistered Scans](scantalk_3d_talking_heads_from_unregistered_scans.md)**

:   提出 ScanTalk，首个能够对**任意拓扑**（包括未配准的3D扫描数据）的3D人脸进行语音驱动动画生成的深度学习框架，核心依赖于 DiffusionNet 的离散化无关特性来突破固定拓扑约束。
