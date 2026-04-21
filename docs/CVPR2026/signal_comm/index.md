---
title: >-
  CVPR2026 信号/通信方向 4篇论文解读
description: >-
  4篇CVPR2026 信号/通信方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**📷 CVPR2026** · **4** 篇论文解读

**[AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)**

:   本文提出 AcTTA，一种基于动态激活函数调制的测试时自适应框架，通过将传统固定激活函数重参数化为可学习形式（包含激活中心偏移和非对称梯度斜率），在推理时自适应调整激活行为以应对分布偏移，在 CIFAR10-C/CIFAR100-C/ImageNet-C 上一致超越基于归一化层的 TTA 方法。

**[ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)**

:   提出 ChartNet，一个包含 150 万条高质量多模态对齐样本的百万级图表理解数据集，通过代码引导的合成管线生成涵盖 24 种图表类型、6 种绘图库的五元组数据（代码、图像、数据表、文本描述、带推理的 QA），在 ChartNet 上微调的 2B 模型可超越 GPT-4o 和 72B 开源模型。

**[Dual-Imbalance Continual Learning for Real-World Food Recognition](dual-imbalance_continual_learning_for_real-world_food_recognition.md)**

:   提出 DIME 框架，通过类别计数引导的光谱适配器合并和秩自适应阈值调制机制，在双重不平衡（类内长尾分布 + 步间类别数不均匀）的持续学习场景下，在四个长尾食物数据集上持续超越 baseline 3% 以上。

**[FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)**

:   提出 FAAR，一种频率感知的多任务参数高效微调方法，通过 Performance-Driven Rank Shrinking (PDRS) 为每个任务和层动态选择最优秩，并设计 Task-Spectral Pyramidal Decoder (TS-PD) 利用 FFT 频率信息增强空间感知和跨任务一致性，以传统微调 1/9 的参数量实现更优性能。
