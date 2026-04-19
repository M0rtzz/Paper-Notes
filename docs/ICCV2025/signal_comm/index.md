---
title: >-
  ICCV2025 信号/通信方向 2篇论文解读
description: >-
  2篇ICCV2025 信号/通信方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**📹 ICCV2025** · **2** 篇论文解读

**[Boosting Multimodal Learning via Disentangled Gradient Learning](boosting_multimodal_learning_via_disentangled_gradient_learning.md)**

:   本文揭示了多模态学习中模态编码器和融合模块之间的优化冲突——融合模块会抑制回传到各模态编码器的梯度，导致即使是优势模态也比单模态模型表现差，并提出解耦梯度学习（DGL）框架通过截断融合模块到编码器的梯度并用独立的单模态损失替代来解决此问题。

**[Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors](generalizable_non-line-of-sight_imaging_with_learnable_physical_priors.md)**

:   提出Learnable Path Compensation (LPC)和Adaptive Phasor Field (APF)两个模块，分别解决NLOS成像中辐射强度衰减的材质依赖性问题和不同信噪比条件下的频域去噪问题，仅在合成数据上训练即可在多种真实数据集上实现SOTA泛化性能。
