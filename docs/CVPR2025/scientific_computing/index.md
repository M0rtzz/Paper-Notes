---
title: >-
  CVPR2025 科学计算方向3篇论文解读
description: >-
  3篇CVPR2025的科学计算方向论文解读，收录 Accurate Differential Operator、Improve Representation for Imb、Learning Phase Distortion with等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "科学计算"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**📷 CVPR2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (19)](../../ICML2026/scientific_computing/index.md) · [📷 CVPR2026 (4)](../../CVPR2026/scientific_computing/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/scientific_computing/index.md) · [🤖 AAAI2026 (8)](../../AAAI2026/scientific_computing/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/scientific_computing/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/scientific_computing/index.md)

**[Accurate Differential Operators for Hybrid Neural Fields](accurate_differential_operators_for_hybrid_neural_fields.md)**

:   揭示混合神经场（如 Instant NGP）中自动微分产生的梯度和曲率存在严重高频噪声问题，提出基于局部多项式拟合的后处理微分算子和自监督微调方法，将梯度误差降低 4 倍、曲率误差降低 4 倍，在渲染和物理模拟中显著消除伪影。

**[Improve Representation for Imbalanced Regression through Geometric Constraints](improve_representation_for_imbalanced_regression_through_geometric_constraints.md)**

:   本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。

**[Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)**

:   提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。
