---
title: >-
  CVPR2025 物理/科学计算方向4篇论文解读
description: >-
  4篇CVPR2025的物理/科学计算方向论文解读，涵盖模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**📷 CVPR2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (5)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (12)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (41)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md)

**[Accurate Differential Operators for Hybrid Neural Fields](accurate_differential_operators_for_hybrid_neural_fields.md)**

:   揭示混合神经场（如 Instant NGP）中自动微分产生的梯度和曲率存在严重高频噪声问题，提出基于局部多项式拟合的后处理微分算子和自监督微调方法，将梯度误差降低 4 倍、曲率误差降低 4 倍，在渲染和物理模拟中显著消除伪影。

**[ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)**

:   提出 ATP（Adaptive Threshold Pruning），在量子数据编码前自适应地剪除低信息量的数据特征，通过 L-BFGS-B 优化阈值，在 MNIST/FashionMNIST/CIFAR/PneumoniaMNIST 四个数据集的二分类任务上取得最高准确率的同时显著降低纠缠熵。

**[Improve Representation for Imbalanced Regression through Geometric Constraints](improve_representation_for_imbalanced_regression_through_geometric_constraints.md)**

:   本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。

**[Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)**

:   提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。
