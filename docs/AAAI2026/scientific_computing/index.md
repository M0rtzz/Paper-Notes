---
title: >-
  AAAI2026 科学计算方向 8篇论文解读
description: >-
  8篇AAAI2026 科学计算方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🤖 AAAI2026** · **8** 篇论文解读

**[Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems](just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)**

:   提出随机稀疏反馈控制框架：控制器在每个时间步仅访问状态向量的随机子集，通过 LMI 联合设计反馈增益矩阵和 Bernoulli 稀疏化参数，在保证渐近均方稳定性（AMSS）的同时最小化所需传感器数量，实验中仅用 0.3% 的状态分量即可达到与全状态反馈可比的性能。

**[Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)**

:   提出 KARMA 框架，在 ViT-MAE 解码器中嵌入线性光谱混合模型 (LSMM) 作为物理约束，结合 Spectral Angle Mapper (SAM) 损失，提升高光谱遥感图像的重建保真度和下游任务迁移性能。

**[Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids](phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)**

:   构建了 Phys-Liquid 数据集（97,200 张物理仿真图像 + 3D mesh），基于 Navier-Stokes 方程模拟透明容器内液体的动态形变，并提出四阶段重建管线（分割→多视角 mask 生成→3D 重建→缩放），在仿真和真实场景中实现高精度液体几何与体积估计。

**[PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)**

:   提出 PhysicsCorrect，一种无需训练的校正框架，通过将 PDE 残差校正建模为线性化逆问题并预计算伪逆缓存，在推理时以 <5% 计算开销实现最高 100× 误差降低，适用于 FNO/UNet/ViT 等任意预训练神经算子。

**[PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)**

:   提出 PIMRL 框架，针对 burst 采样（短段高频+长间隔）的稀疏时空数据，结合宏观尺度潜空间推理和微观尺度物理校正的双模块架构，通过跨尺度消息传递融合信息，在 5 个 PDE 基准上将误差最多降低 80%。

**[SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)**

:   提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力（WA）捕获高频局部细节，与傅里叶注意力（FA）的全局感受野经门控融合互补，在 6 个算子学习基准上取得 SOTA，Navier-Stokes 相对误差比 Transolver 下降 22.3%。

**[Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)**

:   提出物理基线+数据驱动残差的混合建模框架，将海试功率曲线（螺旋桨定律 $P=cV^n$）作为基线，用 XGBoost/NN/PINN 学习残差修正，在稀疏数据区域显著提升外推稳定性和物理一致性。

**[Towards a Foundation Model for Partial Differential Equations Across Physics Domains](towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)**

:   提出 PDE-FM，一个结合空间-频谱 tokenization、物理感知 FiLM 调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准的 12 个跨物理域数据集上平均降低 VRMSE 46%。
