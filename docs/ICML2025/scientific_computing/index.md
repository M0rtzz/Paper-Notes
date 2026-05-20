---
title: >-
  ICML2025 科学计算方向8篇论文解读
description: >-
  8篇ICML2025的科学计算方向论文解读，涵盖少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "科学计算"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🧪 ICML2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (19)](../../ICML2026/scientific_computing/index.md) · [📷 CVPR2026 (4)](../../CVPR2026/scientific_computing/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/scientific_computing/index.md) · [🤖 AAAI2026 (8)](../../AAAI2026/scientific_computing/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/scientific_computing/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/scientific_computing/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)**

:   提出 Causal-PIK，通过将物理因果相似性编码为贝叶斯优化的核函数（Physics-Informed Kernel），使智能体在物理推理任务中仅需极少次尝试即可找到最优动作，在 Virtual Tools 和 PHYRE 基准上超越 SOTA。

**[Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations](closed-form_solutions_a_new_perspective_on_solving_differential_equations.md)**

:   本文提出 SymPDE 框架，利用深度强化学习直接搜索 PDE 的闭式符号解，绕过了 PINNs 数值解精度不足和可解释性差的问题，在 Poisson 方程和热方程上达到 90% 的恢复率。

**[Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)**

:   提出 Kurucz-a1，一个物理约束神经网络（PINN），用于模拟一维恒星大气模型（LTE 假设），解决了可微恒星光谱学中大气结构求解器不可微的关键瓶颈，在流体静力平衡和太阳光谱一致性上甚至优于经典 ATLAS-12 代码。

**[Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)**

:   提出 Erwin，一种基于 ball tree 分层结构的 Transformer 架构，通过将注意力计算限制在固定大小的局部球区域内，实现线性时间复杂度，同时通过渐进式粗化/细化和跨球交互机制捕获多尺度特征，在宇宙学、分子动力学、PDE 求解和粒子流体动力学多个领域达到 SOTA。

**[Improving Memory Efficiency for Training KANs via Meta Learning](improving_memory_efficiency_for_training_kans_via_meta_learning.md)**

:   提出 MetaKANs，用一个小型元学习器（meta-learner）生成 KAN 中所有可学习激活函数的参数，将可训练参数量从 KAN 的 $(G+k+1)$ 倍压缩到接近 MLP 水平（约 1/3 到 1/9），同时保持甚至提升性能。

**[Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators](maximal_update_parametrization_and_zero-shot_hyperparameter_transfer_for_fourier.md)**

:   首次为 Fourier Neural Operator (FNO) 推导了 Maximal Update Parametrization (μP)，使得在小模型上调优的超参数可以零样本迁移到十亿参数级 FNO，将 Navier-Stokes 问题的调参计算量降至 0.30×。

**[OmniArch: Building Foundation Model For Scientific Computing](omniarch_building_foundation_model_for_scientific_computing.md)**

:   OmniArch 是首个在 1D-2D-3D PDE 上进行统一预训练的科学计算基础模型，通过 Fourier 编解码器解决多尺度问题、Temporal Mask 机制处理多物理量耦合、PDE-Aligner 实现物理先验对齐，在 PDEBench 的 11 类 PDE 上达到了 SOTA 性能。

**[Universal Neural Optimal Transport](universal_neural_optimal_transport.md)**

:   提出 UNOT（Universal Neural Optimal Transport），利用 Fourier Neural Operator 学习跨数据集、跨分辨率的熵正则化最优传输对偶势函数，实现对 Sinkhorn 算法最高 7.4× 的加速初始化。
