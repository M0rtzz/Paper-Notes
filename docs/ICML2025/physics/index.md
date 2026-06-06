---
title: >-
  ICML2025 物理/科学计算方向14篇论文解读
description: >-
  14篇ICML2025的物理/科学计算方向论文解读，涵盖少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**🧪 ICML2025** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (5)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (12)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (41)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)**

:   提出 Causal-PIK，通过将物理因果相似性编码为贝叶斯优化的核函数（Physics-Informed Kernel），使智能体在物理推理任务中仅需极少次尝试即可找到最优动作，在 Virtual Tools 和 PHYRE 基准上超越 SOTA。

**[Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations](closed-form_solutions_a_new_perspective_on_solving_differential_equations.md)**

:   本文提出 SymPDE 框架，利用深度强化学习直接搜索 PDE 的闭式符号解，绕过了 PINNs 数值解精度不足和可解释性差的问题，在 Poisson 方程和热方程上达到 90% 的恢复率。

**[Compact Matrix Quantum Group Equivariant Neural Networks](compact_matrix_quantum_group_equivariant_neural_networks.md)**

:   本文将群等变神经网络扩展到**紧致矩阵量子群**的设定下，利用 Woronowicz 形式的 Tannaka-Krein 对偶理论刻画了该类网络的权重矩阵，为非交换几何上的数据学习提供了理论基础。

**[Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)**

:   提出 Kurucz-a1，一个物理约束神经网络（PINN），用于模拟一维恒星大气模型（LTE 假设），解决了可微恒星光谱学中大气结构求解器不可微的关键瓶颈，在流体静力平衡和太阳光谱一致性上甚至优于经典 ATLAS-12 代码。

**[Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)**

:   提出 Erwin，一种基于 ball tree 分层结构的 Transformer 架构，通过将注意力计算限制在固定大小的局部球区域内，实现线性时间复杂度，同时通过渐进式粗化/细化和跨球交互机制捕获多尺度特征，在宇宙学、分子动力学、PDE 求解和粒子流体动力学多个领域达到 SOTA。

**[Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)**

:   首次将 LoRA 应用于恒星光谱基础模型 SpecCLIP，实现以约 100-200 个标注样本将预训练在 LAMOST/Gaia XP 上的模型高效适配到 DESI 巡天数据，证明 LoRA 是跨光谱巡天迁移的轻量而有效策略。

**[Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents](gravity-bench-v1_a_benchmark_on_gravitational_physics_discovery_for_agents.md)**

:   提出 **Gravity-Bench-v1**，一个基于引力动力学模拟的**环境交互式**基准测试，评估 AI Agent 在受限观测预算下进行科学发现（包括 OOD 物理场景）的能力，发现当前模型在观测规划和预算利用方面存在显著不足。

**[Improving Memory Efficiency for Training KANs via Meta Learning](improving_memory_efficiency_for_training_kans_via_meta_learning.md)**

:   提出 MetaKANs，用一个小型元学习器（meta-learner）生成 KAN 中所有可学习激活函数的参数，将可训练参数量从 KAN 的 $(G+k+1)$ 倍压缩到接近 MLP 水平（约 1/3 到 1/9），同时保持甚至提升性能。

**[Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators](maximal_update_parametrization_and_zero-shot_hyperparameter_transfer_for_fourier.md)**

:   首次为 Fourier Neural Operator (FNO) 推导了 Maximal Update Parametrization (μP)，使得在小模型上调优的超参数可以零样本迁移到十亿参数级 FNO，将 Navier-Stokes 问题的调参计算量降至 0.30×。

**[Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)**

:   提出基于 Perceiver-IO 架构的多模态混合专家 VAE（MMVAE），对 Ia 型超新星的光变曲线和光谱进行联合嵌入，实现从光变曲线到光谱的跨模态概率生成，重建精度优于对比学习基线。

**[OmniArch: Building Foundation Model For Scientific Computing](omniarch_building_foundation_model_for_scientific_computing.md)**

:   OmniArch 是首个在 1D-2D-3D PDE 上进行统一预训练的科学计算基础模型，通过 Fourier 编解码器解决多尺度问题、Temporal Mask 机制处理多物理量耦合、PDE-Aligner 实现物理先验对齐，在 PDEBench 的 11 类 PDE 上达到了 SOTA 性能。

**[Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)**

:   在统一量子资源约束下系统性地对比 ML 与 DL 在量子系统学习 (QSL) 任务中的表现，发现传统 ML（Lasso/Ridge/核方法）往往匹配甚至超越 DL，挑战了"大规模量子系统必须用深度学习"的直觉。

**[Teaching LLMs to Speak Spectroscopy](teaching_llms_to_speak_spectroscopy.md)**

:   仅使用 16 GPU 小时和 0.04% 的参数适配，通过 LoRA 将 **LLaMA-3.1-8B** 改造为可从光谱数据预测星系红移的模型，同时保留 85%+ 的语言能力，证明通用 LLM 可高效适配非文本科学模态。

**[Universal Neural Optimal Transport](universal_neural_optimal_transport.md)**

:   提出 UNOT（Universal Neural Optimal Transport），利用 Fourier Neural Operator 学习跨数据集、跨分辨率的熵正则化最优传输对偶势函数，实现对 Sinkhorn 算法最高 7.4× 的加速初始化。
