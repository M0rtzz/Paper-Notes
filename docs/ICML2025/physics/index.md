---
title: >-
  ICML2025 物理学方向 6篇论文解读
description: >-
  6篇ICML2025 物理学方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理学

**🧪 ICML2025** · **6** 篇论文解读

**[Compact Matrix Quantum Group Equivariant Neural Networks](compact_matrix_quantum_group_equivariant_neural_networks.md)**

:   本文将群等变神经网络扩展到**紧致矩阵量子群**的设定下，利用 Woronowicz 形式的 Tannaka-Krein 对偶理论刻画了该类网络的权重矩阵，为非交换几何上的数据学习提供了理论基础。

**[Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)**

:   首次将 LoRA 应用于恒星光谱基础模型 SpecCLIP，实现以约 100-200 个标注样本将预训练在 LAMOST/Gaia XP 上的模型高效适配到 DESI 巡天数据，证明 LoRA 是跨光谱巡天迁移的轻量而有效策略。

**[Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents](gravity-bench-v1_a_benchmark_on_gravitational_physics_discovery_for_agents.md)**

:   提出 **Gravity-Bench-v1**，一个基于引力动力学模拟的**环境交互式**基准测试，评估 AI Agent 在受限观测预算下进行科学发现（包括 OOD 物理场景）的能力，发现当前模型在观测规划和预算利用方面存在显著不足。

**[Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)**

:   提出基于 Perceiver-IO 架构的多模态混合专家 VAE（MMVAE），对 Ia 型超新星的光变曲线和光谱进行联合嵌入，实现从光变曲线到光谱的跨模态概率生成，重建精度优于对比学习基线。

**[Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)**

:   在统一量子资源约束下系统性地对比 ML 与 DL 在量子系统学习 (QSL) 任务中的表现，发现传统 ML（Lasso/Ridge/核方法）往往匹配甚至超越 DL，挑战了"大规模量子系统必须用深度学习"的直觉。

**[Teaching LLMs to Speak Spectroscopy](teaching_llms_to_speak_spectroscopy.md)**

:   仅使用 16 GPU 小时和 0.04% 的参数适配，通过 LoRA 将 **LLaMA-3.1-8B** 改造为可从光谱数据预测星系红移的模型，同时保留 85%+ 的语言能力，证明通用 LLM 可高效适配非文本科学模态。
