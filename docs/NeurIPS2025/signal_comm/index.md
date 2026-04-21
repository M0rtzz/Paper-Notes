---
title: >-
  NeurIPS2025 信号/通信方向 12篇论文解读
description: >-
  12篇NeurIPS2025 信号/通信方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧠 NeurIPS2025** · **12** 篇论文解读

**[Angular Steering: Behavior Control via Rotation in Activation Space](angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

**[Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)**

:   构建了 Infinity-Chat 数据集（26K 开放式真实用户查询 + 31,250 条人类标注），揭示了 LM 在开放式生成中的"Artificial Hivemind"效应——模型内重复和模型间同质化严重，并发现 Reward Model 和 LM Judge 在个体偏好差异大的样本上校准失败。

**[Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[ConTextTab: 语义感知的表格上下文学习器](contexttab_a_semantics-aware_tabular_in-context_learner.md)**

:   ConTextTab 将语义嵌入（列名、分类值的文本编码）融入 table-native ICL 架构，并在大规模真实表格数据（T4, ~2.18M 表）上预训练，在语义丰富的 CARTE 基准上取得新 SOTA，同时在非语义基准上保持与现有方法竞争力。

**[Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)**

:   提出适用于随机OT映射的传输误差指标 $\mathcal{E}_p$（由优化间隙与可行性间隙组成），在无需Brenier映射存在或唯一性的最小假设下，构造了计算高效的rounding估计器达到近最优收敛率 $\tilde{O}(n^{-1/(d+2p)})$，并推广至Hölder连续核与对抗污染场景，建立了首个通用OT映射估计理论。

**[Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)**

:   论文认为时间表格学习真正难的不是“再加一个时间 embedding”这么简单，而是很多特征的语义会随时间漂移，因此提出 feature-aware modulation，通过时间上下文动态生成每个特征的偏移、缩放与非线性形状参数，把跨时间的语义重新对齐，最终在 TabReD 上让深度模型第一次在平均排名上稳定压过 GBDT。

**[Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)**

:   本文提出 Masked Symbol Modeling（MSM），将 BERT 的掩码预测范式应用于通信物理层——将脉冲成形产生的符号间贡献重新定义为"上下文信息"，训练 Transformer 在干净过采样基带信号上学习波形结构，推理时利用学到的上下文来恢复被冲激噪声破坏的符号。

**[Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。

**[Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   构建了包含 134,533 个星系的图像-光谱-红移多模态数据集（GalaxiesML-Spectra），适配多模态掩码自编码器（MMAE）同时进行图像和光谱的联合重建与红移回归，证明在测试时即使光谱完全缺失，仅用 25% 掩码图像即可实现优于 AstroCLIP 的红移预测散度 $\sigma_{NMAD} = 0.016$。

**[Perturbation Bounds for Low-Rank Inverse Approximations under Noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次推导了噪声条件下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，通过新颖的等高线自举 (contour bootstrapping) 技术处理非整函数 $f(z) = 1/z$，在有利条件下比经典界改进 $\sqrt{n}$ 倍。

**[The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)**

:   将可验证奖励的强化学习（RLVR）分解为正样本强化（PSR，增强正确回答概率）和负样本强化（NSR，惩罚错误回答），发现仅用 NSR 就能在整个 Pass@k 谱上持续提升推理性能且通常匹配或超越 PPO/GRPO，据此提出 Weighted-REINFORCE（降低 PSR 权重至 0.1）在 MATH/AIME 2025/AMC23 上取得全面最优。
