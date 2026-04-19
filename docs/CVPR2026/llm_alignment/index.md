---
title: >-
  CVPR2026 对齐/RLHF方向 11篇论文解读
description: >-
  11篇CVPR2026 对齐/RLHF方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐/RLHF

**📷 CVPR2026** · **11** 篇论文解读

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](bases_of_steerable_kernels_for_equivariant_cnns_fr.md)**

:   提出一种绕过 Clebsch-Gordan 系数计算、直接从群表示矩阵元素构造可操纵核显式基的方法，通过"稳定子约束 + Schur 引理 + Steering"三步策略统一覆盖 SO(2)、O(2)、SO(3)、O(3) 和非紧致 Lorentz 群，大幅简化等变 CNN 的核设计流程。

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种绕过 Clebsch-Gordan 系数的方法来求解等变CNN中的可转向核（steerable kernel）约束，通过在稳定子群上求解简单的不变性条件再"转向（steer）"到任意点，为 SO(2) 到 Lorentz 群等不同对称群给出了显式的核基底。

**[GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)**

:   提出 GlyphPrinter，通过构建区域级字形偏好数据集 GlyphCorrector 和区域分组 DPO（R-GDPO）目标函数，在不依赖显式奖励模型的情况下显著提升视觉文本渲染的字形准确度，并引入推理时 Regional Reward Guidance 实现可控生成。

**[MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)**

:   提出 MapReduce LoRA 和 RaTE 两种互补方法来推进多偏好优化的 Pareto 前沿：前者通过"Map（并行训偏好专家）+ Reduce（迭代合并）"的策略渐进推进 Pareto 前沿；后者通过学习奖励感知的 token embedding 实现推理时可组合的偏好控制。

**[Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](mesh-pro_asynchronous_advantage-guided_ranking_preference_optimization_for_artis.md)**

:   提出 Mesh-Pro，首个面向3D四边形网格生成的异步在线强化学习框架，核心算法 ARPO（Advantage-guided Ranking Preference Optimization）通过 Plackett-Luce 排名模型与优势函数加权相结合，在效率（较离线 DPO 快 3.75x）和泛化性上同时取得提升，实现 artist-style 和 dense mesh 的 SOTA 生成质量。

**[LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)**

:   提出LocalDPO，通过对真实高质量视频进行随机时空Bézier掩码的局部腐蚀生成负样本（单次推理、无需外部排序），配合区域感知DPO损失在局部细节级别进行偏好对齐，在Wan2.1和CogVideoX上一致超越传统DPO和SFT的视频质量。

**[MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)**

:   提出 MoD-DPO（Modality-Decoupled DPO），通过不变性正则化、敏感性正则化和语言先验去偏三个机制解耦多模态 LLM 中各模态的贡献，有效缓解跨模态幻觉（如用听觉信息回答视觉问题），并推导出闭式最优策略。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)**

:   将 DPO 偏好优化引入扩散运动生成模型的后训练阶段，通过物理仿真控制器自动构造偏好数据对，使生成的人体运动既符合文本/空间控制指令又满足物理约束，并成功零样本迁移到 Unitree G1 真实机器人。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](physmodpo_physicallyplausible_humanoid_motion_with.md)**

:   提出PhysMoDPO，将预训练的全身控制器（WBC/DeepMimic）集成到扩散运动生成器的后训练流程中，通过物理仿真自动构造偏好对并用DPO微调，使生成运动在WBC执行后同时满足物理可行性和文本/空间条件忠实度，实现零样本迁移到Unitree G1真实机器人。

**[Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)**

:   提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。

**[$\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)**

:   提出 $\varphi$-DPO，将 DPO 作为持续学习范式（以前一步模型为参考策略），并引入受 focal loss 启发的公平性调制因子 $(1-p)^\gamma$ 来平衡不同数据组间的梯度贡献，在理论上证明 $\gamma \to \infty$ 时梯度偏差趋于零，在 CoIN 和 MLLM-CL 基准上达到 SOTA。
