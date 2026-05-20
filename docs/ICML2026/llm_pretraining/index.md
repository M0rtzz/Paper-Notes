---
title: >-
  ICML2026 预训练方向17篇论文解读
description: >-
  17篇ICML2026的预训练方向论文解读，涵盖 LLM、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🧪 ICML2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (10)](../../ACL2026/llm_pretraining/index.md) · [📷 CVPR2026 (8)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/llm_pretraining/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

🔥 **高频主题：** LLM ×7 · 扩散模型 ×3

**[Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)**

:   作者发现 SFT 把模型对齐到一个低熵语义先验上、导致"指令模型越大越无聊"的反向 scaling，于是提出"标注锚定训练"——预训练阶段给文档配语义 tag、SFT 阶段对 tag 部分 mask loss，让推理时先采样语义再生成响应，从而在保持指令跟随能力的同时把语义多样性差距缩小 85%。

**[Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)**

:   本文从表达力与可训练性两个维度系统比较连续扩散、离散掩码扩散、looped transformer，证明"连续扩散"在表达力上严格强于离散扩散并能模拟 looped transformer，但实际性能受限于解码与表征空间；据此提出 **CCDD（Coevolutionary Continuous Discrete Diffusion）**——在离散 token 空间和预训练 LLM 的上下文嵌入空间上同时扩散，由单一模型联合去噪，在 LM1B/OWT 上比 MDLM 困惑度降 25-35%，并以仅 8 步采样超过 MDLM 256 步效果。

**[CoFrGeNet: Continued Fraction Architectures for Language Generation](cofrgenet_continued_fraction_architectures_for_language_generation.md)**

:   本文把"连续分数（continued fraction）"这种具备最优有理逼近性质的函数类引入到语言生成 Transformer 中，分别为多头注意力和 FFN 设计 CoFrNet 替代模块（CAttnU/CAttnM/Cffn），通过"continuants"封闭形式把 $d$ 次除法降为 1 次，在 GPT2-xl 和 Llama-3.2B 上用 $\frac{2}{3}\sim\frac{1}{2}$ 的参数实现持平甚至更优的下游性能。

**[Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)**

:   本文提出 Compute as Teacher（CaT）：把 GRPO 已经在采样的 G 条 rollouts 通过冻结锚模型"合成"出一个伪参考答案，再在非可验证领域用模型自己从该伪参考衍生的二元 rubric 给每条 rollout 打分作为 RL 奖励，从而在没有任何人工标注的情况下把推理算力直接变成监督信号，在 HealthBench 上相对基线最高提升 30%，并以 9× 更低的测试时算力匹配甚至超过 inference-time aggregation。

**[Consistent Diffusion Language Models](consistent_diffusion_language_models.md)**

:   本文指出离散扩散没有连续域 probability-flow ODE 的对应物，因此无法直接做 consistency model；作者提出用**精确闭式 posterior bridge** 作为离散域的"随机版 PF-ODE 替代品"，构造 Multi-Path Discrete Consistency (MPDC) 训练目标，要求 denoiser 在多条 stochastic bridge 路径上的预测在期望上一致，从而单阶段、teacher-free 地训出可在 2-3 步生成高质量文本的 Consistent Diffusion Language Model (CDLM)，在 unconditional / conditional 文本生成上达到 SOTA、对 AR 模型最高 $32\times$ 加速。

**[Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)**

:   本文系统研究 SFT 中数据难度的作用，发现并不存在"普适最优难度"，而是存在一个**随数据规模增大而向更难方向漂移**的最优难度，并用"in-distribution 泛化 gap"与"extrapolation gap"两个 gap 的 trade-off 给出 PAC-Bayes 解释。

**[Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)**

:   论文针对多任务指令微调中的跨任务梯度冲突问题，提出 Badit：先用 SVD 把预训练权重分解为一组天然正交的高奇异值 LoRA "基础能力"专家，再在训练过程中用球面 K-means 对 rank-1 分量做动态正交分组，从而把"按任务隔离参数"的传统思路改为"按基础能力解耦"，在 6 个 LLM 上平均比 GainLoRA 提升 2.68 Rouge。

**[Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)**

:   ME-DLM 给 masked diffusion 语言模型（如 LLaDA）加一个"解码完再编辑修补"的轻量阶段：第一阶段照常 unmask 出粗稿，第二阶段用替换/删除/插入三种 token 级编辑做并行修正，监督信号来自 edit distance 的最短编辑脚本，在只用 1/8 扩散步数的情况下 HumanEval +11.6 / GSM8K +33.6 点反超 LLaDA-Instruct。

**[Focus and Dilution: The Multi-stage Learning Process of Attention](focus_and_dilution_the_multi-stage_learning_process_of_attention.md)**

:   本文在单层 Transformer 学习马尔可夫数据的简化场景下，通过围绕一系列临界点做分阶段线性化的梯度流分析，揭示并严格刻画了注意力训练中反复出现的「聚焦—稀释」循环，并在 WikiText 与 TinyStories 上观察到一致的现象。

**[From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)**

:   本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target，无需额外算力就能在 MEMIT/RECT/PRUNE/AlphaEdit 之上一致涨点。

**[InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)**

:   作者提出 InfoLaw：把"预训练"重新定义为"按桶累积信息"的过程，每桶信息量等于"质量密度 $f_d$ × 唯一 token 数 $M_d$ × $\log K$"再乘上一个随重复次数 $R_d$ 指数衰减的因子，最终把验证损失写成 $L = \alpha\cdot\text{info}^{-\beta}$，能在 252M-1.2B 拟合后外推到 7B / 425B token，平均误差 0.15%、最大 0.96%，并直接用来搜索最优数据配方。

**[Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)**

:   作者用 10,866 个合并模型实测出一条形如 $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ 的双轴幂律：基座规模 $N$ 决定 floor，专家数 $k$ 决定 tail，且四种主流合并方法（Average、TA、TIES、DARE）都共用同一条曲线，从而把"合多少个专家、合到哪一步停"变成一个可预测、可预算的工程问题。

**[On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)**

:   本文用一套精心控制"推理难度恒定、只变 horizon 长度"的 Sudoku/Rush Hour 任务，系统证明**任务 horizon 本身就是 LLM agent RL 训练崩溃的独立根因**，并提出 macro action 与 subgoal decomposition 两种 horizon-reduction 机制——它们不仅稳住训练，还让模型在更长 horizon 上实现强 zero-shot 泛化（horizon generalization）。

**[Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)**

:   本文提出 Noisy Quadratic System (NQS)——一个把 LLM 测试损失建模为 $L(N, B, K)$（模型大小 / 批大小 / 更新步数）的 mechanistic 损失模型，首次在 scaling law 中显式建模 batch size，并在 Pythia + OWT2 上把外推预测能力从 Chinchilla 的 ~20× 算力提升到 ~4000× 算力。

**[Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)**

:   作者把传统 Softmax attention 解构为"非负化 + L1 归一化"两个独立部件，证明真正关键的是 L1 归一化而非指数，于是用 Softplus + 动态长度尺度因子换掉指数得到 LSSA，再用一次幂函数式"重权"对注意力锐化，得到的 LSSAR 在 16× 训练长度上几乎保持 validation loss 不变，并能让 GPT-109M 从轨迹数据中"重新发现"牛顿万有引力定律。

**[Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)**

:   作者在简化单层线性注意力 Transformer 上推出闭式训练动力学，证明正则化方法只能改变收敛速度而无法挪动收敛点（因此在 cFKA 场景几乎注定失效），数据回放则能直接改变收敛点并加大震荡幅度从而稳住旧知识，进而提出按 token 注意力贡献裁切片段、引导预训练模型生成回放语料的 STOC，在合成 + KnowEdit + IndustryCorpus 法律语料上一致比 LAMOL 更能压制遗忘。

**[Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)**

:   作者把 Transformer 自注意力写成 token 间相互作用的平均场粒子系统，把 LoRA 视作低秩扰动，证明遗忘与"扰动模长"和"网络深度"两条相变曲线相关，并给出由 $V$ 的特征值 gap 控制的长时稳定条件。
