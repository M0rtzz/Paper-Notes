---
title: >-
  ICML2025 对齐 / RLHF方向24篇论文解读
description: >-
  24篇ICML2025的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、扩散模型、LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "扩散模型"
  - "LLM"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🧪 ICML2025** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (7)](../../ICML2026/llm_alignment/index.md) · [💬 ACL2026 (26)](../../ACL2026/llm_alignment/index.md) · [📷 CVPR2026 (10)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (41)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (19)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×19 · 扩散模型 ×3 · LLM ×3 · 多模态 ×2

**[ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)**

:   将DPO思想引入扩散式人体网格恢复(HMR)：训练HMR-Scorer评估预测质量，构建偏好数据集(winner/loser对)，用DPO微调基座扩散模型，无需3D标注即可提升in-the-wild图像上的HMR性能。

**[AlphaPO: Reward Shape Matters for LLM Alignment](alphapo_reward_shape_matters_for_llm_alignment.md)**

:   AlphaPO 在 Direct Alignment Algorithms（DAA）框架中引入 $\alpha$ 参数来改变奖励函数的"形状"，从标准的 log 奖励推广到更一般的幂次变换形式，从而细粒度控制 likelihood displacement 和 over-optimization，在 Mistral-7B 和 Llama3-8B 上相对 SimPO 提升 7%-10%，相对 DPO 提升 15%-50%。

**[AMPO: Active Multi-Preference Optimization for Self-play Preference Selection](ampo_active_multi-preference_optimization_for_self-play_preference_selection.md)**

:   提出 AMPO 框架，将在线策略生成、多偏好组对比损失和主动子集选择相结合，通过从大规模候选响应池中智能挑选少量但信息丰富的子集进行偏好优化，在 AlpacaEval 上达到 SOTA。

**[AssistanceZero: Scalably Solving Assistance Games](assistancezero_scalably_solving_assistance_games.md)**

:   提出 AssistanceZero，首次将 assistance game 扩展到复杂环境（Minecraft 建筑辅助，$10^{400}$ 种可能目标），通过扩展 AlphaZero 增加 reward 预测头和人类行为预测头，在 MCTS 下进行不确定性规划，显著优于 PPO 和模仿学习基线，人类实验证明能有效减少用户操作并展现挖地基、推断屋顶、从纠正中学习等涌现行为。

**[BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization](bopo_neural_combinatorial_optimization_via_best-anchored_and_objective-guided_pr.md)**

:   将 preference optimization（偏好优化）引入神经组合优化（NCO），提出 BOPO：通过 (1) best-anchored 偏好对构建（hybrid rollout + uniform filtering + best-anchored pairing）和 (2) objective-guided 自适应缩放损失函数（$\beta = g(y_l)/g(y_w)$），在 JSP/TSP/FJSP 三类经典组合优化问题上全面超越 SOTA，无需 reward model 或参考策略。

**[Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)**

:   发现 RLHF 中 KL 正则化带来的结构性质——策略对最优策略的 coverage 被其次优性控制（$\text{Cov}^{\pi^*|\pi} \leq 1 + \kappa \cdot (J(\pi^*) - J(\pi))/\beta$），据此提出两条迁移学习原则：(1) 选高 policy value 的 transfer policy，(2) self-transfer 从在线数据蒸馏策略。设计 TPO 算法实现早期 $O(W\sqrt{T})$、后期 $O(\sqrt{T})$ 的 regret，可模块化集成 DPO/IPO/XPO，在 T5 summarization 实验上验证有效。

**[Challenges and Future Directions of Data-Centric AI Alignment](challenges_and_future_directions_of_data-centric_ai_alignment.md)**

:   本文是一篇 position paper，倡导将 AI 对齐的研究重心从算法设计转向数据质量，通过对 Anthropic-HH 数据集的定性分析揭示了人类反馈中的六大不可靠来源，并提出了改进数据收集、清洗和验证的未来方向。

**[ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)**

:   提出 ConfPO，通过策略模型自身的置信度分数识别偏好关键 token 并仅对其优化，无需额外模型或计算开销，在 AlpacaEval 2 和 Arena-Hard 上一致优于均匀优化的 DAA 方法，同时缓解奖励黑客问题。

**[D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)**

:   本文提出 D-Fusion 方法，通过 mask 引导的自注意力融合（Self-Attention Fusion）构建视觉一致的偏好数据对并保留去噪轨迹，解决了 DPO 训练扩散模型时因视觉不一致导致效果受限的问题，在多种 RL 算法和 prompt 类型上显著提升了 prompt-image 对齐质量。

**[Diverging Preferences: When do Annotators Disagree and do Models Know?](diverging_preferences_when_do_annotators_disagree_and_do_models_know.md)**

:   本文系统分析了 RLHF 偏好数据集中标注者分歧的原因（建立了包含 10 个类别的分类法），发现超过 75% 的分歧源于个人偏好而非标注噪声，提出了分布式奖励模型（Mean-Var Reward Model）来有效区分分歧偏好与高一致偏好，并揭示了 LLM-as-Judge 评估方法在分歧情况下的系统性偏见。

**[DPO Meets PPO: Reinforced Token Optimization for RLHF](dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)**

:   本文提出 Reinforced Token Optimization (RTO)，将 RLHF 建模为 token 级别的 MDP（而非句子级 bandit），利用 DPO 隐式地提取 token-wise 奖励信号后用 PPO 进行策略优化，在 AlpacaEval 2 上比 PPO 高 7.5 分、在 Arena-Hard 上高 4.1 分，且仅需 1/8 数据量即可达到 PPO 级别性能。

**[Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)**

:   本文提出 Mixture of Agents Alignment（MoAA），利用多个开源 LLM 的集体智慧生成高质量的对齐数据（SFT 数据和偏好数据），显著提升目标模型在 Arena-Hard 和 AlpacaEval2 上的表现，并展示了无需外部强监督的自我提升能力。

**[Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day](instruction_tuning_of_large_language_models_for_tabular_data_generation-in_one_d.md)**

:   本文首次探索用指令微调提升 LLM 的表格数据生成能力，通过构建仅 10K 条高质量指令数据集并在单张 A100 上微调 Llama3.1-8B-Instruct 不到 6 小时，即可达到与 GPT-4o 相当的表格数据生成性能。

**[Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](layer-wise_alignment_examining_safety_alignment_across_image_encoder_layers_in_v.md)**

:   本文发现了 VLM 中图像编码器的"早退出"漏洞（ICET）——跳过图像编码器的部分层会大幅增加有害输出概率，提出 Layer-wise PPO (L-PPO) 修改 Clipped-PPO 算法在不同层级做多模态 RLHF，在 ASR 上降低高达 48%、毒性分数降低 33.64%。

**[M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality](m3hf_multi-agent_reinforcement_learning_from_multi-phase_human_feedback_of_mixed.md)**

:   提出 M³HF 框架，在多智能体强化学习训练过程中整合多阶段、混合质量的人类自然语言反馈，利用 LLM 解析反馈并通过预定义模板和自适应权重更新奖励函数，显著提升多智能体协作性能。

**[MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)**

:   本文提出 MMedPO，一种临床感知的多模态医学偏好优化方法，通过注入可信幻觉和局部病灶加噪构建多模态偏好数据，利用多个医学 LLM 协作评估临床相关性作为加权信号融入 DPO 训练，在 Med-VQA 和报告生成任务上分别平均提升 14.2% 和 51.7%。

**[Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](model_swarms_collaborative_search_to_adapt_llm_experts_via_swarm_intelligence.md)**

:   借鉴粒子群优化（PSO）算法，将多个 LLM 专家视为"粒子"，在权重空间中协作搜索，通过个体最优/全局最优/全局最差三个信号引导专家迭代移动，仅需 200 个样本即可实现无需微调的模型适配，在 9 个任务上平均超越 12 个基线 13.3%。

**[MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment](mpo_an_efficient_post-processing_framework_for_mixing_diverse_preference_alignme.md)**

:   提出 MPO（Mixing Preference Optimization），一个轻量级后处理框架，通过对数线性组合已有单目标策略来实现多偏好对齐，避免了多目标 RLHF 中昂贵的强化学习过程。

**[On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)**

:   提出 Batch-wise Sum-to-Zero Regularization (BSR)，通过约束每个 batch 内奖励分数之和为零来抑制隐状态范数的过度弥散，从根源上解决奖励模型的过优化问题，使 8B 规模 RM 在复杂偏好预测任务上超越 SOTA 5%+，并在 RLHF 下游训练中将生成长度降低 40% 同时提升 7% 胜率。

**[Preference Optimization for Combinatorial Optimization Problems](preference_optimization_for_combinatorial_optimization_problems.md)**

:   将RLHF中的偏好优化思想引入组合优化（COP），把定量奖励信号转化为定性偏好信号，结合熵正则化目标和局部搜索微调，在TSP/CVRP/FFSP等标准基准上实现了1.5x-2.5x的收敛加速和更优解质量。

**[Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)**

:   提出 NS-DPO，通过 Dynamic Bradley-Terry 模型引入单一指数衰减参数 γ 对训练数据进行时序加权，使 DPO 在偏好随时间漂移的场景下仍能鲁棒对齐，同时在平稳场景下不损失性能。

**[Self-Consistency Preference Optimization](self-consistency_preference_optimization.md)**

:   将推理时的自一致性(self-consistency)概念引入训练阶段，通过投票机制构建偏好对并使用加权DPO损失进行迭代训练，在无需金标签的情况下大幅提升LLM的数学和逻辑推理能力。

**[Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)**

:   提出 SmPO-Diffusion，通过平滑偏好建模替代二元偏好标签 + ReNoise Inversion 替代前向加噪估计，在大幅降低训练成本（比 DPO 快 6.5 倍，比 KTO 快 26 倍）的同时实现了 T2I 扩散模型偏好对齐的 SOTA 性能。

**[TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)**

:   将序列级PPO分解为一系列token级近端策略优化问题，并引入token级奖励引导函数 $f(\hat{r}(s_t, a_t))$ 来替代DPO中的固定常数 $\beta$，使不同token根据各自奖励值呈现不同程度的偏离参考策略，在MT-Bench/AlpacaEval 2/Arena-Hard上分别提升最多7.5/6.2/4.3个胜率点。
