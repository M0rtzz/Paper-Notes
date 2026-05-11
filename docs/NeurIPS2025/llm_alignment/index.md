---
title: >-
  NeurIPS2025 对齐 / RLHF方向53篇论文解读
description: >-
  53篇NeurIPS2025的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、LLM、对抗鲁棒、扩散模型、多模态、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🧠 NeurIPS2025** · **53** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (11)](../../ACL2026/llm_alignment/index.md) · [📷 CVPR2026 (10)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (42)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md) · [🧪 ICML2025 (27)](../../ICML2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×24 · LLM ×16 · 对抗鲁棒 ×6 · 扩散模型 ×3 · 多模态 ×3

**[A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)**

:   提出一种自适应 Alpha 聚合策略，在联邦 RLHF 框架中根据各用户群体的历史对齐表现动态调整奖励权重，从而在多元偏好对齐中同时实现高公平性和强对齐性能。

**[Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)**

:   提出基于任务并发（Task Concurrency）的LLM越狱攻击框架 JAIL-CON，通过在词级别交错编码有害任务和良性任务，利用LLM处理并发任务的能力绕过安全防护，同时产生的并发回答在guardrail下具有更强的隐蔽性。

**[Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)**

:   本文提出 CAID（Constrained Alignment via Iterative Dualization），通过迭代对偶方法交替更新 LLM 策略和对偶变量，在理论上证明了对偶方法可以找到最优约束 LLM 策略（至多存在参数化间隙），并在 PKU-SafeRLHF 数据集上显著改善了约束满足和 helpfulness-safety 权衡。

**[Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)**

:   提出基于不确定性的路由框架，用SNGP对pairwise reward model做不确定性量化，将高认知不确定性的样本路由到强LLM judge（DeepSeek-R1），在仅调用9.2%~42.5% judge的成本下显著超越随机路由的准确率，且有效改善下游在线RLHF对齐效果。

**[Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)**

:   提出两阶段微调攻击：第一阶段用10个问题配相同拒绝答案使LLM过拟合到窄最优解（尖锐loss landscape），第二阶段用相同10个问题配正常答案触发灾难性遗忘——安全对齐被"忘掉"，仅用完全良性数据即达94.84%越狱成功率，与恶意微调（97.25%）相当且完全绕过审核模型。

**[Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)**

:   建立了 DPO 在多元人类价值设定下的理论泛化框架——通过分析有限梯度步后 reward margin 的动态轨迹，证明了每种价值所需样本量必须随价值类别数 $K$ 对数增长（$Q = \Theta(\log K)$）才能维持泛化性能，揭示了对齐多元化社会价值的统计代价。

**[Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)**

:   提出奖励特征模型（RFM）：学习共享奖励特征 $\phi_\theta(x,y)$，每个用户通过线性权重 $\mathbf{w}_h$ 组合这些特征得到个性化奖励 $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$，并首次给出多评价者偏好学习的PAC泛化界，证明增加评价者数 $m$ 比增加每人样本数 $n$ 更有效，仅30个样本即可快速适应新用户。

**[DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)**

:   提出DeepVideo-R1，将GRPO重新表述为回归优势值的Reg-GRPO（消除clipping/min等保护机制），同时通过难度感知数据增强缓解优势值消失问题，在视频推理任务上相比标准GRPO提升高达10.1个百分点。

**[DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](densedpo_finegrained_temporal_preference_optimization_for_vi.md)**

:   识别并解决视频 DPO 的运动偏差问题——通过从 GT 视频加噪去噪构造结构对齐的视频对来固定运动维度、在时间片段级标注密集偏好来获取更精准的学习信号、用现成 VLM 自动标注来降低成本，仅用 1/3 标注数据即大幅提升运动生成质量同时匹配视觉质量和文本对齐。

**[Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)**

:   提出 Latent Reward Model (LRM) 和 Latent Preference Optimization (LPO)，将预训练扩散模型本身复用为噪声感知的潜空间奖励模型，在噪声潜在空间直接进行步级偏好优化，相比 Diffusion-DPO 实现 10-28× 训练加速，相比 SPO 实现 2.5-3.5× 加速。

**[DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)**

:   提出 DP²O-SR 框架，利用扩散模型固有的随机性生成多样化超分辨率输出，通过混合感知奖励构建偏好对，并设计层次化偏好优化（HPO）策略自适应加权训练对，在无需人工标注的前提下显著提升真实世界图像超分辨率的感知质量。

**[EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)**

:   提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度函数 + 模拟退火）自动生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，构建的 EvoRefuse-Test 基准比最强基线拒绝触发率高 85.34%、词汇多样性高 34.86%，EvoRefuse-Align 对齐数据通过 SFT/DPO 微调将过度拒绝降低 29.85%-45.96% 且不损害安全性。

**[From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](from_judgment_to_interference_early_stopping_llm_harmful_outputs_via_streaming_c.md)**

:   提出 Streaming Content Monitor (SCM)——首个原生支持部分检测的流式有害内容监控器，通过 FineHarm 数据集（29K 样本含 token 级标注）和层次一致性感知学习，平均仅需看到 18% 的 response tokens 即可达到 0.95+ macro F1，实现对 LLM 有害输出的实时早停。

**[g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)**

:   针对蛋白质语言模型（PLM）应用 DPO 时偏好对数量随样本数二次增长导致训练不可扩展的问题，提出 g-DPO 框架：(1) 通过序列空间 union mask 聚类剪枝冗余偏好对，保留局部邻域中信息量更大的比较；(2) 利用共享 union mask 的分组似然摊销，一次前向传播同时计算组内所有序列的 log-likelihood。在三个蛋白质工程任务上，g-DPO 保持与标准 DPO 统计上不可区分的 in silico 和 in vitro 性能，同时实现 1.7-5.4× 的训练加速。

**[GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)**

:   提出GASP框架，通过训练专用的SuffixLLM生成可读的对抗后缀，利用潜在贝叶斯优化（LBO）在连续嵌入空间中高效搜索并用ORPO迭代微调生成器，在完全黑盒设置下实现高攻击成功率且生成的后缀保持人类可读性。

**[Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models](generalizing_while_preserving_monotonicity_in_comparison-based_preference_learni.md)**

:   提出 **Linear GBT with Diffusion Prior**，一类在保证**单调性**（偏好比较后被偏好方的分数不会反常下降）的同时能**泛化到未比较数据**的偏好学习模型，首次正面回答了"泛化与单调性能否兼得"的核心问题。

**[Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)**

:   证明了在KL正则化的RLHF设置下，直接使用经验估计的贪心采样（无需构建乐观/悲观估计）就能在在线和离线两种设置中实现$O(\log T)$遗憾界和$O(\varepsilon^{-1})$样本复杂度，这是首次在一般偏好模型下达到这些阶数。

**[GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)**

:   通过将 KL 约束奖励最大化的解析解融入梯度权重（零和权重消除配分函数），设计了比 GRPO 更稳定的 LLM 后训练方法 GVPO，在 AIME 上达到 20.72%（GRPO 14.79%），并证明具有唯一全局最优解。

**[Human-assisted Robotic Policy Refinement via Action Preference Optimization](human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)**

:   提出 Action Preference Optimization (APO)，通过人机协作框架收集交互轨迹，利用基于前景理论的二元期望信号和自适应重加权方法对 VLA 模型进行偏好对齐优化，使其能从失败中学习并持续迭代改进。

**[Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)**

:   提出两种互补技术提升 LLM 强化微调（GRPO）的数据效率：(1) DOTS——基于注意力机制预测自适应难度，优先选择中等难度问题以最大化梯度信号；(2) Rollout Replay——复用近期 rollout 降低每步计算开销。两者结合在 6 个模型-数据集组合上平均减少 40.7% 训练时间。

**[Inference-time Alignment in Continuous Space](inference-time_alignment_in_continuous_space.md)**

:   提出 Simple Energy Adaptation (SEA)，将推理时对齐从"离散空间搜索"范式转变为"连续空间优化"范式，通过在连续 logit 空间上进行基于梯度的 Langevin 采样来逼近 RLHF 最优策略，在 AdvBench 上相对最优基线提升 77.51%，在 MATH 上提升 16.36%。

**[Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)**

:   提出基于策略（而非示例）的 LLM 红队评估框架和 Jailbreak-Zero 方法，通过简单的大规模并行采样策略（无需人工越狱策略），在 HarmBench 上对 GPT-4o 和 Claude 3.5 分别达到 99.5% 和 96.0% 的攻击成功率，同时通过微调实现覆盖率、多样性和保真度三个目标的 Pareto 最优。

**[KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)**

:   提出 ε-DPO，通过观察训练时扰动 β 后 logit 作为偏好模型的单调性，实现实例级自适应 KL 惩罚控制，无需额外计算开销即可显著超越 DPO 及大多数直接对齐算法，在 AlpacaEval 2 上达到 46.4% LC win rate（DPO 仅 40.3%）。

**[LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)**

:   将多个奖励模型（RM）的选择建模为上下文多臂老虎机（LinUCB）问题，在迭代 LLM 训练中自适应地为每个 batch 选择最合适的 RM，在推理、指令跟随和长上下文任务上以 2-3 倍效率优势全面超越 RM 集成和单 RM 基线。

**[Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)**

:   提出 LENS 框架，通过在 LLM 嵌入的潜在空间中利用 VAE 合成偏好数据对，绕过昂贵的文本生成过程，以极低计算成本（模型缩小 16000 倍、生成速度提升 18 倍）显著提升 reward model 性能。

**[LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)**

:   建立统一理论框架证明 RLHF/DPO/KTO/BCO 等对齐方法本质上是在估计安全分布 $\mathcal{D}^+$ 与不安全分布 $\mathcal{D}^-$ 之间的散度，由此解释了对齐后隐空间分离现象，并提出基于 KL 散度的 KLDO 对齐方法，在 5 个模型上实现最佳鲁棒性。

**[LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)**

:   LongVPO提出两阶段DPO框架，Stage 1通过锚定短片段构造伪长视频偏好数据并引入anchor-only参考模型近似解决上下文长度不匹配问题，Stage 2通过递归字幕生成和多片段推理任务在真实长视频上自训练，仅用16K合成样本即超越大规模监督训练的长视频模型。

**[Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)**

:   将多方偏好聚合的 RLHF 微调建模为机制设计问题，证明了在社会福利最大化训练规则下各方有动机虚报偏好，并通过扩展 VCG 支付机制实现了占优策略激励相容（DSIC），确保各方如实报告偏好。

**[MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation](metadefense_defending_finetuning-based_jailbreak_attack_before_and_during_genera.md)**

:   提出 MetaDefense，一个两阶段（生成前+生成中）防御框架，通过训练 LLM 自身预测查询和部分响应的有害性来防御基于微调的越狱攻击，无需额外分类器，实现 2× 内存效率。

**[Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)**

:   提出 SymMPO（对称多模态偏好优化），通过对比图像的对称配对偏好学习和偏好边际一致性正则化，解决了现有视觉增强型 DPO 方法中目标函数不严格和间接偏好监督两大局限，在五个幻觉评测基准上取得了一致的性能提升。

**[Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability](multi-environment_pomdps_discrete_model_uncertainty_under_partial_observability.md)**

:   系统研究了多环境 POMDP（ME-POMDP）——一类共享状态/动作/观测空间但转移、观测和奖励函数可任意不同的 POMDP 集合，目标是找到在最坏情况环境下最大化奖励的鲁棒策略。通过引入对抗信念 POMDP（AB-POMDP）统一建模，并证明其与单侧部分可观测随机博弈（POSG）的等价关系，提出精确（值迭代 + LP）和近似（AB-HSVI）算法。

**[On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)**

:   将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

**[ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)**

:   提出 ORPO-Distill，将跨架构 LLM 知识蒸馏重新定义为偏好优化问题：使用教师模型生成正样本推理链、学生模型生成负样本推理链，通过 ORPO 对比损失训练，并引入混合策略（mixed-policy）更新学生负样本，在 5 个 QA 基准上一致超越黑盒 KD 基线。

**[PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)**

:   提出 PolyJuice，首个面向合成图像检测器（SID）的黑盒、图像无关的红队方法，通过在 T2I 模型潜空间中发现并利用"真实感方向"，以通用方式引导生成图像欺骗检测器，成功率高达 84%。

**[Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)**

:   本文将 RLHF 中反复出现的安全-公平-效率冲突形式化为「对齐三难困境」：证明了没有任何 RLHF 系统能同时满足 $\varepsilon$-代表性（忠实反映多元价值）、多项式可处理性（计算可行）和 $\delta$-鲁棒性（抵御对抗攻击），从而为当前 RLHF 系统中偏好坍缩、谄媚等病理现象提供了统一的复杂度理论解释。

**[Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)**

:   系统研究了将谎言检测器（lie detector）整合到LLM偏好学习标注流程中的效果（SOLiD框架），发现训练后模型是变得诚实还是学会规避检测取决于三个关键因素：探索程度（GRPO vs DPO）、检测器准确率（TPR）和KL正则化强度。

**[Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)**

:   将 DPO 重新解释为似然比估计（ratio matching）问题，基于 Bregman 散度框架提出 BPO（Bregman Preference Optimization），包含 DPO 为特例的广义损失函数族，并设计了 SBA（Scaled Basu's Power Divergence）实例，在 Llama-3-8B 上实现 55.9% AlpacaEval2 length-controlled win rate 的 SOTA。

**[Provably Efficient Online RLHF with One-Pass Reward Modeling](provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)**

:   提出一种基于 online mirror descent（OMD）的 one-pass reward modeling 方法，消除了 online RLHF 中需要存储历史数据并重新从头优化的计算瓶颈，实现每次迭代 $\mathcal{O}(1)$ 的时间和存储复杂度，同时在统计效率上也优于 MLE 方法。

**[Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)**

:   RL 微调 LLM 时实际上只更新了 5%-30% 的参数（稀疏子网络），且该子网络在不同种子、数据和算法间具有高度一致性，仅微调子网络即可复现完整微调的模型性能甚至参数值。

**[ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)**

:   提出 ResponseRank 方法,通过利用偏好强度的代理信号（如响应时间和标注者一致性）的局部相对差异来鲁棒地学习效用差值,显著提升奖励模型的样本效率。

**[Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)**

:   针对扩散模型中 DPO 的两个核心问题——有限探索和奖励尺度不平衡,提出稳定参考模型更新策略和时间步感知训练策略,可集成到各种偏好优化算法中。

**[Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)**

:   通过分布鲁棒优化（DRO）框架提出 WDPO（Wasserstein）和 KLDPO（KL散度）两种鲁棒 DPO 变体，解决用户偏好分布转移导致的对齐失败问题，提供 $O(n^{-1/4})$ 收敛保证，在多维对齐任务和 OpenLLM 榜单上显著优于标准 DPO。

**[SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)**

:   通过分析多模态 LLM 中有害 token 的传播机制，发现不到 1% 的 token 在早期-中间层引发越狱行为，由此提出无需训练的 SafePTR 框架，在脆弱层剪枝有害 token 并在后续层恢复良性特征，显著提升安全性而不牺牲任务性能。

**[SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)**

:   首次系统性地将安全强化学习（SafeRL）的 CMDP 框架应用于视觉-语言-动作模型（VLA）的安全对齐，通过建模-激发-约束-保证四阶段集成安全方法（ISA），在移动操作任务上实现 83.58% 的安全违规成本下降同时保持任务性能（+3.85%）。

**[Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)**

:   提出 RRPO（Refined Regularized Preference Optimization），通过子序列级细粒度奖励和 token 级 KL 正则化替代 DPO 的响应级奖励，结合自对齐数据生成框架，在视频理解任务上减少幻觉、提升时间推理能力。

**[Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)**

:   理论证明并实验验证：防御长度 $\Theta(M)$ 的后缀越狱攻击，只需要在长度 $\Theta(\sqrt{M})$ 的对抗后缀上做对抗训练即可，即"短对抗训练防长越狱"——在5个主流LLM上，20 token 对抗训练可将 120 token 越狱成功率降低至少 30%。

**[Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)**

:   发现 NPO（负偏好优化）中的参考模型偏差导致遗忘数据的优化功率分配不均和早期梯度权重平滑失效，提出 SimNPO 通过去除参考模型依赖并采用长度归一化奖励，在 TOFU 上将 FQ 从 0.79 提升至 0.99，在所有基准上一致优于 NPO。

**[Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)**

:   首次从机制设计角度研究 RLHF 中多标注者策略性操纵问题，证明了策略防操纵（strategyproofness）与政策对齐之间存在根本性权衡，并提出 Pessimistic Median of MLEs 算法实现近似策略防操纵。

**[T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)**

:   提出 T-SHIRT 数据选择框架，通过 Selective IFD（仅考虑有信息量的 token）和分层选择策略（偏好邻域一致性高的样本），用 5% 数据微调即可超越全量数据训练，同时仅需 GPT-2 和单 GPU 40 分钟完成选择。

**[Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)**

:   通过机制可解释性视角发现 LLM 中约 5% 的稀疏"安全神经元"，仅修补（patching）这些神经元的激活即可恢复 90% 以上的安全性能，并从神经元重叠角度解释了 alignment tax 现象。

**[Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training](trajectory_balance_with_asynchrony_decoupling_exploration_and_learning_for_fast_.md)**

:   提出 TBA（Trajectory Balance with Asynchrony），将 GFlowNet 的轨迹平衡（TB）目标与异步分布式 RL 架构结合，实现 LLM 后训练中探索与学习的解耦，在数学推理、偏好微调和自动红队测试任务上获得 4-50 倍加速且性能不降反升。

**[Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning](trajectory_bellman_residual_minimization_a_simple_value-based_method_for_llm_rea.md)**

:   TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。

**[What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)**

:   从优化理论角度证明：奖励模型的准确率（accuracy）不足以衡量其作为 RLHF "教师"的质量——即使完美准确的奖励模型，如果诱导的奖励方差（reward variance）过低，也会导致 RLHF 目标函数景观平坦，使 policy gradient 优化极慢；不同的语言模型需要不同的奖励模型。
