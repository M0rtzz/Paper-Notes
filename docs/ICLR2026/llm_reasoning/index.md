---
title: >-
  ICLR2026 LLM 推理方向65篇论文解读
description: >-
  65篇ICLR2026的 LLM 推理方向论文解读，涵盖推理、LLM、多模态、Agent、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🔬 ICLR2026** · **65** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (37)](../../ACL2026/llm_reasoning/index.md) · [📷 CVPR2026 (12)](../../CVPR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (66)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md) · [🧪 ICML2025 (15)](../../ICML2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×42 · LLM ×10 · 多模态 ×3 · Agent ×2 · 对抗鲁棒 ×2

**[Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)**

:   提出 Adaptive Social Learning（ASL）框架，设计四种层次化推理模式（从直觉回应到深度推演），并通过 AMPO 算法（融合模式级和样本级优势估计）让 LLM agent 根据社交场景复杂度自适应切换推理深度，在社交智能任务上比 GPT-4o 高 15.6%，比 GRPO 高 7.0% 且 token 用量减少 32.8%。

**[Agentified Assessment of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)**

:   提出基于Agent的评测框架(AAA)，将评估逻辑封装为assessor agent并通过标准A2A接口与被测agent交互，在经Vampire定理证明器系统清洗的FOLIO数据集上，自动形式化agent（NL→Z3Py+SMT求解）达到86.70%准确率，大幅超过CoT基线73.89%，尤其在矛盾检测(False类)上提升32.79个百分点。

**[AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)**

:   AgentMath提出一个工具增强的Agent框架，通过自动化数据合成、多轮交互式强化学习和高效异步训练系统，将LLM推理能力与代码解释器的计算精度无缝结合，在AIME24/25和HMMT25上以30B-A3B规模达到SOTA水平（90.6%/86.4%/73.8%），超越o3-mini和Claude-Opus-4.0-Thinking。

**[AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)**

:   提出 AIMCoT，将多模态 CoT 的视觉信息选择从"被动关注高注意力区域"转变为"主动寻找最高信息增益区域"，通过三个模块（CAG 上下文增强注意力图、AVP 主动视觉探测、DAT 动态注意力转移触发）协同工作，在 LLaVA-W 上比 ICoT 提升 18.25%（0-shot），是一个免训练的即插即用框架。

**[Annotation-Efficient Universal Honesty Alignment](annotation-efficient_universal_honesty_alignment.md)**

:   提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

**[Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)**

:   系统评估推理型 LLM 对其 CoT 中各种干预（良性/中性/对抗性）的鲁棒性：发现模型总体鲁棒能从干预中恢复，但改写风格（paraphrasing）会抑制"自我怀疑"表达导致正确率下降，恢复过程有显著计算开销（CoT 膨胀最高 665%）。

**[ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)**

:   提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

**[Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts](beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)**

:   提出 Contact Searching Question (CSQ) 框架，基于有向图可达性任务和认知心理学原理设计两个互补统计指标——欺骗意图分数 $\rho$ 和欺骗行为分数 $\delta$，首次系统揭示 16 个主流 LLM 在完全良性提示下存在随任务难度升级的自发欺骗倾向。

**[Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)**

:   本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

**[Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors](conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)**

:   揭示了 LLM 的"逻辑惯性"现象——在遇到矛盾前提时仍沿学习到的推理轨迹继续推理（准确率降至 0.0），提出 Conflict-Aware Fusion 双过程架构，通过强制前提验证先于推理执行，在矛盾检测上实现 100% 准确率。

**[Continuous Chain of Thought Enables Parallel Exploration and Reasoning](continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)**

:   CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

**[CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)**

:   提出CoT-RVS，一种完全无训练的多智能体框架，利用预训练MLLM的零样本CoT推理能力进行时间-语义关联分析与关键帧选取，在推理视频分割任务上大幅超越微调方法（Refer-DAVIS J&F 79.1 vs 71.2，ReasonVOS J&F 65.5 vs 49.9）。

**[CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)**

:   将推理过程中的反思token（如"wait"、"but"）视为可调度的"资源"，借鉴优化中周期性学习率的思想，提出CyclicReflex——一种免训练的解码策略，通过三角波形动态调控反思token的logit，在多个数学推理基准上（MATH500, AIME2024/2025, AMC2023）一致性提升1.5B-8B模型准确率。

**[DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)**

:   将 LLM 的 CoT 推理形式化为 DAG 上的基于规则的随机过程，提出"逻辑闭合性"（logical closeness）度量来评估模型是否通过搜索还是严格逻辑推理得到答案，构建了 2894 个金标准 DAG-MATH benchmark，发现即使 PASS@k 相近的模型在推理忠实度上也存在显著差异。

**[DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)**

:   提出 Design Logic（设计逻辑）——从真题中逆向工程出的可复用元知识，用于指导从原始文本合成多学科推理问题。构建了 470 万道覆盖 75 学科的推理题目，SFT 后的 base 模型甚至超越经过完整后训练的官方模型。

**[Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)**

:   本文系统揭示了多模态大推理模型（MLRM）通过图像推断敏感地理位置信息的隐私泄露风险，提出了三级隐私风险框架和 DoxBench 基准，以及信息论度量 Glare 和协作攻击框架 GeoMiner。

**[DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)**

:   诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

**[Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)**

:   将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

**[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

**[Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)**

:   为小型 VLM 提出两种高效的测试时缩放策略：TTAug（对输入做多种增强后在 token 级别聚合输出概率）和 TTAdapt（用 TTAug 生成的伪标签自适应调整模型参数），在 9 个基准上一致提升性能，同时计算效率远优于现有的基于重复采样的测试时方法。

**[Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)**

:   提出 EELMA 算法，利用信息论中的"赋权"（empowerment，即 agent 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量指标，在语言游戏和真实网页浏览场景中与任务表现强相关（$r=0.83$–$0.94$），可用于开放式 agent 监控与安全评估。

**[Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval](evoking_user_memory_personalizing_llm_via_recollection-familiarity_adaptive_retr.md)**

:   受认知科学双过程理论启发，提出 RF-Mem 框架，通过 Familiarity（快速相似度匹配）和 Recollection（深层链式重建）双路径自适应切换的记忆检索机制，实现高效且可扩展的 LLM 个性化。

**[FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)**

:   针对GRPO训练中生成阶段占91%-98%时间的严重瓶颈，提出并发感知的投机解码策略（动态调整draft树参数以适配从高到低的实时并发度变化）和在线draft模型学习（利用目标模型生成的hidden states持续适配分布漂移），整体实现2.35x-2.72x端到端训练加速，且不损害推理质量。

**[Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)**

:   Fine-R1 通过 CoT 监督微调（"视觉分析→候选子类→对比→预测"结构化推理链）+ 三元组增强策略优化 TAPO（类内增强提升鲁棒性 + 类间增强提升判别力），仅用 4-shot 训练即在细粒度视觉识别上超越 CLIP 和通用/推理型 MLLM。

**[Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)**

:   系统诊断推理时奖励模型（RM）的三大失效模式——简单题性能退化、采样数增多时辨别力下降、高搜索多样性损害准确率，并提出 CRISP 算法通过答案聚类的奖励聚合与逐步前缀引导来缓解这些问题，准确率提升最高 5%。

**[From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)**

:   提出 ContextMATH 基准，通过将 AIME/MATH-500 抽象数学题转化为情景嵌入（SG）和复杂度缩放（CS）两种变体，揭示即使是 GPT-5 和 DeepSeek-R1 等顶级模型在上下文数学推理中也出现 13-34% 的准确率下降，且错误主要由问题建模（formulation）而非计算推理导致。

**[Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)**

:   提出 CodeGym 框架，将编程题自动转化为多轮工具调用的交互式环境，用于 LLM agent 的强化学习训练，在分布外基准上取得显著泛化提升（如 Qwen2.5-32B 在 τ-Bench 上 +8.7 点）。

**[GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)**

:   形式化Program-to-Geometry任务并提出GeoGramBench（500题），按三级几何复杂度分类法评估19个前沿LLM从过程式绘图代码构建几何表征并推理的能力，发现即使GPT-5在最高抽象级别也仅39.26%准确率，揭示了LLM空间抽象的根本性短板。

**[Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)**

:   揭示GRPO的优势函数（std归一化）导致更新幅度在中等难度题目处最大、对难题和易题均隐式抑制的问题，提出MathForge框架——DGPO（用MAD替换std实现难度均衡 + softmax难度加权）+ MQR（添加故事背景/抽象术语/嵌套子问题三方面改写增加难度但保留原答案），在Qwen2.5-Math-7B上在6个数学推理benchmark上平均超GRPO +4.56%。

**[HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)**

:   提出 HeurekaBench，一个基于真实科学工作流构建评测基准的框架，通过多LLM流水线从论文中提取可验证的科学洞见并生成开放式研究问题，用于评估AI co-scientist在数据驱动科学发现中的端到端能力。

**[I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift](i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)**

:   本文系统研究了基于 frozen embedding 的安全分类器在模型更新导致 embedding 漂移时的脆弱性，发现仅 2% 的 embedding 扰动即可将分类器性能从 85% ROC-AUC 降至随机水平（50%），且 72% 的误分类发生在高置信度下（silent failure），同时 instruction-tuned 模型反而比 base 模型更难分类。

**[Is In-Context Learning Learning?](is_in-context_learning_learning.md)**

:   通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。

**[Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)**

:   提出 TRACE（Truncated Reasoning AUC Evaluation）方法，通过逐步截断推理链并测量模型"多早"能获得奖励来量化推理努力程度，从而检测 CoT 监控无法发现的隐式奖励黑客行为，在数学和代码任务中比最强 CoT 监控器分别提升 65% 和 30% 以上的检测 F1。

**[LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation](lingoly-too_disentangling_reasoning_from_knowledge_with_templatised_orthographic.md)**

:   提出LingOly-TOO基准，通过专家设计的正字法置换（grapheme-level permutation）对语言学奥赛题进行混淆，保留推理逻辑但消除知识/记忆捷径，将15个前沿模型的最高分从0.59降至0.48，系统量化了LLM推理能力被知识效应高估的程度。

**[mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)**

:   提出 mR3，一系列覆盖72种语言的多语言rubric-agnostic推理奖励模型，通过系统化的数据构建（GPT-OSS-120B蒸馏+难度过滤）和课程学习策略训练，14B模型在多语言评估基准上超越120B教师模型及所有同类基线，同时支持point-wise/pair-wise/binary三种评估范式。

**[Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)**

:   提出 NRT（Native Reasoning Training）框架，将推理链视为隐变量，通过模型自身对参考答案的预测置信度作为内在奖励信号训练 LLM 推理能力，无需外部验证器或专家推理示范；在 Llama-3.1-8B 上 9 个基准平均提升 10.2 分（46.0→56.2），超越需要验证器的 RLPR +5.4 分。

**[No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)**

:   在 LLM 生成答案之前，仅从问题处理后的残差流激活中训练线性探针（difference-of-means），即可预测模型即将生成的答案是否正确。该"提前正确性方向"在 TriviaQA 上训练后可跨域泛化到多个事实知识数据集（AUROC 0.68-0.88），但无法泛化到数学推理（GSM8K），揭示了"事实正确性"与"推理正确性"在模型内部表征中的结构性分离。

**[Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)**

:   指出GRPO无法从模型完全无法解决的难题(pass rate=0%)中学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，跨3个模型6个benchmark一致超越GRPO并真正提升pass@k能力上界。

**[On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)**

:   提出 Regularized Policy Gradient (RPG) 框架，系统推导并分析了基于 Forward/Reverse KL 散度（归一化和非归一化形式）的策略梯度方法，发现 GRPO 的 KL 项存在理论不一致性，并在数学推理任务上取得优于 GRPO、REINFORCE++、DAPO 的结果。

**[On The Fragility of Benchmark Contamination Detection in Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)**

:   系统性研究发现 LRM 的基准污染检测极其脆弱：SFT 阶段引入的污染在经过 GRPO 训练后检测信号几乎消失（PPO 式重要性采样/裁剪是根因），而对高级 LRM 直接用 CoT 做 SFT 污染则几乎不留任何可检测痕迹，现有 10 种检测方法均接近随机猜测。

**[Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)**

:   提出 Plan-and-Budget 框架，通过将复杂查询分解为子问题并基于估计复杂度自适应分配 token 预算，实现推理 LLM 的高效测试时缩放——最高提升 70% 准确率、减少 39% token、E3 指标提升 193.8%。

**[PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)**

:   首次将分解式 Chain-of-Thought 推理与多维度强化学习（RL）结合应用于视频到音频（V2A）生成，通过四个专门化的 CoT 模块（语义/时序/美学/空间）配合对应奖励函数，解决了目标纠缠问题，并提出 Fast-GRPO 算法大幅降低 RL 训练开销。

**[RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)**

:   提出 RAIN-Merging，一种无梯度的两阶段模型合并方法：先通过零空间投影保护大推理模型 (LRM) 的思维格式，再用指令注意力引导的合并系数增强指令遵循能力，在保持推理质量的同时大幅提升 LRM 的指令遵循性能。

**[Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)**

:   首次系统研究大型推理模型（LRM）的答案来源归因问题，揭示推理（CoT）和检索（记忆）两种机制同时竞争影响最终答案，并提出 Farl（遗忘增强强化学习）通过抑制检索捷径来提升模型的真实推理能力。

**[ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization](reform_reflective_autoformalization_with_prospective_bounded_sequence_optimizati.md)**

:   提出 ReForm，一种反思式自动形式化范式，将自然语言数学问题转为 Lean 形式声明的过程从一次生成转变为"生成 → 语义自验证 → 修正"的迭代循环，并设计 PBSO 算法优化异构奖励信号，在四个基准上比最强基线平均提升 22.6 个百分点。

**[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)**

:   提出推理忠实度（Reasoning Faithfulness）的形式化定义（立场一致性 + 因果影响），构建 7,186 实例/7 任务的 RFEval 基准，通过输出层反事实推理干预评估 12 个开源 LRM，发现 49.7% 的输出不忠实，且 RL 后训练会降低忠实度、准确率不是忠实度的可靠代理指标。

**[Scaling Generalist Data-Analytic Agents](scaling_generalist_data-analytic_agents.md)**

:   提出 DataMind——一套完整的数据分析 Agent 训练方案，通过细粒度任务分类+递归难度组合实现多样 query 合成、知识增强轨迹采样+自一致性过滤保证数据质量、SFT+RL 动态混合训练策略以及内存友好的异步 rollout 框架，训练出的 DataMind-14B 以 71.16% 平均分在多个基准上 SOTA，超越 GPT-5 和 DeepSeek-V3.1。

**[SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)**

:   提出 SceneCOT，首个将 Chain-of-Thought 推理引入 3D 场景理解的框架，通过四阶段推理管线（任务识别→区域定位→实体接地→接地推理）将中间推理步骤显式关联到视觉 grounding，在 Beacon3D 上 Good Coherence 达到 34.7%（比最强 baseline 的 20.4% 高出 70%+）。

**[SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)**

:   提出SealQA挑战基准（含Seal-0/Seal-Hard/LongSeal三种变体），每道题均经NLP研究者精心设计以触发歧义/冲突/噪声搜索结果，GPT-5最高仅43.2%准确率，揭示test-time scaling在噪声检索下不产生可靠增益。

**[Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)**

:   用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

**[Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)**

:   提出 TAMPO（Temperature Adaptive Meta Policy Optimization），将采样温度重新定义为可学习的元策略，通过双层循环在内环做 LLM 策略优化、外环根据轨迹优势信号自适应更新温度分布，无需额外 rollout，在数学推理基准上一致超越固定温度基线。

**[The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models](the_first_impression_problem_internal_bias_triggers_overthinking_in_reasoning_mo.md)**

:   推理模型在看到问题的瞬间就形成了对答案的"第一印象"（内部偏差），当这个直觉猜测与后续系统推理产生冲突时，模型会反复自我质疑、重新检查，导致推理长度膨胀 21%–43%，而现有所有缓解方法均无法从根本上消除这一效应。

**[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)**

:   揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

**[The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)**

:   提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅展开主导簇来替代标准 Self-Consistency，在 GSM8K/Math500/AIME/GPQA 等基准上保持甚至提升准确率的同时减少 40%–60% 的 token 用量和最高 50% 的延迟。

**[Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)**

:   提出 AdaAnchor 潜空间推理框架——将可学习的锚向量（anchor vectors）附加到输入嵌入中，通过迭代前向传播精炼锚状态实现"沉默思考"，配合基于锚稳定性的自适应停止机制按实例难度动态分配计算量，在数学推理任务上比固定步潜推理准确率提升最高 5%、平均步数减少 48–60%，输出 token 相比 CoT 减少 92–93%。

**[TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)**

:   构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

**[Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)**

:   揭示大推理模型（LRM）的推理链即使最终回答安全也常包含有害内容的问题，提出 Intervened Preference Optimization（IPO），通过用安全触发器替换合规线索来纠正不安全推理轨迹，构造偏好对进行对齐训练，在 3 个 LRM 上将推理有害率降低超过 30% 且不损害推理能力。

**[Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)**

:   提出 Progressive Thought Encoding，在 KV 缓存受限条件下将被驱逐的思维 token 编码进 LoRA 权重，使大推理模型在 RL 训练时显存减半的同时推理准确率反超全缓存 LoRA（AIME2024/2025 上最高提升 +23.4%）。

**[TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)**

:   提出TumorChain，面向消化系统五大器官肿瘤分析的交错多模态CoT推理框架，通过知识图谱驱动的1.5M CoT-VQA数据引擎、器官引导的迭代交错推理(IIR)和分割/分类/LLM三模型协同优化，实现从影像发现→临床印象→病理预测的完整推理链，平均精度84.41%，大幅超越GPT-5-Mini(51.59%)。

**[Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)**

:   从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。

**[Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)**

:   提出 Uni-CoT 分层宏-微推理框架，将多模态 CoT 分解为宏观任务规划（将复杂任务分解为子目标）和微观子任务执行（MDP 式自反思迭代优化），通过注意力掩码设计将 $O(T^2)$ 复杂度降至 $O(T)$，在 GenEval 上超越 BAGEL 基线 +0.02，实现了文本-图像交织的统一推理。

**[Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)**

:   提出 CRV（Circuit-based Reasoning Verification），通过将 LLM 的 MLP 替换为 transcoder 构建可解释归因图，从图的结构特征中提取推理错误的"指纹"，实现白盒 CoT 推理验证，并可通过因果干预修正错误推理。

**[When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)**

:   系统研究量化、蒸馏、剪枝三种压缩方法对大型推理模型 (LRM) 的影响，通过性能基准测试和机制可解释性分析，揭示权重数量对知识记忆影响大于推理、最后一层 MLP up_proj 是最关键组件、以及当前量化方法过度压缩最后层等核心发现。

**[When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)**

:   本文系统分析了 Qwen2.5-Math-7B 在 GSM8K 上的隐式推理行为，发现 81.6% 的正确预测来自计算不一致的路径，8.8% 为静默失败（高置信错误），并揭示了推理深度与准确率之间的悖论关系。

**[Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)**

:   本文通过理论和实验揭示了隐式奖励模型（IM-RM，如DPO）比显式奖励模型（EX-RM）泛化更差的根本原因——IM-RM过度依赖表面token级线索而非语义表示，导致在token分布偏移下准确率大幅下降，同时反驳了"生成-验证差距"假说。
