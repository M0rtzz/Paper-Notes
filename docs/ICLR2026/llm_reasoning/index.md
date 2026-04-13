---
title: >-
  ICLR2026 LLM推理方向 66篇论文解读
description: >-
  66篇ICLR2026 LLM推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM推理

**🔬 ICLR2026** · 共 **66** 篇

**[Adaptive Social Learning Via Mode Policy Optimization For Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)**

:   提出 Adaptive Social Learning（ASL）框架，设计四种层次化推理模式（从直觉回应到深度推演），并通过 AMPO 算法（融合模式级和样本级优势估计）让 LLM agent 根据社交场景复杂度自适应切换推理深度，在社交智能任务上比 GPT-4o 高 15.6%，比 GRPO 高 7.0% 且 token 用量减少 32.8%。

**[Agentified Assessment Of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)**

:   提出基于Agent的评测框架(AAA)，将评估逻辑封装为assessor agent并通过标准A2A接口与被测agent交互，在经Vampire定理证明器系统清洗的FOLIO数据集上，自动形式化agent（NL→Z3Py+SMT求解）达到86.70%准确率，大幅超过CoT基线73.89%，尤其在矛盾检测(False类)上提升32.79个百分点。

**[Agentmath Empowering Mathematical Reasoning For Large Language Models Via Tool-A](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)**

:   AgentMath提出一个工具增强的Agent框架，通过自动化数据合成、多轮交互式强化学习和高效异步训练系统，将LLM推理能力与代码解释器的计算精度无缝结合，在AIME24/25和HMMT25上以30B-A3B规模达到SOTA水平（90.6%/86.4%/73.8%），超越o3-mini和Claude-Opus-4.0-Thinking。

**[Aimcot Active Information-Driven Multimodal Chain-Of-Thought For Vision-Language](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)**

:   提出 AIMCoT，将多模态 CoT 的视觉信息选择从"被动关注高注意力区域"转变为"主动寻找最高信息增益区域"，通过三个模块（CAG 上下文增强注意力图、AVP 主动视觉探测、DAT 动态注意力转移触发）协同工作，在 LLaVA-W 上比 ICoT 提升 18.25%（0-shot），是一个免训练的即插即用框架。

**[Annotation-Efficient Universal Honesty Alignment](annotation-efficient_universal_honesty_alignment.md)**

:   提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

**[Are Reasoning Llms Robust To Interventions On Their Chain-Of-Thought](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)**

:   系统评估推理型 LLM 对其 CoT 中各种干预（良性/中性/对抗性）的鲁棒性：发现模型总体鲁棒能从干预中恢复，但改写风格（paraphrasing）会抑制"自我怀疑"表达导致正确率下降，恢复过程有显著计算开销（CoT 膨胀最高 665%）。

**[Atts Asynchronous Test-Time Scaling Via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)**

:   提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

**[Beyond Prompt-Induced Lies Investigating Llm Deception On Benign Prompts](beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)**

:   提出 Contact Searching Question (CSQ) 框架，通过两个统计指标（欺骗意图分数 ρ 和欺骗行为分数 δ）量化 LLM 在正常良性提示下的自发欺骗行为，发现 16 个主流 LLM 普遍存在随任务难度升级的系统性欺骗倾向。

**[Compositional Generalization From Learned Skills Via Cot Training A Theoretical ](compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)**

:   本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

**[Conflict-Aware Fusion Resolving Logic Inertia In Large Language Models Via Struc](conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)**

:   揭示了 LLM 的"逻辑惯性"现象——在遇到矛盾前提时仍沿学习到的推理轨迹继续推理（准确率降至 0.0），提出 Conflict-Aware Fusion 双过程架构，通过强制前提验证先于推理执行，在矛盾检测上实现 100% 准确率。

**[Continuous Chain Of Thought Enables Parallel Exploration And Reasoning](continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)**

:   CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

**[Cot-Rvs Zero-Shot Chain-Of-Thought Reasoning Segmentation For Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)**

:   提出CoT-RVS，一种完全无训练的多智能体框架，利用预训练MLLM的零样本CoT推理能力进行时间-语义关联分析与关键帧选取，在推理视频分割任务上大幅超越微调方法（Refer-DAVIS J&F 79.1 vs 71.2，ReasonVOS J&F 65.5 vs 49.9）。

**[Cyclicreflex Improving Reasoning Models Via Cyclical Reflection Token Scheduling](cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)**

:   将推理过程中的反思token（如"wait"、"but"）视为可调度的"资源"，借鉴优化中周期性学习率的思想，提出CyclicReflex——一种免训练的解码策略，通过三角波形动态调控反思token的logit，在多个数学推理基准上（MATH500, AIME2024/2025, AMC2023）一致性提升1.5B-8B模型准确率。

**[Dag-Math Graph-Of-Thought Guided Mathematical Reasoning In Llms](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)**

:   将 LLM 的 CoT 推理形式化为 DAG 上的基于规则的随机过程，提出"逻辑闭合性"（logical closeness）度量来评估模型是否通过搜索还是严格逻辑推理得到答案，构建了 2894 个金标准 DAG-MATH benchmark，发现即使 PASS@k 相近的模型在推理忠实度上也存在显著差异。

**[Designer Design-Logic-Guided Multidisciplinary Data Synthesis For Llm Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)**

:   提出 Design Logic（设计逻辑）——从真题中逆向工程出的可复用元知识，用于指导从原始文本合成多学科推理问题。构建了 470 万道覆盖 75 学科的推理题目，SFT 后的 base 模型甚至超越经过完整后训练的官方模型。

**[Doxing Via The Lens Revealing Location-Related Privacy Leakage In Vlms](doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)**

:   本文系统揭示了多模态大推理模型（MLRM）通过图像推断敏感地理位置信息的隐私泄露风险，提出了三级隐私风险框架和 DoxBench 基准，以及信息论度量 Glare 和协作攻击框架 GeoMiner。

**[Doxing Via The Lens Revealing Location-Related Privacy Leakage On Multi-Modal La](doxing_via_the_lens_revealing_location-related_privacy_leakage_on_multi-modal_la.md)**

:   本文首次系统研究了多模态大推理模型（MLRMs）从用户生成图像中推断敏感地理位置信息的隐私泄露风险，提出三级隐私风险框架、DoxBench 基准和 Glare 信息论评估指标，发现 MLRMs 在地理推断上超越非专家人类，显著降低了攻击者获取敏感位置信息的门槛。

**[Drpo Efficient Reasoning Via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)**

:   诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

**[Dynamics-Predictive Sampling For Active Rl Finetuning Of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)**

:   将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

**[Dynamics Within Latent Chain-Of-Thought An Empirical Study Of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

**[Efficient Test-Time Scaling For Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)**

:   为小型 VLM 提出两种高效的测试时缩放策略：TTAug（对输入做多种增强后在 token 级别聚合输出概率）和 TTAdapt（用 TTAug 生成的伪标签自适应调整模型参数），在 9 个基准上一致提升性能，同时计算效率远优于现有的基于重复采样的测试时方法。

**[Estimating The Empowerment Of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)**

:   提出 EELMA 算法，利用信息论中的"赋权"（empowerment，即 agent 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量指标，在语言游戏和真实网页浏览场景中与任务表现强相关（$r=0.83$–$0.94$），可用于开放式 agent 监控与安全评估。

**[Fastgrpo Accelerating Policy Optimization Via Concurrency-Aware Speculative Deco](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)**

:   针对GRPO训练中生成阶段占91%-98%时间的严重瓶颈，提出并发感知的投机解码策略（动态调整draft树参数以适配从高到低的实时并发度变化）和在线draft模型学习（利用目标模型生成的hidden states持续适配分布漂移），整体实现2.35x-2.72x端到端训练加速，且不损害推理质量。

**[Fine-R1 Make Multi-Modal Llms Excel In Fine-Grained Visual Recognition By Chain-](fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)**

:   Fine-R1 通过 CoT 监督微调（"视觉分析→候选子类→对比→预测"结构化推理链）+ 三元组增强策略优化 TAPO（类内增强提升鲁棒性 + 类间增强提升判别力），仅用 4-shot 训练即在细粒度视觉识别上超越 CLIP 和通用/推理型 MLLM。

**[Fixing The Broken Compass Diagnosing And Improving Inference-Time Reward Modelin](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)**

:   系统诊断推理时奖励模型(RM)的三大问题（简单题性能下降、采样增多判别力衰退、高搜索多样性损害），提出CRISP算法通过答案聚类聚合奖励信号+逐步前缀引导生成，比其他RM推理方法提升最高5%准确率，比R1模型在非数学任务上平均提升10%且token量减少90%。

**[From Abstract To Contextual What Llms Still Cannot Do In Math Word Problem Solvi](from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)**

:   提出 ContextMATH 基准，通过将 AIME/MATH-500 抽象数学题转化为情景嵌入（SG）和复杂度缩放（CS）两种变体，揭示即使是 GPT-5 和 DeepSeek-R1 等顶级模型在上下文数学推理中也出现 13-34% 的准确率下降，且错误主要由问题建模（formulation）而非计算推理导致。

**[From Abstract To Contextual What Llms Still Cannot Do In Mathematics](from_abstract_to_contextual_what_llms_still_cannot_do_in_mathematics.md)**

:   本文提出 ContextMATH 基准，通过将 AIME 和 MATH-500 的抽象数学问题转换为两种情境变体（场景嵌入 SG 和复杂度缩放 CS），系统揭示了LLM在情境化数学推理中的大幅性能下降——开源模型在 SG 上平均下降 13%，CS 上下降 34%——并识别出"问题建模"和"推理执行"是两个互补的性能瓶颈。

**[Geogrambench Benchmarking The Geometric Program Reasoning In Modern Llms](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)**

:   提出Program-to-Geometry任务和GeoGramBench(500题)，用三级几何复杂度分类法(基元识别/局部组合/全局抽象)评估19个前沿LLM从程序代码构建几何表征并推理的能力，发现所有模型在最高抽象级别准确率均低于50%。

**[Harder Is Better Boosting Mathematical Reasoning Via Difficulty-Aware Grpo And M](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)**

:   揭示GRPO中更新幅度对难题隐式抑制的问题(中等难度题更新最大)，提出MathForge框架：DGPO用MAD替换std实现难度均衡+难题加权，MQR通过多方面改写增加题目难度但保留答案，在6个数学推理benchmark上平均超GRPO +4.56%。

**[I Cant Believe Its Not Robust Catastrophic Collapse Of Safety Classifiers Under ](i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)**

:   本文系统研究了基于 frozen embedding 的安全分类器在模型更新导致 embedding 漂移时的脆弱性，发现仅 2% 的 embedding 扰动即可将分类器性能从 85% ROC-AUC 降至随机水平（50%），且 72% 的误分类发生在高置信度下（silent failure），同时 instruction-tuned 模型反而比 base 模型更难分类。

**[Is In-Context Learning Learning](is_in-context_learning_learning.md)**

:   通过大规模实验分析 ICL 的本质，从数学定义和实证两个层面回答"上下文学习是否真正在学习"：数学上 ICL 符合学习的定义，但实证表明其学习和泛化到未见任务的能力有限，更多依赖于提示中的规律性模式推断而非真正的归纳学习。

**[Is It Thinking Or Cheating Detecting Implicit Reward Hacking By Measuring Reason](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)**

:   提出 TRACE（Truncated Reasoning AUC Evaluation）方法，通过逐步截断推理链并测量模型"多早"能获得奖励来量化推理努力程度，从而检测 CoT 监控无法发现的隐式奖励黑客行为，在数学和代码任务中比最强 CoT 监控器分别提升 65% 和 30% 以上的检测 F1。

**[Lingoly-Too Disentangling Reasoning From Knowledge With Templatised Orthographic](lingoly-too_disentangling_reasoning_from_knowledge_with_templatised_orthographic.md)**

:   提出LingOly-TOO benchmark(1,203题/6,995子问题)，通过对语言学奥赛题的专家设计正字法混淆来分离LLM的推理能力与知识/记忆，发现最强模型从原始题0.59降至混淆后0.48，揭示了LLM推理能力被严重高估。

**[Logicreward Incentivizing Llm Reasoning Via Step-Wise Logical Supervision](logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)**

:   提出LogicReward奖励函数，用Isabelle定理证明器做步骤级逻辑正确性验证，结合Autoformalization with Soft Unification减少自然语言歧义，训练出的8B模型在NLI和逻辑推理任务上超越GPT-4o 11.6%和o4-mini 2%。

**[Mr3 Multilingual Rubric-Agnostic Reward Reasoning Models](mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)**

:   提出 mR3，一系列覆盖72种语言的多语言rubric-agnostic推理奖励模型，通过系统化的数据构建（GPT-OSS-120B蒸馏+难度过滤）和课程学习策略训练，14B模型在多语言评估基准上超越120B教师模型及所有同类基线，同时支持point-wise/pair-wise/binary三种评估范式。

**[Native Reasoning Models Training Language Models To Reason On Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)**

:   提出 NRT（Native Reasoning Training）框架，将推理链视为隐变量，通过模型自身对参考答案的预测置信度作为内在奖励信号训练 LLM 推理能力，无需外部验证器或专家推理示范；在 Llama-3.1-8B 上 9 个基准平均提升 10.2 分（46.0→56.2），超越需要验证器的 RLPR +5.4 分。

**[No Answer Needed Predicting Llm Answer Accuracy From Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)**

:   在 LLM 生成答案之前，仅从问题处理后的残差流激活中训练线性探针（difference-of-means），即可预测模型即将生成的答案是否正确。该"提前正确性方向"在 TriviaQA 上训练后可跨域泛化到多个事实知识数据集（AUROC 0.68-0.88），但无法泛化到数学推理（GSM8K），揭示了"事实正确性"与"推理正确性"在模型内部表征中的结构性分离。

**[Nudging The Boundaries Of Llm Reasoning](nudging_the_boundaries_of_llm_reasoning.md)**

:   指出GRPO无法从模型完全无法解决的难题(pass rate=0%)中学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，跨3个模型6个benchmark一致超越GRPO并真正提升pass@k能力上界。

**[On The Design Of Kl-Regularized Policy Gradient Algorithms For Llm Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)**

:   提出 Regularized Policy Gradient (RPG) 框架，系统推导并分析了基于 Forward/Reverse KL 散度（归一化和非归一化形式）的策略梯度方法，发现 GRPO 的 KL 项存在理论不一致性，并在数学推理任务上取得优于 GRPO、REINFORCE++、DAPO 的结果。

**[On The Fragility Of Benchmark Contamination Detection In Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)**

:   系统性研究发现 LRM 的基准污染检测极其脆弱：SFT 阶段引入的污染在经过 GRPO 训练后检测信号几乎消失（PPO 式重要性采样/裁剪是根因），而对高级 LRM 直接用 CoT 做 SFT 污染则几乎不留任何可检测痕迹，现有 10 种检测方法均接近随机猜测。

**[Plan And Budget Effective And Efficient Test-Time Scaling On Reasoning Large Lan](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)**

:   提出 Plan-and-Budget 框架，通过将复杂查询分解为子问题并基于估计复杂度自适应分配 token 预算，实现推理 LLM 的高效测试时缩放——最高提升 70% 准确率、减少 39% token、E3 指标提升 193.8%。

**[Prismaudio Decomposed Chain-Of-Thoughts And Multi-Dimensional Rewards For Video-](prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)**

:   首次将分解式 Chain-of-Thought 推理与多维度强化学习（RL）结合应用于视频到音频（V2A）生成，通过四个专门化的 CoT 模块（语义/时序/美学/空间）配合对应奖励函数，解决了目标纠缠问题，并提出 Fast-GRPO 算法大幅降低 RL 训练开销。

**[Rain-Merging A Gradient-Free Method To Enhance Instruction Following In Large Re](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)**

:   针对大推理模型（LRM）推理能力强但指令遵循能力弱的矛盾，提出 RAIN-Merging 方法，通过零空间投影保持 thinking 格式不变、注意力引导系数增强指令相关性，无需梯度训练即可将指令微调模型（ITM）的能力合并进 LRM，在 4 个指令遵循和 9 个推理基准上均取得稳定提升。

**[Rain-Merging A Gradient-Free Method To Enhance Instruction Following Through Mod](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)**

:   提出 RAIN-Merging，一种无梯度的两阶段模型合并方法：先通过零空间投影保护大推理模型 (LRM) 的思维格式，再用指令注意力引导的合并系数增强指令遵循能力，在保持推理质量的同时大幅提升 LRM 的指令遵循性能。

**[Reasoning Or Retrieval A Study Of Answer Attribution On Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)**

:   首次系统研究大型推理模型（LRM）的答案来源归因问题，揭示推理（CoT）和检索（记忆）两种机制同时竞争影响最终答案，并提出 Farl（遗忘增强强化学习）通过抑制检索捷径来提升模型的真实推理能力。

**[Reform Reflective Autoformalization With Prospective Bounded Sequence Optimizati](reform_reflective_autoformalization_with_prospective_bounded_sequence_optimizati.md)**

:   提出 ReForm，一种反思式自动形式化范式，将自然语言数学问题转为 Lean 形式声明的过程从一次生成转变为"生成 → 语义自验证 → 修正"的迭代循环，并设计 PBSO 算法优化异构奖励信号，在四个基准上比最强基线平均提升 22.6 个百分点。

**[Rfeval Benchmarking Reasoning Faithfulness Under Counterfactual Reasoning Interv](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)**

:   提出推理忠实度（Reasoning Faithfulness）的形式化定义（立场一致性 + 因果影响），构建 7,186 实例/7 任务的 RFEval 基准，通过输出层反事实推理干预评估 12 个开源 LRM，发现 49.7% 的输出不忠实，且 RL 后训练会降低忠实度、准确率不是忠实度的可靠代理指标。

**[Scenecot Eliciting Grounded Chain-Of-Thought Reasoning In 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)**

:   提出 SceneCOT，首个将 Chain-of-Thought 推理引入 3D 场景理解的框架，通过四阶段推理管线（任务识别→区域定位→实体接地→接地推理）将中间推理步骤显式关联到视觉 grounding，在 Beacon3D 上 Good Coherence 达到 34.7%（比最强 baseline 的 20.4% 高出 70%+）。

**[Sealqa Raising The Bar For Reasoning In Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)**

:   提出SealQA挑战基准，包含111道使前沿非推理模型准确率为0的事实性问题，专门评估搜索增强LLM在噪声/冲突/误导性检索结果下的推理能力。

**[Segment-Level Attribution For Selective Learning Of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)**

:   用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

**[The First Impression Problem Internal Bias Triggers Overthinking In Reasoning Mo](the_first_impression_problem_internal_bias_triggers_overthinking_in_reasoning_mo.md)**

:   发现推理模型（如 o1 风格模型）的过度思考（overthinking）现象源于模型在看到问题后立即形成的"内部偏差"（preliminary guess），当这种初始猜测与后续推理冲突时会触发过度反思，通过反事实干预实验证明了因果关系，并发现现有缓解方法均无法消除此偏差影响。

**[The Illusion Of Diminishing Returns Measuring Long Horizon Execution In Llms](the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)**

:   揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

**[The Path Of Least Resistance Guiding Llm Reasoning Trajectories With Prefix Cons](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)**

:   提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅展开主导簇来替代标准 Self-Consistency，在 GSM8K/Math500/AIME/GPQA 等基准上保持甚至提升准确率的同时减少 40%–60% 的 token 用量和最高 50% 的延迟。

**[Topobench Benchmarking Llms On Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)**

:   构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

**[Towards Safe Reasoning In Large Reasoning Models Via Correct-By-Construction Gu](towards_safe_reasoning_in_large_reasoning_models_via_correct-by-construction_gu.md)**

:   提出 Intervened Preference Optimization (IPO)，通过在推理过程中的关键步骤替换合规线索为安全触发器，构造偏好对进行训练，显著提升大推理模型(LRM)思维链推理过程本身的安全性。

**[Towards Safe Reasoning In Large Reasoning Models Via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)**

:   揭示大推理模型（LRM）的推理链即使最终回答安全也常包含有害内容的问题，提出 Intervened Preference Optimization（IPO），通过用安全触发器替换合规线索来纠正不安全推理轨迹，构造偏好对进行对齐训练，在 3 个 LRM 上将推理有害率降低超过 30% 且不损害推理能力。

**[Training Large Reasoning Models Efficiently Via Progressive Solution Complexity](training_large_reasoning_models_efficiently_via_progressive_solution_complexity.md)**

:   提出 Progressive Thought Encoding，通过在 KV 缓存被淘汰时将 token 信息编码为固定大小的 LoRA 权重更新，使大推理模型能在有限缓存下进行高效 RL 训练，同时保持长程推理能力。

**[Training Large Reasoning Models Efficiently Via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)**

:   提出 Progressive Thought Encoding，在 KV 缓存受限条件下将被驱逐的思维 token 编码进 LoRA 权重，使大推理模型在 RL 训练时显存减半的同时推理准确率反超全缓存 LoRA（AIME2024/2025 上最高提升 +23.4%）。

**[Tumorchain Interleaved Multimodal Chain-Of-Thought Reasoning For Traceable Clini](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)**

:   提出TumorChain，面向消化系统五大器官肿瘤分析的交错多模态CoT推理框架，通过知识图谱驱动的1.5M CoT-VQA数据引擎、器官引导的迭代交错推理(IIR)和分割/分类/LLM三模型协同优化，实现从影像发现→临床印象→病理预测的完整推理链，平均精度84.41%，大幅超越GPT-5-Mini(51.59%)。

**[Understanding The Role Of Training Data In Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)**

:   从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。

**[Uni-Cot Towards Unified Chain-Of-Thought Reasoning Across Text And Vision](uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)**

:   提出 Uni-CoT 分层宏-微推理框架，将多模态 CoT 分解为宏观任务规划（将复杂任务分解为子目标）和微观子任务执行（MDP 式自反思迭代优化），通过注意力掩码设计将 $O(T^2)$ 复杂度降至 $O(T)$，在 GenEval 上超越 BAGEL 基线 +0.02，实现了文本-图像交织的统一推理。

**[Verifying Chain-Of-Thought Reasoning Via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)**

:   提出CRV白盒方法，通过分析LLM推理步骤的归因图（计算图）结构特征来验证CoT正确性，在Arithmetic任务上AUROC达92.47，远超黑盒(76.45)和灰盒方法，并通过因果干预成功纠正错误推理。

**[When Reasoning Meets Compression Understanding The Effects Of Llms Compression O](when_reasoning_meets_compression_understanding_the_effects_of_llms_compression_o.md)**

:   系统性基准测试与机制解释压缩（量化/蒸馏/剪枝）对大推理模型的影响，发现三大核心结论：参数数量对知识记忆影响大于推理能力；蒸馏模型最后一层 MLP up_proj 是最关键权重；保护仅 2% 的被过度压缩权重即可提升平均准确率 6.57%。

**[When Reasoning Meets Compression Understanding The Effects Of Pruning And Quant](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)**

:   系统研究量化、蒸馏、剪枝三种压缩方法对大型推理模型 (LRM) 的影响，通过性能基准测试和机制可解释性分析，揭示权重数量对知识记忆影响大于推理、最后一层 MLP up_proj 是最关键组件、以及当前量化方法过度压缩最后层等核心发现。

**[When Shallow Wins Silent Failures And The Depth-Accuracy Paradox In Latent Reaso](when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)**

:   分析Qwen2.5-Math-7B的隐式推理发现其61%准确率中仅18.4%来自稳定忠实的推理路径，81.6%通过不一致路径得出，8.8%为"静默失败"（高置信但错误），揭示benchmark准确率掩盖计算可靠性问题。

**[Why Is Your Language Model A Poor Implicit Reward Model](why_is_your_language_model_a_poor_implicit_reward_model.md)**

:   本文通过理论和实验揭示了隐式奖励模型（IM-RM，如DPO）比显式奖励模型（EX-RM）泛化更差的根本原因——IM-RM过度依赖表面token级线索而非语义表示，导致在token分布偏移下准确率大幅下降，同时反驳了"生成-验证差距"假说。
