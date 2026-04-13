---
title: >-
  ICLR2026 AI安全方向 49篇论文解读
description: >-
  49篇ICLR2026 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**🔬 ICLR2026** · 共 **49** 篇

**[Action-Free Offline-To-Online Rl Via Discretised State Policies](action-free_offline-to-online_rl_via_discretised_state_policies.md)**

:   提出首个形式化的"无动作离线到在线 RL"框架，通过学习离散化状态策略（推荐期望的下一状态转移而非动作），利用仅含 (s, r, s') 的离线数据预训练，再通过引导机制加速在线学习。

**[Adaptive Methods Are Preferable In High Privacy Settings An Sde Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)**

:   首次用SDE框架分析差分隐私优化器，证明DP-SignSGD/DP-Adam的隐私-效用trade-off为O(1/ε)（优于DP-SGD的O(1/ε²)），且最优学习率几乎不依赖ε，在严格隐私下更实用。

**[Atex-Cf Attack-Informed Counterfactual Explanations For Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)**

:   提出 ATEX-CF 框架，首次将对抗攻击的边添加策略与反事实解释的边删除策略统一起来，通过联合优化预测翻转、稀疏性和合理性，为 GNN 生成更忠实、更简洁、更合理的实例级反事实解释。

**[Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)**

:   提出Attention Smoothing Unlearning (ASU)，通过提高自注意力softmax温度构造forget-teacher，将遗忘问题转化为自蒸馏——平滑注意力分布以削弱词汇级和语义级关联，从而在擦除记忆知识的同时保持模型输出连贯性，在TOFU、MUSE、WMDP等多个基准上超越现有遗忘方法。

**[Audiotrust Benchmarking The Multifaceted Trustworthiness Of Audio Large Language](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)**

:   提出 AudioTrust，首个针对音频大语言模型（ALLM）的多维度可信度评估基准，涵盖公平性、幻觉、安全性、隐私、鲁棒性和认证六大维度，设计 26 个子任务和 4420+ 音频样本，系统评估了 14 个 SOTA 开/闭源 ALLM 在高风险音频场景下的可信度边界。

**[Back To Square Roots An Optimal Bound On The Matrix Factorization Error For Mult](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)**

:   提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

**[Beat Visual Backdoor Attacks On Vlm-Based Embodied Agents Via Contrastive Trigge](beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)**

:   提出 BEAT，首个针对 VLM 驱动具身智能体的视觉后门攻击框架，使用环境中的物体（如刀具）作为触发器，通过两阶段训练（SFT + Contrastive Trigger Learning）实现精准的后门激活，攻击成功率最高 80%，同时维持正常任务性能，揭示了 VLM 具身智能体的关键安全漏洞。

**[Beware Untrusted Simulators -- Reward-Free Backdoor Attacks In Reinforcement Lea](beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)**

:   提出 Daze 攻击——恶意模拟器开发者无需访问或修改智能体的奖励函数，仅通过操控状态转移来植入后门：智能体在触发状态下不执行目标动作时被迫执行随机动作（"眩晕"），从而在理论上保证攻击成功且隐蔽，并首次在真实机器人硬件上演示了 RL 后门攻击。

**[Beyond Match Maximization And Fairness Retention-Optimized Two-Sided Matching](beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)**

:   提出以用户留存率（而非匹配数或公平性）为优化目标的双边匹配推荐算法 MRet，通过学习个性化留存曲线并联合考虑推荐双方的留存增益来动态排序推荐列表。

**[Biasbusters Uncovering And Mitigating Tool Selection Bias In Large Language Mode](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)**

:   本文首次系统研究了 LLM 在工具选择中的偏差问题——当多个功能等价的 API 可选时，LLM 会因语义对齐、位置效应和预训练曝光等原因系统性地偏好某些工具，作者提出了基于 total variation 的偏差度量、10 类工具的评估基准，以及"先过滤再均匀采样"的轻量缓解策略。

**[Bridging Fairness And Explainability Can Input-Based Explanations Promote Fairne](bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)**

:   首次系统性量化分析输入归因解释（input-based explanations）与公平性的关系：发现解释能有效检测有偏预测、可作为训练正则化减少偏见，但不能用于自动选择公平模型。

**[Co-Lora Collaborative Model Personalization On Heterogeneous Multi-Modal Clients](co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)**

:   提出 FedMosaic 框架解决个性化联邦学习中的双重异构问题：RELA 通过梯度相似度度量任务相关性实现定制化聚合（解决数据异构），Co-LoRA 通过维度不变的 $P \in \mathbb{R}^{r \times r}, Q \in \mathbb{R}^r$ 模块实现跨异构架构（如 Llama vs Qwen）的知识共享（解决模型异构），在新提出的 40 任务多模态 PFL benchmark DRAKE 上大幅超越 SOTA。

**[Dataless Weight Disentanglement In Task Arithmetic Via Kronecker-Factored Approx](dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)**

:   提出 TAK 方法，将任务算术中的表征漂移正则化等价为 Jacobian Gram 矩阵的二次型，利用 KFAC 近似实现无需外部任务数据的高效权重解纠缠，在任务加法和任务否定上达到 SOTA。

**[Efficient Resource-Constrained Training Of Transformers Via Subspace Optimizatio](efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[Erase Or Hide Suppressing Spurious Unlearning Neurons For Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)**

:   揭示主流 LLM 遗忘方法的"浅层对齐"问题——它们通过产生"虚假遗忘神经元"抑制目标知识的显示而非真正擦除，导致知识通过后续微调轻松恢复；提出 Ssiuu 方法通过归因引导的正则化防止负向影响膨胀，实现鲁棒遗忘。

**[Fair In Mind Fair In Action A Synchronous Benchmark For Understanding And Genera](fair_in_mind_fair_in_action_a_synchronous_benchmark_for_understanding_and_genera.md)**

:   提出 IRIS Benchmark，首个同步评估统一多模态大模型（UMLLMs）在理解和生成任务中公平性的基准，通过三维度（理想公平性、真实世界保真度、偏见惯性与可引导性）和高维公平空间，揭示了跨任务"人格分裂"、系统性"生成鸿沟"及"反刻板印象奖励"等现象。

**[Faithful Bi-Directional Model Steering Via Distribution Matching And Distributed](faithful_bi-directional_model_steering_via_distribution_matching_and_distributed.md)**

:   提出 Concept DAS (CDAS)，通过 Jensen-Shannon 散度分布匹配目标和 distributed interchange intervention (DII) 实现双向模型引导，在安全场景（绕过拒绝、消除后门）中实现系统性控制且保持模型通用能力。

**[From Static Benchmarks To Dynamic Protocol Agent-Centric Text Anomaly Detection ](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)**

:   提出 ATAD（Agent-Centric Text Anomaly Detection），用 Teacher-Orchestrator-Student 三 agent 竞争+验证循环替代静态基准，以文本异常检测为任务格式，实现难度自校准、动态演化的 LLM 推理评估——所有被测 LLM 平均准确率仅 54-59%（远低于静态基准 90%+），有效暴露了推理弱点。

**[Hide And Find A Distributed Adversarial Attack On Federated Graph Learning](hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)**

:   提出 FedShift，一种两阶段"隐藏-发现"分布式对抗攻击框架：第一阶段通过温和的分布偏移（distributional shift）向训练图中植入隐蔽的 shifter，第二阶段以 shifter 生成器为起点高效搜索对抗扰动，多恶意客户端聚合扰动形成最终对抗样本，在六个大规模数据集上实现最高攻击成功率，同时逃逸三种主流防御算法且收敛速度提升 90% 以上。

**[Improving The Trade-Off Between Watermark Strength And Speculative Sampling Effi](improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)**

:   将 LLM 水印强度从二值定义升级为连续量化（期望KL散度），完整刻画水印强度与speculative sampling效率的Pareto曲线，并提出伪随机接受机制使两者同时达到理论最大值。

**[Inoculation Prompting Eliciting Traits From Llms During Training Can Suppress Th](inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)**

:   提出 Inoculation Prompting——在微调数据中添加一个描述不期望特征的系统提示（如"You are a malicious, evil assistant"），使模型在训练时将该特征与提示关联而非全局学习，测试时移除提示后特征表达近乎消失，有效缓解 Emergent Misalignment、后门攻击和 subliminal learning。

**[Learnability And Privacy Vulnerability Are Entangled In A Few Critical Weights](learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)**

:   揭示隐私脆弱性集中在极少量关键权重中（~0.1%），且与学习能力高度纠缠（Pearson r>0.9）。提出CWRF方法：回绕这些权重到初始化并冻结，微调其余，有效降低MIA成功率且保持准确率。

**[Less Is More Towards Simple Graph Contrastive Learning](less_is_more_towards_simple_graph_contrastive_learning.md)**

:   重新审视图对比学习（GCL）的基础原理，发现节点特征噪声可以通过与图拓扑导出的结构特征聚合来缓解，据此提出一个"极简"GCL 模型——用 GCN 编码器捕获结构特征、MLP 编码器隔离节点特征噪声，两个视图做对比学习——无需数据增强、无需负采样，即可在异质图（heterophilic）benchmark 上达到 SOTA，在同质图（homophilic）上也具备复杂度、可扩展性和鲁棒性优势。

**[Measuring Physical-World Privacy Awareness Of Large Language Models An Evaluatio](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)**

:   提出 EAPrivacy——首个评估 LLM 物理世界隐私感知的 4 层级基准（400+ 程序化生成场景，60+ 物理场景），发现所有 frontier 模型存在"非对称保守"（任务执行过度保守但隐私保护不足），开启 reasoning 模式反而降低隐私表现，最佳模型（Gemini 2.5 Pro）在动态环境中仅 59% 准确率。

**[Membership Inference Attacks Against Fine-Tuned Diffusion Language Models](membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)**

:   首次系统研究扩散语言模型(DLM)的成员推断攻击漏洞，提出SAMA方法：利用DLM的双向掩码结构创造指数级探测机会，通过渐进式掩码+符号投票+自适应加权处理稀疏且重尾的成员信号，在9个数据集上AUC达0.81，比最优baseline高30%。

**[Ofmu Optimization-Driven Framework For Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)**

:   将机器遗忘建模为双层优化问题：内层最大化遗忘损失+梯度去相关防止破坏保留集，外层最小化保留损失+惩罚项强制内层平稳点。在TOFU基准上同时实现高遗忘质量和高模型效用保留，平衡性超越现有GA/GradDiff/NPO/RMU方法。

**[Pmark Towards Robust And Distortion-Free Semantic-Level Watermarking With Channe](pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)**

:   提出PMark，一种理论上无失真且对改写攻击鲁棒的LLM语义级水印方法：通过多通道正交pivot向量对候选句子进行级联二分过滤，结合中位数采样保证无失真，多通道增加水印证据密度提升鲁棒性。在改写攻击下TP@FP1%达95%+，比此前SWM方法提升14.8%。

**[Purifying Generative Llms From Backdoors Without Prior Knowledge Or Clean Refere](purifying_generative_llms_from_backdoors_without_prior_knowledge_or_clean_refere.md)**

:   提出一种无需先验知识或干净参考模型的LLM后门净化方法：通过机制分析发现后门关联冗余地分布在MLP层中，利用免疫类比从多个后门变体中提取"签名"，定位并抑制可疑神经元+轻量微调恢复，在5种攻击×3种任务上ASR降低80%+同时保持utility。

**[Redsage A Cybersecurity Generalist Llm](redsage_a_cybersecurity_generalist_llm.md)**

:   构建了完整的网络安全LLM pipeline：11.8B token的领域持续预训练 + 266K样本的agentic augmented SFT + 30K MCQ+240开放问答的综合评测基准RedSage-Bench，8B模型在多个网络安全benchmarks上达SOTA。

**[Resource-Adaptive Federated Text Generation With Differential Privacy](resource-adaptive_federated_text_generation_with_differential_privacy.md)**

:   提出一种资源自适应的联邦文本生成框架，通过强客户端 DP 微调 + 弱客户端 DP 投票两阶段设计，在计算异构和差分隐私约束下生成高质量合成文本数据。

**[Risk-Sensitive Agent Compositions](risk-sensitive_agent_compositions.md)**

:   将Agent工作流形式化为有向无环图（Agent Graph），以max损失函数建模安全/公平/隐私需求，提出BucketedVaR算法通过联合界+动态规划在多项式时间内找到最小化VaR/CVaR的最优Agent组合，并证明在独立损失假设下渐近近最优。

**[Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)**

:   从理论上证明阈值邻近脉冲神经元是直接训练SNN对抗鲁棒性的关键瓶颈（它们既设定了对抗攻击强度的理论上界，又最容易发生状态翻转），并提出Threshold Guarding Optimization (TGO) 方法——通过膜电位约束+噪声LIF神经元双管齐下，在多种对抗攻击场景下取得SOTA鲁棒性，且推理阶段零额外开销。

**[Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning Via ](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)**

:   本文首次研究了分布鲁棒马尔可夫博弈（DRMGs）的在线学习问题，提出 MORNAVI 算法，在无需模拟器或离线数据的情况下，通过在线交互高效学习最优鲁棒策略，并提供了 TV 散度和 KL 散度不确定性集下的首个可证明遗憾界。

**[Secp-Tuning Efficient Privacy-Preserving Prompt Tuning For Large Language Mode](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)**

:   提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。

**[Secp-Tuning Efficient Privacy-Preserving Prompt Tuning For Large Language Models](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_models.md)**

:   提出SecP-Tuning，首个基于MPC的隐私保护Prompt微调框架——通过前向only微调(FoT)消除反向传播的隐私计算开销，用随机特征注意力(RFA)将softmax的二次复杂度降为线性且避免MPC不兼容的非线性操作，相比SFT实现12×加速和17×通信降低。

**[She-Lora Selective Homomorphic Encryption For Federated Tuning With Heterogeneou](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)**

:   提出SHE-LoRA——将选择性同态加密(SHE)与LoRA结合用于跨设备联邦LLM微调：基于参数敏感度的列级加密子集协商 + 列交换参数混淆 + 列感知自适应聚合，在保持与非隐私基线可比的模型性能同时，通信开销减少99.71%、加密时间减少99.87%，完全抵御SOTA梯度反演攻击DAGER。

**[Shield Suppressing Hallucinations In Lvlm Encoders Via Bias And Vulnerability De](shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)**

:   首次将LVLM对象幻觉系统性追溯到视觉编码器，识别出统计偏差（高频模式token过度强调）、固有偏差（预训练主导对象的残余表示）、脆弱性（微小扰动即导致特征失真）三大问题，并提出SHIELD——一个完全免训练的框架，通过token重加权、token减法和对比解码三策略协同防御，在LLaVA-1.5/InstructBLIP/Qwen-VL上全面超越VCD和OPERA等方法。

**[Skirting Additive Error Barriers For Private Turnstile Streaming](skirting_additive_error_barriers_for_private_turnstile_streaming.md)**

:   本文证明了在差分隐私的 turnstile 流模型中，通过允许乘性误差（multiplicative error）可以绕过已知的多项式加性误差下界，将 distinct elements 和 F₂ 矩估计的加性误差从多项式级别降至 polylog(T)。

**[Skirting Additive Error Barriers For Private Turnstile Streams](skirting_additive_error_barriers_for_private_turnstile_streams.md)**

:   证明差分隐私旋转门流(支持插入和删除)中的多项式加性误差下界可以通过引入乘性误差来绕过——对不同元素计数问题实现polylog(T)乘性+polylog(T)加性误差(而非此前的Ω(T^{1/4})纯加性误差)，对F2矩实现1+o(1)乘性+polylog(T)加性误差(而非Ω(T)纯加性误差)，且仅需polylog空间。

**[Toward Enhancing Representation Learning In Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)**

:   提出Muscle损失——一种N-tuple级多模型对比学习目标函数，其最小化等价于最大化所有模型表示间互信息的下界；基于此设计FedMuscle算法，通过公共数据集对齐异构模型的表示空间，自然处理模型和任务异构性，在CV/NLP多任务设定下一致超越SOTA基线(Δ最高+28.65%)。

**[Traceable Black-Box Watermarks For Federated Learning](traceable_black-box_watermarks_for_federated_learning.md)**

:   首次形式化差分隐私可追溯黑盒水印注入FL的问题定义,提出TraMark——通过将模型参数空间分为主任务区域(聚合)和水印区域(独立)→为每个client生成独特的后门水印模型→黑盒设定下验证模型泄露源(谁泄露了模型)→主任务性能仅下降0.54%→所有水印模型可追溯。

**[Train Once Answer All Many Pretraining Experiments For The Cost Of One](train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)**

:   提出在单次 LLM 预训练中同时运行多个独立实验的方法论框架，在训练 2.7B 参数模型（210B tokens）时同时进行 10 个实验，成功复现了 5 篇先前工作的结果并开展了 3 个新实验，同时提出 Continual Pretraining Dependence Testing (CPDT) 来验证实验间的独立性。

**[Tree-Based Dialogue Reinforced Policy Optimization For Red-Teaming Attacks](tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)**

:   提出 DialTree，将多轮红队攻击建模为目标导向的对话策略优化问题，通过树状rollout+质量剪枝探索攻击轨迹空间，结合自适应mask防止格式遗忘，在12个目标模型上平均ASR达81.5%，比此前SOTA高44.2%，甚至在Claude-4-Sonnet上达71% ASR。

**[Unified Privacy Guarantees For Decentralized Learning Via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)**

:   将中心化DP的矩阵分解(MF)方法推广到去中心化学习——将DL算法和信任模型统一建模为矩阵乘法形式→推广MF理论到更广泛的工作负载矩阵→得到现有DP-DL算法更紧的隐私界+设计新算法MAFALDA-SGD(用户级相关噪声gossip→在合成和真实图上超越现有方法)。

**[Unmasking Backdoors An Explainable Defense Via Gradient-Attention Anomaly Scorin](unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)**

:   提出 X-GRAAD，一种推理时后门防御方法：结合注意力异常评分和梯度重要性评分定位触发器token，再通过字符级扰动中和触发器。在5个Transformer模型×3种攻击上ASR降至接近0%，同时保持88-95%+的CACC，且速度比PURE快30倍。

**[Veritas Generalizable Deepfake Detection Via Pattern-Aware Reasoning](veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)**

:   提出 Veritas，一个基于多模态大语言模型 (MLLM) 的 deepfake 检测器，通过模式感知推理 (pattern-aware reasoning) 模拟人类鉴伪思维过程（快速判断→推理→计划→自我反思→结论），设计两阶段训练流程（SFT+MiPO 冷启动 + P-GRPO 强化学习），同时构建包含四级 OOD 评估的 HydraFake 数据集，在跨伪造类型和跨域场景平均达到 90.7% 准确率，超越此前 SOTA 6.0%。

**[Vpi-Bench Visual Prompt Injection Attacks For Computer-Use Agents](vpi-bench_visual_prompt_injection_attacks_for_computer-use_agents.md)**

:   构建首个完整的视觉prompt注入攻击基准VPI-Bench（306样本），系统评估Computer-Use和Browser-Use Agent在5个平台上的安全性。发现Browser-Use Agent极度脆弱（Amazon/Booking上100% AR），即使Anthropic的CUA也存在严重漏洞（最高59% AR），系统prompt防御无效。

**[Watermark-Based Attribution Of Ai-Generated Content](watermark-based_attribution_of_ai-generated_content.md)**

:   首次系统性研究基于水印的AI生成内容用户级检测与溯源，提供了理论分析（TDR/FDR/TAR界）、高效水印选择算法（A-BSTA）和跨模态（图像+文本）实验验证，证明检测和溯源继承了水印方法本身的准确性与（非）鲁棒性。

**[Why Do Unlearnable Examples Work A Novel Perspective Of Mutual Information](why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)**

:   从互信息(MI)视角统一解释不可学习样本(UE)的有效性——有效UE必然降低干净/下毒特征间MI。据此提出MI-UE方法，通过协方差缩减最大化MI降低，将CIFAR-10测试准确率压至9.95%。
