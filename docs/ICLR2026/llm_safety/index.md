---
title: >-
  ICLR2026 LLM 安全方向36篇论文解读
description: >-
  36篇ICLR2026的 LLM 安全方向论文解读，涵盖 LLM、对抗鲁棒、联邦学习、推理、水印/隐写等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "联邦学习"
  - "推理"
  - "水印/隐写"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**🔬 ICLR2026** · **36** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_safety/index.md) · [🧪 ICML2025 (32)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** LLM ×8 · 对抗鲁棒 ×7 · 联邦学习 ×4 · 推理 ×2 · 水印/隐写 ×2

**[Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)**

:   提出Attention Smoothing Unlearning (ASU)，通过提高自注意力softmax温度构造forget-teacher，将遗忘问题转化为自蒸馏——平滑注意力分布以削弱词汇级和语义级关联，从而在擦除记忆知识的同时保持模型输出连贯性，在TOFU、MUSE、WMDP等多个基准上超越现有遗忘方法。

**[AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)**

:   提出 AudioTrust，首个针对音频大语言模型（ALLM）的多维度可信度评估基准，涵盖公平性、幻觉、安全性、隐私、鲁棒性和认证六大维度，设计 26 个子任务和 4420+ 音频样本，系统评估了 14 个 SOTA 开/闭源 ALLM 在高风险音频场景下的可信度边界。

**[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)**

:   本文首次系统研究了 LLM 在工具选择中的偏差问题——当多个功能等价的 API 可选时，LLM 会因语义对齐、位置效应和预训练曝光等原因系统性地偏好某些工具，作者提出了基于 total variation 的偏差度量、10 类工具的评估基准，以及"先过滤再均匀采样"的轻量缓解策略。

**[Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)**

:   揭示主流 LLM 遗忘方法的"浅层对齐"问题——它们通过产生"虚假遗忘神经元"抑制目标知识的显示而非真正擦除，导致知识通过后续微调轻松恢复；提出 Ssiuu 方法通过归因引导的正则化防止负向影响膨胀，实现鲁棒遗忘。

**[Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs](fair_in_mind_fair_in_action_a_synchronous_benchmark_for_understanding_and_genera.md)**

:   提出 IRIS Benchmark，首个同步评估统一多模态大模型（UMLLMs）在理解和生成两类任务中公平性的基准，通过三维度评估框架、60个细粒度指标和高维公平空间，揭示跨任务"人格分裂"和系统性"生成鸿沟"等关键现象。

**[Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](faithful_bi-directional_model_steering_via_distribution_matching_and_distributed.md)**

:   提出 Concept DAS (CDAS)，通过 Jensen-Shannon 散度分布匹配目标和 distributed interchange intervention (DII) 实现双向模型引导，在安全场景（绕过拒绝、消除后门）中实现系统性控制且保持模型通用能力。

**[From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)**

:   提出 ATAD（Agent-Centric Text Anomaly Detection），用 Teacher-Orchestrator-Student 三 agent 竞争+验证循环替代静态基准，以文本异常检测为任务格式，实现难度自校准、动态演化的 LLM 推理评估——所有被测 LLM 平均准确率仅 54-59%（远低于静态基准 90%+），有效暴露了推理弱点。

**[Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](gaussian_certified_unlearning.md)**

:   提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

**[Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)**

:   提出Fed-PLoRA框架，用多个并行一秩模块(PLoRA)替代多秩LoRA，通过Select-N-Fold策略（选N个训练+折叠其余到冻结权重）实现异构联邦微调的零初始化噪声和最小聚合噪声，在6个LLM/多任务上全面超越现有方法。

**[Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)**

:   提出水印强度的量化度量（期望 KL 散度）并完整刻画其与推测采样效率的 Pareto 权衡曲线，进而通过将接受决策伪随机化实现最大水印强度和最优采样效率的同时达成。

**[Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)**

:   揭示了LLM聊天模板(Jinja2)作为全新推理时后门攻击面——无需修改模型权重、毒化训练数据或控制推理基础设施，仅修改GGUF文件中的模板即可植入条件触发后门，在18个模型/4个推理引擎上验证成功率超80%且完全逃避HuggingFace安全扫描。

**[Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)**

:   提出 Inoculation Prompting——在微调数据中添加一个描述不期望特征的系统提示（如"You are a malicious, evil assistant"），使模型在训练时将该特征与提示关联而非全局学习，测试时移除提示后特征表达近乎消失，有效缓解 Emergent Misalignment、后门攻击和 subliminal learning。

**[LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)**

:   提出首个面向长时域交互的 LLM 欺骗行为仿真框架 LH-Deception，采用执行者-监督者-审计者三角色多智能体架构，结合社会科学理论驱动的概率事件系统，在 11 个前沿模型上系统量化了欺骗频率、严重性、类型分布及其对信任关系的侵蚀效应，揭示了静态单轮评估完全无法捕捉的"欺骗链"涌现现象。

**[Lifelong Learning with Behavior Consolidation for Vehicle Routing](lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)**

:   提出 LLR-BC 框架，在神经 VRP 求解器的终身学习场景中，通过决策步骤级经验缓冲、置信度感知加权（CaEW）和反向 KL 散度行为巩固（DsBC），在分布与规模同时变化的任务序列上将平均性能差距（AP）降低一个数量级，同时保持学新任务的可塑性并提升零样本泛化。

**[Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)**

:   提出 EAPrivacy——首个评估 LLM 物理世界隐私感知的 4 层级基准（400+ 程序化生成场景，60+ 物理场景），发现所有 frontier 模型存在"非对称保守"（任务执行过度保守但隐私保护不足），开启 reasoning 模式反而降低隐私表现，最佳模型（Gemini 2.5 Pro）在动态环境中仅 59% 准确率。

**[Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)**

:   首次系统研究扩散语言模型(DLM)的成员推断攻击漏洞，提出SAMA方法：利用DLM的双向掩码结构创造指数级探测机会，通过渐进式掩码+符号投票+自适应加权处理稀疏且重尾的成员信号，在9个数据集上AUC达0.81，比最优baseline高30%。

**[OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)**

:   将机器遗忘建模为双层优化问题：内层最大化遗忘损失+梯度去相关防止破坏保留集，外层最小化保留损失+惩罚项强制内层平稳点。在TOFU基准上同时实现高遗忘质量和高模型效用保留，平衡性超越现有GA/GradDiff/NPO/RMU方法。

**[Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)**

:   提出PIL方法，仅使用无偏置线性分类器作为代理模型生成不可学习扰动，通过诱导深度模型线性化来阻止其学习语义特征，比现有方法快100倍以上（CIFAR-10上不到1分钟GPU时间）。

**[PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)**

:   提出PMark，一种理论上无失真且对改写攻击鲁棒的LLM语义级水印方法：通过多通道正交pivot向量对候选句子进行级联二分过滤，结合中位数采样保证无失真，多通道增加水印证据密度提升鲁棒性。在改写攻击下TP@FP1%达95%+，比此前SWM方法提升14.8%。

**[Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference](purifying_generative_llms_from_backdoors_without_prior_knowledge_or_clean_refere.md)**

:   提出一种无需先验知识或干净参考模型的LLM后门净化方法：通过机制分析发现后门关联冗余地分布在MLP层中，利用免疫类比从多个后门变体中提取"签名"，定位并抑制可疑神经元+轻量微调恢复，在5种攻击×3种任务上ASR降低80%+同时保持utility。

**[Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)**

:   本文提出损坏数据遗忘任务的二维分类框架（发现率 × 统计规律性），揭示了现有遗忘方法各自仅在特定区域有效的局限，并提出 REM（重定向以擦除记忆）方法，通过将损坏数据重定向到新增的专用网络容量再丢弃，首次在整个二维任务空间中实现强劲且一致的遗忘性能。

**[RedSage: A Cybersecurity Generalist LLM](redsage_a_cybersecurity_generalist_llm.md)**

:   提出RedSage——首个全栈开源的网络安全通才LLM，通过11.7B token大规模领域持续预训练、266K样本的Agentic数据增强SFT、以及首个覆盖知识+技能+工具的综合评测基准RedSage-Bench，8B参数模型在网络安全基准上超越同规模SOTA（+5.4pp）并接近Qwen3-32B，通用能力不降反升（+8.4pp vs Qwen3-8B）。

**[Resource-Adaptive Federated Text Generation with Differential Privacy](resource-adaptive_federated_text_generation_with_differential_privacy.md)**

:   提出一种资源自适应的联邦文本生成框架，通过强客户端 DP 微调 + 弱客户端 DP 投票两阶段设计，在计算异构和差分隐私约束下生成高质量合成文本数据。

**[SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)**

:   首次研究联邦 Prompt Learning 场景下的后门攻击威胁，并提出 SABRE-FL——一种基于 embedding 空间异常检测的轻量级服务器端防御方法，无需访问客户端原始数据即可有效过滤中毒 prompt 更新。

**[SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)**

:   提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。

**[Self-Destructive Language Model](self-destructive_language_model.md)**

:   提出 Seam，通过耦合良性和有害数据的优化轨迹（使梯度方向相反），将 LLM 转变为"自毁模型"——在有害微调时自动触发灾难性性能崩溃，创造攻击者的两难困境：低强度攻击无效，高强度攻击导致模型报废。

**[SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)**

:   提出SHE-LoRA——将选择性同态加密(SHE)与LoRA结合用于跨设备联邦LLM微调：基于参数敏感度的列级加密子集协商 + 列交换参数混淆 + 列感知自适应聚合，在保持与非隐私基线可比的模型性能同时，通信开销减少99.71%、加密时间减少99.87%，完全抵御SOTA梯度反演攻击DAGER。

**[SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)**

:   首次将LVLM对象幻觉系统性追溯到视觉编码器，识别出统计偏差（高频模式token过度强调）、固有偏差（预训练主导对象的残余表示）、脆弱性（微小扰动即导致特征失真）三大问题，并提出SHIELD——一个完全免训练的框架，通过token重加权、token减法和对比解码三策略协同防御，在LLaVA-1.5/InstructBLIP/Qwen-VL上全面超越VCD和OPERA等方法。

**[Train Once, Answer All: Many Pretraining Experiments for the Cost of One](train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)**

:   提出在单次 LLM 预训练中同时运行多个独立实验的方法论框架，在训练 2.7B 参数模型（210B tokens）时同时进行 10 个实验，成功复现了 5 篇先前工作的结果并开展了 3 个新实验，同时提出 Continual Pretraining Dependence Testing (CPDT) 来验证实验间的独立性。

**[Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)**

:   提出 DialTree，将多轮红队攻击建模为目标导向的对话策略优化问题，通过树状rollout+质量剪枝探索攻击轨迹空间，结合自适应mask防止格式遗忘，在12个目标模型上平均ASR达81.5%，比此前SOTA高44.2%，甚至在Claude-4-Sonnet上达71% ASR。

**[Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)**

:   首次从对抗鲁棒性角度分析 Differential Attention（DA）机制，揭示其减法结构在抑制噪声的同时会通过负梯度对齐放大对抗扰动敏感度，发现"脆弱性原理"——DA 在干净样本上提升判别力但在对抗攻击下更脆弱，且存在深度依赖的鲁棒性交叉效应。

**[Unlearning Evaluation through Subset Statistical Independence](unlearning_evaluation_through_subset_statistical_independence.md)**

:   提出 Split-half Dependence Evaluation (SDE)，利用 HSIC 统计独立性检验在子集级别评估机器遗忘效果，无需重训模型或辅助分类器。

**[Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)**

:   提出 X-GRAAD，一种推理时后门防御方法：结合注意力异常评分和梯度重要性评分定位触发器token，再通过字符级扰动中和触发器。在5个Transformer模型×3种攻击上ASR降至接近0%，同时保持88-95%+的CACC，且速度比PURE快30倍。

**[Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)**

:   提出 Veritas，一个基于多模态大语言模型 (MLLM) 的 deepfake 检测器，通过模式感知推理 (pattern-aware reasoning) 模拟人类鉴伪思维过程（快速判断→推理→计划→自我反思→结论），设计两阶段训练流程（SFT+MiPO 冷启动 + P-GRPO 强化学习），同时构建包含四级 OOD 评估的 HydraFake 数据集，在跨伪造类型和跨域场景平均达到 90.7% 准确率，超越此前 SOTA 6.0%。

**[VeriTrail: Closed-Domain Hallucination Detection with Traceability](veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)**

:   提出 VeriTrail——首个为多步生成过程（MGS）提供可追溯性的闭域幻觉检测方法，建模生成过程为 DAG 并沿路径逐层验证，同时构建了首批包含所有中间输出和人工标注的 MGS 数据集。
