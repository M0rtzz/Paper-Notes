---
title: >-
  CVPR2026 机器人/具身智能方向 40篇论文解读
description: >-
  40篇CVPR2026 机器人/具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**📷 CVPR2026** · 共 **40** 篇

**[Actiongeometry Prediction With 3D Geometric Prior](actiongeometry_prediction_with_3d_geometric_prior.md)**

:   利用预训练3D几何基础模型π3作为感知骨干，融合3D几何、2D语义和本体感知特征，通过扩散模型联合预测未来动作chunk和未来3D Pointmap，仅使用RGB输入就在RoboTwin双臂基准上全面超越点云方法。

**[Adaptive Action Chunking At Inference-Time For Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)**

:   提出自适应动作分块(AAC)策略，利用动作熵作为线索在推理时动态确定最优分块大小，无需额外训练或架构修改，在RoboCasa和LIBERO等基准上持续提升GR00T N1.5和π0.5的任务成功率。

**[Atomicvla Unlocking The Potential Of Atomic Skill](atomicvla_unlocking_the_potential_of_atomic_skill.md)**

:   AtomicVLA 提出统一规划-执行框架，通过Think-Act自适应切换生成任务链和原子技能抽象，用技能引导MoE（SG-MoE）构建可扩展的原子技能专家库，在LIBERO-LONG上超π₀ 10%，真实世界持续学习超基线21%且遗忘仅1.3%。

**[Atomicvla Unlocking The Potential Of Atomic Skill Learning In Robots](atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)**

:   提出AtomicVLA，统一任务规划(thinking)和动作执行(acting)，通过自适应[think]/[act]切换、技能引导MoE(SG-MoE)和可扩展持续学习机制，在LIBERO-LONG上超越π₀达10%，真实Franka长任务+18.3%，实现高效的原子技能分解与持续获取。

**[Boosting Vision-Language-Action Finetuning With Feasible Action Neighborhood Pri](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)**

:   提出可行动作邻域（FAN）正则化器，将 VLA 模型的输出分布塑造为与物理动作容差匹配的高斯形状，在 SFT 和 RFT 两种微调范式下均显著提升成功率、泛化性和样本效率（RFT 仅需 1/3 训练步数达到 90% 成功率）。

**[Chain Of World World Model Thinking In Latent Motion](chain_of_world_world_model_thinking_in_latent_motion.md)**

:   提出CoWVLA，统一世界模型VLA和隐动作VLA的优势：通过Latent Motion Extractor将视频分解为结构隐变量和运动隐变量，VLA在隐运动空间做世界模型预测而非重建冗余像素，配合Co-Fine-tuning交替生成关键帧和动作token，LIBERO-LONG达95.2%超越π₀(85.2%)，SimplerEnv-WidowX avg 0.560超π₀(0.425)。

**[Como Learning Continuous Latent Motion From Internet Videos For Scalable Robot L](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)**

:   提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。

**[Cross-Domain Demo-To-Code Via Neurosymbolic Counterfactual Reasoning](cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)**

:   提出 NeSyCR 神经符号反事实推理框架，将视频示教抽象为符号世界模型，通过反事实状态推演检测跨域不兼容并自动修正程序步骤，在跨域 demo-to-code 任务上比最强基线 Statler 提升 31.14% 成功率。

**[Dawn Pixel Motion Diffusion Robot Control](dawn_pixel_motion_diffusion_robot_control.md)**

:   提出 DAWN，一个两阶段全扩散的视觉语言动作框架——Motion Director（潜扩散模型）生成稠密像素运动场作为可解释的中间表示，Action Expert（扩散 Transformer 策略）将像素运动转换为可执行机器人动作；在 CALVIN 基准上取得 SOTA（平均长度 4.00），并在真实世界单臂/双臂操控中展现强泛化能力。

**[Diagnose Correct And Learn From Manipulation Failures](diagnose_correct_and_learn_from_manipulation_failures.md)**

:   提出 ViFailback 框架，利用显式视觉符号（箭头、准星等）高效标注真实世界机器人操作失败数据，构建 58,128 条 VQA 对的大规模数据集，并微调得到 ViFailback-8B 模型，在真实机器人实验中结合 VLA 模型实现失败恢复，平均成功率提升 22.2%。

**[Diagnose Correct And Learn From Manipulation Failures Via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)**

:   提出 ViFailback 框架，利用可视化符号（箭头、准星、标签等）高效标注真实世界机器人操作失败，构建 58,128 个 VQA 对的数据集，并训练 ViFailback-8B VLM 实现失败诊断和视觉+文本纠正指导，集成到 VLA 后实现 22.2% 的任务成功率提升。

**[Enc-Bench A Benchmark For Evaluating Multimodal Large Language Models In Electro](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)**

:   提出 ENC-Bench，首个面向电子航海图 (ENC) 理解的专业级基准，包含 20,490 样本、三级层次评估体系（感知→空间推理→决策），揭示最佳 MLLM 准确率仅 47.88% 的严重能力差距。

**[Expert Pyramid Tuning Efficient Parameter Fine-Tuning For Expertise-Driven Task ](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   针对MoE-LoRA方法中所有expert结构相同（统一rank）导致无法适配不同复杂度任务的问题，提出EPT：通过共享meta-knowledge子空间 + 不同kernel size的反卷积expert构建参数金字塔，配合Adaptive LoRA Pruner和对比学习Task Embedding，在GLUE上以仅0.41M参数/任务达到87.0%平均分，超越所有MoE-LoRA变体。

**[Expert Pyramid Tuning Efficient Parameter Finetuni](expert_pyramid_tuning_efficient_parameter_finetuni.md)**

:   提出 Expert Pyramid Tuning (EPT)，将 CV 中的多尺度特征金字塔思想引入 MoE-LoRA 框架，通过共享元知识子空间 + 不同尺度的反卷积专家 + 对比学习任务嵌入，以仅 0.41M 参数/任务在 GLUE 上达到 87.0% 均分（超越所有 MoE-LoRA 基线）。

**[Fast-Thinkact Efficient Vision-Language-Action Reasoning Via Verbalizable Latent](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)**

:   提出 Fast-ThinkAct，通过将冗长的文本 CoT 推理（~250 token）压缩为 6 个可语言化的连续 latent token，结合 reward-guided preference distillation 和 visual trajectory alignment，实现 89.3% 推理延迟降低（9.3× faster than ThinkAct-7B）同时保持甚至超越 SOTA reasoning VLA 的性能。

**[Force Transferable Visual Jailbreaking Attacks Via Feature Over-Reliance Correct](force_transferable_visual_jailbreaking_attacks_via_feature_over-reliance_correct.md)**

:   通过分析视觉越狱攻击在层特征和频谱域的过度依赖问题，提出FORCE方法纠正非泛化性特征依赖，引导攻击探索更平坦的损失景观，从而显著提升跨模型迁移性。

**[Force Transferable Visual Jailbreaking Attacks Via Feature Over Reliance Correct](force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)**

:   分析发现视觉 jailbreak attack 迁移性差的根因是 attack 处于 high-sharpness loss region——源于浅层特征过度依赖 model-specific 表示和高频信息过度影响；提出 FORCE 方法通过 layer-aware regularization 扩展浅层 feasible region + spectral rescaling 抑制高频非语义成分，引导 attack 进入 flatter loss landscape，显著提升跨模型迁移性。

**[Forcevla2 Unleashing Hybrid Force-Position Control With Force Awareness For Cont](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)**

:   提出ForceVLA2，首个在VLA框架中统一力感知(force awareness)与混合力-位置控制(hybrid force-position control)的端到端模型：通过Force-based Prompts在VLM中构建跨阶段力感知任务概念，Cross-Scale MoE自适应融合任务语义与实时交互力实现闭环力-位置调节，在5个contact-rich任务上平均成功率66%，超π₀和π₀.5分别48.0%和35.0%。

**[Geco-Srt Geometry-Aware Continual Adaptation For Robotic Cross-Task Sim-To-Real ](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)**

:   提出一种基于几何感知的持续适应方法 GeCo-SRT，通过从局部几何特征中提取跨域/跨任务不变知识，在多次 sim-to-real 迁移中实现知识积累，从而高效适应新任务。

**[Gecosrt Geometryaware Continual Adaptation For Rob](gecosrt_geometryaware_continual_adaptation_for_rob.md)**

:   GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。

**[Hif-Vla Hindsight Insight And Foresight Through Motion Representation For Vision](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)**

:   提出 HiF-VLA 框架，通过运动向量（Motion Vector）作为紧凑时间原语，统一回顾（Hindsight）、洞察（Insight）和前瞻（Foresight）三种时间推理能力，实现 VLA 模型的双向时间扩展，在长时操作任务中以极低计算开销大幅超越基线。

**[Influence Malleability In Linearized Attention Dua](influence_malleability_in_linearized_attention_dua.md)**

:   通过NTK框架证明线性化注意力不会收敛到无限宽度核极限（需要宽度m=Ω(κ⁶)），并提出"影响可塑性"指标量化其双面效应：注意力比ReLU网络高6-9倍的数据依赖灵活性，既能降低近似误差也增加对抗脆弱性。

**[Influence Malleability In Linearized Attention Dual Implications Of Non-Converge](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   本文揭示线性化注意力机制在 NTK 框架下不收敛至无穷宽极限，并提出"影响力可塑性"(influence malleability) 度量，证明注意力的强大能力与对抗脆弱性共享同一来源——偏离核regime的数据依赖核结构。

**[Lada Robotic Manipulation](lada_robotic_manipulation.md)**

:   提出LaDA框架，将连续7-DoF动作解耦为平移/旋转/夹爪三个语言锚定的语义原语，通过软标签对比学习和自适应权重策略在共享嵌入空间中对齐跨任务动作表示，在LIBERO上达93.6%成功率（0.6B参数），MimicGen上67%平均成功率，超越所有基线。

**[Language-Grounded Decoupled Action Representation For Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA 框架，将连续 7-DoF 机器人动作解耦为语言描述的可解释运动基元（平移、旋转、夹爪），通过语义引导的软标签对比学习统一视觉-语言-动作表示空间，实现跨任务泛化。

**[Learning To See And Act Task-Aware Virtual View Exploration For Robotic Manipula](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)**

:   提出 TVVE 框架，通过强化学习驱动的多视角探索策略（MVEP）选择最优虚拟相机视角并在线重渲染观测，同时设计任务感知 MoE 视觉编码器（TaskMoE）解决多任务特征干扰问题，在 RLBench 18 个任务上平均成功率达 86.6%。

**[Mergevla Cross-Skill Model Merging Toward A Generalist Vision-Language-Action Ag](mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)**

:   首次系统诊断 VLA 模型不可合并的两大根因（LoRA 自私参数冲突 + 动作专家自注意力导致的任务耦合），提出 MergeVLA——通过任务掩码稀疏激活 LoRA、去自注意力动作专家、无训练测试时路由，将多个单技能 VLA 专家合并为一个通用 agent，在 LIBERO 上达 90.2% 成功率，真机 SO101 达 90%。

**[Mergevla Crossskill Model Merging Toward A General](mergevla_crossskill_model_merging_toward_a_general.md)**

:   MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2% 成功率。

**[Mindpower Enabling Theory-Of-Mind Reasoning In Vlm-Based Embodied Agents](mindpower_enabling_theory-of-mind_reasoning_in_vlm-based_embodied_agents.md)**

:   MindPower 提出以机器人为中心（Robot-Centric）的心智理论推理框架，将感知→信念→欲望→意图→决策→行动组织为三级六层推理层级（MindPower Reasoning Hierarchy），并用 Mind-Reward（基于 GRPO 强化学习）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

**[Mindpower Enabling Theoryofmind Reasoning In Vlmba](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)**

:   MindPower 提出了以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用 Mind-Reward（基于 GRPO）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

**[Panoaffordancenet Towards Holistic Affordance Grou](panoaffordancenet_towards_holistic_affordance_grou.md)**

:   提出PanoAffordanceNet，首次定义360°室内环境中的全局affordance grounding任务，通过失真感知光谱调制器(DASM)和全球面密化头(OSDH)解决ERP几何失真和稀疏激活问题，配合多级训练目标抑制语义漂移，在自建360-AGD数据集上大幅超越现有方法（KLD从2.853→1.270）。

**[Panoaffordancenet Towards Holistic Affordance Grounding In 360 Indoor Environmen](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出首个面向360°全景室内环境的整体affordance定位框架PanoAffordanceNet，通过畸变感知频谱调制器(DASM)和全球面稠密化头(OSDH)系统性解决ERP几何畸变、稀疏功能区域和语义漂移问题，并构建了首个全景affordance数据集360-AGD。

**[Profocus Proactive Perception And Focused Reasoning In Vision-And-Language Navig](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)**

:   通过主动感知（前瞻式环境探索）和聚焦推理（指令相关区域注意力）提升VLN性能

**[Rc-Nf Robot-Conditioned Normalizing Flow For Real-Time Anomaly Detection In Robo](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)**

:   提出 Robot-Conditioned Normalizing Flow (RC-NF)，通过条件归一化流对机器人状态与物体运动轨迹的联合分布建模，实现 <100ms 实时异常检测，可作为 VLA 模型（如 π₀）的即插即用监控模块，支持任务级重规划和状态级轨迹回滚。

**[Rcnf Robot Conditioned Normalizing Flow Anomaly](rcnf_robot_conditioned_normalizing_flow_anomaly.md)**

:   提出RC-NF，一种基于条件归一化流的实时异常检测模型，通过解耦处理机器人状态和物体轨迹特征，仅需正样本无监督训练即可在100ms内检测VLA模型执行中的OOD异常，在LIBERO-Anomaly-10上以约8% AUC和10% AP的优势超越SOTA（包括GPT-5、Gemini 2.5 Pro等VLM基线）。

**[Rehearsevla Simulated Posttraining World Model](rehearsevla_simulated_posttraining_world_model.md)**

:   针对 VLA 模型在数据稀缺场景下的性能退化和真实环境不可重置的限制，提出 RehearseVLA——用物理一致的世界模型模拟器替代真实物理交互进行 RL 后训练，配合 VLM 引导的即时反射器提供奖励信号和终止预测，仅用每个任务 5 个专家演示即可显著提升 VLA 在复杂操控任务上的表现。

**[Sapave Towards Active Perception And Manipulation In Vision-Language-Action Mode](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   提出 SaPaVe 端到端框架，通过解耦相机运动与操控动作的两阶段自底向上学习策略，实现语义驱动的主动感知与视角不变的操控执行，在真实世界任务中超越 GR00T N1 和 π₀ 分别 31.25% 和 40%。

**[Test-Time Ego-Exo-Centric Adaptation For Action Anticipation Via Multi-Label Pro](test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)**

:   首次提出 Test-time Ego-Exo Adaptation for Action Anticipation（TE2A3）任务，设计 DCPGN 网络通过多标签原型增长和双线索（视觉+文本）一致性，在测试时将源视角训练模型在线适配到目标视角进行动作预测，大幅超越现有 TTA 方法。

**[The Coherence Trap When Mllm-Crafted Narratives Exploit Manipulated Visual Conte](the_coherence_trap_when_mllm-crafted_narratives_exploit_manipulated_visual_conte.md)**

:   揭示现有多模态篡改检测忽视了MLLM能生成语义一致的欺骗性叙事这一核心威胁，构建441k样本的MDSM语义对齐篡改数据集，并提出基于Artifact Token和操纵导向推理的AMD框架，在跨域检测中以仅0.27B参数达到88.18 ACC / 60.25 mAP / 61.02 mIoU的最优泛化性能。

**[The Coherence Trap When Mllmcrafted Narratives Exp](the_coherence_trap_when_mllmcrafted_narratives_exp.md)**

:   揭示现有多模态虚假信息检测的两个根本缺陷（低估MLLM生成的语义一致虚假叙事+依赖简单不对齐的伪影），构建441k样本的MDSM数据集（图像篡改+MLLM生成语义对齐文本），并提出AMD框架（Artifact Pre-perception + Manipulation-Oriented Reasoning），在跨域检测中达88.18 ACC / 60.25 mAP / 61.02 mIoU。
