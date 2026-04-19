---
title: >-
  ICLR2026 目标检测方向 30篇论文解读
description: >-
  30篇ICLR2026 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🔬 ICLR2026** · **30** 篇论文解读

**[A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)**

:   提出以问题为导向（而非用户为导向）的优化对构建方法来整合多程序员的策略多样性，并设计锚点验证框架利用"慢但正确的代码"生成测试用例来缓解"优化税"（正确性损失），将优化比从 31.24% 提升到 71.06%，加速比从 2.95x 提升到 6.08x。

**[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)**

:   提出 AdaRank，用可学习二值掩码自适应选择 task vector 的奇异分量（取代启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰，在 ViT-B/32 上达到 89.4% 准确率。

**[Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)**

:   基于 $W_Q$ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 $Q(X)=(X+f_\theta(X))/2$，在不增加参数的情况下超越 +12.5% 参数的基线模型。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），并提出架构无关的频率表征学习（FRL），通过 Nyquist 归一化频率编码使误差随分辨率提升而下降，在 8 种主流架构上验证有效。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。

**[ConFu: Contemplate the Future for Better Speculative Sampling](confu_contemplate_the_future_for_better_speculative_sampling.md)**

:   提出 ConFu，在推测解码的 draft model 中引入 contemplate tokens 让其预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和生成速度。

**[Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)**

:   通过信息流分析揭示扩散多模态大语言模型（dMLLMs）在使用缓存加速时产生"重复诅咒"的内在机制——context token 作为锚点聚合语义信息，缓存破坏了这一信息流模式——并提出 CoTA 方法将重复率降低高达 92%。

**[CORDS: Continuous Representations of Discrete Structures](cords_continuous_representations_of_discrete_structures.md)**

:   提出 CORDS 框架，通过将变大小离散集合（检测框、分子原子）双射映射为连续的密度场和特征场，使模型可在场空间中学习并精确解码回离散集合，避免了固定 slot 或 padding 的限制。

**[Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)**

:   提出 Contrastive Noise Optimization (CNO)，通过在 Tweedie 去噪预测空间上对初始噪声施加 InfoNCE 对比损失，以预处理方式提升扩散模型生成多样性，同时保持保真度，无需修改采样过程或模型本身。

**[Does FLUX Already Know How to Perform Physically Plausible Image Composition?](does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)**

:   提出 SHINE，一个无需训练的图像合成框架，通过 Manifold-Steered Anchor Loss、Degradation-Suppression Guidance 和 Adaptive Background Blending 三个组件，利用预训练 T2I 模型（如 FLUX）内在的物理先验，实现在复杂光照条件下（阴影、水面反射等）的高质量物体插入。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件，为下冠层人员检测模型的训练和评估提供了坚实基础。

**[From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)**

:   发现多模态 LLM 的推理性能与视觉注意力分数（VAS）高度相关（r=0.96），提出 AVAR 框架通过视觉锚定数据合成、注意力引导训练目标和视觉锚定奖励塑造三个阶段提升 VAS，在 77 个基准上平均提升 7%。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)**

:   构建了迄今最大的 AI 生成同行评审数据集（788,984 篇评审），系统评估了 18 种 AI 文本检测方法在同行评审场景下的表现，并提出了利用论文原文作为上下文的 Anchor 检测方法，在低误报率下大幅超越所有基线。

**[Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)**

:   提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

**[Procedural Mistake Detection via Action Effect Modeling](procedural_mistake_detection_via_action_effect_modeling.md)**

:   提出双分支多模态监督的动作效果建模框架，结合视觉分支（目标状态和空间关系特征）和文本分支（GPT-4o 生成的场景图），通过可学习的效果 token 蒸馏外部监督信号，在第一人称程序视频中实现 SOTA 错误检测。

**[SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)**

:   首次研究联邦 Prompt Learning 场景下的后门攻击威胁，并提出 SABRE-FL——一种基于 embedding 空间异常检测的轻量级服务器端防御方法，无需访问客户端原始数据即可有效过滤中毒 prompt 更新。

**[SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)**

:   提出 SAGE，一个统一的 VPR 训练框架：引入轻量 Soft Probing 模块增强局部特征判别力，每个 epoch 在线重建融合地理距离与视觉相似度的亲和图，再通过贪心加权团扩展聚焦最难样本，冻结 DINOv2 骨干仅训练 1.96M 参数即在 8 个基准上全面 SOTA。

**[SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation](serum_simple_efficient_robust_and_unifying_marking_for_diffusion-based_image_gen.md)**

:   提出SERUM水印方法，将唯一水印噪声添加到扩散模型初始噪声中，训练轻量检测器直接从生成图像识别水印（无需昂贵的DDIM反演），在多种攻击下达到最高检测率，且注入/检测极快，支持多用户场景。

**[SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)**

:   提出SpectralGCD，将图像表示为CLIP概念字典上的语义混合（跨模态相似度向量），通过谱过滤自动选择任务相关概念，配合正反知识蒸馏保持语义质量，在6个基准上以与单模态方法可比的计算代价达到多模态SOTA。

**[SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)**

:   提出 SPWOOD 框架统一处理稀疏标注和弱标注（HBox/Point）的旋转目标检测问题，通过自适应旋转目标检测器(SAOD)和空间布局学习策略，在 DOTA 基准上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。

**[Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)**

:   提出 AdaAnchor 潜空间推理框架——将可学习的锚向量（anchor vectors）附加到输入嵌入中，通过迭代前向传播精炼锚状态实现"沉默思考"，配合基于锚稳定性的自适应停止机制按实例难度动态分配计算量，在数学推理任务上比固定步潜推理准确率提升最高 5%、平均步数减少 48–60%，输出 token 相比 CoT 减少 92–93%。

**[Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)**

:   提出 RAGLens，利用稀疏自编码器(SAE)从 LLM 内部激活中解耦出 RAG 幻觉专属特征，通过互信息特征选择 + 广义加性模型(GAM)构建轻量级可解释幻觉检测器，在多个基准上超越现有方法，并支持 token 级可解释反馈与幻觉缓解。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。

**[VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)**

:   VidGuard-R1 是首个采用 GRPO（Group Relative Policy Optimization）强化学习微调 MLLM 的视频真伪检测器，通过构建 14 万无快捷方式的真/假视频对数据集，并设计时序伪影奖励和扩散步数质量奖励两种专用奖励机制，在自建数据集上达到 86.17% 准确率，在 GenVidBench 和 GenVideo 基准上实现 95%+ 的 SOTA 零样本检测性能，同时生成可解释的思维链推理。

**[VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)**

:   提出VLBiMan框架，通过任务感知双臂分解将单次演示拆分为不变/可适应原子技能，利用VLM视觉-语言锚定在新场景中适应物体位置和实例变化，结合运动学感知的轨迹组合实现双臂协调——在10个复杂双臂任务上以1次演示达到85.3%成功率远超需上百次演示的模仿学习基线。

**[What Layers When: Learning to Skip Compute in LLMs with Residual Gates](what_layers_when_learning_to_skip_compute_in_llms_with_residual_gates.md)**

:   提出 GateSkip——在 decoder-only Transformer 每个 Attention/MLP 分支输出处插入一个 sigmoid-linear 门控，微调时联合学习门控稀疏性与语言建模目标，推理时按门控值用分位数阈值确定性跳过低重要性 token，实现 token 级逐层自适应深度；在 Llama 8B 上节省 15% 计算保持 >90% 精度，指令微调模型全计算反而提升精度、约 50% 节省仍匹配基线，且与 INT4 量化/结构化剪枝/自推测解码正交可组合。

**[When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)**

:   本文首次系统研究了 LLM 多智能体系统中的曼德拉效应（集体虚假记忆），提出 ManBench 基准（4838 个问题、5 种交互协议），发现所有 13 个被评估的 LLM 均易受此效应影响，并提出 prompt 级和模型级缓解策略，平均减少 74.40% 的虚假记忆。

**[Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)**

:   提出将目标检测与交互识别完全解耦的零样本 HOI 检测框架 DA-HOI，利用 MLLM 的 VQA 能力替代传统 CLIP 特征做交互识别，核心贡献是确定性生成（training-free 即达 31.50 mAP）、空间感知池化（引入空间先验和跨注意力）和单次确定性匹配（M 次前向变 1 次），在 HICO-DET 四种零样本设定下全面超越 SOTA，且训练后可即插即用切换任意检测器。
