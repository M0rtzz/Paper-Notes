---
title: >-
  ICML2026 多模态 VLM方向30篇论文解读
description: >-
  30篇ICML2026的多模态 VLM 方向论文解读，涵盖多模态、推理、LLM、对抗鲁棒、压缩/编码、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "多模态 VLM"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "LLM"
  - "对抗鲁棒"
  - "压缩/编码"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**🧪 ICML2026** · **30** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (83)](../../ACL2026/multimodal_vlm/index.md) · [📷 CVPR2026 (230)](../../CVPR2026/multimodal_vlm/index.md) · [🔬 ICLR2026 (88)](../../ICLR2026/multimodal_vlm/index.md) · [🤖 AAAI2026 (88)](../../AAAI2026/multimodal_vlm/index.md) · [🧠 NeurIPS2025 (149)](../../NeurIPS2025/multimodal_vlm/index.md) · [📹 ICCV2025 (148)](../../ICCV2025/multimodal_vlm/index.md)

🔥 **高频主题：** 多模态 ×18 · 推理 ×8 · LLM ×5 · 对抗鲁棒 ×3 · 压缩/编码 ×2

**[Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)**

:   作者通过 pilot study 发现"显式把视觉提升到点云、再与 2D patch 融合"是 VLA 注入 3D 信息的最有效方式；为了解决 3D 数据稀缺和不同点云源（仿真/传感器/单目估计）的域差异，提出 Any3D-VLA：用 hybrid point cloud training 学到 source-agnostic 的几何表示，在真实抓取任务上 zero-shot 比最强 baseline 提升 29.2%（62.5% vs 33.3%）。

**[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)**

:   本文把 VLM 的输出强制拆成 `<recognition>` 感知块和 `<think>` 推理块，再用一个"蒙眼"文本推理代理（拿不到图，只看 VLM 写下的感知文字）能不能答对题作为感知奖励 $R_P$，配上结构化语言验证 SVV 作为结果奖励 $R_O$；MoCA 用 $R_P$ 当门控做模态级信用分配，让 7B 模型在 9 个 perception/reasoning/rich-modality benchmark 上同时提升，在多个指标上超过 GPT-4o。

**[Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)**

:   针对统一多模态模型 (unified model) 在 anything-to-image (X2I) 任务上的"理解–生成 gap"（看得懂但生不出），本文提出 Self-Adaptive Interleaved Reasoner：用一个 hierarchical 数据合成 pipeline 在直接生成 / 自我反思 / 多步规划三种模式间分流 5 万条样本，再用 SFT + GRPO 训练并配上 step-wise 推理奖励和 intra-group 复杂度惩罚，让 Emu3.5 在 KRIS-Bench / OmniContext 上超越 GPT-4o、Gemini 2.5 Flash 等闭源模型。

**[Calibrated Multimodal Representation Learning with Missing Modalities](calibrated_multimodal_representation_learning_with_missing_modalities.md)**

:   针对"想用 V-T、A-T 等部分模态数据训练统一多模态对齐"这种现实场景，本文用奇异值扰动给出"缺失模态会导致 anchor shift"的理论上下界，并提出 CalMRL：用概率 PCA 风格的生成模型对缺失模态在表示层做闭式 EM 插补，再把观测 + 插补一起喂给 GRAM/PMRL 的 SVD 对齐目标，在 VAST 之上把跨模态平均 Recall@1 从 44.8 推到 54.2 (+9.4)。

**[DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction](dcer_dual-stage_compression_and_energy-based_reconstruction.md)**

:   DCER 把"模态内频域压缩 + 跨模态 bottleneck token"作为统一的鲁棒融合管道，并用一个学习的能量函数对缺失模态做梯度下降式重建，同时把最终能量值当作内蕴的不确定度，在 MOSI/MOSEI/SIMS 上同时刷新 SOTA。

**[FreeRet: MLLMs as Training-Free Retrievers](freeret_mllms_as_training-free_retrievers.md)**

:   FreeRet 提出一个完全不训练的两阶段多模态检索框架：第一阶段绕过 MLLM 最后一层 MLP 并配合受控生成 prompt 抽取语义忠实的 embedding 做候选检索，第二阶段把 reranking 改成多项选择题来规避 LLM 的 framing 偏置；在 MMEB 上比训练了千万级配对数据的检索模型还要强。

**[FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)**

:   FRISM 把「VLM × LRM 合并」从层级粒度细化到 SVD 子空间粒度：用 LRM 任务向量的 SVD 子空间作为推理先验，再用一个仅含可学习门控的无标签自蒸馏（KL 保视觉 + 谱幅最大化吸收推理）找到最优注入强度，从而在不显著掉视觉的前提下显著提升 VL 推理性能。

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)**

:   本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器），让 INT4 量化的 LLaVA / Qwen-VL 不仅没掉点，反而在多个 benchmark 上超过 BF16 基线，同时实测 3× 吞吐 + 54% 显存节省。

**[Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)**

:   本文把 MLLM 的连续值回归在长尾分布下的"回归到均值"问题转化为分布感知的 RL 问题，在 GRPO 框架内用 Concordance Correlation Coefficient (CCC) 作为批次级奖励——既看相关性、又看方差、又看均值——从而显式惩罚预测分布塌缩；在 4 个长尾回归任务、Qwen2.5-VL-3B/7B 上稳压 SFT、SoftLabel、各种 point-wise RL，特别是 medium/few-shot 区域 MAE 大幅下降。

**[Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)**

:   本文发现 MLLM 中 instruction token 的中间层嵌入能天然过滤视觉端引入的误导信息，据此提出训练无关的 InsLen 分数（Calibrated Local Score + Context Consistency Score），在 5 个 MLLM × 4 个基准上把对象幻觉检测的 AUROC 拉高最多 13.81%。

**[Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)**

:   本文用"信息复杂度 (eRank) + 子空间支持"的几何信息论框架定量诊断 LVLM 的残差流，发现 Attention 几乎只做子空间内重配置而 FFN 才注入新语义维度；更惊人的是把学习到的 attention 权重换成高斯噪声后多数视觉任务性能不降反升，揭示当代 LVLM 在 visual attention 上严重错配冗余。

**[Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)**

:   VISTA 把多模态大模型的自我改进训练改造成"难题靠 prefix 重采样补样本、伪正例靠视觉注意力分数 (VAS) 过滤"的两段式 pipeline，在 Qwen2.5-VL-3B 上把数学/医学多模态推理平均提升 +13.66%。

**[Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)**

:   这篇用 70 万次实验跑遍了 16 种量化方法 × 10 种 VLM × 多项可靠性指标，发现量化不是单纯破坏者——它会通过抑制高 rank 低方差的频谱分量，同时提升 calibration、OOD 检测和噪声鲁棒性，但也会放大对协变量偏移和虚假相关的依赖。

**[LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)**

:   作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉，在三个 AQA 数据集上全面超越依赖完整训练数据的 SOTA。

**[Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)**

:   Model-Dowser 用"权重幅值 × 输入激活 × 输出 Jacobian"三因素给 MLLM 的每个参数打分，冻结高分参数、只更新低分参数，从而在 LLaVA/NVILA 上深层微调时既能学好下游任务又能保留预训练知识，相比 SPIDER、ModelTailor 在 H-score 上稳定领先。

**[Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)**

:   针对 MLLM 在跨场景 VQA 中的视觉遗忘问题，本文构建 MSVQA（高空/水下/低空/室内 4 场景）基准，并提出 Unifier 框架——在视觉 block 里加入 CSR 多分支 + 投影器（VRE）做参数隔离，再用 KL 软约束（VCC）对齐不同分支表征，单次推理即可在 20 步持续学习上把 VQA 提升 2.70-10.62%、F1 提升 3.40-7.69%。

**[MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)**

:   MUSE 把统一视觉 tokenizer 的"理解-生成"零和困境归因于流形错配，提出梯度正交假设——把语义注入 $W_V$ 而结构梯度走 $W_{Q,K}$——并通过 Synergistic Block + DINOv3 拓扑对齐 + NCE 语义锚定彻底解耦，最终 gFID 3.08 与 linear probing 85.2%（甚至超过 InternViT-300M 老师 82.5%）共存，首次实现真正的"互相强化"而非折中。

**[OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)**

:   本文指出现有 Omni-LLM token 压缩方法对音频和视频"对称"处理是次优的，提出 OmniSIFT——先用时空显著性剪掉视频冗余得到"视觉锚点"，再用这些锚点引导音频选择的两阶段非对称压缩框架，仅引入 4.85M 额外参数就在 Qwen2.5-Omni-7B 上保留 25% token 时一致超过现有压缩基线甚至原模型。

**[Perceptual Flow Network for Visually Grounded Reasoning](perceptual_flow_network_for_visually_grounded_reasoning.md)**

:   摒弃"用视觉专家的精确框做硬监督"的传统 RLVR 路线，PFlowNet 把感知行为本身建模为一段结构化的 Perceptual Flow 潜变量，用变分分布 $p_\theta(Z|X)$ 近似面向推理的理想后验，并用 Sub-TB 变分 RL + 多维奖励 + 邻域几何整形 (Vicinal Geometric Shaping) 训练，使得 8B 的 Qwen3-VL 在 V* Bench 拿到 90.6%、MME-RealWorld-lite 67.0% 的新 SOTA。

**[Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)**

:   CSteer 提出一种训练无关的 latent steering 方法,通过在错误/正确指代回答的隐藏激活差上构造"上下文向量",并在推理时分层注入到 query 早期层和 decode 中后期层,让通用 LMM (Qwen3-VL、InternVL-3.5) 在多区域视觉指代任务上反超专门微调的 region LMM。

**[Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)**

:   本文把 LVLM 幻觉重新定义为"被语言先验压制的视觉信息缺失"，用正交投影从原始视觉方向中剔除语言先验得到"纯视觉向量"，再用风险门控只在最优深度的单层做稀疏干预，免训练地把 CHAIRS 幻觉率降 ~19% 同时保住 MM-Vet 通用能力。

**[ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)**

:   本文系统揭示了被广泛使用的 VSI-Bench 因 3D 标注漂移与帧采样不一致而存在结构性失效，进而重新标注 381 个场景、5365 个对象，并设计帧预算自适应 QA 与"删除查询对象帧"的 dummy 视频压力测试，构建出名为 ReVSI 的高保真空间智能基准；评估显示开源 VLM 在 ReVSI 上掉点最多 40%，且在 dummy 视频上幻觉率仍高，暴露出现有空间推理能力被 VSI-Bench 系统性高估。

**[ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision](screenparse_moving_beyond_sparse_grounding_with_complete_screen_parsing_supervis.md)**

:   针对 GUI agent 普遍使用"稀疏 grounding"标注、丢失整屏结构的问题，本文用全自动 Webshot 流水线构建了 771K 截图 / 21M 元素 / 55 类的稠密屏幕解析数据集 ScreenParse，并训练出仅 316M 参数的 ScreenVLM 把整屏解析为 ScreenTag 结构序列，在密集解析与稀疏 grounding 多个 benchmark 上击败 8B 级别的基础 VLM 同时把延迟降到 $\sim 1/4$。

**[Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)**

:   本文用一个"颜色-形状-item"受控合成检索任务复现了"VLM 在纯文本任务上超过其 base LLM"的奇怪现象，并用机制可解释性证明：图像训练让模型把变量绑定策略从"位置捷径"切换到"语义符号匹配"，这一切换在重新接回纯文本后被保留下来，使 OOD 检索准确率从 37.2% 提升到 69.5%；在真实 Qwen2/2.5/3 家族上也观察到一致的"symbolic/positional 比例上升"。

**[Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)**

:   本文借助 Pointwise Partial Information Decomposition 量化视觉-文本模态交互，并提出 Multimodal Interaction Gate：自动挑出「图像独有信息占主导」的样本让 VLM 自我生成 caption 灌入文本侧，从而把 unique 视觉信号转成 redundant 共享信号，使 VLM 在模糊或被污染输入下的视觉幻觉下降 38.3%、一致性提高 16.8%。

**[SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)**

:   SLQ 把一小组"共享潜在查询" $\mathbf{Q}$ 追加到图像/文本 token 序列尾部，借助 MLLM 自身的因果注意力聚合全局上下文，**只训练几千个查询参数**就让冻结的 MLLM 变成检索器，在 COCO/Flickr30K 上胜过全量微调和 LoRA，并配套发布了考验"隐式知识推理"能力的 KARR-Bench。

**[The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)**

:   本文把"VLM 看不清细节"形式化为一个序贯贝叶斯最优实验设计 (S-BOED) 问题,并基于"覆盖率 × 分辨率"的可计算代理目标提出训练免费的 FOVEA 模块,在高分辨率/遥感等基准上稳定超过 Direct 与 ReAct-style 基线。

**[Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)**

:   本文提出 S3 框架，用 MoE 把多模态表征分解为概念级专家（Specialization）、按任务路由激活相关专家（Selection）、并在推理时按路由分数剪枝低贡献路径（Sparsification），在四个 MultiBench 基准上揭示了一条"性能在中间稀疏度达峰"的反 U 型曲线，给出对比学习/InfoMax 之外第三种多模态表征范式。

**[Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)**

:   本文提出 VaLR：在 MLLM 的 CoT 推理每一步之前插入若干"潜在 token"，并用 DINOv3/SigLIP/π³ 等视觉编码器的 patch 特征对这些 token 做表征对齐（REPA），从而在长链推理中持续把视觉信息"喂回"模型，把 Qwen2.5-VL 在 VSI-Bench 上的准确率从 33.0% 拉到 52.9%，并首次让 MLLM 表现出"推理越长越准"的 test-time scaling 行为。

**[What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)**

:   GLANCE 在 VLM agent 的强化学习里加了一个"想-看对齐"自监督头：让 LLM 在 CoT 里产出的"下一状态预测"通过一个轻量 projector 映射到由 EMA target 视觉编码器编码的真实下一帧表示，预测与实际之间的差距同时充当内在好奇心奖励、视觉编码器的训练信号、以及让 internalized world model "落地"的对齐损失；再配合周期性重置 projector 的课程探索机制对抗好奇心衰退，最终在 5 个 agentic 任务上稳定超过现有 exploitation-only 的 VLM-RL 方法。
