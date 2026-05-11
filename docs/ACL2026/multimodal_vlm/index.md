---
title: >-
  ACL2026 多模态 VLM方向50篇论文解读
description: >-
  50篇ACL2026的多模态 VLM 方向论文解读，涵盖多模态、推理、LLM、对抗鲁棒、Agent、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**💬 ACL2026** · **50** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (241)](../../CVPR2026/multimodal_vlm/index.md) · [🔬 ICLR2026 (91)](../../ICLR2026/multimodal_vlm/index.md) · [🤖 AAAI2026 (91)](../../AAAI2026/multimodal_vlm/index.md) · [🧠 NeurIPS2025 (154)](../../NeurIPS2025/multimodal_vlm/index.md) · [📹 ICCV2025 (151)](../../ICCV2025/multimodal_vlm/index.md) · [🧪 ICML2025 (52)](../../ICML2025/multimodal_vlm/index.md)

🔥 **高频主题：** 多模态 ×34 · 推理 ×11 · LLM ×11 · 对抗鲁棒 ×4 · Agent ×3

**[A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)**

:   系统综述基于多模态大语言模型（MLLM）的视觉丰富文档理解（VRDU），从特征表示/融合和训练范式两个维度梳理OCR-based和OCR-free方法，并讨论数据稀缺、多页文档、多语言支持、RAG和智能体等新兴方向。

**[Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)**

:   提出GPRO框架，通过元推理控制器在每个token生成步动态路由计算到三条路径（快速/感知重检/推理反思），解决LVLM的过度思考问题，同时提升精度和效率。

**[AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)**

:   提出 AICA-Bench，一个涵盖情感理解（EU）、情感推理（ER）和情感引导内容生成（EGCG）三个维度的综合基准，评估 23 个 VLM 后发现模型存在强度校准失败和描述浅薄两大缺陷，并提出 GAT Prompting 训练无关框架来缓解这些问题。

**[All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction](all_changes_may_have_invariant_principles_improving_ever-shifting_harmful_meme_d.md)**

:   提出RepMD方法，通过构建设计概念图（DCG）——借鉴攻击树思想描述恶意用户设计有害梗图的步骤和逻辑——来引导MLLM检测不断变化的有害梗图，在GOAT-Bench上达81.1%准确率。

**[Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions](automatic_slide_updating_with_user-defined_dynamic_templates_and_natural_languag.md)**

:   定义了"基于自然语言指令在用户自定义模板上进行动态幻灯片更新"的新任务，构建了包含 20,036 个指令-执行三元组的 DynaSlide 基准，并提出了 SlideAgent 作为强参考基线。

**[Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)**

:   提出 VLM-DeflectionBench，一个包含 2775 个样本的多模态基准，通过四种评估场景（参数化/Oracle/现实/对抗）系统性地评估大型视觉语言模型在证据不足或误导时的拒答（deflection）vs 幻觉（hallucination）行为，实验覆盖 20 个 SOTA LVLM，发现几乎所有模型都无法在噪声证据下可靠拒答。

**[CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)**

:   本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。

**[CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)**

:   CogGen 提出一个模拟人类认知写作过程的多智能体递归框架，通过宏观认知循环实现全局重构、微观认知循环实现并行章节精炼、抽象视觉表示（AVR）实现文本-图表的语义级协同规划，在 OWID 基准上达到人类专家水平并超越 Gemini Deep Research。

**[Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)**

:   提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力，在 WhodunitBench 上显著提升 VLM 的叙事推理、事实提取和欺骗抵御能力。

**[Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)**

:   本文提出 Doc-PP 基准，揭示大型视觉-语言模型（LVLM）在多模态文档问答中存在"推理诱导的安全缺口"——模型在需要跨模态推理时会绕过显式非披露策略泄露敏感信息，并提出 DVA（Decompose–Verify–Aggregation）结构化推理框架来显著降低泄露率。

**[Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)**

:   本文提出VeriGUI框架，通过Thinking-Verification-Action-Expectation（TVAE）闭环推理机制和两阶段训练管线（Robust SFT + GRPO），让GUI Agent能够验证每步操作是否成功并在失败时自我纠正，在3B和7B规模上均显著优于基线。

**[Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)**

:   本文指出现有欺骗检测数据集仅提供受试者级别的情感/人格标签（同一人所有样本共用标签），提出样本级动态标注方案和可靠性加权多模态融合框架 Rel-DDEP，在欺骗检测 F1 上提升 2.53%，情感检测提升 2.66%，人格检测提升 9.30%。

**[Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)**

:   本文提出一个系统性的LVLM推理效率分类体系，围绕编码-预填充-解码三阶段推理流水线分析瓶颈，揭示了"视觉token主导"导致的系统性效率屏障，并梳理了从信息密度塑形、长上下文注意力管理到内存带宽突破的完整优化技术图谱。

**[Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)**

:   本文构建了一个包含11个任务、13万+实例的古汉字演变分析基准，评估了19个MLLM后发现现有模型在字形级识别和演变推理上能力有限，并提出字形驱动对比微调框架GEVO，在2B模型上实现全任务提升。

**[ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)**

:   本文形式化定义了多模态错误检测任务，并构建了 ErrorRadar 基准——包含 2,500 道来自真实学生作答的 K-12 多模态数学题，评估 MLLM 在错误步骤识别（STEP）和错误类型分类（CATE）两个子任务上的能力，发现最强模型 GPT-4o 仍落后人类评估约 10-15%。

**[Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)**

:   本文提出 Faithful-First RPA 框架，通过 FaithEvi 管线在每一步推理中评估感知忠实性（claimed objects 是否在图像中真实存在），以及 FaithAct 机制在推理生成过程中强制执行基于证据的规划和行动，在不降低任务准确率的前提下将感知忠实性提升最高 24%。

**[FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)**

:   FineSteer 将推理时转向分解为两个互补阶段：子空间引导的条件转向（SCS）决定"何时转向"——用 IR 查询子空间的能量比做门控；混合转向专家（MoSE）决定"如何转向"——通过注意力门控网络动态聚合原型专家+残差精炼生成查询特异性转向向量，在安全和真实性 benchmark 上超越 SOTA。

**[From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)**

:   提出 HONES 框架，通过先定位任务关键注意力头再以其为条件引导 FFN 神经元归因，实现了多任务 VLM 中跨异构任务的统一、无梯度的神经元级因果分析和轻量级任务性能提升。

**[From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)**

:   本文提出 MM-Mem，一种受模糊痕迹理论启发的金字塔式多模态记忆架构——将记忆分为感知缓冲层（视觉为主）、情景流层（事件级摘要）和符号图式层（知识图谱）三个层级，通过 SIB-GRPO（语义信息瓶颈+强化学习）自底向上压缩冗余、通过熵驱动自顶向下检索，在 4 个长视频 benchmark 上实现 SOTA。

**[GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)**

:   本文提出 GAMBIT，一种游戏化多模态越狱框架，通过将有害查询分解为拼图图像+隐藏关键词，并嵌入竞争性游戏场景，利用模型的推理激励和认知负荷来绕过安全过滤器，在 Gemini 2.5 Flash 上达到 92.13%、GPT-4o 上达到 85.87% 的攻击成功率，对推理模型和非推理模型均有效。

**[GeoRC: A Benchmark for Geolocation Reasoning Chains](georc_a_benchmark_for_geolocation_reasoning_chains.md)**

:   提出 GeoRC，首个由GeoGuessr冠军级专家撰写的地理定位推理链基准（800条推理链，500个场景），评估VLM生成可审计推理链的能力，发现闭源VLM虽能匹敌人类定位准确率但推理链质量仍大幅落后，开源VLM则几乎等同于纯幻觉基线。

**[HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)**

:   本文发现视觉编码器中存在层级注意力模式——中层关注主体对象、深层关注全局信息，据此提出 HiPrune，一种免训练、模型无关的视觉 token 剪枝方法，通过选择三类 token（Anchor/Buffer/Register）保留不同层级的视觉信息，仅用 1/3 token 保持 99.3% 性能，FLOPs 减少 58.7%。

**[LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)**

:   提出 LaMI，通过后融合架构在预测最后阶段融合视觉特征与 LLM 输出，并在推理时从文本生成多张图像进行基于置信度的聚合，在不损害文本推理能力的前提下显著提升 LLM 的视觉常识推理能力。

**[Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)**

:   提出 ImageProtector，通过在图像中嵌入近不可察觉的对抗扰动作为视觉提示注入攻击，使 MLLM 对被保护图像生成拒绝响应，从而阻止恶意分析者利用开放权重 MLLM 大规模提取图像中的隐私信息。

**[Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)**

:   本文揭示了多模态大模型内容审核中的"对抗走私攻击"（ASA）威胁——将有害内容编码为人可读但 AI 不可读的视觉格式来规避自动检测，构建了包含 1,700 个样本、9 种攻击技术的 SmuggleBench 基准，发现包括 GPT-5 在内的 SOTA 模型攻击成功率超过 90%。

**[MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)**

:   提出 FlowVerse 基准（将数学问题信息分为 DI/EI/RP/OQ 四个组件并构建六个变体版本）和 MathFlow 模块化管线（将感知和推理解耦为独立阶段），训练专门的感知模型 MathFlow-P-7B 从数学图表中提取关键信息，显著提升各类推理模型的视觉数学问题解决能力。

**[MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)**

:   本文提出 MedLayBench-V，首个大规模多模态医学专家-通俗语义对齐基准（79,793 图文对），通过 Structured Concept-Grounded Refinement (SCGR) 流水线将专业放射学报告转化为通俗描述，确保临床语义保真的同时将阅读难度从研究生级别降至高中水平，零样本检索实验表明通俗描述仅带来不到 1% 的性能损失。

**[Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)**

:   本文提出 MPD 框架，通过语义感知正交子空间投影分离幻觉成分，并仅选择性更新与幻觉最相关的少量参数，在减少 23.4% 幻觉的同时保持 97.4% 的通用生成能力，不引入额外推理开销。

**[MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)**

:   本文提出 MMErroR，一个包含 1997 个样本的多模态错误推理基准，每个样本嵌入一个单一推理错误，覆盖 6 大领域和 4 种错误类型，要求 VLM 不仅检测推理链中的错误存在，还要分类错误类型（视觉感知错误/知识应用错误/问题理解错误/推理错误），评测 12 个代表性 VLM 后发现最强模型 Gemini-3-Pro-Preview 也仅达 66.65% 准确率。

**[More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)**

:   本文从认知符号学角度揭示 VLM 的"字面优越偏差"——模型在高保真图像上倾向于字面解读而非隐喻/习语理解，通过引入 DIVA 基准（图标化简化图像）和 Semantic Alignment Gap 指标，证明降低视觉保真度能显著缩小字面与习语解读之间的鸿沟。

**[Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)**

:   本文提出 MT-RL-Judge，一个多任务强化学习框架，通过 GRPO 联合优化多个评估任务训练统一的 MLLM-as-a-Judge 模型，在文本-图像对齐、安全合规和视觉质量评估等六个基准上一致超越 SFT 基线，并在未见过的 MJ-Bench 配对比较格式上展现出强大的分布外泛化能力（Safety 任务 82.23% vs SFT-Unified 的 49.40%）。

**[OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)**

:   本文提出 OMIBench——首个面向奥赛级多图推理的大规模基准，涵盖生物、化学、数学、物理四学科超 1000 道竞赛题，发现即使最强 LVLM（Gemini-3-Pro）也仅达约 50% 准确率，比单图基准下降超 25%。

**[Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)**

:   本文提出 OEA（Omni-Embed-Audio），利用多模态 LLM 作为统一编码器构建检索导向的音频-文本嵌入空间，并引入 User-Intent Queries（UIQ）基准和硬负例区分指标（HNSR/TFR），发现 LLM 主干在 T2T 检索（+22%）和硬负例区分（+4.3%p HNSR@10）上显著优于 CLAP 系列方法。

**[Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)**

:   本文是一篇立场论文（position paper），主张多模态大语言模型（MLLM）可以显著推进跨学科科学推理，提出了四阶段研究路线图（广泛知识识别→类比推理泛化→洞察性推理→创造性假设生成），系统综述了 MLLM 在数学、物理、化学和生物四个领域的应用现状、五大挑战和八个未来方向。

**[Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)**

:   提出表征对比评分（RCS）框架，通过分析 LVLM 内部中间层表征的几何结构，用轻量投影和对比评分区分恶意意图与良性分布偏移，在跨攻击类型泛化的严格评估协议下实现 SOTA 越狱检测性能。

**[SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)**

:   本文提出 SafetyALFRED 基准，在 ALFRED 具身任务中引入六类厨房安全隐患，揭示了多模态大语言模型在静态 QA 中能识别危险（最高 92%）但在具身规划中却难以主动缓解危险（<60%）的严重对齐差距，倡导从 QA 评估范式转向具身安全评估。

**[Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)**

:   提出 Attention-Guided Visual Jailbreaking，通过抑制模型对安全指令的注意力并将注意力锚定到对抗图像特征上，绕过而非强攻安全对齐机制，在 Qwen-VL 上达到 94.4% 攻击成功率，同时减少 45% 的梯度冲突。

**[Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)**

:   提出 DaID (Dual-Anchor Introspective Decoding)，通过挖掘 MLLM 内部不同层的视觉感知差异——Spotlight 层放大视觉信号、Shadow 层抑制语言惯性——在单次前向传播内实现幻觉缓解。

**[Targeted Exploration via Unified Entropy Control for Reinforcement Learning](targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)**

:   本文提出 UEC-RL，一个统一的双向熵控制框架，通过对困难 prompt 进行高温定向探索（增大熵）和通过经验回放稳定器巩固高质量轨迹（减小熵），解决 GRPO 中普遍存在的熵坍塌和训练不稳定问题，在 Geometry3K 上实现 37.9% 的相对提升。

**[TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)**

:   本文提出 TEMA（Text-oriented Entity Mapping Architecture），首个面向多修改文本的组合图像检索（CIR）框架，通过 MMT 解析助手（PA）增强修改实体覆盖、实体映射模块（EM）解决子句-实体对齐问题，并构建了 M-FashionIQ 和 M-CIRR 两个多修改基准数据集，在原始和多修改场景中均取得最优性能。

**[Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding](through_the_magnifying_glass_adaptive_perception_magnification_for_hallucination.md)**

:   本文提出 Perception Magnifier (PM)，一种视觉解码方法，在每个自回归解码步基于多层注意力迭代识别关键视觉区域并自适应放大，通过提升关键区域的有效分辨率来缓解 VLM 的视觉幻觉，同时保持空间结构完整性和推理能力。

**[Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)**

:   提出基于拓扑数据分析的层剪枝框架 TopoVLM，将各层隐藏状态建模为点云并通过 zigzag 持久同调量化层间拓扑一致性，自适应保留关键表征转换层、剪除结构冗余层，在 50-60% 稀疏率下显著优于现有剪枝方法。

**[Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)**

:   本文提出 Tree-of-Evidence（ToE），一种推理时离散束搜索算法，将多模态模型的可解释性形式化为在粗粒度证据单元（生命体征时间窗口、放射报告片段）上的离散优化问题，仅用 5 个证据单元即可保留全输入模型 98% 以上的 AUROC，同时生成可审计的证据追踪路径。

**[TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)**

:   本文提出 TRACE（Textual Representation of Allocentric Context from Egocentric Video），一种提示方法，引导多模态大语言模型从自我中心视频中生成结构化的文本 allocentric 3D 环境表示——包括元上下文、相机轨迹和实体注册表——作为中间推理步骤来增强空间问答能力，在 VSI-Bench 和 OST-Bench 上一致超越已有提示策略。

**[What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)**

:   本文提出 UILoop（UI-in-the-Loop）范式，将 GUI 推理从传统的"屏幕→动作"重构为"屏幕→UI 元素→动作"的循环过程，通过 UI 元素驱动的强化微调教模型显式地定位、理解和利用关键 UI 元素，在 GUI 推理任务上达到 SOTA 性能。

**[What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)**

:   本文通过线性探测发现 VLM 的隐藏表示中编码了丰富的多层次美学属性信息（光照、色彩、构图等），并传播到语言解码器层，基于此提出用简单线性回归实现无需微调的个性化图像美学评估（PIAA），效果显著优于 few-shot 和 LoRA 微调基线。

**[When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)**

:   提出 SaLAD 基准，包含 2013 个真实图文样本覆盖 10 类日常场景，评估多模态大模型在日常辅助中识别隐性安全风险并提供安全警告的能力，揭示即使最强模型在不安全查询上准确率也仅 57.2%。

**[When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning](when_slower_isn39t_truer_inverse_scaling_law_of_truthfulness_in_multimodal_reaso.md)**

:   本文发现多模态推理模型的"逆缩放定律"——慢思考（reasoning）模型在面对误导性视觉输入时比快思考（chat）模型更容易产生不真实输出，并构建了 TruthfulVQA 基准（5000+ 样本、50 名标注员、三层分级提示）和 TruthfulJudge 评估模型（88.4% 准确率）来系统诊断这一现象。

**[When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)**

:   揭示 VLM-as-a-Judge 系统存在严重的"信息量偏见"（informativeness bias）——评判器倾向于选择更详细丰富的回答，即使该回答与图像内容矛盾，提出 BIRCH 范式通过先校正候选答案再进行比较，将偏见减少最高 17%，性能提升最高 9.8%。

**[WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)**

:   提出 WikiSeeker，重新定义 VLM 在多模态 RAG 中的角色——从单纯的答案生成器转变为两个专门化智能体（Refiner 用 RL 训练重写查询、Inspector 验证检索上下文是否可靠），在 EVQA、InfoSeek、M2KR 三个基准上实现 SOTA。
