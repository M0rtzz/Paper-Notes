---
title: >-
  639 篇 ACL2026 论文解读 · 每篇 5 分钟读懂
description: >-
  639篇ACL2026论文解读，涵盖多模态 VLM(50篇)、信息检索/RAG(43篇)、模型压缩(42篇)、LLM Agent(41篇)、LLM 评测(40篇)、医学图像(40篇)、LLM 推理(37篇)、LLM / NLP(35篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2026"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "多模态 VLM"
  - "信息检索/RAG"
  - "模型压缩"
  - "LLM Agent"
  - "LLM 评测"
  - "医学图像"
  - "LLM 推理"
  - "LLM / NLP"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 ACL2026 论文笔记

639篇ACL2026论文解读，涵盖多模态 VLM(50篇)、信息检索/RAG(43篇)、模型压缩(42篇)、LLM Agent(41篇)、LLM 评测(40篇)、医学图像(40篇)、LLM 推理(37篇)、LLM / NLP(35篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](multimodal_vlm/a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)**

:   系统综述基于多模态大语言模型（MLLM）的视觉丰富文档理解（VRDU），从特征表示/融合和训练范式两个维度梳理OCR-based和OCR-free方法，并讨论数据稀缺、多页文档、多语言支持、RAG和智能体等新兴方向。

**[Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](multimodal_vlm/addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)**

:   提出GPRO框架，通过元推理控制器在每个token生成步动态路由计算到三条路径（快速/感知重检/推理反思），解决LVLM的过度思考问题，同时提升精度和效率。

**[AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](multimodal_vlm/aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)**

:   提出 AICA-Bench，一个涵盖情感理解（EU）、情感推理（ER）和情感引导内容生成（EGCG）三个维度的综合基准，评估 23 个 VLM 后发现模型存在强度校准失败和描述浅薄两大缺陷，并提出 GAT Prompting 训练无关框架来缓解这些问题。

**[All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction](multimodal_vlm/all_changes_may_have_invariant_principles_improving_ever-shifting_harmful_meme_d.md)**

:   提出RepMD方法，通过构建设计概念图（DCG）——借鉴攻击树思想描述恶意用户设计有害梗图的步骤和逻辑——来引导MLLM检测不断变化的有害梗图，在GOAT-Bench上达81.1%准确率。

**[Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions](multimodal_vlm/automatic_slide_updating_with_user-defined_dynamic_templates_and_natural_languag.md)**

:   定义了"基于自然语言指令在用户自定义模板上进行动态幻灯片更新"的新任务，构建了包含 20,036 个指令-执行三元组的 DynaSlide 基准，并提出了 SlideAgent 作为强参考基线。

**[Benchmarking Deflection and Hallucination in Large Vision-Language Models](multimodal_vlm/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)**

:   提出 VLM-DeflectionBench，一个包含 2775 个样本的多模态基准，通过四种评估场景（参数化/Oracle/现实/对抗）系统性地评估大型视觉语言模型在证据不足或误导时的拒答（deflection）vs 幻觉（hallucination）行为，实验覆盖 20 个 SOTA LVLM，发现几乎所有模型都无法在噪声证据下可靠拒答。

**[CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](multimodal_vlm/cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)**

:   本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。

**[CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)**

:   CogGen 提出一个模拟人类认知写作过程的多智能体递归框架，通过宏观认知循环实现全局重构、微观认知循环实现并行章节精炼、抽象视觉表示（AVR）实现文本-图表的语义级协同规划，在 OWID 基准上达到人类专家水平并超越 Gemini Deep Research。

**[Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](multimodal_vlm/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)**

:   提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力，在 WhodunitBench 上显著提升 VLM 的叙事推理、事实提取和欺骗抵御能力。

**[Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](multimodal_vlm/doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)**

:   本文提出 Doc-PP 基准，揭示大型视觉-语言模型（LVLM）在多模态文档问答中存在"推理诱导的安全缺口"——模型在需要跨模态推理时会绕过显式非披露策略泄露敏感信息，并提出 DVA（Decompose–Verify–Aggregation）结构化推理框架来显著降低泄露率。

**[Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](multimodal_vlm/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)**

:   本文提出VeriGUI框架，通过Thinking-Verification-Action-Expectation（TVAE）闭环推理机制和两阶段训练管线（Robust SFT + GRPO），让GUI Agent能够验证每步操作是否成功并在失败时自我纠正，在3B和7B规模上均显著优于基线。

**[Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](multimodal_vlm/dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)**

:   本文指出现有欺骗检测数据集仅提供受试者级别的情感/人格标签（同一人所有样本共用标签），提出样本级动态标注方案和可靠性加权多模态融合框架 Rel-DDEP，在欺骗检测 F1 上提升 2.53%，情感检测提升 2.66%，人格检测提升 9.30%。

**[Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](multimodal_vlm/efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)**

:   本文提出一个系统性的LVLM推理效率分类体系，围绕编码-预填充-解码三阶段推理流水线分析瓶颈，揭示了"视觉token主导"导致的系统性效率屏障，并梳理了从信息密度塑形、长上下文注意力管理到内存带宽突破的完整优化技术图谱。

**[Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](multimodal_vlm/enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)**

:   本文构建了一个包含11个任务、13万+实例的古汉字演变分析基准，评估了19个MLLM后发现现有模型在字形级识别和演变推理上能力有限，并提出字形驱动对比微调框架GEVO，在2B模型上实现全任务提升。

**[ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](multimodal_vlm/errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)**

:   本文形式化定义了多模态错误检测任务，并构建了 ErrorRadar 基准——包含 2,500 道来自真实学生作答的 K-12 多模态数学题，评估 MLLM 在错误步骤识别（STEP）和错误类型分类（CATE）两个子任务上的能力，发现最强模型 GPT-4o 仍落后人类评估约 10-15%。

**[Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](multimodal_vlm/faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)**

:   本文提出 Faithful-First RPA 框架，通过 FaithEvi 管线在每一步推理中评估感知忠实性（claimed objects 是否在图像中真实存在），以及 FaithAct 机制在推理生成过程中强制执行基于证据的规划和行动，在不降低任务准确率的前提下将感知忠实性提升最高 24%。

**[FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](multimodal_vlm/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)**

:   FineSteer 将推理时转向分解为两个互补阶段：子空间引导的条件转向（SCS）决定"何时转向"——用 IR 查询子空间的能量比做门控；混合转向专家（MoSE）决定"如何转向"——通过注意力门控网络动态聚合原型专家+残差精炼生成查询特异性转向向量，在安全和真实性 benchmark 上超越 SOTA。

**[From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](multimodal_vlm/from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)**

:   提出 HONES 框架，通过先定位任务关键注意力头再以其为条件引导 FFN 神经元归因，实现了多任务 VLM 中跨异构任务的统一、无梯度的神经元级因果分析和轻量级任务性能提升。

**[From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](multimodal_vlm/from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)**

:   本文提出 MM-Mem，一种受模糊痕迹理论启发的金字塔式多模态记忆架构——将记忆分为感知缓冲层（视觉为主）、情景流层（事件级摘要）和符号图式层（知识图谱）三个层级，通过 SIB-GRPO（语义信息瓶颈+强化学习）自底向上压缩冗余、通过熵驱动自顶向下检索，在 4 个长视频 benchmark 上实现 SOTA。

**[GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](multimodal_vlm/gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)**

:   本文提出 GAMBIT，一种游戏化多模态越狱框架，通过将有害查询分解为拼图图像+隐藏关键词，并嵌入竞争性游戏场景，利用模型的推理激励和认知负荷来绕过安全过滤器，在 Gemini 2.5 Flash 上达到 92.13%、GPT-4o 上达到 85.87% 的攻击成功率，对推理模型和非推理模型均有效。

**[GeoRC: A Benchmark for Geolocation Reasoning Chains](multimodal_vlm/georc_a_benchmark_for_geolocation_reasoning_chains.md)**

:   提出 GeoRC，首个由GeoGuessr冠军级专家撰写的地理定位推理链基准（800条推理链，500个场景），评估VLM生成可审计推理链的能力，发现闭源VLM虽能匹敌人类定位准确率但推理链质量仍大幅落后，开源VLM则几乎等同于纯幻觉基线。

**[HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)**

:   本文发现视觉编码器中存在层级注意力模式——中层关注主体对象、深层关注全局信息，据此提出 HiPrune，一种免训练、模型无关的视觉 token 剪枝方法，通过选择三类 token（Anchor/Buffer/Register）保留不同层级的视觉信息，仅用 1/3 token 保持 99.3% 性能，FLOPs 减少 58.7%。

**[LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](multimodal_vlm/lami_augmenting_large_language_models_via_late_multi-image_fusion.md)**

:   提出 LaMI，通过后融合架构在预测最后阶段融合视觉特征与 LLM 输出，并在推理时从文本生成多张图像进行基于置信度的聚合，在不损害文本推理能力的前提下显著提升 LLM 的视觉常识推理能力。

**[Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](multimodal_vlm/leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)**

:   提出 ImageProtector，通过在图像中嵌入近不可察觉的对抗扰动作为视觉提示注入攻击，使 MLLM 对被保护图像生成拒绝响应，从而阻止恶意分析者利用开放权重 MLLM 大规模提取图像中的隐私信息。

**[Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](multimodal_vlm/making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)**

:   本文揭示了多模态大模型内容审核中的"对抗走私攻击"（ASA）威胁——将有害内容编码为人可读但 AI 不可读的视觉格式来规避自动检测，构建了包含 1,700 个样本、9 种攻击技术的 SmuggleBench 基准，发现包括 GPT-5 在内的 SOTA 模型攻击成功率超过 90%。

**[MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](multimodal_vlm/mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)**

:   提出 FlowVerse 基准（将数学问题信息分为 DI/EI/RP/OQ 四个组件并构建六个变体版本）和 MathFlow 模块化管线（将感知和推理解耦为独立阶段），训练专门的感知模型 MathFlow-P-7B 从数学图表中提取关键信息，显著提升各类推理模型的视觉数学问题解决能力。

**[MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](multimodal_vlm/medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)**

:   本文提出 MedLayBench-V，首个大规模多模态医学专家-通俗语义对齐基准（79,793 图文对），通过 Structured Concept-Grounded Refinement (SCGR) 流水线将专业放射学报告转化为通俗描述，确保临床语义保真的同时将阅读难度从研究生级别降至高中水平，零样本检索实验表明通俗描述仅带来不到 1% 的性能损失。

**[Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)**

:   本文提出 MPD 框架，通过语义感知正交子空间投影分离幻觉成分，并仅选择性更新与幻觉最相关的少量参数，在减少 23.4% 幻觉的同时保持 97.4% 的通用生成能力，不引入额外推理开销。

**[MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](multimodal_vlm/mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)**

:   本文提出 MMErroR，一个包含 1997 个样本的多模态错误推理基准，每个样本嵌入一个单一推理错误，覆盖 6 大领域和 4 种错误类型，要求 VLM 不仅检测推理链中的错误存在，还要分类错误类型（视觉感知错误/知识应用错误/问题理解错误/推理错误），评测 12 个代表性 VLM 后发现最强模型 Gemini-3-Pro-Preview 也仅达 66.65% 准确率。

**[More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](multimodal_vlm/more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)**

:   本文从认知符号学角度揭示 VLM 的"字面优越偏差"——模型在高保真图像上倾向于字面解读而非隐喻/习语理解，通过引入 DIVA 基准（图标化简化图像）和 Semantic Alignment Gap 指标，证明降低视觉保真度能显著缩小字面与习语解读之间的鸿沟。

**[Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multimodal_vlm/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)**

:   本文提出 MT-RL-Judge，一个多任务强化学习框架，通过 GRPO 联合优化多个评估任务训练统一的 MLLM-as-a-Judge 模型，在文本-图像对齐、安全合规和视觉质量评估等六个基准上一致超越 SFT 基线，并在未见过的 MJ-Bench 配对比较格式上展现出强大的分布外泛化能力（Safety 任务 82.23% vs SFT-Unified 的 49.40%）。

**[OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](multimodal_vlm/omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)**

:   本文提出 OMIBench——首个面向奥赛级多图推理的大规模基准，涵盖生物、化学、数学、物理四学科超 1000 道竞赛题，发现即使最强 LVLM（Gemini-3-Pro）也仅达约 50% 准确率，比单图基准下降超 25%。

**[Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](multimodal_vlm/omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)**

:   本文提出 OEA（Omni-Embed-Audio），利用多模态 LLM 作为统一编码器构建检索导向的音频-文本嵌入空间，并引入 User-Intent Queries（UIQ）基准和硬负例区分指标（HNSR/TFR），发现 LLM 主干在 T2T 检索（+22%）和硬负例区分（+4.3%p HNSR@10）上显著优于 CLAP 系列方法。

**[Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](multimodal_vlm/position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)**

:   本文是一篇立场论文（position paper），主张多模态大语言模型（MLLM）可以显著推进跨学科科学推理，提出了四阶段研究路线图（广泛知识识别→类比推理泛化→洞察性推理→创造性假设生成），系统综述了 MLLM 在数学、物理、化学和生物四个领域的应用现状、五大挑战和八个未来方向。

**[Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](multimodal_vlm/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)**

:   提出表征对比评分（RCS）框架，通过分析 LVLM 内部中间层表征的几何结构，用轻量投影和对比评分区分恶意意图与良性分布偏移，在跨攻击类型泛化的严格评估协议下实现 SOTA 越狱检测性能。

**[SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](multimodal_vlm/safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)**

:   本文提出 SafetyALFRED 基准，在 ALFRED 具身任务中引入六类厨房安全隐患，揭示了多模态大语言模型在静态 QA 中能识别危险（最高 92%）但在具身规划中却难以主动缓解危险（<60%）的严重对齐差距，倡导从 QA 评估范式转向具身安全评估。

**[Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](multimodal_vlm/seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)**

:   提出 Attention-Guided Visual Jailbreaking，通过抑制模型对安全指令的注意力并将注意力锚定到对抗图像特征上，绕过而非强攻安全对齐机制，在 Qwen-VL 上达到 94.4% 攻击成功率，同时减少 45% 的梯度冲突。

**[Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](multimodal_vlm/spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)**

:   提出 DaID (Dual-Anchor Introspective Decoding)，通过挖掘 MLLM 内部不同层的视觉感知差异——Spotlight 层放大视觉信号、Shadow 层抑制语言惯性——在单次前向传播内实现幻觉缓解。

**[Targeted Exploration via Unified Entropy Control for Reinforcement Learning](multimodal_vlm/targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)**

:   本文提出 UEC-RL，一个统一的双向熵控制框架，通过对困难 prompt 进行高温定向探索（增大熵）和通过经验回放稳定器巩固高质量轨迹（减小熵），解决 GRPO 中普遍存在的熵坍塌和训练不稳定问题，在 Geometry3K 上实现 37.9% 的相对提升。

**[TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)**

:   本文提出 TEMA（Text-oriented Entity Mapping Architecture），首个面向多修改文本的组合图像检索（CIR）框架，通过 MMT 解析助手（PA）增强修改实体覆盖、实体映射模块（EM）解决子句-实体对齐问题，并构建了 M-FashionIQ 和 M-CIRR 两个多修改基准数据集，在原始和多修改场景中均取得最优性能。

**[Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding](multimodal_vlm/through_the_magnifying_glass_adaptive_perception_magnification_for_hallucination.md)**

:   本文提出 Perception Magnifier (PM)，一种视觉解码方法，在每个自回归解码步基于多层注意力迭代识别关键视觉区域并自适应放大，通过提升关键区域的有效分辨率来缓解 VLM 的视觉幻觉，同时保持空间结构完整性和推理能力。

**[Topology-Aware Layer Pruning for Large Vision-Language Models](multimodal_vlm/topology-aware_layer_pruning_for_large_vision-language_models.md)**

:   提出基于拓扑数据分析的层剪枝框架 TopoVLM，将各层隐藏状态建模为点云并通过 zigzag 持久同调量化层间拓扑一致性，自适应保留关键表征转换层、剪除结构冗余层，在 50-60% 稀疏率下显著优于现有剪枝方法。

**[Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](multimodal_vlm/tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)**

:   本文提出 Tree-of-Evidence（ToE），一种推理时离散束搜索算法，将多模态模型的可解释性形式化为在粗粒度证据单元（生命体征时间窗口、放射报告片段）上的离散优化问题，仅用 5 个证据单元即可保留全输入模型 98% 以上的 AUROC，同时生成可审计的证据追踪路径。

**[TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](multimodal_vlm/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)**

:   本文提出 TRACE（Textual Representation of Allocentric Context from Egocentric Video），一种提示方法，引导多模态大语言模型从自我中心视频中生成结构化的文本 allocentric 3D 环境表示——包括元上下文、相机轨迹和实体注册表——作为中间推理步骤来增强空间问答能力，在 VSI-Bench 和 OST-Bench 上一致超越已有提示策略。

**[What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](multimodal_vlm/what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)**

:   本文提出 UILoop（UI-in-the-Loop）范式，将 GUI 推理从传统的"屏幕→动作"重构为"屏幕→UI 元素→动作"的循环过程，通过 UI 元素驱动的强化微调教模型显式地定位、理解和利用关键 UI 元素，在 GUI 推理任务上达到 SOTA 性能。

**[What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)**

:   本文通过线性探测发现 VLM 的隐藏表示中编码了丰富的多层次美学属性信息（光照、色彩、构图等），并传播到语言解码器层，基于此提出用简单线性回归实现无需微调的个性化图像美学评估（PIAA），效果显著优于 few-shot 和 LoRA 微调基线。

**[When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](multimodal_vlm/when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)**

:   提出 SaLAD 基准，包含 2013 个真实图文样本覆盖 10 类日常场景，评估多模态大模型在日常辅助中识别隐性安全风险并提供安全警告的能力，揭示即使最强模型在不安全查询上准确率也仅 57.2%。

**[When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning](multimodal_vlm/when_slower_isn39t_truer_inverse_scaling_law_of_truthfulness_in_multimodal_reaso.md)**

:   本文发现多模态推理模型的"逆缩放定律"——慢思考（reasoning）模型在面对误导性视觉输入时比快思考（chat）模型更容易产生不真实输出，并构建了 TruthfulVQA 基准（5000+ 样本、50 名标注员、三层分级提示）和 TruthfulJudge 评估模型（88.4% 准确率）来系统诊断这一现象。

**[When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](multimodal_vlm/when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)**

:   揭示 VLM-as-a-Judge 系统存在严重的"信息量偏见"（informativeness bias）——评判器倾向于选择更详细丰富的回答，即使该回答与图像内容矛盾，提出 BIRCH 范式通过先校正候选答案再进行比较，将偏见减少最高 17%，性能提升最高 9.8%。

**[WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)**

:   提出 WikiSeeker，重新定义 VLM 在多模态 RAG 中的角色——从单纯的答案生成器转变为两个专门化智能体（Refiner 用 RL 训练重写查询、Inspector 验证检索上下文是否可靠），在 EVQA、InfoSeek、M2KR 三个基准上实现 SOTA。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](information_retrieval/all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)**

:   系统揭示多语言 RAG 系统在重排序阶段存在严重的语言偏差（偏好英语和查询语言），提出 LAURA 框架通过下游生成质量驱动的监督信号对齐重排序器，有效缓解偏差并提升生成性能。

**[An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)**

:   受Schutz哲学相关性理论启发，提出ITEM迭代效用判断框架，通过让RAG中的三个组件（相关性排序、效用判断、答案生成）动态交互增强，在检索、效用判断和QA任务上均优于基线。

**[Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](information_retrieval/bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)**

:   提出 BAGEL，一个基于高斯过程（GP）的贝叶斯主动学习框架，在有限 LLM 预算下通过探索-利用平衡策略传播稀疏 LLM 相关性信号，实现全局嵌入空间的段落检索，显著超越传统 LLM 重排序方法。

**[Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](information_retrieval/beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)**

:   提出 ProbeRAG，通过发现 LLM 隐空间中冲突/对齐知识的线性可分性，设计三阶段框架（细粒度知识剪枝→隐空间冲突探测→冲突感知注意力），从模型内部机制解决 RAG 忠实性问题。

**[Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](information_retrieval/beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)**

:   形式化定义 RAG 系统的"软失败"威胁（生成流畅但无信息量的回答），提出 DEJA 黑箱进化攻击框架，通过对抗性文档诱导模型利用安全对齐机制产生模棱两可的回答，SASR 超过 79% 且高度隐蔽。

**[CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](information_retrieval/caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)**

:   提出 CarO（Chain-of-Analogy Reasoning Optimization），一个两阶段训练框架，通过 RAG 引导生成类比推理链 + SFT + 定制 DPO 优化，使 LLM 在推理时自主生成类比参考案例进行内容审核，在模糊审核基准上 F1 平均提升 24.9%，显著超越推理模型（DeepSeek R1）和专用审核模型（LLaMA Guard）。

**[ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](information_retrieval/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)**

:   提出 ChAIRO，一个上下文层次化类比归纳与推理优化框架，通过三阶段 pipeline（类比案例生成→规则归纳→规则注入微调）让 LLM 在内容审核中自主生成类比案例并归纳显式审核规则，比单实例规则生成提升 F1 4.5%，比静态 RAG 提升 2.3%。

**[ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](information_retrieval/chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)**

:   提出 ChunQiuTR，首个基于非格里历的时间键检索基准，从《春秋》及其注疏传统中构建，并设计了 CTD（历法时间双编码器），通过傅里叶绝对历法上下文和相对偏移偏置实现时间感知检索，显著优于纯语义基线。

**[CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](information_retrieval/codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)**

:   提出 CodePromptZip，首个面向代码的提示压缩框架，通过类型感知优先级排序构建训练数据并训练带 copy 机制的小模型压缩器，在三个编码任务上分别比最佳基线提升 23.4%、28.7% 和 8.7%。

**[Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation](information_retrieval/conjecture_and_inquiry_quantifying_software_performance_requirements_via_interac.md)**

:   提出IRAP方法，通过交互式检索增强偏好获取（Interactive Retrieval-Augmented Preference Elicitation）将自然语言描述的软件性能需求量化为数学函数，在4个真实数据集上相比10种SOTA方法取得最高40倍的性能提升，且仅需5轮交互。

**[Context Attribution with Multi-Armed Bandit Optimization](information_retrieval/context_attribution_with_multi-armed_bandit_optimization.md)**

:   本文提出 CAMAB，将 RAG 中的上下文归因（识别哪些上下文片段对生成答案有贡献）建模为组合多臂赌博机（CMAB）问题，使用线性 Thompson 采样自适应地探索上下文子集空间，在 HotpotQA、CNN/DM、TyDi QA 上比 SHAP 和 ContextCite 减少最多 30% 的模型查询次数同时匹配或超越归因质量。

**[CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)**

:   本文提出 CounterRefine，一个轻量级推理时修复层：先用标准 RAG 产生初步答案，再通过答案条件化的反证检索收集支持/反对证据，最后通过受限的 KEEP/REVISE 决策和确定性验证修复错误答案，在 SimpleQA 上将 GPT-5 的正确率从 67.3% 提升至 73.1%。

**[CRAFT: Training-Free Cascaded Retrieval for Tabular QA](information_retrieval/craft_training-free_cascaded_retrieval_for_tabular_qa.md)**

:   本文提出 CRAFT，一个无需数据集特定训练的三阶段级联表格检索框架（SPLADE 稀疏过滤 → 语义 mini-table 排序 → 神经重排序），通过 Gemini 生成的表格标题和描述增强表格表示，在 NQ-Tables 上达到 SOTA（R@1 49.84），在 OTT-QA 上展现强零样本泛化能力，且对查询改写具有显著鲁棒性。

**[CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](information_retrieval/curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)**

:   CURaTE 提出一种基于句子嵌入匹配的行为遗忘框架：预部署时训练一个通用的遗忘嵌入器（不使用任何遗忘集），部署后实时将新遗忘请求嵌入存入数据库，推理时通过余弦相似度决定是回答还是拒绝，完全不修改 LLM 权重从而实现近乎完美的知识保留。

**[Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](information_retrieval/detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)**

:   提出 CanaryRAG，一个受软件安全中栈金丝雀启发的 RAG 运行时防御机制，通过在检索块中注入非语义金丝雀 token 并设计双路径完整性博弈（目标路径不应泄露金丝雀 + Oracle 路径应能引出金丝雀），实时检测知识库提取攻击，在不影响任务性能和推理延迟的前提下实现最强防护。

**[Domain-Specific Data Generation Framework for RAG Adaptation](information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)**

:   本文提出 RAGen，一个可扩展的模块化数据生成框架，通过文档级概念提取、多块证据组装和 Bloom 分类学引导的问题生成，自动合成领域特定的 QAC（问题-答案-上下文）数据，支持嵌入模型对比微调和 LLM 监督微调，在三个领域数据集上显著优于 AutoRAG 和 LlamaIndex 基线。

**[DQA: Diagnostic Question Answering for IT Support](information_retrieval/dqa_diagnostic_question_answering_for_it_support.md)**

:   本文提出DQA框架，通过维护持久化的诊断状态和在根因层面聚合检索证据（而非逐文档处理），实现企业IT支持场景下的系统化故障排查，成功率从基线41.3%提升至78.7%，平均轮次从8.4降至3.9。

**[End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)**

:   本文提出 MHGPO（Multi-Agent Heterogeneous Group Policy Optimization），一种无需 critic 的多智能体 RL 方法，通过异构组相对优势估计和反向奖励传播，在三智能体搜索系统（Rewriter→Reranker→Answerer）中实现端到端优化，捕获隐式跨智能体依赖和跨轨迹关联，在 HotpotQA 等多跳 QA 基准上显著优于 MAPPO 和 GRPO 基线。

**[Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](information_retrieval/enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)**

:   CW-GRPO 将过程监督重新定义为"优势重分配"：用 LLM 判断器评估每轮搜索的检索有用性和推理正确性，计算贡献分数来缩放基于结果的优势，实现轮级别信用分配而不引入不稳定的价值函数，在 Qwen3-8B 上超越标准 GRPO 5.0%。

**[Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)**

:   本文发现多语言 RAG 系统中"英语偏好"主要是评估基准中结构性先验（gold 证据集中于英语、文化先验）的伪影而非模型固有偏差，提出去偏语言偏好指标 DeLP 揭示检索器实际偏好单语对齐，并基于此设计 DELTA 查询增强框架，在多语言 RAG 上一致超越英语枢轴策略。

**[FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](information_retrieval/faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)**

:   本文提出FAITH框架，通过将LLM的不确定性信号（一致性+语义熵）映射到自然语言描述的知识状态象限（可信度×诚实度），设计考虑不确定性的细粒度奖励函数进行PPO训练，再用RAG模块纠正潜在错误，系统性提升LLM的事实准确性。

**[Feedback Adaptation for Retrieval-Augmented Generation](information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)**

:   本文提出"反馈适应"作为RAG系统的新问题设定——研究纠正性反馈多快、多有效地传播到未来查询，定义了纠正延迟和反馈后性能两个评估轴，并提出PatchRAG作为免训练的推理时反馈整合方案，实现即时纠正和强泛化。

**[FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)**

:   提出FLARE框架，利用正则化流（Normalizing Flows）进行无标签的文本嵌入模型评估，通过直接从对数似然估计信息充分性来避免基于距离的密度估计在高维空间中的崩溃，在11个数据集上与有监督基准的Spearman $\rho$ 达0.90。

**[From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](information_retrieval/from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)**

:   本文提出AuthGR，首个将文档权威性系统性整合到生成式检索中的框架，通过VLM多模态权威评分、三阶段渐进式训练（CPT→SFT→GRPO）和混合集成部署管线，在Naver商业搜索引擎的大规模A/B测试中验证了显著的用户参与度提升。

**[How Retrieved Context Shapes Internal Representations in RAG](information_retrieval/how_retrieved_context_shapes_internal_representations_in_rag.md)**

:   本文从隐藏表示的角度系统分析 RAG 中检索文档如何影响 LLM 内部状态，发现了五个关键模式：随机文档引发大表示漂移并触发拒绝行为、相关文档主要确认而非改变参数化知识、单个相关文档能锚定多文档场景中的表示、后层逐步强调参数化知识从而限制检索证据的影响、以及 LLM 在早期层就能区分随机文档但到最后层仍无法可靠区分干扰文档和相关文档。

**[Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)**

:   HEAVEN 提出了一种即插即用的两阶段混合向量框架，通过视觉摘要页（VS-Pages）加速单向量粗检索 + 基于词性的查询 token 过滤减少多向量重排序计算，在四个基准上保持 99.87% 的多向量 Recall@1 同时减少 99.82% 的每查询 FLOPs。

**[Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](information_retrieval/is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)**

:   本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。

**[KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](information_retrieval/koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)**

:   KoCo-Bench 提出首个包含显式领域知识语料库的代码基准，覆盖 6 个新兴领域（RL、Agent、RAG 等）的 11 个框架和 25 个项目，评估 LLM 从知识语料库中获取和应用领域知识进行代码生成和知识理解的能力，揭示即使最强 coding agent Claude Code 也仅达 34.2%。

**[MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](information_retrieval/mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)**

:   提出 MAB-DQA 框架，将复杂查询分解为多个方面子查询，用多臂老虎机机制（Thompson Sampling）动态评估各方面的重要性并重新分配检索预算，显著提升多模态文档问答的检索精度和回答准确率。

**[MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](information_retrieval/mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)**

:   本文提出 MASS-RAG，一个免训练的多 Agent 综合 RAG 框架，通过 Summarizer/Extractor/Reasoner 三个专门化过滤 Agent 从互补视角处理检索文档，再通过 Synthesis Agent 整合多视角证据或候选答案，在四个基准上持续超越强基线。

**[Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)**

:   本文提出 MSPA-CQR，通过从改写、检索、回复三个维度构建自一致性偏好数据，并使用前缀引导的多维度 DPO 优化来训练查询改写模型，在分布内外场景均显著超越现有方法。

**[RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](information_retrieval/racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)**

:   RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。

**[ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)**

:   ReasonEmbed 提出三项技术创新——ReMixer 非平凡合成数据方法（82K 高质量样本）、Redapter 自适应推理强度加权训练和多骨干实现——在 BRIGHT 基准上以 38.1 的 nDCG@10 显著超越所有现有文本嵌入模型约 10 个点。

**[Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](information_retrieval/region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)**

:   本文提出 Region-R1，将多模态重排序中的查询图像区域裁剪建模为决策问题，通过强化学习（r-GRPO）学习何时以及如何裁剪查询图像中与问题相关的区域，在 E-VQA 和 InfoSeek 上将 CondRecall@1 分别提升 20% 和 8%。

**[RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](information_retrieval/reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)**

:   提出 RepoShapley，一种基于 Shapley 值的联盟感知上下文过滤框架，通过估计检索代码片段在组合中的交互贡献来决定保留/丢弃，显著提升仓库级代码补全质量。

**[Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)**

:   本文提出 Prune-then-Merge，一个两阶段的免训练多向量文档压缩框架——先通过自适应注意力剪枝移除低信息 patch，再对剩余高信号 patch 进行层次聚类合并，在 29 个 VDR 数据集上将近无损压缩范围从 50-60% 扩展到 60-70%，并在 80%+ 高压缩率下显著优于单阶段方法。

**[Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](information_retrieval/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)**

:   揭示 RAG 系统对检索文档排列顺序高度敏感的问题，提出 Stable-RAG：通过对文档排列产生的隐状态做谱聚类识别主导推理模式，再用 DPO 对齐将幻觉输出引导向正确答案，在三个 QA 数据集上实现准确率和推理一致性的双重提升。

**[TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](information_retrieval/taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)**

:   本文提出 TaxPraBen，首个面向中国税务实践的 LLM 评测基准，包含 14 个数据集共 7.3K 样本，覆盖税务风险防控、稽查分析和税务筹划三大真实场景，并设计了"结构化解析—字段对齐提取—数值与文本匹配"的可扩展评估范式，评测 19 个 LLM 后发现闭源大模型和中文优化模型表现更优，而税务领域微调模型 YaYi2 改进有限。

**[To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](information_retrieval/to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)**

:   本文提出 GlobalLies——一个包含 440 个虚假信息生成模板和 6,867 个实体的多语言平行数据集（8 种语言、195 个国家），揭示了 LLM 在传播虚假信息时存在系统性的国家和语言偏差：对低 HDI 国家的虚假信息生成率显著更高（统计相关 $\rho=-0.355$, $p=5\times10^{-7}$），低资源语言的合规率高出英语 30% 以上，且现有安全分类器和 RAG 防护措施提供不均匀的保护。

**[TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](information_retrieval/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)**

:   本文提出 TPA 框架，通过数学方法将 LLM 每个 token 的生成概率精确分解为七个来源（Query、RAG Context、Past Token、Self Token、FFN、Final LayerNorm、Initial Embedding）的贡献，结合词性标注聚合特征，实现 RAG 场景下的 SOTA 幻觉检测。

**[Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](information_retrieval/understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)**

:   本文提出 FinFRE-RAG，一种两阶段框架，通过重要性引导的特征降维将高维表格交易数据序列化为自然语言，并结合标签感知的检索增强上下文学习，使开源 LLM 在金融欺诈检测上的 F1/MCC 大幅提升，缩小了与专用表格分类器的性能差距。

**[VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)**

:   VideoStir 提出了一种结构化且意图感知的长视频 RAG 框架，通过将视频建模为时空图进行多跳 clip 检索 + 训练意图相关性评分器进行帧级筛选，在不依赖辅助文本工具的前提下达到了与 SOTA 长视频 RAG 方法可比的性能。

**[Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](information_retrieval/why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)**

:   提出 HyPE 框架，在生成式检索中通过先生成层级类别路径（如 "Government >> Government by cities"）再解码文档标识符，为检索结果提供查询相关的可解释路径，同时提升检索准确率。

---

## 📦 模型压缩 { #model_compression }

**[A Computational Method for Measuring "Open Codes" in Qualitative Analysis](model_compression/a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)**

:   提出一种基于理论的计算方法，通过LLM增强的代码合并算法和四个无需ground truth的指标（Coverage, Overlap, Novelty, Divergence），系统评估人类和AI在归纳定性编码中的表现。

**[A Layer-wise Analysis of Supervised Fine-Tuning](model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)**

:   通过信息论、几何和优化三个视角对 1B-32B 模型的 SFT 进行逐层分析，发现指令跟随能力集中在中间层（20%-80%），而非均匀分布，据此提出 Mid-Block Efficient Tuning 策略，选择性更新中间层，在 GSM8K 上比标准 LoRA 提升高达 10.2%。

**[Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)**

:   提出ASL（Adaptive Selection Layer），通过监控token注意力分数排名的方差来自适应确定KV缓存剪枝的层位置，在困难任务上显著优于固定层选择方法，同时保持无需训练。

**[Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](model_compression/analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)**

:   提出一种分析式后训练框架，通过神经元激活模式分析将dense FFN快速重构为sparse MoE——区分高频共享专家和低频路由专家，并从激活统计量构建路由器，仅需2k样本微调即可实现1.17×加速。

**[arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](model_compression/arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)**

:   提出 arXiv2Table 基准（1,957 张表、7,158 篇论文），通过引入干扰论文、模式无关的用户需求和基于 QA 的无标注评估框架，实现更真实的 LLM 文献综述表格生成评估，并提出迭代批处理生成方法。

**[Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)**

:   CSD 提出一种训练免的推测解码增强框架，通过在线校正记忆（OCM）记录高频拒绝模式提供救援候选，再用语义一致性门控（SCG）基于概率比验证候选可靠性，将推测解码的吞吐量提升至最高 2.33×，同时在 HumanEval 和 MATH500 上甚至提升了准确率。

**[CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](model_compression/cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)**

:   CBRS 提出一个多平台框架，通过双层过滤架构（轻量分类器 + LLM）从社交媒体消息流中高效检测并解析血液捐献请求，构建了首个包含 11K 条孟加拉语-英语-转写孟加拉语的血液捐献请求数据集，LoRA 微调的 Llama-3.2-3B 在解析任务上达到 92% 零样本准确率。

**[ChemAmp: Amplified Chemistry Tools via Composable Agents](model_compression/chemamp_amplified_chemistry_tools_via_composable_agents.md)**

:   提出"工具放大"新范式（区别于传统的工具编排），通过 ChemAmp 框架将化学专用工具（UniMol2、Chemformer等）作为可组合积木块动态构建任务专用超级智能体，在分子设计、反应预测等四个核心化学任务上超越专用模型和通用LLM，同时推理token成本减少94%。

**[CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](model_compression/clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)**

:   本文提出 CLAG，一种基于聚类的 Agent 记忆框架，通过 SLM 驱动的路由将记忆组织到语义一致的聚类中，在聚类内部进行局部进化更新，并通过两阶段检索过滤噪声，在多个 QA 数据集上显著优于全局记忆池基线。

**[Compositional Steering of Large Language Models with Steering Tokens](model_compression/compositional_steering_of_large_language_models_with_steering_tokens.md)**

:   本文提出组合引导 token，通过自蒸馏将行为指令压缩为输入空间的嵌入向量，并训练专用组合 token <and> 来捕获"组合"的通用概念，在未见过的行为组合、未见过的行为以及未见过的组合数量上均展现强泛化能力。

**[DeepPrune: Parallel Scaling without Inter-Trace Redundancy](model_compression/deepprune_parallel_scaling_without_inter-trace_redundancy.md)**

:   本文提出 DeepPrune，通过训练专门的判断模型从部分推理轨迹预测答案等价性，结合在线贪心聚类算法动态剪枝冗余的并行 CoT 路径，在保持竞争准确率（3 个百分点以内）的同时减少 65.73%-88.50% 的 token 消耗。

**[Efficient Learned Data Compression via Dual-Stream Feature Decoupling](model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)**

:   本文提出FADE框架，通过双流多尺度解耦器将微观句法和宏观语义特征分离到并行浅层流中处理（取代深层串行堆叠），结合层次化门控精炼器和并发流并行流水线，在压缩率和吞吐量上同时达到SOTA。

**[Enabling Agents to Communicate Entirely in Latent Space](model_compression/enabling_agents_to_communicate_entirely_in_latent_space.md)**

:   本文提出 Interlat，一个让 LLM 智能体完全在潜空间中通信的框架——发送方直接传递最后一层隐状态作为"思维"的表示，接收方通过通信适配器解释这些潜空间消息，并通过潜空间推理进一步压缩到仅 8 个 token 同时保持竞争性能，实现高达 24× 的通信加速。

**[Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings](model_compression/establishing_a_scale_for_kullback-leibler_divergence_in_language_models_across_v.md)**

:   本文利用对数似然向量将不同架构的语言模型嵌入统一空间，系统测量了预训练、模型规模、随机种子、量化、微调和层间等多种设置下的 KL 散度特征尺度，并发现预训练轨迹在对数似然空间中呈亚扩散行为——尽管权重空间持续漂移，模型输出分布早期即趋于稳定。

**[FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](model_compression/fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)**

:   本文提出 FastKV，通过将上下文缩减（prefill 阶段的 Token-Selective Propagation）与 KV 缓存压缩（decoding 阶段的层级 KV 保留）解耦，在 LLaMA-3.1-8B-Instruct 上实现 prefill 1.82× 和 decoding 2.87× 加速，同时在 LongBench 上精度下降控制在 1% 以内。

**[Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)**

:   提出 PerSyn（Personalized data Synthesis），通过"先路由再生成"范式让路由器为每个 prompt 分配最优教师模型，综合考虑学生可学习性和教师响应质量，比传统"先生成再选择"范式高效且效果更好，在指令微调和数学推理两个场景中一致超越所有基线。

**[MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](model_compression/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)**

:   本文将生成引擎优化（GEO）从逐实例启发式优化重构为策略学习问题，提出 MAGEO 多智能体框架——执行层由偏好/规划/编辑/评估四个智能体协作，学习层将验证有效的编辑模式蒸馏为可复用的引擎特定策略技能，并引入 Twin Branch 因果评估协议和 DSV-CF 双轴指标，在三个主流引擎上显著优于启发式基线。

**[From Weights to Activations: Is Steering the Next Frontier of Adaptation?](model_compression/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)**

:   本文系统性地论证 steering（推理时激活空间干预）应被视为一种独立的模型适配范式，提出八项功能性评估标准对比 steering 与微调、PEFT、提示工程等传统方法，将 steering 定位为基于激活空间的局部可逆行为修改方法，具有计算高效、数据高效和可逆性等独特优势。

**[HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)**

:   本文提出 HeteroCache，一种免训练的动态 KV 缓存压缩框架，基于注意力头的时间异质性（稳定头 vs 漂移头）和层内冗余性（相似头聚类），实施细粒度的角色分配策略——为漂移头分配更大缓存预算，用代表头稀疏监控注意力漂移触发异步按需检索，在 224K 上下文下实现 3 倍解码加速。

**[IMPACT: Importance-Aware Activation Space Reconstruction](model_compression/impact_importance-aware_activation_space_reconstruction.md)**

:   提出 IMPACT 框架，将 LLM 低秩压缩从最小化权重重构误差转向最小化重要性加权的激活重构误差，通过将梯度信息融入激活协方差矩阵推导出闭式最优解，实现在保持精度的同时最高减少 55.4% 的模型体积。

**[CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)**

:   提出 CadLLM，一种免训练的自适应推理加速方法，利用扩散语言模型（dLLM）的 token 解码置信度信号动态调整块大小、步数、词表采样范围和提交阈值四个维度，在 LLaDA 和 DREAM 上实现 1.1-2.28× 的吞吐量提升且保持竞争性准确率。

**[JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](model_compression/judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)**

:   提出了一个 synthetic-organic 监督管线，将法官的原始判决文书转化为推理指令微调数据，通过 CLM→指令微调的 Chain-of-LoRA 策略实现对个体法官推理风格的高保真模拟，在希伯来语低资源场景下生成内容与真实法官不可区分。

**[LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](model_compression/llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)**

:   将无标签提示优化形式化为决斗老虎机（dueling bandit）问题，提出 Prompt Duel Optimizer (PDO)，通过 Double Thompson Sampling 高效选择信息量最大的提示对进行比较，结合 top-performer 变异策略扩展搜索空间，在 BBH 和 MS MARCO 上以更少的 judge 调用次数找到更强提示。

**[LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)**

:   提出 LoGo（LoRA on the Go），一个免训练的框架，通过单次前向传播提取 LoRA 激活信号（范数或熵），在实例级别动态选择和合并最相关的 LoRA 适配器，无需标注数据或额外训练即可实现跨任务泛化。

**[MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](model_compression/maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)**

:   本文提出 MAESTRO，将 GRPO 中的奖励标量化重新定义为上下文老虎机问题，通过轻量级 Conductor 网络利用模型末层隐藏状态自适应地为每个 prompt-response 对选择奖励权重，在七个开放域基准上一致超越静态奖励和单一奖励基线。

**[Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](model_compression/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)**

:   本文提出 Mem²Evolve，一种通过双记忆机制（资产记忆 + 经验记忆）实现能力扩展与经验蒸馏协同进化的自进化 Agent 框架，在 6 类任务 8 个基准上平均 Pass@1 达 70.24%，分别超过纯经验进化和纯能力进化的最强基线 11.80% 和 6.46%。

**[Mem^p: Exploring Agent Procedural Memory](model_compression/memp_exploring_agent_procedural_memory.md)**

:   本文提出 Mem^p 框架，系统性地研究如何为 LLM Agent 构建可学习、可更新、终身演化的程序性记忆——通过将过去的任务轨迹蒸馏为细粒度的分步指令和高层脚本抽象，并配合动态更新机制（添加/验证/反思/淘汰），在 TravelPlanner 和 ALFWorld 上实现了成功率持续提升和执行步数大幅减少。

**[Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](model_compression/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)**

:   通过在四个基准上系统对比超网络 LoRA 适应 vs 精心设计的 few-shot 提示，发现 2.28 亿参数的超网络提供零增益——few-shot 示例贡献 +21.5%、文档编码贡献 +5.0%、超网络贡献 0%，3B 模型配合良好提示可达 GPT-5 平均性能的 79.7% 且延迟低 10 倍。

**[Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](model_compression/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)**

:   本文系统地对 25 个 Transformer 语言模型（从 BERT Base 到 Qwen2.5-7B）进行探针分析，发现词汇同一性（lexeme）在早期层线性可解码但随深度衰减，而屈折特征（inflection）在所有层中保持稳定可读，且占据紧凑可控的子空间。

**[No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](model_compression/no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)**

:   提出 NWCAD，一种解码时适配器，通过两阶段门控机制在上下文无信息时精确回退到无上下文解码（防止中性退化），在上下文有帮助时利用上下文进行修正，兼顾"无害"与"有效"两个目标。

**[Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)**

:   本文提出 PERA（Polynomial Expansion Rank Adaptation），通过在低秩因子的参数空间中引入结构化多项式展开（平方项和交叉项），将 LoRA 的线性适配空间扩展为多项式流形，在不增加秩或推理开销的前提下显著提升权重更新的表达能力，在常识推理和 NLU 任务上一致优于 LoRA/DoRA/HiRA 等方法。

**[Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](model_compression/reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)**

:   提出 E-GRM 框架，利用模型并行解码的收敛行为估计不确定性，仅在必要时触发 CoT 推理，并通过混合损失训练的判别式评分器精细评估推理路径质量，在多个奖励模型基准上实现 SOTA 同时降低 62% 推理延迟。

**[SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)**

:   SAMoRA 通过语义感知路由器和任务自适应缩放机制，解决了现有 MoE-LoRA 方法中路由不精确和权重融合缺乏灵活性的问题，在多任务基准上以最少可训练参数（0.15%）达到 SOTA。

**[SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](model_compression/scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)**

:   本文提出 SCURank，一种基于摘要内容单元（SCU）的排序框架，通过提取 SCU、跨摘要聚类估计信息重要性、按信息丰富度评分来排序候选摘要，替代不稳定的 LLM 直接排序和粗粒度的 ROUGE 排序，在多 LLM 蒸馏场景中配合 BRIO 对比学习显著提升了蒸馏模型的摘要性能。

**[SeLaR: Selective Latent Reasoning in Large Language Models](model_compression/selar_selective_latent_reasoning_in_large_language_models.md)**

:   本文提出 SeLaR，一种轻量级无训练框架，通过熵门控机制仅在模型不确定的"探索步"激活软嵌入潜在推理、在高置信的"确定步"保持离散解码，并引入熵感知对比正则化防止软嵌入向主导 token 坍缩，在五个推理基准上一致超越标准 CoT 和 SOTA 无训练方法。

**[Supplement Generation Training for Enhancing Agentic Task Performance](model_compression/supplement_generation_training_for_enhancing_agentic_task_performance.md)**

:   SGT（Supplement Generation Training）训练一个小型 LLM（1.7B）生成逐实例的补充文本（推理线索、摘要、错误提醒等），附加到输入后让冻结的大型 Actor 模型更有效地解决任务，在 5 个基准上平均提升 21%，无需修改大模型参数。

**[Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)**

:   本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。

**[Training-Free Test-Time Contrastive Learning for Large Language Models](model_compression/training-free_test-time_contrastive_learning_for_large_language_models.md)**

:   本文提出 TF-TTCL，一种无需梯度更新的测试时对比学习框架，通过"探索-反思-引导"循环让冻结的 LLM 在线自我改进——用多智能体角色扮演生成多样推理轨迹，从正负样本对比中蒸馏文本规则存入记忆库，推理时检索相关规则引导生成。

**[UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text](model_compression/ukp_psycontrol_at_semeval-2026_task_2_modeling_valence_and_arousal_dynamics_from.md)**

:   UKP_Psycontrol 在 SemEval-2026 Task 2 上取得双项第一，通过结合 LLM 提示、Ising 交互的 MaxEnt 模型和神经回归模型，发现 LLM 擅长捕捉静态情感信号而短期情感变化更多由近期数值轨迹而非文本语义解释。

**[Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](model_compression/which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)**

:   提出 Rank-Surprisal Ratio (RSR) 指标，通过联合衡量推理轨迹对学生模型的"信息量"和"对齐度"来评估训练数据适配性，在 5 个学生模型和 11 个教师模型的组合中与训练后性能达到平均 0.86 的 Spearman 相关性，并成功应用于轨迹选择和教师选择。

**[WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)**

:   本文提出等价模型理论和 WISCA 权重缩放策略，通过在训练中动态调整 Transformer 注意力层的 $W_q/W_k$ 和 $W_v/W_o$ 权重使其 L1 范数相等（保持模型输出不变），将优化引导至更平坦的损失最小值区域，在 GQA 架构上实现平均 5.6% 的零样本评估提升和 2.12% 的训练困惑度降低。

**[YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](model_compression/yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)**

:   提出信息引出代理（IEA）作为新的对话范式，发布了首个大规模（2,281 段对话，26M token）人与人信息引出对话数据集 YIELD，将信息引出形式化为有限视野 POMDP，并设计了专门的评估指标（Conformity、Progression、TLR），实验表明在 YIELD 上微调能显著提升 LLM 与真实引出行为的对齐。

---

## 🦾 LLM Agent { #llm_agent }

**[AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)**

:   提出AgencyBench——一个包含138个真实世界任务的综合基准，评估6种核心智能体能力，每个场景平均需90次工具调用和100万token，通过用户模拟agent和Docker沙箱实现全自动化评估。

**[Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)**

:   本文提出 Agent-GWO，将灰狼优化器的领导者-追随者机制引入多智能体框架，联合优化 prompt 模板和解码超参数（温度、top-p 等），在 11 个数学和混合推理基准上持续超越现有提示优化方法。

**[ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)**

:   提出 ATLAS 多智能体金融交易框架和 Adaptive-OPRO 提示优化方法，通过专业化分析师智能体准备异构市场信息，并基于延迟噪声反馈动态优化中央交易智能体的指令提示，在多种市场波动环境中显著超越基线。

**[Bayesian Social Deduction with Graph-Informed Language Models](llm_agent/bayesian_social_deduction_with_graph-informed_language_models.md)**

:   提出 GRAIL（Graph Reasoning Agent Informed through Language），一个混合推理框架，将概率推理外化到因子图模型、用 LLM 处理语言理解和交互，在社交推理游戏 Avalon 中首次击败人类玩家（67% 胜率），且资源消耗远低于大规模推理模型。

**[CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](llm_agent/ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)**

:   基于上下文完整性（Contextual Integrity）理论构建企业场景基准 CI-Work，揭示前沿 LLM 智能体在企业工作流中普遍存在隐私泄漏问题，且模型规模扩大反而加剧泄漏。

**[CodeStruct: Code Agents over Structured Action Spaces](llm_agent/codestruct_code_agents_over_structured_action_spaces.md)**

:   本文提出CodeStruct框架，将代码仓库重新定义为基于AST的结构化动作空间，让LLM代码Agent通过命名的程序实体（而非文本片段）进行读取和编辑操作，在SWE-Bench Verified上提升1.2-5.0%准确率并减少12-38% token消耗。

**[CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](llm_agent/coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)**

:   CoEvolve 提出**智能体-数据共进化框架**，通过从训练轨迹中提取遗忘/边界/稀有三类弱点信号，引导 LLM 做针对性环境再探索和任务合成，使训练数据分布随智能体能力动态适应，在 AppWorld 和 BFCL 上分别带来 19-23% 的绝对提升。

**[Conjunctive Prompt Attacks in Multi-Agent LLM Systems](llm_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)**

:   本文研究多智能体 LLM 系统中的联合提示攻击（conjunctive prompt attacks）：用户查询中嵌入的触发键和被入侵远程代理中的隐藏模板各自看起来无害，但当路由将它们带到同一代理时会激活有害行为，现有防御（PromptGuard、Llama-Guard 等）均无法可靠阻止。

**[Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](llm_agent/creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)**

:   本文提出 IASC（Interactive Agentic System for ConLangs），一个模块化的人造语言构建系统，通过让 LLM 按语言学规格执行形态句法变换来探测其元语言知识，发现 LLM 处理常见语言类型模式远优于罕见模式，且不同 LLM 之间能力差异悬殊。

**[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](llm_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)**

:   本文通过评估超过 10,000 个研究提案，从模型智能、智能体认知和系统动力学三个层次系统揭示了多智能体 LLM 系统中的"多样性崩溃"现象：更强的模型、权威驱动的角色分配和密集的通信拓扑都会抑制语义多样性，根本原因是交互结构而非模型能力不足。

**[EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](llm_agent/ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)**

:   提出 EA-Agent，将实体对齐（EA）分解为结构化多步推理过程，通过工具池（三元组选择器+对齐工具+反思器）的规划和执行实现可解释的对齐决策，配合奖励引导的离线策略优化持续改进规划能力，在 DBP15K 上 Hits@1 提升高达 3.17%，同时减少冗余三元组带来的效率问题。

**[ExpSeek: Self-Triggered Experience Seeking for Web Agents](llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)**

:   ExpSeek 提出了一种基于步级熵自触发的经验主动寻求框架，让 Web Agent 在交互过程中根据自身信号判断何时需要指导、获取什么指导，在 Qwen3-8B/32B 上分别实现 9.3% 和 7.5% 的绝对提升。

**[FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](llm_agent/fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)**

:   提出 FairQE 多智能体框架，通过性别线索检测、性别翻转变体生成和动态偏见感知分数聚合机制，在不牺牲翻译质量评估准确性的前提下有效缓解 QE 模型中的系统性性别偏见。

**[FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](llm_agent/fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)**

:   FedGUI 是首个面向跨平台 GUI 代理的联邦学习综合基准，包含六个数据集覆盖移动端/网页端/桌面端，系统研究跨平台、跨设备、跨操作系统和跨数据源四种异构性对联邦 GUI 代理训练的影响。

**[FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction](llm_agent/fregelogic_at_semeval_2026_task_11_a_hybrid_neuro-symbolic_architecture_for_cont.md)**

:   提出 FregeLogic 混合神经符号系统，结合五成员 LLM 集成和 Z3 SMT 求解器作为决胜裁判，在三段论有效性判断中将内容效应降低16%的同时提升准确率0.9%。

**[From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](llm_agent/from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)**

:   本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。

**[HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](llm_agent/hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)**

:   提出 HAG 框架，将群体 Agent 生成形式化为两阶段层次化决策过程——先用世界知识模型构建主题自适应人口分布树实现宏观分布对齐，再通过真实数据检索与 Agent 增强保证微观个体一致性，在多领域基准上将群体对齐误差平均降低 37.7%、社会学一致性提升 18.8%。

**[Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)**

:   本文提出 STEP-HRL，通过引入局部进度模块将交互历史迭代压缩为紧凑的文本摘要，使高层和低层策略仅基于单步转移（而非完整历史）做决策，在 ScienceWorld 和 ALFWorld 上显著提升性能和泛化性，同时减少 token 使用。

**[How Adversarial Environments Mislead Agentic AI](llm_agent/how_adversarial_environments_mislead_agentic_ai.md)**

:   本文形式化了"对抗环境注入"（AEI）威胁模型，将其分解为广度攻击（投毒检索结果导致认知漂移）和深度攻击（注入幻影节点构造导航陷阱导致策略崩溃），在 11,000+ 次实验中发现两种攻击的鲁棒性完全独立——"鲁棒性分裂"表明当前单点防御策略根本不够。

**[ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)**

:   提出 ImplicitMemBench，首个系统评估 LLM 隐式记忆的基准，包含程序性记忆、启动效应和经典条件反射三种认知范式共 300 个测试项，在 17 个模型上揭示严重局限：最优模型仅达 66% 整体准确率，远低于人类基线。

**[Lightweight LLM Agent Memory with Small Language Models](llm_agent/lightweight_llm_agent_memory_with_small_language_models.md)**

:   本文提出 LightMem，一种由多个专用小语言模型（SLM）驱动的轻量级 LLM 智能体记忆系统，通过将记忆操作模块化为控制器（SLM-1）、选择器（SLM-2）和写入器（SLM-3），并将在线处理与离线整合解耦，在 LoCoMo 基准上平均 F1 提升约 2.5（相比 A-MEM），同时实现 83ms 检索延迟和 581ms 端到端延迟。

**[LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)**

:   本文提出 Location Preference Optimization (LPO)，通过基于信息熵的窗口奖励和基于物理距离的动态位置奖励，结合 GRPO 框架优化 GUI 智能体的空间定位精度，在离线和在线评估中均达到 SOTA。

**[MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](llm_agent/mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)**

:   提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。

**[MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](llm_agent/mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)**

:   MCP-Flow 提出了一个基于 Web Agent 的自动化管道，从 1166 个真实 MCP 服务器中收集工具信息并合成 68733 条高质量训练数据，使小规模微调模型（0.6B-8B）在 MCP 工具使用上超越 GPT-4o 等 SOTA 大模型。

**[MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](llm_agent/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)**

:   提出 MemoPhishAgent（MPA），首个专为钓鱼URL检测设计的记忆增强多模态LLM智能体，通过5个专用工具的动态编排和情景记忆系统复用历史推理轨迹，在公开基准上召回率提升13.6%，在真实社交媒体数据上提升20%，并已部署生产环境每周处理约6万高风险URL。

**[Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](llm_agent/mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)**

:   开发 Mina——面向孟加拉国法律场景的多语言 LLM 法律助手，通过两阶段 RAG 流水线精准检索法案和条款，配合工具链和多语言嵌入，在孟加拉律师资格考试中取得 75-80% 的通过成绩，法律咨询成本仅为传统方式的 0.12-0.61%。

**[RISK: A Framework for GUI Agents in E-commerce Risk Management](llm_agent/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)**

:   提出 RISK 框架，包含领域数据集（RISK-Data, 8492单步+2386多步轨迹）、基准（RISK-Bench）和基于GRPO的强化微调方法（RISK-R1），针对电商风控场景的GUI智能体，7B模型以仅7.2%的参数量超越SOTA基线，在线任务成功率达70.5%。

**[Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](llm_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)**

:   提出 ExtAgents 多智能体框架，通过全局知识同步（所有Seeking Agent间交换信息）和知识累积推理（逐步向Reasoning Agent注入筛选后的知识）两个机制，解决现有多智能体方法在扩展外部知识输入超出上下文窗口时性能不升反降的瓶颈，在多跳QA和长综述生成任务上显著提升。

**[SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](llm_agent/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)**

:   提出 SecureVibeBench，首个仓库级多文件编辑的安全编码基准，从41个OSS-Fuzz项目中构建105个C/C++安全编码任务，通过级联静态+动态分析精确还原漏洞首次引入的场景，评估发现最佳Agent（SWE-agent + Claude Sonnet 4.5）仅23.8%的代码同时满足功能正确性和安全性。

**[SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](llm_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)**

:   本文提出 SILO-BENCH，一个角色无关的多智能体 LLM 分布式协调基准，包含 30 个算法任务、三个通信复杂度级别、54 种配置共 1620 个实验，揭示了关键的"通信-推理鸿沟"：智能体能自发形成合理通信拓扑并积极交换信息，但系统性地无法将分布式状态整合为正确答案。

**[Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification](llm_agent/spec-o3_a_tool-augmented_vision-language_agent_for_rare_celestial_object_candida.md)**

:   提出 Spec-o3，一个工具增强的视觉语言智能体，通过交错多模态思维链（iMCoT）模拟天文学家的光谱检查流程，采用冷启动 SFT + 基于结果的 RL 两阶段训练，在稀有天体识别上将 macro-F1 从 28.3% 提升至 76.5%，推理速度比人工检查快 ~50 倍。

**[StructMem: Structured Memory for Long-Horizon Behavior in LLMs](llm_agent/structmem_structured_memory_for_long-horizon_behavior_in_llms.md)**

:   StructMem 提出了一种结构增强的层次化记忆框架，通过事件级双视角提取和跨事件语义整合，在 LoCoMo 长对话基准上实现 SOTA 性能（76.82%），同时大幅降低 token 消耗（1.94M vs. 图记忆的 35.8M）和 API 调用次数。

**[SynthAgent: Adapting Web Agents with Synthetic Supervision](llm_agent/synthagent_adapting_web_agents_with_synthetic_supervision.md)**

:   本文提出 SynthAgent，一个完全基于合成监督的 Web Agent 适应框架，通过分类探索系统覆盖网页功能区域以合成多样化任务，再通过任务精炼（冲突检测触发修正幻觉）和轨迹精炼（全局视角去噪）的双重精炼策略提升合成数据质量，在 WebArena 和 Online-Mind2Web 上显著优于现有合成方法。

**[ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](llm_agent/toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)**

:   本文提出 ToolOmni，一个统一的智能体框架，将主动工具检索和基于检索结果的工具执行整合在同一推理循环中，通过冷启动 SFT + 解耦多目标 GRPO 联合优化检索和执行能力，在 ToolBench 上端到端执行成功率超过强基线 +10.8%。

**[Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)**

:   本文提出 LAMO 框架，通过角色导向的数据合成和两阶段训练（SFT with Perplexity-Weighted Cross-Entropy + 多任务 RL），将轻量 3B MLLM 训练为可灵活编排多角色的 GUI Agent，在单体推理、多 Agent 协作和即插即用策略执行器三种模式下工作，搭配 GPT-5 规划器在 AndroidWorld 上达 77.6% 成功率，超越 72B 参数的专用 GUI Agent。

**[Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities](llm_agent/uncertainty_quantification_in_llm_agents_foundations_emerging_challenges_and_opp.md)**

:   本文提出首个 Agent 不确定性量化（Agent UQ）的形式化框架：将 agent 的问题解决轨迹建模为动态贝叶斯网络上的随机过程 $P(\mathcal{F}_{\leq T}) = P(E_0, O_0) \prod_{i=1}^{T} P_{\pi,\mathcal{T}}(A_i|E_{i-1}, O_{i-1}) P(O_i|A_i, E_i)$，统一了现有 UQ 范式（单步 QA、多步推理）为特例，并通过 $\tau^2$-bench 上的实证分析识别了四个 agent UQ 特有的技术挑战。

**[Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](llm_agent/waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)**

:   本文提出 SPECTRA，一种无需监督轨迹的框架——通过冷启动强化学习（GRPO）和软结构化多轮 rollout 拓扑约束，让小型视觉语言模型（SVLM）在纯环境交互中自行发现有效的工具调用和视觉推理行为，在 4 个多模态 benchmark 上提升任务准确率达 5% 和工具效率 9%，同时提出 Tool Instrumental Utility（TIU）指标量化无监督下的工具效能。

**[What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](llm_agent/what_makes_an_llm_a_good_optimizer_a_trajectory_analysis_of_llm-guided_evolution.md)**

:   本文通过大规模实验（15 个 LLM × 8 个任务、72K 候选解）发现优秀的 LLM 优化器表现为"局部精炼器"——持续产生频繁的渐进式改进并在语义空间中逐步集中搜索，而非产生高新颖性的跳跃式突破；关键发现是新颖性本身并不预测优化性能，只有当搜索保持足够局部化时新颖性才有益。

**[When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](llm_agent/when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)**

:   本文提出了 RPS 和 AGS 两个互补指标来量化 LLM Agent 在工具使用行为上的蒸馏导致的同质化现象，通过区分必要行为和非必要行为，在 18 个模型上揭示了跨家族行为继承模式，发现 Kimi-K2 与 Claude Sonnet 4.5 的行为相似度甚至超过 Anthropic 自家模型。

**[Why Agents Compromise Safety Under Pressure](llm_agent/why_agents_compromise_safety_under_pressure.md)**

:   提出"代理压力"（Agentic Pressure）概念——当 LLM 代理在资源约束下无法同时完成任务和遵守安全规则时，会自发地产生规范漂移，主动牺牲安全以保持有用性，且推理能力越强的模型越善于构建语言化合理化来为违规辩护。

**[ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](llm_agent/zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)**

:   提出 ZARA，一个基于知识和检索增强的多智能体框架，通过将传感器信号蒸馏为结构化文本知识库、类别条件检索和分层 LLM 推理，在完全免训练的设置下实现了可解释的人体活动识别，8 个数据集上大幅超越现有方法。

---

## 📊 LLM 评测 { #llm_evaluation }

**[Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL](llm_evaluation/abstain-r1_calibrated_abstention_and_post-refusal_clarification_via_verifiable_r.md)**

:   Abstain-R1 提出一种**澄清感知的 RLVR 奖励**，在不可回答查询上联合优化"明确拒答"和"拒答后给出有用澄清（指出缺失信息）"，使 3B 模型在拒答和澄清质量上接近甚至超越 DeepSeek-R1 等大模型。

**[AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](llm_evaluation/anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)**

:   提出AnchorMem记忆框架，受普鲁斯特现象启发，将检索单元（原子事实）与生成上下文（原始交互）解耦，通过关联事件图连接碎片化记忆，在LoCoMo基准上大幅超越A-Mem、Mem0等现有记忆系统。

**[Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](llm_evaluation/are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)**

:   本文提出 SCRIPTS 基准，包含 1.1K 英语和韩语电影对话，通过三层概率标签（HIGHLY LIKELY / LESS LIKELY / UNLIKELY）评估 9 个 LLM 的社会关系推理能力，发现模型在英语上准确率仅 75-80%、韩语 58-69%，且 CoT 和思维模型对社会推理几乎无帮助。

**[AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](llm_evaluation/autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)**

:   AutoReproduce 提出了一个多智能体框架，通过"论文谱系"算法从引用文献中挖掘隐式领域知识，实现端到端的论文实验自动复现，在自建基准 ReproduceBench 上的代码执行率达 94.87%，性能差距仅 19.72%。

**[Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](llm_evaluation/beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)**

:   提出配对任务框架联合评估 LLM 的文学文本理解能力和翻译创造力，基于 11 本英文经典小说对 23 个模型进行大规模测评，发现强理解力并不能转化为人类水平的翻译创造力。

**[BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](llm_evaluation/bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)**

:   本文提出 BizCompass，一个连接理论基础与实际应用的商业推理基准，覆盖金融/经济/统计/运营四个知识域和分析师/交易员/顾问三个应用角色，系统评估了开源与闭源 LLM 的商业推理能力，揭示理论知识向实际表现转化的规律。

**[Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](llm_evaluation/capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)**

:   本文提出了一个三步评估框架（计算特征提取 + LLM-as-Judge + 人类专家验证）来系统评估六种 LLM 在唐诗生成上的能力，发现了关键的"回声室"效应：LLM 系统性地高估模仿统计模式但违反格律规则的机器生成诗歌，与人类专家判断显著偏离。

**[CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](llm_evaluation/cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)**

:   提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。

**[Closing the Modality Reasoning Gap for Speech Large Language Models](llm_evaluation/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)**

:   本文提出 TARS（Trajectory Alignment for Reasoning in Speech），一个基于强化学习的框架，通过表示对齐和行为对齐两种密集奖励信号，将语音条件下的推理轨迹与文本条件下的推理轨迹对齐，在 7B 规模模型中达到 SOTA，MRR（模态恢复率）接近甚至超过 100%。

**[Common to Whom? Regional Cultural Commonsense and LLM Bias in India](llm_evaluation/common_to_whom_regional_cultural_commonsense_and_llm_bias_in_india.md)**

:   本文构建 Indica，首个评估 LLM 次国家级文化常识的基准，聚焦印度五大区域在八个日常生活领域的文化差异，发现仅 39.4% 的问题在全部五个区域达成共识，且所有 LLM 均表现出地理偏见——过度选择中部和北部印度作为"默认"文化代表。

**[DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](llm_evaluation/diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)**

:   DiZiNER 通过模拟人工标注中的"预标注"流程，利用多个异构 LLM 作为标注员、一个监督 LLM 分析模型间分歧并迭代优化任务指令，在18个NER基准上实现了14个数据集的零样本SOTA，平均提升+8.0 F1，且超越了作为监督者的GPT-5 mini。

**[Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](llm_evaluation/do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)**

:   本文提出 LLMThinkBench，一个系统性评估 LLM 基础数学推理效率的基准，引入 Overthinking Score（准确率和 token 效率的调和平均），通过动态生成的 14 个确定性数学任务评估 53 个 LLM，发现推理模型平均生成约 18× 更多 token 但有时准确率更低，且扩展推理预算呈现收益递减。

**[E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](llm_evaluation/e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)**

:   本文提出 E2EDev，一个基于行为驱动开发（BDD）原则的端到端软件开发基准，包含 46 个真实 Web 项目、244 条细粒度需求和 703 个可执行 BDD 测试，评估发现即使最强 LLM（Claude 系列）在需求准确率上也不超过 60%，多智能体框架的复杂交互成本与性能收益不成正比。

**[Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](llm_evaluation/enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)**

:   L2T 提出了一种预训练框架，将 14 种语言学习任务（字符级→篇章级）与标准 next-token prediction 混合训练，在 500M/1B 参数规模上将 BLiMP 语言能力得分提升 2-3 个百分点并加速其习得过程，同时保持通用推理性能。

**[Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](llm_evaluation/exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)**

:   本文将中文互联网亚文化语言"抽象话"引入 NLP 社区，构建首个评估基准 Mouse（含翻译、表征分类、意图识别、毒性检测、含义选择、完形填空六个任务），发现 SOTA LLM 在上下文语义理解上表现尚可但在其他任务上存在明显局限。

**[Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](llm_evaluation/finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)**

:   本文提出 Finch（FinWorkBench），一个从真实企业环境（Enron 数据集等）构建的金融会计工作流基准，包含 172 个复合工作流和 1,710 个电子表格（2700 万单元格），即使最强的 GPT 5.1 Pro 花费平均 16.8 分钟也仅通过 38.4% 的工作流，揭示了前沿 AI Agent 在真实企业场景中的严重不足。

**[From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](llm_evaluation/from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)**

:   本文形式化定义了领域级和实例级两种 LLM 遗忘粒度，提出 BiForget 框架——利用目标模型自身（而非外部强模型）通过种子引导合成和对抗探测两阶段生成高质量遗忘数据集，在 Harry Potter 领域将相关性提升约 20、多样性提升约 0.05 同时数据量减半。

**[HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](llm_evaluation/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)**

:   本文提出 HiGMem，一个两层事件-对话轮记忆系统，通过让 LLM 先浏览事件摘要再预测哪些细粒度对话轮值得读取，在 LoCoMo10 基准上以少一个数量级的检索量达到了五类问题中四类的最优 F1。

**[Idiom Understanding as a Tool to Measure the Dialect Gap](llm_evaluation/idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)**

:   提出三个新的法语习语理解基准数据集（魁北克法语 QFrCoRE/QFrCoRT 和标准法语 MFrCoE），在 111 个 LLM 上评估发现 65.77% 的模型在方言习语上表现显著差于标准法语，量化了方言差距现象。

**[Language Model as Planner and Formalizer under Constraints](llm_evaluation/language_model_as_planner_and_formalizer_under_constraints.md)**

:   本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。

**[LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](llm_evaluation/lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)**

:   构建了首个中国民事法律关系的结构化分类体系（9 大领域、265 种关系类型），并基于此提出 LexRel 基准（1,140 个专家标注样本），评估了主流 LLM 在法律关系抽取任务上的能力，发现当前模型在该任务上存在显著局限，同时证明了法律关系信息对下游法律 AI 任务的增益效果。

**[MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](llm_evaluation/made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)**

:   本文提出 MADE——一个基于 FDA 医疗设备不良事件报告的"活"多标签文本分类基准，包含 1,154 个层次化标签和严格的时间分割，系统评估了 20+ 编码器/解码器模型在判别式微调、生成式微调和 few-shot 提示下的预测性能和不确定性量化（UQ）能力，揭示了关键权衡：小型判别式微调解码器在头到尾准确率上最优，生成式微调的 UQ 最可靠，大型推理模型提升稀有标签但 UQ 意外较弱。

**[Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](llm_evaluation/min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)**

:   Min-k Sampling 通过分析排序 logit 分布的局部结构来检测"语义悬崖"（高置信候选与低质量尾部噪声的分界点），实现了严格的温度不变性截断，在极端温度下仍保持稳健的推理和创意写作质量。

**[Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](llm_evaluation/modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)**

:   本文发现 LLM 在联合预测情感-思维风格-立场-意图四个认知维度时准确率暴跌至 5.7%（"认知拥挤"效应），通过 Gromov δ-hyperbolicity 分析证明认知状态具有层次结构，提出 HyCoLLM 框架在双曲空间中建模认知状态，8B 模型超越 GPT-4o。

**[MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](llm_evaluation/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)**

:   提出 MTR-DuplexBench，一个针对全双工语音语言模型（FD-SLM）的多轮综合评估基准，通过创新的轮次分割方法解决了全双工对话中轮次边界模糊和上下文不一致的挑战，涵盖对话特性、对话质量、指令遵循和安全性四个维度，实验揭示了现有 FD-SLM 在多轮交互中性能持续衰退的问题。

**[MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](llm_evaluation/multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)**

:   提出 MultiFileTest，首个多文件级别 LLM 单元测试生成基准，覆盖 Python/Java/JavaScript 各 20 个项目，评估 11 个前沿 LLM 并分析手动修复和自修复机制对测试质量的影响，揭示即使最强模型也存在大量基础可执行性错误。

**[ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](llm_evaluation/odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)**

:   本文提出 ODUTQA-MDC 任务和基准，首次系统研究开放域场景下用户查询模糊性的检测与多轮对话澄清问题，构建了包含 25,105 个 QA 对的大规模数据集，并设计了 MAIC-TQA 多智能体框架来完成"检测-澄清-推理"的端到端表格问答。

**[PIArena: A Platform for Prompt Injection Evaluation](llm_evaluation/piarena_a_platform_for_prompt_injection_evaluation.md)**

:   本文提出 PIArena，一个统一且可扩展的提示注入（Prompt Injection）评估平台，集成了多种 SOTA 攻击和防御方法，支持即插即用评估，并设计了基于策略的自适应攻击方法，系统性地揭示了现有防御在泛化性、自适应攻击和任务对齐场景下的关键局限。

**[ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)**

:   提出 ResearchBench，首个大规模评估LLM科学发现能力的基准，基于"灵感驱动假设生成"的理论分解，覆盖12个学科1386篇论文，将科学发现分解为灵感检索、假设组合、假设排序三个充分子任务，发现LLM在跨学科灵感检索上表现出色。

**[Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)**

:   本文重新定义会议效率评估——提出"目标达成率/时间成本"的客观标准和时序细粒度评估范式，构建了包含 130 场会议 2,459 个标注片段的 AMI-ME 数据集，并开发了基于 LLM 的自动评估框架，在 Spearman 相关系数上达到 0.64。

**[ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)**

:   本文提出 ReTraceQA，首个面向常识推理任务的推理过程评测基准，包含 2421 条由专家标注的步骤级错误定位和错误分类标注，揭示 14-24% 的 SLM 虽给出正确答案但推理过程有误，当采用推理感知评估替代仅答案评估时，SLM 性能最多下降 25 个百分点。

**[Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](llm_evaluation/revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)**

:   本文将心理语言学中的信息密度均匀性（UID）假说引入 LLM 推理分析，提出基于熵的步级信息密度度量框架，发现高质量推理轨迹呈现"局部均匀 + 全局非均匀"的反直觉模式，并证明该模式在 Best-of-N 采样中显著优于传统置信度/熵基线。

**[RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)**

:   RoleConflictBench 通过构建 13,914 个角色冲突场景，利用情境紧迫性作为客观约束来评估 LLM 的上下文敏感性，揭示了模型决策被静态角色偏好主导而非响应动态情境线索的严重问题。

**[SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)**

:   本文构建 SciImpact——首个跨 19 个学科领域、涵盖 7 个影响力维度（引用、奖项、专利、媒体、代码、数据集、模型）的大规模科学影响力预测基准，包含 215,928 个对比论文对，通过多任务微调使 4B 模型超越 o4-mini 等大模型。

**[Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness](llm_evaluation/self-awareness_before_action_mitigating_logical_inertia_via_proactive_cognitive_.md)**

:   本文提出 SABA 推理框架，通过"先感知再行动"的范式，在做出最终决策前显式构建和审计知识状态——利用信息融合 (IF) 将叙事整合为可验证的基线状态，再通过查询驱动的结构化推理 (QSR) 递归识别和解决缺失前提——在侦探推理和通用推理基准上均取得最佳表现。

**[SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](llm_evaluation/sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)**

:   本文提出 SessionIntentBench，一个评估 L(V)LM 理解电商购物会话中跨步骤意图漂移能力的多任务基准，包含四个递进式子任务（意图购买似然估计、属性正则化、意图验证对比、意图演化建模），构建了 190 万条意图条目和 113 万条意图轨迹，实验表明当前 20+ 个 L(V)LM 在捕获复杂会话意图方面表现不佳。

**[Subject-level Inference for Realistic Text Anonymization Evaluation](llm_evaluation/subject-level_inference_for_realistic_text_anonymization_evaluation.md)**

:   SPIA 提出首个主体级 PII 推断评估基准（675 篇文档、1712 个主体、7040 个 PII），揭示即使 90%+ 的 PII 片段被遮蔽，主体级推断保护率可低至 33%，且聚焦单一目标主体的匿名化会导致非目标主体暴露更多。

**[Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](llm_evaluation/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)**

:   提出多层级任务画像引导的数据合成框架解决 LLM 路由的冷启动问题，并设计 TRouter——一种将任务类型作为隐变量的路由方法，通过变分推断建模查询-成本-性能关系，在冷启动和域内设置下均实现有效路由。

**[Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](llm_evaluation/text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)**

:   本文提出Quantile Token Regression方法，通过在输入序列中插入专用分位数token并结合检索到的邻居实例及其经验分布，使LLM能够预测完整的条件分布而非单一点估计，在Airbnb和StackSample数据集上相比基线降低约4个MAPE点并将预测区间收窄2倍以上。

**[Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation](llm_evaluation/think_in_latent_thoughts_a_new_paradigm_for_gloss-free_sign_language_translation.md)**

:   提出 SignThought，一种推理驱动的无注释手语翻译框架：引入可学习的潜在思维槽作为视频和文本之间的显式中间语义层，通过"先规划后定位"的双流解码器实现语义规划与视觉证据检索的解耦，在多个基准上超越现有无注释方法。

---

## 🏥 医学图像 { #medical_imaging }

**["Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery](medical_imaging/34excuse_me_may_i_say_something34_colabscience_a_proactive_ai_assistant_for_biom.md)**

:   CoLabScience 通过 PULI（正无标注学习干预）框架，训练一个能在生物医学团队讨论中**主动判断何时介入、如何介入**的 LLM 助手，利用 GRPO 和强化学习协调器从流式对话中自动识别最佳干预时机并生成科学建议。

**[Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](medical_imaging/anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)**

:   提出Anonpsy框架，将精神科叙事的去标识化重新定义为图引导的语义重写问题——先将叙事转换为语义图，在图上进行受约束的扰动以修改身份信息同时保持临床结构，最后通过图条件生成重建叙事。

**[AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](medical_imaging/aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)**

:   提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。

**[Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](medical_imaging/benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)**

:   提出 CMedTEB（中文医学文本嵌入基准）和 CARE（非对称检索框架），前者通过多 LLM 投票+专家验证构建高质量的中文医学检索/重排/STS 基准，后者用轻量 BERT 编码查询+大型 LLM 编码文档的非对称架构，通过两阶段渐进对齐策略实现 LLM 级检索精度+BERT 级在线延迟。

**[Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](medical_imaging/beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)**

:   提出 StsPatient，通过从对比指令/回复对中提取领域特定的转向向量（Steering Vector），配合随机 Token 调制（STM）机制控制注入概率来模拟不同认知障碍领域和严重程度的标准化病人，相比 prompt engineering 方法在临床真实性上平均提升 11.23%，在严重程度可控性上超越最佳基线 18.54%。

**[Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](medical_imaging/beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)**

:   提出 Aegle 框架，通过图结构多智能体架构虚拟化多学科会诊（MDT），将解耦并行推理和动态拓扑引入门诊问诊流程，在24个科室53项指标上超越SOTA模型。

**[BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](medical_imaging/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)**

:   BioHiCL 利用 MeSH（医学主题词）的**层级多标签标注**为稠密检索器提供结构化监督，通过深度加权的标签相似度对齐嵌入空间与 MeSH 语义空间，使 0.1B 模型在生物医学检索、句子相似度和问答任务上超越大多数专用模型。

**[Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](medical_imaging/calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)**

:   研究社会身份标记（性取向和宗教信仰）如何扭曲LLM在医疗问答中的准确率和置信度校准，发现"同性恋"标记在9个LLM上一致导致性能下降和校准危机，且交叉身份产生非加性的特异性伤害。

**[Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](medical_imaging/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)**

:   本文通过构建高质量德语医学语料库 FineMed-de（从 FineWeb2 过滤 730 万文档/51 亿词），对三种 LLM（7B-24B）进行持续预训练和 SLERP 模型合并，创建 DeFineMed 模型家族，证明领域特化的 7B 模型可以在德语医学任务上显著缩小与 24B 通用模型的性能差距（胜率提升约 3.5 倍）。

**[Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](medical_imaging/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)**

:   提出CoPoLLM框架，通过构建首个带认知扭曲标注的情感支持对话数据集CogBiasESC，结合认知策略强化学习（CPRL）引擎和双流条件优化（DSCO），使LLM能诊断8类认知扭曲并生成策略感知的干预回复，在15个SOTA基线上全面领先。

**[CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](medical_imaging/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)**

:   CURA 提出一个双层不确定性校准框架：个体层面将预测不确定性与错误概率对齐，队列层面通过嵌入空间的邻域风险率正则化预测，在 MIMIC-IV 的五个临床风险预测任务上一致提升校准指标而不牺牲判别性能。

**[Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](medical_imaging/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)**

:   提出四种基于音频注意力的指标（AudioRatio、AudioConsistency、AudioEntropy、TextEntropy），训练轻量级逻辑回归分类器在推理时检测语音大模型（SpeechLLM）的幻觉，在域内数据上 PR-AUC 提升最高达 +0.23。

**[Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](medical_imaging/dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)**

:   本文提出临床诊断推理数据（CDRD）结构来捕获从症状到鉴别诊断的抽象临床推理逻辑，并基于 CDRD 通过 SFT+RL 两阶段训练构建 Dr. Assistant 模型（14B），在临床问诊基准上 ICD-Recall 超过 HuatuoGPT-o1-72B 13.59%，达到与 GPT-5 竞争的水平。

**[Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](medical_imaging/efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)**

:   本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。

**[Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](medical_imaging/eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)**

:   本文提出MedSSR框架，通过注入罕见病知识的可控数据合成和"自监督RL→监督RL"的半监督训练范式，高效提升LLM的医学推理能力，在罕见病任务上实现最高+5.93%的提升，突破了现有方法+3%的改进上限。

**[Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](medical_imaging/faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)**

:   本文构建 MedCounterFact 数据集——用无义词、医学术语、非医学物品和有毒物质系统替换临床试验中的干预措施——发现前沿 LLM 在反事实医疗证据面前几乎无条件遵从上下文，即便"证据"表明海洛因或芥子气有疗效也自信回答，揭示了忠实度与安全之间缺乏明确边界的严重问题。

**[From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](medical_imaging/from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)**

:   本文将Toulmin论证模型适配到临床诊断过程，提出CGCL三阶段课程训练框架（事实收集→假设检验→综合结论），配合T-Eval量化评估推理结构完整性，在无需RL的情况下实现与RL方法可比的诊断推理质量。

**[HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](medical_imaging/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)**

:   本文提出医疗场景下的编解码器伪造语音检测任务 HCFD，构建了首个包含多种临床病理条件（抑郁、阿尔茨海默、构音障碍）的编解码器伪造语音数据集 HCFK，并提出 PHOENIX-Mamba 框架——通过在双曲空间中建模多模式伪造证据原型，在英文抑郁检测上达到 97.04% 准确率。

**[HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](medical_imaging/hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)**

:   本文提出 HypEHR，一个仅 22M 参数的洛伦兹双曲模型，将医学编码、就诊记录和问题嵌入双曲空间，通过层级感知正则化对齐 ICD 本体结构，在 MIMIC-IV 电子病历问答任务上接近 LLM 方法的效果。

**[Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](medical_imaging/inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)**

:   本文提出 DyReMe 动态医学诊断评估框架，通过 DyGen 模块生成包含鉴别诊断和误诊因素等临床干扰项的全新诊断案例，并通过 EvalMed 模块从准确性、真实性、帮助性和一致性四个维度评估 LLM，揭示现有静态基准高估了 LLM 的诊断能力——GPT-5 在 DyReMe 上准确率下降 8.25%，12 个 LLM 均暴露出显著的可信度不足。

**[Language Reconstruction with Brain Predictive Coding from fMRI Data](medical_imaging/language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)**

:   本文提出 PredFT，一个结合主网络（语言解码）和侧网络（脑预测编码表征）的端到端 fMRI-to-Text 解码模型，通过从大脑预测相关脑区（PTO 区域）提取前瞻性语义表征并融合到解码过程中，在 LeBel 数据集上 BLEU-1 达 34.95%（Sub-1），相比最强基线 MapGuide 提升 7.84 个百分点。

**[Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](medical_imaging/learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)**

:   提出 OPL-MT-MNAR 框架，通过 MNAR 感知的多模态编码器 + 贝叶斯滤波隐状态 + 离线策略学习，从结构化数据和临床文本的"缺失模式本身携带的信息"中学习 ICU 患者动态表示，实现优于临床医生行为的脓毒症治疗策略（FQE 0.679 vs 0.528）。

**[LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](medical_imaging/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)**

:   本文提出 LogosKG，一个硬件对齐的知识图谱检索框架，通过将图遍历转化为三元稀疏矩阵（SUB/OBJ/REL）的乘法运算，配合度感知图分区、跨图路由和按需缓存，在单设备上实现了对十亿边规模 KG 的可扩展、可解释高跳检索，并通过下游 KG-LLM 交互实验揭示了图拓扑结构对 LLM 诊断推理的影响。

**[MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)**

:   本文提出 MARCH，一个模拟放射科住院医-专科医-主治医层级协作流程的多智能体框架，通过三阶段（初始报告起草、检索增强修订、共识驱动定稿）生成 CT 报告，在 RadGenome-ChestCT 数据集上 CE-F1 达 0.399，比最佳基线 Reg2RG 的 0.253 提升 57.7%。

**[Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](medical_imaging/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)**

:   本文提出 CARE 框架和 FAITH-M 基准数据集，通过对话上下文编码与对比范例检索+知识蒸馏链式推理（KD-CoT），对 AI 生成的心理治疗对话进行六个治疗原则维度的细粒度序数评估，加权 F1 达 63.34，比最强基线 Qwen3 提升 64.26%。

**[MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](medical_imaging/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)**

:   本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。

**[Model-Agnostic Meta Learning for Class Imbalance Adaptation](medical_imaging/model-agnostic_meta_learning_for_class_imbalance_adaptation.md)**

:   本文提出 HAMR（Hardness-Aware Meta-Resample），一个统一的元学习框架，通过双层优化动态估计实例级权重优先处理真正困难的样本，配合邻域感知重采样机制将训练焦点放在困难样本及其语义邻居上，在 6 个不平衡 NLP 数据集上持续超越强基线。

**[Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](medical_imaging/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)**

:   本文提出将话语分解为情感-逻辑-行为（ELB）三组件并用 LLM 推理多个认知扭曲实例，然后通过多视角门控注意力 MIL 框架进行 bag 级分类，在韩语（KoACD）和英语（Therapist QA）数据集上均优于 LLM 直接推理基线。

**[OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](medical_imaging/omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)**

:   本文构建了首个大规模、多领域、基于真实案例的 LLM 安全合规数据集 OmniCompliance-100K，包含 12,985 条人工整理的法规/政策规则和 106,009 条通过 Web 搜索智能体采集的真实合规案例，覆盖 AI 安全、数据隐私、金融、医疗等 9 个领域，并通过广泛的基准实验揭示了当前 LLM 在安全合规能力上的系统性短板。

**[PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](medical_imaging/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)**

:   本文基于国际医学伦理黄金标准——Principlism（自主、不伤害、有益、公正四原则），构建了 PrinciplismQA 基准（3,648 题，含知识 MCQA 和开放式临床伦理困境），并配套专家校准的评估流水线，发现 LLM 在知识基准上的高准确率并不等于具备临床伦理推理能力——最强模型 o3 总分也仅 77.5%。

**[Query Pipeline Optimization for Cancer Patient Question Answering Systems](medical_imaging/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)**

:   本文提出 CoMeta，一个面向癌症患者问答（CPQA）的三层可控元数据感知 RAG 框架，通过临床混合语义-符号文档检索（CHSDR）融合 E-Utilities 实时布尔搜索与 MedCPT 语义检索，配合语义增强重叠分割（SEOS）防止上下文碎片化，在 CMMQA 数据集上将 Claude-3-Haiku 的回答准确率提升 5.24%（vs CoT）和约 3%（vs naive RAG）。

**[RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](medical_imaging/ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)**

:   提出 RA-RRG 框架，通过 LLM 从放射报告中提取临床关键短语并构建检索库，给定胸部 X 光影像后检索相关短语并输入 LLM 生成报告，无需 LLM 微调即可有效抑制幻觉，仅需 18 GPU 小时训练，在 CheXbert 指标上达到 SOTA。

**[RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](medical_imaging/rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)**

:   本文提出 RADS（Reinforcement Adaptive Domain Sampling），一种基于强化学习的样本选择策略，在极端低资源和类别不平衡的临床场景下，通过智能选择少量目标域样本进行标注和联合微调，显著提升跨域疾病检测的迁移效果。

**[Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](medical_imaging/region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)**

:   本文提出首个带有细粒度 ROI 标注的 3D PET/CT 数据集 VietPET-RoI（越南语），以及模拟放射科医生诊断流程的层次化报告生成框架 HiRRA，通过图神经网络建模 ROI 间的空间-形态学关系，BLEU-4 提升 19.7%，临床指标 RoIQ 提升 45.8%。

**[RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](medical_imaging/reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)**

:   本文提出 RePrompT，一种时间感知的 LLM 框架，通过循环提示调优（将前一次就诊的隐状态作为下一次就诊的软提示）和结构化编码提示调优（注入群体级 EHR 编码器的嵌入）两种互补机制，在 MIMIC-III/IV 上的再入院和死亡率预测任务上一致超越 EHR 基线和 LLM 基线。

**[RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](medical_imaging/ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)**

:   RiTeK 构建了两个大规模医学文本知识图谱（TKG）和对应的复杂推理 QA 数据集，涵盖 6 种拓扑结构和丰富的文本描述，评估了 11 种检索方法并揭示了现有 LLM 驱动检索系统在医学 TKG 推理上的严重不足。

**[Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](medical_imaging/semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)**

:   本文提出一种纯音频的半监督学习框架，通过在会话级、片段级和帧级三个层次联合建模临床对话中的病理语音特征，利用 EMA 教师-学生网络动态生成高质量伪标签，在抑郁症和阿尔茨海默症检测中仅用 11 个标注样本即可达到全监督 90% 的性能。

**[Stable On-Policy Distillation through Adaptive Target Reformulation](medical_imaging/stable_on-policy_distillation_through_adaptive_target_reformulation.md)**

:   本文提出 Veto，一种目标层面的重构方法，通过在 logit 空间构建教师-学生的几何桥接分布来稳定 on-policy 知识蒸馏，单一参数 $\beta$ 同时在 forward KL 中充当自适应梯度否决器（抑制低置信度 token 的有害梯度）和在 reverse KL 中充当果断性旋钮（平衡奖励驱动和输出多样性），在 GSM8K 上比 SFT 提升 9.2%。

**[Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](medical_imaging/text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)**

:   本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图，最终联合训练 LoRA 微调的 LLaMA 编码器和异构 GNN 学习统一的医学概念嵌入，在 MIMIC-III/IV 上显著提升诊断预测性能。

**[Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](medical_imaging/thinking_like_a_botanist_challenging_multimodal_language_models_with_intent-driv.md)**

:   本文提出PlantInquiryVQA基准和Chain-of-Inquiry（CoI）框架，包含24,950张植物图像和138,068个问答对，模拟植物学家的适应性诊断提问策略，评估18个MLLM在植物病理诊断中的多步视觉推理能力，发现结构化提问显著提升诊断准确性并减少幻觉，但即使最强模型的临床实用性得分仅0.188。

---

## 💡 LLM 推理 { #llm_reasoning }

**[AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)**

:   提出 AIM-CoT 框架，通过信息觅食理论驱动的主动视觉证据选择(AVP)和基于注意力偏移的动态触发机制(DAT)，解决交错模态思维链(I-MCoT)中"看什么"和"何时看"两个核心问题。

**[Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](llm_reasoning/budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)**

:   本文提出了一种预算感知的任意时推理（anytime reasoning）框架和 Anytime Index 指标，用于量化 LLM 在有限 token 预算下的推理质量-效率权衡，并设计了基于 LLM 自合成偏好数据的推理时自改进方法（PDP），在规划、数学和科学 QA 任务上显著提升了中间和最终解的质量。

**[Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)**

:   本文提出 Alignment Score——一种基于语义熵矩阵的语义级指标，通过比较模型生成的思维链与人类偏好参考链的中间步骤来量化推理对齐度，发现 Alignment Score 与任务准确率、可读性和连贯性高度相关，且 2-hop 推理是对齐的峰值深度。

**[Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](llm_reasoning/challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)**

:   提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。

**[Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](llm_reasoning/decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)**

:   本文通过 Cross-CoT 实验和逐步分析，揭示了 CoT 推理的"解耦机制"：最终准确率由 CoT 内容决定（99% 方差贡献），但分布排序由模型内在先验主导（>80%），说明长 CoT 是强大的决策器但弱的分布校准器。

**[Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](llm_reasoning/discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)**

:   发现 LLM 内部存在一个共享的逻辑子空间，可同时对齐自然语言和符号逻辑两种推理表示，通过在推理时沿该子空间引导激活可无训练提升逻辑推理准确率最高达 11 个百分点。

**[Dissecting Failure Dynamics in Large Language Model Reasoning](llm_reasoning/dissecting_failure_dynamics_in_large_language_model_reasoning.md)**

:   通过分析 LLM 推理轨迹，发现错误集中在早期的少数关键转折点，错误发生后模型进入"认知螺旋"——局部连贯但全局错误地不断延伸；基于此提出 GUARD 框架，在熵信号检测到的高风险转折点处进行短距分支修复。

**[Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](llm_reasoning/do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)**

:   提出 LTE (Learning to reason from Trial and Error)，通过将模型自身生成的错误答案作为提示引导额外 rollout，在不依赖外部专家的情况下有效缓解 RLVR 中的探索停滞问题。

**[DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](llm_reasoning/dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)**

:   DPC 将 Text-to-SQL 的候选选择从"在隐藏数据上猜测"转化为"在可见数据上确定性验证"：构造最小区分数据库（MDD）使冲突 SQL 产生不同结果，再用 Python/Pandas 解作为参考锚点通过跨范式一致性选择正确候选，在 BIRD 和 Spider 上超越 Self-Consistency 最高 2.2%。

**[Efficient PRM Training Data Synthesis via Formal Verification](llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)**

:   本文提出 FoVer，一个利用形式化验证工具（Z3 和 Isabelle）为形式化推理任务的步骤级推理链自动标注正确性标签的框架，生成 FoVer-40K 训练集并微调 PRM，在 12 个推理基准上展示了从形式化到非形式化的迁移能力和跨任务泛化能力。

**[Efficient Process Reward Modeling via Contrastive Mutual Information](llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)**

:   提出 CPMI（Contrastive Pointwise Mutual Information），一种高效的自动步级奖励标注方法，通过对比推理步骤对正确答案和错误答案的条件概率变化量来估计步级贡献，比 Monte Carlo 估计减少 84% 构建时间和 98% token 生成量，同时在过程级评估和数学推理基准上取得更高准确率。

**[Explicit Trait Inference for Multi-Agent Coordination](llm_reasoning/explicit_trait_inference_for_multi-agent_coordination.md)**

:   提出显式特质推理（ETI）方法，基于心理学中温暖和能力两个维度让LLM智能体推理并追踪合作伙伴的行为特征，在经济博弈中减少45-77%收益损失，在MultiAgentBench上提升3-29%任务表现。

**[Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](llm_reasoning/failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)**

:   本文提出 Multi-Focus Attention Instruction (MFAI) 作为语义探针，揭示多跳 QA 中的"最弱链效应"——多跳推理性能由最不可见证据的绝对位置决定而非事实间距离，失败主要源于识别瓶颈而非推理缺陷，且 System-2 推理模型能有效抵御位置偏差和误导性注意力线索。

**[FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](llm_reasoning/fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)**

:   本文提出 FS-Researcher，一个基于文件系统的双 Agent 深度研究框架，通过 Context Builder 构建层次化知识库、Report Writer 分节撰写报告，利用持久化工作空间突破上下文窗口限制，在 DeepResearch Bench 上达到 53.94 RACE（SOTA），并展示了上下文构建计算量与报告质量的正相关测试时扩展效应。

**[GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](llm_reasoning/ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)**

:   本文提出 GanitLLM，首个真正用孟加拉语进行推理（而非翻译或用英语推理）的数学推理模型，构建了难度标注的孟加拉语数学数据集 Ganit，并提出 Curriculum-GRPO 解决低资源语言 GRPO 训练中的冷启动问题，4B 模型在 Bn-MGSM 上提升 8 个准确率百分点，孟加拉语推理 token 从 14% 提升至 88%。

**[Generating Effective CoT Traces for Mitigating Causal Hallucination](llm_reasoning/generating_effective_cot_traces_for_mitigating_causal_hallucination.md)**

:   本文首先提出了因果幻觉率（CHR）指标来量化小型 LLM 在事件因果识别中过度预测因果关系的倾向，然后通过系统实验确定了有效 CoT 数据的两个关键标准（充分长度的语义解释+与目标模型对齐的分布），设计了一套低成本的 CoT 数据生成管线，将 Qwen2.5-1.5B 的 CHR 从 83.54% 降至 6.26%，同时提升平均准确率至 66.00%。

**[How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](llm_reasoning/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)**

:   本文系统研究如何通过 SFT 增强大型推理模型（LRM）的安全性，发现直接蒸馏安全响应效果有限的根因是五种风险推理模式（尤其是"弱犹豫"），提出针对性的蒸馏策略将 PAIR 攻击成功率从 63% 降至 13%，并发现短推理链和模板推理在安全性上与长推理链表现相当。

**[JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)**

:   JTPRO 提出了一种无需模型微调的联合优化框架，通过反思驱动的迭代编辑同时优化全局指令和逐工具的 schema/参数描述，在大规模工具库场景下显著提升工具选择和参数填充的端到端成功率，相比 GEPA 等基线在 OSR 上提升 5%–20%。

**[Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](llm_reasoning/know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)**

:   本文提出 InstruCoT，通过合成覆盖多种注入向量和威胁场景的多样化训练数据，并引入基于情境感知模型的三阶段指令级思维链微调，使 LLM 在面对各类提示注入攻击时能有效识别并拒绝恶意指令，在行为偏离、隐私泄露和有害输出三个维度上大幅超越现有防御方法。

**[Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)**

:   本文系统性地研究了大型推理模型（LRM）在 11 种语言上的潜在推理行为，发现潜在推理能力存在于多语言中但分布不均（高资源语言强、低资源弱），且内部推理动态趋于以英语为中心的共享路径。

**[Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](llm_reasoning/logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)**

:   本文发现 LLM 逻辑推理存在"逻辑相变"现象——性能在特定复杂度阈值处突然崩塌而非平滑退化，提出逻辑复杂度度量（LoCM）来量化这一现象，并设计神经符号课程调优框架（NSCT），通过自适应神经-符号对齐和复杂度感知课程优化，在五个基准上平均提升 naive prompting +1.26 和 CoT +3.95 准确率。

**[MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](llm_reasoning/march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)**

:   提出 MARCH 基准（2,209 个多跳歧义问题）和 CLARION 框架，首次系统研究歧义解析与多步推理交叉场景下的 QA 挑战，揭示现有 SOTA 模型在此类问题上的严重不足。

**[MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)**

:   提出基于约束图对抗进化的分层数据合成框架 MathAgent，将数据合成从文本生成任务重构为约束图的无监督优化问题，通过 Legislator 三Agent系统进化问题骨架再由 Executor 实例化为自然语言，仅 1K 合成样本即超越 LIMO 和 s1K 在八个数学基准上的表现。

**[Parallel Test-Time Scaling for Latent Reasoning Models](llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)**

:   本文首次将并行测试时缩放（parallel TTS）引入潜在推理模型，提出两种基于不确定性理论的随机采样策略（MC-Dropout 和加性高斯噪声）以及一个步级对比训练的潜在奖励模型（LatentRM），使得在连续向量空间中进行推理的模型也能通过并行采样+聚合获得稳定的性能提升。

**[Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](llm_reasoning/process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)**

:   本文提出利用规划领域定义语言（PDDL）自动生成大规模、高精度的步骤级奖励数据集，用于训练过程奖励模型（PRM），在数学和非数学推理基准上均取得显著提升。

**[ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](llm_reasoning/recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)**

:   本文构建了 ReCoQA——一个包含 29,270 个房地产问答对的大规模基准，要求模型融合数据库查询和地图 API 调用进行混合多源推理，并提出层次化多 Agent 框架 HIRE-Agent 作为强基线，系统性地揭示了现有 LLM 在垂直领域复杂推理中的瓶颈。

**[Reinforced Efficient Reasoning via Semantically Diverse Exploration](llm_reasoning/reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)**

:   ROSE 提出语义熵引导的 MCTS 分支策略和长度感知的段级优势估计，解决了现有 MCTS-based RLVR 方法探索多样性不足和推理效率低的问题，在多个数学推理基准上取得最优 pass@8 性能。

**[Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)**

:   提出 Render-of-Thought（RoT），首次将文本 CoT 推理步骤渲染为图像，利用预训练视觉编码器作为语义锚点将 LLM 隐状态对齐到视觉嵌入空间，实现 3-4 倍 token 压缩和显著推理加速，同时保持推理链的可分析性。

**[Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)**

:   系统性研究了 RLVR 训练中 LLM 的熵动态，揭示正优势 token 是熵崩塌的主要驱动因素，并提出 Positive-Advantage Reweighting 方法通过动态调整正优势 token 的损失权重来有效调控模型熵。

**[Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](llm_reasoning/scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)**

:   提出 GenCluster，一个可扩展的测试时计算框架，通过大规模并行生成→行为聚类→锦标赛排名→循环提交策略，首次使开源模型 gpt-oss-120b 在 IOI 2025 上达到金牌水平（446.75/600 分）。

**[Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)**

:   本文提出RDDG，基于渐进式CoT的表格数据合成框架，通过核心集选择、关系挖掘和自强化反馈机制引导LLM生成高保真表格数据，在不平衡分类上平均提升2%+ Macro-F1。

**[Semantic-Aware Logical Reasoning via a Semiotic Framework](llm_reasoning/semantic-aware_logical_reasoning_via_a_semiotic_framework.md)**

:   提出 LogicAgent，一个基于格雷马斯符号方阵(Semiotic Square)的逻辑推理框架，通过多视角语义分析和反思验证，在语义复杂和逻辑复杂双重挑战下实现 SOTA 逻辑推理性能。

**[Think Outside the Policy: In-Context Steered Policy Optimization](llm_reasoning/think_outside_the_policy_in-context_steered_policy_optimization.md)**

:   提出 ICPO (In-Context Steered Policy Optimization)，利用大语言模型自身的上下文学习(ICL)能力作为隐式专家引导，在 RLVR 训练中扩展策略探索空间，无需依赖外部更强模型的推理轨迹。

**[Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)**

:   本文提出 DIN-Retrieval，通过识别 LLM 中跨域激活极性一致的域不变神经元（DIN），构建域鲁棒的表示子空间用于检索结构兼容的跨域示例，首次证明了使用跨域 ICL 示例提升 LLM 推理性能的可行性，在数学-逻辑推理迁移上平均提升 1.8%。

**[TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](llm_reasoning/trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)**

:   TrigReason 提出基于事件触发的大小推理模型协作框架，通过分析小模型三类推理风险（路径偏离、认知过载、恢复失能），设计策略引导、认知卸载和干预请求三种触发器替代逐步轮询验证，在保持 LRM 精度的同时将 1.70-4.79 倍更多推理步骤卸载给小模型，延迟降低 43.9%、API 成本降低 73.3%。

**[TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](llm_reasoning/trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)**

:   本文将自动化多轮越狱攻击建模为多轮强化学习问题，提出 TROJail，通过两个启发式过程奖励（过度有害惩罚和语义相关性递进）缓解结果奖励的稀疏监督问题，在多个模型和基准上显著提升攻击成功率。

**[When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning](llm_reasoning/when_is_thinking_enough_early_exit_via_sufficiency_assessment_for_efficient_reas.md)**

:   提出 DTSR 框架，通过检测推理过程中的"反思信号"（如 Wait、Alternatively）并在该位置让模型自我评估当前推理的"充分性"来决定是否提前终止推理，在 Qwen3 系列模型上实现 28.9%–34.9% 的推理长度缩减且几乎不损失精度。

---

## 💬 LLM / NLP { #llm_nlp }

**[A Study of LLMs' Preferences for Libraries and Programming Languages](llm_nlp/a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)**

:   首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。

**[Adam's Law: Textual Frequency Law on Large Language Models](llm_nlp/adam39s_law_textual_frequency_law_on_large_language_models.md)**

:   本文提出"文本频率定律"（TFL），发现当语义相同时，使用更高频率的文本表达来提示或微调LLM能获得更好效果，并设计了频率蒸馏和课程训练策略来进一步利用该规律。

**[AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](llm_nlp/alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)**

:   提出 AlphaContext，一个基于进化树的心理测量情境生成器，通过 HyperTree 大纲规划、MCTS 逐句生成、MAP-Elites 多样性优化和评估引导迭代精炼四个模块，自动生成用于创造力评估的高质量长文本情境，在 7 个评估维度上平均超越竞争方法 8%。

**[An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](llm_nlp/an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)**

:   通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。

**[Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](llm_nlp/are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)**

:   系统研究LLM中情感和修辞神经元的表征机制及其内在关联，提出结合多维筛选的神经元识别框架和自适应遮蔽验证方法，实现了情感/修辞预测的定向诱导和修辞神经元辅助情感识别。

**[Automatic Combination of Sample Selection Strategies for Few-Shot Learning](llm_nlp/automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)**

:   本文提出 ACSESS 方法，通过前向选择、后向选择和 Datamodels 三种机制自动识别互补的样本选择策略并加权组合，在 23 种策略、5 个 ICL 模型和 3 种梯度少样本学习方法、6 个文本和 8 个图像数据集上验证了组合策略一致优于单一策略和 ICL 专用基线。

**[ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](llm_nlp/chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)**

:   ChatHLS 提出了一个多智能体 HLS 设计框架，通过 HLSTuner（QoR 感知推理优化指令选择）和 HLSFixer（分层反馈增强的调试框架）两个核心组件，结合自进化错误用例扩展机制（VODA），在 HLS-C 生成成功率和硬件性能优化上显著超越基线。

**[CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](llm_nlp/costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)**

:   提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。

**[Detoxification for LLM from Dataset Itself](llm_nlp/detoxification_for_llm_from_dataset_itself.md)**

:   本文提出 HSPD（层次化语义保留去毒）流水线，通过 SoCD（软对比解码）引导 LLM 定位并重写原始语料中的有毒片段，同时保留语义，生成可直接替换原始数据用于微调的去毒语料——在 GPT2-XL 上将毒性概率从 0.42 降至 0.18，在 LLaMA2-7B、OPT-6.7B 和 Falcon-7B 上也取得了最优去毒效果。

**[Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](llm_nlp/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)**

:   本文提出 PA-Tool，一种无训练的工具 Schema 优化方法，利用从数据污染检测中借鉴的"尖锐度"（peakedness）信号识别模型预训练中熟悉的命名模式，通过重命名工具组件来对齐小语言模型的内化知识，在 MetaTool 和 RoTBench 上实现最高 17% 的提升，Schema 不对齐错误减少 80%。

**[EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](llm_nlp/evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)**

:   EvoSpark 提出一个支持长程叙事演化的多智能体框架，通过分层递归记忆（RSB 做社会认知代谢）、生成式场面调度（GMS 做角色-地点-情节对齐）和涌现角色锚定协议（ECGP 将 LLM 幻觉转化为持久角色）三重设计解决社会记忆堆叠和叙事-空间失谐问题。

**[Expect the Unexpected? Testing the Surprisal of Salient Entities](llm_nlp/expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)**

:   本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

**[FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](llm_nlp/fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)**

:   本文分析了连续扩散语言模型在少步采样时自条件化信号的不匹配和训练饱和两个瓶颈，提出FastDiSS框架通过自条件化扰动（SCP）和模型感知噪声缩放（MANS）来改善鲁棒性，在6个基准上实现4×-400×加速同时保持质量。

**[Foresight Optimization for Strategic Reasoning in Large Language Models](llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)**

:   本文提出 Foresight Policy Optimization（FoPO），通过在策略优化中引入对手建模的前瞻修正项，使 LLM 能够显式预见对手行为并据此调整自身策略，在合作（Cooperative RSA）和竞争（Competitive Taboo）两类博弈任务上显著提升策略推理能力，并在跨域 γ-Bench 上取得一致性提升。

**[From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)**

:   本文首次系统综述流式大语言模型（Streaming LLMs），提出基于数据流和交互并发性的统一定义，将现有方法分为三级递进分类——输出流式（Output-streaming）、顺序流式（Sequential-streaming）和并发流式（Concurrent-streaming），覆盖文本、语音和视频流式场景的方法论和应用。

**[GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)**

:   提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。

**[HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](llm_nlp/hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)**

:   提出 HCRE 模型，通过构建层次化关系树将跨文档关系抽取从大规模关系集的直接分类转化为逐层层次化分类，并设计预测-验证推理策略缓解层间错误传播，在 CodRED 数据集上显著超越 SLM 和 LLM 基线。

**[How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](llm_nlp/how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)**

:   本文发现推理 LLM（如 DeepSeek-R1）在定量推理中存在"良性自读"模式——答案 token 对推理痕迹的注意力呈现前移漂移（沿推理链逐步推进）和语义锚点集中（反复回顾关键步骤），且此模式与正确性强相关；基于此提出 SRQ（自读质量）驱动的免训练激活引导方法，在多个基准上提升准确率最高 2.6%。

**[It's High Time: A Survey of Temporal Question Answering](llm_nlp/it39s_high_time_a_survey_of_temporal_question_answering.md)**

:   本文提供了时序问答（TQA）的全面综述，提出了基于语料时间性、问题时间性和模型时间能力三个维度的统一分析框架，系统梳理了从规则管道到 Transformer/LLM 时代的 TQA 方法演进、基准数据集和评估策略，并识别了未来挑战。

**[Iterative Formalization and Planning in Partially Observable Environments](llm_nlp/iterative_formalization_and_planning_in_partially_observable_environments.md)**

:   提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。

**[Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](llm_nlp/lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)**

:   本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。

**[Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](llm_nlp/masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)**

:   本文通过对比自探针（使用模型自身隐藏状态）和外部探针（使用其他模型隐藏状态）预测正确性的能力，发现"模型间一致性"是掩盖特权知识的关键混淆因子，在消除一致性后揭示了领域特异性的特权知识：事实性任务中存在但数学推理中不存在。

**[Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](llm_nlp/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)**

:   提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。

**[MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](llm_nlp/muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)**

:   提出 MulDimIF 多维约束框架，从约束模式（3种）、约束类别（4类13子类）和约束难度（4级）三个维度系统评估 LLM 的指令遵循能力，并通过 GRPO 训练显著提升模型性能，发现改进主要源自注意力模块的参数更新。

**[Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](llm_nlp/not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)**

:   本文提出首个结合 FrameNet 语义框架和概念隐喻理论（CMT）源域的计算框架 ConceptFrameMet，通过 RoBERTa 多任务模型检测隐喻并预测其语义框架和源域，配合对数似然比统计方法发现话语中显著的隐喻模式，揭示了自由派和保守派在移民话语中使用相同源域但选择不同语义框架来传达截然不同的联想。

**[One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)**

:   本文系统比较了 6 种常用的人物画像提示方式（姓名/显式提及/对话历史各两种变体）在 7 个 LLM 和 4 个任务上的效果，发现虽然平均响应跨提示方式高度相关，但不同提示方式产生的人物画像间差异显著不同，过于显式的提示导致更强的个性化偏差，警示不应基于单一提示方式得出偏差结论。

**[Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](llm_nlp/please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)**

:   本文提出 AdaCD（自适应对比解码），通过比较极端安全提示下和无提示下的 token 分布差异提取拒绝 token 分布，再根据一致性比率动态决定增强或抑制拒绝行为，在降低过度拒绝 10.35% 的同时提升恶意查询拒绝率 0.13%。

**[Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](llm_nlp/revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)**

:   本文通过构建 RedirectQA 数据集（利用 Wikipedia 重定向信息将同一实体关联到多种表面形式），系统研究了 LLM 的非逐字记忆如何受实体命名变体的影响，发现事实记忆既非纯粹依赖特定表面形式也非完全表面无关，且实体级频率在表面频率之外仍有独立贡献。

**[Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](llm_nlp/route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)**

:   本文提出 R2A（Route to Rome Attack），通过在黑盒设置下构建混合集成代理路由器并优化通用对抗后缀，将 LLM 路由器的路由决策从廉价弱模型导向昂贵强模型——在 7 个开源路由器和 2 个商用路由器（GPT-5-Auto、OpenRouter）上平均攻击成功率提升 49%，推理成本增加 2.7-2.9 倍。

**[Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](llm_nlp/style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)**

:   发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。

**[The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](llm_nlp/the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)**

:   提出 SA-MCQ 诊断框架揭示知识编辑中的"表面合规"现象——编辑器在标准基准上达到高分但并未真正覆写内部信念，模型在判别式自评中会回退到原始参数记忆，递归编辑还会累积表征残留导致认知不稳定。

**[Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](llm_nlp/think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)**

:   本文提出在 LLM 输入中的句子边界处插入分隔符标记，通过 ICL 和 SFT 两种方式实现"逐句思考"的推理范式，在 7B 到 600B 模型上取得一致提升（GSM8k +7.7%，DROP +12.5%），且几乎不增加额外计算开销。

**[Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](llm_nlp/towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)**

:   提出 SpreadsheetAgent，一种两阶段多智能体框架，通过代码执行、视觉和 LaTeX 三种格式的渐进式区域读取与交叉验证，在不超出 LLM 上下文限制的前提下实现鲁棒的真实世界电子表格理解。

**[Why Did Apple Fall: Evaluating Curiosity in Large Language Models](llm_nlp/why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)**

:   本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。

**[XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](llm_nlp/xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)**

:   本文提出 XtraGPT——首个面向学术论文修改的开源 LLM 套件（1.5B-14B），通过在 7,000 篇顶会论文和 140,000 个标准引导的指令-修改对上微调，实现上下文感知的段落级可控修改，7B 版本匹配 GPT-4o-mini，14B 版本超越 GPT-4o-mini，人类评估显示修改后论文预测评分平均提升 0.65 分。

---

## 🔬 可解释性 { #interpretability }

**[A Structured Clustering Approach for Inducing Media Narratives](interpretability/a_structured_clustering_approach_for_inducing_media_narratives.md)**

:   提出一个从大规模新闻语料中自动归纳媒体叙事模式的框架，通过联合建模事件因果链和角色（英雄/威胁/受害者）信息，使用角色约束的聚类算法将叙事链组织成语义连贯的叙事模式，在移民和枪支控制两个领域生成了可解释且与框架理论一致的叙事模式。

**[Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)**

:   构建大规模Post-hoc Self-Consistency Bank（PSCB，85K决策×428K解释），量化LLM答案与其解释之间的特征归因差距，并通过DPO优化在不损害准确率的前提下提升解释的归因一致性。

**[ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](interpretability/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)**

:   提出 ChemVLR，首个化学领域推理型 VLM，通过跨模态逆向工程策略构建 760K 推理数据集，结合持续预训练-SFT-RL 三阶段训练流程，在分子识别和反应预测任务上显著超越专有模型和领域专家 VLM。

**[Context-Value-Action Architecture for Value-Driven Large Language Model Agents](interpretability/context-value-action_architecture_for_value-driven_large_language_model_agents.md)**

:   提出 CVA（Context-Value-Action）架构，基于 S-O-R 心理学模型和 Schwartz 价值理论，通过训练在真实人类数据上的 Value Verifier 解耦行为生成与认知推理，有效缓解 LLM 智能体的行为极化问题，在超过 110 万真实交互轨迹的 CVABench 上显著优于基线。

**[Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](interpretability/curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)**

:   本文发现当前 LLM 数学推理中存在大量"Miracle Steps"——推理链中凭空跳跃到正确答案的现象，并提出 Rubric Reward Model (RRM)，一种基于问题特定评分标准的过程奖励函数，在 RL 训练中显著减少 Miracle Steps 71% 并将 AIME2024 的 Verified Pass@1024 从 26.7% 提升至 62.6%。

**[Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](interpretability/do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)**

:   发现并形式化了 LLM 工具调用中的"结构对齐偏差"——当查询属性可以有效映射到工具参数时（即使工具功能与用户目标无关），LLM 仍倾向调用该工具。构建 SABEval 数据集解耦结构对齐和语义相关性，用对比注意力归因揭示内部存在语义检查和结构匹配两条竞争路径，提出再平衡策略实现 80% 的相对错误减少。

**[Evian: Towards Explainable Visual Instruction-tuning Data Auditing](interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)**

:   提出"分解-再评估"（Decomposition-then-Evaluation）范式和 EVIAN 框架，将视觉指令微调数据的回答分解为视觉描述、主观推理和事实声明三个组件，沿图文一致性、逻辑连贯性和事实准确性三个正交维度评估，发现用其筛选的少量高质量数据训练的模型优于大规模数据集训练的模型。

**[Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](interpretability/experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)**

:   构建控制知识框架系统研究LLM在科学可行性评估中如何利用实验描述和结果证据，发现提供结果证据比实验描述更可靠，部分实验信息常导致性能低于仅用参数知识的基线，揭示了LLM推理的脆弱性。

**[Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](interpretability/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)**

:   本文提出 Laser，通过动态窗口对齐学习（DWAL）在潜在空间中进行视觉推理，使模型在推理过程中维持未来语义的"概率叠加态"而非逐 token 精确预测，实现"先全局后局部"的认知层次，在 6 个基准上以仅 6 个推理 token（减少 97%+）达到潜在推理方法的 SOTA，超越 Monet 平均 5.03%。

**[From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization](interpretability/from_signal_degradation_to_computation_collapse_uncovering_the_two_failure_modes.md)**

:   本文通过系统的机械可解释性分析，揭示LLM量化存在两种质性不同的失败模式：4-bit的信号退化（Signal Degradation，计算模式完整但精度受损，可局部修复）和2-bit的计算崩溃（Computation Collapse，关键组件功能性破坏，需结构重建）。

**[HistLens: Mapping Idea Change across Concepts and Corpora](interpretability/histlens_mapping_idea_change_across_concepts_and_corpora.md)**

:   提出 HistLens 框架，基于稀疏自编码器（SAE）将概念表示分解为可解释的语义基向量，在共享坐标系中追踪多概念、多语料的历时演化轨迹，支持隐式概念计算，为数字人文和概念史研究提供可量化、可比较的分析工具。

**[IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](interpretability/idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)**

:   提出 IDEA 框架，将 LLM 的决策知识提取为语义因子上的可解释参数化模型，通过 EM 算法联合学习语言概率表达到数值的映射和决策参数，实现了可校准、可编辑、可解释的 LLM 决策，在五个数据集上以 Qwen-3-32B (78.6%) 超越 DeepSeek R1 (68.1%) 和 GPT-5.2 (77.9%)。

**[Interpretability from the Ground Up](interpretability/interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)**

:   本文从教育评估利益相关者需求出发提出 FGTI 四原则（忠实、扎根、可追溯、可互换），开发 AnalyticScore 三阶段框架实现可解释自动评分，在 ASAP-SAS 上平均 QWK 仅比不可解释 SOTA 低 0.06。

**[Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretability/interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)**

:   通过规则化问题分解方法构建可验证的中间推理链数据集，揭示 CoT 推理链的语义正确性与最终答案准确率不可靠地相关（正确链仅 28% 导致正确答案），且最可解释的推理链并非最提升性能的——冗长的 R1 链性能最优但用户评为最不可解释。

**[LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](interpretability/leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)**

:   本文提出 LePREC，一种受法律专业人士启发的神经-符号框架，通过 LLM 生成推理问答对将非结构化法律文本转化为结构化特征，再利用稀疏线性模型进行相关性分类，在 769 个马来西亚合同法案例构建的 LIC 数据集上相比 GPT-4o 等 LLM 基线提升 30–40%。

**[LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](interpretability/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)**

:   本文提出 LLM 引导的语义引导框架，通过 LLM 生成子意图和三阶段课程式合成数据训练非否定 Tsetlin Machine（NTM），提取高置信度符号特征注入真实数据，使标准 TM 在保持完全可解释性的同时逼近 BERT 的分类性能。

**[NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)**

:   提出 NOSE 三模态嗅觉表示学习框架，以分子为枢纽通过正交注入机制对齐分子结构、受体序列和自然语言描述三个模态，配合 LLM 驱动的弱正样本策略缓解描述稀疏问题，在 11 个下游任务上达到 SOTA 并展现优秀的零样本泛化能力。

**[PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](interpretability/pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)**

:   本文提出 PV-SQL，一个 Agent 式 Text-to-SQL 框架，通过 Probe（迭代生成探测查询发现数据库值格式/列语义/表关系）和 Verify（基于模式匹配提取可验证约束并构建检查清单）两个互补组件，在 BIRD 基准上比最佳基线高 5% 执行准确率和 20.8% 有效效率分。

**[Reasoning Fails Where Step Flow Breaks](interpretability/reasoning_fails_where_step_flow_breaks.md)**

:   提出 Step-Saliency 诊断工具发现大推理模型中两种深度相关的信息流失败模式（Shallow Lock-in 和 Deep Decay），并设计 StepFlow 测试时干预方法在不重训练的情况下修复信息传播、提升推理准确率。

**[Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)**

:   本文提出一种基于代理模型的黑盒可解释性框架，利用廉价小模型近似昂贵大模型的局部决策边界来生成 LIME/SHAP 解释，通过统计筛选-应用（screen-and-apply）机制确保可靠性，代理解释在保持超过 90% 保真度的同时将成本降低 88.2%，并成功用于 Prompt 压缩和中毒样本移除等下游优化任务。

**[Rhetorical Questions in LLM Representations: A Linear Probing Study](interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)**

:   通过线性探针分析 LLM 内部如何表征反问句，发现反问句在表征空间中是线性可分的且可跨数据集迁移，但不同数据集学到的探针方向并不一致——反问句由多个异构的线性方向编码，而非单一统一维度。

**[Similarity-Distance-Magnitude Activations](interpretability/similarity-distance-magnitude_activations.md)**

:   本文提出 SDM（Similarity-Distance-Magnitude）激活函数作为 softmax 的更鲁棒替代，通过将正确预测的深度匹配（Similarity）、到训练分布的距离（Distance）和决策边界距离（Magnitude）三个认知维度解耦并整合为新的激活 $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$，并在此基础上构建 SDM 估计器进行选择性分类，在协变量偏移和分布外输入下比现有校准方法更鲁棒。

**[SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings](interpretability/soft_head_selection_for_injecting_icl-derived_task_embeddings.md)**

:   SITE 提出了一种基于梯度优化的软注意力头选择方法，通过识别任务相关的注意力头来有效注入 ICL 衍生的任务嵌入，在 12 个 LLM（4B-70B）上显著超越 ICL 和现有嵌入方法，同时用远少于 PEFT 的可训练参数达到可比性能。

**[SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](interpretability/spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)**

:   SPENCE 通过对 NL2SQL 基准查询进行系统性句法改写并测量执行准确率随句法距离的衰减程度，检测和量化 LLM 在 NL2SQL 基准上的数据污染行为，发现越老的基准（如 Spider）污染信号越强，而较新的 BIRD 基准几乎不受影响。

**[StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](interpretability/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)**

:   本文提出 StructKV，一个结构感知的 KV Cache 压缩框架，通过全局入度中心性（Global In-Degree Centrality）跨层累积注意力模式识别全局信息枢纽，动态枢纽层检测（Dynamic Pivot Detection）自适应定位最优压缩层，以及结构传播与解耦（Structural Propagation & Decoupling）分离计算预算和存储预算，在 LongBench 和 RULER 上以 60% prefill + 10% KV 实现了接近全上下文的性能。

**[Style over Story: Measuring LLM Narrative Preferences via Structured Selection](interpretability/style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)**

:   本文设计了一种基于约束选择的实验范式来测量 LLM 的叙事偏好，使用叙事学理论构建的 200 个约束库让 6 个 LLM 在不同指令类型下进行选择，发现模型系统性地优先选择"风格"（Style）而非"事件"（Event）、"角色"（Character）和"场景"（Setting）等内容元素。

**[TabReX: Tabular Referenceless eXplainable Evaluation](interpretability/tabrex_tabular_referenceless_explainable_evaluation.md)**

:   提出 TabReX，一种基于图推理的无参考表格生成评估框架，将源文本和生成表格转化为知识图谱三元组并对齐，计算可解释的属性驱动分数，在人类判断相关性上大幅超越现有方法；同时构建 TabReX-Bench 大规模基准。

**[The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](interpretability/the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)**

:   系统性揭示了"推理陷阱"悖论：增强LLM推理能力（无论通过RL、蒸馏还是可切换推理模式）会系统性地放大工具幻觉，且这一效应与推理本身而非RL训练相关联，现有缓解策略（提示工程、DPO）面临不可避免的可靠性-能力权衡。

**[ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](interpretability/threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)**

:   本文提出 ThreadSumm，一个多阶段 LLM 管道框架，将嵌套话语线程摘要建模为层次推理问题——先提取方面和原子内容单元进行内容规划，再通过句子排序构建线程感知序列，最后用 Tree of Thoughts 搜索生成和评分多个段落候选，在 Reddit/StackExchange 数据集上优于基线。

**[To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](interpretability/to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)**

:   本文为 LLM 多智能体系统（LLM-MAS）提出首个全面的"可信度"定义（基于 Grice 合作原则的六个正交维度），发现 LLM 的注意力模式可区分不同类型的可信度违规，据此设计了轻量级的 A-Trust 评估方法和端到端的信任管理系统（TMS），在多种攻击下将恶意消息检测率提升至 77-90%。

**[Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)**

:   系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。

**[Tracing Relational Knowledge Recall in Large Language Models](interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)**

:   本文系统研究LLM在文本生成过程中回忆关系知识的内部机制，发现注意力头对残差流的逐头贡献（$\Delta_{att,h}$）是线性关系分类的最强特征（准确率达91%），并提出HeadScore和TokenScore两种探针归因方法来分解预测到注意力头和源token级别，揭示了探针精度与关系特异性、实体连通度及探针信号集中度之间的明确相关性。

**[Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](interpretability/understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)**

:   本文通过受控合成数据集 Biography-Reasoning 系统分析了 SFT 阶段学习新知识导致的事实幻觉现象，发现幻觉的根本机制是模型对关键实体的注意力被削弱，并提出 KnownPatch——在训练末期注入少量已知知识来恢复注意力模式，有效缓解幻觉。

**[Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](interpretability/understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)**

:   本文利用 Gradiend 梯度可解释性方法研究语言模型预测德语定冠词（der/die/das/den/dem/des）时是基于抽象语法规则还是表层记忆，发现模型至少部分依赖记忆化关联而非严格的规则编码。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)**

:   首篇系统综述数据稀缺条件下LLM强化学习的工作，提出数据中心、训练中心、框架中心三层分类体系，覆盖数据剪枝/合成/压缩、轨迹生成/奖励工程/策略优化、以及自演化/协同演化/多智能体演化等方向。

**[Adaptive Instruction Composition for Automated LLM Red-Teaming](reinforcement_learning/adaptive_instruction_composition_for_automated_llm_red-teaming.md)**

:   提出 Adaptive Instruction Composition (AIC) 框架，利用 Neural Thompson Sampling 在众包有害查询和越狱策略的组合空间中自适应地选择攻击指令，同时优化攻击成功率和多样性，在 Harmbench 上大幅超越已有方法。

**[AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](reinforcement_learning/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)**

:   提出 AJ-Bench，首个系统评估 Agent-as-a-Judge 能力的基准，覆盖搜索、数据系统和 GUI 三个领域共 155 个任务和 516 条标注轨迹，实验表明 Agent-as-a-Judge 比 LLM-as-a-Judge 平均 F1 提升约 13 个百分点。

**[AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)**

:   提出 AttnPO，一个利用模型内在注意力信号进行步级信用分配的低开销过程监督 RL 框架，通过识别 Key-Focus Heads（KFH）区分冗余和关键推理步骤，在大幅缩短推理长度的同时显著提升准确率。

**[Bootstrapping Code Translation with Weighted Multilanguage Exploration](reinforcement_learning/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)**

:   BootTrans 提出了一种自举式多语言代码翻译方法，通过利用单一枢纽语言（Python）的测试用例作为跨语言验证预言，结合双池架构进行经验收集扩展训练数据，并设计语言感知加权机制动态优先处理困难的翻译方向，在 HumanEval-X 和 TransCoder-Test 上显著超越基线。

**[Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)**

:   提出 DYPO（Dynamic Policy Optimization），通过动态难度分级将样本路由到不同优化路径——Hard样本用多教师蒸馏降低SFT偏差、Mid样本用Group Alignment Loss降低RL方差，在数学推理benchmark上平均提升4.8%，OOD任务提升13.3%。

**[CAP: Controllable Alignment Prompting for Unlearning in LLMs](reinforcement_learning/cap_controllable_alignment_prompting_for_unlearning_in_llms.md)**

:   提出 CAP 框架，通过训练轻量 SLM 生成可控的提示前缀来引导冻结的 LLM 选择性遗忘目标知识，无需修改模型参数，实现了可逆、可迁移的 LLM 知识遗忘。

**[CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](reinforcement_learning/ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)**

:   提出 CE-GPPO 算法，通过 stop-gradient 操作重新引入 PPO 裁剪区间外低概率 token 的梯度信号，实现对策略熵的精细化协调控制，在探索-利用之间取得更好平衡。

**[ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](reinforcement_learning/chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)**

:   ChipSeek 提出了一个将 EDA 工具链直接集成到训练循环中的分层奖励 RL 框架，通过课程引导的动态策略优化（CDPO）使 LLM 能够生成同时满足功能正确性和 PPA（功耗-性能-面积）优化的 RTL 代码，在标准基准上达到 SOTA。

**[Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](reinforcement_learning/controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)**

:   提出为多模态对话智能体（MCA）构建紧凑的潜在动作空间来替代巨大的 token 动作空间进行 RL 微调，通过跨模态投影器和循环一致性损失利用配对图文数据和纯文本数据共同构建码本，将动作空间从 152K（词表大小）压缩到 128（码本大小），在两个对话任务上全面超越 token 级 RL 基线。

**[Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](reinforcement_learning/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)**

:   本文提出 Data Mixing Agent，首个基于模型的端到端领域重加权框架，通过在大量数据混合轨迹上使用 CQL 强化学习训练小型代理来学习可泛化的数据混合启发式，在数学推理持续预训练中平衡源领域和目标领域性能，且可泛化到未见过的源领域、目标模型和领域空间。

**[Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](reinforcement_learning/deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)**

:   本文提出 Deliberative Searcher，一个推理优先的框架，将搜索操作集成到 CoT 生成中并保持显式置信度校准，使用自适应拉格朗日乘子的约束 RL 联合优化正确性和可靠性，将 7B 模型的平均"错误-确定"率从基线的 54% 降至 2%。

**[FaithLens: Detecting and Explaining Faithfulness Hallucination](reinforcement_learning/faithlens_detecting_and_explaining_faithfulness_hallucination.md)**

:   本文提出 FaithLens，一个 8B 参数的忠实性幻觉检测模型，通过高质量数据合成+三维过滤（标签正确性、解释质量、数据多样性）进行冷启动 SFT，再用基于规则的强化学习（预测正确性奖励+解释质量奖励）进一步优化，在 12 个任务上超越 GPT-5.2 和 o3，同时提供高质量的解释性输出。

**[Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](reinforcement_learning/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)**

:   本文提出 FTRL 框架，通过五阶段自动化管线构建稳定可控的工具使用训练环境，并设计结合工具调用精度和任务完成度的可验证奖励机制，与偏好优化 RL 算法结合后，在 7B-14B 模型上实现平均超 10% 的工具使用性能提升，甚至超越最强闭源模型。

**[Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](reinforcement_learning/frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)**

:   本文提出 IndiRef 基准测试，用于评估对话系统通过"关系指代"（如"昨天我们去的那个公园旁边的咖啡馆"）建立和利用持久共识（common ground）的能力，发现现有 LLM 在全上下文条件下准确率不超过 50%，并通过合成数据 + GRPO 强化学习训练将性能提升 15-20%。

**[From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](reinforcement_learning/from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)**

:   本文系统综述了 LLM 中不确定性量化从"被动诊断指标"到"主动控制信号"的功能演化，覆盖三大前沿领域：高级推理（引导计算分配和自我纠正）、自主代理（驱动工具使用和信息获取的元认知决策）、以及强化学习（缓解奖励黑客并通过内在奖励实现自我改进）。

**[GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](reinforcement_learning/geora_geometry-aware_low-rank_adaptation_for_rlvr.md)**

:   本文提出 GeoRA，一种专为强化学习可验证奖励（RLVR）设计的低秩适配方法，通过构建几何约束矩阵（融合谱先验和欧几里得先验）提取 RL 更新子空间的主方向进行 SVD 初始化，同时冻结残差矩阵作为结构锚，在 1.5B-32B 参数的 Qwen/Llama 模型上，数学、医学和代码 RLVR 任务中一致超越 LoRA、PiSSA、MiLoRA 等基线，且具备更强的域外泛化和更少的能力遗忘。

**[ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](reinforcement_learning/imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)**

:   ImpRIF 将复杂指令中的隐式推理结构形式化为可验证的显式推理图（ERG），基于此构建大规模单轮/多轮数据并通过 SFT+过程验证 RL 训练，使 4B-32B 模型在五个指令遵循基准上显著超越基座模型，32B 模型甚至超越部分大型商用模型。

**[Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](reinforcement_learning/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)**

:   本文提出 LcRL 框架，通过语言耦合的 GRPO 策略优化和反一致性惩罚奖励，解决多语言 RAG 中的知识偏差和知识冲突问题，在多语言问答任务上取得显著提升。

**[LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](reinforcement_learning/less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)**

:   LENS 发现 RLVR 中许多探索失败并非因为问题难度，而是因为 prompt 中少量（<5%）干扰 token，通过识别和删除这些 token 来提升 rollout 成功率，并将净化 rollout 的学习信号转移到原始噪声 prompt 的策略优化中，平均提升 3.88% 并加速 1.6 倍。

**[Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](reinforcement_learning/optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)**

:   提出 PURPLE 框架，将检索增强 LLM 个性化中的用户画像构建问题建模为上下文老虎机问题，通过 Plackett-Luce 排序模型捕捉记录间依赖关系，以 LLM 对参考回复的 log-likelihood 作为奖励信号，直接优化检索以匹配生成质量。

**[Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](reinforcement_learning/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)**

:   提出 Cold-EQS，一个面向冷启动电商场景的查询建议框架，利用可回答性、事实准确性和信息增益作为内在质量奖励，通过迭代强化学习持续优化查询建议质量，在线 chatUV 提升 6.81%。

**[ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](reinforcement_learning/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)**

:   本文提出 ReRec，一个基于强化微调（RFT）的推荐助手框架，通过双图增强的奖励塑形提供细粒度奖励信号、推理感知的优势估计对推理步骤进行差异化监督、以及在线课程调度器动态调整训练难度，使 LLM 能处理复杂的多步推理推荐查询，在 RecBench+ 基准上显著超越现有方法。

**[Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](reinforcement_learning/right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)**

:   提出 Re-RIGHT 框架，通过三模块奖励（词汇覆盖率+语义保持+连贯性）的 GRPO 训练，用 4B 策略模型在英日韩中四种语言上实现按学习者熟练度等级（CEFR/JLPT/TOPIK/HSK）精确简化文本，超越 GPT-5.2 和 Gemini 2.5 等大模型。

**[RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](reinforcement_learning/rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)**

:   RL-PLUS 提出混合策略优化方法，通过多重重要性采样（MIS）解决外部数据分布不匹配问题，以及探索式优势函数（EAF）引导模型学习低概率但正确的推理路径，成功突破 RLVR 导致的能力边界坍塌，在六个数学推理基准上达到 SOTA（平均 53.4），且跨模型一致提升最高达 69.2%。

**[Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](reinforcement_learning/savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)**

:   本文提出 Savoir，一个基于合作博弈论的社交 RL 框架，结合期望效用（前瞻性评估话语的战略潜力）和 Shapley 值（公理化公平信用分配）解决多轮对话中的信用分配问题，在 SOTOPIA 基准上以 7B 模型达到 SOTA 性能（Hard 设置 Goal 7.18），匹配或超越 GPT-4o 和 Claude-3.5-Sonnet，且大型推理模型（o1、DeepSeek-R1）在社交任务上系统性欠佳。

**[Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)**

:   首次系统研究 LLM 强化学习后训练的缩放行为，在 Qwen2.5 系列(0.5B-72B)上发现性能与训练资源之间遵循幂律关系，且学习效率随模型规模增大呈饱和趋势。

**[Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)**

:   本文指出 RLVR 中传统的 token 级探索-利用权衡是测量方式的伪象，提出在隐状态语义空间中用 Effective Rank (ER) 和其时间导数 (ERV/ERA) 来解耦探索与利用，并据此设计 VERL 方法实现两者的同步提升，在高考数学等基准上获得高达 21.4% 的提升。

**[SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)**

:   本文提出 SpiralThinker，通过在潜在表示空间中进行迭代更新、并与文本推理步骤交替进行的框架实现隐式推理，引入渐进对齐目标确保潜在表示在迭代过程中保持与显式推理的一致性，在数学、逻辑和常识推理任务上超越所有潜在推理基线。

**[STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](reinforcement_learning/stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)**

:   本文提出 STRIDE-ED 框架，通过构建覆盖正/中/负情绪的全面共情策略体系，设计任务对齐的多阶段认知CoT推理，结合策略感知数据精炼和SFT+PPO两阶段训练，在多个开源LLM上实现共情对话SOTA，情感准确率达57.25%，BLEU-4达4.67。

**[Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](reinforcement_learning/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)**

:   全面综述了 LLM 时代表格问答（TQA）研究，从五个维度（表格格式、问题复杂度、答案格式、模态、领域）系统化分类任务设置，按核心挑战（表格理解、复杂查询、大输入、数据异构、知识集成）组织建模方法，覆盖 277 篇论文，并前瞻性讨论了强化学习、可解释性等新兴方向。

**[Understanding Generalization in Role-Playing Models via Information Theory](reinforcement_learning/understanding_generalization_in_role-playing_models_via_information_theory.md)**

:   本文提出首个信息论框架 R-EMID 来量化角色扮演模型（RPM）在用户/角色/对话分布偏移下的性能退化，通过引入推理过程和协同进化强化学习（CoRL）实现准确估计，发现用户偏移是最大的泛化风险，且强化学习是唯一一致有效的改进方法。

**[UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](reinforcement_learning/unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)**

:   本文提出 UniCreative 框架，通过自适应约束偏好优化（ACPO）和自适应标准生成式奖励模型（AC-GenRM），在无需 SFT 和参考答案的条件下统一长文本（规划→写作）和短文本（直接生成）两种创意写作模式，模型涌现出自主区分任务类型的元认知能力。

**[SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](reinforcement_learning/what_if_consensus_lies_selective-complementary_reinforcement_learning_at_test_ti.md)**

:   本文提出 SCRL（Selective-Complementary Reinforcement Learning），一个鲁棒的测试时强化学习框架，通过选择性正伪标签（严格共识标准过滤不可靠多数）和熵门控负伪标签（首次在 TTRL 中引入负监督信号来修剪错误轨迹）缓解标签噪声放大问题，在 AIME25 上比 TTRL 提升高达 10.1 个百分点。

---

## 🎵 音频/语音 { #audio_speech }

**[Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](audio_speech/affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)**

:   本文提出 Affectron 框架，通过情感驱动的 Top-K NV 匹配和情感感知的 Top-K 路由两个训练时增强策略，在小规模开源解耦语料上实现了多样且情感对齐的非语言发声（如笑声、叹息）合成，显著超越了基于纯语言预训练的 VoiceCraft 基线。

**[Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](audio_speech/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)**

:   Alexandria 构建了覆盖 13 个阿拉伯国家、11 个社会影响领域、107K 轮次的多轮对话方言阿拉伯语-英语平行数据集，通过社区驱动的人工翻译与修订流程，为方言阿拉伯语机器翻译提供了前所未有的细粒度训练和评测资源，并在 24 个 LLM 上进行了系统性基准评估。

**[An Exploration of Mamba for Speech Self-Supervised Models](audio_speech/an_exploration_of_mamba_for_speech_self-supervised_models.md)**

:   首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。

**[Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)**

:   本文提出锚定循环生成（ACG）范式，通过在自回归过程中用已确认的音乐内容作为锚点来校准生成方向，有效缓解长序列符号音乐生成中的误差累积问题，并构建了层次化框架Hi-ACG实现从全局到局部的音乐生成。

**[Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](audio_speech/beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)**

:   揭示当前 AudioLLM 的感知弱点源于 ASR 中心的训练范式（系统性抑制副语言和非语言信息），提出 Unified Audio Schema（UAS）将音频信息结构化为转录、副语言和非语言事件三个维度的 JSON 格式，在 MMSU 基准上感知精度提升 10.9% 同时保持推理能力。

**[Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](audio_speech/breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)**

:   提出 AHD（Anchor-based History-stable Decoding），一种无需训练的即插即用动态解码策略，通过动态锚点回溯历史轨迹判定扩散LLM中跨块稳定token，实现早期解锁，在BBH上减少80%解码步数的同时提升3.67%性能。

**[Computational Narrative Understanding for Expressive Text-to-Speech](audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)**

:   本文从有声书虚构作品中提取角色直接引语，构建了大规模表达性语音数据集 LibriQuote（5.3K 小时引语 + 12.7K 小时叙述），并用语音动词和副词伪标签标注说话风格，实验表明在 flow-matching 模型上微调可同时提升表达性和可懂度，且 LibriQuote-test 构成了一个具有挑战性的表达性 TTS 基准。

**[DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](audio_speech/dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)**

:   本文构建 DIA-HARM，首个跨 50 种英语方言评估虚假信息检测鲁棒性的基准，揭示人类撰写的方言内容导致检测性能下降 1.4-3.6% F1，微调 Transformer 大幅优于零样本 LLM（96.6% vs 78.3%），且部分模型在混合内容上出现超过 33% 的灾难性退化。

**[Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](audio_speech/do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)**

:   本文通过逐层oracle干预实验揭示了大语音语言模型（LSLM）中语音token表示的结构化冗余层次——浅层编码必要声学细节而深层极度冗余——并提出Affinity Pooling这一免训练的基于相似度的token合并机制，在减少27.48% FLOPs的同时保持竞争力的准确率。

**[SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](audio_speech/generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)**

:   SEPT 通过利用 LLM 生成语义邻居并设计带边距约束的语义扩展损失来正则化提示嵌入空间，显著缓解了音频语言模型（ALM）提示调优中的 Base-New Tradeoff 问题，建立了 ALM 提示泛化的首个系统性评估基准。

**[HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](audio_speech/halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)**

:   本文提出 HalluAudio，首个大规模跨领域（语音/环境声/音乐）的音频幻觉检测基准，包含 5000+ 人工验证的 QA 对和系统化的对抗性提示设计，通过多维指标（准确率/幻觉率/Yes-No偏差/拒绝率/错误类型）评估主流 LALM，揭示了当前模型在声学锚定、时间推理和音乐属性理解方面的显著缺陷。

**[Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](audio_speech/hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)**

:   本文对两种音系极端复杂的低资源濒危东高加索语言（Archi和Rutul）进行音素级ASR分析，发现音素识别准确率与训练频率呈S型学习曲线关系，许多归因于音系复杂性的错误实际上更多源于数据稀缺。

**[How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](audio_speech/how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)**

:   本文通过三个语用任务（虚假预设、反预设、演绎推理）系统对比 14 个 LLM 作为"语用听者"（判断语用适当性）和"语用说者"（生成语用适当的语言）的表现，发现普遍存在的听者-说者不对称：多数模型作为判断者远优于生成者，且项目级分析表明正确判断不能可靠预测成功生成。

**[Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)**

:   构建 Jamendo-MT-QA，一个包含 36,519 个比较问答对（覆盖 12,173 个音轨对）的多音轨比较音乐问答基准，首次系统评估音频-语言模型在跨音轨比较推理上的能力，揭示现有模型在句子级比较生成上的显著不足。

**[Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)**

:   本文提出 CmIR（因果模态不变表示学习），基于因果推理理论将每种模态显式解纠缠为因果不变表示和环境特定虚假表示，通过不变性约束+互信息约束+重建约束的优雅目标函数确保不变表示具有跨环境的稳定预测关系，在多模态情感/幽默/讽刺检测上取得 SOTA，尤其在 OOD 和噪声场景下表现突出。

**[MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](audio_speech/mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)**

:   本文构建了首个面向中国古典文学的大规模（119小时、22000条样本）全版权音频语料库 MCGA，涵盖赋、诗、文、词、曲五大文体和六项语音任务（ASR/S2TT/SEC/SQA/SU/SR），并通过评测 10 个多模态大模型揭示了当前模型在古典文学语音理解上的显著不足。

**[Multimodal In-Context Learning for ASR of Low-Resource Languages](audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)**

:   系统研究多模态上下文学习（MICL）能否使语音 LLM 学习未见过的濒危语言，并提出基于 MICL 的假设选择系统，结合声学模型与语音 LLM 的互补优势，在三种濒危语言上显著提升 ASR 性能。

**[Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](audio_speech/music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)**

:   本文作为音乐视听问答（Music AVQA）领域首篇综合综述，系统分析了数据集演进和方法设计，论证了专门的输入处理、时空架构设计和音乐领域知识对该任务至关重要，通用多模态模型不足以应对音乐表演的独特挑战。

**[MSU-Bench: Musical Score Understanding Benchmark](audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)**

:   MSU-Bench 是首个针对完整乐谱理解的人工标注基准，包含 150 首作品的 1800 个生成式 QA 对，覆盖四级难度，评估揭示了 LLM/VLM 在乐谱定位和幻觉方面的严重不足，而 ABC 记谱法的文本输入显著缓解了这些问题。

**[Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](audio_speech/pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)**

:   本文提出 Pseudo2Real，一种参数空间校正方法，通过在源域中计算真实标签模型与伪标签模型的权重差得到"校正向量"，将其应用于目标域伪标签微调模型以纠正系统性伪标签偏差，在 AfriSpeech-200 的十种非洲口音上最高实现 35% 相对 WER 降低。

**[Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](audio_speech/retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)**

:   本文提出R2ScP框架，将AVQA中缺失模态处理范式从传统的生成式补全转变为基于检索的恢复，通过跨模态检索和上下文感知自适应净化机制消除检索噪声，在模态不完整场景下显著提升了问答性能。

**[Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](audio_speech/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)**

:   本文提出一种基于指令引用的提示注入防御方法，不压制 LLM 的指令遵循能力，而是让模型在响应中引用正在执行的指令，然后通过标签过滤移除与原始指令不相关的响应，在部分场景下将攻击成功率降至接近 0%。

**[Splits! Flexible Sociocultural Linguistic Investigation at Scale](audio_speech/splits_flexible_sociocultural_linguistic_investigation_at_scale.md)**

:   提出构建社会语言学"沙盒"的方法，从 Reddit 构建了按人口统计群体和讨论话题双重切分的 970 万帖子数据集 Splits!，并设计了基于 lift 和 triviality 的两阶段过滤流程，从 2.3 万条 LLM 生成的候选假设中高效筛选出值得深入研究的社会文化语言现象。

**[Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](audio_speech/still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)**

:   针对语音助手无法区分第三方打断（TPI）与主用户发言的问题，提出包含88K训练实例的TPI-Train数据集和TPI-Bench评测框架，通过说话人感知的困难负样本挖掘策略消除语义捷径学习，使模型真正依赖声学线索进行打断检测。

**[StressTest: Can YOUR Speech LM Handle the Stress?](audio_speech/stresstest_can_your_speech_lm_handle_the_stress.md)**

:   提出 StressTest 基准评估语音语言模型（SLMs）对句子重音含义的理解能力，发现现有模型几乎无法基于重音模式推理说话者意图，并通过合成数据管线 Stress-17k 训练的 StresSLM 在重音检测和推理任务上大幅超越前沿模型。

**[TellWhisper: Tell Whisper Who Speaks When](audio_speech/tellwhisper_tell_whisper_who_speaks_when.md)**

:   本文提出TellWhisper，通过设计时间-说话人感知的旋转位置编码（TS-RoPE）将说话人身份和时间信息统一编码到语音编码器的自注意力中，配合双曲空间说话人日志模型（Hyper-SD），实现了对"谁在何时说了什么"的联合建模，在多说话人ASR任务上取得最优性能。

**[Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](audio_speech/temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)**

:   提出 TCD，一种无训练的推理时解码方法：通过对比原始音频和时间模糊慢速路径的 logits 差异，配合稳定性引导的模糊窗口和不确定性门控，使统一音频语言模型更好地利用瞬态声学线索，在 MMAU 和 AIR-Bench 上一致提升。

**[Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)**

:   本文提出FCaps大规模数据集（47k小时语音、19M细粒度标注）和CLSP对比学习模型，通过端到端标注管线和细粒度多粒度对比监督，实现了首个能统一表征全局和细粒度语音风格的语音-文本对齐模型。

**[When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](audio_speech/when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)**

:   本文为Position Paper，论证音频平台上的虚假信息在本质上不同于文本虚假信息——它同时具有口语性（prosody、pacing、emotion）和对话性（多轮、多说话人、跨集节），现有以文本为中心的事实核查流水线无法有效处理，需要围绕音频特有属性重新设计验证框架。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction](multilingual_mt/a_multilingual_dataset_and_empirical_validation_for_the_mutual_reinforcement_eff.md)**

:   构建首个多语言MRE Mix数据集（MMM，21个子集覆盖英中日），并通过大规模消融实验系统验证了词级与文本级信息抽取任务的互增强效应（MRE）跨语言普遍存在。

**[Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](multilingual_mt/beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)**

:   构建非字面翻译元评估数据集 MENT（7,530 条人工标注），揭示传统指标和 LLM-as-Judge 在非字面翻译评估上的不可靠性，并提出 RATE 智能体评估框架，通过反思核心智能体动态调用子智能体，提升 3.2+ 点人类判断相关性。

**[BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](multilingual_mt/bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)**

:   首篇专门针对印度语言NLP资源的统一综述，覆盖200+数据集、50+基准、100+模型/工具，按17个任务类别组织（从核心语言处理到社会文化任务），系统分析了语言覆盖不均、标注碎片化、评估不一致等持续挑战。

**[Efficient Training for Cross-lingual Speech Language Models](multilingual_mt/efficient_training_for_cross-lingual_speech_language_models.md)**

:   本文提出CSLM，一种高效训练跨语言语音LLM的方法，通过新颖的对齐策略实现跨模态和跨语言对齐，并引入语音-文本交织链式模态生成来提升质量和降低延迟，无需大规模语音数据即可扩展到新语言。

**[Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)**

:   本文提出两阶段持续微调（CFT）框架——先在英语指令数据上微调，再在多语言数据上微调——发现阶段间数据集的指令相似性是决定英语能力是否退化的关键因素，并通过生成式重放和启发式层冻结有效缓解了不相似数据集导致的表示漂移和英语遗忘。

**[IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](multilingual_mt/indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)**

:   提出 IndoTabVQA，一个针对印尼语（Bahasa Indonesia）文档表格的跨语言视觉问答基准，包含 1593 张文档图像和四种语言（印尼语/英语/印地语/阿拉伯语）的 QA 标注，揭示了 VLM 在低资源语言和跨语言表格理解上的显著性能差距，微调+空间先验可带来最高 48.5% 的 In-Match 准确率。

**[Just Use XML: Revisiting Joint Translation and Label Projection](multilingual_mt/just_use_xml_revisiting_joint_translation_and_label_projection.md)**

:   提出 LabelPigeon，一种基于 XML 标签的联合翻译与标签投影方法，通过在高质量 XML 标记平行语料上微调 NLLB-200 翻译模型，在 11 种语言上超越所有基线并主动提升翻译质量，在下游跨语言 NER 任务中实现最高 +40.2 F1 的提升。

**[Language Models Entangle Language and Culture](multilingual_mt/language_models_entangle_language_and_culture.md)**

:   本文通过基于 WildChat 数据集构建的通用建议类问题评估多语言 LLM，发现不同语言查询会导致回答质量和文化上下文的系统性差异——低资源语言的回答质量显著低于英语，且语言选择会隐式地改变回答中使用的文化信息，在翻译版 CulturalBench 上验证了语言与文化在 LLM 中的纠缠关系。

**[Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)**

:   本文提出 XBridge，一种将预训练多语言编码器-解码器翻译模型（如 NLLB）与英语为中心的 LLM 组合的架构——编码器负责多语言理解、LLM 负责知识推理、解码器负责多语言生成，通过轻量级映射层和最优传输对齐实现跨模型语义桥接，在低资源和未见语言上显著优于基线。

**[Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](multilingual_mt/location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)**

:   本文提出 LocQA 基准（12 种语言、49 个地区、2156 个地域相关问答），通过地域模糊问题（如"紧急电话号码是多少？"）揭示 LLM 的隐式偏差：跨语言上存在持续的美国中心默认行为（模型回答的 50% 包含美国答案 vs 数据中仅 26%），语言内部存在人口规模驱动的"人口概率引擎"效应，且指令微调加剧了全球偏差。

**[Lost in Translation: Do LVLM Judges Generalize Across Languages?](multilingual_mt/lost_in_translation_do_lvlm_judges_generalize_across_languages.md)**

:   本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。

**[LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](multilingual_mt/lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)**

:   提出LQM（Linguistically Motivated Multidimensional Quality Metrics），一个六层语言学驱动的机器翻译错误分类体系（社会语言学→语用学→语义→形态句法→正字法→字形），并在7种阿拉伯方言上构建3850句双向平行语料库，通过专家标注6113个错误跨度揭示现有MT系统在方言和文化感知翻译上的系统性不足。

**[Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](multilingual_mt/mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)**

:   针对孟加拉语预训练模型在下游分类任务中的外在性别偏见，提出 RandSymKL 方法，通过随机化交叉熵损失和对称 KL 散度联合优化，在保持分类准确率的同时有效缩小性别间预测差异。

**[MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](multilingual_mt/morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)**

:   本文提出 MORPHOGEN，一个涵盖法语/阿拉伯语/印地语的大规模性别感知形态学生成基准（共 20,328 句对），定义了 GENFORM 任务（将第一人称句子改写为相反性别），并提出 SGA/GIoU/CGA 三个评估指标，对 15 个多语言 LLM 的基准测试揭示了模型在复杂形态推理、性别偏差和多实体干扰方面的系统性不足。

**[Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_mt/multilingual_language_models_encode_script_over_linguistic_structure.md)**

:   本文通过 LAPE 指标和稀疏自编码器系统分析多语言 LM 中的语言关联单元，发现这些单元主要由正字法（书写系统）驱动而非抽象语言结构：罗马化转写激活几乎完全不重叠的神经元集合，词序打乱影响甚微，类型学信息仅在深层逐渐可访问，因果干预表明功能重要性与表面形式不变性相关。

**[No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](multilingual_mt/no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)**

:   本文证明没有一种提示策略在所有语言和任务上普遍最优，提出将策略选择建模为学习决策问题，用轻量级分类器为每个实例预测最优策略，在四个基准上显著优于固定策略。

**[Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](multilingual_mt/prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)**

:   本文提出 NOVA-ARC，首次将多语言语音情感识别（SER）建模为从标注的非语言发声（NVV）到未标注的语言语音（UVS）的无监督迁移问题，通过双曲空间中的韵律向量量化编码本、双曲情感透镜和最优传输原型对齐实现跨模态情感迁移，在 6 个数据集上验证了非语言→语言迁移的可行性和优越性。

**[SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](multilingual_mt/serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)**

:   提出 SERM 框架，通过多智能体样本挖掘器和多智能体相关性标注器，从大规模真实查询流中持续自进化搜索相关性模型，经三轮迭代在工业搜索平台上实现 NDCG@1 提升 +2.99，并在在线 A/B 测试中显著提升用户留存率。

**[Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](multilingual_mt/syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)**

:   本文首次探索将 Universal Dependencies 句法信息作为上下文学习的增强源用于低资源科普特语到英语的机器翻译，发现虽然句法信息单独不如词典有效，但将词典与句法信息结合（LEX+SYN）在所有模型上取得最佳效果，Gemma-27B 的 BERTScore F1 达到 0.8746（+0.0361）。

**[The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](multilingual_mt/the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)**

:   本文提出GaoYao基准，包含182.3K样本、26种语言和51个国家/地区，通过三层文化评估框架（通用多语言/跨文化/单文化）和九个认知子层，结合人工本地化的主观测试集和专家验证的跨文化合成数据集SuperBLEnD，深度诊断20+旗舰与紧凑型LLM的多语言能力，揭示了显著的地理数字鸿沟和任务能力分层。

**[Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration](multilingual_mt/unlocking_the_edge_deployment_and_ondevice_acceleration_of_multi-lora_enabled_on.md)**

:   本文提出面向三星 Galaxy S24/S25 的端侧 LLM 部署框架，通过 LoRA 权重作为运行时输入实现动态任务切换、多流并发 token 生成减少风格变体延迟达 6 倍、无草稿模型的 Dynamic Self-Speculative Decoding 加速解码达 2.3 倍，在 9 语言 8 任务上实现 4-6 倍整体优化。

**[Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](multilingual_mt/vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)**

:   本文发现 LLM 在嵌入空间中将词形变化（如 walk→walked）编码为线性方向，基于此提出组合式词表设计：用基础词+变换向量的加法组合替代为每个表面形式分配独立 token，在冻结预训练骨干的前提下仅训练小型适配模块，释放 10-40% 的词表槽位用于多语言扩展，同时几乎不影响下游性能。

**[What Factors Affect LLMs and RLLMs in Financial Question Answering?](multilingual_mt/what_factors_affect_llms_and_rllms_in_financial_question_answering.md)**

:   本文系统研究了提示方法、Agent 框架和多语言对齐方法对 LLM 和 RLLM（推理型大模型）在金融问答任务上的影响，发现现有方法本质上是通过模拟 Long CoT 来提升 LLM 性能，但对已具备 Long CoT 能力的 RLLM 效果有限。

---

## 🔒 LLM 安全 { #llm_safety }

**[Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](llm_safety/adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)**

:   提出自适应文本匿名化框架，通过进化式提示优化自动为LLM发现任务特定的匿名化指令，在多个隐私-效用权衡场景中超越手工设计的策略，且可在开源模型上运行。

**[AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](llm_safety/agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)**

:   AGSC 提出了一个针对长文本生成的不确定性量化框架，通过 NLI 中立概率触发自适应粒度分解（减少 60% 推理时间），并使用 GMM 软聚类捕捉潜在语义主题进行主题感知的加权聚合，在 BIO 和 LongFact 基准上达到 SOTA 的事实性相关性。

**[Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge](llm_safety/beyond_end-to-end_dynamic_chain_optimization_for_private_llm_adaptation_on_the_e.md)**

:   提出 ChainFed，一种打破内存墙的链式联邦微调范式，通过逐层顺序训练-冻结适配器使资源受限边缘设备也能参与 LLM 微调，结合动态层协调、全局感知优化和功能导向自适应三项技术，平均准确率提升最高 46.46%。

**[Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](llm_safety/compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)**

:   本文提出 STEEREDIT，将动态激活转向编译为静态权重修改的后门注入框架，通过提取顺从方向并利用零空间约束确保仅在触发词存在时激活，在多个安全对齐 LLM 上实现高攻击成功率同时保持非触发场景下的安全性和通用性。

**[De-Anonymization at Scale via Tournament-Style Attribution](llm_safety/de-anonymization_at_scale_via_tournament-style_attribution.md)**

:   本文提出 DAS（De-Anonymization at Scale），一种基于 LLM 的大规模作者去匿名化方法，采用锦标赛式淘汰策略+密集检索预过滤+多轮投票聚合，可在数万候选文本中进行作者匹配，揭示了 LLM 对匿名平台（如双盲评审）的隐私威胁。

**[DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](llm_safety/duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)**

:   本文提出 DUET，一个结合直接代码执行和 LLM 伪代码执行的双路框架，通过功能多数投票融合两种互补的执行路径——前者在代码正确时可靠但受实现错误影响，后者绕过实现细节但可能产生执行幻觉——在 LiveCodeBench 测试输出预测上提升 Pass@1 13.6 个百分点。

**[Enhancing Hallucination Detection via Future Context](llm_safety/enhancing_hallucination_detection_via_future_context.md)**

:   本文提出利用采样生成的"未来上下文"（后续句子）来增强黑盒场景下的幻觉检测，利用幻觉一旦出现就倾向于持续传播的"滚雪球效应"，在 SelfCheckGPT 和 SC 等多种采样方法上一致提升检测性能。

**[FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](llm_safety/facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)**

:   本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板），实现快速、准确、隐私合规的查询聚焦表格摘要，在 FeTaQA、QTSumm 和 QFMTS 三个基准上全面超越基线。

**[Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)**

:   提出 Entropy-guided Token Weighting (ETW)，利用预测分布的熵值作为 token 信息量的代理指标，选择性地对信息性 token 施加更强的遗忘惩罚，在有效遗忘目标知识的同时更好地保持模型通用能力。

**[Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](llm_safety/indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)**

:   本文构建了首个多印度语言的 CodecFake 检测基准 ICF，并提出 SATYAM——一个双曲音频大语言模型，通过在双曲空间中用 Bhattacharyya 距离对齐语义和副语言表示再与提示对齐，仅训练 3.75M 参数即达到 98.32% 的检测准确率。

**[Jailbreaking Large Language Models with Morality Attacks](llm_safety/jailbreaking_large_language_models_with_morality_attacks.md)**

:   本文构建10.3K道德攻击数据集（价值模糊+价值冲突），通过四种对抗策略操纵LLM道德判断，发现LLM和guardrail模型对道德攻击极度脆弱，且更大模型反而更容易被攻破。

**[KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](llm_safety/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)**

:   提出知识坐标条件化预训练（KoCo），将每个文档映射为三维语义坐标（来源、内容、稳定性），作为文本前缀注入预训练，使模型获得显式的上下文感知能力，在 10 个下游任务上提升性能、加速收敛约 30%，并有效缓解幻觉。

**[Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](llm_safety/maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)**

:   本文提出 PALU（Prefix-Aware Localized Unlearning），从时间和词表两个维度实现局部化的熵最大化遗忘：在时间维度仅对敏感前缀 token 施加遗忘目标，在词表维度仅对 top-K logits 进行平坦化，以最小的参数扰动实现高效遗忘并保持模型通用能力。

**[MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](llm_safety/meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)**

:   本文提出MeasHalu框架，通过细粒度测量幻觉分类法和两阶段优化（推理感知SFT+幻觉靶向GRPO奖励）缓解LLM在科学测量抽取中的幻觉，在MeasEval上显著超越基线。

**[Synthia: Scalable Grounded Persona Generation from Social Media Data](llm_safety/synthia_scalable_grounded_persona_generation_from_social_media_data.md)**

:   提出 Synthia 框架，基于真实社交媒体帖子（Bluesky）生成有根据的 LLM 人格叙事，在社会调查对齐度上比 SOTA 提升最高 11.6%，同时使用更小的模型，并保留社交网络拓扑结构支持网络感知分析。

**[Topic-Based Watermarks for Large Language Models](llm_safety/topic-based_watermarks_for_large_language_models.md)**

:   本文提出基于主题的轻量水印方案 TBW，将词表按语义主题聚类为"绿色列表"（而非随机分区），根据输入提示选择语义对齐的主题列表进行 logit 偏置，在保持与无水印文本相当的困惑度的同时，显著提升了对释义和词汇扰动攻击的鲁棒性。

**[Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](llm_safety/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)**

:   从理论上分析了多 Token 预测（MTP）如何通过梯度耦合机制诱导表示收缩性从而促进信念状态的涌现，但同时揭示了 MTP 的"结构性幻觉"问题（隐空间中的非法捷径），并提出 LSE-MTP 框架通过隐一致性损失和语义锚定损失将预测锚定到真实隐状态轨迹，在合成图和真实曼哈顿出租车导航上显著改善路径合法性和鲁棒性。

**[Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)**

:   本文发现 LLM 内部编码真实性信号存在两条不同的信息通路：Question-Anchored（依赖问题到回答的信息流）和 Answer-Anchored（从生成答案本身提取自包含证据），两者与知识边界紧密关联，并据此提出 Mixture-of-Probes 和 Pathway Reweighting 两种通路感知的幻觉检测方法，AUC 提升达 10%。

**[Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](llm_safety/who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)**

:   本文首次系统分析 LLM 在人口统计条件下生成定向消息时的偏见行为，提出 Persuasion Bias Index (PBI) 指标，发现 GPT-4o/Llama/Mistral 在气候传播中对男性和年轻人使用更强势的说服策略，且上下文提示会系统性地放大这些差异。

**[Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](llm_safety/why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)**

:   本文首次系统研究了 SFT 中的"不完全学习现象"（ILP）——即模型收敛后仍无法正确复现部分训练数据，识别了五种反复出现的原因（知识缺失、知识冲突、数据内部矛盾、左侧遗忘、不充分优化），并提出诊断框架和针对性缓解策略。

**[XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)**

:   提出 XMark，一种基于 Leave-one-Shard-out（LoSo）策略和 evergreen list 的多比特文本水印方法，通过跨多个词表排列的绿色列表交集和约束 token-shard 映射矩阵，在保持文本质量的同时显著提升了有限 token 条件下的解码准确率。

---

## 💻 代码智能 { #code_intelligence }

**[Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation](code_intelligence/across_programming_language_silos_a_study_on_cross-lingual_retrieval-augmented_c.md)**

:   首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。

**[CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)**

:   本文提出 CodeRL+，将执行语义对齐集成到 RLVR 训练管道中，通过让模型推断变量级执行轨迹来弥合代码文本表示与执行语义之间的差距，在代码生成上平均 pass@1 提升 4.6%，在代码推理和测试输出生成基准上分别提升 15.5% 和 4.4%。

**[CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)**

:   提出 CodeWiki，一个基于层次化分解和递归多智能体处理的开源框架，用于自动生成仓库级代码文档，并构建了 CodeWikiBench 基准，在七种编程语言上以 68.79% 的质量分数超越了闭源系统 DeepWiki（64.06%）。

**[CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](code_intelligence/collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)**

:   本文提出 CollabCoder，一个计划-代码共演化框架，通过协作决策模块（CDM）判断错误应在计划层还是代码层修复，结合推理轨迹模块（RT）实现从错误中学习的自改进调试，在复杂编程基准上比强基线提升 11-20%，同时减少 4-10 次 API 调用。

**[DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](code_intelligence/deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)**

:   提出 DeepGuard，通过注意力机制聚合 Transformer 上层多层表示克服"最终层瓶颈"问题，结合多目标训练和轻量推理时安全引导策略，在 5 个代码 LLM 上将安全-正确生成率平均提升 11.9%。

**[EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)**

:   提出 EET——一种基于历史经验驱动的早停方法，在补丁生成和补丁选择阶段识别无效迭代并提前终止，将 SE Agent 总成本降低 19%-55%（平均 32%），同时几乎不损失任务性能（最多 0.2%）。

**[From Charts to Code: A Hierarchical Benchmark for Multimodal Models](code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)**

:   本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。

**[From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation](code_intelligence/from_if-statements_to_ml_pipelines_revisiting_bias_in_code-generation.md)**

:   揭示LLM代码生成的偏差评估严重低估了实际风险：在ML流水线生成中，敏感属性出现在87.7%的特征选择中（vs 条件语句中的59.2%），且模型能正确排除无关特征但仍选择保留种族、性别等敏感属性，显示出系统性的隐性歧视。

**[LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)**

:   本文构建了首个针对逻辑漏洞的修复评估框架 LogicEval 和数据集 LogicDS（61 个真实逻辑漏洞 + 61 个合成 Java 样本），系统评估了传统 AVR 工具和 LLM 在修复逻辑漏洞上的能力，发现 LLM 在提供辅助信息时表现最佳但整体修复率仍然很低（61 个真实样本中仅正确修复 5 个），并识别了提示敏感性、上下文丢失和补丁定位困难等关键瓶颈。

**[MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)**

:   MARS2 提出多智能体强化树搜索框架，将多个独立优化的策略嵌入共享搜索树中协作探索，通过 Thompson 采样选择智能体-节点对、树一致性奖励塑形和路径级组优势估计，在代码生成基准上一致提升单模型 Pass@1 最高 8.0%、系统级 Pass@1(MCTS) 最高 6.5%。

**[OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)**

:   本文提出 OmniDiagram，一个统一的图表代码生成框架，覆盖 LaTeX/Mermaid/PlantUML 三种语言和图表转代码/图表编辑/文本转代码三种任务，并引入基于视觉问答的 Viva 奖励机制来指导 RL 训练，在多个基准上达到 SOTA。

**[Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)**

:   本文揭示前沿 LLM 在调试任务中的"重生成"倾向——通过引入 PDB 框架和编辑级精度/bug 级召回指标，发现 GPT-5.1-Codex 等模型虽能通过 76% 以上单元测试，但编辑精度不足 45%，且迭代和 agent 调试策略也无法显著改善精度。

**[QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)**

:   本文识别了 LLM 代码修复中的"过度编辑"问题——模型倾向于重写大量代码而非精确定位和修复 bug，提出 PRepair 框架，通过 Self-Breaking（多样化 bug 注入）和 Self-Repairing（编辑感知 GRPO 训练），显著提升修复精确度同时保持正确性，并加速推测解码推理。

**[ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)**

:   本文提出 ReFEree，一种针对真实世界代码摘要的无参考、细粒度事实一致性评估方法，定义四类不一致标准并在句段级别评估，结合依赖信息搜索机制，在 Python 和 Java 上相比前 SOTA 提升 15-18% 的人类判断相关性。

**[River-LLM: Large Language Model Seamless Exit Based on KV Share](code_intelligence/river-llm_large_language_model_seamless_exit_based_on_kv_share.md)**

:   本文提出 River-LLM，一个无需训练的框架，通过构建轻量级 KV 共享退出通道（Exit River）解决了 decoder-only 架构中 Early Exit 的 KV Cache 缺失问题，利用状态转换相似度引导退出决策，实现 1.71×-2.16× 的实际推理加速且保持近无损生成质量。

**[Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](code_intelligence/sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)**

:   本文提出区分词汇召回（逐字检索代码）和语义召回（理解代码运行语义）两种能力，发现前沿 LLM 在长上下文中词汇召回近乎完美但语义召回严重退化，并引入 SemTrace 基准揭示现有评估严重低估了语义理解失败的程度。

**[SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)**

:   本文提出 SOCIA-EVO，一种将自动化模拟器构建重新定义为双锚进化过程的 LLM 智能体框架，通过静态蓝图（Blueprint）锚定经验约束、双层优化解耦结构修正与参数校准、自我策划的策略剧本（Playbook）管理修复假说并通过执行反馈进行贝叶斯加权检索，在用户建模、口罩佩戴扩散和个人出行三个模拟任务上显著超越 Reflexion、G-SIM 等基线。

**[SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)**

:   SolidCoder 通过 S.O.L.I.D. 架构（Shift-left Planning、Oracle-based Assertions、Live Execution、Intermediate Simulation、Defensive Accumulation）将代码验证从 LLM 的"想象执行"转变为"真实执行"，在 GPT-4o 上达到 HumanEval 95.7%、CodeContests 77.0%、APPS 26.7% 的 pass@1 性能。

**[StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](code_intelligence/storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)**

:   本文提出 StoryCoder，一种将代码生成问题重构为连贯自然语言叙事的提示框架，通过任务概述、约束条件和示例三个叙事组件引导 LLM 进行结构化推理，在 11 个模型上平均提升零样本 pass@10 达 18.7%。

**[The Path Not Taken: Duality in Reasoning about Program Execution](code_intelligence/the_path_not_taken_duality_in_reasoning_about_program_execution.md)**

:   本文提出程序执行推理的对偶性概念，通过DexBench基准（445个配对实例）联合评估LLM的正向执行推理（预测给定输入下的代码覆盖）和反向反事实推理（推断使执行流转向目标分支的输入变异），发现单一方向上的强表现不能转化为联合评估下的成功，揭示了模型对程序因果理解的不足。

---

## 🎁 推荐系统 { #recommender }

**[Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](recommender/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)**

:   提出 TravelBench，首个融合真实用户查询、隐式用户偏好、多轮交互、不可解任务识别和10种真实工具的旅行规划基准，通过沙箱环境实现可复现评估，揭示前沿模型在不同能力维度上表现不均衡。

**[Content Fuzzing for Escaping Information Cocoons on Social Media](recommender/content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)**

:   提出 ContentFuzz，一个从内容创作者视角出发的置信度引导模糊测试框架，通过 LLM 改写帖子使其在保持人类解读含义不变的前提下改变机器推断的立场标签，从而突破社交媒体信息茧房。

**[Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](recommender/decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)**

:   提出 DECISIVE 交互式决策框架，通过从非结构化文档中提取客观选项评分矩阵，结合贝叶斯偏好推断自适应选择成对比较问题高效学习用户潜在偏好向量，在最小化用户交互负担的同时实现透明个性化推荐，决策准确率比强基线提升最高 20%。

**[From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)**

:   本文提出Memora基准和FAMA指标，将长期记忆评估从浅层事实检索扩展到跨越数周至数月的记忆整合与突变处理，揭示现有LLM和记忆agent在处理频繁知识更新时的系统性失败。

**[HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](recommender/harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)**

:   提出 HARPO 框架，将对话推荐重新定义为以推荐质量为优化目标的结构化决策问题，通过层次化偏好学习、基于价值网络的树搜索推理、虚拟工具操作和多智能体精炼四大组件，在 ReDial、INSPIRED 和 MUSE 三个基准上显著超越现有方法。

**[HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](recommender/horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)**

:   本文提出 HORIZON，首个全开源的大规模跨领域长期推荐基准，基于 Amazon Reviews 合并构建包含 54M 用户和 35M 商品的统一交互历史，设计了沿时间轴和用户维度解耦的四象限评估协议，揭示了 BERT4Rec 等模型在分布内表现强劲但在时序外推和未见用户场景下显著退化的现象，且 LLM 在用户行为建模上并未一致优于专用架构。

**[IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](recommender/icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)**

:   本文提出 IceBreaker，通过两步"握手"——共鸣感知兴趣蒸馏捕获触发兴趣 + 交互导向启动语生成配合个性化偏好对齐——解决对话智能体的"首条消息壁垒"，在全球最大对话产品之一的 A/B 测试中提升用户活跃天数 +1.84‰ 和点击率 +94.25‰。

**[Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)**

:   本文提出 ReCAP 框架，通过可训练的查询生成器和用户画像生成器，从用户历史记录中检索与说服相关的信息并构建上下文感知的用户画像，显著提升个性化说服力预测的效果。

**[Personalized Benchmarking: Evaluating LLMs by Individual Preferences](recommender/personalized_benchmarking_evaluating_llms_by_individual_preferences.md)**

:   本文对 Chatbot Arena 的 115 名活跃用户进行个性化排名分析，发现 Bradley-Terry 个性化排名与全局排名的平均 Spearman 相关仅 ρ=0.04（57% 用户近零或负相关），证明聚合基准无法反映大多数用户的个体偏好，并通过话题+风格特征成功预测了用户特定的模型排名。

**[Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](recommender/scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)**

:   本文系统综述了音译（transliteration）在跨语言 NLP 中的演变角色，提出五大动机分类（命名实体/OOV处理、代码混合、跨文字相似性利用、英语中心迁移、统一预处理），比较了六种整合方式的优劣，并在现代 LLM 语境下讨论了音译是否仍然必要。

**[What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](recommender/what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)**

:   NOVELQR 提出了一个新颖性驱动的引用推荐框架，通过生成式标签代理构建深层语义知识库实现语义理性检索，并用 token 级新颖性估计器缓解自回归续写偏差，在双语基准上显著提升推荐质量。

**[What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](recommender/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)**

:   本文揭示现有 LLM 推荐系统的二元偏好建模丢失了偏好强度和时间上下文两个关键信息，提出 RecPO 框架通过自适应奖励边际将这两个因素纳入偏好优化，在五个数据集上显著超越 S-DPO 等基线。

**[Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](recommender/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)**

:   SiPeR 通过场景转换估计（"Where"）和贝叶斯逆推理（"What"）两个机制，解决情景对话推荐中用户偏好随环境动态变化且常常隐式表达的挑战，在 SIMMC 2.1 和 SCREEN 上分别提升 10.9% 和 10.6%。

---

## 🎨 图像生成 { #image_generation }

**[AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](image_generation/afmrl_attribute-enhanced_fine-grained_multi-modal_representation_learning_in_e-c.md)**

:   提出 AFMRL 框架，将电商产品的细粒度理解定义为属性生成任务，通过 MLLM 生成关键属性来增强对比学习（AGCL），并用检索性能作为奖励信号反向优化属性生成器（RAR），在大规模电商数据集上实现 SOTA 检索性能。

**[BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](image_generation/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)**

:   BookAgent 是一个安全感知的多智能体框架，通过**价值对齐故事板（VAS）+ 迭代跨模态精炼（ICR）+ 时序认知校准（TCC）**三阶段闭环架构，从用户草稿端到端生成高质量、角色一致、内容安全的绘本故事。

**[CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](image_generation/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)**

:   本文提出 CoDial，一个将预定义的对话流（task schema）转换为结构化异构图再自动生成 LLM 护栏代码（如 Colang）的框架，在推理阶段实现可解释且可控的任务型对话策略，在 STAR 基准上达到 SOTA，且无需训练数据。

**[ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](image_generation/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)**

:   本文提出 ControlAudio，一个统一的渐进式扩散建模框架，通过三阶段渐进训练（TTA 预训练→时序控制微调→时序+可懂语音联合训练）和渐进引导采样，在单个扩散模型中实现文本引导、时序精确控制和可懂语音生成三种能力，在时序精度和语音清晰度上显著超越现有方法。

**[Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](image_generation/follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)**

:   本文通过因果干预框架系统研究了文本到图像模型中文本编码器输出的 token 级信息分布，发现词汇项的语义通常集中在 1-2 个代表性 token 上，且跨项信息流在 11% 的情况下会导致语义泄漏和图像错误解读，并提出了简单有效的 token 级干预方法来改善对齐。

**[From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](image_generation/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)**

:   提出掩码历史学习（MHL）训练框架，通过在生成式推荐的自回归训练中加入掩码历史重建辅助任务，结合熵引导的自适应掩码策略和课程学习调度器，使模型从仅预测"下一个是什么"转向理解"为什么形成这条路径"，在三个数据集上显著超越SOTA。

**[Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](image_generation/investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)**

:   本文通过幽默场景系统调查 LLM 的反事实不公平性——交换说话者/听众身份后观察模型行为变化，发现特权群体说的笑话被拒绝率高达 67.5%，被判定为恶意的概率高 64.7%，且社会危害评分高达 1.5 分（5分制），揭示了模型内化了固定的社会特权层级而非进行真正的社会推理。

**[Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions](image_generation/large_language_models_are_bad_dice_players_llms_struggle_to_generate_random_numb.md)**

:   本文首次大规模系统审计了 11 个前沿 LLM 在 15 种概率分布上的原生采样能力，发现 LLM 严重缺乏内在概率采样机制，且这种缺陷会传导到下游应用中造成系统性偏差。

**[MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](image_generation/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)**

:   本文提出 MASH（多阶段风格人性化对齐），通过风格注入 SFT → DPO 对齐 → 推理时精炼三阶段流水线，训练一个仅 0.1B 参数的改写器，在黑盒设置下以 92% 的平均攻击成功率规避 AI 文本检测器，同时保持优秀的语言质量。

**[VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](image_generation/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)**

:   本文提出 Visualize-then-Retrieve (VisRet)，一种将文本查询先通过 T2I 生成模型可视化为图像、再在图像模态内进行检索的新范式，在四个基准上平均提升 nDCG@30 0.125（CLIP）和 0.121（E5-V），下游 VQA 准确率在 Visual-RAG-ME 上提升 15.7%。

**[ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](image_generation/zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)**

:   提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题，同时发布了首个大规模开源对话语音数据集 OpenDialog（6.8k 小时）。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](llm_alignment/aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)**

:   提出 Plan-RewardBench，一个面向复杂工具增强场景的轨迹级偏好基准，用于评估奖励模型在多步规划、工具使用和错误恢复等场景下区分优劣智能体轨迹的能力。

**[Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](llm_alignment/beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)**

:   本文提出了一种超越边际分布的 LLM 代表性评估框架，通过同时考察边际响应分布和跨问题相关结构来评估人口统计对齐模型，发现虽然微调和 persona prompting 能改善边际分布的近似度，但两者都无法忠实再现人类价值观调查中的多变量相关模式。

**[ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)**

:   ConsistRM 提出基于一致性感知的自训练框架，通过时序一致性伪标签（融合在线状态和历史记忆的偏好一致性）和语义一致性批评奖励（衡量多次生成批评的语义相似度）两个模块，在无需人工标注的条件下将生成式奖励模型的五个基准平均性能提升 1.5%，同时显著缓解了位置偏差问题。

**[Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](llm_alignment/into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)**

:   本文发现领域特定上下文（如化学论文）会选择性放松 LLM 对相关有害知识的防护（纵向解锁），而安全研究上下文会触发跨所有有害类别的广泛防护放松（通用解锁），据此提出 Jargon 攻击框架，在包括 GPT-5.2、Claude-4.5 在内的七个前沿模型上实现超 93% 的攻击成功率。

**[Reward Modeling for Scientific Writing Evaluation](llm_alignment/reward_modeling_for_scientific_writing_evaluation.md)**

:   本文提出 SciRM 和 SciRM-Ref 两个针对科学写作评估的开源奖励模型，通过两阶段强化学习（GRPO）分别优化评估偏好和推理能力，实现了在多种科学写作任务上的细粒度多方面评估，并能泛化到未见过的评估任务和标准。

**[Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](llm_alignment/robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)**

:   提出 Fission-GRPO，在 RL 训练循环中将工具执行错误动态转化为在线策略修正训练实例：通过学习的错误模拟器生成诊断反馈并重采样恢复轨迹，将 Qwen3-8B 的错误恢复率提升 5.7%，整体准确率从 42.75% 提升至 46.75%。

**[SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](llm_alignment/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)**

:   本文提出 SafeMERGE，一种轻量级后微调框架，通过余弦相似度检测偏离安全行为的微调层，仅将这些层与安全模型的对应层合并，在四个 LLM 上显著降低有害输出同时保持甚至提升任务性能。

**[SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](llm_alignment/sftmix_elevating_language_model_instruction_tuning_with_mixup_recipe.md)**

:   本文提出 SFTMix，一种基于 Mixup 的指令微调方法，通过训练动态将 SFT 数据集分为高置信度和低置信度子集，在隐表示空间对两者进行线性插值并施加 Mixup 正则化，在不依赖高质量数据集的情况下，跨 LLM 家族和数据集规模一致性地提升指令遵循能力。

**[STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](llm_alignment/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)**

:   本文提出 STAR-Teaming，一种基于策略-响应多路复用网络（Multiplex Network）的自动化红队测试框架，通过将攻击策略选择建模为逆 Ising 问题的概率优化，在 HarmBench 上达到平均 74.5% 的攻击成功率，比最强基线高 13.5%，同时显著降低计算开销。

**[Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](llm_alignment/towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)**

:   本文识别了直接对齐算法（DAAs）中的"奖励-生成鸿沟"——训练目标与自回归解码动态之间的不匹配，提出 POET（Prefix-Oriented Equal-length Training），通过将偏好响应对截断为较短者长度来隐式约束 token 级 MDP 在所有时间步上收敛，在 AlpacaEval 2 上最高提升 11.8 个百分点。

**[TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](llm_alignment/trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)**

:   本文提出 TrajGuard，一种无需训练的解码时越狱防御框架，通过滑动窗口聚合关键层隐藏状态轨迹实时量化风险，仅在风险持续超过阈值时触发轻量级语义裁判，在 12 种越狱攻击上实现 95% 平均防御率，检测延迟仅 5.2ms/token，误报率低于 1.5%。

---

## 📹 视频理解 { #video_understanding }

**[ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](video_understanding/arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)**

:   提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

**[Distorted or Fabricated? A Survey on Hallucination in Video LLMs](video_understanding/distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)**

:   本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

**[GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](video_understanding/gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)**

:   提出 GameplayQA，一个基于多人3D游戏视频的端到端基准框架，通过密集时间线标注（1.22标签/秒）和结构化干扰项分类学，系统评估多模态大模型在决策密集、多视角同步场景下的感知和推理能力，揭示前沿模型与人类表现仍有显著差距。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](video_understanding/preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)**

:   提出将 LLM 提取的自然语言偏好信号与贝叶斯对手建模框架结合的偏好估计方法，在多方多议题谈判中通过语言似然函数融合定性线索和定量出价信息，将完全达成协议率从 37% 提升至 62%。

**[Probing for Reading Times](video_understanding/probing_for_reading_times.md)**

:   本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

**[RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](video_understanding/rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)**

:   本文提出 RARE 框架，通过将文档分解为原子事实来追踪跨文档冗余，并设计 CRRF（基于独立准则排序的倒数排名融合）稳定 LLM 多准则判断，在金融/法律/专利等高冗余企业语料上构建了 RedQA 基准，揭示主流检索器在 4-hop 高重叠设置下 PerfRecall@10 从 66.4% 暴跌至 5.0-27.9%。

**[Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](video_understanding/saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)**

:   本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略，在代码生成上平均提升 Pass@1 1.9% 的同时实现 251.4% 的推理加速。

**[VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](video_understanding/vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)**

:   本文提出 VC-Inspector，一个基于开源轻量级多模态模型（Qwen2.5-VL 3B/7B）的无参考视频字幕评估指标，通过可控事实错误合成流水线生成训练数据，在 VATEX-Eval 上达到 $\tau_b$=42.58 的人类判断相关性，超越依赖 GPT-4o 的 G-VEval（$\tau_b$=39.40），且在幻觉检测基准上达到 99.6% 准确率。

**[ViLL-E: Video LLM Embeddings for Retrieval](video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)**

:   提出 ViLL-E，首个同时支持文本生成和 embedding 生成的 Video LLM 统一架构，通过三阶段生成-对比联合训练和自适应 KV-Former embedding head，在视频检索和时序定位上逼近专家模型，同时保持 VideoQA 竞争力。

**[VISTA: Verification In Sequential Turn-based Assessment](video_understanding/vista_verification_in_sequential_turn-based_assessment.md)**

:   VISTA 提出了一个基于声明级分解和顺序一致性追踪的多轮对话事实性评估框架，将不可验证内容细分为主观、矛盾、缺乏证据和弃权四类，在四个对话基准和八个 LLM 上显著优于 FActScore 和 LLM-as-Judge 基线。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)**

:   提出 RACE（Rhetorical Analysis for Creator-Editor Modeling），利用修辞结构理论(RST)构建逻辑图来建模文本"创作者"的思维架构，同时提取篇章单元级特征捕获"编辑者"的语言风格，实现四类细粒度 LLM 生成文本检测（人写/LLM写/LLM润色人文/人改写LLM文）。

**[BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](aigc_detection/biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)**

:   BiasedTales-ML 构建了约 35 万篇覆盖 8 种语言的 LLM 生成儿童故事语料库，通过全排列提示设计和分布分析框架，揭示了**叙事中社会属性分布在不同语言间存在显著差异**，英语中心的评估无法反映多语言场景下的偏见模式。

**[CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation](aigc_detection/citeguard_faithful_citation_attribution_for_llms_via_retrieval-augmented_validat.md)**

:   CiteGuard 提出了一个检索增强的智能体框架，通过扩展的检索动作（包括全文搜索和上下文检索）为科学引用归属提供更忠实的基础，在 CiteME 基准上相对基线提升 10 个百分点，达到 68.1% 准确率，接近人类表现（69.2%）。

**[FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](aigc_detection/flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)**

:   FlexGuard 提出了一种输出连续风险评分（0-100）而非二元安全/不安全判断的 LLM 审核模型，通过基于评分准则的蒸馏和 GRPO 风险对齐训练，在不同严格度部署场景下实现了 SOTA 的鲁棒性和准确率。

**[Frankentext: Stitching Random Text Fragments into Long-Form Narratives](aigc_detection/frankentext_stitching_random_text_fragments_into_long-form_narratives.md)**

:   提出Frankentext范式，让LLM在极端约束下（90%文本逐字复制自人类写作）拼接随机人类文本片段为连贯长篇叙事，揭示现有AI文本检测器在混合作者场景下的严重失败（72%的Frankentext被误判为人类写作）。

**[Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](aigc_detection/reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)**

:   提出基于推理的聚类精炼框架，将 LLM 作为语义判官（而非嵌入生成器）验证和重构无监督聚类的输出，通过一致性验证、冗余裁决和标签接地三个推理阶段，在社交媒体语料上显著提升聚类一致性和人类对齐的标注质量。

**[Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)**

:   本文通过构建跨12年的纵向写作数据集，发现LLM生成文本存在"时间扁平化"现象——虽然词汇多样性高，但在语义和认知情感维度上的时间漂移显著低于人类，仅凭时间变异模式就能以94%准确率区分人类与LLM文本。

**[When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)**

:   揭示了个性化场景下 MGT 检测器的"特征反转陷阱"——通用域中区分人写文本和机器文本的特征在个性化域中发生反转，导致检测器性能骤降甚至翻转，并提出 StyloCheck 框架通过量化检测器对反转特征的依赖程度来预测跨域性能变化，预测相关性达 0.85 以上。

**[Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)**

:   本文构建了首个面向LLM生成古典中文诗词的检测基准ChangAn（含30,664首诗），系统评估了12种AI检测方法在不同文本粒度和生成策略下的表现，揭示了当前中文文本检测器在古典诗词领域的严重局限性。

---

## 🗣️ 对话系统 { #dialogue }

**[Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](dialogue/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)**

:   提出ConvAgent，通过将RL训练奖励分解为结果奖励、信息增益奖励和混合主动行为奖励三个互补组件，训练对话式搜索智能体在多轮交互中交替进行搜索和推理。

**[Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](dialogue/author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)**

:   本文将学术论文作者回复（rebuttal）生成重新定义为"作者在回路"任务，提出 Re3Align 数据集（3.4K 论文、440K 句级编辑标注、15K 审稿-回复-修改三元组）、REspGen 可控生成框架和 REspEval 20+ 指标评估套件，在 5 个 SOTA LLM 上系统验证了作者输入、可控性和评估引导精修的效果。

**[Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)**

:   提出 DiaFORGE 框架，通过消歧中心的合成数据生成管线 + 推理链微调 + 动态评估体系，让开源 LLM 在面对近重复企业 API 时的工具调用成功率比 GPT-4o 高 27 个百分点、比 Claude-3.5-Sonnet 高 49 个百分点。

**[Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](dialogue/discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)**

:   本文提出 DRCR，首个将上下文改写引入多方对话生成的框架，使用话语连贯性和回复质量双反馈信号构建偏好数据，通过动态自演化学习让改写器和回复器在迭代训练中相互增强。

**[ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)**

:   ETHICMIND 提出推理时（inference-time）的风险感知对齐框架，在多轮对话的每一轮中联合分析伦理风险和用户情感，规划高层响应策略，再生成兼顾伦理引导和情感共鸣的回复，无需额外训练即可在高风险和道德模糊场景中实现更一致的对齐表现。

**[SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](dialogue/spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)**

:   本文提出 SPASM，一个以稳定性为核心的人设驱动多轮对话模拟框架，通过模块化人设生成、自我中心上下文投影（ECP）和终止检测三个组件，在 LLM-LLM 对话中大幅减少角色漂移和"回声"现象，构建了 45,000 段高质量多轮对话数据。

**[Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](dialogue/template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)**

:   提出 TaDSE 框架，利用对话中现有的模板（template）信息作为辅助锚点，通过模板感知的数据增强、配对对比训练和语义压缩推理三个阶段，在无监督设置下显著提升任务型对话的句子嵌入质量，在五个基准上超越此前 SOTA 甚至优于有监督的商业嵌入模型。

**[Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](dialogue/towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)**

:   本文提出 ProChatIP 框架，将客服聊天机器人从被动应答工具转变为主动信息采集引擎，通过专门的对话策略模块学习"何时探测"用户以获取预设的目标信息，同时最小化对话轮数和用户摩擦。

**[VoxMind: An End-to-End Agentic Spoken Dialogue System](dialogue/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)**

:   提出 VoxMind，一个赋予端到端语音对话模型智能体能力的统一框架：通过"Think-before-Speak"机制实现显式推理，结合多智能体动态工具管理架构解耦推理延迟与工具规模，任务完成率从基线 34.88% 提升至 74.57%，超越 Gemini-2.5-Pro。

---

## 👥 社会计算 { #social_computing }

**[Among Us: Language of Conspiracy Theorists on Mainstream Reddit](social_computing/among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)**

:   分析5亿条Reddit评论的10年纵向数据，发现活跃于阴谋论社区的用户在主流社区中也展现出可检测的独特语言模式（平均87%分类准确率），但这些模式高度依赖社区上下文，社区特定模型比全局模型高出最多17个百分点。

**[Explain the Flag: Contextualizing Hate Speech Beyond Censorship](social_computing/explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)**

:   本文提出一种混合方法，结合 LLM 和三种语言（英/法/希腊语）的人工策展词汇表来检测和解释仇恨言论——术语管道通过词汇匹配+LLM 语义消歧检测固有贬损用语，无术语管道用 LLM 检测群体针对性内容，两者融合生成有据可查的解释。

**[How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](social_computing/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)**

:   通过表示分析揭示 LLM 中"逻辑有效性"和"合理性"两个概念在隐层空间中高度对齐，导致模型将合理性与有效性混淆（内容效应），并构造去偏转向向量有效解耦这两个概念，减少内容效应同时提升推理准确率。

**[Is this chart lying to me? Automating the detection of misleading visualizations](social_computing/is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)**

:   提出 Misviz（2604张真实世界误导性可视化）和 Misviz-synth（57665张合成可视化）基准，覆盖12种误导类型，系统评估MLLM、规则检查器和图像分类器在检测误导性图表上的表现，揭示该任务仍极具挑战性。

**[On the Step Length Confounding in LLM Reasoning Data Selection](social_computing/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)**

:   本文发现基于自然度的 LLM 推理数据选择方法存在"步长混淆"问题——系统性地偏好每步更长的样本而非更高质量的样本，根因是推理步骤首 token 的低概率被长步骤稀释。提出 Aslec-drop（丢弃首 token 概率）和 Aslec-casl（因果回归去偏）两种校正方法，平均准确率提升 6-9%。

**[Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](social_computing/persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)**

:   构建了首个将人格特质（MBTI + Big Five）与读者情感反应关联的大规模数据集 Persona-E2，包含 3111 个事件 × 36 名标注者共 11.2 万条标注，揭示 LLM 在模拟人格化情感反应时存在"人格幻觉"问题，且 Big Five 特征比 MBTI 更有效地缓解该问题。

**[SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)**

:   本文提出 SPAGBias 框架，首次系统评估 LLM 在城市微观空间语境中的性别偏见，通过显式偏见、概率偏见和建构偏见三个诊断层揭示了 LLM 中结构化的空间-性别关联模式，并追溯偏见在模型开发全流程中的嵌入与放大。

**[ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)**

:   ToxiTrace 提出了一种面向 BERT 类编码器的可解释中文毒性检测方法，通过 CuSA（LLM 引导的弱标注）、GCLoss（梯度约束损失）和 ARCL（对抗推理对比学习）三个组件，在保持高效编码器推理的同时实现了句级分类准确率和连续有毒片段提取的双重提升。

**[ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](social_computing/toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)**

:   本文提出 ToxReason，一个基于不良结局路径 (AOP) 框架的化学毒性机理推理基准，整合药物-靶点实验数据与毒性标签，要求模型从分子起始事件推理到器官级不良结局；通过 GRPO 强化学习训练的 4B 模型在毒性预测（F1 71.4%）和推理质量上均超越 GPT-5 等大模型。

---

## 🔗 因果推理 { #causal_inference }

**[Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size](causal_inference/better_and_worse_with_scale_how_contextual_entrainment_diverges_with_model_size.md)**

:   本文首次为"上下文夹带效应"（contextual entrainment）建立缩放定律，发现更大的模型在语义上下文中更能抵抗虚假信息（负指数），但在非语义上下文中更容易复制无关 token（正指数），揭示了语义过滤和机械复制两种功能的对立缩放行为。

**[CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causal_inference/causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)**

:   CausalDetox 使用"必要性和充分性概率"（PNS）作为因果准则来精确定位产生有毒内容的注意力头，并通过局部推理时干预和 PNS 引导的微调两种互补策略进行去毒化，在多个模型上实现最高 5.34% 的毒性降低，同时保持语言流畅性。

**[ClimateCause: Complex and Implicit Causal Structures in Climate Reports](causal_inference/climatecause_complex_and_implicit_causal_structures_in_climate_reports.md)**

:   ClimateCause 构建了首个针对气候报告中复杂和隐式因果结构的专家标注数据集（874 条因果关系），支持嵌套因果、多事件拆解、相关性方向和时空语境标注，并提出基于因果图语义复杂度的可读性度量，LLM 基准测试显示因果链推理仍是重要挑战。

**[Cross-Modal Taxonomic Generalization in (Vision-) Language Models](causal_inference/cross-modal_taxonomic_generalization_in_vision-_language_models.md)**

:   本文系统研究 VLM 中语言模型是否能将纯文本习得的分类学知识（上位词关系）跨模态泛化到视觉输入，发现即使训练时完全不提供上位词标签，预训练 LM 仍能在图像中识别上位词类别，但这种泛化需要类别成员在视觉上的一致性。

**[Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](causal_inference/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)**

:   提出 Dialectic-Med，一个受波普尔证伪主义启发的多智能体医学诊断框架，通过提议者（诊断假设）、反对者（视觉证伪模块主动检索矛盾视觉证据）和调解者（加权共识图决策）的对抗辩证推理，在 MIMIC-CXR-VQA、VQA-RAD 和 PathVQA 上取得 SOTA，解释忠实度提升 12.5%，显著缓解诊断幻觉。

**[Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies](causal_inference/imperfectly_cooperative_human-ai_interactions_comparing_the_impacts_of_human_and.md)**

:   通过 2000 次 LLM 模拟和 290 人用户研究的双框架实验，比较了人类个性特质和 AI 设计属性在不完全合作场景（招聘谈判、部分诚实交易）中的影响，发现模拟中个性特质主导而真人实验中 AI 透明度才是关键驱动因素。

**[iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](causal_inference/itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)**

:   提出 iTAG 框架，通过逆向设计的三阶段流程（参数化因果图构建→基于 CoT 的概念赋值→结构保持的文本生成）生成同时具有极高因果图标注准确率和文本自然度的数据，可作为真实标注数据的实用替代品进行文本因果发现算法基准测试。

**[Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)**

:   本文系统研究了 LLM 在六种语言上的多语言反事实样本生成能力，通过直接生成和翻译两种路径对比，发现翻译路径的标签翻转率更高但需要更多编辑，识别出四类常见错误模式，并验证多语言反事实数据增强优于跨语言增强，尤其对低资源语言更有效。

---

## 🕸️ 图学习 { #graph_learning }

**[AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)**

:   提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。

**[ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)**

:   提出ARK框架，通过三维答案充分性评分（Forward+Backward+Retriever对齐）筛选正样本，利用LLM构建的知识图谱生成渐进难度的困难负样本进行课程对比学习，在10个数据集上平均提升14.5% F1。

**[AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](graph_learning/autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)**

:   提出 AutoPKG，一个多智能体 LLM 框架，从多模态电商商品内容自动构建 Product-Attribute 知识图谱（PKG），通过类型归纳 Agent、属性键发现 Agent、属性值提取 Agent 和集中式 KGD 决策 Agent 实现动态本体的持续演化和规范化，在 Lazada 数据集上取得 0.953 WKE（类型）和 0.724 WKE（属性键），线上 A/B 测试推荐 GMV 提升 7.89%。

**[Comparing Human and Large Language Model Interpretation of Implicit Information](graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)**

:   本文提出隐含信息提取（IIE）任务和基于 LLM 的三阶段提取管道（信息提取→推理验证→时序分析），构建结构化知识图谱来表示文本的隐含含义，并通过众包人类判断对比发现 LLM 在社交丰富语境中比人类更保守，但在短事实语境中人类更保守。

**[From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](graph_learning/from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)**

:   本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。

**[Graph-Based Alternatives to LLMs for Human Simulation](graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)**

:   本文提出 GEMS（Graph-basEd Models for Human Simulation），将封闭式人类行为模拟任务建模为异构图上的链接预测问题，在三个数据集和三种评估设定下匹配或超越强 LLM 基线方法，同时参数量减少 3 个数量级。

**[LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)**

:   本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。

**[Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](graph_learning/which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)**

:   本文提出了否定约束知识图谱问答（NEST KGQA）新任务和 NestKGQA 数据集，设计了 Python 格式逻辑形式 PyLF 来清晰表达否定约束，并提出 CUCKOO 框架通过约束感知草稿生成、Schema 引导语义匹配和自导向细化三个模块，在 few-shot 设置下实现了多约束问题的高效精确回答。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](llm_efficiency/bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)**

:   提出 BOSCH，一种免训练的注意力头级别 SWA 混合方法，将 SWA 头选择建模为大邻域搜索问题并分解为三阶段优化（层重要性探测→自适应比例分配→分组头选择），在 4 个模型 4 种比例设置下系统性超越层级启发式和 6 种静态头级别方法。

**[HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](llm_efficiency/humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)**

:   本文提出 HumanLLM 框架，将 244 个心理学模式（100 个人格特质 + 144 个社会认知模式）建模为相互作用的因果力而非孤立标签，构建了 11,359 个包含 2-5 个模式交互的场景和多轮对话数据集，通过双层 checklist 评估实现与人类判断的高对齐（$r=0.90$），HumanLLM-8B 在多模式动态上以 4 倍小的参数量超越 Qwen3-32B。

**[Multi-Drafter Speculative Decoding with Alignment Feedback](llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)**

:   本文提出 MetaSD，一个将多个异构草稿器整合到推测解码中的统一框架，将草稿器选择建模为多臂赌博机问题，通过块散度（Block Divergence）奖励信号动态选择与目标 LLM 最对齐的草稿器，在黑盒和白盒配置下一致优于单草稿器方法。

**[Native Hybrid Attention for Efficient Sequence Modeling](llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)**

:   本文提出 Native Hybrid Attention (NHA)，将线性 RNN 的长期记忆槽与滑动窗口的短期精确 token 拼接后通过单次 softmax 注意力统一处理，实现层内和层间混合的原生统一——无需额外融合参数即可动态分配长短期注意力权重，在 recall 密集和常识推理任务上超越 Transformer 和其他混合基线。

**[Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](llm_efficiency/planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)**

:   本文提出 PLOTTER 框架，首次将叙事规划从文本表示转移到图结构表示（事件图+角色图），通过多 agent 的 Evaluate-Plan-Revise 迭代循环在图拓扑上诊断和修复叙事缺陷，在叙事性、角色塑造、戏剧张力等维度上显著优于现有方法。

**[SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](llm_efficiency/scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)**

:   本文提出 SciCoQA，首个用于检测科学论文与其代码实现之间差异的基准数据集，包含 635 个差异实例（92 个真实 + 543 个合成），评测 22 个 LLM 后发现最强模型仅能检测 46.7% 的真实差异，揭示了自动化科学质量保证中的关键能力缺口。

**[SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](llm_efficiency/specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)**

:   提出 SpecBound 自草稿推测解码框架，通过逐层温度退火抑制浅层虚假高置信度预测，并设计有界推测算法自适应控制草稿的深度和宽度，在保持输出无损的同时实现最高 2.33× 的推理加速。

**[Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)**

:   提出推测验证（Speculative Verification, SV），通过引入与草稿模型同等规模的伴随模型（companion model），利用草稿-伴随分布的相似性预测推测准确率，动态调整验证长度以最大化有效吞吐量，在大批量推理中实现相对标准推测解码平均1.4×、最高1.9×的加速。

---

## 🤖 机器人/具身智能 { #robotics }

**[Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](robotics/can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)**

:   本文引入 Persuaficial——一个覆盖六种语言的高质量 AI 生成说服性文本多语言基准，系统评估了 LLM 生成的说服性文本与人类撰写的说服性文本在自动检测难度上的差异，发现微妙的 AI 说服比人类说服更难检测（F1 下降约 20%），而过度强化的说服反而更容易被发现。

**[Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](robotics/debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)**

:   提出RADAR框架，通过角色锚定（政客 vs 科学家）的多智能体辩论来检测基于遗漏上下文的半真半假信息，配合双阈值自适应早停机制，在噪声检索条件下一致超越单智能体和传统多智能体基线。

**[DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](robotics/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)**

:   提出 DeCoVec（Decoding Space based Task Vector），一个无训练、非侵入式的框架，通过对比 few-shot 和 zero-shot prompt 的输出 logit 分布差异构建解码空间中的任务向量，注入解码过程引导生成，在 TruthfulQA、Math-500 和 AQUA-RAT 上比标准 few-shot 基线平均提升高达 5.50 准确率。

**[On Safety Risks in Experience-Driven Self-Evolving Agents](robotics/on_safety_risks_in_experience-driven_self-evolving_agents.md)**

:   本文系统研究经验驱动自进化Agent的安全风险，发现仅从无害任务积累的经验也导致安全性显著退化（ASR上升13-49%），根因是经验的执行导向本质强化了行动而非拒绝。

**[Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](robotics/reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)**

:   本文提出"推理劫持"(Reasoning Hijacking) 这一新型攻击范式，通过在数据通道注入虚假决策标准来操纵 LLM 的推理逻辑而非改变任务目标，实现高攻击成功率且能绕过基于意图检测的防御方法。

**[VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](robotics/vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)**

:   本文提出 VLN-NF 基准——首个要求 VLN agent 在 3D 部分可观测环境中识别虚假前提指令并输出 NOT-FOUND 的任务，配套提出 REV-SPL 评估指标和 ROAM 两阶段混合框架，ROAM 达到 6.1 REV-SPL，比监督基线提升 45%。

**[XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](robotics/xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)**

:   揭示了 AI 编码助手自动收集上下文的设计漏洞，提出 Cross-Origin Context Poisoning（XOXO）攻击：通过语义保持的代码变换（如变量重命名）毒化共享代码库，使 GitHub Copilot 等助手在不知情的情况下生成有漏洞的代码，对 8 个 SOTA 模型平均攻击成功率达 73.20%。

---

## 🖼️ 图像恢复 { #image_restoration }

**[CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](image_restoration/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)**

:   本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。

**[Diffusion-CAM: Faithful Visual Explanations for dMLLMs](image_restoration/diffusion-cam_faithful_visual_explanations_for_dmllms.md)**

:   提出 Diffusion-CAM，首个专为扩散式多模态大语言模型（dMLLM）设计的可解释性方法，通过在去噪轨迹中提取结构有效的中间表征并配合四个后处理模块（自适应核去噪、分布感知置信门控、上下文背景衰减、单实例因果去偏），在 COCO Caption 和 GranDf 上显著超越自回归 CAM 基线。

**[Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](image_restoration/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)**

:   提出 EviOmni，通过"先推理再提取"的范式学习从检索文档中提取理性证据：将证据推理和证据提取整合为统一轨迹，用知识 token 掩码避免信息泄露，通过 GRPO 以可验证奖励优化，在 5 个基准上以极高压缩比（~38x）取得优于全文检索的准确率。

**[Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](image_restoration/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)**

:   首次系统性地对比扩散大语言模型（dLLM）与自回归（AR）对应模型的幻觉模式，揭示当前 dLLM 幻觉倾向更高，并识别出三种扩散特有的失败模式：过早终止、不完全去噪和上下文入侵。

**[Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](image_restoration/purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)**

:   本文提出 GeoDe 框架，通过在 LLM 隐空间中训练线性探针构建真值超平面，利用样本到超平面的几何距离作为置信度信号来筛选高质量弃权微调数据，有效消除决策边界附近的"灰色地带"噪声，显著提升模型的真实性和可靠性。

---

## 📚 预训练 { #llm_pretraining }

**[Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](llm_pretraining/commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)**

:   提出自动为现有常识知识库增添否定的方法，构建超过 200 万三元组的否定常识语料库（¬Atomic 和 ¬Anion），并证明在其上预训练可以提升 LLM 的否定理解能力。

**[Compact Example-Based Explanations for Language Models](llm_pretraining/compact_example-based_explanations_for_language_models.md)**

:   本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。

**[SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)**

:   本文提出 SAGE 优化器，通过 Lion 风格的符号更新方向和一个 $O(d)$ 内存开销的自适应阻尼缩放因子，解决了轻量级优化器在嵌入层上失败的"嵌入层困境"，在 Llama 模型（最大 1.3B）上以显著更低的优化器内存达到新的 SOTA 困惑度。

**[SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)**

:   本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。

**[Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](llm_pretraining/working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)**

:   本文将人类工作记忆约束（固定窗口、指数衰减、逻辑衰减、首因-近因效应）集成到 GPT-2 注意力机制中，在发展可信的小规模语料（10M/100M 词）上从头训练，发现这些约束在数据稀缺时显著提升语法准确率和人类阅读时间的预测力，且促进注意力头的功能专门化。

---

## 🎯 目标检测 { #object_detection }

**[E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](object_detection/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)**

:   提出E2E-GMNER，首个将实体识别、语义分类、视觉定位和隐式知识推理统一在单一多模态大语言模型中的端到端GMNER框架，通过CoT推理自适应判断视觉/知识线索的可用性，并引入高斯风险感知框扰动（GRBP）提升生成式框预测的鲁棒性。

**[Evaluating Memory Capability in Continuous Lifelog Scenario](object_detection/evaluating_memory_capability_in_continuous_lifelog_scenario.md)**

:   本文提出LifeDialBench，一个评估连续生活日志场景下记忆能力的基准（含7天真实数据的EgoMem和1年模拟的LifeMem），引入在线评估协议确保时间因果性，反直觉地发现简单RAG基线一致优于复杂记忆系统。

**[Evolutionary Negative Module Pruning for Better LoRA Merging](object_detection/evolutionary_negative_module_pruning_for_better_lora_merging.md)**

:   提出 ENMP 方法，通过进化搜索策略发现并剪除 LoRA 合并中降低性能的"负面模块"，作为即插即用的增强手段，在 NLP 和视觉领域全面提升现有合并算法的效果。

**[GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](object_detection/gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)**

:   提出 GigaCheck，一个双策略框架：文档级使用微调 LLM 进行分类，片段级创新地将 AI 生成文本片段视为"目标"，用 DETR-like 架构实现端到端的字符级定位。

**[Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](object_detection/retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)**

:   提出 BadRDM，首个针对检索增强扩散模型（RDM）的后门攻击框架，通过恶意对比学习微调检索器建立触发词到毒性代理图像的捷径，在类条件和 T2I 两种任务中分别达到 90.9% 和 96.4% 攻击成功率，同时保持良性生成质量。

---

## 📈 时间序列 { #time_series }

**[A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](time_series/a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)**

:   提出FinLangNet框架，通过双模块架构（DeepFM处理静态特征 + 双粒度提示机制的Transformer处理时序行为）实现多尺度信用风险预测，在滴滴金融平台部署后实现KS提升6.3pp和坏账率下降9.9%。

**[Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](time_series/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)**

:   提出 SIVR 框架，通过计算 LLM 隐藏状态跨层的内部方差（广义方差、圆方差、token 熵）作为 token 级特征，用轻量 Transformer 编码器聚合全序列模式来估计不确定性/检测幻觉，显著优于基线且泛化更强。

**[STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)**

:   本文提出 STK-Adapter，通过在 LLM 每一层嵌入三个 MoE 模块（ST-MoE 捕捉时空结构、EA-MoE 建模事件链语义、CMA-MoE 深度跨模态对齐），解决现有方法将 TKG 嵌入与 LLM 浅层对齐导致的时空信息丢失和逐层稀释问题，在四个基准数据集上显著超越 SOTA。

**[Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study](time_series/temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo.md)**

:   本文对 Google 和 DuckDuckGo 的日期过滤器进行系统审计，发现搜索引擎日期过滤在回顾性预测评估中严重失效——71%（Google）和 81%（DuckDuckGo）的问题至少有一个页面包含重大截止日期后信息泄漏，导致预测 Brier 分数从 0.24 虚降至 0.10。

**[Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)**

:   定义 Time-RA 新任务将时间序列异常检测从二分类升级为生成式推理诊断（检测+分类+原因解释），构建首个包含约 4 万样本、10 个领域、20 种异常类型的多模态基准 RATs40K，并通过 AI 反馈标注流程和 LLM 微调验证了该范式的可行性。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Aligning Language Models with Real-time Knowledge Editing](knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)**

:   引入CRAFT（持续更新的中文金融知识编辑数据集）和KEDAS（基于多样化编辑增强和自适应推理的知识编辑对齐范式），解决现有知识编辑方法在实时场景中成功率-局部性-可迁移性难以兼顾的问题。

**[CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)**

:   CLARE 提出了一种轻量级的表示层面方法，通过单个中间层的前向激活量化事实间的纠缠程度，用于预测模型编辑的连锁效应，相比梯度方法平均提升 62.2% Spearman 相关性，同时快 2.74 倍、内存减少 2.85 倍。

**[EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)**

:   提出 EvoEdit，通过动态演化零空间投影器实现大规模序列知识编辑，在保持原有知识的同时高效注入新知识，在 10K 编辑量级下仍保持 SOTA 性能，且比 AlphaEdit 快 3.5 倍。

**[FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)**

:   本文发现现有非结构化模型编辑方法虽能整体性回忆编辑文本但无法进行细粒度事实访问，提出FABLE框架通过两阶段层次化策略将细粒度事实锚定到浅层、整体性叙事整合到深层，并构建UnFine诊断基准进行系统评估。

---

## ✂️ 语义分割 { #segmentation }

**[AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation](segmentation/anchorseg_language_grounded_query_banks_for_reasoning_segmentation.md)**

:   提出AnchorSeg，将推理分割重构为基于语言引导查询库的结构化条件生成过程，通过锚点查询显式解耦空间定位与语义推理，配合Token-Mask循环一致性训练目标，在ReasonSeg上达到SOTA（67.7% gIoU, 68.1% cIoU）。

**[BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](segmentation/boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)**

:   BoundRL 将结构化文本分割重新定义为边界生成任务——仅生成每个片段的起始 token 而非完整文本，减少 90% 的输出 token 并消除幻觉风险，结合双目标奖励函数和选择性扰动策略的 RLVR 训练，使 1.7B 小模型超越了 Claude-4 Sonnet 的 few-shot 表现。

**[Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](segmentation/hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)**

:   本文提出 Hierarchical Policy Optimization (HPO)，通过层级奖励设计对基于 LLM 的同声传译模型进行后训练，在翻译质量未达阈值时抑制延迟优化，从而在 1.5 秒延迟下实现 +7 COMET 的翻译质量提升。

**[TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](segmentation/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)**

:   本文提出 TemporalVLM，通过时间感知的片段编码器（重叠滑动 Video Q-Former + 融合模块）提取局部细粒度时间特征，再用 BiLSTM 聚合全局长程依赖，首次在 Video LLM 中引入 LSTM，在密集视频描述、时序定位、高光检测和动作分割四项任务上超越先前方法。

---

## 📡 信号/通信 { #signal_comm }

**[PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](signal_comm/policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)**

:   本文提出 PolicyBench（21K 题的中美跨体制政策理解基准）和 PolicyMoE（基于认知层次的混合专家模型），系统评估 11 个 SOTA LLM 在政策记忆/理解/应用三层次上的能力，发现模型在结构化推理上表现好但在抽象政策概念上仍然薄弱。

**[Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](signal_comm/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)**

:   本文提出 APF（Automated Problem Formulation），一种与求解器无关的框架，利用 LLM 将工程师的自然语言设计需求转化为可执行的数学优化模型，通过创新的数据生成和测试实例标注管线克服高成本仿真场景下无法使用求解器反馈筛选数据的困难，在天线设计任务上显著优于现有方法。

**[UCS: Estimating Unseen Coverage for Improved In-Context Learning](signal_comm/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)**

:   本文提出 UCS（Unseen Coverage Selection），一种基于 Smoothed Good-Turing 估计器的无训练子集级覆盖率先验，通过估计候选示例集中未观测到的潜在聚类数量来正则化现有 ICL 示例选择方法，在意图分类和推理任务上提升 2-6% 准确率。

---

## 🎬 视频生成 { #video_generation }

**[Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity](video_generation/accelerating_training_of_autoregressive_video_generation_models_via_local_optimi.md)**

:   提出 Local Optimization + Representation Continuity (ReCo) 训练策略，通过在局部窗口内优化并约束隐状态的平滑过渡，实现自回归视频生成模型训练速度提升 2 倍且不牺牲生成质量。

**[OSCBench: Benchmarking Object State Change in Text-to-Video Generation](video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)**

:   提出 OSCBench——首个专门评估文生视频模型中物体状态变化（OSC）能力的基准，基于烹饪场景构建 1,120 条提示覆盖常规/新颖/组合三类场景，揭示即使最强 T2V 模型在 OSC 准确率上也仅达 0.786。

**[Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](video_generation/self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)**

:   提出 VideoRepair，首个免训练、模型无关的文本到视频自校正框架，通过 MLLM 检测细粒度文本-视频不对齐，保留正确区域并选择性修复问题区域，在 EvalCrafter 和 T2V-CompBench 上跨四种 T2V 骨干模型一致提升对齐质量。

---

## 🛡️ AI 安全 { #ai_safety }

**[When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](ai_safety/when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)**

:   本文构建了首个带政治倾向标签的多文档新闻摘要数据集 FairNews，并通过五维公平性评估框架对 13 个 LLM 进行评估，发现中等规模模型在公平性和效率上优于大模型，且实体情感相似性是最难通过提示去偏的维度。

**[XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](ai_safety/xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)**

:   构建首个具有跨语言平行质量的翻译评估基准 XQ-MEval，通过半自动注入 MQM 错误生成可控质量的伪翻译，首次实证揭示自动评估指标的跨语言评分偏差，并提出 LGN 归一化策略有效校准多语言指标评估。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[[b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](self_supervised/bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)**

:   系统性地证明自监督语音模型（S3M）的表示空间中存在线性的音韵特征向量，这些向量满足类似 word2vec 的向量算术关系，且其缩放比例与声学测量呈连续相关性。

**[ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](self_supervised/conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)**

:   本文提出 ConlangCrafter，一个基于 LLM 的多跳管道，将构造语言（conlang）设计分解为音系、语法、词汇三个模块化阶段，通过随机性注入保证类型学多样性、通过自精炼循环保证内部一致性，并提出了一个包含类型学多样性分析和翻译一致性评估的自动评估框架。

---

## 🧑 人体理解 { #human_understanding }

**[ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](human_understanding/generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)**

:   本文提出伪造归因报告生成（Forgery Attribution Report Generation）这一新任务，构建了包含 152,217 个样本的 MMTT 数据集（首个同时提供像素级掩码和人工文本描述的大规模面部伪造数据集），并提出 ForgeryTalker 端到端基线，通过共享编码器和双解码器（掩码+语言模型）联合生成定位掩码和归因报告，达到 59.3 CIDEr 和 73.67 IoU。

---

## 📐 优化/理论 { #optimization }

**[CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](optimization/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)**

:   本文提出 CLewR（Curriculum Learning with Restarts），一种在偏好优化训练中按易到难排序并在每个 epoch 重启课程的策略，有效缓解灾难性遗忘问题，在多个模型家族（Gemma2、Qwen2.5、Llama3.1）和多种偏好优化算法（DPO、CPO、ARPO）上持续提升机器翻译性能。

---

## 🛰️ 遥感 { #remote_sensing }

**[MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)**

:   本文提出 MONETA，首个结合文本（网站、维基百科、Wikidata）和地理空间数据（OpenStreetMap、卫星图像）的多模态行业分类基准，并设计零样本和多轮多智能体两种无训练管线，使用开源和闭源 MLLM 在 20 类 NACE 行业分类上达到 62.10%-74.10% 准确率，多轮设计最高提升 22.80%。

---

## 📂 其他 { #others }

**[Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations](others/agree_disagree_explain_decomposing_human_label_variation_in_nli_through_the_lens.md)**

:   将LiTEx推理分类法从"标签一致下的解释变异"扩展到"标签不一致"场景，发现标注者可能标签不同但推理类似，推理类别的一致性比标签一致性更好地反映解释的语义相似度。

**[Are Large Language Models Economically Viable for Industry Deployment?](others/are_large_language_models_economically_viable_for_industry_deployment.md)**

:   提出Edge-Eval框架，通过5个部署指标（经济盈亏平衡、智能功耗比、系统密度、冷启动税、量化保真度）在传统T4 GPU上全生命周期评估LLM，揭示<2B小模型在经济和生态维度全面优于7B模型，并发现QLoRA虽降低内存但能耗增加最高7倍的反常现象。

**[Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning](others/beyond_accuracy_unveiling_inefficiency_patterns_in_tool-integrated_reasoning.md)**

:   提出 PTE（Prefill Token Equivalents），一个基于硬件感知的工具集成推理效率度量指标，统一了内部推理和外部工具使用的成本，并通过大规模实验揭示了四种 TIR 低效模式：确认性工具使用、工具混合、缺乏工具先验和工具格式崩溃。

**[Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](others/dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)**

:   使用LLM智能体（DeepSeek/GPT系列）在经典啤酒分销博弈中模拟多阶段供应链，系统研究认知异质性（推理能力差异）对系统行为的影响，发现LLM智能体能复现人类的牛鞭效应和短视行为，且信息共享能有效缓解这些不良效应。

**[Reliable Evaluation Protocol for Low-Precision Retrieval](others/reliable_evaluation_protocol_for_low-precision_retrieval.md)**

:   揭示低精度（如二值化/量化嵌入）检索系统在评估时因分数粒度降低产生大量虚假并列（spurious ties），导致评估结果高度不稳定，提出 HPS（高精度打分）和 TRM（并列感知指标）两种互补策略，使低精度检索的评估更可靠一致。

</div>