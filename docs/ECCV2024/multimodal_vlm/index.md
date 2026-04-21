---
title: >-
  ECCV2024 多模态VLM方向 75篇论文解读
description: >-
  75篇ECCV2024 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🎞️ ECCV2024** · **75** 篇论文解读

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   本文构建了一个包含13.7万张作物病害图像和100万条问答对的多模态数据集CDDM，并提出同时对视觉编码器、适配器和语言模型进行LoRA微调的策略，在作物病害诊断任务上将病害分类准确率从5%提升至91.8%。

**[AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)**

:   提出AdaShield框架，通过精心设计的静态防御提示(AdaShield-S)和基于LLM的自适应迭代优化框架(AdaShield-A)，在不微调MLLM或训练额外模块的前提下，有效防御结构化越狱攻击，将攻击成功率从75%以上降至15%以下并保持正常任务性能。

**[AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)**

:   提出 AddressCLIP 框架，通过图像-文本对齐（地址+场景描述的对比学习）和图像-地理匹配（基于GPS距离的流形学习）两大核心组件，将图像地址定位（IAL）问题建模为端到端的视觉-语言对齐任务，在自建的三个IAL数据集上取得最高85.92%的Top-1准确率。

**[AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_visionlanguage_models_for_citywide_im.md)**

:   AddressCLIP 定义了"图像地址定位"(IAL) 新任务，提出端到端框架通过图像-文本对齐（图像↔地址/场景描述的对比学习）和图像-地理匹配（流形学习约束特征空间距离与地理距离一致）直接预测图像拍摄的可读文本地址，在自建的 Pittsburgh 和 San Francisco 数据集上优于现有 VLM 迁移方法。

**[Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_vision-language_models.md)**

:   本文提出Attention Prompting on Image（API），通过辅助模型（如CLIP或LLaVA）根据文本查询生成注意力热力图，将热力图叠加到原始图像上作为视觉提示输入LVLM，在不修改模型参数的情况下在MM-Vet、LLaVA-Bench等多个VL基准上稳定提升多种LVLM的性能（LLaVA-1.5提升3.8%/2.9%）。

**[Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)**

:   提出Attention Prompting on Image（API），用辅助VLM（如CLIP或LLaVA）根据文本查询生成注意力归因热力图，将其叠加到原始图像上作为视觉提示，在无需训练的情况下提升LVLM在多个VL基准上的表现（LLaVA-1.5 在MM-Vet上+3.8%）。

**[Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding](bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_.md)**

:   提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。

**[BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](beaf_observing_before-after_changes_to_evaluate_hallucination_in_vision-language.md)**

:   提出 BEAF 幻觉评估基准，通过图像编辑（移除物体）构造"前后对比"场景，设计 TU/IG/SB/ID 四个变化感知指标，揭示现有 VLM 即使传统 accuracy 高也可能存在严重幻觉。

**[BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](beaf_observing_beforeafter_changes_to_evaluate_hallucination.md)**

:   BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。

**[BLINK: Multimodal Large Language Models Can See but Not Perceive](blink_multimodal_large_language_models_can_see_but_not_perceive.md)**

:   提出BLINK——一个包含14个经典计算机视觉感知任务的多模态评测基准（3807道选择题），这些任务人类可以"眨眼间"解决（95.7%准确率），但最强的GPT-4V仅达51.26%（仅高于随机猜测13.17%），揭示了当前MLLM在核心视觉感知能力上的严重缺失。

**[BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)**

:   本文系统性地分析了不同视觉编码器（CLIP、DINOv2、EVA-CLIP等）对VLM性能的影响，发现没有单一编码器能在所有任务上最优，基于此提出BRAVE方法，通过轻量级的MEQ-Former将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在captioning和VQA任务上取得SOTA，并显著降低视觉幻觉。

**[BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_visionlanguage_model.md)**

:   通过系统benchmarking发现没有单一视觉编码器在所有VLM任务上最优，提出BRAVE方法用Multi-Encoder Querying Transformer（MEQ-Former）将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在多个captioning和VQA基准上达到SOTA。

**[CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)**

:   提出 CAT 模型，通过设计问题相关线索聚合器（Clue Aggregator）捕获细粒度音视频特征，结合混合多模态训练策略和 AI 辅助的模糊感知直接偏好优化（ADPO）策略，显著提升 MLLM 在动态音视频场景中的问答准确性，在多个 AVQA 基准上达到 SOTA。

**[CLAP: Isolating Content from Style Through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learni.md)**

:   从因果生成模型视角出发，提出CLAP（Contrastive Learning with Augmented Prompts），通过文本增强（而非图像增强）在预训练CLIP的特征空间中解耦内容与风格信息，以极低训练成本（<1小时）显著提升CLIP在零样本/少样本分类和对抗鲁棒性上的表现。

**[CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)**

:   从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。

**[Dataset Growth (InfoGrowth)](dataset_growth.md)**

:   提出 InfoGrowth，一种高效的在线数据清洗与选择算法，通过近邻搜索估计每个样本的信息增益，实现数据集的持续增长，同时保证清洁度和多样性，在 CC3M 上仅用 1/6 数据即超过全量训练效果。

**[DeCUR: Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_.md)**

:   将Barlow Twins扩展到多模态场景，通过将嵌入维度显式分为跨模态公共（对齐到identity矩阵）和模态独特（推到零矩阵）两部分，配合模态内自监督训练避免退化，在SAR-光学、RGB-DEM、RGB-深度三类场景中一致超越SimCLR-cross和Barlow Twins基线。

**[Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_self-supervised_lear.md)**

:   提出 DeCUR，在多模态自监督学习中将嵌入维度显式拆分为跨模态共有 (common) 和模态独有 (unique) 两部分，通过互相关矩阵分别驱动对齐与去相关，同时引入模态内训练保证独有维度学到有意义信息，在 SAR-光学、RGB-DEM、RGB-Depth 三类多模态场景上均优于 Barlow Twins / CLIP 等基线。

**[SpLIP: 通过多模态提示学习提升所有零样本草图检索任务](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)**

:   提出 SpLIP，一种基于冻结 CLIP 的双向多模态提示学习框架，通过视觉-文本编码器间的双向知识交换、自适应 margin 的三元组损失和条件跨模态拼图任务，在 ZS-SBIR、GZS-SBIR 和 FG-ZS-SBIR 三种草图检索设定下均取得 SOTA。

**[SpLIP: Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zeroshot_sketchbased_image_retrieval_through_m.md)**

:   提出SpLIP，在冻结CLIP backbone上实现双向prompt共享（视觉→文本、文本→视觉），结合自适应margin三元组损失和条件跨模态拼图任务，首次将多模态prompt learning引入ZS-SBIR，在Sketchy-Ext、TU-Berlin-Ext、QuickDraw-Ext上全面超越现有方法。

**[Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)**

:   提出ECSO（Eyes Closed, Safety On），一种无需训练的MLLM保护方法，通过检测自身响应的安全性，并将不安全查询中的图像自适应转换为文本描述，从而恢复预对齐LLM的内在安全机制，在MM-SafetyBench上实现最高71.3%的安全性提升，且不损害常规性能。

**[Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_imageto.md)**

:   发现MLLM虽易受图像输入的越狱攻击但具备内省能力（能检测自身不安全回复）、且去除图像后安全机制恢复，据此提出ECSO——通过自检不安全回复后将图像转为query-aware文本描述来恢复预对齐LLM的固有安全机制，无需额外训练即可大幅提升安全性。

**[FlexAttention: 面向高效高分辨率视觉语言模型的灵活注意力机制](flexattention_for_efficient_high-resolution_vision-language_models.md)**

:   提出 FlexAttention，通过基于注意力图的高分辨率token动态选择和层次化自注意力融合机制，在保持甚至超越现有高分辨率VLM性能的同时，将计算成本降低近40%。

**[FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_highresolution_visionlanguage_mo.md)**

:   提出FlexAttention注意力机制，通过注意力图引导动态选取约10%的高分辨率token并经层次化自注意力融合到LLM隐状态中，实现计算成本降低约40%的同时在V* Bench等高分辨率基准上超越现有方法。

**[FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)**

:   首次在**完全不使用动捕数据**的情况下，利用 MLLM（GPT-4V）作为关键帧设计师和动画师，结合基于物理的运动跟踪，实现开放集人体运动合成。

**[FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocapfree_human_motion_synthesis_with_multimodal_.md)**

:   FreeMotion首次在不使用任何动捕数据的情况下，利用GPT-4V作为关键帧设计师和动画师，将自然语言指令分解为关键帧序列，再通过插值和基于物理的运动跟踪填充帧间运动，实现了开放集人体动作合成。

**[GENIXER: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powe.md)**

:   Genixer提出一套完整的视觉指令微调数据生成pipeline，通过训练现有MLLM（LLaVA1.5和Shikra）使其具备数据生成能力，无需GPT-4即可生成高质量的VQA和REC指令数据，并通过Fuyu驱动和CLIP驱动的自动过滤框架保证数据质量。

**[Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)**

:   提出 Genixer 数据生成流水线，训练 MLLM 自身作为数据生成器，无需依赖 GPT-4V 即可自动生成高质量视觉指令微调数据，生成的 915K VQA 数据和 350K REC 数据分别提升 LLaVA1.5 和 Shikra 在多个基准上的表现。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal.md)**

:   提出Groma，通过在视觉tokenizer中引入区域提议和区域编码机制，将定位能力嵌入图像token化过程，实现统一的referring和grounding能力，在标准基准上超越同类MLLM。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)**

:   Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

**[IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)**

:   IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。

**[LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers](loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)**

:   LoA-Trans提出一种位置感知的查询选择机制，生成多个可能的目标位置作为位置感知查询（而非仅依赖估计的中心点），并引入TaskSyn网络在解码器中实现指代表达理解（REC）和指代表达分割（RES）的任务协同，显著提升视觉定位的准确性。

**[m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)**

:   提出 m&m's 基准，包含 4K+ 多步骤多模态任务和 33 个可执行工具，系统评估 10 个 LLM 在不同规划策略（多步 vs 逐步）、计划格式（JSON vs 代码）和反馈类型（解析/验证/执行）下的工具使用能力，发现多步JSON规划配合反馈是当前最优设计。

**[MarvelOVD: 融合目标检测器与视觉语言模型实现鲁棒开放词汇目标检测](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)**

:   提出 MarvelOVD 框架，通过将检测器的上下文感知能力和背景识别能力融入 VLM 的伪标签生成与训练流程，在线净化噪声伪标签并自适应重加权训练框，在 COCO 和 LVIS 上大幅超越已有方法。

**[MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)**

:   分析了VLM（CLIP）在局部区域预测中产生噪声伪标签的两大根因——缺乏上下文信息和无"背景"概念，提出MarvelOVD结合检测器的上下文和背景感知能力进行在线伪标签挖掘，配合自适应提案重加权和分层标签分配，在COCO和LVIS上显著超越SOTA。

**[MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)**

:   提出MathVerse——一个包含2612道视觉数学题目（转化为6个版本共15K测试样本）的多模态数学推理评测基准，通过系统性地调控文本与图像中的信息分配来检验MLLM是否真正"看懂"了数学图表，并提出CoT评估策略进行细粒度推理过程评分，揭示了大多数MLLM严重依赖文本而非视觉图表进行数学推理。

**[MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math?](mathverse_does_your_multimodal_llm_truly_see_the_diagrams_in.md)**

:   提出MathVerse——一个专门评估MLLM视觉数学推理能力的基准，通过将每道题转化为6个版本（从文本主导到纯视觉），揭示大多数MLLM严重依赖文本提示而非真正理解数学图表，并提出CoT评估策略进行细粒度推理过程评分。

**[Merlin: Empowering Multimodal LLMs with Foresight Minds](merlin_empowering_multimodal_llms_with_foresight_minds.md)**

:   提出 Foresight Pre-Training (FPT) 和 Foresight Instruction-Tuning (FIT) 两阶段训练范式，通过轨迹建模赋予多模态大语言模型"前瞻性思维"能力，使模型能够基于当前观察预测未来事件并进行推理。

**[Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。

**[Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](metaprompting_for_automating_zeroshot_visual_recognitio.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段元提示策略自动让 LLM 生成任务特定且类别特定的 VLM 提示，在 20 个数据集上将 CLIP 零样本识别提升最高 19.8%，完全消除人工提示设计。

**[MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](mm1_methods_analysis_and_insights_from_multimodal_llm_pre-training.md)**

:   Apple 系统性地消融了 MLLM 构建的三大轴（架构、数据、训练），得出关键设计准则：图像分辨率 > 模型大小 > 训练数据；VL 连接器类型影响甚微；caption/interleaved/text-only 三类数据的精细混合至关重要，最终构建了 3B-30B dense 和最高 64B MoE 的 MM1 模型族，在 few-shot 预训练评测上达到 SOTA。

**[MMBench: Is Your Multi-modal Model an All-Around Player?](mmbench_is_your_multi-modal_model_an_all-around_player.md)**

:   提出 MMBench——一个包含 3217 道多选题、覆盖 20 个细粒度能力维度的双语（英/中）视觉语言模型评测基准，并设计了 CircularEval 循环评测策略和基于 LLM 的选项提取机制，显著提升了评测的鲁棒性和公平性。

**[MMBench: Is Your Multi-modal Model an All-Around Player?](mmbench_is_your_multimodal_model_an_allaround_player.md)**

:   提出MMBench——一个系统设计的双语多模态评测基准，包含3000+多选题覆盖20个能力维度，并引入CircularEval策略和LLM辅助选项匹配，实现对VLM的鲁棒、细粒度评估。

**[MyVLM: Personalizing VLMs for User-Specific Queries](myvlm_personalizing_vlms_for_user-specific_queries.md)**

:   提出MyVLM，通过外部概念识别头（concept head）和可学习的概念嵌入向量（concept embedding），在不修改VLM原始权重的情况下实现个性化视觉语言交互——仅需3-5张图片即可让VLM识别并描述用户特定概念（如"你的狗"、"你的朋友"），在BLIP-2和LLaVA上均取得了显著的个性化效果。

**[MyVLM: Personalizing VLMs for User-Specific Queries](myvlm_personalizing_vlms_for_userspecific_queries.md)**

:   MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。

**[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](navgpt-2_unleashing_navigational_reasoning_capability_for_large_vision-language_.md)**

:   NavGPT-2通过将冻结LLM的隐层表征作为视觉-语言特征输入拓扑图导航策略网络，在保留LLM可解释性导航推理能力的同时，消除了基于LM的智能体与VLN专用模型之间的性能差距，并展现出优异的数据效率。

**[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](navgpt2_unleashing_navigational_reasoning_capability.md)**

:   提出 NavGPT-2，通过将冻结 LLM 与视觉内容对齐，结合拓扑图导航策略网络，在保持 LLM 可解释性推理能力的同时，消除了基于语言模型的导航智能体与 VLN 专用模型之间的性能差距。

**[Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)**

:   OVT通过构建460万多视角图文数据集MVCap和设计minimax优化的跨视角对齐框架，以参数高效微调方式显著提升VLP模型（如CLIP）对3D视角变化的鲁棒性（平均+9-10%），同时几乎不损失原始性能。

**[Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniviewtuning_boosting_viewpoint_invariance_of_visionlangua.md)**

:   构建460万多视角图文对数据集MVCap，提出Omniview-Tuning（OVT）框架，通过minimax式Cross-Viewpoint Alignment目标 + LoRA/VIFormer参数高效微调，在不损失原始性能的前提下将CLIP在视角OOD基准上的准确率平均提升约9-10%。

**[Quantized Prompt for Efficient Generalization of Vision-Language Models](quantized_prompt_for_efficient_generalization_of_vision-language_models.md)**

:   将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。

**[Quantized Prompt for Efficient Generalization of Vision-Language Models](quantized_prompt_for_efficient_generalization_of_visionlangu.md)**

:   发现适度噪声可以抑制VLM prompt tuning中的过拟合和灾难性遗忘，首次将量化误差视为正则化，设计了基于K-Means聚类的量化感知训练算法，在11个数据集上以极小存储开销（0.26KB）超越了众多SOTA方法。

**[REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models](revision_rendering_tools_enable_spatial_fidelity_in_vision-language_models.md)**

:   提出 REVISION 框架，利用 Blender 3D 渲染生成空间关系精确的合成图像，以免训练方式引导 T2I 模型生成空间一致的图像，并构建 RevQA 基准评估 MLLM 的空间推理能力。

**[Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_vision-language_adapters.md)**

:   本文发现CLIP适配方法（Adapter/Prompt Learning/TTA）在OOD场景下严重损害了零样本基线的校准能力，揭示logit范围增大（而非logit范数增大）是误校准的根本原因，并提出三种简单且模型无关的logit范围约束方案（ZS-Norm、Penalty、SaLS），有效缓解误校准同时保持判别性能。

**[Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_visionlanguage_adapters.md)**

:   发现CLIP适配方法（Prompt Learning、Adapters、Test-Time Adaptation）在OOD上的校准退化根因是logit范围（range）增大而非logit范数（norm），提出三种方案——ZS-Norm、Penalty和SaLS（Sample-adaptive Logit Scaling），其中SaLS无需训练即可在推理时将ECE降低50%以上。

**[Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)**

:   提出选择性双教师知识迁移框架（SND），通过衡量预训练VLM和最近微调VLM之间的特征差异，在无标签参考数据集上自适应选择合适的教师进行知识蒸馏，同时缓解灾难性遗忘并保持零样本分类能力。

**[Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)**

:   提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。

**[ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)**

:   ShareGPT4V 构建了一个120万条高质量描述性caption数据集（由GPT4-Vision生成100K种子 + Share-Captioner扩展至1.2M），通过在预训练和SFT两阶段使用该数据集训练LLaVA架构的模型ShareGPT4V-7B，在11个多模态benchmark中9个取得最优，证明了高质量caption是LMM模态对齐的关键瓶颈。

**[ShareGPT4V: Improving Large Multi-modal Models with Better Captions](sharegpt4v_improving_large_multimodal_models_with_better_cap.md)**

:   指出现有LMM训练中低质量caption是模态对齐的瓶颈，构建了1.2M高质量详细描述的ShareGPT4V数据集（100K来自GPT4-Vision + 1.2M来自训练得到的Share-Captioner），在预训练和SFT两阶段使用该数据，以简单架构的7B模型在11个基准中9个取得最优。

**[SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sq-llava_self-questioning_for_large_vision-language_assistant.md)**

:   提出视觉自提问（Visual Self-Questioning）训练范式，让 LLM 不仅学习回答问题，还学习根据图像主动提问，通过充分利用指令数据中问题本身的丰富语义信息来增强视觉-语言对齐。

**[SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sqllava_selfquestioning_for_large_visionlanguage_assistant.md)**

:   提出SQ-LLaVA，首次将指令数据中问题作为额外学习目标，训练MLLM不仅回答问题还学会"自问"，通过视觉自提问（visual self-questioning）任务挖掘指令数据中被忽视的问题上下文信息，配合原型提取器和LoRA微调，在10个VQA基准中9个超越基线。

**[The Hard Positive Truth about Vision-Language Compositionality](the_hard_positive_truth_about_vision-language_compositionality.md)**

:   本文揭示了现有CLIP硬负例微调方法在提升组合性理解时引入了"过敏感"问题——模型将语义不变的硬正例（hard positives）也错误地判为不匹配；通过同时引入硬正例和硬负例进行微调，显著缓解了该问题并实现了更鲁棒的组合性提升。

**[The Hard Positive Truth About Vision-Language Compositionality](the_hard_positive_truth_about_visionlanguage_compositionalit.md)**

:   本文揭示了现有CLIP组合性基准的评估盲区——缺少hard positives测试，发现hard negative微调会导致模型"过敏"（对语义保持的改写也错误地降低匹配分数），并通过同时加入hard positives和hard negatives训练来缓解这一问题。

**[Towards Open-ended Visual Quality Comparison](towards_open-ended_visual_quality_comparison.md)**

:   本文提出 Co-Instruct，首个面向开放式视觉质量比较的大型多模态模型，通过从两种"弱监督源"（LLM合并的单图描述 + GPT-4V伪标签）构建562K指令微调数据集，实现比 GPT-4V（其教师模型）更高的多图质量比较准确率，并提出首个多图比较基准 MICBench。

**[Towards Open-Ended Visual Quality Comparison](towards_openended_visual_quality_comparison.md)**

:   提出 Co-Instruct，首个开源的开放式视觉质量比较大模型，通过构建 Co-Instruct-562K 数据集和 MICBench 基准，使 LMM 在视觉质量比较任务上超越 GPT-4V。

**[Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models](towards_real-world_adverse_weather_image_restoration_enhancing_clearness_and_sem.md)**

:   本文提出WResVLM半监督学习框架，利用视觉-语言模型（VLM）为真实恶劣天气图像提供清晰度评估和语义描述监督信号，通过VLM图像评估+天气提示学习增强清晰度、描述辅助的语义正则化增强语义，在真实去雨/去雾/去雪任务上全面超越现有方法。

**[Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models](towards_realworld_adverse_weather_image_restoration_enhancin.md)**

:   提出WResVLM半监督框架，利用VLM评估图像清晰度和提供语义信息，通过伪标签选择+天气prompt学习增强清晰度、VLM描述引导的语义正则化增强语义，首次有效地将合成数据训练的复原模型泛化到真实恶劣天气场景。

**[UMBRAE: Unified Multimodal Brain Decoding](umbrae_unified_multimodal_brain_decoding.md)**

:   提出UMBRAE，通过通用脑编码器将fMRI信号与图像特征对齐后送入冻结的MLLM，实现多模态脑解码（描述、定位、检索、视觉重建），并创新性地引入跨被试训练策略，使单一模型服务多个被试且优于单被试模型。

**[Uni3DL: Unified Model for 3D and Language Understanding](uni3dl_a_unified_model_for_3d_vision-language_understanding.md)**

:   提出 Uni3DL，一个直接在点云上操作的统一 3D 视觉-语言模型，通过 Query Transformer 学习任务无关的语义/掩码输出，再由 Task Router 组合多个功能头实现语义分割、实例分割、目标检测、视觉定位、3D 描述生成、文本-3D 检索等六大任务，性能达到或超过各任务专用 SOTA。

**[UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_lan.md)**

:   提出UniCode，通过语言驱动的迭代训练范式学习一个统一码本，使LLM的词表可同时量化视觉和文本信号，无需额外对齐模块即可实现多模态理解与生成，并引入上下文图像解压缩任务提升生成质量。

**[UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)**

:   UniCode提出学习一个统一的codebook来同时tokenize视觉和文本信号，通过language-driven iterative training范式将视觉tokenizer的码本与LLM的词表渐进对齐，并引入in-context image decompression预训练任务提升图像生成质量，使MLLM无需额外对齐模块即可实现多模态理解与生成。

**[Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_vision-language_model.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表来扩展 LVLM 的视觉感知能力，使模型在保持通用能力的同时获得文档 OCR、图表理解等细粒度视觉感知能力。

**[Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

**[X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP-ViT（对比学习）和MAE-ViT（掩码图像建模）的互补视觉特征，在仅使用1/10数据量的情况下显著超越BLIP-2在细粒度视觉理解任务上的表现。

**[X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](xformer_unifying_contrastive_and_reconstruction_learning_for.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP（全局语义）和MAE（局部细节）两种视觉编码器的互补特征，结合ITC/ITM/ITG和重建四个损失联合优化，提升MLLM的细粒度视觉理解能力。

**[Zero-shot Object Counting with Good Exemplars (VA-Count)](zero-shot_object_counting_with_good_exemplars.md)**

:   提出VA-Count框架，通过样本增强模块（EEM）利用Grounding DINO发现高质量正负样本，结合噪声抑制模块（NSM）用对比学习区分正负密度图，实现零样本目标计数在FSC-147和CARPK上的SOTA表现。
