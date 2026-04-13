---
title: >-
  ICLR2026 多模态VLM方向 78篇论文解读
description: >-
  78篇ICLR2026 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🔬 ICLR2026** · 共 **78** 篇

**[A-Tpt Angular Diversity Calibration Properties For Test-Time Prompt Tuning Of Vi](a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)**

:   提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

**[Agilepruner An Empirical Study Of Attention And Diversity For Adaptive Visual To](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)**

:   通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

**[Biocap Exploiting Synthetic Captions Beyond Labels In Biological Foundation Mode](biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)**

:   提出 BioCAP，通过用 MLLM 生成 wiki 知识引导的合成描述性 caption（而非仅用物种标签）来训练生物学多模态基础模型，在 10 个物种分类 benchmark 上比 BioCLIP 平均提升 8.8%，在文本-图像检索任务上提升 21.3%。

**[Bongard-Rwr Real-World Representations Of Fine-Grained Concepts In Bongard Probl](bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)**

:   构建 Bongard-RWR+，一个包含 5400 个 Bongard 问题的 benchmark，使用 VLM 流水线（Pixtral-12B + Flux.1-dev）自动生成真实感图像来表示抽象概念，系统评估揭示 SOTA VLM 在辨别细粒度视觉概念（如轮廓、旋转、角度）时表现挣扎，准确率低至 19%。

**[Bootstrapping Mllm For Weakly-Supervised Class-Agnostic Object Counting](bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Breaking The Limits Of Open-Weight Clip An Optimization Framework For Self-Super](breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)**

:   本文提出 TuneCLIP，一个自监督微调（SSFT）框架，通过两阶段设计——先恢复优化器统计量（OSR）消除冷启动偏差，再用带margin的铰链全局对比损失（HGCL）缓解假负样本过度惩罚——在不使用任何标签的条件下持续提升已有开源 CLIP 模型的通用性能，在 ImageNet 及变体上提升最高 +2.5%，在 DataComp 基准上提升 +1.2%。

**[Can Vision-Language Models Answer Face To Face Questions In The Real-World](can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)**

:   提出 QIVD（Qualcomm Interactive Video Dataset），一个面对面实时问答 benchmark（2900 个视频+音频+时间戳标注），揭示现有 VLM 在实时情境理解上远落后人类（最佳模型 60% vs 人类 87%），主要瓶颈在指代消歧、回答时机判断和情境常识，微调可显著缩小差距。

**[Can Vision Language Models Assess Graphic Design Aesthetics A Benchmark Evaluati](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)**

:   提出 AesEval-Bench，首个系统性评估 VLM 图形设计美学评估能力的 benchmark（4维度×12指标×3任务），发现现有 VLM（含推理增强型）在设计美学上表现有限，并通过 human-guided VLM labeling + indicator-grounded reasoning 构建训练数据，微调 7B 模型在精确定位任务上超过 GPT-5。

**[Capacity-Aware Inference Mitigating The Straggler Effect In Mixture Of Experts](capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)**

:   针对 MoE 推理时因 token 分配不均导致的 Straggler Effect（最重负载专家决定整体延迟），提出 Capacity-Aware Token Drop（丢弃过载专家的低分 token）和 Expanded Drop（将溢出 token 重路由到本地低负载专家），在 Mixtral-8×7B 上实现 1.85× 加速且性能提升 0.2%。

**[Chart Deep Research In Lvlms Via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)**

:   提出 PRPO（Parallel Relative Policy Optimization），通过在奖励维度和数据类型两个层面做并行解耦优化，解决 GRPO 在多维奖励信号干扰和异构数据梯度冲突下的训练瓶颈；同时构建 MCDR-Bench，基于"错误唯一性原则"将主观生成评估转化为客观错误识别，实现图表深度研究能力的量化评估。

**[Citylens Evaluating Large Vision-Language Models For Urban Socioeconomic Sensing](citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)**

:   构建 CityLens——迄今最大规模的城市社会经济感知 benchmark（17 城市、6 大领域、11 个预测任务），评估 17 个 LVLM 在直接预测、归一化估计、特征回归三种范式下从卫星/街景图像推断社会经济指标的能力，发现通用 LVLM 在多数任务上仍不及领域特化的对比学习方法。

**[Closing The Modality Gap Aligns Group-Wise Semantics](closing_the_modality_gap_aligns_group-wise_semantics.md)**

:   证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。

**[Contamination Detection For Vlms Using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)**

:   提出多模态语义扰动方法检测 VLM 数据污染：用 LLM 生成密集描述 + Flux ControlNet 在保持图像构图的同时改变答案相关语义元素，污染模型因记忆原始图像-文本对而在扰动版本上失败，首次系统验证现有 LLM 检测方法在 VLM 上不可靠。

**[Customizing Visual Emotion Evaluation For Mllms An Open-Vocabulary Multifaceted ](customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)**

:   提出情感陈述判断（ESJ）任务和 INSETS 自动标注流水线，构建 MVEI benchmark，系统评估 MLLMs 的视觉情感感知能力，揭示当前模型在情感极性辨别和感知主观性理解上的显著不足。

**[Detecting Misbehaviors Of Large Vision-Language Models By Evidential Uncertainty](detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)**

:   提出 EUQ（Evidential Uncertainty Quantification），基于 Dempster-Shafer 证据理论将 LVLM 的认识不确定性分解为冲突（CF，内部矛盾）和无知（IG，信息缺失），单次前向传播即可检测幻觉、越狱、对抗攻击和 OOD 失败四类错误行为，AUROC 相对提升最高 10.5%。

**[Directional Embedding Smoothing For Robust Vision Language Models](directional_embedding_smoothing_for_robust_vision_language_models.md)**

:   将 RESTA（Randomized Embedding Smoothing and Token Aggregation）防御方法从 LLM 扩展到 VLM，发现方向性嵌入噪声（directional noise）在安全-实用性权衡上显著优于各向同性噪声（isotropic noise），可作为推理时的轻量防御层抵御多模态越狱攻击。

**[Diva-Grpo Enhancing Multimodal Reasoning Through Difficulty-Adaptive Variant Adv](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)**

:   提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。

**[Do Vision-Language Models Respect Contextual Integrity In Location Disclosure](do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)**

:   本文提出 VLM-GEOPRIVACY 基准，系统评估了14个主流 VLM 在判断图像位置信息披露适当程度方面的能力，发现这些模型虽然可以精确地理定位图像，但在隐私对齐方面严重不足——经常在敏感场景中过度披露，且容易受到基于提示的攻击。

**[Dynamic Multimodal Activation Steering For Hallucination Mitigation In Large Vis](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)**

:   提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

**[Enhanced Continual Learning Of Vision-Language Models With Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)**

:   提出Continual Decoupling-Unifying（ConDU）框架，首次将模型融合引入VLM持续学习，通过维护统一模型并结合任务触发器进行解耦-统一迭代操作，在MTIL基准上平均性能超SOTA 2%，同时增强了零样本能力。

**[Enhancing Multi-Image Understanding Through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)**

:   通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。

**[Error Notebook-Guided Training-Free Part Retrieval In 3D Cad Assemblies Via Visi](error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)**

:   提出一种无训练的两阶段VLM框架，通过Error Notebook记录纠正后的推理轨迹并结合RAG进行推理时适应，在3D CAD装配体的规格驱动零件检索任务上，GPT-4o准确率从41.7%提升至65.1%（+23.4%），并通过语法约束验证器进一步提升4.5%。

**[Frieda Benchmarking Multi-Step Cartographic Reasoning In Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)**

:   提出 FRIEDA 基准，系统评估大型视觉语言模型在多步骤、跨地图的制图推理能力，发现最强模型 Gemini-2.5-Pro 准确率仅 38.20%，远低于人类 84.87%。

**[Grasp Any Region Towards Precise Contextual Pixel Understanding For Multimodal L](grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)**

:   提出 GAR（Grasp Any Region），通过 RoI-aligned feature replay 在保持全局上下文的同时提取高保真局部特征，实现精准的单区域描述、多区域交互建模和复合推理，1B 模型即超越 InternVL3-78B。

**[Grounding-Iqa Grounding Multimodal Language Model For Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)**

:   将空间定位（referring + grounding）与图像质量评估结合，构建 GIQA-160K 数据集训练多模态 LLM 生成带有边界框的质量描述和空间 VQA，在细粒度质量感知上显著优于通用 MLLM。

**[Gtr-Bench Evaluating Geo-Temporal Reasoning In Vision-Language Mod](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[Hidrop Hierarchical Vision Token Reduction In Mllms Via Late Injection Concave P](hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)**

:   提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

**[Icym2I The Illusion Of Multimodal Informativeness Under Missingness](icym2i_the_illusion_of_multimodal_informativeness_under_missingness.md)**

:   揭示了多模态学习中被忽视的问题：模态缺失（missingness）导致的分布偏移会使模态价值评估产生严重偏差，提出 ICYM2I 框架通过双重逆概率加权（IPW）纠正训练和评估中的偏差，在 MAR 假设下实现对模态预测效用和信息论价值的无偏估计。

**[Index-Preserving Lightweight Token Pruning For Efficient Document Understanding ](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)**

:   提出一种轻量级的 token 剪枝框架，通过二值 patch 分类器移除文档图像中的非文本背景区域，并用 max-pooling 细化步骤恢复碎片化文本区域的空间连贯性，在保持准确率的同时大幅降低 VLM 的计算开销。

**[Ivc-Prune Revealing The Implicit Visual Coordinates In Lvlms For Vision Token Pr](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)**

:   揭示了LVLM中RoPE位置编码隐式建立的视觉坐标系统（IVC tokens），提出一种训练免的、提示感知的视觉token剪枝策略，在保留IVC tokens和语义前景token的同时，削减约50%视觉token并维持≥99%原始性能。

**[Keeplora Continual Learning With Residual Gradient Adaptation](keeplora_continual_learning_with_residual_gradient_adaptation.md)**

:   通过分析预训练模型权重的SVD分解，发现通用知识编码在主子空间、领域特定知识编码在残差子空间，提出KeepLoRA方法将新任务的LoRA更新约束在残差子空间中，同时用梯度信息初始化以保持可塑性，在持续学习中达到前向稳定、后向稳定和可塑性的最优平衡。

**[Liveweb-Ie A Benchmark For Online Web Information Extraction](liveweb-ie_a_benchmark_for_online_web_information_extraction.md)**

:   提出首个面向在线网页的信息抽取（WIE）基准LiveWeb-IE，覆盖文本/图片/超链接等多类数据抽取，并设计Visual Grounding Scraper（VGS）框架，通过模拟人类认知过程——视觉扫描定位区域→精确定位元素→生成XPath——在动态网页上实现鲁棒的信息抽取。

**[Llava-Fa Learning Fourier Approximation For Compressing Large Multimodal Models](llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)**

:   提出 LLaVA-FA，一种在频域进行联合低秩加量化权重近似的高效多模态大模型压缩方法，利用傅里叶变换的去相关性和共轭对称性实现更紧凑准确的权重表示，并引入 PolarQuant（极坐标量化）和 ODC（可选对角校准）方案，在多个基准上以最少的激活参数和计算成本超越现有高效多模态模型。

**[Look Carefully Adaptive Visual Reinforcements In Multimodal Large Language Model](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[Meta-Adaptive Prompt Distillation For Few-Shot Visual Question Answering](meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)**

:   提出 MAPD（Meta-Adaptive Prompt Distillation），一种基于 MAML 元学习的提示蒸馏方法，通过注意力映射器从任务相关的图像特征中蒸馏软提示，使 LMM 在测试时仅用少量梯度步即可适应新的视觉问答任务，性能超越 ICL 21.2%。

**[Metis-Specs Decoupling Multimodal Learning Via Self-Distilled Preference-Based C](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)**

:   提出SPECS框架将VLM的冷启动从SFT替换为DPO偏好训练——通过自蒸馏生成只关注输出格式的偏好数据，DPO冷启动专注表层形式学习(格式/结构/风格)而非内容记忆，为后续GRPO的深层推理学习提供更好的起点，MEGA-Bench+4.1%、MathVista+12.2%。

**[Mixing Importance With Diversity Joint Optimization For Kv Cache Compression In ](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)**

:   发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

**[Mmr-Life Piecing Together Real-Life Scenes For Multimodal Multi-Image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)**

:   提出MMR-Life基准（2646道多图选择题，覆盖7种推理类型21个任务），首次系统评估MLLM在真实生活场景中的多图多类推理能力，发现GPT-5仅58%准确率，在因果/空间/时序推理上存在显著瓶颈。

**[Mmtok Multimodal Coverage Maximization For Efficient Inference Of Vlms](mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)**

:   提出MMTok——一种基于最大覆盖问题（Maximum Coverage Problem）的多模态视觉token选择框架，同时利用文本-视觉和视觉-视觉覆盖信息来选择最具信息量的视觉token子集，在training-free设置下显著优于单模态baseline，甚至超越需要微调的方法。

**[Modal Aphasia Can Unified Multimodal Models Describe Images From Memory](modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)**

:   发现并定义"模态失语"现象——统一多模态模型能精准生成视觉概念（如电影海报）但无法用文字准确描述同一概念，文本描述的错误率是视觉生成的7倍以上，揭示了当前统一模型中知识的跨模态迁移失败和潜在安全隐患。

**[Multimodal Classification Via Total Correlation Maximization](multimodal_classification_via_total_correlation_maximization.md)**

:   从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

**[Multimodal Prompt Optimization Why Not Leverage Multiple Modalities For Mllms](multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)**

:   首次定义并解决"多模态提示优化"问题，提出MPO框架通过对齐保持的联合探索（统一反馈→同步更新文本+非文本提示）和先验继承的贝叶斯UCB选择（父提示性能作为先验warm-start），在图像/视频/分子等10个数据集上全面超越文本only优化。

**[Omnispatial Towards Comprehensive Spatial Reasoning Benchmark For Vision Languag](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)**

:   提出OmniSpatial基准，基于认知心理学系统覆盖4大空间推理维度（动态推理/复杂空间逻辑/空间交互/透视转换）50个子类别的8400+人工标注题目，发现o3/Gemini-2.5-Pro等最强模型在现有基准上>90%但在OmniSpatial上仍显著挣扎。

**[Post-Hoc Probabilistic Vision-Language Models](post-hoc_probabilistic_vision-language_models.md)**

:   提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。

**[Ppe Positional Preservation Embedding For Token Compression In Multimodal Large ](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)**

:   提出PPE(位置保持嵌入)，在MLLM视觉token合并时将多个原始位置ID编码到单个压缩token的不同维度段中（利用RoPE/M-RoPE维度独立性），无参数且即插即用，90%压缩率下在MMBench/TextVQA/VideoMME上比先前方法提升2-5%。

**[Prismm-Bench A Benchmark Of Peer-Review Grounded Multimodal Inconsistencies](prismm-bench_a_benchmark_of_peer-review_grounded_multimodal_inconsistencies.md)**

:   构建PRISMM-Bench——首个基于真实审稿人标记的科学论文多模态不一致性基准：从ICLR 2024/2025的开放评审中挖掘384个跨文本-图表-公式的不一致(而非合成错误),设计关于识别/修复/配对匹配三个任务+JSON去偏答案表示,21个顶级LMM最高仅53.9%准确率→暴露当前模型在科学文档推理上的严重不足。

**[Reasoning-Driven Multimodal Llm For Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)**

:   提出RD-MLDG——首个用MLLM推理链增强域泛化的框架：构建DomainBed-Reasoning数据集(每个样本配GPT-4o生成的类别相关推理链)，发现推理监督比直接标签更难优化且存在推理模式不匹配问题，通过MTCT(多任务交叉训练)和SARR(自对齐推理正则化)解决这两个挑战，在PACS/VLCS/OfficeHome/TerraInc上达SOTA。

**[Ref-Adv Exploring Mllm Visual Reasoning In Referring Expression Tasks](ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks.md)**

:   提出Ref-Adv——消除捷径的现代指称表达理解(REC)基准：通过配对语言非平凡表达+仅含必要信息(无冗余描述符)+硬干扰物的真实图像,暴露当前MLLM在RefCOCO上90%+准确率背后对捷径的依赖——所有模型在Ref-Adv上显著下降,揭示视觉推理和定位的真实能力Gap。

**[Revisit Visual Prompt Tuning The Expressiveness Of Prompt Experts](revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)**

:   从混合专家（MoE）视角揭示 VPT 的局限性——prompt experts 是输入无关的常量函数表达力受限，提出 VAPT 通过 token-wise 投影器和共享特征投影器使 prompt experts 自适应输入，用更少参数实现更优性能，并给出了最优样本效率的理论保证。

**[Seeing Across Views Benchmarking Spatial Reasoning Of Vision-Language Models In ](seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)**

:   提出MV-RoboBench——首个专门评估VLM在机器人操作场景中多视角空间推理能力的基准：1.7K人工策划的QA×8个子任务(空间理解+机器人执行)，发现SOTA模型远低于人类性能，揭示两个关键发现：(1)空间智能与机器人执行正相关,(2)通用单视角基准的强表现不能迁移到多视角机器人场景。

**[Self-Aug Query And Entropy Adaptive Decoding For Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)**

:   提出Self-Aug——无训练的LVLM解码策略减少视觉幻觉：(1)自增强prompting利用模型自身知识选择与文本查询语义对齐的视觉增强(而非随机噪声)→最大化对比解码的信息差异，(2)稀疏度自适应截断(SAT)基于专家logit的熵(而非单一最大值)动态调整候选token集大小→避免对比解码中负logit的虚假放大，在5个LVLM×7个基准上显著增强事实一致性。

**[Self-Evolving Vision-Language Models For Image Quality Assessment Via Voting And](self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)**

:   提出 EvoQuality 框架，通过成对多数投票生成伪排序标签、结合 GRPO 自迭代优化，使 VLM 在无人工标注下自主提升图像质量感知能力，零样本性能提升 31.8% PLCC，在 7 个 IQA 基准中 5 个超越有监督 SOTA。

**[Shuffle-R1 Efficient Rl Framework For Multimodal Large Language Models Via Data-](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)**

:   提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

**[Small Drafts Big Verdict Information-Intensive Visual Reasoning Via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)**

:   提出Speculative Verdict(SV)——受推测解码启发的免训练多模态推理框架：多个轻量VLM作为"草案专家"生成多样推理路径(提供不同的定位和证据)，大型VLM作为"裁决者"综合这些路径产出最终答案→纠正47-53%单模型或投票失败的案例，在InfographicVQA/ChartQA等信息密集基准上超越GPT-4o达10%。

**[Sophiavl-R1 Reinforcing Mllms Reasoning With Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)**

:   提出SophiaVL-R1——在规则基RL训练MLLM推理时引入整体级思维过程奖励：训练Thinking Reward Model从逻辑一致性/冗余度等五维度评估推理质量→提出Trust-GRPO基于正确/错误答案组的思维奖励对比计算可信度权重$\gamma$缓解reward hacking→退火策略$e^{-\text{steps}/T}$渐减思维奖励使后期更依赖准确的规则奖励→7B模型在MathVista(71.3%)和MMMU(61.3%)等多个基准全面超越LLaVA-OneVision-72B。

**[Sparsity Forcing Reinforcing Token Sparsity Of Mllms](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)**

:   提出Sparsity Forcing——基于GRPO的RL后训练框架，将带稀疏注意力的MLLM作为策略模型、原始MLLM作为参考模型，通过多预算rollout探索不同token保留阈值$p$，以效率(token减少率)+性能(答案正确性)为联合奖励做组内对比优化，将Qwen2/2.5-VL的token减少率从20%提升至75%且精度损失极小，实现内存降3×、解码加速3.3×。

**[Spatial-Dise A Unified Benchmark For Evaluating Spatial Reasoning In Vision-Lang](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)**

:   提出Spatial-DISE——基于认知科学DISE分类法(内在-外在×静态-动态四象限)的统一空间推理基准：559评估对+1.2万训练对(Blender自动化pipeline)→评估32个SOTA VLM→揭示所有模型远低于人类(尤其动态内在推理如心理旋转→接近随机)→空间推理失败源于认知过程(规则推理/心理模拟)缺陷而非视觉感知。

**[Spatial Captcha Generatively Benchmarking Spatial Reasoning For Human-Machine Di](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)**

:   提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率，远低于人类表现。

**[Spatial Reasoning Is Not A Free Lunch A Controlled Study On Llava](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过LLaVA框架的受控诊断研究揭示VLM空间推理失败的架构根源——(1)CLIP式编码器优化全局语义对齐而非空间结构→空间推理弱,(2)图像被展平为1D token序列+1D位置编码→丢失2D空间结构→系统性比较CLIP/SigLIP/SigLIP2/AIMv2编码器+2D-RoPE变体→发现编码器目标和位置结构影响空间行为但不能完全解决。

**[Spatialab Can Vision-Language Models Perform Spatial Reasoning In The Wild](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)**

:   提出SpatiaLab，一个包含1400个视觉QA对的真实场景空间推理基准，涵盖6大类30子类空间任务，支持多选和开放式双格式评估，揭示当前最强VLM（InternVL3.5-72B MCQ 54.93%）与人类（87.57%）之间存在巨大空间推理鸿沟，且开放式设置下差距更大。

**[Spinbench Perspective And Rotation As A Lens On Spatial Reasoning In Vlms](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)**

:   提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。

**[Steering And Rectifying Latent Representation Manifolds In Frozen Multi-Modal Ll](steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)**

:   提出 SteerVAD 框架，在完全冻结的多模态大语言模型 (MLLM) 内部，通过识别"潜在异常专家"注意力头并用层次化元控制器动态操控其表示流形，仅用 1% 训练数据即实现免调优视频异常检测的 SOTA。

**[Tabledart Dynamic Adaptive Multi-Modal Routing For Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)**

:   提出TableDART——通过轻量MLP门控网络(2.59M参数)动态选择最优模态路径(文本/图像/融合)的表格理解框架：复用预训练单模态模型(冻结)→每个query-table对动态路由→融合时用LLM agent仲裁/综合两路输出→训练高效(仅训练门控)→7个基准SOTA超最强基线平均4.02%。

**[Thinkomni Lifting Textual Reasoning To Omni-Modal Scenarios Via Guidance Decodin](thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Through The Lens Of Contrast Self-Improving Visual Reasoning In Vlms](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)**

:   提出VC-STaR(Visual Contrastive Self-Taught Reasoner)——利用视觉对比纠正VLM推理中的视觉幻觉：关键发现→VLM在对比VQA对(两张相似图+相似问题)中比单图时更准确地捕捉视觉线索→据此设计三步自改进(生成粗推理→对比分析→LLM精化)→构建VisCoR-55K数据集(5.5万高质量视觉推理样本,覆盖5个VQA域)→微调后超越现有自改进方法和SOTA视觉推理数据集。

**[U-Marvel Unveiling Key Factors For Universal Multimodal Retrieval Via Embedding ](u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)**

:   系统研究MLLM嵌入学习关键设计因素，发现被忽视的核心因子(双向注意力+mean pooling远优于last token; batch/lr/温度交互)，提出U-MARVEL：渐进过渡+过滤硬负+重排蒸馏，M-BEIR大幅超SOTA且零样本迁移CIR和T2V。

**[Unified Vision-Language Modeling Via Concept Space Alignment](unified_vision-language_modeling_via_concept_space_alignment.md)**

:   提出v-Sonar将视觉编码器后置对齐到文本嵌入空间Sonar，使得在Sonar空间上训练的Large Concept Model (LCM)能零样本处理视觉输入，并通过指令微调扩展为v-LCM，在61/62种语言上超越现有VLM。

**[Unihm Unified Dexterous Hand Manipulation With Vision Language Model](unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)**

:   提出UniHM，首个统一的语言条件灵巧手操控框架，通过形态无关VQ codebook将异构机械手映射到共享离散空间，结合VLM进行指令驱动操控序列生成，并通过物理引导动态优化确保物理可行性。

**[Visiomath Benchmarking Figure-Based Mathematical Reasoning In Lmms](visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)**

:   提出VisioMath基准，包含1800道K-12数学题目，所有选项均为高度视觉相似的图表，揭示了LMM在多图像-文本对齐上的核心短板，并探索三种对齐策略实现+12.6%的提升。

**[Vision-R1 Incentivizing Reasoning Capability In Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)**

:   提出Vision-R1，通过Modality Bridging构建200K高质量多模态CoT数据进行冷启动初始化，再用渐进思维抑制训练(PTST)策略结合GRPO强化学习，在7B参数规模达到与OpenAI O1接近的多模态数学推理能力。

**[Vision-Zero Scalable Vlm Self-Improvement Via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)**

:   提出 Vision-Zero，首个无标注的游戏化自博弈框架，通过"谁是卧底"式视觉推理游戏实现 VLM 的可扩展自进化，结合 Iterative-SPO 训练算法在推理、图表理解和视觉中心任务上超越基于人工标注数据的 SOTA 方法。

**[Visjudge-Bench Aesthetics And Quality Assessment Of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)**

:   提出首个面向数据可视化美学与质量评估的综合基准 VisJudge-Bench（3,090 样本，32 种图表类型），并训练 VisJudge 模型，将 MAE 相比 GPT-5 降低 23.9%，与人类专家的一致性提升 60.5%。

**[Visual Prompt-Agnostic Evolution](visual_prompt-agnostic_evolution.md)**

:   提出 Prompt-Agnostic Evolution (PAE)，通过频域感知的任务初始化 (MPA) 和 Koopman-Lyapunov 动力系统 (KLD) 跨层关联 prompt，加速 VPT 收敛（平均 1.41× 加速）并在 25 个数据集上提升 1–3% 精度，且对各类 VPT 变体即插即用、无推理开销。

**[Visual Symbolic Mechanisms Vlm](visual_symbolic_mechanisms_vlm.md)**

:   发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。

**[Webds An End-To-End Benchmark For Web-Based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)**

:   提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。

**[Why Keep Your Doubts To Yourself Trading Visual Uncertainties In Multi-Agent Ban](why_keep_your_doubts_to_yourself_trading_visual_uncertainties_in_multi-agent_ban.md)**

:   提出 Agora 框架，将多智能体 VLM 协调问题重构为去中心化的不确定性交易市场——将认知不确定性铸造为可量化的三维可交易资产（感知/语义/推理），通过利润驱动的交易协议和市场感知的 Thompson Sampling Broker 实现成本高效的均衡分配，在 5 个多模态基准上一致超越启发式方法（如 MMMU 上 +8.5% 准确率同时成本降低 3 倍以上）。

**[Why Keep Your Doubts To Yourself Trading Visual Uncertainty](why_keep_your_doubts_to_yourself_trading_visual_uncertainty.md)**

:   提出 Agora 框架，将多智能体 VLM 协调问题重新建模为去中心化的不确定性交易市场，通过将认知不确定性拆分为可交易资产（感知/语义/推理三维），并用基于盈利性驱动的交易协议和 Thompson Sampling 代理人实现成本感知的最优分配，在五个多模态基准上以超 3 倍成本节省获得至多 +8.5% 准确率提升。

**[Why Reinforcement Fine-Tuning Enables Mllms Preserve Prior Knowledge Better A Da](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)**

:   通过拼图任务系统研究 SFT 与 RFT 对先验知识的影响，揭示 RFT 避免灾难性遗忘的核心在于**数据分布**而非算法差异——RFT 采样的数据天然与基模型概率景观对齐，干扰更小。
