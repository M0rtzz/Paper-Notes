---
title: >-
  ICLR2026 多模态VLM方向 76篇论文解读
description: >-
  76篇ICLR2026 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🔬 ICLR2026** · **76** 篇论文解读

**[A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)**

:   提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

**[AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)**

:   通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

**[BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)**

:   提出 BioCAP，通过用 MLLM 生成 wiki 知识引导的合成描述性 caption（而非仅用物种标签）来训练生物学多模态基础模型，在 10 个物种分类 benchmark 上比 BioCLIP 平均提升 8.8%，在文本-图像检索任务上提升 21.3%。

**[Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)**

:   构建 Bongard-RWR+，一个包含 5400 个 Bongard 问题的 benchmark，使用 VLM 流水线（Pixtral-12B + Flux.1-dev）自动生成真实感图像来表示抽象概念，系统评估揭示 SOTA VLM 在辨别细粒度视觉概念（如轮廓、旋转、角度）时表现挣扎，准确率低至 19%。

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)**

:   本文提出 TuneCLIP，一个自监督微调（SSFT）框架，通过两阶段设计——先恢复优化器统计量（OSR）消除冷启动偏差，再用带margin的铰链全局对比损失（HGCL）缓解假负样本过度惩罚——在不使用任何标签的条件下持续提升已有开源 CLIP 模型的通用性能，在 ImageNet 及变体上提升最高 +2.5%，在 DataComp 基准上提升 +1.2%。

**[Can Vision-Language Models Answer Face to Face Questions in the Real-World?](can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)**

:   提出 QIVD（Qualcomm Interactive Video Dataset），一个面对面实时问答 benchmark（2900 个视频+音频+时间戳标注），揭示现有 VLM 在实时情境理解上远落后人类（最佳模型 60% vs 人类 87%），主要瓶颈在指代消歧、回答时机判断和情境常识，微调可显著缩小差距。

**[Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)**

:   针对 MoE 推理时因 token 分配不均导致的 Straggler Effect（最重负载专家决定整体延迟），提出 Capacity-Aware Token Drop（丢弃过载专家的低分 token）和 Expanded Drop（将溢出 token 重路由到本地低负载专家），在 Mixtral-8×7B 上实现 1.85× 加速且性能提升 0.2%。

**[Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)**

:   提出 PRPO（Parallel Relative Policy Optimization），通过在奖励维度和数据类型两个层面做并行解耦优化，解决 GRPO 在多维奖励信号干扰和异构数据梯度冲突下的训练瓶颈；同时构建 MCDR-Bench，基于"错误唯一性原则"将主观生成评估转化为客观错误识别，实现图表深度研究能力的量化评估。

**[CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)**

:   构建 CityLens——迄今最大规模的城市社会经济感知 benchmark（17 城市、6 大领域、11 个预测任务），评估 17 个 LVLM 在直接预测、归一化估计、特征回归三种范式下从卫星/街景图像推断社会经济指标的能力，发现通用 LVLM 在多数任务上仍不及领域特化的对比学习方法。

**[Closing the Modality Gap Aligns Group-Wise Semantics](closing_the_modality_gap_aligns_group-wise_semantics.md)**

:   证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。

**[Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)**

:   提出多模态语义扰动框架检测VLM数据污染：用LLM生成密集描述 + Flux ControlNet在保持图像构图的同时改变答案相关语义元素，污染模型因记忆原始图文对而在扰动版本上表现骤降，干净模型则因真正推理能力而不受影响。首次系统验证现有LLM污染检测方法在VLM场景下大多不可靠。

**[Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach](customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)**

:   提出情感陈述判断（ESJ）任务与 INSETS 自动标注流水线，将视觉情感评估从"开放式分类"重构为"陈述真伪判断"，构建了 MVEI benchmark（3,086 样本、424 种情感标签、四个认知维度），系统评估 19 个 MLLMs，发现即使 GPT-4o 也与人类（91.6%）存在 13.3% 的准确率差距。

**[Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)**

:   提出 EUQ（Evidential Uncertainty Quantification），基于 Dempster-Shafer 证据理论将 LVLM 的认识不确定性分解为**冲突 CF**（内部矛盾）和**无知 IG**（信息缺失），无需训练、单次前向传播即可检测幻觉/越狱/对抗/OOD 四类错误行为，平均 AUROC 相对最佳基线提升 10.4%/7.5%。

**[Directional Embedding Smoothing for Robust Vision Language Models](directional_embedding_smoothing_for_robust_vision_language_models.md)**

:   将 RESTA（Randomized Embedding Smoothing and Token Aggregation）防御方法从 LLM 扩展到 VLM，发现方向性嵌入噪声（directional noise）在安全-实用性权衡上显著优于各向同性噪声（isotropic noise），可作为推理时的轻量防御层抵御多模态越狱攻击。

**[DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)**

:   提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。

**[Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)**

:   本文基于 Nissenbaum 的上下文完整性（Contextual Integrity）理论构建了 VLM-GEOPRIVACY 基准，通过7个层次递进的上下文感知问题和三级位置披露粒度（拒绝/城市级/精确位置），系统评估14个主流VLM是否能根据图像中的社会规范线索判断适当的位置信息披露级别，结果发现所有模型均严重偏向过度披露（Over-Disclosure率高达46-52%），且恶意提示可将抽象违反率推至100%。

**[Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)**

:   提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

**[Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)**

:   提出Continual Decoupling-Unifying（ConDU）框架，首次将模型融合引入VLM持续学习，通过维护统一模型并结合任务触发器进行解耦-统一迭代操作，在MTIL基准上平均性能超SOTA 2%，同时增强了零样本能力。

**[Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)**

:   通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。

**[Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models](error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)**

:   提出一种无训练的两阶段VLM框架，通过Error Notebook记录纠正后的推理轨迹并结合RAG进行推理时适应，在3D CAD装配体的规格驱动零件检索任务上，GPT-4o准确率从41.7%提升至65.1%（+23.4%），并通过语法约束验证器进一步提升4.5%。

**[FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)**

:   提出 FRIEDA 基准，系统评估大型视觉语言模型在多步骤、跨地图的制图推理能力，发现最强模型 Gemini-2.5-Pro 准确率仅 38.20%，远低于人类 84.87%。

**[Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)**

:   提出 GAR（Grasp Any Region），通过 RoI-aligned feature replay 在保持全局上下文的同时提取高保真局部特征，实现精准的单区域描述、多区域交互建模和复合推理，1B 模型即超越 InternVL3-78B。

**[Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)**

:   将空间定位（referring + grounding）与图像质量评估结合，构建 GIQA-160K 数据集训练多模态 LLM 生成带有边界框的质量描述和空间 VQA，在细粒度质量感知上显著优于通用 MLLM。

**[GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)**

:   提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

**[ICYM2I: The Illusion of Multimodal Informativeness under Missingness](icym2i_the_illusion_of_multimodal_informativeness_under_missingness.md)**

:   揭示了多模态学习中被忽视的问题：模态缺失（missingness）导致的分布偏移会使模态价值评估产生严重偏差，提出 ICYM2I 框架通过双重逆概率加权（IPW）纠正训练和评估中的偏差，在 MAR 假设下实现对模态预测效用和信息论价值的无偏估计。

**[Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)**

:   在 VLM 视觉编码器之前插入一个仅 203K 参数的二值 patch 分类器剔除文档背景 token，再用 $3 \times 3$ max-pooling 恢复碎片化文本区域并保留原始空间索引，在 Qwen2.5-VL 上实现 40-60% FLOPs 缩减且精度损失不超过 ~5%p。

**[IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)**

:   揭示了LVLM中RoPE位置编码隐式建立的视觉坐标系统（IVC tokens），提出一种训练免的、提示感知的视觉token剪枝策略，在保留IVC tokens和语义前景token的同时，削减约50%视觉token并维持≥99%原始性能。

**[KeepLoRA: Continual Learning with Residual Gradient Adaptation](keeplora_continual_learning_with_residual_gradient_adaptation.md)**

:   通过分析预训练模型权重的SVD分解，发现通用知识编码在主子空间、领域特定知识编码在残差子空间，提出KeepLoRA方法将新任务的LoRA更新约束在残差子空间中，同时用梯度信息初始化以保持可塑性，在持续学习中达到前向稳定、后向稳定和可塑性的最优平衡。

**[LiveWeb-IE: A Benchmark For Online Web Information Extraction](liveweb-ie_a_benchmark_for_online_web_information_extraction.md)**

:   提出首个面向在线网页的信息抽取（WIE）基准LiveWeb-IE，覆盖文本/图片/超链接等多类数据抽取，并设计Visual Grounding Scraper（VGS）框架，通过模拟人类认知过程——视觉扫描定位区域→精确定位元素→生成XPath——在动态网页上实现鲁棒的信息抽取。

**[LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)**

:   提出 LLaVA-FA，一种在频域进行联合低秩加量化权重近似的高效多模态大模型压缩方法，利用傅里叶变换的去相关性和共轭对称性实现更紧凑准确的权重表示，并引入 PolarQuant（极坐标量化）和 ODC（可选对角校准）方案，在多个基准上以最少的激活参数和计算成本超越现有高效多模态模型。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)**

:   提出 MAPD（Meta-Adaptive Prompt Distillation），一种基于 MAML 元学习的提示蒸馏方法，通过注意力映射器从任务相关的图像特征中蒸馏软提示，使 LMM 在测试时仅用少量梯度步即可适应新的视觉问答任务，性能超越 ICL 21.2%。

**[Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)**

:   发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

**[MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)**

:   提出 MMR-Life 基准（2646 道 5 选 1 多图选择题，基于 19108 张真实图像，覆盖 7 种推理类型和 21 个任务），首次系统评估 MLLM 在真实生活场景中的多图推理能力，发现最强模型 GPT-5 仅 58.69% 准确率，距人类水平差 14%，并揭示了推理增强方法在大模型上失效、RL 泛化弱于 BoN 等关键发现。

**[MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)**

:   提出MMTok——一种基于最大覆盖问题（Maximum Coverage Problem）的多模态视觉token选择框架，同时利用文本-视觉和视觉-视觉覆盖信息来选择最具信息量的视觉token子集，在training-free设置下显著优于单模态baseline，甚至超越需要微调的方法。

**[Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)**

:   本文发现并系统定义"模态失语"（Modal Aphasia）现象——统一多模态模型能从记忆中近乎完美地生成视觉概念（如电影海报图像），但在文字描述同一概念时错误率高出 7 倍以上，且严重幻觉几乎只出现在文本模态；通过前沿模型（ChatGPT-5）的真实实验和开源模型（Janus-Pro、Harmon）的合成控制实验，证实模态失语是当前统一架构的系统性缺陷而非训练偶然，并展示了该现象对 AI 安全框架的潜在威胁。

**[Multimodal Classification via Total Correlation Maximization](multimodal_classification_via_total_correlation_maximization.md)**

:   从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

**[Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)**

:   首次将自动提示优化（APO）从纯文本空间扩展到多模态空间，提出 MPO 框架：通过对齐保持的联合探索（统一语义梯度同步驱动文本+非文本提示更新，配合 Generation/Edit/Mix 三种算子多样化搜索）和先验继承的贝叶斯 UCB 候选选择（利用父提示性能 warm-start 子提示的 Beta 先验），在图像/视频/分子共 10 个数据集上平均准确率达 65.1%，超越最强文本 APO 基线 ProTeGi 的 60.0%。

**[OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)**

:   基于认知心理学构建OmniSpatial——首个全面空间推理基准，系统覆盖动态推理、复杂空间逻辑、空间交互和透视转换4大维度50个子类别共8.4K人工标注QA对，让o3最强推理模型仅达56.33%而人类达92.63%→揭示复杂空间推理仍是VLM的核心瓶颈。

**[Post-hoc Probabilistic Vision-Language Models](post-hoc_probabilistic_vision-language_models.md)**

:   提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。

**[PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)**

:   提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息。PPE是零参数、即插即用的通用算子，在55%压缩率下图像任务平均仅降3.6%、在90%压缩率下通过级联压缩仍保持可比性能。

**[PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies](prismm-bench_a_benchmark_of_peer-review_grounded_multimodal_inconsistencies.md)**

:   构建首个基于真实审稿人标记的科学论文多模态不一致性基准PRISMM-Bench，从18009条ICLR开放评审中挖掘384个跨模态不一致，设计识别/修复/配对匹配三任务并提出JSON结构化去偏答案表示，21个顶级LMM最高仅53.9%→系统性暴露当前模型在科学文档跨模态推理上的严重不足。

**[Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)**

:   提出 RD-MLDG——首个将 MLLM 推理链引入域泛化的框架。构建 DomainBed-Reasoning 数据集，系统分析推理监督的两大挑战（优化困难 + 推理模式不匹配），通过 MTCT（多任务交叉训练）与 SARR（自对齐推理正则化）协同解决，在 4 个标准 DG 基准上以 86.89% 的平均准确率大幅超越 GPT-4o（83.46%）和所有 CLIP/ViT 方法。

**[Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks](ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks.md)**

:   提出 Ref-Adv 基准数据集，通过 **硬干扰物配对 + LLM 辅助最小充分表达式生成 + 三人一致性人工验证** 的流水线，构建了一个消除"定位捷径"的现代 REC 基准，在该基准上 13 个当代 MLLM（包括 GPT-4o、Gemini 2.5、Qwen2.5-VL-72B 等）的准确率从 RefCOCO(+/g) 上的 90%+ 大幅下降至 50-68%，系统暴露了模型在复杂视觉推理和真实定位能力上的严重不足。

**[Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)**

:   从混合专家（MoE）视角揭示 VPT 的局限性——prompt experts 是输入无关的常量函数表达力受限，提出 VAPT 通过 token-wise 投影器和共享特征投影器使 prompt experts 自适应输入，用更少参数实现更优性能，并给出了最优样本效率的理论保证。

**[Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)**

:   提出 MV-RoboBench，首个结合多视角空间推理与机器人操作任务的基准，系统评估了 40+ 个 VLM（开源+闭源+推理增强），发现最强模型 GPT-5 仅达 56.4% 准确率，远低于人类的 91.0%，并揭示空间与机器人推理正相关、单视角基准表现无法可靠迁移至多视角场景。

**[Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)**

:   提出 Self-Aug，一种免训练的解码策略，通过自增强提示（SAS Prompting）让 LVLM 利用自身知识动态选择与查询语义对齐的视觉增强方式，并提出稀疏度自适应截断（SAT）算法利用输出分布的完整熵信息动态调节候选词集大小，在5个 LVLM 和7个基准上一致超越现有对比解码方法。

**[Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)**

:   提出 EvoQuality 框架，通过成对多数投票生成伪排序标签、结合 GRPO 自迭代优化，使 VLM 在无人工标注下自主提升图像质量感知能力，零样本性能提升 31.8% PLCC，在 7 个 IQA 基准中 5 个超越有监督 SOTA。

**[Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)**

:   提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

**[Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)**

:   借鉴 Speculative Decoding 的 draft-then-verify 范式提出 Speculative Verdict (SV)，用多个轻量 VLM 生成多样推理路径作为 draft，大模型作为 verdict 综合验证并纠错，在信息密集型 VQA 上无需训练即超过 GPT-4o 达 11.9%，且能修复 47-53% 的少数正确案例。

**[SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)**

:   提出SophiaVL-R1——在规则基RL训练MLLM推理时引入整体级思维过程奖励：训练Thinking Reward Model从逻辑一致性/冗余度等五维度评估推理质量→提出Trust-GRPO基于正确/错误答案组的思维奖励对比计算可信度权重$\gamma$缓解reward hacking→退火策略$e^{-\text{steps}/T}$渐减思维奖励使后期更依赖准确的规则奖励→7B模型在MathVista(71.3%)和MMMU(61.3%)等多个基准全面超越LLaVA-OneVision-72B。

**[Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)**

:   提出Sparsity Forcing——基于GRPO的RL后训练框架，将带稀疏注意力的MLLM作为策略模型、原始MLLM作为参考模型，通过多预算rollout探索不同token保留阈值$p$，以效率(token减少率)+性能(答案正确性)为联合奖励做组内对比优化，将Qwen2/2.5-VL的token减少率从20%提升至75%且精度损失极小，实现内存降3×、解码加速3.3×。

**[Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)**

:   提出基于认知科学 2×2 分类法（内在/外在 × 静态/动态）的统一空间推理基准 Spatial-DISE，包含 559 个评估 VQA 对和 12K+ 训练数据，在 32 个 SOTA VLM 上的评测揭示了模型在动态空间推理（尤其是心理旋转和折叠）上与人类的巨大差距。

**[Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)**

:   提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率，远低于人类表现。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架下控制实验，系统性地研究图像编码器训练目标和 2D 位置编码对 VLM 空间推理能力的影响，发现编码器选择主导空间性能、AIMv2 编码器一致性最好，但 2D-RoPE 的改进不稳定，空间推理的失败根植于当前 VLM 流水线的核心设计选择。

**[SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)**

:   提出SpatiaLab，一个包含1400个视觉QA对的真实场景空间推理基准，涵盖6大类30子类空间任务，支持多选和开放式双格式评估，揭示当前最强VLM（InternVL3.5-72B MCQ 54.93%）与人类（87.57%）之间存在巨大空间推理鸿沟，且开放式设置下差距更大。

**[SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)**

:   提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。

**[Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)**

:   提出 SteerVAD 框架，在完全冻结的多模态大语言模型 (MLLM) 内部，通过识别"潜在异常专家"注意力头并用层次化元控制器动态操控其表示流形，仅用 1% 训练数据即实现免调优视频异常检测的 SOTA。

**[TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)**

:   提出 TableDART，通过仅 2.59M 参数的 MLP 门控网络为每个 query-table 对动态选择最优处理路径（Text-only / Image-only / Fusion），复用冻结的单模态专家模型并引入 LLM Agent 进行跨模态融合，在 7 个表格理解 benchmark 上平均超越最强 MLLM 基线 HIPPO 4.02%，同时延迟降低 24.5%。

**[ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)**

:   提出 VC-STaR（Visual Contrastive Self-Taught Reasoner），基于"VLM 在对比两张相似图像时看得更准"的观察，设计了一套对比式自改进框架：通过构造对比 VQA 对让模型在对比中生成更忠实的视觉分析，再由 LLM 将对比分析融入推理路径，产出高质量视觉推理数据集 VisCoR-55K，微调后在 MMVP 上提升 5.7%、Hallusion 上提升 3.2%。

**[U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)**

:   系统消融MLLM嵌入学习的设计空间，揭示双向注意力+mean pooling优于主流last token、可学习温度被严重低估等关键因子，据此构建U-MARVEL三阶段框架（渐进过渡→过滤硬负→重排蒸馏），在M-BEIR上以单模型63.2% Avg大幅超越现有SOTA，零样本迁移CIR和T2V同样领先。

**[Unified Vision-Language Modeling via Concept Space Alignment](unified_vision-language_modeling_via_concept_space_alignment.md)**

:   提出v-Sonar将视觉编码器后置对齐到文本嵌入空间Sonar，使得在Sonar空间上训练的Large Concept Model (LCM)能零样本处理视觉输入，并通过指令微调扩展为v-LCM，在61/62种语言上超越现有VLM。

**[UniHM: Unified Dexterous Hand Manipulation with Vision Language Model](unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)**

:   提出UniHM，首个统一的语言条件灵巧手操控框架，通过形态无关VQ codebook将异构机械手映射到共享离散空间，结合VLM进行指令驱动操控序列生成，并通过物理引导动态优化确保物理可行性。

**[VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)**

:   提出VisioMath基准，包含1800道K-12数学题目，所有选项均为高度视觉相似的图表，揭示了LMM在多图像-文本对齐上的核心短板，并探索三种对齐策略实现+12.6%的提升。

**[Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)**

:   提出Vision-R1，通过Modality Bridging构建200K高质量多模态CoT数据进行冷启动初始化，再用渐进思维抑制训练(PTST)策略结合GRPO强化学习，在7B参数规模达到与OpenAI O1接近的多模态数学推理能力。

**[Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)**

:   提出 Vision-Zero，首个无标注的游戏化自博弈框架，通过"谁是卧底"式视觉推理游戏实现 VLM 的可扩展自进化，结合 Iterative-SPO 训练算法在推理、图表理解和视觉中心任务上超越基于人工标注数据的 SOTA 方法。

**[VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)**

:   提出首个面向数据可视化美学与质量评估的综合基准 VisJudge-Bench（3,090 样本，32 种图表类型），并训练 VisJudge 模型，将 MAE 相比 GPT-5 降低 23.9%，与人类专家的一致性提升 60.5%。

**[Visual Prompt-Agnostic Evolution](visual_prompt-agnostic_evolution.md)**

:   提出 Prompt-Agnostic Evolution (PAE)，通过频域感知的任务初始化 (MPA) 和 Koopman-Lyapunov 动力系统 (KLD) 跨层关联 prompt，加速 VPT 收敛（平均 1.41× 加速）并在 25 个数据集上提升 1–3% 精度，且对各类 VPT 变体即插即用、无推理开销。

**[Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](visual_symbolic_mechanisms_vlm.md)**

:   发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。

**[WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)**

:   提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。

**[Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems](why_keep_your_doubts_to_yourself_trading_visual_uncertainties_in_multi-agent_ban.md)**

:   提出 Agora 框架，将多智能体 VLM 协调问题重构为去中心化的不确定性交易市场——将认知不确定性铸造为可量化的三维可交易资产（感知/语义/推理），通过利润驱动的交易协议和市场感知的 Thompson Sampling Broker 实现成本高效的均衡分配，在 5 个多模态基准上一致超越启发式方法（如 MMMU 上 +8.5% 准确率同时成本降低 3 倍以上）。

**[Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems](why_keep_your_doubts_to_yourself_trading_visual_uncertainty.md)**

:   提出 Agora 框架，将多智能体 VLM 协调问题重新建模为去中心化的不确定性交易市场，通过将认知不确定性拆分为可交易资产（感知/语义/推理三维），并用基于盈利性驱动的交易协议和 Thompson Sampling 代理人实现成本感知的最优分配，在五个多模态基准上以超 3 倍成本节省获得至多 +8.5% 准确率提升。

**[Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)**

:   通过拼图任务系统研究 SFT 与 RFT 对先验知识的影响，揭示 RFT 避免灾难性遗忘的核心在于**数据分布**而非算法差异——RFT 采样的数据天然与基模型概率景观对齐，干扰更小。
