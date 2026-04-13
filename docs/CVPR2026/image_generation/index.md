---
title: >-
  CVPR2026 图像生成方向 164篇论文解读
description: >-
  164篇CVPR2026 图像生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**📷 CVPR2026** · 共 **164** 篇

**[2Ndmatch Finetuning Pruned Diffusion Models Via Second-Order Jacobian Matching](2ndmatch_finetuning_pruned_diffusion_models_via_second-order_jacobian_matching.md)**

:   提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。

**[Accelerating Diffusion Model Training Under Minimal Budgets A Condensation-Based](accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)**

:   提出 D2C（Diffusion Dataset Condensation）——首个面向扩散模型的数据集压缩框架，通过"Select + Attach"两阶段流水线，在仅使用 ImageNet 0.8%–8% 数据的条件下实现 100–233× 的训练加速，同时保持高质量图像生成能力。

**[Adapt Attention Driven Adaptive Prompt Scheduling And Interpolating Orthogonal C](adapt_attention_driven_adaptive_prompt_scheduling_and_interpolating_orthogonal_c.md)**

:   提出 ADAPT 框架，通过注意力驱动的自适应 Prompt 调度（APS）、池化嵌入操控（PEM）和潜空间操控（LSM）三个零样本模块，确定性且语义对齐地控制从通用到罕见概念的生成过渡，在 RareBench 上显著超越 R2F 基线。

**[Adapting A Pre-Trained Single-Cell Foundation Model To Spatial Gene Expression G](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)**

:   提出HINGE框架，首次将预训练的表达空间单细胞基础模型(sc-FM, CellFM)改装为组织学图像条件的空间基因表达生成器，通过恒等初始化的SoftAdaLN调制轻量注入视觉上下文、表达空间掩码扩散过程对齐预训练目标、warm-start课程稳定训练，在三个ST数据集上达SOTA并保持优越的基因共表达一致性。

**[Adaptive Spectral Feature Forecasting For Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)**

:   提出 Spectrum，一种基于切比雪夫多项式的全局谱域特征预测方法，将扩散模型去噪器的中间特征视为时间函数并用岭回归拟合系数，实现误差不随步长增长的长程特征预测，在 FLUX.1 上达到 4.79× 加速、在 Wan2.1-14B 上达到 4.67× 加速而质量几乎无损。

**[Agentic Retoucher For Text-To-Image Generation](agentic_retoucher_for_text-to-image_generation.md)**

:   将 T2I 扩散模型输出的局部失真（手指畸变、面部异常、文字错误等）校正问题建模为感知-推理-行动的多智能体循环系统 Agentic Retoucher，通过 Perception Agent 的上下文感知失真显著性图定位缺陷、Reasoning Agent 的结构化推理诊断失真类型、Action Agent 的工具选择执行修复，并配合 GenBlemish-27K 数据集实现端到端的迭代式自动修正。

**[Agentic Retoucher For Texttoimage Generation](agentic_retoucher_for_texttoimage_generation.md)**

:   Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。

**[Alignvar Towards Globally Consistent Visual Autoregression For Image Super-Resol](alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)**

:   针对视觉自回归（VAR）模型在图像超分辨率中的两个一致性问题——注意力局部偏差导致的空间不连贯和残差监督导致的跨尺度误差累积，提出 AlignVAR 框架，通过空间一致性自回归（SCA）和层级一致性约束（HCC）协同解决，实现比扩散方法快 10× 以上的推理速度且重建质量更优。

**[All-In-One Slider For Attribute Manipulation In Diffusion Models](all-in-one_slider_for_attribute_manipulation_in_diffusion_models.md)**

:   提出 All-in-One Slider 框架，通过在文本编码器中间层嵌入上训练一个轻量级 Attribute Sparse Autoencoder，将属性分解为高维稀疏激活空间中的解耦方向，从而用单一模块实现对多种面部属性的连续、细粒度、可组合控制，并首次展示对未见属性（如种族、名人）的零样本连续操控能力。

**[All In One Slider Attribute Manipulation](all_in_one_slider_attribute_manipulation.md)**

:   提出 All-in-One Slider 框架，通过在文本嵌入空间上训练一个属性稀疏自编码器（Attribute Sparse Autoencoder），将多种人脸属性解耦为稀疏的语义方向，实现单一轻量模块对 52+ 种属性的细粒度连续控制，并支持多属性组合和未见属性的零样本操控。

**[Ani3Dhuman Photorealistic 3D Human Animation With Self-Guided Stochastic Samplin](ani3dhuman_photorealistic_3d_human_animation_with_self-guided_stochastic_samplin.md)**

:   提出 Ani3DHuman 框架，将运动学驱动的网格动画与视频扩散先验相结合，通过自引导随机采样（Self-guided Stochastic Sampling）将低质量的刚体渲染恢复为高保真视频，从而实现逼真的非刚体服装动态建模。

**[Anti-I2V Safeguarding Your Photos From Malicious Image-To-Video Generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)**

:   Anti-I2V 提出了一种针对恶意图像到视频生成的防御方法，通过在 L\*a\*b\* 和频域双空间优化扰动，并设计内部表示崩塌（IRC）和锚定（IRA）损失破坏去噪网络的语义特征传播，在 CogVideoX、DynamiCrafter 和 Open-Sora 三种不同架构上实现 SOTA 防护效果。

**[Apple Attribute-Preserving Pseudo-Labeling For Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)**

:   APPLE 提出了一种基于扩散模型的教师-学生框架，通过条件去模糊（代替传统条件修复）训练教师模型生成属性对齐的伪标签，再利用这些高质量伪标签训练学生模型，在保持身份迁移能力的同时实现了 SOTA 的属性保留性能（FID 2.18, Pose Error 1.85）。

**[Ar2Can An Architect And An Artist Leveraging A Canvas For Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)**

:   Ar2Can 提出将多人图像生成分解为空间规划（Architect）和身份保留渲染（Artist）两阶段，通过 GRPO 强化学习配合基于匈牙利匹配的空间锚定人脸奖励函数训练 Artist 模型，在 MultiHuman-Testbench 上实现了 68.2 的身份保留分数和 90.2 的计数准确率，大幅超越所有基线。

**[As-Bridge A Bidirectional Generative Framework Bridging Next-Generation Astronom](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，一个基于 Brownian Bridge 扩散过程的双向生成框架，在地基 LSST 与空基 Euclid 天文巡天之间建模概率条件分布，实现跨巡天图像翻译和罕见事件检测（引力透镜），并通过 $\epsilon$-prediction 训练目标改进了标准 Brownian Bridge 的似然估计。

**[Asbridge A Bidirectional Generative Framework Brid](asbridge_a_bidirectional_generative_framework_brid.md)**

:   提出 AS-Bridge，基于双向 Brownian Bridge 扩散过程建模地面巡天（LSST）与空间巡天（Euclid）观测之间的随机映射，同时实现跨巡天图像转换和稀有天文事件检测。

**[Attention May I Have Your Decision Localizing Generative Choices In Diffusion Mo](attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)**

:   本文通过线性探针（linear probing）发现扩散模型中**隐式决策**（如未指定性别时默认生成男性）主要由自注意力层而非交叉注意力层控制，并基于此提出 ICM 方法，仅在少量关键自注意力层上进行干预即可实现 SOTA 的去偏见效果，同时最小化图像质量退化。

**[Attribution As Retrieval Model-Agnostic Ai-Generated Image Attribution](attribution_as_retrieval_model-agnostic_ai-generated_image_attribution.md)**

:   将 AI 生成图像归因从分类范式转为实例检索范式，提出 LIDA 框架：利用 RGB 低位平面提取生成器特有指纹作为输入，通过在真实图像上无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上以 1-shot 设置即取得 40.4%/77.5% 的平均 Rank-1 准确率，大幅超越现有方法。

**[Attribution As Retrieval Modelagnostic Aigenerated](attribution_as_retrieval_modelagnostic_aigenerated.md)**

:   将 AI 生成图像归因从分类范式重新定义为实例检索问题，提出 LIDA 框架：利用低位平面提取生成器指纹，通过无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上全面超越现有方法。

**[Autodebias Automated Framework For Debiasing Text-To-Image Models](autodebias_automated_framework_for_debiasing_text-to-image_models.md)**

:   提出 AutoDebias——首个同时检测和缓解 T2I 模型中恶意后门偏见的统一框架，利用 VLM 开放集检测发现触发词-偏见关联并构建查找表，再通过 CLIP 引导的分布对齐训练消除后门关联，在 17 种后门场景中将攻击成功率从 90% 降至接近 0 且保持图像质量。

**[Banana100 Breaking Nr-Iqa Metrics By 100 Iterative Image Replications With Nano ](banana100_breaking_nr-iqa_metrics_by_100_iterative_image_replications_with_nano_.md)**

:   Banana100 通过让 Nano Banana Pro 迭代复制图像 100 次来系统性研究多轮编辑中的质量退化问题，构建了包含 28,000 张退化图像的数据集，并揭示了一个惊人发现：21 种主流无参考图像质量评估（NR-IQA）指标均无法可靠检测迭代退化——大多数指标甚至给噪声图像打出比干净图像更高的分数。

**[Beyond The Golden Data Resolving The Motion-Vision Quality Dilemma Via Timestep ](beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)**

:   发现视频数据中运动质量（MQ）和视觉质量（VQ）呈负相关的"Motion-Vision Quality Dilemma"，通过梯度分析揭示不平衡数据在适当时间步可产生等效学习信号，提出TQD框架使仅用不平衡数据训练即可超越黄金数据训练。

**[Bigain Token Compression](bigain_token_compression.md)**

:   提出BiGain——一个训练免的token压缩框架，通过频域分离（保留高频细节+低中频语义），在扩散模型加速时同时保持生成质量和分类能力。70% token合并下分类精度+7.15%且FID反而更好。

**[Bigain Unified Token Compression For Joint Generation And Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 提出频率感知的 token 压缩框架，通过拉普拉斯门控 token 合并和插值-外推 KV 下采样两个无训练算子，首次在扩散模型加速中同时保持生成质量并显著提升判别分类性能。

**[Bimotion B-Spline Motion For Text-Guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)**

:   提出 BiMotion，用连续可微的 B 样条曲线将变长运动序列压缩为固定数量控制点，配合专用 VAE 和 flow-matching 扩散模型，实现快速、高表达力、语义完整的文本引导动态 3D 角色生成，在质量和效率上均超越现有方法。

**[Biovita Biological Dataset Model And Benchmark For Visual-Textual-Acoustic Align](biovita_biological_dataset_model_and_benchmark_for_visual-textual-acoustic_align.md)**

:   提出 BioVITA 框架，包含百万级三模态（图像-文本-音频）生物数据集、两阶段对齐模型和六方向跨模态物种级检索基准，首次实现生物领域视觉-文本-声音统一表示学习。

**[Blackmirror Black-Box Backdoor Detection For Text-To-Image Models Via Instructio](blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)**

:   提出 BlackMirror 框架，通过细粒度的指令-响应语义偏差检测（MirrorMatch）和跨 prompt 稳定性验证（MirrorVerify）两阶段流程，在黑盒条件下实现对 T2I 模型多种后门攻击的通用检测，F1 平均达 89.46%，大幅超越已有黑盒方法 UFID。

**[Care-Edit Condition-Aware Routing Of Experts For Contextual Image Editing](care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)**

:   提出 CARE-Edit，一种条件感知的专家路由框架，通过异构专家（Text/Mask/Reference/Base）配合轻量级 latent-attention 路由器，在 DiT 骨干上实现动态计算分配，有效解决统一图像编辑器中多条件信号（文本、掩码、参考图）冲突导致的颜色溢出、身份漂移等问题。

**[Careflow Cyclic Adaptive Rectified Flow For Multimodal Fusion](careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)**

:   提出 CaReFlow，首次将 rectified flow 用于多模态分布映射以缩小模态间隙：通过 one-to-many mapping 让源模态数据点观测目标模态全局分布，adaptive relaxed alignment 对不同关联度的模态对施加不同对齐强度，cyclic rectified flow 保证映射后信息不丢失，即使用简单拼接融合也能在多个多模态情感计算 benchmark 上达到 SOTA。

**[Causal Motion Diffusion Models For Autoregressive Motion Generation](causal_motion_diffusion_models_for_autoregressive_motion_generation.md)**

:   提出 CMDM 框架，在运动-语言对齐的因果隐空间中统一扩散去噪与自回归生成，通过帧级独立噪声和因果不确定性采样调度，实现高质量、低延迟的文本到动作生成和长序列流式合成。

**[Cdg Condition Degradation Guidance Diffusion](cdg_condition_degradation_guidance_diffusion.md)**

:   提出CDG替代CFG——用语义退化条件替代空null prompt作为负面引导，将引导信号从粗粒度"好vs空"变为精细"好vs差一点"，在SD3/FLUX/Qwen-Image上显著提升组合精度，零额外计算。

**[Cfg-Ctrl Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)**

:   将 Classifier-Free Guidance (CFG) 重新解释为流匹配扩散模型中的反馈控制过程，提出统一框架 CFG-Ctrl，并基于滑模控制 (SMC) 设计非线性反馈引导机制 SMC-CFG，在大引导尺度下显著提升语义一致性和生成鲁棒性。

**[Chain Of Event-Centric Causal Thought For Physically Plausible Video Generation](chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)**

:   将物理现象建模为因果连接的事件序列，通过物理公式驱动的事件链推理分解复杂物理过程，再用渐进式语义-视觉双提示引导现成视频扩散模型生成物理合理的因果演进视频。

**[Changebridge Spatiotemporal Image Generation With Multimodal Controls For Remote](changebridge_spatiotemporal_image_generation_with_multimodal_controls_for_remote.md)**

:   提出 ChangeBridge，通过漂移异步扩散桥（drift-asynchronous diffusion bridge）实现遥感场景中从前事件到后事件的条件时空图像生成，支持坐标文本、语义掩码、实例布局等多模态控制，并可作为变化检测任务的数据生成引擎。

**[Chordedit One-Step Low-Energy Transport For Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)**

:   基于动态最优传输理论，推导出低能量的 Chord 控制场，将不稳定的朴素编辑场平滑化，首次实现了对蒸馏单步 T2I 模型的无训练、无反演、高保真实时图像编辑。

**[Cinematic Audio Source Separation Using Visual Cues](cinematic_audio_source_separation_using_visual_cues.md)**

:   提出首个音视频影视音频源分离（AV-CASS）框架，利用面部和场景双视频流的视觉线索，通过条件流匹配进行生成式三路音频分离（语音/音效/音乐），仅在合成数据上训练即可泛化到真实电影。

**[Circuit Mechanisms For Spatial Relation Generation In Diffusion Models](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)**

:   通过机械可解释性方法揭示了扩散Transformer（DiT）生成空间关系的内部电路机制：随机嵌入模型使用两阶段模块化电路（关系头+物体生成头），T5编码器模型则将关系信息融合到物体token中通过单token解码，两种机制的鲁棒性差异显著。

**[Circuit Mechanisms For Spatial Relation Generation In Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_transformers.md)**

:   通过机制可解释性方法，揭示了扩散Transformer中空间关系生成的两种截然不同的电路机制：随机文本编码器使用"关系头+物体头"的两阶段模块化电路，而 T5 编码器将关系信息融入物体 token 中通过单 token解码，后者在域外扰动下更脆弱。

**[Cod A Diffusion Foundation Model For Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)**

:   提出首个面向压缩的扩散基础模型 CoD，从零训练学习端到端的压缩-生成联合优化，替换 Stable Diffusion 后在下游扩散编解码器中实现超低码率（0.0039 bpp）下的 SOTA 性能，训练成本仅为 SD 的 0.3%。

**[Codrawagents A Multi-Agent Dialogue Framework For Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)**

:   提出 coDrawAgents，一个交互式多智能体对话框架（Interpreter-Planner-Checker-Painter），通过分而治之的增量布局规划、视觉上下文驱动的空间推理和显式错误纠正机制，大幅提升复杂场景下组合式文本到图像生成的忠实度。

**[Codrawagents A Multiagent Dialogue Framework For C](codrawagents_a_multiagent_dialogue_framework_for_c.md)**

:   提出 coDrawAgents 交互式多智能体对话框架，通过解释器、规划器、检查器、画家四个专业智能体的闭环协作，以分治策略逐步规划布局并基于画布视觉上下文纠错，在 GenEval 上达到 0.94 的 SOTA 组合保真度。

**[Cognitioncapturerpro Towards High-Fidelity Visual Decoding From Eegmeg Via Multi](cognitioncapturerpro_towards_high-fidelity_visual_decoding_from_eegmeg_via_multi.md)**

:   提出 CognitionCapturerPro，通过不确定性加权遮蔽（UM）、多模态融合编码器和共享主干-多头对齐（STH-Align），整合 EEG 信号与图像/文本/深度/边缘四种模态，在 THINGS-EEG 上实现 Top-1 检索准确率 61.2%、Top-5 达 90.8%，较前作 CognitionCapturer 提升 25.9% 和 10.6%。

**[Cognitioncapturerpro Towards Highfidelity Visual D](cognitioncapturerpro_towards_highfidelity_visual_d.md)**

:   提出 CognitionCapturerPro，通过不确定性加权掩蔽、多模态融合编码器、共享主干对齐模块和多分支 IP-Adapter 扩散重建，解决 EEG 视觉解码中的保真度损失和表征偏移问题，在 THINGS-EEG 上 Top-1 检索达 61.2%、Top-5 达 90.8%。

**[Cologen Progressive Learning Of Concept-Localization Duality For Unified Image G](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)**

:   提出 CoLoGen，一个基于"概念-定位对偶性"（Concept-Localization Duality）的统一图像生成框架，通过渐进式分阶段训练和 Progressive Representation Weaving（PRW）动态专家路由架构，在指令编辑、可控生成和个性化生成三大任务上同时达到或超越专用模型水平。

**[Consistcompose Multimodal Layout Control](consistcompose_multimodal_layout_control.md)**

:   提出 ConsistCompose，通过将布局坐标直接嵌入语言prompt（LELG范式），在统一多模态框架中实现布局可控的多实例图像生成；构建340万样本的ConsistCompose3M数据集提供布局+身份监督；配合坐标感知CFG机制，在COCO-Position上实现布局IoU 7.2%提升和AP 13.7%提升，同时保持通用理解能力。

**[Consistcompose Unified Multimodal Layout Control For Image Composition](consistcompose_unified_multimodal_layout_control_for_image_composition.md)**

:   提出 LELG（语言嵌入式布局引导生成）范式，将 bounding box 坐标直接编码为文本 token 嵌入语言流，在统一多模态 Transformer 中实现布局可控的多实例图像生成，无需任何布局专用编码器或分支。

**[Cot-Fm Cluster-Wise Optimal Transport Flow Matching](cot-fm_cluster-wise_optimal_transport_flow_matching.md)**

:   提出 COT-FM，一个即插即用的 Flow Matching 增强框架：通过聚类目标样本、反转预训练模型获取簇级源分布、在簇内近似最优传输，显著拉直传输路径，在不改变模型架构的前提下同时加速采样和提升生成质量。

**[Cross-Modal Emotion Transfer For Emotion Editing In Talking Face Video](cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)**

:   提出 C-MET（Cross-Modal Emotion Transfer），通过建模语音和面部表情空间之间的情感语义向量映射，首次实现了基于语音驱动的扩展情感（如讽刺、魅力）说话人脸视频生成，情感准确率超越 SOTA 14%。

**[Ctcal Rethinking Text-To-Image Diffusion Models Via Cross-Timestep Self-Calibrat](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)**

:   提出 CTCal（Cross-Timestep Self-Calibration），利用扩散模型在小时间步（低噪声）下形成的可靠文本-图像对齐（cross-attention maps）来校准大时间步（高噪声）下的表征学习，为文本到图像生成提供显式的跨时间步自监督，在 T2I-CompBench++ 和 GenEval 上全面超越现有方法。

**[Cubecomposer Spatio-Temporal Autoregressive 4K 360 Video Generation From Perspec](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)**

:   提出 CubeComposer，将360°视频分解为 cubemap 六面表示并按时空自回归方式逐面生成，首次实现从透视视频原生生成4K（3840×1920）分辨率的360°全景视频，无需后处理超分辨率。

**[Cycle-Consistent Tuning For Layered Image Decomposition](cycle-consistent_tuning_for_layered_image_decomposition.md)**

:   提出基于扩散模型的循环一致性微调框架，通过联合训练分解模型和合成模型实现图像层分离（如logo-物体分解），并引入渐进式自改进数据扩增策略，在非线性层交互场景下实现鲁棒分解。

**[D2C Diffusion Dataset Condensation](d2c_diffusion_dataset_condensation.md)**

:   首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。

**[Da-Vae Plug-In Latent Compression For Diffusion Via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)**

:   提出 Detail-Aligned VAE (DA-VAE)，通过结构化潜在空间（base + detail channels）和对齐损失，在不从头训练扩散模型的前提下将预训练 VAE 的压缩率提升至原来的 4 倍，仅需 5 H100-days 即可适配 SD3.5 生成 1024×1024 图像。

**[Denoising As Path Planning Training-Free Acceleration Of Diffusion Models With D](denoising_as_path_planning_training-free_acceleration_of_diffusion_models_with_d.md)**

:   将扩散模型采样加速形式化为全局路径规划问题，构建路径感知代价张量（PACT）量化跳步误差的路径依赖性，通过动态规划选择最优关键步序列，在FLUX上以4.87×加速超越全步基线+0.028 ImageReward。

**[Diff4Splat Controllable 4D Scene Generation With Latent Dynamic Reconstruction M](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)**

:   提出 Diff4Splat，一个前馈式框架，将视频扩散模型与可变形3D高斯场统一到端到端可训练的模型中，从单张图像在约30秒内直接生成动态4D场景表示，比优化方法快60倍。

**[Diffusion Mental Averages](diffusion_mental_averages.md)**

:   提出 Diffusion Mental Averages (DMA)，通过在扩散模型的语义空间中对齐多个去噪轨迹，从预训练扩散模型中提取概念的"心理平均"原型图像——首次实现一致、逼真的概念平均可视化。

**[Diffusion Probe Generated Image Result Prediction Using Cnn Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)**

:   发现扩散模型早期去噪步骤的交叉注意力分布与最终图像质量高度相关，提出 Diffusion Probe——用轻量CNN从早期注意力图预测生成结果质量，实现在完成10%去噪即可预筛选低质量生成路径，加速 Prompt 优化、Seed 选择和 GRPO 训练。

**[Diflowdubber Discrete Flow Matching For Automated Video Dubbing Via Cross-Modal ](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)**

:   提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。

**[Dip Taming Diffusion Models In Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)**

:   提出 DiP，一个高效的像素空间扩散框架，通过将 DiT backbone 在大patch上建模全局结构 + 轻量 Patch Detailer Head 恢复局部细节，实现了与LDM可比的计算效率但无需VAE，在ImageNet 256×256上达到1.79 FID。

**[Disca Accelerating Video Diffusion Transformers Wi](disca_accelerating_video_diffusion_transformers_wi.md)**

:   DisCa 首次提出"可学习特征缓存 + 步蒸馏"兼容的加速方案：用轻量神经预测器替代传统手工缓存策略，并通过 Restricted MeanFlow 稳定大规模视频模型的蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

**[Disca Accelerating Video Diffusion Transformers With Distillation-Compatible Lea](disca_accelerating_video_diffusion_transformers_with_distillation-compatible_lea.md)**

:   提出 DisCa，首次将**可学习特征缓存**与**步骤蒸馏**相结合，通过轻量级神经预测器替代手工缓存策略，并设计 Restricted MeanFlow 稳定大规模视频模型蒸馏，在 HunyuanVideo 上实现 11.8× 加速且几乎无质量损失。

**[Disentangling To Re-Couple Resolving The Similarity-Controllability Paradox In S](disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)**

:   提出 DisCo 框架，通过先解耦文本与视觉信息（用代词替换实体词消除文本对 subject 的干扰）、再用 GRPO + 专用 reward model 重新耦合二者，有效解决了 subject-driven 图像生成中"相似度-可控性"不可兼得的悖论。

**[Dit-Ic Aligned Diffusion Transformer For Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)**

:   提出 DiT-IC，将预训练T2I扩散Transformer通过三种对齐机制（方差引导重建流、自蒸馏对齐、潜表示条件引导）适配为单步图像压缩重建模型，在32×下采样的深层潜空间执行扩散，实现SOTA感知质量且解码速度比现有扩散压缩编解码器快30×。

**[Ditic Aligned Diffusion Transformer For Efficient](ditic_aligned_diffusion_transformer_for_efficient.md)**

:   将预训练文生图 DiT 适配为高效单步图像压缩解码器，通过方差引导重建流、自蒸馏对齐和潜空间条件引导三种对齐机制，在 32× 下采样的深层潜空间中实现 SOTA 感知质量，同时比现有扩散压缩方法解码快 30 倍。

**[Diversity Over Uniformity Rethinking Representation In Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)**

:   提出反特征坍塌学习框架 AFCL，通过信息瓶颈过滤无关特征并抑制不同伪造线索之间的过度重叠，保持判别表征的多样性和互补性，在跨模型生成图像检测上取得显著提升。

**[Dmin Scalable Training Data Influence Estimation For Diffusion Models](dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)**

:   提出 DMin，一个可扩展的扩散模型训练数据影响力估计框架，通过高效梯度压缩将存储需求从数百 TB 降至 MB/KB 级别，首次实现对数十亿参数扩散模型的影响力估计，支持亚秒级 top-k 检索。

**[Dpcache Denoising Path Planning Diffusion Accel](dpcache_denoising_path_planning_diffusion_accel.md)**

:   将扩散采样加速形式化为全局路径规划问题，通过Path-Aware Cost Tensor量化路径依赖的跳步误差，用动态规划选出最优关键时间步序列，在FLUX上实现4.87×加速且ImageReward反超全步基线。

**[Editing Away The Evidence Diffusion-Based Image Manipulation And The Failure Mod](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   本文从理论和实验两方面统一分析了非对抗性扩散编辑如何无意中破坏鲁棒隐形水印，推导了水印 SNR 衰减和互信息衰减的界，并在指令编辑、拖拽编辑、无训练合成等场景下验证了水印恢复的系统性失效。

**[Editing Away The Evidence Diffusionbased Image Man](editing_away_the_evidence_diffusionbased_image_man.md)**

:   本文从理论和实验两方面系统分析了扩散编辑（instruction/drag/composition）如何非对抗性地破坏鲁棒隐形水印，推导出 SNR 衰减和互信息下界，揭示常规后处理鲁棒性不能推广到生成式变换。

**[Effecterase Joint Video Object Removal And Insertion For High-Quality Effect Era](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)**

:   提出 EffectErase 框架，将视频物体插入作为移除的逆辅助任务进行联合学习，并构建包含 60K 视频对的大规模 VOR 数据集，实现对物体及其遮挡、阴影、反射、光照、变形等视觉副效应的高质量擦除。

**[Enhancing Image Aesthetics With Dual-Conditioned Diffusion Models Guided By Mult](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)**

:   提出 DIAE 框架，通过多模态美学感知（MAP）将模糊的美学指令转化为 HSV/轮廓图视觉信号 + 文本联合引导，并构建"不完美配对"数据集 IIAEData 实现弱监督的图像美学增强。

**[Enhancing Image Aesthetics With Dualconditioned Di](enhancing_image_aesthetics_with_dualconditioned_di.md)**

:   提出DIAE——一个基于SD1.5的图像美学增强框架，通过多模态美学感知(MAP)将模糊的美学指令转化为HSV+轮廓图的视觉控制信号，配合"不完美配对"数据集IIAEData和双分支监督训练策略，在美学提升(LAION score +17.4%)和内容一致性(CLIP-I 0.784)上同时优于InstructPix2Pix等SOTA编辑方法。

**[Enhancing Spatial Understanding In Image Generation Via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)**

:   构建 80K 对抗性偏好数据集 SpatialReward-Dataset，训练专门评估空间关系准确性的奖励模型 SpatialScore（准确率超越 GPT-5），并用 top-k 过滤策略结合 GRPO 在线 RL 显著提升 FLUX.1-dev 的空间生成能力。

**[Evatok Adaptive Length Video Tokenization For Efficient Visual Autoregressive Ge](evatok_adaptive_length_video_tokenization_for_efficient_visual_autoregressive_ge.md)**

:   提出 EVATok 四阶段框架，通过代理奖励（proxy reward）定义最优 token 分配，训练轻量路由器预测每段视频的最优 token 预算，实现内容自适应的可变长度视频 tokenization，在 UCF-101 上达到 SOTA 生成质量的同时节省至少 24.4% 的 token 用量。

**[Expportrait Expressive Portrait Generation Via Personalized Representation](expportrait_expressive_portrait_generation_via_personalized_representation.md)**

:   提出高保真度的个性化头部表征（静态身份偏移 + 动态表情偏移），解决 SMPL-X 等参数化模型表达力不足的问题，结合身份自适应表情迁移模块和 DiT 生成器，在人像视频自驱动和跨身份重演任务上取得 SOTA 表现。

**[Face2Scene Using Facial Degradation As An Oracle For Diffusion-Based Scene Resto](face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)**

:   提出 Face2Scene 两阶段框架：先用参考人脸复原模型(Ref-FR)获得 HQ-LQ 人脸对，从中提取退化编码作为"oracle"，再以此条件化单步扩散模型完成包含身体与背景的全场景图像复原。

**[Fastlightgen Fast And Light Video Generation With Fewer Steps And Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)**

:   FastLightGen 提出三阶段蒸馏算法，首次实现采样步数与模型大小的联合蒸馏，通过识别冗余层、动态概率剪枝和 well-guided teacher guidance 分布匹配，将 HunyuanVideo/WanX 压缩为 4 步 30% 参数剪枝的轻量生成器，实现约 35 倍加速且性能超越教师模型。

**[Fdeid-Toolbox Face De-Identification Toolbox](fdeid-toolbox_face_de-identification_toolbox.md)**

:   提出 FDeID-Toolbox，一个模块化的人脸去标识化工具箱，统一集成了 16 种去标识化方法（涵盖朴素/生成式/对抗式/K-Same 四大类）、6 个基准数据集和覆盖隐私保护/属性保持/视觉质量三维度的系统化评估协议，解决了该领域实现碎片化、评估不一致、结果不可比的问题。

**[Fdeidtoolbox Face Deidentification Toolbox](fdeidtoolbox_face_deidentification_toolbox.md)**

:   发布 FDeID-Toolbox，一个模块化人脸去识别研究工具箱，统一了数据加载、方法实现（经典到 SOTA 生成模型）、推理流水线和三维评估协议（隐私/效用/质量），解决该领域实验碎片化和结果不可比的问题。

**[Few-Shot Acoustic Synthesis With Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)**

:   提出 FLAC，首个基于 flow matching 的少样本房间脉冲响应（RIR）生成框架，仅凭单次录音即可在未见场景中合成空间一致的声学响应，并引入 AGREE 联合嵌入用于几何-声学一致性评估。

**[Flash-Unified A Training-Free And Task-Aware Acceleration Framework For Native U](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)**

:   FlashU 首次对原生统一多模态模型进行系统性冗余分析，发现参数特化和计算异质性现象，据此提出免训练任务感知加速框架，通过 FFN 剪枝、动态层跳过、自适应引导缩放和扩散头缓存，在 Show-o2 上实现 1.78x-2.01x 加速同时保持 SOTA 性能。

**[Fractals Made Practical Denoising Diffusion As Par](fractals_made_practical_denoising_diffusion_as_par.md)**

:   证明 DDIM 确定性反向链等价于分区迭代函数系统（PIFS），从分形几何推导出三个可计算量（收缩阈值 $L_t^*$、对角膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），统一解释了余弦调度偏移、分辨率 logSNR 偏移、Min-SNR 损失加权和 Align Your Steps 采样调度四种经验设计选择。

**[Fractals Made Practical Denoising Diffusion As Partitioned Iterated Function Sys](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明了DDIM确定性反向链本质上是一个分区迭代函数系统(PIFS)，并从该框架推导出三个无需模型评估的可计算几何量，从第一性原理统一解释了扩散模型的双阶段去噪动力学、自注意力的有效性，以及四种经验设计选择（cosine schedule offset、分辨率相关logSNR偏移、Min-SNR损失加权、Align Your Steps采样）。

**[Garments2Look A Multi-Reference Dataset For High-Fidelity Outfit-Level Virtual T](garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)**

:   提出 Garments2Look，首个大规模多模态整套搭配级虚拟试穿数据集（80K 对，40 类，300+ 子类），每组包含 3-12 件参考服饰图、模特穿搭图和详细文本标注，揭示现有方法在多层搭配和配饰一致性上的重大不足。

**[Gqir Generative Quanta Image Reconstruc Tion](gqir_generative_quanta_image_reconstruc_tion.md)**

:   将大规模 text-to-image latent diffusion model 适配到单光子雪崩二极管（SPAD）的极端光子受限成像场景，通过三阶段框架（Quanta-aligned VAE → 对抗微调 LoRA U-Net → FusionViT 时空融合）实现从稀疏二值光子检测到高质量 RGB 图像的重建，在 10K-100K fps 极端条件下显著超越所有现有方法。

**[Gqir Generative Quanta Image Reconstruction](gqir_generative_quanta_image_reconstruction.md)**

:   提出 gQIR，一个模块化三阶段框架，将大规模 T2I 扩散模型适配到 SPAD 传感器的极端光子受限域，通过量子对齐 VAE（冻结编码器副本防坍缩）、对抗微调 LoRA U-Net（单步生成）和潜空间 FusionViT（时空融合），从极稀疏二值光子事件重建高质量彩色图像和视频。

**[Guiding Diffusion Models With Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)**

:   提出 Condition-Degradation Guidance (CDG)，用语义退化的条件 $\boldsymbol{c}_{\text{deg}}$ 替代 CFG 中的空提示 $\emptyset$，将引导从粗粒度"好 vs. 空"转变为细粒度"好 vs. 差一点"的对比，通过分层退化策略（先退化内容 token 再退化上下文聚合 token）构建自适应负样本，在 SD3/FLUX/Qwen-Image 等模型上即插即用地提升组合生成精度，几乎零额外开销。

**[Haltnav Reactive Visual Halting Over Lightweight T](haltnav_reactive_visual_halting_over_lightweight_t.md)**

:   提出层级导航框架 HaltNav，结合轻量文本拓扑图 (osmAG) 全局规划 + VLN 模型局部执行，并引入反应式视觉停止 (RVH) 机制在遇到未知障碍时实时中断、更新拓扑、重规划绕行，在仿真和真实机器人上均显著优于基线。

**[Haltnav Reactive Visual Halting Over Lightweight Topological Priors For Robust V](haltnav_reactive_visual_halting_over_lightweight_topological_priors_for_robust_v.md)**

:   提出 HaltNav，一个层级化导航框架，结合轻量级文本拓扑先验（osmAG）做全局规划，用 VLN 模型做局部执行，并通过 Reactive Visual Halting 机制检测意外障碍、动态更新拓扑并重规划，在仿真和真机上均显著提升长程导航鲁棒性。

**[Heterogeneous Decentralized Diffusion Models](heterogeneous_decentralized_diffusion_models.md)**

:   提出异构去中心化扩散框架，允许不同专家使用不同扩散目标（DDPM ε-prediction 与 Flow Matching velocity-prediction）完全独立训练，在推理时通过确定性 schedule-aware 转换统一到速度空间进行融合，相比同构基线同时提升 FID 和生成多样性，并将计算量压缩 16 倍。

**[Hifi-Inpaint Towards High-Fidelity Reference-Based Inpainting For Generating Det](hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)**

:   提出 HiFi-Inpaint 框架，通过共享增强注意力（SEA）利用高频信息增强产品细节特征，结合细节感知损失（DAL）实现像素级高频监督，在人-产品图像生成中达到 SOTA 的细节保真度。

**[High-Fidelity Diffusion Face Swapping With Id-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)**

:   提出身份约束的属性调优框架用于扩散模型人脸替换：先约束身份解空间，再注入属性条件，最后端到端精炼身份损失和对抗损失，结合解耦条件注入设计，在 FFHQ 上实现 SOTA 的 FID（3.61）和身份检索准确率（97.9% Top-1）。

**[Image Generation As A Visual Planner For Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)**

:   将预训练图像生成模型（DiT）通过 LoRA 微调适配为机器人操作的视觉规划器，以 3×3 网格图像形式生成时序连贯的操作序列，支持文本条件和轨迹条件两种控制模式。

**[Improving Text-To-Image Generation With Intrinsic Self-Confidence Rewards](improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)**

:   提出 SOLACE，一种利用文本-图像生成模型自身去噪自信度作为内在奖励的后训练框架，无需外部奖励模型即可在组合生成、文字渲染和文图对齐上获得一致提升，且可与外部奖励互补缓解 reward hacking。

**[Infinity-Rope Action-Controllable Infinite Video Generation Emerges From Autoreg](infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)**

:   提出 ∞-RoPE，一个训练免调的推理时框架，通过 Block-Relativistic RoPE、KV Flush 和 RoPE Cut 三个组件，将仅在5秒视频上训练的自回归视频扩散模型扩展为支持无限时长生成、精细动作控制和电影级场景切换的系统。

**[Innoads-Composer Efficient Condition Composition For E-Commerce Poster Generatio](innoads-composer_efficient_condition_composition_for_e-commerce_poster_generatio.md)**

:   提出 InnoAds-Composer，一个基于 MM-DiT 的单阶段电商海报生成框架，通过统一 token 化将商品主体、字形文本和背景风格三类条件映射到同一空间，结合文本特征增强模块（TFEM）和重要性感知条件注入策略，在保持高质量生成的同时显著降低推理开销。

**[Interedit Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)**

:   提出 InterEdit，首个文本引导的多人3D运动编辑框架，通过语义感知 Plan Token 对齐和交互感知频域 Token 对齐两个机制，在条件扩散模型中实现对双人交互动作的精准编辑，同时保持源运动的一致性和交互协调性。

**[Interedit Navigating Textguided Multihuman 3D Moti](interedit_navigating_textguided_multihuman_3d_moti.md)**

:   首次定义多人3D运动编辑(TMME)任务，构建5161个源-目标-指令三元组的InterEdit3D数据集，提出基于同步无分类器引导的条件扩散模型InterEdit，通过语义感知规划Token对齐和交互感知频域Token对齐两个核心模块，在指令跟随(g2t R@1 30.82%)和源保持(g2s R@1 17.08%)上全面超越基线。

**[Intrinsic Concept Extraction Based On Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)**

:   HyperExpress 提出组合可解释本征概念提取（CI-ICE）新任务，利用双曲空间的层次建模能力和等球面投影模块，从单张图像中提取可组合的物体级和属性级概念，实现可逆的复杂视觉概念分解。

**[Learning Latent Proxies For Controllable Single-Image Relighting](learning_latent_proxies_for_controllable_single-image_relighting.md)**

:   提出 LightCtrl，一个基于扩散模型的单图重光照框架，通过小样本潜在代理编码器（few-shot latent proxy）提供轻量材质-几何先验、光照感知掩码引导空间选择性去噪、DPO 后训练增强物理一致性，实现对光照方向/强度/色温的精确连续控制，在合成和真实场景上均优于现有方法。

**[Learning Latent Transmission And Glare Maps For Lens Veiling Glare Removal](learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)**

:   提出 VeilGen + DeVeiler 框架，通过物理引导的 Stable Diffusion 生成模型学习潜在透射率和眩光图以合成逼真的复合退化训练数据，并用可逆约束训练修复网络，实现简化光学系统中像差与雾化眩光的联合去除。

**[Learning To Generate Via Understanding Understanding-Driven Intrinsic Rewarding ](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)**

:   提出 GvU，利用统一多模态模型（UMM）自身的视觉理解分支作为内在奖励信号，通过 token 级文图对齐概率构建自监督 RL 框架（基于 GRPO），在无外部监督下迭代提升 T2I 生成质量，GenEval++ 上实现 43.3% 提升，且生成增强反过来促进细粒度理解。

**[Lesa Learnable Stage-Aware Predictors For Diffusion Model Acceleration](lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)**

:   提出 LESA 框架，用 KAN（Kolmogorov-Arnold Network）作为可学习时序预测器，结合多阶段多专家架构和两阶段训练策略，在 FLUX 上实现 5× 加速仅 1.0% 质量下降，在 Qwen-Image 上 6.25× 加速比 TaylorSeer 质量提升 20.2%，在 HunyuanVideo 上 5× 加速 PSNR 提升 24.7%。

**[Linvideo A Post-Training Framework Towards On Attention In Efficient Video Gener](linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)**

:   提出 LinVideo，一种无需训练数据的后训练框架，通过选择性地将视频扩散模型中的二次注意力替换为线性注意力，实现 1.43–1.71× 加速，结合蒸馏可达 15.9–20.9× 加速，同时保持生成质量。

**[Linvideo Linear Attention Video Generation](linvideo_linear_attention_video_generation.md)**

:   首个data-free后训练框架LinVideo，通过选择性转移自动选择最适合替换为线性注意力的层+任意时刻分布匹配(ADM)目标函数高效恢复性能，实现Wan 1.3B/14B的1.43-1.71×加速且质量无损，叠加4步蒸馏后达15.9-20.9×加速。

**[Magic Few-Shot Mask-Guided Anomaly Inpainting With Prompt Perturbation Spatially](magic_few-shot_mask-guided_anomaly_inpainting_with_prompt_perturbation_spatially.md)**

:   提出 MAGIC 框架，通过微调 inpainting 扩散模型，结合高斯 prompt 扰动、掩码引导空间噪声注入和上下文感知掩码对齐三个互补模块，在少样本条件下生成高保真、多样化、空间合理的工业异常图像，在 MVTec-AD 下游任务上达到 SOTA。

**[Match-And-Fuse Consistent Generation From Unstructured Image Sets](match-and-fuse_consistent_generation_from_unstructured_image_sets.md)**

:   提出 Match-and-Fuse，首个面向非结构化图像集合的训练无关一致性生成方法。以图为节点、图对为边建立成对一致性图，通过多视角特征融合（MFF）和特征引导在扩散推理中操控内部特征，实现集合级跨图一致性，DINO-MatchSim 达 0.80 远超所有基线。

**[Micon-Bench Benchmarking And Enhancing Multi-Image Context Image Generation In U](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)**

:   提出 MICON-Bench，覆盖 6 项任务（1043 案例）的多图上下文生成基准，配合 MLLM 驱动的 Evaluation-by-Checkpoint 自动评估框架；同时提出 DAR（Dynamic Attention Rebalancing）训练无关机制，通过动态调整推理时注意力权重提升 UMM 的多图生成一致性和质量。

**[Mixture Of States Routing Token-Level Dynamics For Multimodal Generation](mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)**

:   提出 Mixture of States (MoS)——一种基于可学习 token 级稀疏路由的多模态融合范式，使视觉 token 能在每个去噪步骤自适应地从文本编码器任意层选取隐藏状态，仅用 3-5B 参数即可匹敌或超越 20B 级模型。

**[Morphany3D Unleashing The Power Of Structured Latent In 3D Morphing](morphany3d_unleashing_the_power_of_structured_latent_in_3d_morphing.md)**

:   提出 MorphAny3D，首个基于 Structured Latent（SLAT）表示的无训练 3D 变形框架，通过 Morphing Cross-Attention（MCA）融合源/目标信息保证结构合理、Temporal-Fused Self-Attention（TFSA）增强时序一致性、方向校正策略消除突变，在跨类别 3D 变形中实现了 SOTA 质量。

**[Mos Mitigating Optical-Sar Modality Gap For Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)**

:   提出 MOS 框架解决光学-SAR 跨模态船舶重识别问题，包含两个核心模块：(1) MCRL 通过 SAR 图像去噪和类别级模态对齐损失在训练阶段缩小模态差距；(2) CDGF 利用布朗桥扩散模型在推理阶段从光学图像生成伪 SAR 样本并融合特征，在 HOSS ReID 数据集上 SAR→Optical 的 R1 提升 +16.4%。

**[Nova Sparse Control Dense Synthesis For Pair-Free Video Editing](nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)**

:   提出 NOVA，首次形式化"稀疏控制、密集合成"范式用于视频编辑：稀疏分支从用户编辑的多关键帧提供语义引导，密集分支从原始视频注入运动和纹理信息；配合退化模拟训练策略实现无需配对数据的学习，在编辑保真度、运动保持和时序一致性上全面超越现有方法。

**[Oars Process-Aware Online Alignment For Generative Real-World Image Super-Resolu](oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)**

:   提出 OARS 框架，通过基于 MLLM 的过程感知奖励模型 COMPASS 和渐进式在线强化学习（冷启动→有参考 RL→无参考 RL），首次系统解决生成式真实世界图像超分辨率中的人类偏好对齐问题，在保持保真度的同时显著提升感知质量。

**[Oars Processaware Online Alignment For Generative](oars_processaware_online_alignment_for_generative.md)**

:   提出了OARS框架，通过基于MLLM的过程感知奖励模型COMPASS和渐进式在线强化学习，将生成式真实世界超分辨率模型与人类视觉偏好对齐，在感知质量和保真度之间实现自适应平衡。

**[Object-Wiper Training-Free Object And Associated Effect Removal In Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)**

:   提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除，并提出 TokSim 指标和 WIPER-Bench 真实世界基准。

**[One Model Many Budgets Elastic Latent Interfaces F](one_model_many_budgets_elastic_latent_interfaces_f.md)**

:   提出ELIT（Elastic Latent Interface Transformer），通过在DiT中插入可变长度的潜在token接口和轻量级Read/Write交叉注意力层，将计算量与输入分辨率解耦，使单一模型支持多种推理预算，在ImageNet-1K 512px上FID和FDD分别提升35.3%和39.6%。

**[One Model Many Budgets Elastic Latent Interfaces For Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   提出 ELIT（Elastic Latent Interface Transformer），在 DiT 中插入可变长度的潜变量接口（latent interface）和轻量 Read/Write 跨注意力层，使单一模型能在推理时动态调节计算预算，同时将计算非均匀地分配到图像中更难的区域，在 ImageNet 512px 上 FID 最高降低 53%。

**[Parallelised Differentiable Straightest Geodesics For 3D Meshes](parallelised_differentiable_straightest_geodesics_for_3d_meshes.md)**

:   提出 straightest geodesics 的并行 GPU 实现及两种可微分方案（外在代理函数法和测地线有限差分法），使三角网格上的指数映射可高效并行且可微分，并以此构建测地线卷积层、网格上的流匹配方法和二阶优化器三个下游应用。

**[Physical Simulator In-The-Loop Video Generation](physical_simulator_in-the-loop_video_generation.md)**

:   提出 PSIVG，首个将物理模拟器嵌入视频扩散模型推理循环的无训练框架：从模板视频重建 4D 场景和 3D 网格并初始化物理模拟器，生成物理一致轨迹引导视频生成，并通过 Test-Time Texture Consistency Optimization（TTCO）优化前景纹理一致性。用户研究中 82.3% 偏好率远超所有基线。

**[Physics-Consistent Diffusion For Efficient Fluid Super-Resolution Via Multiscale](physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)**

:   提出 ReMD（Residual-Multigrid Diffusion），在扩散模型的每一步反向采样中嵌入多重网格残差修正，利用多小波基构建跨尺度层次结构，无需显式 PDE 即可实现物理一致的高效流体超分辨率。

**[Pixel Motion Diffusion Is What We Need For Robot Control](pixel_motion_diffusion_is_what_we_need_for_robot_control.md)**

:   DAWN 提出两阶段全扩散框架，通过 Motion Director 生成稠密像素运动场作为可解释中间表征，再由 Action Expert 转化为机器人动作序列，在 CALVIN 基准上实现 SOTA（平均长度 4.00）且数据效率极高。

**[Pixelrush Ultra-Fast Training-Free High-Resolution Image Generation Via One-Step](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)**

:   提出 PixelRush，一种无需训练的高分辨率图像生成框架，通过部分反演（partial inversion）+ 少步扩散模型 + 高斯滤波拼接 + 噪声注入四大组件，将 4K 图像生成速度从数分钟压缩到约 20 秒（10×–35× 加速），同时在 FID/IS 指标上超越现有 SOTA。

**[Pixelrush Ultrafast Trainingfree Highresolution Im](pixelrush_ultrafast_trainingfree_highresolution_im.md)**

:   PixelRush 首次实现了免训练的单步高分辨率图像生成，通过部分 DDIM 反转（只扰动到中间时间步而非全噪声）+ 少步扩散模型 + 高斯滤波 patch 融合 + 噪声注入，在单卡 A100 上 20 秒生成 4K 图像，比 SOTA 快 10-35× 且 FID 更优（50.13 vs 52.87）。

**[Pluggable Pruning With Contiguous Layer Distillation For Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)**

:   提出 PPCL 框架，通过线性探针检测 MMDiT 中连续冗余层区间，结合非顺序蒸馏实现深度剪枝（即插即用）和宽度剪枝（用线性投影替换文本流/FFN），将 Qwen-Image 从 20B 压缩到 10B 时性能仅下降 3.29%。

**[Pose-Dive Pose-Diversified Augmentation With Diffusion Model For Person Re-Ident](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)**

:   Pose-dIVE通过SMPL模型联合控制人体姿态和相机视角，利用扩散模型生成具有多样化姿态和视角的行人图像，系统性地弥补Re-ID训练数据中的分布偏差，在多个基准上持续提升任意Re-ID模型的泛化能力。

**[Precise Object And Effect Removal With Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)**

:   提出 ObjectClear 框架，通过自适应目标感知注意力（ATA）将前景移除与背景重建解耦，配合注意力引导融合（AGF）和空间变化去噪强度（SVDS）策略，实现对目标物体及其阴影、反射等附带效果的精准移除，同时构建了首个大规模 Object-Effect Removal 数据集 OBER。

**[Probing And Bridging Geometry-Interaction Cues For Affordance Reasoning In Visio](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)**

:   系统性地探测视觉基础模型（VFM）中的可供性（affordance）能力，发现 DINO 编码了部件级几何结构、Flux 编码了动词条件化的交互先验，并通过 training-free 融合两者实现了可与弱监督方法竞争的零样本可供性估计。

**[Promo Promptable Outfitting For Efficient High-Fidelity Virtual Try-On](promo_promptable_outfitting_for_efficient_high-fidelity_virtual_try-on.md)**

:   基于 Flow Matching DiT 的虚拟试穿框架，通过 latent 多模态条件拼接、时序自参考缓存机制和 3D-RoPE 分组条件注入，在保持高保真度的同时大幅降低推理开销，支持多件服装试穿和文本提示控制穿搭风格。

**[Promo Promptable Virtual Tryon Efficient](promo_promptable_virtual_tryon_efficient.md)**

:   PROMO基于FLUX Flow Matching DiT骨干，通过潜空间多模态条件拼接、时序自参考KV缓存、3D-RoPE分组条件、以及fine-tuned VLM风格提示系统，在去除传统参考网络的前提下实现了高保真且高效的多件服装虚拟试穿，推理速度比无加速版快2.4倍，在VITON-HD和DressCode上超越现有VTON和通用图像编辑方法。

**[Prototype-Guided Concept Erasure In Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)**

:   针对扩散模型中宽泛概念（如暴力、色情）难以彻底擦除的问题，提出基于概念原型的 training-free 擦除方法：通过聚类 CLIP 嵌入空间中的概念差分方向获取图像原型，再优化迁移到文本原型空间，推理时选择最匹配的原型作为负引导信号进行 classifier-free guidance 式的概念抑制。

**[Purecc Pure Learning For Text-To-Image Concept Customization](purecc_pure_learning_for_text-to-image_concept_customization.md)**

:   提出 PureCC 方法，通过分离"目标概念隐式引导"和"原始条件预测"的解耦学习目标，配合冻结表示提取器+可训练流模型的双分支训练管线和自适应引导缩放 $\lambda^{\star}$，实现高保真概念定制的同时最小化对原始模型行为和能力的影响。

**[Raise Requirement-Adaptive Evolutionary Refinement For Training-Free Text-To-Ima](raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)**

:   提出 RAISE 框架，将 T2I 生成建模为需求驱动的自适应进化过程：通过需求分析器将提示词分解为结构化检查清单，用多动作变异（提示重写+噪声重采样+指令编辑）并发进化候选群体，再通过工具增强的视觉验证逐轮淘汰不满足需求的候选，实现自适应推理时缩放——在 GenEval 上达到 0.94 SOTA，同时比反射微调基线减少 30-40% 生成样本和 80% VLM 调用。

**[Razor Ratio-Aware Layer Editing For Targeted Unlearning In Vision Transformers A](razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)**

:   提出 RAZOR，一种基于比率感知的多层/多头选择性编辑框架，可在 CLIP、Stable Diffusion 和 VLM 等 Transformer 视觉模型中高效精准地完成目标遗忘，同时保持模型整体性能与量化鲁棒性。

**[Razor Ratio Aware Unlearning Vit Diffusion](razor_ratio_aware_unlearning_vit_diffusion.md)**

:   提出 RAZOR, 一种基于比率感知梯度评分的多层协调编辑方法, 用于 ViT 和扩散模型的目标遗忘: 通过 forget/retain 梯度的比率和余弦对齐度联合评分, 识别对遗忘贡献最大且对保留损害最小的层/头, 实现一次性高效遗忘, 在 CLIP 身份遗忘上达到 SOTA.

**[Refining Few-Step Text-To-Multiview Diffusion Via Reinforcement Learning](refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)**

:   提出 MVC-ZigAL 框架，通过多视图感知 MDP 建模、zigzag 自反思优势学习和 Lagrangian 对偶约束优化，有效提升少步文本到多视图扩散模型的单视图保真度和跨视图一致性。

**[Rel-Zero Harnessing Patch-Pair Invariance For Robust Zero-Watermarking Against A](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)**

:   本文发现图像patch对之间的关系距离在AI编辑后保持不变，并利用该不变性构建了一种零水印框架Rel-Zero，无需修改原图即可实现对多种生成式编辑的鲁棒内容认证。

**[Reviving Convnext For Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)**

:   本文提出FCDM（Fully Convolutional Diffusion Model），将ConvNeXt架构适配为条件扩散模型backbone，仅用DiT-XL 50%的FLOPs即可在ImageNet上达到竞争性FID（2.03），且能在4块RTX 4090上训练XL模型，展示了全卷积架构在生成建模中被严重低估的效率优势。

**[Seacache Spectral-Evolution-Aware Cache For Accelerating Diffusion Models](seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)**

:   提出 SeaCache，一种基于频谱演化感知（SEA）滤波器的无训练动态缓存策略，通过在频域中分离信号与噪声分量来测量时间步间的冗余度，显著提升扩散模型推理的延迟-质量权衡。

**[Segquant A Semantics-Aware And Generalizable Quantization Framework For Diffusio](segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)**

:   提出 SegQuant 框架，通过基于静态计算图的语义分割量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在不依赖手工规则或运行时动态信息的前提下，实现了跨架构通用、部署管线兼容的扩散模型高保真后训练量化。

**[Segquant Diffusion Model Quantization](segquant_diffusion_model_quantization.md)**

:   提出 SegQuant，一个面向部署的扩散模型后训练量化框架，通过基于计算图静态分析的语义感知分段量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在 SD3.5、FLUX、SDXL 上实现跨架构通用的高保真 W8A8/W4A8 量化，同时保持与 TensorRT 等工业推理引擎的兼容性。

**[Simlbr Learning To Detect Fake Images By Learning To Detect Real Images](simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)**

:   本文提出SimLBR，通过在DINOv3潜空间中将少量假图信息混入真图嵌入作为正则化手段，迫使检测器学习真实图像分布的紧致决策边界，从而实现对未知生成器的强泛化能力，在GenImage上平均准确率达94.54%，在硬测试集Chameleon上比AIDE提升25%准确率和70%召回率。

**[Sjd-Pac Accelerating Speculative Jacobi Decoding Via Proactive Drafting And Adap](sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)**

:   本文分析了 Speculative Jacobi Decoding (SJD) 在文本到图像生成中接受长度分布严重偏斜的瓶颈，提出 SJD-PAC 框架，通过 Proactive Drafting (PD) 和 Adaptive Continuation (AC) 两项技术，在严格无损的前提下实现 3.8× 推理加速，显著超越原始 SJD 的约 2× 加速。

**[Solace Self Confidence Rewards T2I](solace_self_confidence_rewards_t2i.md)**

:   用T2I模型自身的去噪自信心（对注入噪声的恢复精度）作为内在奖励替代外部奖励模型做后训练，在组合生成、文字渲染、文图对齐上获一致提升，且与外部奖励互补可缓解reward hacking。

**[Spatial-Ssrl Enhancing Spatial Understanding Via Self-Supervised Reinforcement L](spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)**

:   本文提出Spatial-SSRL，一种自监督强化学习范式，通过从普通RGB/RGB-D图像自动构造五种pretext任务（patch重排、翻转识别、裁剪修补、深度排序、相对3D位置预测），利用GRPO优化LVLM的空间理解能力，在七个空间benchmark上平均提升3.89%-4.63%，且无需人工标注或外部工具。

**[Switchcraft Training-Free Multi-Event Video Generation With Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)**

:   提出 SwitchCraft，一个无需训练的多事件视频生成框架，通过 Event-Aligned Query Steering (EAQS) 将帧级注意力对齐到对应事件提示、Auto-Balance Strength Solver (ABSS) 自适应平衡引导强度，在不修改模型权重的情况下实现多事件视频的清晰时序切换和场景一致性。

**[Taming Preference Mode Collapse Via Directional Decoupling Alignment In Diffusio](taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)**

:   提出 D2-Align 框架，通过在奖励模型嵌入空间中学习方向性修正向量来纠偏奖励信号，解决扩散模型 RLHF 对齐中的偏好模式坍塌（PMC）问题——即模型过度优化奖励导致生成多样性严重下降；同时提出 DivGenBench 基准用于量化评估生成多样性。

**[Taming Score-Based Denoisers In Admm A Convergent Plug-And-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 三阶段去噪器（自动校正 + 方向校正 + Score 去噪），解决 ADMM 迭代与 score 训练流形不匹配的问题，并首次为 ADMM-PnP + score denoiser 建立了收敛性保证，在多种逆问题上取得 SOTA。

**[Taming Scorebased Denoisers In Admm A Convergent P](taming_scorebased_denoisers_in_admm_a_convergent_p.md)**

:   提出ADMM-PnP with AC-DC去噪器，通过三阶段修正-去噪流程(自动修正+方向修正+基于分数的去噪)将扩散先验集成到ADMM原始-对偶框架中，解决了ADMM迭代与扩散训练流形的几何不匹配问题，同时在两种条件下建立了收敛保证，在7种逆问题上一致优于DAPS/DPS/DiffPIR等基线。

**[Tap A Token-Adaptive Predictor Framework For Training-Free Diffusion Acceleratio](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)**

:   提出 TAP 框架，通过第一层探针（probe）为每个 token 在每一步自适应选择最优预测器（Taylor 展开族），实现无需训练的扩散模型加速，在 FLUX.1-dev 上以 6.24× 加速且无感知质量损失。

**[Taue Training-Free Noise Transplant And Cultivation Diffusion Model](taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)**

:   TAUE 提出一种**免训练**的分层图像生成框架，通过将去噪中间潜变量"移植"到新生成过程的初始噪声中，并结合跨层注意力共享，实现前景、背景和合成图像的三层一致生成，性能匹配甚至超越微调方法。

**[Tc-Padé Trajectory-Consistent Padé Approximation For Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)**

:   提出基于 Padé 有理函数近似的特征残差预测框架 TC-Padé，通过自适应系数调节和分阶段感知策略，在低步数（20-30步）扩散采样场景下实现轨迹一致的加速（FLUX.1-dev 2.88×、Wan2.1 1.72×），显著优于基于 Taylor 展开的现有方法。

**[Textpecker Rewarding Structural Anomaly Quantification For Enhancing Visual Text](textpecker_rewarding_structural_anomaly_quantification_for_enhancing_visual_text.md)**

:   提出 TextPecker——一种即插即用的结构异常感知 RL 策略，通过构建字符级结构异常标注数据集训练结构感知识别器，替代传统 OCR 的噪声奖励信号，联合优化语义对齐和结构保真度，在多个文本到图像模型（FLUX、SD3.5、Qwen-Image）上显著提升视觉文本渲染质量。

**[Tina Text-Free Inversion Attack For Unlearned Text-To-Image Diffusion Models](tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)**

:   提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。

**[Too Vivid To Be Real Benchmarking And Calibrating Generative Color Fidelity](too_vivid_to_be_real_benchmarking_and_calibrating_generative_color_fidelity.md)**

:   针对 T2I 模型生成图像"太鲜艳不像真实照片"的问题，提出 Color Fidelity Dataset (CFD, 130 万图像)、Color Fidelity Metric (CFM, 基于 Qwen2-VL + softrank loss) 和 Color Fidelity Refinement (CFR, 无训练的时空自适应 guidance 调制)，形成评估-改善一体化框架。

**[Trace Structure-Aware Character Encoding For Robust And Generalizable Document W](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)**

:   提出 TRACE——基于字符结构编码的文档水印框架，利用扩散模型（DragDiffusion）精确移动字符骨架关键点来嵌入信息，通过自适应扩散初始化（ADI）、引导扩散编码（GDE）和掩码区域替换（MRR）三大组件，同时实现跨介质传输鲁棒性、多语言/多字体泛化性和高隐蔽性。

**[Training-Free Motion Factorization For Compositional Video Generation](training-free_motion_factorization_for_compositional_video_generation.md)**

:   提出一种无需训练的运动分解框架，将复杂场景运动拆分为静止、刚性运动和非刚性运动三类，通过结构化运动推理 (SMR) 消除提示语义歧义，再借助解耦运动引导 (DMG) 模块分别约束各类运动的生成，实现多实例多运动类别的组合视频合成。

**[Uni-Dad Unified Distillation And Adaptation Of Diffusion Models For Few-Step Few](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)**

:   提出 Uni-DAD，首个将扩散模型蒸馏（distillation）与域适应（adaptation）统一为单阶段流程的方法，通过双域 DMD 损失和多头 GAN 损失，在仅 1–4 步采样下实现少样本域的高质量多样生成。

**[V-Bridge Bridging Video Generative Priors To Versatile Few-Shot Image Restoratio](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像修复重新定义为渐进式视频生成过程，利用预训练视频模型（Wan2.2-TI2V-5B）的丰富视觉先验，仅用 1,000 个多任务训练样本（不到现有方法的 2%）就实现了多种退化类型的全能修复，超越了在百万级数据上训练的专用架构。

**[Vecor -- Velocity Contrastive Regularization For Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)**

:   提出 VeCoR（速度对比正则化），在标准 Flow Matching 训练中引入"负速度"对比信号，通过同时指导模型"该往哪走"和"不该往哪走"，实现更稳定的轨迹演化和更高的感知保真度——在 ImageNet-1K 上 SiT-XL/2 和 REPA-SiT-XL/2 分别获得 22% 和 35% 的 FID 相对降低。

**[Vinedresser3D Agentic Text-Guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)**

:   提出 Vinedresser3D，一个以多模态大语言模型（MLLM）为核心的 3D 编辑智能体，无需用户提供 3D 掩码，通过自动解析编辑意图、定位编辑区域、生成多模态引导，并在原生 3D 生成模型（Trellis）的潜空间中执行基于反演的修补编辑，实现高质量文本引导的 3D 资产编辑。

**[Wadi Weight Direction-Aware Distillation For One-Step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)**

:   通过分析蒸馏过程中权重变化的范数-方向分解，发现方向变化是蒸馏的关键驱动因素（变化幅度比范数大 22×），提出 LoRaD（低秩权重方向旋转）适配器，集成到 VSD 框架中构成 WaDi，仅用 ~10% 可训练参数即在 COCO 上取得一步生成 SOTA FID。

**[When Safety Collides Resolving Multi-Category Harmful Conflicts In Text-To-Image](when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)**

:   提出 Conflict-aware Adaptive Safety Guidance (CASG)，一种无训练的即插即用框架，通过动态识别与当前生成状态最对齐的有害类别并仅沿该方向施加安全引导，解决了现有安全引导方法在多类别聚合时因方向冲突导致的安全性退化问题。

**[When To Lock Attention Training-Free Kv Control In Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，基于扩散模型幻觉检测动态调度背景 KV 缓存融合比例和 CFG 引导强度，在无需训练的前提下同时保证视频编辑的背景一致性和前景生成质量。

**[Wiser Wider Search Deeper Thinking And Adaptive Fusion For Training-Free Zero-Sh](wiser_wider_search_deeper_thinking_and_adaptive_fusion_for_training-free_zero-sh.md)**

:   提出 WISER，一个无训练的零样本组合图像检索（ZS-CIR）框架，通过"检索–验证–精化"迭代循环统一 T2I 和 I2I 双路径检索，利用 VLM 验证器显式建模意图感知和不确定性感知，实现自适应融合与结构化自反思精化。在 CIRCO mAP@5 上相对提升 45%，CIRR Recall@1 上相对提升 57%，甚至超越许多训练式方法。
