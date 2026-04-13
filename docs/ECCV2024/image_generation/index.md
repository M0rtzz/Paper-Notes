---
title: >-
  ECCV2024 图像生成方向 121篇论文解读
description: >-
  121篇ECCV2024 图像生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🎞️ ECCV2024** · 共 **121** 篇

**[2S-Odis Two-Stage Omni-Directional Image Synthesis By Geometric Distortion Corre](2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)**

:   2S-ODIS通过两阶段结构利用预训练VQGAN（无需微调）合成全景图像：第一阶段生成低分辨率粗略ERP图，第二阶段通过生成26个NFoV局部图像并融合来校正几何畸变，训练时间从14天缩短到4天且图像质量更优。

**[A Diffusion Model For Simulation Ready Coronary Anatomy With](a_diffusion_model_for_simulation_ready_coronary_anatomy_with.md)**

:   用潜在扩散模型（LDM）可控生成3D多组织冠状动脉分割图，通过拓扑交互损失保证解剖合理性，通过形态-骨架双通道条件化实现对截面形态和分支结构的解耦控制，并提出自适应空条件引导（ANG）以非可微回归器高效增强条件保真度，最终支持面向有限元仿真的反事实解剖结构编辑。

**[Accdiffusion An Accurate Method For Higher-Resolution Image Generation](accdiffusion_an_accurate_method_for_higher-resolution_image_generation.md)**

:   提出AccDiffusion，通过将全局文本prompt解耦为patch级别的内容感知prompt（利用cross-attention map判断每个词汇是否属于某patch），并引入带窗口交互的膨胀采样来改善全局一致性，在无需额外训练的情况下有效解决patch-wise高分辨率图像生成中的目标重复问题，在SDXL上实现了从2K到4K分辨率的无重复高质量图像外推。

**[Adadiffsr Adaptive Region-Aware Dynamic Acceleration Diffusion Model For Real-Wo](adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)**

:   观察到扩散模型超分中不同图像区域所需去噪步数差异巨大（背景区域早已收敛而前景纹理仍需迭代），提出基于多指标潜在熵（MMLE）感知信息增益来动态跳步的策略，将子区域分为稳定/增长/饱和三类给予不同步长，并通过渐进特征注入（PFJ）平衡保真度与真实感，在DRealSR等数据集上取得与StableSR可比的质量但推理时间和FLOPs分别减少1.5×和2.7×。

**[Adagen Learning Adaptive Policy For Image Synthesis](adagen_learning_adaptive_policy_for_image_synthesis.md)**

:   将多步生成模型（MaskGIT/AR/Diffusion/Rectified Flow）的步级参数调度（温度、mask ratio、CFG scale、timestep等）统一建模为MDP，用轻量RL策略网络实现样本自适应调度，并提出对抗奖励设计防止策略过拟合，在四种生成范式上一致提升性能（VAR FID 1.92→1.59，DiT-XL推理成本降3倍同时性能更优）。

**[Adanat Exploring Adaptive Policy For Token-Based Image Generation](adanat_exploring_adaptive_policy_for_token-based_image_generation.md)**

:   提出AdaNAT，将非自回归Transformer（NAT）的生成策略配置建模为MDP，通过轻量策略网络+PPO强化学习+对抗奖励模型自动为每个样本定制生成策略（重掩码比例、采样温度、CFG权重等），在ImageNet-256上仅用8步达到FID 2.86，相比手工策略实现约40%的相对提升。

**[Anycontrol Create Your Artwork With Versatile Control On Tex](anycontrol_create_your_artwork_with_versatile_control_on_tex.md)**

:   AnyControl提出Multi-Control Encoder，通过交替执行多控制融合块和多控制对齐块，从任意组合的多种空间控制信号中提取统一的多模态embedding，实现高质量、语义对齐的多条件可控图像生成。

**[Anycontrol Create Your Artwork With Versatile Control On Text-To-Image Generatio](anycontrol_create_your_artwork_with_versatile_control_on_text-to-image_generatio.md)**

:   提出 AnyControl，通过 Multi-Control Encoder（fusion + alignment 交替块结构）支持任意组合的多种空间控制信号（深度、边缘、分割、姿态），在 COCO 多控制基准上 FID 44.28 全面超越现有方法。

**[Bridging The Gap Studio-Like Avatar Creation From A Monocular Phone Capture](bridging_the_gap_studio-like_avatar_creation_from_a_monocular_phone_capture.md)**

:   提出从单目手机视频生成类似影棚级质量的面部纹理贴图的方法，结合 StyleGAN2 的 W+ 空间参数化与扩散模型超分辨率，实现从手机扫描到高质量 3D 头像的跨越。

**[Byteedit Boost Comply And Accelerate Generative Image Editing](byteedit_boost_comply_and_accelerate_generative_image_editing.md)**

:   提出 ByteEdit，一个将人类反馈学习引入生成式图像编辑（inpainting/outpainting）的框架，通过美学、对齐、一致性三个奖励模型提升编辑质量，并利用对抗训练和渐进策略加速推理。

**[Challenging Forgets Unveiling The Worst-Case Forget Sets In Machine Unlearning](challenging_forgets_unveiling_the_worst-case_forget_sets_in_machine_unlearning.md)**

:   提出从对抗视角识别"最坏情况遗忘集"的方法，通过双层优化框架找到最难被遗忘的数据子集，利用 SignSGD 将二阶 BLO 简化为一阶问题，从而更可靠地评估机器遗忘方法的真实效能。

**[Coin Control-Inpainting Diffusion Prior For Human And Camera Motion Estimation](coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)**

:   提出COIN方法，通过控制-补绘（Control-Inpainting）的改进版Score Distillation Sampling，结合人-场景关系损失，从单目动态相机视频中同时估计高质量的全局人体运动和相机运动。

**[Collaborative Control For Geometry-Conditioned Pbr Image Generation](collaborative_control_for_geometry-conditioned_pbr_image_generation.md)**

:   提出 Collaborative Control 范式，通过冻结预训练RGB扩散模型并训练一个并行PBR模型，利用双向跨网络通信层联合建模RGB与PBR图像分布，在有限数据下实现高质量的几何条件PBR材质图像生成。

**[Colorpeel Color Prompt Learning With Diffusion Models V](colorpeel_color_prompt_learning_with_diffusion_models_v.md)**

:   提出 ColorPeel，通过在目标颜色的基础几何体上联合学习颜色和形状 token 来实现颜色与形状解耦，使 T2I 扩散模型能精确生成用户指定 RGB 颜色的物体。

**[Colorpeel Color Prompt Learning With Diffusion Models Via Color And Shape Disent](colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)**

:   提出ColorPeel方法，通过在目标颜色的基本几何形状上学习颜色提示token（解耦颜色与形状），并引入交叉注意力对齐损失，使T2I扩散模型能精确生成用户指定RGB颜色的物体。

**[Controlling The World By Sleight Of Hand](controlling_the_world_by_sleight_of_hand.md)**

:   提出 CosHand，通过手部二值掩码作为动作条件，在预训练 Stable Diffusion 上微调，预测手-物交互后的未来图像，并可零样本泛化到机器人末端执行器。

**[Diff-Tracker Text-To-Image Diffusion Models Are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)**

:   提出 Diff-Tracker，首次利用预训练文本到图像扩散模型（Stable Diffusion）中蕴含的丰富视觉语义知识进行无监督目标跟踪，通过学习一个表示目标的 prompt 并在线更新来实现持续跟踪。

**[Diffit Diffusion Vision Transformers For Image Generation](diffit_diffusion_vision_transformers_for_image_generation.md)**

:   提出 DiffiT（Diffusion Vision Transformer），通过引入时间依赖多头自注意力（TMSA）机制，让自注意力在去噪过程的不同阶段动态调整行为，在ImageNet-256上以比DiT/MDT少16-20%的参数量达到了1.73的SOTA FID分数。

**[Difftracker Texttoimage Diffusion Models Are Unsupervised Tr](difftracker_texttoimage_diffusion_models_are_unsupervised_tr.md)**

:   Diff-Tracker利用预训练T2I扩散模型知识进行无监督跟踪，学习prompt在cross-attention上激活目标区域。

**[Diffusion-Based Image-To-Image Translation By Noise Correction Via Prompt Interp](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)**

:   提出PIC（Prompt Interpolation-based Correction），一种无训练的扩散模型图像翻译方法，通过渐进式prompt嵌入插值构造噪声校正项，将其与源图像噪声预测线性组合，实现结构保持的高保真图像编辑，且推理速度（18.1s）优于所有对比方法。

**[Diffusion-Driven Data Replay A Novel Approach To Combat Forgetting In Federated ](diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)**

:   提出 DDDR 框架，首次将预训练扩散模型引入联邦类别持续学习（FCCL），通过 Federated Class Inversion 技术为每个类别学习一个紧凑的 class embedding，利用扩散模型高质量回放历史数据以对抗灾难性遗忘，并通过对比学习弥合生成数据与真实数据的域差距。

**[Distilling Diffusion Models Into Conditional Gans](distilling_diffusion_models_into_conditional_gans.md)**

:   提出 Diffusion2GAN 框架，将多步扩散模型蒸馏为单步条件GAN，核心创新是 E-LatentLPIPS 潜空间感知损失和基于预训练扩散模型的多尺度条件判别器，在零样本 COCO 基准上超越 DMD、SDXL-Turbo 和 SDXL-Lightning。

**[Dreammotion Space-Time Self-Similar Score Distillation For Zero-Shot Video Editi](dreammotion_space-time_self-similar_score_distillation_for_zero-shot_video_editi.md)**

:   提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。

**[Dreammover Leveraging The Prior Of Diffusion Models For Image Interpolation With](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)**

:   提出 DreamMover，基于预训练文本到图像扩散模型实现大运动图像对之间的插值，通过扩散感知光流估计、两级潜空间融合和自注意力拼接替换三个核心组件，生成语义一致的中间帧。

**[Ebdm Exemplar-Guided Image Translation With Brownian-Bridge Diffusion Models](ebdm_exemplar-guided_image_translation_with_brownian-bridge_diffusion_models.md)**

:   提出 EBDM 框架，将样例引导的图像翻译建模为随机布朗桥扩散过程，从结构控制直接翻译为真实感图像，通过 Global Encoder、Exemplar Network 和 Exemplar Attention Module 三个组件有效整合样例的全局风格和细节纹理信息。

**[Echoscene Indoor Scene Generation Via Information Echo Over Scene Graph Diffusio](echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)**

:   提出 EchoScene，一个基于双分支扩散模型的 3D 室内场景生成方法，通过信息回声（Information Echo）机制在场景图扩散过程中实现多个去噪过程间的协作信息交换，生成全局一致且可交互控制的场景。

**[Editable Image Elements For Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)**

:   提出"可编辑图像元素"表示，将输入图像分解为一组语义对齐的 patch embeddings（类似超像素），每个 patch 关联位置和尺寸信息，用户可直接编辑这些属性（移动、缩放、删除），再由基于 Stable Diffusion 的解码器合成真实感图像。

**[Emdm Efficient Motion Diffusion Model For Fast And High-Quality Motion Generatio](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)**

:   提出 EMDM，通过条件去噪扩散 GAN 捕获大步长下的复杂去噪分布，实现仅需不超过 10 步采样即可实时生成高质量人体动作，推理速度较 MDM 提升约 200 倍。

**[Emdm Efficient Motion Diffusion Model For Fast And High](emdm_efficient_motion_diffusion_model_for_fast_and_high.md)**

:   提出 EMDM，通过条件去噪扩散 GAN 捕获大步长采样时的复杂多模态去噪分布，结合几何损失约束，实现 T≤10 步的实时人体运动生成，推理速度提升 60-240 倍，同时保持高质量。

**[Enhancing Diffusion Models With Text-Encoder Reinforcement Learning](enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)**

:   提出 TexForce，通过强化学习（DDPO）结合 LoRA 微调扩散模型的文本编码器以提升图文对齐和视觉质量，并可无缝与已有 U-Net 微调方法组合获得更优效果。

**[Enhancing Perceptual Quality In Video Super-Resolution Through Temporally-Consis](enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)**

:   提出 StableVSR，首次将扩散模型应用于视频超分辨率任务，通过时序条件模块（TCM）和帧级双向采样策略，在显著提升感知质量的同时确保帧间时序一致性。

**[Finematch Aspect-Based Fine-Grained Image And Text Mismatch Detection And Correc](finematch_aspect-based_fine-grained_image_and_text_mismatch_detection_and_correc.md)**

:   提出 FineMatch 基准，定义了基于方面（Aspect）的细粒度图文不匹配检测与纠正任务，包含 49,906 个高质量人工标注的图文对，并展示了现有 VLM 在细粒度组合性理解上的不足。

**[Finematch Aspectbased Finegrained Image And Text Mismat](finematch_aspectbased_finegrained_image_and_text_mismat.md)**

:   提出 FineMatch benchmark，要求模型识别图文对中不匹配的方面短语（Entity/Relation/Attribute/Number）、确定类别并提出修正，构建了 49,906 个人工标注样本，并提出 ITM-IoU 评估指标和 AutoAlign 文生图幻觉检测校正系统。

**[Fouriscale A Frequency Perspective On Training-Free High-Resolution Image Synthe](fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)**

:   提出 FouriScale，从频域分析视角出发，通过膨胀卷积+低通滤波替换预训练扩散模型中的卷积层，实现免训练的任意尺寸高分辨率图像生成，理论上证明了膨胀卷积保持结构一致性的有效性。

**[Freecompose Generic Zero-Shot Image Composition With Diffusion Prior](freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)**

:   提出 FreeCompose，利用预训练扩散模型的生成先验实现通用零样本图像合成，统一覆盖图像和谐化（外观编辑）和语义图像合成（语义编辑），无需额外训练。

**[Freediff Progressive Frequency Truncation For Image Edi](freediff_progressive_frequency_truncation_for_image_edi.md)**

:   提出 FreeDiff，通过渐进式频率截断从频域精化扩散模型的编辑引导信号，无需微调或修改网络结构，实现覆盖多种编辑类型的通用图像编辑方法。

**[Freediff Progressive Frequency Truncation For Image Editing With Diffusion Model](freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)**

:   从频率视角重新审视扩散模型的图像编辑过程，发现去噪网络优先恢复低频分量导致编辑引导与目标区域的misalignment，提出渐进式频率截断（FreeDiff）方法在频率空间精炼引导信号，实现免调优的通用图像编辑。

**[Freeinit Bridging Initialization Gap In Video Diffusion](freeinit_bridging_initialization_gap_in_video_diffusion.md)**

:   提出 FreeInit，一种无需额外训练的推理采样策略，通过迭代精炼初始噪声的时空低频分量来弥合视频扩散模型训练与推理之间的初始化差距，显著提升生成视频的时序一致性。

**[Freeinit Bridging Initialization Gap In Video Diffusion Models](freeinit_bridging_initialization_gap_in_video_diffusion_models.md)**

:   发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。

**[Garmentaligner Text-To-Garment Generation Via Retrieval-Augmented Multi-Level Co](garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)**

:   针对文本到服装图像生成中的细粒度语义错位（组件数量、位置和相互关系），提出 GarmentAligner，通过自动组件提取管线获取空间-数量信息，并结合检索增强对比学习和多级校正损失，实现服装组件的视觉、空间和数量级别的精确对齐。

**[Generating 3D House Wireframes With Semantics](generating_3d_house_wireframes_with_semantics.md)**

:   提出基于自回归模型的 3D 房屋线框生成方法，采用统一的线段（wire）表示替代传统的顶点-边分离建模，通过语义感知的 BFS 序列排列和两阶段 coarse-to-fine Transformer 解码器生成语义丰富的线框结构，可自动分割为墙壁、屋顶、房间等语义组件。

**[Generating Human Interaction Motions In Scenes With Text Control](generating_human_interaction_motions_in_scenes_with_text_control.md)**

:   提出 TeSMo，一个文本控制的场景感知运动生成方法，通过在大规模运动数据上预训练文本-运动扩散模型，再用增强的场景感知分支进行微调，分两阶段（导航+交互）生成角色在 3D 场景中避障行走并与物体交互（如坐下）的真实运动序列。

**[Getting It Right Improving Spatial Consistency In Text-To-Image Models](getting_it_right_improving_spatial_consistency_in_text-to-image_models.md)**

:   系统性调查文本到图像模型的空间关系生成缺陷，发现现有视觉-语言数据集严重缺乏空间描述，据此创建 SPRIGHT 数据集（~600 万张图像重标注空间关系），仅用 <500 张多物体图像微调即在 T2I-CompBench 空间得分上达到 SOTA（0.2133），相比基线提升 41%。

**[Getting It Right Improving Spatial Consistency In Texttoimag](getting_it_right_improving_spatial_consistency_in_texttoimag.md)**

:   发现现有VL数据集严重缺乏空间关系描述（如left/right/above/behind出现率极低），构建了首个空间聚焦的大规模数据集SPRIGHT（600万张图像重描述），仅用0.25%数据微调即可提升22%空间一致性得分，用<500张多物体图像微调达到T2I-CompBench空间SOTA 0.2133。

**[Himo A New Benchmark For Full-Body Human Interacting With Multiple Objects](himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)**

:   提出 HIMO，首个大规模全身人体与多物体交互的 4D MoCap 数据集（3.3K序列，4.08M帧），并附带详细的文本描述和时间段分割标注，提出双分支条件扩散模型及自回归管线，生成协调的多物体交互动作序列。

**[Hybridbooth Hybrid Prompt Inversion For Efficient Subje](hybridbooth_hybrid_prompt_inversion_for_efficient_subje.md)**

:   提出 HybridBooth，融合优化方法和直接回归方法的优势——先用预训练编码器（Word Embedding Probe）生成初始 word embedding，再通过残差精细化（仅 3-5 步）快速适配特定主体，实现高效高保真的 subject-driven 生成。

**[Hybridbooth Hybrid Prompt Inversion For Efficient Subject-Driven Generation](hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)**

:   提出 HybridBooth，一种两阶段混合 prompt inversion 框架，通过先用回归器生成初始词嵌入（Probe），再用残差微调（Refinement）仅需 3-5 步迭代即可高效完成主体驱动的个性化图像生成。

**[Idempotent Unsupervised Representation Learning For Skeleton-Based Action Recogn](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)**

:   提出幂等生成模型（IGM），从理论上建立生成模型与最大熵编码（谱对比学习）的等价关系，通过在骨架数据的特征空间施加幂等约束，使生成模型的特征更紧凑、更适合识别任务，在 NTU 60 xsub 上将准确率从 84.6% 提升至 86.2%。

**[Implicit Style-Content Separation Using B-Lora](implicit_style-content_separation_using_b-lora.md)**

:   提出 B-LoRA，通过分析 SDXL 架构发现仅联合训练两个特定 transformer block 的 LoRA 权重（Block 4 控制内容、Block 5 控制风格）即可隐式实现单张图片的风格-内容分离，支持风格迁移、文本风格化、一致风格生成等多种任务。

**[Infinite-Id Identity-Preserved Personalization Via Id-Semantics Decoupling Parad](infinite-id_identity-preserved_personalization_via_id-semantics_decoupling_parad.md)**

:   提出 Infinite-ID，通过 ID-语义解耦范式将身份信息和文本语义信息分离处理——训练阶段停用文本交叉注意力以专注学习身份嵌入，推理阶段通过混合注意力机制和 AdaIN-mean 操作融合两路信息，在单张参考图下同时实现高保真身份保持和语义一致性。

**[Infiniteid Identitypreserved Personalization Via Idsema](infiniteid_identitypreserved_personalization_via_idsema.md)**

:   提出 Infinite-ID，通过 ID-语义解耦范式将身份信息和文本语义信息分离训练，再通过混合注意力机制和 AdaIN-mean 操作在推理时融合，实现高保真身份保持与精确语义控制的平衡。

**[Inftybrush Controllable Large Image Synthesis With Diffusion](inftybrush_controllable_large_image_synthesis_with_diffusion.md)**

:   提出首个在无限维函数空间中的条件扩散模型 ∞-Brush，通过交叉注意力神经算子实现可控条件生成，仅用 0.4% 像素训练即可在任意分辨率（最高 4096×4096）上生成保持全局结构的大图像。

**[Irgen Generative Modeling For Image Retrieval](irgen_generative_modeling_for_image_retrieval.md)**

:   将图像检索重新定义为生成式建模任务，提出 IRGen——一个序列到序列模型，通过语义图像分词器将图像转化为简短的离散 token 序列，然后自回归地生成查询图像最近邻的标识符，实现端到端可微分的检索并在三个标准基准上达到 SOTA。

**[Latent Guard A Safety Framework For Text-To-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)**

:   提出 Latent Guard 框架，通过在文本编码器的潜在空间中学习黑名单概念与输入提示词的嵌入映射，实现高效、灵活且可抵御对抗攻击的文本到图像生成安全检测。

**[Latent Guard A Safety Framework For Texttoimage Generation](latent_guard_a_safety_framework_for_texttoimage_generation.md)**

:   Latent Guard在T2I文本编码器上学习潜在空间检测黑名单概念。

**[Lazy Diffusion Transformer For Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)**

:   提出 LazyDiffusion，一种非对称编码器-解码器 Transformer 架构，通过上下文编码器压缩全局信息并仅在 mask 区域执行扩散去噪，实现了与全图生成方法质量相当但速度提升 10 倍的交互式图像编辑。

**[Lcm-Lookahead For Encoder-Based Text-To-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)**

:   提出 LCM-Lookahead 机制，利用 Latent Consistency Model 作为快捷路径在扩散训练中反向传播图像空间损失（如身份损失），结合注意力共享和一致性合成数据生成，显著提升基于编码器的人脸个性化中的身份保持和提示词对齐能力。

**[Lcmlookahead For Encoderbased Texttoimage Personalization](lcmlookahead_for_encoderbased_texttoimage_personalization.md)**

:   本文提出利用LCM（Latent Consistency Model）作为"快捷通道"，在扩散模型encoder训练中实现图像空间损失（如身份识别loss）的反向传播，配合自注意力特征共享和一致性数据生成，显著提升encoder-based人脸个性化的身份保持和prompt对齐能力。

**[Learning Differentially Private Diffusion Models Via Stochastic Adversarial Dist](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)**

:   提出 DP-SAD 框架，通过随机对抗蒸馏训练差分隐私扩散模型：利用扩散模型的时间步稀释 DP 噪声影响，引入判别器加速收敛，并结合梯度链式法则与 DP 后处理特性减少随机性引入，在不需要预训练的条件下实现了 SOTA 的隐私保护图像生成质量。

**[Learning Semantic Latent Directions For Accurate And Controllable Human Motion P](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)**

:   提出语义潜在方向（SLD）方法，通过构建一组正交潜在基方向并将未来运动假设表示为这些方向的线性组合，在随机人体运动预测中实现了更准确、更多样且语义可控的运动预测。

**[Learning Trimodal Relation For Audio-Visual Question Answering With Missing Moda](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)**

:   提出基于三模态关系的缺失模态AVQA框架，通过关系感知缺失模态生成器（RMM）和音视觉关系感知扩散模型（AVR），在推理时缺少音频或视觉模态的情况下依然能准确回答问题。

**[Learning Trimodal Relation For Audiovisual Question Answerin](learning_trimodal_relation_for_audiovisual_question_answerin.md)**

:   提出面向音视觉问答（AVQA）的缺失模态处理框架，通过Relation-aware Missing Modal生成器利用三模态关系召回缺失信息，再通过Audio-Visual Relation-aware扩散模型增强特征表示，即使缺少一个模态也能准确回答问题。

**[Lego Learning Egocentric Action Frame Generation Via Vi](lego_learning_egocentric_action_frame_generation_via_vi.md)**

:   提出 LEGO 模型，通过视觉指令微调增强 VLLM 的动作描述能力，并将 VLLM 的图像/文本嵌入作为额外条件注入扩散模型，实现从第一人称视角生成动作执行帧。

**[Lego Learning Egocentric Action Frame Generation Via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_visual_instruction_tuning.md)**

:   提出第一人称视角动作帧生成新问题，通过视觉指令微调 VLLM 生成丰富动作描述并将其嵌入作为扩散模型的额外条件，实现高质量的自我中心动作图像合成。

**[Livephoto Real Image Animation With Text-Guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)**

:   提出 LivePhoto 图像动画框架，通过运动强度估计模块和文本重加权模块解决文本到运动映射的歧义性，实现基于真实图像和文本描述的高质量视频生成，且用户可额外控制运动强度。

**[Local Action-Guided Motion Diffusion Model For Text-To-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)**

:   提出 GuidedMotion，以局部动作作为细粒度控制信号引导全局运动扩散生成，通过语义图解析和图注意力网络估计引导权重，支持连续可调的运动控制，在生成复杂多动作运动时优势显著。

**[M2D2M Multi-Motion Generation From Text With Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)**

:   提出 M2D2M，基于离散扩散模型生成多段连续人体运动序列，通过动态转移概率和两阶段采样策略（TPS）实现动作间平滑过渡，且无需额外的多运动训练数据。

**[Macdiff Unified Skeleton Modeling With Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)**

:   首次将扩散模型用于骨架表征学习，提出 Masked Conditional Diffusion（MacDiff）框架，通过语义编码器提取掩码骨架的表征来引导条件扩散解码器进行去噪，统一了骨架的判别式和生成式建模。

**[Magdiff Multi-Alignment Diffusion For High-Fidelity Video Generation And Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)**

:   提出首个统一视频生成与编辑的多对齐扩散模型 MagDiff，通过主体驱动对齐、自适应提示对齐和高保真对齐三种策略，在单一无微调框架中同时实现高质量视频生成与编辑。

**[Magiceraser Erasing Any Objects Via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)**

:   提出基于扩散模型的对象擦除框架 MagicEraser，通过内容初始化、提示调优和语义感知注意力重聚焦三阶段设计，无需用户输入文本即可高质量擦除对象并生成和谐背景。

**[Memory-Efficient Fine-Tuning For Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)**

:   提出 TuneQDM，首个面向量化扩散模型的内存高效微调方法，通过多通道量化缩放更新和时间步感知缩放策略，在 4-bit 量化模型上实现了接近全精度模型的个性化生成效果。

**[Mixdq Memory-Efficient Few-Step Text-To-Image Diffusion Models With Metric-Decou](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)**

:   针对少步扩散模型量化的特殊挑战，提出 MixDQ 混合精度量化方法，通过 BOS 感知量化处理文本嵌入中的异常值、度量解耦的敏感性分析分离质量与内容影响，在 1-step SDXL-turbo 上实现 W4A8 无损量化。

**[Mixdq Memoryefficient Fewstep Texttoimage Diffusion Models W](mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)**

:   针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。

**[Mofa-Video Controllable Image Animation Via Generative Motion Field Adaptions In](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)**

:   提出 MOFA-Video，通过设计多个领域感知的运动场适配器（MOFA-Adapter），在冻结的 Stable Video Diffusion 上实现多域可控图像动画，支持手绘轨迹、人脸关键点等多种控制信号及其零样本组合。

**[Motionchain Conversational Motion Controllers Via Multimodal](motionchain_conversational_motion_controllers_via_multimodal.md)**

:   MotionChain构建视觉-运动语言模型，通过VQ-VAE将动作token化支持多轮对话运动生成。

**[Motionchain Conversational Motion Controllers Via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)**

:   提出 MotionChain，一个视觉-运动-语言统一模型，通过多模态提示在多轮对话中生成连续、长期的人体运动序列，支持文本、图像和运动的联合理解与生成。

**[Motionlcm Real-Time Controllable Motion Generation Via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)**

:   提出 MotionLCM，首次将一致性蒸馏引入人体运动生成领域，在运动潜在空间中实现单步/少步推理的实时运动生成（~30ms/序列），并通过 Motion ControlNet 实现潜在空间中的实时可控运动生成。

**[Mutual Learning For Acoustic Matching And Dereverberation Via Visual Scene-Drive](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)**

:   提出 MVSD，一个基于扩散模型的互学习框架，将视觉声学匹配（VAM）和去混响作为对称互逆任务联合训练，利用两者的互惠关系克服配对数据稀缺问题，并首次将扩散模型用于视觉引导的混响风格迁移。

**[Neusdfusion A Spatial-Aware Generative Model For 3D Shape Completion Reconstruct](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)**

:   提出 NeuSDFusion，一个基于混合三平面 SDF 表示（NeuSDF）和空间感知 Transformer 自编码器的 3D 形状生成框架，通过保持三平面间的空间对应关系，在无条件生成、多模态形状补全、单视图重建和文本到 3D 生成等任务上达到 SOTA 性能。

**[Nl2Contact Natural Language Guided 3D Hand-Object Contact Modeling With Diffusio](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)**

:   提出 NL2Contact，首次利用自然语言描述来可控地建模 3D 手-物体接触图，通过分阶段扩散模型从文本生成手势姿态和接触区域，并构建了首个带有细粒度语言描述的手-物体接触数据集 ContactDescribe。

**[Omg Occlusion-Friendly Personalized Multi-Concept Generation In Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)**

:   提出 OMG，一种遮挡友好的个性化多概念图像生成框架，通过两阶段采样（布局生成 + 概念噪声融合）实现强身份保持和自然光照协调，且可即插即用地搭配各种单概念模型（LoRA、InstantID）无需额外训练。

**[Omnissr Zero-Shot Omnidirectional Image Super-Resolution Using Stable Diffusion ](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)**

:   提出 OmniSSR，首个基于扩散模型的零样本全向图像超分方法，通过十八面切线投影信息交互（OTII）和梯度分解（GD）校正技术，利用 Stable Diffusion 的图像先验实现保真度和真实感的平衡，无需任何训练或微调。

**[Pixel-Aware Stable Diffusion For Realistic Image Super-Resolution And Personaliz](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)**

:   提出 PASD（Pixel-Aware Stable Diffusion），通过像素感知交叉注意力（PACA）模块使扩散模型在像素级感知图像局部结构，结合退化去除模块和可调噪声调度，实现了真实图像超分辨率和个性化风格化的统一框架，只需替换底座模型即可切换风格。

**[Pixelaware Stable Diffusion For Realistic Image Superre](pixelaware_stable_diffusion_for_realistic_image_superre.md)**

:   提出像素感知稳定扩散（PASD）网络，通过像素感知交叉注意力（PACA）在潜空间中实现像素级结构保持，配合退化移除模块和可调噪声调度，统一解决真实图像超分辨率和个性化风格迁移两大任务。

**[Ponymation Learning Articulated 3D Animal Motions From ](ponymation_learning_articulated_3d_animal_motions_from_.md)**

:   提出从原始、无标签的互联网视频中学习动物 3D 关节运动生成模型的方法，核心是一个视频光几何自编码框架，将训练视频分解为静止姿态形状、关节姿态序列和纹理，实现无需姿态标注的 3D 运动 VAE 学习。

**[Ponymation Learning Articulated 3D Animal Motions From Unlabeled Online Videos](ponymation_learning_articulated_3d_animal_motions_from_unlabeled_online_videos.md)**

:   提出 Ponymation，首次从未标注的网络视频中学习铰接式 3D 动物运动的生成模型，无需姿态标注或参数化形状模板，通过视频光度-几何自编码框架和运动 VAE，能在数秒内从单张图像生成逼真的 4D 动画。

**[Powerful And Flexible Personalized Text-To-Image Generation Via Reinforcement Le](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)**

:   提出基于确定性策略梯度（DPG）的强化学习框架用于个性化文本到图像生成，通过"前瞻"机制和DINO奖励函数捕获长期视觉一致性，大幅提升生成图像的视觉保真度。

**[Powerful And Flexible Personalized Texttoimage Generation Vi](powerful_and_flexible_personalized_texttoimage_generation_vi.md)**

:   将个性化T2I建模为DPG框架，引入Q函数和向前看机制捕获长期视觉一致性。

**[Realistic Human Motion Generation With Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)**

:   提出 CrossDiff 框架，通过统一编码和交叉解码机制融合 3D 与 2D 运动信息，利用交叉扩散实现更精细的全身运动细节捕获，并支持从野外 2D 数据学习 3D 运动生成。

**[Rejection Sampling Imle Designing Priors For Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)**

:   揭示 IMLE 方法中训练/测试时潜在码分布不对齐问题，提出 RS-IMLE 通过拒绝采样改变训练先验分布，在九个少样本图像数据集上平均降低 45.9% FID。

**[Removing Distributional Discrepancies In Captions Improves I](removing_distributional_discrepancies_in_captions_improves_i.md)**

:   发现训练图文对齐模型时正负caption之间存在被忽视的数据集级别分布偏差（如GPT生成负样本时倾向用elephant替换giraffe），提出用纯文本分类器过滤高置信样本来消除偏差，结合替换型+交换型两类负样本微调LLaVA-1.5，在Winoground、SeeTRUE等多个基准上大幅超越现有方法。

**[Removing Distributional Discrepancies In Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)**

:   发现正负描述文本在数据集层面存在分布偏差（如词频差异），提出用纯文本分类器过滤偏差数据，微调 LLaVA-1.5 获得 SOTA 图文对齐评分模型 LLaVA-score。

**[Renoise Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)**

:   提出 ReNoise 迭代重噪方法改进扩散模型的图像反演质量，通过在每个反演步骤多次应用 UNet 并平均预测来提升轨迹估计精度，尤其适用于少步扩散模型（SDXL Turbo、LCM）。

**[Ringid Rethinking Tree-Ring Watermarking For Enhanced Multi-Key Identification](ringid_rethinking_tree-ring_watermarking_for_enhanced_multi-key_identification.md)**

:   本文深入分析了 Tree-Ring 水印方法的鲁棒性来源（发现分布偏移是其验证任务中意外的隐藏助力），揭示其在多密钥识别任务中的严重缺陷，并提出 RingID——一种多通道异构水印框架，通过离散化、无损嵌入和更圆环形设计，将2048个密钥的识别准确率从0.07提升至0.82。

**[Robust-Wide Robust Watermarking Against Instruction-Driven Image Editing](robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)**

:   本文提出 Robust-Wide，首个针对指令驱动图像编辑的鲁棒水印方法，核心创新是部分指令驱动去噪采样引导（PIDSG）模块——在训练中将编辑过程的最后k步梯度打通，迫使水印嵌入到语义感知区域，实现编辑后仅约2.6% 的64位水印误码率。

**[Sair Learning Semantic-Aware Implicit Representation](sair_learning_semantic-aware_implicit_representation.md)**

:   本文提出语义感知隐式表示（SAIR），通过构建语义隐式表示（SIR）和外观隐式表示（AIR）两个模块，将CLIP提取的文本对齐语义嵌入融入隐式神经函数，使其在大面积缺失区域的图像修复任务中远超仅依赖外观信息的方法，在CelebAHQ上 PSNR 提升1.65-2.69dB。

**[Scalable Group Choreography Via Variational Phase Manifold Learning](scalable_group_choreography_via_variational_phase_manifold_learning.md)**

:   本文提出 PDVAE（Phase-conditioned Dance VAE），一种基于相位参数的变分生成模型用于可扩展群舞生成——通过在频域学习舞蹈运动的相位流形（幅度、频率、偏移、相移），实现对**任意数量**舞者的高质量群舞生成，且内存消耗恒定不变，在AIOZ-GDance和AIST-M数据集上全面超越现有方法。

**[Scaledreamer Scalable Text-To-3D Synthesis With Asynchronous Score Distillation](scaledreamer_scalable_text-to-3d_synthesis_with_asynchronous_score_distillation.md)**

:   本文提出异步分数蒸馏（ASD），通过将扩散时间步前移（而非微调扩散模型）来降低噪声预测误差、对齐渲染图像分布，解决了VSD微调破坏文本理解能力的问题，实现了稳定训练且可扩展至10万条文本提示的prompt-amortized 3D生成器训练。

**[Scaledreamer Scalable Textto3D Synthesis With Asynchronous S](scaledreamer_scalable_textto3d_synthesis_with_asynchronous_s.md)**

:   提出异步分数蒸馏(ASD)，通过将扩散时间步前移（而非微调扩散模型）来减小噪声预测误差，实现稳定的3D生成器训练并可扩展到100K文本提示，保持扩散模型的文本理解能力不受损。

**[Shapefusion A 3D Diffusion Model For Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)**

:   提出一种基于掩码扩散训练策略的3D网格局部编辑方法ShapeFusion，通过在顶点空间直接操作实现完全局部化、可解释的3D形状编辑，无需潜在空间优化。

**[Shedding More Light On Robust Classifiers Under The Lens Of Energy-Based Models](shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)**

:   通过将鲁棒判别分类器重新解释为基于能量的模型（EBM），揭示了对抗训练的能量动态规律，提出了基于能量加权的对抗训练方法WEAT，并展示了鲁棒分类器隐含的生成能力。

**[Smoodi Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)**

:   提出SMooDi——首个将预训练文本-动作模型适配为风格化动作生成的扩散模型，通过风格适配器和双重风格引导（无分类器引导+基于分类器引导）实现内容文本与风格动作序列驱动的多样化风格动作生成。

**[Soft Prompt Generation For Domain Generalization](soft_prompt_generation_for_domain_generalization.md)**

:   提出 SPG（Soft Prompt Generation），首次将生成模型引入 VLM 的 prompt learning，通过 CGAN 从图像动态生成实例特定的软提示，将域知识存储在生成模型中而非提示向量中，实现更好的领域泛化性能。

**[Source Prompt Disentangled Inversion For Boosting Image Editability With Diffusi](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)**

:   提出SPDInv——一种源提示解耦反演方法，通过将反演过程建模为不动点搜索问题并利用预训练扩散模型求解，使反演噪声码与源提示解耦，显著提升基于文本驱动的图像编辑质量。

**[Text2Place Affordance-Aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)**

:   提出Text2Place——首个通过文本引导实现真实感人物放置的方法，利用SDS损失优化基于高斯blob的语义掩码学习场景可供性（affordance），再通过主体条件修复实现身份保持的人物放置。

**[Text2Place Affordanceaware Text Guided Human Placement](text2place_affordanceaware_text_guided_human_placement.md)**

:   提出 Text2Place，通过 SDS 损失优化 Gaussian blob 参数化的语义掩码学习场景中的人体 affordance，再结合主体条件修复实现逼真的文本引导人物放置，无需大规模训练。

**[Textdiffuser-2 Unleashing The Power Of Language Models For Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)**

:   利用两个语言模型分别进行布局规划和布局编码，实现更灵活、更多样化的视觉文本渲染，在文本准确性和风格多样性之间取得更好的平衡。

**[Textdiffuser2 Unleashing The Power Of Language Models F](textdiffuser2_unleashing_the_power_of_language_models_f.md)**

:   TextDiffuser-2 利用两个语言模型（一个用于布局规划、一个用于布局编码）实现灵活自动的文本渲染，克服了现有方法在灵活性、布局能力和样式多样性方面的局限。

**[The Fabrication Of Reality And Fantasy Scene Generation With Llm-Assisted Prompt](the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)**

:   提出 Realistic-Fantasy Benchmark (RFBench) 评估扩散模型在创意/知识密集型 prompt 上的表现，并设计 training-free 的 RFNet 框架，通过 LLM 辅助 prompt 解读和语义对齐评估模块来增强扩散模型对抽象和想象性概念的生成能力。

**[Toward Tiny And High-Quality Facial Makeup With Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)**

:   提出 Data Amplify Learning (DAL) 学习范式，用 Diffusion-based Data Amplifier 从仅 5 张标注图像"放大"生成大量配对训练数据，训练出仅 80K 参数的 TinyBeauty 模型，在 iPhone 13 上以 460fps 实现 SOTA 妆容迁移效果。

**[Towards Reliable Advertising Image Generation Using Human Fe](towards_reliable_advertising_image_generation_using_human_fe.md)**

:   针对电商广告图像生成中大量不可用图像（空间不匹配、尺寸不匹配、不显著、形状幻觉）的问题，构建了百万级RF1M数据集训练多模态检测网络RFNet，并提出基于RFNet反馈微调扩散模型的RFFT方法（含Consistent Condition正则化），将可用率从约50%提升至接近100%且不损失美观性。

**[Towards Reliable Advertising Image Generation Using Human Feedback](towards_reliable_advertising_image_generation_using_human_feedback.md)**

:   构建百万级人工标注广告图像数据集 RF1M，提出多模态 RFNet 自动检测生成图像的可用性，并设计 Consistent Condition 正则化驱动的 RFFT 微调方法，将广告图像可用率从 56.4% 提升至 85.5%。

**[Unveiling Advanced Frequency Disentanglement Paradigm For Low-Light Image Enhanc](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)**

:   提出一种通用的频率解耦学习范式，通过轻量级 ACCA 模块进行粗调低频恢复，再通过 LDRM 模块结合低频一致性约束实现高频细化，仅增加 88K 参数即可为六种 SOTA 低光增强模型带来最高 7.68dB 的 PSNR 提升。

**[Vfusion3D Learning Scalable 3D Generative Models From Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)**

:   提出利用预训练视频扩散模型（EMU Video）作为多视图数据引擎，通过微调使其生成3D一致的多视图视频，从而构建约300万合成数据训练前馈式3D生成模型VFusion3D，实现从单张图片秒级生成3D资产，用户偏好率超过90%。

**[Videoshop Localized Semantic Video Editing With Noise-Extrapolated Diffusion Inv](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)**

:   提出Videoshop——一种免训练的局部语义视频编辑方法，用户可通过任意图像编辑工具修改视频首帧，系统基于噪声外推扩散反演和隐变量归一化技术，自动将编辑传播到所有帧，同时保持语义、空间和时序一致性，在10个指标上超越6个基线方法。

**[Webrpg Automatic Web Rendering Parameters Generation For Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)**

:   提出Web渲染参数生成（WebRPG）新任务，旨在根据HTML代码自动生成网页元素的视觉呈现参数（布局、文本样式、颜色），通过VAE压缩渲染参数和定制HTML嵌入捕获语义层次信息，建立自回归和扩散两种基线模型，其中自回归模型显著优于扩散模型和GPT-4。

**[Wildvidfit Video Virtual Try-On In The Wild Via Image-Based Controlled Diffusion](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)**

:   WildVidFit 提出了一个无需视频训练的虚拟试穿框架，利用基于图像的条件扩散模型和扩散引导模块（VideoMAE + DINO-V2），实现了在野外复杂视频中保持时序一致性的服装试穿效果。

**[Xpsr Cross-Modal Priors For Diffusion-Based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)**

:   XPSR 利用多模态大语言模型（MLLM）提取高层和低层语义先验，通过 Semantic-Fusion Attention 和 Degradation-Free Constraint 引导扩散模型实现高保真、高真实感的图像超分辨率。

**[Xpsr Crossmodal Priors For Diffusionbased Image Superresolut](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)**

:   XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。

**[Zigma A Dit-Style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)**

:   ZigMa 提出了一种 DiT 风格的 Zigzag Mamba 扩散模型，通过异构逐层锯齿形扫描方案保持空间连续性，以零参数/显存开销实现优于 Mamba 基线的生成质量，同时相比 Transformer 具备线性复杂度优势。

**[Ziplora Any Subject In Any Style By Effectively Merging Loras](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)**

:   ZipLoRA 提出了一种廉价高效的 LoRA 合并方法，通过学习逐列合并系数并最小化列间余弦相似度，实现了将独立训练的主题 LoRA 和风格 LoRA 无超参数合并，在扩散模型中生成"任意主题 × 任意风格"的个性化图像。
