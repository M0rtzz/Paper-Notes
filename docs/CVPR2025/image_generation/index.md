---
title: >-
  CVPR2025 图像生成方向 163篇论文解读
description: >-
  163篇CVPR2025 图像生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**📷 CVPR2025** · **163** 篇论文解读

**[3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)**

:   提出基于新型原语表示PrimX和Diffusion Transformer的原生3D生成模型3DTopia-XL，能从文本或图像输入生成带有高分辨率几何、纹理和PBR材质的高质量3D资产，在质量和效率上显著超越现有方法。

**[A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)**

:   系统研究了使用decoder-only LLM作为文本到图像扩散模型文本编码器的效果，发现直接使用最后一层embedding效果差于T5，但通过层归一化平均（layer-normalized averaging）聚合所有层的embedding可显著超越T5基线。

**[AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)**

:   提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。

**[HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)**

**[AniDoc: Animation Creation Made Easier](anidoc_animation_creation_made_easier.md)**

**[AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)**

**[Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)**

:   提出一种基于频谱学习的 AI 生成图像检测方法，通过在频域提取分辨率不变的特征，使检测器能够在任意分辨率的输入图像上保持鲁棒性，解决了现有方法对特定分辨率依赖的问题。

**[Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)**

:   本文提出InvSR，通过训练一个噪声预测网络来实现扩散反演（Diffusion Inversion），利用预训练扩散模型的图像先验进行超分辨率，支持1-5步任意步数采样，即使单步采样也能达到或超过现有SOTA方法的效果。

**[ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)**

:   本文提出ArtiFade，首个解决"瑕疵主题驱动生成"问题的方法，通过构建瑕疵-无瑕疵配对数据集、部分微调扩散模型的cross-attention权重并优化artifact-free embedding，使得现有主题驱动方法（Textual Inversion、DreamBooth）能从带水印/贴纸/对抗噪声等瑕疵的图像中生成高质量无伪影的主题图像。

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，用双向布朗桥扩散模型建模地面 LSST 和太空 Euclid 两大天文巡天之间的随机映射关系，实现概率性跨巡天翻译与稀有事件检测（强引力透镜），并证明 epsilon-prediction 训练目标兼具重建质量和似然性优势。

**[AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)**

:   本文提出AutoPresent框架和SlidesBench基准，首次系统研究从自然语言指令生成演示幻灯片的任务——通过让LLM生成Python代码（而非端到端图像生成）来创建PPTX幻灯片，配合SlidesLib工具库和迭代优化，8B参数的开源模型达到接近GPT-4o的效果。

**[AvatarArtist: Open-Domain 4D Avatarization](avatarartist_open-domain_4d_avatarization.md)**

:   提出 AvatarArtist，通过 GAN 和扩散模型协同构建多域 image-triplane 数据集，训练 DiT 生成参数化三平面 + 运动感知跨域渲染器，实现从任意风格单张肖像生成可驱动的 4D 头像。

**[Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)**

:   系统性地将学习式图像处理中卷积的替代/扩展算子组织为五大家族（分解型、自适应加权型、基自适应型、积分/核型和注意力型），并从线性、局部性、等变性、计算成本和任务适用性等多个维度进行比较分析。

**[Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)**

:   提出 ActINR，通过在 INR 中跨帧共享权重、仅用偏置（bias）建模运动的方式实现连续视频表示，在 10× 慢动作、4× 空间超分+2× 时间超分、去噪和修复任务上大幅超越现有方法（平均 3-6dB 提升）。

**[BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 首次将扩散模型的 token 压缩重新定义为生成+分类的双目标优化问题，提出拉普拉斯门控 token 合并（L-GTM）和插值-外推 KV 下采样（IE-KVD）两个频率感知算子，在保持生成质量同时显著提升分类准确率（ImageNet-1K 70%合并比下 Acc +7.15%，FID -0.34）。

**[BootPlace: Bootstrapped Object Placement with Detection Transformers](bootplace_bootstrapped_object_placement_with_detection_transformers.md)**

:   提出 BootPlace，将物体放置问题重新定义为"放置即检测"问题，通过在物体减除背景上训练检测变换器识别候选区域，再用负相关语义互补将目标物体匹配到最佳区域，在 Cityscapes 上 top-5 IOU 比 SOTA 提升约 4×。

**[BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)**

:   提出 BooW-VTON，通过高质量伪数据构建 + 野外数据增广 + 试穿定位损失，训练出无需人体解析掩码的虚拟试穿扩散模型，在 VITON-HD/StreetVTON/WildVTON 多个基准上全面超越现有方法。

**[CacheQuant: Comprehensively Accelerated Diffusion Models](cachequant_comprehensively_accelerated_diffusion_models.md)**

:   提出 CacheQuant，一种无需训练的范式，通过联合优化模型缓存（temporal level）和量化（structural level）来全面加速扩散模型，在 Stable Diffusion 上实现 5.18× 加速和 4× 压缩，CLIP score 仅损失 0.02。

**[CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model](camfreediff_camera-free_image_to_panorama_generation_with_diffusion_model.md)**

:   提出 CamFreeDiff，通过在多视图扩散框架中集成轻量级 3-DoF 单应性估计器，实现从无相机参数的单张图像生成 360° 全景图，FID 从 MVDiffusion 的 42.4 降至 27.0，且无需微调即可泛化到域外数据。

**[Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)**

:   提出 InterPose，利用预训练视频生成模型在两张少/无重叠图像之间"幻想"中间帧，配合自一致性评分选择最佳视频，在 DUSt3R 基础上一致提升四个数据集的位姿估计精度。

**[Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)**

:   提出通道级噪声调度方法，让单一扩散模型架构通过不同噪声调度实现精度优先（SDM, T=4）和多样性优先（PDM, T=1000）两种逆渲染模式，同时引入 ILR 隐式光照表征支持逐像素环境图推理和真实物体插入。

**[ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)**

:   提出 ChatGen，将文本到图像生成中的 prompt 编写、模型选择和参数配置三个繁琐步骤自动化，通过多阶段进化训练策略（ChatGen-Evo）让用户以自由聊天方式描述需求即可获得高质量生成图像。

**[Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](classifier-free_guidance_inside_the_attraction_basin_may_cause_memorization.md)**

:   从动力系统视角提出"吸引盆地"概念解释扩散模型记忆化现象——CFG 在吸引盆地内施加会导致轨迹收敛到记忆化训练图像，通过检测转折点延迟 CFG 启动（配合反向引导 OG）可零额外开销地缓解记忆化。

**[CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)**

:   提出 CleanDIFT，通过轻量级无监督微调（单卡 A100 仅 30 分钟），使扩散模型直接在干净图像上提取高质量语义特征，消除了传统方法需要加噪和调时间步的限制，在语义对应、深度估计、分割等多任务上显著超越标准扩散特征。

**[CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)**

:   系统揭示 CLIP 在多目标场景中的两类偏差——文本编码器偏向先提到的物体、图像编码器偏向大物体，并追溯偏差根源至对比训练过程中训练数据里大物体被先提到的统计规律。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)**

:   提出 coDrawAgents，由 Interpreter、Planner、Checker、Painter 四个专家 agent 组成的交互式多智能体对话框架，通过分而治之的增量布局规划、视觉上下文感知推理和显式错误纠正，在 GenEval 上达到 0.94（SOTA）、DPG-Bench 上 85.17（SOTA）。

**[Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)**

:   提出 CoDe（协同解码），将 VAR 的多尺度推理分解为大模型草稿（低频小尺度）+ 小模型精修（高频大尺度）的协作流程，实现 1.7× 加速、50% 显存降低，FID 仅从 1.95 微增至 1.98。

**[Color Alignment in Diffusion](color_alignment_in_diffusion.md)**

:   提出颜色对齐扩散方法，通过将中间采样或预测结果投影到条件颜色空间（最近邻颜色映射），使扩散模型在保持结构生成自由度的同时严格遵循给定的颜色分布（颜色值+比例），支持重训练、微调和零样本三种设置。

**[Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](community_forensics_using_thousands_of_generators_to_train_fake_image_detectors.md)**

:   构建包含 4803 个生成模型、270 万张图像的 Community Forensics 数据集，发现即使架构相似的模型也能通过增加数量显著提升假图检测泛化性，在多个基准上达到最优平均 mAP 0.966。

**[Composing Parts for Expressive Object Generation](composing_parts_for_expressive_object_generation.md)**

:   提出 PartComposer，一种无需训练的方法，通过并行"部件扩散"从注意力图中定位对象部件，再用区域扩散为每个部件独立生成用户指定的细粒度属性（颜色、风格、描述），实现部件级可控图像合成。

**[Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)**

:   提出基于预训练扩散模型的人体重光照和背景协调统一框架，通过粗到精策略（球谐函数 ControlNet 提供粗光照 + 扩散模型学习精细残差）和无监督运动 ControlNet 实现静态和视频场景的光照一致重光照。

**[Concept Lancet: Image Editing with Compositional Representation Transplant](concept_lancet_image_editing_with_compositional_representation_transplant.md)**

:   提出 Concept Lancet (CoLan)，一种零样本即插即用的图像编辑框架，通过将源图像的隐表示稀疏分解为视觉概念向量的线性组合，然后根据编辑任务（替换/添加/删除）进行定制化概念移植，解决了编辑强度校准难题。

**[Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)**

:   提出 Concept Replacer，通过少样本训练的概念定位器精确识别去噪过程中的敏感概念区域，再用训练免费的双提示交叉注意力（DPCA）将定位区域替换为安全内容，实现精确局部概念替换而非全局图像失真。

**[ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)**

:   提出 ConceptGuard，通过移位嵌入、概念绑定提示、记忆保持正则化和优先队列回放四种策略，实现持续个性化 T2I 生成中灾难性遗忘和概念混淆的缓解，在多概念基准上大幅超越现有方法。

**[Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)**

:   分析 SDXL 自注意力层对风格和结构的敏感度差异，发现仅在最敏感的子集层中注入条件信息即可显著改善多条件生成中的风格-内容 trade-off，无需额外训练。

**[Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)**

:   提出 Cinemo，基于扩散模型的图像动画方法，通过学习运动残差（而非直接预测帧）分布大幅提升与输入图像的时间一致性，配合 SSIM 运动强度控制和 DCT 噪声初始化实现精细可控的 I2V 生成，在 UCF-101 和 MSR-VTT 上全面超越现有方法。

**[Controllable Human Image Generation with Personalized Multi-Garments](controllable_human_image_generation_with_personalized_multi-garments.md)**

:   提出 BootComp，通过两阶段框架（分解模块从真人图像自举生成合成配对数据 + 组合模块用双 SDXL 网络通过扩展自注意力注入多服装特征），实现给定多件参考服装的可控人体图像生成，MP-LPIPS 比基线提升 30%。

**[CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)**

:   提出 CustAny，一种零样本通用物体定制化框架，通过 DINOv2+MAE 双编码器提取全面身份特征、全局+局部双层级注入、以及对比学习身份解耦，从单张参考图生成保持身份的多样化图像，在通用/人脸/试穿三个领域全面超越专用方法。

**[Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales](data-free_group-wise_fully_quantized_winograd_convolution_via_learnable_scales.md)**

:   提出完全量化 Winograd 卷积的方法——通过可学习的对角缩放矩阵均衡 Winograd 域输出的动态范围差异，仅用随机高斯噪声（无真实数据）微调缩放参数，在 InstaFlow/SD v1.5 等扩散模型上实现近无损的 W8A8 Winograd 加速（CPU 卷积层加速 31.3%）。

**[DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)**

:   提出 DeClotH，用 DMTet 表示+3D 模板正则化（SMPLicit/SMPL）分别重建服装和人体，并训练 ClothDiffusion（ControlNet 微调）生成纯服装图像作为 SDS 损失指导，在单张图像下实现可分解的 3D 服装+人体重建，在 4D-DRESS 上超越所有基线。

**[Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning](decouple-then-merge_finetune_diffusion_models_as_multi-task_learning.md)**

:   提出 DeMe，将扩散模型训练视为多任务学习问题——不同时间步是不同"任务"且存在梯度冲突，通过将时间步解耦为 N 个范围分别微调再合并回单一模型，在零推理开销下显著提升生成质量（CIFAR10 FID 从 4.42 降至 3.51）。

**[Decoupling Training-Free Guided Diffusion by ADMM](decoupling_training-free_guided_diffusion_by_admm.md)**

:   提出 ADMMDiff，用 ADMM 框架将训练免费条件生成中的扩散项和引导项解耦为两个变量（x 扩散, z 引导, 约束 x=z），用近端算子分别处理，消除了手动调权重的需求并提供收敛保证，在分割/素描/文本引导生成上大幅超越 DPS 和 MPGD。

**[Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)**

:   提出 DenoisFM，用 DDPM 直接在光谱域去噪生成功能映射矩阵来预测 3D 形状间的点对应，配合无监督符号纠正网络消除拉普拉斯特征函数的符号歧义，在多个形状匹配基准上大幅超越大规模学习方法同时匹配描述符方法。

**[FreeMCG: Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI](derivative-free_diffusion_manifold-constrained_gradient_for_unified_xai.md)**

:   提出 FreeMCG，利用扩散模型的 Tweedie 去噪生成流形上的集成粒子，通过集成卡尔曼滤波（EnKF）理论在无需模型权重的情况下近似分类器的流形约束梯度，统一实现黑盒特征归因和反事实解释，人类评估显著优于白盒方法。

**[DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)**

:   提出 DexGrasp Anything，将三种物理约束（表面拉力、外穿透排斥、自穿透排斥）集成到扩散模型的训练和采样中，配合 LLM 增强的点云+语义物体表示，在 5 个基准上达到 SOTA 灵巧抓取性能（Suc.6 57.5%，比 SceneDiffuser 提升 2.2×）。

**[DiC: Rethinking Conv3x3 Designs in Diffusion Models](dic_rethinking_conv3x3_designs_in_diffusion_models.md)**

:   构建纯 3×3 卷积的扩散模型 DiC，通过沙漏编码解码器架构扩大感受野、稀疏跳跃连接减少冗余、阶段特定条件注入，在 ImageNet 256×256 上以 DiT-XL 的相当 FLOPs 实现 4.7× 吞吐量和更优 FID（3.89 vs 6.24, with CFG）。

**[DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)**

:   提出 DiffLocks，通过自动化 Blender 管线构建 4 万种发型的最大合成头发数据集，用发丝 VAE 将 10 万根发丝编码为 256×256 头皮纹理图，再训练 image-conditioned HDiT 从单张图像扩散去噪生成头皮纹理，实现 2.33 秒 10 万发丝重建，速度比基线快 8-23×。

**[DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)**

:   提出 DiffSensei，两阶段漫画面板生成框架——Stage 1 用掩码交叉注意力注入实现多角色布局控制和对话气泡放置，Stage 2 用 MLLM 适配器根据文本描述动态调整角色特征（表情/姿态/动作），解决了先前方法的"复制粘贴"效果问题。

**[Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)**

:   提出 Diffusion-4K，通过分区 VAE（F=16，无需重训练即避免 OOM）和小波域微调（WLF，用 Haar DWT 分解速度预测增强高频纹理），实现直接 4096×4096 分辨率图像合成，并构建 Aesthetic-4K 基准评估超高分辨率质量。

**[Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)**

:   利用预训练 T2I 模型（FLUX）的上下文网格生成能力自动合成 40 万配对训练数据，再微调该模型为零样本 image+text 条件生成器，在 DreamBench++ 上以零样本方式超越需微调的 DreamBooth LoRA。

**[Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)**

:   提出 DiffLens，用 k-稀疏自编码器将 U-Net 瓶颈层的多义性神经元分解为单义性特征，通过归因分析识别偏见特征并进行干预，实现扩散模型性别偏见的精确缓解（FD 从 0.564 降至 0.046）同时保持甚至提升生成质量。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)**

:   DiT-IC 将预训练 T2I 扩散 Transformer 适配为单步图像压缩重建模型，在 32x 下采样的深层潜空间工作，通过方差引导重建流、自蒸馏对齐和潜变量条件引导三种对齐机制，实现 SOTA 感知质量且解码比现有扩散 codec 快 30 倍。

**[DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)**

:   提出 DiverseFlow，一种训练免费的推理时方法，通过行列式点过程（DPP）在 ODE 求解过程中耦合多个样本使其互相排斥，在固定采样预算下显著提升流匹配模型的模式覆盖多样性（如从 5.64 个模式提升到 7.11 个）。

**[DiVoT: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)**

:   提出 DiVoT，用扩散去噪过程作为视频分词器的自监督训练信号——若分词器的连续表示能引导 U-Net 成功去噪视频则表示有效，统一视频理解和生成，LLM 通过高斯混合模型（而非 MSE 回归）预测连续视频特征。

**[Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)**

:   本文用 SDXL 为 VLN 指令中的视觉地标生成合成图像作为"想象"，通过 ViT 编码后拼接到文本指令 embedding 中输入 VLN agent，配合余弦相似度对齐损失，在 R2R 和 REVERIE 上一致提升导航成功率约 1%，初步验证了视觉想象作为语言与视觉之间桥梁的价值。

**[DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)**

:   提出 DoraCycle 使用两个多模态循环（文→图→文 和 图→文→图）对统一多模态生成模型做无配对域适应，仅用无配对目标域数据即可接近全配对训练效果（FID 27.44 vs 24.93），10% 配对+90% 无配对时几乎无损（FID 25.37）。

**[DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)**

:   提出 DreamCache 通过在单个去噪步（t=1）缓存参考图的 U-Net 中间特征，用轻量 25M 参数的条件适配器在生成时注入缓存特征，实现免微调、免编码器、即插即用的个性化图像生成。

**[DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)**

:   构建统一文生图+多种编辑任务（指令编辑/修补/拖拽/参考生成）的 2.5B DIT 模型，用 Qwen2-VL 替换文本编码器实现统一视觉-语言 prompt 理解，通过合成贴纸数据管线高效创建编辑训练数据，在生成和编辑上同时达到 SOTA。

**[DreamRelation: Bridging Customization and Relation Generation](dreamrelation_bridging_customization_and_relation_generation.md)**

:   DreamRelation 提出了一种关系感知的定制化图像生成框架，通过精心构建的解耦数据引擎、关键点匹配损失（KML）和局部 token 注入三大设计，在保持多目标身份一致性的同时准确生成文本指定的目标间关系（如拥抱、骑行等），在 RelationBench 上全面超越现有方法。

**[DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)**

:   提出 DualAnoDiff，通过双相互关联扩散模型（全局分支生成整体异常图像+异常分支生成局部异常部分）同时生成高质量的异常图像-掩码对，并引入背景补偿模块维持背景和物体形状的一致性，显著提升下游异常检测/定位/分类的性能。

**[Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)**

:   提出 Dual Diffusion Transformer (D-DiT)，在单一 MM-DiT 架构中同时使用连续扩散建模图像分布和离散掩码扩散建模文本分布，是首个端到端的全扩散多模态模型，支持图像生成、图像描述和视觉问答等全套任务。

**[Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)**

:   提出 DPIR，基于 SD3 (Diffusion Transformer) 的图像修复模型，通过轻量级低质量图像条件分支和全局-局部视觉双提示(dual prompting)分支，从多角度引入退化图像信息，首次系统性地将 DiT 应用于图像修复并取得 SOTA 性能。

**[Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)**

:   MotionReFit 提出了首个通用文本引导运动编辑框架，通过 MotionCutMix 数据增强技术动态生成训练三元组，配合自回归扩散模型和运动协调器，实现涵盖身体部位替换、风格迁移和细粒度调整的空间与时序编辑。

**[EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)**

:   EasyCraft 提出了一个端到端的自动角色捏脸框架，通过 MAE 预训练的通用 ViT 编码器将任意风格的面部图像映射为统一特征分布，再转换为游戏引擎捏脸参数，同时集成文本到图像技术支持文本输入，可轻松适配不同游戏引擎。

**[EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)**

:   提出 EDEN，从输入表示、模型架构和训练范式三个维度全面增强扩散模型在视频帧插值中的作用，通过 Transformer tokenizer 压缩中间帧为语义丰富的 1D token 表示、采用 DiT 替代 U-Net 架构、引入双流上下文整合机制（时序注意力 + 帧差嵌入），在 DAVIS 等大运动基准上 LPIPS 降低近 10%，且仅需 2 步去噪即可实现高质量生成。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。

**[Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)**

:   提出一种双层优化框架，将剪枝扩散模型的微调恢复（下层：蒸馏+扩散损失最小化）和不良概念遗忘（上层：引导模型远离目标概念）统一为单一阶段优化，解决了"先微调再遗忘"两阶段方法中微调最优点不等于遗忘最优初始化的循环依赖问题，在风格去除上 CSD 指标降低 27%。

**[Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](efficient_long_video_tokenization_via_coordinate-based_patch_reconstruction.md)**

:   提出 CoordTok，一种可扩展的视频 tokenizer，将视频编码为因子化 triplane 表示，解码器学习从随机采样的 $(x,y,t)$ 坐标到对应 patch 像素的映射（而非一次重建所有帧），使得可以直接在 128 帧长视频上训练大型 tokenizer，将 128 帧视频编码为仅 1280 个 token（基线需要 6144-8192 个），并驱动 DiT 实现 128 帧一次性视频生成（FVD 369.3 SOTA）。

**[Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)**

:   本文提出 ZOODiP，通过零阶优化在量化后的扩散模型上进行个性化（Textual Inversion），利用子空间梯度投影去噪和部分时间步采样加速训练，仅用 2.37GB 显存和前向传播即可达到与梯度方法可比的个性化效果，内存节省最高 8.2 倍。

**[EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)**

:   本文提出 EmoDubber，一个情感可控的电影配音架构，通过时长级对比学习对齐唇动与韵律、发音增强策略提升清晰度、基于流匹配的正负引导机制控制情感类型和强度，在唇形同步和发音清晰度上全面超越现有方法。

**[EmoEdit: Evoking Emotions through Image Manipulation](emoedit_evoking_emotions_through_image_manipulation.md)**

:   本文提出 EmoEdit，首个通过内容修改（而非仅颜色/风格调整）来唤起指定情感的图像操纵框架，构建了 40,120 对的 EmoEditSet 数据集，设计了可即插即用的 Emotion Adapter，在结构保持和情感唤起之间取得了显著平衡。

**[Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)**

:   提出 PN-Diffusion，利用正向播放和反向播放的舞蹈视频分别提取正负节奏条件，设计双向扩散与反向过程来联合训练 U-Net，增强生成音乐与舞蹈动作的节奏一致性和音乐质量，在 AIST++ 和 TikTok 数据集上 BCS 提升 1.80/3.85、BHS 提升 4.22/5.90。

**[Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)**

:   提出 DIAE，通过多模态美学感知模块（MAP）将模糊美学指令转化为 HSV/轮廓图+文本的多模态控制信号，并构建"非完美配对"数据集 IIAEData 配合双分支监督策略实现弱监督美学增强，在 LAION 和 MLLM 美学评分上达 SOTA。

**[Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)**

:   本文提出SPARCL，通过将真实图像特征注入快速T2I模型的padding嵌入来生成高保真微变化合成图像，并设计自适应margin损失过滤噪声合成样本聚焦难样本学习，将CLIP的组合理解准确率在四个基准上平均提升8%以上，在三个基准上超越SOTA 2%。

**[Erasing Undesirable Influence in Diffusion Models (EraseDiff)](erasing_undesirable_influence_in_diffusion_models.md)**

:   本文提出EraseDiff，将扩散模型的数据遗忘问题形式化为基于价值函数的约束优化问题，通过自然的一阶算法同时优化保留性能和擦除效果，在DDPM/Stable Diffusion上比SA快11倍、比SalUn快2倍，同时在保留-遗忘权衡上取得Pareto最优。

**[Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)**

:   本文揭示了扩散驱动TTA方法中源域与合成域之间存在隐性不对齐问题，提出Synthetic-Domain Alignment (SDA)框架，通过Mix of Diffusion (MoD)技术将源模型和目标数据同时对齐到同一个合成域，在分类、分割和多模态大语言模型上均取得了一致的性能提升。

**[EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)**

:   EvoTok 提出了一种基于残差潜在演化（Residual Latent Evolution）的统一图像 tokenizer，通过在共享潜空间中级联残差向量量化，使表示从浅层的像素级细节渐进演化到深层的语义级抽象，在仅用 13M 图像训练的情况下实现了 0.43 rFID 的重建质量，并在 7/9 个理解 benchmark 和 GenEval/GenAI-Bench 上取得优异效果。

**[FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)**

:   提出 FaithDiff，首次释放（fine-tune）预训练扩散模型先验用于图像超分辨率，并设计对齐模块桥接退化图像特征与扩散噪声隐空间，通过联合优化 encoder 和扩散模型实现高保真结构恢复。

**[FilmComposer: LLM-Driven Music Production for Silent Film Clips](filmcomposer_llm-driven_music_production_for_silent_film_clips.md)**

:   FilmComposer 首次将大语言模型多代理系统与波形/符号音乐生成相结合，模拟专业音乐人的工作流程（选点→作曲→编曲→混音），从无声电影片段自动生成高质量（48kHz）、高音乐性、具有发展性的电影配乐。

**[FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)**

:   FineLIP 通过位置编码拉伸（77→248 tokens）、自适应 Token 精炼模块（ATRM）和跨模态 Token 级对齐（CLIM），使 CLIP 模型能够处理长文本描述并实现细粒度视觉-文本匹配，在长描述检索任务上显著超越 Long-CLIP、TULIP 等现有方法。

**[FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)**

:   FlipSketch 首次实现从单张静态草图 + 文本描述自动生成无约束栅格草图动画，通过在 T2V 扩散模型上微调 LoRA、DDIM 反演参考帧机制和双注意力组合三大创新，在保持草图身份的同时生成流畅、动态的动画序列。

**[Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)**

:   提出 Focus-N-Fix，一种区域感知的 T2I 模型微调方法，通过定位问题区域并约束非问题区域不变，实现对伪影、过度性化、暴力等局部质量问题的精准修复，同时避免全局微调带来的灾难性遗忘和奖励黑客现象。

**[Font-Agent: Enhancing Font Understanding with Large Language Models](font-agent_enhancing_font_understanding_with_large_language_models.md)**

:   构建了包含 135,000 个字体-文本对的大规模多模态数据集 DFD，并提出 Font-Agent——一个基于视觉语言模型的字体理解代理，通过边缘感知追踪模块（EAT）捕捉字体笔画细节和动态直接偏好优化策略（D-DPO）精细化模型对字体风格的理解能力。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。

**[Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)**

:   提出一种姿态关联参考选择扩散网络，通过姿态相关性模块计算目标-参考姿态间的关联图并自适应选择最相关的参考特征，支持在大幅视角变化（包括镜头推拉）下进行高质量人体动画生成，同时引入了 MSTed 多机位 TED 视频数据集。

**[FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)**

:   FreeUV 提出了一种不需要 ground-truth UV 纹理数据的面部 UV 纹理恢复框架，通过分别训练关注真实外观的 UV-to-2D 网络和关注结构一致性的 2D-to-UV 网络，在推理时将两者的 UV 相关模块跨装配（Cross-Assembly）到预训练 Stable Diffusion 中，实现高保真的 UV-to-UV 纹理生成。

**[From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)**

:   LaDeCo 将平面设计的分层设计原则引入大型多模态模型（LMM），先用 GPT-4o 对多模态设计元素进行语义层规划，再按层逐步预测元素属性并渲染中间结果反馈给模型，将复杂的设计合成任务分解为可管理的子步骤，在设计合成质量上大幅超越基线方法。

**[From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)**

:   本文定义了文本到图表生成任务，构建了 DiagramGenBenchmark（涵盖 8 类图表），并提出多智能体框架 DiagramAgent（Plan + Code + Check + Diagram-to-Code），在图表生成、编码和编辑任务上显著超越现有文本到图像/代码方法。

**[GCC: Generative Color Constancy via Diffusing a Color Checker](gcc_generative_color_constancy_via_diffusing_a_color_checker.md)**

:   GCC 利用预训练扩散模型的图像先验，通过 inpainting 生成反映场景光照的色卡来估计光照颜色，借助 Laplacian 分解保留色卡结构的同时适应光照变化，在跨相机场景中展现出优越的泛化能力。

**[GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)**

:   本文提出GenDeg，一个基于Stable Diffusion的退化合成框架，能在任意干净图像上生成多种可控退化（雾/雨/雪/运动模糊/低光/雨滴），合成55万+图像构成GenDS数据集，训练在其上的All-In-One复原模型在域外测试集上获得显著性能提升。

**[Generation of Maximal Snake Polyominoes Using a Deep Neural Network](generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)**

:   将 DDPM 应用于生成最大蛇形多联骨牌，提出精简版 Structured Pixel Space Diffusion（SPS Diffusion），在训练到 14x14 正方网格的情况下泛化到 28x28 并生成有效蛇形，部分结果超越已知最大长度下界。

**[Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)**

:   LayerDecomp 提出了一个基于 Diffusion Transformer 的图像图层分解框架，将输入图像分解为干净的 RGB 背景层和带有透明视觉效果（阴影、反射）的 RGBA 前景层，通过一致性损失在无标注数据上也能学到正确的前景表示，大幅超越现有物体移除和空间编辑方法。

**[Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](generative_multimodal_pretraining_with_discrete_diffusion_timestep_tokens.md)**

:   DDT-LLaMA 提出用扩散时间步编码学习具有递归结构的离散视觉 token（DDT），使视觉 token 序列像自然语言一样具有层级依赖关系，从而在统一的 next-token-prediction 框架下同时实现多模态理解和生成的 SOTA 性能。

**[Generative Photomontage](generative_photomontage.md)**

:   提出 Generative Photomontage 框架，允许用户从多张 ControlNet 生成的图像中选取不同区域，通过扩散特征空间的图割分割和自注意力特征注入进行无缝合成，实现对生成图像的精细组合控制。

**[GIF: Generative Inspiration for Face Recognition at Scale](gif_generative_inspiration_for_face_recognition_at_scale.md)**

:   提出将人脸识别中的标量标签替换为结构化身份编码（整数序列），通过CLIP初始化+超球面均匀化生成编码向量，再用层次聚类构建树结构编码，将分类器计算复杂度从$\mathcal{O}(m)$降至$\mathcal{O}(\log m)$，同时解决了少数类坍缩问题。

**[GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)**

:   本文提出 GLASS，一种基于 Slot Attention 的物体中心学习方法，通过在扩散模型生成的图像空间中学习，利用语义引导模块（扩散模型的交叉注意力生成伪语义掩码）和实例引导模块（MLP 重建编码器特征）协同解决过分割和欠分割问题，在真实场景的物体发现和条件/组合生成任务上大幅超越前方法。

**[GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)**

:   提出GlyphMastero字形编码器，通过双流（局部字符级+全局文本行级）特征提取、跨层次注意力交互和多尺度FPN融合，为扩散模型提供笔画级精确的字形引导，在多语言场景文字编辑中句子准确率提升18.02%、FID降低53.28%。

**[Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)**

:   Goku 是字节跳动与港大提出的 rectified flow Transformer 系列模型（2B/8B），首次将 rectified flow 用于图像-视频联合生成，配合全面的数据管线和大规模训练基础设施优化，在 VBench（84.85）和 GenEval（0.76）等基准上达到 SOTA。

**[GPS as a Control Signal for Image Generation](gps_as_a_control_signal_for_image_generation.md)**

:   将照片 EXIF 元数据中的 GPS 坐标作为扩散模型的新型控制信号，训练 GPS+文本联合条件的图像生成模型，能捕捉城市内不同街区/地标的细粒度外观差异，并通过角度条件 SDS 从 2D 模型提取 3D 地标重建。

**[GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)**

:   提出 GraphGPT-o，将多模态属性图（MMAG，节点含图像+文本，边表示关系）的结构信息注入多模态大语言模型（MLLM），通过 PPR 采样、层次化 Q-Former 对齐器和灵活推理策略，实现基于图上下文的文本-图像联合生成。

**[Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)**

:   提出在扩散模型的特定时间步编辑learned score function来隐藏图像的方法，结合梯度感知参数选择和LoRA实现参数高效微调，在提取精度（52.90 dB PSNR）、模型保真度（FID变化仅0.02）和隐藏效率（0.04 GPU小时）三个维度上全面超越现有方法数个量级。

**[Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)**

:   HFD 提出在多尺度上用扩散模型显式去噪双向光流（而非在潜空间直接去噪），结合光流引导的编解码器图像合成器端到端联合训练，在精度上全面超越所有基线，同时推理速度比其他扩散方法快 10+ 倍。

**[HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)**

:   HMAR 将 VAR 的 next-scale 预测重构为 Markov 过程（仅依赖前一尺度的累积重建而非所有前序尺度），并在每个尺度内引入多步掩码生成来消除条件独立假设，配合自定义 IO-aware 块稀疏注意力核，在 ImageNet 上匹配或超越 VAR/DiT 质量的同时实现训练 2.5× 加速和推理 3× 内存缩减。

**[HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)**

:   HSI提出了一种基于全局风格统计特征和逐元素乘法的风格迁移模块，用线性复杂度替代自注意力的二次复杂度，同时通过双关系学习机制提升风格化质量，在效果和效率上均超越现有方法。

**[ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)**

:   提出 ICE 两阶段框架，仅用单个 T2I 扩散模型从单张图像自动定位物体级概念并分解为内在属性（类别、颜色、材质），实现无标注、无额外模型的层次化视觉概念提取。

**[IDEA-Bench: How Far are Generative Models from Professional Designing?](idea-bench_how_far_are_generative_models_from_professional_designing.md)**

:   提出首个面向专业级图像设计的综合基准 IDEA-Bench，涵盖 100 个真实设计任务（海报、绘本、字体、特效等）和 5 种输入输出模式，揭示当前最强模型仅获 22.48/100 分，距离专业设计仍有巨大鸿沟。

**[IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)**

:   IDProtector 提出首个前馈式对抗噪声编码器，通过单次前向传播为人脸照片添加不可感知的对抗扰动，可同时防御 InstantID、IP-Adapter、PhotoMaker 等多种编码器驱动的身份保持生成方法，且对 JPEG 压缩、缩放等变换保持鲁棒。

**[Image Generation Diversity Issues and How to Tame Them](image_generation_diversity_issues_and_how_to_tame_them.md)**

:   本文揭示了当前扩散模型存在严重的多样性不足问题（最先进模型仅覆盖训练数据 77% 的多样性），提出了基于图像检索的 Image Retrieval Score (IRS) 作为可解释的多样性度量指标，并引入 Diversity-Aware Diffusion Models (DiADM) 在不损失生成质量的前提下提升多样性。

**[Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)**

:   本文模仿真实动画制作流程，提出一种基于扩散模型的图像参考草图上色框架，通过分割交叉注意力（Split Cross-Attention）配合可切换LoRA机制分别处理前景和背景的上色，消除了空间纠缠伪影（spatial entanglement），在4.8M图像上训练后在定性、定量和用户研究中均优于现有方法。

**[Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)**

:   本文提出隐式偏见注入攻击框架（IBI-Attacks），通过在文本嵌入空间中预计算一个通用的偏见方向向量，再利用自适应特征选择模块根据不同用户输入动态调整该向量，以即插即用的方式将隐式偏见（如情绪、文化倾向）植入预训练的文生图扩散模型中，同时保持生成内容的原始语义，80%+的攻击成功率下仅35.8%被人类试验者察觉。

**[Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)**

:   本文提出解耦退火后验采样（DAPS），通过在扩散采样过程中解耦相邻步骤的样本依赖关系，允许大幅度的非局部跳跃来修正早期采样错误，在非线性逆问题（如相位恢复）上大幅超越现有方法。

**[Improving Editability in Image Generation with Layer-wise Memory](improving_editability_in_image_generation_with_layer-wise_memory.md)**

:   本文提出基于层级记忆的迭代图像编辑框架，通过存储每步编辑的 latent 和 prompt embedding，结合背景一致性引导（BCG）和多查询解耦注意力（MQD），实现多步顺序编辑中背景保持一致且新对象自然融入的效果。

**[InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)**

:   提出 InsightEdit，构建 250 万级高质量编辑数据集 AdvancedEdit，并设计双流桥接机制将 MLLM 的文本推理特征和图像语义特征同时注入扩散模型，在复杂指令跟随和背景一致性上达到 SOTA。

**[Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)**

:   提出 One Step Control Purification (OSCP) 框架，结合 Gaussian Adversarial Noise Distillation (GAND) 和 Controlled Adversarial Purification (CAP)，在单次 U-Net 推理（~0.1 秒）内完成对抗净化，相比传统扩散净化方法加速 100 倍。

**[InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)**

:   本文提出 InterAct 基准，整合并标准化了 21.81 小时的 3D 人物-物体交互数据（扩展到 30.70 小时），通过统一优化框架校正运动捕捉伪影并增强数据，定义六项生成任务和统一建模方法，在多个 HOI 生成任务上取得 SOTA 表现。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)**

:   提出 InterEdit，首个文本引导的多人 3D 运动交互编辑框架，通过 Semantic-Aware Plan Token Alignment 和 Interaction-Aware Frequency Token Alignment 在扩散模型中实现语义编辑的同时保持多人之间的时空耦合关系。

**[InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)**

:   InterMimic 提出了一个课程式教师-学生蒸馏框架，首次实现了单策略从大规模不完美 MoCap 数据中学习多样化的全身物理人物交互技能，通过教师策略先"完善"每个动作子集，再蒸馏到学生策略，并用 RL 微调超越简单模仿，最终支持零样本泛化和与运动生成器的无缝集成。

**[Interpretable Generative Models through Post-hoc Concept Bottlenecks](interpretable_generative_models_through_post-hoc_concept_bottlenecks.md)**

:   本文提出两种低成本的后置方法——概念瓶颈自编码器(CB-AE)和概念控制器(CC)——将预训练生成模型转化为可解释且可操控的模型，无需从头训练或真实标注数据，在 CelebA/CelebA-HQ/CUB 上的可操控性(steerability)平均超过先前 CBGM 方法约25%，训练速度快4-15倍。

**[IterIS: Iterative Inference-Solving Alignment for LoRA Merging](iteris_iterative_inference-solving_alignment_for_lora_merging.md)**

:   IterIS提出了一种迭代推理-求解的LoRA合并方法，通过直接提取统一适配器的输入特征（而非近似）来建立更准确的优化目标，配合正则化减少样本需求至先前方法的1-5%，并引入自适应权重平衡优化，在文本到图像扩散模型、视觉语言模型和大语言模型的LoRA合并中显著超越基线。

**[JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)**

:   提出 JanusFlow，将 rectified flow 直接嵌入自回归 LLM 框架，通过解耦理解/生成编码器 + 表征对齐正则化，在 1.3B 参数下同时达到多模态理解和图像生成的 SOTA。

**[K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)**

:   提出 K-LoRA，在每个 attention 层通过 Top-K 元素绝对值累加来比较主题 LoRA 和风格 LoRA 的重要性，自适应选择整层 LoRA 权重，配合时间步缩放因子，实现免训练的主题-风格高质量融合。

**[Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)**

:   TexTok 提出在图像分词（tokenization）阶段引入文本描述作为条件，将高层语义信息卸载给文本，使图像 token 专注于编码细粒度视觉细节，从而在保持甚至提升重建质量的同时实现更高的压缩率，在 ImageNet 上取得了 SOTA 的生成 FID 分数 1.46。

**[Latent Space Imaging](latent_space_imaging.md)**

:   Latent Space Imaging (LSI) 提出了一种将光学编码与生成模型解码结合的新成像范式，通过将图像信息直接编码到 StyleGAN 的语义隐空间中，实现 1:100 到 1:16384 的极端压缩比，同时仍能完成人脸重建、属性分类、分割和关键点检测等下游任务。

**[LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)**

:   LaTexBlend 通过在文本编码器后的潜在文本空间（Latent Textual Space）中表示和融合多个定制概念，实现了高保真、高效率的多概念定制图像生成，微调复杂度线性增长且推理无额外开销。

**[LaVin-DiT: Large Vision Diffusion Transformer](lavin-dit_large_vision_diffusion_transformer.md)**

:   LaVin-DiT 提出一种基于扩散 Transformer 的大视觉基础模型，通过空间-时序 VAE 编码、联合扩散 Transformer 去噪、以及 in-context learning 实现超过 20 种视觉任务的统一处理，从 0.1B 扩展至 3.4B 参数，在多项任务上显著超越自回归式大视觉模型 LVM。

**[Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)**

:   提出 Leffa（Learning Flow Fields in Attention），在扩散模型的注意力层中将 attention map 转换为流场并进行像素级正则化监督，显式引导 target query 关注正确的 reference key 区域，**零额外推理开销**地减少细粒度细节（纹理、文字、logo）失真，在虚拟试衣（VITON-HD、DressCode）和姿态迁移（DeepFashion）上均 SOTA。

**[Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)**

:   提出PAG（Prompt Adaptation with GFlowNets），将提示词适配重新定义为概率推断问题，利用GFlowNets从奖励分布中采样而非最大化奖励，结合流重激活、奖励优先采样和奖励分解三大技术解决模式坍塌问题，生成既高质量又多样化的文本到图像提示词。

**[Learning Visual Generative Priors without Text](learning_visual_generative_priors_without_text.md)**

:   提出Lumos框架，通过纯视觉的图像到图像（I2I）自监督预训练学习视觉生成先验，然后仅用1/10的文本-图像对微调即可达到甚至超越现有T2I模型的效果，并在文本无关的视觉任务（I2V、NVS）上展现出优于T2I先验的性能。

**[LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)**

:   提出LEDiff，通过在预训练扩散模型的潜空间中进行曝光融合（而非图像空间），用少量HDR数据微调VAE解码器和去噪器，让现有生成模型具备HDR生成能力，同时实现SOTA级别的LDR到HDR转换。

**[Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)**

:   MVLift提出了一个多阶段框架，仅使用单视角2D姿态序列训练，通过线条件扩散模型→多视角优化→合成数据生成→多视角扩散模型的渐进策略建立多视角一致性，实现无需3D监督的全局3D运动（含关节旋转+根轨迹）估计，在AIST++上根轨迹误差67.6mm超越需要3D监督的WHAM (164.3mm)。

**[LoRACLR: Contrastive Adaptation for Customization of Diffusion Models](loraclr_contrastive_adaptation_for_customization_of_diffusion_models.md)**

:   LoRACLR 提出一种基于对比学习目标的 LoRA 模型合并方法，通过学习一个 delta 权重将多个独立训练的单概念 LoRA 模型融合为一个统一模型，无需重训练或访问原始训练数据，即可实现高保真的多概念图像生成，合并 12 个概念仅需 5 分钟。

**[lbGen: Low-Biased General Annotated Dataset Generation](low-biased_general_annotated_dataset_generation.md)**

:   提出 lbGen 框架，通过双层语义对齐（全局对抗+个体余弦相似度）和质量保证损失微调 Stable Diffusion，仅用类别名称即可生成低偏差的通用标注数据集，预训练骨干比 ImageNet 真实数据平均迁移精度高出 1.7%~2.1%。

**[LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)**

:   提出 LumiNet，将源图像的潜在内在特征（128 维 albedo-like 表征）和目标图像的潜在外在光照码（16 维）注入改造后的 ControlNet，实现仅用图像输入的室内场景级光照迁移，包含镜面高光、阴影和间接照明等复杂效果。

**[MagicQuill: An Intelligent Interactive Image Editing System](magicquill_an_intelligent_interactive_image_editing_system.md)**

:   提出 MagicQuill 智能交互式图像编辑系统，用三种笔触（添加/减去/颜色）表达编辑意图，双分支扩散插件（inpainting + control）实现边缘和颜色的精细控制，MLLM 实时猜测意图自动生成 prompt，形成无需手动输入文字的连续编辑工作流。

**[MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)**

:   MangaNinja 是一个基于扩散模型的参考图引导线稿上色方法，通过渐进式 Patch Shuffling 策略训练模型学会局部语义匹配能力，并引入 PointNet 驱动的点控制机制实现精细颜色对应，在大姿态差异、多参考图、跨角色上色等挑战场景中显著超越现有方法。

**[MARBLE: Material Recomposition and Blending in CLIP-Space](marble_material_recomposition_and_blending_in_clip-space.md)**

:   仅在 CLIP 空间操作材质嵌入，通过定向注入 UNet 中的材质响应层实现材质迁移和混合，并通过轻量 MLP 预测属性编辑方向实现粗糙度/金属度/透明度/发光的参数化控制，无需微调扩散模型。

**[Memories of Forgotten Concepts](memories_of_forgotten_concepts.md)**

:   本文揭示了扩散模型中概念擦除方法的根本缺陷——通过扩散反演找到高似然度的潜变量种子，证明被擦除的概念信息仍然存留在模型中，且可以从多个不同的种子向量重建出被擦除概念的高质量图像。

**[MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis](metashadow_object-centered_shadow_detection_removal_and_synthesis.md)**

:   MetaShadow 提出首个三合一框架，将基于GAN的 Shadow Analyzer（阴影检测+去除）与基于扩散模型的 Shadow Synthesizer（阴影合成）协同结合，通过 GAN 中间特征引导扩散模型进行阴影知识迁移，在三个阴影任务上均达到 SOTA。

**[MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification](mexd_an_expert-infused_diffusion_model_for_whole-slide_image_classification.md)**

:   MExD 首次将生成式扩散模型应用于全切片图像（WSI）分类，通过动态混合专家（Dyn-MoE）聚合器筛选关键实例并提供条件信息，结合扩散分类器（Diff-C）从噪声中迭代还原类别标签，在Camelyon16、TCGA-NSCLC和BRACS三个基准上达到SOTA。

**[MINIMA: Modality Invariant Image Matching](minima_modality_invariant_image_matching.md)**

:   MINIMA 提出了一个统一的跨模态图像匹配框架，通过设计数据引擎从廉价的 RGB 图像对中生成多模态合成数据集 MD-syn（480M 对），使任何现有匹配管线仅需微调即可获得跨模态匹配能力，在 19 种跨模态场景下显著超越模态特定方法。

**[Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)**

:   MinorityPrompt 提出了一种在线 prompt 优化框架，通过在推理过程中迭代优化可学习 token embedding 来最大化似然度损失，引导 T2I 扩散模型生成处于数据分布低密度区域的少数(minority)样本，同时保持语义一致性和生成质量。

**[MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](mirrorverse_pushing_diffusion_models_to_realistically_reflect_the_world.md)**

:   MirrorVerse 通过构建增强的合成数据集 SynMirrorV2（包含随机位姿、旋转和多物体场景），配合三阶段课程式训练策略，训练出 MirrorFusion 2.0 模型，首次使扩散模型能够生成逼真的镜面反射，在合成和真实场景中均显著超越前方法。

**[MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)**

:   提出 MixerMDM，首个可学习的运动扩散模型组合技术，通过 Transformer-based Mixer 模块预测动态混合权重，以对抗训练方式学习如何融合个体运动和交互运动扩散模型，实现细粒度可控的人-人交互运动生成。

**[MLLM-as-a-Judge for Image Safety without Human Labeling](mllm-as-a-judge_for_image_safety_without_human_labeling.md)**

:   提出 CLUE 框架，通过规则客观化、CLIP 相关性扫描、前置条件链分解和去偏 token 概率分析，实现无需人工标注的零样本图像安全判定，在多个 MLLM 上大幅超越基线。

**[MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)**

:   首次将连续图像表示与离散文本表示整合到统一自回归概率建模框架中，通过轻量扩散头替代 VQ 离散化避免信息损失，并推导出 v-prediction 为最优参数化以解决低精度训练下的数值误差问题。

**[MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices](mobileportrait_real-time_one-shot_neural_head_avatars_on_mobile_devices.md)**

:   提出首个可在移动端实时运行的单张人脸头像动画方法 MobilePortrait，通过混合显隐式关键点 + 预计算外观知识，仅用 16 GFLOPs 即匹敌 SOTA（100–600+ GFLOPs）的效果。

**[Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion](mono2stereo_a_benchmark_and_empirical_study_for_stereo_conversion.md)**

:   构建首个大规模立体转换基准 Mono2Stereo（240 万对），提出立体质量指标 SIoU（与人类判断相关性 0.84 Spearman）和双条件扩散模型 + Edge Consistency 损失，同时解决单阶段方法立体效果弱和两阶段方法图像质量差的矛盾。

**[Move-in-2D: 2D-Conditioned Human Motion Generation](move-in-2d_2d-conditioned_human_motion_generation.md)**

:   定义 2D 场景图像+文本条件下的人体运动生成新任务，构建 30 万级 HiC-Motion 数据集，通过 in-context conditioning 扩散 Transformer 生成可自然投影到场景的运动序列，赋能下游人体视频生成。

**[MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)**

:   MTADiffusion通过构建500万张图像的Mask-Text对齐数据集、联合训练修复与边缘预测任务、以及基于VGG Gram矩阵的风格一致性损失，同时解决了对象修复中的语义错位、结构扭曲和风格不一致三大问题，在BrushBench和EditBench上达到SOTA。

**[Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)**

:   MCLD通过将源人物图像解耦为面部区域、外观纹理和整体图像三个焦点条件，设计多焦点条件聚合模块(MFCA)在UNet不同阶段选择性注入不同条件，有效缓解了LDM压缩导致的面部和纹理细节退化问题，在DeepFashion上取得SOTA。

**[Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)**

:   提出 MCA-Ctrl，一种无需微调的图像定制方法，通过三个并行扩散过程的自注意力协同控制，实现文本和图像条件下的高质量主体驱动编辑与生成，同时引入主体定位模块解决复杂视觉场景中的特征泄漏和混淆问题。

**[Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)**

:   本文提出首个支持文本和布局引导的多目标同时合成（compositing）生成模型Multitwine，通过联合训练合成与个性化生成任务，结合跨注意力/自注意力解耦损失，实现同时插入多个对象的自然交互（如拥抱、弹吉他），用户研究中交互真实性偏好率最高达97.1%。

**[MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)**

:   本文提出MVPortrait，一个两阶段文本引导框架（Text2FLAME + FLAME2Video），通过将FLAME 3D参数化面部模型作为中间表示，分别用MotionDM和EmotionDM扩散模型生成运动和表情参数序列，再用多视角视频生成模型将FLAME渲染序列转化为逼真的多视角肖像动画，首次实现文本/语音/视频三种信号兼容的可控肖像动画。

**[Navigating Image Restoration with VAR's Distribution Alignment Prior](navigating_image_restoration_with_vars_distribution_alignment_prior.md)**

:   本文发现Visual AutoRegressive (VAR) 模型的next-scale预测具有天然的多尺度分布对齐能力——低尺度修复全局退化（如低光照、雾霾），高尺度修复局部退化（如噪声、雨滴），基于此构建VarFormer框架，通过Degradation-Aware Enhancement (DAE)自适应选择尺度先验、Adaptive Feature Transformation (AFT)融合先验与退化特征，在6类恢复任务上超越现有multi-task方法。

**[Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)**

:   本文提出FastProtect，首个关注延迟的图像保护框架，通过预训练Mixture-of-Perturbations (MoP)替代传统逐图迭代优化，配合Multi-Layer Protection Loss增强训练效果、Adaptive Targeted Protection和Adaptive Protection Strength优化推理，实现了比现有最快方法PhotoGuard快175×（A100 GPU上0.04秒 vs 7秒处理512²图像）的实时保护，同时保持相当的保护效力和更优的不可见性。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   揭示 DiT 的计算在空间 token 上均匀分配（不会把多余计算重分配到困难区域），提出 ELIT——在 DiT 中插入可变长度的 latent interface（Read/Write 交叉注意力），训练时随机丢弃尾部 latent 学出重要性排序，推理时通过调节 latent 数量实现平滑的质量-FLOPs 权衡，ImageNet 512px 上 FID 降低 53%。

**[Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)**

:   提出 Concept-Gated Visual Distillation (CGVD)，一种无需训练的推理时框架，通过语言指令解析 → SAM3 分割 → 集合论交叉验证 → LaMa 修复的流水线，从 VLA 模型的视觉输入中选择性移除语义干扰物，在高度杂乱场景中将 π₀ 的操作成功率从 43.0% 提升至 77.5%。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 去噪器（Auto-Correction + Directional Correction + Score-Based Denoising 三阶段），解决将 score-based 扩散先验嵌入 ADMM-PnP 框架时的流形不匹配问题，并首次建立了 score-based 去噪器在 ADMM 中的收敛性理论保证，在去噪、修复、去模糊、超分辨、相位恢复、HDR 等逆问题上一致超越现有基线。

**[Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)**

:   提出 FIRM 框架——通过"差异优先"（编辑）和"计划-打分"（生成）的数据构建流水线训练专用奖励模型（FIRM-Edit-8B / FIRM-Gen-8B），配合"Base-and-Bonus"奖励策略（CME/QMA）解决 RL 中的奖励 hacking 问题，在图像编辑和 T2I 生成任务上均取得 SOTA。

**[UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations](unicom_unified_multimodal_modeling_via_compressed_continuous_semantic_representa.md)**

:   提出 UniCom，通过对 VLM 连续语义特征进行**通道维度压缩**（而非空间下采样），构建紧凑连续表示空间，用 Transfusion 架构统一多模态理解与生成，在统一模型中达到 SOTA 生成质量。

**[Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)**

:   提出U2Diff，一个统一的扩散模型框架，能同时处理多智能体轨迹补全和预测任务，通过增强去噪损失提供逐状态不确定性估计，并引入Rank Neural Network对生成的多模态预测进行误差概率排序。

**[V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像复原重新定义为**渐进式视频生成过程**，利用预训练视频生成模型（Wan2.2-TI2V-5B）的先验知识，仅用 1,000 个多任务训练样本（不到现有方法的 2%）即可实现竞争力的多任务图像复原。

**[Visual-ERM: Reward Modeling for Visual Equivalence](visual-erm_reward_modeling_for_visual_equivalence.md)**

:   提出 Visual-ERM，一个多模态生成式奖励模型，在视觉空间中直接评估 vision-to-code 任务的渲染质量，提供细粒度、可解释、任务无关的奖励信号，用于 RL 训练和测试时缩放。
