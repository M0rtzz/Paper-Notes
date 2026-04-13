---
title: >-
  CVPR2025 3D视觉方向 36篇论文解读
description: >-
  36篇CVPR2025 3D视觉方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D视觉

**📷 CVPR2025** · 共 **36** 篇

**[3D-Grand A Million-Scale Dataset For 3D-Llms With Better Grounding And Less Hall](3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[3D-Gsw 3D Gaussian Splatting For Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)**

:   提出3D-GSW，首个专为3D Gaussian Splatting设计的鲁棒数字水印方法，通过频率引导致密化（FGD）移除冗余高斯并在高频区域分裂高斯来增强鲁棒性，结合梯度掩码和小波子带损失保持渲染质量，在Blender/LLFF/Mip-NeRF 360数据集上同时实现了最优的水印鲁棒性和渲染质量。

**[3D-Hgs 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)**

:   提出3D Half-Gaussian (3D-HGS)核函数——用一个分割平面将3D高斯分成两半，每半有独立不透明度，作为**即插即用**的重建核替换标准高斯核，在不牺牲渲染速度的前提下显著提升形状和颜色不连续处的渲染质量，在Mip-NeRF360/T&T/Deep Blending上全面超越所有SOTA方法。

**[3D-Llava Towards Generalist 3D Lmms With Omni Superpoint Transformer](3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)**

:   提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

**[3D-Mem 3D Scene Memory For Embodied Exploration And Reasoning](3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)**

:   提出3D-Mem——基于"记忆快照"的3D场景记忆框架，用少量精选多视角图像紧凑表示已探索区域，结合Frontier Snapshot表示未探索区域，配合VLM实现高效的具身探索与推理。

**[3D-Slnr A Super Lightweight Neural Representation For Large-Scale 3D Mapping](3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)**

:   提出3D-SLNR，一种超轻量神经3D表示——基于锚定在点云支撑点上的带限局部SDF集合定义全局SDF，每个局部SDF仅由一个微型MLP参数化（无隐特征），通过可学习的位置/旋转/缩放适应复杂几何，配合并行查找算法和剪枝-扩展策略，以不到先前方法1/5的内存实现SOTA重建质量。

**[3D Convex Splatting Radiance Field Rendering With 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)**

:   用3D光滑凸体（Smooth Convex）替代高斯基元进行辐射场渲染，通过点集定义凸包+LogSumExp平滑化+自定义CUDA光栅化器，在T&T和Deep Blending上超越3DGS，且所需基元更少。

**[3D Dental Model Segmentation With Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)**

:   提出 CrossTooth，通过基于曲率先验的选择性下采样（边界区域顶点密度提升 10-15%）和多视角渲染图像的跨模态边界特征融合，在 3DTeethSeg'22 公开数据集上实现 95.86% mIoU 和 82.05% boundary IoU，分别比之前 SOTA（ToothGroupNet）提升 2.3% 和 5.7%。

**[3D Gaussian Head Avatars With Expressive Dynamic Appearances By Compact Tensoria](3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)**

:   提出一种紧凑张量表示的3D高斯头部头像方法——用三平面存储中性表情的静态外观，用轻量1D特征线存储每个blendshape的动态纹理（不透明度偏移），仅需**10MB存储**即可实现300FPS实时渲染和准确的动态面部细节捕捉，在Nersemble数据集上PSNR和存储效率全面超越GA、GBS和GHA。

**[3D Gaussian Inpainting With Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)**

:   提出3DGIC，通过**深度引导的跨视角一致修复**框架实现3D高斯场景中的物体移除与修补——利用渲染深度图从其他视角发现被掩码区域中的可见背景像素来精化修补掩码，再用参考视角的2D修补结果通过3D投影约束其他视角的一致性，在SPIn-NeRF数据集上FID和LPIPS全面超越现有方法。

**[3D Student Splatting And Scooping](3d_student_splatting_and_scooping.md)**

:   提出SSS（Student Splatting and Scooping），用前所未有的三重创新改进3DGS范式：(1) 用**Student-t分布**替代高斯分布作为混合组件（可学习的尾部厚度，从Cauchy到Gaussian连续变化）；(2) 引入**负密度组件**（scooping减去颜色）扩展到非单调混合模型；(3) 用**SGHMC采样**替代SGD解耦参数优化，在Mip-NeRF360/T&T/Deep Blending上6/9指标取得最优，且参数效率极高——用**最少18%**的组件数即可匹配或超越3DGS。

**[3Denhancer Consistent Multi-View Diffusion For 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)**

:   提出一个基于多视图潜在扩散模型的3D增强框架，通过姿态感知编码器、多视图行注意力和近视图极线聚合模块，在保持跨视图一致性的前提下显著提升低质量3D生成结果的纹理质量。

**[3Dgut Enabling Distorted Cameras And Secondary Rays In Gaussian Splatting](3dgut_enabling_distorted_cameras_and_secondary_rays_in_gaussian_splatting.md)**

**[4Deform Neural Surface Deformation For Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)**

**[4Dequine Disentangling Motion And Appearance For 4D Equine Reconstruction From M](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

**[4Dtam Non-Rigid Tracking And Mapping Via Dynamic Surface Gaussians](4dtam_non-rigid_tracking_and_mapping_via_dynamic_surface_gaussians.md)**

:   本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。

**[A Lightweight Udf Learning Framework For 3D Reconstruction Based On Local Shape ](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)**

:   本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。

**[A Unified Image-Dense Annotation Generation Model For Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)**

:   本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。

**[Activegamer Active Gaussian Mapping Through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)**

:   提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。

**[Aerialmegadepth Learning Aerial-Ground Reconstruction And View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)**

:   本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。

**[Framevggt Frame Evidence Rolling Memory For Streaming Vggt](framevggt_frame_evidence_rolling_memory_for_streaming_vggt.md)**

:   提出 FrameVGGT，将流式 VGGT 的 KV 缓存从 token 级保留重组为帧级证据块保留，通过中期记忆库+稀疏锚点的双层有界内存结构，在固定内存预算下保持更连贯的几何支撑，实现长序列3D重建/深度/位姿估计的精度-内存最优权衡。

**[Hoi3Dgen Generating High-Quality Human-Object-Interactions In 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)**

:   提出 HOI3DGen 框架，通过MLLM自动标注高质量交互数据 + 视角条件化微调扩散模型 + 3D提升与SMPL配准，首次实现从文本精确控制接触语义的高质量3D人物交互生成，在文本一致性上超越基线4-15倍。

**[Hybrid Etfce-Grf Exact Cluster-Size Retrieval With Analytical P-Values For Voxel](hybrid_etfce-grf_exact_cluster-size_retrieval_with_analytical_p-values_for_voxel.md)**

:   将 eTFCE 的并查集精确聚类大小查询与 pTFCE 的解析 GRF p 值推断结合，首次在单一框架中实现精确聚类检索+无需置换检验的统计推断，速度比置换 TFCE 快 1300 倍，在全脑体素形态测量中保持严格 FWER 控制。

**[Instanthdr Single-Forward Gaussian Splatting For High Dynamic Range 3D Reconstru](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)**

:   提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模进行多曝光融合 + MetaNet 预测场景自适应色调映射器，从未标定多曝光 LDR 图像一次前向推理重建 HDR 3D 高斯，速度比优化方法快 ~700 倍。

**[Jopp-3D Joint Open Vocabulary Semantic Segmentation On Point Clouds And Panorama](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)**

:   提出 JOPP-3D 框架，通过将全景图切线分解为透视图像、利用 SAM+CLIP 进行3D实例-语义对齐，首次实现对3D点云和全景图像的联合开放词汇语义分割，在 Stanford-2D-3D-s 和 ToF-360 数据集上超越现有方法。

**[Mobile-Gs Real-Time Gaussian Splatting For Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)**

:   提出 Mobile-GS，通过深度感知的无序渲染（消除排序瓶颈）+ 神经视角依赖增强 + 一阶SH蒸馏 + 神经向量量化 + 贡献度剪枝，首次在 Snapdragon 8 Gen 3 手机 GPU 上实现 116 FPS 实时高斯溅射渲染，存储仅 4.6MB 且视觉质量与原始 3DGS 相当。

**[Motionanymesh Physics-Grounded Articulation For Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出 MotionAnyMesh，一种零样本框架，通过 SP4D 运动学先验引导 VLM 推理消除幻觉 + 物理约束轨迹优化保证无碰撞，将静态3D网格自动转化为仿真可用的铰接数字孪生，物理可执行率达 87%，是现有最好方法的近两倍。

**[Node-Rf Learning Generalized Continuous Space-Time Scene Dynamics With Neural Od](node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   提出 Node-RF，将 Neural ODE 与动态 NeRF 紧密耦合，用潜在向量的 ODE 演化建模场景连续时间动力学，实现超出训练序列的长程时序外推和跨轨迹泛化，无需光流或深度监督。

**[P-Slcr Unsupervised Point Cloud Semantic Segmentation Via Prototypes Structure L](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)**

:   提出 P-SLCR，一种原型库驱动的无监督点云语义分割方法，通过将点分离为"一致"和"模糊"两类，用一致结构学习对齐一致点与原型 + 语义关系一致性推理约束两个原型库，在 S3DIS 上无监督达 47.1% mIoU，超越全监督 PointNet。

**[Pano360 Perspective To Panoramic Vision With Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，首个在3D摄影测量空间进行全景拼接的 Transformer 框架，利用预训练 VGGT 骨干获取3D感知的多视角特征对齐 + 多特征联合优化接缝检测，支持2到数百张输入图像，在弱纹理/大视差/重复模式场景下成功率达97.8%。

**[Regularizing Inr With Diffusion Prior Self-Supervised 3D Reconstruction Of Neutr](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，将隐式神经表示 (INR/SIREN) 与预训练扩散模型先验结合，通过 proximal loss 在每个 DDIM 时间步用扩散去噪输出正则化 INR 重建，在稀疏视角中子 CT（低至 4-5 个视角）上超越 FBP、纯 INR、DD3IP 和经典 MBIR(qGGMRF) 方法。

**[Rewis3D Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)**

:   Rewis3d 利用前馈 3D 重建（MapAnything）从 2D 视频中获取 3D 点云作为辅助监督信号，通过双 Student-Teacher 架构和加权跨模态一致性 (CMC) 损失，在仅使用稀疏标注（点/涂鸦/粗标记）的情况下将弱监督 2D 语义分割性能提升 2-7% mIoU，推理时仍为纯 2D。

**[Scope Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一个即插即用的背景引导原型富化框架，在基类训练后用类无关分割模型从背景区域挖掘伪实例建立 Instance Prototype Bank (IPB)，当新类别以少样本方式出现时，通过 Contextual Prototype Retrieval (CPR) 和 Attention-Based Prototype Enrichment (APE) 融合背景原型与少样本原型，在 ScanNet/S3DIS 上新类 IoU 提升最高 6.98%。

**[Spectral Defense Against Resource-Targeting Attack In 3D Gaussian Splatting](spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   针对 3DGS 的资源瞄准攻击（通过投毒训练图像触发高斯过度增长导致资源耗尽），提出频域防御：3D 频率滤波器通过将高斯协方差与频谱响应关联实现频率感知剪枝，2D 频谱正则化通过熵惩罚渲染图像的角向能量各向异性来抑制攻击噪声，实现高斯数量压缩 5.92×、内存减少 3.66×、速度提升 4.34×。

**[Towards Spatio-Temporal World Scene Graph Generation From Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[Varsplat Uncertainty-Aware 3D Gaussian Splatting For Robust Rgb-D Slam](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。
