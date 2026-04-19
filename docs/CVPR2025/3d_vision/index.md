---
title: >-
  CVPR2025 3D视觉方向 211篇论文解读
description: >-
  211篇CVPR2025 3D视觉方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D视觉

**📷 CVPR2025** · **211** 篇论文解读

**[3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)**

:   提出3D-GSW，首个专为3D Gaussian Splatting设计的鲁棒数字水印方法，通过频率引导致密化（FGD）移除冗余高斯并在高频区域分裂高斯来增强鲁棒性，结合梯度掩码和小波子带损失保持渲染质量，在Blender/LLFF/Mip-NeRF 360数据集上同时实现了最优的水印鲁棒性和渲染质量。

**[3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)**

:   提出3D Half-Gaussian (3D-HGS)核函数——用一个分割平面将3D高斯分成两半，每半有独立不透明度，作为**即插即用**的重建核替换标准高斯核，在不牺牲渲染速度的前提下显著提升形状和颜色不连续处的渲染质量，在Mip-NeRF360/T&T/Deep Blending上全面超越所有SOTA方法。

**[3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)**

:   提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

**[3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)**

:   提出3D-Mem——基于"记忆快照"的3D场景记忆框架，用少量精选多视角图像紧凑表示已探索区域，结合Frontier Snapshot表示未探索区域，配合VLM实现高效的具身探索与推理。

**[3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)**

:   提出3D-SLNR，一种超轻量神经3D表示——基于锚定在点云支撑点上的带限局部SDF集合定义全局SDF，每个局部SDF仅由一个微型MLP参数化（无隐特征），通过可学习的位置/旋转/缩放适应复杂几何，配合并行查找算法和剪枝-扩展策略，以不到先前方法1/5的内存实现SOTA重建质量。

**[3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)**

:   用3D光滑凸体（Smooth Convex）替代高斯基元进行辐射场渲染，通过点集定义凸包+LogSumExp平滑化+自定义CUDA光栅化器，在T&T和Deep Blending上超越3DGS，且所需基元更少。

**[3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)**

:   提出 CrossTooth，通过基于曲率先验的选择性下采样（边界区域顶点密度提升 10-15%）和多视角渲染图像的跨模态边界特征融合，在 3DTeethSeg'22 公开数据集上实现 95.86% mIoU 和 82.05% boundary IoU，分别比之前 SOTA（ToothGroupNet）提升 2.3% 和 5.7%。

**[3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations](3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)**

:   提出一种紧凑张量表示的3D高斯头部头像方法——用三平面存储中性表情的静态外观，用轻量1D特征线存储每个blendshape的动态纹理（不透明度偏移），仅需**10MB存储**即可实现300FPS实时渲染和准确的动态面部细节捕捉，在Nersemble数据集上PSNR和存储效率全面超越GA、GBS和GHA。

**[3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)**

:   提出3DGIC，通过**深度引导的跨视角一致修复**框架实现3D高斯场景中的物体移除与修补——利用渲染深度图从其他视角发现被掩码区域中的可见背景像素来精化修补掩码，再用参考视角的2D修补结果通过3D投影约束其他视角的一致性，在SPIn-NeRF数据集上FID和LPIPS全面超越现有方法。

**[3D Student Splatting and Scooping (SSS)](3d_student_splatting_and_scooping.md)**

:   提出SSS（Student Splatting and Scooping），用前所未有的三重创新改进3DGS范式：(1) 用**Student-t分布**替代高斯分布作为混合组件（可学习的尾部厚度，从Cauchy到Gaussian连续变化）；(2) 引入**负密度组件**（scooping减去颜色）扩展到非单调混合模型；(3) 用**SGHMC采样**替代SGD解耦参数优化，在Mip-NeRF360/T&T/Deep Blending上6/9指标取得最优，且参数效率极高——用**最少18%**的组件数即可匹配或超越3DGS。

**[3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)**

:   提出一个基于多视图潜在扩散模型的3D增强框架，通过姿态感知编码器、多视图行注意力和近视图极线聚合模块，在保持跨视图一致性的前提下显著提升低质量3D生成结果的纹理质量。

**[3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](3dgut_enabling_distorted_cameras_and_secondary_rays_in_gaussian_splatting.md)**

**[4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)**

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

**[4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians](4dtam_non-rigid_tracking_and_mapping_via_dynamic_surface_gaussians.md)**

:   本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。

**[A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)**

:   本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。

**[A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)**

:   本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。

**[ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)**

:   提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。

**[AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)**

:   本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。

**[AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)**

:   从单张图像生成可动画 3D 人体——先用适配的 CogVideo 生成多视角标准姿态图像（含法线），再将多视角不一致性建模为 4DGS 中的时序变化来提取一致的 canonical 空间高斯模型，最后通过 SMPL-X 蒙皮驱动动画。

**[Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)**

:   提出Any3DIS，通过3D感知的2D掩码跟踪（利用SAM-2追踪每个超点在多帧中的2D分割）替代传统的无监督合并策略，并用动态规划优化3D Proposal，在ScanNet200和ScanNet++上的类别无关、开放词汇、开放式3D实例分割任务中均取得SOTA。

**[ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)**

:   提出ARM框架，将几何和外观生成解耦，在UV纹理空间中通过反投影和全局感受野网络重建高质量纹理，并引入材质先验解决稀疏视角下材质与光照的歧义性，仅用8张H100训练即在GSO和OmniObject3D上超越现有方法。

**[BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)**

:   从错误分析角度重新审视3D语义分割，将分割误差分为四类（区域分类/位移/合并/误响应）并设计对应评估指标，提出BFANet通过边界-语义解耦模块和实时边界伪标签计算增强边界感知，在ScanNet200测试集上达到36.0 mIoU（不含辅助数据训练的最高成绩）。

**[Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)**

:   提出一种基于新型图像块表示 Blurry-Edges 的深度估计方法，通过对散焦边界的平滑度建模，实现在极低光照（光子受限）条件下从一对不同散焦图像中鲁棒地估计物体深度，噪声鲁棒性比现有 DfD 方法高 4 倍以上。

**[CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)**

:   提出 CADDreamer，通过语义增强的多视图扩散模型和几何拓扑提取模块，从单张RGB图像直接生成具有紧凑B-rep表示、清晰结构和锐利边缘的CAD模型，支持平面、圆柱、圆锥、球体、环面五种基元类型。

**[Category-Agnostic Neural Object Rigging](category-agnostic_neural_object_rigging.md)**

:   提出 CANOR（Category-Agnostic Neural Object Rigging），通过将可变形4D物体编码为稀疏的空间定位 blob 集合和实例感知特征体，以完全类别无关、数据驱动的方式自动发现可变形物体的低维姿态空间，实现直观的姿态操控。

**[CMMLoc: Advancing Text-to-PointCloud Localization with Cauchy-Mixture-Model Based Framework](cmmloc_advancing_text-to-pointcloud_localization_with_cauchy-mixture-model_based.md)**

:   提出 CMMLoc，一个基于柯西混合模型（CMM）的不确定性感知文本-点云定位框架，通过将粗检索阶段建模为部分相关检索问题并引入 CMM Transformer 和方位整合模块，在 KITTI360Pose 数据集上实现 SOTA 性能。

**[COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)**

:   提出 COB-GS，一种通过语义梯度统计驱动的边界自适应高斯分裂技术，联合优化语义信息和视觉纹理，解决 3DGS 分割中物体边界模糊的问题，在保持视觉质量的同时实现清晰的物体边界分割。

**[CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)**

:   提出CoCoGaussian，利用物理摄影散焦原理（弥散圆/Circle of Confusion）在3D高斯溅射框架中建模散焦模糊，仅使用散焦图像即可精确重建3D场景并渲染清晰的新视角图像。

**[Coherent 3D Portrait Video Reconstruction via Triplane Fusion](coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)**

:   提出一种基于三平面融合（Triplane Fusion）的方法，将个人化3D先验与逐帧观测融合，在单目RGB视频中同时实现时间一致性和动态外观的忠实重建，用于3D远程呈现。

**[ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)**

:   提出ColabSfM范式——通过3D点云配准（而非视觉描述子匹配）来融合分布式SfM重建结果，并构建了专用的SfM配准数据集生成管线和改进的配准模型RefineRoITr。

**[CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)**

:   提出CoMapGS，利用像素级共视性图（covisibility map）来指导稀疏视角3DGS中初始点云增强和自适应加权监督，首次显式关注并恢复高不确定性的单视角区域。

**[CoMatcher: Multi-View Collaborative Feature Matching](comatcher_multi-view_collaborative_feature_matching.md)**

:   提出CoMatcher，一种多视角协同特征匹配器，从两视角独立匹配范式转向1-to-N协同匹配范式，利用互补视角的上下文线索和跨视角投影一致性约束来提升复杂场景下的匹配可靠性。

**[Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)**

:   提出 Compass Control，通过引入轻量级方向编码器预测 compass token 并结合耦合注意力定位（CALL）机制，实现文本到图像扩散模型中多物体的精确3D方向控制，仅需合成数据训练即可泛化到未见类别和多物体场景。

**[Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)**

:   首次提出面向迭代式立体匹配的一致性感知自训练框架（CST-Stereo），通过多分辨率预测一致性滤波和迭代预测一致性滤波评估伪标签可靠性，结合软加权损失有效利用无标签真实数据提升模型性能和泛化能力。

**[Continuous 3D Perception Model with Persistent State](continuous_3d_perception_model_with_persistent_state.md)**

:   提出CUT3R（Continuous Updating Transformer for 3D Reconstruction），一个维持持续内部状态的循环模型，能从图像流中在线、增量地进行度量级3D重建、相机位姿估计，并能推断未观测区域的3D结构。

**[Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)**

:   揭示跨视图补全（CVC）模型中交叉注意力图（cross-attention map）本质上学到了精确的稠密对应关系，提出ZeroCo在零样本匹配和学习型几何匹配中利用这一发现，显著超越基于编码器/解码器特征的常规用法。

**[CrossOver: 3D Scene Cross-Modal Alignment](crossover_3d_scene_cross-modal_alignment.md)**

:   提出CrossOver框架，通过维度特定编码器和三阶段训练管线，在不要求完整模态配对的条件下，学习RGB图像、点云、CAD模型、平面图和文本描述的统一场景级跨模态嵌入空间，支持灵活的跨模态检索和定位。

**[Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)**

:   通过单张编辑参考图像微调 InstructPix2Pix 模型以"学习"编辑能力，结合两阶段可变形3D高斯优化，实现可控、一致的动态3D场景编辑。

**[DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)**

:   提出 DAGSM，一种文本驱动的解耦数字人生成方法，通过 GS-enhanced Mesh（GSM）分别表示人体和各件衣物，支持换装、真实动画和纹理编辑。

**[DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](dashgaussian_optimizing_3d_gaussian_splatting_in_200_seconds.md)**

:   提出 DashGaussian，一种基于频率分析的渲染分辨率和高斯基元数量联合调度方案，将3DGS优化从逐步拟合高频分量的角度进行重新表述，平均加速 45.7% 且不牺牲渲染质量。

**[Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)**

:   提出DP-Recon，将生成式扩散先验（SDS）引入分解式神经场景重建中，通过可见性引导动态调整逐像素SDS权重，解决重建目标与生成引导之间的冲突，实现稀疏视角下完整的物体几何与外观恢复。

**[DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)**

:   将单目深度基础模型 (Depth Anything V2) 融入循环立体匹配框架 RAFT-Stereo，通过组合特征编码器和尺度更新模块，在保持强泛化能力的同时实现多个基准上排名第一的立体匹配性能。

**[Deformable Radial Kernel Splatting](deformable_radial_kernel_splatting.md)**

:   提出可变形径向核 (DRK) 来泛化传统高斯泼溅，通过可学习的径向基函数、$L_1$/$L_2$ 范数混合和边缘锐化机制，用更少的图元实现更高质量的3D场景渲染。

**[Dense-SfM: Structure from Motion with Dense Consistent Matching](dense-sfm_structure_from_motion_with_dense_consistent_matching.md)**

:   提出 Dense-SfM 框架，通过高斯泼溅进行轨迹扩展解决稠密匹配产生的碎片化轨迹问题，结合基于 Transformer 和高斯过程的多视图核化匹配精炼模块，实现高精度稠密 SfM 重建。

**[Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)**

:   提出 Depth Any Camera (DAC) 框架，通过 ERP 统一表示、Pitch-aware 转换和 FoV 对齐等技术，实现仅用透视图像训练即可零样本泛化到鱼眼和360°相机的度量深度估计，在大视野数据集上 $\delta_1$ 精度提升高达50%。

**[DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)**

:   利用预训练的视频扩散模型 (SVD) 进行视频深度估计，通过三阶段训练策略实现可变长度（最长110帧）的时间一致深度序列生成，并设计分段推理策略支持极长视频，在零样本设置下全面超越现有方法。

**[DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)**

:   提出 DepthCues 基准，通过六个人类单目深度线索任务（高度、光影、遮挡、透视、大小、纹理梯度）系统评估 20 个大规模预训练视觉模型的深度感知能力，揭示了类人深度线索在现代视觉模型中的涌现现象。

**[DepthSplat: Connecting Gaussian Splatting and Depth](depthsplat_connecting_gaussian_splatting_and_depth.md)**

:   将高斯泼溅（3DGS）和深度估计两个通常独立研究的任务统一起来：利用预训练单目深度特征增强多视角深度模型以改善 3DGS 重建质量，同时用 3DGS 的光度渲染损失作为无监督预训练目标来学习强大的深度模型，双任务在多个数据集上均达到 SOTA。

**[DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)**

:   提出 DiET-GS 双阶段框架，通过事件双积分（EDI）先验和预训练扩散模型联合约束 3DGS 优化，从模糊多视角图像和事件流中重建清晰的 3D 表示，实现精确色彩和精细细节的高质量新视角合成。

**[DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)**

:   提出首个能从单张肖像生成一致的 360° 全头部视角的方法，通过双外观控制模块、背视图生成 ControlNet 和连续视角序列训练策略，支持真人、风格化和拟人化角色，并可转化为高质量 NeRF 进行实时自由视角渲染。

**[Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models](difix3d_improving_3d_reconstructions_with_single-step_diffusion_models.md)**

:   提出 Difix3D+，利用微调的单步扩散模型（SD-Turbo）在训练阶段渐进式生成伪训练视角回馈 3D 表示，并在推理阶段作为实时后处理增强器，同时兼容 NeRF 和 3DGS，在 FID 上平均实现 2 倍以上提升。

**[Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)**

:   提出 DTC 数据集，包含 2000 个毫米级几何精度和光真实 PBR 材质的 3D 物体数字孪生模型，配合 DSLR 和自中心 AR 眼镜的多视角评估数据，建立了首个面向数字孪生创建任务的综合真实世界评测基准。

**[Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)**

:   Disco4D 提出将服装（用 Gaussian 模型表示）与人体（用 SMPL-X 模型表示）解耦的 4D 人体生成框架，从单张图像生成可动画、可编辑、分层的3D穿衣人体模型，并支持逼真的4D服装动力学。

**[DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)**

:   提出 DoF-Gaussian，为 3D 高斯表示引入基于几何光学的可学习透镜成像模型，通过逐场景深度先验调整和离焦-对焦自适应策略，实现从浅景深（散焦模糊）输入图像重建清晰 3D 场景，并支持可控景深渲染（重对焦、光圈调节、散焦形状变换等交互应用）。

**[Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)**

:   提出 Doppelgangers++，通过引入多样化的 VisymScenes 日常场景训练数据和利用 MASt3R 多层解码器 3D 感知特征训练 Transformer 分类器，显著提升了 doppelganger（视觉混淆图像对）检测的精度和泛化性，并无缝集成到 COLMAP 和 MASt3R-SfM 管线中改善重复结构场景的 3D 重建质量。

**[Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)**

:   提出 Dr. Splat，绕过渲染过程直接将语言对齐的 CLIP 嵌入注册到 3D 高斯上，结合在大规模图像数据上预训练的乘积量化（PQ）实现 6.25% 的嵌入压缩，在完全不需要逐场景优化的前提下（~10 分钟 vs 现有方法 1-24 小时），在开放词汇 3D 语义分割、3D 物体定位和 3D 物体选择任务上显著超越现有方法。

**[DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)**

:   DroneSplat 提出了一个面向野外无人机影像的鲁棒 3DGS 框架，通过自适应局部-全局掩膜策略消除动态干扰物，结合基于多视图立体的几何感知点采样和体素引导优化策略解决有限视角下的重建质量问题，并提供了 24 个场景的无人机 3D 重建数据集。

**[DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)**

:   DropGaussian 提出了一种无需额外先验的简单正则化方法，通过在 3DGS 训练中随机丢弃高斯并引入不透明度补偿因子，使被遮挡的远距离高斯获得更大梯度和可见性，并采用渐进式丢弃率策略有效缓解稀疏视角下的过拟合问题，在不增加计算复杂度的情况下达到与先验方法可比的性能。

**[DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)**

:   DropoutGS 通过随机 Dropout 正则化（RDR）缓解稀疏视角 3DGS 的过拟合问题，再用边缘引导分裂策略（ESS）补偿低复杂度模型丢失的高频细节，作为即插即用模块可与多种 3DGS 方法结合，在 LLFF、DTU、Blender 上达到 SOTA。

**[DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)**

:   DSPNet 提出了一种双视觉场景感知网络，通过文本引导的多视图融合（TGMF）、自适应双视觉感知（ADVP）和多模态上下文引导推理（MCGR）三个模块，综合利用点云和多视图图像信息来解决 3D 问答中的精细感知和鲁棒推理问题，在 SQA3D 和 ScanQA 数据集上达到 SOTA。

**[Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)**

:   提出双曝光立体成像(Dual-Exposure Stereo)方法，通过自动双曝光控制(ADEC)在交替帧中使用不同曝光，结合运动感知的双曝光特征融合网络进行视差估计，将立体相机的有效动态范围扩展至 160%，实现极端光照条件下的鲁棒 3D 成像。

**[DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)**

:   提出 Dual Point Maps (DualPM) 表示——从单张图像预测一对点图（相机空间 P + 规范空间 Q），将可变形物体的 3D 形状和姿态重建简化为点图预测问题，并引入分层 amodal 点图实现完整形状恢复（含自遮挡部分），仅用 1-2 个合成 3D 模型训练即可泛化到真实图像。

**[DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)**

:   DUNE 提出了异构教师联合蒸馏（co-distillation）框架，将来自不同任务和数据域的 2D（DINOv2）与 3D（MASt3R、Multi-HMR）教师模型统一蒸馏为一个 ViT-Base 通用编码器，在语义分割、深度估计、3D 重建和人体姿态恢复等多任务上均达到或超越各自 ViT-Large 教师的性能。

**[Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)**

:   提出 MultiHeadDepth 和 HomoDepth 两个模型，分别通过硬件友好的多头代价体积（LayerNorm+点积近似余弦相似度 + 分组点卷积）和单应性矩阵估计网络 + 2D 矫正位置编码 (RPE) 来优化立体深度估计中代价体积和预处理的延迟瓶颈，在 AR 眼镜场景下精度提升 11.8-30.3% 的同时端到端延迟降低 44.5%。

**[Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)**

:   提出 Instruct-4DGS，利用 4D 高斯 (4DGS) 中静态 3D 高斯和 Hexplane 变形场的内在可分离性，仅编辑静态典范高斯即可实现高效动态场景编辑，再通过 Coherent-IP2P 驱动的分数蒸馏精炼时序对齐以消除运动伪影，将编辑时间缩短一半以上且仅需单 GPU。

**[EigenGS: Representation from Eigenspace to Gaussian Image Space](eigengs_representation_from_eigenspace_to_gaussian_image_space.md)**

:   本文提出 EigenGS，将经典 PCA 的特征空间表示与 2D 高斯 Splatting 图像表示相桥接，通过在特征基上学习统一的高斯参数实现新图像的即时初始化（无需从头优化），并引入频率感知学习机制避免高分辨率重建伪影，在收敛速度和最终质量上全面超越 GaussianImage。

**[Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)**

:   本文提出利用 RGB-D 视频的相机轨迹自动生成情境感知（situation-aware）数据集 View2Cap（20 万+描述、55 万+ QA），并设计情境定位模块（SG）将位姿估计转为锚点分类任务，使 3D LLM 能理解第一人称视角下的空间关系描述（如"左边""右边"随视角变化），在 SQA3D 上 EM@1 达 54.0%。

**[End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)**

:   提出 HOI-TG 框架，用 Transformer 的自注意力机制隐式学习人物交互关系，并在编码器中嵌入图残差模块分别增强人体和物体的拓扑结构建模，在 BEHAVE 和 InterCap 数据集上实现 SOTA 的 HOI 三维重建。

**[End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)**

:   提出 Meta Weight Transformer (MWT)，通过端到端元学习 SIREN 初始化参数和学习率调度，让 INR 的权重结构同时优化重建质量和分类性能，使用简单标准 Transformer 在 SIREN 权重上分类即可超越所有等变架构方法，首次在高分辨率 ImageNet-1K 上实现 INR 分类。

**[EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)**

:   本文提出EnvGS，用一组环境高斯原语（Environment Gaussian）作为显式3D表示来捕获场景反射，通过基于GPU RT Core的可微光线追踪渲染器联合优化环境高斯和基础高斯，在真实场景中首次实现了实时（26+ FPS）且高质量的镜面反射新视角合成，显著超越所有实时方法。

**[ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)**

:   ERUPT 提出了一种高效的潜在视角合成模型，通过 patch-based 解码器替代像素级解码、可学习的潜在相机位姿以及冻结 DINOv2 特征提取器，在不需要精确相机位姿的情况下仅用 5 张无位姿图像即可实现 600fps 的新视角合成，在 MSN 数据集上达到 SOTA 性能。

**[Estimating Body and Hand Motion in an Ego-sensed World](estimating_body_and_hand_motion_in_an_ego-sensed_world.md)**

:   EgoAllo 提出了一种从头戴设备的自中心 SLAM 位姿和图像估计佩戴者全身姿态、身高和手部参数的系统，通过设计满足空间和时间不变性的头部运动条件化参数，将人体运动估计误差降低高达 18%，并利用运动学约束将手部世界坐标误差降低 40%。

**[Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)**

:   本文提出 Event Fields——一种利用事件相机捕获高速、高分辨率、高动态范围光场的新范式，设计了万花筒（空间复用，捕获时间导数）和振镜（时间复用，捕获角度导数）两种互补光学方案，实现了 250fps 百万像素动态场景重聚焦和 100Hz 实时深度估计等前所未有的能力。

**[EventFly: Event Camera Perception from Ground to the Sky](eventfly_event_camera_perception_from_ground_to_the_sky.md)**

:   EventFly 提出了首个事件相机跨平台域适应框架，通过事件激活先验（EAP）识别高激活区域、EventBlend 混合源/目标域事件数据、EventMatch 双判别器对齐特征分布，在车辆→无人机→四足机器人三个平台间的语义分割任务上，相比 source-only 训练平均提升准确率 23.8%、mIoU 77.1%。

**[Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)**

:   本文提出CarGS，通过发现高斯基元对渲染和重建任务的贡献冲突根源在于协方差，设计了轻量残差结构Lite-Geo来自适应解耦两个任务的几何贡献，并引入法线+SDF双引导的致密化策略，在统一模型中同时实现SOTA的渲染质量和重建精度，且存储仅为双模型方法的9%。

**[Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)**

:   本文提出DeepDeblurRF，首次将DNN去模糊网络引入辐射场构建流程，通过设计RF引导去模糊机制和迭代交替框架，在模糊图像输入下实现高质量新视角合成，训练速度比现有方法快10-100倍，同时支持体素网格和3D高斯溅射等多种3D表示。

**[Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)**

:   本文提出了一种面向真实互联网图像的极端三维旋转估计方法，构建了ExtremeLandmarkPairs (ELP)基准数据集，通过渐进式学习方案（全景裁剪→FoV+外观增强→真实数据微调）和辅助通道增强的Transformer模型，在无重叠视角的互联网图像对上显著超越现有方法。

**[Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)**

:   提出 Fast3R，将 DUSt3R 的配对 pointmap 回归推广到多视图，通过 Transformer 的 all-to-all attention 在单次前向传播中处理 N 张无序无位姿图像，彻底消除了 $O(N^2)$ 配对推理和全局对齐优化。

**[FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection](faster_focal_token_acquiring-and-scaling_transformer_for_long-term_3d_objection_.md)**

:   本文提出FASTer，通过Adaptive Scaling机制自适应选取焦点token并压缩序列、分组层次融合策略渐进式聚合长时序点云信息，在Waymo Open Dataset上以最低延迟（75ms）和显存（2856M）取得了新SOTA性能。

**[Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)**

:   将经典的 quadric error metric（QEM）推导到屏幕空间并以法线贴图为输入，结合最优 Delaunay 三角化实现各向异性网格简化，在 90%+ 压缩率下仍保持亚毫米级精度，将高分辨率法线积分从小时级加速到分钟级。

**[Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)**

:   FADE 提出邻接感知（adjacency-aware）的细粒度概念擦除框架，通过 Concept Neighborhood 识别语义邻近类别并设计 Mesh Modules（Erasing + Adjacency + Guidance 三重损失），在精确删除目标概念的同时保留语义相关概念的生成能力，相比 SOTA 方法在邻接保留性能上提升至少 12%。

**[FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)**

:   FLARE 提出级联学习范式（cascade learning），以相机位姿为桥梁将 3D 重建分解为位姿估计→局部几何→全局几何→高斯外观四个渐进阶段，在 0.5 秒内从 2-8 张未标定稀疏图像实现高质量的相机位姿、几何重建和新视角合成。

**[Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)**

:   提出 ORG 框架，首次从单张图像联合建模物体3D几何、相机参数和物体-地面关系，通过预测像素高度图和透视场两个紧凑的密集表示，解决了重建物体"悬浮/倾斜"的问题，显著提升阴影生成和姿态操控的真实感。

**[Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)**

:   提出 Flow-NeRF，首次在无位姿 NeRF 框架中将场景几何、相机位姿和密集光流作为统一的联合优化目标，通过共享点采样、位姿条件化双射映射和特征消息传递机制，在新视角合成和深度估计上大幅超越先前方法，同时首次定义并实现了新视角光流估计。

**[Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution](flowing_from_words_to_pixels_a_noise-free_framework_for_cross-modality_evolution.md)**

:   提出 CrossFlow，一个通用的跨模态 Flow Matching 框架，直接从一种模态的数据分布演化到另一种模态的分布（而非从噪声出发），无需交叉注意力条件机制，在文本到图像生成上略优于标准 Flow Matching 基线，并展现出更好的模型规模和训练步数的缩放特性。

**[Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)**

:   提出 Floxels，用简单的体素网格替代 MLP 作为场景流的隐式表示，结合多帧距离变换损失和聚类一致性约束，在 Argoverse 2 基准上取得仅次于 EulerFlow 的无监督方法第二名，同时将运行时间从一天缩短到10分钟（60-140倍加速）。

**[FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video](fluidnexus_3d_fluid_reconstruction_and_prediction_from_a_single_video.md)**

:   提出 FluidNexus，首次从单个视频实现3D流体外观和速度场的重建与未来预测，通过结合视频生成模型合成多视角参考视频，以及物理-视觉粒子耦合表示桥接可微分仿真与渲染，在新视角合成和未来预测上大幅超越现有多视角方法。

**[FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)**

:   提出 FoundationStereo，一个大规模立体深度估计基础模型，通过百万级高保真合成数据集、Side-Tuning Adapter 融合单目深度先验、以及混合代价体过滤（含 Axial-Planar Convolution 和 Disparity Transformer），实现了无需目标域微调的强零样本泛化性能。

**[FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation](foundhand_large-scale_domain-specific_learning_for_controllable_hand_image_gener.md)**

:   提出 FoundHand，一个在千万级手部数据集（FoundHand-10M）上训练的领域专用扩散模型，以 2D 关键点热力图为通用控制表示，实现精确的手部姿态/视角控制和外观保持，并展现出修复畸形手、视频生成、手物交互视频等零样本涌现能力。

**[FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT](framevggt_frame_evidence_rolling_memory_for_streaming_vggt.md)**

:   提出 FrameVGGT，将流式 VGGT 的 KV 缓存从 token 级保留重组为帧级证据块保留，通过中期记忆库+稀疏锚点的双层有界内存结构，在固定内存预算下保持更连贯的几何支撑，实现长序列3D重建/深度/位姿估计的精度-内存最优权衡。

**[FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)**

:   提出 FreeGave，一个从多视角动态视频中学习 3D 场景几何、外观和物理速度的通用框架，通过为每个 3D 高斯核引入可学习的物理编码（physics code）并设计无散度（divergence-free）速度场参数化，在不依赖 PINN 损失和目标先验的条件下实现精准的未来帧外推。

**[FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts](freescene_mixed_graph_diffusion_for_3d_scene_synthesis_from_free_prompts.md)**

:   FreeScene 提出了一个用户友好的室内场景合成框架，通过 VLM 驱动的 Graph Designer 将自由形式的文本/图像输入转化为场景图，再用 Mixed Graph Diffusion Transformer (MG-DiT) 在混合连续-离散空间上进行图感知去噪，统一支持 text-to-scene、graph-to-scene 等多种任务，在生成质量和可控性上均超越现有方法。

**[FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)**

:   FruitNinja 首次提出为 3DGS 物体生成内部纹理的方法，通过渐进式截面修复 + 体素平滑 + OpaqueAtom GS 策略，实现切割后实时渲染无需额外优化，在语义对齐和纹理一致性上显著优于基线。

**[FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)**

:   FSFM 提出首个面向人脸安全任务的自监督预训练框架，通过 CRFR-P 面部掩码策略 + MIM/ID 双任务协同学习真实人脸的 3C 表示（区域内一致性、区域间连贯性、局部到全局对应性），在深伪检测、活体检测和扩散伪造检测三大任务上超越任务专用 SOTA。

**[FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)**

:   FSHNet 提出全稀疏混合网络，通过 SlotFormer（槽分区+线性注意力）建立全局范围的稀疏体素交互，配合动态稀疏标签分配和稀疏上采样模块，在 Waymo、nuScenes、Argoverse2 三大基准上超越现有稀疏和密集检测器。

**[Functionality Understanding and Segmentation in 3D Scenes](functionality_understanding_and_segmentation_in_3d_scenes.md)**

:   Fun3DU 首次提出针对 3D 场景功能性理解的方法，通过 LLM 链式思维解析任务描述 + VLM 多视角分割功能性物体 + 2D-3D 投票聚合，在 SceneFun3D 上大幅超越开放词汇 3D 分割基线（mIoU +13.2）。

**[GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)**

:   提出 GA3CE 方法，通过将主体 3D 姿态和场景物体位置编码到以主体为中心的自我中心空间中，并设计方向-距离分解的 D3 位置编码，在 Transformer 中学习 3D 注视方向与场景上下文的空间关系，在无约束设置下将 3D 注视角度误差降低 13%–37%。

**[GASP: Gaussian Avatars with Synthetic Priors](gasp_gaussian_avatars_with_synthetic_priors.md)**

:   提出 GASP，利用合成数据训练 Gaussian Avatar 的生成式先验模型（auto-decoder），通过三阶段拟合过程和学到的 per-Gaussian 语义特征关联来跨越合成-真实域差距，仅从单张图片或短视频即可创建支持 360° 渲染的高质量实时可动画头像（70fps）。

**[GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)**

:   提出 GaussHDR，通过统一 3D 和 2D 局部色调映射来改进 HDR 高斯溅射，设计残差局部色调映射器和不确定性自适应调制机制，同时提升 HDR 重建稳定性和 LDR 拟合质量，在合成和真实场景上大幅超越现有方法。

**[Gaussian Eigen Models for Human Heads](gaussian_eigen_models_for_human_heads.md)**

:   提出 Gaussian Eigen Models (GEM)，通过 PCA 将高质量 CNN-based Gaussian Avatar 蒸馏为轻量级线性特征基表示，仅需低维系数的线性组合即可生成面部动画，实现高质量、超轻量（7MB起）和超快速（200+ fps）的可动画头像，并支持从单目视频的实时跨人表情驱动。

**[Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)**

:   提出 Gaussian Splatting Feature Fields (GSFFs)，将 3DGS 的显式几何与隐式特征场结合，通过自监督对比学习训练尺度感知的 3D 特征和 2D 编码器，并利用基于 Delaunay 图的空间聚类将特征转化为分割标签，实现了高精度的非隐私和隐私保护视觉定位。

**[Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)**

:   本文提出 EOGS，首个基于 3D 高斯溅射的地球观测框架，通过仿射相机近似、阴影映射和三种正则化策略，在卫星图像三维重建任务上达到与 EO-NeRF 相当的精度，同时训练速度快 300 倍（3 分钟 vs 15 小时）。

**[GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting](gaussianudf_inferring_unsigned_distance_functions_through_3d_gaussian_splatting.md)**

:   本文提出 GaussianUDF，通过将 2D 高斯平面贴合到曲面上，利用自监督和梯度推断为近场和远场分别提供无符号距离监督，首次在 3DGS 框架内高效推断连续 UDF，实现高质量开放曲面重建。

**[GauSTAR: Gaussian Surface Tracking and Reconstruction](gaustar_gaussian_surface_tracking_and_reconstruction.md)**

:   GauSTAR 提出一种将高斯原语绑定到网格面上的"高斯曲面"表示，通过自适应解绑和重网格化机制处理拓扑变化，配合基于曲面的场景流初始化，首次实现了动态场景中同时兼顾照片级渲染、精确曲面重建和可靠三维跟踪的统一框架。

**[GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)**

:   GEAL 提出双分支架构，用 3D 高斯溅射将点云渲染为逼真 2D 图像从而利用预训练 2D 基础模型的泛化能力，通过粒度自适应融合和 2D-3D 一致性对齐实现跨模态知识迁移，在标准和腐败数据基准上全面超越现有 3D 功能可供性方法。

**[Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)**

:   本文提出Gen3DEval，一个基于vLLM微调的text-to-3D生成质量评估框架，通过对Llama3模型在合成+人工标注数据上微调，实现对3D物体外观、表面质量和文本一致性的自动评估，在与人类偏好对齐上显著超越GPT-4o等通用模型。

**[Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)**

:   本文提出KFC-W，一种从无位姿互联网照片生成3D一致视频的自监督方法，通过在视频扩散模型上联合训练多视角修复和视角插值两个目标，无需任何3D标注（如相机参数），生成的视频在几何和外观一致性上超越商业模型Luma Dream Machine。

**[Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)**

:   本文提出先用多视图重光照扩散模型将不同光照下拍摄的图像统一到参考光照条件，再用带有"shading embedding"的鲁棒 NeRF 模型重建 3D 表示，在极端光照变化下实现了远超现有方法的高保真外观重建，尤其擅长恢复镜面/高光效果。

**[Generative Omnimatte: Learning to Decompose Video into Layers](generative_omnimatte_learning_to_decompose_video_into_layers.md)**

:   Generative Omnimatte 通过微调视频 inpainting 扩散模型（Casper）学会物体及其关联效果（阴影、反射）的联合移除，结合 trimask 条件和 omnimatte 优化，在无需静态背景假设或相机位姿的前提下实现了高质量的视频图层分解和被遮挡区域补全。

**[GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)**

:   提出 GenFusion，通过重建驱动的视频扩散模型修复 3D 重建伪影并生成不可见区域内容，设计循环融合管线迭代地将生成结果加入训练集，实现稀疏视图下高质量 3D 场景重建和内容扩展。

**[GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)**

:   提出 GenPC 零样本点云补全框架，通过 Depth Prompting 模块将部分点云转化为深度图再生成 RGB 图像作为 Image-to-3D 模型的输入，再通过 Geometric Preserving Fusion 模块将生成的 3D 形状与原始点云对齐融合，实现了比 SDS 优化方法更快更好的真实世界扫描补全。

**[GenVDM: Generating Vector Displacement Maps From a Single Image](genvdm_generating_vector_displacement_maps_from_a_single_image.md)**

:   提出首个从单张图像生成 Vector Displacement Map (VDM) 的方法，通过微调 Zero123++ 生成多视角法线图、使用神经 SDF 重建网格、再用神经变形场参数化为 VDM 图像，并构建了首个学术 VDM 数据集，为 3D 艺术家提供了按需生成自定义几何细节印章的能力。

**[Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)**

:   提出通过优化三角网格的表面法线方向、结合可微分ARAP（dARAP）层恢复顶点位置的方法，实现文本驱动的网格风格化，能产生表达力强但保持形状身份的几何变形。

**[GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)**

:   提出GIFStream，一种基于canonical空间+变形场的4D高斯表示方法，通过为每个anchor点附加时间相关的特征流（feature stream）来增强复杂运动建模能力，同时利用时间对齐的结构和端到端压缩实现30 Mbps高质量1080p沉浸式视频。

**[Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)**

:   提出全局-局部树搜索算法，利用VLM的空间推理能力，通过层次化场景表示和emoji网格的视觉提示，实现高质量3D室内场景布局生成，在用户研究中平均排名第一。

**[Glossy Object Reconstruction with Cost-effective Polarized Acquisition](glossy_object_reconstruction_with_cost-effective_polarized_acquisition.md)**

:   提出一种低成本偏振辅助3D重建方法，仅需在普通RGB相机前加一块线性偏振片，每视角拍摄一张偏振图像（无需校准偏振角），通过神经隐式场端到端优化偏振渲染损失来恢复光泽物体的高保真几何和材质分解。

**[GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector](go-n3rdet_geometry_optimized_nerf-enhanced_3d_object_detector.md)**

:   提出GO-N3RDet，通过位置信息嵌入的体素优化模块（PEOM）、双重重要性采样（DIS）和不透明度优化模块（OOM）三个协同模块，解决基于NeRF的多视图3D检测中缺乏3D位置信息和场景几何感知不足的问题，在ScanNet和ARKitScenes上建立了新SOTA。

**[GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)**

:   提出 GREAT 框架，通过多头 Affordance Chain-of-Thought (MHACoT) 微调 InternVL 推理交互图像中的物体几何属性和潜在交互意图，形成 affordance 知识字典，并通过跨模态自适应融合模块（CMAFM）将知识注入点云和图像特征，实现开放词汇 3D 物体 affordance 定位。同时构建最大规模 3D affordance 数据集 PIADv2（15K 图像 + 38K 点云）。

**[Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)**

:   提出首个多模态多视角 3D 功能区域定位任务和 AGPIL 数据集（30,972 对点云-图像-语言三元组），并设计基于 VLM 的 LMAffordance3D 框架，融合 2D/3D 空间特征与语言语义实现从 full-view 到 partial/rotation-view 的泛化。

**[GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction](gs-2dgs_geometrically_supervised_2dgs_for_reflective_object_reconstruction.md)**

:   在 2DGS 基础上引入基础模型（Marigold + Depth Pro）的深度/法线伪标签监督和延迟着色（Deferred Shading）的物理渲染管线，在反射物体重建上显著超越 GS 方法、媲美 SDF 方法且快了一个数量级。

**[GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)**

:   提出 GuardSplat，通过 CLIP 引导的消息解耦优化（仅训练解码器 5 分钟）和 SH-aware 水印嵌入（仅修改球谐偏移量），实现对 3DGS 资产的高容量、高保真、鲁棒版权保护，总优化时间仅 15 分钟。

**[HandOS: 3D Hand Reconstruction in One Stage](handos_3d_hand_reconstruction_in_one_stage.md)**

:   HandOS 提出了一个端到端的单阶段3D手部重建框架，通过冻结预训练检测器并引入交互式2D-3D解码器，将手部检测、2D姿态估计和3D mesh重建统一到一个pipeline中，消除了传统多阶段方法的冗余计算和累积误差，在 FreiHand 上达到 5.0 PA-MPJPE 的 SOTA 性能。

**[Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)**

:   本文提出了首个基于硬件光栅化的射线型3D高斯泼溅（RayGS）渲染方案 VKRayGS，通过严谨的数学推导在3D空间中构建最小包围四边形，实现了约40倍的渲染加速，同时保持了RayGS的高质量渲染效果，并额外提出了RayGS的MIP抗锯齿方案。

**[Hash3D: Training-free Acceleration for 3D Generation](hash3d_training-free_acceleration_for_3d_generation.md)**

:   Hash3D 发现 SDS 优化过程中扩散模型对相邻相机位姿和时间步的特征高度冗余，通过自适应网格哈希表缓存和复用中间特征，在无需训练的情况下将多种text-to-3D和image-to-3D方法加速1.3~4倍，同时还提升了多视角一致性。

**[HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)**

:   HaWoR 首次实现了从第一人称视频重建世界坐标系下的3D手部运动，通过将任务解耦为相机空间手部重建 + 自适应SLAM相机轨迹估计，并引入运动补全网络处理手部出视野的情况，在 HOT3D 数据集上取得 SOTA 的全局轨迹精度（ATE 3.36mm）和手部重建质量（PA-MPJPE 4.79mm）。

**[HD-EPIC: A Highly-Detailed Egocentric Video Dataset](hd-epic_a_highly-detailed_egocentric_video_dataset.md)**

:   HD-EPIC 提供了41小时非脚本厨房第一人称视频，具有前所未有的标注密度（每分钟263条标注），涵盖食谱步骤、细粒度动作、营养信息、3D数字孪生、物体运动轨迹和注视方向等多层级互联标注，并构建了26K问题的VQA基准——最强的 Gemini Pro 仅达37.6%。

**[Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes](hearing_hands_generating_sounds_from_physical_interactions_in_3d_scenes.md)**

:   本文提出通过在3D重建场景中记录人手交互的动作-声音对，训练基于rectified flow的生成模型，实现从3D手部轨迹预测对应交互声音，生成结果在人类评估中约47%无法与真实声音区分。

**[HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)**

:   提出HeatFormer——一种基于Transformer的神经优化器，通过将SMPL参数估计转化为热力图生成与对齐问题，实现对多视角图像中人体形状和姿态的迭代优化恢复，在Human3.6M上达到29.5mm MPJPE的SOTA精度，且对视角数量、相机配置和遮挡具有强鲁棒性。

**[High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)**

:   GS-RGBN 提出混合 Voxel-Gaussian 表示为无结构高斯提供 3D 空间约束，并设计跨体积融合（CVF）模块在特征层面融合 RGB 语义信息和法线几何信息，从单张图像在数秒内生成高保真 3D 对象，在 GSO 数据集上 PSNR 超出次优方法 5.59dB。

**[HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)**

:   提出 HOI3DGen 框架，通过MLLM自动标注高质量交互数据 + 视角条件化微调扩散模型 + 3D提升与SMPL配准，首次实现从文本精确控制接触语义的高质量3D人物交互生成，在文本一致性上超越基线4-15倍。

**[Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)**

:   本文提出 Horizon-GS，通过粗到精两阶段训练策略、相机分布平衡机制和多分辨率 LOD 结构，首次实现了航空视角和街景视角的统一 3D 高斯溅射重建和实时渲染，在多个城市场景数据集上达到 SOTA 渲染质量。

**[HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)**

:   Meta 发布 HOT3D，首个基于真实头戴设备（Project Aria + Quest 3）的大规模自中心多视角手物交互数据集，包含 833 分钟 370 万+图像、19 名受试者与 33 个物体的交互，并通过实验证明多视角方法在 3D 手部追踪、6DoF 物体位姿估计等任务上显著优于单视角方法。

**[HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)**

:   HRAvatar 提出了一种基于3DGS的单目视频头部重建方法，通过可学习blendshapes和LBS实现灵活变形，结合端到端表情编码器减少追踪误差，并引入物理渲染模型实现高质量实时重光照。

**[Hybrid eTFCE-GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](hybrid_etfce-grf_exact_cluster-size_retrieval_with_analytical_p-values_for_voxel.md)**

:   将 eTFCE 的并查集精确聚类大小查询与 pTFCE 的解析 GRF p 值推断结合，首次在单一框架中实现精确聚类检索+无需置换检验的统计推断，速度比置换 TFCE 快 1300 倍，在全脑体素形态测量中保持严格 FWER 控制。

**[HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)**

:   HybridGS首次提出混合2D+3D高斯表示，用多视角一致的3D高斯建模静态场景、用单视图独立的2D高斯建模瞬态物体，配合多视角监督和多阶段训练实现了含干扰元素场景下SOTA的新视角合成质量。

**[HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)**

:   首次将 3DGS 成功扩展到高光谱新视角合成（HNVS），通过在学习的潜在空间中进行高光谱渲染，配合自适应密度控制与像素级光谱剪枝，实现高维光谱数据的高效准确重建。

**[IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)**

:   构建基于 3DGS 的层次化语义特征场，融合 CLIP/SAM/DINOv2 的语义信息，实现铰接物体的交互式 affordance 预测和跨状态运动参数恢复，支持任意类别、多可动部件的复杂室内场景。

**[Identity-preserving Distillation Sampling by Fixed-Point Iterator](identity-preserving_distillation_sampling_by_fixed-point_iterator.md)**

:   提出 Identity-preserving Distillation Sampling (IDS)，通过不动点迭代正则化（FPR）修正文本条件分数函数中导致身份丢失的梯度误差，生成引导噪声替代随机噪声，在 2D 图像编辑和 3D NeRF 编辑中实现结构和姿态的高度保持。

**[IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)**

:   本文提出IMFine，一种用于无约束场景（包括360°环绕）的3D修复流水线，通过几何先验引导的warping和基于测试时适应的多视角refinement网络生成多视角一致的修复图像，并提出了一种新的修复mask检测技术来精确区分真正需要修复的遮挡区域，在多样化的benchmark上显著超越现有方法。

**[Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)**

:   本文提出局部化点管理（LPM）策略，通过多视角几何约束定位导致渲染误差的 3D 区域，在这些区域内执行针对性的点加密和不透明度重置，作为即插即用模块可提升多种 3DGS 模型的重建质量，同时保持实时渲染速度。

**[IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)**

:   本文提出 IncEventGS，首个仅用单目事件相机、无需已知位姿即可增量重建 3D 高斯溅射场景的方法，采用跟踪-建图 SLAM 范式联合优化相机运动和场景表示，在新视角合成和位姿估计上均超越现有方法。

**[Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)**

:   将 3D 编辑问题转化为多视角一致的 2D inpainting 问题，通过微调 SDXL-inpainting 模型在 2×2 视角网格上同时生成一致的填充内容，再用 LRM 重建 3D，实现约 3 秒完成高质量 3D 编辑——比 SDS 方法快数百倍。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)**

:   提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模进行多曝光融合 + MetaNet 预测场景自适应色调映射器，从未标定多曝光 LDR 图像一次前向推理重建 HDR 3D 高斯，速度比优化方法快 ~700 倍。

**[InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](interactvlm_3d_interaction_reasoning_from_2d_foundational_models.md)**

:   InteractVLM 利用大规模视觉语言模型(VLM)的广泛视觉知识，通过"渲染-定位-提升"(Render-Localize-Lift)框架将2D基础模型的推理能力迁移到3D空间，实现了从单张野外图像估计人体和物体3D接触点，并用于人物交互联合重建，在接触估计任务上F1分数提升20.6%。

**[IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)**

:   本文提出IRGS框架，首次在高斯泼溅中集成完整渲染方程（无简化），通过提出的可微分2D高斯光线追踪技术实时计算入射光的可见性和间接辐射，在多个逆渲染基准上取得了显著优于先前方法的重光照和材质估计效果。

**[IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)**

:   IRIS提出了一个从多视角LDR图像中联合恢复HDR光照、物理材质和相机响应函数的逆渲染框架，通过显式建模色调映射、自动检测发光体和迭代优化策略，在真实和合成室内场景上实现了高质量的材质估计、重光照和虚拟物体插入。

**[iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)**

:   iSegMan提出了一个无需场景特定训练的交互式3DGS分割与操作框架，通过极线引导的交互传播(EIP)和基于可见性的高斯投票(VGV)实现精确的3D区域控制，配合完整的操作工具箱支持语义编辑、上色、缩放、复制粘贴、组合和删除等多种功能。

**[Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)**

:   将相机运动建模为时间连续的角速度和线速度，通过速度积分避免直接优化大范围 camera-to-world 变换，结合时间依赖 NeRF 和 SDF flow 约束，无需深度先验即可从单目视频联合优化位姿和场景几何。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)**

:   提出 JOPP-3D 框架，通过将全景图切线分解为透视图像、利用 SAM+CLIP 进行3D实例-语义对齐，首次实现对3D点云和全景图像的联合开放词汇语义分割，在 Stanford-2D-3D-s 和 ToF-360 数据集上超越现有方法。

**[Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)**

:   将 3D 资产生成转化为 2D 图像生成问题——微调 Flux DiT 模型生成"3D Bundle Image"（四视图 RGB + 法线图拼贴），再用 ISOMER 重建 3D mesh，并通过 ControlNet 扩展支持 3D 增强和编辑。

**[Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)**

:   本文提出 Layered Motion Fusion（LMF），将 2D 运动分割模型的预测融合到分层神经辐射场的动态和半静态层中，结合测试时精修策略，首次证明 3D 方法能在第一人称视频的动态目标分割中超越 2D 基线，动态目标分割 mAP 提升 30.5%。

**[Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)**

:   提出可学习无穷 Taylor 级数（Learnable Infinite Taylor Formula）建模动态场景中高斯基元的位置/旋转/缩放随时间的演化，用三阶 Taylor 展开捕捉大运动、MLP+LBS 构造 Peano 余项补偿高阶项，实现无近似误差的运动建模，N3DV 和 Technicolor 数据集上超越 SOTA。

**[Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)**

:   提出首个统一室内外稀疏监督 3D 目标检测方法 CPDet3D，通过类感知原型聚类（跨场景 Sinkhorn-Knopp 最优传输匹配）挖掘未标注物体的类别，再用多标签协同精化（伪标签 + 原型标签）恢复漏检，仅用每场景 1 个标注即达 ScanNet V2 全监督 78% / SUN RGB-D 90% / KITTI 96% 性能。

**[Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)**

:   Light3R-SfM提出了首个前馈式端到端SfM框架，通过可学习的潜在全局对齐模块替代传统的优化式全局对齐，结合基于检索分数的最短路径树构建场景图，在Tanks&Temples 200图设置下仅需33秒完成重建（比MASt3R-SfM快49倍），同时保持相当的精度。

**[LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)**

:   提出 LIM——首个前馈式跨类别动态 4D 资产重建模型，通过在隐式 triplane 表示间进行 Transformer 插值并引入因果一致性损失，实现秒级高质量连续时间插值与一致拓扑的网格跟踪。

**[LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene (FA-NeRF)](lookcloser_frequency-aware_radiance_field_for_tiny-detail_scene.md)**

:   FA-NeRF 提出一种频率感知的神经辐射场框架，通过 3D 频率量化方法分析场景频率分布，结合频率网格、频率感知特征重加权和自适应光线行进，在单一模型中同时捕捉场景整体结构和高清微小细节，在多频率数据集上显著超越所有基线方法。

**[LT3SD: Latent Trees for 3D Scene Diffusion](lt3sd_latent_trees_for_3d_scene_diffusion.md)**

:   提出 LT3SD，将 3D 场景渐进分解为潜在树（每层包含几何体积 + 高频潜在特征体积），在此表征上训练基于 patch 的扩散模型，实现从粗到细、逐 patch 的高质量无限 3D 场景生成，FID 相对 SOTA 提升 70%。

**[LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)**

:   提出 LUCAS，首个将人脸和头发解耦为分层 mesh 的通用先验 Avatar 模型，通过共享表情编码 + 独立解码实现自然的面部-头发交互，同时支持实时 mesh 渲染（45 FPS mobile）和高保真 Gaussian 渲染，在跨身份零样本驱动中达到 SOTA。

**[MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)**

:   提出 MAC-Ego3D 框架，通过统一的 3D 高斯泼溅（Gaussian Splatting）表示让多个智能体独立构建、对齐和迭代优化局部地图，利用智能体内和智能体间高斯共识机制实现实时协作位姿估计和逼真 3D 重建，达到 15 倍推理加速、位姿误差降低一个数量级、RGB PSNR 提升 4-10 dB。

**[MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)**

:   提出基于刚性可变形3D高斯场景表示的多智能体SLAM系统MAGiC-SLAM，通过新颖的追踪、地图融合机制和基于DinoV2的回环检测，实现了比CP-SLAM快24倍的处理速度、7倍低的GPU占用，以及更精确的轨迹估计和高保真新视角渲染。

**[Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)**

:   Mani-GS 提出了一种基于三角网格操控 3D Gaussian Splatting 的方法——通过在每个三角形上定义局部坐标系来绑定高斯，使得网格变形时高斯的位置、旋转和缩放能自适应调整，从而实现大变形、局部编辑和软体仿真等多种操控类型，同时保持高质量渲染且对网格精度有高容忍度。

**[ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)**

:   提出多层遮挡（MLO）表示学习 3D 手-物遮挡关系，并将 Objaverse 大规模 3D 物体数据整合进训练，实现首个支持灵巧双手操作 + 可泛化物体外观的手-物操作视频生成框架。

**[MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation](mar-3d_progressive_masked_auto-regressor_for_high-resolution_3d_generation.md)**

:   提出金字塔 VAE + 级联 MAR（MAR-LR → MAR-HR）的渐进式 3D 生成框架，通过随机遮罩适配 3D token 的无序特性，并用条件增强策略缓解分辨率上扩展时的累计误差，在开源方法中达到 SOTA。

**[MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation](marvel-40m_multi-level_visual_elaboration_for_high-fidelity_text-to-3d_content_c.md)**

:   构建了包含 890 万 3D 资产、4000 万+多层级文本标注的大规模 3D 描述数据集 MARVEL-40M+，通过多阶段自动标注管线（InternVL2 + Qwen2.5）生成从详细描述到简洁标签的五级标注，并基于此微调 SD3.5 实现 15 秒内的高保真文本到 3D 生成。

**[Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)**

:   提出 MPEC（Masked Point-Entity Contrastive learning），通过跨视角 point-to-entity 对比学习和 entity-to-language 对齐两个层次的对比损失来训练 3D 编码器，在保持实体级几何-空间信息的同时实现开放词汇语义理解，在 ScanNet 上取得 66.0% f-mIoU 的 SOTA 并在 8 个数据集的下游任务上展现强泛化能力。

**[MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)**

:   将 3DGS 中的高斯剪枝从确定性移除改为概率性存在建模，提出 masked-rasterization 技术使未被采样的高斯仍能接收梯度以动态评估其贡献，在 Mip-NeRF360/T&T/DeepBlending 上实现 62-75% 的高斯剪枝率且仅损失 0.02 PSNR。

**[MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)**

:   首个以双视图 3D 重建先验 MASt3R 为基础构建的实时单目稠密 SLAM 系统，通过高效的点图匹配、光线误差跟踪、局部融合、回环检测和二阶全局优化，在无需相机标定的情况下实现 15 FPS 的全局一致位姿估计和稠密几何重建，性能达到 SOTA。

**[MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)**

:   提出 MAtCha Gaussians，将场景表面建模为 2D 流形上的图表集合（Atlas of Charts）并用 2D Gaussian Surfels 渲染，通过单目深度初始化 + 轻量神经变形模型 + 结构保持损失，在仅 3-10 张稀疏视图下数分钟内同时实现高质量表面网格重建和逼真新视角合成。

**[Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)**

:   提出 Material Anything，一个全自动的统一扩散框架，通过三头 U-Net 架构、置信度掩码和渲染损失适配预训练图像扩散模型生成 PBR 材质（albedo/roughness/metallic/bump），配合置信度引导的渐进式多视角生成策略和 UV 空间精化模型，为不同光照条件（无纹理/纯 albedo/扫描/生成）下的 3D 物体统一生成高质量材质贴图。

**[Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)**

:   Matrix3D 提出一个基于多模态扩散 Transformer 的统一摄影测量模型，通过掩码学习策略在单一模型中同时完成位姿估计、深度预测和新视角合成三大任务，在 CO3D 上位姿估计旋转精度达 96.5%，显著超越所有专用方法。

**[MEGA: Masked Generative Autoencoder for Human Mesh Recovery](mega_masked_generative_autoencoder_for_human_mesh_recovery.md)**

:   MEGA 提出了一种基于遮掩生成建模的人体网格恢复方法，通过将人体 mesh 离散化为 token 序列，在自监督预训练后进行图像条件生成，同时支持确定性单次预测和随机多输出生成模式，在两种模式下均达到 SOTA 性能。

**[MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)**

:   MegaSaM 通过在深度视觉 SLAM 框架中集成单目深度先验、运动概率图和不确定性感知全局 BA，实现了对日常拍摄的动态视频进行精确、快速且鲁棒的相机参数和深度图估计，在合成和真实数据集上显著优于现有方法。

**[MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data](megasynth_scaling_up_3d_scene_reconstruction_with_synthesized_data.md)**

:   MegaSynth 提出通过消除语义信息依赖来实现可扩展的 3D 场景数据合成，生成了包含 70 万个场景的数据集（比真实数据集 DL3DV 大 50 倍），用于训练大规模重建模型（LRM），在多个基准上带来 1.2-1.8dB PSNR 的显著提升。

**[Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)**

:   本文提出Mesh Mamba，首个基于状态空间模型（SSM）的统一网格显著性预测模型，通过纹理对齐、子图嵌入和双向SSM，实现对有纹理和无纹理3D网格的高质量视觉注意力预测，并构建了首个系统对比有/无纹理条件下显著性差异的数据集。

**[MeshArt: Generating Articulated Meshes with Structure-Guided Transformers](meshart_generating_articulated_meshes_with_structure-guided_transformers.md)**

:   MeshArt提出了一种层次化Transformer方法，通过将铰接物体分解为高层关节结构和低层部件网格两阶段生成，自回归地生成紧凑、清晰的三角网格铰接物体，在结构覆盖率上提升57.1%，网格FID提升209分。

**[MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)**

:   本文提出MEt3R，一种基于DUSt3R重建和DINO特征比较的多视角一致性评价指标，无需相机位姿即可衡量生成图像的3D一致性，并附带开源了一个多视角潜在扩散模型MV-LDM。

**[MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans](metascenes_towards_automated_replica_creation_for_real-world_3d_scans.md)**

:   MetaScenes 构建了一个大规模可仿真3D场景数据集（15366个物体, 831类），通过从真实扫描中自动替换物体资产实现 Real-to-Sim 转换，并提出多模态对齐模型 Scan2Sim 实现自动化资产选择，在场景合成和VLN跨域迁移任务上验证了数据集的价值。

**[MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)**

:   MICAS 针对 3D 点云 in-context learning 中的任务间（inter-task）和任务内（intra-task）采样敏感性问题，提出了多粒度自适应采样机制——包含任务自适应点采样（Gumbel-softmax 可微采样）和查询特定 prompt 采样（基于概率排序选择最优 prompt），在 ShapeNet 基准上将 part segmentation 提升了 4.1%。

**[MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)**

:   MIDI 将预训练的 image-to-3D 单物体生成模型扩展为多实例扩散模型，通过新颖的多实例注意力机制在 3D 生成过程中直接捕获物体间的空间交互关系，从单张图片同时生成多个具有正确空间布局的 3D 实例，在合成和真实数据集上均大幅超越现有方法。

**[Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)**

:   本文首次探索用 3D Gaussian Splatting (GS) 点云替代传统点云作为 3D 分类的输入表示，利用 GS 中的 scale/rotation 系数区分线状和平坦表面、利用 opacity 区分透明/反射物体，构建了首个真实世界 GS 点云数据集，在多种分类方法上均验证了 GS 点云消除歧义的有效性。

**[MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots](mne-slam_multi-agent_neural_slam_for_mobile_robots.md)**

:   提出首个分布式多智能体协作神经SLAM框架MNE-SLAM，通过联合场景表示、内到外回环检测和多子地图融合实现高质量协作建图与追踪，仅需点对点通信，并发布首个覆盖单/多智能体场景的真实室内神经SLAM (INS) 数据集。

**[Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)**

:   提出 Mobile-GS，通过深度感知的无序渲染（消除排序瓶颈）+ 神经视角依赖增强 + 一阶SH蒸馏 + 神经向量量化 + 贡献度剪枝，首次在 Snapdragon 8 Gen 3 手机 GPU 上实现 116 FPS 实时高斯溅射渲染，存储仅 4.6MB 且视觉质量与原始 3DGS 相当。

**[MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation](monocular_and_generalizable_gaussian_talking_head_animation.md)**

:   提出MGGTalk框架，仅用单目数据集训练即可泛化到未见身份，核心思路是利用深度估计和面部对称先验来弥补单目数据中几何与外观信息的不完整性，实现基于3DGS的高质量说话头动画。

**[MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)**

:   提出MonoPlace3D，一个场景感知的3D数据增强系统，核心是学习一个从场景图像到合理3D边界框分布的放置网络（SA-PlaceNet），配合基于ControlNet的真实感渲染管线，显著提升单目3D检测器性能和数据效率。

**[Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)**

:   提出Morpheus，一种自回归3DGS风格化方法，核心贡献包括：(1) 新的RGBD扩散模型实现外观和形状风格化的独立强度控制；(2) Warp ControlNet通过变形合成帧传播风格；(3) 深度引导的特征共享确保多视角一致性。

**[Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)**

:   提出自动化数据生成管线构建大规模3D mask-text数据集Mosaic3D-5.6M（5.6M对、30K场景），训练语言对齐3D编码器+mask decoder，实现首个单阶段开放词汇3D实例分割。

**[MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)**

:   提出4D Motion Scaffold (MoSca)表示，通过稀疏6-DoF轨迹图紧凑编码场景运动，结合2D基础模型先验和物理正则化，从无位姿的随手拍单目视频实现全自动4D场景重建。

**[MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning](most_efficient_monarch_sparse_tuning_for_3d_representation_learning.md)**

:   提出首个基于重参数化的3D PEFT方法MoST，设计Point Monarch结构化矩阵（在Monarch基础上加入KNN局部特征平滑），仅调3.6%参数在多个3D benchmark上超越全量微调。

**[MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出 MotionAnyMesh，一种零样本框架，通过 SP4D 运动学先验引导 VLM 推理消除幻觉 + 物理约束轨迹优化保证无碰撞，将静态3D网格自动转化为仿真可用的铰接数字孪生，物理可执行率达 87%，是现有最好方法的近两倍。

**[MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond](motionpro_exploring_the_role_of_pressure_in_human_mocap_and_beyond.md)**

:   构建大规模压力-RGB-光学动捕数据集 MotionPRO（70人/400类动作/12.4M帧），并提出 FRAPPE 基线将压力信号与单目 RGB 融合，显著提升全身姿态估计的物理合理性和全局轨迹精度，进一步将压力先验扩展至人形机器人驱动。

**[MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)**

:   针对多物体室内场景的新视角合成（NVS），通过注入结构感知特征（深度+物体掩码）、引入辅助掩码预测任务、设计结构引导的时间步采样调度器三项设计，显著提升跨视角的物体放置和几何一致性。

**[MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)**

:   将单目深度和法线先验紧密集成到经典增量 SfM 中，通过不确定性传播和交替优化突破三视图轨迹的根本限制，首次实现仅凭两视图轨迹的可靠 3D 重建，在极端低重叠和低视差场景下显著超越所有现有方法。

**[Multi-View Pose-Agnostic Change Localization with Zero Labels](multi-view_pose-agnostic_change_localization_with_zero_labels.md)**

:   提出首个无标签、位姿无关的多视角变化检测方法，通过在 3D Gaussian Splatting 中嵌入变化通道（change-aware 3DGS），融合多视角的特征感知和结构感知变化掩码，在复杂多物体场景中实现 1.7× mIoU 和 1.5× F1 的 SOTA 性能提升，并能为未见视角生成变化掩码。

**[Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)**

:   提出 Murre，将 SfM 稀疏点云作为条件注入扩散模型单目深度估计，生成多视角一致的度量深度图后进行 TSDF 融合，在仅用少量合成数据微调后即可在室内、街景、航拍等多种真实场景中超越 SOTA MVS 和神经隐式重建方法。

**[MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)**

:   本文提出MUSt3R，将DUSt3R从成对架构扩展为多视图架构：通过对称化解码器（参数减半）+多层memory机制实现任意数量图像在统一坐标系下的高帧率3D重建，同一网络可同时处理离线SfM和在线Visual Odometry场景，在TUM-RGBD无标定VO中ATE仅5.5cm。

**[MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)**

:   MV-DUSt3R 提出单阶段前馈网络，通过多视图解码器块联合处理任意数量的无位姿输入视图，完全省去 DUSt3R 所需的全局优化，实现比 DUSt3R 快 48~78 倍的场景重建，同时 Chamfer Distance 降低 1.6~3.2 倍；进一步的 MV-DUSt3R+ 引入跨参考视图注意力块，在大场景上进一步提升重建质量。

**[MVBoost: Boost 3D Reconstruction with Multi-View Refinement](mvboost_boost_3d_reconstruction_with_multi-view_refinement.md)**

:   MVBoost 提出了一种通过多视图精炼策略生成伪真值数据来增强 3D 重建的框架，巧妙结合多视图生成模型的高精度和 3D 重建模型的一致性优势，在 GSO 数据集上实现了 SOTA 的单图到 3D 重建效果（PSNR 18.561, CD 0.101）。

**[MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)**

:   MVGenMaster 提出了一种融合度量深度几何先验的多视图扩散模型，配合 160 万场景的 MvD-1M 数据集和无训练的 key-rescaling 技术，能在单次前向推理中从任意参考视图生成多达 100 个新视角，在域内外 NVS 基准上全面超越 CAT3D 和 ViewCrafter。

**[MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)**

:   MVPaint 提出了一个三阶段的 3D 纹理生成框架——同步多视图生成 (SMG) + 空间感知 3D 补绘 (S3I) + UV 精炼 (UVR)，通过在图像域而非 latent 域进行多视图同步、在 3D 点云空间而非 UV 空间进行补绘、以及空间感知的接缝平滑算法，在 Objaverse 和 GSO 两个 T2T 基准上全面超越 SOTA。

**[MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)**

:   本文提出MVSAnywhere (MVSA)，一个通用多视角立体匹配架构，通过Cost Volume Patchifier将代价体信息高效tokenize后与单目ViT特征融合（Mono/Multi Cue Combiner），结合视角数/尺度无关的元数据编码和级联自适应深度范围估计，在Robust MVS Benchmark上取得零样本SOTA，同时支持任意数量的源视角和任意深度范围。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   提出 Node-RF，将 Neural ODE 与动态 NeRF 紧密耦合，用潜在向量的 ODE 演化建模场景连续时间动力学，实现超出训练序列的长程时序外推和跨轨迹泛化，无需光流或深度监督。

**[P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)**

:   提出 P-SLCR，一种原型库驱动的无监督点云语义分割方法，通过将点分离为"一致"和"模糊"两类，用一致结构学习对齐一致点与原型 + 语义关系一致性推理约束两个原型库，在 S3DIS 上无监督达 47.1% mIoU，超越全监督 PointNet。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，首个在3D摄影测量空间进行全景拼接的 Transformer 框架，利用预训练 VGGT 骨干获取3D感知的多视角特征对齐 + 多特征联合优化接缝检测，支持2到数百张输入图像，在弱纹理/大视差/重复模式场景下成功率达97.8%。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，将隐式神经表示 (INR/SIREN) 与预训练扩散模型先验结合，通过 proximal loss 在每个 DDIM 时间步用扩散去噪输出正则化 INR 重建，在稀疏视角中子 CT（低至 4-5 个视角）上超越 FBP、纯 INR、DD3IP 和经典 MBIR(qGGMRF) 方法。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)**

:   Rewis3d 利用前馈 3D 重建（MapAnything）从 2D 视频中获取 3D 点云作为辅助监督信号，通过双 Student-Teacher 架构和加权跨模态一致性 (CMC) 损失，在仅使用稀疏标注（点/涂鸦/粗标记）的情况下将弱监督 2D 语义分割性能提升 2-7% mIoU，推理时仍为纯 2D。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一个即插即用的背景引导原型富化框架，在基类训练后用类无关分割模型从背景区域挖掘伪实例建立 Instance Prototype Bank (IPB)，当新类别以少样本方式出现时，通过 Contextual Prototype Retrieval (CPR) 和 Attention-Based Prototype Enrichment (APE) 融合背景原型与少样本原型，在 ScanNet/S3DIS 上新类 IoU 提升最高 6.98%。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   针对 3DGS 的资源瞄准攻击（通过投毒训练图像触发高斯过度增长导致资源耗尽），提出频域防御：3D 频率滤波器通过将高斯协方差与频谱响应关联实现频率感知剪枝，2D 频谱正则化通过熵惩罚渲染图像的角向能量各向异性来抑制攻击噪声，实现高斯数量压缩 5.92×、内存减少 3.66×、速度提升 4.34×。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。
