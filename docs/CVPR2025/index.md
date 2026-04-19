---
title: >-
  CVPR2025 1184篇论文解读
description: >-
  1184篇CVPR2025论文深度解读，每篇5分钟读懂核心思想，含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📷 CVPR2025 论文笔记

**1184** 篇论文解读，覆盖 **42** 个领域。

<div class="conf-index" markdown>

---

## 🧊 3D视觉 { #3d_vision }

**[3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](3d_vision/3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)**

:   构建了3D-GRAND——首个百万级**密集接地**的3D场景-语言数据集（40K场景、6.2M指令），并提出3D-POPE幻觉评估基准，证明密集接地的指令微调能显著提升3D-LLM的接地能力并减少幻觉，还展示了合成数据到真实场景的迁移效果。

**[3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d_vision/3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)**

:   提出3D-GSW，首个专为3D Gaussian Splatting设计的鲁棒数字水印方法，通过频率引导致密化（FGD）移除冗余高斯并在高频区域分裂高斯来增强鲁棒性，结合梯度掩码和小波子带损失保持渲染质量，在Blender/LLFF/Mip-NeRF 360数据集上同时实现了最优的水印鲁棒性和渲染质量。

**[3D-HGS: 3D Half-Gaussian Splatting](3d_vision/3d-hgs_3d_half-gaussian_splatting.md)**

:   提出3D Half-Gaussian (3D-HGS)核函数——用一个分割平面将3D高斯分成两半，每半有独立不透明度，作为**即插即用**的重建核替换标准高斯核，在不牺牲渲染速度的前提下显著提升形状和颜色不连续处的渲染质量，在Mip-NeRF360/T&T/Deep Blending上全面超越所有SOTA方法。

**[3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](3d_vision/3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)**

:   提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

**[3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](3d_vision/3d-mem_3d_scene_memory_for_embodied_exploration_and_reasoning.md)**

:   提出3D-Mem——基于"记忆快照"的3D场景记忆框架，用少量精选多视角图像紧凑表示已探索区域，结合Frontier Snapshot表示未探索区域，配合VLM实现高效的具身探索与推理。

**[3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d_vision/3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)**

:   提出3D-SLNR，一种超轻量神经3D表示——基于锚定在点云支撑点上的带限局部SDF集合定义全局SDF，每个局部SDF仅由一个微型MLP参数化（无隐特征），通过可学习的位置/旋转/缩放适应复杂几何，配合并行查找算法和剪枝-扩展策略，以不到先前方法1/5的内存实现SOTA重建质量。

**[3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)**

:   用3D光滑凸体（Smooth Convex）替代高斯基元进行辐射场渲染，通过点集定义凸包+LogSumExp平滑化+自定义CUDA光栅化器，在T&T和Deep Blending上超越3DGS，且所需基元更少。

**[3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_vision/3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)**

:   提出 CrossTooth，通过基于曲率先验的选择性下采样（边界区域顶点密度提升 10-15%）和多视角渲染图像的跨模态边界特征融合，在 3DTeethSeg'22 公开数据集上实现 95.86% mIoU 和 82.05% boundary IoU，分别比之前 SOTA（ToothGroupNet）提升 2.3% 和 5.7%。

**[3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations](3d_vision/3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)**

:   提出一种紧凑张量表示的3D高斯头部头像方法——用三平面存储中性表情的静态外观，用轻量1D特征线存储每个blendshape的动态纹理（不透明度偏移），仅需**10MB存储**即可实现300FPS实时渲染和准确的动态面部细节捕捉，在Nersemble数据集上PSNR和存储效率全面超越GA、GBS和GHA。

**[3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_vision/3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)**

:   提出3DGIC，通过**深度引导的跨视角一致修复**框架实现3D高斯场景中的物体移除与修补——利用渲染深度图从其他视角发现被掩码区域中的可见背景像素来精化修补掩码，再用参考视角的2D修补结果通过3D投影约束其他视角的一致性，在SPIn-NeRF数据集上FID和LPIPS全面超越现有方法。

**[3D Student Splatting and Scooping (SSS)](3d_vision/3d_student_splatting_and_scooping.md)**

:   提出SSS（Student Splatting and Scooping），用前所未有的三重创新改进3DGS范式：(1) 用**Student-t分布**替代高斯分布作为混合组件（可学习的尾部厚度，从Cauchy到Gaussian连续变化）；(2) 引入**负密度组件**（scooping减去颜色）扩展到非单调混合模型；(3) 用**SGHMC采样**替代SGD解耦参数优化，在Mip-NeRF360/T&T/Deep Blending上6/9指标取得最优，且参数效率极高——用**最少18%**的组件数即可匹配或超越3DGS。

**[3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3d_vision/3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)**

:   提出一个基于多视图潜在扩散模型的3D增强框架，通过姿态感知编码器、多视图行注意力和近视图极线聚合模块，在保持跨视图一致性的前提下显著提升低质量3D生成结果的纹理质量。

**[3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](3d_vision/3dgut_enabling_distorted_cameras_and_secondary_rays_in_gaussian_splatting.md)**

**[4Deform: Neural Surface Deformation for Robust Shape Interpolation](3d_vision/4deform_neural_surface_deformation_for_robust_shape_interpolation.md)**

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

**[4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians](3d_vision/4dtam_non-rigid_tracking_and_mapping_via_dynamic_surface_gaussians.md)**

:   本文提出了首个基于可微渲染和2D高斯表面基元的4D跟踪与建图方法（4DTAM），通过联合优化相机位姿、场景几何、外观和动态变形场，从单目RGB-D视频流实现非刚性动态场景的实时重建，并发布了全新的合成4D数据集Sim4D用于评估。

**[A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](3d_vision/a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)**

:   本文提出LoSF-UDF，一种基于局部形状函数学习无符号距离场（UDF）的轻量级框架，仅需在合成的局部点云patch上训练一次（653KB参数、0.5GB数据），即可泛化重建各种类型的3D表面，且对噪声和离群点具有鲁棒性。

**[A Unified Image-Dense Annotation Generation Model for Underwater Scenes](3d_vision/a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)**

:   本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。

**[ActiveGAMER: Active GAussian Mapping through Efficient Rendering](3d_vision/activegamer_active_gaussian_mapping_through_efficient_rendering.md)**

:   提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。

**[AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](3d_vision/aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)**

:   本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。

**[AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](3d_vision/anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)**

:   从单张图像生成可动画 3D 人体——先用适配的 CogVideo 生成多视角标准姿态图像（含法线），再将多视角不一致性建模为 4DGS 中的时序变化来提取一致的 canonical 空间高斯模型，最后通过 SMPL-X 蒙皮驱动动画。

**[Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](3d_vision/any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)**

:   提出Any3DIS，通过3D感知的2D掩码跟踪（利用SAM-2追踪每个超点在多帧中的2D分割）替代传统的无监督合并策略，并用动态规划优化3D Proposal，在ScanNet200和ScanNet++上的类别无关、开放词汇、开放式3D实例分割任务中均取得SOTA。

**[ARM: Appearance Reconstruction Model for Relightable 3D Generation](3d_vision/arm_appearance_reconstruction_model_for_relightable_3d_generation.md)**

:   提出ARM框架，将几何和外观生成解耦，在UV纹理空间中通过反投影和全局感受野网络重建高质量纹理，并引入材质先验解决稀疏视角下材质与光照的歧义性，仅用8张H100训练即在GSO和OmniObject3D上超越现有方法。

**[BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](3d_vision/bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)**

:   从错误分析角度重新审视3D语义分割，将分割误差分为四类（区域分类/位移/合并/误响应）并设计对应评估指标，提出BFANet通过边界-语义解耦模块和实时边界伪标签计算增强边界感知，在ScanNet200测试集上达到36.0 mIoU（不含辅助数据训练的最高成绩）。

**[Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](3d_vision/blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)**

:   提出一种基于新型图像块表示 Blurry-Edges 的深度估计方法，通过对散焦边界的平滑度建模，实现在极低光照（光子受限）条件下从一对不同散焦图像中鲁棒地估计物体深度，噪声鲁棒性比现有 DfD 方法高 4 倍以上。

**[CADDreamer: CAD Object Generation from Single-view Images](3d_vision/caddreamer_cad_object_generation_from_single-view_images.md)**

:   提出 CADDreamer，通过语义增强的多视图扩散模型和几何拓扑提取模块，从单张RGB图像直接生成具有紧凑B-rep表示、清晰结构和锐利边缘的CAD模型，支持平面、圆柱、圆锥、球体、环面五种基元类型。

**[Category-Agnostic Neural Object Rigging](3d_vision/category-agnostic_neural_object_rigging.md)**

:   提出 CANOR（Category-Agnostic Neural Object Rigging），通过将可变形4D物体编码为稀疏的空间定位 blob 集合和实例感知特征体，以完全类别无关、数据驱动的方式自动发现可变形物体的低维姿态空间，实现直观的姿态操控。

**[CMMLoc: Advancing Text-to-PointCloud Localization with Cauchy-Mixture-Model Based Framework](3d_vision/cmmloc_advancing_text-to-pointcloud_localization_with_cauchy-mixture-model_based.md)**

:   提出 CMMLoc，一个基于柯西混合模型（CMM）的不确定性感知文本-点云定位框架，通过将粗检索阶段建模为部分相关检索问题并引入 CMM Transformer 和方位整合模块，在 KITTI360Pose 数据集上实现 SOTA 性能。

**[COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](3d_vision/cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)**

:   提出 COB-GS，一种通过语义梯度统计驱动的边界自适应高斯分裂技术，联合优化语义信息和视觉纹理，解决 3DGS 分割中物体边界模糊的问题，在保持视觉质量的同时实现清晰的物体边界分割。

**[CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](3d_vision/cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)**

:   提出CoCoGaussian，利用物理摄影散焦原理（弥散圆/Circle of Confusion）在3D高斯溅射框架中建模散焦模糊，仅使用散焦图像即可精确重建3D场景并渲染清晰的新视角图像。

**[Coherent 3D Portrait Video Reconstruction via Triplane Fusion](3d_vision/coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)**

:   提出一种基于三平面融合（Triplane Fusion）的方法，将个人化3D先验与逐帧观测融合，在单目RGB视频中同时实现时间一致性和动态外观的忠实重建，用于3D远程呈现。

**[ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](3d_vision/colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)**

:   提出ColabSfM范式——通过3D点云配准（而非视觉描述子匹配）来融合分布式SfM重建结果，并构建了专用的SfM配准数据集生成管线和改进的配准模型RefineRoITr。

**[CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](3d_vision/comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)**

:   提出CoMapGS，利用像素级共视性图（covisibility map）来指导稀疏视角3DGS中初始点云增强和自适应加权监督，首次显式关注并恢复高不确定性的单视角区域。

**[CoMatcher: Multi-View Collaborative Feature Matching](3d_vision/comatcher_multi-view_collaborative_feature_matching.md)**

:   提出CoMatcher，一种多视角协同特征匹配器，从两视角独立匹配范式转向1-to-N协同匹配范式，利用互补视角的上下文线索和跨视角投影一致性约束来提升复杂场景下的匹配可靠性。

**[Compass Control: Multi Object Orientation Control for Text-to-Image Generation](3d_vision/compass_control_multi_object_orientation_control_for_text-to-image_generation.md)**

:   提出 Compass Control，通过引入轻量级方向编码器预测 compass token 并结合耦合注意力定位（CALL）机制，实现文本到图像扩散模型中多物体的精确3D方向控制，仅需合成数据训练即可泛化到未见类别和多物体场景。

**[Consistency-aware Self-Training for Iterative-based Stereo Matching](3d_vision/consistency-aware_self-training_for_iterative-based_stereo_matching.md)**

:   首次提出面向迭代式立体匹配的一致性感知自训练框架（CST-Stereo），通过多分辨率预测一致性滤波和迭代预测一致性滤波评估伪标签可靠性，结合软加权损失有效利用无标签真实数据提升模型性能和泛化能力。

**[Continuous 3D Perception Model with Persistent State](3d_vision/continuous_3d_perception_model_with_persistent_state.md)**

:   提出CUT3R（Continuous Updating Transformer for 3D Reconstruction），一个维持持续内部状态的循环模型，能从图像流中在线、增量地进行度量级3D重建、相机位姿估计，并能推断未观测区域的3D结构。

**[Cross-View Completion Models are Zero-shot Correspondence Estimators](3d_vision/cross-view_completion_models_are_zero-shot_correspondence_estimators.md)**

:   揭示跨视图补全（CVC）模型中交叉注意力图（cross-attention map）本质上学到了精确的稠密对应关系，提出ZeroCo在零样本匹配和学习型几何匹配中利用这一发现，显著超越基于编码器/解码器特征的常规用法。

**[CrossOver: 3D Scene Cross-Modal Alignment](3d_vision/crossover_3d_scene_cross-modal_alignment.md)**

:   提出CrossOver框架，通过维度特定编码器和三阶段训练管线，在不要求完整模态配对的条件下，学习RGB图像、点云、CAD模型、平面图和文本描述的统一场景级跨模态嵌入空间，支持灵活的跨模态检索和定位。

**[Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](3d_vision/ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)**

:   通过单张编辑参考图像微调 InstructPix2Pix 模型以"学习"编辑能力，结合两阶段可变形3D高斯优化，实现可控、一致的动态3D场景编辑。

**[DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](3d_vision/dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)**

:   提出 DAGSM，一种文本驱动的解耦数字人生成方法，通过 GS-enhanced Mesh（GSM）分别表示人体和各件衣物，支持换装、真实动画和纹理编辑。

**[DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](3d_vision/dashgaussian_optimizing_3d_gaussian_splatting_in_200_seconds.md)**

:   提出 DashGaussian，一种基于频率分析的渲染分辨率和高斯基元数量联合调度方案，将3DGS优化从逐步拟合高频分量的角度进行重新表述，平均加速 45.7% 且不牺牲渲染质量。

**[Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](3d_vision/decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)**

:   提出DP-Recon，将生成式扩散先验（SDS）引入分解式神经场景重建中，通过可见性引导动态调整逐像素SDS权重，解决重建目标与生成引导之间的冲突，实现稀疏视角下完整的物体几何与外观恢复。

**[DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](3d_vision/defom-stereo_depth_foundation_model_based_stereo_matching.md)**

:   将单目深度基础模型 (Depth Anything V2) 融入循环立体匹配框架 RAFT-Stereo，通过组合特征编码器和尺度更新模块，在保持强泛化能力的同时实现多个基准上排名第一的立体匹配性能。

**[Deformable Radial Kernel Splatting](3d_vision/deformable_radial_kernel_splatting.md)**

:   提出可变形径向核 (DRK) 来泛化传统高斯泼溅，通过可学习的径向基函数、$L_1$/$L_2$ 范数混合和边缘锐化机制，用更少的图元实现更高质量的3D场景渲染。

**[Dense-SfM: Structure from Motion with Dense Consistent Matching](3d_vision/dense-sfm_structure_from_motion_with_dense_consistent_matching.md)**

:   提出 Dense-SfM 框架，通过高斯泼溅进行轨迹扩展解决稠密匹配产生的碎片化轨迹问题，结合基于 Transformer 和高斯过程的多视图核化匹配精炼模块，实现高精度稠密 SfM 重建。

**[Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](3d_vision/depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)**

:   提出 Depth Any Camera (DAC) 框架，通过 ERP 统一表示、Pitch-aware 转换和 FoV 对齐等技术，实现仅用透视图像训练即可零样本泛化到鱼眼和360°相机的度量深度估计，在大视野数据集上 $\delta_1$ 精度提升高达50%。

**[DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](3d_vision/depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)**

:   利用预训练的视频扩散模型 (SVD) 进行视频深度估计，通过三阶段训练策略实现可变长度（最长110帧）的时间一致深度序列生成，并设计分段推理策略支持极长视频，在零样本设置下全面超越现有方法。

**[DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](3d_vision/depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)**

:   提出 DepthCues 基准，通过六个人类单目深度线索任务（高度、光影、遮挡、透视、大小、纹理梯度）系统评估 20 个大规模预训练视觉模型的深度感知能力，揭示了类人深度线索在现代视觉模型中的涌现现象。

**[DepthSplat: Connecting Gaussian Splatting and Depth](3d_vision/depthsplat_connecting_gaussian_splatting_and_depth.md)**

:   将高斯泼溅（3DGS）和深度估计两个通常独立研究的任务统一起来：利用预训练单目深度特征增强多视角深度模型以改善 3DGS 重建质量，同时用 3DGS 的光度渲染损失作为无监督预训练目标来学习强大的深度模型，双任务在多个数据集上均达到 SOTA。

**[DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](3d_vision/diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)**

:   提出 DiET-GS 双阶段框架，通过事件双积分（EDI）先验和预训练扩散模型联合约束 3DGS 优化，从模糊多视角图像和事件流中重建清晰的 3D 表示，实现精确色彩和精细细节的高质量新视角合成。

**[DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](3d_vision/diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)**

:   提出首个能从单张肖像生成一致的 360° 全头部视角的方法，通过双外观控制模块、背视图生成 ControlNet 和连续视角序列训练策略，支持真人、风格化和拟人化角色，并可转化为高质量 NeRF 进行实时自由视角渲染。

**[Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models](3d_vision/difix3d_improving_3d_reconstructions_with_single-step_diffusion_models.md)**

:   提出 Difix3D+，利用微调的单步扩散模型（SD-Turbo）在训练阶段渐进式生成伪训练视角回馈 3D 表示，并在推理阶段作为实时后处理增强器，同时兼容 NeRF 和 3DGS，在 FID 上平均实现 2 倍以上提升。

**[Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](3d_vision/digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)**

:   提出 DTC 数据集，包含 2000 个毫米级几何精度和光真实 PBR 材质的 3D 物体数字孪生模型，配合 DSLR 和自中心 AR 眼镜的多视角评估数据，建立了首个面向数字孪生创建任务的综合真实世界评测基准。

**[Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](3d_vision/disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)**

:   Disco4D 提出将服装（用 Gaussian 模型表示）与人体（用 SMPL-X 模型表示）解耦的 4D 人体生成框架，从单张图像生成可动画、可编辑、分层的3D穿衣人体模型，并支持逼真的4D服装动力学。

**[DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](3d_vision/dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)**

:   提出 DoF-Gaussian，为 3D 高斯表示引入基于几何光学的可学习透镜成像模型，通过逐场景深度先验调整和离焦-对焦自适应策略，实现从浅景深（散焦模糊）输入图像重建清晰 3D 场景，并支持可控景深渲染（重对焦、光圈调节、散焦形状变换等交互应用）。

**[Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](3d_vision/doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)**

:   提出 Doppelgangers++，通过引入多样化的 VisymScenes 日常场景训练数据和利用 MASt3R 多层解码器 3D 感知特征训练 Transformer 分类器，显著提升了 doppelganger（视觉混淆图像对）检测的精度和泛化性，并无缝集成到 COLMAP 和 MASt3R-SfM 管线中改善重复结构场景的 3D 重建质量。

**[Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](3d_vision/dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)**

:   提出 Dr. Splat，绕过渲染过程直接将语言对齐的 CLIP 嵌入注册到 3D 高斯上，结合在大规模图像数据上预训练的乘积量化（PQ）实现 6.25% 的嵌入压缩，在完全不需要逐场景优化的前提下（~10 分钟 vs 现有方法 1-24 小时），在开放词汇 3D 语义分割、3D 物体定位和 3D 物体选择任务上显著超越现有方法。

**[DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](3d_vision/dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)**

:   DroneSplat 提出了一个面向野外无人机影像的鲁棒 3DGS 框架，通过自适应局部-全局掩膜策略消除动态干扰物，结合基于多视图立体的几何感知点采样和体素引导优化策略解决有限视角下的重建质量问题，并提供了 24 个场景的无人机 3D 重建数据集。

**[DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](3d_vision/dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)**

:   DropGaussian 提出了一种无需额外先验的简单正则化方法，通过在 3DGS 训练中随机丢弃高斯并引入不透明度补偿因子，使被遮挡的远距离高斯获得更大梯度和可见性，并采用渐进式丢弃率策略有效缓解稀疏视角下的过拟合问题，在不增加计算复杂度的情况下达到与先验方法可比的性能。

**[DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](3d_vision/dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)**

:   DropoutGS 通过随机 Dropout 正则化（RDR）缓解稀疏视角 3DGS 的过拟合问题，再用边缘引导分裂策略（ESS）补偿低复杂度模型丢失的高频细节，作为即插即用模块可与多种 3DGS 方法结合，在 LLFF、DTU、Blender 上达到 SOTA。

**[DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](3d_vision/dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)**

:   DSPNet 提出了一种双视觉场景感知网络，通过文本引导的多视图融合（TGMF）、自适应双视觉感知（ADVP）和多模态上下文引导推理（MCGR）三个模块，综合利用点云和多视图图像信息来解决 3D 问答中的精细感知和鲁棒推理问题，在 SQA3D 和 ScanQA 数据集上达到 SOTA。

**[Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](3d_vision/dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)**

:   提出双曝光立体成像(Dual-Exposure Stereo)方法，通过自动双曝光控制(ADEC)在交替帧中使用不同曝光，结合运动感知的双曝光特征融合网络进行视差估计，将立体相机的有效动态范围扩展至 160%，实现极端光照条件下的鲁棒 3D 成像。

**[DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](3d_vision/dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)**

:   提出 Dual Point Maps (DualPM) 表示——从单张图像预测一对点图（相机空间 P + 规范空间 Q），将可变形物体的 3D 形状和姿态重建简化为点图预测问题，并引入分层 amodal 点图实现完整形状恢复（含自遮挡部分），仅用 1-2 个合成 3D 模型训练即可泛化到真实图像。

**[DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](3d_vision/dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)**

:   DUNE 提出了异构教师联合蒸馏（co-distillation）框架，将来自不同任务和数据域的 2D（DINOv2）与 3D（MASt3R、Multi-HMR）教师模型统一蒸馏为一个 ViT-Base 通用编码器，在语义分割、深度估计、3D 重建和人体姿态恢复等多任务上均达到或超越各自 ViT-Large 教师的性能。

**[Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](3d_vision/efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)**

:   提出 MultiHeadDepth 和 HomoDepth 两个模型，分别通过硬件友好的多头代价体积（LayerNorm+点积近似余弦相似度 + 分组点卷积）和单应性矩阵估计网络 + 2D 矫正位置编码 (RPE) 来优化立体深度估计中代价体积和预处理的延迟瓶颈，在 AR 眼镜场景下精度提升 11.8-30.3% 的同时端到端延迟降低 44.5%。

**[Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](3d_vision/efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)**

:   提出 Instruct-4DGS，利用 4D 高斯 (4DGS) 中静态 3D 高斯和 Hexplane 变形场的内在可分离性，仅编辑静态典范高斯即可实现高效动态场景编辑，再通过 Coherent-IP2P 驱动的分数蒸馏精炼时序对齐以消除运动伪影，将编辑时间缩短一半以上且仅需单 GPU。

**[EigenGS: Representation from Eigenspace to Gaussian Image Space](3d_vision/eigengs_representation_from_eigenspace_to_gaussian_image_space.md)**

:   本文提出 EigenGS，将经典 PCA 的特征空间表示与 2D 高斯 Splatting 图像表示相桥接，通过在特征基上学习统一的高斯参数实现新图像的即时初始化（无需从头优化），并引入频率感知学习机制避免高分辨率重建伪影，在收敛速度和最终质量上全面超越 GaussianImage。

**[Empowering Large Language Models with 3D Situation Awareness](3d_vision/empowering_large_language_models_with_3d_situation_awareness.md)**

:   本文提出利用 RGB-D 视频的相机轨迹自动生成情境感知（situation-aware）数据集 View2Cap（20 万+描述、55 万+ QA），并设计情境定位模块（SG）将位姿估计转为锚点分类任务，使 3D LLM 能理解第一人称视角下的空间关系描述（如"左边""右边"随视角变化），在 SQA3D 上 EM@1 达 54.0%。

**[End-to-End HOI Reconstruction Transformer with Graph-based Encoding](3d_vision/end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)**

:   提出 HOI-TG 框架，用 Transformer 的自注意力机制隐式学习人物交互关系，并在编码器中嵌入图残差模块分别增强人体和物体的拓扑结构建模，在 BEHAVE 和 InterCap 数据集上实现 SOTA 的 HOI 三维重建。

**[End-to-End Implicit Neural Representations for Classification](3d_vision/end-to-end_implicit_neural_representations_for_classification.md)**

:   提出 Meta Weight Transformer (MWT)，通过端到端元学习 SIREN 初始化参数和学习率调度，让 INR 的权重结构同时优化重建质量和分类性能，使用简单标准 Transformer 在 SIREN 权重上分类即可超越所有等变架构方法，首次在高分辨率 ImageNet-1K 上实现 INR 分类。

**[EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](3d_vision/envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)**

:   本文提出EnvGS，用一组环境高斯原语（Environment Gaussian）作为显式3D表示来捕获场景反射，通过基于GPU RT Core的可微光线追踪渲染器联合优化环境高斯和基础高斯，在真实场景中首次实现了实时（26+ FPS）且高质量的镜面反射新视角合成，显著超越所有实时方法。

**[ERUPT: Efficient Rendering with Unposed Patch Transformer](3d_vision/erupt_efficient_rendering_with_unposed_patch_transformer.md)**

:   ERUPT 提出了一种高效的潜在视角合成模型，通过 patch-based 解码器替代像素级解码、可学习的潜在相机位姿以及冻结 DINOv2 特征提取器，在不需要精确相机位姿的情况下仅用 5 张无位姿图像即可实现 600fps 的新视角合成，在 MSN 数据集上达到 SOTA 性能。

**[Estimating Body and Hand Motion in an Ego-sensed World](3d_vision/estimating_body_and_hand_motion_in_an_ego-sensed_world.md)**

:   EgoAllo 提出了一种从头戴设备的自中心 SLAM 位姿和图像估计佩戴者全身姿态、身高和手部参数的系统，通过设计满足空间和时间不变性的头部运动条件化参数，将人体运动估计误差降低高达 18%，并利用运动学约束将手部世界坐标误差降低 40%。

**[Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](3d_vision/event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)**

:   本文提出 Event Fields——一种利用事件相机捕获高速、高分辨率、高动态范围光场的新范式，设计了万花筒（空间复用，捕获时间导数）和振镜（时间复用，捕获角度导数）两种互补光学方案，实现了 250fps 百万像素动态场景重聚焦和 100Hz 实时深度估计等前所未有的能力。

**[EventFly: Event Camera Perception from Ground to the Sky](3d_vision/eventfly_event_camera_perception_from_ground_to_the_sky.md)**

:   EventFly 提出了首个事件相机跨平台域适应框架，通过事件激活先验（EAP）识别高激活区域、EventBlend 混合源/目标域事件数据、EventMatch 双判别器对齐特征分布，在车辆→无人机→四足机器人三个平台间的语义分割任务上，相比 source-only 训练平均提升准确率 23.8%、mIoU 77.1%。

**[Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](3d_vision/evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)**

:   本文提出CarGS，通过发现高斯基元对渲染和重建任务的贡献冲突根源在于协方差，设计了轻量残差结构Lite-Geo来自适应解耦两个任务的几何贡献，并引入法线+SDF双引导的致密化策略，在统一模型中同时实现SOTA的渲染质量和重建精度，且存储仅为双模型方法的9%。

**[Exploiting Deblurring Networks for Radiance Fields](3d_vision/exploiting_deblurring_networks_for_radiance_fields.md)**

:   本文提出DeepDeblurRF，首次将DNN去模糊网络引入辐射场构建流程，通过设计RF引导去模糊机制和迭代交替框架，在模糊图像输入下实现高质量新视角合成，训练速度比现有方法快10-100倍，同时支持体素网格和3D高斯溅射等多种3D表示。

**[Extreme Rotation Estimation in the Wild](3d_vision/extreme_rotation_estimation_in_the_wild.md)**

:   本文提出了一种面向真实互联网图像的极端三维旋转估计方法，构建了ExtremeLandmarkPairs (ELP)基准数据集，通过渐进式学习方案（全景裁剪→FoV+外观增强→真实数据微调）和辅助通道增强的Transformer模型，在无重叠视角的互联网图像对上显著超越现有方法。

**[Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](3d_vision/fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)**

:   提出 Fast3R，将 DUSt3R 的配对 pointmap 回归推广到多视图，通过 Transformer 的 all-to-all attention 在单次前向传播中处理 N 张无序无位姿图像，彻底消除了 $O(N^2)$ 配对推理和全局对齐优化。

**[FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection](3d_vision/faster_focal_token_acquiring-and-scaling_transformer_for_long-term_3d_objection_.md)**

:   本文提出FASTer，通过Adaptive Scaling机制自适应选取焦点token并压缩序列、分组层次融合策略渐进式聚合长时序点云信息，在Waymo Open Dataset上以最低延迟（75ms）和显存（2856M）取得了新SOTA性能。

**[Feature-Preserving Mesh Decimation for Normal Integration](3d_vision/feature-preserving_mesh_decimation_for_normal_integration.md)**

:   将经典的 quadric error metric（QEM）推导到屏幕空间并以法线贴图为输入，结合最优 Delaunay 三角化实现各向异性网格简化，在 90%+ 压缩率下仍保持亚毫米级精度，将高分辨率法线积分从小时级加速到分钟级。

**[Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](3d_vision/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)**

:   FADE 提出邻接感知（adjacency-aware）的细粒度概念擦除框架，通过 Concept Neighborhood 识别语义邻近类别并设计 Mesh Modules（Erasing + Adjacency + Guidance 三重损失），在精确删除目标概念的同时保留语义相关概念的生成能力，相比 SOTA 方法在邻接保留性能上提升至少 12%。

**[FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](3d_vision/flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)**

:   FLARE 提出级联学习范式（cascade learning），以相机位姿为桥梁将 3D 重建分解为位姿估计→局部几何→全局几何→高斯外观四个渐进阶段，在 0.5 秒内从 2-8 张未标定稀疏图像实现高质量的相机位姿、几何重建和新视角合成。

**[Floating No More: Object-Ground Reconstruction from a Single Image](3d_vision/floating_no_more_object-ground_reconstruction_from_a_single_image.md)**

:   提出 ORG 框架，首次从单张图像联合建模物体3D几何、相机参数和物体-地面关系，通过预测像素高度图和透视场两个紧凑的密集表示，解决了重建物体"悬浮/倾斜"的问题，显著提升阴影生成和姿态操控的真实感。

**[Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](3d_vision/flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)**

:   提出 Flow-NeRF，首次在无位姿 NeRF 框架中将场景几何、相机位姿和密集光流作为统一的联合优化目标，通过共享点采样、位姿条件化双射映射和特征消息传递机制，在新视角合成和深度估计上大幅超越先前方法，同时首次定义并实现了新视角光流估计。

**[Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution](3d_vision/flowing_from_words_to_pixels_a_noise-free_framework_for_cross-modality_evolution.md)**

:   提出 CrossFlow，一个通用的跨模态 Flow Matching 框架，直接从一种模态的数据分布演化到另一种模态的分布（而非从噪声出发），无需交叉注意力条件机制，在文本到图像生成上略优于标准 Flow Matching 基线，并展现出更好的模型规模和训练步数的缩放特性。

**[Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](3d_vision/floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)**

:   提出 Floxels，用简单的体素网格替代 MLP 作为场景流的隐式表示，结合多帧距离变换损失和聚类一致性约束，在 Argoverse 2 基准上取得仅次于 EulerFlow 的无监督方法第二名，同时将运行时间从一天缩短到10分钟（60-140倍加速）。

**[FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video](3d_vision/fluidnexus_3d_fluid_reconstruction_and_prediction_from_a_single_video.md)**

:   提出 FluidNexus，首次从单个视频实现3D流体外观和速度场的重建与未来预测，通过结合视频生成模型合成多视角参考视频，以及物理-视觉粒子耦合表示桥接可微分仿真与渲染，在新视角合成和未来预测上大幅超越现有多视角方法。

**[FoundationStereo: Zero-Shot Stereo Matching](3d_vision/foundationstereo_zero-shot_stereo_matching.md)**

:   提出 FoundationStereo，一个大规模立体深度估计基础模型，通过百万级高保真合成数据集、Side-Tuning Adapter 融合单目深度先验、以及混合代价体过滤（含 Axial-Planar Convolution 和 Disparity Transformer），实现了无需目标域微调的强零样本泛化性能。

**[FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation](3d_vision/foundhand_large-scale_domain-specific_learning_for_controllable_hand_image_gener.md)**

:   提出 FoundHand，一个在千万级手部数据集（FoundHand-10M）上训练的领域专用扩散模型，以 2D 关键点热力图为通用控制表示，实现精确的手部姿态/视角控制和外观保持，并展现出修复畸形手、视频生成、手物交互视频等零样本涌现能力。

**[FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT](3d_vision/framevggt_frame_evidence_rolling_memory_for_streaming_vggt.md)**

:   提出 FrameVGGT，将流式 VGGT 的 KV 缓存从 token 级保留重组为帧级证据块保留，通过中期记忆库+稀疏锚点的双层有界内存结构，在固定内存预算下保持更连贯的几何支撑，实现长序列3D重建/深度/位姿估计的精度-内存最优权衡。

**[FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](3d_vision/freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)**

:   提出 FreeGave，一个从多视角动态视频中学习 3D 场景几何、外观和物理速度的通用框架，通过为每个 3D 高斯核引入可学习的物理编码（physics code）并设计无散度（divergence-free）速度场参数化，在不依赖 PINN 损失和目标先验的条件下实现精准的未来帧外推。

**[FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts](3d_vision/freescene_mixed_graph_diffusion_for_3d_scene_synthesis_from_free_prompts.md)**

:   FreeScene 提出了一个用户友好的室内场景合成框架，通过 VLM 驱动的 Graph Designer 将自由形式的文本/图像输入转化为场景图，再用 Mixed Graph Diffusion Transformer (MG-DiT) 在混合连续-离散空间上进行图感知去噪，统一支持 text-to-scene、graph-to-scene 等多种任务，在生成质量和可控性上均超越现有方法。

**[FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](3d_vision/fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)**

:   FruitNinja 首次提出为 3DGS 物体生成内部纹理的方法，通过渐进式截面修复 + 体素平滑 + OpaqueAtom GS 策略，实现切割后实时渲染无需额外优化，在语义对齐和纹理一致性上显著优于基线。

**[FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](3d_vision/fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)**

:   FSFM 提出首个面向人脸安全任务的自监督预训练框架，通过 CRFR-P 面部掩码策略 + MIM/ID 双任务协同学习真实人脸的 3C 表示（区域内一致性、区域间连贯性、局部到全局对应性），在深伪检测、活体检测和扩散伪造检测三大任务上超越任务专用 SOTA。

**[FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](3d_vision/fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)**

:   FSHNet 提出全稀疏混合网络，通过 SlotFormer（槽分区+线性注意力）建立全局范围的稀疏体素交互，配合动态稀疏标签分配和稀疏上采样模块，在 Waymo、nuScenes、Argoverse2 三大基准上超越现有稀疏和密集检测器。

**[Functionality Understanding and Segmentation in 3D Scenes](3d_vision/functionality_understanding_and_segmentation_in_3d_scenes.md)**

:   Fun3DU 首次提出针对 3D 场景功能性理解的方法，通过 LLM 链式思维解析任务描述 + VLM 多视角分割功能性物体 + 2D-3D 投票聚合，在 SceneFun3D 上大幅超越开放词汇 3D 分割基线（mIoU +13.2）。

**[GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](3d_vision/ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)**

:   提出 GA3CE 方法，通过将主体 3D 姿态和场景物体位置编码到以主体为中心的自我中心空间中，并设计方向-距离分解的 D3 位置编码，在 Transformer 中学习 3D 注视方向与场景上下文的空间关系，在无约束设置下将 3D 注视角度误差降低 13%–37%。

**[GASP: Gaussian Avatars with Synthetic Priors](3d_vision/gasp_gaussian_avatars_with_synthetic_priors.md)**

:   提出 GASP，利用合成数据训练 Gaussian Avatar 的生成式先验模型（auto-decoder），通过三阶段拟合过程和学到的 per-Gaussian 语义特征关联来跨越合成-真实域差距，仅从单张图片或短视频即可创建支持 360° 渲染的高质量实时可动画头像（70fps）。

**[GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](3d_vision/gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)**

:   提出 GaussHDR，通过统一 3D 和 2D 局部色调映射来改进 HDR 高斯溅射，设计残差局部色调映射器和不确定性自适应调制机制，同时提升 HDR 重建稳定性和 LDR 拟合质量，在合成和真实场景上大幅超越现有方法。

**[Gaussian Eigen Models for Human Heads](3d_vision/gaussian_eigen_models_for_human_heads.md)**

:   提出 Gaussian Eigen Models (GEM)，通过 PCA 将高质量 CNN-based Gaussian Avatar 蒸馏为轻量级线性特征基表示，仅需低维系数的线性组合即可生成面部动画，实现高质量、超轻量（7MB起）和超快速（200+ fps）的可动画头像，并支持从单目视频的实时跨人表情驱动。

**[Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](3d_vision/gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)**

:   提出 Gaussian Splatting Feature Fields (GSFFs)，将 3DGS 的显式几何与隐式特征场结合，通过自监督对比学习训练尺度感知的 3D 特征和 2D 编码器，并利用基于 Delaunay 图的空间聚类将特征转化为分割标签，实现了高精度的非隐私和隐私保护视觉定位。

**[Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](3d_vision/gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)**

:   本文提出 EOGS，首个基于 3D 高斯溅射的地球观测框架，通过仿射相机近似、阴影映射和三种正则化策略，在卫星图像三维重建任务上达到与 EO-NeRF 相当的精度，同时训练速度快 300 倍（3 分钟 vs 15 小时）。

**[GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting](3d_vision/gaussianudf_inferring_unsigned_distance_functions_through_3d_gaussian_splatting.md)**

:   本文提出 GaussianUDF，通过将 2D 高斯平面贴合到曲面上，利用自监督和梯度推断为近场和远场分别提供无符号距离监督，首次在 3DGS 框架内高效推断连续 UDF，实现高质量开放曲面重建。

**[GauSTAR: Gaussian Surface Tracking and Reconstruction](3d_vision/gaustar_gaussian_surface_tracking_and_reconstruction.md)**

:   GauSTAR 提出一种将高斯原语绑定到网格面上的"高斯曲面"表示，通过自适应解绑和重网格化机制处理拓扑变化，配合基于曲面的场景流初始化，首次实现了动态场景中同时兼顾照片级渲染、精确曲面重建和可靠三维跟踪的统一框架。

**[GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](3d_vision/geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)**

:   GEAL 提出双分支架构，用 3D 高斯溅射将点云渲染为逼真 2D 图像从而利用预训练 2D 基础模型的泛化能力，通过粒度自适应融合和 2D-3D 一致性对齐实现跨模态知识迁移，在标准和腐败数据基准上全面超越现有 3D 功能可供性方法。

**[Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](3d_vision/gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)**

:   本文提出Gen3DEval，一个基于vLLM微调的text-to-3D生成质量评估框架，通过对Llama3模型在合成+人工标注数据上微调，实现对3D物体外观、表面质量和文本一致性的自动评估，在与人类偏好对齐上显著超越GPT-4o等通用模型。

**[Generating 3D-Consistent Videos from Unposed Internet Photos](3d_vision/generating_3d-consistent_videos_from_unposed_internet_photos.md)**

:   本文提出KFC-W，一种从无位姿互联网照片生成3D一致视频的自监督方法，通过在视频扩散模型上联合训练多视角修复和视角插值两个目标，无需任何3D标注（如相机参数），生成的视频在几何和外观一致性上超越商业模型Luma Dream Machine。

**[Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](3d_vision/generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)**

:   本文提出先用多视图重光照扩散模型将不同光照下拍摄的图像统一到参考光照条件，再用带有"shading embedding"的鲁棒 NeRF 模型重建 3D 表示，在极端光照变化下实现了远超现有方法的高保真外观重建，尤其擅长恢复镜面/高光效果。

**[Generative Omnimatte: Learning to Decompose Video into Layers](3d_vision/generative_omnimatte_learning_to_decompose_video_into_layers.md)**

:   Generative Omnimatte 通过微调视频 inpainting 扩散模型（Casper）学会物体及其关联效果（阴影、反射）的联合移除，结合 trimask 条件和 omnimatte 优化，在无需静态背景假设或相机位姿的前提下实现了高质量的视频图层分解和被遮挡区域补全。

**[GenFusion: Closing the Loop between Reconstruction and Generation via Videos](3d_vision/genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)**

:   提出 GenFusion，通过重建驱动的视频扩散模型修复 3D 重建伪影并生成不可见区域内容，设计循环融合管线迭代地将生成结果加入训练集，实现稀疏视图下高质量 3D 场景重建和内容扩展。

**[GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](3d_vision/genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)**

:   提出 GenPC 零样本点云补全框架，通过 Depth Prompting 模块将部分点云转化为深度图再生成 RGB 图像作为 Image-to-3D 模型的输入，再通过 Geometric Preserving Fusion 模块将生成的 3D 形状与原始点云对齐融合，实现了比 SDS 优化方法更快更好的真实世界扫描补全。

**[GenVDM: Generating Vector Displacement Maps From a Single Image](3d_vision/genvdm_generating_vector_displacement_maps_from_a_single_image.md)**

:   提出首个从单张图像生成 Vector Displacement Map (VDM) 的方法，通过微调 Zero123++ 生成多视角法线图、使用神经 SDF 重建网格、再用神经变形场参数化为 VDM 图像，并构建了首个学术 VDM 数据集，为 3D 艺术家提供了按需生成自定义几何细节印章的能力。

**[Geometry in Style: 3D Stylization via Surface Normal Deformation](3d_vision/geometry_in_style_3d_stylization_via_surface_normal_deformation.md)**

:   提出通过优化三角网格的表面法线方向、结合可微分ARAP（dARAP）层恢复顶点位置的方法，实现文本驱动的网格风格化，能产生表达力强但保持形状身份的几何变形。

**[GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](3d_vision/gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)**

:   提出GIFStream，一种基于canonical空间+变形场的4D高斯表示方法，通过为每个anchor点附加时间相关的特征流（feature stream）来增强复杂运动建模能力，同时利用时间对齐的结构和端到端压缩实现30 Mbps高质量1080p沉浸式视频。

**[Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](3d_vision/global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)**

:   提出全局-局部树搜索算法，利用VLM的空间推理能力，通过层次化场景表示和emoji网格的视觉提示，实现高质量3D室内场景布局生成，在用户研究中平均排名第一。

**[Glossy Object Reconstruction with Cost-effective Polarized Acquisition](3d_vision/glossy_object_reconstruction_with_cost-effective_polarized_acquisition.md)**

:   提出一种低成本偏振辅助3D重建方法，仅需在普通RGB相机前加一块线性偏振片，每视角拍摄一张偏振图像（无需校准偏振角），通过神经隐式场端到端优化偏振渲染损失来恢复光泽物体的高保真几何和材质分解。

**[GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector](3d_vision/go-n3rdet_geometry_optimized_nerf-enhanced_3d_object_detector.md)**

:   提出GO-N3RDet，通过位置信息嵌入的体素优化模块（PEOM）、双重重要性采样（DIS）和不透明度优化模块（OOM）三个协同模块，解决基于NeRF的多视图3D检测中缺乏3D位置信息和场景几何感知不足的问题，在ScanNet和ARKitScenes上建立了新SOTA。

**[GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](3d_vision/great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)**

:   提出 GREAT 框架，通过多头 Affordance Chain-of-Thought (MHACoT) 微调 InternVL 推理交互图像中的物体几何属性和潜在交互意图，形成 affordance 知识字典，并通过跨模态自适应融合模块（CMAFM）将知识注入点云和图像特征，实现开放词汇 3D 物体 affordance 定位。同时构建最大规模 3D affordance 数据集 PIADv2（15K 图像 + 38K 点云）。

**[Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](3d_vision/grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)**

:   提出首个多模态多视角 3D 功能区域定位任务和 AGPIL 数据集（30,972 对点云-图像-语言三元组），并设计基于 VLM 的 LMAffordance3D 框架，融合 2D/3D 空间特征与语言语义实现从 full-view 到 partial/rotation-view 的泛化。

**[GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction](3d_vision/gs-2dgs_geometrically_supervised_2dgs_for_reflective_object_reconstruction.md)**

:   在 2DGS 基础上引入基础模型（Marigold + Depth Pro）的深度/法线伪标签监督和延迟着色（Deferred Shading）的物理渲染管线，在反射物体重建上显著超越 GS 方法、媲美 SDF 方法且快了一个数量级。

**[GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](3d_vision/guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)**

:   提出 GuardSplat，通过 CLIP 引导的消息解耦优化（仅训练解码器 5 分钟）和 SH-aware 水印嵌入（仅修改球谐偏移量），实现对 3DGS 资产的高容量、高保真、鲁棒版权保护，总优化时间仅 15 分钟。

**[HandOS: 3D Hand Reconstruction in One Stage](3d_vision/handos_3d_hand_reconstruction_in_one_stage.md)**

:   HandOS 提出了一个端到端的单阶段3D手部重建框架，通过冻结预训练检测器并引入交互式2D-3D解码器，将手部检测、2D姿态估计和3D mesh重建统一到一个pipeline中，消除了传统多阶段方法的冗余计算和累积误差，在 FreiHand 上达到 5.0 PA-MPJPE 的 SOTA 性能。

**[Hardware-Rasterized Ray-Based Gaussian Splatting](3d_vision/hardware-rasterized_ray-based_gaussian_splatting.md)**

:   本文提出了首个基于硬件光栅化的射线型3D高斯泼溅（RayGS）渲染方案 VKRayGS，通过严谨的数学推导在3D空间中构建最小包围四边形，实现了约40倍的渲染加速，同时保持了RayGS的高质量渲染效果，并额外提出了RayGS的MIP抗锯齿方案。

**[Hash3D: Training-free Acceleration for 3D Generation](3d_vision/hash3d_training-free_acceleration_for_3d_generation.md)**

:   Hash3D 发现 SDS 优化过程中扩散模型对相邻相机位姿和时间步的特征高度冗余，通过自适应网格哈希表缓存和复用中间特征，在无需训练的情况下将多种text-to-3D和image-to-3D方法加速1.3~4倍，同时还提升了多视角一致性。

**[HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](3d_vision/hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)**

:   HaWoR 首次实现了从第一人称视频重建世界坐标系下的3D手部运动，通过将任务解耦为相机空间手部重建 + 自适应SLAM相机轨迹估计，并引入运动补全网络处理手部出视野的情况，在 HOT3D 数据集上取得 SOTA 的全局轨迹精度（ATE 3.36mm）和手部重建质量（PA-MPJPE 4.79mm）。

**[HD-EPIC: A Highly-Detailed Egocentric Video Dataset](3d_vision/hd-epic_a_highly-detailed_egocentric_video_dataset.md)**

:   HD-EPIC 提供了41小时非脚本厨房第一人称视频，具有前所未有的标注密度（每分钟263条标注），涵盖食谱步骤、细粒度动作、营养信息、3D数字孪生、物体运动轨迹和注视方向等多层级互联标注，并构建了26K问题的VQA基准——最强的 Gemini Pro 仅达37.6%。

**[Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes](3d_vision/hearing_hands_generating_sounds_from_physical_interactions_in_3d_scenes.md)**

:   本文提出通过在3D重建场景中记录人手交互的动作-声音对，训练基于rectified flow的生成模型，实现从3D手部轨迹预测对应交互声音，生成结果在人类评估中约47%无法与真实声音区分。

**[HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](3d_vision/heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)**

:   提出HeatFormer——一种基于Transformer的神经优化器，通过将SMPL参数估计转化为热力图生成与对齐问题，实现对多视角图像中人体形状和姿态的迭代优化恢复，在Human3.6M上达到29.5mm MPJPE的SOTA精度，且对视角数量、相机配置和遮挡具有强鲁棒性。

**[High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](3d_vision/high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)**

:   GS-RGBN 提出混合 Voxel-Gaussian 表示为无结构高斯提供 3D 空间约束，并设计跨体积融合（CVF）模块在特征层面融合 RGB 语义信息和法线几何信息，从单张图像在数秒内生成高保真 3D 对象，在 GSO 数据集上 PSNR 超出次优方法 5.59dB。

**[HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)**

:   提出 HOI3DGen 框架，通过MLLM自动标注高质量交互数据 + 视角条件化微调扩散模型 + 3D提升与SMPL配准，首次实现从文本精确控制接触语义的高质量3D人物交互生成，在文本一致性上超越基线4-15倍。

**[Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](3d_vision/horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)**

:   本文提出 Horizon-GS，通过粗到精两阶段训练策略、相机分布平衡机制和多分辨率 LOD 结构，首次实现了航空视角和街景视角的统一 3D 高斯溅射重建和实时渲染，在多个城市场景数据集上达到 SOTA 渲染质量。

**[HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](3d_vision/hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)**

:   Meta 发布 HOT3D，首个基于真实头戴设备（Project Aria + Quest 3）的大规模自中心多视角手物交互数据集，包含 833 分钟 370 万+图像、19 名受试者与 33 个物体的交互，并通过实验证明多视角方法在 3D 手部追踪、6DoF 物体位姿估计等任务上显著优于单视角方法。

**[HRAvatar: High-Quality and Relightable Gaussian Head Avatar](3d_vision/hravatar_high-quality_and_relightable_gaussian_head_avatar.md)**

:   HRAvatar 提出了一种基于3DGS的单目视频头部重建方法，通过可学习blendshapes和LBS实现灵活变形，结合端到端表情编码器减少追踪误差，并引入物理渲染模型实现高质量实时重光照。

**[Hybrid eTFCE-GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](3d_vision/hybrid_etfce-grf_exact_cluster-size_retrieval_with_analytical_p-values_for_voxel.md)**

:   将 eTFCE 的并查集精确聚类大小查询与 pTFCE 的解析 GRF p 值推断结合，首次在单一框架中实现精确聚类检索+无需置换检验的统计推断，速度比置换 TFCE 快 1300 倍，在全脑体素形态测量中保持严格 FWER 控制。

**[HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](3d_vision/hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)**

:   HybridGS首次提出混合2D+3D高斯表示，用多视角一致的3D高斯建模静态场景、用单视图独立的2D高斯建模瞬态物体，配合多视角监督和多阶段训练实现了含干扰元素场景下SOTA的新视角合成质量。

**[HyperGS: Hyperspectral 3D Gaussian Splatting](3d_vision/hypergs_hyperspectral_3d_gaussian_splatting.md)**

:   首次将 3DGS 成功扩展到高光谱新视角合成（HNVS），通过在学习的潜在空间中进行高光谱渲染，配合自适应密度控制与像素级光谱剪枝，实现高维光谱数据的高效准确重建。

**[IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](3d_vision/iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)**

:   构建基于 3DGS 的层次化语义特征场，融合 CLIP/SAM/DINOv2 的语义信息，实现铰接物体的交互式 affordance 预测和跨状态运动参数恢复，支持任意类别、多可动部件的复杂室内场景。

**[Identity-preserving Distillation Sampling by Fixed-Point Iterator](3d_vision/identity-preserving_distillation_sampling_by_fixed-point_iterator.md)**

:   提出 Identity-preserving Distillation Sampling (IDS)，通过不动点迭代正则化（FPR）修正文本条件分数函数中导致身份丢失的梯度误差，生成引导噪声替代随机噪声，在 2D 图像编辑和 3D NeRF 编辑中实现结构和姿态的高度保持。

**[IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](3d_vision/imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)**

:   本文提出IMFine，一种用于无约束场景（包括360°环绕）的3D修复流水线，通过几何先验引导的warping和基于测试时适应的多视角refinement网络生成多视角一致的修复图像，并提出了一种新的修复mask检测技术来精确区分真正需要修复的遮挡区域，在多样化的benchmark上显著超越现有方法。

**[Improving Gaussian Splatting with Localized Points Management](3d_vision/improving_gaussian_splatting_with_localized_points_management.md)**

:   本文提出局部化点管理（LPM）策略，通过多视角几何约束定位导致渲染误差的 3D 区域，在这些区域内执行针对性的点加密和不透明度重置，作为即插即用模块可提升多种 3DGS 模型的重建质量，同时保持实时渲染速度。

**[IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](3d_vision/inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)**

:   本文提出 IncEventGS，首个仅用单目事件相机、无需已知位姿即可增量重建 3D 高斯溅射场景的方法，采用跟踪-建图 SLAM 范式联合优化相机运动和场景表示，在新视角合成和位姿估计上均超越现有方法。

**[Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](3d_vision/instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)**

:   将 3D 编辑问题转化为多视角一致的 2D inpainting 问题，通过微调 SDXL-inpainting 模型在 2×2 视角网格上同时生成一致的填充内容，再用 LRM 重建 3D，实现约 3 秒完成高质量 3D 编辑——比 SDS 方法快数百倍。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)**

:   提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模进行多曝光融合 + MetaNet 预测场景自适应色调映射器，从未标定多曝光 LDR 图像一次前向推理重建 HDR 3D 高斯，速度比优化方法快 ~700 倍。

**[InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](3d_vision/interactvlm_3d_interaction_reasoning_from_2d_foundational_models.md)**

:   InteractVLM 利用大规模视觉语言模型(VLM)的广泛视觉知识，通过"渲染-定位-提升"(Render-Localize-Lift)框架将2D基础模型的推理能力迁移到3D空间，实现了从单张野外图像估计人体和物体3D接触点，并用于人物交互联合重建，在接触估计任务上F1分数提升20.6%。

**[IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](3d_vision/irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)**

:   本文提出IRGS框架，首次在高斯泼溅中集成完整渲染方程（无简化），通过提出的可微分2D高斯光线追踪技术实时计算入射光的可见性和间接辐射，在多个逆渲染基准上取得了显著优于先前方法的重光照和材质估计效果。

**[IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](3d_vision/iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)**

:   IRIS提出了一个从多视角LDR图像中联合恢复HDR光照、物理材质和相机响应函数的逆渲染框架，通过显式建模色调映射、自动检测发光体和迭代优化策略，在真实和合成室内场景上实现了高质量的材质估计、重光照和虚拟物体插入。

**[iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](3d_vision/isegman_interactive_segment-and-manipulate_3d_gaussians.md)**

:   iSegMan提出了一个无需场景特定训练的交互式3DGS分割与操作框架，通过极线引导的交互传播(EIP)和基于可见性的高斯投票(VGV)实现精确的3D区域控制，配合完整的操作工具箱支持语义编辑、上色、缩放、复制粘贴、组合和删除等多种功能。

**[Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](3d_vision/joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)**

:   将相机运动建模为时间连续的角速度和线速度，通过速度积分避免直接优化大范围 camera-to-world 变换，结合时间依赖 NeRF 和 SDF flow 约束，无需深度先验即可从单目视频联合优化位姿和场景几何。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](3d_vision/jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)**

:   提出 JOPP-3D 框架，通过将全景图切线分解为透视图像、利用 SAM+CLIP 进行3D实例-语义对齐，首次实现对3D点云和全景图像的联合开放词汇语义分割，在 Stanford-2D-3D-s 和 ToF-360 数据集上超越现有方法。

**[Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](3d_vision/kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)**

:   将 3D 资产生成转化为 2D 图像生成问题——微调 Flux DiT 模型生成"3D Bundle Image"（四视图 RGB + 法线图拼贴），再用 ISOMER 重建 3D mesh，并通过 ControlNet 扩展支持 3D 增强和编辑。

**[Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](3d_vision/layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)**

:   本文提出 Layered Motion Fusion（LMF），将 2D 运动分割模型的预测融合到分层神经辐射场的动态和半静态层中，结合测试时精修策略，首次证明 3D 方法能在第一人称视频的动态目标分割中超越 2D 基线，动态目标分割 mAP 提升 30.5%。

**[Learnable Infinite Taylor Gaussian for Dynamic View Rendering](3d_vision/learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)**

:   提出可学习无穷 Taylor 级数（Learnable Infinite Taylor Formula）建模动态场景中高斯基元的位置/旋转/缩放随时间的演化，用三阶 Taylor 展开捕捉大运动、MLP+LBS 构造 Peano 余项补偿高阶项，实现无近似误差的运动建模，N3DV 和 Technicolor 数据集上超越 SOTA。

**[Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](3d_vision/learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)**

:   提出首个统一室内外稀疏监督 3D 目标检测方法 CPDet3D，通过类感知原型聚类（跨场景 Sinkhorn-Knopp 最优传输匹配）挖掘未标注物体的类别，再用多标签协同精化（伪标签 + 原型标签）恢复漏检，仅用每场景 1 个标注即达 ScanNet V2 全监督 78% / SUN RGB-D 90% / KITTI 96% 性能。

**[Light3R-SfM: Towards Feed-forward Structure-from-Motion](3d_vision/light3r-sfm_towards_feed-forward_structure-from-motion.md)**

:   Light3R-SfM提出了首个前馈式端到端SfM框架，通过可学习的潜在全局对齐模块替代传统的优化式全局对齐，结合基于检索分数的最短路径树构建场景图，在Tanks&Temples 200图设置下仅需33秒完成重建（比MASt3R-SfM快49倍），同时保持相当的精度。

**[LIM: Large Interpolator Model for Dynamic Reconstruction](3d_vision/lim_large_interpolator_model_for_dynamic_reconstruction.md)**

:   提出 LIM——首个前馈式跨类别动态 4D 资产重建模型，通过在隐式 triplane 表示间进行 Transformer 插值并引入因果一致性损失，实现秒级高质量连续时间插值与一致拓扑的网格跟踪。

**[LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene (FA-NeRF)](3d_vision/lookcloser_frequency-aware_radiance_field_for_tiny-detail_scene.md)**

:   FA-NeRF 提出一种频率感知的神经辐射场框架，通过 3D 频率量化方法分析场景频率分布，结合频率网格、频率感知特征重加权和自适应光线行进，在单一模型中同时捕捉场景整体结构和高清微小细节，在多频率数据集上显著超越所有基线方法。

**[LT3SD: Latent Trees for 3D Scene Diffusion](3d_vision/lt3sd_latent_trees_for_3d_scene_diffusion.md)**

:   提出 LT3SD，将 3D 场景渐进分解为潜在树（每层包含几何体积 + 高频潜在特征体积），在此表征上训练基于 patch 的扩散模型，实现从粗到细、逐 patch 的高质量无限 3D 场景生成，FID 相对 SOTA 提升 70%。

**[LUCAS: Layered Universal Codec Avatars](3d_vision/lucas_layered_universal_codec_avatars.md)**

:   提出 LUCAS，首个将人脸和头发解耦为分层 mesh 的通用先验 Avatar 模型，通过共享表情编码 + 独立解码实现自然的面部-头发交互，同时支持实时 mesh 渲染（45 FPS mobile）和高保真 Gaussian 渲染，在跨身份零样本驱动中达到 SOTA。

**[MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](3d_vision/mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)**

:   提出 MAC-Ego3D 框架，通过统一的 3D 高斯泼溅（Gaussian Splatting）表示让多个智能体独立构建、对齐和迭代优化局部地图，利用智能体内和智能体间高斯共识机制实现实时协作位姿估计和逼真 3D 重建，达到 15 倍推理加速、位姿误差降低一个数量级、RGB PSNR 提升 4-10 dB。

**[MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](3d_vision/magic-slam_multi-agent_gaussian_globally_consistent_slam.md)**

:   提出基于刚性可变形3D高斯场景表示的多智能体SLAM系统MAGiC-SLAM，通过新颖的追踪、地图融合机制和基于DinoV2的回环检测，实现了比CP-SLAM快24倍的处理速度、7倍低的GPU占用，以及更精确的轨迹估计和高保真新视角渲染。

**[Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](3d_vision/mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)**

:   Mani-GS 提出了一种基于三角网格操控 3D Gaussian Splatting 的方法——通过在每个三角形上定义局部坐标系来绑定高斯，使得网格变形时高斯的位置、旋转和缩放能自适应调整，从而实现大变形、局部编辑和软体仿真等多种操控类型，同时保持高质量渲染且对网格精度有高容忍度。

**[ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](3d_vision/manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)**

:   提出多层遮挡（MLO）表示学习 3D 手-物遮挡关系，并将 Objaverse 大规模 3D 物体数据整合进训练，实现首个支持灵巧双手操作 + 可泛化物体外观的手-物操作视频生成框架。

**[MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation](3d_vision/mar-3d_progressive_masked_auto-regressor_for_high-resolution_3d_generation.md)**

:   提出金字塔 VAE + 级联 MAR（MAR-LR → MAR-HR）的渐进式 3D 生成框架，通过随机遮罩适配 3D token 的无序特性，并用条件增强策略缓解分辨率上扩展时的累计误差，在开源方法中达到 SOTA。

**[MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation](3d_vision/marvel-40m_multi-level_visual_elaboration_for_high-fidelity_text-to-3d_content_c.md)**

:   构建了包含 890 万 3D 资产、4000 万+多层级文本标注的大规模 3D 描述数据集 MARVEL-40M+，通过多阶段自动标注管线（InternVL2 + Qwen2.5）生成从详细描述到简洁标签的五级标注，并基于此微调 SD3.5 实现 15 秒内的高保真文本到 3D 生成。

**[Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](3d_vision/masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)**

:   提出 MPEC（Masked Point-Entity Contrastive learning），通过跨视角 point-to-entity 对比学习和 entity-to-language 对齐两个层次的对比损失来训练 3D 编码器，在保持实体级几何-空间信息的同时实现开放词汇语义理解，在 ScanNet 上取得 66.0% f-mIoU 的 SOTA 并在 8 个数据集的下游任务上展现强泛化能力。

**[MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](3d_vision/maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)**

:   将 3DGS 中的高斯剪枝从确定性移除改为概率性存在建模，提出 masked-rasterization 技术使未被采样的高斯仍能接收梯度以动态评估其贡献，在 Mip-NeRF360/T&T/DeepBlending 上实现 62-75% 的高斯剪枝率且仅损失 0.02 PSNR。

**[MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](3d_vision/mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)**

:   首个以双视图 3D 重建先验 MASt3R 为基础构建的实时单目稠密 SLAM 系统，通过高效的点图匹配、光线误差跟踪、局部融合、回环检测和二阶全局优化，在无需相机标定的情况下实现 15 FPS 的全局一致位姿估计和稠密几何重建，性能达到 SOTA。

**[MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](3d_vision/matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)**

:   提出 MAtCha Gaussians，将场景表面建模为 2D 流形上的图表集合（Atlas of Charts）并用 2D Gaussian Surfels 渲染，通过单目深度初始化 + 轻量神经变形模型 + 结构保持损失，在仅 3-10 张稀疏视图下数分钟内同时实现高质量表面网格重建和逼真新视角合成。

**[Material Anything: Generating Materials for Any 3D Object via Diffusion](3d_vision/material_anything_generating_materials_for_any_3d_object_via_diffusion.md)**

:   提出 Material Anything，一个全自动的统一扩散框架，通过三头 U-Net 架构、置信度掩码和渲染损失适配预训练图像扩散模型生成 PBR 材质（albedo/roughness/metallic/bump），配合置信度引导的渐进式多视角生成策略和 UV 空间精化模型，为不同光照条件（无纹理/纯 albedo/扫描/生成）下的 3D 物体统一生成高质量材质贴图。

**[Matrix3D: Large Photogrammetry Model All-in-One](3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)**

:   Matrix3D 提出一个基于多模态扩散 Transformer 的统一摄影测量模型，通过掩码学习策略在单一模型中同时完成位姿估计、深度预测和新视角合成三大任务，在 CO3D 上位姿估计旋转精度达 96.5%，显著超越所有专用方法。

**[MEGA: Masked Generative Autoencoder for Human Mesh Recovery](3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)**

:   MEGA 提出了一种基于遮掩生成建模的人体网格恢复方法，通过将人体 mesh 离散化为 token 序列，在自监督预训练后进行图像条件生成，同时支持确定性单次预测和随机多输出生成模式，在两种模式下均达到 SOTA 性能。

**[MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](3d_vision/megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)**

:   MegaSaM 通过在深度视觉 SLAM 框架中集成单目深度先验、运动概率图和不确定性感知全局 BA，实现了对日常拍摄的动态视频进行精确、快速且鲁棒的相机参数和深度图估计，在合成和真实数据集上显著优于现有方法。

**[MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data](3d_vision/megasynth_scaling_up_3d_scene_reconstruction_with_synthesized_data.md)**

:   MegaSynth 提出通过消除语义信息依赖来实现可扩展的 3D 场景数据合成，生成了包含 70 万个场景的数据集（比真实数据集 DL3DV 大 50 倍），用于训练大规模重建模型（LRM），在多个基准上带来 1.2-1.8dB PSNR 的显著提升。

**[Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](3d_vision/mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)**

:   本文提出Mesh Mamba，首个基于状态空间模型（SSM）的统一网格显著性预测模型，通过纹理对齐、子图嵌入和双向SSM，实现对有纹理和无纹理3D网格的高质量视觉注意力预测，并构建了首个系统对比有/无纹理条件下显著性差异的数据集。

**[MeshArt: Generating Articulated Meshes with Structure-Guided Transformers](3d_vision/meshart_generating_articulated_meshes_with_structure-guided_transformers.md)**

:   MeshArt提出了一种层次化Transformer方法，通过将铰接物体分解为高层关节结构和低层部件网格两阶段生成，自回归地生成紧凑、清晰的三角网格铰接物体，在结构覆盖率上提升57.1%，网格FID提升209分。

**[MEt3R: Measuring Multi-View Consistency in Generated Images](3d_vision/met3r_measuring_multi-view_consistency_in_generated_images.md)**

:   本文提出MEt3R，一种基于DUSt3R重建和DINO特征比较的多视角一致性评价指标，无需相机位姿即可衡量生成图像的3D一致性，并附带开源了一个多视角潜在扩散模型MV-LDM。

**[MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans](3d_vision/metascenes_towards_automated_replica_creation_for_real-world_3d_scans.md)**

:   MetaScenes 构建了一个大规模可仿真3D场景数据集（15366个物体, 831类），通过从真实扫描中自动替换物体资产实现 Real-to-Sim 转换，并提出多模态对齐模型 Scan2Sim 实现自动化资产选择，在场景合成和VLN跨域迁移任务上验证了数据集的价值。

**[MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](3d_vision/micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)**

:   MICAS 针对 3D 点云 in-context learning 中的任务间（inter-task）和任务内（intra-task）采样敏感性问题，提出了多粒度自适应采样机制——包含任务自适应点采样（Gumbel-softmax 可微采样）和查询特定 prompt 采样（基于概率排序选择最优 prompt），在 ShapeNet 基准上将 part segmentation 提升了 4.1%。

**[MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](3d_vision/midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)**

:   MIDI 将预训练的 image-to-3D 单物体生成模型扩展为多实例扩散模型，通过新颖的多实例注意力机制在 3D 生成过程中直接捕获物体间的空间交互关系，从单张图片同时生成多个具有正确空间布局的 3D 实例，在合成和真实数据集上均大幅超越现有方法。

**[Mitigating Ambiguities in 3D Classification with Gaussian Splatting](3d_vision/mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)**

:   本文首次探索用 3D Gaussian Splatting (GS) 点云替代传统点云作为 3D 分类的输入表示，利用 GS 中的 scale/rotation 系数区分线状和平坦表面、利用 opacity 区分透明/反射物体，构建了首个真实世界 GS 点云数据集，在多种分类方法上均验证了 GS 点云消除歧义的有效性。

**[MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots](3d_vision/mne-slam_multi-agent_neural_slam_for_mobile_robots.md)**

:   提出首个分布式多智能体协作神经SLAM框架MNE-SLAM，通过联合场景表示、内到外回环检测和多子地图融合实现高质量协作建图与追踪，仅需点对点通信，并发布首个覆盖单/多智能体场景的真实室内神经SLAM (INS) 数据集。

**[Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](3d_vision/mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)**

:   提出 Mobile-GS，通过深度感知的无序渲染（消除排序瓶颈）+ 神经视角依赖增强 + 一阶SH蒸馏 + 神经向量量化 + 贡献度剪枝，首次在 Snapdragon 8 Gen 3 手机 GPU 上实现 116 FPS 实时高斯溅射渲染，存储仅 4.6MB 且视觉质量与原始 3DGS 相当。

**[MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation](3d_vision/monocular_and_generalizable_gaussian_talking_head_animation.md)**

:   提出MGGTalk框架，仅用单目数据集训练即可泛化到未见身份，核心思路是利用深度估计和面部对称先验来弥补单目数据中几何与外观信息的不完整性，实现基于3DGS的高质量说话头动画。

**[MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](3d_vision/monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)**

:   提出MonoPlace3D，一个场景感知的3D数据增强系统，核心是学习一个从场景图像到合理3D边界框分布的放置网络（SA-PlaceNet），配合基于ControlNet的真实感渲染管线，显著提升单目3D检测器性能和数据效率。

**[Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](3d_vision/morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)**

:   提出Morpheus，一种自回归3DGS风格化方法，核心贡献包括：(1) 新的RGBD扩散模型实现外观和形状风格化的独立强度控制；(2) Warp ControlNet通过变形合成帧传播风格；(3) 深度引导的特征共享确保多视角一致性。

**[Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](3d_vision/mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)**

:   提出自动化数据生成管线构建大规模3D mask-text数据集Mosaic3D-5.6M（5.6M对、30K场景），训练语言对齐3D编码器+mask decoder，实现首个单阶段开放词汇3D实例分割。

**[MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](3d_vision/mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)**

:   提出4D Motion Scaffold (MoSca)表示，通过稀疏6-DoF轨迹图紧凑编码场景运动，结合2D基础模型先验和物理正则化，从无位姿的随手拍单目视频实现全自动4D场景重建。

**[MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning](3d_vision/most_efficient_monarch_sparse_tuning_for_3d_representation_learning.md)**

:   提出首个基于重参数化的3D PEFT方法MoST，设计Point Monarch结构化矩阵（在Monarch基础上加入KNN局部特征平滑），仅调3.6%参数在多个3D benchmark上超越全量微调。

**[MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出 MotionAnyMesh，一种零样本框架，通过 SP4D 运动学先验引导 VLM 推理消除幻觉 + 物理约束轨迹优化保证无碰撞，将静态3D网格自动转化为仿真可用的铰接数字孪生，物理可执行率达 87%，是现有最好方法的近两倍。

**[MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond](3d_vision/motionpro_exploring_the_role_of_pressure_in_human_mocap_and_beyond.md)**

:   构建大规模压力-RGB-光学动捕数据集 MotionPRO（70人/400类动作/12.4M帧），并提出 FRAPPE 基线将压力信号与单目 RGB 融合，显著提升全身姿态估计的物理合理性和全局轨迹精度，进一步将压力先验扩展至人形机器人驱动。

**[MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](3d_vision/movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)**

:   针对多物体室内场景的新视角合成（NVS），通过注入结构感知特征（深度+物体掩码）、引入辅助掩码预测任务、设计结构引导的时间步采样调度器三项设计，显著提升跨视角的物体放置和几何一致性。

**[MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](3d_vision/mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)**

:   将单目深度和法线先验紧密集成到经典增量 SfM 中，通过不确定性传播和交替优化突破三视图轨迹的根本限制，首次实现仅凭两视图轨迹的可靠 3D 重建，在极端低重叠和低视差场景下显著超越所有现有方法。

**[Multi-View Pose-Agnostic Change Localization with Zero Labels](3d_vision/multi-view_pose-agnostic_change_localization_with_zero_labels.md)**

:   提出首个无标签、位姿无关的多视角变化检测方法，通过在 3D Gaussian Splatting 中嵌入变化通道（change-aware 3DGS），融合多视角的特征感知和结构感知变化掩码，在复杂多物体场景中实现 1.7× mIoU 和 1.5× F1 的 SOTA 性能提升，并能为未见视角生成变化掩码。

**[Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)**

:   提出 Murre，将 SfM 稀疏点云作为条件注入扩散模型单目深度估计，生成多视角一致的度量深度图后进行 TSDF 融合，在仅用少量合成数据微调后即可在室内、街景、航拍等多种真实场景中超越 SOTA MVS 和神经隐式重建方法。

**[MUSt3R: Multi-view Network for Stereo 3D Reconstruction](3d_vision/must3r_multi-view_network_for_stereo_3d_reconstruction.md)**

:   本文提出MUSt3R，将DUSt3R从成对架构扩展为多视图架构：通过对称化解码器（参数减半）+多层memory机制实现任意数量图像在统一坐标系下的高帧率3D重建，同一网络可同时处理离线SfM和在线Visual Odometry场景，在TUM-RGBD无标定VO中ATE仅5.5cm。

**[MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](3d_vision/mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)**

:   MV-DUSt3R 提出单阶段前馈网络，通过多视图解码器块联合处理任意数量的无位姿输入视图，完全省去 DUSt3R 所需的全局优化，实现比 DUSt3R 快 48~78 倍的场景重建，同时 Chamfer Distance 降低 1.6~3.2 倍；进一步的 MV-DUSt3R+ 引入跨参考视图注意力块，在大场景上进一步提升重建质量。

**[MVBoost: Boost 3D Reconstruction with Multi-View Refinement](3d_vision/mvboost_boost_3d_reconstruction_with_multi-view_refinement.md)**

:   MVBoost 提出了一种通过多视图精炼策略生成伪真值数据来增强 3D 重建的框架，巧妙结合多视图生成模型的高精度和 3D 重建模型的一致性优势，在 GSO 数据集上实现了 SOTA 的单图到 3D 重建效果（PSNR 18.561, CD 0.101）。

**[MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](3d_vision/mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)**

:   MVGenMaster 提出了一种融合度量深度几何先验的多视图扩散模型，配合 160 万场景的 MvD-1M 数据集和无训练的 key-rescaling 技术，能在单次前向推理中从任意参考视图生成多达 100 个新视角，在域内外 NVS 基准上全面超越 CAT3D 和 ViewCrafter。

**[MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](3d_vision/mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)**

:   MVPaint 提出了一个三阶段的 3D 纹理生成框架——同步多视图生成 (SMG) + 空间感知 3D 补绘 (S3I) + UV 精炼 (UVR)，通过在图像域而非 latent 域进行多视图同步、在 3D 点云空间而非 UV 空间进行补绘、以及空间感知的接缝平滑算法，在 Objaverse 和 GSO 两个 T2T 基准上全面超越 SOTA。

**[MVSAnywhere: Zero-Shot Multi-View Stereo](3d_vision/mvsanywhere_zero-shot_multi-view_stereo.md)**

:   本文提出MVSAnywhere (MVSA)，一个通用多视角立体匹配架构，通过Cost Volume Patchifier将代价体信息高效tokenize后与单目ViT特征融合（Mono/Multi Cue Combiner），结合视角数/尺度无关的元数据编码和级联自适应深度范围估计，在Robust MVS Benchmark上取得零样本SOTA，同时支持任意数量的源视角和任意深度范围。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](3d_vision/node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   提出 Node-RF，将 Neural ODE 与动态 NeRF 紧密耦合，用潜在向量的 ODE 演化建模场景连续时间动力学，实现超出训练序列的长程时序外推和跨轨迹泛化，无需光流或深度监督。

**[P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](3d_vision/p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)**

:   提出 P-SLCR，一种原型库驱动的无监督点云语义分割方法，通过将点分离为"一致"和"模糊"两类，用一致结构学习对齐一致点与原型 + 语义关系一致性推理约束两个原型库，在 S3DIS 上无监督达 47.1% mIoU，超越全监督 PointNet。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，首个在3D摄影测量空间进行全景拼接的 Transformer 框架，利用预训练 VGGT 骨干获取3D感知的多视角特征对齐 + 多特征联合优化接缝检测，支持2到数百张输入图像，在弱纹理/大视差/重复模式场景下成功率达97.8%。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，将隐式神经表示 (INR/SIREN) 与预训练扩散模型先验结合，通过 proximal loss 在每个 DDIM 时间步用扩散去噪输出正则化 INR 重建，在稀疏视角中子 CT（低至 4-5 个视角）上超越 FBP、纯 INR、DD3IP 和经典 MBIR(qGGMRF) 方法。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](3d_vision/rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)**

:   Rewis3d 利用前馈 3D 重建（MapAnything）从 2D 视频中获取 3D 点云作为辅助监督信号，通过双 Student-Teacher 架构和加权跨模态一致性 (CMC) 损失，在仅使用稀疏标注（点/涂鸦/粗标记）的情况下将弱监督 2D 语义分割性能提升 2-7% mIoU，推理时仍为纯 2D。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](3d_vision/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一个即插即用的背景引导原型富化框架，在基类训练后用类无关分割模型从背景区域挖掘伪实例建立 Instance Prototype Bank (IPB)，当新类别以少样本方式出现时，通过 Contextual Prototype Retrieval (CPR) 和 Attention-Based Prototype Enrichment (APE) 融合背景原型与少样本原型，在 ScanNet/S3DIS 上新类 IoU 提升最高 6.98%。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   针对 3DGS 的资源瞄准攻击（通过投毒训练图像触发高斯过度增长导致资源耗尽），提出频域防御：3D 频率滤波器通过将高斯协方差与频谱响应关联实现频率感知剪枝，2D 频谱正则化通过熵惩罚渲染图像的角向能量各向异性来抑制攻击噪声，实现高斯数量压缩 5.92×、内存减少 3.66×、速度提升 4.34×。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](3d_vision/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

**[VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。

---

## 🎨 图像生成 { #image_generation }

**[3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](image_generation/3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)**

:   提出基于新型原语表示PrimX和Diffusion Transformer的原生3D生成模型3DTopia-XL，能从文本或图像输入生成带有高分辨率几何、纹理和PBR材质的高质量3D资产，在质量和效率上显著超越现有方法。

**[A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](image_generation/a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)**

:   系统研究了使用decoder-only LLM作为文本到图像扩散模型文本编码器的效果，发现直接使用最后一层embedding效果差于T5，但通过层归一化平均（layer-normalized averaging）聚合所有层的embedding可显著超越T5基线。

**[AMO Sampler: Enhancing Text Rendering with Overshooting](image_generation/amo_sampler_enhancing_text_rendering_with_overshooting.md)**

:   提出AMO（Attention-Modulated Overshooting）采样器，一种无需训练的推理时增强方法，通过在rectified flow模型的采样过程中引入过冲-噪声补偿的Langevin动力学校正，并利用文本-图像交叉注意力分数自适应控制过冲强度，显著提升文本渲染的准确率，同时保持生成图像的整体质量。

**[HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](image_generation/an_image-like_diffusion_method_for_human-object_interaction_detection.md)**

**[AniDoc: Animation Creation Made Easier](image_generation/anidoc_animation_creation_made_easier.md)**

**[AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](image_generation/animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)**

**[Any-Resolution AI-Generated Image Detection by Spectral Learning](image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)**

:   提出一种基于频谱学习的 AI 生成图像检测方法，通过在频域提取分辨率不变的特征，使检测器能够在任意分辨率的输入图像上保持鲁棒性，解决了现有方法对特定分辨率依赖的问题。

**[Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](image_generation/arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)**

:   本文提出InvSR，通过训练一个噪声预测网络来实现扩散反演（Diffusion Inversion），利用预训练扩散模型的图像先验进行超分辨率，支持1-5步任意步数采样，即使单步采样也能达到或超过现有SOTA方法的效果。

**[ArtiFade: Learning to Generate High-quality Subject from Blemished Images](image_generation/artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)**

:   本文提出ArtiFade，首个解决"瑕疵主题驱动生成"问题的方法，通过构建瑕疵-无瑕疵配对数据集、部分微调扩散模型的cross-attention权重并优化artifact-free embedding，使得现有主题驱动方法（Textual Inversion、DreamBooth）能从带水印/贴纸/对抗噪声等瑕疵的图像中生成高质量无伪影的主题图像。

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，用双向布朗桥扩散模型建模地面 LSST 和太空 Euclid 两大天文巡天之间的随机映射关系，实现概率性跨巡天翻译与稀有事件检测（强引力透镜），并证明 epsilon-prediction 训练目标兼具重建质量和似然性优势。

**[AutoPresent: Designing Structured Visuals from Scratch](image_generation/autopresent_designing_structured_visuals_from_scratch.md)**

:   本文提出AutoPresent框架和SlidesBench基准，首次系统研究从自然语言指令生成演示幻灯片的任务——通过让LLM生成Python代码（而非端到端图像生成）来创建PPTX幻灯片，配合SlidesLib工具库和迭代优化，8B参数的开源模型达到接近GPT-4o的效果。

**[AvatarArtist: Open-Domain 4D Avatarization](image_generation/avatarartist_open-domain_4d_avatarization.md)**

:   提出 AvatarArtist，通过 GAN 和扩散模型协同构建多域 image-triplane 数据集，训练 DiT 生成参数化三平面 + 运动感知跨域渲染器，实现从任意风格单张肖像生成可驱动的 4D 头像。

**[Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](image_generation/beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)**

:   系统性地将学习式图像处理中卷积的替代/扩展算子组织为五大家族（分解型、自适应加权型、基自适应型、积分/核型和注意力型），并从线性、局部性、等变性、计算成本和任务适用性等多个维度进行比较分析。

**[Bias for Action: Video Implicit Neural Representations with Bias Modulation](image_generation/bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)**

:   提出 ActINR，通过在 INR 中跨帧共享权重、仅用偏置（bias）建模运动的方式实现连续视频表示，在 10× 慢动作、4× 空间超分+2× 时间超分、去噪和修复任务上大幅超越现有方法（平均 3-6dB 提升）。

**[BiGain: Unified Token Compression for Joint Generation and Classification](image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 首次将扩散模型的 token 压缩重新定义为生成+分类的双目标优化问题，提出拉普拉斯门控 token 合并（L-GTM）和插值-外推 KV 下采样（IE-KVD）两个频率感知算子，在保持生成质量同时显著提升分类准确率（ImageNet-1K 70%合并比下 Acc +7.15%，FID -0.34）。

**[BootPlace: Bootstrapped Object Placement with Detection Transformers](image_generation/bootplace_bootstrapped_object_placement_with_detection_transformers.md)**

:   提出 BootPlace，将物体放置问题重新定义为"放置即检测"问题，通过在物体减除背景上训练检测变换器识别候选区域，再用负相关语义互补将目标物体匹配到最佳区域，在 Cityscapes 上 top-5 IOU 比 SOTA 提升约 4×。

**[BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](image_generation/boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)**

:   提出 BooW-VTON，通过高质量伪数据构建 + 野外数据增广 + 试穿定位损失，训练出无需人体解析掩码的虚拟试穿扩散模型，在 VITON-HD/StreetVTON/WildVTON 多个基准上全面超越现有方法。

**[CacheQuant: Comprehensively Accelerated Diffusion Models](image_generation/cachequant_comprehensively_accelerated_diffusion_models.md)**

:   提出 CacheQuant，一种无需训练的范式，通过联合优化模型缓存（temporal level）和量化（structural level）来全面加速扩散模型，在 Stable Diffusion 上实现 5.18× 加速和 4× 压缩，CLIP score 仅损失 0.02。

**[CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model](image_generation/camfreediff_camera-free_image_to_panorama_generation_with_diffusion_model.md)**

:   提出 CamFreeDiff，通过在多视图扩散框架中集成轻量级 3-DoF 单应性估计器，实现从无相机参数的单张图像生成 360° 全景图，FID 从 MVDiffusion 的 42.4 降至 27.0，且无需微调即可泛化到域外数据。

**[Can Generative Video Models Help Pose Estimation?](image_generation/can_generative_video_models_help_pose_estimation.md)**

:   提出 InterPose，利用预训练视频生成模型在两张少/无重叠图像之间"幻想"中间帧，配合自一致性评分选择最佳视频，在 DUSt3R 基础上一致提升四个数据集的位姿估计精度。

**[Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](image_generation/channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)**

:   提出通道级噪声调度方法，让单一扩散模型架构通过不同噪声调度实现精度优先（SDM, T=4）和多样性优先（PDM, T=1000）两种逆渲染模式，同时引入 ILR 隐式光照表征支持逐像素环境图推理和真实物体插入。

**[ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](image_generation/chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)**

:   提出 ChatGen，将文本到图像生成中的 prompt 编写、模型选择和参数配置三个繁琐步骤自动化，通过多阶段进化训练策略（ChatGen-Evo）让用户以自由聊天方式描述需求即可获得高质量生成图像。

**[Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](image_generation/classifier-free_guidance_inside_the_attraction_basin_may_cause_memorization.md)**

:   从动力系统视角提出"吸引盆地"概念解释扩散模型记忆化现象——CFG 在吸引盆地内施加会导致轨迹收敛到记忆化训练图像，通过检测转折点延迟 CFG 启动（配合反向引导 OG）可零额外开销地缓解记忆化。

**[CleanDIFT: Diffusion Features without Noise](image_generation/cleandift_diffusion_features_without_noise.md)**

:   提出 CleanDIFT，通过轻量级无监督微调（单卡 A100 仅 30 分钟），使扩散模型直接在干净图像上提取高质量语义特征，消除了传统方法需要加噪和调时间步的限制，在语义对应、深度估计、分割等多任务上显著超越标准扩散特征。

**[CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](image_generation/clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)**

:   系统揭示 CLIP 在多目标场景中的两类偏差——文本编码器偏向先提到的物体、图像编码器偏向大物体，并追溯偏差根源至对比训练过程中训练数据里大物体被先提到的统计规律。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)**

:   提出 coDrawAgents，由 Interpreter、Planner、Checker、Painter 四个专家 agent 组成的交互式多智能体对话框架，通过分而治之的增量布局规划、视觉上下文感知推理和显式错误纠正，在 GenEval 上达到 0.94（SOTA）、DPG-Bench 上 85.17（SOTA）。

**[Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](image_generation/collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)**

:   提出 CoDe（协同解码），将 VAR 的多尺度推理分解为大模型草稿（低频小尺度）+ 小模型精修（高频大尺度）的协作流程，实现 1.7× 加速、50% 显存降低，FID 仅从 1.95 微增至 1.98。

**[Color Alignment in Diffusion](image_generation/color_alignment_in_diffusion.md)**

:   提出颜色对齐扩散方法，通过将中间采样或预测结果投影到条件颜色空间（最近邻颜色映射），使扩散模型在保持结构生成自由度的同时严格遵循给定的颜色分布（颜色值+比例），支持重训练、微调和零样本三种设置。

**[Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](image_generation/community_forensics_using_thousands_of_generators_to_train_fake_image_detectors.md)**

:   构建包含 4803 个生成模型、270 万张图像的 Community Forensics 数据集，发现即使架构相似的模型也能通过增加数量显著提升假图检测泛化性，在多个基准上达到最优平均 mAP 0.966。

**[Composing Parts for Expressive Object Generation](image_generation/composing_parts_for_expressive_object_generation.md)**

:   提出 PartComposer，一种无需训练的方法，通过并行"部件扩散"从注意力图中定位对象部件，再用区域扩散为每个部件独立生成用户指定的细粒度属性（颜色、风格、描述），实现部件级可控图像合成。

**[Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](image_generation/comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)**

:   提出基于预训练扩散模型的人体重光照和背景协调统一框架，通过粗到精策略（球谐函数 ControlNet 提供粗光照 + 扩散模型学习精细残差）和无监督运动 ControlNet 实现静态和视频场景的光照一致重光照。

**[Concept Lancet: Image Editing with Compositional Representation Transplant](image_generation/concept_lancet_image_editing_with_compositional_representation_transplant.md)**

:   提出 Concept Lancet (CoLan)，一种零样本即插即用的图像编辑框架，通过将源图像的隐表示稀疏分解为视觉概念向量的线性组合，然后根据编辑任务（替换/添加/删除）进行定制化概念移植，解决了编辑强度校准难题。

**[Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](image_generation/concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)**

:   提出 Concept Replacer，通过少样本训练的概念定位器精确识别去噪过程中的敏感概念区域，再用训练免费的双提示交叉注意力（DPCA）将定位区域替换为安全内容，实现精确局部概念替换而非全局图像失真。

**[ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](image_generation/conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)**

:   提出 ConceptGuard，通过移位嵌入、概念绑定提示、记忆保持正则化和优先队列回放四种策略，实现持续个性化 T2I 生成中灾难性遗忘和概念混淆的缓解，在多概念基准上大幅超越现有方法。

**[Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](image_generation/conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)**

:   分析 SDXL 自注意力层对风格和结构的敏感度差异，发现仅在最敏感的子集层中注入条件信息即可显著改善多条件生成中的风格-内容 trade-off，无需额外训练。

**[Consistent and Controllable Image Animation with Motion Diffusion Models](image_generation/consistent_and_controllable_image_animation_with_motion_diffusion_models.md)**

:   提出 Cinemo，基于扩散模型的图像动画方法，通过学习运动残差（而非直接预测帧）分布大幅提升与输入图像的时间一致性，配合 SSIM 运动强度控制和 DCT 噪声初始化实现精细可控的 I2V 生成，在 UCF-101 和 MSR-VTT 上全面超越现有方法。

**[Controllable Human Image Generation with Personalized Multi-Garments](image_generation/controllable_human_image_generation_with_personalized_multi-garments.md)**

:   提出 BootComp，通过两阶段框架（分解模块从真人图像自举生成合成配对数据 + 组合模块用双 SDXL 网络通过扩展自注意力注入多服装特征），实现给定多件参考服装的可控人体图像生成，MP-LPIPS 比基线提升 30%。

**[CustAny: Customizing Anything from A Single Example](image_generation/custany_customizing_anything_from_a_single_example.md)**

:   提出 CustAny，一种零样本通用物体定制化框架，通过 DINOv2+MAE 双编码器提取全面身份特征、全局+局部双层级注入、以及对比学习身份解耦，从单张参考图生成保持身份的多样化图像，在通用/人脸/试穿三个领域全面超越专用方法。

**[Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales](image_generation/data-free_group-wise_fully_quantized_winograd_convolution_via_learnable_scales.md)**

:   提出完全量化 Winograd 卷积的方法——通过可学习的对角缩放矩阵均衡 Winograd 域输出的动态范围差异，仅用随机高斯噪声（无真实数据）微调缩放参数，在 InstaFlow/SD v1.5 等扩散模型上实现近无损的 W8A8 Winograd 加速（CPU 卷积层加速 31.3%）。

**[DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](image_generation/decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)**

:   提出 DeClotH，用 DMTet 表示+3D 模板正则化（SMPLicit/SMPL）分别重建服装和人体，并训练 ClothDiffusion（ControlNet 微调）生成纯服装图像作为 SDS 损失指导，在单张图像下实现可分解的 3D 服装+人体重建，在 4D-DRESS 上超越所有基线。

**[Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning](image_generation/decouple-then-merge_finetune_diffusion_models_as_multi-task_learning.md)**

:   提出 DeMe，将扩散模型训练视为多任务学习问题——不同时间步是不同"任务"且存在梯度冲突，通过将时间步解耦为 N 个范围分别微调再合并回单一模型，在零推理开销下显著提升生成质量（CIFAR10 FID 从 4.42 降至 3.51）。

**[Decoupling Training-Free Guided Diffusion by ADMM](image_generation/decoupling_training-free_guided_diffusion_by_admm.md)**

:   提出 ADMMDiff，用 ADMM 框架将训练免费条件生成中的扩散项和引导项解耦为两个变量（x 扩散, z 引导, 约束 x=z），用近端算子分别处理，消除了手动调权重的需求并提供收敛保证，在分割/素描/文本引导生成上大幅超越 DPS 和 MPGD。

**[Denoising Functional Maps: Diffusion Models for Shape Correspondence](image_generation/denoising_functional_maps_diffusion_models_for_shape_correspondence.md)**

:   提出 DenoisFM，用 DDPM 直接在光谱域去噪生成功能映射矩阵来预测 3D 形状间的点对应，配合无监督符号纠正网络消除拉普拉斯特征函数的符号歧义，在多个形状匹配基准上大幅超越大规模学习方法同时匹配描述符方法。

**[FreeMCG: Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI](image_generation/derivative-free_diffusion_manifold-constrained_gradient_for_unified_xai.md)**

:   提出 FreeMCG，利用扩散模型的 Tweedie 去噪生成流形上的集成粒子，通过集成卡尔曼滤波（EnKF）理论在无需模型权重的情况下近似分类器的流形约束梯度，统一实现黑盒特征归因和反事实解释，人类评估显著优于白盒方法。

**[DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](image_generation/dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)**

:   提出 DexGrasp Anything，将三种物理约束（表面拉力、外穿透排斥、自穿透排斥）集成到扩散模型的训练和采样中，配合 LLM 增强的点云+语义物体表示，在 5 个基准上达到 SOTA 灵巧抓取性能（Suc.6 57.5%，比 SceneDiffuser 提升 2.2×）。

**[DiC: Rethinking Conv3x3 Designs in Diffusion Models](image_generation/dic_rethinking_conv3x3_designs_in_diffusion_models.md)**

:   构建纯 3×3 卷积的扩散模型 DiC，通过沙漏编码解码器架构扩大感受野、稀疏跳跃连接减少冗余、阶段特定条件注入，在 ImageNet 256×256 上以 DiT-XL 的相当 FLOPs 实现 4.7× 吞吐量和更优 FID（3.89 vs 6.24, with CFG）。

**[DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](image_generation/difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)**

:   提出 DiffLocks，通过自动化 Blender 管线构建 4 万种发型的最大合成头发数据集，用发丝 VAE 将 10 万根发丝编码为 256×256 头皮纹理图，再训练 image-conditioned HDiT 从单张图像扩散去噪生成头皮纹理，实现 2.33 秒 10 万发丝重建，速度比基线快 8-23×。

**[DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](image_generation/diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)**

:   提出 DiffSensei，两阶段漫画面板生成框架——Stage 1 用掩码交叉注意力注入实现多角色布局控制和对话气泡放置，Stage 2 用 MLLM 适配器根据文本描述动态调整角色特征（表情/姿态/动作），解决了先前方法的"复制粘贴"效果问题。

**[Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](image_generation/diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)**

:   提出 Diffusion-4K，通过分区 VAE（F=16，无需重训练即避免 OOM）和小波域微调（WLF，用 Haar DWT 分解速度预测增强高频纹理），实现直接 4096×4096 分辨率图像合成，并构建 Aesthetic-4K 基准评估超高分辨率质量。

**[Diffusion Self-Distillation for Zero-Shot Customized Image Generation](image_generation/diffusion_self-distillation_for_zero-shot_customized_image_generation.md)**

:   利用预训练 T2I 模型（FLUX）的上下文网格生成能力自动合成 40 万配对训练数据，再微调该模型为零样本 image+text 条件生成器，在 DreamBench++ 上以零样本方式超越需微调的 DreamBooth LoRA。

**[Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](image_generation/dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)**

:   提出 DiffLens，用 k-稀疏自编码器将 U-Net 瓶颈层的多义性神经元分解为单义性特征，通过归因分析识别偏见特征并进行干预，实现扩散模型性别偏见的精确缓解（FD 从 0.564 降至 0.046）同时保持甚至提升生成质量。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)**

:   DiT-IC 将预训练 T2I 扩散 Transformer 适配为单步图像压缩重建模型，在 32x 下采样的深层潜空间工作，通过方差引导重建流、自蒸馏对齐和潜变量条件引导三种对齐机制，实现 SOTA 感知质量且解码比现有扩散 codec 快 30 倍。

**[DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](image_generation/diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)**

:   提出 DiverseFlow，一种训练免费的推理时方法，通过行列式点过程（DPP）在 ODE 求解过程中耦合多个样本使其互相排斥，在固定采样预算下显著提升流匹配模型的模式覆盖多样性（如从 5.64 个模式提升到 7.11 个）。

**[DiVoT: Diffusion Powers Video Tokenizer for Comprehension and Generation](image_generation/divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)**

:   提出 DiVoT，用扩散去噪过程作为视频分词器的自监督训练信号——若分词器的连续表示能引导 U-Net 成功去噪视频则表示有效，统一视频理解和生成，LLM 通过高斯混合模型（而非 MSE 回归）预测连续视频特征。

**[Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](image_generation/do_visual_imaginations_improve_vision-and-language_navigation_agents.md)**

:   本文用 SDXL 为 VLN 指令中的视觉地标生成合成图像作为"想象"，通过 ViT 编码后拼接到文本指令 embedding 中输入 VLN agent，配合余弦相似度对齐损失，在 R2R 和 REVERIE 上一致提升导航成功率约 1%，初步验证了视觉想象作为语言与视觉之间桥梁的价值。

**[DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](image_generation/doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)**

:   提出 DoraCycle 使用两个多模态循环（文→图→文 和 图→文→图）对统一多模态生成模型做无配对域适应，仅用无配对目标域数据即可接近全配对训练效果（FID 27.44 vs 24.93），10% 配对+90% 无配对时几乎无损（FID 25.37）。

**[DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](image_generation/dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)**

:   提出 DreamCache 通过在单个去噪步（t=1）缓存参考图的 U-Net 中间特征，用轻量 25M 参数的条件适配器在生成时注入缓存特征，实现免微调、免编码器、即插即用的个性化图像生成。

**[DreamOmni: Unified Image Generation and Editing](image_generation/dreamomni_unified_image_generation_and_editing.md)**

:   构建统一文生图+多种编辑任务（指令编辑/修补/拖拽/参考生成）的 2.5B DIT 模型，用 Qwen2-VL 替换文本编码器实现统一视觉-语言 prompt 理解，通过合成贴纸数据管线高效创建编辑训练数据，在生成和编辑上同时达到 SOTA。

**[DreamRelation: Bridging Customization and Relation Generation](image_generation/dreamrelation_bridging_customization_and_relation_generation.md)**

:   DreamRelation 提出了一种关系感知的定制化图像生成框架，通过精心构建的解耦数据引擎、关键点匹配损失（KML）和局部 token 注入三大设计，在保持多目标身份一致性的同时准确生成文本指定的目标间关系（如拥抱、骑行等），在 RelationBench 上全面超越现有方法。

**[DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)**

:   提出 DualAnoDiff，通过双相互关联扩散模型（全局分支生成整体异常图像+异常分支生成局部异常部分）同时生成高质量的异常图像-掩码对，并引入背景补偿模块维持背景和物体形状的一致性，显著提升下游异常检测/定位/分类的性能。

**[Dual Diffusion for Unified Image Generation and Understanding](image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)**

:   提出 Dual Diffusion Transformer (D-DiT)，在单一 MM-DiT 架构中同时使用连续扩散建模图像分布和离散掩码扩散建模文本分布，是首个端到端的全扩散多模态模型，支持图像生成、图像描述和视觉问答等全套任务。

**[Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](image_generation/dual_prompting_image_restoration_with_diffusion_transformers.md)**

:   提出 DPIR，基于 SD3 (Diffusion Transformer) 的图像修复模型，通过轻量级低质量图像条件分支和全局-局部视觉双提示(dual prompting)分支，从多角度引入退化图像信息，首次系统性地将 DiT 应用于图像修复并取得 SOTA 性能。

**[Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](image_generation/dynamic_motion_blending_for_versatile_motion_editing.md)**

:   MotionReFit 提出了首个通用文本引导运动编辑框架，通过 MotionCutMix 数据增强技术动态生成训练三元组，配合自回归扩散模型和运动协调器，实现涵盖身体部位替换、风格迁移和细粒度调整的空间与时序编辑。

**[EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](image_generation/easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)**

:   EasyCraft 提出了一个端到端的自动角色捏脸框架，通过 MAE 预训练的通用 ViT 编码器将任意风格的面部图像映射为统一特征分布，再转换为游戏引擎捏脸参数，同时集成文本到图像技术支持文本输入，可轻松适配不同游戏引擎。

**[EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](image_generation/eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)**

:   提出 EDEN，从输入表示、模型架构和训练范式三个维度全面增强扩散模型在视频帧插值中的作用，通过 Transformer tokenizer 压缩中间帧为语义丰富的 1D token 表示、采用 DiT 替代 U-Net 架构、引入双流上下文整合机制（时序注意力 + 帧差嵌入），在 DAVIS 等大运动基准上 LPIPS 降低近 10%，且仅需 2 步去噪即可实现高质量生成。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。

**[Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)**

:   提出一种双层优化框架，将剪枝扩散模型的微调恢复（下层：蒸馏+扩散损失最小化）和不良概念遗忘（上层：引导模型远离目标概念）统一为单一阶段优化，解决了"先微调再遗忘"两阶段方法中微调最优点不等于遗忘最优初始化的循环依赖问题，在风格去除上 CSD 指标降低 27%。

**[Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](image_generation/efficient_long_video_tokenization_via_coordinate-based_patch_reconstruction.md)**

:   提出 CoordTok，一种可扩展的视频 tokenizer，将视频编码为因子化 triplane 表示，解码器学习从随机采样的 $(x,y,t)$ 坐标到对应 patch 像素的映射（而非一次重建所有帧），使得可以直接在 128 帧长视频上训练大型 tokenizer，将 128 帧视频编码为仅 1280 个 token（基线需要 6144-8192 个），并驱动 DiT 实现 128 帧一次性视频生成（FVD 369.3 SOTA）。

**[Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](image_generation/efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)**

:   本文提出 ZOODiP，通过零阶优化在量化后的扩散模型上进行个性化（Textual Inversion），利用子空间梯度投影去噪和部分时间步采样加速训练，仅用 2.37GB 显存和前向传播即可达到与梯度方法可比的个性化效果，内存节省最高 8.2 倍。

**[EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](image_generation/emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)**

:   本文提出 EmoDubber，一个情感可控的电影配音架构，通过时长级对比学习对齐唇动与韵律、发音增强策略提升清晰度、基于流匹配的正负引导机制控制情感类型和强度，在唇形同步和发音清晰度上全面超越现有方法。

**[EmoEdit: Evoking Emotions through Image Manipulation](image_generation/emoedit_evoking_emotions_through_image_manipulation.md)**

:   本文提出 EmoEdit，首个通过内容修改（而非仅颜色/风格调整）来唤起指定情感的图像操纵框架，构建了 40,120 对的 EmoEditSet 数据集，设计了可即插即用的 Emotion Adapter，在结构保持和情感唤起之间取得了显著平衡。

**[Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](image_generation/enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)**

:   提出 PN-Diffusion，利用正向播放和反向播放的舞蹈视频分别提取正负节奏条件，设计双向扩散与反向过程来联合训练 U-Net，增强生成音乐与舞蹈动作的节奏一致性和音乐质量，在 AIST++ 和 TikTok 数据集上 BCS 提升 1.80/3.85、BHS 提升 4.22/5.90。

**[Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](image_generation/enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)**

:   提出 DIAE，通过多模态美学感知模块（MAP）将模糊美学指令转化为 HSV/轮廓图+文本的多模态控制信号，并构建"非完美配对"数据集 IIAEData 配合双分支监督策略实现弱监督美学增强，在 LAION 和 MLLM 美学评分上达 SOTA。

**[Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](image_generation/enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)**

:   本文提出SPARCL，通过将真实图像特征注入快速T2I模型的padding嵌入来生成高保真微变化合成图像，并设计自适应margin损失过滤噪声合成样本聚焦难样本学习，将CLIP的组合理解准确率在四个基准上平均提升8%以上，在三个基准上超越SOTA 2%。

**[Erasing Undesirable Influence in Diffusion Models (EraseDiff)](image_generation/erasing_undesirable_influence_in_diffusion_models.md)**

:   本文提出EraseDiff，将扩散模型的数据遗忘问题形式化为基于价值函数的约束优化问题，通过自然的一阶算法同时优化保留性能和擦除效果，在DDPM/Stable Diffusion上比SA快11倍、比SalUn快2倍，同时在保留-遗忘权衡上取得Pareto最优。

**[Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](image_generation/everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)**

:   本文揭示了扩散驱动TTA方法中源域与合成域之间存在隐性不对齐问题，提出Synthetic-Domain Alignment (SDA)框架，通过Mix of Diffusion (MoD)技术将源模型和目标数据同时对齐到同一个合成域，在分类、分割和多模态大语言模型上均取得了一致的性能提升。

**[EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](image_generation/evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)**

:   EvoTok 提出了一种基于残差潜在演化（Residual Latent Evolution）的统一图像 tokenizer，通过在共享潜空间中级联残差向量量化，使表示从浅层的像素级细节渐进演化到深层的语义级抽象，在仅用 13M 图像训练的情况下实现了 0.43 rFID 的重建质量，并在 7/9 个理解 benchmark 和 GenEval/GenAI-Bench 上取得优异效果。

**[FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)**

:   提出 FaithDiff，首次释放（fine-tune）预训练扩散模型先验用于图像超分辨率，并设计对齐模块桥接退化图像特征与扩散噪声隐空间，通过联合优化 encoder 和扩散模型实现高保真结构恢复。

**[FilmComposer: LLM-Driven Music Production for Silent Film Clips](image_generation/filmcomposer_llm-driven_music_production_for_silent_film_clips.md)**

:   FilmComposer 首次将大语言模型多代理系统与波形/符号音乐生成相结合，模拟专业音乐人的工作流程（选点→作曲→编曲→混音），从无声电影片段自动生成高质量（48kHz）、高音乐性、具有发展性的电影配乐。

**[FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](image_generation/finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)**

:   FineLIP 通过位置编码拉伸（77→248 tokens）、自适应 Token 精炼模块（ATRM）和跨模态 Token 级对齐（CLIM），使 CLIP 模型能够处理长文本描述并实现细粒度视觉-文本匹配，在长描述检索任务上显著超越 Long-CLIP、TULIP 等现有方法。

**[FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](image_generation/flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)**

:   FlipSketch 首次实现从单张静态草图 + 文本描述自动生成无约束栅格草图动画，通过在 T2V 扩散模型上微调 LoRA、DDIM 反演参考帧机制和双注意力组合三大创新，在保持草图身份的同时生成流畅、动态的动画序列。

**[Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](image_generation/focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)**

:   提出 Focus-N-Fix，一种区域感知的 T2I 模型微调方法，通过定位问题区域并约束非问题区域不变，实现对伪影、过度性化、暴力等局部质量问题的精准修复，同时避免全局微调带来的灾难性遗忘和奖励黑客现象。

**[Font-Agent: Enhancing Font Understanding with Large Language Models](image_generation/font-agent_enhancing_font_understanding_with_large_language_models.md)**

:   构建了包含 135,000 个字体-文本对的大规模多模态数据集 DFD，并提出 Font-Agent——一个基于视觉语言模型的字体理解代理，通过边缘感知追踪模块（EAT）捕捉字体笔画细节和动态直接偏好优化策略（D-DPO）精细化模型对字体风格的理解能力。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](image_generation/fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。

**[Free-viewpoint Human Animation with Pose-correlated Reference Selection](image_generation/free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)**

:   提出一种姿态关联参考选择扩散网络，通过姿态相关性模块计算目标-参考姿态间的关联图并自适应选择最相关的参考特征，支持在大幅视角变化（包括镜头推拉）下进行高质量人体动画生成，同时引入了 MSTed 多机位 TED 视频数据集。

**[FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](image_generation/freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)**

:   FreeUV 提出了一种不需要 ground-truth UV 纹理数据的面部 UV 纹理恢复框架，通过分别训练关注真实外观的 UV-to-2D 网络和关注结构一致性的 2D-to-UV 网络，在推理时将两者的 UV 相关模块跨装配（Cross-Assembly）到预训练 Stable Diffusion 中，实现高保真的 UV-to-UV 纹理生成。

**[From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](image_generation/from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)**

:   LaDeCo 将平面设计的分层设计原则引入大型多模态模型（LMM），先用 GPT-4o 对多模态设计元素进行语义层规划，再按层逐步预测元素属性并渲染中间结果反馈给模型，将复杂的设计合成任务分解为可管理的子步骤，在设计合成质量上大幅超越基线方法。

**[From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](image_generation/from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)**

:   本文定义了文本到图表生成任务，构建了 DiagramGenBenchmark（涵盖 8 类图表），并提出多智能体框架 DiagramAgent（Plan + Code + Check + Diagram-to-Code），在图表生成、编码和编辑任务上显著超越现有文本到图像/代码方法。

**[GCC: Generative Color Constancy via Diffusing a Color Checker](image_generation/gcc_generative_color_constancy_via_diffusing_a_color_checker.md)**

:   GCC 利用预训练扩散模型的图像先验，通过 inpainting 生成反映场景光照的色卡来估计光照颜色，借助 Laplacian 分解保留色卡结构的同时适应光照变化，在跨相机场景中展现出优越的泛化能力。

**[GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](image_generation/gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)**

:   本文提出GenDeg，一个基于Stable Diffusion的退化合成框架，能在任意干净图像上生成多种可控退化（雾/雨/雪/运动模糊/低光/雨滴），合成55万+图像构成GenDS数据集，训练在其上的All-In-One复原模型在域外测试集上获得显著性能提升。

**[Generation of Maximal Snake Polyominoes Using a Deep Neural Network](image_generation/generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)**

:   将 DDPM 应用于生成最大蛇形多联骨牌，提出精简版 Structured Pixel Space Diffusion（SPS Diffusion），在训练到 14x14 正方网格的情况下泛化到 28x28 并生成有效蛇形，部分结果超越已知最大长度下界。

**[Generative Image Layer Decomposition with Visual Effects](image_generation/generative_image_layer_decomposition_with_visual_effects.md)**

:   LayerDecomp 提出了一个基于 Diffusion Transformer 的图像图层分解框架，将输入图像分解为干净的 RGB 背景层和带有透明视觉效果（阴影、反射）的 RGBA 前景层，通过一致性损失在无标注数据上也能学到正确的前景表示，大幅超越现有物体移除和空间编辑方法。

**[Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](image_generation/generative_multimodal_pretraining_with_discrete_diffusion_timestep_tokens.md)**

:   DDT-LLaMA 提出用扩散时间步编码学习具有递归结构的离散视觉 token（DDT），使视觉 token 序列像自然语言一样具有层级依赖关系，从而在统一的 next-token-prediction 框架下同时实现多模态理解和生成的 SOTA 性能。

**[Generative Photomontage](image_generation/generative_photomontage.md)**

:   提出 Generative Photomontage 框架，允许用户从多张 ControlNet 生成的图像中选取不同区域，通过扩散特征空间的图割分割和自注意力特征注入进行无缝合成，实现对生成图像的精细组合控制。

**[GIF: Generative Inspiration for Face Recognition at Scale](image_generation/gif_generative_inspiration_for_face_recognition_at_scale.md)**

:   提出将人脸识别中的标量标签替换为结构化身份编码（整数序列），通过CLIP初始化+超球面均匀化生成编码向量，再用层次聚类构建树结构编码，将分类器计算复杂度从$\mathcal{O}(m)$降至$\mathcal{O}(\log m)$，同时解决了少数类坍缩问题。

**[GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](image_generation/glass_guided_latent_slot_diffusion_for_object-centric_learning.md)**

:   本文提出 GLASS，一种基于 Slot Attention 的物体中心学习方法，通过在扩散模型生成的图像空间中学习，利用语义引导模块（扩散模型的交叉注意力生成伪语义掩码）和实例引导模块（MLP 重建编码器特征）协同解决过分割和欠分割问题，在真实场景的物体发现和条件/组合生成任务上大幅超越前方法。

**[GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)**

:   提出GlyphMastero字形编码器，通过双流（局部字符级+全局文本行级）特征提取、跨层次注意力交互和多尺度FPN融合，为扩散模型提供笔画级精确的字形引导，在多语言场景文字编辑中句子准确率提升18.02%、FID降低53.28%。

**[Goku: Flow Based Video Generative Foundation Models](image_generation/goku_flow_based_video_generative_foundation_models.md)**

:   Goku 是字节跳动与港大提出的 rectified flow Transformer 系列模型（2B/8B），首次将 rectified flow 用于图像-视频联合生成，配合全面的数据管线和大规模训练基础设施优化，在 VBench（84.85）和 GenEval（0.76）等基准上达到 SOTA。

**[GPS as a Control Signal for Image Generation](image_generation/gps_as_a_control_signal_for_image_generation.md)**

:   将照片 EXIF 元数据中的 GPS 坐标作为扩散模型的新型控制信号，训练 GPS+文本联合条件的图像生成模型，能捕捉城市内不同街区/地标的细粒度外观差异，并通过角度条件 SDS 从 2D 模型提取 3D 地标重建。

**[GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](image_generation/graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)**

:   提出 GraphGPT-o，将多模态属性图（MMAG，节点含图像+文本，边表示关系）的结构信息注入多模态大语言模型（MLLM），通过 PPR 采样、层次化 Q-Former 对齐器和灵活推理策略，实现基于图上下文的文本-图像联合生成。

**[Hiding Images in Diffusion Models by Editing Learned Score Functions](image_generation/hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)**

:   提出在扩散模型的特定时间步编辑learned score function来隐藏图像的方法，结合梯度感知参数选择和LoRA实现参数高效微调，在提取精度（52.90 dB PSNR）、模型保真度（FID变化仅0.02）和隐藏效率（0.04 GPU小时）三个维度上全面超越现有方法数个量级。

**[Hierarchical Flow Diffusion for Efficient Frame Interpolation](image_generation/hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)**

:   HFD 提出在多尺度上用扩散模型显式去噪双向光流（而非在潜空间直接去噪），结合光流引导的编解码器图像合成器端到端联合训练，在精度上全面超越所有基线，同时推理速度比其他扩散方法快 10+ 倍。

**[HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](image_generation/hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)**

:   HMAR 将 VAR 的 next-scale 预测重构为 Markov 过程（仅依赖前一尺度的累积重建而非所有前序尺度），并在每个尺度内引入多步掩码生成来消除条件独立假设，配合自定义 IO-aware 块稀疏注意力核，在 ImageNet 上匹配或超越 VAR/DiT 质量的同时实现训练 2.5× 加速和推理 3× 内存缩减。

**[HSI: A Holistic Style Injector for Arbitrary Style Transfer](image_generation/hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)**

:   HSI提出了一种基于全局风格统计特征和逐元素乘法的风格迁移模块，用线性复杂度替代自注意力的二次复杂度，同时通过双关系学习机制提升风格化质量，在效果和效率上均超越现有方法。

**[ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](image_generation/ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)**

:   提出 ICE 两阶段框架，仅用单个 T2I 扩散模型从单张图像自动定位物体级概念并分解为内在属性（类别、颜色、材质），实现无标注、无额外模型的层次化视觉概念提取。

**[IDEA-Bench: How Far are Generative Models from Professional Designing?](image_generation/idea-bench_how_far_are_generative_models_from_professional_designing.md)**

:   提出首个面向专业级图像设计的综合基准 IDEA-Bench，涵盖 100 个真实设计任务（海报、绘本、字体、特效等）和 5 种输入输出模式，揭示当前最强模型仅获 22.48/100 分，距离专业设计仍有巨大鸿沟。

**[IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](image_generation/idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)**

:   IDProtector 提出首个前馈式对抗噪声编码器，通过单次前向传播为人脸照片添加不可感知的对抗扰动，可同时防御 InstantID、IP-Adapter、PhotoMaker 等多种编码器驱动的身份保持生成方法，且对 JPEG 压缩、缩放等变换保持鲁棒。

**[Image Generation Diversity Issues and How to Tame Them](image_generation/image_generation_diversity_issues_and_how_to_tame_them.md)**

:   本文揭示了当前扩散模型存在严重的多样性不足问题（最先进模型仅覆盖训练数据 77% 的多样性），提出了基于图像检索的 Image Retrieval Score (IRS) 作为可解释的多样性度量指标，并引入 Diversity-Aware Diffusion Models (DiADM) 在不损失生成质量的前提下提升多样性。

**[Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_generation/image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)**

:   本文模仿真实动画制作流程，提出一种基于扩散模型的图像参考草图上色框架，通过分割交叉注意力（Split Cross-Attention）配合可切换LoRA机制分别处理前景和背景的上色，消除了空间纠缠伪影（spatial entanglement），在4.8M图像上训练后在定性、定量和用户研究中均优于现有方法。

**[Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](image_generation/implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)**

:   本文提出隐式偏见注入攻击框架（IBI-Attacks），通过在文本嵌入空间中预计算一个通用的偏见方向向量，再利用自适应特征选择模块根据不同用户输入动态调整该向量，以即插即用的方式将隐式偏见（如情绪、文化倾向）植入预训练的文生图扩散模型中，同时保持生成内容的原始语义，80%+的攻击成功率下仅35.8%被人类试验者察觉。

**[Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](image_generation/improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)**

:   本文提出解耦退火后验采样（DAPS），通过在扩散采样过程中解耦相邻步骤的样本依赖关系，允许大幅度的非局部跳跃来修正早期采样错误，在非线性逆问题（如相位恢复）上大幅超越现有方法。

**[Improving Editability in Image Generation with Layer-wise Memory](image_generation/improving_editability_in_image_generation_with_layer-wise_memory.md)**

:   本文提出基于层级记忆的迭代图像编辑框架，通过存储每步编辑的 latent 和 prompt embedding，结合背景一致性引导（BCG）和多查询解耦注意力（MQD），实现多步顺序编辑中背景保持一致且新对象自然融入的效果。

**[InsightEdit: Towards Better Instruction Following for Image Editing](image_generation/insightedit_towards_better_instruction_following_for_image_editing.md)**

:   提出 InsightEdit，构建 250 万级高质量编辑数据集 AdvancedEdit，并设计双流桥接机制将 MLLM 的文本推理特征和图像语义特征同时注入扩散模型，在复杂指令跟随和背景一致性上达到 SOTA。

**[Instant Adversarial Purification with Adversarial Consistency Distillation](image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)**

:   提出 One Step Control Purification (OSCP) 框架，结合 Gaussian Adversarial Noise Distillation (GAND) 和 Controlled Adversarial Purification (CAP)，在单次 U-Net 推理（~0.1 秒）内完成对抗净化，相比传统扩散净化方法加速 100 倍。

**[InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](image_generation/interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)**

:   本文提出 InterAct 基准，整合并标准化了 21.81 小时的 3D 人物-物体交互数据（扩展到 30.70 小时），通过统一优化框架校正运动捕捉伪影并增强数据，定义六项生成任务和统一建模方法，在多个 HOI 生成任务上取得 SOTA 表现。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](image_generation/interedit_navigating_text-guided_multi-human_3d_motion_editing.md)**

:   提出 InterEdit，首个文本引导的多人 3D 运动交互编辑框架，通过 Semantic-Aware Plan Token Alignment 和 Interaction-Aware Frequency Token Alignment 在扩散模型中实现语义编辑的同时保持多人之间的时空耦合关系。

**[InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](image_generation/intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)**

:   InterMimic 提出了一个课程式教师-学生蒸馏框架，首次实现了单策略从大规模不完美 MoCap 数据中学习多样化的全身物理人物交互技能，通过教师策略先"完善"每个动作子集，再蒸馏到学生策略，并用 RL 微调超越简单模仿，最终支持零样本泛化和与运动生成器的无缝集成。

**[Interpretable Generative Models through Post-hoc Concept Bottlenecks](image_generation/interpretable_generative_models_through_post-hoc_concept_bottlenecks.md)**

:   本文提出两种低成本的后置方法——概念瓶颈自编码器(CB-AE)和概念控制器(CC)——将预训练生成模型转化为可解释且可操控的模型，无需从头训练或真实标注数据，在 CelebA/CelebA-HQ/CUB 上的可操控性(steerability)平均超过先前 CBGM 方法约25%，训练速度快4-15倍。

**[IterIS: Iterative Inference-Solving Alignment for LoRA Merging](image_generation/iteris_iterative_inference-solving_alignment_for_lora_merging.md)**

:   IterIS提出了一种迭代推理-求解的LoRA合并方法，通过直接提取统一适配器的输入特征（而非近似）来建立更准确的优化目标，配合正则化减少样本需求至先前方法的1-5%，并引入自适应权重平衡优化，在文本到图像扩散模型、视觉语言模型和大语言模型的LoRA合并中显著超越基线。

**[JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](image_generation/janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)**

:   提出 JanusFlow，将 rectified flow 直接嵌入自回归 LLM 框架，通过解耦理解/生成编码器 + 表征对齐正则化，在 1.3B 参数下同时达到多模态理解和图像生成的 SOTA。

**[K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](image_generation/k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)**

:   提出 K-LoRA，在每个 attention 层通过 Top-K 元素绝对值累加来比较主题 LoRA 和风格 LoRA 的重要性，自适应选择整层 LoRA 权重，配合时间步缩放因子，实现免训练的主题-风格高质量融合。

**[Language-Guided Image Tokenization for Generation](image_generation/language-guided_image_tokenization_for_generation.md)**

:   TexTok 提出在图像分词（tokenization）阶段引入文本描述作为条件，将高层语义信息卸载给文本，使图像 token 专注于编码细粒度视觉细节，从而在保持甚至提升重建质量的同时实现更高的压缩率，在 ImageNet 上取得了 SOTA 的生成 FID 分数 1.46。

**[Latent Space Imaging](image_generation/latent_space_imaging.md)**

:   Latent Space Imaging (LSI) 提出了一种将光学编码与生成模型解码结合的新成像范式，通过将图像信息直接编码到 StyleGAN 的语义隐空间中，实现 1:100 到 1:16384 的极端压缩比，同时仍能完成人脸重建、属性分类、分割和关键点检测等下游任务。

**[LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](image_generation/latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)**

:   LaTexBlend 通过在文本编码器后的潜在文本空间（Latent Textual Space）中表示和融合多个定制概念，实现了高保真、高效率的多概念定制图像生成，微调复杂度线性增长且推理无额外开销。

**[LaVin-DiT: Large Vision Diffusion Transformer](image_generation/lavin-dit_large_vision_diffusion_transformer.md)**

:   LaVin-DiT 提出一种基于扩散 Transformer 的大视觉基础模型，通过空间-时序 VAE 编码、联合扩散 Transformer 去噪、以及 in-context learning 实现超过 20 种视觉任务的统一处理，从 0.1B 扩展至 3.4B 参数，在多项任务上显著超越自回归式大视觉模型 LVM。

**[Learning Flow Fields in Attention for Controllable Person Image Generation](image_generation/learning_flow_fields_in_attention_for_controllable_person_image_generation.md)**

:   提出 Leffa（Learning Flow Fields in Attention），在扩散模型的注意力层中将 attention map 转换为流场并进行像素级正则化监督，显式引导 target query 关注正确的 reference key 区域，**零额外推理开销**地减少细粒度细节（纹理、文字、logo）失真，在虚拟试衣（VITON-HD、DressCode）和姿态迁移（DeepFashion）上均 SOTA。

**[Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](image_generation/learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)**

:   提出PAG（Prompt Adaptation with GFlowNets），将提示词适配重新定义为概率推断问题，利用GFlowNets从奖励分布中采样而非最大化奖励，结合流重激活、奖励优先采样和奖励分解三大技术解决模式坍塌问题，生成既高质量又多样化的文本到图像提示词。

**[Learning Visual Generative Priors without Text](image_generation/learning_visual_generative_priors_without_text.md)**

:   提出Lumos框架，通过纯视觉的图像到图像（I2I）自监督预训练学习视觉生成先验，然后仅用1/10的文本-图像对微调即可达到甚至超越现有T2I模型的效果，并在文本无关的视觉任务（I2V、NVS）上展现出优于T2I先验的性能。

**[LEDiff: Latent Exposure Diffusion for HDR Generation](image_generation/lediff_latent_exposure_diffusion_for_hdr_generation.md)**

:   提出LEDiff，通过在预训练扩散模型的潜空间中进行曝光融合（而非图像空间），用少量HDR数据微调VAE解码器和去噪器，让现有生成模型具备HDR生成能力，同时实现SOTA级别的LDR到HDR转换。

**[Lifting Motion to the 3D World via 2D Diffusion](image_generation/lifting_motion_to_the_3d_world_via_2d_diffusion.md)**

:   MVLift提出了一个多阶段框架，仅使用单视角2D姿态序列训练，通过线条件扩散模型→多视角优化→合成数据生成→多视角扩散模型的渐进策略建立多视角一致性，实现无需3D监督的全局3D运动（含关节旋转+根轨迹）估计，在AIST++上根轨迹误差67.6mm超越需要3D监督的WHAM (164.3mm)。

**[LoRACLR: Contrastive Adaptation for Customization of Diffusion Models](image_generation/loraclr_contrastive_adaptation_for_customization_of_diffusion_models.md)**

:   LoRACLR 提出一种基于对比学习目标的 LoRA 模型合并方法，通过学习一个 delta 权重将多个独立训练的单概念 LoRA 模型融合为一个统一模型，无需重训练或访问原始训练数据，即可实现高保真的多概念图像生成，合并 12 个概念仅需 5 分钟。

**[lbGen: Low-Biased General Annotated Dataset Generation](image_generation/low-biased_general_annotated_dataset_generation.md)**

:   提出 lbGen 框架，通过双层语义对齐（全局对抗+个体余弦相似度）和质量保证损失微调 Stable Diffusion，仅用类别名称即可生成低偏差的通用标注数据集，预训练骨干比 ImageNet 真实数据平均迁移精度高出 1.7%~2.1%。

**[LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](image_generation/luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)**

:   提出 LumiNet，将源图像的潜在内在特征（128 维 albedo-like 表征）和目标图像的潜在外在光照码（16 维）注入改造后的 ControlNet，实现仅用图像输入的室内场景级光照迁移，包含镜面高光、阴影和间接照明等复杂效果。

**[MagicQuill: An Intelligent Interactive Image Editing System](image_generation/magicquill_an_intelligent_interactive_image_editing_system.md)**

:   提出 MagicQuill 智能交互式图像编辑系统，用三种笔触（添加/减去/颜色）表达编辑意图，双分支扩散插件（inpainting + control）实现边缘和颜色的精细控制，MLLM 实时猜测意图自动生成 prompt，形成无需手动输入文字的连续编辑工作流。

**[MangaNinja: Line Art Colorization with Precise Reference Following](image_generation/manganinja_line_art_colorization_with_precise_reference_following.md)**

:   MangaNinja 是一个基于扩散模型的参考图引导线稿上色方法，通过渐进式 Patch Shuffling 策略训练模型学会局部语义匹配能力，并引入 PointNet 驱动的点控制机制实现精细颜色对应，在大姿态差异、多参考图、跨角色上色等挑战场景中显著超越现有方法。

**[MARBLE: Material Recomposition and Blending in CLIP-Space](image_generation/marble_material_recomposition_and_blending_in_clip-space.md)**

:   仅在 CLIP 空间操作材质嵌入，通过定向注入 UNet 中的材质响应层实现材质迁移和混合，并通过轻量 MLP 预测属性编辑方向实现粗糙度/金属度/透明度/发光的参数化控制，无需微调扩散模型。

**[Memories of Forgotten Concepts](image_generation/memories_of_forgotten_concepts.md)**

:   本文揭示了扩散模型中概念擦除方法的根本缺陷——通过扩散反演找到高似然度的潜变量种子，证明被擦除的概念信息仍然存留在模型中，且可以从多个不同的种子向量重建出被擦除概念的高质量图像。

**[MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis](image_generation/metashadow_object-centered_shadow_detection_removal_and_synthesis.md)**

:   MetaShadow 提出首个三合一框架，将基于GAN的 Shadow Analyzer（阴影检测+去除）与基于扩散模型的 Shadow Synthesizer（阴影合成）协同结合，通过 GAN 中间特征引导扩散模型进行阴影知识迁移，在三个阴影任务上均达到 SOTA。

**[MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification](image_generation/mexd_an_expert-infused_diffusion_model_for_whole-slide_image_classification.md)**

:   MExD 首次将生成式扩散模型应用于全切片图像（WSI）分类，通过动态混合专家（Dyn-MoE）聚合器筛选关键实例并提供条件信息，结合扩散分类器（Diff-C）从噪声中迭代还原类别标签，在Camelyon16、TCGA-NSCLC和BRACS三个基准上达到SOTA。

**[MINIMA: Modality Invariant Image Matching](image_generation/minima_modality_invariant_image_matching.md)**

:   MINIMA 提出了一个统一的跨模态图像匹配框架，通过设计数据引擎从廉价的 RGB 图像对中生成多模态合成数据集 MD-syn（480M 对），使任何现有匹配管线仅需微调即可获得跨模态匹配能力，在 19 种跨模态场景下显著超越模态特定方法。

**[Minority-Focused Text-to-Image Generation via Prompt Optimization](image_generation/minority-focused_text-to-image_generation_via_prompt_optimization.md)**

:   MinorityPrompt 提出了一种在线 prompt 优化框架，通过在推理过程中迭代优化可学习 token embedding 来最大化似然度损失，引导 T2I 扩散模型生成处于数据分布低密度区域的少数(minority)样本，同时保持语义一致性和生成质量。

**[MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](image_generation/mirrorverse_pushing_diffusion_models_to_realistically_reflect_the_world.md)**

:   MirrorVerse 通过构建增强的合成数据集 SynMirrorV2（包含随机位姿、旋转和多物体场景），配合三阶段课程式训练策略，训练出 MirrorFusion 2.0 模型，首次使扩散模型能够生成逼真的镜面反射，在合成和真实场景中均显著超越前方法。

**[MixerMDM: Learnable Composition of Human Motion Diffusion Models](image_generation/mixermdm_learnable_composition_of_human_motion_diffusion_models.md)**

:   提出 MixerMDM，首个可学习的运动扩散模型组合技术，通过 Transformer-based Mixer 模块预测动态混合权重，以对抗训练方式学习如何融合个体运动和交互运动扩散模型，实现细粒度可控的人-人交互运动生成。

**[MLLM-as-a-Judge for Image Safety without Human Labeling](image_generation/mllm-as-a-judge_for_image_safety_without_human_labeling.md)**

:   提出 CLUE 框架，通过规则客观化、CLIP 相关性扫描、前置条件链分解和去偏 token 概率分析，实现无需人工标注的零样本图像安全判定，在多个 MLLM 上大幅超越基线。

**[MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](image_generation/mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)**

:   首次将连续图像表示与离散文本表示整合到统一自回归概率建模框架中，通过轻量扩散头替代 VQ 离散化避免信息损失，并推导出 v-prediction 为最优参数化以解决低精度训练下的数值误差问题。

**[MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices](image_generation/mobileportrait_real-time_one-shot_neural_head_avatars_on_mobile_devices.md)**

:   提出首个可在移动端实时运行的单张人脸头像动画方法 MobilePortrait，通过混合显隐式关键点 + 预计算外观知识，仅用 16 GFLOPs 即匹敌 SOTA（100–600+ GFLOPs）的效果。

**[Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion](image_generation/mono2stereo_a_benchmark_and_empirical_study_for_stereo_conversion.md)**

:   构建首个大规模立体转换基准 Mono2Stereo（240 万对），提出立体质量指标 SIoU（与人类判断相关性 0.84 Spearman）和双条件扩散模型 + Edge Consistency 损失，同时解决单阶段方法立体效果弱和两阶段方法图像质量差的矛盾。

**[Move-in-2D: 2D-Conditioned Human Motion Generation](image_generation/move-in-2d_2d-conditioned_human_motion_generation.md)**

:   定义 2D 场景图像+文本条件下的人体运动生成新任务，构建 30 万级 HiC-Motion 数据集，通过 in-context conditioning 扩散 Transformer 生成可自然投影到场景的运动序列，赋能下游人体视频生成。

**[MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](image_generation/mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)**

:   MTADiffusion通过构建500万张图像的Mask-Text对齐数据集、联合训练修复与边缘预测任务、以及基于VGG Gram矩阵的风格一致性损失，同时解决了对象修复中的语义错位、结构扭曲和风格不一致三大问题，在BrushBench和EditBench上达到SOTA。

**[Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](image_generation/multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)**

:   MCLD通过将源人物图像解耦为面部区域、外观纹理和整体图像三个焦点条件，设计多焦点条件聚合模块(MFCA)在UNet不同阶段选择性注入不同条件，有效缓解了LDM压缩导致的面部和纹理细节退化问题，在DeepFashion上取得SOTA。

**[Multi-party Collaborative Attention Control for Image Customization](image_generation/multi-party_collaborative_attention_control_for_image_customization.md)**

:   提出 MCA-Ctrl，一种无需微调的图像定制方法，通过三个并行扩散过程的自注意力协同控制，实现文本和图像条件下的高质量主体驱动编辑与生成，同时引入主体定位模块解决复杂视觉场景中的特征泄漏和混淆问题。

**[Multitwine: Multi-Object Compositing with Text and Layout Control](image_generation/multitwine_multi-object_compositing_with_text_and_layout_control.md)**

:   本文提出首个支持文本和布局引导的多目标同时合成（compositing）生成模型Multitwine，通过联合训练合成与个性化生成任务，结合跨注意力/自注意力解耦损失，实现同时插入多个对象的自然交互（如拥抱、弹吉他），用户研究中交互真实性偏好率最高达97.1%。

**[MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](image_generation/mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)**

:   本文提出MVPortrait，一个两阶段文本引导框架（Text2FLAME + FLAME2Video），通过将FLAME 3D参数化面部模型作为中间表示，分别用MotionDM和EmotionDM扩散模型生成运动和表情参数序列，再用多视角视频生成模型将FLAME渲染序列转化为逼真的多视角肖像动画，首次实现文本/语音/视频三种信号兼容的可控肖像动画。

**[Navigating Image Restoration with VAR's Distribution Alignment Prior](image_generation/navigating_image_restoration_with_vars_distribution_alignment_prior.md)**

:   本文发现Visual AutoRegressive (VAR) 模型的next-scale预测具有天然的多尺度分布对齐能力——低尺度修复全局退化（如低光照、雾霾），高尺度修复局部退化（如噪声、雨滴），基于此构建VarFormer框架，通过Degradation-Aware Enhancement (DAE)自适应选择尺度先验、Adaptive Feature Transformation (AFT)融合先验与退化特征，在6类恢复任务上超越现有multi-task方法。

**[Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](image_generation/nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)**

:   本文提出FastProtect，首个关注延迟的图像保护框架，通过预训练Mixture-of-Perturbations (MoP)替代传统逐图迭代优化，配合Multi-Layer Protection Loss增强训练效果、Adaptive Targeted Protection和Adaptive Protection Strength优化推理，实现了比现有最快方法PhotoGuard快175×（A100 GPU上0.04秒 vs 7秒处理512²图像）的实时保护，同时保持相当的保护效力和更优的不可见性。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   揭示 DiT 的计算在空间 token 上均匀分配（不会把多余计算重分配到困难区域），提出 ELIT——在 DiT 中插入可变长度的 latent interface（Read/Write 交叉注意力），训练时随机丢弃尾部 latent 学出重要性排序，推理时通过调节 latent 数量实现平滑的质量-FLOPs 权衡，ImageNet 512px 上 FID 降低 53%。

**[Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](image_generation/overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)**

:   提出 Concept-Gated Visual Distillation (CGVD)，一种无需训练的推理时框架，通过语言指令解析 → SAM3 分割 → 集合论交叉验证 → LaMa 修复的流水线，从 VLA 模型的视觉输入中选择性移除语义干扰物，在高度杂乱场景中将 π₀ 的操作成功率从 43.0% 提升至 77.5%。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 去噪器（Auto-Correction + Directional Correction + Score-Based Denoising 三阶段），解决将 score-based 扩散先验嵌入 ADMM-PnP 框架时的流形不匹配问题，并首次建立了 score-based 去噪器在 ADMM 中的收敛性理论保证，在去噪、修复、去模糊、超分辨、相位恢复、HDR 等逆问题上一致超越现有基线。

**[Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](image_generation/trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)**

:   提出 FIRM 框架——通过"差异优先"（编辑）和"计划-打分"（生成）的数据构建流水线训练专用奖励模型（FIRM-Edit-8B / FIRM-Gen-8B），配合"Base-and-Bonus"奖励策略（CME/QMA）解决 RL 中的奖励 hacking 问题，在图像编辑和 T2I 生成任务上均取得 SOTA。

**[UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations](image_generation/unicom_unified_multimodal_modeling_via_compressed_continuous_semantic_representa.md)**

:   提出 UniCom，通过对 VLM 连续语义特征进行**通道维度压缩**（而非空间下采样），构建紧凑连续表示空间，用 Transfusion 架构统一多模态理解与生成，在统一模型中达到 SOTA 生成质量。

**[Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](image_generation/unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)**

:   提出U2Diff，一个统一的扩散模型框架，能同时处理多智能体轨迹补全和预测任务，通过增强去噪损失提供逐状态不确定性估计，并引入Rank Neural Network对生成的多模态预测进行误差概率排序。

**[V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像复原重新定义为**渐进式视频生成过程**，利用预训练视频生成模型（Wan2.2-TI2V-5B）的先验知识，仅用 1,000 个多任务训练样本（不到现有方法的 2%）即可实现竞争力的多任务图像复原。

**[Visual-ERM: Reward Modeling for Visual Equivalence](image_generation/visual-erm_reward_modeling_for_visual_equivalence.md)**

:   提出 Visual-ERM，一个多模态生成式奖励模型，在视觉空间中直接评估 vision-to-code 任务的渲染质量，提供细粒度、可解释、任务无关的奖励信号，用于 RL 训练和测试时缩放。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](multimodal_vlm/4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)**

:   提出4D LangSplat，通过多模态大语言模型生成逐物体视频caption来构建4D语言场，结合状态可变形网络建模语义的时间连续演变，首次实现动态场景中时间敏感和时间无关的开放词汇查询。

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](multimodal_vlm/a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

**[Active Data Curation Effectively Distills Large-Scale Multimodal Models](multimodal_vlm/active_data_curation_effectively_distills_large-scale_multimodal_models.md)**

:   提出 ACID（主动数据筛选即隐式蒸馏）和 ACED（结合显式蒸馏），证明用大模型作为参考来主动筛选训练数据是一种比传统知识蒸馏更有效的多模态模型压缩方式，两者互补结合后在 27 个零样本任务上以更少推理 FLOPs 达到 SOTA。

**[Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](multimodal_vlm/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)**

:   提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

**[Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning](multimodal_vlm/beyond_words_augmenting_discriminative_richness_via_diffusions_in_unsupervised_p.md)**

:   提出AiR（Augmenting discriminative Richness）方法，利用LoRA微调的Stable Diffusion生成合成图像构建辅助分类器，与文本分类器互补融合，将无监督prompt learning中的文本-图像匹配扩展为图像-图像匹配，显著提升细粒度/遥感等困难数据集上的分类准确率。

**[Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](multimodal_vlm/can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)**

:   系统研究了VLM在语义定位任务中的自我纠错能力，发现内在自我纠错（无外部反馈）反而损害性能（-7至-17点），但通过同一VLM作为二值验证器提供反馈的迭代纠错最多可提升8.4个百分点，揭示了反馈质量是自我纠错的关键瓶颈。

**[COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection](multimodal_vlm/coap_memory-efficient_training_with_correlation-aware_gradient_projection.md)**

**[CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)**

:   针对现有交错图文数据集（MMC4/OBELICS）叙事连贯性差、实体风格不一致的核心问题，构建 CoMM 数据集（227K 文档、2.28M 图片），通过定向采集指令型内容 + 三维质量过滤策略确保文本连贯、图像一致、图文对齐，并提出 4 个交错生成评测任务。

**[Completion as Enhancement: A Degradation-Aware Selective Image Guided Network](multimodal_vlm/completion_as_enhancement_a_degradation-aware_selective_image_guided_network_for.md)**

:   将图像增强重构为'补全'范式，通过退化感知选择机制引导网络聚焦于需要增强的区域，避免对已清晰区域的过度处理

**[Context-Aware Multimodal Pretraining](multimodal_vlm/context-aware_multimodal_pretraining.md)**

:   本文提出LIxP（Language-Image Contextual Pretraining），通过在对比式图文预训练中引入交叉注意力上下文化机制，使视觉-语言模型在不损失零样本性能的前提下，显著提升了基于度量的few-shot适应能力（21个下游任务平均提升5%以上，样本效率提升可达4倍）。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL 框架，通过对抗性锚点（DPGD）精准探测新旧任务语义边界的脆弱区域，结合跨模态几何蒸馏（ACGD）和文本语义正则化（TSGR）保护 VLM 的跨模态几何结构，在五个持续学习 benchmark 上达到 SOTA。

**[COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](multimodal_vlm/counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)**

:   本文构建了COUNTS——一个包含14种自然分布偏移、222K+样本和119万+标注框的大规模OOD数据集，并提出O(OD)²和OODG两个基准，系统评估了目标检测器和多模态大模型在分布偏移下的泛化能力，发现即使是GPT-4o也仅能达到56.7%的定位准确率。

**[Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning](multimodal_vlm/critic-v_vlm_critics_help_catch_vlm_errors_in_multimodal_reasoning.md)**

:   本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。

**[Cropper: Vision-Language Model for Image Cropping through In-Context Learning](multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)**

:   本文提出Cropper框架，首次利用大型视觉-语言模型（VLM）的上下文学习（ICL）能力来解决图像裁剪任务，通过高效的prompt检索和基于反馈的迭代裁剪优化策略，无需任何训练即可在自由裁剪、主体感知裁剪和宽高比裁剪三种任务上大幅超越有监督SOTA方法。

**[Cross-modal Information Flow in Multimodal Large Language Models](multimodal_vlm/cross-modal_information_flow_in_multimodal_large_language_models.md)**

:   通过"attention knockout"方法系统性地追踪 MLLM 中视觉和语言信息的流动路径，发现视觉信息分两阶段（先全局后局部）融入语言表征，最终在中间层由问题位置传播到最后位置生成答案。

**[Data Distributional Properties as Inductive Bias for Systematic Generalization](multimodal_vlm/data_distributional_properties_as_inductive_bias_for_systematic_generalization.md)**

:   发现仅通过操纵训练数据的分布性质（多样性、突发性、潜在干预）就能诱导多模态遮蔽语言模型实现系统性泛化，其中增加属性多样性可将 OOD 形状预测准确率从 0.6% 提升到 90%，无需任何模型架构或训练策略修改。

**[Distraction is All You Need for Multimodal Large Language Model Jailbreaking](multimodal_vlm/distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)**

:   提出"分散假说"——通过构造高对比度多子图复合输入增加视觉复杂度来制造 OOD 效果，配合查询分解和精心设计的无害指令，实现对 GPT-4o 等闭源 MLLM 高达 42-64% 攻击成功率的黑盒越狱。

**[Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](multimodal_vlm/document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)**

:   提出 DocHaystack 和 InfoHaystack 两个大规模文档检索基准（每个问题对应 1000+ 文档），以及 V-RAG——一个视觉中心的检索增强生成框架，在 Recall@1 上比最佳基线提升 9%-11%。

**[DocVLM: Make Your VLM an Efficient Reader](multimodal_vlm/docvlm_make_your_vlm_an_efficient_reader.md)**

:   提出一种模型无关的 OCR 编码模块，将 OCR 提取的文本和布局信息压缩为 64 个 learned query token 并注入冻结的 VLM，在极低视觉 token 数量下大幅提升文档理解能力（DocVQA 最高 +30.6 分），并零样本泛化到多页文档。

**[DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](multimodal_vlm/dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)**

:   提出双提示协作（DPC）框架，通过冻结原始调优提示保持新类泛化、训练并行提示强化基类性能，配合加权解耦推理机制，作为即插即用模块在 4 种 prompt tuning 基线上一致提升 base-new 调和均值。

**[DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution](multimodal_vlm/dynrefer_delving_into_region-level_multimodal_tasks_via_dynamic_resolution.md)**

:   模拟人眼"注视+扫视"的动态分辨率机制，围绕目标区域构建多层嵌套视图并在训练时随机采样、推理时根据任务或图像先验选择性组合，以 4.2B 参数在区域描述、属性检测、密集描述等任务上全面超越 7B+ 模型。

**[EgoLM: Multi-Modal Language Model of Egocentric Motions](multimodal_vlm/egolm_multi-modal_language_model_of_egocentric_motions.md)**

:   提出统一自我中心动作追踪（稀疏传感器→全身动作）和动作理解（动作→语言描述）的多模态语言模型框架，通过 VQ-VAE 动作 tokenizer + GPT-2 骨干实现四种模态（文本、动作 token、传感器、视频）的联合建模，加入自我中心视频后追踪误差降低 10-20mm。

**[Embodied Scene Understanding for Vision Language Models via MetaVQA](multimodal_vlm/embodied_scene_understanding_for_vision_language_models_via_metavqa.md)**

:   构建了一个基于 Set-of-Mark 标注和场景图的大规模 VQA 基准（430 万问题），系统评估 VLM 的空间推理和具身理解能力，发现在 MetaVQA 上微调可显著提升空间推理（+28 点），且训练于仿真数据的能力可零样本迁移到真实场景和未见过的闭环驾驶任务。

**[ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)**

:   提出 Espire，一个基于仿真环境的具身空间推理诊断基准，将 VLM 评估分解为定位和执行两阶段，通过全生成式范式系统评估 VLM 在多种空间推理维度和粒度上的能力。

**[Evaluating Model Perception of Color Illusions in Photorealistic Scenes](multimodal_vlm/evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)**

:   本文提出了一套自动化框架生成包含 19,000 张真实感颜色错觉图像的 RCID 数据集，首次系统性揭示了 VLM 确实存在类人的颜色感知偏差，并通过混合训练方法使模型能同时理解人类感知和真实像素值。

**[EventGPT: Event Stream Understanding with Multimodal Large Language Models](multimodal_vlm/eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)**

:   首个专为事件相机流设计的 MLLM，通过三阶段渐进训练范式（视觉-语言对齐→事件-语言对齐→指令微调）跨越异步事件数据与语言之间的巨大领域差距，在事件场景描述和 VQA 上大幅超越通用 MLLM。

**[Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond](multimodal_vlm/every_sam_drop_counts_embracing_semantic_priors_for_multi-modality_image_fusion_.md)**

:   利用 SAM 的语义先验通过持久注意力模块增强红外-可见光图像融合，再通过双层优化知识蒸馏将语义知识转移到仅 0.136M 参数的轻量子网络，实现无需 SAM 的 10.47ms 推理，同时在分割任务上超越所有专用融合方法 3+ mIoU。

**[FastVLM: Efficient Vision Encoding for Vision Language Models](multimodal_vlm/fastvlm_efficient_vision_encoding_for_vision_language_models.md)**

:   提出混合卷积-Transformer视觉编码器 FastViTHD，通过 5 阶段架构实现 32× 空间下采样，在同等精度下比 ViT-L/14 生成 16× 更少的视觉 token 且编码速度提升 3.7×，TTFT 降低高达 85×。

**[Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation](multimodal_vlm/finer-cam_spotting_the_difference_reveals_finer_details_for_visual_explanation.md)**

:   将 CAM 的解释目标从单类 logit $y^c$ 改为类间差值 $y^c - \gamma \cdot y^d$（目标类与相似类的 logit 差），零额外参数地将任何 CAM 方法升级为细粒度版本，使激活图从"整体轮廓"细化到"区分性局部细节"。

**[FLAIR: VLM with Fine-grained Language-informed Image Representations](multimodal_vlm/flair_vlm_with_fine-grained_language-informed_image_representations.md)**

:   提出文本条件注意力池化（text-conditioned attention pooling），用文本 embedding 作为 query 从局部图像 token 中自适应聚合相关视觉信息，仅用 30M 合成描述数据训练就在细粒度检索和零样本分割上大幅超越用数十亿数据训练的 SigLIP/OpenCLIP。

**[Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](multimodal_vlm/florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)**

:   用生成式视觉基础模型 Florence-2 替换 CLIP 作为 VLM 视觉编码器，通过"深度-广度融合"（DBFusion）整合底层 DaViT 特征和三种任务提示（描述/OCR/定位）的高层特征，以单编码器 576 token 实现超越多编码器方案的性能。

**[ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](multimodal_vlm/forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)**

:   发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

**[Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](multimodal_vlm/free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)**

:   FreeTTA 提出一种无需训练、无需存储历史数据的测试时适应方法，通过在线 EM 算法显式建模目标域分布，利用 CLIP 零样本预测作为先验迭代估计每个类别的高斯分布参数，在 15 个数据集上稳定超越现有 TTA 方法。

**[From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](multimodal_vlm/from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)**

:   GEA 将预训练的多模态 LLM（LLaVA-OneVision）通过学习式多具身动作分词器适配到操控/导航/游戏/UI控制/规划五大领域，先用 220 万条跨域专家轨迹 SFT，再用在线 PPO 强化学习微调，单模型在多个基准上超越或接近领域专用模型。

**[Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](multimodal_vlm/generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)**

:   GFS-VL 提出一种广义小样本 3D 点云分割框架，通过将 3D 视觉语言模型（3D VLM）生成的稠密但有噪声的伪标签与精确但稀疏的小样本标注协同融合——经由原型引导的伪标签筛选、自适应填充和 novel-base 混合增强——在现有和新设的高难度 benchmark 上取得了 SOTA 性能。

**[GeoMM: On Geodesic Perspective for Multi-Modal Learning](multimodal_vlm/geomm_on_geodesic_perspective_for_multi-modal_learning.md)**

:   首次将测地距离（Geodesic Distance）引入多模态对比学习，通过构建层次化图结构高效计算样本间的流形距离，替代传统余弦距离，从而更准确地挖掘正负样本关系，提升图文检索、VQA等下游任务性能。

**[Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels](multimodal_vlm/ground-v_teaching_vlms_to_ground_complex_instructions_in_pixels.md)**

:   构建了Ground-V，一个包含50万指令-分割对的数据集，系统性解决真实世界指代分割中的五大挑战（幻觉引用、多对象、推理、多粒度、部件引用），训练后的VLM在gRefCOCO上N-Acc超越前SOTA 20%以上。

**[HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](multimodal_vlm/halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)**

:   提出HalLoc，一个15.5万样本、覆盖VQA/指令跟随/图像描述三类任务的token级幻觉标注数据集，并基于此训练了轻量级幻觉检测模型HalLocalizer，可在不影响效率的前提下即插即用地集成到现有VLM中实现实时概率化幻觉检测。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](multimodal_vlm/hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)**

:   通过对 attention 公式的精确分解，揭示 ICL 的效果本质上是 query-dependent 的标准自注意力输出与上下文 value 的动态混合，据此提出直接参数化"虚拟 KV 对"（低秩分解）来高保真模拟 ICL，仅 2.2M 参数即超越 MimIC/LoRA，且训练快 7.5 倍。

**[HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](multimodal_vlm/homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)**

:   提出HomeSafe-Bench——一个包含438个案例覆盖六大家庭功能区域的不安全动作检测基准，并设计HD-Guard双脑分层流式架构，以轻量FastBrain做高频筛查、大模型SlowBrain做深度推理，平衡家庭机器人安全监控的实时性与准确性。

**[Hyperbolic Safety-Aware Vision-Language Models](multimodal_vlm/hyperbolic_safety-aware_vision-language_models.md)**

:   HySAC 提出在双曲空间中构建安全感知的视觉语言模型，通过蕴含锥（entailment cone）将安全/不安全内容映射到双曲空间的不同区域（安全内容靠近原点、不安全内容远离原点），使模型具备安全内容分类和动态重定向能力，在检索安全性和NSFW检测上显著超越现有遗忘方法。

**[Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](multimodal_vlm/identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)**

:   本文发现多图大视觉语言模型（LVLM）存在严重的位置偏差——开源模型偏重后置图片、闭源模型忽视中间图片——并提出了一种无需训练的SoFt Attention（SoFA）方法，通过在图像间因果注意力与双向注意力之间做线性插值来缓解该偏差，在多个基准上提升了2~3%的平均准确率。

**[Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](multimodal_vlm/img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)**

:   提出一种受对比学习启发的数据合成方法，自动生成包含细微物体差异的相似图像对及其差异描述，用于微调MLLM后在MMVP上超越GPT-4V和Gemini达12个点，并在8个通用MLLM基准上平均提升3.06%。

**[Improving Personalized Search with Regularized Low-Rank Parameter Updates](multimodal_vlm/improving_personalized_search_with_regularized_low-rank_parameter_updates.md)**

:   本文提出POLAR方法，通过对CLIP文本编码器**最后一层**的value矩阵施加**rank-1的LoRA更新**加正则化，仅用少量样本即可学习个性化概念并保留通用知识，在DeepFashion2和ConCon-Chi基准上超越基于文本反转的先前方法4%~22%。

**[Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](multimodal_vlm/insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)**

:   Insight-V 提出一个包含数据生成 pipeline 和多智能体推理系统的视觉推理增强方案：通过渐进式生成+多粒度评估构建高质量长链推理数据，设计推理Agent和总结Agent协作解题，配合迭代DPO进一步提升推理质量，在7个视觉推理基准上实现平均7%的提升。

**[It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](multimodal_vlm/its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)**

:   本文首次系统研究了在**完全无配对数据**的情况下，仅利用视觉和语言嵌入空间各自内部的成对距离进行"盲匹配"的可行性，提出了一种分解式Hahn-Grant QAP求解器（内存从 $O(N^4)$ 降到 $O(N^3)$），并在33个视觉模型×27个语言模型的大规模实验中证明了该匹配的可行性，甚至实现了无监督图像分类。

**[LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](multimodal_vlm/layoutvlm_differentiable_optimization_of_3d_layout_via_vision-language_models.md)**

:   提出LayoutVLM，利用VLM的语义知识生成包含数值位姿估计和空间关系约束的双重场景布局表示，通过可微分优化联合优化语义目标和物理合理性约束，在11种房间类型上显著超越现有方法。

**[LLaVA-Critic: Learning to Evaluate Multimodal Models](multimodal_vlm/llava-critic_learning_to_evaluate_multimodal_models.md)**

:   LLaVA-Critic 是首个开源的通用多模态评估模型，通过在精心构建的113k评估指令数据上训练，使开源LMM具备了接近GPT-4o水平的Pointwise评分和Pairwise排序能力，并可作为奖励模型为迭代DPO提供有效的偏好信号，超越基于人类反馈训练的LLaVA-RLHF奖励模型。

**[MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)**

:   提出VQAMask预训练范式，在VQA文本解析基础上引入辅助的Mask生成任务（推理时丢弃），通过显式的空间对齐监督增强视觉编码器对文档图像中文字区域的感知能力，建立Marten模型在多项文档理解任务上达到8B级MLLM的SOTA。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](multimodal_vlm/mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   构建首个包含正负语义描述的视觉定位数据集 D-Negation，并提出 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义约束显著增强 grounding 模型对否定语义的理解能力。

**[MBQ: Modality-Balanced Quantization for Large Vision-Language Models](multimodal_vlm/mbq_modality-balanced_quantization_for_large_vision-language_models.md)**

:   发现大型VLM中视觉token和语言token对量化误差的敏感度差异超过10倍，提出MBQ方法在量化校准过程中引入基于梯度的模态平衡因子，在W3A16和W4A8设置下分别提升精度最高4.4%和11.6%，并实现1.4倍端到端加速。

**[Mimic In-Context Learning for Multimodal Tasks](multimodal_vlm/mimic_in-context_learning_for_multimodal_tasks.md)**

:   本文从数学角度分析了ICL中in-context demonstrations (ICDs)对自注意力的"移位效应"，并提出MimIC方法通过在每个注意力头插入可学习移位向量+query依赖的缩放因子来模拟ICL行为，在VQA和Captioning任务上以仅0.26M参数超越32-shot ICL和所有现有移位向量方法。

**[MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](multimodal_vlm/mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)**

:   本文提出MIMO——首个同时支持"视觉引用多模态输入"（用户通过点/框指定感兴趣区域）和"像素级定位多模态输出"（模型在文本回答中嵌入分割mask）的医学视觉语言模型，并构建了895K样本的MIMOSeg数据集，在多种医学VQA和分割任务上展示了独特的referring+grounding能力。

**[MMRL: Multi-Modal Representation Learning for Vision-Language Models](multimodal_vlm/mmrl_multi-modal_representation_learning_for_vision-language_models.md)**

:   MMRL 提出了一个共享的、模态无关的可学习表征空间，将表征 token 投影到图像和文本编码器的高层（保留低层泛化知识），并通过解耦推理策略（基类用表征+类别特征，新类只用类别特征）在 15 个数据集上实现了 few-shot 适配与泛化的最优平衡，刷新了 base-to-novel 泛化的 SOTA。

**[Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)**

:   本文提出Molmo系列VLM和PixMo数据集，完全不依赖闭源VLM的合成数据，通过创新的数据收集方式（语音描述图像、交互式问答标注、2D指向标注）从零构建高质量训练数据，其72B模型在学术基准和人类评估中超越Claude 3.5 Sonnet和Gemini 1.5 Pro，仅次于GPT-4o。

**[Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](multimodal_vlm/mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)**

:   本文提出MM-Graph——首个同时包含文本和视觉节点属性的综合性图学习基准，涵盖7个不同规模的真实数据集和3类图任务（链接预测/节点分类/知识图谱补全），系统评估了视觉信息对图学习的影响，揭示了"多模态GNN不如传统GNN"和"特征对齐至关重要"等关键发现。

**[MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](multimodal_vlm/move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)**

:   本文提出MoVE-KD——首个从知识蒸馏角度将多个视觉编码器（CLIP/EVA/ConvNeXt/SAM）的特长融合到单个编码器的框架，通过Mixture-of-LoRA-Experts (MoLE)缓解多教师知识冲突、利用CLIP的[CLS]注意力自适应加权蒸馏token和教师，在LLaVA/LLaVA-NeXT上实现一致提升。

**[Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)**

:   本文系统研究了多模态 LLM 中多层视觉特征融合的两个核心问题：**(1) 如何选择最有效的视觉层**和 **(2) 如何最好地融合到语言模型中**，发现从不同表示相似性阶段各选一层 + 外部直接融合是最优实践。

**[Multimodal OCR: Parse Anything from Documents](multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)**

:   提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。

**[MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](multimodal_vlm/mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)**

:   本文提出 MV-MATH 基准，包含 2,009 道高质量多图数学题（来自真实 K-12 场景），系统评估了 25 个多模态大模型在多图数学推理场景下的能力，发现所有模型远低于人类水平（最佳 Claude 仅 33.9%），揭示了多图数学推理仍是 MLLM 的重大挑战。

**[NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)**

:   本文发现在 CLIP 提示学习中简单替换 MAE 损失就能显著提升对噪声标签的鲁棒性，并通过特征学习理论证明了这一现象，进而提出 NLPrompt 方法——结合基于最优传输的数据净化（PromptOT）将数据分为干净/噪声子集后分别用 CE 和 MAE 损失训练，在多种噪声设置下大幅超越现有方法。

**[NVILA: Efficient Frontier Visual Language Models](multimodal_vlm/nvila_efficient_frontier_visual_language_models.md)**

:   NVILA 提出"先放大再压缩"(Scale-then-Compress)的范式，通过提升空间和时间分辨率后再压缩视觉Token，在保持甚至超越SOTA精度的同时，将训练成本降低1.9-5.1倍、推理预填充延迟降低1.6-2.2倍、解码延迟降低1.2-2.8倍。

**[Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](multimodal_vlm/octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)**

:   本文揭示了多模态大模型幻觉的混合特性——不同样本甚至同一回答中的不同 token 面临不同类型的幻觉挑战（语言先验、视觉信息丢失、注意力偏差），据此提出 Octopus 框架，通过可学习的"眼睛"模块自适应识别幻觉类型，动态选择最适合的对比解码策略（"触手"），在四个基准上实现了 SOTA。

**[ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](multimodal_vlm/ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)**

:   本文提出 ODE（Open-set Dynamic Evaluation）协议，通过图结构建模现实世界物体概念及其分布关联，从中动态提取概念组合并生成合成测试图像，实现了开放集、持续更新的多模态幻觉评估，有效避免了现有静态基准可能存在的数据污染问题。

**[OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)**

:   本文提出 OpenING 基准（5,400 条人工标注实例、56 个真实场景任务）和 IntJudge 评判模型（与人类判断一致率 82.42%），填补了开放式图文交错生成评估的真空，发现当前集成管线（如 Gemini+Flux）大幅领先端到端模型，但所有方法仍远不及人类标注质量。

**[Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy](multimodal_vlm/optimus-2_multimodal_minecraft_agent_with_goal-observation-action_conditioned_po.md)**

:   提出Optimus-2，通过MLLM进行高层规划，结合Goal-Observation-Action Conditioned Policy (GOAP)进行底层控制，其中GOAP使用Action-guided Behavior Encoder建模观察-动作因果关系，并用MLLM对齐行为token与语言指令，在Minecraft原子任务上平均提升27%、长程任务提升10%、开放指令任务提升18%。

**[PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](multimodal_vlm/parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)**

:   提出PARC框架，通过**11种语言/视觉提示变异**、**可靠性评分**和**指标校准**三大支柱，首次系统量化分析了22个VLM在7个数据集上的提示敏感性，发现VLM继承了LLM的语言敏感性并在视觉域呈现对称表现，InternVL2家族对提示变化最鲁棒。

**[Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](multimodal_vlm/period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)**

:   提出Period-LLM——首个具备周期性感知能力的MLLM，采用"从易到难"渐进式训练范式（文本重复→宏观周期视频→微观周期信号），配合"抵抗逻辑遗忘"（RLO）梯度优化策略，在重复动作计数、rPPG心率估计等跨模态周期任务上显著超越现有MLLM。

**[Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](multimodal_vlm/playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)**

:   提出 JOOD 框架，通过将恶意输入进行分布外（OOD）化变换（如图像/文本混合），大幅提升模型不确定性，从而绕过 LLM 和 MLLM 的安全对齐防护，实现高成功率的黑盒越狱攻击。

**[Post-pre-training for Modality Alignment in Vision-Language Foundation Models](multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)**

:   提出 CLIP-Refine，一种介于预训练和微调之间的"后预训练"方法，通过随机特征对齐（RaFA）和混合对比蒸馏（HyCD）两个技术，仅用 1 个 epoch 在小数据集上训练即可缩小 CLIP 的模态间隙并提升零样本性能。

**[Quantization without Tears](multimodal_vlm/quantization_without_tears.md)**

:   提出 QwT（Quantization without Tears）方法，通过在量化网络的每个 block 后添加一个轻量级线性补偿层来弥补量化信息损失，该补偿层参数可通过闭式解在2分钟内求得，在视觉、语言、多模态等多种任务上均显著提升了 PTQ 精度。

**[RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models](multimodal_vlm/rap_retrieval-augmented_personalization_for_multimodal_large_language_models.md)**

:   提出 RAP（Retrieval-Augmented Personalization）框架，通过"记忆-检索-生成"三步实现 MLLM 的个性化：用外部数据库存储用户概念，用多模态检索器动态检索相关概念信息，再注入 MLLM 生成个性化响应，每个概念仅需1张图+描述即可，且支持实时更新。

**[Realistic Test-Time Adaptation of Vision-Language Models](multimodal_vlm/realistic_test-time_adaptation_of_vision-language_models.md)**

:   本文揭示现有VLM测试时适应（TTA）/转导方法在realistic场景下（有效类数可变、非i.i.d.数据流）会严重损害CLIP的零样本鲁棒性，并提出StatA方法，通过在高斯聚类模型参数上引入基于文本编码器知识的KL散度正则化（统计锚），在所有部署场景中保持稳定提升。

**[Reasoning to Attend: Try to Understand How \<SEG\> Token Works](multimodal_vlm/reasoning_to_attend_try_to_understand_how_seg_token_works.md)**

:   深入分析了 \<SEG\> token 在推理分割任务中的工作机制——发现其学到了与文本直接提及相似的语义特征并用于图像-文本语义对齐，在此基础上提出 READ 方法，将 \<SEG\> token 与图像 token 的相似度图转换为点提示，以即插即用方式指导 SAM 解码器生成更精确的分割掩码。

**[Recognition-Synergistic Scene Text Editing](multimodal_vlm/recognition-synergistic_scene_text_editing.md)**

:   提出 RS-STE（Recognition-Synergistic Scene Text Editing）方法，将文字识别与文字编辑统一到一个多模态并行解码器中，利用识别模型隐式解耦风格与内容的天然能力来辅助编辑，并设计循环自监督微调策略使模型能在无配对标注的真实数据上有效训练。

**[Relation-Rich Visual Document Generator for Visual Information Extraction](multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)**

:   提出 RIDGE，一个关系丰富的视觉文档生成器，通过 LLM 生成层次化结构文本内容 + 自监督学习生成内容驱动的布局，合成带有实体类别和链接标注的文档图像，显著提升 VIE 模型在多个基准上的性能。

**[Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](multimodal_vlm/rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)**

:   通过分析 PEFT 在少样本适配中的学习动态，发现训练过程天然分为"任务级特征提取"和"可用类别特化"两个阶段，据此提出 2SFS：先调 LayerNorm 学通用特征，再训练线性分类器提升已知类判别，在 base-to-novel 和 all-to-all 两种设定下均达到或超越 SOTA。

**[Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](multimodal_vlm/rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)**

:   提出 M2F2-Det，首个同时输出深度伪造检测得分和文本解释的多模态人脸伪造检测器，通过 Forgery Prompt Learning 适配 CLIP 学习伪造特征、Bridge Adapter 融合 CLIP 与 deepfake 编码器特征、频域 token 引导 LLM 生成可信解释。

**[ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](multimodal_vlm/revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)**

:   提出 ReVisionLLM，首个能在小时级长视频中进行时序定位的视觉语言模型，模仿人类搜索策略递归处理视频——先粗粒度锁定相关片段，再逐级细化至精确时间边界，在 MAD 数据集上超越 SOTA +2.6% R1@0.1。

**[Revisiting Model Stitching in the Foundation Model Era](multimodal_vlm/revisiting_model_stitching_in_the_foundation_model_era.md)**

:   系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。

**[RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness](multimodal_vlm/rlaif-v_open-source_ai_feedback_leads_to_super_gpt-4v_trustworthiness.md)**

:   RLAIF-V 提出一套完全基于开源MLLM的反馈对齐框架，通过去混淆的候选回复生成策略和分治式反馈标注方法来产生高质量偏好数据，并结合DPO迭代训练与自反馈推理时扩展，使7B模型幻觉率降低80.7%，12B模型仅用自身反馈即超越GPT-4V的可信度。

**[RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)**

:   RoboSpatial 构建了一个包含 1M 图像、5k 3D 扫描和 3M 空间关系标注的大规模机器人空间理解数据集，通过自动化 pipeline 从已有 3D 场景数据中生成三类空间问答对（空间上下文/兼容性/配置），并引入三种参考坐标系（自我/世界/物体），在多个 2D 和 3D VLM 上训练后显著提升空间推理性能，并在真实机器人操作实验中验证了有效性。

**[Seeing the Abstract: Translating the Abstract Language for Vision Language Models](multimodal_vlm/seeing_the_abstract_translating_the_abstract_language_for_vision_language_models.md)**

:   提出 ACT（Abstract-to-Concrete Translator），通过 PCA 分析抽象-具象文本在 VLM 隐空间的表征差异，在推理时无训练地将抽象描述的表征向具象方向偏移，解决 VLM 对抽象语言理解不足的问题，在时尚领域文本-图像检索任务上显著超越微调模型。

**[Self-Evolving Visual Concept Library using Vision-Language Critics](multimodal_vlm/self-evolving_visual_concept_library_using_vision-language_critics.md)**

:   提出 Escher 框架，通过 VLM 作为评判者 + LLM 作为概念生成器的迭代循环，自动进化视觉概念库以提升概念瓶颈模型在图像分类中的表现，在 CUB 数据集上将 LM4CV 从 63.26% 提升至 83.17%（+19.91%）。

**[Self-Supervised Spatial Correspondence Across Modalities](multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)**

:   将对比随机游走（CRW）框架扩展到跨模态像素级对应问题，通过同时学习模态内和模态间的循环一致性特征表示，在无需配对标注的情况下实现 RGB-Depth、RGB-Thermal、Photo-Sketch 等跨模态密集匹配，显著超越现有方法。

**[SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](multimodal_vlm/seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)**

:   提出 Sequential 3D Affordance Reasoning 任务，构建180K指令-点云对基准，通过在3D MLLM中引入 `<SEG>` token 和多粒度语言-点云融合模块，从复杂人类指令中推理并分割出序列化的affordance区域。

**[Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](multimodal_vlm/single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)**

:   提出首个面向少样本计数的单域泛化模型URM，通过将CLIP的通用视觉-语言表征蒸馏到可学习原型中参与相关性构建，在不损失域内性能的前提下大幅提升跨域泛化能力（MAE降低27.5%）。

**[SketchAgent: Language-Driven Sequential Sketch Generation](multimodal_vlm/sketchagent_language-driven_sequential_sketch_generation.md)**

:   SketchAgent 无需任何训练或微调，通过为预训练多模态 LLM 设计网格画布坐标系统 + 上下文示例 + 贝塞尔曲线拟合的后处理流水线，使模型以逐笔画方式生成语义丰富、接近人类风格的草图，Top-1 识别率达人类水平的 85%，并支持交互式协作绘图和对话编辑。

**[Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](multimodal_vlm/skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)**

:   揭示 prompt tuning 冻结 VLM 参数既不促进知识迁移也未显著提升效率（仅减 6% 内存/16% 时间），提出 Skip Tuning 通过层级跳过（LSkip）和类别跳过（CSkip）缩短全微调的梯度传播流，实现 15× 时间效率和 6.4× 内存效率提升的同时精度更优。

**[SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](multimodal_vlm/spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)**

:   SPA-VL 构建了一个包含 100,788 个四元组（问题、图像、优选回答、劣选回答）的大规模VLM安全偏好对齐数据集，覆盖6大领域/13类/53子类有害内容，基于12个VLM的多样化回答和全自动化标注流程，使用DPO/PPO训练后模型在安全性上大幅提升同时保持帮助性。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](multimodal_vlm/spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架中系统替换图像编码器（CLIP/SigLIP/SigLIP2/AIMv2）和引入 2D-RoPE 位置编码，发现 VLM 的空间推理能力主要由编码器的训练目标决定，指望仅靠 2D 位置结构改善空间理解是不够的。

**[StarVector: Generating Scalable Vector Graphics Code from Images and Text](multimodal_vlm/starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)**

:   提出 StarVector，一个基于多模态大语言模型的 SVG 生成框架，将图像矢量化重新定义为逆渲染+代码生成任务，通过视觉语义理解直接生成包含丰富SVG基元（圆形、多边形、文本等）的紧凑SVG代码，在10个数据集3个任务上建立了新的SOTA。

**[BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](multimodal_vlm/stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)**

:   首次揭示SSL视觉编码器对LVLM的后门安全威胁，提出BadVision——通过双层触发器优化和触发器聚焦后门学习机制，仅篡改视觉编码器即可使下游LVLM产生自由文本形式的视觉幻觉（ASR>99%），同时绕过SOTA检测方法。

**[Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](multimodal_vlm/steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_.md)**

:   提出ASTRA，通过**图像归因**定位对抗图像中与越狱最相关的视觉token，构建**转向向量**表征有害响应方向，并在推理时进行**自适应激活转向**将模型远离有害方向，实现了比JailGuard低12%毒性分数、低18% ASR且快9倍的SOTA防御效果。

**[STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](multimodal_vlm/sting-bee_towards_vision-language_model_for_real-world_x-ray_baggage_security_in.md)**

:   构建了首个多模态X射线行李安全数据集**STCray**（46,642张图像-描述对，21类威胁含IED和3D打印枪），设计**STING协议**系统生成领域感知的高质量描述，并训练领域特化VLM **STING-BEE**，在场景理解、威胁定位、视觉定地和VQA四项任务上建立新基线，并展现SOTA跨域泛化能力。

**[Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target](multimodal_vlm/stop_learning_it_all_to_mitigate_visual_hallucination_focus_on_the_hallucination.md)**

:   提出**TL-DPO**（Target-Learning DPO），将传统DPO的全句级偏好学习限制到**幻觉发生的目标chunk**和**对应的图像区域**，通过目标生成损失和目标条件损失排除无关信号，在LLaVA-1.5上将CHAIR_s从66.8降至20.1，同时LLaVA-Bench从63.4提升至71.2。

**[SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation](multimodal_vlm/svlta_benchmarking_vision-language_temporal_alignment_via_synthetic_video_situat.md)**

:   提出**SVLTA**，一个通过合成模拟环境生成的视觉-语言时序对齐基准，包含25.3K动态场景、96种组合动作和77.1K高质量时序标注，具备**可控、组合、无偏**的时序分布，从时序问答、分布偏移敏感性和时序适应三个维度揭示当前VidLLM严重缺乏时序对齐能力（最强GPT-4o在IoU=0.5时R@1仅11.69%）。

**[Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](multimodal_vlm/synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)**

:   用 Stable Diffusion 从类名生成合成图像，通过对比蒸馏 + 图文对齐约束 + 自适应权重固化进行知识蒸馏，仅用每任务 1K 合成图像就超越使用 100K 真实 ImageNet 图像的持续学习方法 ZSCL。

**[Synthetic Visual Genome](multimodal_vlm/synthetic_visual_genome.md)**

:   提出**SVG**（Synthetic Visual Genome）数据引擎，通过GPT-4在已有人工标注基础上**补全缺失关系**（Stage 1）和**Robin自蒸馏+GPT-4编辑**（Stage 2/SG-Edit）两阶段管道，生成146K图像、2.6M物体、5.6M关系的密集场景图数据集，训练的**Robin-3B**模型仅用<3M实例即超越300M实例训练的同尺寸模型，在指代表达理解上达到88.9的SOTA。

**[TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](multimodal_vlm/tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)**

:   首个 VLM 测试时对抗防御方法，通过最小化多视图增强的熵一致性 + 对抗-干净 embedding 统计对齐来学习每个测试样本的防御性 prompt，仅需一步优化即可将 CLIP 对 AutoAttack 的鲁棒性从 0.1% 提升到 48.9%。

**[Taxonomy-Aware Evaluation of Vision-Language Models](multimodal_vlm/taxonomy-aware_evaluation_of_vision-language_models.md)**

:   提出taxonomy-aware VLM评估框架，通过将VLM的自由文本输出映射到分类学树上，利用**层次精度(hP)**和**层次召回(hR)**来量化预测的正确性和具体性，解决了传统精确匹配/文本相似度无法给"部分正确"答案打分的问题。

**[Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](multimodal_vlm/teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)**

:   提出DeQA-Score，通过将质量分数的**高斯分布离散化为soft label**（替代Q-Align的one-hot label），大幅减少离散化信息损失（10-35倍），并引入基于Thurstone模型的**fidelity loss**实现多IQA数据集联合训练，在分数回归任务上全面超越基线。

**[CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

**[Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](multimodal_vlm/thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)**

:   本文提出 VSI-Bench，一个基于视频的视觉空间智能基准（5000+ QA对），系统评估了 MLLM 的空间推理能力，发现空间推理是主要瓶颈，传统语言推理技术（CoT等）无法提升性能，但显式生成认知地图可改善空间距离推理。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   发现现有 VLM（包括 GPT-5.2、Gemini-2.5）在拓扑异常检测上几乎为零（F1@0.5 < 1.5%），提出 Topo-R1 框架通过 SFT + GRPO（含拓扑感知复合 reward，集成 type-aware Hungarian matching + clDice）赋予 VLM 拓扑感知能力，最佳 F1@0.5 达 45.2%。

**[Towards Understanding How Knowledge Evolves in Large Vision-Language Models](multimodal_vlm/towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)**

:   首次系统分析LVLM中多模态知识的演化过程，从**单token概率**、**token概率分布**和**特征编码**三个层次揭示知识演化的"关键层-突变层"双节点模式，将演化过程划分为**快速演化→稳定→突变**三个阶段，并发现深层突变与幻觉现象密切相关。

**[Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](multimodal_vlm/towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)**

:   首个专用于零样本异常检测和推理的 MLLM（Anomaly-OV），通过 Look-Twice Feature Matching 机制生成异常显著性图，配合视觉 Token 选择器聚焦可疑区域，在 9 个基准上实现 88.6% 平均 AUROC 的零样本异常检测 SOTA。

**[UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](multimodal_vlm/unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)**

:   提出UNEM，将广义EM（GEM）算法的每次迭代展开为神经网络的一层，通过端到端学习自动优化**类平衡超参数λ**和**温度缩放T**，在11个细粒度数据集上实现vision-language设置下平均77.8%的准确率（vs. EM-Dirichlet的73.6%），vision-only设置下提升最高达10%。

**[Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly](multimodal_vlm/unveiling_the_ignorance_of_mllms_seeing_clearly_answering_incorrectly.md)**

:   揭示MLLM"**理解了视觉内容但仍给出错误回答**"的普遍现象，构建包含12类正负样本对的**MMVU基准**，发现根因在于训练数据正样本偏倚和视觉token注意力不足，提出**MMVU-Train数据集**（112K正负样本对）+ **内容引导精炼（CGR）**+ **视觉注意力精炼（VAR）**三管齐下的解决方案。

**[UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](multimodal_vlm/upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)**

:   提出UPME框架，通过**无监督同行评审机制**、**视觉-语言评分系统**和**动态权重优化**，仅使用图像数据就能让多个MLLM互相出题评审，在MMStar上与人工评估的Pearson相关性达0.944，有效缓解了MLLM评估对人工标注的依赖和评审偏差问题。

**[VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](multimodal_vlm/vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)**

:   提出VidComposition基准，专门评估MLLM对**编辑合成视频**（影视、动画等）的构图理解能力，涵盖**5大类15个子任务**（镜头运动、叙事结构、角色理解等），对33个MLLM的评测揭示了模型与人类在电影级视频理解上的巨大差距（最佳模型63.3% vs 人类86.3%）。

**[Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](multimodal_vlm/video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)**

:   利用 LLM 内部的 KV 稀疏化能力实现长视频 token 压缩——引入视觉摘要 token（VST）将每段视频的视觉信息压缩到其 KV 中并卸载原始视觉 KV，配合动态压缩和课程学习，在单 A100 上处理 2048 帧，MLVU Dev 上超越 GPT-4o。

**[VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](multimodal_vlm/vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)**

:   提出VILA-M3框架，通过四阶段训练方案将医学领域专家模型（分割/分类）的知识按需集成到通用VLM中，在VQA、报告生成、分类等多个医学基准上以远小于Med-Gemini的模型规模（3B-40B vs 1.5T）实现了平均约9%的SOTA提升。

**[Vision-Language Model IP Protection via Prompt-based Learning](multimodal_vlm/vision-language_model_ip_protection_via_prompt-based_learning.md)**

:   提出IP-CLIP框架，通过轻量级IP-Prompt学习（域token+图像token）和风格增强分支，在冻结CLIP骨干上实现VLM的知识产权保护——让模型在授权域保持高准确率的同时故意降低在非授权域的性能，授权域准确率下降为0%。

**[VisionZip: Longer is Better but Not Necessary in Vision Language Models](multimodal_vlm/visionzip_longer_is_better_but_not_necessary_in_vision_language_models.md)**

:   VisionZip 发现视觉编码器（CLIP/SigLIP）生成的视觉Token存在严重冗余——仅少数Token聚集了绝大部分注意力和信息，基于此提出一种文本无关的Token选择与合并方法，在仅保留10%Token的情况下保持95%的模型性能，并实现8倍预填充加速。

**[Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](multimodal_vlm/visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)**

:   提出视觉语义提示协作网络（VSPCN），通过在预训练ViT中同时学习视觉提示和语义提示，并设计浅层弱融合+深层强融合机制，高效适配ViT提取语义相关的判别性视觉特征，在CUB/SUN/AWA2三个GZSL基准上均达到SOTA。

**[VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](multimodal_vlm/vlsi_verbalized_layers-to-interactions_from_large_to_small_vision_language_model.md)**

:   VLsI 提出了一种基于自然语言的层间蒸馏方法，通过在大小 VLM 的中间层引入 "verbalizer" 将特征映射到语言空间，并采用自适应层匹配策略对齐推理过程，使 2B/7B 小模型在 10 个 VL 基准上平均超过 GPT-4V 达 11.0%/17.4%，无需改变架构或增加参数。

**[What's in the Image? A Deep-Dive into the Vision of Vision Language Models](multimodal_vlm/whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)**

:   本文通过 Attention Knockout 实验系统分析了 VLM（InternVL2-76B 和 LLaVA-1.5-7B）的视觉信息处理机制，揭示了三个关键发现：(1) query text token 充当全局图像描述器压缩高层视觉信息，(2) 中间层（约 25%）主导跨模态信息传递而早晚层贡献极少，(3) 细粒度物体细节通过空间局部化的方式从 image token 中提取。基于这些发现提出了 Image Re-prompting 应用，用仅 5% 的 image token 即可保持 96% 的 VQA 性能。

**[Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](multimodal_vlm/words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)**

:   本文发现VLM存在"盲目信任文本"现象——当视觉与文本输入不一致时，模型系统性地偏向文本（即使文本是错误的），通过构建包含Match/Corruption/Irrelevance三类文本变体的benchmark评估了10个VLM，分析了5个影响因素，并证明SFT+文本增强可有效缓解，同时从理论上解释了该现象源于纯文本与多模态训练数据的不平衡。

**[Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](multimodal_vlm/your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)**

:   发现冻结 LVLM 中天然存在少量"定位头"（localization heads）持续捕捉文本语义对应的物体位置，仅用 3 个注意力头的注意力图即可实现超越微调 LISA-7B 的无训练视觉定位，RefCOCO val 达 86.5%。

---

## 🏥 医学图像 { #medical_imaging }

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

**[AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](medical_imaging/aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)**

:   提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](medical_imaging/accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)**

:   借鉴基础模型范式，在大规模公开脑 MRI 数据上预训练扩散概率模型（DPM），再在仅 20 例中风患者数据上微调，实现数据受限场景下加速 MRI 重建，临床读者研究证实 2× 加速图像质量不劣于标准治疗。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](medical_imaging/adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](medical_imaging/association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)**

:   在两个大规模肺癌筛查队列（NLST 7980 例、SUMMIT 8561 例）中验证了基于深度学习自动量化的 PPFE（胸膜肺实质纤维弹性组织增生）进展与全因死亡率独立相关，提出 PPFE 纵向变化可作为筛查人群中识别高呼吸发病风险个体的影像生物标志物。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](medical_imaging/automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)**

:   使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   BiCLIP 提出了一种双向一致性视觉-语言分割框架，通过双向多模态融合（BMF，让视觉特征反向精炼文本嵌入）和图像增强一致性（IAC，跨弱/强扰动正则化），在 COVID-19 CT 分割上以仅 1% 标注数据即可保持鲁棒性能，且对临床图像退化（噪声/模糊）具有容忍力。

**[Boltzmann Attention Sampling for Image Analysis with Small Objects](medical_imaging/boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)**

:   提出BoltzFormer——一种新型transformer decoder架构，通过玻尔兹曼分布动态采样稀疏注意力区域来聚焦小目标，结合退火温度调度（早期层探索、后期层利用）和PiGMA多query聚合模块，在占图像面积<0.1%的小目标分割上比SOTA提升3-12% Dice分数，同时减少一个数量级的注意力计算。

**[CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)**

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE 框架，将缺失模态分割的鲁棒性问题重新定义为决策层专家一致性控制问题，通过全局模态专家一致性(MEC)和区域专家一致性(REC)双分支约束减少专家漂移，并用轻量门控网络将一致性分数转化为可靠性权重指导特征融合，在 BraTS 2020 和 MSD Prostate 上超越 SOTA。

**[CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections](medical_imaging/crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)**

:   提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。

**[CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](medical_imaging/cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)**

:   提出 CycleULM，首个统一的无标签深度学习超声定位显微(ULM)框架，通过 CycleGAN 学习 CEUS 帧到简化微泡域的物理仿真双向翻译来弥合仿真-真实域差距，实现微泡定位精度提升达40% recall、46% precision，并以18.3 fps 实现实时处理。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为核心的混合 Transformer-CNN-Mamba 架构，通过 Co-Attention Gate、Vision State Space Module 和可变形卷积精炼块增强解码器能力，并引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA 性能，同时保持适中的模型复杂度。

**[Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出 ScleraGluNet，通过五方向巩膜血管图像结合多分支 CNN + MRFO 特征筛选 + Transformer 跨视图融合，实现三分类代谢状态判别（93.8% 准确率）和连续空腹血糖估计（MAE = 6.42 mg/dL），为无创血糖监测提供了新途径。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建了最大规模 PET 分割数据集 PETWB-Seg11K（11,041 例全身 PET + 59,831 个分割掩码），并提出 SegAnyPET——基于 3D 架构 + prompt 工程的 PET 通用分割基础模型，在多中心、多示踪剂、多疾病场景下展现强零样本泛化能力。

**[DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)**

:   提出 DFLMoE 在去中心化联邦学习中使用混合专家（MoE）机制处理医疗数据异质性，无需中心服务器即可在保护隐私的前提下协同训练

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   提出一种结合非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量 CNN 分类和扩散式特征空间去噪的脑肿瘤分类框架，在保持 ~85% 干净准确率的同时，将 AutoAttack 下的鲁棒准确率从 0.47% 提升至 59.5%。

**[DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](medical_imaging/din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)**

:   本文提出DiN框架，首次将扩散模型应用于医学VQA的噪声标签场景（NM-VQA），通过扩散式答案分类器从生成视角进行粗到细的答案筛选，配合噪声标签精炼模块动态修正标签，在10%语义噪声下VQA-RAD准确率达74.24%，超越SNLC的69.65%。

**[Distilled Prompt Learning for Incomplete Multimodal Survival Prediction](medical_imaging/distilled_prompt_learning_for_incomplete_multimodal_survival_prediction.md)**

:   本文提出DisPro (Distilled Prompt Learning)，通过两阶段提示学习——UniPro蒸馏各模态知识分布 + MultiPro利用LLM从可用模态推断缺失模态——同时补偿缺失模态的特异性和共享信息，在5个TCGA生存预测数据集上取得SOTA。

**[Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data](medical_imaging/domain_adaptive_diabetic_retinopathy_grading_with_model_absence_and_flowing_data.md)**

:   本文提出 GUES（Generative Unadversarial Examples）方法，在无法访问源模型参数和标签、目标数据以流式到达的极端在线无模型领域自适应（OMG-DA）场景下，通过 VAE 生成个性化非对抗性扰动并以显著性图作为伪监督，提升冻结源模型在目标域上的糖尿病视网膜病变（DR）分级性能。

**[EchoONE: Segmenting Multiple Echocardiography Planes in One Model](medical_imaging/echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)**

:   本文提出 EchoONE，首次用一个统一模型解决超声心动图多切面分割（MPS）问题，通过先验可组合掩码学习（PC-Mask）模块生成语义感知的稠密 prompt，并设计局部特征融合与适配（LFFA）模块将 CNN 局部特征注入 SAM 解码器，在 6 个切面上持续达到 SOTA 性能。

**[EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance](medical_imaging/echoworld_learning_motion-aware_world_models_for_echocardiography_probe_guidance.md)**

:   本文提出 EchoWorld，一种面向超声心动图探头引导的运动感知世界建模框架：先通过空间世界建模（掩码重建）和运动世界建模（探头运动与视觉变化预测）进行预训练以编码心脏解剖知识，然后在微调阶段引入运动感知注意力机制融合历史视觉-运动序列，在 10 个标准切面的引导任务上显著降低引导误差。

**[Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](medical_imaging/enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)**

:   提出 MLRG 两阶段框架，通过多视角纵向对比学习融合当前多视角图像的空间信息和历史纵向数据的时间信息进行视觉-文本预训练，并用 tokenized absence encoding 灵活处理缺失的患者先验知识，在 MIMIC-CXR 上 BLEU-4 提升 2.3%，MIMIC-ABN 上 F1 提升 5.5%。

**[Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling](medical_imaging/enhancing_virtual_try-on_with_synthetic_pairs_and_error-aware_noise_scheduling.md)**

:   本文提出通过人体图像反向提取合成服装对来增强虚拟试穿训练数据，并设计了基于错误感知噪声调度的Schrödinger Bridge精炼模型（EARSB），对已有试穿模型的生成结果进行局部纠错，在VITON-HD和DressCode上取得了SOTA效果且用户更偏好本文结果（59%）。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出 EquivAnIA，一种基于 cake wavelet 和 ridge filter 的频谱方法，用于对图像进行旋转等变的各向异性分析，在合成和真实图像（含 CT）上展现出优于传统 angular binning 的旋转鲁棒性。

**[Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)**

:   提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 联邦学习框架，通过模态专属编码器（全局联邦）和部分个性化融合解码器，同时解决多模态 MRI 脑肿瘤分割中的模态间异质性和客户端个性化问题，在 BraTS 2018/2020 上客户端平均 mDSC 达 75.70%/75.90%。

**[Few-Shot Personalized Scanpath Prediction](medical_imaging/few-shot_personalized_scanpath_prediction.md)**

:   提出少样本个性化扫视路径预测（FS-PSP）任务 和 Subject-Embedding Network（SE-Net），通过将主体嵌入学习与扫视路径预测解耦，仅需 1-10 张图像的注视数据即可适配新用户，在 OSIE、COCO-FreeView、COCO-Search18 三个数据集上 ScanMatch 指标超越第二名 5.9%-7.9%，且适配时间仅 3.6 秒、无需微调。

**[FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](medical_imaging/ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)**

:   提出 FFaceNeRF，一种基于 NeRF 的面部编辑方法，通过几何适配器（geometry adapter）+ 三平面特征注入 + 潜码混合增强（LMTA），仅需 10 张标注样本即可适配到任意自定义分割 mask 布局，实现灵活的 3D 感知面部编辑。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](medical_imaging/giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)**

:   提出 GIIM，一种基于多异构图（MHG）的多视图医学图像分类框架，同时建模视图内（intra-view）和视图间（inter-view）的病灶依赖关系，在肝脏 CT、乳腺 X 线和乳腺 MRI 三种模态上均显著优于现有多视图方法，并对缺失视图具有鲁棒性。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）理论量化因果覆盖差距，并将人类专家知识与 MedGemma-4B 视觉语言模型结合，实现单源域泛化（SDG），在糖尿病视网膜病变分级（8 个数据集）和癫痫灶检测（2 个数据集）上大幅超越现有方法。

**[Interactive Medical Image Analysis with Concept-based Similarity Reasoning](medical_imaging/interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)**

:   本文提出 CSR（Concept-based Similarity Reasoning）网络，通过学习概念原型在图像局部区域的相似性来进行分类推理，同时支持医生在训练和测试时从空间级和概念级两个维度进行交互式干预，在三个医学数据集上以高达 4.5% 的 F1 提升超越了现有可解释方法。

**[Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](medical_imaging/interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)**

:   本文提出 IMed-361M，一个包含 640 万张图像和 3.61 亿个 mask（平均每张 56 个）的大规模交互式医学图像分割基准数据集，覆盖 14 种成像模态和 204 个分割目标，并基于此开发了支持点击、边框、文本及组合交互的 IMIS 基线网络，在多个场景下超越现有视觉基础模型。

**[Knowledge Bridger: Towards Training-Free Missing Modality Completion](medical_imaging/knowledge_bridger_towards_training-free_missing_modality_completion.md)**

:   本文提出 Knowledge Bridger，一个免训练的缺失模态补全框架，通过利用大型多模态模型（LMM）自动挖掘多模态知识、构建知识图谱来指导缺失模态的生成与排序，在通用场景和医学OOD场景下均超越了现有方法。

**[Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](medical_imaging/latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)**

:   本文提出 Latent Drifting (LD)，通过在扩散模型的前向和反向过程中引入一个标量偏移参数 δ 来弥合预训练自然图像模型与医学图像目标分布之间的差距，显著提升了多种微调方案下的医学图像生成和反事实图像合成效果。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](medical_imaging/mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)**

:   提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

**[MoEdit: On Learning Quantity Perception for Multi-Object Image Editing](medical_imaging/moedit_on_learning_quantity_perception_for_multi-object_image_editing.md)**

:   提出无辅助工具的多物体图像编辑框架 MoEdit，通过 FeCom 模块补偿 CLIP 编码中物体属性的交叉混淆、QTTN 模块注入数量感知到 U-Net，实现编辑前后物体数量一致且属性互不干扰。

**[Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](medical_imaging/multi-modal_vision_pre-training_for_medical_image_analysis.md)**

:   BrainMVP提出首个多模态视觉预训练范式，通过跨模态掩码重建、模态模板蒸馏和模态感知对比学习三个代理任务，在16,022例多参数脑MRI扫描(240万+图像)上预训练ViT，在六个分割和四个分类下游任务上均超越SOTA，Dice Score提升最高达14.47%。

**[Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)**

:   提出 MR-PLIP，首个在多分辨率（5×/10×/20×/40×）下进行病理-语言预训练的视觉语言模型，通过跨分辨率视觉-文本对齐（CVTA）和多分辨率文本引导视觉表示对齐（MRTVA），在 34M 图文对上训练后，在 26 个基准数据集上全面超越 SOTA 基础模型。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D 深度学习模型，融合纵向 MRI 数据与放疗剂量分布图，用于区分胶质母细胞瘤术后放射性对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出 ERBA 适配器，将酶动力学预测建模为"底物识别→构象适应"的分阶段条件化过程，通过 MRCA 注入底物语义、G-MoE 融合活性位点3D几何、ESDA 保持 PLM 先验，在 kcat/Km/Ki 三个动力学端点上一致超越现有方法。

**[MultiMorph: On-demand Atlas Construction](medical_imaging/multimorph_on-demand_atlas_construction.md)**

:   本文提出MultiMorph，一种前馈式脑图谱构建模型，通过线性复杂度的GroupBlock特征共享层和Centrality Layer实现任意数量3D脑图像的单次前向传播即生成无偏群组图谱，速度比传统优化方法快100倍，且无需微调即可泛化到未见模态和人群。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM 框架，在潜在空间中显式解耦风格与结构信息，通过高频注入块 (HFIB)、多模态结构特征融合 (MMSF) 和多尺度结构增强 (MSSE) 提取模态不变的多尺度结构先验来引导扩散过程，解决任意模态缺失下 MRI 翻译的解剖不一致和纹理退化问题。

**[NOIR: Neural Operator Mapping for Implicit Representations](medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)**

:   NOIR 将医学图像计算任务重新建模为连续函数空间之间的算子学习问题，通过隐式神经表示(INR)将离散医学信号嵌入连续函数空间，再用神经算子(NO)学习函数间的映射，实现分辨率无关的分割、形状补全、图像翻译和合成。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   本文将 Singleton 和 Batch Processing 设计模式集成到基于 Python 的 RPA 自动化管道中，结合 EfficientNetV2B1 模型实现口腔癌病灶检测，相比 UiPath/Automation Anywhere 等传统 RPA 平台实现 60-100× 的推理加速。

**[Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era](medical_imaging/nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)**

:   Nyxus 是一个面向大数据和 AI 时代的下一代图像特征提取库，支持 2D/3D 数据的 out-of-core 可扩展提取，覆盖 radiomics 和细胞分析两大领域共 261+ 特征，在速度上比 CellProfiler 快 3–131×、比 PyRadiomics/MITK 快数倍至数百倍。

**[LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments](medical_imaging/paper_title_lov3d_grounding_cognitive_prognosis_reasoning_in_longitudinal_3d_bra.md)**

:   LoV3D 提出一套端到端纵向 3D 脑 MRI 视觉-语言模型管线，通过结构化可验证输出设计实现解剖区域评估 + 纵向对比 + 三分类诊断推理，并利用临床加权 Verifier 驱动 DPO 训练（无需人工标注），在 ADNI 上达到 93.7% 三分类准确率且零非相邻诊断错误。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](medical_imaging/prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)**

:   ProtoSR 提出从大规模自由文本放射学报告中挖掘模板对齐的原型知识库，并通过原型条件化的后期融合残差模块注入结构化报告预测，在 Rad-ReStruct 基准上实现 SOTA，尤其在细粒度属性问题 (L3) 上获得 72.1% 的相对提升。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](medical_imaging/reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)**

:   将深度学习模块（SynthStrip/SynthSeg）模块化替换 SIENA 管线中的经典颅骨剥离和组织分割步骤，在保留管线可解释性的前提下显著提升纵向脑萎缩（PBVC）估计的临床敏感性和鲁棒性。在 ADNI 和 PPMI 两个纵向队列上验证。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

**[SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](medical_imaging/salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)**

:   提出 SALIENT，一个基于小波域扩散的掩码条件生成框架，通过频率感知的可解释优化目标和配对的病灶-掩码体积生成，实现长尾 CT 检测中可控、高效的合成数据增强与精度拯救。首次系统表征增强剂量-反应曲线。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)**

:   提出 SCDL 即插即用模块，通过学习类条件代理分布并进行双向对齐（CDBA）+ 语义锚约束（SAC），在嵌入空间显式重塑类条件特征结构，缓解半监督医学影像分割中的监督偏差和表示不平衡。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 多教师多学生半监督框架，通过 Stricter Weighted-Confidence Constraint 实现多源 CBCT 牙齿分割的跨域泛化。

**[Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](medical_imaging/transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   首次将 SegFormer 用于 HR-pQCT 影像的多区域（骨+软组织）自动分割与放射组学分析，发现肌腱组织特征在骨质疏松分类中优于传统骨指标。

**[UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](medical_imaging/ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)**

:   提出 UltrasoundAgents 层次化多智能体框架，通过主智能体定位病灶+子智能体识别属性+证据链推理的流程，对齐乳腺超声临床诊断工作流并实现可追溯的 BI-RADS 分级与良恶性判断。

**[Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的教师-学生结构结合文本概念提示、置信度感知一致性正则化和双流时序一致性，实现 X 光冠脉造影视频的半监督血管分割。

**[UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结病理基础模型 UNI 的稠密空间 token 作为生成器的直接条件信号，实现 H&E 到 IHC 的虚拟染色，单一统一模型同时服务四种 IHC 标记物并达到 SOTA。

**[Unleashing Video Language Models for Fine-grained HRCT Report Generation](medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

**[Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)**

:   通过从 13 个癌症病理基准数据集中裁剪 20×20 像素的背景区域（不含任何临床诊断信息）训练 ResNet50/DenseNet121/InceptionV3/VGG16 四种 CNN，发现分类准确率远高于随机猜测（最高达 93%），系统性揭示了 CNN 在癌症病理分析中可能依赖数据集采集偏差（如染色协议、扫描仪差异）而非真正的病理特征进行判断。

---

## 🧑 人体理解 { #human_understanding }

**[3D Face Reconstruction From Radar Images](human_understanding/3d_face_reconstruction_from_radar_images.md)**

:   首次从毫米波雷达图像进行3D人脸重建：用物理雷达渲染器生成合成数据集训练CNN编码器估计BFM参数，再通过学习一个可微分雷达渲染器构建model-based autoencoder，在合成数据上实现2.56mm平均点距精度，并可在推理时无监督优化参数。

**[Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](human_understanding/analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)**

:   首次系统研究3D手势估计中合成数据到真实数据的域差距，通过可控数据合成管线分解并分析了前臂、频谱统计、手势分布、物体遮挡四个关键因素的影响，证明合理整合这些因素后纯合成数据可达到与真实数据同等的精度。

**[Anomize: Better Open Vocabulary Video Anomaly Detection](human_understanding/anomize_better_open_vocabulary_video_anomaly_detection.md)**

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](human_understanding/breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   用预训练的Foundation Model（TabPFN）替代传统手工先验，实现零超参数调优的电路Yield Multi-Corner Analysis：冻结backbone做in-context learning，自动跨corner迁移知识，结合自动特征选择（1152D→48D），在SRAM benchmarks上达到SOTA精度（MRE低至0.11%）且验证成本降低10倍以上。

**[ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)**

**[Co-op: Correspondence-Based Novel Object Pose Estimation](human_understanding/co-op_correspondence-based_novel_object_pose_estimation.md)**

**[Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](human_understanding/collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)**

:   提出 Cooperative Tree Search (CoTS) 框架，将修改版蒙特卡洛树搜索与 LLM 驱动的奖励函数结合，引导多个具身智能体进行长期战略规划和高效协作，并通过计划评估模块避免频繁计划更新带来的行为混乱，在 CWAH 和 TDW-MAT 环境上显著超越现有方法。

**[Conformal Prediction for Zero-Shot Models](human_understanding/conformal_prediction_for_zero-shot_models.md)**

:   将保形预测（Conformal Prediction）应用于零样本模型，为 CLIP 等模型的预测提供有理论保证的不确定性量化和校准预测集

**[ControlFace: Harnessing Facial Parametric Control for Face Rigging](human_understanding/controlface_harnessing_facial_parametric_control_for_face_rigging.md)**

:   提出 ControlFace，利用双分支 U-Net（FaceNet + 去噪 U-Net）结合 3DMM 渲染条件，实现无需微调即可灵活编辑人脸姿态、表情和光照，同时精确保留身份和语义细节。

**[CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)**

:   提出 CRISP，一个类别无关的物体姿态与形状估计 pipeline，核心创新在于基于 active shape model 的优化校正器和 correct-and-certify 自训练策略，可在测试时自适应弥合大的域差距。

**[CryptoFace: End-to-End Encrypted Face Recognition](human_understanding/cryptoface_end-to-end_encrypted_face_recognition.md)**

:   提出 CryptoFace，首个端到端全同态加密（FHE）人脸识别系统，通过混合浅层 patch CNN 架构（CryptoFaceNet）大幅降低乘法深度，实现比 SOTA FHE 网络快 7 倍的加密推理，同时提升验证精度。

**[Design2GarmentCode: Turning Design Concepts to Tangible Garments Through Program Synthesis](human_understanding/design2garmentcode_turning_design_concepts_to_tangible_garments_through_program_.md)**

:   提出 Design2GarmentCode，首个神经符号方法将多模态设计输入（文本/图像/草图）转化为参数化服装制版程序（GarmentCode DSL），实现 100% 仿真成功率和 88.67% 的用户满意度，且生成的程序可编辑、可参数化。

**[DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](human_understanding/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)**

:   提出 DualTalk——首个统一建模说话者和倾听者行为的多轮双人交互 3D 说话人头生成框架，配套构建了包含 50 小时、1000+ 身份的双人对话数据集。

**[Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](human_understanding/efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)**

:   本文提出一种基于 3D-VQGAN 的高效盲人脸视频增强框架，通过设计空间-时间双码本记录高质量肖像特征和运动残差信息，配合边际先验正则化缓解码本崩溃问题，在 BFVR 和去闪烁任务上实现了 SOTA 效果且推理速度提升 2-140 倍。

**[EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](human_understanding/egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)**

:   EgoPressure 提出首个第一人称视角的手部触觉压力和姿态数据集，包含 21 名参与者 5 小时的 RGB-D 交互数据、基于多视角优化的高保真 MANO 手部网格标注和压力传感器的真实压力映射，并建立了从 RGB 图像估计手部压力和姿态的基准模型。

**[Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](human_understanding/enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)**

:   提出一种两阶段自训练弱监督框架 ST-WSGE，利用 2D 注视跟随数据集（如 GazeFollow）生成 3D 伪标签来增强野外 3D 注视估计的泛化能力，同时设计了模态无关的 Gaze Transformer（GaT）统一处理图像和视频输入，在 Gaze360、GFIE、MPIIFaceGaze 等数据集上取得 SOTA。

**[FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](human_understanding/fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)**

:   提出 FATE，从单目视频重建可动画化的全头高斯化身，通过基于采样的密化策略（替代阈值分裂）、神经烘焙（将离散高斯转为连续UV纹理图以支持编辑）和通用补全框架（合成后脑外观），实现仅 49K 高斯即达到 28.37dB PSNR 的高效高质量重建。

**[Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](human_understanding/forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)**

:   提出 Forensics Adapter，一个仅 5.7M 参数的轻量适配器网络，与冻结 CLIP 并行学习人脸伪造的融合边界特征，通过掩码边界预测+逐块对比+样本级对比三重目标实现跨数据集的高泛化性人脸伪造检测，CDF-v1 上 AUC 达 0.914。

**[FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](human_understanding/freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)**

:   提出 FreeCloth 混合框架，将人体表面分为"裸露/变形/生成"三类区域，对贴身衣物用 LBS 变形、对宽松服装（裙子、长裙）用无 LBS 约束的自由形态生成器建模，在 ReSynth 数据集上取得 SOTA，尤其在宽松服装场景下大幅超越现有方法。

**[FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](human_understanding/fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)**

:   提出 FRESA，通过学习一个通用着装人体先验模型，从少量图像前馈式（18秒）联合推理个性化 canonical 形状、蒙皮权重和姿态依赖变形，实现零样本泛化到手机照片的高质量可动画化 3D 人体 Avatar 重建。

**[FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones](human_understanding/fsboard_over_3_million_characters_of_asl_fingerspelling_collected_via_smartphone.md)**

:   发布 FSboard——迄今最大的 ASL 指拼（fingerspelling）识别数据集（320万字符、266小时视频、147位聋人签名者用智能手机自拍录制），聚焦手机文字输入场景，基线模型用 MediaPipe + ByT5 达到 11.1% CER，为指拼作为手机输入方式提供了坚实的数据基础。

**[GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](human_understanding/gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)**

:   提出 GaussianIP 两阶段框架，通过自适应人体蒸馏采样（AHDS）从人体中心扩散模型高效生成身份一致的 3D 高斯人体，再通过视角一致性精炼（VCR）机制利用 mutual attention 增强面部和服饰纹理细节，在 40 分钟内完成训练并显著优于现有方法。

**[GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)**

:   GCE-Pose 提出一种"先补全再聚合"的策略，通过语义形状重建（SSR）模块将部分观测补全为完整的几何+语义 3D 表示，再通过全局上下文增强（GCE）特征融合模块将全局信息注入局部关键点特征，在 HouseCat6D 和 NOCS-REAL275 上显著超越现有方法。

**[GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](human_understanding/gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)**

:   GigaHands 是迄今为止最大的双手活动数据集，通过设计"指令-标注"程序化采集策略和 51 相机无标记捕捉系统，收集了 34 小时、56 名被试、417 个物体的双手活动数据，包含 1.83 亿帧 RGB 图像和 84K 条详细文本标注，在文本驱动手部动作生成和动作描述任务上展示了数据规模的价值。

**[Hearing Anywhere in Any Environment](human_understanding/hearing_anywhere_in_any_environment.md)**

:   提出 xRIR，一个可跨房间泛化的声脉冲响应（RIR）预测统一模型，结合全景深度图的几何特征提取器和少量参考 RIR 的声学编码器，配合新构建的 AcousticRooms 数据集（260 个房间、30 万+ RIR），在已见/未见模拟环境和真实环境中均大幅超越基线方法。

**[HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator](human_understanding/heie_mllm-based_hierarchical_explainable_aigc_image_implausibility_evaluator.md)**

:   提出HEIE——基于多模态大语言模型（MLLM）的层次化可解释AIGC图像不合理性评估器，通过CoT驱动的三位一体评估器同时输出热力图、评分和文字解释，并用自适应层次化不合理性映射器实现全局-局部缺陷的精准定位，在RichHF-18K和AbHuman数据集上达到SOTA。

**[HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](human_understanding/hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)**

:   HiPART 提出从稀疏 2D 姿态（17 关节）生成层次化稠密 2D 姿态（48→96 关节）的自回归生成方案，用丰富的骨架上下文替代复杂的时序/视觉编码器来解决遮挡问题，在单帧 3D HPE 上达到 SOTA 且超越多数多帧方法，同时参数量和计算量更小。

**[Homogeneous Dynamics Space for Heterogeneous Humans](human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)**

:   本文提出 HDyS（Homogeneous Dynamics Space），通过聚合来自生物力学和强化学习的异构人体运动数据，训练一个同构潜空间来统一不同运动学和动力学表征，实现了从运动学到动力学的高质量双向映射，并在逆动力学估计、地面反力预测等下游任务上展现了有效性。

**[HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](human_understanding/hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)**

:   本文提出 HOP，一种基于异构拓扑的多模态纠缠方法，通过将音频作为桥梁，利用重编程模块对齐音频-文本语义、利用时空图网络对齐音频-动作节奏，实现更自然连贯的语音伴随手势生成，在 FGD、BC 和多样性指标上达到 SOTA。

**[HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition](human_understanding/hotspot_signed_distance_function_optimization_with_an_asymptotically_sufficient_.md)**

:   提出 HotSpot，基于 screened Poisson 方程与距离的经典关系设计新的 SDF 优化损失，提供了渐近充分条件保证收敛到真正的距离函数（而非仅满足 Eikonal 的伪解），同时自然惩罚多余表面积，在复杂形状上显著优于 SAL/DiGS/StEik。

**[Improve Representation for Imbalanced Regression through Geometric Constraints](human_understanding/improve_representation_for_imbalanced_regression_through_geometric_constraints.md)**

:   本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。

**[L2GTX: From Local to Global Time Series Explanations](human_understanding/l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的时间序列分类全局解释方法，通过聚合 LOMATCE 产生的参数化时间事件原语（PEPs）构建类级全局解释，在六个基准数据集上保持稳定的全局忠实度（R²）。

**[Learning Affine Correspondences by Integrating Geometric Constraints](human_understanding/learning_affine_correspondences_by_integrating_geometric_constraints.md)**

:   提出一种融合稠密匹配与几何约束的仿射对应估计新框架（DenseAffine），采用两阶段解耦训练：先用 Sampson 距离损失训练稠密点匹配器，再冻结匹配器、用仿射 Sampson 距离损失训练局部仿射变换提取器，在 HPatches 匹配和 MegaDepth 位姿估计上均取得 SOTA。

**[Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](human_understanding/learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)**

:   提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。

**[Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](human_understanding/learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)**

:   仅使用约 30 秒的行走 MoCap 数据，通过将行走动作中的可迁移运动模式（浅层网络特征对齐）与运动学方法生成的抓取姿态（主动数据扩充策略）相结合，实现了物理可行、自然流畅的全身人体接近-抓取运动生成，在简单场景下抓取成功率达 99.8%。

**[Less is More: Efficient Model Merging with Binary Task Switch](human_understanding/less_is_more_efficient_model_merging_with_binary_task_switch.md)**

:   通过控制实验发现任务向量具有"脉冲特性"——只有幅度超过阈值的参数对任务有正贡献，据此提出T-Switch方法将任务向量二值化为激活开关、极性开关和缩放旋钮三个组件，仅需1-3%的存储空间即可实现显著优于现有基线的动态模型合并效果。

**[MagicArticulate: Make Your 3D Models Articulation-Ready](human_understanding/magicarticulate_make_your_3d_models_articulation-ready.md)**

:   提出 MagicArticulate 两阶段框架，第一阶段用自回归 Transformer 将骨架生成建模为序列预测任务，第二阶段用函数扩散过程结合体积测地距离先验预测蒙皮权重，搭配 33K+ 大规模 Articulation-XL 数据集，实现静态 3D 模型到可动画化资产的自动转换。

**[ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](human_understanding/maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)**

:   提出 ManipTrans，两阶段残差学习框架将人手动捕数据迁移到灵巧机器手的双手操作：Stage-1 在纯手轨迹上预训练模仿模型（手腕+手指跟踪+平滑奖励），Stage-2 通过残差模块+课程学习加入物体交互约束（物体跟踪+接触力），在 OakInk-V2 上物体旋转误差仅 8.60°、双手成功率 39.5%。

**[MaRI: Material Retrieval Integration across Domains](human_understanding/mari_material_retrieval_integration_across_domains.md)**

:   提出 MaRI 框架，用双 DINOv2 编码器（图像 + 材质）通过对比学习构建共享嵌入空间，结合 Blender 合成数据和 ZeST 生成的真实世界材质数据，实现跨域准确的 PBR 材质检索。

**[MDP: Multidimensional Vision Model Pruning with Latency Constraint](human_understanding/mdp_multidimensional_vision_model_pruning_with_latency_constraint.md)**

:   MDP 提出多维度剪枝范式，将通道、注意力头、Q/K/V、嵌入维度和整个 block 等不同粒度的结构化剪枝统一建模为混合整数非线性规划(MINLP)问题，在严格延迟约束下联合求解全局最优剪枝结构，在高剪枝比下大幅超越已有方法。

**[MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](human_understanding/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)**

:   MG-MotionLLM 提出了一个统一的多粒度动作-语言模型，通过 Motion VQ-VAE + T5 语言模型的架构和精心设计的多粒度协同预训练方案（含 28 种任务），同时支持粗粒度和细粒度的动作理解与生成，在经典任务上达到 SOTA 的同时开启了细粒度动作编辑等新应用。

**[MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](human_understanding/mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co.md)**

:   MM-CondChain 是首个针对视觉基础深层组合推理的 MLLM 基准，通过可验证程序中间表示（VPIR）自动构建多层条件链和链式硬负样本，最强模型仅获 53.33 Path F1，揭示深层组合推理是根本挑战。

**[MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](human_understanding/moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)**

:   提出情绪混合专家（MoEE）模型，为 6 种基础情绪各训练一个专家网络并通过 Soft MoE 门控组合，配合 150 小时专业情绪数据集和多模态情绪条件模块，实现对单一及复合情绪的精确、自然控制。

**[MotionMap: Representing Multimodality in Human Pose Forecasting](human_understanding/motionmap_representing_multimodality_in_human_pose_forecasting.md)**

:   提出MotionMap——用热力图表示运动空间分布的新范式，通过t-SNE降维+codebook实现可变数量模式预测和置信度量化，以最少采样实现最佳模式覆盖。

**[Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](human_understanding/multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)**

:   提出 MulSen-AD，首个融合 RGB 相机、激光扫描仪和红外热成像三种传感器的工业物体异常检测数据集（15 类产品、14 种异常），并设计 MulSen-TripleAD 决策级融合基线方法，实现 96.1% AUROC，证明多传感器融合显著优于单传感器方法。

**[NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction](human_understanding/nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)**

:   NBAvatar 提出 Neural Billboard 原语——将可学习平面几何原语与神经纹理延迟渲染结合，实现手脸交互场景下的照片级真实头部 avatar 渲染，在百万像素分辨率下 LPIPS 比 Gaussian 方法降低 30%。

**[One2Any: One-Reference 6D Pose Estimation for Any Object](human_understanding/one2any_one-reference_6d_pose_estimation_for_any_object.md)**

:   提出 One2Any，仅需单张参考图像即可估计任意新物体的 6D 位姿——用参考物体坐标（ROC，以参考相机帧为基准而非规范坐标）编码参考姿态，通过 VQVAE+U-Net 条件生成密集 ROC 图，再用 Umeyama 算法恢复位姿，在 YCB-Video 上 93.7% ADD-S AUC，推理仅 0.09 秒。

**[Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception](human_understanding/perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)**

:   提出一种面向人机协作的感知调度框架，基于信息增益和计算代价的权衡来选择性激活感知模块（目标检测/姿态估计），在流式感知场景下将计算延迟降低最多 27.52%，同时 MMPose 激活召回提升 72.73%。

**[PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](human_understanding/posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)**

:   提出 PoseBH，通过非参数关键点原型（Sinkhorn-Knopp 在线聚类）和跨类型自监督（CSS）实现人/动物/手部等不同骨骼定义数据集的统一训练，在 APT-36K 动物视频数据集上比 ViTPose++ 提升 11.2 AP，证明跨类型知识迁移的有效性。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

**[Removing Reflections from RAW Photos](human_understanding/removing_reflections_from_raw_photos.md)**

:   提出首个基于 RAW 图像的端到端去反射系统：在 XYZ 色彩空间中模拟逼真的反射（含 Fresnel/双反射/WB/曝光），训练 EfficientNet+BiFPN 基础模型分离透射/反射层，再用高斯金字塔上采样器保留高分辨率细节，利用可选的自拍相机上下文图辅助判断，PSNR 30.62dB。

**[Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](human_understanding/scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)**

:   MONDAY 框架从 YouTube 教学视频自动生成移动端导航数据集——通过 OCR 场景转换检测和 GPT-4o 的 3 步动作识别流程，以人工标注 1/17 的成本（$0.34 vs $5.76/视频）构建了覆盖 iOS/Android 双平台的 313K 标注帧，预训练后 agent 在未见的 Windows Mobile 上提升 18.11%。

**[ShowMak3r: Compositional TV Show Reconstruction](human_understanding/showmak3r_compositional_tv_show_reconstruction.md)**

:   提出 ShowMak3r，首个从网络视频（芭蕾/跑酷/电影片段）中进行组合式 4D 重建的完整流水线：3DGS 舞台重建 + 深度引导的 SMPL 时空定位 + 跨镜头演员匹配 + 隐式 MLP 面部表情精化，舞台 PSNR 19.65，面部 PSNR 24.34（比 ExAvatar +4.17）。

**[ShowUI: One Vision-Language-Action Model for GUI Visual Agent](human_understanding/showui_one_vision-language-action_model_for_gui_visual_agent.md)**

:   ShowUI 基于 Qwen2-VL-2B，通过 UI 连通图引导的视觉 token 选择减少 33% 冗余 token 并加速 1.4 倍，配合交错式视觉-语言-动作流和精选 256K 训练数据，仅 2B 参数即在零样本 ScreenSpot 上达到 75.1% 的 SOTA 精度。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](human_understanding/team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

**[Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](human_understanding/two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)**

:   提出 EED（Efficient Ensemble Defense），从单个基础模型通过不同剪枝策略（NIS/ERM/ASE/BNSF）生成多个子模型并动态集成——在 80% 稀疏度下 CIFAR-10 PGD 鲁棒准确率 55.71%（接近未压缩基线），推理加速 1.86 倍。

**[VI3NR: Variance Informed Initialization for Implicit Neural Representations](human_understanding/vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)**

:   推导了适用于任意激活函数的隐式神经表示（INR）初始化方法 VI3NR，将 Xavier/Kaiming 初始化推广到 Gaussian/Sinc 等非标准激活——通过控制前向和反向传播的方差一致性，用一个自由度 $\sigma_p^2$ 同时满足两个方向的稳定性，显著改善 INR 的收敛速度和重建质量。

---

## ✂️ 语义分割 { #segmentation }

**[2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](segmentation/2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)**

:   提出2DMamba，首个具有高效并行算法的**原生2D选择性状态空间模型**，通过保持2D空间连续性（而非展平为1D序列）来建模WSI中的patch间关系，在10个公共病理数据集上全面超越1D Mamba方法，并在ImageNet分类和ADE20K分割上也有提升。

**[A Distractor-Aware Memory for Visual Object Tracking with SAM2](segmentation/a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)**

:   提出SAM2.1++的干扰物感知记忆模型（DAM），将SAM2的记忆拆分为近期外观记忆（RAM，确保分割精度）和干扰物解析记忆（DRM，确保跟踪鲁棒性），通过内省式更新策略检测干扰物并自动存储锚帧，在7个基准上设立新SOTA。

**[SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](segmentation/assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)**

**[Audio-Visual Instance Segmentation](segmentation/audio-visual_instance_segmentation.md)**

**[G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](segmentation/binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)**

:   提出 G2HFNet，通过多尺度细节增强 (MDE)、双分支几何-粒度互补 (DGC)、深层语义感知 (DSP) 和局部-全局引导融合 (LGF) 四个模块，针对不同层级特征设计差异化优化策略，在三个遥感显著性检测数据集上全面超越 SOTA。

**[Condensing Action Segmentation Datasets via Generative Network Inversion](segmentation/condensing_action_segmentation_datasets_via_generative_network_inversion.md)**

**[Continuous Locomotive Crowd Behavior Generation](segmentation/continuous_locomotive_crowd_behavior_generation.md)**

:   生成连续的人群运动行为，实现轨迹和动作的联合合成，产生自然且多样的群体运动模式

**[COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)**

:   COSMOS 提出了一种跨模态自蒸馏框架，通过文本裁剪策略和交叉注意力模块在学生-教师结构中学习细粒度的跨模态表征，在仅使用 30M 数据预训练的情况下，在零样本检索、分类和语义分割任务上全面超越 CLIP 类基线，甚至超越在数十亿数据上训练的 OpenCLIP。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](segmentation/crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，基于物理引导的稀疏混合专家 (MoE) 架构，构建了包含 200K 图像的训练集和 22 个子基准的评估体系，在 20/22 个跨域语义分割基准上达到 SOTA。

**[DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](segmentation/declip_decoupled_learning_for_open-vocabulary_dense_perception.md)**

:   DeCLIP 发现 CLIP 的自注意力中存在"代理 token"现象导致图像 token 无法聚合空间相关信息，提出将自注意力模块解耦为"内容"和"上下文"特征并分别用 CLIP 自蒸馏和视觉基础模型蒸馏进行优化的框架，在开放词汇目标检测和语义分割上全面超越现有方法。

**[DefMamba: Deformable Visual State Space Model](segmentation/defmamba_deformable_visual_state_space_model.md)**

:   DefMamba 提出了一种基于可变形机制的视觉状态空间模型，通过可变形扫描策略动态调整扫描路径（参考点偏移 + 扫描顺序偏移），克服了现有 Visual Mamba 方法使用固定扫描顺序导致的空间结构信息丢失问题，在 ImageNet 分类、COCO 检测和 ADE20K 分割上达到 SOTA。

**[DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](segmentation/dformerv2_geometry_self-attention_for_rgbd_semantic_segmentation.md)**

:   提出将深度图作为几何先验而非通过神经网络编码，设计几何自注意力（GSA）将深度距离和空间距离融合为衰减因子调制注意力权重，以约一半 FLOPs 匹配或超越双编码器 RGBD 分割方法。

**[DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)**

:   提出 dino.txt，通过冻结 DINOv2 视觉编码器 + 从头训练文本编码器的 LiT 策略，创新性地用 [CLS]+平均池化拼接作为图像表征，结合文本+图像双模态数据平衡，仅用 50K 迭代（CLIP 训练成本的几分之一）即在零样本分类和开放词汇分割上达到 SOTA。

**[DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](segmentation/dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)**

:   DPSeg 提出在开放词汇语义分割中同时利用文本提示和 Stable Diffusion 生成的视觉提示来构建双提示代价体积，通过多尺度视觉代价体积引导解码器和两轮推理的语义精炼策略，在 5 个公开数据集上全面超越现有方法。

**[Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)**

:   提出 Dual-Agent Optimization (DATO) 框架，包含一致性互聚合（CMA）模块学习跨域不变特征以增强表示，以及相关性修正策略（CRS）将 support-query 匹配转移到域不敏感的特征空间，有效提升跨域小样本分割的泛化能力。

**[Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](segmentation/dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)**

:   DDESeg 从音频的本质特性出发，针对混合音频的特征混淆和同物体不同声音的类内变异两大问题，提出动态推导模块从混合信号中衍生独立声源表征并增强判别性，再通过动态消除模块过滤掉画外音等无关音频语义，在 AVS 所有基准上取得 SOTA。

**[EdgeTAM: On-Device Track Anything Model](segmentation/edgetam_on-device_track_anything_model.md)**

:   EdgeTAM 通过详细的延迟分析发现 SAM 2 的瓶颈在记忆注意力而非图像编码器，提出 2D Spatial Perceiver 将帧级记忆从 64×64 维压缩到 ~500 个 token（保留空间结构），配合两阶段知识蒸馏，在 iPhone 15 Pro Max 上实现 16 FPS 的实时 Track Anything。

**[EditAR: Unified Conditional Generation with Autoregressive Models](segmentation/editar_unified_conditional_generation_with_autoregressive_models.md)**

:   提出 EditAR——首个将图像编辑（纹理修改、物体替换/移除、局部编辑）和图像翻译（深度/边缘/分割图到图像）统一在单一自回归框架中的方法，通过在 LlamaGen 基础上引入条件图像 token 前缀和 DINOv2 蒸馏损失，在标准 next-token prediction 范式下即可对多种条件生成任务取得与专用模型竞争的性能。

**[Effective SAM Combination for Open-Vocabulary Semantic Segmentation](segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)**

:   提出 ESC-Net，一种单阶段开放词汇语义分割模型，通过从 CLIP 图像-文本相关性图中生成伪提示（pseudo prompts）并将其嵌入预训练 SAM 解码器 block 中，高效利用 SAM 的类无关分割能力来增强空间聚合，配合 Vision-Language Fusion (VLF) 模块实现精确的掩码预测，在 ADE20K、PASCAL-VOC、PASCAL-Context 上均取得 SOTA 性能。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](segmentation/efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一个高效 RGB-D 多任务场景理解网络，通过改进融合编码器利用冗余特征加速推理，引入归一化聚焦通道层 (NFCL) 和上下文特征交互层 (CFIL) 进行跨维度特征引导，并设计多任务自适应损失函数动态调整任务权重，在 NYUv2/SUN RGB-D/Cityscapes 上达到 SOTA。

**[Exploiting Temporal State Space Sharing for Video Semantic Segmentation](segmentation/exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)**

:   提出 TV3S（Temporal Video State Space Sharing）架构，利用 Mamba 状态空间模型实现跨视频帧的高效时序信息共享，通过独立处理空间 patch 并结合 shifted window 机制实现高度并行化计算，在 VSPW 和 Cityscapes 数据集上以良好的精度-效率平衡超越了现有的 Transformer 和 RNN 方法。

**[Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](segmentation/exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)**

:   ExCEL 提出利用 patch-text 对齐范式（而非传统 image-text 对齐）挖掘 CLIP 的密集知识用于弱监督语义分割，通过文本语义扩充（TSE）和视觉校准（VC）两个模块增强密集对齐能力，在仅需 3.2GB 显存和 6% 训练时间的条件下，在 PASCAL VOC 和 MS COCO 上大幅超越 SOTA。

**[F-LMM: Grounding Frozen Large Multimodal Models](segmentation/f-lmm_grounding_frozen_large_multimodal_models.md)**

:   F-LMM 冻结现成 LMM 的所有参数，仅训练轻量 CNN mask decoder 将 LMM 注意力图中固有的词-像素对应关系翻译为分割 mask，在完全保持对话能力的同时获得 competitive 的视觉定位性能。

**[Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](segmentation/fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)**

:   PartCATSeg 通过将物体级和部件级的图文代价体积解耦聚合、引入组合损失约束部件构成关系、并利用 DINO 特征提供结构引导，在多个开放词汇部件分割基准上将 h-IoU 提升超过 10%。

**[FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity](segmentation/finecaption_compositional_image_captioning_focusing_on_wherever_you_want_at_any_.md)**

:   FineCaption 提出一种支持任意 mask 引用和高分辨率图像输入的视觉语言模型，结合 mask 感知 CLIP 编码器、ConvNeXT 和 SAM 高分辨率编码器，以及新构建的 CompositionCap 数据集，实现了多粒度组合式区域图像描述任务。

**[Foveated Instance Segmentation](segmentation/foveated_instance_segmentation.md)**

:   FSNet 提出一种模拟人眼中央凹视觉机制的实例分割框架，通过可学习的显著性图引导非均匀下采样，在注视目标区域保持高分辨率细节、在外围降低分辨率，实现了在不同预训练分割网络上的即插即用式效率提升。

**[Fractal Calibration for Long-Tailed Object Detection](segmentation/fractal_calibration_for_long-tailed_object_detection.md)**

:   提出 FRACAL（FRActal CALibration），一种无需训练的后处理方法，首次将分形维数引入长尾目标检测的后校准中，通过对称校准频率轴（类别频率）和空间轴（类别位置均匀度），在 LVIS 数据集上将稀有类 mask AP 提升高达 8.6%，并在 COCO、V3Det、OpenImages 上展示泛化性。

**[Frequency Dynamic Convolution for Dense Image Prediction](segmentation/frequency_dynamic_convolution_for_dense_image_prediction.md)**

:   FDConv 从频率域角度重新设计动态卷积，通过傅里叶不相交权重（FDW）在不增加参数的前提下构建频率多样的卷积核，结合核空间调制（KSM）和频带调制（FBM）实现精细的频率自适应，仅增加 3.6M 参数即超越需要 65-90M 额外参数的现有动态卷积方法。

**[Generative Video Propagation](segmentation/generative_video_propagation.md)**

:   提出 GenProp 框架，通过选择性内容编码器（SCE）与 I2V 生成模型的配合，将首帧编辑统一传播到整个视频，在一个模型中同时支持视频编辑、目标去除、目标插入、目标跟踪等多种视频任务。

**[GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](segmentation/glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)**

:   提出GLUS框架，通过"上下文帧（全局推理）+ 查询帧（局部追踪）"的帧划分策略，将全局理解和局部时序一致性统一到单个MLLM中，结合端到端训练的VOS记忆库模块，在MeViS上大幅超越所有MLLM-based方法（J&F 51.3%）。

**[Golden Cudgel Network for Real-Time Semantic Segmentation](segmentation/golden_cudgel_network_for_real-time_semantic_segmentation.md)**

:   提出 GCNet，核心是 Golden Cudgel Block (GCBlock)，训练时自膨胀（多卷积多路径）提升学习能力，推理时自收缩（重参数化为单个 3×3 卷积）加速推理，无需外部教师模型即成为"自蒸馏"方案，在 Cityscapes 上以 77.3% mIoU / 193.3 FPS 超越现有实时分割模型。

**[GroupMamba: Efficient Group-Based Visual State Space Model](segmentation/groupmamba_efficient_group-based_visual_state_space_model.md)**

:   提出 Modulated Group Mamba 层，将输入通道分为四组分别按四个方向执行单向 SSM 扫描，通过 Channel Affinity Modulation（CAM）增强跨组通道交互，配合蒸馏训练目标解决大模型不稳定问题，在 ImageNet-1K 上以 23M 参数达到 83.3% Top-1 精度。

**[HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](segmentation/hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)**

:   HFP-SAM 提出分层频率提示的 SAM 框架，通过频率引导适配器（FGA）注入海洋场景信息、频率感知点选择（FPS）自动生成高质量点提示、全视图 Mamba（FVM）高效解码，在四个海洋动物分割数据集上取得 SOTA。

**[Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning](segmentation/hierarchical_compact_clustering_attention_coca_for_unsupervised_object-centric_l.md)**

:   COCA-Net 提出基于物理紧凑性（compactness）的层级聚类注意力层，通过自底向上的层级合并策略发现物体中心，解决了 Slot Attention 在初始化敏感性、slot 数量预设和背景分割等方面的固有缺陷，在六个无监督物体发现数据集上达到 SOTA。

**[Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](segmentation/holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)**

:   本文提出 Holmes-VAU，构建了包含 70k+ 多粒度标注的视频异常理解基准 HIVAU-70k，并设计异常聚焦时序采样器（ATS）让多模态 VLM 集中关注异常密集区域，在长视频异常检测和推理任务上大幅超越现有方法。

**[Image Quality Assessment: From Human to Machine Preference](segmentation/image_quality_assessment_from_human_to_machine_preference.md)**

:   本文首次提出面向机器视觉的图像质量评估（IQA for MVS），构建了包含 225 万细粒度标注和 3 万参考/失真图像对的 Machine Preference Database (MPD)，实验证明现有 HVS-centric IQA 指标无法准确表征机器偏好，揭示了人类与机器视觉系统间的根本性差异。

**[Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](segmentation/leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)**

:   本文提出了一个利用3D几何先验的旋转对称性检测模型，通过在3D空间中直接预测旋转中心和顶点并投影回2D，结合基于种子点和旋转轴的顶点重建模块，在DENDI数据集上以F1-score 33.2超越了之前基于分割的SOTA方法EquiSym (22.5)。

**[LiVOS: Light Video Object Segmentation with Gated Linear Matching](segmentation/livos_light_video_object_segmentation_with_gated_linear_matching.md)**

:   提出 LiVOS——首个使用门控线性注意力替代 softmax 注意力进行内存匹配的轻量 VOS 网络，将时空注意力矩阵压缩为恒定大小的 2D 状态矩阵，实现任意长视频的恒定内存占用，并在 32G 消费级 GPU 上支持 4096p 推理。

**[MambaOut: Do We Really Need Mamba for Vision?](segmentation/mambaout_do_we_really_need_mamba_for_vision.md)**

:   本文通过概念分析指出 Mamba 的 SSM 机制适用于长序列+自回归任务，而 ImageNet 图像分类两者都不满足，因此构建了去掉 SSM 的 MambaOut（纯 Gated CNN）系列模型，在图像分类上全面超越所有视觉 Mamba 模型，有力证明了 SSM 对视觉分类是不必要的。

**[MambaVision: A Hybrid Mamba-Transformer Vision Backbone](segmentation/mambavision_a_hybrid_mamba-transformer_vision_backbone.md)**

:   NVIDIA 提出 MambaVision——首个系统研究 Mamba 与 Transformer 混合方式的视觉骨干网络，通过重设计的 MambaVision Mixer + 在最后几层加入 self-attention 来弥补 SSM 的全局上下文不足，在 ImageNet-1K 上达到精度-吞吐量的新 Pareto 前沿，同时在检测和分割下游任务中也优于同等规模的竞争模型。

**[MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](segmentation/mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)**

:   本文提出 MammAlps——一个来自瑞士国家公园的多模态多视角野生哺乳动物行为监测数据集（8.5 小时稠密标注，5 个物种，11 种活动 + 19 种动作），以及两个基准任务：多模态物种+层级行为识别（B1）和首个多视角长期事件理解（B2），填补了野生动物视频行为分析在层级行为标注、多模态和多视角方面的空白。

**[Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](segmentation/mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)**

:   揭示了开放词汇分割中 mask pooling 方法的性能上界瓶颈——精确 mask 往往无法获得准确分类，提出 Mask-Adapter 从 proposal mask 和 CLIP 特征中提取语义激活图来替代直接 mask pooling，以即插即用方式显著提升多种 OVS 方法的分类准确率。

**[MaSS13K: A Matting-level Semantic Segmentation Benchmark](segmentation/mass13k_a_matting-level_semantic_segmentation_benchmark.md)**

:   构建了包含 13,348 张 4K 分辨率图像的 matting 级语义分割数据集 MaSS13K（掩码复杂度比现有数据集高 20-50 倍），并提出 MaSSFormer 模型通过双分支像素解码器（全局语义 + 局部结构）在保持计算效率的同时实现了高分辨率场景下精细边界的高质量分割。

**[MatAnyone: Stable Video Matting with Consistent Memory Propagation](segmentation/matanyone_stable_video_matting_with_consistent_memory_propagation.md)**

:   提出 MatAnyone 框架，通过区域自适应记忆融合机制在记忆空间中实现一致性传播（核心区域保持语义稳定，边界区域捕获精细 alpha 细节），配合新数据集 VM800 和利用分割数据直接监督 matting head 的训练策略，实现了鲁棒且高质量的目标指定视频抠图。

**[A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](segmentation/multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)**

:   本文提出渐进式两阶段预训练策略——第一阶段用patch级对比学习对齐RGB和深度模态的跨模态表示，第二阶段用掩码自编码+受扩散模型启发的去噪+特征蒸馏联合训练，在ScanNet语义分割上比Mask3D提升+1.3% mIoU，在多个RGB-D下游任务上达到SOTA。

**[PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](segmentation/picosam3_real-time_in-sensor_region-of-interest_segmentation.md)**

:   PicoSAM3 是一个 1.3M 参数的超轻量可提示分割模型，通过 ROI 隐式提示编码、密集 CNN 架构（无 Transformer）、SAM3 知识蒸馏和 INT8 量化，在 COCO 上达 65.45% mIoU，并实现在 Sony IMX500 视觉传感器上 11.82ms 实时推理。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](segmentation/prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)**

:   SAM FTI-FDet 提出基于轻量 SAM 的自动提示实例分割框架，通过 Transformer 解码器式的提示生成器自动产生任务相关提示、自适应特征分发器融合多尺度特征、TinyViT backbone 降低计算开销，在货运列车故障检测数据集上达 74.6 $AP^{box}$ / 74.2 $AP^{mask}$。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](segmentation/rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   RDNet 针对遥感图像中目标尺度剧烈变化的问题，提出区域比例感知的动态自适应显著性检测网络，通过动态自适应细节感知模块（DAD，根据目标区域比例选择不同大小卷积核组合）、频率匹配上下文增强模块（FCE，小波域特征交互）和区域比例感知定位模块（RPL，交叉注意力+比例引导），在 EORSSD/ORSSD/ORSI-4199 三个数据集上取得 SOTA。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出区域引导选择性优化网络 RSONet，通过两阶段（区域引导+显著性生成）解决 RGB 与热红外图像中显著区域不一致问题，利用相似度分数自动选择信息更准确的模态主导后续融合。

**[SAP: Segment Any 4K Panorama](segmentation/sap_segment_any_4k_panorama.md)**

:   将 360° 全景图分割重新定义为透视视频分割问题，通过沿 zigzag 轨迹分解全景图为重叠 patch 序列并微调 SAM2 的 memory 模块，配合 183K 合成 4K 全景图的大规模训练，实现零样本全景分割 +17.2 mIoU 的提升。

**[SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](segmentation/segagent_exploring_pixel_understanding_capabilities_in_mllms_by_imitating_human_.md)**

:   SegAgent 将 referring expression segmentation 建模为人类标注员的迭代操作过程——MLLM 观察当前 mask 状态后预测下一个点击位置，交互式分割模型据此更新 mask，经过多轮迭代得到最终分割结果；通过 StaR+ 策略改进和 PRM+树搜索，在复杂场景下大幅提升分割精度。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出SGMA框架，通过语义引导融合(SGF)模块构建全局语义原型估计模态鲁棒性并自适应加权融合，以及模态感知采样(MAS)模块动态优先训练脆弱模态，解决遥感不完整多模态分割中的模态不平衡、类内变化和跨模态异质性三大挑战。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](segmentation/sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出SPARROW框架，通过目标特定跟踪特征(TSF)和双提示(BOX+SEG)机制，解决视频MLLM中时序引用一致性差和首帧初始化不稳定的问题，在6个基准上对3个主流视频MLLM均取得一致提升。

**[Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在预训练视觉语言模型中引入轻量级表达感知的混合专家（MoE）精细化，分别在 backbone 层（SERA-Adapter）和融合层（SERA-Fusion）进行专家路由，仅更新 <1% 参数即在参考图像分割基准上达到 SOTA。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](autonomous_driving/3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)**

:   提出3D-AVS，首个针对LiDAR点云的**自动词表分割**方法：无需用户指定目标类别，系统自动从图像和点云中识别场景中存在的语义实体并生成词表，再用开放词表分割器完成逐点语义分割，在nuScenes和ScanNet200上展示了生成精细语义类别的能力。

**[ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](autonomous_driving/3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)**

:   提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

**[A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](autonomous_driving/a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)**

:   本文提出首个将 ASP 符号推理决策以可学习嵌入形式直接嵌入端到端规划器轨迹解码的神经-符号框架，用 LLM 动态提取场景规则、Clingo 求解器进行逻辑仲裁、可微 KBM 生成物理可行轨迹并配合神经残差修正，在 nuScenes 上 L₂ 误差 0.57m、碰撞率 0.075%、TPC 0.47m 全面超越 MomAD。

**[PAP: A Prediction-as-Perception Framework for 3D Object Detection](autonomous_driving/a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   PAP 受人脑"预测性感知"启发，将上一帧轨迹预测结果作为当前帧感知模块的 query 输入替代部分随机 query，在 UniAD 上实现 AMOTA 提升 10%（0.359→0.395）、推理速度提升 15%（14→16 FPS）和训练时间缩短 14%。

**[CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](autonomous_driving/cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)**

:   CAWM-Mamba 首次提出端到端统一处理红外-可见光图像融合与复合恶劣天气（如雾+雨、雨+雪）场景的框架，通过天气感知预处理、跨模态特征交互和小波域频率-SSM 解耦多频退化，在 AWMM-100K 和标准融合数据集上全面超越 SOTA。

**[Certified Human Trajectory Prediction](autonomous_driving/certified_human_trajectory_prediction.md)**

:   首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。

**[ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate](autonomous_driving/climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_coordinate.md)**

:   提出首个攀岩运动多模态数据集 AscendMotion（412K 帧 RGB+LiDAR+IMU，22 名专业攀岩者，12 面攀岩墙），以及 ClimbingCap 方法通过分离坐标解码、三重后处理优化和半监督训练实现世界坐标系下的 3D 攀岩动作恢复，MPJPE 达 75.45mm。

**[Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](autonomous_driving/closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)**

**[CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](autonomous_driving/composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)**

:   CompoSIA 提出一种基于 Flow Matching DiT 的组合式驾驶视频生成框架，通过解耦结构（3D bbox）、身份（单参考图像）和自车动作（相机轨迹）三类控制信号的注入方式，实现精细独立控制和组合编辑，用于系统化合成对抗性驾驶场景，FVD 提升 17%，碰撞率增加 173%。

**[Cubify Anything: Scaling Indoor 3D Object Detection](autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)**

**[DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](autonomous_driving/decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)**

:   将 3DGS 场景中的物体与背景解耦，使物体支持物理仿真（碰撞、抓取等），同时保持场景的高质量渲染

**[DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)**

:   本文提出DiffusionDrive，通过截断扩散策略（将去噪步骤从20步减少到2步）和级联扩散解码器，首次将扩散模型成功应用于端到端自动驾驶的实时多模态轨迹规划，在NAVSIM数据集上以88.1 PDMS刷新记录，同时保持45 FPS的实时速度。

**[Distilling Monocular Foundation Model for Fine-grained Depth Completion](autonomous_driving/distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)**

:   本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。

**[Distilling Multi-modal Large Language Models for Autonomous Driving](autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)**

:   本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器，在nuScenes上实现L2轨迹误差降低37%、碰撞率降低80%。

**[Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](autonomous_driving/driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)**

:   本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。

**[DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](autonomous_driving/drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)**

:   构建基于 4D 占用网格的高保真闭环驾驶仿真框架——用 OccDreamer 从 BEV 生成静态场景占用、用 Actor Bank 组合动态物体、用 VideoDreamer 从占用条件生成多视角视频，FVD 降低 44%，物体检测 mAP 提升 33%。

**[EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](autonomous_driving/ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)**

:   首次将事件相机引入3D目标检测，提出 Virtual 3D Event Fusion（V3D-EF）将异步事件投影到3D体素空间与LiDAR特征融合，在帧间"盲区时间"内以100FPS持续检测物体，填补了传感器帧间~100ms的感知空白。

**[EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](autonomous_driving/evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)**

:   提出 EVolSplat，一个基于稀疏3D卷积的前馈城市场景3D高斯泼溅方法，通过全局统一体素预测高斯参数（而非像素对齐），结合遮挡感知的基于图像的渲染（IBR）着色，在 KITTI-360 上达到 23.26dB PSNR / 83.81 FPS。

**[Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](autonomous_driving/exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)**

:   提出 AIScene 框架利用场景内一致性（点擦除策略）和场景间关联（MixPatch + InsFill 跨场景增强），在仅 1% 标注的 SemanticKITTI 上将半监督 LiDAR 分割提升 1.9 mIoU。

**[ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images](autonomous_driving/forestlpr_lidar_place_recognition_in_forests_attentioning_multiple_bev_density_i.md)**

:   本文提出ForestLPR，通过将点云在不同高度切片生成多张BEV密度图，利用ViT提取局部特征后经multi-BEV交互模块自适应关注不同高度的判别性特征，实现森林环境下鲁棒的LiDAR位置识别，在多个数据集上大幅超越SOTA。

**[FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](autonomous_driving/freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)**

:   本文提出FreeSim，通过将挑战性的偏离轨迹新视角生成问题重新表述为生成式图像增强问题，配合piece-wise高斯重建的训练数据构造和渐进式视角扩展策略，首次实现了驾驶场景中超过3米横向偏移的高质量自由视角渲染。

**[g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](autonomous_driving/g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)**

:   本文提出g3D-LF，通过在约5K室内3D场景和近100万语言描述上进行多级对比学习预训练，构建了可泛化到未知环境的3D-语言特征场，在VLN（单目/全景）、零样本物体导航和情境问答四种具身任务上均取得SOTA或接近SOTA表现。

**[GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](autonomous_driving/gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)**

:   本文提出GaussianFormer-2，从概率视角重新诠释3D语义高斯：每个高斯表示其邻域被占用的概率分布，通过概率乘法聚合几何预测、高斯混合模型归一化语义预测，彻底消除了高斯描述空区域和相互冗余重叠的问题，以仅8.9%的高斯数量达到SOTA。

**[GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](autonomous_driving/gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)**

:   提出 GaussianWorld，将 3D 占用预测重新定义为以当前传感器输入为条件的 4D 占用预测问题，通过将场景演化分解为自车运动对齐、动态物体运动和新区域补全三个因素，在 3D 高斯空间中用世界模型显式建模场景变化，在 nuScenes 上不增加额外计算量的前提下将单帧方法的 mIoU 提升超过 2%。

**[Generating Multimodal Driving Scenes via Next-Scene Prediction](autonomous_driving/generating_multimodal_driving_scenes_via_next-scene_prediction.md)**

:   提出 UMGen，一个统一的多模态驾驶场景生成框架，将自车动作、地图、交通参与者和图像四种模态进行 token 化，通过帧间时序自回归（TAR）和帧内有序自回归（OAR）两阶段策略逐场景生成，同时引入动作感知地图对齐（AMA）模块保持自车运动与地图的一致性，可自主生成长达 60 秒的连贯驾驶序列。

**[Generative Gaussian Splatting for Unbounded 3D City Generation](autonomous_driving/generative_gaussian_splatting_for_unbounded_3d_city_generation.md)**

:   提出 GaussianCity，首个将 3D 高斯溅射应用于无界 3D 城市生成的框架，通过引入 BEV-Point 紧凑中间表示使显存占用与场景规模解耦（保持恒定），并设计 Point Serializer 将无序 BEV 点转为有序序列以捕获结构和上下文特征，在无人机视角和街景视角的城市生成中达到 SOTA，渲染速度比 CityDreamer（基于 NeRF）快 60 倍。

**[GLane3D: Detecting Lanes with Graph of 3D Keypoints](autonomous_driving/glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)**

:   提出GLane3D，一种基于关键点的3D车道线检测方法，通过检测车道关键点并预测它们之间的有向连接构建图结构，利用PointNMS去除冗余关键点提议后用Dijkstra最短路径提取车道实例，在OpenLane和Apollo数据集上达到SOTA的F1分数且泛化能力优越。

**[Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](autonomous_driving/helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)**

:   提出Helvipad——首个用于全景立体深度估计的真实世界数据集（40K帧、上下双360°相机+LiDAR），并引入极角输入和环形填充两个适配策略来改进立体匹配模型处理等距矩形投影图像，所提360-IGEV-Stereo在所有指标上达到最佳。

**[InteractionMap: Improving Online Vectorized HDMap Construction with Interaction](autonomous_driving/interactionmap_improving_online_vectorized_hdmap_construction_with_interaction.md)**

:   本文提出InteractionMap，通过点级和实例级关系嵌入、关键帧分层时序融合和几何感知分类-定位对齐三个模块，全面增强在线矢量化HD地图构建中的信息交互，在nuScenes (mAP 71.8) 和Argoverse2 (mAP 74.7) 上均取得SOTA。

**[Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels](autonomous_driving/learning_to_detect_objects_from_multi-agent_lidar_scans_without_manual_labels.md)**

:   提出 DOtA（Detect Objects from Multi-Agent），一种无需人工标注的多智能体 LiDAR 3D 目标检测方法：利用协作智能体内部共享的自车位姿和车身形状完成检测器初始化，再通过智能体间互补观测进行多尺度编码，解码出高低质量伪标签分别指导特征学习，实现完全无监督的高质量 3D 目标检测。

**[LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation](autonomous_driving/lidar-rt_gaussian-based_ray_tracing_for_dynamic_lidar_re-simulation.md)**

:   本文提出LiDAR-RT，将3D高斯原语与NVIDIA OptiX硬件加速光线追踪相结合，首次实现动态驾驶场景下实时且物理精确的LiDAR重新仿真，渲染速度达30 FPS，训练仅需2小时，远超NeRF方案的0.2 FPS和15小时。

**[LightLoc: Learning Outdoor LiDAR Localization at Light Speed](autonomous_driving/lightloc_learning_outdoor_lidar_localization_at_light_speed.md)**

:   本文提出LightLoc，通过样本分类引导 (SCG) 减少视觉相似区域的回归歧义，以及冗余样本下采样 (RSD) 剔除已学好的帧，实现大规模室外LiDAR定位训练50倍加速（1小时 vs 2天），同时达到0.83m SOTA位置精度。

**[LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes](autonomous_driving/limoe_mixture_of_lidar_representation_learners_from_automotive_scenes.md)**

:   提出 LiMoE，通过混合专家（MoE）机制融合三种互补的 LiDAR 表示（距离图/稀疏体素/原始点云），三阶段训练（图像→LiDAR 预训练 → 对比混合学习 → 语义混合监督），在 nuScenes 分割上达到 51.4% mIoU，跨域泛化到 7 个数据集。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出基于 LiDAR 反射率引导的显著高斯泼溅方法，引入结构感知的显著高斯表示（由 LiDAR 几何和反射率特征点初始化）和光照不变的反射率通道作为额外约束，在 Waymo 数据集挑战场景（复杂光照）上 PSNR 超越 OmniRe 1.18 dB。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](autonomous_driving/m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对多相机输入不完整时的语义占用预测问题，提出多视角掩码重建（MMR）模块利用相邻相机重叠区域恢复缺失视角特征，以及特征记忆模块（FMM）通过类级语义原型精炼不确定体素特征，在缺失后视角设置下 IoU 提升 4.93%。

**[MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)**

:   MapGCLR 提出地理空间对比学习方法，通过强制多次行驶中地理空间重叠区域的 BEV 特征一致性来改善在线矢量化 HD 地图构建的 BEV 编码器，在仅 5% 标注数据下实现 42% 的相对 mAP 提升。

**[Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](autonomous_driving/modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)**

:   提出 Human Annotator Modeling (HAM) 方法，通过对人类标注描述进行风格特征提取和聚类，用可学习提示让 MLLM 模拟数千种人类标注风格，再结合 Uniform Prototype Sampling (UPS) 进一步增加风格多样性，自动构建大规模高质量文本-图像行人 ReID 数据集，在多个基准上大幅提升了 ReID 模型的泛化能力。

**[ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](autonomous_driving/modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)**

:   提出 ModeSeq——一种将轨迹模式建模为序列的全新范式，通过逐步解码多模态轨迹（而非一次性并行解码）来显式捕捉模式间关联，并配合 Early-Match-Take-All (EMTA) 训练策略，在不依赖密集模式预测或启发式后处理的前提下，显著提升了稀疏多模态运动预测的轨迹多样性和置信度校准。

**[Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](autonomous_driving/multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)**

:   本文提出首个用于行人轨迹预测的多模态知识蒸馏框架——用轨迹+人体姿态+文本描述训练全模态教师模型，将其知识蒸馏到仅用轨迹或轨迹+姿态的学生模型，在JRDB/SIT/ETH-UCY三个数据集上最高提升约13%预测精度。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出纯视觉端到端的全向开放词汇占用预测框架，通过极坐标螺旋 Mamba（PsM）建模全向空间连续性、占用代价聚合（OCA）统一几何和语义监督、以及无梯度自然模态对齐（NMA）桥接像素-体素-文本模态间隙，在 QuadOcc 和 Human360Occ 上达到 SOTA。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)**

:   首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

**[Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](autonomous_driving/point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)**

:   发现半监督人群计数中点到点（P2P）匹配导致模型对未标注数据过度激活（通过 PSAM 梯度诊断可视化），提出点到区域（P2R）匹配——将每个 GT/伪标签点扩展为局部区域并传播置信度，在 ShanghaiTech-A 5% 标注下 MAE 69.9（前 SOTA 83.7），且比 P2P 快 68 倍。

**[GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](autonomous_driving/rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)**

:   提出 GDFusion，将 RNN 重新解释为梯度下降步骤，统一三种时序线索（场景级/运动/几何）的融合方式，在 Occ3D 上比非时序基线提升 1.4-4.8% mIoU 同时减少 27-72% 推理内存，比 SOLOFusion 等多帧方法更高效。

**[Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](autonomous_driving/scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)**

:   提出 Scenario Dreamer，将自动驾驶仿真环境生成分解为三部分：向量化潜扩散模型生成初始场景（车道+智能体）、回报条件的 CtRL-Sim 生成闭环行为、场景修补实现无界环境扩展，在 nuPlan 上 Frechet Distance 0.67（基线 SLEDGE 1.44），生成仅需 0.16 秒。

**[SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](autonomous_driving/scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)**

:   提出 SceneDiffuser++，一个端到端的城市级交通仿真扩散模型，通过软裁剪（soft clipping）处理稀疏张量中的智能体出入场问题，实现 60 秒以上的行程级（trip-level）交通仿真，在 WOMD-XLMap 上达到 0.2423 综合 JS 散度。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](autonomous_driving/single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)**

:   利用 microLED-on-CMOS 超快数字光投影器实现基于单像素成像（SPI）的 MNIST 图像分类，在 1.2 kfps 帧率下达到 >90% 分类精度，完全绕过图像重建直接从时序光信号分类。

**[SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](autonomous_driving/solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)**

:   提出 SOLVE，通过共享 SQ-Former 视觉编码器实现 VLM 和端到端驾驶模型的特征级协同，用 Trajectory Chain-of-Thought（T-CoT）将 VLM 的长程轨迹作为 E2E 模型的初始化先验，在 nuScenes 上达到 0.28m 平均 L2 误差 SOTA。

**[Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)**

:   SG-NLF 提出一种无需精确位姿的 LiDAR NeRF 框架，通过混合频谱-几何表征重建平滑几何、置信度感知位姿图实现全局对齐、对抗学习增强跨帧一致性，在低频 LiDAR 场景下重建质量和位姿精度分别超越 SOTA 35.8% 和 68.8%。

**[UniScene: Unified Occupancy-centric Driving Scene Generation](autonomous_driving/uniscene_unified_occupancy-centric_driving_scene_generation.md)**

:   提出 UniScene，以占用网格为统一中间表示的两阶段驾驶场景生成：Occupancy Diffusion Transformer 从 BEV 布局生成语义占用，再通过高斯泼溅联合渲染语义+深度图条件化双扩散模型生成视频和 LiDAR，FVD 71.94（前 SOTA Drive-WM 122.70），下游数据增强提升 3D 检测 mAP 3.62%。

**[VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](autonomous_driving/vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   VIRD 通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视角不变表征，实现无需方向先验的全向跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。

---

## 📦 模型压缩 { #model_compression }

**[Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)**

:   提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。

**[Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks](model_compression/alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)**

:   提出基于交替梯度流(AGF)的统一效用度量，将特征空间总变差作为结构化剪枝指标，并结合置信度级联路由实现离线拓扑构建与在线动态推理的解耦，在ImageNet-1K极端压缩下避免传统指标导致的结构崩溃，在ImageNet-100动态推理中以0.92x计算代价匹配全模型精度。

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](model_compression/an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移向量(DV)搜索模块的FPGA架构实现，采用四级流水线设计和优化的存储组织方式，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐量和277 mW功耗，为IPC实际硬件部署和ASIC转化奠定基础。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](model_compression/arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出ARCHE端到端学习型图像压缩框架，在统一概率架构中整合分层Hyperprior、掩码空间自回归上下文、通道条件化和SE激励通道重校准，无需Transformer或循环组件，在Kodak上相对Ballé基线BD-Rate降低约48%，相对VVC Intra降低约5.6%，仅95M参数和222ms解码时间。

**[BHViT: Binarized Hybrid Vision Transformer](model_compression/bhvit_binarized_hybrid_vision_transformer.md)**

:   针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。

**[Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment](model_compression/charm_the_missing_piece_in_vit_fine-tuning_for_image_aesthetic_assessment.md)**

**[CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)**

:   提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。

**[CoA: Towards Real Image Dehazing via Compression-and-Adaptation](model_compression/coa_towards_real_image_dehazing_via_compression-and-adaptation.md)**

:   提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率

**[Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](model_compression/curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)**

:   提出CCFS方法，通过课程学习框架渐进式地从原始数据集中选择合适的真实样本补充蒸馏数据，解决高IPC场景下蒸馏数据与真实数据的不兼容问题，在CIFAR-10/100和Tiny-ImageNet上大幅超越SOTA（最高+6.6%）。

**[Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](model_compression/dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)**

:   提出NCFM方法，通过在复平面上用神经网络参数化的特征函数差异（NCFD）作为分布距离度量，将数据集蒸馏重构为minmax对抗优化问题，同时对齐相位（真实性）和幅值（多样性）信息，在ImageNet子集上最高提升20.5%，且GPU内存降低300倍以上。

**[DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](model_compression/delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)**

:   提出EarlyLate训练策略，通过让不同IPC子批次从不同优化起点开始、经历不同迭代次数来生成难度各异的合成图像，在batch-to-global匹配框架下显著提升类内多样性，同时减少39.3%计算时间，在ImageNet-1K上以IPC=50达到66.1%（ResNet-101，超越RDED 4.9%）。

**[DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models](model_compression/ders_towards_extremely_efficient_upcycled_mixture-of-experts_models.md)**

:   提出DeRS（Decompose-Replace-Synthesis）范式，利用upcycled MoE专家间的极高相似性（余弦相似度>0.999），将N个专家分解为1个共享基础权重+N个轻量delta权重，通过稀疏化/量化/低秩表示压缩delta权重，在MoE层参数减少65%的同时性能不降，或训练时额外参数减少2270倍。

**[Distilling Long-tailed Datasets](model_compression/distilling_long-tailed_datasets.md)**

:   首次系统研究长尾数据集蒸馏问题，发现现有方法在长尾场景下严重退化（甚至不如随机选择），提出Distribution-agnostic Matching（DAM）和Expert Decoupling（ED）两个策略，在CIFAR-10/100-LT和Tiny-ImageNet-LT上大幅超越现有方法（如在imbalance factor=100时超越DATM 19.7%）。

**[DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](model_compression/dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)**

:   提出DyCoke，一种免训练的动态视觉Token压缩方法，通过两阶段策略——时序Token合并（消除跨帧冗余50-60%）和KV Cache动态剪枝（在每个解码步动态保留最相关的token，进一步减少70-90%），将视频LLM的每帧平均token数降至15个，实现1.5倍加速且性能不降反微升。

**[ECVC: Exploiting Non-Local Correlations in Multiple Frames for Contextual Video Compression](model_compression/ecvc_exploiting_non-local_correlations_in_multiple_frames_for_contextual_video_c.md)**

:   提出ECVC视频压缩模型，通过多帧非局部上下文挖掘（MNLC）和多头线性交叉注意力（MHLCA）捕获多参考帧间的非局部相关性，结合部分级联微调策略（PCFS）解决训练-测试序列长度不匹配问题，在IP=32和IP=-1设置下分别比DCVC-FM节省10.5%和11.5%码率。

**[EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](model_compression/efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)**

:   提出EfficientViM，通过将SSD层中的通道混合操作从token空间（$O(LD^2)$）迁移到压缩的隐藏状态空间（$O(ND^2)$，$N \ll L$），实现了比现有Vision Mamba模型快2-4倍的推理速度，同时保持竞争性精度（ImageNet-1K上M3模型77.9%/11952 img/s）。

**[Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning](model_compression/embracing_collaboration_over_competition_condensing_multiple_prompts_for_visual_.md)**

:   提出 Condenser 将多个 Visual ICL 的 prompt 候选通过 Patch-wise 跨注意力凝聚为单一 prompt，实现多 prompt 协作而非竞争选择，在分割/检测/上色等任务上以 16 个 prompt 输入达到 46.63 mIoU（vs 单 prompt 44.14），推理速度比逐一评估快 15×。

**[Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](model_compression/emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)**

:   提出EDF方法，通过Common Pattern Dropout（丢弃轨迹匹配中低损失的通用模式参数梯度）和Discriminative Area Enhancement（用Grad-CAM加权放大判别性区域的梯度），解决数据集蒸馏在复杂场景（ImageNet子集）上的性能退化问题，在ImageMeow/ImageYellow等数据集上仅用23%数据实现无损压缩。

**[Enhancing Dataset Distillation via Non-Critical Region Refinement](model_compression/enhancing_dataset_distillation_via_non-critical_region_refinement.md)**

:   提出NRR-DD三阶段框架：用CAM选低置信度patch初始化合成图像、固定关键区域仅优化非关键区域提升信息密度、用2个距离值替代1000维软标签实现500倍存储压缩。在ImageNet-1K上IPC=10时达到46.1%（超RDED 25.7%），软标签存储从120GB降至0.2GB。

**[Exploration-Driven Generative Interactive Environments](model_compression/exploration-driven_generative_interactive_environments.md)**

:   开源实现 Genie 世界模型（GenieRedux），增加真实动作条件、Token 距离交叉熵（TDCE）损失和 token 跳连得到 GenieRedux-G，并提出 AutoExplore 探索智能体用世界模型的 token 预测不确定性作为内在奖励驱动多样数据收集，将仿真质量提升高达 7.4 PSNR。

**[Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)](model_compression/exploring_contextual_attribute_density_in_referring_expression_counting.md)**

:   提出上下文属性密度（Contextual Attribute Density, CAD）概念来增强指代表达计数（Referring Expression Counting），通过 U 形密度估计器、CAD 注意力和动态查询初始化三个模块，在 REC-8K 数据集上相比 GroundingREC 降低了约 30% 的计数误差（MAE 从 6.80 降至 5.43）。

**[Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)**

:   提出 FPET（Faster Parameter-Efficient Tuning），在参数高效微调（PET）中引入即插即用的 token 冗余压缩模块——在 ViT 中间层用可微的二分匹配策略合并约一半的 token，实现比原始 backbone 更快 20% 的推理速度、减少约 40% GPU显存、且精度与 SOTA PET 方法持平。

**[FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](model_compression/fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)**

:   提出 FIMA-Q，通过对角+低秩（DPLR）的 Fisher 信息矩阵近似替代传统对角近似，更准确地捕捉量化误差对输出分布的影响，在 3-bit 极低比特 ViT 量化中大幅超越现有方法（ViT-B 77.63% vs QDrop 74.75%）。

**[Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](model_compression/gaze-lle_gaze_target_estimation_via_large-scale_learned_encoders.md)**

:   提出 Gaze-LLE，一个基于冻结 DINOv2 编码器的极简视线目标估计框架——仅用 ~2.8M 可训练参数（比先前方法少 1-2 个数量级）、无需辅助深度/姿态模型、无需独立头部编码器，通过人物位置提示 + 轻量 transformer 解码器即在 GazeFollow/VideoAttentionTarget 等基准上达到 SOTA（AUC 0.958）。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](model_compression/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。

**[HOT: Hadamard-based Optimized Training](model_compression/hot_hadamard-based_optimized_training.md)**

:   提出HOT方法，通过对反向传播中不同梯度路径（激活梯度$g_x$和权重梯度$g_m$）的差异化灵敏度分析，选择性地应用Hadamard变换+量化——$g_x$用HT+INT4加速计算、$g_m$用HLA+INT8节省激活内存，实现75%激活内存节省和2.6倍GPU加速，ViT-B在ImageNet上精度仅降0.17%。

**[HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](model_compression/hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)**

:   提出 HyperLoRA，一种通过自适应网络直接生成 LoRA 权重的零样本个性化肖像生成方法——将 LoRA 参数投影到低维线性空间（原参数的 1.2%），用 perceiver resampler 从输入人脸预测组合系数，并将 LoRA 显式分解为 ID-LoRA 和 Base-LoRA 以解耦身份与无关信息，实现高保真度+高可编辑性+快速推理的平衡。

**[Incremental Object Keypoint Learning (KAMP)](model_compression/incremental_object_keypoint_learning.md)**

:   首次定义增量关键点学习（IKL）范式——新任务只标注新关键点、不保留旧数据的增量训练，提出 KAMP 框架通过知识关联网络（KA-Net）建模新旧关键点间的解剖学空间关系，配合关键点导向的空间蒸馏损失，在 4 个数据集上不仅有效防遗忘，甚至实现了对旧关键点的正向迁移提升（MPII AAA 79.93% vs LWF 75.75%）。

**[InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](model_compression/instag_learning_personalized_3d_talking_head_from_few-second_video.md)**

:   提出 InsTaG，通过 Identity-Free Pre-training 从多人长视频中提取通用运动先验，再通过 Motion-Aligned Adaptation 仅用 5 秒视频即可快速学习高保真个性化 3D 说话人头像，实现 82.5 FPS 实时推理。

**[JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)**

:   JamMa提出了基于Joint Mamba的超轻量级半密集特征匹配器，通过JEGO扫描-合并策略实现跨视角联合扫描、高效四方向扫描、全局感受野和全方向特征表示，以不到50%的参数和FLOPs实现了优于Transformer-based匹配器的性能-效率平衡。

**[Layered Image Vectorization via Semantic Simplification](model_compression/layered_image_vectorization_via_semantic_simplification.md)**

:   本文提出一种渐进式图像矢量化方法，利用 Score Distillation Sampling（SDS）的特征平均效应生成逐级简化的图像序列，以此引导从宏观语义结构到精细细节的分层矢量重建，在视觉保真度、语义对齐和紧凑分层表示上显著优于现有方法。

**[Learned Image Compression with Dictionary-based Entropy Model](model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)**

:   提出基于字典的交叉注意力熵模型 (DCAE)，引入可学习字典从训练数据集中提取自然图像的典型纹理结构先验，通过多尺度特征聚合 + 交叉注意力实现精确的概率分布估计，在编解码速度仅 193ms 的条件下实现 -17.0%/-21.1%/-19.7% 的 BD-rate（Kodak/Tecnick/CLIC），全面超越 SOTA。

**[PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval](model_compression/learning_compatible_multi-prize_subnetworks_for_asymmetric_retrieval.md)**

:   提出 PrunNet（可剪枝网络），通过为每个权重学习重要性分数并结合冲突感知梯度集成，训练一个可以在任意容量（20%-100%）下产生兼容子网络的统一模型，在 GLDv2 上 46.29 mAP 超越密集网络基线，且所有容量子网络间特征兼容。

**[LALIC: Linear Attention Modeling for Learned Image Compression](model_compression/linear_attention_modeling_for_learned_image_compression.md)**

:   首次将 RWKV 线性注意力机制引入学习图像压缩，设计 Bi-RWKV 变换块实现线性复杂度的全局感受野特征提取，配合 RWKV 时空通道上下文熵模型，以较低复杂度超越 VTM-9.1 达 15.26% BD-rate。

**[Logits DeConfusion with CLIP for Few-Shot Learning](model_compression/logits_deconfusion_with_clip_for_few-shot_learning.md)**

:   发现 CLIP 在下游任务中 logits 存在严重的类间混淆问题，提出 Logits DeConfusion（LDC）方法，通过多层级 Adapter 融合（MAF）增强特征表示，结合类间去混淆模块（ICD）以残差结构学习并消除混淆模式，在 11 个基准上取得 SOTA。

**[LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)**

:   LoRA-DRS 提出"LoRA 减法"操作——在学习新任务前将旧任务的 LoRA 权重从预训练权重中减去以构建漂移抵抗空间（DRS），然后在该空间中通过梯度投影训练新任务的 LoRA，结合增强三元组损失提升可塑性，在无样本持续学习中实现了 SOTA 性能，尤其在长任务序列上优势显著。

**[LSNet: See Large, Focus Small](model_compression/lsnet_see_large_focus_small.md)**

:   受人类视觉外周（广域感知）-中央（精细聚合）的双尺度机制启发，提出 LS 卷积（大核深度卷积感知 + 小核动态卷积聚合），构建 LSNet 轻量网络家族，在 0.3~1.3G FLOPs 下全面超越现有 SOTA 轻量模型。

**[Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](model_compression/mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)**

:   提出 Mamba-Adaptor，通过两个模块增强 Vision Mamba/SSM：Adaptor-T（时序）用可学习记忆选择机制保留关键历史状态，Adaptor-S（空间）用多尺度空心深度卷积增强空间局部性，在 ImageNet 上 83.0% Top-1（Mamba-Adaptor-b2），检测/分割+迁移学习全面提升。

**[MambaIC: State Space Models for High-Performance Learned Image Compression](model_compression/mambaic_state_space_models_for_high-performance_learned_image_compression.md)**

:   首次将 SSM 同时整合到学习型图像压缩的非线性变换和上下文模型中，通过 VSS block 增强通道-空间上下文建模 + 窗口局部注意力消除空间冗余，在 Kodak 上比 VVC 节省 12.52% BD-rate，且高分辨率图像压缩优势更加显著。

**[Masking Meets Supervision: A Strong Learning Alliance](model_compression/masking_meets_supervision_a_strong_learning_alliance.md)**

:   提出 Masked Sub-branch (MaskSub)——在监督学习中引入高比例 (50%) mask 增强的通用框架，通过主分支(无mask)和子分支(有mask)的自蒸馏结构解决强 mask 增强导致训练不稳定的问题，在 DeiT-III、MAE 微调、CLIP 微调、BERT 训练以及 ResNet/Swin 等多种场景中均取得一致性能提升。

**[MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](model_compression/mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)**

:   提出 MobileMamba 轻量级视觉网络，通过三阶段粗粒度架构设计和 MRFFI 细粒度模块（融合 Mamba 全局建模、多核卷积多尺度感知和 Identity 冗余消除），在分类和下游高分辨率任务上实现速度与精度的最优平衡。

**[MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation](model_compression/mutri_multi-view_tri-alignment_for_oct_to_octa_3d_image_translation.md)**

:   本文提出MuTri，首次将向量量化（VQ）引入OCT到OCTA的3D体积翻译任务，通过两阶段训练——先预训练OCT和OCTA重建VQVAE提供多视图先验，再用对比语义对齐（3D OCT/OCTA视图）和血管结构对齐（2D OCTA投影图视图）三视图指导翻译VQVAE的码本学习，在三个数据集上全面超越SOTA。

**[NADER: Neural Architecture Design via Multi-Agent Collaboration](model_compression/nader_neural_architecture_design_via_multi-agent_collaboration.md)**

:   NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验，仅 10 次试验即突破 NAS-Bench-201 搜索空间的准确率上限，在 CIFAR-100 上达 74.51%（搜索空间最优 73.51%）。

**[Targeted Forgetting of Image Subgroups in CLIP Models](model_compression/targeted_forgetting_of_image_subgroups_in_clip_models.md)**

:   提出三阶段 CLIP 模型子群遗忘方法：遗忘阶段用相对 Fisher 信息定位关键层+LoRA 微调遗忘目标类；提醒阶段用分布对齐在保留集上恢复；恢复阶段用模型汤（model souping）恢复零样本能力。在 ImageNet-1K 上遗忘分数达 91.0（基线 68.9）。

**[Understanding Multi-Layered Transmission Matrices](model_compression/understanding_multi-layered_transmission_matrices.md)**

:   分析体散射介质的多层传输矩阵近似，发现"缺失锥"（光学孔径导致的频率域约束）使所需 SLM 层数远少于 Nyquist 采样预测——200μm 组织理论需要 100 层但实际只需 3-4 层即可实现良好聚焦，视场从 1×1μm 扩展到 13×13μm。

---

## 📹 视频理解 { #video_understanding }

**[BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)**

:   提出 BehaviorVLM，一个统一的无需微调的视觉语言框架，通过多阶段结构化推理管线同时解决动物姿态估计和行为理解两大任务，仅需 3 帧人工标注即可实现可靠的关键点追踪，并通过深度嵌入聚类 + VLM 描述 + LLM 语义合并实现可解释的多动物行为分割。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   提出 R-MSD（Reliable Multi-Sample Distillation），通过对每个输入采样多个教师响应并结合任务自适应质量匹配，解决视频 LVLM 黑盒蒸馏中单样本教师监督不可靠的问题，4B 学生模型在 VideoMME (+1.5%)、Video-MMMU (+3.2%)、MathVerse (+3.6%) 等基准上取得一致提升。

**[BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](video_understanding/bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)**

**[Bimba: Selective-Scan Compression for Long-Range Video Question Answering](video_understanding/bimba_selective-scan_compression_for_long-range_video_question_answering.md)**

**[Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-Grained View-Invariant Video Representations](video_understanding/bootstrap_your_own_views_masked_ego-exo_modeling_for_fine-grained_view-invariant.md)**

:   通过掩码建模在自我中心和外部视角之间学习细粒度视图不变表示，无需配对标注即可从两种视角的关联中自监督学习

**[Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models](video_understanding/coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)**

:   本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

**[Context-Enhanced Memory-Refined Transformer for Online Action Detection](video_understanding/context-enhanced_memory-refined_transformer_for_online_action_detection.md)**

:   本文揭示了现有在线动作检测（OAD）方法中的训练-推理不一致问题——短时记忆帧的不均衡上下文暴露和伪未来引入的非因果信息泄漏导致学习偏向中间帧——并提出CMeRT通过近过去上下文增强编码器和基于近未来的记忆精炼解码器来解决该问题，在THUMOS'14、CrossTask和EK100上实现SOTA。

**[Cross-modal Causal Relation Alignment for Video Question Grounding](video_understanding/cross-modal_causal_relation_alignment_for_video_question_grounding.md)**

:   通过因果干预消除视频问答定位（VideoQG）中的虚假跨模态关联，引入高斯平滑定位、跨模态对齐和显式因果干预三个模块，在 NextGQA 上同时提升定位（+2.2 Acc@GQA）和问答（+0.9 Acc@VQA）性能。

**[DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)**

:   提出DeCafNet，通过**delegate-and-conquer双编码器策略**（轻量sidekick encoder密集提特征+生成显著性图，expert encoder仅处理top-c%关键clip），配合**DeCaf-Grounder**统一不同时序分辨率特征，在长视频时序定位任务上以**减少47% TFLOPs**的代价超越所有先前方法。

**[DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](video_understanding/divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)**

:   将视觉token剪枝问题重新建模为**Max-Min Diversity Problem (MMDP)**，通过精确求解使保留token集合的**最小pair-wise距离最大化**，实现无需训练/校准的即插即用剪枝方案，在16个多模态基准上实现SOTA，特别是在≥80%极端剪枝率下显著优于所有基线。

**[DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework](video_understanding/dpflow_adaptive_optical_flow_estimation_with_a_dual-pyramid_framework.md)**

:   提出DPFlow，结合**图像金字塔**和**特征金字塔**的双金字塔循环编码器，配合纯卷积的**Cross-Gated Unit (CGU)**，仅用标准分辨率训练即可自适应泛化至8K分辨率输入，在Sintel、KITTI、Spring等基准上达到SOTA，同时发布**Kubric-NK**多分辨率光流评测数据集首次支持定量高分辨率评估。

**[DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](video_understanding/dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)**

:   提出**Dynamic Prototype Updating (DPU)**框架，通过**Cohesive-Separate对比训练**建立稳健表示空间、**动态原型逼近**自适应更新类中心、**Pro-ratio差异增强**按样本到原型的距离调节多模态预测差异的放大强度，作为即插即用模块在5个数据集×9种基础OOD方法上全面提升性能，Far-OOD检测提升最高达**80%**。

**[DrVideo: Document Retrieval Based Long Video Understanding](video_understanding/drvideo_document_retrieval_based_long_video_understanding.md)**

:   提出DrVideo，将**长视频理解转化为长文档理解**任务：先将视频帧转为文本文档，通过**文档检索**定位关键帧并**增强信息**，再通过**Planning-Interaction双Agent循环**迭代补充缺失信息，最终以CoT方式回答问题。在EgoSchema（3分钟）、MovieChat-1K（10分钟）和Video-MME长视频分割（平均44分钟）上大幅超越现有LLM-based SOTA。

**[Dynamic Updates for Language Adaptation in Visual-Language Tracking](video_understanding/dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)**

:   提出DUTrack，通过动态更新多模态参考信息（模板帧+语言描述）来解决视觉语言跟踪中静态参考与动态目标之间的语义不一致问题，首次让VL跟踪器在LaSOT上超越最佳纯视觉跟踪器。

**[DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](video_understanding/dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)**

:   提出DynFocus，一个基于LLM的动态协作视频编码网络，通过DPE模块动态选择与问答相关的关键帧，CCE模块对关键帧用细粒度token编码（类似视锥细胞Cones）、对冗余帧用极少token粗粒度编码（类似视杆细胞Rods），在有限token预算下平衡空间细节与时序动态。

**[EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](video_understanding/edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)**

:   提出EDCFlow，利用相邻事件帧之间时间密集的特征差分图与低分辨率代价体积的互补性，在1/4分辨率上实现高质量且轻量的事件光流估计。

**[Efficient Transfer Learning for Video-language Foundation Models](video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)**

:   提出多模态时空适配器MSTA，通过视觉-语言共享投影层和时空描述引导的一致性约束，以仅2-7%的可训练参数实现视频-语言基础模型向下游任务的高效迁移。

**[Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](video_understanding/ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)**

:   提出Ego4o统一框架，从穿戴设备的多模态输入（1-3个IMU + 第一人称图像 + 运动描述）同时实现人体运动捕捉和运动描述生成，且两个任务可互相增强。

**[EgoLife: Towards Egocentric Life Assistant](video_understanding/egolife_towards_egocentric_life_assistant.md)**

:   发布EgoLife数据集（6名参与者共居一周、300小时第一人称多模态视频）和EgoLifeQA基准，提出EgoButler系统（EgoGPT + EgoRAG）探索超长上下文第一人称视觉生活助手的建设路径。

**[EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)**

:   提出 EgoTextVQA 基准，包含 1.5K 第一人称视频和 7K 场景文字相关问答对，揭示了当前 MLLM 在以自我中心视角进行实时场景文字问答辅助时的严重不足（最佳模型 Gemini 1.5 Pro 仅约 33% 准确率）。

**[Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](video_understanding/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)**

:   AoTD 用 LLM agent 将复杂视频问题分解为子任务、调用专家视觉模型执行并收集中间结果作为推理链（CoT），经 LLM 质量过滤后蒸馏到 Video-LLM 中，让端到端模型同时获得准确答案和可解释的多步推理能力。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一个轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的外观特征过滤和重叠 tracklet 对内的相似度比较，在线纠正因目标重叠导致的检测-轨迹错误匹配，将长期身份切换比例从 36.86% 降至 29.55%，同时在 MOT17/MOT20 上保持 SOTA 性能。

**[FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](video_understanding/fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)**

:   提出 FSAnno/FSBench，首个面向花样滑冰的细粒度、多模态、多层次基准数据集，覆盖从先验知识测试、单个动作识别/评估/解说到整体表演评估/解说的完整任务链，揭示了现有 LLM 在艺术体育理解上的显著不足。

**[GG-SSMs: Graph-Generating State Space Models](video_understanding/gg-ssms_graph-generating_state_space_models.md)**

:   提出 Graph-Generating State Space Models (GG-SSMs)，通过基于特征相似度动态构建最小生成树（MST）来替代传统 SSM 中固定的一维扫描路径，实现对高维数据中复杂非局部依赖的高效建模，在 11 个数据集上取得 SOTA 性能。

**[H-MoRe: Learning Human-centric Motion Representation for Action Analysis](video_understanding/h-more_learning_human-centric_motion_representation_for_action_analysis.md)**

:   提出 H-MoRe（Human-centric Motion Representation），通过骨骼约束和边界约束的联合自监督学习框架，从真实场景中学习精确的以人为中心的运动表示（world-local flows），在步态识别（CL@R1 +16.01%）、动作识别（Acc@1 +8.92%）和视频生成（FVD -67.07%）上均大幅超越传统光流方法。

**[Heterogeneous Skeleton-Based Action Representation Learning](video_understanding/heterogeneous_skeleton-based_action_representation_learning.md)**

:   首次研究人体骨架数据的异构性问题（不同关节数、不同坐标维度），提出通过 3D 姿态估计模块统一维度、骨架特定 prompt 统一拓扑、语义运动编码引入语义信息三大组件，结合自监督统一表示学习框架，在 NTU-60/120 和 PKU-MMD II 上取得显著提升。

**[HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](video_understanding/hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)**

:   提出 HierarQ，一种任务感知的层次化 Q-Former 框架，通过双流语言引导特征调制器（实体流 + 场景流）和短/长期记忆库实现自回归式逐帧视频处理，无需帧采样即可绕过 LLM 上下文长度限制，在 10 个视频理解基准上取得 SOTA 或接近 SOTA 的性能。

**[HuMoCon: Concept Discovery for Human Motion Understanding](video_understanding/humocon_concept_discovery_for_human_motion_understanding.md)**

:   HuMoCon 是一个面向人体行为分析的运动-视频理解框架，其核心创新是在编码器预训练阶段通过显式的视频-运动特征对齐和基于速度重建的高频信息保持机制来发现语义化的运动概念（codebook），从而显著提升下游 LLM 的人体运动理解和推理能力。

**[HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)**

:   HyperGLM 提出将实体场景图（捕捉空间关系）和程序图（建模因果时序转换）统一为超图 (HyperGraph)，并将其注入多模态 LLM 实现视频场景图的生成、预测和推理，同时发布包含 190 万帧的 VSGR 数据集支持五类任务。

**[KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](video_understanding/keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)**

:   KeyFace 提出一个两阶段扩散框架——先以低帧率生成捕捉关键表情的锚帧，再通过插值模型填充中间帧——解决了现有音频驱动面部动画方法在长序列中身份漂移和质量退化的问题，同时首次支持连续情感（valence/arousal）建模和多种非语音发声 (NSV) 的动画生成。

**[Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)**

:   提出 AVIGATE 框架，通过门控注意力机制选择性地融合音频与视觉信息（过滤无用音频噪声），并设计自适应间距对比损失处理视频-文本之间模糊的正负关系，在多个视频-文本检索基准上取得 SOTA。

**[Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](video_understanding/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)**

:   提出 ORTrack 框架，通过基于空间 Cox 过程的随机遮罩来学习遮挡鲁棒的 ViT 特征表征（训练时加遮罩约束、推理时零开销），并设计自适应特征蒸馏方法将大模型压缩为轻量级学生模型 ORTrack-D，在多个无人机跟踪基准上实现 SOTA 精度与实时速度的最佳平衡。

**[LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](video_understanding/lion-fs_fast_slow_video-language_thinker_as_online_video_assistant.md)**

:   提出 LION-FS 在线视频助手框架，借鉴"快思考-慢思考"认知理论，用 Fast Path（基于路由的 Token 聚合与丢弃）实现高效实时响应判断，用 Slow Path（多粒度关键帧增强）在响应生成时注入细粒度空间和交互特征，在 Ego4D/Ego-Exo4D 基准上全面超越现有方法。

**[LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](video_understanding/llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)**

:   针对日常生活活动（ADL）理解，构建了多视角多模态指令微调数据集 ADL-X，提出 LLAVIDAL 模型融合视频、3D 骨架和 HOI 线索，采用 MMPro 渐进式训练策略实现 SOTA 性能。

**[Locality-Aware Zero-Shot Human-Object Interaction Detection](video_understanding/locality-aware_zero-shot_human-object_interaction_detection.md)**

:   提出 LAIN 框架，通过局部适配器（LA）和交互适配器（IA）增强 CLIP 表示的局部细粒度感知和交互推理能力，在多种零样本 HOI 检测设定下达到 SOTA。

**[Localizing Events in Videos with Multimodal Queries](video_understanding/localizing_events_in_videos_with_multimodal_queries.md)**

:   提出 ICQ 基准和 ICQ-Highlight 数据集，首次系统研究用多模态查询（图像+文本）替代纯文本查询进行视频事件定位，并设计 3 种查询适配方法和 SUIT 代理微调策略。

**[M-LLM Based Video Frame Selection for Efficient Video Understanding](video_understanding/m-llm_based_video_frame_selection_for_efficient_video_understanding.md)**

:   提出一个轻量级 M-LLM 帧选择器，通过空间和时序伪标签训练，自适应地为下游视频 LLM 选取与问题最相关的帧，无需微调下游模型即可提升多个视频 QA 基准性能。

**[MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)**

:   首个基于 Mamba 的视觉语言跟踪器 MambaVLT，利用状态空间的时间演化特性实现长时序目标信息记忆和多模态参考特征的自适应更新，在多个视觉语言跟踪基准上达到 SOTA。

**[MITracker: Multi-View Integration for Visual Object Tracking](video_understanding/mitracker_multi-view_integration_for_visual_object_tracking.md)**

:   提出多视角目标跟踪数据集 MVTrack（234K 帧，27 类目标）和方法 MITracker，通过将 2D 特征投影到 3D 特征体并压缩为 BEV 平面进行跨视角融合，结合空间增强注意力修正各视角跟踪结果，实现从遮挡中快速恢复跟踪。

**[MLVU: Benchmarking Multi-task Long Video Understanding](video_understanding/mlvu_benchmarking_multi-task_long_video_understanding.md)**

:   提出 MLVU 基准，通过9种多样化评测任务、多种视频类型和灵活的时长设置，系统评估多模态大模型在长视频理解上的能力，揭示现有模型在处理长视频时的显著不足。

**[MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](video_understanding/mmvu_measuring_expert-level_multi-discipline_video_understanding.md)**

:   提出 MMVU 基准，包含 3,000 个专家标注的跨 27 个学科的视频理解题目，评估多模态基础模型在专业领域视频中的专家级知识推理能力，揭示即使最强模型仍显著落后于人类专家。

**[MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](video_understanding/must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)**

:   提出首个大规模多光谱无人机单目标跟踪数据集 MUST（250 序列、43K 帧、8 光谱波段），并设计 UNTrack 统一框架融合光谱、空间、时序特征，通过非对称 Transformer 和光谱提示编码器实现高效鲁棒跟踪。

**[Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](video_understanding/reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](video_understanding/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)**

:   VCBench 将计数重新定位为诊断视频模型"时空状态维护"能力的最小探针，提出了覆盖物体计数（当前状态/身份追踪）和事件计数（瞬时事件/周期活动）的 8 种子类别，通过沿时间线的流式多点查询观察模型预测轨迹，在 406 个视频/4576 个查询点上评估主流模型，发现当前模型在时空状态维护上仍存在显著缺陷。

**[Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](video_understanding/video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)**

:   提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。

---

## 🎬 视频生成 { #video_generation }

**[4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](video_generation/4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)**

:   提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。

**[AnimateAnything: Consistent and Controllable Animation for Video Generation](video_generation/animateanything_consistent_and_controllable_animation_for_video_generation.md)**

:   提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。

**[Articulated Kinematics Distillation from Video Diffusion Models](video_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)**

:   提出 AKD，将视频扩散模型（CogVideoX-5B）的运动先验通过 SDS 蒸馏到 3D 铰链角色的关节角度参数中，结合可微分前向运动学+高斯散射渲染+物理约束，实现文本驱动的真实角色动画，VideoPhy SA 得分从 TC4D 的 0.40 提升到 0.81。

**[BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](video_generation/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)**

:   提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。

**[Can Text-to-Video Generation Help Video-Language Alignment?](video_generation/can_text-to-video_generation_help_video-language_alignment.md)**

:   提出 SynViTA 框架探索文本到视频生成模型产生的合成视频能否改善视频-语言对齐（VLA），通过基于对齐质量的样本加权和语义一致性正则化解决合成视频的语义不一致和外观偏差问题，在时序挑战性任务上提升 4+ 个点。

**[ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](video_generation/conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)**

:   ConMo提出了一种零样本运动迁移框架，通过将参考视频中的复合运动解耦为独立的主体运动和背景（相机）运动，再在目标视频生成时可控地重组这些运动，实现了多主体运动迁移、语义/形状变换、主体去除、相机运动模拟等多种应用，在运动保真度和文本对齐上显著超越现有方法。

**[Dynamic Camera Poses and Where to Find Them](video_generation/dynamic_camera_poses_and_where_to_find_them.md)**

:   提出DynPose-100K——一个包含10万个动态互联网视频及其相机位姿标注的大规模数据集，通过专用模型组合+VLM的视频过滤管线和集成最新点跟踪+动态掩码+全局BA的位姿估计管线实现。

**[DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes](video_generation/dynamicscaler_seamless_and_scalable_video_generation_for_panoramic_scenes.md)**

:   DynamicScaler 提出了一个无需微调的统一框架，通过偏移移位去噪器（OSD）和全局运动引导（GMG）实现任意分辨率/宽高比的全景动态场景合成，支持常规全景和 360° 视野视频生成，同时保持恒定 VRAM 消耗。

**[Exploring Temporally-Aware Features for Point Tracking](video_generation/exploring_temporally-aware_features_for_point_tracking.md)**

:   提出 Chrono，一个为点跟踪设计的时序感知特征骨干网络，通过在 DINOv2 的 Transformer 块间插入时序适配器（2D 卷积下采样 + 1D 局部时序注意力 + 2D 卷积上采样），仅通过简单的特征匹配（soft-argmax）即可在无精炼器设定下达到 SOTA 表现。

**[FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](video_generation/fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)**

:   提出 FADE，一种免训练的视频编辑方法，通过分析 T2V 模型中各 transformer block 的频率角色（sketching vs sharpening），利用频谱引导调制在频域中分离保留与编辑内容，实现高质量的外观和运动编辑。

**[From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](video_generation/from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)**

:   CausVid 通过非对称蒸馏将预训练的双向视频扩散 Transformer 蒸馏为因果自回归 4 步生成器，结合 ODE 初始化和 KV 缓存，实现 9.4 FPS 的流式视频生成（比 CogVideoX 快 160×），在 VBench-Long 基准上以 84.27 分排名第一。

**[Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](video_generation/generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)**

:   提出 FCVG，通过从两个关键帧中提取匹配线段并逐帧线性插值作为帧级条件，注入 SVD 视频生成模型，显著消解了生成式中间帧合成中前向/反向路径的模糊性，实现时序稳定的视频插帧。

**[HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](video_generation/hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)**

:   HOIGen-1M 是首个面向人物交互 (HOI) 视频生成的百万级高质量数据集，通过高效数据筛选管线和 Mixture-of-Multimodal-Experts (MoME) 字幕策略解决了 HOI 视频数据稀缺和描述幻觉问题，并提出 CoarseHOIScore/FineHOIScore 两个评估指标来量化生成视频中交互的质量。

**[HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](video_generation/hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)**

:   HunyuanPortrait提出了首个基于Stable Video Diffusion的隐式条件肖像动画框架，通过强度感知运动编码器和ID感知多尺度适配器实现了对精细面部动态的高保真控制和强身份一致性。

**[HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks](video_generation/hypernvd_accelerating_neural_video_decomposition_via_hypernetworks.md)**

:   HyperNVD 提出利用超网络 (Hypernetwork) 根据 VideoMAE 编码的视频嵌入动态生成隐式神经表示 (INR) 的参数，实现跨视频的通用视频分解模型，在新视频上可比从头训练快 30+ 分钟达到相同 PSNR，同时最终性能平均提升 0.8dB。

**[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](video_generation/identity-preserving_text-to-video_generation_by_frequency_decomposition.md)**

:   ConsisID 提出基于频率分解的 DiT 控制方案，将人脸特征解耦为低频全局信息和高频内在身份信息，分别注入 DiT 的不同位置，实现免微调的身份保持文本到视频生成，在身份保持、文本相关性和视觉质量上全面超越现有方法。

**[IDOL: Instant Photorealistic 3D Human Creation from a Single Image](video_generation/idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)**

:   IDOL 通过构建包含 10 万人体的大规模多视角数据集 HuGe100K，训练基于 Transformer 的前馈模型在单张图片输入下实现即时（<1秒）的高保真可动画 3D 人体重建，在质量和泛化能力上大幅超越现有方法。

**[Improved Video VAE for Latent Video Diffusion Model](video_generation/improved_video_vae_for_latent_video_diffusion_model.md)**

:   本文提出 IV-VAE，通过关键帧时序压缩架构（KTC）和组因果卷积（GCConv）解决现有视频 VAE 中图像权重初始化抑制时序压缩学习、以及因果卷积导致帧间性能不均衡的问题，在多个基准上实现 SOTA 视频重建和生成质量。

**[InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](video_generation/interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)**

:   InterDyn 提出将视频扩散模型作为隐式物理引擎，通过在 Stable Video Diffusion 上引入交互控制分支（ControlNet-like），从单帧图像和驱动运动信号生成物理上合理的交互动力学视频，在 Something-Something-v2 数据集上 FVD 指标超过基线 CosHand 达 77%。

**[Learning from Streaming Video with Orthogonal Gradients](video_generation/learning_from_streaming_video_with_orthogonal_gradients.md)**

:   针对流式视频学习中连续帧高度相关导致梯度冗余、模型崩溃的问题，提出正交梯度优化器（Orthogonal Optimizer），通过将当前梯度投影到历史梯度的正交分量来去相关，可无缝集成到 SGD/AdamW 中，在 DoRA、VideoMAE、未来预测三个场景下均显著恢复了从打乱训练到顺序训练的性能损失。

**[Learning Temporally Consistent Video Depth from Video Diffusion Priors](video_generation/learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)**

:   提出 ChronoDepth——基于 Stable Video Diffusion (SVD) 的视频深度估计方法，通过在训练时为每帧独立采样噪声水平并在推理时使用无噪声前序帧作为上下文（Consistent Context-Aware Strategy），在保持空间精度的同时实现了 SOTA 的时序一致性，MFC 指标平均排名第一。

**[LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](video_generation/levitor_3d_trajectory_oriented_image-to-video_synthesis.md)**

:   LeviTor首次在image-to-video合成中引入3D物体轨迹控制，通过将物体mask用K-means聚类为少量代表点并结合深度信息作为控制信号注入SVD模型，实现了遮挡关系、前后移动和环绕等复杂3D运动的精准控制，在DAVIS上FID/FVD分别达到25.41/190.44。

**[Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](video_generation/long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)**

:   Presto 提出分段交叉注意力（SCA）策略，将隐状态沿时间维度分段并与对应子描述分别交叉注意力，结合精心策展的 261K 高质量长视频数据集 LongTake-HD，实现了 15 秒内容丰富且长程连贯的视频生成，在 VBench 语义得分达到 78.5%、Dynamic Degree 达到 100%。

**[LongDiff: Training-Free Long Video Generation in One Go](video_generation/longdiff_training-free_long_video_generation_in_one_go.md)**

:   LongDiff 通过理论分析揭示短视频模型生成长视频时的两个关键挑战——时序位置模糊和信息稀释，并提出 Position Mapping（GROUP+SHIFT）和 Informative Frame Selection（IFS）两个简洁的时序注意力修改策略，无需训练即可让短视频模型一次性生成高质量长视频。

**[Mimir: Improving Video Diffusion Models for Precise Text Understanding](video_generation/mimir_improving_video_diffusion_models_for_precise_text_understanding.md)**

:   Mimir 提出一个端到端训练框架，通过精心设计的 Token Fuser 将 decoder-only LLM（Phi-3.5）的强文本理解能力与传统 text encoder（T5）的稳定特征无损融合，显著提升视频扩散模型的文本理解精度，尤其在多物体、空间关系和时序理解上大幅领先现有方法。

**[MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](video_generation/mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)**

:   MIMO 提出一种基于空间分解建模的角色视频合成框架，将 2D 视频按 3D 深度分层为人物、场景和遮挡物三个空间组件，通过解耦编码和组合解码实现了对角色身份、3D 运动和交互场景的灵活控制，在复杂运动和场景交互上显著超越先前方法。

**[Mind the Time: Temporally-Controlled Multi-Event Video Generation](video_generation/mind_the_time_temporally-controlled_multi-event_video_generation.md)**

:   提出 MinT，首个支持事件时间控制的多事件视频生成器，通过 Rescaled RoPE (ReRoPE) 位置编码将事件描述绑定到特定时间段，在预训练视频 DiT 上微调实现平滑连贯的多事件视频合成。

**[MotiF: Making Text Count in Image Animation with Motion Focal Loss](video_generation/motif_making_text_count_in_image_animation_with_motion_focal_loss.md)**

:   提出 Motion Focal Loss (MotiF)，通过光流生成运动热力图对扩散损失进行空间加权，引导模型关注高运动区域，显著提升 Text-Image-to-Video 生成中的文本遵循和运动质量，并构建 TI2V-Bench 评测基准。

**[Motion Modes: What Could Happen Next?](video_generation/motion_modes_what_could_happen_next.md)**

:   提出 Motion Modes，一种免训练方法，通过设计四种引导能量函数探索预训练图像到视频生成器的潜在分布，从单张图像中发现物体的多种合理且多样的运动模式，同时将物体运动与相机运动解耦。

**[Motion Prompting: Controlling Video Generation with Motion Trajectories](video_generation/motion_prompting_controlling_video_generation_with_motion_trajectories.md)**

:   将时空稀疏/稠密点轨迹作为"运动提示"训练ControlNet，用单一模型实现物体控制、相机控制、运动迁移、拖拽编辑等多种运动控制能力，并展现出逼真物理行为的涌现特性。

**[MotionPro: A Precise Motion Controller for Image-to-Video Generation](video_generation/motionpro_a_precise_motion_controller_for_image-to-video_generation.md)**

:   提出 MotionPro，利用区域级轨迹（region-wise trajectory）和运动掩码（motion mask）双重信号，实现细粒度、可区分物体/相机运动的精确可控图像到视频生成。

**[MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](video_generation/motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)**

:   提出 MotionStone，通过训练独立的运动强度估计器将视频运动解耦为物体运动和相机运动两个维度，并以解耦方式注入 Diffusion Transformer，实现精细的运动强度可控 I2V 生成。

**[Multi-subject Open-set Personalization in Video Generation](video_generation/multi-subject_open-set_personalization_in_video_generation.md)**

:   提出 Video Alchemist，在 Diffusion Transformer 架构中内置多主体、开放集的视频个性化生成能力，支持前景物体和背景的定制，无需测试时优化。

**[Navigation World Models](video_generation/navigation_world_models.md)**

:   本文提出Navigation World Model (NWM)，一个10亿参数的Conditional Diffusion Transformer (CDiT)，在多个机器人导航数据集和Ego4D无标签视频上联合训练，通过预测给定动作下的未来视觉观测来模拟导航轨迹，可用于MPC规划或对外部策略（如NoMaD）的轨迹排序，在RECON数据集上的ATE（1.13）和RPE（0.35）均显著优于现有导航策略。

**[NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification](video_generation/neuro-symbolic_evaluation_of_text-to-video_models_using_formal_verification.md)**

:   提出 NeuS-V，首个用形式化验证（时序逻辑+概率模型检验）评估文本到视频（T2V）模型时序一致性的框架——将文本提示转为时序逻辑规范，用 VLM 评分原子命题，构建视频自动机后形式化验证满足概率，在 Gen-3 上与人类标注 Pearson 相关 0.71（VBench 仅 0.47）。

**[SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](video_generation/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)**

:   提出 SAW（Surgical Action World），通过四种轻量级条件信号（语言提示、参考帧、组织功能图、工具轨迹）驱动视频扩散模型，实现可控、可扩展的手术动作视频生成，用于罕见动作增强和手术仿真。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](video_generation/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出一种LLM驱动的自适应多模态语义卫星传输系统，通过双流生成架构（V2A/A2V）灵活切换视频驱动音频生成和音频驱动视频生成，结合动态知识库更新机制，在极低带宽卫星链路上实现高保真音视频同步重建。

**[ViReS: Video Instance Repainting via Sketch and Text Guided Generation](video_generation/vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)**

:   提出ViReS框架，通过草图和文本双重引导实现视频中特定实例的重绘，利用时序注意力和实例掩码保持背景不变和时间一致性，在多种视频编辑场景下生成高质量结果。

**[When to Lock Attention: Training-Free KV Control in Video Diffusion](video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，一种基于扩散幻觉检测的免训练视频编辑框架，通过动态调度 KV 缓存融合比例和 CFG 引导尺度，在保持背景一致性的同时增强前景生成质量。

**[World2Act: Latent Action Post-Training via Skill-Compositional World Models](video_generation/world2act_latent_action_post-training_via_skill-compositional_world_models.md)**

:   World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。

---

## 💬 LLM/NLP { #llm_nlp }

**[Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Generalization](llm_nlp/attribute-formed_class-specific_concept_space_endowing_language_bottleneck_model.md)**

:   针对语言瓶颈模型（LBM）中所有概念混合在一起导致的虚假线索推理和零样本泛化差的问题，提出属性构成的类特定概念空间，将概念按属性维度为每个类别组织独立空间。

**[Breaking the Low-Rank Dilemma of Linear Attention](llm_nlp/breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   从理论上揭示线性注意力性能不及 Softmax 注意力的根本原因是输出特征的低秩问题，提出秩增强线性注意力（RALA），通过增强 KV 缓存秩和输出特征秩两种互补策略，在保持线性复杂度的同时追平甚至超越 Softmax 注意力的表现。

**[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](llm_nlp/bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)**

:   提出不确定性感知的模糊先验，为从脑信号（fMRI）重建视觉刺激提供物理合理的图像退化模型，缓解大脑编码过程中高频信息丢失对重建质量的影响。

**[Building Vision Models upon Heat Conduction](llm_nlp/building_vision_models_upon_heat_conduction.md)**

:   提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

**[Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)**

:   本文提出基于对话的行人检索（ChatPR）新范式，构建了首个对话-图像配对数据集ChatPedes，并设计了DiaNA框架通过自适应属性精炼器实现对话与图像间的细粒度跨模态对齐，显著优于传统单句文本检索方法。

**[Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](llm_nlp/classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)**

:   本文提出Classifier-to-Bias（C2B），首个无需任何标注数据的视觉分类器偏差自动发现框架，仅依靠分类任务的文本描述，利用LLM生成偏差候选并通过检索增强的生成-验证流程来识别预训练模型中的系统性偏差。

**[ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices](llm_nlp/comrope_scalable_and_robust_rotary_position_embedding_parameterized_by_trainable.md)**

:   ComRoPE将RoPE从固定的2D旋转矩阵推广到SO(n)群的更大子群，证明交换性是保持相对位置鲁棒性的充要条件，提出AP和LD两种可训练参数化方案，在ImageNet分类（+1.6%）、COCO检测（+0.2 AP）上均优于LieRE。

**[Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders](llm_nlp/dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Empowering LLMs to Understand and Generate Complex Vector Graphics](llm_nlp/empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   本文提出LLM4SVG，首个支持任意LLM进行SVG理解与生成的统一框架，通过可学习语义token精确编码SVG组件属性，配合模块化架构和580K条SVG指令微调数据（SVGX-SFT），显著超越GPT-4等基线在复杂矢量图形生成上的表现。

**[Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](llm_nlp/exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)**

:   本文提出Exposure-slot框架，将Slot Attention算法扩展为层次化的slot-in-slot结构，通过可学习的曝光prompt引导特征聚类，实现以曝光为中心的区域感知表征学习，在欠曝/过曝图像矫正任务上取得SOTA性能。

**[Guiding Human-Object Interactions with Rich Geometry and Relations](llm_nlp/guiding_human-object_interactions_with_rich_geometry_and_relations.md)**

:   本文提出ROG框架，通过在物体网格上采样富含几何信息的关键点构建交互距离场（IDF），并利用基于扩散的关系模型在推理时引导运动生成模型产生关系感知且语义对齐的人物-物体交互动作，在FullBodyManipulation数据集上显著超越SOTA。

**[Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](llm_nlp/imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

**[Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](llm_nlp/improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)**

:   本文深入分析LLM框架下视觉embedding空间的特性，发现视觉token间的相关性有助于实现更稳定的生成，据此提出IAR方法，通过码本重排（Codebook Rearrangement）和簇导向token预测（Cluster-Oriented Token Prediction）提升自回归视觉生成的效率和质量。

**[L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers](llm_nlp/l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas.md)**

:   本文提出L-SWAG（Layer-Sample Wise Activation with Gradients），一种新型通用零代价代理，通过结合层级和样本级的激活值与梯度信息来评估网络架构质量，首次将零代价NAS系统性地扩展到Vision Transformer搜索空间，并在Autoformer搜索空间的6个任务上建立了新的benchmark。

**[Learning Textual Prompts for Open-World Semi-Supervised Learning](llm_nlp/learning_textual_prompts_for_open-world_semi-supervised_learning.md)**

:   本文提出了一种针对开放世界半监督学习（OWSSL）的新方法，通过全局-局部文本提示学习策略增强图文对齐效果，并设计前向-反向策略降低无标签样本中图文匹配的噪声，在多个细粒度数据集上显著超越SOTA。

**[Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness](llm_nlp/let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)**

:   提出NSF方法，通过利用样本在特征空间中的聚类特性自动识别依赖虚假特征的样本组，无需组标注即可训练出对虚假相关性鲁棒的分类器，最差组准确度显著超越ERM基线。

**[Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](llm_nlp/lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)**

:   通过引入背景视频描述、历史翻译和伪词汇表三种上下文线索，结合Llama3-8B的LoRA微调，实现了连续手语到文本的精确翻译，在BOBSL数据集上相比SOTA提升40%以上。

**[Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration](llm_nlp/making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res.md)**

:   本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。

**[PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](llm_nlp/postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)**

:   本文提出PosterO，一种以布局为中心的海报生成方法，将数据集中的布局结构化为SVG语言的层次化树表示，通过通用形状表示、设计意图向量化和层次节点描述三大机制，使LLM能够通过in-context learning在推理时生成多样化的内容感知布局。

**[Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](llm_nlp/rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)**

:   本文深入分析了点积在脉冲查询-键对中因大量"非脉冲事件"导致相似度度量失效的根本原因，提出专为脉冲序列设计的a-XNOR相似度度量，将非脉冲对的相关性重定义为特定值a，在多种脉冲Transformer架构和数据集上显著提升性能。

**[RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives](llm_nlp/roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_.md)**

:   本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。

**[Robust Message Embedding via Attention Flow-Based Steganography](llm_nlp/robust_message_embedding_via_attention_flow-based_steganography.md)**

:   本文提出RMSteg（Robust Message Steganography）框架，首次将Transformer注意力机制集成到归一化流网络中（AttnFlow），配合可逆QR码转换和可逆Token融合模块，实现了高质量、高容量且鲁棒的消息-图像隐写，隐写图像即使经过打印-拍照等极端扭曲仍可准确解码。

**[SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](llm_nlp/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)**

:   本文提出SATA（Spatial Autocorrelation Token Analysis），一种免训练的ViT鲁棒性增强方法，通过空间自相关分析将token按空间关联模式分组，利用分组信息重新加权token表示，提升ViT在分布偏移和对抗攻击下的鲁棒性，且不影响干净样本性能。

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](llm_nlp/scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   首次在人类动作生成领域系统验证缩放律，提出包含Motion FSQ-VAE（解决codebook collapse）、260小时MotionUnion数据集和文本前缀自回归Transformer的可扩展系统ScaMo，发现归一化测试损失与FLOPs的对数律以及词汇参数/模型参数/数据量与FLOPs的幂律关系，并在$1\times 10^{18}$FLOPs预算下成功预测最优配置。

**[SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](llm_nlp/sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)**

:   提出 SEC-Prompt（SEmantic Complementary Prompt）框架，学习两组语义互补的提示——判别性提示（D-Prompt）和非判别性提示（ND-Prompt），通过自适应查询机制协同工作，分别强化类间区分和促进新类泛化，在三个基准数据集上取得 SOTA 性能。

**[SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](llm_nlp/softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)**

:   提出SoftShadow框架，用连续灰度软掩码替代传统二值硬掩码来表示阴影区域，通过SAM+LoRA预测软掩码并引入半影形成约束损失联合训练检测与去阴影网络，在SRD/ISTD+/LRSS/UIUC四个数据集上达到SOTA且无需外部掩码输入。

**[Spiking Transformer with Spatial-Temporal Attention](llm_nlp/spiking_transformer_with_spatial-temporal_attention.md)**

:   将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

**[STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](llm_nlp/staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   通过在SNN中集成全局上下文自注意(GC)、位置编码(PE)、步骤注意(SA)和时间步随机退出(TSRD)四大模块，STAA-SNN在CIFAR-10/100和ImageNet上达到97.14%/82.05%/70.40%的SNN SOTA性能。

**[Test-Time Visual In-Context Tuning](llm_nlp/test-time_visual_in-context_tuning.md)**

:   首次系统研究VICL模型在分布偏移下的鲁棒性问题，提出VICT方法利用循环一致性自监督信号在测试时进行单样本微调，在6个视觉任务和15种图像破坏下显著改进Painter等VICL模型。

**[The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation](llm_nlp/the_change_you_want_to_detect_semantic_change_detection_in_earth_observation_wit.md)**

:   本文提出HySCDG（Hybrid Semantic Change Detection Data Generation），一种混合数据生成流水线，结合真实超高分辨率（VHR）遥感影像和图像inpainting技术生成大规模语义变化检测训练数据，在简洁的架构设计下实现了强大的时间和空间泛化能力。

**[TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](llm_nlp/tide_training_locally_interpretable_domain_generalization_models_enables_test-ti.md)**

:   本文提出TIDE，一种针对单源域泛化的新型训练方案，利用扩散模型和LLM自动生成类别级概念标注（如"鸟类=尖嘴+翅膀+爪子"），通过概念显著性对齐损失训练模型关注域不变的局部概念而非全局背景特征，使模型在测试时能通过概念显著图自动矫正域偏移导致的错误预测。

---

## 🎯 目标检测 { #object_detection }

**[A Bias-Free Training Paradigm for More General AI-generated Image Detection](object_detection/a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)**

:   提出B-Free训练范式——通过stable diffusion的自条件重构从真实图像生成语义对齐的假图，结合inpainting内容增强，消除格式/内容/分辨率等偏差，使检测器聚焦于生成器特有的伪影痕迹，在27种生成模型（含FLUX、SD 3.5等最新模型）上泛化AUC>99%，balanced accuracy达95.2%。

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA（Aligned Basis Relocation for Adaptation），通过在权重空间中进行 SVD 分解与正交旋转对齐，将源域的类别特定检测知识"传送"到无标注数据的目标域，实现零样本跨域目标检测。

**[Any6D: Model-free 6D Pose Estimation of Novel Objects](object_detection/any6d_model-free_6d_pose_estimation_of_novel_objects.md)**

:   提出 Any6D 框架，仅从单张 RGB-D 锚点图像即可估计未知物体的 6D 位姿和尺寸，通过 InstantMesh 3D 重建 + 朝向包围盒粗对齐 + 联合尺寸-位姿精细化，在 HO3D 上 ADD-S 达 98.7% 远超 GEDI 的 71.9%。

**[BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs](object_detection/bacon_improving_clarity_of_image_captions_via_bag-of-concept_graphs.md)**

:   提出BACON提示方法，将VLM生成的冗长图像描述解构为物体、关系、风格、主题等解耦结构化元素（JSON字典格式），使下游模型无需强文本编码能力即可高效利用描述信息，在开放词汇目标检测中帮助GroundingDINO实现1.51倍的召回率提升。

**[Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)**

:   发现在域增量学习中选择最优参数子集比微调全部参数更有效，提出参数选择策略解决域增量目标检测的灾难性遗忘

**[Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](object_detection/co-spy_combining_semantic_and_pixel_features_to_detect_synthetic_images_by_ai.md)**

:   提出 Co-Spy 融合 VAE 重建伪影特征和 CLIP 语义特征两条互补检测路径——VAE 伪影跨模型泛化但怕 JPEG 压缩，CLIP 语义抗 JPEG 但泛化差——自适应调节器根据输入动态分配两路权重，在 22 个生成模型上建立新 SOTA。

**[DEIM: DETR with Improved Matching for Fast Convergence](object_detection/deim_detr_with_improved_matching_for_fast_convergence.md)**

:   通过两个简单改进加速 DETR 训练收敛——Dense O2O（用数据增强增加每图目标数实现稠密一对一匹配）和 MAL（替代 VFL 更好地优化低质量匹配），训练 epoch 减半同时性能提升（COCO AP 56.5 with D-FINE-X）。

**[Detecting Adversarial Data Using Perturbation Forgery](object_detection/detecting_adversarial_data_using_perturbation_forgery.md)**

:   通过建模对抗噪声的高斯分布并证明其近邻性，提出 Perturbation Forgery 方法在训练时持续扰动噪声分布形成开覆盖，配合稀疏掩码生成伪对抗数据训练二分类器，仅需 FGSM 一种攻击的噪声分布就能泛化检测梯度、GAN、扩散和物理等各类未见攻击，AUROC 达 0.99+ 且推理开销极低。

**[Detecting Out-of-Distribution through the Lens of Neural Collapse](object_detection/detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)**

:   从 Neural Collapse 理论出发，发现中心化后的 ID 特征聚集在预测类别的权重向量附近且远离原点（形成 simplex ETF），据此设计 NCI 检测器——结合特征与权重向量的角度近邻度（pScore）和特征范数过滤，在 CIFAR-10/100 和 ImageNet 多架构上实现最佳综合 OOD 检测性能且推理延迟与 softmax 基线持平。

**[DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](object_detection/diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)**

:   提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。

**[DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](object_detection/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)**

:   提出 DreamVideo-Omni，通过渐进式两阶段训练范式（Omni-Motion SFT + Latent Identity Reward Feedback Learning），在统一的 DiT 框架中实现多主体定制与全运动控制（全局 bbox + 局部轨迹 + 相机运动）的协同生成。

**[Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](object_detection/efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)**

:   提出首个面向大规模基准的混合 SNN-ANN 目标检测模型，设计注意力桥接模块（ASAB）将 SNN 的稀疏脉冲表示通过时空注意力转换为 ANN 可处理的密集特征，在 Gen1/Gen4 数据集上以仅 6.6M 参数大幅超越 SNN 方法并接近 ANN/RNN 方法的精度，同时 SNN 部分可部署在 Intel Loihi 2 神经形态芯片上实现低功耗推理。

**[Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](object_detection/efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)**

:   提出一种高效的持续测试时自适应目标检测（CTTA-OD）方法，发现源模型中某些特征通道对域偏移敏感且会损害跨域性能，通过在图像级和实例级度量通道敏感性来引导加权稀疏正则化实现选择性剪枝，辅以随机通道重激活机制防止误剪，在减少 12% 计算量的同时超越 SOTA 方法的自适应精度。

**[Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](object_detection/enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)**

:   本文提出 PRSS 方法，通过 Prompt Re-anchoring（将记忆化 prompt 重新用作 CFG 的锚点引导生成偏离记忆内容）和 Semantic Prompt Search（用 LLM 搜索语义相似但不触发记忆的替代 prompt）两个策略，在不修改模型和不需要训练数据的推理阶段改进 CFG 方程，实现了扩散模型记忆化缓解中的最优隐私-效用平衡。

**[ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](object_detection/escape_equivariant_shape_completion_via_anchor_point_encoding.md)**

:   ESCAPE 提出了一种基于锚点距离编码的旋转等变点云补全方法，通过将点云表示为到高曲率锚点的距离矩阵，使 Transformer 在旋转不变的距离空间中预测完整形状，再通过优化恢复 3D 坐标，在任意旋转输入下大幅超越现有方法（PCN 数据集 CD-L1 从 26.65 降至 10.58）。

**[Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)**

:   本文首次将扩散模型引入域泛化目标检测，通过提取扩散过程的多时间步中间特征构建域不变的检测器，并设计特征级+目标级对齐的知识迁移框架将泛化能力蒸馏到轻量检测器中，在6个DG基准上平均提升14.0% mAP，甚至超越大多数域适应方法。

**[HumanMM: Global Human Motion Recovery from Multi-shot Videos](object_detection/humanmm_global_human_motion_recovery_from_multi-shot_videos.md)**

:   HumanMM首次提出从多镜头视频中恢复世界坐标系下3D人体运动的框架，通过镜头转换检测器、增强SLAM、基于立体标定的朝向对齐和运动积分器，实现了跨镜头的连续运动重建。

**[Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays](object_detection/image_reconstruction_from_readout-multiplexed_single-photon_detector_arrays.md)**

:   本文将行列读出复用的单光子探测器阵列中的多光子碰巧分辨问题形式化为逆成像问题，提出了一种概率性的多光子估计器（Multiphoton Estimator），能够解析最多4个同时入射的光子的空间位置，在32×32阵列上相比传统方法提升3-4 dB PSNR，并将所需帧数减少约4倍。

**[Interpreting Object-level Foundation Models via Visual Precision Search](object_detection/interpreting_object-level_foundation_models_via_visual_precision_search.md)**

:   针对 Grounding DINO 和 Florence-2 等目标级基础模型的可解释性问题，本文提出 Visual Precision Search (VPS) 方法，通过超像素稀疏化+子模函数引导的贪心搜索精确定位关键决策子区域，在 MS COCO/RefCOCO/LVIS 上的忠实度指标(Insertion)分别超过 SOTA 方法 D-RISE 达 23.7%/20.1%/31.6%。

**[Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](object_detection/large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)**

:   DINO Teacher 提出用冻结的 DINOv2 大模型替代传统 Mean Teacher 框架中的 EMA 教师，一方面作为更准确的伪标签生成器，另一方面作为特征对齐的代理目标，在多个域自适应目标检测基准上取得了 SOTA 性能（BDD100k 上 +7.6%）。

**[MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](object_detection/mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)**

:   提出MCCD，通过MLLM驱动的多智能体协作场景解析模块和层次化组合扩散机制（高斯掩码+区域增强滤波），以免训练方式显著提升扩散模型在多物体、多属性、多关系复杂场景下的文生图生成质量。

**[MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](object_detection/mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)**

:   MI-DETR 提出了并行多次查询（MI）机制替代传统 DETR 级联解码器架构，让 object queries 通过多个参数独立的 inquiry heads 并行地从图像特征中学习多模式信息，配合 U-like Feature Interaction（UFI），在 COCO 上以 ResNet-50 backbone 达到 52.7 AP，超越所有已有 DETR 变体。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](object_detection/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)**

:   提出 RAPTA（训练时基于目标检测的区域感知 prompt 变体增强）和 ADMCD（推理时三流注意力融合的多模态复制检测），从缓解和检测两个角度端到端地应对文生图扩散模型的训练数据记忆化问题。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](object_detection/mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)**

:   提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。

**[Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](object_detection/mr_detr_instructive_multi-route_training_for_detection_transformers.md)**

:   系统研究 DETR 解码器各组件在 one-to-one/one-to-many 多任务框架下的角色，发现任何单独组件都能有效协调两个目标；基于此提出多路由训练（Instructive Self-Attention + Independent FFN + Route-Aware MoE），推理时丢弃辅助路由不增加任何开销。

**[Multiple Object Tracking as ID Prediction](object_detection/multiple_object_tracking_as_id_prediction.md)**

:   本文提出MOTIP，将多目标跟踪中的目标关联问题重新定义为in-context ID预测任务：给定携带ID嵌入的历史轨迹，直接用标准Transformer解码器预测当前检测的ID标签，无需启发式匹配算法即在DanceTrack上以69.6 HOTA大幅超越前SOTA CO-MOT (65.3)。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一种轻量级可见光-红外融合网络，通过 MEAF 模块（可学习空间掩码+空间注意力的像素级融合）和训练时结构表示增强（SR，推理时无开销的超分辅助监督），在 VEDAI 上达到 84.71% mAP 同时参数量仅 5.1M（减少 93.6%）。

---

## 🤖 具身智能 { #robotics }

**[3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](robotics/3d-mvp_3d_multiview_pretraining_for_manipulation.md)**

:   提出3D-MVP，将Masked Autoencoder预训练从2D扩展到3D多视角设定——在Objaverse的200K个3D物体上预训练RVT的多视角Transformer编码器，下游微调后在RLBench上平均成功率从62.9%提升到67.5%，在COLOSSEUM上显著提升对纹理、大小、光照等环境变化的鲁棒性。

**[A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](robotics/a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)**

:   通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。

**[ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection](robotics/asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)**

**[Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs](robotics/chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)**

**[DRAWER: Digital Reconstruction and Articulation with Environment Realism](robotics/drawer_digital_reconstruction_and_articulation_with_environment_realism.md)**

:   提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](robotics/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   提出 Expert Pyramid Tuning (EPT)，将计算机视觉中的多尺度特征金字塔思想引入 LoRA-based MoE，通过共享元知识子空间 + 反卷积金字塔投影机制构建不同粒度的专家，实现更高效的多任务参数微调。

**[Foundations of the Theory of Performance-Based Ranking](robotics/foundations_of_the_theory_of_performance-based_ranking.md)**

:   建立基于概率测度和序理论的性能排名公理化框架，用 6 大支柱（性能测度/偏序/满意度/评价函数/分数/重要性）统一了准确率/精度/召回率/F1 等指标，证明了何种评分函数适合用于排名、何种不适合。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[Instruction-based Image Manipulation by Watching How Things Move](robotics/instruction-based_image_manipulation_by_watching_how_things_move.md)**

:   本文提出 InstructMove，通过从视频中采样帧对并用多模态大模型生成编辑指令来构建大规模真实图像编辑数据集，结合空间条件化策略微调 T2I 模型，在姿态调整、视角变换等非刚性编辑任务上实现了 SOTA 效果。

**[LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA，将 7-DoF 机器人动作解耦为平移/旋转/夹爪三类运动原语并与语言语义建立对应，通过软标签对比学习和自适应损失加权，以 1.3B 参数在 LIBERO 上达到 93.6% 平均成功率。

**[Let Humanoids Hike! Integrative Skill Development on Complex Trails](robotics/let_humanoids_hike_integrative_skill_development_on_complex_trails.md)**

:   提出 LEGO-H 框架，通过 TC-ViT（时序条件 ViT）统一导航感知和低层运动控制，结合层次潜空间匹配（HLM）从 oracle 策略高效蒸馏，使 Unitree H1 人形机器人在复杂户外山径上达到 68.4% 成功率。

**[Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](robotics/lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)**

:   Lift3D提出了一个两阶段框架，先通过任务感知MAE重建深度信息增强2D基础模型的隐式3D感知能力，再通过将3D点云投影到虚拟平面建立与2D位置嵌入的映射关系来直接让2D模型编码点云数据，在MetaWorld上平均成功率达83.9%（超越前SOTA DP3的65.3%达18.6个百分点）。

**[Magma: A Foundation Model for Multimodal AI Agents](robotics/magma_a_foundation_model_for_multimodal_ai_agents.md)**

:   Magma 通过在图像上标注可交互区域（Set-of-Mark）和在视频中标注运动轨迹（Trace-of-Mark），将 UI 截图、机器人数据和人类操作视频统一到同一个预训练框架中，使单一模型同时具备多模态理解和跨域动作预测能力，在 UI 导航和机器人操控上均取得 SOTA。

**[Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](robotics/mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)**

:   提出 HR-Align 适配范式，利用配对人-机器人视频数据和对比对齐损失，以参数高效的方式弥合人类数据预训练模型与机器人域之间的语义差距，在 20 个仿真任务和 5 个真实任务上平均成功率提升 7%+。

**[MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](robotics/momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)**

:   提出 MoManipVLA，将预训练的固定基座 VLA 模型迁移到移动操作场景，通过双层轨迹优化联合规划底盘移动和机械臂轨迹（优化可达性/平滑性/碰撞避免），在 OVMM 基准上达到 66.1% 成功率（+4.2%），仅需 50 条演示即可在真实世界部署。

**[One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](robotics/one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

**[Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](robotics/phoenix_a_motion-based_self-reflection_framework_for_fine-grained_robotic_action.md)**

:   提出 Phoenix 框架，用运动指令作为桥梁连接 MLLM 的高层语义反思和底层机器人动作纠正，通过双过程运动调整机制+运动条件扩散策略实现精细粒度的操作失败恢复，并支持终身学习自我提升。

**[Prof. Robot: Differentiable Robot Rendering without Static and Self-Collisions](robotics/prof_robot_differentiable_robot_rendering_without_static_and_self-collisions.md)**

:   提出 Prof. Robot，首个结合碰撞约束的可微机器人渲染框架——将 3D 高斯点绑定到机器人 URDF 模型的各连杆上实现可微渲染，同时在优化中加入静态碰撞（与环境）和自碰撞（机器人自身）约束，将碰撞率从 24% 降至 0%，同时保持视觉保真度。

**[RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](robotics/roboground_robotic_manipulation_with_grounded_vision-language_priors.md)**

:   提出 RoboGround，一个两阶段框架：先用 Grounded VLM（GLaMM）从图像和文本指令中生成目标物体和放置区域的分割掩码，再通过 Grounded Perceiver 将掩码作为中间表示引导机器人策略网络执行操作，在复杂语义操作任务上实现 60-100% 的相对提升。

**[SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

**[SortScrews: A Dataset and Baseline for Real-time Screw Classification](robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)**

:   提出SortScrews数据集——一个包含560张512×512 RGB图像、覆盖6类螺丝的工业分类数据集，配套可复用的数据采集流水线，并以迁移学习的EfficientNet-B0和ResNet-18作为基线，ResNet-18在该数据集上达到96.4%验证准确率。

**[Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation](robotics/think_small_act_big_primitive_prompt_learning_for_lifelong_robot_manipulation.md)**

:   提出 Primitive Prompt Learning (PPL)，通过将运动原语编码为可复用的提示向量，结合光流感知的 Motion-Aware Prompting（MAP）实现跨技能运动原语共享，用冻结-扩展机制支持终身机器人操作学习，在 LIBERO 和真实世界中均优于 LoRA、经验回放等基线。

**[TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)**

:   在 ESP32 微控制器上部署端到端量化 CNN，仅用 23k 参数和 ToF 深度相机实现 30ms 延迟的实时自主导航。

**[Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](robotics/towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)**

:   定义长程视觉语言导航（LH-VLN）任务，构建 NavGen 自动生成平台和 LHPR-VLN 基准（3260 个多阶段任务，平均 150 步），提出 MGDM 方法通过短期记忆模糊+长期记忆检索+CoT反馈实现多阶段导航，在 ISR 指标上超越 NaviLLM 23%。

---

## 🖼️ 图像恢复 { #image_restoration }

**[A Flag Decomposition for Hierarchical Datasets](image_restoration/a_flag_decomposition_for_hierarchical_datasets.md)**

:   提出Flag Decomposition（FD）——一种保持层次结构的矩阵分解方法，将具有嵌套列层次的数据矩阵分解为Stiefel坐标表示的flag（嵌套子空间序列）、块上三角矩阵和置换矩阵，在去噪、聚类和小样本学习任务上优于SVD等标准方法。

**[A Physics-Informed Blur Learning Framework for Imaging Systems](image_restoration/a_physics-informed_blur_learning_framework_for_imaging_systems.md)**

:   提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。

**[EQ-Reg: A Regularization-Guided Equivariant Approach for Image Restoration](image_restoration/a_regularization-guided_equivariant_approach_for_image_restoration.md)**

**[AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)**

:   提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。

**[Augmenting Perceptual Super-Resolution via Image Quality Predictors](image_restoration/augmenting_perceptual_super-resolution_via_image_quality_predictors.md)**

:   利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。

**[Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)**

:   重新审视经典视频去噪方法并与现代ML工具结合，实现鲁棒、快速且噪声级别可控的视频去噪

**[Complexity Experts are Task-Discriminative Learners for Any Image Restoration](image_restoration/complexity_experts_are_task-discriminative_learners_for_any_image_restoration.md)**

:   提出 MoCE-IR，用具有不同计算复杂度和感受野大小的"复杂度专家"替代传统均匀 MoE 的统一架构，配合偏向低复杂度的弹簧式路由机制，意外地实现了任务判别性分配——不同退化类型自动路由到适当复杂度的专家，可在推理时跳过无关专家。

**[DarkIR: Robust Low-Light Image Restoration](image_restoration/darkir_robust_low-light_image_restoration.md)**

:   提出 DarkIR，一种面向低光照多任务修复（去噪+去模糊+增强）的 CNN 端到端框架，编码器用频域注意力 FreMLP 纠正光照、解码器用膨胀空间注意力 Di-SpAM 去模糊，以 Restormer 1/5 参数和 1/5 计算量超越其 PSNR。

**[Degradation-Aware Feature Perturbation for All-in-One Image Restoration](image_restoration/degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)**

:   提出 DFPIR，通过退化引导的特征扰动——通道级扰动（退化引导的通道混洗在高维空间中重组特征）和注意力级扰动（top-K 掩码交叉注意力选择性传递信息）——将特征空间对齐到统一的参数空间，在 3/5 任务全合一修复中平均 PSNR 超越 InstructIR +0.45/+1.09 dB。

**[Detail-Preserving Latent Diffusion for Stable Shadow Removal](image_restoration/detail-preserving_latent_diffusion_for_stable_shadow_removal.md)**

:   提出两阶段阴影去除框架，Stage 1 微调 Stable Diffusion 去噪器以 z₀-prediction 方式生成粗阴影去除结果（低方差高稳定性），Stage 2 用 Detail Injection 模块通过 RRDB 块选择性地将阴影感知高频细节注入冻结 VAE 解码器，在四个基准上达到无掩码方法最优。

**[DiffFNO: Diffusion Fourier Neural Operator](image_restoration/difffno_diffusion_fourier_neural_operator.md)**

:   提出 DiffFNO，将加权傅里叶神经算子（WFNO，通过模式重平衡保留高频傅里叶模式）与扩散框架结合，配合门控融合机制和自适应时间步 ODE 求解器，实现任意尺度图像超分辨率，在 DIV2K 多尺度上比 SRNO/LMI 提升 0.3-1.0 dB PSNR。

**[EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation](image_restoration/echomimicv2_towards_striking_simplified_and_semi-body_human_animation.md)**

:   提出 Audio-Pose Dynamic Harmonization（APDH）策略渐进式将控制权从全身姿态转移到音频——逐步移除关键点（保留手部）同时扩大音频控制范围（从唇部到全身），实现仅需音频+参考图+手部姿态的高质量半身动画。

**[Efficient Diffusion as Low Light Enhancer (ReDDiT)](image_restoration/efficient_diffusion_as_low_light_enhancer.md)**

:   提出 ReDDiT 将扩散式低光增强从 10+ 步蒸馏到 2-4 步——通过线性外推修正拟合误差、用 Retinex 分解的反射率做轨迹精炼弥合推理间隙，4 步即在 10 个基准上全面达到 SOTA。

**[Efficient Visual State Space Model for Image Deblurring](image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)**

:   本文提出一种高效视觉状态空间模型 EVSSM，通过几何变换替代多方向扫描策略捕获非局部信息，并设计高效频域前馈网络增强局部细节，在图像去模糊任务上以仅四分之一的计算代价超越现有 SSM 方法，达到 SOTA 效果。

**[FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems](image_restoration/fire_fixed-points_of_restoration_priors_for_solving_inverse_problems.md)**

:   本文提出 FiRe 框架，通过将通用图像恢复模型（去模糊、超分、修复等）与其训练时的退化算子复合，利用不动点理论推导出显式先验公式，扩展了传统 PnP 中仅限去噪先验的范围，并支持多恢复模型的集成，在多种逆问题上显著超越现有 PnP 和扩散方法。

**[Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise](image_restoration/generalized_recorrupted-to-recorrupted_self-supervised_learning_beyond_gaussian_.md)**

:   本文提出Generalized R2R (GR2R)，将原始R2R自监督去噪框架从高斯噪声推广到自然指数族（NEF）分布——包括Poisson/Gamma/Binomial噪声，证明GR2R损失是有监督损失的无偏估计，并且SURE可视为其特例，在低光成像和SAR等应用中达到接近监督学习的性能。

**[Gyro-based Neural Single Image Deblurring](image_restoration/gyro-based_neural_single_image_deblurring.md)**

:   提出 GyroDeblurNet，通过新颖的相机运动场嵌入表示复杂手抖、陀螺仪细化模块利用图像模糊信息校正陀螺仪误差、陀螺仪去模糊模块用校正后的运动信息去除模糊，配合课程学习策略，在合成和真实数据集上大幅超越现有方法。

**[INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations](image_restoration/infp_audio-driven_interactive_head_generation_in_dyadic_conversations.md)**

:   INFP 提出了一个统一的音频驱动交互式头部生成框架，通过双轨音频（agent + 对话伙伴）驱动 agent 在说话和倾听状态间自然切换，无需手动角色分配或显式角色切换，同时引入大规模 DyConv 数据集支持研究。

**[Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](image_restoration/iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)**

:   IPC-Dehaze 提出了一种基于 VQGAN 码本先验的迭代式 Predictor-Critic 解码框架，通过 Code-Critic 评估码本序列间的相互关联来决定哪些码应保留或重采样，实现了从清晰区域到密集雾区的由易到难渐进去雾，在真实场景中显著超越 SOTA。

**[MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](image_restoration/mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)**

:   提出 MaIR，核心创新是嵌套 S 形扫描策略（NSS）通过条带划分保持局部性 + S 形路径保持连续性，以及序列洗牌注意力（SSA）通过通道级注意力智能聚合不同扫描方向的序列，在超分、去噪、去模糊、去雾 4 大任务 14 个数据集上达到 SOTA。

**[MambaIRv2: Attentive State Space Restoration](image_restoration/mambairv2_attentive_state_space_restoration.md)**

:   提出 MambaIRv2，通过 Attentive State-space Equation（ASE）在 Mamba 的输出矩阵 $\mathbf{C}$ 中注入可学习 prompt 实现类似注意力的非因果全局查询，并用 Semantic Guided Neighboring（SGN）按语义标签重排序列缓解长距离衰减，仅需单方向扫描即超越多方向方法，轻量 SR 上以 9.3% 更少参数超 SRFormer 0.35dB。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery](image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+/++，通过**分块训练+拼接推理**和**arcsinh 非线性变换**两项改进，使深度学习方法首次能处理宽视场（12,960×12,960 像素）、高动态范围（~10⁶）的射电干涉成像，并展示了超分辨率对强引力透镜发现的 10× 提升潜力。

**[OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras](image_restoration/towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)**

:   扩展 OptiFusion 自动设计 120 种多样化镜头，提出 ODE 综合评估指标和大规模 benchmark，系统对比 24 种算法，发现 CNN 模型在像差校正中提供最佳速度-精度权衡，反直觉地超越 Transformer。

**[Variational Garrote for Sparse Inverse Problems](image_restoration/variational_garrote_for_sparse_inverse_problems.md)**

:   系统比较 $\ell_1$ 正则化 (LASSO) 与 Variational Garrote (VG, 概率 $\ell_0$ 近似) 在信号重采样、去噪和稀疏视角 CT 重建三种逆问题上的表现，发现 VG 在强欠定情况下（采样率低/角度稀疏）通常获得更低的泛化误差，因为 spike-and-slab 先验与真实稀疏分布更匹配。

---

## 🛡️ AI安全 { #ai_safety }

**[A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](ai_safety/a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)**

:   提出FedRDN——一种极其简单的联邦学习数据增强方法，在训练时随机使用其他客户端的通道级均值/标准差做数据归一化（而非固定用本地统计），仅需几行代码即可显著缓解特征分布偏移问题，在多种FL方法上一致提升性能。

**[Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior](ai_safety/data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior.md)**

:   提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。

**[DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging](ai_safety/deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)**

:   提出 DEAL（Data-Efficient Adversarial Learning），一种仅需 50 张清晰红外图像训练的对抗学习框架，通过动态对抗退化合成和双通道交互网络（Scale Transform + Spiking Neurons），以 0.96M 超轻量参数同时处理条纹噪声、低分辨率和低对比度三种红外退化。

**[DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](ai_safety/dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)**

**[Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)**

:   提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。

**[Dynamic Integration of Task-Specific Adapters for Class Incremental Learning](ai_safety/dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)**

:   通过动态集成任务特定适配器实现类增量学习，每个任务训练轻量适配器，推理时动态选择和组合相关适配器

**[ESC: Erasing Space Concept for Knowledge Deletion](ai_safety/esc_erasing_space_concept_for_knowledge_deletion.md)**

:   提出 ESC（Erasing Space Concept），通过 SVD 分解待遗忘数据的特征空间并移除主成分方向，实现训练无关的特征级知识删除，首次定义了"知识删除"（Knowledge Deletion）任务并提出 Knowledge Retention Score 评估特征级遗忘效果。

**[FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](ai_safety/fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_.md)**

:   提出 FedAWA，受任务算术（task arithmetic）启发，用客户端向量（本地参数与全局参数的差值）来自适应优化联邦学习中的聚合权重——与全局优化方向一致的客户端获得更高权重，在 non-IID 场景下稳定提升 FedAvg 1-4 个点。

**[Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](ai_safety/geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)**

:   在联邦学习中通过从局部协方差矩阵精确重建全局协方差来获取全局嵌入分布的几何形状，沿全局主方向生成增强样本本地化全局分布信息，在 CIFAR-100 极端异质场景（β=0.01）下提升 17 个百分点。

**[Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning](ai_safety/gradient_inversion_attacks_on_parameter-efficient_fine-tuning.md)**

:   首次证明 Adapter-based PEFT 在联邦学习中不是隐私安全的——恶意服务器可以将预训练模型设计为恒等映射使 patch embedding 原样传播到 adapter 层，从 adapter 梯度中解析式恢复训练图像（CIFAR-100 SSIM 0.88）。

**[Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)**

:   本文首次研究了联邦学习中非合作多标签后门攻击(MBA)场景，揭示了现有单标签后门攻击方法扩展到多标签场景时因构建相似的分布外(OOD)映射而导致攻击者间相互排斥的内在缺陷，提出 Mirage 方法通过构建分布内(ID)后门映射，使多个攻击者可以独立且持久地植入后门，平均攻击成功率超过97%且在900轮后仍保持90%以上。

**[INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](ai_safety/invisible_backdoor_attack_against_self-supervised_learning.md)**

:   提出 INACTIVE，首个对自监督学习（SSL）有效的不可见后门攻击——通过在 HSV/HSL 色彩空间中设计触发器以逃离 SSL 数据增强的分布空间，实现 99.09% 平均攻击成功率，同时保持 SSIM 0.9763/PSNR 41.07dB 的高隐蔽性，抵抗 7 种防御方法。

**[Lyapunov Stable Graph Neural Flow](ai_safety/lyapunov_stable_graph_neural_flow.md)**

:   将 Lyapunov 稳定性理论（整数阶和分数阶）与图神经流集成，通过可学习 Lyapunov 函数和投影机制将 GNN 特征动态约束在稳定空间中，首次为图神经流提供可证明的对抗鲁棒性保证，且与对抗训练正交可叠加。

**[Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](ai_safety/mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)**

:   本文提出了一种基于查询更新模式(而非输入模式)的黑盒对抗攻击检测框架 GWAD，引入 Delta Similarity 指标来捕获基于查询的攻击中零阶优化的固有模式，在8种SOTA攻击(包括自适应攻击OARS)上实现了接近100%的检测率且误报率极低，显著优于现有的状态化防御方法。

**[MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](ai_safety/mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)**

:   提出MOS Attack框架，将对抗攻击建模为多目标集合优化问题，结合smooth max/min近似实现多损失函数联合优化，并自动发现损失函数间的协同模式，在CIFAR-10和ImageNet上超越现有SOTA单目标攻击和集成攻击。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](ai_safety/neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[NoT: Federated Unlearning via Weight Negation](ai_safety/not_federated_unlearning_via_weight_negation.md)**

:   提出 NoT（Not），通过将选定层的权重取反（乘 -1）再在保留数据上微调来实现联邦遗忘，无需访问遗忘数据或存储额外信息，在 CIFAR-10 CNN 上与从头重训的差距仅 0.29%，计算量减少约一半。

**[Rethinking VLMs for Image Forgery Detection and Localization](ai_safety/rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM，揭示 VLM 先验对伪造检测/定位几乎无益，通过将检测/定位与语言解释解耦的两阶段框架，用 ViT+SAM 专家模型做检测定位、再将定位 mask 作为辅助输入增强 VLM 训练以生成可解释文字说明。

**[Split Adaptation for Pre-trained Vision Transformers](ai_safety/split_adaptation_for_pre-trained_vision_transformers.md)**

:   提出 Split Adaptation，将预训练 ViT 分为客户端前端（量化+噪声）和服务端后端，通过 OOD 增强量化、Hilbert 变换增强和 patch 级检索增强实现少样本适应，在保护数据隐私和模型知识产权的同时在 CIFAR-100 5-shot 上达到 81.98%，计算量仅为线性探测的 4%。

---

## 🔄 自监督 { #self_supervised }

**[AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)**

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[CheXWorld: Image World Modeling for Radiograph Representation Learning](self_supervised/chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)**

**[Do Your Best and Get Enough Rest for Continual Learning](self_supervised/do_your_best_and_get_enough_rest_for_continual_learning.md)**

:   受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

**[Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](self_supervised/escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)**

**[Few-Shot Implicit Function Generation via Equivariance](self_supervised/few-shot_implicit_function_generation_via_equivariance.md)**

:   通过等变性约束从少量样本生成隐式函数（NeRF/SDF），利用对称性先验减少对数据的需求

**[From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](self_supervised/from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)**

:   提出原型驱动的 MAE 课程学习——用 K-means 聚类识别数据集中的"原型"样本（靠近聚类中心的代表性图像），通过温度控制的采样策略从原型逐步过渡到全分布训练，实现 8× 训练加速（200 epoch 原型课程 ≈ 800 epoch 标准 MAE）。

**[Hyperbolic Category Discovery](self_supervised/hyperbolic_category_discovery.md)**

:   提出HypCD框架，将广义类别发现（GCD）中的表示学习从欧氏/球面空间迁移到双曲空间（Poincaré球模型），利用双曲空间指数级体积增长天然适合编码层次结构的特性，通过距离-角度混合相似度学习和双曲分类器，在CUB上将SelEx从69.1%提升到71.8%，在ImageNet-100上从87.1%提升到88.3%。

**[Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry](self_supervised/learning_to_normalize_on_the_spd_manifold_under_bures-wasserstein_geometry.md)**

:   提出 GBWBN，在 SPD（对称正定）流形上用广义 Bures-Wasserstein 距离实现黎曼批归一化，通过可学习度量参数 M 和矩阵幂变形 θ 处理病态 SPD 矩阵问题，在骨架动作识别/脑电分类/无人机识别等 SPD 学习任务上一致提升。

**[MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)**

:   提出 Masked Autoregressive Pretraining（MAP），通过局部 MAE 建模 + 行级自回归解码的层次化预训练目标，首次有效预训练混合 Mamba-Transformer 视觉骨干，显著超越 MAE 和 AR 单一策略。

**[MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](self_supervised/metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)**

:   MetaWriter 将手写文字识别的个性化适配形式化为 prompt tuning 问题，结合 MAE 自监督辅助任务实现无标签测试时适应，并用元学习优化 prompt 初始化使自监督损失与识别损失对齐，仅更新不到1%参数即在IAM和RIMES上达到SOTA。

**[Representation Learning for Spatiotemporal Physical Systems](self_supervised/representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](self_supervised/smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)**

:   提出 SMILE，通过合成运动增强（在视频上叠加沿随机轨迹运动的分割物体）和 CLIP 特征重建目标来增强掩码视频建模，结合轨迹引导的掩码策略，在 K400 线性探测上大幅提升至 56.2%（前 SOTA 47.5%）。

**[Spectral State Space Model for Rotation-Invariant Visual Representation Learning](self_supervised/spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)**

:   提出 Spectral VMamba，用谱图拉普拉斯的特征向量排序 patch 遍历顺序（替代预定义扫描线），结合旋转特征归一化器（RFN，聚合 4 个正则旋转的特征），在 miniImageNet 上达到 87.86% 准确率且对正则旋转完全不变。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。

**[Transformers without Normalization](self_supervised/transformers_without_normalization.md)**

:   发现 LayerNorm 的输入-输出映射呈 tanh 形状，提出 Dynamic Tanh (DyT) 作为归一化层的即插即用替代：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，在视觉/语言/扩散/语音等多任务中与 LN 性能持平甚至更优。

---

## 🎵 音频/语音 { #audio_speech }

**[Contextual AD Narration with Interleaved Multimodal Sequence](audio_speech/contextual_ad_narration_with_interleaved_multimodal_sequence.md)**

:   提出 Uni-AD 统一框架，以交错多模态序列（视频特征+文本+角色库+上下文）作为输入，通过视觉映射网络对齐特征 + 角色精化模块识别主要角色 + 对比损失增强上下文一致性，在 MAD-eval-Named 上达到 SOTA。

**[Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](audio_speech/crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)**

:   提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

**[DistinctAD: Distinctive Audio Description Generation in Contexts](audio_speech/distinctad_distinctive_audio_description_generation_in_contexts.md)**

:   生成上下文中有区分度的音频描述（AD），避免生成泛化无特色的描述，通过对比学习鼓励与前后AD的差异性

**[EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions](audio_speech/emova_empowering_language_models_to_see_hear_and_speak_with_vivid_emotions.md)**

:   提出 EMoVA，首个端到端的全模态 LLM，通过语义-声学解耦的语音 tokenizer 同时实现视觉理解、语音识别和情感可控的语音合成，在视觉语言基准上超越 GPT-4o，语音识别 WER 达 2.9%。

**[Exploring Timeline Control for Facial Motion Generation](audio_speech/exploring_timeline_control_for_facial_motion_generation.md)**

:   本文首次提出面部动作生成的时间线控制方式——用户指定多轨道时间轴上各面部动作的精确帧区间，通过TICC时序聚类实现省力的帧级面部动作标注，并设计base-branch扩散模型在解耦各面部区域的同时保留自然耦合，生成精确对齐时间线且自然流畅的面部动作。

**[Improving Sound Source Localization with Joint Slot Attention on Image and Audio](audio_speech/improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)**

:   提出联合槽注意力机制将图像和音频同时分解为目标/非目标表示，通过跨模态注意力匹配和对比学习实现精确声源定位，在 Flickr-SoundNet 上达到 65.16% AUC、86.00% cIoU SOTA。

**[ImViD: Immersive Volumetric Videos for Enhanced VR Engagement](audio_speech/imvid_immersive_volumetric_videos_for_enhanced_vr_engagement.md)**

:   构建首个沉浸式体积视频数据集——用 46 台同步 GoPro 的移动多视角系统拍摄 7 个场景（含室内/室外），提出 STG++ 增加可学习仿射颜色变换解决跨相机颜色不一致，实现 110.47 FPS 渲染/387MB 存储，并集成 HRTF 空间音频。

**[Learning to Highlight Audio by Watching Movies](audio_speech/learning_to_highlight_audio_by_watching_movies.md)**

:   提出视觉引导的声学高亮任务(visually-guided acoustic highlighting)，利用电影中精心制作的音视频数据作为免费监督，通过基于Transformer的多模态框架VisAH，将"混音不佳"的音频转换为视觉语义对齐的高亮音频，在所有指标上显著超越基线方法。

**[LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](audio_speech/livecc_learning_video_llm_with_streaming_speech_transcription_at_scale.md)**

:   提出 LiveCC，通过将 ASR 转录词与视频帧沿时间轴密集交织训练视频 LLM，构建了 Live-CC-5M 预训练数据集，使 7B 模型在实时视频解说任务上超越 72B 模型（包括 Qwen2.5-VL-72B）。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](audio_speech/team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   本文提出面向视频级矛盾/犹豫（A/H）识别的多模态方法，整合场景（VideoMAE）、面部（EmotionEfficientNetB0）、音频（EmotionWav2Vec2.0+Mamba）和文本（EmotionDistilRoBERTa）四种模态，通过原型增强的 Transformer 融合模型实现 83.25% 平均 MF1，最终以五模型集成在测试集达到 71.43%。

**[Towards Lossless Implicit Neural Representation via Bit Plane Decomposition](audio_speech/towards_lossless_implicit_neural_representation_via_bit_plane_decomposition.md)**

:   发现隐式神经表示（INR）的模型容量上界随比特精度指数增长（$\mathcal{P}(f_\theta) \propto 2^n$），提出比特平面分解——将 n-bit 信号分解为 n 个独立的 1-bit 平面分别训练 INR，首次实现 16-bit 图像的无损（BER=0）隐式神经表示。

**[Towards Open-Vocabulary Audio-Visual Event Localization](audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)**

:   首次定义开放词汇音视频事件定位（OV-AVEL）任务，构建了包含 24800 个视频、67 类事件的 OV-AVEBench 基准，并提出基于 ImageBind 的训练免和微调两种基线方法，其中仅用 1 层时序 Transformer 微调即达 57.8% 平均性能。

**[UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)**

:   提出 UWAV，一个弱监督音视频视频解析框架，通过在大规模标注数据上预训练时序感知模块生成高质量伪标签，再用不确定性加权软标签+类别平衡重加权+特征混合三种技术提升弱监督训练效果，在 LLP 数据集上刷新 SOTA。

**[MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](audio_speech/video-guided_foley_sound_generation_with_multimodal_controls.md)**

:   提出 MultiFoley，基于 Diffusion Transformer 的视频引导 Foley 音效生成系统，支持文本语义控制和参考音频风格控制，通过联合训练视频-音频和文本-音频数据集实现 48kHz 高质量音频生成，在人类评估中以 90% 胜率碾压现有方法。

**[VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](audio_speech/vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)**

:   提出 VinTAGe，首个联合视频+文本条件的音频生成模型，通过可学习层权重平衡视觉/文本引导，用教师-学生框架缓解模态偏置，在画内音和画外音生成上实现全面最优（FAD 3.05，MOS 3.36）。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](llm_alignment/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)**

:   本文提出 Step-by-step Preference Optimization（SPO），在每个去噪步中从同一噪声潜变量采样多个候选，用 step-aware 偏好模型选择 win/lose 对来指导扩散模型微调，从通用偏好数据中隐式蒸馏美学信息，在 SD-1.5 和 SDXL 上显著提升美学质量且收敛速度远快于 DPO。

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](llm_alignment/bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种求解可转向等变 CNN 核约束方程的替代方法，通过在不动点处求解更简单的不变性条件再"转向"到任意点，绕过了计算 Clebsch-Gordan 系数的需要，为 SO(2)、O(2)、SO(3)、O(3) 及 Lorentz 群给出了显式的核基底公式。

**[Boost Your Human Image Generation Model via Direct Preference Optimization](llm_alignment/boost_your_human_image_generation_model_via_direct_preference_optimization.md)**

:   提出 HG-DPO，以真实人像作为 DPO 的 winning image（而非生成图像对）+ 三阶段课程学习（Easy/Normal/Hard）渐进弥合生成-真实图像分布 gap + 统计匹配损失解决色偏，FID 从 37.34 降至 29.41（-21.4%），CI-Q 0.906→0.934，win-rate 超越 Diffusion-DPO 达 99.97%。

**[CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation](llm_alignment/cad-llama_leveraging_large_language_models_for_computer-aided_design_parametric_.md)**

:   本文提出 CAD-Llama 框架，通过层次化标注管线将 3D CAD 模型转化为富含语义描述的 Python 风格代码（SPCC），再用自适应预训练和指令微调将 LLaMA3-8B 转化为参数化 CAD 模型生成器，在 text-to-CAD 任务上精度超出先前方法约 14%，并支持补全、添加、删除等多种 CAD 编辑任务。

**[Calibrated Multi-Preference Optimization for Aligning Diffusion Models](llm_alignment/calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)**

:   本文提出 Calibrated Preference Optimization（CaPO），通过 win-rate 校准将不同奖励模型的分数统一为期望胜率，并设计基于 Pareto 前沿的配对采样策略（FRS）来处理多奖励信号间的冲突，在 SDXL 和 SD3-Medium 上一致地超越 DPO 和 IPO 方法。

**[Continual SFT Matches Multimodal RLHF with Negative Supervision](llm_alignment/continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)**

:   通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

**[Curriculum Direct Preference Optimization for Diffusion and Consistency Models](llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)**

:   首次将课程学习引入 DPO 并首次将 DPO 适配到一致性模型，通过从"容易区分的偏好对"到"难以区分的偏好对"渐进训练，在文本对齐、美学和人类偏好上全面超越标准 DPO 和 DDPO，且仅需 1/10 训练数据量。

**[Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](llm_alignment/debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)**

:   NaPO 针对MLLM的模态偏差问题（过度依赖语言先验或视觉细节），通过mask模态信息构造偏差数据集RLAIF-V-Bias，并提出基于负Box-Cox变换的噪声感知偏好优化算法，在自动构造的含噪数据上实现鲁棒训练，在去偏和减幻觉上均取得显著效果。

**[Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal Large Language Models?](llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)**

:   本文通过系统分析揭示MLLM的安全对齐缺口主要源于微调数据的分布偏差（而非图像内容、回复质量或对比行为），仅需用少量良性指令跟随数据（将回复替换为简单拒绝句）即可显著提升5种架构MLLM的安全性，无需费力收集恶意数据。

**[Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation](llm_alignment/enhancing_sam_with_efficient_prompting_and_preference_optimization_for_semi-supe.md)**

:   提出一种增强 SAM 的半监督医学图像分割框架：通过 CLIP 和 VQA 无监督生成包含语义、位置和形状信息的高效提示（无需专家标注），再用 DPO 偏好优化技术配合虚拟标注器（代替人类标注者提供排名/评分）训练最优分割策略，在肺分割、乳腺肿瘤分割、器官分割等多模态任务上达到 SOTA。

**[InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](llm_alignment/inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)**

:   提出 InPO（Inversion Preference Optimization），通过 DDIM 反演的重参数化技巧将偏好优化从需要完整去噪链的长马尔可夫过程简化为单步优化，在训练效率和生成质量上同时优于现有 Diffusion-DPO 方法。

**[Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](llm_alignment/jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)**

:   提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](llm_alignment/physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)**

:   提出 PhysMoDPO，将 Direct Preference Optimization 应用于文本驱动的人体运动生成，通过将全身控制器（WBC）集成到训练 pipeline 中计算基于物理的奖励来构造偏好数据，使生成运动同时满足物理约束和文本指令，并在 Unitree G1 机器人上实现零样本部署。

**[SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](llm_alignment/symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)**

:   SymDPO 发现LMM在多模态ICL中存在"视觉上下文忽视"问题（用空白图替换示例图不影响性能），提出将示例中的文本答案替换为无语义随机符号，迫使模型必须理解视觉内容才能正确匹配符号与答案，通过DPO训练在OpenFlamingo和IDEFICS上一致提升了多模态ICL效果。

**[Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](llm_alignment/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)**

:   提出 Task Preference Optimization（TPO），通过可学习的任务 token 将视觉任务专用头（区域定位/时序定位/分割）接入 MLLM，利用视觉任务标注作为"任务偏好"反向优化 MLLM，在不损害对话能力的前提下大幅提升细粒度视觉理解，VideoChat 基线上平均提升 14.6%。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](information_retrieval/advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)**

:   将CLIP从传统的一对一(image, text)对比学习升级为多对多(multi-image-embeddings, multi-texts)对比学习范式，通过VLM生成多视角多层次的描述文本、多分支视觉编码器输出多种视觉embedding，实现更全面的视觉语言对齐，在检索/分类/密集任务上大幅超越baseline。

**[ChatHuman: Chatting about 3D Humans with Tools](information_retrieval/chathuman_chatting_about_3d_humans_with_tools.md)**

:   提出 ChatHuman，一个基于 LLM 的语言驱动系统，通过自动选择和集成专门的 3D 人体分析工具（3D 姿态估计、形状恢复、接触检测、人物交互分析、情感识别等），利用学术论文作为工具使用说明和 RAG（检索增强生成）创建 in-context 示例以管理新工具，在工具选择准确率和整体 3D 人体任务性能上超越现有 LLM 模型。

**[DocoPilot: Improving Multimodal Models for Document-Level Understanding](information_retrieval/docopilot_improving_multimodal_models_for_document-level_understanding.md)**

:   本文构建了 Doc-750K——一个包含 758K 问答对和 3.1M 图像的高质量文档级多模态数据集，并基于此训练原生文档理解模型 Docopilot，在 MM-NIAH 上超越 InternVL2-8B 达 19.9 个百分点，无需 RAG 即可高效处理多页文档。

**[EZSR: Event-based Zero-Shot Recognition](information_retrieval/ezsr_event-based_zero-shot_recognition.md)**

:   提出 EZSR，通过将事件相机数据与 CLIP 的 RGB 嵌入空间对齐，实现基于事件数据的零样本物体识别。

**[Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](information_retrieval/few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)**

:   本文首次将检索增强学习（RAL）扩展到少样本识别（FSR），揭示了检索数据的分布不平衡和域差距两大挑战，提出两阶段方法 SWAT（先在混合数据上微调视觉编码器、再在少量标注数据上重训分类器），在 9 个基准上以 >6% 的优势超越所有先前方法。

**[GENIUS: A Generative Framework for Universal Multimodal Search](information_retrieval/genius_a_generative_framework_for_universal_multimodal_search.md)**

:   首个通用生成式多模态检索框架，通过模态解耦的语义量化将多模态数据编码为离散 ID，用自回归解码器直接从查询生成目标 ID，在 Flickr30K 文本→图像检索上超越先前生成式方法 25+ 个点，存储开销比 CLIP 降低 99%。

**[GOAL: Global-Local Object Alignment Learning](information_retrieval/goal_global-local_object_alignment_learning.md)**

:   提出GOAL方法，通过局部图-句匹配（LISM）和Token相似性学习（TSL）两个模块增强CLIP对长文本描述的理解能力，在全局对齐的基础上引入局部语义对齐，大幅提升图文检索性能。

**[Joint Vision-Language Social Bias Removal for CLIP](information_retrieval/joint_vision-language_social_bias_removal_for_clip.md)**

:   本文揭示了现有CLIP去偏方法因图文偏差分布不一致导致的"过度去偏"问题，提出先对齐图文偏差再联合移除的双模态去偏框架，在多个骨干网络上显著提升ABLE综合指标，实现了偏差消除与V-L对齐能力的良好平衡。

**[LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](information_retrieval/lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)**

:   将生成式大语言模型（LMM）改造为通用多模态检索器+重排器，通过两阶段训练（语言预训练+多模态指令微调）和联合逐点/列表重排训练，仅插入轻量LoRA模块即可在16种检索任务上显著超越双编码器方法，且在10个未见数据集上展现强泛化能力。

**[NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](information_retrieval/neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)**

:   提出 NeighborRetr，通过三重机制解决跨模态检索中的 Hubness 问题（少数样本垄断近邻）：中心性加权损失（降低 hub 样本的训练权重）、邻域调整损失（区分好/坏 hub）和均匀正则化（确保每个样本被公平检索），在 MSR-VTT 文本→视频 R@1 达 49.5%（+0.9% SOTA）。

**[Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)**

:   提出 CRPL 框架，通过源域增强的伪标签和基于最优传输的聚类保持策略，改进 CLIP 在无监督域适应（UDA）中的 prompt 学习，使得目标域 prompt 的文本嵌入能更好地覆盖视觉嵌入的聚类结构。

**[Towards Smart Point-and-Shoot Photography](information_retrieval/towards_smart_point-and-shoot_photography.md)**

:   提出智能"傻瓜相机"摄影系统：先用 CLIP 文本嵌入的构图质量评估器（CCQA）判断当前构图质量，再用专家混合（MoE）相机姿态调整模型（CPAM）预测偏航/俯仰调整角度，在 PCARD 数据集（320K 图像，从 4K 全景图生成）上实现 79.3% AUC 的调整建议和 0.613 IoU 的调整精度。

**[VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](information_retrieval/vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)**

:   构建首个直接以文档图像（而非解析文本）为输入的 RAG 框架，用 LVLM 作为双编码器检索器 + 两种自监督预训练任务（对比+生成）实现文档图像检索，在 ChartQA 上比文本 RAG 高 24 个点。

**[VladVA: Discriminative Fine-tuning of LVLMs](information_retrieval/vladva_discriminative_fine-tuning_of_lvlms.md)**

:   提出VladVA框架，通过混合短/长caption数据策略、对比损失+自回归损失的联合训练、以及soft prompting+LoRA的参数高效适配，将生成式LVLM（LLaVA）转化为强判别式模型，在图文检索和组合性理解基准上大幅超越CLIP类模型和18B EVA-CLIP。

---

## 📊 LLM评测 { #llm_evaluation }

**[ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](llm_evaluation/comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)**

:   ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。

**[ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](llm_evaluation/context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)**

:   提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

**[Dense Match Summarization for Faster Two-view Estimation](llm_evaluation/dense_match_summarization_for_faster_two-view_estimation.md)**

:   通过将密集匹配聚类为代表性子集，并用9×9矩阵编码每个聚类的几何约束，在保持精度的前提下将RANSAC鲁棒估计加速10-100倍。

**[Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](llm_evaluation/do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)**

:   提出层次化频率捷径搜索方法（HFSS），首次在ImageNet-1K规模上高效发现CNN和Transformer学到的频率捷径（仅5%频率即可正确分类），揭示频率捷径在保留纹理的OOD测试中反而有益但在风格化测试（IN-R/IN-S）上有害，指出现有OOD评估框架忽视了频率捷径的影响。

**[Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](llm_evaluation/dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)**

:   提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。

**[Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)](llm_evaluation/erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways.md)**

:   本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799），在复杂真实场景中显著优于SD2-Inpaint和LaMa。

**[Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](llm_evaluation/event_ellipsometer_event-based_mueller-matrix_video_imaging.md)**

:   首个实现 30fps 视频级穆勒矩阵成像的系统——用事件相机捕捉快速旋转 QWP 产生的光强调制，将事件时间差映射到穆勒矩阵比值，通过 SVD 估计+时空传播重建物理有效的穆勒矩阵视频。

**[Gradient-Guided Annealing for Domain Generalization](llm_evaluation/gradient-guided_annealing_for_domain_generalization.md)**

:   提出GGA方法，在训练早期通过模拟退火搜索参数空间中梯度跨域对齐的点（最小化域间梯度余弦相似度的最小值），引导模型在优化初期找到域不变特征的起始点，从而在无需数据增强的情况下提升域泛化性，可与现有DG方法组合获得显著提升。

**[LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](llm_evaluation/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)**

:   提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150），且提出 RF-JSD 免重训评估指标（与 JSD Pearson 相关 0.92）。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](llm_evaluation/out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   StEvo-Bench 提出了一个评估视频世界模型"不可观测状态演化"能力的 benchmark——测试当物理过程不被观察时（相机移开/遮挡/关灯），世界模型能否继续正确推理状态变化，结果发现当前所有前沿模型（Veo 3、Sora 2 Pro 等）的任务成功率均低于 10%，揭示了"眼不见，心不在"的严重缺陷。

**[Potential Field Based Deep Metric Learning](llm_evaluation/potential_field_based_deep_metric_learning.md)**

:   提出 PFML，用物理势能场概念替代传统的 tuple mining 进行度量学习——每个样本在嵌入空间中创建连续的引力场（同类）和斥力场（异类），具有距离衰减特性（远处交互力弱），在 Cars-196 上 R@1 达 92.7%（前 SOTA 89.6%）。

**[TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](llm_evaluation/traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)**

:   提出TraF-Align框架，通过预测特征级目标轨迹来学习特征的流动路径，生成沿时间排列的采样点将当前时刻query引导至相关历史特征，解决异步多智能体感知中的时空失配问题。

---

## 🔬 可解释性 { #interpretability }

**[Differentiable Inverse Rendering with Interpretable Basis BRDFs](interpretability/differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)**

:   提出基于可解释基 BRDF 的可微逆渲染方法，将材质分解为有物理意义的基函数组合，实现可解释的材质估计

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)**

:   提出一个从基准构建、诊断到注入的完整框架，通过 3D 基础模型（VGGT）提取相机运动线索并以结构化提示注入 VideoLLM，实现无需训练的相机运动感知增强。

**[Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)**

:   本文提出一种基于非参数原型学习的可解释图像分类框架，通过对自监督ViT特征进行最优传输聚类来发现语义上不同的物体部件原型，解决了现有ProtoPNet方法中原型重复冗余的问题，同时引入了Distinctiveness和Comprehensiveness两个新指标来量化解释质量。

**[KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](interpretability/kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)**

:   KVQ 受人类视觉系统启发，将视频全局质量显式解耦为视觉显著性和局部纹理两个因素，通过 Fusion-Window Attention 提取跨区域显著性、Local Perception Constraint 增强独立区域的纹理感知，在五个 VQA benchmark 上显著超越 SOTA。

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](interpretability/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。

**[Learning on Model Weights using Tree Experts](interpretability/learning_on_model_weights_using_tree_experts.md)**

:   发现公开模型大多属于少数 Model Tree（从共同祖先微调而来），在同一 Tree 内学习权重远比跨 Tree 简单；提出 ProbeX——首个针对单隐藏层权重的轻量 probing 方法，通过 Tucker 张量分解实现参数量 30 倍压缩，并首次实现了将模型权重与文本表示对齐的零样本模型分类（89.8% 准确率）。

**[Learning Visual Composition through Improved Semantic Guidance](interpretability/learning_visual_composition_through_improved_semantic_guidance.md)**

:   本文提出通过改善训练数据的语义监督信号（使用基础模型重新生成高质量描述+使用预训练文本编码器替代从头训练）来大幅提升标准 CLIP 模型的视觉组合理解能力，在 ARO 基准上从CLIP的59%/63%提升到92%/94%，在DOCCI图像检索上从58.4%提升到94.5% recall@1，且无需任何架构改动。

**[Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](interpretability/prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)**

:   提出 Prompt-CAM，通过为预训练 ViT 注入类别特定的可学习 prompt token，利用最后一层的多头注意力图来识别和定位区分细粒度类别的关键特征（traits），实现了近乎"免费"的可解释细粒度分析。

**[Scaling Vision Pre-Training to 4K Resolution](interpretability/scaling_vision_pre-training_to_4k_resolution.md)**

:   本文提出PS3（Pre-training with Scale-Selective Scaling），通过局部区域与局部caption的对比学习代替全图对比，以近常数的计算开销将CLIP式视觉预训练扩展到4K分辨率，并结合top-down/bottom-up patch选择机制构建VILA-HD多模态大模型，在高分辨率感知任务上大幅超越GPT-4o和Qwen2.5-VL。

**[Towards Faithful Multimodal Concept Bottleneck Models](interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出 f-CBM，一个基于 CLIP 的忠实多模态 Concept Bottleneck Model 框架，通过可微分的 leakage 损失和 Kolmogorov-Arnold Network 预测头联合解决概念检测准确性和信息泄漏问题，在任务精度、概念检测和 leakage 三者间达到最优权衡。

**[Why Does It Look There? Structured Explanations for Image Classification](interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   本文提出 I2X 框架，通过在训练检查点上追踪从 GradCAM 显著性图中提取的抽象原型（prototype）的强度变化与模型置信度的对应关系，将非结构化的可解释性转化为结构化的可解释性，并利用识别出的"不确定原型"来指导微调、减少类间混淆、提升分类精度。

---

## ⚡ LLM效率 { #llm_efficiency }

**[Associative Transformer](llm_efficiency/associative_transformer.md)**

:   提出 Associative Transformer (AiT)，通过在 Transformer 中引入可学习的显式记忆模块和 Hopfield 网络进行 token 重建，以更少的参数实现优于 ViT 的分类和关系推理性能。

**[CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](llm_efficiency/care_transformer_mobile-friendly_linear_visual_transformer_via_decoupled_dual_in.md)**

:   本文提出CARE（deCoupled duAl-interactive lineaR attEntion）机制，通过非对称特征解耦策略将局部归纳偏置和长程依赖的学习过程分而治之，配合动态记忆单元和双交互模块充分利用跨特征互补性，在ImageNet-1K上以0.7/1.9 GMACs达到78.4/82.1% top-1精度，在移动端实现极低延迟。

**[Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](llm_efficiency/efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](llm_efficiency/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)**

:   提出 Diff-DML（Differentiated Deep Mutual Learning），通过差异化训练策略（DTS）和多样性保持学习目标（DPLO）两个核心设计，在保持集成模型预测多样性的同时，同时提升准确率和不确定性校准质量。

**[KAC: Kolmogorov-Arnold Classifier for Continual Learning](llm_efficiency/kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

**[Language Guided Concept Bottleneck Models for Interpretable Continual Learning](llm_efficiency/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)**

:   提出语言引导的概念瓶颈模型（Language Guided Concept Bottleneck Model），将概念瓶颈网络的可解释性与持续学习结合，通过语言模型引导的概念定义实现任务间的知识迁移和可解释的增量学习。

**[LOCORE: Image Re-ranking with Long-Context Sequence Modeling](llm_efficiency/locore_image_re-ranking_with_long-context_sequence_modeling.md)**

:   提出 LoCoRe（Long-Context Re-ranker），首次实现基于局部描述子的列表级（list-wise）图像重排序，利用 Longformer 长上下文序列模型同时处理查询图像和整个候选列表的局部描述子，通过捕获候选图像间的传递关系显著提升重排序性能。

**[Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](llm_efficiency/low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)**

:   针对全同态加密（Leveled FHE）场景下多线性算子网络的灾难性遗忘问题，提出了一种结合低秩适应（LoRA）和梯度投影记忆（GPM）机制的增量学习方法，在保障数据安全的前提下实现持续学习。

**[Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](llm_efficiency/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)**

:   提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。

**[Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](llm_efficiency/spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)**

:   本文提出 Spatial-TTT，通过测试时训练（TTT）机制将模型的部分参数（快速权重）作为紧凑非线性记忆，配合混合架构和空间预测机制，从无界视频流中持续积累和组织3D空间证据，在视频空间理解基准上达到 SOTA。

**[Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](llm_efficiency/spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)**

:   本文提出 Accurate Addition-Only Spiking Self-Attention（A²OS²A），通过融合二值、ReLU 和三值脉冲神经元的混合策略，在保持纯加法计算（无乘法）的前提下显著提升脉冲Transformer精度，ImageNet-1K 上达到 78.66%。

---

## 🦾 LLM Agent { #llm_agent }

**[ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Inpainting](llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)**

:   提出 ATA（Adaptive Transformation Agent），解决文本引导的主体位置可变背景修复任务，通过 PosAgent Block 自适应预测位移、Reverse Displacement Transform 模块和 Position Switch Embedding，在保持修复质量的同时实现主体位置的灵活调整。

**[Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields](llm_agent/feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_.md)**

:   提出 Feature4X，一个通用框架，从任意单目视频通过动态优化策略将多种 2D 视觉基础模型（SAM2、InternVideo2 等）的功能蒸馏到统一的 4D 高斯特征场中，首次实现基于 Gaussian Splatting 的视频基础模型 4D 特征提升，支持新视角下的 segment anything、几何/外观编辑和自由形式 VQA。

**[GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)**

:   提出 GUI-Xplore 数据集（312 个应用、32K+ QA 对、五层级任务）和 Xplore-Agent 框架（Action-aware GUI 建模 + GUI Transition Graph 推理），通过模拟"先探索再推理"的人类策略，在陌生应用上比 SOTA GUI Agent 提升约 10% StepSR。

**[RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression](llm_agent/rl-rc-dot_a_block-level_rl_agent_for_task-aware_video_compression.md)**

:   提出 RL-RC-DoT，一个基于强化学习的宏块级量化参数（QP）控制 agent，用于任务感知视频压缩。通过将 QP 选择建模为 RL 的顺序决策问题，agent 学习在给定码率约束下为任务相关区域分配更多码率，在车辆检测和 ROI 显著性编码两个任务上显著提升性能。关键优势在于推理时不需要运行下游任务模型，适合边缘设备部署。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](llm_agent/sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，一个基于视觉反馈的闭环 agentic 框架，通过为 VLM 设计一套功能完备的 Action API（13个原子操作覆盖物体增删、6DoF空间操作、相机控制），让 VLM 以 ReAct 范式迭代生成开放词汇的 3D 场景，在室内（偏好率61.25%）和开放域（偏好率65.00%）场景中均大幅优于 Holodeck 和 SceneWeaver。

**[Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)**

:   提出 Sketchtopia 大规模数据集（20K+ 游戏会话、263K 草图、916 名玩家）和三组件 Agent 框架（ActionDecider + DRAWBOT + GUESSBOT），在 Pictionary 场景下研究异步、目标驱动的多模态协作通信，引入 AAO/FRS/MATS 三个新评估指标。

**[SpiritSight Agent: Advanced GUI Agent with One Look](llm_agent/spiritsight_agent_advanced_gui_agent_with_one_look.md)**

:   提出 SpiritSight，一个基于视觉的端到端 GUI agent，通过 573 万样本的多层级数据集 GUI-Lasagne 和 Universal Block Parsing (UBP) 方法解决动态高分辨率输入的定位歧义，SpiritSight-8B 在 Multimodal-Mind2Web 上非候选元素设置下 Step SR 达 52.7%，全面超越所有视觉/语言/混合方法。

**[TANGO: Training-free Embodied AI Agents for Open-world Tasks](llm_agent/tango_training-free_embodied_ai_agents_for_open-world_tasks.md)**

:   提出 TANGO，通过 LLM 的程序组合能力编排两个最小化的导航基础原语（PointGoal Navigation + 记忆驱动探索策略），无需任何任务特定训练，仅用 few-shot 示例即可在 Open-Set ObjectGoal Navigation、Multi-Modal Lifelong Navigation 和 Open Embodied QA 三个不同的具身 AI 任务上达到 SOTA，体现了"最小原语集 + LLM 组合"的通用性。

**[V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](llm_agent/v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)**

:   提出 V-Stylist，一个基于 MLLM 多 agent 协作和反思的视频风格化系统，通过 Video Parser（视频分镜）、Style Parser（风格树搜索）和 Style Artist（多轮自反思渲染）三个角色协作，在复杂转场视频和开放风格描述上实现 SOTA，整体指标超越 FRESCO 6.05%。

**[Visual Agentic AI for Spatial Reasoning with a Dynamic API](llm_agent/visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)**

:   提出 VADAR，一种 agentic 程序合成方法用于 3D 空间推理。多个 LLM agent 协作生成 Pythonic API 并在求解过程中动态扩展新函数来解决常见子问题，克服了 VisProg/ViperGPT 等先前方法依赖静态人工定义 API 的局限。同时引入涉及多步空间定位和推理的新 benchmark，在 3D 理解任务上超越现有零样本方法。

---

## 🛰️ 遥感 { #remote_sensing }

**[Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes](remote_sensing/dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)**

:   提出 Dense Dispersed Structured Light（DDSL）方法，利用廉价衍射光栅薄膜（<\$20）+ 立体 RGB 相机 + RGB 投影仪，设计光谱复用 DDSL 图案大幅减少所需投影帧数，实现 6.6fps 实时高光谱 3D 成像，光谱分辨率 15.5nm FWHM，深度误差 4mm。

**[DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery](remote_sensing/disciple_learning_interpretable_programs_for_scientific_visual_discovery.md)**

:   提出 DiSciPLE 框架，利用 LLM 引导的进化算法自动合成可解释的 Python 程序来分析视觉数据，在人口密度估计等科学任务上以比最近基线低 35% 的误差实现了 SOTA，且程序完全可解释。

**[EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](remote_sensing/earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)**

:   提出 EarthDial，一个专为地球观测 (EO) 数据设计的对话式视觉语言模型，支持多光谱 (SAR/NIR/红外)、多时序和多分辨率遥感影像的统一理解，基于 1111 万条指令微调数据集，在 44 个下游数据集上超越现有遥感 VLM。

**[Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](remote_sensing/hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)**

:   提出 UAV 场景变化描述（UAV-SCC）新任务及 HDC-CL 框架，通过动态自适应布局 Transformer 建模移动视角下的图像对重叠/非重叠区域，结合层级跨模态方向一致性校准增强视角偏移方向感知，并构建了专用基准数据集。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](remote_sensing/joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   研究多 LEO 卫星联合服务多天线地面用户的分布式 MIMO 下行通信，提出联合传输与流式传输两种模式：前者通过 WMMSE 迭代优化预编码器最大化和频谱效率，后者通过匈牙利算法的流-卫星关联减少前传开销，实现性能与前传负载的灵活权衡。

**[Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](remote_sensing/meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)**

:   MetaPEFT提出了一种元学习框架，将PEFT中的离散位置选择和连续缩放因子统一为可微分的调制器（modulator），通过双层优化自动搜索最优的PEFT超参数配置，在遥感和自然图像的长尾分布适应任务上取得SOTA。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](remote_sensing/metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   提出 MetaSpectra+，一种基于超表面-折射混合光学的紧凑多功能相机，通过双层超表面独立控制各通道色散/曝光/偏振，在约 250nm 可见光带宽内实现快照式高光谱+HDR 或高光谱+偏振联合成像，重建精度在基准数据集上达到 SOTA。

**[MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](remote_sensing/mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)**

:   MFogHub 构建了首个多区域（15个沿海区域）多卫星（6颗地球同步卫星）的全球海雾检测与预测数据集，包含超过68000个高分辨率样本和11600+像素级标注，通过16个基线模型的大规模实验揭示了区域差异和卫星变化对模型泛化能力的影响。

**[Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing](remote_sensing/think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)**

:   构建遥感多实体推理定位基准 ME-RSRG（首个显式标注主体-客体角色的遥感定位数据集），提出 Entity-Aware Reasoning (EAR) 框架，结合 SFT 冷启动与实体感知奖励驱动的 GRPO 优化，实现结构化推理链输出和主-客体联合定位，Qwen2.5-VL 系列在 EAR 优化后 mAcc@0.5 提升超 10%。

---

## 💡 LLM推理 { #llm_reasoning }

**[Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)**

:   Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。

**[CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](llm_reasoning/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   提出 CoT-VLA，将视觉思维链推理引入视觉-语言-动作模型，通过两阶段推理——先预测子目标图像再生成动作序列——结合混合注意力和动作分块策略，在 LIBERO 基准上实现 81.13% 平均成功率，显著超越现有方法。

**[Interleaved-Modal Chain-of-Thought](llm_reasoning/interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework](llm_reasoning/learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte.md)**

:   提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。

**[Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval](llm_reasoning/reason-before-retrieve_one-stage_reflective_chain-of-thoughts_for_training-free_.md)**

:   OSrCIR提出单阶段反思性链式思维推理，让MLLM同时处理参考图像和修改文本（避免两阶段caption→推理的信息丢失），通过"描述→思考→反思→目标描述"四步CoT生成准确的目标图像描述，在CIRCO上mAP@5达23.87%超越CIReVL 26.2%，且完全免训练。

**[Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](llm_reasoning/style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。

**[VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](llm_reasoning/videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)**

:   VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。

---

## 📐 优化/理论 { #optimization }

**[Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)**

:   提出 GETA 框架实现自动联合结构化剪枝和量化感知训练：量化感知依赖图（QADG）构建通用剪枝搜索空间 + 部分投影 SGD 保证逐层比特约束 + 可解释的联合学习策略，在 CNN 和 Transformer 上均达到竞争力或领先的压缩性能。

**[Federated Learning with Domain Shift Eraser](optimization/federated_learning_with_domain_shift_eraser.md)**

:   提出FDSE方法，将每层网络分解为域无关特征提取器（DFE，全局聚合增强共识）和域特异偏移消除器（DSE，个性化聚合保留本地特性），结合BN一致性正则化，在DomainNet上达到76.77%（超Ditto 1.6%），在Office-Caltech10上达到91.58%（超FedBN 4.6%）。

**[Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](optimization/leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)**

:   发现 OOD 样本的检测得分比 IND 样本更容易被对抗扰动降低，提出 PRO 方法——在推理时用梯度下降搜索 ε-球内的最小 OOD 得分，增强 IND/OOD 可分性，在 CIFAR-10 上 FPR@95 从 44.35% 降至 19.95%。

**[Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning](optimization/mind_the_gap_confidence_discrepancy_can_guide_federated_semi-supervised_learning.md)**

:   提出 TABASCO，一个两阶段二维样本选择框架解决同时存在标签噪声和长尾分布的联邦半监督学习：用加权 JSD（WJSD）和自适应质心距离（ACD）两个互补指标识别干净样本，GMM 聚类后以半监督方式利用剩余噪声数据，在 CIFAR-10（0.1 不平衡+0.4 噪声）上达 85.53%。

**[Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](optimization/model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)**

:   发现现有联邦学习模型投毒攻击因跨轮次方向不一致导致效果自相抵消，提出 PoisonedFL 通过固定随机方向向量 + 动态幅度调节 + 假设检验机制实现多轮一致性攻击，在无需任何真实客户端信息的前提下击穿 8 种 SOTA 防御。

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 提出了一种面向联邦学习的语义 coreset 选择框架，利用 VLM（MobileCLIP-S2）零样本提取三种标量指标（表示分数、多样性分数、边界接近度），通过服务器聚合全局共识后指导客户端进行两阶段剪枝（异常过滤+冗余消除），在 128-512× 上行带宽减少和 7.72× 加速的同时保持竞争精度。

**[Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent](optimization/stop_walking_in_circles_bailing_out_early_in_projected_gradient_descent.md)**

:   发现 PGD 攻击在 L∞ 球上对鲁棒样本会产生循环行为，通过哈希检测循环实现提前终止（PGD_CD），在保持完全相同鲁棒性评估结果的前提下实现最高 96% 的迭代次数减少。

---

## 🔗 因果推理 { #causal_inference }

**[Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency](causal_inference/adventurer_optimizing_vision_mamba_architecture_designs_for_efficiency.md)**

:   提出 Adventurer 系列视觉模型，通过"头部平均池化 token"和"层间翻转"两个简单设计将图像输入适配到单向因果扫描框架中，使 Mamba 架构在视觉任务上实现 4-6 倍于现有 Vision Mamba 的训练速度，同时保持与 ViT 相当甚至更优的精度。

**[Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition](causal_inference/antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua.md)**

:   提出 Antidote 统一框架，专门解决大型视觉语言模型在面对含反事实预设的问题（如"图中的大象是什么颜色的？"而图中没有大象）时的幻觉生成问题。

**[Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual](causal_inference/image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)**

:   提出基于溯因反事实推理的全参考图像质量评估方法（FR-IQA），通过建模"如果没有某种失真，感知质量会如何变化"的因果关系来更准确地评估图像质量。

**[Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](causal_inference/joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)**

:   提出 JSCPT，通过联合调度因果提示（消除虚假相关）和任务学习顺序（利用动态任务关系），优化多任务提示学习的性能。

**[Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](causal_inference/seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)**

:   将 MLLM 幻觉分为初始幻觉（首次出现的错误描述）和雪球幻觉（基于已有错误的累积放大），提出注意力因果解码方法分别针对两种类型进行缓解。

**[Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)**

:   针对模型错误分类的场景，提出细粒度的反事实解释方法 — 不仅展示"模型看了什么"，还能回答"如果哪些区域不同，模型就能正确分类"。

---

## 📚 预训练 { #llm_pretraining }

**[3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](llm_pretraining/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)**

:   提出跨任务少样本2D视线估计——利用预训练3D视线模型作为先验，通过**基于物理的可微投影模块**（6个可学习屏幕参数）将3D视线方向投影到2D屏幕坐标，仅需10张标注图像即可在未知设备上适配2D视线估计，在MPIIGaze/EVE/GazeCapture上比EFE和IVGaze提升超25%。

**[A Unified Framework for Heterogeneous Semi-supervised Learning](llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)**

:   提出异构半监督学习(HSSL)新问题设定——标记数据和无标记数据来自不同分布的域，目标是训练能在两个域上都泛化的模型；通过将C类问题扩展为2C类分类（每个域的同一语义类视为不同类），结合WMA伪标签、跨域原型对齐和渐进式跨域Mixup三个组件统一解决。

**[HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](llm_pretraining/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)**

:   HSEmotion 团队在 ABAW-10 竞赛中提出了一个轻量级 pipeline：用预训练 EfficientNet 提取面部 embedding，结合 MLP + GLA（Generalized Logit Adjustment）+ 滑窗平滑，在四项任务（EXPR/VA/AU/VD）上均大幅超过官方 baseline，其中暴力检测任务使用 ConvNeXt-T + TCN 达到 0.783 macro F1。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](llm_pretraining/mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)**

:   MXNorm 提出复用 MXFP 量化过程中已计算的 block absmax 来近似 RMS，将归一化与 MX 量化融合为单次统计收集操作，实现 RMSNorm 的 drop-in 替换，在 Llama 3 8B 预训练中保持训练精度的同时获得最高 2.4× 的 kernel 加速。

**[Robust Message Embedding via Attention Flow-Based Steganography](llm_pretraining/robust_message_embedding_via_attention_flow-based_steganography.md)**

:   首次将Transformer注意力机制与归一化流结合，通过可逆QR码转换、可逆令牌融合和注意力仿射耦合块，实现高质量高容量的隐写，在打印-拍摄等极端物理扭曲下仍能准确还原消息。

**[The Scene Language: Representing Scenes with Programs, Words, and Embeddings](llm_pretraining/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)**

:   提出 Scene Language——一种用程序（P, 编码层级结构）+ 词语（W, 语义类别）+ 嵌入（Z, 视觉身份）三元组 $\Phi(s)=(W,P,Z)$ 表示视觉场景的新范式，通过 Claude 3.5 Sonnet 的 training-free 推理从文本/图像输入生成场景表示，支持传统/神经/混合渲染，在 3D/4D 场景生成质量和可控编辑上超越场景图等现有表示。

---

## 🎮 强化学习 { #reinforcement_learning }

**[CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](reinforcement_learning/citywalker_learning_embodied_urban_navigation_from_web-scale_videos.md)**

:   利用互联网上超过 2000 小时的城市步行和驾驶视频，通过视觉里程计 (VO) 自动提取动作标签进行大规模模仿学习，训练出能在复杂动态城市环境中导航的具身智能体，真实部署成功率达 77.3%，显著超越现有方法。

**[Decision SpikeFormer: Spike-Driven Transformer for Decision Making](reinforcement_learning/decision_spikeformer_spike-driven_transformer_for_decision_making.md)**

:   提出 DSFormer，首个用于离线强化学习的脉冲驱动 Transformer，设计了时序脉冲自注意力 (TSSA) 和位置脉冲自注意力 (PSSA) 来捕获 RL 中的时序/位置依赖，并引入渐进式阈值依赖批归一化 (PTBN) 解决归一化与脉冲特性的冲突，在 D4RL 基准上超越 ANN 对手且节省 78.4% 能耗。

**[Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging](reinforcement_learning/gazing_at_rewards_eye_movements_as_a_lens_into_human_and_ai_decision-making_in_h.md)**

:   提出Visual Forager（VF）模型，通过目标特征调制、目标价值调制和ViT-based Actor-Critic决策网络模拟人类混合视觉搜索任务中的眼动策略，在归一化得分上达到72.6%（人类87.4%），扫视大小仅差0.01°（4.06° vs 人类4.05°），首次揭示目标价值和出现率如何联合影响人类搜索决策。

**[Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](reinforcement_learning/neural_motion_simulator_pushing_the_limit_of_world_models_in_reinforcement_learn.md)**

:   提出 MoSim，一个基于刚体动力学先验和 Neural ODE 的世界模型，可在物理状态空间中进行高精度长时域预测，首次实现零样本强化学习——不需任何真实环境交互即可训练策略。

**[SkillMimic: Learning Basketball Interaction Skills from Demonstrations](reinforcement_learning/skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)**

:   提出 SkillMimic，一个纯数据驱动的框架，通过统一的 HOI 模仿奖励（特别是创新的接触图奖励）从动捕数据中学习多样的篮球交互技能，并通过高层控制器组合技能实现连续得分等复杂长程任务。

**[ThinkStream: Thinking in Streaming Video](reinforcement_learning/thinking_in_streaming_video.md)**

:   提出 ThinkStream，采用 Watch-Think-Speak 范式实现流式视频的实时连续推理，通过 RCSM（推理压缩流式记忆）将推理 trace 作为紧凑语义锚点替代旧视觉 token，配合 Streaming RLVR 训练策略，在保持低延迟/低内存的同时超越现有在线视频模型。

---

## 📡 信号/通信 { #signal_comm }

**[ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](signal_comm/abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)**

:   提出 ABC-Former，利用辅助双模态信息（全局色温和局部色彩）通过跨域 Transformer 和交互通道注意力实现高质量的 sRGB 图像白平衡矫正。

**[Continuous Space-Time Video Resampling with Invertible Motion Steganography](signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)**

:   提出基于可逆运动隐写术的连续时空视频重采样方法，在时空下采样过程中将运动信息隐藏在低分辨率帧中，上采样时恢复隐藏的运动细节，实现灵活的时空重采样。

**[DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](signal_comm/ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)**

:   提出 DiTASK，利用连续分段仿射 (CPAB) 微分同胚变换对预训练权重矩阵的奇异值进行平滑变换而保持奇异向量不变，以每层仅约 32 个参数实现全秩更新的多任务微调，在 PASCAL MTL 上以 75% 更少的参数超越 MTLoRA 26.27%。

**[Neural Video Compression with Context Modulation](signal_comm/neural_video_compression_with_context_modulation.md)**

:   提出上下文调制机制，增强神经视频编码器（NVC）中时间上下文的传播和利用能力，解决现有条件编码方法时间上下文利用不充分的问题。

**[Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling](signal_comm/radio_frequency_ray_tracing_with_neural_object_representation_for_enhanced_rf_mo.md)**

:   提出 RFScape 框架，将神经场景表示（如 NeRF）扩展到射频（RF）频段，通过适配电磁传播物理特性实现增强的 RF 传播建模。

**[Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](signal_comm/tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)**

:   针对正弦神经网络（如 SIREN）的初始化和训练不稳定问题，提出基于理论分析的频率调谐方法，使训练过程更鲁棒、更可预测。

---

## 📈 时间序列 { #time_series }

**[Competition-Aware CPC Forecasting with Near-Market Coverage](time_series/competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   将付费搜索CPC预测重构为"部分可观测竞争下的预测"问题，通过语义邻域（Transformer嵌入）、行为邻域（DTW对齐）和地理意图三类竞争代理逼近不可观测的竞争状态，在1811个关键词×127周的Google Ads数据上显示竞争感知增强在中长期预测（6/12周）上显著优于单变量和弱上下文baseline。

**[DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](time_series/dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)**

:   提出 DejaVid，一种编码器无关的轻量级视频分类增强方法：将视频表示为变长时序嵌入序列 (TSE) 而非单个嵌入，通过学习每个时间步、每个特征维度的重要性权重，结合改进的可微分 DTW 算法做时序对齐分类，仅增加 <1.8% 参数就在 SSV2 达到 77.2%、K400 达到 89.1% 的 SOTA。

**[FLAVC: Learned Video Compression with Feature Level Attention](time_series/flavc_learned_video_compression_with_feature_level_attention.md)**

:   提出 FLAVC，在学习型视频压缩框架中引入特征级注意力机制，在特征域（而非像素域）中更高效地利用运动估计和时空上下文信息。

**[Learning Extremely High Density Crowds as Active Matters](time_series/learning_extremely_high_density_crowds_as_active_matters.md)**

:   将极高密度人群类比为物理学中的"主动物质"（active matter），从质量参差的野外视频中学习人群的集体动力学行为模式，用于人群分析和预测。

**[Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](time_series/reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)**

:   通过262个真实机器人导航episode的大规模实验，深入分析端到端RL训练的导航智能体内部涌现出的推理能力——包括类Kalman滤波的动力学模型、场景结构的潜在记忆、有限水平的规划能力以及与长期规划相关的价值函数。

---

## 🕸️ 图学习 { #graph_learning }

**[Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models](graph_learning/coeff-tuning_a_graph_filter_subspace_view_for_tuning_attention-based_large_model.md)**

:   将多头注意力重新解释为图卷积滤波器子空间，通过学习一组极小的子空间组合系数（$H \times H$ 矩阵）来线性组合预训练的注意力图，突破 softmax 造成的凸包约束从而扩展特征空间，以几乎零参数量的代价即插即用地提升各种 PEFT 方法的性能。

**[DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](graph_learning/dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)**

:   提出 DVHGNN，一种利用多尺度膨胀超图捕获图像 patch 间高阶相关性的视觉骨干网络，通过聚类+膨胀超图构造 (DHGC) 获取多尺度超边、动态超图卷积实现自适应特征交换，在 ImageNet-1K 上以 30.2M 参数达到 83.1% top-1 准确率，超越 ViG-S 1.0% 和 ViHGNN-S 0.6%。

**[Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](graph_learning/hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges.md)**

:   提出HgVT，将层次化二部超图结构嵌入ViT中，通过主图像patch顶点和虚拟顶点的分离处理、动态余弦邻接构建和超边通信池三层注意力机制，无需聚类即可捕获patch间高阶语义关系，在ImageNet-1K上HgVT-Ti以7.7M参数达到76.2%准确率（超ViHGNN-Ti 1.9%），并在图像检索中达到73.23% mAP@10。

**[Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](graph_learning/unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)**

:   提出 VISA 框架，从视觉（记忆引导序列建模 MGSM 降低特征方差）和语义（迭代关系生成器 IRG 引入层次上下文减少对偏置先验的依赖）双重角度对视频场景图生成进行去偏置，在 Action Genome 等数据集上大幅提升尾部类别性能。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](multilingual_mt/harnessing_frozen_unimodal_encoders_for_flexible_multimodal_alignment.md)**

:   提出一种新的视觉-语言对齐框架：冻结预训练好的单模态视觉编码器（DINOv2）和语言编码器（All-Roberta-Large），仅训练轻量MLP投影层实现多模态对齐，以20倍数据缩减和65倍计算缩减达到了CLIP级别甚至超越的性能。

**[Semantic and Expressive Variation in Image Captions Across Languages](multilingual_mt/semantic_and_expressive_variations_in_image_captions_across_languages.md)**

:   系统性证明了不同语言的图像描述在语义内容（对象、关系、属性）和表达方式（具象度、语调、真实性）上存在显著的分布差异，多语言描述集相比单语言提供更丰富的视觉信息（+46% 对象、+66.1% 关系、+66.8% 属性），为多语言数据训练视觉模型提供了实证支撑。

**[SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity](multilingual_mt/smtpd_a_new_benchmark_for_temporal_prediction_of_social_media_popularity.md)**

:   构建首个时间对齐的社交媒体流行度时序预测基准SMTPD（282K YouTube样本，30天连续观测），并提出基于多模态特征提取+LSTM时序回归的baseline框架，发现早期流行度（EP）是准确预测后续流行度的关键。

---

## ✍️ 文本生成 { #nlp_generation }

**[ArtFormer: Controllable Generation of Diverse 3D Articulated Objects](nlp_generation/artformer_controllable_generation_of_diverse_3d_articulated_objects.md)**

:   提出ArtFormer框架，通过树结构参数化和条件扩散Shape Prior，从文本/图像描述生成高质量、多样化且运动学关系准确的3D关节物体，在生成质量和多样性上显著超越现有方法。

**[Dense Match Summarization for Faster Two-view Estimation](nlp_generation/dense_match_summarization_for_faster_two-view_estimation.md)**

:   提出高效的稠密匹配摘要方案，从大量稠密对应中生成一小组代表性匹配子集，在保持位姿估计精度的同时大幅加速 RANSAC 鲁棒估计。

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](nlp_generation/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。

---

## 👥 社会计算 { #social_computing }

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

**[Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification](social_computing/classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)**

:   提出 Classifier-guided CLIP Distillation（CCD），通过 CAM 引导的局部视图标签聚合和 CLIP 预测去偏两项核心技术，在完全无标注的条件下达到与全监督方法持平的多标签分类性能（VOC12 上 90.1% mAP）。

**[Learning from Neighbors: Category Extrapolation for Long-Tail Learning](social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)**

:   发现更细粒度的类别划分天然减轻长尾不平衡的影响，提出用 LLM 发现与现有类别相关的细粒度辅助类 + 网络爬虫收集图像 + 邻近静默损失防止辅助类喧宾夺主，在 ImageNet-LT 上 Few 类提升 16 个百分点（41.4→57.4）。

---

## 🔎 AIGC检测 { #aigc_detection }

**[Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration](aigc_detection/enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit.md)**

:   提出无需额外训练的双层模态校准框架，利用 CLIP 等预训练视觉语言模型的跨模态对齐能力实现小样本类增量学习，在避免灾难性遗忘的同时学习新类。

**[ProAPO: Progressively Automatic Prompt Optimization for Visual Classification](aigc_detection/proapo_progressively_automatic_prompt_optimization_for_visual_classification.md)**

:   提出 ProAPO 渐进式自动提示优化方法，通过 LLM 生成视觉描述并迭代优化，自动为 VLM 发现最优分类提示，无需人工设计或梯度训练。

---

## 🔒 LLM安全 { #llm_safety }

**[Empowering LLMs to Understand and Generate Complex Vector Graphics](llm_safety/empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   通过引入55个SVG语义token、构建580k条指令微调数据集SVGX-SFT，使任意LLM能准确理解和生成复杂矢量图形，在文本对齐度和美观度上超越GPT-4o和Claude，推理速度比优化方法快50-150倍。

**[Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](llm_safety/order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)**

:   提出 GDDSG，用图着色理论将类按相似度分组——同组内类别尽量不相似（减少干扰），每组独立用 NCM 分类器+LoRA 适配器学习，在 CIFAR-100 10-step 上达到 94.00% 准确率和仅 0.78% 遗忘率（前 SOTA RanPAC 90.50%/3.49%）。

---

## ⚛️ 物理学 { #physics }

**[ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](physics/atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)**

:   提出自适应阈值剪枝（ATP）编码方法，通过自适应地剪除低信息量的数据维度来减少量子神经网络所需的量子比特数，在保持分类性能的同时提升 QNN 的可扩展性和效率。

**[Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding](physics/galaxy_walker_geometry-aware_vlms_for_galaxy-scale_understanding.md)**

:   提出 Galaxy Walker，一种几何感知的视觉语言模型，将球面空间（行星轨道）和双曲空间（黑洞）等非欧几何引入 VLM 框架，实现对天文现象的多尺度多几何理解。

---

## 🎁 推荐系统 { #recommender }

**[FineVQ: Fine-Grained User Generated Content Video Quality Assessment](recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)**

:   提出 FineVQ 细粒度 UGC 视频质量评估方法，将质量评估从单一整体评分扩展到多个感知质量维度（清晰度、稳定性、色彩、构图等），为视频推荐和优化提供更精细的质量信号。

**[VisionArena: 230k Real World User-VLM Conversations with Preference Labels](recommender/visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)**

:   发布 VisionArena，当前最大规模的众包真实用户-VLM 对话数据集（230K 条对话+偏好标签），为 VLM 评估提供反映真实使用场景的基准。

---

## 💻 代码智能 { #code_intelligence }

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](code_intelligence/codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过 scaling 分析发现 STEM 视觉推理的真正瓶颈是感知而非推理，提出用可执行 Python 代码作为精确感知媒介——构建 ICC-1M 数据集（Image-Caption-Code 三元组）训练模型，在 STEM 感知基准上 CodePercept-8B 比 Qwen3-VL-8B 提升 +3.0%-12.3%。

---

## 🧮 科学计算 { #scientific_computing }

**[Accurate Differential Operators for Hybrid Neural Fields](scientific_computing/accurate_differential_operators_for_hybrid_neural_fields.md)**

:   解决混合神经场（如 Instant NGP）中因离散特征网格与连续 MLP 耦合导致的微分算子不精确问题，提出适用于离散-连续混合表示的精确微分方法。

---

## 📂 其他 { #others }

**[A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](others/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含100万+复杂CAD模型、超1000万多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的A2Z数据集，是目前最大的CAD逆向工程数据集，并基于此训练了BRep边界和角点检测的基础模型。

**[AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](others/anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)**

:   提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。

**[BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出一个面向板金弯曲工艺的可制造性度量分类法（按配置依赖性×可行性/复杂度两个维度划分为四象限），并构建首个包含20,000个零件（含可制造与不可制造样本）的合成数据集BenDFM，基准测试表明图结构表示（UV-Net）优于点云（PointNext），配置依赖性指标的预测更具挑战性。

**[Bounds on Agreement between Subjective and Objective Measurements](others/bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   通过仅假设投票均值收敛于真实质量，推导出主观测试（MOS）与客观估计器之间PCC（上界）和MSE（下界）的数学界限，并提出基于二项分布的投票模型BinoVotes，使得即使在投票方差不可用时也能计算这些界限，18个主观测试数据的验证表明BinoVotes界限与全数据驱动界限高度吻合。

**[CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](others/cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)**

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](others/deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   通过提供完美的oracle噪声转移矩阵T，证明Forward Correction在理想条件下仍会训练崩塌（先升后降最终与无校正基线收敛），从宏观（收敛终态）、微观（梯度动力学）、信息论（噪声信道不可逆信息损失）三个层面系统诊断了失败的根本原因——这不是T估计不准的问题，而是有限样本下高容量网络的结构性缺陷。

**[Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](others/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)**

:   提出DPDL方法，通过学习多高斯分布原型并用Schrödinger桥将正常样本扩散映射到原型空间（同时推开异常样本），结合超球空间上的离散特征学习增强泛化性，在9个公开异常检测数据集上取得SOTA（如AITEX上超越AHL 5.0%、ELPV上超越8.7%）。

**[EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](others/edm_equirectangular_projection-oriented_dense_kernelized_feature_matching.md)**

:   提出EDM，首个基于学习的等距柱状投影（ERP）全景图像密集特征匹配方法，通过球面空间对齐模块（SSAM，使用3D笛卡尔坐标的球面位置编码+高斯过程回归）和测地线流细化处理ERP的极区畸变，在Matterport3D上AUC@5°超越DKM 26.72%、在Stanford2D3D上超越42.62%。

**[Effortless Active Labeling for Long-Term Test-Time Adaptation](others/effortless_active_labeling_for_long-term_test-time_adaptation.md)**

:   提出EATTA方法，在长期测试时适应（TTA）中通过特征扰动敏感度每批次仅标注1个最有价值样本（而非多个），结合梯度范数去偏策略平衡监督和无监督损失的梯度，在ImageNet-C上以极低标注代价实现50.9%的平均错误率（超过标注3倍的SimATTA 3.9%）。

**[EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](others/evos_efficient_implicit_neural_training_via_evolutionary_selector.md)**

:   提出EVOS方法，通过进化选择范式（稀疏适应度评估+频率引导交叉+增强无偏变异）对INR训练样本进行智能稀疏采样，在保持甚至提升重建质量（PSNR 37.81 vs 标准37.10）的同时将训练时间减少48-66%（180秒→97秒）。

**[Feature Selection for Latent Factor Models](others/feature_selection_for_latent_factor_models.md)**

:   提出基于信噪比（SNR）的类特异性特征选择方法用于低秩生成模型（PPCA/LFA/ELF），每新增一个类只需$O(1)$计算（不需重训旧类模型），避免了灾难性遗忘，并提出新的非参数潜因子模型ELF，在微阵列癌症分类和高维特征选择上验证了有效性。

**[Focal Split: Untethered Snapshot Depth from Differential Defocus](others/focal_split_untethered_snapshot_depth_from_differential_defocus.md)**

:   受跳蛛视觉启发，构建首个无线（电池供电）的快照式差分离焦深度相机 Focal Split，用分光镜将光路分给两个不同焦距的传感器，仅需 500 FLOPs/像素和 4.9W 功率即可在树莓派上实时估计深度。

**[Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](others/full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)**

:   提出首个仅用事件流估计完整6-DoF自运动（角速度+线速度）的几何求解器方法，通过建立事件扇形流形上的线段几何约束——入射关系和新颖的共面关系，设计最少仅需8个事件的稀疏求解器，无需IMU即可解耦旋转和平移估计。

**[H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](others/h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)**

:   提出H2ST方法，用层次化的两样本检验架构实现增量学习中的OOD检测——每个任务对应一个特征级别的源-目标二分类器层，通过Clopper-Pearson置信区间假设检验自动判定ID/OOD（无需手动阈值），同时提供任务ID预测能力，在7个基准上优于MSP/Energy/ODIN且计算效率提升$(T+1)/2$倍。

**[Improving Transferable Targeted Attacks with Feature Tuning Mixup](others/improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)**

:   提出 FTM（Feature Tuning Mixup）通过在代理模型的特征空间中混合优化的攻击专用扰动和随机干净扰动来提升有目标对抗攻击的迁移性，使用动量式随机更新策略保持计算效率，14 个黑盒模型上平均成功率从 74.6% 提升到 77.4%。

**[Instance-wise Supervision-level Optimization in Active Learning](others/instance-wise_supervision-level_optimization_in_active_learning.md)**

:   本文提出 ISO (Instance-wise Supervision-level Optimization) 框架，在主动学习中不仅选择哪些样本标注，还为每个样本自动决定最优的标注级别（精确标签 vs 粗标签），通过价值-成本比(VCR)和多样性感知的批次选择算法，在固定预算约束下达到比传统主动学习高10%+的准确率。

**[Integral Fast Fourier Color Constancy](others/integral_fast_fourier_color_constancy.md)**

:   本文提出 IFFCC，将 FFCC 算法扩展到多光源场景，通过积分 UV 直方图加速区域直方图计算并行化傅里叶卷积操作，实现了与像素级神经网络相当的精度，同时参数量减少 400 倍、速度提升 20-100 倍的实时多光源自动白平衡。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](others/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，在制药 BFS 高速产线上实现了仅用正常样本训练、单 patch 推理 0.17ms 的实时在线质检部署，通过 Perlin 噪声增强和 Noise Loss 优化重建质量。

**[Joint Out-of-Distribution Filtering and Data Discovery Active Learning](others/joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)**

:   提出 Open-Set Discovery Active Learning (OSDAL) 场景，并设计 Joda 算法，通过训练-过滤-选择三阶段流程，用单一模型同时过滤 OOD 数据和发现新类别，无需额外辅助模型，在 18 个配置上持续达到最高准确率。

**[LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos](others/latte-mv_learning_to_anticipate_table_tennis_hits_from_monocular_videos.md)**

:   LATTE-MV 提出一套从单目乒乓球比赛视频中重建 3D 比赛数据的可扩展系统，并训练 Transformer 模型预判对手击球意图，结合共形预测实现不确定性感知的预判式控制，将仿真中机器人回球率从 49.9% 提升至 59.0%。

**[Locally Orderless Images for Optimization in Differentiable Rendering](others/locally_orderless_images_for_optimization_in_differentiable_rendering.md)**

:   提出利用局部无序图像（LOI）的三维尺度空间（内尺度 σ、色调尺度 β、范围尺度 α）进行直方图匹配的逆渲染优化方法，无需修改可微渲染器即可扩展稀疏梯度的支持范围，有效避免局部最优。

**[MOS: Modeling Object-Scene Associations in Generalized Category Discovery](others/mos_modeling_object-scene_associations_in_generalized_category_discovery.md)**

:   挑战了GCD中"场景信息是噪声"的传统观点，发现场景被误解为噪声是因为"歧义挑战"（目标与场景的base/novel关系冲突），提出MOS框架通过双分支网络+MLP场景感知模块有效利用场景信息，在细粒度GCD上平均提升4%。

**[Open Set Label Shift with Test Time Out-of-Distribution Reference](others/open_set_label_shift_with_test_time_out-of-distribution_reference.md)**

:   提出 NoT 三阶段框架解决开集标签偏移（OSLS）：通过参考数据集估计源域 ID/OOD 比例，用 EM 算法最大似然估计目标域标签分布，再校正不完美 OOD 分类器的偏差，无需重训练分类器即可适应标签分布变化。

**[Order-One Rolling Shutter Cameras](others/order-one_rolling_shutter_cameras.md)**

:   建立了卷帘快门（Rolling Shutter, RS）相机的统一代数理论，定义了"一阶RS相机"（RS1）——将空间点一一映射到图像点的卷帘模型，完整分类了其三种类型（I/II/III），并发现了 31 个 2-5 相机间相对位姿估计的最小问题。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](others/rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   建立基于风洞 PIV 实验数据（非 CFD 模拟）的屋顶风场重建框架，系统对比 Kriging 插值与三种深度学习模型（UNet、ViTAE、CWGAN）在 5-30 个稀疏传感器下的重建性能，发现混合风向训练（MDT）使深度学习全面超越 Kriging（SSIM 提升最高 32.7%），并用 QR 分解优化传感器布局提升鲁棒性达 27.8%。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](others/sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶作为刚体的物理先验，在 ViT 中间层提取尺度不变的梯度能量统计量作为跨模态几何锚点，并在终端层将特征解耦为模态不变共享特征和模态特定特征后通过加性残差融合，实现光学-SAR 船舶重识别 SOTA。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](others/shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，利用图拉普拉斯算子的谱嵌入技术，从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群 $C_n$，在多个公开数据集上实现了接近原子分辨率的从头螺旋结构重建。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   本文构建了一个包含24万+工业零件的大规模多模态CAD数据集 SldprtNet，每个样本对齐了3D模型、多视角图像、参数化建模脚本和自然语言描述四种模态，并开发了支持13种CAD操作的编码器/解码器工具实现无损双向转换，实验证明多模态输入显著优于纯文本输入。

**[STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](others/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)**

:   STRAP-ViT 提出一种无需训练的即插即用 ViT 防御模块，利用 Jensen-Shannon 散度将受对抗补丁影响的 token 从正常 token 中分离出来，再通过随机复合变换消除其对抗效应，在多种 ViT 架构和攻击方法下实现了接近干净基线 2-3% 的鲁棒精度。

**[Test-Time Augmentation Improves Efficiency in Conformal Prediction](others/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)**

:   发现测试时数据增强（TTA）可以系统性地提升共形预测的效率——通过在校准集上学习增强权重来优化增强聚合策略，在 ImageNet ResNet-50 上将预测集大小减少 10-17%，同时严格保持覆盖率保证。

**[UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](others/uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)**

:   提出 UniPhy，首个统一的潜变量条件本构模型，在共享潜空间中编码弹性体/沙子/塑料/牛顿/非牛顿流体等多种材料属性，推理时通过可微 MPM 仿真器优化潜变量以匹配观测粒子轨迹，重建误差比 NCLaw 低 1-2 个数量级。

**[Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](others/wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)**

:   针对柔性磨料翻页轮的复杂磨损模式，提出三级层次化深度学习分类框架，将磨损评估分解为使用状态检测、磨损类型识别和严重程度评估三个子任务，使用EfficientNetV2迁移学习实现93.8%–99.3%的分类精度。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，将零阶优化策略性地整合到 SAM 的扰动步骤中，仅需一次反向传播即可获得 SAM 的平坦最小值优势，在稀疏训练场景下将计算开销减半的同时提升精度和鲁棒性。

</div>