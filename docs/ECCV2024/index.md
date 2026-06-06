---
title: >-
  868 篇 ECCV2024 论文解读 · 每篇 5 分钟读懂
description: >-
  868篇ECCV2024论文解读，涵盖 3D 视觉(175篇)、图像生成(117篇)、自动驾驶(54篇)、语义分割(53篇)、人体理解(51篇)、多模态 VLM(48篇)、视频理解(47篇)、目标检测(29篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2024"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "3D 视觉"
  - "图像生成"
  - "自动驾驶"
  - "语义分割"
  - "人体理解"
  - "多模态 VLM"
  - "视频理解"
  - "目标检测"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎞️ ECCV2024 论文笔记

868篇ECCV2024论文解读，涵盖 3D 视觉(175篇)、图像生成(117篇)、自动驾驶(54篇)、语义分割(53篇)、人体理解(51篇)、多模态 VLM(48篇)、视频理解(47篇)、目标检测(29篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🧊 3D 视觉 { #3d_vision }

**[3D Congealing: 3D-Aware Image Alignment in the Wild](3d_vision/3d_congealing_3d-aware_image_alignment_in_the_wild.md)**

:   3D Congealing将一组语义相似的无标注互联网图像对齐到共享的3D canonical空间，通过结合预训练扩散模型的SDS指导获得3D形状 + DINO语义特征匹配估计位姿和坐标映射，无需模板、位姿标注或相机参数。

**[3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_vision/3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)**

:   提出HORSE框架，通过从野外视频中提取多视角2D mask监督（以手部姿态作为物体姿态代理）和从合成3D形状集合中学习2D切片对抗形状先验，训练occupancy网络从单张RGB图像重建手持物体3D形状，在不使用任何真实世界3D标注的情况下，在MOW数据集上超越使用3D监督的方法11.6%。

**[3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_vision/3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)**

:   HVTrack首次探索高时间变化场景下的3D单目标跟踪，通过相对位姿感知记忆模块(RPM)、基础-扩展特征交叉注意力(BEA)和上下文点引导自注意力(CPA)三个模块，分别解决点云形状剧变、相似物体干扰和背景噪声问题，在KITTI-HV 5帧间隔下比SOTA提升11.3%/15.7% Success/Precision。

**[3DEgo: 3D Editing on the Go!](3d_vision/3dego_3d_editing_on_the_go.md)**

:   3DEgo将传统三阶段3D编辑流程（COLMAP位姿估计→未编辑场景初始化→迭代编辑更新）压缩为单阶段框架：先用自回归噪声混合模块对视频帧进行多视角一致的2D编辑，再用COLMAP-free的3DGS从编辑后帧直接重建3D场景，速度提升约10倍且支持任意来源视频。

**[3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting](3d_vision/3igs_factorised_tensorial_illumination_for_3d_gaussian_splatting.md)**

:   3iGS 用基于张量分解的连续入射光照场替代 3DGS 中每个高斯体独立优化的球谐系数，结合可学习 BRDF 特征和轻量神经渲染器来建模出射辐射，在保持实时渲染速度的同时显著提升了镜面反射等视角依赖效果的渲染质量。

**[3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)**

:   提出了一种无需训练的3D物体部件分割方法3-By-2，利用扩散模型(DIFT)的2D语义对应关系从已标注2D数据集或少量3D标注对象中迁移部件标签到3D，在zero-shot和few-shot设置下均达到SOTA。

**[4Diff: 3D-Aware Diffusion Model for Third-to-First Viewpoint Translation](3d_vision/4diff_3d-aware_diffusion_model_for_third-to-first_viewpoint_translation.md)**

:   本文提出 4Diff，一个结合3D几何先验的 transformer-based 扩散模型，通过自中心点云光栅化和3D感知旋转交叉注意力两个机制，将第三人称（外视角）图像转换为第一人称（自中心视角）图像，在 Ego-Exo4D 数据集上达到 SOTA 并展现出对新环境的强泛化能力。

**[6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](3d_vision/6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)**

:   提出6DGS，通过反转3DGS渲染流程——从椭球体表面均匀发射光线（Ellicell），利用注意力机制将光线与目标图像像素绑定，再用加权最小二乘闭式求解相机位姿，无需迭代和初始位姿，在真实场景上旋转精度提升12%、平移精度提升22%，达到15fps近实时性能。

**[A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](3d_vision/a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)**

:   将3DGS中的位置和旋转参数建模为时间的函数（位置用Fourier逼近、旋转用线性逼近），使动态场景的存储复杂度从O(TN)降低到O(LN)，在D-NeRF/DyNeRF/HyperNeRF三个数据集上实现了与NeRF方法匹敌的渲染质量，同时保持118+ FPS的实时渲染速度。

**[A Probability-guided Sampler for Neural Implicit Surface Rendering](3d_vision/a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)**

:   提出一种概率引导的光线采样器（Probability-guided Sampler），在3D图像投影空间中建模概率密度函数来指导光线采样朝向感兴趣区域，同时设计了包含近表面和空白空间两个分量的新型表面重建损失，可作为插件集成到现有神经隐式表面渲染器中，显著提升重建精度和渲染质量。

**[AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](3d_vision/aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)**

:   提出AEDNet，通过自适应点云嵌入与多视角感知解耦（AED）模块，在编码器和解码器中分别对点云进行全局嵌入和局部解耦，利用从单位球面生成的3D视点从外部观察点云，实现了对3D对象几何的全面理解，在MVP和PCN数据集上达到SOTA。

**[Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](3d_vision/analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)**

:   提出Analysis-by-Synthesis Transformer（AST），在统一框架中通过Shape Transformer和Texture Transformer分别建模像素到形状和像素到纹理的关系，仅使用2D标注就能实现高质量的网格重建和纹理生成，在CUB-200-2011和ShapeNet上超越已有方法。

**[Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](3d_vision/analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)**

:   通过使用条件 logistic 函数解析近似高斯信号在像素窗口上的积分，替代 3DGS 的像素中心点采样，实现无混叠的 3D 高斯泼溅，在多尺度渲染上超越 Mip-Splatting。

**[AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](3d_vision/animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)**

:   提出 AnimatableDreamer，通过 Canonical Score Distillation (CSD) 技术，从单目视频提取骨骼和运动后生成文本引导的可动画化 3D 非刚体模型，在生成质量和时序一致性上全面超越现有方法。

**[BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](3d_vision/bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)**

:   首次将运动模糊物理成像模型引入 3D Gaussian Splatting 框架，联合优化场景 Gaussian 参数与曝光时间内的相机运动轨迹，从模糊图像中恢复清晰 3D 场景并实现实时渲染。

**[BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](3d_vision/benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)**

:   提出 BeNeRF，仅从**单张模糊图像**及其对应的事件流（event stream）联合恢复神经辐射场与相机运动轨迹，无需多视角输入或已知位姿，即可实现高质量去模糊与新视角合成。

**[Bi-directional Contextual Attention for 3D Dense Captioning](3d_vision/bi-directional_contextual_attention_for_3d_dense_captioning.md)**

:   提出 BiCA，通过双向上下文注意力机制将 instance query 和 context query 解耦并行解码，解决了 3D 密集描述中定位与描述生成之间的目标冲突，在 ScanRefer 和 Nr3D 两个基准上取得 SOTA。

**[Binomial Self-compensation for Motion Error in Dynamic 3D Scanning](3d_vision/binomial_self-compensation_for_motion_error_in_dynamic_3d_scanning.md)**

:   提出二项式自补偿(BSC)算法,通过对运动受影响的相位序列按二项式系数加权求和,无需任何中间变量即可指数级消除四步相位移轮廓术中的运动误差,实现与相机帧率相同的高精度动态3D扫描。

**[CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering](3d_vision/caesarnerf_calibrated_semantic_representation_for_few-shot_generalizable_neural_.md)**

:   提出 CaesarNeRF，在可泛化 NeRF（GNT）基础上引入场景级语义表征，通过相机位姿校准（特征旋转对齐到目标视角）和序列细化（跨 Transformer 层逐步更新全局特征），在 1-view 设置下 PSNR 比 GNT 提升 1.74dB（LLFF），且可即插即用地增强 IBRNet、MatchNeRF 等其他基线。

**[Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation](3d_vision/camera_height_doesnapost_change_unsupervised_training_for_metric_monocular_road-.md)**

:   提出FUMET训练框架,利用道路上检测到的车辆尺寸先验聚合为相机高度估计,并利用相机高度在同一视频序列中不变的事实作为度量尺度监督,使任意单目深度网络无需辅助传感器即可学习绝对尺度。

**[CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images](3d_vision/canonicalfusion_generating_drivable_3d_human_avatars_from_multiple_images.md)**

:   提出CanonicalFusion框架,通过联合预测深度图和压缩LBS权重映射图实现直接规范化,并利用前向蒙皮可微渲染融合多张图像信息,从多张输入图像生成可驱动的3D人体Avatar。

**[CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](3d_vision/cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)**

:   提出CG-SLAM,基于一致性和几何稳定性优化的不确定性感知3D高斯场,实现高效稠密RGB-D SLAM,在定位精度和建图质量上均达到SOTA,跟踪速度最高15Hz。

**[CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](3d_vision/citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)**

:   提出 CityGaussian (CityGS)，通过分治训练策略和 block-wise Level-of-Detail 机制，首次实现了城市级大规模场景（>1.5 km²）的高质量 3D Gaussian Splatting 训练与跨尺度实时渲染。

**[Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)**

:   提出Click-Gaussian，通过学习两级粒度（粗/细）的可区分3D特征场，结合全局特征引导学习(GFL)解决跨视角mask不一致问题，实现每次点击仅需10ms的实时3D高斯交互式分割，速度比现有方法快15-130倍，同时显著提升分割精度。

**[CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](3d_vision/cloudfixer_test-time_adaptation_for_3d_point_clouds_via_diffusion-guided_geometr.md)**

:   本文提出CloudFixer，首个针对3D点云的测试时输入适应方法，通过预训练扩散模型引导的几何变换参数优化，将分布偏移的测试点云变换回源域，同时避免了扩散模型的反向传播，实现了不到1秒的单实例适应速度。

**[CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](3d_vision/coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)**

:   提出CoherentGS，通过为3DGS引入结构化表示（每像素一个高斯）并利用隐式卷积解码器和全变差损失构建单视图和多视图一致性约束，结合基于单目深度的初始化策略，在极稀疏输入（如3张图像）下实现高质量新视角合成，LPIPS指标显著优于现有NeRF方法。

**[ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance](3d_vision/comboverse_compositional_3d_assets_creation_using_spatially-aware_diffusion_guid.md)**

:   提出ComboVerse，一个组合式3D资产生成框架：先将包含多个物体的输入图像分解并独立重建为单物体3D模型，再通过空间感知的Score Distillation Sampling (SSDS)引导物体的位置、缩放和旋转参数优化，实现高质量多物体组合3D资产创建，在CLIP Score和人类评估中均显著优于现有方法。

**[Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](3d_vision/compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)**

:   提出一种高度压缩的 triplane 潜空间自编码器，配合两阶段扩散模型（先生成 shape embedding 再生成 triplane latent），仅需 7 秒即可从单张图像生成高质量 3D 资产，且训练数据和时间远少于同类方法。

**[CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)**

:   发现同时训练两个 3DGS 辐射场时它们在高斯位置和渲染结果上的差异（disagreement）与重建质量负相关，据此提出 CoR-GS 通过协同剪枝和伪视角协同正则化来抑制不准确重建，在稀疏视角下实现 SOTA 新视角合成。

**[CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model](3d_vision/crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)**

:   提出卷积重建模型 CRM，利用 triplane 与六个正交视图之间的空间对齐先验，用 U-Net 替代 Transformer 直接从六视图映射到 triplane，结合 FlexiCubes 端到端训练，10 秒内从单张图像生成高保真纹理网格，训练成本仅为 LRM 的 1/8。

**[CrossScore: Towards Multi-View Image Evaluation and Scoring](3d_vision/crossscore_towards_multi-view_image_evaluation_and_scoring.md)**

:   提出 Cross-Reference（CR）图像质量评估新范式，通过对比查询图像与多个不同视角参考图像，利用 cross-attention 神经网络预测与 SSIM 高度相关的像素级质量分数，无需 ground truth 参考图像即可评估新视角合成质量。

**[D-SCo: Dual-Stream Conditional Diffusion for Monocular Hand-Held Object Reconstruction](3d_vision/d-sco_dual-stream_conditional_diffusion_for_monocular_hand-held_object_reconstru.md)**

:   提出双流条件扩散模型 D-SCo 从单张 RGB 图像重建手持物体点云，通过统一手-物语义嵌入和手关节几何嵌入两个分支分别提供语义和几何先验，配合手约束质心固定策略稳定扩散过程，在 ObMan 上 F-5 达 0.61（超 DDF-HO 10.9%），真实数据集 HO3D/MOW 上也大幅领先。

**[DATENeRF: Depth-Aware Text-based Editing of NeRFs](3d_vision/datenerf_depth-aware_text-based_editing_of_nerfs.md)**

:   利用NeRF重建的场景深度信息来引导基于文本的2D图像编辑（通过深度条件化的ControlNet + 投影修复方案），从而实现多视角一致的高质量NeRF场景编辑。

**[Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](3d_vision/deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)**

:   提出 Deblur e-NeRF，通过物理精确的像素带宽模型来建模事件相机的运动模糊，首次实现从运动模糊的事件流中直接有效地重建无模糊 NeRF。

**[Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](3d_vision/deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)**

:   利用微调的 Stable Diffusion + ControlNet 将粗糙 NeRF/3DGS 渲染结果转化为高质量伪观测图像，将稀疏输入视图增密 5-10 倍后重新训练，在 Hypersim/LLFF/ScanNet 等数据集上超越 FreeNeRF 等方法 1-2dB PSNR，训练速度比扩散正则化方法快约 10 倍。

**[Deep Patch Visual SLAM](3d_vision/deep_patch_visual_slam.md)**

:   基于 DPVO 视觉里程计系统，通过高效的邻近回环检测和经典回环检测机制，将其扩展为完整的 SLAM 系统 DPV-SLAM，在单 GPU 上实现实时、高精度、低显存的单目视觉 SLAM。

**[DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](3d_vision/dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)**

:   提出 DG-PIC，首个在统一模型中同时处理多领域多任务的点云理解框架，通过双层次源域原型估计和测试时特征平移机制，在不更新模型的情况下提升对未知域的泛化能力。

**[Differentiable Convex Polyhedra Optimization from Multi-view Images](3d_vision/differentiable_convex_polyhedra_optimization_from_multi-view_images.md)**

:   提出一种基于对偶变换和三平面交点求解的可微凸多面体构造方法，绕过隐式场监督，直接利用多视角图像损失进行梯度优化，实现高保真的凸多面体形状表示。

**[Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](3d_vision/diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)**

:   利用text-to-image扩散模型（ControlNet/T2I-Adapter）将简单场景图像转化为保持同一3D结构的恶劣条件图像，通过自蒸馏微调现有单目深度估计网络，统一解决恶劣天气和非朗伯表面等分布外挑战。

**[DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](3d_vision/diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)**

:   本文首次将扩散去噪过程引入单目深度估计任务，通过在潜在深度空间中执行视觉条件引导的迭代去噪，并提出自扩散机制解决稀疏GT深度的模式崩塌问题，在KITTI和NYU-Depth-V2上达到SOTA性能。

**[Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images](3d_vision/divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)**

:   提出"分而治之"的自底向上人体网格重建方法，通过独立重建各身体部位后融合，有效解决人体大面积不可见时传统自顶向下方法（如SMPL）失效的问题。

**[DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](3d_vision/dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)**

:   提出DreamDissector框架，通过Neural Category Field和Deep Concept Mining将包含多物体交互的text-to-3D NeRF解耦为独立的带纹理网格，实现物体级别的3D编辑控制。

**[DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](3d_vision/dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)**

:   提出DreamScene360，利用全景图像作为中间表示，结合GPT-4V自精炼机制和全景3D高斯溅射技术，实现从文本到沉浸式360°3D场景的快速生成。

**[DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](3d_vision/dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)**

:   提出DreamView，通过自适应文本引导注入模块，将视角特定的文本描述和全局文本描述协同注入扩散模型，实现可定制化且多视角一致的文本到3D生成。

**[Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation](3d_vision/dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)**

:   提出双层自适应自标注方法，通过半松弛最优传输处理类别不平衡问题，并结合区域级表示增强点级分类器的学习，在点云分割中实现高效的新类发现。

**[Dynamic Neural Radiance Field from Defocused Monocular Video](3d_vision/dynamic_neural_radiance_field_from_defocused_monocular_video.md)**

:   提出 $D^2RF$，首个从散焦单目视频中恢复清晰动态NeRF的方法，通过将景深(DoF)渲染与体积渲染统一，引入分层DoF体积渲染来建模散焦模糊并恢复清晰新视角。

**[Efficient Depth-Guided Urban View Synthesis (EDUS)](3d_vision/efficient_depth-guided_urban_view_synthesis.md)**

:   提出EDUS方法，利用噪声几何先验（单目/双目深度）引导可泛化NeRF，通过前景3D CNN + 背景/天空图像渲染的三部分分解，实现稀疏街景视角下的快速前馈推理和高效逐场景微调。

**[Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](3d_vision/equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)**

:   提出基于SE(3)等变图神经网络的稀疏点云配准方法Equi-GSPR，通过等变消息传播、低秩特征变换（LRFT）和隐式特征空间相似度匹配，在室内外数据集上以低模型复杂度实现SOTA配准性能。

**[Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](3d_vision/explicitly_guided_information_interaction_network_for_cross-modal_point_cloud_co.md)**

:   提出EGIInet框架，通过统一编码器实现模态对齐，并利用显式引导的信息交互策略（FT-Loss）让网络精准识别图像中的关键结构信息，在视图引导点云补全任务上以更少参数实现了超越XMFnet 16% CD的性能。

**[External Knowledge Enhanced 3D Scene Generation from Sketch](3d_vision/external_knowledge_enhanced_3d_scene_generation_from_sketch.md)**

:   提出SEK框架，结合手绘草图和外部物体关系知识库作为扩散模型的条件，通过知识增强图推理和频谱滤波器，端到端地同时生成3D室内场景的布局和物体几何形状。

**[FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](3d_vision/falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)**

:   提出FALIP（Foveal-Attention CLIP），一种免训练方法，通过在CLIP的多头自注意力模块中插入类似人类中央凹视觉的注意力掩码，在不修改原始图像的情况下增强CLIP的区域感知能力，在指代表达理解、图像分类和3D点云识别等零样本任务上均取得提升。

**[FastCAD: Real-Time CAD Retrieval and Alignment from Scans and Videos](3d_vision/fastcad_real-time_cad_retrieval_and_alignment_from_scans_and_videos.md)**

:   提出FastCAD，通过对比学习嵌入空间蒸馏和直接参数预测，实现50ms内完成场景中所有物体的CAD模型检索与对齐，比现有方法快50倍且精度更优。

**[Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering](3d_vision/flash_cache_reducing_bias_in_radiance_cache_based_inverse_rendering.md)**

:   提出一种无偏的辐射缓存逆渲染方法，通过遮挡感知的vMF重要性采样和快速缓存控制变量技术，在保持计算效率的同时消除现有方法中的渲染偏差，提升材质和光照分解的质量。

**[FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally](3d_vision/flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)**

:   将3D高斯溅射的2D-to-3D分割问题建模为整数线性规划，利用alpha混合的线性性质得到闭式最优解，仅需30秒完成优化，比现有方法快50倍。

**[FlashTex: Fast Relightable Mesh Texturing with LightControlNet](3d_vision/flashtex_fast_relightable_mesh_texturing_with_lightcontrolnet.md)**

:   提出LightControlNet——一种光照感知的ControlNet变体，结合两阶段纹理优化pipeline，能在约4分钟内为3D网格生成高质量、可重光照的PBR纹理，速度比现有方法快3-10倍。

**[FLAT: Flux-Aware Imperceptible Adversarial Attacks on 3D Point Clouds](3d_vision/flat_flux-aware_imperceptible_adversarial_attacks_on_3d_point_clouds.md)**

:   本文提出FLAT框架，从通量（flux）的角度解决3D点云对抗攻击中的不可感知性问题——通过计算局部扰动向量场的通量来评估均匀性变化，并在检测到高通量（均匀性破坏）时调整扰动方向，生成远比现有方法更难被察觉的对抗点云。

**[Flying with Photons: Rendering Novel Views of Propagating Light](3d_vision/flying_with_photons_rendering_novel_views_of_propagating_light.md)**

:   提出瞬态场（Transient Field）表示，结合首创的多视点超快成像数据集，首次实现从动态新视角渲染真实场景中传播光的视频，能处理散射、反射、折射和衍射等复杂光传输效果。

**[Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis](3d_vision/forest2seq_revitalizing_order_prior_for_sequential_indoor_scene_synthesis.md)**

:   提出Forest2Seq框架，通过将无序的室内场景物体组织为层次化的场景树/森林结构，用广度优先遍历导出有意义的排列顺序作为先验知识，配合Transformer自回归解码器显著提升室内场景合成质量。

**[Formula-Supervised Visual-Geometric Pre-training (FSVGP)](3d_vision/formula-supervised_visual-geometric_pre-training.md)**

:   提出FSVGP，利用分形几何的数学公式自动生成对齐的合成图像和点云，通过公式监督一致性标签在统一Transformer上实现跨模态视觉-几何预训练，在图像和3D物体的分类、检测、分割六项任务上均超越单模态FDSL方法。

**[FutureDepth: Learning to Predict the Future Improves Video Depth Estimation](3d_vision/futuredepth_learning_to_predict_the_future_improves_video_depth_estimation.md)**

:   提出FutureDepth，通过未来预测网络(F-Net)学习运动线索和重建网络(R-Net)学习多帧对应关系，将隐式的运动和场景特征注入深度解码器，在NYUDv2、KITTI、DDAD、Sintel四个数据集上达到SOTA精度和时序一致性，且推理效率显著优于现有视频深度方法。

**[G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](3d_vision/g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)**

:   提出了G²fR（Generalized Grid-based Frequency Regularization），通过理论分析建立频率正则化与网格特征编码NeRF的联系，解决了GFE-NeRF在相机位姿优化和少样本重建中的核心问题。

**[G3R: Gradient Guided Generalizable Reconstruction](3d_vision/g3r_gradient_guided_generalizable_reconstruction.md)**

:   提出G3R，一种梯度引导的可泛化重建方法，通过学习一个重建网络迭代地利用可微渲染的3D梯度反馈更新3D Neural Gaussians表示，在大规模场景（>10,000m²）上实现2分钟内重建，加速至少10倍且达到与3DGS可比或更优的渲染质量。

**[GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views](3d_vision/gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)**

:   提出GAURA，一种基于可泛化NeRF的统一复原与渲染框架，通过可学习的退化感知latent codes在特征聚合和渲染阶段动态适应不同图像退化类型，无需逐场景优化即可从退化图像中渲染清晰的新视角。

**[GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](3d_vision/gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)**

:   提出GaussCtrl，利用深度条件化的ControlNet编辑和注意力对齐模块实现多视角一致的文本驱动3DGS场景编辑，支持一次编辑所有视角并仅需一次3D模型更新。

**[Gaussian Grouping: Segment and Edit Anything in 3D Scenes](3d_vision/gaussian_grouping_segment_and_edit_anything_in_3d_scenes.md)**

:   为 3D Gaussian Splatting 中的每个高斯学习 16 维 Identity Encoding 实现实例级分组，使用 SAM + DEVA 视频跟踪生成多视图一致的 2D 伪标签做监督，在 LERF-Mask 开放词汇分割上 mIoU 达 69-77%（超 LERF 2 倍+），全景分割超 Panoptic Lifting 4.9% mIoU 且 14× 更快，同时支持 3D 物体移除/修复/着色/风格迁移等多种编辑。

**[GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)**

:   提出GaussianImage，首次将2D Gaussian Splatting用于图像表示与压缩，通过紧凑的8参数2D高斯和累积求和光栅化算法，实现了2000 FPS的解码速度，同时与INR方法在表示质量和压缩性能上持平。

**[GaussReg: Fast 3D Registration with Gaussian Splatting](3d_vision/gaussreg_fast_3d_registration_with_gaussian_splatting.md)**

:   首次探索3D Gaussian Splatting场景之间的配准问题，提出粗到精的GaussReg框架——粗阶段利用点云配准方法估计初始变换，精阶段通过渲染图像提取体积特征进行精细对齐，速度比HLoc快44倍且精度可比。

**[Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](3d_vision/generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)**

:   提出GCD（Generative Camera Dolly），通过微调Stable Video Diffusion模型实现从单目视频生成任意视角的同步动态新视角视频，支持最高180°的极端相机变换，无需深度输入或显式3D建模。

**[GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](3d_vision/geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)**

:   提出GeometrySticker，将二进制版权信息"贴"在NeRF的**几何组件**（而非颜色组件）上，使得即使NeRF被重着色（recolorization），原始创建者仍能从渲染图像中提取水印来主张所有权。

**[GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image](3d_vision/geowizard_unleashing_the_diffusion_priors_for_3d_geometry_estimation_from_a_sing.md)**

:   本文提出GeoWizard，一个基于Stable Diffusion先验的几何估计基础模型，通过几何切换器（Geometry Switcher）实现单一模型联合预测深度和法线，并通过场景分布解耦策略（Scene Distribution Decoupler）消除混合场景布局的歧义，在零样本深度和法线基准上达到SOTA。

**[Global-to-Pixel Regression for Human Mesh Recovery](3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)**

:   提出一种从全局特征到像素级特征的两阶段回归框架，通过自适应2D关键点引导的局部编码模块捕获细粒度身体部位信息，并引入动态匹配策略改善视觉-网格对齐，在Human3.6M和3DPW上取得SOTA。

**[GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding](3d_vision/gpsformer_a_global_perception_and_local_structure_fitting-based_transformer_for_.md)**

:   提出GPSFormer，通过全局感知模块(GPM)学习点云短程和长程依赖，结合Taylor级数启发的局部结构拟合卷积(LSFConv)精确捕获局部几何细节，在ScanObjectNN上以纯监督学习方式达到95.4%准确率，超越所有预训练方法。

**[GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](3d_vision/grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)**

:   提出GRM，一种基于纯Transformer架构的前馈式3D重建模型，将稀疏视图(4张图)的像素通过pixel-aligned Gaussians转化为稠密的3D高斯表示，约0.1秒完成重建，结合多视图扩散模型可实现文本/图像到3D生成。

**[GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](3d_vision/gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)**

:   本文提出GS-LRM，一个极其简洁的基于Transformer的大规模重建模型，将多视角图像patch化后通过自注意力直接回归逐像素3D高斯参数，在物体级（超Triplane-LRM 4dB PSNR）和场景级（超pixelSplat 2.2dB PSNR）重建中均大幅超越SOTA，单张A100上0.23秒完成推理。

**[GVGEN: Text-to-3D Generation with Volumetric Representation](3d_vision/gvgen_text-to-3d_generation_with_volumetric_representation.md)**

:   提出GVGEN，首个直接从文本前馈生成3D高斯的框架，通过将无序高斯组织为结构化体积表示(GaussianVolume)，并设计从粗到精的生成管线（先生成几何体积再预测高斯属性），在约7秒内完成文本到3D生成。

**[HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](3d_vision/hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)**

:   利用结构化二值哈希网格为无序的3DGS锚点建立空间上下文关系，通过条件概率建模和自适应量化实现高效熵编码，达到相比vanilla 3DGS **75×** 的压缩率，同时保持甚至提升渲染质量。

**[HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](3d_vision/headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)**

:   提出HeadGaS，通过为每个3D高斯基元配备可学习的潜在特征基底，利用表情参数线性混合特征并经MLP预测表情相关的颜色和不透明度，实现**实时（250+ fps）**且高质量的可动画头部重建，PSNR超越基线约2 dB。

**[Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](3d_vision/heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)**

:   提出 3D-HetSGP 框架，将3D场景图预测建模为异构图学习问题，通过两阶段的异构图结构学习（HGSL）和异构图推理（HGR），解决了现有同构全连接图方法中不加区分的消息传递导致的次优性能问题。

**[Hiding Imperceptible Noise in Curvature-Aware Patches for 3D Point Cloud Attack](3d_vision/hiding_imperceptible_noise_in_curvature-aware_patches_for_3d_point_cloud_attack.md)**

:   提出 Wavelet Patches Attack（WPA）方法，利用小波变换分析点云的局部曲率结构，将对抗扰动隐藏在曲率一致的patch中——在平滑区域沿切平面扰动、在尖锐区域沿法向量扰动——实现比现有方法更不可感知的3D点云攻击。

**[High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](3d_vision/high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)**

:   提出RPrDepth，在训练阶段利用多帧/高分辨率等"富资源"模型的特征和预测作为先验，通过先验深度融合模块和富资源引导损失，使仅用**低分辨率单张图像**推理的模型达到甚至超过多帧高分辨率模型的深度估计精度。

**[High-Resolution and Few-shot View Synthesis from Asymmetric Dual-Lens Inputs](3d_vision/high-resolution_and_few-shot_view_synthesis_from_asymmetric_dual-lens_inputs.md)**

:   本文提出 DL-GS（Dual-Lens 3D-GS），利用移动设备上常见的非对称双镜头系统（广角+长焦）提供的立体几何约束和高分辨率引导，解决了 3D-GS 在少样本训练和超分辨率渲染上的两大难题，通过一致性感知训练策略和多参考引导细化模块实现了 SOTA 性能。

**[Human Hair Reconstruction with Strand-Aligned 3D Gaussians](3d_vision/human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)**

:   本文提出 Gaussian Haircut，通过经典发丝多段线和 3D 高斯基元的双表示（strand-aligned 3D Gaussians），结合 3D 方向场提升和粗到细的发丝拟合优化策略，从多视角图像重建出可直接用于图形引擎编辑、渲染和物理仿真的高保真发丝级发型，速度比之前方法快 10 倍以上。

**[Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM](3d_vision/hyperion_-_a_fast_versatile_symbolic_gaussian_belief_propagation_framework_for_c.md)**

:   本文提出Hyperion，一个基于SymForce符号计算框架自动生成超高效B/Z样条实现的连续时间高斯置信传播（GBP）SLAM框架，在运动跟踪和定位场景中达到与传统NLLS求解器（Ceres）相当的精度，同时天然支持分布式多智能体推理。

**[I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](3d_vision/i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)**

:   提出I²-SLAM，将物理成像过程（运动模糊建模+色调映射）集成到视觉SLAM系统中，通过HDR辐射场地图、多虚拟相机运动模糊模拟和可微分色调映射的联合优化，从手持随意拍摄的退化视频中重建出清晰的HDR 3D地图和更精确的相机轨迹。

**[IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation](3d_vision/idol_unified_dual-modal_latent_diffusion_for_human-centric_joint_video-depth_gen.md)**

:   提出IDOL框架，通过统一双模态U-Net和运动一致性损失，实现以人为中心的视频与深度图联合生成，显著优于现有方法。

**[Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](3d_vision/implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)**

:   提出一种非线性隐式滤波器，在不需要法线的情况下对神经SDF的隐式场进行平滑同时保留尖锐几何细节，并通过扩展到非零等值面实现全场一致性正则化。

**[Improving 2D Feature Representations by 3D-Aware Fine-Tuning](3d_vision/improving_2d_feature_representations_by_3d-aware_fine-tuning.md)**

:   通过将2D基础模型特征提升到3D Gaussian表示中实现多视角融合，再用渲染的3D感知特征反向微调2D模型，以线性探测即可提升语义分割和深度估计性能。

**[Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](3d_vision/improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)**

:   提出 SCAT 框架，通过缩放深度网络（SDN）降低 UNet 跳跃连接对扰动的敏感性，并引入冲突梯度手术（CGS）解决对抗增强导致的双重优化冲突，首次将对抗数据增强成功应用于自监督单目深度估计以提升跨域泛化能力。

**[Interactive 3D Object Detection with Prompts](3d_vision/interactive_3d_object_detection_with_prompts.md)**

:   提出"2D提示，3D检测"+"3D检测，3D精化"的多模态交互式 3D 目标检测框架，通过简单的 2D 交互提示（点击或框选）桥接 2D-3D 复杂性差距，并支持迭代精化，大幅降低 3D 标注成本，在 nuScenes 上验证了有效性且展示了出色的开放集能力。

**[Invertible Neural Warp for NeRF](3d_vision/invertible_neural_warp_for_nerf.md)**

:   提出用可逆神经网络（INN）过参数化相机位姿的刚性变换函数，在 NeRF 联合优化中显著提升位姿估计精度和重建质量，证明可逆性是 MLP 建模刚性 warp 的关键约束。

**[JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation](3d_vision/jointdreamer_ensuring_geometry_consistency_and_text_congruence_in_text-to-3d_gen.md)**

:   提出联合分数蒸馏（JSD），通过能量函数建模多视图去噪图像的联合分布，将 SDS 从单视图独立优化扩展为多视图联合优化，有效解决 3D 生成中的 Janus 问题，同时保持对复杂文本的生成保真度。

**[Lagrangian Hashing for Compressed Neural Field Representations](3d_vision/lagrangian_hashing_for_compressed_neural_field_representations.md)**

:   将InstantNGP的欧拉网格哈希表与拉格朗日点云表示相结合，在哈希桶中存储可移动的高斯特征点，实现**参数量减少1.8-2.8倍**但重建质量不降的紧凑神经场表示。

**[Language-Driven 6-DoF Grasp Detection Using Negative Prompt Guidance](3d_vision/language-driven_6-dof_grasp_detection_using_negative_prompt_guidance.md)**

:   提出大规模语言驱动6-DoF抓取数据集Grasp-Anything-6D（1M场景、200M抓取姿态），以及基于扩散模型的LGrasp6D方法，核心创新是**负提示引导（Negative Prompt Guidance）**策略，在推理时引导抓取姿态远离非目标物体。

**[LaRa: Efficient Large-Baseline Radiance Fields](3d_vision/lara_efficient_large-baseline_radiance_fields.md)**

:   提出LaRa前馈重建模型，通过**高斯体积（Gaussian Volume）**表示和**分组注意力层（Group Attention Layer）**统一局部与全局推理，仅需4张图像即可从大基线视角重建360°辐射场，且仅用**4×A100训练2天**即可超越LGM等费时方法。

**[Learning 3D-Aware GANs from Unposed Images with Template Feature Field](3d_vision/learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)**

:   提出模板特征场(TeFF)，通过联合学习生成辐射场和语义特征场，从无姿态标注的野外图像中自动提取3D模板并在线估计相机位姿，从而实现完整3D几何的生成对抗学习。

**[Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](3d_vision/learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)**

:   提出 GScream 框架，通过单目深度引导训练和交叉注意力特征正则化，在 3D Gaussian Splatting 表示下实现高质量的物体移除，同时保持几何一致性和纹理连贯性。

**[Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation](3d_vision/learning_to_generate_conditional_tri-plane_for_3d-aware_expression_controllable_.md)**

:   提出 Export3D，通过对比预训练获取与外观解耦的表情表示（CLeBS），结合表情自适应层归一化（EAdaLN）直接生成条件tri-plane，实现无外观交换的跨身份3D-aware人像表情动画。

**[LEIA: Latent View-Invariant Embeddings for Implicit 3D Articulation](3d_vision/leia_latent_view-invariant_embeddings_for_implicit_3d_articulation.md)**

:   提出LEIA方法，通过学习视角不变的潜在嵌入来表征铰接物体的不同状态，利用超网络(HyperNetwork)调制NeRF权重，实现在未见过的铰接配置之间进行平滑插值，无需任何运动先验或3D监督。

**[LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](3d_vision/lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)**

:   本文提出LGM，一个基于非对称U-Net架构的多视角3D高斯重建模型，从4张正交视角图像预测65536个3D高斯原语，在512分辨率下5秒内完成从文本/图像到高分辨率3D模型的生成，通过数据增强策略弥合训练-推理域差异。

**[LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](3d_vision/ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)**

:   提出LN3Diff++框架，通过3D感知的VAE将多视角图像压缩到紧凑的3D潜在空间，在该空间上训练扩散模型（U-Net或DiT），实现高质量、快速、通用的条件3D生成，包括文本到3D和图像到3D。

**[MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References](3d_vision/mariner_enhancing_novel_views_by_matching_rendered_images_with_nearby_references.md)**

:   提出MaRINeR方法，利用附近参考图像通过深度特征匹配和层次化细节传输来增强3D重建的渲染图像质量，适用于显式（mesh）和隐式（NeRF）等多种3D表示的渲染后处理。

**[MegaScenes: Scene-Level View Synthesis at Scale](3d_vision/megascenes_scene-level_view_synthesis_at_scale.md)**

:   从Wikimedia Commons互联网照片构建包含10万+SfM重建的大规模场景级3D数据集MegaScenes，并结合warp条件和位姿条件提升场景级新视角合成的位姿一致性。

**[Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](3d_vision/mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)**

:   提出Mesh2NeRF，通过解析解直接从纹理网格(textured mesh)构造GT辐射场，用occupancy函数建模密度场、用反射模型建模颜色场，为NeRF表示与生成任务提供精确的3D逐点监督。

**[MeshFeat: Multi-Resolution Features for Neural Fields on Meshes](3d_vision/meshfeat_multi-resolution_features_for_neural_fields_on_meshes.md)**

:   提出MeshFeat，一种适用于mesh上神经场的参数化多分辨率特征编码方法，利用网格简化算法构建多分辨率特征表示，在保持重建质量的同时实现13倍推理加速。

**[milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing](3d_vision/milliflow_scene_flow_estimation_on_mmwave_radar_point_cloud_for_human_motion_sen.md)**

:   提出首个毫米波雷达点云场景流估计方法 milliFlow，通过多尺度特征提取、全局聚合、GRU 时序传播和约束回归，在自建数据集上将 EPE3D 从次优 0.107m 降至 0.046m（cm 级精度），并展示场景流特征对人体活动识别（+7.9%）、人体部位解析（+3.6%）、人体追踪等下游任务的增强效果。

**[Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)**

:   Multi-HMR是首个单阶段多人全身（含手部和面部表情）3D人体网格恢复方法，使用ViT骨干网络和交叉注意力预测头（HPH），结合新的CUFFS合成数据集解决手部姿态学习困难，在多人和全身两类基准上同时达到SOTA。

**[MVDD: Multi-View Depth Diffusion Models](3d_vision/mvdd_multi-view_depth_diffusion_models.md)**

:   提出MVDD，一个基于多视角深度图表示的扩散模型，通过极线"线段"注意力和去噪深度融合实现3D一致的高质量形状生成，可生成20K+点的稠密点云。

**[MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](3d_vision/mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)**

:   MVDiffusion++提出了一种无需相机位姿的多视图潜在扩散模型，通过"无位姿架构"和"视图丢弃训练策略"两个简洁的想法，从单张或少量输入图像生成密集（32张）高分辨率（512×512）的多视图图像，进而实现高质量3D物体重建。

**[MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](3d_vision/mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)**

:   将MVS的代价体深度估计与3D高斯溅射结合，通过混合渲染(splatting+volume rendering)提升泛化性，并提出基于多视图几何一致性的点云聚合策略，使per-scene优化仅需45秒就超越3D-GS的10分钟效果。

**[MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](3d_vision/mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)**

:   提出MVSplat，通过plane-sweep构建代价体（cost volume）来精确定位Gaussian中心，以极少参数量（pixelSplat的1/10）和最快推理速度（22fps）实现了稀疏视角前馈式3D Gaussian预测的SOTA。

**[NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](3d_vision/ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)**

:   提出NGP-RT，通过轻量注意力机制聚合多级显式哈希特征替代per-point MLP，并引入占用距离网格减少光线行进中的内存访问，在Mip-NeRF 360数据集上实现1080p 108fps的实时NeRF渲染。

**[NOVUM: Neural Object Volumes for Robust Object Classification](3d_vision/novum_neural_object_volumes_for_robust_object_classification.md)**

:   提出 NOVUM 架构，为每个物体类别维护一个由 3D 高斯组成的神经体积表征，通过将图像特征与各类别的高斯特征匹配实现分类，在遮挡/损坏/真实 OOD 场景下相比 ResNet/ViT/Swin 等标准架构分类准确率提升 6-33%，同时支持 3D 位姿估计和可解释性可视化。

**[nuCraft: Crafting High Resolution 3D Semantic Occupancy for Unified 3D Scene Understanding](3d_vision/nucraft_crafting_high_resolution_3d_semantic_occupancy_for_unified_3d_scene_unde.md)**

:   本文构建了基于nuScenes的高精度3D语义占用数据集nuCraft（分辨率达0.1m体素、8倍于现有benchmark），并提出VQ-Occ方法利用VQ-VAE将占用数据编码到紧凑潜在空间中进行预测，首次实现了无需后处理上采样的高分辨率语义占用直接生成。

**[Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](3d_vision/omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)**

:   提出Omni-Recon框架，通过基于图像的渲染（IBR）管线构建通用NeRF，利用解耦的几何/外观双分支设计，首次在单一模型中实现可泛化3D重建、零样本多任务场景理解和实时渲染、场景编辑等多种下游3D任务的适配。

**[Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation](3d_vision/omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)**

:   构建了 **Omni6D**——首个大规模类别级 6DoF 姿态估计 RGBD 数据集，覆盖 **166 个类别、4688 个实例、80 万张图像**，远超现有 NOCS 等数据集（仅 6 类），并提出对称感知评估指标和渐进式微调策略。

**[Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models](3d_vision/open-vocabulary_3d_semantic_segmentation_with_text-to-image_diffusion_models.md)**

:   提出 Diff2Scene，首次将预训练的文本-图像扩散模型（Stable Diffusion）用于开放词汇3D语义分割，通过创新的掩码蒸馏方法将2D基础模型的语义丰富mask嵌入迁移到3D几何感知mask模型，在 ScanNet200 上超越 SOTA 12%。

**[Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation](3d_vision/open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)**

:   提出 GGSD 框架，利用3D几何先验（超点语义一致性）引导从2D模型到3D模型的知识蒸馏，并通过自蒸馏机制进一步挖掘3D数据的表征优势，在室内外开放词汇3D场景理解任务上大幅超越现有方法。

**[P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](3d_vision/p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)**

:   提出 P2P-Bridge，将点云去噪建模为 Schrödinger Bridge 问题，学习噪声点云到干净点云之间的最优传输计划，首次引入数据到数据（而非数据到噪声）的扩散范式，在合成数据和真实室内场景（ScanNet++、ARKitScenes）上均大幅超越现有方法。

**[PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](3d_vision/pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)**

:   提出 PCF-Lift，通过概率特征嵌入（多元高斯分布）替代确定性特征，结合概率乘积核（PP Kernel）的对比损失和跨视图约束，有效应对2D分割中的不一致分割和不一致ID问题，在 ScanNet 和 Messy Room 数据集上显著超越前沿方法。

**[Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](3d_vision/per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)**

:   提出基于逐高斯嵌入（Per-Gaussian Embedding）的形变表示方法，将形变定义为逐高斯潜在嵌入与时间嵌入的函数，辅以粗细形变分解和局部平滑正则化，在多个动态场景数据集上取得了质量、速度和模型容量的全面优势。

**[Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](3d_vision/pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)**

:   Pixel-GS通过在3DGS的点云生长判定条件中引入像素覆盖数量作为梯度加权因子，解决了大高斯体在初始点云稀疏区域无法有效分裂的问题，同时通过距离感知的梯度缩放抑制相机附近浮点伪影的产生。

**[PointLLM: Empowering Large Language Models to Understand Point Clouds](3d_vision/pointllm_empowering_large_language_models_to_understand_point_clouds.md)**

:   将点云编码器（Point-BERT）通过 MLP 投影层对接 LLaMA 大语言模型，构建 PointLLM；利用 730K 指令数据（660K 简述 + 70K 复杂指令）两阶段训练后，在 3D 物体分类上达到 53.4% 生成式准确率（超越 LLaVA-13B 的 44.2%），在物体描述任务上人类评估胜率 55%（超越人工标注）。

**[ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](3d_vision/prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)**

:   提出一种概率融合框架 ProDepth，通过辅助解码器推断动态区域不确定性，以加权几何均值自适应融合单帧和多帧深度概率分布来修正代价体中的错误匹配代价，并配合不确定性感知的损失重加权策略，在自监督多帧单目深度估计中取得 SOTA。

**[Progressive Classifier and Feature Extractor Adaptation for Unsupervised Domain Adaptation on Point Clouds](3d_vision/progressive_classifier_and_feature_extractor_adaptation_for_unsupervised_domain_.md)**

:   提出 PCFEA 方法用于点云无监督域自适应，通过渐进构建从源域到目标域的中间域，在宏观层面用目标风格特征增强训练分类器（PTFA），微观层面引导特征提取器向中间域对齐（IDFA），在 PointDA-10 上均值准确率达 76.5%（超 SOTA +2.9%），GraspNetPC-10 上达 87.6%（超 SOTA +13.7%）。

**[Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model](3d_vision/protecting_nerfsapos_copyright_via_plug-and-play_watermarking_base_model.md)**

:   提出 NeRFProtector，利用预训练的水印基础模型（message extractor）以即插即用方式在 NeRF 创建过程中同步嵌入二进制水印，通过渐进式全局渲染（PGR）将水印知识蒸馏到 NeRF 表示中，无需修改 NeRF 架构即可实现高比特精度的版权保护。

**[Ray-Distance Volume Rendering for Neural Scene Reconstruction](3d_vision/ray-distance_volume_rendering_for_neural_scene_reconstruction.md)**

:   提出 RS-Recon 方法，用射线方向相关的有符号射线距离函数（SRDF）替代传统 SDF 来参数化体渲染中的密度函数，结合 SRDF-SDF 一致性损失和自监督可见性任务，在多物体室内场景重建中取得更准确的表面和视图合成。

**[Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians](3d_vision/reconstruction_and_simulation_of_elastic_objects_with_spring-mass_3d_gaussians.md)**

:   提出 Spring-Gaus，将可学习的 3D 弹簧-质点模型集成到 3D Gaussian Splatting 中，从多视角视频重建弹性物体的外观、几何和物理动力学参数，支持未来预测和不同条件下的仿真。

**[Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation](3d_vision/reliable_spatial-temporal_voxels_for_multi-modal_test-time_adaptation.md)**

:   本文提出 Latte（ReLiable Spatial-temporal Voxels），一种多模态测试时适应方法，通过滑动窗口帧聚合构建时空体素（ST voxels）并计算时空熵（ST entropy）来评估预测可靠性，进而实现自适应跨模态学习，在三个 MM-TTA 基准上取得 SOTA 性能。

**[Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting](3d_vision/repaint123_fast_and_high-quality_one_image_to_3d_generation_with_progressive_con.md)**

:   Repaint123 提出了一种渐进式可控重绘策略，用 2D 扩散模型生成多视角一致的高质量图像，再通过简单的 MSE 损失快速优化 3D 表征，仅需 2 分钟即可从单张图像生成纹理精细、多视角一致的 3D 内容，大幅超越基于 SDS 的方法。

**[RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation](3d_vision/risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)**

:   提出 RISurConv，通过构建局部三角表面并提取高表达力旋转不变表面属性（RISP），结合注意力增强卷积，实现首次在精度上超越非旋转不变方法的旋转不变点云分析网络。

**[RoGUENeRF: A Robust Geometry-Consistent Universal Enhancer for NeRF](3d_vision/roguenerf_a_robust_geometry-consistent_universal_enhancer_for_nerf.md)**

:   本文提出RoGUENeRF，一种结合3D重投影对齐、非刚性光流精炼和几何感知注意力的NeRF后处理增强器，能在保持视角一致性的同时显著提升多种NeRF方法的图像渲染质量，且对相机标定误差具有鲁棒性。

**[S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](3d_vision/s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)**

:   提出 S³D-NeRF，利用分层面部外观编码器、跨模态面部形变场和唇音同步判别器，实现了仅需一张图片即可由语音驱动生成高保真说话头视频的 NeRF 方法，在视频质量和唇音同步方面超越了现有单图方法。

**[SAGS: Structure-Aware 3D Gaussian Splatting](3d_vision/sags_structure-aware_3d_gaussian_splatting.md)**

:   提出 SAGS，通过局部-全局图表示和图神经网络隐式编码场景几何结构，在保持实时渲染的同时提升3DGS的渲染质量、减少存储需求（最高24×压缩），并显著抑制浮点伪影。

**[Sapiens: Foundation for Human Vision Models](3d_vision/sapiens_foundation_for_human_vision_models.md)**

:   Sapiens 提出了一个以人为中心的视觉基础模型家族（0.3B-2B参数），通过在3亿张人体图像上进行 MAE 自监督预训练，原生支持1K高分辨率推理，在2D姿态估计、身体部位分割、深度估计和表面法线预测四个人体视觉任务上全面超越现有SOTA。

**[SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer](3d_vision/sc4d_sparse-controlled_video-to-4d_generation_and_motion_transfer.md)**

:   SC4D提出了一种基于稀疏控制点的视频到4D生成框架，通过将动态3D物体的运动和外观解耦为稀疏控制点（~512个）和密集高斯体（~50k个），结合自适应高斯初始化（AG）和高斯对齐损失（GA）解决形状退化问题，并实现了基于控制点运动的跨实体运动迁移应用。

**[ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities](3d_vision/scanreason_empowering_3d_visual_grounding_with_reasoning_capabilities.md)**

:   提出 3D reasoning grounding 新任务和 ScanReason 基准（10K+ QA-location pairs，5种推理类型），设计 ReGround3D 框架将 MLLM 推理与 3D grounding 模块通过 Chain-of-Grounding 机制协同，在隐式指令下实现准确的 3D 目标定位。

**[ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention](3d_vision/scatterformer_efficient_voxel_transformer_with_scattered_linear_attention.md)**

:   提出 ScatterFormer，首个直接对跨窗口的变长体素序列施加线性注意力的体素 Transformer，通过 Scattered Linear Attention (SLA) 模块和 chunk-wise 矩阵乘法算法实现亚毫秒级延迟，配合 Cross-Window Interaction (CWI) 模块替代窗口平移，在 Waymo 和 nuScenes 上达到 SOTA 精度的同时保持 23 FPS 的检测速度。

**[SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](3d_vision/scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)**

:   提出 SceneGraphLoc，将查询图像在由多模态 3D 场景图组成的参考地图中进行粗定位，在不依赖大规模图像数据库的前提下，实现了接近 SOTA 图像级方法的定位精度，同时存储需求降低三个数量级。

**[SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](3d_vision/sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)**

:   提出首个百万级 3D 视觉-语言数据集 SceneVerse（68K 室内场景 + 2.5M 场景-语言对），结合多层级对比预训练框架 GPS，在 3D visual grounding 和 QA 任务上取得 SOTA，并展现零样本迁移能力。

**[SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](3d_vision/sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)**

:   提出 SEDiff，首次利用扩散模型提取域不变的结构信息，通过结构一致的风格迁移消除合成数据与真实数据之间的域差距，实现了高性能的域自适应单目深度估计。

**[SEED: A Simple and Effective 3D DETR in Point Clouds](3d_vision/seed_a_simple_and_effective_3d_detr_in_point_clouds.md)**

:   SEED 提出了一种简洁高效的 3D DETR 检测器，通过双重查询选择（DQS）模块以粗到精方式获取高质量查询，结合可变形网格注意力（DGA）模块利用 3D 物体的几何结构信息实现灵活的查询交互，在 Waymo 和 nuScenes 上达到新 SOTA。

**[SegPoint: Segment Any Point Cloud via Large Language Model](3d_vision/segpoint_segment_any_point_cloud_via_large_language_model.md)**

:   提出 SegPoint，首个利用多模态 LLM 推理能力在统一框架中完成 3D 指令分割、引用分割、语义分割和开放词汇分割四种任务的模型，并构建 Instruct3D 基准测试（2,565 对），mIoU 达 27.5%。

**[SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation](3d_vision/semantichuman-hd_high-resolution_semantic_disentangled_3d_human_generation.md)**

:   提出SemanticHuman-HD，首个实现语义解耦的3D人体图像合成方法，通过K个独立局部生成器和3D感知超分模块，实现1024²分辨率的语义可控人体生成。

**[SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)**

:   提出SGS-SLAM，首个基于Gaussian Splatting的语义视觉SLAM系统，通过多通道优化融合外观、几何和语义特征，在相机姿态估计、地图重建和语义分割方面均达到SOTA。

**[ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](3d_vision/shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)**

:   提出一种基于掩码扩散训练策略的3D网格局部编辑方法ShapeFusion，通过在顶点空间直接操作实现完全局部化、可解释的3D形状编辑，无需潜在空间优化。

**[SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark](3d_vision/signavatars_a_large-scale_3d_sign_language_holistic_motion_dataset_and_benchmark.md)**

:   提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。

**[SINDER: Repairing the Singular Defects of DINOv2](3d_vision/sinder_repairing_the_singular_defects_of_dinov2.md)**

:   揭示DINOv2特征图中高范数缺陷token的根源是网络权重的主左奇异向量（singular defect），并提出SINDER——仅需小数据集微调奇异值即可修复缺陷，同时保持特征质量。

**[SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](3d_vision/slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)**

:   提出SlotLifter，通过将2D特征提升为3D并与Slot Attention结合的slot-guided feature lifting设计，在场景分解和新视角合成上同时达到SOTA，且训练效率提升约5倍。

**[SparseSSP: 3D Subcellular Structure Prediction from Sparse-View Transmitted Light Images](3d_vision/sparsessp_3d_subcellular_structure_prediction_from_sparse-view_transmitted_light.md)**

:   提出 SparseSSP，一种混合维度拓扑的高效框架，通过 Z 轴深度到通道变换将 3D 亚细胞结构预测转化为 2D 网络任务，最多减少 87.5% 的成像频次同时保持领先精度。

**[SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](3d_vision/splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)**

:   SplatFields发现3D高斯溅射（3DGS）在稀疏视图设置下的性能瓶颈源于splat特征缺乏空间自相关性，提出通过隐式神经场预测splat特征来引入空间正则化，在静态3D和动态4D的稀疏重建场景中一致提升了重建质量。

**[SuperGaussian: Repurposing Video Models for 3D Super Resolution](3d_vision/supergaussian_repurposing_video_models_for_3d_super_resolution.md)**

:   提出SuperGaussian，通过复用预训练视频上采样模型实现3D超分辨率，无需类别特定训练，可处理多种3D输入格式（高斯、NeRF、网格等），输出高质量Gaussian Splat模型。

**[Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints](3d_vision/surface_reconstruction_from_3d_gaussian_splatting_via_local_structural_hints.md)**

:   针对3DGS在表面重建质量差的问题，提出利用单目法向/深度先验来增强高斯原语的几何组织性，并通过移动最小二乘（MLS）构建局部符号距离场，再联合学习神经隐式网络进行正则化，显著提升了3DGS的表面重建精度。

**[T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning](3d_vision/t-mae_temporal_masked_autoencoders_for_point_cloud_representation_learning.md)**

:   T-MAE 提出了一种时序掩码自编码器预训练策略，以时序相邻两帧为输入，通过掩码当前帧并借助历史帧信息重建来学习时序依赖关系，配合设计的 SiamWCA（孪生编码器+窗口交叉注意力）架构，在 Waymo 和 ONCE 数据集上以更少标注数据和更少训练迭代次数超越 SOTA 自监督方法。

**[TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](3d_vision/talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)**

:   提出TalkingGaussian，基于3D高斯溅射的形变驱动说话人头部合成框架，通过对持久性高斯基元施加平滑形变表示面部运动，并分解面部和口腔内部区域以解决运动不一致问题。

**[MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](3d_vision/taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)**

:   提出MALD-NeRF，通过掩码对抗训练和场景定制的潜在扩散模型实现高质量NeRF修复，有效解决扩散模型的多视角不一致和纹理偏移问题。

**[TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection](3d_vision/tcc-det_temporarily_consistent_cues_for_weakly-supervised_3d_detection.md)**

:   本文提出一种完全不需要人工3D标注的弱监督3D目标检测方法，通过利用现成的2D检测器（Mask-RCNN）和多帧时间一致性线索生成高质量伪3D标签，然后用于训练3D点云检测器（Voxel-RCNN），在KITTI和Waymo上超越所有弱监督方法并显著缩小与全监督方法的差距。

**[TC-Stereo: Temporally Consistent Stereo Matching](3d_vision/temporally_consistent_stereo_matching.md)**

:   提出TC-Stereo，通过时序视差补全提供良好初始化、时序状态融合保持隐藏态连贯性，以及双空间（视差+视差梯度）迭代精炼改善病态区域，实现时间一致的立体匹配。

**[Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)**

:   提出Texture-GS，首次为3D高斯溅射解耦几何与纹理，通过UV映射MLP和局部Taylor展开将外观表示为2D纹理图，实现实时纹理替换和编辑（58 FPS，RTX 2080 Ti）。

**[The NeRFect Match: Exploring NeRF Features for Visual Localization](3d_vision/the_nerfect_match_exploring_nerf_features_for_visual_localization.md)**

:   提出NeRFMatch，探索NeRF内部特征作为3D描述子的潜力，建立基于注意力机制的2D-3D匹配网络，在Cambridge Landmarks上实现有竞争力的定位性能，验证了NeRF作为定位场景表示的可行性。

**[Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](3d_vision/thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)**

:   提出Thermal3D-GS，通过神经网络建模大气传输效应和热传导物理过程，并引入温度一致性约束，实现热红外图像的高质量新视角合成，创建了首个大规模热红外新视角合成数据集TI-NSD。

**[TPA3D: Triplane Attention for Fast Text-to-3D Generation](3d_vision/tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)**

:   提出TPA3D，一个基于GAN的文本引导3D生成框架，通过三平面注意力（TPA）模块在句子级和词级特征上进行逐层细化，实现快速且细粒度的文本到3D纹理网格生成。

**[Track Everything Everywhere Fast and Robustly](3d_vision/track_everything_everywhere_fast_and_robustly.md)**

:   提出一种高效鲁棒的测试时优化像素跟踪方法，通过引入CaDeX++可逆变形网络、单目深度先验和DINOv2长期语义一致性，将训练速度提升10倍以上，同时显著提高了跟踪精度和鲁棒性。

**[TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](3d_vision/tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)**

:   提出TrackNeRF，将SfM中的特征轨迹（feature tracks）引入NeRF训练，通过全局多视角重投影一致性损失替代传统的成对对应损失，显著提升稀疏+有噪声位姿下的NeRF重建质量和位姿优化精度。

**[TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](3d_vision/tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)**

:   提出TRAM，一个两阶段方法，通过鲁棒化SLAM恢复度量尺度相机运动 + 视频Transformer（VIMO）回归相机坐标系下的人体运动，组合两者实现准确的世界坐标系3D人体全局轨迹与动作重建。

**[Transferable 3D Adversarial Shape Completion using Diffusion Models](3d_vision/transferable_3d_adversarial_shape_completion_using_diffusion_models.md)**

:   提出3DAdvDiff，利用3D扩散模型通过对抗性形状补全生成高质量的迁移性3D对抗点云，结合模型不确定性、集成对抗引导和显著性评分策略，在黑盒设置下对最新3D模型实现SOTA攻击成功率。

**[UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](3d_vision/unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)**

:   提出UniDream，通过训练albedo-法线对齐的多视角扩散模型（AN-MVM），结合Transformer重建模型和分阶段SDS优化，实现可重光照的文本到3D生成，生成的3D物体具有干净的albedo纹理和PBR材质。

**[VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](3d_vision/vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)**

:   提出VCD-Texture，在Stable Diffusion去噪过程中统一2D和3D自注意力学习（JNP），通过方差对齐（VA）解决光栅化引起的方差衰减问题，并用修复细化处理不一致区域，实现高保真、高一致性的3D纹理合成。

**[VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](3d_vision/versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)**

:   本文提出 VersatileGaussian，通过为 3D 高斯赋予共享多任务特征并设计任务相关注意力（Task Correlation Attention）模块实现跨任务信息流动，在 ScanNet 和 Replica 数据集上同时达到了多任务标签预测的 SOTA 精度和 35 FPS 的实时渲染速度。

**[View Selection for 3D Captioning via Diffusion Ranking](3d_vision/view_selection_for_3d_captioning_via_diffusion_ranking.md)**

:   提出DiffuRank方法，利用预训练text-to-3D扩散模型（Shap·E）对3D物体渲染视角进行对齐度评分和排序，选出最具代表性的Top-6视角送入GPT4-Vision生成高质量字幕，修正Cap3D中约200k错误标注并扩展至150万条字幕。

**[Vista3D: Unravel the 3D Darkside of a Single Image](3d_vision/vista3d_unravel_the_3d_darkside_of_a_single_image.md)**

:   提出Vista3D，通过粗到细的两阶段框架（高斯溅射→FlexiCubes等值面细化+解耦纹理），结合角度扩散先验组合，5分钟内从单张图像生成多样且一致的高保真3D网格。

**[WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](3d_vision/wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)**

:   提出WaSt-3D，利用3D高斯溅射表示将风格迁移重新定义为两个高斯分布之间的最优传输问题，通过Sinkhorn散度匹配内容场景和风格场景的3D分布，首次实现了3D场景到场景的几何风格迁移。

**[When Do We Not Need Larger Vision Models?](3d_vision/when_do_we_not_need_larger_vision_models.md)**

:   提出 Scaling on Scales (S2) 策略：冻结小模型（如 ViT-B）在多个图像尺度上运行并拼接特征，无需增加参数即可在分类、分割、深度估计、MLLM 等任务上匹敌甚至超越大模型（ViT-H/G），并从理论和实验上论证了大模型学到的表征大部分可由多尺度小模型线性近似。

**[WordRobe: Text-Guided Generation of Textured 3D Garments](3d_vision/wordrobe_text-guided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。

**[Zero-Shot Multi-Object Scene Completion](3d_vision/zero-shot_multi-object_scene_completion.md)**

:   提出OctMAE，一种融合Octree U-Net和隐空间3D MAE的混合架构，从单张RGB-D图像实现高质量近实时的多物体场景形状补全，通过遮挡掩码策略和3D旋转位置编码显著提升效率和泛化能力。

**[ZeST: Zero-Shot Material Transfer from a Single Image](3d_vision/zest_zero-shot_material_transfer_from_a_single_image.md)**

:   提出ZeST，一种零样本免训练的材质迁移方法，通过IP-Adapter提取材质表示、ControlNet提供几何引导、前景灰度图提供光照线索，三条分支组合实现从单张材质样本图像到目标物体的2D材质迁移。

---

## 🎨 图像生成 { #image_generation }

**[2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction](image_generation/2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)**

:   2S-ODIS通过两阶段结构利用预训练VQGAN（无需微调）合成全景图像：第一阶段生成低分辨率粗略ERP图，第二阶段通过生成26个NFoV局部图像并融合来校正几何畸变，训练时间从14天缩短到4天且图像质量更优。

**[A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks](image_generation/a_closer_look_at_gan_priors_exploiting_intermediate_features.md)**

:   提出 IF-GMI，将预训练 StyleGAN2 的生成器拆解为多个 block，在中间特征层逐层优化（配合 $\ell_1$ 球约束防止图像崩塌），把模型反演攻击的搜索空间从潜码扩展到中间特征，在 OOD 场景下攻击准确率提升高达 38.8%。

**[A Diffusion Model for Simulation Ready Coronary Anatomy with Morpho-skeletal Control](image_generation/a_diffusion_model_for_simulation_ready_coronary_anatomy_with.md)**

:   用潜在扩散模型（LDM）可控生成3D多组织冠状动脉分割图，通过拓扑交互损失保证解剖合理性，通过形态-骨架双通道条件化实现对截面形态和分支结构的解耦控制，并提出自适应空条件引导（ANG）以非可微回归器高效增强条件保真度，最终支持面向有限元仿真的反事实解剖结构编辑。

**[A High-Quality Robust Diffusion Framework for Corrupted Dataset](image_generation/a_highquality_robust_diffusion_framework_for_corrupted_datas.md)**

:   提出 RDUOT 框架，首次将非平衡最优传输(UOT)融入扩散模型(DDGAN)中，通过学习 $q(x_0|x_t)$ 而非 $q(x_{t-1}|x_t)$ 来有效过滤训练数据中的离群值，在污染数据集上实现鲁棒生成的同时，在干净数据集上也超越了 DDGAN 基线。

**[AccDiffusion: An Accurate Method for Higher-Resolution Image Generation](image_generation/accdiffusion_an_accurate_method_for_higher-resolution_image_generation.md)**

:   提出AccDiffusion，通过将全局文本prompt解耦为patch级别的内容感知prompt（利用cross-attention map判断每个词汇是否属于某patch），并引入带窗口交互的膨胀采样来改善全局一致性，在无需额外训练的情况下有效解决patch-wise高分辨率图像生成中的目标重复问题，在SDXL上实现了从2K到4K分辨率的无重复高质量图像外推。

**[AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)**

:   观察到扩散模型超分中不同图像区域所需去噪步数差异巨大（背景区域早已收敛而前景纹理仍需迭代），提出基于多指标潜在熵（MMLE）感知信息增益来动态跳步的策略，将子区域分为稳定/增长/饱和三类给予不同步长，并通过渐进特征注入（PFJ）平衡保真度与真实感，在DRealSR等数据集上取得与StableSR可比的质量但推理时间和FLOPs分别减少1.5×和2.7×。

**[AdaGen: Learning Adaptive Policy for Image Synthesis](image_generation/adagen_learning_adaptive_policy_for_image_synthesis.md)**

:   将多步生成模型（MaskGIT/AR/Diffusion/Rectified Flow）的步级参数调度（温度、mask ratio、CFG scale、timestep等）统一建模为MDP，用轻量RL策略网络实现样本自适应调度，并提出对抗奖励设计防止策略过拟合，在四种生成范式上一致提升性能（VAR FID 1.92→1.59，DiT-XL推理成本降3倍同时性能更优）。

**[AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](image_generation/adanat_exploring_adaptive_policy_for_token-based_image_generation.md)**

:   提出AdaNAT，将非自回归Transformer（NAT）的生成策略配置建模为MDP，通过轻量策略网络+PPO强化学习+对抗奖励模型自动为每个样本定制生成策略（重掩码比例、采样温度、CFG权重等），在ImageNet-256上仅用8步达到FID 2.86，相比手工策略实现约40%的相对提升。

**[AFreeCA: Annotation-Free Counting for All](image_generation/afreeca_annotation-free_counting_for_all.md)**

:   利用 Stable Diffusion 生成合成排序/计数数据，通过先学排序再学计数的两阶段策略 + 密度引导的图像分块，实现了首个适用于任意类别物体的无标注计数方法，在人群计数上超越已有无监督方法。

**[AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation](image_generation/anycontrol_create_your_artwork_with_versatile_control_on_text-to-image_generatio.md)**

:   提出 AnyControl，通过 Multi-Control Encoder（fusion + alignment 交替块结构）支持任意组合的多种空间控制信号（深度、边缘、分割、姿态），在 COCO 多控制基准上 FID 44.28 全面超越现有方法。

**[Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](image_generation/be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)**

:   Be Yourself深入分析了扩散模型中Cross-Attention和Self-Attention导致的多主体语义泄漏问题，提出Bounded Attention机制，通过在去噪过程中限制不同主体间的信息流动来生成语义独立的多主体图像，免训练即可生成5+个语义相似主体。

**[Beta-Tuned Timestep Diffusion Model](image_generation/beta-tuned_timestep_diffusion_model.md)**

:   本文对扩散模型前向过程进行了深入的理论分析，发现分布变化在早期阶段最为剧烈，据此提出 B-TTDM（Beta-Tuned Timestep Diffusion Model），使用 Beta 分布替代均匀分布进行时间步采样，使训练更好地对齐前向扩散过程的特性，在多个基准数据集上验证了其有效性。

**[Bridging the Gap: Studio-Like Avatar Creation from a Monocular Phone Capture](image_generation/bridging_the_gap_studio-like_avatar_creation_from_a_monocular_phone_capture.md)**

:   提出从单目手机视频生成类似影棚级质量的面部纹理贴图的方法，结合 StyleGAN2 的 W+ 空间参数化与扩散模型超分辨率，实现从手机扫描到高质量 3D 头像的跨越。

**[ByteEdit: Boost, Comply and Accelerate Generative Image Editing](image_generation/byteedit_boost_comply_and_accelerate_generative_image_editing.md)**

:   提出 ByteEdit，一个将人类反馈学习引入生成式图像编辑（inpainting/outpainting）的框架，通过美学、对齐、一致性三个奖励模型提升编辑质量，并利用对抗训练和渐进策略加速推理。

**[Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning](image_generation/challenging_forgets_unveiling_the_worst-case_forget_sets_in_machine_unlearning.md)**

:   提出从对抗视角识别"最坏情况遗忘集"的方法，通过双层优化框架找到最难被遗忘的数据子集，利用 SignSGD 将二阶 BLO 简化为一阶问题，从而更可靠地评估机器遗忘方法的真实效能。

**[COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation](image_generation/coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)**

:   提出COIN方法，通过控制-补绘（Control-Inpainting）的改进版Score Distillation Sampling，结合人-场景关系损失，从单目动态相机视频中同时估计高质量的全局人体运动和相机运动。

**[Collaborative Control for Geometry-Conditioned PBR Image Generation](image_generation/collaborative_control_for_geometry-conditioned_pbr_image_generation.md)**

:   提出 Collaborative Control 范式，通过冻结预训练RGB扩散模型并训练一个并行PBR模型，利用双向跨网络通信层联合建模RGB与PBR图像分布，在有限数据下实现高质量的几何条件PBR材质图像生成。

**[ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement](image_generation/colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)**

:   提出ColorPeel方法，通过在目标颜色的基本几何形状上学习颜色提示token（解耦颜色与形状），并引入交叉注意力对齐损失，使T2I扩散模型能精确生成用户指定RGB颜色的物体。

**[Controlling the World by Sleight of Hand](image_generation/controlling_the_world_by_sleight_of_hand.md)**

:   提出 CosHand，通过手部二值掩码作为动作条件，在预训练 Stable Diffusion 上微调，预测手-物交互后的未来图像，并可零样本泛化到机器人末端执行器。

**[DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](image_generation/dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)**

:   提出 DCDM（Diffusion-Conditioned-Diffusion Model），通过双扩散架构学习高分辨率场景文字图像的分布：第一个潜在扩散模型生成字符级文本嵌入作为条件，第二个扩散模型在此条件和低分辨率图像的联合引导下生成高清文字图像，在 TextZoom 和 Real-CE 数据集上超越 SOTA。

**[Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](image_generation/diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)**

:   提出 Diff-Tracker，首次利用预训练文本到图像扩散模型（Stable Diffusion）中蕴含的丰富视觉语义知识进行无监督目标跟踪，通过学习一个表示目标的 prompt 并在线更新来实现持续跟踪。

**[DiffiT: Diffusion Vision Transformers for Image Generation](image_generation/diffit_diffusion_vision_transformers_for_image_generation.md)**

:   提出 DiffiT（Diffusion Vision Transformer），通过引入时间依赖多头自注意力（TMSA）机制，让自注意力在去噪过程的不同阶段动态调整行为，在ImageNet-256上以比DiT/MDT少16-20%的参数量达到了1.73的SOTA FID分数。

**[Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](image_generation/diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)**

:   提出PIC（Prompt Interpolation-based Correction），一种无训练的扩散模型图像翻译方法，通过渐进式prompt嵌入插值构造噪声校正项，将其与源图像噪声预测线性组合，实现结构保持的高保真图像编辑，且推理速度（18.1s）优于所有对比方法。

**[Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](image_generation/diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)**

:   提出 DDDR 框架，首次将预训练扩散模型引入联邦类别持续学习（FCCL），通过 Federated Class Inversion 技术为每个类别学习一个紧凑的 class embedding，利用扩散模型高质量回放历史数据以对抗灾难性遗忘，并通过对比学习弥合生成数据与真实数据的域差距。

**[Distilling Diffusion Models into Conditional GANs](image_generation/distilling_diffusion_models_into_conditional_gans.md)**

:   提出 Diffusion2GAN 框架，将多步扩散模型蒸馏为单步条件GAN，核心创新是 E-LatentLPIPS 潜空间感知损失和基于预训练扩散模型的多尺度条件判别器，在零样本 COCO 基准上超越 DMD、SDXL-Turbo 和 SDXL-Lightning。

**[DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment](image_generation/dreamdiffusion_high-quality_eeg-to-image_generation_with_temporal_masked_signal_.md)**

:   本文提出 DreamDiffusion，利用时序掩码信号建模对EEG编码器进行大规模预训练学习鲁棒的脑电表征，再通过CLIP图像编码器提供额外监督将EEG-文本-图像空间对齐，最终借助预训练的Stable Diffusion从脑电信号直接生成高质量图像，实现便携低成本的"思维转图像"。

**[DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators](image_generation/dreamdrone_text-to-image_diffusion_models_are_zero-shot_perpetual_view_generator.md)**

:   提出DreamDrone——一个零样本、无需训练的无限飞行场景生成管线，通过直接对预训练扩散模型的中间latent code进行warping（而非图像级warping），结合特征对应引导和高通滤波策略，实现高质量、几何一致的无界场景生成。

**[DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](image_generation/dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)**

:   提出 DreamMover，基于预训练文本到图像扩散模型实现大运动图像对之间的插值，通过扩散感知光流估计、两级潜空间融合和自注意力拼接替换三个核心组件，生成语义一致的中间帧。

**[EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models](image_generation/ebdm_exemplar-guided_image_translation_with_brownian-bridge_diffusion_models.md)**

:   提出 EBDM 框架，将样例引导的图像翻译建模为随机布朗桥扩散过程，从结构控制直接翻译为真实感图像，通过 Global Encoder、Exemplar Network 和 Exemplar Attention Module 三个组件有效整合样例的全局风格和细节纹理信息。

**[EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)**

:   提出 EchoScene，一个基于双分支扩散模型的 3D 室内场景生成方法，通过信息回声（Information Echo）机制在场景图扩散过程中实现多个去噪过程间的协作信息交换，生成全局一致且可交互控制的场景。

**[Editable Image Elements for Controllable Synthesis](image_generation/editable_image_elements_for_controllable_synthesis.md)**

:   提出"可编辑图像元素"表示，将输入图像分解为一组语义对齐的 patch embeddings（类似超像素），每个 patch 关联位置和尺寸信息，用户可直接编辑这些属性（移动、缩放、删除），再由基于 Stable Diffusion 的解码器合成真实感图像。

**[EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](image_generation/emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)**

:   提出 EMDM，通过条件去噪扩散 GAN 捕获大步长下的复杂去噪分布，实现仅需不超过 10 步采样即可实时生成高质量人体动作，推理速度较 MDM 提升约 200 倍。

**[Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](image_generation/enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)**

:   提出 TexForce，通过强化学习（DDPO）结合 LoRA 微调扩散模型的文本编码器以提升图文对齐和视觉质量，并可无缝与已有 U-Net 微调方法组合获得更优效果。

**[Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models](image_generation/enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)**

:   提出 StableVSR，首次将扩散模型应用于视频超分辨率任务，通过时序条件模块（TCM）和帧级双向采样策略，在显著提升感知质量的同时确保帧间时序一致性。

**[Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing](image_generation/eta_inversion_designing_an_optimal_eta_function_for_diffusion-based_real_image_e.md)**

:   通过理论分析DDIM采样方程中η参数的作用，设计时间和区域依赖的η函数，实现更灵活精确的真实图像编辑。

**[FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction](image_generation/finematch_aspect-based_fine-grained_image_and_text_mismatch_detection_and_correc.md)**

:   提出 FineMatch 基准，定义了基于方面（Aspect）的细粒度图文不匹配检测与纠正任务，包含 49,906 个高质量人工标注的图文对，并展示了现有 VLM 在细粒度组合性理解上的不足。

**[FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)**

:   提出 FouriScale，从频域分析视角出发，通过膨胀卷积+低通滤波替换预训练扩散模型中的卷积层，实现免训练的任意尺寸高分辨率图像生成，理论上证明了膨胀卷积保持结构一致性的有效性。

**[FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior](image_generation/freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)**

:   提出 FreeCompose，利用预训练扩散模型的生成先验实现通用零样本图像合成，统一覆盖图像和谐化（外观编辑）和语义图像合成（语义编辑），无需额外训练。

**[FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models](image_generation/freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)**

:   从频率视角重新审视扩散模型的图像编辑过程，发现去噪网络优先恢复低频分量导致编辑引导与目标区域的misalignment，提出渐进式频率截断（FreeDiff）方法在频率空间精炼引导信号，实现免调优的通用图像编辑。

**[GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections](image_generation/garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)**

:   针对文本到服装图像生成中的细粒度语义错位（组件数量、位置和相互关系），提出 GarmentAligner，通过自动组件提取管线获取空间-数量信息，并结合检索增强对比学习和多级校正损失，实现服装组件的视觉、空间和数量级别的精确对齐。

**[Generating 3D House Wireframes with Semantics](image_generation/generating_3d_house_wireframes_with_semantics.md)**

:   提出基于自回归模型的 3D 房屋线框生成方法，采用统一的线段（wire）表示替代传统的顶点-边分离建模，通过语义感知的 BFS 序列排列和两阶段 coarse-to-fine Transformer 解码器生成语义丰富的线框结构，可自动分割为墙壁、屋顶、房间等语义组件。

**[Generating Human Interaction Motions in Scenes with Text Control](image_generation/generating_human_interaction_motions_in_scenes_with_text_control.md)**

:   提出 TeSMo，一个文本控制的场景感知运动生成方法，通过在大规模运动数据上预训练文本-运动扩散模型，再用增强的场景感知分支进行微调，分两阶段（导航+交互）生成角色在 3D 场景中避障行走并与物体交互（如坐下）的真实运动序列。

**[Getting it Right: Improving Spatial Consistency in Text-to-Image Models](image_generation/getting_it_right_improving_spatial_consistency_in_text-to-image_models.md)**

:   系统性调查文本到图像模型的空间关系生成缺陷，发现现有视觉-语言数据集严重缺乏空间描述，据此创建 SPRIGHT 数据集（~600 万张图像重标注空间关系），仅用 <500 张多物体图像微调即在 T2I-CompBench 空间得分上达到 SOTA（0.2133），相比基线提升 41%。

**[Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation](image_generation/harnessing_text-to-image_diffusion_models_for_category-agnostic_pose_estimation.md)**

:   提出 Prompt Pose Matching（PPM）框架，利用预训练文本到图像扩散模型中的丰富知识来解决类别无关姿态估计（CAPE），通过学习与关键点对应的伪提示（pseudo prompts）实现零训练基础类别的少样本关键点检测。

**[HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects](image_generation/himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)**

:   提出 HIMO，首个大规模全身人体与多物体交互的 4D MoCap 数据集（3.3K序列，4.08M帧），并附带详细的文本描述和时间段分割标注，提出双分支条件扩散模型及自回归管线，生成协调的多物体交互动作序列。

**[HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](image_generation/hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)**

:   提出 HybridBooth，一种两阶段混合 prompt inversion 框架，通过先用回归器生成初始词嵌入（Probe），再用残差微调（Refinement）仅需 3-5 步迭代即可高效完成主体驱动的个性化图像生成。

**[Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](image_generation/idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)**

:   提出幂等生成模型（IGM），从理论上建立生成模型与最大熵编码（谱对比学习）的等价关系，通过在骨架数据的特征空间施加幂等约束，使生成模型的特征更紧凑、更适合识别任务，在 NTU 60 xsub 上将准确率从 84.6% 提升至 86.2%。

**[Implicit Concept Removal of Diffusion Models](image_generation/implicit_concept_removal_of_diffusion_models.md)**

:   提出 Geom-Erasing 方法，通过引入外部分类器/检测器提供隐式概念的存在性和几何位置信息，将其编码为文本条件中的位置 token 并作为负提示使用，有效消除扩散模型中水印、不安全内容等"隐式概念"的生成，在 I2P 和自建 ICD 基准上达到 SOTA。

**[Implicit Style-Content Separation using B-LoRA](image_generation/implicit_style-content_separation_using_b-lora.md)**

:   提出 B-LoRA，通过分析 SDXL 架构发现仅联合训练两个特定 transformer block 的 LoRA 权重（Block 4 控制内容、Block 5 控制风格）即可隐式实现单张图片的风格-内容分离，支持风格迁移、文本风格化、一致风格生成等多种任务。

**[Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm](image_generation/infinite-id_identity-preserved_personalization_via_id-semantics_decoupling_parad.md)**

:   提出 Infinite-ID，通过 ID-语义解耦范式将身份信息和文本语义信息分离处理——训练阶段停用文本交叉注意力以专注学习身份嵌入，推理阶段通过混合注意力机制和 AdaIN-mean 操作融合两路信息，在单张参考图下同时实现高保真身份保持和语义一致性。

**[∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions](image_generation/inftybrush_controllable_large_image_synthesis_with_diffusion.md)**

:   提出首个在无限维函数空间中的条件扩散模型 ∞-Brush，通过交叉注意力神经算子实现可控条件生成，仅用 0.4% 像素训练即可在任意分辨率（最高 4096×4096）上生成保持全局结构的大图像。

**[IRGen: Generative Modeling for Image Retrieval](image_generation/irgen_generative_modeling_for_image_retrieval.md)**

:   将图像检索重新定义为生成式建模任务，提出 IRGen——一个序列到序列模型，通过语义图像分词器将图像转化为简短的离散 token 序列，然后自回归地生成查询图像最近邻的标识符，实现端到端可微分的检索并在三个标准基准上达到 SOTA。

**[L-DiffER: Single Image Reflection Removal with Language-Based Diffusion Model](image_generation/l-differ_single_image_reflection_removal_with_language-based_diffusion_model.md)**

:   提出 L-DiffER，一种语言引导的扩散模型，通过迭代条件精化策略解决控制条件不准确问题，结合多条件约束机制保证图像恢复的颜色和结构保真度，同时保留扩散模型的生成能力以处理低透射率反射。

**[Latent Guard: A Safety Framework for Text-to-Image Generation](image_generation/latent_guard_a_safety_framework_for_texttoimage_generation.md)**

:   提出Latent Guard框架，在T2I模型文本编码器之上学习一个潜在空间，通过对比学习将黑名单概念与包含该概念的输入prompt映射到相近位置，实现高效的不安全prompt检测（ID Explicit AUC 0.985），支持黑名单测试时灵活更新且无需重训练。

**[LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](image_generation/layoutdetr_detection_transformer_is_a_good_multimodal_layout_designer.md)**

:   将目标检测框架 DETR 与生成模型（GAN/VAE）统一，提出 LayoutDETR 用于多模态条件下的图形布局自动设计，以背景图像为约束、前景图文元素为驱动，在广告横幅和 UI 布局生成上达到 SOTA。

**[Lazy Diffusion Transformer for Interactive Image Editing](image_generation/lazy_diffusion_transformer_for_interactive_image_editing.md)**

:   提出 LazyDiffusion，一种非对称编码器-解码器 Transformer 架构，通过上下文编码器压缩全局信息并仅在 mask 区域执行扩散去噪，实现了与全图生成方法质量相当但速度提升 10 倍的交互式图像编辑。

**[LCM-Lookahead for Encoder-Based Text-to-Image Personalization](image_generation/lcmlookahead_for_encoderbased_texttoimage_personalization.md)**

:   本文提出利用LCM（Latent Consistency Model）作为"快捷通道"，在扩散模型encoder训练中实现图像空间损失（如身份识别loss）的反向传播，配合自注意力特征共享和一致性数据生成，显著提升encoder-based人脸个性化的身份保持和prompt对齐能力。

**[Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)**

:   提出 DP-SAD 框架，通过随机对抗蒸馏训练差分隐私扩散模型：利用扩散模型的时间步稀释 DP 噪声影响，引入判别器加速收敛，并结合梯度链式法则与 DP 后处理特性减少随机性引入，在不需要预训练的条件下实现了 SOTA 的隐私保护图像生成质量。

**[Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](image_generation/learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)**

:   提出语义潜在方向（SLD）方法，通过构建一组正交潜在基方向并将未来运动假设表示为这些方向的线性组合，在随机人体运动预测中实现了更准确、更多样且语义可控的运动预测。

**[Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](image_generation/learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)**

:   提出基于三模态关系的缺失模态AVQA框架，通过关系感知缺失模态生成器（RMM）和音视觉关系感知扩散模型（AVR），在推理时缺少音频或视觉模态的情况下依然能准确回答问题。

**[LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](image_generation/lego_learning_egocentric_action_frame_generation_via_vi.md)**

:   提出 LEGO 模型，通过视觉指令微调增强 VLLM 的动作描述能力，并将 VLLM 的图像/文本嵌入作为额外条件注入扩散模型，实现从第一人称视角生成动作执行帧。

**[Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](image_generation/lego_learning_to_disentangle_and_invert_personalized_concepts_beyond_object_appe.md)**

:   提出Lego方法，通过主体分离和上下文损失实现超越外观的个性化概念（如形容词、动词）的解纠缠与反演，用于扩散模型的个性化内容生成。

**[Linearly Controllable GAN: Unsupervised Feature Categorization and Decomposition for Image Generation and Manipulation](image_generation/linearly_controllable_gan_unsupervised_feature_categorization_and_decomposition_.md)**

:   本文提出LC-GAN，通过对比特征分类和谱正则化实现GAN潜在空间的无监督几何-外观特征解耦，使得生成图像的各个属性可以被线性独立控制，在FFHQ、CelebA-HQ和AFHQ-V2上达到SOTA生成质量。

**[LivePhoto: Real Image Animation with Text-guided Motion Control](image_generation/livephoto_real_image_animation_with_text-guided_motion_control.md)**

:   提出 LivePhoto 图像动画框架，通过运动强度估计模块和文本重加权模块解决文本到运动映射的歧义性，实现基于真实图像和文本描述的高质量视频生成，且用户可额外控制运动强度。

**[Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](image_generation/local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)**

:   提出 GuidedMotion，以局部动作作为细粒度控制信号引导全局运动扩散生成，通过语义图解析和图注意力网络估计引导权重，支持连续可调的运动控制，在生成复杂多动作运动时优势显著。

**[M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](image_generation/m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)**

:   提出 M2D2M，基于离散扩散模型生成多段连续人体运动序列，通过动态转移概率和两阶段采样策略（TPS）实现动作间平滑过渡，且无需额外的多运动训练数据。

**[MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](image_generation/macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)**

:   首次将扩散模型用于骨架表征学习，提出 Masked Conditional Diffusion（MacDiff）框架，通过语义编码器提取掩码骨架的表征来引导条件扩散解码器进行去噪，统一了骨架的判别式和生成式建模。

**[MagicEraser: Erasing Any Objects via Semantics-Aware Control](image_generation/magiceraser_erasing_any_objects_via_semantics-aware_control.md)**

:   提出基于扩散模型的对象擦除框架 MagicEraser，通过内容初始化、提示调优和语义感知注意力重聚焦三阶段设计，无需用户输入文本即可高质量擦除对象并生成和谐背景。

**[Memory-Efficient Fine-Tuning for Quantized Diffusion Model](image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)**

:   提出 TuneQDM，首个面向量化扩散模型的内存高效微调方法，通过多通道量化缩放更新和时间步感知缩放策略，在 4-bit 量化模型上实现了接近全精度模型的个性化生成效果。

**[MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization](image_generation/mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)**

:   针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。

**[MotionChain: Conversational Motion Controllers via Multimodal Prompts](image_generation/motionchain_conversational_motion_controllers_via_multimodal_prompts.md)**

:   提出 MotionChain，一个视觉-运动-语言统一模型，通过多模态提示在多轮对话中生成连续、长期的人体运动序列，支持文本、图像和运动的联合理解与生成。

**[MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](image_generation/motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)**

:   提出 MotionLCM，首次将一致性蒸馏引入人体运动生成领域，在运动潜在空间中实现单步/少步推理的实时运动生成（~30ms/序列），并通过 Motion ControlNet 实现潜在空间中的实时可控运动生成。

**[MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](image_generation/multigen_zero-shot_image_generation_from_multi-modal_prompts.md)**

:   本文提出 MultiGen，通过为每个物体构建"增广token"（融合文本、坐标和图像信息），并训练坐标模型和特征模型来处理推理时的模态缺失，首次实现了从多物体多模态提示进行零样本图像生成，支持纯文本或任意模态组合的灵活输入。

**[Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](image_generation/mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)**

:   提出 MVSD，一个基于扩散模型的互学习框架，将视觉声学匹配（VAM）和去混响作为对称互逆任务联合训练，利用两者的互惠关系克服配对数据稀缺问题，并首次将扩散模型用于视觉引导的混响风格迁移。

**[NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](image_generation/neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)**

:   提出 NeuSDFusion，一个基于混合三平面 SDF 表示（NeuSDF）和空间感知 Transformer 自编码器的 3D 形状生成框架，通过保持三平面间的空间对应关系，在无条件生成、多模态形状补全、单视图重建和文本到 3D 生成等任务上达到 SOTA 性能。

**[NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](image_generation/nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)**

:   提出 NL2Contact，首次利用自然语言描述来可控地建模 3D 手-物体接触图，通过分阶段扩散模型从文本生成手势姿态和接触区域，并构建了首个带有细粒度语言描述的手-物体接触数据集 ContactDescribe。

**[OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](image_generation/omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)**

:   提出 OMG，一种遮挡友好的个性化多概念图像生成框架，通过两阶段采样（布局生成 + 概念噪声融合）实现强身份保持和自然光照协调，且可即插即用地搭配各种单概念模型（LoRA、InstantID）无需额外训练。

**[OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](image_generation/omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)**

:   提出 OmniSSR，首个基于扩散模型的零样本全向图像超分方法，通过十八面切线投影信息交互（OTII）和梯度分解（GD）校正技术，利用 Stable Diffusion 的图像先验实现保真度和真实感的平衡，无需任何训练或微调。

**[PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance](image_generation/panofree_tuning-free_holistic_multi-view_image_generation_with_cross-view_self-g.md)**

:   提出PanoFree，一种无需微调的多视图图像生成方法，通过迭代变形-修补、跨视图引导和对称双向生成策略，高效生成一致的全景图像。

**[Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](image_generation/pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)**

:   提出 PASD（Pixel-Aware Stable Diffusion），通过像素感知交叉注意力（PACA）模块使扩散模型在像素级感知图像局部结构，结合退化去除模块和可调噪声调度，实现了真实图像超分辨率和个性化风格化的统一框架，只需替换底座模型即可切换风格。

**[Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](image_generation/ponymation_learning_articulated_3d_animal_motions_from_.md)**

:   本文提出一种从无标注互联网视频中学习关节式3D动物运动生成模型的方法，通过视频Photo-Geometric自编码框架将视频分解为静态形状、外观和运动隐编码，无需任何姿态标注或参数化形状模型即可在推理时从单张图像生成多样的4D动画。

**[Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](image_generation/powerful_and_flexible_personalized_texttoimage_generation_vi.md)**

:   将个性化T2I生成建模为确定性策略梯度（DPG）框架——扩散模型作为策略、去噪步骤作为动作——引入"向前看"机制捕获长期视觉一致性和DINO相似度奖励，在DreamBooth基准上DINO得分从0.694提升至0.738（+6.3%），CLIP-I从0.762提升至0.797（+4.6%）。

**[Probabilistic Weather Forecasting with Deterministic Guidance-Based Diffusion Model](image_generation/probabilistic_weather_forecasting_with_deterministic_guidance-based_diffusion_mo.md)**

:   本文提出DGDM(Deterministic Guidance Diffusion Model)，通过将确定性预测分支与基于布朗桥的概率扩散分支联合训练，利用确定性预测结果截断扩散反向过程来控制不确定性范围，同时实现精确和概率性的气象预报，并在全球和区域预报任务中达到SOTA。

**[Prompting Future Driven Diffusion Model for Hand Motion Prediction](image_generation/prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)**

:   本文提出PromptFDDM，一个基于prompt的未来驱动扩散模型用于手部运动预测，通过空间-时间提取网络(STEN)结合Ground Truth提取网络(GTEN)和参考数据生成网络(RDGN)的引导机制，以及交互式prompt增强，在第一和第三人称视角的手部运动预测中达到SOTA。

**[Realistic Human Motion Generation with Cross-Diffusion Models](image_generation/realistic_human_motion_generation_with_cross-diffusion_models.md)**

:   提出 CrossDiff 框架，通过统一编码和交叉解码机制融合 3D 与 2D 运动信息，利用交叉扩散实现更精细的全身运动细节捕获，并支持从野外 2D 数据学习 3D 运动生成。

**[RegionDrag: Fast Region-Based Image Editing with Diffusion Models](image_generation/regiondrag_fast_region-based_image_editing_with_diffusion_models.md)**

:   提出基于区域的拷贝-粘贴拖拽编辑方法RegionDrag，用区域指令替代点拖拽指令，实现更快（100倍以上）、更精确且意图更清晰的图像编辑。

**[Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](image_generation/rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)**

:   揭示 IMLE 方法中训练/测试时潜在码分布不对齐问题，提出 RS-IMLE 通过拒绝采样改变训练先验分布，在九个少样本图像数据集上平均降低 45.9% FID。

**[Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](image_generation/removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)**

:   发现正负描述文本在数据集层面存在分布偏差（如词频差异），提出用纯文本分类器过滤偏差数据，微调 LLaVA-1.5 获得 SOTA 图文对齐评分模型 LLaVA-score。

**[ReNoise: Real Image Inversion Through Iterative Noising](image_generation/renoise_real_image_inversion_through_iterative_noising.md)**

:   提出 ReNoise 迭代重噪方法改进扩散模型的图像反演质量，通过在每个反演步骤多次应用 UNet 并平均预测来提升轨迹估计精度，尤其适用于少步扩散模型（SDXL Turbo、LCM）。

**[RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](image_generation/ringid_rethinking_tree-ring_watermarking_for_enhanced_multi-key_identification.md)**

:   本文深入分析了 Tree-Ring 水印方法的鲁棒性来源（发现分布偏移是其验证任务中意外的隐藏助力），揭示其在多密钥识别任务中的严重缺陷，并提出 RingID——一种多通道异构水印框架，通过离散化、无损嵌入和更圆环形设计，将2048个密钥的识别准确率从0.07提升至0.82。

**[Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)**

:   本文提出 Robust-Wide，首个针对指令驱动图像编辑的鲁棒水印方法，核心创新是部分指令驱动去噪采样引导（PIDSG）模块——在训练中将编辑过程的最后k步梯度打通，迫使水印嵌入到语义感知区域，实现编辑后仅约2.6% 的64位水印误码率。

**[RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models](image_generation/rodinhd_high-fidelity_3d_avatar_generation_with_diffusion_models.md)**

:   提出RodinHD，解决triplane解码器的灾难性遗忘问题，并通过层级化肖像表示注入实现高保真3D头像生成。

**[RPBG: Towards Robust Neural Point-based Graphics in the Wild](image_generation/rpbg_towards_robust_neural_point-based_graphics_in_the_wild.md)**

:   本文针对Neural Point-based Graphics (NPBG)在真实场景中的鲁棒性不足问题，提出RPBG，通过退化感知卷积模块、注意力驱动的点可见性校正、轻量级背景建模和点云增强，在不修改点栅格化流程的前提下显著提升了点云神经重渲染在多种wild数据集上的质量和稳定性。

**[SAIR: Learning Semantic-aware Implicit Representation](image_generation/sair_learning_semantic-aware_implicit_representation.md)**

:   本文提出语义感知隐式表示（SAIR），通过构建语义隐式表示（SIR）和外观隐式表示（AIR）两个模块，将CLIP提取的文本对齐语义嵌入融入隐式神经函数，使其在大面积缺失区域的图像修复任务中远超仅依赖外观信息的方法，在CelebAHQ上 PSNR 提升1.65-2.69dB。

**[Scalable Group Choreography via Variational Phase Manifold Learning](image_generation/scalable_group_choreography_via_variational_phase_manifold_learning.md)**

:   本文提出 PDVAE（Phase-conditioned Dance VAE），一种基于相位参数的变分生成模型用于可扩展群舞生成——通过在频域学习舞蹈运动的相位流形（幅度、频率、偏移、相移），实现对**任意数量**舞者的高质量群舞生成，且内存消耗恒定不变，在AIOZ-GDance和AIST-M数据集上全面超越现有方法。

**[ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](image_generation/scaledreamer_scalable_text-to-3d_synthesis_with_asynchronous_score_distillation.md)**

:   本文提出异步分数蒸馏（ASD），通过将扩散时间步前移（而非微调扩散模型）来降低噪声预测误差、对齐渲染图像分布，解决了VSD微调破坏文本理解能力的问题，实现了稳定训练且可扩展至10万条文本提示的prompt-amortized 3D生成器训练。

**[Shedding More Light on Robust Classifiers under the lens of Energy-based Models](image_generation/shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)**

:   通过将鲁棒判别分类器重新解释为基于能量的模型（EBM），揭示了对抗训练的能量动态规律，提出了基于能量加权的对抗训练方法WEAT，并展示了鲁棒分类器隐含的生成能力。

**[SMooDi: Stylized Motion Diffusion Model](image_generation/smoodi_stylized_motion_diffusion_model.md)**

:   提出SMooDi——首个将预训练文本-动作模型适配为风格化动作生成的扩散模型，通过风格适配器和双重风格引导（无分类器引导+基于分类器引导）实现内容文本与风格动作序列驱动的多样化风格动作生成。

**[Soft Prompt Generation for Domain Generalization](image_generation/soft_prompt_generation_for_domain_generalization.md)**

:   提出 SPG（Soft Prompt Generation），首次将生成模型引入 VLM 的 prompt learning，通过 CGAN 从图像动态生成实例特定的软提示，将域知识存储在生成模型中而非提示向量中，实现更好的领域泛化性能。

**[Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](image_generation/source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)**

:   提出SPDInv——一种源提示解耦反演方法，通过将反演过程建模为不动点搜索问题并利用预训练扩散模型求解，使反演噪声码与源提示解耦，显著提升基于文本驱动的图像编辑质量。

**[Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis](image_generation/stable_preference_redefining_training_paradigm_of_human_preference_model_for_tex.md)**

:   重新定义了文本到图像生成中人类偏好模型的训练范式，通过引入质量感知的margin机制和抗干扰损失函数，解决了传统交叉熵训练中"相似质量图像对的盲目惩罚"和"对视觉扰动不鲁棒"两大问题，在主流人类偏好数据集上取得了SOTA性能。

**[StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models](image_generation/styletokenizer_defining_image_style_by_a_single_instance_for_controlling_diffusi.md)**

:   提出StyleTokenizer，通过将风格定义为可学习的token嵌入，实现仅用一张参考图即可控制扩散模型的风格生成，同时精确分离内容和风格。

**[Text2Place: Affordance-aware Text Guided Human Placement](image_generation/text2place_affordance-aware_text_guided_human_placement.md)**

:   提出Text2Place——首个通过文本引导实现真实感人物放置的方法，利用SDS损失优化基于高斯blob的语义掩码学习场景可供性（affordance），再通过主体条件修复实现身份保持的人物放置。

**[TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](image_generation/textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)**

:   利用两个语言模型分别进行布局规划和布局编码，实现更灵活、更多样化的视觉文本渲染，在文本准确性和风格多样性之间取得更好的平衡。

**[Textual-Visual Logic Challenge: Understanding and Reasoning in Text-to-Image Generation](image_generation/textual-visual_logic_challenge_understanding_and_reasoning_in_text-to-image_gene.md)**

:   本文提出了一个新任务——逻辑丰富的文本到图像生成（Logic-Rich T2I），构建了Textual-Visual Logic数据集来评估模型处理复杂关系描述的能力，并设计了包含关系理解模块、多模态融合模块和负样本判别器三个核心组件的基线模型，显著提升了复杂逻辑文本的图像生成质量。

**[The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation](image_generation/the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)**

:   提出 Realistic-Fantasy Benchmark (RFBench) 评估扩散模型在创意/知识密集型 prompt 上的表现，并设计 training-free 的 RFNet 框架，通过 LLM 辅助 prompt 解读和语义对齐评估模块来增强扩散模型对抽象和想象性概念的生成能力。

**[Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](image_generation/toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)**

:   提出 Data Amplify Learning (DAL) 学习范式，用 Diffusion-based Data Amplifier 从仅 5 张标注图像"放大"生成大量配对训练数据，训练出仅 80K 参数的 TinyBeauty 模型，在 iPhone 13 上以 460fps 实现 SOTA 妆容迁移效果。

**[Towards Reliable Advertising Image Generation Using Human Feedback](image_generation/towards_reliable_advertising_image_generation_using_human_feedback.md)**

:   构建百万级人工标注广告图像数据集 RF1M，提出多模态 RFNet 自动检测生成图像的可用性，并设计 Consistent Condition 正则化驱动的 RFFT 微调方法，将广告图像可用率从 56.4% 提升至 85.5%。

**[UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models](image_generation/udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)**

:   提出 UDiffText，通过设计轻量级字符级文本编码器替换 CLIP encoder、引入基于字符分割图的 local attention loss 和 STR loss 微调 cross-attention 层，并在推理阶段对 noised latent 进行 refinement，实现在任意图像中合成高精度、视觉协调的文本，SeqAcc 全面超越 SOTA。

**[Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](image_generation/unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)**

:   提出一种通用的频率解耦学习范式，通过轻量级 ACCA 模块进行粗调低频恢复，再通过 LDRM 模块结合低频一致性约束实现高频细化，仅增加 88K 参数即可为六种 SOTA 低光增强模型带来最高 7.68dB 的 PSNR 提升。

**[WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](image_generation/webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)**

:   提出Web渲染参数生成（WebRPG）新任务，旨在根据HTML代码自动生成网页元素的视觉呈现参数（布局、文本样式、颜色），通过VAE压缩渲染参数和定制HTML嵌入捕获语义层次信息，建立自回归和扩散两种基线模型，其中自回归模型显著优于扩散模型和GPT-4。

**[WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](image_generation/wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)**

:   WildVidFit 提出了一个无需视频训练的虚拟试穿框架，利用基于图像的条件扩散模型和扩散引导模块（VideoMAE + DINO-V2），实现了在野外复杂视频中保持时序一致性的服装试穿效果。

**[XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)**

:   XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。

**[You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation](image_generation/you_only_need_one_step_fast_super-resolution_with_stable_diffusion_via_scale_dis.md)**

:   提出YONOS-SR方法，通过尺度蒸馏（Scale Distillation）策略训练基于Stable Diffusion的超分辨率模型，仅需一步DDIM即可获得SOTA结果，速度比传统方法快200倍。

**[Zero-Shot Detection of AI-Generated Images](image_generation/zero-shot_detection_of_ai-generated_images.md)**

:   本文提出了零样本熵检测器ZED（Zero-shot Entropy-based Detector），通过无损图像编码器估计每个像素在给定上下文下的概率分布，用"图像对真实图像模型的意外程度"作为判别特征，无需任何AI生成训练数据即可检测多种生成器生成的图像，在广泛的生成模型上比SOTA平均准确率提升超过3%。

**[ZigMa: A DiT-style Zigzag Mamba Diffusion Model](image_generation/zigma_a_dit-style_zigzag_mamba_diffusion_model.md)**

:   ZigMa 提出了一种 DiT 风格的 Zigzag Mamba 扩散模型，通过异构逐层锯齿形扫描方案保持空间连续性，以零参数/显存开销实现优于 Mamba 基线的生成质量，同时相比 Transformer 具备线性复杂度优势。

**[ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)**

:   ZipLoRA 提出了一种廉价高效的 LoRA 合并方法，通过学习逐列合并系数并最小化列间余弦相似度，实现了将独立训练的主题 LoRA 和风格 LoRA 无超参数合并，在扩散模型中生成"任意主题 × 任意风格"的个性化图像。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[4D Contrastive Superflows are Dense 3D Representation Learners](autonomous_driving/4d_contrastive_superflows_are_dense_3d_representation_learners.md)**

:   提出SuperFlow框架，通过视图一致性对齐、稠密-稀疏一致性正则化、和基于流的时空对比学习三个模块，利用连续LiDAR-相机对建立4D预训练目标，在11个异构LiDAR数据集上全面超越了之前的Image-to-LiDAR预训练方法。

**[Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](autonomous_driving/accelerating_online_mapping_and_behavior_prediction_via_dire.md)**

:   提出直接将在线地图估计模型内部的BEV特征暴露给下游轨迹预测模型（而非仅传递解码后的矢量化地图），通过三种BEV特征注入策略实现推理加速最高73%、预测精度提升最高29%。

**[Adaptive Human Trajectory Prediction via Latent Corridors](autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)**

:   将prompt tuning思想引入行人轨迹预测，通过在预训练轨迹预测器的输入端添加可学习的低秩图像prompt（称为latent corridors），以不到0.1%的额外参数实现对部署场景特定行为模式的高效自适应，在合成和真实数据上分别取得最高23.9%和26.8%的ADE提升。

**[Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](autonomous_driving/approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)**

:   提出 LiSe 方法，将 2D 图像信息引入无监督 3D 目标检测，通过自步学习（self-paced learning）中的自适应采样和弱模型聚合策略，大幅提升远距离和小目标的检测能力。

**[CarFormer: Self-Driving with Learned Object-Centric Representations](autonomous_driving/carformer_self-driving_with_learned_object-centric_representations.md)**

:   提出 CarFormer，首次将自监督 slot attention 学到的 object-centric 表征用于自动驾驶，在 CARLA Longest6 基准上超越了使用精确物体属性的 PlanT，同时具备世界模型预测未来状态的能力。

**[CSOT: Cross-Scan Object Transfer for Semi-Supervised LiDAR Object Detection](autonomous_driving/csot_cross-scan_object_transfer_for_semi-supervised_lidar_object_detection.md)**

:   提出 CSOT（Cross-Scan Object Transfer）范式，通过 Transformer 网络预测语义一致的物体放置位置和适配度，首次在 LiDAR 半监督目标检测中成功实现了 object copy-paste 增强，配合空间感知分类损失，仅用 1% 标注数据即可达到全监督基线的检测性能。

**[Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](autonomous_driving/detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)**

:   本文从数据标注过程中总结出"回归任务不应使用图像特征"的基本原则，提出 DAL 范式——将检测过程类比为标注过程，用点云特征独立完成回归预测、用融合特征完成分类预测，结合简洁的训练流程，在 nuScenes 上以 74.0 NDS（val）和 74.8 NDS（test）大幅刷新 SOTA。

**[DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment](autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)**

:   提出基于聚类的 Local-to-Global 融合网络 DVLO，通过双向结构对齐（图像→伪点云 + 点云→伪图像）解决视觉与 LiDAR 的数据结构不一致问题，在 KITTI 里程计和 FlyingThings3D 场景流任务上均取得 SOTA。

**[DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](autonomous_driving/dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)**

:   DySeT 提出了一种动态掩码自蒸馏方法，通过强化学习驱动的信息性 token 优先采样和从完整到掩码表示的知识蒸馏，显著提升了自动驾驶场景下轨迹预测模型的泛化能力和鲁棒性。

**[Enhancing Vectorized Map Perception with Historical Rasterized Maps](autonomous_driving/enhancing_vectorized_map_perception_with_historical_rasterized_maps.md)**

:   提出 HRMapNet，通过维护一张低成本的全局历史栅格化地图（historical rasterized map），为在线矢量化地图感知提供互补先验信息，在 BEV 特征聚合和 query 初始化两个层面增强现有方法，在 nuScenes 和 Argoverse 2 上取得显著提升。

**[Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](autonomous_driving/equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)**

:   E-SSL3D 提出一种时空联合等变自监督预训练框架，通过空间等变（对旋转用分类目标、对平移/缩放/翻转用对比目标）和时间等变（用 3D 场景流约束相邻帧特征变换一致性）联合训练 3D 特征编码器，在低数据场景下仅用 20% 标注数据就能达到接近 100% 数据从头训练的检测性能。

**[FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](autonomous_driving/fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)**

:   提出前景自蒸馏（FSD）框架，在同一模型内构建教师-学生分支共享图像特征，避免跨模态蒸馏中的分布差异问题，配合点云增强和多尺度前景增强模块，在 nuScenes 上取得 SOTA 性能。

**[Fully Sparse 3D Occupancy Prediction](autonomous_driving/fully_sparse_3d_occupancy_prediction.md)**

:   提出 SparseOcc，首个完全稀疏的 3D 占用预测网络，通过稀疏体素解码器和掩码引导的 Mask Transformer 实现高效占用预测，并设计了 RayIoU 评价指标解决传统 mIoU 的深度方向不一致惩罚问题。

**[GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)**

:   提出以物体为中心的 3D 语义高斯表示替代传统密集体素，用一组稀疏的 3D 语义高斯描述场景并通过高斯到体素的 splatting 生成占用预测，在性能可比的情况下将内存消耗降低 75%-82%。

**[GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](autonomous_driving/graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)**

:   针对多模态BEV融合中LiDAR与相机标定误差导致的特征错位问题，提出GraphBEV框架，通过LocalAlign（基于KD-Tree的邻域深度图匹配）和GlobalAlign（可学习偏移量全局对齐）两个模块，在nuScenes上达到70.1% mAP（超BEVFusion 1.6%），在噪声错位场景下超BEVFusion 8.3%。

**[H-V2X: A Large Scale Highway Dataset for BEV Perception](autonomous_driving/h-v2x_a_large_scale_highway_dataset_for_bev_perception.md)**

:   提出首个大规模真实世界高速公路 BEV 感知数据集 H-V2X，覆盖100+公里高速路段，含190万+细粒度标注样本，并设计了BEV检测、跟踪和轨迹预测三个基准任务及融合矢量地图的创新方法。

**[Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion](autonomous_driving/hierarchical_temporal_context_learning_for_camera-based_semantic_scene_completio.md)**

:   针对相机语义场景补全（SSC）中时序信息利用粗糙的问题，提出层级式时序上下文学习（HTCL）范式：先通过跨帧模式亲和度（CPA）度量当前帧与历史帧的细粒度对应关系，再通过基于亲和度的动态精炼（ADR）自适应采样补偿不完整观测，在SemanticKITTI上排名第1，甚至在OpenOccupancy上mIoU超过LiDAR方法。

**[Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving](autonomous_driving/improving_agent_behaviors_with_rl_fine-tuning_for_autonomous_driving.md)**

:   通过闭环强化学习微调改善监督学习训练的交通智能体行为模型，解决开环训练的分布偏移问题，在Waymo仿真基准上取得SOTA。

**[ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](autonomous_driving/ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)**

:   提出IT2框架，通过利用LiDAR数据的对等表示（range image + voxel grid）之间的一致性学习作为新型扰动形式，并引入基于高斯混合模型的跨分布对比学习，大幅提升半监督LiDAR语义分割性能。

**[LiDAR-Event Stereo Fusion with Hallucinations](autonomous_driving/lidar-event_stereo_fusion_with_hallucinations.md)**

:   提出将LiDAR稀疏深度点与事件立体相机融合的首个框架，通过在事件堆叠表示（VSH）或原始事件流（BTH）中"幻觉"（插入虚构事件）来弥补事件相机在无运动/无纹理区域的信息缺失，大幅提升事件立体匹配精度。

**[LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment](autonomous_driving/livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)**

:   提出 LiveHPS++，一种基于单 LiDAR 的鲁棒人体动作捕捉方法，通过轨迹引导身体追踪器、噪声不敏感速度预测器和运动学感知姿态优化器三个模块，隐式和显式建模人体运动的动力学和运动学特征，在复杂噪声环境下实现精确且连贯的全局人体运动捕捉。

**[MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](autonomous_driving/mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)**

:   首次将知识蒸馏引入 HD 地图构建任务，提出 MapDistill 框架，通过双 BEV 变换模块、跨模态关系蒸馏、双层特征蒸馏和 Map Head 蒸馏，将相机-LiDAR 融合教师模型的知识迁移至轻量纯相机学生模型，在 nuScenes 上实现 **+7.7 mAP** 或 **4.5倍加速**。

**[MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping](autonomous_driving/maptracker_tracking_with_strided_memory_fusion_for_consistent_vector_hd_mapping.md)**

:   将在线向量高精地图构建重新定义为追踪任务，通过双表示（BEV栅格+道路元素向量）的步进式记忆缓冲区融合机制实现时间一致的高精地图重建，在nuScenes和Argoverse2上分别以76.1和76.9 mAP大幅超越现有方法。

**[Monocular Occupancy Prediction for Scalable Indoor Scenes](autonomous_driving/monocular_occupancy_prediction_for_scalable_indoor_scenes.md)**

:   提出 ISO（Indoor Scene Occupancy）方法，通过预训练深度模型和 D-FLoSP（双特征视线投影）模块实现室内场景的单目 3D 占用预测，并构建了规模比 NYUv2 大 40 倍的 Occ-ScanNet 基准数据集。

**[MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](autonomous_driving/monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)**

:   提出 MonoWAD，通过天气码本学习晴天知识作为参考，结合天气自适应扩散模型将雾效建模为噪声进行特征增强，实现在各种天气条件下鲁棒的单目3D目标检测。

**[Navigation Instruction Generation with BEV Perception and Large Language Models](autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)**

:   提出 BEVInstructor，将鸟瞰图 (BEV) 特征融合到多模态大语言模型中，通过 Perspective-BEV 融合编码器、参数高效的 Prompt Tuning 以及实例引导的迭代优化策略，在室内外导航指令生成任务上取得 SOTA。

**[Neural Volumetric World Models for Autonomous Driving](autonomous_driving/neural_volumetric_world_models_for_autonomous_driving.md)**

:   本文提出 NeMo（Neural Volumetric World Model），一种基于体积表示的端到端自动驾驶框架，通过 3D 体素表征场景、运动流模块建模动态、时间注意力整合未来预测信息，以自监督方式训练并在 nuScenes 和 CARLA 上实现了超越前人方法 18%+ 的驾驶性能。

**[NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](autonomous_driving/neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)**

:   提出 NeuroNCAP，一个基于 NeRF 渲染的真实感闭环自动驾驶安全测试框架，受 Euro NCAP 碰撞避免协议启发设计三类安全关键场景（静止/正面/侧面碰撞），揭示当前 SOTA 端到端规划器（UniAD、VAD）在闭环安全场景中严重失败——碰撞率高达 88-92%——尽管其感知模块准确运行。

**[OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](autonomous_driving/occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)**

:   OccGen 将 3D 语义占用预测重新定义为"noise-to-occupancy"的生成式范式，通过条件编码器提取多模态特征、渐进式精炼解码器执行扩散去噪，以由粗到精的方式逐步生成占用图，在 nuScenes-Occupancy 上多模态/纯LiDAR/纯相机设置下分别相对提升 9.5%/6.3%/13.3% 的 mIoU。

**[OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](autonomous_driving/occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)**

:   OccWorld 提出在 3D 占用空间中学习世界模型，用 VQ-VAE 对 3D occupancy 进行 token 化，再通过 GPT 风格的时空生成 Transformer 自回归预测未来场景演化和自车轨迹，在 nuScenes 上无需实例和地图标注即可实现有竞争力的规划性能。

**[OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](autonomous_driving/open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)**

:   提出 OPEN，通过目标级深度编码器（ODE）从像素级深度先验中预测物体中心深度，并设计目标级位置编码（OPE）将该信息注入 Transformer 解码器，生成 3D 目标感知特征，在 nuScenes 上达到 64.4% NDS 的 SOTA 性能。

**[Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](autonomous_driving/optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)**

:   本文提出 Optimal Gaussian Diffusion (OGD) 和 Estimated Clean Manifold (ECM) Guidance 两项技术，分别通过优化扩散先验分布和在干净流形上直接注入引导梯度，将联合轨迹预测的扩散步数减少到原来的 1/12，引导采样步数减少到 1/5，同时在 Argoverse 2 上取得更优性能。

**[PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation](autonomous_driving/panovos_bridging_non-panoramic_and_panoramic_views_with_transformer_for_video_se.md)**

:   提出首个全景视频目标分割数据集 PanoVOS（150个视频、19K实例标注），揭示现有 VOS 模型无法处理全景视频的像素不连续和严重畸变问题，并设计 PSCFormer 利用全景空间一致性注意力解决左右边界连续性问题。

**[Progressive Pretext Task Learning for Human Trajectory Prediction](autonomous_driving/progressive_pretext_task_learning_for_human_trajectory_prediction.md)**

:   提出渐进式前置任务学习框架 PPT，通过三阶段训练（逐步下一位置预测 → 目的地预测 → 完整轨迹预测）逐步增强模型对短期动态和长期依赖的捕获能力，配合高效的两步非自回归 Transformer 预测器，在多个行人轨迹预测基准上取得 SOTA。

**[Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes](autonomous_driving/random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)**

:   提出 Random Walk on Pixel Manifolds (RWPM)，利用随机游走捕获像素嵌入的流形结构来修正因驾驶场景多样性导致的流形畸变，从而提升异常分割评分函数的准确性，无需额外训练即可即插即用地集成到现有异常分割框架中。

**[RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation](autonomous_driving/rapid-seg_range-aware_pointwise_distance_distribution_networks_for_3d_lidar_segm.md)**

:   本文提出 RAPiD 特征（Range-Aware Pointwise Distance Distribution），一种对刚体变换不变且适应点密度变化的 LiDAR 点云局部几何特征，配合双层嵌套自编码器和通道注意力融合，在 SemanticKITTI（76.1 mIoU）和 nuScenes（83.6 mIoU）上达到 SOTA 分割性能。

**[Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving](autonomous_driving/reason2drive_towards_interpretable_and_chainbased_reasoning.md)**

:   构建 Reason2Drive 基准数据集（600K+ 视频-文本对，覆盖感知-预测-推理链式任务），提出 ADRScore 评估链式推理正确性的新指标，并设计 Prior Tokenizer + Instructed Vision Decoder 框架增强 VLM 的目标级感知和推理能力，在自动驾驶推理任务上显著超越所有基线。

**[Reliability in Semantic Segmentation: Can We Use Synthetic Data?](autonomous_driving/reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)**

:   首次系统地利用 Stable Diffusion 生成合成 OOD 数据来全面评估语义分割模型的可靠性，包括协变量偏移下的鲁棒性评估、OOD 物体检测评估和模型校准，并证明合成数据与真实 OOD 数据的评估结果高度相关。

**[Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](autonomous_driving/rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)**

:   通过数据中心分析识别出恶劣天气对 LiDAR 的两大核心干扰模式（几何扰动和点丢失），提出 Selective Jittering 和 Learnable Point Drop 两种针对性数据增强方法，在 SemanticKITTI→SemanticSTF 基准上将 baseline 提升 8.1 mIoU 达到 SOTA。

**[Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](autonomous_driving/rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)**

:   提出密度判别特征嵌入（DDFE）模块，利用单一 LiDAR 源域点云中固有的密度多样性（近处密/远处疏），学习密度感知的特征表示，实现对不同传感器配置下未见域的泛化，无需目标域数据。

**[Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](autonomous_driving/risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)**

:   RaSc 提出风险感知自一致模仿学习框架，通过 TTC（碰撞时间）预测分支学习人类驾驶行为背后的风险规避动机，并通过自一致性约束使规划器理解自身动作的物理后果，在 nuPlan 数据集的开环和闭环评估中均超越了先前的学习型方法。

**[RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion](autonomous_driving/roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)**

:   RoofDiffusion 提出了一种基于条件扩散概率模型的端到端自监督方法，用于从严重稀疏（最高99%缺失）、不完整（80%区域遮挡）且含噪的屋顶高程图中恢复完整干净的高程信息，在自建的 PoznanRD 数据集和 BuildingNet 上显著超越传统插值方法和现有深度补全方法。

**[Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](autonomous_driving/safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)**

:   Safe-Sim 提出了一个基于扩散模型的闭环安全关键仿真框架，通过在扩散去噪过程中引入对抗项和部分扩散（Partial Diffusion）机制，实现了对抗车辆行为类型（碰撞角度、相对速度、碰撞类型）的细粒度控制，在 nuScenes 和 nuPlan 上验证了对多种 planner 的有效评估能力。

**[SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](autonomous_driving/seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)**

:   SeFlow 提出将传统的基于 ray-casting 的动态点分类融入自监督场景流学习管线，通过专门的动态/静态损失函数和基于聚类的物体级运动一致性约束，在 Argoverse 2 和 Waymo 上以实时速度（48ms/帧）取得自监督场景流 SOTA 性能，甚至超越部分有监督方法。

**[SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](autonomous_driving/sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)**

:   SFPNet 提出稀疏焦点调制（SFPM）替代 window-attention，通过多层级上下文提取和门控自适应聚合来避免针对特定 LiDAR 类型的归纳偏置设计，在机械旋转式、固态和混合固态三种 LiDAR 数据集上均取得领先或竞争性性能，并发布了首个混合固态 LiDAR 语义分割数据集 S.MID。

**[SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](autonomous_driving/simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)**

:   提出 SimPB 统一模型，通过混合解码器（multi-view 2D decoder + 3D decoder）以循环 3D→2D→3D 的方式同时完成多相机 2D 检测和 BEV 空间 3D 检测，在 nuScenes 上两项任务均取得优秀结果。

**[SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic](autonomous_driving/sledge_synthesizing_driving_environments_with_generative_models_and_rule-based_t.md)**

:   SLEDGE 提出了首个基于生成模型的驾驶仿真器，通过 Raster-to-Vector 自编码器将驾驶场景编码为栅格化潜在图（RLM），再利用 Diffusion Transformer 生成高质量的车道图和交通参与者，实现了比 nuPlan 少 500 倍存储（<4GB）的仿真环境，同时支持 500m 长路线测试，暴露了 SOTA 规划器 PDM-Closed 超过 40% 的失败率。

**[Stream Query Denoising for Vectorized HD-Map Construction](autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)**

:   提出 Stream Query Denoising (SQD) 策略，通过对前一帧 GT 添加噪声并训练网络恢复当前帧 GT 来增强流式 HD 地图构建中的时序一致性建模，在 nuScenes 和 Argoverse2 上全面超越 StreamMapNet。

**[TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes](autonomous_driving/tod3cap_towards_3d_dense_captioning_in_outdoor_scenes.md)**

:   首次提出户外 3D 密集描述任务，构建百万级 TOD3Cap 数据集（850 场景 2.3M 描述），设计基于 BEV 特征 + Relation Q-Former + LLaMA-Adapter 的端到端网络，超越适配后的室内方法 +9.6 CIDEr@0.5IoU。

**[Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation](autonomous_driving/train_till_you_drop_towards_stable_and_robust_source-free_unsupervised_3d_domain.md)**

:   针对无源数据的3D语义分割域自适应（SFUDA）中训练后期性能退化问题，提出正则化策略和基于参考模型一致性的验证准则，实现稳定且鲁棒的自适应。

**[UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving](autonomous_driving/unim2ae_multi-modal_masked_autoencoders_with_unified_3d_representation_for_3d_pe.md)**

:   本文提出 UniM2AE，一个多模态自监督预训练框架，通过将图像和 LiDAR 点云特征统一投影到 3D 体素空间（比 BEV 多保留高度维度），并设计 Multi-modal 3D Interactive Module（MMIM）进行高效跨模态交互，实现了比独立预训练和简单拼接的前序方法更强的 3D 检测（+1.2% NDS）和 BEV 分割（+6.5% mIoU）提升。

**[UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](autonomous_driving/unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)**

:   UniTraj 构建了一个统一多数据集（nuScenes、Argoverse 2、WOMD）、多模型（AutoBot、MTR、Wayformer）和多评估策略的车辆轨迹预测框架，揭示模型跨数据集泛化能力显著下降，但通过扩大数据规模和多样性可大幅提升性能，合并训练在 nuScenes 排行榜达到第 1 名。

**[VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](autonomous_driving/visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)**

:   提出 VisionTrap，将环视相机图像和文本描述引入轨迹预测任务，通过 BEV 视觉语义编码器和文本驱动的去偏对比学习引导模型学习视觉语义线索（如行人姿态、转向灯等），在保持 53ms 实时推理的同时显著提升预测精度并发布 nuScenes-Text 数据集。

**[Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](autonomous_driving/weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)**

:   提出 VG-W3D 框架，仅使用 2D 标注（无需任何 3D 标签），通过特征级、输出级和训练级三层视觉引导来训练 3D 目标检测器，在 KITTI 上取得了与使用 500 帧 3D 标注方法相当的性能。

---

## ✂️ 语义分割 { #segmentation }

**[A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties](segmentation/a_semantic_space_is_worth_256_language_descriptions_make_str.md)**

:   ProLab 用 LLM 生成类别的常识性描述，通过句子嵌入和 K-Means 聚类将其压缩为 256 个可解释的描述性属性，构建属性级多热标签空间替代传统 one-hot 类别标签来监督分割模型，在五个经典基准上一致超越类别级监督且涌现出域外泛化能力。

**[A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](segmentation/a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)**

:   基于Stable Diffusion构建了一个极简的潜在扩散分割框架LDMSeg，通过浅层自编码器将分割mask压缩到潜空间、再训练图像条件扩散模型来生成全景分割结果，避免了传统方法中的目标检测模块、匈牙利匹配和复杂后处理，并天然支持mask inpainting和多任务扩展。

**[ActionVOS: Actions as Prompts for Video Object Segmentation](segmentation/actionvos_actions_as_prompts_for_video_object_segmentation.md)**

:   提出ActionVOS——一种以人类动作叙述作为额外语言提示的Referring Video Object Segmentation新设定，通过无参数的动作感知标注模块生成伪标签，并设计动作引导的focal loss来抑制假阳性，在VISOR上将非活跃物体的误分割降低35.6% mIoU，同时在VOST/VSCOS上对状态变化物体的分割提升3.0% mIoU。

**[Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images](segmentation/active_coarsetofine_segmentation_of_moveable_parts_from_real.md)**

:   提出首个面向真实室内场景RGB图像中可运动部件实例分割的主动学习框架，通过姿态感知masked attention网络实现由粗到细的分割，仅需人工标注11.45%的图像即可获得全量验证的高质量分割结果，相比最优非AL方法节省60%人工时间。

**[Attention Decomposition for Cross-Domain Semantic Segmentation](segmentation/attention_decomposition_for_cross-domain_semantic_segmentation.md)**

:   本文提出 ADFormer，一种用于跨域语义分割的新型 Transformer 架构，通过将解码器中的交叉注意力分解为域无关和域特定两部分，结合梯度反转对抗学习，有效缩小源域和目标域之间的分布差异，在 GTA→Cityscapes 和 SYNTHIA→Cityscapes 两个基准上以显著更低的复杂度超越了现有无 proposal 方法。

**[CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection](segmentation/cola_conditional_dropout_and_language-driven_robust_dual-modal_salient_object_de.md)**

:   提出 CoLA 框架，通过语言驱动的质量评估（LQA）和条件性 Dropout（CD）两个核心模块，首次在双模态显著性目标检测中同时解决噪声输入和模态缺失两大鲁棒性问题。

**[ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders](segmentation/colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)**

:   提出 ColorMAE，通过对随机噪声施加不同频域滤波器生成具有空间与语义先验的数据无关遮罩模式，在不增加任何参数和计算开销的前提下，显著提升 MAE 的下游任务表现，尤其在语义分割任务上相比随机遮罩提升 2.72 mIoU。

**[ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](segmentation/controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)**

:   提出 ControlNet++，通过像素级循环一致性损失显式优化条件可控生成质量：用预训练判别模型从生成图像中提取条件并与输入条件对齐，并设计高效单步去噪 reward 策略避免多步采样的巨大显存开销，在分割掩码、边缘、深度等多种条件控制下显著提升可控性（如分割 mIoU +11.1%）。

**[CoReS: Orchestrating the Dance of Reasoning and Segmentation](segmentation/cores_orchestrating_the_dance_of_reasoning_and_segmentation.md)**

:   提出 CoReS（Chains of Reasoning and Segmenting），一种双链结构的多模态思维链框架，通过推理链和分割链的层次化协作，结合 in-context 引导策略，实现对复杂推理文本中目标物体的渐进式精确分割，在 ReasonSeg 数据集上超越 LISA 6.5%。

**[CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](segmentation/cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)**

:   提出 CPM（Class-conditional Prompting Machine），通过结合类无关查询与基于 GMM 采样的类条件查询来增强 Mask2Former 在音视频分割中的二部图匹配稳定性和跨模态注意力效力，同时设计音频条件提示（ACP）、视觉条件提示（VCP）和提示对比学习（PCL）三个辅助任务，在 AVSBench 和 VPO 基准上达到 SOTA。

**[Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](segmentation/cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)**

:   提出 Cs2K 框架，从类别特有知识（原型引导伪标签 + 原型引导类别适应）和类别共享知识（权重引导选择性整合）两个方面协同缓解增量语义分割中的灾难性遗忘与新类欠拟合问题。

**[Dataset Enhancement with Instance-Level Augmentations](segmentation/dataset_enhancement_with_instance-level_augmentations.md)**

:   提出一种基于预训练扩散模型的实例级数据增强方法，通过在保持原始标注不变的前提下逐个重绘图像中的目标实例，显著提升了显著性目标检测、语义分割和目标检测的性能，同时支持数据匿名化。

**[Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time](segmentation/deep_nets_with_subsampling_layers_unwittingly_discard_useful_activations_at_test.md)**

:   发现深度网络中下采样层在默认前向传播中丢弃了大量有用激活，提出一个搜索+聚合框架在测试时利用这些被丢弃的激活图来提升分类和分割性能，与传统TTA方法正交互补。

**[DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs](segmentation/densenets_reloaded_paradigm_shift_beyond_resnets_and_vits.md)**

:   重新审视 DenseNet 的密集拼接连接（concatenation shortcut），通过系统性现代化改造（加宽减深、现代化 block、扩大中间维度、更多 transition 层等），提出 RDNet（Revitalized DenseNet），在 ImageNet-1K 上超越 Swin Transformer、ConvNeXt、DeiT-III，证明了拼接连接作为一种被低估的范式具有强大潜力。

**[Diffusion Models for Open-Vocabulary Segmentation](segmentation/diffusion_models_for_open-vocabulary_segmentation.md)**

:   本文提出 OVDiff，利用预训练的文本到图像扩散模型为任意文本类别生成支持图像集，从中提取多层次原型（类级、实例级、部件级），结合背景原型实现无训练的开放词汇语义分割，在 PASCAL VOC 上超越先前方法 10% 以上。

**[DreamLIP: Language-Image Pre-training with Long Captions](segmentation/dreamlip_language-image_pre-training_with_long_captions.md)**

:   通过 MLLM 为 30M 图像生成长文本描述，提出动态子描述采样的多正样本对比学习和子描述特定分组损失，实现细粒度视觉-语言对齐，仅用 30M 数据在检索和语义分割上达到甚至超越 CLIP 400M 的性能。

**[EAFormer: Scene Text Segmentation with Edge-Aware Transformers](segmentation/eaformer_scene_text_segmentation_with_edge-aware_transformers.md)**

:   提出边缘感知Transformer（EAFormer），通过文本边缘提取器过滤非文本区域边缘、对称交叉注意力在编码器中融合文本边缘信息，显著提升文字边缘区域的分割精度，并重标注COCO_TS和MLT_S数据集以实现更公平评估。

**[Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](segmentation/early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)**

:   提出NeST（New claSsifier pre-Tuning）方法，在正式训练前通过学习从所有旧分类器到新分类器的线性变换来初始化新分类器权重，并设计基于跨任务类别相似性的变换矩阵初始化策略，在Pascal VOC和ADE20K上显著提升多种CISS方法的性能。

**[Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)**

:   R-Adapter 通过在 CLIP 模型中插入轻量级 adapter 模块并结合三种自集成策略（Adapter Dropping、权重累积、权重缩放重参数化），在仅微调 13% 参数的前提下同时实现了 ID 高精度和 OOD 强鲁棒性，并首次将鲁棒微调扩展到分类之外的跨模态检索和开放词汇分割任务。

**[Eliminating Feature Ambiguity for Few-Shot Segmentation](segmentation/eliminating_feature_ambiguity_for_few-shot_segmentation.md)**

:   提出AENet插件网络，通过挖掘判别性查询前景区域来消除特征歧义，增强交叉注意力中的前景-前景匹配，可即插即用地提升现有少样本分割方法性能（SCCAN 1-shot在PASCAL-5i上+3.0%）。

**[Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](segmentation/frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)**

:   提出频率-空间纠缠学习（FSEL）框架，通过在频率域和空间域之间进行纠缠学习（entanglement learning），利用全局频率特征弥补空间特征的局部性和敏感性限制，在三个COD基准上超越21个SOTA方法。

**[FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](segmentation/frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)**

:   提出 FREST，一种面向多种恶劣条件（雾、雨、雪、夜间）的源无关域自适应语义分割框架，通过交替学习条件嵌入空间（分离条件信息）和特征恢复（将恶劣条件特征恢复为正常条件），逐步消除恶劣条件对特征的影响，在 ACDC 和 RobotCar 基准上均达到新的 SOTA。

**[General and Task-Oriented Video Segmentation](segmentation/general_and_task-oriented_video_segmentation.md)**

:   GvSeg 提出了一个通用视频分割框架，通过将分割目标解耦为外观、形状和位置三个因素，并根据任务需求（VIS/VSS/VPS/EVS）动态调整这三个因素在查询初始化、匹配和采样中的参与度，在统一架构下实现了四种视频分割任务的SOTA性能。

**[GiT: Towards Generalist Vision Transformer through Universal Language Interface](segmentation/git_towards_generalist_vision_transformer_through_universal_language_interface.md)**

:   提出 GiT 框架，通过通用语言接口将图像描述、目标检测、实例分割、语义分割和视觉定位五大视觉任务统一为自回归序列生成，仅用纯 ViT（无任何任务特定模块）实现多任务联合训练，且任务间互相增强。

**[LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](segmentation/lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)**

:   本文提出 LASS3D，在 MeanTeacher 半监督 3D 语义分割框架中引入大语言视觉模型（LVM）生成多层级文本描述来增强 3D 特征，并通过渐进式负学习策略有效利用低置信度伪标签点，在室内外数据集上取得显著提升。

**[Learning Camouflaged Object Detection from Noisy Pseudo Label](segmentation/learning_camouflaged_object_detection_from_noisy_pseudo_label.md)**

:   提出首个弱半监督伪装目标检测方法 (WSSCOD)，仅用 20% 像素级标注 + 80% 框标注即可达到全监督 SOTA 的可比性能，核心贡献是一个自适应噪声校正损失 $\mathcal{L}_{NC}$，可在早期学习和记忆化两个阶段分别优化。

**[Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](segmentation/learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)**

:   首次提出完全使用网络图像（而非精心设计的数据集图像）进行弱监督增量语义分割，通过傅里叶域判别器筛选网络图像 + caption 驱动的 rehearsal 策略保持旧类知识，在 PASCAL VOC 15-5 设定下达到 73.4% mIoU。

**[LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors](segmentation/lift_a_surprisingly_simple_lightweight_feature_transform_for_dense_vit_descripto.md)**

:   提出 LiFT，一种极其简单的轻量级后处理网络（仅 1.2M 参数），通过自监督多尺度重建目标训练，融合冻结 ViT 的粗粒度语义特征与 CNN 提取的细粒度图像特征，以仅增加 5.7% 参数和 22% FLOPs 的代价将 ViT 特征分辨率翻倍，在关键点匹配、检测、分割和目标发现等密集任务上均获得显著性能提升。

**[Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](segmentation/long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)**

:   首次系统性地解决时序动作分割中的长尾问题，提出 Group-wise Temporal Logit Adjustment (G-TLA) 框架，利用活动标签进行分组分类并结合动作时序先验进行 logit 调整，在大幅提升尾部类别性能的同时不损失头部类别。

**[Occlusion-Aware Seamless Segmentation](segmentation/occlusion-aware_seamless_segmentation.md)**

:   提出 Occlusion-Aware Seamless Segmentation (OASS) 新任务与 UnmaskFormer 框架，同时解决全景图像窄视场解锁、遮挡物体完整分割和针孔-全景跨域适应三大挑战，在自建 BlendPASS 数据集上达到 SOTA。

**[OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](segmentation/olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)**

:   提出即插即用框架 OLAF，通过将前景/边缘掩码作为额外输入通道、引入低层稠密特征提取模块 LDF 和针对性权重适配策略，在不改变基础架构的前提下为多种分割网络（CNN/U-Net/Transformer）带来显著的多物体多部件分割增益，在最具挑战的 Pascal-Parts-201 上超越 SOTA 达 4.0 mIoU。

**[OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](segmentation/openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)**

:   首次定义开放集全景场景图生成（OpenPSG）任务，利用 BLIP-2 作为多模态关系解码器，结合关系查询 Transformer（RelQ-Former）实现开放集关系预测，在 PSG 数据集 PredCls R@100 达到 79.3%，闭集场景超越先前 SOTA 26.6%。

**[Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](segmentation/part2object_hierarchical_unsupervised_3d_instance_segmentation.md)**

:   提出 Part2Object 层次聚类框架，利用自监督特征和3D物体性先验（objectness prior），从零件级过分割逐层合并到物体级实例，生成高质量伪标签用于自训练 Hi-Mask3D，实现无需人工标注的3D实例分割。

**[PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](segmentation/partstad_2d-to-3d_part_segmentation_task_adaptation.md)**

:   PartSTAD 提出了一种 2D-to-3D 部件分割的任务适配方法：通过为 GLIP 的 2D 检测框引入可学习权重预测网络（以 3D mRIoU 为目标优化），并集成 SAM 获取精确前景掩码，在 PartNet-Mobility 上实现了语义分割 mIoU 提升 7.0%p、实例分割 mAP50 提升 5.2%p（相对 PartSLIP）。

**[Point-Supervised Panoptic Segmentation via Estimating Pseudo Labels from Learnable Distance](segmentation/point-supervised_panoptic_segmentation_via_estimating_pseudo_labels_from_learnab.md)**

:   本文提出一种基于可学习距离的点监督全景分割方法，用 anchor query 表示每个实例，通过交叉注意力预测像素到实例的距离，并以端到端方式由点标签监督距离学习，结合迭代的查询聚合和增强过程持续优化伪标签质量，取得了点监督全景分割的 SOTA 结果。

**[ReMamber: Referring Image Segmentation with Mamba Twister](segmentation/remamber_referring_image_segmentation_with_mamba_twister.md)**

:   本文首次将 Mamba 架构引入指称图像分割（RIS）任务，提出 Mamba Twister 模块通过通道扫描和空间扫描的"扭转"机制实现高效的视觉-语言特征融合，在 RefCOCO/RefCOCO+/G-Ref 三个基准上取得了超越 Transformer 方法的竞争性结果，同时保持线性计算复杂度。

**[Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](segmentation/representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)**

:   利用分形理论将分形维数（FD）从图像级扩展到像素级，生成分形特征图（FFM）作为深度学习模型的额外输入和损失权重，并设计包含边缘解码器和骨架解码器的多解码器网络（MD-Net），在五个管状结构数据集上显著提升分割性能。

**[Rotary Position Embedding for Vision Transformer](segmentation/rotary_position_embedding_for_vision_transformer.md)**

:   本文系统研究了将 RoPE（Rotary Position Embedding）从1D语言模型扩展到2D视觉任务的方法，提出 RoPE-Mixed（混合可学习频率）替代传统的 Axial 频率分配，在 ViT 和 Swin Transformer 上实现了显著的分辨率外推性能提升，在 ImageNet 分类、COCO 检测和 ADE20k 分割上均带来一致增益。

**[SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)**

:   发现 CLIP 的密集预测失败源于自注意力导致的空间位置错位问题，提出 Correlative Self-Attention (CSA) 机制——仅修改最后一层自注意力的计算方式（无需训练），将 CLIP 的零样本语义分割从 14.1% 平均 mIoU 提升至 38.2%，超越所有已有方法。

**[SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis](segmentation/seggen_supercharging_segmentation_models_with_text2mask_and_mask2img_synthesis.md)**

:   提出 SegGen 数据生成框架，反转传统"先生图再标注"的流程为"先从文本生成分割掩码，再从掩码生成图像"，打破分割数据合成的"鸡生蛋"瓶颈，在 ADE20K 上将 Mask2Former R50 的 mIoU 从 47.2 提升至 49.9（+2.7）。

**[Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills](segmentation/segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)**

:   提出分割引导的矢量化框架，通过梯度感知分割子程序引导 Bézier 路径的初始化和优化，首次在保持分层拓扑的逐层矢量化方法中支持径向渐变填充，使矢量图形在更少路径数下达到更高的视觉质量。

**[SeiT++: Masked Token Modeling Improves Storage-Efficient Training](segmentation/seit_masked_token_modeling_improves_storage-efficient_training.md)**

:   在 SeiT 的 token 化训练框架上引入掩码 token 建模（MTM）自监督预训练，并设计 TokenAdapt 和 ColorAdapt 两种 token 专用数据增强策略，在仅 1% 存储空间（1.4GB）下将 ImageNet-1k 分类准确率从 74.0% 提升至 77.8%，有效解决了 token 域数据增强的难题。

**[Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](segmentation/self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)**

:   提出 SCoSPARC——一个两阶段自监督共显著目标检测模型，通过 patch 级和 region 级 ViT 特征对应关系检测图像组中的共显著物体，在 CoCA 数据集上 F-measure 比无监督 SOTA 高 13.7%，甚至超越多个有监督方法。

**[SiLC: Improving Vision Language Pretraining with Self-Distillation](segmentation/silc_improving_vision_language_pretraining_with_self-distillation.md)**

:   提出SiLC框架，在CLIP式图文对比学习中加入局部到全局的自蒸馏，显著提升密集预测任务（检测、分割）的性能，同时改善分类和检索。

**[SOS: Segment Object System for Open-World Instance Segmentation With Object Priors](segmentation/sos_segment_object_system_for_open-world_instance_segmentation_with_object_prior.md)**

:   提出 SOS 方法，通过用 DINO 自注意力图作为物体先验生成聚焦于物体的 SAM 提示点，从而产出高质量伪标注来训练标准实例分割系统，在 COCO/LVIS/ADE20k 跨类别/跨数据集设置下大幅超越 SOTA，精度提升高达 81.6%。

**[SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images](segmentation/spin_hierarchical_segmentation_with_subpart_granularity_in_natural_images.md)**

:   SPIN 构建了首个自然图像子部件（subpart）级层级语义分割数据集 SubPartImageNet——包含 203 个子部件类别和 10.6 万条标注——并提出两个层级一致性评估指标（SpCS / SeCS），在 20+ 现代模型上全面基准测试，揭示了当前模型在子部件层面的严重不足。

**[Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](segmentation/un-evimo_unsupervised_event-based_independent_motion_segmentation.md)**

:   首个无需标注的事件相机独立运动物体(IMO)分割框架，利用光流与几何约束生成伪标签训练分割网络，在 EVIMO 数据集上取得与有监督方法可比的性能。

**[UniFS: Universal Few-Shot Instance Perception with Point Representations](segmentation/unifs_universal_few-shot_instance_perception_with_point_representations.md)**

:   提出UniFS——首个通用少样本实例感知模型，通过将目标检测、实例分割、姿态估计和目标计数统一为动态点表示学习范式，并引入结构感知点学习(SAPL)损失来捕获点间高阶结构关系，在最小任务假设下达到接近专家模型的性能。

**[Unsupervised Moving Object Segmentation with Atmospheric Turbulence](segmentation/unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)**

:   本文提出一种无监督方法，通过"检测-生长"（detect-then-grow）策略分割大气湍流视频中的运动目标：先用基于 Sampson 距离的极线几何一致性检查分离真实运动与湍流运动，再从高置信种子像素出发区域生长生成分割掩码，最后用时空一致性损失精细化，在首个真实湍流视频数据集 DOST 上大幅超越现有方法（IoU 提升 60.1%）。

**[VISA: Reasoning Video Object Segmentation via Large Language Models](segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)**

:   提出 ReasonVOS 新任务和 VISA 模型，利用多模态 LLM 的世界知识推理能力实现基于隐式文本查询的视频目标分割与跟踪。

**[VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](segmentation/visage_video_instance_segmentation_with_appearance-guided_enhancement.md)**

:   针对在线视频实例分割(VIS)中现有方法过度依赖位置信息导致的关联错误，提出VISAGE通过从骨干特征中显式提取外观嵌入、结合对比学习和简化tracker来增强实例关联准确性，在YTVIS和OVIS基准上取得SOTA。

**[VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network](segmentation/vp-sam_taming_segment_anything_model_for_video_polyp_segmentation_via_disentangl.md)**

:   本文提出 VP-SAM，通过语义解耦适配器（SDA）利用傅里叶频谱的幅度信息帮助 SAM 区分低对比度的息肉与背景，同时设计时空侧网络（STSN）为 SAM 注入视频帧间时序信息，在 SUN-SEG、CVC-612 和 CVC-300 等数据集上达到 SOTA。

**[You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception](segmentation/you_only_learn_one_query_learning_unified_human_query_for_single-stage_multi-per.md)**

:   提出 HQNet 框架，通过学习统一的 Human Query 表示，在单阶段单模型中同时完成行人检测、实例分割、2D 姿态估计、3D Mesh 恢复、属性识别等多种以人为中心的感知任务，并构建了首个全面的多任务人体感知基准 COCO-UniHuman。

---

## 🧑 人体理解 { #human_understanding }

**[3D Hand Pose Estimation in Everyday Egocentric Images](human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)**

:   通过系统研究裁剪输入、相机内参感知位置编码(KPE)、辅助监督(手部分割+抓握标签)和多数据集联合训练这四个实践，提出WildHands系统，在仅用ResNet50和少量数据的条件下，实现了对野外第一人称图像中3D手部姿态的鲁棒估计，零样本泛化超过FrankMocap全部指标且与10倍大的HaMeR竞争。

**[3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)**

:   提出将视线估计重新表述为密集3D眼球网格回归，并通过从大规模野外人脸图像中自动提取伪标签+HeadGAN合成多视图进行弱监督训练，在跨域场景下比SOTA提升最多30%。

**[3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](human_understanding/3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)**

:   本文提出3D空间注意力模块（3DSA），通过3D空间细分算法将特征体积划分为多个区域并为其分配基于视角的注意力权重，解决多视图3D人体姿态估计中不同视角对不同空间区域贡献不均的问题，在 CMU Panoptic Studio 数据集上达到 SOTA。

**[A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](human_understanding/a_simple_baseline_for_spoken_language_to_sign_language_trans.md)**

:   提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

**[AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](human_understanding/adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)**

:   提出AdaDistill，将知识蒸馏概念嵌入margin penalty softmax loss中，通过基于EMA的自适应类中心（早期用sample-sample简单知识、后期用sample-center复杂知识）和困难样本感知机制，无需额外超参数即可提升轻量级人脸识别模型的判别能力，在IJB-B/C和ICCV21-MFR等挑战性基准上超越SOTA蒸馏方法。

**[Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](human_understanding/adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)**

:   提出自适应高频Transformer（AdaFreq），通过频域混合增强、目标感知的高频token动态选择、特征均衡损失三大策略，将高频信息（毛皮纹理、轮廓边缘等）统一用于多种野生动物的重识别，在8个跨物种数据集上超越现有ReID方法。

**[ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](human_understanding/aden_adaptive_density_representations_for_sparseview_camera.md)**

:   提出ADen框架，通过生成器输出多个位姿假设+判别器评分选择最佳的方式，统一了位姿回归和概率估计范式，仅需500个自适应样本即超越需要500K均匀采样的方法，同时实现实时推理。

**[Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences](human_understanding/alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)**

:   提出 Alignist，首个利用 CAD 模型信息（SDF + SurfEmb 对应特征）训练隐式分布网络来推断 SO(3) 上姿态分布的方法，通过 product of experts 融合几何和特征对齐，在低数据场景下显著优于对比学习方法。

**[Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](human_understanding/audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)**

:   提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。

**[Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](human_understanding/avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)**

:   本文定义了"Avatar指纹识别"这一新任务——验证合成说话头视频中驱动表情的真实身份，贡献了迄今最大规模的面部重建数据集NVFAIR（161个身份），并提出基于归一化面部关键点距离和时序CNN的基线方法，通过学习与外观无关的面部运动签名实现身份验证（平均AUC 0.85），且能泛化到未见过的生成器（AUC 0.83）。

**[Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](human_understanding/bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)**

:   本文提出运动学短语（Kinematic Phrases, KP）作为人体运动与动作语义之间的中间表示，KP基于客观运动学事实，具有适当抽象性、可解释性和通用性，并据此构建了运动理解系统和白盒运动生成评估基准KPG。

**[Combining Generative and Geometry Priors for Wide-Angle Portrait Correction](human_understanding/combining_generative_and_geometry_priors_for_wide-angle_portrait_correction.md)**

:   提出结合 StyleGAN 生成式先验（用于人脸矫正）和几何对称先验（用于背景直线矫正）的双模块框架，大幅提升广角人像畸变校正的视觉质量和定量指标。

**[CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing](human_understanding/como_controllable_motion_generation_through_language_guided_pose_code_editing.md)**

:   提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。

**[Cut Out the Middleman: Revisiting Pose-Based Gait Recognition](human_understanding/cut_out_the_middleman_revisiting_pose-based_gait_recognition.md)**

:   重新审视基于姿态的步态识别方法，提出 GaitHeat 框架，用热力图（heatmap）取代传统的骨架关键点坐标来编码人体姿态，通过改进的预处理流程和姿态引导热力图对齐模块大幅提升性能和泛化能力，使基于姿态的方法首次接近轮廓（silhouette）方法的精度。

**[De-confounded Gaze Estimation](human_understanding/de-confounded_gaze_estimation.md)**

:   本文提出基于因果干预的视线估计框架 FSCI，通过特征分离将视线相关特征与身份/光照等无关特征解耦，并利用动态混杂因子库对无关特征进行因果干预，在跨域设置下较基线提升36.2%、较SOTA提升11.5%。

**[Diffusion Model is a Good Pose Estimator from 3D RF-Vision](human_understanding/diffusion_model_is_a_good_pose_estimator_from_3d_rf-vision.md)**

:   提出mmDiff，一种基于扩散模型的毫米波雷达人体姿态估计框架，通过全局-局部雷达上下文提取和结构-运动一致性约束，有效应对雷达点云稀疏、噪声大和信号不一致的挑战，显著超越现有SOTA。

**[EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](human_understanding/edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)**

:   提出基于正交可学习基向量的高效解耦框架 EDTalk，将人脸动态分解为嘴型、头部姿态和情感表情三个独立潜空间，同时支持视频驱动和音频驱动的情感说话人头像生成。

**[Event-based Head Pose Estimation: Benchmark and Method](human_understanding/event-based_head_pose_estimation_benchmark_and_method.md)**

:   针对事件相机头部姿态估计（HPE）领域缺乏大规模数据集和专用方法的问题，构建了两个大规模多场景事件HPE基准数据集，并提出包含事件时空融合（ESTF）和事件运动感知注意力（EMPA）两个核心模块的专用网络，在多种挑战场景下取得优异性能。

**[Facial Affective Behavior Analysis with Instruction Tuning](human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)**

:   提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。

**[FoundPose: Unseen Object Pose Estimation with Foundation Features](human_understanding/foundpose_unseen_object_pose_estimation_with_foundation_features.md)**

:   FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。

**[FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](human_understanding/freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)**

:   提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。

**[Generalizable Facial Expression Recognition](human_understanding/generalizable_facial_expression_recognition.md)**

:   提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。

**[GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)**

:   提出GS-Pose方法，利用预训练视觉基础模型（DINOv2）的2D语义特征投影到3D空间，结合几何特征通过Transformer匹配网络进行类别级物体9D姿态估计，仅需10个合成3D模型训练即可在多个真实数据集上达到SOTA级别性能。

**[How Video Meetings Change Your Expression](human_understanding/how_video_meetings_change_your_expression.md)**

:   提出 FacET（Facial Explanations through Translations），一种基于生成式域翻译的可解释框架，通过学习解耦的面部空间特征和可解释的时空线性变换，自动发现视频会议（VC）与面对面（F2F）交流之间的细微面部表情差异模式，并支持将 VC 视频转换为 F2F 风格的"去zoom化"。

**[HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation](human_understanding/hpe-li_wifi-enabled_lightweight_dual_selective_kernel_convolution_for_human_pose.md)**

:   本文提出 HPE-Li，一种基于 WiFi 信号的轻量化人体姿态估计方法，通过创新的双选择性核注意力（SKA）机制构建多分支 CNN，能够根据输入的 WiFi CSI 数据特征动态调整感受野大小，在 MM-Fi 和 WiPose 两个基准上以极低的计算开销超越了 SOTA 方法。

**[Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](human_understanding/human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)**

:   提出HoCoTTA框架，通过多域稳态评估和隔离参数优化策略，在持续变化的目标域中实现人体运动预测的鲁棒自适应，有效缓解了灾难性遗忘和误差累积问题。

**[HUMOS: Human Motion Model Conditioned on Body Shape](human_understanding/humos_human_motion_model_conditioned_on_body_shape.md)**

:   提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。

**[LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](human_understanding/lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)**

:   提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。

**[Large Motion Model for Unified Multi-Modal Motion Generation](human_understanding/large_motion_model_for_unified_multi-modal_motion_generation.md)**

:   提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。

**[MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation](human_understanding/manikin_biomechanically_accurate_neural_inverse_kinematics_for_human_motion_esti.md)**

:   本文提出MANIKIN，通过在SMPL参数模型中嵌入解剖学约束并设计基于旋转角预测的神经逆运动学求解器，从头部和手部的稀疏末端执行器姿态精确恢复全身运动，同时保证生物力学合理性和地面非穿透性。

**[MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition](human_understanding/migs_multi-identity_gaussian_splatting_via_tensor_decomposition.md)**

:   提出MIGS，通过CP张量分解将多个人体身份的3DGS参数统一到一个低秩张量中，在大幅减少参数量的同时实现了对未见姿态的鲁棒动画。

**[Modeling and Driving Human Body Soundfields through Acoustic Primitives](human_understanding/modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)**

:   提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。

**[Motion Mamba: Efficient and Long Sequence Motion Generation](human_understanding/motion_mamba_efficient_and_long_sequence_motion_generation.md)**

:   本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。

**[Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](human_understanding/multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)**

:   提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。

**[Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](human_understanding/occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)**

:   针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。

**[Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)**

:   提出了一个自监督学习基准，同时评估语义分类和姿态估计能力，并设计视角轨迹正则化损失(trajectory loss)，利用相邻视角的图像三元组约束特征空间中的局部线性性，使学到的表征既保持语义分类精度又获得 emergent 的全局姿态感知能力，在域内和域外姿态估计上均提升4%。

**[PoseSOR: Human Pose Can Guide Our Attention](human_understanding/posesor_human_pose_can_guide_our_attention.md)**

:   本文首次将人体姿态信息引入显著目标排序(SOR)任务，通过提出姿态感知交互模块(PAI)和姿态驱动排序模块(PDR)来建模人体活动与注意力转移的关系，在复杂场景中显著提升了SOR性能并达到SOTA。

**[ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](human_understanding/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)**

:   提出 ReLoo，通过分层神经人体表示和非层级虚拟骨骼变形模块，从单目野外视频中重建穿着宽松服装的高质量3D人体模型。

**[RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](human_understanding/repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)**

:   RePOSE 提出用时空相对深度一致性损失替代传统的绝对深度监督信号，将遮挡场景下的 3D 人体姿态估计从"学习绝对深度值"转变为"学习关键点的相对深度顺序"，以极简的实现（仅需几行代码）显著提升遮挡条件下的姿态估计鲁棒性和精度。

**[ScanTalk: 3D Talking Heads from Unregistered Scans](human_understanding/scantalk_3d_talking_heads_from_unregistered_scans.md)**

:   提出 ScanTalk，首个能够对**任意拓扑**（包括未配准的3D扫描数据）的3D人脸进行语音驱动动画生成的深度学习框架，核心依赖于 DiffusionNet 的离散化无关特性来突破固定拓扑约束。

**[SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](human_understanding/scape_a_simple_and_strong_category-agnostic_pose_estimator.md)**

:   通过将类别无关姿态估计(CAPE)简化为纯自注意力特征匹配问题，抛弃显式相似度匹配和两阶段框架，引入全局关键点特征感知器(GKP)和关键点注意力精炼器(KAR)以提升注意力质量，在MP-100数据集上1-shot/5-shot设置下分别超越SOTA 2.2/1.3 PCK，同时减少参数量和提升推理速度。

**[Spectral Subsurface Scattering for Material Classification](human_understanding/spectral_subsurface_scattering_for_material_classification.md)**

:   提出利用Spectral Sub-Surface Scattering（S4，光谱次表面散射）进行材质分类的方法，证明了次表面散射的强光谱依赖性可以提供高度判别性的特征，并设计了一种新型成像装置通过2D投影高效获取S4测量数据，无需耗时的高光谱扫描。

**[TELA: Text to Layer-wise 3D Clothed Human Generation](human_understanding/tela_text_to_layer-wise_3d_clothed_human_generation.md)**

:   TELA提出了分层的3D穿衣人体表示方法和渐进优化策略，从文本描述生成服装可解耦的3D人体模型，支持逐层穿衣生成和虚拟试衣等编辑应用。

**[TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)**

:   本文提出TF-FAS框架，通过双重语义元素（内容元素和类别元素）的细粒度引导来增强人脸反欺骗的跨域泛化能力，其中CEDM模块探索并解耦内容相关特征，FCEM模块挖掘类别内的细粒度差异，在多个跨域FAS基准上达到SOTA。

**[Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)**

:   本文提出MMA-FAS框架解决多模态人脸反欺骗中的模态缺失问题，通过模态解耦适配器从频率分解角度分离模态不变和模态特有特征，结合LBP引导的对比损失和自适应模态组合采样策略，在所有模态缺失场景下均达到SOTA。

**[U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](human_understanding/u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)**

:   本文提出 U-COPE，首个统一处理刚性和铰接物体的类别级 9D 位姿估计框架，通过将刚性物体视为单部件铰接物体来统一问题定义，利用 Point Pair Features（PPF）独立提取各部件特征并通过通用投票策略预测关键位姿参数，在合成和真实数据集上均达到 SOTA。

**[UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](human_understanding/upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)**

:   提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。

**[Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving](human_understanding/upper-body_hierarchical_graph_for_skeleton_based_emotion_recognition_in_assistiv.md)**

:   本文针对辅助驾驶场景提出 UbH-GCN，利用上半身骨骼序列构建层次化图结构（UbH-Graph）动态建模关节运动与情感的关系，并引入类别特定变化机制平衡不均衡数据分布，在 AIDE 辅助驾驶数据集上超越现有多模态方法。

**[VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)**

:   VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。

**[Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](human_understanding/wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)**

:   提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。

**[WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](human_understanding/worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)**

:   利用2022年FIFA世界杯体育场部署的多视角静态摄像机基础设施，构建了首个大规模多人全局3D姿态估计数据集WorldPose，包含约250万个3D姿态和超过120公里的全局轨迹，并揭示了现有全局姿态估计方法在多人场景下面临的严峻挑战。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](multimodal_vlm/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)**

:   提出AdaShield框架，通过精心设计的静态防御提示(AdaShield-S)和基于LLM的自适应迭代优化框架(AdaShield-A)，在不微调MLLM或训练额外模块的前提下，有效防御结构化越狱攻击，将攻击成功率从75%以上降至15%以下并保持正常任务性能。

**[AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](multimodal_vlm/addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)**

:   提出 AddressCLIP 框架，通过图像-文本对齐（地址+场景描述的对比学习）和图像-地理匹配（基于GPS距离的流形学习）两大核心组件，将图像地址定位（IAL）问题建模为端到端的视觉-语言对齐任务，在自建的三个IAL数据集上取得最高85.92%的Top-1准确率。

**[ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](multimodal_vlm/artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)**

:   本文提出将视觉属性识别问题重新建模为基于图像条件的前缀语言模型（PrefixLM）下的句子生成概率问题，通过"生成式检索"（Generative Retrieval）替代传统的"对比式检索"（Contrastive Retrieval），显式建模物体-属性间的条件依赖关系，在VAW和新提出的VGARank数据集上显著超越对比检索方法。

**[Attention Prompting on Image for Large Vision-Language Models](multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)**

:   提出Attention Prompting on Image（API），用辅助VLM（如CLIP或LLaVA）根据文本查询生成注意力归因热力图，将其叠加到原始图像上作为视觉提示，在无需训练的情况下提升LVLM在多个VL基准上的表现（LLaVA-1.5 在MM-Vet上+3.8%）。

**[Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding](multimodal_vlm/bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_.md)**

:   提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。

**[BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](multimodal_vlm/beaf_observing_beforeafter_changes_to_evaluate_hallucination.md)**

:   BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。

**[BLINK: Multimodal Large Language Models Can See but Not Perceive](multimodal_vlm/blink_multimodal_large_language_models_can_see_but_not_perceive.md)**

:   提出BLINK——一个包含14个经典计算机视觉感知任务的多模态评测基准（3807道选择题），这些任务人类可以"眨眼间"解决（95.7%准确率），但最强的GPT-4V仅达51.26%（仅高于随机猜测13.17%），揭示了当前MLLM在核心视觉感知能力上的严重缺失。

**[BRAVE: Broadening the Visual Encoding of Vision-Language Models](multimodal_vlm/brave_broadening_the_visual_encoding_of_vision-language_models.md)**

:   本文系统性地分析了不同视觉编码器（CLIP、DINOv2、EVA-CLIP等）对VLM性能的影响，发现没有单一编码器能在所有任务上最优，基于此提出BRAVE方法，通过轻量级的MEQ-Former将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在captioning和VQA任务上取得SOTA，并显著降低视觉幻觉。

**[CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](multimodal_vlm/cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)**

:   提出 CAT 模型，通过设计问题相关线索聚合器（Clue Aggregator）捕获细粒度音视频特征，结合混合多模态训练策略和 AI 辅助的模糊感知直接偏好优化（ADPO）策略，显著提升 MLLM 在动态音视频场景中的问答准确性，在多个 AVQA 基准上达到 SOTA。

**[CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](multimodal_vlm/clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)**

:   从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。

**[Dataset Growth (InfoGrowth)](multimodal_vlm/dataset_growth.md)**

:   提出 InfoGrowth，一种高效的在线数据清洗与选择算法，通过近邻搜索估计每个样本的信息增益，实现数据集的持续增长，同时保证清洁度和多样性，在 CC3M 上仅用 1/6 数据即超过全量训练效果。

**[Efficient Inference of Vision Instruction-Following Models with Elastic Cache](multimodal_vlm/efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)**

:   Elastic Cache 提出一种针对多模态指令遵循模型的 KV Cache 管理方法，在指令编码阶段采用基于重要性的 cache 合并策略（而非丢弃），在输出生成阶段采用固定点淘汰策略，以"一个序列、两种策略"实现任意加速比的高效推理，在 KV Cache 预算仅 0.2 时实现 78% 的实际速度提升且保持生成质量。

**[Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](multimodal_vlm/elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)**

:   提出 SpLIP，一种基于冻结 CLIP 的双向多模态提示学习框架，通过视觉-文本编码器间的双向知识交换、自适应 margin 的三元组损失和条件跨模态拼图任务，在 ZS-SBIR、GZS-SBIR 和 FG-ZS-SBIR 三种草图检索设定下均取得 SOTA。

**[Elysium: Exploring Object-level Perception in Videos via MLLM](multimodal_vlm/elysium_exploring_object-level_perception_in_videos_via_mllm.md)**

:   提出 Elysium，一个端到端可训练的 MLLM，通过构建百万级视频目标感知数据集 ElysiumTrack-1M 和设计 T-Selector 视觉 token 压缩网络，将 MLLM 的目标级感知能力从图像扩展到视频领域，支持单目标跟踪 (SOT)、引用式单目标跟踪 (RSOT) 和视频引用表达生成 (Video-REG) 等任务。

**[Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](multimodal_vlm/eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)**

:   提出ECSO（Eyes Closed, Safety On），一种无需训练的MLLM保护方法，通过检测自身响应的安全性，并将不安全查询中的图像自适应转换为文本描述，从而恢复预对齐LLM的内在安全机制，在MM-SafetyBench上实现最高71.3%的安全性提升，且不损害常规性能。

**[FlexAttention for Efficient High-Resolution Vision-Language Models](multimodal_vlm/flexattention_for_efficient_high-resolution_vision-language_models.md)**

:   提出 FlexAttention，通过基于注意力图的高分辨率token动态选择和层次化自注意力融合机制，在保持甚至超越现有高分辨率VLM性能的同时，将计算成本降低近40%。

**[FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](multimodal_vlm/freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)**

:   首次在**完全不使用动捕数据**的情况下，利用 MLLM（GPT-4V）作为关键帧设计师和动画师，结合基于物理的运动跟踪，实现开放集人体运动合成。

**[Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](multimodal_vlm/genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)**

:   提出 Genixer 数据生成流水线，训练 MLLM 自身作为数据生成器，无需依赖 GPT-4V 即可自动生成高质量视觉指令微调数据，生成的 915K VQA 数据和 350K REC 数据分别提升 LLaVA1.5 和 Shikra 在多个基准上的表现。

**[Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](multimodal_vlm/groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)**

:   Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

**[IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](multimodal_vlm/ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)**

:   IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。

**[LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers](multimodal_vlm/loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)**

:   LoA-Trans提出一种位置感知的查询选择机制，生成多个可能的目标位置作为位置感知查询（而非仅依赖估计的中心点），并引入TaskSyn网络在解码器中实现指代表达理解（REC）和指代表达分割（RES）的任务协同，显著提升视觉定位的准确性。

**[m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](multimodal_vlm/m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)**

:   提出 m&m's 基准，包含 4K+ 多步骤多模态任务和 33 个可执行工具，系统评估 10 个 LLM 在不同规划策略（多步 vs 逐步）、计划格式（JSON vs 代码）和反馈类型（解析/验证/执行）下的工具使用能力，发现多步JSON规划配合反馈是当前最优设计。

**[MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](multimodal_vlm/marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)**

:   提出 MarvelOVD 框架，通过将检测器的上下文感知能力和背景识别能力融入 VLM 的伪标签生成与训练流程，在线净化噪声伪标签并自适应重加权训练框，在 COCO 和 LVIS 上大幅超越已有方法。

**[MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](multimodal_vlm/mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)**

:   提出MathVerse——一个包含2612道视觉数学题目（转化为6个版本共15K测试样本）的多模态数学推理评测基准，通过系统性地调控文本与图像中的信息分配来检验MLLM是否真正"看懂"了数学图表，并提出CoT评估策略进行细粒度推理过程评分，揭示了大多数MLLM严重依赖文本而非视觉图表进行数学推理。

**[Merlin: Empowering Multimodal LLMs with Foresight Minds](multimodal_vlm/merlin_empowering_multimodal_llms_with_foresight_minds.md)**

:   提出 Foresight Pre-Training (FPT) 和 Foresight Instruction-Tuning (FIT) 两阶段训练范式，通过轨迹建模赋予多模态大语言模型"前瞻性思维"能力，使模型能够基于当前观察预测未来事件并进行推理。

**[Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](multimodal_vlm/meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。

**[MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](multimodal_vlm/mm1_methods_analysis_and_insights_from_multimodal_llm_pre-training.md)**

:   Apple 系统性地消融了 MLLM 构建的三大轴（架构、数据、训练），得出关键设计准则：图像分辨率 > 模型大小 > 训练数据；VL 连接器类型影响甚微；caption/interleaved/text-only 三类数据的精细混合至关重要，最终构建了 3B-30B dense 和最高 64B MoE 的 MM1 模型族，在 few-shot 预训练评测上达到 SOTA。

**[MMBench: Is Your Multi-modal Model an All-Around Player?](multimodal_vlm/mmbench_is_your_multi-modal_model_an_all-around_player.md)**

:   提出 MMBench——一个包含 3217 道多选题、覆盖 20 个细粒度能力维度的双语（英/中）视觉语言模型评测基准，并设计了 CircularEval 循环评测策略和基于 LLM 的选项提取机制，显著提升了评测的鲁棒性和公平性。

**[MyVLM: Personalizing VLMs for User-Specific Queries](multimodal_vlm/myvlm_personalizing_vlms_for_userspecific_queries.md)**

:   MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。

**[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](multimodal_vlm/navgpt-2_unleashing_navigational_reasoning_capability_for_large_vision-language_.md)**

:   NavGPT-2通过将冻结LLM的隐层表征作为视觉-语言特征输入拓扑图导航策略网络，在保留LLM可解释性导航推理能力的同时，消除了基于LM的智能体与VLN专用模型之间的性能差距，并展现出优异的数据效率。

**[Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](multimodal_vlm/nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)**

:   Nymeria 是目前世界最大的野外人体运动数据集（300 小时、264 名参与者），首次提供同步定位的多设备多模态自我中心数据（Project Aria 眼镜+腕带+动捕服），并配套 310.5K 句层次化运动语言描述。

**[Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](multimodal_vlm/omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)**

:   OVT通过构建460万多视角图文数据集MVCap和设计minimax优化的跨视角对齐框架，以参数高效微调方式显著提升VLP模型（如CLIP）对3D视角变化的鲁棒性（平均+9-10%），同时几乎不损失原始性能。

**[Quantized Prompt for Efficient Generalization of Vision-Language Models](multimodal_vlm/quantized_prompt_for_efficient_generalization_of_vision-language_models.md)**

:   将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。

**[REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models](multimodal_vlm/revision_rendering_tools_enable_spatial_fidelity_in_vision-language_models.md)**

:   提出 REVISION 框架，利用 Blender 3D 渲染生成空间关系精确的合成图像，以免训练方式引导 T2I 模型生成空间一致的图像，并构建 RevQA 基准评估 MLLM 的空间推理能力。

**[Robust Calibration of Large Vision-Language Adapters](multimodal_vlm/robust_calibration_of_large_vision-language_adapters.md)**

:   本文发现CLIP适配方法（Adapter/Prompt Learning/TTA）在OOD场景下严重损害了零样本基线的校准能力，揭示logit范围增大（而非logit范数增大）是误校准的根本原因，并提出三种简单且模型无关的logit范围约束方案（ZS-Norm、Penalty、SaLS），有效缓解误校准同时保持判别性能。

**[Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](multimodal_vlm/select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)**

:   提出选择性双教师知识迁移框架（SND），通过衡量预训练VLM和最近微调VLM之间的特征差异，在无标签参考数据集上自适应选择合适的教师进行知识蒸馏，同时缓解灾难性遗忘并保持零样本分类能力。

**[Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](multimodal_vlm/self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)**

:   提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。

**[ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](multimodal_vlm/sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)**

:   ShareGPT4V 构建了一个120万条高质量描述性caption数据集（由GPT4-Vision生成100K种子 + Share-Captioner扩展至1.2M），通过在预训练和SFT两阶段使用该数据集训练LLaVA架构的模型ShareGPT4V-7B，在11个多模态benchmark中9个取得最优，证明了高质量caption是LMM模态对齐的关键瓶颈。

**[SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](multimodal_vlm/sq-llava_self-questioning_for_large_vision-language_assistant.md)**

:   提出视觉自提问（Visual Self-Questioning）训练范式，让 LLM 不仅学习回答问题，还学习根据图像主动提问，通过充分利用指令数据中问题本身的丰富语义信息来增强视觉-语言对齐。

**[The Hard Positive Truth About Vision-Language Compositionality](multimodal_vlm/the_hard_positive_truth_about_visionlanguage_compositionalit.md)**

:   本文揭示了现有CLIP组合性基准的评估盲区——缺少hard positives测试，发现hard negative微调会导致模型"过敏"（对语义保持的改写也错误地降低匹配分数），并通过同时加入hard positives和hard negatives训练来缓解这一问题。

**[Towards Open-ended Visual Quality Comparison](multimodal_vlm/towards_open-ended_visual_quality_comparison.md)**

:   本文提出 Co-Instruct，首个面向开放式视觉质量比较的大型多模态模型，通过从两种"弱监督源"（LLM合并的单图描述 + GPT-4V伪标签）构建562K指令微调数据集，实现比 GPT-4V（其教师模型）更高的多图质量比较准确率，并提出首个多图比较基准 MICBench。

**[Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models](multimodal_vlm/towards_real-world_adverse_weather_image_restoration_enhancing_clearness_and_sem.md)**

:   本文提出WResVLM半监督学习框架，利用视觉-语言模型（VLM）为真实恶劣天气图像提供清晰度评估和语义描述监督信号，通过VLM图像评估+天气提示学习增强清晰度、描述辅助的语义正则化增强语义，在真实去雨/去雾/去雪任务上全面超越现有方法。

**[Uni3DL: Unified Model for 3D and Language Understanding](multimodal_vlm/uni3dl_a_unified_model_for_3d_vision-language_understanding.md)**

:   提出 Uni3DL，一个直接在点云上操作的统一 3D 视觉-语言模型，通过 Query Transformer 学习任务无关的语义/掩码输出，再由 Task Router 组合多个功能头实现语义分割、实例分割、目标检测、视觉定位、3D 描述生成、文本-3D 检索等六大任务，性能达到或超过各任务专用 SOTA。

**[UniCode: Learning a Unified Codebook for Multimodal Large Language Models](multimodal_vlm/unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)**

:   UniCode提出学习一个统一的codebook来同时tokenize视觉和文本信号，通过language-driven iterative training范式将视觉tokenizer的码本与LLM的词表渐进对齐，并引入in-context image decompression预训练任务提升图像生成质量，使MLLM无需额外对齐模块即可实现多模态理解与生成。

**[Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](multimodal_vlm/vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

**[X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](multimodal_vlm/x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP-ViT（对比学习）和MAE-ViT（掩码图像建模）的互补视觉特征，在仅使用1/10数据量的情况下显著超越BLIP-2在细粒度视觉理解任务上的表现。

**[Zero-shot Object Counting with Good Exemplars (VA-Count)](multimodal_vlm/zero-shot_object_counting_with_good_exemplars.md)**

:   提出VA-Count框架，通过样本增强模块（EEM）利用Grounding DINO发现高质量正负样本，结合噪声抑制模块（NSM）用对比学习区分正负密度图，实现零样本目标计数在FSC-147和CARPK上的SOTA表现。

---

## 📹 视频理解 { #video_understanding }

**[ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](video_understanding/actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)**

:   提出 ActionSwitch——首个无需类别信息即可检测流式视频中重叠动作实例的在线时序动作定位（On-TAL）框架，核心将多动作检测建模为有限状态机的状态分类问题，并辅以 conservativeness loss 减少碎片化误检，在 THUMOS14、FineAction、Epic-Kitchens 100 等数据集上在 OAD 扩展方法中达到 SOTA。

**[Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts](video_understanding/adapt2reward_adapting_videolanguage_models_to_generalizable.md)**

:   提出 Adapt2Reward，通过可学习的失败提示（failure prompts）将预训练视频语言模型适配为可泛化的语言条件奖励函数，仅需少量单一环境的机器人数据即可泛化到新环境和新任务，在 MetaWorld 上比前方法高出约 28%。

**[AMEGO: Active Memory from Long EGOcentric Videos](video_understanding/amego_active_memory_from_long_egocentric_videos.md)**

:   提出 AMEGO，一种从长第一人称视频中在线构建结构化"活跃记忆"的方法，通过 HOI tracklet + 位置分段 + 语义无关的视觉查询，在新提出的 AMB benchmark 上超越 Video QA baselines 12.7%。

**[Bayesian Evidential Deep Learning for Online Action Detection](video_understanding/bayesian_evidential_deep_learning_for_online_action_detection.md)**

:   本文提出 BEDL（Bayesian Evidential Deep Learning）框架，通过贝叶斯教师-证据学生架构，在在线动作检测任务中实现了准确高效的推理与可靠的不确定性量化，并设计了基于贝叶斯互信息的注意力模块用于主动特征选择。

**[Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects](video_understanding/benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)**

:   基于 HANDS23 挑战赛（AssemblyHands + ARCTIC 数据集），系统性地对第一人称视角下手-物体交互的 3D 姿态估计方法进行了基准测试和深入分析，揭示了畸变校正、高容量 Transformer 和多视角融合的有效性，以及快速运动、遮挡和窄视角下物体重建等仍未解决的挑战。

**[Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](video_understanding/boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)**

:   本文提出了一个统一的3D单目标跟踪（SOT）框架，通过3D生成式预训练和2D预训练基础跟踪器的匹配知识蒸馏，解决了点云数据稀缺和LiDAR扫描稀疏不完整的问题，在KITTI、Waymo和nuScenes上达到SOTA性能。

**[Classification Matters: Improving Video Action Detection with Class-Specific Attention](video_understanding/classification_matters_improving_video_action_detection_with_class-specific_atte.md)**

:   提出类别专属查询（class queries）机制，通过为每个动作类别分配独立的可学习查询，让模型动态关注与各类别相关的上下文区域，显著提升视频动作检测中的分类性能。

**[CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](video_understanding/crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)**

:   提出CrossGLG框架，利用LLM生成的文本描述以"全局→局部→全局"的方式引导骨架特征学习，在单样本3D动作识别中以仅2.8%的SOTA模型参数量大幅超越对手。

**[Data Collection-Free Masked Video Modeling](video_understanding/data_collection-free_masked_video_modeling.md)**

:   提出基于伪运动生成器（PMG）从静态图像递归生成伪运动视频，结合掩码视频建模（VideoMAE）进行自监督预训练，完全摆脱真实视频数据的采集成本和隐私/版权顾虑，甚至可用合成图像实现有效的视频Transformer预训练。

**[DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](video_understanding/dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)**

:   提出DINO-Tracker，将预训练DINOv2的语义特征与测试时单视频优化相结合，通过Delta-DINO残差微调和多源自监督损失实现长程稠密点追踪，在自监督方法中达到SOTA且可媲美有监督追踪器，尤其在长期遮挡场景中大幅领先。

**[Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](video_understanding/efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)**

:   EMP-Net 提出了一种高效多层级后推理网络，通过后推理机制避免大部分梯度回传来降低 CLIP 在小样本动作识别中的领域对齐开销，同时利用多层级表示（全局、patch、帧级别）提升特征判别力，在效率和性能之间取得了最优平衡。

**[EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere](video_understanding/egoposer_robust_real-time_egocentric_pose_estimation_from_sparse_and_intermitten.md)**

:   提出 EgoPoser，仅从头显设备的头部和手部稀疏且间歇性追踪信号中，鲁棒地估计全身姿态，通过全局运动分解、真实视野建模、SlowFast时序融合和体型感知优化四大核心设计，在大规模真实场景中实现SOTA性能，推理速度超600fps。

**[Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](video_understanding/exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)**

:   本文提出FERMT（Feature Extraction and Relation Modeling Tracker），通过将one-stream tracker中的注意力机制分解为四个功能不同的子模块——浅层专注特征提取、深层专注关系建模——并引入双注意力单元进行特征预处理，在GOT-10k上以69.6%的AO分数超越领先实时跟踪器5.6%，同时CPU速度提升54%。

**[FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)**

:   提出 FinePseudo 框架，利用基于时序对齐性（temporal alignability）的度量学习来改善伪标签质量，首次系统性地解决半监督细粒度动作识别问题，在四个细粒度数据集上显著超越现有方法。

**[Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](video_understanding/goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)**

:   提出 Goldfish 框架，通过将长视频分割为短 clip 并利用基于文本相似度的检索机制选取与问题最相关的 top-k 片段，实现对任意长度视频的高效理解，同时提出 MiniGPT4-Video 短视频模型和 TVQA-long 长视频评测基准。

**[HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](video_understanding/hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)**

:   提出HAT——首个在Online Temporal Action Localization（OnTAL）中引入长期历史上下文的anchor-based Transformer框架，通过动作预期引导的历史压缩和未来驱动的历史精炼，在程序性自我中心数据集（EGTEA/EK100）上显著超越OAT，在标准数据集（THUMOS/MUSES）上达到可比或更优性能。

**[IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](video_understanding/iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)**

:   提出IAM-VFI框架，通过引入运动复杂度图（Motion Complexity Map）来感知局部运动的难度级别，对不同复杂度区域自适应分配计算资源和处理策略，实现对任意运动模式的鲁棒视频帧插值。

**[LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow](video_understanding/layeredflow_a_real-world_benchmark_for_non-lambertian_multi-layer_optical_flow.md)**

:   提出 LayeredFlow——首个包含多层光流标注的真实世界非朗伯体基准数据集（150k 光流对，185 个场景，360 个物体），并提出多层光流任务定义、大规模合成训练数据集和基于 RAFT 的多层光流基线方法。

**[Leveraging Temporal Contextualization for Video Action Recognition](video_understanding/leveraging_temporal_contextualization_for_video_action_recognition.md)**

:   提出 **TC-CLIP** 框架，通过**时序上下文化(TC)** 机制将全局视频动作线索压缩为少量 context tokens 注入 CLIP 编码过程，并设计**视频条件提示(VP)** 模块将视觉信息注入文本端，在零样本、小样本、base-to-novel 和全监督四种设定下全面超越现有 CLIP-based 视频识别方法。

**[Local All-Pair Correspondence for Point Tracking](video_understanding/local_all-pair_correspondence_for_point_tracking.md)**

:   本文提出LocoTrack，通过局部4D相关性体（local 4D correlation）实现视频中任意点的全对应匹配，结合轻量级相关性编码器和长度可泛化的Transformer，在所有TAP-Vid基准测试上达到最高精度，同时比SOTA方法快近6倍。

**[Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](video_understanding/masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)**

:   提出 EVI-MAE，首个联合第一人称视频与身体穿戴 IMU 的多模态表示学习方法，通过 MAE 自监督预训练学习视频-IMU 跨模态对齐，并用图神经网络建模多 IMU 设备间的协同运动关系，在动作识别中取得 SOTA 且具备优秀的鲁棒性。

**[Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](video_understanding/motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)**

:   本文提出一种将非线性运动先验（轨迹参数函数）引入对比度最大化框架的自监督方法，用于事件相机的稠密连续时间运动估计，在真实世界数据集 EVIMO2 上将合成数据预训练模型的零样本性能提升了 29%。

**[Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](video_understanding/occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)**

:   本文从动作检测的视角重新审视遮挡步态识别问题，提出GaitMoE方法通过时序专家混合(MTE)自适应构建动作锚点和动作专家混合(MAE)生成动作提议，仅使用ID标签进行端到端训练即可有效应对各种遮挡场景，并构建了首个统一的遮挡步态数据库OccGait。

**[On the Utility of 3D Hand Poses for Action Recognition](video_understanding/on_the_utility_of_3d_hand_poses_for_action_recognition.md)**

:   提出 HandFormer，一种轻量级多模态 Transformer，将密集采样的 3D 手部姿态（捕捉细粒度动作）与稀疏采样的 RGB 帧（提供场景语义）结合，通过 micro-action 时序分解和 trajectory 编码高效建模手-物交互，在 Assembly101 和 H2O 上达到 SOTA，且纯 pose 模型以 5× 更少 FLOPs 超越已有骨架方法。

**[OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](video_understanding/onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)**

:   本文深入分析了端到端3D跟踪器中检测与跟踪任务之间性能冲突的根本原因——二者在正样本分配上的微妙差异导致了分类梯度的矛盾，并提出OneTrack通过梯度协调、查询分组和注意力掩码等策略，首次实现了检测和跟踪在统一特征表示下的无冲突联合优化，在nuScenes上取得了SOTA性能。

**[Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition](video_understanding/optimizing_factorized_encoder_models_time_and_memory_reduction_for_scalable_and_.md)**

:   本文通过冻结 ViViT 因子化编码器中的空间 Transformer 并引入合理的时间 Transformer 初始化策略和紧凑的适配器模块，在保持甚至略微提升精度的同时大幅降低了训练成本和内存消耗，为资源受限的研究者提供了更高效的动作识别训练方案。

**[PiTe: Pixel-Temporal Alignment for Large Video-Language Model](video_understanding/pite_pixel-temporal_alignment_for_large_video-language_model.md)**

:   提出 PiTe 模型，通过物体运动轨迹在像素级别实现视频与语言的时空对齐，构建 PiTe-143k 数据集，在零样本 QA、时序定位和密集描述任务上大幅超越现有方法。

**[R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)**

:   提出 R²-Tuning，通过在冻结 CLIP 的后几层反向递归附加轻量 R² Block（仅 1.5% 总参数），实现查询调制的空间池化和粗到细的时序精炼，在 6 个 VTG 基准 3 个任务上以 2.7M 参数超越了需要额外时序骨干网络的 SOTA 方法。

**[Referring Atomic Video Action Recognition](video_understanding/referring_atomic_video_action_recognition.md)**

:   提出"基于文本引用的原子视频动作识别"（RAVAR）新任务和 RefAVA 数据集（36,630 实例），以及 RefAtomNet 方法，通过跨流 agent 注意力融合视觉、文本和位置-语义三路 token，在 mAP 上比最佳基线 BLIPv2 提升 3.85%/3.17%。

**[Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data](video_understanding/rethinking_video-text_understanding_retrieval_from_counterfactually_augmented_da.md)**

:   提出反事实增强数据检索（RCAD）任务和 Feint6K 数据集，揭示 SOTA 视频文本模型在动作语义理解上远落后于人类（InternVideo 58.2% vs 人类 95.2%），并提出 LLM-teacher 通过 LLM 知识蒸馏改善动作嵌入学习。

**[RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](video_understanding/rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)**

:   提出 RGNet 将长视频时序定位的片段检索和时序定位两个阶段深度统一到单一网络中，通过 RG-Encoder 的稀疏注意力和对比片段采样实现端到端优化，在 MAD 和 Ego4D 上取得 SOTA。

**[SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](video_understanding/sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)**

:   SA-DVAE 首次将特征解耦引入骨架零样本动作识别，通过双头 VAE 将骨架特征分离为语义相关和语义无关两个独立部分，仅用语义相关部分与文本对齐，配合对抗性总相关惩罚增强解耦效果，在 NTU RGB+D 60/120 和 PKU-MMD 三个基准上达到 SOTA。

**[SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](video_understanding/safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)**

:   SAFNet 提出选择性对齐融合策略，通过金字塔解码器联合精炼有价值区域掩码和跨曝光光流，仅在有价值区域进行精确对齐后显式融合 HDR 图像，在 Kalantari 17 和自建 Challenge123 数据集上超越 SOTA 的同时推理速度快一个数量级。

**[SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](video_understanding/sea-raft_simple_efficient_accurate_raft_for_optical_flow.md)**

:   SEA-RAFT 通过混合拉普拉斯损失(MoL)、直接回归初始光流和刚性流预训练三项改进，在保持简洁架构的同时实现了 SOTA 精度，并比现有方法快 2.3× 以上。

**[Self-Supervised Any-Point Tracking by Contrastive Random Walks](video_understanding/self-supervised_any-point_tracking_by_contrastive_random_walks.md)**

:   提出 GMRW（Global Matching Random Walk），将全局匹配 Transformer 架构与对比随机游走自监督目标结合，首次在无标注的情况下实现了强劲的"任意点跟踪"（TAP）性能，并设计 label warping 数据增强来避免 Transformer 的捷径解。

**[SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild](video_understanding/semtrack_a_large-scale_dataset_for_semantic_tracking_in_the_wild.md)**

:   提出 SemTrack 数据集和 SemTracker 方法，将传统目标跟踪从"定位目标在哪里"扩展到"理解目标在做什么"——跟踪目标的同时捕获其语义轨迹（与谁/什么交互、何时何地如何交互），并引入元学习策略应对长尾交互类别的挑战。

**[SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking](video_understanding/slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)**

:   SLAck 提出在多目标跟踪的关联阶段早期统一融合语义、位置和外观三种线索，通过轻量级时空目标图（STOG）学习隐式运动先验和跨线索协同，无需后处理启发式规则，在开放词汇 MOT 和 TAO TETA 基准上显著提升新类别跟踪性能。

**[SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow](video_understanding/spamming_labels_efficient_annotations_for_the_trackers_of_tomorrow.md)**

:   提出 SPAM 视频标注引擎，将合成数据预训练、伪标签自训练和基于图层级的主动学习相结合，仅需 3-20% 的人工标注量即可产生接近 GT 质量的多目标跟踪标注。

**[Spherical World-Locking for Audio-Visual Localization in Egocentric Videos](video_understanding/spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)**

:   提出球面世界锁定（Spherical World-Locking, SWL）框架，通过将多模态感知流隐式变换到世界锁定的球面坐标系中，消除自身运动带来的挑战，实现更精准的第一人称视频中的音视觉定位。

**[Text-Guided Video Masked Autoencoder](video_understanding/text-guided_video_masked_autoencoder.md)**

:   提出文本引导掩码策略（TGM）利用自然语言描述替代运动先验来掩码视频显著区域，并统一 MAE 与视频-文本对比学习，在五个动作识别和一个自中心数据集上取得最佳相对性能。

**[TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning](video_understanding/timecraft_navigate_weakly-supervised_temporal_grounded_video_question_answering_.md)**

:   本文提出一种双向推理框架TimeCraft来解决弱监督时序定位视频问答（temporal grounded VQA）任务，通过构建两条对称的推理路径（前向：时序定位→回答；反向：回答→时序定位）并用循环一致性约束提供自监督信号，在不需要时序标注的情况下同时定位回答依据的视频片段并给出正确答案。

**[Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](video_understanding/towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)**

:   提出异构模型数据集压缩（HMDC）方法，通过同时使用两个结构不同的模型（如 ConvNet 和 ViT）进行数据集压缩，并设计梯度平衡模块和互蒸馏机制，生成对各种模型普遍适用的压缩图像，解决传统方法过度适配单一模型的问题。

**[Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance](video_understanding/tracking_meets_lora_faster_training_larger_model_stronger_performance.md)**

:   LoRAT 首次将 LoRA 引入视觉目标跟踪，通过解耦位置编码（共享空间 + 独立类型嵌入）和纯 MLP 检测头两个 LoRA-友好设计，使得在实验室级资源上训练 ViT-g 骨干的跟踪器成为可能，在 LaSOT 上达到 0.762 SUC（新 SOTA），最轻变体 LoRAT-B-224 以 209 FPS 运行。

**[UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation](video_understanding/uniinr_event-guided_unified_rolling_shutter_correction_deblurring_and_interpolat.md)**

:   提出 UniINR 框架，利用统一的时空隐式神经表征（INR）从单张卷帘快门模糊帧和配对事件流中，一次性同时完成卷帘快门校正、去模糊和任意帧率的视频帧插值。

**[Vamos: Versatile Action Models for Video Understanding](video_understanding/vamos_versatile_action_models_for_video_understanding.md)**

:   提出 Vamos 框架，以大语言模型作为推理器，灵活统一视觉嵌入和通用文本描述作为视频表征，发现纯文本表征在多个视频理解基准上一致性地取得竞争甚至更优性能，并设计 Token Bottleneck Model 实现可解释证据选择与 5 倍推理加速。

**[VideoMamba: Spatio-Temporal Selective State Space Model](video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)**

:   提出基于纯 Mamba 架构的视频识别模型 VideoMamba（KAIST 版），通过设计时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM）来有效处理视频中非序列空间信息与序列时间信息的复杂交互，以线性复杂度实现了与 Transformer 竞争的性能。

**[VideoMamba: State Space Model for Efficient Video Understanding](video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)**

:   将 Mamba 的选择性状态空间模型创新性地适配到视频领域，提出纯 SSM 架构的 VideoMamba，以线性复杂度实现高效的时空上下文建模，在短视频和长视频理解任务上均展现出优越性能。

---

## 🎯 目标检测 { #object_detection }

**[Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](object_detection/adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   本文提出一种两步共形预测框架用于多目标检测的不确定性量化：第一步生成类别标签的共形预测集合以处理分类错误，第二步基于集成和分位数回归生成自适应的边界框不确定性区间，在保证覆盖率的同时提供实际可用的紧致预测区间。

**[Adaptive Multi-task Learning for Few-Shot Object Detection](object_detection/adaptive_multi-task_learning_for_few-shot_object_detection.md)**

:   本文提出了一种自适应多任务学习方法(MTL-FSOD)，通过精度驱动的梯度平衡器动态调整分类和定位任务的梯度比例来缓解两者的冲突，并引入基于 CLIP 的知识蒸馏和分类精化方案来增强各任务的能力，在多个小样本检测基准上取得了一致的性能提升。

**[AugDETR: Improving Multi-scale Learning for Detection Transformer](object_detection/augdetr_improving_multi-scale_learning_for_detection_transformer.md)**

:   本文提出 AugDETR（Augmented DETR），通过混合注意力编码器（Hybrid Attention Encoder）扩大可变形编码器的感受野并引入全局上下文特征增强特征表示，再通过编码器混合交叉注意力（Encoder-Mixing Cross-Attention）自适应利用多层编码器信息加速收敛，在 COCO 上为 DINO、AlignDETR、DDQ 分别带来 1.2/1.1/1.0 AP 的提升。

**[BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](object_detection/bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)**

:   提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。

**[Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](object_detection/bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)**

:   提出 Bridge Past and Future (BPF) 方法，通过伪标签桥接过去阶段、注意力机制排除未来潜在物体，并结合双教师蒸馏（Distillation with Future），解决增量目标检测中跨阶段信息不对称导致的优化目标不一致问题。

**[Can OOD Object Detectors Learn from Foundation Models?](object_detection/can_ood_object_detectors_learn_from_foundation_models.md)**

:   SyncOOD 提出一种自动化数据策展方法，利用 LLM 想象语义新颖的 OOD 概念，通过 Stable Diffusion Inpainting 在 ID 图像上进行区域级编辑合成场景级 OOD 样本，再经 SAM 精炼框和特征相似度过滤后训练轻量 MLP 分类器，在多个 OOD 检测基准上以极少量合成数据大幅超越 SOTA。

**[DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](object_detection/damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)**

:   DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。

**[DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](object_detection/dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)**

:   提出动态空间剪枝（DSP）策略，在多级 3D 检测器的解码器中逐级移除已检测到大物体区域的体素特征，使检测器能以高空间分辨率处理场景、大幅提升小目标检测精度（ScanNet 小目标 mAP@0.25 从 27.5% 提升到 44.8%），同时通过剪枝将显存降低为同分辨率方法的 1/5。

**[GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](object_detection/gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)**

:   提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。

**[LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](object_detection/lami-detr_open-vocabulary_detection_with_language_model_instruction.md)**

:   提出 LaMI-DETR，通过利用 GPT 生成视觉概念描述和 T5 挖掘类间视觉相似性关系，解决开放词汇目标检测中概念表示不足和基类过拟合两大问题，在 OV-LVIS 上以 43.4 的 rare AP 超越前最佳方法 7.8 个点。

**[MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)**

:   提出 MutDet，一种面向遥感旋转目标检测的互优化预训练框架，通过双向交叉注意力融合 object embeddings 与 encoder 特征、对比对齐损失、以及辅助孪生头，系统性地缓解了检测预训练中 object embeddings 与 detector features 之间的特征差异问题。

**[Nonverbal Interaction Detection](object_detection/nonverbal_interaction_detection.md)**

:   首次系统性研究人类非语言交互（手势、表情、注视、姿态、触碰），提出大规模数据集 NVI、新任务 NVI-DET 和基于双重多尺度超图的检测模型 NVI-DEHR，在非语言交互检测和 HOI 检测任务上均取得最优性能。

**[On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines](object_detection/on_calibration_of_object_detectors_pitfalls_evaluation_and_baselines.md)**

:   本文系统性地揭示了当前目标检测器校准研究中评估框架、评估指标和温度缩放（Temperature Scaling）使用方面的重大缺陷，提出了原则性的联合评估框架以及专为目标检测定制的后处理校准方法（Platt Scaling和Isotonic Regression），证明了正确设计和评估的后处理校准器远优于近期训练时校准方法。

**[OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)**

:   提出 OpenKD 模型，从模态（视觉+文本）、语义（seen vs. unseen）、语言（多样化文本）三个维度开放 prompt 多样性，通过多模态 prototype set、辅助关键点-文本插值和 LLM 文本解析，实现通用的 zero- and few-shot keypoint detection，在 Animal Pose、AwA、CUB、NABird 上取得 SOTA。

**[Plain-Det: A Plain Multi-Dataset Object Detector](object_detection/plain-det_a_plain_multi-dataset_object_detector.md)**

:   Plain-Det 提出了一个简洁灵活的多数据集目标检测框架，通过语义空间校准、类感知查询组合器和基于难度的动态采样策略，在 COCO 上达到 51.9 mAP（匹配当时 SOTA），并可灵活扩展到新数据集且保持鲁棒性能。

**[Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](object_detection/portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)**

:   提出一种利用**伪多视角视频**来训练前馈式单图4D头部合成器的新学习范式：先用合成数据学一个3D头部合成器将单目视频转为多视角，再利用伪多视角视频通过**跨视角自重演**学习4D合成器，避免了对3DMM的过度依赖，在重建保真度、几何一致性和运动控制精度上大幅超越先前方法。

**[Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](object_detection/projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)**

:   提出点-轴（Point-Axis）表示方法，将旋转目标的位置（点集）和方向（轴编码）解耦，配合 Max-Projection Loss 和 Cross-Axis Loss 实现无需额外标注的优化，并基于此设计 Oriented DETR 模型，解决传统旋转框表示的损失不连续问题。

**[Rectify the Regression Bias in Long-Tailed Object Detection](object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)**

:   首次揭示并系统解决长尾目标检测中被忽视的**回归偏差**问题：稀有类别的类别专属(class-specific)回归头参数因样本不足导致泛化能力差，通过添加额外的类别不可知(class-agnostic)回归分支进行权衡，在LVIS等数据集上取得了SOTA性能。

**[ReGround: Improving Textual and Spatial Grounding at No Cost](object_detection/reground_improving_textual_and_spatial_grounding_at_no_cost.md)**

:   通过将 GLIGEN 中 Gated Self-Attention (GSA) 与 Cross-Attention (CA) 的串行连接改为并行连接（网络重连），在不引入任何新参数、不需要微调、不增加计算开销的前提下，显著缓解了文本定位与空间定位之间的权衡问题。

**[Responsible Visual Editing](object_detection/responsible_visual_editing.md)**

:   定义"负责任视觉编辑"新任务，提出CoEditor认知编辑器，通过感知-行为双阶段认知过程将有害图像转换为负责任的版本，同时最小化修改。

**[SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](object_detection/shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)**

:   针对组合时序定位任务中现有方法负样本构造不合理、DETR 模型对负查询无法产生合理显著性响应的问题，提出利用 LLM（GPT-3.5 Turbo）生成语义可行的分层硬负样本，并设计粗到细的显著性排序策略建立视频片段与层次负查询之间的多粒度语义关系，显著提升组合泛化能力。

**[Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](object_detection/stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)**

:   针对点监督时序动作定位中稀疏标注导致的动作边界语义模糊问题，提出逐步多粒度边界检测器（SMBD），通过背景锚点生成器（BAG）和双边界检测器（DBD）为训练提供细粒度的边界监督信号，在THUMOS'14等数据集上达到SOTA。

**[TAPTR: Tracking Any Point with Transformers as Detection](object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)**

:   TAPTR 将 Tracking Any Point (TAP) 任务重新建模为类 DETR 的检测问题，将每个跟踪点表示为包含位置和内容的 point query，通过多层 Transformer 解码器逐层优化，结合 cost volume 和滑动窗口特征更新策略，在 TAP-Vid 基准上达到 SOTA 且推理速度更快。

**[Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](object_detection/tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)**

:   提出张量模板匹配（TTM）算法，通过对称张量场将模板在所有旋转下的信息整合为固定数量的相关计算，使得计算复杂度与旋转精度无关，在3D断层扫描图像中实现快速且准确的目标检测与旋转估计。

**[Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](object_detection/towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)**

:   构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。

**[Visible and Clear: Finding Tiny Objects in Difference Map](object_detection/visible_and_clear_finding_tiny_objects_in_difference_map.md)**

:   SR-TOD 首次将图像自重建机制引入目标检测，发现重建差异图与微小目标之间的强相关性，并设计差异图引导的特征增强（DGFE）模块，在自建反无人机数据集 DroneSwarms 和 VisDrone2019、AI-TOD 上均取得显著提升。

**[WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](object_detection/walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)**

:   本文提出Walker——首个自监督多目标跟踪器，通过构建准稠密的时序物体外观图（temporal appearance graph），设计多正样本对比损失优化图上的随机游走来学习实例相似度，并引入互斥连接约束和运动约束双向游走推理策略，在MOT17、DanceTrack和BDD100K上达到自监督跟踪的竞争性能，且在标注需求减少400倍的情况下仍超越之前的自监督方法。

**[Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](object_detection/weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)**

:   提出 WSCL 框架：利用 LLM 生成多样文本描述 + 扩散模型生成对应图像 + 弱检测器分解短语生成伪标框，构建密集合成三元组（image, description, bbox），配合组合对比学习显著提升语言引导目标检测性能，OmniLabel 上 GLIP-T 提升 +5.0AP。

**[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](object_detection/yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)**

:   YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。

---

## 🖼️ 图像恢复 { #image_restoration }

**[A New Dataset and Framework for Real-World Blurred Images Super-Resolution](image_restoration/a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)**

:   针对现有盲超分方法在处理含模糊（散焦/运动模糊）图像时过度纹理化、破坏模糊区域感知质量的问题，构建了包含近3000张模糊图像的ReBlurSR数据集，并提出PBaSR框架，通过双分支解耦训练（CDM）和基于权重插值的跨分支融合（CFM），在不增加任何推理开销的前提下，同时提升模糊图像和普通图像的超分效果，LPIPS提升0.02~0.10。

**[Accelerating Image Super-Resolution Networks with Pixel-Level Classification](image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)**

:   提出PCSR——首个像素级计算资源分配的超分方法，用轻量MLP分类器逐像素判断恢复难度并分配到不同容量的上采样器，在PSNR几乎不掉的情况下将FLOPs压低至原始模型的18%~57%，大幅优于现有patch级方法ClassSR和ARM。

**[Asymmetric Mask Scheme for Self-supervised Real Image Denoising](image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)**

:   提出非对称掩码方案 AMSNet，训练时用单掩码、推理时用多掩码互补，突破了 blind spot network 对网络感受野的结构限制，在真实图像自监督去噪任务上取得 SOTA。

**[BAMM: Bidirectional Autoregressive Motion Model](image_restoration/bamm_bidirectional_autoregressive_motion_model.md)**

:   提出 BAMM（双向自回归运动模型），通过统一生成掩码建模和自回归建模的混合注意力掩码策略，在一个框架中同时实现高质量运动生成、自适应长度预测和零样本运动编辑，在 HumanML3D 和 KIT-ML 上全面超越 SOTA。

**[Blind Image Deblurring with Noise-Robust Kernel Estimation](image_restoration/blind_image_deblurring_with_noise-robust_kernel_estimation.md)**

:   本文提出一种基于噪声鲁棒核估计函数和深度图像先验（DIP）的盲去模糊方法，通过设计能在强噪声下仍能准确估计模糊核的核估计函数，结合多核估计方案处理未知噪声水平，在模拟和真实图像上取得了优越的去模糊性能。

**[BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion](image_restoration/brushnet_a_plug-and-play_image_inpainting_model_with_decomposed_dual-branch_diff.md)**

:   提出 BrushNet，一种即插即用的双分支扩散模型图像修复架构，通过将遮罩图像特征提取与图像生成解耦到独立分支，实现逐层像素级特征注入，在图像质量、遮罩区域保持和文本对齐三方面全面超越已有方法。

**[Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](image_restoration/contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)**

:   针对红外图像超分辨率的特殊挑战，提出 CoRPLE 框架，利用 Contourlet 变换进行多尺度多方向的红外频谱残差增强，并引入基于视觉语言模型的提示学习范式来捕获红外图像的固有特征，在红外 SR 任务上达到 SOTA 性能。

**[DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](image_restoration/denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)**

:   提出 DenoiSplit，首个将语义图像分解（image splitting）和无监督去噪（unsupervised denoising）联合解决的方法，通过在层次化 VAE 中整合像素噪声模型和改进的 KL 散度损失加权策略，在荧光显微镜图像上实现了端到端的去噪+分解，性能显著优于先去噪再分解的串行方案。

**[Domain-Adaptive Video Deblurring via Test-Time Blurring](image_restoration/domain-adaptive_video_deblurring_via_test-time_blurring.md)**

:   提出基于扩散模糊模型的测试时域适应方法，通过从模糊视频中检测相对清晰区域作为伪清晰图像，并生成域自适应的模糊条件来合成训练对，实现在未知域上对去模糊模型的微调，在 5 个真实数据集上最高提升 7.54dB。

**[EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](image_restoration/edformer_transformer-based_event_denoising_across_varied_noise_levels.md)**

:   EDformer 提出了一种基于 Transformer 的逐事件去噪模型，通过学习事件之间的时空相关性来处理不同噪声水平下的事件相机噪声，并首次构建了包含 21 个噪声等级的真实世界事件去噪数据集 ED24。

**[Efficient Cascaded Multiscale Adaptive Network for Image Restoration](image_restoration/efficient_cascaded_multiscale_adaptive_network_for_image_restoration.md)**

:   ECMA 提出了一种高效级联多尺度自适应网络，通过局部自适应模块（LAM）动态调整卷积核来处理空间变化的退化，并以级联多尺度的方式捕捉不同尺度的特征，在去模糊、去噪和超分辨率等多种图像复原任务上以 1.2×-9.7× 的计算量减少实现了与 SOTA 可比甚至更优的性能。

**[Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)**

:   发现 Diffusion Transformer 中 query-key 交互存在显著冗余（尤其在去噪早期），提出 Attention Mediator 机制将注意力复杂度降至线性，并设计逐步动态调整策略，在 SiT-XL/2 上实现 SOTA FID 2.01，同时减少计算量。

**[Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](image_restoration/exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)**

:   提出首个基于学习的多帧ToF深度去噪框架，通过双相关性估计模块（利用帧内和帧间相关性）和置信度引导的残差回归模块，有效利用多帧ToF数据之间的关联来指导噪声去除，在强噪声区域显著优于现有单帧方法。

**[Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography](image_restoration/joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)**

:   提出 JDM-HDRNet，通过联合 RGB-光谱分解模型从低分辨率多光谱图像（Lr-MSI）中提取 shading、reflectance 和材质语义三种先验，将它们分别融入 HDRNet 以增强动态范围、色彩映射和语义网格专家学习，并构建了首个 RGB-高光谱配对的 Mobile-Spec 数据集。

**[Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](image_restoration/learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)**

:   本文提出 Exhaustive Correlation Transformer (ECT)，通过光谱方向非连续3D切分策略 (SD3D) 建模统一的空间-光谱相关性，并通过动态低秩映射模块 (DLRM) 捕获多token间的线性依赖关系，在光谱超分辨率任务上以最少的参数量和最低的推理延迟实现了 SOTA 性能。

**[Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams](image_restoration/learning_to_robustly_reconstruct_dynamic_scenes_from_low-light_spike_streams.md)**

:   本文针对脉冲相机在低光环境下信息稀疏导致重建困难的问题，提出了一种双向循环重建框架，其核心是光鲁棒表示（LR-Rep）通过全局脉冲间隔（GISI）聚合时域信息，配合特征融合模块提取时序特征，并构建了专门的低光高速数据集，在合成和真实数据上均大幅超越现有方法。

**[MambaIR: A Simple Baseline for Image Restoration with State-Space Model](image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)**

:   本文首次将 Mamba（选择性状态空间模型）引入底层图像修复任务，通过设计残差状态空间块（RSSB）中的局部卷积增强和通道注意力机制，解决了 vanilla Mamba 在 2D 图像上的局部像素遗忘和通道冗余问题，在图像超分辨率和去噪任务上以线性复杂度实现了与 Transformer 方法相当甚至更优的性能（SR 上超过 SwinIR 0.45dB）。

**[MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration](image_restoration/moe-diffir_task-customized_diffusion_priors_for_universal_compressed_image_resto.md)**

:   提出 MoE-DiffIR，首个基于扩散模型的通用压缩图像复原（CIR）框架，通过混合专家（MoE）Prompt 模块从 Stable Diffusion 中挖掘任务定制化的扩散先验，结合 Visual-to-Text 适配器激活 SD 的跨模态生成先验，并构建了覆盖 7 种编解码器 × 3 个压缩级别共 21 种退化的首个通用 CIR 基准数据集。

**[OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](image_restoration/oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)**

:   针对双重 JPEG 压缩图像恢复问题，提出 OAPT，通过预测两次压缩之间的像素偏移量，将每个 8×8 block 中的四种不同模式进行聚类分组后分别进行自注意力处理，在双重 JPEG 恢复任务上超越 SOTA 方法 0.16 dB。

**[Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](image_restoration/overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)**

:   本文提出 ODM 框架，通过协同失配正则化（cooperative mismatch regularization）和逐层权重裁剪校正（weight clipping correction）两个简单策略，在不引入推理时动态模块的前提下解决 SR 网络量化中的分布失配问题，以极小的额外开销达到 SOTA。

**[Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](image_restoration/pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)**

:   提出成对距离蒸馏框架，通过蒸馏专用模型和通用模型之间的内部和模型间距离关系，实现无监督真实世界图像超分辨率的退化自适应。

**[Restoring Images in Adverse Weather Conditions via Histogram Transformer](image_restoration/restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)**

:   提出 Histoformer，一种基于直方图自注意力机制的高效 Transformer，通过将空间特征按像素强度排序分箱（bin），在箱内和箱间执行自注意力，实现动态范围的空间注意力以高效处理天气退化像素，配合动态范围卷积和 Pearson 相关性损失，在去雪/去雨雾/去雨滴三大任务上统一建模并达到 SOTA。

**[Rethinking Image Super-Resolution from Training Data Perspectives](image_restoration/rethinking_image_super-resolution_from_training_data_perspectives.md)**

:   从训练数据角度重新思考图像超分辨率，提出自动化数据评估流水线构建 DiverSeg 数据集（低分辨率但高质量、目标多样的图像），证明在该数据集上训练的 SR 模型可以超越使用高分辨率数据集（DF2K、LSDIR）训练的模型。

**[Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)**

:   提出 FPro，通过频域视角的 prompt learning 指导图像复原：使用 Gated Dynamic Decoupler 将特征解耦为低频/高频分量，再通过 Dual Prompt Block（HPM + LPM）分别对两个频带注入可学习 prompt 并与解码器特征交互，在去雨、去雨滴、去摩尔纹、去模糊、去雾 5 个任务上全面超越 SOTA。

**[Spatially-Variant Degradation Model for Dataset-free Super-resolution](image_restoration/spatially-variant_degradation_model_for_dataset-free_super-resolution.md)**

:   提出首个无需数据集训练的空间变化退化模型 SVDSR，每个像素的退化核由可学习的原子核字典的线性组合表示，系数矩阵通过模糊集的隶属函数从图像纹理信息推导，在 MAP 框架下用 Monte Carlo EM 算法推断，$2\times$ 超分平均提升 1 dB。

**[Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](image_restoration/towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)**

:   本文首次提出事件相机引导的低光视频增强与去模糊联合任务，构建了基于分光棱镜的真实世界数据集 RELED，并设计了包含事件引导可变形时序对齐 (ED-TFA) 和频谱滤波跨模态增强 (SFCM-FE) 两个核心模块的端到端框架，在 PSNR 上比此前最佳方法提升 1.2dB 以上。

**[TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](image_restoration/ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)**

:   本文提出 TTT-MIM，在训练阶段联合优化监督去噪损失和自监督掩码图像建模（MIM）损失，在测试时通过最小化 MIM 自监督损失对单张噪声图像进行适应性微调，从而显著提升对分布外噪声（如真实相机噪声、显微镜噪声）的去噪性能，且速度远超零样本方法。

**[Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](image_restoration/unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)**

:   提出 UDU-Net，将低光视频增强建模为 MAP 优化问题并展开为深度网络，通过 Intra/Inter 子网分别处理空间（光照）和时序（一致性）退化，支持无配对训练和人类感知反馈的可控增强。

---

## 🏥 医学图像 { #medical_imaging }

**[A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image](medical_imaging/a_cephalometric_landmark_regression_method_based_on_dual-encoder_for_high-resolu.md)**

:   本文提出 D-CeLR，一种基于双编码器（Dual-Encoder）的端到端回归方法，仅利用 Transformer 编码器设计特征提取+参考编码器+精调编码器的三阶段架构，实现从粗到细的头影测量标志点检测，在 Mean Radical Error (MRE) 和 2mm Success Detection Rate (SDR) 指标上显著超越现有 SOTA。

**[A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](medical_imaging/a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)**

:   本文提出 SRRM-ViT，通过在 ViT 中引入统计旋转不变性增强机制(SRRM)，自适应选择关键区域并融合直方图统计特征，实现了对食管癌内镜超声图像中任意径向位置病灶的无偏细粒度分类，在临床和公开数据集上取得了显著性能提升。

**[Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](medical_imaging/adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)**

:   针对医学图像无监督配准中噪声、遮挡等干扰因素导致的虚假重建误差问题，提出了一个自适应对应关系评分框架（AdaCS），通过学习像素级的对应置信度图来重新加权误差残差，以即插即用方式一致提升三种主流配准架构在三个数据集上的性能。

**[Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](medical_imaging/alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)**

:   提出 AD-MT（Alternate Diverse Mean Teacher），通过随机周期性交替更新两个教师模型 + 基于熵的冲突调和策略，在半监督医学分割中解决 confirmation bias 问题，在 ACDC/LA/Pancreas 上全面超越 SOTA。

**[Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization](medical_imaging/architecture-agnostic_untrained_network_priors_for_image_reconstruction_with_fre.md)**

:   提出三种与架构无关的频率正则化技术（带宽受限输入、带宽可控上采样、Lipschitz 正则化卷积层），统一解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，在 MRI 重建任务中显著缩小不同架构间的性能差距。

**[Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](medical_imaging/brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)**

:   本文提出 Brain-ID，一种对比度无关的脑解剖表征学习模型，通过"轻度到重度"的受试者内图像合成策略，在全合成数据上训练获得对MRI对比度、分辨率、方向和伪影鲁棒的解剖特征，仅需一层适配即可在四种下游任务和六个公开数据集上达到 SOTA。

**[CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos](medical_imaging/cardiacnet_learning_to_reconstruct_abnormalities_for_cardiac_disease_assessment_.md)**

:   提出基于重建的心脏疾病评估框架 CardiacNet，通过 Consistency Deformation Codebook (CDC) 和 Consistency Deformation Discriminator (CDD) 学习正常与异常心脏超声视频之间的结构和运动差异，在射血分数预测、肺动脉高压和房间隔缺损分类三个任务上达到 SOTA。

**[Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild](medical_imaging/chameleon_a_data-efficient_generalist_for_dense_visual_prediction_in_the_wild.md)**

:   提出 Chameleon，一个基于 meta-learning 和 token matching 的数据高效视觉通才模型，仅需几十张标注图像即可适应全新的密集预测任务（包括医学图像、视频、3D 等），在六个下游基准上显著超越现有通才方法。

**[CheX: Interactive Localization and Region Description in Chest X-rays](medical_imaging/chex_interactive_localization_and_region_description_in_chest_x-rays.md)**

:   提出ChEX——一个同时支持文本提示和边界框查询的交互式胸部X光解释模型，通过DETR风格的prompt检测器和多任务联合训练，在9个胸部X光任务上与SOTA竞争，同时提供独特的定位可解释性和用户交互能力。

**[Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model](medical_imaging/co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)**

:   提出一种上下文条件化的联合扩散模型，能够同时合成组织病理学细胞核图像、语义标签和距离图，通过点图（centroid layout）和文本提示两种条件实现对合成过程的精确控制，并生成高质量的实例级标签用于下游核分割和分类任务。

**[Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction](medical_imaging/domesticating_sam_for_breast_ultrasound_image_segmentation_via_spatial-frequency.md)**

:   本文提出 SF-RecSAM 模型，通过空间-频率特征融合模块弥补SAM在低级特征提取上的不足，并设计双假校正器（Dual False Corrector）利用不确定性估计识别并修正假阳性和假阴性区域，在BUSI和UDIAT两个乳腺超声数据集上显著超越SOTA方法。

**[Energy-induced Explicit Quantification for Multi-modality MRI Fusion](medical_imaging/energy-induced_explicit_quantification_for_multi-modality_mri_fusion.md)**

:   提出能量引导的显式传播与对齐框架E²PA，通过能量引导的层级融合（EHF）和能量正则化的空间对齐（ESA）两个模块，显式量化并优化多模态MRI融合中的模态间依赖传播和信息流一致性，在三个公开数据集上超越SOTA。

**[GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation](medical_imaging/gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)**

:   提出基于异构图的全模态生物医学表征学习框架 GTP-4o，通过异构图嵌入显式建模跨模态关系，利用图提示机制补全缺失模态，并设计知识引导的层次化跨模态聚合，在胶质瘤分级和生存预测任务上取得SOTA。

**[I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](medical_imaging/i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)**

:   提出 I-MedSAM，将 SAM 的强泛化能力与隐式神经表示（INR）的连续空间预测优势结合，通过频率适配器增强边界高频信息、不确定性引导采样精细化分割，仅用 1.6M 可训练参数即超越现有离散和隐式方法。

**[Improving Medical Multi-modal Contrastive Learning with Expert Annotations](medical_imaging/improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)**

:   提出 eCLIP，通过整合放射科专家的眼动注视热力图作为额外监督信号，结合 mixup 增强和课程学习策略，在不修改 CLIP 核心架构的前提下增强医学多模态对比学习的表征质量。

**[Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data](medical_imaging/is_user_feedback_always_informative_retrieval_latent_defending_for_semi-supervis.md)**

:   发现用户反馈在域适应中并非总是有益——偏向纠正错误预测的"负偏反馈"(NBF)会导致现有半监督域适应方法性能下降，并提出 Retrieval Latent Defending 方法，通过在每个 mini-batch 中加入伪标签防御样本来平衡监督信号。

**[OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding](medical_imaging/ophnet_a_large-scale_video_benchmark_for_ophthalmic_surgical_workflow_understand.md)**

:   构建了OphNet——目前最大规模的眼科手术视频基准数据集（2278个视频、285小时、66种手术类型、102种手术阶段、150种精细操作），支持手术类型识别、阶段识别、时序定位和阶段预测四大任务，其规模约为现有最大手术工作流分析基准的20倍。

**[Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](medical_imaging/pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)**

:   提出 PEMP 框架，将病理学先验知识（视觉样例 + 文本描述）融入 patch 级和 slide 级的 prompt 中，结合 CLIP 进行多实例 prompt learning，在少样本弱监督 WSI 分类任务上平均超越 SOTA 方法 4%。

**[RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](medical_imaging/radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)**

:   提出 RadEdit，一种基于扩散模型的医学图像编辑方法，通过引入 edit mask 和 keep mask 的双重掩码机制，打破数据中的虚假关联（spurious correlations），生成高质量的合成测试集来压力测试（stress-test）生物医学视觉模型对数据集偏移的鲁棒性。

**[Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](medical_imaging/radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)**

:   提出 X-Gaussian，首个将 3D 高斯泼溅（3DGS）应用于 X 射线新视角合成的框架，通过设计辐射高斯点云模型（替代球谐函数）和角度位姿长方体均匀初始化策略（替代 SfM），在性能上超越 SOTA NeRF 方法 6.5 dB 的同时，实现 73× 推理加速和仅 15% 的训练时间。

**[Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments](medical_imaging/shape-guided_configuration-aware_learning_for_endoscopic-image-based_pose_estima.md)**

:   利用柔性机器人的3D形状先验引导图像特征学习，通过部件级几何表示提取和动态形状变形机制，实现了高精度的内窥镜图像柔性机器人位姿估计，在外部朝向和内部弯曲角度估计上显著超越了关键点、骨架和直接回归等基线方法。

**[NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)**

:   NePhi 提出用神经隐式函数（SIREN）替代传统体素形变场来表示图像配准中的形变，通过编码器预测潜码实现快速推理、通过实例优化提升精度，在肺部和脑部 3D 配准任务中匹配 SOTA 精度的同时将训练内存降低 5 倍，且天然产生近似微分同胚的光滑形变。

**[TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data](medical_imaging/tip_tabular-image_pre-training_for_multimodal_classification_with_incomplete_dat.md)**

:   提出 TIP 框架，通过掩码表格重建、图像-表格匹配和对比学习三种自监督任务，对表格数据和图像进行联合预训练，学习对不完整表格数据鲁棒的多模态表征，用于下游分类任务。

**[Topology-Preserving Downsampling of Binary Images](medical_imaging/topology-preserving_downsampling_of_binary_images.md)**

:   提出首个基于离散优化（整数规划）的拓扑保持二值图像下采样方法，通过将下采样像素的黑白决策编码为布尔变量、拓扑保持作为硬约束、与原图相似度作为目标函数来求解，保证下采样结果具有与原图完全相同的Betti数（连通分量数和孔洞数），同时保持与传统方法竞争性的像素级相似度。

**[UMBRAE: Unified Multimodal Brain Decoding](medical_imaging/umbrae_unified_multimodal_brain_decoding.md)**

:   提出UMBRAE，通过通用脑编码器将fMRI信号与图像特征对齐后送入冻结的MLLM，实现多模态脑解码（描述、定位、检索、视觉重建），并创新性地引入跨被试训练策略，使单一模型服务多个被试且优于单被试模型。

**[Unleashing the Power of Prompt-driven Nucleus Instance Segmentation](medical_imaging/unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)**

:   提出 PromptNucSeg 框架，通过训练一个 prompter 自动生成细胞核中心点 prompt，并微调 SAM 进行逐核分割，同时引入相邻核作为 negative prompt 解决重叠核分割问题，无需复杂后处理即在三个 benchmark 上达到 SOTA。

**[Unsupervised Multi-modal Medical Image Registration via Invertible Translation](medical_imaging/unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)**

:   本文提出 INNReg，通过可逆神经网络将多模态医学图像翻译为单模态，再利用单模态图像进行配准，结合基于归一化互信息的 barrier 损失函数，在 MRI T1/T2 和 MRI/CT 数据集上取得了优于现有方法的配准精度。

---

## 📦 模型压缩 { #model_compression }

**[A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](model_compression/a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)**

:   首个面向视频快照压缩成像（Video SCI）重建任务的低比特量化框架Q-SCI，通过高质量特征提取模块、精确视频重建模块和Transformer分支的query/key分布偏移操作，在4-bit量化下实现7.8倍理论加速且性能仅下降2.3%。

**[AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](model_compression/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)**

:   提出自适应对数底量化器AdaLog，通过可搜索的对数底替代固定log₂/log√2量化器来处理ViT中post-Softmax和post-GELU激活的幂律分布，并设计快速渐进组合搜索(FPCS)策略高效确定量化超参，在极低比特(3/4-bit)下显著优于现有ViT PTQ方法。

**[Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)**

:   本文提出 AdaSense，利用预训练扩散模型的零样本后验采样能力来量化重建不确定性，从而自适应地选择最优测量矩阵，在人脸图像、MRI 和 CT 等多个领域实现了无需额外训练的自适应压缩感知，性能超越非自适应方法甚至基于 PCA 的最优非自适应方案。

**[Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)**

:   本文提出"自适应选择采样-重建对"($\mathcal{H}_{1.5}$)框架，利用超分辨率空间生成模型量化高频贝叶斯不确定性，为每个输入数据选择最佳的采样掩码-重建网络对，在理论和实验上同时优于非自适应联合优化方法（$\mathcal{H}_1$）和自适应采样方法（$\mathcal{H}_2$），在人脸图像和多线圈 MRI 重建中取得显著 SSIM 提升。

**[Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](model_compression/adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)**

:   本文提出了一种基于特征分布统计对齐的对抗鲁棒知识蒸馏方法，通过减小 student 和 teacher 模型在对抗样本和干净样本之间的特征方差差距(variance gap)来提升 student 模型的对抗鲁棒性，发现鲁棒精度与方差差距存在强负相关线性关系。

**[Anytime Continual Learning for Open Vocabulary Classification](model_compression/anytime_continual_learning_for_open_vocabulary_classification.md)**

:   提出 AnytimeCL 框架，通过部分微调 CLIP 最后一个 transformer block 并动态加权融合微调模型与原始模型的预测，实现任意时刻接收样本、任意标签集推理的开放词汇持续学习。

**[Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](model_compression/auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)**

:   本文提出 Auto-DAS，一个基于进化算法的自动化代理发现框架，用于免训练的蒸馏感知架构搜索（DAS），通过在由学生内在统计量和师生交互统计量构成的搜索空间中自动发现最优代理指标，避免了手工设计代理的局限性，在 ResNet、ViT、NAS-Bench-101/201 等多种架构和搜索空间上达到了 SOTA 的排序相关性和搜索精度。

**[BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](model_compression/basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)**

:   本文提出 BaSIC 框架，通过学习神经图像压缩（NIC）系统的贝叶斯网络结构，同时控制骨干网络复杂度和自回归单元的并行计算能力，首次实现了对 NIC 全流程的计算可扩展性控制。

**[Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](model_compression/bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)**

:   提出双向对称的立体图像压缩框架 BiSIC，采用 3D 卷积联合编解码器和跨维度熵模型，在 PSNR 和 MS-SSIM 上均超越传统标准和已有学习方法，同时消除了单向方法中左右视图压缩质量不平衡的问题。

**[Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](model_compression/category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)**

:   提出 CAMP 方法，通过可学习投影器蒸馏与类别中心适应网络的协同组合，在广义持续类别发现（GCCD）场景中显著提升了新类别学习与旧知识保持之间的平衡。

**[ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration](model_compression/else_efficient_deep_neural_network_inference_through_line-based_sparsity_explora.md)**

:   提出基于行稀疏性探索的事件抑制方法ELSE，利用激活图中相邻行的空间相关性来减少非零激活（事件）数量，在目标检测和姿态估计任务上实现3.14~6.49倍的计算节省，且可与现有事件抑制方法互补。

**[Improving Knowledge Distillation via Regularizing Feature Direction and Norm](model_compression/improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)**

:   提出 ND 损失函数，通过同时对齐学生特征方向至教师类均值方向并鼓励学生产生大范数特征，显著提升了现有知识蒸馏方法在 ImageNet、CIFAR100 和 COCO 上的性能。

**[Improving Zero-Shot Generalization for CLIP with Variational Adapter](model_compression/improving_zero-shot_generalization_for_clip_with_variational_adapter.md)**

:   提出 Prompt-based Variational Adapter (PVA)，通过变分适配器将 base 和 novel 类别样本在隐空间中分离，采用分治策略分别处理，结合残差连接增强 novel 类别的迁移能力，在广义零样本学习和跨数据集迁移学习基准上达到 SOTA。

**[Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images](model_compression/is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)**

:   提出 SCAR（Selective-distillation for Class and Architecture-agnostic unleaRning），一种无需保留集的近似遗忘算法，通过 Mahalanobis 距离引导遗忘样本特征向量向最近错误类分布迁移，并利用 OOD 图像蒸馏保持模型性能。

**[Isomorphic Pruning for Vision Models](model_compression/isomorphic_pruning_for_vision_models.md)**

:   提出 Isomorphic Pruning，通过将网络子结构建模为图并按图同构性分组，在同构组内独立排序剪枝，解决异构子结构间重要性不可比的问题，在 ViT 和 CNN 上均取得优于专门设计的剪枝方法的效果。

**[Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation](model_compression/leveraging_hierarchical_feature_sharing_for_efficient_dataset_condensation.md)**

:   提出层级记忆网络（HMN），将数据蒸馏中的合成数据存储为三层结构（数据集级-类级-实例级记忆），通过层级化特征共享提升存储效率，并利用实例级剪枝进一步去除冗余，仅用低GPU内存的 batch-based loss 即超越所有基线方法。

**[MetaAug: Meta-Data Augmentation for Post-Training Quantization](model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)**

:   提出 MetaAug，一种基于元学习的训练后量化（PTQ）方法，通过可学习的变换网络对校准数据进行增强，并以双层优化框架同时优化变换网络和量化模型，有效缓解 PTQ 在小校准集上的过拟合问题。

**[PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](model_compression/papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)**

:   提出 PaPr，利用轻量级 ConvNet 的卷积特征图生成 Patch Significance Map (PSM)，在**无需重训练**的情况下对 ViT/ConvNet/混合架构进行**一步式** patch 剪枝，实现显著的计算量削减（视频场景最高 3.7× FLOPs 减少），且精度损失极小。

**[PQ-SAM: Post-training Quantization for Segment Anything Model](model_compression/pq-sam_post-training_quantization_for_segment_anything_model.md)**

:   本文提出PQ-SAM，首个专为Segment Anything Model定制的训练后量化方法，通过分组激活分布变换(GADT)和两阶段异常值层次聚类(OHC)方案解决SAM的高度不对称激活分布和有害异常值问题，将4-bit量化的SAM推进到可用水平。

**[Simple Unsupervised Knowledge Distillation With Space Similarity](model_compression/simple_unsupervised_knowledge_distillation_with_space_similarity.md)**

:   CoSS 提出在无监督知识蒸馏中，除了常规的**特征维度余弦相似度**外，额外引入一个**空间维度余弦相似度（Space Similarity）**损失——将特征矩阵转置后在维度方向上对齐，从而弥补 $L_2$ 归一化导致的流形结构信息丢失，以极简的方式在多个 UKD benchmark 上达到 SOTA。

**[SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](model_compression/spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)**

:   提出 SpaceJAM，一种仅约 16K 可训练参数的无监督图像联合对齐方法，无需正则化项或 atlas 维护，在 SPair-71K 和 CUB 数据集上匹配现有方法的对齐能力同时实现 10 倍以上加速。

**[Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](model_compression/token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)**

:   提出 ToCom（Token Compensator），一个模型算术框架的轻量插件，通过快速的参数高效自蒸馏获得，可在推理时直接插入任意下游已训练模型以弥补 token 压缩度不匹配造成的性能损失，无需重新训练。

**[Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer](model_compression/uncertainty-driven_spectral_compressive_imaging_with_spatial-frequency_transform.md)**

:   本文提出 Specformer，通过并行的空间局部窗口自注意力（LWSA）和频率域自注意力（FWSA）模块充分捕获高光谱图像（HSI）的空间稀疏性和光谱间相似性先验，并引入不确定性驱动的损失函数增强网络对纹理丰富和边缘区域的重建能力，在模拟和真实 HSI 数据集上以更低计算量超越 SOTA。

**[UNIC: Universal Classification Models via Multi-teacher Distillation](model_compression/unic_universal_classification_models_via_multi-teacher_distillation.md)**

:   提出UNIC框架，通过改进的多教师蒸馏策略（包括梯形投影器和教师丢弃技术），将多个互补预训练模型的知识融合到单一学生模型中，实现跨任务的通用分类。

---

## 📊 LLM 评测 { #llm_evaluation }

**[ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](llm_evaluation/colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)**

:   提出 ColorMNet，一种基于记忆机制的时空特征传播网络，通过预训练大视觉模型引导的特征提取（PVGFE）、基于记忆的特征传播（MFP）和局部注意力（LA）三个模块，在显著降低 GPU 显存消耗（仅需 1.9G）的同时实现了优于 SOTA 的视频上色效果。

**[Deep Cost Ray Fusion for Sparse Depth Video Completion](llm_evaluation/deep_cost_ray_fusion_for_sparse_depth_video_completion.md)**

:   本文提出 RayFusion 框架，通过在 cost volume 上沿射线方向施加 self-attention 和 cross-attention 实现时序融合，以仅 1.15M 参数在 KITTI、VOID、ScanNetV2 三个数据集上全面超越或持平 SOTA 稀疏深度补全方法。

**[Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)**

:   提出分布对齐（DA）损失将测试时特征分布拉回源域分布，配合域偏移检测机制，在非 i.i.d. 动态数据流和连续域偏移场景下大幅超越现有 TTA 方法。

**[Eliminating Warping Shakes for Unsupervised Online Video Stitching](llm_evaluation/eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)**

:   定义了视频拼接中的"warping shake"新问题（图像拼接扩展到视频时非重叠区域的时域抖动），提出StabStitch首个无监督在线视频拼接框架，通过拼接轨迹生成与平滑同时实现视频拼接和稳定，达到实时28.2ms/帧。

**[EvSign: Sign Language Recognition and Translation with Streaming Events](llm_evaluation/evsign_sign_language_recognition_and_translation_with_streaming_events.md)**

:   首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

**[Gradient-Regularized Out-of-Distribution Detection](llm_evaluation/gradient-regularized_out-of-distribution_detection.md)**

:   提出 GReg/GReg+，通过正则化 OOD 评分函数的输入梯度范数来学习评分流形的局部平滑性，并结合基于能量评分的聚类采样策略选取高信息量辅助样本，在 CIFAR 和 ImageNet OOD 检测基准上取得 SOTA。

**[Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](llm_evaluation/image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)**

:   本文提出 IFMatch，在传统图像级弱到强一致性范式基础上引入特征级扰动并构建三分支结构，通过置信度策略区分朴素/困难样本，在多个 SSL 基准上显著提升已有方法（如 FixMatch、FreeMatch 等）的性能。

**[Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems](llm_evaluation/imaging_interiors_an_implicit_solution_to_electromagnetic_inverse_scattering_pro.md)**

:   提出基于隐式神经表示（INR）的电磁逆散射问题（EISP）求解方案，通过将散射体的相对介电常数建模为连续隐式表示并在前向框架中优化，有效避免了逆估计的困难和离散化导致的低分辨率问题。

**[Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation](llm_evaluation/instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es.md)**

:   本文提出一种基于概率图模型的噪声率估计方法，可自动估计训练集标签噪声率，并利用估计值指导样本选择策略的课程设计，可无缝集成到 DivideMix、InstanceGM 等 SOTA 噪声标签学习方法中，在合成和真实世界基准上提升其分类精度。

**[Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](llm_evaluation/learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)**

:   提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。

**[MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](llm_evaluation/merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)**

:   提出单阶段注意力沙漏网络MERLiN，从单张图像联合估计空间变化BRDF参数并进行物理正确的重打光，首次利用重打光图像驱动光度立体方法实现单图法向估计，弥合了Shape from Shading与Photometric Stereo之间的鸿沟。

**[OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](llm_evaluation/ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)**

:   提出 OGNI-DC，通过"优化引导的神经迭代"（OGNI）框架，结合 ConvGRU 迭代精炼深度梯度场和可微深度积分器（DDI）来实现深度补全，同时达到 SOTA 精度和强泛化能力。

**[PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](llm_evaluation/petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)**

:   构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

**[R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations](llm_evaluation/r2-bench_benchmarking_the_robustness_of_referring_perception_models_under_pertur.md)**

:   提出 R²-Bench，一个系统评估指代感知模型（RPM）在各种扰动下鲁棒性的综合基准，包含完整的扰动分类体系、通用的扰动合成工具箱和基于 LLM 的自动化评估代理 R²-Agent，覆盖五大关键任务，揭示了当前 RPM 在噪声条件下的脆弱性。

**[SIGMA: Sinkhorn-Guided Masked Video Modeling](llm_evaluation/sigma_sinkhorn-guided_masked_video_modeling.md)**

:   本文提出 SIGMA，通过引入投影网络将 masked video modeling 的重建目标从像素级升级为可学习的深层特征聚类分配，利用 Sinkhorn 算法的最优传输实施高熵正则化避免坍缩，在 10 个数据集 3 个 benchmark 上全面超越 VideoMAE 等 SOTA 方法。

**[Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](llm_evaluation/spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)**

:   提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。

**[Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](llm_evaluation/sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)**

:   提出可对齐视频检索（Alignable Video Retrieval, AVR）任务，通过 DRAQ 对齐质量指标从大规模视频数据库中识别并检索出最适合与查询视频进行时序对齐的视频，同时提出特征上下文化方法提升对齐性能。

**[Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](llm_evaluation/versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)**

:   首次定义 Versatile Incremental Learning (VIL) 场景——后续任务的类别或领域增量类型未知，并提出 ICON 框架，通过 CAST 损失控制学习方向避免与历史任务冲突、IC 增量分类器动态扩展输出节点处理跨域同类覆写问题，在三个基准上全面超越现有 CIL/DIL 方法。

**[VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](llm_evaluation/visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)**

:   VisFocus提出了一种提示引导的视觉编码方法用于OCR-free文档理解：通过将用户提示（prompt）直接注入视觉编码器的patch merging层（ViLMA层），配合局部掩码提示建模（LMPM）预训练任务，使视觉编码器学会聚焦于与提示相关的文本区域，在多个文档VQA基准上达到同规模SOTA。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Adaptive Multi-head Contrastive Learning](self_supervised/adaptive_multihead_contrastive_learning.md)**

:   本文提出AMCL（Adaptive Multi-head Contrastive Learning），通过多个投影头产生不同特征视角，配合基于MLE推导的自适应温度机制为每对样本独立加权，有效解决了多种数据增强下正负样本相似度分布重叠的问题，一致提升SimCLR、MoCo和Barlow Twins的性能。

**[COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation](self_supervised/coho_context-sensitive_city-scale_hierarchical_urban_layout_generation.md)**

:   提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。

**[Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](self_supervised/efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)**

:   提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。

**[Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](self_supervised/exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)**

:   提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

**[FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](self_supervised/flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)**

:   提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。

**[InfMAE: A Foundation Model in the Infrared Modality](self_supervised/infmae_a_foundation_model_in_the_infrared_modality.md)**

:   提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。

**[MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](self_supervised/marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)**

:   本文提出MarineInst，一个面向海洋图像分析的基础模型，能够同时输出实例掩码和语义描述；并构建了MarineInst20M——迄今最大的海洋图像数据集（2000万张），支持从图像级场景理解到区域级实例理解的多层次海洋视觉分析任务。

**[PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer](self_supervised/posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)**

:   提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。

**[PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](self_supervised/promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)**

:   提出PromptCCD框架，利用高斯混合模型（GMM）作为提示池，实现在无标签数据流中的持续新类别发现，同时缓解灾难性遗忘。

**[Rethinking Unsupervised Outlier Detection via Multiple Thresholding](self_supervised/rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)**

:   提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。

**[Revisiting Supervision for Continual Representation Learning](self_supervised/revisiting_supervision_for_continual_representation_learning.md)**

:   挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。

**[SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](self_supervised/scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)**

:   提出 SCPNet，通过模内自监督学习（intra-modal self-supervised learning）、相关性网络和一致性特征图投影三个关键组件的协同，首次在卫星-地图等大模态差距数据集上实现了有效的无监督跨模态单应性估计，MACE 比监督方法 MHN 低 14%。

**[Self-supervised Video Copy Localization with Regional Token Representation](self_supervised/self-supervised_video_copy_localization_with_regional_token_representation.md)**

:   提出了一种自监督视频拷贝定位框架，通过在 Vision Transformer 中引入 Regional Token 捕获局部区域信息，并利用传递性（Transitivity Property）自动生成训练数据，在无需人工标注的情况下超越了有监督方法的性能。

**[ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](self_supervised/vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)**

:   ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。

**[WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](self_supervised/wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)**

:   提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。

---

## 🎬 视频生成 { #video_generation }

**[BlazeBVD: Make Scale-Time Equalization Great Again for Blind Video Deflickering](video_generation/blazebvd_make_scale-time_equalization_great_again_for_blind_video_deflickering.md)**

:   提出 BlazeBVD，利用经典 Scale-Time Equalization (STE) 在光照直方图空间提取 deflickering 先验（滤波光照图、曝光图、闪烁帧索引），将复杂的视频时空学习简化为 2D 空间网络逐帧处理 + 轻量 3D 时序一致性网络，在盲视频去闪烁任务上实现 SOTA 质量且推理速度比基线快 10 倍以上。

**[DragAnything: Motion Control for Anything using Entity Representation](video_generation/draganything_motion_control_for_anything_using_entity_representation.md)**

:   提出DragAnything，利用扩散模型的隐空间特征作为实体表征（Entity Representation）来实现实体级运动控制，解决了现有轨迹驱动方法仅拖拽像素而无法精确控制目标对象运动的问题，在VIPSeg上实现SOTA的FVD/FID指标，用户研究中运动控制投票超出DragNUWA 26%。

**[DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing](video_generation/dreammotion_space-time_self-similar_score_distillation_for_zero-shot_video_editi.md)**

:   提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。

**[Evaluating Text-to-Visual Generation with Image-to-Text Generation](video_generation/evaluating_text-to-visual_generation_with_image-to-text_generation.md)**

:   提出VQAScore，利用VQA模型替代CLIP来评估文本-视觉生成质量，在复杂组合性提示上大幅超越CLIPScore，并发布GenAI-Bench基准。

**[Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](video_generation/exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)**

:   本文首次探索预训练文本到视频（T2V）扩散模型的视觉特征用于视频理解任务，提出 VD-IT 框架，通过文本引导的图像投影和视频特定噪声预测两项关键设计，从固定的 T2V 扩散模型中提取具有优越时序语义一致性的视觉特征，在 R-VOS 四大基准上超越了使用判别式预训练视频骨干网络（如 Video Swin Transformer）的 SOTA 方法。

**[FreeInit: Bridging Initialization Gap in Video Diffusion Models](video_generation/freeinit_bridging_initialization_gap_in_video_diffusion_models.md)**

:   发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。

**[Kalman-Inspired Feature Propagation for Video Face Super-Resolution](video_generation/kalman-inspired_feature_propagation_for_video_face_super-resolution.md)**

:   本文提出 KEEP 框架，借鉴卡尔曼滤波原理在隐空间中递归融合前帧先验与当前帧观测，实现视频人脸超分辨率中面部细节的高保真恢复与时序一致性，在 VFHQ 数据集上 PSNR 超过此前最优方法 0.8 dB。

**[MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](video_generation/magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)**

:   提出首个统一视频生成与编辑的多对齐扩散模型 MagDiff，通过主体驱动对齐、自适应提示对齐和高保真对齐三种策略，在单一无微调框架中同时实现高质量视频生成与编辑。

**[MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](video_generation/mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)**

:   提出 MOFA-Video，通过设计多个领域感知的运动场适配器（MOFA-Adapter），在冻结的 Stable Video Diffusion 上实现多域可控图像动画，支持手绘轨迹、人脸关键点等多种控制信号及其零样本组合。

**[PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](video_generation/physdreamer_physics-based_interaction_with_3d_objects_via_video_generation.md)**

:   利用视频生成模型中隐含的物理动力学先验，为静态3D高斯对象估计空间变化的杨氏模量材料场，从而实现物理合理的交互式3D动力学合成。

**[RealViformer: Investigating Attention for Real-World Video Super-Resolution](video_generation/realviformer_investigating_attention_for_real-world_video_super-resolution.md)**

:   本文系统研究了空间注意力和通道注意力在真实世界视频超分辨率（RWVSR）中的行为差异，发现通道注意力对退化伪影更鲁棒但会导致特征冗余，据此提出了带有改进通道注意力（ICA）和通道注意力融合（CAF）模块的 RealViformer，以更少的参数和更快的速度达到 SOTA。

**[SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)**

:   提出SV3D，将图像到视频扩散模型适配为多视图合成和3D生成，利用视频模型的泛化能力和多视图一致性，并引入显式相机控制。

**[VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](video_generation/vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)**

:   提出利用预训练视频扩散模型（EMU Video）作为多视图数据引擎，通过微调使其生成3D一致的多视图视频，从而构建约300万合成数据训练前馈式3D生成模型VFusion3D，实现从单张图片秒级生成3D资产，用户偏好率超过90%。

**[Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](video_generation/videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)**

:   提出Videoshop——一种免训练的局部语义视频编辑方法，用户可通过任意图像编辑工具修改视频首帧，系统基于噪声外推扩散反演和隐变量归一化技术，自动将编辑传播到所有帧，同时保持语义、空间和时序一致性，在10个指标上超越6个基线方法。

---

## 🤖 机器人/具身智能 { #robotics }

**[AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation](robotics/aff-ttention_affordances_and_attention_models_for_short-term_object_interaction_.md)**

:   提出 STAformer 架构和两个基于 affordance 的模块（环境 affordance 数据库 + 交互热点），将第一人称视频中的短期物体交互预测（STA）在 Ego4D 和 EPIC-Kitchens 上提升了 30-45% 的相对性能。

**[An Economic Framework for 6-DoF Grasp Detection](robotics/an_economic_framework_for_6-dof_grasp_detection.md)**

:   提出EconomicGrasp框架，通过发现密集监督中的歧义问题（ambiguity problem）是性能与资源矛盾的根源，设计经济监督范式（保留所有视角但裁剪角度/深度）和焦点表示模块（交互式抓取头+复合评分），在GraspNet-1Billion上以1/4训练时间、1/8内存成本超越SOTA约3AP。

**[Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation](robotics/decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)**

:   提出 Decomposed VQ-VAE (DVQ-VAE)，通过将手部分解为六个部分分别编码到独立码本，并设计双阶段解码策略（先姿态后位置），在四个基准数据集上质量指标相对提升约14.1%。

**[DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control](robotics/disco_embodied_navigation_and_interaction_via_differentiable_scene_semantics_and.md)**

:   提出 DISCO 框架，通过可微分场景语义表示和双层粗-细动作控制，在 ALFRED 基准上实现具身导航与交互的显著性能提升（未见场景成功率超越 SOTA +8.6%，且无需逐步指令）。

**[GraspXL: Generating Grasping Motions for Diverse Objects at Scale](robotics/graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)**

:   提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。

**[Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos](robotics/hierarchically_structured_neural_bones_for_reconstructing_animatable_objects_fro.md)**

:   提出层次化神经骨骼（Hierarchical Neural Bones）框架，通过树状结构的骨骼系统以粗到细的方式分解物体运动，从随手拍摄的视频中重建可操控的高质量 3D 模型。

**[Learning Cross-Hand Policies of High-DOF Reaching and Grasping](robotics/learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)**

:   提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。

**[LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)**

:   本文提出VLN-Copilot框架，让视觉语言导航智能体在粗粒度（简短模糊）指令下遇到困惑时主动向LLM求助，LLM作为副驾驶实时生成细粒度导航指导，在两个粗粒度VLN数据集上显著提升导航成功率。

**[Prioritized Semantic Learning for Zero-shot Instance Navigation](robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)**

:   提出Prioritized Semantic Learning (PSL)方法，通过语义增强的Agent架构、优先语义训练策略和语义扩展推理方案，显著提升零样本目标/实例导航中Agent的语义感知能力，在ObjectNav和新提出的InstanceNav任务上实现SOTA。

**[QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](robotics/quar-vla_vision-language-action_model_for_quadruped_robots.md)**

:   首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。

**[ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments](robotics/realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)**

:   提出 ReALFRED 基准，使用 150 个真实世界 3D 扫描的多房间可交互环境替代 ALFRED 的合成单房间场景，提供 30,696 条自由格式语言指令，揭示了现有具身指令跟随方法在真实环境中性能显著下降的问题。

**[See and Think: Embodied Agent in Virtual Environment](robotics/see_and_think_embodied_agent_in_virtual_environment.md)**

:   提出 STEVE，一个基于视觉感知、语言指令和代码动作三大组件的 Minecraft 开放世界具身智能体，通过 STEVE-21K 数据集微调 LLaMA-2 并结合视觉编码器和技能数据库，在科技树解锁和方块搜索任务上大幅超越现有方法。

**[SemGrasp: Semantic Grasp Generation via Language Aligned Discretization](robotics/semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)**

:   提出SemGrasp方法，设计层次化VQ-VAE将抓取姿态离散为"方向-方式-精修"三个语义token，然后微调多模态大语言模型(MLLM)在统一语义空间中融合物体、抓取与语言，实现根据自然语言指令生成物理合理且语义一致的人类抓取姿态。

---

## 🛡️ AI 安全 { #ai_safety }

**[Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection](ai_safety/any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)**

:   提出 GAKer，首个可泛化到未知目标类别的定向对抗攻击生成器，通过在 UNet 中间层注入目标特征（latent infection）+ 余弦距离损失替代交叉熵实现类别无关训练，在未知类上的攻击成功率比 HGN 高 14.13%。

**[Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement](ai_safety/bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)**

:   提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

**[CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](ai_safety/clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)**

:   提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。

**[Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](ai_safety/fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)**

:   本文提出Self-Driven Fisher Calibration（SDFC），利用Fisher信息度量参数对不同分布的重要程度差异，在异质联邦学习场景中有效区分恶意后门客户端并进行参数校准，突破了现有防御方法依赖数据同质性和恶意节点少数假设的局限。

**[Event Trojan: Asynchronous Event-based Backdoor Attacks](ai_safety/genq_quantization_in_low_data_regimes_with_generative_synthetic_data.md)**

:   提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。

**[Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](ai_safety/noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)**

:   本文提出 CLIP-IFDL，一种基于 CLIP 的图像篡改检测与定位模型，通过实例感知的双流提示学习和伪造增强噪声适配器来弥补 CLIP 在篡改检测领域的提示缺失和伪造感知不足问题，将 CLIP 的开放世界泛化能力迁移到篡改检测任务中。

**[One-stage Prompt-based Continual Learning](ai_safety/one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](ai_safety/preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)**

:   从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。

**[Resilience of Entropy Model in Distributed Neural Networks](ai_safety/resilience_of_entropy_model_in_distributed_neural_networks.md)**

:   首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平，准确率仅下降约 2%。

**[SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](ai_safety/skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)**

:   提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。

**[Towards Multi-modal Transformers in Federated Learning](ai_safety/towards_multi-modal_transformers_in_federated_learning.md)**

:   提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。

**[Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](ai_safety/unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)**

:   本文揭示了随机神经网络（SNNs）在联邦学习中同样容易遭受梯度反演攻击，提出 ISG 方法通过将 SNN 的随机训练过程等价为传统 NN 训练的变体来重建训练数据，并引入特征约束策略提升重建保真度。

---

## 💬 LLM / NLP { #llm_nlp }

**[AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](llm_nlp/adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)**

:   在CLIP中同时引入静态（全局共享）和动态（逐图生成）两种可学习提示，用辅助异常检测数据训练后，在14个工业+医学异常检测数据集上实现零样本SOTA，核心在于"任务级+实例级"双层自适应的混合提示设计。

**[APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](llm_nlp/apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)**

:   本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。

**[Cultural Value Differences of LLMs: Prompt, Language, and Model Size](llm_nlp/cultural_value_differences_llms.md)**

:   本文使用 Hofstede 文化维度问卷系统性地研究 LLM 表达文化价值观的行为模式，发现提示语言（中文 vs 英文）和模型规模对文化价值差异的影响远大于模型架构差异和问题顺序变化。

**[FreestyleRet: Retrieving Images from Style-Diversified Queries](llm_nlp/freestyleret_retrieving_images_from_style-diversified_queries.md)**

:   提出首个风格多样化查询图像检索（Style-Diversified QBIR）任务及数据集DSR，设计了轻量即插即用的FreestyleRet框架，通过Gram矩阵提取查询的纹理/风格特征，构建风格空间并以此初始化prompt token，使冻结的视觉编码器能适配文本、草图、低分辨率、艺术画等多种查询风格的检索。

**[FunQA: Towards Surprising Video Comprehension](llm_nlp/funqa_towards_surprising_video_comprehension.md)**

:   构建了大规模反直觉视频问答基准 FunQA（4.3K 视频、312K QA 对），覆盖幽默/创意/魔术三类令人惊讶的视频，并提出 FunMentor 智能体通过多轮对话增强 VLM 的反常识推理能力。

**[PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](llm_nlp/promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)**

:   提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

**[Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos](llm_nlp/propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)**

:   VidAssist提出"提议-评估-搜索"三步框架，利用LLM作为知识库和评估工具，结合广度优先搜索算法，在教学视频的目标导向规划任务中以零/少样本方式超越全监督SOTA，few-shot在COIN上比全监督VLaMP高+7.7% SR。

**[Reprojection Errors as Prompts for Efficient Scene Coordinate Regression](llm_nlp/reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)**

:   本文提出 EGFS（Error-Guided Feature Selection）机制，利用低重投影误差区域作为 SAM 的 point prompts 扩展为语义掩码，迭代地筛选可靠训练样本，在 Cambridge Landmarks 和 Indoor6 数据集上以更小模型和更少训练时间超越现有无 3D 信息依赖的 SCR 方法。

**[RoadPainter: Points Are Ideal Navigators for Topology Transformer](llm_nlp/roadpainter_points_are_ideal_navigators_for_topology_transformer.md)**

:   提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。

**[Stripe Observation Guided Inference Cost-Free Attention Mechanism](llm_nlp/stripe_observation_guided_inference_cost-free_attention_mechanism.md)**

:   本文通过深入分析Transformer中注意力权重矩阵的条纹（stripe）模式现象，提出一种推理阶段完全无额外计算开销的注意力增强机制——仅在训练阶段通过辅助模块学习条纹引导的注意力修正，并在推理时将其重参数化融入标准注意力权重中，实现"免费午餐"式的性能提升。

**[Zero-Shot Object Counting with Good Exemplars (VA-Count)](llm_nlp/zeroshot_object_counting_with_good_exemplars.md)**

:   提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。

---

## 🎵 音频/语音 { #audio_speech }

**[Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](audio_speech/action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation](audio_speech/beat-it_beat-synchronized_multi-condition_3d_dance_generation.md)**

:   提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。

**[CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)**

:   提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。

**[ControlLLM: Augment Language Models with Tools by Searching on Graphs](audio_speech/controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)**

:   提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

**[Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)**

:   提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。

**[Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](audio_speech/latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)**

:   提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。

**[Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](audio_speech/listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)**

:   提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。

**[Siamese Vision Transformers are Scalable Audio-Visual Learners](audio_speech/siamese_vision_transformers_are_scalable_audio-visual_learners.md)**

:   提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。

---

## 📚 预训练 { #llm_pretraining }

**[Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)**

:   提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。

**[DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](llm_pretraining/dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)**

:   DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。

**[I Can't Believe It's Not Scene Flow!](llm_pretraining/i_canapost_believe_itaposs_not_scene_flow.md)**

:   揭示现有场景流方法在行人等小目标上的灾难性失败被现有评估指标所掩盖，提出类别感知且速度归一化的Bucket Normalized EPE评估协议，以及一个简单但SOTA的TrackFlow基线（检测器+跟踪器生成场景流），在行人运动描述上实现1.5倍提升。

**[Learning to Obstruct Few-Shot Image Classification over Restricted Classes](llm_pretraining/learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)**

:   提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。

**[Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)**

:   本文提出 PRO-Motion 分治框架，将文本到动作生成分解为三个阶段：LLM 驱动的动作规划（Plan）、基于脚本的姿态扩散生成（Posture）、以及全身平移旋转估计（Go），通过降低各阶段的复杂度实现了开放词汇的高质量动作生成。

**[PreLAR: World Model Pre-training with Learnable Action Representation](llm_pretraining/prelar_world_model_pre-training_with_learnable_action_representation.md)**

:   本文提出PreLAR，在无动作标签的视频上进行世界模型预训练时，通过从相邻帧编码隐式动作表示并设计动作-状态一致性损失来弥合无动作预训练与有动作微调之间的差距，显著提升了下游视觉控制任务的样本效率。

**[Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](llm_pretraining/prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)**

:   本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。

**[Scaling Backwards: Minimal Synthetic Pre-training?](llm_pretraining/scaling_backwards_minimal_synthetic_pre-training.md)**

:   提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。

---

## 🛰️ 遥感 { #remote_sensing }

**[Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](remote_sensing/adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)**

:   针对细粒度跨视角定位模型在新区域部署时精度下降的问题，提出基于知识自蒸馏的弱监督学习方法——通过模式化伪GT生成、粗粒度监督和离群值过滤三个策略，仅使用目标区域的地面-航拍图像对（无需精确GT），即可在VIGOR和KITTI上将定位误差降低12%~20%。

**[ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](remote_sensing/congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)**

:   提出 ConGeo，一种模型无关的单视图+跨视图对比学习框架，通过强制同一地点不同地面视角变体之间的特征一致性，使单一模型即可在任意朝向和任意视场角(FoV)下实现鲁棒的跨视图地理定位。

**[Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach](remote_sensing/cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach.md)**

:   构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。

**[Learning Representations of Satellite Images From Metadata Supervision](remote_sensing/learning_representations_of_satellite_images_from_metadata_supervision.md)**

:   本文提出了 SatMIP（Satellite Metadata-Image Pretraining），将卫星图像的元数据（如时间、地理位置、传感器信息等）表示为文本描述，通过图像-元数据对比学习任务在共享嵌入空间中对齐图像和元数据，学习到既包含视觉特征又编码语义信息的卫星图像表征，并进一步提出 SatMIPS（结合图像自监督和元数据监督），在多个遥感下游任务上超越了 SimCLR 等纯视觉自监督方法。

**[Masked Angle-Aware Autoencoder for Remote Sensing Images](remote_sensing/masked_angle-aware_autoencoder_for_remote_sensing_images.md)**

:   提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。

**[Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)**

:   提出首个弱监督的地面-卫星图像配准定位方法，通过卫星-卫星自监督训练旋转估计器、对比学习训练平移估计器，在无需精确GT姿态标签的条件下实现最佳跨区域泛化能力，超越大多数全监督SOTA方法。

---

## 📡 信号/通信 { #signal_comm }

**[Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](signal_comm/defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)**

:   本文构建了 Defect Spectrum 数据集，在四个工业基准之上提供精细的、语义丰富的、大规模的多类缺陷标注（125种缺陷类别，3518+1920张），并提出两阶段扩散生成器 Defect-Gen 在少样本条件下合成高质量多样性缺陷图像，合成数据将缺陷分割 mIoU 最高提升 9.85。

**[Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](signal_comm/optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)**

:   本文提出从双曝光 HDR 图像对中提取一种简洁的双曝光特征（DEF），并基于此构建了两个超轻量级光源估计器 EMLP 和 ECCC，在仅使用几百到几千个参数的情况下即可达到或超越需要数十万参数的先前方法的性能。

**[PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](signal_comm/pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)**

:   本文提出 PYRA，通过并行生成解耦的自适应调制权重并以 re-activation 策略调节待合并 token 的特征，实现了 Vision Transformer 在下游任务适配时同时兼顾训练效率（仅调 0.4% 参数）和推理效率（约 1.7-3.2 倍加速），性能与不压缩的 PEFT 方法持平甚至更优。

**[QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](signal_comm/querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)**

:   提出QueryCDR网络，通过可学习查询机制（DLQM）和两种可控调制模块（CCMB/CAMB），首次实现不同畸变程度的鱼眼图像在**不重训**的情况下进行高质量可控矫正。

**[RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](signal_comm/raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)**

:   提出 RAW-Adapter，通过输入级适配器（可学习 ISP 阶段）和模型级适配器（ISP 中间特征注入骨干网络），以极小参数量（0.2-0.8M）将 sRGB 预训练模型高效适配到 Camera RAW 图像，在正常光/暗光/过曝等多种光照条件下的检测和分割任务上达到 SOTA。

**[Unsupervised Exposure Correction](signal_comm/unsupervised_exposure_correction.md)**

:   提出首个无监督曝光校正（UEC）方法，利用ISP管线自由生成的多曝光序列让图像互为ground truth进行训练，设计仅含19K参数的像素级变换函数保留图像细节，在曝光校正和下游边缘检测上超越有监督SOTA。

---

## 🔬 可解释性 { #interpretability }

**[DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration](interpretability/detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)**

:   提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。

**[EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding](interpretability/egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)**

:   提出 EgoExo-Fitness 数据集，包含同步的第一人称和第三人称健身视频，提供两级时间边界标注和创新性的可解释动作评判标注（技术关键点验证、自然语言评论、质量评分），并构建五个基准任务。

**[Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](interpretability/improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)**

:   本文发现 Concept Bottleneck Models (CBMs) 中人工干预效率低下的原因在于干预时各概念独立处理、忽视了概念间关联，提出了一个轻量级的 Concept Intervention Realignment Module (CIRM)，在干预后自动重新对齐相关概念的预测值，将达到目标性能所需的干预次数最多减少 70%。

**[PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery](interpretability/plot_text-based_person_search_with_part_slot_attention_for_corresponding_part_di.md)**

:   提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。

**[POA: Pre-training Once for Models of All Sizes](interpretability/poa_pre-training_once_for_models_of_all_sizes.md)**

:   POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。

---

## 🔗 因果推理 { #causal_inference }

**[Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation](causal_inference/distill_gold_from_massive_ores_bi-level_data_pruning_towards_efficient_dataset_d.md)**

:   提出双层数据剪枝策略 BiLP，通过经验损失静态剪枝和基于因果效应 (ITE) 的动态剪枝，高效选择对数据集蒸馏最有价值的真实样本，以即插即用方式一致性提升现有蒸馏方法性能并降低计算开销。

**[Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](causal_inference/integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)**

:   提出 CMBRL 框架，在隐空间中发现马尔可夫毯（Markov Blanket）特征——目标变量的最小充分统计量——代替现有方法中仅选择因果/反因果变量的做法，构建不变预测机制实现跨域泛化。

**[Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](causal_inference/learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)**

:   本文提出了反事实偏差鲁棒推理数据集（CoBRa）和反事实思维链方法（CoCT），通过构造编辑后的知识图谱和图像内容来评估和缓解大型视觉语言模型（LVLM）中的知识偏差，使模型能够逐步推理而非依赖偏见知识，在需要知识偏差下推理的任务上显著优于现有方法。

**[Understanding Physical Dynamics with Counterfactual World Modeling](causal_inference/understanding_physical_dynamics_with_counterfactual_world_modeling.md)**

:   本文提出反事实世界建模（Counterfactual World Modeling, CWM），通过时序分解的遮蔽策略训练视频掩码预测器，并设计"反事实提示"机制从单一预训练模型中无需微调即可提取光流、分割、关键点等多种视觉结构，在物理动力学理解任务Physion基准上达到最优性能。

---

## 🕸️ 图学习 { #graph_learning }

**[Confidence Self-Calibration for Multi-Label Class-Incremental Learning](graph_learning/confidence_self-calibration_for_multi-label_class-incremental_learning.md)**

:   针对多标签类增量学习(MLCIL)中部分标签导致的过度自信预测和假阳性错误问题，提出 Confidence Self-Calibration (CSC) 框架，通过类增量图卷积网络(CI-GCN)校准标签关系 + 最大熵正则化校准置信度，在 MS-COCO 和 VOC 上大幅超越 SOTA。

**[GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](graph_learning/gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)**

:   提出首个全图卷积多标签识别模型 GKGNet，通过 Group KNN 机制动态构建标签与图像区域间的图结构，在 MS-COCO 和 VOC2007 上以更低计算量取得 SOTA。

**[SENC: Handling Self-collision in Neural Cloth Simulation](graph_learning/senc_handling_self-collision_in_neural_cloth_simulation.md)**

:   提出 SENC，通过基于 Global Intersection Analysis (GIA) 的自碰撞损失和自碰撞感知图神经网络，首次在自监督神经布料模拟中有效解决布料自碰撞问题。

**[Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching](graph_learning/synchronous_diffusion_for_unsupervised_smooth_non-rigid_3d_shape_matching.md)**

:   提出同步扩散正则化方法用于无监督非刚性3D形状匹配，核心思想是"在两个形状上同步地扩散同一函数应产生一致输出"，通过这一简单而高效的正则化可以显著提升现有深度功能映射方法的匹配平滑性，在FAUST、SCAPE、TOPKIDS等多个数据集上达到SOTA。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Grounding Language Models for Visual Entity Recognition](information_retrieval/grounding_language_models_for_visual_entity_recognition.md)**

:   提出 AutoVER——首个将多模态大语言模型（MLLM）应用于大规模视觉实体识别的方法，通过将检索能力集成到 MLLM 内部，结合对比训练和前缀树约束解码，在 Oven-Wiki 基准上大幅超越 PaLI-17B 等先前方法。

**[Multi-Label Cluster Discrimination for Visual Representation Learning](information_retrieval/multi-label_cluster_discrimination_for_visual_representation_learning.md)**

:   提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。

**[OneRestore: A Universal Restoration Framework for Composite Degradation](information_retrieval/onerestore_a_universal_restoration_framework_for_composite_degradation.md)**

:   提出 OneRestore，一种基于 Transformer 的通用图像复原框架，通过场景描述符引导的交叉注意力机制和复合退化复原损失，能在单一模型中自适应地处理低光照、雾、雨、雪及其任意组合的复合退化场景，并支持文本/视觉双模式的可控复原。

**[Towards Open-Ended Visual Recognition with Large Language Model](information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)**

:   提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。

---

## 🦾 LLM Agent { #llm_agent }

**[Agent3D-Zero: An Agent for Zero-shot 3D Understanding](llm_agent/agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)**

:   Agent3D-Zero 提出一个基于 VLM 的零样本 3D 场景理解 Agent 框架，通过鸟瞰图上的 Set-of-Line 视觉提示引导 VLM 主动选择观察视角，并综合多视角图像进行 3D 推理，在 ScanQA 等任务上超越了需要微调的 3D-LLM 方法。

**[HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning](llm_agent/hydra_a_hyper_agent_for_dynamic_compositional_visual_reasoning.md)**

:   （注：基于摘要的简要笔记）提出 HYDRA，一种多阶段动态组合式视觉推理框架，通过规划器（Planner）、强化学习认知控制器（RL Agent）和推理器（Reasoner）三模块协作，实现可靠且渐进式的视觉推理，在 RefCOCO/RefCOCO+、OK-VQA、GQA 等多个数据集上取得 SOTA。

**[VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](llm_agent/videoagent_a_memory-augmented_multimodal_agent_for_video_understanding.md)**

:   提出 VideoAgent，一个记忆增强的多模态 Agent，通过构建结构化记忆（temporal memory 存储事件描述 + object memory 存储物体跟踪状态）并利用 4 个工具与记忆交互，零样本完成长视频问答任务，在 NExT-QA 上平均 +6.6%、EgoSchema 上 +26.0%，接近 Gemini 1.5 Pro 的性能。

---

## 🎮 强化学习 { #reinforcement_learning }

**[AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale](reinforcement_learning/adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale.md)**

:   提出AdaGlimpse，利用Soft Actor-Critic强化学习从连续动作空间中选择任意位置和尺度的glimpse，结合弹性位置编码的ViT编码器实现多任务（重建/分类/分割）的主动视觉探索，以仅6%像素超越了使用18%像素的SOTA方法。

**[Octopus: Embodied Vision-Language Programmer from Environmental Feedback](reinforcement_learning/octopus_embodied_vision-language_programmer_from_environmental_feedback.md)**

:   提出 Octopus，一个具身视觉-语言编程模型，通过生成可执行代码来连接高层规划与底层操控，并引入 Reinforcement Learning with Environmental Feedback (RLEF) 训练方案来提升决策质量。

**[Visual Grounding for Object-Level Generalization in Reinforcement Learning](reinforcement_learning/visual_grounding_for_object-level_generalization_in_reinforcement_learning.md)**

:   利用视觉语言模型 (MineCLIP) 的 visual grounding 能力生成目标物体的 confidence map，通过奖励设计和任务表征两条路径将 VLM 知识迁移到强化学习中，实现对未见物体和指令的零样本泛化。

---

## 📈 时间序列 { #time_series }

**[Multi-person Pose Forecasting with Individual Interaction Perceptron and Prior Learning](time_series/multi-person_pose_forecasting_with_individual_interaction_perceptron_and_prior_l.md)**

:   本文提出 IAFormer（Interaction-Aware Pose Forecasting Transformer），通过设计交互感知模块（IPM）来评估每个人与事件的交互程度，并引入交互先验学习模块（IPLM）来积累高频交互模式的先验知识，从而实现语义层面的多人姿态预测，在多个多人场景数据集上显著超越现有方法。

**[OmniSat: Self-Supervised Modality Fusion for Earth Observation](time_series/omnisat_self-supervised_modality_fusion_for_earth_observation.md)**

:   提出OmniSat统一框架，通过模态特异编码器+跨模态对比自监督预训练，将多光谱时序（S2）、SAR时序（S1）、高分辨率单时相（SPOT/Aerial）等异构遥感数据融合为统一表示，在语义分割和作物分类上超越所有单模态和多模态基线。

**[Semantically Guided Representation Learning For Action Anticipation](time_series/semantically_guided_representation_learning_for_action_anticipation.md)**

:   提出 S-GEAR 框架，通过学习视觉动作原型并利用语言模型的语义关联来引导原型之间的几何关系，使模型理解动作间的语义互联性，从而提升动作预测性能，在 Epic-Kitchens 55/100、EGTEA Gaze+、50 Salads 四个基准上取得 SOTA 或极具竞争力的结果。

---

## 📐 优化/理论 { #optimization }

**[Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction](optimization/fine-grained_scene_graph_generation_via_sample-level_bias_prediction.md)**

:   提出样本级偏置预测方法 SBP，通过 Bias-Oriented GAN 利用物体对 union region 的上下文信息预测样本特异性纠偏向量，将粗粒度关系修正为细粒度关系，在 VG/GQA/VG-1800 上相比数据集级纠偏方法平均提升 5.6%/3.9%/3.2% 的 Average@K。

**[Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework](optimization/handling_the_non-smooth_challenge_in_tensor_svd_a_multi-objective_tensor_recover.md)**

:   提出基于可学习张量核范数的多目标张量恢复框架 (MOTC)，通过引入可学习酉矩阵替代固定变换来解决 t-SVD 方法在非光滑张量数据上的性能退化问题，并通过多目标优化有效利用张量各维度的低秩性。

---

## 👥 社会计算 { #social_computing }

**[Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)**

:   提出 DaSC 框架，通过分布感知的类中心估计（DaCC）和置信度感知的对比学习（SBCL + MIDL），同时解决长尾分布和噪声标签的联合问题，在 CIFAR 和真实噪声数据集上达到 SOTA。

**[GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering](social_computing/grace_graph-based_contextual_debiasing_for_fair_visual_question_answering.md)**

:   提出 GRACE（GRAph-based Contextual DEbiasing），一种基于图结构的上下文去偏方法，通过无监督上下文图学习和基于图的多样化 in-context example 选择，解决知识增强 VQA 系统中大语言模型继承的数据偏差问题。

---

## 💻 代码智能 { #code_intelligence }

**[DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation](code_intelligence/dreamstruct_understanding_slides_and_user_interfaces_via_synthetic_data_generati.md)**

:   提出利用代码生成合成结构化视觉数据（幻灯片和UI），用于训练理解模型，减少人工标注需求。

---

## 🗣️ 对话系统 { #dialogue }

**[BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](dialogue/bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)**

:   提出 BI-MDRG 框架，通过桥接图像历史信息来增强多模态对话中文本回复的图像 grounding 能力和连续图像回复中物体的一致性。

---

## 🌍 地球科学 { #earth_science }

**[Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization](earth_science/semi-supervised_video_desnowing_network_via_temporal_decoupling_experts_and_dist.md)**

:   提出首个半监督视频去雪框架 SemiVDN，通过物理先验引导的时序解耦专家模块和分布驱动的对比正则化，利用无标签真实雪景视频缩小合成-真实域差距，在合成与真实数据集上均超越现有方法。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Controllable Navigation Instruction Generation with Chain of Thought Prompting](llm_reasoning/controllable_navigation_instruction_generation_with_chain_of_thought_prompting.md)**

:   提出 C-Instructor，利用 LLM 的思维链提示实现风格和内容可控的导航指令生成，通过 CoTL（带地标的思维链）、STMT（空间拓扑建模）和 SMT（混合风格训练）三大机制，在四个室内外导航数据集上全面超越已有方法。

---

## 🔒 LLM 安全 { #llm_safety }

**[MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment](llm_safety/magr_manifold-aligned_graph_regularization_for_continual_action_quality_assessme.md)**

:   提出 MAGR 方法，通过流形对齐投影器和 Intra-Inter-Joint 图正则化器，解决持续动作质量评估（CAQA）中特征回放导致的旧特征与当前特征流形不对齐问题，在四个数据集上显著超越现有基线。

---

## 📖 NLP 理解 { #nlp_understanding }

**[SLIMER: Show Less, Instruct More - Enriching Prompts with Definitions and Guidelines for Zero-Shot NER](nlp_understanding/slimer_zero_shot_ner.md)**

:   SLIMER 通过在提示中注入实体定义和标注指南来增强 LLM 的零样本命名实体识别能力，仅用 391 个实体类别训练即可在从未见过的实体标签上达到与使用 13000+ 实体类别训练的 SOTA 方法相当的性能。

---

## ⚛️ 物理/科学计算 { #physics }

**[Robust Fitting on a Gate Quantum Computer](physics/robust_fitting_on_a_gate_quantum_computer.md)**

:   首次在真实门量子计算机（IonQ Aria）上实现鲁棒拟合：提出用于一维 $\ell_\infty$ 可行性检验的量子电路，填补了 Bernstein-Vazirani（BV）电路计算 Boolean influence 的关键空缺，并展示如何将一维 influence 累积到高维非线性模型（如基础矩阵估计）。

---

## 🎁 推荐系统 { #recommender }

**[AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)**

:   首次提出图像内容吸引力评估（ICAA）任务，区别于传统美学评估（IAA），设计了一套自动化数据集生成 + 吸引力估计 + 吸引力增强的完整 pipeline，用 Stable Diffusion + Textual Inversion 实现零人工标注的大规模数据集构建。

---

## 📂 其他 { #others }

**[3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views](others/3dfg-pifu_3d_feature_grids_for_human_digitization_from_sparse_views.md)**

:   本文提出 3DFG-PIFu，通过引入3D特征网格（3D Feature Grids）在整个 pipeline 中全局融合多视图特征，替代传统逐点局部融合方式，并结合迭代网格精炼机制和基于 SDF 的 SMPL-X 特征，显著超越现有稀疏视图人体数字化 SOTA 方法。

**[A Direct Approach to Viewing Graph Solvability](others/a_direct_approach_to_viewing_graph_solvability.md)**

:   本文对视图图（Viewing Graph）可解性问题提出了一种比以往更直接的新形式化方法，引入了新概念用于理解实际 SfM 图的可解性，并给出了更高效的不可解情况检测与分解算法。

**[A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](others/a_framework_for_efficient_model_evaluation_through_stratific.md)**

:   提出一个统计框架，通过分层（stratification）、采样设计（sampling）和估计器（estimation）三个组件的协同设计，在仅标注少量测试样本的情况下精确估计CV模型准确率，最高可实现10倍的效率增益（即用1/10的标注量达到同等精度）。

**[ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](others/abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)**

:   提出首个无需样例图像即可同时计数图像中多类未知物体的方法ABC123，通过ViT回归多通道密度图+匈牙利匹配训练+SAM示例发现机制，在自建合成数据集MCAC上大幅超越需要样例的方法，且能泛化到FSC-147真实数据集。

**[Active Generation for Image Classification](others/active_generation_for_image_classification.md)**

:   本文提出ActGen，将主动学习思想融入扩散模型的图像生成过程，通过识别模型误分类的验证样本作为引导图像、结合注意力引导和基于梯度的生成控制，仅用10%的合成图像即可在ImageNet上实现+2.26%的准确率提升，超过了使用94%合成数据的先前方法。

**[AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](others/addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)**

:   本文提出 AddMe，一个基于扩散模型的零样本人像生成器，通过身份解耦适配器和增强型人像注意力模块，能够将给定的人像自然地插入到现有场景图像的指定位置，同时保持身份一致性和群体交互的合理性。

**[ADMap: Anti-disturbance Framework for Vectorized HD Map Construction](others/admap_anti-disturbance_framework_for_vectorized_hd_map_construction.md)**

:   本文提出 ADMap 框架，通过多尺度感知颈部(MPN)、实例交互注意力(IIA)和矢量方向差异损失(VDDL)三个模块，从实例间和实例内两个层面级联式监控点序列预测过程，有效缓解了矢量化高精地图构建中的点序列抖动/锯齿问题，在 nuScenes 和 Argoverse2 上取得了 SOTA 性能。

**[Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](others/align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)**

:   提出NEAT——一种模型无关的轻量级插件，通过重要性引导的查询提议、可变形特征对齐和区域交叉注意力增强三个模块，显式解决协同感知中因位姿误差和通信延迟导致的特征级空间错位问题，在四个协同3D检测数据集的噪声设置下为多种基线方法带来一致性增益。

**[An Incremental Unified Framework for Small Defect Inspection](others/an_incremental_unified_framework_for_small_defect_inspection.md)**

:   提出增量统一框架IUF，首次将增量学习集成到统一重建式缺陷检测方法中，通过目标感知自注意力（OASA）建立语义边界、语义压缩损失（SCL）压缩非主要语义空间、以及基于SVD的权重更新策略保护旧对象特征，在MVTec-AD和VisA上实现图像级和像素级的SOTA增量缺陷检测性能。

**[AttnZero: Efficient Attention Discovery for Vision Transformers](others/attnzero_efficient_attention_discovery_for_vision_transformers.md)**

:   本文提出 AttnZero，首个自动发现高效注意力模块的框架，通过构建包含六类计算图和丰富算子的搜索空间、利用进化算法进行多目标搜索，自动发现了适用于多种 ViT 的线性注意力公式，在 DeiT/PVT/Swin/CSwin 上分别达到 74.9%/78.1%/82.1%/82.9% 的 ImageNet top-1 准确率，并构建了包含 2000 种注意力变体的 Attn-Bench-101 基准。

**[Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](others/auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)**

:   本文提出 Auto-GAS，首个面向生成模型（GAN）的免训练架构搜索框架，通过自动发现并优化零成本代理指标来替代传统训练式搜索，实现 110 倍搜索加速，同时保持与训练式方法相当的生成质量。

**[Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)**

:   提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。

**[Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](others/brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)**

:   本文提出了一种从功能磁共振成像（fMRI）信号重建视频的新方法，通过多数据集多被试训练和三阶段pipeline，利用预训练的文本到视频和视频到视频模型，实现了跨数据集和跨被试的SOTA视频重建能力。

**[CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](others/clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)**

:   本文提出了CLR-GAN训练范式，通过让判别器恢复生成器的预定义隐码、让生成器重建真实输入，建立了G和D隐空间之间的一致性约束，使GAN训练更公平稳定，在CIFAR10上FID提升31.22%，在AFHQ-Cat上提升39.5%。

**[COIN-Matting: Confounder Intervention for Image Matting](others/coin-matting_confounder_intervention_for_image_matting.md)**

:   本文从因果推断角度分析图像抠图任务中的数据集偏差问题，识别出对比度偏差和透明度偏差两种典型偏差及其根源——混淆因子，并通过后门调整提出模型无关的 COIN 抠图框架，显著缓解偏差影响、提升现有抠图模型性能。

**[DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation](others/dc-solver_improving_predictor-corrector_diffusion_sampler_via_dynamic_compensati.md)**

:   提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。

**[Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](others/decoupling_common_and_unique_representations_for_multimodal_self-supervised_lear.md)**

:   提出 DeCUR，在多模态自监督学习中将嵌入维度显式拆分为跨模态共有 (common) 和模态独有 (unique) 两部分，通过互相关矩阵分别驱动对齐与去相关，同时引入模态内训练保证独有维度学到有意义信息，在 SAR-光学、RGB-DEM、RGB-Depth 三类多模态场景上均优于 Barlow Twins / CLIP 等基线。

**[Docling Technical Report](others/docling_pdf_document_conversion.md)**

:   Docling 是一个开源的 PDF 文档转换工具，集成了基于 DocLayNet 的布局分析模型和 TableFormer 表格结构识别模型，可在普通硬件上高效地将 PDF 转换为结构化的 JSON 或 Markdown 格式。

**[Domain Reduction Strategy for Non-Line-of-Sight Imaging](others/domain_reduction_strategy_for_non-line-of-sight_imaging.md)**

:   提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。

**[Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](others/dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)**

:   本文提出 DMLoRA（Dropout-Mixture Low-Rank Adaptation），通过引入多分支上下投影结构并在训练过程中逐步dropout分支来平衡精度与正则化，配合两阶段学习缩放因子策略优化每层的缩放系数，在VTAB-1k和FGVC视觉微调基准上取得SOTA性能且推理无额外开销。

**[Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese Handwriting](others/elegantly_written_disentangling_writer_and_character_styles_for_enhancing_online.md)**

:   本文提出了一种基于序列模型的在线中文手写轨迹美化方法，通过交叉注意力机制解耦书写者风格和字符结构风格，将用户潦草的手写轨迹转化为保持个人风格的美观书写，同时通过笛卡尔积分解有效去除冗余风格特征。

**[Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](others/enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)**

:   提出Diode优化器，专为二值神经网络（BNN）设计，通过利用梯度符号的低阶矩估计实现无潜在权重（latent-weight-free）的参数更新，在ImageNet上将BNext-18的Top-1准确率提升0.96%且训练迭代次数减少8倍，并在NLP任务上达到新SOTA。

**[ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](others/et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)**

:   提出首个从真实电影中提取的**相机-角色轨迹数据集 E.T.**（115K 样本，11M 帧），以及基于扩散模型的 **Director** 方法，能根据文本描述和角色轨迹生成复杂的相机运动轨迹，同时设计了 **CLaTr** 对比嵌入用于轨迹生成质量评估。

**[Event-based Mosaicing Bundle Adjustment](others/event-based_mosaicing_bundle_adjustment.md)**

:   提出 EMBA，首个针对纯旋转事件相机的光度 Bundle Adjustment 方法，利用线性化事件生成模型将问题形式化为正则化非线性最小二乘优化，并利用法方程矩阵的块对角稀疏结构设计高效求解器，同时优化相机旋转轨迹和全景梯度图。

**[Exploring Guided Sampling of Conditional GANs](others/exploring_guided_sampling_of_conditional_gans.md)**

:   本文提出在条件GAN中引入类似扩散模型的引导采样（guided sampling）策略，通过隐空间向量运算估计数据-条件联合分布，无需预训练分类器或学习无条件模型，即可显著提升GAN生成质量，将ImageNet 64×64上的FID从8.87降至4.37。

**[FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](others/fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)**

:   本文提出FisherRF，利用Fisher信息直接量化辐射场（Radiance Fields）模型参数的观测信息量，通过最大化期望信息增益（Expected Information Gain）选择最优视角，在视角选择、主动建图和不确定性量化三个任务上均达到SOTA，且通过稀疏性利用和自定义CUDA核实现了70 fps的视角评估速度。

**[Foster Adaptivity and Balance in Learning with Noisy Labels](others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)**

:   提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。

**[Free-Viewpoint Video of Outdoor Sports Using a Flying Camera](others/free-viewpoint_video_of_outdoor_sports_using_a_flying_camera.md)**

:   提出了一种基于无人机RGB相机的系统，能够重建户外运动场景中的4D动态人体和3D无界背景，实现任意时刻的自由视点视频渲染。

**[FreeAugment: Data Augmentation Search Across All Degrees of Freedom](others/freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)**

:   提出 FreeAugment，首个能够同时全局优化数据增强策略的四个自由度（变换数量/类型/顺序/强度）的全可微搜索方法，通过 Gumbel-Softmax 学习深度分布、Gumbel-Sinkhorn 学习排列分布来避免重复采样，在多个基准上取得 SOTA。

**[Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery](others/functional_transform-based_low-rank_tensor_factorization_for_multi-dimensional_d.md)**

:   提出了基于函数变换的低秩张量分解方法（FLRTF），利用隐式神经表示替代传统离散变换来捕获数据在第三维度上的连续平滑性，有效解决时间/光谱退化问题。

**[Gaze Target Detection Based on Head-Local-Global Coordination](others/gaze_target_detection_based_on_head-local-global_coordination.md)**

:   提出了一种基于头部-局部-全局三视图协调的注视目标检测方法，通过引入基于FOV（视野范围）的局部视图，并设计全局-局部位置与表示一致性机制，显著提升了注视目标预测的准确性。

**[GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths](others/gazexplain_learning_to_predict_natural_language_explanations_of_visual_scanpaths.md)**

:   提出GazeXplain，首次将视觉扫描路径预测与自然语言解释结合，通过注意力-语言解码器、语义对齐机制和跨数据集联合训练，实现对人类注视行为的可解释预测。

**[HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images](others/hiei_a_universal_framework_for_generating_high-quality_emerging_images_from_natu.md)**

:   本文提出了一个通用框架 HiEI，通过人类中心的颜色量化模块（TTNet）、感知难度控制模块（PDC）和模板矢量化模块（TV），将自然图像转化为高质量的新兴图像（Emerging Images），在内容和风格质量上超越现有方法，同时可有效对抗深度视觉模型的攻击，适用于 CAPTCHA 机制。

**[High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding](others/high-fidelity_3d_textured_shapes_generation_by_sparse_encoding_and_adversarial_d.md)**

:   本文提出了一种基于稀疏编码模块和对抗解码模块的 3D 纹理形状生成框架，通过对 StableDiffusion 的最小适配扩展到 3D 领域，在 ShapeNet 和 G-Objaverse（200K 样本）上实现了开放词汇的高保真 3D 生成，超越了现有 SOTA 方法。

**[HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](others/hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)**

:   提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。

**[Image Demoiréing in RAW and sRGB Domains](others/image_demoiréing_in_raw_and_srgb_domains.md)**

:   提出RRID框架联合利用RAW和sRGB双域数据进行图像去摩尔纹，设计了带GFM（门控反馈）和FSM（频域选择）的SCDM去摩尔纹模块，以及RGISP实现设备相关ISP学习辅助颜色恢复，在PSNR上超越SOTA 0.62dB。

**[Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](others/improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)**

:   提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。

**[Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](others/interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)**

:   提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。

**[Intrinsic Single-Image HDR Reconstruction](others/intrinsic_single-image_hdr_reconstruction.md)**

:   > 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。

**[Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](others/learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)**

:   本文提出了一种基于"正常性先验"的无监督视频异常检测方法（LANP），通过利用"视频首尾段大概率为正常事件"这一数据无关先验知识生成初始正常标签，再通过正常性传播将正常知识扩散到全部片段，最后配合损失重加权策略训练异常检测器，在 ShanghaiTech 和 UCF-Crime 上取得了优异性能。

**[Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)**

:   提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。

**[MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)**

:   本文提出 MemBN（Memory-based Batch Normalization），通过在每个 BN 层中维护统计量记忆队列并设计专用的记忆管理与聚合算法，使得 TTA 方法在各种批量大小下都能稳健估计测试域的统计量，大幅提升小批量场景下的准确率和鲁棒性。

**[Momentum Auxiliary Network for Supervised Local Learning](others/momentum_auxiliary_network_for_supervised_local_learning.md)**

:   本文提出动量辅助网络（MAN），通过指数移动平均（EMA）将相邻局部块的参数信息传递到当前块的辅助网络，并引入可学习偏置弥补跨块特征差异，解决了监督局部学习中块间信息交换缺失导致的"短视"问题，在 ImageNet 上以不到 E2E 训练一半的 GPU 显存实现更高性能。

**[Non-parametric Sensor Noise Modeling and Synthesis](others/non-parametric_sensor_noise_modeling_and_synthesis.md)**

:   本文提出一种非参数传感器噪声模型，通过直接从实拍图像中为每个亮度级别构建概率质量函数(PMF)来建模真实噪声分布，无需假设特定噪声分布形式，并提出了ISO插值和在含噪图像上合成噪声的方法，在下游去噪任务上显著优于现有参数化噪声模型。

**[Object-Aware NIR-to-Visible Translation](others/object-aware_nir-to-visible_translation.md)**

:   本文提出一种对象感知的近红外(NIR)到可见光图像翻译框架，通过将可见光图像分解为与对象无关的光照分量和对象特定的反射分量分别处理，结合分割先验知识，在缺乏大规模配对数据的条件下实现了高质量的NIR彩色化，并构建了首个完全对齐的NIR-可见光大规模配对数据集。

**[On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](others/on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)**

:   从数学上系统分析3D Gaussian Splatting中局部仿射近似引入的投影误差，证明误差函数在Gaussian均值方向与投影平面法线重合时取极小值，据此提出每个Gaussian投影到各自切平面的最优投影策略(Optimal Gaussian Splatting)，在不影响实时性能的前提下显著降低渲染伪影。

**[Online Temporal Action Localization with Memory-Augmented Transformer](others/online_temporal_action_localization_with_memory-augmented_transformer.md)**

:   本文提出 MATR（Memory-Augmented Transformer），通过记忆队列选择性地保存历史片段特征来建模长期上下文，并采用双 Transformer 解码器分别定位动作的结束和起始时间，在 THUMOS14 和 MUSES 两个在线时序动作定位基准上刷新了 SOTA，甚至可与部分离线方法媲美。

**[Operational Open-Set Recognition and PostMax Refinement](others/operational_open-set_recognition_and_postmax_refinement.md)**

:   本文提出了一种面向实际部署场景的开放集识别评估指标 OOSA（Operational Open-Set Accuracy）以及后处理算法 PostMax，通过对最大类别 logit 进行深度特征幅度归一化和广义 Pareto 分布映射，将 logit 转化为合理的概率估计，在大规模评估中取得了统计显著的 SOTA 性能。

**[PartCraft: Crafting Creative Objects by Parts](others/partcraft_crafting_creative_objects_by_parts.md)**

:   提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。

**[PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](others/pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)**

:   提出PISR方法，利用偏振光的几何约束（偏振角与法线方位角的对应关系）直接正则化神经隐式表面形状，结合哈希网格加速和图像空间法线平滑，在无纹理和镜面物体上实现了0.5mm Chamfer距离和99.5% F-score的高精度重建，速度比此前偏振方法快4~30倍。

**[Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](others/power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)**

:   提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。

**[Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](others/raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)**

:   提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。

**[Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes](others/real-data-driven_2000_fps_color_video_from_mosaicked_chromatic_spikes.md)**

:   针对马赛克彩色脉冲相机（mosaicked chromatic spikes），提出一种完全基于真实数据驱动的 2000FPS 彩色高动态范围视频重建方法，通过自监督去噪模块和渐进式配准模块解决短时帧噪声和运动模糊问题，无需合成数据即可重建高质量高速彩色视频。

**[Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](others/rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)**

:   本文提出 RECD 算法，通过蒙特卡洛近似估计未标注数据的未知类别分布，基于估计分布重新平衡分类器，并引入特征聚类压缩缓解特征图不平衡，在标注-未标注数据类别分布失配的半监督学习场景中取得 SOTA 性能。

**[Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](others/rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)**

:   本文提出"Undercover Bias"数据集水印方法，通过在训练数据中嵌入与目标任务无关但与标签对应的隐蔽水印图案，使未授权使用者训练的模型不自觉地学会分类这些水印，水印分类能力作为未授权使用的不可抵赖证据，实现了隐蔽、模型无关、对目标任务无损的数据集版权保护。

**[Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](others/self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。

**[Shifted Autoencoders for Point Annotation Restoration in Object Counting](others/shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)**

:   提出**Shifted AutoEncoders (SAE)**，一种受MAE启发的点标注修复方法：通过随机位移点标注后训练UNet恢复，使模型学到"通用位置知识"而忽略个体标注噪声；用训练好的SAE修复原始标注使其更一致，可为任意计数模型（密度图/定位型）稳定提升性能，在9个数据集上创下新记录。

**[SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding](others/spatialformer_towards_generalizable_vision_transformers_with_explicit_spatial_un.md)**

:   提出SpatialFormer架构，通过引入自适应空间token显式建模场景的全局空间关系，采用decoder-only架构与双边交叉注意力块实现上下文与空间信息的高效交互，在分类、分割和检测任务上展示了优异的泛化性和可迁移性。

**[Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](others/spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)**

:   提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。

**[SpectraM-PS: Spectrally Multiplexed Photometric Stereo Under Unknown Spectral Composition](others/spectram-ps_spectrally_multiplexed_photometric_stereo_under_unknown_spectral_com.md)**

:   提出一种无需物理模型约束的光谱复用光度立体方法（SpectraM-PS），在光源光谱组成完全未知的条件下，通过数据驱动的方式从单张RGB图像中恢复表面法线，实现了传统多次拍摄光度立体到单次拍摄的突破。

**[STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](others/stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)**

:   提出空间-时间子空间投影（STSP）方法解决视频类增量学习中的灾难性遗忘问题，通过时间子空间分类器（TSC）用正交子空间基表示每个类别，并通过空间梯度投影（SGP）将梯度约束在旧任务特征的零空间中，在HMDB51、UCF101和SSv2上达到SOTA。

**[Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)**

:   提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。

**[Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](others/synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)**

:   提出了 IntCLIP 框架，通过双分支编码策略将 CLIP 中的"视觉感知"（Sight）知识迁移到"语义中心"（Semantic）的多标签意图理解任务中，结合层次化类别整合和视觉辅助聚合，在标准 MIU benchmark 和图像情感识别任务上显著超越 SOTA。

**[Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](others/teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)**

:   提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。

**[Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing](others/wavelength-embedding-guided_filter-array_transformer_for_spectral_demosaicing.md)**

:   本文提出 WeFAT，通过波长嵌入引导的多头自注意力（We-MSA）赋予模型"波长记忆"能力，配合滤波器阵列注意力机制（MaM）聚焦高质量光谱区域，仅在 ARAD 数据集上训练就能在不同相机和不同光谱分布下保持稳定性能，超越现有 SOTA。

</div>