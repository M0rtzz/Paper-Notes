---
title: >-
  ECCV2024 779篇论文解读
description: >-
  779篇ECCV2024论文深度解读，每篇5分钟读懂核心思想。覆盖3D视觉、图像生成、多模态VLM、语义分割、自动驾驶、人体理解等36个研究领域，每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎞️ ECCV2024 论文笔记

共 **779** 篇笔记，覆盖 **36** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 🧊 [3D视觉](#3d_vision) | 156 |
| 🎨 [图像生成](#image_generation) | 121 |
| 🧩 [多模态VLM](#multimodal_vlm) | 76 |
| ✂️ [语义分割](#segmentation) | 53 |
| 🚗 [自动驾驶](#autonomous_driving) | 52 |
| 🧑 [人体理解](#human_understanding) | 52 |
| 🎬 [视频理解](#video_understanding) | 42 |
| 🎯 [目标检测](#object_detection) | 37 |
| 🏥 [医学图像](#medical_imaging) | 23 |
| 🖼️ [图像恢复](#image_restoration) | 20 |
| 📦 [模型压缩](#model_compression) | 15 |
| 📊 [LLM评测](#llm_evaluation) | 12 |
| 🤖 [机器人/具身智能](#robotics) | 10 |
| 🛡️ [AI安全](#ai_safety) | 9 |
| 🎵 [音频/语音](#audio_speech) | 9 |
| 🔄 [自监督/表示学习](#self_supervised) | 9 |
| 💬 [LLM/NLP](#llm_nlp) | 7 |
| 🔍 [信息检索/RAG](#information_retrieval) | 6 |
| 📚 [预训练/数据](#llm_pretraining) | 6 |
| 🛰️ [遥感](#remote_sensing) | 5 |
| 🕸️ [图学习](#graph_learning) | 4 |
| 🔬 [可解释性](#interpretability) | 4 |
| 🎮 [强化学习](#reinforcement_learning) | 4 |
| 📡 [信号/通信](#signal_comm) | 4 |
| 🦾 [LLM Agent](#llm_agent) | 3 |
| 💡 [LLM推理](#llm_reasoning) | 3 |
| 🗣️ [对话系统](#dialogue) | 2 |
| 👥 [社会计算](#social_computing) | 2 |
| 📈 [时间序列](#time_series) | 2 |
| 🔗 [因果推理](#causal_inference) | 1 |
| 🌍 [地球科学](#earth_science) | 1 |
| 🔒 [LLM安全](#llm_safety) | 1 |
| 📐 [优化/理论](#optimization) | 1 |
| ⚛️ [物理学](#physics) | 1 |
| 🎁 [推荐系统](#recommender) | 1 |
| 📂 [其他](#others) | 25 |

---

## 🧊 3D视觉 { #3d_vision }

**[3D Congealing 3D-Aware Image Alignment In The Wild](3d_vision/3d_congealing_3d-aware_image_alignment_in_the_wild.md)**

:   3D Congealing将一组语义相似的无标注互联网图像对齐到共享的3D canonical空间，通过结合预训练扩散模型的SDS指导获得3D形状 + DINO语义特征匹配估计位姿和坐标映射，无需模板、位姿标注或相机参数。

**[3D Reconstruction Of Objects In Hands Without Real World 3D](3d_vision/3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)**

:   提出HORSE框架，通过从野外视频中提取多视角2D mask监督（以手部姿态作为物体姿态代理）和从合成3D形状集合中学习2D切片对抗形状先验，训练occupancy网络从单张RGB图像重建手持物体3D形状，在不使用任何真实世界3D标注的情况下，在MOW数据集上超越使用3D监督的方法11.6%。

**[3D Single-Object Tracking In Point Clouds With High Temporal Variation](3d_vision/3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)**

:   HVTrack首次探索高时间变化场景下的3D单目标跟踪，通过相对位姿感知记忆模块(RPM)、基础-扩展特征交叉注意力(BEA)和上下文点引导自注意力(CPA)三个模块，分别解决点云形状剧变、相似物体干扰和背景噪声问题，在KITTI-HV 5帧间隔下比SOTA提升11.3%/15.7% Success/Precision。

**[3Dego 3D Editing On The Go](3d_vision/3dego_3d_editing_on_the_go.md)**

:   3DEgo将传统三阶段3D编辑流程（COLMAP位姿估计→未编辑场景初始化→迭代编辑更新）压缩为单阶段框架：先用自回归噪声混合模块对视频帧进行多视角一致的2D编辑，再用COLMAP-free的3DGS从编辑后帧直接重建3D场景，速度提升约10倍且支持任意来源视频。

**[3Igs Factorised Tensorial Illumination For 3D Gaussian Splatting](3d_vision/3igs_factorised_tensorial_illumination_for_3d_gaussian_splatting.md)**

:   3iGS 用基于张量分解的连续入射光照场替代 3DGS 中每个高斯体独立优化的球谐系数，结合可学习 BRDF 特征和轻量神经渲染器来建模出射辐射，在保持实时渲染速度的同时显著提升了镜面反射等视角依赖效果的渲染质量。

**[3X2 3D Object Part Segmentation By 2D Semantic Correspondenc](3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)**

:   提出了一种无需训练的3D物体部件分割方法3-By-2，利用扩散模型(DIFT)的2D语义对应关系从已标注2D数据集或少量3D标注对象中迁移部件标签到3D，在zero-shot和few-shot设置下均达到SOTA。

**[6Dgs 6D Pose Estimation From A Single Image And A 3D Gaussia](3d_vision/6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)**

:   提出6DGS，通过反转3DGS渲染流程——从椭球体表面均匀发射光线（Ellicell），利用注意力机制将光线与目标图像像素绑定，再用加权最小二乘闭式求解相机位姿，无需迭代和初始位姿，在真实场景上旋转精度提升12%、平移精度提升22%，达到15fps近实时性能。

**[A Compact Dynamic 3D Gaussian Representation For Realtime Dy](3d_vision/a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)**

:   将3DGS中的位置和旋转参数建模为时间的函数（位置用Fourier逼近、旋转用线性逼近），使动态场景的存储复杂度从O(TN)降低到O(LN)，在D-NeRF/DyNeRF/HyperNeRF三个数据集上实现了与NeRF方法匹敌的渲染质量，同时保持118+ FPS的实时渲染速度。

**[Analytic-Splatting Anti-Aliased 3D Gaussian Splatting Via Analytic Integration](3d_vision/analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)**

:   通过使用条件 logistic 函数解析近似高斯信号在像素窗口上的积分，替代 3DGS 的像素中心点采样，实现无混叠的 3D 高斯泼溅，在多尺度渲染上超越 Mip-Splatting。

**[Animatabledreamer Text-Guided Non-Rigid 3D Model Generation And Reconstruction W](3d_vision/animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)**

:   提出 AnimatableDreamer，通过 Canonical Score Distillation (CSD) 技术，从单目视频提取骨骼和运动后生成文本引导的可动画化 3D 非刚体模型，在生成质量和时序一致性上全面超越现有方法。

**[Bad-Gaussians Bundle Adjusted Deblur Gaussian Splatting](3d_vision/bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)**

:   首次将运动模糊物理成像模型引入 3D Gaussian Splatting 框架，联合优化场景 Gaussian 参数与曝光时间内的相机运动轨迹，从模糊图像中恢复清晰 3D 场景并实现实时渲染。

**[Benerf Neural Radiance Fields From A Single Blurry Image And Event Stream](3d_vision/benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)**

:   提出 BeNeRF，仅从**单张模糊图像**及其对应的事件流（event stream）联合恢复神经辐射场与相机运动轨迹，无需多视角输入或已知位姿，即可实现高质量去模糊与新视角合成。

**[Bi-Directional Contextual Attention For 3D Dense Captioning](3d_vision/bi-directional_contextual_attention_for_3d_dense_captioning.md)**

:   提出 BiCA，通过双向上下文注意力机制将 instance query 和 context query 解耦并行解码，解决了 3D 密集描述中定位与描述生成之间的目标冲突，在 ScanRefer 和 Nr3D 两个基准上取得 SOTA。

**[Binomial Self-Compensation For Motion Error In Dynamic 3D Scanning](3d_vision/binomial_self-compensation_for_motion_error_in_dynamic_3d_scanning.md)**

:   提出二项式自补偿(BSC)算法,通过对运动受影响的相位序列按二项式系数加权求和,无需任何中间变量即可指数级消除四步相位移轮廓术中的运动误差,实现与相机帧率相同的高精度动态3D扫描。

**[Caesarnerf Calibrated Semantic Representation For Few-Shot Generalizable Neural ](3d_vision/caesarnerf_calibrated_semantic_representation_for_few-shot_generalizable_neural_.md)**

:   提出 CaesarNeRF，在可泛化 NeRF（GNT）基础上引入场景级语义表征，通过相机位姿校准（特征旋转对齐到目标视角）和序列细化（跨 Transformer 层逐步更新全局特征），在 1-view 设置下 PSNR 比 GNT 提升 1.74dB（LLFF），且可即插即用地增强 IBRNet、MatchNeRF 等其他基线。

**[Camera Height Doesnapost Change Unsupervised Training For Metric Monocular Road-](3d_vision/camera_height_doesnapost_change_unsupervised_training_for_metric_monocular_road-.md)**

:   提出FUMET训练框架,利用道路上检测到的车辆尺寸先验聚合为相机高度估计,并利用相机高度在同一视频序列中不变的事实作为度量尺度监督,使任意单目深度网络无需辅助传感器即可学习绝对尺度。

**[Canonicalfusion Generating Drivable 3D Human Avatars From Multiple Images](3d_vision/canonicalfusion_generating_drivable_3d_human_avatars_from_multiple_images.md)**

:   提出CanonicalFusion框架,通过联合预测深度图和压缩LBS权重映射图实现直接规范化,并利用前向蒙皮可微渲染融合多张图像信息,从多张输入图像生成可驱动的3D人体Avatar。

**[Cg-Slam Efficient Dense Rgb-D Slam In A Consistent Uncertainty-Aware 3D Gaussian](3d_vision/cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)**

:   提出CG-SLAM,基于一致性和几何稳定性优化的不确定性感知3D高斯场,实现高效稠密RGB-D SLAM,在定位精度和建图质量上均达到SOTA,跟踪速度最高15Hz。

**[Citygaussian Real-Time High-Quality Large-Scale Scene Rendering With Gaussians](3d_vision/citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)**

:   提出 CityGaussian (CityGS)，通过分治训练策略和 block-wise Level-of-Detail 机制，首次实现了城市级大规模场景（>1.5 km²）的高质量 3D Gaussian Splatting 训练与跨尺度实时渲染。

**[Click-Gaussian Interactive Segmentation To Any 3D Gaussians](3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)**

:   提出Click-Gaussian，通过学习两级粒度（粗/细）的可区分3D特征场，结合全局特征引导学习(GFL)解决跨视角mask不一致问题，实现每次点击仅需10ms的实时3D高斯交互式分割，速度比现有方法快15-130倍，同时显著提升分割精度。

**[Coherentgs Sparse Novel View Synthesis With Coherent 3D Gaussians](3d_vision/coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)**

:   提出CoherentGS，通过为3DGS引入结构化表示（每像素一个高斯）并利用隐式卷积解码器和全变差损失构建单视图和多视图一致性约束，结合基于单目深度的初始化策略，在极稀疏输入（如3张图像）下实现高质量新视角合成，LPIPS指标显著优于现有NeRF方法。

**[Comboverse Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guid](3d_vision/comboverse_compositional_3d_assets_creation_using_spatially-aware_diffusion_guid.md)**

:   提出ComboVerse，一个组合式3D资产生成框架：先将包含多个物体的输入图像分解并独立重建为单物体3D模型，再通过空间感知的Score Distillation Sampling (SSDS)引导物体的位置、缩放和旋转参数优化，实现高质量多物体组合3D资产创建，在CLIP Score和人类评估中均显著优于现有方法。

**[Compress3D A Compressed Latent Space For 3D Generation From A Single Image](3d_vision/compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)**

:   提出一种高度压缩的 triplane 潜空间自编码器，配合两阶段扩散模型（先生成 shape embedding 再生成 triplane latent），仅需 7 秒即可从单张图像生成高质量 3D 资产，且训练数据和时间远少于同类方法。

**[Cor-Gs Sparse-View 3D Gaussian Splatting Via Co-Regularization](3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)**

:   发现同时训练两个 3DGS 辐射场时它们在高斯位置和渲染结果上的差异（disagreement）与重建质量负相关，据此提出 CoR-GS 通过协同剪枝和伪视角协同正则化来抑制不准确重建，在稀疏视角下实现 SOTA 新视角合成。

**[Crm Single Image To 3D Textured Mesh With Convolutional Reconstruction Model](3d_vision/crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)**

:   提出卷积重建模型 CRM，利用 triplane 与六个正交视图之间的空间对齐先验，用 U-Net 替代 Transformer 直接从六视图映射到 triplane，结合 FlexiCubes 端到端训练，10 秒内从单张图像生成高保真纹理网格，训练成本仅为 LRM 的 1/8。

**[Crossscore Towards Multi-View Image Evaluation And Scoring](3d_vision/crossscore_towards_multi-view_image_evaluation_and_scoring.md)**

:   提出 Cross-Reference（CR）图像质量评估新范式，通过对比查询图像与多个不同视角参考图像，利用 cross-attention 神经网络预测与 SSIM 高度相关的像素级质量分数，无需 ground truth 参考图像即可评估新视角合成质量。

**[Crossscore Towards Multiview Image Evaluation And Scori](3d_vision/crossscore_towards_multiview_image_evaluation_and_scori.md)**

:   提出 CrossScore——一种新型的交叉参考图像质量评估方法，利用多视角参考图像替代真实参考图，通过 cross-attention 机制预测 SSIM 分数图，在无需 ground truth 的条件下实现接近全参考指标的评估精度。

**[D-Sco Dual-Stream Conditional Diffusion For Monocular Hand-Held Object Reconstru](3d_vision/d-sco_dual-stream_conditional_diffusion_for_monocular_hand-held_object_reconstru.md)**

:   提出双流条件扩散模型 D-SCo 从单张 RGB 图像重建手持物体点云，通过统一手-物语义嵌入和手关节几何嵌入两个分支分别提供语义和几何先验，配合手约束质心固定策略稳定扩散过程，在 ObMan 上 F-5 达 0.61（超 DDF-HO 10.9%），真实数据集 HO3D/MOW 上也大幅领先。

**[Datenerf Depth-Aware Text-Based Editing Of Nerfs](3d_vision/datenerf_depth-aware_text-based_editing_of_nerfs.md)**

:   利用NeRF重建的场景深度信息来引导基于文本的2D图像编辑（通过深度条件化的ControlNet + 投影修复方案），从而实现多视角一致的高质量NeRF场景编辑。

**[Deblur E-Nerf Nerf From Motion-Blurred Events Under High-Speed Or Low-Light Cond](3d_vision/deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)**

:   提出 Deblur e-NeRF，通过物理精确的像素带宽模型来建模事件相机的运动模糊，首次实现从运动模糊的事件流中直接有效地重建无模糊 NeRF。

**[Deceptive-Nerf3Dgs Diffusion-Generated Pseudo-Observations For High-Quality Spar](3d_vision/deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)**

:   利用微调的 Stable Diffusion + ControlNet 将粗糙 NeRF/3DGS 渲染结果转化为高质量伪观测图像，将稀疏输入视图增密 5-10 倍后重新训练，在 Hypersim/LLFF/ScanNet 等数据集上超越 FreeNeRF 等方法 1-2dB PSNR，训练速度比扩散正则化方法快约 10 倍。

**[Deep Patch Visual Slam](3d_vision/deep_patch_visual_slam.md)**

:   基于 DPVO 视觉里程计系统，通过高效的邻近回环检测和经典回环检测机制，将其扩展为完整的 SLAM 系统 DPV-SLAM，在单 GPU 上实现实时、高精度、低显存的单目视觉 SLAM。

**[Dg-Pic Domain Generalized Point-In-Context Learning For Point Cloud Understandin](3d_vision/dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)**

:   提出 DG-PIC，首个在统一模型中同时处理多领域多任务的点云理解框架，通过双层次源域原型估计和测试时特征平移机制，在不更新模型的情况下提升对未知域的泛化能力。

**[Dgpic Domain Generalized Pointincontext Learning For Po](3d_vision/dgpic_domain_generalized_pointincontext_learning_for_po.md)**

:   提出 DG-PIC，首个在统一模型中同时处理多域多任务点云理解的方法，通过双层源域原型估计和双层测试时特征平移机制，在无需模型更新的情况下提升对未见域的泛化能力。

**[Differentiable Convex Polyhedra Optimization From Multi-View Images](3d_vision/differentiable_convex_polyhedra_optimization_from_multi-view_images.md)**

:   提出一种基于对偶变换和三平面交点求解的可微凸多面体构造方法，绕过隐式场监督，直接利用多视角图像损失进行梯度优化，实现高保真的凸多面体形状表示。

**[Diffusion Model Is A Good Pose Estimator From 3D Rf-Vision](3d_vision/diffusion_model_is_a_good_pose_estimator_from_3d_rf-vision.md)**

:   提出mmDiff，一种基于扩散模型的毫米波雷达人体姿态估计框架，通过全局-局部雷达上下文提取和结构-运动一致性约束，有效应对雷达点云稀疏、噪声大和信号不一致的挑战，显著超越现有SOTA。

**[Diffusion Models For Monocular Depth Estimation Overcoming Challenging Condition](3d_vision/diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)**

:   利用text-to-image扩散模型（ControlNet/T2I-Adapter）将简单场景图像转化为保持同一3D结构的恶劣条件图像，通过自蒸馏微调现有单目深度估计网络，统一解决恶劣天气和非朗伯表面等分布外挑战。

**[Divide And Fuse Body Part Mesh Recovery From Partially Visible Human Images](3d_vision/divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)**

:   提出"分而治之"的自底向上人体网格重建方法，通过独立重建各身体部位后融合，有效解决人体大面积不可见时传统自顶向下方法（如SMPL）失效的问题。

**[Dreamdissector Learning Disentangled Text-To-3D Generation From 2D Diffusion Pri](3d_vision/dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)**

:   提出DreamDissector框架，通过Neural Category Field和Deep Concept Mining将包含多物体交互的text-to-3D NeRF解耦为独立的带纹理网格，实现物体级别的3D编辑控制。

**[Dreamdrone Text-To-Image Diffusion Models Are Zero-Shot Perpetual View Generator](3d_vision/dreamdrone_text-to-image_diffusion_models_are_zero-shot_perpetual_view_generator.md)**

:   提出DreamDrone——一个零样本、无需训练的无限飞行场景生成管线，通过直接对预训练扩散模型的中间latent code进行warping（而非图像级warping），结合特征对应引导和高通滤波策略，实现高质量、几何一致的无界场景生成。

**[Dreamdrone Texttoimage Diffusion Models Are Zeroshot Perpetu](3d_vision/dreamdrone_texttoimage_diffusion_models_are_zeroshot_perpetu.md)**

:   DreamDrone提出零样本、免训练的无限场景飞越生成pipeline，核心创新是在扩散模型的latent空间进行视角变换（而非像素空间），并通过特征对应引导和高通滤波策略保证帧间的几何一致性和高频细节一致性。

**[Dreamscene360 Unconstrained Text-To-3D Scene Generation With Panoramic Gaussian ](3d_vision/dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)**

:   提出DreamScene360，利用全景图像作为中间表示，结合GPT-4V自精炼机制和全景3D高斯溅射技术，实现从文本到沉浸式360°3D场景的快速生成。

**[Dreamview Injecting View-Specific Text Guidance Into Text-To-3D Generation](3d_vision/dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)**

:   提出DreamView，通过自适应文本引导注入模块，将视角特定的文本描述和全局文本描述协同注入扩散模型，实现可定制化且多视角一致的文本到3D生成。

**[Dreamview Injecting Viewspecific Text Guidance Into Textto3D](3d_vision/dreamview_injecting_viewspecific_text_guidance_into_textto3d.md)**

:   DreamView通过自适应引导注入模块协调全局和视角特定文本实现3D定制化生成。

**[Dspdet3D 3D Small Object Detection With Dynamic Spatial Pruning](3d_vision/dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)**

:   提出动态空间剪枝（DSP）策略，在多级 3D 检测器的解码器中逐级移除已检测到大物体区域的体素特征，使检测器能以高空间分辨率处理场景、大幅提升小目标检测精度（ScanNet 小目标 mAP@0.25 从 27.5% 提升到 44.8%），同时通过剪枝将显存降低为同分辨率方法的 1/5。

**[Dual-Level Adaptive Self-Labeling For Novel Class Discovery In Point Cloud Segme](3d_vision/dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)**

:   提出双层自适应自标注方法，通过半松弛最优传输处理类别不平衡问题，并结合区域级表示增强点级分类器的学习，在点云分割中实现高效的新类发现。

**[Dynamic Neural Radiance Field From Defocused Monocular Video](3d_vision/dynamic_neural_radiance_field_from_defocused_monocular_video.md)**

:   提出 $D^2RF$，首个从散焦单目视频中恢复清晰动态NeRF的方法，通过将景深(DoF)渲染与体积渲染统一，引入分层DoF体积渲染来建模散焦模糊并恢复清晰新视角。

**[Efficient Depth-Guided Urban View Synthesis](3d_vision/efficient_depth-guided_urban_view_synthesis.md)**

:   提出EDUS方法，利用噪声几何先验（单目/双目深度）引导可泛化NeRF，通过前景3D CNN + 背景/天空图像渲染的三部分分解，实现稀疏街景视角下的快速前馈推理和高效逐场景微调。

**[Equi-Gspr Equivariant Se3 Graph Network Model For Sparse Point Cloud Registratio](3d_vision/equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)**

:   提出基于SE(3)等变图神经网络的稀疏点云配准方法Equi-GSPR，通过等变消息传播、低秩特征变换（LRFT）和隐式特征空间相似度匹配，在室内外数据集上以低模型复杂度实现SOTA配准性能。

**[Explicitly Guided Information Interaction Network For Cross-Modal Point Cloud Co](3d_vision/explicitly_guided_information_interaction_network_for_cross-modal_point_cloud_co.md)**

:   提出EGIInet框架，通过统一编码器实现模态对齐，并利用显式引导的信息交互策略（FT-Loss）让网络精准识别图像中的关键结构信息，在视图引导点云补全任务上以更少参数实现了超越XMFnet 16% CD的性能。

**[External Knowledge Enhanced 3D Scene Generation From Sketch](3d_vision/external_knowledge_enhanced_3d_scene_generation_from_sketch.md)**

:   提出SEK框架，结合手绘草图和外部物体关系知识库作为扩散模型的条件，通过知识增强图推理和频谱滤波器，端到端地同时生成3D室内场景的布局和物体几何形状。

**[Falip Visual Prompt As Foveal Attention Boosts Clip Zer](3d_vision/falip_visual_prompt_as_foveal_attention_boosts_clip_zer.md)**

:   提出 FALIP（Foveal-Attention CLIP），通过在 CLIP 的多头自注意力模块中插入类似人眼中央凹的注意力掩码，在不修改原始图像内容的前提下引导模型关注特定区域，显著提升指代表达理解、图像分类和 3D 点云识别等零样本任务的性能。

**[Falip Visual Prompt As Foveal Attention Boosts Clip Zero-Shot Performance](3d_vision/falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)**

:   提出FALIP（Foveal-Attention CLIP），一种免训练方法，通过在CLIP的多头自注意力模块中插入类似人类中央凹视觉的注意力掩码，在不修改原始图像的情况下增强CLIP的区域感知能力，在指代表达理解、图像分类和3D点云识别等零样本任务上均取得提升。

**[Fastcad Real-Time Cad Retrieval And Alignment From Scans And Videos](3d_vision/fastcad_real-time_cad_retrieval_and_alignment_from_scans_and_videos.md)**

:   提出FastCAD，通过对比学习嵌入空间蒸馏和直接参数预测，实现50ms内完成场景中所有物体的CAD模型检索与对齐，比现有方法快50倍且精度更优。

**[Flash Cache Reducing Bias In Radiance Cache Based Inverse Rendering](3d_vision/flash_cache_reducing_bias_in_radiance_cache_based_inverse_rendering.md)**

:   提出一种无偏的辐射缓存逆渲染方法，通过遮挡感知的vMF重要性采样和快速缓存控制变量技术，在保持计算效率的同时消除现有方法中的渲染偏差，提升材质和光照分解的质量。

**[Flashsplat 2D To 3D Gaussian Splatting Segmentation Solved Optimally](3d_vision/flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)**

:   将3D高斯溅射的2D-to-3D分割问题建模为整数线性规划，利用alpha混合的线性性质得到闭式最优解，仅需30秒完成优化，比现有方法快50倍。

**[Flashtex Fast Relightable Mesh Texturing With Lightcontrolnet](3d_vision/flashtex_fast_relightable_mesh_texturing_with_lightcontrolnet.md)**

:   提出LightControlNet——一种光照感知的ControlNet变体，结合两阶段纹理优化pipeline，能在约4分钟内为3D网格生成高质量、可重光照的PBR纹理，速度比现有方法快3-10倍。

**[Flying With Photons Rendering Novel Views Of Propagating Light](3d_vision/flying_with_photons_rendering_novel_views_of_propagating_light.md)**

:   提出瞬态场（Transient Field）表示，结合首创的多视点超快成像数据集，首次实现从动态新视角渲染真实场景中传播光的视频，能处理散射、反射、折射和衍射等复杂光传输效果。

**[Forest2Seq Revitalizing Order Prior For Sequential Indoor Scene Synthesis](3d_vision/forest2seq_revitalizing_order_prior_for_sequential_indoor_scene_synthesis.md)**

:   提出Forest2Seq框架，通过将无序的室内场景物体组织为层次化的场景树/森林结构，用广度优先遍历导出有意义的排列顺序作为先验知识，配合Transformer自回归解码器显著提升室内场景合成质量。

**[Formula-Supervised Visual-Geometric Pre-Training](3d_vision/formula-supervised_visual-geometric_pre-training.md)**

:   提出FSVGP，利用分形几何的数学公式自动生成对齐的合成图像和点云，通过公式监督一致性标签在统一Transformer上实现跨模态视觉-几何预训练，在图像和3D物体的分类、检测、分割六项任务上均超越单模态FDSL方法。

**[Futuredepth Learning To Predict The Future Improves Video Depth Estimation](3d_vision/futuredepth_learning_to_predict_the_future_improves_video_depth_estimation.md)**

:   提出FutureDepth，通过未来预测网络(F-Net)学习运动线索和重建网络(R-Net)学习多帧对应关系，将隐式的运动和场景特征注入深度解码器，在NYUDv2、KITTI、DDAD、Sintel四个数据集上达到SOTA精度和时序一致性，且推理效率显著优于现有视频深度方法。

**[G3R Gradient Guided Generalizable Reconstruction](3d_vision/g3r_gradient_guided_generalizable_reconstruction.md)**

:   提出G3R，一种梯度引导的可泛化重建方法，通过学习一个重建网络迭代地利用可微渲染的3D梯度反馈更新3D Neural Gaussians表示，在大规模场景（>10,000m²）上实现2分钟内重建，加速至少10倍且达到与3DGS可比或更优的渲染质量。

**[Gaura Generalizable Approach For Unified Restoration And Rendering Of Arbitrary ](3d_vision/gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)**

:   提出GAURA，一种基于可泛化NeRF的统一复原与渲染框架，通过可学习的退化感知latent codes在特征聚合和渲染阶段动态适应不同图像退化类型，无需逐场景优化即可从退化图像中渲染清晰的新视角。

**[Gaussctrl Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](3d_vision/gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)**

:   提出GaussCtrl，利用深度条件化的ControlNet编辑和注意力对齐模块实现多视角一致的文本驱动3DGS场景编辑，支持一次编辑所有视角并仅需一次3D模型更新。

**[Gaussian Grouping Segment And Edit Anything In 3D Scenes](3d_vision/gaussian_grouping_segment_and_edit_anything_in_3d_scenes.md)**

:   为 3D Gaussian Splatting 中的每个高斯学习 16 维 Identity Encoding 实现实例级分组，使用 SAM + DEVA 视频跟踪生成多视图一致的 2D 伪标签做监督，在 LERF-Mask 开放词汇分割上 mIoU 达 69-77%（超 LERF 2 倍+），全景分割超 Panoptic Lifting 4.9% mIoU 且 14× 更快，同时支持 3D 物体移除/修复/着色/风格迁移等多种编辑。

**[Gaussianimage 1000 Fps Image Representation And Compression By 2D Gaussian Splat](3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)**

:   提出GaussianImage，首次将2D Gaussian Splatting用于图像表示与压缩，通过紧凑的8参数2D高斯和累积求和光栅化算法，实现了2000 FPS的解码速度，同时与INR方法在表示质量和压缩性能上持平。

**[Gaussreg Fast 3D Registration With Gaussian Splatting](3d_vision/gaussreg_fast_3d_registration_with_gaussian_splatting.md)**

:   首次探索3D Gaussian Splatting场景之间的配准问题，提出粗到精的GaussReg框架——粗阶段利用点云配准方法估计初始变换，精阶段通过渲染图像提取体积特征进行精细对齐，速度比HLoc快44倍且精度可比。

**[Generative Camera Dolly Extreme Monocular Dynamic Novel View Synthesis](3d_vision/generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)**

:   提出GCD（Generative Camera Dolly），通过微调Stable Video Diffusion模型实现从单目视频生成任意视角的同步动态新视角视频，支持最高180°的极端相机变换，无需深度输入或显式3D建模。

**[Geometrysticker Enabling Ownership Claim Of Recolorized Neural Radiance Fields](3d_vision/geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)**

:   提出GeometrySticker，将二进制版权信息"贴"在NeRF的**几何组件**（而非颜色组件）上，使得即使NeRF被重着色（recolorization），原始创建者仍能从渲染图像中提取水印来主张所有权。

**[Gpsformer A Global Perception And Local Structure Fitting-Based Transformer For ](3d_vision/gpsformer_a_global_perception_and_local_structure_fitting-based_transformer_for_.md)**

:   提出GPSFormer，通过全局感知模块(GPM)学习点云短程和长程依赖，结合Taylor级数启发的局部结构拟合卷积(LSFConv)精确捕获局部几何细节，在ScanObjectNN上以纯监督学习方式达到95.4%准确率，超越所有预训练方法。

**[Grm Large Gaussian Reconstruction Model For Efficient 3D Reconstruction And Gene](3d_vision/grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)**

:   提出GRM，一种基于纯Transformer架构的前馈式3D重建模型，将稀疏视图(4张图)的像素通过pixel-aligned Gaussians转化为稠密的3D高斯表示，约0.1秒完成重建，结合多视图扩散模型可实现文本/图像到3D生成。

**[Gvgen Text-To-3D Generation With Volumetric Representation](3d_vision/gvgen_text-to-3d_generation_with_volumetric_representation.md)**

:   提出GVGEN，首个直接从文本前馈生成3D高斯的框架，通过将无序高斯组织为结构化体积表示(GaussianVolume)，并设计从粗到精的生成管线（先生成几何体积再预测高斯属性），在约7秒内完成文本到3D生成。

**[Hac Hash-Grid Assisted Context For 3D Gaussian Splatting Compression](3d_vision/hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)**

:   利用结构化二值哈希网格为无序的3DGS锚点建立空间上下文关系，通过条件概率建模和自适应量化实现高效熵编码，达到相比vanilla 3DGS **75×** 的压缩率，同时保持甚至提升渲染质量。

**[Headgas Real-Time Animatable Head Avatars Via 3D Gaussian Splatting](3d_vision/headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)**

:   提出HeadGaS，通过为每个3D高斯基元配备可学习的潜在特征基底，利用表情参数线性混合特征并经MLP预测表情相关的颜色和不透明度，实现**实时（250+ fps）**且高质量的可动画头部重建，PSNR超越基线约2 dB。

**[High-Precision Self-Supervised Monocular Depth Estimation With Rich-Resource Pri](3d_vision/high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)**

:   提出RPrDepth，在训练阶段利用多帧/高分辨率等"富资源"模型的特征和预测作为先验，通过先验深度融合模块和富资源引导损失，使仅用**低分辨率单张图像**推理的模型达到甚至超过多帧高分辨率模型的深度估计精度。

**[Idol Unified Dual-Modal Latent Diffusion For Human-Centric Joint Video-Depth Gen](3d_vision/idol_unified_dual-modal_latent_diffusion_for_human-centric_joint_video-depth_gen.md)**

:   提出IDOL框架，通过统一双模态U-Net和运动一致性损失，实现以人为中心的视频与深度图联合生成，显著优于现有方法。

**[Implicit Filtering For Learning Neural Signed Distance Functions From 3D Point C](3d_vision/implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)**

:   提出一种非线性隐式滤波器，在不需要法线的情况下对神经SDF的隐式场进行平滑同时保留尖锐几何细节，并通过扩展到非零等值面实现全场一致性正则化。

**[Improving 2D Feature Representations By 3D-Aware Fine-Tuning](3d_vision/improving_2d_feature_representations_by_3d-aware_fine-tuning.md)**

:   通过将2D基础模型特征提升到3D Gaussian表示中实现多视角融合，再用渲染的3D感知特征反向微调2D模型，以线性探测即可提升语义分割和深度估计性能。

**[Improving Domain Generalization In Self-Supervised Monocular Depth Estimation Vi](3d_vision/improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)**

:   提出 SCAT 框架，通过缩放深度网络（SDN）降低 UNet 跳跃连接对扰动的敏感性，并引入冲突梯度手术（CGS）解决对抗增强导致的双重优化冲突，首次将对抗数据增强成功应用于自监督单目深度估计以提升跨域泛化能力。

**[Invertible Neural Warp For Nerf](3d_vision/invertible_neural_warp_for_nerf.md)**

:   提出用可逆神经网络（INN）过参数化相机位姿的刚性变换函数，在 NeRF 联合优化中显著提升位姿估计精度和重建质量，证明可逆性是 MLP 建模刚性 warp 的关键约束。

**[Jointdreamer Ensuring Geometry Consistency And Text Congruen](3d_vision/jointdreamer_ensuring_geometry_consistency_and_text_congruen.md)**

:   JointDreamer提出JSD通过能量函数建模多视角联合分布确保3D一致性。

**[Jointdreamer Ensuring Geometry Consistency And Text Congruence In Text-To-3D Gen](3d_vision/jointdreamer_ensuring_geometry_consistency_and_text_congruence_in_text-to-3d_gen.md)**

:   提出联合分数蒸馏（JSD），通过能量函数建模多视图去噪图像的联合分布，将 SDS 从单视图独立优化扩展为多视图联合优化，有效解决 3D 生成中的 Janus 问题，同时保持对复杂文本的生成保真度。

**[Lagrangian Hashing For Compressed Neural Field Representations](3d_vision/lagrangian_hashing_for_compressed_neural_field_representations.md)**

:   将InstantNGP的欧拉网格哈希表与拉格朗日点云表示相结合，在哈希桶中存储可移动的高斯特征点，实现**参数量减少1.8-2.8倍**但重建质量不降的紧凑神经场表示。

**[Language-Driven 6-Dof Grasp Detection Using Negative Prompt Guidance](3d_vision/language-driven_6-dof_grasp_detection_using_negative_prompt_guidance.md)**

:   提出大规模语言驱动6-DoF抓取数据集Grasp-Anything-6D（1M场景、200M抓取姿态），以及基于扩散模型的LGrasp6D方法，核心创新是**负提示引导（Negative Prompt Guidance）**策略，在推理时引导抓取姿态远离非目标物体。

**[Lara Efficient Large-Baseline Radiance Fields](3d_vision/lara_efficient_large-baseline_radiance_fields.md)**

:   提出LaRa前馈重建模型，通过**高斯体积（Gaussian Volume）**表示和**分组注意力层（Group Attention Layer）**统一局部与全局推理，仅需4张图像即可从大基线视角重建360°辐射场，且仅用**4×A100训练2天**即可超越LGM等费时方法。

**[Learning 3D-Aware Gans From Unposed Images With Template Feature Field](3d_vision/learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)**

:   提出模板特征场(TeFF)，通过联合学习生成辐射场和语义特征场，从无姿态标注的野外图像中自动提取3D模板并在线估计相机位姿，从而实现完整3D几何的生成对抗学习。

**[Learning 3D Geometry And Feature Consistent Gaussian Splatting For Object Remova](3d_vision/learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)**

:   提出 GScream 框架，通过单目深度引导训练和交叉注意力特征正则化，在 3D Gaussian Splatting 表示下实现高质量的物体移除，同时保持几何一致性和纹理连贯性。

**[Learning To Generate Conditional Tri-Plane For 3D-Aware Expression Controllable ](3d_vision/learning_to_generate_conditional_tri-plane_for_3d-aware_expression_controllable_.md)**

:   提出 Export3D，通过对比预训练获取与外观解耦的表情表示（CLeBS），结合表情自适应层归一化（EAdaLN）直接生成条件tri-plane，实现无外观交换的跨身份3D-aware人像表情动画。

**[Leia Latent View-Invariant Embeddings For Implicit 3D Articulation](3d_vision/leia_latent_view-invariant_embeddings_for_implicit_3d_articulation.md)**

:   提出LEIA方法，通过学习视角不变的潜在嵌入来表征铰接物体的不同状态，利用超网络(HyperNetwork)调制NeRF权重，实现在未见过的铰接配置之间进行平滑插值，无需任何运动先验或3D监督。

**[Ln3Diff Scalable Latent Neural Fields Diffusion For Speedy 3D Generation](3d_vision/ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)**

:   提出LN3Diff++框架，通过3D感知的VAE将多视角图像压缩到紧凑的3D潜在空间，在该空间上训练扩散模型（U-Net或DiT），实现高质量、快速、通用的条件3D生成，包括文本到3D和图像到3D。

**[Mariner Enhancing Novel Views By Matching Rendered Images With Nearby References](3d_vision/mariner_enhancing_novel_views_by_matching_rendered_images_with_nearby_references.md)**

:   提出MaRINeR方法，利用附近参考图像通过深度特征匹配和层次化细节传输来增强3D重建的渲染图像质量，适用于显式（mesh）和隐式（NeRF）等多种3D表示的渲染后处理。

**[Megascenes Scene-Level View Synthesis At Scale](3d_vision/megascenes_scene-level_view_synthesis_at_scale.md)**

:   从Wikimedia Commons互联网照片构建包含10万+SfM重建的大规模场景级3D数据集MegaScenes，并结合warp条件和位姿条件提升场景级新视角合成的位姿一致性。

**[Mesh2Nerf Direct Mesh Supervision For Neural Radiance Field Representation And G](3d_vision/mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)**

:   提出Mesh2NeRF，通过解析解直接从纹理网格(textured mesh)构造GT辐射场，用occupancy函数建模密度场、用反射模型建模颜色场，为NeRF表示与生成任务提供精确的3D逐点监督。

**[Meshfeat Multi-Resolution Features For Neural Fields On Meshes](3d_vision/meshfeat_multi-resolution_features_for_neural_fields_on_meshes.md)**

:   提出MeshFeat，一种适用于mesh上神经场的参数化多分辨率特征编码方法，利用网格简化算法构建多分辨率特征表示，在保持重建质量的同时实现13倍推理加速。

**[Migs Multi-Identity Gaussian Splatting Via Tensor Decomposition](3d_vision/migs_multi-identity_gaussian_splatting_via_tensor_decomposition.md)**

:   提出MIGS，通过CP张量分解将多个人体身份的3DGS参数统一到一个低秩张量中，在大幅减少参数量的同时实现了对未见姿态的鲁棒动画。

**[Milliflow Scene Flow Estimation On Mmwave Radar Point Cloud For Human Motion Sen](3d_vision/milliflow_scene_flow_estimation_on_mmwave_radar_point_cloud_for_human_motion_sen.md)**

:   提出首个毫米波雷达点云场景流估计方法 milliFlow，通过多尺度特征提取、全局聚合、GRU 时序传播和约束回归，在自建数据集上将 EPE3D 从次优 0.107m 降至 0.046m（cm 级精度），并展示场景流特征对人体活动识别（+7.9%）、人体部位解析（+3.6%）、人体追踪等下游任务的增强效果。

**[Mvdd Multi-View Depth Diffusion Models](3d_vision/mvdd_multi-view_depth_diffusion_models.md)**

:   提出MVDD，一个基于多视角深度图表示的扩散模型，通过极线"线段"注意力和去噪深度融合实现3D一致的高质量形状生成，可生成20K+点的稠密点云。

**[Mvsgaussian Fast Generalizable Gaussian Splatting Reconstruction From Multi-View](3d_vision/mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)**

:   将MVS的代价体深度估计与3D高斯溅射结合，通过混合渲染(splatting+volume rendering)提升泛化性，并提出基于多视图几何一致性的点云聚合策略，使per-scene优化仅需45秒就超越3D-GS的10分钟效果。

**[Mvsplat Efficient 3D Gaussian Splatting From Sparse Multi-View Images](3d_vision/mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)**

:   提出MVSplat，通过plane-sweep构建代价体（cost volume）来精确定位Gaussian中心，以极少参数量（pixelSplat的1/10）和最快推理速度（22fps）实现了稀疏视角前馈式3D Gaussian预测的SOTA。

**[Ngp-Rt Fusing Multi-Level Hash Features With Lightweight Attention For Real-Time](3d_vision/ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)**

:   提出NGP-RT，通过轻量注意力机制聚合多级显式哈希特征替代per-point MLP，并引入占用距离网格减少光线行进中的内存访问，在Mip-NeRF 360数据集上实现1080p 108fps的实时NeRF渲染。

**[Novum Neural Object Volumes For Robust Object Classification](3d_vision/novum_neural_object_volumes_for_robust_object_classification.md)**

:   提出 NOVUM 架构，为每个物体类别维护一个由 3D 高斯组成的神经体积表征，通过将图像特征与各类别的高斯特征匹配实现分类，在遮挡/损坏/真实 OOD 场景下相比 ResNet/ViT/Swin 等标准架构分类准确率提升 6-33%，同时支持 3D 位姿估计和可解释性可视化。

**[Omni-Recon Harnessing Image-Based Rendering For General-Purpose Neural Radiance ](3d_vision/omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)**

:   提出Omni-Recon框架，通过基于图像的渲染（IBR）管线构建通用NeRF，利用解耦的几何/外观双分支设计，首次在单一模型中实现可泛化3D重建、零样本多任务场景理解和实时渲染、场景编辑等多种下游3D任务的适配。

**[Omni6D Large-Vocabulary 3D Object Dataset For Category-Level 6D Object Pose Esti](3d_vision/omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)**

:   构建了 **Omni6D**——首个大规模类别级 6DoF 姿态估计 RGBD 数据集，覆盖 **166 个类别、4688 个实例、80 万张图像**，远超现有 NOCS 等数据集（仅 6 类），并提出对称感知评估指标和渐进式微调策略。

**[On The Error Analysis Of 3D Gaussian Splatting And An Optimal Projection Strateg](3d_vision/on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)**

:   从数学上系统分析3D Gaussian Splatting中局部仿射近似引入的投影误差，证明误差函数在Gaussian均值方向与投影平面法线重合时取极小值，据此提出每个Gaussian投影到各自切平面的最优投影策略(Optimal Gaussian Splatting)，在不影响实时性能的前提下显著降低渲染伪影。

**[Open-Vocabulary 3D Semantic Segmentation With Text-To-Image Diffusion Models](3d_vision/open-vocabulary_3d_semantic_segmentation_with_text-to-image_diffusion_models.md)**

:   提出 Diff2Scene，首次将预训练的文本-图像扩散模型（Stable Diffusion）用于开放词汇3D语义分割，通过创新的掩码蒸馏方法将2D基础模型的语义丰富mask嵌入迁移到3D几何感知mask模型，在 ScanNet200 上超越 SOTA 12%。

**[Open Vocabulary 3D Scene Understanding Via Geometry Guided Self-Distillation](3d_vision/open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)**

:   提出 GGSD 框架，利用3D几何先验（超点语义一致性）引导从2D模型到3D模型的知识蒸馏，并通过自蒸馏机制进一步挖掘3D数据的表征优势，在室内外开放词汇3D场景理解任务上大幅超越现有方法。

**[P2P-Bridge Diffusion Bridges For 3D Point Cloud Denoising](3d_vision/p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)**

:   提出 P2P-Bridge，将点云去噪建模为 Schrödinger Bridge 问题，学习噪声点云到干净点云之间的最优传输计划，首次引入数据到数据（而非数据到噪声）的扩散范式，在合成数据和真实室内场景（ScanNet++、ARKitScenes）上均大幅超越现有方法。

**[Part2Object Hierarchical Unsupervised 3D Instance Segmentation](3d_vision/part2object_hierarchical_unsupervised_3d_instance_segmentation.md)**

:   提出 Part2Object 层次聚类框架，利用自监督特征和3D物体性先验（objectness prior），从零件级过分割逐层合并到物体级实例，生成高质量伪标签用于自训练 Hi-Mask3D，实现无需人工标注的3D实例分割。

**[Pcf-Lift Panoptic Lifting By Probabilistic Contrastive Fusion](3d_vision/pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)**

:   提出 PCF-Lift，通过概率特征嵌入（多元高斯分布）替代确定性特征，结合概率乘积核（PP Kernel）的对比损失和跨视图约束，有效应对2D分割中的不一致分割和不一致ID问题，在 ScanNet 和 Messy Room 数据集上显著超越前沿方法。

**[Per-Gaussian Embedding-Based Deformation For Deformable 3D Gaussian Splatting](3d_vision/per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)**

:   提出基于逐高斯嵌入（Per-Gaussian Embedding）的形变表示方法，将形变定义为逐高斯潜在嵌入与时间嵌入的函数，辅以粗细形变分解和局部平滑正则化，在多个动态场景数据集上取得了质量、速度和模型容量的全面优势。

**[Physdreamer Physics-Based Interaction With 3D Objects Via Video Generation](3d_vision/physdreamer_physics-based_interaction_with_3d_objects_via_video_generation.md)**

:   利用视频生成模型中隐含的物理动力学先验，为静态3D高斯对象估计空间变化的杨氏模量材料场，从而实现物理合理的交互式3D动力学合成。

**[Pointllm Empowering Large Language Models To Understand Point Clouds](3d_vision/pointllm_empowering_large_language_models_to_understand_point_clouds.md)**

:   将点云编码器（Point-BERT）通过 MLP 投影层对接 LLaMA 大语言模型，构建 PointLLM；利用 730K 指令数据（660K 简述 + 70K 复杂指令）两阶段训练后，在 3D 物体分类上达到 53.4% 生成式准确率（超越 LLaVA-13B 的 44.2%），在物体描述任务上人类评估胜率 55%（超越人工标注）。

**[Prodepth Boosting Self-Supervised Multi-Frame Monocular Depth With Probabilistic](3d_vision/prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)**

:   提出一种概率融合框架 ProDepth，通过辅助解码器推断动态区域不确定性，以加权几何均值自适应融合单帧和多帧深度概率分布来修正代价体中的错误匹配代价，并配合不确定性感知的损失重加权策略，在自监督多帧单目深度估计中取得 SOTA。

**[Progressive Classifier And Feature Extractor Adaptation For Unsupervised Domain ](3d_vision/progressive_classifier_and_feature_extractor_adaptation_for_unsupervised_domain_.md)**

:   提出 PCFEA 方法用于点云无监督域自适应，通过渐进构建从源域到目标域的中间域，在宏观层面用目标风格特征增强训练分类器（PTFA），微观层面引导特征提取器向中间域对齐（IDFA），在 PointDA-10 上均值准确率达 76.5%（超 SOTA +2.9%），GraspNetPC-10 上达 87.6%（超 SOTA +13.7%）。

**[Ray-Distance Volume Rendering For Neural Scene Reconstruction](3d_vision/ray-distance_volume_rendering_for_neural_scene_reconstruction.md)**

:   提出 RS-Recon 方法，用射线方向相关的有符号射线距离函数（SRDF）替代传统 SDF 来参数化体渲染中的密度函数，结合 SRDF-SDF 一致性损失和自监督可见性任务，在多物体室内场景重建中取得更准确的表面和视图合成。

**[Reconstruction And Simulation Of Elastic Objects With Spring-Mass 3D Gaussians](3d_vision/reconstruction_and_simulation_of_elastic_objects_with_spring-mass_3d_gaussians.md)**

:   提出 Spring-Gaus，将3D弹簧-质点模型集成到3D高斯表示中，从多视角视频中重建弹性物体的外观、几何和物理动力学属性，实现未来预测和不同环境条件下的仿真。

**[Reloo Reconstructing Humans Dressed In Loose Garments From Monocular Video In Th](3d_vision/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)**

:   提出 ReLoo，通过分层神经人体表示和非层级虚拟骨骼变形模块，从单目野外视频中重建穿着宽松服装的高质量3D人体模型。

**[Risurconv Rotation Invariant Surface Attention-Augmented Convolutions For 3D Poi](3d_vision/risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)**

:   提出 RISurConv，通过构建局部三角表面并提取高表达力旋转不变表面属性（RISP），结合注意力增强卷积，实现首次在精度上超越非旋转不变方法的旋转不变点云分析网络。

**[Roguenerf A Robust Geometry-Consistent Universal Enhancer For Nerf](3d_vision/roguenerf_a_robust_geometry-consistent_universal_enhancer_for_nerf.md)**

:   提出 RoGUENeRF，通过结合3D对齐、非刚性精炼和几何感知注意力机制，作为NeRF的后处理增强器，显著提升多种NeRF基线的渲染质量，同时保持视角一致性和对不准确相机标定的鲁棒性。

**[Rpbg Towards Robust Neural Point-Based Graphics In The Wild](3d_vision/rpbg_towards_robust_neural_point-based_graphics_in_the_wild.md)**

:   提出 RPBG，通过退化感知卷积（DAC）模块增强神经渲染器、轻量环境建模和点云增强策略，使基于点的神经重渲染在各类野外场景中表现稳定优越。

**[Sags Structure-Aware 3D Gaussian Splatting](3d_vision/sags_structure-aware_3d_gaussian_splatting.md)**

:   提出 SAGS，通过局部-全局图表示和图神经网络隐式编码场景几何结构，在保持实时渲染的同时提升3DGS的渲染质量、减少存储需求（最高24×压缩），并显著抑制浮点伪影。

**[Sapiens Foundation For Human Vision Models](3d_vision/sapiens_foundation_for_human_vision_models.md)**

:   Sapiens 提出了一个以人为中心的视觉基础模型家族（0.3B-2B参数），通过在3亿张人体图像上进行 MAE 自监督预训练，原生支持1K高分辨率推理，在2D姿态估计、身体部位分割、深度估计和表面法线预测四个人体视觉任务上全面超越现有SOTA。

**[Scanreason Empowering 3D Visual Grounding With Reasoning Capabilities](3d_vision/scanreason_empowering_3d_visual_grounding_with_reasoning_capabilities.md)**

:   提出 3D reasoning grounding 新任务和 ScanReason 基准（10K+ QA-location pairs，5种推理类型），设计 ReGround3D 框架将 MLLM 推理与 3D grounding 模块通过 Chain-of-Grounding 机制协同，在隐式指令下实现准确的 3D 目标定位。

**[Scatterformer Efficient Voxel Transformer With Scattered Linear Attention](3d_vision/scatterformer_efficient_voxel_transformer_with_scattered_linear_attention.md)**

:   提出 ScatterFormer，首个直接对跨窗口的变长体素序列施加线性注意力的体素 Transformer，通过 Scattered Linear Attention (SLA) 模块和 chunk-wise 矩阵乘法算法实现亚毫秒级延迟，配合 Cross-Window Interaction (CWI) 模块替代窗口平移，在 Waymo 和 nuScenes 上达到 SOTA 精度的同时保持 23 FPS 的检测速度。

**[Scenegraphloc Cross-Modal Coarse Visual Localization On 3D Scene Graphs](3d_vision/scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)**

:   提出 SceneGraphLoc，将查询图像在由多模态 3D 场景图组成的参考地图中进行粗定位，在不依赖大规模图像数据库的前提下，实现了接近 SOTA 图像级方法的定位精度，同时存储需求降低三个数量级。

**[Scenegraphloc Crossmodal Coarse Visual Localization On 3D Sc](3d_vision/scenegraphloc_crossmodal_coarse_visual_localization_on_3d_sc.md)**

:   提出SceneGraphLoc，首次将queryimage在多模态3D场景图数据库中进行粗定位，通过学习场景图节点和图像patch的统一嵌入空间，在存储效率提升1000倍的同时接近图像检索方法的定位精度。

**[Sceneverse Scaling 3D Vision-Language Learning For Grounded Scene Understanding](3d_vision/sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)**

:   提出首个百万级 3D 视觉-语言数据集 SceneVerse（68K 室内场景 + 2.5M 场景-语言对），结合多层级对比预训练框架 GPS，在 3D visual grounding 和 QA 任务上取得 SOTA，并展现零样本迁移能力。

**[Sceneverse Scaling 3D Visionlanguage Learning For Grounded S](3d_vision/sceneverse_scaling_3d_visionlanguage_learning_for_grounded_s.md)**

:   提出SceneVerse——首个百万级3D视觉语言数据集（68K场景+250万语言描述），通过结合人工标注和基于场景图的自动生成pipeline构建多粒度描述，并设计GPS预训练框架实现多层次场景-文本对齐，在3D grounding和QA基准上达到SOTA。

**[Seed A Simple And Effective 3D Detr In Point Clouds](3d_vision/seed_a_simple_and_effective_3d_detr_in_point_clouds.md)**

:   SEED 提出了一种简洁高效的 3D DETR 检测器，通过双重查询选择（DQS）模块以粗到精方式获取高质量查询，结合可变形网格注意力（DGA）模块利用 3D 物体的几何结构信息实现灵活的查询交互，在 Waymo 和 nuScenes 上达到新 SOTA。

**[Segpoint Segment Any Point Cloud Via Large Language Model](3d_vision/segpoint_segment_any_point_cloud_via_large_language_model.md)**

:   提出 SegPoint，首个利用多模态 LLM 推理能力在统一框架中完成 3D 指令分割、引用分割、语义分割和开放词汇分割四种任务的模型，并构建 Instruct3D 基准测试（2,565 对），mIoU 达 27.5%。

**[Semantichuman-Hd High-Resolution Semantic Disentangled 3D Human Generation](3d_vision/semantichuman-hd_high-resolution_semantic_disentangled_3d_human_generation.md)**

:   提出SemanticHuman-HD，首个实现语义解耦的3D人体图像合成方法，通过K个独立局部生成器和3D感知超分模块，实现1024²分辨率的语义可控人体生成。

**[Sgs-Slam Semantic Gaussian Splatting For Neural Dense Slam](3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)**

:   提出SGS-SLAM，首个基于Gaussian Splatting的语义视觉SLAM系统，通过多通道优化融合外观、几何和语义特征，在相机姿态估计、地图重建和语义分割方面均达到SOTA。

**[Sinder Repairing The Singular Defects Of Dinov2](3d_vision/sinder_repairing_the_singular_defects_of_dinov2.md)**

:   揭示DINOv2特征图中高范数缺陷token的根源是网络权重的主左奇异向量（singular defect），并提出SINDER——仅需小数据集微调奇异值即可修复缺陷，同时保持特征质量。

**[Slotlifter Slot-Guided Feature Lifting For Learning Object-Centric Radiance Fiel](3d_vision/slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)**

:   提出SlotLifter，通过将2D特征提升为3D并与Slot Attention结合的slot-guided feature lifting设计，在场景分解和新视角合成上同时达到SOTA，且训练效率提升约5倍。

**[Sparsessp 3D Subcellular Structure Prediction From Sparse-View Transmitted Light](3d_vision/sparsessp_3d_subcellular_structure_prediction_from_sparse-view_transmitted_light.md)**

:   提出SparseSSP，通过混合维度拓扑（3D编码+2D解码）和一对多体素映射，从稀疏透射光切片预测3D荧光体素网格，最多减少87.5%成像频率且保持领先性能。

**[Supergaussian Repurposing Video Models For 3D Super Resolution](3d_vision/supergaussian_repurposing_video_models_for_3d_super_resolution.md)**

:   提出SuperGaussian，通过复用预训练视频上采样模型实现3D超分辨率，无需类别特定训练，可处理多种3D输入格式（高斯、NeRF、网格等），输出高质量Gaussian Splat模型。

**[T-Mae Temporal Masked Autoencoders For Point Cloud Representation Learning](3d_vision/t-mae_temporal_masked_autoencoders_for_point_cloud_representation_learning.md)**

:   T-MAE 提出了一种时序掩码自编码器预训练策略，以时序相邻两帧为输入，通过掩码当前帧并借助历史帧信息重建来学习时序依赖关系，配合设计的 SiamWCA（孪生编码器+窗口交叉注意力）架构，在 Waymo 和 ONCE 数据集上以更少标注数据和更少训练迭代次数超越 SOTA 自监督方法。

**[Talkinggaussian Structure-Persistent 3D Talking Head Synthesis Via Gaussian Spla](3d_vision/talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)**

:   提出TalkingGaussian，基于3D高斯溅射的形变驱动说话人头部合成框架，通过对持久性高斯基元施加平滑形变表示面部运动，并分解面部和口腔内部区域以解决运动不一致问题。

**[Taming Latent Diffusion Model For Neural Radiance Field Inpainting](3d_vision/taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)**

:   提出MALD-NeRF，通过掩码对抗训练和场景定制的潜在扩散模型实现高质量NeRF修复，有效解决扩散模型的多视角不一致和纹理偏移问题。

**[Temporally Consistent Stereo Matching](3d_vision/temporally_consistent_stereo_matching.md)**

:   提出TC-Stereo，通过时序视差补全提供良好初始化、时序状态融合保持隐藏态连贯性，以及双空间（视差+视差梯度）迭代精炼改善病态区域，实现时间一致的立体匹配。

**[Texture-Gs Disentangling The Geometry And Texture For 3D Gaussian Splatting Edit](3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)**

:   提出Texture-GS，首次为3D高斯溅射解耦几何与纹理，通过UV映射MLP和局部Taylor展开将外观表示为2D纹理图，实现实时纹理替换和编辑（58 FPS，RTX 2080 Ti）。

**[The Nerfect Match Exploring Nerf Features For Visual Localization](3d_vision/the_nerfect_match_exploring_nerf_features_for_visual_localization.md)**

:   提出NeRFMatch，探索NeRF内部特征作为3D描述子的潜力，建立基于注意力机制的2D-3D匹配网络，在Cambridge Landmarks上实现有竞争力的定位性能，验证了NeRF作为定位场景表示的可行性。

**[Thermal3D-Gs Physics-Induced 3D Gaussians For Thermal Infrared Novel-View Synthe](3d_vision/thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)**

:   提出Thermal3D-GS，通过神经网络建模大气传输效应和热传导物理过程，并引入温度一致性约束，实现热红外图像的高质量新视角合成，创建了首个大规模热红外新视角合成数据集TI-NSD。

**[Tpa3D Triplane Attention For Fast Text-To-3D Generation](3d_vision/tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)**

:   提出TPA3D，一个基于GAN的文本引导3D生成框架，通过三平面注意力（TPA）模块在句子级和词级特征上进行逐层细化，实现快速且细粒度的文本到3D纹理网格生成。

**[Track Everything Everywhere Fast And Robustly](3d_vision/track_everything_everywhere_fast_and_robustly.md)**

:   提出一种高效鲁棒的测试时优化像素跟踪方法，通过引入CaDeX++可逆变形网络、单目深度先验和DINOv2长期语义一致性，将训练速度提升10倍以上，同时显著提高了跟踪精度和鲁棒性。

**[Tracknerf Bundle Adjusting Nerf From Sparse And Noisy Views Via Feature Tracks](3d_vision/tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)**

:   提出TrackNeRF，将SfM中的特征轨迹（feature tracks）引入NeRF训练，通过全局多视角重投影一致性损失替代传统的成对对应损失，显著提升稀疏+有噪声位姿下的NeRF重建质量和位姿优化精度。

**[Tram Global Trajectory And Motion Of 3D Humans From In-The-Wild Videos](3d_vision/tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)**

:   提出TRAM，一个两阶段方法，通过鲁棒化SLAM恢复度量尺度相机运动 + 视频Transformer（VIMO）回归相机坐标系下的人体运动，组合两者实现准确的世界坐标系3D人体全局轨迹与动作重建。

**[Transferable 3D Adversarial Shape Completion Using Diffusion Models](3d_vision/transferable_3d_adversarial_shape_completion_using_diffusion_models.md)**

:   提出3DAdvDiff，利用3D扩散模型通过对抗性形状补全生成高质量的迁移性3D对抗点云，结合模型不确定性、集成对抗引导和显著性评分策略，在黑盒设置下对最新3D模型实现SOTA攻击成功率。

**[Unidream Unifying Diffusion Priors For Relightable Text-To-3D Generation](3d_vision/unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)**

:   提出UniDream，通过训练albedo-法线对齐的多视角扩散模型（AN-MVM），结合Transformer重建模型和分阶段SDS优化，实现可重光照的文本到3D生成，生成的3D物体具有干净的albedo纹理和PBR材质。

**[Vcd-Texture Variance Alignment Based 3D-2D Co-Denoising For Text-Guided Texturin](3d_vision/vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)**

:   提出VCD-Texture，在Stable Diffusion去噪过程中统一2D和3D自注意力学习（JNP），通过方差对齐（VA）解决光栅化引起的方差衰减问题，并用修复细化处理不一致区域，实现高保真、高一致性的3D纹理合成。

**[View Selection For 3D Captioning Via Diffusion Ranking](3d_vision/view_selection_for_3d_captioning_via_diffusion_ranking.md)**

:   DiffuRank用预训练text-to-3D扩散模型评估视角对齐度选择最佳视角减少幻觉。

**[Vista3D Unravel The 3D Darkside Of A Single Image](3d_vision/vista3d_unravel_the_3d_darkside_of_a_single_image.md)**

:   提出Vista3D，通过粗到细的两阶段框架（高斯溅射→FlexiCubes等值面细化+解耦纹理），结合角度扩散先验组合，5分钟内从单张图像生成多样且一致的高保真3D网格。

**[Wast-3D Wasserstein-2 Distance For Scene-To-Scene Stylization On 3D Gaussians](3d_vision/wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)**

:   提出WaSt-3D，利用3D高斯溅射表示将风格迁移重新定义为两个高斯分布之间的最优传输问题，通过Sinkhorn散度匹配内容场景和风格场景的3D分布，首次实现了3D场景到场景的几何风格迁移。

**[When Do We Not Need Larger Vision Models](3d_vision/when_do_we_not_need_larger_vision_models.md)**

:   提出 Scaling on Scales (S2) 策略：冻结小模型（如 ViT-B）在多个图像尺度上运行并拼接特征，无需增加参数即可在分类、分割、深度估计、MLLM 等任务上匹敌甚至超越大模型（ViT-H/G），并从理论和实验上论证了大模型学到的表征大部分可由多尺度小模型线性近似。

**[Zero-Shot Multi-Object Scene Completion](3d_vision/zero-shot_multi-object_scene_completion.md)**

:   提出OctMAE，一种融合Octree U-Net和隐空间3D MAE的混合架构，从单张RGB-D图像实现高质量近实时的多物体场景形状补全，通过遮挡掩码策略和3D旋转位置编码显著提升效率和泛化能力。

**[Zest Zero-Shot Material Transfer From A Single Image](3d_vision/zest_zero-shot_material_transfer_from_a_single_image.md)**

:   提出ZeST，一种零样本免训练的材质迁移方法，通过IP-Adapter提取材质表示、ControlNet提供几何引导、前景灰度图提供光照线索，三条分支组合实现从单张材质样本图像到目标物体的2D材质迁移。

---

## 🎨 图像生成 { #image_generation }

**[2S-Odis Two-Stage Omni-Directional Image Synthesis By Geometric Distortion Corre](image_generation/2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)**

:   2S-ODIS通过两阶段结构利用预训练VQGAN（无需微调）合成全景图像：第一阶段生成低分辨率粗略ERP图，第二阶段通过生成26个NFoV局部图像并融合来校正几何畸变，训练时间从14天缩短到4天且图像质量更优。

**[A Diffusion Model For Simulation Ready Coronary Anatomy With](image_generation/a_diffusion_model_for_simulation_ready_coronary_anatomy_with.md)**

:   用潜在扩散模型（LDM）可控生成3D多组织冠状动脉分割图，通过拓扑交互损失保证解剖合理性，通过形态-骨架双通道条件化实现对截面形态和分支结构的解耦控制，并提出自适应空条件引导（ANG）以非可微回归器高效增强条件保真度，最终支持面向有限元仿真的反事实解剖结构编辑。

**[Accdiffusion An Accurate Method For Higher-Resolution Image Generation](image_generation/accdiffusion_an_accurate_method_for_higher-resolution_image_generation.md)**

:   提出AccDiffusion，通过将全局文本prompt解耦为patch级别的内容感知prompt（利用cross-attention map判断每个词汇是否属于某patch），并引入带窗口交互的膨胀采样来改善全局一致性，在无需额外训练的情况下有效解决patch-wise高分辨率图像生成中的目标重复问题，在SDXL上实现了从2K到4K分辨率的无重复高质量图像外推。

**[Adadiffsr Adaptive Region-Aware Dynamic Acceleration Diffusion Model For Real-Wo](image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)**

:   观察到扩散模型超分中不同图像区域所需去噪步数差异巨大（背景区域早已收敛而前景纹理仍需迭代），提出基于多指标潜在熵（MMLE）感知信息增益来动态跳步的策略，将子区域分为稳定/增长/饱和三类给予不同步长，并通过渐进特征注入（PFJ）平衡保真度与真实感，在DRealSR等数据集上取得与StableSR可比的质量但推理时间和FLOPs分别减少1.5×和2.7×。

**[Adagen Learning Adaptive Policy For Image Synthesis](image_generation/adagen_learning_adaptive_policy_for_image_synthesis.md)**

:   将多步生成模型（MaskGIT/AR/Diffusion/Rectified Flow）的步级参数调度（温度、mask ratio、CFG scale、timestep等）统一建模为MDP，用轻量RL策略网络实现样本自适应调度，并提出对抗奖励设计防止策略过拟合，在四种生成范式上一致提升性能（VAR FID 1.92→1.59，DiT-XL推理成本降3倍同时性能更优）。

**[Adanat Exploring Adaptive Policy For Token-Based Image Generation](image_generation/adanat_exploring_adaptive_policy_for_token-based_image_generation.md)**

:   提出AdaNAT，将非自回归Transformer（NAT）的生成策略配置建模为MDP，通过轻量策略网络+PPO强化学习+对抗奖励模型自动为每个样本定制生成策略（重掩码比例、采样温度、CFG权重等），在ImageNet-256上仅用8步达到FID 2.86，相比手工策略实现约40%的相对提升。

**[Anycontrol Create Your Artwork With Versatile Control On Tex](image_generation/anycontrol_create_your_artwork_with_versatile_control_on_tex.md)**

:   AnyControl提出Multi-Control Encoder，通过交替执行多控制融合块和多控制对齐块，从任意组合的多种空间控制信号中提取统一的多模态embedding，实现高质量、语义对齐的多条件可控图像生成。

**[Anycontrol Create Your Artwork With Versatile Control On Text-To-Image Generatio](image_generation/anycontrol_create_your_artwork_with_versatile_control_on_text-to-image_generatio.md)**

:   提出 AnyControl，通过 Multi-Control Encoder（fusion + alignment 交替块结构）支持任意组合的多种空间控制信号（深度、边缘、分割、姿态），在 COCO 多控制基准上 FID 44.28 全面超越现有方法。

**[Bridging The Gap Studio-Like Avatar Creation From A Monocular Phone Capture](image_generation/bridging_the_gap_studio-like_avatar_creation_from_a_monocular_phone_capture.md)**

:   提出从单目手机视频生成类似影棚级质量的面部纹理贴图的方法，结合 StyleGAN2 的 W+ 空间参数化与扩散模型超分辨率，实现从手机扫描到高质量 3D 头像的跨越。

**[Byteedit Boost Comply And Accelerate Generative Image Editing](image_generation/byteedit_boost_comply_and_accelerate_generative_image_editing.md)**

:   提出 ByteEdit，一个将人类反馈学习引入生成式图像编辑（inpainting/outpainting）的框架，通过美学、对齐、一致性三个奖励模型提升编辑质量，并利用对抗训练和渐进策略加速推理。

**[Challenging Forgets Unveiling The Worst-Case Forget Sets In Machine Unlearning](image_generation/challenging_forgets_unveiling_the_worst-case_forget_sets_in_machine_unlearning.md)**

:   提出从对抗视角识别"最坏情况遗忘集"的方法，通过双层优化框架找到最难被遗忘的数据子集，利用 SignSGD 将二阶 BLO 简化为一阶问题，从而更可靠地评估机器遗忘方法的真实效能。

**[Coin Control-Inpainting Diffusion Prior For Human And Camera Motion Estimation](image_generation/coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)**

:   提出COIN方法，通过控制-补绘（Control-Inpainting）的改进版Score Distillation Sampling，结合人-场景关系损失，从单目动态相机视频中同时估计高质量的全局人体运动和相机运动。

**[Collaborative Control For Geometry-Conditioned Pbr Image Generation](image_generation/collaborative_control_for_geometry-conditioned_pbr_image_generation.md)**

:   提出 Collaborative Control 范式，通过冻结预训练RGB扩散模型并训练一个并行PBR模型，利用双向跨网络通信层联合建模RGB与PBR图像分布，在有限数据下实现高质量的几何条件PBR材质图像生成。

**[Colorpeel Color Prompt Learning With Diffusion Models V](image_generation/colorpeel_color_prompt_learning_with_diffusion_models_v.md)**

:   提出 ColorPeel，通过在目标颜色的基础几何体上联合学习颜色和形状 token 来实现颜色与形状解耦，使 T2I 扩散模型能精确生成用户指定 RGB 颜色的物体。

**[Colorpeel Color Prompt Learning With Diffusion Models Via Color And Shape Disent](image_generation/colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)**

:   提出ColorPeel方法，通过在目标颜色的基本几何形状上学习颜色提示token（解耦颜色与形状），并引入交叉注意力对齐损失，使T2I扩散模型能精确生成用户指定RGB颜色的物体。

**[Controlling The World By Sleight Of Hand](image_generation/controlling_the_world_by_sleight_of_hand.md)**

:   提出 CosHand，通过手部二值掩码作为动作条件，在预训练 Stable Diffusion 上微调，预测手-物交互后的未来图像，并可零样本泛化到机器人末端执行器。

**[Diff-Tracker Text-To-Image Diffusion Models Are Unsupervised Trackers](image_generation/diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)**

:   提出 Diff-Tracker，首次利用预训练文本到图像扩散模型（Stable Diffusion）中蕴含的丰富视觉语义知识进行无监督目标跟踪，通过学习一个表示目标的 prompt 并在线更新来实现持续跟踪。

**[Diffit Diffusion Vision Transformers For Image Generation](image_generation/diffit_diffusion_vision_transformers_for_image_generation.md)**

:   提出 DiffiT（Diffusion Vision Transformer），通过引入时间依赖多头自注意力（TMSA）机制，让自注意力在去噪过程的不同阶段动态调整行为，在ImageNet-256上以比DiT/MDT少16-20%的参数量达到了1.73的SOTA FID分数。

**[Difftracker Texttoimage Diffusion Models Are Unsupervised Tr](image_generation/difftracker_texttoimage_diffusion_models_are_unsupervised_tr.md)**

:   Diff-Tracker利用预训练T2I扩散模型知识进行无监督跟踪，学习prompt在cross-attention上激活目标区域。

**[Diffusion-Based Image-To-Image Translation By Noise Correction Via Prompt Interp](image_generation/diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)**

:   提出PIC（Prompt Interpolation-based Correction），一种无训练的扩散模型图像翻译方法，通过渐进式prompt嵌入插值构造噪声校正项，将其与源图像噪声预测线性组合，实现结构保持的高保真图像编辑，且推理速度（18.1s）优于所有对比方法。

**[Diffusion-Driven Data Replay A Novel Approach To Combat Forgetting In Federated ](image_generation/diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)**

:   提出 DDDR 框架，首次将预训练扩散模型引入联邦类别持续学习（FCCL），通过 Federated Class Inversion 技术为每个类别学习一个紧凑的 class embedding，利用扩散模型高质量回放历史数据以对抗灾难性遗忘，并通过对比学习弥合生成数据与真实数据的域差距。

**[Distilling Diffusion Models Into Conditional Gans](image_generation/distilling_diffusion_models_into_conditional_gans.md)**

:   提出 Diffusion2GAN 框架，将多步扩散模型蒸馏为单步条件GAN，核心创新是 E-LatentLPIPS 潜空间感知损失和基于预训练扩散模型的多尺度条件判别器，在零样本 COCO 基准上超越 DMD、SDXL-Turbo 和 SDXL-Lightning。

**[Dreammotion Space-Time Self-Similar Score Distillation For Zero-Shot Video Editi](image_generation/dreammotion_space-time_self-similar_score_distillation_for_zero-shot_video_editi.md)**

:   提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。

**[Dreammover Leveraging The Prior Of Diffusion Models For Image Interpolation With](image_generation/dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)**

:   提出 DreamMover，基于预训练文本到图像扩散模型实现大运动图像对之间的插值，通过扩散感知光流估计、两级潜空间融合和自注意力拼接替换三个核心组件，生成语义一致的中间帧。

**[Ebdm Exemplar-Guided Image Translation With Brownian-Bridge Diffusion Models](image_generation/ebdm_exemplar-guided_image_translation_with_brownian-bridge_diffusion_models.md)**

:   提出 EBDM 框架，将样例引导的图像翻译建模为随机布朗桥扩散过程，从结构控制直接翻译为真实感图像，通过 Global Encoder、Exemplar Network 和 Exemplar Attention Module 三个组件有效整合样例的全局风格和细节纹理信息。

**[Echoscene Indoor Scene Generation Via Information Echo Over Scene Graph Diffusio](image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)**

:   提出 EchoScene，一个基于双分支扩散模型的 3D 室内场景生成方法，通过信息回声（Information Echo）机制在场景图扩散过程中实现多个去噪过程间的协作信息交换，生成全局一致且可交互控制的场景。

**[Editable Image Elements For Controllable Synthesis](image_generation/editable_image_elements_for_controllable_synthesis.md)**

:   提出"可编辑图像元素"表示，将输入图像分解为一组语义对齐的 patch embeddings（类似超像素），每个 patch 关联位置和尺寸信息，用户可直接编辑这些属性（移动、缩放、删除），再由基于 Stable Diffusion 的解码器合成真实感图像。

**[Emdm Efficient Motion Diffusion Model For Fast And High-Quality Motion Generatio](image_generation/emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)**

:   提出 EMDM，通过条件去噪扩散 GAN 捕获大步长下的复杂去噪分布，实现仅需不超过 10 步采样即可实时生成高质量人体动作，推理速度较 MDM 提升约 200 倍。

**[Emdm Efficient Motion Diffusion Model For Fast And High](image_generation/emdm_efficient_motion_diffusion_model_for_fast_and_high.md)**

:   提出 EMDM，通过条件去噪扩散 GAN 捕获大步长采样时的复杂多模态去噪分布，结合几何损失约束，实现 T≤10 步的实时人体运动生成，推理速度提升 60-240 倍，同时保持高质量。

**[Enhancing Diffusion Models With Text-Encoder Reinforcement Learning](image_generation/enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)**

:   提出 TexForce，通过强化学习（DDPO）结合 LoRA 微调扩散模型的文本编码器以提升图文对齐和视觉质量，并可无缝与已有 U-Net 微调方法组合获得更优效果。

**[Enhancing Perceptual Quality In Video Super-Resolution Through Temporally-Consis](image_generation/enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)**

:   提出 StableVSR，首次将扩散模型应用于视频超分辨率任务，通过时序条件模块（TCM）和帧级双向采样策略，在显著提升感知质量的同时确保帧间时序一致性。

**[Finematch Aspect-Based Fine-Grained Image And Text Mismatch Detection And Correc](image_generation/finematch_aspect-based_fine-grained_image_and_text_mismatch_detection_and_correc.md)**

:   提出 FineMatch 基准，定义了基于方面（Aspect）的细粒度图文不匹配检测与纠正任务，包含 49,906 个高质量人工标注的图文对，并展示了现有 VLM 在细粒度组合性理解上的不足。

**[Finematch Aspectbased Finegrained Image And Text Mismat](image_generation/finematch_aspectbased_finegrained_image_and_text_mismat.md)**

:   提出 FineMatch benchmark，要求模型识别图文对中不匹配的方面短语（Entity/Relation/Attribute/Number）、确定类别并提出修正，构建了 49,906 个人工标注样本，并提出 ITM-IoU 评估指标和 AutoAlign 文生图幻觉检测校正系统。

**[Fouriscale A Frequency Perspective On Training-Free High-Resolution Image Synthe](image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)**

:   提出 FouriScale，从频域分析视角出发，通过膨胀卷积+低通滤波替换预训练扩散模型中的卷积层，实现免训练的任意尺寸高分辨率图像生成，理论上证明了膨胀卷积保持结构一致性的有效性。

**[Freecompose Generic Zero-Shot Image Composition With Diffusion Prior](image_generation/freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)**

:   提出 FreeCompose，利用预训练扩散模型的生成先验实现通用零样本图像合成，统一覆盖图像和谐化（外观编辑）和语义图像合成（语义编辑），无需额外训练。

**[Freediff Progressive Frequency Truncation For Image Edi](image_generation/freediff_progressive_frequency_truncation_for_image_edi.md)**

:   提出 FreeDiff，通过渐进式频率截断从频域精化扩散模型的编辑引导信号，无需微调或修改网络结构，实现覆盖多种编辑类型的通用图像编辑方法。

**[Freediff Progressive Frequency Truncation For Image Editing With Diffusion Model](image_generation/freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)**

:   从频率视角重新审视扩散模型的图像编辑过程，发现去噪网络优先恢复低频分量导致编辑引导与目标区域的misalignment，提出渐进式频率截断（FreeDiff）方法在频率空间精炼引导信号，实现免调优的通用图像编辑。

**[Freeinit Bridging Initialization Gap In Video Diffusion](image_generation/freeinit_bridging_initialization_gap_in_video_diffusion.md)**

:   提出 FreeInit，一种无需额外训练的推理采样策略，通过迭代精炼初始噪声的时空低频分量来弥合视频扩散模型训练与推理之间的初始化差距，显著提升生成视频的时序一致性。

**[Freeinit Bridging Initialization Gap In Video Diffusion Models](image_generation/freeinit_bridging_initialization_gap_in_video_diffusion_models.md)**

:   发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。

**[Garmentaligner Text-To-Garment Generation Via Retrieval-Augmented Multi-Level Co](image_generation/garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)**

:   针对文本到服装图像生成中的细粒度语义错位（组件数量、位置和相互关系），提出 GarmentAligner，通过自动组件提取管线获取空间-数量信息，并结合检索增强对比学习和多级校正损失，实现服装组件的视觉、空间和数量级别的精确对齐。

**[Generating 3D House Wireframes With Semantics](image_generation/generating_3d_house_wireframes_with_semantics.md)**

:   提出基于自回归模型的 3D 房屋线框生成方法，采用统一的线段（wire）表示替代传统的顶点-边分离建模，通过语义感知的 BFS 序列排列和两阶段 coarse-to-fine Transformer 解码器生成语义丰富的线框结构，可自动分割为墙壁、屋顶、房间等语义组件。

**[Generating Human Interaction Motions In Scenes With Text Control](image_generation/generating_human_interaction_motions_in_scenes_with_text_control.md)**

:   提出 TeSMo，一个文本控制的场景感知运动生成方法，通过在大规模运动数据上预训练文本-运动扩散模型，再用增强的场景感知分支进行微调，分两阶段（导航+交互）生成角色在 3D 场景中避障行走并与物体交互（如坐下）的真实运动序列。

**[Getting It Right Improving Spatial Consistency In Text-To-Image Models](image_generation/getting_it_right_improving_spatial_consistency_in_text-to-image_models.md)**

:   系统性调查文本到图像模型的空间关系生成缺陷，发现现有视觉-语言数据集严重缺乏空间描述，据此创建 SPRIGHT 数据集（~600 万张图像重标注空间关系），仅用 <500 张多物体图像微调即在 T2I-CompBench 空间得分上达到 SOTA（0.2133），相比基线提升 41%。

**[Getting It Right Improving Spatial Consistency In Texttoimag](image_generation/getting_it_right_improving_spatial_consistency_in_texttoimag.md)**

:   发现现有VL数据集严重缺乏空间关系描述（如left/right/above/behind出现率极低），构建了首个空间聚焦的大规模数据集SPRIGHT（600万张图像重描述），仅用0.25%数据微调即可提升22%空间一致性得分，用<500张多物体图像微调达到T2I-CompBench空间SOTA 0.2133。

**[Himo A New Benchmark For Full-Body Human Interacting With Multiple Objects](image_generation/himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)**

:   提出 HIMO，首个大规模全身人体与多物体交互的 4D MoCap 数据集（3.3K序列，4.08M帧），并附带详细的文本描述和时间段分割标注，提出双分支条件扩散模型及自回归管线，生成协调的多物体交互动作序列。

**[Hybridbooth Hybrid Prompt Inversion For Efficient Subje](image_generation/hybridbooth_hybrid_prompt_inversion_for_efficient_subje.md)**

:   提出 HybridBooth，融合优化方法和直接回归方法的优势——先用预训练编码器（Word Embedding Probe）生成初始 word embedding，再通过残差精细化（仅 3-5 步）快速适配特定主体，实现高效高保真的 subject-driven 生成。

**[Hybridbooth Hybrid Prompt Inversion For Efficient Subject-Driven Generation](image_generation/hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)**

:   提出 HybridBooth，一种两阶段混合 prompt inversion 框架，通过先用回归器生成初始词嵌入（Probe），再用残差微调（Refinement）仅需 3-5 步迭代即可高效完成主体驱动的个性化图像生成。

**[Idempotent Unsupervised Representation Learning For Skeleton-Based Action Recogn](image_generation/idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)**

:   提出幂等生成模型（IGM），从理论上建立生成模型与最大熵编码（谱对比学习）的等价关系，通过在骨架数据的特征空间施加幂等约束，使生成模型的特征更紧凑、更适合识别任务，在 NTU 60 xsub 上将准确率从 84.6% 提升至 86.2%。

**[Implicit Style-Content Separation Using B-Lora](image_generation/implicit_style-content_separation_using_b-lora.md)**

:   提出 B-LoRA，通过分析 SDXL 架构发现仅联合训练两个特定 transformer block 的 LoRA 权重（Block 4 控制内容、Block 5 控制风格）即可隐式实现单张图片的风格-内容分离，支持风格迁移、文本风格化、一致风格生成等多种任务。

**[Infinite-Id Identity-Preserved Personalization Via Id-Semantics Decoupling Parad](image_generation/infinite-id_identity-preserved_personalization_via_id-semantics_decoupling_parad.md)**

:   提出 Infinite-ID，通过 ID-语义解耦范式将身份信息和文本语义信息分离处理——训练阶段停用文本交叉注意力以专注学习身份嵌入，推理阶段通过混合注意力机制和 AdaIN-mean 操作融合两路信息，在单张参考图下同时实现高保真身份保持和语义一致性。

**[Infiniteid Identitypreserved Personalization Via Idsema](image_generation/infiniteid_identitypreserved_personalization_via_idsema.md)**

:   提出 Infinite-ID，通过 ID-语义解耦范式将身份信息和文本语义信息分离训练，再通过混合注意力机制和 AdaIN-mean 操作在推理时融合，实现高保真身份保持与精确语义控制的平衡。

**[Inftybrush Controllable Large Image Synthesis With Diffusion](image_generation/inftybrush_controllable_large_image_synthesis_with_diffusion.md)**

:   提出首个在无限维函数空间中的条件扩散模型 ∞-Brush，通过交叉注意力神经算子实现可控条件生成，仅用 0.4% 像素训练即可在任意分辨率（最高 4096×4096）上生成保持全局结构的大图像。

**[Irgen Generative Modeling For Image Retrieval](image_generation/irgen_generative_modeling_for_image_retrieval.md)**

:   将图像检索重新定义为生成式建模任务，提出 IRGen——一个序列到序列模型，通过语义图像分词器将图像转化为简短的离散 token 序列，然后自回归地生成查询图像最近邻的标识符，实现端到端可微分的检索并在三个标准基准上达到 SOTA。

**[Latent Guard A Safety Framework For Text-To-Image Generation](image_generation/latent_guard_a_safety_framework_for_text-to-image_generation.md)**

:   提出 Latent Guard 框架，通过在文本编码器的潜在空间中学习黑名单概念与输入提示词的嵌入映射，实现高效、灵活且可抵御对抗攻击的文本到图像生成安全检测。

**[Latent Guard A Safety Framework For Texttoimage Generation](image_generation/latent_guard_a_safety_framework_for_texttoimage_generation.md)**

:   Latent Guard在T2I文本编码器上学习潜在空间检测黑名单概念。

**[Lazy Diffusion Transformer For Interactive Image Editing](image_generation/lazy_diffusion_transformer_for_interactive_image_editing.md)**

:   提出 LazyDiffusion，一种非对称编码器-解码器 Transformer 架构，通过上下文编码器压缩全局信息并仅在 mask 区域执行扩散去噪，实现了与全图生成方法质量相当但速度提升 10 倍的交互式图像编辑。

**[Lcm-Lookahead For Encoder-Based Text-To-Image Personalization](image_generation/lcm-lookahead_for_encoder-based_text-to-image_personalization.md)**

:   提出 LCM-Lookahead 机制，利用 Latent Consistency Model 作为快捷路径在扩散训练中反向传播图像空间损失（如身份损失），结合注意力共享和一致性合成数据生成，显著提升基于编码器的人脸个性化中的身份保持和提示词对齐能力。

**[Lcmlookahead For Encoderbased Texttoimage Personalization](image_generation/lcmlookahead_for_encoderbased_texttoimage_personalization.md)**

:   本文提出利用LCM（Latent Consistency Model）作为"快捷通道"，在扩散模型encoder训练中实现图像空间损失（如身份识别loss）的反向传播，配合自注意力特征共享和一致性数据生成，显著提升encoder-based人脸个性化的身份保持和prompt对齐能力。

**[Learning Differentially Private Diffusion Models Via Stochastic Adversarial Dist](image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)**

:   提出 DP-SAD 框架，通过随机对抗蒸馏训练差分隐私扩散模型：利用扩散模型的时间步稀释 DP 噪声影响，引入判别器加速收敛，并结合梯度链式法则与 DP 后处理特性减少随机性引入，在不需要预训练的条件下实现了 SOTA 的隐私保护图像生成质量。

**[Learning Semantic Latent Directions For Accurate And Controllable Human Motion P](image_generation/learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)**

:   提出语义潜在方向（SLD）方法，通过构建一组正交潜在基方向并将未来运动假设表示为这些方向的线性组合，在随机人体运动预测中实现了更准确、更多样且语义可控的运动预测。

**[Learning Trimodal Relation For Audio-Visual Question Answering With Missing Moda](image_generation/learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)**

:   提出基于三模态关系的缺失模态AVQA框架，通过关系感知缺失模态生成器（RMM）和音视觉关系感知扩散模型（AVR），在推理时缺少音频或视觉模态的情况下依然能准确回答问题。

**[Learning Trimodal Relation For Audiovisual Question Answerin](image_generation/learning_trimodal_relation_for_audiovisual_question_answerin.md)**

:   提出面向音视觉问答（AVQA）的缺失模态处理框架，通过Relation-aware Missing Modal生成器利用三模态关系召回缺失信息，再通过Audio-Visual Relation-aware扩散模型增强特征表示，即使缺少一个模态也能准确回答问题。

**[Lego Learning Egocentric Action Frame Generation Via Vi](image_generation/lego_learning_egocentric_action_frame_generation_via_vi.md)**

:   提出 LEGO 模型，通过视觉指令微调增强 VLLM 的动作描述能力，并将 VLLM 的图像/文本嵌入作为额外条件注入扩散模型，实现从第一人称视角生成动作执行帧。

**[Lego Learning Egocentric Action Frame Generation Via Visual Instruction Tuning](image_generation/lego_learning_egocentric_action_frame_generation_via_visual_instruction_tuning.md)**

:   提出第一人称视角动作帧生成新问题，通过视觉指令微调 VLLM 生成丰富动作描述并将其嵌入作为扩散模型的额外条件，实现高质量的自我中心动作图像合成。

**[Livephoto Real Image Animation With Text-Guided Motion Control](image_generation/livephoto_real_image_animation_with_text-guided_motion_control.md)**

:   提出 LivePhoto 图像动画框架，通过运动强度估计模块和文本重加权模块解决文本到运动映射的歧义性，实现基于真实图像和文本描述的高质量视频生成，且用户可额外控制运动强度。

**[Local Action-Guided Motion Diffusion Model For Text-To-Motion Generation](image_generation/local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)**

:   提出 GuidedMotion，以局部动作作为细粒度控制信号引导全局运动扩散生成，通过语义图解析和图注意力网络估计引导权重，支持连续可调的运动控制，在生成复杂多动作运动时优势显著。

**[M2D2M Multi-Motion Generation From Text With Discrete Diffusion Models](image_generation/m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)**

:   提出 M2D2M，基于离散扩散模型生成多段连续人体运动序列，通过动态转移概率和两阶段采样策略（TPS）实现动作间平滑过渡，且无需额外的多运动训练数据。

**[Macdiff Unified Skeleton Modeling With Masked Conditional Diffusion](image_generation/macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)**

:   首次将扩散模型用于骨架表征学习，提出 Masked Conditional Diffusion（MacDiff）框架，通过语义编码器提取掩码骨架的表征来引导条件扩散解码器进行去噪，统一了骨架的判别式和生成式建模。

**[Magdiff Multi-Alignment Diffusion For High-Fidelity Video Generation And Editing](image_generation/magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)**

:   提出首个统一视频生成与编辑的多对齐扩散模型 MagDiff，通过主体驱动对齐、自适应提示对齐和高保真对齐三种策略，在单一无微调框架中同时实现高质量视频生成与编辑。

**[Magiceraser Erasing Any Objects Via Semantics-Aware Control](image_generation/magiceraser_erasing_any_objects_via_semantics-aware_control.md)**

:   提出基于扩散模型的对象擦除框架 MagicEraser，通过内容初始化、提示调优和语义感知注意力重聚焦三阶段设计，无需用户输入文本即可高质量擦除对象并生成和谐背景。

**[Memory-Efficient Fine-Tuning For Quantized Diffusion Model](image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)**

:   提出 TuneQDM，首个面向量化扩散模型的内存高效微调方法，通过多通道量化缩放更新和时间步感知缩放策略，在 4-bit 量化模型上实现了接近全精度模型的个性化生成效果。

**[Mixdq Memory-Efficient Few-Step Text-To-Image Diffusion Models With Metric-Decou](image_generation/mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)**

:   针对少步扩散模型量化的特殊挑战，提出 MixDQ 混合精度量化方法，通过 BOS 感知量化处理文本嵌入中的异常值、度量解耦的敏感性分析分离质量与内容影响，在 1-step SDXL-turbo 上实现 W4A8 无损量化。

**[Mixdq Memoryefficient Fewstep Texttoimage Diffusion Models W](image_generation/mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)**

:   针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。

**[Mofa-Video Controllable Image Animation Via Generative Motion Field Adaptions In](image_generation/mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)**

:   提出 MOFA-Video，通过设计多个领域感知的运动场适配器（MOFA-Adapter），在冻结的 Stable Video Diffusion 上实现多域可控图像动画，支持手绘轨迹、人脸关键点等多种控制信号及其零样本组合。

**[Motionchain Conversational Motion Controllers Via Multimodal](image_generation/motionchain_conversational_motion_controllers_via_multimodal.md)**

:   MotionChain构建视觉-运动语言模型，通过VQ-VAE将动作token化支持多轮对话运动生成。

**[Motionchain Conversational Motion Controllers Via Multimodal Prompts](image_generation/motionchain_conversational_motion_controllers_via_multimodal_prompts.md)**

:   提出 MotionChain，一个视觉-运动-语言统一模型，通过多模态提示在多轮对话中生成连续、长期的人体运动序列，支持文本、图像和运动的联合理解与生成。

**[Motionlcm Real-Time Controllable Motion Generation Via Latent Consistency Model](image_generation/motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)**

:   提出 MotionLCM，首次将一致性蒸馏引入人体运动生成领域，在运动潜在空间中实现单步/少步推理的实时运动生成（~30ms/序列），并通过 Motion ControlNet 实现潜在空间中的实时可控运动生成。

**[Mutual Learning For Acoustic Matching And Dereverberation Via Visual Scene-Drive](image_generation/mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)**

:   提出 MVSD，一个基于扩散模型的互学习框架，将视觉声学匹配（VAM）和去混响作为对称互逆任务联合训练，利用两者的互惠关系克服配对数据稀缺问题，并首次将扩散模型用于视觉引导的混响风格迁移。

**[Neusdfusion A Spatial-Aware Generative Model For 3D Shape Completion Reconstruct](image_generation/neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)**

:   提出 NeuSDFusion，一个基于混合三平面 SDF 表示（NeuSDF）和空间感知 Transformer 自编码器的 3D 形状生成框架，通过保持三平面间的空间对应关系，在无条件生成、多模态形状补全、单视图重建和文本到 3D 生成等任务上达到 SOTA 性能。

**[Nl2Contact Natural Language Guided 3D Hand-Object Contact Modeling With Diffusio](image_generation/nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)**

:   提出 NL2Contact，首次利用自然语言描述来可控地建模 3D 手-物体接触图，通过分阶段扩散模型从文本生成手势姿态和接触区域，并构建了首个带有细粒度语言描述的手-物体接触数据集 ContactDescribe。

**[Omg Occlusion-Friendly Personalized Multi-Concept Generation In Diffusion Models](image_generation/omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)**

:   提出 OMG，一种遮挡友好的个性化多概念图像生成框架，通过两阶段采样（布局生成 + 概念噪声融合）实现强身份保持和自然光照协调，且可即插即用地搭配各种单概念模型（LoRA、InstantID）无需额外训练。

**[Omnissr Zero-Shot Omnidirectional Image Super-Resolution Using Stable Diffusion ](image_generation/omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)**

:   提出 OmniSSR，首个基于扩散模型的零样本全向图像超分方法，通过十八面切线投影信息交互（OTII）和梯度分解（GD）校正技术，利用 Stable Diffusion 的图像先验实现保真度和真实感的平衡，无需任何训练或微调。

**[Pixel-Aware Stable Diffusion For Realistic Image Super-Resolution And Personaliz](image_generation/pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)**

:   提出 PASD（Pixel-Aware Stable Diffusion），通过像素感知交叉注意力（PACA）模块使扩散模型在像素级感知图像局部结构，结合退化去除模块和可调噪声调度，实现了真实图像超分辨率和个性化风格化的统一框架，只需替换底座模型即可切换风格。

**[Pixelaware Stable Diffusion For Realistic Image Superre](image_generation/pixelaware_stable_diffusion_for_realistic_image_superre.md)**

:   提出像素感知稳定扩散（PASD）网络，通过像素感知交叉注意力（PACA）在潜空间中实现像素级结构保持，配合退化移除模块和可调噪声调度，统一解决真实图像超分辨率和个性化风格迁移两大任务。

**[Ponymation Learning Articulated 3D Animal Motions From ](image_generation/ponymation_learning_articulated_3d_animal_motions_from_.md)**

:   提出从原始、无标签的互联网视频中学习动物 3D 关节运动生成模型的方法，核心是一个视频光几何自编码框架，将训练视频分解为静止姿态形状、关节姿态序列和纹理，实现无需姿态标注的 3D 运动 VAE 学习。

**[Ponymation Learning Articulated 3D Animal Motions From Unlabeled Online Videos](image_generation/ponymation_learning_articulated_3d_animal_motions_from_unlabeled_online_videos.md)**

:   提出 Ponymation，首次从未标注的网络视频中学习铰接式 3D 动物运动的生成模型，无需姿态标注或参数化形状模板，通过视频光度-几何自编码框架和运动 VAE，能在数秒内从单张图像生成逼真的 4D 动画。

**[Powerful And Flexible Personalized Text-To-Image Generation Via Reinforcement Le](image_generation/powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)**

:   提出基于确定性策略梯度（DPG）的强化学习框架用于个性化文本到图像生成，通过"前瞻"机制和DINO奖励函数捕获长期视觉一致性，大幅提升生成图像的视觉保真度。

**[Powerful And Flexible Personalized Texttoimage Generation Vi](image_generation/powerful_and_flexible_personalized_texttoimage_generation_vi.md)**

:   将个性化T2I建模为DPG框架，引入Q函数和向前看机制捕获长期视觉一致性。

**[Realistic Human Motion Generation With Cross-Diffusion Models](image_generation/realistic_human_motion_generation_with_cross-diffusion_models.md)**

:   提出 CrossDiff 框架，通过统一编码和交叉解码机制融合 3D 与 2D 运动信息，利用交叉扩散实现更精细的全身运动细节捕获，并支持从野外 2D 数据学习 3D 运动生成。

**[Rejection Sampling Imle Designing Priors For Better Few-Shot Image Synthesis](image_generation/rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)**

:   揭示 IMLE 方法中训练/测试时潜在码分布不对齐问题，提出 RS-IMLE 通过拒绝采样改变训练先验分布，在九个少样本图像数据集上平均降低 45.9% FID。

**[Removing Distributional Discrepancies In Captions Improves I](image_generation/removing_distributional_discrepancies_in_captions_improves_i.md)**

:   发现训练图文对齐模型时正负caption之间存在被忽视的数据集级别分布偏差（如GPT生成负样本时倾向用elephant替换giraffe），提出用纯文本分类器过滤高置信样本来消除偏差，结合替换型+交换型两类负样本微调LLaVA-1.5，在Winoground、SeeTRUE等多个基准上大幅超越现有方法。

**[Removing Distributional Discrepancies In Captions Improves Image-Text Alignment](image_generation/removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)**

:   发现正负描述文本在数据集层面存在分布偏差（如词频差异），提出用纯文本分类器过滤偏差数据，微调 LLaVA-1.5 获得 SOTA 图文对齐评分模型 LLaVA-score。

**[Renoise Real Image Inversion Through Iterative Noising](image_generation/renoise_real_image_inversion_through_iterative_noising.md)**

:   提出 ReNoise 迭代重噪方法改进扩散模型的图像反演质量，通过在每个反演步骤多次应用 UNet 并平均预测来提升轨迹估计精度，尤其适用于少步扩散模型（SDXL Turbo、LCM）。

**[Ringid Rethinking Tree-Ring Watermarking For Enhanced Multi-Key Identification](image_generation/ringid_rethinking_tree-ring_watermarking_for_enhanced_multi-key_identification.md)**

:   本文深入分析了 Tree-Ring 水印方法的鲁棒性来源（发现分布偏移是其验证任务中意外的隐藏助力），揭示其在多密钥识别任务中的严重缺陷，并提出 RingID——一种多通道异构水印框架，通过离散化、无损嵌入和更圆环形设计，将2048个密钥的识别准确率从0.07提升至0.82。

**[Robust-Wide Robust Watermarking Against Instruction-Driven Image Editing](image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)**

:   本文提出 Robust-Wide，首个针对指令驱动图像编辑的鲁棒水印方法，核心创新是部分指令驱动去噪采样引导（PIDSG）模块——在训练中将编辑过程的最后k步梯度打通，迫使水印嵌入到语义感知区域，实现编辑后仅约2.6% 的64位水印误码率。

**[Sair Learning Semantic-Aware Implicit Representation](image_generation/sair_learning_semantic-aware_implicit_representation.md)**

:   本文提出语义感知隐式表示（SAIR），通过构建语义隐式表示（SIR）和外观隐式表示（AIR）两个模块，将CLIP提取的文本对齐语义嵌入融入隐式神经函数，使其在大面积缺失区域的图像修复任务中远超仅依赖外观信息的方法，在CelebAHQ上 PSNR 提升1.65-2.69dB。

**[Scalable Group Choreography Via Variational Phase Manifold Learning](image_generation/scalable_group_choreography_via_variational_phase_manifold_learning.md)**

:   本文提出 PDVAE（Phase-conditioned Dance VAE），一种基于相位参数的变分生成模型用于可扩展群舞生成——通过在频域学习舞蹈运动的相位流形（幅度、频率、偏移、相移），实现对**任意数量**舞者的高质量群舞生成，且内存消耗恒定不变，在AIOZ-GDance和AIST-M数据集上全面超越现有方法。

**[Scaledreamer Scalable Text-To-3D Synthesis With Asynchronous Score Distillation](image_generation/scaledreamer_scalable_text-to-3d_synthesis_with_asynchronous_score_distillation.md)**

:   本文提出异步分数蒸馏（ASD），通过将扩散时间步前移（而非微调扩散模型）来降低噪声预测误差、对齐渲染图像分布，解决了VSD微调破坏文本理解能力的问题，实现了稳定训练且可扩展至10万条文本提示的prompt-amortized 3D生成器训练。

**[Scaledreamer Scalable Textto3D Synthesis With Asynchronous S](image_generation/scaledreamer_scalable_textto3d_synthesis_with_asynchronous_s.md)**

:   提出异步分数蒸馏(ASD)，通过将扩散时间步前移（而非微调扩散模型）来减小噪声预测误差，实现稳定的3D生成器训练并可扩展到100K文本提示，保持扩散模型的文本理解能力不受损。

**[Shapefusion A 3D Diffusion Model For Localized Shape Editing](image_generation/shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)**

:   提出一种基于掩码扩散训练策略的3D网格局部编辑方法ShapeFusion，通过在顶点空间直接操作实现完全局部化、可解释的3D形状编辑，无需潜在空间优化。

**[Shedding More Light On Robust Classifiers Under The Lens Of Energy-Based Models](image_generation/shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)**

:   通过将鲁棒判别分类器重新解释为基于能量的模型（EBM），揭示了对抗训练的能量动态规律，提出了基于能量加权的对抗训练方法WEAT，并展示了鲁棒分类器隐含的生成能力。

**[Smoodi Stylized Motion Diffusion Model](image_generation/smoodi_stylized_motion_diffusion_model.md)**

:   提出SMooDi——首个将预训练文本-动作模型适配为风格化动作生成的扩散模型，通过风格适配器和双重风格引导（无分类器引导+基于分类器引导）实现内容文本与风格动作序列驱动的多样化风格动作生成。

**[Soft Prompt Generation For Domain Generalization](image_generation/soft_prompt_generation_for_domain_generalization.md)**

:   提出 SPG（Soft Prompt Generation），首次将生成模型引入 VLM 的 prompt learning，通过 CGAN 从图像动态生成实例特定的软提示，将域知识存储在生成模型中而非提示向量中，实现更好的领域泛化性能。

**[Source Prompt Disentangled Inversion For Boosting Image Editability With Diffusi](image_generation/source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)**

:   提出SPDInv——一种源提示解耦反演方法，通过将反演过程建模为不动点搜索问题并利用预训练扩散模型求解，使反演噪声码与源提示解耦，显著提升基于文本驱动的图像编辑质量。

**[Text2Place Affordance-Aware Text Guided Human Placement](image_generation/text2place_affordance-aware_text_guided_human_placement.md)**

:   提出Text2Place——首个通过文本引导实现真实感人物放置的方法，利用SDS损失优化基于高斯blob的语义掩码学习场景可供性（affordance），再通过主体条件修复实现身份保持的人物放置。

**[Text2Place Affordanceaware Text Guided Human Placement](image_generation/text2place_affordanceaware_text_guided_human_placement.md)**

:   提出 Text2Place，通过 SDS 损失优化 Gaussian blob 参数化的语义掩码学习场景中的人体 affordance，再结合主体条件修复实现逼真的文本引导人物放置，无需大规模训练。

**[Textdiffuser-2 Unleashing The Power Of Language Models For Text Rendering](image_generation/textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)**

:   利用两个语言模型分别进行布局规划和布局编码，实现更灵活、更多样化的视觉文本渲染，在文本准确性和风格多样性之间取得更好的平衡。

**[Textdiffuser2 Unleashing The Power Of Language Models F](image_generation/textdiffuser2_unleashing_the_power_of_language_models_f.md)**

:   TextDiffuser-2 利用两个语言模型（一个用于布局规划、一个用于布局编码）实现灵活自动的文本渲染，克服了现有方法在灵活性、布局能力和样式多样性方面的局限。

**[The Fabrication Of Reality And Fantasy Scene Generation With Llm-Assisted Prompt](image_generation/the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)**

:   提出 Realistic-Fantasy Benchmark (RFBench) 评估扩散模型在创意/知识密集型 prompt 上的表现，并设计 training-free 的 RFNet 框架，通过 LLM 辅助 prompt 解读和语义对齐评估模块来增强扩散模型对抽象和想象性概念的生成能力。

**[Toward Tiny And High-Quality Facial Makeup With Data Amplify Learning](image_generation/toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)**

:   提出 Data Amplify Learning (DAL) 学习范式，用 Diffusion-based Data Amplifier 从仅 5 张标注图像"放大"生成大量配对训练数据，训练出仅 80K 参数的 TinyBeauty 模型，在 iPhone 13 上以 460fps 实现 SOTA 妆容迁移效果。

**[Towards Reliable Advertising Image Generation Using Human Fe](image_generation/towards_reliable_advertising_image_generation_using_human_fe.md)**

:   针对电商广告图像生成中大量不可用图像（空间不匹配、尺寸不匹配、不显著、形状幻觉）的问题，构建了百万级RF1M数据集训练多模态检测网络RFNet，并提出基于RFNet反馈微调扩散模型的RFFT方法（含Consistent Condition正则化），将可用率从约50%提升至接近100%且不损失美观性。

**[Towards Reliable Advertising Image Generation Using Human Feedback](image_generation/towards_reliable_advertising_image_generation_using_human_feedback.md)**

:   构建百万级人工标注广告图像数据集 RF1M，提出多模态 RFNet 自动检测生成图像的可用性，并设计 Consistent Condition 正则化驱动的 RFFT 微调方法，将广告图像可用率从 56.4% 提升至 85.5%。

**[Unveiling Advanced Frequency Disentanglement Paradigm For Low-Light Image Enhanc](image_generation/unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)**

:   提出一种通用的频率解耦学习范式，通过轻量级 ACCA 模块进行粗调低频恢复，再通过 LDRM 模块结合低频一致性约束实现高频细化，仅增加 88K 参数即可为六种 SOTA 低光增强模型带来最高 7.68dB 的 PSNR 提升。

**[Vfusion3D Learning Scalable 3D Generative Models From Video Diffusion Models](image_generation/vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)**

:   提出利用预训练视频扩散模型（EMU Video）作为多视图数据引擎，通过微调使其生成3D一致的多视图视频，从而构建约300万合成数据训练前馈式3D生成模型VFusion3D，实现从单张图片秒级生成3D资产，用户偏好率超过90%。

**[Videoshop Localized Semantic Video Editing With Noise-Extrapolated Diffusion Inv](image_generation/videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)**

:   提出Videoshop——一种免训练的局部语义视频编辑方法，用户可通过任意图像编辑工具修改视频首帧，系统基于噪声外推扩散反演和隐变量归一化技术，自动将编辑传播到所有帧，同时保持语义、空间和时序一致性，在10个指标上超越6个基线方法。

**[Webrpg Automatic Web Rendering Parameters Generation For Visual Presentation](image_generation/webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)**

:   提出Web渲染参数生成（WebRPG）新任务，旨在根据HTML代码自动生成网页元素的视觉呈现参数（布局、文本样式、颜色），通过VAE压缩渲染参数和定制HTML嵌入捕获语义层次信息，建立自回归和扩散两种基线模型，其中自回归模型显著优于扩散模型和GPT-4。

**[Wildvidfit Video Virtual Try-On In The Wild Via Image-Based Controlled Diffusion](image_generation/wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)**

:   WildVidFit 提出了一个无需视频训练的虚拟试穿框架，利用基于图像的条件扩散模型和扩散引导模块（VideoMAE + DINO-V2），实现了在野外复杂视频中保持时序一致性的服装试穿效果。

**[Xpsr Cross-Modal Priors For Diffusion-Based Image Super-Resolution](image_generation/xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)**

:   XPSR 利用多模态大语言模型（MLLM）提取高层和低层语义先验，通过 Semantic-Fusion Attention 和 Degradation-Free Constraint 引导扩散模型实现高保真、高真实感的图像超分辨率。

**[Xpsr Crossmodal Priors For Diffusionbased Image Superresolut](image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)**

:   XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。

**[Zigma A Dit-Style Zigzag Mamba Diffusion Model](image_generation/zigma_a_dit-style_zigzag_mamba_diffusion_model.md)**

:   ZigMa 提出了一种 DiT 风格的 Zigzag Mamba 扩散模型，通过异构逐层锯齿形扫描方案保持空间连续性，以零参数/显存开销实现优于 Mamba 基线的生成质量，同时相比 Transformer 具备线性复杂度优势。

**[Ziplora Any Subject In Any Style By Effectively Merging Loras](image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)**

:   ZipLoRA 提出了一种廉价高效的 LoRA 合并方法，通过学习逐列合并系数并最小化列间余弦相似度，实现了将独立训练的主题 LoRA 和风格 LoRA 无超参数合并，在扩散模型中生成"任意主题 × 任意风格"的个性化图像。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[A Multimodal Benchmark Dataset And Model For Crop Disease Di](multimodal_vlm/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[Adashield Safeguarding Multimodal Large Language Models From Structure-Based Att](multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)**

:   AdaShield通过在MLLM输入前添加防御提示(defense prompt)来防御结构化越狱攻击（图像中嵌入有害文本），提出静态手动提示和自适应自动精化框架两种方案，无需微调模型即可显著提升安全性且不损害正常能力。

**[Addressclip Empowering Vision-Language Models For City-Wide Image Address Locali](multimodal_vlm/addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)**

:   提出 AddressCLIP 框架，通过图像-文本对齐（地址+场景描述的对比学习）和图像-地理匹配（基于GPS距离的流形学习）两大核心组件，将图像地址定位（IAL）问题建模为端到端的视觉-语言对齐任务，在自建的三个IAL数据集上取得最高85.92%的Top-1准确率。

**[Addressclip Empowering Visionlanguage Models For Citywide Im](multimodal_vlm/addressclip_empowering_visionlanguage_models_for_citywide_im.md)**

:   AddressCLIP 定义了"图像地址定位"(IAL) 新任务，提出端到端框架通过图像-文本对齐（图像↔地址/场景描述的对比学习）和图像-地理匹配（流形学习约束特征空间距离与地理距离一致）直接预测图像拍摄的可读文本地址，在自建的 Pittsburgh 和 San Francisco 数据集上优于现有 VLM 迁移方法。

**[Attention Prompting On Image For Large Vision-Language Models](multimodal_vlm/attention_prompting_on_image_for_large_vision-language_models.md)**

:   本文提出Attention Prompting on Image（API），通过辅助模型（如CLIP或LLaVA）根据文本查询生成注意力热力图，将热力图叠加到原始图像上作为视觉提示输入LVLM，在不修改模型参数的情况下在MM-Vet、LLaVA-Bench等多个VL基准上稳定提升多种LVLM的性能（LLaVA-1.5提升3.8%/2.9%）。

**[Attention Prompting On Image For Large Visionlanguage Models](multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)**

:   API用辅助VLM根据文本查询生成注意力热力图叠加原图，引导LVLM关注相关区域。

**[Bad Students Make Great Teachers Active Learning Accelerates Large-Scale Visual ](multimodal_vlm/bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_.md)**

:   提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。

**[Beaf Observing Before-After Changes To Evaluate Hallucination In Vision-Language](multimodal_vlm/beaf_observing_before-after_changes_to_evaluate_hallucination_in_vision-language.md)**

:   提出 BEAF 幻觉评估基准，通过图像编辑（移除物体）构造"前后对比"场景，设计 TU/IG/SB/ID 四个变化感知指标，揭示现有 VLM 即使传统 accuracy 高也可能存在严重幻觉。

**[Beaf Observing Beforeafter Changes To Evaluate Hallucination](multimodal_vlm/beaf_observing_beforeafter_changes_to_evaluate_hallucination.md)**

:   BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。

**[Blink Multimodal Large Language Models Can See But Not Perceive](multimodal_vlm/blink_multimodal_large_language_models_can_see_but_not_perceive.md)**

:   提出BLINK——一个包含14个经典计算机视觉感知任务的多模态评测基准（3807道选择题），这些任务人类可以"眨眼间"解决（95.7%准确率），但最强的GPT-4V仅达51.26%（仅高于随机猜测13.17%），揭示了当前MLLM在核心视觉感知能力上的严重缺失。

**[Brave Broadening The Visual Encoding Of Vision-Language Models](multimodal_vlm/brave_broadening_the_visual_encoding_of_vision-language_models.md)**

:   本文系统性地分析了不同视觉编码器（CLIP、DINOv2、EVA-CLIP等）对VLM性能的影响，发现没有单一编码器能在所有任务上最优，基于此提出BRAVE方法，通过轻量级的MEQ-Former将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在captioning和VQA任务上取得SOTA，并显著降低视觉幻觉。

**[Brave Broadening The Visual Encoding Of Visionlanguage Model](multimodal_vlm/brave_broadening_the_visual_encoding_of_visionlanguage_model.md)**

:   通过系统benchmarking发现没有单一视觉编码器在所有VLM任务上最优，提出BRAVE方法用Multi-Encoder Querying Transformer（MEQ-Former）将多个冻结编码器的特征融合为紧凑表示，以仅116M可训练参数在多个captioning和VQA基准上达到SOTA。

**[Cat Enhancing Multimodal Large Language Model To Answer Questions In Dynamic Aud](multimodal_vlm/cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)**

:   提出 CAT 模型，通过设计问题相关线索聚合器（Clue Aggregator）捕获细粒度音视频特征，结合混合多模态训练策略和 AI 辅助的模糊感知直接偏好优化（ADPO）策略，显著提升 MLLM 在动态音视频场景中的问答准确性，在多个 AVQA 基准上达到 SOTA。

**[Clap Isolating Content From Style Through Contrastive Learni](multimodal_vlm/clap_isolating_content_from_style_through_contrastive_learni.md)**

:   从因果生成模型视角出发，提出CLAP（Contrastive Learning with Augmented Prompts），通过文本增强（而非图像增强）在预训练CLIP的特征空间中解耦内容与风格信息，以极低训练成本（<1小时）显著提升CLIP在零样本/少样本分类和对抗鲁棒性上的表现。

**[Clap Isolating Content From Style Through Contrastive Learning With Augmented Pr](multimodal_vlm/clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)**

:   从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。

**[Dataset Growth](multimodal_vlm/dataset_growth.md)**

:   提出 InfoGrowth，一种高效的在线数据清洗与选择算法，通过近邻搜索估计每个样本的信息增益，实现数据集的持续增长，同时保证清洁度和多样性，在 CC3M 上仅用 1/6 数据即超过全量训练效果。

**[Decoupling Common And Unique Representations For Multimodal ](multimodal_vlm/decoupling_common_and_unique_representations_for_multimodal_.md)**

:   DeCUR将嵌入维度分为跨模态公共和模态独特维度进行多模态自监督学习。

**[Decoupling Common And Unique Representations For Multimodal Self-Supervised Lear](multimodal_vlm/decoupling_common_and_unique_representations_for_multimodal_self-supervised_lear.md)**

:   提出 DeCUR，在多模态自监督学习中将嵌入维度显式拆分为跨模态共有 (common) 和模态独有 (unique) 两部分，通过互相关矩阵分别驱动对齐与去相关，同时引入模态内训练保证独有维度学到有意义信息，在 SAR-光学、RGB-DEM、RGB-Depth 三类多模态场景上均优于 Barlow Twins / CLIP 等基线。

**[Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt L](multimodal_vlm/elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)**

:   提出 SpLIP，一种基于冻结 CLIP 的双向多模态提示学习框架，通过视觉-文本编码器间的双向知识交换、自适应 margin 的三元组损失和条件跨模态拼图任务，在 ZS-SBIR、GZS-SBIR 和 FG-ZS-SBIR 三种草图检索设定下均取得 SOTA。

**[Elevating All Zeroshot Sketchbased Image Retrieval Through M](multimodal_vlm/elevating_all_zeroshot_sketchbased_image_retrieval_through_m.md)**

:   SpLIP提出双向prompt共享用于零样本sketch检索，配合自适应margin和跨模态拼图任务。

**[Eyes Closed Safety On Protecting Multimodal Llms Via Image-To-Text Transformatio](multimodal_vlm/eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)**

:   提出ECSO（Eyes Closed, Safety On），一种无需训练的MLLM保护方法，通过检测自身响应的安全性，并将不安全查询中的图像自适应转换为文本描述，从而恢复预对齐LLM的内在安全机制，在MM-SafetyBench上实现最高71.3%的安全性提升，且不损害常规性能。

**[Eyes Closed Safety On Protecting Multimodal Llms Via Imageto](multimodal_vlm/eyes_closed_safety_on_protecting_multimodal_llms_via_imageto.md)**

:   发现MLLM虽易受图像输入的越狱攻击但具备内省能力（能检测自身不安全回复）、且去除图像后安全机制恢复，据此提出ECSO——通过自检不安全回复后将图像转为query-aware文本描述来恢复预对齐LLM的固有安全机制，无需额外训练即可大幅提升安全性。

**[Flexattention For Efficient High-Resolution Vision-Language Models](multimodal_vlm/flexattention_for_efficient_high-resolution_vision-language_models.md)**

:   提出 FlexAttention，通过基于注意力图的高分辨率token动态选择和层次化自注意力融合机制，在保持甚至超越现有高分辨率VLM性能的同时，将计算成本降低近40%。

**[Flexattention For Efficient Highresolution Visionlanguage Mo](multimodal_vlm/flexattention_for_efficient_highresolution_visionlanguage_mo.md)**

:   FlexAttention动态选择约10%高分辨率token进行层次自注意力，计算成本降40%且性能超越。

**[Freemotion Mocap-Free Human Motion Synthesis With Multimodal Large Language Mode](multimodal_vlm/freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)**

:   首次在**完全不使用动捕数据**的情况下，利用 MLLM（GPT-4V）作为关键帧设计师和动画师，结合基于物理的运动跟踪，实现开放集人体运动合成。

**[Freemotion Mocapfree Human Motion Synthesis With Multimodal ](multimodal_vlm/freemotion_mocapfree_human_motion_synthesis_with_multimodal_.md)**

:   FreeMotion首次在不使用任何动捕数据的情况下，利用GPT-4V作为关键帧设计师和动画师，将自然语言指令分解为关键帧序列，再通过插值和基于物理的运动跟踪填充帧间运动，实现了开放集人体动作合成。

**[Genixer Empowering Multimodal Large Language Model As A Powe](multimodal_vlm/genixer_empowering_multimodal_large_language_model_as_a_powe.md)**

:   Genixer提出一套完整的视觉指令微调数据生成pipeline，通过训练现有MLLM（LLaVA1.5和Shikra）使其具备数据生成能力，无需GPT-4即可生成高质量的VQA和REC指令数据，并通过Fuyu驱动和CLIP驱动的自动过滤框架保证数据质量。

**[Genixer Empowering Multimodal Large Language Model As A Powerful Data Generator](multimodal_vlm/genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)**

:   提出 Genixer 数据生成流水线，训练 MLLM 自身作为数据生成器，无需依赖 GPT-4V 即可自动生成高质量视觉指令微调数据，生成的 915K VQA 数据和 350K REC 数据分别提升 LLaVA1.5 和 Shikra 在多个基准上的表现。

**[Groma Localized Visual Tokenization For Grounding Multimodal](multimodal_vlm/groma_localized_visual_tokenization_for_grounding_multimodal.md)**

:   提出Groma，通过在视觉tokenizer中引入区域提议和区域编码机制，将定位能力嵌入图像token化过程，实现统一的referring和grounding能力，在标准基准上超越同类MLLM。

**[Groma Localized Visual Tokenization For Grounding Multimodal Large Language Mode](multimodal_vlm/groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)**

:   Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

**[Grounding Language Models For Visual Entity Recognition](multimodal_vlm/grounding_language_models_for_visual_entity_recognition.md)**

:   提出 AutoVER，在多模态大语言模型中统一集成对比检索和前缀树约束解码，将 600 万级 Wikipedia 实体空间先缩小到数百候选再做受限生成，在 Oven-Wiki 上将 entity seen 准确率从 PaLI-17B 的 30.6% 翻倍到 61.5%，同时在 unseen/query split 上也大幅领先。

**[Ivtp Instruction-Guided Visual Token Pruning For Large Vision-Language Models](multimodal_vlm/ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)**

:   IVTP提出在大型视觉语言模型的推理过程中，利用文本指令（instruction）信息动态评估各视觉token的重要性并剪枝冗余token，实现与任务相关的自适应视觉信息压缩，在大幅减少计算量的同时保持甚至提升模型性能。

**[Loa-Trans Enhancing Visual Grounding By Location-Aware Transformers](multimodal_vlm/loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)**

:   LoA-Trans提出一种位置感知的查询选择机制，生成多个可能的目标位置作为位置感知查询（而非仅依赖估计的中心点），并引入TaskSyn网络在解码器中实现指代表达理解（REC）和指代表达分割（RES）的任务协同，显著提升视觉定位的准确性。

**[M Ampmaposs A Benchmark To Evaluate Tool-Use For Multi-Step Multi-Modal Tasks](multimodal_vlm/m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)**

:   提出 m&m's 基准，包含 4K+ 多步骤多模态任务和 33 个可执行工具，系统评估 10 个 LLM 在不同规划策略（多步 vs 逐步）、计划格式（JSON vs 代码）和反馈类型（解析/验证/执行）下的工具使用能力，发现多步JSON规划配合反馈是当前最优设计。

**[Marvelovd Marrying Object Recognition And Vision-Language Models For Robust Open](multimodal_vlm/marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)**

:   提出 MarvelOVD 框架，通过将检测器的上下文感知能力和背景识别能力融入 VLM 的伪标签生成与训练流程，在线净化噪声伪标签并自适应重加权训练框，在 COCO 和 LVIS 上大幅超越已有方法。

**[Marvelovd Marrying Object Recognition And Visionlanguage Mod](multimodal_vlm/marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)**

:   分析了VLM（CLIP）在局部区域预测中产生噪声伪标签的两大根因——缺乏上下文信息和无"背景"概念，提出MarvelOVD结合检测器的上下文和背景感知能力进行在线伪标签挖掘，配合自适应提案重加权和分层标签分配，在COCO和LVIS上显著超越SOTA。

**[Mathverse Does Your Multi-Modal Llm Truly See The Diagrams In Visual Math Proble](multimodal_vlm/mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)**

:   提出MathVerse——一个包含2612道视觉数学题目（转化为6个版本共15K测试样本）的多模态数学推理评测基准，通过系统性地调控文本与图像中的信息分配来检验MLLM是否真正"看懂"了数学图表，并提出CoT评估策略进行细粒度推理过程评分，揭示了大多数MLLM严重依赖文本而非视觉图表进行数学推理。

**[Mathverse Does Your Multimodal Llm Truly See The Diagrams In](multimodal_vlm/mathverse_does_your_multimodal_llm_truly_see_the_diagrams_in.md)**

:   提出MathVerse——一个专门评估MLLM视觉数学推理能力的基准，通过将每道题转化为6个版本（从文本主导到纯视觉），揭示大多数MLLM严重依赖文本提示而非真正理解数学图表，并提出CoT评估策略进行细粒度推理过程评分。

**[Merlin Empowering Multimodal Llms With Foresight Minds](multimodal_vlm/merlin_empowering_multimodal_llms_with_foresight_minds.md)**

:   提出 Foresight Pre-Training (FPT) 和 Foresight Instruction-Tuning (FIT) 两阶段训练范式，通过轨迹建模赋予多模态大语言模型"前瞻性思维"能力，使模型能够基于当前观察预测未来事件并进行推理。

**[Meta-Prompting For Automating Zero-Shot Visual Recognition With Llms](multimodal_vlm/meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。

**[Metaprompting For Automating Zeroshot Visual Recognitio](multimodal_vlm/metaprompting_for_automating_zeroshot_visual_recognitio.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段元提示策略自动让 LLM 生成任务特定且类别特定的 VLM 提示，在 20 个数据集上将 CLIP 零样本识别提升最高 19.8%，完全消除人工提示设计。

**[Mm1 Methods Analysis And Insights From Multimodal Llm Pre-Training](multimodal_vlm/mm1_methods_analysis_and_insights_from_multimodal_llm_pre-training.md)**

:   Apple 系统性地消融了 MLLM 构建的三大轴（架构、数据、训练），得出关键设计准则：图像分辨率 > 模型大小 > 训练数据；VL 连接器类型影响甚微；caption/interleaved/text-only 三类数据的精细混合至关重要，最终构建了 3B-30B dense 和最高 64B MoE 的 MM1 模型族，在 few-shot 预训练评测上达到 SOTA。

**[Mmbench Is Your Multi-Modal Model An All-Around Player](multimodal_vlm/mmbench_is_your_multi-modal_model_an_all-around_player.md)**

:   提出 MMBench——一个包含 3217 道多选题、覆盖 20 个细粒度能力维度的双语（英/中）视觉语言模型评测基准，并设计了 CircularEval 循环评测策略和基于 LLM 的选项提取机制，显著提升了评测的鲁棒性和公平性。

**[Mmbench Is Your Multimodal Model An Allaround Player](multimodal_vlm/mmbench_is_your_multimodal_model_an_allaround_player.md)**

:   提出MMBench——一个系统设计的双语多模态评测基准，包含3000+多选题覆盖20个能力维度，并引入CircularEval策略和LLM辅助选项匹配，实现对VLM的鲁棒、细粒度评估。

**[Myvlm Personalizing Vlms For User-Specific Queries](multimodal_vlm/myvlm_personalizing_vlms_for_user-specific_queries.md)**

:   提出MyVLM，通过外部概念识别头（concept head）和可学习的概念嵌入向量（concept embedding），在不修改VLM原始权重的情况下实现个性化视觉语言交互——仅需3-5张图片即可让VLM识别并描述用户特定概念（如"你的狗"、"你的朋友"），在BLIP-2和LLaVA上均取得了显著的个性化效果。

**[Myvlm Personalizing Vlms For Userspecific Queries](multimodal_vlm/myvlm_personalizing_vlms_for_userspecific_queries.md)**

:   MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。

**[Navgpt-2 Unleashing Navigational Reasoning Capability For Large Vision-Language ](multimodal_vlm/navgpt-2_unleashing_navigational_reasoning_capability_for_large_vision-language_.md)**

:   NavGPT-2通过将冻结LLM的隐层表征作为视觉-语言特征输入拓扑图导航策略网络，在保留LLM可解释性导航推理能力的同时，消除了基于LM的智能体与VLN专用模型之间的性能差距，并展现出优异的数据效率。

**[Navgpt2 Unleashing Navigational Reasoning Capability](multimodal_vlm/navgpt2_unleashing_navigational_reasoning_capability.md)**

:   提出 NavGPT-2，通过将冻结 LLM 与视觉内容对齐，结合拓扑图导航策略网络，在保持 LLM 可解释性推理能力的同时，消除了基于语言模型的导航智能体与 VLN 专用模型之间的性能差距。

**[Omniview-Tuning Boosting Viewpoint Invariance Of Vision-Language Pre-Training Mo](multimodal_vlm/omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)**

:   OVT通过构建460万多视角图文数据集MVCap和设计minimax优化的跨视角对齐框架，以参数高效微调方式显著提升VLP模型（如CLIP）对3D视角变化的鲁棒性（平均+9-10%），同时几乎不损失原始性能。

**[Omniviewtuning Boosting Viewpoint Invariance Of Visionlangua](multimodal_vlm/omniviewtuning_boosting_viewpoint_invariance_of_visionlangua.md)**

:   OVT构建400万+MVCap数据集+Cross-Viewpoint Alignment提升VLP视角不变性。

**[Quantized Prompt For Efficient Generalization Of Vision-Language Models](multimodal_vlm/quantized_prompt_for_efficient_generalization_of_vision-language_models.md)**

:   将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。

**[Quantized Prompt For Efficient Generalization Of Visionlangu](multimodal_vlm/quantized_prompt_for_efficient_generalization_of_visionlangu.md)**

:   发现适度噪声可以抑制VLM prompt tuning中的过拟合和灾难性遗忘，首次将量化误差视为正则化，设计了基于K-Means聚类的量化感知训练算法，在11个数据集上以极小存储开销（0.26KB）超越了众多SOTA方法。

**[Revision Rendering Tools Enable Spatial Fidelity In Vision-Language Models](multimodal_vlm/revision_rendering_tools_enable_spatial_fidelity_in_vision-language_models.md)**

:   提出 REVISION 框架，利用 Blender 3D 渲染生成空间关系精确的合成图像，以免训练方式引导 T2I 模型生成空间一致的图像，并构建 RevQA 基准评估 MLLM 的空间推理能力。

**[Robust Calibration Of Large Vision-Language Adapters](multimodal_vlm/robust_calibration_of_large_vision-language_adapters.md)**

:   本文发现CLIP适配方法（Adapter/Prompt Learning/TTA）在OOD场景下严重损害了零样本基线的校准能力，揭示logit范围增大（而非logit范数增大）是误校准的根本原因，并提出三种简单且模型无关的logit范围约束方案（ZS-Norm、Penalty、SaLS），有效缓解误校准同时保持判别性能。

**[Robust Calibration Of Large Visionlanguage Adapters](multimodal_vlm/robust_calibration_of_large_visionlanguage_adapters.md)**

:   CLIP适配方法OOD校准退化的根因是logit范围增大，提出SaLS等方案。

**[Select And Distill Selective Dual-Teacher Knowledge Transfer For Continual Learn](multimodal_vlm/select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)**

:   提出选择性双教师知识迁移框架（SND），通过衡量预训练VLM和最近微调VLM之间的特征差异，在无标签参考数据集上自适应选择合适的教师进行知识蒸馏，同时缓解灾难性遗忘并保持零样本分类能力。

**[Self-Adapting Large Visual-Language Models To Edge Devices Across Visual Modalit](multimodal_vlm/self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)**

:   提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。

**[Sharegpt4V Improving Large Multi-Modal Models With Better Captions](multimodal_vlm/sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)**

:   ShareGPT4V 构建了一个120万条高质量描述性caption数据集（由GPT4-Vision生成100K种子 + Share-Captioner扩展至1.2M），通过在预训练和SFT两阶段使用该数据集训练LLaVA架构的模型ShareGPT4V-7B，在11个多模态benchmark中9个取得最优，证明了高质量caption是LMM模态对齐的关键瓶颈。

**[Sharegpt4V Improving Large Multimodal Models With Better Cap](multimodal_vlm/sharegpt4v_improving_large_multimodal_models_with_better_cap.md)**

:   指出现有LMM训练中低质量caption是模态对齐的瓶颈，构建了1.2M高质量详细描述的ShareGPT4V数据集（100K来自GPT4-Vision + 1.2M来自训练得到的Share-Captioner），在预训练和SFT两阶段使用该数据，以简单架构的7B模型在11个基准中9个取得最优。

**[Sq-Llava Self-Questioning For Large Vision-Language Assistant](multimodal_vlm/sq-llava_self-questioning_for_large_vision-language_assistant.md)**

:   提出视觉自提问（Visual Self-Questioning）训练范式，让 LLM 不仅学习回答问题，还学习根据图像主动提问，通过充分利用指令数据中问题本身的丰富语义信息来增强视觉-语言对齐。

**[Sqllava Selfquestioning For Large Visionlanguage Assistant](multimodal_vlm/sqllava_selfquestioning_for_large_visionlanguage_assistant.md)**

:   提出SQ-LLaVA，首次将指令数据中问题作为额外学习目标，训练MLLM不仅回答问题还学会"自问"，通过视觉自提问（visual self-questioning）任务挖掘指令数据中被忽视的问题上下文信息，配合原型提取器和LoRA微调，在10个VQA基准中9个超越基线。

**[The Hard Positive Truth About Vision-Language Compositionality](multimodal_vlm/the_hard_positive_truth_about_vision-language_compositionality.md)**

:   本文揭示了现有CLIP硬负例微调方法在提升组合性理解时引入了"过敏感"问题——模型将语义不变的硬正例（hard positives）也错误地判为不匹配；通过同时引入硬正例和硬负例进行微调，显著缓解了该问题并实现了更鲁棒的组合性提升。

**[The Hard Positive Truth About Visionlanguage Compositionalit](multimodal_vlm/the_hard_positive_truth_about_visionlanguage_compositionalit.md)**

:   本文揭示了现有CLIP组合性基准的评估盲区——缺少hard positives测试，发现hard negative微调会导致模型"过敏"（对语义保持的改写也错误地降低匹配分数），并通过同时加入hard positives和hard negatives训练来缓解这一问题。

**[Towards Open-Ended Visual Quality Comparison](multimodal_vlm/towards_open-ended_visual_quality_comparison.md)**

:   本文提出 Co-Instruct，首个面向开放式视觉质量比较的大型多模态模型，通过从两种"弱监督源"（LLM合并的单图描述 + GPT-4V伪标签）构建562K指令微调数据集，实现比 GPT-4V（其教师模型）更高的多图质量比较准确率，并提出首个多图比较基准 MICBench。

**[Towards Openended Visual Quality Comparison](multimodal_vlm/towards_openended_visual_quality_comparison.md)**

:   提出 Co-Instruct，首个开源的开放式视觉质量比较大模型，通过构建 Co-Instruct-562K 数据集和 MICBench 基准，使 LMM 在视觉质量比较任务上超越 GPT-4V。

**[Towards Real-World Adverse Weather Image Restoration Enhancing Clearness And Sem](multimodal_vlm/towards_real-world_adverse_weather_image_restoration_enhancing_clearness_and_sem.md)**

:   本文提出WResVLM半监督学习框架，利用视觉-语言模型（VLM）为真实恶劣天气图像提供清晰度评估和语义描述监督信号，通过VLM图像评估+天气提示学习增强清晰度、描述辅助的语义正则化增强语义，在真实去雨/去雾/去雪任务上全面超越现有方法。

**[Towards Realworld Adverse Weather Image Restoration Enhancin](multimodal_vlm/towards_realworld_adverse_weather_image_restoration_enhancin.md)**

:   提出WResVLM半监督框架，利用VLM评估图像清晰度和提供语义信息，通过伪标签选择+天气prompt学习增强清晰度、VLM描述引导的语义正则化增强语义，首次有效地将合成数据训练的复原模型泛化到真实恶劣天气场景。

**[Umbrae Unified Multimodal Brain Decoding](multimodal_vlm/umbrae_unified_multimodal_brain_decoding.md)**

:   提出UMBRAE，通过通用脑编码器将fMRI信号与图像特征对齐后送入冻结的MLLM，实现多模态脑解码（描述、定位、检索、视觉重建），并创新性地引入跨被试训练策略，使单一模型服务多个被试且优于单被试模型。

**[Uni3Dl A Unified Model For 3D Vision-Language Understanding](multimodal_vlm/uni3dl_a_unified_model_for_3d_vision-language_understanding.md)**

:   提出 Uni3DL，一个直接在点云上操作的统一 3D 视觉-语言模型，通过 Query Transformer 学习任务无关的语义/掩码输出，再由 Task Router 组合多个功能头实现语义分割、实例分割、目标检测、视觉定位、3D 描述生成、文本-3D 检索等六大任务，性能达到或超过各任务专用 SOTA。

**[Unicode Learning A Unified Codebook For Multimodal Large Lan](multimodal_vlm/unicode_learning_a_unified_codebook_for_multimodal_large_lan.md)**

:   提出UniCode，通过语言驱动的迭代训练范式学习一个统一码本，使LLM的词表可同时量化视觉和文本信号，无需额外对齐模块即可实现多模态理解与生成，并引入上下文图像解压缩任务提升生成质量。

**[Unicode Learning A Unified Codebook For Multimodal Large Language Models](multimodal_vlm/unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)**

:   UniCode提出学习一个统一的codebook来同时tokenize视觉和文本信号，通过language-driven iterative training范式将视觉tokenizer的码本与LLM的词表渐进对齐，并引入in-context image decompression预训练任务提升图像生成质量，使MLLM无需额外对齐模块即可实现多模态理解与生成。

**[Vary Scaling Up The Vision Vocabulary For Large Vision-Language Model](multimodal_vlm/vary_scaling_up_the_vision_vocabulary_for_large_vision-language_model.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表来扩展 LVLM 的视觉感知能力，使模型在保持通用能力的同时获得文档 OCR、图表理解等细粒度视觉感知能力。

**[Vary Scaling Up The Vision Vocabulary For Large Visionlanguag](multimodal_vlm/vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)**

:   提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

**[X-Former Unifying Contrastive And Reconstruction Learning For Mllms](multimodal_vlm/x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP-ViT（对比学习）和MAE-ViT（掩码图像建模）的互补视觉特征，在仅使用1/10数据量的情况下显著超越BLIP-2在细粒度视觉理解任务上的表现。

**[Xformer Unifying Contrastive And Reconstruction Learning For](multimodal_vlm/xformer_unifying_contrastive_and_reconstruction_learning_for.md)**

:   提出X-Former，一个轻量级Transformer模块，通过双交叉注意力机制融合CLIP（全局语义）和MAE（局部细节）两种视觉编码器的互补特征，结合ITC/ITM/ITG和重建四个损失联合优化，提升MLLM的细粒度视觉理解能力。

**[Zero-Shot Object Counting With Good Exemplars](multimodal_vlm/zero-shot_object_counting_with_good_exemplars.md)**

:   提出VA-Count框架，通过样本增强模块（EEM）利用Grounding DINO发现高质量正负样本，结合噪声抑制模块（NSM）用对比学习区分正负密度图，实现零样本目标计数在FSC-147和CARPK上的SOTA表现。

---

## ✂️ 语义分割 { #segmentation }

**[A Semantic Space Is Worth 256 Language Descriptions Make Str](segmentation/a_semantic_space_is_worth_256_language_descriptions_make_str.md)**

:   ProLab 用 LLM 生成类别的常识性描述，通过句子嵌入和 K-Means 聚类将其压缩为 256 个可解释的描述性属性，构建属性级多热标签空间替代传统 one-hot 类别标签来监督分割模型，在五个经典基准上一致超越类别级监督且涌现出域外泛化能力。

**[A Simple Latent Diffusion Approach For Panoptic Segmentation And Mask Inpainting](segmentation/a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)**

:   基于Stable Diffusion构建了一个极简的潜在扩散分割框架LDMSeg，通过浅层自编码器将分割mask压缩到潜空间、再训练图像条件扩散模型来生成全景分割结果，避免了传统方法中的目标检测模块、匈牙利匹配和复杂后处理，并天然支持mask inpainting和多任务扩展。

**[Actionvos Actions As Prompts For Video Object Segmentation](segmentation/actionvos_actions_as_prompts_for_video_object_segmentation.md)**

:   提出ActionVOS——一种以人类动作叙述作为额外语言提示的Referring Video Object Segmentation新设定，通过无参数的动作感知标注模块生成伪标签，并设计动作引导的focal loss来抑制假阳性，在VISOR上将非活跃物体的误分割降低35.6% mIoU，同时在VOST/VSCOS上对状态变化物体的分割提升3.0% mIoU。

**[Active Coarsetofine Segmentation Of Moveable Parts From Real](segmentation/active_coarsetofine_segmentation_of_moveable_parts_from_real.md)**

:   提出首个面向真实室内场景RGB图像中可运动部件实例分割的主动学习框架，通过姿态感知masked attention网络实现由粗到细的分割，仅需人工标注11.45%的图像即可获得全量验证的高质量分割结果，相比最优非AL方法节省60%人工时间。

**[Adalog Post-Training Quantization For Vision Transformers With Adaptive Logarith](segmentation/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)**

:   提出自适应对数底量化器AdaLog，通过可搜索的对数底替代固定log₂/log√2量化器来处理ViT中post-Softmax和post-GELU激活的幂律分布，并设计快速渐进组合搜索(FPCS)策略高效确定量化超参，在极低比特(3/4-bit)下显著优于现有ViT PTQ方法。

**[Brushnet A Plug-And-Play Image Inpainting Model With Decomposed Dual-Branch Diff](segmentation/brushnet_a_plug-and-play_image_inpainting_model_with_decomposed_dual-branch_diff.md)**

:   提出 BrushNet，一种即插即用的双分支扩散模型图像修复架构，通过将遮罩图像特征提取与图像生成解耦到独立分支，实现逐层像素级特征注入，在图像质量、遮罩区域保持和文本对齐三方面全面超越已有方法。

**[Cola Conditional Dropout And Language-Driven Robust Dual-Modal Salient Object De](segmentation/cola_conditional_dropout_and_language-driven_robust_dual-modal_salient_object_de.md)**

:   提出 CoLA 框架，通过语言驱动的质量评估（LQA）和条件性 Dropout（CD）两个核心模块，首次在双模态显著性目标检测中同时解决噪声输入和模态缺失两大鲁棒性问题。

**[Colormae Exploring Data-Independent Masking Strategies In Masked Autoencoders](segmentation/colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)**

:   提出 ColorMAE，通过对随机噪声施加不同频域滤波器生成具有空间与语义先验的数据无关遮罩模式，在不增加任何参数和计算开销的前提下，显著提升 MAE 的下游任务表现，尤其在语义分割任务上相比随机遮罩提升 2.72 mIoU。

**[Controlnet Improving Conditional Controls With Efficien](segmentation/controlnet_improving_conditional_controls_with_efficien.md)**

:   提出 ControlNet++，通过预训练判别模型提取生成图像的条件并优化像素级循环一致性损失来显式提升可控生成的精度，同时提出高效单步去噪奖励策略避免多步采样的巨大开销。

**[Controlnet Improving Conditional Controls With Efficient Consistency Feedback](segmentation/controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)**

:   提出 ControlNet++，通过像素级循环一致性损失显式优化条件可控生成质量：用预训练判别模型从生成图像中提取条件并与输入条件对齐，并设计高效单步去噪 reward 策略避免多步采样的巨大显存开销，在分割掩码、边缘、深度等多种条件控制下显著提升可控性（如分割 mIoU +11.1%）。

**[Cores Orchestrating The Dance Of Reasoning And Segmentation](segmentation/cores_orchestrating_the_dance_of_reasoning_and_segmentation.md)**

:   提出 CoReS（Chains of Reasoning and Segmenting），一种双链结构的多模态思维链框架，通过推理链和分割链的层次化协作，结合 in-context 引导策略，实现对复杂推理文本中目标物体的渐进式精确分割，在 ReasonSeg 数据集上超越 LISA 6.5%。

**[Cpm Class-Conditional Prompting Machine For Audio-Visual Segmentation](segmentation/cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)**

:   提出 CPM（Class-conditional Prompting Machine），通过结合类无关查询与基于 GMM 采样的类条件查询来增强 Mask2Former 在音视频分割中的二部图匹配稳定性和跨模态注意力效力，同时设计音频条件提示（ACP）、视觉条件提示（VCP）和提示对比学习（PCL）三个辅助任务，在 AVSBench 和 VPO 基准上达到 SOTA。

**[Cs2K Class-Specific And Class-Shared Knowledge Guidance For Incremental Semantic](segmentation/cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)**

:   提出 Cs2K 框架，从类别特有知识（原型引导伪标签 + 原型引导类别适应）和类别共享知识（权重引导选择性整合）两个方面协同缓解增量语义分割中的灾难性遗忘与新类欠拟合问题。

**[Dataset Enhancement With Instance-Level Augmentations](segmentation/dataset_enhancement_with_instance-level_augmentations.md)**

:   提出一种基于预训练扩散模型的实例级数据增强方法，通过在保持原始标注不变的前提下逐个重绘图像中的目标实例，显著提升了显著性目标检测、语义分割和目标检测的性能，同时支持数据匿名化。

**[Deep Nets With Subsampling Layers Unwittingly Discard Useful Activations At Test](segmentation/deep_nets_with_subsampling_layers_unwittingly_discard_useful_activations_at_test.md)**

:   发现深度网络中下采样层在默认前向传播中丢弃了大量有用激活，提出一个搜索+聚合框架在测试时利用这些被丢弃的激活图来提升分类和分割性能，与传统TTA方法正交互补。

**[Densenets Reloaded Paradigm Shift Beyond Resnets And Vits](segmentation/densenets_reloaded_paradigm_shift_beyond_resnets_and_vits.md)**

:   重新审视 DenseNet 的密集拼接连接（concatenation shortcut），通过系统性现代化改造（加宽减深、现代化 block、扩大中间维度、更多 transition 层等），提出 RDNet（Revitalized DenseNet），在 ImageNet-1K 上超越 Swin Transformer、ConvNeXt、DeiT-III，证明了拼接连接作为一种被低估的范式具有强大潜力。

**[Dreamlip Language-Image Pre-Training With Long Captions](segmentation/dreamlip_language-image_pre-training_with_long_captions.md)**

:   通过 MLLM 为 30M 图像生成长文本描述，提出动态子描述采样的多正样本对比学习和子描述特定分组损失，实现细粒度视觉-语言对齐，仅用 30M 数据在检索和语义分割上达到甚至超越 CLIP 400M 的性能。

**[Eaformer Scene Text Segmentation With Edge-Aware Transformers](segmentation/eaformer_scene_text_segmentation_with_edge-aware_transformers.md)**

:   提出边缘感知Transformer（EAFormer），通过文本边缘提取器过滤非文本区域边缘、对称交叉注意力在编码器中融合文本边缘信息，显著提升文字边缘区域的分割精度，并重标注COCO_TS和MLT_S数据集以实现更公平评估。

**[Early Preparation Pays Off New Classifier Pre-Tuning For Class Incremental Seman](segmentation/early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)**

:   提出NeST（New claSsifier pre-Tuning）方法，在正式训练前通过学习从所有旧分类器到新分类器的线性变换来初始化新分类器权重，并设计基于跨任务类别相似性的变换矩阵初始化策略，在Pascal VOC和ADE20K上显著提升多种CISS方法的性能。

**[Efficient And Versatile Robust Fine-Tuning Of Zero-Shot Models](segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)**

:   R-Adapter 通过在 CLIP 模型中插入轻量级 adapter 模块并结合三种自集成策略（Adapter Dropping、权重累积、权重缩放重参数化），在仅微调 13% 参数的前提下同时实现了 ID 高精度和 OOD 强鲁棒性，并首次将鲁棒微调扩展到分类之外的跨模态检索和开放词汇分割任务。

**[Eliminating Feature Ambiguity For Few-Shot Segmentation](segmentation/eliminating_feature_ambiguity_for_few-shot_segmentation.md)**

:   提出AENet插件网络，通过挖掘判别性查询前景区域来消除特征歧义，增强交叉注意力中的前景-前景匹配，可即插即用地提升现有少样本分割方法性能（SCCAN 1-shot在PASCAL-5i上+3.0%）。

**[Exploring Pre-Trained Text-To-Video Diffusion Models For Referring Video Object ](segmentation/exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)**

:   首次探索**预训练文本到视频(T2V)扩散模型**的特征表示用于视频理解任务，提出 VD-IT 框架，通过文本引导图像投影和视频专属噪声预测两个设计，在 R-VOS 任务上实现了优于判别式视频骨干网络的时序一致性和分割性能。

**[Exploring Pretrained Texttovideo Diffusion Models For Referr](segmentation/exploring_pretrained_texttovideo_diffusion_models_for_referr.md)**

:   VD-IT首次探索预训练T2V扩散模型（ModelScopeT2V）在视频理解任务中的应用，通过Text-Guided Image Projection和Video-specific Noise Prediction设计，从固定T2V模型中提取语义对齐、时序一致的视频特征，在Referring VOS任务上超越传统判别式backbone。

**[Frequency-Spatial Entanglement Learning For Camouflaged Object Detection](segmentation/frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)**

:   提出频率-空间纠缠学习（FSEL）框架，通过在频率域和空间域之间进行纠缠学习（entanglement learning），利用全局频率特征弥补空间特征的局部性和敏感性限制，在三个COD基准上超越21个SOTA方法。

**[Frest Feature Restoration For Semantic Segmentation Under Multiple Adverse Condi](segmentation/frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)**

:   提出 FREST，一种面向多种恶劣条件（雾、雨、雪、夜间）的源无关域自适应语义分割框架，通过交替学习条件嵌入空间（分离条件信息）和特征恢复（将恶劣条件特征恢复为正常条件），逐步消除恶劣条件对特征的影响，在 ACDC 和 RobotCar 基准上均达到新的 SOTA。

**[General And Task-Oriented Video Segmentation](segmentation/general_and_task-oriented_video_segmentation.md)**

:   GvSeg 提出了一个通用视频分割框架，通过将分割目标解耦为外观、形状和位置三个因素，并根据任务需求（VIS/VSS/VPS/EVS）动态调整这三个因素在查询初始化、匹配和采样中的参与度，在统一架构下实现了四种视频分割任务的SOTA性能。

**[Git Towards Generalist Vision Transformer Through Universal Language Interface](segmentation/git_towards_generalist_vision_transformer_through_universal_language_interface.md)**

:   提出 GiT 框架，通过通用语言接口将图像描述、目标检测、实例分割、语义分割和视觉定位五大视觉任务统一为自回归序列生成，仅用纯 ViT（无任何任务特定模块）实现多任务联合训练，且任务间互相增强。

**[Learning Camouflaged Object Detection From Noisy Pseudo Label](segmentation/learning_camouflaged_object_detection_from_noisy_pseudo_label.md)**

:   提出首个弱半监督伪装目标检测方法 (WSSCOD)，仅用 20% 像素级标注 + 80% 框标注即可达到全监督 SOTA 的可比性能，核心贡献是一个自适应噪声校正损失 $\mathcal{L}_{NC}$，可在早期学习和记忆化两个阶段分别优化。

**[Learning From The Web Language Drives Weakly-Supervised Incremental Learning For](segmentation/learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)**

:   首次提出完全使用网络图像（而非精心设计的数据集图像）进行弱监督增量语义分割，通过傅里叶域判别器筛选网络图像 + caption 驱动的 rehearsal 策略保持旧类知识，在 PASCAL VOC 15-5 设定下达到 73.4% mIoU。

**[Lift A Surprisingly Simple Lightweight Feature Transform For Dense Vit Descripto](segmentation/lift_a_surprisingly_simple_lightweight_feature_transform_for_dense_vit_descripto.md)**

:   提出 LiFT，一种极其简单的轻量级后处理网络（仅 1.2M 参数），通过自监督多尺度重建目标训练，融合冻结 ViT 的粗粒度语义特征与 CNN 提取的细粒度图像特征，以仅增加 5.7% 参数和 22% FLOPs 的代价将 ViT 特征分辨率翻倍，在关键点匹配、检测、分割和目标发现等密集任务上均获得显著性能提升。

**[Long-Tail Temporal Action Segmentation With Group-Wise Temporal Logit Adjustment](segmentation/long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)**

:   首次系统性地解决时序动作分割中的长尾问题，提出 Group-wise Temporal Logit Adjustment (G-TLA) 框架，利用活动标签进行分组分类并结合动作时序先验进行 logit 调整，在大幅提升尾部类别性能的同时不损失头部类别。

**[Occlusion-Aware Seamless Segmentation](segmentation/occlusion-aware_seamless_segmentation.md)**

:   提出 Occlusion-Aware Seamless Segmentation (OASS) 新任务与 UnmaskFormer 框架，同时解决全景图像窄视场解锁、遮挡物体完整分割和针孔-全景跨域适应三大挑战，在自建 BlendPASS 数据集上达到 SOTA。

**[Olaf A Plug-And-Play Framework For Enhanced Multi-Object Multi-Part Scene Parsin](segmentation/olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)**

:   提出即插即用框架 OLAF，通过将前景/边缘掩码作为额外输入通道、引入低层稠密特征提取模块 LDF 和针对性权重适配策略，在不改变基础架构的前提下为多种分割网络（CNN/U-Net/Transformer）带来显著的多物体多部件分割增益，在最具挑战的 Pascal-Parts-201 上超越 SOTA 达 4.0 mIoU。

**[Openpsg Open-Set Panoptic Scene Graph Generation Via Large Multimodal Models](segmentation/openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)**

:   首次定义开放集全景场景图生成（OpenPSG）任务，利用 BLIP-2 作为多模态关系解码器，结合关系查询 Transformer（RelQ-Former）实现开放集关系预测，在 PSG 数据集 PredCls R@100 达到 79.3%，闭集场景超越先前 SOTA 26.6%。

**[Openpsg Openset Panoptic Scene Graph Generation Via Large Mu](segmentation/openpsg_openset_panoptic_scene_graph_generation_via_large_mu.md)**

:   本文首次提出开放集全景场景图生成任务（OpenPSG），利用大型多模态模型（BLIP-2）以自回归方式预测物体间的开放集关系，通过关系查询Transformer高效提取物体对特征并过滤无关对，在闭集和开放集设置下均取得SOTA。

**[Partstad 2D-To-3D Part Segmentation Task Adaptation](segmentation/partstad_2d-to-3d_part_segmentation_task_adaptation.md)**

:   PartSTAD 提出了一种 2D-to-3D 部件分割的任务适配方法：通过为 GLIP 的 2D 检测框引入可学习权重预测网络（以 3D mRIoU 为目标优化），并集成 SAM 获取精确前景掩码，在 PartNet-Mobility 上实现了语义分割 mIoU 提升 7.0%p、实例分割 mAP50 提升 5.2%p（相对 PartSLIP）。

**[Plain-Det A Plain Multi-Dataset Object Detector](segmentation/plain-det_a_plain_multi-dataset_object_detector.md)**

:   Plain-Det 提出了一个简洁灵活的多数据集目标检测框架，通过语义空间校准、类感知查询组合器和基于难度的动态采样策略，在 COCO 上达到 51.9 mAP（匹配当时 SOTA），并可灵活扩展到新数据集且保持鲁棒性能。

**[Representing Topological Self-Similarity Using Fractal Feature Maps For Accurate](segmentation/representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)**

:   利用分形理论将分形维数（FD）从图像级扩展到像素级，生成分形特征图（FFM）作为深度学习模型的额外输入和损失权重，并设计包含边缘解码器和骨架解码器的多解码器网络（MD-Net），在五个管状结构数据集上显著提升分割性能。

**[Rotary Position Embedding For Vision Transformer](segmentation/rotary_position_embedding_for_vision_transformer.md)**

:   本文系统研究了将 RoPE（Rotary Position Embedding）从1D语言模型扩展到2D视觉任务的方法，提出 RoPE-Mixed（混合可学习频率）替代传统的 Axial 频率分配，在 ViT 和 Swin Transformer 上实现了显著的分辨率外推性能提升，在 ImageNet 分类、COCO 检测和 ADE20k 分割上均带来一致增益。

**[Sclip Rethinking Self-Attention For Dense Vision-Language Inference](segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)**

:   发现 CLIP 的密集预测失败源于自注意力导致的空间位置错位问题，提出 Correlative Self-Attention (CSA) 机制——仅修改最后一层自注意力的计算方式（无需训练），将 CLIP 的零样本语义分割从 14.1% 平均 mIoU 提升至 38.2%，超越所有已有方法。

**[Sclip Rethinking Selfattention For Dense Visionlanguage Infe](segmentation/sclip_rethinking_selfattention_for_dense_visionlanguage_infe.md)**

:   发现CLIP在密集预测中失败的根因是自注意力机制导致的空间位置错配（spatial-invariant features），提出Correlative Self-Attention(CSA)机制——仅用一个投影矩阵计算token间相关性作为注意力分数，无需任何训练/额外参数即可将CLIP的零样本语义分割mIoU从14.1%提升至38.2%（8个基准平均），大幅超越现有SOTA的33.9%。

**[Seggen Supercharging Segmentation Models With Text2Mask And Mask2Img Synthesis](segmentation/seggen_supercharging_segmentation_models_with_text2mask_and_mask2img_synthesis.md)**

:   提出 SegGen 数据生成框架，反转传统"先生图再标注"的流程为"先从文本生成分割掩码，再从掩码生成图像"，打破分割数据合成的"鸡生蛋"瓶颈，在 ADE20K 上将 Mask2Former R50 的 mIoU 从 47.2 提升至 49.9（+2.7）。

**[Segmentation-Guided Layer-Wise Image Vectorization With Gradient Fills](segmentation/segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)**

:   提出分割引导的矢量化框架，通过梯度感知分割子程序引导 Bézier 路径的初始化和优化，首次在保持分层拓扑的逐层矢量化方法中支持径向渐变填充，使矢量图形在更少路径数下达到更高的视觉质量。

**[Seit Masked Token Modeling Improves Storage-Efficient Training](segmentation/seit_masked_token_modeling_improves_storage-efficient_training.md)**

:   在 SeiT 的 token 化训练框架上引入掩码 token 建模（MTM）自监督预训练，并设计 TokenAdapt 和 ColorAdapt 两种 token 专用数据增强策略，在仅 1% 存储空间（1.4GB）下将 ImageNet-1k 分类准确率从 74.0% 提升至 77.8%，有效解决了 token 域数据增强的难题。

**[Self-Supervised Co-Salient Object Detection Via Feature Correspondences At Multi](segmentation/self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)**

:   提出 SCoSPARC——一个两阶段自监督共显著目标检测模型，通过 patch 级和 region 级 ViT 特征对应关系检测图像组中的共显著物体，在 CoCA 数据集上 F-measure 比无监督 SOTA 高 13.7%，甚至超越多个有监督方法。

**[Sos Segment Object System For Open-World Instance Segmentation With Object Prior](segmentation/sos_segment_object_system_for_open-world_instance_segmentation_with_object_prior.md)**

:   提出 SOS 方法，通过用 DINO 自注意力图作为物体先验生成聚焦于物体的 SAM 提示点，从而产出高质量伪标注来训练标准实例分割系统，在 COCO/LVIS/ADE20k 跨类别/跨数据集设置下大幅超越 SOTA，精度提升高达 81.6%。

**[Spin Hierarchical Segmentation With Subpart Granularity In Natural Images](segmentation/spin_hierarchical_segmentation_with_subpart_granularity_in_natural_images.md)**

:   SPIN 构建了首个自然图像子部件（subpart）级层级语义分割数据集 SubPartImageNet——包含 203 个子部件类别和 10.6 万条标注——并提出两个层级一致性评估指标（SpCS / SeCS），在 20+ 现代模型上全面基准测试，揭示了当前模型在子部件层面的严重不足。

**[Udifftext A Unified Framework For High-Quality Text Synthesis In Arbitrary Image](segmentation/udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)**

:   提出 UDiffText，通过设计轻量级字符级文本编码器替换 CLIP encoder、引入基于字符分割图的 local attention loss 和 STR loss 微调 cross-attention 层，并在推理阶段对 noised latent 进行 refinement，实现在任意图像中合成高精度、视觉协调的文本，SeqAcc 全面超越 SOTA。

**[Un-Evimo Unsupervised Event-Based Independent Motion Segmentation](segmentation/un-evimo_unsupervised_event-based_independent_motion_segmentation.md)**

:   首个无需标注的事件相机独立运动物体(IMO)分割框架，利用光流与几何约束生成伪标签训练分割网络，在 EVIMO 数据集上取得与有监督方法可比的性能。

**[Unifs Universal Few-Shot Instance Perception With Point Representations](segmentation/unifs_universal_few-shot_instance_perception_with_point_representations.md)**

:   提出UniFS——首个通用少样本实例感知模型，通过将目标检测、实例分割、姿态估计和目标计数统一为动态点表示学习范式，并引入结构感知点学习(SAPL)损失来捕获点间高阶结构关系，在最小任务假设下达到接近专家模型的性能。

**[Visa Reasoning Video Object Segmentation Via Large Language Models](segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)**

:   提出 ReasonVOS 新任务和 VISA 模型，利用多模态 LLM 的世界知识推理能力实现基于隐式文本查询的视频目标分割与跟踪。

**[Visage Video Instance Segmentation With Appearance-Guided Enhancement](segmentation/visage_video_instance_segmentation_with_appearance-guided_enhancement.md)**

:   针对在线视频实例分割(VIS)中现有方法过度依赖位置信息导致的关联错误，提出VISAGE通过从骨干特征中显式提取外观嵌入、结合对比学习和简化tracker来增强实例关联准确性，在YTVIS和OVIS基准上取得SOTA。

**[You Only Learn One Query Learning Unified Human Query For Single-Stage Multi-Per](segmentation/you_only_learn_one_query_learning_unified_human_query_for_single-stage_multi-per.md)**

:   提出 HQNet 框架，通过学习统一的 Human Query 表示，在单阶段单模型中同时完成行人检测、实例分割、2D 姿态估计、3D Mesh 恢复、属性识别等多种以人为中心的感知任务，并构建了首个全面的多任务人体感知基准 COCO-UniHuman。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[4D Contrastive Superflows Are Dense 3D Representation Learners](autonomous_driving/4d_contrastive_superflows_are_dense_3d_representation_learners.md)**

:   提出SuperFlow框架，通过视图一致性对齐、稠密-稀疏一致性正则化、和基于流的时空对比学习三个模块，利用连续LiDAR-相机对建立4D预训练目标，在11个异构LiDAR数据集上全面超越了之前的Image-to-LiDAR预训练方法。

**[Accelerating Online Mapping And Behavior Prediction Via Dire](autonomous_driving/accelerating_online_mapping_and_behavior_prediction_via_dire.md)**

:   提出直接将在线地图估计模型内部的BEV特征暴露给下游轨迹预测模型（而非仅传递解码后的矢量化地图），通过三种BEV特征注入策略实现推理加速最高73%、预测精度提升最高29%。

**[Adaptive Human Trajectory Prediction Via Latent Corridors](autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)**

:   将prompt tuning思想引入行人轨迹预测，通过在预训练轨迹预测器的输入端添加可学习的低秩图像prompt（称为latent corridors），以不到0.1%的额外参数实现对部署场景特定行为模式的高效自适应，在合成和真实数据上分别取得最高23.9%和26.8%的ADE提升。

**[Approaching Outside Scaling Unsupervised 3D Object Detection From 2D Scene](autonomous_driving/approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)**

:   提出 LiSe 方法，将 2D 图像信息引入无监督 3D 目标检测，通过自步学习（self-paced learning）中的自适应采样和弱模型聚合策略，大幅提升远距离和小目标的检测能力。

**[Carformer Self-Driving With Learned Object-Centric Representations](autonomous_driving/carformer_self-driving_with_learned_object-centric_representations.md)**

:   提出 CarFormer，首次将自监督 slot attention 学到的 object-centric 表征用于自动驾驶，在 CARLA Longest6 基准上超越了使用精确物体属性的 PlanT，同时具备世界模型预测未来状态的能力。

**[Dvlo Deep Visual-Lidar Odometry With Local-To-Global Feature Fusion And Bi-Direc](autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)**

:   提出基于聚类的 Local-to-Global 融合网络 DVLO，通过双向结构对齐（图像→伪点云 + 点云→伪图像）解决视觉与 LiDAR 的数据结构不一致问题，在 KITTI 里程计和 FlyingThings3D 场景流任务上均取得 SOTA。

**[Dvlo Deep Visuallidar Odometry With Localtoglobal Featu](autonomous_driving/dvlo_deep_visuallidar_odometry_with_localtoglobal_featu.md)**

:   提出 DVLO——基于从局部到全局融合 + 双向结构对齐的视觉-LiDAR 里程计网络，通过将图像视为伪点云进行局部聚类融合、将点云投影为伪图像进行全局自适应融合，解决了两种模态间固有的数据结构不一致问题。

**[Enhancing Vectorized Map Perception With Historical Rasterized Maps](autonomous_driving/enhancing_vectorized_map_perception_with_historical_rasterized_maps.md)**

:   提出 HRMapNet，通过维护一张低成本的全局历史栅格化地图（historical rasterized map），为在线矢量化地图感知提供互补先验信息，在 BEV 特征聚合和 query 初始化两个层面增强现有方法，在 nuScenes 和 Argoverse 2 上取得显著提升。

**[Equivariant Spatio-Temporal Self-Supervision For Lidar Object Detection](autonomous_driving/equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)**

:   E-SSL3D 提出一种时空联合等变自监督预训练框架，通过空间等变（对旋转用分类目标、对平移/缩放/翻转用对比目标）和时间等变（用 3D 场景流约束相邻帧特征变换一致性）联合训练 3D 特征编码器，在低数据场景下仅用 20% 标注数据就能达到接近 100% 数据从头训练的检测性能。

**[Fsd-Bev Foreground Self-Distillation For Multi-View 3D Object Detection](autonomous_driving/fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)**

:   提出前景自蒸馏（FSD）框架，在同一模型内构建教师-学生分支共享图像特征，避免跨模态蒸馏中的分布差异问题，配合点云增强和多尺度前景增强模块，在 nuScenes 上取得 SOTA 性能。

**[Fully Sparse 3D Occupancy Prediction](autonomous_driving/fully_sparse_3d_occupancy_prediction.md)**

:   提出 SparseOcc，首个完全稀疏的 3D 占用预测网络，通过稀疏体素解码器和掩码引导的 Mask Transformer 实现高效占用预测，并设计了 RayIoU 评价指标解决传统 mIoU 的深度方向不一致惩罚问题。

**[Gaussianformer Scene As Gaussians For Vision-Based 3D Semantic Occupancy Predict](autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)**

:   提出以物体为中心的 3D 语义高斯表示替代传统密集体素，用一组稀疏的 3D 语义高斯描述场景并通过高斯到体素的 splatting 生成占用预测，在性能可比的情况下将内存消耗降低 75%-82%。

**[Graphbev Towards Robust Bev Feature Alignment For Multi-Modal 3D Object Detectio](autonomous_driving/graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)**

:   针对多模态BEV融合中LiDAR与相机标定误差导致的特征错位问题，提出GraphBEV框架，通过LocalAlign（基于KD-Tree的邻域深度图匹配）和GlobalAlign（可学习偏移量全局对齐）两个模块，在nuScenes上达到70.1% mAP（超BEVFusion 1.6%），在噪声错位场景下超BEVFusion 8.3%。

**[Hierarchical Temporal Context Learning For Camera-Based Semantic Scene Completio](autonomous_driving/hierarchical_temporal_context_learning_for_camera-based_semantic_scene_completio.md)**

:   针对相机语义场景补全（SSC）中时序信息利用粗糙的问题，提出层级式时序上下文学习（HTCL）范式：先通过跨帧模式亲和度（CPA）度量当前帧与历史帧的细粒度对应关系，再通过基于亲和度的动态精炼（ADR）自适应采样补偿不完整观测，在SemanticKITTI上排名第1，甚至在OpenOccupancy上mIoU超过LiDAR方法。

**[Ittakestwo Leveraging Peer Representations For Semi-Supervised Lidar Semantic Se](autonomous_driving/ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)**

:   提出IT2框架，通过利用LiDAR数据的对等表示（range image + voxel grid）之间的一致性学习作为新型扰动形式，并引入基于高斯混合模型的跨分布对比学习，大幅提升半监督LiDAR语义分割性能。

**[Lidar-Event Stereo Fusion With Hallucinations](autonomous_driving/lidar-event_stereo_fusion_with_hallucinations.md)**

:   提出将LiDAR稀疏深度点与事件立体相机融合的首个框架，通过在事件堆叠表示（VSH）或原始事件流（BTH）中"幻觉"（插入虚构事件）来弥补事件相机在无运动/无纹理区域的信息缺失，大幅提升事件立体匹配精度。

**[Lidarevent Stereo Fusion With Hallucinations](autonomous_driving/lidarevent_stereo_fusion_with_hallucinations.md)**

:   首次探索 LiDAR 与事件立体相机的融合，提出虚拟堆叠幻觉（VSH）和回溯时间幻觉（BTH）两种策略，通过在事件流/堆叠中注入虚拟事件来增强匹配可辨别性，大幅提升事件立体匹配精度。

**[Livehps Robust And Coherent Motion Capture In Dynamic Free Environment](autonomous_driving/livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)**

:   提出 LiveHPS++，一种基于单 LiDAR 的鲁棒人体动作捕捉方法，通过轨迹引导身体追踪器、噪声不敏感速度预测器和运动学感知姿态优化器三个模块，隐式和显式建模人体运动的动力学和运动学特征，在复杂噪声环境下实现精确且连贯的全局人体运动捕捉。

**[Mapdistill Boosting Efficient Camera-Based Hd Map Construction Via Camera-Lidar ](autonomous_driving/mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)**

:   首次将知识蒸馏引入 HD 地图构建任务，提出 MapDistill 框架，通过双 BEV 变换模块、跨模态关系蒸馏、双层特征蒸馏和 Map Head 蒸馏，将相机-LiDAR 融合教师模型的知识迁移至轻量纯相机学生模型，在 nuScenes 上实现 **+7.7 mAP** 或 **4.5倍加速**。

**[Maptracker Tracking With Strided Memory Fusion For Consistent Vector Hd Mapping](autonomous_driving/maptracker_tracking_with_strided_memory_fusion_for_consistent_vector_hd_mapping.md)**

:   将在线向量高精地图构建重新定义为追踪任务，通过双表示（BEV栅格+道路元素向量）的步进式记忆缓冲区融合机制实现时间一致的高精地图重建，在nuScenes和Argoverse2上分别以76.1和76.9 mAP大幅超越现有方法。

**[Monocular Occupancy Prediction For Scalable Indoor Scenes](autonomous_driving/monocular_occupancy_prediction_for_scalable_indoor_scenes.md)**

:   提出 ISO（Indoor Scene Occupancy）方法，通过预训练深度模型和 D-FLoSP（双特征视线投影）模块实现室内场景的单目 3D 占用预测，并构建了规模比 NYUv2 大 40 倍的 Occ-ScanNet 基准数据集。

**[Monowad Weather-Adaptive Diffusion Model For Robust Monocular 3D Object Detectio](autonomous_driving/monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)**

:   提出 MonoWAD，通过天气码本学习晴天知识作为参考，结合天气自适应扩散模型将雾效建模为噪声进行特征增强，实现在各种天气条件下鲁棒的单目3D目标检测。

**[Navigation Instruction Generation With Bev](autonomous_driving/navigation_instruction_generation_with_bev.md)**

:   提出 BEVInstructor，将鸟瞰图 (BEV) 特征融入多模态大语言模型 (MLLM) 用于导航指令生成，通过 Perspective-BEV 视觉编码、参数高效 prompt tuning 和实例引导的迭代精化，在室内外多个数据集上全面超越 SOTA。

**[Navigation Instruction Generation With Bev Perception And Large Language Models](autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)**

:   提出 BEVInstructor，将鸟瞰图 (BEV) 特征融合到多模态大语言模型中，通过 Perspective-BEV 融合编码器、参数高效的 Prompt Tuning 以及实例引导的迭代优化策略，在室内外导航指令生成任务上取得 SOTA。

**[Neuroncap Photorealistic Closed-Loop Safety Testing For Autonomous Driving](autonomous_driving/neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)**

:   提出 NeuroNCAP，一个基于 NeRF 渲染的真实感闭环自动驾驶安全测试框架，受 Euro NCAP 碰撞避免协议启发设计三类安全关键场景（静止/正面/侧面碰撞），揭示当前 SOTA 端到端规划器（UniAD、VAD）在闭环安全场景中严重失败——碰撞率高达 88-92%——尽管其感知模块准确运行。

**[Occgen Generative Multi-Modal 3D Occupancy Prediction For Autonomous Driving](autonomous_driving/occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)**

:   OccGen 将 3D 语义占用预测重新定义为"noise-to-occupancy"的生成式范式，通过条件编码器提取多模态特征、渐进式精炼解码器执行扩散去噪，以由粗到精的方式逐步生成占用图，在 nuScenes-Occupancy 上多模态/纯LiDAR/纯相机设置下分别相对提升 9.5%/6.3%/13.3% 的 mIoU。

**[Occgen Generative Multimodal 3D Occupancy Prediction For Aut](autonomous_driving/occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)**

:   提出OccGen，首次将扩散模型的"噪声到占据"生成范式引入3D语义占据预测任务，通过条件编码器+渐进式精炼解码器实现由粗到精的占据图生成，在nuScenes-Occupancy上多模态/纯LiDAR/纯相机设置下分别提升mIoU 9.5%/6.3%/13.3%。

**[Occworld Learning A 3D Occupancy World Model For Autonomous Driving](autonomous_driving/occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)**

:   OccWorld 提出在 3D 占用空间中学习世界模型，用 VQ-VAE 对 3D occupancy 进行 token 化，再通过 GPT 风格的时空生成 Transformer 自回归预测未来场景演化和自车轨迹，在 nuScenes 上无需实例和地图标注即可实现有竞争力的规划性能。

**[Open Object-Wise Position Embedding For Multi-View 3D Object Detection](autonomous_driving/open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)**

:   提出 OPEN，通过目标级深度编码器（ODE）从像素级深度先验中预测物体中心深度，并设计目标级位置编码（OPE）将该信息注入 Transformer 解码器，生成 3D 目标感知特征，在 nuScenes 上达到 64.4% NDS 的 SOTA 性能。

**[Optimizing Diffusion Models For Joint Trajectory Prediction And Controllable Gen](autonomous_driving/optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)**

:   本文提出 Optimal Gaussian Diffusion (OGD) 和 Estimated Clean Manifold (ECM) Guidance 两项技术，分别通过优化扩散先验分布和在干净流形上直接注入引导梯度，将联合轨迹预测的扩散步数减少到原来的 1/12，引导采样步数减少到 1/5，同时在 Argoverse 2 上取得更优性能。

**[Panovos Bridging Non-Panoramic And Panoramic Views With Transformer For Video Se](autonomous_driving/panovos_bridging_non-panoramic_and_panoramic_views_with_transformer_for_video_se.md)**

:   提出首个全景视频目标分割数据集 PanoVOS（150个视频、19K实例标注），揭示现有 VOS 模型无法处理全景视频的像素不连续和严重畸变问题，并设计 PSCFormer 利用全景空间一致性注意力解决左右边界连续性问题。

**[Progressive Pretext Task Learning For Human Trajectory Prediction](autonomous_driving/progressive_pretext_task_learning_for_human_trajectory_prediction.md)**

:   提出渐进式前置任务学习框架 PPT，通过三阶段训练（逐步下一位置预测 → 目的地预测 → 完整轨迹预测）逐步增强模型对短期动态和长期依赖的捕获能力，配合高效的两步非自回归 Transformer 预测器，在多个行人轨迹预测基准上取得 SOTA。

**[Random Walk On Pixel Manifolds For Anomaly Segmentation Of Complex Driving Scene](autonomous_driving/random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)**

:   提出 Random Walk on Pixel Manifolds (RWPM)，利用随机游走捕获像素嵌入的流形结构来修正因驾驶场景多样性导致的流形畸变，从而提升异常分割评分函数的准确性，无需额外训练即可即插即用地集成到现有异常分割框架中。

**[Rapid-Seg Range-Aware Pointwise Distance Distribution Networks For 3D Lidar Segm](autonomous_driving/rapid-seg_range-aware_pointwise_distance_distribution_networks_for_3d_lidar_segm.md)**

:   本文提出 RAPiD 特征（Range-Aware Pointwise Distance Distribution），一种对刚体变换不变且适应点密度变化的 LiDAR 点云局部几何特征，配合双层嵌套自编码器和通道注意力融合，在 SemanticKITTI（76.1 mIoU）和 nuScenes（83.6 mIoU）上达到 SOTA 分割性能。

**[Reason2Drive Towards Interpretable And Chain-Based Reasoning For Autonomous Driv](autonomous_driving/reason2drive_towards_interpretable_and_chain-based_reasoning_for_autonomous_driv.md)**

:   Reason2Drive 构建了一个包含 60 万+视频-文本对的大规模自动驾驶推理数据集，将驾驶决策拆解为感知→预测→推理的链式过程，并提出 ADRScore 评估指标和带 prior tokenizer + instructed vision decoder 的 VLM 框架，显著提升了驾驶场景的链式推理准确性。

**[Reason2Drive Towards Interpretable And Chainbased Reasoning](autonomous_driving/reason2drive_towards_interpretable_and_chainbased_reasoning.md)**

:   构建 Reason2Drive 基准数据集（600K+ 视频-文本对，覆盖感知-预测-推理链式任务），提出 ADRScore 评估链式推理正确性的新指标，并设计 Prior Tokenizer + Instructed Vision Decoder 框架增强 VLM 的目标级感知和推理能力，在自动驾驶推理任务上显著超越所有基线。

**[Reliability In Semantic Segmentation Can We Use Synthetic Data](autonomous_driving/reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)**

:   首次系统地利用 Stable Diffusion 生成合成 OOD 数据来全面评估语义分割模型的可靠性，包括协变量偏移下的鲁棒性评估、OOD 物体检测评估和模型校准，并证明合成数据与真实 OOD 数据的评估结果高度相关。

**[Rethinking Data Augmentation For Robust Lidar Semantic Segmentation In Adverse W](autonomous_driving/rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)**

:   通过数据中心分析识别出恶劣天气对 LiDAR 的两大核心干扰模式（几何扰动和点丢失），提出 Selective Jittering 和 Learnable Point Drop 两种针对性数据增强方法，在 SemanticKITTI→SemanticSTF 基准上将 baseline 提升 8.1 mIoU 达到 SOTA。

**[Rethinking Lidar Domain Generalization Single Source As Multiple Density Domains](autonomous_driving/rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)**

:   提出密度判别特征嵌入（DDFE）模块，利用单一 LiDAR 源域点云中固有的密度多样性（近处密/远处疏），学习密度感知的特征表示，实现对不同传感器配置下未见域的泛化，无需目标域数据。

**[Roofdiffusion Constructing Roofs From Severely Corrupted Point Data Via Diffusio](autonomous_driving/roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)**

:   RoofDiffusion 提出了一种基于条件扩散概率模型的端到端自监督方法，用于从严重稀疏（最高99%缺失）、不完整（80%区域遮挡）且含噪的屋顶高程图中恢复完整干净的高程信息，在自建的 PoznanRD 数据集和 BuildingNet 上显著超越传统插值方法和现有深度补全方法。

**[Safe-Sim Safety-Critical Closed-Loop Traffic Simulation With Diffusion-Cont](autonomous_driving/safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)**

:   Safe-Sim 提出了一个基于扩散模型的闭环安全关键仿真框架，通过在扩散去噪过程中引入对抗项和部分扩散（Partial Diffusion）机制，实现了对抗车辆行为类型（碰撞角度、相对速度、碰撞类型）的细粒度控制，在 nuScenes 和 nuPlan 上验证了对多种 planner 的有效评估能力。

**[Safe-Sim Safety-Critical Closed-Loop Traffic Simulation With Diffusion-Controlla](autonomous_driving/safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-controlla.md)**

:   Safe-Sim 提出了一种基于扩散模型的闭环安全关键仿真框架，通过在去噪过程中注入对抗性引导目标和部分扩散（Partial Diffusion）机制，生成真实且可控的对抗场景来评估自动驾驶规划算法，支持控制碰撞类型、相对速度和 TTC 等关键参数。

**[Seflow A Self-Supervised Scene Flow Method In Autonomous Driving](autonomous_driving/seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)**

:   SeFlow 提出将传统的基于 ray-casting 的动态点分类融入自监督场景流学习管线，通过专门的动态/静态损失函数和基于聚类的物体级运动一致性约束，在 Argoverse 2 和 Waymo 上以实时速度（48ms/帧）取得自监督场景流 SOTA 性能，甚至超越部分有监督方法。

**[Sfpnet Sparse Focal Point Network For Semantic Segmentation On General Lidar Poi](autonomous_driving/sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)**

:   SFPNet 提出稀疏焦点调制（SFPM）替代 window-attention，通过多层级上下文提取和门控自适应聚合来避免针对特定 LiDAR 类型的归纳偏置设计，在机械旋转式、固态和混合固态三种 LiDAR 数据集上均取得领先或竞争性性能，并发布了首个混合固态 LiDAR 语义分割数据集 S.MID。

**[Simpb A Single Model For 2D And 3D Object Detection From Multiple Cameras](autonomous_driving/simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)**

:   提出 SimPB 统一模型，通过混合解码器（multi-view 2D decoder + 3D decoder）以循环 3D→2D→3D 的方式同时完成多相机 2D 检测和 BEV 空间 3D 检测，在 nuScenes 上两项任务均取得优秀结果。

**[Sledge Synthesizing Driving Environments With Generative Models And Rule-Based T](autonomous_driving/sledge_synthesizing_driving_environments_with_generative_models_and_rule-based_t.md)**

:   SLEDGE 提出了首个基于生成模型的驾驶仿真器，通过 Raster-to-Vector 自编码器将驾驶场景编码为栅格化潜在图（RLM），再利用 Diffusion Transformer 生成高质量的车道图和交通参与者，实现了比 nuPlan 少 500 倍存储（<4GB）的仿真环境，同时支持 500m 长路线测试，暴露了 SOTA 规划器 PDM-Closed 超过 40% 的失败率。

**[Stream Query Denoising For Vectorized Hd-Map Construction](autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)**

:   提出 Stream Query Denoising (SQD) 策略，通过对前一帧 GT 添加噪声并训练网络恢复当前帧 GT 来增强流式 HD 地图构建中的时序一致性建模，在 nuScenes 和 Argoverse2 上全面超越 StreamMapNet。

**[Tod3Cap Towards 3D Dense Captioning In Outdoor Scenes](autonomous_driving/tod3cap_towards_3d_dense_captioning_in_outdoor_scenes.md)**

:   首次提出户外 3D 密集描述任务，构建百万级 TOD3Cap 数据集（850 场景 2.3M 描述），设计基于 BEV 特征 + Relation Q-Former + LLaMA-Adapter 的端到端网络，超越适配后的室内方法 +9.6 CIDEr@0.5IoU。

**[Unitraj A Unified Framework For Scalable Vehicle Trajectory Prediction](autonomous_driving/unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)**

:   UniTraj 构建了一个统一多数据集（nuScenes、Argoverse 2、WOMD）、多模型（AutoBot、MTR、Wayformer）和多评估策略的车辆轨迹预测框架，揭示模型跨数据集泛化能力显著下降，但通过扩大数据规模和多样性可大幅提升性能，合并训练在 nuScenes 排行榜达到第 1 名。

**[Visiontrap Vision-Augmented Trajectory Prediction Guided By Textual Descriptions](autonomous_driving/visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)**

:   提出 VisionTrap，将环视相机图像和文本描述引入轨迹预测任务，通过 BEV 视觉语义编码器和文本驱动的去偏对比学习引导模型学习视觉语义线索（如行人姿态、转向灯等），在保持 53ms 实时推理的同时显著提升预测精度并发布 nuScenes-Text 数据集。

**[Visiontrap Visionaugmented Trajectory Prediction Guided](autonomous_driving/visiontrap_visionaugmented_trajectory_prediction_guided.md)**

:   提出 VisionTrap，利用环视相机视觉输入和 VLM/LLM 生成的文本描述作为训练监督，增强自动驾驶场景下的多智能体轨迹预测，同时保持 53ms 实时推理速度。

**[Weakly Supervised 3D Object Detection Via Multi-Level Visual Guidance](autonomous_driving/weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)**

:   提出 VG-W3D 框架，仅使用 2D 标注（无需任何 3D 标签），通过特征级、输出级和训练级三层视觉引导来训练 3D 目标检测器，在 KITTI 上取得了与使用 500 帧 3D 标注方法相当的性能。

---

## 🧑 人体理解 { #human_understanding }

**[3D Hand Pose Estimation In Everyday Egocentric Images](human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)**

:   通过系统研究裁剪输入、相机内参感知位置编码(KPE)、辅助监督(手部分割+抓握标签)和多数据集联合训练这四个实践，提出WildHands系统，在仅用ResNet50和少量数据的条件下，实现了对野外第一人称图像中3D手部姿态的鲁棒估计，零样本泛化超过FrankMocap全部指标且与10倍大的HaMeR竞争。

**[3Dgazenet Generalizing 3D Gaze Estimation With Weak-Supervision From Synthetic V](human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)**

:   提出将视线估计重新表述为密集3D眼球网格回归，并通过从大规模野外人脸图像中自动提取伪标签+HeadGAN合成多视图进行弱监督训练，在跨域场景下比SOTA提升最多30%。

**[A Probabilityguided Sampler For Neural Implicit Surface Rend](human_understanding/a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)**

:   提出一种概率引导的光线采样器（Probability-guided Sampler），在3D图像投影空间中建模概率密度函数来指导光线采样朝向感兴趣区域，同时设计了包含近表面和空白空间两个分量的新型表面重建损失，可作为插件集成到现有神经隐式表面渲染器中，显著提升重建精度和渲染质量。

**[A Simple Baseline For Spoken Language To Sign Language Trans](human_understanding/a_simple_baseline_for_spoken_language_to_sign_language_trans.md)**

:   提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

**[Adadistill Adaptive Knowledge Distillation For Deep Face Rec](human_understanding/adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)**

:   提出AdaDistill，将知识蒸馏概念嵌入margin penalty softmax loss中，通过基于EMA的自适应类中心（早期用sample-sample简单知识、后期用sample-center复杂知识）和困难样本感知机制，无需额外超参数即可提升轻量级人脸识别模型的判别能力，在IJB-B/C和ICCV21-MFR等挑战性基准上超越SOTA蒸馏方法。

**[Aden Adaptive Density Representations For Sparseview Camera](human_understanding/aden_adaptive_density_representations_for_sparseview_camera.md)**

:   ADen提出生成器-判别器框架统一位姿回归和概率位姿估计：生成器输出多个6DoF位姿假设来建模多模态分布（处理对称歧义），判别器选出最佳假设，在稀疏视角位姿估计上同时实现了更高精度和更低运行时间。

**[Alignist Cad-Informed Orientation Distribution Estimation By Fusing Shape And Co](human_understanding/alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)**

:   提出 Alignist，首个利用 CAD 模型信息（SDF + SurfEmb 对应特征）训练隐式分布网络来推断 SO(3) 上姿态分布的方法，通过 product of experts 融合几何和特征对齐，在低数据场景下显著优于对比学习方法。

**[Audio-Driven Talking Face Generation With Stabilized Synchronization Loss](human_understanding/audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)**

:   提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。

**[Bi-Tta Bidirectional Test-Time Adapter For Remote Physiological Measurement](human_understanding/bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)**

:   提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

**[Combining Generative And Geometry Priors For Wide-Angle Portrait Correction](human_understanding/combining_generative_and_geometry_priors_for_wide-angle_portrait_correction.md)**

:   提出结合 StyleGAN 生成式先验（用于人脸矫正）和几何对称先验（用于背景直线矫正）的双模块框架，大幅提升广角人像畸变校正的视觉质量和定量指标。

**[Como Controllable Motion Generation Through Language Guided Pose Code Editing](human_understanding/como_controllable_motion_generation_through_language_guided_pose_code_editing.md)**

:   提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。

**[Decomposed Vector-Quantized Variational Autoencoder For Human Grasp Generation](human_understanding/decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)**

:   提出 Decomposed VQ-VAE (DVQ-VAE)，通过将手部分解为六个部分分别编码到独立码本，并设计双阶段解码策略（先姿态后位置），在四个基准数据集上质量指标相对提升约14.1%。

**[Domain Reduction Strategy For Non-Line-Of-Sight Imaging](human_understanding/domain_reduction_strategy_for_non-line-of-sight_imaging.md)**

:   提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。

**[Egoexo-Fitness Towards Egocentric And Exocentric Full-Body Action Understanding](human_understanding/egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)**

:   提出 EgoExo-Fitness 数据集，包含同步的第一人称和第三人称健身视频，提供两级时间边界标注和创新性的可解释动作评判标注（技术关键点验证、自然语言评论、质量评分），并构建五个基准任务。

**[Evsign Sign Language Recognition And Translation With Streaming Events](human_understanding/evsign_sign_language_recognition_and_translation_with_streaming_events.md)**

:   首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

**[Exemplar-Free Continual Representation Learning Via Learnable Drift Compensation](human_understanding/exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)**

:   提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

**[Facial Affective Behavior Analysis With Instruction Tuning](human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)**

:   提出首个面向面部情感行为分析（FABA）的指令微调数据集 FABA-Instruct、评测基准 FABA-Bench 以及高效 MLLM 架构 EmoLA，通过面部先验专家模块和 LoRA 适配实现了对情绪与 AU 的细粒度描述与识别。

**[Foundpose Unseen Object Pose Estimation With Foundation Features](human_understanding/foundpose_unseen_object_pose_estimation_with_foundation_features.md)**

:   FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。

**[Freemotion A Unified Framework For Number-Free Text-To-Motion Synthesis](human_understanding/freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)**

:   提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。

**[Generalizable Facial Expression Recognition](human_understanding/generalizable_facial_expression_recognition.md)**

:   提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。

**[Graspxl Generating Grasping Motions For Diverse Objects At Scale](human_understanding/graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)**

:   提出 GraspXL，一个基于强化学习的抓取动作生成框架，仅用58个物体训练即可泛化到50万+未见物体，同时支持多运动目标（抓取区域、朝向、手腕旋转、手部位置）控制和多种灵巧手平台。

**[Gs-Pose Category-Level Object Pose Estimation Via Geometric And Semantic Corresp](human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)**

:   提出GS-Pose方法，利用预训练视觉基础模型（DINOv2）的2D语义特征投影到3D空间，结合几何特征通过Transformer匹配网络进行类别级物体9D姿态估计，仅需10个合成3D模型训练即可在多个真实数据集上达到SOTA级别性能。

**[How Video Meetings Change Your Expression](human_understanding/how_video_meetings_change_your_expression.md)**

:   提出 FacET（Facial Explanations through Translations），一种基于生成式域翻译的可解释框架，通过学习解耦的面部空间特征和可解释的时空线性变换，自动发现视频会议（VC）与面对面（F2F）交流之间的细微面部表情差异模式，并支持将 VC 视频转换为 F2F 风格的"去zoom化"。

**[Humos Human Motion Model Conditioned On Body Shape](human_understanding/humos_human_motion_model_conditioned_on_body_shape.md)**

:   提出 HUMOS，一种基于体型条件化的人体运动生成模型，通过循环一致性损失和可微分的直觉物理/动态稳定性约束，在无配对训练数据的情况下学习体型与运动之间的相关性，生成物理可信且动态稳定的人体运动。

**[Improving Point-Based Crowd Counting And Localization Based On Auxiliary Point G](human_understanding/improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)**

:   提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。

**[Interleaving One-Class And Weakly-Supervised Models With Adaptive Thresholding F](human_understanding/interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)**

:   提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。

**[Lapose Laplacian Mixture Shape Modeling For Rgb-Based Category-Level Object Pose](human_understanding/lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)**

:   提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。

**[Large Motion Model For Unified Multi-Modal Motion Generation](human_understanding/large_motion_model_for_unified_multi-modal_motion_generation.md)**

:   提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。

**[Large Motion Model For Unified Multimodal Motion Generation](human_understanding/large_motion_model_for_unified_multimodal_motion_generation.md)**

:   LMM是首个多模态通用人体动作生成模型，统一了文本/动作/音乐/语音等10种任务、16个数据集（320K序列/1亿帧），通过身体部位感知的ArtAttention机制和可变帧率+随机遮掩的预训练策略，在多个标准benchmark上与专家模型竞争甚至超越。

**[Learning Cross-Hand Policies Of High-Dof Reaching And Grasping](human_understanding/learning_cross-hand_policies_of_high-dof_reaching_and_grasping.md)**

:   提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。

**[Merlin Single-Shot Material Estimation And Relighting For Photometric Stereo](human_understanding/merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)**

:   提出单阶段注意力沙漏网络MERLiN，从单张图像联合估计空间变化BRDF参数并进行物理正确的重打光，首次利用重打光图像驱动光度立体方法实现单图法向估计，弥合了Shape from Shading与Photometric Stereo之间的鸿沟。

**[Modeling And Driving Human Body Soundfields Through Acoustic Primitives](human_understanding/modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)**

:   提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。

**[Motion Mamba Efficient And Long Sequence Motion Generation](human_understanding/motion_mamba_efficient_and_long_sequence_motion_generation.md)**

:   本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。

**[Multi-Memory Matching For Unsupervised Visible-Infrared Person Re-Identification](human_understanding/multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)**

:   提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。

**[Nymeria A Massive Collection Of Multimodal Egocentric Daily Motion In The Wild](human_understanding/nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)**

:   提出 Nymeria 数据集——目前最大规模的野外多模态自我中心人体日常运动数据集，包含 300 小时、264 人、50 个场景，提供全身精确动作捕捉、多设备同步多模态数据和 310.5K 句分层语言描述，并在 body tracking、motion synthesis 等任务上建立 baseline。

**[Occlusion Handling In 3D Human Pose Estimation With Perturbed Positional Encodin](human_understanding/occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)**

:   针对人体关节遮挡导致2D骨架图边缺失、传统图拉普拉斯位置编码失效的问题，提出PerturbPE方法，利用瑞利-薛定谔微扰定理多次随机扰动并求平均来提取图拉普拉斯特征基的一致性部分作为位置编码，在完整骨架上优于MöbiusGCN，在边缺失场景下性能提升达12%。

**[One-Stage Prompt-Based Continual Learning](human_understanding/one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[Petface A Large-Scale Dataset And Benchmark For Animal Identification](human_understanding/petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)**

:   构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

**[Pisr Polarimetric Neural Implicit Surface Reconstruction For Textureless And Spe](human_understanding/pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)**

:   提出PISR方法，利用偏振光的几何约束（偏振角与法线方位角的对应关系）直接正则化神经隐式表面形状，结合哈希网格加速和图像空间法线平滑，在无纹理和镜面物体上实现了0.5mm Chamfer距离和99.5% F-score的高精度重建，速度比此前偏振方法快4~30倍。

**[Pose-Aware Self-Supervised Learning With Viewpoint Trajectory Regularization](human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)**

:   提出了一个自监督学习基准，同时评估语义分类和姿态估计能力，并设计视角轨迹正则化损失(trajectory loss)，利用相邻视角的图像三元组约束特征空间中的局部线性性，使学到的表征既保持语义分类精度又获得 emergent 的全局姿态感知能力，在域内和域外姿态估计上均提升4%。

**[Quar-Vla Vision-Language-Action Model For Quadruped Robots](human_understanding/quar-vla_vision-language-action_model_for_quadruped_robots.md)**

:   首次提出四足机器人视觉-语言-动作（QUAR-VLA）范式，构建 259K episode 的多任务数据集 QUARD 和基于预训练多模态大模型的 QUART 模型，实现感知、导航、全身操作等多任务统一控制。

**[Quarvla Visionlanguageaction Model For Quadruped Robots](human_understanding/quarvla_visionlanguageaction_model_for_quadruped_robots.md)**

:   提出 QUAR-VLA 范式，首次将视觉、语言指令和动作生成统一到四足机器人中，构建了大规模多任务数据集 QUARD（259K episodes），训练 QUART 模型（基于 8B VLM）实现感知、导航、全身操控等多种任务，并展示了从仿真到真实的迁移能力。

**[Scape A Simple And Strong Category-Agnostic Pose Estimator](human_understanding/scape_a_simple_and_strong_category-agnostic_pose_estimator.md)**

:   通过将类别无关姿态估计(CAPE)简化为纯自注意力特征匹配问题，抛弃显式相似度匹配和两阶段框架，引入全局关键点特征感知器(GKP)和关键点注意力精炼器(KAR)以提升注意力质量，在MP-100数据集上1-shot/5-shot设置下分别超越SOTA 2.2/1.3 PCK，同时减少参数量和提升推理速度。

**[Self-Supervised Feature Adaptation For 3D Industrial Anomaly Detection](human_understanding/self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。

**[Selfsupervised Feature Adaptation For 3D Industrial Ano](human_understanding/selfsupervised_feature_adaptation_for_3d_industrial_ano.md)**

:   提出 LSFA（Local-to-global Self-supervised Feature Adaptation），通过模态内特征紧致化（IFC）和跨模态局部到全局一致性对齐（CLC）微调适配器，学习面向异常检测的任务导向表示，在 MVTec-3D AD 上达到 97.1% I-AUROC（+3.4%）。

**[Signavatars A Large-Scale 3D Sign Language Holistic Motion Dataset And Benchmark](human_understanding/signavatars_a_large-scale_3d_sign_language_holistic_motion_dataset_and_benchmark.md)**

:   提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。

**[Upose3D Uncertainty-Aware 3D Human Pose Estimation With Cross-View And Temporal ](human_understanding/upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)**

:   提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。

**[Videoclusternet Self-Supervised And Adaptive Face Clustering For Videos](human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)**

:   VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。

**[Wear-Any-Way Manipulable Virtual Try-On Via Sparse Correspondence Alignment](human_understanding/wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)**

:   提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。

**[Wordrobe Text-Guided Generation Of Textured 3D Garments](human_understanding/wordrobe_text-guided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。

**[Wordrobe Textguided Generation Of Textured 3D Garments](human_understanding/wordrobe_textguided_generation_of_textured_3d_garments.md)**

:   提出 WordRobe 框架，通过学习 3D 服装潜在空间并与 CLIP 嵌入对齐，实现文本驱动的带纹理 3D 服装网格生成，并利用 ControlNet 的单步前向推理实现高效视角一致的纹理合成。

**[Worldpose A World Cup Dataset For Global 3D Human Pose Estimation](human_understanding/worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)**

:   利用2022年FIFA世界杯体育场部署的多视角静态摄像机基础设施，构建了首个大规模多人全局3D姿态估计数据集WorldPose，包含约250万个3D姿态和超过120公里的全局轨迹，并揭示了现有全局姿态估计方法在多人场景下面临的严峻挑战。

---

## 🎬 视频理解 { #video_understanding }

**[Actionswitch Class-Agnostic Detection Of Simultaneous Actions In Streaming Video](video_understanding/actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)**

:   提出 ActionSwitch——首个无需类别信息即可检测流式视频中重叠动作实例的在线时序动作定位（On-TAL）框架，核心将多动作检测建模为有限状态机的状态分类问题，并辅以 conservativeness loss 减少碎片化误检，在 THUMOS14、FineAction、Epic-Kitchens 100 等数据集上在 OAD 扩展方法中达到 SOTA。

**[Adapt2Reward Adapting Videolanguage Models To Generalizable](video_understanding/adapt2reward_adapting_videolanguage_models_to_generalizable.md)**

:   提出 Adapt2Reward，通过可学习的失败提示（failure prompts）将预训练视频语言模型适配为可泛化的语言条件奖励函数，仅需少量单一环境的机器人数据即可泛化到新环境和新任务，在 MetaWorld 上比前方法高出约 28%。

**[Amego Active Memory From Long Egocentric Videos](video_understanding/amego_active_memory_from_long_egocentric_videos.md)**

:   提出 AMEGO，一种从长第一人称视频中在线构建结构化"活跃记忆"的方法，通过 HOI tracklet + 位置分段 + 语义无关的视觉查询，在新提出的 AMB benchmark 上超越 Video QA baselines 12.7%。

**[Benchmarks And Challenges In Pose Estimation For Egocentric Hand Interactions Wi](video_understanding/benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)**

:   基于 HANDS23 挑战赛（AssemblyHands + ARCTIC 数据集），系统性地对第一人称视角下手-物体交互的 3D 姿态估计方法进行了基准测试和深入分析，揭示了畸变校正、高容量 Transformer 和多视角融合的有效性，以及快速运动、遮挡和窄视角下物体重建等仍未解决的挑战。

**[Blazebvd Make Scale-Time Equalization Great Again For Blind Video Deflickering](video_understanding/blazebvd_make_scale-time_equalization_great_again_for_blind_video_deflickering.md)**

:   提出 BlazeBVD，利用经典 Scale-Time Equalization (STE) 在光照直方图空间提取 deflickering 先验（滤波光照图、曝光图、闪烁帧索引），将复杂的视频时空学习简化为 2D 空间网络逐帧处理 + 轻量 3D 时序一致性网络，在盲视频去闪烁任务上实现 SOTA 质量且推理速度比基线快 10 倍以上。

**[Classification Matters Improving Video Action Detection With Class-Specific Atte](video_understanding/classification_matters_improving_video_action_detection_with_class-specific_atte.md)**

:   提出类别专属查询（class queries）机制，通过为每个动作类别分配独立的可学习查询，让模型动态关注与各类别相关的上下文区域，显著提升视频动作检测中的分类性能。

**[Crossglg Llm Guides One-Shot Skeleton-Based 3D Action Recognition In A Cross-Lev](video_understanding/crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)**

:   提出CrossGLG框架，利用LLM生成的文本描述以"全局→局部→全局"的方式引导骨架特征学习，在单样本3D动作识别中以仅2.8%的SOTA模型参数量大幅超越对手。

**[Data Collection-Free Masked Video Modeling](video_understanding/data_collection-free_masked_video_modeling.md)**

:   提出基于伪运动生成器（PMG）从静态图像递归生成伪运动视频，结合掩码视频建模（VideoMAE）进行自监督预训练，完全摆脱真实视频数据的采集成本和隐私/版权顾虑，甚至可用合成图像实现有效的视频Transformer预训练。

**[Dino-Tracker Taming Dino For Self-Supervised Point Tracking In A Single Video](video_understanding/dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)**

:   提出DINO-Tracker，将预训练DINOv2的语义特征与测试时单视频优化相结合，通过Delta-DINO残差微调和多源自监督损失实现长程稠密点追踪，在自监督方法中达到SOTA且可媲美有监督追踪器，尤其在长期遮挡场景中大幅领先。

**[Draganything Motion Control For Anything Using Entity Representation](video_understanding/draganything_motion_control_for_anything_using_entity_representation.md)**

:   提出DragAnything，利用扩散模型的隐空间特征作为实体表征（Entity Representation）来实现实体级运动控制，解决了现有轨迹驱动方法仅拖拽像素而无法精确控制目标对象运动的问题，在VIPSeg上实现SOTA的FVD/FID指标，用户研究中运动控制投票超出DragNUWA 26%。

**[Egoposer Robust Real-Time Egocentric Pose Estimation From Sparse And Intermitten](video_understanding/egoposer_robust_real-time_egocentric_pose_estimation_from_sparse_and_intermitten.md)**

:   提出 EgoPoser，仅从头显设备的头部和手部稀疏且间歇性追踪信号中，鲁棒地估计全身姿态，通过全局运动分解、真实视野建模、SlowFast时序融合和体型感知优化四大核心设计，在大规模真实场景中实现SOTA性能，推理速度超600fps。

**[Elysium Exploring Object-Level Perception In Videos Via Mllm](video_understanding/elysium_exploring_object-level_perception_in_videos_via_mllm.md)**

:   提出 Elysium，一个端到端可训练的 MLLM，通过构建百万级视频目标感知数据集 ElysiumTrack-1M 和设计 T-Selector 视觉 token 压缩网络，将 MLLM 的目标级感知能力从图像扩展到视频领域，支持单目标跟踪 (SOT)、引用式单目标跟踪 (RSOT) 和视频引用表达生成 (Video-REG) 等任务。

**[Elysium Exploring Objectlevel Perception In Videos Via Mllm](video_understanding/elysium_exploring_objectlevel_perception_in_videos_via_mllm.md)**

:   提出Elysium，首个端到端可训练的多模态大语言模型系统化处理视频目标级任务（如目标跟踪），构建了百万级ElysiumTrack-1M视频数据集支持SOT/RSOT/Video-REG三类任务，并设计T-Selector token压缩网络在保持性能的同时大幅减少视觉token消耗。

**[Finepseudo Improving Pseudo-Labelling Through Temporal-Alignablity For Semi-Supe](video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)**

:   提出 FinePseudo 框架，利用基于时序对齐性（temporal alignability）的度量学习来改善伪标签质量，首次系统性地解决半监督细粒度动作识别问题，在四个细粒度数据集上显著超越现有方法。

**[Goldfish Vision-Language Understanding Of Arbitrarily Long Videos](video_understanding/goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)**

:   提出 Goldfish 框架，通过将长视频分割为短 clip 并利用基于文本相似度的检索机制选取与问题最相关的 top-k 片段，实现对任意长度视频的高效理解，同时提出 MiniGPT4-Video 短视频模型和 TVQA-long 长视频评测基准。

**[Layeredflow A Real-World Benchmark For Non-Lambertian Multi-Layer Optical Flow](video_understanding/layeredflow_a_real-world_benchmark_for_non-lambertian_multi-layer_optical_flow.md)**

:   提出 LayeredFlow——首个包含多层光流标注的真实世界非朗伯体基准数据集（150k 光流对，185 个场景，360 个物体），并提出多层光流任务定义、大规模合成训练数据集和基于 RAFT 的多层光流基线方法。

**[Leveraging Temporal Contextualization For Video Action Recognition](video_understanding/leveraging_temporal_contextualization_for_video_action_recognition.md)**

:   提出 **TC-CLIP** 框架，通过**时序上下文化(TC)** 机制将全局视频动作线索压缩为少量 context tokens 注入 CLIP 编码过程，并设计**视频条件提示(VP)** 模块将视觉信息注入文本端，在零样本、小样本、base-to-novel 和全监督四种设定下全面超越现有 CLIP-based 视频识别方法。

**[Masked Video And Body-Worn Imu Autoencoder For Egocentric Action Recognition](video_understanding/masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)**

:   提出 EVI-MAE，首个联合第一人称视频与身体穿戴 IMU 的多模态表示学习方法，通过 MAE 自监督预训练学习视频-IMU 跨模态对齐，并用图神经网络建模多 IMU 设备间的协同运动关系，在动作识别中取得 SOTA 且具备优秀的鲁棒性。

**[Motion-Prior Contrast Maximization For Dense Continuous-Time Motion Estimation](video_understanding/motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)**

:   本文提出一种将非线性运动先验（轨迹参数函数）引入对比度最大化框架的自监督方法，用于事件相机的稠密连续时间运动估计，在真实世界数据集 EVIMO2 上将合成数据预训练模型的零样本性能提升了 29%。

**[Nymeria A Massive Collection Of Multimodal Egocentric Daily ](video_understanding/nymeria_a_massive_collection_of_multimodal_egocentric_daily_.md)**

:   Nymeria是全球最大野外人体运动数据集，300h/264人多设备多模态自我中心数据和310.5K句语言描述。

**[Nymeria A Massive Collection Of Multimodal Egocentric Daily Motion In The Wild](video_understanding/nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)**

:   Nymeria 是目前世界最大的野外人体运动数据集（300 小时、264 名参与者），首次提供同步定位的多设备多模态自我中心数据（Project Aria 眼镜+腕带+动捕服），并配套 310.5K 句层次化运动语言描述。

**[On The Utility Of 3D Hand Poses For Action Recognition](video_understanding/on_the_utility_of_3d_hand_poses_for_action_recognition.md)**

:   提出 HandFormer，一种轻量级多模态 Transformer，将密集采样的 3D 手部姿态（捕捉细粒度动作）与稀疏采样的 RGB 帧（提供场景语义）结合，通过 micro-action 时序分解和 trajectory 编码高效建模手-物交互，在 Assembly101 和 H2O 上达到 SOTA，且纯 pose 模型以 5× 更少 FLOPs 超越已有骨架方法。

**[Pite Pixel-Temporal Alignment For Large Video-Language Model](video_understanding/pite_pixel-temporal_alignment_for_large_video-language_model.md)**

:   提出 PiTe 模型，通过物体运动轨迹在像素级别实现视频与语言的时空对齐，构建 PiTe-143k 数据集，在零样本 QA、时序定位和密集描述任务上大幅超越现有方法。

**[Pite Pixeltemporal Alignment For Large Videolanguage Mo](video_understanding/pite_pixeltemporal_alignment_for_large_videolanguage_mo.md)**

:   提出 PiTe，一种通过物体轨迹引导的像素-时序对齐方法，利用自动构建的 PiTe-143K 数据集在空间和时间维度上实现视频与语言的精细对齐，显著提升视频理解能力。

**[R2Tuning Efficient Imagetovideo Transfer Learning For Video](video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)**

:   R²-Tuning提出了一个仅需1.5%参数的轻量R²Block，通过从CLIP后层向前层的逆向递归方式聚合多层空间特征并精化时序关联，在6个VTG基准上以2.7M参数超越了使用额外时序骨干的4倍大方法。

**[Referring Atomic Video Action Recognition](video_understanding/referring_atomic_video_action_recognition.md)**

:   提出"基于文本引用的原子视频动作识别"（RAVAR）新任务和 RefAVA 数据集（36,630 实例），以及 RefAtomNet 方法，通过跨流 agent 注意力融合视觉、文本和位置-语义三路 token，在 mAP 上比最佳基线 BLIPv2 提升 3.85%/3.17%。

**[Rethinking Video-Text Understanding Retrieval From Counterfactually Augmented Da](video_understanding/rethinking_video-text_understanding_retrieval_from_counterfactually_augmented_da.md)**

:   提出反事实增强数据检索（RCAD）任务和 Feint6K 数据集，揭示 SOTA 视频文本模型在动作语义理解上远落后于人类（InternVideo 58.2% vs 人类 95.2%），并提出 LLM-teacher 通过 LLM 知识蒸馏改善动作嵌入学习。

**[Rgnet A Unified Clip Retrieval And Grounding Network For Long Videos](video_understanding/rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)**

:   提出 RGNet 将长视频时序定位的片段检索和时序定位两个阶段深度统一到单一网络中，通过 RG-Encoder 的稀疏注意力和对比片段采样实现端到端优化，在 MAD 和 Ego4D 上取得 SOTA。

**[Sa-Dvae Improving Zero-Shot Skeleton-Based Action Recognition By Disentangled Va](video_understanding/sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)**

:   SA-DVAE 首次将特征解耦引入骨架零样本动作识别，通过双头 VAE 将骨架特征分离为语义相关和语义无关两个独立部分，仅用语义相关部分与文本对齐，配合对抗性总相关惩罚增强解耦效果，在 NTU RGB+D 60/120 和 PKU-MMD 三个基准上达到 SOTA。

**[Safnet Selective Alignment Fusion Network For Efficient Hdr Imaging](video_understanding/safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)**

:   SAFNet 提出选择性对齐融合策略，通过金字塔解码器联合精炼有价值区域掩码和跨曝光光流，仅在有价值区域进行精确对齐后显式融合 HDR 图像，在 Kalantari 17 和自建 Challenge123 数据集上超越 SOTA 的同时推理速度快一个数量级。

**[Sea-Raft Simple Efficient Accurate Raft For Optical Flow](video_understanding/sea-raft_simple_efficient_accurate_raft_for_optical_flow.md)**

:   SEA-RAFT 通过混合拉普拉斯损失(MoL)、直接回归初始光流和刚性流预训练三项改进，在保持简洁架构的同时实现了 SOTA 精度，并比现有方法快 2.3× 以上。

**[Self-Supervised Any-Point Tracking By Contrastive Random Walks](video_understanding/self-supervised_any-point_tracking_by_contrastive_random_walks.md)**

:   提出 GMRW（Global Matching Random Walk），将全局匹配 Transformer 架构与对比随机游走自监督目标结合，首次在无标注的情况下实现了强劲的"任意点跟踪"（TAP）性能，并设计 label warping 数据增强来避免 Transformer 的捷径解。

**[Slack Semantic Location And Appearance Aware Open-Vocabulary Tracking](video_understanding/slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)**

:   SLAck 提出在多目标跟踪的关联阶段早期统一融合语义、位置和外观三种线索，通过轻量级时空目标图（STOG）学习隐式运动先验和跨线索协同，无需后处理启发式规则，在开放词汇 MOT 和 TAO TETA 基准上显著提升新类别跟踪性能。

**[Spamming Labels Efficient Annotations For The Trackers Of Tomorrow](video_understanding/spamming_labels_efficient_annotations_for_the_trackers_of_tomorrow.md)**

:   提出 SPAM 视频标注引擎，将合成数据预训练、伪标签自训练和基于图层级的主动学习相结合，仅需 3-20% 的人工标注量即可产生接近 GT 质量的多目标跟踪标注。

**[Spherical World-Locking For Audio-Visual Localization In Egocentric Videos](video_understanding/spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)**

:   提出球面世界锁定（Spherical World-Locking, SWL）框架，通过将多模态感知流隐式变换到世界锁定的球面坐标系中，消除自身运动带来的挑战，实现更精准的第一人称视频中的音视觉定位。

**[Sync From The Sea Retrieving Alignable Videos From Large-Scale Datasets](video_understanding/sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)**

:   提出可对齐视频检索（Alignable Video Retrieval, AVR）任务，通过 DRAQ 对齐质量指标从大规模视频数据库中识别并检索出最适合与查询视频进行时序对齐的视频，同时提出特征上下文化方法提升对齐性能。

**[Text-Guided Video Masked Autoencoder](video_understanding/text-guided_video_masked_autoencoder.md)**

:   提出文本引导掩码策略（TGM）利用自然语言描述替代运动先验来掩码视频显著区域，并统一 MAE 与视频-文本对比学习，在五个动作识别和一个自中心数据集上取得最佳相对性能。

**[Towards Model-Agnostic Dataset Condensation By Heterogeneous Models](video_understanding/towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)**

:   提出异构模型数据集压缩（HMDC）方法，通过同时使用两个结构不同的模型（如 ConvNet 和 ViT）进行数据集压缩，并设计梯度平衡模块和互蒸馏机制，生成对各种模型普遍适用的压缩图像，解决传统方法过度适配单一模型的问题。

**[Uniinr Event-Guided Unified Rolling Shutter Correction Deblurring And Interpolat](video_understanding/uniinr_event-guided_unified_rolling_shutter_correction_deblurring_and_interpolat.md)**

:   提出 UniINR 框架，利用统一的时空隐式神经表征（INR）从单张卷帘快门模糊帧和配对事件流中，一次性同时完成卷帘快门校正、去模糊和任意帧率的视频帧插值。

**[Vamos Versatile Action Models For Video Understanding](video_understanding/vamos_versatile_action_models_for_video_understanding.md)**

:   提出 Vamos 框架，以大语言模型作为推理器，灵活统一视觉嵌入和通用文本描述作为视频表征，发现纯文本表征在多个视频理解基准上一致性地取得竞争甚至更优性能，并设计 Token Bottleneck Model 实现可解释证据选择与 5 倍推理加速。

**[Videomamba Spatio-Temporal Selective State Space Model](video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)**

:   提出基于纯 Mamba 架构的视频识别模型 VideoMamba（KAIST 版），通过设计时空前向-后向 SSM（Spatio-Temporal Forward and Backward SSM）来有效处理视频中非序列空间信息与序列时间信息的复杂交互，以线性复杂度实现了与 Transformer 竞争的性能。

**[Videomamba State Space Model For Efficient Video Understanding](video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)**

:   将 Mamba 的选择性状态空间模型创新性地适配到视频领域，提出纯 SSM 架构的 VideoMamba，以线性复杂度实现高效的时空上下文建模，在短视频和长视频理解任务上均展现出优越性能。

---

## 🎯 目标检测 { #object_detection }

**[A New Dataset And Framework For Real-World Blurred Images Super-Resolution](object_detection/a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)**

:   针对现有盲超分方法在处理含模糊（散焦/运动模糊）图像时过度纹理化、破坏模糊区域感知质量的问题，构建了包含近3000张模糊图像的ReBlurSR数据集，并提出PBaSR框架，通过双分支解耦训练（CDM）和基于权重插值的跨分支融合（CFM），在不增加任何推理开销的前提下，同时提升模糊图像和普通图像的超分效果，LPIPS提升0.02~0.10。

**[Adaptive Bounding Box Uncertainties Via Twostep Conformal Pr](object_detection/adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   提出两步共形预测框架为多类目标检测的边界框生成带理论覆盖率保证的自适应不确定性区间——第一步用共形分类集处理类别误判风险，第二步用集成/分位数回归等方法构建自适应于目标尺寸的边界框预测区间，在COCO/Cityscapes/BDD100k上达到约90%目标覆盖率且区间实际可用。

**[Afreeca Annotation-Free Counting For All](object_detection/afreeca_annotation-free_counting_for_all.md)**

:   利用 Stable Diffusion 生成合成排序/计数数据，通过先学排序再学计数的两阶段策略 + 密度引导的图像分块，实现了首个适用于任意类别物体的无标注计数方法，在人群计数上超越已有无监督方法。

**[Afreeca Annotationfree Counting For All](object_detection/afreeca_annotationfree_counting_for_all.md)**

:   利用潜在扩散模型（LDM）生成合成计数和排序数据，提出首个可适用于任意物体类别的无监督计数方法，无需任何人工标注即可实现准确计数。

**[Bam-Detr Boundary-Aligned Moment Detection Transformer For Temporal Sentence Gro](object_detection/bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)**

:   提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。

**[Be Yourself Bounded Attention For Multi-Subject Text-To-Image Generation](object_detection/be_yourself_bounded_attention_for_multi-subject_text-to-image_generation.md)**

:   提出 Bounded Attention，一种无需训练的注意力约束方法，通过在去噪过程中限制 cross-attention 和 self-attention 的信息流动来解决多主体文本到图像生成中的语义泄漏问题。

**[Be Yourself Bounded Attention For Multisubject Texttoimage G](object_detection/be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)**

:   Be Yourself深入分析了扩散模型中Cross-Attention和Self-Attention导致的多主体语义泄漏问题，提出Bounded Attention机制，通过在去噪过程中限制不同主体间的信息流动来生成语义独立的多主体图像，免训练即可生成5+个语义相似主体。

**[Bridge Past And Future Overcoming Information Asymmetry In Incremental Object De](object_detection/bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)**

:   提出 Bridge Past and Future (BPF) 方法，通过伪标签桥接过去阶段、注意力机制排除未来潜在物体，并结合双教师蒸馏（Distillation with Future），解决增量目标检测中跨阶段信息不对称导致的优化目标不一致问题。

**[Can Ood Object Detectors Learn From Foundation Models](object_detection/can_ood_object_detectors_learn_from_foundation_models.md)**

:   SyncOOD 提出一种自动化数据策展方法，利用 LLM 想象语义新颖的 OOD 概念，通过 Stable Diffusion Inpainting 在 ID 图像上进行区域级编辑合成场景级 OOD 样本，再经 SAM 精炼框和特征相似度过滤后训练轻量 MLP 分类器，在多个 OOD 检测基准上以极少量合成数据大幅超越 SOTA。

**[Damsdet Dynamic Adaptive Multispectral Detection Transformer With Competitive Qu](object_detection/damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)**

:   DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。

**[Efficient Inference Of Vision Instruction-Following Models With Elastic Cache](object_detection/efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)**

:   Elastic Cache 提出一种针对多模态指令遵循模型的 KV Cache 管理方法，在指令编码阶段采用基于重要性的 cache 合并策略（而非丢弃），在输出生成阶段采用固定点淘汰策略，以"一个序列、两种策略"实现任意加速比的高效推理，在 KV Cache 预算仅 0.2 时实现 78% 的实际速度提升且保持生成质量。

**[Gra Detecting Oriented Objects Through Group-Wise Rotating And Attention](object_detection/gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)**

:   提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。

**[Hat History-Augmented Anchor Transformer For Online Temporal Action Localization](object_detection/hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)**

:   提出HAT——首个在Online Temporal Action Localization（OnTAL）中引入长期历史上下文的anchor-based Transformer框架，通过动作预期引导的历史压缩和未来驱动的历史精炼，在程序性自我中心数据集（EGTEA/EK100）上显著超越OAT，在标准数据集（THUMOS/MUSES）上达到可比或更优性能。

**[Implicit Concept Removal Of Diffusion Models](object_detection/implicit_concept_removal_of_diffusion_models.md)**

:   提出 Geom-Erasing 方法，通过引入外部分类器/检测器提供隐式概念的存在性和几何位置信息，将其编码为文本条件中的位置 token 并作为负提示使用，有效消除扩散模型中水印、不安全内容等"隐式概念"的生成，在 I2P 和自建 ICD 基准上达到 SOTA。

**[Lami-Detr Open-Vocabulary Detection With Language Model Instruction](object_detection/lami-detr_open-vocabulary_detection_with_language_model_instruction.md)**

:   提出 LaMI-DETR，通过利用 GPT 生成视觉概念描述和 T5 挖掘类间视觉相似性关系，解决开放词汇目标检测中概念表示不足和基类过拟合两大问题，在 OV-LVIS 上以 43.4 的 rare AP 超越前最佳方法 7.8 个点。

**[Layoutdetr Detection Transformer Is A Good Multimodal Layout](object_detection/layoutdetr_detection_transformer_is_a_good_multimodal_layout.md)**

:   将版式设计问题重新构建为基于背景图像的目标检测问题，提出LayoutDETR框架，利用DETR的transformer编解码器结构结合GAN/VAE生成先验，以多模态前景元素（图像+文本）为输入，生成考虑背景语义的排版布局，在公开基准和自建广告横幅数据集上均达到SOTA。

**[Layoutdetr Detection Transformer Is A Good Multimodal Layout Designer](object_detection/layoutdetr_detection_transformer_is_a_good_multimodal_layout_designer.md)**

:   将目标检测框架 DETR 与生成模型（GAN/VAE）统一，提出 LayoutDETR 用于多模态条件下的图形布局自动设计，以背景图像为约束、前景图文元素为驱动，在广告横幅和 UI 布局生成上达到 SOTA。

**[Learn From The Learnt Source-Free Active Domain Adaptation Via Contrastive Sampl](object_detection/learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)**

:   提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。

**[Mutdet Mutually Optimizing Pre-Training For Remote Sensing Object Detection](object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)**

:   提出 MutDet，一种面向遥感旋转目标检测的互优化预训练框架，通过双向交叉注意力融合 object embeddings 与 encoder 特征、对比对齐损失、以及辅助孪生头，系统性地缓解了检测预训练中 object embeddings 与 detector features 之间的特征差异问题。

**[Nonverbal Interaction Detection](object_detection/nonverbal_interaction_detection.md)**

:   首次系统性研究人类非语言交互（手势、表情、注视、姿态、触碰），提出大规模数据集 NVI、新任务 NVI-DET 和基于双重多尺度超图的检测模型 NVI-DEHR，在非语言交互检测和 HOI 检测任务上均取得最优性能。

**[Openkd Opening Prompt Diversity For Zero- And Few-Shot Keypoint Detection](object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)**

:   提出 OpenKD 模型，从模态（视觉+文本）、语义（seen vs. unseen）、语言（多样化文本）三个维度开放 prompt 多样性，通过多模态 prototype set、辅助关键点-文本插值和 LLM 文本解析，实现通用的 zero- and few-shot keypoint detection，在 Animal Pose、AwA、CUB、NABird 上取得 SOTA。

**[Portrait4D-V2 Pseudo Multi-View Data Creates Better 4D Head Synthesizer](object_detection/portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)**

:   提出一种利用**伪多视角视频**来训练前馈式单图4D头部合成器的新学习范式：先用合成数据学一个3D头部合成器将单目视频转为多视角，再利用伪多视角视频通过**跨视角自重演**学习4D合成器，避免了对3DMM的过度依赖，在重建保真度、几何一致性和运动控制精度上大幅超越先前方法。

**[Projecting Points To Axes Oriented Object Detection Via Point-Axis Representatio](object_detection/projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)**

:   提出点-轴（Point-Axis）表示方法，将旋转目标的位置（点集）和方向（轴编码）解耦，配合 Max-Projection Loss 和 Cross-Axis Loss 实现无需额外标注的优化，并基于此设计 Oriented DETR 模型，解决传统旋转框表示的损失不连续问题。

**[Rectify The Regression Bias In Long-Tailed Object Detection](object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)**

:   首次揭示并系统解决长尾目标检测中被忽视的**回归偏差**问题：稀有类别的类别专属(class-specific)回归头参数因样本不足导致泛化能力差，通过添加额外的类别不可知(class-agnostic)回归分支进行权衡，在LVIS等数据集上取得了SOTA性能。

**[Reground Improving Textual And Spatial Grounding At No Cost](object_detection/reground_improving_textual_and_spatial_grounding_at_no_cost.md)**

:   通过将 GLIGEN 中 Gated Self-Attention (GSA) 与 Cross-Attention (CA) 的串行连接改为并行连接（网络重连），在不引入任何新参数、不需要微调、不增加计算开销的前提下，显著缓解了文本定位与空间定位之间的权衡问题。

**[Shine Saliency-Aware Hierarchical Negative Ranking For Compositional Temporal Gr](object_detection/shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)**

:   针对组合时序定位任务中现有方法负样本构造不合理、DETR 模型对负查询无法产生合理显著性响应的问题，提出利用 LLM（GPT-3.5 Turbo）生成语义可行的分层硬负样本，并设计粗到细的显著性排序策略建立视频片段与层次负查询之间的多粒度语义关系，显著提升组合泛化能力。

**[Spherical Linear Interpolation And Text-Anchoring For Zero-Shot Composed Image R](object_detection/spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)**

:   提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。

**[Taptr Tracking Any Point With Transformers As Detection](object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)**

:   TAPTR 将 Tracking Any Point (TAP) 任务重新建模为类 DETR 的检测问题，将每个跟踪点表示为包含位置和内容的 point query，通过多层 Transformer 解码器逐层优化，结合 cost volume 和滑动窗口特征更新策略，在 TAP-Vid 基准上达到 SOTA 且推理速度更快。

**[Tensorial Template Matching For Fast Cross-Correlation With Rotations And Its Ap](object_detection/tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)**

:   提出张量模板匹配（TTM）算法，通过对称张量场将模板在所有旋转下的信息整合为固定数量的相关计算，使得计算复杂度与旋转精度无关，在3D断层扫描图像中实现快速且准确的目标检测与旋转估计。

**[Towards Natural Language-Guided Drones Geotext-1652 Benchmark With Spatial Relat](object_detection/towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)**

:   构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。

**[Towards Natural Languageguided Drones Geotext1652 Bench](object_detection/towards_natural_languageguided_drones_geotext1652_bench.md)**

:   构建 GeoText-1652 多视角自然语言引导地理定位基准数据集（276K text-bbox 对），提出利用区域级空间关系匹配（grounding loss + spatial loss）进行精细化文本-图像跨模态检索的方法，实现自然语言控制无人机导航。

**[Tracking Meets Lora Faster Training Larger Model Strong](object_detection/tracking_meets_lora_faster_training_larger_model_strong.md)**

:   首次将 LoRA 引入视觉目标跟踪领域，通过解耦位置编码和设计 MLP-only 头网络，使大规模 ViT 模型（最大 ViT-g）在实验室级资源下实现高效训练和 SOTA 跟踪性能。

**[Tracking Meets Lora Faster Training Larger Model Stronger Performance](object_detection/tracking_meets_lora_faster_training_larger_model_stronger_performance.md)**

:   LoRAT 首次将 LoRA 引入视觉目标跟踪，通过解耦位置编码（共享空间 + 独立类型嵌入）和纯 MLP 检测头两个 LoRA-友好设计，使得在实验室级资源上训练 ViT-g 骨干的跟踪器成为可能，在 LaSOT 上达到 0.762 SUC（新 SOTA），最轻变体 LoRAT-B-224 以 209 FPS 运行。

**[Visible And Clear Finding Tiny Objects In Difference Map](object_detection/visible_and_clear_finding_tiny_objects_in_difference_map.md)**

:   SR-TOD 首次将图像自重建机制引入目标检测，发现重建差异图与微小目标之间的强相关性，并设计差异图引导的特征增强（DGFE）模块，在自建反无人机数据集 DroneSwarms 和 VisDrone2019、AI-TOD 上均取得显著提升。

**[Weak-To-Strong Compositional Learning From Generative Models For Language-Based ](object_detection/weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)**

:   提出 WSCL 框架：利用 LLM 生成多样文本描述 + 扩散模型生成对应图像 + 弱检测器分解短语生成伪标框，构建密集合成三元组（image, description, bbox），配合组合对比学习显著提升语言引导目标检测性能，OmniLabel 上 GLIP-T 提升 +5.0AP。

**[Wecromcl Weakly Supervised Cross-Modality Contrastive Learning For Transcription](object_detection/wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)**

:   提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。

**[Yolov9 Learning What You Want To Learn Using Programmable Gradient Information](object_detection/yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)**

:   YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。

---

## 🏥 医学图像 { #medical_imaging }

**[Adaptive Correspondence Scoring For Unsupervised Medical Ima](medical_imaging/adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)**

:   针对医学图像无监督配准中噪声、遮挡等干扰因素导致的虚假重建误差问题，提出了一个自适应对应关系评分框架（AdaCS），通过学习像素级的对应置信度图来重新加权误差残差，以即插即用方式一致提升三种主流配准架构在三个数据集上的性能。

**[Alternate Diverse Teaching For Semi-Supervised Medical Image Segmentation](medical_imaging/alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)**

:   提出 AD-MT（Alternate Diverse Mean Teacher），通过随机周期性交替更新两个教师模型 + 基于熵的冲突调和策略，在半监督医学分割中解决 confirmation bias 问题，在 ACDC/LA/Pancreas 上全面超越 SOTA。

**[Architecture-Agnostic Untrained Network Priors For Image Reconstruction With Fre](medical_imaging/architecture-agnostic_untrained_network_priors_for_image_reconstruction_with_fre.md)**

:   提出三种与架构无关的频率正则化技术（带宽受限输入、带宽可控上采样、Lipschitz 正则化卷积层），统一解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，在 MRI 重建任务中显著缩小不同架构间的性能差距。

**[Cardiacnet Learning To Reconstruct Abnormalities For Cardiac Disease Assessment ](medical_imaging/cardiacnet_learning_to_reconstruct_abnormalities_for_cardiac_disease_assessment_.md)**

:   提出基于重建的心脏疾病评估框架 CardiacNet，通过 Consistency Deformation Codebook (CDC) 和 Consistency Deformation Discriminator (CDD) 学习正常与异常心脏超声视频之间的结构和运动差异，在射血分数预测、肺动脉高压和房间隔缺损分类三个任务上达到 SOTA。

**[Chameleon A Data-Efficient Generalist For Dense Visual Prediction In The Wild](medical_imaging/chameleon_a_data-efficient_generalist_for_dense_visual_prediction_in_the_wild.md)**

:   提出 Chameleon，一个基于 meta-learning 和 token matching 的数据高效视觉通才模型，仅需几十张标注图像即可适应全新的密集预测任务（包括医学图像、视频、3D 等），在六个下游基准上显著超越现有通才方法。

**[Chex Interactive Localization And Region Description In Chest X-Rays](medical_imaging/chex_interactive_localization_and_region_description_in_chest_x-rays.md)**

:   提出ChEX——一个同时支持文本提示和边界框查询的交互式胸部X光解释模型，通过DETR风格的prompt检测器和多任务联合训练，在9个胸部X光任务上与SOTA竞争，同时提供独特的定位可解释性和用户交互能力。

**[Co-Synthesis Of Histopathology Nuclei Image-Label Pairs Using A Context-Conditio](medical_imaging/co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)**

:   提出一种上下文条件化的联合扩散模型，能够同时合成组织病理学细胞核图像、语义标签和距离图，通过点图（centroid layout）和文本提示两种条件实现对合成过程的精确控制，并生成高质量的实例级标签用于下游核分割和分类任务。

**[Gtp-4O Modality-Prompted Heterogeneous Graph Learning For Omni-Modal Biomedical ](medical_imaging/gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)**

:   提出基于异构图的全模态生物医学表征学习框架 GTP-4o，通过异构图嵌入显式建模跨模态关系，利用图提示机制补全缺失模态，并设计知识引导的层次化跨模态聚合，在胶质瘤分级和生存预测任务上取得SOTA。

**[Gtp4O Modalityprompted Heterogeneous Graph Learning For](medical_imaging/gtp4o_modalityprompted_heterogeneous_graph_learning_for.md)**

:   提出 GTP-4o，一种基于模态提示的异构图学习框架，通过异构图嵌入、图提示补全缺失模态、知识引导的层级聚合，实现基因组学-病理图像-细胞图-文本等多种临床模态的统一表示学习。

**[I-Medsam Implicit Medical Image Segmentation With Segment Anything](medical_imaging/i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)**

:   提出 I-MedSAM，将 SAM 的强泛化能力与隐式神经表示（INR）的连续空间预测优势结合，通过频率适配器增强边界高频信息、不确定性引导采样精细化分割，仅用 1.6M 可训练参数即超越现有离散和隐式方法。

**[Improving Medical Multi-Modal Contrastive Learning With Expert Annotations](medical_imaging/improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)**

:   提出 eCLIP，通过整合放射科专家的眼动注视热力图作为额外监督信号，结合 mixup 增强和课程学习策略，在不修改 CLIP 核心架构的前提下增强医学多模态对比学习的表征质量。

**[Improving Medical Multimodal Contrastive Learning With Exper](medical_imaging/improving_medical_multimodal_contrastive_learning_with_exper.md)**

:   提出eCLIP，通过引入放射科医生的眼动热力图（eye-gaze heatmap）作为专家标注，利用热力图处理器和mixup增强策略扩充高质量正样本对，有效缓解医学CLIP中的"模态间隙"问题，在零样本推理、线性探测、跨模态检索和RAG报告生成等任务上取得一致性提升。

**[Is User Feedback Always Informative Retrieval Latent Defending For Semi-Supervis](medical_imaging/is_user_feedback_always_informative_retrieval_latent_defending_for_semi-supervis.md)**

:   发现用户反馈在域适应中并非总是有益——偏向纠正错误预测的"负偏反馈"(NBF)会导致现有半监督域适应方法性能下降，并提出 Retrieval Latent Defending 方法，通过在每个 mini-batch 中加入伪标签防御样本来平衡监督信号。

**[Ophnet A Large-Scale Video Benchmark For Ophthalmic Surgical Workflow Understand](medical_imaging/ophnet_a_large-scale_video_benchmark_for_ophthalmic_surgical_workflow_understand.md)**

:   构建了OphNet——目前最大规模的眼科手术视频基准数据集（2278个视频、285小时、66种手术类型、102种手术阶段、150种精细操作），支持手术类型识别、阶段识别、时序定位和阶段预测四大任务，其规模约为现有最大手术工作流分析基准的20倍。

**[Pathology-Knowledge Enhanced Multi-Instance Prompt Learning For Few-Shot Whole S](medical_imaging/pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)**

:   提出 PEMP 框架，将病理学先验知识（视觉样例 + 文本描述）融入 patch 级和 slide 级的 prompt 中，结合 CLIP 进行多实例 prompt learning，在少样本弱监督 WSI 分类任务上平均超越 SOTA 方法 4%。

**[Pathologyknowledge Enhanced Multiinstance Prompt Learni](medical_imaging/pathologyknowledge_enhanced_multiinstance_prompt_learni.md)**

:   提出 PEMP——病理知识增强的多实例提示学习框架，将视觉和文本病理先验（典型 patch/slide 示例 + 语言描述）注入 CLIP 的提示中，在 patch 和 slide 两个层级进行对比学习，显著提升少样本全切片图像（WSI）分类性能。

**[Radedit Stress-Testing Biomedical Vision Models Via Diffusion Image Editing](medical_imaging/radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)**

:   提出 RadEdit，一种基于扩散模型的医学图像编辑方法，通过引入 edit mask 和 keep mask 的双重掩码机制，打破数据中的虚假关联（spurious correlations），生成高质量的合成测试集来压力测试（stress-test）生物医学视觉模型对数据集偏移的鲁棒性。

**[Radiative Gaussian Splatting For Efficient X-Ray Novel View Synthesis](medical_imaging/radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)**

:   提出 X-Gaussian，首个将 3D 高斯泼溅（3DGS）应用于 X 射线新视角合成的框架，通过设计辐射高斯点云模型（替代球谐函数）和角度位姿长方体均匀初始化策略（替代 SfM），在性能上超越 SOTA NeRF 方法 6.5 dB 的同时，实现 73× 推理加速和仅 15% 的训练时间。

**[Textttnephi Neural Deformation Fields For Approximately Diff](medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)**

:   NePhi用隐式神经网络（SIREN）替代传统的体素化形变场来表示配准变换，通过编码器预测latent code + 可选的测试时优化实现快速且近似微分同胚的医学图像配准，在多分辨率设置下与SOTA精度相当但内存降低5倍。

**[Tip Tabular-Image Pre-Training For Multimodal Classification With Incomplete Dat](medical_imaging/tip_tabular-image_pre-training_for_multimodal_classification_with_incomplete_dat.md)**

:   提出 TIP 框架，通过掩码表格重建、图像-表格匹配和对比学习三种自监督任务，对表格数据和图像进行联合预训练，学习对不完整表格数据鲁棒的多模态表征，用于下游分类任务。

**[Tip Tabularimage Pretraining For Multimodal Classification W](medical_imaging/tip_tabularimage_pretraining_for_multimodal_classification_w.md)**

:   提出TIP框架，通过掩码表格重建、图像-表格匹配和对比学习三个自监督任务，在表格数据不完整的条件下学习鲁棒的多模态表示，在自然图像和医学图像分类任务上超越现有方法。

**[Topology-Preserving Downsampling Of Binary Images](medical_imaging/topology-preserving_downsampling_of_binary_images.md)**

:   提出首个基于离散优化（整数规划）的拓扑保持二值图像下采样方法，通过将下采样像素的黑白决策编码为布尔变量、拓扑保持作为硬约束、与原图相似度作为目标函数来求解，保证下采样结果具有与原图完全相同的Betti数（连通分量数和孔洞数），同时保持与传统方法竞争性的像素级相似度。

**[Unleashing The Power Of Prompt-Driven Nucleus Instance Segmentation](medical_imaging/unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)**

:   提出 PromptNucSeg 框架，通过训练一个 prompter 自动生成细胞核中心点 prompt，并微调 SAM 进行逐核分割，同时引入相邻核作为 negative prompt 解决重叠核分割问题，无需复杂后处理即在三个 benchmark 上达到 SOTA。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Accelerating Image Super-Resolution Networks With Pixel-Level Classification](image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)**

:   提出PCSR——首个像素级计算资源分配的超分方法，用轻量MLP分类器逐像素判断恢复难度并分配到不同容量的上采样器，在PSNR几乎不掉的情况下将FLOPs压低至原始模型的18%~57%，大幅优于现有patch级方法ClassSR和ARM。

**[Asymmetric Mask Scheme For Self-Supervised Real Image Denoising](image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)**

:   提出非对称掩码方案 AMSNet，训练时用单掩码、推理时用多掩码互补，突破了 blind spot network 对网络感受野的结构限制，在真实图像自监督去噪任务上取得 SOTA。

**[Bamm Bidirectional Autoregressive Motion Model](image_restoration/bamm_bidirectional_autoregressive_motion_model.md)**

:   提出 BAMM（双向自回归运动模型），通过统一生成掩码建模和自回归建模的混合注意力掩码策略，在一个框架中同时实现高质量运动生成、自适应长度预测和零样本运动编辑，在 HumanML3D 和 KIT-ML 上全面超越 SOTA。

**[Denoisplit A Method For Joint Microscopy Image Splitting And Unsupervised Denois](image_restoration/denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)**

:   提出 DenoiSplit，首个将语义图像分解（image splitting）和无监督去噪（unsupervised denoising）联合解决的方法，通过在层次化 VAE 中整合像素噪声模型和改进的 KL 散度损失加权策略，在荧光显微镜图像上实现了端到端的去噪+分解，性能显著优于先去噪再分解的串行方案。

**[Domain-Adaptive Video Deblurring Via Test-Time Blurring](image_restoration/domain-adaptive_video_deblurring_via_test-time_blurring.md)**

:   提出基于扩散模糊模型的测试时域适应方法，通过从模糊视频中检测相对清晰区域作为伪清晰图像，并生成域自适应的模糊条件来合成训练对，实现在未知域上对去模糊模型的微调，在 5 个真实数据集上最高提升 7.54dB。

**[Efficient Diffusion Transformer With Step-Wise Dynamic Attention Mediators](image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)**

:   发现 Diffusion Transformer 中 query-key 交互存在显著冗余（尤其在去噪早期），提出 Attention Mediator 机制将注意力复杂度降至线性，并设计逐步动态调整策略，在 SiT-XL/2 上实现 SOTA FID 2.01，同时减少计算量。

**[Joint Rgb-Spectral Decomposition Model Guided Image Enhancement In Mobile Photog](image_restoration/joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)**

:   提出 JDM-HDRNet，通过联合 RGB-光谱分解模型从低分辨率多光谱图像（Lr-MSI）中提取 shading、reflectance 和材质语义三种先验，将它们分别融入 HDRNet 以增强动态范围、色彩映射和语义网格专家学习，并构建了首个 RGB-高光谱配对的 Mobile-Spec 数据集。

**[Kalman-Inspired Feature Propagation For Video Face Super-Resolution](image_restoration/kalman-inspired_feature_propagation_for_video_face_super-resolution.md)**

:   受卡尔曼滤波启发，提出 KEEP 框架，通过在隐空间中循环融合先前帧的恢复信息和当前帧的观测信息来维持稳定的人脸先验，在视频人脸超分辨率任务上实现了高保真度和时序一致性的双重提升。

**[Learning Exhaustive Correlation For Spectral Super-Resolution Where Spatial-Spec](image_restoration/learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)**

:   本文提出 Exhaustive Correlation Transformer (ECT)，通过光谱方向非连续3D切分策略 (SD3D) 建模统一的空间-光谱相关性，并通过动态低秩映射模块 (DLRM) 捕获多token间的线性依赖关系，在光谱超分辨率任务上以最少的参数量和最低的推理延迟实现了 SOTA 性能。

**[Mambair A Simple Baseline For Image Restoration With State-Space Model](image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)**

:   本文首次将 Mamba（选择性状态空间模型）引入底层图像修复任务，通过设计残差状态空间块（RSSB）中的局部卷积增强和通道注意力机制，解决了 vanilla Mamba 在 2D 图像上的局部像素遗忘和通道冗余问题，在图像超分辨率和去噪任务上以线性复杂度实现了与 Transformer 方法相当甚至更优的性能（SR 上超过 SwinIR 0.45dB）。

**[Moe-Diffir Task-Customized Diffusion Priors For Universal Compressed Image Resto](image_restoration/moe-diffir_task-customized_diffusion_priors_for_universal_compressed_image_resto.md)**

:   提出 MoE-DiffIR，首个基于扩散模型的通用压缩图像复原（CIR）框架，通过混合专家（MoE）Prompt 模块从 Stable Diffusion 中挖掘任务定制化的扩散先验，结合 Visual-to-Text 适配器激活 SD 的跨模态生成先验，并构建了覆盖 7 种编解码器 × 3 个压缩级别共 21 种退化的首个通用 CIR 基准数据集。

**[Oapt Offset-Aware Partition Transformer For Double Jpeg Artifacts Removal](image_restoration/oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)**

:   针对双重 JPEG 压缩图像恢复问题，提出 OAPT，通过预测两次压缩之间的像素偏移量，将每个 8×8 block 中的四种不同模式进行聚类分组后分别进行自注意力处理，在双重 JPEG 恢复任务上超越 SOTA 方法 0.16 dB。

**[Overcoming Distribution Mismatch In Quantizing Image Super-Resolution Networks](image_restoration/overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)**

:   本文提出 ODM 框架，通过协同失配正则化（cooperative mismatch regularization）和逐层权重裁剪校正（weight clipping correction）两个简单策略，在不引入推理时动态模块的前提下解决 SR 网络量化中的分布失配问题，以极小的额外开销达到 SOTA。

**[Realviformer Investigating Attention For Real-World Video Super-Resolution](image_restoration/realviformer_investigating_attention_for_real-world_video_super-resolution.md)**

:   本文系统研究了空间注意力和通道注意力在真实世界视频超分辨率（RWVSR）中的行为差异，发现通道注意力对退化伪影更鲁棒但会导致特征冗余，据此提出了带有改进通道注意力（ICA）和通道注意力融合（CAF）模块的 RealViformer，以更少的参数和更快的速度达到 SOTA。

**[Restoring Images In Adverse Weather Conditions Via Histogram Transformer](image_restoration/restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)**

:   提出 Histoformer，一种基于直方图自注意力机制的高效 Transformer，通过将空间特征按像素强度排序分箱（bin），在箱内和箱间执行自注意力，实现动态范围的空间注意力以高效处理天气退化像素，配合动态范围卷积和 Pearson 相关性损失，在去雪/去雨雾/去雨滴三大任务上统一建模并达到 SOTA。

**[Rethinking Image Super-Resolution From Training Data Perspectives](image_restoration/rethinking_image_super-resolution_from_training_data_perspectives.md)**

:   从训练数据角度重新思考图像超分辨率，提出自动化数据评估流水线构建 DiverSeg 数据集（低分辨率但高质量、目标多样的图像），证明在该数据集上训练的 SR 模型可以超越使用高分辨率数据集（DF2K、LSDIR）训练的模型。

**[Seeing The Unseen A Frequency Prompt Guided Transformer For Image Restoration](image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)**

:   提出 FPro，通过频域视角的 prompt learning 指导图像复原：使用 Gated Dynamic Decoupler 将特征解耦为低频/高频分量，再通过 Dual Prompt Block（HPM + LPM）分别对两个频带注入可学习 prompt 并与解码器特征交互，在去雨、去雨滴、去摩尔纹、去模糊、去雾 5 个任务上全面超越 SOTA。

**[Spatially-Variant Degradation Model For Dataset-Free Super-Resolution](image_restoration/spatially-variant_degradation_model_for_dataset-free_super-resolution.md)**

:   提出首个无需数据集训练的空间变化退化模型 SVDSR，每个像素的退化核由可学习的原子核字典的线性组合表示，系数矩阵通过模糊集的隶属函数从图像纹理信息推导，在 MAP 框架下用 Monte Carlo EM 算法推断，$2\times$ 超分平均提升 1 dB。

**[Towards Real-World Event-Guided Low-Light Video Enhancement And Deblurring](image_restoration/towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)**

:   本文首次提出事件相机引导的低光视频增强与去模糊联合任务，构建了基于分光棱镜的真实世界数据集 RELED，并设计了包含事件引导可变形时序对齐 (ED-TFA) 和频谱滤波跨模态增强 (SFCM-FE) 两个核心模块的端到端框架，在 PSNR 上比此前最佳方法提升 1.2dB 以上。

**[Unrolled Decomposed Unpaired Learning For Controllable Low-Light Video Enhanceme](image_restoration/unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)**

:   提出 UDU-Net，将低光视频增强建模为 MAP 优化问题并展开为深度网络，通过 Intra/Inter 子网分别处理空间（光照）和时序（一致性）退化，支持无配对训练和人类感知反馈的可控增强。

---

## 📦 模型压缩 { #model_compression }

**[A Simple Lowbit Quantization Framework For Video Snapshot Co](model_compression/a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)**

:   首个面向视频快照压缩成像（Video SCI）重建任务的低比特量化框架Q-SCI，通过高质量特征提取模块、精确视频重建模块和Transformer分支的query/key分布偏移操作，在4-bit量化下实现7.8倍理论加速且性能仅下降2.3%。

**[Adaptive Compressed Sensing With Diffusionbased Posterior Sa](model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)**

:   提出AdaSense，利用预训练扩散模型的零样本后验采样来量化重建不确定性，从而自适应地选择最优测量矩阵，无需额外训练即可在人脸图像、MRI和CT等多领域实现优于非自适应方法的压缩感知重建。

**[Adaptive Selection Of Samplingreconstruction In Fourier Comp](model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)**

:   提出ℋ1.5框架：为每个输入数据自适应选择最佳采样mask-重建网络对（J=3对），利用超分辨率空间生成模型量化高频贝叶斯不确定性来决定采样策略，理论证明优于联合优化ℋ1（非自适应）和自适应采样ℋ2（Pareto次优）。

**[Anytime Continual Learning For Open Vocabulary Classification](model_compression/anytime_continual_learning_for_open_vocabulary_classification.md)**

:   提出 AnytimeCL 框架，通过部分微调 CLIP 最后一个 transformer block 并动态加权融合微调模型与原始模型的预测，实现任意时刻接收样本、任意标签集推理的开放词汇持续学习。

**[Bidirectional Stereo Image Compression With Cross-Dimensional Entropy Model](model_compression/bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)**

:   提出双向对称的立体图像压缩框架 BiSIC，采用 3D 卷积联合编解码器和跨维度熵模型，在 PSNR 和 MS-SSIM 上均超越传统标准和已有学习方法，同时消除了单向方法中左右视图压缩质量不平衡的问题。

**[Category Adaptation Meets Projected Distillation In Generalized Continual Catego](model_compression/category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)**

:   提出 CAMP 方法，通过可学习投影器蒸馏与类别中心适应网络的协同组合，在广义持续类别发现（GCCD）场景中显著提升了新类别学习与旧知识保持之间的平衡。

**[Freestyleret Retrieving Images From Style-Diversified Queries](model_compression/freestyleret_retrieving_images_from_style-diversified_queries.md)**

:   提出首个风格多样化查询图像检索（Style-Diversified QBIR）任务及数据集DSR，设计了轻量即插即用的FreestyleRet框架，通过Gram矩阵提取查询的纹理/风格特征，构建风格空间并以此初始化prompt token，使冻结的视觉编码器能适配文本、草图、低分辨率、艺术画等多种查询风格的检索。

**[Is Retain Set All You Need In Machine Unlearning Restoring Performance Of Unlear](model_compression/is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)**

:   提出 SCAR（Selective-distillation for Class and Architecture-agnostic unleaRning），一种无需保留集的近似遗忘算法，通过 Mahalanobis 距离引导遗忘样本特征向量向最近错误类分布迁移，并利用 OOD 图像蒸馏保持模型性能。

**[Isomorphic Pruning For Vision Models](model_compression/isomorphic_pruning_for_vision_models.md)**

:   提出 Isomorphic Pruning，通过将网络子结构建模为图并按图同构性分组，在同构组内独立排序剪枝，解决异构子结构间重要性不可比的问题，在 ViT 和 CNN 上均取得优于专门设计的剪枝方法的效果。

**[Leveraging Hierarchical Feature Sharing For Efficient Dataset Condensation](model_compression/leveraging_hierarchical_feature_sharing_for_efficient_dataset_condensation.md)**

:   提出层级记忆网络（HMN），将数据蒸馏中的合成数据存储为三层结构（数据集级-类级-实例级记忆），通过层级化特征共享提升存储效率，并利用实例级剪枝进一步去除冗余，仅用低GPU内存的 batch-based loss 即超越所有基线方法。

**[Metaaug Meta-Data Augmentation For Post-Training Quantization](model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)**

:   提出 MetaAug，一种基于元学习的训练后量化（PTQ）方法，通过可学习的变换网络对校准数据进行增强，并以双层优化框架同时优化变换网络和量化模型，有效缓解 PTQ 在小校准集上的过拟合问题。

**[Papr Training-Free One-Step Patch Pruning With Lightweight Convnets For Faster I](model_compression/papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)**

:   提出 PaPr，利用轻量级 ConvNet 的卷积特征图生成 Patch Significance Map (PSM)，在**无需重训练**的情况下对 ViT/ConvNet/混合架构进行**一步式** patch 剪枝，实现显著的计算量削减（视频场景最高 3.7× FLOPs 减少），且精度损失极小。

**[Simple Unsupervised Knowledge Distillation With Space Similarity](model_compression/simple_unsupervised_knowledge_distillation_with_space_similarity.md)**

:   CoSS 提出在无监督知识蒸馏中，除了常规的**特征维度余弦相似度**外，额外引入一个**空间维度余弦相似度（Space Similarity）**损失——将特征矩阵转置后在维度方向上对齐，从而弥补 $L_2$ 归一化导致的流形结构信息丢失，以极简的方式在多个 UKD benchmark 上达到 SOTA。

**[Spacejam A Lightweight And Regularization-Free Method For Fast Joint Alignment O](model_compression/spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)**

:   提出 SpaceJAM，一种仅约 16K 可训练参数的无监督图像联合对齐方法，无需正则化项或 atlas 维护，在 SPair-71K 和 CUB 数据集上匹配现有方法的对齐能力同时实现 10 倍以上加速。

**[Token Compensator Altering Inference Cost Of Vision Transformer Without Re-Tunin](model_compression/token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)**

:   提出 ToCom（Token Compensator），一个模型算术框架的轻量插件，通过快速的参数高效自蒸馏获得，可在推理时直接插入任意下游已训练模型以弥补 token 压缩度不匹配造成的性能损失，无需重新训练。

---

## 📊 LLM评测 { #llm_evaluation }

**[Colormnet A Memory-Based Deep Spatial-Temporal Feature Propagation Network For V](llm_evaluation/colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)**

:   提出 ColorMNet，一种基于记忆机制的时空特征传播网络，通过预训练大视觉模型引导的特征提取（PVGFE）、基于记忆的特征传播（MFP）和局部注意力（LA）三个模块，在显著降低 GPU 显存消耗（仅需 1.9G）的同时实现了优于 SOTA 的视频上色效果。

**[Deep Cost Ray Fusion For Sparse Depth Video Completion](llm_evaluation/deep_cost_ray_fusion_for_sparse_depth_video_completion.md)**

:   本文提出 RayFusion 框架，通过在 cost volume 上沿射线方向施加 self-attention 和 cross-attention 实现时序融合，以仅 1.15M 参数在 KITTI、VOID、ScanNetV2 三个数据集上全面超越或持平 SOTA 稀疏深度补全方法。

**[Distribution Alignment For Fully Test-Time Adaptation With Dynamic Online Data S](llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)**

:   提出分布对齐（DA）损失将测试时特征分布拉回源域分布，配合域偏移检测机制，在非 i.i.d. 动态数据流和连续域偏移场景下大幅超越现有 TTA 方法。

**[Eliminating Warping Shakes For Unsupervised Online Video Stitching](llm_evaluation/eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)**

:   定义了视频拼接中的"warping shake"新问题（图像拼接扩展到视频时非重叠区域的时域抖动），提出StabStitch首个无监督在线视频拼接框架，通过拼接轨迹生成与平滑同时实现视频拼接和稳定，达到实时28.2ms/帧。

**[Gradient-Regularized Out-Of-Distribution Detection](llm_evaluation/gradient-regularized_out-of-distribution_detection.md)**

:   提出 GReg/GReg+，通过正则化 OOD 评分函数的输入梯度范数来学习评分流形的局部平滑性，并结合基于能量评分的聚类采样策略选取高信息量辅助样本，在 CIFAR 和 ImageNet OOD 检测基准上取得 SOTA。

**[Image-Feature Weak-To-Strong Consistency An Enhanced Paradigm For Semi-Supervise](llm_evaluation/image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)**

:   本文提出 IFMatch，在传统图像级弱到强一致性范式基础上引入特征级扰动并构建三分支结构，通过置信度策略区分朴素/困难样本，在多个 SSL 基准上显著提升已有方法（如 FixMatch、FreeMatch 等）的性能。

**[Imaging Interiors An Implicit Solution To Electromagnetic Inverse Scattering Pro](llm_evaluation/imaging_interiors_an_implicit_solution_to_electromagnetic_inverse_scattering_pro.md)**

:   提出基于隐式神经表示（INR）的电磁逆散射问题（EISP）求解方案，通过将散射体的相对介电常数建模为连续隐式表示并在前向框架中优化，有效避免了逆估计的困难和离散化导致的低分辨率问题。

**[Ogni-Dc Robust Depth Completion With Optimization-Guided Neural Iterations](llm_evaluation/ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)**

:   提出 OGNI-DC，通过"优化引导的神经迭代"（OGNI）框架，结合 ConvGRU 迭代精炼深度梯度场和可微深度积分器（DDI）来实现深度补全，同时达到 SOTA 精度和强泛化能力。

**[Sigma Sinkhorn-Guided Masked Video Modeling](llm_evaluation/sigma_sinkhorn-guided_masked_video_modeling.md)**

:   本文提出 SIGMA，通过引入投影网络将 masked video modeling 的重建目标从像素级升级为可学习的深层特征聚类分配，利用 Sinkhorn 算法的最优传输实施高熵正则化避免坍缩，在 10 个数据集 3 个 benchmark 上全面超越 VideoMAE 等 SOTA 方法。

**[Versatile Incremental Learning Towards Class And Domain-Agnostic Incremental Lea](llm_evaluation/versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)**

:   首次定义 Versatile Incremental Learning (VIL) 场景——后续任务的类别或领域增量类型未知，并提出 ICON 框架，通过 CAST 损失控制学习方向避免与历史任务冲突、IC 增量分类器动态扩展输出节点处理跨域同类覆写问题，在三个基准上全面超越现有 CIL/DIL 方法。

**[Visfocus Prompt-Guided Vision Encoders For Ocr-Free Dense Document Understanding](llm_evaluation/visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)**

:   VisFocus提出了一种提示引导的视觉编码方法用于OCR-free文档理解：通过将用户提示（prompt）直接注入视觉编码器的patch merging层（ViLMA层），配合局部掩码提示建模（LMPM）预训练任务，使视觉编码器学会聚焦于与提示相关的文本区域，在多个文档VQA基准上达到同规模SOTA。

**[Visfocus Promptguided Vision Encoders For Ocrfree Dense](llm_evaluation/visfocus_promptguided_vision_encoders_for_ocrfree_dense.md)**

:   提出 VisFocus，通过在视觉编码器的 patch merging 层引入 prompt 感知的 ViLMA 层，并设计 LMPM 预训练任务，使 OCR-Free 文档理解模型能聚焦于与用户查询相关的文本区域，在多个文档 VQA 基准上达到同规模 SOTA。

---

## 🤖 机器人/具身智能 { #robotics }

**[Aff-Ttention Affordances And Attention Models For Short-Term Object Interaction ](robotics/aff-ttention_affordances_and_attention_models_for_short-term_object_interaction_.md)**

:   提出 STAformer 架构和两个基于 affordance 的模块（环境 affordance 数据库 + 交互热点），将第一人称视频中的短期物体交互预测（STA）在 Ego4D 和 EPIC-Kitchens 上提升了 30-45% 的相对性能。

**[Disco Embodied Navigation And Interaction](robotics/disco_embodied_navigation_and_interaction.md)**

:   提出 DISCO，通过可微分场景语义表征（包含物体和 affordance）实现动态场景建模，结合全局-局部双层粗到细控制策略实现高效移动操作，在 ALFRED benchmark 的 unseen scenes 上以 +8.6% 成功率超越使用分步指令的 SOTA，且无需分步指令。

**[Disco Embodied Navigation And Interaction Via Differentiable Scene Semantics And](robotics/disco_embodied_navigation_and_interaction_via_differentiable_scene_semantics_and.md)**

:   提出 DISCO 框架，通过可微分场景语义表示和双层粗-细动作控制，在 ALFRED 基准上实现具身导航与交互的显著性能提升（未见场景成功率超越 SOTA +8.6%，且无需逐步指令）。

**[Hierarchically Structured Neural Bones For Reconstructing Animatable Objects Fro](robotics/hierarchically_structured_neural_bones_for_reconstructing_animatable_objects_fro.md)**

:   提出层次化神经骨骼（Hierarchical Neural Bones）框架，通过树状结构的骨骼系统以粗到细的方式分解物体运动，从随手拍摄的视频中重建可操控的高质量 3D 模型。

**[Prioritized Semantic Learning For Zero-Shot Instance Navigation](robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)**

:   提出Prioritized Semantic Learning (PSL)方法，通过语义增强的Agent架构、优先语义训练策略和语义扩展推理方案，显著提升零样本目标/实例导航中Agent的语义感知能力，在ObjectNav和新提出的InstanceNav任务上实现SOTA。

**[Prioritized Semantic Learning For Zeroshot Instance Navigation](robotics/prioritized_semantic_learning_for_zeroshot_instance_navigation.md)**

:   提出 Prioritized Semantic Learning (PSL) 方法，通过语义感知智能体架构、优先语义训练策略和语义扩展推理方案，显著提升导航智能体的语义感知能力，在零样本 ObjectNav 上超越 SOTA  66%（SR），并提出了更具挑战性的 InstanceNav 任务。

**[Realfred An Embodied Instruction Following Benchmark In Photo-Realistic Environm](robotics/realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)**

:   提出 ReALFRED 基准，使用 150 个真实世界 3D 扫描的多房间可交互环境替代 ALFRED 的合成单房间场景，提供 30,696 条自由格式语言指令，揭示了现有具身指令跟随方法在真实环境中性能显著下降的问题。

**[See And Think Embodied Agent In Virtual Environment](robotics/see_and_think_embodied_agent_in_virtual_environment.md)**

:   提出 STEVE，一个基于视觉感知、语言指令和代码动作三大组件的 Minecraft 开放世界具身智能体，通过 STEVE-21K 数据集微调 LLaMA-2 并结合视觉编码器和技能数据库，在科技树解锁和方块搜索任务上大幅超越现有方法。

**[Semgrasp Semantic Grasp Generation Via Language Aligned](robotics/semgrasp_semantic_grasp_generation_via_language_aligned.md)**

:   提出 SemGrasp，通过层次化 VQ-VAE 将抓取姿态离散化为三个语义对齐的 token（方向/方式/精修），并微调多模态大语言模型实现基于语言指令的语义抓取生成。

**[Semgrasp Semantic Grasp Generation Via Language Aligned Discretization](robotics/semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)**

:   提出SemGrasp方法，设计层次化VQ-VAE将抓取姿态离散为"方向-方式-精修"三个语义token，然后微调多模态大语言模型(MLLM)在统一语义空间中融合物体、抓取与语言，实现根据自然语言指令生成物理合理且语义一致的人类抓取姿态。

---

## 🛡️ AI安全 { #ai_safety }

**[Any Target Can Be Offense Adversarial Example Generation Via Generalized Latent ](ai_safety/any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)**

:   提出 GAKer，首个可泛化到未知目标类别的定向对抗攻击生成器，通过在 UNet 中间层注入目标特征（latent infection）+ 余弦距离损失替代交叉熵实现类别无关训练，在未知类上的攻击成功率比 HGN 高 14.13%。

**[Clip-Guided Generative Networks For Transferable Targeted Adversarial Attacks](ai_safety/clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)**

:   提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。

**[Event Trojan Asynchronous Event-Based Backdoor Attacks](ai_safety/event_trojan_asynchronous_event-based_backdoor_attacks.md)**

:   提出 Event Trojan 框架，首次研究直接在异步事件数据流中注入后门触发器（immutable trigger 和 mutable trigger），揭示了事件相机视觉任务面临的后门攻击安全风险。

**[Genq Quantization In Low Data Regimes With Generative Synthetic Data](ai_safety/genq_quantization_in_low_data_regimes_with_generative_synthetic_data.md)**

:   提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。

**[Preventing Catastrophic Overfitting In Fast Adversarial Training A Bi-Level Opti](ai_safety/preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)**

:   从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。

**[Resilience Of Entropy Model In Distributed Neural Networks](ai_safety/resilience_of_entropy_model_in_distributed_neural_networks.md)**

:   首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平，准确率仅下降约 2%。

**[Skymask Attack-Agnostic Robust Federated Learning With Fine-Grained Learnable Ma](ai_safety/skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)**

:   提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。

**[Towards Multi-Modal Transformers In Federated Learning](ai_safety/towards_multi-modal_transformers_in_federated_learning.md)**

:   提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。

**[Towards Multimodal Transformers In Federated Learning](ai_safety/towards_multimodal_transformers_in_federated_learning.md)**

:   首次探索Transformer架构在转移式多模态联邦学习中的应用，提出FedCola框架，通过互补式本地训练（利用跨模态Transformer blocks）和协作式服务器聚合（选择性聚合self-attention层），在保护数据隐私的前提下有效训练多模态Transformer。

---

## 🎵 音频/语音 { #audio_speech }

**[Beat-It Beat-Synchronized Multi-Condition 3D Dance Generation](audio_speech/beat-it_beat-synchronized_multi-condition_3d_dance_generation.md)**

:   提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。

**[Coleaf A Contrastive-Collaborative Learning Framework For Weakly Supervised Audi](audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)**

:   提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。

**[Controlllm Augment Language Models With Tools](audio_speech/controlllm_augment_language_models_with_tools.md)**

:   提出 ControlLLM 框架，通过任务分解、Thoughts-on-Graph (ToG) 图搜索范式和执行引擎三大组件，让 LLM 在预构建的工具图上搜索最优解决方案路径，准确高效地调用多模态工具完成复杂任务，在困难任务上达到 93% 的解决方案成功率。

**[Controlllm Augment Language Models With Tools By Searching On Graphs](audio_speech/controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)**

:   提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

**[Edtalk Efficient Disentanglement For Emotional Talking Head Synthesis](audio_speech/edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)**

:   提出基于正交可学习基向量的高效解耦框架 EDTalk，将人脸动态分解为嘴型、头部姿态和情感表情三个独立潜空间，同时支持视频驱动和音频驱动的情感说话人头像生成。

**[Label-Anticipated Event Disentanglement For Audio-Visual Video Parsing](audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)**

:   提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。

**[Latent-Inr A Flexible Framework For Implicit Representations Of Videos With Disc](audio_speech/latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)**

:   提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。

**[Listen To Look Into The Future Audio-Visual Egocentric Gaze Anticipation](audio_speech/listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)**

:   提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。

**[Siamese Vision Transformers Are Scalable Audio-Visual Learners](audio_speech/siamese_vision_transformers_are_scalable_audio-visual_learners.md)**

:   提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Adaptive Multihead Contrastive Learning](self_supervised/adaptive_multihead_contrastive_learning.md)**

:   AMCL提出使用多个投影头（各自产生不同特征）+ 对每个样本对和每个头自适应学习温度参数，从最大似然估计推导出损失函数，作为通用插件在SimCLR/MoCo/Barlow Twins/CAN/LGP上一致提升1-5%性能。

**[Coho Context-Sensitive City-Scale Hierarchical Urban Layout Generation](self_supervised/coho_context-sensitive_city-scale_hierarchical_urban_layout_generation.md)**

:   提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。

**[Efficient Image Pre-Training With Siamese Cropped Masked Autoencoders](self_supervised/efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)**

:   提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。

**[Flowcon Out-Of-Distribution Detection Using Flow-Based Contrastive Learning](self_supervised/flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)**

:   提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。

**[Infmae A Foundation Model In The Infrared Modality](self_supervised/infmae_a_foundation_model_in_the_infrared_modality.md)**

:   提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。

**[Posformer Recognizing Complex Handwritten Mathematical Expression With Position ](self_supervised/posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)**

:   提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。

**[Rethinking Unsupervised Outlier Detection Via Multiple Thresholding](self_supervised/rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)**

:   提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。

**[Revisiting Supervision For Continual Representation Learning](self_supervised/revisiting_supervision_for_continual_representation_learning.md)**

:   挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。

**[Vic-Mae Self-Supervised Representation Learning From Images And Video With Contr](self_supervised/vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)**

:   ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。

---

## 💬 LLM/NLP { #llm_nlp }

**[Adaclip Adapting Clip With Hybrid Learnable Prompts For Zero](llm_nlp/adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)**

:   在CLIP中同时引入静态（全局共享）和动态（逐图生成）两种可学习提示，用辅助异常检测数据训练后，在14个工业+医学异常检测数据集上实现零样本SOTA，核心在于"任务级+实例级"双层自适应的混合提示设计。

**[Funqa Towards Surprising Video Comprehension](llm_nlp/funqa_towards_surprising_video_comprehension.md)**

:   构建了大规模反直觉视频问答基准 FunQA（4.3K 视频、312K QA 对），覆盖幽默/创意/魔术三类令人惊讶的视频，并提出 FunMentor 智能体通过多轮对话增强 VLM 的反常识推理能力。

**[Promptiqa Boosting The Performance And Generalization For No-Reference Image Qua](llm_nlp/promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)**

:   提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

**[Propose Assess Search Harnessing Llms For Goal-Oriented Planning In Instructiona](llm_nlp/propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)**

:   VidAssist提出"提议-评估-搜索"三步框架，利用LLM作为知识库和评估工具，结合广度优先搜索算法，在教学视频的目标导向规划任务中以零/少样本方式超越全监督SOTA，few-shot在COIN上比全监督VLaMP高+7.7% SR。

**[Reprojection Errors As Prompts For Efficient Scene Coordinate Regression](llm_nlp/reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)**

:   本文提出 EGFS（Error-Guided Feature Selection）机制，利用低重投影误差区域作为 SAM 的 point prompts 扩展为语义掩码，迭代地筛选可靠训练样本，在 Cambridge Landmarks 和 Indoor6 数据集上以更小模型和更少训练时间超越现有无 3D 信息依赖的 SCR 方法。

**[Stripe Observation Guided Inference Cost-Free Attention Mechanism](llm_nlp/stripe_observation_guided_inference_cost-free_attention_mechanism.md)**

:   本文通过深入分析Transformer中注意力权重矩阵的条纹（stripe）模式现象，提出一种推理阶段完全无额外计算开销的注意力增强机制——仅在训练阶段通过辅助模块学习条纹引导的注意力修正，并在推理时将其重参数化融入标准注意力权重中，实现"免费午餐"式的性能提升。

**[Zeroshot Object Counting With Good Exemplars](llm_nlp/zeroshot_object_counting_with_good_exemplars.md)**

:   提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Artvlm Attribute Recognition Through Vision-Based Prefix Language Modeling](information_retrieval/artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)**

:   本文提出将视觉属性识别问题重新建模为基于图像条件的前缀语言模型（PrefixLM）下的句子生成概率问题，通过"生成式检索"（Generative Retrieval）替代传统的"对比式检索"（Contrastive Retrieval），显式建模物体-属性间的条件依赖关系，在VAW和新提出的VGARank数据集上显著超越对比检索方法。

**[Artvlm Attribute Recognition Through Visionbased Prefix Lang](information_retrieval/artvlm_attribute_recognition_through_visionbased_prefix_lang.md)**

:   ArtVLM将属性识别定义为语言建模问题，PrefixLM生成式检索灵活建模物体-属性条件依赖。

**[Grounding Language Models For Visual Entity Recognition](information_retrieval/grounding_language_models_for_visual_entity_recognition.md)**

:   提出 AutoVER——首个将多模态大语言模型（MLLM）应用于大规模视觉实体识别的方法，通过将检索能力集成到 MLLM 内部，结合对比训练和前缀树约束解码，在 Oven-Wiki 基准上大幅超越 PaLI-17B 等先前方法。

**[Multi-Label Cluster Discrimination For Visual Representation Learning](information_retrieval/multi-label_cluster_discrimination_for_visual_representation_learning.md)**

:   提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。

**[Onerestore A Universal Restoration Framework For Composite Degradation](information_retrieval/onerestore_a_universal_restoration_framework_for_composite_degradation.md)**

:   提出 OneRestore，一种基于 Transformer 的通用图像复原框架，通过场景描述符引导的交叉注意力机制和复合退化复原损失，能在单一模型中自适应地处理低光照、雾、雨、雪及其任意组合的复合退化场景，并支持文本/视觉双模式的可控复原。

**[Towards Open-Ended Visual Recognition With Large Language Models](information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)**

:   提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。

---

## 📚 预训练/数据 { #llm_pretraining }

**[Cross-Domain Learning For Video Anomaly Detection With Limited Supervision](llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)**

:   提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。

**[Dragapart Learning A Part-Level Motion Prior For Articulated Objects](llm_pretraining/dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)**

:   DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。

**[Learning To Obstruct Few-Shot Image Classification Over Restricted Classes](llm_pretraining/learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)**

:   提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。

**[Prompting Language-Informed Distribution For Compositional Zero-Shot Learning](llm_pretraining/prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)**

:   本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。

**[Scaling Backwards Minimal Synthetic Pre-Training](llm_pretraining/scaling_backwards_minimal_synthetic_pre-training.md)**

:   提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。

**[Scantalk 3D Talking Heads From Unregistered Scans](llm_pretraining/scantalk_3d_talking_heads_from_unregistered_scans.md)**

:   提出 ScanTalk，首个能够对**任意拓扑**（包括未配准的3D扫描数据）的3D人脸进行语音驱动动画生成的深度学习框架，核心依赖于 DiffusionNet 的离散化无关特性来突破固定拓扑约束。

---

## 🛰️ 遥感 { #remote_sensing }

**[Adapting Fine-Grained Cross-View Localization To Areas Without Fine Ground Truth](remote_sensing/adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)**

:   针对细粒度跨视角定位模型在新区域部署时精度下降的问题，提出基于知识自蒸馏的弱监督学习方法——通过模式化伪GT生成、粗粒度监督和离群值过滤三个策略，仅使用目标区域的地面-航拍图像对（无需精确GT），即可在VIGOR和KITTI上将定位误差降低12%~20%。

**[Congeo Robust Cross-View Geo-Localization Across Ground View Variations](remote_sensing/congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)**

:   提出 ConGeo，一种模型无关的单视图+跨视图对比学习框架，通过强制同一地点不同地面视角变体之间的特征一致性，使单一模型即可在任意朝向和任意视场角(FoV)下实现鲁棒的跨视图地理定位。

**[Cross-Platform Video Person Reid A New Benchmark Dataset And Adaptation Approach](remote_sensing/cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach.md)**

:   构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。

**[Masked Angle-Aware Autoencoder For Remote Sensing Images](remote_sensing/masked_angle-aware_autoencoder_for_remote_sensing_images.md)**

:   提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。

**[Weakly-Supervised Camera Localization By Ground-To-Satellite Image Registration](remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)**

:   提出首个弱监督的地面-卫星图像配准定位方法，通过卫星-卫星自监督训练旋转估计器、对比学习训练平移估计器，在无需精确GT姿态标签的条件下实现最佳跨区域泛化能力，超越大多数全监督SOTA方法。

---

## 🕸️ 图学习 { #graph_learning }

**[Confidence Self-Calibration For Multi-Label Class-Incremental Learning](graph_learning/confidence_self-calibration_for_multi-label_class-incremental_learning.md)**

:   针对多标签类增量学习(MLCIL)中部分标签导致的过度自信预测和假阳性错误问题，提出 Confidence Self-Calibration (CSC) 框架，通过类增量图卷积网络(CI-GCN)校准标签关系 + 最大熵正则化校准置信度，在 MS-COCO 和 VOC 上大幅超越 SOTA。

**[Fine-Grained Scene Graph Generation Via Sample-Level Bias Prediction](graph_learning/fine-grained_scene_graph_generation_via_sample-level_bias_prediction.md)**

:   提出样本级偏置预测方法 SBP，通过 Bias-Oriented GAN 利用物体对 union region 的上下文信息预测样本特异性纠偏向量，将粗粒度关系修正为细粒度关系，在 VG/GQA/VG-1800 上相比数据集级纠偏方法平均提升 5.6%/3.9%/3.2% 的 Average@K。

**[Gkgnet Group K-Nearest Neighbor Based Graph Convolutional Network For Multi-Labe](graph_learning/gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)**

:   提出首个全图卷积多标签识别模型 GKGNet，通过 Group KNN 机制动态构建标签与图像区域间的图结构，在 MS-COCO 和 VOC2007 上以更低计算量取得 SOTA。

**[Senc Handling Self-Collision In Neural Cloth Simulation](graph_learning/senc_handling_self-collision_in_neural_cloth_simulation.md)**

:   提出 SENC，通过基于 Global Intersection Analysis (GIA) 的自碰撞损失和自碰撞感知图神经网络，首次在自监督神经布料模拟中有效解决布料自碰撞问题。

---

## 🔬 可解释性 { #interpretability }

**[Detailsemnet Elevating Signature Verification Through Detail-Semantic Integratio](interpretability/detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)**

:   提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。

**[Improving Intervention Efficacy Via Concept Realignment In Concept Bottleneck Mo](interpretability/improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)**

:   本文发现 Concept Bottleneck Models (CBMs) 中人工干预效率低下的原因在于干预时各概念独立处理、忽视了概念间关联，提出了一个轻量级的 Concept Intervention Realignment Module (CIRM)，在干预后自动重新对齐相关概念的预测值，将达到目标性能所需的干预次数最多减少 70%。

**[Plot Text-Based Person Search With Part Slot Attention For Corresponding Part Di](interpretability/plot_text-based_person_search_with_part_slot_attention_for_corresponding_part_di.md)**

:   提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。

**[Poa Pre-Training Once For Models Of All Sizes](interpretability/poa_pre-training_once_for_models_of_all_sizes.md)**

:   POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Adaglimpse Active Visual Exploration With Arbitrary Glimpse Position And Scale](reinforcement_learning/adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale.md)**

:   提出AdaGlimpse，利用Soft Actor-Critic强化学习从连续动作空间中选择任意位置和尺度的glimpse，结合弹性位置编码的ViT编码器实现多任务（重建/分类/分割）的主动视觉探索，以仅6%像素超越了使用18%像素的SOTA方法。

**[Octopus Embodied Vision-Language Programmer From Environmental Feedback](reinforcement_learning/octopus_embodied_vision-language_programmer_from_environmental_feedback.md)**

:   提出 Octopus，一个具身视觉-语言编程模型，通过生成可执行代码来连接高层规划与底层操控，并引入 Reinforcement Learning with Environmental Feedback (RLEF) 训练方案来提升决策质量。

**[Octopus Embodied Visionlanguage Programmer From Environmental Feedback](reinforcement_learning/octopus_embodied_visionlanguage_programmer_from_environmental_feedback.md)**

:   Octopus 是一个具身视觉-语言编程模型，通过将 VLM 与可执行代码生成相结合，利用 GPT-4 收集训练数据并引入 RLEF（环境反馈强化学习）进行微调，在三个不同模拟器（OmniGibson、Minecraft、GTA-V）中实现了端到端的视觉感知→计划→代码生成→执行闭环。

**[Visual Grounding For Object-Level Generalization In Reinforcement Learning](reinforcement_learning/visual_grounding_for_object-level_generalization_in_reinforcement_learning.md)**

:   利用视觉语言模型 (MineCLIP) 的 visual grounding 能力生成目标物体的 confidence map，通过奖励设计和任务表征两条路径将 VLM 知识迁移到强化学习中，实现对未见物体和指令的零样本泛化。

---

## 📡 信号/通信 { #signal_comm }

**[Pyra Parallel Yielding Re-Activation For Training-Inference Efficient Task Adapt](signal_comm/pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)**

:   提出PYRA方法同时实现训练高效和推理高效的任务适配，通过并行生成通道和token维度的自适应调制权重，在token合并前对特征进行re-activation校准，在ViT-L/16上1.7×加速仅掉0.1%精度、3×加速下消除"逆向压缩"现象。

**[Querycdr Query-Based Controllable Distortion Rectification Network For Fisheye I](signal_comm/querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)**

:   提出QueryCDR网络，通过可学习查询机制（DLQM）和两种可控调制模块（CCMB/CAMB），首次实现不同畸变程度的鱼眼图像在**不重训**的情况下进行高质量可控矫正。

**[Raw-Adapter Adapting Pre-Trained Visual Model To Camera Raw Images](signal_comm/raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)**

:   提出 RAW-Adapter，通过输入级适配器（可学习 ISP 阶段）和模型级适配器（ISP 中间特征注入骨干网络），以极小参数量（0.2-0.8M）将 sRGB 预训练模型高效适配到 Camera RAW 图像，在正常光/暗光/过曝等多种光照条件下的检测和分割任务上达到 SOTA。

**[Unsupervised Exposure Correction](signal_comm/unsupervised_exposure_correction.md)**

:   提出首个无监督曝光校正（UEC）方法，利用ISP管线自由生成的多曝光序列让图像互为ground truth进行训练，设计仅含19K参数的像素级变换函数保留图像细节，在曝光校正和下游边缘检测上超越有监督SOTA。

---

## 🦾 LLM Agent { #llm_agent }

**[Agent3D-Zero An Agent For Zero-Shot 3D Understanding](llm_agent/agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)**

:   Agent3D-Zero 提出一个基于 VLM 的零样本 3D 场景理解 Agent 框架，通过鸟瞰图上的 Set-of-Line 视觉提示引导 VLM 主动选择观察视角，并综合多视角图像进行 3D 推理，在 ScanQA 等任务上超越了需要微调的 3D-LLM 方法。

**[Hydra A Hyper Agent For Dynamic Compositional Visual Reasoning](llm_agent/hydra_a_hyper_agent_for_dynamic_compositional_visual_reasoning.md)**

:   （注：基于摘要的简要笔记）提出 HYDRA，一种多阶段动态组合式视觉推理框架，通过规划器（Planner）、强化学习认知控制器（RL Agent）和推理器（Reasoner）三模块协作，实现可靠且渐进式的视觉推理，在 RefCOCO/RefCOCO+、OK-VQA、GQA 等多个数据集上取得 SOTA。

**[Videoagent A Memory-Augmented Multimodal Agent For Video Understanding](llm_agent/videoagent_a_memory-augmented_multimodal_agent_for_video_understanding.md)**

:   提出 VideoAgent，一个记忆增强的多模态 Agent，通过构建结构化记忆（temporal memory 存储事件描述 + object memory 存储物体跟踪状态）并利用 4 个工具与记忆交互，零样本完成长视频问答任务，在 NExT-QA 上平均 +6.6%、EgoSchema 上 +26.0%，接近 Gemini 1.5 Pro 的性能。

---

## 💡 LLM推理 { #llm_reasoning }

**[Controllable Navigation Instruction Generation](llm_reasoning/controllable_navigation_instruction_generation.md)**

:   提出 C-Instructor，利用 Chain-of-Thought with Landmarks (CoTL) 机制引导 LLM 先识别关键地标再生成指令，结合空间拓扑建模任务 (STMT) 和风格混合训练 (SMT)，实现风格可控和内容可控的导航指令生成，在四个室内外 benchmark 上全面超越 SOTA。

**[Controllable Navigation Instruction Generation With Chain Of Thought Prompting](llm_reasoning/controllable_navigation_instruction_generation_with_chain_of_thought_prompting.md)**

:   提出 C-Instructor，利用 LLM 的思维链提示实现风格和内容可控的导航指令生成，通过 CoTL（带地标的思维链）、STMT（空间拓扑建模）和 SMT（混合风格训练）三大机制，在四个室内外导航数据集上全面超越已有方法。

**[Roadpainter Points Are Ideal Navigators For Topology Transformer](llm_reasoning/roadpainter_points_are_ideal_navigators_for_topology_transformer.md)**

:   提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。

---

## 🗣️ 对话系统 { #dialogue }

**[Bi-Mdrg Bridging Image History In Multimodal Dialogue Response Generation](dialogue/bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)**

:   提出 BI-MDRG 框架，通过桥接图像历史信息来增强多模态对话中文本回复的图像 grounding 能力和连续图像回复中物体的一致性。

**[Bimdrg Bridging Image History In Multimodal Dialogue Respons](dialogue/bimdrg_bridging_image_history_in_multimodal_dialogue_respons.md)**

:   BI-MDRG通过视觉交叉注意力和Citation Module桥接图像历史增强多模态对话。

---

## 👥 社会计算 { #social_computing }

**[Distribution-Aware Robust Learning From Long-Tailed Data With Noisy Labels](social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)**

:   提出 DaSC 框架，通过分布感知的类中心估计（DaCC）和置信度感知的对比学习（SBCL + MIDL），同时解决长尾分布和噪声标签的联合问题，在 CIFAR 和真实噪声数据集上达到 SOTA。

**[Grace Graph-Based Contextual Debiasing For Fair Visual Question Answering](social_computing/grace_graph-based_contextual_debiasing_for_fair_visual_question_answering.md)**

:   提出 GRACE（GRAph-based Contextual DEbiasing），一种基于图结构的上下文去偏方法，通过无监督上下文图学习和基于图的多样化 in-context example 选择，解决知识增强 VQA 系统中大语言模型继承的数据偏差问题。

---

## 📈 时间序列 { #time_series }

**[Omnisat Self-Supervised Modality Fusion For Earth Observation](time_series/omnisat_self-supervised_modality_fusion_for_earth_observation.md)**

:   提出OmniSat统一框架，通过模态特异编码器+跨模态对比自监督预训练，将多光谱时序（S2）、SAR时序（S1）、高分辨率单时相（SPOT/Aerial）等异构遥感数据融合为统一表示，在语义分割和作物分类上超越所有单模态和多模态基线。

**[Semantically Guided Representation Learning For Action Anticipation](time_series/semantically_guided_representation_learning_for_action_anticipation.md)**

:   提出 S-GEAR 框架，通过学习视觉动作原型并利用语言模型的语义关联来引导原型之间的几何关系，使模型理解动作间的语义互联性，从而提升动作预测性能，在 Epic-Kitchens 55/100、EGTEA Gaze+、50 Salads 四个基准上取得 SOTA 或极具竞争力的结果。

---

## 🔗 因果推理 { #causal_inference }

**[Distill Gold From Massive Ores Bi-Level Data Pruning Towards Efficient Dataset D](causal_inference/distill_gold_from_massive_ores_bi-level_data_pruning_towards_efficient_dataset_d.md)**

:   提出双层数据剪枝策略 BiLP，通过经验损失静态剪枝和基于因果效应 (ITE) 的动态剪枝，高效选择对数据集蒸馏最有价值的真实样本，以即插即用方式一致性提升现有蒸馏方法性能并降低计算开销。

---

## 🌍 地球科学 { #earth_science }

**[Semi-Supervised Video Desnowing Network Via Temporal Decoupling Experts And Dist](earth_science/semi-supervised_video_desnowing_network_via_temporal_decoupling_experts_and_dist.md)**

:   提出首个半监督视频去雪框架 SemiVDN，通过物理先验引导的时序解耦专家模块和分布驱动的对比正则化，利用无标签真实雪景视频缩小合成-真实域差距，在合成与真实数据集上均超越现有方法。

---

## 🔒 LLM安全 { #llm_safety }

**[Magr Manifold-Aligned Graph Regularization For Continual Action Quality Assessme](llm_safety/magr_manifold-aligned_graph_regularization_for_continual_action_quality_assessme.md)**

:   提出 MAGR 方法，通过流形对齐投影器和 Intra-Inter-Joint 图正则化器，解决持续动作质量评估（CAQA）中特征回放导致的旧特征与当前特征流形不对齐问题，在四个数据集上显著超越现有基线。

---

## 📐 优化/理论 { #optimization }

**[Handling The Non-Smooth Challenge In Tensor Svd A Multi-Objective Tensor Recover](optimization/handling_the_non-smooth_challenge_in_tensor_svd_a_multi-objective_tensor_recover.md)**

:   提出基于可学习张量核范数的多目标张量恢复框架 (MOTC)，通过引入可学习酉矩阵替代固定变换来解决 t-SVD 方法在非光滑张量数据上的性能退化问题，并通过多目标优化有效利用张量各维度的低秩性。

---

## ⚛️ 物理学 { #physics }

**[Robust Fitting On A Gate Quantum Computer](physics/robust_fitting_on_a_gate_quantum_computer.md)**

:   首次在真实门量子计算机（IonQ Aria）上实现鲁棒拟合：提出用于一维 $\ell_\infty$ 可行性检验的量子电路，填补了 Bernstein-Vazirani（BV）电路计算 Boolean influence 的关键空缺，并展示如何将一维 influence 累积到高维非线性模型（如基础矩阵估计）。

---

## 🎁 推荐系统 { #recommender }

**[Aid-Appeal Automatic Image Dataset And Algorithm For Content Appeal Enhancement ](recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)**

:   首次提出图像内容吸引力评估（ICAA）任务，区别于传统美学评估（IAA），设计了一套自动化数据集生成 + 吸引力估计 + 吸引力增强的完整 pipeline，用 Stable Diffusion + Textual Inversion 实现零人工标注的大规模数据集构建。

---

## 📂 其他 { #others }

**[A Closer Look At Gan Priors Exploiting Intermediate Features](others/a_closer_look_at_gan_priors_exploiting_intermediate_features.md)**

:   提出 IF-GMI，将预训练 StyleGAN2 的生成器拆解为多个 block，在中间特征层逐层优化（配合 $\ell_1$ 球约束防止图像崩塌），把模型反演攻击的搜索空间从潜码扩展到中间特征，在 OOD 场景下攻击准确率提升高达 38.8%。

**[A Framework For Efficient Model Evaluation Through Stratific](others/a_framework_for_efficient_model_evaluation_through_stratific.md)**

:   提出一个统计框架，通过分层（stratification）、采样设计（sampling）和估计器（estimation）三个组件的协同设计，在仅标注少量测试样本的情况下精确估计CV模型准确率，最高可实现10倍的效率增益（即用1/10的标注量达到同等精度）。

**[A Highquality Robust Diffusion Framework For Corrupted Datas](others/a_highquality_robust_diffusion_framework_for_corrupted_datas.md)**

:   提出 RDUOT 框架，首次将非平衡最优传输(UOT)融入扩散模型(DDGAN)中，通过学习 $q(x_0|x_t)$ 而非 $q(x_{t-1}|x_t)$ 来有效过滤训练数据中的离群值，在污染数据集上实现鲁棒生成的同时，在干净数据集上也超越了 DDGAN 基线。

**[Abc Easy As 123 A Blind Counter For Exemplar-Free Multi-Class Class-Agnostic Cou](others/abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)**

:   提出首个无需样例图像即可同时计数图像中多类未知物体的方法ABC123，通过ViT回归多通道密度图+匈牙利匹配训练+SAM示例发现机制，在自建合成数据集MCAC上大幅超越需要样例的方法，且能泛化到FSC-147真实数据集。

**[Action2Sound Ambientaware Generation Of Action Sounds From E](others/action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[Active Generation For Image Classification](others/active_generation_for_image_classification.md)**

:   ActGen将主动学习思想引入扩散模型数据增强，通过识别分类器的错分样本并以注意力掩码引导+梯度对抗引导生成"难样本"，仅用10%的合成数据量即超越了此前需要近等量合成数据的方法，在ImageNet上ResNet-50获得+2.26%的精度提升。

**[Adaptive Highfrequency Transformer For Diverse Wildlife Reid](others/adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)**

:   提出自适应高频Transformer（AdaFreq），通过频域混合增强、目标感知的高频token动态选择、特征均衡损失三大策略，将高频信息（毛皮纹理、轮廓边缘等）统一用于多种野生动物的重识别，在8个跨物种数据集上超越现有ReID方法。

**[Bidirectional Uncertainty-Based Active Learning For Open-Set Annotation](others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)**

:   提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。

**[Dc-Solver Improving Predictor-Corrector Diffusion Sampler Via Dynamic Compensati](others/dc-solver_improving_predictor-corrector_diffusion_sampler_via_dynamic_compensati.md)**

:   提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。

**[Et The Exceptional Trajectories Text-To-Camera-Trajectory Generation With Charac](others/et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)**

:   提出首个从真实电影中提取的**相机-角色轨迹数据集 E.T.**（115K 样本，11M 帧），以及基于扩散模型的 **Director** 方法，能根据文本描述和角色轨迹生成复杂的相机运动轨迹，同时设计了 **CLaTr** 对比嵌入用于轨迹生成质量评估。

**[Event-Based Mosaicing Bundle Adjustment](others/event-based_mosaicing_bundle_adjustment.md)**

:   提出 EMBA，首个针对纯旋转事件相机的光度 Bundle Adjustment 方法，利用线性化事件生成模型将问题形式化为正则化非线性最小二乘优化，并利用法方程矩阵的块对角稀疏结构设计高效求解器，同时优化相机旋转轨迹和全景梯度图。

**[Foster Adaptivity And Balance In Learning With Noisy Labels](others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)**

:   提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。

**[Freeaugment Data Augmentation Search Across All Degrees Of Freedom](others/freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)**

:   提出 FreeAugment，首个能够同时全局优化数据增强策略的四个自由度（变换数量/类型/顺序/强度）的全可微搜索方法，通过 Gumbel-Softmax 学习深度分布、Gumbel-Sinkhorn 学习排列分布来避免重复采样，在多个基准上取得 SOTA。

**[Hpff Hierarchical Locally Supervised Learning With Patch Feature Fusion](others/hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)**

:   提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。

**[Intrinsic Single-Image Hdr Reconstruction](others/intrinsic_single-image_hdr_reconstruction.md)**

:   > 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。

**[Mahalanobis Distance-Based Multi-View Optimal Transport For Multi-View Crowd Loc](others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)**

:   提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。

**[Online Temporal Action Localization With Memory-Augmented Transformer](others/online_temporal_action_localization_with_memory-augmented_transformer.md)**

:   提出 MATR（Memory-Augmented Transformer），通过记忆队列存储过去片段的特征来利用长期上下文，并采用分离的 Start/End Transformer 解码器进行动作实例定位，在在线时序动作定位（On-TAL）任务上取得 SOTA，甚至可比肩部分离线方法。

**[Partcraft Crafting Creative Objects By Parts](others/partcraft_crafting_creative_objects_by_parts.md)**

:   提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。

**[Power Variable Projection For Initialization-Free Large-Scale Bundle Adjustment](others/power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)**

:   提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。

**[Raindrop Clarity A Dual-Focused Dataset For Day And Night Raindrop Removal](others/raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)**

:   提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。

**[Shifted Autoencoders For Point Annotation Restoration In Object Counting](others/shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)**

:   提出**Shifted AutoEncoders (SAE)**，一种受MAE启发的点标注修复方法：通过随机位移点标注后训练UNet恢复，使模型学到"通用位置知识"而忽略个体标注噪声；用训练好的SAE修复原始标注使其更一致，可为任意计数模型（密度图/定位型）稳定提升性能，在9个数据集上创下新记录。

**[Spatio-Temporal Proximity-Aware Dual-Path Model For Panoramic Activity Recogniti](others/spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)**

:   提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。

**[Superpixel-Informed Implicit Neural Representation For Multi-Dimensional Data](others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)**

:   提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。

**[Teaching Tailored To Talent Adverse Weather Restoration](others/teaching_tailored_to_talent_adverse_weather_restoration.md)**

:   提出 T3-DiffWeather，采用 prompt pool 自主组合子 prompt 构建天气退化信息，结合 Depth-Anything 约束的通用 prompt 提供场景信息，以对比 prompt 损失约束两类 prompt，在恶劣天气图像恢复任务上仅用 WeatherDiffusion 十分之一的采样步数达到 SOTA。

**[Teaching Tailored To Talent Adverse Weather Restoration Via Prompt Pool And Dept](others/teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)**

:   提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。
