---
title: >-
  ICCV2025 3D视觉方向 292篇论文解读
description: >-
  292篇ICCV2025 3D视觉方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D视觉

**📹 ICCV2025** · 共 **292** 篇

**[2D Gaussian Splattingbased Sparseview Transparent Object Dep](2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)**

:   提出TRAN-D，一种基于2D Gaussian Splatting的稀疏视角透明物体深度重建方法，通过分割引导的object-aware损失优化遮挡区域Gaussian分布，并利用物理仿真（MPM）实现物体移除后的场景动态更新，仅需单张图像即可完成场景刷新。

**[3D Gaussian Map With Openset Semantic Grouping For Visionlan](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)**

:   提出基于3D高斯溅射的场景地图表示（3D Gaussian Map），结合开放集语义分组机制，为视觉-语言导航（VLN）构建兼顾几何结构与丰富语义的3D环境表示，并设计多层级动作预测策略（Multi-Level Action Prediction）融合多粒度空间-语义线索辅助导航决策。

**[3D Mesh Editing Using Masked Lrms](3d_mesh_editing_using_masked_lrms.md)**

:   提出MaskedLRM，将3D形状编辑重构为条件重建问题——训练时随机生成3D遮挡物遮盖多视角输入，用一张干净条件视图引导被遮挡区域的补全；推理时用户定义编辑区域并提供单张编辑图像，模型在**<3秒单次前传**中完成3D网格编辑，比优化方法快2-10倍，能执行拓扑变化编辑（加孔/加把手），重建质量与SOTA持平。

**[3D Test-Time Adaptation Via Graph Spectral Driven Point Shift](3d_test-time_adaptation_via_graph_spectral_driven_point_shift.md)**

:   提出 GSDTTA，将3D点云测试时自适应从空间域转移到图谱域，仅优化最低10%频率分量即可适配点云的全局结构，配合特征图引导的自训练策略，在 ModelNet40-C 和 ScanObjectNN-C 上达到 SOTA。

**[3D Testtime Adaptation Via Graph Spectral Driven Point Shift](3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)**

:   提出GSDTTA，首次将3D点云的测试时适应从空间域转移到图谱域，通过仅优化最低10%频率分量（减少约90%参数）实现全局结构调整，并结合特征图引导的自训练策略生成伪标签，在ModelNet40-C和ScanObjectNN-C上显著超越现有3D TTA方法。

**[3Dgraphllm Combining Semantic Graphs And Large Language Mode](3dgraphllm_combining_semantic_graphs_and_large_language_mode.md)**

:   提出3DGraphLLM，首个将**3D语义场景图的可学习表示**直接输入LLM的方法——通过k近邻子图+三元组(object1, relation, object2)编码物体间语义关系，然后投影到LLM的token嵌入空间。在ScanRefer上Acc@0.5提升+6.4%（vs无语义关系的Chat-Scene），在Multi3DRefer上F1@0.5提升+7.5%，推理速度比GPT4Scene-HDM快5倍。

**[3Dgraphllm Combining Semantic Graphs And Large Language Models For 3D Scene Unde](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)**

:   本文提出3DGraphLLM，将3D场景中物体间的语义关系编码为可学习的图表示并输入LLM，在object grounding、场景描述和视觉问答等多个3D视觉-语言任务上显著超越不使用语义关系的基线方法，同时推理速度比LVLM方法快5倍。

**[3Dgs-Lm Faster Gaussian-Splatting Optimization With Levenberg-Marquardt](3dgs-lm_faster_gaussian-splatting_optimization_with_levenberg-marquardt.md)**

:   本文提出3DGS-LM，用定制的Levenberg-Marquardt优化器替换3DGS中的ADAM优化器，通过高效的GPU缓存驱动并行化方案实现Jacobian-向量积的快速计算，在保持相同重建质量的前提下将3DGS优化速度提升20%。

**[3Dgslm Faster Gaussiansplatting Optimization With Levenbergm](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)**

:   将3D Gaussian Splatting的ADAM优化器替换为定制化的Levenberg-Marquardt（LM）二阶优化器，通过高效CUDA并行化的PCG算法和梯度缓存结构实现Jacobian-向量积加速，在保持相同重建质量的前提下将优化时间缩短约20%。

**[4D Gaussian Splatting Slam](4d_gaussian_splatting_slam.md)**

:   提出首个完整的4D Gaussian Splatting SLAM系统，在动态场景中同时进行相机位姿跟踪和4D高斯辐射场重建——将高斯原语分为静态/动态集合，通过稀疏控制点+MLP建模动态物体运动，并创新性地设计2D光流图渲染算法来监督动态高斯的运动学习。

**[4D Visual Pre-Training For Robot Learning](4d_visual_pre-training_for_robot_learning.md)**

:   FVP提出了一种基于4D（3D空间+时间）点云预测的视觉预训练框架，通过将预训练目标建模为"下一帧点云预测"并用扩散模型实现，显著提升了多种3D模仿学习方法在真实机器人操作任务上的成功率（DP3平均提升28%）。

**[4D Visual Pretraining For Robot Learning](4d_visual_pretraining_for_robot_learning.md)**

:   FVP提出将3D视觉预训练建模为"下一帧点云预测"问题，用条件扩散模型从历史帧点云预测未来帧点云来学习3D视觉表示，在12个真实世界操作任务中将DP3的平均成功率提升28%，达到SOTA水平。

**[7Dgs Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)**

:   将3DGS扩展到7维（空间3D+时间1D+方向3D），通过条件切片机制将7D高斯投影为与3DGS管线兼容的3D高斯，在具有视角依赖效果的动态场景上PSNR提升最高7.36dB，同时维持401 FPS实时渲染。

**[7Dgs Unified Spatialtemporalangular Gaussian Splatting](7dgs_unified_spatialtemporalangular_gaussian_splatting.md)**

:   提出7DGS，将场景元素建模为**7维高斯分布**（3D空间+1D时间+3D视角方向），通过条件切片机制将7D高斯转换为与时间和视角相关的条件3D高斯，统一处理动态场景+视角依赖效果，在自定义7DGS-PBR数据集上比4DGS PSNR提升高达7.36dB，仅用15.3%的高斯点数，401FPS实时渲染。

**[A3Gs Arbitrary Artistic Style Into Arbitrary 3D Gaussian Spl](a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)**

:   提出A³GS，首个**前馈式零样本3DGS风格迁移**网络——使用图卷积网络(GCN)自编码器将3DGS场景编码到潜在空间，通过AdaIN注入任意风格图像特征，仅需**10秒**即可将任意风格迁移到任意3D场景，无需逐风格优化，可处理大规模3DGS场景。

**[A Lesson In Splats Teacher-Guided Diffusion For 3D Gaussian Splats Generation Wi](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)**

:   本文提出了一种仅使用2D图像监督来训练3D扩散模型的新框架——通过将确定性3D重建模型作为"噪声教师"生成3D噪声样本，并结合多步去噪策略和循环一致性正则化，实现了超越教师模型的3D高斯喷溅生成质量（PSNR提升0.5-0.85）。

**[A Lesson In Splats Teacherguided Diffusion For 3D Gaussian S](a_lesson_in_splats_teacherguided_diffusion_for_3d_gaussian_s.md)**

:   提出一种用2D图像监督训练3D扩散模型的框架：利用预训练的确定性3D重建模型作为"噪声教师"生成3D噪声样本，通过多步去噪策略和渲染损失实现跨模态（3D去噪+2D监督）训练，在用更小模型的情况下超越教师模型0.5-0.85 PSNR。

**[A Recipe For Generating 3D Worlds From A Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)**

:   将单图到3D世界生成分解为两个更简单的子问题——全景合成（无训练in-context learning）和点云条件修复（仅5k步微调ControlNet），结合3DGS重建出可在VR中2米立方体范围内导航的沉浸式3D环境，在图像质量指标上全面超越WonderJourney和DimensionX等SOTA方法。

**[A Simple Yet Mighty Hartley Diffusion Versatilist For Genera](a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)**

:   提出HarDiff——基于离散Hartley变换的频域学习策略，通过低频训练（从源域提取结构先验）和高频采样（利用目标域细节引导）增强扩散模型在稠密视觉任务上的跨域泛化能力，在语义分割、深度估计和去雾等12个基准上取得SOTA。

**[A Unified Interpretation Of Training-Time Out-Of-Distribution Detection](a_unified_interpretation_of_training-time_out-of-distribution_detection.md)**

:   从输入变量间"交互"的新视角出发，统一解释了不同训练时 OOD 检测方法为何有效——它们都促使模型编码更多高阶交互，并进一步验证了高阶交互在 OOD 检测中的主导作用，以及 near-OOD 样本难以检测的交互分布原因。

**[Aaa-Gaussians Anti-Aliased And Artifact-Free 3D Gaussian Rendering](aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)**

:   AAA-Gaussians提出了一种统一的3D高斯光栅化框架，通过自适应3D平滑滤波器、视空间透视正确边界计算和基于视锥体的3D裁剪，在单一框架内同时解决了3DGS的锯齿、投影畸变和闪烁三大顽疾，在分布外视角评估中大幅领先其他方法，同时保持实时渲染性能。

**[Aaagaussians Antialiased And Artifactfree 3D Gaussian Render](aaagaussians_antialiased_and_artifactfree_3d_gaussian_render.md)**

:   通过在3DGS渲染管线的所有环节中融入完整的3D评估（而非2D splat近似），提出自适应3D平滑滤波器、视空间边界计算和基于视锥的tile剔除，统一解决了3DGS中的锯齿、投影伪影和弹出伪影（popping），在OOD视角下大幅优于现有方法，同时保持实时渲染（>100 FPS）。

**[Accelerate 3D Object Detection Models Via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)**

:   提出 tgGBC（trim keys gradually Guided By Classification scores），一种零样本运行时剪枝方法，利用分类分数与注意力图的乘积计算键重要性，逐层剪除不重要的键，在多个3D检测器上实现Transformer解码器近2×加速且性能损失<1%。

**[Adahuman Animatable Detailed 3D Human Generation With Compos](adahuman_animatable_detailed_3d_human_generation_with_compos.md)**

:   提出AdaHuman框架，通过姿态条件的联合3D扩散模型（在扩散过程中同步进行多视角图像生成与3DGS重建以保证3D一致性）和组合式3DGS细化模块（利用crop-aware camera ray map融合局部精细细节），从单张野外图片生成高保真可动画的3D人体avatar，在重建和重姿态任务上全面超越现有SOTA。

**[Adahuman Animatable Detailed 3D Human Generation With Compositional Multiview Di](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)**

:   提出AdaHuman框架，通过姿态条件化的3D联合扩散模型和组合式3DGS细化模块，从单张图片生成高精度、可动画化的3D人体虚拟人。

**[Advancing Text-To-3D Generation With Linearized Lookahead Variational Score Dist](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)**

:   通过分析 VSD 中 LoRA 模型与 3D 模型的优化顺序不匹配问题，提出线性化前瞻（Linearized Lookahead）修正项 $L^2$-VSD，仅需额外一次前向传播即可显著提升 text-to-3D 生成质量。

**[Adversarial Exploitation Of Data Diversity Improves Visual L](adversarial_exploitation_of_data_diversity_improves_visual_l.md)**

:   提出RAP（Robust Absolute Pose regression）——基于外观感知3DGS的双分支联合训练框架，通过对抗判别器弥合合成-真实域差距+外观/位姿增强数据作为额外监督，在Cambridge Landmarks上平移/旋转误差分别降低38-50%/41-44%，在日夜场景和驾驶场景中表现尤为突出。

**[Adversarial Exploitation Of Data Diversity Improves Visual Localization](adversarial_exploitation_of_data_diversity_improves_visual_localization.md)**

:   提出RAP框架，通过外观可变的3DGS合成多样化训练数据，并引入对抗判别器弥合合成-真实域差距，使绝对姿态回归方法在多个数据集上大幅超越SOTA——室内平移/旋转误差降低50%/41%，室外降低38%/44%。

**[Ajahr Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)**

:   首个面向截肢者的3D人体网格恢复框架——通过合成100万+截肢者图像(A3D)、设计BPAC-Net截肢分类器区分截肢与遮挡、以及双Tokenizer切换策略分别编码截肢/正常位姿先验，在截肢者数据上大幅领先(ITW-amputee上MVE比TokenHMR低16.87)，非截肢者数据上也保持竞争力。

**[Amodal3R Amodal 3D Reconstruction From Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)**

:   提出Amodal3R，一个端到端的遮挡感知3D重建模型，通过在TRELLIS基础上引入mask加权交叉注意力和遮挡感知注意力层，直接在3D潜空间中从部分遮挡的2D图像重建完整的3D物体形状和外观，大幅超越先前"2D补全→3D重建"的两阶段方法。

**[Amodal Depth Anything Amodal Depth Estimation In The Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)**

:   提出非模态相对深度估计新范式，构建大规模真实数据集ADIW（564K），基于Depth Anything V2和DepthFM设计两个互补框架（Amodal-DAV2和Amodal-DepthFM），通过最小化修改预训练模型实现遮挡区域深度预测，在ADIW上RMSE比之前SOTA提升27.4%。

**[Animateanymesh A Feedforward 4D Foundation Model For Textdri](animateanymesh_a_feedforward_4d_foundation_model_for_textdri.md)**

:   提出AnimateAnyMesh，首个前馈式文本驱动通用Mesh动画框架：通过DyMeshVAE将动态Mesh分解为初始位置和相对轨迹并压缩到潜空间，再用基于Rectified Flow的MMDiT模型学习文本条件下的轨迹分布，配合4M+规模的DyMesh数据集训练，在6秒内即可为任意拓扑Mesh生成高质量动画，全面碾压DG4D、L4GM和Animate3D。

**[Anyi2V Animating Any Conditional Image With Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)**

:   提出AnyI2V，一个无需训练的框架，可接受任意模态图像（mesh、点云、深度图、骨架等）作为首帧条件，结合用户定义的轨迹实现运动控制的视频生成，在FID/FVD/ObjMC指标上优于现有training-free方法并与训练方法竞争。

**[Ar1To3 Single Image To Consistent 3D Object Via Nextview Pre](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)**

:   提出AR-1-to-3，一种基于扩散模型的自回归下一视角预测框架，通过"先近后远"的渐进式生成策略，配合Stacked-LE（堆叠局部特征编码）和LSTM-GE（全局特征编码）两种条件注入机制，显著提升了单图到多视角生成的一致性，在GSO数据集上PSNR达13.18（相比InstantMesh的10.67提升23.5%），Chamfer Distance降至0.063（InstantMesh为0.117）。

**[Articulate3D Holistic Understanding Of 3D Scenes As Universa](articulate3d_holistic_understanding_of_3d_scenes_as_universa.md)**

:   提出Articulate3D（280个真实室内场景、8类铰接标注的大规模数据集）和USDNet（基于Mask3D扩展的统一框架），通过密集逐点预测机制同时完成可动零件分割和运动参数估计，在铰接参数预测上比Mask3D†提升5.7%，并支持LLM场景编辑和机器人策略训练。

**[Articulate3D Holistic Understanding Of 3D Scenes As Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)**

:   本文提出Articulate3D——首个大规模真实世界室内场景铰接标注数据集（280个高质量扫描），以及USDNet统一框架，能从3D点云同时预测可移动/可交互部件分割和运动参数，为具身AI的物理仿真提供了simulation-ready的场景数据。

**[Atlas Decoupling Skeletal And Shape Parameters For Expressiv](atlas_decoupling_skeletal_and_shape_parameters_for_expressiv.md)**

:   提出ATLAS参数化人体模型，通过显式解耦外部表面形状和内部骨骼参数，并引入稀疏非线性姿态校正变形，在60万高分辨率扫描数据上训练，实现了比SMPL-X更精确、更可控的3D人体建模。

**[Atlas Decoupling Skeletal And Shape Parameters For Expressive Parametric Human M](atlas_decoupling_skeletal_and_shape_parameters_for_expressive_parametric_human_m.md)**

:   提出 ATLAS 参数化人体模型，通过显式解耦外部表面形状与内部骨骼参数，结合稀疏非线性姿态校正，在 60 万高分辨率扫描上训练，实现比 SMPL-X 更精确可控的人体建模。

**[Auto-Regressively Generating Multi-View Consistent Images](auto-regressively_generating_multi-view_consistent_images.md)**

:   提出 MV-AR，首次将自回归模型引入多视图图像生成，利用所有先前视图作为条件逐步生成后续视图，配合统一的多模态条件注入模块和 Shuffle View 数据增强，在文本/图像/形状条件下均达到与扩散模型可比的一致性。

**[Autoocc Automatic Openended Semantic Occupancy Annotation Vi](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)**

:   提出AutoOcc，一个以视觉为中心的全自动开放式语义占据标注流水线，通过视觉-语言模型引导的可微高斯泼溅（VL-GS）实现无需人工标签的3D语义占据生成，在Occ3D-nuScenes上以纯视觉输入就达到IoU 83.01/mIoU 20.92，大幅超越现有自动标注方法。

**[Back On Track Bundle Adjustment For Dynamic Scene Reconstruc](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruc.md)**

:   提出BA-Track框架，通过学习型3D点追踪器将观测到的运动解耦为相机引起的运动和物体自身运动，使传统束调整(BA)能够无差别地处理静态和动态点，在相机位姿估计(ATE在Sintel上达到0.034，较SOTA降低一半以上)和稠密3D重建上取得显著提升。

**[Back On Track Bundle Adjustment For Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)**

:   提出 BA-Track 框架，利用 3D 点追踪器将观测运动分解为相机运动和物体运动，使传统 Bundle Adjustment 能同时处理静态与动态场景元素，实现精确的相机位姿估计和时间一致的稠密重建。

**[Baking Gaussian Splatting Into Diffusion Denoiser For Fast A](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)**

:   提出DiffusionGS，将3D高斯点云直接嵌入扩散模型的去噪器中，通过单阶段3D扩散实现从单张图片到3D物体生成和场景重建，在ABO/GSO上PSNR超越SOTA 2.20/1.25 dB，RealEstate10K上超1.34 dB，推理速度约6秒（A100）。

**[Baking Gaussian Splatting Into Diffusion Denoiser For Fast And Scalable Single-S](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)**

:   提出DiffusionGS，将3D高斯点云"烘焙"进扩散模型的去噪器中，实现单阶段、视图一致的单视图3D物体生成和场景重建，配合场景-物体混合训练策略和RPPC相机条件编码，在PSNR/FID上大幅超越现有方法，推理速度仅需约6秒。

**[Banet Bilateral Aggregation Network For Mobile Stereo Matchi](banet_bilateral_aggregation_network_for_mobile_stereo_matchi.md)**

:   提出双边聚合网络BANet，通过将代价体分离为高频细节体和低频平滑体分别聚合再融合，仅使用2D卷积即可在移动设备上实现实时高精度立体匹配（骁龙8 Gen 3上45ms，KITTI 2015 D1-all=1.83%，比MobileStereoNet-2D精度高35.3%）。

**[Banet Bilateral Aggregation Network For Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)**

:   提出双边聚合网络BANet，通过空间注意力将代价体分离为高频细节体和低频平滑体并分别聚合，仅使用2D卷积即可在移动设备上实时运行并大幅超越MobileStereoNet-2D（KITTI 2015上精度提升35.3%），3D版本在GPU上达到实时方法最高精度。

**[Benchmarking And Learning Multi-Dimensional Quality Evaluator For Text-To-3D Gen](benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)**

:   构建了包含1280个文本生成3D模型的多维度基准MATE-3D（8类prompt × 8种方法 × 4维评分 × 21名标注者），并提出基于超网络的多维度质量评估器HyperScore，通过条件特征融合和自适应映射在所有评估维度上超越现有指标。

**[Benchmarking And Learning Multidimensional Quality Evaluator](benchmarking_and_learning_multidimensional_quality_evaluator.md)**

:   构建MATE-3D基准（8类prompt×8种方法=1280个textured mesh，4维度×21人主观评分=107520标注）并提出HyperScore多维质量评估器：通过可学习条件特征+条件特征融合(模拟注意力转移)+超网络生成维度自适应映射函数(模拟决策过程变化)，在语义对齐、几何、纹理、整体4个维度上全面超越现有指标。

**[Benchmarking Egocentric Visualinertial Slam At City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)**

:   提出 LaMAria——首个城市尺度的第一人称多传感器 VIO/SLAM 基准数据集，利用测绘级控制点提供厘米精度的地面真值，系统评估了学术界主流 SLAM 方案在真实第一人称场景下的表现，揭示了现有方法与商业系统之间的巨大差距。

**[Beziergs Dynamic Urban Scene Reconstruction With Bezier Curv](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)**

:   用可学习的Bézier曲线显式建模动态物体的运动轨迹，替代传统依赖精确bbox标注的范式，实现了对自动驾驶街景中动/静态成分的准确分离与高保真重建。

**[Beziergs Dynamic Urban Scene Reconstruction With Bezier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)**

:   提出用可学习的Bézier曲线建模动态物体运动轨迹的3D高斯溅射方法（BezierGS），摆脱对精确目标标注框的依赖，在Waymo和nuPlan数据集上的动态和静态场景重建均达到SOTA。

**[Billboard Splatting Bbsplat Learnable Textured Primitives Fo](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)**

:   提出BBSplat——用可学习的RGB纹理和alpha贴图替代2D Gaussian Splatting中的高斯分布不透明度，使每个平面基元具有任意形状和逐像素颜色控制，在用更少基元的情况下弥补2DGS与3DGS之间的渲染质量差距，同时保留精确网格提取能力并实现最高×17的存储压缩。

**[Blended Point Cloud Diffusion For Localized Textguided Shape](blended_point_cloud_diffusion_for_localized_textguided_shape.md)**

:   提出 BlendedPC，将局部文本引导的3D形状编辑重新定义为语义inpainting问题，通过在Point·E基础上训练Inpaint-E模型，并在推理时引入无需反演(inversion-free)的坐标混合(coordinate blending)机制，在保持原始形状身份的同时实现精准局部编辑，在ShapeTalk数据集上全面超越现有方法。

**[Bokehdiff Neural Lens Blur With One-Step Diffusion](bokehdiff_neural_lens_blur_with_one-step_diffusion.md)**

:   BokehDiff提出基于预训练扩散模型的单步推理散景渲染方法，通过物理启发的自注意力模块（PISA）融入能量守恒、弥散圆约束和自遮挡效果，配合合成前景数据训练，在深度不连续区域显著优于传统方法。

**[Bolt3D Generating 3D Scenes In Seconds](bolt3d_generating_3d_scenes_in_seconds.md)**

:   提出一种基于潜在扩散模型的前馈式3D场景生成方法，通过将3D场景表示为多组Splatter Image并使用专门训练的几何VAE，**在单GPU上7秒内生成完整3D场景**，推理成本比优化式方法（CAT3D）降低300倍。

**[Boost 3D Reconstruction Using Diffusion-Based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)**

:   提出 DM-Calib，利用 Stable Diffusion 先验进行单目相机内参估计，设计了 Camera Image 表示将内参无损编码为图像，结合 RANSAC 解算焦距和光心，在5个零样本数据集上大幅超越现有标定方法，并推进了度量深度估计、位姿估计和稀疏视图重建等下游任务。

**[Boost 3D Reconstruction Using Diffusionbased Monocular Camer](boost_3d_reconstruction_using_diffusionbased_monocular_camer.md)**

:   提出DM-Calib——基于扩散模型的单目相机内参估计方法：设计Camera Image表示（将内参无损编码为3通道图像=方位角+仰角+灰度图），微调Stable Diffusion生成Camera Image，用RANSAC提取内参，在5个零样本数据集上超越所有基线，并将相机标定扩展到度量深度估计、位姿估计和稀疏视角3D重建。

**[Boosting Multi-View Indoor 3D Object Detection Via Adaptive 3D Volume](boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume.md)**

:   提出SGCDet框架，通过几何与上下文感知聚合模块（自适应特征提升）和稀疏体素构建策略（粗到细的自适应体素选择），在不依赖GT场景几何的前提下，实现了高效且高精度的多视图室内3D目标检测。

**[Boosting Multi-View Indoor 3D Object Detection Via Adaptive 3D Volume Constructi](boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume_constructi.md)**

:   SGCDet 通过自适应稀疏3D体素构建和几何-上下文感知聚合，实现了高效精准的多视图室内3D目标检测，无需真实几何监督即超越现有方法。

**[Boosting Multiview Indoor 3D Object Detection Via Adaptive 3](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)**

:   SGCDet通过几何与上下文感知的聚合模块（3D可变形注意力+多视角注意力融合）和基于占据概率的稀疏体素构建策略，在无需ground-truth几何监督的情况下，实现了多视角室内3D目标检测的SOTA性能，同时大幅降低计算开销。

**[Bootstrap3D Improving Multi-View Diffusion Model With Synthetic Data](bootstrap3d_improving_multi-view_diffusion_model_with_synthetic_data.md)**

:   提出 Bootstrap3D 框架，利用 2D/视频扩散模型自动生成 100 万张高质量多视角图像配精细文本描述，并通过训练时间步重调度（TTR）策略在微调多视角扩散模型时平衡图像质量与视角一致性，显著提升文本到 3D 生成的质量。

**[Bootstrap3D Improving Multiview Diffusion Model With Synthet](bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)**

:   提出Bootstrap3D框架，利用视频扩散模型生成合成多视图数据，并通过微调的MV-LLaVA进行质量过滤与密集描述重写，结合Training Timestep Reschedule (TTR)策略训练多视图扩散模型，在不牺牲视图一致性的前提下大幅提升图像质量和文本对齐能力。

**[Boxdreamer Dreaming Box Corners For Generalizable Object Pos](boxdreamer_dreaming_box_corners_for_generalizable_object_pos.md)**

:   提出以3D包围盒角点作为中间表示，通过Transformer解码器预测查询视图中角点的2D投影热图，结合PnP算法实现可泛化的稀疏视角6DoF物体位姿估计，在遮挡和稀疏视角场景下显著优于现有方法。

**[Boxdreamer Dreaming Box Corners For Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)**

:   提出 BoxDreamer，以 3D 包围盒角点作为中间表示，通过基于参考视角的点合成器预测查询图像中的 2D 角点投影，建立 2D-3D 对应关系后用 PnP 算法恢复物体位姿，在稀疏视角和严重遮挡场景下显著优于现有方法。

**[Bridging 3D Anomaly Localization And Repair Via High-Qualit](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)**

:   提出 PASDF 框架，通过姿态感知的签名距离函数（SDF）实现连续几何表征，统一了3D异常检测与修复任务，在 Real3D-AD 和 Anomaly-ShapeNet 上取得 SOTA。

**[Bridging 3D Anomaly Localization And Repair Via High-Quality Continuous Geometri](bridging_3d_anomaly_localization_and_repair_via_high-quality_continuous_geometri.md)**

:   提出 PASDF 框架，通过姿态感知的签名距离函数（SDF）实现连续几何表征，统一了3D异常检测与修复任务，在 Real3D-AD 和 Anomaly-ShapeNet 上取得 SOTA。

**[Bridging 3D Anomaly Localization And Repair Via Highquality](bridging_3d_anomaly_localization_and_repair_via_highquality.md)**

:   提出PASDF框架，通过姿态对齐模块(PAM)将点云对齐到标准姿态 + 神经SDF网络学习连续几何表示 + 基于SDF偏差的异常评分，统一实现3D点云异常检测与异常修复(Marching Cubes提取零等值面作为修复模板)，在Real3D-AD上O-AUROC 80.2%、Anomaly-ShapeNet上90.0%均达SOTA。

**[Bridging Diffusion Models And 3D Representations A 3D Consis](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)**

:   提出3DSR——将扩散超分模型与3DGS表示交替迭代实现3D一致超分：每步去噪后将SR图像训练到3DGS中获得3D一致渲染→重编码回潜在空间引导下一步去噪，无需微调任何模型即显式保证跨视角一致性，在LLFF上PSNR提升1.16dB+FID降低50%(vs StableSR)。

**[Bridging Diffusion Models And 3D Representations A 3D Consistent Super-Resolutio](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)**

:   提出3DSR框架，将扩散模型的2D超分辨率与3D高斯溅射（3DGS）表示相结合，在每个扩散去噪步骤中通过3DGS渲染来强制多视图3D一致性，实现高保真且空间一致的3D场景超分辨率。

**[Bring Your Rear Cameras For Egocentric 3D Human Pose Estimat](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimat.md)**

:   首次研究HMD后置相机对全身姿态追踪的价值，提出Transformer-based多视角热力图精炼模块(利用可变形注意力+不确定性感知遮罩)，解决后视角2D关节检测不可靠的问题，并发布两个大规模数据集(Ego4View-Syn/RW)，在Ego4View-RW上MPJPE比SOTA EgoPoseFormer提升>10%(63.38→56.94mm)。

**[Bring Your Rear Cameras For Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)**

:   首次研究HMD后置相机对自中心3D全身姿态估计的价值，提出基于Transformer的多视图热图细化方法，结合不确定性感知掩码机制，在新建的Ego4View数据集上实现>10% MPJPE提升。

**[Buffer-X Towards Zero-Shot Point Cloud Registration In Diverse Scenes](buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)**

:   提出 BUFFER-X，一种无需人工参数调优的零样本点云配准方法，通过自适应体素大小/搜索半径估计、FPS 替代学习型关键点检测器、以及 patch 级坐标归一化，在 11 个数据集上实现开箱即用的跨域泛化。

**[Bufferx Towards Zeroshot Point Cloud Registration In Diverse](bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)**

:   通过几何自适应bootstrapping确定体素大小/搜索半径、用FPS替代学习型关键点检测器、以及patch级坐标归一化，构建了一个无需人工调参即可在11个跨域数据集上实现零样本点云配准的pipeline BUFFER-X，在室内外多传感器多场景下取得了平均排名第一的成功率。

**[Ca-I2P Channel-Adaptive Registration Network With Global Optimal Selection](ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)**

:   提出 CA-I2P，通过 Channel Adaptive Adjustment Module (CAA) 增强并过滤图像-点云特征的通道差异，并用 Global Optimal Selection (GOS) 基于最优传输替代 top-k 选择减少多对一匹配误差，在 RGB-D Scenes V2 和 7-Scenes 上实现图像-点云配准 SOTA。

**[Cad-Recode Reverse Engineering Cad Code From Point Clouds](cad-recode_reverse_engineering_cad_code_from_point_clouds.md)**

:   提出 CAD-Recode，将点云翻译为可执行的 Python CadQuery 代码来重建 CAD 模型，利用预训练 LLM（Qwen2-1.5B）作为解码器配合轻量级点云编码器，在 DeepCAD、Fusion360 和 CC3D 三个基准上实现了 10 倍以上的 Chamfer Distance 降低。

**[CAD-Recode: Reverse Engineering CAD Code from Point Clouds](cadrecode_reverse_engineering_cad_code_from_point_clouds.md)**

:   将CAD sketch-extrude序列表示为Python代码，利用轻量级点云投影器 + 预训练LLM解码器将点云翻译为可执行Python代码来重建CAD模型，在DeepCAD/Fusion360/真实世界CC3D数据集上显著超越现有方法，且输出代码可被通用LLM理解用于CAD编辑和问答。

**[Can3Tok Canonical 3D Tokenization And Latent Modeling Of Sce](can3tok_canonical_3d_tokenization_and_latent_modeling_of_sce.md)**

:   提出Can3Tok——首个场景级3DGS VAE：通过cross-attention将大量(40K)无序3D Gaussian压缩到低维canonical token(256×768→64×64×4) + 3DGS归一化解决跨场景尺度不一致 + 语义感知过滤去除floater噪声，在DL3DV-10K上唯一成功的场景级3DGS潜在建模方法(L2=30.1, 失败率2.5%)，支持text-to-3DGS和image-to-3DGS前馈生成。

**[Can3Tok Canonical 3D Tokenization And Latent Modeling Of Scene-Level 3D Gaussian](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)**

:   提出 Can3Tok，首个可将场景级3DGS编码到低维潜空间的变分自编码器，通过规范化查询（canonical query）的交叉注意力实现高效tokenization，配合3DGS归一化和语义感知过滤解决尺度不一致问题，在DL3DV-10K上成功泛化到新场景。

**[Casp Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspo](casp_improving_semi-dense_feature_matching_pipeline_leveraging_cascaded_correspo.md)**

:   提出 CasP，一种级联匹配流水线，将匹配阶段分解为 1/16 尺度的一对多先验匹配和 1/8 尺度的一对一精细匹配，在保持精度的同时实现最高 2.2× 加速，并显著提升跨域泛化能力。

**[Catsplat Contextaware Transformer With Spatial Guidance For](catsplat_contextaware_transformer_with_spatial_guidance_for.md)**

:   提出CATSplat——单视图前馈3DGS重建的泛化Transformer框架：利用VLM文本嵌入（上下文先验）和3D点云特征（空间先验）通过双重cross-attention增强图像特征，在RE10K等数据集上在PSNR/SSIM/LPIPS全面超越Flash3D，且跨数据集泛化性优异。

**[Cf3 Compact And Fast 3D Feature Fields](cf3_compact_and_fast_3d_feature_fields.md)**

:   提出 CF³ 管线，通过 top-down 特征提升、per-Gaussian 自编码器压缩和自适应稀疏化，仅使用原始 Gaussian 数量的 5% 即可构建紧凑高速的 3D 特征场，实现 121–245× 的存储压缩和实时渲染。

**[Charm3R Towards Unseen Camera Height Robust Monocular 3D Det](charm3r_towards_unseen_camera_height_robust_monocular_3d_det.md)**

:   通过数学推导发现回归深度和地面深度在相机高度变化时呈现方向相反的误差趋势，CHARM3R 直接在模型内对两种深度做简单平均来抵消趋势，从而大幅提升单目3D检测器对未见相机高度的泛化能力（CARLA 上提升超过 45%）。

**[Charm3R Towards Unseen Camera Height Robust Monocular 3D Detector](charm3r_towards_unseen_camera_height_robust_monocular_3d_detector.md)**

:   通过数学证明回归深度和地平面深度在相机高度变化时具有相反的外推趋势，提出CHARM3R在模型内简单平均两种深度估计来抵消趋势，实现Mono3D对未见相机高度的鲁棒泛化，AP3D提升超过45%。

**[Cl-Splats Continual Learning Of Gaussian Splatting With Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)**

:   提出 CL-Splats，一种基于 3D Gaussian Splatting 的持续学习框架，通过 DINOv2 变化检测、2D→3D 掩码提升和球体约束的局部优化，从稀疏新视图高效增量更新场景重建，在合成和真实场景上大幅超越 CL-NeRF 等方法（PSNR：40.1 vs 30.1 dB），并支持历史恢复和并发更新等应用。

**[Clip-Gs Unifying Vision-Language Representation With 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)**

:   提出 CLIP-GS，首个基于 3D Gaussian Splatting (3DGS) 的多模态表示学习框架。通过 GS Tokenizer 将 3DGS 序列化为 token，结合图像投票损失 (Image Voting Loss) 进行多模态对齐，在跨模态检索、零样本和少样本 3D 分类任务上全面超越基于点云的方法。

**[Cmt A Cascade Mar With Topology Predictor For Multimodal Conditional Cad Generat](cmt_a_cascade_mar_with_topology_predictor_for_multimodal_conditional_cad_generat.md)**

:   提出 CMT，首个基于 B-Rep 表示的多模态 CAD 生成框架，通过级联 MAR（先边后面）和拓扑预测器实现精确拓扑和几何生成，并构建了 130 万级多模态 CAD 数据集 mmABC。

**[Comogaussian Continuous Motionaware Gaussian Splatting From](comogaussian_continuous_motionaware_gaussian_splatting_from.md)**

:   用Neural ODE建模曝光时间内的连续相机运动轨迹，结合刚体变换和可学习的连续运动修正(CMR)变换，从运动模糊图像重建清晰3D高斯场景，在所有benchmark上大幅超越SOTA。

**[Compression Of 3D Gaussian Splatting With Optimized Feature Planes And Standard ](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)**

:   本文提出 CodecGS，通过将 3DGS 的所有高斯属性用紧凑的 Tri-plane 特征平面表示，并结合频率域 DCT 熵建模和通道级比特分配策略，使特征平面能高效利用标准视频编解码器（HEVC）压缩，实现在保持高渲染质量的同时将存储大小减少至约10MB以内（相比原始3DGS压缩比高达146×）。

**[Constraint-Aware Feature Learning For Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)**

:   提出首个面向参数化点云的约束感知特征学习方法 CstNet，将 CAD 约束编码为点级别的 MAD-Adj-PT 三元组表示，通过两阶段网络（约束获取 + 约束特征学习）在自建的 Param20K 数据集上实现分类精度 +3.49%、旋转鲁棒性 +26.17% 的 SOTA 提升。

**[Contact-Aware Amodal Completion For Human-Object Interaction Via Multi-Regional ](contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)**

:   提出首个面向人物交互（HOI）场景的非模态补全框架，利用人体拓扑和接触信息通过凸包操作识别遮挡区域，结合多区域修复策略在预训练扩散模型上无需额外训练即可完成高质量的遮挡物体补全。

**[Curve-Aware Gaussian Splatting For 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)**

:   提出 CurveGaussian，通过在参数曲线与边缘导向高斯原语之间建立双向耦合机制，实现从多视图边缘图直接端到端优化 3D 参数曲线的一阶段方法，消除了两阶段管线的误差累积，在精度、效率和紧凑性上全面超越先前方法。

**[Cuts3D Cutting Semantics In 3D For 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)**

:   提出CutS3D方法，首次将3D信息（单目深度估计）引入无监督实例分割，通过在3D点云中切割语义区域来分离2D中重叠的实例，并引入空间置信度机制提升伪标签质量，在多个基准上超越CutLER等SoTA。

**[Dap-Mae Domain-Adaptive Point Cloud Masked Autoencoder For E](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_e.md)**

:   提出 DAP-MAE，通过异构域适配器（HDA）和域特征生成器（DFG）协同学习多域点云数据，仅需一次预训练即可适配物体分类、表情识别、部件分割和3D检测等多种下游任务。

**[Dap-Mae Domain-Adaptive Point Cloud Masked Autoencoder For Effective Cross-Domai](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)**

:   提出 DAP-MAE，通过异构域适配器（HDA）和域特征生成器（DFG）协同学习多域点云数据，仅需一次预训练即可适配物体分类、表情识别、部件分割和3D检测等多种下游任务。

**[Dapmae Domainadaptive Point Cloud Masked Autoencoder For Eff](dapmae_domainadaptive_point_cloud_masked_autoencoder_for_eff.md)**

:   提出一种域自适应点云MAE框架（DAP-MAE），通过异构域适配器（HDA）和域特征生成器（DFG）两个模块，让一次跨域预训练即可在物体分类、人脸表情识别、部件分割、目标检测等多个不同域的下游任务上都达到SOTA。

**[David Data-Efficient And Accurate Vision Models From Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)**

:   证明通过高保真**程序化合成数据**即可训练出精度媲美基础模型（如 Sapiens-2B）的以人为中心的稠密预测模型，仅需 **30 万合成图像**、**0.3B 参数**、训练成本不到同级方案的 1/16，在深度估计、表面法线估计、软前景分割三项任务上实现 SOTA 或近 SOTA 性能。

**[Deepmesh Auto-Regressive Artist-Mesh Creation With Reinforcement Learning](deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)**

:   提出 DeepMesh 框架，通过改进的高效mesh tokenization算法（72%压缩率）和首次将DPO强化学习引入3D网格生成来实现人类偏好对齐，能够生成最高3万面的高质量Artist-like三角网格。

**[Degauss Dynamic-Static Decomposition With Gaussian Splatting For Distractor-Free](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)**

:   提出 DeGauss，一种基于解耦的动态-静态高斯泼溅的自监督框架，通过前景动态高斯和背景静态高斯的概率掩码组合，实现从随意捕获的图像集到高度动态的自我中心视频的广泛场景下的无干扰 3D 重建。

**[Demeter A Parametric Model Of Crop Plant Morphology From The Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)**

:   Demeter 是一个数据驱动的参数化植物形态模型，将植物形态分解为拓扑、关节、形状和变形四个因素，支持形状生成、3D 重建和生物物理仿真。

**[Depth Anyevent A Cross-Modal Distillation Paradigm For Event-Based Monocular Dep](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)**

:   提出跨模态蒸馏范式，利用图像域的视觉基础模型（Depth Anything v2）生成伪标签来训练事件相机深度估计网络，并设计了基于 VFM 的循环架构 DepthAnyEvent-R，在无需昂贵深度标注的情况下实现了事件相机单目深度估计的 SOTA 性能。

**[Describe Adapt And Combine Empowering Clip Encoders For Open-Set 3D Object Retri](describe_adapt_and_combine_empowering_clip_encoders_for_open-set_3d_object_retri.md)**

:   提出 DAC 框架，通过 "描述-适配-组合" 三步策略协同 CLIP 与多模态大语言模型 (MLLM)，仅使用多视图图像即可在开放集 3D 物体检索任务上大幅超越此前使用全模态（点云+体素+图像）的 SOTA 方法，平均 mAP 提升超过 +10%。

**[Diorama Unleashing Zero-Shot Single-View 3D Indoor Scene Modeling](diorama_unleashing_zero-shot_single-view_3d_indoor_scene_modeling.md)**

:   提出Diorama，首个零样本开放世界系统，从单张RGB图像通过模块化管线（开放世界感知+基于CAD的场景建模）生成完整的3D室内场景，包含建筑结构和物体摆放，无需端到端训练或人工标注。

**[Diorama Unleashing Zeroshot Singleview 3D Indoor Scene Model](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)**

:   提出首个零样本开放世界系统 Diorama，通过模块化地组合 foundation model（GPT-4o、SAM、DinoV2、Metric3D 等），将单张 RGB 图像转化为包含建筑结构和 CAD 物体的完整可组合 3D 室内场景，无需任何端到端训练或人工标注。

**[Discretized Gaussian Representation For Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)**

:   提出离散化高斯表示（DGR）用于 CT 重建，通过离散化高斯函数直接端到端重建 3D 体素，并设计高度并行化的快速体积重建技术，在稀疏视角和有限角度 CT 场景中以零训练数据超越深度学习和实例重建方法。

**[Disentangling Instance And Scene Contexts For 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)**

:   提出 DISC，一种基于类别感知的双流架构用于 3D 语义场景补全，通过将实例类别和场景类别解耦到独立的查询流中并设计针对性的解码模块，在 SemanticKITTI 上仅用单帧输入即超越多帧 SOTA 方法，实例类别 mIoU 提升 17.9%。

**[Diving Into The Fusion Of Monocular Priors For Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)**

:   深入分析单目先验融合中的三大问题（仿射不变性 vs 绝对深度的对齐、迭代更新中的局部最优、噪声视差对融合的干扰），提出二元局部排序图和全局配准模块，在 SceneFlow→Middlebury/Booster 泛化实验中将 bad2 错误减半甚至更多，且几乎不增加计算开销。

**[Dmesh An Efficient Differentiable Mesh For Complex Shapes](dmesh_an_efficient_differentiable_mesh_for_complex_shapes.md)**

:   本文提出DMesh++，通过Minimum-Ball算法替代加权Delaunay三角剖分实现可微网格的tessellation函数，将计算复杂度从 $O(N)$ 降至 $O(\log N)$，在处理复杂形状时速度提升最高32倍，同时保持无自交叉和少薄三角形的优良特性。

**[Do It Yourself Learning Semantic Correspondence From Pseudo-Labels](do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)**

:   本文提出 DIY-SC，通过3D感知的伪标签生成策略（链式传播+松弛循环一致性+球面原型过滤）训练轻量适配器来改进基础模型特征的语义对应能力，在 SPair-71k 上实现了超越先前 SOTA 4.5%（PCK@0.1 per-keypoint）的性能，且无需人工关键点标注。

**[Driving View Synthesis On Free-Form Trajectories With Generative Prior](driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)**

:   提出驾驶视图合成框架DriveX，通过渐进式地将视频扩散模型的生成先验蒸馏到3DGS表示中——设计inpainting-based视频修复任务来生成新轨迹伪标注，迭代优化3D重建，实现自由轨迹上的高质量实时渲染。

**[Dso Aligning 3D Generators With Simulation Feedback For Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)**

:   提出 Direct Simulation Optimization (DSO) 框架，利用物理仿真器的（非可微）稳定性反馈作为奖励信号，通过 DPO 或新提出的 DRO 目标函数微调 3D 生成器，使其前馈式地直接输出物理上自支撑的 3D 物体，无需测试时优化。

**[Dynamic Point Maps A Versatile Representation For Dynamic 3D Reconstruction](dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)**

:   提出 Dynamic Point Maps (DPM)，将 DUSt3R 的视点不变点图扩展为同时控制视点和时间的时空不变表示，仅通过预测4组点图即可在前馈方式下同时解决深度估计、场景流、运动分割和3D目标跟踪等多种4D任务。

**[Easi3R Estimating Disentangled Motion From Dust3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)**

:   提出 Easi3R，一种无需训练的即插即用方法，通过解耦 DUSt3R 注意力层中隐含编码的相机运动与物体运动信息，实现动态视频的4D重建、运动分割和相机位姿估计。

**[Easy3D A Simple Yet Effective Method For 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)**

:   提出 Easy3D，一种简洁高效的 3D 交互式实例分割方法，结合体素稀疏编码器、轻量 Transformer 解码器和隐式点击融合策略，在域内和域外数据集上一致性地超越 SOTA，并首次将学习的负嵌入 (learned negative embedding) 成功应用于隐式点击融合。

**[Efficient Spiking Point Mamba For Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)**

:   SPM（Spiking Point Mamba）提出首个基于 Mamba 的 3D 脉冲神经网络框架，通过层次化动态编码（HDE）和脉冲 Mamba 模块（SMB），在大幅降低能耗（3.5× 以上）的同时，在 ScanObjectNN 上比前 SOTA SNN 方法提升 6-7% 的准确率。

**[Egocentric Action-Aware Inertial Localization In Point Clouds With Vision-Langua](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)**

:   EAIL 框架利用头戴式 IMU 信号中的第一人称动作线索，通过层次化多模态对齐（视觉-语言引导）学习动作与环境结构的关联，在 3D 点云中实现精确的惯性定位，同时附带动作识别能力。

**[Egom2P Egocentric Multimodal Multitask Pretraining](egom2p_egocentric_multimodal_multitask_pretraining.md)**

:   EgoM2P 是首个面向自我中心(egocentric)4D理解的多模态多任务大模型，通过时序感知的掩码建模框架统一处理 RGB 视频、深度、注视和相机轨迹四种模态，在多个下游任务上匹配或超越专用模型且快一个数量级。

**[Embodiedsplat Personalized Real-To-Sim-To-Real Navigation With Gaussian Splats F](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)**

:   提出 EmbodiedSplat，一个利用 iPhone 手机拍摄视频 → 3D 高斯溅射重建 mesh → 在 Habitat-Sim 中微调导航策略 → 部署到真实世界的完整流程，在真实场景 ImageNav 任务上比零样本基线提升 20%-40% 绝对成功率，sim-vs-real 相关系数达 0.87-0.97。

**[Etch Generalizing Body Fitting To Clothed Humans Via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)**

:   提出ETCH框架，通过建模从衣物表面到体表的SE(3)等变紧密度向量(tightness vector)，将穿衣人体的body fitting简化为tightness-aware的稀疏marker拟合任务，在CAPE和4D-Dress数据集上相比SOTA方法（含tightness-agnostic和tightness-aware方法）在宽松衣物上提升16.7%~69.5%的关节误差，形状精度平均提升49.9%。

**[Evagaussians Event Stream Assisted Gaussian Splatting From Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)**

:   提出EvaGaussians框架，利用事件相机的高时间分辨率事件流辅助3D高斯泼溅从运动模糊图像中学习，通过事件辅助初始化、模糊/事件联合重建损失和事件辅助几何正则化，实现高保真新视图合成并保持实时渲染效率。

**[Event-Based Tiny Object Detection A Benchmark Dataset And Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)**

:   提出首个大规模事件相机反无人机小目标检测基准EV-UAV数据集（147序列/230万事件级标注/平均目标仅6.8×5.4像素），并设计EV-SpSegNet——基于稀疏3D点云分割的检测框架，利用小目标在时空事件点云中形成连续曲线的特征，配合时空相关性损失(STC loss)引导网络保留目标事件，在IoU/ACC/检测概率上全面超越13种SOTA方法，推理速度快10-100倍。

**[Event-Boosted Deformable 3D Gaussians For Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)**

:   首次将事件相机与可变形 3D 高斯溅射（3D-GS）结合用于动态场景重建，提出 GS-阈值联合建模策略和动静分解策略，在新构建的事件-4D 基准上实现了 SOTA 的渲染质量和速度（合成数据平均 PSNR 提升 2.73dB，渲染速度达 4D-GS 的 1.71 倍）。

**[Event-Driven Storytelling With Multiple Lifelike Humans In A 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)**

:   提出基于事件驱动的LLM框架，将3D场景中多角色行为规划分解为叙述者逐事件生成和事件解析器的精细空间推理两个模块，首次实现了大规模多房间3D场景中4-5+角色的长时序自然交互运动生成。

**[Excap3D Expressive 3D Scene Understanding Via Object Captioning With Varying Det](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)**

:   提出 ExCap3D，一个在 3D 室内场景中对物体生成多粒度描述的方法，包含物体级和部件级两个描述层次，通过部件→物体的信息共享和语义/文本一致性损失确保描述的准确性与一致性，在新建的 190K 描述数据集上 CIDEr 评分比 SOTA 分别提升 17% 和 124%。

**[Facelift Learning Generalizable Single Image 3D Face Reconstruction From Synthet](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)**

:   提出 FaceLift，一种仅在合成数据上训练但能良好泛化到真实图像的单图360度高质量3D人头重建方法，通过多视图潜扩散模型生成身份一致的多视角图像，再用基于 Transformer 的重建器生成像素对齐的3D高斯表示。

**[Faster And Better 3D Splatting Via Group Training](faster_and_better_3d_splatting_via_group_training.md)**

:   提出 **Group Training** 策略，通过将高斯基元周期性分组为"训练组"和"缓存组"来加速 3DGS 训练，结合**基于透明度的优先采样**（OPS），在4个标准数据集上实现约 **30% 训练加速**的同时**提升渲染质量**和**减少模型体积**，且可即插即用于 3DGS 和 Mip-Splatting 等框架。

**[Fiffdepth Feed-Forward Transformation Of Diffusion-Based Generators For Detailed](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)**

:   提出FiffDepth，将预训练的扩散模型转化为确定性前馈架构进行单目深度估计，通过保持扩散轨迹维持細节生成能力，并引入可学习滤波器蒸馏DINOv2的鲁棒泛化能力到扩散骨干网络，在效率、精度和细节丰富度三方面同时超越现有方法。

**[Find Any Part In 3D](find_any_part_in_3d.md)**

:   提出Find3D，构建了一个由2D基础模型（SAM + Gemini）驱动的自动化3D数据标注引擎，生成210万个部件标注，训练出首个同时具备开放世界、跨类别、部件级和前馈推理能力的3D分割模型，零样本mIoU提升260%，推理速度比现有方法快6-300倍。

**[Fish2Mesh Transformer 3D Human Mesh Recovery From Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)**

:   提出 Fish2Mesh，一种鱼眼感知的 Transformer 模型，通过新颖的自我中心位置编码（EPE）将等距柱状投影的3D球面信息嵌入 Swin Transformer，实现从头戴式鱼眼相机图像精确恢复3D人体网格。

**[Flashdepth Real-Time Streaming Video Depth Estimation At 2K Resolution](flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)**

:   提出FlashDepth，在Depth Anything v2基础上添加Mamba循环模块实现帧间尺度一致性，并设计Small-Large混合架构在2K分辨率下达到24 FPS的实时流式视频深度估计，边界清晰度远超现有方法。

**[Flexgen Flexible Multi-View Generation From Text And Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)**

:   本文提出 FlexGen，一个灵活的多视角图像生成框架，通过 GPT-4V 生成 3D-aware 文本标注并设计自适应双控制模块，支持单图、文本或二者联合控制生成一致的多视角图像，实现未可见区域补全、材质编辑和纹理控制等多种可控能力。

**[Free-Form Motion Control Controlling The 6D Poses Of Camera And Objects In Video](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)**

:   提出 SynFMC 合成数据集（首个包含相机和物体完整 6D 位姿标注的视频数据集）和 FMC 方法，实现了在文本到视频生成中独立或同时控制相机和物体的 6D 位姿，在多种场景下生成高保真视频，且兼容多种个性化 T2I 模型。

**[From Gallery To Wrist Realistic 3D Bracelet Insertion In Videos](from_gallery_to_wrist_realistic_3d_bracelet_insertion_in_videos.md)**

:   提出一种混合管线将 3D 手镯逼真插入视频：利用 3D 高斯泼溅（3DGS）保证时序一致性，用 2D 扩散模型增强光照真实感，并通过光照驱动（Shading-Driven）管线分离 albedo/shading/反射残差分别优化，在用户研究中以 81.7% 的真实感偏好率大幅超越现有方法。

**[From Image To Video An Empirical Study Of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)**

:   系统对比了相同架构(WALT)在图像 vs 视频生成目标下训练的扩散模型在下游视觉理解任务上的表现，发现视频扩散模型在所有任务上一致优于图像对应物，尤其在需要运动和3D空间理解的任务上优势显著（点跟踪+68%、相机位姿+60%）。

**[From One To More Contextual Part Latents For 3D Generation](from_one_to_more_contextual_part_latents_for_3d_generation.md)**

:   提出CoPart框架，通过上下文部件潜码表示3D物体并利用互引导策略微调预训练扩散模型，实现高质量的部件级3D生成，同时支持部件编辑、铰接体生成和小场景生成。

**[Fross Faster-Than-Real-Time Online 3D Semantic Scene Graph Generation From Rgb-D](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)**

:   提出FROSS方法，通过将2D场景图直接提升到3D空间并用高斯分布表示物体，实现了超实时（144 FPS）的在线3D语义场景图生成，无需精确点云重建。

**[G2Sf Geometry-Guided Score Fusion For Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)**

:   提出 G2SF 框架，将基于 memory bank 的异常分数重新解释为局部特征空间中的各向同性欧氏距离，进而通过 Local Scale Prediction Network (LSPN) 学习方向感知的缩放因子，将其渐进演化为各向异性的统一融合度量，实现多模态工业异常检测 SOTA。

**[Gap Gaussianize Any Point Clouds With Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)**

:   提出GAP框架,利用深度感知图像扩散模型将无色点云转化为高保真3D Gaussian表示,通过表面锚定机制确保几何精度,并设计基于扩散的inpainting策略补全难以观测区域。

**[Gas Generative Avatar Synthesis From A Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)**

:   提出GAS框架，通过将泛化NeRF重建的密集外观线索与视频扩散模型结合，统一新视角和新姿态合成为视频生成任务，配合模态切换器解耦两种任务，实现从单张图像生成视角一致和时序连贯的人体Avatar。

**[Gaussian Splatting With Discretized Sdf For Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)**

:   本文提出将连续SDF离散化为高斯基元的额外属性，通过SDF-to-opacity变换统一高斯和SDF表示，配合投影一致性损失和球面初始化，在仅用4G显存的前提下实现了超越现有高斯逆渲染方法的重光照质量。

**[Gaussian Variation Field Diffusion For High-Fidelity Video-To-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)**

:   提出一种视频到4D生成框架，通过Direct 4DMesh-to-GS Variation Field VAE将动画数据直接编码为紧凑的高斯变化场潜在空间，再训练时序感知的扩散模型生成动态3D内容，在4.5秒内实现高保真4D合成，并展示了对真实视频输入的优越泛化能力。

**[Gaussianproperty Integrating Physical Properties To 3D Gaussians With Lmms](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)**

:   GaussianProperty 提出了一个免训练框架，利用 SAM 分割和 GPT-4V 识别能力，通过全局-局部推理模块和多视角投票策略，将物理属性（密度、弹性模量、摩擦系数等）赋予 3D Gaussians，支持物理仿真和机器人抓取两大下游任务。

**[Gaussianupdate Continual 3D Gaussian Splatting Update For Changing Environments](gaussianupdate_continual_3d_gaussian_splatting_update_for_changing_environments.md)**

:   提出GaussianUpdate，首次将3D高斯表示与持续学习结合，通过三阶段更新策略（外观更新→几何布局更新→联合精炼）和可见性感知生成式回放，实现时变场景的实时渲染和变化可视化。

**[Gazegaussian High-Fidelity Gaze Redirection With 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)**

:   提出GazeGaussian，首个基于3D高斯溅射（3DGS）的高保真视线重定向方法，通过双流3DGS模型分别建模面部和眼部区域，设计显式的高斯眼球旋转表示和表情引导神经渲染器，在视线精度、合成质量和渲染速度上全面超越现有方法。

**[Generating Physically Stable And Buildable Brick Structures From Text](generating_physically_stable_and_buildable_brick_structures_from_text.md)**

:   BrickGPT 首次实现从文本提示生成物理稳定且可组装的互锁砖块结构，核心思想是将积木组装问题建模为自回归文本生成任务，并在推理时集成物理感知的有效性检查和回滚机制，确保生成结构的稳定性和可构建性。

**[Geo4D Leveraging Video Generators For Geometric 4D Scene Reconstruction](geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)**

:   将预训练视频扩散模型(DynamiCrafter)改造为单目4D动态场景重建器——同时预测点云图、视差图和射线图三种互补几何模态，通过多模态对齐融合算法和滑动窗口推理，仅用合成数据训练即可零样本泛化至真实视频，大幅超越当前视频深度估计SOTA。

**[Geometry Distributions](geometry_distributions.md)**

:   提出Geometry Distributions (GeomDist)，将3D几何建模为表面点的概率分布并用扩散模型学习，无需假设亏格、连通性或边界条件，可从高斯噪声采样无限多表面点来表示任意拓扑的几何。

**[Geoprog3D Compositional Visual Reasoning For City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)**

:   提出 GeoProg3D，首个支持城市级高保真3D场景自然语言交互的视觉编程框架，通过地理感知的城市级3D语言场（GCLF）和地理视觉API（GV-APIs），结合LLM推理引擎实现组合式地理空间推理，在新提出的952条查询的GeoEval3D基准上全面超越现有3D语言场和VLM方法。

**[Geosplatting Towards Geometry Guided Gaussian Splatting For Physically-Based Inv](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)**

:   提出 GeoSplatting，通过从可优化的显式网格可微分地生成表面对齐的高斯点，为3DGS提供精确的几何引导，实现SOTA逆渲染性能（材质-光照解耦），同时训练仅需10-15分钟。

**[Global-Aware Monocular Semantic Scene Completion With State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)**

:   提出GA-MonoSSC，一种结合Transformer（2D全局上下文）和Mamba（3D长程依赖）的混合架构用于室内单目语义场景补全，创新引入Frustum Mamba Layer解决体素序列化中的特征不连续性问题，在Occ-ScanNet和NYUv2上达到SOTA。

**[Global Motion Corresponder For 3D Point-Based Scene Interpolation Under Large Mo](global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)**

:   提出Global Motion Corresponder (GMC),通过学习将两个时刻的3D Gaussian映射到共享规范空间的一元势场,实现大运动条件下的鲁棒场景插值和外推。

**[Gsot3D Towards Generic 3D Single Object Tracking In The Wild](gsot3d_towards_generic_3d_single_object_tracking_in_the_wild.md)**

:   提出 GSOT3D，目前最大的通用3D单目标跟踪基准，包含620个多模态序列（点云+RGB+深度）覆盖54类物体，支持PC/RGB-PC/RGB-D三种3D跟踪任务，并提出渐进式时空跟踪器PROT3D以9DoF包围盒实现最优性能。

**[Guava Generalizable Upper Body 3D Gaussian Avatar](guava_generalizable_upper_body_3d_gaussian_avatar.md)**

:   提出 GUAVA，首个从单张图像通过前馈推理快速重建可动画上半身3D高斯虚拟人的框架，结合模板高斯和 UV 高斯表示，支持丰富面部表情和手势驱动，约0.1s完成重建并实时渲染。

**[Guiding Diffusion-Based Articulated Object Generation By Partial Point Cloud Ali](guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)**

:   提出 PhysNAP，通过点云对齐损失和基于SDF的物理合理性约束（部件穿透+关节移动）引导预训练扩散模型 NAP 的逆扩散过程，实现类别感知的铰接物体生成，在对齐精度和物理合理性上显著优于无引导基线。

**[Hierarchical Material Recognition From Local Appearance](hierarchical_material_recognition_from_local_appearance.md)**

:   提出面向视觉应用的层级式材质分类学体系(taxonomy)与野外数据集 Matador（含深度图的 ~7200 张材质图像，57类），并基于图注意力网络(GAT)利用分类学的层级亲缘关系进行材质识别，在多个基准数据集上达到 SOTA，同时支持新材质的小样本学习和场景中任意点的材质探测。

**[His-Gpt Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)**

:   提出 HIS-GPT，首个面向3D人-场景联合理解的多模态大语言模型，通过辅助交互模块(AInt)和布局-轨迹位置编码(LTP)捕获人场交互线索，并构建首个系统性基准 HIS-Bench，在HIS-QA任务上大幅超越GPT-4o等基线。

**[Hort Monocular Hand-Held Objects Reconstruction With Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)**

:   提出 HORT，基于 Transformer 的粗到细框架，从单目图像高效重建手持物体的稠密3D点云，通过整合图像特征和3D手部几何信息联合预测物体点云及其相对手部的位姿，在准确率和推理速度上均达到 SOTA。

**[Housetour A Virtual Real Estate Aigent](housetour_a_virtual_real_estate_aigent.md)**

:   提出 HouseTour，给定一组已知位姿的室内图像，联合生成类人的3D相机轨迹和房地产文字描述，通过 Residual Diffuser 进行基于扩散的轨迹规划并将空间特征集成到 Qwen2-VL-3D 中生成3D-grounded文本摘要。

**[How Far Are Ai-Generated Videos From Simulating The 3D Visual World A Learned 3D](how_far_are_ai-generated_videos_from_simulating_the_3d_visual_world_a_learned_3d.md)**

:   提出 Learned 3D Evaluation (L3DE)，一种基于单目3D线索（运动、深度、外观）和对比学习的客观可量化评估方法，用于衡量AI生成视频在3D视觉一致性方面与真实视频的差距，无需人工标注缺陷或质量标签。

**[Humanolat A Large-Scale Dataset For Full-Body Human Relighting And Novel-View Sy](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)**

:   提出HumanOLAT——首个公开可用的大规模全身人体多视角OLAT(One-Light-at-a-Time)数据集，包含21个被试×3个姿态×40视角×344种光照≈850K帧，为人体重打光和新视角合成提供了高质量基准。

**[Identity Preserving 3D Head Stylization With Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)**

:   提出基于负对数似然蒸馏(LD)的3D头部风格化框架，通过多视角网格评分、镜像梯度和秩加权评分张量，实现在360度一致渲染下的高质量风格化与身份保持。

**[Im360 Large-Scale Indoor Mapping With 360 Cameras](im360_large-scale_indoor_mapping_with_360_cameras.md)**

:   本文提出 IM360，一个面向稀疏扫描大规模室内环境的三维建图流水线，通过将球面相机模型深度集成到 SfM 各环节、结合稠密特征匹配和可微渲染纹理优化，在 Matterport3D 和 Stanford2D3D 上实现了远超现有方法的相机定位准确率和渲染质量（PSNR 提升 3.5）。

**[Image-Guided Shape-From-Template Using Mesh Inextensibility Constraints](image-guided_shape-from-template_using_mesh_inextensibility_constraints.md)**

:   提出一种纯图像引导的无监督 Shape-from-Template (SfT) 方法，仅利用颜色、梯度和轮廓等视觉线索配合网格不可伸展性约束来重建变形物体 3D 形状，比最优无监督方法快 400 倍且精度大幅领先。

**[Image As An Imu Estimating Camera Motion From A Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)**

:   本文将运动模糊从"不需要的伪影"转变为"有价值的运动线索"，通过从单张模糊图像预测稠密光流场和单目深度图，再用可微分最小二乘求解器恢复相机6DoF瞬时速度，实现媲美甚至超越IMU的运动估计精度和30FPS实时性能。

**[Instascene Towards Complete 3D Instance Decomposition And Reconstruction From Cl](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)**

:   InstaScene 提出统一的杂乱场景实例分解与完整重建框架，通过追踪高斯光栅化构建空间对比学习实现精准实例分割，并设计 in-situ 生成管线利用已知观测和几何线索引导 3D 生成模型重建完整物体。

**[Jointdit Enhancing Rgb-Depth Joint Modeling With Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)**

:   JointDiT 基于 Flux 扩散 Transformer 构建 RGB-Depth 联合分布模型，通过自适应调度权重和非平衡时间步采样策略，使单一模型通过控制各模态的时间步即可灵活执行联合生成、深度估计和深度条件图像生成三种任务。

**[Kh Symmetry Understanding Of 3D Shapes Via Chirality Disentanglement](kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)**

:   提出无监督手性特征提取管线,从2D基础模型特征中蒸馏左右手性信息用于装饰3D形状顶点描述子,有效解决形状分析中的左右歧义问题。

**[Laconic A 3D Layout Adapter For Controllable Image Creation](laconic_a_3d_layout_adapter_for_controllable_image_creation.md)**

:   提出 LACONIC，一种基于参数化 3D 语义包围盒的轻量级适配器，通过解耦交叉注意力机制将显式 3D 几何信息注入预训练 text-to-image 扩散模型，首次实现了相机控制、3D 物体级语义引导以及对屏幕外物体的全面场景上下文建模，在 FID 上比 SceneCraft 降低 75.8%。

**[Layerlock Non-Collapsing Representation Learning With Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)**

:   提出 LayerLock，一种通过渐进式冻结网络层并动态切换预测目标（从像素到越来越深的中间层特征）的自监督视频表征学习方法，兼具像素预测的稳定性和潜变量预测的高效语义捕获能力，用于训练高达 4B 参数的视频模型。

**[Learning 3D Object Spatial Relationships From Pre-Trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)**

:   提出从预训练 2D 扩散模型合成图像中学习物体间 3D 空间关系（OOR），通过 3D 提升管线构建配对数据集，训练文本条件化的 score-based 扩散模型对物体对的相对位姿和尺度分布建模，并扩展至多物体场景布局和场景编辑。

**[Learning 3D Scene Analogies With Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)**

:   提出3D场景类比任务，通过神经上下文场景映射（neural contextual scene maps）在共享相似语义上下文的场景区域间建立稠密三维映射，支持轨迹迁移与物体放置迁移等下游应用。

**[Learning 4D Embodied World Models](learning_4d_embodied_world_models.md)**

:   提出 TesserAct——一种 4D 具身世界模型，通过训练视频生成模型联合预测 RGB、深度和法线视频，再转换为高质量 4D 场景，实现空间-时间一致的 3D 世界动态模拟和机器人动作规划。

**[Learning Robust Stereo Matching In The Wild With Selective Mixture-Of-Experts](learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)**

:   提出 SMoEStereo，通过在冻结的视觉基础模型(VFM)中集成变秩MoE-LoRA和变核MoE-Adapter，结合轻量决策网络选择性激活MoE模块，实现了场景自适应的鲁棒立体匹配，在跨域和联合泛化上达到SOTA。

**[Lightweight Gradient-Aware Upscaling Of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)**

:   提出专门为3DGS设计的轻量图像上采样技术，利用高斯原语的解析图像梯度进行梯度感知双三次样条插值，无需深度学习推理即可实现3-4倍渲染加速，且重建质量优于标准双三次插值和DL-based上采样。

**[Linr-Pcgc Lossless Implicit Neural Representations For Point Cloud Geometry Comp](linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)**

:   LINR-PCGC 提出了首个基于隐式神经表征（INR）的点云几何无损压缩方法，通过设计轻量级多尺度 SparseConv 网络（含尺度上下文提取 SCE 和子节点预测 CNP 模块），结合 GoP 级帧共享解码器和初始化策略，在不依赖特定训练数据分布的前提下，在 MVUB 数据集上比 G-PCC TMC13v23 降低 21.21% 码率，比 SparsePCGC 降低 21.95%。

**[Llava-3D A Simple Yet Effective Pathway To Empowering Lmms With 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)**

:   本文提出 LLaVA-3D，通过将 3D 位置嵌入注入 2D CLIP patch 特征构建"3D Patch"，以最小改动将 2D LMM（LLaVA-Video）扩展为统一的 2D/3D 理解模型，训练收敛速度比现有 3D LMM 快 3.5 倍，在多个 3D 基准上达到 SOTA 且保持 2D 能力不下降。

**[Localdygs Multi-View Global Dynamic Scene Modeling Via Adaptive Local Implicit F](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)**

:   提出 LocalDyGS——将全局复杂动态场景分解为种子点定义的局部空间、并通过静态-动态特征解耦生成时序高斯来建模各局部运动的框架，首次实现了大尺度复杂动态场景的高质量重建。

**[Long3R Long Sequence Streaming 3D Reconstruction](long3r_long_sequence_streaming_3d_reconstruction.md)**

:   提出 LONG3R，一种基于循环记忆机制的流式多视图3D重建模型，通过记忆门控、双源精炼解码器和3D时空记忆三大创新，在保持实时推理速度的同时显著提升长序列重建质量。

**[Longsplat Robust Unposed 3D Gaussian Splatting For Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)**

:   LongSplat 针对无相机位姿的随拍长视频场景，提出增量联合优化框架同时优化相机位姿和 3DGS，设计基于 MASt3R 先验的鲁棒位姿估计模块和自适应八叉树锚点形成机制，解决位姿漂移、几何初始化不准和内存限制问题。

**[Maskhand Generative Masked Modeling For Robust Hand Mesh Reconstruction In The W](maskhand_generative_masked_modeling_for_robust_hand_mesh_reconstruction_in_the_w.md)**

:   提出 MaskHand，首个将生成式掩码建模引入 3D 手部网格重建的方法，通过 VQ-MANO 将连续手部姿态离散化为 token，再利用上下文引导的掩码 Transformer 学习 2D-to-3D 映射的概率分布，在推理时通过置信度引导的迭代采样生成高精度手部网格，在 HO3Dv3 零样本评估中 PA-MPJPE 降低 19.5%。

**[Materialmvp Illumination-Invariant Material Generation Via Multi-View Pbr Diffus](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)**

:   MaterialMVP 是一个端到端的多视图 PBR 纹理生成模型，通过参考注意力、一致性正则化训练和双通道材质生成框架，从3D网格和图像提示生成光照不变且多视图一致的高质量 PBR 材质。

**[Mega Memory-Efficient 4D Gaussian Splatting For Dynamic Scenes](mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)**

:   提出 MEGA，一个面向4D Gaussian Splatting的内存高效框架，通过DC-AC颜色分解消除冗余球谐系数（8×压缩），结合熵约束Gaussian形变技术扩大每个Gaussian的作用范围并减少数量，最终在Technicolor和Neural 3D Video数据集上分别实现约190×和125×存储压缩，同时保持可比的渲染质量和实时速度。

**[Meshanything V2 Artist-Created Mesh Generation With Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)**

:   MeshAnything V2 提出 Adjacent Mesh Tokenization (AMT)，通过用单个顶点（而非传统三个顶点）表示相邻面，将网格的 token 序列长度平均缩短一半，从而在不增加计算成本的前提下将最大生成面数从 800 提升到 1600，显著提高了自回归网格生成的效率和质量。

**[Meshmamba State Space Models For Articulated 3D Mesh Generation And Reconstructi](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)**

:   MeshMamba 提出基于 Mamba 状态空间模型的 3D 关节体网格生成与重建方法，通过设计基于身体部位 UV 图和模板网格坐标的顶点序列化技术，实现了万级顶点网格的高效生成和重建，速度比 Transformer 快 6-9 倍。

**[Meshpad Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation And Ed](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)**

:   MeshPad 将草图驱动的 3D 网格创建与编辑解耦为"添加"和"删除"两个子任务，基于三角序列表示和 Transformer 自回归生成，并提出顶点对齐推测解码器实现 2.2× 加速，让交互式网格编辑在几秒内完成。

**[Mincd-Pnp Learning 2D-3D Correspondences With Approximate Blind Pnp](mincd-pnp_learning_2d-3d_correspondences_with_approximate_blind_pnp.md)**

:   本文提出 MinCD-PnP，通过三重近似将计算昂贵的 Blind PnP 简化为最小化 2D-3D 关键点间 Chamfer 距离的问题，设计轻量级多任务学习模块 MinCD-Net 集成到现有 I2P 配准框架中，在跨场景和跨数据集设置下显著提升内点率和配准召回率。

**[Moga 3D Generative Avatar Prior For Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)**

:   提出MoGA，通过学习生成式3D头像先验并将其作为初始化、正则化和姿态优化的强约束，从单张图像重建高保真3D高斯头像，显著超越现有方法。

**[Momentum-Gs Momentum Gaussian Self-Distillation For High-Quality Large Scene Rec](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)**

:   Momentum-GS 提出基于动量的自蒸馏机制来解决大规模场景3D高斯溅射中分块并行训练的一致性问题，通过动量教师高斯解码器提供全局引导并解耦分块数量与GPU数量的限制，在多个大规模场景数据集上取得SOTA，LPIPS较CityGaussian提升18.7%。

**[Monocular Semantic Scene Completion Via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)**

:   提出 MonoMRN，一个两阶段单目语义场景补全框架：先做粗粒度预测，再用 Masked Sparse GRU（MS-GRU）循环精炼被遮挡区域，并引入距离注意力投影减少深度投影误差，在 NYUv2 和 SemanticKITTI 上均达到 SOTA。

**[Monomobility Zero-Shot 3D Mobility Analysis From Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)**

:   MonoMobility 提出首个从单目视频零样本分析关节物体运动部件及运动属性（运动轴、运动类型）的框架，通过2D高斯泼溅场景表示和端到端动态场景优化算法，无需标注数据即可处理旋转、平移及复合运动。

**[Mugs Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)**

:   MuGS 是首个支持多基线设置（小基线到大基线）的泛化3D高斯泼溅方法，通过融合 MVS 和 MDE 特征、投影-采样深度一致性网络和参考视图损失，在不同基线数据集上均达到 SOTA。

**[Multi-Identity Human Image Animation With Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)**

:   提出Structural Video Diffusion,通过身份特定嵌入和RGB-深度-法线联合学习,首次实现多身份人体视频生成中的外观一致性保持和3D感知的人-物交互建模。

**[Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)**

:   提出 MVTracker——首个数据驱动的多视角3D点跟踪器，通过将多视图深度图反投影为统一的3D特征点云，利用 kNN 关联和 Transformer 迭代优化，在仅需4个相机的实用配置下实现鲁棒的长程3D点轨迹估计，在 Panoptic Studio 和 DexYCB 上分别达到 3.1 cm 和 2.0 cm 的中位轨迹误差。

**[Mv-Adapter Multi-View Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)**

:   提出首个基于Adapter的多视角图像生成方案MV-Adapter，通过复制self-attention层+并行注意力架构实现即插即用的多视角生成，在SDXL上达到768分辨率，兼容各种T2I衍生模型。

**[Mvgbench A Comprehensive Benchmark For Multi-View Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)**

:   提出 MVGBench——多视图生成模型的综合评估框架，创新性地引入基于 3DGS 自一致性的 3D 一致性指标（无需 3D GT），系统评估了 12 个 SOTA 方法在最佳性能、泛化和鲁棒性三方面的表现，并基于分析提出的最佳实践构建了新方法 ViFiGen。

**[Nautilus Locality-Aware Autoencoder For Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)**

:   Nautilus 提出一种局部性感知的自编码器进行可扩展的 artist-like 网格生成，通过 Nautilus 式壳结构网格分词算法将序列长度压缩到 1/4，并结合双流点云条件器提高局部结构保真度，首次实现最多 5000 面的高质量网格直接生成。

**[Neural Compression For 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)**

:   提出NeCGS,首个能将包含数千个多样3D网格模型的几何集合压缩高达900倍的神经压缩范式,通过TSDF-Def隐式表示和量化感知自解码器实现高精度保持。

**[Neuraleaf Neural Parametric Leaf Models With Shape And Deformation Disentangleme](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)**

:   NeuraLeaf 将叶片的 3D 几何解耦为 2D 基础形状和 3D 变形两个潜在空间，利用大量 2D 叶片图像数据集学习形状空间，提出无骨架蒙皮模型处理叶片的高度柔性变形，并构建了首个专注叶片变形建模的 3D 数据集 DeformLeaf。

**[No Pose At All Self-Supervised Pose-Free 3D Gaussian Splatting From Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)**

:   提出SPFSplat,首个在训练和推理时均不需要真值位姿的自监督3DGS框架,通过共享ViT骨干同时预测Gaussian基元和相机位姿,在极端视角变化下超越需要位姿的SOTA方法。

**[Noise2Score3D Tweedies Approach For Unsupervised Point Cloud Denoising](noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)**

:   提出Noise2Score3D,基于Tweedie公式的全无监督点云去噪框架,从噪声数据直接学习得分函数,实现单步去噪;引入点云全变分度量估计未知噪声参数。

**[Not All Frame Features Are Equal Video-To-4D Generation Via Decoupling Dynamic-S](not_all_frame_features_are_equal_video-to-4d_generation_via_decoupling_dynamic-s.md)**

:   DS4D 首次提出在video-to-4D生成中沿时间轴和空间轴解耦动静态特征，通过动静态特征解耦模块（DSFD）获取动态表征，并通过时空相似性融合模块（TSSF）跨视角自适应聚合动态信息，在Consistent4D和Objaverse数据集上达到SOTA。

**[Occlugaussian Occlusion-Aware Gaussian Splatting For Large Scene Reconstruction ](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)**

:   提出遮挡感知的场景划分策略和基于区域的渲染技术,通过相机共可见性图聚类实现与场景布局对齐的分区,显著提升大场景3DGS重建质量和渲染速度。

**[One Look Is Enough Seamless Patchwise Refinement For Zero-Shot Monocular Depth E](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)**

:   提出 PRO（Patch Refine Once），通过分组块一致性训练（GPCT）和无偏遮罩（BFM）策略，在高分辨率图像上实现无缝的逐块深度精炼，仅需每块单次精炼即可消除边界伪影，推理速度比 PatchRefiner 快12倍。

**[Online Language Splatting](online_language_splatting.md)**

:   首个在 3DGS-SLAM 系统中实现**在线、近实时、开放词汇**语言建图的框架，通过高分辨率 CLIP 嵌入、两阶段在线自编码器压缩和颜色-语言解耦优化三项创新，在精度超越离线 SOTA 的同时实现 40×–200× 的效率提升。

**[Open-Vocabulary Octree-Graph For 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)**

:   提出 Octree-Graph，一种将自适应八叉树与图结构结合的新颖场景表示，通过时序分组式段合并(CGSM)和实例特征聚合(IFA)获取准确的语义对象，实现高效的开放词汇3D场景理解。

**[Outdoor Monocular Slam With Global Scale-Consistent 3D Gaussian Pointmaps](outdoor_monocular_slam_with_global_scale-consistent_3d_gaussian_pointmaps.md)**

:   提出 S3PO-GS，通过将 3DGS 渲染的 pointmap 作为锚点建立尺度自一致的跟踪模块，结合基于 patch 的 pointmap 动态建图机制，在 RGB-only 室外场景中实现了无累积尺度漂移的高精度定位与高保真新视角合成。

**[Panst3R Multi-View Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)**

:   基于MUSt3R构建PanSt3R，在**单次前向传播**中同时完成3D重建和多视角全景分割，无需相机参数、无需测试时优化，比现有方法快数个量级。

**[Pcr-Gs Colmap-Free 3D Gaussian Splatting Via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)**

:   提出 PCR-GS，通过 DINO 特征重投影正则化和基于小波变换的频率正则化对相机位姿进行协同约束，在无需 COLMAP 先验的条件下实现了复杂相机轨迹场景的高质量 3D-GS 重建与位姿估计。

**[Perspose 3D Human Pose Estimation With Perspective Encoding And Perspective Rota](perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)**

:   提出PersPose框架，通过透视编码(PE)将裁剪后相机内参编码为2D映射、透视旋转(PR)将人体居中以消除透视畸变，解决了现有方法忽略FOV信息导致深度估计不准确的问题。

**[Phd Personalized 3D Human Body Fitting With Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)**

:   提出个性化3D人体姿态估计范式PHD——先通过SHAPify校准用户体型，再用体型条件化的点扩散模型PointDiT作为3D先验，结合Point Distillation Sampling损失迭代优化姿态，在绝对姿态精度上达到EMDB数据集SOTA。

**[Physsplat Efficient Physics Simulation For 3D Scenes Via Mllm-Guided Gaussian Sp](physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)**

:   提出PhysSplat，首次利用多模态大语言模型(MLLM)零样本估计3D场景中物体的物理属性，结合物理-几何自适应采样策略在单GPU上2分钟内实现逼真的物理仿真。

**[Placeit3D Language-Guided Object Placement In Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)**

:   提出语言引导的真实3D场景中物体放置任务（PlaceIt3D），包含基准测试、大规模数据集和基于3D LLM的基线方法PlaceWizard，实现对场景、物体和语言指令的联合推理。

**[Polaranything Diffusion-Based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)**

:   提出 PolarAnything，首个基于单张 RGB 图像生成偏振图像的扩散模型框架，通过对编码后的 AoLP 和 DoLP 进行去噪扩散，实现了物理准确且逼真的偏振属性合成，无需 3D 资产或偏振相机。

**[Proactive Scene Decomposition And Reconstruction](proactive_scene_decomposition_and_reconstruction.md)**

:   提出基于主动人-物交互的在线场景分解与重建任务,通过观察自我中心视角下的交互行为来定义分解粒度,实现渐进式对象解耦和高质量全局重建。

**[Pseudomaptrainer Learning Online Mapping Without Hd Maps](pseudomaptrainer_learning_online_mapping_without_hd_maps.md)**

:   提出 PseudoMapTrainer，首次实现**完全不依赖 GT HD Map** 训练在线建图模型：利用 2D Gaussian Splatting（RoGS）从多视角相机图像重建道路表面并结合预训练语义分割（Mask2Former）生成矢量化伪标签，同时设计 mask-aware 匹配算法与损失函数处理部分遮挡的伪标签，支持单次行程和多次行程（众包数据）两种模式。

**[Rayletdf Raylet Distance Fields For Generalizable 3D Surface Reconstruction From](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)**

:   提出 RayletDF，一种基于"raylet"（光线片段）距离场的泛化3D表面重建方法，通过raylet特征提取器、距离场预测器和多raylet混合器三个模块，从点云或3D高斯直接预测表面点，在未见数据集上实现单次前向传播的高精度跨数据集泛化。

**[Rayzer A Self-Supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)**

:   提出 RayZer，一个无需任何3D监督（无相机位姿/无场景几何标注）的自监督多视角3D视觉模型，通过将图像解耦为相机参数和场景表示实现3D感知自编码，在新视角合成任务上达到甚至超越依赖位姿标注的"oracle"方法。

**[Realcam-I2V Real-World Image-To-Video Generation With Interactive Complex Camera](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)**

:   提出 RealCam-I2V，通过集成单目度量深度估计构建3D场景实现度量尺度对齐训练，并提供交互式3D场景轨迹绘制界面和场景约束噪声整形机制，解决了现有轨迹引导I2V方法的尺度不一致和真实世界可用性问题。

**[Reggs Unposed Sparse Views Gaussian Splatting With 3Dgs Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)**

:   提出 RegGS 框架，通过基于最优传输 MW2 距离的可微 3DGS 配准模块，将前馈网络生成的局部3D高斯增量式地对齐到全局一致的3D表示中，实现无位姿稀疏视角的高质量3D重建。

**[Relative Illumination Fields Learning Medium And Light Independent Underwater Sc](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)**

:   提出相对光照场（Relative Illumination Fields），通过在相机局部坐标系中用MLP建模非均匀光照分布，结合体积介质表示，实现对水下场景的干净重建——去除光源和介质的影响。

**[Reparo Compositional 3D Assets Generation With Differentiable 3D Layout Alignmen](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)**

:   提出REPARO，通过先分别重建单个物体3D网格再利用基于最优传输的可微渲染进行布局对齐，实现从单张图像生成多物体组合式3D资产。

**[Reposed Efficient Relative Pose Estimation With Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)**

:   本文提出了一组高效的相对位姿最小求解器，将单目深度估计（MDE）的尺度和仿射参数与相对位姿联合估计，在标定/共焦距/不同焦距三种配置下均超越SOTA深度感知求解器，并通过大规模实验回答了"MDE深度是否有助于相对位姿估计"这一核心问题。

**[Representing 3D Shapes With 64 Latent Vectors For 3D Diffusion Models](representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)**

:   提出COD-VAE，通过两阶段自编码器方案（渐进式编码器 + Triplane解码器 + 不确定性引导Token剪枝），将3D形状编码为仅64个1D潜在向量，在保持重建质量的同时实现16×压缩比和20.8×生成加速。

**[Repurposing 2D Diffusion Models With Gaussian Atlas For 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)**

:   提出 Gaussian Atlas 表示法，将无序3D高斯通过最优传输映射到球面再展平为规整2D网格，从而直接微调预训练2D Latent Diffusion模型实现高质量文本到3D生成。

**[Resgs Residual Densification Of 3D Gaussian For Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)**

:   提出残差分裂（residual split）操作替代3D-GS中split/clone的二元选择机制，配合图像金字塔渐进监督和可变梯度阈值选择策略，自适应地同时解决过重建和欠重建问题，在减少高斯数量的同时实现SOTA渲染质量。

**[Revisiting Point Cloud Completion Are We Ready For The Real-World](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)**

:   通过代数拓扑和持久同调（PH）工具揭示现有合成点云数据集缺乏真实世界中丰富的拓扑特征，贡献了首个真实世界工业点云补全数据集RealPC（~40,000对、21类），并提出BOSHNet通过采样代理同调骨架作为拓扑先验，在真实世界点云补全上取得显著改进。

**[Ri3D Few-Shot Gaussian Splatting With Repair And Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)**

:   提出 RI3D，将稀疏视图合成分解为"修复可见区域"和"补全缺失区域"两个子任务，引入两个个性化扩散模型（repair + inpainting）配合两阶段优化策略，在极端稀疏输入下实现高质量 3DGS 重建。

**[Robopearls Editable Video Simulation For Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)**

:   提出 RoboPearls，基于 3D 高斯溅射（3DGS）构建的可编辑视频仿真框架，从演示视频中构建照片级真实感仿真环境，通过增量语义蒸馏（ISD）和3D正则化NNFM损失支持丰富的场景编辑操作，并利用 LLM 智能体自动化仿真生成流程，形成以 VLM 闭环驱动的机器人学习增强系统。

**[Robotron-Mani All-In-One Multimodal Large Model For Robotic Manipulation](robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)**

:   提出多模态机器人操作模型 RoboTron-Mani 和综合数据集 RoboData，通过相机参数与占用监督增强3D感知、Modality-Isolation-Mask 实现灵活多模态融合，首次作为通才策略在多个数据集上同时超越专家模型。

**[Robust And Efficient 3D Gaussian Splatting For Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)**

:   提出一套面向城市级场景的高效鲁棒3DGS重建框架——通过可见性分区策略、可控LOD生成、细粒度外观变换模块及多种正则化技术，实现了在外观差异大、含瞬态物体的城市数据上高质量重建与实时渲染。

**[Robustereo Robust Zero-Shot Stereo Matching Under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)**

:   提出 RobuSTereo 框架，通过基于扩散模型的立体数据生成管线和结合去噪 ViT 与 VGG19 的鲁棒特征编码器，大幅提升立体匹配模型在雨、雾、雪等恶劣天气下的零样本泛化能力。

**[Robustsplat Decoupling Densification And Dynamics For Transient-Free 3Dgs](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)**

:   本文发现 3DGS 的高斯致密化过程是导致瞬态物体伪影的关键因素，提出延迟高斯生长策略和尺度级联掩码自举方法来解耦致密化与动态区域建模，在多个基准数据集上实现了最优的无瞬态新视角合成效果。

**[Roco-Sim Enhancing Roadside Collaborative Perception Through Foreground Simulati](roco-sim_enhancing_roadside_collaborative_perception_through_foreground_simulati.md)**

:   RoCo-Sim 是首个面向路侧协同感知的仿真框架，通过相机外参优化、多视图遮挡感知采样、DepthSAM 深度渲染和可扩展后处理工具包，从单张图像生成多样且多视图一致的路侧仿真数据，将路侧3D检测性能提升 83%+。

**[Ross3D Reconstructive Visual Instruction Tuning With 3D-Awareness](ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)**

:   Ross3D 提出将3D感知的视觉重建预训练任务（跨视图重建 + 全局BEV重建）注入2D大型多模态模型的训练流程中，在不修改输入表示的前提下通过输出级监督信号显著提升3D场景理解能力，在SQA3D、ScanQA、Scan2Cap、ScanRefer、Multi3DRefer五个基准上均达到SOTA。

**[S3E Self-Supervised State Estimation For Radar-Inertial System](s3e_self-supervised_state_estimation_for_radar-inertial_system.md)**

:   提出S3E，首次实现从雷达信号频谱和惯性数据的互补自监督状态估计，通过基于旋转的跨融合技术增强有限角分辨率下的空间结构信息。

**[S3R-Gs Streamlining The Pipeline For Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)**

:   S3R-GS 通过识别传统街景重建管线中的三大计算冗余（不必要的局部-全局坐标变换、过多的3D-2D投影、低效的远距离内容渲染），提出实例特定投影、时序可见性过滤和自适应LOD策略，将重建时间降至竞争方法的20%-50%，同时保持SOTA渲染质量。

**[Sas Segment Any 3D Scene With Integrated 2D Priors](sas_segment_any_3d_scene_with_integrated_2d_priors.md)**

:   提出 SAS 框架，首次整合多个 2D 开放词汇模型的互补能力来学习更好的 3D 表示：通过 Model Alignment via Text 对齐不同模型的特征空间，通过 Annotation-Free Model Capability Construction 利用扩散模型合成图像来量化各模型识别不同类别的能力，以此指导多模型特征融合和 3D 蒸馏，在 ScanNet v2/Matterport3D/nuScenes 上大幅超越前作。

**[Sat2City 3D City Generation From A Single Satellite Image With Cascaded Latent D](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)**

:   提出 Sat2City，首个从单张卫星图像同时生成城市级几何和外观的3D生成框架，通过将稀疏体素与级联潜扩散模型结合，引入 Re-Hash 多尺度特征网格和逆采样策略，在自建3D城市数据集上实现了优于现有方法的高保真生成。

**[Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)**

:   提出场景坐标回归(SCR)的概率化训练框架，引入手工设计的深度分布先验和基于3D点云扩散模型的学习先验，在多视角约束不足时显著改善场景重建质量、相机位姿估计和下游任务表现。

**[Scenemi Motion In-Betweening For Modeling Human-Scene Interaction](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)**

:   首次正式研究场景感知运动插值（scene-aware motion in-betweening）问题，提出 SceneMI 框架，通过双层场景描述符（全局体素 + 局部 BPS）全面编码场景上下文，利用扩散模型的去噪特性处理含噪关键帧，在 TRUMANS 上碰撞帧率降低 56.9%，在真实世界 GIMO 上脚部滑动减少 37.5%、抖动减少 56.5%。

**[Seeing And Seeing Through The Glass Real And Synthetic Data For Multi-Layer Dept](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)**

:   提出多层深度估计(multi-layer depth estimation)新任务，构建了包含1500张真实图像的LayeredDepth基准和程序化合成数据生成器，揭示了现有深度估计方法在透明物体上的严重不足。

**[Segmentdreamer Towards High-Fidelity Text-To-3D Synthesis With Segmented Consist](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)**

:   本文提出SegmentDreamer，通过分段一致性轨迹蒸馏（SCTD）重新表述SDS损失，解决了现有一致性蒸馏（CD）方法中自一致性和交叉一致性之间的不平衡问题，在单张A100 GPU上仅需~32分钟即可通过3DGS生成高保真3D资产。

**[Sehdr Single-Exposure Hdr Novel View Synthesis Via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)**

:   SeHDR 是首个从单曝光多视图 LDR 图像生成 HDR 新视角的3DGS框架，通过在3D空间扩展包围曝光原理（Bracketed 3D Gaussians）并设计可微分神经曝光融合（NeEF）在球谐空间融合，无需 HDR 监督即超越现有方法 14.3dB。

**[Self-Ensembling Gaussian Splatting For Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)**

:   SE-GS 通过不确定性感知扰动策略在训练过程中动态生成多样化的 3DGS 模型，并利用自集成机制使 Σ-model 聚合扰动模型的信息，有效缓解稀疏视角下的过拟合问题，在多个数据集上实现 SOTA 的少样本新视角合成性能。

**[Self-Supervised Learning Of Hybrid Part-Aware 3D Representations Of 2D Gaussians](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)**

:   提出 PartGS，一个自监督的部件感知3D重建框架，将2D Gaussian Splatting与超二次曲面混合耦合，通过参数共享和多种正则化实现同时高质量几何分解和纹理重建，在DTU、ShapeNet和真实场景上在重建精度上比SOTA提升75.9%，PSNR提升16.13dB。

**[Sequential Gaussian Avatars With Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)**

:   提出 SeqAvatar，利用显式3DGS表示结合层次化运动上下文（粗粒度骨骼运动 + 细粒度逐点速度）建模人体化身的运动相关外观变化，并通过时空多尺度采样增强运动条件的鲁棒性，在多个数据集上取得SOTA渲染质量同时保持实时渲染速度。

**[Shape Of Motion 4D Reconstruction From A Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)**

:   提出基于 $\mathbb{SE}(3)$ 运动基的动态 3D 高斯表示，从单目视频中恢复全局一致的 3D 运动轨迹，同时实现实时新视角合成和长程 3D 跟踪，在 iPhone 和 Kubric 数据集上全面超越先前方法。

**[Sheap Self-Supervised Head Geometry Predictor Learned Via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)**

:   提出SHeaP，利用2D Gaussian Splatting替代传统可微mesh渲染进行自监督3DMM预测训练，通过将Gaussians绑定到3DMM mesh上实现重动画，并设计graph卷积Gaussians生成器和几何一致性正则化，在NoW和Nersemble基准上超越所有自监督方法。

**[Sim3D Single-Instance Multiview Multimodal And Multisetup 3D Anomaly Detection B](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)**

:   提出 SiM3D，首个面向多视角多模态3D异常检测与分割的基准，聚焦工业制造中的单实例场景，通过工业级传感器采集高分辨率数据，使用体素化异常体积(Anomaly Volume)替代2D异常图，并首次支持合成到真实的跨域评估。

**[Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)**

:   Sdirt 提出基于光线追踪的双像素（DP）图像模拟方案，通过精确计算包含像差和相位分裂信息的空间变化 DP PSF，弥合仿真与真实 DP 数据之间的域间差距，使深度估计模型在真实 DP 图像上具有更好的泛化能力。

**[Skysense V2 A Unified Foundation Model For Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)**

:   本文提出SkySense V2，使用单一统一Transformer骨干网络处理高分辨率光学/多光谱/SAR三种遥感模态数据，通过自适应Patch合并、模态特异性Prompt Token和基于Query的语义聚合对比学习（QSACL）进行预训练，仅用665M参数（相比前作SkySense的1.26B）在16个数据集7种任务上平均提升1.8分。

**[Sl2A-Inr Single-Layer Learnable Activation For Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)**

:   提出SL2A-INR，通过单层基于Chebyshev多项式的可学习激活函数块与ReLU-MLP融合块的混合架构，有效缓解隐式神经表示中的频谱偏差问题，在图像拟合、3D形状重建和新视角合成任务上达到SOTA。

**[Sparfels Fast Reconstruction From Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)**

:   提出Sparfels方法，将3D基础模型（MASt3R）与高效的测试时优化（2DGS）相结合，通过MASt3R提供初始化点云/相机和对应关系引导优化，并创新性地引入泼溅色彩方差损失，在3分钟内从稀疏无位姿图像实现SOTA几何重建。

**[Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)**

:   提出 4D Diffusion Policy（DP4），通过动态高斯世界模型为扩散策略注入3D空间和4D时空感知能力，在17个仿真任务和3个真实机器人任务上大幅超越基线（Adroit +16.4%, DexArt +14%, RLBench +6.45%, 真实任务 +8.6%）。

**[Spatialsplat Efficient Semantic 3D From Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)**

:   提出SpatialSplat,通过双场语义表示和选择性Gaussian机制,从稀疏无位姿图像前馈生成紧凑的语义3D Gaussian,将表示参数量减少60%同时超越SOTA方法。

**[Spinmeround Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)**

:   提出 SpinMeRound，一种基于身份嵌入的多视角扩散模型，能从单张或少量人脸图像生成 360° 全头部一致性肖像及对应法线图，在人脸新视角合成任务上超越现有多视角扩散方法。

**[Splattalk 3D Vqa With Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)**

:   提出SplatTalk，利用可泛化的3D Gaussian Splatting框架生成与LLM兼容的3D token，仅需多视角RGB图像即可实现零样本3D视觉问答，性能超越2D LMM方法并接近3D LMM。

**[Stable Score Distillation](stable_score_distillation.md)**

:   提出 Stable Score Distillation (SSD)，通过单分类器跨提示词引导和 null-text 分支的跨轨迹正则化，实现更稳定精准的文本引导 2D/3D 编辑，在保持源内容结构的同时提升编辑对齐度。

**[Stealthattack Robust 3D Gaussian Splatting Poisoning Via Density-Guided Illusion](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)**

:   首次针对3D高斯泼溅(3DGS)提出密度引导的投毒攻击方法，通过在低密度区域注入幻觉物体的高斯点并引入自适应噪声破坏多视角一致性，实现从目标视角清晰可见而不干扰其余视角的隐蔽攻击。

**[Stereo Any Video Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)**

:   提出Stereo Any Video框架，通过融合单目视频深度基础模型先验(Video Depth Anything)、全对全配对相关(all-to-all-pair correlation)和时序凸上采样(temporal convex upsampling)三大核心模块，在不依赖相机位姿或光流的前提下实现空间精确且时序一致的视频立体匹配，在多个数据集零样本设定下达到SOTA。

**[Stochasticsplats Stochastic Rasterization For Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)**

:   StochasticSplats 将随机透明度（Stochastic Transparency）引入 3DGS，通过无偏 Monte Carlo 估计替代深度排序的 alpha 混合，实现免排序、无 popping 的渲染，在 1 SPP 下比标准 CUDA 3DGS 快 4×，并可通过采样数灵活权衡质量与速度。

**[Strandhead Text To Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)**

:   提出 StrandHead，首个通过蒸馏人体特定2D扩散模型来生成发丝级3D头部化身的框架，提出可微棱柱化算法实现发丝到水密网格的转换和梯度反传，并设计基于统计发丝几何先验的正则化损失保证发型的真实性。

**[Strumamba3D Exploring Structural Mamba For Self-Supervised Point Cloud Represent](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)**

:   提出 StruMamba3D，通过为 SSM 的隐含状态赋予空间位置属性（空间状态）来维护 3D 点的邻接关系，并引入序列长度自适应策略解决预训练与下游任务之间的序列长度差异问题，在 ScanObjectNN 最难分割上达到 92.75% 准确率，ModelNet40 达到 95.1%，均为单模态 SOTA。

**[Superdec 3D Scene Decomposition With Superquadrics Primitives](superdec_3d_scene_decomposition_with_superquadrics_primitives.md)**

:   提出SuperDec,基于Transformer的学习方法将点云分解为紧凑的超二次曲面基元集合,在ShapeNet上训练即可泛化到真实场景,支持机器人操作和可控生成。

**[Supermat Physically Consistent Pbr Material Estimation At Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)**

:   提出SuperMat，一个单步推理的PBR材质分解框架，通过结构化专家分支和调度器修正实现端到端训练，引入re-render loss确保物理一致性，将推理速度从秒级提升至毫秒级。

**[Svg-Head Hybrid Surface-Volumetric Gaussians For High-Fidelity Head Reconstructi](svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)**

:   提出SVG-Head，通过表面高斯(显式纹理图)和体积高斯(非朗伯区域补充建模)的混合表示，首次实现高保真高斯头部化身的实时外观编辑。

**[Tar3D Creating High-Quality 3D Assets Via Next-Part Prediction](tar3d_creating_high-quality_3d_assets_via_next-part_prediction.md)**

:   提出TAR3D框架——首次将三平面表示量化为离散几何部件并用GPT自回归生成，通过3D VQ-VAE编码任意面数网格为固定长度序列+TriPE位置编码保留3D空间信息，在文本/图像→3D任务上全面超越现有方法。

**[Text2Vdm Text To Vector Displacement Maps For Expressive And Interactive 3D Scul](text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)**

:   提出Text2VDM,首个从文本生成VDM雕刻笔刷的框架,通过Sobolev预条件网格变形和语义增强SDS损失解决子对象结构生成中的语义耦合问题。

**[Textured 3D Regenerative Morphing With 3D Diffusion Prior](textured_3d_regenerative_morphing_with_3d_diffusion_prior.md)**

:   提出基于3D扩散先验的再生式3D morphing方法，通过在初始噪声、模型参数和条件特征三个层级进行插值，结合Attention Fusion、Token Reordering和Low-Frequency Enhancement三种策略，首次实现了跨类别纹理3D物体的平滑、合理变形序列生成。

**[Timeformer Capturing Temporal Relationships Of Deformable 3D Gaussians For Robus](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)**

:   提出TimeFormer模块,通过跨时间Transformer编码器隐式学习可变形3D Gaussian的时序关系,并设计双流优化策略在训练时迁移运动知识,推理时无额外开销。

**[Tokenunify Scaling Up Autoregressive Pretraining For Neuron Segmentation](tokenunify_scaling_up_autoregressive_pretraining_for_neuron_segmentation.md)**

:   提出 TokenUnify，通过统一随机 token 预测、下一 token 预测和下一全部 token 预测三种互补学习目标，在大规模电子显微镜数据上实现层次化预测编码，将自回归误差累积从 O(K) 降至 O(√K)，下游神经元分割提升 44%。

**[Towards More Diverse And Challenging Pre-Training For Point Cloud Learning Self-](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)**

:   提出Point-PQAE，首个将跨视图重建（Cross Reconstruction）引入3D生成式自监督学习的框架，通过点云裁剪机制生成解耦视图、设计视图相对位置编码（VRPE）和位置查询模块，使预训练更具挑战性和信息量，在ScanObjectNN上以Mlp-Linear协议平均超越Point-MAE 6.7%。

**[Towards Scalable Spatial Intelligence Via 2D-To-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)**

:   提出一个可扩展的数据生成管线，通过集成深度估计、相机标定和尺度校准，将单视图2D图像自动转换为包含点云、相机位姿、深度图的尺度真实3D表示，生成了约200万场景的COCO-3D和Objects365-v2-3D数据集，显著提升多种3D任务性能。

**[Trace3D Consistent Segmentation Lifting Via Gaussian Instance Tracing](trace3d_consistent_segmentation_lifting_via_gaussian_instance_tracing.md)**

:   提出Gaussian Instance Tracing (GIT)机制，通过反向光栅化为每个高斯核维护跨视角的实例权重矩阵，统一解决2D分割多视角不一致和边界高斯模糊两大问题，在离线对比学习和在线自提示两种设定下均显著提升3D分割质量。

**[Trace Learning 3D Gaussian Physical Dynamics From Multi-View Videos](trace_learning_3d_gaussian_physical_dynamics_from_multi-view_videos.md)**

:   提出TRACE框架，将每个3D高斯核视为刚性粒子并为其学习独立的平移-旋转动力学系统（包含速度、加速度、角速度、角加速度等完整物理参数），无需任何人工标注即可从多视角动态视频中学习3D场景的物理运动规律并准确外推未来帧。

**[Tridi Trilateral Diffusion Of 3D Humans Objects And Interactions](tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)**

:   提出 TriDi，首个建模人体(H)、物体(O)和交互(I)三变量联合分布的统一扩散模型，一个网络覆盖 7 种条件生成模式，超越各专用单向基线。

**[Tune-Your-Style Intensity-Tunable 3D Style Transfer With Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)**

:   提出 Tune-Your-Style，首个强度可调的 3D 风格迁移范式，通过 Gaussian 神经元显式建模风格强度并参数化可学习 style tuner，配合两阶段优化策略，实现用户自由调节风格注入的程度。

**[Turboreg Turboclique For Robust And Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)**

:   提出 TurboReg 框架，通过定义轻量级 3-clique（TurboClique）替代传统最大团搜索，并设计高度可并行的 Pivot-Guided Search（PGS）算法，在保持SOTA配准精度的同时将速度提升 208× 以上。

**[Uniegomotion A Unified Model For Egocentric Motion Reconstruction Forecasting An](uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)**

:   提出 UniEgoMotion，首个统一的自中心运动模型，通过条件运动扩散框架和头部中心运动表示，在单一模型中实现自中心视角下的3D人体运动重建、预测和生成三项任务，并发布大规模EE4D-Motion数据集。

**[Unified Category-Level Object Detection And Pose Estimation From Rgb Images Usin](unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)**

:   首次提出将物体检测与类别级位姿估计统一到单一模型的 RGB-only 框架，利用 Neural Mesh Models 作为3D原型表示，通过特征匹配和多模型 RANSAC PnP 同时实现检测和 9D 位姿估计，在 REAL275 上所有 scale-agnostic 指标均超越 SOTA。

**[Univg A Generalist Diffusion Model For Unified Image Generation And Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)**

:   提出UniVG,基于MM-DiT的统一图像生成模型,通过通道维拼接输入、渐进式多任务训练和外部条件注入,用单套权重支持T2I生成、编辑、ID保持、布局引导、深度估计等多种任务。

**[Unleashing Vecset Diffusion Model For Fast Shape Generation](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)**

:   FlashVDM 提出系统性框架加速 Vecset Diffusion Model（VDM）的 DiT 采样和 VAE 解码：通过渐进式流蒸馏将扩散步骤降至 5 步，通过自适应 KV 选择 + 层次体素解码 + 高效解码器将 VAE 解码加速 45×，整体实现 32× 加速至 1 秒内生成高质量 3D 形状。

**[Upp Unified Point-Level Prompting For Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)**

:   提出统一点级提示方法UPP，将点云去噪和补全重新定义为下游任务的提示机制，通过Rectification Prompter过滤噪声、Completion Prompter补全缺失、Shape-Aware Unit捕获几何特征，在噪声和不完整点云上以6.3%参数实现超越全量微调的鲁棒分析。

**[Ust-Ssm Unified Spatio-Temporal State Space Models For Point Cloud Video Modelin](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)**

:   提出UST-SSM，通过时空选择扫描(STSS)、时空结构聚合(STSA)和时序交互采样(TIS)三个核心模块，将选择性状态空间模型扩展到点云视频分析，以线性复杂度实现优于Transformer的性能。

**[Vertexregen Mesh Generation With Continuous Level Of Detail](vertexregen_mesh_generation_with_continuous_level_of_detail.md)**

:   提出VertexRegen，受渐进网格启发将网格生成重新定义为边折叠(edge collapse)的逆操作——顶点分裂(vertex split)的学习，实现连续细节层级的"随时停止"网格生成。

**[Vit-Split Unleashing The Power Of Vision Foundation Models Via Efficient Splitti](vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)**

:   基于"VFM 层可分为低层特征提取器和高层任务适配器"的关键观察，提出 ViT-Split，通过冻结 VFM + task head（复制最后 $K_t$ 层）+ prior head（轻量 CNN 聚合多尺度先验特征）的设计，在 ADE20K 上仅用线性头即达到 58.2 mIoU（DINOv2-L），训练速度提升 4 倍，可训练参数仅为传统适配器的 1/4~1/5。

**[Vivid4D Improving 4D Reconstruction From Monocular Video By Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)**

:   本文提出Vivid4D，将单目视频的多视角增广任务转化为视频修复（inpainting）问题——先用单目深度先验将视频warp到新视角，再用视频扩散模型修复遮挡区域，通过迭代视角扩展策略和鲁棒重建损失显著改善了单目4D动态场景的重建质量。

**[Volume - Authentic 3D Video Calls From Live Gaussian Splat Prediction](volume_-_authentic_3d_video_calls_from_live_gaussian_splat_prediction.md)**

:   微软提出首个从单目2D摄像头实时预测3D高斯泼溅重建的方法，实现真实感、保真性、实时性和时序稳定性四项要求的统一，使任何人仅用标准笔记本摄像头即可进行体积3D视频通话。

**[Volumetricsmpl A Neural Volumetric Body Model For Efficient Interactions Contact](volumetricsmpl_a_neural_volumetric_body_model_for_efficient_interactions_contact.md)**

:   提出 VolumetricSMPL，一种基于 Neural Blend Weights（NBW）的高效神经体积人体模型，相比前代 COAP 实现 10× 推理加速、6× 显存节省，并通过 SDF（而非占据函数）表示提供更精确的可微碰撞建模。

**[Wildseg3D Segment Any 3D Objects In The Wild From 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)**

:   提出 WildSeg3D，首个前馈式3D分割模型，无需场景特定训练，通过动态全局对齐(DGA)解决多视角点图对齐误差，结合多视角组映射(MGM)实现实时交互式3D分割，比现有SOTA快40倍且精度更优。

**[Wonderplay Dynamic 3D Scene Generation From A Single Image And Actions](wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)**

:   WonderPlay 提出混合生成模拟器（Hybrid Generative Simulator），将物理求解器的粗糙3D动态仿真与视频扩散模型的高质量生成相结合，实现从单张图像加用户动作输入生成逼真多材质动态3D场景，支持刚体、布料、液体、烟雾、颗粒等多种材质。

**[Zero-Shot Inexact Cad Model Alignment From A Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)**

:   提出一种弱监督的9-DoF CAD模型对齐方法，通过增强DINOv2特征的几何感知能力并在归一化物体坐标（NOC）空间进行稠密对齐优化，实现无需位姿标注、可泛化到未见类别的零样本3D对齐。

**[Zerostereo Zero-Shot Stereo Matching From Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)**

:   提出 ZeroStereo 管线：从任意单张图像出发，利用单目深度估计生成伪视差，再用微调的扩散修复模型合成高质量右视图，实现只需 35K 合成数据即达到 SOTA 零样本立体匹配泛化性能。
