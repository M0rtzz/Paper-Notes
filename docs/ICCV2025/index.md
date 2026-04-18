---
title: >-
  ICCV2025 1313篇论文解读
description: >-
  1313篇ICCV2025论文深度解读，每篇5分钟读懂核心思想，含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 ICCV2025 论文笔记

**1313** 篇论文解读，覆盖 **41** 个领域。

<div class="conf-index" markdown>

---

## 🧊 3D视觉 { #3d_vision }

**[2D Gaussian Splattingbased Sparseview Transparent Object Dep](3d_vision/2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)**

:   提出TRAN-D，一种基于2D Gaussian Splatting的稀疏视角透明物体深度重建方法，通过分割引导的object-aware损失优化遮挡区域Gaussian分布，并利用物理仿真（MPM）实现物体移除后的场景动态更新，仅需单张图像即可完成场景刷新。

**[3D Gaussian Map With Openset Semantic Grouping For Visionlan](3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)**

:   提出基于3D高斯溅射的场景地图表示（3D Gaussian Map），结合开放集语义分组机制，为视觉-语言导航（VLN）构建兼顾几何结构与丰富语义的3D环境表示，并设计多层级动作预测策略（Multi-Level Action Prediction）融合多粒度空间-语义线索辅助导航决策。

**[3D Mesh Editing Using Masked Lrms](3d_vision/3d_mesh_editing_using_masked_lrms.md)**

:   提出MaskedLRM，将3D形状编辑重构为条件重建问题——训练时随机生成3D遮挡物遮盖多视角输入，用一张干净条件视图引导被遮挡区域的补全；推理时用户定义编辑区域并提供单张编辑图像，模型在**<3秒单次前传**中完成3D网格编辑，比优化方法快2-10倍，能执行拓扑变化编辑（加孔/加把手），重建质量与SOTA持平。

**[3D Test-Time Adaptation Via Graph Spectral Driven Point Shift](3d_vision/3d_test-time_adaptation_via_graph_spectral_driven_point_shift.md)**

:   提出 GSDTTA，将3D点云测试时自适应从空间域转移到图谱域，仅优化最低10%频率分量即可适配点云的全局结构，配合特征图引导的自训练策略，在 ModelNet40-C 和 ScanObjectNN-C 上达到 SOTA。

**[3D Testtime Adaptation Via Graph Spectral Driven Point Shift](3d_vision/3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)**

:   提出GSDTTA，首次将3D点云的测试时适应从空间域转移到图谱域，通过仅优化最低10%频率分量（减少约90%参数）实现全局结构调整，并结合特征图引导的自训练策略生成伪标签，在ModelNet40-C和ScanObjectNN-C上显著超越现有3D TTA方法。

**[3Dgraphllm Combining Semantic Graphs And Large Language Mode](3d_vision/3dgraphllm_combining_semantic_graphs_and_large_language_mode.md)**

:   提出3DGraphLLM，首个将**3D语义场景图的可学习表示**直接输入LLM的方法——通过k近邻子图+三元组(object1, relation, object2)编码物体间语义关系，然后投影到LLM的token嵌入空间。在ScanRefer上Acc@0.5提升+6.4%（vs无语义关系的Chat-Scene），在Multi3DRefer上F1@0.5提升+7.5%，推理速度比GPT4Scene-HDM快5倍。

**[3Dgraphllm Combining Semantic Graphs And Large Language Models For 3D Scene Unde](3d_vision/3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)**

:   本文提出3DGraphLLM，将3D场景中物体间的语义关系编码为可学习的图表示并输入LLM，在object grounding、场景描述和视觉问答等多个3D视觉-语言任务上显著超越不使用语义关系的基线方法，同时推理速度比LVLM方法快5倍。

**[3Dgs-Lm Faster Gaussian-Splatting Optimization With Levenberg-Marquardt](3d_vision/3dgs-lm_faster_gaussian-splatting_optimization_with_levenberg-marquardt.md)**

:   本文提出3DGS-LM，用定制的Levenberg-Marquardt优化器替换3DGS中的ADAM优化器，通过高效的GPU缓存驱动并行化方案实现Jacobian-向量积的快速计算，在保持相同重建质量的前提下将3DGS优化速度提升20%。

**[3Dgslm Faster Gaussiansplatting Optimization With Levenbergm](3d_vision/3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)**

:   将3D Gaussian Splatting的ADAM优化器替换为定制化的Levenberg-Marquardt（LM）二阶优化器，通过高效CUDA并行化的PCG算法和梯度缓存结构实现Jacobian-向量积加速，在保持相同重建质量的前提下将优化时间缩短约20%。

**[4D Gaussian Splatting Slam](3d_vision/4d_gaussian_splatting_slam.md)**

:   提出首个完整的4D Gaussian Splatting SLAM系统，在动态场景中同时进行相机位姿跟踪和4D高斯辐射场重建——将高斯原语分为静态/动态集合，通过稀疏控制点+MLP建模动态物体运动，并创新性地设计2D光流图渲染算法来监督动态高斯的运动学习。

**[4D Visual Pre-Training For Robot Learning](3d_vision/4d_visual_pre-training_for_robot_learning.md)**

:   FVP提出了一种基于4D（3D空间+时间）点云预测的视觉预训练框架，通过将预训练目标建模为"下一帧点云预测"并用扩散模型实现，显著提升了多种3D模仿学习方法在真实机器人操作任务上的成功率（DP3平均提升28%）。

**[4D Visual Pretraining For Robot Learning](3d_vision/4d_visual_pretraining_for_robot_learning.md)**

:   FVP提出将3D视觉预训练建模为"下一帧点云预测"问题，用条件扩散模型从历史帧点云预测未来帧点云来学习3D视觉表示，在12个真实世界操作任务中将DP3的平均成功率提升28%，达到SOTA水平。

**[7Dgs Unified Spatial-Temporal-Angular Gaussian Splatting](3d_vision/7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)**

:   将3DGS扩展到7维（空间3D+时间1D+方向3D），通过条件切片机制将7D高斯投影为与3DGS管线兼容的3D高斯，在具有视角依赖效果的动态场景上PSNR提升最高7.36dB，同时维持401 FPS实时渲染。

**[7Dgs Unified Spatialtemporalangular Gaussian Splatting](3d_vision/7dgs_unified_spatialtemporalangular_gaussian_splatting.md)**

:   提出7DGS，将场景元素建模为**7维高斯分布**（3D空间+1D时间+3D视角方向），通过条件切片机制将7D高斯转换为与时间和视角相关的条件3D高斯，统一处理动态场景+视角依赖效果，在自定义7DGS-PBR数据集上比4DGS PSNR提升高达7.36dB，仅用15.3%的高斯点数，401FPS实时渲染。

**[A3Gs Arbitrary Artistic Style Into Arbitrary 3D Gaussian Spl](3d_vision/a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)**

:   提出A³GS，首个**前馈式零样本3DGS风格迁移**网络——使用图卷积网络(GCN)自编码器将3DGS场景编码到潜在空间，通过AdaIN注入任意风格图像特征，仅需**10秒**即可将任意风格迁移到任意3D场景，无需逐风格优化，可处理大规模3DGS场景。

**[A Lesson In Splats Teacher-Guided Diffusion For 3D Gaussian Splats Generation Wi](3d_vision/a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)**

:   本文提出了一种仅使用2D图像监督来训练3D扩散模型的新框架——通过将确定性3D重建模型作为"噪声教师"生成3D噪声样本，并结合多步去噪策略和循环一致性正则化，实现了超越教师模型的3D高斯喷溅生成质量（PSNR提升0.5-0.85）。

**[A Lesson In Splats Teacherguided Diffusion For 3D Gaussian S](3d_vision/a_lesson_in_splats_teacherguided_diffusion_for_3d_gaussian_s.md)**

:   提出一种用2D图像监督训练3D扩散模型的框架：利用预训练的确定性3D重建模型作为"噪声教师"生成3D噪声样本，通过多步去噪策略和渲染损失实现跨模态（3D去噪+2D监督）训练，在用更小模型的情况下超越教师模型0.5-0.85 PSNR。

**[A Recipe For Generating 3D Worlds From A Single Image](3d_vision/a_recipe_for_generating_3d_worlds_from_a_single_image.md)**

:   将单图到3D世界生成分解为两个更简单的子问题——全景合成（无训练in-context learning）和点云条件修复（仅5k步微调ControlNet），结合3DGS重建出可在VR中2米立方体范围内导航的沉浸式3D环境，在图像质量指标上全面超越WonderJourney和DimensionX等SOTA方法。

**[A Simple Yet Mighty Hartley Diffusion Versatilist For Genera](3d_vision/a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)**

:   提出HarDiff——基于离散Hartley变换的频域学习策略，通过低频训练（从源域提取结构先验）和高频采样（利用目标域细节引导）增强扩散模型在稠密视觉任务上的跨域泛化能力，在语义分割、深度估计和去雾等12个基准上取得SOTA。

**[A Unified Interpretation Of Training-Time Out-Of-Distribution Detection](3d_vision/a_unified_interpretation_of_training-time_out-of-distribution_detection.md)**

:   从输入变量间"交互"的新视角出发，统一解释了不同训练时 OOD 检测方法为何有效——它们都促使模型编码更多高阶交互，并进一步验证了高阶交互在 OOD 检测中的主导作用，以及 near-OOD 样本难以检测的交互分布原因。

**[Aaa-Gaussians Anti-Aliased And Artifact-Free 3D Gaussian Rendering](3d_vision/aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)**

:   AAA-Gaussians提出了一种统一的3D高斯光栅化框架，通过自适应3D平滑滤波器、视空间透视正确边界计算和基于视锥体的3D裁剪，在单一框架内同时解决了3DGS的锯齿、投影畸变和闪烁三大顽疾，在分布外视角评估中大幅领先其他方法，同时保持实时渲染性能。

**[Aaagaussians Antialiased And Artifactfree 3D Gaussian Render](3d_vision/aaagaussians_antialiased_and_artifactfree_3d_gaussian_render.md)**

:   通过在3DGS渲染管线的所有环节中融入完整的3D评估（而非2D splat近似），提出自适应3D平滑滤波器、视空间边界计算和基于视锥的tile剔除，统一解决了3DGS中的锯齿、投影伪影和弹出伪影（popping），在OOD视角下大幅优于现有方法，同时保持实时渲染（>100 FPS）。

**[Accelerate 3D Object Detection Models Via Zero-Shot Attention Key Pruning](3d_vision/accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)**

:   提出 tgGBC（trim keys gradually Guided By Classification scores），一种零样本运行时剪枝方法，利用分类分数与注意力图的乘积计算键重要性，逐层剪除不重要的键，在多个3D检测器上实现Transformer解码器近2×加速且性能损失<1%。

**[Adahuman Animatable Detailed 3D Human Generation With Compos](3d_vision/adahuman_animatable_detailed_3d_human_generation_with_compos.md)**

:   提出AdaHuman框架，通过姿态条件的联合3D扩散模型（在扩散过程中同步进行多视角图像生成与3DGS重建以保证3D一致性）和组合式3DGS细化模块（利用crop-aware camera ray map融合局部精细细节），从单张野外图片生成高保真可动画的3D人体avatar，在重建和重姿态任务上全面超越现有SOTA。

**[Adahuman Animatable Detailed 3D Human Generation With Compositional Multiview Di](3d_vision/adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)**

:   提出AdaHuman框架，通过姿态条件化的3D联合扩散模型和组合式3DGS细化模块，从单张图片生成高精度、可动画化的3D人体虚拟人。

**[Advancing Text-To-3D Generation With Linearized Lookahead Variational Score Dist](3d_vision/advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)**

:   通过分析 VSD 中 LoRA 模型与 3D 模型的优化顺序不匹配问题，提出线性化前瞻（Linearized Lookahead）修正项 $L^2$-VSD，仅需额外一次前向传播即可显著提升 text-to-3D 生成质量。

**[Adversarial Exploitation Of Data Diversity Improves Visual L](3d_vision/adversarial_exploitation_of_data_diversity_improves_visual_l.md)**

:   提出RAP（Robust Absolute Pose regression）——基于外观感知3DGS的双分支联合训练框架，通过对抗判别器弥合合成-真实域差距+外观/位姿增强数据作为额外监督，在Cambridge Landmarks上平移/旋转误差分别降低38-50%/41-44%，在日夜场景和驾驶场景中表现尤为突出。

**[Adversarial Exploitation Of Data Diversity Improves Visual Localization](3d_vision/adversarial_exploitation_of_data_diversity_improves_visual_localization.md)**

:   提出RAP框架，通过外观可变的3DGS合成多样化训练数据，并引入对抗判别器弥合合成-真实域差距，使绝对姿态回归方法在多个数据集上大幅超越SOTA——室内平移/旋转误差降低50%/41%，室外降低38%/44%。

**[Ajahr Amputated Joint Aware 3D Human Mesh Recovery](3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)**

:   首个面向截肢者的3D人体网格恢复框架——通过合成100万+截肢者图像(A3D)、设计BPAC-Net截肢分类器区分截肢与遮挡、以及双Tokenizer切换策略分别编码截肢/正常位姿先验，在截肢者数据上大幅领先(ITW-amputee上MVE比TokenHMR低16.87)，非截肢者数据上也保持竞争力。

**[Amodal3R Amodal 3D Reconstruction From Occluded 2D Images](3d_vision/amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)**

:   提出Amodal3R，一个端到端的遮挡感知3D重建模型，通过在TRELLIS基础上引入mask加权交叉注意力和遮挡感知注意力层，直接在3D潜空间中从部分遮挡的2D图像重建完整的3D物体形状和外观，大幅超越先前"2D补全→3D重建"的两阶段方法。

**[Amodal Depth Anything Amodal Depth Estimation In The Wild](3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)**

:   提出非模态相对深度估计新范式，构建大规模真实数据集ADIW（564K），基于Depth Anything V2和DepthFM设计两个互补框架（Amodal-DAV2和Amodal-DepthFM），通过最小化修改预训练模型实现遮挡区域深度预测，在ADIW上RMSE比之前SOTA提升27.4%。

**[Animateanymesh A Feedforward 4D Foundation Model For Textdri](3d_vision/animateanymesh_a_feedforward_4d_foundation_model_for_textdri.md)**

:   提出AnimateAnyMesh，首个前馈式文本驱动通用Mesh动画框架：通过DyMeshVAE将动态Mesh分解为初始位置和相对轨迹并压缩到潜空间，再用基于Rectified Flow的MMDiT模型学习文本条件下的轨迹分布，配合4M+规模的DyMesh数据集训练，在6秒内即可为任意拓扑Mesh生成高质量动画，全面碾压DG4D、L4GM和Animate3D。

**[Anyi2V Animating Any Conditional Image With Motion Control](3d_vision/anyi2v_animating_any_conditional_image_with_motion_control.md)**

:   提出AnyI2V，一个无需训练的框架，可接受任意模态图像（mesh、点云、深度图、骨架等）作为首帧条件，结合用户定义的轨迹实现运动控制的视频生成，在FID/FVD/ObjMC指标上优于现有training-free方法并与训练方法竞争。

**[Ar1To3 Single Image To Consistent 3D Object Via Nextview Pre](3d_vision/ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)**

:   提出AR-1-to-3，一种基于扩散模型的自回归下一视角预测框架，通过"先近后远"的渐进式生成策略，配合Stacked-LE（堆叠局部特征编码）和LSTM-GE（全局特征编码）两种条件注入机制，显著提升了单图到多视角生成的一致性，在GSO数据集上PSNR达13.18（相比InstantMesh的10.67提升23.5%），Chamfer Distance降至0.063（InstantMesh为0.117）。

**[Articulate3D Holistic Understanding Of 3D Scenes As Universa](3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universa.md)**

:   提出Articulate3D（280个真实室内场景、8类铰接标注的大规模数据集）和USDNet（基于Mask3D扩展的统一框架），通过密集逐点预测机制同时完成可动零件分割和运动参数估计，在铰接参数预测上比Mask3D†提升5.7%，并支持LLM场景编辑和机器人策略训练。

**[Articulate3D Holistic Understanding Of 3D Scenes As Universal Scene Description](3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)**

:   本文提出Articulate3D——首个大规模真实世界室内场景铰接标注数据集（280个高质量扫描），以及USDNet统一框架，能从3D点云同时预测可移动/可交互部件分割和运动参数，为具身AI的物理仿真提供了simulation-ready的场景数据。

**[Atlas Decoupling Skeletal And Shape Parameters For Expressiv](3d_vision/atlas_decoupling_skeletal_and_shape_parameters_for_expressiv.md)**

:   提出ATLAS参数化人体模型，通过显式解耦外部表面形状和内部骨骼参数，并引入稀疏非线性姿态校正变形，在60万高分辨率扫描数据上训练，实现了比SMPL-X更精确、更可控的3D人体建模。

**[Atlas Decoupling Skeletal And Shape Parameters For Expressive Parametric Human M](3d_vision/atlas_decoupling_skeletal_and_shape_parameters_for_expressive_parametric_human_m.md)**

:   提出 ATLAS 参数化人体模型，通过显式解耦外部表面形状与内部骨骼参数，结合稀疏非线性姿态校正，在 60 万高分辨率扫描上训练，实现比 SMPL-X 更精确可控的人体建模。

**[Auto-Regressively Generating Multi-View Consistent Images](3d_vision/auto-regressively_generating_multi-view_consistent_images.md)**

:   提出 MV-AR，首次将自回归模型引入多视图图像生成，利用所有先前视图作为条件逐步生成后续视图，配合统一的多模态条件注入模块和 Shuffle View 数据增强，在文本/图像/形状条件下均达到与扩散模型可比的一致性。

**[Autoocc Automatic Openended Semantic Occupancy Annotation Vi](3d_vision/autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)**

:   提出AutoOcc，一个以视觉为中心的全自动开放式语义占据标注流水线，通过视觉-语言模型引导的可微高斯泼溅（VL-GS）实现无需人工标签的3D语义占据生成，在Occ3D-nuScenes上以纯视觉输入就达到IoU 83.01/mIoU 20.92，大幅超越现有自动标注方法。

**[Back On Track Bundle Adjustment For Dynamic Scene Reconstruc](3d_vision/back_on_track_bundle_adjustment_for_dynamic_scene_reconstruc.md)**

:   提出BA-Track框架，通过学习型3D点追踪器将观测到的运动解耦为相机引起的运动和物体自身运动，使传统束调整(BA)能够无差别地处理静态和动态点，在相机位姿估计(ATE在Sintel上达到0.034，较SOTA降低一半以上)和稠密3D重建上取得显著提升。

**[Back On Track Bundle Adjustment For Dynamic Scene Reconstruction](3d_vision/back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)**

:   提出 BA-Track 框架，利用 3D 点追踪器将观测运动分解为相机运动和物体运动，使传统 Bundle Adjustment 能同时处理静态与动态场景元素，实现精确的相机位姿估计和时间一致的稠密重建。

**[Baking Gaussian Splatting Into Diffusion Denoiser For Fast A](3d_vision/baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)**

:   提出DiffusionGS，将3D高斯点云直接嵌入扩散模型的去噪器中，通过单阶段3D扩散实现从单张图片到3D物体生成和场景重建，在ABO/GSO上PSNR超越SOTA 2.20/1.25 dB，RealEstate10K上超1.34 dB，推理速度约6秒（A100）。

**[Baking Gaussian Splatting Into Diffusion Denoiser For Fast And Scalable Single-S](3d_vision/baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)**

:   提出DiffusionGS，将3D高斯点云"烘焙"进扩散模型的去噪器中，实现单阶段、视图一致的单视图3D物体生成和场景重建，配合场景-物体混合训练策略和RPPC相机条件编码，在PSNR/FID上大幅超越现有方法，推理速度仅需约6秒。

**[Banet Bilateral Aggregation Network For Mobile Stereo Matchi](3d_vision/banet_bilateral_aggregation_network_for_mobile_stereo_matchi.md)**

:   提出双边聚合网络BANet，通过将代价体分离为高频细节体和低频平滑体分别聚合再融合，仅使用2D卷积即可在移动设备上实现实时高精度立体匹配（骁龙8 Gen 3上45ms，KITTI 2015 D1-all=1.83%，比MobileStereoNet-2D精度高35.3%）。

**[Banet Bilateral Aggregation Network For Mobile Stereo Matching](3d_vision/banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)**

:   提出双边聚合网络BANet，通过空间注意力将代价体分离为高频细节体和低频平滑体并分别聚合，仅使用2D卷积即可在移动设备上实时运行并大幅超越MobileStereoNet-2D（KITTI 2015上精度提升35.3%），3D版本在GPU上达到实时方法最高精度。

**[Benchmarking And Learning Multi-Dimensional Quality Evaluator For Text-To-3D Gen](3d_vision/benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)**

:   构建了包含1280个文本生成3D模型的多维度基准MATE-3D（8类prompt × 8种方法 × 4维评分 × 21名标注者），并提出基于超网络的多维度质量评估器HyperScore，通过条件特征融合和自适应映射在所有评估维度上超越现有指标。

**[Benchmarking And Learning Multidimensional Quality Evaluator](3d_vision/benchmarking_and_learning_multidimensional_quality_evaluator.md)**

:   构建MATE-3D基准（8类prompt×8种方法=1280个textured mesh，4维度×21人主观评分=107520标注）并提出HyperScore多维质量评估器：通过可学习条件特征+条件特征融合(模拟注意力转移)+超网络生成维度自适应映射函数(模拟决策过程变化)，在语义对齐、几何、纹理、整体4个维度上全面超越现有指标。

**[Benchmarking Egocentric Visualinertial Slam At City Scale](3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)**

:   提出 LaMAria——首个城市尺度的第一人称多传感器 VIO/SLAM 基准数据集，利用测绘级控制点提供厘米精度的地面真值，系统评估了学术界主流 SLAM 方案在真实第一人称场景下的表现，揭示了现有方法与商业系统之间的巨大差距。

**[Beziergs Dynamic Urban Scene Reconstruction With Bezier Curv](3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)**

:   用可学习的Bézier曲线显式建模动态物体的运动轨迹，替代传统依赖精确bbox标注的范式，实现了对自动驾驶街景中动/静态成分的准确分离与高保真重建。

**[Beziergs Dynamic Urban Scene Reconstruction With Bezier Curve Gaussian Splatting](3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)**

:   提出用可学习的Bézier曲线建模动态物体运动轨迹的3D高斯溅射方法（BezierGS），摆脱对精确目标标注框的依赖，在Waymo和nuPlan数据集上的动态和静态场景重建均达到SOTA。

**[Billboard Splatting Bbsplat Learnable Textured Primitives Fo](3d_vision/billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)**

:   提出BBSplat——用可学习的RGB纹理和alpha贴图替代2D Gaussian Splatting中的高斯分布不透明度，使每个平面基元具有任意形状和逐像素颜色控制，在用更少基元的情况下弥补2DGS与3DGS之间的渲染质量差距，同时保留精确网格提取能力并实现最高×17的存储压缩。

**[Blended Point Cloud Diffusion For Localized Textguided Shape](3d_vision/blended_point_cloud_diffusion_for_localized_textguided_shape.md)**

:   提出 BlendedPC，将局部文本引导的3D形状编辑重新定义为语义inpainting问题，通过在Point·E基础上训练Inpaint-E模型，并在推理时引入无需反演(inversion-free)的坐标混合(coordinate blending)机制，在保持原始形状身份的同时实现精准局部编辑，在ShapeTalk数据集上全面超越现有方法。

**[Bokehdiff Neural Lens Blur With One-Step Diffusion](3d_vision/bokehdiff_neural_lens_blur_with_one-step_diffusion.md)**

:   BokehDiff提出基于预训练扩散模型的单步推理散景渲染方法，通过物理启发的自注意力模块（PISA）融入能量守恒、弥散圆约束和自遮挡效果，配合合成前景数据训练，在深度不连续区域显著优于传统方法。

**[Bolt3D Generating 3D Scenes In Seconds](3d_vision/bolt3d_generating_3d_scenes_in_seconds.md)**

:   提出一种基于潜在扩散模型的前馈式3D场景生成方法，通过将3D场景表示为多组Splatter Image并使用专门训练的几何VAE，**在单GPU上7秒内生成完整3D场景**，推理成本比优化式方法（CAT3D）降低300倍。

**[Boost 3D Reconstruction Using Diffusion-Based Monocular Camera Calibration](3d_vision/boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)**

:   提出 DM-Calib，利用 Stable Diffusion 先验进行单目相机内参估计，设计了 Camera Image 表示将内参无损编码为图像，结合 RANSAC 解算焦距和光心，在5个零样本数据集上大幅超越现有标定方法，并推进了度量深度估计、位姿估计和稀疏视图重建等下游任务。

**[Boost 3D Reconstruction Using Diffusionbased Monocular Camer](3d_vision/boost_3d_reconstruction_using_diffusionbased_monocular_camer.md)**

:   提出DM-Calib——基于扩散模型的单目相机内参估计方法：设计Camera Image表示（将内参无损编码为3通道图像=方位角+仰角+灰度图），微调Stable Diffusion生成Camera Image，用RANSAC提取内参，在5个零样本数据集上超越所有基线，并将相机标定扩展到度量深度估计、位姿估计和稀疏视角3D重建。

**[Boosting Multi-View Indoor 3D Object Detection Via Adaptive 3D Volume Constructi](3d_vision/boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume_constructi.md)**

:   SGCDet 通过自适应稀疏3D体素构建和几何-上下文感知聚合，实现了高效精准的多视图室内3D目标检测，无需真实几何监督即超越现有方法。

**[Boosting Multiview Indoor 3D Object Detection Via Adaptive 3](3d_vision/boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)**

:   SGCDet通过几何与上下文感知的聚合模块（3D可变形注意力+多视角注意力融合）和基于占据概率的稀疏体素构建策略，在无需ground-truth几何监督的情况下，实现了多视角室内3D目标检测的SOTA性能，同时大幅降低计算开销。

**[Bootstrap3D Improving Multi-View Diffusion Model With Synthetic Data](3d_vision/bootstrap3d_improving_multi-view_diffusion_model_with_synthetic_data.md)**

:   提出 Bootstrap3D 框架，利用 2D/视频扩散模型自动生成 100 万张高质量多视角图像配精细文本描述，并通过训练时间步重调度（TTR）策略在微调多视角扩散模型时平衡图像质量与视角一致性，显著提升文本到 3D 生成的质量。

**[Bootstrap3D Improving Multiview Diffusion Model With Synthet](3d_vision/bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)**

:   提出Bootstrap3D框架，利用视频扩散模型生成合成多视图数据，并通过微调的MV-LLaVA进行质量过滤与密集描述重写，结合Training Timestep Reschedule (TTR)策略训练多视图扩散模型，在不牺牲视图一致性的前提下大幅提升图像质量和文本对齐能力。

**[Boxdreamer Dreaming Box Corners For Generalizable Object Pos](3d_vision/boxdreamer_dreaming_box_corners_for_generalizable_object_pos.md)**

:   提出以3D包围盒角点作为中间表示，通过Transformer解码器预测查询视图中角点的2D投影热图，结合PnP算法实现可泛化的稀疏视角6DoF物体位姿估计，在遮挡和稀疏视角场景下显著优于现有方法。

**[Boxdreamer Dreaming Box Corners For Generalizable Object Pose Estimation](3d_vision/boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)**

:   提出 BoxDreamer，以 3D 包围盒角点作为中间表示，通过基于参考视角的点合成器预测查询图像中的 2D 角点投影，建立 2D-3D 对应关系后用 PnP 算法恢复物体位姿，在稀疏视角和严重遮挡场景下显著优于现有方法。

**[Bridging 3D Anomaly Localization And Repair Via High-Quality Continuous Geometri](3d_vision/bridging_3d_anomaly_localization_and_repair_via_high-quality_continuous_geometri.md)**

:   提出 PASDF 框架，通过姿态感知的签名距离函数（SDF）实现连续几何表征，统一了3D异常检测与修复任务，在 Real3D-AD 和 Anomaly-ShapeNet 上取得 SOTA。

**[Bridging 3D Anomaly Localization And Repair Via Highquality](3d_vision/bridging_3d_anomaly_localization_and_repair_via_highquality.md)**

:   提出PASDF框架，通过姿态对齐模块(PAM)将点云对齐到标准姿态 + 神经SDF网络学习连续几何表示 + 基于SDF偏差的异常评分，统一实现3D点云异常检测与异常修复(Marching Cubes提取零等值面作为修复模板)，在Real3D-AD上O-AUROC 80.2%、Anomaly-ShapeNet上90.0%均达SOTA。

**[Bridging Diffusion Models And 3D Representations A 3D Consis](3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consis.md)**

:   提出3DSR——将扩散超分模型与3DGS表示交替迭代实现3D一致超分：每步去噪后将SR图像训练到3DGS中获得3D一致渲染→重编码回潜在空间引导下一步去噪，无需微调任何模型即显式保证跨视角一致性，在LLFF上PSNR提升1.16dB+FID降低50%(vs StableSR)。

**[Bridging Diffusion Models And 3D Representations A 3D Consistent Super-Resolutio](3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)**

:   提出3DSR框架，将扩散模型的2D超分辨率与3D高斯溅射（3DGS）表示相结合，在每个扩散去噪步骤中通过3DGS渲染来强制多视图3D一致性，实现高保真且空间一致的3D场景超分辨率。

**[Bring Your Rear Cameras For Egocentric 3D Human Pose Estimat](3d_vision/bring_your_rear_cameras_for_egocentric_3d_human_pose_estimat.md)**

:   首次研究HMD后置相机对全身姿态追踪的价值，提出Transformer-based多视角热力图精炼模块(利用可变形注意力+不确定性感知遮罩)，解决后视角2D关节检测不可靠的问题，并发布两个大规模数据集(Ego4View-Syn/RW)，在Ego4View-RW上MPJPE比SOTA EgoPoseFormer提升>10%(63.38→56.94mm)。

**[Bring Your Rear Cameras For Egocentric 3D Human Pose Estimation](3d_vision/bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)**

:   首次研究HMD后置相机对自中心3D全身姿态估计的价值，提出基于Transformer的多视图热图细化方法，结合不确定性感知掩码机制，在新建的Ego4View数据集上实现>10% MPJPE提升。

**[Buffer-X Towards Zero-Shot Point Cloud Registration In Diverse Scenes](3d_vision/buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)**

:   提出 BUFFER-X，一种无需人工参数调优的零样本点云配准方法，通过自适应体素大小/搜索半径估计、FPS 替代学习型关键点检测器、以及 patch 级坐标归一化，在 11 个数据集上实现开箱即用的跨域泛化。

**[Bufferx Towards Zeroshot Point Cloud Registration In Diverse](3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)**

:   通过几何自适应bootstrapping确定体素大小/搜索半径、用FPS替代学习型关键点检测器、以及patch级坐标归一化，构建了一个无需人工调参即可在11个跨域数据集上实现零样本点云配准的pipeline BUFFER-X，在室内外多传感器多场景下取得了平均排名第一的成功率。

**[Ca-I2P Channel-Adaptive Registration Network With Global Optimal Selection](3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)**

:   提出 CA-I2P，通过 Channel Adaptive Adjustment Module (CAA) 增强并过滤图像-点云特征的通道差异，并用 Global Optimal Selection (GOS) 基于最优传输替代 top-k 选择减少多对一匹配误差，在 RGB-D Scenes V2 和 7-Scenes 上实现图像-点云配准 SOTA。

**[Cad-Recode Reverse Engineering Cad Code From Point Clouds](3d_vision/cad-recode_reverse_engineering_cad_code_from_point_clouds.md)**

:   提出 CAD-Recode，将点云翻译为可执行的 Python CadQuery 代码来重建 CAD 模型，利用预训练 LLM（Qwen2-1.5B）作为解码器配合轻量级点云编码器，在 DeepCAD、Fusion360 和 CC3D 三个基准上实现了 10 倍以上的 Chamfer Distance 降低。

**[CAD-Recode: Reverse Engineering CAD Code from Point Clouds](3d_vision/cadrecode_reverse_engineering_cad_code_from_point_clouds.md)**

:   将CAD sketch-extrude序列表示为Python代码，利用轻量级点云投影器 + 预训练LLM解码器将点云翻译为可执行Python代码来重建CAD模型，在DeepCAD/Fusion360/真实世界CC3D数据集上显著超越现有方法，且输出代码可被通用LLM理解用于CAD编辑和问答。

**[Can3Tok Canonical 3D Tokenization And Latent Modeling Of Sce](3d_vision/can3tok_canonical_3d_tokenization_and_latent_modeling_of_sce.md)**

:   提出Can3Tok——首个场景级3DGS VAE：通过cross-attention将大量(40K)无序3D Gaussian压缩到低维canonical token(256×768→64×64×4) + 3DGS归一化解决跨场景尺度不一致 + 语义感知过滤去除floater噪声，在DL3DV-10K上唯一成功的场景级3DGS潜在建模方法(L2=30.1, 失败率2.5%)，支持text-to-3DGS和image-to-3DGS前馈生成。

**[Can3Tok Canonical 3D Tokenization And Latent Modeling Of Scene-Level 3D Gaussian](3d_vision/can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)**

:   提出 Can3Tok，首个可将场景级3DGS编码到低维潜空间的变分自编码器，通过规范化查询（canonical query）的交叉注意力实现高效tokenization，配合3DGS归一化和语义感知过滤解决尺度不一致问题，在DL3DV-10K上成功泛化到新场景。

**[Casp Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspo](3d_vision/casp_improving_semi-dense_feature_matching_pipeline_leveraging_cascaded_correspo.md)**

:   提出 CasP，一种级联匹配流水线，将匹配阶段分解为 1/16 尺度的一对多先验匹配和 1/8 尺度的一对一精细匹配，在保持精度的同时实现最高 2.2× 加速，并显著提升跨域泛化能力。

**[Catsplat Contextaware Transformer With Spatial Guidance For](3d_vision/catsplat_contextaware_transformer_with_spatial_guidance_for.md)**

:   提出CATSplat——单视图前馈3DGS重建的泛化Transformer框架：利用VLM文本嵌入（上下文先验）和3D点云特征（空间先验）通过双重cross-attention增强图像特征，在RE10K等数据集上在PSNR/SSIM/LPIPS全面超越Flash3D，且跨数据集泛化性优异。

**[Cf3 Compact And Fast 3D Feature Fields](3d_vision/cf3_compact_and_fast_3d_feature_fields.md)**

:   提出 CF³ 管线，通过 top-down 特征提升、per-Gaussian 自编码器压缩和自适应稀疏化，仅使用原始 Gaussian 数量的 5% 即可构建紧凑高速的 3D 特征场，实现 121–245× 的存储压缩和实时渲染。

**[Charm3R Towards Unseen Camera Height Robust Monocular 3D Det](3d_vision/charm3r_towards_unseen_camera_height_robust_monocular_3d_det.md)**

:   通过数学推导发现回归深度和地面深度在相机高度变化时呈现方向相反的误差趋势，CHARM3R 直接在模型内对两种深度做简单平均来抵消趋势，从而大幅提升单目3D检测器对未见相机高度的泛化能力（CARLA 上提升超过 45%）。

**[Charm3R Towards Unseen Camera Height Robust Monocular 3D Detector](3d_vision/charm3r_towards_unseen_camera_height_robust_monocular_3d_detector.md)**

:   通过数学证明回归深度和地平面深度在相机高度变化时具有相反的外推趋势，提出CHARM3R在模型内简单平均两种深度估计来抵消趋势，实现Mono3D对未见相机高度的鲁棒泛化，AP3D提升超过45%。

**[Cl-Splats Continual Learning Of Gaussian Splatting With Local Optimization](3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)**

:   提出 CL-Splats，一种基于 3D Gaussian Splatting 的持续学习框架，通过 DINOv2 变化检测、2D→3D 掩码提升和球体约束的局部优化，从稀疏新视图高效增量更新场景重建，在合成和真实场景上大幅超越 CL-NeRF 等方法（PSNR：40.1 vs 30.1 dB），并支持历史恢复和并发更新等应用。

**[Clip-Gs Unifying Vision-Language Representation With 3D Gaussian Splatting](3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)**

:   提出 CLIP-GS，首个基于 3D Gaussian Splatting (3DGS) 的多模态表示学习框架。通过 GS Tokenizer 将 3DGS 序列化为 token，结合图像投票损失 (Image Voting Loss) 进行多模态对齐，在跨模态检索、零样本和少样本 3D 分类任务上全面超越基于点云的方法。

**[Cmt A Cascade Mar With Topology Predictor For Multimodal Conditional Cad Generat](3d_vision/cmt_a_cascade_mar_with_topology_predictor_for_multimodal_conditional_cad_generat.md)**

:   提出 CMT，首个基于 B-Rep 表示的多模态 CAD 生成框架，通过级联 MAR（先边后面）和拓扑预测器实现精确拓扑和几何生成，并构建了 130 万级多模态 CAD 数据集 mmABC。

**[Comogaussian Continuous Motionaware Gaussian Splatting From](3d_vision/comogaussian_continuous_motionaware_gaussian_splatting_from.md)**

:   用Neural ODE建模曝光时间内的连续相机运动轨迹，结合刚体变换和可学习的连续运动修正(CMR)变换，从运动模糊图像重建清晰3D高斯场景，在所有benchmark上大幅超越SOTA。

**[Compression Of 3D Gaussian Splatting With Optimized Feature Planes And Standard ](3d_vision/compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)**

:   本文提出 CodecGS，通过将 3DGS 的所有高斯属性用紧凑的 Tri-plane 特征平面表示，并结合频率域 DCT 熵建模和通道级比特分配策略，使特征平面能高效利用标准视频编解码器（HEVC）压缩，实现在保持高渲染质量的同时将存储大小减少至约10MB以内（相比原始3DGS压缩比高达146×）。

**[Constraint-Aware Feature Learning For Parametric Point Cloud](3d_vision/constraint-aware_feature_learning_for_parametric_point_cloud.md)**

:   提出首个面向参数化点云的约束感知特征学习方法 CstNet，将 CAD 约束编码为点级别的 MAD-Adj-PT 三元组表示，通过两阶段网络（约束获取 + 约束特征学习）在自建的 Param20K 数据集上实现分类精度 +3.49%、旋转鲁棒性 +26.17% 的 SOTA 提升。

**[Contact-Aware Amodal Completion For Human-Object Interaction Via Multi-Regional ](3d_vision/contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)**

:   提出首个面向人物交互（HOI）场景的非模态补全框架，利用人体拓扑和接触信息通过凸包操作识别遮挡区域，结合多区域修复策略在预训练扩散模型上无需额外训练即可完成高质量的遮挡物体补全。

**[Curve-Aware Gaussian Splatting For 3D Parametric Curve Reconstruction](3d_vision/curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)**

:   提出 CurveGaussian，通过在参数曲线与边缘导向高斯原语之间建立双向耦合机制，实现从多视图边缘图直接端到端优化 3D 参数曲线的一阶段方法，消除了两阶段管线的误差累积，在精度、效率和紧凑性上全面超越先前方法。

**[Cuts3D Cutting Semantics In 3D For 2D Unsupervised Instance Segmentation](3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)**

:   提出CutS3D方法，首次将3D信息（单目深度估计）引入无监督实例分割，通过在3D点云中切割语义区域来分离2D中重叠的实例，并引入空间置信度机制提升伪标签质量，在多个基准上超越CutLER等SoTA。

**[Dap-Mae Domain-Adaptive Point Cloud Masked Autoencoder For Effective Cross-Domai](3d_vision/dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)**

:   提出 DAP-MAE，通过异构域适配器（HDA）和域特征生成器（DFG）协同学习多域点云数据，仅需一次预训练即可适配物体分类、表情识别、部件分割和3D检测等多种下游任务。

**[Dapmae Domainadaptive Point Cloud Masked Autoencoder For Eff](3d_vision/dapmae_domainadaptive_point_cloud_masked_autoencoder_for_eff.md)**

:   提出一种域自适应点云MAE框架（DAP-MAE），通过异构域适配器（HDA）和域特征生成器（DFG）两个模块，让一次跨域预训练即可在物体分类、人脸表情识别、部件分割、目标检测等多个不同域的下游任务上都达到SOTA。

**[David Data-Efficient And Accurate Vision Models From Synthetic Data](3d_vision/david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)**

:   证明通过高保真**程序化合成数据**即可训练出精度媲美基础模型（如 Sapiens-2B）的以人为中心的稠密预测模型，仅需 **30 万合成图像**、**0.3B 参数**、训练成本不到同级方案的 1/16，在深度估计、表面法线估计、软前景分割三项任务上实现 SOTA 或近 SOTA 性能。

**[Deepmesh Auto-Regressive Artist-Mesh Creation With Reinforcement Learning](3d_vision/deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)**

:   提出 DeepMesh 框架，通过改进的高效mesh tokenization算法（72%压缩率）和首次将DPO强化学习引入3D网格生成来实现人类偏好对齐，能够生成最高3万面的高质量Artist-like三角网格。

**[Degauss Dynamic-Static Decomposition With Gaussian Splatting For Distractor-Free](3d_vision/degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)**

:   提出 DeGauss，一种基于解耦的动态-静态高斯泼溅的自监督框架，通过前景动态高斯和背景静态高斯的概率掩码组合，实现从随意捕获的图像集到高度动态的自我中心视频的广泛场景下的无干扰 3D 重建。

**[Demeter A Parametric Model Of Crop Plant Morphology From The Real World](3d_vision/demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)**

:   Demeter 是一个数据驱动的参数化植物形态模型，将植物形态分解为拓扑、关节、形状和变形四个因素，支持形状生成、3D 重建和生物物理仿真。

**[Depth Anyevent A Cross-Modal Distillation Paradigm For Event-Based Monocular Dep](3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)**

:   提出跨模态蒸馏范式，利用图像域的视觉基础模型（Depth Anything v2）生成伪标签来训练事件相机深度估计网络，并设计了基于 VFM 的循环架构 DepthAnyEvent-R，在无需昂贵深度标注的情况下实现了事件相机单目深度估计的 SOTA 性能。

**[Describe Adapt And Combine Empowering Clip Encoders For Open-Set 3D Object Retri](3d_vision/describe_adapt_and_combine_empowering_clip_encoders_for_open-set_3d_object_retri.md)**

:   提出 DAC 框架，通过 "描述-适配-组合" 三步策略协同 CLIP 与多模态大语言模型 (MLLM)，仅使用多视图图像即可在开放集 3D 物体检索任务上大幅超越此前使用全模态（点云+体素+图像）的 SOTA 方法，平均 mAP 提升超过 +10%。

**[Diorama Unleashing Zero-Shot Single-View 3D Indoor Scene Modeling](3d_vision/diorama_unleashing_zero-shot_single-view_3d_indoor_scene_modeling.md)**

:   提出Diorama，首个零样本开放世界系统，从单张RGB图像通过模块化管线（开放世界感知+基于CAD的场景建模）生成完整的3D室内场景，包含建筑结构和物体摆放，无需端到端训练或人工标注。

**[Diorama Unleashing Zeroshot Singleview 3D Indoor Scene Model](3d_vision/diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)**

:   提出首个零样本开放世界系统 Diorama，通过模块化地组合 foundation model（GPT-4o、SAM、DinoV2、Metric3D 等），将单张 RGB 图像转化为包含建筑结构和 CAD 物体的完整可组合 3D 室内场景，无需任何端到端训练或人工标注。

**[Discretized Gaussian Representation For Tomographic Reconstruction](3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)**

:   提出离散化高斯表示（DGR）用于 CT 重建，通过离散化高斯函数直接端到端重建 3D 体素，并设计高度并行化的快速体积重建技术，在稀疏视角和有限角度 CT 场景中以零训练数据超越深度学习和实例重建方法。

**[Disentangling Instance And Scene Contexts For 3D Semantic Scene Completion](3d_vision/disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)**

:   提出 DISC，一种基于类别感知的双流架构用于 3D 语义场景补全，通过将实例类别和场景类别解耦到独立的查询流中并设计针对性的解码模块，在 SemanticKITTI 上仅用单帧输入即超越多帧 SOTA 方法，实例类别 mIoU 提升 17.9%。

**[Diving Into The Fusion Of Monocular Priors For Generalized Stereo Matching](3d_vision/diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)**

:   深入分析单目先验融合中的三大问题（仿射不变性 vs 绝对深度的对齐、迭代更新中的局部最优、噪声视差对融合的干扰），提出二元局部排序图和全局配准模块，在 SceneFlow→Middlebury/Booster 泛化实验中将 bad2 错误减半甚至更多，且几乎不增加计算开销。

**[Dmesh An Efficient Differentiable Mesh For Complex Shapes](3d_vision/dmesh_an_efficient_differentiable_mesh_for_complex_shapes.md)**

:   本文提出DMesh++，通过Minimum-Ball算法替代加权Delaunay三角剖分实现可微网格的tessellation函数，将计算复杂度从 $O(N)$ 降至 $O(\log N)$，在处理复杂形状时速度提升最高32倍，同时保持无自交叉和少薄三角形的优良特性。

**[Do It Yourself Learning Semantic Correspondence From Pseudo-Labels](3d_vision/do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)**

:   本文提出 DIY-SC，通过3D感知的伪标签生成策略（链式传播+松弛循环一致性+球面原型过滤）训练轻量适配器来改进基础模型特征的语义对应能力，在 SPair-71k 上实现了超越先前 SOTA 4.5%（PCK@0.1 per-keypoint）的性能，且无需人工关键点标注。

**[Driving View Synthesis On Free-Form Trajectories With Generative Prior](3d_vision/driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)**

:   提出驾驶视图合成框架DriveX，通过渐进式地将视频扩散模型的生成先验蒸馏到3DGS表示中——设计inpainting-based视频修复任务来生成新轨迹伪标注，迭代优化3D重建，实现自由轨迹上的高质量实时渲染。

**[Dso Aligning 3D Generators With Simulation Feedback For Physical Soundness](3d_vision/dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)**

:   提出 Direct Simulation Optimization (DSO) 框架，利用物理仿真器的（非可微）稳定性反馈作为奖励信号，通过 DPO 或新提出的 DRO 目标函数微调 3D 生成器，使其前馈式地直接输出物理上自支撑的 3D 物体，无需测试时优化。

**[Dynamic Point Maps A Versatile Representation For Dynamic 3D Reconstruction](3d_vision/dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)**

:   提出 Dynamic Point Maps (DPM)，将 DUSt3R 的视点不变点图扩展为同时控制视点和时间的时空不变表示，仅通过预测4组点图即可在前馈方式下同时解决深度估计、场景流、运动分割和3D目标跟踪等多种4D任务。

**[Easi3R Estimating Disentangled Motion From Dust3R Without Training](3d_vision/easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)**

:   提出 Easi3R，一种无需训练的即插即用方法，通过解耦 DUSt3R 注意力层中隐含编码的相机运动与物体运动信息，实现动态视频的4D重建、运动分割和相机位姿估计。

**[Easy3D A Simple Yet Effective Method For 3D Interactive Segmentation](3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)**

:   提出 Easy3D，一种简洁高效的 3D 交互式实例分割方法，结合体素稀疏编码器、轻量 Transformer 解码器和隐式点击融合策略，在域内和域外数据集上一致性地超越 SOTA，并首次将学习的负嵌入 (learned negative embedding) 成功应用于隐式点击融合。

**[Efficient Spiking Point Mamba For Point Cloud Analysis](3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)**

:   SPM（Spiking Point Mamba）提出首个基于 Mamba 的 3D 脉冲神经网络框架，通过层次化动态编码（HDE）和脉冲 Mamba 模块（SMB），在大幅降低能耗（3.5× 以上）的同时，在 ScanObjectNN 上比前 SOTA SNN 方法提升 6-7% 的准确率。

**[Egocentric Action-Aware Inertial Localization In Point Clouds With Vision-Langua](3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)**

:   EAIL 框架利用头戴式 IMU 信号中的第一人称动作线索，通过层次化多模态对齐（视觉-语言引导）学习动作与环境结构的关联，在 3D 点云中实现精确的惯性定位，同时附带动作识别能力。

**[Egom2P Egocentric Multimodal Multitask Pretraining](3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)**

:   EgoM2P 是首个面向自我中心(egocentric)4D理解的多模态多任务大模型，通过时序感知的掩码建模框架统一处理 RGB 视频、深度、注视和相机轨迹四种模态，在多个下游任务上匹配或超越专用模型且快一个数量级。

**[Embodiedsplat Personalized Real-To-Sim-To-Real Navigation With Gaussian Splats F](3d_vision/embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)**

:   提出 EmbodiedSplat，一个利用 iPhone 手机拍摄视频 → 3D 高斯溅射重建 mesh → 在 Habitat-Sim 中微调导航策略 → 部署到真实世界的完整流程，在真实场景 ImageNav 任务上比零样本基线提升 20%-40% 绝对成功率，sim-vs-real 相关系数达 0.87-0.97。

**[Etch Generalizing Body Fitting To Clothed Humans Via Equivariant Tightness](3d_vision/etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)**

:   提出ETCH框架，通过建模从衣物表面到体表的SE(3)等变紧密度向量(tightness vector)，将穿衣人体的body fitting简化为tightness-aware的稀疏marker拟合任务，在CAPE和4D-Dress数据集上相比SOTA方法（含tightness-agnostic和tightness-aware方法）在宽松衣物上提升16.7%~69.5%的关节误差，形状精度平均提升49.9%。

**[Evagaussians Event Stream Assisted Gaussian Splatting From Blurry Images](3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)**

:   提出EvaGaussians框架，利用事件相机的高时间分辨率事件流辅助3D高斯泼溅从运动模糊图像中学习，通过事件辅助初始化、模糊/事件联合重建损失和事件辅助几何正则化，实现高保真新视图合成并保持实时渲染效率。

**[Event-Based Tiny Object Detection A Benchmark Dataset And Baseline](3d_vision/event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)**

:   提出首个大规模事件相机反无人机小目标检测基准EV-UAV数据集（147序列/230万事件级标注/平均目标仅6.8×5.4像素），并设计EV-SpSegNet——基于稀疏3D点云分割的检测框架，利用小目标在时空事件点云中形成连续曲线的特征，配合时空相关性损失(STC loss)引导网络保留目标事件，在IoU/ACC/检测概率上全面超越13种SOTA方法，推理速度快10-100倍。

**[Event-Boosted Deformable 3D Gaussians For Dynamic Scene Reconstruction](3d_vision/event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)**

:   首次将事件相机与可变形 3D 高斯溅射（3D-GS）结合用于动态场景重建，提出 GS-阈值联合建模策略和动静分解策略，在新构建的事件-4D 基准上实现了 SOTA 的渲染质量和速度（合成数据平均 PSNR 提升 2.73dB，渲染速度达 4D-GS 的 1.71 倍）。

**[Event-Driven Storytelling With Multiple Lifelike Humans In A 3D Scene](3d_vision/event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)**

:   提出基于事件驱动的LLM框架，将3D场景中多角色行为规划分解为叙述者逐事件生成和事件解析器的精细空间推理两个模块，首次实现了大规模多房间3D场景中4-5+角色的长时序自然交互运动生成。

**[Excap3D Expressive 3D Scene Understanding Via Object Captioning With Varying Det](3d_vision/excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)**

:   提出 ExCap3D，一个在 3D 室内场景中对物体生成多粒度描述的方法，包含物体级和部件级两个描述层次，通过部件→物体的信息共享和语义/文本一致性损失确保描述的准确性与一致性，在新建的 190K 描述数据集上 CIDEr 评分比 SOTA 分别提升 17% 和 124%。

**[Facelift Learning Generalizable Single Image 3D Face Reconstruction From Synthet](3d_vision/facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)**

:   提出 FaceLift，一种仅在合成数据上训练但能良好泛化到真实图像的单图360度高质量3D人头重建方法，通过多视图潜扩散模型生成身份一致的多视角图像，再用基于 Transformer 的重建器生成像素对齐的3D高斯表示。

**[Faster And Better 3D Splatting Via Group Training](3d_vision/faster_and_better_3d_splatting_via_group_training.md)**

:   提出 **Group Training** 策略，通过将高斯基元周期性分组为"训练组"和"缓存组"来加速 3DGS 训练，结合**基于透明度的优先采样**（OPS），在4个标准数据集上实现约 **30% 训练加速**的同时**提升渲染质量**和**减少模型体积**，且可即插即用于 3DGS 和 Mip-Splatting 等框架。

**[Fiffdepth Feed-Forward Transformation Of Diffusion-Based Generators For Detailed](3d_vision/fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)**

:   提出FiffDepth，将预训练的扩散模型转化为确定性前馈架构进行单目深度估计，通过保持扩散轨迹维持細节生成能力，并引入可学习滤波器蒸馏DINOv2的鲁棒泛化能力到扩散骨干网络，在效率、精度和细节丰富度三方面同时超越现有方法。

**[Find Any Part In 3D](3d_vision/find_any_part_in_3d.md)**

:   提出Find3D，构建了一个由2D基础模型（SAM + Gemini）驱动的自动化3D数据标注引擎，生成210万个部件标注，训练出首个同时具备开放世界、跨类别、部件级和前馈推理能力的3D分割模型，零样本mIoU提升260%，推理速度比现有方法快6-300倍。

**[Fish2Mesh Transformer 3D Human Mesh Recovery From Egocentric Vision](3d_vision/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)**

:   提出 Fish2Mesh，一种鱼眼感知的 Transformer 模型，通过新颖的自我中心位置编码（EPE）将等距柱状投影的3D球面信息嵌入 Swin Transformer，实现从头戴式鱼眼相机图像精确恢复3D人体网格。

**[Flashdepth Real-Time Streaming Video Depth Estimation At 2K Resolution](3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)**

:   提出FlashDepth，在Depth Anything v2基础上添加Mamba循环模块实现帧间尺度一致性，并设计Small-Large混合架构在2K分辨率下达到24 FPS的实时流式视频深度估计，边界清晰度远超现有方法。

**[Flexgen Flexible Multi-View Generation From Text And Image Inputs](3d_vision/flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)**

:   本文提出 FlexGen，一个灵活的多视角图像生成框架，通过 GPT-4V 生成 3D-aware 文本标注并设计自适应双控制模块，支持单图、文本或二者联合控制生成一致的多视角图像，实现未可见区域补全、材质编辑和纹理控制等多种可控能力。

**[From Gallery To Wrist Realistic 3D Bracelet Insertion In Videos](3d_vision/from_gallery_to_wrist_realistic_3d_bracelet_insertion_in_videos.md)**

:   提出一种混合管线将 3D 手镯逼真插入视频：利用 3D 高斯泼溅（3DGS）保证时序一致性，用 2D 扩散模型增强光照真实感，并通过光照驱动（Shading-Driven）管线分离 albedo/shading/反射残差分别优化，在用户研究中以 81.7% 的真实感偏好率大幅超越现有方法。

**[From Image To Video An Empirical Study Of Diffusion Representations](3d_vision/from_image_to_video_an_empirical_study_of_diffusion_representations.md)**

:   系统对比了相同架构(WALT)在图像 vs 视频生成目标下训练的扩散模型在下游视觉理解任务上的表现，发现视频扩散模型在所有任务上一致优于图像对应物，尤其在需要运动和3D空间理解的任务上优势显著（点跟踪+68%、相机位姿+60%）。

**[From One To More Contextual Part Latents For 3D Generation](3d_vision/from_one_to_more_contextual_part_latents_for_3d_generation.md)**

:   提出CoPart框架，通过上下文部件潜码表示3D物体并利用互引导策略微调预训练扩散模型，实现高质量的部件级3D生成，同时支持部件编辑、铰接体生成和小场景生成。

**[Fross Faster-Than-Real-Time Online 3D Semantic Scene Graph Generation From Rgb-D](3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)**

:   提出FROSS方法，通过将2D场景图直接提升到3D空间并用高斯分布表示物体，实现了超实时（144 FPS）的在线3D语义场景图生成，无需精确点云重建。

**[G2Sf Geometry-Guided Score Fusion For Multimodal Industrial Anomaly Detection](3d_vision/g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)**

:   提出 G2SF 框架，将基于 memory bank 的异常分数重新解释为局部特征空间中的各向同性欧氏距离，进而通过 Local Scale Prediction Network (LSPN) 学习方向感知的缩放因子，将其渐进演化为各向异性的统一融合度量，实现多模态工业异常检测 SOTA。

**[Gap Gaussianize Any Point Clouds With Text Guidance](3d_vision/gap_gaussianize_any_point_clouds_with_text_guidance.md)**

:   提出GAP框架,利用深度感知图像扩散模型将无色点云转化为高保真3D Gaussian表示,通过表面锚定机制确保几何精度,并设计基于扩散的inpainting策略补全难以观测区域。

**[Gas Generative Avatar Synthesis From A Single Image](3d_vision/gas_generative_avatar_synthesis_from_a_single_image.md)**

:   提出GAS框架，通过将泛化NeRF重建的密集外观线索与视频扩散模型结合，统一新视角和新姿态合成为视频生成任务，配合模态切换器解耦两种任务，实现从单张图像生成视角一致和时序连贯的人体Avatar。

**[Gaussian Splatting With Discretized Sdf For Relightable Assets](3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)**

:   本文提出将连续SDF离散化为高斯基元的额外属性，通过SDF-to-opacity变换统一高斯和SDF表示，配合投影一致性损失和球面初始化，在仅用4G显存的前提下实现了超越现有高斯逆渲染方法的重光照质量。

**[Gaussian Variation Field Diffusion For High-Fidelity Video-To-4D Synthesis](3d_vision/gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)**

:   提出一种视频到4D生成框架，通过Direct 4DMesh-to-GS Variation Field VAE将动画数据直接编码为紧凑的高斯变化场潜在空间，再训练时序感知的扩散模型生成动态3D内容，在4.5秒内实现高保真4D合成，并展示了对真实视频输入的优越泛化能力。

**[Gaussianproperty Integrating Physical Properties To 3D Gaussians With Lmms](3d_vision/gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)**

:   GaussianProperty 提出了一个免训练框架，利用 SAM 分割和 GPT-4V 识别能力，通过全局-局部推理模块和多视角投票策略，将物理属性（密度、弹性模量、摩擦系数等）赋予 3D Gaussians，支持物理仿真和机器人抓取两大下游任务。

**[Gaussianupdate Continual 3D Gaussian Splatting Update For Changing Environments](3d_vision/gaussianupdate_continual_3d_gaussian_splatting_update_for_changing_environments.md)**

:   提出GaussianUpdate，首次将3D高斯表示与持续学习结合，通过三阶段更新策略（外观更新→几何布局更新→联合精炼）和可见性感知生成式回放，实现时变场景的实时渲染和变化可视化。

**[Gazegaussian High-Fidelity Gaze Redirection With 3D Gaussian Splatting](3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)**

:   提出GazeGaussian，首个基于3D高斯溅射（3DGS）的高保真视线重定向方法，通过双流3DGS模型分别建模面部和眼部区域，设计显式的高斯眼球旋转表示和表情引导神经渲染器，在视线精度、合成质量和渲染速度上全面超越现有方法。

**[Generating Physically Stable And Buildable Brick Structures From Text](3d_vision/generating_physically_stable_and_buildable_brick_structures_from_text.md)**

:   BrickGPT 首次实现从文本提示生成物理稳定且可组装的互锁砖块结构，核心思想是将积木组装问题建模为自回归文本生成任务，并在推理时集成物理感知的有效性检查和回滚机制，确保生成结构的稳定性和可构建性。

**[Geo4D Leveraging Video Generators For Geometric 4D Scene Reconstruction](3d_vision/geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)**

:   将预训练视频扩散模型(DynamiCrafter)改造为单目4D动态场景重建器——同时预测点云图、视差图和射线图三种互补几何模态，通过多模态对齐融合算法和滑动窗口推理，仅用合成数据训练即可零样本泛化至真实视频，大幅超越当前视频深度估计SOTA。

**[Geometry Distributions](3d_vision/geometry_distributions.md)**

:   提出Geometry Distributions (GeomDist)，将3D几何建模为表面点的概率分布并用扩散模型学习，无需假设亏格、连通性或边界条件，可从高斯噪声采样无限多表面点来表示任意拓扑的几何。

**[Geoprog3D Compositional Visual Reasoning For City-Scale 3D Language Fields](3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)**

:   提出 GeoProg3D，首个支持城市级高保真3D场景自然语言交互的视觉编程框架，通过地理感知的城市级3D语言场（GCLF）和地理视觉API（GV-APIs），结合LLM推理引擎实现组合式地理空间推理，在新提出的952条查询的GeoEval3D基准上全面超越现有3D语言场和VLM方法。

**[Geosplatting Towards Geometry Guided Gaussian Splatting For Physically-Based Inv](3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)**

:   提出 GeoSplatting，通过从可优化的显式网格可微分地生成表面对齐的高斯点，为3DGS提供精确的几何引导，实现SOTA逆渲染性能（材质-光照解耦），同时训练仅需10-15分钟。

**[Global-Aware Monocular Semantic Scene Completion With State Space Models](3d_vision/global-aware_monocular_semantic_scene_completion_with_state_space_models.md)**

:   提出GA-MonoSSC，一种结合Transformer（2D全局上下文）和Mamba（3D长程依赖）的混合架构用于室内单目语义场景补全，创新引入Frustum Mamba Layer解决体素序列化中的特征不连续性问题，在Occ-ScanNet和NYUv2上达到SOTA。

**[Global Motion Corresponder For 3D Point-Based Scene Interpolation Under Large Mo](3d_vision/global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)**

:   提出Global Motion Corresponder (GMC),通过学习将两个时刻的3D Gaussian映射到共享规范空间的一元势场,实现大运动条件下的鲁棒场景插值和外推。

**[Gsot3D Towards Generic 3D Single Object Tracking In The Wild](3d_vision/gsot3d_towards_generic_3d_single_object_tracking_in_the_wild.md)**

:   提出 GSOT3D，目前最大的通用3D单目标跟踪基准，包含620个多模态序列（点云+RGB+深度）覆盖54类物体，支持PC/RGB-PC/RGB-D三种3D跟踪任务，并提出渐进式时空跟踪器PROT3D以9DoF包围盒实现最优性能。

**[Guava Generalizable Upper Body 3D Gaussian Avatar](3d_vision/guava_generalizable_upper_body_3d_gaussian_avatar.md)**

:   提出 GUAVA，首个从单张图像通过前馈推理快速重建可动画上半身3D高斯虚拟人的框架，结合模板高斯和 UV 高斯表示，支持丰富面部表情和手势驱动，约0.1s完成重建并实时渲染。

**[Guiding Diffusion-Based Articulated Object Generation By Partial Point Cloud Ali](3d_vision/guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)**

:   提出 PhysNAP，通过点云对齐损失和基于SDF的物理合理性约束（部件穿透+关节移动）引导预训练扩散模型 NAP 的逆扩散过程，实现类别感知的铰接物体生成，在对齐精度和物理合理性上显著优于无引导基线。

**[Hierarchical Material Recognition From Local Appearance](3d_vision/hierarchical_material_recognition_from_local_appearance.md)**

:   提出面向视觉应用的层级式材质分类学体系(taxonomy)与野外数据集 Matador（含深度图的 ~7200 张材质图像，57类），并基于图注意力网络(GAT)利用分类学的层级亲缘关系进行材质识别，在多个基准数据集上达到 SOTA，同时支持新材质的小样本学习和场景中任意点的材质探测。

**[His-Gpt Towards 3D Human-In-Scene Multimodal Understanding](3d_vision/his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)**

:   提出 HIS-GPT，首个面向3D人-场景联合理解的多模态大语言模型，通过辅助交互模块(AInt)和布局-轨迹位置编码(LTP)捕获人场交互线索，并构建首个系统性基准 HIS-Bench，在HIS-QA任务上大幅超越GPT-4o等基线。

**[Hort Monocular Hand-Held Objects Reconstruction With Transformers](3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)**

:   提出 HORT，基于 Transformer 的粗到细框架，从单目图像高效重建手持物体的稠密3D点云，通过整合图像特征和3D手部几何信息联合预测物体点云及其相对手部的位姿，在准确率和推理速度上均达到 SOTA。

**[Housetour A Virtual Real Estate Aigent](3d_vision/housetour_a_virtual_real_estate_aigent.md)**

:   提出 HouseTour，给定一组已知位姿的室内图像，联合生成类人的3D相机轨迹和房地产文字描述，通过 Residual Diffuser 进行基于扩散的轨迹规划并将空间特征集成到 Qwen2-VL-3D 中生成3D-grounded文本摘要。

**[How Far Are Ai-Generated Videos From Simulating The 3D Visual World A Learned 3D](3d_vision/how_far_are_ai-generated_videos_from_simulating_the_3d_visual_world_a_learned_3d.md)**

:   提出 Learned 3D Evaluation (L3DE)，一种基于单目3D线索（运动、深度、外观）和对比学习的客观可量化评估方法，用于衡量AI生成视频在3D视觉一致性方面与真实视频的差距，无需人工标注缺陷或质量标签。

**[Humanolat A Large-Scale Dataset For Full-Body Human Relighting And Novel-View Sy](3d_vision/humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)**

:   提出HumanOLAT——首个公开可用的大规模全身人体多视角OLAT(One-Light-at-a-Time)数据集，包含21个被试×3个姿态×40视角×344种光照≈850K帧，为人体重打光和新视角合成提供了高质量基准。

**[Identity Preserving 3D Head Stylization With Multiview Score Distillation](3d_vision/identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)**

:   提出基于负对数似然蒸馏(LD)的3D头部风格化框架，通过多视角网格评分、镜像梯度和秩加权评分张量，实现在360度一致渲染下的高质量风格化与身份保持。

**[Im360 Large-Scale Indoor Mapping With 360 Cameras](3d_vision/im360_large-scale_indoor_mapping_with_360_cameras.md)**

:   本文提出 IM360，一个面向稀疏扫描大规模室内环境的三维建图流水线，通过将球面相机模型深度集成到 SfM 各环节、结合稠密特征匹配和可微渲染纹理优化，在 Matterport3D 和 Stanford2D3D 上实现了远超现有方法的相机定位准确率和渲染质量（PSNR 提升 3.5）。

**[Image-Guided Shape-From-Template Using Mesh Inextensibility Constraints](3d_vision/image-guided_shape-from-template_using_mesh_inextensibility_constraints.md)**

:   提出一种纯图像引导的无监督 Shape-from-Template (SfT) 方法，仅利用颜色、梯度和轮廓等视觉线索配合网格不可伸展性约束来重建变形物体 3D 形状，比最优无监督方法快 400 倍且精度大幅领先。

**[Image As An Imu Estimating Camera Motion From A Single Motion-Blurred Image](3d_vision/image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)**

:   本文将运动模糊从"不需要的伪影"转变为"有价值的运动线索"，通过从单张模糊图像预测稠密光流场和单目深度图，再用可微分最小二乘求解器恢复相机6DoF瞬时速度，实现媲美甚至超越IMU的运动估计精度和30FPS实时性能。

**[Instascene Towards Complete 3D Instance Decomposition And Reconstruction From Cl](3d_vision/instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)**

:   InstaScene 提出统一的杂乱场景实例分解与完整重建框架，通过追踪高斯光栅化构建空间对比学习实现精准实例分割，并设计 in-situ 生成管线利用已知观测和几何线索引导 3D 生成模型重建完整物体。

**[Jointdit Enhancing Rgb-Depth Joint Modeling With Diffusion Transformers](3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)**

:   JointDiT 基于 Flux 扩散 Transformer 构建 RGB-Depth 联合分布模型，通过自适应调度权重和非平衡时间步采样策略，使单一模型通过控制各模态的时间步即可灵活执行联合生成、深度估计和深度条件图像生成三种任务。

**[Kh Symmetry Understanding Of 3D Shapes Via Chirality Disentanglement](3d_vision/kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)**

:   提出无监督手性特征提取管线,从2D基础模型特征中蒸馏左右手性信息用于装饰3D形状顶点描述子,有效解决形状分析中的左右歧义问题。

**[Laconic A 3D Layout Adapter For Controllable Image Creation](3d_vision/laconic_a_3d_layout_adapter_for_controllable_image_creation.md)**

:   提出 LACONIC，一种基于参数化 3D 语义包围盒的轻量级适配器，通过解耦交叉注意力机制将显式 3D 几何信息注入预训练 text-to-image 扩散模型，首次实现了相机控制、3D 物体级语义引导以及对屏幕外物体的全面场景上下文建模，在 FID 上比 SceneCraft 降低 75.8%。

**[Layerlock Non-Collapsing Representation Learning With Progressive Freezing](3d_vision/layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)**

:   提出 LayerLock，一种通过渐进式冻结网络层并动态切换预测目标（从像素到越来越深的中间层特征）的自监督视频表征学习方法，兼具像素预测的稳定性和潜变量预测的高效语义捕获能力，用于训练高达 4B 参数的视频模型。

**[Learning 3D Object Spatial Relationships From Pre-Trained 2D Diffusion Models](3d_vision/learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)**

:   提出从预训练 2D 扩散模型合成图像中学习物体间 3D 空间关系（OOR），通过 3D 提升管线构建配对数据集，训练文本条件化的 score-based 扩散模型对物体对的相对位姿和尺度分布建模，并扩展至多物体场景布局和场景编辑。

**[Learning 3D Scene Analogies With Neural Contextual Scene Maps](3d_vision/learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)**

:   提出3D场景类比任务，通过神经上下文场景映射（neural contextual scene maps）在共享相似语义上下文的场景区域间建立稠密三维映射，支持轨迹迁移与物体放置迁移等下游应用。

**[Learning 4D Embodied World Models](3d_vision/learning_4d_embodied_world_models.md)**

:   提出 TesserAct——一种 4D 具身世界模型，通过训练视频生成模型联合预测 RGB、深度和法线视频，再转换为高质量 4D 场景，实现空间-时间一致的 3D 世界动态模拟和机器人动作规划。

**[Learning Robust Stereo Matching In The Wild With Selective Mixture-Of-Experts](3d_vision/learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)**

:   提出 SMoEStereo，通过在冻结的视觉基础模型(VFM)中集成变秩MoE-LoRA和变核MoE-Adapter，结合轻量决策网络选择性激活MoE模块，实现了场景自适应的鲁棒立体匹配，在跨域和联合泛化上达到SOTA。

**[Lightweight Gradient-Aware Upscaling Of 3D Gaussian Splatting Images](3d_vision/lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)**

:   提出专门为3DGS设计的轻量图像上采样技术，利用高斯原语的解析图像梯度进行梯度感知双三次样条插值，无需深度学习推理即可实现3-4倍渲染加速，且重建质量优于标准双三次插值和DL-based上采样。

**[Linr-Pcgc Lossless Implicit Neural Representations For Point Cloud Geometry Comp](3d_vision/linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)**

:   LINR-PCGC 提出了首个基于隐式神经表征（INR）的点云几何无损压缩方法，通过设计轻量级多尺度 SparseConv 网络（含尺度上下文提取 SCE 和子节点预测 CNP 模块），结合 GoP 级帧共享解码器和初始化策略，在不依赖特定训练数据分布的前提下，在 MVUB 数据集上比 G-PCC TMC13v23 降低 21.21% 码率，比 SparsePCGC 降低 21.95%。

**[Llava-3D A Simple Yet Effective Pathway To Empowering Lmms With 3D Capabilities](3d_vision/llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)**

:   本文提出 LLaVA-3D，通过将 3D 位置嵌入注入 2D CLIP patch 特征构建"3D Patch"，以最小改动将 2D LMM（LLaVA-Video）扩展为统一的 2D/3D 理解模型，训练收敛速度比现有 3D LMM 快 3.5 倍，在多个 3D 基准上达到 SOTA 且保持 2D 能力不下降。

**[Localdygs Multi-View Global Dynamic Scene Modeling Via Adaptive Local Implicit F](3d_vision/localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)**

:   提出 LocalDyGS——将全局复杂动态场景分解为种子点定义的局部空间、并通过静态-动态特征解耦生成时序高斯来建模各局部运动的框架，首次实现了大尺度复杂动态场景的高质量重建。

**[Long3R Long Sequence Streaming 3D Reconstruction](3d_vision/long3r_long_sequence_streaming_3d_reconstruction.md)**

:   提出 LONG3R，一种基于循环记忆机制的流式多视图3D重建模型，通过记忆门控、双源精炼解码器和3D时空记忆三大创新，在保持实时推理速度的同时显著提升长序列重建质量。

**[Longsplat Robust Unposed 3D Gaussian Splatting For Casual Long Videos](3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)**

:   LongSplat 针对无相机位姿的随拍长视频场景，提出增量联合优化框架同时优化相机位姿和 3DGS，设计基于 MASt3R 先验的鲁棒位姿估计模块和自适应八叉树锚点形成机制，解决位姿漂移、几何初始化不准和内存限制问题。

**[Maskhand Generative Masked Modeling For Robust Hand Mesh Reconstruction In The W](3d_vision/maskhand_generative_masked_modeling_for_robust_hand_mesh_reconstruction_in_the_w.md)**

:   提出 MaskHand，首个将生成式掩码建模引入 3D 手部网格重建的方法，通过 VQ-MANO 将连续手部姿态离散化为 token，再利用上下文引导的掩码 Transformer 学习 2D-to-3D 映射的概率分布，在推理时通过置信度引导的迭代采样生成高精度手部网格，在 HO3Dv3 零样本评估中 PA-MPJPE 降低 19.5%。

**[Materialmvp Illumination-Invariant Material Generation Via Multi-View Pbr Diffus](3d_vision/materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)**

:   MaterialMVP是一个端到端的多视图PBR纹理生成模型，通过一致性正则化训练解耦光照、双通道材质生成框架（MCAA + Learnable Material Embeddings）对齐albedo和metallic-roughness贴图，从3D网格和图像prompt一步生成高质量、光照不变、多视图一致的PBR材质。

**[Mega Memory-Efficient 4D Gaussian Splatting For Dynamic Scenes](3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)**

:   提出 MEGA，一个面向4D Gaussian Splatting的内存高效框架，通过DC-AC颜色分解消除冗余球谐系数（8×压缩），结合熵约束Gaussian形变技术扩大每个Gaussian的作用范围并减少数量，最终在Technicolor和Neural 3D Video数据集上分别实现约190×和125×存储压缩，同时保持可比的渲染质量和实时速度。

**[Meshanything V2 Artist-Created Mesh Generation With Adjacent Mesh Tokenization](3d_vision/meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)**

:   MeshAnything V2 提出 Adjacent Mesh Tokenization (AMT)，通过用单个顶点（而非传统三个顶点）表示相邻面，将网格的 token 序列长度平均缩短一半，从而在不增加计算成本的前提下将最大生成面数从 800 提升到 1600，显著提高了自回归网格生成的效率和质量。

**[Meshmamba State Space Models For Articulated 3D Mesh Generation And Reconstructi](3d_vision/meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)**

:   MeshMamba 提出基于 Mamba 状态空间模型的 3D 关节体网格生成与重建方法，通过设计基于身体部位 UV 图和模板网格坐标的顶点序列化技术，实现了万级顶点网格的高效生成和重建，速度比 Transformer 快 6-9 倍。

**[Meshpad Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation And Ed](3d_vision/meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)**

:   MeshPad 将草图驱动的 3D 网格创建与编辑解耦为"添加"和"删除"两个子任务，基于三角序列表示和 Transformer 自回归生成，并提出顶点对齐推测解码器实现 2.2× 加速，让交互式网格编辑在几秒内完成。

**[Mincd-Pnp Learning 2D-3D Correspondences With Approximate Blind Pnp](3d_vision/mincd-pnp_learning_2d-3d_correspondences_with_approximate_blind_pnp.md)**

:   本文提出 MinCD-PnP，通过三重近似将计算昂贵的 Blind PnP 简化为最小化 2D-3D 关键点间 Chamfer 距离的问题，设计轻量级多任务学习模块 MinCD-Net 集成到现有 I2P 配准框架中，在跨场景和跨数据集设置下显著提升内点率和配准召回率。

**[Moga 3D Generative Avatar Prior For Monocular Gaussian Avatar Reconstruction](3d_vision/moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)**

:   提出MoGA，通过学习生成式3D头像先验并将其作为初始化、正则化和姿态优化的强约束，从单张图像重建高保真3D高斯头像，显著超越现有方法。

**[Momentum-Gs Momentum Gaussian Self-Distillation For High-Quality Large Scene Rec](3d_vision/momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)**

:   Momentum-GS 提出基于动量的自蒸馏机制来解决大规模场景3D高斯溅射中分块并行训练的一致性问题，通过动量教师高斯解码器提供全局引导并解耦分块数量与GPU数量的限制，在多个大规模场景数据集上取得SOTA，LPIPS较CityGaussian提升18.7%。

**[Monocular Semantic Scene Completion Via Masked Recurrent Networks](3d_vision/monocular_semantic_scene_completion_via_masked_recurrent_networks.md)**

:   提出 MonoMRN，一个两阶段单目语义场景补全框架：先做粗粒度预测，再用 Masked Sparse GRU（MS-GRU）循环精炼被遮挡区域，并引入距离注意力投影减少深度投影误差，在 NYUv2 和 SemanticKITTI 上均达到 SOTA。

**[Monomobility Zero-Shot 3D Mobility Analysis From Monocular Videos](3d_vision/monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)**

:   MonoMobility提出首个从单目视频零样本分析关节物体运动部件及运动属性（运动轴和运动类型）的框架，通过组合深度估计、光流分割等现成工具进行初始分析，再用2D高斯泼溅和专门设计的关节物体动态场景优化算法自监督精细化结果，无需任何标注数据即可处理旋转、平移和复合运动。

**[Mugs Multi-Baseline Generalizable Gaussian Splatting Reconstruction](3d_vision/mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)**

:   MuGS 是首个支持多基线设置（小基线到大基线）的泛化3D高斯泼溅方法，通过融合 MVS 和 MDE 特征、投影-采样深度一致性网络和参考视图损失，在不同基线数据集上均达到 SOTA。

**[Multi-View 3D Point Tracking](3d_vision/multi-view_3d_point_tracking.md)**

:   提出 MVTracker——首个数据驱动的多视角3D点跟踪器，通过将多视图深度图反投影为统一的3D特征点云，利用 kNN 关联和 Transformer 迭代优化，在仅需4个相机的实用配置下实现鲁棒的长程3D点轨迹估计，在 Panoptic Studio 和 DexYCB 上分别达到 3.1 cm 和 2.0 cm 的中位轨迹误差。

**[Mv-Adapter Multi-View Consistent Image Generation Made Easy](3d_vision/mv-adapter_multi-view_consistent_image_generation_made_easy.md)**

:   提出首个基于Adapter的多视角图像生成方案MV-Adapter，通过复制self-attention层+并行注意力架构实现即插即用的多视角生成，在SDXL上达到768分辨率，兼容各种T2I衍生模型。

**[Mvgbench A Comprehensive Benchmark For Multi-View Generation Models](3d_vision/mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)**

:   提出 MVGBench——多视图生成模型的综合评估框架，创新性地引入基于 3DGS 自一致性的 3D 一致性指标（无需 3D GT），系统评估了 12 个 SOTA 方法在最佳性能、泛化和鲁棒性三方面的表现，并基于分析提出的最佳实践构建了新方法 ViFiGen。

**[Nautilus Locality-Aware Autoencoder For Scalable Mesh Generation](3d_vision/nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)**

:   Nautilus 提出一种局部性感知的自编码器进行可扩展的 artist-like 网格生成，通过 Nautilus 式壳结构网格分词算法将序列长度压缩到 1/4，并结合双流点云条件器提高局部结构保真度，首次实现最多 5000 面的高质量网格直接生成。

**[Neural Compression For 3D Geometry Sets](3d_vision/neural_compression_for_3d_geometry_sets.md)**

:   提出NeCGS,首个能将包含数千个多样3D网格模型的几何集合压缩高达900倍的神经压缩范式,通过TSDF-Def隐式表示和量化感知自解码器实现高精度保持。

**[Neuraleaf Neural Parametric Leaf Models With Shape And Deformation Disentangleme](3d_vision/neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)**

:   NeuraLeaf 将叶片的 3D 几何解耦为 2D 基础形状和 3D 变形两个潜在空间，利用大量 2D 叶片图像数据集学习形状空间，提出无骨架蒙皮模型处理叶片的高度柔性变形，并构建了首个专注叶片变形建模的 3D 数据集 DeformLeaf。

**[No Pose At All Self-Supervised Pose-Free 3D Gaussian Splatting From Sparse Views](3d_vision/no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)**

:   提出SPFSplat,首个在训练和推理时均不需要真值位姿的自监督3DGS框架,通过共享ViT骨干同时预测Gaussian基元和相机位姿,在极端视角变化下超越需要位姿的SOTA方法。

**[Noise2Score3D Tweedies Approach For Unsupervised Point Cloud Denoising](3d_vision/noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)**

:   提出Noise2Score3D,基于Tweedie公式的全无监督点云去噪框架,从噪声数据直接学习得分函数,实现单步去噪;引入点云全变分度量估计未知噪声参数。

**[Not All Frame Features Are Equal Video-To-4D Generation Via Decoupling Dynamic-S](3d_vision/not_all_frame_features_are_equal_video-to-4d_generation_via_decoupling_dynamic-s.md)**

:   DS4D 首次提出在video-to-4D生成中沿时间轴和空间轴解耦动静态特征，通过动静态特征解耦模块（DSFD）获取动态表征，并通过时空相似性融合模块（TSSF）跨视角自适应聚合动态信息，在Consistent4D和Objaverse数据集上达到SOTA。

**[Occlugaussian Occlusion-Aware Gaussian Splatting For Large Scene Reconstruction ](3d_vision/occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)**

:   提出遮挡感知的场景划分策略和基于区域的渲染技术,通过相机共可见性图聚类实现与场景布局对齐的分区,显著提升大场景3DGS重建质量和渲染速度。

**[One Look Is Enough Seamless Patchwise Refinement For Zero-Shot Monocular Depth E](3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)**

:   提出 PRO（Patch Refine Once），通过分组块一致性训练（GPCT）和无偏遮罩（BFM）策略，在高分辨率图像上实现无缝的逐块深度精炼，仅需每块单次精炼即可消除边界伪影，推理速度比 PatchRefiner 快12倍。

**[Online Language Splatting](3d_vision/online_language_splatting.md)**

:   首个在 3DGS-SLAM 系统中实现**在线、近实时、开放词汇**语言建图的框架，通过高分辨率 CLIP 嵌入、两阶段在线自编码器压缩和颜色-语言解耦优化三项创新，在精度超越离线 SOTA 的同时实现 40×–200× 的效率提升。

**[Open-Vocabulary Octree-Graph For 3D Scene Understanding](3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)**

:   提出 Octree-Graph，一种将自适应八叉树与图结构结合的新颖场景表示，通过时序分组式段合并(CGSM)和实例特征聚合(IFA)获取准确的语义对象，实现高效的开放词汇3D场景理解。

**[Outdoor Monocular Slam With Global Scale-Consistent 3D Gaussian Pointmaps](3d_vision/outdoor_monocular_slam_with_global_scale-consistent_3d_gaussian_pointmaps.md)**

:   提出 S3PO-GS，通过将 3DGS 渲染的 pointmap 作为锚点建立尺度自一致的跟踪模块，结合基于 patch 的 pointmap 动态建图机制，在 RGB-only 室外场景中实现了无累积尺度漂移的高精度定位与高保真新视角合成。

**[Panst3R Multi-View Consistent Panoptic Segmentation](3d_vision/panst3r_multi-view_consistent_panoptic_segmentation.md)**

:   基于MUSt3R构建PanSt3R，在**单次前向传播**中同时完成3D重建和多视角全景分割，无需相机参数、无需测试时优化，比现有方法快数个量级。

**[Pcr-Gs Colmap-Free 3D Gaussian Splatting Via Pose Co-Regularizations](3d_vision/pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)**

:   提出 PCR-GS，通过 DINO 特征重投影正则化和基于小波变换的频率正则化对相机位姿进行协同约束，在无需 COLMAP 先验的条件下实现了复杂相机轨迹场景的高质量 3D-GS 重建与位姿估计。

**[Perspose 3D Human Pose Estimation With Perspective Encoding And Perspective Rota](3d_vision/perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)**

:   提出PersPose框架，通过透视编码(PE)将裁剪后相机内参编码为2D映射、透视旋转(PR)将人体居中以消除透视畸变，解决了现有方法忽略FOV信息导致深度估计不准确的问题。

**[Phd Personalized 3D Human Body Fitting With Point Diffusion](3d_vision/phd_personalized_3d_human_body_fitting_with_point_diffusion.md)**

:   提出个性化3D人体姿态估计范式PHD——先通过SHAPify校准用户体型，再用体型条件化的点扩散模型PointDiT作为3D先验，结合Point Distillation Sampling损失迭代优化姿态，在绝对姿态精度上达到EMDB数据集SOTA。

**[Physsplat Efficient Physics Simulation For 3D Scenes Via Mllm-Guided Gaussian Sp](3d_vision/physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)**

:   提出PhysSplat，首次利用多模态大语言模型(MLLM)零样本估计3D场景中物体的物理属性，结合物理-几何自适应采样策略在单GPU上2分钟内实现逼真的物理仿真。

**[Placeit3D Language-Guided Object Placement In Real 3D Scenes](3d_vision/placeit3d_language-guided_object_placement_in_real_3d_scenes.md)**

:   提出语言引导的真实3D场景中物体放置任务（PlaceIt3D），包含基准测试、大规模数据集和基于3D LLM的基线方法PlaceWizard，实现对场景、物体和语言指令的联合推理。

**[Polaranything Diffusion-Based Polarimetric Image Synthesis](3d_vision/polaranything_diffusion-based_polarimetric_image_synthesis.md)**

:   提出 PolarAnything，首个基于单张 RGB 图像生成偏振图像的扩散模型框架，通过对编码后的 AoLP 和 DoLP 进行去噪扩散，实现了物理准确且逼真的偏振属性合成，无需 3D 资产或偏振相机。

**[Proactive Scene Decomposition And Reconstruction](3d_vision/proactive_scene_decomposition_and_reconstruction.md)**

:   提出基于主动人-物交互的在线场景分解与重建任务,通过观察自我中心视角下的交互行为来定义分解粒度,实现渐进式对象解耦和高质量全局重建。

**[Pseudomaptrainer Learning Online Mapping Without Hd Maps](3d_vision/pseudomaptrainer_learning_online_mapping_without_hd_maps.md)**

:   提出 PseudoMapTrainer，首次实现**完全不依赖 GT HD Map** 训练在线建图模型：利用 2D Gaussian Splatting（RoGS）从多视角相机图像重建道路表面并结合预训练语义分割（Mask2Former）生成矢量化伪标签，同时设计 mask-aware 匹配算法与损失函数处理部分遮挡的伪标签，支持单次行程和多次行程（众包数据）两种模式。

**[Rayletdf Raylet Distance Fields For Generalizable 3D Surface Reconstruction From](3d_vision/rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)**

:   提出 RayletDF，一种基于"raylet"（光线片段）距离场的泛化3D表面重建方法，通过raylet特征提取器、距离场预测器和多raylet混合器三个模块，从点云或3D高斯直接预测表面点，在未见数据集上实现单次前向传播的高精度跨数据集泛化。

**[Rayzer A Self-Supervised Large View Synthesis Model](3d_vision/rayzer_a_self-supervised_large_view_synthesis_model.md)**

:   提出 RayZer，一个无需任何3D监督（无相机位姿/无场景几何标注）的自监督多视角3D视觉模型，通过将图像解耦为相机参数和场景表示实现3D感知自编码，在新视角合成任务上达到甚至超越依赖位姿标注的"oracle"方法。

**[Reggs Unposed Sparse Views Gaussian Splatting With 3Dgs Registration](3d_vision/reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)**

:   提出 RegGS 框架，通过基于最优传输 MW2 距离的可微 3DGS 配准模块，将前馈网络生成的局部3D高斯增量式地对齐到全局一致的3D表示中，实现无位姿稀疏视角的高质量3D重建。

**[Relative Illumination Fields Learning Medium And Light Independent Underwater Sc](3d_vision/relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)**

:   提出相对光照场（Relative Illumination Fields），通过在相机局部坐标系中用MLP建模非均匀光照分布，结合体积介质表示，实现对水下场景的干净重建——去除光源和介质的影响。

**[Reparo Compositional 3D Assets Generation With Differentiable 3D Layout Alignmen](3d_vision/reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)**

:   提出REPARO，通过先分别重建单个物体3D网格再利用基于最优传输的可微渲染进行布局对齐，实现从单张图像生成多物体组合式3D资产。

**[Reposed Efficient Relative Pose Estimation With Known Depth Information](3d_vision/reposed_efficient_relative_pose_estimation_with_known_depth_information.md)**

:   本文提出了一组高效的相对位姿最小求解器，将单目深度估计（MDE）的尺度和仿射参数与相对位姿联合估计，在标定/共焦距/不同焦距三种配置下均超越SOTA深度感知求解器，并通过大规模实验回答了"MDE深度是否有助于相对位姿估计"这一核心问题。

**[Representing 3D Shapes With 64 Latent Vectors For 3D Diffusion Models](3d_vision/representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)**

:   提出COD-VAE，通过两阶段自编码器方案（渐进式编码器 + Triplane解码器 + 不确定性引导Token剪枝），将3D形状编码为仅64个1D潜在向量，在保持重建质量的同时实现16×压缩比和20.8×生成加速。

**[Repurposing 2D Diffusion Models With Gaussian Atlas For 3D Generation](3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)**

:   提出 Gaussian Atlas 表示法，将无序3D高斯通过最优传输映射到球面再展平为规整2D网格，从而直接微调预训练2D Latent Diffusion模型实现高质量文本到3D生成。

**[Resgs Residual Densification Of 3D Gaussian For Efficient Detail Recovery](3d_vision/resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)**

:   提出残差分裂（residual split）操作替代3D-GS中split/clone的二元选择机制，配合图像金字塔渐进监督和可变梯度阈值选择策略，自适应地同时解决过重建和欠重建问题，在减少高斯数量的同时实现SOTA渲染质量。

**[Revisiting Point Cloud Completion Are We Ready For The Real-World](3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)**

:   通过代数拓扑和持久同调（PH）工具揭示现有合成点云数据集缺乏真实世界中丰富的拓扑特征，贡献了首个真实世界工业点云补全数据集RealPC（~40,000对、21类），并提出BOSHNet通过采样代理同调骨架作为拓扑先验，在真实世界点云补全上取得显著改进。

**[Ri3D Few-Shot Gaussian Splatting With Repair And Inpainting Diffusion Priors](3d_vision/ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)**

:   提出 RI3D，将稀疏视图合成分解为"修复可见区域"和"补全缺失区域"两个子任务，引入两个个性化扩散模型（repair + inpainting）配合两阶段优化策略，在极端稀疏输入下实现高质量 3DGS 重建。

**[Robopearls Editable Video Simulation For Robot Manipulation](3d_vision/robopearls_editable_video_simulation_for_robot_manipulation.md)**

:   提出 RoboPearls，基于 3D 高斯溅射（3DGS）构建的可编辑视频仿真框架，从演示视频中构建照片级真实感仿真环境，通过增量语义蒸馏（ISD）和3D正则化NNFM损失支持丰富的场景编辑操作，并利用 LLM 智能体自动化仿真生成流程，形成以 VLM 闭环驱动的机器人学习增强系统。

**[Robotron-Mani All-In-One Multimodal Large Model For Robotic Manipulation](3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)**

:   提出多模态机器人操作模型 RoboTron-Mani 和综合数据集 RoboData，通过相机参数与占用监督增强3D感知、Modality-Isolation-Mask 实现灵活多模态融合，首次作为通才策略在多个数据集上同时超越专家模型。

**[Robust And Efficient 3D Gaussian Splatting For Urban Scene Reconstruction](3d_vision/robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)**

:   提出一套面向城市级场景的高效鲁棒3DGS重建框架——通过可见性分区策略、可控LOD生成、细粒度外观变换模块及多种正则化技术，实现了在外观差异大、含瞬态物体的城市数据上高质量重建与实时渲染。

**[Robustereo Robust Zero-Shot Stereo Matching Under Adverse Weather](3d_vision/robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)**

:   提出 RobuSTereo 框架，通过基于扩散模型的立体数据生成管线和结合去噪 ViT 与 VGG19 的鲁棒特征编码器，大幅提升立体匹配模型在雨、雾、雪等恶劣天气下的零样本泛化能力。

**[Robustsplat Decoupling Densification And Dynamics For Transient-Free 3Dgs](3d_vision/robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)**

:   本文发现 3DGS 的高斯致密化过程是导致瞬态物体伪影的关键因素，提出延迟高斯生长策略和尺度级联掩码自举方法来解耦致密化与动态区域建模，在多个基准数据集上实现了最优的无瞬态新视角合成效果。

**[Roco-Sim Enhancing Roadside Collaborative Perception Through Foreground Simulati](3d_vision/roco-sim_enhancing_roadside_collaborative_perception_through_foreground_simulati.md)**

:   RoCo-Sim 是首个面向路侧协同感知的仿真框架，通过相机外参优化、多视图遮挡感知采样、DepthSAM 深度渲染和可扩展后处理工具包，从单张图像生成多样且多视图一致的路侧仿真数据，将路侧3D检测性能提升 83%+。

**[Ross3D Reconstructive Visual Instruction Tuning With 3D-Awareness](3d_vision/ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)**

:   Ross3D 提出将3D感知的视觉重建预训练任务（跨视图重建 + 全局BEV重建）注入2D大型多模态模型的训练流程中，在不修改输入表示的前提下通过输出级监督信号显著提升3D场景理解能力，在SQA3D、ScanQA、Scan2Cap、ScanRefer、Multi3DRefer五个基准上均达到SOTA。

**[S3E Self-Supervised State Estimation For Radar-Inertial System](3d_vision/s3e_self-supervised_state_estimation_for_radar-inertial_system.md)**

:   提出S3E，首次实现从雷达信号频谱和惯性数据的互补自监督状态估计，通过基于旋转的跨融合技术增强有限角分辨率下的空间结构信息。

**[S3R-Gs Streamlining The Pipeline For Large-Scale Street Scene Reconstruction](3d_vision/s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)**

:   S3R-GS 通过识别传统街景重建管线中的三大计算冗余（不必要的局部-全局坐标变换、过多的3D-2D投影、低效的远距离内容渲染），提出实例特定投影、时序可见性过滤和自适应LOD策略，将重建时间降至竞争方法的20%-50%，同时保持SOTA渲染质量。

**[Sas Segment Any 3D Scene With Integrated 2D Priors](3d_vision/sas_segment_any_3d_scene_with_integrated_2d_priors.md)**

:   提出 SAS 框架，首次整合多个 2D 开放词汇模型的互补能力来学习更好的 3D 表示：通过 Model Alignment via Text 对齐不同模型的特征空间，通过 Annotation-Free Model Capability Construction 利用扩散模型合成图像来量化各模型识别不同类别的能力，以此指导多模型特征融合和 3D 蒸馏，在 ScanNet v2/Matterport3D/nuScenes 上大幅超越前作。

**[Sat2City 3D City Generation From A Single Satellite Image With Cascaded Latent D](3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)**

:   提出 Sat2City，首个从单张卫星图像同时生成城市级几何和外观的3D生成框架，通过将稀疏体素与级联潜扩散模型结合，引入 Re-Hash 多尺度特征网格和逆采样策略，在自建3D城市数据集上实现了优于现有方法的高保真生成。

**[Scene Coordinate Reconstruction Priors](3d_vision/scene_coordinate_reconstruction_priors.md)**

:   提出场景坐标回归(SCR)的概率化训练框架，引入手工设计的深度分布先验和基于3D点云扩散模型的学习先验，在多视角约束不足时显著改善场景重建质量、相机位姿估计和下游任务表现。

**[Scenemi Motion In-Betweening For Modeling Human-Scene Interaction](3d_vision/scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)**

:   首次正式研究场景感知运动插值（scene-aware motion in-betweening）问题，提出 SceneMI 框架，通过双层场景描述符（全局体素 + 局部 BPS）全面编码场景上下文，利用扩散模型的去噪特性处理含噪关键帧，在 TRUMANS 上碰撞帧率降低 56.9%，在真实世界 GIMO 上脚部滑动减少 37.5%、抖动减少 56.5%。

**[Seeing And Seeing Through The Glass Real And Synthetic Data For Multi-Layer Dept](3d_vision/seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)**

:   提出多层深度估计(multi-layer depth estimation)新任务，构建了包含1500张真实图像的LayeredDepth基准和程序化合成数据生成器，揭示了现有深度估计方法在透明物体上的严重不足。

**[Segmentdreamer Towards High-Fidelity Text-To-3D Synthesis With Segmented Consist](3d_vision/segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)**

:   本文提出SegmentDreamer，通过分段一致性轨迹蒸馏（SCTD）重新表述SDS损失，解决了现有一致性蒸馏（CD）方法中自一致性和交叉一致性之间的不平衡问题，在单张A100 GPU上仅需~32分钟即可通过3DGS生成高保真3D资产。

**[Sehdr Single-Exposure Hdr Novel View Synthesis Via 3D Gaussian Bracketing](3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)**

:   SeHDR 是首个从单曝光多视图 LDR 图像生成 HDR 新视角的3DGS框架，通过在3D空间扩展包围曝光原理（Bracketed 3D Gaussians）并设计可微分神经曝光融合（NeEF）在球谐空间融合，无需 HDR 监督即超越现有方法 14.3dB。

**[Self-Ensembling Gaussian Splatting For Few-Shot Novel View Synthesis](3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)**

:   SE-GS 通过不确定性感知扰动策略在训练过程中动态生成多样化的 3DGS 模型，并利用自集成机制使 Σ-model 聚合扰动模型的信息，有效缓解稀疏视角下的过拟合问题，在多个数据集上实现 SOTA 的少样本新视角合成性能。

**[Self-Supervised Learning Of Hybrid Part-Aware 3D Representations Of 2D Gaussians](3d_vision/self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)**

:   提出 PartGS，一个自监督的部件感知3D重建框架，将2D Gaussian Splatting与超二次曲面混合耦合，通过参数共享和多种正则化实现同时高质量几何分解和纹理重建，在DTU、ShapeNet和真实场景上在重建精度上比SOTA提升75.9%，PSNR提升16.13dB。

**[Sequential Gaussian Avatars With Hierarchical Motion Context](3d_vision/sequential_gaussian_avatars_with_hierarchical_motion_context.md)**

:   提出 SeqAvatar，利用显式3DGS表示结合层次化运动上下文（粗粒度骨骼运动 + 细粒度逐点速度）建模人体化身的运动相关外观变化，并通过时空多尺度采样增强运动条件的鲁棒性，在多个数据集上取得SOTA渲染质量同时保持实时渲染速度。

**[Shape Of Motion 4D Reconstruction From A Single Video](3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)**

:   提出基于 $\mathbb{SE}(3)$ 运动基的动态 3D 高斯表示，从单目视频中恢复全局一致的 3D 运动轨迹，同时实现实时新视角合成和长程 3D 跟踪，在 iPhone 和 Kubric 数据集上全面超越先前方法。

**[Sheap Self-Supervised Head Geometry Predictor Learned Via 2D Gaussians](3d_vision/sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)**

:   提出SHeaP，利用2D Gaussian Splatting替代传统可微mesh渲染进行自监督3DMM预测训练，通过将Gaussians绑定到3DMM mesh上实现重动画，并设计graph卷积Gaussians生成器和几何一致性正则化，在NoW和Nersemble基准上超越所有自监督方法。

**[Sim3D Single-Instance Multiview Multimodal And Multisetup 3D Anomaly Detection B](3d_vision/sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)**

:   提出 SiM3D，首个面向多视角多模态3D异常检测与分割的基准，聚焦工业制造中的单实例场景，通过工业级传感器采集高分辨率数据，使用体素化异常体积(Anomaly Volume)替代2D异常图，并首次支持合成到真实的跨域评估。

**[Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)**

:   Sdirt 提出基于光线追踪的双像素（DP）图像模拟方案，通过精确计算包含像差和相位分裂信息的空间变化 DP PSF，弥合仿真与真实 DP 数据之间的域间差距，使深度估计模型在真实 DP 图像上具有更好的泛化能力。

**[Skysense V2 A Unified Foundation Model For Multi-Modal Remote Sensing](3d_vision/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)**

:   本文提出SkySense V2，使用单一统一Transformer骨干网络处理高分辨率光学/多光谱/SAR三种遥感模态数据，通过自适应Patch合并、模态特异性Prompt Token和基于Query的语义聚合对比学习（QSACL）进行预训练，仅用665M参数（相比前作SkySense的1.26B）在16个数据集7种任务上平均提升1.8分。

**[Sl2A-Inr Single-Layer Learnable Activation For Implicit Neural Representation](3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)**

:   提出SL2A-INR，通过单层基于Chebyshev多项式的可学习激活函数块与ReLU-MLP融合块的混合架构，有效缓解隐式神经表示中的频谱偏差问题，在图像拟合、3D形状重建和新视角合成任务上达到SOTA。

**[Sparfels Fast Reconstruction From Sparse Unposed Imagery](3d_vision/sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)**

:   提出Sparfels方法，将3D基础模型（MASt3R）与高效的测试时优化（2DGS）相结合，通过MASt3R提供初始化点云/相机和对应关系引导优化，并创新性地引入泼溅色彩方差损失，在3分钟内从稀疏无位姿图像实现SOTA几何重建。

**[Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](3d_vision/spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)**

:   提出 4D Diffusion Policy（DP4），通过动态高斯世界模型为扩散策略注入3D空间和4D时空感知能力，在17个仿真任务和3个真实机器人任务上大幅超越基线（Adroit +16.4%, DexArt +14%, RLBench +6.45%, 真实任务 +8.6%）。

**[Spatialsplat Efficient Semantic 3D From Sparse Unposed Images](3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)**

:   提出SpatialSplat,通过双场语义表示和选择性Gaussian机制,从稀疏无位姿图像前馈生成紧凑的语义3D Gaussian,将表示参数量减少60%同时超越SOTA方法。

**[Spinmeround Consistent Multi-View Identity Generation Using Diffusion Models](3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)**

:   提出 SpinMeRound，一种基于身份嵌入的多视角扩散模型，能从单张或少量人脸图像生成 360° 全头部一致性肖像及对应法线图，在人脸新视角合成任务上超越现有多视角扩散方法。

**[Splattalk 3D Vqa With Gaussian Splatting](3d_vision/splattalk_3d_vqa_with_gaussian_splatting.md)**

:   提出SplatTalk，利用可泛化的3D Gaussian Splatting框架生成与LLM兼容的3D token，仅需多视角RGB图像即可实现零样本3D视觉问答，性能超越2D LMM方法并接近3D LMM。

**[Stable Score Distillation](3d_vision/stable_score_distillation.md)**

:   提出 Stable Score Distillation (SSD)，通过单分类器跨提示词引导和 null-text 分支的跨轨迹正则化，实现更稳定精准的文本引导 2D/3D 编辑，在保持源内容结构的同时提升编辑对齐度。

**[Stealthattack Robust 3D Gaussian Splatting Poisoning Via Density-Guided Illusion](3d_vision/stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)**

:   首次针对3D高斯泼溅(3DGS)提出密度引导的投毒攻击方法，通过在低密度区域注入幻觉物体的高斯点并引入自适应噪声破坏多视角一致性，实现从目标视角清晰可见而不干扰其余视角的隐蔽攻击。

**[Stereo Any Video Temporally Consistent Stereo Matching](3d_vision/stereo_any_video_temporally_consistent_stereo_matching.md)**

:   提出Stereo Any Video框架，通过融合单目视频深度基础模型先验(Video Depth Anything)、全对全配对相关(all-to-all-pair correlation)和时序凸上采样(temporal convex upsampling)三大核心模块，在不依赖相机位姿或光流的前提下实现空间精确且时序一致的视频立体匹配，在多个数据集零样本设定下达到SOTA。

**[Stochasticsplats Stochastic Rasterization For Sorting-Free 3D Gaussian Splatting](3d_vision/stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)**

:   StochasticSplats 将随机透明度（Stochastic Transparency）引入 3DGS，通过无偏 Monte Carlo 估计替代深度排序的 alpha 混合，实现免排序、无 popping 的渲染，在 1 SPP 下比标准 CUDA 3DGS 快 4×，并可通过采样数灵活权衡质量与速度。

**[Strandhead Text To Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)**

:   提出 StrandHead，首个通过蒸馏人体特定2D扩散模型来生成发丝级3D头部化身的框架，提出可微棱柱化算法实现发丝到水密网格的转换和梯度反传，并设计基于统计发丝几何先验的正则化损失保证发型的真实性。

**[Strumamba3D Exploring Structural Mamba For Self-Supervised Point Cloud Represent](3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)**

:   提出 StruMamba3D，通过为 SSM 的隐含状态赋予空间位置属性（空间状态）来维护 3D 点的邻接关系，并引入序列长度自适应策略解决预训练与下游任务之间的序列长度差异问题，在 ScanObjectNN 最难分割上达到 92.75% 准确率，ModelNet40 达到 95.1%，均为单模态 SOTA。

**[Superdec 3D Scene Decomposition With Superquadrics Primitives](3d_vision/superdec_3d_scene_decomposition_with_superquadrics_primitives.md)**

:   提出SuperDec,基于Transformer的学习方法将点云分解为紧凑的超二次曲面基元集合,在ShapeNet上训练即可泛化到真实场景,支持机器人操作和可控生成。

**[Supermat Physically Consistent Pbr Material Estimation At Interactive Rates](3d_vision/supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)**

:   提出SuperMat，一个单步推理的PBR材质分解框架，通过结构化专家分支和调度器修正实现端到端训练，引入re-render loss确保物理一致性，将推理速度从秒级提升至毫秒级。

**[Svg-Head Hybrid Surface-Volumetric Gaussians For High-Fidelity Head Reconstructi](3d_vision/svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)**

:   提出SVG-Head，通过表面高斯(显式纹理图)和体积高斯(非朗伯区域补充建模)的混合表示，首次实现高保真高斯头部化身的实时外观编辑。

**[Tar3D Creating High-Quality 3D Assets Via Next-Part Prediction](3d_vision/tar3d_creating_high-quality_3d_assets_via_next-part_prediction.md)**

:   提出TAR3D框架——首次将三平面表示量化为离散几何部件并用GPT自回归生成，通过3D VQ-VAE编码任意面数网格为固定长度序列+TriPE位置编码保留3D空间信息，在文本/图像→3D任务上全面超越现有方法。

**[Text2Vdm Text To Vector Displacement Maps For Expressive And Interactive 3D Scul](3d_vision/text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)**

:   提出Text2VDM,首个从文本生成VDM雕刻笔刷的框架,通过Sobolev预条件网格变形和语义增强SDS损失解决子对象结构生成中的语义耦合问题。

**[Textured 3D Regenerative Morphing With 3D Diffusion Prior](3d_vision/textured_3d_regenerative_morphing_with_3d_diffusion_prior.md)**

:   提出基于3D扩散先验的再生式3D morphing方法，通过在初始噪声、模型参数和条件特征三个层级进行插值，结合Attention Fusion、Token Reordering和Low-Frequency Enhancement三种策略，首次实现了跨类别纹理3D物体的平滑、合理变形序列生成。

**[Timeformer Capturing Temporal Relationships Of Deformable 3D Gaussians For Robus](3d_vision/timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)**

:   提出TimeFormer模块,通过跨时间Transformer编码器隐式学习可变形3D Gaussian的时序关系,并设计双流优化策略在训练时迁移运动知识,推理时无额外开销。

**[Tokenunify Scaling Up Autoregressive Pretraining For Neuron Segmentation](3d_vision/tokenunify_scaling_up_autoregressive_pretraining_for_neuron_segmentation.md)**

:   提出 TokenUnify，通过统一随机 token 预测、下一 token 预测和下一全部 token 预测三种互补学习目标，在大规模电子显微镜数据上实现层次化预测编码，将自回归误差累积从 O(K) 降至 O(√K)，下游神经元分割提升 44%。

**[Towards More Diverse And Challenging Pre-Training For Point Cloud Learning Self-](3d_vision/towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)**

:   提出Point-PQAE，首个将跨视图重建（Cross Reconstruction）引入3D生成式自监督学习的框架，通过点云裁剪机制生成解耦视图、设计视图相对位置编码（VRPE）和位置查询模块，使预训练更具挑战性和信息量，在ScanObjectNN上以Mlp-Linear协议平均超越Point-MAE 6.7%。

**[Towards Scalable Spatial Intelligence Via 2D-To-3D Data Lifting](3d_vision/towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)**

:   提出一个可扩展的数据生成管线，通过集成深度估计、相机标定和尺度校准，将单视图2D图像自动转换为包含点云、相机位姿、深度图的尺度真实3D表示，生成了约200万场景的COCO-3D和Objects365-v2-3D数据集，显著提升多种3D任务性能。

**[Trace3D Consistent Segmentation Lifting Via Gaussian Instance Tracing](3d_vision/trace3d_consistent_segmentation_lifting_via_gaussian_instance_tracing.md)**

:   提出Gaussian Instance Tracing (GIT)机制，通过反向光栅化为每个高斯核维护跨视角的实例权重矩阵，统一解决2D分割多视角不一致和边界高斯模糊两大问题，在离线对比学习和在线自提示两种设定下均显著提升3D分割质量。

**[Trace Learning 3D Gaussian Physical Dynamics From Multi-View Videos](3d_vision/trace_learning_3d_gaussian_physical_dynamics_from_multi-view_videos.md)**

:   提出TRACE框架，将每个3D高斯核视为刚性粒子并为其学习独立的平移-旋转动力学系统（包含速度、加速度、角速度、角加速度等完整物理参数），无需任何人工标注即可从多视角动态视频中学习3D场景的物理运动规律并准确外推未来帧。

**[Tridi Trilateral Diffusion Of 3D Humans Objects And Interactions](3d_vision/tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)**

:   提出 TriDi，首个建模人体(H)、物体(O)和交互(I)三变量联合分布的统一扩散模型，一个网络覆盖 7 种条件生成模式，超越各专用单向基线。

**[Tune-Your-Style Intensity-Tunable 3D Style Transfer With Gaussian Splatting](3d_vision/tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)**

:   提出 Tune-Your-Style，首个强度可调的 3D 风格迁移范式，通过 Gaussian 神经元显式建模风格强度并参数化可学习 style tuner，配合两阶段优化策略，实现用户自由调节风格注入的程度。

**[Turboreg Turboclique For Robust And Efficient Point Cloud Registration](3d_vision/turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)**

:   提出 TurboReg 框架，通过定义轻量级 3-clique（TurboClique）替代传统最大团搜索，并设计高度可并行的 Pivot-Guided Search（PGS）算法，在保持SOTA配准精度的同时将速度提升 208× 以上。

**[Uniegomotion A Unified Model For Egocentric Motion Reconstruction Forecasting An](3d_vision/uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)**

:   提出 UniEgoMotion，首个统一的自中心运动模型，通过条件运动扩散框架和头部中心运动表示，在单一模型中实现自中心视角下的3D人体运动重建、预测和生成三项任务，并发布大规模EE4D-Motion数据集。

**[Unified Category-Level Object Detection And Pose Estimation From Rgb Images Usin](3d_vision/unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)**

:   首次提出将物体检测与类别级位姿估计统一到单一模型的 RGB-only 框架，利用 Neural Mesh Models 作为3D原型表示，通过特征匹配和多模型 RANSAC PnP 同时实现检测和 9D 位姿估计，在 REAL275 上所有 scale-agnostic 指标均超越 SOTA。

**[Univg A Generalist Diffusion Model For Unified Image Generation And Editing](3d_vision/univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)**

:   提出UniVG,基于MM-DiT的统一图像生成模型,通过通道维拼接输入、渐进式多任务训练和外部条件注入,用单套权重支持T2I生成、编辑、ID保持、布局引导、深度估计等多种任务。

**[Unleashing Vecset Diffusion Model For Fast Shape Generation](3d_vision/unleashing_vecset_diffusion_model_for_fast_shape_generation.md)**

:   FlashVDM 提出系统性框架加速 Vecset Diffusion Model（VDM）的 DiT 采样和 VAE 解码：通过渐进式流蒸馏将扩散步骤降至 5 步，通过自适应 KV 选择 + 层次体素解码 + 高效解码器将 VAE 解码加速 45×，整体实现 32× 加速至 1 秒内生成高质量 3D 形状。

**[Upp Unified Point-Level Prompting For Robust Point Cloud Analysis](3d_vision/upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)**

:   提出统一点级提示方法UPP，将点云去噪和补全重新定义为下游任务的提示机制，通过Rectification Prompter过滤噪声、Completion Prompter补全缺失、Shape-Aware Unit捕获几何特征，在噪声和不完整点云上以6.3%参数实现超越全量微调的鲁棒分析。

**[Ust-Ssm Unified Spatio-Temporal State Space Models For Point Cloud Video Modelin](3d_vision/ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)**

:   提出UST-SSM，通过时空选择扫描(STSS)、时空结构聚合(STSA)和时序交互采样(TIS)三个核心模块，将选择性状态空间模型扩展到点云视频分析，以线性复杂度实现优于Transformer的性能。

**[Vertexregen Mesh Generation With Continuous Level Of Detail](3d_vision/vertexregen_mesh_generation_with_continuous_level_of_detail.md)**

:   提出VertexRegen，受渐进网格启发将网格生成重新定义为边折叠(edge collapse)的逆操作——顶点分裂(vertex split)的学习，实现连续细节层级的"随时停止"网格生成。

**[Vit-Split Unleashing The Power Of Vision Foundation Models Via Efficient Splitti](3d_vision/vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)**

:   基于"VFM 层可分为低层特征提取器和高层任务适配器"的关键观察，提出 ViT-Split，通过冻结 VFM + task head（复制最后 $K_t$ 层）+ prior head（轻量 CNN 聚合多尺度先验特征）的设计，在 ADE20K 上仅用线性头即达到 58.2 mIoU（DINOv2-L），训练速度提升 4 倍，可训练参数仅为传统适配器的 1/4~1/5。

**[Vivid4D Improving 4D Reconstruction From Monocular Video By Video Inpainting](3d_vision/vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)**

:   本文提出Vivid4D，将单目视频的多视角增广任务转化为视频修复（inpainting）问题——先用单目深度先验将视频warp到新视角，再用视频扩散模型修复遮挡区域，通过迭代视角扩展策略和鲁棒重建损失显著改善了单目4D动态场景的重建质量。

**[Volume - Authentic 3D Video Calls From Live Gaussian Splat Prediction](3d_vision/volume_-_authentic_3d_video_calls_from_live_gaussian_splat_prediction.md)**

:   微软提出首个从单目2D摄像头实时预测3D高斯泼溅重建的方法，实现真实感、保真性、实时性和时序稳定性四项要求的统一，使任何人仅用标准笔记本摄像头即可进行体积3D视频通话。

**[Volumetricsmpl A Neural Volumetric Body Model For Efficient Interactions Contact](3d_vision/volumetricsmpl_a_neural_volumetric_body_model_for_efficient_interactions_contact.md)**

:   提出 VolumetricSMPL，一种基于 Neural Blend Weights（NBW）的高效神经体积人体模型，相比前代 COAP 实现 10× 推理加速、6× 显存节省，并通过 SDF（而非占据函数）表示提供更精确的可微碰撞建模。

**[Wildseg3D Segment Any 3D Objects In The Wild From 2D Images](3d_vision/wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)**

:   提出 WildSeg3D，首个前馈式3D分割模型，无需场景特定训练，通过动态全局对齐(DGA)解决多视角点图对齐误差，结合多视角组映射(MGM)实现实时交互式3D分割，比现有SOTA快40倍且精度更优。

**[Wonderplay Dynamic 3D Scene Generation From A Single Image And Actions](3d_vision/wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)**

:   WonderPlay 提出混合生成模拟器（Hybrid Generative Simulator），将物理求解器的粗糙3D动态仿真与视频扩散模型的高质量生成相结合，实现从单张图像加用户动作输入生成逼真多材质动态3D场景，支持刚体、布料、液体、烟雾、颗粒等多种材质。

**[Zero-Shot Inexact Cad Model Alignment From A Single Image](3d_vision/zero-shot_inexact_cad_model_alignment_from_a_single_image.md)**

:   提出一种弱监督的9-DoF CAD模型对齐方法，通过增强DINOv2特征的几何感知能力并在归一化物体坐标（NOC）空间进行稠密对齐优化，实现无需位姿标注、可泛化到未见类别的零样本3D对齐。

**[Zerostereo Zero-Shot Stereo Matching From Single Images](3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)**

:   提出 ZeroStereo 管线：从任意单张图像出发，利用单目深度估计生成伪视差，再用微调的扩散修复模型合成高质量右视图，实现只需 35K 合成数据即达到 SOTA 零样本立体匹配泛化性能。

---

## 🎨 图像生成 { #image_generation }

**[A0 An Affordance-Aware Hierarchical Model For General Robotic Manipulation](image_generation/a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)**

:   提出 A0，一个层次化可供性感知扩散模型，通过预测以物体为中心的接触点和后接触轨迹（Embodiment-Agnostic Affordance Representation），将操控任务分解为高层空间理解和低层动作执行，在 100 万接触点数据上预训练后可跨 Franka/Kinova/Realman/Dobot 四种平台泛化。

**[A Unified Framework For Motion Reasoning And Generation In Human Interaction](image_generation/a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)**

:   提出 MoLaM，一个统一的交互式动作-语言模型，通过三阶段训练和新构建的 Inter-MT² 数据集（82.7K 多轮指令），首次在单一框架内同时实现双人交互动作的理解、生成、编辑和推理。

**[Accelerating Diffusion Sampling Via Exploiting Local Transition Coherence](image_generation/accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)**

:   提出 LTC-Accel，一种基于"局部转移一致性"(Local Transition Coherence) 现象的免训练扩散采样加速方法，通过利用相邻去噪步之间转移算子的强相关性来近似替代当前步的计算，在 Stable Diffusion v2 上实现 1.67× 加速，与蒸馏模型结合可在视频生成中达到 10× 加速。

**[Adaptive Routing Of Text-To-Image Generation Requests Between Large Cloud Model ](image_generation/adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)**

:   提出 RouteT2I，首个面向文本到图像生成的边缘-云模型路由框架，通过多维质量度量、Pareto 相对优越性和双门控 token 选择 MoE 架构，在控制成本的同时最大化图像生成质量。

**[Addressing Text Embedding Leakage In Diffusion-Based Image Editing](image_generation/addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)**

:   揭示了基于扩散模型的文本图像编辑中属性泄露的根本原因——自回归文本编码器中 EOS 嵌入的语义纠缠，并提出 ALE 框架（ORE + RGB-CAM + BB），从嵌入解耦、注意力遮罩和背景混合三个层面彻底消除属性泄露。

**[Adiee Automatic Dataset Creation And Scorer For Instruction-Guided Image Editing](image_generation/adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)**

:   本文提出 ADIEE，一种自动化构建指令引导图像编辑评估数据集的方法，并基于超过 10 万样本微调 LLaVA-NeXT-8B 模型作为评分器，在多个基准上超越所有开源 VLM 和 Gemini-Pro 1.5，同时可作为奖励模型提升图像编辑模型性能。

**[Aether Geometric-Aware Unified World Modeling](image_generation/aether_geometric-aware_unified_world_modeling.md)**

:   Aether 提出一个几何感知的统一世界模型框架，通过在合成 4D 数据上联合训练重建、预测和规划三大能力，基于 CogVideoX 后训练实现零样本泛化到真实场景。

**[Aether Geometricaware Unified World Modeling](image_generation/aether_geometricaware_unified_world_modeling.md)**

:   提出Aether统一世界模型，在合成RGB-D数据上后训练CogVideoX视频扩散模型，通过随机组合输入/输出模态的多任务训练策略，同时实现4D重建、动作条件视频预测和目标条件视觉规划，且零样本迁移到真实世界数据达到与领域专用模型可比的性能。

**[Animegamer Infinite Anime Life Simulation With Next Game State Prediction](image_generation/animegamer_infinite_anime_life_simulation_with_next_game_state_prediction.md)**

:   提出 AnimeGamer，基于多模态大语言模型(MLLM)的无限动漫生活模拟系统，通过动作感知的多模态表征预测下一轮游戏状态（动态动画镜头 + 角色状态更新），实现持续一致的交互式动漫游戏体验。

**[Anti-Tamper Protection For Unauthorized Individual Image Generation](image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)**

:   提出Anti-Tamper Perturbation (ATP)，在频域中将保护扰动（阻止伪造生成）和授权扰动（检测净化篡改）分离嵌入，当攻击者尝试净化保护信息时触发防篡改机制拒绝服务，在各种净化攻击下实现100%保护成功率。

**[Anyportal Zero-Shot Consistent Video Background Replacement](image_generation/anyportal_zero-shot_consistent_video_background_replacement.md)**

:   AnyPortal 提出了一个零样本、免训练的视频背景替换框架，通过协同利用 IC-Light 的重光照能力和视频扩散模型（CogVideoX）的时序先验，配合新提出的 Refinement Projection Algorithm (RPA) 实现像素级前景保持，在单张 24GB GPU 上即可高效运行。

**[Autoprompt Automated Red-Teaming Of Text-To-Image Models Via Llm-Driven Adversar](image_generation/autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)**

:   本文提出APT（AutoPrompT），一种基于LLM的黑盒红队测试框架，通过"优化-微调"交替训练管线和双规避策略，自动生成可被人类阅读且不被内容过滤器拦截的对抗性后缀，有效突破T2I模型的安全机制，并具有零样本跨提示迁移能力。

**[Balanced Image Stylization With Style Matching Score](image_generation/balanced_image_stylization_with_style_matching_score.md)**

:   提出 Style Matching Score（SMS），将图像风格化重铸为风格分布匹配问题，通过渐进频谱正则化和语义感知梯度精炼，在风格对齐与内容保持之间取得卓越平衡，并可蒸馏到轻量前馈网络实现一步风格化。

**[Bitrate-Controlled Diffusion For Disentangling Motion And Content In Video](image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)**

:   提出BCD（Bitrate-Controlled Diffusion），一种通用的自监督视频解耦框架，通过低码率矢量量化作为信息瓶颈来分离视频中的逐帧运动特征和全局内容特征，并以条件扩散模型重建视频，在说话人头部视频和像素风格卡通数据集上展示了高质量的运动迁移和自回归视频生成能力。

**[Bridging The Skeleton-Text Modality Gap Diffusion-Powered Modality Alignment For](image_generation/bridging_the_skeleton-text_modality_gap_diffusion-powered_modality_alignment_for.md)**

:   提出TDSM（Triplet Diffusion for Skeleton-Text Matching），首次将扩散模型应用于零样本骨骼动作识别，通过反向扩散过程实现骨骼特征与文本prompt的隐式对齐，并引入triplet diffusion损失增强判别力，在NTU-60/120和PKU-MMD上大幅超越SOTA（2.36%到13.05%的提升幅度）。

**[Bvinet Unlocking Blind Video Inpainting With Zero Annotations](image_generation/bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)**

:   首次定义并解决"盲视频修复"（blind video inpainting）任务——在无需任何损坏区域标注的情况下，端到端地同时完成"哪里需要修复"和"如何修复"，通过 mask 预测网络与视频补全网络的一致性约束互相增强，在合成数据和真实应用（弹幕去除/划痕修复）中均取得优异效果。

**[Calibrating Mllm-As-A-Judge Via Multimodal Bayesian Prompt Ensembles](image_generation/calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)**

:   提出Multimodal Mixture-of-Bayesian Prompt Ensembles (MMB)，通过基于图像聚类的多模态感知提示权重学习，显著改善MLLM作为评判者时的校准性和判断准确性，解决了标准提示集成方法在多模态场景下失效的问题。

**[Cao2 Rectifying Inconsistencies In Diffusion-Based Dataset Distillation](image_generation/cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)**

:   揭示了基于扩散模型的数据集蒸馏中存在的"目标不一致"和"条件不一致"两个关键问题，提出两阶段框架CaO2：第一阶段通过分类器引导的样本选择缓解目标不一致，第二阶段通过隐空间优化最大化条件似然缓解条件不一致，在ImageNet上平均提升2.3%。

**[Cap Evaluation Of Persuasive And Creative Image Generation](image_generation/cap_evaluation_of_persuasive_and_creative_image_generation.md)**

:   针对广告图像生成任务，提出三个新评估指标（创意性、对齐度、说服力），并用LLM扩展隐式消息为显式视觉描述来改善T2I模型的广告生成效果，在人类标注一致性上显著优于CLIPScore等基线指标。

**[Characonsist Fine-Grained Consistent Character Generation](image_generation/characonsist_fine-grained_consistent_character_generation.md)**

:   提出一种免训练的细粒度一致性角色生成方法，通过点跟踪注意力（Point-Tracking Attention）、自适应 token 合并和前景-背景解耦控制，首次在 DiT 架构（FLUX.1）上实现了高质量的跨图像角色一致性生成。

**[Chords Diffusion Sampling Accelerator With Multi-Core Hierarchical Ode Solvers](image_generation/chords_diffusion_sampling_accelerator_with_multi-core_hierarchical_ode_solvers.md)**

:   提出 Chords，一种基于多核层次 ODE 求解器的扩散采样加速框架，通过慢到快的核间纠正机制（inter-core rectification），在 4-8 个 GPU 上实现 2.1×~2.9× 加速，且不牺牲生成质量。

**[Cns-Bench Benchmarking Image Classifier Robustness Under Continuous Nuisance Shi](image_generation/cns-bench_benchmarking_image_classifier_robustness_under_continuous_nuisance_shi.md)**

:   提出 CNS-Bench，首个利用 LoRA 适配器对扩散模型施加**连续**且**逼真**的干扰偏移（nuisance shift）来系统评估图像分类器 OOD 鲁棒性的基准，覆盖 14 种偏移类型、5 个严重度级别和 40+ 分类器。

**[Compass Enhancing Spatial Understanding In Text-To-Image Diffusion Models](image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)**

:   CoMPaSS通过SCOP数据引擎筛选空间关系无歧义的训练数据，并提出无参数的TENOR模块将token顺序信息注入注意力机制，大幅提升T2I扩散模型的空间关系生成准确率（VISOR +98%、GenEval Position +131%）。

**[Completeme Reference-Based Human Image Completion](image_generation/completeme_reference-based_human_image_completion.md)**

:   提出CompleteMe框架，通过双U-Net架构和Region-focused Attention（RFA）Block，利用参考图像中的细粒度人物细节（衣物纹理、纹身等），实现高保真的参考引导人体图像补全。

**[Compression-Aware One-Step Diffusion Model For Jpeg Artifact Removal](image_generation/compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)**

:   提出 CODiff，一种压缩感知的单步扩散模型用于 JPEG 伪影去除，核心是设计了压缩感知视觉嵌入器（CaVE）通过显式+隐式双重学习策略提取 JPEG 压缩先验，引导扩散模型实现高质量复原，在 LIVE-1、Urban100、DIV2K-Val 上全面超越现有方法同时推理效率极高。

**[Compslider Compositional Slider For Disentangled Multiple-Attribute Image Genera](image_generation/compslider_compositional_slider_for_disentangled_multiple-attribute_image_genera.md)**

:   提出 CompSlider，一个组合式滑块模型，通过生成条件先验来实现对 T2I 基础模型中多个属性的同时、独立、细粒度控制，利用解耦损失和结构损失来解决多属性之间的纠缠问题。

**[Contrastive Flow Matching](image_generation/contrastive_flow_matching.md)**

:   在 Flow Matching 的训练目标中引入对比正则项，强制不同条件的流场互相远离，从而在零额外推理开销下实现 9× 训练加速、5× 更少采样步数、FID 最多降低 8.9。

**[Csd-Var Content-Style Decomposition In Visual Autoregressive Models](image_generation/csd-var_content-style_decomposition_in_visual_autoregressive_models.md)**

:   首次在视觉自回归模型（VAR）中探索内容-风格分解（CSD），通过尺度感知交替优化、SVD风格嵌入修正和增强型K-V记忆三项创新，实现优于扩散模型方法的内容保持与风格迁移效果。

**[Cure Cultural Gaps In The Long Tail Of Text-To-Image Systems](image_generation/cure_cultural_gaps_in_the_long_tail_of_text-to-image_systems.md)**

:   提出 CURE 基准与评分套件，利用**属性规范的边际效用**（Marginal Information Attribution）作为人类判断的代理指标，系统评估 T2I 系统在全球文化长尾分布上的代表性能力。

**[Cycle Consistency As Reward Learning Image-Text Alignment Without Human Preferen](image_generation/cycle_consistency_as_reward_learning_image-text_alignment_without_human_preferen.md)**

:   利用循环一致性（图→文→图或文→图→文的重建相似度）作为替代人类偏好的监督信号，构建866K偏好数据集CyclePrefDB，训练的CycleReward模型在详细描述生成评估上超越所有现有方法，并可通过DPO提升VLM和扩散模型。

**[Cycle Consistency As Reward Learning Imagetext Alignment Wit](image_generation/cycle_consistency_as_reward_learning_imagetext_alignment_wit.md)**

:   提出CycleReward，利用cycle consistency作为自监督信号替代人工偏好标注——将caption用T2I模型重建为图像再比较相似度来排序，构建866K偏好对数据集CyclePrefDB，训练的奖励模型在detailed captioning上比HPSv2/PickScore/ImageReward高6%+，且DPO训练后提升VLM在多个VL任务上的性能，无需任何人工标注。

**[Dc-Ar Efficient Masked Autoregressive Image Generation With Deep Compression Hyb](image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)**

:   提出 DC-AR，一个基于深度压缩混合标记器（DC-HT，32× 空间压缩）的掩码自回归文本到图像生成框架，通过离散 token 生成结构 + 残差 token 精细化的混合流程，在 MJHQ-30K 上取得 SOTA gFID 5.49，同时吞吐量比扩散模型高 1.5-7.9×。

**[Dct-Shield A Robust Frequency Domain Defense Against Malicious Image Editing](image_generation/dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)**

:   提出 DCT-Shield，在离散余弦变换（DCT）域中引入对抗扰动而非像素空间，使免疫噪声高度不可感知，并天然具备JPEG压缩鲁棒性，有效抵御基于扩散模型的恶意图像编辑。

**[Deeply Supervised Flow-Based Generative Models](image_generation/deeply_supervised_flow-based_generative_models.md)**

:   DeepFlow 通过在 flow-based 模型的 Transformer 层间引入深度监督和 VeRA（Velocity Refiner with Acceleration）模块，利用二阶 ODE 动力学对齐中间层速度特征，在不依赖外部预训练模型的情况下实现 8 倍训练加速和显著 FID 提升。

**[Deepshield Fortifying Deepfake Video Detection With Local And Global Forgery Ana](image_generation/deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)**

:   提出 DeepShield，一种结合局部 patch 级引导（LPG）和全局伪造多样化（GFD）的深度伪造视频检测框架，通过时空伪影建模提供 patch 级监督、分布级特征增强合成多样伪造表征，在跨数据集和跨操控类型评估中显著超越 SOTA。

**[Dense2Moe Restructuring Diffusion Transformer To Moe For Eff](image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_eff.md)**

:   首次将预训练的dense DiT（FLUX.1 [dev] 12B参数）通过三步蒸馏pipeline转换为结构化稀疏的MoE架构——用MoE层替换FFN实现token级稀疏、用Mixture of Blocks（MoB）实现block级动态跳过——激活参数从12B降至5.2B（减少56%+）的同时保持原始性能，全面超越同等压缩比的剪枝方法。

**[Dense2Moe Restructuring Diffusion Transformer To Moe For Efficient Text-To-Image](image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)**

:   首次提出将密集型扩散Transformer（DiT）转化为MoE稀疏结构的范式Dense2MoE，通过FFN替换为MoE层+Transformer块分组为MoB（Mixture of Blocks），配合多阶段蒸馏流水线，将FLUX.1的12B参数压缩至5.2B激活参数同时保持原始性能，全面超越剪枝方法。

**[Dense Policy Bidirectional Autoregressive Learning Of Actions](image_generation/dense_policy_bidirectional_autoregressive_learning_of_actions.md)**

:   提出 Dense Policy，一种基于双向自回归扩展的机器人操作策略，通过对数时间的粗到细层次化动作生成，在仿真和真实世界任务中超越 Diffusion Policy 和 ACT 等主流生成式策略。

**[Describe Dont Dictate Semantic Image Editing With Natural Language Intent](image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)**

:   提出 DescriptiveEdit，将"指令式图像编辑"重新定义为"参考图像条件下的文本到图像生成"，通过 Cross-Attentive UNet 引入注意力桥接层将参考图像特征注入生成过程，仅需 75M 可训练参数即可实现高保真描述式编辑，并与 ControlNet、IP-Adapter 等社区工具无缝兼容。

**[Dice Staleness-Centric Optimizations For Parallel Diffusion Moe Inference](image_generation/dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)**

:   针对 MoE 扩散模型并行推理中的"陈旧性"问题 (staleness)，提出 DICE 框架，通过步级交织并行、层级选择性同步和 token 级条件通信三层优化策略，在 DiT-MoE 上实现 1.26× 加速且质量损失极小。

**[Diffsim Taming Diffusion Models For Evaluating Visual Similarity](image_generation/diffsim_taming_diffusion_models_for_evaluating_visual_similarity.md)**

:   DiffSim 首次发现预训练扩散模型（Stable Diffusion）的注意力层特征可用于测量视觉相似度，提出 Aligned Attention Score (AAS) 在 U-Net 的 self-attention / cross-attention 层中对齐两张图像特征后计算余弦相似度，在人类感知一致性、风格相似度和实例一致性等多个 benchmark 上达到 SOTA。

**[Diffumatch Category-Agnostic Spectral Diffusion Priors For Robust Non-Rigid Shap](image_generation/diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)**

:   提出在功能映射（Functional Map）的谱域上训练无条件扩散模型，通过蒸馏学习到的结构先验替代传统公理化正则项（如拉普拉斯交换性、正交性），实现跨类别零样本非刚性形状匹配。

**[Diffusion-Based 3D Hand Motion Recovery With Intuitive Physics](image_generation/diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)**

:   提出一种物理增强的条件扩散模型，通过迭代去噪过程将逐帧 3D 手部重建结果细化为时序一致的运动序列，并结合直觉物理约束（运动学约束和稳定性约束）大幅提升重建精度和物理合理性。

**[Diffusion Image Prior](image_generation/diffusion_image_prior.md)**

:   发现预训练扩散模型在重建退化图像时存在类似 Deep Image Prior 的隐式偏置——迭代优化过程中先生成干净图像再过拟合到退化输入——且比 DIP 适用更广泛的退化类型，据此提出完全盲（无需退化模型）的图像复原方法 DIIP。

**[Disrupting Model Merging A Parameter-Level Defense Without Sacrificing Accuracy](image_generation/disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)**

:   提出 PaRaMS（Parameter Rearrangement & Random Multi-head Scaling），一种参数级主动防御方法，通过功能等价的参数变换将模型推离共享损失盆地，使得被保护模型在合并时性能严重退化，同时保持未合并时的原始性能。

**[Ditfastattnv2 Head-Wise Attention Compression For Multi-Modality Diffusion Trans](image_generation/ditfastattnv2_head-wise_attention_compression_for_multi-modality_diffusion_trans.md)**

:   针对多模态扩散Transformer（MMDiT）提出DiTFastAttnV2，通过Head-wise Arrow Attention和Head-wise Caching机制实现细粒度的注意力压缩，在2K图像生成中减少68%注意力FLOPs并实现1.5倍端到端加速，且不损失视觉质量。

**[Dmq Dissecting Outliers Of Diffusion Models For Post-Training Quantization](image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)**

:   提出 DMQ 框架，结合学习型等价缩放（LES）和通道级 Power-of-Two 缩放（PTS）来处理扩散模型量化中的异常值问题，首次在 W4A6 低比特设定下实现稳定的高质量图像生成。

**[Domain Generalizable Portrait Style Transfer](image_generation/domain_generalizable_portrait_style_transfer.md)**

:   DGPST 提出了一个基于扩散模型的人像风格迁移框架，通过 semantic adapter 建立跨域稠密语义对应来扭曲参考图像，配合 AdaIN-Wavelet Transform 进行潜空间初始化以平衡风格化与内容保持，结合 ControlNet（高频结构引导）和 style adapter（风格引导）的双条件扩散模型生成最终结果，仅在 30K 真实肖像照片上训练即可泛化到照片、卡通、素描、动漫等多种域。

**[Dposer-X Diffusion Model As Robust 3D Whole-Body Human Pose Prior](image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)**

:   提出 DPoser-X，基于无条件扩散模型的 3D 全身人体姿态先验，将各种姿态相关任务统一为逆问题，通过变分扩散采样的截断时间步调度进行测试时优化，并引入混合训练策略有效结合全身和部位数据集，在身体、手、脸和全身建模的 8 个基准上取得最高 61% 的提升。

**[Dreamdance Animating Human Images By Enriching 3D Geometry Cues From 2D Poses](image_generation/dreamdance_animating_human_images_by_enriching_3d_geometry_cues_from_2d_poses.md)**

:   DreamDance 提出一种仅以 2D 骨架姿态序列为输入的人体图像动画框架：先通过 Mutually Aligned Geometry Diffusion Model 从 2D 姿态生成相互对齐的深度图和法线图以丰富 3D 几何引导，再通过基于 SVD 的 Cross-Domain Controlled Video Diffusion Model 整合多层次引导信号生成高质量人体动画，在 TikTok 数据集上取得 SOTA（FVD 153.07 vs Champ 170.20）。

**[Dual Recursive Feedback On Generation And Appearance Latents For Pose-Robust Tex](image_generation/dual_recursive_feedback_on_generation_and_appearance_latents_for_pose-robust_tex.md)**

:   提出 **Dual Recursive Feedback (DRF)**，一种无需训练的双递归反馈系统，通过**外观反馈**和**生成反馈**递归精修中间隐变量，解决可控 T2I 扩散模型在跨类别（class-invariant）场景下结构/外观分离不彻底的问题，实现细粒度的姿态迁移和外观融合。

**[Dynamicid Zero-Shot Multi-Id Image Personalization With Flexible Facial Editabil](image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)**

:   DynamicID 通过语义激活注意力（SAA）和身份-运动重构器（IMR）两个核心组件，实现了零样本的单/多身份个性化图像生成，同时保持高保真度和灵活的面部可编辑性。

**[Early Timestep Zero-Shot Candidate Selection For Instruction-Guided Image Editin](image_generation/early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)**

:   本文提出 ELECT（Early-timestep Latent Evaluation for Candidate selecTion），一个零样本框架，通过在扩散去噪的早期时间步估计背景不一致性来选择最优种子，将计算开销降低 41%（最高 61%），同时提升背景一致性和编辑指令遵循度，且无需外部监督或额外训练。

**[Ec-Flow Enabling Versatile Robotic Manipulation From Action-Unlabeled Videos Via](image_generation/ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)**

:   EC-Flow 提出了"具身中心光流"范式，从无动作标注的 RGB 视频中预测机器人本体的像素级运动轨迹，结合 URDF 运动学约束将视觉预测转化为可执行动作，在可变形物体、遮挡和非位移操作等场景中大幅超越物体中心方法。

**[Edit Efficient Diffusion Transformers With Linear Compressed Attention](image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)**

:   EDiT 提出线性压缩注意力机制，通过 ConvFusion 增强 query 的局部信息并用 Spatial Compressor 压缩 key/value token，实现对 DiT 和 MM-DiT 的高效加速（最高 2.2 倍），同时保持可比的图像质量。

**[Eedit Rethinking The Spatial And Temporal Redundancy For Efficient Image Editing](image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)**

:   提出 EEdit 高效图像编辑框架，通过空间局部性缓存（SLoC）跳过未编辑区域计算、Token 索引预处理（TIP）无损加速缓存操作、以及反演步跳过（ISS）减少反演冗余，在 prompt 引导、拖拽、图像合成等多种编辑任务上实现平均 2.46× 加速且无质量损失。

**[Efficient Autoregressive Shape Generation Via Octree-Based Adaptive Tokenization](image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)**

:   OAT 提出基于二次误差度量（quadric error）的自适应八叉树 tokenization，根据局部几何复杂度动态分配 token 预算，在减少 50% token 的同时保持重建质量，并在此基础上构建 OctreeGPT 实现高质量文本到 3D 生成。

**[Efficient Input-Level Backdoor Defense On Text-To-Image Synthesis Via Neuron Act](image_generation/efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)**

:   NaviT2I 发现了文生图扩散模型中后门触发器导致的"早期步骤激活变化"（Early-step Activation Variation）现象，基于此提出了一种仅需分析第一步扩散迭代的高效输入级后门防御框架，在 8 种主流攻击上平均 AUROC 达 96.3%，耗时仅为已有方法的 3.8%~16.7%。

**[Emoticrafter Text-To-Emotional-Image Generation Based On Valence-Arousal Model](image_generation/emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)**

:   提出 EmotiCrafter，首个基于连续 Valence-Arousal (V-A) 模型的情感图像生成方法，通过情感嵌入映射网络将 V-A 值融合到文本特征中，注入 Stable Diffusion XL 实现精确的内容+情感双重控制，生成图像在情感连续性和可控性上显著优于现有方法。

**[End-To-End Multi-Modal Diffusion Mamba](image_generation/end-to-end_multi-modal_diffusion_mamba.md)**

:   提出 Multi-Modal Diffusion Mamba（MDM），一种基于 Mamba 架构的端到端多模态模型，通过统一的 VAE 编解码器和多步选择性扩散模型，实现图像和文本的同时生成，计算复杂度为 $\mathcal{O}(MLN^2)$，在图像生成、图像描述、VQA 等多任务上超越现有端到端模型。

**[Enhancing Reward Models For High-Quality Image Generation Beyond Text-Image Alig](image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)**

:   本文揭示了基于 CLIP/BLIP 的奖励模型在评估高质量图像时的「评分悖论」——细节丰富的高质量图像反而得低分，并提出 ICT Score（Image-Contained-Text，评估图像包含文本信息的程度）和 HP Score（纯图像模态的人类偏好评分）两个新指标，在 Pick-High 数据集上训练后，偏好预测准确率提升超过 10%，并成功优化 SD3.5-Turbo 生成更高质量的图像。

**[Erasing More Than Intended How Concept Erasure Degrades The Generation Of Non-Ta](image_generation/erasing_more_than_intended_how_concept_erasure_degrades_the_generation_of_non-ta.md)**

:   系统分析了文本到图像模型中概念擦除技术对非目标概念的意外负面影响（溢出退化），提出EraseBench基准测试框架覆盖视觉相似、二项关联、语义关联等多维度，揭示当前SOTA擦除方法在保留非目标概念的生成质量方面仍不可靠。

**[Exploring Multimodal Diffusion Transformers For Enhanced Prompt-Based Image Edit](image_generation/exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)**

:   系统分析了多模态扩散Transformer（MM-DiT）的注意力机制，将注意力矩阵分解为四个功能性子块（I2I/T2I/I2T/T2T），并基于分析结果提出了一种高效的、通过替换图像输入投影（$\mathbf{q}_i, \mathbf{k}_i$）实现的prompt-based图像编辑方法，适用于SD3系列和Flux.1等多种MM-DiT变体。

**[Facecraft4D Animated 3D Facial Avatar Generation From A Single Image](image_generation/facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)**

:   本文提出 FaceCraft4D 框架，通过组合 3D形状先验（PanoHead GAN反演）、2D图像先验（扩散模型增强纹理）和视频先验（LivePortrait 生成表情动画），从单张图像生成可动画的360度4D面部头像，并提出 COIN 训练策略解决多视角数据不一致问题，实现高质量实时渲染（156 FPS）。

**[Fair Generation Without Unfair Distortions Debiasing Text-To-Image Generation Wi](image_generation/fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)**

:   提出 Entanglement-Free Attention（EFA），一种推理时应用的去偏见方法，通过修改跨注意力机制将目标属性（如性别、种族）注入人物区域，同时保持非目标属性（如背景、物品）不变，在消除生成偏见的同时避免引入新的不公平关联。

**[Feddifrc Unlocking The Potential Of Text-To-Image Diffusion Models In Heterogene](image_generation/feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)**

:   首次将预训练文本到图像扩散模型（Stable Diffusion）的内部表示引入联邦学习，提出 FedDifRC 框架，通过文本驱动的扩散对比学习（TDCL）和噪声驱动的扩散一致性正则化（NDCR）两个互补模块，有效缓解数据异质性问题，在多种 non-iid 场景下显著提升全局模型性能。

**[Fewer Denoising Steps Or Cheaper Per-Step Inference Towards Compute-Optimal Diff](image_generation/fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)**

:   本文提出 PostDiff——一个无需训练的扩散模型加速框架，在输入层面通过混合分辨率去噪策略（早期低分辨率→后期高分辨率）和模块层面通过混合缓存策略（DeepCache + 交叉注意力缓存）减少冗余，系统性地回答了"减少去噪步数 vs 降低每步计算成本哪个更有效"这一关键问题——答案是后者在大多数效率范围内更优。

**[Ficgen Frequency-Inspired Contextual Disentanglement For Layout-Driven Degraded ](image_generation/ficgen_frequency-inspired_contextual_disentanglement_for_layout-driven_degraded_.md)**

:   提出 FICGen，首次解决退化场景（低光照/水下/遥感/恶劣天气等）Layout-to-Image 生成中的"上下文幻觉困境"，通过可学习双查询机制提取退化场景的高低频原型，经视觉-频率增强注意力注入 latent 扩散空间，并使用实例一致性图 + 空间-频率自适应聚合实现前景-背景解耦，在 5 个退化数据集上全面超越现有 L2I 方法。

**[Fix-Clip Dual-Branch Hierarchical Contrastive Learning Via Synthetic Captions Fo](image_generation/fix-clip_dual-branch_hierarchical_contrastive_learning_via_synthetic_captions_fo.md)**

:   Fix-CLIP 通过三大创新模块提升 CLIP 的长文本理解能力：（1）双分支训练管线用短文本配合 masked 图像、长文本配合原始图像分别对齐；（2）带单向掩码的可学习区域提示（Regional Prompts）提取局部视觉特征；（3）层级特征对齐模块对齐中间层多尺度特征。在 30M 合成长文本数据上增量训练后，长文本检索和短文本检索均大幅超越 SOTA，文本编码器可即插即用提升扩散模型长文本生成质量。

**[Float Generative Motion Latent Flow Matching For Audio-Driven Talking Portrait](image_generation/float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)**

:   提出 FLOAT，基于流匹配（Flow Matching）的音频驱动说话肖像生成方法，在正交运动潜空间中用 Transformer 架构预测向量场，实现高效（~10 步采样）、时序一致的高质量说话视频生成，并支持语音驱动的情绪增强和测试时头部姿态编辑。

**[Flowdps Flow-Driven Posterior Sampling For Inverse Problems](image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)**

:   FlowDPS 通过推导 Flow 模型的 Tweedie 公式将 Flow ODE 分解为干净图像估计和噪声估计两个分量，在干净图像分量中注入似然梯度、在噪声分量中引入随机噪声，实现了基于 Flow 模型的后验采样逆问题求解，在 SD3.0 上的四种线性逆问题中超越所有已有方法。

**[Flowedit Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)**

:   FlowEdit 提出一种无需反转（inversion-free）、无需优化、模型无关的文本编辑方法，直接在预训练 Flow 模型的源/目标分布之间构建 ODE 路径，实现比 inversion 更低传输代价的结构保持编辑。

**[Flowtok Flowing Seamlessly Across Text And Image Tokens](image_generation/flowtok_flowing_seamlessly_across_text_and_image_tokens.md)**

:   FlowTok 提出将文本和图像都编码为紧凑的 1D token 表示（77×16），通过 flow matching 直接在文本与图像 token 之间进行流动转换，无需复杂的条件机制或噪声调度，实现了高效的跨模态生成。

**[Free4D Tuning-Free 4D Scene Generation With Spatial-Temporal Consistency](image_generation/free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)**

:   提出 Free4D，首个无需微调的单图像 4D 场景生成框架，通过 4D 几何结构初始化、自适应引导去噪保证空间一致性、参考潜变量替换保证时序一致性、基于调制的精化融合多视角信息为一致的 4D 高斯表示，实现实时可控渲染。

**[Freemorph Tuning-Free Generalized Image Morphing With Diffusion Model](image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)**

:   FreeMorph 提出首个无需微调的通用图像变形方法，通过引导感知球面插值和步骤导向变化趋势两个创新设计，实现了 30 秒内在任意语义/布局的图像对之间生成平滑过渡序列，速度比现有方法快 10-50 倍。

**[Freescale Unleashing The Resolution Of Diffusion Models Via Tuning-Free Scale Fu](image_generation/freescale_unleashing_the_resolution_of_diffusion_models_via_tuning-free_scale_fu.md)**

:   提出 FreeScale，一种无需训练的推理范式，通过尺度融合（Scale Fusion）机制从不同感受野尺度提取并融合信息（全局高频 + 局部低频），配合定制化级联上采样和约束膨胀卷积，首次在单张 A800 GPU 上实现了 8K 分辨率的文本到图像生成，同时支持高分辨率视频生成。

**[From Reusing To Forecasting Accelerating Diffusion Models With Taylorseers](image_generation/from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)**

:   提出 TaylorSeer，将扩散模型特征缓存范式从"缓存-重用"升级为"缓存-预测"——利用 Taylor 级数展开基于历史特征的高阶有限差分来预测未来时间步的中间特征，在 FLUX 上实现近乎无损的 4.99× 加速、在 HunyuanVideo 上实现 5.00× 加速，且完全无需额外训练。

**[Gamefactory Creating New Games With Generative Interactive Videos](image_generation/gamefactory_creating_new_games_with_generative_interactive_videos.md)**

:   提出 GameFactory，通过在预训练视频扩散模型上**解耦游戏风格与动作控制**的多阶段训练策略，实现了从小规模 Minecraft 数据学到的动作控制能力**泛化到开放域任意场景**的交互式游戏视频生成——这是首个提供完整技术论文且验证复杂动作空间（7键+鼠标）的场景泛化方法。

**[Generating Multi-Image Synthetic Data For Text-To-Image Customization](image_generation/generating_multi-image_synthetic_data_for_text-to-image_customization.md)**

:   提出 SynCD（合成定制数据集）及其生成管线，利用共享注意力和 3D 资产先验合成多图一致性对象数据集，训练的编码器模型在无需测试时优化的情况下超越现有编码器方法。

**[Generative Modeling Of Shape-Dependent Self-Contact Human Poses](image_generation/generative_modeling_of_shape-dependent_self-contact_human_poses.md)**

:   构建首个大规模精确形状标注的自接触姿态数据集Goliath-SC（383K姿态/130个subject），提出形状条件的部件感知潜在扩散模型PAPoseDiff来建模体型依赖的自接触姿态分布，并利用学到的扩散先验进行单视角姿态refinement，在unseen subject上超越BUDDI和SMPLer-X等SOTA方法。

**[Genflowrl Shaping Rewards With Generative Object-Centric Flow In Visual Reinforc](image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)**

:   提出 GenFlowRL，通过从跨具身数据集训练的流生成模型中提取的 δ-flow 表示进行奖励塑形，将生成式物体中心光流与强化学习结合，实现了鲁棒且可泛化的机器人操控策略学习，在 10 个操控任务上显著优于流式模仿学习和视频引导 RL 方法。

**[Genhancer Imperfect Generative Models Are Secretly Strong Vision-Centric Enhance](image_generation/genhancer_imperfect_generative_models_are_secretly_strong_vision-centric_enhance.md)**

:   发现"完美的图像重建并不总带来最佳视觉表征"，提出 GenHancer——一种仅用轻量级随机初始化去噪器（约预训练重型去噪器 1/10 参数）和全局 [CLS] token 条件的两阶段后训练方法，通过自监督重建任务增强 CLIP 的细粒度视觉感知能力，在 MMVP-VLM 上比 DIVA 提升 6.0%。

**[Golden Noise For Diffusion Models A Learning Framework](image_generation/golden_noise_for_diffusion_models_a_learning_framework.md)**

:   本文提出"噪声提示"（Noise Prompt）概念，设计了一个轻量级噪声提示网络（NPNet），通过 Re-denoise Sampling 收集 10 万对噪声数据训练 NPNet，将随机高斯噪声转化为承载语义信息的"黄金噪声"，作为即插即用模块提升 SDXL 等多种扩散模型的生成质量，仅增加 3% 推理时间。

**[Grouped Speculative Decoding For Autoregressive Image Generation](image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)**

:   提出 Grouped Speculative Decoding (GSD)，一种免训练的自回归图像生成加速方法，通过在语义有效的 token 簇级别（而非单一最可能 token）进行推测验证，平均实现 3.7× 加速且不损失图像质量。

**[Guiding Noisy Label Conditional Diffusion Models With Score-Based Discriminator ](image_generation/guiding_noisy_label_conditional_diffusion_models_with_score-based_discriminator_.md)**

:   提出Score-based Discriminator Correction (SBDC)，通过训练一个轻量判别器在推理时校正噪声标签条件扩散模型的生成轨迹，利用噪声检测将训练集分为干净/腐败子集来训练判别器，并发现仅在采样过程的早中期阶段施加引导即可获得最优效果。

**[Holistic Tokenizer For Autoregressive Image Generation](image_generation/holistic_tokenizer_for_autoregressive_image_generation.md)**

:   提出 Hita，一种全局-局部（holistic-to-local）图像 tokenizer，通过可学习全局查询捕获纹理/材质/形状等全局属性，结合双码本量化和因果注意力融合模块，在不修改 AR 模型架构的前提下，将 ImageNet 256×256 生成 FID 降至 2.59、训练收敛加速 2.1 倍，并支持零样本风格迁移和图像补全。

**[Holistic Unlearning Benchmark A Multi-Faceted Evaluation For Text-To-Image Diffu](image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)**

:   HUB 提出了首个全面评估文生图扩散模型概念遗忘（concept unlearning）方法的基准框架，覆盖 33 个目标概念和 6 大评估维度（忠实度、对齐性、精确性、多语言鲁棒性、对抗鲁棒性、效率），每个概念使用 16,000 条 prompt，发现没有任何单一方法能在所有维度上占优。

**[Hpsv3 Towards Wide-Spectrum Human Preference Score](image_generation/hpsv3_towards_wide-spectrum_human_preference_score.md)**

:   HPSv3 构建了首个宽谱人类偏好数据集 HPDv3（1.08M 图文对、1.17M 标注对），采用 VLM 骨干（Qwen2-VL）+ 不确定性感知排序损失训练偏好模型，并提出 CoHP 链式思维迭代生成方法，显著提升图像生成评估的准确性和覆盖范围。

**[Hypdae Hyperbolic Diffusion Autoencoders For Hierarchical Few-Shot Image Generat](image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)**

:   将双曲空间的层级表示学习能力与扩散自编码器的高质量生成能力结合，通过在 Poincaré 圆盘中操控潜码的半径和方向，实现可控、多样且保持类别一致性的小样本图像生成。

**[Illume Illuminating Your Llms To See Draw And Self-Enhance](image_generation/illume_illuminating_your_llms_to_see_draw_and_self-enhance.md)**

:   提出 ILLUME，一个通过统一的下一 token 预测范式将多模态理解和生成能力整合进单个 LLM 的统一 MLLM。通过**语义视觉分词器**（减少4倍预训练数据量至15M）和**自增强多模态对齐方案**（让模型自评自生成图像与文本的一致性），在多种理解、生成和编辑任务上达到了State-of-the-art统一模型的竞争力甚至超越。

**[Imagegem In-The-Wild Generative Image Interaction Dataset For Generative Model P](image_generation/imagegem_in-the-wild_generative_image_interaction_dataset_for_generative_model_p.md)**

:   提出 **ImageGem**，首个大规模真实用户生成式交互数据集（57K用户 × 242K定制LoRA × 3M文本提示 × 5M生成图像），利用个体用户偏好标注实现三大应用：**聚合偏好对齐**超越 Pick-a-Pic、**个性化检索与生成式推荐**（VLM排序显著提升）、以及首次提出的**生成模型个性化**——在 LoRA 潜权重空间（W2W）中学习偏好编辑方向以定制扩散模型。

**[Improved Noise Schedule For Diffusion Training](image_generation/improved_noise_schedule_for_diffusion_training.md)**

:   提出从概率分布视角统一分析和设计扩散模型噪声调度的框架，发现将采样概率集中在 $\log\text{SNR}=0$ 附近（信号与噪声临界点）的 Laplace 噪声调度，在相同训练预算下比标准 cosine 调度 FID 提升 26.6%，且优于所有损失权重调整方法。

**[Inference-Time Diffusion Model Distillation](image_generation/inference-time_diffusion_model_distillation.md)**

:   提出 Distillation++，一种推理时扩散蒸馏框架，在采样过程中利用预训练教师模型的引导来修正学生蒸馏模型的去噪路径，无需额外训练数据或微调即可显著缩小师生模型间的性能差距。

**[Infgen A Resolution-Agnostic Paradigm For Scalable Image Synthesis](image_generation/infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)**

:   提出InfGen，一种"第二代生成"范式，用一个基于Transformer的生成器替换VAE解码器，从固定大小的latent一步解码出任意分辨率图像，无需修改或重新训练扩散模型，将4K图像生成时间压缩至10秒以内，比现有最快方法UltraPixel提速10倍以上。

**[Infinidreamer Arbitrarily Long Human Motion Generation Via Segment Score Distill](image_generation/infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)**

:   InfiniDreamer 通过将预训练的短序列运动扩散模型作为先验，提出 Segment Score Distillation (SSD) 优化方法，对粗初始化的长运动序列中的重叠短片段进行迭代优化，实现了无需额外长序列训练数据的任意长度人体运动生成。

**[Inpaint4Drag Repurposing Inpainting Models For Drag-Based Image Editing Via Bidi](image_generation/inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)**

:   提出Inpaint4Drag，将拖拽式图像编辑分解为像素空间双向warp和图像修复两个阶段，受弹性物体变形启发设计双向warping算法实现实时预览（0.01s）和高效生成（0.3s），比现有方法快600倍，且可作为任意修复模型的通用适配器。

**[Introstyle Training-Free Introspective Style Attribution Using Diffusion Feature](image_generation/introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)**

:   提出 IntroStyle，一种无需训练的风格归因方法，仅利用扩散模型自身中间层特征的通道级均值和方差统计量，通过 2-Wasserstein 距离度量图像间的风格相似性，在 WikiArt 和 DomainNet 上大幅超越需要专门训练的 SOTA 方法。

**[Invisible Watermarks Visible Gains Steering Machine Unlearning With Bi-Level Wat](image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)**

:   提出 Water4MU，通过双层优化（BLO）框架将数字水印机制与机器遗忘（MU）相结合，在上层优化水印网络使其有利于遗忘，在下层执行遗忘优化，从而在不显著损害模型效用的前提下显著提升遗忘效果。

**[Irgpt Understanding Real-World Infrared Image With Bi-Cross-Modal Curriculum On ](image_generation/irgpt_understanding_real-world_infrared_image_with_bi-cross-modal_curriculum_on_.md)**

:   提出 IRGPT，首个基于真实红外图像的多模态大语言模型，构建了包含 260K+ 图像-文本对的大规模红外-文本数据集 IR-TD，并设计了双跨模态课程迁移学习策略（Bi-cross-modal Curriculum），在 9 个红外任务基准上取得 SOTA 性能，零样本 psum 比基线 InternVL2-8B 提升 76.35。

**[Joint Diffusion Models In Continual Learning](image_generation/joint_diffusion_models_in_continual_learning.md)**

:   > 提出 JDCL，将分类器与扩散生成模型统一为一个联合参数化的网络，结合知识蒸馏和两阶段训练策略，在生成重放式持续学习中大幅缓解灾难性遗忘，超越现有生成重放方法。

**[Larender Training-Free Occlusion Control In Image Generation Via Latent Renderin](image_generation/larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)**

:   提出 LaRender，一种基于体渲染原理的免训练图像生成方法，通过在潜空间中对物体特征进行"渲染"来精确控制图像中物体之间的遮挡关系。该方法仅替换预训练扩散模型的交叉注意力层，不引入任何可学习参数，在遮挡精度上显著超越现有 SOTA 方法，且能实现语义透明度控制等丰富效果。

**[Latent Diffusion Models With Masked Autoencoders](image_generation/latent_diffusion_models_with_masked_autoencoders.md)**

:   系统性地分析了 LDM 中自编码器应具备的三个关键属性（潜空间平滑性、感知压缩质量、重建质量），发现现有自编码器无法同时满足，提出 Variational Masked AutoEncoders (VMAEs)，结合 MAE 的层次化特征和 VAE 的概率编码，在仅 13.4% 参数和 4.1% GFLOPs 的条件下显著提升生成质量（ImageNet-1K gFID: 5.98 vs SD-VAE 的 6.49）。

**[Lay-Your-Scene Natural Scene Layout Generation With Diffusion Transformers](image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)**

:   提出 LayouSyn，基于轻量开源语言模型提取场景元素、结合宽高比感知扩散 Transformer 的开放词汇文本到布局生成流水线，在空间推理和数量推理基准上达到 SOTA。

**[Lazymar Accelerating Masked Autoregressive Models Via Feature Caching](image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)**

:   LazyMAR针对Masked Autoregressive（MAR）模型的推理效率瓶颈，利用两种冗余——token冗余（相邻解码步中大部分token特征高度相似）和条件冗余（classifier-free guidance中条件/无条件输出的残差在相邻步间变化极小），设计了token cache和condition cache两种缓存机制，实现2.83×加速且几乎不损失生成质量。

**[Ld-Rps Zero-Shot Unified Image Restoration Via Latent Diffusion Recurrent Poster](image_generation/ld-rps_zero-shot_unified_image_restoration_via_latent_diffusion_recurrent_poster.md)**

:   LD-RPS 提出一种零样本、无数据集的统一图像复原方法，利用预训练潜在扩散模型进行循环后验采样，通过多模态大模型提供语义先验、可学习 F-PAM 模块对齐退化域，实现多种退化类型的高质量盲复原。

**[Learning Few-Step Diffusion Models By Trajectory Distribution Matching](image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)**

:   提出 Trajectory Distribution Matching（TDM），一种统一轨迹蒸馏和分布匹配的新范式，在分布层面对齐学生与教师的 ODE 轨迹，实现高效的少步扩散模型蒸馏，仅需 2 A800 小时即可将 PixArt-α 蒸馏为超越教师的 4 步生成器。

**[Learning To See In The Extremely Dark](image_generation/learning_to_see_in_the_extremely_dark.md)**

:   提出配对到配对的数据合成管线构建极暗场景（低至0.0001 lux）RAW图像增强数据集SIED，并设计基于扩散模型的框架，通过自适应光照校正模块（AICM）和颜色一致性损失实现极低信噪比RAW图像的高质量恢复。

**[Less-To-More Generalization Unlocking More Controllability By In-Context Generat](image_generation/less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)**

:   本文提出 UNO，一种基于 DiT 的通用定制化生成模型，通过"模型-数据共进化"范式——利用较弱模型生成的合成数据逐步训练更强模型——结合渐进式跨模态对齐和 Universal RoPE，实现了单主体和多主体驱动图像生成的 SOTA 性能（DreamBench DINO 0.760, CLIP-I 0.835）。

**[Less Is More Improving Motion Diffusion Models With Sparse Keyframes](image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)**

:   提出 sMDM，一种以稀疏关键帧为核心的运动扩散框架，通过 masking-interpolation 策略和 Visvalingam-Whyatt 关键帧选择算法，减少冗余帧处理，在文本对齐和运动质量上持续超越密集帧基线。

**[Lift Latent Implicit Functions For Task- And Data-Agnostic Encoding](image_generation/lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)**

:   LIFT 提出了一个基于元学习的多尺度隐式神经表示框架，通过并行局部隐式函数和层次化潜变量生成器，实现跨任务（生成、分类）和跨数据模态（2D 图像、3D 体素）的统一编码，在重建和生成任务上均达到 SOTA 且计算成本大幅降低。

**[Lit Delving Into A Simple Linear Diffusion Transformer For Image Generation](image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)**

:   > 系统研究如何将预训练 DiT 安全高效地转换为线性注意力版本 LiT，提出 5 条实践指南（深度卷积增强、少头策略、权重继承、选择性加载、混合蒸馏），仅需 DiT 训练步数的 20% 即可达到可比性能。

**[Long-Context State-Space Video World Models](image_generation/long-context_state-space_video_world_models.md)**

:   本文提出将状态空间模型（SSM/Mamba）引入视频世界模型，通过 block-wise SSM 扫描方案在空间一致性和时序记忆之间权衡，配合局部帧注意力，实现了线性训练复杂度、常数推理开销下的长期空间记忆保持，在 Memory Maze 和 Minecraft 数据集上大幅超越有限上下文的 Transformer 基线。

**[Looking In The Mirror A Faithful Counterfactual Explanation Method For Interpret](image_generation/looking_in_the_mirror_a_faithful_counterfactual_explanation_method_for_interpret.md)**

:   将分类器的决策边界视为"镜面"，通过将特征表示"反射"到镜面另一侧生成反事实解释（CFE），并设计三角测量损失保持潜在空间到图像空间的距离关系，实现忠实、可控且可动画化的反事实解释。

**[Loraverse A Submodular Framework To Retrieve Diverse Adapters For Diffusion Mode](image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)**

:   将从100K+ LoRA适配器库中检索相关且多样化的LoRA组合建模为组合优化问题，提出基于子模函数最大化的LoRAverse框架，通过概念提取+子模检索实现兼顾相关性和多样性的LoRA选择。

**[M2Sformer Multi-Spectral And Multi-Scale Attention With Edge-Aware Difficulty Gu](image_generation/m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)**

:   提出 M2SFormer，在编码器-解码器的 skip connection 中统一多光谱（2D DCT 频域）和多尺度（SIFT 风格空间金字塔）注意力机制，并设计基于边缘感知曲率的难度引导注意力解码器，在图像篡改定位任务中实现跨域泛化性能 SOTA（CASIAv2 训练方案下 unseen 域平均 DSC 43.0%，mIoU 34.3%）。

**[Make Me Happier Evoking Emotions Through Image Diffusion Models](image_generation/make_me_happier_evoking_emotions_through_image_diffusion_models.md)**

:   EmoEditor 提出首个系统性的**情感驱动图像生成**框架，通过双分支扩散模型（全局情感条件 + 局部语义特征）实现仅输入源图和目标情感即可生成具有目标情感的图像，无需手工文本指令或参考图，并构建了 340K 情感标注图对的 EmoPair 数据集。

**[Mamtiff-Cad Multi-Scale Latent Diffusion With Mamba For Complex Parametric Seque](image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)**

:   提出MamTiff-CAD框架，结合Mamba+编码器与Transformer解码器的自编码器学习CAD命令序列的潜表示，再用多尺度Transformer扩散模型生成，首次实现60-256命令长度的复杂CAD模型生成。

**[Maskcontrol Spatio-Temporal Control For Masked Motion Synthesis](image_generation/maskcontrol_spatio-temporal_control_for_masked_motion_synthesis.md)**

:   MaskControl 首次将空间可控性引入生成式掩码运动模型，通过 Logits Regularizer（训练时隐式对齐）和 Logits Optimization（推理时显式优化）两个核心组件操控 token 分类器的 logits，同时实现高质量运动生成（FID 降低 77%）和高精度关节控制（平均误差 0.91cm vs 1.08cm）。

**[Matchdiffusion Training-Free Generation Of Match-Cuts](image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)**

:   提出MatchDiffusion，利用扩散模型早期去噪步骤定义场景宏观结构、后期步骤添加细节的特性，通过Joint Diffusion和Disjoint Diffusion两阶段无训练方法实现自动match-cut视频生成。

**[Mavflow Preserving Paralinguistic Elements With Conditional Flow Matching For Ze](image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)**

:   提出 MAVFlow，基于条件流匹配（CFM）的零样本音视觉渲染器，通过音频说话人嵌入和视觉情感嵌入的双模态引导，在多语言 AV2AV 翻译中保持说话人一致性。

**[Meta-Unlearning On Diffusion Models Preventing Relearning Unlearned Concepts](image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)**

:   本文提出了扩散模型的元遗忘（Meta-Unlearning）框架，在标准遗忘目标之外增加一个元目标，使得模型在被恶意微调时，与遗忘概念相关的良性知识会自毁，从而阻止已遗忘概念的重新学习，该框架兼容大多数现有遗忘方法且仅需添加一个简单的元目标。

**[Mind The Gap Aligning Vision Foundation Models To Image Feature Matching](image_generation/mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)**

:   本文发现视觉基础模型（如 DINOv2）在图像特征匹配中存在"对齐偏差"——基于对比学习的模型丢失了实例级细节且缺乏跨图像交互机制，导致多实例场景匹配失败。为此提出 IMD 框架，利用扩散模型作为特征提取器保留实例级细节，并设计跨图像交互提示模块（CIPM）实现双向信息交互，在标准基准和新提出的多实例基准 IMIM 上均达到 SOTA，多实例场景提升 12%。

**[Mmaif Multi-Task And Multi-Degradation All-In-One For Image Fusion With Language](image_generation/mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)**

:   MMAIF 提出统一的多任务、多退化、语言引导图像融合框架，通过实际退化流水线和现代化 DiT 架构在潜在空间操作，同时提供回归和 Flow Matching 两个版本，在各类退化融合任务上超越现有 restoration+fusion 流水线。

**[Mofrr Mixture Of Diffusion Models For Face Retouching Restoration](image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)**

:   本文首次提出人脸修图还原(FRR)任务，并设计 MoFRR 框架——借鉴 DeepSeek MoE 思想，通过路由器激活特定修图类型的专家（小波 DDIM）和共享专家（通用 DDIM），在新构建的百万级 RetouchingFFHQ++ 数据集上实现了修图人脸的近真实还原。

**[Mosaicdiff Training-Free Structural Pruning For Diffusion Model Acceleration Ref](image_generation/mosaicdiff_training-free_structural_pruning_for_diffusion_model_acceleration_ref.md)**

:   本文提出 MosaicDiff，一种免训练的扩散模型结构化剪枝方法，通过将推理过程按预训练学习速度动态分为三个阶段并对各阶段应用不同稀疏度的子网络，实现了在 DiT 和 SDXL 上的显著加速而不牺牲生成质量。

**[Motiondiff Training-Free Zero-Shot Interactive Motion Editing Via Flow-Assisted ](image_generation/motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)**

:   MotionDiff 提出一种免训练、零样本的多视图运动编辑方法，通过点运动学模型（PKM）从静态场景估计多视图光流，再利用解耦运动表示引导 Stable Diffusion 生成高质量、多视图一致的运动编辑结果。

**[Motionstreamer Streaming Motion Generation Via Diffusion-Based Autoregressive Mo](image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)**

:   提出 MotionStreamer，将连续因果潜空间与扩散头结合到自回归框架中，实现文本条件下的流式人体动作生成，支持在线多轮生成和动态运动组合。

**[Multi-Turn Consistent Image Editing](image_generation/multi-turn_consistent_image_editing.md)**

:   提出基于 flow matching 的多轮图像编辑框架，通过双目标 LQR 引导和自适应注意力机制，有效抑制多轮编辑中的误差累积，在保持内容一致性的同时实现灵活可控的迭代编辑。

**[Multimodal Latent Diffusion Model For Complex Sewing Pattern Generation](image_generation/multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)**

:   提出 SewingLDM，一个多模态条件潜空间扩散模型，通过扩展缝纫版型表示和两阶段训练策略，实现在文本、草图、体型条件控制下合复杂缝纫版型，并可无缝集成到 CG 仿真管线。

**[Munba Machine Unlearning Via Nash Bargaining](image_generation/munba_machine_unlearning_via_nash_bargaining.md)**

:   将机器遗忘（Machine Unlearning）建模为双玩家合作博弈问题，利用 Nash 讨价还价理论推导闭式解来同时解决遗忘目标与保留目标之间的梯度冲突和梯度支配问题，在分类和生成任务上实现遗忘与保留的最优平衡。

**[Music-Aligned Holistic 3D Dance Generation Via Hierarchical Motion Modeling](image_generation/music-aligned_holistic_3d_dance_generation_via_hierarchical_motion_modeling.md)**

:   提出 SoulDance 数据集（首个含身体+手部+面部的高质量3D舞蹈数据集）和 SoulNet 框架（层次化残差向量量化 + 音乐对齐生成模型 + 跨模态检索），实现首个面部表情与身体手部动作协调一致、与音乐节奏情感对齐的全身3D舞蹈生成。

**[Nuiscene Exploring Efficient Generation Of Unbounded Outdoor Scenes](image_generation/nuiscene_exploring_efficient_generation_of_unbounded_outdoor_scenes.md)**

:   NuiScene 提出使用向量集（vector set）编码场景块的高效方法，配合显式 outpainting 扩散模型实现快速无界户外场景生成，并策划了 NuiScene43 高质量户外场景数据集。

**[Nullswap Proactive Identity Cloaking Against Deepfake Face Swapping](image_generation/nullswap_proactive_identity_cloaking_against_deepfake_face_swapping.md)**

:   提出 NullSwap，通过在源图像中嵌入身份引导的不可见扰动来伪装面部身份信息，使 Deepfake 换脸模型无法提取正确身份，从而在纯黑盒场景下主动防御换脸攻击。

**[Omegance A Single Parameter For Various Granularities In Diffusion-Based Synthes](image_generation/omegance_a_single_parameter_for_various_granularities_in_diffusion-based_synthes.md)**

:   Omegance 提出仅通过一个参数 $\omega$ 缩放扩散模型去噪步骤中的噪声预测，即可无需重训练地实现对生成图像/视频细节粒度的全局、空间和时序精细控制，方法与架构无关且兼容 SDXL、SD3、FLUX 等多种模型。

**[Ominicontrol Minimal And Universal Control For Diffusion Transformer](image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)**

:   提出OminiControl，仅需0.1%额外参数即可在DiT架构上实现空间对齐和非对齐两类图像控制任务的统一处理，核心创新包括统一序列处理、动态位置编码和注意力偏置控制机制。

**[Omnipaint Mastering Object-Oriented Editing Via Disentangled Insertion-Removal I](image_generation/omnipaint_mastering_object-oriented_editing_via_disentangled_insertion-removal_i.md)**

:   提出 OmniPaint 统一框架，将物体移除与插入重新定义为互逆互补的关联任务，基于 FLUX 扩散先验并引入 CycleFlow 无配对训练机制和 CFD 无参考评估指标，仅用 3K 真实配对样本即可实现高保真的物体编辑，尤其擅长处理阴影、反射等复杂物理效果。

**[Omnivton Training-Free Universal Virtual Try-On](image_generation/omnivton_training-free_universal_virtual_try-on.md)**

:   OmniVTON 提出首个无需训练的通用虚拟试穿框架，通过解耦服装纹理与姿态条件，利用结构化服装变形、连续边界缝合和频谱姿态注入三大模块，在 in-shop 和 in-the-wild 场景中均实现高保真试穿，并首次支持多人试穿。

**[Ouroboros Single-Step Diffusion Models For Cycle-Consistent Forward And Inverse ](image_generation/ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)**

:   本文提出 Ouroboros，一个由两个单步扩散模型（分别负责逆渲染 RGB→X 和前向渲染 X→RGB）组成的统一框架，通过循环一致性训练确保双向渲染的一致性，在多个数据集上取得 SOTA 的同时推理速度比多步扩散方法快 50 倍，并可零样本迁移到视频分解。

**[Panollama Generating Endless And Coherent Panoramas With Next-Token-Prediction L](image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)**

:   提出 PanoLlama，通过 token 重定向策略将固定尺寸的视觉自回归（VAR）模型扩展为无限全景生成，实现免训练的 next-crop prediction，在连贯性、保真度和美学上超越联合扩散等方法。

**[Patchscaler An Efficient Patch-Independent Diffusion Model For Image Super-Resol](image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)**

:   本文提出 PatchScaler，一种 Patch 级独立扩散超分管线，通过全局修复模块生成置信度图量化各区域重建难度，并将 Patch 分组为简单/中等/困难三组分配不同采样步数，搭配纹理提示检索机制，在 RealSR 上仅 0.23× ResShift 运行时间达到更优质量。

**[Penalizing Boundary Activation For Object Completeness In Diffusion Models](image_generation/penalizing_boundary_activation_for_object_completeness_in_diffusion_models.md)**

:   本文深入分析了扩散模型生成不完整物体的根本原因——训练中使用的 RandomCrop 数据增强，并提出一种训练免费的边界激活惩罚方法，通过在早期去噪步骤中利用交叉注意力和自注意力约束抑制物体在图像边缘生成，将 SDv2.1 的物体不完整率从 45.7% 降至 17.3%。

**[Personalvideo High Id-Fidelity Video Customization Without Dynamic And Semantic ](image_generation/personalvideo_high_id-fidelity_video_customization_without_dynamic_and_semantic_.md)**

:   本文提出 PersonalVideo 框架，通过混合奖励监督（身份一致性奖励+语义一致性奖励）直接对生成视频施加反馈，消除了传统方法中 T2I 调优与 T2V 推理之间的分布差距，在保持高身份保真度的同时避免了运动动态和语义跟随的退化。

**[Pino Person-Interaction Noise Optimization For Long-Duration And Customizable Mo](image_generation/pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)**

:   提出Person-Interaction Noise Optimization（PINO），一种无需训练的框架，将复杂的多人群体交互分解为语义明确的两人交互对，利用预训练的两人交互扩散模型通过噪声优化和物理惩罚项顺序合成任意规模的群体交互运动，支持精细化用户控制和长时序运动生成。

**[Pla Prompt Learning Attack Against Text-To-Image Generative Models](image_generation/pla_prompt_learning_attack_against_text-to-image_generative_models.md)**

:   本文提出 PLA（Prompt Learning Attack），一种针对黑盒 T2I 模型的梯度驱动对抗攻击框架，通过敏感知识编码和多模态相似度损失来学习对抗性 prompt，从而绕过 prompt 过滤器和后置安全检查器，平均 ASR-4 达 90%+，远超现有方法。

**[Pretrained Reversible Generation As Unsupervised Visual Representation Learning](image_generation/pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)**

:   PRG 通过**反转预训练连续生成模型**（扩散/流模型）的生成过程来提取无监督视觉表示，实现模型无关的判别任务适配，在 ImageNet 64×64 上达到 78% top-1 准确率，为基于生成模型的方法中 SOTA。

**[Randomized Autoregressive Visual Generation](image_generation/randomized_autoregressive_visual_generation.md)**

:   提出 Randomized AutoRegressive modeling (RAR)：在标准自回归训练中以随机排列输入序列并逐步退火回光栅扫描顺序，使模型学习双向上下文，在 ImageNet-256 上以 FID 1.48 刷新自回归图像生成 SOTA，同时保持与语言模型框架的完全兼容。

**[Reducio Generating 1K Video Within 16 Seconds Using Extremely Compressed Motion ](image_generation/reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)**

:   提出 Reducio-VAE，一种以内容帧为条件的 3D 视频自编码器，将视频压缩至比标准 2D VAE 小 64 倍的运动潜空间，配合 Reducio-DiT 在单张 A100 上 15.5 秒内生成 16 帧 1024x1024 视频，训练仅需 3200 A100 GPU 小时。

**[Reflex Text-Guided Editing Of Real Images In Rectified Flow Via Mid-Step Feature](image_generation/reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)**

:   针对 Rectified Flow（ReFlow）模型的真实图像编辑难题，通过系统分析 MM-DiT 的中间表示，识别出三个关键特征（I2I-SA、I2T-CA、残差特征），并提出中间步特征提取（mid-step feature extraction）和两种注意力适配技术，在 FLUX 模型上实现了无需训练、无需用户掩码的高质量真实图像编辑，人类评估中 68.2% 优选率远超其他方法。

**[Regen Learning Compact Video Embedding With Re-Generative Decoder](image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)**

:   提出 REGEN，用扩散 Transformer（DiT）替代传统 VAE 解码器作为视频的再生式解码器，通过"生成而非精确重建"的学习范式突破视频时序压缩瓶颈，实现最高 32× 时序压缩。

**[Repa-E Unlocking Vae For End-To-End Tuning Of Latent Diffusion Transformers](image_generation/repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)**

:   本文提出 REPA-E，通过表示对齐（REPA）损失实现 VAE 和潜在扩散 Transformer 的端到端联合训练，训练速度分别比 REPA 和普通训练快 17× 和 45×，在 ImageNet 256×256 上达到 FID 1.12 的新SOTA。

**[Repae Unlocking Vae For Endtoend Tuning Of Latent Diffusion](image_generation/repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)**

:   回答了"潜空间扩散模型能否与VAE端到端联合训练"的基础问题——发现标准扩散loss无法端到端训练但表示对齐（REPA）loss可以，提出REPA-E实现VAE+DiT联合训练，训练速度比REPA快17倍、比vanilla快45倍，在ImageNet 256×256上达到1.12 FID（w/ CFG）的新SOTA。

**[Rethink Sparse Signals For Pose-Guided Text-To-Image Generation](image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)**

:   提出 SP-Ctrl（Spatial-Pose ControlNet），通过可学习空间姿态表示（SPR）替换 OpenPose 的固定 RGB 编码，并引入关键点概念学习（KCL）策略利用交叉注意力热力图约束增强关键点对齐，使稀疏姿态信号达到与密集信号（深度图/DensePose）相当的姿态控制精度，同时保持图像多样性和跨物种生成能力。

**[Rethinking Cross-Modal Interaction In Multimodal Diffusion Transformers](image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)**

:   分析发现 MM-DiT 架构（FLUX、SD3.5）中视觉与文本 token 数量不对称导致交叉注意力被抑制、且注意力权重对时间步不敏感，提出 TACA（Temperature-Adjusted Cross-modal Attention）通过温度缩放和时间步自适应调整重新平衡多模态交互，结合 LoRA 微调在 T2I-CompBench 上显著提升文图对齐（空间关系+16.4%、形状+5.9%），且几乎无额外计算开销。

**[Rethinking Layered Graphic Design Generation With A Top-Down Approach](image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)**

:   提出Accordion框架，采用自顶向下策略将AI生成的栅格化设计图转换为可编辑的分层设计（含背景、前景对象、矢量化文本层），由VLM在参考创建、设计规划和层生成三个阶段扮演不同角色。

**[Rethinking The Embodied Gap In Vision-And-Language Navigation A Holistic Study O](image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)**

:   > 提出 VLN-PE，首个物理真实的视觉-语言导航平台，支持人形、四足和轮式机器人，系统评估现有 VLN 方法在真实物理约束下的性能，揭示了仿真到物理部署中 34% 的成功率下降。

**[Revelio Interpreting And Leveraging Semantic Information In Diffusion Models](image_generation/revelio_interpreting_and_leveraging_semantic_information_in_diffusion_models.md)**

:   Revelio 使用 k-稀疏自编码器（k-SAE）揭示扩散模型不同层和时间步中蕴含的单语义（monosemantic）可解释特征，并通过轻量分类器 Diff-C 验证这些特征的迁移学习价值，实现对黑盒扩散模型的深度解读。

**[Sa-Lut Spatial Adaptive 4D Look-Up Table For Photorealistic Style Transfer](image_generation/sa-lut_spatial_adaptive_4d_look-up_table_for_photorealistic_style_transfer.md)**

:   本文提出 SA-LUT，通过风格引导的 4D 查找表和内容-风格交叉注意力生成的上下文映射，实现空间自适应的写实风格迁移，在新提出的 PST50 基准上 LPIPS 相比 3D LUT 方法降低 66.7%，同时支持 4K 视频 16 FPS 实时处理。

**[Sana-Sprint One-Step Diffusion With Continuous-Time Consistency Distillation](image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)**

:   SANA-Sprint 提出混合蒸馏框架（连续时间一致性模型 + 潜空间对抗蒸馏），将预训练 Flow Matching 模型无损转换为 TrigFlow 并通过 sCM+LADD 联合训练，实现 1-4 步统一自适应高质量文本到图像生成，H100 上单步仅需 0.1 秒。

**[Sanasprint Onestep Diffusion With Continuoustime Consistency](image_generation/sanasprint_onestep_diffusion_with_continuoustime_consistency.md)**

:   将预训练的SANA flow matching模型通过无损数学变换转化为TrigFlow，结合连续时间一致性蒸馏（sCM）和潜空间对抗蒸馏（LADD）的混合策略，实现统一的1-4步自适应高质量图像生成，1步生成1024×1024图像仅需0.1s（H100），以7.59 FID和0.74 GenEval超越FLUX-schnell且速度快10倍。

**[Scflow Implicitly Learning Style And Content Disentanglement With Flow Models](image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)**

:   提出SCFlow，通过Flow Matching学习风格和内容的可逆合并映射，利用映射的可逆性让解耦作为合并过程的自然涌现属性，无需显式解耦监督。

**[Scorehoi Physically Plausible Reconstruction Of Human-Object Interaction Via Sco](image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)**

:   ScoreHOI 利用 score-based 扩散模型作为优化器，结合 DDIM 逆向-正向采样与物理约束（接触、穿透、地面接触）引导去噪过程，并通过接触驱动的迭代细化策略，从单目图像实现物理合理的人体-物体交互三维重建，在 BEHAVE 上接触 F-Score 提升 9%。

**[Sdmatte Grafting Diffusion Models For Interactive Matting](image_generation/sdmatte_grafting_diffusion_models_for_interactive_matting.md)**

:   本文提出 SDMatte，基于 Stable Diffusion 的交互式抠图模型，通过视觉提示驱动交叉注意力、坐标/不透明度嵌入和掩码自注意力三项设计，将扩散模型的文本交互能力转化为视觉提示交互能力，在多个数据集上显著超越 SAM-based 方法。

**[Semantic Watermarking Reinvented Enhancing Robustness And Generation Quality Wit](image_generation/semantic_watermarking_reinvented_enhancing_robustness_and_generation_quality_wit.md)**

:   针对潜扩散模型（LDM）的语义水印方法因丢弃虚部而导致频率完整性缺失的问题，提出厄密对称傅里叶水印（SFW）和中心感知嵌入策略，在维持频域完整性的同时增强检测鲁棒性和生成质量。

**[Shortft Diffusion Model Alignment Via Shortcut-Based Fine-Tuning](image_generation/shortft_diffusion_model_alignment_via_shortcut-based_fine-tuning.md)**

:   提出 ShortFT，利用轨迹保持少步扩散模型构建去噪捷径（shortcut），将原本冗长的去噪链大幅缩短，从而实现完整的端到端奖励梯度反向传播，高效且有效地将扩散模型与奖励函数对齐。

**[Smgdiff Soccer Motion Generation Using Diffusion Probabilistic Models](image_generation/smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)**

:   提出 SMGDiff，一个两阶段扩散模型框架，能够根据用户控制信号实时生成高质量、多样化的足球运动动画，同时通过接触引导模块优化球-脚交互细节。

**[Spectral Image Tokenizer](image_generation/spectral_image_tokenizer.md)**

:   提出 Spectral Image Tokenizer (SIT)，用离散小波变换 (DWT) 将图像从空域转换到频域后再进行 token 化，使 token 序列天然地按"粗到细"排列，从而支持多分辨率重建、渐进式生成、文本引导上采样与编辑等传统 raster-scan tokenizer 无法实现的能力。

**[Straighten Viscous Rectified Flow Via Noise Optimization](image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)**

:   本文提出 VRFNO（Viscous Rectified Flow via Noise Optimization），通过引入历史速度项增强轨迹区分度、并用编码器联合训练来优化噪声构建最优耦合，有效拉直 Rectified Flow 的推理轨迹，在 CIFAR-10 和 AFHQ 上取得单步/少步生成的 SOTA 性能（单步 FID 4.50，无需蒸馏）。

**[Streamdiffusion A Pipeline-Level Solution For Real-Time Interactive Generation](image_generation/streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)**

:   StreamDiffusion 提出管线级实时扩散框架，通过 Stream Batch（去噪步骤批处理）、R-CFG（残差无分类器引导）和 SSF（随机相似性过滤）等策略，在单张 RTX 4090 上实现高达 91 fps 的实时图像生成，比 Diffusers AutoPipeline 快 59.6 倍。

**[Structure-Guided Diffusion Models For High-Fidelity Portrait Shadow Removal](image_generation/structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)**

:   本文将人像阴影去除建模为扩散 Inpainting 问题，通过训练光照无关的结构提取网络获取排除阴影边界的结构图、以结构图引导 Inpainting 扩散模型修复阴影区域，再用梯度引导细节恢复扩散模型补回精细面部细节，在基准数据集上显著超越现有方法。

**[Stylekeeper Prevent Content Leakage Using Negative Visual Query Guidance](image_generation/stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)**

:   提出 **负视觉查询引导（NVQG）** 方法，通过在 self-attention 层中将参考图的 query 注入作为负向引导来抑制内容泄漏，实现了无需训练的高质量视觉风格提示，在风格相似度和文本对齐上均优于现有方法。

**[Stylemotif Multi-Modal Motion Stylization Using Style-Content Cross Fusion](image_generation/stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)**

:   提出 StyleMotif，一个单分支运动潜在扩散框架，通过风格-内容交叉归一化机制统一内容生成与多模态（文本/图片/视频/音频/运动）风格注入，相比 SMooDi 双分支设计减少 43.9% 可训练参数并提速 22.5%，同时在风格识别准确率（SRA）上提升 5.23%。

**[Summdiff Generative Modeling Of Video Summarization With Diffusion](image_generation/summdiff_generative_modeling_of_video_summarization_with_diffusion.md)**

:   SummDiff 首次将扩散模型引入视频摘要任务，将其定义为条件生成问题，通过学习"好摘要"的分布来生成多种合理摘要，更好地反映视频摘要任务固有的主观性。

**[Superedit Rectifying And Facilitating Supervision For Instruction-Based Image Ed](image_generation/superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)**

:   SuperEdit 通过利用扩散生成先验引导 VLM 修正编辑指令、并构建对比监督信号（正/负指令 + triplet loss）来解决指令式图像编辑中的噪声监督问题，以更少数据和更小模型超越 SmartEdit 9.19%。

**[Synthesizing Near-Boundary Ood Samples For Out-Of-Distribution Detection](image_generation/synthesizing_near-boundary_ood_samples_for_out-of-distribution_detection.md)**

:   本文提出 SynOOD，利用 MLLM 提取上下文语义 + 扩散模型迭代 inpainting + OOD 梯度引导，合成靠近 InD/OOD 边界的挑战性 OOD 样本，用于微调 CLIP 图像编码器和负标签特征，在 ImageNet 基准上 AUROC 提升 2.80%、FPR95 降低 11.13%。

**[Taxadiffusion Progressively Trained Diffusion Model For Fine-Grained Species Gen](image_generation/taxadiffusion_progressively_trained_diffusion_model_for_fine-grained_species_gen.md)**

:   TaxaDiffusion 利用生物分类学的层级结构（Kingdom→Phylum→Class→Order→Family→Genus→Species）渐进式训练扩散模型，从高层共有特征逐步细化到物种级别的细微差异，实现了高精度的细粒度动物图像生成，在 FishNet 数据集上 FID 降至 31.87（vs LoRA 的 43.91），BioCLIP 对齐分数提升 37%，且对样本极少（甚至仅 1 张）的稀有物种同样有效。

**[Teefusion Blending Text Embeddings To Distill Classifier-Free Guidance](image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)**

:   本文提出 TeEFusion，通过将 CFG 的引导幅度直接编码为条件/无条件文本嵌入的线性组合来替代双重前向传播，实现零额外参数的高效 CFG 蒸馏，同时兼容教师模型的复杂采样策略（如 Z-Sampling、W2SD），使学生模型推理速度达教师的 6 倍。

**[Tera Rethinking Text-Guided Realistic 3D Avatar Generation](image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)**

:   提出TeRA，首个基于隐空间扩散模型的文本引导3D真人头像生成框架，通过蒸馏大规模人体重建模型构建结构化隐空间，12秒生成写实3D人物，比SDS方法快两个数量级。

**[Text Embedding Knows How To Quantize Text-Guided Diffusion Models](image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)**

:   首次利用文本提示（text prompt）指导扩散模型的动态量化比特分配——通过预测文本对应的生成图像质量，为不同层和时间步自适应选择高/中/低比特精度，在降低计算复杂度的同时保持甚至提升生成质量。

**[The Silent Assistant Noisequery As Implicit Guidance For Goal-Driven Image Gener](image_generation/the_silent_assistant_noisequery_as_implicit_guidance_for_goal-driven_image_gener.md)**

:   本文提出 NoiseQuery，一种免训练的 T2I 生成增强方法，通过预构建大规模噪声库并在推理时检索与用户目标最匹配的初始噪声，实现高级语义和低级视觉属性的细粒度控制，仅需 0.002 秒/prompt 的额外开销即可提升多种 T2I 模型和增强技术的效果。

**[Timestep-Aware Diffusion Model For Extreme Image Rescaling](image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)**

:   提出 TADM，在预训练 SD 的潜空间中执行极端图像缩放（16×/32×），通过解耦特征缩放模块和时间步自适应对齐策略，动态分配扩散模型的生成能力以应对空间非均匀退化。

**[Tlb-Vfi Temporal-Aware Latent Brownian Bridge Diffusion For Video Frame Interpol](image_generation/tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)**

:   提出 TLB-VFI，一种高效的视频扩散模型用于帧插值：通过时域感知自编码器（隐空间时域块+像素空间3D小波门控）提取丰富的时间信息，结合重新设计的布朗桥扩散过程，在参数量仅 46.7M（比图像扩散方法少 3×、比视频扩散方法少 20×）的情况下，在 SNU-FILM extreme 和 Xiph-4K 上 FID 提升约 20%。

**[Towards Robust Defense Against Customization Via Protective Perturbation Resista](image_generation/towards_robust_defense_against_customization_via_protective_perturbation_resista.md)**

:   提出AntiPure——一种抗净化（anti-purification）保护性扰动方法，通过Patch-wise频域引导和错误时间步引导两种机制，使保护性扰动在被扩散净化后仍能有效干扰定制化生成（DreamBooth/LoRA），实现最小感知差异和最大输出失真的双重目标。

**[Trade-Offs In Image Generation How Do Different Dimensions Interact](image_generation/trade-offs_in_image_generation_how_do_different_dimensions_interact.md)**

:   提出 TRIG-Bench 基准（40,200 样本，10 个评估维度，132 个成对维度子集），以及 VLM-as-Judge 指标 TRIGScore，首次系统性地揭示和分析了图像生成模型在不同评估维度（如真实性、关系对齐、风格等）之间的权衡关系，并通过维度权衡图（DTM）指导微调实现性能提升。

**[Trans-Adapter A Plug-And-Play Framework For Transparent Image Inpainting](image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)**

:   提出Trans-Adapter，一种即插即用的适配器模块，使基于扩散的图像修复模型能直接处理透明（RGBA）图像，同时引入LayerBench基准和Alpha边缘质量（AEQ）度量指标。

**[Transformed Low-Rank Adaptation Via Tensor Decomposition And Its Applications To](image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)**

:   提出 TLoRA 方法，将预训练权重的微调分解为 **变换（Transform）** 和 **残差（Residual）** 两个适应部分，分别采用张量环矩阵（TRM）和张量环（TR）分解进行参数化，在 SDXL 上实现了仅 0.4M 参数的超参数高效微调，同时性能优于 LoRA 等基线方法。

**[Trce Towards Reliable Malicious Concept Erasure In Text-To-Image Diffusion Model](image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)**

:   提出 TRCE，通过两阶段概念擦除策略（文本语义擦除 + 去噪轨迹转向），在可靠擦除恶意概念的同时最小化对模型正常生成能力的影响。

**[Understanding Flatness In Generative Models Its Role And Benefits](image_generation/understanding_flatness_in_generative_models_its_role_and_benefits.md)**

:   本文首次系统研究损失景观平坦性在生成模型（尤其是扩散模型）中的角色与优势，理论证明平坦极小值可增强对先验分布扰动的鲁棒性，实验表明 SAM 能有效提升扩散模型的平坦性，从而改善生成质量、降低暴露偏差和量化误差。

**[Unicombine Unified Multi-Conditional Combination With Diffusion Transformer](image_generation/unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)**

:   UniCombine 提出基于 DiT 的多条件可控生成框架，通过 Conditional MMDiT Attention 机制和 LoRA Switching 模块，实现任意条件组合（文本+空间图+主体图像）的统一生成，支持 training-free 和 training-based 两种模式，并构建了首个多条件生成数据集 SubjectSpatial200K。

**[Unlocking The Potential Of Diffusion Priors In Blind Face Restoration](image_generation/unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)**

:   本文提出 FLIPNET，一个基于 T2I 扩散模型的统一框架，通过翻转输入在修复模式（BoostHub 选择性融合 LQ 特征 + BFR-oriented 面部嵌入）和退化模式（从真实退化数据集学习并合成退化图像）之间切换，同时解决 HQ/LQ 分布差距和合成/真实退化差距两大难题。

**[Video Color Grading Via Look-Up Table Generation](image_generation/video_color_grading_via_look-up_table_generation.md)**

:   提出基于扩散模型显式生成 LUT 的视频调色框架：通过 GS-Extractor 提取参考场景的高层风格特征，用 L-Diffuser 生成色彩查找表（LUT），一次生成即可无损应用于全部视频帧，并支持文本 prompt 进行亮度/对比度等细粒度调整。

**[Video Motion Graphs](image_generation/video_motion_graphs.md)**

:   Video Motion Graphs 提出了一个基于检索+生成的通用人体运动视频系统，通过将参考视频构建为运动图结构并进行条件化路径搜索获取关键帧，再利用 HMInterp（一个双分支扩散帧插值模型，结合运动扩散模型的骨骼引导和渐进式条件训练）来无缝连接不连续帧，在多种条件（音乐、语音、动作标签）下生成高质量人体运动视频，显著优于生成式和检索式基线。

**[Vigface Virtual Identity Generation For Privacy-Free Face Recognition Dataset](image_generation/vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)**

:   提出 VIGFace 框架，通过在人脸识别模型的特征空间中预先分配与真实身份正交的虚拟原型（virtual prototypes），训练扩散模型从虚拟原型生成不存在于真实世界的人脸图像，实现隐私无忧的人脸识别数据集构建和数据增强。

**[Visualcloze A Universal Image Generation Framework Via Visua](image_generation/visualcloze_a_universal_image_generation_framework_via_visua.md)**

:   提出 VisualCloze，将多种图像生成任务统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像 infilling 模型实现统一生成，并构建 Graph200K 图结构数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。

**[Visualcloze A Universal Image Generation Framework Via Visual In-Context Learnin](image_generation/visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)**

:   提出 VisualCloze，将多种图像生成任务统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像 infilling 模型实现统一生成，并构建 Graph200K 图结构数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。

**[What Makes For Text To 360-Degree Panorama Generation With Stable Diffusion](image_generation/what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)**

:   通过系统分析LoRA微调中$W_{\{q,k,v,o\}}$各组件的行为，揭示了$W_v$和$W_o$负责学习全景球面结构而$W_q$和$W_k$保留透视域共享知识的机制，并据此提出高效的单分支全景生成框架UniPano。

**[Whats In A Latent Leveraging Diffusion Latent Space For Domain Generalization](image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)**

:   深入分析了不同预训练模型（CLIP、DiT、SD、MAE、DINOv2、ResNet）隐空间的域分离能力，发现扩散模型特征在无监督情况下最擅长分离域信息，并提出 GUIDE 框架——用扩散特征发现伪域表征并增广分类器特征，在 5 个 DomainBed 数据集上无需域标签即取得 66.3% 平均准确率（超越 ERM 基线 +2.6%，在 TerraIncognita 上 +4.3%），且优于大多数需要域标签的方法。

**[Your Text Encoder Can Be An Object-Level Watermarking Controller](image_generation/your_text_encoder_can_be_an_object-level_watermarking_controller.md)**

:   通过仅微调文本编码器中的伪 token 嵌入 $\mathcal{W}_*$，实现对 T2I 扩散模型生成图像的对象级不可见水印嵌入，以 $10^5\times$ 更少的参数达到 99% 的比特准确率（48 bits）。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[Acknowledging Focus Ambiguity In Visual Questions](multimodal_vlm/acknowledging_focus_ambiguity_in_visual_questions.md)**

:   首次定义并系统研究视觉问答中的**焦点歧义**（focus ambiguity）问题——当问题中的语言描述可能指向图像中多个合理区域时，现有 VQA 系统完全忽略了这种歧义。作者构建了 VQ-FocusAmbiguity 数据集（5,500 样本 + 12,880 实例分割），并证明现代模型在识别和定位焦点歧义方面表现很差。

**[Adaptive Prompt Learning Via Gaussian Outlier Synthesis For Out-Of-Distribution ](multimodal_vlm/adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)**

:   提出 APLGOS 框架，利用 ChatGPT 标准化 Q&A 对来初始化可学习 ID 提示，并在类条件高斯分布的低似然区域合成虚拟 OOD 提示和图像，通过对比学习对齐文本-图像嵌入，实现更紧凑的 ID/OOD 决策边界。

**[Advdreamer Unveils Are Visionlanguage Models Truly Ready For](multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)**

:   提出AdvDreamer框架从单张图像生成物理可复现的对抗性3D变换(Adv-3DT)样本，通过零样本单目姿态操作+自然度奖励模型+逆语义概率损失，揭示当前VLM（包括GPT-4o）在3D变化下性能下降高达50-80%，并建立首个3D变化鲁棒性VQA基准MM3DTBench。

**[Aigi-Holmes Towards Explainable And Generalizable Ai-Generated Image Detection V](multimodal_vlm/aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)**

:   提出 AIGI-Holmes，通过构建包含解释性标注的 Holmes-Set 数据集和精心设计的三阶段训练流程（视觉专家预训练 → SFT → DPO），将 MLLM 改造为既能准确检测 AI 生成图像又能提供人类可验证解释的"福尔摩斯"检测器，推理阶段通过协同解码策略进一步增强泛化能力。

**[Aircache Activating Inter-Modal Relevancy Kv Cache Compression For Efficient Lar](multimodal_vlm/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)**

:   提出 AirCache，一种面向 LVLM 的 KV Cache 压缩方法，通过精英观察窗口（Elite Observation Window）评估视觉 token 重要性，结合基于重要性分数分布强度与偏度的自适应层级预算分配，在仅保留 10% 视觉 KV Cache 时性能损失不超过 1%，解码延迟降低 29%-66%。

**[Analyzing Finetuning Representation Shift For Multimodal Llms Steering](multimodal_vlm/analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)**

:   提出一个无需训练的框架，通过概念级别分析揭示多模态大语言模型微调时的表征偏移，并利用偏移向量实现模型行为的轻量级引导（去偏、安全控制）。

**[Are They The Same Exploring Visual Correspondence Shortcomings Of Multimodal Llm](multimodal_vlm/are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)**

:   本文首次系统研究了多模态大模型（MLLM）在视觉对应匹配方面的不足，构建了含1510样本的MMVM基准和220K匹配数据集，并提出CoLVA方法通过目标级对比学习和细粒度视觉专家显著提升了MLLM的跨图像实例匹配能力。

**[Attention To The Burstiness In Visual Prompt Tuning](multimodal_vlm/attention_to_the_burstiness_in_visual_prompt_tuning.md)**

:   本文揭示了视觉Prompt Tuning中自注意力模块数据的"爆发性"（burstiness）和非高斯分布问题，提出通过数据白化和双线性模型来学习"爆发性prompt"，在多个基准上大幅超越VPT及其变体，如CUB数据集上从42.15%提升至77.86%。

**[Autocompose Automatic Generation Of Pose Transition Descriptions For Composed Po](multimodal_vlm/autocompose_automatic_generation_of_pose_transition_descriptions_for_composed_po.md)**

:   本文提出AutoComPose，首个利用多模态大语言模型（MLLM）自动生成人体姿态转换描述的框架，通过身体部位级描述生成、多样化增强和循环一致性损失，在取代昂贵的人工标注的同时实现了更优的组合姿态检索性能。

**[Babyvlm Data-Efficient Pretraining Of Vlms Inspired By Infant Learning](multimodal_vlm/babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)**

:   受人类婴儿高效学习能力的启发，提出BabyVLM框架，包括合成训练数据集（将通用数据转化为儿童导向的格式）和多个发展对齐的评估基准，实现了紧凑VLM在有限数据下的高效预训练，性能优于仅用SAYCam或通用数据训练的模型。

**[Background Invariance Testing According To Semantic Proximity](multimodal_vlm/background_invariance_testing_according_to_semantic_proximity.md)**

:   本文提出基于语义邻近度的背景不变性测试方法，通过关联分析构建关键词本体来系统采样背景场景，实现兼顾测试多样性（recall）和人类判断一致性（precision）的最优平衡，并验证可视化测试框架比全局统计指标更具信息量。

**[Basic Boosting Visual Alignment With Intrinsic Refined Embeddings In Multimodal ](multimodal_vlm/basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)**

:   通过分析 LLM 浅层对视觉嵌入的语义精炼过程，提出 BASIC 方法，利用 LLM 内部精炼后的视觉嵌入作为监督信号，从方向对齐和语义分布两个维度直接指导视觉投射器生成更好的初始视觉嵌入。

**[Bidirectional Likelihood Estimation With Multi-Modal Large Language Models For T](multimodal_vlm/bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)**

:   揭示了基于MLLM的检索系统中"候选先验偏差"问题——候选似然估计倾向于选择先验概率高而非语义最相关的候选，提出BLiM（双向似然估计）和CPN（候选先验归一化）模块来解决此问题，在四个文本-视频检索基准上平均R@1提升6.4。

**[Boosting Mllm Reasoning With Text-Debiased Hint-Grpo](multimodal_vlm/boosting_mllm_reasoning_with_text-debiased_hint-grpo.md)**

:   揭示GRPO在MLLM推理中的两大问题——低数据利用率（难题上所有输出均错误导致梯度无效）和文本偏差（模型忽视图像仅依赖文本推理），提出Hint-GRPO（自适应提供推理提示）和文本偏差校准（测试时增强图像条件）两套方案，在3个基座MLLM上的11个数据集上显著提升推理能力。

**[Cad-Assistant Tool-Augmented Vllms As Generic Cad Task Solvers](multimodal_vlm/cad-assistant_tool-augmented_vllms_as_generic_cad_task_solvers.md)**

:   提出CAD-Assistant，首个面向通用CAD任务的工具增强视觉大语言模型框架，通过集成CAD专用工具集（草图参数化器、渲染模块、约束检查器等）和FreeCAD Python API，在零样本设置下超越了监督式任务特定方法。

**[Capellm Support-Free Category-Agnostic Pose Estimation With Multimodal Large Lan](multimodal_vlm/capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)**

:   首次将多模态大语言模型（MLLM）引入类别无关姿态估计（CAPE），仅需查询图像和文本描述即可预测任意类别的关键点位置，无需传统的支持图像和标注，在MP-100基准上超越5-shot SOTA。

**[Captionsmiths Flexibly Controlling Language Pattern In Image Captioning](multimodal_vlm/captionsmiths_flexibly_controlling_language_pattern_in_image_captioning.md)**

:   提出CaptionSmiths框架，通过连续标量插值（而非离散聚类）对图像描述的长度、描述性和词汇独特性三个属性进行滑块式灵活控制，在多数据集联合训练下实现比基线更精确的属性控制和更高的词汇对齐质量。

**[Capture Evaluating Spatial Reasoning In Vision Language Models Via Occluded Obje](multimodal_vlm/capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)**

:   本文提出CAPTURe基准，通过要求VLM在遮挡场景中对规律排列的物体进行"模态补全计数"（amodal counting），系统评估VLM的空间推理和世界模型构建能力，发现即使最强的GPT-4o在遮挡场景下也有14.75%的计数误差，而人类几乎无误差。

**[Causal Disentanglement And Cross-Modal Alignment For Enhanced Few-Shot Learning](multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)**

:   提出 Causal CLIP Adapter (CCA)，利用 ICA 对 CLIP 视觉特征进行因果解纠缠，并通过单向微调文本分类器和双向交叉注意力增强跨模态对齐，在 11 个基准数据集上实现了少样本分类 SOTA。

**[Chimera Improving Generalist Model With Domain-Specific Experts](multimodal_vlm/chimera_improving_generalist_model_with_domain-specific_experts.md)**

:   提出 Chimera，一个可扩展的低成本多模态管道，通过轻量路由模块动态选择领域专家模型、渐进式训练策略以及 Generalist-Specialist Collaboration Masking（GSCM）机制，将领域专家知识（表格、图表、数学、文档）集成到通用多模态大模型中，在 MathVista 上达到 64.9%（SOTA），在多个视觉结构提取任务上也达到或超越专家模型水平。

**[Clipsym Delving Into Symmetry Detection With Clip](multimodal_vlm/clipsym_delving_into_symmetry_detection_with_clip.md)**

:   提出 CLIPSym，首次利用预训练 CLIP 模型的多模态理解能力进行反射和旋转对称性检测，设计语义感知提示分组 (SAPG) 策略整合文本语义线索，并引入具有旋转等变保证的解码器，在 DENDI、SDRW、LDRS 三个基准上达到 SOTA。

**[Coa-Vla Improving Vision-Language-Action Models Via Visual-Text Chain-Of-Afforda](multimodal_vlm/coa-vla_improving_vision-language-action_models_via_visual-text_chain-of-afforda.md)**

:   提出 CoA-VLA，将四类机器人可供性（物体、抓取、空间、运动）组织为链式推理，通过视觉-文本协同注入模块融合到扩散策略网络中，显著提升 VLA 模型在多任务操作中的精度和泛化能力。

**[Coavla Improving Visionlanguageaction Models Via Visualtext](multimodal_vlm/coavla_improving_visionlanguageaction_models_via_visualtext.md)**

:   提出Chain-of-Affordance（CoA-VLA）框架，将四类机器人affordance（物体、抓取、空间、运动）以文本和视觉双模态形式注入VLA模型的策略网络，在真实机器人7任务多任务学习中达到85.54%成功率，比OpenVLA高30.65%，并展现出对未见物体姿态和障碍物的泛化能力。

**[Compcap Improving Multimodal Large Language Models With Composite Captions](multimodal_vlm/compcap_improving_multimodal_large_language_models_with_composite_captions.md)**

:   提出 CompCap 框架，自动合成6类复合图像（拼贴、图文混合、图表、表格、代码、流程图）及其高质量描述文本，构建 CompCap-118K 数据集，通过在 SFT 阶段引入该数据集显著提升 MLLM 对复合图像的理解能力。

**[Controlling Multimodal Llms Via Reward-Guided Decoding](multimodal_vlm/controlling_multimodal_llms_via_reward-guided_decoding.md)**

:   提出多模态奖励引导解码 (MRGD)，通过构建两个奖励模型分别控制物体精度和召回率，在推理时实现对 MLLM 输出的细粒度可控性，同时显著降低物体幻觉。

**[Controlling Multimodal Llms Via Rewardguided Decoding](multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)**

:   提出MRGD（Multimodal Reward-Guided Decoding），通过训练一个基于PaliGemma的物体幻觉奖励模型和一个基于OWLv2的物体召回奖励模型，在MLLM推理时通过线性加权组合两个奖励来逐句搜索最优候选输出，在CHAIR上将LLaVA-1.5的CHAIRi从15.05降至4.53（降70%）且支持精度-召回率的动态可控权衡。

**[Cvpt Cross Visual Prompt Tuning](multimodal_vlm/cvpt_cross_visual_prompt_tuning.md)**

:   针对 Visual Prompt Tuning (VPT) 中 prompt token 参与 self-attention 导致的计算冗余和注意力破坏问题，提出 CVPT，通过 cross-attention 解耦 prompt 与 image token 的交互，并利用权重共享机制初始化 cross-attention，在 25 个数据集上显著超越 VPT，性能媲美主流 adapter 方法。

**[Dash Detection And Assessment Of Systematic Hallucinations Of Vlms](multimodal_vlm/dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)**

:   提出DASH自动化流水线，通过LLM生成文本查询（DASH-LLM）和扩散模型优化图像查询（DASH-OPT）两种策略，在ReLAION-5B中系统性地发现VLM的假阳性对象幻觉聚类，共发现19k+聚类和950k+图像，并构建了更具挑战性的DASH-B基准。

**[Disenq Disentangling Q-Former For Activity-Biometrics](multimodal_vlm/disenq_disentangling_q-former_for_activity-biometrics.md)**

:   提出 DisenQ（Disentangling Q-Former），通过结构化语言引导将视频特征解纠缠为生物特征、动作和非生物特征三个独立空间，无需额外视觉模态即可实现活动感知的行人识别 SOTA。

**[Dita Scaling Diffusion Transformer For Generalist Visionlang](multimodal_vlm/dita_scaling_diffusion_transformer_for_generalist_visionlang.md)**

:   提出Dita(Diffusion Transformer Policy)，区别于先前方法用浅层网络在embedding上去噪，采用in-context conditioning让去噪直接条件化于原始视觉token，通过causal Transformer处理语言+图像+timestep+噪声动作的完整token序列，334M参数在SimplerEnv零样本/LIBERO/CALVIN等benchmark上达到SOTA或可比性能。

**[Docthinker Explainable Multimodal Large Language Models With](multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)**

:   提出DocThinker，首个将GRPO（Group Relative Policy Optimization）强化学习应用于文档理解的框架，通过四目标规则奖励（格式、答案准确度、RoI IoU、问题改写质量）训练MLLM自主生成可解释的推理过程，仅用4K训练数据在DocVQA上将Qwen2.5-VL-7B从0.355提升到0.579（RL vs SFT: 0.579 vs 0.355），并在视觉定位任务上达到82.4%精度。

**[Dogr Towards Versatile Visual Document Grounding And Referring](multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)**

:   提出文档定位与指代数据引擎 DOGR-Engine，构建首个全面评估文档定位/指代能力的基准 DOGR-Bench（7类任务×3种文档），并开发首个兼具精准文本定位和交互式grounding/referring能力的文档理解MLLM——DOGR。

**[Dynamic-Vlm Simple Dynamic Visual Token Compression For Videollm](multimodal_vlm/dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)**

:   提出 Dynamic-VLM，通过动态视觉Token压缩器根据视频长度灵活调整每帧Token数量，配合200万级高质量合成视频QA数据集，在 VideoMME 上比 LLaVA-OneVision 提升 2.7%，在 MuirBench 上提升 10.7%。

**[Dynamic Group Detection Using Vlm-Augmented Temporal Groupness Graph](multimodal_vlm/dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)**

:   本文提出基于VLM增强的时序群组图（temporal groupness graph）进行视频中的动态人群群组检测，核心创新是用CLIP提取包含人对和背景的groupness-augmented特征来估计成组概率，并通过全帧时序图的Louvain聚类实现动态变化群组的检测。

**[Dynamic Multimodal Prototype Learning In Vision-Language Models](multimodal_vlm/dynamic_multimodal_prototype_learning_in_vision-language_models.md)**

:   提出 ProtoMM，一个 training-free 的多模态原型学习框架，通过将原型建模为文本描述和视觉粒子的离散分布，利用最优传输动态更新多模态原型，在 15 个 zero-shot 基准上达到 SOTA。

**[Effective Training Data Synthesis For Improving Mllm Chart Understanding](multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)**

:   提出模块化的五步图表数据合成流水线，生成包含10k+图表图像和300k+ QA对的高质量训练集ECD（Effective Chart Dataset），在多种开源MLLM上一致提升图表理解能力。

**[Enhancing Few-Shot Vision-Language Classification With Large Multimodal Model Fe](multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)**

:   提出稀疏注意力向量（SAVs）——一种无需微调的方法，从冻结的生成式大型多模态模型（LMM）的注意力头中提取不到 5% 的头作为强特征表示，仅需约 20 个标注样本即可在视觉语言分类任务上达到 SOTA，平均超越 LoRA 微调 7%（在 BLINK、VLGuard、NaturalBench 等挑战性基准上）。

**[Enrich And Detect Video Temporal Grounding With Multimodal Llms](multimodal_vlm/enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)**

:   提出 ED-VTG，将视频时序定位分为"先丰富查询、再预测时间区间"两阶段，利用多模态 LLM 的描述能力增补查询细节，配合轻量区间解码器和多实例学习框架，在多个基准上首次让 LLM 方法全面追平甚至超越专用模型。

**[Evading Data Provenance In Deep Neural Networks](multimodal_vlm/evading_data_provenance_in_deep_neural_networks.md)**

:   揭示了当前数据集所有权验证（DOV）方法的安全假象——通过一个统一的规避框架 Escaping DOV，利用教师模型在 OOD 数据集上向代理学生传输任务相关但标识无关的知识，成功同时绕过所有 11 种 DOV 方法。

**[Evev2 Improved Baselines For Encoderfree Visionlanguage Mode](multimodal_vlm/evev2_improved_baselines_for_encoderfree_visionlanguage_mode.md)**

:   系统性地探索无视觉编码器VLM的最优架构和训练策略，提出Divide-and-Conquer架构将transformer完全分解为模态专用组件（attention/FFN/LayerNorm各模态独立），在仅100M公开数据下超越所有encoder-free同类并接近encoder-based VLM性能。

**[Exploiting Vision Language Model For Training-Free 3D Point Cloud Ood Detection ](multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)**

:   提出 Graph Score Propagation（GSP），一种无需训练的框架，通过在类原型和测试数据构成的图上进行分数传播，结合 prompt 聚类和自训练负提示策略，利用 VLM 在 3D 点云上实现高效 OOD 检测，在合成和真实世界数据集上一致超越现有 SOTA。

**[Fa Forced Prompt Learning Of Vision-Language Models For Out-Of-Distribution Dete](multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)**

:   提出FA（Forced prompt leArning），通过引入一个可学习的"强制提示"并迫使其产生比冻结原始提示更高的ID类别匹配度，使提示学到超越标签文本语义的丰富ID类别描述，在无需外部辅助数据和额外参数的条件下显著提升基于CLIP的少样本OOD检测性能。

**[Falcon Resolving Visual Redundancy And Fragmentation In High](multimodal_vlm/falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)**

:   提出 FALCON，通过在 ViT 中引入可学习的视觉寄存器（Visual Register），利用 ReCompact 机制在编码阶段直接消除视觉冗余（9 倍 token 压缩），并用 ReAtten 模块通过寄存器间交互解决裁切导致的视觉碎片化问题。

**[Feather The Throttle Revisiting Visual Token Pruning For Vis](multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vis.md)**

:   揭示了VLM中视觉token剪枝方法（如FastV）因RoPE的长程衰减特性导致系统性地保留图像底部token的严重缺陷，并提出FEATHER方法通过去除RoPE+均匀采样+两阶段剪枝修复该问题，在定位任务上实现5倍以上的性能提升。

**[Feather The Throttle Revisiting Visual Token Pruning For Vision-Language Model A](multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)**

:   揭示了 VLM 中早期视觉 token 剪枝存在系统性位置偏差（RoPE 导致倾向保留图像底部 token），并提出 FEATHER 方法通过去除 RoPE + 均匀采样 + 多阶段剪枝解决该问题，在定位任务上实现 5× 以上性能提升。

**[Fedmvp Federated Multimodal Visual Prompt Tuning For Vision-Language Models](multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)**

:   提出FedMVP，在联邦学习场景下通过PromptFormer网络融合图像视觉特征和LLM生成的类别属性文本特征，生成动态多模态视觉提示注入CLIP的视觉编码器，在20个数据集、三种泛化设置下显著超越现有联邦提示学习方法1.57%-2.26%。

**[Finmmr Make Financial Numerical Reasoning More Multimodal Comprehensive And Chal](multimodal_vlm/finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal.md)**

:   提出 FinMMR，一个双语（中英文）多模态金融数值推理基准，包含 4,300 道题目、8,700+ 张金融图表、14 个金融子领域，系统评估了 15 个 MLLM 在专业领域复杂推理中的瓶颈，并提出视觉过滤、知识增强和模型协作三种改进策略。

**[Folder Accelerating Multi-Modal Large Language Models With Enhanced Performance](multimodal_vlm/folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)**

:   提出 FOLDER——一种即插即用的视觉 token 压缩模块，通过系统分析信息损失的三个关键因素（压缩影响、传播效应、聚合方式），在视觉编码器的最后几层进行激进的 token 合并，实现最多 70% 的 token 削减，同时保持甚至提升模型性能。

**[From Holistic To Localized Local Enhanced Adapters For Efficient Visual Instruct](multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)**

:   提出 Dual-LoRA 和 Visual Cue Enhancement (VCE) 两个模块，通过"从整体到局部"的范式解决高效视觉指令微调中的数据冲突问题，以仅 1.16× 推理时间开销超越 LoRA-MoE 方法。

**[G2D Boosting Multimodal Learning With Gradient-Guided Distillation](multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)**

:   提出G2D（Gradient-Guided Distillation），通过融合单模态教师到多模态学生的特征蒸馏+logit蒸馏损失，并结合基于单模态教师置信度分数的Sequential Modality Prioritization（SMP）梯度调制策略，解决多模态学习中的模态不平衡问题，在CREMA-D上实现85.89%准确率、超越所有专注模态不平衡的SOTA方法。

**[Gendop Auto-Regressive Camera Trajectory Generation As A Director Of Photography](multimodal_vlm/gendop_auto-regressive_camera_trajectory_generation_as_a_director_of_photography.md)**

:   提出 DataDoP 数据集（29K 真实电影镜头的自由运动相机轨迹+描述）和 GenDoP 自回归 Transformer 模型，通过文本和/或 RGBD 输入生成艺术化、高质量的相机运动轨迹，在可控性、运动稳定性和复杂度上超越现有方法。

**[Generalizable Object Re-Identification Via Visual In-Context Prompting](multimodal_vlm/generalizable_object_re-identification_via_visual_in-context_prompting.md)**

:   VICP 提出了一种可泛化的目标重识别框架，通过 LLM 从少量正负样本对中推理出身份判别规则，然后将其转化为动态视觉提示注入冻结的视觉基础模型（DINOv2），实现无需参数更新即可泛化到未见类别的 ReID。

**[Generate Transduct Adapt Iterative Transduction With Vlms](multimodal_vlm/generate_transduct_adapt_iterative_transduction_with_vlms.md)**

:   提出 GTA-CLIP，通过迭代执行"LLM 属性生成→属性增强传导推理→编码器微调"三步，在 12 个数据集上 zero-shot 平均提升 9.5%，few-shot 提升 3-4%，首次在零标签场景下统一了属性发现、传导推理和模型适配。

**[Geobench-Vlm Benchmarking Vision-Language Models For Geospatial Tasks](multimodal_vlm/geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)**

:   提出GEOBench-VLM，一个专为评估VLM地理空间任务能力而设计的综合基准，覆盖8大类31个子任务、超过10,000条人工验证指令，揭示了现有SOTA VLM（包括GPT-4o）在地理空间任务上仍然表现不佳（最高仅41.7%准确率）。

**[Global And Local Entailment Learning For Natural World Imagery](multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)**

:   提出 Radial Cross-Modal Embeddings（RCME）框架，通过显式建模蕴含关系的传递性（transitivity），在视觉-语言模型中学习层次化表示，使模型能够在生命之树（Tree of Life）的任意分类等级上推理，在层次分类和检索任务上超越现有 SOTA。

**[Grab A Challenging Graph Analysis Benchmark For Large Multimodal Models](multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)**

:   GRAB 是一个面向大型多模态模型（LMM）的图表分析基准测试，包含 3284 道合成题目覆盖 5 个任务和 23 个图形属性，当前最强模型 Claude 3.5 Sonnet 仅达到 21.0% 的准确率，揭示了 LMM 在视觉分析推理方面的严重不足。

**[Growing A Twig To Accelerate Large Vision-Language Models](multimodal_vlm/growing_a_twig_to_accelerate_large_vision-language_models.md)**

:   提出 TwigVLM，通过在 VLM 早期层上"生长"一个轻量级 twig 模块，同时实现 twig 引导的视觉 token 剪枝（TTP，prefilling 加速）和自推测解码（SSD，decoding 加速），在 LLaVA-1.5-7B 上剪枝 88.9% 视觉 token 后保留 96% 精度，长回答生成速度提升 154%，在精度和速度上均大幅超越现有方法。

**[Gtr Guided Thought Reinforcement Prevents Thought Collapse In Rl-Based Vlm Agent](multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)**

:   发现 VLM 智能体在 RL 训练中仅依赖结果奖励会导致"思维崩塌"（thought collapse），提出 GTR 框架通过外部 VLM 纠正器自动纠正推理过程并结合 PPO + SFT 联合训练思维和行动，在 24 点游戏和 ALFWorld 环境中实现 3-5 倍的任务成功率提升。

**[Harmonizing Visual Representations For Unified Multimodal Un](multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)**

:   发现掩码自回归（MAR）模型的编码器天然兼具生成所需的细粒度图像特征和理解所需的高层语义表示，据此提出Harmon——以共享MAR编码器统一图像生成与理解的自回归框架，通过三阶段渐进训练在GenEval上以0.76 Overall超越所有统一模型，同时理解能力匹配使用独立SigLIP编码器的Janus系列。

**[Hints Of Prompt Enhancing Visual Representation For Multimodal Llms In Autonomou](multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)**

:   提出Hints of Prompt（HoP）框架，通过三种层次化提示（Affinity/Semantic/Question hint）增强CLIP视觉表征的实例级结构、领域语义和问题相关性，在自动驾驶VQA任务上仅用25%数据即超越基线全数据性能。

**[Ideator Jailbreaking And Benchmarking Large Vision-Language Models Using Themsel](multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_vision-language_models_using_themsel.md)**

:   提出IDEATOR，利用VLM自身作为红队模型自主生成多模态越狱图文对，以94%成功率突破MiniGPT-4的安全机制，并基于此构建了包含3654个样本的VLJailbreakBench安全评估基准。

**[Ideator Jailbreaking And Benchmarking Large Visionlanguage M](multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)**

:   提出IDEATOR，首个用VLM自身做红队攻击VLM的黑盒越狱框架——利用一个弱安全对齐的VLM（MiniGPT-4）作为攻击者，结合Stable Diffusion生成语义丰富的图文越狱对，通过breadth-depth探索策略迭代优化，在MiniGPT-4上达94%攻击成功率（平均5.34次查询），迁移到LLaVA/InstructBLIP/Chameleon达75-88%，并构建VLJailbreakBench（3654样本）揭示11个VLM的安全漏洞。

**[Improving Large Vision And Language Models By Learning From A Panel Of Peers](multimodal_vlm/improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)**

:   提出 Panel-of-Peers (PoP) 学习框架，利用多个性能相近的 LVLM 互相生成候选答案、互相评分、构建偏好数据，并通过 SimPO 迭代自我改进，在 15 个基准上将平均分从 48% 提升至 57%，无需人工标注数据。

**[Information Density Principle For Mllm Benchmarks](multimodal_vlm/information_density_principle_for_mllm_benchmarks.md)**

:   提出"信息密度"原则从 Fallacy（错误）、Difficulty（难度）、Redundancy（冗余）、Diversity（多样性）四个维度评估 MLLM benchmark 质量，构建了一套 Human-Model-Data 三级自动化评估流水线，对 19 个主流 benchmark 进行了系统性的"benchmark for benchmark"分析。

**[Instruction-Grounded Visual Projectors For Continual Learning Of Generative Visi](multimodal_vlm/instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)**

:   提出 MVP（Mixture of Visual Projectors），一种基于指令上下文的视觉投影器混合专家框架，通过专家推荐策略和专家剪枝机制，使生成式 VLM 在持续学习新视觉-语言任务时避免灾难性遗忘，同时保持对不同指令类型的响应能力，在分类/描述/问答等任务上全面超越现有方法。

**[Instruction-Oriented Preference Alignment For Enhancing Multi-Modal Comprehensio](multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)**

:   提出**指令导向偏好对齐（IPA）**框架，通过自动化偏好构建机制和渐进式偏好数据收集管线，将对齐信号锚定在**指令完成效能**而非仅局限于幻觉因素，在 Qwen2VL-7B 上跨 9 个基准（幻觉评估、通用VQA、文本理解）实现一致性提升。

**[Interpretable Zero-Shot Learning With Locally-Aligned Vision-Language Model](multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)**

:   提出 LaZSL，通过最优传输（Optimal Transport）实现局部视觉区域与语义属性之间的细粒度对齐，在无需额外训练的前提下构建可解释的零样本分类器，在9个数据集上取得了兼顾准确性、可解释性和域泛化的优异表现。

**[Jailbreaking Multimodal Large Language Models Via Shuffle Inconsistency](multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)**

:   发现多模态大语言模型(MLLMs)在理解能力和安全能力之间存在**打乱不一致性(Shuffle Inconsistency)**——模型能理解打乱后的有害指令，但安全机制却无法防御；据此提出基于查询的黑盒越狱攻击方法 SI-Attack，在开源和闭源商用模型上均显著提升攻击成功率。

**[Large Multi-Modal Models Can Interpret Features In Large Multi-Modal Models](multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)**

:   本文提出了首个面向多模态大模型（LMM）的自动化特征解释框架，使用稀疏自编码器（SAE）分解 LMM 的内部表征为单语义特征，并利用更大的 LMM 对这些特征进行自动解释，还展示了特征引导可修正模型幻觉。

**[Latte Collaborative Test-Time Adaptation Of Vision-Language Models In Federated ](multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)**

:   提出 Latte 框架，在联邦学习的去中心化场景下，通过本地记忆与外部记忆的协同机制，实现视觉语言模型（如 CLIP）的协作式测试时自适应，兼顾跨客户端知识共享与个性化。

**[Llava-Kd A Framework Of Distilling Multimodal Large Language Models](multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)**

:   提出 LLaVA-KD 框架，通过多模态蒸馏(MDist)和关系蒸馏(RDist)策略配合三阶段训练方案(DPT-SFT-DFT)，将大规模 MLLM 的知识迁移到小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。

**[Llava-Prumerge Adaptive Token Reduction For Efficient Large Multimodal Models](multimodal_vlm/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)**

:   利用视觉编码器中CLS token与空间token之间注意力分数的稀疏性，自适应地剪枝和合并视觉token，在仅保留5.5%视觉token的情况下维持LMM的可比性能。

**[Llavacot Let Vision Language Models Reason Stepbystep](multimodal_vlm/llavacot_let_vision_language_models_reason_stepbystep.md)**

:   通过构建包含结构化推理标注的LLaVA-CoT-100k数据集，训练VLM自主执行"总结→视觉解读→逻辑推理→结论"四阶段推理，配合测试时SWIRES搜索策略，11B模型超越GPT-4o-mini和Gemini-1.5-pro等大模型。

**[Llavaprumerge Adaptive Token Reduction For Efficient Large M](multimodal_vlm/llavaprumerge_adaptive_token_reduction_for_efficient_large_m.md)**

:   利用CLIP-ViT中[CLS] token与视觉token之间注意力分数的稀疏特性，通过IQR异常值检测自适应选择重要视觉token，再用k-近邻聚类将被剪除token的信息合并回保留token，实现视觉token 14倍压缩且性能几乎不降。

**[Mastering Collaborative Multi-Modal Data Selection A Focus On Informativeness Un](multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)**

:   提出 DataTailor——基于信息性（informativeness）、唯一性（uniqueness）和代表性（representativeness）三大原则的协同多模态数据选择框架，仅用 15% 数据即可达到全量数据微调 101.3% 的性能，充分体现"Less is More"理念。

**[Matvlm Hybrid Mamba-Transformer For Efficient Vision-Language Modeling](multimodal_vlm/matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)**

:   提出MaTVLM，将预训练VLM中部分Transformer层替换为Mamba-2层并通过单阶段知识蒸馏训练，在保持竞争性性能的同时实现3.6倍推理加速和27.5%显存降低。

**[Mavias Mitigate Any Visual Bias](multimodal_vlm/mavias_mitigate_any_visual_bias.md)**

:   提出 MAVias，一个开放集视觉偏差缓解框架：利用图像标注基础模型提取视觉属性标签，用 LLM 筛选与目标类别无关的标签作为潜在偏差，再通过 vision-language embedding 编码偏差并融入训练过程以学习偏差不变表示，在 CelebA、Waterbirds、UrbanCars 和 ImageNet9 上大幅超越现有方法。

**[Mc-Bench A Benchmark For Multi-Context Visual Grounding In The Era Of Mllms](multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)**

:   提出多上下文视觉定位（Multi-Context Visual Grounding）这一新任务和 MC-Bench 基准——包含 2000 个人工标注样本、3 种文本描述风格、20 项实用技能，评估了 20+ 个 MLLM 和基础模型，揭示现有模型与人类之间存在显著性能差距（人类 AP50=41.3% vs. 最优端到端模型 AP50=30.7%），并提供了一个 GPT-4o + G-DINO 的 agentic 基线（AP50=36.2%）。

**[Metamorph Multimodal Understanding And Generation Via Instru](multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instru.md)**

:   提出Visual-Predictive Instruction Tuning (VPiT)——一种简单有效的视觉指令微调扩展，让预训练LLM同时预测离散文本token和连续视觉token，发现视觉生成能力是视觉理解能力提升的自然副产物，少量生成数据即可解锁，LLM的预训练知识可以迁移到视觉生成中克服常见失败模式。

**[Metamorph Multimodal Understanding And Generation Via Instruction Tuning](multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)**

:   提出 Visual-Predictive Instruction Tuning（VPiT），仅通过轻量级指令微调即可将预训练 LLM 扩展为同时理解和生成视觉 token 的统一模型 MetaMorph，发现视觉生成能力是视觉理解的自然副产物且两者互利不对称。

**[Mitigating Object Hallucinations Via Sentence-Level Early Intervention](multimodal_vlm/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)**

:   本文提出SENTINEL框架，基于"幻觉在生成早期出现并向后传播"的关键观察，通过域内候选引导、双检测器交叉验证构建句子级偏好数据，使用上下文感知DPO（C-DPO）实现早期干预，在Object HalBench上减少92%幻觉且保持通用能力。

**[Mm-Ifengine Towards Multimodal Instruction Following](multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)**

:   提出 MM-IFEngine 管线，系统性地生成高质量的图像-指令对数据（含 SFT 和 DPO 版本），并构建 MM-IFEval 基准，显著提升 MLLM 在多模态指令遵循任务上的表现。

**[Mm-Spatial Exploring 3D Spatial Understanding In Multimodal Llms](multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)**

:   Apple 提出 CA-VQA 数据集和 MM-Spatial 模型，利用高质量 3D 场景数据和开放集标注生成涵盖空间关系预测、度量估计和 3D grounding 的训练/评估数据集，训练出一个通用型 MLLM，在 3D 空间理解 benchmark 上达到 SOTA，同时保持其他任务的竞争力。

**[Mmat1M A Large Reasoning Dataset For Multimodal Agent Tuning](multimodal_vlm/mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning.md)**

:   提出首个百万规模的多模态agent调优数据集MMAT-1M，通过四阶段数据引擎（基础数据→推理轨迹生成→反思纠错→格式整合）为MLLM注入CoT推理、工具调用和反思能力，在InternVL2.5-8B上平均提升2.7%，RAG任务上提升8.8%。

**[Mmone Representing Multiple Modalities In One Scene](multimodal_vlm/mmone_representing_multiple_modalities_in_one_scene.md)**

:   提出 MMOne 通用框架，通过模态建模模块（含模态指示器）和多模态分解机制解决多模态场景表示中的属性差异和粒度差异问题，在单一 3DGS 表示中同时建模 RGB、热成像和语言等多种模态并均获提升。

**[Molparser End-To-End Visual Recognition Of Molecule Structures In The Wild](multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)**

:   提出 MolParser，一个端到端的光学化学结构识别 (OCSR) 方法，通过扩展 SMILES 表示（E-SMILES）处理 Markush 结构、构建 700 万级大规模训练集 MolParser-7M，并利用主动学习引入真实文献数据，在 WildMol 基准上以 76.9% 准确率显著超越现有方法。

**[Multi-Cache Enhanced Prototype Learning For Test-Time Generalization Of Vision-L](multimodal_vlm/multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)**

:   提出 MCP/MCP++ 多缓存增强的原型学习框架，通过 entropy cache、align cache 和 negative cache 三种互补缓存机制构建紧致的类内分布，并引入跨模态残差学习进一步优化视觉和文本原型对齐，在 15 个下游任务上实现了 SOTA 的零样本泛化性能。

**[Multimodal Llms As Customized Reward Models For Text-To-Image Generation](multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)**

:   提出 LLaVA-Reward，利用预训练 MLLM 的隐藏状态（而非文本生成）直接输出奖励值，通过 Skip-connection Cross Attention (SkipCA) 增强双向视觉-文本交互，配合 LoRA 适配不同评估维度，在文本-图像对齐、保真度和安全性评估上达到 SOTA，并可用于扩散模型推理时缩放。

**[Multiverse A Multi-Turn Conversation Benchmark For Evaluating Large Vision And L](multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)**

:   提出 MultiVerse 多轮对话评估基准，从 12 个 VLM 评估数据集中收集 647 段对话，覆盖 484 种任务和 484 种交互目标，采用 checklist 评估方法发现即使最强的 GPT-4o 在复杂多轮对话中仅达 50% 的成功率。

**[Muse-Vl Modeling Unified Vlm Through Semantic Discrete Encoding](multimodal_vlm/muse-vl_modeling_unified_vlm_through_semantic_discrete_encoding.md)**

:   提出语义离散编码 (SDE)，通过在视觉 tokenizer 的量化过程中融入预训练 CLIP 语义特征，使离散视觉 token 与语言 token 天然对齐，在仅用 24M 图文对的情况下实现了统一理解与生成的 SOTA 性能。

**[Musevl Modeling Unified Vlm Through Semantic Discrete Encodi](multimodal_vlm/musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)**

:   提出语义离散编码（SDE）视觉tokenizer，在VQGAN基础上加入SigLIP语义特征约束，使离散视觉token与语言token语义对齐，构建统一的自回归VLM（MUSE-VL），在仅用24M数据的条件下理解性能比Emu3提升4.8%，超过LLaVA-NeXT 34B专用理解模型3.7%，同时支持图像生成。

**[One Perturbation Is Enough On Generating Universal Adversarial Perturbations Aga](multimodal_vlm/one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)**

:   本文提出 C-PGC 框架，通过恶意对比学习训练条件扰动生成器，生成一对通用图文对抗扰动（UAP），能够从根本上破坏 VLP 模型的多模态对齐关系，在白盒和黑盒场景下对多种 VLP 模型和下游任务均取得优异攻击效果。

**[Only Onelayer Intervention Sufficiently Mitigates Hallucinat](multimodal_vlm/only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)**

:   提出ONLY，一种training-free的单层干预解码方法——通过Text-to-Visual Entropy Ratio（TVER）选择偏向文本的attention head生成textually-enhanced logits，然后与原始logits做自适应对比/协作解码，仅增加1.07×推理时间就在POPE上比VCD/M3ID高3.14%，在CHAIR上降低CHAIR_S 6.2个点。

**[Openvision A Fully-Open Cost-Effective Family Of Advanced Vision Encoders For Mu](multimodal_vlm/openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)**

:   本文发布 OpenVision——一个完全开源（数据、训练代码、权重）的视觉编码器家族（5.9M-632.1M参数），基于 CLIPS 框架和 Recap-DataComp-1B 数据集训练，在集成到 LLaVA 等多模态框架时匹配甚至超越 OpenAI CLIP 和 Google SigLIP 的性能，为社区提供透明、灵活的视觉骨干替代方案。

**[Oraclefusion Assisting The Decipherment Of Oracle Bone Script With Structurally ](multimodal_vlm/oraclefusion_assisting_the_decipherment_of_oracle_bone_script_with_structurally_.md)**

:   提出OracleFusion，一个两阶段语义字体排印框架：第一阶段利用MLLM增强的空间感知推理（SAR）分析甲骨文字形结构并定位关键部件；第二阶段提出Structural Oracle Vector Fusion（SOVF），通过字形结构约束和骨架保持损失生成语义丰富的矢量字体，在保持原始字形完整性的同时传达语义，辅助专家解读未释甲骨文。

**[Physics Context Builders A Modular Framework For Physical Reasoning In Vision-La](multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)**

:   提出 Physics Context Builders (PCBs)，一种模块化框架，通过微调小型专用 VLM 从仿真数据中学习生成详细的物理场景描述，作为物理上下文增强大型基础 VLM（如 GPT-4o）的物理推理能力，无需修改大模型本身。

**[Pi-Gps Enhancing Geometry Problem Solving By Unleashing The Power Of Diagrammati](multimodal_vlm/pi-gps_enhancing_geometry_problem_solving_by_unleashing_the_power_of_diagrammati.md)**

:   Pi-GPS 提出利用几何图形信息消解文本描述中的歧义，通过"纠正器+验证器"微模块解决了先前被忽视的文本模糊性问题，在 Geometry3K 上比此前最优神经符号方法提升近 10%。

**[Pro-Vpt Distribution-Adaptive Visual Prompt Tuning Via Prompt Relocation](multimodal_vlm/pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)**

:   提出 PRO-VPT 框架，通过嵌套优化将提示分布优化 (ADO) 与视觉提示调优 (VPT) 协同设计，利用闲置分数剪枝和强化学习分配策略迭代重定位提示，在 VTAB-1k 和 FGVC 上较 VPT 分别提升 1.6pp 和 2.0pp。

**[Probres Probabilistic Jump Diffusion For Open-World Egocentric Activity Recognit](multimodal_vlm/probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)**

:   提出 ProbRes 框架，通过基于跳跃扩散的概率残差搜索策略，结合 ConceptNet 常识先验与 VLM 似然估计，在开放世界第一人称活动识别中高效导航大规模搜索空间，大幅减少 VLM 查询次数的同时提升识别准确率。

**[R1-Vl Learning To Reason With Multimodal Large Language Models Via Step-Wise Gro](multimodal_vlm/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)**

:   提出 StepGRPO，一种新的在线强化学习框架，通过两种无需过程奖励模型的规则化步级推理奖励（StepRAR 步级推理准确性奖励 + StepRVR 步级推理有效性奖励），解决 MLLM 在 RL 训练中的稀疏奖励问题，使 MLLM 能够自主探索和改进推理能力。

**[Reasonvqa A Multi-Hop Reasoning Benchmark With Structural Knowledge For Visual Q](multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)**

:   提出 ReasonVQA 数据集，通过低成本可扩展框架将结构化百科知识（Wikidata）与图像自动融合，生成 1/2/3 跳的多跳推理问题，包含 598K 图像和 4.2M 问题，显著挑战了现有 VQA 模型。

**[Safeguarding Vision-Language Models Mitigating Vulnerabilities To Gaussian Noise](multimodal_vlm/safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)**

:   发现主流VLM普遍缺乏高斯噪声鲁棒性，提出Robust-VLGuard安全数据集（含图文对齐/不对齐场景）配合噪声增强微调提升高斯噪声鲁棒性，再结合DiffPure将对抗噪声转化为高斯噪声，构建DiffPure-VLM通用防御框架，有效抵御多种强度的对抗攻击。

**[Sc-Captioner Improving Image Captioning With Self-Correction By Reinforcement Le](multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)**

:   SC-Captioner 提出了一种基于策略梯度的多轮强化学习框架，通过设计包含正确性奖励和错误惩罚的纠错奖励函数，使大型视觉语言模型获得图像描述的自纠错能力，同时提出改进的 CAPTURE 评估指标。

**[Scaling Inference-Time Search With Vision Value Model For Improved Visual Compre](multimodal_vlm/scaling_inference-time_search_with_vision_value_model_for_improved_visual_compre.md)**

:   提出 Vision Value Model (VisVM)，一个基于时序差分（TD）学习训练的视觉价值模型，用于在推理时指导 VLM 逐句搜索生成更高质量的描述性标注——相比贪心解码和 CLIP-PRM，VisVM 搜索显著减少幻觉（CHAIRs 从 32.4 降至 26.2），且生成的数据用于自训练可在 9 个基准上平均提升 10.8%。

**[Scaling Inferencetime Search With Vision Value Model For Imp](multimodal_vlm/scaling_inferencetime_search_with_vision_value_model_for_imp.md)**

:   提出Vision Value Model（VisVM），用TD learning训练一个能预测VLM生成句子长期价值的价值网络，指导推理时逐句搜索生成更少幻觉、更丰富细节的图像描述，并进一步将VisVM生成的高质量caption用于自训练，在9个benchmark上平均提升LLaVA-Next 10.8%。

**[Scaling Laws For Native Multimodal Models](multimodal_vlm/scaling_laws_for_native_multimodal_models.md)**

:   通过训练457个不同架构、规模和训练配比的模型，系统研究Native Multimodal Models（NMM）的scaling law，发现early-fusion架构（不依赖预训练视觉编码器）在小参数规模时优于late-fusion，训练更高效，部署更简单，引入MoE可进一步显著提升性能。

**[Scan Bootstrapping Contrastive Pre-Training For Data Efficiency](multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)**

:   提出SCAN，一种动态自举数据集剪枝方法，通过迭代的剪枝候选识别和数据集突变操作，在CLIP和MoCo对比预训练中以30-35%的数据剪枝率实现平均不到1%的性能下降。

**[Shortv Efficient Multimodal Large Language Models By Freezin](multimodal_vlm/shortv_efficient_multimodal_large_language_models_by_freezin.md)**

:   发现MLLM中约60%的层对视觉token的变换几乎不影响模型输出（Layer Contribution极低），提出ShortV方法在这些"ineffective layers"中冻结视觉token（不参与attention query和FFN），在LLaVA-NeXT-13B上实现50% FLOPs降低且性能几乎不变，且与token剪枝方法（如FastV）正交可叠加。

**[Shortv Efficient Multimodal Large Language Models By Freezing Visual Tokens In I](multimodal_vlm/shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)**

:   发现 MLLM 中存在显著的**层级冗余**——多数层对视觉 token 的变换贡献极小，据此提出 ShortV：在约 60% 的层中冻结视觉 token（跳过其注意力和 FFN 计算），在 LLaVA-NeXT-13B 上实现 50% FLOPs 减少，性能几乎无损。方法免训练，且与 token 剪枝方法正交可叠加。

**[Smolora Exploring And Defying Dual Catastrophic Forgetting In Continual Visual I](multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)**

:   发现多模态大模型持续视觉指令微调（CVIT）中存在"双重灾难性遗忘"——视觉理解能力和指令遵循能力同时退化，提出SMoLoRA通过可分离路由的LoRA专家混合方法有效缓解该问题。

**[Sparsemm Head Sparsity Emerges From Visual Concept Responses](multimodal_vlm/sparsemm_head_sparsity_emerges_from_visual_concept_responses.md)**

:   发现MLLM中仅约5%的attention head主动参与视觉理解（称为"visual heads"），提出基于OCR任务的training-free识别方法量化每个head的视觉相关性，并设计SparseMM——按visual score非对称分配KV-Cache预算的策略，在DocVQA上仅用5.3%的cache（256/4830）即可维持Qwen2-VL的性能，实现1.87×加速和50%内存减少。

**[Sparsemm Head Sparsity Emerges From Visual Concept Responses In Mllms](multimodal_vlm/sparsemm_head_sparsity_emerges_from_visual_concept_responses_in_mllms.md)**

:   揭示了多模态大语言模型(MLLM)中仅约5%的注意力头实际参与视觉理解的"visual head sparsity"现象，提出基于OCR任务的免训练visual head识别框架，并设计SparseMM——一种按视觉分数对不同head分配不对称KV-Cache预算的加速策略，实现1.38×实时加速和52%显存降低，同时保持性能不降。

**[Sparsevila Decoupling Visual Sparsity For Efficient Vlm Infe](multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_infe.md)**

:   提出SparseVILA，将VLM推理时的视觉token稀疏化解耦为两个阶段——prefill阶段做query-agnostic剪枝（去冗余）、decode阶段做query-aware检索（精选相关token），在长视频任务上实现4.0×prefill加速、2.5×decode加速、2.6×端到端加速，同时在视频理解benchmark上精度不降反升。

**[Sparsevila Decoupling Visual Sparsity For Efficient Vlm Inference](multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)**

:   提出SparseVILA——首个解耦prefill和decode阶段视觉稀疏性的VLM推理加速框架：prefill阶段进行query-agnostic的冗余token剪枝，decode阶段进行query-aware的相关token检索，实现最高4.0×prefill加速、2.5×decode吞吐提升、2.6×端到端加速，同时在多轮对话场景中保持精度（现有方法因永久删除token而在多轮中急剧退化）。

**[Sparsity Outperforms Low-Rank Projections In Few-Shot Adaptation](multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)**

:   提出稀疏优化（SO）框架，通过动态稀疏梯度选择和基于重要性的动量剪枝来替代低秩适配方法（如LoRA），在11个数据集上的少样本VLM适配任务中实现了SOTA，同时降低了内存开销。

**[Synergistic Prompting For Robust Visual Recognition With Missing Modalities](multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)**

:   提出Synergistic Prompting（SyP）框架，通过动态适配器生成自适应缩放因子来调节基础prompt（动态prompt），并与共享跨模态特征的静态prompt协同，实现在模态缺失场景下的鲁棒视觉识别，在MM-IMDb/Food101/Hateful Memes三个数据集上全面超越DCP等SOTA。

**[Tab Transformer Attention Bottlenecks Enable User Intervention And Debugging In ](multimodal_vlm/tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)**

:   提出TAB（Transformer Attention Bottleneck），一个插入标准MHSA之后的单头co-attention瓶颈层，通过移除skip connection并将注意力约束到[0,1]区间，实现VLM注意力的精确可视化、真值监督训练、以及测试时用户编辑干预，在变化描述任务上首次建立了注意力值与VLM输出之间的因果关系。

**[The Inter-Intra Modal Measure A Predictive Lens On Fine-Tuning Outcomes In Visio](multimodal_vlm/the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)**

:   提出 Inter-Intra Modal Measure（IIMM）——一个仅需单次前向推理即可预测视觉-语言双编码器模型微调后性能增益和灾难性遗忘程度的指标，通过量化模态内图像嵌入相似性和模态间错误标签对齐程度，在 4 个基础模型和 5 种微调策略下展现出强线性预测能力（$R^2 > 0.85$）。

**[Toolvqa A Dataset For Multi-Step Reasoning Vqa With External Tools](multimodal_vlm/toolvqa_a_dataset_for_multi-step_reasoning_vqa_with_external_tools.md)**

:   提出 ToolVQA——一个包含 23K 样本的大规模多模态工具增强 VQA 数据集，通过 ToolEngine 管道（结合图像引导的 DFS 和 LCS 示例匹配）自动生成真实场景下的多步推理数据，在其上微调的 LLaVA-7B 在 5 个 OOD 基准上超越 GPT-3.5-Turbo。

**[Toolvqa A Dataset For Multistep Reasoning Vqa With External](multimodal_vlm/toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)**

:   提出ToolVQA，一个23K样本的多模态工具使用VQA数据集，通过ToolEngine数据生成pipeline（图像引导DFS + LCS示例匹配）从真实图像中构造隐式多步推理问题（平均2.78步），在该数据上微调LLaVA-7B后在5个OOD benchmark上超过GPT-3.5-Turbo，并揭示了当前LFM在参数预测和答案总结方面的瓶颈。

**[Training-Free Personalization Via Retrieval And Reasoning On Fingerprints](multimodal_vlm/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)**

:   提出R2P，首个免训练的VLM个性化方法，利用VLM自身的世界知识提取概念"指纹"属性，通过检索-推理范式和跨模态属性验证实现个人概念识别，无需任何微调或大规模预训练。

**[Trust But Verify Programmatic Vlm Evaluation In The Wild](multimodal_vlm/trust_but_verify_programmatic_vlm_evaluation_in_the_wild.md)**

:   提出 PROVE（Programmatic VLM Evaluation）评测范式，通过从超详细图像描述构建高保真场景图，并利用 LLM 生成可编程验证的开放式视觉问答对，在统一的场景图框架内同时评估 VLM 回答的**有用性**（helpfulness）和**真实性**（truthfulness），揭示当前模型在两者之间难以取得良好平衡。

**[Understanding Museum Exhibits Using Vision-Language Reasoning](multimodal_vlm/understanding_museum_exhibits_using_vision-language_reasoning.md)**

:   构建了一个包含 6500 万张图片和 2 亿个问答对的大规模博物馆展品数据集 Museum-65，并通过在该数据集上微调 BLIP 和 LLaVA 证明：领域特定的大规模数据集显著优于零样本 SOTA VLM，微调后的 LLaVA 在展品标题和产地识别上分别达到 57% 和 70% 的准确率（vs. GPT-4o 的 22% 和 33%）。

**[Unified Multimodal Understanding Via Byte-Pair Visual Encoding](multimodal_vlm/unified_multimodal_understanding_via_byte-pair_visual_encoding.md)**

:   将 NLP 中的 Byte-Pair Encoding (BPE) 策略应用于视觉 token 化，提出优先级引导的编码方案（融合频率和空间一致性）、课程式数据混合和渐进式参数解冻三阶段训练策略，构建的 Being-VL-0.5（8B）在离散 token 路线上接近连续 embedding 方法的主流水平。

**[Vision-Language Models Cant See The Obvious](multimodal_vlm/vision-language_models_cant_see_the_obvious.md)**

:   提出 SalBench 基准测试，发现当前大型视觉-语言模型（LVLM）在检测对人类而言显而易见的视觉显著特征（如颜色、方向、大小差异）上表现极差——最先进的 GPT-4o 在检测任务上仅达到 47.6% 准确率，揭示了 LVLM 与人类视觉注意力之间的根本差距。

**[Visnumbench Evaluating Number Sense Of Multimodal Large Language Models](multimodal_vlm/visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)**

:   本文提出 VisNumBench，一个包含约 1900 道多选题的基准，覆盖 7 种视觉数值属性和 4 类视觉数值估计任务，系统评估了 17 个 MLLM 的直觉数字感知能力，发现即使最先进的模型也远低于人类水平。

**[Visual-Oriented Fine-Grained Knowledge Editing For Multimodal Large Language Mod](multimodal_vlm/visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)**

:   提出面向视觉的细粒度多模态知识编辑任务及 FGVEdit 基准，设计 MSCKE 框架通过多模态范围分类器融合视觉与文本信息，实现对图像中多个交互实体的精确知识更新，显著优于纯文本编辑方法。

**[Visual Chronicles Using Multimodal Llms To Analyze Massive Collections Of Images](multimodal_vlm/visual_chronicles_using_multimodal_llms_to_analyze_massive_collections_of_images.md)**

:   提出 Visual Chronicles 系统，首次利用多模态大语言模型（MLLM）分析超过 2000 万张街景图像的海量数据库，通过自底向上的分层策略（局部变化检测 + 趋势发现）和高效的文本嵌入-MLLM 混合验证算法，无标签、开放式地发现城市十年间的视觉变化趋势（如旧金山新增太阳能板、高架桥被刷成蓝色等），将 MLLM 推理成本降低 2000 倍同时保持 93.9% 的验证准确率。

**[Visual Intention Grounding For Egocentric Assistants](multimodal_vlm/visual_intention_grounding_for_egocentric_assistants.md)**

:   提出首个面向**自我中心视觉意图定位**（egocentric visual intention grounding）的任务和数据集 **EgoIntention**（26K 图像 + 52K 意图描述 + 89K 边界框），揭示现有 MLLM 在隐式意图推理和第一人称视觉定位上的重大不足，并提出 **Reason-to-Ground (RoG)** 指令微调方法，通过解耦意图推理和物体定位显著提升性能。

**[Visual Interestingness Decoded How Gpt-4O Mirrors Human Interests](multimodal_vlm/visual_interestingness_decoded_how_gpt-4o_mirrors_human_interests.md)**

:   系统性研究了 GPT-4o 等大型多模态模型对"图像有趣性"这一主观视觉概念的理解程度，发现 GPT-4o 与人类评判有中等正相关（配对图像一致率 73.8%），并提出利用 GPT-4o 自动标注图像对训练 learning-to-rank 模型来预测图像有趣性，超越了所有现有方法。

**[Why Lvlms Are More Prone To Hallucinations In Longer Responses The Role Of Conte](multimodal_vlm/why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte.md)**

:   深入探究 LVLM 长文本生成中幻觉频发的根本原因——不是长度本身，而是上下文的连贯性（coherence）和完备性（completeness）需求驱动模型外推产生幻觉，并据此提出 HalTrapper 的"诱导-检测-抑制"三阶段框架。

**[Wikiautogen Towards Multi-Modal Wikipedia-Style Article Generation](multimodal_vlm/wikiautogen_towards_multi-modal_wikipedia-style_article_generation.md)**

:   提出 WikiAutoGen 多智能体框架，通过整合文本和图像的多模态检索与多视角自反思机制，自动生成高质量的多模态 Wikipedia 风格文章，在自建基准 WikiSeek 上相比已有方法提升 8%–29%。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage G](autonomous_driving/3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)**

:   提出PGA，首个基于3DGS的物理对抗攻击框架，通过快速准确重建目标+解决Gaussians互/自遮挡问题+min-max背景对抗优化策略，生成跨视角鲁棒的物理对抗迷彩，在数字和物理域均超越SOTA方法。

**[3D Gaussian Splatting Driven Multiview Robust Physical Adver](autonomous_driving/3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)**

:   提出首个基于3D高斯体（3DGS）的物理对抗攻击框架PGA，通过解决高斯体的互遮挡和自遮挡问题保证跨视角一致性，并设计min-max优化策略过滤非鲁棒对抗特征，在数字域和物理域均大幅超越SOTA方法。

**[3Drealcar An In-The-Wild Rgb-D Car Dataset With 360-Degree Views](autonomous_driving/3drealcar_an_in-the-wild_rgb-d_car_dataset_with_360-degree_views.md)**

:   本文提出首个大规模真实3D车辆数据集3DRealCar，包含2500辆来自100+品牌的真实车辆，每辆车约200张高分辨率360度RGB-D视图，覆盖反光/标准/暗光三种光照条件，并提供13类车辆解析标注，支持3D重建、检测、生成等多种任务。

**[3Drealcar An Inthewild Rgbd Car Dataset With 360Degree Views](autonomous_driving/3drealcar_an_inthewild_rgbd_car_dataset_with_360degree_views.md)**

:   提出首个大规模3D真实汽车数据集3DRealCar，包含2,500辆真实汽车的高分辨率（1920×1440）360度RGB-D扫描（平均每辆200张视角），覆盖100+品牌和三种光照条件（标准/高反光/暗光），提供点云、解析图等丰富标注，并基准测试了多种3D重建方法，揭示了反光和暗光条件下的重建挑战。

**[4Dsegstreamer Streaming 4D Panoptic Segmentation Via Dual Th](autonomous_driving/4dsegstreamer_streaming_4d_panoptic_segmentation_via_dual_th.md)**

:   提出4DSegStreamer，一种通用的**双线程**流式4D全景分割框架——预测线程维护几何和运动记忆并预测未来动态，推理线程通过自我位姿对齐和逆向前向流迭代实现对新到帧的实时查询，可即插即用地集成到现有3D/4D分割方法中，在SemanticKITTI上sLSTQ比PTv3高7.7-15.2%，在高FPS场景下性能鲁棒性远超现有方法。

**[4Dsegstreamer Streaming 4D Panoptic Segmentation Via Dual Threads](autonomous_driving/4dsegstreamer_streaming_4d_panoptic_segmentation_via_dual_threads.md)**

:   提出4DSegStreamer，一种基于双线程系统（预测线程+推理线程）的流式4D全景分割框架，通过几何与运动记忆维护、自车位姿预测和逆向前向光流迭代实现实时高质量4D全景分割。

**[6Dope-Gs Online 6D Object Pose Estimation Using Gaussian Splatting](autonomous_driving/6dope-gs_online_6d_object_pose_estimation_using_gaussian_splatting.md)**

:   提出6DOPE-GS，一种利用2D高斯溅射（2DGS）联合优化6D物体位姿和3D重建的model-free在线追踪方法，通过动态关键帧选择和基于透明度百分位的密度控制实现5倍加速，同时保持SOTA精度。

**[6Dopegs Online 6D Object Pose Estimation Using Gaussian Spla](autonomous_driving/6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)**

:   利用2D Gaussian Splatting的高效可微渲染能力，提出一种无需CAD模型的在线6D物体位姿估计与跟踪方法，通过联合优化高斯物体场和关键帧位姿，实现比BundleSDF快约5倍的速度同时保持可比精度。

**[A Constrained Optimization Approach For Gaussian Splatting F](autonomous_driving/a_constrained_optimization_approach_for_gaussian_splatting_f.md)**

:   提出一种无需SfM的约束优化方法，通过相机位姿分解、灵敏度预调节、对数障碍约束和几何约束，从多相机SLAM系统输出的粗糙位姿和噪声点云中联合优化相机参数与3DGS场景重建。

**[A Constrained Optimization Approach For Gaussian Splatting From Coarsely-Posed I](autonomous_driving/a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)**

:   提出一种无需SfM的约束优化方法，通过相机位姿分解、灵敏度预调节、对数障碍约束和几何约束，从多相机SLAM系统输出的粗糙位姿和噪声点云中联合优化相机参数与3DGS场景重建。

**[Acam-Kd Adaptive And Cooperative Attention Masking For Knowledge Distillation](autonomous_driving/acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)**

:   提出 ACAM-KD，一种自适应学生-教师协作注意力掩码知识蒸馏方法，通过跨注意力特征融合（STCA-FF）和自适应空间-通道掩码（ASCM）动态调整蒸馏焦点，在 COCO 检测上超越 SOTA 最高 1.4 mAP，在 Cityscapes 分割上提升 3.09 mIoU。

**[Ad-Gs Object-Aware B-Spline Gaussian Splatting For Self-Supervised Autonomous Dr](autonomous_driving/ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)**

:   本文提出 AD-GS，一种基于 3D Gaussian Splatting 的自监督自动驾驶场景渲染框架，核心创新是将可学习 B-spline 曲线与三角函数结合进行局部-全局运动建模，并通过简化的二值伪分割实现鲁棒的场景分解，在不依赖人工 3D 标注的条件下大幅超越现有自监督方法。

**[Adadrive Self-Adaptive Slow-Fast System For Language-Grounded Autonomous Driving](autonomous_driving/adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)**

:   AdaDrive提出了首个自适应慢-快架构的LLM增强自动驾驶框架，通过两个自适应连接器动态决定"何时激活LLM"（Connector-W）和"LLM贡献多少"（Connector-H），在语言引导驾驶基准上实现了SOTA性能（驾驶分数80.9%），同时将推理延迟降低至189ms、显存降至6.79GB。

**[Adaptive Dual Uncertainty Optimization Boosting Monocular 3D](autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d.md)**

:   提出DUO（Dual Uncertainty Optimization），首个面向单目3D检测（M3OD）的测试时自适应框架，通过共轭焦点损失减少语义不确定性和法线场一致性约束减少几何不确定性，形成互补优化闭环。

**[Adaptive Dual Uncertainty Optimization Boosting Monocular 3D Object Detection Un](autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)**

:   提出 DUO（Dual Uncertainty Optimization），首个联合最小化语义不确定性和几何不确定性的测试时自适应框架，通过共轭焦点损失和法向场约束实现鲁棒的单目3D目标检测。

**[Ago Adaptive Grounding For Open World 3D Occupancy Predictio](autonomous_driving/ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)**

:   提出AGO框架，通过噪声增强的接地训练(grounding training)处理已知类别 + 模态适配器的自适应对齐处理未知类别，并用基于信息熵的开放世界识别器在推理时动态选择最佳特征，在Occ3D-nuScenes自监督基准上超越VEON 4.09 mIoU，同时具备开放世界零样本/少样本迁移能力。

**[Ccl-Lgs Contrastive Codebook Learning For 3D Language Gaussian Splatting](autonomous_driving/ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)**

:   提出CCL-LGS框架，通过零样本跟踪器实现跨视角掩码关联，并利用对比码本学习（CCL）模块蒸馏出类内紧凑、类间可区分的语义特征，从而解决基于2D先验的3D语义场重建中因遮挡、模糊和视角变化导致的跨视角语义不一致问题。

**[Colmdriver Llm-Based Negotiation Benefits Cooperative Autonomous Driving](autonomous_driving/colmdriver_llm-based_negotiation_benefits_cooperative_autonomous_driving.md)**

:   首个全流程 LLM 驱动的协作驾驶系统，通过 Actor-Critic 范式的语言协商模块和意图引导的轨迹生成器，在多种 V2V 交互场景中实现比现有方法高 11% 的成功率。

**[Controllable 3D Outdoor Scene Generation Via Scene Graphs](autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)**

:   首次提出以场景图（Scene Graph）作为控制信号生成大规模3D室外场景的方法——通过GNN将稀疏场景图编码为BEV嵌入图，再经2D→3D级联离散扩散模型生成语义3D场景，并配套交互系统让用户直接编辑场景图来控制生成。

**[Cooptrack Exploring End-To-End Learning For Efficient Cooperative Sequential Per](autonomous_driving/cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)**

:   提出 CoopTrack，首个完全实例级端到端协同 3D 多目标跟踪框架，通过可学习的图注意力关联模块和多维特征提取实现跨Agent实例匹配与融合，在 V2X-Seq 上达到 SOTA。

**[Counting Stacked Objects](autonomous_driving/counting_stacked_objects.md)**

:   将堆叠物体计数问题分解为"体积估计"和"占空比估计"两个子问题，前者用多视角3D重建解决，后者用深度图驱动的神经网络从可见表面推断，首次实现了对不可见堆叠物体的准确计数，性能远超人类。

**[Cvfusion Cross-View Fusion Of 4D Radar And Camera For 3D Object Detection](autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)**

:   提出CVFusion——首个4D雷达-相机两阶段融合网络，第一阶段通过雷达引导迭代（RGIter）BEV融合生成高召回率提案框，第二阶段利用点引导融合（PGF）和网格引导融合（GGF）聚合多视角异构特征进行提案精化，在VoD和TJ4DRadSet上分别取得9.10%和3.68%的mAP提升。

**[Damap Distance-Aware Mapnet For High Quality Hd Map Construction](autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)**

:   揭示当前HD地图构建方法在高质量预测上的两大固有缺陷——不恰当的分类标签与次优的任务特征，提出DAMap（含DAFL、HLS、TMDA三个组件）系统性地解决任务错位问题，在NuScenes和Argoverse2上多个基线方法上一致提升2-3 mAP。

**[Dchm Depth-Consistent Human Modeling For Multiview Detection](autonomous_driving/dchm_depth-consistent_human_modeling_for_multiview_detection.md)**

:   提出 DCHM，一种无需 3D 标注的深度一致性人体建模框架，通过超像素级高斯溅射生成伪深度标签来微调单目深度估计网络，结合多视角标签匹配实现稀疏视角、遮挡严重场景下的高精度行人检测，在 Wildtrack 上 MODA 达 84.2%，MODP 较 UMPD 提升 31.2%。

**[Decoupled Diffusion Sparks Adaptive Scene Generation](autonomous_driving/decoupled_diffusion_sparks_adaptive_scene_generation.md)**

:   提出 Nexus，一个基于解耦扩散的自适应驾驶场景生成框架，通过独立噪声状态实现目标导向与实时响应的统一，将位移误差降低 40%，并构建了包含 540 小时安全关键驾驶数据的 Nexus-Data。

**[Detect Anything 3D In The Wild](autonomous_driving/detect_anything_3d_in_the_wild.md)**

:   DetAny3D 是一个可提示（promptable）的3D检测基础模型，通过融合SAM和depth-pretrained DINO两个2D基础模型的先验知识，并提出2D Aggregator和Zero-Embedding Mapping机制实现稳定的2D-to-3D知识迁移，仅用单目图像即可在任意场景和相机配置下实现零样本3D目标检测，在新类别上零样本AP3D超越基线最多21%。

**[Dist-4D Disentangled Spatiotemporal Diffusion With Metric Depth For 4D Driving S](autonomous_driving/dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)**

:   提出DiST-4D，首个前馈式4D驾驶场景生成框架，通过将时间预测（DiST-T）和空间新视角合成（DiST-S）解耦为两个扩散过程，以度量深度（metric depth）为几何桥梁，在nuScenes上同时实现SOTA的时间视频生成（FVD 22.67）和空间NVS（FID 10.12），无需逐场景优化。

**[Donut A Decoder-Only Model For Trajectory Prediction](autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)**

:   DONUT受LLM中decoder-only架构启发，提出用统一的自回归模型处理历史和未来轨迹，配合"过预测（overprediction）"策略让模型更好预判未来，在Argoverse 2基准上取得SOTA。

**[Drivex Omni Scene Modeling For Learning Generalizable World Knowledge In Autonom](autonomous_driving/drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)**

:   提出 DriveX，一个自监督世界模型框架，通过 Omni Scene Modeling（联合3D点云预测、2D语义表示和图像生成）在 BEV 潜在空间学习可迁移的通用场景表征，并设计 Future Spatial Attention 范式将预测的未来状态无缝集成到占据预测、流估计和端到端驾驶等多种下游任务中，在多个任务上达到 SOTA。

**[Duet Dual Incremental Object Detection Via Exemplar-Free Task Arithmetic](autonomous_driving/duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)**

:   提出 DuET 框架，首次以无样本（exemplar-free）的任务算术（Task Arithmetic）模型合并方式，同时解决目标检测中的类别增量和域增量问题（Dual Incremental Object Detection, DuIOD），并引入方向一致性损失（Directional Consistency Loss）缓解符号冲突，在 Pascal Series 和 Diverse Weather Series 上大幅超越现有方法。

**[Embodiedocc Embodied 3D Occupancy Prediction For Vision-Based Online Scene Under](autonomous_driving/embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)**

:   提出EmbodiedOcc框架，利用3D语义高斯作为全局记忆，通过逐步探索和局部更新实现基于单目视觉输入的在线室内场景三维占据预测。

**[Emd Explicit Motion Modeling For High-Quality Street Gaussian Splatting](autonomous_driving/emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)**

:   提出显式运动分解（EMD）模块，通过可学习运动嵌入和双尺度形变框架为每个 Gaussian 基元建模其运动特性，作为即插即用模块可无缝集成到自监督和监督街景高斯溅射方法中，在 Waymo 和 KITTI 数据集上达到自监督设置的 SOTA 性能。

**[Epona Autoregressive Diffusion World Model For Autonomous Driving](autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)**

:   提出 Epona，一种自回归扩散世界模型，通过解耦时空建模和异步多模态生成，实现高分辨率长时程驾驶视频生成与实时轨迹规划的统一框架。

**[Eta Efficiency Through Thinking Ahead A Dual Approach To Self-Driving With Large](autonomous_driving/eta_efficiency_through_thinking_ahead_a_dual_approach_to_self-driving_with_large.md)**

:   提出ETA双系统框架，通过将大模型的当前帧计算转移到前序时间步并进行批量推理，使大模型特征在每帧都可用，在Bench2Drive上以50ms延迟达到69.53驾驶分数，提升SOTA 8%。

**[Evt Efficient View Transformation For Multi-Modal 3D Object Detection](autonomous_driving/evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)**

:   提出EVT框架，通过自适应采样与自适应投影(ASAP)实现高效的LiDAR引导视图变换，结合分组混合查询选择和几何感知交叉注意力，在nuScenes测试集上以实时速度达到75.3% NDS的SOTA性能。

**[Extrapolated Urban View Synthesis Benchmark](autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)**

:   提出首个外推式城市视图合成（EUVS）基准，利用多遍历/多车辆/多相机公开数据集系统评估外推场景下 3DGS 及 NeRF 方法的泛化能力，揭示当前方法严重过拟合训练视角。

**[Foresight In Motion Reinforcing Trajectory Prediction With Reward Heuristics](autonomous_driving/foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)**

:   提出"先推理，后预测"（First Reasoning, Then Forecasting）策略，通过基于查询中心的逆强化学习（QIRL）推断驾驶意图的奖励分布，并结合 Bi-Mamba 增强的 DETR 式轨迹解码器，显著提升轨迹预测的置信度和准确性。

**[Frequency-Aligned Knowledge Distillation For Lightweight Spatiotemporal Forecast](autonomous_driving/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)**

:   提出SDKD（频域解耦知识蒸馏）框架，通过频率感知的教师模型和频率对齐的蒸馏策略，将复杂时空预测模型的多尺度频域知识迁移到轻量级学生网络，在Navier-Stokes数据集上MSE最高降低81.3%。

**[Future-Aware Interaction Network For Motion Forecasting](autonomous_driving/future-aware_interaction_network_for_motion_forecasting.md)**

:   提出 FINet，将潜在未来轨迹提前建模并融入场景编码阶段进行联合优化，同时引入 Mamba 架构替代 Transformer 进行时空建模，实现了高效且准确的运动预测。

**[Gaussianflowocc Sparse And Weakly Supervised Occupancy Estimation Using Gaussian](autonomous_driving/gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)**

:   提出 GaussianFlowOcc，用稀疏 3D Gaussian 分布替代密集体素网格进行占用估计，通过 Gaussian Transformer 高效建模场景，引入 Temporal Module 估计每个 Gaussian 的 3D 时序流处理动态物体，在 nuScenes 上以弱监督方式大幅超越现有方法（mIoU 提升 51%+），推理速度快 50 倍。

**[Gaussrender Learning 3D Occupancy With Gaussian Rendering](autonomous_driving/gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)**

:   提出 GaussRender，一个即插即用的可微高斯渲染模块，通过将预测和真值的 3D occupancy 投影到 2D 视图并施加语义和深度一致性约束，消除浮空体素等视觉伪影，在多个 benchmark 上显著提升几何保真度，尤其在 RayIoU 等表面敏感指标上提升突出。

**[Generative Active Learning For Long-Tail Trajectory Prediction Via Controllable ](autonomous_driving/generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)**

:   提出 GALTraj，首个将生成式主动学习应用于轨迹预测的方法——在训练过程中动态识别模型失败的尾部样本，利用可控扩散模型生成保持尾部特征且符合交通规则的新样本，有效缓解长尾数据不平衡，在提升尾部性能的同时也改善整体准确性。

**[Gm-Moe Low-Light Enhancement With Gated-Mechanism Mixture-Of-Experts](autonomous_driving/gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)**

:   首次将混合专家（MoE）网络引入低光图像增强任务，通过三个专门的子专家网络分别处理颜色修复、细节增强和高级特征增强，并利用动态门控机制自适应调整各专家的权重，在5个基准数据集上取得了SOTA的PSNR表现。

**[Gs-Livm Real-Time Photo-Realistic Lidar-Inertial-Visual Mapping With Gaussian Sp](autonomous_driving/gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)**

:   提出 GS-LIVM，首个为大规模无界室外场景设计的实时光真实感 LiDAR-惯性-视觉建图框架，通过体素级高斯过程回归（Voxel-GPR）解决 LiDAR 点云稀疏不均匀问题，利用协方差中心化设计快速初始化 3D 高斯参数，在多个室外数据集上达到 SOTA 的建图效率和渲染质量。

**[Gs-Occ3D Scaling Vision-Only Occupancy Reconstruction With Gaussian Splatting](autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)**

:   提出 GS-Occ3D，一个可扩展的纯视觉 occupancy 重建框架，通过 Octree-based Gaussian Surfel 表示和地面/静态/动态三层解耦建模，实现了全 Waymo 数据集的纯视觉 occupancy 标注生成，在下游任务上达到与 LiDAR 标注可比甚至更好的零样本泛化性能。

**[Hermes A Unified Self-Driving World Model For Simultaneous 3D Scene Understandin](autonomous_driving/hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)**

:   提出 Hermes，第一个统一 3D 场景理解（VQA/描述）和未来场景生成（点云预测）的驾驶世界模型，通过 BEV 表征和 world queries 将 LLM 的世界知识注入未来场景生成，3s 点云生成误差降低 32.4%，场景理解 CIDEr 提升 8.0%。

**[Igl-Nav Incremental 3D Gaussian Localization For Image-Goal Navigation](autonomous_driving/igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)**

:   提出 IGL-Nav，基于增量式 3D 高斯表示构建可渲染场景记忆，并通过粗到精的目标定位策略高效解决图像目标导航问题，同时支持任意相机视角的自由视图设定。

**[Instinct Instance-Level Interaction Architecture For Query-Based Collaborative P](autonomous_driving/instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)**

:   提出 INSTINCT，一种基于 LiDAR 的实例级交互协作感知框架，通过质量感知过滤、双分支检测路由和跨智能体局部实例融合三个核心模块，在多个数据集上实现 SOTA 性能的同时将通信带宽降低至现有方法的 1/264~1/281。

**[Langtraj Diffusion Model And Dataset For Language-Conditioned Trajectory Simulat](autonomous_driving/langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)**

:   提出 LangTraj，首个在训练阶段直接以自然语言为条件的扩散模型轨迹仿真器，并构建了包含 150K 人工标注交互行为的 InterDrive 数据集，支持语言可控的多智能体交互仿真和安全关键场景生成。

**[Language Driven Occupancy Prediction](autonomous_driving/language_driven_occupancy_prediction.md)**

:   提出LOcc，一个有效且可泛化的开放词汇占据(OVO)预测框架，核心是设计了语义传递标注管线（LVLM+OV-Seg→LiDAR→voxel），生成密集细粒度的3D语言占据伪GT，替代了噪声大且稀疏的传统中间特征蒸馏，在Occ3D-nuScenes上全面超越SOTA。

**[Leveraging 2D Priors And Sdf Guidance For Urban Scene Rendering](autonomous_driving/leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)**

:   提出 UGSDF 方法，将 SDF 网络与 3D Gaussian Splatting 联合学习来建模动态城市场景中的物体，仅使用 2D 先验（深度网络+点跟踪器）即可实现 SOTA 渲染效果，无需 LiDAR 数据、3D 运动标注或人体模板。

**[Leveraging Bev Paradigm For Ground-To-Aerial Image Synthesis](autonomous_driving/leveraging_bev_paradigm_for_ground-to-aerial_image_synthesis.md)**

:   提出SkyDiffusion，结合Curved-BEV变换和BEV引导扩散模型，实现从地面街景图像到航拍/卫星图像的高质量跨视角合成，并引入Ground2Aerial-3多场景数据集。

**[Lightsout Diffusion-Based Outpainting For Enhanced Lens Flare Removal](autonomous_driving/lightsout_diffusion-based_outpainting_for_enhanced_lens_flare_removal.md)**

:   提出 LightsOut，一个基于扩散模型的图像外推框架，通过预测和重建画面外的光源来增强现有单图光斑去除(SIFR)方法的性能，作为即插即用的预处理模块无需额外训练即可提升任意 SIFR 模型的效果。

**[Long-Term Traffic Simulation With Interleaved Autoregressive Motion And Scenario](autonomous_driving/long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)**

:   提出 InfGen，一个统一的自回归 next-token prediction 模型，通过交替进行闭环运动仿真和场景生成（智能体的动态插入与移除），首次实现稳定的长时程（30秒）交通仿真，在短时程任务上达到 SOTA 水平，在长时程任务上显著超越所有现有方法。

**[Lookout Real-World Humanoid Egocentric Navigation](autonomous_driving/lookout_real-world_humanoid_egocentric_navigation.md)**

:   LookOut 提出从第一人称带位姿视频中预测未来 4.5 秒内的 6D 头部姿态序列（平移 + 旋转），通过将 DINOv2 特征反投影到 3D 空间再压缩为 BEV 表示来理解场景几何与语义，在自采集的 4 小时真实世界动态场景数据集上学习到等待、绕行、过马路前左右张望等类人导航行为。

**[Maestro Task-Relevant Optimization Via Adaptive Feature Enhancement And Suppress](autonomous_driving/maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)**

:   提出 MAESTRO 框架，通过类别原型生成（CPG）、任务特定特征生成（TSFG）和场景原型聚合（SPA）三个模块，在多任务3D感知中生成任务特定特征并抑制任务间干扰，在3D目标检测、BEV地图分割和3D占用预测三个任务上同时超越单任务模型。

**[Mcam Multimodal Causal Analysis Model For Ego-Vehicle-Level Driving Video Unders](autonomous_driving/mcam_multimodal_causal_analysis_model_for_ego-vehicle-level_driving_video_unders.md)**

:   提出 MCAM，通过驾驶状态有向无环图 (DSDAG) 构建视觉-语言模态间的因果结构，结合多层级特征提取和因果分析模块，用于自车级别驾驶视频理解中的行为描述与原因推理。

**[Mgsfm Multi-Camera Geometry Driven Global Structure-From-Motion](autonomous_driving/mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)**

:   提出 MGSfM，一个面向多相机系统的全局 Structure-from-Motion (SfM) 框架，通过**解耦旋转平均 (DMRA)** 和**混合平移平均 (MGP)** 两个核心模块，充分利用多相机刚性约束，在大规模场景中实现与增量式 SfM 媲美甚至更优的精度，同时速度提升约 10 倍。

**[Mixed Signals A Diverse Point Cloud Dataset For Heterogeneous Lidar V2X Collabor](autonomous_driving/mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)**

:   Mixed Signals 是首个包含异构 LiDAR 配置（不同高度和倾斜角）的真实世界 V2X 数据集，由 3 辆自动驾驶车 + 路侧单元采集，提供 4.51 万点云帧和 24.06 万标注框，同时也是首个左行交通国家（澳大利亚）的 V2X 数据集。

**[Monosowa Scalable Monocular 3D Object Detector Without Human Annotations](autonomous_driving/monosowa_scalable_monocular_3d_object_detector_without_human_annotations.md)**

:   提出首个完全不依赖人工标注（包括 2D 和 3D）的单目 3D 物体检测方法，通过新提出的局部目标运动模型（LOMM）解耦帧间运动来源，自动标注速度比前人快 700 倍，并通过规范目标空间（COS）融合不同相机设置的多数据集训练。

**[Occupancy Learning With Spatiotemporal Memory](autonomous_driving/occupancy_learning_with_spatiotemporal_memory.md)**

:   提出 ST-Occ，一个场景级时空占用表示学习框架，通过统一时序建模（Unified Temporal Modeling）范式，使用场景坐标系下的时空记忆库和具有不确定性/动态感知的记忆注意力机制，在 Occ3D 基准上比 SOTA 提升 3 mIoU，同时将时序不一致性降低 29%。

**[Od-Rase Ontology-Driven Risk Assessment And Safety Enhancement For Autonomous Dr](autonomous_driving/od-rase_ontology-driven_risk_assessment_and_safety_enhancement_for_autonomous_dr.md)**

:   提出 OD-RASE 框架，通过构建道路交通专家知识本体(ontology)来过滤 LVLM 生成的道路基础设施改善方案，实现对事故风险道路结构的前瞻性识别与改善建议生成。

**[Passing The Driving Knowledge Test](autonomous_driving/passing_the_driving_knowledge_test.md)**

:   构建DriveQA——首个大规模文本+视觉双模态驾驶知识测试基准（26K文本QA + 448K图像QA），系统评估LLM/MLLM在交通规则、标志识别和路权判断等驾驶知识上的能力，揭示其在数值推理和复杂路权场景中的显著不足，并展示DriveQA预训练对下游驾驶任务的泛化增益。

**[Pbcat Patch-Based Composite Adversarial Training Against Physically Realizable A](autonomous_driving/pbcat_patch-based_composite_adversarial_training_against_physically_realizable_a.md)**

:   提出 PBCAT（Patch-Based Composite Adversarial Training），通过结合小面积梯度引导对抗补丁和全局不可感知扰动进行对抗训练，统一防御多种物理可实现攻击（对抗补丁+对抗纹理），在行人检测任务上比之前 SOTA 防御方法提升 29.7% AP。

**[Recondreamer Harmonizing Generative And Reconstructive Models For Driving Scene ](autonomous_driving/recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)**

:   ReconDreamer++ 在 ReconDreamer 基础上，通过引入新轨迹可变形网络（NTDNet）弥合生成数据与真实观测的域差距，并独立建模地面以保留几何先验，在 Waymo 上实现了原始轨迹性能持平 Street Gaussians、新轨迹 NTA-IoU 提升 6.1%、FID 改善 23.0% 的显著效果。

**[Referring Expression Comprehension For Small Objects](autonomous_driving/referring_expression_comprehension_for_small_objects.md)**

:   提出 SOREC 数据集（10万对小目标指称表达和 bounding box）和 PIZA 适配器模块（渐进式迭代缩放），使 GroundingDINO 等预训练模型能以自回归方式逐步放大定位极小目标，在自动驾驶场景中大幅提升小目标 REC 精度。

**[Rescue Crowd Evacuation Simulation Via Controlling Sdm-United Characters](autonomous_driving/rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)**

:   提出首个在线 SDM（感知-决策-运动）统一 3D 疏散仿真框架 RESCUE，集成 3D 自适应社会力模型和个性化步态控制器，实现数百智能体的实时个性化疏散模拟。

**[Resonance Learning To Predict Social-Aware Pedestrian Trajectories As Co-Vibrati](autonomous_driving/resonance_learning_to_predict_social-aware_pedestrian_trajectories_as_co-vibrati.md)**

:   受振动系统和共振现象启发，提出 Resonance 模型将行人轨迹编码和预测建模为"共振动"形式，通过轨迹分解为独立振动分量、模拟共振现象学习社交交互表示。

**[Robotron-Sim Improving Real-World Driving Via Simulated Hard-Case](autonomous_driving/robotron-sim_improving_real-world_driving_via_simulated_hard-case.md)**

:   提出RoboTron-Sim框架，通过构建困难场景仿真数据集HASS、场景感知提示工程SPE和图像到自车编码器I2E，使MLLM有效利用仿真困难案例提升真实世界自动驾驶性能，在nuScenes困难场景下L2距离降低~48%、碰撞率降低~46%，达到开环规划SOTA。

**[Robust 3D Object Detection Using Probabilistic Point Clouds From Single-Photon L](autonomous_driving/robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)**

:   提出概率点云(PPC)表示——将单光子LiDAR原始时间直方图中的测量置信度作为概率属性附加到每个3D点上，配合轻量级NPD滤波和FPPS采样方法，实现低信噪比(SBR)下鲁棒的3D目标检测，在SUN RGB-D和KITTI上大幅超越点云去噪基线，且几乎不增加计算开销。

**[Rtmap Real-Time Recursive Mapping With Change Detection And Localization](autonomous_driving/rtmap_real-time_recursive_mapping_with_change_detection_and_localization.md)**

:   提出RTMap——首个端到端框架，同时解决多次遍历在线HD地图构建中的三大核心挑战：基于先验地图的定位、道路结构变化检测和概率感知众包地图融合，在TbV和nuScenes上同时提升地图质量和定位精度。

**[Saliency-Aware Quantized Imitation Learning For Efficient Robotic Control](autonomous_driving/saliency-aware_quantized_imitation_learning_for_efficient_robotic_control.md)**

:   提出 SQIL（Saliency-Aware Quantized Imitation Learning），通过显著性评分识别任务关键状态并在量化感知训练中加权蒸馏，使 4-bit 量化的 VLA 策略模型在机器人操控和自动驾驶中恢复全精度性能，同时实现 2.5-3.7 倍加速。

**[Sam4D Segment Anything In Camera And Lidar Streams](autonomous_driving/sam4d_segment_anything_in_camera_and_lidar_streams.md)**

:   提出 SAM4D，首个面向相机和 LiDAR 流的可提示多模态分割基础模型，通过统一多模态位置编码（UMPE）实现跨模态提示与交互，通过运动感知跨模态记忆注意力（MCMA）确保时序一致性，并构建包含 30 万+ masklet 的 Waymo-4DSeg 数据集，在跨模态分割和数据标注方面展示了强大能力。

**[Self-Supervised Sparse Sensor Fusion For Long Range Perception](autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)**

:   LRS4Fusion 提出基于稀疏体素表示的长距离LiDAR-Camera融合方法，配合自监督预训练策略（通过稀疏占用和速度场重建），在250米感知范围内实现了目标检测 mAP 提升 26.6%、LiDAR预测 Chamfer Distance 降低 30.5% 的SOTA性能。

**[Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](autonomous_driving/semantic_causality-aware_vision-based_3d_occupancy_prediction.md)**

:   从因果关系视角分析视觉3D占用预测中2D到3D变换的语义歧义问题，提出因果损失（Causal Loss）实现端到端语义一致性监督，并设计SCAT模块（通道分组提升、可学习相机偏移、归一化卷积）显著提升占用预测精度和相机扰动鲁棒性。

**[Seqgrowgraph Learning Lane Topology As A Chain Of Graph Expansions](autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)**

:   模拟人类绘图过程，将车道拓扑建模为逐步图扩展的链式序列，通过自回归变换器增量构建有向车道图，克服 DAG 方法无法表达环路和双向车道的局限。

**[Sparselanestp Leveraging Spatio-Temporal Priors With Sparse Transformers For 3D ](autonomous_driving/sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)**

:   提出 SparseLaneSTP，将车道线几何先验（平行性、连续性）和时序信息融合进稀疏 Transformer 架构，通过 Catmull-Rom 样条表示、时空注意力机制和时序正则化，在多个 3D 车道线检测基准上取得 SOTA。

**[Splat-Loam Gaussian Splatting Lidar Odometry And Mapping](autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)**

:   首个纯基于 2D Gaussian 原语的 LiDAR 里程计与建图管线，通过球面投影驱动的可微分光栅化器同时实现高精度位姿估计和轻量化场景重建。

**[Srefiner Soft-Braid Attention For Multi-Agent Trajectory Refinement](autonomous_driving/srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)**

:   提出 Soft-Braid Attention，通过"软交叉点"显式建模轨迹间和轨迹与车道间的时空拓扑关系来指导多智能体轨迹精炼，在 Argoverse v2 和 INTERACTION 两个数据集上对四种基线方法均实现显著提升，建立了轨迹精炼任务的新 SOTA。

**[Sti-Bench Are Mllms Ready For Precise Spatial-Temporal World Understanding](autonomous_driving/sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)**

:   提出 STI-Bench，一个评估多模态大语言模型（MLLM）精确时空理解能力的基准，涵盖桌面/室内/户外三大场景、8类静态+动态任务超 2000 道 QA 对，揭示当前最强 MLLM（Gemini-2.5-Pro）平均准确率仅 41.4%，在精确空间量化和时序动态理解上存在根本性不足。

**[Tars Traffic-Aware Radar Scene Flow Estimation](autonomous_driving/tars_traffic-aware_radar_scene_flow_estimation.md)**

:   提出 TARS，一种交通感知的雷达场景流估计方法，通过联合目标检测构建交通向量场（TVF），在交通层面而非实例层面捕获刚体运动，在 VOD 和专有数据集上分别超越 SOTA 15% 和 23%。

**[Towards Open-World Generation Of Stereo Images And Unsupervised Matching](autonomous_driving/towards_open-world_generation_of_stereo_images_and_unsupervised_matching.md)**

:   提出 GenStereo，一种基于扩散模型的立体图像生成框架，通过视差感知坐标嵌入、跨视图注意力和自适应融合机制，同时实现高视觉质量和高几何精度的立体图像生成，并推动无监督立体匹配达到新 SOTA。

**[Trackany3D Transferring Pretrained 3D Models For Category-Unified 3D Point Cloud](autonomous_driving/trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)**

:   TrackAny3D 首次将大规模预训练3D模型迁移到类别无关的3D单目标跟踪任务，通过双路径适配器、混合几何专家（MoGE）和时序上下文优化策略，在单一模型上实现了跨类别统一跟踪的SOTA性能。

**[Trafficloc Localizing Traffic Surveillance Cameras In 3D Scenes](autonomous_driving/trafficloc_localizing_traffic_surveillance_cameras_in_3d_scenes.md)**

:   提出 TrafficLoc，一种粗到细的图像-点云配准方法，通过几何引导注意力损失(GAL)、模态间-模态内对比学习(ICL)和稠密训练对齐(DTA)，实现交通监控相机在3D参考地图中的高精度定位，在自建 Carla Intersection 数据集上较 SOTA 提升达 86%。

**[Uavscenes A Multi-Modal Dataset For Uavs](autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)**

:   UAVScenes 是首个同时提供逐帧图像和 LiDAR 点云语义标注及精确 6-DoF 位姿的大规模多模态无人机数据集，包含超 12 万帧标注数据，支持语义分割、深度估计、定位、场景识别和新视角合成等六类感知任务。

**[Uniocc A Unified Benchmark For Occupancy Forecasting And Prediction In Autonomou](autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)**

:   提出 UniOcc，首个统一的 2D/3D 占据预测与预报基准，整合 nuScenes、Waymo、CARLA 和 OpenCOOD 四个数据源，引入逐体素流标注和不依赖真值标签的评估指标，通过大规模实验揭示了体素级流信息和跨域训练对占据任务的重要价值。

**[Unleashing The Temporal Potential Of Stereo Event Cameras For Continuous-Time 3D](autonomous_driving/unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)**

:   提出首个仅依赖立体事件相机的 3D 目标检测框架，通过语义-几何双重滤波模块和目标中心 ROI 对齐，在 blind time 期间实现连续时间 3D 检测，在动态大运动场景下显著优于依赖同步传感器的方法（Ev-3DOD），行人 AP3D 甚至超越使用 LiDAR+RGB+Event 的方法。

**[Unraveling The Effects Of Synthetic Data On End-To-End Autonomous Driving](autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)**

:   提出 SceneCrafter，一个基于 3DGS 的统一仿真框架，通过自适应运动学模型和双向交互式智能体控制，同时支持合成数据生成和闭环评估，实验证明合成数据可显著提升端到端自动驾驶模型的泛化能力（Route Completion 提升 18%）。

**[Wavelet Policy Lifting Scheme For Policy Learning In Long-Horizon Tasks](autonomous_driving/wavelet_policy_lifting_scheme_for_policy_learning_in_long-horizon_tasks.md)**

:   Wavelet Policy 首次将小波分析引入具身智能的策略学习，设计了基于可学习提升方案（lifting scheme）的多尺度策略网络，通过将观测序列分解为不同频率分量后逐层合成动作序列，在自动驾驶（CARLA）、机器人操作、多机器人协作等5个长horizon任务上取得了优于或持平基线的性能。

**[Where Am I Cross-View Geo-Localization With Natural Language Descriptions](autonomous_driving/where_am_i_cross-view_geo-localization_with_natural_language_descriptions.md)**

:   引入基于自然语言描述的跨视角地理定位新任务，构建覆盖3个城市3万+坐标的CVG-Text多模态数据集（街景+卫星+OSM+文本），并提出CrossText2Loc方法——通过扩展位置嵌入处理长文本和可解释检索模块提供定位理由，Top-1召回率提升超10%。

**[Where What Why Towards Explainable Driver Attention Prediction](autonomous_driving/where_what_why_towards_explainable_driver_attention_prediction.md)**

:   本文提出了"可解释驾驶员注意力预测"新范式，构建了首个大规模 W³DA 数据集并设计了 LLada 框架，将空间注意力预测（Where）、语义解析（What）和认知推理（Why）统一在一个端到端的大语言模型驱动架构中。

**[World4Drive End-To-End Autonomous Driving Via Intention-Aware Physical Latent Wo](autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)**

:   构建意图感知的潜在世界模型 World4Drive，利用视觉基础模型的空间-语义先验，在无感知标注条件下实现端到端规划，L2误差降低18.1%，碰撞率降低46.7%。

---

## ✂️ 语义分割 { #segmentation }

**[2Handedafforder Learning Precise Actionable Bimanual Affordances From Human Vide](segmentation/2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)**

:   本文提出从人类活动视频中自动提取精确的双手可操作区域(affordance)数据集 2HANDS，并训练基于 VLM 的 2HandedAfforder 模型，实现根据文本提示预测双手抓握的精确物体区域分割，在新提出的 ActAffordance 基准上显著优于现有方法。

**[A Plug-And-Play Physical Motion Restoration Approach For In-The-Wild High-Diffic](segmentation/a_plug-and-play_physical_motion_restoration_approach_for_in-the-wild_high-diffic.md)**

:   提出即插即用的物理动作恢复方法，通过Mask条件运动修正模块（MCM）修复视频动捕中的瑕疵帧，以及基于预训练+测试时适应的物理运动迁移模块（PTM）实现高难度野外动作的物理仿真，显著提升动作的物理真实性。

**[A Plugandplay Physical Motion Restoration Approach For Inthe](segmentation/a_plugandplay_physical_motion_restoration_approach_for_inthe.md)**

:   提出即插即用的物理运动修复框架，通过Mask条件运动校正模块（MCM）修复视频运动捕捉中的缺陷帧，结合基于RL测试时适应的物理运动传输模块（PTM），首次实现对野外高难度运动（如体操、武术后空翻）的物理仿真修复。

**[Advancing Visual Large Language Model For Multi-Granular Versatile Perception](segmentation/advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)**

:   本文提出 MVP-LM，一个基于视觉大语言模型的多粒度通用感知框架，通过创新的多粒度解码器和 CoT 启发的数据统一策略，首次在单一模型中同时支持词级/句级指令下的框/掩膜预测四种感知组合，在全景分割、目标检测、视觉定位和指示表达分割等任务上取得有竞争力的性能。

**[Animalclue Recognizing Animals By Their Traces](segmentation/animalclue_recognizing_animals_by_their_traces.md)**

:   提出 AnimalClue，首个大规模动物痕迹识别数据集，包含 159,605 个边界框覆盖 968 个物种的五类间接线索（脚印、粪便、蛋、骨骼、羽毛），并建立了分类、检测、实例分割和特征预测四项基准。

**[Auto-Vocabulary Semantic Segmentation](segmentation/auto-vocabulary_semantic_segmentation.md)**

:   本文提出 Auto-Vocabulary Semantic Segmentation (AVS) 新任务，通过 AutoSeg 框架自动从图像中发现目标类别并分割，无需人为指定词汇表，在 PASCAL VOC 上达到 87.1 mIoU，远超唯一同类方法 ZeroSeg (20.1)，甚至超越部分需要指定类别的开放词汇方法。

**[Beyond Single Images Retrieval Self-Augmented Unsupervised Camouflaged Object De](segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)**

:   本文提出 RISE——一种检索自增强的无监督伪装目标检测范式，通过从训练集本身构建前景/背景原型库并利用 KNN 检索生成伪标签，在无任何标注的条件下大幅超越现有无监督和基于提示的方法。

**[Can Generative Geospatial Diffusion Models Excel As Discriminative Geospatial Fo](segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)**

:   提出SatDiFuser框架，将生成式地理空间扩散模型（DiffusionSat）转化为判别式遥感基础模型，通过系统分析多阶段多时间步扩散特征并设计三种融合策略（全局加权、局部加权、MoE联合融合），在语义分割和分类任务上优于现有SOTA遥感基础模型，最高提升+5.7% mIoU和+7.9% F1。

**[Cavis Context-Aware Video Instance Segmentation](segmentation/cavis_context-aware_video_instance_segmentation.md)**

:   提出CAVIS，通过引入上下文感知实例追踪器（CAIT）融合物体边界周围的上下文信息来增强实例关联，并设计原型化跨帧对比损失（PCC）保证跨帧特征一致性，在VIS和VPS任务上全面刷新SOTA。

**[Clot Closed Loop Optimal Transport For Unsupervised Action Segmentation](segmentation/clot_closed_loop_optimal_transport_for_unsupervised_action_segmentation.md)**

:   提出闭环最优传输（CLOT）框架，通过三级循环特征学习（帧嵌入→段嵌入→交叉注意力精化帧嵌入）联合求解三个OT问题，在帧级和段级表征之间建立显式反馈循环，显著提升无监督动作分割的边界检测和聚类质量。

**[Conformalsam Unlocking The Potential Of Foundational Segmentation Models In Semi](segmentation/conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)**

:   提出ConformalSAM框架，利用Conformal Prediction校准基础分割模型SEEM在目标域的输出不确定性，筛除不可靠像素标签后作为未标注数据的监督信号，配合后期自依赖训练策略，在PASCAL VOC上1/16标注设定下达到81.21 mIoU。

**[Corrclip Reconstructing Patch Correlations In Clip For Openv](segmentation/corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)**

:   揭示CLIP用于分割时patch间"类间相关性"是性能瓶颈的根本原因，提出CorrCLIP通过SAM限制patch交互范围（scope reconstruction）+DINO计算更一致的相似度值（value reconstruction）+空间/语义特征增强+SAM mask后处理，在8个benchmark上training-free方法平均mIoU从48.6%提升到53.6%。

**[Correspondence As Video Test-Time Adaption On Sam2 For Reference Segmentation In](segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)**

:   CAV-SAM 将参考-目标图像对之间的对应关系表示为伪视频序列，通过基于扩散模型的语义过渡模块（DBST）桥接语义差异，以及测试时几何对齐模块（TTGA）对齐几何变化，使SAM2的视频分割能力零训练地适配参考分割任务，在跨域少样本分割基准上超越SOTA约5% mIoU。

**[Correspondence As Video Testtime Adaption On Sam2 For Refere](segmentation/correspondence_as_video_testtime_adaption_on_sam2_for_refere.md)**

:   将reference-target图像对之间的对应关系表示为用扩散模型生成的伪视频序列，利用SAM2的iVOS能力进行分割，结合test-time轻量微调对齐几何变化，在跨域few-shot分割上比SOTA方法提升约5% mIoU，且无需meta-training。

**[Ddb Diffusion Driven Balancing To Address Spurious Correlations](segmentation/ddb_diffusion_driven_balancing_to_address_spurious_correlations.md)**

:   提出Diffusion Driven Balancing（DDB）方法，利用Stable Diffusion的文本反演和图像修复能力，自动生成少数组样本来平衡数据集中的虚假相关性，结合基于ERM模型预测概率和积分梯度的双重剪枝策略确保生成质量，在Waterbirds和MetaShift上达到最优最差组准确率。

**[Deris Decoupling Perception And Cognition For Enhanced Referring Image Segmentat](segmentation/deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)**

:   提出DeRIS框架，将指代图像分割任务解耦为感知（perception）和认知（cognition）两个分支，通过回环协同（Loopback Synergy）机制迭代增强两分支的交互，并引入非指代样本转换增强策略，在RefCOCO/+/g和gRefCOCO数据集上取得SOTA。

**[E-Sam Training-Free Segment Every Entity Model](segmentation/e-sam_training-free_segment_every_entity_model.md)**

:   E-SAM 是一个无需额外训练的框架，通过三个级联模块——多层级掩码生成（MMG）、实体级掩码精炼（EMR）和欠分割修复（USR）——系统性地解决 SAM 自动掩码生成（AMG）中的过分割和欠分割问题，在基准指标上超越现有实体分割方法 **+30.1 分**。

**[Enhancing Transformers Through Conditioned Embedded Tokens](segmentation/enhancing_transformers_through_conditioned_embedded_tokens.md)**

:   揭示 Transformer 自注意力矩阵存在固有的 ill-conditioning 问题，通过理论分析建立自注意力条件数与嵌入令牌条件数的直接关系，提出 Conditioned Embedded Tokens 方法（对嵌入矩阵添加基于 SVD 的修正项），在图像分类、目标检测、实例分割和 NLP 等多种任务上一致提升性能。

**[Exploiting Domain Properties In Language-Driven Domain Generalization For Semant](segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)**

:   提出DPMFormer框架，通过域感知提示学习将输入图像的域特有属性转化为文本上下文提示，并结合域鲁棒一致性学习，解决语言驱动域泛化语义分割中视觉与文本上下文的语义错位问题。

**[Exploring Probabilistic Modeling Beyond Domain Generalization For Semantic Segme](segmentation/exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)**

:   提出 PDAF（概率扩散对齐框架），通过概率扩散建模显式估计潜在域先验（LDP），为现有分割网络提供域偏移补偿，在不需要目标域配对样本的情况下实现跨域泛化的 SOTA 性能。

**[Floss Free Lunch In Openvocabulary Semantic Segmentation](segmentation/floss_free_lunch_in_openvocabulary_semantic_segmentation.md)**

:   挑战OVSS中"平均80个模板"的默认做法，发现每个类别存在特定的"专家模板"（class-expert）远优于平均分类器，提出用预测熵无监督选择专家模板+融合专家预测的FLOSS方法，在不需要标签和训练的情况下一致提升现有OVSS方法。

**[Harnessing Massive Satellite Imagery With Efficient Masked Image Modeling](segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)**

:   提出一个遥感模型预训练流水线，包括 1300 万张光学遥感图像数据集 OpticalRS-13M 和基于语义丰富度选择性编码/重建的高效 MIM 方法 SelectiveMAE，仅用 40% 图像 patch 即可训练出与全量 patch 相当的模型，同时实现 2 倍以上加速。

**[Hierarchical Visual Prompt Learning For Continual Video Instance Segmentation](segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)**

:   提出持续视频实例分割（CVIS）新问题，设计分层视觉提示学习（HVPL）模型，通过帧级和视频级两个层面的遗忘补偿机制，有效缓解旧类别的灾难性遗忘。

**[Himtok Learning Hierarchical Mask Tokens For Image Segmentation With Large Multi](segmentation/himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)**

:   提出HiMTok（分层掩码Token化器），将分割掩码表示为最多32个由粗到细的离散token，使LMM像生成文本一样直接生成分割结果，无需额外的图像条件掩码解码器，在多个分割基准上达到SOTA。

**[How Do Optical Flow And Textual Prompts Collaborate To Assist In Audio-Visual Se](segmentation/how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)**

:   提出 SSP (Stepping Stone Plus) 框架，将光流作为辅助掩码提示与两类文本提示协同工作，配合视觉-文本对齐模块 (VTA)，在音视频语义分割任务中实现 SOTA 性能。

**[Hybrid-Tta Continual Test-Time Adaptation Via Dynamic Domain Shift Detection](segmentation/hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)**

:   Hybrid-TTA 提出一种持续测试时自适应（CTTA）框架，通过动态域偏移检测（DDSD）模块判断当前输入是否来自新域，自适应地在全参数微调（Full Tuning）和高效微调（Adapter Tuning）之间切换；同时引入掩码图像建模自适应（MIMA）作为辅助任务增强模型稳定性，在 Cityscapes-to-ACDC 基准上达到 62.2% mIoU，且推理速度比可比方法快约 **20 倍**。

**[Inter2Former Dynamic Hybrid Attention For Efficient High-Precision Interactive S](segmentation/inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)**

:   提出 Inter2Former，通过动态混合注意力（DHA）将边界 token 路由到全注意力、非边界 token 路由到线性复杂度的 BSQ 注意力，配合动态提示嵌入（DPE）、混合专家（HMoE）和动态局部上采样（DLU），在 CPU 设备上实现高精度交互式分割的 SOTA 性能与高效推理。

**[Joint Self-Supervised Video Alignment And Action Segmentation](segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)**

:   提出 VAOT/VASOT 框架，基于融合 Gromov-Wasserstein 最优传输和结构先验，首次将自监督视频对齐和动作分割统一到单一模型中，视频对齐性能优于现有方法，动作分割也达到 SOTA。

**[Know No Better A Data-Driven Approach For Enhancing Negation Awareness In Clip](segmentation/know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)**

:   通过分析 CLIP 预训练数据中否定表达的稀缺和错位问题，设计两条基于 LLM/MLLM 的否定数据生成管线来微调 CLIP 文本编码器，开发出 NegationCLIP，在增强否定理解能力的同时保持通用性能，并提出 NegRefCOCOg 基准用于全面评估否定理解。

**[Know Your Attention Maps Class-Specific Token Masking For Weakly Supervised Sema](segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)**

:   提出一种端到端的弱监督语义分割方法，通过在 ViT 中引入多个 [CLS] token（每个类别一个）、对 [CLS] token 输出嵌入进行随机掩码以及剪枝冗余注意力头，直接利用自注意力图生成类别特定的伪分割掩码，无需额外的 CAM 模块。

**[Latent Expression Generation For Referring Image Segmentation And Grounding](segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)**

:   提出 Latent-VG 框架，通过从单个文本描述生成多个潜在表达式（共享同一主语、但具有不同视觉属性），利用互补的视觉细节弥补稀疏文本与丰富视觉信息之间的语义差距，在指代图像分割和指代表达理解任务上同时达到 SOTA。

**[Lawdis Language-Window-Based Controllable Dichotomous Image Segmentation](segmentation/lawdis_language-window-based_controllable_dichotomous_image_segmentation.md)**

:   提出 LawDIS，一个基于潜在扩散模型的可控二分图像分割框架，通过宏观语言控制（LS）和微观窗口细化（WR）两种模式的协同，实现高质量前景目标掩码生成，在 DIS5K 基准上全面超越 11 种 SOTA 方法。

**[Layeranimate Layer-Level Control For Animation](segmentation/layeranimate_layer-level_control_for_animation.md)**

:   提出LayerAnimate框架，将传统动画生产中的图层分离理念与视频扩散模型结合，实现图层级别的精细控制（运动分数、轨迹、草图），并设计自动化数据策划pipeline解决图层数据稀缺问题，在6种视频生成任务中全面超越现有方法。

**[Learn2Synth Learning Optimal Data Synthesis Using Hypergradients For Brain Image](segmentation/learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)**

:   提出Learn2Synth训练框架，通过超梯度（hypergradients）学习最优的合成数据增强参数，使在合成数据上训练的分割网络在真实数据上达到最优精度，兼顾域内高精度和域外强泛化，在脑MRI分割任务中全面超越SynthSeg和监督学习基线。

**[Learning Precise Affordances From Egocentric Videos For Robotic Manipulation](segmentation/learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)**

:   提出一套完整的 affordance 学习系统：(1) 从第一人称视频自动提取精确的可抓取/功能性 affordance 分割标注，(2) 基于 DINOv2 + 深度几何引导的 GAT 模型实现跨域 affordance 分割（mIoU 提升 13.8%），(3) Aff-Grasp 框架在 179 次真实机器人试验中达到 77.1% 抓取成功率。

**[Legion Learning To Ground And Explain For Synthetic Image Detection](segmentation/legion_learning_to_ground_and_explain_for_synthetic_image_detection.md)**

:   提出 LEGION 框架和 SynthScars 数据集，利用多模态大语言模型（MLLM）实现合成图像的伪影检测、像素级分割和文本解释三位一体，并创新性地将检测器从"防御者"扩展为"控制者"，引导生成模型产出更高质量的图像。

**[Legrad An Explainability Method For Vision Transformers Via Feature Formation Se](segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)**

:   提出LeGrad——一种专为ViT设计的逐层可解释性方法，通过计算激活值对各层注意力图的梯度作为解释信号，并跨层聚合以生成高质量的空间显著性图，在分割、扰动和开放词汇场景均展现出优越的空间保真度。

**[Move Motion-Guided Few-Shot Video Object Segmentation](segmentation/move_motion-guided_few-shot_video_object_segmentation.md)**

:   本文提出运动引导的少样本视频目标分割新任务及大规模数据集 MOVE（224 类运动、4300 视频、314K mask），并设计解耦运动-外观网络 DMA，通过帧差提取运动原型+外观原型的双分支架构，在新基准上显著优于现有 FSVOS 方法。

**[O-Mama Learning Object Mask Matching Between Egocentric And Exocentric Views](segmentation/o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)**

:   将跨视角（ego-exo）物体分割任务重新定义为 mask matching 问题，利用 FastSAM 生成候选 mask、DINOv2 提取语义特征、对比学习匹配跨视角物体，在 Ego-Exo4D 基准上以仅 1% 可训练参数实现 SOTA。

**[Object-Level Correlation For Few-Shot Segmentation](segmentation/object-level_correlation_for_few-shot_segmentation.md)**

:   提出 OCNet，通过模仿生物视觉过程构建**目标级别**（而非图像级别）的 support-query 关联，先挖掘查询图像中的通用物体，再从中识别目标物体，有效抑制背景中的无关物体噪声。

**[Omnisam Omnidirectional Segment Anything Model For Uda In Panoramic Semantic Seg](segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)**

:   提出 OmniSAM，首次将 SAM2 应用于全景语义分割的无监督域适应任务，通过滑动窗口将全景图切分为 patch 序列并利用 SAM2 的记忆机制捕获跨 patch 对应关系，结合 FoV-based 原型自适应和动态伪标签更新策略，在室内外场景均大幅超越 SOTA（+10.22% / +6.58%）。

**[On The Generalization Of Representation Uncertainty In Earth Observation](segmentation/on_the_generalization_of_representation_uncertainty_in_earth_observation.md)**

:   系统研究了预训练表示不确定性在地球观测（EO）领域的泛化能力，发现 EO 预训练的不确定性在不同地理位置、EO 任务和目标粒度上具备强泛化能力，同时对地面采样距离（GSD）高度敏感。

**[Online Generic Event Boundary Detection](segmentation/online_generic_event_boundary_detection.md)**

:   本文提出在线通用事件边界检测（On-GEBD）这一新任务——在流式视频中实时检测事件边界，并设计了基于认知科学事件分割理论（EST）的 ESTimator 框架，通过一致事件预测器（CEA）和在线边界判别器（OBD）的协同，在 Kinetics-GEBD 上 Avg F1 达到 0.748，超越所有在线基线且接近离线方法的性能。

**[Online Reasoning Video Segmentation With Just-In-Time Digital Twins](segmentation/online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)**

:   提出一种基于"即时数字孪生(Just-in-Time Digital Twin)"概念的多智能体框架，将感知和推理解耦，无需 LLM 微调即可实现在线视频推理分割，在语义、空间、时间三类推理任务中全面超越现有方法。

**[Open-World Skill Discovery From Unsegmented Demonstration Videos](segmentation/open-world_skill_discovery_from_unsegmented_demonstration_videos.md)**

:   受人类认知事件分割理论（EST）启发，提出 Skill Boundary Detection (SBD) 算法，利用预训练无条件动作预测模型的**预测误差跳变**来自动识别未分割演示视频中的技能边界，在 Minecraft 中显著提升条件策略和层级智能体的表现。

**[Partfield Learning 3D Feature Fields For Part Segmentation And Beyond](segmentation/partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)**

:   PartField 通过前馈模型学习连续 3D 特征场，用对比学习从混合的 2D/3D 部件提案中蒸馏知识，在类别无关的 3D 部件分割上比现有方法精度提高 20%+ 同时推理速度快数个数量级。

**[Prompt Guidance And Human Proximal Perception For Hot Prediction With Regional J](segmentation/prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)**

:   提出 P3HOT 框架，通过文本 prompt 引导关注人体接触部位、深度感知模块过滤无关背景、以及 Regional Joint Loss 保证区域内类别一致性，在 HOT（Human-Object Contact）检测任务上取得 SOTA。

**[Ragnet Large-Scale Reasoning-Based Affordance Segmentation Benchmark Towards Gen](segmentation/ragnet_large-scale_reasoning-based_affordance_segmentation_benchmark_towards_gen.md)**

:   构建了首个大规模推理式 affordance 分割基准 RAGNet（273k 图像、180 类别、26k 推理指令），并提出 AffordanceNet 框架，将 VLM 预训练的 affordance 预测与抓取姿态生成相结合，展现出强大的开放世界泛化和推理能力。

**[Refer To Any Segmentation Mask Group With Vision-Language Prompts](segmentation/refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)**

:   提出全模态指代表达分割（ORES）任务及 RAS 框架，通过掩码级 LMM 和非自回归解码机制，根据视觉-语言混合提示从候选掩码中选择目标掩码组，在新 ORES 数据集及经典 RES/GRES 基准上取得 SOTA。

**[Referdino Referring Video Object Segmentation With Visual Grounding Foundations](segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)**

:   提出ReferDINO，通过将GroundingDINO视觉定位基础模型端到端适配到指代视频目标分割（RVOS）任务，设计定位引导可变形掩码解码器、目标一致性时序增强器和置信度查询剪枝策略，在五个基准上显著超越SOTA（Ref-YouTube-VOS上+3.9% $\mathcal{J}\&\mathcal{F}$），并实现51 FPS实时推理。

**[Refereverything Towards Segmenting Everything We Can Speak Of In Videos](segmentation/refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)**

:   利用视频扩散模型中学到的通用视觉-语言映射，通过保留完整生成模型架构并将目标从预测噪声转变为预测掩码潜变量，实现对视频中任意可用语言描述的概念（包括非物体的动态过程）进行开放世界指代分割。

**[Region-Based Cluster Discrimination For Visual Representation Learning](segmentation/region-based_cluster_discrimination_for_visual_representation_learning.md)**

:   提出 RICE（Region-Aware Cluster Discrimination），通过构建十亿级区域数据集、设计 Region Transformer 层和统一区域聚类判别损失，联合优化目标感知和 OCR 能力，显著提升视觉编码器在分割、检测和 MLLM 多任务上的表现。

**[Rethinking Detecting Salient And Camouflaged Objects In Unconstrained Scenes](segmentation/rethinking_detecting_salient_and_camouflaged_objects_in_unconstrained_scenes.md)**

:   构建首个无约束显著性和伪装目标检测数据集USC12K（覆盖四种场景类型），提出基于SAM的USCNet网络，通过属性关系建模（ARM）模块显式建模显著和伪装目标的关系，并设计新指标CSCS衡量混淆程度，在所有场景中达到SOTA。

**[Roadwork A Dataset And Benchmark For Learning To Recognize Observe Analyze And D](segmentation/roadwork_a_dataset_and_benchmark_for_learning_to_recognize_observe_analyze_and_d.md)**

:   提出首个大规模施工区域（work zone）数据集ROADWork，涵盖4375段视频、9650张丰富标注图像和129K带路径图像，揭示基础模型在施工场景下严重失效（AP仅2.9-4.2），微调后性能大幅提升（+32.2 AP），并提出识别、观察、分析、驾驶四层认知框架。

**[Sam2Long Enhancing Sam 2 For Long Video Segmentation With A](segmentation/sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)**

:   针对SAM 2在长视频中因贪心选择策略导致的错误累积问题，提出一种training-free的约束树搜索记忆策略，维护多条分割路径并在视频级别选择最优结果，在9个VOS和3个VOT benchmark上平均提升3.7 J&F，长视频场景最高提升5.3。

**[Score Scene Context Matters In Open-Vocabulary Remote Sensing Instance Segmentat](segmentation/score_scene_context_matters_in_open-vocabulary_remote_sensing_instance_segmentat.md)**

:   提出 SCORE 框架，利用多粒度场景上下文（区域上下文+全局上下文）增强开放词汇遥感实例分割，通过 Region-Aware Integration 和 Global Context Adaptation 两个模块分别强化视觉和文本表示。

**[Score Scene Context Matters In Openvocabulary Remote Sensing](segmentation/score_scene_context_matters_in_openvocabulary_remote_sensing.md)**

:   提出SCORE框架，通过引入区域上下文（RAI）和全局上下文适配（GCA）两个模块，将遥感专用CLIP的多粒度场景知识注入到开放词汇实例分割pipeline中，在多个遥感数据集上的跨数据集评估中平均mAP超越前SOTA 5.53%。

**[Skeleton Motion Words For Unsupervised Skeleton-Based Temporal Action Segmentati](segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)**

:   提出 Skeleton Motion Quantization (SMQ) 方法，通过关节解耦的时序自编码器和骨架运动词量化模块，实现无监督骨架序列时序动作分割，在 HuGaDB、LARa 和 BABEL 三个数据集上大幅超越现有无监督方法。

**[Spade Spatial-Aware Denoising Network For Open-Vocabulary Panoptic Scene Graph G](segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)**

:   提出SPADE——一种面向开放词汇全景场景图生成（PSG）的空间感知去噪网络，通过DDIM逆向校准将预训练扩散模型适配为PSG特定的空间先验提取器，并设计关系图Transformer捕获长程和局部上下文，在闭集和开集场景中均大幅超越SOTA，尤其在空间关系预测上表现突出。

**[Stepping Out Of Similar Semantic Space For Open-Vocabulary Segmentation](segmentation/stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)**

:   揭示现有开放词汇分割（OVS）测试集与训练语义空间高度相似的评估偏差，提出新基准 OpenBench 和方法 OVSNet，通过梯度无关聚合（GFA）融合异构特征和代理校准（PC）零成本扩展训练空间，在已有基准和 OpenBench 上均取得 SOTA。

**[Tavis Text-Bridged Audio-Visual Segmentation With Foundation Models](segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)**

:   提出 TAViS，一种文本桥接的音频-视觉分割框架，通过耦合 ImageBind 的跨模态对齐能力与 SAM2 的精确分割能力，引入文本桥接的混合提示机制和对齐监督策略，在单源、多源、语义及零样本分割场景上均取得 SOTA 性能。

**[Temporal Rate Reduction Clustering For Human Motion Segmentation](segmentation/temporal_rate_reduction_clustering_for_human_motion_segmentation.md)**

:   提出 Temporal Rate Reduction Clustering (TR²C) 方法，将最大编码率约简（MCR²）原理与时序连续性正则化相结合，联合学习符合子空间联合（UoS）分布的时序一致表示与亲和度矩阵，在五个基准上大幅刷新人体运动分割 SOTA。

**[Tinyvim Frequency Decoupling For Tiny Hybrid Vision Mamba](segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)**

:   提出 TinyViM，一种基于频率解耦的轻量级卷积-Mamba 混合视觉骨干，通过拉普拉斯混合器将低频分量输入 Mamba 建模全局上下文、高频分量用深度卷积增强，配合频率斜坡 Inception 结构逐层调节频率配比，在分类/检测/分割任务上以 2-3 倍吞吐量超越现有 Mamba 模型。

**[Topotta Topology-Enhanced Test-Time Adaptation For Tubular Structure Segmentatio](segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)**

:   首个针对管状结构分割（TSS）的测试时适应（TTA）框架，通过拓扑元差分卷积（TopoMDCs）适应跨域拓扑结构差异，并通过拓扑硬样本生成（TopoHG）策略修复拓扑连续性断裂，在10个数据集上平均clDice提升31.81%。

**[Towards Omnimodal Expressions And Reasoning In Referring Audio-Visual Segmentati](segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)**

:   提出 OmniAVS 数据集和 OISA 模型，将指代音频-视觉分割从简单声学属性感知拓展至**全模态表达（文本/语音/声音/图像的任意组合）**和**深度推理（理解声音内容+世界知识）**，在新基准及多个相关任务上取得 SOTA。

**[Training-Free Class Purification For Open-Vocabulary Semantic Segmentation](segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)**

:   提出 FreeCP，一种无需训练的类别净化框架，通过冗余净化和歧义净化两阶段策略，解决开放词汇语义分割中因过完备词汇表导致的类别冗余和视觉-语言歧义问题，作为即插即用模块在八个基准上显著提升现有方法性能。

**[Understanding Personal Concept In Open-Vocabulary Semantic Segmentation](segmentation/understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)**

:   首次提出个性化开放词汇语义分割（Personalized OVSS）任务，设计基于文本提示调优的即插即用方法，通过"负掩码提案"抑制假阳性和视觉嵌入注入丰富个性化概念表征，仅用少量图像-掩码对即可识别用户感兴趣的特定物体实例，同时保持原有OVSS性能。

**[Uniglyph Unified Segmentation-Conditioned Diffusion For Precise Visual Text Synt](segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)**

:   提出 UniGlyph，一种以分割掩码为统一条件信号的视觉文本生成框架，通过自适应字形条件（AGC）和字形区域损失（GRL）替代传统的渲染字形条件，实现单一 ControlNet 架构下中英文文字图像生成的 SOTA，尤其在小字体和复杂排版场景大幅领先。

**[Veggie Instructional Editing And Reasoning Video Concepts With Grounded Generati](segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)**

:   VEGGIE 提出了一个端到端统一框架，将 MLLM 与视频扩散模型连接，仅用扩散损失就能在单一模型中同时完成指令式视频编辑、概念定位和推理分割等 8 种任务。

**[Vsc Visual Search Compositional Text-To-Image Diffusion Model](segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)**

:   提出 VSC，一种基于视觉搜索的组合文本到图像扩散生成方法，通过为每个属性-对象对单独生成参考图像并融合视觉原型嵌入，结合分割引导的交叉注意力定位训练，显著提升多属性-对象绑定的准确性和扩展性。

**[Vssd Vision Mamba With Non-Causal State Space Duality](segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)**

:   提出非因果状态空间对偶（NC-SSD），通过保留 token 贡献的相对权重取代隐状态的累积衰减，将 Mamba2 的 SSD 无缝转化为非因果形式，构建 VSSD 视觉骨干，在分类/检测/分割多任务上超越现有 SSM 模型，同时训练速度提升 20%-50%。

**[What If Understanding Motion Through Sparse Interactions](segmentation/what_if_understanding_motion_through_sparse_interactions.md)**

:   提出 Flow Poke Transformer (FPT)，直接预测场景中物体运动的**多模态概率分布**（而非单一确定性结果），通过稀疏"戳动 (poke)"交互条件化，实现可解释的运动理解和运动部件分割。

**[Zim Zero-Shot Image Matting For Anything](segmentation/zim_zero-shot_image_matting_for_anything.md)**

:   提出ZIM——一种零样本图像抠图模型，通过标签转换器将SA1B分割标签转为精细抠图标签构建SA1B-Matte数据集，并设计层次像素解码器和提示感知遮罩注意力机制，在保持零样本泛化能力的同时实现微观级精细抠图。

---

## 🧑 人体理解 { #human_understanding }

**[A Quality-Guided Mixture Of Score-Fusion Experts Framework For Human Recognition](human_understanding/a_quality-guided_mixture_of_score-fusion_experts_framework_for_human_recognition.md)**

:   本文提出 QME（Quality-guided Mixture of score-fusion Experts）框架，通过可学习的分数融合策略和基于模态质量的 MoE 路由机制，动态整合人脸、步态和行人重识别等多模态生物特征的相似度分数，在多个全身识别基准上实现了 SOTA 性能。

**[A Qualityguided Mixture Of Scorefusion Experts Framework For](human_understanding/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)**

:   提出 Quality-guided Mixture of score-fusion Experts (QME) 框架，通过质量引导的 MoE 策略对来自不同生物特征模态（人脸、步态、身体）的相似度分数进行可学习融合，配合伪质量损失和分数三元组损失，在多个全身生物特征识别基准上达到 SOTA。

**[Align Your Rhythm Generating Highly Aligned Dance Poses With Gating-Enhanced Rhy](human_understanding/align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)**

:   提出Danceba框架，通过基于相位的节奏提取（PRE）、时序门控因果注意力（TGCA）和并行Mamba运动建模（PMMM）三个核心模块，实现音乐驱动的高节奏对齐、高多样性舞蹈生成，在AIST++数据集上FIDk提升48.68%、BAS提升12%。

**[Ar-Vrm Imitating Human Motions For Visual Robot Manipulation With Analogical Rea](human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)**

:   提出 AR-VRM，首个通过显式模仿人类手部关键点来增强视觉机器人操控的方法，采用关键点视觉语言模型预训练从大规模人类动作视频中学习动作知识，并通过类比推理(Analogical Reasoning)建立人手关键点与机器人组件的映射。

**[Avat3R Large Animatable Gaussian Reconstruction Model For Hi](human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)**

:   提出Avat3r——首个可动画的大型3D重建模型(LRM)，仅需4张输入图像即可在前馈方式下回归出高质量可驱动的3D高斯头部头像，通过整合DUSt3R位置图和Sapiens语义特征作为先验、并用简单的cross-attention建模表情动画，在Ava256和NeRSemble数据集上大幅超越现有方法。

**[Bi-Level Optimization For Self-Supervised Ai-Generated Face Detection](human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)**

:   提出BLADES方法，通过双层优化（bi-level optimization）将自监督预训练与AI生成人脸检测目标显式对齐：内层优化视觉编码器学习EXIF分类/排序和人脸篡改检测等前置任务，外层优化各任务权重以提升代理检测任务性能，实现不依赖合成人脸的跨生成器泛化检测。

**[Cargait Cross-Attention Based Re-Ranking For Gait Recognition](human_understanding/cargait_cross-attention_based_re-ranking_for_gait_recognition.md)**

:   提出CarGait，一种基于交叉注意力的步态识别重排序方法，通过probe与候选序列之间的strip-wise交叉注意力学习细粒度的步态对应关系，将预训练单阶段模型的全局特征映射到新的判别性嵌入空间，在Gait3D、GREW和OU-MVLP三大基准上对七种步态模型均取得一致的Rank-1/5精度提升。

**[Cargait Crossattention Based Reranking For Gait Recognition](human_understanding/cargait_crossattention_based_reranking_for_gait_recognition.md)**

:   提出CarGait——基于cross-attention的步态识别重排序方法：对任意单阶段步态模型的top-K检索结果，通过probe与候选间步态条带(gait strip)的cross-attention学习细粒度pair-wise交互，生成新的条件化表征并重新计算距离进行重排序。在Gait3D/GREW/OU-MVLP三个数据集、7种基线模型上一致提升Rank-1/5准确率，推理速度6.5ms/probe远超现有重排序方法。

**[Cleanpose Category-Level Object Pose Estimation Via Causal Learning And Knowledg](human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)**

:   首次将因果推理引入类别级物体位姿估计（COPE），通过基于前门调整的因果推理模块消除数据偏差导致的虚假关联，并利用3D基础模型ULIP-2的残差知识蒸馏提供无偏的类别语义监督，在REAL275的严格指标5°2cm上达到61.7%，超越SOTA 4.7%。

**[Combinative Matching For Geometric Shape Assembly](human_understanding/combinative_matching_for_geometric_shape_assembly.md)**

:   提出组合匹配（Combinative Matching）方法，同时建模互锁部件的"表面形状一致性"和"体积占用相反性"两大属性，通过等变网络学习方向对齐、形状匹配与占用匹配三个目标，大幅减少几何组装中的局部歧义。

**[Contact-Aware Refinement Of Human Pose Pseudo-Ground Truth Via Bioimpedance Sens](human_understanding/contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)**

:   提出BioTUCH框架，通过手腕间生物阻抗传感检测自接触事件，结合视觉姿态估计器进行接触感知的3D手臂姿态优化，平均提升重建精度11.7%。

**[Controllable And Expressive One-Shot Video Head Swapping](human_understanding/controllable_and_expressive_one-shot_video_head_swapping.md)**

:   本文提出一个基于扩散模型的多条件可控视频头部替换框架（SwapAnyHead），通过形状无关掩码策略、发型增强策略和表情感知的3DMM驱动landmark重定向模块，实现了高保真的身份保持、无缝背景融合和精确的跨身份表情迁移与编辑。

**[Dadm Dual Alignment Of Domain And Modality For Face Anti-Spoofing](human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)**

:   提出 DADM 框架，通过互信息掩码（MIM）模块和域-模态双对齐优化策略，同时解决多模态人脸反欺骗中的域内模态不对齐和域间模态不对齐问题，在四种协议下取得 SOTA 性能。

**[Degradation-Modeled Multipath Diffusion For Tunable Metalens Photography](human_understanding/degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)**

:   提出DMDiff框架，利用预训练扩散模型的自然图像先验，通过正/中/负三路径多提示扩散策略和空间变化退化感知注意力（SVDA）模块，实现毫米级超透镜相机的高保真可调图像重建，在多项指标上超越现有方法。

**[Dreamactor-M1 Holistic Expressive And Robust Human Image Animation With Hybrid G](human_understanding/dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)**

:   DreamActor-M1提出基于DiT架构的人体图像动画框架，通过隐式面部表征+3D头部球体+3D身体骨架的混合控制信号实现精细面部和身体控制，结合互补外观引导和渐进式训练策略支持肖像到全身的多尺度生成。

**[Dynamic Reconstruction Of Hand-Object Interaction With Distributed Force-Aware C](human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)**

:   提出 ViTaM-D，一个视觉-触觉融合框架，通过新提出的分布式力感知接触表示（DF-Field）和两阶段流程（视觉动态跟踪+力感知优化），实现刚性和可变形物体的手物交互动态重建，并引入 HOT 数据集填补可变形物体手物交互的评测空白。

**[Dynfacerestore Balancing Fidelity And Quality In Diffusion-Guided Blind Face Res](human_understanding/dynfacerestore_balancing_fidelity_and_quality_in_diffusion-guided_blind_face_res.md)**

:   提出 DynFaceRestore，通过动态模糊等级映射（DBLM）将盲退化转化为高斯去模糊问题，结合动态起始步查找表（DSST）和区域自适应引导缩放器（DGSA），在扩散模型采样中实现保真度与感知质量的最优平衡。

**[Egoagent A Joint Predictive Agent Model In Egocentric Worlds](human_understanding/egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)**

:   提出EgoAgent，一个统一的预测式智能体模型，在单个Transformer中同时学习表征第一人称视觉观测、预测未来世界状态和生成3D人体动作。

**[External Knowledge Injection For Clip-Based Class-Incremental Learning](human_understanding/external_knowledge_injection_for_clip-based_class-incremental_learning.md)**

:   提出 Engine（ExterNal knowledGe INjEction）框架，通过双分支注入调优（视觉分支用数据增强、文本分支用 GPT-4 生成判别性描述）和推理时后调优知识注入（成对判别特征重排序），在无需存储历史样本的条件下，在 9 个基准数据集上以 3-10% 的优势超越所有 CLIP-based 类增量学习方法。

**[Free-Merging Fourier Transform For Efficient Model Merging](human_understanding/free-merging_fourier_transform_for_efficient_model_merging.md)**

:   首次发现模型合并中任务干扰在频域上的表现，提出 FR-Merging 通过高通滤波去除低频干扰构建高质量合并骨干网络，并结合轻量级任务专家模块（FREE-Merging），在视觉、语言和多模态任务上实现性能-成本的最优平衡。

**[From Easy To Hard The Mir Benchmark For Progressive Interleaved Multi-Image Reas](human_understanding/from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)**

:   提出 MIR 基准，包含 22,257 个多图像交错推理问答对及五阶段推理步骤，并设计渐进式课程学习策略，从"简单到困难"逐步提升 MLLM 的多图像交错推理能力。

**[Fw-Merging Scaling Model Merging With Frank-Wolfe Optimization](human_understanding/fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)**

:   将模型合并形式化为约束优化问题，引入Frank-Wolfe优化启发的FW-Merging方法，通过迭代选择最相关模型并局部合并，实现在大规模黑盒模型池中的可扩展、鲁棒合并，合并20个ViT模型时超越数据感知方法Adamerging 8.39%。

**[Genm3 Generative Pretrained Multi-Path Motion Model For Text Conditional Human M](human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)**

:   提出 GenM3 框架，通过 Multi-Expert VQ-VAE (MEVQ-VAE) 学习统一的离散运动表示，以及 Multi-path Motion Transformer (MMT) 处理模态内变异和跨模态对齐，整合 11 个运动数据集（约 220 小时），在 HumanML3D 上达到 SOTA FID 0.035。

**[Genmo A Generalist Model For Human Motion](human_understanding/genmo_a_generalist_model_for_human_motion.md)**

:   提出 GENMO，首个统一人体运动估计（从视频/2D 关键点恢复运动）和运动生成（从文本/音乐/关键帧合成运动）的通用模型，通过双模式训练范式（回归+扩散）在单一模型中同时实现精确估计和多样生成。

**[Gesturehydra Semantic Co-Speech Gesture Synthesis Via Hybrid Modality Diffusion ](human_understanding/gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)**

:   提出 GestureHYDRA，一个基于混合模态扩散 Transformer 和级联同步检索增强生成的共语手势合成系统，能够可靠地激活语义明确的手势（如数字和方向指示）。

**[Ggtalker Talking Head Systhesis With Generalizable Gaussian Priors And Identity-](human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)**

:   GGTalker 提出先验-适配两阶段训练策略，从大规模数据集学习通用的音频-表情先验和表情-视觉先验，再快速适配到特定身份，在渲染质量、3D 一致性、唇同步和训练效率上全面达到 SOTA，仅需 20 分钟适配即可生成 120 FPS 的逼真说话头视频。

**[Guiodyssey A Comprehensive Dataset For Cross-App Gui Navigation On Mobile Device](human_understanding/guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)**

:   提出 GUIOdyssey，首个面向移动端跨应用 GUI 导航的综合数据集（8334 episodes、212 apps、1357 app 组合），以及 OdysseyAgent——配备历史重采样模块的多模态导航智能体，在平衡性能与推理效率的同时显著提升跨应用任务表现。

**[Hcceposebf Predicting Front Back Surfaces To Construct Ultra-Dense 2D-3D Corresp](human_understanding/hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)**

:   提出同时预测物体前后表面的3D坐标并在两表面间密集采样，构建超密集2D-3D对应关系，配合新颖的层级连续坐标编码（HCCE），在BOP七大核心数据集上超越现有SOTA方法。

**[Hi3Dgen High-Fidelity 3D Geometry Generation From Images Via Normal Bridging](human_understanding/hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)**

:   提出 Hi3DGen 框架，以法线图作为中间表示桥接 2D 图像到 3D 几何的映射，通过噪声注入回归式法线估计器（NiRNE）和法线正则化潜在扩散（NoRLD）两大核心组件，显著提升生成 3D 模型的几何细节保真度。

**[High-Resolution Spatiotemporal Modeling With Global-Local State Space Models For](human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)**

:   提出 GLSMamba，首个纯 Mamba 的视频人体姿态估计框架，通过 Global Spatiotemporal Mamba（6D 选择性时空扫描 + 时空调制融合）和 Local Refinement Mamba（窗口化时空扫描）分别建模全局动态上下文和局部关键点细节，在四个基准上以线性复杂度达到 SOTA。

**[Hineus High-Fidelity Neural Surface Mitigating Low-Texture And Reflective Ambigu](human_understanding/hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)**

:   提出 HiNeuS，一个统一的神经表面重建框架，通过 SDF 引导的可见性验证、平面共形正则化和渲染优先的 Eikonal 松弛三项创新，同时解决反射歧义、低纹理退化和细节保留三大核心挑战。

**[Humoto A 4D Dataset Of Mocap Human Object Interactions](human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)**

:   提出 HUMOTO，一个高保真 4D 人物交互数据集，包含 735 段序列（7875 秒，30fps），涵盖 63 个精确建模物体和 72 个可动部件，创新性地使用 LLM 驱动的场景脚本生成流程和多传感器捕获系统，在手部姿态精度和交互质量上显著超越现有数据集。

**[Idface Face Template Protection For Efficient And Secure Identification](human_understanding/idface_face_template_protection_for_efficient_and_secure_identification.md)**

:   提出 IDFace，一种基于同态加密（HE）的人脸模板保护方法，通过近等距变换（实值向量→三值向量）和空间高效编码两项技术，使 100 万加密模板的检索仅需 126ms，相比无保护仅 2× 开销。

**[Imhead A Large-Scale Implicit Morphable Model For Localized Head Modeling](human_understanding/imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)**

:   imHead 提出首个大规模隐式 3D 头部形变模型，通过全局-局部解耦架构在 4,000 个身份的数据集上训练，实现了紧凑的隐式表示与局部面部编辑的兼顾，在重建精度和编辑灵活性上超越现有方法。

**[Intra-View And Inter-View Correlation Guided Multi-View Novel Class Discovery](human_understanding/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)**

:   提出 IICMVNCD 框架，首次将新类发现（NCD）扩展到多视图设定，通过视图内矩阵分解捕捉已知/新类的分布一致性，以及视图间权重学习传递已知类的视图关系到新类，避免了对伪标签的依赖。

**[Iris Breaking Gui Complexity With Adaptive Focus And Self-Refining](human_understanding/iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)**

:   Iris 提出信息敏感裁剪（ISC）和自精炼双重学习（SRDL）两大核心创新，仅用 850K 标注数据即在多个 GUI 理解基准上达到 SOTA，性能匹敌使用 10 倍以上数据的方法，同时将处理时间从 3 秒缩短至 1 秒。

**[Kinmo Kinematic-Aware Human Motion Understanding And Generation](human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)**

:   提出 KinMo 框架，将人体运动分解为六大运动学组及其交互的层级可描述表示，通过自动标注管线生成细粒度文本描述，结合层级文本-运动对齐和由粗到细的运动生成策略，显著提升运动理解和细粒度运动生成能力。

**[Lay2Story Extending Diffusion Transformers For Layout-Togglable Story Generation](human_understanding/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)**

:   Lay2Story 提出布局可切换的故事生成任务，构建了超 100 万张高分辨率图像的 Lay2Story-1M 数据集，并基于 DiT 架构设计全局-主体双分支框架，在一致性、语义相关性和美学质量上全面超越现有方法。

**[Learning Visual Proxy For Compositional Zero-Shot Learning](human_understanding/learning_visual_proxy_for_compositional_zero-shot_learning.md)**

:   提出 Visual Proxy（视觉代理）概念，在 CZSL 任务中首次引入文本引导的视觉类中心，并通过跨模态联合学习（CMJL）协同优化文本原型与视觉代理，在四个 CZSL 基准上达到闭世界 SOTA。

**[Lvface Progressive Cluster Optimization For Large Vision Models In Face Recognit](human_understanding/lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)**

:   提出 LVFace，通过渐进式聚类优化（PCO）策略解决 ViT 在大规模人脸识别中训练不稳定的问题，将训练分解为特征对齐、质心稳定和边界精炼三个阶段，在多个基准上取得 SOTA。

**[Magshield Towards Better Robustness In Sparse Inertial Motion Capture Under Magn](human_understanding/magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)**

:   提出 MagShield，首个针对稀疏惯性运动捕捉系统中磁场干扰问题的方法，采用"检测-校正"两阶段策略：通过多 IMU 联合分析检测磁场扰动，再利用人体运动先验网络校正方向误差，可即插即用地增强现有稀疏 IMU 动捕系统的鲁棒性。

**[Mdd A Dataset For Text-And-Music Conditioned Duet Dance Generation](human_understanding/mdd_a_dataset_for_text-and-music_conditioned_duet_dance_generation.md)**

:   介绍 Multimodal DuetDance (MDD)，首个同时整合动作、音乐和文本描述的大规模专业级双人舞蹈数据集，包含 620 分钟动捕数据、15 种舞蹈类型和超过 10K 条细粒度文本标注，并提出 Text-to-Duet 和 Text-to-Dance Accompaniment 两个新任务。

**[Mitigating Object Hallucinations Via Sentence-Level Early Intervention](human_understanding/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)**

:   提出 SENTINEL 框架，通过句子级早期干预和域内偏好学习有效缓解 MLLM 的物体幻觉，在 Object HalBench 上将幻觉率降低超过 90%，同时保持甚至提升通用能力。

**[Mixri Mixing Features Of Reference Images For Novel Object Pose Estimation](human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)**

:   提出 MixRI，一个仅需 12 张参考图像和 5.3M 参数的轻量级网络，通过多视角特征融合策略直接建立多参考图与查询图之间的 2D-3D 对应关系，在 BOP 挑战的 7 个核心数据集上实现了与需要数百张参考图的方法相当的位姿估计性能。

**[Monocular Facial Appearance Capture In The Wild](human_understanding/monocular_facial_appearance_capture_in_the_wild.md)**

:   提出一种从单目头部旋转视频重建面部外观属性（漫反射反照率、高光强度、高光粗糙度）的方法，通过提出遮挡感知的 split-sum 近似着色模型，在不对光照环境做任何简化假设的情况下实现了逼近工作室级别的面部外观捕捉质量。

**[Multi-View Gaze Target Estimation](human_understanding/multi-view_gaze_target_estimation.md)**

:   本文首次将注视目标估计（GTE）从单视角扩展到多视角，通过头部信息聚合（HIA）、基于不确定性的注视选择（UGS）和基于极线的场景注意力（ESA）三个模块融合多相机信息，在自建 MVGT 数据集上显著超越单视角 SOTA，并实现了单视角方法无法处理的跨视角估计。

**[Negrefine Refining Negative Label-Based Zero-Shot Ood Detection](human_understanding/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)**

:   本文提出 NegRefine，通过 LLM 过滤负标签集中的专有名词和子类别标签，并设计多标签匹配评分函数来处理图像同时匹配分布内和负标签的情况，在 ImageNet-1K 基准上平均 AUROC 提升 1.82%、FPR95 降低 4.35%，刷新了零样本 OOD 检测 SOTA。

**[Neural Multi-View Self-Calibrated Photometric Stereo Without Photometric Stereo ](human_understanding/neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)**

:   提出一种端到端的神经逆渲染框架，从多视图变化光照图像中联合恢复几何、空间变化反射率和光照参数，无需光源标定或中间光度立体线索（如法线图），超越了现有的分阶段 MVPS 方法。

**[Ngd Neural Gradient Based Deformation For Monocular Garment Reconstruction](human_understanding/ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)**

:   提出 NGD，一种基于神经梯度的变形方法，通过将 Jacobian 场分解为帧不变的静态分量和帧相关的动态分量，结合自适应重网格化策略，从单目视频重建高保真动态纺织品几何与纹理，在宽松服装等困难场景上显著优于现有 SOTA。

**[On Large Multimodal Models As Open-World Image Classifiers](human_understanding/on_large_multimodal_models_as_open-world_image_classifiers.md)**

:   系统性地评估了 13 个大型多模态模型（LMM）在开放世界图像分类任务上的表现，提出包含 4 种互补指标的评估协议，揭示了 LMM 在粒度判断和细粒度区分上的系统性错误模式。

**[One-Shot Knowledge Transfer For Scalable Person Re-Identification](human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)**

:   提出 OSKT（One-Shot Knowledge Transfer），通过将教师模型知识精炼为"权重链"（weight chain）作为中间载体，实现一次计算即可生成任意尺寸学生模型的行人重识别模型压缩方案。

**[Openanimals Revisiting Person Re-Identification For Animals Towards Better Gener](human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)**

:   > 本文开发了 OpenAnimals 开源框架，系统回顾行人重识别方法在动物重识别中的迁移效果，提出面向动物的强基线模型 ARBase，在多个基准上大幅超越现有行人 ReID 方法。

**[Orderchain Towards General Instruct-Tuning For Stimulating The Ordinal Understan](human_understanding/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)**

:   提出 OrderChain 提示范式，通过任务感知提示和范围优化思维链（RO-CoT）增强多模态大语言模型的序数理解能力，首次实现跨任务统一序数回归模型。

**[Posesyn Synthesizing Diverse 3D Pose Data From In-The-Wild 2D Data](human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)**

:   提出 PoseSyn 框架，通过误差提取模块（EEM）从野外 2D 姿态数据中识别目标估计器的困难样本，再通过运动合成模块（MSM）将不准确的伪标签扩展为多样化的运动序列，最终借助人体动画模型生成带有准确 3D 标注的合成训练数据，在多个真实场景基准上将 3D 姿态估计精度提升最多 14%。

**[Rapverse Coherent Vocals And Whole-Body Motion Generation From Text](human_understanding/rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)**

:   构建大规模说唱数据集 RapVerse 并提出统一自回归变换器框架，首次实现从歌词文本同时生成连贯的歌声和全身3D运动。

**[Raypose Ray Bundling Diffusion For Template Views In Unseen 6D Object Pose Estim](human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)**

:   将未见物体6D位姿估计重新建模为射线对齐问题，提出物体中心的射线参数化方案，运用扩散变换器从多个已知位姿模板中推断查询图像的6D位姿。

**[Samo A Lightweight Sharpness-Aware Approach For Multi-Task Optimization With Joi](human_understanding/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)**

:   提出 SAMO，一种轻量级锐度感知多任务优化方法，通过全局-局部联合扰动缓解任务梯度冲突，并利用零阶梯度近似和层级归一化大幅降低计算开销。

**[Semges Semantics-Aware Co-Speech Gesture Generation Using Semantic Coherence And](human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)**

:   > SemGes 提出两阶段框架，通过语义一致性和语义相关性学习在全局和细粒度层面整合语义信息，生成与语音语义对齐的共语手势，在 BEAT 和 TED-Expressive 两个基准上超越现有方法。

**[Semtalk Holistic Co-Speech Motion Generation With Frame-Level Semantic Emphasis](human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)**

:   > SemTalk 将共语动作分解为节奏相关的基础动作和语义感知的稀疏动作，通过学得的语义分数自适应融合两者，实现帧级语义强调的高质量全身共语动作生成。

**[Sequential Keypoint Density Estimator An Overlooked Baseline Of Skeleton-Based V](human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)**

:   > SeeKer 提出将骨架序列的联合密度在关键点级别进行自回归分解，通过预测后续关键点的条件高斯分布来检测异常人体行为，在 UBnormal 和 MSAD-HR 数据集上大幅超越现有方法。

**[Signrep Enhancing Self-Supervised Sign Representations](human_understanding/signrep_enhancing_self-supervised_sign_representations.md)**

:   提出 SignRep，一个可扩展的自监督手语表征学习框架，通过在 Masked Autoencoder 预训练中利用手语骨架先验、特征正则化和对抗式风格无关损失，仅用单一 RGB 模态即超越了复杂的多模态/多分支方法，在手语识别、字典检索和手语翻译三大任务上均取得 SOTA。

**[Signs As Tokens A Retrieval-Enhanced Multilingual Sign Language Generator](human_understanding/signs_as_tokens_a_retrieval-enhanced_multilingual_sign_language_generator.md)**

:   提出 SOKE，一种基于预训练语言模型的多语言手语生成框架，通过解耦式 tokenizer 将连续手语动作离散化为 token 序列，结合多头解码和检索增强策略，实现从文本到多语种 3D 手语 avatar 的高质量生成。

**[Switch-A-View View Selection Learned From Unlabeled In-The-Wild Videos](human_understanding/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)**

:   提出 Switch-a-view 模型，通过从大规模无标注的互联网教学视频中学习视角切换模式（ego/exo），实现多视图教学视频的自动视角选择，无需显式的最佳视角标注。

**[Syncdiff Synchronized Motion Diffusion For Multi-Body Human-Object Interaction S](human_understanding/syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)**

:   提出 SyncDiff，一个统一的多体人体-物体交互运动合成框架，通过对齐分数（alignment scores）和显式同步策略实现多体运动的精确同步，并引入频域分解来建模高频交互语义。

**[Udc-Vit A Real-World Video Dataset For Under-Display Cameras](human_understanding/udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)**

:   提出首个真实世界屏下摄像头（UDC）视频数据集 UDC-VIT，包含 647 个视频片段共 116,460 帧，通过精心设计的双摄像头-分光器采集系统实现精确的时空对齐，并以人脸识别为核心应用场景，揭示了合成数据集在模拟真实 UDC 退化方面的不足。

**[Understanding Co-Speech Gestures In-The-Wild](human_understanding/understanding_co-speech_gestures_in-the-wild.md)**

:   本文提出 JEGAL——一个联合手势-语音-文本的三模态嵌入空间，通过全局短语对比损失和局部手势-词耦合损失在弱监督条件下学习共语手势表征，定义了三个新的手势理解任务和基准，超越了包括大型视觉语言模型在内的多种方法。

**[Weakly Supervised Visible-Infrared Person Re-Identification Via Heterogeneous Ex](human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)**

:   提出首个弱监督可见光-红外行人重识别（VIReID）范式，仅使用各模态内部的身份标注（无需跨模态对应标注），通过异构专家协同一致性学习框架建立跨模态身份对应关系，性能接近全监督方法。

**[Whats Making That Sound Right Now Video-Centric Audio-Visual Localization](human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)**

:   提出视频级音视频定位基准 AVATAR 和时序感知模型 TAVLO，通过高分辨率时序建模解决传统 AVL 方法忽略时间动态的问题。

**[Wir3D Visually-Informed And Geometry-Aware 3D Shape Abstraction](human_understanding/wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)**

:   > WIR3D 通过优化一组 3D Bézier 曲线参数，在 CLIP 中间层激活的空间引导下，从任意视角忠实表示 3D 形状的几何结构和视觉显著特征（包括纹理），实现稀疏但语义丰富的 3D 形状抽象。

---

## 📹 视频理解 { #video_understanding }

**[4D-Bench Benchmarking Multi-Modal Large Language Models For 4D Object Understand](video_understanding/4d-bench_benchmarking_multi-modal_large_language_models_for_4d_object_understand.md)**

:   4D-Bench 是首个评估多模态大语言模型（MLLM）4D 物体理解能力的基准，包含 4D 物体问答和描述两大任务，揭示了即使 SOTA GPT-4o 也仅达 63% 准确率（人类基线 91%），暴露了当前 MLLM 在多视角时序推理上的显著不足。

**[4Dbench Benchmarking Multimodal Large Language Models For 4D](video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)**

:   提出 4D-Bench，首个评估多模态大语言模型对4D物体（具有时间演化的3D物体）理解能力的基准，包含4D物体问答（751 QA对）和4D物体描述（580物体×5标注）两大任务，发现即使SOTA的GPT-4o也仅达63%准确率（人类91%），揭示了MLLM在多视角时空理解上的巨大差距。

**[Adaptive Hyper-Graph Convolution Network For Skeleton-Based Human Action Recogni](video_understanding/adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)**

:   提出 Hyper-GCN，通过**自适应非均匀超图**替代传统二元图来建模骨骼拓扑，并引入**虚拟超关节**（hyper joints）创建虚拟连接，使多关节协同关系得以直接建模，在 NTU-60/120 和 NW-UCLA 上以最轻量的 GCN 设计实现 SOTA（base 版仅 1.1M 参数、1.63 GFLOPs）。

**[Aim Adaptive Inference Of Multi-Modal Llms Via Token Merging And Pruning](video_understanding/aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)**

:   提出一种无需训练的自适应推理方法，通过 LLM 前基于嵌入相似度的迭代式 token 合并 + LLM 层内基于 PageRank 多模态重要性的渐进式 token 剪枝，实现多模态 LLM 在 40 倍 FLOPs 减少范围内的灵活精度-效率权衡，在视频和图像理解任务上均取得优异表现。

**[Aligning Effective Tokens With Video Anomaly In Large Language Models](video_understanding/aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)**

:   提出VA-GPT，通过空间有效Token选择（SETS）和时间有效Token生成（TETG）两个模块，在MLLM中高效对齐与视频异常相关的关键Token，实现对异常事件的精准检测、描述和时间定位。

**[Alltracker Efficient Dense Point Tracking At High Resolution](video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)**

:   提出AllTracker，将点跟踪重新表述为多帧长程光流问题，在低分辨率网格上通过2D卷积+像素对齐时序注意力迭代优化对应估计再上采样，仅16M参数即实现SOTA准确率和高分辨率（768×1024）全像素密集跟踪，跟踪速度接近光流方法。

**[An Empirical Study Of Autoregressive Pre-Training From Videos](video_understanding/an_empirical_study_of_autoregressive_pre-training_from_videos.md)**

:   系统性地研究了从视频进行自回归预训练的方法（称为Toto），在超过1万亿视觉token上训练因果Transformer，发现尽管归纳偏置极少，自回归预训练在图像识别、视频分类、目标跟踪和机器人操控等多个下游任务上均具有竞争力，且展现出类似语言模型的缩放规律（但速率较慢）。

**[Attention To Trajectory Trajectory-Aware Open-Vocabulary Tracking](video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)**

:   本文提出TRACT，一种利用轨迹级信息增强开放词汇多目标跟踪（OV-MOT）的方法，通过轨迹一致性强化（TCR）改善关联、通过轨迹特征聚合（TFA）和轨迹语义丰富（TSE）改善分类，在OV-TAO基准上显著提升了跟踪性能，尤其是分类准确率。

**[Beyond Label Semantics Language-Guided Action Anatomy For Few-Shot Action Recogn](video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)**

:   提出 Language-Guided Action Anatomy (LGA) 框架，利用大语言模型将动作标签解剖为原子级动作描述（主体-动作-对象三要素），同时在视频端通过聚类分割将帧序列划分为对应的原子动作阶段，在原子级别进行多模态融合和匹配，显著提升小样本动作识别性能。

**[Beyond The Frame Generating 360Deg Panoramic Videos From Perspective Videos](video_understanding/beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)**

:   提出 Argus 模型，首次实现从普通透视视频生成完整 360° 全景视频，通过相机运动模拟、视角对齐帧校准和混合解码三大几何-运动感知技术，在基于扩散模型的框架上让生成的全景视频具备空间一致性和时序连贯性。

**[Blinktrack Feature Tracking Over 80 Fps Via Events And Images](video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)**

:   提出 BlinkTrack，将可微卡尔曼滤波引入学习框架，有效解决事件相机和传统相机异步数据的关联与不确定性感知融合，实现超过 80 FPS 的高帧率特征跟踪，并在遮挡场景中显著优于现有方法。

**[Breaking The Encoder Barrier For Seamless Video-Language Understanding](video_understanding/breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)**

:   提出 ELVA，首个无编码器（encoder-free）的视频大语言模型，通过层级 token 合并、视频引导监督和混合分辨率推理机制，仅用 7M 公开视频-文本对数据即可达到与有编码器架构相当的性能，同时将 FLOPs 降低 95%、推理延迟降低 92%。

**[Despite Exploring Contrastive Deep Skeleton-Pointcloud-Imu-Text Embeddings For A](video_understanding/despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)**

:   DeSPITE 提出了一种隐私保护的多模态对比预训练模型，将 LiDAR 点云、骨架姿态、IMU 和文本四种模态对齐到统一嵌入空间，实现了跨模态匹配、检索以及人体活动识别的预训练范式。

**[Distime Distribution-Based Time Representation For Video Large Language Models](video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)**

:   提出DisTime框架，通过一个可学习的时间token和基于分布的时间解码器，在Video-LLM中实现连续时间表示，配合大规模自动标注数据集InternVid-TG（125万事件），在时刻检索、密集视频描述、Grounded-VQA三类时间敏感任务上达到SOTA。

**[Dynimg Key Frames With Visual Prompts Are Good Representation For Multi-Modal Vi](video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)**

:   DynImg 提出了一种新颖的视频表示方法，将非关键帧作为"时序视觉提示"叠加在关键帧下方形成动态图像，在视觉编码器内部实现细粒度时空交互（而非高层token级交互），配合4D旋转位置编码维护正确的时空序列关系，在多个视频理解基准上以更少的视觉token超越SOTA约2%。

**[Egoadapt Adaptive Multisensory Distillation And Policy Learning For Efficient Eg](video_understanding/egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)**

:   提出 EgoAdapt 框架，将跨模态蒸馏与策略学习联合训练，自适应选择最优模态组合，在自我中心感知任务中实现最高 89% GMACs 缩减的同时保持与 SOTA 持平甚至更优的性能。

**[Egoppg Heart Rate Estimation From Eye-Tracking Cameras In Egocentric Systems To ](video_understanding/egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)**

:   提出egoPPG这一新的自中心视觉任务，通过PulseFormer方法从未修改的自中心头戴设备的眼部追踪摄像头估计心率（MAE=7.67 bpm），并证明心率估计在EgoExo4D的技能水平评估下游任务中可提升14.1%的准确率。

**[Emotive Event-Guided Trajectory Modeling For 3D Motion Estimation](video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)**

:   本文提出 EMoTive，一个基于事件相机的 3D 运动估计框架，通过 Event Kymograph 编码精细时序演化信息，并使用事件密度引导的非均匀 NURBS 参数曲线建模时空轨迹，从轨迹中导出光流和深度运动场，在自建 CarlaEvent3D 数据集和真实世界基准上取得 SOTA 性能。

**[Estimating 2D Camera Motion With Hybrid Motion Basis](video_understanding/estimating_2d_camera_motion_with_hybrid_motion_basis.md)**

:   提出 CamFlow，通过混合运动基（12 个物理基 + 随机噪声基）表示复杂的 2D 相机运动，揭示了多个单应性流场叠加的非线性特性，结合基于 Laplace 分布的概率损失函数，在标准和跨数据集零样本条件下均大幅超越现有单应性和 meshflow 方法。

**[Factorized Learning For Temporally Grounded Video-Language Models](video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)**

:   提出D2VLM框架，通过将视频理解分解为"先定位证据再基于证据生成回答"的范式，引入证据token捕捉事件级视觉语义，并设计分解式偏好优化(FPO)同时提升时序定位和文本回答能力。

**[Fine-Grained Spatiotemporal Grounding On Egocentric Videos](video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)**

:   提出 EgoMask，首个面向自我中心视频的像素级时空定位基准，包含短/中/长时视频评测集和大规模训练集 EgoMask-Train，通过系统分析揭示了自我中心与外中心视频之间的关键差异，并证明微调后模型性能可大幅提升。

**[Flow4Agent Long-Form Video Understanding Via Motion Prior From Optical Flow](video_understanding/flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)**

:   Flow4Agent 首次将光流运动先验引入 LLM-based 视频理解，通过时域粒度优化（TGO）利用粗粒度光流聚类视频事件并用语义先验过滤冗余场景，通过运动 Token 剪枝（MTP）利用细粒度光流去除帧内静态冗余 token，在 VideoMME/MLVU/LongVideoBench 等长视频基准上取得领先表现。

**[Flowseek Optical Flow Made Easier With Depth Foundation Models And Motion Bases](video_understanding/flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)**

:   FlowSeek 将深度基础模型（Depth Anything V2）的先验知识和经典的低维运动参数化（motion bases）融入光流网络，在仅使用单张消费级 GPU 训练的条件下即可实现 SOTA 的跨数据集泛化性能。

**[Free-Moref Instantly Multiplexing Context Perception Capabilities Of Video-Mllms](video_understanding/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)**

:   提出免训练方法Free-MoRef，受MoE启发将长视频token分割为多个短序列作为多参考(multi-reference)，通过MoRef注意力机制并行查询并融合统一激活值，在单卡A100上实现2×到8×更长帧输入的高效全面理解，在VideoMME/MLVU/LongVideoBench上超越专训长视频模型。

**[General Compression Framework For Efficient Transformer Object Tracking](video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)**

:   提出 CompressTracker，一个通用 Transformer 跟踪器压缩框架，通过阶段划分、替换训练和特征模仿三个递进创新，实现结构无关的高效压缩——压缩 SUTrack 后保持约 99% 精度同时加速 2.42 倍。

**[Hermes Temporal-Coherent Long-Form Understanding With Episodes And Semantics](video_understanding/hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)**

:   提出 HERMES 框架，通过情景压缩器 (ECO) 和语义检索器 (SeTR) 两个通用模块分别捕获视频的情景记忆和语义信息，既可作为独立系统达到 SOTA，也可即插即用地增强现有视频语言模型，同时降低推理延迟达 43% 和内存消耗达 46%。

**[Hierarchical Event Memory For Accurate And Low-Latency Online Video Temporal Gro](video_understanding/hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)**

:   针对在线视频时序定位（OnVTG）任务，提出层级事件记忆机制存储不同时间尺度的历史事件信息，结合基于段树结构的事件提议和未来预测分支，在TACoS、ActivityNet Captions和MAD三大数据集上实现了SOTA的定位精度和低延迟预测。

**[Learning To Generalize Without Bias For Open-Vocabulary Action Recognition](video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)**

:   本文提出 Open-MeDe，一个基于元学习的开放词汇动作识别框架，通过跨批次元优化模拟"已知到开放"的泛化任务，并结合高斯自集成稳定化策略，在不依赖 CLIP 正则化的情况下同时提升上下文内和上下文外场景的泛化能力。

**[Memfof High-Resolution Training For Memory-Efficient Multi-Frame Optical Flow Es](video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)**

:   MEMFOF 是首个面向显存效率的多帧光流方法，通过降低相关体积分辨率并引入高分辨率训练策略，在 1080p 推理仅需 2.09GB 显存的同时在 Spring、Sintel、KITTI 等基准上达到 SOTA 精度。

**[Mikudance Animating Character Art With Mixed Motion Dynamics](video_understanding/mikudance_animating_character_art_with_mixed_motion_dynamics.md)**

:   提出 MikuDance，一种基于扩散模型的角色艺术动画系统，通过 Mixed Motion Modeling（将角色运动和 3D 相机运动统一到像素空间表示）和 Mixed-Control Diffusion（在 Reference UNet 中隐式对齐角色形状/尺度与运动引导），实现了复杂角色画作的高动态动画生成。

**[Miore Var-Miore Benchmarks To Push The Boundaries Of Restoration](video_understanding/miore_var-miore_benchmarks_to_push_the_boundaries_of_restoration.md)**

:   提出 MIORe 和 VAR-MIORe 两个多任务运动复原基准数据集，使用 1000fps 工业级相机和专业镜头阵列采集，涵盖从极微到极端的全运动幅度谱，通过自适应帧平均机制生成一致运动模糊，为去模糊、帧插值和光流估计提供统一评估平台。

**[Mobileviclip An Efficient Video-Text Model For Mobile Devices](video_understanding/mobileviclip_an_efficient_video-text_model_for_mobile_devices.md)**

:   将时空结构重参数化引入高效图像-文本模型MobileCLIP，在大规模视频-文本数据集上训练，得到可在移动端运行的视频-文本模型MobileViCLIP，在零样本检索和动作识别上取得与大模型相当的性能。

**[Moment Quantization For Video Temporal Grounding](video_understanding/moment_quantization_for_video_temporal_grounding.md)**

:   提出 MQVTG，首次将向量量化引入视频时序定位任务，通过时刻码本和软量化将视频片段映射为离散向量，增强前景/背景的区分度，在 6 个基准上取得 SOTA。

**[Multi-Modal Multi-Platform Person Re-Identification Benchmark And Method](video_understanding/multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)**

:   提出首个多模态多平台行人重识别基准 MP-ReID（含 RGB、红外、热成像三种模态 + 地面和无人机两种平台）和统一提示学习框架 Uni-Prompt ReID，通过模态感知、平台感知和视觉增强提示显著提升复杂场景下的 ReID 性能。

**[No More Sibling Rivalry Debiasing Human-Object Interaction Detection](video_understanding/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)**

:   发现并系统分析了 HOI 检测中的"有毒兄弟"偏差问题——高度相似的 HOI 三元组在输入端和输出端相互干扰竞争，提出"对比后校准"（C2C）和"合并后拆分"（M2S）两种去偏学习目标，在 HICO-DET 上超越 baseline +9.18% mAP、超越前 SOTA +3.59%。

**[Online Dense Point Tracking With Streaming Memory](video_understanding/online_dense_point_tracking_with_streaming_memory.md)**

:   提出 SPOT 框架，通过定制的记忆读取模块、感知记忆（sensory memory）和可见性引导的 splatting 实现在线稠密长程点跟踪，以 10× 更少参数和 2× 更快速度达到 CVO 基准上的 SOTA，在多个稀疏跟踪基准上也超越或媲美离线方法。

**[Ovg-Hq Online Video Grounding With Hybrid-Modal Queries](video_understanding/ovg-hq_online_video_grounding_with_hybrid-modal_queries.md)**

:   提出在线视频定位新任务 OVG-HQ，支持文本/图像/视频片段等混合模态查询，通过参数化记忆块（PMB）保留历史信息和混合蒸馏策略缓解模态不平衡，在流式视频中实时定位目标片段。

**[Prior-Flow Enhancing Primitive Panoramic Optical Flow With Orthogonal View](video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)**

:   提出双分支框架 PriOr-Flow，利用正交视图的低畸变先验来补偿 ERP 全景图像极区的严重畸变，从而显著提升全景光流估计精度，在 MPFDataset 和 FlowScape 上分别降低 EPE 30.0% 和 29.6%。

**[Q-Frame Query-Aware Frame Selection And Multi-Resolution Adaptation For Video-Ll](video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)**

:   提出 Q-Frame，一种无需训练的即插即用视频帧选择与多分辨率自适应框架，利用 CLIP 跨模态匹配和 Gumbel-Max 技巧实现查询感知的帧选择，使 Video-LLM 在相同计算预算下处理更多关键帧，在 MLVU、LongVideoBench、Video-MME 三个基准上显著提升性能。

**[Rainbowprompt Diversity-Enhanced Prompt-Evolving For Continual Learning](video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)**

:   提出 RainbowPrompt，通过注意力变换和任务引导对齐的提示演化机制，将多个任务特定提示整合为多样性增强的统一提示，在图像分类和视频动作识别任务上平均超越现有方法 8.23%。

**[Residualvit For Efficient Temporally Dense Video Encoding](video_understanding/residualvit_for_efficient_temporally_dense_video_encoding.md)**

:   本文提出 ResidualViT，通过类比视频压缩中的 I帧/P帧 策略，交替使用完整 ViT 和轻量残差 ViT 编码视频帧，在保持接近原始 CLIP 精度的同时，实现最高 60% 的计算成本降低和 2.5 倍推理加速。

**[Simultaneous Motion And Noise Estimation With Event Cameras](video_understanding/simultaneous_motion_and_noise_estimation_with_event_cameras.md)**

:   首次提出事件相机运动估计与噪声估计的联合方法，利用对比度最大化（CMax）框架中运动补偿后的局部对比度对每个事件评分，通过交替优化同时获得运动参数和信号/噪声分类，在 E-MLB 去噪基准上达到 SOTA。

**[Timeexpert An Expert-Guided Video Llm For Video Temporal Grounding](video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)**

:   提出TimeExpert——首个基于MoE的Video-LLM框架，通过**任务感知动态门控**和**token自适应路由**将时间戳、显著性分数和文本描述路由到专门的专家，配合任务依赖辅助损失，在Dense Video Captioning、Moment Retrieval和Video Highlight Detection三类VTG任务上全面超越SOTA。

**[Toga Temporally Grounded Open-Ended Video Qa With Weak Supervision](video_understanding/toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)**

:   提出TOGA——一种弱监督条件下的视觉语言模型，通过多尺度视觉语言连接器和一致性约束生成伪时序标签，在**无需任何时序标注**的情况下联合生成开放式答案与时间定位，在NExT-GQA、MSVD-QA和ActivityNet-QA上取得SOTA。

**[Towards Efficient General Feature Prediction In Masked Skeleton Modeling](video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)**

:   提出 GFP（General Feature Prediction）框架，将掩码骨架建模的重建目标从低层关节坐标提升为多层次高层语义特征预测，配合轻量级目标生成网络和信息最大化约束，实现 6.2 倍训练加速的同时达到 SOTA 性能。

**[Towards Video Thinking Test A Holistic Benchmark For Advanced Video Reasoning An](video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)**

:   提出 Video Thinking Test (Video-TT)，一个评估视频大语言模型正确性和鲁棒性的基准，包含 1000 个 YouTube Shorts 视频和 5000 个问题，通过视觉/叙事复杂性因子和自然对抗问题揭示了当前最强模型（GPT-4o 36.6%）与人类（84.3%）之间的巨大差距。

**[Trokens Semantic-Aware Relational Trajectory Tokens For Few-Shot Action Recognit](video_understanding/trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)**

:   提出Trokens框架，通过**语义感知的轨迹点采样**和**关系运动建模**（包含轨迹内HoD和轨迹间相对位移描述子），将点轨迹转化为语义感知的关系token，与外观特征融合后在6个few-shot动作识别基准上取得SOTA。

**[Umdatrack Unified Multi-Domain Adaptive Tracking Under Adverse Weather Condition](video_understanding/umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)**

:   UMDATrack 提出了首个统一多域自适应跟踪框架，利用文本引导扩散模型合成少量（<2% 帧）多天气条件无标注视频，通过域定制适配器（DCA）高效迁移目标表征到不同天气域，并引入基于最优传输的目标感知置信度对齐（TCA）增强跨域定位一致性，在夜间/雾天/雨天等场景中大幅超越现有 SOTA 跟踪器。

**[Unsupervised Joint Learning Of Optical Flow And Intensity With Event Cameras](video_understanding/unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)**

:   提出首个基于单一网络的无监督学习框架，从事件相机数据中联合估计光流和图像亮度，核心是新推导的事件光度误差（PhE）与对比度最大化（CMax）的互补损失函数。

**[Vamba Understanding Hour-Long Videos With Hybrid Mamba-Transformers](video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)**

:   提出 Vamba —— 一种混合 Mamba-Transformer 架构的大型多模态模型，用 Mamba-2 块以线性复杂度编码视频 token、用交叉注意力更新文本 token，在单 GPU 上可处理 1024 帧视频，在小时级视频理解基准上超越所有高效 LMM 方法。

**[Videollamb Long Streaming Video Understanding With Recurrent Memory Bridges](video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)**

:   提出 VideoLLaMB，通过 SceneTiling 语义分段、循环记忆桥接层和记忆缓存检索机制，以线性 GPU 内存扩展实现长流式视频理解，在 4 个 VideoQA 基准上平均提升 4.2 分。

**[Videominer Iteratively Grounding Key Frames Of Hour-Long Videos Via Tree-Based G](video_understanding/videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)**

:   提出VideoMiner——基于强化学习的长视频理解树结构框架，通过迭代分割-描述-聚类构建层次化视频树，并提出T-GRPO（树结构Group Relative Policy Optimization）引导策略模型自适应探索关键帧，在4个长视频基准上取得SOTA，并发现T-GRPO可自发激发推理链。

**[Vtimecot Thinking By Drawing For Video Temporal Grounding And Reasoning](video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)**

:   > 提出 VTimeCoT，一种无需训练的视觉-时间链式思维框架，通过在视频帧底部叠加可视化进度条和高亮关键片段，使多模态大模型能准确感知时间戳，在时间定位和推理问答任务上大幅超越 GPT-4o 和 Qwen2VL-7B 基线。

**[What You Have Is What You Track Adaptive And Robust Multimodal Tracking](video_understanding/what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)**

:   提出FlexTrack——首个系统研究**时序性不完整多模态数据**下跟踪性能的框架，通过异构MoE融合机制（HMoE）实现自适应计算复杂度，配合视频级masking训练策略，在9个基准上取得SOTA，完整模态提升2.6%，缺失模态场景提升10.2%。

**[Xtrack Multimodal Training Boosts Rgb-X Video Object Trackers](video_understanding/xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)**

:   提出 XTrack，通过 Mixture of Modal Experts (MeME) 框架和软路由分类器，实现 RGB-D/T/E 跨模态知识共享，使推理时仅用单模态即可受益于多模态训练知识，平均精度提升 3%。

---

## 🎬 视频生成 { #video_generation }

**[Adversarial Distribution Matching For Diffusion Distillation Towards Efficient I](video_generation/adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)**

:   本文提出对抗分布匹配（ADM）框架，通过基于扩散模型的判别器以对抗方式对齐真假分数估计器的潜在预测，替代DMD中预定义的KL散度，结合对抗蒸馏预训练（ADP），在SDXL上实现一步生成超越DMD2，并在SD3和CogVideoX上刷新多步蒸馏基准。

**[Aid Adapting Image2Video Diffusion Models For Instruction-Guided Video Predictio](video_generation/aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)**

:   提出AID框架，将预训练的Image2Video扩散模型（SVD）迁移至文本引导视频预测任务，通过MLLM辅助的视频状态预测、双查询Transformer条件注入和时空适配器，在多个数据集上FVD指标超越前SOTA 50%以上。

**[Aligning Moments In Time Using Video Queries](video_generation/aligning_moments_in_time_using_video_queries.md)**

:   本文提出MATR（Moment Alignment TRansformer），通过双阶段序列对齐（soft-DTW）将目标视频表示条件化于查询视频特征，实现视频到视频的时刻检索（Vid2VidMR），并设计自监督预训练策略，在ActivityNet-VRL上R@1提升13.1%、mIoU提升8.1%。

**[Badvideo Stealthy Backdoor Attack Against Text-To-Video Generation](video_generation/badvideo_stealthy_backdoor_attack_against_text-to-video_generation.md)**

:   首次提出针对文本到视频（T2V）生成模型的后门攻击框架BadVideo，利用视频中固有的静态和动态冗余信息（如未被文本指定的环境元素、运动轨迹等），通过时空组合和动态元素转换两类策略隐蔽地嵌入恶意内容，在LaVie和Open-Sora上实现高达93.5%的人类评估攻击成功率，同时有效规避现有内容审核系统。

**[Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis](video_generation/causal-entity_reflected_egocentric_traffic_accident_video_synthesis.md)**

:   本文提出Causal-VidSyn扩散模型，通过事故原因问答（ArA）模块和驾驶员注视条件的视觉token选择机制实现因果实体定位，并构建了包含154万帧注视数据的Drive-Gaze数据集，在事故视频编辑、正常到事故视频扩散、文本到视频生成三个任务中超越SOTA。

**[D3 Training-Free Ai-Generated Video Detection Using Second-Order Features](video_generation/d3_training-free_ai-generated_video_detection_using_second-order_features.md)**

:   本文从牛顿力学的二阶控制系统出发，发现真实视频和 AI 生成视频在二阶时序特征（"加速度"）上存在本质差异——真实视频波动大而生成视频平坦，据此提出 D3，一种完全免训练的 AI 生成视频检测方法，仅需计算帧间特征的二阶差分标准差即可判别，在 40 个测试子集上达到 SOTA。

**[Dacon Dino For Anime Paint Bucket Colorization With Any Number Of Reference Imag](video_generation/dacon_dino_for_anime_paint_bucket_colorization_with_any_number_of_reference_imag.md)**

:   提出DACoN，利用DINOv2基础模型的语义特征与U-Net的高分辨率空间特征融合，实现支持任意数量参考图像的动画线稿自动上色，在关键帧和连续帧上色任务中均超越现有方法。

**[Decouple And Track Benchmarking And Improving Video Diffusion Transformers For M](video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)**

:   针对 DiT 模型中 3D 全注意力机制导致的运动-外观难以解耦问题，提出共享时序核（Shared Temporal Kernel）和稠密点跟踪损失（Dense Point Tracking Loss），同时建立了更全面的运动迁移基准 MTBench 和混合运动保真度指标。

**[Dh-Facevid-1K A Large-Scale High-Quality Dataset For Face Video Generation](video_generation/dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)**

:   推出 DH-FaceVid-1K，一个包含 1,200+ 小时、270,043 个视频片段、20,000+ 个人身份的大规模高质量人脸视频数据集，重点解决现有数据集中亚洲人脸严重不足的问题，并通过系统实验验证了数据规模与模型参数的缩放定律。

**[Disentangled World Models Learning To Transfer Semantic Knowledge From Distracti](video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)**

:   提出DisWM框架，通过从"干扰视频"中预训练解纠缠表示，然后通过离线到在线的潜空间蒸馏将语义知识迁移到下游世界模型，提升视觉强化学习在环境变化下的样本效率和鲁棒性。

**[Dive Taming Dino For Subject-Driven Video Editing](video_generation/dive_taming_dino_for_subject-driven_video_editing.md)**

:   提出DIVE框架，利用预训练DINOv2模型的语义特征作为隐式对应关系来引导主体驱动的视频编辑，通过DINO特征进行时序运动建模和目标主体身份注册，实现高质量的主体替换同时保持运动一致性。

**[Dollar Fewstep Video Generation Via Distillation And Latent](video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)**

:   结合变分分数蒸馏（VSD）和一致性蒸馏实现少步视频生成，同时提出潜空间奖励模型微调方法进一步优化特定质量维度，4步student模型在VBench上达82.57分超越teacher模型和Gen-3/Kling等商业基线，1步蒸馏实现278.6倍采样加速。

**[Dreamrelation Relation-Centric Video Customization](video_generation/dreamrelation_relation-centric_video_customization.md)**

:   提出 DreamRelation，首个关系中心的视频定制方法，通过 Relation LoRA Triplet + Hybrid Mask Training 实现关系与外观的解耦，并通过时空关系对比损失增强关系动态学习，使动物能模仿人类交互。

**[Dual-Expert Consistency Model For Efficient And High-Quality Video Generation](video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)**

:   本文分析一致性模型蒸馏中高/低噪声水平的优化冲突，提出参数高效的双专家一致性模型（DCM），语义专家负责布局和运动、细节专家负责精细细节，配合时序一致性损失和GAN+特征匹配损失，在HunyuanVideo（13B）上实现4步采样接近50步基线质量。

**[Dualreal Adaptive Joint Training For Lossless Identity-Motion Fusion In Video Cu](video_generation/dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)**

:   DualReal 首次提出身份与运动的自适应联合训练框架，通过 Dual-aware Adaptation 和 StageBlender Controller 实现两个维度的无损融合，在 CLIP-I 和 DINO-I 指标上平均提升 21.7% 和 31.8%。

**[Efficientmt Efficient Temporal Adaptation For Motion Transfer In Text-To-Video D](video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)**

:   提出 EfficientMT，一个高效的端到端视频运动迁移框架，通过复用预训练 T2V 模型骨干提取时序运动特征，结合 scaler 模块和时序集成机制，仅用少量合成配对数据即可实现零样本运动迁移，推理时间较优化方法提速 10 倍以上。

**[Etva Evaluation Of Text-To-Video Alignment Via Fine-Grained Question Generation ](video_generation/etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)**

:   提出ETVA，一种基于细粒度问题生成与回答的文本-视频对齐评估方法，通过多智能体场景图遍历生成原子问题、知识增强多阶段推理回答问题，在与人类判断的相关性上大幅超越现有指标（Spearman's ρ 58.47 vs 31.0），并构建了包含2k prompts和12k问题的评估基准。

**[Free-Form Motion Control Controlling The 6D Poses Of Camera And Objects In Video](video_generation/free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)**

:   提出 SynFMC 合成数据集（首个包含相机和物体完整 6D 位姿标注的视频数据集）和 FMC 方法，实现了在文本到视频生成中独立或同时控制相机和物体的 6D 位姿，在多种场景下生成高保真视频，且兼容多种个性化 T2I 模型。

**[Fuxi-Rtm A Physics-Guided Prediction Framework With Radiative Transfer Modeling](video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)**

:   提出 FuXi-RTM，首个将深度学习辐射传输模型 (DLRTM) 作为可微物理正则化器集成到天气预报框架中的混合物理引导体系，在 88.51% 的变量-预报时效组合上超越无约束基线。

**[Generating Fast And Slow Scalable Parallel Video Generation With Video Interface](video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)**

:   提出 Video Interface Networks (VINs)，一种类似"快思考"的抽象模块，在每个扩散步中将长视频编码为固定大小的全局 token，引导 DiT 并行生成多个视频 chunk，实现高效且时序一致的长视频生成。

**[Leanvae An Ultra-Efficient Reconstruction Vae For Video Diffusion Models](video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)**

:   > 提出 LeanVAE，基于非重叠 Patch 操作、邻域感知前馈（NAF）模块、小波变换和压缩感知技术，构建超高效视频 VAE，在仅 40M 参数下实现 FLOPs 减少 50 倍、推理速度加快 44 倍，同时保持有竞争力的重建质量。

**[Long Context Tuning For Video Generation](video_generation/long_context_tuning_for_video_generation.md)**

:   本文提出Long Context Tuning（LCT），将预训练单镜头视频扩散模型的上下文窗口扩展到场景级别，通过交错3D位置嵌入和异步噪声策略实现跨镜头视觉/时序一致性，无需额外参数即支持联合和自回归多镜头生成，并展现出组合生成等涌现能力。

**[Magicdrive-V2 High-Resolution Long Video Generation For Autonomous Driving With ](video_generation/magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)**

:   MagicDrive-V2 提出了基于 DiT + 3D VAE 的多视角驾驶视频生成框架，通过时空条件编码模块和渐进式训练策略，实现了 848×1600×6 视角、241 帧的高分辨率长视频生成，显著超越现有方法的分辨率和帧数限制。

**[Motionagent Fine-Grained Controllable Video Generation Via Motion Field Agent](video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)**

:   提出 MotionAgent，通过运动场代理（Motion Field Agent）将文本中的运动描述转化为物体轨迹和相机外参，再经解析式光流合成模块统一为光流图，实现仅凭文本输入即可对 I2V 生成中的物体运动和相机运动进行细粒度精确控制。

**[Motionshot Adaptive Motion Transfer Across Arbitrary Objects For Text-To-Video G](video_generation/motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)**

:   提出 MotionShot，一个无需训练的运动迁移框架，通过高层语义对齐和低层形态对齐的两级运动对齐策略，实现在外观和结构差异显著的任意参考-目标物体对之间的高保真运动迁移。

**[Multi-Identity Human Image Animation With Structural Video Diffusion](video_generation/multi-identity_human_image_animation_with_structural_video_diffusion.md)**

:   本文提出Structural Video Diffusion框架，通过基于掩码引导的身份特定嵌入保持多人外观一致性，联合学习RGB/深度/法线三模态几何结构信息建模人物-物体交互，配合25K多人交互视频数据集Multi-HumanVid，实现多身份人体视频生成。

**[Normalcrafter Learning Temporally Consistent Normals From Video Diffusion Priors](video_generation/normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)**

:   NormalCrafter 基于视频扩散模型（SVD）提出视频法线估计方法，通过语义特征正则化（SFR）和两阶段训练策略，生成具有精细细节和时序一致性的法线序列，在视频基准上大幅超越现有单帧方法。

**[Ock Unsupervised Dynamic Video Prediction With Object-Centric Kinematics](video_generation/ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)**

:   提出 OCK（Object-Centric Kinematics），在以对象为中心的视频预测中引入显式的运动学属性（位置、速度、加速度）作为 Slot 表示的补充，通过 Joint-OCK 和 Cross-OCK 两种 Transformer 变体融合外观与运动信息，在复杂合成和真实场景中显著提升动态视频预测质量。

**[Omnihuman-1 Rethinking The Scaling-Up Of One-Stage Conditioned Human Animation M](video_generation/omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)**

:   提出 OmniHuman，一种基于 Diffusion Transformer 的多条件人体动画生成框架，通过混合文本/音频/姿态等运动相关条件的全条件训练策略实现数据规模化，首次实现单一模型支持任意身体比例、任意宽高比输入的音频驱动人体视频生成，在肖像和半身动画任务上均达到 SOTA。

**[Prompt-A-Video Prompt Your Video Diffusion Model Via Preference-Aligned Llm](video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)**

:   提出Prompt-A-Video，通过奖励引导的提示词进化流水线自动构建训练数据，经过SFT和DPO两阶段优化LLM，生成针对特定视频扩散模型偏好对齐的增强提示词。

**[Quantifying And Narrowing The Unknown Interactive Text-To-Video Retrieval Via Un](video_generation/quantifying_and_narrowing_the_unknown_interactive_text-to-video_retrieval_via_un.md)**

:   本文提出UMIVR框架，显式量化文本视频检索中的三种不确定性——文本歧义（语义熵）、映射不确定性（JS散度）和帧不确定性（时序质量帧采样），基于量化的不确定性自适应生成澄清问题，迭代精炼查询，在MSR-VTT-1k上经10轮交互达到69.2% R@1。

**[Realcam-I2V Real-World Image-To-Video Generation With Interactive Complex Camera](video_generation/realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)**

:   提出 RealCam-I2V，通过集成单目度量深度估计构建3D场景实现度量尺度对齐训练，并提供交互式3D场景轨迹绘制界面和场景约束噪声整形机制，解决了现有轨迹引导I2V方法的尺度不一致和真实世界可用性问题。

**[Reangle-A-Video 4D Video Generation As Video-To-Video Translation](video_generation/reangle-a-video_4d_video_generation_as_video-to-video_translation.md)**

:   Reangle-A-Video 将多视角视频生成重新定义为视频到视频翻译问题，通过自监督微调视频扩散模型学习视角不变运动，配合 DUSt3R 引导的多视角一致性 inpainting，从单目视频生成同步多视角视频。

**[Recammaster Camera-Controlled Generative Rendering From A Single Video](video_generation/recammaster_camera-controlled_generative_rendering_from_a_single_video.md)**

:   提出 ReCamMaster，通过帧维度拼接的视频条件注入机制和 UE5 合成的多相机同步数据集，实现从单视频输入以新相机轨迹重新生成视频，显著超越现有方法。

**[Stiv Scalable Text And Image Conditioned Video Generation](video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)**

:   本文提出 STIV，一个基于 Diffusion Transformer 的统一文本-图像条件视频生成框架，通过帧替换策略整合图像条件并引入联合图像-文本 classifier-free guidance，在单一模型中同时实现 T2V 和 TI2V 生成，8.7B 参数模型在 VBench T2V 和 I2V 上分别达到 83.1 和 90.1 的 SOTA 成绩。

**[Sweettok Semantic-Aware Spatial-Temporal Tokenizer For Compact Video Discretizat](video_generation/sweettok_semantic-aware_spatial-temporal_tokenizer_for_compact_video_discretizat.md)**

:   提出 SweetTok 视频 tokenizer，通过解耦查询自编码器（DQAE）分离空间和时间信息压缩、运动增强语言码本（MLC）按词性分配码字，在仅使用 25% token 数量的情况下，rFVD 改善 42.8%，gFVD 改善 15.1%，实现压缩率与重建保真度的最佳平衡。

**[Tip-I2V A Million-Scale Real Text And Image Prompt Dataset For Image-To-Video Ge](video_generation/tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)**

:   构建了首个百万规模的真实用户文本和图像Prompt数据集TIP-I2V（170万+唯一prompt对），包含5个SOTA图像到视频模型的生成视频，并基于此提出了TIP-Eval评估基准、用户偏好分析、以及视频真伪检测等多个研究方向。

**[Vace All-In-One Video Creation And Editing](video_generation/vace_all-in-one_video_creation_and_editing.md)**

:   本文提出VACE，一个基于Diffusion Transformer的视频生成与编辑一体化框架，通过统一的Video Condition Unit (VCU)接口和可插拔的Context Adapter结构，用单一模型覆盖参考生成、视频编辑、mask编辑等12+种视频任务，性能与任务专用模型持平。

**[Vace Allinone Video Creation And Editing](video_generation/vace_allinone_video_creation_and_editing.md)**

:   提出VACE统一视频创建和编辑框架，通过Video Condition Unit(VCU)将文本/图像/视频/掩码统一为条件输入，结合Context Adapter注入任务概念到DiT模型，首次在单一视频DiT中同时支持参考生成、视频编辑、掩码编辑及其自由组合。

**[Versatile Transition Generation With Image-To-Video Diffusion](video_generation/versatile_transition_generation_with_image-to-video_diffusion.md)**

:   本文提出VTG统一过渡视频生成框架，基于图像到视频扩散模型，通过插值初始化（噪声SLERP+LoRA插值+文本SLERP）、双向运动微调和DINOv2表征对齐正则化，在物体变形、运动预测、概念融合、场景过渡四类任务上实现平滑高保真过渡。

**[Vip Iterative Online Preference Distillation For Efficient Video Diffusion Model](video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)**

:   > 提出 ReDPO 损失函数和 V.I.P. 迭代在线偏好蒸馏框架，将偏好学习 (DPO) 与 SFT 正则化相结合用于剪枝后视频扩散模型的蒸馏，在参数减少 36.2%-67.5% 的情况下匹配甚至超越完整模型性能。

**[Vmbench A Benchmark For Perception-Aligned Video Motion Generation](video_generation/vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)**

:   提出 VMBench——首个面向视频运动质量评估的综合基准，包含五维感知对齐运动指标（PMM）和元信息引导的运动提示生成框架（MMPG），覆盖 969 类运动类型，在 Spearman 相关系数上比现有方法平均提升 35.3%。

**[Vpo Aligning Text-To-Video Generation Models With Prompt Optimization](video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)**

:   > 提出 VPO 框架，基于三大原则（无害、准确、有用）系统性优化视频生成的文本提示，通过原则导向的SFT和多反馈偏好优化，显著提升生成视频的安全性、对齐度和质量。

**[Vsrm A Robust Mamba-Based Framework For Video Super-Resolution](video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)**

:   首次将 Mamba 引入视频超分辨率（VSR），提出 VSRM 框架，通过双聚合Mamba块实现高效时空建模，结合可变形交叉Mamba对齐和频域损失，在多个基准上取得 SOTA。

**[Worldscore A Unified Evaluation Benchmark For World Generation](video_generation/worldscore_a_unified_evaluation_benchmark_for_world_generation.md)**

:   提出 WorldScore —— 首个统一的世界生成评估基准，将世界生成分解为一系列"下一场景生成"任务，支持对 3D、4D、I2V 和 T2V 模型的统一评测，并涵盖 3000 个测试样本和 10 项指标。

**[X-Dancer Expressive Music To Human Dance Video Generation](video_generation/x-dancer_expressive_music_to_human_dance_video_generation.md)**

:   X-Dancer 提出了一个统一的 Transformer-扩散框架，从单张静态图像和音乐输入出发，通过自回归 Transformer 生成与音乐节拍同步的 2D 全身舞蹈姿态 token 序列，再利用扩散模型将这些 token 转化为高保真的舞蹈视频，在多样性、表达力和视频质量上均超越了现有方法。

---

## 📦 模型压缩 { #model_compression }

**[A Good Teacher Adapts Their Knowledge For Distillation](model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)**

:   本文揭示了知识蒸馏中教师-学生容量差距问题的本质原因在于**输出分布的类内分布不匹配**，并提出 AID（Adapted Intra-class Distribution）方法，在蒸馏前对教师模型进行微调以优化其类内分布使之更符合学生的学习能力，在多种架构组合上取得了SOTA性能。

**[Achieving More With Less Additive Prompt Tuning For Rehearsal-Free Class-Increme](model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)**

:   提出 APT（Additive Prompt Tuning），用加法操作替代传统的提示拼接范式，仅在 CLS token 的 key/value 上添加两个可学习向量，在大幅降低计算开销（GFLOPs 减少 41.5%）和可训练参数（减少 78.2%）的同时实现 SOTA 的类增量学习性能。

**[Argmatch Adaptive Refinement Gathering For Efficient Dense Matching](model_compression/argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)**

:   提出自适应精炼聚合（Adaptive Refinement Gathering）管线，包含内容感知偏移估计器、局部一致匹配校正器和局部一致上采样器三个模块，配合自适应门控机制，大幅减少了稠密匹配对重量级特征提取器和全局匹配器的依赖，以轻量级模型实现与SOTA可比的性能。

**[B-Vllm A Vision Large Language Model With Balanced Spatio-Temporal Tokens](model_compression/b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)**

:   本文提出B-VLLM框架，通过文本条件自适应帧选择、时序帧Token合并和空间Token采样三个模块，在VLLM的上下文窗口限制内动态平衡视频的时空线索，在MVBench上带来10%的性能提升。

**[Beyond Low-Rank Tuning Model Prior-Guided Rank Allocation For Effective Transfer](model_compression/beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)**

:   提出SR-LoRA（Stable Rank-Guided LoRA），利用预训练权重矩阵的稳定秩（Stable Rank）作为自然先验为每层LoRA模块分配最优秩，无需搜索即可实现灵活的逐层秩分配，在大域差距+少样本迁移场景（如医学影像）中显著优于固定低秩LoRA和其他自适应秩方法。

**[Bridging Continuous And Discrete Tokens For Autoregressive V](model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_v.md)**

:   TokenBridge提出对预训练VAE连续特征进行后训练维度级量化，将连续token无损转化为离散token，再通过轻量级维度级自回归头高效建模指数级大词表空间，在ImageNet 256×256上用标准交叉熵损失达到了与连续token方法（如MAR）相当的生成质量（FID=1.55），且推理快5.94倍。

**[Bridging Continuous And Discrete Tokens For Autoregressive Visual Generation](model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)**

:   提出TokenBridge，通过对预训练连续VAE特征进行后训练维度级量化，将连续token转化为离散token，在保持连续token高保真表示能力的同时，使用标准交叉熵损失进行简洁的自回归建模，在ImageNet 256×256上达到与连续方法可比的生成质量。

**[Ciard Cyclic Iterative Adversarial Robustness Distillation](model_compression/ciard_cyclic_iterative_adversarial_robustness_distillation.md)**

:   提出CIARD，通过对比推离损失（Contrastive Push Loss）解决双教师ARD框架中clean teacher和robust teacher的优化目标冲突，并设计迭代教师训练（ITT）策略持续更新robust teacher以防止性能退化，在CIFAR-10/100和Tiny-ImageNet上同时提升对抗鲁棒性+3.53%和干净准确率+5.87%。

**[Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks](model_compression/color_matching_using_hypernetwork-based_kolmogorov-arnold_networks.md)**

:   提出cmKAN，利用超网络驱动的Kolmogorov-Arnold Network进行颜色匹配，通过生成器预测空间变化的KAN样条参数，支持有监督/无监督/配对优化三种场景和raw-to-raw/raw-to-sRGB/sRGB-to-sRGB三种任务，在所有任务上平均超越现有方法37.3%且极轻量（76.4K参数）。

**[Colors See Colors Ignore Clothes Changing Reid With Color Disentanglement](model_compression/colors_see_colors_ignore_clothes_changing_reid_with_color_disentanglement.md)**

:   提出CSCI方法，通过引入Color token学习颜色表示（Color See），并利用新颖的S2A自注意力机制将颜色信息与ReID特征解耦（Color Ignore），在无需外部标注的情况下有效消除换衣行人重识别中的外观偏差。

**[Competitive Distillation A Simple Learning Strategy For Improving Visual Classif](model_compression/competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)**

:   提出竞争蒸馏策略，在多网络联合训练中，每个迭代动态选择表现最好的网络作为教师，配合随机扰动机制引入类似遗传算法的变异操作，显著提升视觉分类性能。

**[Context Guided Transformer Entropy Modeling For Video Compression](model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)**

:   提出Context Guided Transformer (CGT) 条件熵模型，通过时间上下文重采样器降低计算开销、依赖加权空间上下文分配器显式建模空间依赖关系，在视频压缩中将熵建模时间减少约65%，同时实现11% BD-Rate改进。

**[Cross-Architecture Distillation Made Simple With Redundancy Suppression](model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)**

:   提出RSD（Redundancy Suppression Distillation），通过跨架构不变性最大化和特征去相关来提取架构无关知识，仅用一个简单的RSD损失和轻量MLP解耦模块，在CIFAR-100和ImageNet-1k上大幅超越跨架构蒸馏先驱方法OFA，且参数开销仅为其小部分。

**[Dataset Distillation Via The Wasserstein Metric](model_compression/dataset_distillation_via_the_wasserstein_metric.md)**

:   提出 WMDD（Wasserstein Metric-based Dataset Distillation），使用 Wasserstein 重心替代 MMD 进行分布匹配，结合逐类 BatchNorm 正则化，在 ImageNet-1K 等大规模数据集上达到 SOTA 数据集蒸馏性能。

**[Dlf Extreme Image Compression With Dual-Generative Latent Fusion](model_compression/dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)**

:   提出双分支生成式隐空间融合（DLF）框架，将图像隐空间分解为语义和细节两个分支分别压缩，通过跨分支交互设计消除冗余，在极低码率（<0.01 bpp）下实现了超越 MS-ILLM 高达 67.82% BD-Rate 节省的 SOTA 重建质量，同时解码速度远快于扩散模型方案。

**[Duolora Cycle-Consistent And Rank-Disentangled Content-Style Personalization](model_compression/duolora_cycle-consistent_and_rank-disentangled_content-style_personalization.md)**

:   DuoLoRA 提出在 LoRA 的秩维度上学习掩码（ZipRank），结合 SDXL 层先验信息和循环一致性损失（Constyle loss），实现了高效的内容-风格 LoRA 合并，在多个基准上超过 ZipLoRA 等 SOTA 方法，且可训练参数减少 19 倍。

**[Ea-Vit Efficient Adaptation For Elastic Vision Transformer](model_compression/ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)**

:   提出首个在适配（adaptation）阶段引入弹性结构的ViT框架，通过多维弹性架构+课程学习+轻量路由器，一次适配即生成覆盖10^26种配置的子模型，在多个下游任务上持续优于现有弹性方法。

**[Efficient Adaptation Of Pre-Trained Vision Transformer Underpinned By Approximat](model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)**

:   本文发现预训练 ViT 权重矩阵的行/列向量具有近似正交性，而 LoRA/Adapter 的投影矩阵不具备此性质；提出 AOFT 策略，用单个可学习向量生成近似正交的下/上投影矩阵，使其与骨干网络性质对齐，从而降低泛化误差上界，在 FGVC 和 VTAB-1k 上用更少参数达到竞争性能。

**[Fastvar Linear Visual Autoregressive Modeling Via Cached Token Pruning](model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)**

:   FastVAR 提出一种无需训练的后处理加速方法，通过观察 VAR 模型中大尺度步骤主要建模高频纹理且对剪枝鲁棒的特性，利用频域引导的关键 token 选择（PTS）仅保留高频 token 参与前向，并用缓存的早期尺度 token 恢复被剪枝的位置（CTR），在 FlashAttention 基础上实现额外 2.7× 加速且性能损失 <1%，并首次实现单张 3090 GPU 上 1.5 秒生成 2K 图像。

**[Fuse Before Transfer Knowledge Fusion For Heterogeneous Distillation](model_compression/fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)**

:   提出 FBT（Fuse Before Transfer），通过在知识传递前先融合异构教师和学生的模块（CNN/MSA/MLP），构建一个自适应的中间融合模型来缓解跨架构蒸馏（CAKD）中的特征差距，并用空间无关的 InfoNCE 损失替代传统 MSE 损失，在 CIFAR-100 上平均提升 8.38%，在 ImageNet-1K 上平均提升 2.31%。

**[Gain-Mlp Improving Hdr Gain Map Encoding Via A Lightweight Mlp](model_compression/gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp.md)**

:   提出使用 10KB 轻量级 MLP 网络替代传统 JPEG/HEIC 压缩来编码 HDR gain map，以 SDR 图像的颜色和位置坐标 (r,g,b,x,y) 作为输入，结合指数残差编码（gamma map），在多个 HDR 重建指标上超越现有方法和传统压缩技术。

**[Generalized Tensor-Based Parameter-Efficient Fine-Tuning Via Lie Group Transform](model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)**

:   提出 LieRA，利用李群理论将矩阵级 PEFT 方法（如 LoRA）推广到高维参数空间（如卷积核），通过在李代数中表示扰动并用指数映射回李群，在保持参数空间结构性质的同时实现高效微调。

**[Heavy Labels Out Dataset Distillation With Label Space Lightening](model_compression/heavy_labels_out_dataset_distillation_with_label_space_lightening.md)**

:   提出 HeLlO 框架，利用 CLIP 预训练模型和 LoRA-like 低秩知识迁移构建轻量级图像-标签投影器，将数据集蒸馏中软标签的存储需求降低至原来的 0.003%，同时保持甚至超越 SOTA 性能。

**[Integrating Task-Specific And Universal Adapters For Pre-Trained Model-Based Cla](model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)**

:   提出 TUNA 方法，通过为每个增量任务训练正交的 task-specific adapter，并将它们融合为一个 universal adapter，结合基于熵的 adapter 选择机制和双 adapter 集成推理策略，在无 exemplar 的 PTM-based CIL 中实现 SOTA。

**[Knowledge Distillation With Refined Logits](model_compression/knowledge_distillation_with_refined_logits.md)**

:   RLD 通过 Sample Confidence（样本置信度）和 Masked Correlation（掩码相关性）两种精炼知识，在不破坏类别相关性的前提下修正教师错误预测的负面影响，在 CIFAR-100 和 ImageNet 上全面超越现有 logit 蒸馏方法。

**[Learned Image Compression With Hierarchical Progressive Context Modeling](model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)**

:   提出分层渐进上下文模型 (HPCM)，通过将 latent 划分为多尺度子表征并从小到大依次编码，结合跨编码步的渐进上下文融合机制（基于交叉注意力），实现更高效的远程依赖建模和更准确的熵参数估计，在图像压缩性能和计算复杂度之间取得更好的平衡。

**[Local Dense Logit Relations For Enhanced Knowledge Distillation](model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)**

:   本文提出局部稠密关系 logit 蒸馏（LDRLD），通过递归解耦和重组 logit 知识来捕获细粒度的类间关系，结合自适应衰减权重（ADW）策略对关键类别对赋予更高权重，在 CIFAR-100、ImageNet-1K 和 Tiny-ImageNet 上持续优于现有 logit 蒸馏 SOTA。

**[Mixa-Q Revisiting Activation Sparsity For Vision Transformers From A Mixed-Preci](model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)**

:   提出 MixA-Q，一种混合精度激活量化框架，将窗口级激活稀疏性（原本用于剪枝）转化为量化维度的利用——对不重要的窗口分配更低比特宽度而非完全跳过计算，在 COCO 目标检测上实现 PTQ 无损 1.35× 加速和 QAT 无损 1.25× 加速，同时具有更好的 OOD 鲁棒性。

**[Motionfollower Editing Video Motion Via Score-Guided Diffusion](model_compression/motionfollower_editing_video_motion_via_score-guided_diffusion.md)**

:   提出 MotionFollower，通过两个轻量卷积控制器（姿态+外观）和基于分数函数正则化的一致性引导机制，实现视频运动编辑，在 GPU 显存消耗减少约 80% 的同时超越 MotionEditor 等强基线。

**[Msq Memory-Efficient Bit Sparsification Quantization](model_compression/msq_memory-efficient_bit_sparsification_quantization.md)**

:   提出MSQ，通过RoundClamp量化器从权重直接计算最低有效位(LSB)并施加L1正则化诱导稀疏性，无需显式创建bit-level可训练参数即可实现混合精度量化发现，训练参数减少8倍、训练时间减少86%，同时保持竞争性的精度-压缩权衡。

**[Ouromamba A Data-Free Quantization Framework For Vision Mamba](model_compression/ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)**

:   首个面向 Vision Mamba 模型（VMM）的无数据后训练量化框架，通过增强隐式注意力生成高质量合成数据，并结合动态异常值检测的混合精度量化方案，在 W4A4 设置下显著超越现有数据驱动 PTQ 方法。

**[Partial Forward Blocking A Novel Data Pruning Paradigm For Lossless Training Acc](model_compression/partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)**

:   提出 Partial Forward Blocking (PFB)，在前向传播的浅层阶段计算样本重要性并剪枝，阻断被剪枝样本的后续深层前向传播，实现 ImageNet 上 40% 剪枝下 0.5% 精度提升 + 33% 训练时间缩减。

**[Perspective-Aware Teaching Adapting Knowledge For Heterogeneous Distillation](model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)**

:   提出PAT（Perspective-Aware Teaching）框架，通过区域感知注意力（RAA）解决异构架构间的视角不匹配问题，通过自适应反馈提示（AFP）解决教师无感知问题，使得特征级蒸馏首次在异构知识蒸馏场景中全面超越logits级方法。

**[Scheduling Weight Transitions For Quantization-Aware Training](model_compression/scheduling_weight_transitions_for_quantization-aware_training.md)**

:   指出传统学习率调度对量化感知训练（QAT）中量化权重的有效步长控制失效，提出转换率（Transition Rate）调度技术，通过自适应学习率（TALR）显式控制量化权重的离散跳变次数，显著提升低比特量化模型性能。

**[Soft Separation And Distillation Toward Global Uniformity In Federated Unsupervi](model_compression/soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)**

:   提出 Soft Separation and Distillation (SSD) 框架，通过维度缩放正则化 (DSR) 和投影器蒸馏 (PD) 两个模块，解决联邦无监督学习中客户端间 (inter-client) 表示均匀性不足的问题，在不增加通信开销的前提下显著提升全局表示质量。

**[Ssvq Unleashing The Potential Of Vector Quantization With Sign-Splitting](model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)**

:   提出 Sign-Splitting Vector Quantization (SSVQ)，将权重的符号位与码本解耦，引入可学习符号位和增强的迭代冻结策略，使 VQ 微调时每个量化权重可以沿各自梯度方向独立更新，在极端压缩率下显著优于传统 VQ 和标量量化。

**[Stolenlora Exploring Lora Extraction Attacks Via Synthetic Data](model_compression/stolenlora_exploring_lora_extraction_attacks_via_synthetic_data.md)**

:   StolenLoRA 首次提出针对 LoRA 自适应模型的模型提取攻击方向，利用 LLM 驱动的 Stable Diffusion 生成高质量合成数据替代真实数据集搜索，并设计基于分歧的半监督学习（DSL）策略通过选择性查询最大化信息增益，仅需 10k 次查询即可达到高达 96.60% 的攻击成功率，揭示了 LoRA 适配模型的严重安全漏洞。

**[Syncity Training-Free Generation Of 3D Worlds](model_compression/syncity_training-free_generation_of_3d_worlds.md)**

:   SynCity 提出了一种无需训练和优化的3D世界生成方法，通过精心设计的提示工程策略，组合预训练的语言模型、2D图像生成器（Flux）和3D生成器（TRELLIS），以tile-by-tile的方式自回归地生成大规模、高质量、可自由导航的3D场景。

**[Task Vector Quantization For Memory-Efficient Model Merging](model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)**

:   本文提出对任务向量（fine-tuned 与 pre-trained 权重之差）而非 fine-tuned 权重本身进行量化，利用任务向量更窄的数值范围实现低至 3-bit 的量化而不损失精度；进一步提出残差任务向量量化（RTVQ），将任务向量分解为共享高精度基向量和低精度偏移量，在仅用 8% 原始存储的情况下维持甚至提升模型合并性能。

**[Time-Aware Auto White Balance In Mobile Photography](model_compression/time-aware_auto_white_balance_in_mobile_photography.md)**

:   本文提出一种利用手机上下文元数据（时间戳和地理位置）辅助图像颜色信息的轻量化光照估计方法（约 5K 参数），在自建 3224 张智能手机数据集上达到或超过大模型性能，且可在旗舰手机 DSP 上 0.25ms 内完成推理。

**[Tr-Pts Task-Relevant Parameter And Token Selection For Efficient Tuning](model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)**

:   提出 TR-PTS 框架，通过 Fisher 信息矩阵进行任务驱动的逐层参数选择，同时利用 CLS 注意力分数动态筛选/合并 token，在仅微调 0.34%-0.60% 参数的情况下超越全量微调 3.40%（FGVC）和 10.35%（VTAB）。

**[Uniconvnet Expanding Effective Receptive Field While Maintaining Asymptotically ](model_compression/uniconvnet_expanding_effective_receptive_field_while_maintaining_asymptotically_.md)**

:   提出UniConvNet，通过合理组合较小卷积核（7×7, 9×9, 11×11）的三层感受野聚合器（RFA），在扩大有效感受野（ERF）的同时保持其渐近高斯分布（AGD），从而在轻量级到大规模模型上全面超越现有CNN和ViT。

**[Variance-Based Pruning For Accelerating And Compressing Trained Networks](model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)**

:   提出基于方差的一次性结构化剪枝方法（VBP），通过移除MLP隐藏层中方差最小的神经元，并将其均值激活补偿到下一层偏置中，以极少微调（10 epoch）即可恢复99%原始精度，同时减少35%计算量和36%参数。

**[Vit-Linearizer Distilling Quadratic Knowledge Into Linear-Time Vision Models](model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)**

:   提出 ViT-Linearizer，一种跨架构蒸馏框架，通过**激活匹配**和**掩码预测**两个核心机制，将 ViT 自注意力中学习到的"二次知识"高效迁移到线性复杂度的循环模型（Mamba-based Adventurer），在 ImageNet 上达到 84.3% 准确率，同时在高分辨率任务中实现最高 4.2× 的推理加速。

**[Vq-Sgen A Vector Quantized Stroke Representation For Creative Sketch Generation](model_compression/vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)**

:   > 提出 VQ-SGen，将每个笔画视为独立实体并解耦其形状与位置信息，通过向量量化（VQ）构建紧凑离散的笔画码本，再用级联自回归 Transformer 逐步生成笔画的语义标签、形状和位置，在 CreativeSketch 数据集上显著超越现有方法。

---

## 🏥 医学图像 { #medical_imaging }

**[Aczerots Active Learning For Zeroshot Tissue Segmentation In](medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)**

:   提出AcZeroTS框架，将主动学习与基于VLM的原型引导零样本分割模型ProZS结合，通过同时考虑不确定性、多样性和原型覆盖unseen类的能力来选择最有价值的标注样本，以最少标注实现seen和unseen组织类型的高质量分割。

**[Alleviating Textual Reliance In Medical Language-Guided Segmentation Via Prototy](medical_imaging/alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)**

:   提出ProLearn框架，首次通过原型驱动的语义近似（PSA）模块从根本上缓解医学语言引导分割对文本的依赖——仅需少量图文配对数据初始化原型空间，训练和推理均可无文本输入，在1%文本可用性下仍保持强劲性能（QaTa-COV19 Dice=0.857），且参数量比LLM方案减少1000倍，推理速度快100倍。

**[An Openmind For 3D Medical Vision Self-Supervised Learning](medical_imaging/an_openmind_for_3d_medical_vision_self-supervised_learning.md)**

:   发布了最大的公开 3D 医学影像预训练数据集 OpenMind（114k 脑部 MRI），并系统性地对比了多种 3D 自监督学习方法在 CNN（ResEnc-L）和 Transformer（Primus-M）上的表现，证明 MAE 预训练在分割任务上最优、对比学习在分类任务上最优，且首次展示预训练 Transformer 可在部分数据集上超越从头训练的 CNN。

**[An Openmind For 3D Medical Vision Selfsupervised Learning](medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)**

:   发布了最大的公开3D医学影像预训练数据集OpenMind（114k脑MRI体积），并在该数据集上系统性benchmark了现有3D SSL方法在最先进CNN（ResEnc-L）和Transformer（Primus-M）架构上的表现，明确了3D医学图像SSL的当前SOTA。

**[Beyond Brain Decoding Visualsemantic Reconstructions To Ment](medical_imaging/beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)**

:   提出NeuroCreat——一种结合LLM视觉与文本能力的脑多模态架构，将fMRI解码从单一的视觉刺激重建扩展到**图像重建 + 文本描述（captioning）+ 心理创造（creation）**三个层次，通过Prompt Variant Alignment模块有效弥合fMRI低分辨率信号与高级语义表征之间的鸿沟。

**[Boosting Vision Semantic Density With Anatomy Normality Mode](medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_mode.md)**

:   提出 ViSD-Boost，通过疾病级视觉对比学习增强视觉语义 + VQ-VAE 建模解剖正常性分布来放大异常信号，解决医学 VLP 中视觉语义密度低导致的对齐偏差，在腹部 CT 54 种疾病零样本诊断达到 84.9% AUC。

**[Boosting Vision Semantic Density With Anatomy Normality Modeling For Medical Vis](medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)**

:   提出 ViSD-Boost 方法，通过疾病级视觉对比学习增强视觉语义、以及基于 VQ-VAE 的解剖正常性建模来放大异常信号，解决医学视觉语言预训练中视觉模态语义密度低导致的对齐偏差问题，在 15 个器官 54 种疾病的零样本诊断上达到 84.9% AUC。

**[Coin Confidence Score-Guided Distillation For Annotation-Free Cell Segmentation](medical_imaging/coin_confidence_score-guided_distillation_for_annotation-free_cell_segmentation.md)**

:   提出COIN框架，通过无监督语义分割+最优传输的像素级细胞传播、基于模型-SAM一致性的实例级置信度评分、以及置信度引导的递归自蒸馏三步策略，解决了无标注细胞实例分割中"无错误实例缺失"的关键问题，在MoNuSeg和TNBC上超越半监督/弱监督方法。

**[Controllable Latent Space Augmentation For Digital Pathology](medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)**

:   提出HistAug——一种基于Transformer的轻量级潜在空间增强模型，通过条件式跨注意力机制在特征空间中模拟真实图像变换（色相、腐蚀等），以极低计算开销为病理MIL训练提供可控且高效的数据增强。

**[Coordinate-Based Speed Of Sound Recovery For Aberration-Corrected Photoacoustic ](medical_imaging/coordinate-based_speed_of_sound_recovery_for_aberration-corrected_photoacoustic_.md)**

:   本文提出一种高效的自监督联合重建方法，通过将声速（SOS）参数化为像素网格或神经场，并通过可微成像前向模型反向传播梯度来恢复SOS和高质量光声图像，在精度上超越现有SOTA的同时实现35倍加速（40秒 vs 23分钟）。

**[Cryofastar Fast Cryo-Em Ab Initio Reconstruction Made Easy](medical_imaging/cryofastar_fast_cryo-em_ab_initio_reconstruction_made_easy.md)**

:   提出CryoFastAR，首个面向冷冻电镜（cryo-EM）的几何基础模型，通过ViT架构直接从多视图噪声粒子图像前馈式预测Fourier Planar Map实现位姿估计，在合成和真实数据集上达到可比质量的同时实现10倍以上加速。

**[Cryofastar Fast Cryoem Ab Initio Reconstruction Made Easy](medical_imaging/cryofastar_fast_cryoem_ab_initio_reconstruction_made_easy.md)**

:   首个将DUSt3R式的几何基础模型范式引入冷冻电镜(cryo-EM)领域的工作，通过ViT编码器+跨视图注意力解码器直接从大量含噪粒子图像前馈预测姿态（无需迭代优化），实现了比传统方法快10-33倍的ab initio蛋白质三维重建。

**[Cumperlay Learning Cubical Multiparameter Persistence Vectorizations](medical_imaging/cumperlay_learning_cubical_multiparameter_persistence_vectorizations.md)**

:   提出 CuMPerLay，一个可微的立方多参数持久同调 (Cubical Multiparameter Persistence, CMP) 向量化层，将 CMP 分解为多条可学习的单参数持久同调线，通过联合学习双滤过 (bifiltration) 函数实现端到端训练，嵌入 Swin Transformer 后在医学图像分类和语义分割任务上（尤其小数据场景）取得显著提升。

**[Dictas A Framework For Class-Generalizable Few-Shot Anomaly Segmentation Via Dic](medical_imaging/dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)**

:   受人类检查员"查字典"直觉启发，提出 DictAS 框架，将少样本异常分割重新定义为字典查询任务——若查询特征无法从正常样本字典中检索到则判定为异常——通过自监督训练获得类别无关的字典查询能力，在 7 个工业和医学数据集上的 FSAS 性能和推理速度均达到 SOTA。

**[G2Pdiffusion Cross-Species Genotype-To-Phenotype Prediction Via Evolutionary Dif](medical_imaging/g2pdiffusion_cross-species_genotype-to-phenotype_prediction_via_evolutionary_dif.md)**

:   提出G2PDiffusion，首个基于扩散模型的跨物种基因型到表型预测框架，通过进化信号（多序列比对MSA和环境上下文）条件化生成形态学图像，实现从DNA序列预测物种外观。

**[Gdkvm Echocardiography Video Segmentation Via Spatiotemporal Key-Value Memory Wi](medical_imaging/gdkvm_echocardiography_video_segmentation_via_spatiotemporal_key-value_memory_wi.md)**

:   提出 GDKVM，一种基于线性键值关联和门控 Delta 规则的心脏超声视频分割架构，通过高效的内存管理和多尺度特征融合，在 CAMUS 和 EchoNet-Dynamic 上实现 SOTA 性能，同时保持实时推理速度。

**[Gecko Gigapixel Vision-Concept Contrastive Pretraining In Histopathology](medical_imaging/gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)**

:   提出GECKO，一种无需额外临床数据模态的WSI级MIL聚合器预训练方法，通过从H&E WSI自动提取可解释的概念先验(Concept Prior)并与深度特征对比对齐，在5个分类任务上超越现有单模态和多模态预训练方法，同时提供病理学家可解释的WSI级描述。

**[Gemex A Large-Scale Groundable And Explainable Medical Vqa Benchmark For Chest X](medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)**

:   构建了当前最大的胸部X光 VQA 数据集 GEMeX（151K 图像、1.6M 问题），首次同时提供文本推理解释和视觉区域定位，涵盖四种问题类型，并系统评估了 12 个代表性大视觉语言模型。

**[Hrscene How Far Are Vlms From Effective High-Resolution Image Understanding](medical_imaging/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)**

:   提出 HRScene 基准，涵盖 25 个真实场景和 2 个诊断数据集（分辨率 1K-35K），评估 28 个 VLM 后发现：当前最强模型在真实高分辨率任务上平均准确率仅约 50%，且存在显著的区域差异和 lost-in-middle 问题。

**[Idf Iterative Dynamic Filtering Networks For Generalizable Image Denoising](medical_imaging/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)**

:   提出迭代动态滤波网络 (IDF)，仅用约 0.04M 参数，通过逐像素动态核预测 + 自适应迭代精炼策略，仅在单一级别高斯噪声上训练即可泛化到各种未见噪声类型（高斯/泊松/椒盐/蒙特卡洛渲染/真实噪声），实现出色的 OOD 去噪性能。

**[Integrating Biological Knowledge For Robust Microscopy Image Profiling On De Nov](medical_imaging/integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)**

:   提出将外部生物知识（蛋白质互作图谱+单细胞基础模型的转录组特征）整合到显微图像预训练中，显式解耦扰动特异性和细胞系特异性表征，提升模型在未见细胞系上的扰动筛查泛化能力。

**[Mrgen Segmentation Data Engine For Underrepresented Mri Modalities](medical_imaging/mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)**

:   针对稀缺 MRI 模态缺乏分割标注的难题，构建了大规模放射影像数据集 MRGen-DB（~25 万张切片、100+ 模态），并训练了可控扩散数据引擎 MRGen，通过文本+掩码双条件控制生成目标模态的高质量 MR 图像用于训练分割模型，在 10 对跨模态实验中平均 DSC 从 10%~27% 提升至 43%~45%，实现了标注稀缺模态的"零样本"分割。

**[Multiverseg Scalable Interactive Segmentation Of Biomedical Imaging Datasets Wit](medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)**

:   提出 MultiverSeg，一个渐进式交互分割系统：用户每标注一张图像，后续图像所需的交互次数就会减少，通过将已分割图像作为上下文输入模型实现"越用越好"的效果，在 12 个未见数据集上相比 ScribblePrompt 将点击数减少 36%、涂鸦步骤减少 25%。

**[Neurons Emulating The Human Visual Cortex Improves Fidelity And Interpretability](medical_imaging/neurons_emulating_the_human_visual_cortex_improves_fidelity_and_interpretability.md)**

:   提出 NEURONS 框架，受人类视觉皮层层级结构启发，将 fMRI 到视频的重建解耦为四个子任务（关键物体分割、概念识别、场景描述、模糊视频重建），模拟 V1/V2/V4/ITC 等脑区的功能特化，在视频一致性（26.6%）和语义准确度（19.1%）上显著超越 SOTA。

**[Predict-Optimize-Distill A Self-Improving Cycle For 4D Object Understanding](medical_imaging/predict-optimize-distill_a_self-improving_cycle_for_4d_object_understanding.md)**

:   提出 Predict-Optimize-Distill (POD) 框架，通过预测-优化-蒸馏的自改进循环，从单目长视频中恢复铰接物体的4D部件姿态，性能随视频长度和迭代次数持续提升。

**[Progait A Multi-Purpose Video Dataset And Benchmark For Transfemoral Prosthesis ](medical_imaging/progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)**

:   提出ProGait——首个面向大腿截肢假肢用户的多用途视频数据集，支持视频目标分割、2D人体姿态估计和步态分析三项任务，并提供基线模型证明数据集对改善假肢检测的有效性。

**[Progressive Test Time Energy Adaptation For Medical Image Segmentation](medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)**

:   提出一种基于能量模型的渐进式测试时自适应方法，训练一个形状能量模型作为分布内/外判别器，在测试时通过最小化能量值引导分割模型适应目标域，在心脏、脊髓、肺部等 8 个公共数据集上持续超越基线。

**[Pvchat Personalized Video Chat With One-Shot Learning](medical_imaging/pvchat_personalized_video_chat_with_one-shot_learning.md)**

:   提出 PVChat，首个支持从单个参考视频进行个性化主体学习的视频大语言模型，通过 ReLU 路由混合注意力头（ReMoH）机制、系统化的数据增强管道和渐进式图像到视频训练策略，实现身份感知的视频问答，在医疗、电视剧、动漫等多种场景中超越现有 SOTA ViLLM。

**[Radgpt Constructing 3D Image-Text Tumor Datasets](medical_imaging/radgpt_constructing_3d_image-text_tumor_datasets.md)**

:   本文提出 RadGPT——一个解剖感知的 VL AI 管线，通过将放射科医师修订的肿瘤分割 mask 经由确定性算法转化为结构化报告、再由 LLM 适配为叙述性报告，构建了首个大规模公开腹部 CT 图文肿瘤数据集 AbdomenAtlas 3.0（9,262 例 CT、每体素标注 + 报告），并证明分割辅助可显著提升 AI 报告中的肿瘤检测率。

**[Scaling Tumor Segmentation Best Lessons From Real And Synthetic Data](medical_imaging/scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)**

:   通过在大规模私有数据集上系统研究数据缩放定律，发现合成肿瘤可大幅降低真实标注需求（从 1500 降至 500 例），并据此构建了 AbdomenAtlas 2.0——首个涵盖 6 种器官肿瘤的万级 CT 大规模人工标注数据集，在分布内和分布外测试上均取得显著提升。

**[Scivid Cross-Domain Evaluation Of Video Models In Scientific Applications](medical_imaging/scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)**

:   提出 SciVid 基准，包含动物行为分类、组织追踪、天气预测等 5 个跨学科科学视频任务，系统评估 6 类视频基础模型（ViFM），发现用简单可训练 readout 适配冻结的 ViFM backbone 即可在多个科学应用中达到 SOTA，首次证明通用 ViFM 在科学领域的可迁移性。

**[Seganypet Universal Promptable Segmentation From Positron Emission Tomography Im](medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)**

:   本文构建了迄今最大的PET分割数据集PETS-5k（5731例3D全身PET图像，超130万张2D切片），并提出SegAnyPET——首个针对PET影像的3D可提示分割基础模型，通过跨提示置信学习（CPCL）策略处理标注质量不一致问题，在已见和未见目标上均大幅超越现有基础模型和任务专用模型。

**[Semi-Supervised Deep Transfer For Regression Without Domain Alignment](medical_imaging/semi-supervised_deep_transfer_for_regression_without_domain_alignment.md)**

:   提出 CRAFT（Contradistinguisher-based Regularization Approach for Flexible Training），一种无需源数据、无需域对齐的半监督迁移学习框架，专门面向回归任务，通过联合优化监督损失和基于 Contradistinguisher 的无监督正则项在标签稀缺场景下显著提升预测性能。

**[Sic Similarity-Based Interpretable Image Classification With Neural Networks](medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)**

:   提出 SIC，一个同时提供局部、全局和忠实解释的内在可解释神经网络：通过从训练图像中提取类别代表性的支持向量，基于 B-cos 变换计算输入与支持向量的相似度进行分类，在保持与黑盒模型相当准确率的同时，提供像素级贡献图和基于案例推理的全局解释，在 FunnyBirds 基准上 9 项可解释性指标中 8 项超越 ProtoPNet。

**[Simmlm A Simple Framework For Multi-Modal Learning With Missing Modality](medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)**

:   提出 SimMLM，一个简洁高效的多模态缺失学习框架，由动态模态专家混合架构（DMoME）和 More vs. Fewer（MoFe）排序损失组成，在脑肿瘤分割和多模态分类任务上以更少参数和计算量全面超越 SOTA，同时提供模态重要性可解释性。

**[Teethgenerator A Two-Stage Framework For Paired Pre- And Post-Orthodontic 3D Den](medical_imaging/teethgenerator_a_two-stage_framework_for_paired_pre-_and_post-orthodontic_3d_den.md)**

:   提出 TeethGenerator，一个两阶段框架用于生成配对的正畸前后 3D 牙齿点云模型，Stage I 用 VQ-VAE+扩散模型生成矫正后牙齿形态，Stage II 用 Transformer 根据风格模型生成对应的矫正前牙齿排列。

**[Toward Long-Tailed Online Anomaly Detection Through Class-Agnostic Concepts](medical_imaging/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)**

:   本文提出长尾在线异常检测（LTOAD）新任务和benchmark，核心创新是用可学习的"类无关概念集"替代传统的类标签依赖，结合Concept VQ-VAE和综合prompt学习框架，在不需要类标签的情况下于offline和online场景下均达到SOTA。

**[Ukbob One Billion Mri Labeled Masks For Generalizable 3D Medical Image Segmentat](medical_imaging/ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)**

:   本文构建了UKBOB——迄今最大的标注医学影像分割数据集（51,761个MRI 3D样本，72类器官，13.7亿2D分割mask），提出Specialized Organ Label Filter (SOLF)清洗自动标注和Entropy Test-Time Adaptation (ETTA)处理带噪标签的域迁移，训练的Swin-BOB基础模型在BRATS和BTCV基准上达到SOTA。

**[Vector Contrastive Learning For Pixel-Wise Pretraining In Medical Vision](medical_imaging/vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)**

:   提出向量对比学习（Vector CL），将标准对比学习从二值优化问题重新表述为向量回归问题，通过建模特征距离来量化分散程度，解决像素级医学视觉预训练中的"过度分散"问题，在 8 个下游任务上显著优于 17 种方法。

**[Victr Vital Consistency Transfer For Pathology Aware Image Synthesis](medical_imaging/victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)**

:   > 提出 ViCTr 两阶段框架，结合 Rectified Flow 与 Tweedie 校正的扩散过程实现高保真的病理感知医学图像合成，将推理步数从50步降至3-4步，并首次实现分级严重程度的腹部MRI病理合成。

**[Visual Surface Wave Elastography Revealing Subsurface Physical Properties Via Vi](medical_imaging/visual_surface_wave_elastography_revealing_subsurface_physical_properties_via_vi.md)**

:   本文提出 VSWE（Visual Surface Wave Elastography），仅通过一段表面波传播的视频，提取色散关系并结合基于物理的有限元优化，推断介质的亚表面厚度和刚度参数，在模拟和真实明胶实验中均实现了高精度的参数恢复，为居家健康监测提供了概念验证。

---

## 🎯 目标检测 { #object_detection }

**[3Dmood Lifting 2D To 3D For Monocular Openset Object Detecti](object_detection/3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)**

:   提出首个端到端的单目开放集3D目标检测器3D-MOOD，通过将开放集2D检测"提升"到3D空间，结合几何感知3D query生成与canonical image space设计，在Omni3D闭集和Argoverse 2/ScanNet开集基准上均达到SOTA。

**[Advancing Textual Prompt Learning With Anchored Attributes](object_detection/advancing_textual_prompt_learning_with_anchored_attributes.md)**

:   本文提出 ATPrompt，通过在文本 prompt 中嵌入通用属性 token（如颜色、形状），将软 prompt 的学习空间从一维类别级别拓展到多维属性级别，作为即插即用的模块可无缝集成到现有文本 prompt 学习方法中，在 11 个数据集上一致性提升基线性能。

**[Adversarial Attention Perturbations For Large Object Detection Transformers](object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)**

:   本文提出 AFOG（Attention-Focused Offensive Gradient），一种架构无关的对抗攻击方法，通过可学习注意力机制聚焦扰动到图像脆弱区域，仅需 10 次迭代即可在视觉不可察觉的扰动下将 12 种检测 Transformer 的 mAP 最高降低 37.8 倍，同时在 CNN 检测器上也优于现有方法。

**[Anchor Token Matching Implicit Structure Locking For Training-Free Ar Image Edit](object_detection/anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)**

:   提出 ISLock，首个面向自回归(AR)视觉生成模型的无训练图像编辑方法，通过锚点 Token 匹配(ATM)在隐空间中隐式对齐自注意力模式，实现结构一致的文本引导图像编辑。

**[Attention To Neural Plagiarism Diffusion Models Can Plagiarize Your Copyrighted ](object_detection/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)**

:   揭示"神经抄袭"威胁——扩散模型可轻松复制受版权保护的图像（包括受水印保护的图像），提出基于"锚点与垫片"的通用攻击框架，通过在交叉注意力机制中搜索扰动实现从粗到细的语义修改，绕过从可见商标到隐形水印的各类版权保护。

**[Augmenting Moment Retrieval Zero-Dependency Two-Stage Learning](object_detection/augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)**

:   提出 AMR 框架，通过 Splice-and-Boost 数据增强策略和冷启动-蒸馏两阶段训练，在不依赖任何外部数据/预训练模型的前提下，大幅提升视频时刻检索的边界感知能力和语义辨别力，在 QVHighlights 上超越 SOTA +5%。

**[Automated Model Evaluation For Object Detection Via Prediction Consistency And R](object_detection/automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)**

:   本文提出PCR（Prediction Consistency and Reliability），一种无需人工标注即可估计目标检测模型性能的自动化评估方法，通过分析NMS前后边界框的空间一致性和置信度可靠性来估计mAP，并构建了基于图像腐蚀的元数据集以实现更现实和可扩展的评估。

**[Chartpoint Guiding Mllms With Grounding Reflection For Chart Reasoning](object_detection/chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)**

:   提出PointCoT方法，将反思性视觉定位（bounding box）集成到图表推理的思维链中，使MLLM在每个推理步骤都能与图表视觉内容交互验证，并构建了包含19.2K高质量样本的ChartPoint-SFT-62k数据集，在ChartBench上实现+5.04%的提升。

**[Diffdoctor Diagnosing Image Diffusion Models Before Treating](object_detection/diffdoctor_diagnosing_image_diffusion_models_before_treating.md)**

:   提出 DiffDoctor，首个利用像素级反馈微调扩散模型的方法：先训练鲁棒的 artifact 检测器（1M+ 样本，类别平衡策略），再通过最小化合成图中每个像素的 artifact 置信度反向传播梯度到扩散模型，使其在未见 prompt 上也能显著减少 artifact 生成。

**[Diffusion Curriculum Synthetic-To-Real Data Curriculum Via Image-Guided Diffusio](object_detection/diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)**

:   利用扩散模型的图像引导强度控制生成从合成到真实的连续谱系数据，设计"扩散课程学习（DisCL）"策略在训练不同阶段自适应选择最优引导级别的合成数据，有效解决长尾分类和低质量数据学习问题。

**[Distil Data-Free Inversion Of Suspicious Trojan Inputs Via Latent Diffusion](object_detection/distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)**

:   DISTIL 提出一种无需干净数据的木马触发器反演方法，通过在预训练引导扩散模型的潜空间中搜索触发器模式（而非像素空间），并注入均匀噪声正则化，有效区分真实后门触发器和对抗扰动，在 BackdoorBench 上精度最高提升 7.1%。

**[Dynamicdino Finegrained Mixture Of Experts Tuning For Realti](object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)**

:   首次将Mixture of Experts引入实时开放词汇目标检测器，通过MoE-Tuning将Grounding DINO 1.5 Edge从dense模型扩展为动态推理框架，提出细粒度专家分解和预训练权重分配策略，仅用1.56M开源数据超越使用20M私有数据训练的原版模型。

**[Ea-Kd Entropy-Based Adaptive Knowledge Distillation](object_detection/ea-kd_entropy-based_adaptive_knowledge_distillation.md)**

:   提出 EA-KD，一种基于信息熵的即插即用知识蒸馏方法：通过结合 teacher 和 student 输出的熵值动态重加权蒸馏损失，优先学习高熵（高信息量）样本，在图像分类、目标检测和 LLM 蒸馏任务上均一致提升多种 KD 框架的性能，且计算开销可忽略。

**[Evrt-Detr Latent Space Adaptation Of Image Detectors For Event-Based Vision](object_detection/evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)**

:   提出I2EvDet框架，通过在冻结的RT-DETR检测器的潜空间中插入轻量级RNN时序模块，以最小的架构修改将主流图像检测器适配为事件相机视频检测模型，在Gen1和1Mpx基准上分别取得+2.3和+1.4 mAP的SOTA。

**[Fakeradar Probing Forgery Outliers To Detect Unknown Deepfake Videos](object_detection/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)**

:   提出FakeRadar深度伪造视频检测框架，通过Forgery Outlier Probing在特征空间中主动生成模拟未知伪造的异常值样本，并设计Outlier-Guided Tri-Training三分类优化策略，在跨数据集/跨操纵类型评估中显著超越现有方法。

**[Few-Shot Pattern Detection Via Template Matching And Regression](object_detection/few-shot_pattern_detection_via_template_matching_and_regression.md)**

:   本文提出TMR方法，通过经典模板匹配结合支持条件化边界框回归，实现了对任意模式（包括非物体级模式）的小样本检测，同时引入RPINE数据集覆盖更广泛的重复模式，在多个基准上超越现有FSCD方法并展现出强大的跨数据集泛化能力。

**[Forgelens Data-Efficient Forgery Focus For Generalizable Forgery Image Detection](object_detection/forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)**

:   提出 ForgeLens，一个基于冻结 CLIP-ViT 的特征引导框架，通过轻量级的权重共享引导模块（WSGM）和伪造感知特征集成器（FAFormer），引导冻结预训练网络聚焦伪造特征，仅用 1% 训练数据即达到 SOTA 泛化性能。

**[Intervening In Black Box Concept Bottleneck Model For Enhancing Human Neural Net](object_detection/intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)**

:   提出 CBM-HNMU 框架，通过概念瓶颈模型（CBM）逼近黑盒模型的推理过程，自动识别并修正有害概念，再将修正后的知识蒸馏回黑盒模型，实现超越样本级别的系统性模型干预与准确率提升。

**[Is Less More Exploring Token Condensation As Training-Free Test-Time Adaptation](object_detection/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)**

:   提出 Token Condensation as Adaptation（TCA），一种免训练的测试时自适应方法，通过领域感知的 token 库（DTR）引导跨头 token 裁剪/合并和 logits 自校正，在不修改模型参数的情况下，将 CLIP/SigLIP 系列的跨数据集性能提升最高 21.4%，同时减少 12.2%-48.9% 的 GFLOPs。

**[Large-Scale Pre-Training For Grounded Video Caption Generation](object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)**

:   提出 GROVE 模型和大规模自动标注方法，构建包含 1M 视频的 HowToGround1M 预训练数据集和 3513 个视频的手动标注 iGround 数据集，实现联合视频字幕生成与多目标时空边界框定位，在 iGround、VidSTG、ActivityNet-Entities 等数据集上取得 SOTA。

**[Lmm-Det Make Large Multimodal Models Excel In Object Detection](object_detection/lmm-det_make_large_multimodal_models_excel_in_object_detection.md)**

:   提出 LMM-Det，通过系统分析发现大型多模态模型在目标检测中核心瓶颈是低召回率，并通过数据分布调整（伪标签增强）和推理优化（按类别逐一检测）将 LMM 的 COCO AP 从 0.2 提升至 47.5，无需任何额外专用检测模块。

**[Measuring The Impact Of Rotation Equivariance On Aerial Object Detection](object_detection/measuring_the_impact_of_rotation_equivariance_on_aerial_object_detection.md)**

:   提出 MessDet，一个基于旋转等变网络的航空目标检测器，通过新型下采样过程实现严格旋转等变性，并引入旋转等变通道注意力和多分支检测头，在 DOTA 等数据集上以极低参数量达到 SOTA 性能。

**[Openrsd Towards Open-Prompts For Object Detection In Remote Sensing Images](object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)**

:   提出OpenRSD通用遥感开放提示目标检测框架，支持文本和图像多模态提示，集成对齐头和融合头平衡速度与精度，配合三阶段训练流水线和47万张图像的ORSD+数据集，在7个公开数据集上取得最优平均性能，同时保持20.8 FPS实时推理。

**[Pasg A Closed-Loop Framework For Automated Geometric Primitive Extraction And Se](object_detection/pasg_a_closed-loop_framework_for_automated_geometric_primitive_extraction_and_se.md)**

:   提出 PASG（Primitive-Aware Semantic Grounding），一个闭环框架，通过自动化几何基元提取（关键点、功能轴、主轴）和 VLM 驱动的语义锚定，将低层几何特征与高层任务语义动态耦合，在机器人操作任务中实现了接近人工标注的性能，并构建了 Robocasa-PA 基准和微调模型 Qwen2.5VL-PA。

**[Revisiting Adversarial Patch Defenses On Object Detectors Unified Evaluation Lar](object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)**

:   系统性重新审视 11 种对抗补丁防御方法，建立首个补丁防御基准（含 13 种攻击、11 个检测器、4 种度量），构建 94,000 张图像的大规模 APDE 数据集，并揭示三个关键新发现：自然补丁防御难点在于数据分布而非高频、补丁检测精度与防御性能不一致、自适应攻击可绕过大多数现有防御。

**[Semantic Discrepancy-Aware Detector For Image Forgery Identification](object_detection/semantic_discrepancy-aware_detector_for_image_forgery_identification.md)**

:   提出语义差异感知检测器（SDD），通过语义 token 采样、概念级伪造差异学习和低层伪造特征增强三个模块，利用重建学习将 CLIP 的视觉语义概念空间与伪造空间进行细粒度对齐，在 UnivFD 和 SynRIS 基准上取得 SOTA 性能（$ap_m$ 98.51%，AUROC 95.1%）。

**[Sfuod Source-Free Unknown Object Detection](object_detection/sfuod_source-free_unknown_object_detection.md)**

:   提出 Source-Free Unknown Object Detection (SFUOD) 新场景，并设计 CollaPAUL 框架，通过协作调优融合源域和目标域知识 + 基于主轴的未知物体伪标签分配，在无源数据条件下同时检测已知和未知物体。

**[Sim-Detr Unlock Detr For Temporal Sentence Grounding](object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)**

:   系统分析了 DETR 在时序语句定位 (TSG) 任务中的异常行为根因——查询间冲突和查询内全局-局部矛盾，并提出两个简单修改（Query Grouping & Ranking + Global-Local Bridging）构成 Sim-DETR，解锁 DETR 在 TSG 任务的全部潜力。

**[Sketchsplat 3D Edge Reconstruction Via Differentiable Multi-View Sketch Splattin](object_detection/sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)**

:   提出 SketchSplat，将 3D 边缘表示为参数化 sketch（直线+Bézier曲线），通过从 sketch 采样高斯点进行可微渲染来直接优化边缘参数，同时提出自适应拓扑控制和改进的 2D 边缘检测器，在 CAD 数据集上实现 SOTA 的准确性、完整性和紧凑性。

**[Sparse-Dense Side-Tuner For Efficient Video Temporal Grounding](object_detection/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)**

:   提出 SDST（Sparse-Dense Side-Tuner），首个无锚框（anchor-free）的 Side-Tuning 架构，通过稀疏-稠密双流设计同时处理时刻检索（MR）和高光检测（HD），并提出 Reference-based Deformable Self-Attention（RDSA）解决可变形注意力的上下文缺失问题，在 QVHighlights、TACoS、Charades-STA 上取得 SOTA 或高度竞争性结果，同时将可训练参数量减少至现有 SOTA 的 27%。

**[Synchronization Of Multiple Videos](object_detection/synchronization_of_multiple_videos.md)**

:   提出 Temporal Prototype Learning (TPL)，一个基于原型的视频同步框架，从预训练模型提取的高维嵌入中构建共享的紧凑1D表征，通过学习统一的原型序列锚定关键动作阶段来对齐多个视频，首次解决了生成式AI视频的同步问题。

**[The Devil Is In The Spurious Correlations Boosting Moment Retrieval With Dynamic](object_detection/the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)**

:   首次揭示文本查询与视频背景帧之间的虚假相关性是时刻检索性能瓶颈的根本原因，提出 TD-DETR 框架通过动态上下文视频合成和文本-动态交互增强两个策略来缓解该问题，在 QVHighlights 和 Charades-STA 上达到 SOTA。

**[Uncertainty-Aware Gradient Stabilization For Small Object Detection](object_detection/uncertainty-aware_gradient_stabilization_for_small_object_detection.md)**

:   揭示了传统目标定位方法在小目标上存在因损失曲率陡峭导致的梯度不稳定问题，提出 UGS（不确定性感知梯度稳定化）框架，通过分类式定位 + 不确定性最小化 + 不确定性引导精炼三个组件来稳定梯度，显著提升小目标检测性能。

**[Upre Zero-Shot Domain Adaptation For Object Detection Via Unified Prompt And Rep](object_detection/upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)**

:   提出 UPRE 框架，通过联合优化多视角域提示（MDP）和统一表示增强（URE）来同时缓解零样本域自适应目标检测中的检测偏差和域偏差，在恶劣天气、跨城市、虚拟到现实三类场景的九个数据集上取得 SOTA 性能。

**[Viewsrd 3D Visual Grounding Via Structured Multi-View Decomposition](object_detection/viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)**

:   提出 ViewSRD 框架，将 3D 视觉定位建模为结构化多视角分解过程：通过 SRD 模块将复杂多锚点查询解耦为简单单锚点查询，并引入跨模态一致视角 token (CCVT) 解决视角变化导致的空间描述不一致问题。

**[Visrl Intention-Driven Visual Perception Via Reinforced Reasoning](object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)**

:   VisRL是首个将强化学习应用于意图驱动视觉感知的框架，通过迭代DPO训练让大多模态模型学会根据查询意图自主选择关注区域（预测bounding box），无需昂贵的中间bounding box标注即可实现比SFT更强的视觉推理能力。

**[Visual-Rft Visual Reinforcement Fine-Tuning](object_detection/visual-rft_visual_reinforcement_fine-tuning.md)**

:   Visual-RFT将DeepSeek R1的强化学习+可验证奖励(RLVR)范式从数学/代码领域扩展到视觉感知任务，设计了IoU奖励（目标检测）和CLS奖励（分类）等任务特异的可验证奖励函数，在细粒度分类、少样本检测、推理定位等任务上以极少数据大幅超越SFT。

**[Visual Modality Prompt For Adapting Vision-Language Object Detectors](object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)**

:   提出 ModPrompt，一种基于编码器-解码器的视觉提示策略，将视觉-语言目标检测器（如 YOLO-World、Grounding DINO）适应到红外和深度等新模态，同时保留零样本检测能力。

**[Voccl3D A Video Benchmark Dataset For 3D Human Pose And Shape Estimation Under R](object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)**

:   提出 VOccl3D，一个基于3DGS渲染的大规模合成视频数据集（25万帧，400视频序列），专注于真实遮挡场景的3D人体姿态与形状估计，在该数据集上微调的模型显著提升了遮挡场景下的HPS性能。

**[Yolo-Count Differentiable Object Counting For Text-To-Image Generation](object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)**

:   提出 YOLO-Count，一个基于 YOLO 架构的全可微分开放词汇目标计数模型，通过创新的"基数图"（cardinality map）回归目标和混合强弱监督训练策略，在通用计数和文本到图像生成的数量控制两个任务上均达到 SOTA。

**[Yoloe Realtime Seeing Anything](object_detection/yoloe_realtime_seeing_anything.md)**

:   提出YOLOE，在YOLO架构中统一支持文本提示、视觉提示和无提示三种开放场景的检测和分割，通过RepRTA（可重参数化区域-文本对齐）、SAVPE（语义激活视觉提示编码器）和LRPC（懒惰区域-提示对比）三个设计实现高效率高性能，以3x更少的训练成本在LVIS上超越YOLO-World v2。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Alocc Adaptive Lifting-Based 3D Semantic Occupancy And Cost Volume-Based Flow Pr](image_restoration/alocc_adaptive_lifting-based_3d_semantic_occupancy_and_cost_volume-based_flow_pr.md)**

:   提出ALOcc框架，通过遮挡感知自适应提升机制、语义原型占用头和BEV代价体积流预测三项创新，在多个3D语义占用和占用流预测基准上取得SOTA，同时提供实时到高精度的多种模型变体。

**[Alocc Adaptive Liftingbased 3D Semantic Occupancy And Cost V](image_restoration/alocc_adaptive_liftingbased_3d_semantic_occupancy_and_cost_v.md)**

:   提出ALOcc框架，通过遮挡感知的自适应提升机制、语义原型对齐和BEV代价体flow预测三个改进，在多个占据预测基准上取得SOTA，同时保持较高推理速度。

**[Benchmarking Burst Superresolution For Polarization Images N](image_restoration/benchmarking_burst_superresolution_for_polarization_images_n.md)**

:   针对偏振相机"光效低、分辨率低、噪声大"的硬件瓶颈，构建了两个专用数据集（PolarNS用于噪声统计分析，PolarBurstSR用于burst超分的训练/评测），提出偏振噪声传播分析模型，并将5种SOTA burst超分方法适配到偏振域，证明偏振专用训练在强度图(s0)和偏振角(AoLP)重建上显著优于RGB通用训练。

**[Blind2Sound Self-Supervised Image Denoising Without Residual Noise](image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)**

:   提出 Blind2Sound 框架，通过自适应重可见损失（adaptive re-visible loss）感知噪声水平并实现个性化去噪，配合 Cramer Gaussian 损失提升噪声参数估计精度，在自监督盲去噪中消除残余噪声，性能超越同期所有自监督方法甚至部分有监督基线。

**[Blind Noisy Image Deblurring Using Residual Guidance Strateg](image_restoration/blind_noisy_image_deblurring_using_residual_guidance_strateg.md)**

:   提出残差引导策略（RGS），在图像金字塔的粗到细估计过程中，利用相邻粗尺度的卷积残差经 guided filter 去噪后校正当前尺度的模糊图像，从而在高噪声（σ=0.1）下显著提升盲去模糊的核估计精度和恢复质量，无需训练即超越多种深度学习方法。

**[Closed-Loop Transfer For Weakly-Supervised Affordance Grounding](image_restoration/closed-loop_transfer_for_weakly-supervised_affordance_grounding.md)**

:   提出LoopTrans闭环知识迁移框架，通过共享CAM实现外中心-自中心图像的统一知识激活，利用像素级伪掩码将粗激活精炼为精确定位，并通过去噪蒸馏将自中心定位反馈增强外中心知识提取，在AGD20K上全面超越SOTA。

**[Consistent Time-Of-Flight Depth Denoising Via Graph-Informed Geometric Attention](image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)**

:   GIGA-ToF 提出了一种基于运动不变图结构融合的 ToF 深度去噪网络，通过跨帧图注意力机制和 MAP 问题的算法展开，同时增强了时序稳定性和空间锐度，并在合成和真实数据上展现了优秀的泛化能力。

**[Cwnet Causal Wavelet Network For Low-Light Image Enhancement](image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)**

:   提出因果小波网络CWNet，通过结构因果模型将低光增强中的语义信息视为因果因子、亮度/颜色退化视为非因果因子，结合小波变换骨干网络实现频域特征的精细化恢复。

**[Decouple To Reconstruct High Quality Uhd Restoration Via Active Feature Disentan](image_restoration/decouple_to_reconstruct_high_quality_uhd_restoration_via_active_feature_disentan.md)**

:   提出 D²R-UHDNet 框架，通过 Controlled Differential Disentangled VAE（CD²-VAE）将退化图像主动解耦为退化主导潜空间和背景主导特征，并利用复数域可逆多尺度融合网络处理背景特征，在仅 1M 参数下实现六项 UHD 复原任务的 SOTA。

**[Devil Is In The Uniformity Exploring Diverse Learners Within Transformer For Ima](image_restoration/devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)**

:   针对标准Multi-Head Attention (MHA)中各head使用均匀子空间导致的冗余问题，提出HINT模型，通过异构层级多头注意力(HMHA)和Query-Key缓存更新(QKCU)机制增强head间多样性与交互，在5类图像恢复任务的12个benchmark上取得SOTA结果。

**[Eamamba Efficient All-Around Vision State Space Model For Image Restoration](image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)**

:   本文提出EAMamba框架，通过多头选择性扫描模块（MHSSM）和全方位扫描策略（all-around scanning），在不增加计算复杂度和参数量的情况下实现多方向扫描，解决了Vision Mamba在图像恢复中的计算开销和局部像素遗忘问题，在超分辨率、去噪、去模糊、去雾等任务上取得了31-89%的FLOPs降低同时保持优异性能。

**[Efficient Concertormer For Image Deblurring And Beyond](image_restoration/efficient_concertormer_for_image_deblurring_and_beyond.md)**

:   提出 Concertormer，通过将自注意力分解为全局 Concertino 和局部 Ripieno 两个分量，同时引入跨维度通信模块和门控深度卷积 MLP，实现了线性复杂度下的全局-局部特征建模，在去模糊及其他图像复原任务上取得 SOTA 性能。

**[Emulating Self-Attention With Convolution For Efficient Image Super-Resolution](image_restoration/emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)**

:   观察到自注意力在相邻层之间的特征和注意力图高度相似（89%/87%），提出用共享大核卷积和动态卷积核组成的 ConvAttn 模块替代大部分自注意力，同时首次在轻量级超分辨率中引入 Flash Attention 将窗口扩展到 32×32，以极低延迟和内存代价实现了 SOTA 性能。

**[Enhancing Image Restoration Transformer Via Adaptive Translation Equivariance](image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)**

:   系统研究了平移等变性（Translation Equivariance, TE）对图像修复网络收敛速度和泛化能力的影响，提出滑动键值自注意力（SkvSA）及其自适应版本（ASkvSA）和下采样自注意力（DSA），构建了 TEAFormer，在超分、去模糊、去噪等多个任务上取得 SOTA，同时保持线性复杂度。

**[Exploiting Diffusion Prior For Task-Driven Image Restoration](image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)**

:   提出 EDTR 方法，通过预修复+部分扩散和短步去噪策略，有效利用扩散模型先验恢复与高层视觉任务相关的细节，在复杂退化场景下显著提升分类、分割和检测性能。

**[Foundir Unleashing Million-Scale Training Data To Advance Foundation Models For ](image_restoration/foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)**

:   构建了首个百万级真实世界配对图像修复数据集（含 20 种退化类型），并提出 FoundIR 框架，通过退化无关的泛化器模型与退化感知的专家模型协同，在 24 个基准上突破了图像修复的性能天花板。

**[Generic Event Boundary Detection Via Denoising Diffusion](image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)**

:   DiffGEBD 首次将扩散模型引入通用事件边界检测（GEBD），通过将边界预测建模为从随机噪声到合理边界分布的去噪过程，利用 Classifier-Free Guidance 控制预测多样性，并提出了对称 F1 和 Diversity Score 两项新评估指标来衡量多预测场景下的质量与多样性。

**[Learning Pixel-Adaptive Multi-Layer Perceptrons For Real-Time Image Enhancement](image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)**

:   提出 BPAM 框架，将双边网格的空间建模能力与 MLP 的非线性映射能力相结合，通过为每个像素动态生成独特的微型 MLP 参数实现高质量、实时的图像增强。

**[Lightweight And Fast Real-Time Image Enhancement Via Decomposition Of The Spatia](image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)**

:   通过将3D LUT分解为2D LUT的线性组合并进一步做SVD，结合缓存高效的空间特征融合结构，实现了在保持空间感知能力的同时将模型参数减少84%、4K分辨率推理加速2.8倍的轻量实时图像增强。

**[Low-Light Image Enhancement Using Event-Based Illumination Estimation](image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)**

:   RetinEV 提出利用事件相机的"时间映射事件"（temporal-mapping events，由透射率调制触发）而非传统"运动事件"进行光照估计，结合 Retinex 理论将低光照图像分解为光照和反射率分量，通过光照辅助反射率增强（IRE）模块实现高质量低光照图像增强，在 640×480 分辨率下达到 35.6 FPS 实时速度。

**[Metric Convolutions A Unifying Theory To Adaptive Image Convolutions](image_restoration/metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)**

:   从度量几何视角统一解释现有各种自适应卷积（标准/膨胀/平移/可变形），并基于显式 Randers 度量的单位球采样提出 Metric Convolution，以更少参数实现更好的几何正则化和泛化能力。

**[Mobileie An Extremely Lightweight And Effective Convnet For Real-Time Image Enha](image_restoration/mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)**

:   提出 MobileIE，一个仅有约 4K 参数的极致轻量 CNN 框架，通过多分支重参数化卷积（MBRConv）、特征自变换（FST）模块、分层双路径注意力（HDPA）以及增量权重优化（IWO）策略，首次在移动设备上实现超过 1100 FPS 的实时图像增强，同时在低光增强、水下增强和 ISP 三个任务上取得最优的速度-性能平衡。

**[Mp-Hsir A Multi-Prompt Framework For Universal Hyperspectral Image Restoration](image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)**

:   提出 MP-HSIR 框架，通过整合光谱提示（通用低秩光谱模式）、文本提示和视觉提示三种模态的引导信息，构建了统一的高光谱图像复原模型，在包含去噪、去模糊、超分辨率、修复、去雾、波段补全等 9 个 HSI 复原任务上全面超越现有 all-in-one 方法和多个任务专用方法。

**[Outlier-Aware Post-Training Quantization For Image Super-Resolution](image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)**

:   提出一种面向图像超分辨率的离群值感知后训练量化方法，通过双区域分段线性量化平衡离群值保留与正常激活精度，并引入敏感度感知微调策略使模型关注量化敏感层，在 W4A4 设置下大幅超越现有 PTQ 方法并接近 QAT 性能。

**[Pre-Mamba A 4D State Space Model For Ultra-High-Frequent Event Camera Deraining](image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)**

:   首个基于点的事件相机去雨框架，利用4D事件云表示和多尺度状态空间模型（MS3M），在保持微秒级时间精度的同时实现高效去雨，仅0.26M参数即达到SOTA性能。

**[Robust Adverse Weather Removal Via Spectral-Based Spatial Grouping](image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)**

:   SSGformer 提出一种基于光谱分解和分组注意力的 All-in-One 恶劣天气图像复原方法：利用 Sobel 算子提取高频边缘信息和 SVD 分析低频退化纹理，将二者融合后生成空间分组掩码（grouping-mask），在组内执行通道和空间注意力以实现对多种天气退化（雨、雪、雾、雨滴）的鲁棒去除。

**[Self-Calibrated Variance-Stabilizing Transformations For Real-World Image Denois](image_restoration/self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)**

:   提出 Noise2VST 框架，通过自监督学习一个无模型假设的方差稳定化变换（VST），使现成的高斯去噪器无需额外训练即可高效处理真实世界噪声图像。

**[Towards A Universal Image Degradation Model Via Content-Degradation Disentanglem](image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)**

:   提出首个通用图像退化模型，通过"压缩解纠缠"方法分离退化信息与图像内容，引入 IDEN 和 IDA 层处理非均匀退化，实现跨退化类型的编码、合成和迁移，可作为 plug-in 模块将非盲图像恢复方法转化为盲方法。

**[Uniphys Unified Planner And Controller With Diffusion For Flexible Physics-Based](image_restoration/uniphys_unified_planner_and_controller_with_diffusion_for_flexible_physics-based.md)**

:   提出 UniPhys，一个基于扩散模型的行为克隆框架，将运动规划和物理控制统一到单一模型中，通过 Diffusion Forcing 训练范式处理累积预测误差，实现了灵活的文本驱动、速度控制、目标达到和动态避障等多任务物理角色运动生成。

**[Unires Universal Image Restoration For Complex Degradations](image_restoration/unires_universal_image_restoration_for_complex_degradations.md)**

:   提出 UniRes——一个基于扩散模型的通用图像复原框架，通过多任务训练学习超分辨率、运动去模糊、散焦去模糊和去噪等专家知识，推理时通过灵活组合不同任务的隐空间预测权重来端到端地处理真实世界中的任意复杂退化组合。

---

## 🛡️ AI安全 { #ai_safety }

**[A Framework For Double-Blind Federated Adaptation Of Foundation Models](ai_safety/a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)**

:   本文提出BlindFed框架，通过全同态加密（FHE）友好的架构改造、两阶段分割学习和隐私增强策略，实现了基础模型的"双盲"联邦适配——数据方看不到模型，服务方看不到数据，在CIFAR-10上达到94.28%准确率，接近LoRA的95.92%。

**[A Framework For Doubleblind Federated Adaptation Of Foundati](ai_safety/a_framework_for_doubleblind_federated_adaptation_of_foundati.md)**

:   BlindFed提出了双盲联邦基础模型适配框架：通过FHE友好的架构重设计（多项式近似非线性操作）+ 两阶段分割学习（离线知识蒸馏 + 在线加密推理）+ 隐私增强（样本置换 + 随机块采样），在数据方看不到模型、模型方看不到数据的约束下实现了接近LoRA的适配精度。

**[Active Membership Inference Test Amint Enhancing Model Audit](ai_safety/active_membership_inference_test_amint_enhancing_model_audit.md)**

:   提出Active MINT（aMINT），将成员推断检测作为训练时的优化目标，通过多任务学习让被审计模型与MINT模型联合训练、共享早期特征层，在不显著损失主任务性能的前提下，将训练数据的识别准确率从被动MINT的~60%大幅提升至80%以上。

**[Active Membership Inference Test Amint Enhancing Model Auditability With Multi-T](ai_safety/active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)**

:   本文提出 Active MINT（aMINT），一种多任务学习框架，在训练审核模型的同时联合训练 MINT 模型，使模型能够以超过 80% 的准确率检测特定数据是否被用于训练，显著优于現有的被动 MINT 和成员推断攻击方法。

**[Ask And Remember A Questions-Only Replay Strategy For Continual Visual Question ](ai_safety/ask_and_remember_a_questions-only_replay_strategy_for_continual_visual_question_.md)**

:   提出QUAD——一种仅存储过去任务问题（不存储图像）的持续VQA方法，通过问题重放和注意力一致性蒸馏，在保护隐私的同时超越存储图像的现有方法。

**[Backdoor Mitigation By Distance-Driven Detoxification](ai_safety/backdoor_mitigation_by_distance-driven_detoxification.md)**

:   本文提出Distance-Driven Detoxification（D3），将后门防御重新表述为约束优化问题——最大化微调后模型权重与中毒初始权重的距离，同时约束干净样本损失不超过阈值，从而有效逃逸"后门区域"，在7种SOTA攻击上取得最优或次优防御效果。

**[Backdooring Self-Supervised Contrastive Learning By Noisy Alignment](ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)**

:   提出Noisy Alignment（NA）方法，通过显式压缩投毒图像中的噪声成分来增强自监督对比学习的后门攻击效果，将攻击建模为二维图像布局优化问题，并推导出理论最优参数，在ImageNet-100上ASR提升最高达45.9%。

**[Client2Vec Improving Federated Learning By Distribution Shifts Aware Client Inde](ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)**

:   提出Client2Vec机制，在联邦学习训练前利用CLIP编码器和分布偏移感知索引生成网络（DSA-IGN）为每个客户端生成包含标签和特征分布信息的索引向量，进而改善客户端采样、模型聚合和本地训练三个关键阶段。

**[Controllable Feature Whitening For Hyperparameter-Free Bias Mitigation](ai_safety/controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)**

:   提出可控特征白化(CFW)框架，通过白化变换消除目标特征与偏差特征之间的线性相关性来缓解模型偏差，无需对抗学习或额外正则化超参数，且可通过加权系数平滑控制demographic parity和equalized odds之间的权衡。

**[Enhancing Adversarial Transferability By Balancing Exploration And Exploitation ](ai_safety/enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)**

:   提出Gradient-Guided Sampling (GGS)内迭代采样策略，通过使用上一内迭代的梯度方向引导采样，在平衡Exploitation（攻击强度/损失极大值）和Exploration（跨模型泛化/平坦损失面）的困境中取得突破，在CNN/ViT/MLLM等多架构上显著超越现有迁移攻击方法。

**[Fedmenf Privacy-Preserving Federated Meta-Learning For Neural Fields](ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)**

:   本文首次研究在私有数据场景下的联邦神经场（Neural Fields）元学习问题，揭示了现有联邦元学习方法在神经场任务中的严重隐私泄露机制，并提出FedMeNF，通过隐私保护损失函数正则化局部元梯度中的隐私信息，在保持快速优化能力的同时有效保护客户端数据隐私。

**[Find A Scapegoat Poisoning Membership Inference Attack And Defense To Federated ](ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)**

:   提出 FedPoisonMIA，一种基于角度偏差最大化的联邦学习投毒成员推理攻击，同时提出 Angular Trimmed-mean (ATM) 防御机制，通过角度距离过滤恶意梯度。

**[Fret Feature Redundancy Elimination For Test Time Adaptation](ai_safety/fret_feature_redundancy_elimination_for_test_time_adaptation.md)**

:   本文提出特征冗余消除（FRET）作为测试时自适应（TTA）的新视角，发现分布偏移时嵌入特征冗余度显著增加，并设计了S-FRET（直接最小化冗余分数）和G-FRET（基于GCN的注意力-冗余分解+双层优化）两种方法，G-FRET在多种架构和数据集上达到SOTA性能。

**[Geminio Language-Guided Gradient Inversion Attacks In Federated Learning](ai_safety/geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)**

:   本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。

**[Lora-Fair Federated Lora Fine-Tuning With Aggregation And Initialization Refinem](ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)**

:   本文提出LoRA-FAIR方法，通过在服务器端引入残差校正项 $\Delta\mathbf{B}$ 来同时解决联邦学习+LoRA微调中的服务器端聚合偏差和客户端初始化滞后两大挑战，在ViT和MLP-Mixer模型上一致超越现有联邦微调方法，且不增加通信开销。

**[Mind The Cost Of Scaffold Benign Clients May Even Become Accomplices Of Backdoor](ai_safety/mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)**

:   提出 BadSFL，首个针对 Scaffold 联邦学习算法的后门攻击方法，通过篡改控制变量（control variate）将良性客户端变为"帮凶"，结合 GAN 数据增强和预测全局模型收敛方向的优化策略，在 non-IID 场景下实现了攻击停止后仍持续 60+ 轮的后门效果，持久性是基线方法的 3 倍。

**[Oasis One Image Is All You Need For Multimodal Instruction Data Synthesis](ai_safety/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)**

:   提出Oasis方法，仅需输入图像（无需任何文本提示）即可诱导MLLM自回归生成高质量多模态指令跟随数据，配合精细的指令质量控制机制，合成50万数据给LLaVA-NeXT带来平均3.1%的全面性能提升，且超越其他合成方法。

**[Semantic Alignment And Reinforcement For Data-Free Quantization Of Vision Transf](ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)**

:   提出 SARDFQ 方法解决 ViT 无数据量化（DFQ）中合成图像的**语义失真**和**语义不足**问题，通过注意力先验对齐（APA）引导合成图像的注意力模式与真实图像对齐，通过多语义增强（MSR）优化局部 patch 丰富图像语义，在 ImageNet W4A4 ViT-B 上提升 15.52% Top-1 准确率。

**[Specguard Spectral Projection-Based Advanced Invisible Watermarking](ai_safety/specguard_spectral_projection-based_advanced_invisible_watermarking.md)**

:   SpecGuard 提出将水印信息嵌入到小波分解后的高频子带的频谱域中（通过 FFT 近似的频谱投影），编码端用强度因子增强鲁棒性，解码端利用 Parseval 定理设计可学习阈值进行比特恢复，在保持高图像质量（PSNR>42dB）的同时实现了对畸变、再生成和对抗攻击的全面鲁棒性，超越了现有 SOTA 方法。

**[Staining And Locking Computer Vision Models Without Retraining](ai_safety/staining_and_locking_computer_vision_models_without_retraining.md)**

:   本文提出了无需重训练或微调即可对预训练视觉模型进行"染色"（水印嵌入）和"锁定"（使用保护）的新算法，通过直接修改少量权重植入高选择性检测神经元，并提供了可计算的误报率理论保证，在图像分类和目标检测模型上验证了有效性。

**[Synfer Towards Boosting Facial Expression Recognition With Synthetic Data](ai_safety/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)**

:   提出 SynFER，一个基于扩散模型的面部表情合成框架，通过文本描述 + 面部动作单元 (FAU) 的双重控制实现细粒度表情生成，并引入 FERAnno 标签校准器确保标注可靠性，在自监督、监督、零样本和少样本四种学习范式下均证明合成数据对 FER 的有效性。

**[Towards Adversarial Robustness Via Debiased High-Confidence Logit Alignment](ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)**

:   揭示了逆向对抗攻击（inverse adversarial attack）在对抗训练中导致模型注意力偏移至背景特征的虚假相关性问题，提出 DHAT 方法通过去偏高置信度 logit 正则化（DHLR）和前景 logit 正交增强（FLOE）两个组件来消除这种偏差，在 CIFAR-10/100 和 ImageNet-1K 上取得了 SOTA 的对抗鲁棒性。

**[Vulnerability-Aware Spatio-Temporal Learning For Generalizable Deepfake Video De](ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)**

:   本文提出FakeSTormer，一个细粒度的生成式深度伪造视频检测框架，通过多任务学习同时建模时间和空间脆弱性区域，配合自混合视频（SBV）数据合成策略生成高质量伪造样本，仅用真实数据训练即可在多个跨数据集基准上达到SOTA泛化性能。

---

## 🤖 具身智能 { #robotics }

**[Adaptive Articulated Object Manipulation On The Fly With Foundation Model Reason](robotics/adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)**

:   本文提出 AdaRPG 框架，利用基础视觉-语言模型对铰接物体进行零件级分割和可操作性推理，并借助 GPT-4o 生成高层控制代码以自适应调度原子操作技能，在仿真和真实环境中实现了跨类别零样本泛化操作。

**[Anybimanual Transferring Unimanual Policy For General Bimanual Manipulation](robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)**

:   提出 AnyBimanual，一个即插即用的框架，通过技能管理器和视觉对齐器将预训练的单臂操控策略迁移到通用双臂操控场景，在仅有少量双臂示范的情况下实现显著的多任务泛化能力。

**[Beyond Losses Reweighting Empowering Multi-Task Learning Via The Generalization ](robotics/beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)**

:   从泛化角度出发，将锐度感知最小化（SAM）引入多任务学习，通过分解每个任务的 SAM 梯度为"低损失方向"和"平坦方向"并分别聚合，减少梯度冲突并引导模型进入跨任务共同平坦低损失区域。

**[Bridging Domain Generalization To Multimodal Domain Generalization Via Unified R](robotics/bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)**

:   提出URMMDG框架，通过监督对比学习构建跨模态统一表示空间，并利用互信息最小化解耦类别通用信息与模态/域特定信息，将传统单模态域泛化方法（Mixup、JiGen、IBN-Net）有效迁移到多模态域泛化场景，在EPIC-Kitchens和HAC基准上取得SOTA。

**[Certifiably Optimal Anisotropic Rotation Averaging](robotics/certifiably_optimal_anisotropic_rotation_averaging.md)**

:   提出了一种新的SDP松弛方法，通过强制解落在SO(3)的凸包conv(SO(3))内，首次实现了各向异性代价下的可证明全局最优旋转平均，解决了传统O(3)松弛在各向异性场景下完全失效的问题。

**[Combatvla An Efficient Vision-Language-Action Model For Combat Tasks In 3D Actio](robotics/combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)**

:   提出CombatVLA，一个针对3D动作角色扮演游戏战斗任务的高效3B参数VLA模型，通过Action-of-Thought数据格式和截断推理策略，实现比现有VLM游戏框架快50倍的推理速度，且战斗成功率超越人类玩家。

**[Cosmo Combination Of Selective Memorization For Low-Cost Vision-And-Language Nav](robotics/cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)**

:   提出 COSMO，一种结合选择性记忆的低成本 VLN 架构，通过两个定制化的选择性状态空间模块——Round Selective Scan（RSS，单轮扫描捕获全局上下文）和 Cross-modal Selective State Space Module（CS3，双流跨模态交互）——替代 Transformer 中的高成本注意力机制，以仅 15.5% 参数和 9.3% FLOPs 实现超越基线 DUET 的导航性能。

**[Dexvlg Dexterous Vision-Language-Grasp Model At Scale](robotics/dexvlg_dexterous_vision-language-grasp_model_at_scale.md)**

:   提出DexVLG——首个大规模视觉-语言-灵巧抓取模型，构建了包含174K物体、1.7亿抓取姿态的DexGraspNet 3.0数据集（带部件级语义标注），结合VLM和Flow Matching姿态预测头，在仿真中实现76%+零样本执行成功率，并在真实世界中完成语义对齐的灵巧抓取。

**[Embodied Representation Alignment With Mirror Neurons](robotics/embodied_representation_alignment_with_mirror_neurons.md)**

:   本文受镜像神经元启发，通过对比学习将动作理解（观察他人行为）和具身执行（自主执行动作）的中间表征对齐到共享潜在空间，发现两类模型的表征存在自发对齐现象且与任务成功率相关，显式对齐后在动作识别（+3.3%）和机器人操作（+3.5%）上均获提升。

**[Evolvinggrasp Evolutionary Grasp Generation Via Efficient Preference Alignment](robotics/evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)**

:   提出 EvolvingGrasp，通过 Handpose-wise Preference Optimization (HPO) 和 Physics-Aware Consistency Model (PCM) 实现灵巧抓取姿态的高效进化式生成与人类偏好对齐，在四个基准数据集上取得 SOTA，并实现 30 倍加速。

**[Imanip Skill-Incremental Learning For Robotic Manipulation](robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)**

:   提出 iManip 框架，通过时序回放策略和可扩展 PerceiverIO 架构，使机器人能够在不重新训练的情况下持续学习新的操作技能，同时缓解对已学技能的灾难性遗忘，在 RLBench 上比传统增量基线平均提升 9.4%。

**[Interaction-Merged Motion Planning Effectively Leveraging Diverse Motion Dataset](robotics/interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)**

:   提出 IMMP（Interaction-Merged Motion Planning），通过两阶段策略——交互保持预合并（构建多指标检查点池）和交互迁移合并（按交互模块分组的任务向量加权合并）——将来自不同轨迹数据集的智能体行为和交互知识迁移到目标域，有效提升运动规划的跨域适应性。

**[Moto Latent Motion Token As The Bridging Language For Learning Robot Manipulatio](robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)**

:   提出 Moto 框架，通过无监督学习的潜在运动 Token（Latent Motion Token）将视频帧间的视觉运动编码为离散序列，利用 GPT 式自回归预训练学习运动先验，再通过 co-fine-tuning 策略将学到的运动知识迁移到真实机器人操作，在 SIMPLER 和 CALVIN 基准上取得与 55B 参数大模型匹敌的性能（仅 98M 参数）。

**[Navmorph A Self-Evolving World Model For Vision-And-Language Navigation In Conti](robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)**

:   提出 NavMorph，一种基于 RSSM 的**自进化世界模型**，通过 World-aware Navigator 和 Foresight Action Planner 在隐空间建模连续环境动态，并引入上下文进化记忆（CEM）实现在线测试时的快速适应。

**[Pacgdc Label-Efficient Generalizable Depth Completion With Projection Ambiguity ](robotics/pacgdc_label-efficient_generalizable_depth_completion_with_projection_ambiguity_.md)**

:   提出 PacGDC，利用 2D 到 3D 投影中固有的形状歧义和位置歧义来合成大量伪几何数据（通过多个深度基础模型作为尺度操纵器），以最小的标注代价实现可泛化的深度补全，在零样本和少样本设置中均达到 SOTA。

**[Rep-Mtl Unleashing The Power Of Representation-Level Task Saliency For Multi-Tas](robotics/rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)**

:   提出 Rep-MTL，一种基于表示空间任务显著性（task saliency）的多任务优化方法，通过熵正则化保留任务特定学习模式（TSR）和样本级跨任务对比对齐（CSA）来缓解负迁移并显式促进任务互补性，无需修改优化器或网络架构。

**[Resolving Token-Space Gradient Conflicts Token Space Manipulation For Transforme](robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)**

:   提出 DTME-MTL 框架，通过在 token 空间中识别和分类梯度冲突（值域空间冲突 vs 零空间冲突），分别采用 Token Modulation（仿射变换）和 Token Expansion（添加任务特定token）来缓解 Transformer 多任务学习中的负迁移问题，以极低参数开销实现一致性能提升。

**[Selective Contrastive Learning For Weakly Supervised Affordance Grounding](robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)**

:   提出选择性对比学习方法用于弱监督可供性定位，通过原型级对比学习和像素级对比学习，在目标和部件两个粒度上自适应学习可供性相关线索，有效避免模型关注与动作无关的显著特征，在 AGD20K 和 HICO-IIF 上全面超越了使用更强基础模型（GPT-4、LLAVA 等）的竞争方法。

**[Site Towards Spatial Intelligence Thorough Evaluation](robotics/site_towards_spatial_intelligence_thorough_evaluation.md)**

:   本文提出 SITE，一个基于认知科学三重分类体系的空间智能综合基准，涵盖 8,068 个多选 VQA 任务（覆盖 31 个数据集、图像+视频），评估结果显示当前最强 VLM（GPT-4o）在整体空间推理上仍落后人类专家约 32%，且 VLM 的空间智能与机器人操控任务的成功率呈高度正相关（Pearson $r=0.902$）。

**[Transit Transient Transformer For Non-Line-Of-Sight Videography](robotics/transit_transient_transformer_for_non-line-of-sight_videography.md)**

:   设计了 TransiT 架构，通过瞬态信号压缩、帧间特征融合和时空 Transformer，实现从稀疏快速扫描（16×16、0.4ms/点）的 NLOS 瞬态信号实时重建 64×64 分辨率的隐藏场景视频（10 FPS），并提出基于 MMD 的迁移学习方法弥合合成与真实数据的分布差距。

**[Unziplora Separating Content And Style From A Single Image](robotics/unziplora_separating_content_and_style_from_a_single_image.md)**

:   提出 UnZipLoRA 方法，从单张图像中同时训练两个解耦且兼容的 LoRA（内容 LoRA 和风格 LoRA），通过 prompt 分离、列分离和块分离三种策略实现内容与风格的有效解耦，支持独立操控和自由重组，用户偏好率全面超越 DreamBooth-LoRA、Inspiration Tree 和 B-LoRA。

**[Vq-Vla Improving Vision-Language-Action Models Via Scaling Vector-Quantized Acti](robotics/vq-vla_improving_vision-language-action_models_via_scaling_vector-quantized_acti.md)**

:   本文提出基于卷积残差 VQ-VAE 的动作 tokenizer，在比先前方法多 100 倍的训练数据（含大量合成数据）上训练后可零样本迁移到各种下游 VLA 任务，在真实机器人上将长时域任务成功率提升最高 30%，推理速度提升近 3 倍。

**[Weakly-Supervised Learning Of Dense Functional Correspondences](robotics/weakly-supervised_learning_of_dense_functional_correspondences.md)**

:   定义了"稠密功能对应"（Dense Functional Correspondence）任务——基于物体功能（如"倒水"）在不同类别物体之间建立像素级稠密对应，并提出一种弱监督学习框架，通过 VLM 伪标注功能部件 + 多视角对比学习来蒸馏功能和结构知识到新模型中。

---

## 📊 LLM评测 { #llm_evaluation }

**[3Dsrbench A Comprehensive 3D Spatial Reasoning Benchmark](llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)**

:   提出首个全面的3D空间推理基准3DSRBench，包含2,772个人工标注的VQA对（12种问题类型），通过平衡数据分布和新型FlipEval策略实现鲁棒评估，揭示SOTA LMM（包括GPT-4o、Gemini）在3D空间推理上远落后于人类水平（≈52% vs 95.7%），且在非常规视角下性能显著退化。

**[A Conditional Probability Framework For Compositional Zero-Shot Learning](llm_evaluation/a_conditional_probability_framework_for_compositional_zero-shot_learning.md)**

:   本文提出条件概率框架CPF，将组合零样本学习中的组合似然分解为物体似然和条件属性似然，通过文本增强的物体学习和物体引导的属性学习模块显式建模属性-物体的语义约束和上下文依赖，在UT-Zappos50K上AUC提升17.9%，在MIT-States上Unseen Accuracy提升5.5%。

**[A Conditional Probability Framework For Compositional Zerosh](llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)**

:   提出条件概率框架（CPF），将组合识别概率分解为对象似然 p(o|x) 和属性条件似然 p(a|o,x) 两部分，通过文本增强对象学习和对象引导属性学习两个模块显式建模属性-对象依赖关系，在三个 CZSL 基准上全面超越 SOTA。

**[A Real-World Display Inverse Rendering Dataset](llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)**

:   本文构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个不同材质物体在OLAT照明模式下的立体偏振图像及高精度几何真值，并提出了一个简单有效的显示器逆渲染基线方法，超越了现有逆渲染方法。

**[A Realworld Display Inverse Rendering Dataset](llm_evaluation/a_realworld_display_inverse_rendering_dataset.md)**

:   构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个物体的OLAT（逐像素点亮）采集图像、偏振信息和GT几何，并提出简单有效的基线方法（基于Cook-Torrance BRDF的可微渲染优化），在150秒内超越现有逆渲染方法。

**[Batclip Bimodal Online Test-Time Adaptation For Clip](llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)**

:   提出BATCLIP，一种针对CLIP的双模态在线测试时自适应（TTA）方法，通过同时适应视觉编码器和文本编码器的LayerNorm参数，引入投影匹配损失和类间可分性损失来增强图文特征对齐和类别区分度，在CIFAR-10C/100C/ImageNet-C上达到SOTA效果。

**[Discopatch Taming Adversarially-Driven Batch Statistics For Improved Out-Of-Dist](llm_evaluation/discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)**

:   提出DisCoPatch框架，利用对抗性VAE中BatchNorm对批统计量的内在偏向性来区分ID和OOD样本，通过推理时将同一图像的多个patch组成batch来保证分布一致性，在协变量偏移OOD检测（ImageNet-1K(-C) 95.5% AUROC）和近分布OOD检测（95.0% AUROC）上达到SOTA，模型仅25MB且延迟低一个数量级。

**[Dista-Net Dynamic Closely-Spaced Infrared Small Target Unmixing](llm_evaluation/dista-net_dynamic_closely-spaced_infrared_small_target_unmixing.md)**

:   DISTA-Net提出动态深度展开网络，将ISTA稀疏重建中的非线性变换和阈值参数从静态改为根据输入自适应生成，实现密集红外小目标的首个深度学习解混方法，并建立了包含数据集、评估指标和工具包的首个开源生态。

**[Forcennet Foreground-Centric Network For Document Image Rectification](llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)**

:   提出以前景为中心的文档矫正网络ForCenNet，通过前景标签生成、掩码引导Transformer解码器和曲率一致性损失三大创新，仅需无畸变图像即可高效训练，在DocUNet、DIR300、WarpDoc、DocReal四个基准上达到SOTA。

**[Generative Zoo](llm_evaluation/generative_zoo.md)**

:   提出一种利用条件图像生成模型（FLUX + ControlNet）合成动物 3D 姿态和形状训练数据的可扩展流水线，生成百万级 GenZoo 数据集，仅用合成数据训练即在真实世界基准上达到 SOTA。

**[Hiero Understanding The Hierarchy Of Human Behavior Enhances Reasoning On Egocen](llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)**

:   提出 HiERO，一种弱监督的层次化图架构，通过对齐视频片段与叙述文本来学习功能性活动线索的层次结构，使视频片段特征编码多尺度的行为依赖关系，在程序学习任务的零样本评估中大幅超越全监督方法（EgoProceL 上 F1 提升 +12.5%），在视频-文本对齐基准上也达到了 SOTA。

**[Imbalance In Balance Online Concept Balancing In Generation Models](llm_evaluation/imbalance_in_balance_online_concept_balancing_in_generation_models.md)**

:   通过精心设计的因果实验揭示了数据分布（而非模型规模或数据量）是扩散模型概念组合能力的决定性因素，并提出 IMBA Loss——一种在线的、概念级别的均衡损失函数，通过条件与无条件分布差异（IMBA 距离）自适应调整 token 级损失权重，只需几行代码修改即可显著提升模型的多概念生成能力。

**[Intersyn Interleaved Learning For Dynamic Motion Synthesis In The Wild](llm_evaluation/intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)**

:   提出 InterSyn 框架，通过交错学习策略（Interleaved Learning）将单人与多人动作在统一序列中联合建模，配合相对协调精修（REC）模块，生成更自然、更协调的人体交互动作，在 InterHuman 测试集上 FID 较 FreeMotion 降低 6.1%，R Precision Top-1 提升 2.8%。

**[Odp-Bench Benchmarking Out-Of-Distribution Performance Prediction](llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)**

:   构建了首个全面的OOD性能预测基准ODP-Bench，涵盖29个OOD数据集、10种预测算法和1,444个预训练模型，揭示现有算法在合成corruption上表现较好但在自然分布偏移上普遍失效的关键发现。

**[Omnidiff A Comprehensive Benchmark For Fine-Grained Image Difference Captioning](llm_evaluation/omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)**

:   提出包含324个多样场景（真实+3D合成）的细粒度图像差异描述数据集 OmniDiff，并设计即插即用的多尺度差异感知（MDP）模块嵌入 MLLM 构建 M3Diff 模型，在 OmniDiff 及多个公开基准上取得 SOTA。

**[On The Robustness Tradeoff In Fine-Tuning](llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)**

:   首次系统研究微调过程中对抗鲁棒性与准确率的权衡关系，在231个模型、7种微调策略和6个数据集上揭示：(1)微调初期鲁棒性先升后降；(2)不同PEFT策略和任务复杂度导致不同的Pareto前沿；(3)OOD鲁棒性不存在类似权衡而是紧跟准确率变化。

**[Shadowhack Hacking Shadows Via Luminance-Color Divide And Conquer](llm_evaluation/shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)**

:   提出ShadowHack框架，将阴影去除分解为亮度恢复和颜色修复两个子任务，通过带有纠偏外展注意力的LRNet恢复亮度和纹理，再用跨注意力驱动的CRNet重建准确颜色，在ISTD+和SRD数据集上取得SOTA。

**[Spectral Sensitivity Estimation With An Uncalibrated Diffraction Grating](llm_evaluation/spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)**

:   提出一种使用未标定衍射光栅片估计相机光谱灵敏度的实用方法，通过联合估计光谱灵敏度和光栅效率，仅需一次已知光谱光源拍摄即可获得准确的闭式解，性能显著优于传统色卡方法且设备成本不到5美元。

**[Supercharging Floorplan Localization With Semantic Rays](llm_evaluation/supercharging_floorplan_localization_with_semantic_rays.md)**

:   提出一种语义感知的平面图定位框架，将语义光线预测与深度光线融合为结构-语义概率体，配合由粗到细策略，在两个标准数据集上实现了2-3倍的性能提升。

**[Svtrv2 Ctc Beats Encoder-Decoder Models In Scene Text Recognition](llm_evaluation/svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)**

:   提出 SVTRv2，通过多尺寸resize策略（MSR）、特征重排模块（FRM）和语义引导模块（SGM）三大设计，让 CTC 模型首次在多场景基准上全面超越编码器-解码器方法，同时保持推理速度优势。

---

## 🛰️ 遥感 { #remote_sensing }

**[Astroloc Robust Space To Ground Image Localizer](remote_sensing/astroloc_robust_space_to_ground_image_localizer.md)**

:   提出AstroLoc，首个利用30万张人工标注宇航员照片进行训练的太空对地定位模型，通过查询-卫星配对损失和无监督挖掘技术学习鲁棒的地球表面特征表征，在recall@1上平均提升35%，recall@100持续超过99%，已在实际中完成50万+照片的定位。

**[Citynav A Large-Scale Dataset For Real-World Aerial Navigation](remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)**

:   构建了首个面向真实城市环境的大规模空中视觉语言导航数据集 CityNav（32,637 条人类演示轨迹，覆盖 4.65 km²），并提出地理语义地图（GSM）辅助表示，显著提升基线模型的导航性能。

**[Geodistill Geometry-Guided Self-Distillation For Weakly Supervised Cross-View Lo](remote_sensing/geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)**

:   提出GeoDistill框架，通过基于视场角（FoV）遮挡的教师-学生自蒸馏范式增强局部判别性特征学习，在弱监督条件下（仅需粗略GPS标注）实现稳健的跨视角定位，性能提升超过10%且可即插即用于不同定位框架。

**[Geoexplorer Active Geo-Localization With Curiosity-Driven Exploration](remote_sensing/geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)**

:   提出 GeoExplorer，一个结合目标导向和好奇心驱动内在奖励的主动地理定位（AGL）智能体，通过联合动作-状态动力学建模和好奇心探索实现更鲁棒的 UAV 搜索策略，在未知目标和环境中展现出优越的泛化能力。

**[Information-Bottleneck Driven Binary Neural Network For Change Detection](remote_sensing/information-bottleneck_driven_binary_neural_network_for_change_detection.md)**

:   提出 BiCD，首个专为变化检测设计的二值神经网络，通过信息瓶颈（IB）原理引导的辅助目标模块提升 BNN 的特征表示能力和可分离性，在街景和遥感变化检测数据集上达到 BNN 领域的 SOTA，同时实现 30× 内存压缩和 2.5× 推理加速。

**[Pan-Crafter Learning Modality-Consistent Alignment For Pan-Sharpening](remote_sensing/pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)**

:   PAN-Crafter 提出模态一致性对齐框架，通过模态自适应重建（MARs）和跨模态对齐感知注意力（CM3A）显式处理 PAN 和 MS 图像的跨模态错位问题，在多个遥感基准数据集上达到 SOTA，且推理速度比扩散模型快 **1110×**。

**[Rs-Vheat Heat Conduction Guided Efficient Remote Sensing Foundation Model](remote_sensing/rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)**

:   首次将物理热传导过程引入遥感基础模型，提出 RS-vHeat，用热传导算子（HCO）替代注意力机制来建模遥感图像中的局部区域相关性，在 4 个任务 10 个数据集上取得优异性能的同时，相比注意力基线减少 84% 显存、24% FLOPs、提升 2.7 倍吞吐量。

**[Smarties Spectrum-Aware Multi-Sensor Auto-Encoder For Remote Sensing Images](remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)**

:   提出 SMARTIES，一个统一的传感器无关遥感基础模型，通过光谱感知投影将异构传感器数据映射到共享空间，结合跨传感器 token 混合和掩码重建进行自监督预训练，在单模态和多模态任务上超越专用传感器模型，并可泛化到预训练未见过的传感器。

**[Towards A Unified Copernicus Foundation Model For Earth Visi](remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_visi.md)**

:   提出由Copernicus-Pretrain（1870万张覆盖全部Sentinel任务的对齐图像）、Copernicus-FM（通过扩展动态超网络和Fourier元数据编码处理任意光谱/非光谱传感器的统一基础模型）、Copernicus-Bench（15个分层下游任务基准）三位一体的完整EO基础模型体系，首次实现从地表到大气的跨模态联合预训练，在15个下游任务中11个以冻结编码器超越全参数监督训练。

**[Towards A Unified Copernicus Foundation Model For Earth Vision](remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)**

:   构建了涵盖所有主要Copernicus Sentinel任务的统一地球观测基础模型体系，包括1870万对齐图像的Copernicus-Pretrain数据集、支持任意光谱/非光谱传感器的Copernicus-FM模型、以及覆盖15个层级化下游任务的Copernicus-Bench评估基准。

**[Wildsat Learning Satellite Image Representations From Wildlife Observations](remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)**

:   提出 WildSAT，利用公民科学平台上的数百万地理标记野生动物观测数据，通过对比学习将卫星图像、物种位置和文本描述对齐，显著提升遥感图像表征质量，并支持零样本文本检索。

---

## 🎵 音频/语音 { #audio_speech }

**[25 Years In Class A Multimodal Textbook For Vision-Language Pretraining](audio_speech/25_years_in_class_a_multimodal_textbook_for_vision-language_pretraining.md)**

:   从YouTube教学视频中提取关键帧和文本（ASR+OCR），构建高质量交错图文格式的"多模态教材"数据集，用于VLM预训练，在知识密集型和推理任务上大幅领先网页爬取的交错数据集。

**[25 Years In Class A Multimodal Textbook For Visionlanguage P](audio_speech/25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)**

:   从YouTube收集2.5年(22,000课时)的教学视频，通过LLM驱动的多级抽取与过滤管线构建高质量交错图文"多模态教科书"语料(6.5M关键帧 + 0.75B文本token)，显著提升VLM在知识密集型和推理任务上的预训练效果，尤其在ScienceQA和MathVista上带来大幅提升。

**[How Would It Sound Material-Controlled Multimodal Acoustic Profile Generation Fo](audio_speech/how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)**

:   提出材质可控的声学特征生成任务（M-CAPA），给定室内场景的音视觉观测和用户定义的新材质配置，生成反映材质变化的目标房间脉冲响应（RIR），并构建了配套的 Acoustic Wonderland 数据集。

**[Latent Swap Joint Diffusion For 2D Long-Form Latent Generation](audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)**

:   提出SaFa（Swap Forward），一种模态无关的高效方法，通过两种潜空间交换算子（Self-Loop Latent Swap和Reference-Guided Latent Swap）替代传统联合扩散中的均值化操作，解决频谱混叠问题并保持跨视图一致性，在长音频和全景图生成中显著优于现有方法。

**[Lyra An Efficient And Speech-Centric Framework For Omni-Cognition](audio_speech/lyra_an_efficient_and_speech-centric_framework_for_omni-cognition.md)**

:   提出 Lyra，一个以语音为中心的高效全模态 MLLM 框架，通过多模态 LoRA、潜在跨模态正则化器和潜在多模态提取器三大策略，使用更少的训练数据实现视觉-语言-语音多模态的 SOTA 性能，并首次支持长达数小时的语音输入。

**[Lyra An Efficient And Speechcentric Framework For Omnicognit](audio_speech/lyra_an_efficient_and_speechcentric_framework_for_omnicognit.md)**

:   提出Lyra，一个以语音为中心的全模态MLLM框架，通过三大核心组件（DTW-based跨模态正则化器、多模态LoRA、Latent多模态提取器）和首个12K长语音SFT数据集，在仅用2.7M数据和少量训练的情况下，同时在视觉-语言、视觉-语音、语音-语言benchmark上达到SOTA，并能处理长达2小时的语音输入。

**[Memorytalker Personalized Speech-Driven 3D Facial Animation Via Audio-Guided Sty](audio_speech/memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)**

:   提出 MemoryTalker，通过两阶段训练策略（Memorizing + Animating）利用键值记忆网络存储通用面部运动，并通过音频驱动的风格化记忆实现仅凭音频即可生成个性化 3D 面部动画，无需任何额外先验信息。

**[Mug Pseudo Labeling Augmented Audio-Visual Mamba Network For Audio-Visual Video ](audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)**

:   提出MUG框架，通过伪标签增强的跨模态随机组合数据增强策略和音视频Mamba网络，同时提升弱监督音视频解析任务中段级和事件级的预测性能。

**[Vggsounder Audio-Visual Evaluations For Foundation Models](audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)**

:   针对 VGGSound 数据集在多标签缺失、类别重叠和模态错位方面的局限性，构建了 VGGSounder——一个带有模态标注的多标签音视频分类基准，并提出"模态混淆"度量来揭示基础模型在多模态融合上的不足。

**[Zero-Avsr Zero-Shot Audio-Visual Speech Recognition With Llms By Learning Langua](audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)**

:   提出 Zero-AVSR 框架，通过将语音转写为语言无关的罗马化文本（Roman text），再利用 LLM 将罗马文本转换为目标语言文字，实现无需目标语言语音数据的零样本视听语音识别，并构建了覆盖 82 种语言、2916 小时的 MARC 数据集。

---

## 🔄 自监督 { #self_supervised }

**[A Tokenlevel Text Image Foundation Model For Document Unders](self_supervised/a_tokenlevel_text_image_foundation_model_for_document_unders.md)**

:   提出首个 token 级别文本图像基础模型 TokenFD，通过在 2000 万图像、18 亿 BPE token-mask 对上进行 token 级视觉-语言对齐预训练，实现 image-as-text 语义能力，并基于此构建文档理解 MLLM TokenVL，在 OCRBench 上得分 860（8B 组最高），在 DocVQA 等十项 VQA 任务上平均提升 8.8%。

**[Always Skip Attention](self_supervised/always_skip_attention.md)**

:   本文从理论上证明了 Vision Transformer 中的自注意力机制是本质上病态的（ill-conditioned），在无 skip connection 时会导致训练崩溃，并提出 Token Graying（TG）方法通过改善输入 token 的条件数来进一步增强 ViT 的训练稳定性和性能。

**[Cobl Toward Zero-Shot Ordinal Layering Without User Prompting](self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)**

:   本文提出 CObL，一种基于多个冻结 Stable Diffusion UNet 并行生成的架构，能在无需用户提示、不知物体数量的前提下，从单张图像推断出遮挡排序的物体层叠表示（每层一个 amodal 完整物体），并且仅用数千张合成桌面场景就能零样本泛化到真实世界照片。

**[From Linearity To Non-Linearity How Masked Autoencoders Capture Spatial Correlat](self_supervised/from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat.md)**

:   从理论角度分析 MAE 如何学习图像中的空间相关性，推导出线性 MAE 的解析解，揭示了掩码比例和 patch 大小如何选择短距离和长距离空间特征，并将分析扩展到非线性 MAE，为实践中的超参数选择提供了理论指导。

**[Loftup Learning A Coordinatebased Feature Upsampler For Visi](self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)**

:   提出LoftUp，通过坐标-cross-attention架构直接将低分辨率VFM特征映射到任意高分辨率，并用class-agnostic mask精炼+自蒸馏构建全分辨率伪GT进行训练，在6个下游任务上平均提升10-20%且在视频目标分割上提升近50%。

**[Manual-Pa Learning 3D Part Assembly From Instruction Diagrams](self_supervised/manual-pa_learning_3d_part_assembly_from_instruction_diagrams.md)**

:   提出 Manual-PA，一个基于 Transformer 的说明书引导 3D 零件组装框架：通过对比学习将 3D 零件与说明书步骤图对齐来推断组装顺序，再以学到的顺序作为位置编码的软引导进行 6DoF 位姿预测，在 PartNet 上显著超越现有方法。

**[Mosic Optimal-Transport Motion Trajectory For Dense Self-Supervised Learning](self_supervised/mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)**

:   MoSiC 利用离线点跟踪器提取长程运动轨迹，通过基于最优传输（Sinkhorn-Knopp）的聚类机制在时间维度上传播聚类分配，从而在视频数据上学习空间-时间一致的稠密表征，仅用视频训练即可将 DINOv2 在多个图像/视频基准上提升 1%–6%。

**[Scaling Languagefree Visual Representation Learning](self_supervised/scaling_languagefree_visual_representation_learning.md)**

:   通过在MetaCLIP的20亿web图像上训练DINOv2/MAE系列模型（1B-7B参数），系统性地证明纯视觉自监督学习在模型和数据规模上展现优于CLIP的scaling behavior，5B+参数时在VQA平均性能上超越CLIP——包括传统认为需要语言监督的OCR/Chart任务。

**[To Label Or Not To Label Palm - A Predictive Model For Evaluating Sample Efficie](self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)**

:   提出 PALM——一个用4个可解释参数（最大精度 $A_{\max}$、覆盖效率 $\delta$、初始学习偏移 $\alpha$、扩展性 $\beta$）描述主动学习轨迹的统一数学模型，能从有限标注数据预测完整学习曲线，实现主动学习策略的定量公平比较。

---

## 🔬 可解释性 { #interpretability }

**[Aim Amending Inherent Interpretability Via Self-Supervised Masking](interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)**

:   本文提出 AIM，一种基于自监督二值掩码的 top-down 特征选择机制，无需额外标注即可引导 CNN 聚焦真实判别特征、抑制虚假相关，同时获得内在可解释性和更强的 OOD 泛化能力。

**[Argotweak Towards Self-Updating Hd Maps Through Structured Priors](interpretability/argotweak_towards_self-updating_hd_maps_through_structured_priors.md)**

:   提出 ArgoTweak，首个提供"旧地图先验 + 当前传感器数据 + 最新真值地图"完整三元组的 HD 地图数据集，通过双射映射框架将大规模地图修改分解为元素级原子变化，并引入可解释的评测指标（mAPC/mACC），将模型在 ArgoTweak 上训练后的 sim2real 差距降低 10 倍以上。

**[Ce-Fam Concept-Based Explanation Via Fusion Of Activation Maps](interpretability/ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)**

:   提出CE-FAM概念解释方法，通过训练与图像分类器共享激活图的分支网络来模拟VLM嵌入，实现概念预测→概念区域（激活图加权和）→概念贡献（对分类分数影响）的一一对应，并提出新的NRA评估指标，在零样本概念推理上超越现有方法。

**[Granular Concept Circuits Toward A Fine-Grained Circuit Discovery For Concept Re](interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)**

:   提出 Granular Concept Circuit (GCC) 方法，通过迭代评估神经元间的功能依赖性（Neuron Sensitivity Score）和语义一致性（Semantic Flow Score），自动发现深度视觉模型中编码特定概念的细粒度视觉电路——这是首个能在单个query中发现多个概念级电路的方法。

**[Learnable Fractional Reaction-Diffusion Dynamics For Under-Display Tof Imaging A](interpretability/learnable_fractional_reaction-diffusion_dynamics_for_under-display_tof_imaging_a.md)**

:   LFRD² 提出一种混合框架，将可学习的时间分数阶反应-扩散方程与神经网络结合，用于屏下 ToF（UD-ToF）深度图恢复。通过分数阶微积分捕获迭代过程中的长期记忆依赖，并引入高效的连续卷积算子替代离散卷积，在 UD-ToF 深度恢复、ToF 去噪和深度超分辨率任务上均取得最优性能。

**[Minerva Evaluating Complex Video Reasoning](interpretability/minerva_evaluating_complex_video_reasoning.md)**

:   提出 Minerva——一个包含 1515 个手工标注的复杂视频推理问答数据集，每题配有 5 个选项和详细推理链（reasoning trace），用于评估多模态大模型的视频推理能力，并建立了视频推理错误分类体系（Temporal/Perceptual/Logical/Completeness）和 MiRA 自动评估框架。

**[Svip Semantically Contextualized Visual Patches For Zero-Shot Learning](interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)**

:   提出SVIP框架，通过在**输入阶段**识别并替换语义无关的图像patch（用属性级word embedding初始化的可学习嵌入替代），从根源上解决零样本学习中的语义错位问题。

**[Vital More Understandable Feature Visualization Through Distribution Alignment A](interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)**

:   提出VITAL方法，通过将特征可视化重新定义为真实图像特征分布对齐问题（而非传统的激活最大化），并结合相关性评分过滤无关特征，生成对人类更易理解的神经元可视化结果。

---

## 📐 优化/理论 { #optimization }

**[Addressing Representation Collapse In Vector Quantized Models With One Linear La](optimization/addressing_representation_collapse_in_vector_quantized_models_with_one_linear_la.md)**

:   提出SimVQ方法，通过一个可学习的线性变换层对码本向量进行重参数化（$\bm{C}\bm{W}$），将码本的不相交优化转化为联合空间优化，从根本上解决VQ模型中的表示崩塌问题，实现接近100%的码本利用率。

**[Adversarial Data Augmentation For Single Domain Generalization Via Lyapunov Expo](optimization/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)**

:   提出 LEAwareSGD 优化器，利用 Lyapunov 指数（LE）动态调节学习率，引导模型训练在混沌边缘附近，在对抗数据增强框架下实现更广泛的参数空间探索，显著提升单域泛化（SDG）性能。

**[Class-Wise Federated Averaging For Efficient Personalization](optimization/class-wise_federated_averaging_for_efficient_personalization.md)**

:   cwFedAvg 将 FedAvg 从"按客户端聚合"扩展为"按类别聚合"，为每个类别创建专属全局模型，再根据各客户端的类别分布加权组合成个性化模型，配合权重分布正则化（WDR）增强类别分布与权重范数的关联，在保持 FedAvg 通信开销的同时显著提升非 IID 场景下的个性化性能。

**[Cooperative Pseudo Labeling For Unsupervised Federated Classification](optimization/cooperative_pseudo_labeling_for_unsupervised_federated_classification.md)**

:   FedCoPL 首次将无监督联邦学习扩展到分类任务，通过协作伪标签策略（全局分配伪标签确保类别平衡）和部分 prompt 聚合协议（仅聚合视觉 prompt、保留文本 prompt 本地化）有效应对 CLIP 固有偏差和标签偏移挑战。

**[Federated Continual Instruction Tuning](optimization/federated_continual_instruction_tuning.md)**

:   首次提出联邦持续指令微调（FCIT）基准，涵盖 2 种场景、4 种设置和 12 个数据集，并设计 DISCO 框架通过动态知识组织（DKO）和子空间选择性激活（SSA）有效解决数据异构性和灾难性遗忘。

**[Federated Prompt-Tuning With Heterogeneous And Incomplete Multimodal Client Data](optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)**

:   提出 FED-PRIME，一个面向多模态数据模态缺失场景的联邦 Prompt-Tuning 框架，通过 inter-client 和 intra-client 两组 prompt 分别捕获跨客户端可对齐的缺失模式和客户端内特有的缺失模式，并通过聚类-对齐机制进行服务端聚合，在多种缺失数据设置下大幅超越现有基线。

**[Learning Interpretable Queries For Explainable Image Classification With Informa](optimization/learning_interpretable_queries_for_explainable_image_classification_with_informa.md)**

:   在CLIP语义嵌入空间中将信息追踪（Information Pursuit）的查询字典参数化为可学习向量，通过交替优化算法学习任务充分的可解释查询字典，缩小了可解释分类器与黑盒分类器的性能差距。

**[Memory-Efficient 4-Bit Preconditioned Stochastic Optimization](optimization/memory-efficient_4-bit_preconditioned_stochastic_optimization.md)**

:   提出基于 Cholesky 分解 + 误差反馈的 4-bit 量化方案，将 Shampoo 优化器的预条件矩阵压缩至 4-bit 精度，在大幅降低 GPU 显存的同时保持与 32-bit Shampoo 接近的训练性能，并给出了光滑与非光滑两种场景下的收敛性证明。

---

## ⚡ LLM效率 { #llm_efficiency }

**[Asynchronous Event Error-Minimizing Noise For Safeguarding Event Dataset](llm_efficiency/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)**

:   提出首个面向异步事件数据的不可学习样本生成方法（UEvs），设计了事件误差最小化噪声（E²MN）及自适应投影机制，使事件数据集在保持合法使用功能的同时阻止未授权模型从中学习。

**[Layertracer Cognitive-Aligned Layered Svg Synthesis Via Diffusion Transformer](llm_efficiency/layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)**

:   LayerTracer 提出首个基于 Diffusion Transformer（DiT）的认知对齐分层 SVG 生成框架：通过构建 2 万+ 设计师操作序列数据集，训练 DiT 生成模拟设计师工作流程的多阶段光栅化蓝图，再通过逐层矢量化和路径去重转换为干净可编辑的分层 SVG；同时支持文本驱动生成和图像到分层 SVG 的转换。

**[Mixant Observation-Dependent Memory Propagation For Stochastic Dense Action Anti](llm_efficiency/mixant_observation-dependent_memory_propagation_for_stochastic_dense_action_anti.md)**

:   提出 MixANT，通过混合专家方法为 Mamba 的遗忘门（A 矩阵）引入输入依赖性，动态选择上下文相关的 A 矩阵控制时序记忆传播，在 50Salads、Breakfast 和 Assembly101 三个密集动作预测数据集上全面超越 SOTA。

**[Phatnet A Physics-Guided Haze Transfer Network For Domain-Adaptive Real-World Im](llm_efficiency/phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)**

:   提出物理引导的雾迁移网络PHATNet，通过将大气散射模型（ASM）扩展到潜空间来解耦和迁移雾模式，生成域自适应的微调数据集，使去雾模型在测试时有效适应未见过的真实世界雾场景。

**[Rectifying Magnitude Neglect In Linear Attention](llm_efficiency/rectifying_magnitude_neglect_in_linear_attention.md)**

:   揭示 Linear Attention 完全忽略 Query 幅值信息导致注意力分数分布与 Softmax Attention 显著偏离，提出 Magnitude-Aware Linear Attention (MALA)，通过引入缩放因子 β 和偏移项 γ 使线性注意力恢复幅值感知能力，在分类、检测、分割、NLP、语音、图像生成等任务上全面超越现有方法。

**[Streammind Unlocking Full Frame Rate Streaming Video Dialogue Through Event-Gate](llm_efficiency/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)**

:   StreamMind 提出"事件门控 LLM 调用"范式替代现有的"逐帧 LLM 调用"，通过在视频编码器和 LLM 之间插入认知门控网络（Cognition Gate），仅在查询相关事件发生时才调用 LLM，配合基于状态空间方法的事件保持特征提取器（EPFE）实现常量感知成本，在单张 A100 上达到 **100 fps** 的流式视频处理速度。

**[Stroke2Sketch Harnessing Stroke Attributes For Training-Free Sketch Generation](llm_efficiency/stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)**

:   提出 Stroke2Sketch，一个无训练的参考式素描生成框架，通过跨图像笔触注意力（CSA）、指导性注意力模块（DAM）和语义保持模块（SPM）三个模块协同工作，在预训练扩散模型中实现精细的笔触属性迁移与内容结构保持。

---

## 📚 预训练 { #llm_pretraining }

**[Aceg Improving Generalization Of Scene Coordinate Regression](llm_pretraining/aceg_improving_generalization_of_scene_coordinate_regression.md)**

:   将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。

**[Conststyle Robust Domain Generalization With Unified Style Transformation](llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)**

:   提出ConstStyle框架，通过构建一个理论驱动的"统一域"（Unified Domain），在训练时将所有样本风格对齐到该统一域，测试时将未见域样本部分投影到统一域，有效缩小域间差距并提升泛化性能。

**[Dataset Ownership Verification For Pre-Trained Masked Models](llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)**

:   DOV4MM 提出了首个针对掩码预训练模型的数据集所有权验证方法，通过比较"见过"与"未见过"样本在嵌入空间中遮掩信息重构难度的差异，利用配对 t 检验判断黑盒模型是否使用了特定数据集进行预训练，在 10 种掩码图像模型和 4 种掩码语言模型上均实现 p 值远低于 0.05 的准确验证。

**[Eta Energy-Based Test-Time Adaptation For Depth Completion](llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)**

:   提出ETA方法，利用能量模型量化深度预测属于源域分布的可能性，并在测试时通过最小化目标域预测的能量值来引导预训练深度补全模型适配到新环境，在室外和室内场景平均比先前SOTA分别提升6.94%和10.23%。

**[Flow To The Mode Mode-Seeking Diffusion Autoencoders For State-Of-The-Art Image ](llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)**

:   提出 FlowMo，一种基于 Transformer 的扩散自编码器 (diffusion autoencoder)，通过两阶段训练（mode-matching 预训练 + mode-seeking 后训练），首次实现扩散自编码器在 ImageNet-1K 离散图像 tokenization 上的 SOTA 性能，无需使用卷积、对抗损失、2D 空间对齐 latent 或从其他 tokenizer 蒸馏。

**[Image Intrinsic Scale Assessment Bridging The Gap Between Quality And Resolution](llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)**

:   本文定义了图像内在尺度（IIS）这一新概念——即图像展现最高感知质量的最大缩放比例，并提出 IISA 任务、构建了 785 张图像的数据集，以及基于弱标签的 WIISA 训练策略，在多个 NR-IQA 方法上一致提升了 IIS 预测性能。

**[Make Your Training Flexible Towards Deployment-Efficient Video Models](llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)**

:   本文提出Flux——一种使视频模型训练灵活化的数据增强工具，通过灵活采样网格+组动态token选择，使单一模型在不同计算预算下都能高效工作；并提出Token Optimization新测试范式，在1/4 token下即可匹配前SOTA性能，节省约90%计算。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Aligning Information Capacity Between Vision And Language Via Dense-To-Sparse Fe](information_retrieval/aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)**

:   提出D2S-VSE框架，通过两阶段训练（稠密文本预训练+稠密到稀疏特征蒸馏微调）增强视觉语义嵌入的信息容量，解决图文匹配中图像与文本信息密度不对称的核心问题。

**[Langbridge Interpreting Image As A Combination Of Language Embeddings](information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)**

:   LangBridge 通过将视觉特征显式分解为 LLM 词汇嵌入的线性组合，实现了可解释的视觉-语言对齐，并支持跨 LLM 的预训练无关适配器迁移。

**[Monster A Unified Model For Motion Scene Text Retrieval](information_retrieval/monster_a_unified_model_for_motion_scene_text_retrieval.md)**

:   提出 **MonSTeR**——首个**运动-场景-文本三模态检索模型**，通过受拓扑深度学习启发的高阶关系建模，构建统一隐空间以捕获三模态之间的内在依赖关系，在多项检索任务上大幅超越仅依赖单模态表征的基线，并可用于人-场景交互模型的评估。

**[Ocr Hinders Rag Evaluating The Cascading Impact Of Ocr On Retrieval-Augmented Ge](information_retrieval/ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)**

:   提出 OHRBench——首个评估 OCR 对 RAG 系统级联影响的基准，包含 7 个领域的 8561 张文档图像和 8498 个 QA 对，系统性地揭示了 OCR 产生的语义噪声（Semantic Noise）和格式噪声（Formatting Noise）对检索和生成两阶段的不同影响模式。

**[Representation Shift Unifying Token Compression With Flashattention](information_retrieval/representation_shift_unifying_token_compression_with_flashattention.md)**

:   提出 Representation Shift，一种无需训练、模型无关的 token 重要性度量方法，通过计算 token 在网络层前后的表征变化量来衡量重要性，从而首次实现 token 压缩与 FlashAttention 的兼容，在视频理解和图像分类上取得高达 5.5× 的加速。

**[Vilu Learning Vision-Language Uncertainties For Failure Prediction](information_retrieval/vilu_learning_vision-language_uncertainties_for_failure_prediction.md)**

:   提出 ViLU，一个针对 VLM 零样本预测的后验不确定性量化框架，通过交叉注意力融合视觉嵌入、预测文本嵌入和图像条件文本表示，构建不确定性感知的多模态表征，在 13 个分类数据集和大规模图文数据集上显著超越现有失败预测方法。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Embodied Navigation With Auxiliary Task Of Action Description Prediction](reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)**

:   DescRL 将动作描述生成作为强化学习导航的辅助任务，通过从预训练的视觉-语言模型蒸馏知识来训练 ADPredictor，使导航智能体在生成可解释动作描述的同时提升导航性能，在语义音频-视觉导航（SAVNav）等多个任务上实现 SOTA。

**[Mdp3 A Training-Free Approach For List-Wise Frame Selection In Video-Llms](reinforcement_learning/mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)**

:   提出 mDP3，一种免训练、模型无关的视频帧选择方法，通过条件高斯核在 RKHS 中估计帧相似度，结合行列式点过程（DPP）捕获查询相关性和列表级多样性，再通过马尔可夫决策过程（MDP）建模时序性，在多个长视频 benchmark 上以仅 8 帧输入显著超越均匀采样和现有帧选择方法。

**[Navq Learning A Q-Model For Foresighted Vision-And-Language Navigation](reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)**

:   提出 NavQ，一种前瞻性 VLN 智能体，通过 Q-model 在单次前向传播中预测每个候选动作的长期未来语义聚合特征（Q-feature），结合 A* 式搜索策略在目标导向导航中取得显著提升。

**[Progressor A Perceptually Guided Reward Estimator With Self-Supervised Online Re](reinforcement_learning/progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)**

:   提出Progressor框架，从无标注视频中自监督学习任务无关的奖励函数，通过预测任务进度分布提供稠密奖励信号，并在在线RL训练中通过对抗性push-back策略应对分布偏移问题。

**[R1-Onevision Advancing Generalized Multimodal Reasoning Through Cross-Modal Form](reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)**

:   提出 R1-Onevision，通过跨模态推理管线将图像转换为形式化文本表示，结合 SFT + 基于规则的强化学习（GRPO）的两阶段后训练策略，显著提升视觉语言模型的多模态推理能力，在多个数学推理基准上超越 GPT-4o。

**[Reinforcement Learning-Guided Data Selection Via Redundancy Assessment](reinforcement_learning/reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)**

:   提出 RL-Selector，引入 ε-sample cover 概念量化样本冗余度，将数据选择建模为强化学习过程，通过轻量 A2C 策略网络自适应优化选择策略，在多个基准数据集上以更少数据达到接近甚至超越全量训练的泛化性能。

---

## 🦾 LLM Agent { #llm_agent }

**[Embodied Image Captioning Self-Supervised Learning Agents For Spatially Coherent](llm_agent/embodied_image_captioning_self-supervised_learning_agents_for_spatially_coherent.md)**

:   提出一个三阶段自监督框架，通过agent自主导航收集多视角观测、LLM共识机制生成伪标注、对比学习微调captioner，显著提升室内环境中同一物体跨视角描述的一致性和准确性。

**[Gtr Guided Thought Reinforcement Prevents Thought Collapse I](llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)**

:   发现RL训练VLM Agent时的"思维坍塌"现象——CoT推理迅速退化为与状态无关的模板化思维并导致无效动作，提出GTR框架用VLM纠正器自动修正思维(SFT) + PPO优化动作的双目标训练，在24点游戏和ALFWorld上实现3-5倍的成功率提升。

**[Less Is More Empowering Gui Agent With Context-Aware Simplification](llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)**

:   提出 SimpAgent——一种上下文感知的简化框架，通过基于遮挡的元素剪枝（训练时随机遮挡无关元素区域）和一致性引导的历史压缩（在 LLM 中间层直接丢弃历史视觉 token + KL散度一致性约束），在降低27% FLOPs 的同时取得多个 GUI 导航基准的 SOTA。

**[Uipro Unleashing Superior Interaction Capability For Gui Agents](llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)**

:   提出 UIPro，通过构建 2060 万 GUI 理解样本进行预训练并提出统一动作空间整合异构 GUI agent 任务数据，实现跨移动端、Web 端和桌面端的 SOTA GUI 交互性能。

---

## 💡 LLM推理 { #llm_reasoning }

**[Corvid Improving Multimodal Large Language Models Towards Ch](llm_reasoning/corvid_improving_multimodal_large_language_models_towards_ch.md)**

:   提出Corvid MLLM，通过混合视觉编码器(SigLIP+ConvNeXt)+GateMixer门控连接器+MCoT-Instruct-287K高质量CoT数据+推理时自验证策略，三阶段训练实现超越同规模o1-like模型的CoT推理能力。

**[Corvid Improving Multimodal Large Language Models Towards Chain-Of-Thought Reaso](llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)**

:   提出 Corvid，通过混合视觉编码器 + GateMixer 连接器 + 高质量 CoT 数据集 + 推理时自验证策略，全面提升 MLLM 的链式推理能力，在数学推理和科学问题求解上超越同参数量级的开源模型。

**[Unsupervised Visual Chain-Of-Thought Reasoning Via Preference Optimization](llm_reasoning/unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization.md)**

:   提出UV-CoT框架，通过自动生成偏好数据和改进的Score-DPO损失函数，在不需要人工标注bounding box的情况下实现图像级链式思维（Visual CoT）推理，在6个基准上超越有监督的Visual-CoT方法。

**[Video-T1 Test-Time Scaling For Video Generation](llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)**

:   将LLM中的测试时缩放(TTS)思想迁移到视频生成领域，将TTS重新定义为从高斯噪声空间到目标视频分布的搜索问题，提出Tree-of-Frames (ToF)搜索算法实现高效的推理时计算扩展，在VBench上持续稳定提升各类视频生成模型的质量。

---

## 🔒 LLM安全 { #llm_safety }

**[Adversarial Robust Memory-Based Continual Learner](llm_safety/adversarial_robust_memory-based_continual_learner.md)**

:   揭示持续学习与对抗训练结合时的双重挑战（加速遗忘 + 梯度混淆），提出抗遗忘 Logit 校准（AFLC）和鲁棒感知经验回放（RAER）两个即插即用模块，在 Split-CIFAR10/100 和 Split-Tiny-ImageNet 上有效提升对抗鲁棒性达 8.13%。

**[Chartcap Mitigating Hallucination Of Dense Chart Captioning](llm_safety/chartcap_mitigating_hallucination_of_dense_chart_captioning.md)**

:   构建了包含56.5万张真实图表-描述对的大规模数据集ChartCap，通过类型特定的描述模式排除无关信息、强调结构与关键洞察，并提出无参考的Visual Consistency Score评估指标，有效减少VLM在图表描述中的幻觉问题。

**[Forgetting Through Transforming Enabling Federated Unlearning Via Class-Aware Re](llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)**

:   提出 FUCRT 方法，通过类感知表征变换实现联邦遗忘：将遗忘类的表征“变换”到语义最近的保留类，而非直接消除，配合双重对比学习对齐跨客户端的变换一致性，在四个数据集上实现 100% 遗忘保障的同时保持甚至提升剩余类性能。

**[Temporal Unlearnable Examples Preventing Personal Video Data From Unauthorized E](llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)**

:   本文首次研究防止视频数据被深度跟踪器未授权使用的问题，提出基于 DiT 的生成式框架生成时序不可学习样本（TUE），通过时间对比损失使跟踪器依赖扰动噪声进行时序匹配而非学习真实数据结构，实现了跨模型、跨数据集和跨任务的强可迁移性。

---

## 📈 时间序列 { #time_series }

**[I2-World Intra-Inter Tokenization For Efficient Dynamic 4D Scene Forecasting](time_series/i2-world_intra-inter_tokenization_for_efficient_dynamic_4d_scene_forecasting.md)**

:   提出 I²-World，通过将 3D 场景 tokenization 解耦为帧内（intra-scene）多尺度残差量化和帧间（inter-scene）时序量化两个互补过程，在保持 3D tokenizer 高压缩率的同时获得 4D tokenizer 的时序建模能力，实现高效且高质量的 4D occupancy 预测。

**[V2Xpnp Vehicle-To-Everything Spatio-Temporal Fusion For Multi-Agent Perception A](time_series/v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)**

:   提出 V2XPnP，一个基于统一 Transformer 架构的 V2X 时空融合框架，在单步通信策略下实现多智能体端到端感知与预测，同时构建了首个支持所有 V2X 协作模式的大规模真实世界时序数据集，在感知和预测任务上达到 SOTA。

**[Va-Moe Variables-Adaptive Mixture Of Experts For Incremental Weather Forecasting](time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)**

:   提出增量天气预报新范式和VA-MoE框架，通过变量自适应的MoE架构和索引嵌入机制，实现在仅25%可训练参数和50%初始训练数据的条件下达到与全量训练可比的预报精度。

**[Vlrmbench A Comprehensive And Challenging Benchmark For Vision-Language Reward M](time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)**

:   提出 VLRMBench，一个包含 12634 个问题、12 项任务的综合且具有挑战性的视觉语言奖励模型（VLRM）基准，覆盖过程理解、结果判断和批评生成三大方面，在 26 个模型上的广泛实验揭示了当前 VLRM 的显著不足。

---

## 🕸️ 图学习 { #graph_learning }

**[Pasta Part-Aware Sketch-To-3D Shape Generation With Text-Aligned Prior](graph_learning/pasta_part-aware_sketch-to-3d_shape_generation_with_text-aligned_prior.md)**

:   提出PASTA框架，通过VLM文本先验补充草图缺失的语义信息，并用ISG-Net（IndivGCN+PartGCN）建模部件间关系，实现了草图到3D形状生成的SOTA性能，支持部件级编辑。

**[Taming The Untamed Graph-Based Knowledge Retrieval And Reasoning For Mllms To Co](graph_learning/taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co.md)**

:   以《怪物猎人：世界》为测试平台，构建了包含文本、图像、视频和复杂实体关系的多模态知识图谱(MH-MMKG)，设计了238个复杂查询和多智能体知识检索方法，揭示了当前MLLM在领域特定任务中的知识检索与推理能力不足。

---

## 💬 LLM/NLP { #llm_nlp }

**[Any-Ssr How Recursive Least Squares Works In Continual Learning Of Large Languag](llm_nlp/any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag.md)**

:   提出Analytic Subspace Routing（Any-SSR）框架，通过为每个任务分配独立的LoRA子空间消除任务间干扰，并利用递归最小二乘（RLS）闭式解训练一个零遗忘的解析路由器，实现LLM的无回放持续学习。

**[Zeroth-Order Fine-Tuning Of Llms In Random Subspaces](llm_nlp/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)**

:   提出 SubZero（random Subspace Zeroth-order），通过逐层低秩扰动在随机子空间中估计梯度，显著降低零阶优化的梯度方差和角度误差，以接近推理的内存开销实现 LLM 的高效微调。

---

## 📡 信号/通信 { #signal_comm }

**[Boosting Multimodal Learning Via Disentangled Gradient Learning](signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)**

:   本文揭示了多模态学习中模态编码器和融合模块之间的优化冲突——融合模块会抑制回传到各模态编码器的梯度，导致即使是优势模态也比单模态模型表现差，并提出解耦梯度学习（DGL）框架通过截断融合模块到编码器的梯度并用独立的单模态损失替代来解决此问题。

**[Generalizable Non-Line-Of-Sight Imaging With Learnable Physical Priors](signal_comm/generalizable_non-line-of-sight_imaging_with_learnable_physical_priors.md)**

:   提出Learnable Path Compensation (LPC)和Adaptive Phasor Field (APF)两个模块，分别解决NLOS成像中辐射强度衰减的材质依赖性问题和不同信噪比条件下的频域去噪问题，仅在合成数据上训练即可在多种真实数据集上实现SOTA泛化性能。

---

## 👥 社会计算 { #social_computing }

**[Gradient Extrapolation For Debiased Representation Learning](social_computing/gradient_extrapolation_for_debiased_representation_learning.md)**

:   提出 GERNE 方法，通过构建具有不同虚假相关程度的两个 batch 并对其梯度进行线性外推，引导模型学习去偏差表征，在已知和未知属性情况下均优于 SOTA。

**[Propvg End-To-End Proposal-Driven Visual Grounding With Multi-Granularity Discri](social_computing/propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)**

:   提出PropVG，首个无需预训练检测器的端到端proposal-based视觉定位框架，将视觉定位分解为前景proposal生成+基于对比学习的指代评分两阶段，并引入多粒度目标判别模块（MTD）融合物体级和语义级信息判断目标是否存在，在10个数据集上刷新SOTA且推理速度比传统proposal方法快4倍。

---

## 🔎 AIGC检测 { #aigc_detection }

**[Spatial Preference Rewarding For Mllms Spatial Understanding](aigc_detection/spatial_preference_rewarding_for_mllms_spatial_understanding.md)**

:   提出 SPR（Spatial Preference Rewarding）框架，通过语义分数和定位分数自动构建偏好数据对，利用 DPO 训练 MLLM 区分高精度定位（正样本）和模糊/错误定位（负样本），大幅提升细粒度空间理解能力，尤其在高 IoU 阈值下效果显著。

---

## 🔗 因果推理 { #causal_inference }

**[A Visual Leap In Clip Compositionality Reasoning Through Gen](causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)**

:   提出基于LLM+扩散模型的block-based diffusion方法自动生成高质量反事实图文对数据集，配套设计set-aware损失函数，无需人工标注即可显著提升CLIP的组合推理能力，在ARO/VL-Checklist等benchmark上以更少数据超越SOTA。

---

## 💻 代码智能 { #code_intelligence }

**[Tikzero Zero-Shot Text-Guided Graphics Program Synthesis](code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)**

:   提出 TikZero，通过将图像表示作为中间桥梁，将图形程序生成与文本理解解耦，实现零样本文本引导的 TikZ 图形程序合成，在无需文本对齐训练数据的情况下大幅超越基线方法，经端到端微调后的 TikZero+ 达到甚至超越 GPT-4o 等大型商业模型的性能。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[Magicid Hybrid Preference Optimization For Id-Consistent And Dynamic-Preserved V](llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)**

:   提出 MagicID 框架，通过构建身份偏好和动态偏好的混合视频对数据，并设计两阶段混合偏好优化（HPO）训练策略，首次将 DPO 应用于身份定制化视频生成，同时解决传统自重建训练导致的身份退化和动态减弱问题。

---

## ✍️ 文本生成 { #nlp_generation }

**[Beyond Isolated Words Diffusion Brush For Handwritten Text-Line Generation](nlp_generation/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)**

:   提出 DiffBrush，首个基于扩散模型的手写文本行生成方法，通过内容解耦的风格学习（列/行掩码）和多尺度内容判别器（行/词级别），在风格模仿和内容准确性上大幅超越现有方法。

---

## 📖 NLP理解 { #nlp_understanding }

**[Balancing Task-Invariant Interaction And Task-Specific Adaptation For Unified Im](nlp_understanding/balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)**

:   TITA 提出了一种无需任务标识的统一图像融合框架，通过交互增强像素注意力（IPA）模块探索任务不变的互补信息提取，并通过基于操作的自适应融合（OAF）模块动态适配任务特定需求，同时采用 FAMO 策略缓解多任务梯度冲突。

---

## ⚛️ 物理学 { #physics }

**[Resq A Novel Framework To Implement Residual Neural Networks On Analog Rydberg A](physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)**

:   提出 ResQ——首个利用模拟 Rydberg 原子量子计算机的连续时间哈密顿演化来原生实现残差神经网络（ResNet）的框架，通过分段参数化激光脉冲编码输入特征和训练参数，在 MNIST/FashionMNIST/医疗数据集的分类任务上相比同等规模经典模型平均提升50%。

---

## 🧮 科学计算 { #scientific_computing }

**[Jpeg Processing Neural Operator For Backward-Compatible Coding](scientific_computing/jpeg_processing_neural_operator_for_backward-compatible_coding.md)**

:   提出JPNeO，一个完全后向兼容JPEG格式的下一代编解码器，通过在编码和解码阶段分别引入神经算子(JENO和JDNO)以及可训练量化矩阵，显著提升JPEG重建质量（尤其是色度分量），同时保持低内存和少参数量的优势。

---

## 📂 其他 { #others }

**[A Hidden Stumbling Block In Generalized Category Discovery D](others/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)**

:   发现GCD中未标注数据（尤其是未知类别）的ViT注意力会分散到背景区域（distracted attention），提出Attention Focusing（AF）模块通过多尺度token重要性度量+自适应剪枝来纠正注意力，作为即插即用模块在SimGCD上最高带来15.4%的性能提升。

**[A Hyperdimensional One Place Signature To Represent Them All Stackable Descripto](others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)**

:   本文提出 HOPS（Hyperdimensional One Place Signatures），利用超维计算（HDC）框架将同一地点在不同环境条件下采集的多个参考描述子融合为统一表示，在不增加计算量和存储开销的前提下，大幅提升视觉场所识别（VPR）的鲁棒性与召回率。

**[A Linear N-Point Solver For Structure And Motion From Asynchronous Tracks](others/a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)**

:   本文提出了一种统一的线性 N-point 求解器，能够从具有任意时间戳的 2D 点对应中恢复相机线速度和 3D 点结构，适用于全局快门、滚动快门和事件相机等多种传感器模式。

**[Adaptiveae An Adaptive Exposure Strategy For Hdr Capturing I](others/adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)**

:   本文提出AdaptiveAE，利用深度强化学习将HDR曝光包围拍摄建模为马尔可夫决策过程（MDP），同时优化ISO和快门速度的组合，在用户定义的时间预算内自适应地为动态场景选择最优曝光参数，在HDRV数据集上达到PSNR 39.70，比之前最好的方法Hasinoff et al. (37.59) 高出2.1 dB。

**[Afunet Crossiterative Alignmentfusion Synergy For Hdr Recons](others/afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)**

:   将多曝光HDR重建从MAP估计视角建模，通过空间对应先验将问题分解为对齐和融合两个交替子问题，再展开为端到端可训练的AFUNet（含SAM空间对齐+CFM通道融合+DCM数据一致性模块），在三个HDR基准上取得SOTA，PSNR-μ达44.91dB（Kalantari数据集）。

**[Autoregressively Generating Multiview Consistent Images](others/autoregressively_generating_multiview_consistent_images.md)**

:   首次将自回归（AR）模型引入多视角图像生成任务，通过逐视角生成利用所有前序视角信息来增强远距离视角间的一致性，同时设计了统一的多模态条件注入架构和Shuffle Views数据增强策略，使单一模型可同时处理文本/图像/几何形状条件。

**[C4D 4D Made From 3D Through Dual Correspondences](others/c4d_4d_made_from_3d_through_dual_correspondences.md)**

:   提出C4D框架，通过在DUSt3R的3D pointmap预测基础上联合捕获双重时序对应(短时光流+动态感知长时点跟踪DynPT)，生成运动掩码分离动静区域，并引入相机运动对齐/相机轨迹平滑/点轨迹平滑三个优化目标，将现有3D重建范式升级为完整4D重建(逐帧点云+相机参数+2D/3D轨迹)，在深度/位姿/跟踪多个下游任务上达competitive性能。

**[Despite Exploring Contrastive Deep Skeletonpointcloudimutext](others/despite_exploring_contrastive_deep_skeletonpointcloudimutext.md)**

:   提出 DeSPITE，一个将 LiDAR 点云、骨架姿态、IMU 信号和文本四种模态对齐到联合嵌入空间的对比学习框架，首次以 LiDAR（而非 RGB）作为核心视觉模态，实现了跨模态匹配/检索等此前不可能的任务，同时作为有效的 HAR 预训练策略在 MSR-Action3D 和 HMPEAR 上取得 SOTA。

**[Doodle Your Keypoints Sketch-Based Few-Shot Keypoint Detection](others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)**

:   提出首个基于草图的跨模态少样本关键点检测框架，利用原型网络、网格定位器、原型域适应和去风格化网络，仅需少量带标注草图即可在真实照片中检测新类别的新关键点。

**[Edffdnet Towards Accurate And Efficient Unsupervised Multi-Grid Image Registrati](others/edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)**

:   提出 EDFFDNet，采用指数衰减自由形变 (EDFFD) 替代传统 B-spline FFD 和 TPS 进行图像配准，配合自适应稀疏运动聚合器 (ASMA) 和渐进式相关策略，在 UDIS-D 数据集上以减少 70.5% 参数、32.6% 显存的代价实现 +0.5dB PSNR 提升。

**[Failure Cases Are Better Learned But Boundary Says Sorry Facilitating Smooth Per](others/failure_cases_are_better_learned_but_boundary_says_sorry_facilitating_smooth_per.md)**

:   揭示了对抗训练中一个反直觉现象——失败样本的模型感知变化反而比成功样本更小（即被"过度学习"），据此提出 Robust Perception Adversarial Training (RPAT)，通过鼓励感知随扰动平滑变化来缓解准确率-鲁棒性权衡问题。

**[Fixtalk Taming Identity Leakage For High-Quality Talking Head Generation In Extr](others/fixtalk_taming_identity_leakage_for_high-quality_talking_head_generation_in_extr.md)**

:   提出FixTalk框架，通过增强运动指示器（EMI）和增强细节指示器（EDI）两个轻量级即插即用模块，将GAN模型中的身份泄漏问题"化害为利"——EMI消除运动特征中的身份信息以解决身份泄漏，EDI利用泄漏的身份信息在极端姿态下补充缺失细节以消除渲染伪影。

**[From Easy To Hard Progressive Active Learning Framework For Infrared Small Targe](others/from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)**

:   提出渐进式主动学习（PAL）框架，通过"模型预启动→模型增强→模型精炼"三阶段训练策略，驱动红外小目标检测网络从易到难地主动识别和学习困难样本，在单点监督条件下显著缩小了与全监督方法之间的性能差距（IoU 提升 8.53%–29.1%）。

**[Generate Refine And Encode Leveraging Synthesized Novel Samples For On-The-Fly F](others/generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)**

:   提出基于扩散模型的即时类别发现框架 DiffGRE，通过属性组合生成（ACG）合成包含虚拟类别信息的新样本、多样性驱动精炼（DDR）过滤低质量样本、半监督Leader编码（SLE）注入额外类别知识，在 6 个细粒度数据集上显著提升了已有 OCD 方法的性能（平均 ACC-ALL 提升 6.5%）。

**[Hytip Hybrid Temporal Information Propagation For Masked Conditional Residual Vi](others/hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)**

:   提出 HyTIP 框架，将输出回归（显式缓冲解码帧）和隐状态传播（隐式缓冲潜在特征）两种时序信息传播机制统一到同一学习式视频编码框架中，仅用 SOTA 方法 14% 的缓冲区大小即可达到可比的编码性能。

**[I Am Big You Are Little I Am Right You Are Wrong](others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)**

:   利用因果推理 XAI 工具 rex 提取图像分类模型的最小充分像素集（MPS），系统比较 5 种架构 15 个模型的"注意力集中度"，发现大型模型（EVA/ConvNext）仅用图像 5% 像素即可做出分类，且不同架构的 MPS 在大小和位置上存在统计显著差异。

**[Iap Invisible Adversarial Patch Attack Through Perceptibility-Aware Localization](others/iap_invisible_adversarial_patch_attack_through_perceptibility-aware_localization.md)**

:   提出 IAP 框架，通过**感知感知（perceptibility-aware）的贴片定位**和**保色梯度更新**，首次实现在目标攻击场景下生成真正不可见的对抗补丁，同时能绕过多种 SOTA 补丁防御方法。

**[Is Meta-Learning Out Rethinking Unsupervised Few-Shot Classification With Limite](others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)**

:   本文通过提出"熵受限监督设定"建立了元学习与全类训练（WCT）的公平比较框架，从理论上证明了元学习有更紧的泛化界，并揭示了其对标签噪声更鲁棒、更适合异构任务的特性，据此提出 MINO 框架在无监督少样本和零样本任务上取得了 SOTA。

**[Jigsaw Imagining Complete Shape Priors For Object Reassembly](others/jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)**

:   Jigsaw++ 提出了一种基于生成模型的完整形状先验学习方法，通过"retargeting"策略将部分组装的碎片点云映射到完整物体的形状空间，与现有组装算法正交地提升重组质量。

**[Joint Asymmetric Loss For Learning With Noisy Labels](others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)**

:   将非对称损失函数扩展到更复杂的被动损失场景，提出非对称均方误差（AMSE），严格建立其满足非对称条件的充要条件，并将 AMSE 嵌入 APL 框架构建联合非对称损失（JAL），在 CIFAR-10/100 等多个数据集上全面超越现有鲁棒损失函数方法。

**[Kaputt A Large-Scale Dataset For Visual Defect Detection](others/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)**

:   Kaputt 发布了一个包含 23 万+ 图像、4.8 万+ 独立商品的大规模零售物流缺陷检测数据集，规模是 MVTec-AD 的 40 倍，首次引入显著的姿态和外观变化，使得 SOTA 异常检测方法的 AUROC 不超过 56.96%，揭示了现有方法在真实零售场景中的严重不足。

**[Lacoot Layer Collapse Through Optimal Transport](others/lacoot_layer_collapse_through_optimal_transport.md)**

:   提出 LaCoOT，一种基于最优传输的正则化策略，通过最小化网络内部中间特征分布之间的 Max-Sliced Wasserstein 距离，使得训练后可以直接移除整个网络层，在保持性能的同时显著减少模型深度和推理时间。

**[Layerd Decomposing Raster Graphic Designs Into Layers](others/layerd_decomposing_raster_graphic_designs_into_layers.md)**

:   提出 LayerD，通过迭代提取未遮挡顶层和背景补全来分解栅格图形设计为可编辑图层，并利用图形设计的域先验（纹理平坦区域）进行精炼，同时提出了基于 DTW 的层级评估协议。

**[Learning Visual Hierarchies In Hyperbolic Space For Image Retrieval](others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)**

:   首次提出在双曲空间中编码用户定义的多层视觉层次结构的学习范式，通过基于角度的 entailment 对比损失在无需显式层次标签的情况下学习 scene→object→part 层次，并引入基于最优传输的层次检索评估指标。

**[Loss Functions For Predictor-Based Neural Architecture Search](others/loss_functions_for_predictor-based_neural_architecture_search.md)**

:   首次对性能预测器中8种损失函数进行全面系统性研究，涵盖回归、排序和加权三大类，在5个搜索空间的13个任务上揭示了各类损失函数的特性与互补性，并提出分段损失（PW loss）组合方法PWLNAS，在多个基准上超越现有SOTA。

**[Magic Insert Style-Aware Drag-And-Drop](others/magic_insert_style-aware_drag-and-drop.md)**

:   提出Magic Insert方法，首次形式化和解决"风格感知拖放"问题——将任意风格的主体拖入不同风格的目标图像中，主体自动适应目标风格且插入效果物理合理，核心包括风格感知个性化（LoRA+IP-Adapter风格注入）和Bootstrap Domain Adaptation（将真实图像训练的插入模型适配到风格化图像领域）。

**[Membership Inference Attacks With False Discovery Rate Control](others/membership_inference_attacks_with_false_discovery_rate_control.md)**

:   提出MIAFdR，首个能提供错误发现率（FDR）理论保证的成员推理攻击方法，通过设计新颖的非成员一致性分数函数和基于调整的成员判定策略来控制FDR，可作为即插即用的wrapper无缝集成到现有MIA方法中，在保持攻击性能的同时提供FDR控制。

**[Nappure Adversarial Purification For Robust Image Classification Under Non-Addit](others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)**

:   提出 NAPPure 框架，通过联合优化底层干净图像和扰动参数（基于似然最大化），将对抗纯化从仅处理加性扰动扩展到模糊、遮挡、几何扭曲等非加性扰动，在GTSRB上实现73.93%的平均鲁棒准确率（传统方法仅43.2%）。

**[Omni-Dc Highly Robust Depth Completion With Multiresolution Depth Integration](others/omni-dc_highly_robust_depth_completion_with_multiresolution_depth_integration.md)**

:   提出 OMNI-DC，通过多分辨率深度积分器（Multi-res DDI）、Laplacian 损失和尺度归一化技术，构建了一个能够零样本泛化到不同数据集和稀疏深度模式的高鲁棒深度补全模型。

**[On The Complexity-Faithfulness Trade-Off Of Gradient-Based Explanations](others/on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)**

:   提出统一的频谱框架来系统性分析和量化梯度解释的平滑性（复杂度）与忠实度之间的权衡，引入期望频率（EF）度量网络对高频信息的依赖程度，并通过将 ReLU 与高斯函数卷积来控制解释复杂度，同时定义"解释间隙"来量化替代模型导致的忠实度损失。

**[Ph-Gan Physics-Inspired Gan For Generating Sar Images Under Limited Data](others/ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)**

:   提出Φ-GAN，将SAR的理想点散射中心（PSC）电磁散射物理模型以可微神经模块形式集成到GAN训练中，通过双物理损失（生成器物理一致性约束+判别器电磁特征蒸馏）显著提升数据稀缺场景下SAR图像生成的质量和稳定性。

**[Recover Biological Structure From Sparse-View Diffraction Images With Neural Vol](others/recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)**

:   提出Neural Volumetric Prior (NVP)，通过融合显式3D特征网格与隐式MLP的混合神经表示，结合基于衍射光学的物理渲染方程，首次实现了从稀疏视角（仅6-7张荧光图像）对半透明生物样本3D折射率的高保真体积重建，所需图像数量减少约50倍、处理时间缩短3倍。

**[Recovering Parametric Scenes From Very Few Time-Of-Flight Pixels](others/recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)**

:   本文探索用极少量（低至 15 个像素）低成本广视场 ToF 传感器恢复 3D 参数化场景几何的可行性，设计了前馈预测+可微渲染的分析-合成框架，在 6D 物体位姿估计等任务上展示了令人惊讶的效果。

**[Thermal Polarimetric Multi-View Stereo](others/thermal_polarimetric_multi-view_stereo.md)**

:   提出利用热偏振（长波红外偏振）线索进行精细三维形状重建的方法，理论证明 LWIR 偏振观测不受光照环境和材质光学属性的影响，从而实现对透明、半透明和异质材料物体的高精度三维重建，显著优于可见光偏振方法。

**[Toward Material-Agnostic System Identification From Videos](others/toward_material-agnostic_system_identification_from_videos.md)**

:   提出 MASIV，首个无需预定义材质先验的视觉系统辨识框架：采用可学习的神经本构模型替代手工设计的弹性/塑性方程，通过重建连续体粒子轨迹提供时间密集的几何约束，从多视角视频中推断物体的内在动力学特性。

**[You Share Beliefs I Adapt Progressive Heterogeneous Collaborative Perception](others/you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)**

:   提出PHCP框架，首次在推理阶段解决异构协同感知的域差距问题——通过agent的伪标签做few-shot无监督域适应，自训练适配器对齐特征空间，无需联合训练即在OPV2V上仅用少量无标注数据达到接近SOTA(HEAL)的性能。

</div>