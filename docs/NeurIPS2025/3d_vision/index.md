---
title: >-
  NeurIPS2025 3D 视觉方向116篇论文解读
description: >-
  116篇NeurIPS2025的 3D 视觉方向论文解读，涵盖 3D 高斯渲染、扩散模型、动态场景、新视角合成、三维重建、点云等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D 视觉

**🧠 NeurIPS2025** · **116** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (230)](../../CVPR2026/3d_vision/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/3d_vision/index.md) · [🤖 AAAI2026 (74)](../../AAAI2026/3d_vision/index.md) · [📹 ICCV2025 (254)](../../ICCV2025/3d_vision/index.md) · [🧪 ICML2025 (11)](../../ICML2025/3d_vision/index.md) · [💬 ACL2025 (1)](../../ACL2025/3d_vision/index.md)

🔥 **高频主题：** 3D 高斯渲染 ×19 · 扩散模型 ×10 · 动态场景 ×9 · 新视角合成 ×7 · 三维重建 ×6

**[3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)**

:   提出 Tri-MARF 三模态多智能体框架，通过 VLM 标注 Agent（多视角多候选描述）+ 信息聚合 Agent（BERT 聚类 + CLIP 加权 + UCB1 多臂赌博机选择）+ 点云门控 Agent（Uni3D 文本-点云对齐过滤幻觉），实现 CLIPScore 88.7（超越人类标注 82.4）、吞吐量 12k 物体/小时，已标注约 200 万 3D 模型。

**[3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)**

:   揭示了3D视觉错觉（如墙面彩绘、屏幕重播、镜面反射等）会严重欺骗现有SOTA单目和双目深度估计方法，构建了包含约3k场景/200k图像的大规模数据集，并提出基于VLM常识推理的单目-双目自适应融合框架，在各类错觉场景下达到SOTA。

**[Anti-Aliased 2D Gaussian Splatting](anti-aliased_2d_gaussian_splatting.md)**

:   提出 AA-2DGS，通过世界空间平坦平滑核和物体空间 Mip 滤波器两个互补机制，解决 2D Gaussian Splatting 在不同采样率下渲染时的严重锯齿问题，在保持 2DGS 几何精度优势的同时显著提升多尺度渲染质量。

**[ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)**

:   提出将 3D mesh 生成建模为"由粗到精"的逐级细化过程（next-level-of-detail prediction），通过反转广义网格简化算法（GSlim）获得渐进式细化序列，再用 Transformer 自回归学习，从单个点开始逐步增加几何与拓扑细节生成完整网格。

**[AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)**

:   提出 AtlasGS，通过将 Atlanta-world 结构先验引入隐式结构化高斯表示（implicit-structured Gaussians），在室内和城市场景中实现平滑且保留高频细节的高质量表面重建，全面超越已有隐式和显式方法。

**[BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)**

:   提出 BecomingLit，基于 3D Gaussian 原语和混合神经着色（neural diffuse BRDF + 解析 Cook-Torrance specular）从低成本 light stage 多视角序列重建可重光照、实时渲染的高保真头部 avatar，并发布了新的公开 OLAT 人脸数据集。

**[CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](clipgaussian_universal_and_multimodal_style_transfer_based_on_gaussian_splatting.md)**

:   CLIPGaussian 提出首个基于 Gaussian Splatting 的统一风格迁移框架，支持文本和图像引导的 2D 图像、视频、3D 物体和 4D 动态场景的风格化，作为即插即用模块集成到现有 GS 管线中，无需大规模生成模型或从头重训，且不改变模型大小。

**[Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)**

:   Concerto 将 3D 点云模态内自蒸馏与 2D-3D 跨模态联合嵌入预测相结合，以极简设计让单一点云编码器（PTv3）涌现出超越 2D/3D 单模态甚至两者拼接的空间表征，在多个 3D 场景理解基准上刷新 SOTA（ScanNet 语义分割 80.7% mIoU）。

**[Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)**

:   本文提出 Copresheaf Topological Neural Networks (CTNNs)，基于代数拓扑中的余预层（copresheaf）概念，在组合复形（combinatorial complex）上定义方向性、异质的消息传递机制，统一了 CNN、GNN、Transformer、Sheaf Neural Networks 和拓扑神经网络等多种深度学习架构，并在物理模拟、图分类和高阶复形分类任务上超越传统基线。

**[CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning](cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)**

:   提出 CosmoBench——目前最大的宇宙学几何深度学习基准，包含 3.4 万点云和 2.5 万有向树，覆盖多尺度、多视角、多任务，并揭示简单线性模型有时能超越大型 GNN。

**[Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)**

:   Cue3D是首个模型无关的图像线索重要性量化框架，通过系统性扰动光照、纹理、轮廓、透视、边缘和局部连续性6种视觉线索，在涵盖回归式/多视图/原生3D生成三大范式的7个方法上揭示了关键洞察：形状意义性而非纹理决定泛化能力，光照比纹理更重要，且模型过度依赖输入轮廓。

**[D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)**

:   提出 Static-Dynamic Aligned Pointmap (SDAP) 表示，将静态和动态区域的 3D 对齐统一建模，使 DUSt3R 系列方法能够在动态场景中实现准确的稠密三维重建与对应关系估计。

**[DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)**

:   提出基于方向一致性（Directional Consistency）的自适应密度控制方法 DC4GS，通过利用位置梯度的角度相干性来改进 3DGS 中的 primitive 分裂决策和分裂位置选择，在减少最多 30% primitive 数量的同时提升重建质量。

**[DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)**

:   提出 Dynamic Gaussian Hair (DGH)，一个数据驱动的 coarse-to-fine 框架，通过体素隐式变形模型学习头发动力学，并结合柱状 Gaussian 表示与曲率混合策略实现动态头发的逼真新视角渲染。

**[DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints](dualfocus_depth_from_focus_with_spatio-focal_dual_variational_constraints.md)**

:   提出 DualFocus，通过空间变分约束（利用焦距相关梯度模式区分深度边缘与纹理伪影）和焦距变分约束（强制单峰单调的对焦概率分布）双重约束，实现从焦距堆栈中鲁棒精确的深度估计。

**[Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)**

:   提出统一框架，通过可学习模糊核卷积联合建模散焦模糊和运动模糊，结合动态高斯致密化策略和未见视角约束，从模糊单目视频中实现高质量动态 3DGS 新视角合成。

**[DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)**

:   提出 DynaRend，通过掩码重建和未来预测两个互补目标，利用可微体渲染在 triplane 表征上联合学习 3D 几何、语义和动态信息，预训练后可高效迁移到下游机器人操控任务。

**[E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization](e-moflow_learning_egomotion_and_optical_flow_from_event_data_via_implicit_regula.md)**

:   提出 E-MoFlow，通过将光流建模为隐式神经表示、自运动建模为连续样条，并利用微分几何约束联合优化两者，在无监督范式下实现事件数据的 6-DoF 自运动和稠密光流联合估计。

**[EA3D: Online Open-World 3D Object Extraction from Streaming Videos](ea3d_online_open-world_3d_object_extraction_from_streaming_videos.md)**

:   提出 EA3D（ExtractAnything3D），一个在线开放世界 3D 物体提取框架，通过知识集成特征图、在线视觉里程计和循环联合优化，从流式视频中同时进行几何重建和全面场景理解。

**[EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)**

:   EAG3R 将事件相机的异步事件流融入 MonST3R 点图重建框架，通过 Retinex 增强模块 + SNR 感知融合机制 + 事件光度一致性损失，在极端低光动态场景下实现鲁棒的深度估计、位姿跟踪和 4D 重建，零样本迁移夜间场景即可大幅超越 RGB-only 方法。

**[EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)**

:   EF-3DGS 首次将事件相机引入自由轨迹场景重建，通过事件生成模型（EGM）重建帧间潜在图像做连续监督、对比度最大化（CMax）结合线性事件模型（LEGM）挖掘运动信息校准位姿，以及光度 BA + Fixed-GS 策略解决颜色不一致问题，在高速场景下 PSNR 提升 3dB、ATE 降低 40%。

**[ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals](electra_a_cartesian_network_for_3d_charge_density_prediction_with_floating_orbit.md)**

:   提出 ELECTRA（Electronic Tensor Reconstruction Algorithm），一种等变笛卡尔张量网络，通过预测浮动高斯轨道的位置、权重和协方差矩阵来重构电子密度，在 QM9 基准上精度比 SOTA 方法 SCDP 高 2.4 倍且推理速度快 4.4-11 倍，并将 DFT 的 SCF 迭代次数减少 50.72%。

**[EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation](enerverse_envisioning_embodied_future_space_for_robotics_manipulation.md)**

:   EnerVerse 是一个生成式机器人基础模型，通过 chunk-wise 自回归视频扩散 + 稀疏上下文记忆 + 多视角生成先验构建 4D 具身空间，结合 4DGS 数据飞轮缩小 Sim2Real 差距，最终通过策略头将 4D 世界表示转化为物理动作，在 LIBERO 基准上达到 SOTA。

**[EUGens: Efficient, Unified, and General Dense Layers](eugens_efficient_unified_and_general_dense_layers.md)**

:   EUGens 提出一类新的高效稠密层，利用随机特征（Random Features）将全连接前馈层的推理复杂度从二次降到线性，统一了已有的高效 FFL 扩展，在 LLM 预训练、ViT 图像分类、NeRF/iSDF 三维重建等任务中实现高达 27% 加速和 30% 参数压缩，且支持无需反向传播的层级知识蒸馏。

**[Evaluation of Vision-LLMs in Surveillance Video](evaluation_of_vision-llms_in_surveillance_video.md)**

:   提出一个无训练的两阶段框架，利用小型 Vision-LLM 生成视频文本描述 + NLI 分类器零样本评分，系统评估了提示策略和隐私保护滤镜对监控视频异常行为识别的影响。

**[Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation](every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)**

:   提出 4D Gaussian Ray Tracing (4D-GRT)，将 4D Gaussian Splatting 与物理光线追踪结合，从多视角视频重建动态场景后，以可控参数生成鱼眼畸变、景深模糊、卷帘快门等物理精确的相机效果视频数据。

**[Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)**

:   提出 Fin3R，通过冻结 decoder 并用带重归一化的 LoRA 适配器对 encoder 进行单目知识蒸馏微调，以统一且轻量的方式提升 DUSt3R/MASt3R/CUT3R/VGGT 等前馈式 3D 重建模型的几何精度和鲁棒性。

**[FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering](flarex_a_physics-informed_dataset_for_lens_flare_removal_via_2d_synthesis_and_3d.md)**

:   提出 FlareX 数据集，通过参数化模板创建、基于光照定律的 2D 合成和基于物理引擎的 3D 渲染三个阶段生成物理真实的镜头光晕数据，训练的模型在真实世界测试集上显著超越此前所有数据集。

**[Flux4D: Flow-based Unsupervised 4D Reconstruction](flux4d_flow-based_unsupervised_4d_reconstruction.md)**

:   提出 Flux4D，一个无监督且可泛化的 4D 动态驾驶场景重建框架，通过前馈网络直接预测 3D 高斯及其运动速度，仅用光度损失和静态偏好正则化实现大规模场景重建，在 PandaSet 和 Waymo 上超越所有无监督方法并接近有监督方法的性能。

**[From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes](from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)**

:   提出 Anywhere3D-Bench，首个涵盖区域/空间/物体/部件四个层级的 3D 视觉定位基准，揭示即使最强的 Gemini-2.5-Pro 和 o3 在空间级任务上仅达约 30% 准确率、部件级约 40%，远低于人类的 95%。

**[From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy](from_pixels_to_views_learning_angular-aware_and_physics-consistent_representatio.md)**

:   提出 XLFM-Former，通过 **视角级 Masked View Modeling（MVM-LF）** 自监督预训练学习 XLFM 的角度–空间先验，并设计基于 PSF 可微渲染的 **光学渲染一致性损失（ORC Loss）** 约束重建体积的物理合理性，在自建的首个 XLFM-Zebrafish 标准化基准上，平均 PSNR 达 54.04 dB，较最佳基线 ConvNeXt（50.16 dB）提升 **7.7%**。

**[From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries](from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)**

:   提出 FactoredScenes，将真实世界 3D 场景生成分解为五步因式分解——从合成数据学布局程序库、LLM 生成场景程序、执行程序获得轴对齐布局、程序条件化层次姿态预测、物体检索放置，在卧室上 FID 改善 38.3%、KID 改善 80.4%，人类仅 67% 能区分生成与真实 ScanNet。

**[Fully Dynamic Algorithms for Chamfer Distance](fully_dynamic_algorithms_for_chamfer_distance.md)**

:   提出首个全动态 Chamfer 距离维护算法，将问题归约为近似最近邻（ANN）查询，实现 $(1+\epsilon)$ 近似且更新时间 $\tilde{O}(\epsilon^{-d})$，大幅突破了静态重算的线性时间下界，在真实数据集上误差 <10% 且速度比朴素方法快数个数量级。

**[Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model](galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)**

:   提出一个多模态 Transformer 编解码框架，以廉价的暗物质 N-body 模拟的密度场和速度场为输入，自回归生成星系目录（位置 + 物理属性），在多种统计指标上忠实再现流体动力学模拟结果，计算加速约 100 倍。

**[GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)**

:   提出 GauDP，通过从多智能体的去中心化 RGB 观测中构建全局一致的 3D 高斯场，并将高斯属性动态分配回各智能体的局部视角，实现可扩展的、感知增强的多智能体协作模仿学习。

**[Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders](gaussian-augmented_physics_simulation_and_system_identification_with_complex_col.md)**

:   提出 AS-DiffMPM，一种支持任意形状刚体碰撞体的可微物质点法（MPM）框架，结合多种新视角合成方法实现从视觉观测中估计物体物理参数的系统辨识。

**[Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)**

:   提出 EgoSpanLift 方法，将第一人称 2D 注视预测提升到 3D 空间，构建多层级体积视觉跨度表示，结合 3D U-Net 和单向 Transformer 实现对未来 3D 视觉关注区域的预测。

**[GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)**

:   提出 GeoComplete，通过将投影点云作为几何条件注入双分支扩散模型，并结合 target-aware masking 策略，实现几何一致的参考驱动图像补全，PSNR 提升 17.1%。

**[GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction](geosvr_taming_sparse_voxels_for_geometrically_accurate_surface_reconstruction.md)**

:   提出基于稀疏体素的显式表面重建框架 GeoSVR，通过体素不确定性深度约束和稀疏体素表面正则化，在几何精度、细节保留和重建完整性方面全面超越现有基于 3DGS 和 SDF 的方法。

**[GOATex: Geometry & Occlusion-Aware Texturing](goatex_geometry_occlusion-aware_texturing.md)**

:   GOATex 提出首个遮挡感知的 3D 网格纹理生成框架，通过基于光线投射的 hit level 分层机制将网格分解为由外到内的可见性层，配合法线翻转和残差面聚类的两阶段可见性控制策略以及基于可见性权重的 UV 空间融合，实现了对外表面和被遮挡内表面的高质量纹理生成。

**[HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)**

:   HAIF-GS 提出基于稀疏运动锚点的动态 3DGS 框架，通过锚点过滤器区分动静区域、自监督诱导场景流引导时序一致变形、以及分层锚点加密捕捉精细非刚性运动，在 NeRF-DS 和 D-NeRF 基准上取得 SOTA 渲染质量。

**[High Resolution UDF Meshing via Iterative Networks](high_resolution_udf_meshing_via_iterative_networks.md)**

:   本文提出首个针对无符号距离场（UDF）的迭代式网格化方法，通过多轮次前向传播逐步将邻域信息传播到局部体素的伪符号预测中，有效解决了高分辨率下神经 UDF 噪声导致的表面空洞和不连续问题，在多个数据集上显著优于现有单遍方法。

**[How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?](how_many_tokens_do_3d_point_cloud_transformer_architectures_really_need.md)**

:   本文系统性地揭示了 3D 点云 Transformer（如 PTv3、Sonata）中存在 90-95% 的 token 冗余，并提出 gitmerge3D——一种全局信息感知的图 token 合并方法，通过能量分数自适应合并策略实现了高达 5.3× FLOPs 降低和 6.4× 显存节省而几乎不损失精度。

**[Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics](hybrid_physical-neural_simulator_for_fast_cosmological_hydrodynamics.md)**

:   提出一种混合物理-神经宇宙学模拟器，用可微分粒子网格（PM）方法处理引力动力学，用物理约束的神经网络参数化气体的有效压力场，仅需单次参考模拟即可训练，在场级别和统计量级别均优于 EGD 基线。

**[HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis](hyplanehead_rethinking_tri-plane-like_representations_in_full-head_image_synthes.md)**

:   系统分析了 tri-plane 类表征在 3D 感知头部合成中的三大问题（镜像伪影、不均匀映射、特征穿透），提出 hy-plane 混合表征（平面+球面）结合 unify-split 策略和近等面积映射，在全头图像合成中达到 SOTA。

**[HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)**

:   提出混合辐射场（HyRF），将紧凑的显式高斯体（仅存储8个参数）与解耦的网格神经场相结合，在实现 20× 模型压缩的同时达到 SOTA 渲染质量和实时性能。

**[IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)**

:   提出基于图像的高斯泼溅方法（IBGS），通过从邻近训练图像中学习颜色残差来增强标准3DGS的渲染质量，在不增加存储开销的前提下显著提升高频细节和视角依赖效果的建模能力。

**[IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants](indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)**

:   提出IndEgo——首个面向真实工业场景的大规模多模态第一人称视觉数据集，包含3,460段自我中心录像（约197小时）和1,092段外部视角录像（约97小时），覆盖装配/拆卸、物流、检修、木工等五大类任务及协作场景，并建立了错误检测、推理问答和协作理解三项基准。

**[Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)**

:   提出一类通用的稳定化适配器（Stabilization Adapters），可插入几乎任何图像模型架构中，通过冻结基础网络仅训练适配器参数，配合统一的精度-稳定性-鲁棒性损失函数，使帧级模型获得视频时序一致性和腐蚀鲁棒性。

**[Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)**

:   首次将Stable Diffusion的视觉先验引入自监督单目深度估计框架，提出Mix-Batch Image Reconstruction（MIR）代理任务保护SD先验不被重投影噪声损坏，并设计Scale-Shift GRU（SSG）桥接SD的尺度-偏移不变性（SSI）与自监督的尺度不变性（SI）深度分布，在KITTI上AbsRel=0.090达到所有SSMDE方法SOTA，且零样本泛化全面超越Marigold、E2E FT、Lotus等有监督SD方法。

**[LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](langsplatv2_high-dimensional_3d_language_gaussian_splatting_with_450_fps.md)**

:   通过将每个3D高斯视为全局字典上的稀疏编码，LangSplatV2用稀疏系数场替代重量级解码器，实现476.2 FPS的高维特征溅射和384.6 FPS的3D开放词汇查询，较LangSplat加速47倍。

**[Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)**

:   提出Fuse-and-Refine模块，通过混合Splat-Voxel表征将像素对齐的高斯基元聚合到粗到细的体素层次结构中，用稀疏体素Transformer在15ms内融合约20万基元并提升约2dB PSNR，且仅在静态场景训练即可零样本泛化到流式动态场景重建。

**[Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)**

:   提出神经曝光场（NExF），通过学习每个 3D 点的最优曝光值（而非每张图像的曝光），实现 3D 一致的高质量视图合成，在高动态范围场景中相比 SOTA 方法 PSNR 提升 3.5+，速度快 50 倍。

**[Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)**

:   提出 CDIM，一种基于 DDIM 的线性逆问题求解算法，通过将残差能量与前向扩散过程的 $\chi^2$ 分布对齐来自适应控制投影步数和步长，实现比 DPS 快 10-50 倍的推理速度，同时在无噪声情况下精确满足测量约束。

**[LinPrim: Linear Primitives for Differentiable Volumetric Rendering](linprim_linear_primitives_for_differentiable_volumetric_rendering.md)**

:   提出 LinPrim，用线性基元（八面体和四面体）替代3D高斯核作为新视角合成的场景表示，通过可微光栅化pipeline实现端到端优化，在真实数据集上以更少的基元数量达到与3DGS可比的重建质量，同时保持实时渲染能力。

**[Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)**

:   通过将 LSH 与 Point Transformer 结合，提出 HEPTv2 实现粒子轨迹重建的端到端学习，消除了 DBScan 聚类后处理瓶颈，在保持竞争性追踪效率的同时实现 28.9 倍加速。

**[LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)**

:   提出 LODGE，通过层次化 LOD（Level-of-Detail）策略对 3D Gaussian Splatting 进行多尺度管理，根据相机距离动态选择合适粒度的 Gaussian 表示，实现大规模场景的高质量实时渲染。

**[Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)**

:   Look and Tell 构建了一个多模态数据集，在厨房场景中使用 Meta Aria 智能眼镜和固定 GoPro 摄像头同步采集 25 名参与者的注视、语音和双视角视频，结合 3D 场景重建和多层级标注流水线，提供了首个跨第一人称/第三人称视角的指称交际研究基准。

**[MaNGO: Adaptable Graph Network Simulators via Meta-Learning](mango_-_adaptable_graph_network_simulators_via_meta-learning.md)**

:   提出 MaNGO（Meta Neural Graph Operator），通过元学习和条件神经过程（CNP）学习不同物理参数下仿真任务的共享潜在结构，实现对新物理参数的快速适应，无需重新训练。

**[MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)**

:   提出MaterialRefGS，通过多视角一致的材质推断约束和基于2DGS光线追踪的环境建模策略，实现反射表面的高保真新视角合成和精确光照分解。

**[Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)**

:   提出 Mesh-RFT 框架，通过拓扑感知评分系统和掩码直接偏好优化（M-DPO）实现面级别的细粒度网格质量优化，显著提升生成网格的几何完整性和拓扑规则性。

**[Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)**

:   提出 MIGN 框架，通过网格插值策略将不规则气象站数据映射到规则 HEALPix 网格上进行消息传递，并引入参数化球谐函数位置编码增强空间泛化能力，在全球天气预报任务中显著超越现有方法。

**[Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)**

:   提出BraInCoRL（Brain In-Context Representation Learning），一种基于Transformer的元学习框架，通过上下文学习（in-context learning）从少量刺激-响应样本直接预测新被试的体素级神经响应，无需微调即可适应新被试和新刺激，仅用100张图片就接近在9000张图片上完整训练的参考模型的性能。

**[MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)**

:   提出MetaGS，通过将可微Blinn-Phong反射模型嵌入3D高斯splatting并结合双层优化的元学习训练策略，实现在分布外（OOD）光照条件下的高质量3D场景重光照。

**[Metropolis-Hastings Sampling for 3D Gaussian Reconstruction](metropolis-hastings_sampling_for_3d_gaussian_reconstruction.md)**

:   提出自适应Metropolis-Hastings框架替代3DGS中的启发式密度控制机制，通过多视角光度误差驱动的概率采样实现更高效的高斯分布推断，收敛速度快于3DGS-MCMC。

**[More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)**

:   Merge提出了一种即插即用的框架，在固定的预训练T2I扩散模型前插入轻量级可学习的Converter，仅用约12%的额外参数就能赋予模型深度估计能力，同时完美保留原有的图像生成能力，在多个零样本深度估计基准上达到了统一模型的SOTA。

**[Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding](motion4d_learning_3d-consistent_motion_and_semantics_for_4d_scene_understanding.md)**

:   Motion4D提出了一个统一的4D高斯溅射框架，通过迭代优化策略将2D基础模型的先验（语义掩码、点追踪、深度）融入3D表示，实现了时空一致的运动和语义建模，在视频对象分割、点追踪和新视角合成任务上显著超越了现有方法。

**[Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)**

:   提出ComGS框架，利用动态场景中运动的局部性和一致性，通过仅约200个关键点驱动整个运动区域的高斯点运动，实现了相比3DGStream 159倍、相比QUEEN 14倍的存储压缩，同时保持了竞争性的视觉质量和渲染速度。

**[MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)**

:   MPMAvatar 将 Material Point Method (MPM) 物理仿真器与 3D 高斯溅射渲染相结合，通过各向异性本构模型和面向网格碰撞体的新碰撞处理算法，实现宽松衣物的精确鲁棒物理动画——在 ActorsHQ 和 4D-DRESS 上几何和外观全面超越 PhysAvatar，仿真成功率 100% vs 37.6%，单帧仿真仅需 1.1 秒。

**[NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)**

:   提出NerfBaselines评测框架，通过统一的评估协议、环境隔离和原始代码封装，解决了新视角合成领域因评估协议差异导致的不公平比较问题，并通过实验揭示了微小的协议差异（如图像缩放方式、背景颜色）可以显著改变方法排名。

**[Neural Green's Functions](neural_greens_functions.md)**

:   提出 Neural Green's Function，一种基于特征分解的可学习线性 PDE 解算子：从域几何中提取逐点特征来预测 Green 函数的特征分解，一次训练即可对任意源函数和边界条件通过数值积分求解，在机械零件热分析上比 SOTA 神经算子误差降低 13.9% 且比数值求解器快 350 倍。

**[Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)**

:   将稀疏输入新视角合成重新定义为测试时自然视频补全问题，利用预训练视频扩散模型的先验生成中间伪视图，并通过不确定性感知机制与 3D 高斯泼溅（3D-GS）迭代优化，在极稀疏输入下实现高保真场景重建。

**[Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction](object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)**

:   通过实证分析揭示物体特征可区分性是 3D 场景图谓词预测的关键瓶颈（物体分类错误导致 92%+ 的谓词错误），提出独立对比预训练的物体编码器（3D-2D-Text 三模态对齐）+ 几何正则化关系编码器 + 双向边门控 GNN，在 3DSSG 上 Object R@1 59.53%、Predicate R@50 91.40% 均达新 SOTA。

**[On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)**

:   提出 Geometry Encoding Mixer (GEM)，一种专为3D点云Transformer设计的几何感知PEFT模块，通过空间适配器捕获局部几何细节和上下文适配器注入全局场景信息，仅更新1.6%参数即可达到甚至超越全量微调性能。

**[Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)**

:   将在线3D实例分割重新建模为实例跟踪问题（AutoSeg3D），通过长期记忆进行实例关联、短期记忆进行实例更新、以及空间一致性学习缓解VFM过分割，在ScanNet200上超越ESAM 2.8 AP并保持实时性。

**[OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)**

:   提出 OnlineSplatter，一个无需相机位姿、深度先验或全局优化的前馈式在线3D重建框架，通过双键记忆模块（外观-几何潜在键 + 方向键）实现自由移动物体的恒定时间增量重建。

**[OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)**

:   提出 OpenLex3D，一个面向开放词汇 3D 场景表示的分层评测基准，在 Replica、ScanNet++、HM3D 三个数据集上提供 13 倍于原始标注的丰富语言标签，支持开放集 3D 语义分割和目标检索两项任务评测。

**[Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)**

:   提出 OriGS (Orientation-anchored Gaussian Splatting)，通过全局方向场引导和方向感知超维高斯表示，实现从随手拍摄的单目视频中进行高质量4D动态场景重建。

**[Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)**

:   提出朝向对齐3D物体生成任务，构建了跨1008个类别14832个朝向对齐3D模型的Objaverse-OA数据集，通过微调Trellis和Wonder3D两种主流3D生成框架实现朝向对齐的物体生成，并展示零样本朝向估计和箭头旋转操控两个下游应用。

**[PhysX-3D: Physical-Grounded 3D Asset Generation](physx-3d_physical-grounded_3d_asset_generation.md)**

:   PhysX提出了首个端到端的物理属性驱动3D资产生成范式，包括PhysXNet（首个系统标注了绝对尺度、材料、功能可供性、运动学和功能描述五个维度的物理3D数据集，含26K+对象）和PhysXGen（双分支前馈生成框架，将物理知识注入预训练的3D结构空间中）。

**[Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers](pixel-perfect_depth_with_semantics-prompted_diffusion_transformers.md)**

:   提出Pixel-Perfect Depth——在像素空间（而非潜空间）直接做扩散生成的单目深度估计模型，通过语义提示DiT（SP-DiT）引入视觉基础模型的高层语义表示和级联DiT设计，生成无飞点（flying-pixel-free）的深度图，在五个benchmark上超越所有已发表的生成式模型。

**[Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)**

:   提出Plana3R，一个无需位姿和平面标注的前馈框架，从未配对的双视角图像中预测稀疏3D平面基元和度量尺度相对位姿，实现室内场景的零样本度量平面3D重建。

**[PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)**

:   利用视觉语言基础模型（GroundedSAM）检测平面区域，结合DUSt3R多视图深度先验，通过共面约束和几何先验监督优化3DGS，实现室内场景的高保真表面重建。

**[PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)**

:   提出 PointMAC，首个将元辅助学习和测试时适应（TTA）引入点云补全的框架：通过 Bi-Aux Units（随机掩码重建+噪声去除）提供自监督信号，MAML 对齐辅助目标与主任务，推理时仅更新共享编码器实现样本级精化，在合成/模拟/真实数据上达到 SOTA。

**[Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)**

:   本文揭示了稀疏视角 3D Gaussian Splatting 中外观伪影的核心成因——高斯体之间的协同适应（co-adaptation）现象，提出了 Co-Adaptation Score（CA）度量指标来量化这一纠缠程度，并设计了 Gaussian Dropout 和不透明度乘性噪声注入两种即插即用的正则化策略，在五种基线方法和三个数据集上均显著降低了 co-adaptation 并提升了新视角渲染质量。

**[Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)**

:   提出 CogNVS，将动态场景新视角合成分解为三阶段管线——3D 重建（获取可见像素）→ 视频扩散修复（生成遮挡区域）→ 测试时微调（适应目标视频的分布），用纯 2D 视频自监督训练修复模型，实现零样本泛化到新测试视频。

**[Reconstructing the Local Density Field with Combined Convolutional and Point Cloud Architecture](reconstructing_the_local_density_field_with_combined_convolutional_and_point_clo.md)**

:   提出一种混合卷积（U-Net）与点云（DeepSets）的神经网络架构，用于从暗物质晕的视线方向特异速度重建局部暗物质密度场，在小尺度上显著优于纯卷积和线性重建方法。

**[Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)**

:   提出 Rectified Point Flow，一种统一的生成式框架，将成对点云配准和多部件形状组装统一为条件生成问题，通过学习连续点级速度场来估计部件位姿。

**[RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes](rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)**

:   ROS-Cam 提出仅用单个RGB视频作为监督的动态场景相机参数（焦距+位姿）优化方法，通过Patch-wise跟踪过滤器建立稀疏鲁棒对应关系、Cauchy分布异常值感知联合优化自适应降权运动物体、以及基于Softplus/凸极小分析的两阶段优化策略，在5个数据集上以最少监督实现最优精度和最快速度。

**[RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data](riganyface_scaling_neural_facial_mesh_auto-rigging_with_unlabeled_data.md)**

:   提出RigAnyFace（RAF），一个可扩展的面部网格自动绑定框架，通过2D监督策略利用无标注中性网格扩大训练规模，实现对多种拓扑和断连组件（如眼球）的高质量FACS混合形状绑定。

**[Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)**

:   AsymGS利用一个关键观察——野外训练数据引起的重建伪影具有随机性——提出非对称双3DGS框架，通过互补掩码策略和一致性约束抑制伪影，并引入Dynamic EMA Proxy实现高效训练，在多个野外数据集上显著超越现有方法。

**[ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)**

:   本文提出ROGR，利用多视角扩散重光照模型生成多光照条件下的一致图像，训练一个光照条件化的NeRF，实现任意环境光照下的前馈式3D物体重光照，在TensoIR和Stanford-ORB基准上达到SOTA性能且支持交互式渲染。

**[Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion](scaffold_diffusion_sparse_multi-category_voxel_structure_generation_with_discret.md)**

:   提出Scaffold Diffusion，将稀疏多类别3D体素视为token序列，使用Masked Diffusion Language Model（MDLM）配合3D正弦位置编码，在条件占用图上生成空间连贯的多类别体素结构，在极端稀疏（>98%背景）的Minecraft房屋数据集上显著优于自回归和传统离散扩散baseline。

**[Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)**

:   提出首个用于体素级全脑4D fMRI条件生成的扩散Transformer，结合3D VQ-GAN潜空间压缩、CNN-Transformer混合骨干网络和AdaLN-Zero+交叉注意力的强条件注入，在HCP七种认知任务上实现任务激活图相关0.83、RSA达0.98和完美条件特异性。

**[SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions](sceneforge_enhancing_3d-text_alignment_with_structured_scene_compositions.md)**

:   提出SceneForge框架，通过将单个3D点云对象组合成带显式空间关系的多物体场景，配合LLM精炼的组合描述，增强3D-文本对比学习的数据多样性和复杂度，在多个下游任务上带来一致性能提升。

**[SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent](sceneweaver_all-in-one_3d_scene_synthesis_with_an_extensible_and_self-reflective.md)**

:   提出SceneWeaver，首个用于3D场景合成的反思型智能体框架，通过标准化可扩展的工具接口统一多种场景生成范式，并利用reason-act-reflect闭环迭代优化，在物理合理性、视觉真实感和语义对齐上全面超越现有方法。

**[Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)**

:   提出"先分割再重建"的新范式，在3D高斯溅射重建之前就将高斯分配到不同目标集合，从而消除几何和语义歧义，实现静态和动态场景的统一3D开放词汇分割。

**[Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)**

:   提出 Shallow Flow Matching（SFM），在粗到细 TTS 框架中利用弱生成器输出构建 flow matching 中间状态，使推理从中间状态而非纯噪声出发，同时提升合成质量和加速推理。

**[SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference](singref6d_monocular_novel_object_pose_estimation_with_a_single_rgb_reference.md)**

:   提出SingRef6D，一个仅需单张RGB参考图像的轻量级6D位姿估计流水线，通过token-scaler微调Depth-Anything v2实现鲁棒深度预测，并引入深度感知匹配增强LoFTR的空间推理能力，在透明/反光物体场景中大幅超越现有方法。

**[SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)**

:   提出"语义朝向"(Semantic Orientation)概念，用自然语言描述物体方向（如 USB 的"插入方向"、杯子的"把手方向"），构建 OrienText300K 大规模数据集训练 PointSO 模型实现零样本朝向预测，并集成为 SoFar 系统实现 6-DoF 场景理解与机器人操作。

**[Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles](styl3r_instant_3d_stylized_reconstruction_for_arbitrary_scenes_and_styles.md)**

:   提出Styl3R前馈网络，通过结构-外观双分支架构将3D重建与风格化解耦，仅用未标定的稀疏视角图像和任意风格图像，在0.15秒内完成风格化3D重建。

**[SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction](synchuman_synchronizing_2d_and_3d_generative_models_for_single-view_human_recons.md)**

:   SyncHuman首次将2D多视图生成模型与3D原生生成模型统一在一个框架中，通过像素对齐的2D-3D同步注意力机制实现互补增强，在复杂人体姿态下实现了高保真纹理网格重建，几何精度和视觉质量均超越现有方法。

**[TAPIP3D: Tracking Any Point in Persistent 3D Geometry](tapip3d_tracking_any_point_in_persistent_3d_geometry.md)**

:   提出TAPIP3D，将视频表示为相机稳定化的时空3D特征点云，通过3D邻域到邻域（N2N）注意力机制在持久3D几何空间中迭代精化多帧点轨迹，显著超越现有3D点跟踪方法。

**[Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)**

:   提出首个端到端率失真（RD）优化的 4D 高斯泼溅压缩框架，通过 Haar 小波变换利用动态点轨迹的时序平滑先验，在 Ex4DGS 基础上实现高达 91× 的压缩率（平均模型仅约原始 1.1%），同时保持合理的渲染质量和灵活的率-质量权衡控制。

**[Towards 3D Objectness Learning in an Open World](towards_3d_objectness_learning_in_an_open_world.md)**

:   提出 OP3Det，一个无需文本提示的类无关开放世界 3D 检测器，通过 2D 基础模型进行 3D 物体发现，并设计跨模态混合专家（MoE）动态融合点云与图像特征，大幅提升新类别物体的召回率。

**[TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making](tp-mddn_task-preferenced_multi-demand-driven_navigation_with_autonomous_decision.md)**

:   提出任务偏好多需求驱动导航（TP-MDDN）基准和AWMSystem自主决策系统，通过指令分解、动态目标选择和任务状态监控三个LLM模块配合多维度累积语义地图，实现长程多子任务导航。

**[TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)**

:   提出TRIM（Trajectory Reduction and Instance Mask denoising），一种后训练框架，通过时间维度的轨迹预筛选和空间维度的背景token裁剪来加速3D高斯扩散模型推理，同时提升生成质量，在T3Bench文本生成3D和GSO图像生成3D任务上均优于DiffSplat等基线。

**[U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)**

:   提出 U-CAN 无监督点云去噪框架，通过 Noise2Noise 匹配方案和几何一致性约束实现多步去噪路径推断，性能逼近有监督方法，且一致性约束可泛化到 2D 图像去噪。

**[UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss](ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)**

:   提出 UGM2N 无监督网格移动网络，通过局部化 Node Patch 表示和 M-Uniform 损失函数实现无监督训练，在无需预适应网格数据的条件下实现跨 PDE 类型和跨网格几何的零样本泛化，且不产生网格缠绕。

**[UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)**

:   提出 UMAMI，一个统一掩码自回归模型（MAR）和确定性渲染的混合框架用于稀疏视角新视角合成：双向 Transformer 编码多视角图像 Token 和 Plücker 射线嵌入，两个轻量级 MLP 头分别处理可见区域（确定性回归）和遮挡区域（MAR 扩散生成），渲染速度比全生成基线快一个数量级。

**[URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model](urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)**

:   提出URDF-Anything，首个基于3D多模态大语言模型（MLLM）的端到端关节物体重建框架，通过[SEG] token机制实现几何分割与运动学参数的联合预测，在分割精度（mIoU提升17%）、参数误差（降低29%）和物理可执行性（超越基线50%）上均达到SOTA。

**[VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)**

:   通过引入边缘感知图像监督、可见性感知的多视图光度对齐、法线约束和深度图像特征对齐四种视图对齐（View Alignment）策略，显著提升3D高斯溅射的几何表示精度，在表面重建和新视图合成上取得SOTA。

**[VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)**

:   VisualSync提出了一个基于对极几何约束的多相机时间同步框架，利用预训练视觉模型（VGGT、CoTracker3、MAST3R）提取运动轨迹和跨视角对应关系，通过最小化Sampson误差来估计各相机的时间偏移，在四个数据集上达到了中位误差低于50ms的毫秒级同步精度。

**[Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)**

:   从理论上证明SDS是Schrödinger Bridge的特例，并基于此提出TraCe框架——在当前渲染和文本条件目标之间构建显式扩散桥，通过LoRA微调学习桥轨迹的score dynamics，在低CFG值下实现高质量text-to-3D生成。

**[WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)**

:   提出WildCAT3D，通过显式建模图像的全局外观条件，扩展多视角扩散模型（CAT3D）从野外互联网数据（如旅游照片）中学习场景级新视角合成，同时支持外观控制生成。

**[ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)**

:   从信息瓶颈（Information Bottleneck）原理出发分析前馈式3DGS的容量瓶颈，提出轻量级、与架构无关的ZPressor模块，通过将多视角输入压缩为紧凑的锚点视角表示，使现有模型能扩展到100+输入视角（480P，80GB GPU），在DL3DV-10K和RealEstate10K上持续提升性能。
