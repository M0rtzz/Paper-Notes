---
title: >-
  ICCV2025 其他方向 36篇论文解读
description: >-
  36篇ICCV2025 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📹 ICCV2025** · **36** 篇论文解读

**[A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)**

:   发现GCD中未标注数据（尤其是未知类别）的ViT注意力会分散到背景区域（distracted attention），提出Attention Focusing（AF）模块通过多尺度token重要性度量+自适应剪枝来纠正注意力，作为即插即用模块在SimGCD上最高带来15.4%的性能提升。

**[A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)**

:   本文提出 HOPS（Hyperdimensional One Place Signatures），利用超维计算（HDC）框架将同一地点在不同环境条件下采集的多个参考描述子融合为统一表示，在不增加计算量和存储开销的前提下，大幅提升视觉场所识别（VPR）的鲁棒性与召回率。

**[A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)**

:   本文提出了一种统一的线性 N-point 求解器，能够从具有任意时间戳的 2D 点对应中恢复相机线速度和 3D 点结构，适用于全局快门、滚动快门和事件相机等多种传感器模式。

**[AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)**

:   本文提出AdaptiveAE，利用深度强化学习将HDR曝光包围拍摄建模为马尔可夫决策过程（MDP），同时优化ISO和快门速度的组合，在用户定义的时间预算内自适应地为动态场景选择最优曝光参数，在HDRV数据集上达到PSNR 39.70，比之前最好的方法Hasinoff et al. (37.59) 高出2.1 dB。

**[AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)**

:   将多曝光HDR重建从MAP估计视角建模，通过空间对应先验将问题分解为对齐和融合两个交替子问题，再展开为端到端可训练的AFUNet（含SAM空间对齐+CFM通道融合+DCM数据一致性模块），在三个HDR基准上取得SOTA，PSNR-μ达44.91dB（Kalantari数据集）。

**[Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)**

:   首次将自回归（AR）模型引入多视角图像生成任务，通过逐视角生成利用所有前序视角信息来增强远距离视角间的一致性，同时设计了统一的多模态条件注入架构和Shuffle Views数据增强策略，使单一模型可同时处理文本/图像/几何形状条件。

**[C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)**

:   提出C4D框架，通过在DUSt3R的3D pointmap预测基础上联合捕获双重时序对应(短时光流+动态感知长时点跟踪DynPT)，生成运动掩码分离动静区域，并引入相机运动对齐/相机轨迹平滑/点轨迹平滑三个优化目标，将现有3D重建范式升级为完整4D重建(逐帧点云+相机参数+2D/3D轨迹)，在深度/位姿/跟踪多个下游任务上达competitive性能。

**[despite exploring contrastive deep skeletonpointcloudimutext](despite_exploring_contrastive_deep_skeletonpointcloudimutext.md)**

:   提出 DeSPITE，一个将 LiDAR 点云、骨架姿态、IMU 信号和文本四种模态对齐到联合嵌入空间的对比学习框架，首次以 LiDAR（而非 RGB）作为核心视觉模态，实现了跨模态匹配/检索等此前不可能的任务，同时作为有效的 HAR 预训练策略在 MSR-Action3D 和 HMPEAR 上取得 SOTA。

**[Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)**

:   提出首个基于草图的跨模态少样本关键点检测框架，利用原型网络、网格定位器、原型域适应和去风格化网络，仅需少量带标注草图即可在真实照片中检测新类别的新关键点。

**[EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)**

:   提出 EDFFDNet，采用指数衰减自由形变 (EDFFD) 替代传统 B-spline FFD 和 TPS 进行图像配准，配合自适应稀疏运动聚合器 (ASMA) 和渐进式相关策略，在 UDIS-D 数据集上以减少 70.5% 参数、32.6% 显存的代价实现 +0.5dB PSNR 提升。

**[Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training](failure_cases_are_better_learned_but_boundary_says_sorry_facilitating_smooth_per.md)**

:   揭示了对抗训练中一个反直觉现象——失败样本的模型感知变化反而比成功样本更小（即被"过度学习"），据此提出 Robust Perception Adversarial Training (RPAT)，通过鼓励感知随扰动平滑变化来缓解准确率-鲁棒性权衡问题。

**[FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases](fixtalk_taming_identity_leakage_for_high-quality_talking_head_generation_in_extr.md)**

:   提出FixTalk框架，通过增强运动指示器（EMI）和增强细节指示器（EDI）两个轻量级即插即用模块，将GAN模型中的身份泄漏问题"化害为利"——EMI消除运动特征中的身份信息以解决身份泄漏，EDI利用泄漏的身份信息在极端姿态下补充缺失细节以消除渲染伪影。

**[From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)**

:   提出渐进式主动学习（PAL）框架，通过"模型预启动→模型增强→模型精炼"三阶段训练策略，驱动红外小目标检测网络从易到难地主动识别和学习困难样本，在单点监督条件下显著缩小了与全监督方法之间的性能差距（IoU 提升 8.53%–29.1%）。

**[Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)**

:   提出基于扩散模型的即时类别发现框架 DiffGRE，通过属性组合生成（ACG）合成包含虚拟类别信息的新样本、多样性驱动精炼（DDR）过滤低质量样本、半监督Leader编码（SLE）注入额外类别知识，在 6 个细粒度数据集上显著提升了已有 OCD 方法的性能（平均 ACC-ALL 提升 6.5%）。

**[HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding](hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)**

:   提出 HyTIP 框架，将输出回归（显式缓冲解码帧）和隐状态传播（隐式缓冲潜在特征）两种时序信息传播机制统一到同一学习式视频编码框架中，仅用 SOTA 方法 14% 的缓冲区大小即可达到可比的编码性能。

**[I Am Big, You Are Little; I Am Right, You Are Wrong](i_am_big_you_are_little_i_am_right_you_are_wrong.md)**

:   利用因果推理 XAI 工具 rex 提取图像分类模型的最小充分像素集（MPS），系统比较 5 种架构 15 个模型的"注意力集中度"，发现大型模型（EVA/ConvNext）仅用图像 5% 像素即可做出分类，且不同架构的 MPS 在大小和位置上存在统计显著差异。

**[IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization](iap_invisible_adversarial_patch_attack_through_perceptibility-aware_localization.md)**

:   提出 IAP 框架，通过**感知感知（perceptibility-aware）的贴片定位**和**保色梯度更新**，首次实现在目标攻击场景下生成真正不可见的对抗补丁，同时能绕过多种 SOTA 补丁防御方法。

**[Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)**

:   本文通过提出"熵受限监督设定"建立了元学习与全类训练（WCT）的公平比较框架，从理论上证明了元学习有更紧的泛化界，并揭示了其对标签噪声更鲁棒、更适合异构任务的特性，据此提出 MINO 框架在无监督少样本和零样本任务上取得了 SOTA。

**[Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)**

:   Jigsaw++ 提出了一种基于生成模型的完整形状先验学习方法，通过"retargeting"策略将部分组装的碎片点云映射到完整物体的形状空间，与现有组装算法正交地提升重组质量。

**[Joint Asymmetric Loss for Learning with Noisy Labels](joint_asymmetric_loss_for_learning_with_noisy_labels.md)**

:   将非对称损失函数扩展到更复杂的被动损失场景，提出非对称均方误差（AMSE），严格建立其满足非对称条件的充要条件，并将 AMSE 嵌入 APL 框架构建联合非对称损失（JAL），在 CIFAR-10/100 等多个数据集上全面超越现有鲁棒损失函数方法。

**[Kaputt: A Large-Scale Dataset for Visual Defect Detection](kaputt_a_large-scale_dataset_for_visual_defect_detection.md)**

:   Kaputt 发布了一个包含 23 万+ 图像、4.8 万+ 独立商品的大规模零售物流缺陷检测数据集，规模是 MVTec-AD 的 40 倍，首次引入显著的姿态和外观变化，使得 SOTA 异常检测方法的 AUROC 不超过 56.96%，揭示了现有方法在真实零售场景中的严重不足。

**[LaCoOT: Layer Collapse through Optimal Transport](lacoot_layer_collapse_through_optimal_transport.md)**

:   提出 LaCoOT，一种基于最优传输的正则化策略，通过最小化网络内部中间特征分布之间的 Max-Sliced Wasserstein 距离，使得训练后可以直接移除整个网络层，在保持性能的同时显著减少模型深度和推理时间。

**[LayerD: Decomposing Raster Graphic Designs into Layers](layerd_decomposing_raster_graphic_designs_into_layers.md)**

:   提出 LayerD，通过迭代提取未遮挡顶层和背景补全来分解栅格图形设计为可编辑图层，并利用图形设计的域先验（纹理平坦区域）进行精炼，同时提出了基于 DTW 的层级评估协议。

**[Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)**

:   首次提出在双曲空间中编码用户定义的多层视觉层次结构的学习范式，通过基于角度的 entailment 对比损失在无需显式层次标签的情况下学习 scene→object→part 层次，并引入基于最优传输的层次检索评估指标。

**[Loss Functions for Predictor-based Neural Architecture Search](loss_functions_for_predictor-based_neural_architecture_search.md)**

:   首次对性能预测器中8种损失函数进行全面系统性研究，涵盖回归、排序和加权三大类，在5个搜索空间的13个任务上揭示了各类损失函数的特性与互补性，并提出分段损失（PW loss）组合方法PWLNAS，在多个基准上超越现有SOTA。

**[Magic Insert: Style-Aware Drag-and-Drop](magic_insert_style-aware_drag-and-drop.md)**

:   提出Magic Insert方法，首次形式化和解决"风格感知拖放"问题——将任意风格的主体拖入不同风格的目标图像中，主体自动适应目标风格且插入效果物理合理，核心包括风格感知个性化（LoRA+IP-Adapter风格注入）和Bootstrap Domain Adaptation（将真实图像训练的插入模型适配到风格化图像领域）。

**[Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)**

:   提出MIAFdR，首个能提供错误发现率（FDR）理论保证的成员推理攻击方法，通过设计新颖的非成员一致性分数函数和基于调整的成员判定策略来控制FDR，可作为即插即用的wrapper无缝集成到现有MIA方法中，在保持攻击性能的同时提供FDR控制。

**[NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)**

:   提出 NAPPure 框架，通过联合优化底层干净图像和扰动参数（基于似然最大化），将对抗纯化从仅处理加性扰动扩展到模糊、遮挡、几何扭曲等非加性扰动，在GTSRB上实现73.93%的平均鲁棒准确率（传统方法仅43.2%）。

**[Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration](omni-dc_highly_robust_depth_completion_with_multiresolution_depth_integration.md)**

:   提出 OMNI-DC，通过多分辨率深度积分器（Multi-res DDI）、Laplacian 损失和尺度归一化技术，构建了一个能够零样本泛化到不同数据集和稀疏深度模式的高鲁棒深度补全模型。

**[On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)**

:   提出统一的频谱框架来系统性分析和量化梯度解释的平滑性（复杂度）与忠实度之间的权衡，引入期望频率（EF）度量网络对高频信息的依赖程度，并通过将 ReLU 与高斯函数卷积来控制解释复杂度，同时定义"解释间隙"来量化替代模型导致的忠实度损失。

**[Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)**

:   提出Φ-GAN，将SAR的理想点散射中心（PSC）电磁散射物理模型以可微神经模块形式集成到GAN训练中，通过双物理损失（生成器物理一致性约束+判别器电磁特征蒸馏）显著提升数据稀缺场景下SAR图像生成的质量和稳定性。

**[Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)**

:   提出Neural Volumetric Prior (NVP)，通过融合显式3D特征网格与隐式MLP的混合神经表示，结合基于衍射光学的物理渲染方程，首次实现了从稀疏视角（仅6-7张荧光图像）对半透明生物样本3D折射率的高保真体积重建，所需图像数量减少约50倍、处理时间缩短3倍。

**[Recovering Parametric Scenes from Very Few Time-of-Flight Pixels](recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)**

:   本文探索用极少量（低至 15 个像素）低成本广视场 ToF 传感器恢复 3D 参数化场景几何的可行性，设计了前馈预测+可微渲染的分析-合成框架，在 6D 物体位姿估计等任务上展示了令人惊讶的效果。

**[Thermal Polarimetric Multi-view Stereo](thermal_polarimetric_multi-view_stereo.md)**

:   提出利用热偏振（长波红外偏振）线索进行精细三维形状重建的方法，理论证明 LWIR 偏振观测不受光照环境和材质光学属性的影响，从而实现对透明、半透明和异质材料物体的高精度三维重建，显著优于可见光偏振方法。

**[Toward Material-Agnostic System Identification from Videos](toward_material-agnostic_system_identification_from_videos.md)**

:   提出 MASIV，首个无需预定义材质先验的视觉系统辨识框架：采用可学习的神经本构模型替代手工设计的弹性/塑性方程，通过重建连续体粒子轨迹提供时间密集的几何约束，从多视角视频中推断物体的内在动力学特性。

**[You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception](you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)**

:   提出PHCP框架，首次在推理阶段解决异构协同感知的域差距问题——通过agent的伪标签做few-shot无监督域适应，自训练适配器对齐特征空间，无需联合训练即在OPV2V上仅用少量无标注数据达到接近SOTA(HEAL)的性能。
