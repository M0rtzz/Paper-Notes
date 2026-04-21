---
title: >-
  CVPR2025 其他方向 42篇论文解读
description: >-
  42篇CVPR2025 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📷 CVPR2025** · **42** 篇论文解读

**[A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含100万+复杂CAD模型、超1000万多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的A2Z数据集，是目前最大的CAD逆向工程数据集，并基于此训练了BRep边界和角点检测的基础模型。

**[AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)**

:   提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。

**[BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出一个面向板金弯曲工艺的可制造性度量分类法（按配置依赖性×可行性/复杂度两个维度划分为四象限），并构建首个包含20,000个零件（含可制造与不可制造样本）的合成数据集BenDFM，基准测试表明图结构表示（UV-Net）优于点云（PointNext），配置依赖性指标的预测更具挑战性。

**[Bounds on Agreement between Subjective and Objective Measurements](bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   通过仅假设投票均值收敛于真实质量，推导出主观测试（MOS）与客观估计器之间PCC（上界）和MSE（下界）的数学界限，并提出基于二项分布的投票模型BinoVotes，使得即使在投票方差不可用时也能计算这些界限，18个主观测试数据的验证表明BinoVotes界限与全数据驱动界限高度吻合。

**[CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)**

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   通过提供完美的oracle噪声转移矩阵T，证明Forward Correction在理想条件下仍会训练崩塌（先升后降最终与无校正基线收敛），从宏观（收敛终态）、微观（梯度动力学）、信息论（噪声信道不可逆信息损失）三个层面系统诊断了失败的根本原因——这不是T估计不准的问题，而是有限样本下高容量网络的结构性缺陷。

**[Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)**

:   提出DPDL方法，通过学习多高斯分布原型并用Schrödinger桥将正常样本扩散映射到原型空间（同时推开异常样本），结合超球空间上的离散特征学习增强泛化性，在9个公开异常检测数据集上取得SOTA（如AITEX上超越AHL 5.0%、ELPV上超越8.7%）。

**[EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](edm_equirectangular_projection-oriented_dense_kernelized_feature_matching.md)**

:   提出EDM，首个基于学习的等距柱状投影（ERP）全景图像密集特征匹配方法，通过球面空间对齐模块（SSAM，使用3D笛卡尔坐标的球面位置编码+高斯过程回归）和测地线流细化处理ERP的极区畸变，在Matterport3D上AUC@5°超越DKM 26.72%、在Stanford2D3D上超越42.62%。

**[Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)**

:   提出EATTA方法，在长期测试时适应（TTA）中通过特征扰动敏感度每批次仅标注1个最有价值样本（而非多个），结合梯度范数去偏策略平衡监督和无监督损失的梯度，在ImageNet-C上以极低标注代价实现50.9%的平均错误率（超过标注3倍的SimATTA 3.9%）。

**[EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)**

:   提出EVOS方法，通过进化选择范式（稀疏适应度评估+频率引导交叉+增强无偏变异）对INR训练样本进行智能稀疏采样，在保持甚至提升重建质量（PSNR 37.81 vs 标准37.10）的同时将训练时间减少48-66%（180秒→97秒）。

**[Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)**

:   提出基于信噪比（SNR）的类特异性特征选择方法用于低秩生成模型（PPCA/LFA/ELF），每新增一个类只需$O(1)$计算（不需重训旧类模型），避免了灾难性遗忘，并提出新的非参数潜因子模型ELF，在微阵列癌症分类和高维特征选择上验证了有效性。

**[Focal Split: Untethered Snapshot Depth from Differential Defocus](focal_split_untethered_snapshot_depth_from_differential_defocus.md)**

:   受跳蛛视觉启发，构建首个无线（电池供电）的快照式差分离焦深度相机 Focal Split，用分光镜将光路分给两个不同焦距的传感器，仅需 500 FLOPs/像素和 4.9W 功率即可在树莓派上实时估计深度。

**[Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)**

:   提出首个仅用事件流估计完整6-DoF自运动（角速度+线速度）的几何求解器方法，通过建立事件扇形流形上的线段几何约束——入射关系和新颖的共面关系，设计最少仅需8个事件的稀疏求解器，无需IMU即可解耦旋转和平移估计。

**[H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)**

:   提出H2ST方法，用层次化的两样本检验架构实现增量学习中的OOD检测——每个任务对应一个特征级别的源-目标二分类器层，通过Clopper-Pearson置信区间假设检验自动判定ID/OOD（无需手动阈值），同时提供任务ID预测能力，在7个基准上优于MSP/Energy/ODIN且计算效率提升$(T+1)/2$倍。

**[Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)**

:   提出 FTM（Feature Tuning Mixup）通过在代理模型的特征空间中混合优化的攻击专用扰动和随机干净扰动来提升有目标对抗攻击的迁移性，使用动量式随机更新策略保持计算效率，14 个黑盒模型上平均成功率从 74.6% 提升到 77.4%。

**[Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)**

:   本文提出 ISO (Instance-wise Supervision-level Optimization) 框架，在主动学习中不仅选择哪些样本标注，还为每个样本自动决定最优的标注级别（精确标签 vs 粗标签），通过价值-成本比(VCR)和多样性感知的批次选择算法，在固定预算约束下达到比传统主动学习高10%+的准确率。

**[Integral Fast Fourier Color Constancy](integral_fast_fourier_color_constancy.md)**

:   本文提出 IFFCC，将 FFCC 算法扩展到多光源场景，通过积分 UV 直方图加速区域直方图计算并行化傅里叶卷积操作，实现了与像素级神经网络相当的精度，同时参数量减少 400 倍、速度提升 20-100 倍的实时多光源自动白平衡。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，在制药 BFS 高速产线上实现了仅用正常样本训练、单 patch 推理 0.17ms 的实时在线质检部署，通过 Perlin 噪声增强和 Noise Loss 优化重建质量。

**[Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)**

:   提出 Open-Set Discovery Active Learning (OSDAL) 场景，并设计 Joda 算法，通过训练-过滤-选择三阶段流程，用单一模型同时过滤 OOD 数据和发现新类别，无需额外辅助模型，在 18 个配置上持续达到最高准确率。

**[LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos](latte-mv_learning_to_anticipate_table_tennis_hits_from_monocular_videos.md)**

:   LATTE-MV 提出一套从单目乒乓球比赛视频中重建 3D 比赛数据的可扩展系统，并训练 Transformer 模型预判对手击球意图，结合共形预测实现不确定性感知的预判式控制，将仿真中机器人回球率从 49.9% 提升至 59.0%。

**[Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)**

:   提出利用局部无序图像（LOI）的三维尺度空间（内尺度 σ、色调尺度 β、范围尺度 α）进行直方图匹配的逆渲染优化方法，无需修改可微渲染器即可扩展稀疏梯度的支持范围，有效避免局部最优。

**[MOS: Modeling Object-Scene Associations in Generalized Category Discovery](mos_modeling_object-scene_associations_in_generalized_category_discovery.md)**

:   挑战了GCD中"场景信息是噪声"的传统观点，发现场景被误解为噪声是因为"歧义挑战"（目标与场景的base/novel关系冲突），提出MOS框架通过双分支网络+MLP场景感知模块有效利用场景信息，在细粒度GCD上平均提升4%。

**[NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)**

:   NeISF++ 将偏振逆渲染从仅支持介电体扩展到同时支持导体和介电体，通过引入二元控制变量 $m$ 的广义 pBRDF 模型、复折射率建模和 DoLP 几何初始化，在合成导体场景上法线角度误差降至 1.789°（比 NeISF 的 10.303°低 83%）。

**[Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)**

:   提出 NoT 三阶段框架解决开集标签偏移（OSLS）：通过参考数据集估计源域 ID/OOD 比例，用 EM 算法最大似然估计目标域标签分布，再校正不完美 OOD 分类器的偏差，无需重训练分类器即可适应标签分布变化。

**[Order-One Rolling Shutter Cameras](order-one_rolling_shutter_cameras.md)**

:   建立了卷帘快门（Rolling Shutter, RS）相机的统一代数理论，定义了"一阶RS相机"（RS1）——将空间点一一映射到图像点的卷帘模型，完整分类了其三种类型（I/II/III），并发现了 31 个 2-5 相机间相对位姿估计的最小问题。

**[PLeaS: Merging Models with Permutations and Least Squares](pleas_-_merging_models_with_permutations_and_least_squares.md)**

:   提出 PLeaS，一种两步模型合并算法：第一步利用置换对称性部分匹配两个模型的特征（相似特征合并、不相似特征保留），第二步通过逐层最小二乘优化使合并模型的特征逼近原模型置换后的集成特征，在相同模型大小下比现有方法提升高达 15 个百分点。

**[Regor: Progressive Correspondence Regenerator for Robust 3D Registration](progressive_correspondence_regenerator_for_robust_3d_registration.md)**

:   Regor提出了一种渐进式对应关系再生策略，不同于传统的"自上而下"外点剔除方法，通过"自下而上"地在局部球体内迭代生成更多高质量对应关系，生成的正确匹配数量是现有方法的10倍，即使在弱特征条件下也能实现鲁棒配准。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   建立基于风洞 PIV 实验数据（非 CFD 模拟）的屋顶风场重建框架，系统对比 Kriging 插值与三种深度学习模型（UNet、ViTAE、CWGAN）在 5-30 个稀疏传感器下的重建性能，发现混合风向训练（MDT）使深度学习全面超越 Kriging（SSIM 提升最高 32.7%），并用 QR 分解优化传感器布局提升鲁棒性达 27.8%。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶作为刚体的物理先验，在 ViT 中间层提取尺度不变的梯度能量统计量作为跨模态几何锚点，并在终端层将特征解耦为模态不变共享特征和模态特定特征后通过加性残差融合，实现光学-SAR 船舶重识别 SOTA。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，利用图拉普拉斯算子的谱嵌入技术，从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群 $C_n$，在多个公开数据集上实现了接近原子分辨率的从头螺旋结构重建。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   本文构建了一个包含24万+工业零件的大规模多模态CAD数据集 SldprtNet，每个样本对齐了3D模型、多视角图像、参数化建模脚本和自然语言描述四种模态，并开发了支持13种CAD操作的编码器/解码器工具实现无损双向转换，实验证明多模态输入显著优于纯文本输入。

**[STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)**

:   STRAP-ViT 提出一种无需训练的即插即用 ViT 防御模块，利用 Jensen-Shannon 散度将受对抗补丁影响的 token 从正常 token 中分离出来，再通过随机复合变换消除其对抗效应，在多种 ViT 架构和攻击方法下实现了接近干净基线 2-3% 的鲁棒精度。

**[TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)**

:   提出TAET两阶段对抗均衡训练框架：先用交叉熵损失稳定早期训练，再用层级对抗鲁棒学习(HARL)联合BCL/HDL/RCEL三种损失均衡各类性能，并引入平衡鲁棒性(Balanced Robustness)评估指标，解决长尾分布下对抗训练的尾部类鲁棒性不足问题。

**[Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)**

:   提出TagFex框架，通过持续自监督学习捕获任务无关(task-agnostic)特征，并利用merge attention将其与任务特定特征融合后蒸馏回推理模型，缓解扩展式类增量学习中的特征碰撞问题。

**[Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)**

:   发现测试时数据增强（TTA）可以系统性地提升共形预测的效率——通过在校准集上学习增强权重来优化增强聚合策略，在 ImageNet ResNet-50 上将预测集大小减少 10-17%，同时严格保持覆盖率保证。

**[Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)**

:   提出从三视图单应性矩阵中恢复焦距的高效求解器，利用法向量一致性约束推导出新的显式约束，将问题转化为单变量或双变量多项式求解，速度比现有方法快 80-270 倍。

**[Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)**

:   本文提出 Probability Margin Attack (PMA)，在概率空间而非 logits 空间定义对抗边际损失函数，其梯度等价于无目标和有目标交叉熵损失的自适应加权组合，一致性地超越现有个体攻击方法；基于此构建百万级评估数据集 CC1M，首次开展对抗训练模型的百万规模白盒鲁棒性评估。

**[VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)**

:   VKDNW提出了一种基于Fisher信息矩阵（FIM）特征值谱熵的训练无关NAS代理，首次成功地将Fisher信息理论应用于大规模深度网络架构搜索，无需任何训练即可评估网络分类精度潜力，并提出了更适合NAS任务的nDCG评估指标。

**[UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)**

:   提出 UniPhy，首个统一的潜变量条件本构模型，在共享潜空间中编码弹性体/沙子/塑料/牛顿/非牛顿流体等多种材料属性，推理时通过可微 MPM 仿真器优化潜变量以匹配观测粒子轨迹，重建误差比 NCLaw 低 1-2 个数量级。

**[Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)**

:   针对柔性磨料翻页轮的复杂磨损模式，提出三级层次化深度学习分类框架，将磨损评估分解为使用状态检测、磨损类型识别和严重程度评估三个子任务，使用EfficientNetV2迁移学习实现93.8%–99.3%的分类精度。

**[Which Viewpoint Shows It Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)**

:   LangView 提出用语言作为弱监督信号训练多视角教学视频的视角选择模型：通过比较各视角预测字幕与视角无关叙述的匹配度生成伪标签，无需人工最佳视角标注即可学习自动切镜。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，将零阶优化策略性地整合到 SAM 的扰动步骤中，仅需一次反向传播即可获得 SAM 的平坦最小值优势，在稀疏训练场景下将计算开销减半的同时提升精度和鲁棒性。
