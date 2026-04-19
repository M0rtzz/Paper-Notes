---
title: >-
  CVPR2026 其他方向 44篇论文解读
description: >-
  44篇CVPR2026 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📷 CVPR2026** · **44** 篇论文解读

**[A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含 1000 万+ 多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的 100 万+ CAD 模型数据集 A2Z，为 Scan-to-BRep 逆向工程和多模态 BRep 学习提供了前所未有的数据基础，并训练基础模型在边界/角点检测上大幅超越现有方法。

**[AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments](adasformer_adaptive_serialized_transformers_for_monocular_semantic_scene_complet.md)**

:   提出AdaSFormer，一种针对室内单目语义场景补全(MSSC)的序列化Transformer框架，通过自适应序列化注意力(可学习偏移量)、中心相对位置编码和卷积调制层归一化三个核心设计，在NYUv2和Occ-ScanNet上达到SOTA。

**[AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](assistmimic_physics_grounded_humanoid_assistance.md)**

:   首个在物理仿真中实现接触式人-人辅助行为模仿学习的多智能体RL框架，通过运动先验初始化、动态参考重定向和接触促进奖励使MARL在高接触设置中可行。

**[BenDFM: A Taxonomy and Synthetic CAD Dataset for Manufacturability Assessment in Sheet Metal Bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_ma.md)**

:   提出可制造性指标的二维分类法（配置依赖性 x 可行性/复杂度）和首个钣金弯曲合成 CAD 数据集 BenDFM（20,000 零件，含可制造和不可制造设计），基准测试显示拓扑感知的图表示（UV-Net, AUC 0.896）在四类任务上全面优于点云方法（PointNext, AUC 0.844）。

**[BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出可制造性度量的二维分类法（配置依赖性 × 可行性/复杂度），并构建首个面向钣金弯曲的合成数据集 BenDFM（20k零件），基准测试表明图结构表示（UV-Net）优于点云表示（PointNext），且配置依赖型指标更难预测。

**[Bounds on Agreement between Subjective and Objective Measurements](bounds_on_agreement_between_subjective_and_objecti.md)**

:   从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。

**[Bounds on Agreement between Subjective and Objective Measurements](bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   推导了主观测试 MOS 值与任意客观质量估计器之间 PCC 上界和 MSE 下界的数学闭式解，并提出基于二项分布的投票模型 BinoVotes 在缺少投票方差信息时估算该界。

**[U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)**

:   提出TextUnlock方法，通过训练轻量MLP将任意冻结视觉分类器的特征投射到文本嵌入空间（同时保持原分类器分布不变），无需CLIP、无需标注、无需训练线性探针，即可将任何legacy分类器转化为可解释的概念瓶颈模型——在40+架构上测试，超越甚至有监督的CLIP基CBM。

**[Coded-E2LF: Coded Aperture Light Field Imaging from Events](coded-e2lf_coded_aperture_light_field_imaging_from_events.md)**

:   首次证明仅用 event camera（无需传统 intensity 图像）即可重建像素级精度的 4D 光场，提出 Coded-E2LF 系统：通过编码光圈序列触发 events 并累积为 event images，利用全黑 pattern 建立 event-based 与 intensity-based coded aperture imaging 的数学等价性，结合端到端 deep optics 训练实现 8×8 视点光场重建。

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correcti.md)**

:   通过宏观收敛态、微观梯度动力学和信息论三个层次，严格证明了即使给定完美噪声转移矩阵，前向校正（FC）仍不可避免地塌缩到与无校正相同的次优水平，根本原因在于有限样本下的记忆化和噪声信道的信息损失。

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   本文通过受控实验证明，即使给定完美的噪声转移矩阵 T，前向校正方法仍会在训练后期发生性能崩溃，并从宏观收敛状态、微观优化动力学、信息论三个层面系统诊断了这一失败的根本原因。

**[DiffBMP: Differentiable Rendering with Bitmap Primitives](diffbmp_differentiable_rendering_with_bitmap_primitives.md)**

:   提出 DiffBMP——首个面向**位图图元**的通用可微渲染引擎，通过自定义 CUDA 并行管线实现对数千张位图图元的位置、旋转、缩放、颜色和透明度的高效梯度优化，填补了 2D 可微渲染仅限矢量图形的空白。

**[DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](dirpa_addressing_prior_shift_in_imbalanced_few-shot_crop-type_classification.md)**

:   提出 Dirichlet Prior Augmentation (DirPA)，通过在小样本学习训练过程中用 Dirichlet 分布采样模拟未知的长尾标签分布偏移，主动缓解训练集人工均衡与真实世界极端类别不平衡之间的先验偏移 (prior shift)，并在多个欧盟国家的作物分类任务上验证了跨区域的有效性。

**[DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)**

:   提出 Dirichlet 先验增强（DirPA），在少样本学习训练阶段从 Dirichlet 分布中采样类别比例向量来构造不平衡 episode，主动模拟真实世界长尾分布以消除先验偏移，在欧盟多个国家的作物分类任务中展示了一致的鲁棒性提升和稀有类别精度改善。

**[Dual-Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions](dual_band_thermal_videography_separating_time-varying_reflection_and_emission_ne.md)**

:   提出一种双波段长波红外视频分析框架，利用光谱线索（双波段发射率比恒定）和时间线索（物体辐射平滑变化、背景辐射突变）联合约束，首次实现近环境温度条件下动态场景中反射与发射分量的逐像素分离，并恢复物体发射率和温度场。

**[ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](enhancing_outofdistribution_detection_with_extende.md)**

:   诊断LogitNorm的特征坍缩问题(维度坍缩+原点坍缩)，提出ELogitNorm——用到决策边界的平均距离(而非特征范数)做自适应温度缩放，无超参数、兼容所有post-hoc OOD检测方法——CIFAR-10上far-OOD AUROC提升10.48%(SCALE)，ImageNet-1K上FPR95从51.45%降至27.74%，同时改善分类精度和ECE校准。

**[GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)**

:   本文提出 GazeOnce360，一个端到端的双分辨率 CNN 模型，用于从单个朝上放置的桌面鱼眼相机进行 360° 多人视线方向估计，同时构建了首个面向该场景的大规模合成数据集 MPSGaze360，在精度和速度两方面均大幅超越现有多阶段方法 GAM360。

**[HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)**

:   本文提出 HypeVPR，一个基于双曲空间层次化嵌入的视觉位置识别框架，专门解决透视图像（查询）与全景图像（数据库）之间的跨视场匹配问题，通过在 Poincaré 球中从局部到全局构建多级描述子，实现精度-效率-存储的灵活平衡，检索速度比滑窗基线快数倍且精度相当。

**[Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_a.md)**

:   基于GRD-Net改进的GAN+密集瓶颈残差自编码器（DRAE），在制药BFS生产线上实现半监督异常检测，用281万训练patch在500ms时间约束内完成推理（0.17ms/patch），达到97.62%平衡准确率。

**[Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   本文提出一个基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，专门设计用于制药行业 Blow-Fill-Seal（BFS）产线的高速在线质量检测，仅用合格品训练即可实现 96.4% 的准确率，单 patch 推理仅 0.17ms，满足 500ms 检测周期的严格工业约束。

**[IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)**

:   提出IrisFP模型指纹框架，通过将指纹放置在多类决策边界交叉点处、构建复合样本指纹、以及基于统计可分性的指纹筛选三项创新，同时增强指纹的唯一性和鲁棒性，在5个数据集上AUC一致超过SOTA方法。

**[Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)**

:   针对实例依赖偏标签学习 (ID-PLL) 中相似类别实例因特征和候选标签重叠导致的"实例纠缠"问题，提出 CAD 框架，通过类别特定增强的类内对齐和加权惩罚损失的类间分离，双管齐下缓解类混淆。

**[NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries](nailia_multimodal_nail_design_retrieval_based_on_dense_intent_descriptions_and_p.md)**

:   提出 NaiLIA，一种面向美甲设计图像的多模态检索方法，通过密集意图描述和调色板查询实现细粒度匹配，引入基于置信度分数的松弛对比损失（CRC loss）处理未标注正样本问题，在自建 NAIL-STAR 基准和 Marqo Fashion200K 上大幅超越现有方法。

**[Neural Collapse in Test-Time Adaptation](neural_collapse_in_test-time_adaptation.md)**

:   将神经坍缩 (Neural Collapse) 理论从类级别扩展到样本级别，发现了NC3+现象（样本特征嵌入与对应分类器权重对齐），基于此揭示了分布偏移下性能退化的根本原因是样本级特征-分类器错位，并提出NCTTA方法通过几何邻近度与预测置信度的混合目标引导特征重新对齐，在ImageNet-C上比Tent提升14.52%。

**[Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)**

:   MoScale 提出了一种 next-scale 自回归动作生成框架，替代传统 next-token 预测，通过从粗到细的层次化因果生成来捕获全局语义结构，并引入跨尺度层次精化和尺度内时间精化，在 HumanML3D 和 KIT-ML 上达到 SOTA（Top-1 0.540，FID 0.046）。

**[Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)**

:   针对工业异常检测中"正常"定义随规格变更而变化的实际问题，提出了两种新场景（A2N/N2A）、一个新评价指标（S-AUROC）和一种训练增强方法 RePaste，通过将高异常分数区域重新粘贴到训练图片中来增加其出现频率，使模型灵活适应正常样本定义的变化。

**[Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)**

:   提出 VRSketch2Shape 框架，首次建模 VR 草图的笔画时序信息，通过序列感知的 BERT 编码器与基于扩散的 3D 生成器（SDFusion），从有序 VR 草图生成高保真 3D 形状，同时贡献了包含 20k 合成 + 900 真实草图的多类别数据集。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)**

:   POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。

**[Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](rethinking_snn_online_training_and_deployment_grad.md)**

:   提出 HD-LIF（混合驱动 LIF）脉冲神经元模型族，通过在阈值上下区域采用不同脉冲计算机制，理论证明梯度可分离性和对齐性，解决 SNN 在线训练的前后向传播不一致问题，同时实现学习精度、内存复杂度和功耗的全阶段优化——以 10× 参数压缩、11× 功耗降低和 30% NOPs 节省达到 CIFAR-100 上 78.61% 精度。

**[Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](rethinking_snn_online_training_and_deployment_gradient-coherent_learning_via_hyb.md)**

:   提出 Hybrid-Driven LIF (HD-LIF) 模型族，通过在阈值上下区域采用不同脉冲计算机制实现梯度可分离性和对齐性，解决了 SNN 在线训练中前向-反向传播不一致的根本问题，同时实现了训练精度、内存复杂度和推理功耗的全阶段优化。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)**

:   建立基于PIV风洞实验数据的学习-观测框架，系统比较Kriging插值与三种深度学习模型（UNet/ViTAE/CWGAN）在5–30个稀疏传感器下的屋顶风场重建能力，揭示混合风向训练（MDT）下深度学习一致优于Kriging（SSIM提升18–34%），并通过QR分解优化传感器布局提升系统鲁棒性达27.8%。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   基于风洞PIV实验数据，系统比较了Kriging插值与三种深度学习方法（UNet、ViTAE、CWGAN）在稀疏传感器条件下的屋顶风场重建性能，并提出QR分解优化传感器布局以增强鲁棒性。

**[Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)**

:   提出 FECO 框架，通过鞋款风格–内容随机化（对抗训练）和地面感知学习（像素高度图 + 地面法线），从单张 RGB 图像实现鲁棒的密集足部接触估计，在多个基准上显著超越现有方法。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，通过谱嵌入（spectral embedding）从冷冻电镜 2D 投影图像中直接恢复螺旋分子片段的投影角度，无需预先知道螺旋对称参数（rise/twist），实现了真正的 ab-initio 螺旋结构重建。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embeddingbased_approach_for_abini.md)**

:   SHREC利用谱嵌入技术从2D冷冻电镜投影图像直接恢复螺旋分子的投影角度（无需螺旋对称参数先验），通过证明螺旋片段投影构成一维闭合流形(同胚于圆)实现角度恢复，在TMV、VipA/VipB和MakA三个公开数据集上实现接近发表水平的高分辨率重建(3.66Å–8.23Å)。

**[SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)**

:   提出 SimRecon 框架，通过"感知→生成→仿真"三阶段流水线，从真实视频自动构建仿真就绪的组合式 3D 场景，核心创新在于主动视角优化（AVO）为单物体生成寻找最优投影视角和场景图合成器（SGS）引导物理可信的层级化组装。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   构建了包含 242,000+ 工业零件的大规模多模态 CAD 数据集 SldprtNet，每个样本包含 .sldprt/.step 3D 模型、七视图合成图像、参数化建模脚本和自然语言描述四种模态的完整对齐数据，配套开发支持 13 种 CAD 命令的无损编码器/解码器工具，baseline 实验验证了多模态输入相比纯文本输入在 CAD 生成任务上的显著优势。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_largescale_multimodal_dataset_for_cad.md)**

:   构建SldprtNet——含242K+工业CAD零件的大规模多模态数据集，每个样本包含.sldprt/.step模型、7视角合成图、参数化建模脚本(13种命令无损编解码)和Qwen2.5-VL生成的自然语言描述，baseline实验验证多模态输入(图+文)在CAD生成上优于纯文本输入。

**[What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)**

:   本文从排名理论角度系统研究了 $F_\beta$ 分数族作为 Precision 与 Recall 排名折中的性质，证明 $F_\beta$ 诱导的排名构成 Precision 和 Recall 排名之间的测地线（最短路径），进而提出闭式公式来找到最优的 $\beta$ 值，并证明常用的 $F_1$ 和 skew-insensitive $F_1$ 在大多数情况下都不是最优排名折中。

**[What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)**

:   系统分析了现有渲染合成数据在语料、字体、布局多样性上的不足，提出 UnionST 合成引擎和自演化学习框架（SEL），仅用合成数据即大幅超越传统合成集，结合 SEL 仅需 9% 真实标注即可逼近全监督性能。

**[WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)**

:   提出 WildCap，通过混合逆渲染框架（数据驱动 SwitchLight 去光照 + 基于模型的 texel grid lighting 优化 + 扩散先验采样），从手机野外视频中重建高质量 4K 面部漫反射 albedo 贴图，大幅缩小野外捕捉与受控光照方法之间的质量差距。

**[Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)**

:   通过能量景观分析揭示 AT 和 JEM 的互补性（AT 对齐 clean-adv 能量分布 → 鲁棒性；JEM 对齐 clean-generated 能量分布 → 精度+生成），提出 EB-JDAT 建模联合分布 $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$ 并用 min-max 能量优化对齐三种数据能量分布，CIFAR-10 AutoAttack 鲁棒性 68.76%（超 SOTA AT +10.78%），同时保持 90.39% 清洁精度和 FID=27.42 的竞争力生成质量。

**[Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the_gaps_in_classification_robustn.md)**

:   提出 EB-JDAT 框架，通过建模干净样本、对抗样本和生成样本的联合能量分布 $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$，首次在单个模型中同时实现高分类精度、强对抗鲁棒性和具有竞争力的生成能力，在 CIFAR-10 上 AutoAttack 鲁棒性达 66.12%，超越 SOTA AT 方法超 10 个百分点。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，在 SAM 的扰动步骤中用零阶梯度估计替代反向传播，将 SAM 的计算开销从 2 次反传减少为 1 次，首次让 SAM 在稀疏训练中变得实用，在 CIFAR-10/100 和 ImageNet-1K 上一致提升所有主流稀疏训练方法 0.38%-2.54%。
