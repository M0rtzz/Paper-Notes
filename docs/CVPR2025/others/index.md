---
title: >-
  CVPR2025 其他方向 15篇论文解读
description: >-
  15篇CVPR2025 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**📷 CVPR2025** · 共 **15** 篇

**[4Real-Video Learning Generalizable Photo-Realistic 4D Video Diffusion](4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)**

:   提出4Real-Video，一种基于双流架构的4D视频生成框架，通过将视频token分为时间流和视角流并行处理，引入hard/soft同步层协调两流信息，约1分钟即可生成8×8的高质量时空视频网格，在视觉质量和多视角一致性上超越现有方法。

**[A2Z-10M Geometric Deep Learning With A-To-Z Brep Annotations For Ai-Assisted Cad](a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含100万+复杂CAD模型、超1000万多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的A2Z数据集，是目前最大的CAD逆向工程数据集，并基于此训练了BRep边界和角点检测的基础模型。

**[Anomalyncd Towards Novel Anomaly Class Discovery In Industrial Scenarios](anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)**

:   提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。

**[Bendfm A Taxonomy And Synthetic Cad Dataset For Manufacturability Assessment In ](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)**

:   提出一个面向板金弯曲工艺的可制造性度量分类法（按配置依赖性×可行性/复杂度两个维度划分为四象限），并构建首个包含20,000个零件（含可制造与不可制造样本）的合成数据集BenDFM，基准测试表明图结构表示（UV-Net）优于点云（PointNext），配置依赖性指标的预测更具挑战性。

**[Bounds On Agreement Between Subjective And Objective Measurements](bounds_on_agreement_between_subjective_and_objective_measurements.md)**

:   通过仅假设投票均值收敛于真实质量，推导出主观测试（MOS）与客观估计器之间PCC（上界）和MSE（下界）的数学界限，并提出基于二项分布的投票模型BinoVotes，使得即使在投票方差不可用时也能计算这些界限，18个主观测试数据的验证表明BinoVotes界限与全数据驱动界限高度吻合。

**[Cadcrafter Generating Computer-Aided Design Models From Unconstrained Images](cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)**

**[Deconstructing The Failure Of Ideal Noise Correction A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   通过提供完美的oracle噪声转移矩阵T，证明Forward Correction在理想条件下仍会训练崩塌（先升后降最终与无校正基线收敛），从宏观（收敛终态）、微观（梯度动力学）、信息论（噪声信道不可逆信息损失）三个层面系统诊断了失败的根本原因——这不是T估计不准的问题，而是有限样本下高容量网络的结构性缺陷。

**[Integration Of Deep Generative Anomaly Detection Algorithm In High-Speed Industr](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，在制药 BFS 高速产线上实现了仅用正常样本训练、单 patch 推理 0.17ms 的实时在线质检部署，通过 Perlin 噪声增强和 Noise Loss 优化重建质量。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors From Deterministic To Gen](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)**

:   提出基于风洞PIV实验数据的学习框架，系统对比Kriging插值与三种深度学习模型（UNet、ViTAE、CWGAN）在稀疏传感器屋顶风场重建任务中的表现，并结合QR分解优化传感器布局。深度学习在混合风向训练下全面优于Kriging，SSIM提升最高33.5%。

**[Sdf-Net Structure-Aware Disentangled Feature Learning For Opticall-Sar Ship Re-I](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶作为刚体的物理先验，在 ViT 中间层提取尺度不变的梯度能量统计量作为跨模态几何锚点，并在终端层将特征解耦为模态不变共享特征和模态特定特征后通过加性残差融合，实现光学-SAR 船舶重识别 SOTA。

**[Shrec A Spectral Embedding-Based Approach For Ab-Initio Reconstruction Of Helica](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出SHREC算法，利用谱嵌入技术从冷冻电镜二维投影图像中直接恢复螺旋分子的投影角度，无需预知螺旋对称参数（rise/twist），仅需已知轴对称群Cn，实现从头(ab-initio)螺旋结构重建。

**[Sldprtnet A Large-Scale Multimodal Dataset For Cad Generation In Language-Driven](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   构建包含24.2万工业零件的大规模多模态CAD数据集SldprtNet，提供3D模型、多视图图像、参数化文本脚本和自然语言描述的完整对齐，支持语义驱动的CAD建模任务。

**[Strap-Vit Segregated Tokens With Randomized -- Transformations For Defense Again](strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)**

:   STRAP-ViT 提出一种无需训练的即插即用 ViT 防御模块，利用 Jensen-Shannon 散度将受对抗补丁影响的 token 从正常 token 中分离出来，再通过随机复合变换消除其对抗效应，在多种 ViT 架构和攻击方法下实现了接近干净基线 2-3% 的鲁棒精度。

**[Wear Classification Of Abrasive Flap Wheels Using A Hierarchical Deep Learning A](wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)**

:   针对柔性磨料翻页轮的复杂磨损模式，提出三级层次化深度学习分类框架，将磨损评估分解为使用状态检测、磨损类型识别和严重程度评估三个子任务，使用EfficientNetV2迁移学习实现93.8%–99.3%的分类精度。

**[Zo-Sam Zero-Order Sharpness-Aware Minimization For Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，将零阶优化策略性地整合到 SAM 的扰动步骤中，仅需一次反向传播即可获得 SAM 的平坦最小值优势，在稀疏训练场景下将计算开销减半的同时提升精度和鲁棒性。
