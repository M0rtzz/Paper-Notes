---
title: >-
  NeurIPS2025 物理学方向19篇论文解读
description: >-
  19篇NeurIPS2025的物理学方向论文解读，涵盖域适应、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "物理学"
  - "论文解读"
  - "论文笔记"
  - "域适应"
  - "自监督学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理学

**🧠 NeurIPS2025** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (2)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/physics/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (6)](../../ICML2025/physics/index.md)

🔥 **高频主题：** 域适应 ×2

**[AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)**

:   提出 AstroCo，一种将 Conformer（注意力 + 深度可分离卷积 + 门控）引入天文不规则光变曲线的自监督编码器，在 MACHO 数据集上重建误差比 Astromer v1/v2 降低 61-70%，少样本分类 macro-F1 提升约 7%。

**[Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)**

:   用条件可逆神经网络（cINN）训练于15,777颗合成行星数据，从观测量（行星质量、轨道距离）快速推断行星形成参数（盘质量、湍流α、尘气比），实现比物理模型快~10⁶倍的概率性参数回溯，并证明多行星系统数据比单行星数据更鲁棒。

**[FAIR Universe HiggsML Uncertainty Dataset and Competition](fair_universe_higgsml_uncertainty_dataset_and_competition.md)**

:   提供2.8亿模拟LHC碰撞事件的标准化数据集和竞赛平台，包含6种参数化系统偏差（探测器校准+背景成分）及不对称覆盖惩罚评估指标，要求参赛者为Higgs信号强度$\mu$估计鲁棒的68.27%置信区间，优胜方案通过无聚焦替代建模实现比传统binned方法窄约20%的置信区间。

**[FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)**

:   提出 FEAT 框架，利用随机插值学习两个热力学系统之间的传输映射，基于 escorted Jarzynski 等式和 controlled Crooks 定理提供一致、最小方差的自由能差估计器及变分上下界，统一了平衡与非平衡方法。

**[From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)**

:   构建从模拟星系（TNG50）到真实巡天观测（SDSS）的域适应 pipeline，通过特征级对齐（欧几里得距离 + 最优传输 + top-$k$ 软匹配损失）和可训练权重调度，将星系形态分类的目标域准确率从 46.8%（无适应）提升到 87.3%，Macro F1 从 0.298 提升到 0.626。

**[Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)**

:   提出PHAZE框架，利用密码学哈希（Rabin指纹）和零知识机器学习（zkML）实现LHC触发器级别的可验证早退出推理，理论延迟降至~152-253ns量级，同时内建异常检测能力。

**[Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer](latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)**

:   将掩码点云 Transformer 自编码器引入重离子碰撞分析，通过自监督预训练+监督微调的两阶段范式，学习到比 PointNet 更强的非线性潜在表征（PC1 分布重叠从 2.42% 降至 0.27%），为 QGP 性质研究提供了通用特征学习框架。

**[Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器（MMAE）应用于星系图像（HSC-PDR2五波段）和光谱（DESI-DR1）的联合建模，构建134,533个星系的跨模态数据集GalaxiesML-Spectra，在75%掩码率下重建光谱主要发射线和图像形态，在光谱完全缺失时仅用图像实现 $\sigma_{\text{NMAD}}=0.016$ 的红移预测，优于AstroCLIP且红移范围首次扩展到 $z \sim 4$。

**[Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)**

:   提出一种神经网络方法，将 Nuker 星系轮廓参数映射为可解析反投影的 Multi Gaussian Expansion (MGE) 分量，从而在无需光学成像的情况下实现星系恒星质量建模，并集成到可微分动力学建模管道 SuperMAGE 中，对超大质量黑洞 (SMBH) 质量进行贝叶斯推断。

**[POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)**

:   构建首个系外行星偏振成像ML基准数据集POLARIS（921张VLT/SPHERE/IRDIS偏振图像+75,910张预处理曝光），提出Diff-SimCLR框架（扩散模型增强对比学习），在参考星vs目标星分类任务上达到93%准确率，仅需<10%手动标注。

**[Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)**

:   提出QDSFormer（量子双随机Transformer），用变分量子电路QontOT替代softmax生成双随机注意力矩阵，理论和实验证明量子电路生成的DSM更多样、更好保持信息，在多个小规模视觉识别任务上一致超越标准ViT和Sinkformer。

**[Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)**

:   首次将基于仿真的推断（SBI）应用于中微子相互作用模型参数调优，使用神经后验估计（NPE）从200K个GENIE模拟的58-bin直方图中学习4个物理参数的后验分布，在MicroBooNE Tune的mock数据上准确恢复了真实参数值。

**[The Pareto Frontier of Resilient Jet Tagging](the_pareto_frontier_of_resilient_jet_tagging.md)**

:   系统评估LHC射流标记任务中多种架构（DNN/PFN/EFN/ParT）的AUC-鲁棒性权衡，揭示更复杂模型虽AUC更高但对蒙特卡洛模型依赖性更强，构建Pareto前沿并通过案例研究证明低鲁棒性分类器即使校准后仍在下游参数估计中产生偏差。

**[The Platonic Universe: Do Foundation Models See the Same Sky?](the_platonic_universe_do_foundation_models_see_the_same_sky.md)**

:   在天文学场景下验证柏拉图表征假说（PRH）：使用JWST、HSC、Legacy Survey和DESI光谱数据，测量6种基础模型（ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer）的表征对齐度，发现模态内和跨模态MKNN分数随模型规模一致增加（p=3.31×10⁻⁵），支持不同架构和模态向共享表征收敛的假说。

**[TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)**

:   提出TITAN框架，用深度学习模型预测VQE中的"冻结参数"（训练过程中始终不活跃的参数），在初始化阶段即冻结40-60%参数，实现最高3倍收敛加速和40-60%电路评估量减少，在30量子比特的分子系统上匹配或超越基线精度。

**[Toward Complete Merger Identification at Cosmic Noon with Deep Learning](toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)**

:   在 IllustrisTNG50 模拟生成的模拟 HST CANDELS 图像上训练 ResNet18，首次证明深度学习可以在高红移 $1<z<1.5$ 下成功识别包括小质量比合并（minor merger, $\mu \geq 1/10$）和低质量星系（$M_\star > 10^8 M_\odot$）在内的星系合并，总体准确率约 73%，并通过 Grad-CAM 和 UMAP 深入分析了模型行为。

**[Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)**

:   研究从标准宇宙学模型（ΛCDM）预训练的神经网络能否迁移到超越标准模型的场景（大质量中微子、修改引力、原初非高斯性），发现dummy node架构可将模拟需求降低一个数量级，但当参数存在强物理简并（如σ₈-Mν）时会出现负迁移。

**[Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)**

:   用变分自编码器(VAE)对 2743 条 JWST 高红移($z>4$)星系光谱进行无监督聚类，发现 12 个不同的天体物理类别，使已知的后星暴星系、Lyman-α 发射星系、极端发射线星系、Little Red Dots 等稀有种群数量翻倍。

**[Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)**

:   首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。
