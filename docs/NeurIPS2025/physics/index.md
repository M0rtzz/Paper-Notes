---
title: >-
  NeurIPS2025 物理/科学计算方向41篇论文解读
description: >-
  41篇NeurIPS2025的物理/科学计算方向论文解读，涵盖域适应、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
  - "域适应"
  - "自监督学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**🧠 NeurIPS2025** · **41** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (5)](../../CVPR2026/physics/index.md) · [🔬 ICLR2026 (12)](../../ICLR2026/physics/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (14)](../../ICML2025/physics/index.md)

🔥 **高频主题：** 域适应 ×3

**[A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees](a_regularized_newton_method_for_nonconvex_optimization_with.md)**

:   提出一类基于当前与历史梯度构造的新型正则化器，结合带负曲率监测的共轭梯度法求解正则化Newton方程，在不需要Hessian Lipschitz常数先验知识的自适应框架下，首次同时实现了$O(\epsilon^{-3/2})$最优全局迭代复杂度和二次局部收敛速率。

**[AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)**

:   提出 AstroCo，一种将 Conformer（注意力 + 深度可分离卷积 + 门控）引入天文不规则光变曲线的自监督编码器，在 MACHO 数据集上重建误差比 Astromer v1/v2 降低 61-70%，少样本分类 macro-F1 提升约 7%。

**[Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios](bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)**

:   提出基于贝叶斯神经网络（BNN）的代理模型，用于替代昂贵的非线性有限元分析（NLFEA），实现对老化桥梁组合的快速、不确定性感知的结构安全预评估，在真实铁路案例中为单座桥梁节省约37万美元。

**[Collapsing Taylor Mode Automatic Differentiation](collapsing_taylor_mode_automatic_differentiation.md)**

:   提出 Taylor mode 自动微分的"折叠"(collapsing)优化技术，通过重写计算图将导数求和操作向上传播，大幅加速 PDE 算子（如 Laplacian、一般线性 PDE 算子）的计算，实现速度优于嵌套反向传播同时保持前向模式的低内存优势。

**[DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)**

:   提出 DeltaPhi 框架：不直接学习 PDE 的输入→输出映射，而是学习**相似物理状态之间的残差**，利用物理系统稳定性实现隐式数据增强，在数据稀缺场景下显著提升各类神经算子的性能。

**[EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)**

:   提出 EddyFormer，一种基于谱元法 (SEM) 的 Transformer 架构，将流场分解为 LES（大尺度）和 SGS（小尺度）两路并行流，在 256³ 分辨率 3D 湍流上达到 DNS 级精度且加速 30 倍，并在未见的 4× 更大域上泛化良好。

**[Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)**

:   提出两种无需训练的后处理投影方法（非线性LBFGS优化和局部线性化投影），将神经PDE求解器的输出投影到满足控制方程约束的可行流形上，在Lorenz/KS/Navier-Stokes上大幅降低约束违反并提升精度，且效果显著优于physics-informed训练。

**[Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)**

:   用条件可逆神经网络（cINN）训练于15,777颗合成行星数据，从观测量（行星质量、轨道距离）快速推断行星形成参数（盘质量、湍流α、尘气比），实现比物理模型快~10⁶倍的概率性参数回溯，并证明多行星系统数据比单行星数据更鲁棒。

**[F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)**

:   本文首次系统研究了科学机器学习中预训练大型算子模型(LOM)的参数高效微调(PEFT)，发现 LoRA 在傅里叶层中存在深度放大的近似误差下界，而 Adapter 保留了通用逼近能力；据此提出频率自适应 Adapter（F-Adapter），按频谱能量分配 Adapter 容量，在 3D Navier-Stokes 预测任务上仅调参不到 2% 即达到 SOTA。

**[FAIR Universe HiggsML Uncertainty Dataset and Competition](fair_universe_higgsml_uncertainty_dataset_and_competition.md)**

:   提供2.8亿模拟LHC碰撞事件的标准化数据集和竞赛平台，包含6种参数化系统偏差（探测器校准+背景成分）及不对称覆盖惩罚评估指标，要求参赛者为Higgs信号强度$\mu$估计鲁棒的68.27%置信区间，优胜方案通过无聚焦替代建模实现比传统binned方法窄约20%的置信区间。

**[FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)**

:   提出 FEAT 框架，利用随机插值学习两个热力学系统之间的传输映射，基于 escorted Jarzynski 等式和 controlled Crooks 定理提供一致、最小方差的自由能差估计器及变分上下界，统一了平衡与非平衡方法。

**[From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)**

:   提出基于 Neural Operator 的「子网格黑洞」模型，学习小尺度 (GR)MHD 时间演化算子，替代手工闭合规则嵌入多层级直接数值模拟框架，首次实现吸积驱动反馈的内禀变异性捕获，加速比达 $\sim 10^5$ 倍。

**[From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows](from_images_to_physics_probabilistic_inference_of_galaxy_parameters_and_emission.md)**

:   提出 VAE–Normalizing Flow 混合框架，从 SDSS gri 图像和测光数据出发，以概率方式联合推断星系物理参数（恒星质量、SFR、红移、气相金属丰度、中心黑洞质量）和发射线流量（Hα、Hβ、[N II]、[O III]），速度比 SED 拟合快 100 倍以上且提供校准良好的后验分布。

**[From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)**

:   构建从模拟星系（TNG50）到真实巡天观测（SDSS）的域适应 pipeline，通过特征级对齐（欧几里得距离 + 最优传输 + top-$k$ 软匹配损失）和可训练权重调度，将星系形态分类的目标域准确率从 46.8%（无适应）提升到 87.3%，Macro F1 从 0.298 提升到 0.626。

**[GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)**

:   首次提出可扩展的5D神经网络代理模型 GyroSwin，将 Swin Transformer 扩展至5维回旋动力学相空间，通过交叉注意力实现3D↔5D交互、通道式模态分离捕获带状流，在等离子体湍流模拟中实现比传统准线性方法更高的精度，且比数值求解器（GKW）快3个数量级。

**[Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)**

:   基于 Riesz 表示定理，用可学习核积分（Integral Kernel Functional）近似无限维 Hamiltonian 泛函，通过自动微分获取泛函导数，实现保能量的神经 PDE 求解器（HNS），在 1D/2D PDE 上展现出优越的稳定性和泛化能力。

**[INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)**

:   提出间接神经校正器(INC)，将学习到的校正项嵌入PDE的右端项（而非直接修改状态），理论证明误差放大降低$\mathcal{O}(\Delta t^{-1}+L)$倍，在6个PDE系统（1D混沌到3D湍流）上大幅改善长期轨迹性能（R²提升达158.7%），实现最高330×加速。

**[Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)**

:   揭示了标准 BSDE 方法性能不如 PINNs 的根本原因是 Euler-Maruyama 积分引入的不可消除离散化偏差，提出基于 Stratonovich 形式的 Heun-BSDE 方法彻底消除该偏差，在高维 PDE 上与 PINNs 竞争。

**[Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)**

:   提出PHAZE框架，利用密码学哈希（Rabin指纹）和零知识机器学习（zkML）实现LHC触发器级别的可验证早退出推理，理论延迟降至~152-253ns量级，同时内建异常检测能力。

**[Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer](latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)**

:   将掩码点云 Transformer 自编码器引入重离子碰撞分析，通过自监督预训练+监督微调的两阶段范式，学习到比 PointNet 更强的非线性潜在表征（PC1 分布重叠从 2.42% 降至 0.27%），为 QGP 性质研究提供了通用特征学习框架。

**[Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)**

:   针对最优交易执行中 HJB 方程的硬零终端库存约束（$X_T=0$），提出 Multi-Trajectory PINN (MT-PINN)，通过基于轨迹展开的终端损失与 $\lambda$-curriculum 训练策略，在合成数据和 SPY 实盘回测中显著优于 vanilla PINN，终端库存违规率大幅降低。

**[Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)**

:   提出一种神经网络方法，将 Nuker 星系轮廓参数映射为可解析反投影的 Multi Gaussian Expansion (MGE) 分量，从而在无需光学成像的情况下实现星系恒星质量建模，并集成到可微分动力学建模管道 SuperMAGE 中，对超大质量黑洞 (SMBH) 质量进行贝叶斯推断。

**[Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)**

:   挑战了"神经 PDE 模拟器精度受限于训练数据（数值求解器）精度"的传统认知，发现并严格定义了 **emulator superiority** 现象——仅在低精度求解器数据上训练的神经网络，在以高精度参考解评估时竟能超越其训练求解器本身。

**[Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)**

:   NeuSA 将经典谱方法与 Neural ODE 结合，先将 PDE 投影到谱基（Fourier）上得到 ODE 系统，再用 NODE 学习动力学演化，从架构层面解决了传统 PINN 的谱偏差和因果性问题，在波动方程/Burgers方程/sine-Gordon方程上误差比 baseline 低 1-2 个数量级且训练更快。

**[One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)**

:   将微扰理论与 PINNs 结合，将非线性PDE分解为线性子问题序列，用 Multi-Head PINN 学习线性算子的潜空间后，对新的PDE实例可通过闭式解在0.2秒内完成迁移，达到 $10^{-3}$ 量级误差。

**[Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)**

:   提出混合 ML–EPM 框架：用轻量 CNN 学习从 RANS 湍流动能场到 DNS 真值的修正映射，以此调制特征空间扰动法（EPM）的扰动幅度，在保持物理一致性的前提下将湍流模型不确定性估计的误差降低 1–2 个数量级。

**[Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)**

:   提出 Spectral PINNsformer (S-Pformer)，用 Fourier 特征嵌入替换 PINNsformer 的编码器，结合仅解码器 Transformer 架构，在减少 18.6% 参数量的同时在多个 PDE benchmark 上取得更优性能，有效缓解了频谱偏置问题。

**[POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)**

:   构建首个系外行星偏振成像ML基准数据集POLARIS（921张VLT/SPHERE/IRDIS偏振图像+75,910张预处理曝光），提出Diff-SimCLR框架（扩散模型增强对比学习），在参考星vs目标星分类任务上达到93%准确率，仅需<10%手动标注。

**[Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)**

:   提出QDSFormer（量子双随机Transformer），用变分量子电路QontOT替代softmax生成双随机注意力矩阵，理论和实验证明量子电路生成的DSM更多样、更好保持信息，在多个小规模视觉识别任务上一致超越标准ViT和Sinkformer。

**[Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)**

:   首次将基于仿真的推断（SBI）应用于中微子相互作用模型参数调优，使用神经后验估计（NPE）从200K个GENIE模拟的58-bin直方图中学习4个物理参数的后验分布，在MicroBooNE Tune的mock数据上准确恢复了真实参数值。

**[Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)**

:   本文研究了两层过参数化 ReLU 网络中稳定极小值（flat minima）的泛化性质，证明虽然平坦性确实蕴含泛化，但其收敛速率随输入维度指数级恶化（即存在维度灾难），与不受维度灾难影响的低范数解（weight decay）形成指数级分离；并揭示了"neural shattering"现象作为高维失败的几何机制。

**[Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)**

:   利用符号回归（Symbolic Regression）从数值相对论模拟数据中自动发现双中子星并合后吸积盘质量的解析标定关系，所得紧凑表达式在预测精度、泛化能力和可解释性上全面超越文献中已有的经验拟合公式。

**[The Pareto Frontier of Resilient Jet Tagging](the_pareto_frontier_of_resilient_jet_tagging.md)**

:   系统评估LHC射流标记任务中多种架构（DNN/PFN/EFN/ParT）的AUC-鲁棒性权衡，揭示更复杂模型虽AUC更高但对蒙特卡洛模型依赖性更强，构建Pareto前沿并通过案例研究证明低鲁棒性分类器即使校准后仍在下游参数估计中产生偏差。

**[The Platonic Universe: Do Foundation Models See the Same Sky?](the_platonic_universe_do_foundation_models_see_the_same_sky.md)**

:   在天文学场景下验证柏拉图表征假说（PRH）：使用JWST、HSC、Legacy Survey和DESI光谱数据，测量6种基础模型（ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer）的表征对齐度，发现模态内和跨模态MKNN分数随模型规模一致增加（p=3.31×10⁻⁵），支持不同架构和模态向共享表征收敛的假说。

**[The Primacy of Magnitude in Low-Rank Adaptation](the_primacy_of_magnitude_in_low-rank_adaptation.md)**

:   揭示 LoRA 中权重更新幅度（magnitude）是性能的根本驱动因素，统一了学习率、缩放因子和初始化策略对 LoRA 的影响机制，并提出 LoRAM——一种基于确定性正交基和幅度缩放的高效初始化方法，无需 SVD 即可匹敌甚至超越谱初始化方法。

**[TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)**

:   提出TITAN框架，用深度学习模型预测VQE中的"冻结参数"（训练过程中始终不活跃的参数），在初始化阶段即冻结40-60%参数，实现最高3倍收敛加速和40-60%电路评估量减少，在30量子比特的分子系统上匹配或超越基线精度。

**[Toward Complete Merger Identification at Cosmic Noon with Deep Learning](toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)**

:   在 IllustrisTNG50 模拟生成的模拟 HST CANDELS 图像上训练 ResNet18，首次证明深度学习可以在高红移 $1<z<1.5$ 下成功识别包括小质量比合并（minor merger, $\mu \geq 1/10$）和低质量星系（$M_\star > 10^8 M_\odot$）在内的星系合并，总体准确率约 73%，并通过 Grad-CAM 和 UMAP 深入分析了模型行为。

**[Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)**

:   提出基于 adapter 的多物理场预训练框架，通过将 lifting/projection 层作为问题特定适配器、冻结共享的核积分算子层，实现跨 PDE 问题的迁移学习，显著降低微调成本并提升泛化能力。

**[Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)**

:   研究从标准宇宙学模型（ΛCDM）预训练的神经网络能否迁移到超越标准模型的场景（大质量中微子、修改引力、原初非高斯性），发现dummy node架构可将模拟需求降低一个数量级，但当参数存在强物理简并（如σ₈-Mν）时会出现负迁移。

**[Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)**

:   用变分自编码器(VAE)对 2743 条 JWST 高红移($z>4$)星系光谱进行无监督聚类，发现 12 个不同的天体物理类别，使已知的后星暴星系、Lyman-α 发射星系、极端发射线星系、Little Red Dots 等稀有种群数量翻倍。

**[Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)**

:   首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。
