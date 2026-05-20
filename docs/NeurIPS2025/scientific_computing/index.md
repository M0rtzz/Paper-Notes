---
title: >-
  NeurIPS2025 科学计算方向23篇论文解读
description: >-
  23篇NeurIPS2025的科学计算方向论文解读，涵盖域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "科学计算"
  - "论文解读"
  - "论文笔记"
  - "域适应"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🧠 NeurIPS2025** · **23** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (19)](../../ICML2026/scientific_computing/index.md) · [📷 CVPR2026 (4)](../../CVPR2026/scientific_computing/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/scientific_computing/index.md) · [🤖 AAAI2026 (8)](../../AAAI2026/scientific_computing/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/scientific_computing/index.md) · [🧪 ICML2025 (8)](../../ICML2025/scientific_computing/index.md)

**[A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees](a_regularized_newton_method_for_nonconvex_optimization_with.md)**

:   提出一类基于当前与历史梯度构造的新型正则化器，结合带负曲率监测的共轭梯度法求解正则化Newton方程，在不需要Hessian Lipschitz常数先验知识的自适应框架下，首次同时实现了$O(\epsilon^{-3/2})$最优全局迭代复杂度和二次局部收敛速率。

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

**[F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)**

:   本文首次系统研究了科学机器学习中预训练大型算子模型(LOM)的参数高效微调(PEFT)，发现 LoRA 在傅里叶层中存在深度放大的近似误差下界，而 Adapter 保留了通用逼近能力；据此提出频率自适应 Adapter（F-Adapter），按频谱能量分配 Adapter 容量，在 3D Navier-Stokes 预测任务上仅调参不到 2% 即达到 SOTA。

**[From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)**

:   提出基于 Neural Operator 的「子网格黑洞」模型，学习小尺度 (GR)MHD 时间演化算子，替代手工闭合规则嵌入多层级直接数值模拟框架，首次实现吸积驱动反馈的内禀变异性捕获，加速比达 $\sim 10^5$ 倍。

**[From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows](from_images_to_physics_probabilistic_inference_of_galaxy_parameters_and_emission.md)**

:   提出 VAE–Normalizing Flow 混合框架，从 SDSS gri 图像和测光数据出发，以概率方式联合推断星系物理参数（恒星质量、SFR、红移、气相金属丰度、中心黑洞质量）和发射线流量（Hα、Hβ、[N II]、[O III]），速度比 SED 拟合快 100 倍以上且提供校准良好的后验分布。

**[GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)**

:   首次提出可扩展的5D神经网络代理模型 GyroSwin，将 Swin Transformer 扩展至5维回旋动力学相空间，通过交叉注意力实现3D↔5D交互、通道式模态分离捕获带状流，在等离子体湍流模拟中实现比传统准线性方法更高的精度，且比数值求解器（GKW）快3个数量级。

**[Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)**

:   基于 Riesz 表示定理，用可学习核积分（Integral Kernel Functional）近似无限维 Hamiltonian 泛函，通过自动微分获取泛函导数，实现保能量的神经 PDE 求解器（HNS），在 1D/2D PDE 上展现出优越的稳定性和泛化能力。

**[INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)**

:   提出间接神经校正器(INC)，将学习到的校正项嵌入PDE的右端项（而非直接修改状态），理论证明误差放大降低$\mathcal{O}(\Delta t^{-1}+L)$倍，在6个PDE系统（1D混沌到3D湍流）上大幅改善长期轨迹性能（R²提升达158.7%），实现最高330×加速。

**[Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)**

:   揭示了标准 BSDE 方法性能不如 PINNs 的根本原因是 Euler-Maruyama 积分引入的不可消除离散化偏差，提出基于 Stratonovich 形式的 Heun-BSDE 方法彻底消除该偏差，在高维 PDE 上与 PINNs 竞争。

**[Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)**

:   针对最优交易执行中 HJB 方程的硬零终端库存约束（$X_T=0$），提出 Multi-Trajectory PINN (MT-PINN)，通过基于轨迹展开的终端损失与 $\lambda$-curriculum 训练策略，在合成数据和 SPY 实盘回测中显著优于 vanilla PINN，终端库存违规率大幅降低。

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

**[Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)**

:   本文研究了两层过参数化 ReLU 网络中稳定极小值（flat minima）的泛化性质，证明虽然平坦性确实蕴含泛化，但其收敛速率随输入维度指数级恶化（即存在维度灾难），与不受维度灾难影响的低范数解（weight decay）形成指数级分离；并揭示了"neural shattering"现象作为高维失败的几何机制。

**[Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)**

:   利用符号回归（Symbolic Regression）从数值相对论模拟数据中自动发现双中子星并合后吸积盘质量的解析标定关系，所得紧凑表达式在预测精度、泛化能力和可解释性上全面超越文献中已有的经验拟合公式。

**[The Primacy of Magnitude in Low-Rank Adaptation](the_primacy_of_magnitude_in_low-rank_adaptation.md)**

:   揭示 LoRA 中权重更新幅度（magnitude）是性能的根本驱动因素，统一了学习率、缩放因子和初始化策略对 LoRA 的影响机制，并提出 LoRAM——一种基于确定性正交基和幅度缩放的高效初始化方法，无需 SVD 即可匹敌甚至超越谱初始化方法。

**[Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)**

:   提出基于 adapter 的多物理场预训练框架，通过将 lifting/projection 层作为问题特定适配器、冻结共享的核积分算子层，实现跨 PDE 问题的迁移学习，显著降低微调成本并提升泛化能力。
