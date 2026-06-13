---
title: >-
  ICLR2026 物理/科学计算论文汇总 · 17篇论文解读
description: >-
  17篇ICLR2026的物理/科学计算方向论文解读，收录 Astral、Augmenting Representations wit、Contact Wasserstein Geodesics等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "astral_training_physics-informed_neural_networks_with_error_majorants/"
    t: "Astral: Training Physics-Informed Neural Networks with Error Majorants"
  - u: "augmenting_representations_with_scientific_papers/"
    t: "Augmenting Representations with Scientific Papers"
  - u: "contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges/"
    t: "Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges"
  - u: "deep_learning_for_subspace_regression/"
    t: "Deep Learning for Subspace Regression"
  - u: "dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes/"
    t: "DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs"
  - u: "drift-net_a_spectral--coupled_neural_operator_for_pdes_learning/"
    t: "DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning"
  - u: "empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r/"
    t: "Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery"
  - u: "feedback-driven_recurrent_quantum_neural_network_universality/"
    t: "Feedback-driven Recurrent Quantum Neural Network Universality"
  - u: "hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi/"
    t: "HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning"
  - u: "initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study/"
    t: "Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study"
  - u: "learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems/"
    t: "Learning-guided Kansa Collocation for Forward and Inverse PDE Problems"
  - u: "mosiv_multi-object_system_identification_from_videos/"
    t: "MOSIV: Multi-Object System Identification from Videos"
  - u: "one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd/"
    t: "One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers"
  - u: "policy_myopia_as_a_mechanism_of_gradual_disempowerment_in_post-agi_governance_ci/"
    t: "Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance"
  - u: "stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan/"
    t: "Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance"
  - u: "sublinear_time_quantum_algorithm_for_attention_approximation/"
    t: "Sublinear Time Quantum Algorithm for Attention Approximation"
  - u: "supervised_metric_regularization_through_alternating_optimization_for_multi-regi/"
    t: "Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**🔬 ICLR2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (26)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (8)](../../CVPR2026/physics/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (20)](../../ICML2025/physics/index.md)

**[Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)**

:   提出 Astral 损失函数（基于函数型后验误差上界/error majorant），替代传统 PiNN 中的残差损失来训练物理信息神经网络，实现训练过程中可靠的误差估计，并在扩散方程、Maxwell 方程等多种 PDE 上取得了更好或相当的精度。

**[Augmenting Representations with Scientific Papers](augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

**[Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)**

:   提出非守恒广义 Schrödinger 桥 (NCGSB)——基于接触哈密顿力学允许能量随时间变化，通过 Contact Wasserstein Geodesic (CWG) 将桥问题转化为有限维 Jacobi 度量上的测地线计算，用 ResNet 参数化实现近线性复杂度且支持引导生成，在流形导航、分子动力学、图像生成等任务上大幅超越迭代式 SB 求解器。

**[Deep Learning for Subspace Regression](deep_learning_for_subspace_regression.md)**

:   将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。

**[DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)**

:   基于Green函数理论，将叠加原理嵌入物理-神经混合架构，构建离散Green网络DGNet，在仅用数十条训练轨迹的条件下实现SOTA精度，并展现对未见源项的鲁棒零样本泛化。

**[DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)**

:   提出 DRIFT-Net 双分支神经算子，通过受控低频混合（谱分支）和局部细节保真（图像分支）的带宽融合（radial gating），解决窗口注意力中全局谱耦合不足导致的自回归漂移问题，在 Navier-Stokes 基准上误差降低 7%-54%。

**[Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)**

:   在硬约束递归物理信息架构（HRPINN）中系统评估vanilla KAN替代MLP作为残差分支的效果——通过3项互补研究×100随机种子发现KAN在单变量可分残差（Duffing的 $-0.3x^3$）上的表现具有竞争力，但在乘法耦合残差（Van der Pol的 $(1-x^2)v$）上系统性失败且超参数极度脆弱，标准MLP在几乎所有配置下稳定性远优。

**[Feedback-driven Recurrent Quantum Neural Network Universality](feedback-driven_recurrent_quantum_neural_network_universality.md)**

:   本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。

**[HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)**

:   提出 HyperKKL，用超网络（hypernetwork）编码外源输入信号并即时生成 KKL 观测器的变换映射参数，使非自治非线性系统的状态估计无需重新训练或在线梯度更新，在 Duffing、Van der Pol、Lorenz、Rössler 四个经典非线性系统上验证了方法的有效性和局限性。

**[Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)**

:   首次对样条KAN的初始化策略进行系统性研究，提出LeCun/Glorot启发的方差保持方案和可调幂律初始化族，在126K+模型实例的大规模实验中证明幂律初始化在函数拟合和PDE求解上全面超越基线，Glorot方案在大参数量模型上增益显著，NTK特征谱分析揭示了其背后的优化动力学机制。

**[Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)**

:   将基于径向基函数(RBF)的无网格Kansa方法从单变量线性PDE扩展到耦合多变量和非线性PDE场景，结合自调参技术和多种时间步进方案，并系统对比了与PINN、FNO等神经PDE求解器在正问题和反问题上的表现。

**[MOSIV: Multi-Object System Identification from Videos](mosiv_multi-object_system_identification_from_videos.md)**

:   提出MOSIV——首个从多视角视频进行多物体系统辨识的完整框架：(1) 物体感知的4D动态高斯重建每个物体的几何与运动 → (2) 高斯到连续体提升构建MPM仿真粒子 → (3) 可微MPM模拟器前向滚动+几何对齐目标(3D Chamfer + 2D轮廓)反传优化每个物体的连续材料参数($E, \nu, \mu$) → 在包含弹性/塑性/流体/沙粒四种材料的接触丰富合成基准上，PSNR 达30.51 vs OmniPhysGS 25.93，Chamfer距离降低9.4倍，建立多物体长期物理仿真新基准。

**[One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)**

:   论证神经 PDE 求解器在边界条件变化时学到的不是单一的解算子，而是由边界条件索引的算子族，并从学习理论角度形式化了 ERM 下边界分布偏移导致的不可辨识性问题。

**[Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance](policy_myopia_as_a_mechanism_of_gradual_disempowerment_in_post-agi_governance_ci.md)**

:   论证政策短视（policy myopia）不是注意力分配问题，而是一个制度性机制——通过显著性捕获、能力级联和价值锁死三个耦合的正反馈循环，在后AGI时代系统性地、不可逆地剥夺人类的治理参与能力，而标准的缓解措施只能延缓但无法阻止这一过程。

**[Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)**

:   提出 Stretch-and-Squeeze（SnS）算法，一个无梯度、模型无关的双目标优化框架，通过在不同处理层级"拉伸"表征同时"压缩"目标单元激活来系统性地探测视觉系统的不变性流形，揭示了标准与鲁棒 CNN 之间不变性可解释性的分层差异。

**[Sublinear Time Quantum Algorithm for Attention Approximation](sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。

**[Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)**

:   提出拓扑感知 PINN (TAPINN)，通过监督度量正则化（Triplet Loss）结构化潜空间 + 交替优化调度稳定训练，在 Duffing 振荡器多域问题上物理残差降低约 49%（0.082 vs 0.160），梯度方差降低 2.18×。
