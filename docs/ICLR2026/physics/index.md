---
title: >-
  ICLR2026 物理/科学计算方向12篇论文解读
description: >-
  12篇ICLR2026的物理/科学计算方向论文解读，收录 Astral、Augmenting Representations wit、Deep Learning for Subspace Reg等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**🔬 ICLR2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/physics/index.md) · [📷 CVPR2026 (5)](../../CVPR2026/physics/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (41)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (14)](../../ICML2025/physics/index.md)

**[Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)**

:   提出 Astral 损失函数（基于函数型后验误差上界/error majorant），替代传统 PiNN 中的残差损失来训练物理信息神经网络，实现训练过程中可靠的误差估计，并在扩散方程、Maxwell 方程等多种 PDE 上取得了更好或相当的精度。

**[Augmenting Representations with Scientific Papers](augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

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

**[Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)**

:   将基于径向基函数(RBF)的无网格Kansa方法从单变量线性PDE扩展到耦合多变量和非线性PDE场景，结合自调参技术和多种时间步进方案，并系统对比了与PINN、FNO等神经PDE求解器在正问题和反问题上的表现。

**[One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)**

:   论证神经 PDE 求解器在边界条件变化时学到的不是单一的解算子，而是由边界条件索引的算子族，并从学习理论角度形式化了 ERM 下边界分布偏移导致的不可辨识性问题。

**[Sublinear Time Quantum Algorithm for Attention Approximation](sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。

**[Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)**

:   提出拓扑感知 PINN (TAPINN)，通过监督度量正则化（Triplet Loss）结构化潜空间 + 交替优化调度稳定训练，在 Duffing 振荡器多域问题上物理残差降低约 49%（0.082 vs 0.160），梯度方差降低 2.18×。
