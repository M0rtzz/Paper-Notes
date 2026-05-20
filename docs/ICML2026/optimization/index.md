---
title: >-
  ICML2026 优化/理论方向15篇论文解读
description: >-
  15篇ICML2026的优化/理论方向论文解读，涵盖图神经网络、压缩/编码、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "图神经网络"
  - "压缩/编码"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🧪 ICML2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (44)](../../ICLR2026/optimization/index.md) · [🤖 AAAI2026 (22)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (114)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md) · [🧪 ICML2025 (56)](../../ICML2025/optimization/index.md)

**[Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](adaptive_estimation_and_inference_in_semi-parametric_heterogeneous_clustered_mul.md)**

:   本文桥接双重机器学习与聚类多任务学习，提出自适应框架结合 Neyman 正交性与数据驱动的配对融合罚项，在异质（可能无限维）噪声的半参数设置中精确恢复任务潜在聚类、以汇总率达到预言水平，并建立渐近正态性，实现有效统计推断。

**[Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)**

:   首次给"预算受限 + 私有成本"的子模社会福利最大化采购拍卖给出有近似比保证的真值机制 BFM-SWM——用几何递增阈值的降序时钟拍卖 + 单点保护 + 价/付率参数 $\beta$ 实现非负盈余 + 预算可行，一般子模函数 0.0328-近似、单调子模 0.0877-近似；副产品 BFM-VM 把估值最大化的确定性最佳近似比从 1/64 提升到 $1/(12+4\sqrt{3})\approx 0.0528$，并将运行时间从 $\mathcal{O}(n^2\log n)$ 降到 $\mathcal{O}(n\log n)$。

**[FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)**

:   本文提出 FAB——首个面向时变有向图分布式双层优化的纯一阶算法，将 AB/Push-Pull 通信与值函数惩罚法相结合，给出非渐近 $\mathcal{O}(K^{-2/3})$ 收敛率，并顺带解决了 AB/Push-Pull 在时变有向图非凸场景下收敛率长期悬而未决的开放问题。

**[Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)**

:   训练一个轻量网络预测线性指派问题 (LAP) 的对偶变量 $\hat{u}$，用 Min-Trick 构造可行对偶 $\hat{v}$，将其作为 LAPJV 精确求解器的暖启动，从而在保持最优性的同时把 $N=16{,}384$ 规模实例端到端加速 $2\times$ 以上。

**[Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)**

:   本文用 empirical NTK 作为统一视角，证明 zeroth-order SGD 引出的 eNTK 等价于把 first-order eNTK 投影到由微扰张成的随机子空间，从而通过 Johnson-Lindenstrauss 引理解释为何 ZO 方法在十亿参数 LLM 上仍然 work：误差只取决于输出维度 $V$ 和微扰数 $P$，与模型维度 $d$ 无关。

**[Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)**

:   本文为 Uniform Facility Location 设计了一个把经典近似算法 SimpleUniformFL 神经化的 MPNN，**既可用无监督期望成本损失端到端训练，也具备 $\mathcal{O}(\log n)$（递归版还能到 $\mathcal{O}(1)$）的可证明近似界**，实验上既打过 SimpleUniformFL 经典算法、也逼近 ILP 最优。

**[On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)**

:   本文首次在不假设 adapter 矩阵有界、不要求重参数化损失 Lipschitz 平滑的前提下，证明了原始 LoRA 梯度下降的最小梯度范数以 $O(1/\log T)$ 速率收敛（若参数范数有界则恢复经典 $O(1/T)$），并据此设计了与理论严格对应的自适应/归一化学习率，在 logistic regression、ResNet-18、TinyLlama 上验证了训练加速与稳定性提升。

**[Probing Neural TSP Representations for Prescriptive Decision Support](probing_neural_tsp_representations_for_prescriptive_decision_support.md)**

:   作者把训练好的 TSP 神经求解器视作"可迁移编码器",用冻结表征 + 轻量探针预测两类昂贵的运筹敏感性查询(节点移除与边禁用),系统证明探针准确率随求解器质量单调提升,可以与传统启发式集成达到 SOTA。

**[RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)**

:   本文提出 RL-SPH —— 一种不依赖外部 ILP 求解器、能独立产出 100% 可行解的端到端强化学习启发式算法，用「可行性奖励 + 双阶段策略 + 可行性感知邻域搜索」让 Graph Transformer Agent 在包含非二元整数变量的 ILP 上把 primal gap 平均降低 28.6 倍。

**[RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)**

:   本文基于 Transformer 层级 Hessian 的「行块对角占优」结构，把 Muon 优化器里昂贵的 Newton-Schulz 正交化换成一次行级 $\ell_2$ 归一化，将每步预条件复杂度从 $\mathcal{O}(mn\min(m,n))$ 降到 $\mathcal{O}(mn)$，在 GPT-2 / LLaMA 预训练上 wall-clock 提速 13–44×、ppl 不降反略升。

**[Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)**

:   Stream-SW 是首个能在"样本流"上估计 sliced Wasserstein 距离的算法：每个一维投影上用 KLL/quantile sketch 维护近似分位函数，把 1D Wasserstein 的闭式积分变成可流式更新的估计量，空间复杂度对样本数仅对数级，从而把 SOT 带入 IoT / 边缘设备等"看一次就丢掉"的场景。

**[Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)**

:   SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里，使离线黑盒优化在 Design-Bench 和 LLM 数据混合任务上稳定达到 SOTA。

**[Test time training enhances in-context learning of nonlinear functions](test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)**

:   本文给单层 softmax-attention transformer + LoRA 测试时微调的组合建立了首个严格泛化界，证明在 single-index 多项式任务上 TTT 把 ICL 的样本复杂度从 $r^{\Theta(\mathrm{ie}(\sigma_*))}$ 压到 $r^{\Theta(\mathrm{ge}(\sigma_*))}$ 并允许 link 函数逐任务变化、推理误差可随上下文长度 $\to$ 噪声水平。

**[Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)**

:   本文提出 T-LVMOGP：把多输出 GP 的核心建模问题——跨输出协方差 $k_{p,p'}(x, x')$ 的构造——转化成"在 Lipschitz 正则的 RCNN 嵌入空间里用单个标量基核做内积"，并完整嵌入 SVGP 框架，使 MOGP 第一次能可扩展且高表达力地处理 $P > 10000$ 输出（含 ZINB 似然的空间转录组数据），同时全面胜过 SV-LMC / OILMM / GS-LVMOGP 等基线。

**[Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)**

:   本文把"陈旧的"块循环坐标下降梯度估计存进 FIFO buffer，配上 momentum 衰减重用，并证明这等价于带 warm-start 的 BCCD；同时给出反直觉结论——更大的有限差分步长 $\epsilon$ 会隐式平滑 loss landscape、降低有效 Lipschitz 常数，从而让 stale gradient 反而能换来稳定下降。
