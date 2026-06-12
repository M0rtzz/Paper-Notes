---
title: >-
  ICML2026 学习理论论文汇总 · 17篇论文解读
description: >-
  17篇ICML2026的学习理论方向论文解读，涵盖对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "学习理论"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
item_list:
  - u: "a_perturbation_approach_to_unconstrained_linear_bandits/"
    t: "A Perturbation Approach to Unconstrained Linear Bandits"
  - u: "conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat/"
    t: "Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding"
  - u: "core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations/"
    t: "CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations"
  - u: "correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference/"
    t: "Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference"
  - u: "estimating_correlation_clustering_cost_in_node-arrival_stream/"
    t: "Estimating Correlation Clustering Cost in Node-Arrival Stream"
  - u: "expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif/"
    t: "Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift"
  - u: "matroid_algorithms_under_size-sensitive_independence_oracles/"
    t: "Matroid Algorithms Under Size-Sensitive Independence Oracles"
  - u: "mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t/"
    t: "MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation"
  - u: "multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne/"
    t: "Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety"
  - u: "on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective/"
    t: "On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective"
  - u: "optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_/"
    t: "Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification"
  - u: "parsimonious_learning-augmented_online_metric_matching/"
    t: "Parsimonious Learning-Augmented Online Metric Matching"
  - u: "provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi/"
    t: "Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function"
  - u: "realizable_bayes-consistency_for_general_metric_losses/"
    t: "Realizable Bayes-Consistency for General Metric Losses"
  - u: "semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain/"
    t: "Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain"
  - u: "simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation/"
    t: "Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering"
  - u: "towards_optimal_robustness_in_learning-augmented_paging/"
    t: "Towards Optimal Robustness in Learning-Augmented Paging"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 学习理论

**🧪 ICML2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (7)](../../ICLR2026/learning_theory/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/learning_theory/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/learning_theory/index.md) · [🧪 ICML2025 (16)](../../ICML2025/learning_theory/index.md)

🔥 **高频主题：** 对抗鲁棒 ×2

**[A Perturbation Approach to Unconstrained Linear Bandits](a_perturbation_approach_to_unconstrained_linear_bandits.md)**

:   本文重新审视 Abernethy 等人的扰动式 bandit linear optimization 思路，提出 PABLO 归约，把无约束线性 bandit 转成可调用任意 OLO 子程序的问题，并由此得到 comparator-adaptive 静态/动态 regret、高概率界以及若干下界讨论。

**[Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)**

:   本文提出条件核岭回归（Conditional KRR）框架，将一组非惩罚特征注入核方法中，通过残差核将其归约为标准 KRR，证明了归约代价为 $\mathcal{O}(1/\sqrt{N})$，并在硬阈值（top-k 本征函数）和软阈值（随机高斯特征）两种设定下验证了条件 KRR 优于标准 KRR 的充分条件。

**[CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)**

:   作者把多任务学习里"负迁移"的根因从"梯度冲突"重新归到"共享表征里语义和噪声纠缠"，提出 CORE-MTL：双流编码器把表征拆成语义 $\hat{Z}_s$ 和残差 $\hat{Z}_r$，用 CKA 独立性约束 + 反事实风格替换 + 反演渲染重构来落地"因果正交"，理论上给出比梯度平衡更紧的 OOD 上界，实验上在 NYUv2/Cityscapes 的 ID 与 GTA5→Cityscapes、Cityscapes-C 的 OOD 设定上同时压过 PCGrad/GradNorm/STCH/FairGrad 等十种 baseline。

**[Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)**

:   作者指出经典 Hoeffding Tree（HT）在数据流上分裂时使用的"固定样本量"集中不等式被它自己采用的"数据相关停止规则"破坏，于是用 testing-by-betting + Universal Portfolio 重写分裂判据，让单棵树和 Adaptive Random Forest 都能在任意停止时刻保持 Type-I 错误可控，同时在 12 个真实流上更准且树更小。

**[Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)**

:   本文研究「节点到达」数据流模型下相关聚类（correlation clustering）代价的近似估计问题：作者提出 C4Approx 算法，用 $O(n^{(3+\alpha)/4}\log n)$ 词的**亚线性**空间和常数遍数得到 $(O(1), n^{1-\alpha})$-近似，并配套两个匹配下界证明多遍与加性误差都不可避免；在真实数据上仅存 2% 节点即达 Pivot 同等效果。

**[Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)**

:   ECL 证明在协变量漂移下完整对齐输入分布 $P_s(X) = P_t(X)$ 并非校准的必要条件，只要"在每个置信度水平集上 $P(Y_k=1|X)$ 的条件期望两域一致"即可，并据此构造一个对 canonical / class-wise / top-label 三类校准都通用、可微、且 mini-batch 梯度无偏的损失 ECL。

**[Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)**

:   作者提出「查询代价随查询集合大小线性增长」的尺寸敏感拟阵 oracle 模型，证明在该模型下找基、估计秩、估计划分数的最优查询代价都是 $\tilde{\Theta}(n^2)$，并对有界周长 $c$ 的拟阵给出 $\mathcal{O}(n^{2-1/c}\log n)$ 的最大权基算法突破二次下界。

**[MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)**

:   论文为 test-time adaptation 提供了第一份"目标风险 ≤ 源经验风险 + KL 复杂度 + MMD 分布偏移项"的 PAC-Bayes 上界，并把 MMD-球解读为 Walley 意义下的 credal set，从而用"上下风险区间"自然分离 aleatoric 与 epistemic 不确定性，给出"何时应当 adapt、何时该 abstain"的可计算判据。

**[Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)**

:   本文提出一种以 $\|\theta_j-\beta\|_{\bm\Sigma_j}$（矩阵加权范数）为正则项的鲁棒多任务线性回归估计器，用一个相对的"平衡度常数" $B$ 取代了既往工作中刚硬的"每个任务二阶矩最小特征值 $\Omega(1)$"假设，在病态/低秩/带离群任务的高维场景下同时给出最小最大率（minimax）、自适应、和回退到独立任务学习（ITL）的安全保证。

**[On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)**

:   本文首次为测试时自适应（TTA）建立可学习性理论框架，用 $(\epsilon,\delta)$-Recovery Complexity 衡量分布漂移后模型把超额风险压到 $\epsilon$ 所需时间，并配合 $(\epsilon,\rho)$-TTA Learnability 把局部恢复推广到整条非平稳测试流，导出匹配阶的 minimax 上/下界，揭示了 TTA 的"适应速度—信息约束"内在权衡。

**[Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)**

:   在多项式逻辑斯蒂（MNL）bandit 的组合动作空间里首次给出**计算可行**的 G-optimal 实验设计——把 Frank–Wolfe 线性最大化谱写成 0–1 MILP 或多项式时间 Schur 补松弛——并据此造出第一个面向"线性效用 + 非均匀收益"的最佳组合识别算法，样本复杂度 $\tilde{\mathcal{O}}(d\log N / \Delta^2)$。

**[Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)**

:   本文回答了 Im et al. (2022) 留下的公开问题：把"按动作预测的"在线度量匹配带进"节俭预测"框架——预测被昂贵地按 $k$ 步一次发放——并通过 Follow-the-Prediction 框架 + 自动补齐"虚拟预测"的元算法，给出与已知下界基本匹配的确定性和随机性竞争比上界。

**[Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)**

:   本文用「实代数几何 + 一阶谓词逻辑量词消去」给多维超参数调参第一次给出可证明的 generalization bound，把过去只能处理一维标量超参的 Balcan 2025 框架推广到任意 $p$ 维、双层验证损失、近似内层优化等多种实际场景，并配出第一条匹配上界的下界。

**[Realizable Bayes-Consistency for General Metric Losses](realizable_bayes-consistency_for_general_metric_losses.md)**

:   本文对"在一般（可能无界）度量损失下，假设类 $\mathcal{H}$ 何时存在分布无关的强通用 Bayes 一致学习算法"这一开放问题在 realizable 情形下给出锐刻画——充分必要条件是 $\mathcal{H}$ 不包含一种新的"无界 gap Littlestone 树"组合障碍。

**[Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)**

:   作者把"从高斯噪声生成的合成域"当作半监督迁移学习里的替代源域，先证明这种"无语义但有判别结构"的噪声能给目标域带来可量化的泛化界改进，再用三损失的 Noise Adaptation Framework（NAF）联合优化两域风险与分布差异，使 CIFAR-10 上 4-shot ResNet-18 比 ERM 提升 12.35%。

**[Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)**

:   本文为有符号图上的"坏三角形覆盖"问题（Bad Triangle Transversal, BTT）给出两个仅需单次解 LP 的简洁 2-近似算法，证明在完全图上 BTT 与 Correlation Clustering、MinSTC、Cluster Deletion 同时具有 $\tfrac{2137}{2136}$ 的 NP-难逼近下界，并构造了一种新的 pivot 流程把任意可行 BTT 覆盖转化为最多 $\tfrac{3}{2}|F|$ 错误的聚类，从而把 BTT 与 CC 最优值的差距从 2 收紧到 $3/2$。

**[Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)**

:   本文为带预测的随机化在线调页提出统一的「相对预测预算」(RPB) 视角，并基于 OnlineMin 设计 RPB-OnOPT 框架，把可证的鲁棒竞争比从既有的 $2H_k+O(1)$ 一举推到信息论下界附近的 $H_k+O(1)$，同时保持 1-一致性。
