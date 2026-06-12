---
title: >-
  ICLR2026 学习理论论文汇总 · 7篇论文解读
description: >-
  7篇ICLR2026的学习理论方向论文解读，涵盖目标跟踪、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "学习理论"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "对抗鲁棒"
item_list:
  - u: "an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p/"
    t: "An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem"
  - u: "deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding/"
    t: "Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding"
  - u: "function_spaces_without_kernels_learning_compact_hilbert_space_representations/"
    t: "Function Spaces Without Kernels: Learning Compact Hilbert Space Representations"
  - u: "lipschitz_bandits_with_stochastic_delayed_feedback/"
    t: "Lipschitz Bandits with Stochastic Delayed Feedback"
  - u: "scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w/"
    t: "Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees"
  - u: "the_expressive_limits_of_diagonal_ssms_for_state-tracking/"
    t: "The Expressive Limits of Diagonal SSMs for State-Tracking"
  - u: "the_price_of_robustness_stable_classifiers_need_overparameterization/"
    t: "The Price of Robustness: Stable Classifiers Need Overparameterization"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 学习理论

**🔬 ICLR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (17)](../../ICML2026/learning_theory/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/learning_theory/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/learning_theory/index.md) · [🧪 ICML2025 (16)](../../ICML2025/learning_theory/index.md)

**[An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)**

:   提出增量单元枚举算法（ICE），首个具有严格证明的独立算法，可以在 $O(N^{D+1})$ 时间内精确求解0-1损失线性分类问题的全局最优解，并扩展到多项式超曲面分类。

**[Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)**

:   提出 FlexQP——基于 $\ell_1$ 弹性松弛的"永远可行"凸二次规划（QP）求解器，结合深度展开（deep unfolding）学习 LSTM 反馈策略加速收敛得到 Deep FlexQP；在 SQP 框架中作为子模块，解非线性轨迹优化比 OSQP 快 4-16 倍，预测安全滤波器的安全违规减少 70%+、任务完成率提升 43%。

**[Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)**

:   证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。

**[Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)**

:   首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。

**[Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)**

:   提出 Random Wavelet Features (RWF)，通过从小波族中随机采样构建可扩展的非平稳核近似，保留随机特征的线性时间复杂度，同时具有正定性、无偏性和一致收敛保证。

**[The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)**

:   建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。

**[The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)**

:   建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。
