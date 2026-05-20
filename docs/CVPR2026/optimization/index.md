---
title: >-
  CVPR2026 优化/理论方向8篇论文解读
description: >-
  8篇CVPR2026的优化/理论方向论文解读，涵盖联邦学习、对抗鲁棒、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "对抗鲁棒"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**📷 CVPR2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (15)](../../ICML2026/optimization/index.md) · [🔬 ICLR2026 (44)](../../ICLR2026/optimization/index.md) · [🤖 AAAI2026 (22)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (114)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md) · [🧪 ICML2025 (56)](../../ICML2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×4 · 对抗鲁棒 ×2

**[BlazeFL: Fast and Deterministic Federated Learning Simulation](blazefl_fast_and_deterministic_federated_learning_simulation.md)**

:   提出 BlazeFL，一个基于 Python free-threading 的轻量级单机联邦学习仿真框架，通过共享内存执行和客户端隔离 RNG 流实现最高 3.1× 加速与比特级可复现。

**[Dynamic Momentum Recalibration in Online Gradient Learning](dynamic_momentum_recalibration_in_online_gradient_learning.md)**

:   从信号处理视角揭示固定动量系数在偏差-方差权衡上的固有缺陷，提出SGDF优化器，通过在线计算最优时变增益（基于最小均方误差原则）动态平衡梯度估计的噪声抑制和信号保持，在多种视觉任务上超越SGD动量和Adam变体。

**[Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)**

:   针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。

**[Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)**

:   提出 Fed-ADE 框架，通过 uncertainty dynamics estimation 和 representation dynamics estimation 两个轻量级分布漂移信号，为每个客户端在每个时间步自适应调整学习率，实现联邦部署后无监督适应。

**[OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)**

:   将视觉 token 裁剪建模为最优传输（OT）下的分布对齐问题，通过最小化完整与裁剪后 token 集合间的 2-Wasserstein 距离，以 Gaussian 代理 + log-det 子模目标 + 贪心 Cholesky 选择实现 training-free、$O(mk^2)$ 复杂度的高效裁剪，在 11 个多模态基准上取得 SOTA 精度-效率折中。

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 用一个零训练的视觉语言几何打分器，把每个样本压缩成表示性、多样性和负类边界接近度三个标量，再由服务器只聚合这些轻量统计量形成全局共识，指导各客户端先删语义异常样本、再删多数类冗余样本，从而在强非 IID 和长尾联邦场景下兼顾精度、鲁棒性和极低通信开销。

**[The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)**

:   将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。

**[UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)**

:   提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。
