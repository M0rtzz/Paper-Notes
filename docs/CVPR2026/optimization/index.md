---
title: >-
  CVPR2026 优化/理论论文汇总 · 18篇论文解读
description: >-
  18篇CVPR2026的优化/理论方向论文解读，涵盖联邦学习、对抗鲁棒、自监督学习、压缩/编码、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "对抗鲁棒"
  - "自监督学习"
  - "压缩/编码"
  - "模型压缩"
item_list:
  - u: "ace-merging_data-free_model_merging_with_adaptive_covariance_estimation/"
    t: "ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation"
  - u: "bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea/"
    t: "BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning"
  - u: "beyond_euclidean_gossip_kl-barycentric_consensus_on_heterogeneous_and_imbalanced/"
    t: "Beyond Euclidean Gossip: KL-Barycentric Consensus on Heterogeneous and Imbalanced Images"
  - u: "blazefl_fast_and_deterministic_federated_learning_simulation/"
    t: "BlazeFL: Fast and Deterministic Federated Learning Simulation"
  - u: "conditional_factuality_controlled_llms_with_generalization_certificates_via_conf/"
    t: "Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling"
  - u: "dc-merge_improving_model_merging_with_directional_consistency/"
    t: "DC-Merge: Improving Model Merging with Directional Consistency"
  - u: "defending_unauthorized_model_merging_via_dual-stage_weight_protection/"
    t: "Defending Unauthorized Model Merging via Dual-Stage Weight Protection"
  - u: "dynamic_momentum_recalibration_in_online_gradient_learning/"
    t: "Dynamic Momentum Recalibration in Online Gradient Learning"
  - u: "enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p/"
    t: "Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning"
  - u: "fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_/"
    t: "Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift"
  - u: "label-free_cross-task_lora_merging_with_null-space_compression/"
    t: "Label-Free Cross-Task LoRA Merging with Null-Space Compression"
  - u: "model_merging_in_the_essential_subspace/"
    t: "Model Merging in the Essential Subspace"
  - u: "otprune_distribution-aligned_visual_token_pruning_via_optimal_transport/"
    t: "OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport"
  - u: "scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn/"
    t: "SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning"
  - u: "semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score/"
    t: "Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score"
  - u: "the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s/"
    t: "The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers"
  - u: "unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source/"
    t: "UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation"
  - u: "zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training/"
    t: "ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**📷 CVPR2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (64)](../../ICML2026/optimization/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md) · [🧪 ICML2025 (61)](../../ICML2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×4 · 对抗鲁棒 ×2

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)**

:   提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。

**[Beyond Euclidean Gossip: KL-Barycentric Consensus on Heterogeneous and Imbalanced Images](beyond_euclidean_gossip_kl-barycentric_consensus_on_heterogeneous_and_imbalanced.md)**

:   针对完全去中心化训练在 non-i.i.d. 数据和客户端规模不均衡下崩坏的问题，本文把"邻居间平均模型参数"这个 Euclidean gossip 操作，换成在指数族期望参数空间里做线性混合——它恰好等价于一次曲率感知的 KL 重心共识（自然梯度步），无需构造或求逆 Fisher 矩阵就把单轮复杂度从 $O(d^3)$ 降到 $O(d)$，并给出一个开销与 Adam 几乎相同的实现 KL-consensus Adam，在 CIFAR-100 上比 Euclidean 共识基线高出约 20% 准确率。

**[BlazeFL: Fast and Deterministic Federated Learning Simulation](blazefl_fast_and_deterministic_federated_learning_simulation.md)**

:   提出 BlazeFL，一个基于 Python free-threading 的轻量级单机联邦学习仿真框架，通过共享内存执行和客户端隔离 RNG 流实现最高 3.1× 加速与比特级可复现。

**[Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)**

:   提出 CFC（Conditional Factuality Control），一种后验保形框架，通过增广分位数回归学习特征条件化的接受阈值，为LLM/VLM采样输出提供条件覆盖率保证，在保持紧凑预测集的同时显著改善难题子群的可靠性。

**[DC-Merge: Improving Model Merging with Directional Consistency](dc-merge_improving_model_merging_with_directional_consistency.md)**

:   DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。

**[Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)**

:   提出 MergeGuard，一种主动式双阶段权重保护框架：Stage 1通过L2正则化分散任务关键权重，Stage 2注入结构化扰动破坏合并兼容性，在保持保护模型<1.5%性能损失的同时使合并模型精度下降高达90%。

**[Dynamic Momentum Recalibration in Online Gradient Learning](dynamic_momentum_recalibration_in_online_gradient_learning.md)**

:   从信号处理视角揭示固定动量系数在偏差-方差权衡上的固有缺陷，提出SGDF优化器，通过在线计算最优时变增益（基于最小均方误差原则）动态平衡梯度估计的噪声抑制和信号保持，在多种视觉任务上超越SGD动量和Adam变体。

**[Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)**

:   针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。

**[Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)**

:   提出 Fed-ADE 框架，通过 uncertainty dynamics estimation 和 representation dynamics estimation 两个轻量级分布漂移信号，为每个客户端在每个时间步自适应调整学习率，实现联邦部署后无监督适应。

**[Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)**

:   观察到LoRA微调过程中下投影矩阵A的零空间比率随训练下降且与性能强相关，据此提出NSC Merging，一种无标签、任务无关的LoRA合并方法，在20个异构视觉任务、6个NLI任务和VLM评估上达到SOTA。

**[Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)**

:   提出 ESM 框架，通过对参数更新引起的激活偏移做 PCA 构建"本质子空间"（而非直接对参数做 SVD），并用三级极化缩放增强关键参数、抑制噪声，在 ViT-B/32 的 20 任务合并中比 Iso-CTS 提升 3.2%（绝对准确率）。

**[OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)**

:   将视觉 token 裁剪建模为最优传输（OT）下的分布对齐问题，通过最小化完整与裁剪后 token 集合间的 2-Wasserstein 距离，以 Gaussian 代理 + log-det 子模目标 + 贪心 Cholesky 选择实现 training-free、$O(mk^2)$ 复杂度的高效裁剪，在 11 个多模态基准上取得 SOTA 精度-效率折中。

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 用一个零训练的视觉语言几何打分器，把每个样本压缩成表示性、多样性和负类边界接近度三个标量，再由服务器只聚合这些轻量统计量形成全局共识，指导各客户端先删语义异常样本、再删多数类冗余样本，从而在强非 IID 和长尾联邦场景下兼顾精度、鲁棒性和极低通信开销。

**[Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)**

:   提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

**[The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)**

:   将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。

**[UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)**

:   提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，在 SAM 的扰动步骤中用零阶梯度估计替代反向传播，将 SAM 的计算开销从 2 次反传减少为 1 次，首次让 SAM 在稀疏训练中变得实用，在 CIFAR-10/100 和 ImageNet-1K 上一致提升所有主流稀疏训练方法 0.38%-2.54%。
