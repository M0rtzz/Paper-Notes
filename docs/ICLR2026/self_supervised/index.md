---
title: >-
  ICLR2026 自监督/表示学习论文汇总 · 16篇论文解读
description: >-
  16篇ICLR2026的自监督/表示学习方向论文解读，涵盖自监督学习、少样本学习、扩散模型、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "少样本学习"
  - "扩散模型"
  - "时序预测"
item_list:
  - u: "adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat/"
    t: "Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts"
  - u: "chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization/"
    t: "Chart Deep Research in LVLMs via Parallel Relative Policy Optimization"
  - u: "difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect/"
    t: "Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective"
  - u: "exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c/"
    t: "Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification"
  - u: "fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce/"
    t: "Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning"
  - u: "infonce_induces_gaussian_distribution/"
    t: "InfoNCE Induces Gaussian Distribution"
  - u: "maximizing_asynchronicity_in_event-based_neural_networks/"
    t: "Maximizing Asynchronicity in Event-based Neural Networks"
  - u: "maximizing_incremental_information_entropy_for_contrastive_learning/"
    t: "Maximizing Incremental Information Entropy for Contrastive Learning"
  - u: "no_other_representation_component_is_needed_diffusion_transformers_can_provide_r/"
    t: "No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves"
  - u: "ponderlm_pretraining_language_models_to_ponder_in_continuous_space/"
    t: "PonderLM: Pretraining Language Models to Ponder in Continuous Space"
  - u: "regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found/"
    t: "Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models"
  - u: "snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i/"
    t: "SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty"
  - u: "soft_equivariance_regularization_for_invariant_self-supervised_learning/"
    t: "Soft Equivariance Regularization for Invariant Self-Supervised Learning"
  - u: "temporal_slowness_in_central_vision_drives_semantic_object_learning/"
    t: "Temporal Slowness in Central Vision Drives Semantic Object Learning"
  - u: "test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting/"
    t: "Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting"
  - u: "why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic/"
    t: "Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning"
item_total: 16
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🔬 ICLR2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [📷 CVPR2026 (34)](../../CVPR2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×5

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)**

:   提出 PRPO（Parallel Relative Policy Optimization），通过在奖励维度和数据类型两个层面做并行解耦优化，解决 GRPO 在多维奖励信号干扰和异构数据梯度冲突下的训练瓶颈；同时构建 MCDR-Bench，基于"错误唯一性原则"将主观生成评估转化为客观错误识别，实现图表深度研究能力的量化评估。

**[Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)**

:   通过相似度图模型理论分析严格证明"困难样本"（跨类高相似度样本对）会损害无监督对比学习性能——困难样本使泛化误差界严格恶化，提出删除困难样本、调节 margin 和温度缩放三种理论指导的缓解策略，在 TinyImageNet 上带来高达 10.42% 的线性探测准确率提升。这一发现是反直觉的：深度学习中通常"更多数据更好"，但对比学习中精心移除困难样本反而有益。

**[Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification](exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c.md)**

:   发现病理基础模型特征具有低维流形几何结构（有效秩仅29.7/512维），而线性层会破坏这种结构导致少样本过拟合，提出即插即用的MR Block（冻结随机矩阵做几何锚+低秩残差路径做任务适配）在少样本WSI分类上达到SOTA。

**[Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)**

:   受果蝇嗅觉回路启发，提出 Fly-CL 框架，通过稀疏随机投影+top-k操作+流式岭分类三阶段渐进去相关，在预训练模型持续学习中大幅降低训练时间的同时达到SOTA水平。

**[InfoNCE Induces Gaussian Distribution](infonce_induces_gaussian_distribution.md)**

:   从理论上证明 InfoNCE 损失函数在两种互补机制下会诱导表征趋向高斯分布：经验理想化路线（对齐+球面均匀性→高斯）和正则化路线（消失正则项→各向同性高斯），并在合成数据和 CIFAR-10 上验证。

**[Maximizing Asynchronicity in Event-based Neural Networks](maximizing_asynchronicity_in_event-based_neural_networks.md)**

:   提出EVA框架，将事件类比为语言token，用基于RWKV-6的线性注意力异步编码器实现逐事件特征更新，结合多表示预测(MRP)+下一表示预测(NRP)的自监督学习获得可泛化特征，首次在异步-同步(A2S)范式中成功完成高难度目标检测任务(Gen1数据集0.477 mAP)。

**[Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)**

:   提出IE-CL（Incremental-Entropy Contrastive Learning）框架，通过显式优化增强视图间的熵增益（而非仅最大化互信息），将编码器视为信息瓶颈并联合优化可学习变换（生成熵）与编码器正则化器（保留熵），在小batch设置下一致提升CIFAR-10/100、STL-10和ImageNet上的对比学习性能，且核心模块可即插即用集成到现有框架。

**[No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](no_other_representation_component_is_needed_diffusion_transformers_can_provide_r.md)**

:   提出 Self-Representation Alignment (SRA)，发现扩散 Transformer 内部表征沿"层数增加 + 噪声降低"两个维度呈现从差到好的判别质量梯度，据此将学生网络早层高噪声表征对齐到 EMA 教师晚层低噪声表征，**完全不需要任何外部表征组件（DINOv2/CLIP/MAE）**，即可在 DiT 和 SiT 上大幅加速收敛并提升生成质量（SiT-XL/2 在 800 epoch 达到 FID 1.58，可比依赖 DINOv2 的 REPA）。

**[PonderLM: Pretraining Language Models to Ponder in Continuous Space](ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)**

:   提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。

**[Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)**

:   提出 Regularized Latent Dynamics Prediction (RLDP)，通过在自监督的潜空间下一状态预测目标上添加简单的正交正则化来维持特征多样性，在零样本 RL 中匹配甚至超越复杂的 SOTA 表示学习方法，特别是在低覆盖率场景下优势显著。

**[SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)**

:   SNAP-UQ 提出一种面向 TinyML 场景的单次前向传播不确定性估计方法：在骨干网络的选定层附加微型 int8 预测头，用自监督方式预测下一层的激活统计量，将实际激活与预测之间的偏差（"surprisal"）聚合为不确定性分数，无需额外前向传播、时序缓冲或集成，仅增加几十 KB 闪存即可在微控制器上实现可靠的分布偏移检测和故障检测。

**[Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)**

:   提出 SER（Soft Equivariance Regularization），通过在 ViT 中间层施加软等变正则化、在最终层保持不变性目标的层解耦设计，在不引入额外模块的情况下，为不变性 SSL 方法（MoCo-v3, DINO, Barlow Twins）带来一致的分类精度和鲁棒性提升。

**[Temporal Slowness in Central Vision Drives Semantic Object Learning](temporal_slowness_in_central_vision_drives_semantic_object_learning.md)**

:   通过模拟人类中央视觉（注视点裁剪）和时间慢性原则（时间对比学习），在 Ego4D 数据上训练 SSL 模型，发现两者组合能有效提升语义对象表征——中央视觉强化前景提取，时间慢性在注视凝视期间蒸馏语义信息。

**[Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)**

:   提出 Chroma——小型预训练时序模型组合（portfolio）框架：从通用模型通过后训练（post-training）产出频率/领域专家（训练加速 10×），测试时通过模型选择或贪心集成组合，4M 参数的 portfolio 在 Chronos Benchmark II 上匹配 205M-500M 参数的大型单体模型性能，同时推理计算远低于 test-time fine-tuning。

**[Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)**

:   诊断出原型自监督学习中部分原型坍缩的根因是编码器与原型的联合优化导致的快捷学习，提出全解耦训练策略——用在线 GMM 独立估计原型——彻底消除坍缩并提升下游性能。
