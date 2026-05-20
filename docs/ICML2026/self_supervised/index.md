---
title: >-
  ICML2026 自监督/表示学习方向9篇论文解读
description: >-
  9篇ICML2026的自监督/表示学习方向论文解读，涵盖自监督学习、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🧪 ICML2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (30)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (14)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (13)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (32)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md) · [🧪 ICML2025 (24)](../../ICML2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×2

**[A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)**

:   本文改进了监督对比学习（在有限标注数据池中构造元组）的样本复杂度上界，通过两个不同的U-统计量估计器，在极值多类场景下实现从依赖最小类概率的界到仅依赖类别数或样本规模的界的突破。

**[Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)**

:   本文提出 SAGE，把"估计未标注数据分布"换成"在表征空间做结构推断"，用 simplex ETF 几何锚 + 高阶图传播 + 分布无关可靠性加权三件套，在极端标签稀缺且未标注分布任意的 UniSSL 设定下取得平均 8.52% 的准确率提升。

**[Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)**

:   作者证明对比学习里的"预定义数据增强 (旋转/裁剪/翻转)"等价于对 Positive-incentive Noise (π-noise) 的点估计, 然后把 π-noise 从"点估计"升级为可学习分布, 训练一个 π-noise 生成器在原图上加可学噪声当增强 (PiNDA), 使 SimCLR / BYOL / SimSiam / MoCo / DINO 在 vision 上稳定涨点, 且天然适配 HAR / Reuters / Epsilon 等无人工增强的非视觉数据。

**[How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)**

:   作者把一只"小白鼠视觉皮层的 SOTA 基础模型（FNN）"当成生理学实验对象，用解码流形 / 编码流形 / 解码轨迹三件套挨个分析它的 encoder / recurrent / readout，发现 FNN 的拟合精度主要靠 readout 那一堆同质 feature map 撑起来，而真正"像大脑"的只有 recurrent 模块；并用新提出的 tubularity 指标定量地说"早期编码层缺少生物级时间结构"，给未来神经基础模型给出"早期加 recurrence、readout 减少 feature 维度"的明确建议。

**[Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)**

:   作者证明:对比学习里典型的三元组任务,只要嵌入维度 $d$ 小于真维度 $D$ 的某个常数倍,无论用什么优化器,准确率都会"坍缩"到 1 维随机嵌入的 50% baseline,而且在算法层面这件事在 Unique Games 假设下也无法被多项式时间逼近。

**[Statistical Consistency and Generalization of Contrastive Representation Learning](statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)**

:   本文首次为对比表示学习 (CRL) 建立了"上游对比损失最小化等价于下游 AUC 型检索性能最优"的 Fisher / 统计一致性, 并给出依赖于正样本数 $n$ 和负样本数 $m$ 的精细泛化界 $O(1/m+1/\sqrt n)$ (监督) 与 $O(1/\sqrt m+1/\sqrt n)$ (自监督), 从而首次从理论上解释了 CLIP / SimCLR 使用上万负样本能持续涨点的现象。

**[Text-Conditional JEPA for Learning Semantically Rich Visual Representations](text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)**

:   本文提出 TC-JEPA，把 I-JEPA 的 mask 特征预测器额外条件化在图像 caption 上，通过多层稀疏跨注意力让 patch 表示在文本"提示"下变得可预测，从而在不用对比损失的前提下学到语义更丰富、对密集预测尤其友好的视觉表征。

**[The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)**

:   本文用测度论框架把 InfoNCE 损失提升到表示分布上的确定性"种群能量"，证明 unimodal 情形是凸的且收敛到唯一 Gibbs 平衡，而对称多模态情形会出现持续的负对称 KL 耦合，从几何上必然产生 modality gap。

**[Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)**

:   作者把对比 / 非对比 / 预测式 SSL 统一为"潜在分布匹配 (LDM)"：最大化样本在假设潜在模型下的对数概率 (alignment) + 最大化潜在熵 (uniformity)，并基于此推出带 Kalman 预测器的非线性可识别预测式 SSL。
