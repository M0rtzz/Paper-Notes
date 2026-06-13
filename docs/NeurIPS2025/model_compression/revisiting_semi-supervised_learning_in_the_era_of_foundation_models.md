---
title: >-
  [论文解读] Revisiting Semi-Supervised Learning in the Era of Foundation Models
description: >-
  [NeurIPS 2025][模型压缩][半监督学习] 系统性研究发现传统 SSL 方法在 VFM 时代效益有限——仅用有标签数据的 PEFT 即可匹敌 SSL——由此提出 V-PET：集成多种 PEFT 方法和多种 VFM 的伪标签来实现简洁高效的半监督学习。 半监督学习（SSL）利用大量无标签数据配合少量标签数据提升模型…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "半监督学习"
  - "视觉基础模型"
  - "参数高效微调"
  - "伪标签集成"
  - "自训练"
---

# Revisiting Semi-Supervised Learning in the Era of Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2503.09707](https://arxiv.org/abs/2503.09707)  
**代码**: [https://github.com/OSU-MLB/SSL-Foundation-Models](https://github.com/OSU-MLB/SSL-Foundation-Models)  
**领域**: 模型压缩  
**关键词**: 半监督学习, 视觉基础模型, 参数高效微调, 伪标签集成, 自训练

## 一句话总结

系统性研究发现传统 SSL 方法在 VFM 时代效益有限——仅用有标签数据的 PEFT 即可匹敌 SSL——由此提出 V-PET：集成多种 PEFT 方法和多种 VFM 的伪标签来实现简洁高效的半监督学习。

## 研究背景与动机

半监督学习（SSL）利用大量无标签数据配合少量标签数据提升模型性能，是深度学习时代的重要范式（FixMatch、FlexMatch、SoftMatch 等）。但这些方法大多设计用于**从零训练**神经网络。如今视觉基础模型（VFM）如 CLIP 和 DINOv2 已成为现代视觉应用的核心，那么：

1. 现有 SSL 算法在使用 VFM 作为骨干时是否仍然有效？
2. 需要哪些调整来提升性能？
3. 能否利用 VFM 的力量设计更简单有效的 SSL 算法？

作者指出传统 SSL 评估基准（CIFAR-10/100、Food101）在 VFM 时代已**不够有挑战性**——冻结 VFM 的线性探测就能达到很高准确率，域覆盖也太窄。

## 方法详解

### 整体框架

V-PET 遵循四阶段流水线：(a) 用有标签数据通过多种 PEFT 方法微调多种 VFM；(b) 用微调模型为无标签数据生成伪标签；(c) 集成多个模型的伪标签；(d) 用集成伪标签自训练最终模型。核心思想是利用 VFM 和 PEFT 方法的**多样性和互补性**来获得高质量伪标签，从而避免复杂的伪标签选择策略。

### 关键设计

1. **基于 VTAB 的新 SSL 基准**: 从 VTAB 三个类别中各选 2 个数据集：Natural（DTD、SUN397）、Specialized（RESISC45、Retinopathy）、Structured（CLEVR-C、KITTI），选择冻结 VFM 表现不佳的任务，覆盖纹理识别、场景理解、遥感、医学影像、合成推理、自动驾驶 6 个领域，12 种 shot 配置组合。这些任务对冻结 VFM 具有实质性挑战，真正需要 SSL 来释放 VFM 潜力。

2. **无监督超参数调优协议**: 传统 SSL 超参调优存在数据泄露风险（使用有标签验证集调参）。提出整合 7 种无监督标准——5 个特征空间指标（AMI、ARI、V-Measure、FMI、BNM）和 2 个 logit 指标（RankMe、CHI）——对每种超参配置在无标签验证集上计算全部 7 个指标的排名，选择平均排名最低的配置。完全无需标签信息。

3. **Mean Labels 集成策略**: 不同 VFM 和 PEFT 方法虽然总体准确率相似，但对个体样本的预测差异很大（互补性），且输出分布的尺度不同（Mean Logits 和 Mean Probabilities 集成会被某些模型主导）。Mean Labels 将每个模型的预测先转为 one-hot 编码（统一尺度），再平均得到 soft 伪标签。这种集成方式极其简单，但有效消除了尺度不一致问题，且集成越多模型、伪标签质量越高。

### 损失函数 / 训练策略

- 自训练阶段使用**全部伪标签**（$\tau = 0$，无置信度阈值过滤），只需一轮自训练。
- 自训练时从原始预训练权重重新初始化，而非沿用 PEFT 微调后的权重。
- PEFT 方法包括 LoRA 和 AdaptFormer，VFM 包括 CLIP ViT-B/16 和 DINOv2 ViT-B/14。
- 使用 AdamW 优化器，batch size 32，训练 35 轮。
- V-PET 总时间开销仅约为其他 SSL 方法的 1.16 倍。

## 实验关键数据

### 主实验

| 方法 | 6 个数据集 12 配置平均 | 最频繁 rank-1 次数 | 说明 |
|------|----------------------|-------------------|------|
| V-PET (CLIP+DINOv2, LoRA+AdaptFormer) | **60.5-61.0%** | 最多 | 跨 VFM + PEFT 集成 |
| PET (单 VFM, 多 PEFT) | 59.3-59.7% | 次多 | 单 VFM 内集成 |
| Labeled-Only PEFT | 55.6-55.7% | — | 无 SSL |
| FixMatch | 53.7% | — | 传统 SSL |
| FlexMatch | 53.9-56.2% | — | 传统 SSL |
| SoftMatch | 56.3-59.7% | — | 传统 SSL |
| FineSSL | 51.6-53.9% | — | 近期 VFM SSL |

### 消融实验

| 配置 | 说明 | 观察 |
|------|------|------|
| 全量微调 vs PEFT | SSL with VFM | PEFT 一致优于全量微调 |
| PEFT Label-Only vs SSL | 有无无标签数据 | Label-Only PEFT 已匹敌 SSL |
| ST → PET → V-PET | 集成规模递增 | 伪标签质量递增，性能递增 |
| Mean Labels vs Mean Logits vs Mean Probs | 集成方式 | Mean Labels 最优（尺度不变） |

### 关键发现

- **震惊发现**：在公平比较下，仅用有标签数据的全量微调就能匹配甚至超越 SSL 方法——无标签数据在现有 SSL 框架下对 VFM 几乎无益。
- **PEFT 解释**：SSL 允许更新 VFM 所有参数可能反而损害其内建的泛化性（无标签数据的噪声监督信号导致退化），PEFT 通过限制更新范围保护了 VFM 的泛化能力。
- **多样性是关键**：不同 VFM + PEFT 的预测多样性是集成有效的根基——top 20% 高置信预测的 Venn 图显示重叠有限。
- V-PET 不在每个单独设定都是第一，但其 mean rank 最低、rank-1 频率最高，稳定性最好。

## 亮点与洞察

- **重要的 negative result**：系统性验证了传统 SSL 在 VFM 时代的失效，对社区有重要指导；不是简单否定，而是深入分析原因并给出替代方案。
- **极致简洁**：V-PET 无需复杂的一致性约束、数据增强策略或伪标签过滤，仅用标准 PEFT + 集成 + 自训练，概念清晰、实现简单。
- **无监督调参协议**：7 指标排名融合的调参方案解决了 SSL 长期的数据泄露问题，可独立使用。
- 揭示了 PEFT 方法和 VFM 之间的互补特性，为模型选择和集成提供了新视角。

## 局限与展望

- 主要关注分类任务，分割、检测等密集预测任务待验证。
- 需要训练多个 PEFT 模型再集成，尽管时间开销小但模型管理复杂度增加。
- 基准仅 6 个数据集、12 配置，规模尚可扩大。
- 未探索大规模 VFM（ViT-Large/Huge）或最新的 VFM（SigLIP、InternViT 等）。
- Mean Labels 集成假设模型间独立性，当模型高度相关时，集成增益可能递减。

## 相关工作与启发

- 与 FineSSL 的区别：后者仅使用 CLIP 视觉编码器并在简单数据集（CIFAR）上验证，本文扩展到多种 VFM 和更难的基准。
- 基于 VFM 多样性的集成思路呼应了 "Eyes Wide Shut"（Tong et al.）和 Cambrian 的发现——不同 VFM 的能力确实互补。
- 对实际少标签场景（医学影像、遥感等）的 SSL 工作流有直接指导价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 对 VFM 时代 SSL 的系统性重新审视，V-PET 方案虽简单但洞察深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ 12 设定 × 多 SSL/PEFT/VFM 组合，公平调参协议，分析极为细致
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑递进清晰，从观察到洞察到方案一气呵成
- **价值**: ⭐⭐⭐⭐⭐ 对 SSL 社区和 VFM 实践者的双重指导意义，baseline 级别的参考工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)
- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
