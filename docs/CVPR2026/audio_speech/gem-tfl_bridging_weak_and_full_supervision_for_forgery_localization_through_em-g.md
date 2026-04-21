---
title: >-
  [论文解读] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization
description: >-
  [CVPR 2026][语音][时序伪造定位] 提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。
tags:
  - CVPR 2026
  - 语音
  - 时序伪造定位
  - 弱监督
  - EM算法
  - 图扩散
  - 时序一致性
---

# GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization

**会议**: CVPR 2026  
**arXiv**: [2603.05095](https://arxiv.org/abs/2603.05095)  
**代码**: 无  
**领域**: 语音/音频  
**关键词**: 时序伪造定位, 弱监督, EM算法, 图扩散, 时序一致性

## 一句话总结

提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。

## 研究背景与动机

时序伪造定位 (TFL) 旨在精确定位视频/音频中的篡改片段。全监督方法需要逐帧标注，成本高昂。弱监督 TFL (WS-TFL) 仅用视频级二元标签训练，但面临：

1. 训练目标（分类）与推理目标（定位）不匹配
2. 二元标签监督信息过弱
3. top-k 聚合不可微导致梯度阻断
4. 提案独立生成导致碎片化

## 方法详解

### 整体框架

两阶段：分类阶段（MIL + LAD + TCR + GPR 生成伪提案）→ 定位阶段（回归分支在伪提案监督下训练，推理时仅用回归分支）。

### 关键设计

#### 1. 潜在属性分解 (LAD)

将二元标签分解为 $(m+1)$ 维潜在属性集：0 代表真实，$1...m$ 代表 $m$ 个可学习的伪造属性。通过 EM 优化：

- E-Step：计算后验 $P(c|x,y;\theta^{(t)})$——真实样本分配到类0，伪造样本按模型置信度分配到多个属性
- M-Step：最小化 $\mathcal{L}_{bin} + \lambda_1 \mathcal{L}_{nll} + \lambda_2 \mathcal{L}_{ent}$，更新参数+EMA更新属性先验

#### 2. 时序一致性精化 (TCR)

解决 top-k 聚合不可微问题。将帧级属性预测 $S_t$ 重新对齐到视频级属性先验 $q$，通过 KL-based Bregman 投影问题建模，用迭代比例缩放 (IPS) 求解。训练无关（training-free），交替投影到行/列约束空间直到收敛。

#### 3. 图扩散提案精化 (GPR)

构造无向图 $G=(V,E)$，节点为提案，边权结合时序相似度（DIoU）和语义相似度。通过迭代扩散传播置信度：$\omega^{(t+1)} = \beta \mathcal{T} \omega^{(t)} + (1-\beta) \omega^{(0)}$，闭式解 $\omega^* = (1-\beta)(I - \beta\mathcal{T})^{-1}\omega^{(0)}$。

### 损失函数

定位阶段：$\mathcal{L} = \mathcal{L}_{bce}(\hat{y},y) + \gamma \cdot \mathcal{L}_{main}(\hat{\mathcal{P}}, \mathcal{P})$，$\gamma$ 从 0.5 线性增长到 1.0。

## 实验关键数据

### LAV-DF 数据集

| 方法 | 监督 | Avg. mAP |
|------|------|----------|
| UMMAFormer | 全监督 | 96.8 |
| MFMS | 全监督 | 97.3 |
| MDP | 弱监督 | 60.0 |
| WMMT | 弱监督 | 73.3 |
| **GEM-TFL** | 弱监督 | **77.6** |

### AV-Deepfake1M 数据集

| 方法 | 监督 | Avg. mAP |
|------|------|----------|
| GEM-TFL vs 上一SOTA | 弱监督 | +8% 绝对提升 |

### 关键发现

- 两阶段设计有效弥合训练-推理鸿沟
- EM 分解使二元标签的监督信号增强
- TCR 的训练无关特性避免了梯度阻断

## 亮点与洞察

1. EM 将二元标签分解为多维属性——巧妙地从弱监督中挖掘更丰富语义
2. 图扩散替代 OIC 分数的硬编码外区域设置，减少人为偏差
3. TCR 的训练无关特性——后处理级别的精化不增加训练开销

## 局限与展望

1. 与全监督方法仍有约 20% mAP 差距
2. 潜在属性数量 m 需要手动设置
3. 图扩散中的 beta 等超参需要调优

## 相关工作与启发

- 相比 PseudoFormer：增加了 EM 属性分解和图扩散，提升了伪提案质量
- EM 潜在属性分解的思路可迁移到其他弱监督任务

## 评分

- 新颖性: ⭐⭐⭐⭐ EM分解+图扩散+TCR的组合有创新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集充分对比
- 写作质量: ⭐⭐⭐⭐ 框架图清晰
- 价值: ⭐⭐⭐⭐ 弥合弱/全监督差距的方向重要

<!-- RELATED:START -->

## 相关论文

- [DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization](../../AAAI2026/audio_speech/deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)
- [Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition](../../ICML2025/audio_speech/bridging_the_language_gap_synthetic_voice_diversity_via_latent_mixup_for_equitab.md)
- [Towards Open-Vocabulary Audio-Visual Event Localization](../../CVPR2025/audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)
- [Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](../../ICML2025/audio_speech/sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)

<!-- RELATED:END -->
