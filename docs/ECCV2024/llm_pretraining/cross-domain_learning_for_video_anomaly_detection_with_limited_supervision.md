---
title: >-
  [论文解读] Cross-Domain Learning for Video Anomaly Detection with Limited Supervision
description: >-
  [ECCV2024][Video Anomaly Detection] 提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。
tags:
  - ECCV2024
  - Video Anomaly Detection
  - Cross-Domain Learning
  - 伪标签
  - uncertainty estimation
  - Weakly-Supervised Learning
---

# Cross-Domain Learning for Video Anomaly Detection with Limited Supervision

**会议**: ECCV2024  
**arXiv**: [2408.05191](https://arxiv.org/abs/2408.05191)  
**代码**: 待确认  
**领域**: others  
**关键词**: Video Anomaly Detection, Cross-Domain Learning, Pseudo-Labeling, uncertainty estimation, Weakly-Supervised Learning

## 一句话总结

提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。

## 背景与动机

视频异常检测（VAD）旨在自动定位视频中的异常事件（如事故、爆炸等安全威胁），是视频监控的核心任务。现有方法分为无监督（仅用正常视频建模）和弱监督（利用视频级标签）两类。弱监督方法在单域内表现优异，但在跨域场景下性能大幅下降，原因有三：(1) 异常定义依赖上下文，简单迁移无法捕获跨域差异；(2) 异常事件稀少导致类别不平衡，跨域时更加严重；(3) 弱标注数据量有限，模型对open-set异常的学习能力受约束。

现有跨域 VAD 方法均基于无监督技术，缺乏对异常特征的显式建模，性能不足以满足实际需求。同时大量无标注视频唾手可得，如何将有限的弱监督数据与丰富的无标注数据结合，是一个重要且实际的研究方向。

## 核心问题

如何在仅有少量弱标注源域数据的条件下，利用大量无标注外部视频提升 VAD 模型的跨域与 open-set 泛化能力？关键挑战在于：外部数据的伪标签含噪声，直接使用会导致确认偏差（Confirmation Bias），需要一种自适应机制来量化并抑制伪标签的不确定性。

## 方法详解

### 整体框架

CDL 框架包含两个阶段的迭代训练：

1. **CDL Step 0**：在弱标注数据 $\mathcal{D}_l$ 上使用 Ranking Loss 分别训练主模型 $P_m$（基于 CLIP backbone）和辅助模型 $P_a$（基于 I3D backbone），然后对外部数据 $\mathcal{D}_u$ 生成 segment 级伪标签。
2. **CDL Step k (k>0)**：在 $\mathcal{D}_l \cup \mathcal{D}_u$ 上迭代训练。每个 CDL step 包含多个 epoch，每 epoch 重新计算不确定性正则化分数，用于自适应加权外部数据的损失。每个 CDL step 结束后重新生成伪标签，逐步提升伪标签质量。

### 双 Backbone 特征提取

- **CLIP backbone**（ViT-B/32）：提取帧级特征，通过双线性插值池化到固定 $n_s$ 个 segment
- **I3D backbone**：提取 segment 级 3D 卷积特征

两个 backbone 具有对比性归纳偏置（Transformer vs 3D CNN），为不确定性估计提供多样化预测。推理阶段仅使用 CLIP backbone。

### 预测偏差估计

将外部数据的预测偏差分解为两部分：模型预测与伪标签之间的差异（可优化项）和伪标签与真实标签的差异（视为常数）。使用 BCE 损失估计每个 segment 的预测偏差。

### 高维空间不确定性估计

传统方法在概率空间（二分类后验概率）计算预测方差，但 VAD 是二分类任务，概率分布支撑有限。本文提出在高维特征空间量化不确定性：

- 取 $P_m$ 和 $P_a$ 倒数第二层的 segment 表示 $Z_m$ 和 $Z_a$
- 计算 segment 间的余弦相似度
- 通过指数变换得到不确定性正则化分数：$s^j = e^{\tau(\langle z_m^j, z_a^j \rangle - 1)}$

分数越高（两模型编码一致）表示伪标签越可靠；分数越低表示不确定性越大。

### 训练目标

外部数据损失为不确定性加权的 BCE 损失加上余弦相似度正则项：

$$\mathcal{L}_{\text{ext}} = \mathbb{E}[S \cdot \mathcal{L}_{\text{bce}} - \lambda_3 \cdot \langle Z_m, Z_a \rangle]$$

总损失为 Ranking Loss（标注数据）加上外部数据损失：$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{rank}} + \lambda_4 \cdot \mathcal{L}_{\text{ext}}$

### 推理

使用 $P_m$ 计算 segment 级异常分数，按帧数均匀映射到帧级。

## 实验关键数据

### 跨域实验

**UCF-Crime 为源域，XD-Violence 为跨域目标**（Table 2）：

| 方法 | 特征 | XDV AP(%) |
|---|---|---|
| zxVAD（无监督 SOTA） | - | 40.68 |
| SSRL（弱监督域内 SOTA） | I3D | 51.60 |
| CDL (UCF+HACS) | CLIP | **65.14** |
| CDL (UCF+XDV) | CLIP | **68.37** |

在 XDV 上比无监督跨域 SOTA 提升 **+27.69%**，比弱监督域内最优方法提升 **+16.77%**。

**XD-Violence 为源域，UCF-Crime 为跨域目标**（Table 3）：

| 方法 | UCF-R AUC(%) |
|---|---|
| zxVAD（无监督 SOTA） | 63.61 |
| CDL (XDV+HACS) | **88.50** |

在 UCF-R 上比无监督跨域 SOTA 绝对提升 **+24.89%**。

### Open-Set 实验（Table 4）

仅使用 1 类异常训练时，CDL 在 UCF-R 上达到 85.39% AUC（无 CDL 为 84.32%），始终优于所有 baseline。

### 消融实验（Table 5，open-set c=1）

| 外部数据 | 不确定性加权 | 余弦损失 | UCF-R AUC(%) |
|---|---|---|---|
| ✗ | ✗ | ✗ | 84.32 |
| ✓ | ✗ | ✗ | 84.67 |
| ✓ | ✓ | ✗ | 84.80 |
| ✓ | ✓ | ✓ | **85.39** |

每个组件均有正向贡献，余弦相似度损失带来最大增益（+0.59%）。

## 亮点

- **实用场景建模**：弱监督+无标注外部数据的组合非常贴合实际部署需求，比纯无监督跨域方法实用得多
- **高维不确定性估计**：没有在概率空间（二分类支撑有限）而是在特征空间计算双模型预测差异，更鲁棒
- **自适应伪标签精炼**：不确定性分数作为自动阈值动态加权损失，避免手动调参，且与伪标签质量有强负相关性
- **测试集重标注**：发现 UCF-Crime 测试集标注噪声严重（异常帧占比从 7.58% 提升到 16.55%），提供了更准确的 UCF-R 标注

## 局限与展望

- 依赖 I3D 和 CLIP 两个 backbone，训练开销较大（推理仅用 CLIP）
- 在 XDV 作为源域时，XDV 上的域内性能（78.61%）低于域内 SOTA（80.67%），可能因架构简单导致域内欠拟合
- 迭代伪标签精炼需要 40 个 CDL step，训练效率有待提升
- 仅在定义相近的大规模数据集（UCF-Crime/XD-Violence）间验证，未涉及异常定义差异大的小规模数据集（如 ShanghaiTech）
- 外部数据量的选择（11,000 视频）通过消融确定，缺少理论指导

## 与相关工作的对比

| 维度 | 本文 CDL | zxVAD | RTFM / S3R 等 |
|---|---|---|---|
| 监督方式 | 弱监督 | 无监督 | 弱监督 |
| 外部数据 | ✓（无标注） | ✓（生成伪异常帧） | ✗ |
| 跨域能力 | 强（显著提升） | 中等 | 弱（性能大幅下降） |
| 不确定性建模 | 高维特征空间 | 无 | 无 |
| 推理 backbone | CLIP | 自有 | I3D |

## 启发与关联

- 双模型预测差异量化不确定性的思路可推广到其他弱监督/半监督视频理解任务
- 高维特征空间的不确定性估计比概率空间更适用于二分类或少类任务，值得在其他低类别数任务中探索
- 测试集重标注的工作提醒我们：基准数据集的标注质量直接影响方法评估的公正性

## 评分
- 新颖性: ⭐⭐⭐⭐ — 弱监督跨域+不确定性驱动伪标签是新颖组合，高维不确定性估计有创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 跨域/open-set/消融/相关性分析/CDF演变等全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 实用性强，跨域提升显著，但场景覆盖面有限

<!-- RELATED:START -->

## 相关论文

- [Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](../../NeurIPS2025/llm_pretraining/learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)
- [Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)
- [DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)
- [Learning to Obstruct Few-Shot Image Classification over Restricted Classes](learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)

<!-- RELATED:END -->
