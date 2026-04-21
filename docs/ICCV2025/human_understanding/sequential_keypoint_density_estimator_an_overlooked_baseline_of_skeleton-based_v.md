---
title: >-
  [论文解读] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection
description: >-
  [人体理解] > SeeKer 提出将骨架序列的联合密度在关键点级别进行自回归分解，通过预测后续关键点的条件高斯分布来检测异常人体行为，在 UBnormal 和 MSAD-HR 数据集上大幅超越现有方法。
tags:
  - 人体理解
---

# Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection

- **会议**: ICCV 2025
- **arXiv**: 2506.18368
- **代码**: https://github.com/adelic99/seeker
- **领域**: 人体理解
- **关键词**: 视频异常检测, 骨架序列, 自回归密度估计, 关键点级别, 高斯分布

## 一句话总结

> SeeKer 提出将骨架序列的联合密度在关键点级别进行自回归分解，通过预测后续关键点的条件高斯分布来检测异常人体行为，在 UBnormal 和 MSAD-HR 数据集上大幅超越现有方法。

## 研究背景与动机

视频异常检测是安全关键的计算机视觉任务，广泛应用于医疗监护、工作场所安全和公共监控等场景。基于骨架的方法因其低维性、匿名性和域不变性而具有独特优势。

然而，现有的骨架异常检测方法（如 STG-NF、MoCoDAD）存在以下问题：

**忽略骨架的组合性质**：将骨架作为整体图结构处理，使用图神经网络进行单体似然估计，忽略了骨架由关键点组成的组合特性

**缺乏因果建模**：未充分利用骨架序列在时间域中固有的因果关系

**无法处理关键点检测的不确定性**：异常分数无法考虑骨架关键点检测器的置信度差异，导致遮挡或检测错误时产生误判

作者观察到，人体异常行为通常反映为异常的身体姿态——即关键点出现在不太可能的位置。这启发了一种更自然的建模方式：在关键点级别进行密度估计。

## 方法详解

### 整体框架

SeeKer（Sequential Keypoint Density Estimator）的核心思想是：给定前序关键点，预测后续关键点位置的条件多元高斯分布。将此方法应用于整个骨架序列，即可得到关键点级别的自回归密度分解。

每个骨架 $X_t$ 是一个 $N \times D$ 矩阵（$N=18$ 个关键点，$D=2$ 维坐标）。骨架序列 $\mathbf{X}$ 的联合密度通过两层自回归分解：

**第一层——时间自回归**：

$$p_\theta(\mathbf{X}) = \prod_{t=1}^{T} p_\theta(X_t | \mathbf{X}_\Delta)$$

其中 $\mathbf{X}_\Delta$ 是过去 $\Delta$ 帧的骨架。

**第二层——关键点自回归**：

$$p_\theta(X_t | \mathbf{X}_\Delta) = \prod_{n=1}^{N} p_\theta(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

每个关键点 $X_{t,n}$ 的条件分布定义为多元高斯：

$$p_\theta(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta) := \mathcal{N}(X_{t,n} | \boldsymbol{\mu}_\theta, \Sigma_\theta)$$

### 损失函数

训练目标为最大化序列似然，简化后的损失函数为：

$$L(\theta; \mathcal{D}) \cong \sum_{\mathbf{X},t,n} (X_{t,n} - \boldsymbol{\mu}_\theta)^\top \Sigma_\theta^{-1} (X_{t,n} - \boldsymbol{\mu}_\theta) + \ln \det \Sigma_\theta$$

第一项为马氏距离，第二项为协方差行列式的对数，起正则化作用防止退化解。

### 复合异常分数

基础异常分数为各关键点负对数似然之和：

$$s'(X_t | \mathbf{X}_\Delta) = -\sum_n \ln p_{\theta_{\text{MLE}}}(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

考虑关键点检测器置信度后的加权异常分数：

$$s(X_t | \mathbf{X}_\Delta) = -\sum_n c_{t,n} \ln p_{\theta_{\text{MLE}}}(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

其中 $c_{t,n} \in [0,1]$ 是关键点检测置信度，将检测不确定性融入最终决策。多人场景取帧内最大异常分数。

### 自回归架构

采用因果掩码全连接网络，通过固定的分块三角掩码实现因果约束，允许同一关键点的坐标相互依赖。实验验证该简单架构优于 Transformer 和 RNN。

## 实验

### 主实验结果

| 数据集 | 方法 | AUROC | AP |
|--------|------|-------|-----|
| UBnormal (Full) | STG-NF | 71.8 | 62.7 |
| | MoCoDAD | 68.3 | - |
| | MULDE | 72.8 | - |
| | **SeeKer** | **77.9** | **80.3** |
| ShanghaiTech (Full) | STG-NF | 85.9 | 77.6 |
| | **SeeKer** | **85.5** | **80.0** |
| MSAD-HR | STG-NF | 55.7 | 56.5 |
| | **SeeKer** | **61.1** | **60.1** |

在 UBnormal 上 SeeKer AUROC 比 STG-NF 高 **6.1pp**，AP 高 **17.6pp**；在 MSAD-HR 上 AUROC 高 **5.4pp**。

### 消融实验

| 协方差类型 | UBnormal (Full) | UBnormal (HR) | ShanghaiTech (Full) | ShanghaiTech (HR) |
|-----------|----------------|---------------|--------------------|--------------------|
| 固定（单位矩阵） | 61.4 | 63.4 | 74.1 | 74.4 |
| 对角可学习 | 77.1 | 78.1 | 85.3 | 86.2 |
| 全可学习 | **77.9** | **78.9** | **85.5** | **86.9** |

学习协方差比固定协方差提升超 **10pp**，充分说明了学习协方差的重要性。

### 关键发现

1. **概念简洁但效果显著**：SeeKer 的核心思想——关键点级自回归高斯密度估计——虽然简单，但在多个基准上取得 SOTA
2. **可解释性强**：异常分数可分解到每个关键点的贡献，能定位触发异常检测的具体关节
3. **置信度加权有效**：利用关键点检测器的置信度加权，在遮挡和检测错误场景下更为鲁棒
4. **关键点排列不影响性能**：实验证实不同的关键点排列顺序不影响模型表达能力

## 亮点与洞察

- 将密度估计从"整体骨架"细化到"逐关键点"是关键创新，带来了可解释性和更好的异常检测粒度
- 置信度加权的异常分数设计优雅地整合了管道中所有组件的不确定性
- 掩码全连接模型优于更复杂的 Transformer，体现了任务适配的重要性
- 在原始 2D 空间操作使得预测可视化直观

## 局限性

- 依赖外部骨架检测器（AlphaPose）和跟踪器的质量，在拥挤场景（如 MSAD）中性能受限
- 仅使用 2D 骨架信息，无法建模深度方向的异常动作
- 建模假设条件高斯分布可能不够灵活以表示复杂的多模态关键点分布
- 时间窗口 $\Delta$ 固定，无法动态适应不同速度的动作

## 相关工作

- **骨架异常检测**: STG-NF（图归一化流）、MoCoDAD（扩散模型预测）、MULDE（能量模型+去噪分数匹配）
- **视频异常检测**: 基于深度特征、光流和多模态融合的方法
- **自回归密度估计**: 从 NLP/语音中的序列建模思想迁移到骨架序列

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐⭐ |
| 清晰度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 8.5/10 |

<!-- RELATED:START -->

## 相关论文

- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](../../CVPR2025/human_understanding/multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/human_understanding/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)

<!-- RELATED:END -->
