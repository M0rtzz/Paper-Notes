---
title: >-
  [论文解读] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment
description: >-
  [NeurIPS 2025][时间序列][时间序列] 提出 Time-o1，通过将标签序列变换为去相关且按重要性排序的主成分，解决时间序列预测中 TMSE 损失的自相关偏差和任务过载问题，实现与多种预测模型兼容的 SOTA 性能。
tags:
  - NeurIPS 2025
  - 时间序列
  - 学习目标
  - 标签自相关
  - SVD变换
  - 去相关
---

# Time-O1: Time-Series Forecasting Needs Transformed Label Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2505.17847](https://arxiv.org/abs/2505.17847)  
**代码**: [有](https://github.com/Master-PLC/Time-o1)  
**领域**: 时间序列预测  
**关键词**: 时间序列, 学习目标, 标签自相关, SVD变换, 去相关

## 一句话总结

提出 Time-o1，通过将标签序列变换为去相关且按重要性排序的主成分，解决时间序列预测中 TMSE 损失的自相关偏差和任务过载问题，实现与多种预测模型兼容的 SOTA 性能。

## 研究背景与动机

时间序列预测模型的训练通常使用时序均方误差（TMSE）作为学习目标，即逐步计算预测与标签序列的差异。然而 TMSE 存在两个根本性缺陷：

**缺陷一：标签自相关导致偏差。** 时间序列天然具有自相关性（相邻步骤高度相关），而 TMSE 将每步视为独立任务，忽略了步间相关性。根据 Theorem 3.1，TMSE 与标签序列真实似然之间的偏差为：

$$\text{Bias} = \|Y - \hat{Y}\|_{\Sigma^{-1}}^2 - \|Y - \hat{Y}\|^2 - \frac{1}{2}\log|\Sigma|$$

当标签步间去相关时偏差才消失。

**缺陷二：预测步数增加导致优化困难。** 长期预测时，预测步数 T 可达 720，TMSE 将每步视为独立任务，而多任务学习在任务过多时梯度冲突加剧，收敛困难。

先前工作 FreDF 提出在频域对齐来解决偏差，但频域成分仅在 $T \to \infty$ 时才完全去相关，有限预测步长下仍有残余相关。且频域变换不减少任务数量，优化困难未解决。

## 方法详解

### 整体框架

Time-o1 的核心思路是：将标签序列通过最优投影矩阵变换为去相关且按重要性排序的主成分，然后仅对齐最重要的 K 个成分进行训练。最终损失为变换域损失与 TMSE 的加权融合。

### 关键设计

1. **最优投影矩阵求解**: 对归一化后的标签矩阵 $\mathbf{Y} \in \mathbb{R}^{m \times T}$，通过约束优化求解投影矩阵 $\mathbf{P}^*$：逐个求最大化成分方差的投影方向，并要求相互正交。公式为 $\mathbf{P}_p^* = \arg\max_{\mathbf{P}_p} (\mathbf{Y}\mathbf{P}_p)^\top(\mathbf{Y}\mathbf{P}_p)$，约束 $\|\mathbf{P}_p\|^2 = 1$ 和 $\mathbf{P}_p^\top \mathbf{P}_j = 0$。

    由 Lemma 3.3，$\mathbf{P}^*$ 可通过 SVD 高效计算：$\mathbf{Y} = \mathbf{U}\mathbf{\Lambda}(\mathbf{P}^*)^\top$。变换后的成分 $\mathbf{Z} = \mathbf{Y}\mathbf{P}^*$ 满足去相关性（Lemma 3.2），且按方差从大到小排列。

2. **显著成分选择与任务削减**: 保留前 $K = \text{round}(\gamma \cdot T)$ 个最重要成分，$\gamma$ 控制保留比例。变换域损失仅对齐这 K 个成分：$\mathcal{L}_{\text{trans},\gamma} = \|\hat{\mathbf{Z}}_{\cdot,1:K} - \mathbf{Z}_{\cdot,1:K}\|_1$。使用 L1 范数而非 L2，因为不同成分方差差异极大，L2 会导致不稳定。

3. **融合损失**: 最终学习目标为变换域损失与原始 TMSE 的加权融合：$\mathcal{L}_{\alpha,\gamma} = \alpha \cdot \mathcal{L}_{\text{trans},\gamma} + (1-\alpha) \cdot \mathcal{L}_{\text{tmse}}$。$\alpha$ 控制两者的相对权重。

### 损失函数 / 训练策略

Time-o1 是模型无关的——可直接替换任意预测模型的训练损失。实现流程简洁：标准化标签 → SVD 计算主成分 → 投影预测和标签 → 计算融合损失。仅需调整 $\alpha$ 和 $\gamma$ 两个超参数。

## 实验关键数据

### 主实验（长期预测，8个数据集）

| 模型 | ETTm1 MSE | ETTm2 MSE | ETTh1 MSE | ECL MSE | Weather MSE |
|------|-----------|-----------|-----------|---------|-------------|
| **Time-o1** | **0.380** | **0.272** | **0.431** | **0.170** | **0.241** |
| Fredformer | 0.387 | 0.280 | 0.447 | 0.191 | 0.261 |
| iTransformer | 0.411 | 0.295 | 0.452 | 0.179 | 0.269 |
| DLinear | 0.403 | 0.342 | 0.456 | 0.212 | 0.265 |
| TimesNet | 0.438 | 0.302 | 0.472 | 0.212 | 0.271 |

### 消融实验（与其他学习目标对比）

| 损失函数 | ETTm1 MSE | ETTh1 MSE | Weather MSE | 说明 |
|---------|-----------|-----------|-------------|------|
| **Time-o1** | **0.379** | **0.431** | **—** | 变换+TMSE融合 |
| FreDF | 0.384 | 0.447 | — | 频域对齐 |
| Koopman | 0.389 | 0.452 | — | Koopman算子 |
| Dilate | 0.389 | — | — | 形状对齐 |
| DF（TMSE） | 0.387 | — | — | 基线TMSE |

### 关键发现

- Time-o1 在所有8个长期预测数据集上均提升基线模型性能
- 在 ETTh1 上将 Fredformer 的 MSE 从 0.447 降至 0.431（下降 3.6%）
- 仅修改学习目标即可获得与架构创新相当甚至更好的提升
- 相比 FreDF，Time-o1 的去相关效果更好（有限步长下频域成分仍有残余相关）
- ETTh1 数据集上约 50.5% 的标签步间相关系数超过 0.25，验证了自相关问题的严重性
- 变换后的成分仅少数具有大方差，可用 $\gamma<1$ 有效削减任务量

## 亮点与洞察

- **将 PCA 用于标签空间是核心创新**：传统 PCA 用于输入特征降维，本文创新性地用于标签序列去相关和显著性区分
- 理论分析清晰：从 Theorem 3.1（偏差定量分析）到 Lemma 3.2/3.3（去相关保证和 SVD 实现）形成完整理论链
- 方法极其轻量——仅需一次 SVD 和矩阵乘法，无额外参数
- 模型无关性使其具有广泛适用性，可作为标准训练技巧推广

## 局限与展望

- SVD 投影矩阵基于训练集标签计算，假设测试集标签分布一致
- 多变量场景中各变量独立处理，未考虑跨变量相关性
- $\gamma$ 的选择可能因数据集而异，目前需要在验证集上搜索
- 仅验证了 L1 范数，未系统比较其他鲁棒损失函数

## 相关工作与启发

- FreDF 是最直接的前驱工作，提出频域损失缓解偏差，但去相关不完全
- 该方法启发：在任何需要序列对齐的任务中，考虑标签空间的相关结构可能带来提升
- PCA 的标签空间应用可推广到语音、NLP 等其他序列预测任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — PCA 用于标签序列去相关+显著性区分是新思路
- 实验充分度: ⭐⭐⭐⭐⭐ — 8个数据集、多种基线模型、5个以上学习目标对比、消融详尽
- 写作质量: ⭐⭐⭐⭐ — 理论推导严密，动机阐述清楚
- 价值: ⭐⭐⭐⭐⭐ — 模型无关的训练技巧，可直接应用于现有系统

<!-- RELATED:START -->

## 相关论文

- [Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning](../../ICML2025/time_series/timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des.md)

<!-- RELATED:END -->
