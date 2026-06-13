---
title: >-
  [论文解读] Exploring Structural Degradation in Dense Representations for Self-supervised Learning
description: >-
  [NeurIPS 2025][语义分割][自监督学习] 发现并系统研究了自监督学习中"稠密退化"（SDD）现象——训练越久分类越好但稠密任务性能反而下降，提出 DSE 度量和基于 DSE 的模型选择/正则化策略，平均提升 mIoU 3.0%。 自监督学习（SSL）已在图像级表示学习中取得巨大成功，但稠密（patch/pixe…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "自监督学习"
  - "稠密表示"
  - "性能退化"
  - "模型选择"
  - "正则化"
---

# Exploring Structural Degradation in Dense Representations for Self-supervised Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.17299](https://arxiv.org/abs/2510.17299)  
**代码**: [GitHub](https://github.com/EldercatSAM/SSL-Degradation)  
**领域**: 图像分割  
**关键词**: 自监督学习, 稠密表示, 性能退化, 模型选择, 正则化

## 一句话总结

发现并系统研究了自监督学习中"稠密退化"（SDD）现象——训练越久分类越好但稠密任务性能反而下降，提出 DSE 度量和基于 DSE 的模型选择/正则化策略，平均提升 mIoU 3.0%。

## 研究背景与动机

自监督学习（SSL）已在图像级表示学习中取得巨大成功，但稠密（patch/pixel 级）表示学习改善有限。本文发现了一个反直觉现象：

**Self-supervised Dense Degradation (SDD)**：虽然训练损失收敛且分类性能稳步提升，但语义分割等稠密任务的性能在训练后期反而下降

**普遍性**：SDD 在 16 种 SOTA SSL 方法中一致出现，涵盖对比学习（MoCo v3, DenseCL）、非对比学习（BYOL, SimSiam, DINO）、基于聚类（SwAV）、掩码建模（MAE, I-JEPA）等各种范式

**非过拟合**：即使训练和评估使用同一数据集（COCO），SDD 仍然存在（DINO 下降 4.0% mIoU）

**现有度量失效**：α-REQ、RankMe、Lidar 等指标主要面向图像级任务，与稠密性能负相关

## 方法详解

### 整体框架

基于误差率分解理论，提出 Dense representation Structure Estimator（DSE），由类可分性度量和有效维度度量组成：

$$\text{DSE} = M_{inter} - M_{intra} + \lambda \cdot M_{dim}$$

### 关键设计

**理论基础**：
- **定理 2**（类相关度量）：证明当类内半径（通过归一化表示矩阵的trace估计）小于类间距离时，简单的 NN 分类器即可正确分类
- **推论 5**（维度影响）：证明下游错误率随表示维度 $d$ 指数衰减：$\text{Err} \leq \delta + 2K\exp(-\tilde{C}_\delta \cdot d)$

**类可分性度量**：
- 使用 k-means 聚类生成伪标签
- 类内半径：$M_{intra} = \frac{1}{k}\sum_{j=1}^{k} \frac{\sum_{i=1}^{\min(\tilde{N}_j, d)} \sigma_i(\tilde{Z}_c^j)}{(\tilde{N}_j - 1)}$
- 类间距离：$M_{inter} = \frac{1}{k}\sum_{j=1}^{k} \frac{1}{N_j}\sum_{z \in \tilde{Z}_j} \min_{i \neq j} \|z - \tilde{\mu}_i\|^2$

**有效维度度量**：
- 随机采样 $B'$ 个独立稠密表示，计算有效秩：$M_{dim} = \text{Erank}(\bar{Z}) = \exp(-\sum_i p_i \log p_i)$

**自适应缩放**：$\lambda = \text{Std}(M_{inter} - M_{intra}) / \text{Std}(M_{dim})$

### 损失函数 / 训练策略

**DSE 引导的模型选择**（离线）：
1. 对所有检查点计算 DSE
2. 选择 DSE 的局部最大值点作为候选
3. 取 top-3 最高 DSE 值的检查点

**DSE 正则化**（在线）：
$$\mathcal{L} = \mathcal{L}_{original} - \beta \cdot \text{DSE}$$
其中 $\lambda = 1$，$\beta = 0.001$，从最佳初始性能的检查点开始训练 10 个 epoch。

## 实验关键数据

### 主实验

**16 种 SSL 方法的 SDD 现象**（COCO-Stuff/PASCAL VOC/ADE20k/Cityscapes 上的 Best vs Last mIoU 差距）：

| 方法 | COCO Diff | VOC Diff | ADE20k Diff | Cityscapes Diff |
|------|-----------|----------|-------------|-----------------|
| MoCo v3 | -22.0 | -45.2 | -14.4 | -11.5 |
| DINO | -4.4 | -11.3 | -4.2 | -0.1 |
| iBOT | -2.5 | -3.0 | -3.7 | -3.2 |
| I-JEPA | -5.6 | -7.6 | -4.5 | -3.9 |
| BYOL | -6.4 | -6.7 | -7.9 | -7.5 |
| MAE | -0.4 | -1.3 | -0.7 | -2.1 |

**DSE 模型选择效果**（+MS 表示模型选择后的提升）：

| 方法 | COCO mIoU | VOC mIoU |
|------|-----------|----------|
| MoCo v3 | 15.1 → **30.9** (+15.8) | 5.9 → **42.0** (+36.1) |
| BYOL | 30.7 → **37.1** (+6.4) | 45.4 → **51.1** (+5.7) |
| I-JEPA | 34.0 → **39.6** (+5.6) | 52.6 → **59.3** (+6.7) |
| EsViT | 33.4 → **41.6** (+8.2) | 54.3 → **59.8** (+5.5) |

### 消融实验

**DSE 与其他度量对比**（平均 Kendall's τ）：

| 度量 | COCO | VOC | ADE20k | City | 平均 |
|------|------|-----|--------|------|------|
| α-ReQ | -0.07 | -0.05 | -0.05 | 0.09 | -0.02 |
| RankMe | -0.10 | -0.09 | -0.14 | 0.00 | -0.08 |
| Lidar | -0.37 | -0.36 | -0.26 | -0.21 | -0.30 |
| RankMe† (稠密适配) | 0.25 | 0.26 | 0.22 | 0.23 | 0.24 |
| **DSE (Ours)** | **0.58** | **0.60** | **0.56** | **0.49** | **0.57** |

**DSE 组件消融**（平均 Kendall's τ）：

| 类可分性 | 有效维度 | COCO | VOC | ADE20k | City | 平均 |
|----------|----------|------|-----|--------|------|------|
| ✓ | ✗ | 0.45 | 0.42 | 0.33 | 0.37 | 0.39 |
| ✗ | ✓ | 0.25 | 0.26 | 0.22 | 0.23 | 0.24 |
| ✓ | ✓ | **0.58** | **0.60** | **0.56** | **0.49** | **0.57** |

**效率对比**：

| 方法 | 平均改善 | 计算开销 (GPU·h) |
|------|----------|-----------------|
| Loss-based | -1.0 | 0.0 |
| Supervised | +3.6 | 2.43 |
| DSE (Ours) | **+3.0** | **0.025** (~97× 加速) |

### 关键发现

1. **退化原因因方法而异**：MoCo v3 是维度坍缩导致，DINO 是类可分性下降导致
2. **DSE 正则化能逆转退化趋势**：在 iBOT 和 I-JEPA 上，添加 DSE 正则化后性能不再下降
3. **DSE 也推广到图像级任务**：在 ImageNet k-NN 评估上平均 Kendall's τ 达 0.86，优于 RankMe 的 0.79
4. **仅需极少数据**：2048 张图像（~0.16% 训练数据）即可准确计算 DSE

## 亮点与洞察

1. **现象发现的广泛性**：16 种方法 × 4 个数据集 × 多种评估协议统一验证了 SDD 的存在，这是一个社区级别的重要发现
2. **理论与实践的优雅统一**：从误差率分解出发，推导出类可分性和维度两个因素，然后设计出可直接优化的 DSE 度量
3. **实用价值极高**：模型选择仅需 0.025 GPU·h，即可平均提升 3.0% mIoU
4. **命题 1 的洞察**：揭示了用 k-means 伪标签的实例级距离度量永远预测准确率为 1 的根本问题，进而设计类级半径度量

## 局限与展望

1. 理论分析主要针对 linear probing 设置，未充分考虑 transfer learning 中的分布偏移
2. DSE 在深度估计等回归任务上的预测能力相对较弱（Kendall's τ 较低）
3. 训练时 DSE 正则化需要模型特定的适配（如学生模型的稠密表示提取方式）
4. 尚未深入分析各方法维度坍缩或可分性退化的具体机理

## 相关工作与启发

- DINOv3（并发工作）从 iBOT/DINOv2 家族角度研究退化，使用 gram 矩阵蒸馏解决，与本文互补
- RankMe 本质是本文 $M_{dim}$ 的稠密版本，但无法捕捉类可分性的退化
- 本文发现为 SSL 训练策略设计提供了新视角：应在类可分性和维度坍缩之间寻求折中

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 发现了社区未充分认识的重要现象
- 实验完整度：⭐⭐⭐⭐⭐ — 16 种方法 × 4 数据集的系统性验证
- 实用性：⭐⭐⭐⭐⭐ — 几乎零成本的模型选择策略
- 写作质量：⭐⭐⭐⭐⭐ — 从现象到理论到方法的逻辑链条非常清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](../../CVPR2025/segmentation/exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)
- [\[NeurIPS 2025\] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](vision_transformers_with_self-distilled_registers.md)

</div>

<!-- RELATED:END -->
