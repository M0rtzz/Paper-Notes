---
title: >-
  [论文解读] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation
description: >-
  [NeurIPS 2025][图像分割][无监督域适应] 提出 DiDA，通过将图像退化操作形式化为扩散前向过程来构建源域和目标域之间的连续中间域，结合语义偏移补偿机制，作为即插即用模块显著提升现有 UDA 语义分割方法的性能。
tags:
  - NeurIPS 2025
  - 图像分割
  - 无监督域适应
  - 语义分割
  - 扩散过程
  - 图像退化
  - 域桥接
---

# Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2412.10339](https://arxiv.org/abs/2412.10339)  
**代码**: [有](https://github.com/Woof6/DiDA)  
**领域**: Segmentation / Domain Adaptation  
**关键词**: 无监督域适应, 语义分割, 扩散过程, 图像退化, 域桥接

## 一句话总结

提出 DiDA，通过将图像退化操作形式化为扩散前向过程来构建源域和目标域之间的连续中间域，结合语义偏移补偿机制，作为即插即用模块显著提升现有 UDA 语义分割方法的性能。

## 研究背景与动机

语义分割模型在跨域部署时面临严重的性能下降。虽然自训练（Self-Training）已成为 UDA 的主流范式（如 DAFormer、HRDA、MIC 系列），但这些方法**忽略了域共享特征提取的显式建模**。

从因果表示学习角度分析：观察到的特征 $x = \Phi(c, e)$，其中 $c$ 是决定类别身份的因果特征（如形状），$e$ 是域特定特征（如纹理）。由于 $e_S \neq e_T$，导致 $x_S \neq x_T$，阻碍了域不变特征的学习。

**核心洞察**来自扩散模型的前向过程：逐步添加噪声会按粒度顺序移除属性——**细粒度的域特定属性（纹理）先丢失，粗粒度的域不变属性（形状）后丢失**。这意味着退化操作创建的中间域分布的重叠区域可以作为**域共享分布的先验**。

但直接使用退化作为域桥接面临两大挑战：(1) 需要在宽范围退化级别下保持稳定的特征表示；(2) 退化不可避免地损害域不变特征，导致**语义偏移**问题。

## 方法详解

### 整体框架

DiDA 集成到标准自训练(ST) UDA 流程中，包含两个核心模块：(1) 基于退化的中间域构建，通过扩散前向过程创建连续中间域；(2) 语义偏移补偿，使用扩散编码器解耦并补偿退化引起的语义信息丢失。在推理时，仅使用骨干分割网络 $f_\theta = h \circ g$，无需额外计算开销。

### 关键设计

1. **基于退化的中间域构建 (Degradation-based Intermediate Domain Construction)**：将扩散前向过程 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 产生的中间状态 $X_1, X_2, \ldots, X_T$ 视为中间域。随着时间步增大，不同域分布的重叠面积逐渐扩大，消除域特定属性。基于理论命题（属性丢失与时间步的单调关系），退化操作构建了从源/目标域到共享域的连续桥接。

2. **语义偏移补偿 (Semantic Shift Compensation)**：引入可训练的扩散编码器 $g'$，以时间嵌入模块条件化，对退化图像 $x_t$ 提取语义偏移信息：
   $$\hat{z}_{(t,i)} = z'_{(t,i)} (MLP_s^i \circ \text{Embed}(t) + 1) + MLP_b^i \circ \text{Embed}(t)$$
   通过残差连接在多层级上融合特征 $g + g'$，用重建损失 $\mathcal{L}^R = \|f_\theta(x_t, t) - \epsilon\|_2^2$ 监督。设计动机：时间嵌入使网络能精确解耦不同退化程度对应的语义损失，从而针对性地补偿。

3. **退化图像一致性损失 (Degraded Image Consistency, DIC)**：
   $$\mathcal{L}^D = \sum_{i}^{N_S} \mathcal{L}_{ce}(\bar{f}_\theta(x_{i,t}^S, t), y_i^S) + \sum_{i}^{N_T} \mathcal{L}_{ce}(\bar{f}_\theta(x_{i,t}^T, t), p_i^T, q^T)$$
   其中 $\bar{f}_\theta = h \circ (g + g')$，强制退化图像和原始图像的预测一致。

### 损失函数 / 训练策略

总训练损失为四项的加权和：
$$\mathcal{L} = \mathcal{L}^S + \mathcal{L}^T + \lambda_D \mathcal{L}^D + \lambda_R \mathcal{L}^R$$

- $\mathcal{L}^S$：源域监督损失
- $\mathcal{L}^T$：目标域伪标签自训练损失
- $\lambda_D = 0.5$，$\lambda_R$ 根据架构调整（DAFormer: 5, DeepLabV2: 1）
- 噪声调度：$T=100$，使用 sigmoid schedule
- 推理时完全移除 $g'$ 和 $h'$，零额外开销

## 实验关键数据

### 主实验

**跨方法、跨架构的一致性提升 (mIoU)**

| 方法 | GTA→CS (CNN) | GTA→CS (Trans) | SYN→CS (CNN) | SYN→CS (Trans) | CS→ACDC (Trans) |
|------|-------------|----------------|-------------|----------------|-----------------|
| DAFormer | 56.0 | 68.3 | 54.7 | 60.9 | 55.4 |
| +DiDA | **58.3** (+2.3) | **70.3** (+2.0) | **57.6** (+2.9) | **63.1** (+2.2) | **59.1** (+3.7) |
| HRDA | 63.0 | 73.8 | 61.2 | 65.8 | 68.0 |
| +DiDA | **64.3** (+1.3) | **75.4** (+1.6) | **62.6** (+1.4) | **67.8** (+2.0) | **70.7** (+2.7) |
| MIC | 64.2 | 75.5 | 62.4 | 67.3 | 69.8 |
| +DiDA | **65.0** (+0.8) | **76.8** (+1.3) | **63.5** (+1.1) | **68.6** (+1.3) | **72.1** (+2.3) |

### 消融实验

**GTA→CS (Transformer), 基于 DAFormer**

| $\mathcal{L}^D$ | $\mathcal{L}^R$ | $g_{time}$ | $g'$ | $h'$ | mIoU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| - | - | - | - | - | 68.3 |
| ✓ | - | - | - | - | 66.5 |
| ✓ | - | ✓ | - | - | 69.5 |
| ✓ | ✓ | ✓ | - | - | 69.4 |
| ✓ | ✓ | ✓ | - | ✓ | 69.9 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **70.3** |

### 关键发现

- **即插即用有效性**：DiDA 在所有 3 种 UDA 方法 × 2 种架构 × 5 种设置中均带来提升
- **天气适应场景提升最大**：CS→ACDC 上提升可达 +3.7 mIoU，说明退化桥接对域差异大的场景尤其有效
- **语义偏移补偿至关重要**：不加时间嵌入的 DIC 损失反而下降 1.8 mIoU（66.5 vs 68.3），加入时间嵌入后恢复并超越基线
- **扩展性好**：可替换为模糊、inpainting 等任意退化操作，均有效

## 亮点与洞察

- **优雅的理论动机**：从扩散模型属性丢失的理论命题出发，将"退化=域桥接"这一直觉形式化
- **推理零开销**：退化编码器和重建头仅在训练时使用，不增加部署成本
- **通用性极强**：兼容 CNN 和 Transformer 架构，兼容多种 UDA 基线方法，支持多种退化操作
- 域桥接视角相比传统的对抗训练和风格迁移提供了新思路

## 局限与展望

- 退化级别 $T=100$ 和噪声调度的选择基于经验，理论上最优退化策略尚不明确
- 扩散编码器 $g'$ 与骨干编码器 $g$ 结构相同，参数量翻倍（虽然推理时移除）
- 在强基线 MIC 上的提升相对较小（+0.8~1.3），可能接近性能天花板
- 未探索与非自训练 UDA 方法（如对抗训练）的结合

## 相关工作与启发

- **与 MIC/HRDA 的关系**：DiDA 作为 plug-and-play 补充而非替代，与一致性正则化思路正交
- **与扩散模型在分割中的应用**：不同于利用扩散生成数据或提取内部特征的方法，DiDA 将扩散策略直接整合进 UDA 训练流程
- **启发**：退化作为域桥接的思想可推广到检测、分类等其他跨域任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （退化=域桥接视角新颖，与扩散理论的结合自然）
- 实验充分度: ⭐⭐⭐⭐⭐ （多方法/多架构/多基准/详细消融）
- 写作质量: ⭐⭐⭐⭐ （动机阐述清晰，理论与实践结合好）
- 价值: ⭐⭐⭐⭐⭐ （即插即用的通用UDA增强策略，实际价值高）

<!-- RELATED:START -->

## 相关论文

- [Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [Exploring Structural Degradation in Dense Representations for Self-supervised Learning](exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [Dual form Complementary Masking for Domain-Adaptive Image Segmentation](../../ICML2025/segmentation/dual_form_complementary_masking_for_domain-adaptive_image_segmentation.md)
- [Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)

<!-- RELATED:END -->
