---
title: >-
  [论文解读] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation
description: >-
  [NeurIPS 2025][医学图像][半监督分割] 提出 VQ-Seg，首次将向量量化引入半监督医学图像分割，用量化扰动模块（QPM）替代传统 dropout 实现更可控的特征扰动，并结合双分支架构和基础模型引导对齐来弥补量化信息损失。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "半监督分割"
  - "向量量化"
  - "特征扰动"
  - "一致性学习"
  - "医学图像分割"
---

# VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2601.10124](https://arxiv.org/abs/2601.10124)  
**代码**: [GitHub](https://github.com/script-Yang/VQ-Seg)  
**领域**: 医学图像  
**关键词**: 半监督分割, 向量量化, 特征扰动, 一致性学习, 医学图像分割

## 一句话总结

提出 VQ-Seg，首次将向量量化引入半监督医学图像分割，用量化扰动模块（QPM）替代传统 dropout 实现更可控的特征扰动，并结合双分支架构和基础模型引导对齐来弥补量化信息损失。

## 研究背景与动机

半监督医学图像分割中，一致性学习配合特征扰动是广泛使用的策略。然而，现有方法大量依赖 dropout 进行特征级扰动，存在根本性问题：

**低 dropout 率（如 0.3, 0.5）**：扰动不足，对分割性能几乎无影响，无法提供有意义的正则化

**高 dropout 率（如 ≥0.7）**：性能急剧下降，Dice 和 Jaccard 骤降，HD95 和 ASD 大幅上升，DR=0.9 时输出完全不可用

**找到最优 dropout 率极其困难**：依赖数据集、任务和网络架构，需要大量手动调参

从理论上看，对于 dropout，后验与先验之间的 KL 散度可近似为：

$$D_{KL}(P||Q) \approx \frac{1}{2}\left(\frac{p}{1-p} + \log(1-p)\right)$$

当 $p$ 增大时 KL 散度急剧增长，导致过度正则化和学习退化。这激发了在离散向量量化空间中进行更可控扰动的思路。

## 方法详解

### 整体框架

VQ-Seg 包含四个核心组件：（1）VQ 编码器将连续特征量化为离散码本空间；（2）量化扰动模块（QPM）在码本索引空间内进行可控扰动；（3）双分支架构共享后量化空间，联合优化重建和分割；（4）基础模型引导的后量化特征适配器（PFA）弥补语义损失。采用教师-学生框架进行一致性学习。

### 关键设计

1. **量化扰动模块（QPM）**：编码器输出 $z = f_{\text{enc}}(x)$ 经 VQ 量化为最近码字索引 $i = \arg\min_j \|z - c_j\|$。QPM 定义扰动策略——给定原始码字 $c_i$，以概率 $\pi(j|i)$ 替换为另一个码字 $c_j$：

$$\pi(j|i) = \begin{cases} 1 - \epsilon, & \text{if } j = i \\ \frac{\epsilon \exp(-d(c_i, c_j))}{Z_i}, & \text{if } j \neq i \end{cases}$$

   其中 $\epsilon \in [0,1]$ 控制扰动强度，$d(c_i, c_j)$ 是码字距离，$Z_i$ 是归一化因子。核心优势是：与 dropout 不同，QPM 的扰动分布 $Q(c_j|\epsilon)$ 始终有界，受学习到的码本结构引导，在语义相近的码字间进行替换，更可控且可解释。例如扰动强度 $\epsilon=0.7$ 时，最近码字 c₂ 有 49% 概率被选中。

2. **双分支共享后量化空间**：VQ 量化可能丢失细粒度视觉信息。为此设计双分支架构，后量化特征同时输入图像解码器 $D_i$ 和分割解码器 $D_s$：

$$\hat{x} = D_i(q(\mathbf{z})), \quad \hat{y} = D_s(q(\mathbf{z}))$$

   有标签数据：$\mathcal{L}_l = \mathcal{L}_{rec}(x_l, \hat{x}_l^S) + \mathcal{L}_{seg}(y_l, \hat{y}_l^S)$

   无标签数据使用教师网络生成伪标签 $\tilde{y}_u$：$\mathcal{L}_u = \mathcal{L}_{rec}(x_u, \hat{x}_u^S) + \mathcal{L}_{seg}(\tilde{y}_u, \hat{y}_u^S) + \mathcal{L}_{seg}(\tilde{y}_u, \hat{y}_a^S)$

   重建分支作为自监督信号，鼓励 VQ 编码器学习更好的表征。

3. **基础模型引导的后量化特征适配器（PFA）**：使用冻结的 DINOv2 作为外部语义先验。PFA 通过 resize + 1×1 卷积匹配特征分辨率和通道维度，然后通过逐 patch 对比学习对齐量化特征与 FM 特征：

$$\mathcal{L}_{\text{align}} = -\frac{1}{HW} \sum_{i=1}^{HW} \log \frac{\exp(\text{sim}(f_i^{\text{pfa}}, f_i^{\text{fm}})/\tau)}{\sum_{j=1}^{HW} \exp(\text{sim}(f_i^{\text{pfa}}, f_j^{\text{fm}})/\tau)}$$

   这种局部化的语义监督能弥补量化过程中的细节和语义漂移。

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{db} + \lambda_a \mathcal{L}_{\text{align}}$，其中 $\mathcal{L}_{db} = \mathcal{L}_l + \lambda_u \mathcal{L}_u$。$\mathcal{L}_{rec}$ 为 L1 损失，$\mathcal{L}_{seg}$ 为交叉熵损失。教师网络通过 EMA 更新（$\alpha=0.996$）。码本大小 $K=16384$，训练 100K 迭代，AdamW 优化器，4 张 RTX 4090 GPU。

## 实验关键数据

### 主实验（LC 肺癌数据集）

| 方法 | 5% 标注 Dice↑ | 5% Jaccard↑ | 5% HD95↓ | 10% Dice↑ | 10% Jaccard↑ | 10% HD95↓ |
|------|-------------|------------|---------|----------|-------------|---------|
| UNet-F(全监督) | 0.8345 | 0.7386 | 6.9634 | 0.8345 | 0.7386 | 6.9634 |
| UNet-S | 0.4343 | 0.3118 | 26.0498 | 0.6490 | 0.5175 | 21.4063 |
| UA-MT | 0.6029 | 0.4647 | 48.6681 | 0.7222 | 0.5989 | 11.6724 |
| Unimatch | 0.6493 | 0.5071 | 17.8700 | 0.7511 | 0.6333 | 17.0178 |
| **VQ-Seg** | **0.6643** | **0.5257** | **12.2525** | **0.7852** | **0.6731** | **11.6179** |

在 10% 标注下 Dice 提升 2.97%，Jaccard 提升 3.17%。

### 消融实验

| Base | QPM | DB | PFA | Dice↑ | Jaccard↑ | HD95↓ | ASD↓ |
|------|-----|-----|-----|-------|---------|-------|------|
| ✓ | | | | 0.7443 | 0.6238 | 14.2153 | 5.2301 |
| ✓ | ✓ | | | 0.7701 | 0.6559 | 13.0246 | 4.9378 |
| ✓ | ✓ | ✓ | | 0.7784 | 0.6620 | 12.4728 | 4.6013 |
| ✓ | ✓ | ✓ | ✓ | **0.7852** | **0.6731** | **11.6179** | **4.2094** |

QPM 贡献最大（+2.58% Dice），三个模块协同效果最优。

### 关键发现

- 扰动强度 $\epsilon=0.7$ 为最优，$\epsilon=0.9$ 时性能明显下降，但远不如 dropout 的急剧恶化
- DINOv2 作为语义先验在多个基础模型（CLIP、BiomedCLIP、MAE、Rad-DINO）中表现最优
- 码本大小 $K=16384$ 最优：太小（1024）表达力不足，太大（65536）利用率下降到 92%
- 新收集的肺癌数据集（828 例 CT）为中央型肺癌标注，具有临床实用价值

## 亮点与洞察

- **VQ 在半监督分割中的首次应用**：将离散码本空间作为扰动载体，比 dropout 更结构化、可控
- **从理论到实做的完整论证**：从 KL 散度分析 dropout 不稳定性 → 提出 QPM → 实验验证，逻辑链完整
- **双分支设计巧妙应对量化信息损失**：重建分支不仅保留视觉信息，还为 VQ 编码器提供自监督信号
- **新数据集贡献**：828 例肺癌 CT 是有价值的临床数据集，填补中央型肺癌分割数据空白

## 局限与展望

- 仅在 2D 切片上实验，未验证 3D 情况下 VQ 的表现
- 码本学习的稳定性和效率可能在更复杂任务中成为瓶颈
- QPM 的扰动策略基于码字距离，对码本质量敏感
- 仅使用 DINOv2 作为基础模型先验，未探索多模型集成或医学专用大模型

## 相关工作与启发

与 UA-MT（Monte Carlo Dropout）、BCP、Unimatch 等方法相比，VQ-Seg 的核心差异在于用离散空间的结构化扰动替代连续空间的随机丢弃。与通用 VQ-VAE 文献相比，VQ-Seg 巧妙地将 VQ 空间同时服务于扰动和重建。启发：当连续空间中的正则化策略表现不稳定时，映射到离散空间可能提供更可控的替代方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 VQ 引入半监督分割做扰动，理论分析支撑充分
- 实验充分度: ⭐⭐⭐⭐ 消融细致，但仅两个数据集（LC 和 ACDC），可更多样化
- 写作质量: ⭐⭐⭐⭐ 动机清晰，图示直观，结构完整
- 价值: ⭐⭐⭐⭐ 为半监督分割中的扰动策略提供新思路，新数据集有额外贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[ECCV 2024\] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](../../ECCV2024/medical_imaging/alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)
- [\[ICML 2026\] Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](../../ICML2026/medical_imaging/are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](../../AAAI2026/medical_imaging/bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)

</div>

<!-- RELATED:END -->
