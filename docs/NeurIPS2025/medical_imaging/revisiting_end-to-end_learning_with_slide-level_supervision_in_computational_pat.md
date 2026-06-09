---
title: >-
  [论文解读] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology
description: >-
  [NeurIPS 2025][医学图像][计算病理] 重新审视计算病理中切片级监督的端到端(E2E)学习，首次揭示稀疏注意力MIL在E2E训练中导致的优化困难，提出ABMILX通过多头注意力和全局注意力校正模块解决该问题，使E2E训练的ResNet在多个基准上超越SOTA基础模型。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "计算病理"
  - "端到端学习"
  - "多实例学习"
  - "稀疏注意力"
  - "ABMILX"
---

# Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology

**会议**: NeurIPS 2025  
**arXiv**: [2506.02408](https://arxiv.org/abs/2506.02408)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: 计算病理, 端到端学习, 多实例学习, 稀疏注意力, ABMILX

## 一句话总结

重新审视计算病理中切片级监督的端到端(E2E)学习，首次揭示稀疏注意力MIL在E2E训练中导致的优化困难，提出ABMILX通过多头注意力和全局注意力校正模块解决该问题，使E2E训练的ResNet在多个基准上超越SOTA基础模型。

## 研究背景与动机

计算病理(CPath)的主流范式是**两阶段方法**：先用预训练编码器离线提取切片中所有patch的特征，再用MIL聚合器进行切片级预测。近年来，基础模型(FM)如UNI(1亿patch)、GigaPath(170K切片)大幅提升了离线特征质量，但这种范式存在根本性局限：

**编码器未针对下游任务微调**：通用预训练特征对特定任务的适配不足

**编码器与MIL的优化脱节**：两阶段独立训练，无法端到端联合优化

**基础模型性能饱和**：即使将数据量扩展到170K切片、模型扩大到1B参数，在PANDA等挑战性任务上性能提升有限

**端到端(E2E)学习**是解决上述问题的直觉方案——联合训练编码器和MIL。然而，E2E学习面临计算成本高和性能不佳的双重困境，目前仍被忽视。

本文的关键洞察是：**E2E学习性能不佳的根本原因不是采样策略不好，而是稀疏注意力MIL引入的优化挑战被忽视了**。具体来说：

- 稀疏注意力（如ABMIL）在两阶段方法中表现优异，因为它能从成千上万的patch中聚焦关键区域
- 但在E2E训练中，MIL充当"软实例选择器"，其注意力分布影响编码器收到的梯度。**过度稀疏的注意力**使编码器过度拟合于有限的判别区域，且容易被冗余区域误导
- 更差的特征进一步恶化注意力准确性，形成**恶性优化循环**

## 方法详解

### 整体框架

E2E训练流水线包含：多尺度随机patch采样 → 编码器特征提取 → ABMILX聚合 → 任务头预测。联合优化编码器参数 $\theta$、MIL参数 $\phi$ 和任务头参数 $\eta$：

$$\{\hat{\theta}, \hat{\phi}, \hat{\eta}\} \leftarrow \arg\min_{\theta, \phi, \eta} \sum_{i=1}^{n} \mathcal{L}(y_i, \hat{y}_i)$$

### 关键设计

1. **多尺度随机patch采样 (MRIS)**

   采用简单随机采样替代复杂的注意力/聚类采样策略，引入多尺度信息。给定目标采样数 $s$ 和尺度集合 $\{I_1, \ldots, I_t\}$，按比例 $\{\sigma_1, \ldots, \sigma_t\}$ 分配每个尺度的采样数：

    $\hat{s}_j = \lceil s \times \sigma_j \rceil, \quad \sum_{j=1}^{t} \sigma_j = 1$

   不同尺度的patch统一resize到同一分辨率，合并为最终采样集。这模拟了病理学家多尺度观察的诊断习惯，且计算开销远低于注意力采样(9h vs 68h)。

2. **多头局部注意力 (MHLA)**

   将特征 $\mathbf{E}$ 分为 $m$ 个头特征 $\{\mathbf{H}^1, \ldots, \mathbf{H}^m\}$，每个头独立计算稀疏注意力 $\mathbf{A}^j = \text{MLP}(\mathbf{H}^j)$，然后在各头内独立聚合：

    $\mathbf{Z} = \text{Concat}(\mathbf{Z}^1, \ldots, \mathbf{Z}^m), \quad \mathbf{Z}^j = \text{Softmax}(\mathcal{G}(\mathbf{A}^j))^T \mathbf{H}^j$

   设计动机：E2E训练中MIL注意力的"假阳性"通常呈随机分布。多个独立头的投票可以**抑制对冗余实例的过度关注**，同时从不同特征子空间提供更全面的判别区域覆盖。

3. **全局注意力校正模块 (Attention Plus, A+)**

   利用patch间的全局相似性来校正局部稀疏注意力。计算头内的相似性矩阵 $\mathbf{U}^j$，将注意力在相似patch间传播：

    $\mathcal{G}(\mathbf{A}^j) = \mathbf{A}^j + \alpha \cdot \text{Softmax}\left(\frac{\mathbf{Q}^j {\mathbf{K}^j}^T}{\sqrt{\lceil D'/m \rceil}}\right) \mathbf{A}^j$

   其中 $\alpha$ 是可学习的缩放因子。关键insight是：具有相似病理特征的组织通常具有高度相似的特征，因此判别性patch的注意力应传播到其相似邻居。A+模块中传播权重 $P_{abx}(i) = \mathbf{A}_k^j \sum_{k=1}^{s} \mathbf{U}_{k,i}^j$ 同时受注意力值和相似性调节，确保只有高注意力区域才大量传播。

### 损失函数 / 训练策略

根据具体任务使用标准损失（分类用交叉熵、生存分析用NLL Survival Loss）。E2E训练总计不超过10 RTX3090 GPU小时（TCGA-BRCA上）。编码器使用ImageNet-1K预训练的ResNet-18/50。

## 实验关键数据

### 主实验——亚型分类

| 编码器 | 方法 | E2E | TCGA-BRCA AUC | TCGA-NSCLC AUC |
|--------|------|-----|---------------|----------------|
| ResNet-50 | Best-of-two-stage | ✗ | 89.35 | 95.21 |
| CHIEF | RRTMIL | ✗ | 92.49 | 97.00 |
| UNI | RRTMIL | ✗ | 94.61 | 97.88 |
| GigaPath | RRTMIL | ✗ | 94.82 | 97.63 |
| **ResNet-18** | **ABMILX (ours)** | **✓** | **93.97** | **97.09** |
| **ResNet-50** | **ABMILX (ours)** | **✓** | **95.17** | **97.06** |

### 主实验——癌症分级 & 生存分析

| 编码器 | 方法 | E2E | PANDA Acc | LUAD C-index | BRCA C-index |
|--------|------|-----|-----------|-------------|-------------|
| ResNet-50 | Best-of-two-stage | ✗ | 62.72 | 64.15 | 64.93 |
| UNI | 2DMamba | ✗ | 76.37 | 61.05 | 64.69 |
| GigaPath | 2DMamba | ✗ | 75.72 | 64.49 | 65.35 |
| **ResNet-18** | **ABMILX** | **✓** | **78.34** | **64.91** | **67.78** |

### 消融实验

| E2E中的MIL | 稀疏度 | 分类Acc ↑ | 生存C-index ↑ |
|-----------|--------|----------|-------------|
| ABMIL | 80 (极端稀疏) | 89.23 | 62.70 |
| TransMIL | 13 (全局注意力) | 91.44 | 63.42 |
| MHLA only ($\alpha=0$) | 61 | 91.58 | 63.80 |
| MHLA + A+ ($\alpha=1$) | 29 | 92.84 | 65.49 |
| **MHLA + A+ (learnable $\alpha$)** | **36** | **93.97** | **67.78** |

### 关键发现

1. **E2E训练的ResNet-50超越了基础模型**：在PANDA上E2E ResNet-50 (78.83%) > UNI (76.37%) > GigaPath (75.72%)，在BRCA亚型分类上E2E ResNet-50 (95.17%) > GigaPath+RRTMIL (94.82%)
2. **MIL的选择对E2E训练影响远大于采样策略**：注意力采样vs随机采样差距小(93.14 vs 92.72)，但不同MIL差距大(89.23 vs 93.97)
3. **稀疏度是核心矛盾**：ABMIL过度稀疏(80)导致E2E优化崩溃，TransMIL虽缓解但全局注意力被冗余实例分散，ABMILX通过适度稀疏(36)取得最优平衡
4. **推理速度优势巨大**：E2E ResNet-18推理1.7s/slide vs GigaPath 83s/slide，速度快近50倍
5. **外部验证**：从TCGA训练到CPTAC测试，E2E ResNet-50 (85.19 AUC) > UNI+TransMIL (85.24 AUC)，泛化能力与FM持平

## 亮点与洞察

- 首次指出"稀疏注意力MIL在E2E训练中引发优化困难"这一被忽视的问题，分析深入且实验验证充分
- ABMILX设计精妙：多头局部注意力抑制稀疏误差 + 全局A+模块利用patch相似性传播正确注意力，且保持了稀疏特性
- 打破了"CPath必须用大规模基础模型"的思维定式：ImageNet预训练的ResNet通过E2E学习即可达到FM水平
- 成本对比极具说服力：不需要额外预训练，总训练时间9h(3090)，推理速度快50倍

## 局限与展望

- 生存分析任务上E2E提升有限（受采样数量影响），未来可探索更多采样量
- ABMILX的全局注意力矩阵计算为 $O(s^2)$，大规模采样时可能成为瓶颈
- 仅使用ResNet作为编码器，未测试ViT等Transformer编码器的E2E训练效果
- 未与最新的E2E方法（需要多GPU集群的方法）在相同算力下对比

## 相关工作与启发

- 该工作为CPath领域的E2E学习提供了坚实的起点：优化挑战的识别 + 简单有效的解决方案
- ABMILX中"注意力传播+可学习稀疏度"的设计可推广到其他需要注意力的弱监督任务
- E2E学习可能也受益于上游预训练，结合基础模型+E2E微调是自然的下一步
- 稀疏注意力在E2E中的恶性循环问题可能也存在于其他weak MIL场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示E2E优化困难的根源并提出优雅解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、3种任务、外部验证、详尽消融、成本对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 可能改变CPath领域"重基础模型、轻E2E"的研究范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[CVPR 2026\] A protocol for evaluating robustness to H&E staining variation in computational pathology models](../../CVPR2026/medical_imaging/a_protocol_for_evaluating_robustness_to_he_stainin.md)
- [\[NeurIPS 2025\] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression](imagenet-trained_cnns_are_not_biased_towards_texture_revisiting_feature_reliance.md)
- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[ECCV 2024\] Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](../../ECCV2024/medical_imaging/pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)

</div>

<!-- RELATED:END -->
