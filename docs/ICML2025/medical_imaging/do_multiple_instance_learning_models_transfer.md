---
title: >-
  [论文解读] Do Multiple Instance Learning Models Transfer?
description: >-
  [ICML2025][医学图像][Multiple Instance Learning] 首次系统评估计算病理学中 MIL 模型的迁移学习能力，发现在 pancancer 数据集上预训练的 MIL 模型能够跨器官、跨任务泛化，以不到 10% 的预训练数据超越自监督 slide foundation model（CHIEF、GigaPath）。
tags:
  - ICML2025
  - 医学图像
  - Multiple Instance Learning
  - 迁移学习
  - Computational Pathology
  - Slide Foundation Model
  - Pancancer Pretraining
---

# Do Multiple Instance Learning Models Transfer?

**会议**: ICML2025  
**arXiv**: [2506.09022](https://arxiv.org/abs/2506.09022)  
**代码**: [mahmoodlab/MIL-Lab](https://github.com/mahmoodlab/MIL-Lab)  
**领域**: medical_imaging / 计算病理学 / 多实例学习  
**关键词**: Multiple Instance Learning, Transfer Learning, Computational Pathology, Slide Foundation Model, Pancancer Pretraining

## 一句话总结

首次系统评估计算病理学中 MIL 模型的迁移学习能力，发现在 pancancer 数据集上预训练的 MIL 模型能够跨器官、跨任务泛化，以不到 10% 的预训练数据超越自监督 slide foundation model（CHIEF、GigaPath）。

## 研究背景与动机

**核心问题**：MIL（多实例学习）是计算病理学处理千兆像素全切片图像（WSI）的基石范式，但受制于小规模弱监督临床数据集，模型性能受限。在 NLP 和传统 CV 中迁移学习已广泛用于应对数据稀缺，但 MIL 模型的可迁移性却几乎未被研究过——目前随机初始化仍是 MIL 模型开发和评估的标准做法。

**研究动机**：

- 尽管 patch 级别编码器（UNI、Virchow 等）的迁移学习已被广泛采用，但 slide 级别聚合器的迁移学习完全被忽视
- 自监督 slide foundation model（CHIEF、GigaPath）需要数万甚至十几万张 WSI 进行预训练，数据和计算成本极高
- 作者假设：在大规模多类别 pancancer 分类任务上有监督预训练的 MIL 模型，可以作为一种简单而有效的 slide foundation model 替代方案

**MIL 工作流程回顾**：给定一张 WSI，首先用预训练 patch encoder 将其分割为 patch 并提取特征（$\sim$1000–10000 个 patch），然后通过可训练的聚合器将所有 patch 特征池化为一个 slide 级别表示，用于下游分类。

## 方法详解

### 实验框架：有监督 MIL 迁移

对于 MIL 架构 $f$、预训练任务 $s$、目标任务 $t$，研究回答三个核心问题：

1. **$f_{s \to t}$ vs. $f_{\text{rand} \to t}$**：预训练是否优于从头训练？
2. **$f_{s \to t}$ vs. $f_{s' \to t}$**：不同预训练任务迁移效果如何？
3. **$f_{s \to t}$ vs. $f'_{s \to t}$**：不同架构迁移能力有何差异？

### 评估设置

- **11 种 MIL 架构**：ABMIL、CLAM、DSMIL、DFTD、TransMIL、Transformer、ILRA、RRT、WIKG、MeanMIL、MaxMIL
- **21 个预训练任务 + 19 个目标任务**：涵盖乳腺、肺、前列腺、脑 4 个器官，包含癌症分类、分级、分子亚型预测等
- **Pancancer 预训练任务**：PC-43（43 类）和 PC-108（108 类 OncoTree 编码），来自 17 种器官的 3,499 张 WSI
- **两种评估方式**：端到端微调 + 冻结特征 KNN 评估

### 标准化实现

- Patch 切分：$256 \times 256$，20× 放大（0.5 μm/pixel）
- Patch 编码器：UNI（DINOv2 预训练 ViT-L/16）
- 优化器：AdamW，学习率 $1 \times 10^{-4}$，cosine 衰减
- 最大 20 epoch，早停 patience = 5

## 实验关键数据

### 预训练任务质量对比（KNN 冻结特征评估）

| 预训练策略 | 相对 baseline 平均提升 |
|:---|:---|
| PC-108 pancancer | **+9.8%** |
| PC-43 pancancer | +8.6% |
| 单器官任务（域内） | +3–6% |
| 单器官任务（域外） | +1–4% |
| 随机初始化 baseline | 0% |

**关键发现**：即使是跨器官的预训练（如肺→乳腺），也能带来显著提升。

### 11 种架构微调迁移（PC-108 预训练 vs 随机初始化）

| 架构 | 随机初始化 | PC-108 | Δ |
|:---|:---|:---|:---|
| ABMIL | 71.7 | 75.5 | **+3.8** |
| DFTD | 69.6 | 76.6 | **+7.0** |
| TransMIL | 68.1 | 73.9 | **+5.8** |
| Transformer | 68.5 | 74.3 | **+5.8** |
| DSMIL | 72.3 | 73.0 | +0.7 |
| CLAM | 69.0 | 70.5 | +1.5 |
| WIKG | 69.3 | 74.7 | +5.4 |
| 所有模型平均 | 70.1 | 73.4 | **+3.3** |

### Few-shot 学习（K=4,16,32 samples/class）

- DFTD 在 K=4 时，PC-108 预训练相比随机初始化提升 **171%**
- 所有 5 种方法在所有 shot 数下，pancancer 预训练均优于随机初始化
- PC-108 始终优于 PC-43，说明更细粒度的分类任务带来更好的数据效率

### 对比 Slide Foundation Model

| 项目 | PC-108 ABMIL | CHIEF | GigaPath |
|:---|:---|:---|:---|
| 预训练数据量 | **3,944 WSI** | 60,530 WSI | 171,189 WSI |
| 预训练方式 | 有监督分类 | 对比学习+CLIP | 自监督 MAE |
| KNN 胜出任务数 | **12/15** vs CHIEF | 3/15 | 2/15 |
| 微调胜出任务数 | **11/15** vs CHIEF | 4/15 | 5/15 |
| KNN 平均提升 | — | +5.9% over CHIEF | +9.7% over GigaPath |

PC-108 仅用 CHIEF 6.5%、GigaPath 2.3% 的预训练数据，在大多数任务上取得更优结果。

### 模型规模与迁移

- 随机初始化在不同模型规模上性能波动大
- PC-108 预训练下，性能从 0.1M 到 5M 参数单调递增，展现出良好的 scaling 趋势
- 9M 参数时性能略降，但仍大幅优于随机初始化

### 迁移的关键组件分析

通过逐层重置实验（ABMIL 四层结构）：

| 重置策略 | 相对完整迁移的性能下降 |
|:---|:---|
| 重置 Attention 层 | **-5.0%** |
| 重置 Attention + 线性层3 | -5.2% |
| 重置 Attention + 线性层2+3 | -6.6% |
| 全部重置（=随机初始化） | -8.3% |

**Attention 聚合层是迁移知识的核心载体**，与 CNN 迁移中后层不重要的结论不同。

## 亮点与洞察

1. **"预训练比架构重要"**：随机初始化最优架构（DSMIL 72.3）低于 9/11 种预训练后架构的性能，说明好的初始化比好的架构更重要
2. **简单架构 + 好初始化 = 最优**：ABMIL 作为最简单的注意力池化方法，在预训练后表现最佳，验证了"强 patch encoder + 简单聚合器"的有效性
3. **有监督 pancancer 预训练 > 大规模自监督**：以极少数据（~4k WSI）超越用 6–17 万 WSI 预训练的 foundation model，说明精心设计的分类任务比堆数据更有效
4. **注意力热力图可视化**：预训练模型在微调前就已关注肿瘤区域，而随机初始化模型注意力弥散——预训练帮助模型避免虚假相关
5. **跨 patch encoder 一致有效**：在 ResNet-50、CTransPath、GigaPath ViT、UNIv2、CONCHv1.5 五种编码器上均观察到 PC-108 预训练带来的提升

## 局限与展望

1. **缺少 State-Space MIL 模型**：如 Mamba 系列架构未纳入评估
2. **未评估生存预测任务**：仅覆盖分类/分级，未涉及 Cox 回归等生存分析
3. **预训练数据来源单一**：PC-108 全部来自 Brigham and Women's Hospital，可能存在机构偏差
4. **未探索增强预训练策略**：如数据增广、自监督+有监督混合预训练可能进一步提升
5. **Patch encoder 冻结**：全程使用冻结的预训练 patch encoder，未探索端到端联合微调

## 相关工作与启发

- **Patch Foundation Model**：UNI (Chen et al., 2024)、Virchow (Vorontsov et al., 2024) — 本文关注的是 slide 级别迁移，与 patch 级别迁移互补
- **Slide Foundation Model**：CHIEF (Wang et al., 2024)、GigaPath (Xu et al., 2024) — 本文证明有监督预训练可作为更高效的替代方案
- **MIL 架构**：ABMIL → CLAM → TransMIL → WIKG — 本文发现架构差异对迁移后性能影响有限
- **NLP/CV 迁移学习**：ImageNet 预训练范式 — 本文将 PC-108 类比为病理学的 "ImageNet"

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个系统研究病理 MIL 迁移学习的工作，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ — 11 架构 × 21 任务 × 多种编码器，规模极大且设计严谨
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，结论有力，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 对病理 AI 社区有直接实用价值，开源权重和代码

<!-- RELATED:START -->

## 相关论文

- [MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](../../CVPR2025/medical_imaging/mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](../../CVPR2026/medical_imaging/every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](../../CVPR2026/medical_imaging/fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](../../ACL2025/medical_imaging/cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [The Four Color Theorem for Cell Instance Segmentation](the_four_color_theorem_for_cell_instance_segmentation.md)

<!-- RELATED:END -->
